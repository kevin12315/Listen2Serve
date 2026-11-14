"""批量运行：并发 / checkpoint 续跑 / 任务级重试 / 网关审计落盘。

运行产物布局（每个任务独立目录）：
runs/<run_id>/<scenario_id>/sim_status.json （轨迹）
runs/<run_id>/<scenario_id>/ticks.json （tick 时间线）
runs/<run_id>/<scenario_id>/audio/*.wav （音频）
runs/<run_id>/gateway_calls.jsonl （LLM/TTS 调用审计）
runs/<run_id>/results.json （聚合结果）
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import zlib
from pathlib import Path
from typing import Any

from listen2serve.domains.base import DomainSpec, render_customer_profile
from listen2serve.gateway.base import JsonlCallSink, summarize_identities
from listen2serve.gateway.llm_gateway import LLMGateway
from listen2serve.gateway.tts_gateway import TTSGateway
from listen2serve.model_registry import (
    FULL_DUPLEX_NO_TEXT_INPUT_PLATFORMS,
    resolve_model,
    supported_genders,
    user_voice_for,
    voice_for,
)
from listen2serve.runtime.agent import Agent, voice_call_instruction
from listen2serve.runtime.config import Settings, get_settings
from listen2serve.runtime.manifest import (
    build_run_manifest,
    load_run_manifest,
    sanitize_settings,
    sha256_dir,
    write_run_manifest,
)
from listen2serve.runtime.orchestrator import FullDuplexOrchestrator, Trajectory
from listen2serve.runtime.tts_instructions import load_table as load_instruction_table

logger = logging.getLogger(__name__)


def stable_seed(scenario_id: str, mod: int = 100000) -> int:
    """跨进程稳定的场景 seed（Python hash() 有随机盐，不可复现）。"""
    return zlib.crc32(scenario_id.encode("utf-8")) % mod


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _instruction_table_fingerprint() -> dict[str, str]:
    """记录实际使用的韵律指令表。"""
    table = load_instruction_table()
    return {"dataset_version": table.dataset_version, "sha256": table.sha256}


def assign_voices(scenario: dict, settings: Settings, model_spec: str) -> dict[str, str | None]:
    """按场景身份确定性分配双侧音色，显式配置优先。"""
    sid = str(scenario["scenario_id"])
    user_gender = scenario.get("user_gender") or (
        "male" if zlib.crc32(f"{sid}-user".encode()) % 2 == 0 else "female"
    )
    agent_gender = "male" if zlib.crc32(f"{sid}-agent".encode()) % 2 == 0 else "female"
    realtime_name = model_spec.partition("/")[2] or model_spec
    # 单性别模型回退时，记录**实际生效**性别（避免审计误导）
    rt_genders = supported_genders(realtime_name)
    if rt_genders and agent_gender not in rt_genders:
        agent_gender = rt_genders[0] if len(rt_genders) == 1 else agent_gender
    tts_model_name = "cosyvoice-v2" if settings.tts_backend == "cosyvoice" else settings.tts_model
    tts_genders = supported_genders(tts_model_name)
    effective_user_gender = user_gender
    if tts_genders and user_gender not in tts_genders and len(tts_genders) == 1:
        effective_user_gender = tts_genders[0]
    # 用户音色映射优先于模型的性别默认音色。
    user_voice = (
        settings.tts_voice_user
        or user_voice_for(sid, tts_model_name, user_gender)
        or voice_for(tts_model_name, user_gender)
    )
    return {
        "user_gender": effective_user_gender,
        "user_voice": user_voice,
        "agent_gender": agent_gender,
        "agent_voice": settings.realtime_voice or voice_for(realtime_name, agent_gender),
    }


class BatchRunner:
    """批量评测运行器。"""

    def __init__(
        self,
        settings: Settings | None = None,
        run_id: str | None = None,
        concurrency: int | None = None,
        max_retries: int = 2,
        save_audio: bool = True,
        sim_model: str | None = None,
        prosody_arm: str = "state",
        sim_thinking: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.concurrency = concurrency or self.settings.concurrency
        self.max_retries = max_retries
        self.save_audio = save_audio
        self.sim_model = sim_model or self.settings.sim_model or self.settings.llm_model
        self.prosody_arm = prosody_arm
        self.sim_thinking = sim_thinking
        self._identities: dict[str, dict[str, Any]] = {}
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.runs_root = Path("runs") / self.run_id
        self.runs_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _retry_backoff_s(model_spec: str, attempt: int) -> float:
        """连续音频端点采用较长退避，避免重试撞上尚未释放的会话。"""
        try:
            backend, _ = resolve_model(str(model_spec), "")
        except Exception:  # noqa: BLE001 - 解析不出来的模型按普通端点退避，不静默升级
            backend = ""
        base = 20.0 if backend in FULL_DUPLEX_NO_TEXT_INPUT_PLATFORMS else 2.0
        return base * (attempt + 1)

    def _simulator(
        self, scenario: dict, voices: dict, llm: LLMGateway, tts: TTSGateway, artifact_dir: Path | None = None
    ):
        from listen2serve.runtime.user_simulator import UserSimulator

        return UserSimulator(
            user_script=scenario["user_script"],
            llm_gateway=llm,
            tts_gateway=tts,
            llm_model=self.sim_model,
            voice=voices["user_voice"],
            audio_mode="control",
            seed=stable_seed(str(scenario["scenario_id"])),
            role=scenario.get("role", ""),
            dynamics=scenario.get("dynamics", ""),
            db_seed=scenario.get("db_seed"),
            scenario_id=str(scenario["scenario_id"]),
            base_scenario_id=str(scenario.get("base_scenario_id", "")),
            leakage_label=scenario.get("leakage_label", ""),
            user_gender=scenario.get("user_gender", ""),
            sub_domain=scenario.get("sub_domain", ""),
            prompt_log_path=artifact_dir / "user_prompts.jsonl" if artifact_dir else None,
            role_type=scenario.get("role_type", ""),
            prosody_arm=self.prosody_arm,
            sim_thinking=self.sim_thinking,
        )

    @staticmethod
    def _agent_prompt(scenario: dict, domain: DomainSpec) -> str:
        profile = render_customer_profile(domain, scenario.get("db_seed"))
        policy = domain.render_policy_prompt(
            scenario.get("agent_policy") or {},
            turn_budget=(scenario.get("user_script") or {}).get("turn_budget"),
        )
        return f"{profile}\n\n{policy}" if profile else policy

    def generation_identity(
        self, scenario: dict, model_spec: str, domain: DomainSpec, llm: LLMGateway, tts: TTSGateway
    ) -> dict[str, Any]:
        """绑定真实生成输入；在任何网络请求和旧产物覆盖之前计算。"""
        voices = assign_voices(scenario, self.settings, model_spec)
        simulator = self._simulator(scenario, voices, llm, tts)
        messages = {
            "initial": simulator._build_messages(1, "", forced=False),
            "forced": simulator._build_messages(simulator.hard_budget - 2, "", forced=True),
            "soft_budget": simulator._build_messages(simulator.soft_budget, "", forced=False),
        }
        simulator._key_turn_no = 2
        messages["closing"] = simulator._build_messages(simulator.soft_budget, "", forced=False)
        # 评测模型、凭据和并发不改变生成条件，其余生效配置纳入续跑身份。
        settings = {
            k: v
            for k, v in sanitize_settings(self.settings).items()
            if k not in {"judge_model", "voice_judge_model", "asr_model", "concurrency"}
        }
        target_backend, target_model = resolve_model(model_spec, self.settings.llm_model)
        sim_backend, sim_model = resolve_model(self.sim_model, self.settings.llm_model)
        tts_model = "cosyvoice-v2" if self.settings.tts_backend == "cosyvoice" else self.settings.tts_model
        identity = {
            "protocol": "multi-turn",
            "source_tree_sha256": sha256_dir(Path(__file__).resolve().parents[1]),
            "scenario_sha256": _digest(scenario),
            "models": {
                "target": f"{target_backend}/{target_model}",
                "sim": f"{sim_backend}/{sim_model}",
                "tts": f"{self.settings.tts_backend}/{tts_model}",
            },
            "voices": voices,
            "prompts": {
                "agent_sha256": _digest(
                    voice_call_instruction() + "\n\n" + self._agent_prompt(scenario, domain)
                ),
                "sim_sha256": _digest(messages),
            },
            "tts_instructions": _instruction_table_fingerprint(),
            "settings": settings,
            "prosody_arm": self.prosody_arm,
            "sim_thinking": self.sim_thinking,
            "save_audio": self.save_audio,
        }
        identity["digest"] = _digest(identity)
        return identity

    def _prepare_run(
        self,
        scenarios: list[dict],
        model_spec: str,
        domain: DomainSpec | None,
        llm: LLMGateway,
        tts: TTSGateway,
        manifest: dict | None,
    ) -> None:
        from listen2serve.domains import get_domain

        sids = [str(s["scenario_id"]) for s in scenarios]
        if len(sids) != len(set(sids)):
            raise ValueError("场景清单含重复 scenario_id")
        identities = {
            str(s["scenario_id"]): self.generation_identity(
                s, model_spec, domain or get_domain(s["role"]), llm, tts
            )
            for s in scenarios
        }
        # 先校验全部任务，避免一部分已开始调用后才发现另一条缓存不兼容。
        for sid, identity in identities.items():
            path = self.runs_root / sid / "sim_status.json"
            if not path.exists():
                continue
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"已有轨迹无法读取：{sid}；请使用新的 run-id") from exc
            if not isinstance(previous, dict):
                raise ValueError(f"已有轨迹格式错误：{sid}；请使用新的 run-id")  # noqa: TRY004 - malformed JSON artifact
            stored = previous.get("generation_identity") or {}
            try:
                backend, name = resolve_model(previous.get("model"), self.settings.llm_model)
                stored_model = f"{backend}/{name}"
            except ValueError:
                stored_model = ""
            if (
                stored != identity
                or previous.get("scenario_id") != sid
                or stored_model != identity["models"]["target"]
            ):
                raise ValueError(f"生成条件与已有轨迹不一致：{sid}；请使用新的 run-id")
        old_manifest = load_run_manifest(self.runs_root)
        manifest_path = self.runs_root / "run_manifest.json"
        if manifest_path.exists() and old_manifest is None:
            raise ValueError("已有 run 的 manifest 无法读取；请使用新的 run-id")
        if old_manifest is None and any(self.runs_root.glob("*/sim_status.json")):
            raise ValueError("已有 run 缺少 manifest；请使用新的 run-id")
        # 同一个 run 的新增场景也必须使用相同的模型和生成配置。
        common = {
            k: v
            for k, v in next(iter(identities.values())).items()
            if k not in {"scenario_sha256", "voices", "prompts", "digest"}
        }
        common_sha = _digest(common)
        if old_manifest and old_manifest.get("generation"):
            generation = old_manifest["generation"]
            if not isinstance(generation, dict) or generation.get("settings_sha256") != common_sha:
                raise ValueError("生成条件与已有 run 不一致；请使用新的 run-id")
        elif old_manifest and any(self.runs_root.glob("*/sim_status.json")):
            raise ValueError("已有 run 缺少可验证的生成身份；请使用新的 run-id")
        self._identities = identities
        if manifest is None:
            manifest = build_run_manifest(self.run_id, {}, {}, self.settings)
        # 保留已完成评测的真实模型和指纹，不用生成期占位覆盖它们。
        if old_manifest and old_manifest.get("evaluation"):
            manifest["evaluation"] = old_manifest["evaluation"]
        manifest["generation"] = {
            "protocol": "multi-turn",
            "settings_sha256": common_sha,
            "models": common["models"],
            "fingerprints": {sid: value["digest"] for sid, value in identities.items()},
        }
        manifest["models"].update(common["models"])
        manifest["prompts"].update(
            {
                "agent_prompt_sha256": _digest(
                    {sid: v["prompts"]["agent_sha256"] for sid, v in identities.items()}
                ),
                "sim_prompt_sha256": _digest(
                    {sid: v["prompts"]["sim_sha256"] for sid, v in identities.items()}
                ),
            }
        )
        write_run_manifest(self.runs_root, manifest)

    async def _run_one(
        self,
        scenario: dict[str, dict],
        model_spec: str,
        api_key: str,
        domain: DomainSpec | None,
        llm: LLMGateway,
        tts: TTSGateway,
    ) -> Trajectory:
        from listen2serve.domains import get_domain

        # 域按场景角色逐条解析（支持跨角色混跑，如三角色冒烟）；显式传入时优先
        domain = domain or get_domain(scenario["role"])

        artifact_dir = self.runs_root / str(scenario["scenario_id"])
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # 双侧音色按性别均衡分配，并落盘供审计/分层分析
        voices = assign_voices(scenario, self.settings, model_spec)
        (artifact_dir / "voices.json").write_text(
            json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        db_seed = scenario.get("db_seed")
        system_prompt = self._agent_prompt(scenario, domain)
        agent = Agent(
            model_spec=model_spec,
            api_key=api_key,
            domain=domain,
            system_prompt=system_prompt,
            voice=voices["agent_voice"],  # 客服侧：按场景哈希分配性别（单性别模型自动回退）
            tick_ms=self.settings.tick_ms,
            tools_enabled=self.settings.tools_enabled,
        )
        # 工具/DB 预留机制：仅开关打开时注入场景 db_seed 初始化的 DB（行格式 list → tuple）；
        # 关闭时不注入（未注册工具本不会触发；TOOL_DISABLED_OUTPUT 兜底保留）
        if self.settings.tools_enabled:
            agent.attach_db(domain.init_db(_normalize_db_seed(db_seed)))
        simulator = self._simulator(scenario, voices, llm, tts, artifact_dir)
        orchestrator = FullDuplexOrchestrator(
            settings=self.settings, artifact_dir=str(artifact_dir), save_audio=self.save_audio
        )
        traj = await orchestrator.run_scenario(scenario, agent, simulator)
        # 模拟器协议健康度计数随轨迹落盘。
        traj.sim_stats = dict(getattr(simulator, "stats", {}) or {})
        return traj

    # ---- 批量 ----
    async def run_batch(
        self,
        scenarios: list[dict[str, dict]],
        model_spec: str,
        api_key: str,
        domain: DomainSpec | None = None,
        resume: bool = True,
        manifest: dict[str, Any] | None = None,
    ) -> list[Trajectory]:
        """并发运行场景；响应超时在本 run 内不重试或覆盖，新 run-id 可重新生成。"""
        if not scenarios:
            return []
        sem = asyncio.Semaphore(self.concurrency)
        sink = JsonlCallSink(str(self.runs_root / "gateway_calls.jsonl"))
        llm = LLMGateway(self.settings, sink=sink)
        tts = TTSGateway(self.settings, sink=sink)
        self._prepare_run(scenarios, model_spec, domain, llm, tts, manifest)
        # 已在 worker 内落过盘的任务无需再次写入。
        saved: set[str] = set()

        async def worker(scenario: dict[str, dict]) -> Trajectory:
            async with sem:
                sid = str(scenario["scenario_id"])
                # checkpoint 续跑：已完成且成功的任务加载历史轨迹（不覆盖、不重跑）
                status_path = self.runs_root / sid / "sim_status.json"
                if status_path.exists():
                    try:
                        data = json.loads(status_path.read_text(encoding="utf-8"))
                        previous = Trajectory.from_dict(data)
                        # 截断证据保留在原 run；重新生成应使用新 run-id。
                        if (
                            previous.status == "partial"
                            or previous.termination_reason == "response_timeout"
                            or (resume and previous.status == "ok")
                        ):
                            logger.info("续跑加载已完成任务: %s", sid)
                            traj = previous
                            traj.resumed = True
                            traj.dataset_version = data.get("dataset_version", "")
                            traj.rt_diag = data.get("rt_diag") or {}
                            return traj
                    except json.JSONDecodeError:
                        pass
                last: Trajectory | None = None
                for attempt in range(self.max_retries + 1):
                    last = await self._run_one(scenario, model_spec, api_key, domain, llm, tts)
                    if last.status in {"ok", "partial"} or last.termination_reason == "response_timeout":
                        break
                    logger.warning("任务 %s 第 %d 次失败: %s", sid, attempt + 1, last.error)
                    if attempt < self.max_retries:
                        await asyncio.sleep(self._retry_backoff_s(model_spec, attempt))
                assert last is not None
                last.dataset_version = str(scenario.get("dataset_version", ""))
                # 每个任务完成即写盘，失败任务保留并在续跑时重试。
                self._save_trajectory(last)
                saved.add(last.scenario_id)
                return last

        tasks = [worker(s) for s in scenarios]
        results = list(await asyncio.gather(*tasks))

        # 落盘（续跑加载的历史结果不重复写，避免覆盖；worker 里已写过的也不再重写）
        for traj in results:
            if not traj.resumed and traj.scenario_id not in saved:
                self._save_trajectory(traj)
        self._save_results(results)
        current = load_run_manifest(self.runs_root)
        if current is not None:
            records = []
            calls_path = self.runs_root / "gateway_calls.jsonl"
            if calls_path.is_file():
                for line in calls_path.read_text(encoding="utf-8").splitlines():
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            records.append(record)
                    except json.JSONDecodeError:
                        continue
            current.setdefault("model_identity", {})["generation"] = {
                "schema": "requested-response-v1",
                "targets": {t.scenario_id: t.provider_identity for t in results},
                "gateway_calls": summarize_identities(records),
            }
            write_run_manifest(self.runs_root, current)
        return results

    def _save_trajectory(self, traj: Trajectory) -> None:
        path = self.runs_root / traj.scenario_id / "sim_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = traj.as_dict()
        data["dataset_version"] = getattr(traj, "dataset_version", "")
        data["tts_instructions"] = _instruction_table_fingerprint()
        data["prosody_arm"] = self.prosody_arm
        data["sim_thinking"] = self.sim_thinking
        data["protocol"] = "multi-turn"
        data["generation_identity"] = self._identities[traj.scenario_id]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_results(self, results: list[Trajectory]) -> None:
        path = self.runs_root / "results.json"
        path.write_text(
            json.dumps([r.as_dict() for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ok = sum(1 for r in results if r.status == "ok")
        partial = sum(1 for r in results if r.status == "partial")
        logger.info(
            "批量完成: %d 完整、%d 部分可评、%d 失败（含续跑加载 %d）",
            ok,
            partial,
            len(results) - ok - partial,
            sum(1 for r in results if r.resumed),
        )


def _normalize_db_seed(db_seed: dict | None) -> dict[str, list[tuple]] | None:
    """scenarios.jsonl 中的 db_seed（行为 list）→ init_db 的 seed_data（行为 tuple）。"""
    if not db_seed:
        return None
    return {table: [tuple(row) for row in rows] for table, rows in db_seed.items()}
