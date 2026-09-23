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
import json
import logging
import time
import zlib
from pathlib import Path

from listen2serve.domains.base import DomainSpec, render_customer_profile
from listen2serve.gateway.base import JsonlCallSink
from listen2serve.gateway.llm_gateway import LLMGateway
from listen2serve.gateway.tts_gateway import TTSGateway
from listen2serve.model_registry import (
    FULL_DUPLEX_NO_TEXT_INPUT_PLATFORMS,
    resolve_model,
    supported_genders,
    user_voice_for,
    voice_for,
)
from listen2serve.runtime.agent import Agent
from listen2serve.runtime.config import Settings, get_settings
from listen2serve.runtime.orchestrator import FullDuplexOrchestrator, Trajectory
from listen2serve.runtime.tts_instructions import load_table as load_instruction_table

logger = logging.getLogger(__name__)


def stable_seed(scenario_id: str, mod: int = 100000) -> int:
    """跨进程稳定的场景 seed（Python hash() 有随机盐，不可复现）。"""
    return zlib.crc32(scenario_id.encode("utf-8")) % mod


def _instruction_table_fingerprint() -> dict[str, str]:
    """内部方案：instruction 表的版本与 sha256（换表即换韵律，必须可回溯）。

    表本体的硬失败已经在模拟器构造期发生过；这里只是落盘时补记指纹，读不到不应把
    已跑完的一批结果抹掉，因此此处降级记空值。
    """
    try:
        table = load_instruction_table()
    except (OSError, ValueError) as exc:
        logger.warning("instruction 表指纹读取失败（落空值）：%s", exc)
        return {"dataset_version": "", "sha256": ""}
    return {"dataset_version": table.dataset_version, "sha256": table.sha256}


def assign_voices(scenario: dict, settings: Settings, model_spec: str) -> dict[str, str | None]:
    """按性别均衡策略分配双侧音色（2026-08-10，除非模型只支持单性别）：

    - 用户侧：性别取场景 `user_gender`（数据构造期与姓名一致）→ TTS 模型的对应音色；
    - 客服侧：按场景哈希 50/50 确定性分配 → Realtime 模型的对应音色；
    - .env 的 TTS_VOICE_USER / REALTIME_VOICE 仍可整体覆盖（试验用）。
    """
    sid = str(scenario["scenario_id"])
    user_gender = scenario.get("user_gender") or ("male" if zlib.crc32(f"{sid}-user".encode()) % 2 == 0 else "female")
    agent_gender = "male" if zlib.crc32(f"{sid}-agent".encode()) % 2 == 0 else "female"
    realtime_name = model_spec.partition("/")[2] or model_spec
    # 单性别模型回退时，记录**实际生效**性别（避免审计误导）
    rt_genders = supported_genders(realtime_name)
    if rt_genders and agent_gender not in rt_genders:
        agent_gender = rt_genders[0] if len(rt_genders) == 1 else agent_gender
    # 用户侧音色按**生效 TTS 后端**的绑定模型解析（备选后端 → 该后端绑定音色，
    # 2026-08-14），保证 voices.json 审计与实际合成音色一致
    tts_model_name = settings.tts_model
    tts_genders = supported_genders(tts_model_name)
    effective_user_gender = user_gender
    if tts_genders and user_gender not in tts_genders and len(tts_genders) == 1:
        effective_user_gender = tts_genders[0]
    # 用户侧音色优先级：.env 临时覆盖 > user_voice_map（内部方案 / v5.5，persona
    # 性别×年龄匹配，T1/T2/T3 共享）> 性别默认音色（兼容回落；正式评测 144
    # 基础场景必须全命中 map，测试断言）
    user_voice = (settings.tts_voice_user
                  or user_voice_for(sid, tts_model_name, user_gender)
                  or voice_for(tts_model_name, user_gender))
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
        sim_prompt_schema: str = "v6.0",
        prosody_arm: str = "state",
        sim_thinking: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.concurrency = concurrency or self.settings.concurrency
        self.max_retries = max_retries
        self.save_audio = save_audio
        # 模拟器演绎模型：--sim-model > 配置 SIM_MODEL > 默认跟随 llm_model
        self.sim_model = sim_model or self.settings.sim_model or self.settings.llm_model
        # 内部方案：演绎 prompt schema（v6.0 旧区块式 / v6.1 四块式，A/B 后切默认）
        self.sim_prompt_schema = sim_prompt_schema
        # 内部方案 步骤 4：2×2 四格的韵律因子臂（state=B/D 正式口径 / neutral=A/C 中性臂）。
        # 默认 state，不改任何既有跑法；合法性在 UserSimulator 构造期再校验一次。
        self.prosody_arm = prosody_arm
        # planL 台账 C 第 7 项：演用户 LLM 的 reasoning 开关（被测 agent 不受影响）
        self.sim_thinking = sim_thinking
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.runs_root = Path("runs") / self.run_id
        self.runs_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _retry_backoff_s(model_spec: str, attempt: int) -> float:
        """任务级重试退避：volc（豆包）按 20s/40s，其余保持 2s/4s。

        2026-09-08 实测依据：`mtLN_doubao` 48/137 通失败里 26 通是 `WebSocket 意外关闭`，
        而旧重试间隔 2/4s 后立刻重开一路，**撞的是同一次服务端抖动**（同一会话额度、
        同一排队尖峰）⇒ 34% 是“三次连撞”而非偶发，单单重跑（隔 6h）又能救回 12/40。
        既然拖尾量级是分钟级，退避也得是分钟级。非 volc 端点行为不变。
        """
        # 端点族从 model_registry 的单一事实源派生（⚠️ 原先写 `startswith("volc")`，
        # 字符串前缀判断既漏了同族新端点、又在裸名/带前缀两种写法下行为不一致）。
        try:
            backend, _ = resolve_model(str(model_spec), "")
        except Exception: # noqa: BLE001 - 解析不出来的模型按普通端点退避，不静默升级
            backend = ""
        base = 20.0 if backend in FULL_DUPLEX_NO_TEXT_INPUT_PLATFORMS else 2.0
        return base * (attempt + 1)

    # ---- 单任务 ----
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
        from listen2serve.runtime.user_simulator import UserSimulator

        # 域按场景角色逐条解析（支持跨角色混跑，如三角色冒烟）；显式传入时优先
        domain = domain or get_domain(scenario["role"])

        artifact_dir = self.runs_root / str(scenario["scenario_id"])
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # 双侧音色按性别均衡分配，并落盘供审计/分层分析
        voices = assign_voices(scenario, self.settings, model_spec)
        (artifact_dir / "voices.json").write_text(
            json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")

        # 客户信息档案（2026-08-14：评测默认不发起 function call）：db_seed 单行记录
        # 自然语言化后拼在策略指令**前面**，直接作为初始 prompt 的事实依据；
        # db_seed 为空时档案为空串，旧场景行为不变。
        db_seed = scenario.get("db_seed")
        profile = render_customer_profile(domain, db_seed)
        # 内部方案：turn_budget.soft 进客服侧 prompt（节奏引导行，）
        policy_prompt = domain.render_policy_prompt(
            scenario.get("agent_policy") or {},
            turn_budget=(scenario.get("user_script") or {}).get("turn_budget"),
            variant=self.settings.agent_prompt,
        )
        system_prompt = f"{profile}\n\n{policy_prompt}" if profile else policy_prompt
        agent = Agent(
            model_spec=model_spec,
            api_key=api_key,
            domain=domain,
            system_prompt=system_prompt,
            voice=voices["agent_voice"], # 客服侧：按场景哈希分配性别（单性别模型自动回退）
            tick_ms=self.settings.tick_ms,
            tools_enabled=self.settings.tools_enabled,
            prompt_variant=self.settings.agent_prompt,
        )
        # 工具/DB 预留机制：仅开关打开时注入场景 db_seed 初始化的 DB（行格式 list → tuple）；
        # 关闭时不注入（未注册工具本不会触发；TOOL_DISABLED_OUTPUT 兜底保留）
        if self.settings.tools_enabled:
            agent.attach_db(domain.init_db(_normalize_db_seed(db_seed)))
        simulator = UserSimulator(
            user_script=scenario["user_script"],
            llm_gateway=llm,
            tts_gateway=tts,
            llm_model=self.sim_model,
            voice=voices["user_voice"], # 用户侧：性别随场景 user_gender（与姓名一致）
            audio_mode="control",
            seed=stable_seed(str(scenario["scenario_id"])),
            # B3：场景级上下文透传（few-shot 选样/事实锚定；缺失时模拟器内部自行回落）
            role=scenario.get("role", ""),
            dynamics=scenario.get("dynamics", ""),
            db_seed=scenario.get("db_seed"),
            scenario_id=str(scenario.get("scenario_id", "")), # tts_control 越界报错的定位上下文
            # 内部方案：instruction 表的索引键（不含 -T1/-T2/-T3）。数据里已有该字段，
            # 显式透传而不让模拟器从 scenario_id 反推，避开命名规则变更时的静默错位。
            base_scenario_id=str(scenario.get("base_scenario_id", "")),
            # 整改 P0：透明度层级显式透传（数据已有 leakage_label 字段），
            # 演绎约束与 tts 回落直接消费，不再从 text_policy 自由文本反推
            leakage_label=scenario.get("leakage_label", ""),
            user_gender=scenario.get("user_gender", ""), # tts 回落角色短语（与生成期同款）
            sub_domain=scenario.get("sub_domain", ""),
            # 内部方案：逐轮 elicit prompt 落盘（网页逐轮明细与审计消费）
            prompt_log_path=artifact_dir / "user_prompts.jsonl",
            # 内部方案：outbound 首轮被动接听判定依据（role_type 显式透传）
            role_type=scenario.get("role_type", ""),
            # 内部方案：演绎 prompt schema（A/B 验证后切默认）+ 共识背景注入
            prompt_schema=self.sim_prompt_schema,
            background_consensus=(scenario.get("background_card") or {}).get("consensus", ""),
            # 内部方案 步骤 4：韵律因子臂透传（neutral 时 instruct 恒取 non_key.neutral）
            prosody_arm=self.prosody_arm,
            sim_thinking=self.sim_thinking,
        )
        orchestrator = FullDuplexOrchestrator(
            settings=self.settings, artifact_dir=str(artifact_dir), save_audio=self.save_audio
        )
        traj = await orchestrator.run_scenario(scenario, agent, simulator)
        # 内部方案：模拟器协议健康度计数随轨迹落盘（reflect_smoke 消费）
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
    ) -> list[Trajectory]:
        """并发运行全部场景（支持 checkpoint 续跑；domain=None 时按场景角色逐条解析）。"""
        sem = asyncio.Semaphore(self.concurrency)
        sink = JsonlCallSink(str(self.runs_root / "gateway_calls.jsonl"))
        llm = LLMGateway(self.settings, sink=sink)
        tts = TTSGateway(self.settings, sink=sink)
        # 已在 worker 内落过盘的 scenario_id（末尾就不必再写一遍）
        saved: set[str] = set()

        async def worker(scenario: dict[str, dict]) -> Trajectory:
            async with sem:
                sid = str(scenario["scenario_id"])
                # checkpoint 续跑：已完成且成功的任务加载历史轨迹（不覆盖、不重跑）
                status_path = self.runs_root / sid / "sim_status.json"
                if resume and status_path.exists():
                    try:
                        data = json.loads(status_path.read_text(encoding="utf-8"))
                        if data.get("status") == "ok":
                            logger.info("续跑加载已完成任务: %s", sid)
                            traj = Trajectory.from_dict(data)
                            traj.resumed = True
                            # B0：历史产物透传 dataset_version（旧产物无该字段回落 v4）
                            traj.dataset_version = data.get("dataset_version") or "v4"
                            return traj
                    except json.JSONDecodeError:
                        pass
                last: Trajectory | None = None
                for attempt in range(self.max_retries + 1):
                    last = await self._run_one(scenario, model_spec, api_key, domain, llm, tts)
                    if last.status == "ok":
                        break
                    logger.warning("任务 %s 第 %d 次失败: %s", sid, attempt + 1, last.error)
                    if attempt < self.max_retries:
                        await asyncio.sleep(self._retry_backoff_s(model_spec, attempt))
                # B0：sim_status 透传 dataset_version（读自场景，旧场景无该字段回落 "v4"）
                last.dataset_version = str(scenario.get("dataset_version") or "v4")
                # ⚠️ **逐场景增量落盘**（原先只在 gather 之后统一写）。
                # 2026-09-07 通宵批被这件事咬了两次，都不是理论风险：
                # ① 发现豆包换 key 必崩（tick_ms）时想立刻重启，但"整轮才落盘"意味着
                # 杀掉进程 = 已跑完的 119 通全部丢失（内存里没有第二份）⇒ 只能干等
                # 它把 145 通全崩完；实测那批 0/145，等于白跑一整轮。
                # ② Layer R 需要抽 Layer L 的冻结 history，而 history 来自 sim_status.json
                # ⇒ 生成没结束就抽不到，接力队列只能每 180s 重试派发。
                # 写的是各场景自己目录下的文件、且 _save_trajectory 是同内容覆盖
                # ⇒ 协程之间无争用（本方法是同步函数，也不跨 await 点）。
                # 失败的轨迹同样落盘：resume 只跳过 status=="ok"，所以 fail 会被重跑。
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
        return results

    def _save_trajectory(self, traj: Trajectory) -> None:
        path = self.runs_root / traj.scenario_id / "sim_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = traj.as_dict()
        # B0：仅追加可选字段 dataset_version，不改任何既有字段（results.json 仍用 as_dict 原样）
        data["dataset_version"] = getattr(traj, "dataset_version", "v4")
        data["tts_instructions"] = _instruction_table_fingerprint()
        # 内部方案 步骤 4：本轮所属四格韵律臂随轨迹落盘（B/D=state, A/C=neutral），供分层比对
        data["prosody_arm"] = self.prosody_arm
        # planL 台账 C 第 7 项：随轨迹落盘，否则分析侧只能靠命令行回忆本臂开没开 thinking
        data["sim_thinking"] = self.sim_thinking
        # v7：被测 Agent prompt 版本随轨迹落盘。重评时裁判契约必须按生成时的版本渲染，
        # 否则 v6 run 会被 v7 契约评（默认值漂移即静默错评）。
        data["agent_prompt"] = self.settings.agent_prompt
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_results(self, results: list[Trajectory]) -> None:
        path = self.runs_root / "results.json"
        path.write_text(
            json.dumps([r.as_dict() for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ok = sum(1 for r in results if r.status == "ok")
        logger.info("批量完成: %d/%d 成功（含续跑加载 %d）", ok, len(results),
                    sum(1 for r in results if r.resumed))


def _normalize_db_seed(db_seed: dict | None) -> dict[str, list[tuple]] | None:
    """scenarios.jsonl 中的 db_seed（行为 list）→ init_db 的 seed_data（行为 tuple）。"""
    if not db_seed:
        return None
    return {table: [tuple(row) for row in rows] for table, rows in db_seed.items()}
