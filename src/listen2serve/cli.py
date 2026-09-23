"""Listen2Serve CLI 入口。

子命令：
    run-eval 批量运行评测（被测模型对话）
    evaluate 对轨迹运行 Judge（Policy/Voice/对话级）
    report 聚合渲染报告（md + json）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from listen2serve.domains import get_domain
from listen2serve.domains.base import AGENT_PROMPT_VERSIONS, render_customer_profile
from listen2serve.gateway.asr_gateway import ASRGateway


def _load_scenarios(path: str, limit: int | None, role: str | None = None,
                    leakage: str | None = None, scenario_ids: str | None = None,
                    per_role: int | None = None) -> list[dict]:
    rows = []
    wanted = {s.strip() for s in scenario_ids.split(",")} if scenario_ids else None
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        s = json.loads(line)
        if wanted is not None and s.get("scenario_id") not in wanted \
                and s.get("base_scenario_id") not in wanted:
            continue # 支持精确 ID（含 T 后缀）或 base ID（命中全部透明度变体）
        if role and s.get("role") != role:
            continue
        if leakage and s.get("leakage_level") != leakage:
            continue
        rows.append(s)
    if per_role:
        # 每角色取前 N 条（最小冒烟约束：三角色至少各一条）
        picked: list[dict] = []
        count: dict[str, int] = {}
        for s in rows:
            r = s.get("role", "")
            if count.get(r, 0) < per_role:
                picked.append(s)
                count[r] = count.get(r, 0) + 1
        rows = picked
    if limit:
        rows = rows[:limit]
    return rows


async def _run_eval(args: argparse.Namespace) -> int:
    from listen2serve.runtime.config import get_settings
    from listen2serve.runtime.run_batch import BatchRunner

    settings = get_settings()
    scenarios = _load_scenarios(args.scenarios, args.limit, args.role, args.leakage,
                                getattr(args, "scenario_ids", None), getattr(args, "per_role", None))
    if not scenarios:
        print(f"无场景可运行: {args.scenarios}")
        return 1
    # api_key 按**被测模型的平台**路由（2026-09-06 修）。
    # 原先这里硬编码 `settings.dashscope_api_key`，于是跑 volc（豆包 Seeduplex）时会把
    # dashscope 的 key 交给 VolcRealtimeProvider ⇒ X-Api-Key 鉴权必然失败，
    # 而且报错长得像「端点不通/账户没开通」，会把人引到错误的排查方向（内部方案 踩过同类坑）。
    from listen2serve.model_registry import resolve_model

    _target = args.model or ""
    _backend, _mname = "dashscope", _target
    if _target:
        try:
            _backend, _mname = resolve_model(_target, settings.llm_model)
        except Exception as exc: # 模型未登记 ⇒ 如实报错，不回落
            print(f"被测模型无法解析：{exc}")
            return 1
    if _backend == "volc":
        api_key = settings.volc_key() or ""
        _missing = "缺少 VOLC_API_KEY / VOLC_API_KEY2（.env）"
    else:
        api_key = settings.dashscope_api_key or ""
        _missing = "缺少 DASHSCOPE_API_KEY（.env）"
    if not api_key:
        print(_missing)
        return 1
    if _backend == "volc":
        # ⚠️ 只打前 6 位，绝不打印完整 key（本仓纪律）；打出把数是为了让日志能看出故障转移有没有发生
        print(f"被测平台=volc（豆包），生效 key 前缀={(api_key or '')[:6]}…，"
              f"共配置 {len(settings.volc_keys())} 把（连接失败自动切下一把）", flush=True)
    roles = {s["role"] for s in scenarios}
    print(f"运行 {len(scenarios)} 条场景（角色：{'/'.join(sorted(roles))}）→ runs/<run_id> 逐场景落盘", flush=True)
    runner = BatchRunner(
        settings=settings,
        run_id=args.run_id,
        concurrency=args.concurrency,
        save_audio=args.save_audio,
        sim_model=getattr(args, "sim_model", None),
        sim_prompt_schema=getattr(args, "sim_prompt_schema", "v6.0"),
        prosody_arm=getattr(args, "prosody_arm", "state"),
        sim_thinking=bool(getattr(args, "sim_thinking", False)),
        # None ⇒ 用 BatchRunner 自己的默认值（2 次重试）；显式给 0 用于"外层已按轮补跑"的
        # 场景：批内再重试只是重复烧时间，且立刻重试更容易撞上尚未释放的僵尸会话槽。
        **({"max_retries": args.max_retries}
           if getattr(args, "max_retries", None) is not None else {}),
    )
    # 内部方案：run_manifest.json 版本指纹（代码/数据/prompt/模型/CLI 参数/生效配置）
    from listen2serve.runtime.manifest import build_run_manifest, write_run_manifest

    manifest = build_run_manifest(
        run_id=runner.run_id,
        models={
            "target": args.model or "",
            "sim": runner.sim_model or "",
            # ⚠️ 生成阶段**还不知道**会用哪个裁判 —— 裁判是评测步的 `--judge-model` 决定的。
            # 原先这里预填 `settings.llm_model`，于是任何"只生成、或用别的裁判评"的 run，
            # manifest 都会声称一个没参与判定的裁判；补跑还会把该字段重置回默认值
            # （生成步重建 manifest），使谎报反复出现。留 None 才是诚实口径：裁判身份由
            # `run_experiment.sh` 的回写步骤在评测后填，并拿 verdicts 的 judge_meta 反向核对
            # （不一致拒绝写）。
            "judge": None,
            "voice_judge": None,
            "tts": f"{settings.tts_backend}/{settings.tts_model}",
        },
        cli_args={k: v for k, v in vars(args).items() if k != "save_audio"},
        settings=settings,
    )
    write_run_manifest(runner.runs_root, manifest)
    print(f"版本指纹: runs/{runner.run_id}/run_manifest.json", flush=True)
    # 域按场景角色逐条解析（支持跨角色混跑）
    results = await runner.run_batch(scenarios, args.model, api_key, domain=None)
    ok = sum(1 for r in results if r.status == "ok")
    print(f"运行完成: {ok}/{len(results)} 成功 → runs/{runner.run_id}")
    return 0


def _warn_dataset_version_mismatch(runs_root: Path, scenarios: list[dict],
                                    scenarios_path: str) -> None:
    """旧 run 重评防护：逐条比对轨迹 dataset_version 与当前场景文件版本。

    旧 run（更早数据集版本的产物）用当前 scenarios.jsonl 重评时，oracle 与关键轮台词
    已大面积变更，裁判会以错配口径打分且不报错；此处打印显式警告（不阻断执行），
    建议改用 --scenarios 指定与该 run 同版本的场景文件重评。"""
    by_version: dict[str, int] = {}
    scenario_versions = {str(s.get("dataset_version") or "v4") for s in scenarios}
    for status_path in sorted(runs_root.glob("*/sim_status.json")):
        try:
            traj = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        v = str(traj.get("dataset_version") or "v4")
        if v not in scenario_versions:
            by_version[v] = by_version.get(v, 0) + 1
    if not by_version:
        return
    detail = "、".join(f"{v}×{n} 条" for v, n in sorted(by_version.items()))
    print(
        "⚠ 数据集版本错配警告：该 run 的轨迹 dataset_version（"
        f"{detail}）与当前场景文件（{scenarios_path}，版本 "
        f"{'/'.join(sorted(scenario_versions))}）不一致；oracle 状态与关键轮台词已变更，"
        "重评结果以错配口径打分，仅供对照。",
        flush=True,
    )
    print(
        " 建议：用 --scenarios 指定与该 run 同数据集版本的场景文件重评，"
        "否则结果仅供对照（版本以 run_manifest.json 的 dataset_version 为准）。",
        flush=True,
    )


def _warn_manifest_data_mismatch(runs_root: Path) -> None:
    """内部方案：run_manifest 数据版本错配警告（不阻断）。

    读 runs/<run_id>/run_manifest.json 的 data.dataset_version，与当前
    data/benchmark/version.json 不一致时显式提示（重评口径可能已变更）；
    旧 run 无 manifest 时静默（由 report 展示缺失标记）。"""
    from listen2serve.runtime.manifest import DATA_VERSION_FILE, load_run_manifest

    manifest = load_run_manifest(runs_root)
    if manifest is None:
        return
    run_version = str((manifest.get("data") or {}).get("dataset_version") or "")
    try:
        current = json.loads(
            Path(DATA_VERSION_FILE).read_text(encoding="utf-8")
        ).get("dataset_version", "")
    except (OSError, json.JSONDecodeError):
        return
    if run_version and current and run_version != str(current):
        print(
            f"⚠ manifest 数据版本错配警告：该 run 的 run_manifest 记录数据版本 "
            f"{run_version}，与当前 data/benchmark/version.json（{current}）不一致；"
            "重评结果以当前数据口径为准，仅供对照。",
            flush=True,
        )


@dataclass
class _EvalTaskPayload:
    """单场景评测载荷（调度层透传，judge 语义由 _judge_scenario 负责）。"""

    scenario: dict
    domain: object
    traj: dict
    key_trace: dict
    history_with_agent: list
    history_before_critical: list


def _dialogue_metric(scenario: dict) -> str | None:
    """场景 dynamics → 对话级指标名（与 _judge_scenario 内的分支口径一致）。"""
    if scenario.get("dynamics") in ("all_positive", "all_negative"):
        return "task_completion"
    if scenario.get("dynamics") in ("pos_to_neg", "neg_to_pos"):
        return "transition_score"
    return None


def _build_eval_tasks(runs_root: Path, by_id: dict[str, dict], policy_model: str,
                      voice_model: str, judge_schema: str = "v2.2-I") -> list:
    """按 内部方案 前的遍历顺序与过滤条件构建评测任务（轨迹须 ok、场景须可索引、关键轮须存在）。"""
    from listen2serve.evaluation.incremental import EvalTask
    from listen2serve.evaluation.prompts import build_judge_meta

    tasks = []
    key_missing = key_nonunique = 0
    for status_path in sorted(runs_root.glob("*/sim_status.json")):
        traj = json.loads(status_path.read_text(encoding="utf-8"))
        if traj.get("status") != "ok":
            continue
        sid = traj["scenario_id"]
        scenario = by_id.get(sid)
        if scenario is None:
            print(f"跳过（无场景定义）: {sid}", flush=True)
            continue
        domain = get_domain(scenario["role"])
        # 关键轮提取（内部方案 v6.0：运行期 is_key_turn 唯一定位；
        # 旧 run 无该字段时回落场景预定义 critical_turn 兼容重评）
        key_traces = [t for t in traj["turns"] if t.get("is_key_turn")]
        if key_traces:
            if len(key_traces) > 1: # 防御：唯一性由模拟器状态机保证
                key_nonunique += 1
                print(f"跳过（关键轮非唯一 {len(key_traces)}）: {sid}", flush=True)
                continue
            key_trace = key_traces[0]
            critical = key_trace["turn"]
        elif any("is_key_turn" in t for t in traj["turns"]):
            key_missing += 1
            print(f"跳过（key_turn_missing）: {sid}", flush=True)
            continue
        else:
            critical = (scenario.get("measurement") or {}).get("critical_turn")
            key_trace = next((t for t in traj["turns"] if t["turn"] == critical), None)
        if key_trace is None:
            continue
        # 完整对话历史（user/assistant 逐轮），及关键轮回复前的历史切片
        history_with_agent = []
        history_before_critical = []
        for t in traj["turns"]:
            history_with_agent.append({"role": "user", "content": t["user_text"]})
            if t["turn"] <= critical:
                history_before_critical.append({"role": "user", "content": t["user_text"]})
            if t["agent_text"]:
                history_with_agent.append({"role": "assistant", "content": t["agent_text"]})
                if t["turn"] < critical:
                    history_before_critical.append({"role": "assistant", "content": t["agent_text"]})
        # 续评幂等指纹：judge prompt hash + 裁判模型名（与 verdict 的 judge_meta 同源构建）
        expected_meta = build_judge_meta(
            policy_model=policy_model or "",
            voice_model=voice_model or "",
            dialogue_model=policy_model or "",
            dialogue_metric=_dialogue_metric(scenario),
            profile_injected=bool(render_customer_profile(domain, scenario.get("db_seed"))),
            judge_schema=judge_schema,
        )
        tasks.append(EvalTask(
            scenario_id=sid,
            expected_meta=expected_meta,
            payload=_EvalTaskPayload(
                scenario=scenario, domain=domain, traj=traj, key_trace=key_trace,
                history_with_agent=history_with_agent,
                history_before_critical=history_before_critical,
            ),
        ))
    if key_missing or key_nonunique:
        print(f"关键轮异常跳过：key_turn_missing {key_missing} 条、非唯一 {key_nonunique} 条",
              flush=True)
    return tasks


def _evaluate(args: argparse.Namespace) -> int:
    """对 runs/<run_id>/ 的轨迹运行全部 Judge。

    内部方案：逐场景增量落盘（verdicts_partial_<modality>.jsonl）+ 指纹校验断点续评
    + 可选并发（每 worker 独立 judge 实例）+ 每场景进度行；最终 verdicts_*.json
    格式与 内部方案 前逐字段一致，裁判逻辑与 prompt 不动。
    """
    from listen2serve.evaluation.incremental import (
        JudgeOutcome,
        PartialRejected,
        PartialStore,
        WorkerLocalPool,
        partial_path,
        run_incremental,
    )
    from listen2serve.evaluation.closing_check import check_closing
    from listen2serve.evaluation.lexicon_check import check_user_lexicon
    from listen2serve.report.export_json import export_json
    from listen2serve.runtime.config import get_settings

    settings = get_settings()
    runs_root = Path("runs") / args.run_id
    if not runs_root.exists():
        print(f"runs/{args.run_id} 不存在")
        return 1

    # 内部方案：裁判 schema 切换（默认 v2.3-J 新判据；--legacy-judges 切回 v2.2-I）；
    # --report-id 支持重评产物写独立目录（不覆盖旧判定，风险缓解条款）。
    judge_schema = "v2.2-I" if getattr(args, "legacy_judges", False) else "v2.3-J"
    # 截断观察窗（2026-09-11）：只把对话前 K 轮交给 FlowJudge。用于「会话寿命装不下一通
    # 完整对话」的端点（实测服务端会话只能活 91–146 s，而跑完一通需 176–238 s）。
    # 0/None = 整通口径 ⇒ 三家既有读数零影响；跨端点比较时**四家必须给同一个 K**。
    flow_turn_cap = getattr(args, "flow_turn_cap", 0) or None
    report_dir = Path("reports") / (getattr(args, "report_id", None) or args.run_id)
    store = PartialStore(partial_path(report_dir, args.modality))
    policy_model = args.judge_model or settings.judge_model or settings.llm_model
    voice_model = args.voice_judge_model or settings.voice_judge_model

    # 加载场景（按 scenario_id 索引）
    scenarios = _load_scenarios(args.scenarios, None)
    by_id = {s["scenario_id"]: s for s in scenarios}
    # v5.6：旧 run 重评 dataset_version 错配显式警告（不阻断执行）
    _warn_dataset_version_mismatch(runs_root, scenarios, args.scenarios)
    # 内部方案：run_manifest 数据版本与当前 version.json 错配警告（沿用不阻断策略）
    _warn_manifest_data_mismatch(runs_root)

    tasks = _build_eval_tasks(runs_root, by_id, policy_model, voice_model, judge_schema=judge_schema)
    if not tasks:
        print("无可评测轨迹（status=ok 且场景/关键轮齐全）")
        return 1
    if args.fresh and store.path.exists():
        stale = store.path.with_suffix(f".stale_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        store.path.rename(stale)
        print(f"--fresh: 既有 partial 归档为 {stale}（保留审计，不复用）", flush=True)

    # 每 worker 独立 judge 实例（parse_retries 等实例计数器线程不安全，禁止共享）
    asr_fail_lock = threading.Lock()
    asr_fail = 0
    # K18c：泄露探针失败计数（附加度量，不阻断主评测）
    probe_fail_lock = threading.Lock()
    probe_fail = 0

    def _make_judges():
        from listen2serve.evaluation.voice_judge import VoiceJudge

        judges: dict = {
            "voice": VoiceJudge(llm_model=voice_model),
            "asr": ASRGateway(settings) if (args.modality == "text" or args.asr) else None,
        }
        if judge_schema == "v2.3-J":
            from listen2serve.evaluation.flow_judge import FlowJudge
            from listen2serve.evaluation.policy_judge import KeyTurnJudge

            judges["keyturn"] = KeyTurnJudge(llm_model=policy_model)
            judges["flow"] = FlowJudge(llm_model=policy_model)
            # K18c：泄露探针（不评客服，只盲评用户关键轮文本的字面态度表露量）
            from listen2serve.evaluation.leakage_probe import LeakageProbe

            judges["leakage"] = LeakageProbe(llm_model=policy_model)
        else:
            from listen2serve.evaluation.dialogue_metrics import DialogueMetrics
            from listen2serve.evaluation.policy_judge import PolicyJudge

            judges["policy"] = PolicyJudge(llm_model=policy_model)
            judges["dialogue"] = DialogueMetrics(llm_model=policy_model)
        return judges

    pool = WorkerLocalPool(_make_judges)

    def _judge_scenario(task) -> JudgeOutcome:
        nonlocal asr_fail, probe_fail
        from listen2serve.evaluation.prompts import build_judge_meta

        p: _EvalTaskPayload = task.payload
        b = pool.get()
        voice_judge = b["voice"]
        asr_gateway = b["asr"]
        scenario, domain, traj, key_trace = p.scenario, p.domain, p.traj, p.key_trace
        # 内部方案：关键轮号取 run 产物（is_key_turn 唯一定位）；旧 run 回落 critical_turn
        critical = key_trace["turn"]
        timings: dict[str, float] = {}

        # ---- ASR（文本模态或 --asr 对比分析）----
        asr_text = None
        if asr_gateway is not None and key_trace.get("agent_audio_path"):
            try:
                asr_text = asr_gateway.transcribe(key_trace["agent_audio_path"]).text
            except Exception as exc: # noqa: BLE001 - ASR 失败不影响主流程
                with asr_fail_lock:
                    asr_fail += 1
                print(f" [ASR 失败] {task.scenario_id}: {exc}", flush=True)
        # 文本模态：用 ASR 转写替代真实文本（失败则回退真实文本）
        eval_agent_text = asr_text or key_trace["agent_text"]

        # 关键轮级
        t0 = time.perf_counter()
        ktv = pv = None
        if judge_schema == "v2.3-J":
            ktv = b["keyturn"].judge(
                domain, scenario, p.history_before_critical, eval_agent_text
            )
        else:
            pv = b["policy"].judge(
                domain, scenario, p.history_before_critical, eval_agent_text
            )
        timings["policy"] = time.perf_counter() - t0
        if args.modality == "text":
            vv = None # 文本模态不评声音
        else:
            # B2：voice_requirements/communication_style 注入 Voice Judge（声音锚点不再硬编码）
            ap = scenario.get("agent_policy") or {}
            t0 = time.perf_counter()
            vv = voice_judge.judge(
                key_trace["agent_audio_path"] or "",
                ap.get("role_description", ""),
                key_trace["user_state"],
                key_trace["agent_text"],
                voice_requirements=ap.get("voice_requirements") or domain.default_voice_requirements,
                communication_style=ap.get("communication_style"),
            ) if key_trace["agent_audio_path"] else None
            timings["voice"] = time.perf_counter() - t0

        # 对话级（文本模态同样用 ASR 文本）
        dv = None
        fv = None
        history_for_dialogue = p.history_with_agent
        if args.modality == "text" and asr_text and key_trace["turn"] == critical:
            # 关键轮客服文本替换为 ASR 转写
            history_for_dialogue = [
                h if h["role"] != "assistant" or h["content"] != key_trace["agent_text"]
                else {"role": "assistant", "content": asr_text}
                for h in p.history_with_agent
            ]
        t0 = time.perf_counter()
        if judge_schema == "v2.3-J":
            # FlowJudge：全对话口径一次产出 flow_items + task_score（无 dynamics 分支）
            fv = b["flow"].judge(domain, scenario, history_for_dialogue, flow_turn_cap)
        elif scenario.get("dynamics") in ("all_positive", "all_negative"):
            dv = b["dialogue"].task_completion(domain, scenario, history_for_dialogue)
        elif scenario.get("dynamics") in ("pos_to_neg", "neg_to_pos"):
            dv = b["dialogue"].transition_score(domain, scenario, history_for_dialogue)
        if fv is not None or dv is not None:
            timings["dialogue"] = time.perf_counter() - t0

        # 非关键轮用户话术事后词表校验（B3，附加式度量：量化不拦截，词表动态读 listen2serve.vocab）
        lex = check_user_lexicon(scenario, traj)

        # 收尾轮确定性检查（v7 强制收尾轮规则的可测化；纯词面规则，零 API 成本）
        clo = check_closing(traj)

        # K18c 泄露探针（操纵有效性检验）：盲评用户关键轮文本的字面态度表露量。
        # 与客服表现无关，用于把「客服听不出弦外之音」和「用户根本没把 T3 演成 T3」
        # 两种归因分开；只在 v2.3-J 下产出（judge_meta 的 leakage 指纹亦仅在该 schema 下登记）。
        lv = None
        if judge_schema == "v2.3-J":
            t0 = time.perf_counter()
            prev_agent = next(
                (h["content"] for h in reversed(p.history_before_critical)
                 if h["role"] == "assistant"), "")
            try:
                lv = b["leakage"].probe(
                    key_trace["user_text"], prev_agent, key_trace.get("user_state"))
            except Exception as exc: # noqa: BLE001 - 探针是附加度量，失败不影响主评测
                with probe_fail_lock:
                    probe_fail += 1
                print(f" [泄露探针失败] {task.scenario_id}: {exc}", flush=True)
            timings["leakage"] = time.perf_counter() - t0

        verdict = {
            "scenario_id": task.scenario_id,
            "role": scenario["role"],
            "leakage_level": scenario["leakage_level"],
            "leakage_label": scenario.get("leakage_label", ""),
            "dynamics": scenario.get("dynamics", ""),
            "modality": args.modality,
            "asr_text": asr_text,
            "agent_text": key_trace["agent_text"],
            # 旧 schema 字段保留（零破坏）：v2.3-J 下退役名置 None，聚合层改读新键
            "policy_pass": pv.passed if pv else None,
            "policy_detail": pv.as_dict() if pv else None,
            "voice_pass": vv.passed if vv else None,
            "voice_detail": vv.as_dict() if vv else None,
            "joint_pass": ((pv.passed if pv else False) and vv.passed) if (pv and vv) else None,
            "task_completion": dv.score if dv and dv.metric == "task_completion" else None,
            "transition_score": dv.score if dv and dv.metric == "transition_score" else None,
            "dialogue_detail": dv.as_dict() if dv else None,
            # ---- v2.3-J 增量键；v2.5-K 增 key_turn_score（1-5 MOS）----
            "key_turn_pass": ktv.passed if ktv else None,
            "key_turn_score": ktv.key_turn_score if ktv else None,
            "key_turn_detail": ktv.as_dict() if ktv else None,
            "key_behavior_met": (ktv.key_behavior_met if ktv else None),
            "flow_items": fv.flow_items if fv else None,
            "flow_rate": fv.flow_rate if fv else None,
            "task_score": fv.task_score if fv else None,
            # 截断窗读数的落盘（整通口径下这三个是 None，不改动既有字段）
            "flow_turn_cap": fv.turn_cap if fv else None,
            "flow_rate_in_window": fv.flow_rate_in_window if fv else None,
            "n_items_in_window": fv.n_items_in_window if fv else None,
            "flow_detail": fv.as_dict() if fv else None,
            "interaction": traj.get("interaction") or {},
            "termination_reason": traj.get("termination_reason", ""),
            # F1：裁判 prompt 版本追溯（policy/voice/dialogue 各自的版本与 hash）；
            # v5.9 契约内容维度标记 profile_injected：本条 verdict 的 policy/dialogue
            # 裁判契约末尾是否追加了同源客户信息档案（v5.8 断点可程序化甄别）
            "judge_meta": build_judge_meta(
                policy_model=policy_model or "",
                voice_model=voice_model or "",
                dialogue_model=policy_model or "",
                dialogue_metric=dv.metric if dv else None,
                profile_injected=bool(render_customer_profile(domain, scenario.get("db_seed"))),
                judge_schema=judge_schema,
            ),
            # 非关键轮词表校验（B3）：T2/T3 用户实际话术命中明确情绪词的违规度量
            "vocab_checked_turns": lex["vocab_checked_turns"],
            "vocab_violation_turns": lex["vocab_violation_turns"],
            "vocab_violation_rate": lex["vocab_violation_rate"],
            "vocab_violations": lex["vocab_violations"],
            # ---- 收尾轮确定性检查（closing_check）：用户已示意收尾时客服是否真的收尾 ----
            # 无词面收尾信号时 dangling/farewell/ok 为 None，聚合层按「不进分母」处理。
            "closing_signal": clo["closing_signal"],
            "closing_dangling": clo["closing_dangling"],
            "closing_farewell": clo["closing_farewell"],
            "closing_ok": clo["closing_ok"],
            "closing_last_agent_text": clo["closing_last_agent_text"],
            # ---- K18c 泄露探针（操纵有效性检验；期望 T1 > T2 > T3 单调）----
            # surface_score=0 表示解析失败 → 记 None，聚合层按「无有效值」剔除而非记 1 分
            "leakage_score": (lv.surface_score or None) if lv else None,
            "leakage_state_guess": (lv.state_guess or None) if lv else None,
            "leakage_state_correct": lv.state_correct if lv else None,
            "leakage_stance_markers": lv.stance_markers if lv else None,
            "leakage_detail": lv.as_dict() if lv else None,
            "user_key_text": key_trace["user_text"], # 探针被评对象（逐样本审阅用）
        }
        return JudgeOutcome(verdict=verdict, timings=timings)

    try:
        verdicts = run_incremental(
            tasks, _judge_scenario, store,
            concurrency=args.concurrency, fresh=args.fresh,
            log=lambda msg: print(msg, flush=True),
        )
    except PartialRejected as exc:
        print(exc)
        return 1

    out = report_dir / f"verdicts_{args.modality}.json"
    export_json(verdicts, out)
    if args.asr and args.modality == "audio":
        export_json(verdicts, report_dir / "verdicts_audio_with_asr.json")
    print(f"判定输出: {out}（{len(verdicts)} 条；逐场景增量落盘 {store.path} 保留审计）")
    # F5 解析失败统计：各 worker 独立计数器按 worker 聚合后汇总
    judge_names = ("keyturn", "flow", "voice") if judge_schema == "v2.3-J" else ("policy", "voice", "dialogue")

    def _per_worker(attr: str) -> str:
        return "/".join(
            f"{name}={sum(getattr(b[name], attr) for b in pool.instances)}"
            for name in judge_names
        )

    def _sum_counter(attr: str) -> int:
        return sum(getattr(b[name], attr) for b in pool.instances for name in judge_names)

    _retries = _sum_counter("parse_retries")
    _failures = _sum_counter("parse_failures")
    print(
        "裁判 JSON 解析（F5）: 重试 " + _per_worker("parse_retries")
        + "；重试后仍失败（落默认值）: " + _per_worker("parse_failures")
        + f"；合计 retries={_retries} failures={_failures}（worker 数 {len(pool.instances)}）"
    )
    # token 经济性：文本裁判链每场景平均输入 token（新旧链路对照验收）
    _text_names = [n for n in judge_names if n != "voice"]
    for name in _text_names:
        toks = sum(getattr(b[name], "input_tokens", 0) for b in pool.instances)
        calls = sum(getattr(b[name], "judge_calls", 0) for b in pool.instances)
        print(f"token 经济性: {name} 裁判输入合计 {toks} tokens / {calls} 次调用"
              f"（每场景均 {toks / len(verdicts):.0f}）")
    # 词表违规概览（内部方案：全轮同口径，含 T1 关键轮正向校验）：微平均，量化不拦截
    _vc = sum(v.get("vocab_checked_turns") or 0 for v in verdicts)
    _vv = sum(v.get("vocab_violation_turns") or 0 for v in verdicts)
    if _vc:
        print(f"用户话术词表校验（全轮）: {_vv}/{_vc} 轮违规（vocab_violation_rate={_vv / _vc:.4f}）")
    # K18c 泄露探针概览：把「三层透明度是否真的分级」当场打出来——若不单调，
    # 该批次客服侧的层间差异不可解读，需先回头修数据/演绎侧而不是直接写结论。
    _by_level: dict[str, list[int]] = {}
    for v in verdicts:
        s = v.get("leakage_score")
        if isinstance(s, (int, float)):
            # 用 leakage_label（T1/T2/T3）而非 leakage_level（explicit/implicit_consistent/
            # prosody_only）：两者字母序恰好同序，但 label 可读、与论文口径一致。
            _by_level.setdefault(str(v.get("leakage_label", "")), []).append(s)
    if _by_level:
        parts = "、".join(
            f"{lv}={sum(xs) / len(xs):.2f}(n={len(xs)})"
            for lv, xs in sorted(_by_level.items())
        )
        _means = [sum(xs) / len(xs) for _, xs in sorted(_by_level.items())]
        mono = "单调递减" if all(a > b for a, b in zip(_means, _means[1:])) else "非单调！"
        print(f"泄露探针（字面态度表露量 1-5）: {parts} → {mono}")
        _correct = [v.get("leakage_state_correct") for v in verdicts]
        _correct = [c for c in _correct if c is not None]
        if _correct:
            print(f"仅凭文字猜中真实状态: {sum(1 for c in _correct if c)}/{len(_correct)}")
    if probe_fail:
        print(f"泄露探针失败 {probe_fail} 条（附加度量，不影响主指标）")
    return 0


def _report(args: argparse.Namespace) -> int:
    from listen2serve.report.aggregate import aggregate_results
    from listen2serve.report.render_md import export_json, render_md_report
    from listen2serve.runtime.manifest import load_run_manifest

    verdicts_path = Path("reports") / args.run_id / f"verdicts_{args.modality}.json"
    if not verdicts_path.exists():
        print(f"缺少判定结果: {verdicts_path}（先运行 evaluate）")
        return 1
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    aggregated = aggregate_results(verdicts)
    # 内部方案：报告头附运行溯源（git_commit 前 8 位 / dataset_version / models）；
    # 旧 run 无 manifest 时显示「manifest 缺失（旧 run）」
    manifest = load_run_manifest(Path("runs") / args.run_id)
    aggregated["provenance"] = _provenance_summary(args.run_id, manifest)
    out_dir = Path("reports") / args.run_id
    export_json(aggregated, out_dir / f"aggregated_{args.modality}.json")
    (out_dir / f"report_{args.modality}.md").write_text(render_md_report(aggregated), encoding="utf-8")
    print(f"报告输出: {out_dir}/report_{args.modality}.md + aggregated_{args.modality}.json")
    return 0


def _provenance_summary(run_id: str, manifest: dict | None) -> dict:
    """内部方案：报告头溯源摘要；旧 run 无 manifest 时如实标记缺失。"""
    if manifest is None:
        return {"run_id": run_id, "manifest": "缺失（旧 run）"}
    code = manifest.get("code") or {}
    commit = code.get("git_commit")
    from listen2serve.runtime.manifest import observed_judge
    oj = observed_judge(manifest)
    models = dict(manifest.get("models") or {})
    # 展示用的裁判一栏取**实测/回写值**：models.judge 是生成步写的占位，
    # 旧 run 里它是占位默认名，照原样渲染会把没参与判定的模型
    # 呈现给评审。三者不一致时把两个值都留下（conflict 标记），不做静默取舍。
    if oj.get("judge"):
        models["judge"] = oj["judge"]
    if oj.get("voice_judge"):
        models["voice_judge"] = oj["voice_judge"]
    if oj.get("conflict"):
        models["judge_generation_claim"] = oj.get("generation_claim")
        models["judge_observed"] = oj.get("observed")
    return {
        "run_id": run_id,
        "manifest": "ok",
        "git_commit": commit[:8] if commit else None,
        "git_dirty": bool(code.get("git_dirty")),
        "fallback_tree_hash": (code.get("fallback_tree_hash") or "")[:8] or None,
        "dataset_version": str((manifest.get("data") or {}).get("dataset_version") or ""),
        "models": models,
        "created_at": manifest.get("created_at", ""),
    }




















def main() -> int:
    parser = argparse.ArgumentParser(prog="Listen2Serve", description="Listen2Serve CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)


    p_run = sub.add_parser("run-eval", help="批量运行评测")
    p_run.add_argument("--scenarios", default="data/benchmark/scenarios.jsonl")
    p_run.add_argument("--model", default=None, help="被测模型，如 dashscope/qwen-audio-3.0-realtime-plus")
    p_run.add_argument("--role", default=None, choices=["collection", "marketing", "hotline"])
    p_run.add_argument("--leakage", default=None, choices=["explicit", "implicit_consistent", "prosody_only"])
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--scenario-ids", default=None,
                       help="逗号分隔 scenario_id 精确选取（优先于其他过滤）")
    p_run.add_argument("--per-role", type=int, default=None,
                       help="每角色取前 N 条（最小冒烟约束：三角色至少各 1 条，即 --per-role 1）")
    p_run.add_argument("--run-id", default=None,
                       help="默认 run_%%Y%%m%%d_%%H%%M%%S；冒烟建议 smoke_%%Y%%m%%d_%%H%%M%%S（按时间排序）")
    p_run.add_argument("--concurrency", type=int, default=None)
    p_run.add_argument("--sim-model", default=None,
                       help="用户模拟器演绎模型（覆盖 SIM_MODEL/llm_model），如 qwen-plus（裸名按绑定表解析，未收录报错）")
    p_run.add_argument("--sim-prompt-schema", default="v6.0", choices=["v6.0", "v6.1", "v7"],
                       help="演绎 prompt schema：v6.0 旧区块式（默认）/ v6.1 四块式减负，A/B 验收后切默认")
    p_run.add_argument("--prosody-arm", default="state", choices=["state", "neutral"],
                       help="2×2 韵律因子臂：state（默认，B/D 两格、带 state 韵律）/ neutral（A/C 两格、韵律恒中性，instruct 取 non_key.neutral 且不附强制标签）")
    p_run.add_argument("--sim-thinking", action="store_true",
                       help="开演用户 LLM 的 reasoning（A/B 对照用）。只作用于用户模拟器，"
                            "被测 agent 走 realtime 音频通路不受影响；默认关以控成本")
    p_run.add_argument("--no-audio", action="store_true", help="不保存音频（评测仍需音频，仅跳过落盘）")
    # choices 直接取 AGENT_PROMPT_VERSIONS（单一来源）：此前这里硬编码版本清单，与 base.py
    # 的版本元组各写一份，新增 v9 时就会漏（口径分散在多份手写副本里）。
    p_run.add_argument("--agent-prompt", default=None, choices=list(AGENT_PROMPT_VERSIONS),
                       help="被测 Agent prompt 版本（默认取 .env AGENT_PROMPT，缺省 v9）："
                            "v9=本轮正式口径（v7 精简板块 + 与裁判同源的「状态→动作速查表」，"
                            "不再渲染 state_playbook）；v9_notable=v9 减速查表（测内化臂）；"
                            "v7=听→做为任务主体、状态优先于流程、强制收尾轮、文本策略精简；"
                            "v6=旧口径；"
                            "⚠️ v8 含 variant 透传缺陷，仅供历史 run 复算，禁止用于新实验。"
                            "evaluate 侧须传同一版本（裁判契约同源）")

    p_eval = sub.add_parser("evaluate", help="对轨迹运行 Judge")
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument("--scenarios", default="data/benchmark/scenarios.jsonl")
    p_eval.add_argument("--modality", default="audio", choices=["audio", "text"],
                        help="audio=原始音频+真实转写；text=对音频 ASR 后用转写文本评测（Voice 跳过）")
    p_eval.add_argument("--asr", action="store_true", help="audio 模态下也记录 ASR 转写（用于对比分析）")
    p_eval.add_argument("--judge-model", default=None,
                        help="Policy/对话级裁判模型（无自动回退），裸名按绑定表解析平台，如 qwen-plus（dashscope 百炼）")
    p_run.add_argument("--max-retries", type=int, default=None,
                        help="单场景批内重试次数（缺省沿用 BatchRunner 默认 2）。设 0 用于外层已按轮"
                             "补跑的场合：批内重试会重复烧时间，且立刻重试更易撞未释放的僵尸会话槽")
    p_eval.add_argument("--flow-turn-cap", type=int, default=0,
                        help="只把对话前 K 轮交给 FlowJudge（截断观察窗）；0=整通（默认，既有口径）。"
                             "用于会话寿命不足以跑完整通的端点；跨端点比较时四家必须同一个 K")
    p_eval.add_argument("--voice-judge-model", default=None,
                        help="Voice 裁判模型（无自动回退）；默认 qwen3.5-omni-plus（dashscope），备选 openai/<名>（配 OPENAI_BASE_URL/OPENAI_API_KEY）")
    p_eval.add_argument("--concurrency", type=int, default=1,
                        help="并发 worker 数（默认 1 保守；IO 等待型负载建议 4）；每 worker 独立 judge 实例")
    p_eval.add_argument("--fresh", action="store_true",
                        help="忽略既有 partial 全量重评（既有 partial 归档保留审计）")
    p_eval.add_argument("--legacy-judges", action="store_true",
                        help="切回 v2.2-I 旧裁判链路（PolicyJudge+对话级双裁判）；默认 v2.3-J（KeyTurn+Flow）")
    p_eval.add_argument("--report-id", default=None,
                        help="报告输出目录名（默认同 run-id）；重评新裁判时建议独立目录不覆盖旧判定")
    p_eval.add_argument("--agent-prompt", default=None, choices=list(AGENT_PROMPT_VERSIONS),
                        help="裁判契约按哪个版本的 Agent prompt 渲染（默认取 .env AGENT_PROMPT，缺省 v9）。"
                             "必须与该 run 生成时的版本一致，否则裁判会按被测方当时没收到的规则打分："
                             "v7 之前产出的 run 重评须显式传 v6；v7/v8 产出的 run 重评须显式传 v7/v8；"
                             "v9 产出的 run 重评须传 v9"
                             "（v9 的契约同步剔除了字数约束与红线重复项，传 v7 会多判几条禁止）。"
                             "⚠️ 2026-09-06 起缺省值已是 v9，重评 v9 之前的旧 run **不传参即口径错配**")

    p_rep = sub.add_parser("report", help="聚合渲染报告")
    p_rep.add_argument("--run-id", required=True)
    p_rep.add_argument("--modality", default="audio", choices=["audio", "text"])










    args = parser.parse_args()
    # --agent-prompt 经环境变量落到 Settings：被测 prompt 与裁判契约都在渲染期读配置，
    # 走同一入口才能保证两侧口径一致（Settings 有 lru_cache，需先清）
    if getattr(args, "agent_prompt", None):
        import os

        from listen2serve.runtime.config import get_settings

        os.environ["AGENT_PROMPT"] = args.agent_prompt
        get_settings.cache_clear()

        _sys.argv = ["probe"] + (["--out", args.out] if args.out else [])
        return probe_main()
    if args.cmd == "run-eval":
        if args.model is None:
            from listen2serve.runtime.realtime.registry import default_model_spec

            args.model = default_model_spec() # 默认被测模型（Realtime），而非裁判 LLM
        args.save_audio = not getattr(args, "no_audio", False)
        return asyncio.run(_run_eval(args))
    if args.cmd == "evaluate":
        return _evaluate(args)
    if args.cmd == "report":
        return _report(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
