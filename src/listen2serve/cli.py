"""Run multi-turn scenarios, evaluate saved trajectories, and render reports."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from listen2serve.conditions import condition_code, wording_for
from listen2serve.domains import get_domain
from listen2serve.domains.base import render_customer_profile, render_judge_contract
from listen2serve.gateway.asr_gateway import ASRGateway


def _load_scenarios(
    path: str,
    limit: int | None,
    role: str | None = None,
    wording: str | None = None,
    scenario_ids: str | None = None,
    per_role: int | None = None,
) -> list[dict]:
    wanted = {s.strip() for s in scenario_ids.split(",")} if scenario_ids else None
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            scenario = json.loads(line)
            if (
                wanted is not None
                and scenario.get("scenario_id") not in wanted
                and scenario.get("base_scenario_id") not in wanted
            ):
                continue
            if role and scenario.get("role") != role:
                continue
            if wording and wording_for(scenario) != wording:
                continue
            rows.append(scenario)
    if wanted:
        found = {s.get("scenario_id") for s in rows} | {s.get("base_scenario_id") for s in rows}
        missing = wanted - found
        if missing:
            raise ValueError(f"未找到请求的场景 ID: {', '.join(sorted(missing))}")
    if per_role:
        picked, counts = [], {}
        for scenario in rows:
            r = scenario["role"]
            if counts.get(r, 0) < per_role:
                picked.append(scenario)
                counts[r] = counts.get(r, 0) + 1
        rows = picked
    return rows[:limit] if limit else rows


async def _run_eval(args: argparse.Namespace) -> int:
    from listen2serve.model_registry import resolve_model
    from listen2serve.runtime.config import get_settings
    from listen2serve.runtime.manifest import build_run_manifest
    from listen2serve.runtime.preflight import require_generation_credentials
    from listen2serve.runtime.run_batch import BatchRunner

    settings = get_settings()
    scenarios = _load_scenarios(
        args.scenarios, args.limit, args.role, args.wording, args.scenario_ids, args.per_role
    )
    if not scenarios:
        print(f"无场景可运行: {args.scenarios}")
        return 1
    require_generation_credentials(settings, [args.model], args.sim_model)
    backend, _ = resolve_model(args.model, settings.llm_model)
    api_key = settings.volc_key() if backend == "volc" else settings.dashscope_api_key
    runner = BatchRunner(
        settings=settings,
        run_id=args.run_id,
        concurrency=args.concurrency,
        save_audio=not args.no_audio,
        sim_model=args.sim_model,
        prosody_arm=args.prosody_arm,
        sim_thinking=args.sim_thinking,
        **({"max_retries": args.max_retries} if args.max_retries is not None else {}),
    )
    manifest = build_run_manifest(
        run_id=runner.run_id,
        models={
            "target": args.model,
            "sim": runner.sim_model or "",
            "judge": None,
            "voice_judge": None,
            "tts": f"{settings.tts_backend}/{settings.tts_model}",
        },
        cli_args=vars(args),
        settings=settings,
    )
    print(f"运行 {len(scenarios)} 条场景 → runs/{runner.run_id}", flush=True)
    results = await runner.run_batch(scenarios, args.model, api_key, domain=None, manifest=manifest)
    complete = sum(r.status == "ok" and r.termination_reason != "endpoint_session_closed" for r in results)
    partial = sum(
        r.status == "partial" or (r.status == "ok" and r.termination_reason == "endpoint_session_closed")
        for r in results
    )
    failed = len(results) - complete - partial
    print(f"运行完成: 完整 {complete}，部分 {partial}，失败 {failed} → runs/{runner.run_id}")
    return 0 if failed == 0 else 1


@dataclass
class _EvalTaskPayload:
    scenario: dict
    domain: object
    traj: dict
    key_trace: dict
    history_with_agent: list
    history_before_critical: list


def _content_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _audio_hashes(traj: dict) -> dict:
    """Bind cached verdicts to file contents, including files changed in place."""
    hashes = {}
    for i, turn in enumerate(traj.get("turns") or []):
        for key in ("agent_audio_path", "user_audio_path"):
            raw = turn.get(key)
            if raw:
                path = Path(raw)
                hashes[f"{i}:{key}"] = (
                    hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                )
    return hashes


def _build_eval_tasks(
    runs_root: Path,
    by_id: dict[str, dict],
    policy_model: str,
    voice_model: str,
    *,
    modality: str = "audio",
    use_asr: bool = False,
    flow_turn_cap: int | None = None,
    asr_model: str = "",
    backend_config: dict | None = None,
) -> list:
    from listen2serve.evaluation.incremental import EvalTask, dialogue_is_complete
    from listen2serve.evaluation.prompts import build_judge_meta
    from listen2serve.runtime.manifest import sha256_dir

    source_digest = sha256_dir(Path(__file__).resolve().parent)
    tasks = []
    for status_path in sorted(runs_root.glob("*/sim_status.json")):
        traj = json.loads(status_path.read_text(encoding="utf-8"))
        if traj.get("status") not in ("ok", "partial"):
            continue
        sid = traj["scenario_id"]
        if sid not in by_id:
            raise ValueError(f"轨迹缺少场景定义: {sid}")
        scenario = by_id[sid]
        domain = get_domain(scenario["role"])
        key_traces = [t for t in traj["turns"] if t.get("is_key_turn")]
        if len(key_traces) != 1:
            raise ValueError(f"轨迹必须有且只有一个关键轮: {sid}（实际 {len(key_traces)}）")
        key_trace = key_traces[0]
        critical = key_trace["turn"]
        earlier_timeouts = [
            turn["turn"] for turn in traj["turns"] if turn.get("timed_out") and turn["turn"] < critical
        ]
        if earlier_timeouts:
            raise ValueError(
                f"关键轮之前存在响应超时: {sid}（第 {', '.join(map(str, earlier_timeouts))} 轮）；"
                "关键轮可能混入前轮残片，拒绝评测"
            )
        if key_trace.get("endpoint_interrupted") or key_trace.get("timed_out"):
            raise ValueError(f"关键轮客服回复不完整: {sid}")
        if not dialogue_is_complete(traj) and not str(key_trace.get("agent_text") or "").strip():
            raise ValueError(f"部分轨迹缺少关键轮客服回复: {sid}")
        history, before = [], []
        for turn in traj["turns"]:
            if turn.get("endpoint_interrupted") or (
                not dialogue_is_complete(traj) and turn["turn"] > critical
            ):
                continue
            user = {"role": "user", "content": turn["user_text"]}
            history.append(user)
            if turn["turn"] <= critical:
                before.append(user)
            if turn.get("agent_text"):
                reply = {"role": "assistant", "content": turn["agent_text"]}
                history.append(reply)
                if turn["turn"] < critical:
                    before.append(reply)
        meta = build_judge_meta(
            policy_model,
            voice_model,
            profile_injected=bool(render_customer_profile(domain, scenario.get("db_seed"))),
        )
        meta["source_tree_sha256"] = source_digest
        meta["inputs"] = {
            "trajectory_sha256": _content_hash(traj),
            "scenario_sha256": _content_hash(scenario),
            "contract_sha256": _content_hash(render_judge_contract(domain, scenario)),
            "audio_sha256": _audio_hashes(traj),
        }
        meta["configuration"] = {
            "modality": modality,
            "asr": use_asr,
            "asr_model": asr_model if use_asr else None,
            "flow_turn_cap": flow_turn_cap,
            "backends": backend_config or {},
        }
        tasks.append(
            EvalTask(sid, meta, _EvalTaskPayload(scenario, domain, traj, key_trace, history, before))
        )
    return tasks


def _asr_histories(payload: _EvalTaskPayload, gateway: ASRGateway) -> tuple[str, list, list]:
    """Use one ASR source for every agent utterance passed to the text judges."""
    from listen2serve.evaluation.incremental import dialogue_is_complete

    history, before = [], []
    key_text = None
    critical = payload.key_trace["turn"]
    for turn in payload.traj["turns"]:
        if turn.get("endpoint_interrupted") or (
            not dialogue_is_complete(payload.traj) and turn["turn"] > critical
        ):
            continue
        user = {"role": "user", "content": turn["user_text"]}
        history.append(user)
        if turn["turn"] <= critical:
            before.append(user)
        if not turn.get("agent_text") and not turn.get("agent_audio_path"):
            continue
        audio = turn.get("agent_audio_path")
        if not audio or not Path(audio).is_file():
            raise ValueError(f"ASR 缺少客服音频: {payload.traj['scenario_id']} 第 {turn['turn']} 轮")
        result = gateway.transcribe(audio)
        text = result.text.strip() if isinstance(result.text, str) else ""
        if not text:
            raise ValueError(f"ASR 返回空文本: {payload.traj['scenario_id']} 第 {turn['turn']} 轮")
        reply = {"role": "assistant", "content": text}
        history.append(reply)
        if turn["turn"] < critical:
            before.append(reply)
        if turn["turn"] == critical:
            key_text = text
    if key_text is None:
        raise ValueError("ASR 未得到关键轮客服文本")
    return key_text, history, before


def _evaluate(args: argparse.Namespace) -> int:
    from listen2serve.evaluation.closing_check import check_closing
    from listen2serve.evaluation.flow_judge import FlowJudge
    from listen2serve.evaluation.incremental import (
        JudgeOutcome,
        PartialStore,
        WorkerLocalPool,
        dialogue_is_complete,
        partial_path,
        run_incremental,
    )
    from listen2serve.evaluation.leakage_probe import LeakageProbe
    from listen2serve.evaluation.lexicon_check import check_user_lexicon
    from listen2serve.evaluation.policy_judge import KeyTurnJudge
    from listen2serve.evaluation.voice_judge import VoiceJudge
    from listen2serve.model_registry import resolve_model
    from listen2serve.report.export_json import export_json
    from listen2serve.runtime.config import get_settings
    from listen2serve.runtime.manifest import load_run_manifest, write_run_manifest
    from listen2serve.runtime.preflight import require_evaluation_credentials

    settings = get_settings()
    runs_root = Path("runs") / args.run_id
    if not runs_root.is_dir():
        raise ValueError(f"runs/{args.run_id} 不存在")
    flow_turn_cap = args.flow_turn_cap or None
    if flow_turn_cap is not None and flow_turn_cap < 1:
        raise ValueError("--flow-turn-cap 必须为非负整数")
    policy_model = "/".join(resolve_model(args.judge_model or settings.judge_model, settings.llm_model))
    voice_model = "/".join(resolve_model(args.voice_judge_model, settings.voice_judge_model))
    use_asr = args.modality == "text" or args.asr
    scenarios = _load_scenarios(args.scenarios, None)
    by_id = {s["scenario_id"]: s for s in scenarios}
    # Endpoint changes also invalidate cached verdicts; credentials are never recorded.
    backends = {
        key: value
        for key, value in settings.model_dump().items()
        if key.endswith("_base_url") or key == "asr_model"
    }
    tasks = _build_eval_tasks(
        runs_root,
        by_id,
        policy_model,
        voice_model,
        modality=args.modality,
        use_asr=use_asr,
        flow_turn_cap=flow_turn_cap,
        asr_model=settings.asr_model,
        backend_config=backends,
    )
    if not tasks:
        raise ValueError("无可评测轨迹（需要 status=ok 且关键轮齐全）")
    report_dir = Path("reports") / (args.report_id or args.run_id)
    store = PartialStore(partial_path(report_dir, args.modality))
    # A fully cached evaluation can be read without credentials. Mismatched rows
    # are rejected by run_incremental before any API call, preserving its error.
    records = {} if args.fresh else store.load()[0]
    mismatched = any(
        task.scenario_id in records and records[task.scenario_id].get("judge_meta") != task.expected_meta
        for task in tasks
    )
    pending = any(task.scenario_id not in records for task in tasks)
    if pending and not mismatched:
        require_evaluation_credentials(
            settings,
            judge_model=policy_model,
            voice_judge_model=voice_model,
            modality=args.modality,
            use_asr=use_asr,
        )
    if args.fresh and store.path.exists():
        stale = store.path.with_suffix(f".stale_{time.time_ns()}.jsonl")
        store.path.rename(stale)
        print(f"--fresh: 既有 partial 归档为 {stale}", flush=True)
    probe_fail_lock, probe_fail = threading.Lock(), 0

    def make_judges():
        return {
            "voice": VoiceJudge(llm_model=voice_model),
            "asr": ASRGateway(settings) if use_asr else None,
            "keyturn": KeyTurnJudge(llm_model=policy_model),
            "flow": FlowJudge(llm_model=policy_model),
            "leakage": LeakageProbe(llm_model=policy_model),
        }

    pool = WorkerLocalPool(make_judges)

    def judge_scenario(task) -> JudgeOutcome:
        nonlocal probe_fail
        p = task.payload
        judges = pool.get()
        # Capture only this task's calls from worker-local judge instances.
        offsets = {
            service: len(getattr(getattr(getattr(judge, "llm", judge), "sink", None), "records", []))
            for service, judge in judges.items()
        }
        scenario, domain, traj, key = p.scenario, p.domain, p.traj, p.key_trace
        timings = {}
        text, history, before = key["agent_text"], p.history_with_agent, p.history_before_critical
        asr_text = None
        if use_asr:
            t0 = time.perf_counter()
            text, history, before = _asr_histories(p, judges["asr"])
            asr_text = text
            timings["asr"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        keyturn = judges["keyturn"].judge(domain, scenario, before, text)
        timings["keyturn"] = time.perf_counter() - t0
        voice = None
        if args.modality == "audio":
            audio = key.get("agent_audio_path")
            if not audio or not Path(audio).is_file():
                raise ValueError(f"Voice 裁判缺少关键轮音频: {task.scenario_id}")
            ap = scenario.get("agent_policy") or {}
            t0 = time.perf_counter()
            voice = judges["voice"].judge(
                audio,
                ap.get("role_description", ""),
                key["user_state"],
                text,
                voice_requirements=ap.get("voice_requirements") or domain.default_voice_requirements,
                communication_style=ap.get("communication_style"),
            )
            timings["voice"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        complete = dialogue_is_complete(traj)
        flow = judges["flow"].judge(domain, scenario, history, flow_turn_cap) if complete else None
        if complete:
            timings["flow"] = time.perf_counter() - t0
        lex = check_user_lexicon(scenario, traj)
        closing = (
            check_closing(traj)
            if complete
            else dict.fromkeys(
                (
                    "closing_signal",
                    "closing_dangling",
                    "closing_farewell",
                    "closing_ok",
                    "closing_last_agent_text",
                )
            )
        )
        leakage = None
        t0 = time.perf_counter()
        try:
            previous = next((h["content"] for h in reversed(before) if h["role"] == "assistant"), "")
            leakage = judges["leakage"].probe(key["user_text"], previous, key.get("user_state"))
        except Exception as exc:  # noqa: BLE001 - auxiliary probe failures are reported separately
            with probe_fail_lock:
                probe_fail += 1
            print(f"附加文字表露量探针失败 {task.scenario_id}: {exc}", flush=True)
        timings["leakage"] = time.perf_counter() - t0
        identities = {}
        for service, judge in judges.items():
            sink = getattr(getattr(judge, "llm", judge), "sink", None)
            identities[service] = [
                record.as_dict() for record in getattr(sink, "records", [])[offsets[service] :]
            ]
            identities[service] = [
                {
                    key: value
                    for key, value in record.items()
                    if key
                    in {
                        "service",
                        "backend",
                        "requested_model",
                        "response_model",
                        "requested_host",
                        "provider_host",
                        "request_id",
                        "response_model_differs",
                    }
                }
                for record in identities[service]
            ]
        verdict = {
            "model_identity_schema": "requested-response-v1",
            "model_identity": identities,
            "scenario_id": task.scenario_id,
            "role": scenario["role"],
            "wording": wording_for(scenario),
            "condition": condition_code(wording_for(scenario), traj["prosody_arm"])
            if traj.get("prosody_arm")
            else None,
            "dynamics": scenario.get("dynamics", ""),
            "modality": args.modality,
            "asr_text": asr_text,
            "agent_text": key["agent_text"],
            "evaluated_text": text,
            "text_source": "asr" if use_asr else "endpoint",
            "voice_pass": voice.passed if voice else None,
            "voice_detail": voice.as_dict() if voice else None,
            "key_turn_pass": keyturn.passed,
            "key_turn_score": keyturn.key_turn_score,
            "key_turn_detail": keyturn.as_dict(),
            "key_behavior_met": keyturn.key_behavior_met,
            "flow_items": flow.flow_items if flow else None,
            "flow_rate": flow.flow_rate if flow else None,
            "task_score": flow.task_score if flow else None,
            "flow_turn_cap": flow.turn_cap if flow else flow_turn_cap,
            "flow_rate_in_window": flow.flow_rate_in_window if flow else None,
            "n_items_in_window": flow.n_items_in_window if flow else None,
            "flow_detail": flow.as_dict() if flow else None,
            "trajectory_status": "ok" if complete else "partial",
            "dialogue_complete": complete,
            "interaction": traj.get("interaction") or {},
            "termination_reason": traj.get("termination_reason", ""),
            "judge_meta": task.expected_meta,
            "vocab_checked_turns": lex["vocab_checked_turns"],
            "vocab_violation_turns": lex["vocab_violation_turns"],
            "vocab_violation_rate": lex["vocab_violation_rate"],
            "vocab_violations": lex["vocab_violations"],
            **{
                name: closing[name]
                for name in (
                    "closing_signal",
                    "closing_dangling",
                    "closing_farewell",
                    "closing_ok",
                    "closing_last_agent_text",
                )
            },
            "leakage_score": leakage.surface_score or None if leakage else None,
            "leakage_state_guess": leakage.state_guess or None if leakage else None,
            "leakage_state_correct": leakage.state_correct if leakage else None,
            "leakage_stance_markers": leakage.stance_markers if leakage else None,
            "leakage_detail": leakage.as_dict() if leakage else None,
            "user_key_text": key["user_text"],
        }
        return JudgeOutcome(verdict, timings)

    verdicts = run_incremental(
        tasks,
        judge_scenario,
        store,
        concurrency=args.concurrency,
        fresh=args.fresh,
        log=lambda msg: print(msg, flush=True),
    )
    out = report_dir / f"verdicts_{args.modality}.json"
    export_json(verdicts, out)
    if use_asr and args.modality == "audio":
        export_json(verdicts, report_dir / "verdicts_audio_with_asr.json")
    manifest = load_run_manifest(runs_root)
    if manifest is not None:
        manifest["evaluation"] = {
            "judge_model": policy_model,
            "model_identity_schema": "requested-response-v1",
            "keyturn_model_requested": policy_model,
            "keyturn_model_observed": _single_reported_model(verdicts, "keyturn"),
            "voice_model_observed": _single_reported_model(verdicts, "voice"),
            "voice_judge_model": voice_model if args.modality == "audio" else None,
            "modality": args.modality,
            "asr": use_asr,
            "asr_model": settings.asr_model if use_asr else None,
            "flow_turn_cap": flow_turn_cap,
            "verdicts_path": str(out),
            "verdict_count": len(verdicts),
            "verdicts_sha256": _content_hash(verdicts),
            "judge_meta_sha256": _content_hash([v["judge_meta"] for v in verdicts]),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model_identity": {
                service: _report_identity_rows(verdicts, service)
                for service in ("keyturn", "voice", "flow", "asr", "leakage")
            },
        }
        write_run_manifest(runs_root, manifest)
    print(f"判定输出: {out}（{len(verdicts)} 条）")
    for name in ("keyturn", "flow", "voice"):
        retries = sum(getattr(j[name], "parse_retries", 0) for j in pool.instances)
        print(f"{name}: JSON 重试 {retries} 次")
    if probe_fail:
        print(f"附加文字表露量探针失败 {probe_fail} 条（相应值为空）")
    return 0


def _report(args: argparse.Namespace) -> int:
    from listen2serve.report.aggregate import aggregate_results
    from listen2serve.report.export_json import export_json
    from listen2serve.report.render_md import render_md_report
    from listen2serve.runtime.manifest import load_run_manifest

    report_id = args.report_id or args.run_id
    out_dir = Path("reports") / report_id
    verdicts_path = out_dir / f"verdicts_{args.modality}.json"
    if not verdicts_path.is_file():
        raise ValueError(f"缺少判定结果: {verdicts_path}（先运行 evaluate）")
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    aggregated = aggregate_results(verdicts)
    aggregated["provenance"] = _provenance_summary(
        args.run_id, load_run_manifest(Path("runs") / args.run_id), verdicts
    )
    export_json(aggregated, out_dir / f"aggregated_{args.modality}.json")
    (out_dir / f"report_{args.modality}.md").write_text(render_md_report(aggregated), encoding="utf-8")
    print(f"报告输出: {out_dir}/report_{args.modality}.md + aggregated_{args.modality}.json")
    return 0


def _report_identity_rows(verdicts: list[dict], service: str) -> list[dict]:
    from listen2serve.gateway.base import summarize_identities

    return summarize_identities(
        [
            record
            for verdict in verdicts
            if verdict.get("model_identity_schema") == "requested-response-v1"
            for record in (verdict.get("model_identity") or {}).get(service, [])
        ]
    )


def _single_reported_model(verdicts: list[dict], service: str) -> str | None:
    models = {row.get("response_model") for row in _report_identity_rows(verdicts, service)} - {None}
    return next(iter(models)) if len(models) == 1 else None


def _provenance_summary(run_id: str, manifest: dict | None, verdicts: list[dict] | None = None) -> dict:
    if manifest is None:
        return {"run_id": run_id, "manifest": "缺失"}
    from listen2serve.runtime.manifest import observed_judge

    code, models = manifest.get("code") or {}, dict(manifest.get("models") or {})
    observed = observed_judge(manifest)
    if observed.get("judge"):
        models["judge"] = observed["judge"]
    if observed.get("voice_judge"):
        models["voice_judge"] = observed["voice_judge"]
    if verdicts:
        # A run can have several report directories; use this report's own judge identities.
        keyturn_models = sorted(
            {(v.get("judge_meta") or {}).get("keyturn", {}).get("model", "") for v in verdicts} - {""}
        )
        voice_models = sorted(
            {
                (v.get("judge_meta") or {}).get("voice", {}).get("model", "")
                for v in verdicts
                if v.get("voice_detail") is not None
            }
            - {""}
        )
        if keyturn_models:
            models["judge"] = ", ".join(keyturn_models)
        models["voice_judge"] = ", ".join(voice_models) or None
    models["judge_response_model"] = (
        _single_reported_model(verdicts, "keyturn") if verdicts else observed.get("observed")
    )
    models["voice_judge_response_model"] = (
        _single_reported_model(verdicts, "voice") if verdicts else observed.get("voice_observed")
    )
    return {
        "run_id": run_id,
        "manifest": "ok",
        "model_identity": {
            service: _report_identity_rows(verdicts or [], service)
            for service in ("keyturn", "voice", "flow", "asr", "leakage")
        },
        "git_commit": (code.get("git_commit") or "")[:8] or None,
        "git_dirty": bool(code.get("git_dirty")),
        "fallback_tree_hash": (code.get("fallback_tree_hash") or "")[:8] or None,
        "dataset_version": str((manifest.get("data") or {}).get("dataset_version") or ""),
        "models": models,
        "created_at": manifest.get("created_at", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="l2s", description="Listen2Serve multi-turn evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-eval", help="运行多轮场景")
    run.add_argument("--scenarios", default="data/benchmark/scenarios.jsonl")
    run.add_argument("--model", default=None, help="被测实时语音模型，platform/model")
    run.add_argument("--role", choices=["collection", "marketing", "hotline"])
    run.add_argument("--wording", choices=["explicit", "neutral"], help="论文中的显性或中性措辞条件")
    run.add_argument("--limit", type=int)
    run.add_argument("--scenario-ids", help="逗号分隔的 scenario_id 或 base_scenario_id")
    run.add_argument("--per-role", type=int, help="每个角色最多运行 N 条")
    run.add_argument("--run-id", help="输出目录名；默认使用运行时间")
    run.add_argument("--concurrency", type=int)
    run.add_argument("--sim-model", help="用户模拟器模型；默认 kimi-k3")
    run.add_argument(
        "--prosody-arm", default="state", choices=["state", "neutral"], help="状态韵律或中性韵律"
    )
    run.add_argument("--sim-thinking", action="store_true", help="启用用户模拟器的 reasoning")
    run.add_argument("--no-audio", action="store_true", help="运行时不保存音频，后续音频评分不可用")
    run.add_argument("--max-retries", type=int, help="单场景生成重试次数；默认 2")
    evaluate = sub.add_parser("evaluate", help="对已保存轨迹评分")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--scenarios", default="data/benchmark/scenarios.jsonl")
    evaluate.add_argument(
        "--modality", default="audio", choices=["audio", "text"], help="text 使用 ASR 文本并跳过声音评分"
    )
    evaluate.add_argument("--asr", action="store_true", help="使用统一 ASR 作为文本裁判的输入；失败时中止")
    evaluate.add_argument("--judge-model", help="关键轮和全对话裁判模型；默认 google/gemini-3.8-flash")
    evaluate.add_argument("--voice-judge-model", help="语音裁判模型；默认 google/gemini-3.8-flash")
    evaluate.add_argument(
        "--flow-turn-cap", type=int, default=0, help="全对话裁判仅观察前 K 轮；0 表示完整对话"
    )
    evaluate.add_argument("--concurrency", type=int, default=1)
    evaluate.add_argument("--fresh", action="store_true", help="归档已有 partial 并重新评分")
    evaluate.add_argument("--report-id", help="报告输出目录名；默认同 run-id")
    report = sub.add_parser("report", help="聚合并渲染报告")
    report.add_argument("--run-id", required=True)
    report.add_argument("--report-id", help="读取指定报告目录；默认同 run-id")
    report.add_argument("--modality", default="audio", choices=["audio", "text"])
    args = parser.parse_args()
    try:
        if args.cmd == "run-eval":
            if args.model is None:
                from listen2serve.runtime.realtime.registry import default_model_spec

                args.model = default_model_spec()
            return asyncio.run(_run_eval(args))
        if args.cmd == "evaluate":
            return _evaluate(args)
        return _report(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
