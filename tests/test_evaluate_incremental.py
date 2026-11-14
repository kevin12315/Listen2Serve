"""Offline regression checks for evaluate incremental."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from listen2serve.evaluation.incremental import (
    EvalTask,
    JudgeOutcome,
    PartialRejected,
    PartialStore,
    WorkerLocalPool,
    backoff_delays,
    is_rate_limited,
    judge_with_backoff,
    partial_path,
    run_incremental,
    strip_scheduling_keys,
)


def _meta(version: str = "v2.1-B8", model: str = "stub-judge-model") -> dict:
    """meta."""
    return {
        "policy": {"version": version, "prompt_hash": "ph_policy", "model": model},
        "voice": {"version": version, "prompt_hash": "ph_voice", "model": model},
        "dialogue": None,
    }


def _tasks(n: int, meta: dict | None = None) -> list[EvalTask]:
    return [EvalTask(scenario_id=f"S{i:02d}", expected_meta=meta or _meta(), payload=i) for i in range(n)]


def _make_judge_fn(calls: list[str], meta: dict | None = None):
    def judge_fn(task: EvalTask) -> JudgeOutcome:
        calls.append(task.scenario_id)
        return JudgeOutcome(
            verdict={
                "scenario_id": task.scenario_id,
                "policy_pass": True,
                "judge_meta": meta or _meta(),
            },
            timings={"policy": 1.0, "voice": 2.0},
        )

    return judge_fn


class TestPartialStore:
    def test_append_load_roundtrip(self, tmp_path):
        store = PartialStore(tmp_path / "verdicts_partial_audio.jsonl")
        store.append({"scenario_id": "S01", "policy_pass": True})
        store.append({"scenario_id": "S02", "policy_pass": False})
        records, corrupted = store.load()
        assert corrupted == 0
        assert set(records) == {"S01", "S02"}
        assert records["S02"]["policy_pass"] is False

    def test_load_tolerates_corrupt_line(self, tmp_path):
        path = tmp_path / "verdicts_partial_audio.jsonl"
        store = PartialStore(path)
        store.append({"scenario_id": "S01", "judge_meta": _meta()})

        with open(path, "a", encoding="utf-8") as f:
            f.write('{"scenario_id": "S02", "judge_me\n')
        records, corrupted = store.load()
        assert corrupted == 1
        assert set(records) == {"S01"}

    def test_partial_path_isolated_by_modality(self, tmp_path):
        assert partial_path(tmp_path, "audio").name == "verdicts_partial_audio.jsonl"
        assert partial_path(tmp_path, "text").name == "verdicts_partial_text.jsonl"


class TestResume:
    def test_resume_skips_matching_fingerprint(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        store.append({"scenario_id": "S00", "policy_pass": True, "judge_meta": _meta()})
        calls: list[str] = []
        verdicts = run_incremental(
            _tasks(3),
            _make_judge_fn(calls),
            store,
            log=lambda msg: None,
            sleep_fn=lambda s: None,
        )
        assert calls == ["S01", "S02"]
        assert [v["scenario_id"] for v in verdicts] == ["S00", "S01", "S02"]

    def test_fingerprint_mismatch_rejects_whole_partial(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")

        store.append({"scenario_id": "S00", "policy_pass": True, "judge_meta": _meta(model="qwen3-max")})
        with pytest.raises(PartialRejected, match="--fresh"):
            run_incremental(
                _tasks(3), _make_judge_fn([]), store, log=lambda msg: None, sleep_fn=lambda s: None
            )

    def test_fresh_ignores_mismatched_partial(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        store.append({"scenario_id": "S00", "policy_pass": True, "judge_meta": _meta(model="qwen3-max")})
        calls: list[str] = []
        verdicts = run_incremental(
            _tasks(2),
            _make_judge_fn(calls),
            store,
            fresh=True,
            log=lambda msg: None,
            sleep_fn=lambda s: None,
        )
        assert calls == ["S00", "S01"]
        assert len(verdicts) == 2

    def test_strips_scheduling_keys_on_reuse(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        store.append({"scenario_id": "S00", "judge_meta": _meta(), "_elapsed": {"policy": 1.0}})
        verdicts = run_incremental(
            _tasks(1),
            _make_judge_fn([]),
            store,
            log=lambda msg: None,
            sleep_fn=lambda s: None,
        )
        assert "_elapsed" not in verdicts[0]
        assert strip_scheduling_keys({"a": 1, "_elapsed": {}}) == {"a": 1}


class TestConcurrency:
    def test_concurrent_completeness_order_and_partial(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        calls: list[str] = []
        tasks = _tasks(8)
        verdicts = run_incremental(
            tasks,
            _make_judge_fn(calls),
            store,
            concurrency=4,
            log=lambda msg: None,
            sleep_fn=lambda s: None,
        )
        assert sorted(calls) == sorted(t.scenario_id for t in tasks)
        assert [v["scenario_id"] for v in verdicts] == [t.scenario_id for t in tasks]
        records, corrupted = store.load()
        assert corrupted == 0 and len(records) == 8

        assert "_elapsed" in records["S00"]

    def test_worker_local_pool_isolated_instances_and_aggregation(self):
        """Test worker local pool isolated instances and aggregation."""

        class FakeJudge:
            def __init__(self) -> None:
                self.parse_retries = 0

        pool = WorkerLocalPool(FakeJudge)
        seen: list[FakeJudge] = []
        lock = threading.Lock()

        def work(_: int) -> None:
            inst = pool.get()
            inst.parse_retries += 1
            with lock:
                seen.append(inst)

        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(work, range(16)))

        assert len(pool.instances) <= 4
        assert len(set(map(id, pool.instances))) == len(pool.instances)
        assert sum(i.parse_retries for i in pool.instances) == 16
        assert set(seen) <= set(pool.instances)


class TestRateLimitBackoff:
    def test_backoff_delays_sequence(self):
        assert backoff_delays() == [5.0, 10.0, 20.0, 40.0, 60.0]

    def test_is_rate_limited(self):
        assert is_rate_limited(RuntimeError("DashScope omni 评分失败 HTTP 429: xxx"))
        assert is_rate_limited(RuntimeError("Error code: 429 - rate limit exceeded"))

        class RateLimitError(Exception):
            pass

        assert is_rate_limited(RateLimitError("too many requests"))
        assert is_rate_limited(RuntimeError("Throttling: quota exceeded"))
        assert not is_rate_limited(ValueError("JSON parse error"))

    def test_retry_then_success(self):
        sleeps: list[float] = []
        n = {"calls": 0}

        def flaky(task: EvalTask) -> JudgeOutcome:
            n["calls"] += 1
            if n["calls"] <= 2:
                raise RuntimeError("HTTP 429: rate limited")
            return JudgeOutcome(verdict={"scenario_id": task.scenario_id})

        task = EvalTask(scenario_id="S01", expected_meta=_meta())
        out = judge_with_backoff(flaky, task, log=lambda msg: None, sleep_fn=sleeps.append)
        assert out.verdict["scenario_id"] == "S01"
        assert sleeps == [5.0, 10.0]

    def test_backoff_exhausted_raises(self):
        sleeps: list[float] = []

        def always_429(task: EvalTask) -> JudgeOutcome:
            raise RuntimeError("HTTP 429: rate limited")

        task = EvalTask(scenario_id="S01", expected_meta=_meta())
        with pytest.raises(RuntimeError, match="429"):
            judge_with_backoff(always_429, task, log=lambda msg: None, sleep_fn=sleeps.append)
        assert len(sleeps) == 5 and sleeps[-1] == 60.0

    def test_non_rate_limit_error_not_retried(self):
        sleeps: list[float] = []
        n = {"calls": 0}

        def boom(task: EvalTask) -> JudgeOutcome:
            n["calls"] += 1
            raise ValueError("judge 调用失败")

        task = EvalTask(scenario_id="S01", expected_meta=_meta())
        with pytest.raises(ValueError):
            judge_with_backoff(boom, task, log=lambda msg: None, sleep_fn=sleeps.append)
        assert n["calls"] == 1 and sleeps == []


class TestPartialLineFormat:
    def test_partial_line_json_serializable_single_line(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        store.append(
            {
                "scenario_id": "S01",
                "judge_meta": _meta(),
                "policy_detail": {"reason": "中文理由"},
                "_elapsed": {"policy": 1.5},
            }
        )
        lines = store.path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["policy_detail"]["reason"] == "中文理由"


def _trajectory_fixture(tmp_path):
    from listen2serve.cli import _build_eval_tasks

    scenario = {
        "scenario_id": "case",
        "role": "collection",
        "leakage_level": "explicit",
        "oracle_state": "displeased",
        "agent_policy": {"forbidden_actions": ["威胁"]},
    }
    audio = tmp_path / "agent.wav"
    audio.write_bytes(b"first audio")
    traj = {
        "scenario_id": "case",
        "status": "ok",
        "turns": [
            {
                "turn": 1,
                "user_text": "用户说话",
                "agent_text": "客服回答",
                "user_state": "displeased",
                "agent_audio_path": str(audio),
                "is_key_turn": True,
            },
        ],
    }
    run = tmp_path / "run"
    status = run / "case" / "sim_status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps(traj))

    def build(**kwargs):
        return _build_eval_tasks(
            run, {"case": scenario}, "google/gemini-3.8-flash", "openai/audio", **kwargs
        )[0]

    return scenario, traj, status, audio, build


@pytest.mark.parametrize("change", ["reply", "history", "scenario", "audio", "asr", "cap", "endpoint"])
def test_resume_fingerprint_binds_actual_inputs_and_options(tmp_path, change):
    scenario, traj, status, audio, build = _trajectory_fixture(tmp_path)
    original = build()
    kwargs = {}
    if change == "reply":
        traj["turns"][0]["agent_text"] = "新的回答"
        status.write_text(json.dumps(traj))
    elif change == "history":
        traj["turns"][0]["user_text"] = "不同的上下文"
        status.write_text(json.dumps(traj))
    elif change == "scenario":
        scenario["oracle_state"] = "cooperative"
    elif change == "audio":
        audio.write_bytes(b"different same-path audio")
    elif change == "asr":
        kwargs.update(use_asr=True, asr_model="asr-a")
    elif change == "cap":
        kwargs["flow_turn_cap"] = 3
    else:
        kwargs["backend_config"] = {"openai_base_url": "http://127.0.0.1:9999/v1"}
    changed = build(**kwargs)
    store = PartialStore(tmp_path / "partial.jsonl")
    store.append({"scenario_id": "case", "judge_meta": original.expected_meta})
    with pytest.raises(PartialRejected):
        run_incremental(
            [changed], lambda _: pytest.fail("stale verdict must be rejected before API calls"), store
        )


def test_asr_replaces_each_agent_utterance_without_equal_text_collisions(tmp_path):
    from types import SimpleNamespace

    from listen2serve.cli import _asr_histories, _EvalTaskPayload

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"audio")
    turns = [
        {"turn": n, "user_text": f"用户{n}", "agent_text": "重复的端点文本", "agent_audio_path": str(audio)}
        for n in (1, 2, 3)
    ]
    payload = _EvalTaskPayload({}, None, {"scenario_id": "case", "turns": turns}, turns[1], [], [])
    gateway = SimpleNamespace(
        transcribe=iter(
            [SimpleNamespace(text="转写1"), SimpleNamespace(text="转写2"), SimpleNamespace(text="转写3")]
        ).__next__
    )
    outputs = iter(["转写1", "转写2", "转写3"])
    gateway.transcribe = lambda _: SimpleNamespace(text=next(outputs))
    key, history, before = _asr_histories(payload, gateway)
    assert key == "转写2"
    assert [h["content"] for h in history if h["role"] == "assistant"] == ["转写1", "转写2", "转写3"]
    assert [h["content"] for h in before if h["role"] == "assistant"] == ["转写1"]


@pytest.mark.parametrize("failure", ["empty", "missing_audio", "api_failure"])
def test_explicit_asr_failure_never_falls_back_to_endpoint_text(tmp_path, failure):
    from types import SimpleNamespace

    from listen2serve.cli import _asr_histories, _EvalTaskPayload

    audio = tmp_path / "a.wav"
    if failure != "missing_audio":
        audio.write_bytes(b"audio")
    turn = {"turn": 1, "user_text": "用户", "agent_text": "端点旧文本", "agent_audio_path": str(audio)}
    payload = _EvalTaskPayload({}, None, {"scenario_id": "case", "turns": [turn]}, turn, [], [])

    def transcribe(_):
        if failure == "api_failure":
            raise RuntimeError("ASR API failed")
        return SimpleNamespace(text=" ")

    with pytest.raises((ValueError, RuntimeError)):
        _asr_histories(payload, SimpleNamespace(transcribe=transcribe))


def _evaluation_cli_fixture(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from listen2serve.gateway.asr_gateway import ASRGateway
    from listen2serve.gateway.llm_gateway import LLMGateway, LLMResult
    from listen2serve.runtime import config
    from listen2serve.runtime.config import Settings

    monkeypatch.chdir(tmp_path)
    scenario = {
        "scenario_id": "case",
        "role": "collection",
        "leakage_level": "explicit",
        "oracle_state": "displeased",
        "dynamics": "all_negative",
        "agent_policy": {"required_actions": ["确认身份"], "forbidden_actions": ["威胁"]},
    }
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(json.dumps(scenario))
    audio = tmp_path / "agent.wav"
    audio.write_bytes(b"fixture audio")
    run = tmp_path / "runs" / "case"
    status = run / "case" / "sim_status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "scenario_id": "case",
                "status": "ok",
                "turns": [
                    {
                        "turn": 1,
                        "user_text": "我真烦",
                        "agent_text": "端点文本",
                        "user_state": "displeased",
                        "agent_audio_path": str(audio),
                        "is_key_turn": True,
                    },
                ],
            }
        )
    )
    manifest = run / "run_manifest.json"
    manifest.write_text(json.dumps({"run_id": "case", "models": {"target": "target", "judge": None}}))
    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: Settings(
            _env_file=None,
            dashscope_api_key="offline-test-only",
            gemini_api_key="offline-test-only",
        ),
    )
    asr_calls, prompts = [], []

    def transcribe(_self, path):
        asr_calls.append(path)
        return SimpleNamespace(text="统一转写文本")

    monkeypatch.setattr(ASRGateway, "transcribe", transcribe)

    def chat(_self, messages, **kwargs):
        prompt = messages[-1]["content"]
        prompts.append(prompt)
        if isinstance(prompt, list):
            data = {"role_voice_match": 4, "state_voice_fit": 4, "naturalness": 4, "reason": "自然"}
        elif "## 一、过程清单" in prompt:
            data = {
                "flow_items": [{"action": "确认身份", "hit": "是", "turn": 1, "evidence": "统一转写文本"}],
                "task_score": 2,
                "reason": "完成",
            }
        elif "surface_score" in prompt:
            data = {"surface_score": 5, "state_guess": "displeased", "cue": "烦"}
        else:
            data = {
                "key_turn_score": 4,
                "key_turn_evidence": "统一转写文本",
                "forbidden_items": [{"action": "威胁", "triggered": "否", "evidence": "无"}],
                "state_fit": "是",
                "reason": "接住",
            }
        return LLMResult(text=json.dumps(data), model=kwargs["model"])

    monkeypatch.setattr(LLMGateway, "chat", chat)
    args = SimpleNamespace(
        run_id="case",
        scenarios=str(scenarios),
        modality="audio",
        asr=True,
        flow_turn_cap=0,
        report_id=None,
        judge_model=None,
        voice_judge_model=None,
        fresh=False,
        concurrency=1,
    )
    return args, manifest, asr_calls, prompts


def test_evaluate_writes_real_judge_metadata_after_success(tmp_path, monkeypatch):
    from listen2serve.cli import _evaluate

    args, manifest_path, asr_calls, prompts = _evaluation_cli_fixture(tmp_path, monkeypatch)
    assert _evaluate(args) == 0
    verdicts = json.loads((tmp_path / "reports/case/verdicts_audio.json").read_text())
    verdict = verdicts[0]
    assert verdict["wording"] == "explicit" and "leakage_label" not in verdict
    assert verdict["evaluated_text"] == "统一转写文本" and verdict["text_source"] == "asr"
    assert len(asr_calls) == 1
    assert any("统一转写文本" in p for p in prompts if isinstance(p, str) and "## 一、过程清单" in p)
    evaluation = json.loads(manifest_path.read_text())["evaluation"]
    assert evaluation["judge_model"] == evaluation["keyturn_model_requested"] == "google/gemini-3.8-flash"
    assert (
        evaluation["keyturn_model_observed"] is None
    )  # The fake gateway did not report a response identity.
    assert evaluation["model_identity_schema"] == "requested-response-v1"
    assert evaluation["voice_judge_model"] == "google/gemini-3.8-flash"
    assert evaluation["verdict_count"] == 1 and evaluation["asr"] is True
    assert len(evaluation["verdicts_sha256"]) == 64
    # A matching second call must reuse the validated partial and avoid all APIs.
    prompts.clear()
    assert _evaluate(args) == 0
    assert not prompts and len(asr_calls) == 1


def test_failed_evaluation_writes_no_final_scores_or_success_metadata(tmp_path, monkeypatch):
    from listen2serve.cli import _evaluate
    from listen2serve.evaluation.validation import JudgeResponseError
    from listen2serve.gateway.llm_gateway import LLMGateway, LLMResult

    args, manifest_path, _, _ = _evaluation_cli_fixture(tmp_path, monkeypatch)
    calls = []

    def incomplete(_self, messages, **kwargs):
        calls.append(kwargs["model"])
        return LLMResult(text='{"key_turn_score":4}')

    monkeypatch.setattr(LLMGateway, "chat", incomplete)
    with pytest.raises(JudgeResponseError):
        _evaluate(args)
    assert calls == ["google/gemini-3.8-flash"] * 2
    assert not (tmp_path / "reports/case/verdicts_audio.json").exists()
    assert "evaluation" not in json.loads(manifest_path.read_text())


def test_explicit_asr_failure_stops_before_any_judge_call(tmp_path, monkeypatch):
    from listen2serve.cli import _evaluate
    from listen2serve.gateway.asr_gateway import ASRGateway

    args, manifest_path, _, prompts = _evaluation_cli_fixture(tmp_path, monkeypatch)

    def failed(_self, _path):
        raise RuntimeError("ASR fixture failure")

    monkeypatch.setattr(ASRGateway, "transcribe", failed)
    with pytest.raises(RuntimeError, match="ASR fixture failure"):
        _evaluate(args)
    assert not prompts
    assert "evaluation" not in json.loads(manifest_path.read_text())


@pytest.mark.parametrize(
    "record,expected",
    [
        ({}, True),
        ({"status": "ok", "termination_reason": "user_closed"}, True),
        ({"status": "partial"}, False),
        ({"trajectory_status": "partial"}, False),
        ({"status": "ok", "termination_reason": "endpoint_session_closed"}, False),
        ({"dialogue_complete": False}, False),
    ],
)
def test_dialogue_completion_handles_current_and_legacy_records(record, expected):
    from listen2serve.evaluation.incremental import dialogue_is_complete

    assert dialogue_is_complete(record) is expected
