"""Generation caches must retain the exact input and model identity."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from listen2serve.runtime.config import PROJECT_ROOT, Settings
from listen2serve.runtime.manifest import load_run_manifest
from listen2serve.runtime.orchestrator import Trajectory
from listen2serve.runtime.run_batch import BatchRunner

MODEL = "dashscope/qwen-audio-3.0-realtime-plus"


@pytest.fixture
def scenario() -> dict:
    path = PROJECT_ROOT / "data/benchmark/scenarios.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    async def fake_run(self, scenario, model_spec, api_key, domain, llm, tts):
        calls.append(scenario["scenario_id"])
        traj = Trajectory(scenario_id=scenario["scenario_id"], model=model_spec, role=scenario["role"])
        traj.rt_diag = {"frames": 7}
        return traj

    monkeypatch.setattr(BatchRunner, "_run_one", fake_run)
    settings = Settings(_env_file=None, llm_model="qwen-plus")
    return settings, calls


def run(runner, scenarios, model=MODEL, **kwargs):
    return asyncio.run(runner.run_batch(scenarios, model, "test-only-key", **kwargs))


@pytest.mark.parametrize("resume", [False, True])
def test_response_timeout_is_terminal_in_run_and_preserved_on_resume(batch, scenario, monkeypatch, resume):
    settings, calls = batch

    async def timed_out(self, scenario, model_spec, *args):
        calls.append("timeout")
        return Trajectory(
            scenario_id=scenario["scenario_id"],
            model=model_spec,
            role=scenario["role"],
            status="error",
            termination_reason="response_timeout",
            error="fixture response timeout",
        )

    monkeypatch.setattr(BatchRunner, "_run_one", timed_out)
    runner = BatchRunner(settings=settings, run_id="timeout", max_retries=3)
    first = run(runner, [scenario])[0]
    assert first.status == "error" and calls == ["timeout"]
    path = runner.runs_root / scenario["scenario_id"] / "sim_status.json"
    original = path.read_bytes()
    loaded = run(BatchRunner(settings=settings, run_id="timeout", max_retries=3), [scenario], resume=resume)[
        0
    ]
    assert loaded.resumed and loaded.termination_reason == "response_timeout"
    assert calls == ["timeout"] and path.read_bytes() == original
    fresh = run(BatchRunner(settings=settings, run_id="new-timeout", max_retries=3), [scenario])[0]
    assert not fresh.resumed and calls == ["timeout", "timeout"]


def test_other_errors_still_use_configured_retries(batch, scenario, monkeypatch):
    settings, calls = batch

    async def fail_then_succeed(self, scenario, model_spec, *args):
        calls.append("attempt")
        return Trajectory(
            scenario_id=scenario["scenario_id"],
            model=model_spec,
            role=scenario["role"],
            status="error" if len(calls) == 1 else "ok",
            termination_reason="error" if len(calls) == 1 else "user_closed",
        )

    monkeypatch.setattr(BatchRunner, "_run_one", fail_then_succeed)
    monkeypatch.setattr(BatchRunner, "_retry_backoff_s", lambda *_args: 0)
    result = run(BatchRunner(settings=settings, run_id="ordinary-error", max_retries=1), [scenario])[0]
    assert result.status == "ok" and calls == ["attempt", "attempt"]


def test_resume_keeps_artifacts_and_evaluation(batch, scenario):
    settings, calls = batch
    runner = BatchRunner(settings=settings, run_id="test")
    run(runner, [scenario])
    path = runner.runs_root / scenario["scenario_id"] / "sim_status.json"
    original = path.read_bytes()
    manifest_path = runner.runs_root / "run_manifest.json"
    manifest = load_run_manifest(runner.runs_root)
    manifest["evaluation"] = {"judge_model": "openai/test-judge", "fingerprint": "saved"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    resumed = run(BatchRunner(settings=settings, run_id="test"), [scenario])[0]
    assert resumed.resumed and resumed.rt_diag == {"frames": 7}
    assert calls == [scenario["scenario_id"]]
    assert path.read_bytes() == original
    current = load_run_manifest(runner.runs_root)
    assert current["evaluation"] == manifest["evaluation"]
    assert current["models"]["target"] == MODEL
    assert current["prompts"]["agent_prompt_sha256"]
    assert current["prompts"]["sim_prompt_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        "target",
        "sim",
        "tts",
        "scenario",
        "prosody",
        "thinking",
        "voice",
        "timing",
        "tokens",
        "prompt",
        "shared-prompt",
        "sim-prompt",
        "audio",
    ],
)
def test_resume_rejects_changed_input_before_any_write(batch, scenario, monkeypatch, change):
    settings, calls = batch
    first = BatchRunner(settings=settings, run_id="test")
    run(first, [scenario])
    manifest_path = first.runs_root / "run_manifest.json"
    original = manifest_path.read_bytes()
    second_settings = settings
    changed = copy.deepcopy(scenario)
    options = {}
    model = MODEL
    if change == "target":
        model = "dashscope/qwen3.5-omni-plus-realtime"
    elif change == "sim":
        options["sim_model"] = "qwen-max"
    elif change == "tts":
        second_settings = settings.model_copy(update={"tts_model": "other-tts"})
    elif change == "scenario":
        changed["user_script"]["identity"] += "（内容已修改）"
    elif change == "prosody":
        options["prosody_arm"] = "neutral"
    elif change == "thinking":
        options["sim_thinking"] = True
    elif change == "voice":
        second_settings = settings.model_copy(update={"tts_voice_user": "another-voice"})
    elif change == "timing":
        second_settings = settings.model_copy(update={"tick_ms": settings.tick_ms + 20})
    elif change == "tokens":
        second_settings = settings.model_copy(update={"max_session_tokens": settings.max_session_tokens + 1})
    elif change == "prompt":
        original_prompt = BatchRunner._agent_prompt
        monkeypatch.setattr(
            BatchRunner, "_agent_prompt", staticmethod(lambda s, d: original_prompt(s, d) + " changed")
        )
    elif change == "shared-prompt":
        from listen2serve.runtime import run_batch as module

        original_prompt = module.voice_call_instruction
        monkeypatch.setattr(module, "voice_call_instruction", lambda: original_prompt() + " changed")
    elif change == "sim-prompt":
        from listen2serve.runtime.user_simulator import UserSimulator

        original_messages = UserSimulator._build_messages

        def changed_messages(self, *args, **kwargs):
            messages = original_messages(self, *args, **kwargs)
            messages[0]["content"] += " changed"
            return messages

        monkeypatch.setattr(UserSimulator, "_build_messages", changed_messages)
    elif change == "audio":
        options["save_audio"] = False
    second = BatchRunner(settings=second_settings, run_id="test", **options)
    with pytest.raises(ValueError, match="生成条件"):
        run(second, [changed], model)
    assert calls == [scenario["scenario_id"]]
    assert manifest_path.read_bytes() == original


def test_legacy_cache_cannot_be_silently_reused(batch, scenario):
    settings, calls = batch
    runner = BatchRunner(settings=settings, run_id="old")
    path = runner.runs_root / scenario["scenario_id"] / "sim_status.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"status": "ok", "scenario_id": scenario["scenario_id"], "model": "old-target"})
    )
    with pytest.raises(ValueError, match="生成条件"):
        run(runner, [scenario])
    assert calls == []
    assert not (runner.runs_root / "run_manifest.json").exists()


def test_failed_scenario_retries_without_reusing_error(batch, scenario, monkeypatch):
    settings, calls = batch
    runner = BatchRunner(settings=settings, run_id="retry", max_retries=0)
    original_run = BatchRunner._run_one

    async def fail(self, scenario, model_spec, *args):
        calls.append("failed")
        return Trajectory(
            scenario["scenario_id"], model_spec, scenario["role"], status="error", error="stub failure"
        )

    monkeypatch.setattr(BatchRunner, "_run_one", fail)
    assert run(runner, [scenario])[0].status == "error"
    monkeypatch.setattr(BatchRunner, "_run_one", original_run)
    assert run(runner, [scenario])[0].status == "ok"
    assert calls == ["failed", scenario["scenario_id"]]


def test_model_change_when_adding_new_scenario_is_rejected(batch, scenario):
    settings, calls = batch
    runner = BatchRunner(settings=settings, run_id="shared")
    run(runner, [scenario])
    new_scenario = copy.deepcopy(scenario)
    new_scenario["scenario_id"] += "-new"
    with pytest.raises(ValueError, match="已有 run"):
        run(runner, [new_scenario], "dashscope/qwen3.5-omni-plus-realtime")
    assert len(calls) == 1


def test_corrupt_manifest_is_rejected_before_overwrite(batch, scenario):
    settings, calls = batch
    runner = BatchRunner(settings=settings, run_id="damaged")
    run(runner, [scenario])
    path = runner.runs_root / "run_manifest.json"
    path.write_text("{broken")
    with pytest.raises(ValueError, match="manifest 无法读取"):
        run(runner, [scenario])
    assert path.read_text() == "{broken"
    assert len(calls) == 1


def test_changed_trajectory_model_is_rejected_even_if_identity_survives(batch, scenario):
    settings, calls = batch
    runner = BatchRunner(settings=settings, run_id="changed")
    run(runner, [scenario])
    path = runner.runs_root / scenario["scenario_id"] / "sim_status.json"
    data = json.loads(path.read_text())
    data["model"] = "dashscope/other-model"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="生成条件"):
        run(runner, [scenario])
    assert len(calls) == 1


@pytest.mark.parametrize("resume", [True, False])
def test_partial_call_is_saved_once_and_preserved_on_resume(batch, scenario, monkeypatch, resume):
    settings, calls = batch

    async def partial(self, scenario, model_spec, *args):
        calls.append("partial")
        artifact = self.runs_root / scenario["scenario_id"] / "evidence.wav"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"first-attempt-evidence")
        return Trajectory(
            scenario["scenario_id"],
            model_spec,
            scenario["role"],
            status="partial",
            termination_reason="endpoint_session_closed",
        )

    monkeypatch.setattr(BatchRunner, "_run_one", partial)
    runner = BatchRunner(settings=settings, run_id="partial", max_retries=3)
    result = run(runner, [scenario])[0]
    assert result.status == "partial" and calls == ["partial"]
    path = runner.runs_root / scenario["scenario_id"] / "sim_status.json"
    before = path.read_bytes()
    second = run(runner, [scenario], resume=resume)[0]
    assert second.status == "partial" and second.resumed
    assert calls == ["partial"] and path.read_bytes() == before
    assert (path.parent / "evidence.wav").read_bytes() == b"first-attempt-evidence"


def test_legacy_successful_session_closure_is_resumed_as_partial(batch, scenario):
    settings, calls = batch
    runner = BatchRunner(settings=settings, run_id="legacy-closed")
    run(runner, [scenario])
    path = runner.runs_root / scenario["scenario_id"] / "sim_status.json"
    data = json.loads(path.read_text())
    data["termination_reason"] = "endpoint_session_closed"
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    resumed = run(runner, [scenario])[0]
    assert resumed.status == "partial" and resumed.resumed
    assert calls == [scenario["scenario_id"]]
    assert path.read_bytes() == before
