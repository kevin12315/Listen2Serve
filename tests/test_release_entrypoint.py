"""Exercise the public shell pipeline with network-isolated backend doubles."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from listen2serve.runtime.config import PROJECT_ROOT

# Backend doubles live only in the temporary test project. The actual simulator,
# orchestrator, strict judges, cache, CLI and report writer run unchanged.
_OFFLINE_BACKENDS = r"""
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace


def no_network(*args, **kwargs):
    raise AssertionError("Offline regression attempted network access")

socket.socket.connect = no_network
socket.create_connection = no_network
socket.getaddrinfo = no_network

from listen2serve.gateway.asr_gateway import ASRGateway, ASRResult
from listen2serve.gateway.llm_gateway import LLMGateway, LLMResult
from listen2serve.gateway.tts_gateway import TTSGateway, TTSResult
from listen2serve.runtime.agent import AgentTickResult
from listen2serve.runtime.errors import EndpointSessionClosed
import listen2serve.runtime.run_batch as batch_module

SCENARIO = json.loads(Path("offline_scenario.json").read_text(encoding="utf-8"))
AGENT_TEXT = "好的，我们先核对您的信息，再一起确认下一步安排。"
CALLS = Path("offline_calls.jsonl")


def record(kind, **values):
    with CALLS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"backend": "offline-mock", "kind": kind, **values}, ensure_ascii=False) + "\n")


class OfflineAgent:
    def __init__(self, model_spec, **kwargs):
        self.model_spec = model_spec
        self.provider = SimpleNamespace(needs_continuous_audio=False)
        self.session_usage = {}
        self.pending = False
        self.user_turns = 0

    async def start(self):
        record("realtime", requested_model=self.model_spec)
        if os.environ.get("L2S_STUB_FAIL") == "generation":
            raise RuntimeError("intentional offline generation failure")

    async def stop(self):
        pass

    async def cancel_response(self):
        pass

    async def run_tick(self, pcm16, tick):
        if any(pcm16):
            self.user_turns += 1
            if os.environ.get("L2S_STUB_PARTIAL") and self.user_turns == 3:
                raise EndpointSessionClosed("offline fixture session limit after completed key turn")
            self.pending = True
        elif self.pending:
            self.pending = False
            return AgentTickResult(audio=b"\x01\x00" * 4800, transcript=AGENT_TEXT,
                                   response_done=True, done_status="completed", agent_output=True)
        return AgentTickResult()

    def attach_db(self, db):
        pass

batch_module.Agent = OfflineAgent
_turn = 0


def offline_chat(self, messages, model=None, **kwargs):
    global _turn
    record("llm", requested_model=model)
    if os.environ.get("L2S_STUB_FAIL") == "evaluation" and "裁判" in str(messages):
        return LLMResult(text='{"key_turn_score":99}', model=model or "offline-mock")
    first = messages[0]["content"]
    if isinstance(first, list):
        payload = {"role_voice_match": 4, "state_voice_fit": 4, "naturalness": 4,
                   "reason": "OFFLINE MOCK: synthetic voice score"}
    elif "扮演打/接电话的用户" in first:
        _turn += 1
        if _turn == 2:
            state = SCENARIO["user_script"]["key_event"]["state"]
        else:
            state = "neutral"
        payload = {"text": ("这项安排具体要怎么处理，请您再跟我说明一下。" if _turn < 3 else
                            "好的，您刚才说的这些我知道了，那我们就先这样，再见。"),
                   "state": state, "is_key_turn": _turn == 2,
                   "end_call": _turn >= 3, "covered": [1], "interrupt": False}
    elif "客服关键轮应对裁判" in first:
        payload = {"key_turn_score": 4, "key_turn_evidence": AGENT_TEXT,
                   "forbidden_items": [{"action": action, "triggered": "否", "evidence": "无"}
                       for action in SCENARIO["agent_policy"].get("forbidden_actions", [])],
                   "state_fit": "是", "reason": "OFFLINE MOCK: synthetic key-turn score"}
    elif "客服全对话裁判" in first:
        if os.environ.get("L2S_STUB_PARTIAL"):
            raise AssertionError("Partial call must not invoke the whole-dialogue judge")
        checklist = SCENARIO.get("required_actions_effective") or SCENARIO["agent_policy"].get("required_actions", [])
        payload = {"flow_items": [{"action": action, "hit": "否", "turn": None, "evidence": "无"}
                                  for action in checklist],
                   "task_score": 0, "reason": "OFFLINE MOCK: synthetic dialogue score"}
    else:
        payload = {"surface_score": 1, "state_guess": "neutral", "cue": "OFFLINE MOCK"}
    return LLMResult(text=json.dumps(payload, ensure_ascii=False), model=model or "offline-mock",
                     prompt_tokens=1, completion_tokens=1, finish_reason="stop")


def offline_tts(self, text, **kwargs):
    record("tts", requested_model=self.settings.tts_model)
    return TTSResult(audio=b"\x01\x00" * 3200, sample_rate=16000, backend="offline-mock")


def offline_asr(self, audio_path, **kwargs):
    if not Path(audio_path).is_file():
        raise AssertionError("ASR input audio missing")
    record("asr", audio=str(audio_path))
    return ASRResult(text=AGENT_TEXT, backend="offline-mock")

LLMGateway.chat = offline_chat
TTSGateway.synthesize = offline_tts
ASRGateway.transcribe = offline_asr
"""


def prepare_offline_project(root: Path) -> dict:
    """Prepare an isolated project; callers may retain it as a labelled format example."""
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "scripts/run_experiment.sh", root / "scripts/run_experiment.sh")
    (root / "src").symlink_to(PROJECT_ROOT / "src", target_is_directory=True)
    (root / "data").symlink_to(PROJECT_ROOT / "data", target_is_directory=True)
    (root / "scripts/sitecustomize.py").write_text(_OFFLINE_BACKENDS, encoding="utf-8")
    scenarios = [
        json.loads(line)
        for line in (PROJECT_ROOT / "data/benchmark/scenarios.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    neutral = next(
        s for s in scenarios if s.get("leakage_level") == "prosody_only" and s.get("role") == "hotline"
    )
    (root / "offline_scenario.json").write_text(json.dumps(neutral, ensure_ascii=False), encoding="utf-8")
    (root / "subset.json").write_text(
        json.dumps({"scenario_ids": [neutral["scenario_id"]]}), encoding="utf-8"
    )
    config = {
        "wording": "explicit",
        "prosody_arm": "state",
        "asr_for_judge_input": True,
        "scenario_ids_file": "subset.json",
        "endpoints": ["dashscope/offline-a", "dashscope/offline-b"],
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "OFFLINE-MOCK.txt").write_text(
        "All backend responses and scores are synthetic test doubles. No commercial model was called.\n"
    )
    return neutral


def offline_env() -> dict[str, str]:
    return {
        **os.environ,
        "PY": sys.executable,
        "DASHSCOPE_API_KEY": "offline-test-only",
        "OPENAI_API_KEY": "offline-test-only",
        "GEMINI_API_KEY": "offline-test-only",
        "SIM_MODEL": "openai/offline-simulator",
        "JUDGE_MODEL": "openai/offline-judge",
        "VOICE_JUDGE_MODEL": "openai/offline-voice",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def invoke_pipeline(root: Path, *extra: str, env: dict | None = None):
    # /bin/bash is macOS's system Bash 3.2; Linux uses its installed Bash.
    return subprocess.run(
        [
            "/bin/bash",
            "scripts/run_experiment.sh",
            "--run-id",
            "offline",
            "--config",
            "config.json",
            "--model",
            "all",
            "--wording",
            "neutral",
            "--prosody-arm",
            "neutral",
            "--max-retries",
            "0",
            *extra,
        ],
        check=False,
        cwd=root,
        env=env or offline_env(),
        text=True,
        capture_output=True,
        timeout=90,
    )


def test_public_entrypoint_generation_scoring_reporting(tmp_path):
    scenario = prepare_offline_project(tmp_path)
    result = invoke_pipeline(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / "offline_calls.jsonl").read_text().splitlines()]
    assert {"realtime", "tts", "llm", "asr"} <= {call["kind"] for call in calls}
    for endpoint in ("dashscope_offline-a", "dashscope_offline-b"):
        rid = f"offline__{endpoint}"
        paths = list((tmp_path / "runs" / rid).glob("*/sim_status.json"))
        assert len(paths) == 1  # Exact subset file, rather than the whole wording condition.
        trajectory = json.loads(paths[0].read_text())
        assert trajectory["scenario_id"] == scenario["scenario_id"]
        assert trajectory["status"] == "ok" and len(trajectory["turns"]) == 3
        assert trajectory["prosody_arm"] == "neutral"  # CLI override of config.state.
        manifest = json.loads((tmp_path / "runs" / rid / "run_manifest.json").read_text())
        assert manifest["models"]["sim"] == "openai/offline-simulator"
        assert manifest["evaluation"]["judge_model"] == "openai/offline-judge"
        report = (tmp_path / "reports" / rid / "report_audio.md").read_text()
        assert all(metric in report for metric in ("KeyTurnPass", "VoicePass", "FlowRate", "TaskScore"))
    assert len([call for call in calls if call["kind"] == "asr"]) == 6


def test_explicit_asr_override_and_single_model(tmp_path):
    prepare_offline_project(tmp_path)
    result = invoke_pipeline(tmp_path, "--no-asr", "--model", "dashscope/offline-single")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / "offline_calls.jsonl").read_text().splitlines()]
    assert not any(call["kind"] == "asr" for call in calls)
    assert (tmp_path / "reports/offline/report_audio.md").exists()


@pytest.mark.parametrize("stage", ["generation", "evaluation"])
def test_entrypoint_stops_on_backend_failure(tmp_path, stage):
    prepare_offline_project(tmp_path)
    result = invoke_pipeline(
        tmp_path, "--model", "dashscope/offline-single", env={**offline_env(), "L2S_STUB_FAIL": stage}
    )
    assert result.returncode != 0
    assert not (tmp_path / "reports/offline/report_audio.md").exists()
    if stage == "generation":
        assert not (tmp_path / "reports/offline/verdicts_audio.json").exists()


def test_missing_subset_fails_before_generation(tmp_path):
    prepare_offline_project(tmp_path)
    (tmp_path / "subset.json").unlink()
    result = invoke_pipeline(tmp_path)
    assert result.returncode != 0
    assert not (tmp_path / "offline_calls.jsonl").exists()


def test_pipeline_missing_judge_credentials_stops_before_generation(tmp_path):
    prepare_offline_project(tmp_path)
    env = {**offline_env(), "JUDGE_MODEL": "google/gemini-3.8-flash",
           "VOICE_JUDGE_MODEL": "google/gemini-3.8-flash", "GEMINI_API_KEY": ""}
    result = invoke_pipeline(tmp_path, "--model", "dashscope/offline-single", env=env)
    assert result.returncode != 0
    assert "GEMINI_API_KEY" in result.stderr
    assert "fake-" not in result.stderr and "offline-test-only" not in result.stderr
    assert not (tmp_path / "offline_calls.jsonl").exists()
    assert not (tmp_path / "runs").exists()


def test_all_endpoints_are_checked_before_first_generation(tmp_path):
    prepare_offline_project(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["endpoints"][-1] = "doubao-seeduplex-3.0"
    config_path.write_text(json.dumps(config))
    result = invoke_pipeline(tmp_path, env={**offline_env(), "VOLC_API_KEY": "", "VOLC_API_KEY2": ""})
    assert result.returncode != 0
    assert "VOLC_API_KEY" in result.stderr and "VOLC_API_KEY2" in result.stderr
    assert not (tmp_path / "offline_calls.jsonl").exists()
    assert not (tmp_path / "runs").exists()


def test_pipeline_preflight_respects_explicit_simulator_override(tmp_path):
    prepare_offline_project(tmp_path)
    result = invoke_pipeline(
        tmp_path, "--model", "dashscope/offline-single", "--sim-model", "openai/offline-simulator",
        env={**offline_env(), "SIM_MODEL": "kimi-k3", "MOONSHOT_API_KEY": ""},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((tmp_path / "runs/offline/run_manifest.json").read_text())
    assert manifest["models"]["sim"] == "openai/offline-simulator"


def test_public_entrypoint_preserves_partial_key_evidence_and_reports_denominators(tmp_path):
    prepare_offline_project(tmp_path)
    result = invoke_pipeline(
        tmp_path, "--model", "dashscope/offline-single",
        env={**offline_env(), "L2S_STUB_PARTIAL": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "完整 0，部分 1，失败 0" in result.stdout
    assert "0 complete, 1 partial scored scenarios" in result.stdout
    trajectory = json.loads(next((tmp_path / "runs/offline").glob("*/sim_status.json")).read_text())
    assert trajectory["status"] == "partial"
    assert trajectory["termination_reason"] == "endpoint_session_closed"
    assert trajectory["turns"][-1]["endpoint_interrupted"] is True
    verdict = json.loads((tmp_path / "reports/offline/verdicts_audio.json").read_text())[0]
    assert verdict["dialogue_complete"] is False and verdict["trajectory_status"] == "partial"
    assert verdict["key_turn_pass"] is True and verdict["voice_pass"] is True
    assert verdict["flow_rate"] is None and verdict["task_score"] is None
    summary = json.loads((tmp_path / "reports/offline/aggregated_audio.json").read_text())["summary"]
    assert summary["n_complete"] == 0 and summary["n_partial"] == 1
    assert summary["key_turn_n"] == 1 and summary["voice_n"] == 1
    assert summary["flow_n"] == 0 and summary["task_n"] == 0
    report = (tmp_path / "reports/offline/report_audio.md").read_text()
    assert "完整通话：0" in report
    assert "FlowRate=0" in report and "TaskScore=0" in report
