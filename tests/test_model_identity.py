"""Provider-reported model identity must never be inferred from a request."""

from __future__ import annotations

import base64
import io
import json
import wave
from types import SimpleNamespace

import httpx
import pytest

from listen2serve.cli import _provenance_summary, _single_reported_model
from listen2serve.gateway.asr_gateway import ASRGateway
from listen2serve.gateway.base import CallRecord, model_identity, summarize_identities
from listen2serve.gateway.llm_gateway import LLMGateway, LLMResult
from listen2serve.gateway.tts_gateway import TTSGateway
from listen2serve.runtime.agent import Agent
from listen2serve.runtime.config import Settings
from listen2serve.runtime.manifest import observed_judge
from listen2serve.runtime.realtime.qwen_provider import QwenRealtimeProvider
from listen2serve.runtime.realtime.registry import create_provider


def wav_bytes():
    data = io.BytesIO()
    with wave.open(data, "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x00" * 160)
    return data.getvalue()


def test_identity_sanitizes_metadata_and_never_infers_an_observation():
    result = model_identity(
        {}, "requested", requested_url="https://user:secret@example.test/api?token=private"
    )
    assert result["requested_host"] == "example.test"
    assert result["provider_host"] is None and result["response_model"] is None
    assert result["response_model_differs"] is None
    assert "secret" not in json.dumps(result) and "private" not in json.dumps(result)
    invalid = model_identity(
        {"model": "https://user:secret@private", "request_id": "Bearer secret"}, "requested"
    )
    assert invalid["response_model"] is None and invalid["request_id"] is None
    assert CallRecord("llm", "openai", "requested").as_dict()["response_model"] is None
    assert LLMResult(text="cached legacy", model="openai/configured").response_model is None


@pytest.mark.parametrize("reported", [None, "qwen-plus-2026-11-01"])
def test_llm_keeps_the_wire_request_and_returned_snapshot_separate(monkeypatch, caplog, reported):
    calls = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=None,
        _request_id="reported-request-123",
    )
    if reported:
        response.model = reported

    def create(**kwargs):
        calls.append(kwargs)
        return response

    gw = LLMGateway(Settings(_env_file=None))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(gw, "_client", lambda *args: client)
    result = gw.chat([{"role": "user", "content": "test"}], model="qwen-plus")
    assert calls[0]["model"] == "qwen-plus"
    assert result.model == "dashscope/qwen-plus"  # Compatibility alias remains the requested model.
    assert result.requested_model == "qwen-plus" and result.response_model == reported
    assert result.request_id == "reported-request-123"
    assert result.record.as_dict()["response_model"] == reported
    if reported:
        assert "verify the provider's alias/snapshot mapping" in caplog.text
    else:
        assert result.response_model_differs is None


def test_asr_reports_real_backend_and_response_identity(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(wav_bytes())
    response = httpx.Response(
        200,
        json={"model": "asr-snapshot", "request_id": "reported-123", "output": {"text": "你好"}},
        request=httpx.Request("POST", "https://actual.example.test/path?secret=omit"),
    )
    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: response)
    result = ASRGateway(Settings(_env_file=None, asr_model="configured-asr")).transcribe(audio)
    assert result.backend == "dashscope"
    assert result.requested_model == "configured-asr" and result.response_model == "asr-snapshot"
    assert result.provider_host == "actual.example.test" and result.request_id == "reported-123"
    assert result.as_dict()["requested_model"] == "configured-asr"


@pytest.mark.parametrize("binary", [True, False])
def test_tts_binary_response_has_no_inferred_model(monkeypatch, binary):
    response = (
        httpx.Response(200, content=wav_bytes())
        if binary
        else httpx.Response(
            200,
            json={
                "model": "tts-snapshot",
                "output": {"audio": {"data": base64.b64encode(wav_bytes()).decode()}},
            },
        )
    )
    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: response)
    result = TTSGateway(Settings(_env_file=None)).synthesize("你好")
    assert result.requested_model == "qwen-audio-3.0-tts-plus"
    assert result.response_model == (None if binary else "tts-snapshot")
    assert result.record.response_model == result.response_model


def test_realtime_session_identity_preserves_qwen_and_doubao_wire_names():
    qwen = QwenRealtimeProvider("fixture", model="qwen-requested")
    assert qwen.identity["response_model"] is None
    qwen._capture_identity(
        {"type": "session.created", "session": {"model": "qwen-reported-snapshot", "id": "private-session"}}
    )
    qwen._capture_identity({"type": "session.updated", "session": {"id": "private-session"}})
    assert qwen.identity["response_model"] == "qwen-reported-snapshot"
    assert qwen.model == "qwen-requested"
    assert "private-session" not in json.dumps(qwen.identity)
    provider = create_provider("doubao-seeduplex-3.0", "fixture")
    agent = Agent.__new__(Agent)
    agent.provider, agent.model_spec = provider, "doubao-seeduplex-3.0"
    assert agent.provider_identity["requested_model"] == "doubao-seeduplex-3.0"
    assert agent.provider_identity["requested_wire_model"] == "1.2.6.1"
    assert agent.provider_identity["response_model"] is None
    provider._capture_identity({"type": "session.created", "session": {"model": "1.2.6.2"}})
    assert agent.provider_identity["response_model"] == "1.2.6.2"


def test_manifest_and_report_do_not_treat_old_configured_labels_as_observed():
    old = {"evaluation": {"judge_model": "google/requested", "keyturn_model_observed": "google/requested"}}
    assert observed_judge(old)["observed"] is None
    assert _provenance_summary("run", old)["models"]["judge_response_model"] is None
    configured_only = [{"judge_meta": {"keyturn": {"model": "google/configured"}}}]
    assert _single_reported_model(configured_only, "keyturn") is None
    true_response = [
        {
            "model_identity_schema": "requested-response-v1",
            "model_identity": {
                "keyturn": [{"requested_model": "requested", "response_model": "reported-snapshot"}]
            },
        }
    ]
    assert _single_reported_model(true_response, "keyturn") == "reported-snapshot"
    summary = summarize_identities([{"service": "llm", "model": "old-request"}])
    assert summary[0]["requested_model"] == "old-request" and summary[0]["response_model"] is None


def test_identity_resummarization_preserves_five_asr_calls():
    from listen2serve.cli import _report_identity_rows

    rows = [
        {"service": "asr", "backend": "dashscope", "requested_model": "asr-request", "response_model": None}
        for _ in range(5)
    ]
    once = summarize_identities(rows)
    assert once[0]["calls"] == 5
    assert summarize_identities(once)[0]["calls"] == 5
    verdicts = [{"model_identity_schema": "requested-response-v1", "model_identity": {"asr": once}}]
    assert _report_identity_rows(verdicts, "asr")[0]["calls"] == 5


def test_evaluation_persists_reported_identity_without_relabelling_requests(tmp_path, monkeypatch):
    from test_evaluate_incremental import _evaluation_cli_fixture

    from listen2serve.cli import _evaluate
    from listen2serve.model_registry import resolve_model

    args, manifest_path, _, _ = _evaluation_cli_fixture(tmp_path, monkeypatch)
    original = LLMGateway.chat

    def audited(self, messages, **kwargs):
        result = original(self, messages, **kwargs)
        backend, name = resolve_model(kwargs["model"], "")
        self.sink.append(CallRecord("llm", backend, name, response_model="reported-snapshot"))
        return result

    monkeypatch.setattr(LLMGateway, "chat", audited)
    assert _evaluate(args) == 0
    evaluation = json.loads(manifest_path.read_text())["evaluation"]
    assert evaluation["keyturn_model_requested"] == "google/gemini-3.8-flash"
    assert evaluation["keyturn_model_observed"] == "reported-snapshot"
    assert evaluation["voice_model_observed"] == "reported-snapshot"
    verdicts = json.loads((tmp_path / "reports/case/verdicts_audio.json").read_text())
    row = verdicts[0]["model_identity"]["keyturn"][0]
    assert row["requested_model"] == "gemini-3.8-flash" and row["response_model"] == "reported-snapshot"


@pytest.mark.asyncio
async def test_target_and_gateway_identity_reach_saved_trajectory_and_manifest(tmp_path, monkeypatch):
    from listen2serve.runtime.config import PROJECT_ROOT
    from listen2serve.runtime.manifest import load_run_manifest
    from listen2serve.runtime.orchestrator import Trajectory
    from listen2serve.runtime.run_batch import BatchRunner

    scenario = json.loads((PROJECT_ROOT / "data/benchmark/scenarios.jsonl").read_text().splitlines()[0])
    monkeypatch.chdir(tmp_path)

    async def fake_run(self, scenario, model_spec, api_key, domain, llm, tts):
        llm.sink.append(CallRecord("llm", "moonshot", "kimi-k3", response_model="kimi-reported-snapshot"))
        tts.sink.append(CallRecord("tts", "dashscope", "qwen-audio-3.0-tts-plus"))
        return Trajectory(
            scenario["scenario_id"],
            model_spec,
            scenario["role"],
            provider_identity={
                "requested_model": model_spec,
                "requested_wire_model": "qwen-audio-3.0-realtime-plus",
                "response_model": "qwen-reported-snapshot",
            },
        )

    monkeypatch.setattr(BatchRunner, "_run_one", fake_run)
    runner = BatchRunner(Settings(_env_file=None), run_id="identity")
    await runner.run_batch([scenario], "dashscope/qwen-audio-3.0-realtime-plus", "fixture")
    path = runner.runs_root / scenario["scenario_id"] / "sim_status.json"
    trajectory = json.loads(path.read_text())
    assert trajectory["provider_identity"]["response_model"] == "qwen-reported-snapshot"
    audit = load_run_manifest(runner.runs_root)["model_identity"]["generation"]
    assert audit["targets"][scenario["scenario_id"]] == trajectory["provider_identity"]
    by_service = {r["service"]: r for r in audit["gateway_calls"]}
    assert by_service["llm"]["response_model"] == "kimi-reported-snapshot"
    assert by_service["tts"]["response_model"] is None
