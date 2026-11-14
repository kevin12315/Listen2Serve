"""Verify Kimi's mode-dependent temperature at SDK and audit boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from listen2serve.gateway import llm_gateway
from listen2serve.gateway.base import MemoryCallSink
from listen2serve.runtime.config import Settings


def gateway(monkeypatch, *, fail=False):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if fail:
            raise RuntimeError("mock provider rejection")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"text":"测试输出"}'), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm_gateway, "OpenAI", lambda **kwargs: fake_client)
    config = Settings(
        _env_file=None, moonshot_api_key="fake-moonshot", dashscope_api_key="fake-dashscope",
        openai_api_key="fake-openai", gemini_api_key="fake-gemini", volc_api_key=None, volc_api_key2=None,
    )
    sink = MemoryCallSink()
    return llm_gateway.LLMGateway(config, sink=sink), calls, sink


@pytest.mark.parametrize("model", ["kimi-k3", "moonshot/kimi-k3"])
@pytest.mark.parametrize("thinking,effective,mode", [
    (False, 0.6, "disabled"), (True, 1.0, "enabled"), (None, 1.0, "default"),
])
def test_kimi_sdk_temperature_matches_thinking_and_audit(monkeypatch, model, thinking, effective, mode):
    client, calls, sink = gateway(monkeypatch)
    messages = [{"role": "user", "content": "保持原提示词"}]
    result = client.chat(messages, model=model, temperature=0.7, max_tokens=320, seed=123, thinking=thinking)
    assert calls[0]["model"] == "kimi-k3"
    assert calls[0]["temperature"] == effective
    assert calls[0]["messages"] is messages
    assert calls[0]["max_tokens"] == 320 and calls[0]["seed"] == 123
    if thinking is None:
        assert "extra_body" not in calls[0]
    else:
        assert calls[0]["extra_body"] == {"thinking": {"type": mode}}
    assert messages == [{"role": "user", "content": "保持原提示词"}]
    assert result.record is sink.records[0]
    assert f"temperature 0.7→{effective}" in result.record.param_note
    assert f"thinking={mode}" in sink.dump()[0]["param_note"]


@pytest.mark.parametrize("model,temperature,thinking", [
    ("kimi-k3", 0.6, False), ("kimi-k3", 1.0, True), ("kimi-k3", 1.0, None),
    ("qwen-plus", 0.7, False), ("qwen-plus", 0.7, True), ("qwen-plus", 0.7, None),
    ("google/gemini-3.8-flash", 0.0, None), ("openai/other-model", 0.7, None),
])
def test_accepted_or_unregistered_models_keep_requested_temperature(monkeypatch, model, temperature, thinking):
    client, calls, sink = gateway(monkeypatch)
    client.chat([{"role": "user", "content": "测试"}], model=model, temperature=temperature, thinking=thinking)
    assert calls[0]["temperature"] == temperature
    assert sink.records[0].param_note == ""


def test_other_registered_domains_keep_existing_nearest_value_behavior(monkeypatch):
    monkeypatch.setitem(llm_gateway.TEMPERATURE_DOMAIN, "other-model", (0.0, 0.4, 0.9))
    client, calls, sink = gateway(monkeypatch)
    client.chat([{"role": "user", "content": "测试"}], model="openai/other-model", temperature=0.7, thinking=True)
    assert calls[0]["temperature"] == 0.9
    assert "temperature 0.7→0.9" in sink.records[0].param_note
    assert "thinking=" not in sink.records[0].param_note


def test_every_adapted_call_keeps_note_even_after_warning_was_deduplicated(monkeypatch):
    client, calls, sink = gateway(monkeypatch)
    for _ in range(2):
        client.chat([{"role": "user", "content": "测试"}], model="kimi-k3", temperature=0.7, thinking=False)
    assert [call["temperature"] for call in calls] == [0.6, 0.6]
    assert all("temperature 0.7→0.6" in record.param_note for record in sink.records)
    assert all("thinking=disabled" in record.param_note for record in sink.records)


def test_failed_call_still_records_effective_temperature_and_mode(monkeypatch):
    client, calls, sink = gateway(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="mock provider rejection"):
        client.chat([{"role": "user", "content": "测试"}], model="kimi-k3", temperature=0.7, thinking=False)
    assert calls[0]["temperature"] == 0.6
    assert "temperature 0.7→0.6" in sink.records[0].param_note
    assert "thinking=disabled" in sink.records[0].param_note
    assert sink.records[0].error == "RuntimeError: mock provider rejection"
