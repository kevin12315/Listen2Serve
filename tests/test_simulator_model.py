"""Verify paper-model routing and wire requests without external network calls."""

from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from listen2serve.gateway.llm_gateway import LLMGateway
from listen2serve.model_registry import resolve_model
from listen2serve.runtime.config import Settings


def test_default_simulator_is_paper_model():
    settings = Settings(_env_file=None)
    assert resolve_model(settings.sim_model or settings.llm_model, settings.llm_model) == (
        "moonshot",
        "kimi-k3",
    )


def test_moonshot_request_contains_actual_paper_id(monkeypatch):
    import json

    captured = []

    def handle(request):
        captured.append((str(request.url), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "stub",
                "object": "chat.completion",
                "model": "kimi-k3",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                ],
            },
        )

    gw = LLMGateway(Settings(_env_file=None, moonshot_api_key="fixture"))
    client = OpenAI(
        api_key="fixture",
        base_url="https://api.moonshot.cn/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr(gw, "_client", lambda backend, name: client if backend == "moonshot" else None)
    result = gw.chat([{"role": "user", "content": "你好"}], thinking=False)
    assert result.model == "moonshot/kimi-k3"
    assert captured[0][0] == "https://api.moonshot.cn/v1/chat/completions"
    assert captured[0][1]["model"] == "kimi-k3"
    assert captured[0][1]["thinking"] == {"type": "disabled"}


def test_same_api_key_across_platforms_does_not_share_client(monkeypatch):
    settings = Settings(_env_file=None, openai_api_key="fixture", gemini_api_key="fixture")
    clients = []

    def make_client(**kwargs):
        client = SimpleNamespace(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("listen2serve.gateway.llm_gateway.OpenAI", make_client)
    gw = LLMGateway(settings)
    assert gw._client("openai") is not gw._client("google")
    assert clients[0].base_url != clients[1].base_url


def test_missing_simulator_key_has_clear_error():
    with pytest.raises(ValueError, match="Missing API key for moonshot"):
        LLMGateway(Settings(_env_file=None))._client("moonshot")
