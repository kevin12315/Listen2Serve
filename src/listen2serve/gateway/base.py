"""网关抽象：LLM/TTS 调用统一接口与调用记录。

所有网关调用（LLM chat / TTS 合成）都经过这里定义的抽象，并记录
model/version/prompt_hash/latency/tokens/成本，供审计与报告使用。
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit


@dataclass
class CallRecord:
    """一次网关调用的审计记录。"""

    service: str  # "llm" | "tts"
    backend: str  # "google" | "openai" | "dashscope" | "local"
    model: str
    prompt_hash: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0
    error: str | None = None
    # 网关按模型适配表改写请求参数时的留痕（如 temperature 被收敛到该模型的合法值域）。
    # 空串 = 未改写。**改写必须留痕，不许静默** —— 否则落盘 meta 记的参数与实际发出的不一致。
    param_note: str = ""
    ts: float = field(default_factory=time.time)
    # model remains a compatibility alias for the requested wire model.
    requested_model: str | None = None
    response_model: str | None = None
    requested_host: str | None = None
    provider_host: str | None = None
    request_id: str | None = None
    response_model_differs: bool | None = None

    def __post_init__(self) -> None:
        if self.requested_model is None:
            self.requested_model = self.model

    def identity_dict(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "response_model": self.response_model,
            "requested_host": self.requested_host,
            "provider_host": self.provider_host,
            "request_id": self.request_id,
            "response_model_differs": self.response_model_differs,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "backend": self.backend,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "latency_ms": round(self.latency_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_cny": round(self.cost_cny, 6),
            "error": self.error,
            "param_note": self.param_note,
            "ts": self.ts,
            **self.identity_dict(),
        }


_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,255}\Z")
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_IDENTITY_KEYS = (
    "requested_model",
    "response_model",
    "requested_host",
    "provider_host",
    "response_model_differs",
)


def safe_model_name(value: Any) -> str | None:
    """Retain a reported model identifier, never arbitrary response metadata."""
    if not isinstance(value, str) or not _MODEL_NAME.fullmatch(value) or value.startswith("sk-"):
        return None
    return value


def safe_request_id(value: Any) -> str | None:
    if not isinstance(value, str) or not _REQUEST_ID.fullmatch(value) or value.startswith("sk-"):
        return None
    return value


def host_of(url: Any) -> str | None:
    """Hostname only: omit credentials, paths, query strings and fragments."""
    if not isinstance(url, str):
        return None
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def _field(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def model_identity(
    payload: Any, requested_model: str, *, response: Any = None, requested_url: str | None = None
) -> dict[str, Any]:
    """Keep provider-reported identity separate from the request and configuration.

    No response model is inferred from a URL, the backend name, a successful call,
    or a request echo in our own settings. Nested output.model is accepted only
    when it is present in the provider's response payload.
    """
    reported = safe_model_name(_field(payload, "model"))
    output = _field(payload, "output")
    if reported is None and isinstance(output, dict):
        reported = safe_model_name(output.get("model"))
    request_id = safe_request_id(_field(payload, "request_id")) or safe_request_id(
        _field(payload, "_request_id")
    )
    actual_host = None
    if response is not None:
        try:
            actual_host = host_of(str(response.request.url))
        except (AttributeError, RuntimeError):
            pass
        headers = getattr(response, "headers", {})
        for key in ("x-request-id", "x-dashscope-request-id", "request-id"):
            request_id = request_id or safe_request_id(headers.get(key))
    differs = reported != requested_model if reported is not None else None
    if differs:
        logging.getLogger(__name__).warning(
            "Provider reported model %s for requested model %s; verify the provider's alias/snapshot mapping.",
            reported,
            requested_model,
        )
    return {
        "requested_model": requested_model,
        "response_model": reported,
        "requested_host": host_of(requested_url),
        "provider_host": actual_host,
        "request_id": request_id,
        "response_model_differs": differs,
    }


def summarize_identities(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact audit rows; old records supply requests, never observed responses."""
    groups: dict[tuple, dict[str, Any]] = {}
    for record in records:
        row = {key: record.get(key) for key in _IDENTITY_KEYS}
        row["requested_model"] = row["requested_model"] or record.get("model")
        row["response_model"] = safe_model_name(row["response_model"])
        row.update(service=record.get("service"), backend=record.get("backend"))
        key = tuple(row.items())
        calls = record.get("calls", 1)
        if type(calls) is not int or calls < 1:
            raise ValueError("Identity call count must be a positive integer")
        groups.setdefault(key, {**row, "calls": 0})["calls"] += calls
    return list(groups.values())


def prompt_hash(text: str) -> str:
    """稳定散列，用于审计 prompt 版本。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class CallSink(Protocol):
    """调用记录接收方（默认内存列表，可由 run_batch 注入文件 sink）。"""

    def append(self, record: CallRecord) -> None: ...


class MemoryCallSink:
    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def append(self, record: CallRecord) -> None:
        self.records.append(record)

    def dump(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.records]


class JsonlCallSink:
    """追加式 JSONL 文件 sink（批跑审计落盘，run_batch 注入）。"""

    def __init__(self, path: str) -> None:
        from pathlib import Path

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CallRecord) -> None:
        import json

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
