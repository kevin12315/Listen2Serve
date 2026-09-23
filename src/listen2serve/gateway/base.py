"""网关抽象：LLM/TTS 调用统一接口与调用记录。

所有网关调用（LLM chat / TTS 合成）都经过这里定义的抽象，并记录
model/version/prompt_hash/latency/tokens/成本，供审计与报告使用。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class CallRecord:
    """一次网关调用的审计记录。"""

    service: str # "llm" | "tts"
    backend: str # "google" | "openai" | "dashscope" | "local"
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
        }


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
