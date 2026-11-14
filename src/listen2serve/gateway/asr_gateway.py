"ASR 网关：音频 → 文本转写（qwen-audio-3.0-asr-flash，DashScope 云端）"

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from listen2serve.gateway.base import CallRecord, CallSink, MemoryCallSink, model_identity, prompt_hash
from listen2serve.runtime.config import Settings, get_settings

logger = logging.getLogger(__name__)

ASR_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


@dataclass
class ASRResult:
    text: str
    language: str = "zh"
    duration_ms: int = 0
    words: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "dashscope"
    requested_model: str | None = None
    response_model: str | None = None
    requested_host: str | None = None
    provider_host: str | None = None
    request_id: str | None = None
    response_model_differs: bool | None = None
    record: CallRecord | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration_ms": self.duration_ms,
            "words": self.words,
            "backend": self.backend,
            "requested_model": self.requested_model,
            "response_model": self.response_model,
            "requested_host": self.requested_host,
            "provider_host": self.provider_host,
            "request_id": self.request_id,
            "response_model_differs": self.response_model_differs,
        }


class ASRGateway:
    """云端 ASR 网关（qwen-audio-3.0-asr-flash）。"""

    def __init__(self, settings: Settings | None = None, sink: CallSink | None = None) -> None:
        self.settings = settings or get_settings()
        self.sink = sink or MemoryCallSink()

    def transcribe(
        self,
        audio_path: str | Path,
        format: str = "wav",
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> ASRResult:
        """转写音频文件（WAV/PCM16 16k）。

        Args:
            language: 语种提示（如 "zh"），按官方字段 language_hints 以数组下发；
                None = 不下发，由模型自动识别语种。
        """
        audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode()
        body: dict[str, Any] = {
            "model": self.settings.asr_model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": f"data:audio/{format};base64,{audio_b64}"},
                            }
                        ],
                    }
                ]
            },
            "parameters": {"format": format, "sample_rate": str(sample_rate)},
        }
        if language:
            body["parameters"]["language_hints"] = [language]
        ph = prompt_hash(str(Path(audio_path).stat().st_size) + audio_b64[:64])
        resp = httpx.post(
            ASR_URL,
            headers={"Authorization": f"Bearer {self.settings.dashscope_api_key or ''}"},
            json=body,
            timeout=180,
        )
        if resp.status_code != 200:
            self.sink.append(
                CallRecord(
                    service="asr",
                    backend="dashscope",
                    model=self.settings.asr_model,
                    prompt_hash=ph,
                    error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                    **model_identity(None, self.settings.asr_model, response=resp, requested_url=ASR_URL),
                )
            )
            raise RuntimeError(f"ASR 失败 HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        out = data.get("output") or {}

        nested = out.get("output") if isinstance(out.get("output"), dict) else {}
        sentence = out.get("sentence") or nested.get("sentence") or {}
        text = out.get("text") or sentence.get("text") or nested.get("text") or ""
        words = sentence.get("words", [])
        duration_ms = sentence.get("end_time", 0) - sentence.get("begin_time", 0)
        identity = model_identity(data, self.settings.asr_model, response=resp, requested_url=ASR_URL)
        record = CallRecord(
            service="asr",
            backend="dashscope",
            model=self.settings.asr_model,
            prompt_hash=ph,
            **identity,
        )
        self.sink.append(record)
        return ASRResult(
            text=text,
            duration_ms=duration_ms,
            words=words,
            backend="dashscope",
            record=record,
            **identity,
        )
