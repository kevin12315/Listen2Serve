"""ASR 网关：音频 → 文本转写（qwen-audio-3.0-asr-flash，DashScope 云端）。

调用格式（官方文档验证，2026-08-05；字段复核 2026-08-26）：
- 端点：multimodal-generation/generation（同步）
- content 用 OpenAI 风格 `{"type": "input_audio", "input_audio": {"data": ...}}`
- **format/sample_rate 在顶层 parameters 中**（不在 content 内）
- data 支持 base64 data URL 与公网 URL
- 语种指定的官方字段是 `parameters.language_hints`（**数组**，Qwen-Audio-3.0-ASR-Flash
  最多 4 个）；旧写法 `parameters.language`（标量）不在文档字段表内，且本端点对未知
  parameters 一律静默接受（真机验证：传 bogus_field 也回 HTTP 200），故写错不报错、只是不生效
- 响应：文档口径为 `output.text`（全量文本）+ `output.sentence`（句级详情/词时间戳）；
  真机另额外回一层历史兼容的 `output.output.{text,sentence}`（内容与外层一致），
  解析以文档口径优先、嵌套层作回退
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from listen2serve.gateway.base import CallRecord, CallSink, MemoryCallSink, prompt_hash
from listen2serve.runtime.config import Settings, get_settings

logger = logging.getLogger(__name__)

ASR_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


@dataclass
class ASRResult:
    text: str
    language: str = "zh"
    duration_ms: int = 0
    words: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "qwen-audio-3.0-asr-flash"
    record: CallRecord | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration_ms": self.duration_ms,
            "words": self.words,
            "backend": self.backend,
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
            self.sink.append(CallRecord(
                service="asr", backend="dashscope", model=self.settings.asr_model,
                prompt_hash=ph, error=f"HTTP {resp.status_code}: {resp.text[:300]}",
            ))
            raise RuntimeError(f"ASR 失败 HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        out = data.get("output") or {}
        # 文档口径 output.{text,sentence} 优先；历史兼容层 output.output 作回退
        nested = out.get("output") if isinstance(out.get("output"), dict) else {}
        sentence = out.get("sentence") or nested.get("sentence") or {}
        text = out.get("text") or sentence.get("text") or nested.get("text") or ""
        words = sentence.get("words", [])
        duration_ms = sentence.get("end_time", 0) - sentence.get("begin_time", 0)
        self.sink.append(CallRecord(
            service="asr", backend="dashscope", model=self.settings.asr_model, prompt_hash=ph,
        ))
        return ASRResult(
            text=text,
            duration_ms=duration_ms,
            words=words,
        )
