"""Qwen Realtime WebSocket 事件解析。

事件模型对齐 DashScope Realtime 协议（OpenAI 兼容变体）：
- session.created / session.updated
- input_audio_buffer.speech_started / speech_stopped / committed
- response.created / response.done / response.cancelled
- response.audio.delta / response.audio.done
- response.audio_transcript.delta / response.audio_transcript.done
- response.function_call_arguments.delta / response.function_call_arguments.done
- response.output_item.added / response.output_item.done
- conversation.item.created / error

自研实现（协议细节参考 tau2-bench 验证结论，MIT License © Sierra）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QwenEvent:
    """统一事件结构。"""

    type: str
    data: dict[str, Any] = field(default_factory=dict)


class QwenTimeout(QwenEvent):
    """接收超时（非真实服务端事件）。"""

    def __init__(self) -> None:
        super().__init__(type="timeout")


def parse_qwen_event(raw: str | bytes | dict[str, Any]) -> QwenEvent:
    """解析服务端原始消息为 QwenEvent。"""
    if isinstance(raw, (str, bytes)):
        import json

        data = json.loads(raw)
    else:
        data = raw
    return QwenEvent(type=data.get("type", "unknown"), data=data)


# ---- 便捷访问器 ----
def event_audio_delta(event: QwenEvent) -> bytes | None:
    """response.audio.delta → PCM24 音频块。"""
    if event.type != "response.audio.delta":
        return None
    import base64

    try:
        return base64.b64decode(event.data.get("delta", ""))
    except Exception: # noqa: BLE001
        return None


def event_transcript_delta(event: QwenEvent) -> str | None:
    """response.audio_transcript.delta → 文本片段。"""
    if event.type != "response.audio_transcript.delta":
        return None
    return event.data.get("delta", "")


def event_function_call(event: QwenEvent) -> dict[str, Any] | None:
    """response.output_item.done 中携带完整 function_call（嵌套 FC 格式）。"""
    if event.type != "response.output_item.done":
        return None
    item = event.data.get("item", {})
    if item.get("type") != "function_call":
        return None
    return {
        "call_id": item.get("call_id", ""),
        "name": item.get("name", ""),
        "arguments": item.get("arguments", "{}"),
    }
