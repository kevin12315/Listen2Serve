"""Qwen Realtime WebSocket Provider（被测 Agent 接入）。

协议要点（DashScope Realtime，OpenAI 兼容变体）：
- WSS: `wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={model}`，Bearer 鉴权
- 输入 PCM16 16kHz（base64），输出 PCM24 24kHz
- session.update 的 tools 采用嵌套格式 `{"type": "function", "function": {...}}`
- tool result 前需先发 `response.cancel` 取消进行中的响应
- ping_interval=10s / ping_timeout=30s / max_size=16MB
- 支持 server_vad（barge-in 打断）与 manual 两种轮转模式

基于 tau2-bench 验证的协议细节自研实现（MIT License © Sierra，参考其架构模式）。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, AsyncGenerator

import websockets

from listen2serve.runtime.realtime.events import QwenEvent, QwenTimeout, parse_qwen_event

logger = logging.getLogger(__name__)

QWEN_REALTIME_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
QWEN_INPUT_SAMPLE_RATE = 16000 # 服务端固定
QWEN_OUTPUT_SAMPLE_RATE = 24000 # 服务端固定


class QwenRealtimeProvider:
    """Qwen Realtime 全双工 Provider：音频流 + 文本 + 函数调用。"""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-audio-3.0-realtime-plus",
        voice: str = "longanxiaoxin",
        base_url: str = QWEN_REALTIME_URL,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.base_url = base_url
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.session_id: str | None = None

    @property
    def is_connected(self) -> bool:
        if self.ws is None:
            return False
        try:
            from websockets.protocol import State

            return self.ws.state == State.OPEN
        except Exception: # noqa: BLE001
            return False

    async def connect(self, max_retries: int = 3) -> None:
        """建立连接并等待 session.created。"""
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}?model={self.model}"
                self.ws = await websockets.connect(
                    url,
                    additional_headers={"Authorization": f"Bearer {self.api_key}"},
                    ping_interval=10,
                    ping_timeout=30,
                    close_timeout=5,
                    max_size=16 * 1024 * 1024,
                )
                raw = await asyncio.wait_for(self.ws.recv(), timeout=30)
                data = json.loads(raw)
                if data.get("type") != "session.created":
                    raise RuntimeError(f"握手失败，期望 session.created，实际 {data.get('type')}")
                self.session_id = data.get("session", {}).get("id")
                logger.info("Qwen Realtime 已连接 session=%s", self.session_id)
                return
            except Exception as exc: # noqa: BLE001
                last_exc = exc
                await self.disconnect()
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Qwen Realtime 连接失败（重试 {max_retries} 次）: {last_exc}")

    async def disconnect(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception: # noqa: BLE001
                pass
            self.ws = None
            self.session_id = None

    # ---- 发送 ----
    async def _send(self, payload: dict[str, Any]) -> None:
        if not self.is_connected:
            raise RuntimeError("未连接 Qwen Realtime")
        await self.ws.send(json.dumps(payload, ensure_ascii=False))

    async def configure_session(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        voice: str | None = None,
        vad_mode: str = "manual",
        modalities: list[str] | None = None,
    ) -> None:
        """配置会话：指令/工具/音色/轮转检测。等待 session.updated 确认。"""
        formatted = []
        for t in tools or []:
            formatted.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        turn_detection = None
        if vad_mode == "server_vad":
            turn_detection = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 800,
            }
        session_config = {
            "type": "session.update",
            "session": {
                "instructions": system_prompt,
                "modalities": modalities or ["text", "audio"],
                "voice": voice or self.voice,
                "tools": formatted,
                "turn_detection": turn_detection,
            },
        }
        await self._send(session_config)
        # 等待 session.updated 确认
        deadline = asyncio.get_event_loop().time() + 30
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=30)
            data = json.loads(raw)
            if data.get("type") == "session.updated":
                return
            if data.get("type") == "error":
                raise RuntimeError(f"会话配置失败: {data.get('error')}")
        raise RuntimeError("等待 session.updated 超时")

    async def send_audio(self, pcm16: bytes) -> None:
        """追加输入音频（PCM16 16k）。"""
        await self._send(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16).decode()}
        )

    async def commit_audio(self) -> None:
        """提交音频缓冲并请求响应。"""
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})

    async def clear_audio_buffer(self) -> None:
        await self._send({"type": "input_audio_buffer.clear"})

    async def send_text(self, text: str, commit: bool = True) -> None:
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        if commit:
            await self._send({"type": "response.create"})

    async def send_tool_result(self, call_id: str, output: str) -> None:
        """发送工具结果；先 cancel 进行中的响应再提交（协议要求）。

        若当前无活动响应（response 已 done），cancel 会返回错误，此时忽略继续。
        """
        try:
            await self._send({"type": "response.cancel"})
        except Exception as exc: # noqa: BLE001 - 无活动响应时服务端拒绝 cancel，忽略
            logger.debug("response.cancel 无活动响应（忽略）: %s", exc)
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            }
        )
        await self._send({"type": "response.create"})

    async def cancel_response(self) -> None:
        await self._send({"type": "response.cancel"})

    async def request_response(self) -> None:
        await self._send({"type": "response.create"})

    # ---- 接收 ----
    async def receive_events(self, poll_s: float = 0.01) -> AsyncGenerator[QwenEvent, None]:
        """轮询接收事件；无事件时产出 QwenTimeout。"""
        if not self.is_connected:
            raise RuntimeError("未连接 Qwen Realtime")
        while self.is_connected:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=poll_s)
                yield parse_qwen_event(raw)
            except asyncio.TimeoutError:
                yield QwenTimeout()
            except websockets.ConnectionClosed as exc:
                logger.error("Qwen Realtime 连接关闭: %s", exc)
                raise RuntimeError(f"WebSocket 连接意外关闭: {exc}") from exc
