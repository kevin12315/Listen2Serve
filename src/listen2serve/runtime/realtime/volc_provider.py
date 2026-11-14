"豆包 Seeduplex 3.0（火山引擎「端到端实时语音-全双工版本」）WebSocket Provider"

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import websockets

from listen2serve.gateway.base import model_identity
from listen2serve.runtime.realtime.events import QwenEvent, QwenTimeout

logger = logging.getLogger(__name__)

VOLC_REALTIME_URL = "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue"
VOLC_INPUT_SAMPLE_RATE = 16000  # 服务端固定（asr pcm 16k）
VOLC_OUTPUT_SAMPLE_RATE = 24000  # 服务端固定（tts pcm_s16le 24k）
VOLC_MODEL = "1.2.6.1"  # 全双工版本固定单值（官方《接入必读》）
VOLC_DEFAULT_VOICE = "zh_female_xiaohe_jupiter_bigtts"  # demo 默认音色

# 豆包事件类型（官方 realtime_client.py 常量，逐字对齐）
_T_SESSION_CREATED = "session.created"
_T_SESSION_UPDATED = "session.updated"
_T_SESSION_CLOSED = "session.closed"
_T_INPUT_COMMITTED = "input_audio_buffer.committed"
_T_ASR_STARTED = "conversation.item.input_audio_transcription.started"
_T_ASR_DELTA = "conversation.item.input_audio_transcription.delta"
_T_ASR_COMPLETED = "conversation.item.input_audio_transcription.completed"
_T_ASR_FAILED = "conversation.item.input_audio_transcription.failed"
_T_OUT_TEXT_DELTA = "response.output_text.delta"
_T_OUT_TEXT_DONE = "response.output_text.done"
_T_OUT_AUDIO_STARTED = "response.output_audio.started"
_T_OUT_AUDIO_DELTA = "response.output_audio.delta"
_T_OUT_AUDIO_DONE = "response.output_audio.done"
_T_FC_DONE = "response.function_call_arguments.done"
_T_RESP_CANCELED = "response.canceled"
_T_RESP_DONE = "response.done"
_T_ERROR = "error"

# 输入保活阈值：业务侧每 20ms 一块，连续 50ms 没有上行就视为“停滞”并接管静音。
# 取 50ms 而非更小：避开 asyncio 调度抖动导致的短暂间隙，不至于在业务帧之间插静音。
_IDLE_PUMP_AFTER_S = 0.05
# 单次追赶封顶（5 帧 = 100ms 静音）：既跟得上真实时间，又不会在长停滞后来一个大 burst。
_IDLE_PUMP_MAX_FRAMES = 5

_IDLE_PUMP_MAX_DEBT_S = 1.0
_PUMP_FRAME_S = 0.02


def _pump_catch_up_frames(debt_s: float) -> int:
    """欠了多少秒 → 本次应补几帧（封顶 `_IDLE_PUMP_MAX_FRAMES`，但至少 1 帧）。"""
    return min(int(debt_s // _PUMP_FRAME_S), _IDLE_PUMP_MAX_FRAMES) or 1


class VolcRealtimeProvider:
    """豆包 Seeduplex 全双工 Provider：音频流 + 文本 + 函数调用。

    对外接口与 `QwenRealtimeProvider` 完全一致（connect / configure_session / send_audio /
    commit_audio / send_text / send_tool_result / cancel_response / receive_events /
    disconnect），故 `QwenTickAdapter` 可直接包裹。
    """

    # 全双工音频服务端要求**持续**有音频输入；纯文本轮会话若断流会报
    # 52000033 AudioServerNoAudioInputTooLongError。探针/编排器据此在文本轮期间补静音流。
    needs_continuous_audio = True

    def __init__(
        self,
        api_key: str,
        model: str = VOLC_MODEL,
        voice: str = VOLC_DEFAULT_VOICE,
        base_url: str = VOLC_REALTIME_URL,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.base_url = base_url
        self.identity = model_identity(None, model, requested_url=base_url)
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.session_id: str = str(uuid.uuid4())  # 客户端生成，session.create 携带
        self.dialog_id: str | None = None  # session.created 返回的服务端 id
        self._event_id = 0
        self._write_lock = asyncio.Lock()
        # 本轮回复的 response_id：文本/started 类事件带它，音频 delta 不带它。
        self._active_response_id: str = ""
        # 输入保活（needs_continuous_audio）：最近一次上行音频的时刻 + 保活任务。
        self._last_input_at: float = 0.0
        self._pump_task: asyncio.Task | None = None
        # 保活补发的帧数（坐实“这一路真补过静音”：这个故障模式全是静默的，
        # 没计数的话下次再出现仍只能靠猜）。每帧 20ms。
        self._pump_frames = 0

    def _next_event_id(self) -> str:
        self._event_id += 1
        return f"event_{self._event_id}"

    def _capture_identity(self, data: dict[str, Any]) -> None:
        payload = data.get("session") if data.get("type", "").startswith("session.") else data.get("response")
        if not isinstance(payload, dict):
            return
        identity = model_identity(
            payload, self.model, requested_url=self.base_url, response=getattr(self.ws, "response", None)
        )
        for key, value in identity.items():
            if value is not None:
                self.identity[key] = value

    @property
    def is_connected(self) -> bool:
        if self.ws is None:
            return False
        try:
            from websockets.protocol import State

            return self.ws.state == State.OPEN
        except Exception:  # noqa: BLE001
            return False

    async def _send(self, payload: dict[str, Any]) -> None:
        if not self.is_connected:
            raise RuntimeError("未连接豆包 Realtime")
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        async with self._write_lock:
            await self.ws.send(data)

    # ---- 生命周期 ----
    async def connect(self, max_retries: int = 3) -> None:
        """仅建立 WS 握手（携 X-Api-Key）。豆包的 session.created 在 session.create **之后**
        才回（与 Qwen「连上即推 session.created」不同），故握手与建会话分属 connect /
        configure_session 两步 —— 正好对齐 QwenRealtimeProvider 的两步接口。"""
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                self.ws = await websockets.connect(
                    self.base_url,
                    additional_headers={"X-Api-Key": self.api_key},
                    ping_interval=None,
                    close_timeout=5,
                    max_size=16 * 1024 * 1024,
                )
                try:
                    logid = self.ws.response_headers.get("X-Tt-Logid")
                    if logid:
                        logger.info("豆包 Realtime 已连接 logid=%s", logid)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Optional response metadata unavailable: %s", type(exc).__name__)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                await self.disconnect()
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
        # 404 且 body 含 "does not exist" ⇒ 账户未开通全双工（邀测未批），给出可诊断信息
        msg = str(last_exc)
        hint = ""
        resp = getattr(last_exc, "response", None)
        body = getattr(resp, "body", b"") or b""
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8", "replace")
        if getattr(resp, "status_code", None) == 404 or "does not exist" in body:
            hint = (
                f"；服务端响应：{body[:160]!r} ⇒ 该 API Key 账户**未开通**全双工"
                f"（Seeduplex 3.0 邀测），需在火山控制台申请/开通后重试"
            )
        raise RuntimeError(f"豆包 Realtime 连接失败（重试 {max_retries} 次）: {msg}{hint}")

    async def disconnect(self) -> None:
        """优雅关闭：先 session.close 并等 session.closed（带超时），再关 WS。
        直接断开会触发服务端 ContextCanceled(55000001)（官方《接入必读》）。"""
        if self._pump_task is not None and not self._pump_task.done():
            self._pump_task.cancel()  # 保活任务必须在 session.close 前停，否则会往已关会话里发帧
        if self._pump_frames:
            # 留一条可审计的流水：断流补救到底有没有发生，不能只靠“这批复活了”反推。
            logger.info(
                "豆包会话结束：静音保活补发 %d 帧（≈%.1fs 输入）dialog_id=%s",
                self._pump_frames,
                self._pump_frames * 0.02,
                self.dialog_id,
            )
        self._pump_task = None
        if self.ws is None:
            return
        try:
            if self.is_connected:
                await self._send({"type": "session.close", "event_id": self._next_event_id()})
                await self._wait_session_closed(timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("session.close 忽略: %s", exc)
        try:
            await self.ws.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Transport close: %s", type(exc).__name__)
        self.ws = None

    async def _wait_session_closed(self, timeout: float = 3.0) -> None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            except Exception:  # noqa: BLE001
                return
            ev = self._loads(raw)
            if ev and ev.get("type") == _T_SESSION_CLOSED:
                return

    @staticmethod
    def _loads(raw: str | bytes) -> dict[str, Any]:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    # ---- 会话配置 ----
    async def configure_session(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        voice: str | None = None,
        vad_mode: str = "server_vad",
        modalities: list[str] | None = None,
    ) -> None:
        "发 session.create 并等待 session.created"
        session = {
            "id": self.session_id,
            "model": self.model,
            "instructions": system_prompt,
            "audio": {
                "input": {"format": {"type": "pcm", "rate": VOLC_INPUT_SAMPLE_RATE}},
                "output": {
                    "format": {"type": "pcm_s16le", "rate": VOLC_OUTPUT_SAMPLE_RATE},
                    "voice": voice or self.voice,
                },
            },
            "tools": self._format_tools(tools),
        }
        extension = {
            "asr": {"extra": {}},
            "tts": {"extra": {}},
            "dialog": {
                "location": {"city": "北京", "country": "中国", "country_code": "CN"},
                "extra": {"enable_loudness_norm": True, "enable_music": False},
            },
        }
        await self._send(
            {
                "type": "session.create",
                "event_id": self._next_event_id(),
                "session": session,
                "extension": extension,
            }
        )
        deadline = asyncio.get_event_loop().time() + 30
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=30)
            ev = self._loads(raw)
            t = ev.get("type")
            if t == _T_SESSION_CREATED:
                self._capture_identity(ev)
                self.dialog_id = (ev.get("session") or {}).get("id") or self.session_id
                logger.info("豆包 session.created dialog_id=%s", self.dialog_id)
                self._start_idle_pump()  # 会话已活：从此保证输入流不断
                return
            if t == _T_ERROR:
                raise RuntimeError(f"豆包会话建立失败: {ev.get('error')}")
        raise RuntimeError("等待 session.created 超时（30s）")

    @staticmethod
    def _format_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in tools or []:
            out.append(
                {
                    "type": "function",
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return out

    async def update_session(self, session: dict[str, Any], extension: dict[str, Any] | None = None) -> None:
        """会话中动态改配置（session.update，全量覆盖 tools）。"""
        session = dict(session)
        session.setdefault("id", self.dialog_id or self.session_id)
        await self._send(
            {
                "type": "session.update",
                "event_id": self._next_event_id(),
                "session": session,
                "extension": extension,
            }
        )

    # ---- 发送 ----
    async def send_audio(self, pcm16: bytes) -> None:
        "追加输入音频（PCM16 16k，base64）= input_audio_buffer.append(=TaskRequest)"
        chunk = 640
        loop = asyncio.get_event_loop()
        for i in range(0, len(pcm16), chunk):
            await self._send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16[i : i + chunk]).decode("ascii"),
                }
            )
            self._last_input_at = loop.time()
            await asyncio.sleep(0.02)

    # ---- 输入保活（`needs_continuous_audio` 的实现点）----
    def _start_idle_pump(self) -> None:
        "启一个后台静音保活任务（会话建立后调，`disconnect` 里负责取消）"
        if self._pump_task is None or self._pump_task.done():
            self._last_input_at = asyncio.get_event_loop().time()
            self._pump_task = asyncio.create_task(self._idle_pump())

    def stream_diag(self) -> dict[str, Any]:
        "输入保活的可审计计数（进 sim_status.rt_diag）"
        return {
            "silence_keepalive_frames": self._pump_frames,
            "silence_keepalive_s": round(self._pump_frames * 0.02, 1),
        }

    async def _idle_pump(self) -> None:
        "业务音频停滞后，按实时速率补静音帧，保证输入流不中断"
        silence = base64.b64encode(b"\x00" * 640).decode("ascii")
        while self.is_connected:
            await asyncio.sleep(0.02)
            loop = asyncio.get_event_loop()
            # 阈值 50ms > 业务侧 20ms pacing：正常 tick 期间永不介入，只在停滞时接管。
            debt = loop.time() - self._last_input_at
            if debt < _IDLE_PUMP_AFTER_S:
                continue
            for _ in range(_pump_catch_up_frames(debt)):
                try:
                    await self._send({"type": "input_audio_buffer.append", "audio": silence})
                except Exception as exc:  # noqa: BLE001 - 连接已断则自然停止，不抛
                    logger.debug("静音保活帧发送失败（停止保活）: %s", exc)
                    return
                self._pump_frames += 1
                self._last_input_at += _PUMP_FRAME_S  # 记账式推进：欠多少补多少，不抹零

            if loop.time() - self._last_input_at > _IDLE_PUMP_MAX_DEBT_S:
                self._last_input_at = loop.time() - _IDLE_PUMP_MAX_DEBT_S
            await asyncio.sleep(0.01)  # 避免追赶完仍满速转圈

    async def commit_audio(self) -> None:
        "提交音频缓冲（=EndASR）+ 补一段静音尾触发服务端判停"
        await self._send({"type": "input_audio_buffer.commit", "event_id": self._next_event_id()})
        loop = asyncio.get_event_loop()
        for _ in range(75):  # 75 × 640B × 20ms ≈ 1.5s 静音尾
            await self._send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(b"\x00" * 640).decode("ascii"),
                }
            )
            self._last_input_at = loop.time()
            await asyncio.sleep(0.02)

    async def clear_audio_buffer(self) -> None:
        # 豆包无对应「清空输入缓冲」事件；打断走 cancel_response()。保留空实现以对齐接口。
        return None

    async def send_text(self, text: str, commit: bool = True) -> None:
        """以 user 文本条目注入（conversation.item.create）。豆包无独立 response.create，
        服务端据 VAD/条目自动续答；commit 参数仅为接口兼容。"""
        await self._send(
            {
                "type": "conversation.item.create",
                "event_id": self._next_event_id(),
                "items": [
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
                ],
            }
        )

    async def send_tool_result(self, call_id: str, output: str) -> None:
        """回传函数结果（role=tool，call_id 必须与下行 response.function_call_arguments.done 一致）。"""
        await self._send(
            {
                "type": "conversation.item.create",
                "event_id": self._next_event_id(),
                "items": [
                    {
                        "type": "message",
                        "role": "tool",
                        "call_id": call_id,
                        "content": [{"type": "input_text", "text": output}],
                    }
                ],
            }
        )

    async def cancel_response(self) -> None:
        """客户端打断（response.cancel = 原 ClientInterrupt）。"""
        await self._send({"type": "response.cancel", "event_id": self._next_event_id()})

    async def request_response(self) -> None:
        # 豆包无 response.create（服务端自驱）；保留空实现以对齐接口。
        return None

    # ---- 接收（归一化为 QwenEvent 类型名，供 QwenTickAdapter 直接消费）----
    async def receive_events(self, poll_s: float = 0.01) -> AsyncGenerator[QwenEvent, None]:
        if not self.is_connected:
            raise RuntimeError("未连接豆包 Realtime")
        while self.is_connected:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=poll_s)
            except TimeoutError:
                yield QwenTimeout()
                continue
            except websockets.ConnectionClosed as exc:
                logger.error("豆包 Realtime 连接关闭: %s", exc)
                raise RuntimeError(f"WebSocket 连接意外关闭: {exc}") from exc
            ev = self._loads(raw)
            if not ev:
                continue
            for qev in self._normalize(ev):
                yield qev

    def _normalize(self, ev: dict[str, Any]) -> list[QwenEvent]:
        """豆包事件 → 一个或多个 QwenEvent（类型名对齐 DashScope/Qwen Realtime）。

        [UNTESTED] 标注项以官方 demo 字段推导，开通后按真机校正：
          - response.done 的 usage 具体结构（PDF 为截图）：这里把整个 usage 透传到
            data.response.usage，阳性对照 A1 用**递归**查找 audio token 字段，可容忍结构差异。
          - barge-in：豆包无 Qwen 的 input_audio_buffer.speech_started；用 ASR started
            （模型识别到用户首字）作为「用户开始说话」的打断信号映射。
        """
        self._capture_identity(ev)
        t = ev.get("type", "")

        if ev.get("response_id"):
            self._active_response_id = str(ev["response_id"])
        rid = self._active_response_id or self.dialog_id or ""

        if t in (_T_RESP_DONE, _T_RESP_CANCELED):
            self._active_response_id = ""

        if t == _T_OUT_AUDIO_DELTA:
            return [
                QwenEvent(type="response.audio.delta", data={"delta": ev.get("delta", ""), "item_id": rid})
            ]
        if t == _T_OUT_TEXT_DELTA:
            return [
                QwenEvent(
                    type="response.audio_transcript.delta",
                    data={"delta": ev.get("delta", ""), "item_id": rid},
                )
            ]
        if t == _T_OUT_TEXT_DONE:
            return [
                QwenEvent(
                    type="response.audio_transcript.done",
                    data={"transcript": ev.get("text", ""), "item_id": rid},
                )
            ]
        if t == _T_OUT_AUDIO_STARTED:
            return [QwenEvent(type="response.created", data={"response": {"id": rid}})]
        if t == _T_OUT_AUDIO_DONE:
            return [QwenEvent(type="response.audio.done", data={"item_id": rid})]
        if t == _T_ASR_STARTED:
            return [QwenEvent(type="input_audio_buffer.speech_started", data={})]
        if t == _T_ASR_COMPLETED:
            return [
                QwenEvent(
                    type="input_audio_buffer.speech_stopped",
                    data={"transcript": ev.get("transcript") or ev.get("text") or ""},
                )
            ]
        if t == _T_FC_DONE:
            out: list[QwenEvent] = []
            for item in ev.get("items") or []:
                out.append(
                    QwenEvent(
                        type="response.function_call_arguments.done",
                        data={
                            "call_id": str(item.get("call_id", "")),
                            "name": str(item.get("name", "")),
                            "arguments": str(item.get("arguments", "{}")),
                        },
                    )
                )
            return out
        if t == _T_RESP_DONE:
            usage = ev.get("usage") or {}
            return [
                QwenEvent(
                    type="response.done",
                    data={"response": {"id": rid, "status": "completed", "usage": usage}},
                )
            ]
        if t == _T_RESP_CANCELED:
            return [QwenEvent(type="response.cancelled", data={"response": {"id": rid}})]
        if t == _T_ERROR:
            return [QwenEvent(type="error", data={"error": ev.get("error") or ev})]
        if t in (
            _T_SESSION_CREATED,
            _T_SESSION_UPDATED,
            _T_SESSION_CLOSED,
            _T_INPUT_COMMITTED,
            _T_ASR_DELTA,
            _T_ASR_FAILED,
        ):
            return [QwenEvent(type=t, data=ev)]
        # conversation.item.added/retrieved/updated/deleted 等：原样透传（tick adapter 忽略）
        return [QwenEvent(type=t, data=ev)]
