"""Qwen Realtime 离散时间 tick 适配器（对齐 tau-voice DiscreteTimeAdapter 语义）。

职责（每 tick 一次 run_tick 调用）：
- 发送本 tick 的用户音频块（16k PCM16，200ms）；
- 在 tick 预算内收集服务端事件（音频 delta / 转写 delta / VAD / 工具调用 / done）；
- **音频封顶与缓冲**：每 tick 最多播出 bytes_per_tick 字节客服音频，超出部分
  缓冲到后续 tick（tau-voice buffer_excess_audio 同语义）；
- **比例转写**：按本 tick 实际播出的音频字节数，等比切分该 utterance 的转写文本
  （tau-voice UtteranceTranscript 同语义）；
- **打断（barge-in）**：server_vad 检测到用户说话（speech_started）时，丢弃未播出
  的缓冲音频并跳过被截断 item 的后续音频（tau-voice truncation 同语义）；
- **工具结果投递**：结果在下一 tick 开始时投递，投递前先 response.cancel
  （协议要求，tau-voice _flush_pending_tool_results 同语义）。

自研实现（协议细节参考 tau2-bench 验证结论，MIT License © Sierra）。
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import Any

from listen2serve.runtime.realtime.events import QwenEvent, QwenTimeout
from listen2serve.runtime.realtime.qwen_provider import (
    QWEN_OUTPUT_SAMPLE_RATE,
    QwenRealtimeProvider,
)

logger = logging.getLogger(__name__)


@dataclass
class UtteranceTranscript:
    """单个 utterance（item）的音频/转写比例分配跟踪。"""

    item_id: str
    audio_bytes_received: int = 0
    transcript_received: str = ""
    audio_bytes_played: int = 0
    chars_shown: int = 0

    def add_audio(self, num_bytes: int) -> None:
        self.audio_bytes_received += num_bytes

    def add_transcript(self, text: str) -> None:
        self.transcript_received += text

    def take_for_audio(self, bytes_played: int) -> str:
        """按累计播出比例取出应显示的转写增量。"""
        self.audio_bytes_played += bytes_played
        if self.audio_bytes_received == 0 or not self.transcript_received:
            return ""
        ratio = min(1.0, self.audio_bytes_played / self.audio_bytes_received)
        target_chars = int(len(self.transcript_received) * ratio)
        if target_chars <= self.chars_shown:
            return ""
        text = self.transcript_received[self.chars_shown : target_chars]
        self.chars_shown = target_chars
        return text

    def flush_remaining(self) -> str:
        """utterance 结束时取出尚未显示的转写尾部。"""
        text = self.transcript_received[self.chars_shown :]
        self.chars_shown = len(self.transcript_received)
        return text


@dataclass
class AdapterTickResult:
    """一个 tick 的适配器输出。"""

    tick: int = 0
    agent_audio: bytes = b"" # 本 tick 播出的客服音频（24k PCM16，封顶未填充）
    transcript: str = "" # 与本 tick 播出音频成比例的转写增量
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    response_done: bool = False
    done_status: str = "" # response.done 携带的状态（completed/cancelled/failed/incomplete，审计用）
    speech_started: bool = False # server VAD 检测到用户语音（打断信号）
    was_truncated: bool = False
    truncated_bytes: int = 0
    has_activity: bool = False # 本 tick 是否收到任何真实事件
    agent_output: bool = False # 本 tick 是否收到**下行产出**事件（音频帧/转写增量/response.done）
    # has_activity 与 agent_output 的差别不是文字游戏：has_activity 含输入侧事件
    # （如 server VAD 的 speech_started），且各端点“还有哪些事件”并不一致；而
    # 「客服这一轮是否还在产出」必须是纯下行口径，才能给编排器当判轮依据。
    buffer_bytes: int = 0 # 尚未播出的缓冲音频字节数
    tools_flushed: int = 0 # 本 tick 投递的工具结果数（供编排器判定工具轮后续语音窗口）
    errors: list[str] = field(default_factory=list)


class QwenTickAdapter:
    """Qwen Realtime 的 tick 级全双工适配器。"""

    def __init__(
        self,
        provider: QwenRealtimeProvider,
        tick_ms: int = 200,
        output_sample_rate: int = QWEN_OUTPUT_SAMPLE_RATE,
    ) -> None:
        self.provider = provider
        self.tick_ms = tick_ms
        self.bytes_per_tick = int(output_sample_rate * 2 * tick_ms / 1000)
        # 缓冲与转写状态
        self._buffer: list[tuple[bytes, str | None]] = [] # (音频, item_id)
        self._utterances: dict[str, UtteranceTranscript] = {}
        self._current_item_id: str | None = None
        self._skip_item_id: str | None = None
        self._pending_tool_results: list[tuple[str, str]] = [] # (call_id, output)
        self._seen_call_ids: set[str] = set()
        self._last_flush_count: int = 0 # 本 tick 投递的工具结果数
        # 会话 token 用量（response.done 携带的 usage 累计，审计/熔断用）
        self._usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "responses": 0}

    @property
    def session_usage(self) -> dict[str, int]:
        """本会话累计 token 用量（来自服务端 response.done usage 上报）。"""
        return dict(self._usage)

    # ---- 生命周期 ----
    async def connect(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        voice: str | None = None,
        vad_mode: str = "server_vad",
    ) -> None:
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt, tools=tools, voice=voice, vad_mode=vad_mode
        )

    async def disconnect(self) -> None:
        await self.provider.disconnect()
        self._buffer.clear()
        self._utterances.clear()
        self._pending_tool_results.clear()
        self._seen_call_ids.clear()

    @property
    def buffer_bytes(self) -> int:
        return sum(len(d) for d, _ in self._buffer)

    def queue_tool_result(self, call_id: str, output: str) -> None:
        """登记工具结果，下一 tick 开始时投递（先 cancel 再投递）。"""
        self._pending_tool_results.append((call_id, output))

    async def cancel_response(self) -> None:
        try:
            await self.provider.cancel_response()
        except Exception as exc: # noqa: BLE001 - 无活动响应时服务端拒绝，忽略
            logger.debug("response.cancel 忽略: %s", exc)

    # ---- 主入口 ----
    async def run_tick(self, user_pcm16: bytes, tick: int = 0) -> AdapterTickResult:
        """运行一个 tick：投递工具结果 → 发送用户音频 → 收集事件 → 封顶输出。"""
        result = AdapterTickResult(tick=tick)
        loop = asyncio.get_event_loop()
        tick_start = loop.time()
        self._last_flush_count = 0
        await self._flush_pending_tool_results()

        # 并发：发送用户音频 + 在剩余预算内收集事件（对齐 tau-voice _execute_tick）
        async def _collect() -> list[QwenEvent]:
            elapsed = loop.time() - tick_start
            budget = max(0.01, self.tick_ms / 1000 - elapsed)
            return await self._collect_events(budget)

        _, events = await asyncio.gather(self.provider.send_audio(user_pcm16), _collect())

        for event in events:
            self._process_event(result, event)

        # 封顶：本 tick 最多播出 bytes_per_tick 字节
        played = self._pop_played()
        result.agent_audio = b"".join(d for d, _ in played)
        result.transcript = self._proportional_transcript(played)
        result.buffer_bytes = self.buffer_bytes
        result.tools_flushed = self._last_flush_count
        return result

    # ---- 内部 ----
    async def _flush_pending_tool_results(self) -> None:
        if not self._pending_tool_results:
            return
        await self.cancel_response()
        await asyncio.sleep(0.1) # 等 cancel 生效（tau-voice 同款）
        n = len(self._pending_tool_results)
        for call_id, output in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, output)
        self._pending_tool_results.clear()
        self._last_flush_count = n # 供 run_tick 写入结果（投递后模型会再出语音）

    async def _collect_events(self, budget_s: float) -> list[QwenEvent]:
        events: list[QwenEvent] = []
        loop = asyncio.get_event_loop()
        end = loop.time() + budget_s
        gen = self.provider.receive_events(poll_s=0.01)
        try:
            async for ev in gen:
                if not isinstance(ev, QwenTimeout):
                    events.append(ev)
                if loop.time() >= end:
                    break
        finally:
            await gen.aclose()
        return events

    def _process_event(self, result: AdapterTickResult, event: QwenEvent) -> None:
        etype = event.type
        if etype != "timeout":
            result.has_activity = True
        # 产出侧信号先行置位：必须在下面那些 `return`（空音频、被截断 item 的迟到
        # 音频丢弃）之前，否则「模型还在说、只是这一帧被我方丢了」会被读成静默。
        if etype in ("response.audio.delta", "response.audio_transcript.delta",
                     "response.audio_transcript.done", "response.done",
                     "response.output_item.done"):
            result.agent_output = True

        if etype == "response.created":
            # 新响应起点 = 上一个被打断 item 的归属结束。**必须复位 `_skip_item_id`**：
            # 全文件除 init 外只有下面一处赋值（无其他复位点），而豆包的音频帧不带 id，
            # 其 utterance 键由 provider 侧逐轮刷新（一个会话只有一个 dialog_id 兼底）——
            # 不复位则一次真打断会让**该会话其后所有客服音频**都被当成
            # “被截断 item 的迟到音频”静默丢弃（下面 `item_id == self._skip_item_id` 那一支）。
            # 对 qwen 无害：`response.created` 总在该响应第一个音频帧之前，而 qwen 本来
            # 每 item 一个 id（skip 值与本应放行新 item 不相等）。
            self._skip_item_id = None

        if etype == "response.audio.delta":
            item_id = event.data.get("item_id") or self._current_item_id
            try:
                audio = base64.b64decode(event.data.get("delta", ""))
            except Exception: # noqa: BLE001
                return
            if not audio:
                return
            if self._skip_item_id is not None and item_id == self._skip_item_id:
                result.truncated_bytes += len(audio) # 被截断 item 的后续音频丢弃
                return
            self._buffer.append((audio, item_id))
            if item_id:
                self._current_item_id = item_id
                self._utterance(item_id).add_audio(len(audio))

        elif etype == "response.audio_transcript.delta":
            item_id = event.data.get("item_id") or self._current_item_id
            delta = event.data.get("delta", "")
            if item_id and delta:
                self._utterance(item_id).add_transcript(delta)

        elif etype == "response.audio_transcript.done":
            # 兜底：只在该 utterance **一个字都没收到 delta** 时拿 done 的整句顶上。
            # 不做覆盖/补齐：done 带的是**计划播出的整句**，打断轮里尾部音频已被丢弃，
            # 补齐会把“用户根本没听到的文字”泄进 agent_text（比例转写就是为了防这件事）。
            item_id = event.data.get("item_id") or self._current_item_id
            text = event.data.get("transcript", "")
            if item_id and text:
                ut = self._utterance(item_id)
                if not ut.transcript_received:
                    ut.add_transcript(text)

        elif etype == "input_audio_buffer.speech_started":
            result.speech_started = True
            # 打断：丢弃未播出的缓冲音频，跳过被截断 item 的后续音频
            if self._buffer:
                result.truncated_bytes += self.buffer_bytes
                self._buffer.clear()
                result.was_truncated = True
                self._skip_item_id = self._current_item_id
                logger.debug("barge-in：丢弃缓冲音频，skip_item=%s", self._skip_item_id)

        elif etype == "response.function_call_arguments.done":
            self._add_tool_call(
                result,
                call_id=event.data.get("call_id", ""),
                name=event.data.get("name", ""),
                arguments=event.data.get("arguments", "{}"),
            )

        elif etype == "response.output_item.done":
            item = event.data.get("item", {})
            if item.get("type") == "function_call":
                self._add_tool_call(
                    result,
                    call_id=item.get("call_id", ""),
                    name=item.get("name", ""),
                    arguments=item.get("arguments", "{}"),
                )

        elif etype == "response.done":
            result.response_done = True
            response = event.data.get("response") or {}
            result.done_status = response.get("status", "") or result.done_status
            # 记录服务端上报的 token 用量（Realtime 审计缺口修复）
            usage = response.get("usage") or {}
            if usage:
                self._usage["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                self._usage["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                self._usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
                self._usage["responses"] += 1

        elif etype == "error":
            err = event.data.get("error", {})
            msg = str(err.get("message", err))
            # 无活动响应的 cancel 报错属预期，不计入
            if "no active response" not in msg.lower():
                result.errors.append(msg)
                logger.warning("Qwen Realtime error: %s", msg)

    def _add_tool_call(self, result: AdapterTickResult, call_id: str, name: str, arguments: str) -> None:
        """记录工具调用（两种事件来源去重）。"""
        if not call_id or call_id in self._seen_call_ids:
            return
        self._seen_call_ids.add(call_id)
        result.tool_calls.append({"call_id": call_id, "name": name, "arguments": arguments})
        logger.debug("工具调用: %s(%s)", name, call_id)

    def _utterance(self, item_id: str) -> UtteranceTranscript:
        if item_id not in self._utterances:
            self._utterances[item_id] = UtteranceTranscript(item_id=item_id)
        return self._utterances[item_id]

    def _pop_played(self) -> list[tuple[bytes, str | None]]:
        """从缓冲取出本 tick 播出的音频（≤ bytes_per_tick），其余留缓冲。"""
        played: list[tuple[bytes, str | None]] = []
        total = 0
        while self._buffer and total < self.bytes_per_tick:
            data, item_id = self._buffer[0]
            space = self.bytes_per_tick - total
            if len(data) <= space:
                played.append((data, item_id))
                total += len(data)
                self._buffer.pop(0)
            else:
                played.append((data[:space], item_id))
                self._buffer[0] = (data[space:], item_id)
                total = self.bytes_per_tick
        return played

    def _proportional_transcript(self, played: list[tuple[bytes, str | None]]) -> str:
        """按本 tick 各 item 播出字节数，等比取转写增量。"""
        if not played:
            return ""
        by_item: dict[str, int] = {}
        for data, item_id in played:
            if item_id:
                by_item[item_id] = by_item.get(item_id, 0) + len(data)
        parts: list[str] = []
        for item_id, n in by_item.items():
            ut = self._utterances.get(item_id)
            if ut is None:
                continue
            # 该 item 已播完（缓冲无剩余）→ 补齐尾部转写，避免文本残留
            remaining = sum(len(d) for d, i in self._buffer if i == item_id)
            text = ut.take_for_audio(n)
            if remaining == 0 and ut.audio_bytes_received > 0 and ut.audio_bytes_played >= ut.audio_bytes_received:
                text += ut.flush_remaining()
            if text:
                parts.append(text)
        return "".join(parts)
