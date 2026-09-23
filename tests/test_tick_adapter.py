"""tick 适配器单测：音频封顶缓冲、比例转写、打断截断（fake provider，不触网）。"""

from __future__ import annotations

import base64

import pytest

from listen2serve.evaluation.objective import aggregate_pass_hat_k, pass_hat_k
from listen2serve.runtime.realtime.events import QwenEvent, QwenTimeout
from listen2serve.runtime.realtime.tick_adapter import QwenTickAdapter


class FakeProvider:
    """按 tick 吐出预置事件批次。"""

    def __init__(self, batches: list[list[QwenEvent]]) -> None:
        self.batches = batches
        self.tool_results: list[tuple[str, str]] = []
        self.cancelled = 0

    async def send_audio(self, pcm16: bytes) -> None:
        pass

    async def cancel_response(self) -> None:
        self.cancelled += 1

    async def send_tool_result(self, call_id: str, output: str) -> None:
        self.tool_results.append((call_id, output))

    async def receive_events(self, poll_s: float = 0.01):
        batch = self.batches.pop(0) if self.batches else []
        for ev in batch:
            yield ev
        while True:
            yield QwenTimeout()


def audio_delta(data: bytes, item_id: str = "i1") -> QwenEvent:
    return QwenEvent(
        type="response.audio.delta",
        data={"delta": base64.b64encode(data).decode(), "item_id": item_id},
    )


def transcript_delta(text: str, item_id: str = "i1") -> QwenEvent:
    return QwenEvent(
        type="response.audio_transcript.delta", data={"delta": text, "item_id": item_id}
    )


def make_adapter(batches: list[list[QwenEvent]], tick_ms: int = 200) -> QwenTickAdapter:
    return QwenTickAdapter(FakeProvider(batches), tick_ms=tick_ms)


@pytest.mark.asyncio
async def test_audio_capping_and_buffering():
    """单 tick 收到超量音频 → 封顶 bytes_per_tick，超出缓冲至后续 tick。"""
    big = b"\x01\x00" * 10000  # 20000 字节 > 9600（200ms@24k PCM16）
    adapter = make_adapter([[audio_delta(big), transcript_delta("你好世界一二三四五十")]])

    r1 = await adapter.run_tick(b"\x00" * 6400, 0)
    assert len(r1.agent_audio) == adapter.bytes_per_tick == 9600
    assert r1.buffer_bytes == 20000 - 9600
    assert 0 < len(r1.transcript) < 10  # 比例转写：只出部分文本

    r2 = await adapter.run_tick(b"\x00" * 6400, 1)
    assert len(r2.agent_audio) == 9600
    r3 = await adapter.run_tick(b"\x00" * 6400, 2)
    assert len(r3.agent_audio) == 20000 - 9600 * 2
    assert r3.buffer_bytes == 0
    # 播完后转写全部吐出
    assert (r1.transcript + r2.transcript + r3.transcript) == "你好世界一二三四五十"


@pytest.mark.asyncio
async def test_barge_in_truncation():
    """speech_started（用户插话）→ 丢弃缓冲，跳过被截断 item 的后续音频。"""
    big = b"\x01\x00" * 10000
    speech_started = QwenEvent(type="input_audio_buffer.speech_started", data={})
    late_audio = audio_delta(b"\x02\x00" * 1000, item_id="i1")  # 截断后同 item 的音频
    adapter = make_adapter([
        [audio_delta(big)],
        [speech_started, late_audio],
        [audio_delta(b"\x03\x00" * 1000, item_id="i2")],  # 新 item 正常通过
    ])

    r1 = await adapter.run_tick(b"\x00" * 6400, 0)
    assert r1.buffer_bytes > 0

    r2 = await adapter.run_tick(b"\x00" * 6400, 1)
    assert r2.speech_started is True
    assert r2.was_truncated is True
    assert r2.truncated_bytes >= 20000 - 9600  # 缓冲 + 迟到音频均被丢弃
    assert r2.agent_audio == b""
    assert r2.buffer_bytes == 0

    r3 = await adapter.run_tick(b"\x00" * 6400, 2)
    assert len(r3.agent_audio) == 2000  # 新 item 不受 skip 影响


@pytest.mark.asyncio
async def test_tool_call_recorded_and_result_queued():
    """工具调用记录保留（name/arguments/call_id），结果下一 tick 先 cancel 再投递。"""
    fc = QwenEvent(
        type="response.output_item.done",
        data={"item": {"type": "function_call", "call_id": "c1", "name": "query_debt",
                       "arguments": '{"customer_name": "张三"}'}},
    )
    adapter = make_adapter([[fc], []])
    r1 = await adapter.run_tick(b"\x00" * 6400, 0)
    assert r1.tool_calls == [{"call_id": "c1", "name": "query_debt",
                              "arguments": '{"customer_name": "张三"}'}]

    adapter.queue_tool_result("c1", "欠款 8000 元")
    provider: FakeProvider = adapter.provider  # type: ignore[assignment]
    await adapter.run_tick(b"\x00" * 6400, 1)
    assert provider.cancelled == 1
    assert provider.tool_results == [("c1", "欠款 8000 元")]


@pytest.mark.asyncio
async def test_session_usage_accumulated():
    """response.done 携带的 usage 被累计（Realtime 用量审计）。"""
    done1 = QwenEvent(type="response.done",
                      data={"response": {"usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}})
    done2 = QwenEvent(type="response.done",
                      data={"response": {"usage": {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280}}})
    adapter = make_adapter([[done1], [done2]])
    await adapter.run_tick(b"\x00" * 6400, 0)
    await adapter.run_tick(b"\x00" * 6400, 1)
    assert adapter.session_usage == {
        "input_tokens": 300, "output_tokens": 130, "total_tokens": 430, "responses": 2,
    }


class TestPassHatK:
    def test_values(self):
        assert pass_hat_k(4, 3, 1) == 0.75
        assert pass_hat_k(4, 3, 2) == 0.5  # C(3,2)/C(4,2) = 3/6
        assert pass_hat_k(4, 4, 4) == 1.0
        assert pass_hat_k(4, 1, 2) == 0.0  # 成功数 < k

    def test_invalid(self):
        with pytest.raises(ValueError):
            pass_hat_k(2, 1, 3)

    def test_aggregate(self):
        mean, per_task = aggregate_pass_hat_k(
            {"a": [True, True, False, True], "b": [True, True, True, True], "c": [True]},
            k=2,
        )
        assert per_task == {"a": 0.5, "b": 1.0}  # c 试验数不足 k，跳过
        assert mean == 0.75
