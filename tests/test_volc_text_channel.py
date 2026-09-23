"""豆包下行文本通道的回归测试（零 API，事件形态取自 2026-09-08 真机下行实录）。

为什么单独一个文件而不是塞进 test_tick_adapter.py：这里守的是**跨两个模块的配对不变量**
（`VolcRealtimeProvider._normalize` 给的 utterance 键 ↔ `QwenTickAdapter` 的音频/转写配对），
任何一侧"看起来更合理"的改动都会把它拆开，而拆开的症状是**静默的**：
`trace.agent_text` 恒空、不报错、音频一切正常。

这个缺陷的实际代价（2026-09-08 定位）：豆包 Layer L 三条件的 `agent_text` 全空，于是
① FlowRate/TaskScore/closing_ok 被当成"结构性失效"（实测 0.008/0.012）；
② 更重的是生成侧 —— 用户模拟器 `next_turn(agent_reply="")` 全程看不到客服说了什么，
   且 `peek_interrupt` 要求 ≥8 字 ⇒ 脚本标注的 barge-in 在豆包上物理不可能触发。
两条都被归因成"端点只回音频不回文本""豆包明显更安静"，而事实是端点一直在发文本。

真机下行事件的关键形态（本文件的 fixture 即按此写死）：
  · `response.output_text.delta` keys = [delta, event_id, question_id, **response_id**, type]
  · `response.output_audio.delta` keys = [delta, event_id, type]        ← **不带任何 id**
  · 顺序：文本增量先到 → `response.output_audio.started`(带 response_id) → 音频增量
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from listen2serve.runtime.realtime.events import QwenEvent, QwenTimeout
from listen2serve.runtime.realtime.tick_adapter import QwenTickAdapter
from listen2serve.runtime.realtime.volc_provider import (
    _IDLE_PUMP_AFTER_S,
    VolcRealtimeProvider,
)

RESP_ID = "54810143316620290"      # 真机：本轮客服回复的 response_id
QUESTION_ID = "54810143316620546"  # 真机：本轮用户条目的 item_id
DIALOG_ID = "6e07f577-3bd6-402f-aaae-8cc32030f773"  # 真机：session.created 的 session.id

TEXT_PARTS = ["您可以直接", "通过当时办理的手机银行APP", "找到对应普惠贷款账户操作还款。"]
TEXT_FULL = "".join(TEXT_PARTS)

ONE_FRAME = 9600  # 200ms @ 24k PCM16 = adapter 单 tick 封顶播出量


def volc_provider() -> VolcRealtimeProvider:
    p = VolcRealtimeProvider(api_key="x")
    p.dialog_id = DIALOG_ID  # configure_session 收到 session.created 之后的状态
    return p


def raw(kind: str, **fields) -> dict:
    return {"type": kind, **fields}


def audio_b64(n_bytes: int = ONE_FRAME, filler: bytes = b"\x01\x00") -> str:
    return base64.b64encode(filler * (n_bytes // 2)).decode()


def doubao_batches(p: VolcRealtimeProvider) -> list[list[QwenEvent]]:
    """真机顺序的豆包原始事件 → 逐条过真实 `_normalize` → 按 tick 分组的 QwenEvent。

    注意：`_normalize` 是有状态的（active response_id 靠它累积），所以必须**按到达顺序
    一次性预归一化**，得到的事件再喂给适配器 —— 这样测的才是真机链路。
    """
    ev = lambda kind, **f: p._normalize(raw(kind, **f))[0]  # noqa: E731
    head = [
        [ev("response.output_text.delta", delta=TEXT_PARTS[0],
            response_id=RESP_ID, question_id=QUESTION_ID)],
        [ev("response.output_text.delta", delta=TEXT_PARTS[1], response_id=RESP_ID),
         ev("response.output_text.delta", delta=TEXT_PARTS[2], response_id=RESP_ID)],
        [ev("response.output_text.done", text=TEXT_FULL, response_id=RESP_ID),
         ev("response.output_audio.started", response_id=RESP_ID, tts_type="default")],
    ]
    # 6 帧音频：每帧占一个 tick（= tick 适配器每 tick 只收一帧的节奏）
    return head + [[ev("response.output_audio.delta", delta=audio_b64())] for _ in range(6)]


class FakeStreamProvider:
    """按 tick 吐出预置事件批次（与 test_tick_adapter.FakeProvider 同形）。"""

    def __init__(self, batches: list[list[QwenEvent]]) -> None:
        self.batches = batches
        self.cancelled = 0

    async def send_audio(self, pcm16: bytes) -> None:
        pass

    async def cancel_response(self) -> None:
        self.cancelled += 1

    async def send_tool_result(self, call_id: str, output: str) -> None:
        pass

    async def receive_events(self, poll_s: float = 0.01):
        batch = self.batches.pop(0) if self.batches else []
        for e in batch:
            yield e
        while True:
            yield QwenTimeout()


def make_adapter(batches: list[list[QwenEvent]]) -> QwenTickAdapter:
    return QwenTickAdapter(FakeStreamProvider(batches), tick_ms=200)


async def drain(adapter: QwenTickAdapter, ticks: int) -> tuple[str, int]:
    """跑 ticks 个 tick，返回（累加出的 agent_text, 播出的客服音频字节数）。"""
    texts, audio = [], 0
    for tick in range(ticks):
        r = await adapter.run_tick(b"\x00" * 6400, tick)
        texts.append(r.transcript)
        audio += len(r.agent_audio)
    return "".join(texts), audio


@pytest.mark.asyncio
async def test_doubao_agent_text_is_not_empty():
    """核心回归：豆包一轮回复的 agent_text 必须等于下行文本（历史上恒为 ''）。"""
    adapter = make_adapter(doubao_batches(volc_provider()))
    text, audio = await drain(adapter, 9)

    assert text == TEXT_FULL, f"agent_text 漏/错：{text!r}"
    assert audio == ONE_FRAME * 6, "音频侧被键对齐改动影响"


@pytest.mark.asyncio
async def test_transcript_is_released_proportionally_to_played_audio():
    """整段音频一次到达时，转写只能按已播出比例释放，不许把没播出的先吐完。

    这条守的是"打断轮不泄文本"的性质：若为了省事把文本改成直接累加到 trace，
    本条会失败，而错误只会以"客服看起来说了更多"的形式暴露出来。
    必须用单帧大音频才能测到比例：“每 tick 收一帧、播一帧”时比例恒为 1。
    """
    p = volc_provider()
    ev = lambda kind, **f: p._normalize(raw(kind, **f))[0]  # noqa: E731
    adapter = make_adapter([
        [ev("response.output_text.delta", delta=TEXT_FULL, response_id=RESP_ID)],
        [ev("response.output_audio.delta", delta=audio_b64(ONE_FRAME * 3))],  # 整轮 = 3 tick 量
    ])
    # 共 4 个 tick：tick0 只有文本（不出字）、tick1–3 分三次把 3 帧量播完
    texts = [(await adapter.run_tick(b"\x00" * 6400, t)).transcript for t in range(4)]

    # 文本跑得比音频快也不能先泄：这个 tick 一个字都没播出（音频还没到）
    assert texts[0] == "", f"没播出音频就不该出文本：{texts[0]!r}"
    assert texts[1] and texts[1] != TEXT_FULL, f"第二 tick 只应出一小段：{texts[1]!r}"
    assert "".join(texts) == TEXT_FULL
    assert TEXT_FULL.startswith(texts[1])
    assert TEXT_FULL.startswith(texts[1] + texts[2]), "释放的文本必须是原文的连续前缀"


@pytest.mark.asyncio
async def test_barge_in_skip_resets_on_next_response():
    """打断后：被取消 item 的迟到音频仍丢弃，但**新响应的音频必须放行**。

    原实现 `_skip_item_id` 除 init 外无复位点。豆包音频帧不带 id、全靠 provider 给的
    逐轮键 ⇒ 一次真打断会让该会话其后所有客服音频都被当成"被截断 item 的迟到音频"丢掉。
    """
    def a_delta(item_id: str, filler: bytes) -> QwenEvent:
        return QwenEvent(type="response.audio.delta",
                         data={"delta": audio_b64(filler=filler), "item_id": item_id})

    adapter = make_adapter([
        [a_delta("item-A", b"\x01\x02"), a_delta("item-A", b"\x01\x02")],  # 两帧：留一帧在缓冲
        [QwenEvent(type="input_audio_buffer.speech_started", data={})],  # 用户插话 → skip=A
        [a_delta("item-A", b"\x03\x04")],                              # A 的迟到音频 → 应丢
        [QwenEvent(type="response.created", data={"response": {"id": "resp-B"}})],  # 复位
        [a_delta("item-B", b"\x05\x06")],                              # B 的音频 → 应放行
    ])

    r1 = await adapter.run_tick(b"\x00" * 6400, 0)
    assert len(r1.agent_audio) == ONE_FRAME and r1.buffer_bytes == ONE_FRAME
    r2 = await adapter.run_tick(b"\x00" * 6400, 1)
    assert r2.was_truncated is True and r2.buffer_bytes == 0
    r3 = await adapter.run_tick(b"\x00" * 6400, 2)
    assert r3.agent_audio == b"", "被取消 item 的迟到音频不该播出"
    r4 = await adapter.run_tick(b"\x00" * 6400, 3)   # 仅 response.created
    assert r4.agent_audio == b""
    r5 = await adapter.run_tick(b"\x00" * 6400, 4)
    assert len(r5.agent_audio) == ONE_FRAME, "新响应的音频被 stale skip 误杀"


@pytest.mark.asyncio
async def test_transcript_done_fills_only_an_empty_utterance():
    """`.done` 只在**该 utterance 一个字都没收到 delta** 时兜底，绝不覆盖/补齐。"""
    p = volc_provider()
    ev = lambda kind, **f: p._normalize(raw(kind, **f))[0]  # noqa: E731

    # 场景 1：只有 done 没有 delta → 整句可用
    only_done = make_adapter(
        [[ev("response.output_text.done", text=TEXT_FULL, response_id=RESP_ID)],
         [ev("response.output_audio.started", response_id=RESP_ID, tts_type="default")]]
        + [[ev("response.output_audio.delta", delta=audio_b64())] for _ in range(6)])
    text, _ = await drain(only_done, 9)
    assert text == TEXT_FULL

    # 场景 2：delta 已到一半，done 带整句 → 不得把没播出的尾部补进来
    partial = make_adapter([
        [QwenEvent(type="response.audio_transcript.delta",
                   data={"delta": TEXT_PARTS[0], "item_id": "i1"})],
        [QwenEvent(type="response.audio_transcript.done",
                   data={"transcript": TEXT_FULL, "item_id": "i1"})],
    ])
    await partial.run_tick(b"\x00" * 6400, 0)
    await partial.run_tick(b"\x00" * 6400, 1)
    assert partial._utterances["i1"].transcript_received == TEXT_PARTS[0], \
        "done 覆盖/补齐了已有 delta（会泄出未播出的文字）"


def test_provider_response_id_scopes_per_turn():
    """轮边界：canceled/done 后清空 active 键；无主音频回落 dialog 级键（旧行为）。"""
    p = volc_provider()
    p._normalize(raw("response.output_text.delta", delta="x", response_id=RESP_ID))
    audio = p._normalize(raw("response.output_audio.delta", delta=""))[0]
    assert audio.data["item_id"] == RESP_ID, "音频帧没继承本轮 response_id（原缺陷）"

    p._normalize(raw("response.canceled", response_id=RESP_ID))
    assert p._active_response_id == "", "轮结束后没清空，下一轮会串上一轮的文本"
    orphan = p._normalize(raw("response.output_audio.delta", delta=""))[0]
    assert orphan.data["item_id"] == DIALOG_ID


def test_idle_pump_fills_only_input_gaps():
    """静音保活：业务帧在走时一个字节都不插；停滞 ≥ 阈值后才按 20ms 补静音。

    `is_connected` 是 property，不能拿实例属性盖 ⇒ 用子类把连接状态固定为真。
    """

    class AlwaysConnected(VolcRealtimeProvider):
        @property
        def is_connected(self) -> bool:  # noqa: D401 - 测试替身
            return True

    async def scenario():
        sent: list[str] = []
        p = AlwaysConnected(api_key="x")

        async def fake_send(payload):
            sent.append(payload["type"])

        p._send = fake_send  # type: ignore[assignment]
        loop = asyncio.get_event_loop()
        p._last_input_at = loop.time()
        task = asyncio.create_task(p._idle_pump())

        # ① 业务上行持续（每 20ms 刷时间戳）→ 保活不得介入
        for _ in range(int(0.12 / 0.02)):
            await asyncio.sleep(0.02)
            p._last_input_at = loop.time()
        assert sent == [], f"业务期间被插了静音帧：{sent}"

        # ② 停滞（= 编排器阻塞在 simulator.next_turn 的那 6.7~68s 的缩影）→ 保活接管
        await asyncio.sleep(_IDLE_PUMP_AFTER_S + 0.08)
        assert len(sent) >= 2, f"停滞期没补静音：{sent}"
        assert set(sent) == {"input_audio_buffer.append"}
        assert p.stream_diag()["silence_keepalive_frames"] == len(sent), \
            "保活帧没进可审计计数（这个故障模式全静默，没计数就无从自证）"
        task.cancel()

    asyncio.run(scenario())


def test_pump_catch_up_frames_covers_time_without_bursts():
    """保活必须能**追赶**：单帧实际周期（20ms + `_send` 开销）比 20ms 长，
    每轮只发一帧会越欠越多。2026-09-08 实测：不追赶时某通断流 76.6s 只补到 24.0s。
    封顶 5 帧防止在长停滞后来一个“单帧过大”的 burst（旧坑 `52000033`）。
    """
    from listen2serve.runtime.realtime.volc_provider import (
        _IDLE_PUMP_MAX_FRAMES, _pump_catch_up_frames)

    assert _pump_catch_up_frames(0.03) == 1      # 只欠一帧就补一帧
    assert _pump_catch_up_frames(0.05) == 2
    assert _pump_catch_up_frames(0.13) == _IDLE_PUMP_MAX_FRAMES
    assert _pump_catch_up_frames(30.0) == _IDLE_PUMP_MAX_FRAMES, "长停滞不得一次性补满"


class TestDoubaoStabilityKnobs:
    """同一份缺陷报告带的两个稳定性旋钮（拖尾等待 / 重试退避）。

    背对背实测：豆包首响 p50 1.5s，但同一条输入出现过 24.7s 的拖尾（旧 30s 阈值
    正好撞上 ⇒ 旧批 17/48 「Agent 无响应」）；而旧重试间隔 2/4s 三次连撞同一次抖动。
    """

    def test_streaming_endpoints_wait_longer(self):
        from types import SimpleNamespace

        from listen2serve.runtime.orchestrator import (
            STREAMING_GRACE_TICKS, FullDuplexOrchestrator)
        s = SimpleNamespace(max_response_ticks=150)

        class Plain:
            pass

        class Streaming:
            needs_continuous_audio = True

        limit = FullDuplexOrchestrator._response_wait_ticks
        assert limit(Plain(), s) == 150, "不声明连续输入的端点不得改变时序口径"
        assert limit(Streaming(), s) == 150 + STREAMING_GRACE_TICKS
        # 缺属性（假 provider / 旧 provider）不得抛错，按一般端点处理
        assert limit(object(), s) == 150

    def test_volc_retry_backoff_is_minutes_not_seconds(self):
        from listen2serve.runtime.run_batch import BatchRunner

        backoff = BatchRunner._retry_backoff_s
        assert backoff("dashscope/qwen-audio-3.0-realtime-plus", 0) == 2.0  # qwen 不变
        assert backoff("volc/doubao-seeduplex-3.0", 0) == 20.0
        assert backoff("volc/doubao-seeduplex-3.0", 1) == 40.0
