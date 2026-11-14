"""Item identity changes interruption diagnostics without changing playback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_tick_adapter import audio_delta, make_adapter, transcript_delta

from listen2serve.runtime.agent import Agent, AgentTickResult
from listen2serve.runtime.config import Settings
from listen2serve.runtime.orchestrator import FullDuplexOrchestrator
from listen2serve.runtime.realtime.events import QwenEvent
from listen2serve.runtime.realtime.tick_adapter import AdapterTickResult
from listen2serve.runtime.user_simulator import UserTurnResult


async def test_adapter_reports_all_played_and_dropped_items_without_changing_output():
    first, second = b"\x01\x00" * 1000, b"\x02\x00" * 6000
    adapter = make_adapter(
        [
            [
                audio_delta(first, "first"),
                transcript_delta("甲", "first"),
                audio_delta(second, "second"),
                transcript_delta("乙乙乙乙乙乙", "second"),
            ],
            [
                audio_delta(b"\x03\x00" * 1000, "unplayed"),
                QwenEvent(type="input_audio_buffer.speech_started", data={}),
            ],
        ]
    )
    played = await adapter.run_tick(b"\x00" * 6400, 0)
    assert played.played_item_ids == ["first", "second"]
    assert played.truncated_item_ids == []
    assert played.agent_audio == first + second[:7600]
    assert played.transcript == "甲乙乙乙"
    assert played.buffer_bytes == 4400
    truncated = await adapter.run_tick(b"\x00" * 6400, 1)
    assert truncated.played_item_ids == []
    assert truncated.truncated_item_ids == ["second", "unplayed"]
    assert truncated.was_truncated and truncated.truncated_bytes == 6400
    assert truncated.agent_audio == b"" and truncated.transcript == "" and truncated.buffer_bytes == 0


async def test_adapter_unknown_item_is_capable_but_has_no_identity():
    adapter = make_adapter(
        [
            [audio_delta(b"\x01\x00" * 6000, "")],
            [QwenEvent(type="input_audio_buffer.speech_started", data={})],
        ]
    )
    played = await adapter.run_tick(b"\x00" * 6400, 0)
    assert played.agent_audio and played.played_item_ids == []
    truncated = await adapter.run_tick(b"\x00" * 6400, 1)
    assert truncated.was_truncated and truncated.truncated_item_ids == []


@pytest.mark.parametrize("legacy", [False, True])
async def test_agent_wrapper_preserves_item_identity_capability(legacy):
    result = AdapterTickResult(agent_audio=b"pcm", transcript="原文", was_truncated=True)
    result.played_item_ids = ["played", "also-played"]
    result.truncated_item_ids = ["truncated", "also-truncated"]
    wrapped = SimpleNamespace(**vars(result))
    if legacy:
        del wrapped.played_item_ids
        del wrapped.truncated_item_ids
    agent = Agent.__new__(Agent)
    agent.adapter = SimpleNamespace(run_tick=AsyncMock(return_value=wrapped))
    actual = await agent.run_tick(b"user", 4)
    assert actual.audio == b"pcm" and actual.transcript == "原文" and actual.was_truncated
    assert actual.played_item_ids == (None if legacy else ["played", "also-played"])
    assert actual.truncated_item_ids == (None if legacy else ["truncated", "also-truncated"])
    if not legacy:
        assert actual.played_item_ids is not result.played_item_ids
        assert actual.truncated_item_ids is not result.truncated_item_ids


class Simulator:
    def __init__(self):
        self.calls = 0

    def next_turn(self, *_args):
        self.calls += 1
        return UserTurnResult(
            text=f"用户{self.calls}", pcm16=b"\x01\x00" * 3200 * (1 if self.calls == 1 else 4)
        )

    def peek_interrupt(self, *_args):
        return False


class EventAgent:
    model_spec = "fake/item-diagnostics"

    def __init__(self, played_id, truncated_ids, *, capable=True):
        self.played_id = played_id
        self.truncated_ids = truncated_ids
        self.capable = capable

    async def start(self):
        pass

    async def stop(self):
        pass

    async def run_tick(self, chunk, tick=0):
        events = {
            1: AgentTickResult(audio=b"\x02\x00" * 4800, transcript="上轮。", played_item_ids=["old"]),
            5: AgentTickResult(
                audio=b"\x03\x00" * 4800, transcript="本轮开头", played_item_ids=[self.played_id]
            ),
            6: AgentTickResult(
                was_truncated=True,
                speech_started=True,
                truncated_item_ids=self.truncated_ids,
                audio=b"\x05\x00" * 4800 if self.truncated_ids == ["unplayed"] else b"",
                played_item_ids=["another-new"] if self.truncated_ids == ["unplayed"] else [],
            ),
            8: AgentTickResult(audio=b"\x04\x00" * 4800, transcript="本轮完成。", played_item_ids=["final"]),
        }
        result = events.get(tick, AgentTickResult(played_item_ids=[], truncated_item_ids=[]))
        if not self.capable:
            result.played_item_ids = None
            result.truncated_item_ids = None
        return result


@pytest.mark.parametrize(
    "played_id,truncated_ids,expected",
    [
        ("new", ["new"], [False, True]),
        ("old", ["old"], [True, False]),
        ("new", ["unplayed"], [False, False]),
        ("new", [], [False, False]),
        ("new", ["old", "new"], [True, True]),
    ],
)
async def test_truncation_belongs_to_item_first_playback_not_current_phase(
    tmp_path, played_id, truncated_ids, expected
):
    settings = Settings(_env_file=None, idle_end_ticks=1, turn_gap_ticks=0)
    scenario = {"scenario_id": "fixture", "role": "hotline"}
    current_dir = tmp_path / "current"
    legacy_dir = tmp_path / "legacy"
    current = await FullDuplexOrchestrator(settings, str(current_dir)).run_scenario(
        scenario, EventAgent(played_id, truncated_ids), Simulator(), max_turns=2
    )
    legacy = await FullDuplexOrchestrator(settings, str(legacy_dir)).run_scenario(
        scenario, EventAgent(played_id, truncated_ids, capable=False), Simulator(), max_turns=2
    )
    assert current.status == legacy.status == "ok"
    assert [trace.interrupted for trace in current.traces] == expected
    assert [trace.interrupted for trace in legacy.traces] == [True, False]
    # Existing audio, transcript, tick transitions and tool records stay identical.
    for name in ("conversation.wav", "both.wav"):
        assert (current_dir / name).read_bytes() == (legacy_dir / name).read_bytes()
    assert (current_dir / "ticks.json").read_bytes() == (legacy_dir / "ticks.json").read_bytes()
    for actual, before in zip(current.traces, legacy.traces, strict=True):
        assert actual.agent_text == before.agent_text
        assert actual.user_text == before.user_text
        assert (actual.tick_start, actual.user_end_tick, actual.tick_end) == (
            before.tick_start,
            before.user_end_tick,
            before.tick_end,
        )
        assert actual.tool_calls == before.tool_calls and actual.tool_outputs == before.tool_outputs
        for who in ("user", "agent"):
            assert (
                Path(getattr(actual, f"{who}_audio_path")).read_bytes()
                == Path(getattr(before, f"{who}_audio_path")).read_bytes()
            )


async def test_known_old_item_truncation_keeps_script_priority(tmp_path):
    class ScriptSimulator(Simulator):
        def peek_interrupt(self, *_args):
            return self.calls == 1

    settings = Settings(_env_file=None, idle_end_ticks=1, turn_gap_ticks=0, interrupt_delay_ticks=0)
    trajectory = await FullDuplexOrchestrator(settings, str(tmp_path)).run_scenario(
        {"scenario_id": "fixture", "role": "hotline"},
        EventAgent("old", ["old"]),
        ScriptSimulator(),
        max_turns=2,
    )
    assert trajectory.status == "ok"
    assert trajectory.traces[0].interrupted and trajectory.traces[0].interrupt_source == "script"
    assert not trajectory.traces[1].interrupted
