"""A timed-out response must never spill into a later user turn."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from listen2serve.runtime.agent import AgentTickResult
from listen2serve.runtime.config import Settings
from listen2serve.runtime.orchestrator import FullDuplexOrchestrator, Trajectory, TurnTrace
from listen2serve.runtime.user_simulator import UserTurnResult


class ScriptSimulator:
    def __init__(self, key_turn):
        self.key_turn = key_turn
        self.calls = 0

    def next_turn(self, *_args):
        self.calls += 1
        return UserTurnResult(
            text=f"用户第{self.calls}轮。",
            pcm16=b"\x01\x00" * 3200,
            is_key_turn=self.calls == self.key_turn,
        )

    def peek_interrupt(self, *_args):
        return False


class TimeoutAgent:
    model_spec = "fake/timeout"

    def __init__(self, timeout_turn, *, cancel_fails=False, has_fragment=True):
        self.timeout_turn = timeout_turn
        self.cancel_fails = cancel_fails
        self.has_fragment = has_fragment
        self.turn = 0
        self.response_ticks = 0
        self.cancelled = False
        self.stopped = False
        self.reads_after_cancel = 0

    async def start(self):
        pass

    async def stop(self):
        self.stopped = True

    async def cancel_response(self):
        self.cancelled = True
        if self.cancel_fails:
            raise RuntimeError("fixture cancellation failure")

    async def run_tick(self, chunk, tick=0):
        if self.cancelled:
            self.reads_after_cancel += 1
            return AgentTickResult(transcript="节，并在下一轮继续。", audio=b"\x02\x00" * 4800)
        if any(chunk):
            self.turn += 1
            self.response_ticks = 0
            return AgentTickResult()
        self.response_ticks += 1
        if self.turn == self.timeout_turn:
            if not self.has_fragment:
                return AgentTickResult()
            return AgentTickResult(
                transcript="故障细" if self.response_ticks == 1 else "",
                audio=b"\x02\x00" * 4800,
                buffer_bytes=4800,
            )
        if self.response_ticks == 1:
            return AgentTickResult(transcript="已登记。", audio=b"\x02\x00" * 4800)
        return AgentTickResult()


@pytest.mark.parametrize(
    "timeout_turn,key_turn,status,key_missing",
    [(1, 2, "error", True), (2, 2, "error", False), (3, 2, "partial", False)],
)
@pytest.mark.parametrize("cancel_fails", [False, True])
async def test_timeout_stops_session_and_preserves_fragment(
    tmp_path, timeout_turn, key_turn, status, key_missing, cancel_fails
):
    settings = Settings(
        _env_file=None,
        max_response_ticks=4,
        idle_end_ticks=2,
        turn_gap_ticks=0,
    )
    simulator = ScriptSimulator(key_turn)
    agent = TimeoutAgent(timeout_turn, cancel_fails=cancel_fails)
    trajectory = await FullDuplexOrchestrator(settings, str(tmp_path)).run_scenario(
        {"scenario_id": "fixture", "role": "hotline"}, agent, simulator
    )
    assert trajectory.status == status
    assert trajectory.termination_reason == "response_timeout"
    assert trajectory.key_turn_missing is key_missing
    assert simulator.calls == timeout_turn
    assert len(trajectory.traces) == timeout_turn
    final = trajectory.traces[-1]
    assert final.timed_out and final.agent_text == "故障细"
    assert final.agent_audio_pcm24_len > 0
    assert final.tick_end - final.user_end_tick == settings.max_response_ticks
    assert Path(final.agent_audio_path).stat().st_size > 44
    assert agent.cancelled and agent.stopped and not agent.reads_after_cancel
    ticks = json.loads((tmp_path / "ticks.json").read_text())
    assert max(t["turn"] for t in ticks) == timeout_turn
    assert max(t["tick"] for t in ticks) == final.tick_end
    assert "下一轮继续" not in "".join(t.agent_text for t in trajectory.traces)


async def test_zero_output_timeout_keeps_current_turn_evidence(tmp_path):
    settings = Settings(_env_file=None, max_response_ticks=4, idle_end_ticks=2)
    agent = TimeoutAgent(1, has_fragment=False)
    trajectory = await FullDuplexOrchestrator(settings, str(tmp_path)).run_scenario(
        {"scenario_id": "fixture", "role": "hotline"}, agent, ScriptSimulator(1)
    )
    assert trajectory.status == "error" and trajectory.termination_reason == "response_timeout"
    assert not trajectory.key_turn_missing
    assert len(trajectory.traces) == 1 and trajectory.traces[0].timed_out
    assert Path(trajectory.traces[0].agent_audio_path).exists()
    assert agent.cancelled and agent.stopped


@pytest.mark.parametrize(
    "timeout_turn,key_turn,status", [(1, 2, "error"), (2, 2, "error"), (3, 2, "partial")]
)
def test_legacy_ok_timeout_is_classified_in_memory(timeout_turn, key_turn, status):
    trajectory = Trajectory(scenario_id="legacy", model="fixture", role="hotline")
    trajectory.traces = [
        TurnTrace(
            turn=turn,
            user_state="neutral",
            user_text="用户文本。",
            agent_text="故障细" if turn == timeout_turn else "已登记。",
            is_key_turn=turn == key_turn,
            timed_out=turn == timeout_turn,
        )
        for turn in range(1, 5)
    ]
    original = trajectory.as_dict()
    loaded = Trajectory.from_dict(original)
    assert loaded.status == status and loaded.termination_reason == "response_timeout"
    assert original["status"] == "ok" and original["termination_reason"] == ""
