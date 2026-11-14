"""Legacy response timeouts cannot leak fragments into paid evaluation stages."""

from __future__ import annotations

import json

import pytest
from test_evaluate_incremental import _evaluation_cli_fixture

from listen2serve.cli import _build_eval_tasks, _evaluate
from listen2serve.evaluation.incremental import dialogue_is_complete
from listen2serve.report.aggregate import aggregate_results


def timed_trajectory(tmp_path, monkeypatch, fault_turn=None, fault="timed_out"):
    args, manifest, asr_calls, judge_calls = _evaluation_cli_fixture(tmp_path, monkeypatch)
    status = tmp_path / "runs/case/case/sim_status.json"
    trajectory = json.loads(status.read_text())
    original = trajectory["turns"][0]
    trajectory["termination_reason"] = "user_closed"
    trajectory["turns"] = [
        {**original, "turn": turn, "is_key_turn": turn == 2, fault: turn == fault_turn} for turn in (1, 2, 3)
    ]
    trajectory["turns"][2]["agent_text"] = "超时后的残片不可进入完整通话评测"
    status.write_text(json.dumps(trajectory))
    return args, manifest, asr_calls, judge_calls, status, trajectory


@pytest.mark.parametrize(
    "turn,fault,message",
    [
        (1, "timed_out", "关键轮之前存在响应超时"),
        (2, "timed_out", "关键轮客服回复不完整"),
        (2, "endpoint_interrupted", "关键轮客服回复不完整"),
    ],
)
def test_bad_stream_before_or_at_key_is_rejected_before_asr_or_judges(
    tmp_path, monkeypatch, turn, fault, message
):
    args, manifest, asr_calls, judge_calls, status, _ = timed_trajectory(tmp_path, monkeypatch, turn, fault)
    original_bytes = status.read_bytes()
    with pytest.raises(ValueError, match=message):
        _evaluate(args)
    assert asr_calls == [] and judge_calls == []
    assert not (tmp_path / "reports/case/verdicts_audio.json").exists()
    assert "evaluation" not in json.loads(manifest.read_text())
    assert status.read_bytes() == original_bytes


def test_legacy_ok_timeout_after_complete_key_only_scores_the_valid_key(tmp_path, monkeypatch):
    from listen2serve.evaluation.flow_judge import FlowJudge

    args, _, asr_calls, judge_calls, status, trajectory = timed_trajectory(tmp_path, monkeypatch, 3)
    original_bytes = status.read_bytes()
    monkeypatch.setattr(FlowJudge, "judge", lambda *a, **k: pytest.fail("Whole-call judge must not run"))
    assert _evaluate(args) == 0
    assert len(asr_calls) == 2  # Only the valid prefix through the complete critical reply.
    assert len(judge_calls) == 3  # Key, voice and auxiliary probe, without whole-call scoring.
    verdict = json.loads((tmp_path / "reports/case/verdicts_audio.json").read_text())[0]
    assert verdict["trajectory_status"] == "partial" and verdict["dialogue_complete"] is False
    assert verdict["key_turn_pass"] is True and verdict["voice_pass"] is True
    assert verdict["flow_items"] is None and verdict["flow_detail"] is None
    assert verdict["flow_rate"] is None and verdict["task_score"] is None
    assert verdict["closing_signal"] is None and verdict["closing_ok"] is None
    assert not dialogue_is_complete(trajectory)
    summary = aggregate_results([verdict])["summary"]
    assert summary["n_complete"] == 0 and summary["n_partial"] == 1
    assert summary["key_turn_n"] == summary["voice_n"] == 1
    assert summary["flow_n"] == summary["task_n"] == 0
    assert summary["flow_rate_mean"] is None and summary["task_score_mean"] is None
    assert status.read_bytes() == original_bytes


def test_legacy_post_key_timeout_history_excludes_the_failed_suffix(tmp_path, monkeypatch):
    _, _, _, _, status, _ = timed_trajectory(tmp_path, monkeypatch, 3)
    scenario = json.loads((tmp_path / "scenarios.jsonl").read_text())
    task = _build_eval_tasks(
        status.parents[1], {"case": scenario}, "google/gemini-3.8-flash", "google/gemini-3.8-flash"
    )[0]
    assert len(task.payload.history_with_agent) == 4
    assert len(task.payload.history_before_critical) == 3
    assert all("超时后的残片" not in item["content"] for item in task.payload.history_with_agent)


def test_normal_complete_dialogue_keeps_full_asr_flow_and_task_scoring(tmp_path, monkeypatch):
    args, _, asr_calls, judge_calls, _, trajectory = timed_trajectory(tmp_path, monkeypatch)
    assert _evaluate(args) == 0
    assert len(asr_calls) == 3 and len(judge_calls) == 4
    verdict = json.loads((tmp_path / "reports/case/verdicts_audio.json").read_text())[0]
    assert verdict["dialogue_complete"] is True and dialogue_is_complete(trajectory)
    assert verdict["flow_rate"] == verdict["task_score"] == 1


def test_scripted_interruption_without_response_timeout_remains_complete():
    assert dialogue_is_complete({"status": "ok", "turns": [{"timed_out": False, "interrupted": True}]})
