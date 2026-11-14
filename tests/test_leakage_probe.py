"""Offline regression checks for leakage probe."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from listen2serve.evaluation.leakage_probe import LeakageProbe, stance_marker_hits
from listen2serve.evaluation.validation import JudgeResponseError
from listen2serve.report.aggregate import aggregate_results


def _probe(payload: dict | str) -> LeakageProbe:
    llm = MagicMock()
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    llm.chat.return_value = MagicMock(text=text, prompt_tokens=100, finish_reason="stop")
    return LeakageProbe(llm=llm, llm_model="stub-judge-model")


def test_probe_parses_and_scores():
    p = _probe({"surface_score": 5, "state_guess": "displeased", "cue": "烦"})
    v = p.probe("你们别再打了，真烦人。", "先生这个产品很划算的", oracle_state="displeased")
    assert v.surface_score == 5
    assert v.state_guess == "displeased"
    assert v.state_correct is True
    assert v.cue == "烦"
    assert p.judge_calls == 1


def test_probe_neutral_wording_does_not_reveal_state():
    """Test probe neutral wording does not reveal state."""
    p = _probe({"surface_score": 1, "state_guess": "neutral", "cue": "无"})
    v = p.probe("我这号码在你们那儿登记成什么了？", "我们是根据登记信息联系您的", oracle_state="displeased")
    assert v.surface_score == 1
    assert v.state_correct is False


def test_probe_rejects_out_of_range_and_unknown_state():
    """Test probe rejects out of range and unknown state."""
    p = _probe({"surface_score": 9, "state_guess": "angry", "cue": "x"})
    with pytest.raises(JudgeResponseError):
        p.probe("随便一句", oracle_state="displeased")
    assert p.parse_retries == p.parse_failures == 1


def test_probe_unparsable_retries_then_reports_failure():
    p = _probe("这不是 JSON")
    with pytest.raises(JudgeResponseError):
        p.probe("随便一句")
    assert p.parse_retries == p.parse_failures == 1


def test_stance_marker_hits_is_deterministic():
    """Test stance marker hits is deterministic."""
    assert stance_marker_hits("我不需要，以后别再打了") == ["不需要", "别再"]
    assert stance_marker_hits("这个数字是按什么算出来的？") == []
    assert stance_marker_hits("") == []
    assert stance_marker_hits(None) == []


def test_aggregate_reads_leakage_metrics():
    verdicts = [
        {
            "role": "marketing",
            "leakage_level": "explicit",
            "dynamics": "all_negative",
            "leakage_score": 5,
            "leakage_state_correct": True,
            "leakage_stance_markers": 2,
        },
        {
            "role": "marketing",
            "leakage_level": "prosody_only",
            "dynamics": "all_negative",
            "leakage_score": 1,
            "leakage_state_correct": False,
            "leakage_stance_markers": 0,
        },
    ]
    agg = aggregate_results(verdicts)
    assert agg["summary"]["leakage_score_mean"] == 3.0
    assert agg["summary"]["leakage_state_correct_rate"] == 0.5
    assert agg["summary"]["leakage_stance_marker_mean"] == 1.0
    by_level = {r["wording"]: r for r in agg["rows"]}
    assert by_level["explicit"]["leakage_score_mean"] == 5.0
    assert by_level["neutral"]["leakage_score_mean"] == 1.0


def test_aggregate_old_verdicts_have_no_leakage_keys():
    """Test aggregate old verdicts have no leakage keys."""
    agg = aggregate_results(
        [
            {"role": "collection", "leakage_level": "explicit", "dynamics": "all_positive"},
        ]
    )
    assert agg["summary"]["leakage_score_mean"] is None
    assert agg["summary"]["leakage_state_correct_rate"] is None


def test_probe_prompt_hides_ground_truth():
    """Test probe prompt hides ground truth."""
    from listen2serve.evaluation.prompts import LEAKAGE_PROBE_PROMPT

    rendered = LEAKAGE_PROBE_PROMPT.format(agent_text="客服话", user_text="用户话")
    assert "客服话" in rendered and "用户话" in rendered
    for leaked in ("leakage_label", "T1", "T2", "T3", "key_event", "intensity"):
        assert leaked not in rendered
