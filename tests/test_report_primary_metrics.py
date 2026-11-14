"""The public report shows all four multi-turn primary metrics."""

from listen2serve.report.aggregate import aggregate_cross_model, aggregate_marginals, aggregate_results
from listen2serve.report.render_md import render_md_report


def verdict(sid="A", passed=True, wording="explicit", model="model-a", score=4):
    return {
        "scenario_id": sid,
        "role": "collection",
        "wording": wording,
        "dynamics": "all_positive",
        "model": model,
        "key_turn_score": score,
        "key_turn_pass": passed,
        "key_behavior_met": "是",
        "flow_rate": 1.0 if passed else 0.5,
        "task_score": 1.0 if passed else 0.5,
        "voice_pass": passed,
        "voice_detail": {"role_voice_match": 4, "state_voice_fit": 4, "naturalness": 4, "mean": 4.0},
    }


def test_primary_metrics_aggregate_with_unchanged_formulas():
    summary = aggregate_results([verdict(), verdict("B", False)])["summary"]
    assert summary["key_turn_pass"] == summary["voice_pass"] == 0.5
    assert summary["flow_rate_mean"] == summary["task_score_mean"] == 0.75
    assert summary["voice_mean"] == 4.0


def test_voice_pass_always_renders_in_primary_and_stratified_tables():
    md = render_md_report(aggregate_results([verdict()]))
    overall, grouped = md.split("## 分层结果")
    assert "VoicePass" in overall and "VoicePass" in grouped
    assert "100.0%" in overall
    for name in ("KeyTurnPass", "VoicePass", "FlowRate", "TaskScore"):
        assert name in overall and name in grouped
    for removed in ("退役指标", "v2.3", "T1", "T3", "PolicyPass", "JointPass"):
        assert removed not in md


def test_missing_voice_is_na_and_never_fabricated_as_zero():
    v = verdict()
    v["voice_pass"], v["voice_detail"] = None, None
    agg = aggregate_results([v])
    assert agg["summary"]["voice_pass"] is None
    assert agg["summary"]["voice_mean"] is None
    assert "N/A" in render_md_report(agg)


def test_report_exposes_paper_wording_for_immutable_source_labels():
    v = verdict()
    del v["wording"]
    v.update(leakage_level="prosody_only", leakage_label="T3")
    agg = aggregate_results([v])
    assert agg["rows"][0]["wording"] == "neutral"
    assert "prosody_only" not in render_md_report(agg)


def test_cross_model_and_marginal_summaries_keep_pairing():
    rows = [
        verdict("A"),
        verdict("B", wording="neutral"),
        verdict("A", model="model-b", score=5),
        verdict("B", wording="neutral", model="model-b", score=5),
        verdict("unpaired"),
    ]
    marginal = aggregate_marginals(rows)
    assert [r["group"] for r in marginal["by_wording"]] == ["explicit", "neutral"]
    cross = aggregate_cross_model(rows)
    assert cross["paired"][0]["n_paired"] == 2
    assert cross["paired"][0]["d_key_turn_score"] == 1.0
    assert cross["paired"][0]["win"] == 2


def test_empty_aggregation_is_safe():
    assert aggregate_results([])["summary"]["voice_pass"] is None
    assert aggregate_cross_model([])["models"] == []


def test_report_provenance_uses_its_own_judges_after_a_different_reevaluation():
    from listen2serve.cli import _provenance_summary

    manifest = {
        "models": {"target": "target", "judge": None},
        "evaluation": {"keyturn_model_observed": "google/new-judge", "voice_judge_model": "openai/new-audio"},
    }
    rows = [
        {
            "judge_meta": {
                "keyturn": {"model": "google/original-judge"},
                "voice": {"model": "openai/original-audio"},
            },
            "voice_detail": {"mean": 4},
        }
    ]
    summary = _provenance_summary("run", manifest, rows)
    assert summary["models"]["judge"] == "google/original-judge"
    assert summary["models"]["voice_judge"] == "openai/original-audio"


def test_existing_prosody_condition_is_preserved_in_report_groups():
    first, second = verdict(), verdict("B")
    first.update(wording="neutral", condition="N/E")
    second.update(wording="neutral", condition="N/N")
    agg = aggregate_results([first, second])
    assert len(agg["rows"]) == 2
    md = render_md_report(agg)
    assert "N/E" in md and "N/N" in md


def test_partial_calls_keep_key_and_voice_but_never_enter_whole_call_metrics():
    complete, partial = verdict(), verdict("partial", False)
    partial.update(trajectory_status="partial", termination_reason="endpoint_session_closed",
                   flow_rate=0.0, task_score=0.0, closing_ok=False)
    complete["closing_ok"] = True
    agg = aggregate_results([complete, partial])
    summary = agg["summary"]
    assert summary["n_complete"] == 1 and summary["n_partial"] == 1
    assert summary["key_turn_n"] == summary["voice_n"] == 2
    assert summary["flow_n"] == summary["task_n"] == 1
    assert summary["key_turn_pass"] == summary["voice_pass"] == 0.5
    assert summary["flow_rate_mean"] == summary["task_score_mean"] == summary["closing_ok_rate"] == 1
    md = render_md_report(agg)
    assert "完整通话：1" in md and "端点截断、关键轮可评：1" in md
    assert "FlowRate=1" in md and "TaskScore=1" in md


def test_legacy_ok_session_closure_cannot_contaminate_flow_or_paired_scores():
    complete, legacy = verdict("A"), verdict("A", model="model-b", score=5)
    legacy.update(status="ok", termination_reason="endpoint_session_closed")
    summary = aggregate_results([legacy])["summary"]
    assert summary["n_partial"] == 1
    assert summary["flow_rate_mean"] is None and summary["task_score_mean"] is None
    assert summary["key_turn_pass"] == 1 and summary["voice_pass"] == 1
    paired = aggregate_cross_model([complete, legacy])["paired"][0]
    assert paired["n_key_turn_score"] == 1 and paired["d_key_turn_score"] == 1
    assert paired["n_flow_rate"] == paired["n_task_score"] == 0
