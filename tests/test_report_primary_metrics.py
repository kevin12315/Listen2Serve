"""report 层主指标口径测试。

Plan J（v2.3-J）：对外主指标 = KeyTurnPass/FlowRate/TaskScore/VoiceMean，量纲统一
（文本 0-1 / 声音 MOS 1-5）；PolicyPass/RequiredRate 等退役名仅作旧 verdict 兼容渲染。
"""

from __future__ import annotations

from listen2serve.report.aggregate import (
    aggregate_cross_model,
    aggregate_marginals,
    aggregate_results,
)
from listen2serve.report.render_md import render_md_report


def _verdict(sid: str, req_rate: float | None, voice: dict | None,
             policy_pass: bool = True, voice_pass: bool = True) -> dict:
    """旧 schema verdict（v2.2-I）：无 v2.3-J 增量键。"""
    return {
        "scenario_id": sid, "role": "collection", "leakage_level": "explicit",
        "dynamics": "all_positive", "policy_pass": policy_pass, "voice_pass": voice_pass,
        "joint_pass": policy_pass and voice_pass, "task_completion": 2, "transition_score": 1,
        "policy_detail": {"required_rate": req_rate} if req_rate is not None else {},
        "voice_detail": voice or {},
    }


def _verdict_v23(sid: str, key_turn_pass: bool, flow_rate: float, task_score: float,
                 key_behavior_met: str = "是") -> dict:
    """新 schema verdict（v2.3-J）：退役名置 None。"""
    return {
        "scenario_id": sid, "role": "collection", "leakage_level": "explicit",
        "dynamics": "all_positive",
        "policy_pass": None, "policy_detail": None, "task_completion": None, "transition_score": None,
        "key_turn_pass": key_turn_pass, "key_behavior_met": key_behavior_met,
        "flow_rate": flow_rate, "task_score": task_score,
        "voice_detail": {},
    }


class TestPrimaryMetrics:
    def test_required_rate_mean_macro_legacy(self):
        """旧口径兼容：逐场景 required_rate 宏平均（明细口径）。"""
        verdicts = [_verdict("A", 1.0, {"mean": 4.0}), _verdict("B", 0.5, {"mean": 3.0})]
        agg = aggregate_results(verdicts)
        assert agg["summary"]["required_rate_mean"] == 0.75
        assert agg["rows"][0]["required_rate_mean"] == 0.75

    def test_voice_mean_and_dims(self):
        """VoiceMean 与三维单项均分。"""
        v1 = {"role_voice_match": 5, "state_voice_fit": 4, "naturalness": 3, "mean": 4.0}
        v2 = {"role_voice_match": 3, "state_voice_fit": 2, "naturalness": 4, "mean": 3.0}
        agg = aggregate_results([_verdict("A", 0.8, v1), _verdict("B", 0.6, v2)])
        s = agg["summary"]
        assert s["voice_mean"] == 3.5
        assert s["voice_role_match_mean"] == 4.0
        assert s["voice_state_fit_mean"] == 3.0
        assert s["voice_naturalness_mean"] == 3.5

    def test_missing_detail_renders_na(self):
        """旧 run 无 policy_detail.required_rate / voice_detail 时主指标为 None，渲染 N/A。"""
        agg = aggregate_results([_verdict("A", None, None)])
        assert agg["summary"]["required_rate_mean"] is None
        assert agg["summary"]["voice_mean"] is None
        md = render_md_report(agg)
        assert "N/A" in md

    def test_binary_metrics_kept_as_secondary(self):
        """PolicyPass/VoicePass 旧键聚合行为不变（旧 verdict 兼容）。"""
        agg = aggregate_results([_verdict("A", 1.0, {"mean": 5.0}, True, True),
                                 _verdict("B", 0.5, {"mean": 2.0}, False, False)])
        assert agg["summary"]["policy_pass"] == 0.5
        assert agg["summary"]["voice_pass"] == 0.5

    def test_render_primary_first(self):
        """渲染顺序：主指标块先于退役指标块（旧 verdict 时退役块存在）。"""
        agg = aggregate_results([_verdict("A", 1.0, {"role_voice_match": 5,
                                                      "state_voice_fit": 5,
                                                      "naturalness": 5, "mean": 5.0})])
        md = render_md_report(agg)
        assert md.index("主指标（v2.3-J") < md.index("退役指标")
        assert "KeyTurnPass" in md.split("## 分层结果")[1]  # 分层表含新主指标列


class TestV23JMetrics:
    """Plan J v2.3-J：新主指标聚合与渲染。"""

    def test_new_metrics_aggregate(self):
        agg = aggregate_results([
            _verdict_v23("A", True, 1.0, 1.0),
            _verdict_v23("B", False, 0.5, 0.5, key_behavior_met="部分"),
        ])
        s = agg["summary"]
        assert s["key_turn_pass"] == 0.5
        assert s["key_behavior_met_rate"] == 1.0  # 是/部分 均计命中
        assert s["flow_rate_mean"] == 0.75
        assert s["task_score_mean"] == 0.75

    def test_new_metrics_none_on_legacy_verdicts(self):
        """旧 verdict 无增量键：新主指标一律 None（渲染 N/A），不假性记 0。"""
        agg = aggregate_results([_verdict("A", 1.0, {"mean": 4.0})])
        s = agg["summary"]
        assert s["key_turn_pass"] is None
        assert s["key_behavior_met_rate"] is None
        assert s["flow_rate_mean"] is None
        assert s["task_score_mean"] is None
        md = render_md_report(agg)
        assert "退役指标" in md  # 旧 verdict 渲染退役对照块

    def test_v23_render_no_legacy_block(self):
        """纯 v2.3-J verdict：报告不渲染退役指标块。"""
        md = render_md_report(aggregate_results([_verdict_v23("A", True, 1.0, 1.0)]))
        assert "退役指标" not in md
        assert "KeyTurnPass" in md and "FlowRate" in md and "TaskScore" in md

    def test_retired_names_none_not_false_zero(self):
        """Plan L §0.17d：退役名在新批次必须聚成 None，不得兜底成假零 0.0。

        假零的危险在于旧批次（v2.2-I）的真值本就贴近零（实测 PolicyPass 0.0556），
        横向对照页的 0.0% 无法告诉读者「未测量」还是「几乎全不通过」。
        """
        s = aggregate_results([_verdict_v23("A", True, 1.0, 1.0),
                              _verdict_v23("B", False, 0.5, 0.5)])["summary"]
        for key in ("policy_pass", "voice_pass", "joint_pass",
                    "task_completion_mean", "transition_score_mean"):
            assert s[key] is None, f"{key} 假零: {s[key]!r}"

    def test_retired_names_keep_legacy_values(self):
        """反向担保：旧 verdict 有真值时，退役名仍照旧口径算出数（不被误伤成 None）。"""
        s = aggregate_results([_verdict("A", 1.0, {"mean": 5.0}, True, True),
                              _verdict("B", 0.5, {"mean": 2.0}, False, False)])["summary"]
        assert s["policy_pass"] == 0.5 and s["voice_pass"] == 0.5 and s["joint_pass"] == 0.5
        assert s["task_completion_mean"] == 2.0 and s["transition_score_mean"] == 1.0


class TestCrossModel:
    """跨模型对照（n5）：同一份数据在多个被测模型上的横向切分与配对差值。"""

    @staticmethod
    def _v(sid, model, kt, level, label, leak):
        return {"scenario_id": sid, "model": model, "role": "collection",
                "dynamics": "stable", "leakage_level": level, "leakage_label": label,
                "key_turn_score": kt, "key_turn_pass": kt >= 3.0,
                "flow_rate": 0.5, "task_score": 0.6, "leakage_score": leak}

    def _pair_set(self):
        """两模型 × 三层同数据；另加一条只有 A 侧有判定的场景（应被配对排除）。"""
        vs = []
        for i, (lv, lb, ls) in enumerate([("explicit", "T1", 4.5),
                                          ("implicit_consistent", "T2", 3.0),
                                          ("prosody_only", "T3", 1.5)]):
            vs.append(self._v(f"S{i}", "A/plus", 3.0, lv, lb, ls))
            vs.append(self._v(f"S{i}", "B/flash", 4.0, lv, lb, ls))
        vs.append(self._v("S9", "A/plus", 2.0, "explicit", "T1", 4.0))
        return vs

    def test_marginals_include_by_model(self):
        m = aggregate_marginals(self._pair_set())
        assert [r["group"] for r in m["by_model"]] == ["A/plus", "B/flash"]

    def test_paired_excludes_single_side(self):
        """单侧缺判定的场景不进配对——否则差值混进的是「谁失败得多」。"""
        cm = aggregate_cross_model(self._pair_set())
        assert cm["models"] == ["A/plus", "B/flash"] and cm["baseline"] == "A/plus"
        row = cm["paired"][0]
        assert row["n_paired"] == 3          # S9 只有 A 侧，排除
        assert row["d_key_turn_score"] == 1.0
        assert (row["win"], row["tie"], row["loss"]) == (3, 0, 0)

    def test_monotonic_computed_per_model(self):
        cm = aggregate_cross_model(self._pair_set())
        assert cm["monotonic_by_model"] == {"A/plus": True, "B/flash": True}
        assert len(cm["by_model_leakage"]) == 6   # 2 模型 × 3 层

    def test_empty_verdicts_safe(self):
        cm = aggregate_cross_model([])
        assert cm["models"] == [] and cm["paired"] == [] and cm["baseline"] == ""
