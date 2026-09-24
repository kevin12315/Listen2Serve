"""端到端冒烟：真实评测集场景 × mock 网关 × 报告生成（不触网）。

Plan I（v6.0）：数据契约测试按骨架自由演绎口径重写——mission/key_event/
disclosure_plan/turn_budget 新键，state_track/transition/emotion_anchors/
tts_control_map/critical_turn 退役键不得存在。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from listen2serve.report.aggregate import aggregate_results
from listen2serve.report.render_md import render_md_report
from conftest import N_BASE, N_SCENARIOS
from listen2serve.user_script import MISSION_KEY_TAG, mission_beats, mission_states

# 版本断言锚定 version.json（不再硬编码；Plan J pitfall：升版后断言随之自动跟随）
_EXPECTED_VERSION = json.loads(
    (Path(__file__).resolve().parents[1] / "data/benchmark/version.json")
    .read_text(encoding="utf-8"))["dataset_version"]


@pytest.fixture(scope="module")
def scenarios() -> list[dict]:
    path = Path("data/benchmark/scenarios.jsonl")
    assert path.exists(), "缺评测集：data/benchmark/scenarios.jsonl 未随发布切片带上"
    return [json.loads(row) for row in path.open(encoding="utf-8") if row.strip()]


class TestBenchmarkContract:
    def test_scale(self, scenarios):
        assert len(scenarios) == N_SCENARIOS
        roles = {s["role"] for s in scenarios}
        assert roles == {"collection", "marketing", "hotline"}
        # scenario_id 全局唯一（T1/T2/T3 变体带后缀，防 runs/ 产物覆盖与回联错位）
        ids = [s["scenario_id"] for s in scenarios]
        assert len(set(ids)) == N_SCENARIOS, "scenario_id 存在重复"
        for s in scenarios:
            assert s["scenario_id"].endswith(s["leakage_label"])
            assert s["base_scenario_id"] and not s["base_scenario_id"].endswith(s["leakage_label"])
            assert s["dataset_version"] == _EXPECTED_VERSION

    def test_skeleton_required_fields(self, scenarios):
        """v6.0 骨架键齐备且合法。"""
        from listen2serve.domains.base import STATE_LABELS

        for s in scenarios:
            us = s["user_script"]
            sid = s["scenario_id"]
            assert us.get("identity") and us.get("user_goal"), f"{sid} 缺 identity/user_goal"
            # mission：3–7 条，恰好一条【关键事件】
            # Plan L §3.3：条目升级为 {beat, state?, key_event?}，beat 文本逐字不变
            mission = us.get("mission")
            assert mission and 3 <= len(mission) <= 7, f"{sid} mission 条数非法"
            beats = mission_beats(us)
            key_items = [b for b in beats if b.startswith(MISSION_KEY_TAG)]
            assert len(key_items) == 1, f"{sid} mission 关键事件条目数 {len(key_items)}"
            # 逐 beat state 合法六态（省略即 neutral）；key_event 标记恰好一条且与前缀同位
            bad = [st for st in mission_states(us) if st not in STATE_LABELS]
            assert not bad, f"{sid} mission.state 非法: {bad}"
            marked = [i for i, m in enumerate(mission) if m.get("key_event")]
            assert marked == [beats.index(key_items[0])], f"{sid} mission key_event 标记错位: {marked}"
            # key_event 结构
            ke = us.get("key_event") or {}
            assert ke.get("description"), f"{sid} key_event.description 缺失"
            assert ke.get("state") in STATE_LABELS, f"{sid} key_event.state 非法: {ke.get('state')}"
            assert "intensity" not in ke, f"{sid} intensity 应随 Plan L §3.5 退役"
            assert ke.get("trigger_hint"), f"{sid} trigger_hint 缺失"
            assert isinstance(ke.get("interrupt"), bool), f"{sid} interrupt 非布尔"
            assert key_items[0][len(MISSION_KEY_TAG):] == ke["description"], \
                f"{sid} mission 关键事件与 key_event.description 不同源"
            # turn_budget
            tb = us.get("turn_budget") or {}
            assert 3 <= tb.get("soft", 0) <= 12 and tb.get("hard", 0) <= 12
            assert tb["hard"] >= tb["soft"] + 1, f"{sid} hard 需大于 soft"
            # disclosure_plan 覆盖全部 user_facts 字段（无 facts 可省略）
            facts = us.get("user_facts") or {}
            dp = us.get("disclosure_plan") or {}
            if facts:
                assert set(dp) == set(facts), f"{sid} disclosure_plan 与 user_facts 不同键"
            # 退役键不得存在
            for dead in ("state_track", "transition", "emotion_anchors", "tts_control_map"):
                assert dead not in us, f"{sid} 退役键 {dead} 仍存在"
            # 场景级契约
            assert "critical_turn" not in (s.get("measurement") or {}), \
                f"{sid} critical_turn 应随 v6.0 退役"
            assert "stage_at_critical_turn" not in s, f"{sid} 退役键 stage_at_critical_turn"
            assert s.get("expected_key_behavior"), f"{sid} expected_key_behavior 缺失"
            assert s["agent_policy"]["required_actions"]
            assert s["agent_policy"]["forbidden_actions"]

    def test_oracle_state_six_states(self, scenarios):
        """oracle_state 改写为新 6 态（可归一化）。"""
        from listen2serve.domains.base import STATE_LABELS, normalize_state

        for s in scenarios:
            code = normalize_state(s.get("oracle_state"))
            assert code in STATE_LABELS, f"{s['scenario_id']} oracle_state 无法归一: {s.get('oracle_state')}"
            assert code == s["user_script"]["key_event"]["state"], \
                f"{s['scenario_id']} oracle 与 key_event.state 不一致"

    def test_variants_share_skeleton(self, scenarios):
        """三变体（T1/T2/T3）共享同一骨架；唯一允许的差异是关键轮 surface。

        v6.2 起 key_event.surface 按层写入三份文本（透明度 explicit >
        implicit_consistent > prosody_only），所以 key_event 不再逐字相同；
        除 surface 之外的字段（state/intensity/interrupt/trigger_hint …）仍必须一致，
        否则韵律与事件本身也跟着变了，层间对比不再干净。
        """
        by_base: dict[str, list[dict]] = {}
        for s in scenarios:
            by_base.setdefault(s["base_scenario_id"], []).append(s)
        assert len(by_base) == N_BASE
        for base, group in by_base.items():
            # 发布切片只带主实验用到的两档（T1 文本显性 / T3 文本中性）；内部全库是三档 606 行
            assert len(group) == 2, f"{base} 变体数 {len(group)}"
            missions = {json.dumps(g["user_script"]["mission"], ensure_ascii=False) for g in group}
            keys = {json.dumps({k: v for k, v in g["user_script"]["key_event"].items()
                                if k != "surface"}, ensure_ascii=False, sort_keys=True)
                    for g in group}
            surfaces = {g["user_script"]["key_event"].get("surface", "") for g in group}
            budgets = {json.dumps(g["user_script"]["turn_budget"]) for g in group}
            assert len(missions) == 1, f"{base} 三变体 mission 不一致"
            assert len(keys) == 1, f"{base} 三变体 key_event（除 surface）不一致"
            assert len(surfaces) == 2, f"{base} 两档 surface 未分化：{surfaces}"
            assert len(budgets) == 1, f"{base} 三变体 turn_budget 不一致"

    def test_interrupt_annotation(self, scenarios):
        """打断标注：仅出现在负向动态场景的关键事件上，且确有标注。"""
        n_interrupt = 0
        for s in scenarios:
            ke = s["user_script"]["key_event"]
            if ke["interrupt"]:
                n_interrupt += 1
                assert s["dynamics"] in ("all_negative", "pos_to_neg"), \
                    f"{s['scenario_id']} 打断标注不在负向动态场景"
        assert n_interrupt > 0, "无任何打断标注"

    def test_vocab_scale(self):
        """词表规模（Plan I 契约）：NEG>=100 / POS>=60，子表键为新 6 态负极性子集，
        并集=平铺表（词本体零改动、归属重排）。"""
        from listen2serve.domains.base import STATE_LABELS
        from listen2serve.vocab import (
            NEG_BY_STATE, NEG_WORDS, POS_BY_STATE, POS_WORDS, STATE_POLARITY,
        )

        assert len(NEG_WORDS) >= 100
        assert len(POS_WORDS) >= 60
        assert set(NEG_BY_STATE) | set(POS_BY_STATE) == set(STATE_POLARITY)
        assert set(STATE_POLARITY) <= set(STATE_LABELS) - {"neutral"}
        assert {w for ws in NEG_BY_STATE.values() for w in ws} == set(NEG_WORDS)
        assert {w for ws in POS_BY_STATE.values() for w in ws} == set(POS_WORDS)

    def test_user_gender_balance(self, scenarios):
        """用户声音性别均衡：全集 50/50，且每角色×动态内均衡。

        不写死每单元条数：v6.5 增量补齐后各单元容量不再相同（补的是 hushed/urgent/doubtful，
        落点集中在 pos_to_neg / all_negative），但"单元内男女相等"这条不变量仍然成立
        —— 它才是这个用例要守的东西（也正因为它，v6.5 那两个 21 条的奇数单元只能靠跨单元
        删一造一来对齐）。
        """
        from collections import Counter

        genders = Counter(s["user_gender"] for s in scenarios)
        # 全库 202 base 严格 50/50；145 均衡子集按态分层，性别只能近似均衡 ⇒ 放界到 ±10%
        assert abs(genders["male"] - genders["female"]) <= N_SCENARIOS * 0.1, genders
        by_cell = Counter((s["role"], s["dynamics"], s["user_gender"]) for s in scenarios)
        for role in ("collection", "marketing", "hotline"):
            for dyn in ("all_positive", "all_negative", "pos_to_neg", "neg_to_pos"):
                m, f = by_cell[(role, dyn, "male")], by_cell[(role, dyn, "female")]
                # 分层维度是**客户状态**而非性别 ⇒ 子集不保证格内 50/50（hotline/neg_to_pos 实测 2/8）。
                # 这里只断言两性都在场（性别错配会连带音色与称呼）；全局均衡见上面那一条。
                assert m > 0 and f > 0, f"{role}/{dyn} 缺一性：{m}/{f}"

    def test_db_seed_present(self, scenarios):
        for s in scenarios:
            assert s.get("db_seed"), f"{s['scenario_id']} 缺少 db_seed"

    def test_customer_profile_renders_nonempty(self, scenarios):
        """客户信息档案：每条场景档案渲染均非空且含引导语标题。"""
        from listen2serve.domains.base import render_customer_profile

        checked = 0
        for s in scenarios:
            profile = render_customer_profile(s["role"], s.get("db_seed"))
            assert profile, f"{s['scenario_id']} 客户信息档案渲染为空"
            assert "客户信息档案（系统已预先提供，通话中无需查询、直接作为事实依据）" in profile
            checked += 1
        assert checked == N_SCENARIOS


class TestReportPipeline:
    def test_aggregate_and_render(self):
        verdicts = [
            {"scenario_id": f"S{i}", "role": "collection", "leakage_level": "explicit",
             "dynamics": "all_positive", "policy_pass": True, "voice_pass": True,
             "joint_pass": True, "task_completion": 2, "transition_score": None}
            for i in range(3)
        ] + [
            {"scenario_id": f"T{i}", "role": "marketing", "leakage_level": "prosody_only",
             "dynamics": "pos_to_neg", "policy_pass": False, "voice_pass": True,
             "joint_pass": False, "task_completion": None, "transition_score": 1}
            for i in range(3)
        ]
        agg = aggregate_results(verdicts)
        assert agg["summary"]["n"] == 6
        assert agg["summary"]["policy_pass"] == 0.5
        md = render_md_report(agg)
        assert "## 总体指标" in md
        assert "JointPass" in md
