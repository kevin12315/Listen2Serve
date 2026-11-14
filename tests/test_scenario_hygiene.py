"§5：演绎侧数据卫生 CI —— 七类缺陷 + 三条新增断言，任一命中即失败"

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import N_BASE, N_SCENARIOS

from listen2serve.data import hygiene as H

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "benchmark" / "scenarios_base.jsonl"

_PRE_FIX_HITS = {"F1": 1, "F2": 6, "F3": 24, "F4": 59, "F5": 288, "F6": 85, "F7": 159}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    path = ROOT / "data" / "benchmark" / "scenarios.jsonl"
    assert path.exists(), "评测集缺失"
    return [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]


@pytest.fixture(scope="module")
def base_rows() -> list[dict]:
    assert BASE_PATH.exists(), "基础骨架缺失"
    return [json.loads(ln) for ln in BASE_PATH.open(encoding="utf-8") if ln.strip()]


@pytest.fixture(scope="module")
def flags(rows) -> dict[str, list[str]]:
    return H.audit_rows(rows)


def _t1t3(rows) -> list[dict]:
    return [r for r in rows if r.get("leakage_label") in H.REPAIRED_LAYERS]


class TestSevenDefects:
    "断言 1–7：F1–F7 各自命中数（ §5 表的前七行）"

    def test_f1_surface_no_identity_name(self, flags):
        """F1：surface 里不许出现用户本人姓名（改前 1 行 / 1 base）。"""
        assert flags["F1"] == [], f"F1 命中 {flags['F1']}"

    def test_f2_third_party_has_debtor(self, flags):
        """F2：自称第三方时 db_seed 首行不能就是接电话的人（改前 6 行 / 2 base）。"""
        assert flags["F2"] == [], f"F2 命中 {flags['F2']}"

    def test_f3_surface_not_written_style(self, flags, rows):
        "Test f3 surface not written style."
        scope = {r["scenario_id"] for r in _t1t3(rows)}
        hit = [s for s in flags["F3"] if s in scope]
        assert hit == [], f"F3 在 T1/T3 命中 {hit}"

    def test_f4_no_self_contradiction(self, flags):
        """F4：surface 说「不表态」而 description 要「表达/明确」= prompt 自相矛盾（改前 59 行）。"""
        assert flags["F4"] == [], f"F4 命中 {flags['F4']}"

    def test_f5_db_seed_has_truth(self, flags):
        """F5：db_seed 不许是占位符（改前 288 行 / 96 base，业务事实没有真值）。"""
        assert flags["F5"] == [], f"F5 命中 {flags['F5'][:8]}（共 {len(flags['F5'])} 行）"

    def test_f6_form_matches_key_form(self, flags, rows):
        """F6：`key_form=statement` 的 surface 不许是疑问式。**作用域 T1+T3**。"""
        scope = {r["scenario_id"] for r in _t1t3(rows)}
        hit = [s for s in flags["F6"] if s in scope]
        assert hit == [], f"F6 在 T1/T3 命中 {hit}"

    def test_f7_identity_digits_migrated(self, flags):
        """F7：identity 里数字 ≥3 个的要把处境类数字外迁到 user_facts（改前 159 行 / 53 base）。"""
        assert flags["F7"] == [], f"F7 命中 {flags['F7'][:8]}（共 {len(flags['F7'])} 行）"

    def test_no_base_hit_any(self, flags, rows):
        "改前 176/202 base 至少命中一条（ §1）；改后在**修复面**上必须是 0 个 base"
        assert set(flags) == set(H.DEFECT_CODES), "缺陷码集合与 hygiene 不一致"
        t1t3 = {r["scenario_id"] for r in _t1t3(rows)}
        hit: set[str] = set()
        for code in H.DEFECT_CODES:
            sids = flags[code] if code not in H.SURFACE_TEXT_CODES else [s for s in flags[code] if s in t1t3]
            hit |= {H.base_id(s) for s in sids}
        assert hit == set(), f"修复面上仍有 {len(hit)} 个 base 命中缺陷：{sorted(hit)[:8]}"
        assert len({H.base_id(r["scenario_id"]) for r in rows}) == N_BASE


class TestNewAssertions:
    "断言 8–10： §5 表里新增的三条"

    def test_8_base_layer_has_no_surface(self, base_rows):
        "断言 8：`scenarios_base.jsonl` 里不许再有 `key_event.surface`（改前 138 条有值）"
        holders = H.base_layer_surface_ids(base_rows)
        assert holders == [], f"base 层仍有 {len(holders)} 条带 surface：{holders[:8]}"
        assert len(base_rows) == N_BASE

    def test_9_objective_intent_full_coverage(self, rows):
        """断言 9：606 条全有 `objective` 与 `intent`（改动 2 的新字段，description 保留原值）。"""
        missing = H.objective_intent_missing(rows)
        assert len(rows) == N_SCENARIOS
        assert missing == [], f"{len(missing)} 条缺 objective/intent：{missing[:8]}"

    def test_10_surface_style_all_pass(self, rows):
        "断言 10：所有 T1/T3 的 surface ≤20 字、数字 ≤1、不以「你」开头（404/404）"
        scope = _t1t3(rows)
        assert len(scope) == 2 * N_BASE, f"T1/T3 应为 {2 * N_BASE} 行，实为 {len(scope)}"
        bad = H.surface_style_violations(scope)
        assert bad == [], f"{len(bad)} 条 surface 违规：{bad[:5]}"


class TestFrozenAndInvariant:
    """断言 11–14：本次执行中补的护栏（不替代上面十条，只加严）。"""

    def test_12_base_layer_mirrors_expansion(self, base_rows, rows):
        "断言 12：base 层与展开层除 `surface` 外逐字段相同（含三层之间）"
        drift = H.shared_field_drift(base_rows, rows)
        assert drift == {}, f"base↔展开层字段漂移：{ {k: v[:5] for k, v in drift.items()} }"

    def test_13_kb_family_matches_sub_domain(self, rows):
        "断言 13（F8）：marketing 各 sub_domain 的 kb 家族必须是本域那一套"
        bad = H.kb_family_mismatches(rows)
        assert bad == [], f"{len(bad)} 个 base 的 kb 家族与 sub_domain 错配：{bad[:5]}"

    def test_14_product_appears_in_kb(self, rows):
        "断言 14：marketing 的产品名必须同时出现在 db_seed 与 knowledge_base"
        bad = H.product_kb_missing(rows)
        assert bad == [], f"{len(bad)} 个 marketing base 的产品没进 kb：{bad[:8]}"

    def test_15_background_card_has_no_placeholder(self, rows):
        """断言 15：`background_card` 里不许再有占位商品名（F5 的判据只扫 db_seed，扫不到这里）。

        执行期实测：改完 `db_seed` 之后仍有 **96 个 base** 的卡片写着「产品99 / 示例商品」，
        而 `background_card.consensus` 是经 `run_batch.py:197` 进用户 prompt 的（渲染成「背景：…」）
        ⇒ 只修 db_seed 会让同一条业务事实在一份数据里有两种说法，比占位符更糟。
        判据用 `PLACEHOLDER_IN_PROSE`（比 F5 窄，不含「测试」这个常用词，避免假违规）。
        """
        bad = H.derived_view_placeholders(rows)
        assert bad == [], f"{len(bad)} 个 base 的 background_card 仍有占位名：{bad[:3]}"

    def test_16_background_card_mirrors_db_seed(self, rows):
        "断言 16：`background_card.world_state` 镜像的商品/产品名必须与 `db_seed` 逐字相同"
        bad = H.background_card_db_mismatch(rows)
        assert bad == [], f"{len(bad)} 个 base 的卡片与 db_seed 说法不一致：{bad[:3]}"

    def test_17_t3_surface_has_no_own_state_ban_term(self, rows):
        "断言 17（F13a）：T3 的 `surface` 不许含**本态**的动作/结论禁词"
        bad = H.t3_ban_hits(rows, scope="own")
        assert bad == [], f"{len(bad)} 条 T3 surface 含本态禁词：{bad[:5]}"

    def test_18_cross_state_and_prompt_field_ban_hits_not_worse(self, rows):
        "断言 18（F13b/F13c 的**棘轮**，不是归零门）：跨态与 objective/intent 的本态禁词命中数不许变多"
        cross = H.t3_ban_hits(rows, scope="cross")
        obj = H.field_ban_hits(rows, "objective")
        inte = H.field_ban_hits(rows, "intent")
        assert len(cross) <= 15, f"T3 surface 跨态禁词命中变多了（{len(cross)} > 15）：{cross[:5]}"
        assert len(obj) <= 23, f"T3 objective 本态禁词命中变多了（{len(obj)} > 23）：{obj[:5]}"
        assert len(inte) <= 6, f"T3 intent 本态禁词命中变多了（{len(inte)} > 6）：{inte[:5]}"

    def test_20_urgent_intent_has_no_own_state_ban_term(self, rows):
        "Test 20 urgent intent has no own state ban term."
        bad = H.field_ban_hits(rows, "intent", states=("urgent",))
        assert bad == [], f"{len(bad)} 条 urgent 的 T3 intent 含本态禁词：{bad[:5]}"

    def test_21_trigger_hint_ban_hits_are_registered_not_judged(self, rows):
        "断言 21（F13d 的**棘轮**）：`trigger_hint` 的本态禁词命中数不许变多"
        hits = H.trigger_hint_ban_hits(rows)
        assert len(hits) <= 28, f"trigger_hint 本态禁词命中变多了（{len(hits)} > 28）：{hits[:5]}"

    def test_22_surfaces_are_unique_across_bases(self, rows):
        "中性措辞的表述指令在基础场景间保持唯一；显式措辞保留既有上限"
        dup3 = H.duplicate_surfaces(rows, layers=("T3",))
        assert not dup3, (
            f"T3 有 {len(dup3)} 组跨 base 重复的 surface：{list(dup3.items())[:3]}"
        )
        dup1 = H.duplicate_surfaces(rows, layers=("T1",))
        assert len(dup1) <= 16, (
            f"T1 跨 base 重复的 surface 变多了（{len(dup1)} > 16 组）：{list(dup1.items())[:3]}"
        )


class TestBaselineDocumented:
    """改前基线数字留档（不参与判定，只保证报告里的「改前」有据可查）。"""

    def test_pre_fix_hits_are_documented(self):
        assert sum(_PRE_FIX_HITS.values()) == 622
        assert _PRE_FIX_HITS == {
            "F1": 1,
            "F2": 6,
            "F3": 24,
            "F4": 59,
            "F5": 288,
            "F6": 85,
            "F7": 159,
        }
