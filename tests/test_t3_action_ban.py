"""T3 禁止动作表（`data/benchmark/t3_action_ban.json`）的发布契约。

这张表是 KeyTurnPass 的放行门：裁判给到 4 分以上、但当轮命中"禁止动作"仍判不通过。
它必须与场景数据同源分发，并能被生成侧/检测侧代码直接 import（全仓唯一数据源，
不允许各抄一份副本）。内部仓里那些"合并前快照比对 / 阈值裁定"的回归测试属数据治理
历史，不在发布范围；本文件只守公开契约：能加载、五态齐全、条目带出处、渲染出的
条款含条目原文、命中筛查可用、被剔除的词不会顺手回来。
"""
from __future__ import annotations

import pytest

from listen2serve.data import t3_action_ban as ban

STATES = {"cooperative", "displeased", "doubtful", "urgent", "hushed"}


def test_loads_and_covers_five_states():
    doc = ban.load()
    assert doc["schema"] == "t3_action_ban/v1"
    assert set(ban.states()) == STATES, ban.states()
    assert set(doc["tier_definitions"]) == set(ban.TIERS)


@pytest.mark.parametrize("state", sorted(STATES))
def test_entries_have_provenance(state):
    rows = ban.entries(state, tiers=ban.TIERS)
    assert rows, f"{state} 一条记录都没有"
    for r in rows:
        assert r["tier"] in ban.TIERS
        assert r["term"], f"{state} 有条目没有词面"
        # 出处三件套：来自哪份源词表、属哪个轴、是否已并进 v9 动作摘要
        assert {"from", "axis", "maps_to_v9"} <= set(r), f"{state}/{r['term']} 缺出处字段"
        assert r["axis"] in ban.AXES, (r["term"], r["axis"])


@pytest.mark.parametrize("state", sorted(STATES))
def test_ban_state_has_ban_tier(state):
    """五态都得有真禁止项；某态为 0 说明口径塌了（而不是"该态没有禁止动作"）。"""
    assert ban.terms(state, ("ban",)), f"{state} 的 ban 层为空"


@pytest.mark.parametrize("state", sorted(STATES))
def test_rendered_clause_contains_terms(state):
    clause = ban.ban_clause([state])
    assert clause.strip()
    for t in ban.terms(state, ("ban",))[:3]:
        assert t in clause, f"{state} 渲染出的条款缺 {t!r} ⇒ 裁判看不到该禁止项"


def test_screen_hits_finds_banned_and_passes_clean():
    state = "hushed"
    term = ban.terms(state, ("ban",))[0]
    hits = ban.screen_hits(state, f"客服：{term}，然后我们继续。")
    assert hits.get("ban") == [term], hits
    # 干净句：ban 层必须空；screen_only 层命中（"方便""短"这类排队线索词）是设计如此，
    # 它只把样本送进第二关（LLM 精判），不直接判不通过。
    clean = ban.screen_hits(state, "客服：您现在方便讲话吗？我长话短说。")
    assert clean.get("ban", []) == [], clean


def test_rejected_terms_are_not_in_default_screen():
    """rejected/candidate 两层必须显式点名才生效：默认筛查面不能把它们捞回来。"""
    for state in ban.states():
        rejected = set(ban.terms(state, ("rejected",)))
        if not rejected:
            continue
        hits = ban.screen_hits(state, "".join(sorted(rejected)))
        assert not (set(hits.get("ban", [])) | set(hits.get("screen_only", []))) & rejected, \
            (state, hits)


def test_coverage_counts_are_derived_not_hardcoded():
    cov = ban.coverage()
    assert set(cov) == STATES
    for state, row in cov.items():
        assert row["ban"] == len(ban.terms(state, ("ban",))), (state, row)
        assert all(isinstance(v, int) and v >= 0 for v in row.values())


def test_conflicts_are_recorded():
    """合并时的分歧必须留在表里可查，不能悄悄选一个。"""
    rows = ban.conflicts()
    assert rows, "conflicts 为空：合并过程声称零分歧，可信度反而更低"
    for c in rows:
        assert {"id", "term", "state", "resolution"} <= set(c), c
    assert {c["resolution"] for c in rows} & {"rejected", "kept", "absorbed", "split"}


def test_action_summary_is_single_line():
    for state in STATES:
        summary = ban.v9_action_summary(state)
        assert summary and "\n" not in summary.strip(), (state, summary[:60])
