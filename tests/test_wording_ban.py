"T3 禁止动作表（`data/benchmark/neutral_wording_ban.json`）的发布契约"

from __future__ import annotations

import pytest

from listen2serve.data import wording_ban as ban

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
        assert not (set(hits.get("ban", [])) | set(hits.get("screen_only", []))) & rejected, (state, hits)


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
        summary = ban.action_summary(state)
        assert summary and "\n" not in summary.strip(), (state, summary[:60])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("具体怎么还、最晚什么时候还，你说一下，我配合。", "我配合"),
        ("我会配合。请先说一下办理流程。", "我会配合"),
        ("按这个流程办，我这边积极配合。", "我这边积极配合"),
        ("按这个流程办，这边我配合。", "这边我配合"),
        ("我不配合，你先把记录给我。", "我不配合"),
    ],
)
def test_explicit_cooperation_declarations_are_identified(text, expected):
    assert ban.explicit_state_declaration_hits(text) == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "行，就按这个流程办。",
        "可以，三号还六千，请帮我登记。",
        "我配合你核对材料，先核对哪一项？",
        "这边我配合你核对材料，先核对哪一项？",
        "如果需要我配合，请告诉我要哪些材料。",
        "需不需要我配合核对身份？",
        "你们需要我配合吗？",
        "页面上写着‘我配合’，这个按钮有什么用？",
        "他们说要配合办理，具体怎么操作？",
        "我配合的材料都已经上传了。",
        "",
    ],
)
def test_business_agreement_and_references_are_not_state_declarations(text):
    assert ban.explicit_state_declaration_hits(text) == []
