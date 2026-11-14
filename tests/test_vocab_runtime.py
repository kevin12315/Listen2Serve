"""运行时情绪词与业务语境豁免检查。"""

from __future__ import annotations

import pytest

from listen2serve.vocab import (
    NEG_CONTEXT_EXEMPT,
    NEG_WORDS,
    POS_WORDS,
    RUNTIME_CONTEXT_EXEMPT,
    RUNTIME_PHRASE_EXEMPT,
    neutral_wording_hits,
    runtime_emotion_hits,
    validate_level,
    validate_level_runtime,
    validate_wording,
)


class TestRuntimeExemptTable:
    """逐项检查业务语境和道别表达的豁免。"""

    def test_gua_le_farewell_exempt(self):
        # 「挂了」道别语：左邻 先/我/，/。 × 右邻 。/啊/句尾
        for text in (
            "那可不，三万我下个月十号前后还上，先这样，挂了。",
            "行，那就先帮我登记上，回头你们联系吧，我先挂了啊。",
            "把你工号报一下，我记着，后面我再跟进。挂了啊。",
            "先挂了。",
        ):
            ok, _ = validate_level_runtime(text, "T3", "")
            assert ok, text

    def test_gua_le_guard_invalidates_exempt(self):
        # 守卫词：句内含投诉/骚扰/别打/再打等抗拒语境 → 「挂了」不豁免
        for text in (
            "行，那就别再打了，我先挂了。",  # 含「再打」
            "以后不要再打来了，我先挂了。",  # 含「不要再打」
            "不然我就投诉了，先挂了。",  # 含「投诉」
            "那个，再打我就投诉骚扰电话了，挂了啊。",  # 含「骚扰」「再打」
        ):
            ok, reason = validate_level_runtime(text, "T3", "")
            assert not ok, text
            assert "挂了" in reason or "投诉" in reason or "骚扰" in reason

    def test_fan_in_mafen_exempt(self):
        # 「烦」⊂ 礼貌用语「麻烦」
        assert validate_level_runtime("行，那就这么办，麻烦你了，再见。", "T3", "")[0]

    def test_buxuyao_in_xuyu_buxuyao_exempt(self):
        # 「不需要」⊂ 中性征询「需不需要」
        assert validate_level_runtime("你们这渠道我打过几次了，需不需要我再补点啥材料？", "T3", "")[0]

    def test_cui_business_confirmation_exempt(self):
        # 「催这期…还款」催收业务确认问句
        assert validate_level_runtime("喂，我是韩超，你们是催这期还款的吧？", "T3", "")[0]

    def test_tuozhe_promise_exempt(self):
        # 「不拖着」承诺不拖延；抱怨语境不豁免
        assert validate_level_runtime("嗯，你把账核对清楚，我就交，不拖着。", "T3", "")[0]
        assert not validate_level_runtime("问题好几天了，一直拖着没处理好。", "T3", "")[0]

    def test_bu_fangbian_time_statement_exempt(self):
        # 「这会儿/今天 + 不方便」客观时间陈述
        assert validate_level_runtime("这会儿不方便，回头再联系吧，先挂了啊。", "T3", "")[0]

    def test_every_entry_has_source(self):
        # 豁免表显式可查、逐条有来源（验收标准）
        for word, rules in RUNTIME_CONTEXT_EXEMPT.items():
            assert word in NEG_WORDS + POS_WORDS, word
            for rule in rules:
                assert rule.get("source"), f"{word} 缺来源标注"
                assert set(rule) <= {"left", "right", "guard", "source"}
        for word, phrases in RUNTIME_PHRASE_EXEMPT.items():
            assert word in NEG_WORDS + POS_WORDS, word
            for p in phrases:
                assert word in p, f"短语豁免 {p!r} 不含命中词 {word!r}"


class TestRuntimeTrueViolation:
    """真实情绪义命中在运行期口径下仍计违规（豁免不得过度）。"""

    def test_true_violations_still_caught(self):
        for text, level in (
            ("这事真烦人，我要投诉。", "T3"),
            ("你们这么打电话催，合法吗？", "T3"),
            ("你们平台先别催了。", "T3"),
            ("我都没钱，你们还天天来，太欺负人了。", "T3"),
        ):
            ok, reason = validate_level_runtime(text, level, "")
            assert not ok, text
            assert "命中明确情绪词" in reason

    def test_runtime_hits_subset_of_raw(self):
        # 运行期命中 ⊆ 原始子串命中（豁免只减不增）
        text = "行，那就这么办，麻烦你了，先挂了啊。"
        raw = [w for w in NEG_WORDS + POS_WORDS if w in text]
        assert set(runtime_emotion_hits(text)) <= set(raw)


class TestWordingContracts:
    """原始匹配、文本条件和运行时匹配保持各自语义。"""

    def test_raw_emotion_matching(self):
        # 旧口径无豁免：「麻烦」按「烦」计、道别语「挂了」按命中计
        assert validate_level("麻烦你了", "T3", "")[0] is False
        assert validate_level("先这样，挂了。", "T3", "")[0] is False
        # T1 极性命中语义不变
        assert validate_level("我真是太失望了", "T1", "N1_frustrated")[0] is True

    def test_wording_matching(self):

        assert NEG_CONTEXT_EXEMPT == {"挂了": ("先挂了",)}
        assert validate_wording("麻烦你帮我确认下流程。", "T3", "")[0] is False

    def test_runtime_explicit_matching(self):
        texts = ["我真是太失望了", "这个业务怎么办", "没问题，我配合"]
        for t in texts:
            for state in ("N1_frustrated", "P1_cooperative"):
                assert validate_level_runtime(t, "T1", state) == validate_level(t, "T1", state), t


@pytest.mark.parametrize("level", ["T2", "T3"])
def test_runtime_neutral_contract_rejects_explicit_cooperation(level):
    text = "具体还款方式你说一下，我配合。"
    ok, reason = validate_level_runtime(text, level, "cooperative")
    assert not ok
    assert "明确状态自述" in reason
    assert "我配合" in reason
    assert neutral_wording_hits(text) == ["我配合"]


def test_prepared_neutral_wording_rejects_the_same_declaration():
    assert not validate_wording("还款方式你说一下，我配合。", "T3", "cooperative")[0]


def test_original_vocabulary_and_explicit_contract_are_preserved():
    assert "我配合" not in POS_WORDS
    assert "我配合" not in runtime_emotion_hits("我配合。")
    assert validate_level_runtime("我愿意配合办理还款。", "T1", "cooperative")[0]
    assert validate_wording("我愿意配合办理还款。", "T1", "cooperative")[0]


@pytest.mark.parametrize(
    "text",
    [
        "行，就按这个流程办。",
        "可以，三号还六千，请帮我登记。",
        "我配合你核对材料，先核对哪一项？",
        "如果需要我配合，请告诉我要哪些材料。",
    ],
)
def test_neutral_business_actions_remain_valid(text):
    assert neutral_wording_hits(text) == []
    assert validate_level_runtime(text, "T3", "cooperative")[0]


def test_saved_neutral_turn_is_flagged_without_changing_the_trajectory():
    from copy import deepcopy

    from listen2serve.evaluation.lexicon_check import check_user_lexicon

    trajectory = {
        "turns": [
            {
                "turn": 4,
                "user_state": "cooperative",
                "is_key_turn": True,
                "user_text": "行，那我就在官方应用里操作。具体怎么还、最晚什么时候还，你说一下，我配合。",
            }
        ]
    }
    original = deepcopy(trajectory)
    result = check_user_lexicon(
        {"leakage_label": "T3", "user_script": {"key_event": {"state": "cooperative"}}}, trajectory
    )
    assert result["vocab_checked_turns"] == 1
    assert result["vocab_violation_turns"] == 1
    assert result["vocab_violations"][0]["is_key_turn"] is True
    assert "我配合" in result["vocab_violations"][0]["reason"]
    assert trajectory == original
