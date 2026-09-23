"""Plan G：运行期词表豁免口径测试（豁免表显式可查 + 关键轮契约零改动）。

覆盖：
1. RUNTIME_CONTEXT_EXEMPT / RUNTIME_PHRASE_EXEMPT 逐条豁免生效（条目来源：
   Plan G Step 2 误报聚类 （内部产物，未随本仓发布））；
2. 守卫词失效：抗拒/威胁语境下「挂了」不豁免；
3. validate_level（v4 三方共用契约）与 validate_level_v2（关键轮预写文本契约）
   语义零改动；validate_level_runtime 的 T1 分支与旧口径一致；
4. 真实违规语境（情绪义命中）在运行期口径下仍计违规。
"""

from __future__ import annotations

from listen2serve.vocab import (
    NEG_CONTEXT_EXEMPT,
    NEG_WORDS,
    POS_WORDS,
    RUNTIME_CONTEXT_EXEMPT,
    RUNTIME_PHRASE_EXEMPT,
    runtime_emotion_hits,
    validate_level,
    validate_level_runtime,
    validate_level_v2,
)


class TestRuntimeExemptTable:
    """豁免表逐条生效（Step 2 误报聚类的代表性样例）。"""

    def test_gua_le_farewell_exempt(self):
        # 「挂了」道别语：左邻 先/我/，/。 × 右邻 。/啊/句尾（Step 2 误报 52/64）
        for text in (
            "那可不，三万我下个月十号前后还上，先这样，挂了。",
            "行，那就先帮我登记上，回头你们联系吧，我先挂了啊。",
            "把你工号报一下，我记着，后面我再跟进。挂了啊。",
            "先挂了。",
        ):
            ok, _ = validate_level_runtime(text, "T2", "")
            assert ok, text

    def test_gua_le_guard_invalidates_exempt(self):
        # 守卫词：句内含投诉/骚扰/别打/再打等抗拒语境 → 「挂了」不豁免
        for text in (
            "行，那就别再打了，我先挂了。",          # 含「再打」
            "以后不要再打来了，我先挂了。",          # 含「不要再打」
            "不然我就投诉了，先挂了。",              # 含「投诉」
            "那个，再打我就投诉骚扰电话了，挂了啊。",  # 含「骚扰」「再打」
        ):
            ok, reason = validate_level_runtime(text, "T2", "")
            assert not ok, text
            assert "挂了" in reason or "投诉" in reason or "骚扰" in reason

    def test_fan_in_mafen_exempt(self):
        # 「烦」⊂ 礼貌用语「麻烦」（Step 2 误报 5/5）
        assert validate_level_runtime("行，那就这么办，麻烦你了，再见。", "T3", "")[0]

    def test_buxuyao_in_xuyu_buxuyao_exempt(self):
        # 「不需要」⊂ 中性征询「需不需要」（Step 2 误报 1/1）
        assert validate_level_runtime("你们这渠道我打过几次了，需不需要我再补点啥材料？", "T2", "")[0]

    def test_cui_business_confirmation_exempt(self):
        # 「催这期…还款」催收业务确认问句（Step 2 误报 4/9）
        assert validate_level_runtime("喂，我是韩超，你们是催这期还款的吧？", "T2", "")[0]

    def test_tuozhe_promise_exempt(self):
        # 「不拖着」承诺不拖延（Step 2 误报 3/5）；抱怨语境不豁免
        assert validate_level_runtime("嗯，你把账核对清楚，我就交，不拖着。", "T2", "")[0]
        assert not validate_level_runtime("问题好几天了，一直拖着没处理好。", "T3", "")[0]

    def test_bu_fangbian_time_statement_exempt(self):
        # 「这会儿/今天 + 不方便」客观时间陈述（Step 2 误报 2/2）
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
            ("你们这么打电话催，合法吗？", "T2"),
            ("你们平台先别催了。", "T2"),
            ("我都没钱，你们还天天来，太欺负人了。", "T2"),
        ):
            ok, reason = validate_level_runtime(text, level, "")
            assert not ok, text
            assert "命中明确情绪词" in reason

    def test_runtime_hits_subset_of_raw(self):
        # 运行期命中 ⊆ 原始子串命中（豁免只减不增）
        text = "行，那就这么办，麻烦你了，先挂了啊。"
        raw = [w for w in NEG_WORDS + POS_WORDS if w in text]
        assert set(runtime_emotion_hits(text)) <= set(raw)


class TestLegacyContractsUnchanged:
    """铁律：v4 三方共用契约与 v2 关键轮契约语义零改动。"""

    def test_validate_level_v4_unchanged(self):
        # 旧口径无豁免：「麻烦」按「烦」计、道别语「挂了」按命中计
        assert validate_level("麻烦你了", "T3", "")[0] is False
        assert validate_level("先这样，挂了。", "T2", "")[0] is False
        # T1 极性命中语义不变
        assert validate_level("我真是太失望了", "T1", "N1_frustrated")[0] is True

    def test_validate_level_v2_unchanged(self):
        # v2 仅沿用 NEG_CONTEXT_EXEMPT（「先挂了」），不受 Plan G 新豁免表影响：
        # 「麻烦」在 v2 下仍按「烦」计违规（新豁免不进 v2；句内带业务白名单词
        # 排除 T3 下限干扰）
        assert NEG_CONTEXT_EXEMPT == {"挂了": ("先挂了",)}
        assert validate_level_v2("麻烦你帮我确认下流程。", "T3", "")[0] is False

    def test_runtime_t1_branch_equals_v4(self):
        texts = ["我真是太失望了", "这个业务怎么办", "没问题，我配合"]
        for t in texts:
            for state in ("N1_frustrated", "P1_cooperative"):
                assert (validate_level_runtime(t, "T1", state)
                        == validate_level(t, "T1", state)), t
