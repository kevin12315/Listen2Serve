"""lexicon_check 行为测试（Plan I v6.0：关键轮不再跳过，全轮同口径）。"""

from __future__ import annotations

from listen2serve.evaluation.lexicon_check import check_user_lexicon


def _scenario(label: str = "T2", key_state: str = "displeased") -> dict:
    return {
        "leakage_label": label,
        "leakage_level": {"T1": "explicit", "T2": "implicit_consistent", "T3": "prosody_only"}[label],
        "user_script": {"key_event": {"state": key_state}},
    }


def _traj(turns: list[dict]) -> dict:
    return {"turns": turns}


class TestLexiconCheck:
    def test_t2_all_turns_zero_hit(self):
        """T2 全轮（含关键轮）命中明确情绪词 → 计违规。"""
        traj = _traj([
            {"turn": 1, "user_text": "那个，我现在手头有点紧。", "user_state": "neutral"},
            {"turn": 2, "user_text": "你们这样我真烦。", "user_state": "displeased"},
            {"turn": 3, "user_text": "这事真闹心，就这么着吧。", "user_state": "displeased",
             "is_key_turn": True},  # 关键轮同样校验（Plan I）
        ])
        r = check_user_lexicon(_scenario("T2"), traj)
        assert r["vocab_checked_turns"] == 3
        assert r["vocab_violation_turns"] == 2
        assert {v["turn"] for v in r["vocab_violations"]} == {2, 3}

    def test_t1_key_positive_check(self):
        """T1 关键轮：正向命中 key_event.state 子表词通过；未命中计违规。"""
        # displeased 子表含「烦」
        ok_traj = _traj([
            {"turn": 1, "user_text": "嗯，你先说说什么情况。", "user_state": "neutral"},
            {"turn": 2, "user_text": "我真是烦透了，这钱的事没完。",
             "user_state": "displeased", "is_key_turn": True},
        ])
        r = check_user_lexicon(_scenario("T1"), ok_traj)
        assert r["vocab_violation_turns"] == 0
        bad_traj = _traj([
            {"turn": 1, "user_text": "行吧，那就先这样处理一下。",
             "user_state": "displeased", "is_key_turn": True},
        ])
        r2 = check_user_lexicon(_scenario("T1"), bad_traj)
        assert r2["vocab_violation_turns"] == 1
        assert r2["vocab_violations"][0]["is_key_turn"] is True

    def test_t1_non_key_zero_hit(self):
        """T1 非关键轮（v6.0 克制中性）：命中情绪词计违规。"""
        traj = _traj([
            {"turn": 1, "user_text": "我真的很烦。", "user_state": "displeased"},
            {"turn": 2, "user_text": "嗯，你先说说情况。", "user_state": "neutral",
             "is_key_turn": True},  # neutral 关键事件无子表词要求
        ])
        r = check_user_lexicon(_scenario("T1", key_state="neutral"), traj)
        assert r["vocab_checked_turns"] == 2
        assert r["vocab_violation_turns"] == 1
        assert r["vocab_violations"][0]["turn"] == 1

    def test_legacy_run_fallback_critical_turn(self):
        """旧 run（无 is_key_turn 字段）回落场景 critical_turn 定位。"""
        sc = _scenario("T2")
        sc["measurement"] = {"critical_turn": 2}
        sc.pop("user_script")
        sc["oracle_state"] = "用户处于N1_frustrated状态"
        traj = _traj([
            {"turn": 1, "user_text": "你们这样我真烦。", "user_state": "N1_frustrated"},
            {"turn": 2, "user_text": "烦死人的事情又来了。", "user_state": "N1_frustrated"},
        ])
        r = check_user_lexicon(sc, traj)
        # T2 全轮校验（关键轮也不跳过）
        assert r["vocab_checked_turns"] == 2
        assert r["vocab_violation_turns"] == 2

    def test_empty_text_and_missing_fields_fallback(self):
        """空文本不计入校验集；缺字段轨迹不报错。"""
        r = check_user_lexicon(_scenario("T2"), {"turns": [{"turn": 1, "user_text": ""}]})
        assert r["vocab_checked_turns"] == 0
        r2 = check_user_lexicon({}, {})
        assert r2["vocab_violation_rate"] is None

    def test_vocab_words_read_dynamically(self):
        """词表动态读自 listen2serve.vocab（monkeypatch 扩容立即生效，证明无硬编码副本）。"""
        import listen2serve.vocab as vocab

        traj = _traj([{"turn": 1, "user_text": "这事儿真闹心。", "user_state": "displeased"}])
        assert check_user_lexicon(_scenario("T2"), traj)["vocab_violation_turns"] == 1
        orig_neg, orig_pos = vocab.NEG_WORDS, vocab.POS_WORDS
        try:
            vocab.NEG_WORDS, vocab.POS_WORDS = [], []
            r = check_user_lexicon(_scenario("T2"), traj)
            assert r["vocab_violation_turns"] == 0
        finally:
            vocab.NEG_WORDS, vocab.POS_WORDS = orig_neg, orig_pos
