# -*- coding: utf-8 -*-
"""Plan J §8.2：TTS 文本规范化单测（纯函数，无 API）。"""

from __future__ import annotations

from listen2serve.runtime.tts_normalize import _int_cn, normalize_tts_text


class TestIntCn:
    def test_basic(self):
        assert _int_cn(0) == "零"
        assert _int_cn(10) == "十"
        assert _int_cn(28) == "二十八"
        assert _int_cn(100) == "一百"
        assert _int_cn(3600) == "三千六百"
        assert _int_cn(9800) == "九千八百"
        assert _int_cn(10001) == "一万零一"
        assert _int_cn(26000) == "二万六千"
        assert _int_cn(15000) == "一万五千"


class TestNormalize:
    def test_money(self):
        assert normalize_tts_text("欠款3600元") == "欠款三千六百元"
        assert normalize_tts_text("99.9元") == "九十九点九元"

    def test_percent(self):
        assert normalize_tts_text("优惠30%") == "优惠百分之三十"

    def test_date(self):
        assert normalize_tts_text("2025-05-28到期") == "二零二五年五月二十八日到期"

    def test_ordinal_and_unit(self):
        assert normalize_tts_text("下个月10号") == "下个月十号"
        assert normalize_tts_text("逾期20天") == "逾期二十天"
        assert normalize_tts_text("第3轮") == "第三轮"

    def test_code_digits(self):
        assert normalize_tts_text("订单ORD0005") == "订单ORD零零零五"
        assert normalize_tts_text("工号1024") == "工号一零二四"

    def test_polyphone(self):
        assert normalize_tts_text("我想问下还款方式") == "我想问下归还方式"
        assert normalize_tts_text("帮我处理一下") == "帮我办理一下"
        # 豁免词不替换
        assert normalize_tts_text("还有别的事吗") == "还有别的事吗"
        assert normalize_tts_text("我提供了资料") == "我提供了资料"

    def test_idempotent_and_safe(self):
        """已规范化文本重跑不叠加；无数字文本原样。"""
        once = normalize_tts_text("欠款3600元，逾期20天")
        assert normalize_tts_text(once) == once
        assert normalize_tts_text("你好，哪位？") == "你好，哪位？"
        assert normalize_tts_text("") == ""
