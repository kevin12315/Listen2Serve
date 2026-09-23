"""Plan L §4.3 emotion 模块口径测试（6 态底色 + 生成期种子句式）。

防回归点：
1. STATE_EMOTION 与 STATE_LABELS 6 态同码（导入期 _validate_library 已校验）；
2. tts_control_for 的种子口径：一律「情绪克制自然」，必含语速词/音量词、句间停顿短、
   总长 ≤80；不再接 level 参数（按透明度改韵律的行为已退役，§0.3）；
3. 韵律硬约束：hushed 态豁免「音量≥正常」（用「音量压低」），其余态维持语速≥中等、
   音量≥正常；
4. 退役面：TONE_WORDS / INTENSITY_PROSODY 已从模块移除，不得复活。
"""

import pytest

from listen2serve.domains.base import STATE_LABELS
from listen2serve.emotion import STATE_EMOTION, tts_control_for

# 韵律关键词约定（与运行时 _prosody_from_control 启发式对齐）
RATE_WORDS = ("语速中等", "语速偏快", "语速快")
VOL_WORDS = ("音量正常", "音量略升", "音量大", "音量压低")
# 违反「语速≥中等、句间无长停顿」或属内部术语/元语言的禁用表述
BANNED = ("偏慢", "小声", "音量降", "停顿长", "停顿略长", "一度", "二度", "三度",
          "情境：", "句尾收稳", "不加掩饰", "起伏大", "句尾下沉")


def _has_rate(text: str) -> bool:
    return any(w in text for w in RATE_WORDS)


def _has_vol(text: str) -> bool:
    return any(w in text for w in VOL_WORDS)


def test_state_tables_six_states():
    """6 态同码：STATE_EMOTION 与 STATE_LABELS 一致。"""
    assert set(STATE_EMOTION) == set(STATE_LABELS)
    assert set(STATE_LABELS) == {
        "neutral", "cooperative", "displeased", "urgent", "doubtful", "hushed"}


def test_control_for_all_states():
    """全量 6 态（含未知状态码回落）：关键词合规、无禁用表述、总长 ≤80。"""
    for state in list(STATE_EMOTION) + ["未知状态码"]:
        s = tts_control_for(state)
        assert len(s) <= 80, f"{state} 超长: {s}"
        assert _has_rate(s) and _has_vol(s) and "句间停顿短" in s, f"{state}: {s}"
        for b in BANNED:
            assert b not in s, f"{state} 含禁用表述 {b}: {s}"


def test_control_for_restrained_tone():
    """种子句式一律「情绪克制自然」口径（不随透明度分档）。"""
    for state in STATE_EMOTION:
        assert "情绪克制自然" in tts_control_for(state), state


def test_control_for_role_param():
    """支持 role 参数注入角色设定（句首），缺省为通用角色。"""
    s = tts_control_for("neutral", role="一位38岁男性欠款客户")
    assert s.startswith("一位38岁男性欠款客户，")
    assert tts_control_for("neutral").startswith("一位普通电话用户，")
    assert len(s) <= 80


def test_hushed_volume_exemption():
    """韵律硬约束：hushed 态豁免「音量≥正常」，用「音量压低」；其余态不得压低。"""
    assert "音量压低" in tts_control_for("hushed")
    for state in set(STATE_EMOTION) - {"hushed"}:
        s = tts_control_for(state)
        assert "音量压低" not in s, f"{state}: {s}"
        assert any(w in s for w in ("音量正常", "音量略升", "音量大")), s


def test_retired_symbols_absent():
    """Plan L §4.3 退役面：候选词池与强度三档韵律表不得再出现在模块里。

    这两个符号的消费者（tts.tone 槽位、key_event.intensity 字段）都已随 Plan L 删除，
    留着它们只会给「每轮抽词 → 同条目跑出不同文本」这类不可重现行为留后门。
    """
    import listen2serve.emotion as em

    for name in ("TONE_WORDS", "INTENSITY_PROSODY", "_INTENSITY_NAME", "EMOTION_LIBRARY"):
        assert not hasattr(em, name), f"退役符号复活: {name}"


def test_control_for_rejects_level_arg():
    """旧签名 tts_control_for(state, level) 必须不再被接受（避免静默按位置吞掉 level）。"""
    with pytest.raises(TypeError):
        tts_control_for("neutral", "T3", "一位普通电话用户")  # type: ignore[call-arg]
