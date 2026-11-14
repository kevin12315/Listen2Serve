"§4.3 emotion 模块口径测试（6 态底色 + 生成期种子句式）"

import pytest

from listen2serve.domains.base import STATE_LABELS
from listen2serve.emotion import STATE_EMOTION, tts_control_for

# 韵律关键词约定（与运行时 _prosody_from_control 启发式对齐）
RATE_WORDS = ("语速中等", "语速偏快", "语速快")
VOL_WORDS = ("音量正常", "音量略升", "音量大", "音量压低")

BANNED = (
    "偏慢",
    "小声",
    "音量降",
    "停顿长",
    "停顿略长",
    "一度",
    "二度",
    "三度",
    "情境：",
    "句尾收稳",
    "不加掩饰",
    "起伏大",
    "句尾下沉",
)


def _has_rate(text: str) -> bool:
    return any(w in text for w in RATE_WORDS)


def _has_vol(text: str) -> bool:
    return any(w in text for w in VOL_WORDS)


def test_state_tables_six_states():
    """6 态同码：STATE_EMOTION 与 STATE_LABELS 一致。"""
    assert set(STATE_EMOTION) == set(STATE_LABELS)
    assert set(STATE_LABELS) == {"neutral", "cooperative", "displeased", "urgent", "doubtful", "hushed"}


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
    "§4.3 退役面：候选词池与强度三档韵律表不得再出现在模块里"
    import listen2serve.emotion as em

    for name in ("TONE_WORDS", "INTENSITY_PROSODY", "_INTENSITY_NAME", "EMOTION_LIBRARY"):
        assert not hasattr(em, name), f"退役符号复活: {name}"


def test_control_for_rejects_level_arg():
    "Test control for rejects level arg."
    with pytest.raises(TypeError):
        tts_control_for("neutral", "T3", "一位普通电话用户")  # type: ignore[call-arg]
