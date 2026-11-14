"情绪状态底色"

from __future__ import annotations

# F2：中文状态标签唯一来源 domains.base.STATE_LABELS（严禁复制副本）
from listen2serve.domains.base import STATE_LABELS

STATE_EMOTION: dict[str, str] = {
    "neutral": "自然平实、平稳",
    "cooperative": "平和配合、稳定",
    "displeased": "不满不耐、发冲",
    "urgent": "急切催促、发急",
    "doubtful": "将信将疑、字斟句酌",
    "hushed": "谨慎收敛、压着嗓子",
}

# ---- 韵律关键词约定（与 runtime 的 _prosody_from_control 启发式提取对齐）----
# 语速词 ∈ {"语速中等", "语速偏快", "语速快"}；音量词 ∈ {"音量正常", "音量略升", "音量大", "音量压低"}。
_SEED_PROSODY = "语速中等，音量正常，句间停顿短"
_SEED_PROSODY_HUSHED = "语速中等，音量压低，句间停顿短"


def tts_control_for(state: str, role: str = "一位普通电话用户") -> str:
    """Render a state-specific synthesis instruction."""
    emotion = STATE_EMOTION.get(state, STATE_EMOTION["neutral"])
    prosody = _SEED_PROSODY_HUSHED if state == "hushed" else _SEED_PROSODY
    return f"{role}，语气{emotion}，情绪克制自然，{prosody}"


def _validate_library() -> None:
    """导入时校验（F2 单一来源）：状态码与 STATE_LABELS 逐一对应，违规即阻断管线。"""
    assert set(STATE_EMOTION) == set(STATE_LABELS), "STATE_EMOTION 与 STATE_LABELS 状态码不一致"


_validate_library()
