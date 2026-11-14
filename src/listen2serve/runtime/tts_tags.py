"TTS 情感标签白名单与四重校验"

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 官方控制类标签（23 个）：作用于其后所有文本，直到下一个控制标签或长句被切分。
CONTROL_TAGS: frozenset[str] = frozenset(
    {
        "sad",
        "amazed",
        "trembling",
        "angry",
        "excited",
        "sarcastic",
        "curious",
        "bored",
        "tired",
        "scornful",
        "shouting",
        "deep and loud shouting",
        "asmr",
        "panicked",
        "mischievously",
        "empathetic",
        "whispers",
        "reluctantly",
        "crying",
        "serious",
        "very slowly",
        "very fast",
        "like dracula",
    }
)

RICH_TAGS: frozenset[str] = frozenset(
    {
        "gasp",
        "sighing",
        "clears throat",
        "giggles",
        "laughing",
        "cough",
        "snorts",
    }
)

# 支持富语言标签的模型（能力门控用）。
RICH_TAG_MODELS: frozenset[str] = frozenset(
    {
        "qwen-audio-3.0-tts-plus",
        "qwen-audio-3.0-tts-flash",
    }
)

STATE_TAG_ALLOWED: dict[str, str | None] = {
    "neutral": None,
    "cooperative": None,
    "displeased": "angry",
    "urgent": "very fast",
    "doubtful": "curious",
    "hushed": "whispers",
}

MAX_TAGS_PER_UTTERANCE = 1

_TAG_RE = re.compile(r"\[([^\[\]]{1,32})\]")


@dataclass
class TagResult:
    """标签校验结果：text 供评委/ASR 对齐，text_tts 实际送 TTS。"""

    text: str
    text_tts: str
    applied: str | None = None
    stripped: list[tuple[str, str]] = field(default_factory=list)  # (tag, reason)

    @property
    def has_violation(self) -> bool:
        return bool(self.stripped)


def find_tags(text: str) -> list[str]:
    """抽出文本里出现的全部方括号标记（保序，不去重）。

    供运行期记 `tag_raw`：LLM 协议已不含 `tag` 字段，它自行输出的标记属违约，
    得先留档再剔，否则“剔了什么”在报告里不可见。
    """
    return [m.group(1).strip() for m in _TAG_RE.finditer(text)]


def allowed_tags_from_entry(entry: dict) -> list[str]:
    "本 base 的标签允许集 = instruction 表里声明的强制标签（去重保序）"
    out: list[str] = []
    for tag in list((entry.get("tags") or {}).values()) + [entry.get("key_tag")]:
        if tag and tag not in out:
            out.append(str(tag))
    return out


def apply_tag_policy(
    text: str,
    *,
    allowed: list[str] | tuple[str, ...],
    model: str = "",
    enable_rich: bool = False,
) -> TagResult:
    "按四重校验裁剪台词中的标签，返回干净文本与实际送 TTS 文本"
    found = list(_TAG_RE.finditer(text))
    if not found:
        return TagResult(text=text, text_tts=text)

    allow = set(allowed)
    rich_ok = enable_rich and model in RICH_TAG_MODELS
    kept: re.Match[str] | None = None
    stripped: list[tuple[str, str]] = []

    for m in found:
        tag = m.group(1).strip()
        if tag in RICH_TAGS:
            if not rich_ok:
                stripped.append((tag, "rich_disabled"))
                continue
        elif tag not in CONTROL_TAGS:
            # 白名单外：正是「静默吞字」的来源，必须剥离。
            stripped.append((tag, "not_whitelisted"))
            continue
        elif tag not in allow:
            stripped.append((tag, "not_in_candidate_set"))
            continue
        # 位置：只允许句首（控制标签作用至下一个标签，句中插入会把一句切成两种语气）。
        if m.start() != 0:
            stripped.append((tag, "not_at_head"))
            continue
        if kept is not None:
            stripped.append((tag, "exceeds_max_count"))
            continue
        kept = m

    clean = _TAG_RE.sub("", text).strip()
    if kept is None:
        return TagResult(text=clean, text_tts=clean, applied=None, stripped=stripped)
    tag = kept.group(1).strip()
    return TagResult(text=clean, text_tts=f"[{tag}]{clean}", applied=tag, stripped=stripped)


def _validate_library() -> None:
    """启动即校验标签库自洽（与 emotion._validate_library 同机制）。"""
    assert len(CONTROL_TAGS) == 23, f"控制类标签应为 23 个，实为 {len(CONTROL_TAGS)}"
    assert len(RICH_TAGS) == 7, f"富语言类标签应为 7 个，实为 {len(RICH_TAGS)}"
    assert not (CONTROL_TAGS & RICH_TAGS), "两类标签不得重叠"
    for state, tag in STATE_TAG_ALLOWED.items():
        assert tag is None or tag in CONTROL_TAGS, f"{state} 的允许标签 {tag!r} 不在控制类清单内"
    # 期望值按 dict 定义序写（不是字母序），否则改表时容易误判。
    non_empty = [s for s, t in STATE_TAG_ALLOWED.items() if t]
    assert non_empty == ["displeased", "urgent", "doubtful", "hushed"], (
        f"非空标签项应恒为 4 个负面态（cooperative 仅 instruction 文本控制），实为 {non_empty}"
    )


_validate_library()
