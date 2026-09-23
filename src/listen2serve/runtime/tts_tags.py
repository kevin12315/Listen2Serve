"""TTS 情感标签白名单与四重校验。

标签是独立于 instruction 文本的第二条韵律通道：instruction 描述整句基调，标签在句首
切一次具体情绪。两者可叠加，但标签必须被严格框住——实测表明写错标签时后端不报错、
不念字面，而是**吞掉紧随其后的第一个字**（`[urgent]` 使句首「哦」消失），属于静默
污染音频，靠事后听测很难发现。

校验四重（顺序执行）：白名单 / 能力门控（富语言标签仅 plus/flash 支持，内部方案 决策 d 下
一律剥离）→ 本 base 允许集 → 位置（句首）→ 数量（≤1）。违规一律**剥离**而非抛错：
与 state 不同，标签缺失时台词依然完整可合成，剥离后回到的是合法的默认态（无标签），
不是猜出来的值；剥离原因全部计数上报。

本模块只提供常量与纯函数，不含 I/O，供运行期与生成期共用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 官方控制类标签（23 个）：作用于其后所有文本，直到下一个控制标签或长句被切分。
CONTROL_TAGS: frozenset[str] = frozenset({
    "sad", "amazed", "trembling", "angry", "excited", "sarcastic", "curious",
    "bored", "tired", "scornful", "shouting", "deep and loud shouting", "asmr",
    "panicked", "mischievously", "empathetic", "whispers", "reluctantly",
    "crying", "serious", "very slowly", "very fast", "like dracula",
})

# 官方富语言类标签（7 个）：在当前位置插一段拟声，不影响前后风格。
# 内部方案 决策 d：暂不启用（仅 qwen-audio-3.0-tts-plus / flash 支持，会造成跨后端不可比）。
RICH_TAGS: frozenset[str] = frozenset({
    "gasp", "sighing", "clears throat", "giggles", "laughing", "cough", "snorts",
})

# 支持富语言标签的模型（能力门控用）。
RICH_TAG_MODELS: frozenset[str] = frozenset({
    "qwen-audio-3.0-tts-plus", "qwen-audio-3.0-tts-flash",
})

# state → 该态允许的唯一标签（全局硬约束，）。值为裸标签名，方括号在合成前由
# apply_tag_policy 拼上；这样全链路（表、校验、剥离）用同一种表示，不存在去括号的边界。
#
# 听测一度把非空项收敛到 2 个（只有 hushed/displeased 有标签）；Step2（v6.4）按人耳
# 裁决加回 doubtful，v6.6 按 实测加回 urgent —— 判据是「模型能否听出 / 声学上量到操纵」
# 而非声学闸门（内部方案 修订 4）。至此 4 个负面态全部有标签、cooperative 仍只走 instruction。
# null 态下仍不存在“LLM 可选加标签”的空间，韵律的唯一输入回到 state 本身（原则 1）。
STATE_TAG_ALLOWED: dict[str, str | None] = {
    "neutral": None,
    # 2026-09-02 用户裁决：cooperative 不再用 [excited] 标签（催收/客服场景里"带着笑意"语用突兀），
    # 改为仅 instruction 文本控制（"语气平和爽快、积极配合"）。标签通道只保留 4 个负面态。
    "cooperative": None,
    "displeased": "angry", # Step2（v6.4）：模型金标准 87.5–100% + 人耳裁决；抽身类 9 base 的 reluctantly 退役（声学方向错）
    # 2026-09-05 用户裁决：urgent 由「无标签」改为 [very fast]。旧注释写的「两后端均无法可靠
    # 编码语速」说的是 instruct-only 通道 —— planT 实测 instruct 的语速 Δ=−0.091 字/秒（p=0.784）
    # 等于零操纵，内部方案 因此在脚本侧覆写成 [very fast] 并量到语速 +37.5%（载体句）/ +27.9%（台词）。
    # 生产表原先没跟上这次改动，导致 内部方案 的刺激口径与生产表不一致；此处把生产表对齐 内部方案。
    "urgent": "very fast",
    "doubtful": "curious", # Step2（v6.4）：人耳裁决标签胜出（模型判别也仅标签通道显著）
    "hushed": "whispers", # 压声类 5 base（表驱动强制），人耳复核确认优于 instruction-v2
}

MAX_TAGS_PER_UTTERANCE = 1

_TAG_RE = re.compile(r"\[([^\[\]]{1,32})\]")


@dataclass
class TagResult:
    """标签校验结果：text 供评委/ASR 对齐，text_tts 实际送 TTS。"""

    text: str
    text_tts: str
    applied: str | None = None
    stripped: list[tuple[str, str]] = field(default_factory=list) # (tag, reason)

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
    """本 base 的标签允许集 = instruction 表里声明的强制标签（去重保序）。

    不再由 candidate_states 推导：同一个 hushed 内部就分压声类（需 [whispers]）与抽身类
    （不能加）两种语用（a），“按 state 查代码常量表”这个粒度表达不了；强制标签
    住在表里（），代码只守“一态一标签”这个全局不变量。Step2（v6.4）口径下非空的
    是 41 个 base（按「态 × 动态」语用适配挂 key_tag），其余 103 个为空列表。
    """
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
    """按四重校验裁剪台词中的标签，返回干净文本与实际送 TTS 文本。

    allowed 为本 base 的标签候选集；enable_rich 为 False 时富语言标签一律剥离
    （内部方案 决策 d）。
    """
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
    assert non_empty == ["displeased", "urgent", "doubtful", "hushed"], \
        f"非空标签项应恒为 4 个负面态（cooperative 仅 instruction 文本控制），实为 {non_empty}"


_validate_library()
