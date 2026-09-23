"""情绪状态底色（内部方案 v6.0：11 态 39 标签 → 6 态；内部方案：收缩为生成期种子）。

单一来源原则（F2 硬约束）：
- 中文状态标签一律 import `domains.base.STATE_LABELS`，本模块不复制副本；
- T1 词表关联一律引用 `vocab.NEG_BY_STATE / POS_BY_STATE`，本模块不复制词表。

内部方案 瘦身（本模块退出运行期）：
- 语气定性词候选池（6 态 × 10 词）退役：它唯一的消费者是已删除的 tts.tone 槽位。
  「每轮从 10 词里抽一个拼进 prompt」使同一条目每次跑出不同文本（不可重现），
  而细腻差异现已改由 144 base × 2–3 态的 instruction 文本承载；
- 强度三档韵律表退役：随 key_event.intensity 字段一同移除（）；
- tts_control_for 的透明度分支退役：按透明度减弱韵律直接违反「文本透明度递减而
  情绪强度恒定」不变量（）；函数本体保留，但**只允许生成期脚本**当作撰写
  instruction 的种子句式调用，运行期路径 0 引用（由源码级单测断言守住）。

韵律硬约束（语速≥中等、音量≥正常、句间停顿短；hushed 态豁免音量条款、改用「音量
压低」）本身未变，现在由生成期的 scripts/validate_tts_contract.py 逐条把守。
"""

from __future__ import annotations

# F2：中文状态标签唯一来源 domains.base.STATE_LABELS（严禁复制副本）
from listen2serve.domains.base import STATE_LABELS

# ---- canonical 状态情绪底色（只写语气定性、不写韵律词）----
# 内部方案 审阅修正：值内不再自带「语气」二字。旧值形如「自然平实，语气平稳」，经
# tts_control_for 模板 f"{role}，语气{emotion}…" 拼装后产出「语气自然平实，语气平稳」
# 的自我重复 instruct（K18 演绎侧审阅 P3）；改为「定性词、定性词」并列式。
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
    """状态情绪底色的种子句式（内部方案：**仅限生成期脚本**调用）。

    运行期的 instruct 一律按 base_scenario_id × state 查预生成表，不再拼模板；本函数
    只作为人工撰写 instruction 文本时的句式参考（因此也不再接 level 参数：按透明度
    改韵律恰好是需要被除掉的那个行为）。输出满足 ≤80 字符与语速词/音量词可提取契约。
    """
    emotion = STATE_EMOTION.get(state, STATE_EMOTION["neutral"])
    prosody = _SEED_PROSODY_HUSHED if state == "hushed" else _SEED_PROSODY
    return f"{role}，语气{emotion}，情绪克制自然，{prosody}"


def _validate_library() -> None:
    """导入时校验（F2 单一来源）：状态码与 STATE_LABELS 逐一对应，违规即阻断管线。"""
    assert set(STATE_EMOTION) == set(STATE_LABELS), "STATE_EMOTION 与 STATE_LABELS 状态码不一致"


_validate_library()
