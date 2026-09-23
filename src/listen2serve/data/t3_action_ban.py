"""T3 关键轮「动作语用」禁止词表的唯一读取口。

数据住在 `data/benchmark/t3_action_ban.json`，本模块只做三件事：读、按 tier 取、渲染。
**不在这里再抄一份词表** —— 那份 JSON 是三份历史副本（`内部净化脚本` 的
`改写 prompt` 规则 1、`内部检测脚本` 的 `动作泄露检测词表`、
`内部台词审计脚本` 的 `台词审计 prompt` leak 例子）逐条合并的结果，
合并前的原貌冻结在 `data/benchmark/validation/t3_action_ban_sources_pre_merge.json`，
「有没有丢条目」由 `tests/test_t3_action_ban.py` 拿那份快照比对，不靠人记。

四个 tier 的语义差别很大，用错就等于把闸门放宽（内部方案 纪律 3）：

  ban 生成时不许出现；第一关命中即排进优先审队列
  screen_only 只配排队，误报率高到不能参与判定（改 / 换 / 短 / 期限 …）
  rejected 逐条判后剔除但留档（例：doubtful 的「确认」——它正是 T3 规定句式的动词）
  candidate 内部方案 按实证漏检提出的新增条目，**未经决策记录，默认不生效**

动作文本（rubric）不在本模块，一律从 `listen2serve.domains.base.ACTION_V9` 取 ——
那是 内部方案 收口的 state→动作唯一真值源，第二关审计与第三关 C0 必须共用同一个动作定义，
否则「审的是 A 动作、判的是 B 动作」，两关的读数对不上却看不出哪里错。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BAN_PATH = ROOT / "data" / "benchmark" / "t3_action_ban.json"
SOURCES_SNAPSHOT = ROOT / "data" / "benchmark" / "validation" / "t3_action_ban_sources_pre_merge.json"

TIERS = ("ban", "screen_only", "rejected", "candidate")
# 默认只吃 ban + screen_only：前者是禁止项，后者是排队线索。
# rejected / candidate 一律要显式点名才进匹配面，避免"顺手全都要"把已剔除的词又放回来。
DEFAULT_SCREEN_TIERS = ("ban", "screen_only")
AXES = ("action", "T", "T+action")


@lru_cache(maxsize=1)
def load() -> dict:
    """读整份词表（含 provenance 与冲突裁定记录）。"""
    return json.loads(BAN_PATH.read_text(encoding="utf-8"))


def states() -> list[str]:
    """五个态，顺序固定为 JSON 里的定义序（hushed/doubtful/cooperative/displeased/urgent）。"""
    return list(load()["states"])


def entries(state: str, tiers: tuple[str, ...] = ("ban",)) -> list[dict]:
    """取某态某几个 tier 的条目原文（含 from / axis / maps_to_v9 / note）。"""
    if state not in load()["states"]:
        raise KeyError(f"未知态 {state}；本表覆盖 {states()}")
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        raise ValueError(f"未知 tier {unknown}；合法值 {TIERS}")
    return [e for e in load()["states"][state]["entries"] if e["tier"] in tiers]


def terms(state: str, tiers: tuple[str, ...] = ("ban",)) -> list[str]:
    """取某态某几个 tier 的词面（去重、保序）。"""
    seen: list[str] = []
    for e in entries(state, tiers):
        if e["term"] not in seen:
            seen.append(e["term"])
    return seen


def screen_hits(state: str, text: str,
                tiers: tuple[str, ...] = DEFAULT_SCREEN_TIERS) -> dict[str, list[str]]:
    """第一关（0 成本预筛）：按 tier 分桶返回命中的词。

    ⚠ 返回值只用来**排队**，不用来判定。命中 ban 不等于泄露，
    未命中也不等于干净 —— 最终判据是第三关的 C0 行为门。
    rejected / candidate 默认不参与匹配，但传进来就会照算，方便报告里把
    「被剔除的词本来会命中多少条」摊开给人看。
    """
    out: dict[str, list[str]] = {t: [] for t in tiers}
    for t in tiers:
        out[t] = [w for w in terms(state, (t,)) if w in (text or "")]
    return out


def axis_note(state: str) -> str:
    """该态的轴归属说明（displeased 有一条整态级的裁定，读它别读 entries）。"""
    return str(load()["states"][state].get("axis_note", ""))


def v9_action_summary(state: str) -> str:
    """从 `domains.base.ACTION_V9` 现场渲染该态的动作定义（不复制文本）。"""
    from listen2serve.domains.base import ACTION_V9 # 延迟 import：避免 data↔domains 环

    if state not in ACTION_V9:
        raise KeyError(f"ACTION_V9 里没有 {state}")
    a = ACTION_V9[state]
    return f"opening={a['opening']}；follow={a['follow']}"


def ban_clause(states_: list[str] | None = None) -> str:
    """把 ban 层渲染成 `改写 prompt` 规则 1 那种一句话逐态禁止词表。

    存在的理由：`内部净化脚本` 的规则 1 原本把这份表**手写**在 prompt 字符串里，
    是三个分叉源头之一。改成调用本函数后，prompt 里的词表与 JSON 逐字同源。
    ⚠ 内部方案 实测「例词清单被演绎模型当成推荐模板」（填充词起头率 24%→92%），
    所以这段**不要**塞进演绎 prompt；消费点是改写器与本仓的代码侧词表防线。
    """
    todo = states_ or states()
    parts = [f"{s} 不许提" + "/".join(terms(s, ("ban",))) for s in todo]
    return "；".join(parts)


def coverage() -> dict[str, dict[str, int]]:
    """每态 × 每 tier 的条数，供报告与 CI 用（数字从数据现算，不写死）。"""
    return {s: {t: len(terms(s, (t,))) for t in TIERS} for s in states()}


def conflicts() -> list[dict]:
    """合并时的冲突裁定记录（C1–C6）。内部方案 要求逐条列进报告。"""
    return list(load()["conflicts"])
