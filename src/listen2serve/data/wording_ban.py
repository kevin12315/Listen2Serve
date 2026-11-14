"T3 关键轮「动作语用」禁止词表的唯一读取口"

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BAN_PATH = ROOT / "data" / "benchmark" / "neutral_wording_ban.json"

TIERS = ("ban", "screen_only", "rejected", "candidate")
# 默认只吃 ban + screen_only：前者是禁止项，后者是排队线索。
# rejected / candidate 一律要显式点名才进匹配面，避免"顺手全都要"把已剔除的词又放回来。
DEFAULT_SCREEN_TIERS = ("ban", "screen_only")
AXES = ("action", "T", "T+action")

# Match a self-declaration, not a task, quotation, or question about cooperation.
_COOPERATION_DECLARATION = re.compile(
    r"(?:^|[，,。.!！?？;；:：\n])\s*"
    r"(?P<declaration>(?:这边)?我(?:这边)?"
    r"(?:现在|也|都|一定|肯定|会|可以|愿意|很|挺|积极|完全|全力|不|不会|不愿意|无法|没法){0,2}"
    r"配合(?:你们|你)?(?:的|啊|呀|呢|了)?)"
    r"(?=\s*(?:$|[，,。.!！?？;；:：]))"
)


def explicit_state_declaration_hits(text: str) -> list[str]:
    """Find standalone first-person cooperation declarations missed by the lexicon.

    Concrete tasks (我配合你核对材料), conditional requests and ordinary agreement
    remain outside this narrow check. It supplements the reviewed vocabulary;
    the source screening table and its tier semantics stay unchanged.
    """
    return list(dict.fromkeys(m["declaration"] for m in _COOPERATION_DECLARATION.finditer(text)))


_QUOTED_TEXT = re.compile(r'“[^”]*”|‘[^’]*’|"[^"\n]*"|\'[^\'\n]*\'')
_BUSINESS_TOPIC = (
    r"(?:(?:(?:眼下|当前|目前)?(?:这|那)(?:个|项)?|(?:眼下|当前|目前)的)?"
    r"(?:还款方案|方案|安排|要求|条件|还款计划))"
)
_KEY_STATE_DECLARATION = re.compile(
    r"(?P<declaration>(?:" + _BUSINESS_TOPIC + r")?我(?:这边)?(?:现在|目前|也|就是|实在|真的)?"
    r"(?:(?:不接受|不同意)(?:"
    + _BUSINESS_TOPIC
    + r")?|(?:等不了|没法等|不能等)(?:这么久|那么久|这么长时间|那么长时间)?)"
    r"(?:的|啊|呀|呢|了|吧)?)"
)


def explicit_key_state_declaration_hits(text: str) -> list[str]:
    """Find the observed refusal/waiting declarations only for neutral key turns.

    Questions, quoted text, conditional statements and business actions remain
    outside this narrow check. Runtime callers must keep it scoped to the key
    turn: a later refusal or request to end a call can be part of the script.
    """
    unquoted = _QUOTED_TEXT.sub(lambda match: " " * len(match.group()), text)
    hits = []
    for sentence in re.findall(r"[^。.!！?？;；\n]+[。.!！?？;；\n]?", unquoted):
        if sentence.endswith(("?", "？")) or re.search(r"如果|假如|要是|倘若|只要|除非", sentence):
            continue
        for clause in re.split(r"[，,：:]", sentence.rstrip("。.!！;；\n")):
            match = _KEY_STATE_DECLARATION.fullmatch(clause.strip())
            if match and match["declaration"] not in hits:
                hits.append(match["declaration"])
    return hits


@lru_cache(maxsize=1)
def load() -> dict:
    "Load."
    return json.loads(BAN_PATH.read_text(encoding="utf-8"))


def states() -> list[str]:
    """五个态，顺序固定为 JSON 里的定义序（hushed/doubtful/cooperative/displeased/urgent）。"""
    return list(load()["states"])


def entries(state: str, tiers: tuple[str, ...] = ("ban",)) -> list[dict]:
    "取某态某几个 tier 的条目原文（含 from / axis / maps_to_ / note）"
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


def screen_hits(state: str, text: str, tiers: tuple[str, ...] = DEFAULT_SCREEN_TIERS) -> dict[str, list[str]]:
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
    "Axis note."
    return str(load()["states"][state].get("axis_note", ""))


def action_summary(state: str) -> str:
    """从 `domains.base.STATE_ACTIONS` 现场渲染该态的动作定义（不复制文本）。"""
    from listen2serve.domains.base import (
        STATE_ACTIONS,
    )  # 延迟 import：避免 data↔domains 环

    if state not in STATE_ACTIONS:
        raise KeyError(f"STATE_ACTIONS 里没有 {state}")
    a = STATE_ACTIONS[state]
    return f"opening={a['opening']}；follow={a['follow']}"


def ban_clause(states_: list[str] | None = None) -> str:
    "把 ban 层渲染成 `改写 prompt` 规则 1 那种一句话逐态禁止词表"
    todo = states_ or states()
    parts = [f"{s} 不许提" + "/".join(terms(s, ("ban",))) for s in todo]
    return "；".join(parts)


def coverage() -> dict[str, dict[str, int]]:
    """每态 × 每 tier 的条数，供报告与 CI 用（数字从数据现算，不写死）。"""
    return {s: {t: len(terms(s, (t,))) for t in TIERS} for s in states()}


def conflicts() -> list[dict]:
    "Conflicts."
    return list(load()["conflicts"])
