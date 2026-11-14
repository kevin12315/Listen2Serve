"""Bound affirmative balance-clearing claims in source-written partial plans.

This narrow guard reads source facts and mission beats, never conversation
history. It leaves questions, quotations and conditional negotiations unchanged.
"""

from __future__ import annotations

import math
import re
from typing import Any

_REST = re.compile(r"剩下|剩余|余下|余款|尾款|其余")
_CLEAR = re.compile(r"补齐|补上|还清|还完|结清|付清|交清|清偿|全部还|全额还")
_TIME = re.compile(r"周[一二三四五六日天]|星期[一二三四五六日天]|今天|明天|今晚|月底|月末|月初|晚点|稍后")
_NEGATIVE = re.compile(r"不(?:能|会|是|保证|确定|承诺)?|没(?:有|法|定)?|尚未|无法|难以|未必")


def _amount(value: Any) -> float | None:
    """Read only an explicit source amount, including ordinary Chinese integers."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) and value > 0 else None
    text = re.sub(r"[,，\s]|(?:元|块钱|块)$", "", str(value or ""))
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    if not re.fullmatch(r"[零〇一二两三四五六七八九十百千万]+", text):
        return None
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9), strict=True))
    total = section = number = 0
    for char in text:
        if char in digits:
            number = digits[char]
        elif char == "万":
            total += (section + number) * 10000
            section = number = 0
        else:
            section += (number or 1) * {"十": 10, "百": 100, "千": 1000}[char]
            number = 0
    return float(total + section + number) or None


def _clauses(text: str):
    # Quoted agent words are not customer commitments. A separate unquoted
    # assertion remains visible and can still be checked.
    text = re.sub(r'“[^”]*”|「[^」]*」|"[^"\n]*"', "", text)
    for sentence in re.split(r"(?<=[。！!？?])|\n", text):
        if re.search(r"如果|假如|要是|(?:^|，|,)若", sentence):
            continue
        if re.search(
            r"(?:^|[，,])(?:这个(?:安排|方案)|该(?:安排|方案)|这样(?:安排|还款|补缴)?)?"
            r"(?:行不行|可不可以|可以吗|行吗|可行吗|能否这样(?:安排|还款)?|能不能这样(?:安排|还款)?)"
            r"\s*[？?]\s*$",
            sentence,
        ):
            continue
        for clause in re.split(r"[，,；;]", sentence):
            if re.search(r"[？?]|吗|会不会|能否|怎么|如何|是否|是不是|行不行", clause):
                continue
            if re.search(r"(?:你们|你|您|客服|对方)(?:刚才|刚|之前)?(?:说|提到|要求|建议|让我|叫我)", clause):
                continue
            if re.search(
                r"希望|想.{0,14}(?:还清|结清|补齐|补上)|争取|尽量|未确认|还没.{0,4}(?:确认|定)|再.{0,2}(?:商量|安排|定)",
                clause,
            ):
                continue
            yield clause


def _source_balance_plans(facts: dict[str, Any], mission: list[str]) -> list[str]:
    # A generic goal to pay the debt, a cash-receipt date, or an agent repeating
    # an invention cannot authorize a balance plan. Require a source-written
    # remainder payment action. Existing POS68 explicitly has such a beat.
    sources = [str(value) for value in facts.values()] + mission
    return [
        clause
        for source in sources
        for clause in _clauses(source)
        if _REST.search(clause)
        and re.search(r"补|还(?:款|清|完|上)?|结清|付清|交清", clause)
        and not _NEGATIVE.search(clause)
    ]


def unsupported_repayment_completion(
    text: str, facts: dict[str, Any], mission: list[str], db_seed: dict[str, Any] | None = None
) -> list[str]:
    """Reject a newly invented definite clearing plan when only partial pay is given.

    Source-provided remainder plans stay available; literal time anchors stop a
    different explicit day from borrowing their authorization. Arbitrary date
    interpretation and general paraphrase classification are outside this check.
    """
    planned, debt = _amount(facts.get("planned_amount")), _amount(facts.get("debt_amount"))
    if debt is None:
        customers = (db_seed or {}).get("customers") or []
        if customers and isinstance(customers[0], (list, tuple)) and len(customers[0]) > 1:
            debt = _amount(customers[0][1])
    if planned is None or debt is None or planned >= debt:
        return []
    authorized = _source_balance_plans(facts, mission)
    for clause in _clauses(text):
        action = _CLEAR.search(clause)
        if action is None:
            continue
        prefix = clause[: action.start()]
        if _NEGATIVE.search(prefix):
            continue
        rest = _REST.search(clause)
        full = re.search(r"全(?:部|额)|整笔|一次性|一口气|(?:这笔|欠款).{0,8}(?:还清|结清|还完)", clause)
        if rest is None and full is None:
            continue
        if (
            rest is None
            and re.search(r"这部分|首期|这一期|第一期|本期", clause)
            and not re.search(r"(?:全部|所有|整笔)欠款", clause)
        ):
            continue
        amounts = [
            _amount(token)
            for token in re.findall(r"\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+", clause)
        ]
        if (
            rest is None
            and planned in amounts
            and debt not in amounts
            and not re.search(r"(?:全部|所有|整笔)欠款", clause)
        ):
            continue  # Clearing the explicitly stated installment is allowed.
        times = set(_TIME.findall(clause))
        if any(not times or times <= set(_TIME.findall(plan)) for plan in authorized):
            continue
        return [
            (
                "源只给出部分还款金额与日期，不能新增余款补齐或整笔结清承诺；"
                "资金回款时间不是余款清偿计划，保留拟还安排，余款仅按源明确计划回应或说明待协商"
            )
        ]
    return []
