"""Strict parsing shared by the published judges."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import Any


class JudgeResponseError(ValueError):
    """A judge returned an incomplete or invalid verdict."""


def json_object(text: str) -> dict[str, Any]:
    """Accept a JSON object, optionally enclosed in a Markdown code fence."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise JudgeResponseError("Judge response must be valid JSON") from exc
    if not isinstance(data, dict):
        raise JudgeResponseError("Judge response must be a JSON object")
    return data


def integer(data: dict, key: str, low: int, high: int) -> int:
    value = data.get(key)
    if type(value) is not int or not low <= value <= high:
        raise JudgeResponseError(f"{key} must be an integer in [{low}, {high}]")
    return value


def string(data: dict, key: str, choices: tuple[str, ...] | None = None) -> str:
    value = data.get(key)
    if not isinstance(value, str) or (choices is not None and value not in choices):
        raise JudgeResponseError(f"Invalid or missing {key}")
    return value


def action_items(data: dict, key: str, expected: list[str] | None = None) -> list[dict]:
    items = data.get(key)
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise JudgeResponseError(f"{key} must be a list of objects")
    actions = [string(item, "action") for item in items]
    if expected is not None and Counter(actions) != Counter(expected):
        raise JudgeResponseError(f"{key} must cover every supplied action exactly once")
    return items


_QUOTED_EVIDENCE = re.compile(
    r'“([^”]+)”|‘([^’]+)’|「([^」]+)」|『([^』]+)』|"([^"]+)"|`([^`]+)`'
    r"|(?<![A-Za-z0-9])'((?:[^']|'(?=[A-Za-z]))*?)'(?![A-Za-z0-9])"
)
_ELLIPSIS = re.compile(r"(?:\.{3,}|…+)")
_EVIDENCE_LABEL = re.compile(
    r"^(?:(?:第\s*\d+\s*轮)\s*)?(?:客服(?:回复|发言)?|用户(?:表态|回复)?|证据|依据|原文)\s*[:：]\s*"
)
_NO_EVIDENCE = frozenset(
    {"", "无", "无证据", "无依据", "无相关证据", "无相关依据", "无回复", "未回复", "没有回复", "none", "na"}
)


def _normalize_quote(text: str) -> str:
    """Ignore presentation differences while preserving numeric punctuation and units."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    chars = []
    for i, char in enumerate(normalized):
        if char.isspace():
            continue
        if unicodedata.category(char).startswith("P"):
            adjacent_digit = (i > 0 and normalized[i - 1].isdigit()) or (
                i + 1 < len(normalized) and normalized[i + 1].isdigit()
            )
            if char not in ".,:/+-％%" or not adjacent_digit:
                continue
        chars.append(char)
    return "".join(chars)


def _numeric_boundaries_match(fragment: str, source: str, position: int) -> bool:
    """A quotation cannot take only the trailing or leading digits of a number."""
    before = source[position - 1] if position else ""
    after_pos = position + len(fragment)
    after = source[after_pos] if after_pos < len(source) else ""
    if fragment[0].isdigit() and before:
        if before.isdigit() or before in "+-":
            return False
        if before in ".," and position > 1 and source[position - 2].isdigit():
            return False
    if fragment[-1].isdigit() and after:
        if after.isdigit():
            return False
        if after in ".," and after_pos + 1 < len(source) and source[after_pos + 1].isdigit():
            return False
    return True


def _quote_matches(quote: str, sources: list[str]) -> bool:
    """An omitted span may separate ordered verbatim fragments within an utterance."""
    fragments = [_normalize_quote(part) for part in _ELLIPSIS.split(quote)]
    fragments = [part for part in fragments if part]
    if not fragments:
        return False
    for source in sources:
        offset = 0
        for fragment in fragments:
            position = source.find(fragment, offset)
            while position >= 0 and not _numeric_boundaries_match(fragment, source, position):
                position = source.find(fragment, position + 1)
            if position < 0:
                break
            offset = position + len(fragment)
        else:
            return True
    return False


def grounded_evidence(evidence: str, sources: list[str], *, required: bool, field: str) -> None:
    """Check quotation provenance, not whether the quoted behavior deserves a score.

    Accept whitespace/punctuation changes, ellipses, and commentary around explicitly
    quoted excerpts. Every excerpt must occur in an allowed source utterance; loose
    lexical overlap or a paraphrase is insufficient to establish provenance.
    """
    text = _EVIDENCE_LABEL.sub("", evidence.strip())
    if _normalize_quote(text) in _NO_EVIDENCE:
        if required:
            raise JudgeResponseError(f"{field} requires evidence from the supplied text")
        return
    quoted = [
        next(value for value in match.groups() if value is not None)
        for match in _QUOTED_EVIDENCE.finditer(text)
    ]
    excerpts = quoted or [text]
    normalized_sources = [_normalize_quote(source) for source in sources if source.strip()]
    if any(not _quote_matches(excerpt, normalized_sources) for excerpt in excerpts):
        raise JudgeResponseError(f"{field} contains a quotation absent from the allowed evidence window")
