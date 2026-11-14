"`user_script.mission` 的读取口径"

from __future__ import annotations

from typing import Any

# mission 关键事件条目的文本前缀（数据面唯一标记，与 `key_event: true` 冗余互校）。
MISSION_KEY_TAG = "【关键事件】"

DEFAULT_STATE = "neutral"


def mission_entries(user_script: dict[str, Any]) -> list[dict[str, Any]]:
    """归一后的 mission 条目列表（每项必有 beat 键；不改原对象）。"""
    return [dict(m) if isinstance(m, dict) else {"beat": str(m)} for m in (user_script.get("mission") or [])]


def mission_beats(user_script: dict[str, Any]) -> list[str]:
    """各条 beat 文本（含 `【关键事件】` 前缀，逐字不变）。"""
    return [str(e.get("beat") or "") for e in mission_entries(user_script)]


def mission_states(user_script: dict[str, Any]) -> list[str]:
    """各条 beat 的 state，省略项补 `neutral`（规则 1）。"""
    return [str(e.get("state") or DEFAULT_STATE) for e in mission_entries(user_script)]


def key_beat_index(user_script: dict[str, Any]) -> int:
    "关键事件条目下标（0 基）；找不到返回 -1"
    entries = mission_entries(user_script)
    for i, e in enumerate(entries):
        if e.get("key_event"):
            return i
    for i, e in enumerate(entries):
        if str(e.get("beat") or "").startswith(MISSION_KEY_TAG):
            return i
    return -1


def is_key_beat(entry: dict[str, Any] | str) -> bool:
    """单条 mission 条目是否为关键事件条（两种格式通吃）。"""
    if isinstance(entry, str):
        return entry.startswith(MISSION_KEY_TAG)
    return bool(entry.get("key_event")) or str(entry.get("beat") or "").startswith(MISSION_KEY_TAG)
