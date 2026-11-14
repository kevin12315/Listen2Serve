"""Deterministic final-turn checks for farewell and dangling questions.

The recorded end_call flag defines the closing-turn denominator. When the flag
is absent, explicit farewell phrases provide a text-based fallback.
"""

from __future__ import annotations

import re
from typing import Any

_USER_CLOSING = (
    "先这样",
    "就这样",
    "没别的事",
    "没别的了",
    "没其他事",
    "没什么事",
    "没事了",
    "我挂了",
    "先挂了",
    "那我挂",
    "挂电话",
    "不聊了",
    "不说了",
    "再见",
    "拜拜",
    "知道了就这",
    "行了就这",
)


_AGENT_FAREWELL = (
    "再见",
    "拜拜",
    "祝您",
    "祝你",
    "感谢您",
    "谢谢您的",
    "谢谢你的",
    "祝生活愉快",
    "祝顺利",
    "有需要随时",
    "再联系",
    "慢走",
)


_TAIL_QUESTION = re.compile(r"(吗|呢|吧|对吗|对吧|好吗|好吧|可以吗|行吗)[\s。，,.!！~]*$")


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def _is_question(text: str) -> bool:
    """客服末句是否把话轮抛回：含问号，或以疑问语气词收句。"""
    if "？" in text or "?" in text:
        return True
    return bool(_TAIL_QUESTION.search(text.strip()))


def check_closing(traj: dict[str, Any]) -> dict[str, Any]:
    """对单条轨迹做收尾轮检查。

    Returns:
        dict：closing_signal / closing_dangling / closing_farewell / closing_ok
        / closing_last_agent_text。无收尾信号时后三项为 None（不进分母）。
    """
    turns = traj.get("turns") or []
    last = turns[-1] if turns else {}
    user_text = (last.get("user_text") or "").strip()
    agent_text = (last.get("agent_text") or "").strip()
    if "end_call" in last:
        signal = bool(last["end_call"])
    else:
        signal = bool(user_text) and _has_any(user_text, _USER_CLOSING)
    out: dict[str, Any] = {
        "closing_signal": signal,
        "closing_last_agent_text": agent_text,
        "closing_dangling": None,
        "closing_farewell": None,
        "closing_ok": None,
    }
    if not signal:
        return out
    dangling = _is_question(agent_text)
    farewell = _has_any(agent_text, _AGENT_FAREWELL)
    out["closing_dangling"] = dangling
    out["closing_farewell"] = farewell
    out["closing_ok"] = farewell and not dangling
    return out
