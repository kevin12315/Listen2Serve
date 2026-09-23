"""收尾轮确定性检查（v7 收尾规则的可测化；零 API 成本，可重算历史批次）。

为什么要确定性而不是交给裁判：收尾轮是**对话模拟本身是否成立**的问题，不是策略优劣
问题——用户已明说要挂断，客服还在追问，这条轨迹作为「一通电话」就是残缺的。这类缺陷
必须能在任意历史 run 上按同一口径重算，因此只用词面规则，不进 LLM。

口径（只看最后一轮）：
- 收尾信号 closing_signal：最后一轮用户是否在**宣告收尾**。判据取用户模拟器自己落盘的
  `end_call`（协议字段，语义即「本轮为道别收尾轮，其后不再发言」）——这是刺激侧真值，
  不是对措辞的猜测；仅当轨迹没有该字段（早期 run）才回落词面意图词表。
  未命中则本条不计入分母（closing_ok/closing_dangling 皆为 None）。
- 悬空 closing_dangling：客服最后一句把话轮又抛回给用户（含问号，或以 吗/呢/吧/对吗
  一类疑问尾收句）——即「用户要挂，客服还在问」。
- 道别 closing_farewell：客服最后一句出现道别用语。
- closing_ok = closing_farewell and not closing_dangling。

为什么不用词面词表当分母：在 step4 两批次（各 288 条）上实测，`end_call` 命中 288/288，
而词面词表只命中 147/142 —— **漏检约 50%**，且漏掉的正是用户没把「再见」说出口、
只靠语气与内容收尾的隐式收尾（对客服而言恰恰是难的那一半）。用词表当分母会把指标
算在偏易的一半样本上。
"""

from __future__ import annotations

import re
from typing import Any

# 用户收尾意图（词面）：**仅供无 end_call 字段的早期 run 回落**。明说挂断/宣告无事/道别；
# 不含「好的/嗯」这类单纯应答，那属于声音侧的收尾意向，词面不可判。
_USER_CLOSING = (
    "先这样", "就这样", "没别的事", "没别的了", "没其他事", "没什么事", "没事了",
    "我挂了", "先挂了", "那我挂", "挂电话", "不聊了", "不说了", "再见", "拜拜",
    "知道了就这", "行了就这",
)

# 客服道别用语
_AGENT_FAREWELL = (
    "再见", "拜拜", "祝您", "祝你", "感谢您", "谢谢您的", "谢谢你的",
    "祝生活愉快", "祝顺利", "有需要随时", "再联系", "慢走",
)

# 疑问尾：句末（允许尾随标点/语气词）出现的疑问标记
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
