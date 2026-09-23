"""FlowJudge：全对话裁判（内部方案 v2.3-J，）。

单次 LLM 调用合并产出（控 token）：
- flow_items：数据侧固定清单逐条判定（hit 是/部分/否 + 命中轮次 + 证据），
  裁判不得增删条目；清单源 = 场景键 required_actions_effective（v6.1 条件化，
  ），缺失时回落 agent_policy.required_actions（与 v2.2-I 清单同源）；
- task_score：终局结果分（0-2 原始分；聚合层归一 0/0.5/1），双向场景锚点
  「转变后仍达成/调整目标」已写入 prompt 判据；TransitionScore 退役不再单独产出。

FlowRate = flow_items 命中比例（"部分"计 0.5）。
替代 v2.2-I 的 DIALOGUE_TASK/TRANSITION 两个独立裁判（完整对话正文只进一次 prompt）。
复用 内部方案 并发/断点续评设施（调度在 cli 层，本模块只负责单场景判定）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from listen2serve.domains.base import DomainSpec, render_customer_profile, render_judge_contract
from listen2serve.evaluation.prompts import FLOW_JUDGE_PROMPT, FLOW_JUDGE_WINDOW_SUFFIX
from listen2serve.gateway.llm_gateway import LLMGateway, LLMResult


@dataclass
class FlowVerdict:
    """FlowJudge 判定结果（flow_items + FlowRate + task_score）。"""

    flow_items: list[dict[str, Any]] = field(default_factory=list)
    flow_rate: float = 0.0 # 命中比例（部分计 0.5）
    task_score_raw: int = 0 # 裁判原始 0-2 分
    task_score: float | None = 0.0 # 归一分（0/0.5/1）；截断窗模式下为 None（终局未发生）
    # ---- 截断观察窗（2026-09-11，给「会话寿命装不下一通完整对话」的端点留可比口径）----
    turn_cap: int | None = None # None = 整通口径（三家既有读数走的这条）
    n_items_in_window: int | None = None # 清单里"到第 K 轮为止该做"的条目数（= 窗口分母）
    flow_rate_in_window: float | None = None
    reason: str = ""
    parse_retried: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "flow_items": self.flow_items,
            "flow_rate": self.flow_rate,
            "task_score_raw": self.task_score_raw,
            "task_score": self.task_score,
            "turn_cap": self.turn_cap,
            "n_items_in_window": self.n_items_in_window,
            "flow_rate_in_window": self.flow_rate_in_window,
            "reason": self.reason,
            "parse_retried": self.parse_retried,
        }


def flow_checklist(scenario: dict[str, Any]) -> list[str]:
    """FlowJudge 固定清单（单一来源）：required_actions_effective（v6.1，
    条件化裁剪产物）优先；缺失回落 agent_policy.required_actions（v6.0 口径）。"""
    eff = scenario.get("required_actions_effective")
    if isinstance(eff, list) and eff:
        return [str(a) for a in eff]
    return [str(a) for a in (scenario.get("agent_policy") or {}).get("required_actions") or []]


class FlowJudge:
    """全对话裁判（文本 LLM）：过程清单 + 终局结果分一次产出。"""

    def __init__(self, llm: LLMGateway | None = None, llm_model: str | None = None) -> None:
        self.llm = llm or LLMGateway()
        self.llm_model = llm_model
        self.parse_retries = 0
        self.parse_failures = 0
        # 内部方案 token 经济性统计（验收）
        self.input_tokens = 0
        self.judge_calls = 0

    def judge(
        self,
        domain: DomainSpec,
        scenario: dict[str, Any],
        history: list[dict[str, str]],
        turn_cap: int | None = None,
    ) -> FlowVerdict:
        """整通判定（turn_cap=None，三家既有读数走的口径）或截断窗判定（turn_cap=K）。

        turn_cap 只截对话、不改清单：**四端点必须用同一个 K** 才可比，所以 K 由调用方
        （cli 的 --flow-turn-cap）统一给，不在这里按端点/按场景自适应——那样分母就随
        被测端点变化，等于把「谁走得远」又偷偷算回能力分。
        """
        checklist = flow_checklist(scenario)
        prompt = FLOW_JUDGE_PROMPT.format(
            checklist=_render_checklist(checklist),
            task_goal=scenario.get("task_goal") or _default_goal(scenario),
            policy_rules=_render_rules(domain, scenario),
            history=_render_history_turn_numbered(history, turn_cap),
        )
        if turn_cap:
            prompt += FLOW_JUDGE_WINDOW_SUFFIX.format(turn_cap=turn_cap)
        messages = [
            {"role": "system", "content": "你是客服全对话裁判，只输出 JSON，不要输出 markdown 代码块或其他内容。"},
            {"role": "user", "content": prompt},
        ]
        data: dict[str, Any] = {}
        retried = False
        for attempt in range(2):
            result: LLMResult = self.llm.chat(
                messages,
                model=self.llm_model,
                temperature=0.0,
                max_tokens=4096, # checklist 逐条输出较长，预留足够输出空间
            )
            self.input_tokens += result.prompt_tokens
            self.judge_calls += 1
            data = _extract_json(result.text)
            if data:
                break
            if attempt == 0:
                retried = True
                self.parse_retries += 1
        if not data:
            self.parse_failures += 1
        verdict = _parse_flow_json(data, checklist, turn_cap)
        verdict.parse_retried = retried
        return verdict


def _render_checklist(checklist: list[str]) -> str:
    return "\n".join(f"{i}. {a}" for i, a in enumerate(checklist, 1)) or "（无）"


def _render_history_turn_numbered(history: list[dict[str, str]],
                                  turn_cap: int | None = None) -> str:
    """完整对话渲染（标注轮号，供 flow_items 命中轮次标注）。

    turn_cap=K 时只保留前 K 轮：一轮 = 一条 user 起、到下一条 user 前为止（含其中的
    assistant 行）。**不能按下标切**——history 是 user/assistant 交替，按下标截会把
    某轮的客服发言截掉半句，而这正是本次要修的故障形态。
    """
    if turn_cap:
        kept: list[dict[str, str]] = []
        turn = 0
        for h in history:
            if h["role"] == "user":
                turn += 1
                if turn > turn_cap:
                    break
            kept.append(h)
        history = kept
    lines: list[str] = []
    turn = 0
    for h in history:
        if h["role"] == "user":
            turn += 1
            lines.append(f"[第{turn}轮] 用户: {h['content']}")
        else:
            lines.append(f"[第{turn}轮] 客服: {h['content']}")
    return "\n".join(lines) or "（无）"


def _render_rules(domain: DomainSpec, scenario: dict[str, Any]) -> str:
    """契约 + 同源客户档案（与 policy_judge/dialogue_metrics 同口径）。"""
    contract = render_judge_contract(domain, scenario)
    profile = render_customer_profile(domain, scenario.get("db_seed"))
    if not profile:
        return contract
    return f"{contract}\n\n{profile}"


def _default_goal(scenario: dict[str, Any]) -> str:
    role = scenario.get("role", "")
    goals = {
        "collection": "促成客户给出具体的还款承诺（金额+时间）",
        "marketing": "完成产品推介并促成转化或预约",
        "hotline": "解决用户问题并确认满意",
    }
    return goals.get(role, "完成客服任务目标")


def _parse_flow_json(data: dict[str, Any], checklist: list[str],
                     turn_cap: int | None = None) -> FlowVerdict:
    """解析 FlowJudge JSON：按固定清单对齐条目（防裁判改写成清单长度变化）。"""
    raw_items = data.get("flow_items") or []
    flow_items: list[dict[str, Any]] = []
    for idx, action in enumerate(checklist):
        it = raw_items[idx] if idx < len(raw_items) and isinstance(raw_items[idx], dict) else {}
        hit = str(it.get("hit", "否"))
        if hit not in ("是", "部分", "否"):
            hit = "否"
        turn = it.get("turn")
        try:
            turn = int(turn) if turn is not None else None
        except (TypeError, ValueError):
            turn = None
        in_window = bool(it.get("in_window", True))
        # 自洽校验（防比率被人为抬高）：窗口内已命中的条目**必然**在窗口内可判。
        # 裁判若把命中的条目报成 in_window=false，分母就会漏掉它的得分 ⇒ 比率虚高。
        if hit != "否":
            in_window = True
        flow_items.append({
            "action": action, # 以固定清单原文为准（裁判改写不采信）
            "hit": hit,
            "turn": turn,
            "evidence": str(it.get("evidence", "")),
            "in_window": in_window,
        })
    scores = {"是": 1.0, "部分": 0.5, "否": 0.0}
    flow_rate = round(sum(scores[it["hit"]] for it in flow_items) / len(flow_items), 4) if flow_items else 0.0
    n_win = sum(1 for it in flow_items if it["in_window"])
    flow_rate_in_window = (
        round(sum(scores[it["hit"]] for it in flow_items if it["in_window"]) / n_win, 4)
        if turn_cap and n_win else None)
    # 窗口内不给终局分：对话没结束，"任务是否达成"无从判定，记 0 会把端点故障记成能力差
    task_score_raw = 0 if turn_cap else _clamp_score(data.get("task_score", 0))
    return FlowVerdict(
        flow_items=flow_items,
        flow_rate=flow_rate,
        task_score_raw=task_score_raw,
        task_score=None if turn_cap else round(task_score_raw / 2, 2), # 0-2 → 0/0.5/1 归一
        turn_cap=turn_cap,
        n_items_in_window=n_win if turn_cap else None,
        flow_rate_in_window=flow_rate_in_window,
        reason=data.get("reason", ""),
        raw=data,
    )


def _clamp_score(value: Any) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(2, v))


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
