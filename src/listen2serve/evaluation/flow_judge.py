"""Whole-dialogue checklist coverage and normalized task completion scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from listen2serve.domains.base import DomainSpec, render_customer_profile, render_judge_contract
from listen2serve.evaluation.prompts import FLOW_JUDGE_PROMPT, FLOW_JUDGE_WINDOW_SUFFIX
from listen2serve.evaluation.validation import (
    JudgeResponseError,
    action_items,
    grounded_evidence,
    integer,
    json_object,
    string,
)
from listen2serve.gateway.llm_gateway import LLMGateway


@dataclass
class FlowVerdict:
    """FlowJudge 判定结果（flow_items + FlowRate + task_score）。"""

    flow_items: list[dict[str, Any]] = field(default_factory=list)
    flow_rate: float = 0.0
    task_score_raw: int = 0
    task_score: float | None = 0.0

    turn_cap: int | None = None
    n_items_in_window: int | None = None
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
    """Use the conditional checklist when present, otherwise the scenario action list."""
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

        self.input_tokens = 0
        self.judge_calls = 0

    def judge(
        self,
        domain: DomainSpec,
        scenario: dict[str, Any],
        history: list[dict[str, str]],
        turn_cap: int | None = None,
    ) -> FlowVerdict:
        """Score the full dialogue or a fixed turn window; keep the checklist unchanged."""
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
            {
                "role": "system",
                "content": "你是客服全对话裁判，只输出 JSON，不要输出 markdown 代码块或其他内容。",
            },
            {"role": "user", "content": prompt},
        ]
        for attempt in range(2):
            result = self.llm.chat(messages, model=self.llm_model, temperature=0.0, max_tokens=4096)
            self.input_tokens += result.prompt_tokens
            self.judge_calls += 1
            try:
                verdict = _parse_flow_json(json_object(result.text), checklist, turn_cap, history=history)
                verdict.parse_retried = bool(attempt)
                return verdict
            except JudgeResponseError as exc:
                if attempt == 0:
                    self.parse_retries += 1
                else:
                    self.parse_failures += 1
                    raise JudgeResponseError(f"Flow judge invalid after one retry: {exc}") from exc
        raise AssertionError("unreachable")


def _render_checklist(checklist: list[str]) -> str:
    return "\n".join(f"{i}. {a}" for i, a in enumerate(checklist, 1)) or "（无）"


def _history_in_window(history: list[dict[str, str]], turn_cap: int | None) -> list[dict[str, str]]:
    """Select the same complete user turns for rendering and evidence validation."""
    if turn_cap:
        kept: list[dict[str, str]] = []
        turn = 0
        for h in history:
            if h["role"] == "user":
                turn += 1
                if turn > turn_cap:
                    break
            kept.append(h)
        return kept
    return history


def _render_history_turn_numbered(history: list[dict[str, str]], turn_cap: int | None = None) -> str:
    """完整对话渲染（标注轮号，供 flow_items 命中轮次标注）。

    turn_cap=K 时只保留前 K 轮：一轮 = 一条 user 起、到下一条 user 前为止（含其中的
    assistant 行）。**不能按下标切**——history 是 user/assistant 交替，按下标截会把
    某轮的客服发言截掉半句，而这正是本次要修的故障形态。
    """
    history = _history_in_window(history, turn_cap)
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
    """Render the shared service contract with the agent's customer facts."""
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


def _parse_flow_json(
    data: dict[str, Any], checklist: list[str], turn_cap: int | None = None, *, history: list[dict[str, str]]
) -> FlowVerdict:
    """解析 FlowJudge JSON：按固定清单对齐条目（防裁判改写成清单长度变化）。"""
    visible_history = _history_in_window(history, turn_cap)
    n_turns = 0
    annotated = []
    for message in visible_history:
        n_turns += message["role"] == "user"
        annotated.append((n_turns, message["role"], message["content"]))
    raw_items = action_items(data, "flow_items", checklist)
    if [item["action"] for item in raw_items] != checklist:
        raise JudgeResponseError("flow_items must follow the supplied checklist order")
    flow_items: list[dict[str, Any]] = []
    for action, item in zip(checklist, raw_items):
        hit = string(item, "hit", ("是", "部分", "否"))
        evidence = string(item, "evidence")
        turn = item.get("turn")
        if "turn" not in item or (turn is not None and (type(turn) is not int or turn < 1)):
            raise JudgeResponseError("flow item turn must be a positive integer or null")
        if hit != "否" and turn is None:
            raise JudgeResponseError("Matched flow item requires a turn")
        if hit == "否" and turn is not None:
            raise JudgeResponseError("Unmatched flow item turn must be null")
        in_window = item.get("in_window", True)
        if turn_cap and type(item.get("in_window")) is not bool:
            raise JudgeResponseError("Windowed flow item requires boolean in_window")
        if turn_cap and turn is not None and turn > turn_cap:
            raise JudgeResponseError("flow item turn exceeds the observation window")
        if turn is not None and turn > n_turns:
            raise JudgeResponseError("flow item turn exceeds the supplied dialogue")
        if hit != "否":
            # Agent quotations must identify the cited turn. The rubric also allows
            # the user's same-turn or later feedback to establish implicit completion.
            sources = [
                text
                for number, role, text in annotated
                if (role == "assistant" and number == turn) or (role == "user" and number >= turn)
            ]
            grounded_evidence(evidence, sources, required=True, field="flow_items.evidence")
            in_window = True
        flow_items.append(
            {
                "action": action,
                "hit": hit,
                "turn": turn,
                "evidence": evidence,
                "in_window": in_window,
            }
        )
    scores = {"是": 1.0, "部分": 0.5, "否": 0.0}
    flow_rate = round(sum(scores[it["hit"]] for it in flow_items) / len(flow_items), 4) if flow_items else 0.0
    n_win = sum(1 for it in flow_items if it["in_window"])
    flow_rate_in_window = (
        round(sum(scores[it["hit"]] for it in flow_items if it["in_window"]) / n_win, 4)
        if turn_cap and n_win
        else None
    )

    if turn_cap and ("task_score" not in data or data["task_score"] is not None):
        raise JudgeResponseError("Windowed flow verdict requires null task_score")
    task_score_raw = 0 if turn_cap else integer(data, "task_score", 0, 2)
    return FlowVerdict(
        flow_items=flow_items,
        flow_rate=flow_rate,
        task_score_raw=task_score_raw,
        task_score=None if turn_cap else round(task_score_raw / 2, 2),
        turn_cap=turn_cap,
        n_items_in_window=n_win if turn_cap else None,
        flow_rate_in_window=flow_rate_in_window,
        reason=string(data, "reason"),
        raw=data,
    )
