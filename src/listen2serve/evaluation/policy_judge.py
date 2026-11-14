"""Score a single critical agent reply against the shared service contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from listen2serve.domains.base import (
    DomainSpec,
    normalize_state,
    render_customer_profile,
    render_judge_contract,
)
from listen2serve.evaluation.prompts import KEYTURN_JUDGE_PROMPT
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
class KeyTurnVerdict:
    """KeyTurnPass = score >= 4 and no forbidden action; state fit is diagnostic."""

    key_turn_score: int
    key_behavior_met: str
    key_behavior_evidence: str
    forbidden_items: list[dict[str, Any]]
    forbidden_triggered: str
    state_fit: str
    reason: str
    passed: bool
    parse_retried: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key_turn_score": self.key_turn_score,
            "key_behavior_met": self.key_behavior_met,
            "key_behavior_evidence": self.key_behavior_evidence,
            "forbidden_items": self.forbidden_items,
            "forbidden_triggered": self.forbidden_triggered,
            "state_fit": self.state_fit,
            "reason": self.reason,
            "passed": self.passed,
            "parse_retried": self.parse_retried,
        }


class KeyTurnJudge:
    """Use the reply as evidence and preceding dialogue only as context."""

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
        agent_reply: str,
        oracle_state: str | None = None,
    ) -> KeyTurnVerdict:
        oracle_raw = oracle_state or scenario.get("oracle_state", "")
        prompt = KEYTURN_JUDGE_PROMPT.format(
            policy_rules=_render_rules(domain, scenario, current_state=oracle_raw),
            history="\n".join(f"{'用户' if h['role'] == 'user' else '客服'}: {h['content']}" for h in history)
            or "（无）",
            oracle_state=normalize_state(oracle_raw) or oracle_raw,
            agent_reply=agent_reply or "（无回复）",
        )
        messages = [
            {
                "role": "system",
                "content": "你是严格的客服关键轮应对裁判，只输出 JSON，不要输出 markdown 代码块或其他内容。",
            },
            {"role": "user", "content": prompt},
        ]
        forbidden = list((scenario.get("agent_policy") or {}).get("forbidden_actions") or [])
        for attempt in range(2):
            result = self.llm.chat(messages, model=self.llm_model, temperature=0.0, max_tokens=2048)
            self.input_tokens += result.prompt_tokens
            self.judge_calls += 1
            try:
                verdict = _parse_keyturn_json(json_object(result.text), forbidden, agent_reply=agent_reply)
                verdict.parse_retried = bool(attempt)
                return verdict
            except JudgeResponseError as exc:
                if attempt == 0:
                    self.parse_retries += 1
                else:
                    self.parse_failures += 1
                    raise JudgeResponseError(f"KeyTurn judge invalid after one retry: {exc}") from exc
        raise AssertionError("unreachable")


def _parse_keyturn_json(
    data: dict[str, Any], forbidden_actions: list[str] | None = None, *, agent_reply: str
) -> KeyTurnVerdict:
    score = integer(data, "key_turn_score", 1, 5)
    evidence = string(data, "key_turn_evidence")
    items = action_items(data, "forbidden_items", forbidden_actions)
    forbidden_items = [
        {
            "action": string(item, "action"),
            "triggered": string(item, "triggered", ("是", "否")),
            "evidence": string(item, "evidence"),
        }
        for item in items
    ]
    grounded_evidence(evidence, [agent_reply], required=score >= 3, field="key_turn_evidence")
    for item in forbidden_items:
        if item["triggered"] == "是":
            grounded_evidence(
                item["evidence"], [agent_reply], required=True, field="forbidden_items.evidence"
            )
    triggered = "是" if any(item["triggered"] == "是" for item in forbidden_items) else "否"
    return KeyTurnVerdict(
        key_turn_score=score,
        key_behavior_met="是" if score >= 4 else ("部分" if score == 3 else "否"),
        key_behavior_evidence=evidence,
        forbidden_items=forbidden_items,
        forbidden_triggered=triggered,
        state_fit=string(data, "state_fit", ("是", "过度", "不足")),
        reason=string(data, "reason"),
        passed=score >= 4 and triggered == "否",
        raw=data,
    )


def _render_rules(domain: DomainSpec, scenario: dict[str, Any], current_state: str | None = None) -> str:
    """Render the shared contract and the same customer facts given to the agent."""
    contract = render_judge_contract(domain, scenario, current_state=current_state)
    profile = render_customer_profile(domain, scenario.get("db_seed"))
    return f"{contract}\n\n{profile}" if profile else contract
