"""对话级指标：TaskCompletion（非动态场景）/ TransitionScore（动态场景）。

- TaskCompletion（0-2）：仅全正/全负场景（50%）
- TransitionScore（0-2）：仅动态场景（正→负 / 负→正）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from listen2serve.domains.base import DomainSpec, render_customer_profile, render_judge_contract
from listen2serve.evaluation.prompts import (
    DIALOGUE_TASK_JUDGE_PROMPT,
    DIALOGUE_TRANSITION_JUDGE_PROMPT,
)
from listen2serve.gateway.llm_gateway import LLMGateway, LLMResult


@dataclass
class DialogueVerdict:
    """对话级指标判定结果。"""

    metric: str # task_completion | transition_score
    score: int = 0
    reason: str = ""
    raw: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "score": self.score, "reason": self.reason}


class DialogueMetrics:
    """对话级指标裁判。"""

    def __init__(self, llm: LLMGateway | None = None, llm_model: str | None = None) -> None:
        self.llm = llm or LLMGateway()
        self.llm_model = llm_model
        # F5 统计（evaluate 结束汇总输出）
        self.parse_retries = 0
        self.parse_failures = 0
        # 内部方案 token 经济性统计（旧链路基线对照）
        self.input_tokens = 0
        self.judge_calls = 0

    def _chat_json(self, system: str, prompt: str) -> dict[str, Any]:
        """调用裁判 LLM 并解析 JSON；F5：解析失败重试一次再落默认值。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(2):
            result: LLMResult = self.llm.chat(
                messages,
                model=self.llm_model,
                temperature=0.0,
                max_tokens=2048, # Gemini reasoning 消耗大，需预留输出空间
            )
            self.input_tokens += result.prompt_tokens
            self.judge_calls += 1
            data = _extract_json(result.text)
            if data:
                return data
            if attempt == 0: # 首次解析失败 → 记一次重试；重试再失败才落默认值
                self.parse_retries += 1
        self.parse_failures += 1
        return {}

    def task_completion(
        self,
        domain: DomainSpec,
        scenario: dict[str, Any],
        history: list[dict[str, str]],
    ) -> DialogueVerdict:
        """TaskCompletion（全正/全负场景）。"""
        goal = scenario.get("task_goal") or _default_goal(scenario)
        prompt = DIALOGUE_TASK_JUDGE_PROMPT.format(
            task_goal=goal,
            policy_rules=_render_rules(domain, scenario),
            history=_render_history(history),
        )
        data = self._chat_json("你是客服任务完成度裁判，只输出 JSON，不要输出 markdown 代码块或其他内容。", prompt)
        return DialogueVerdict(
            metric="task_completion",
            score=_clamp_score(data.get("score", 0)),
            reason=data.get("reason", ""),
            raw=data,
        )

    def transition_score(
        self,
        domain: DomainSpec,
        scenario: dict[str, Any],
        history: list[dict[str, str]],
    ) -> DialogueVerdict:
        """TransitionScore（动态场景）。"""
        # 内部方案（v6.0）：transition 键退役，转变描述由 dynamics + key_event 语义
        # 构造；旧数据仍带 transition 时沿用旧口径（旧 run 重评兼容）。
        us = scenario.get("user_script") or {}
        transition = us.get("transition") or scenario.get("transition") or {}
        if transition:
            desc = f"{transition.get('type', '')}：{transition.get('trigger', '')}"
        else:
            dynamics = scenario.get("dynamics", "")
            ke = us.get("key_event") or {}
            direction = {
                "pos_to_neg": "用户态度由配合转向负面（不满/质疑/急迫等）",
                "neg_to_pos": "用户态度由负面转向配合",
            }.get(dynamics, "用户态度发生转变")
            hint = ke.get("trigger_hint") or ke.get("description") or ""
            desc = f"{dynamics}：{direction}" + (f"（关键事件：{hint}）" if hint else "")
        prompt = DIALOGUE_TRANSITION_JUDGE_PROMPT.format(
            transition_desc=desc,
            policy_rules=_render_rules(domain, scenario),
            history=_render_history(history),
        )
        data = self._chat_json("你是客服状态转变适应度裁判，只输出 JSON，不要输出 markdown 代码块或其他内容。", prompt)
        return DialogueVerdict(
            metric="transition_score",
            score=_clamp_score(data.get("score", 0)),
            reason=data.get("reason", ""),
            raw=data,
        )


def _render_history(history: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{'用户' if h['role'] == 'user' else '客服'}: {h['content']}" for h in history
    ) or "（无）"


def _render_rules(domain: DomainSpec, scenario: dict[str, Any]) -> str:
    """薄封装：委托 render_judge_contract（裁判与被测模型单一契约来源），
    并在返回值末尾**追加**同源客户信息档案板块（2026-08-14，与 policy_judge 同口径）。

    保留 B1 归一化行为：规则键归一化为 canonical 英文码（中文显示名由
    STATE_LABELS 派生），兼容旧格式中文键（存量数据/旧 run 重评），与 policy_judge 同口径。
    render_judge_contract 本体不动（基线 digest 冻结）；无 db_seed 时档案为空，
    输出与旧行为逐字节一致。
    """
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
