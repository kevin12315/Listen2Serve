"""Auxiliary text-only probe of how clearly user wording reveals their attitude.

The probe sees only the preceding agent text and critical user text. Ground truth
is used after scoring to check the guessed state. Failures are reported separately
from the primary service metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from listen2serve.evaluation.prompts import LEAKAGE_PROBE_PROMPT
from listen2serve.evaluation.validation import JudgeResponseError, integer, json_object, string
from listen2serve.gateway.llm_gateway import LLMGateway
from listen2serve.runtime.user_simulator import NEUTRAL_STANCE_MARKERS, STATES


@dataclass
class LeakageVerdict:
    """探针判定结果。"""

    surface_score: int = 0
    state_guess: str = ""
    state_correct: bool | None = None
    stance_markers: int = 0
    cue: str = ""
    parse_retried: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface_score": self.surface_score,
            "state_guess": self.state_guess,
            "state_correct": self.state_correct,
            "stance_markers": self.stance_markers,
            "cue": self.cue,
            "parse_retried": self.parse_retried,
        }


def stance_marker_hits(text: str) -> list[str]:
    """Find explicit stance markers using the simulator's neutral-wording list.

    只认显式结论词，命中数为 0 不代表没泄露态度（词表防线的固有局限，见 vocab.py），
    故它只作 LLM 主观分的客观对照，不单独当判据。
    """
    return [m for m in NEUTRAL_STANCE_MARKERS if m in (text or "")]


class LeakageProbe:
    """盲评探针（文本 LLM）：只看两句文字，评字面态度表露量。"""

    def __init__(self, llm: LLMGateway | None = None, llm_model: str | None = None) -> None:
        self.llm = llm or LLMGateway()
        self.llm_model = llm_model
        self.parse_retries = 0
        self.parse_failures = 0
        self.input_tokens = 0
        self.judge_calls = 0

    def probe(
        self,
        user_text: str,
        prev_agent_text: str = "",
        oracle_state: str | None = None,
    ) -> LeakageVerdict:
        """盲评单条关键轮文本。

        Args:
            user_text: 用户关键轮台词（被评对象）
            prev_agent_text: 客服上一轮（最小上下文；缺则写「（无）」。
                不给更多历史是有意的——上下文越长，探针越容易从剧情反推态度，
                度量的就不再是「这一句文字露了多少」）
            oracle_state: 真实 user_state，仅用于事后比对猜中与否，不进 prompt
        """
        markers = stance_marker_hits(user_text)
        prompt = LEAKAGE_PROBE_PROMPT.format(
            agent_text=prev_agent_text or "（无）",
            user_text=user_text or "（无）",
        )
        messages = [
            {"role": "system", "content": "你只输出 JSON，不要输出 markdown 代码块或其他内容。"},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(2):
            result = self.llm.chat(messages, model=self.llm_model, temperature=0.0, max_tokens=2048)
            self.input_tokens += result.prompt_tokens
            self.judge_calls += 1
            try:
                data = json_object(result.text)
                score = integer(data, "surface_score", 1, 5)
                guess = string(data, "state_guess", tuple(STATES))
                cue = string(data, "cue")
                return LeakageVerdict(
                    surface_score=score,
                    state_guess=guess,
                    state_correct=(guess == oracle_state) if oracle_state else None,
                    stance_markers=len(markers),
                    cue=cue,
                    parse_retried=bool(attempt),
                    raw=data,
                )
            except JudgeResponseError as exc:
                if attempt == 0:
                    self.parse_retries += 1
                else:
                    self.parse_failures += 1
                    raise JudgeResponseError(f"Wording probe invalid after one retry: {exc}") from exc
        raise AssertionError("unreachable")
