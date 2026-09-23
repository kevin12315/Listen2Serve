"""LeakageProbe：透明度操纵有效性检验（K18c 新增）。

与其它裁判的区别：**不评客服**。它盲评的是用户模拟器自己产出的关键轮文本，
回答一个与被测模型完全无关的问题——「仅凭文字，这句话把说话人的态度露出了多少」。

为什么需要独立度量：现有指标（KeyTurnScore/FlowRate/VoiceMean）全部是「客服表现」，
一旦 T3 分数偏低，无法区分两种根本不同的归因：
  (a) 客服确实听不出弦外之音（这是我们想测的能力缺口）；
  (b) 用户模拟器根本没把 T3 演成 T3，文字里照样把结论说全了（操纵失效，指标无效）。
探针把 (b) 单独量出来：leakage_score 若不呈 T1 > T2 > T3 的单调梯度，则该批次的
透明度对照本身不成立，客服侧的层间差异不可解读。

三个产出：
- leakage_score（1-5，主）：字面态度表露量。探针只看「客服上一轮 + 用户关键轮」两句，
  拿不到语气、state 标签、key_event 意图，故不会被真值提示污染。
- leakage_state_correct（bool，辅）：仅凭文字猜六态是否猜中真实 user_state。
  T3 应显著低于 T1——这正是「非全双工链路（ASR→文本）会丢掉什么」的直接证据。
- leakage_stance_markers（int，程序化）：关键轮文本命中结论式标记的个数，零成本、
  确定性可复现，作为 LLM 主观分的客观对照（论文里可并列汇报）。

已知盲区（K18b 真机校准发现，读数时必须记住）：surface_score 只答「字面露了多少」，
不答「露的方向对不对」。一条把立场演反的 T3（本该不满却问得像有兴趣）与一条演得干净的
T3 同样得 1 分，且 state_guess 同样猜不中。方向是否回正只能靠人工逐样本读关键轮文本，
或看 KeyTurnScore／FlowJudge 的定性文字，探针给不出。

失败不阻断：LLM 异常时三个字段置 None，与 ASR 失败同构——探针是附加度量，
不能让它把主评测拖挂。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from listen2serve.evaluation.prompts import LEAKAGE_PROBE_PROMPT
from listen2serve.gateway.llm_gateway import LLMGateway, LLMResult
from listen2serve.runtime.user_simulator import STATES, _T3_STANCE_MARKERS

# 分数值域（与 prompt 中的 1-5 锚点同步）
_MIN_SCORE, _MAX_SCORE = 1, 5


@dataclass
class LeakageVerdict:
    """探针判定结果。"""

    surface_score: int = 0 # 1-5：字面态度表露量
    state_guess: str = "" # 仅凭文字猜的六态之一
    state_correct: bool | None = None # 与真实 user_state 比对；无真值时 None
    stance_markers: int = 0 # 程序化：结论式标记命中数
    cue: str = "" # 探针摘引的字面依据
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
    """关键轮文本命中的结论式标记（与 T3 禁用清单同源，程序化零成本度量）。

    只认显式结论词，命中数为 0 不代表没泄露态度（词表防线的固有局限，见 vocab.py），
    故它只作 LLM 主观分的客观对照，不单独当判据。
    """
    return [m for m in _T3_STANCE_MARKERS if m in (text or "")]


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
        data: dict[str, Any] = {}
        retried = False
        for attempt in range(2):
            result: LLMResult = self.llm.chat(
                # max_tokens 2048（2026-09-06 从 512 提高）：其余四个裁判
                # （policy 2048 / keyturn 2048 / flow 4096 / dialogue 2048 / voice 2048）
                # 早已为 Gemini reasoning 预留了输出空间，**只有本文件漏了**。
                # 实测代价：用带长思考的音频裁判时 LeakageScore **15/15 全部为空**
                # （reasoning tokens 吃光 512 预算 ⇒ _extract_json 拿不到 JSON）；
                # 换用不带长思考的型号后缺 0/15 ⇒ 判定为空是型号特性，不是数据问题。
                # 实测结论，样本量 15。
                messages, model=self.llm_model, temperature=0.0, max_tokens=2048,
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
        guess = str(data.get("state_guess", "")).strip()
        if guess not in STATES:
            guess = "" # 越界值不采信（不强行回落 neutral，免得污染猜中率分母）
        return LeakageVerdict(
            surface_score=_clamp(data.get("surface_score")),
            state_guess=guess,
            state_correct=(guess == oracle_state) if (guess and oracle_state) else None,
            stance_markers=len(markers),
            cue=str(data.get("cue", "")),
            parse_retried=retried,
            raw=data,
        )


def _clamp(value: Any) -> int:
    """1-5 截断；不可解析返回 0（聚合层按「无有效值」剔除，不记 1 分）。"""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 0
    if v < _MIN_SCORE or v > _MAX_SCORE:
        return 0
    return v


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
