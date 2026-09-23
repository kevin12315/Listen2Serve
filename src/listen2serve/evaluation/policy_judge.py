"""PolicyPass 判定：文本 LLM Judge（四维判定 + B2 逐动作分解）。

输入 = 统一策略契约 + 对话历史 + oracle 状态 + Agent 关键轮回复文本。
PolicyPass = 必需全满足 ∧ 禁止零触发 ∧ 状态适配=是
（等价于新格式明细判据：required_rate==1 ∧ 零触发 ∧ state_fit=是）。

B2：POLICY_JUDGE_PROMPT v2 输出逐条明细（required_items/forbidden_items），
顶层字段 required_satisfied/forbidden_triggered 由明细聚合（旧格式 JSON 回落旧判据解析）；
F5：解析失败重试一次再落默认值。

内部方案（2026-09-06，经用户裁定放行）：KeyTurnJudge.judge() 增三个**仅关键字**参数
variant / seed / thinking，缺省值下行为与改动前逐字节一致（见该方法 docstring）。
判据本体（1-5 量规、passed = score≥4 ∧ 禁止零触发）与 _parse_keyturn_json **一字未动** ——
plan「不要改 policy_judge.py」的原意是"这套判据是对的，别去改判据"，本次只开参数通道：
探针复算历史 run 需要把契约版本对齐到该 run 的 agent_prompt，且裁定 8 要求裁判链带 seed。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from listen2serve.domains.base import DomainSpec, normalize_state, render_customer_profile, render_judge_contract
from listen2serve.evaluation.prompts import KEYTURN_JUDGE_PROMPT, POLICY_JUDGE_PROMPT
from listen2serve.gateway.llm_gateway import LLMGateway, LLMResult


@dataclass
class PolicyVerdict:
    """PolicyPass 判定结果（旧顶层字段全保留；B2 明细进增量键）。"""

    required_satisfied: str = "否" # 是/否/部分
    forbidden_triggered: str = "否" # 是/否
    state_fit: str = "否" # 是/否/过度/不足
    context_coherent: str = "否" # 是/否
    reason: str = ""
    passed: bool = False
    # ---- B2 增量键：逐动作分解明细 ----
    required_items: list[dict[str, Any]] = field(default_factory=list)
    forbidden_items: list[dict[str, Any]] = field(default_factory=list)
    required_rate: float | None = None
    parse_retried: bool = False # F5：是否触发过解析重试
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_satisfied": self.required_satisfied,
            "forbidden_triggered": self.forbidden_triggered,
            "state_fit": self.state_fit,
            "context_coherent": self.context_coherent,
            "reason": self.reason,
            "passed": self.passed,
            "required_items": self.required_items,
            "forbidden_items": self.forbidden_items,
            "required_rate": self.required_rate,
            "parse_retried": self.parse_retried,
        }


class PolicyJudge:
    """策略合规裁判（文本 LLM）。"""

    def __init__(self, llm: LLMGateway | None = None, llm_model: str | None = None) -> None:
        self.llm = llm or LLMGateway()
        self.llm_model = llm_model
        # F5 统计（evaluate 结束汇总输出）
        self.parse_retries = 0
        self.parse_failures = 0
        # 内部方案 token 经济性统计（验收：裁判链每场景输入 token ≤ 旧链路 110%）
        self.input_tokens = 0
        self.judge_calls = 0

    def judge(
        self,
        domain: DomainSpec,
        scenario: dict[str, Any],
        history: list[dict[str, str]],
        agent_reply: str,
        oracle_state: str | None = None,
    ) -> PolicyVerdict:
        """判定关键轮回复是否满足策略（逐动作分解）。"""
        # oracle 状态归一化为 canonical 英文码（兼容旧格式存量数据/旧 run 重评）
        oracle_raw = oracle_state or scenario.get("oracle_state", "")
        oracle = normalize_state(oracle_raw) or oracle_raw
        policy_contract = _render_rules(domain, scenario, current_state=oracle_raw)
        history_text = "\n".join(
            f"{'用户' if h['role'] == 'user' else '客服'}: {h['content']}" for h in history
        )
        prompt = POLICY_JUDGE_PROMPT.format(
            policy_rules=policy_contract,
            history=history_text or "（无）",
            oracle_state=oracle,
            agent_reply=agent_reply or "（无回复）",
        )
        messages = [
            {"role": "system", "content": "你是严格的客服策略合规裁判，只输出 JSON，不要输出 markdown 代码块或其他内容。"},
            {"role": "user", "content": prompt},
        ]
        # F5：解析失败重试一次再落默认值
        data: dict[str, Any] = {}
        retried = False
        for attempt in range(2):
            result: LLMResult = self.llm.chat(
                messages,
                model=self.llm_model,
                temperature=0.0,
                max_tokens=2048, # Gemini 3.1 Pro reasoning 消耗 ~1000 tokens，需预留足够输出空间
            )
            self.input_tokens += result.prompt_tokens
            self.judge_calls += 1
            data = _extract_json(result.text)
            if data:
                break
            if attempt == 0: # 首次解析失败 → 记一次重试；重试再失败才落默认值
                retried = True
                self.parse_retries += 1
        if not data:
            self.parse_failures += 1
        verdict = _parse_policy_json(data)
        verdict.parse_retried = retried
        return verdict


def _parse_policy_json(data: dict[str, Any]) -> PolicyVerdict:
    """解析裁判 JSON：新格式（逐条明细）聚合顶层字段；旧格式回落旧判据。"""
    required_items_raw = data.get("required_items")
    forbidden_items_raw = data.get("forbidden_items")
    if isinstance(required_items_raw, list) and required_items_raw:
        # ---- 新格式（v2 逐条清单）：明细聚合出顶层字段 ----
        required_items = [
            {
                "action": str(it.get("action", "")),
                "hit": str(it.get("hit", "否")),
                "evidence": str(it.get("evidence", "")),
            }
            for it in required_items_raw
            if isinstance(it, dict)
        ]
        hits = [it["hit"] for it in required_items]
        if hits and all(h == "是" for h in hits):
            required_satisfied = "是"
        elif any(h in ("是", "部分") for h in hits):
            required_satisfied = "部分"
        else:
            required_satisfied = "否"
        computed_rate = round(sum(1 for h in hits if h == "是") / len(hits), 2) if hits else 0.0
        # v5.6：required_rate 以明细聚合（computed_rate）为唯一来源；LLM 自报值
        # 可能与逐条明细矛盾，仅保留在 raw 中供对照（PolicyPass 判据本就只依赖
        # 聚合顶层字段，不受影响）。
        required_rate = computed_rate
        forbidden_items = [
            {
                "action": str(it.get("action", "")),
                "triggered": str(it.get("triggered", "否")),
                "evidence": str(it.get("evidence", "")),
            }
            for it in (forbidden_items_raw or [])
            if isinstance(it, dict)
        ]
        forbidden_triggered = "是" if any(
            it["triggered"] in ("是", "true", "True", "1", "已触发") for it in forbidden_items
        ) else "否"
        verdict = PolicyVerdict(
            required_satisfied=required_satisfied,
            forbidden_triggered=forbidden_triggered,
            state_fit=str(data.get("state_fit", "否")),
            context_coherent=str(data.get("context_coherent", "否")),
            reason=data.get("reason", ""),
            required_items=required_items,
            forbidden_items=forbidden_items,
            required_rate=required_rate,
            raw=data,
        )
    else:
        # ---- 旧格式 JSON：回落旧判据解析（向后兼容存量 mock/旧输出）----
        verdict = PolicyVerdict(
            required_satisfied=data.get("required_satisfied", "否"),
            forbidden_triggered=data.get("forbidden_triggered", "否"),
            state_fit=data.get("state_fit", "否"),
            context_coherent=data.get("context_coherent", "否"),
            reason=data.get("reason", ""),
            raw=data,
        )
    # PolicyPass 判据（旧顶层字段口径，等价新明细 required_rate==1 ∧ 零触发 ∧ state_fit=是）
    verdict.passed = (
        verdict.required_satisfied == "是"
        and verdict.forbidden_triggered == "否"
        and verdict.state_fit == "是"
    )
    return verdict


# ---- 内部方案（v2.3-J）：KeyTurnPass 判定（关键轮单条回复 × 单锚点）----


@dataclass
class KeyTurnVerdict:
    """KeyTurnPass 判定结果（v2.5-K：1-5 锚定式量规，MOS 口径）。

    key_turn_score：弦外之音听出与承接质量（1-5）；
    KeyTurnPass(passed) = key_turn_score ≥ 4 ∧ 禁止零触发；
    state_fit 为诊断维度不计入 pass（分寸已含量规内，避免双重计费）。
    key_behavior_met 为兼容派生字段（score≥4→是，3→部分，≤2→否）。
    """

    key_turn_score: int = 0 # 1-5（0 = 解析失败兜底，聚合时按缺失处理）
    key_behavior_met: str = "否" # 兼容派生：是/部分/否
    key_behavior_evidence: str = ""
    forbidden_items: list[dict[str, Any]] = field(default_factory=list)
    forbidden_triggered: str = "否" # 是/否（聚合自 forbidden_items）
    state_fit: str = "否" # 是/过度/不足（语气与力度适配，诊断维度）
    reason: str = ""
    passed: bool = False # KeyTurnPass 布尔（score≥4 ∧ 禁止零触发）
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
    """关键轮应对裁判（v2.3-J，文本 LLM）。

    与 PolicyJudge 的差异（planJ）：判据收敛为三项（key_behavior_met/
    forbidden_items/state_fit），删除 required_items 逐条判定（移入 FlowJudge）；
    prompt 显式声明证据窗口（历史不作命中证据）。输入面与 PolicyJudge 一致
    （契约 + history_before_critical + 关键轮回复 + oracle），便于 CLI 同位替换。
    """

    def __init__(self, llm: LLMGateway | None = None, llm_model: str | None = None) -> None:
        self.llm = llm or LLMGateway()
        self.llm_model = llm_model
        self.parse_retries = 0
        self.parse_failures = 0
        # 内部方案 token 经济性统计
        self.input_tokens = 0
        self.judge_calls = 0

    def judge(
        self,
        domain: DomainSpec,
        scenario: dict[str, Any],
        history: list[dict[str, str]],
        agent_reply: str,
        oracle_state: str | None = None,
        *,
        variant: str | None = None,
        seed: int | None = None,
        thinking: bool | None = None,
    ) -> KeyTurnVerdict:
        """判定关键轮回复。

        内部方案 新增三个**仅关键字**参数（全部缺省 = 改动前行为，逐字节不变）：
          variant —— 裁判契约的 Agent prompt 版本。render_judge_contract 的版本决定契约
                      内容（实测同一场景 v7/v8/v9 = 2748/3186/2628 字：v8 无双通道规则、
                      v9 已剔字数约束）。缺省 None → 走 Settings.agent_prompt，与旧行为一致；
                      探针脚本复算历史 run 时必须显式传该 run 记录的 agent_prompt，
                      否则裁判会拿被测模型根本没见过的约束去判（内部方案 §口径核对 坑 2）。
          seed —— 确定性种子（裁定 8：裁判链 seed=20260903）。缺省 None = 不下发。
          thinking —— 显式开关 reasoning（qwen3 系经 extra_body.enable_thinking 传递）。
        """
        oracle_raw = oracle_state or scenario.get("oracle_state", "")
        oracle = normalize_state(oracle_raw) or oracle_raw
        policy_contract = _render_rules(domain, scenario, current_state=oracle_raw, variant=variant)
        history_text = "\n".join(
            f"{'用户' if h['role'] == 'user' else '客服'}: {h['content']}" for h in history
        )
        prompt = KEYTURN_JUDGE_PROMPT.format(
            policy_rules=policy_contract,
            history=history_text or "（无）",
            oracle_state=oracle,
            agent_reply=agent_reply or "（无回复）",
        )
        messages = [
            {"role": "system", "content": "你是严格的客服关键轮应对裁判，只输出 JSON，不要输出 markdown 代码块或其他内容。"},
            {"role": "user", "content": prompt},
        ]
        data: dict[str, Any] = {}
        retried = False
        for attempt in range(2):
            result: LLMResult = self.llm.chat(
                messages,
                model=self.llm_model,
                temperature=0.0,
                max_tokens=2048,
                seed=seed,
                thinking=thinking,
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
        verdict = _parse_keyturn_json(data)
        verdict.parse_retried = retried
        return verdict


def _parse_keyturn_json(data: dict[str, Any]) -> KeyTurnVerdict:
    """解析 KeyTurnJudge JSON（v2.5-K：key_turn_score 1-5；宽容旧格式 key_behavior_met 回落映射）。"""
    forbidden_items = [
        {
            "action": str(it.get("action", "")),
            "triggered": str(it.get("triggered", "否")),
            "evidence": str(it.get("evidence", "")),
        }
        for it in (data.get("forbidden_items") or [])
        if isinstance(it, dict)
    ]
    forbidden_triggered = "是" if any(
        it["triggered"] in ("是", "true", "True", "1", "已触发") for it in forbidden_items
    ) else "否"
    # v2.5-K：优先读 1-5 分；旧格式（key_behavior_met）回落映射 是→5/部分→3/否→1
    score_raw = data.get("key_turn_score")
    try:
        score = int(score_raw) if score_raw is not None else 0
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(5, score))
    if score == 0:
        legacy = str(data.get("key_behavior_met", ""))
        score = {"是": 5, "部分": 3, "否": 1}.get(legacy, 0)
    key_behavior_met = "是" if score >= 4 else ("部分" if score == 3 else "否")
    state_fit = str(data.get("state_fit", "否"))
    verdict = KeyTurnVerdict(
        key_turn_score=score,
        key_behavior_met=key_behavior_met,
        key_behavior_evidence=str(data.get("key_turn_evidence") or data.get("key_behavior_evidence", "")),
        forbidden_items=forbidden_items,
        forbidden_triggered=forbidden_triggered,
        state_fit=state_fit,
        reason=data.get("reason", ""),
        raw=data,
    )
    # KeyTurnPass 判据（v2.5-K）：弦外之音量规 ≥4 ∧ 禁止零触发（state_fit 为诊断维度不计入）
    verdict.passed = score >= 4 and forbidden_triggered == "否"
    return verdict


def _render_rules(domain: DomainSpec, scenario: dict[str, Any], current_state: str | None = None,
                  variant: str | None = None) -> str:
    """薄封装：委托 render_judge_contract（裁判与被测模型单一契约来源），
    并在返回值末尾**追加**同源客户信息档案板块（2026-08-14）。

    保留 B1 归一化行为：规则键归一化为 canonical 英文码（与 oracle_state 同码可
    程序化对齐），中文显示名由 STATE_LABELS 派生；旧格式中文键同样归一后命中。
    B2 契约在旧规则文本基础上补全 general_constraints/allowed_actions/声音要求。

    档案同源注入：评测默认不发起 function call，被测侧将 db_seed 档案注入初始
    prompt；裁判契约复用 render_customer_profile 追加同一档案，使裁判能核实
    被测回复对档案事实的使用（金额/订单状态/产品促销等）。render_judge_contract
    本体不动（基线 digest 冻结）；无 db_seed 时档案为空，输出与旧行为逐字节一致。

    variant：透传给 render_judge_contract 决定契约的 Agent prompt 版本；
    缺省 None = 由 Settings.agent_prompt 决定，与改动前逐字节一致（PolicyJudge 走的就是
    这条缺省路径，未受影响）。
    """
    contract = render_judge_contract(domain, scenario, current_state=current_state, variant=variant)
    profile = render_customer_profile(domain, scenario.get("db_seed"))
    if not profile:
        return contract
    return f"{contract}\n\n{profile}"


def _extract_json(text: str) -> dict[str, Any]:
    """从 LLM 输出中提取 JSON（容忍 markdown 围栏）。"""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
