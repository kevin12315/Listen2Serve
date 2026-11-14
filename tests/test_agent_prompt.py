"""发布版客服提示词、共享动作表与约束过滤的回归检查。"""

from __future__ import annotations

import hashlib
import re

import pytest

from listen2serve.domains import available_roles, get_domain
from listen2serve.domains.base import (
    _CONSTRAINT_COVERED_BY_RED_LINE,
    _UNSCORED_CONSTRAINT_SUBSTR,
    ACTION_TABLE_ORDER,
    AGENT_CLOSING_RULE,
    AGENT_DUAL_RESPONSE_RULE,
    AGENT_STATE_FIRST_RULE,
    STATE_ACTIONS,
    STATE_LABELS,
    action_table,
    filter_constraints,
    render_judge_contract,
    render_policy_sections,
)
from listen2serve.runtime.agent import (
    _STATE_ACTION_TABLE,
    VOICE_CALL_INSTRUCTION,
    voice_call_instruction,
)

GOLDEN_ACTION_TABLE = "f9d451f7ede7c86492f57e150378ca7c8e0cf0d1e37eed94089f594d74b84b3a"


_AP = {
    "role_description": "合成域客服",
    "communication_style": "温和",
    "success_criteria": ["收口标准甲"],
    "business_flow": [
        {
            "id": "S1",
            "name": "开场与身份确认",
            "goal": "确认接听人身份",
            "precondition": "无",
            "key_points": ["要点甲"],
            "exit_criteria": "客户确认身份",
            "on_fail": "改约回电",
        }
    ],
    "red_lines": ["RL1 不威胁恐吓", "RL2 严格按流程顺序推进，不得跳阶段"],
    "knowledge_base": ["用户类型应对要点：按文字线索索引的旧状态表", "业务知识条目甲"],
    "speech_style": ["说话风格条目甲"],
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_action_table_bytes_frozen():
    """验证发布版共享动作与约束的契约。"""
    assert _sha(action_table()) == GOLDEN_ACTION_TABLE


def test_table_is_rendered_from_single_source():
    """验证发布版共享动作与约束的契约。"""
    heading = _STATE_ACTION_TABLE.split("\n", 1)[0]
    assert _STATE_ACTION_TABLE == action_table(heading=heading)

    rows = _STATE_ACTION_TABLE.split("\n")[1:]
    assert len(rows) == len(ACTION_TABLE_ORDER) == 5
    for row, state in zip(rows, ACTION_TABLE_ORDER, strict=True):
        assert row == (
            f"- 听出【{STATE_LABELS[state]}】：{STATE_ACTIONS[state]['opening']}；"
            f"{STATE_ACTIONS[state]['follow']}。"
        ), f"{state} 行与 STATE_ACTIONS 不同源"


def test_table_has_five_states_no_fallback_no_neutral():
    """验证发布版共享动作与约束的契约。"""
    assert _STATE_ACTION_TABLE.count("听出【") == 5
    assert "未出现以上特殊状态" not in VOICE_CALL_INSTRUCTION, "5 选 1 口径下不许留兜底行"
    assert "中立" not in _STATE_ACTION_TABLE
    assert set(re.findall(r"听出【(.+?)】", _STATE_ACTION_TABLE)) == {
        STATE_LABELS[s] for s in ACTION_TABLE_ORDER
    }


def test_dedup_keys_match_real_domain_constraints():
    """验证发布版共享动作与约束的契约。"""
    for role, mapping in _CONSTRAINT_COVERED_BY_RED_LINE.items():
        real = set(get_domain(role).general_constraints)
        unknown = set(mapping) - real
        assert not unknown, (
            f"{role} 的剔除清单里有 {len(unknown)} 个键在域通用约束中已不存在（去重会静默失效）："
            f"{sorted(unknown)}"
        )


def test_constraint_dedup_is_conditional_on_red_line_coverage():
    """验证发布版共享动作与约束的契约。"""
    constraints = list(get_domain("collection").general_constraints)
    target = "不得承诺减免或延期（无权限）"
    assert target in constraints
    covering = ["RL3 不越权承诺减免延期，一律走申请口径"]
    non_covering = ["RL1 不冒充法院工作人员：只以调解中心身份沟通"]
    assert target not in filter_constraints("collection", constraints, covering)
    assert target in filter_constraints("collection", constraints, non_covering)

    kept = filter_constraints("collection", constraints, [])
    assert target in kept
    assert not any(_UNSCORED_CONSTRAINT_SUBSTR in c for c in kept)


@pytest.mark.parametrize("role", sorted(available_roles()))
def test_keeps_constraints_not_covered_by_red_lines(role):
    """验证发布版共享动作与约束的契约。"""
    domain = get_domain(role)
    prompt = domain.render_policy_prompt(_AP)
    kept_expected = [
        c
        for c in domain.general_constraints
        if c not in _CONSTRAINT_COVERED_BY_RED_LINE.get(role, {}) and _UNSCORED_CONSTRAINT_SUBSTR not in c
    ]
    for c in kept_expected:
        assert f"- {c}" in prompt, f"{role} 误删了未被红线覆盖的通用约束：{c}"


def test_call_instruction_matches_retained_behavior():
    assert voice_call_instruction() == VOICE_CALL_INSTRUCTION
    assert _sha(VOICE_CALL_INSTRUCTION) == "de34e96b77d6c749649eb8ac457909e3841e1cba0560aca890bb4484b3096cf2"
    assert "按下方「用户状态应对策略」执行" not in VOICE_CALL_INSTRUCTION
    for rule in (AGENT_DUAL_RESPONSE_RULE, AGENT_STATE_FIRST_RULE, AGENT_CLOSING_RULE):
        assert rule in VOICE_CALL_INSTRUCTION


def test_policy_sections_use_compact_business_flow():
    text = "\n".join(render_policy_sections(_AP))
    assert "- S1 开场与身份确认：确认接听人身份" in text
    assert not re.search(r"^### S\d", text, flags=re.MULTILINE)
    assert "## 说话风格" not in text
    assert "严格按流程顺序推进" not in text
    assert "用户类型应对要点" not in text
    assert "业务知识条目甲" in text


@pytest.mark.parametrize("role", sorted(available_roles()))
def test_agent_and_judge_share_constraints(role):
    domain = get_domain(role)
    prompt = domain.render_policy_prompt(_AP)
    contract = render_judge_contract(domain, {"agent_policy": _AP, "role": role})
    assert "# 用户状态应对策略" not in prompt
    for text in (prompt, contract):
        assert _UNSCORED_CONSTRAINT_SUBSTR not in text
        assert not re.search(r"^### S\d", text, flags=re.MULTILINE)
    for rule in (AGENT_DUAL_RESPONSE_RULE, AGENT_STATE_FIRST_RULE, AGENT_CLOSING_RULE):
        assert rule in contract
    assert set(domain.policy_rules) == set(STATE_LABELS)
