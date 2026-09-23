"""planS2（客服侧 prompt 瘦身）回归测试：v9 口径 + v6/v7/v8 字节冻结。

防回归点（逐条对应 planS2 §7 的验收项）：

1. **V7（plan 定为最重要的一条）**：v6/v7/v8 三个旧版本的通话指令字节不变 —— 用 golden
   sha256 锁死。历史 run 的 prompt hash 全靠这条才可复算。
2. **v8 的 variant 透传缺陷是"有意冻结"而不是"已修好"**：v8 必须**继续**掉进 v6 全量分支
   （planS2 §3.2 / §5.1）。哪天有人"顺手修好"了 v8，这条测试就会红 —— 那等于改掉历史字节。
3. **V3**：v9/v9_notable 不再渲染「# 用户状态应对策略」整节（三份状态规范收成一份）。
4. **V4**：v9 速查表 5 行 5 态，无「未出现以上特殊状态」兜底行、无 [中立]。
5. **V5**：v9 剔除裁判不判的字数约束（「回复控制在…」）。
6. **V6**：v9 的 business_flow 保持一行一阶段（不落 v6 全量分支）。
7. **改动 4.2 的条件剔除**：红线覆盖才删、不覆盖必须留；且剔除清单的键必须逐字命中域里
   真实存在的通用约束 —— 否则改了域文本后剔除会**静默失效**（键对不上就等于没剔）。
8. **改动 2 的同源要求**：v9 的速查表必须由 `domains.base.action_table()` 渲染，不许手抄；
   `action_table()` 本身用 golden sha256 锁死（planS2 §5.3：本轮不改词）。
9. **单变量**：v9 与 v9_notable 的差异**只有**那张速查表。
10. 裁判契约（render_judge_contract）与被测 prompt 同步瘦身，且 v9 仍带三条 v7 会话级规则。

设计纪律（照抄 tests/conftest.py 的教训）：**不写死场景条数、不依赖 scenarios.jsonl**。
所有断言只吃 src/ 下的代码常量与本文件内合成的 agent_policy —— 数据扩样不该让这些用例红。
"""

from __future__ import annotations

import hashlib
import re

import pytest

from listen2serve.domains import available_roles, get_domain
from listen2serve.domains.base import (
    ACTION_TABLE_ORDER,
    ACTION_V9,
    AGENT_CLOSING_RULE,
    AGENT_DUAL_RESPONSE_RULE,
    AGENT_PROMPT_VERSIONS,
    AGENT_STATE_FIRST_RULE,
    STATE_LABELS,
    _V9_CONSTRAINT_COVERED_BY_RED_LINE,
    _V9_DROP_CONSTRAINT_SUBSTR,
    _v9_filter_constraints,
    action_table,
    render_judge_contract,
    render_policy_sections,
)
from listen2serve.runtime.agent import (
    VOICE_CALL_INSTRUCTION_V6,
    VOICE_CALL_INSTRUCTION_V7,
    VOICE_CALL_INSTRUCTION_V8,
    VOICE_CALL_INSTRUCTION_V9,
    VOICE_CALL_INSTRUCTION_V9_NOTABLE,
    _V9_ACTION_TABLE,
    voice_call_instruction,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- golden 指纹：改动前（2026-09-06，planS2 执行前）实测值，**不得随代码更新** ----
# 这三条是 planS2 §7 V7 的代码层护栏。要改 v6/v7/v8 的字节，必须先改这里并在报告里说明
# 为什么历史 run 不再需要复算 —— 那是一次裁定级决定，不是一次重构。
GOLDEN_LEGACY_INSTRUCTIONS = {
    "v6": "815e02a1c4d6a0c16049248f49355b255ca3911e48c5d39481196be936cc1d74",
    "v7": "8a36ce1ceb1ed58a7c29623826e3b3cc1bc406f95cc66a13fdf5db63fb7eb9a8",
    "v8": "9634877205ff5c88d911f45a3d74a7a1bd2a1d69988c1e527da897651bd32023",
}
# action_table() 的 golden 指纹：搬家前（住在 scripts/planU_common.py 时）实测同一值，
# 锁住「上移定义处但输出逐字节不变」+「本轮不改 ACTION_V9 的词」（planS2 §5.3）。
GOLDEN_ACTION_TABLE = "f9d451f7ede7c86492f57e150378ca7c8e0cf0d1e37eed94089f594d74b84b3a"

# 合成 agent_policy：覆盖全部 5 个可选键，且每个键都塞一条"v7 应当剔除"的内容，
# 用来判定走的是精简分支还是 v6 全量分支。不读 scenarios.jsonl。
_AP = {
    "role_description": "合成域客服",
    "communication_style": "温和",
    "success_criteria": ["收口标准甲"],
    "business_flow": [
        {
            "id": "S1", "name": "开场与身份确认", "goal": "确认接听人身份",
            "precondition": "无", "key_points": ["要点甲"],
            "exit_criteria": "客户确认身份", "on_fail": "改约回电",
        }
    ],
    "red_lines": ["RL1 不威胁恐吓", "RL2 严格按流程顺序推进，不得跳阶段"],
    "knowledge_base": ["用户类型应对要点：按文字线索索引的旧状态表", "业务知识条目甲"],
    "speech_style": ["说话风格条目甲"],
}


# ================== 1. 旧版本字节冻结（planS2 §7 V7）==================
@pytest.mark.parametrize("variant", ["v6", "v7", "v8"])
def test_legacy_instruction_bytes_frozen(variant):
    """v6/v7/v8 通话指令字节不变 —— 历史 run 的 prompt hash 必须仍可复算。"""
    actual = voice_call_instruction(variant)
    assert _sha(actual) == GOLDEN_LEGACY_INSTRUCTIONS[variant], (
        f"{variant} 的通话指令字节被改动了。planS2 §5.1 明令不许原地修 v8、§7 V7 要求 "
        f"v6/v7/v8 三者字节逐一不变；确需改动请先走裁定并在报告里说明历史 run 如何复算。"
    )


def test_legacy_constants_still_exported():
    """三个旧常量仍可直接 import（export_web / 旧引用沿用这些名字）。"""
    assert _sha(VOICE_CALL_INSTRUCTION_V6) == GOLDEN_LEGACY_INSTRUCTIONS["v6"]
    assert _sha(VOICE_CALL_INSTRUCTION_V7) == GOLDEN_LEGACY_INSTRUCTIONS["v7"]
    assert _sha(VOICE_CALL_INSTRUCTION_V8) == GOLDEN_LEGACY_INSTRUCTIONS["v8"]


# ============ 2. v8 的 variant 透传缺陷是**有意冻结**（planS2 §3.2 / §5.1）============
def test_v8_still_falls_into_v6_full_branch():
    """v8 必须**继续**走 v6 全量分支 —— 这是被冻结的缺陷，不是待修的 bug。

    planS2 §3.2：v8 名义上是「v7 + 速查表」，实际因 render_policy_sections 的分支判断
    写成 `== "v7"` 而掉进 v6 全量分支，多灌 1325 字。§5.1 明令不许原地修（修了历史 run 的
    prompt hash 就对不上），修好的版本另起 v9。所以这里的"相等"是**要求**，不是缺陷。
    """
    assert render_policy_sections(_AP, variant="v8") == render_policy_sections(_AP, variant="v6")
    assert render_policy_sections(_AP, variant="v8") != render_policy_sections(_AP, variant="v7")
    # v6 全量分支的四项特征在 v8 里都在（这正是被灌回来的 1325 字）
    v8 = "\n".join(render_policy_sections(_AP, variant="v8"))
    assert "### S1 开场与身份确认" in v8, "v8 应保留业务流程六阶段展开（缺陷冻结）"
    assert "## 说话风格" in v8, "v8 应保留「## 说话风格」整节（缺陷冻结）"
    assert "严格按流程顺序推进" in v8, "v8 应保留流程顺序红线（缺陷冻结）"
    assert "用户类型应对要点" in v8, "v8 应保留旧状态表知识条（缺陷冻结）"


@pytest.mark.parametrize("variant", ["v7", "v9", "v9_notable"])
def test_lite_policy_branch_drops_v6_baggage(variant):
    """v7/v9/v9_notable 走精简分支：四项 v6 包袱都不进 prompt（planS2 §7 V6）。"""
    text = "\n".join(render_policy_sections(_AP, variant=variant))
    assert not re.search(r"^### S\d", text, flags=re.M), f"{variant} 的业务流程应是一行一阶段"
    assert "- S1 开场与身份确认：确认接听人身份" in text
    assert "## 说话风格" not in text
    assert "严格按流程顺序推进" not in text, f"{variant} 应剔除与状态优先矛盾的流程顺序红线"
    assert "用户类型应对要点" not in text, f"{variant} 应剔除与声音状态表竞争的旧知识条"
    assert "业务知识条目甲" in text, "业务知识是真值来源，planS2 §3.5 明令保留"


def test_v9_policy_sections_identical_to_v7():
    """v9 的板块口径 = v7（planS2 改动 1：v9 就是修好缺陷后的「v7 + 速查表」）。"""
    assert render_policy_sections(_AP, variant="v9") == render_policy_sections(_AP, variant="v7")
    assert (render_policy_sections(_AP, variant="v9_notable")
            == render_policy_sections(_AP, variant="v7"))


# ================== 3. v9 不再渲染 state_playbook（planS2 §7 V3）==================
@pytest.mark.parametrize("role", sorted(available_roles()))
def test_v9_drops_state_playbook_section(role):
    """v9/v9_notable 不渲染「# 用户状态应对策略」；v6/v7/v8 仍渲染（字节冻结）。"""
    domain = get_domain(role)
    for variant in ("v9", "v9_notable"):
        text = domain.render_policy_prompt(_AP, variant=variant)
        assert "# 用户状态应对策略" not in text, f"{role}/{variant} 不应再渲染 playbook 整节"
        assert "当你从客户声音中听出" not in text, f"{role}/{variant} 不应再渲染 playbook 小节标题"
        assert "### 必须执行：" not in text
    for variant in ("v6", "v7", "v8"):
        text = domain.render_policy_prompt(_AP, variant=variant)
        assert "# 用户状态应对策略" in text, f"{role}/{variant} 必须保留 playbook（字节冻结）"
        # 6 态全渲染（含中立）—— 演绎侧仍用 6 态，客服侧 v9 才统一为 5（planS2 改动 3）。
        # 数小节标题而不是数「时：」：v6 全量分支的业务流程里也有「未达成时：」，会多数一个。
        head = "## 当用户表现出[" if variant == "v6" else "## 当你从客户声音中听出["
        assert text.count(head) == len(domain.policy_rules), f"{role}/{variant} playbook 态数不对"


def test_v9_prompt_still_keeps_policy_rules_field_intact():
    """只停渲染、不动数据：三个域的 policy_rules 仍是 6 态全覆盖（planS2 §5.4）。"""
    for role in available_roles():
        assert set(get_domain(role).policy_rules) == set(STATE_LABELS)
        assert "neutral" in get_domain(role).policy_rules


# ============ 4. 速查表：同源、5 态、无兜底行（planS2 §7 V4 / 改动 2、3）============
def test_action_table_bytes_frozen():
    """action_table() 输出逐字节冻结：搬家不改字、本轮不改词（planS2 §5.3）。"""
    assert _sha(action_table()) == GOLDEN_ACTION_TABLE


def test_v9_instruction_is_notable_plus_table():
    """单变量：v9 = v9_notable + 空行 + 速查表，差异**只有**那张表（planS2 改动 5）。"""
    assert VOICE_CALL_INSTRUCTION_V9 == (
        VOICE_CALL_INSTRUCTION_V9_NOTABLE + "\n\n" + _V9_ACTION_TABLE
    )


def test_v9_table_is_rendered_from_single_source():
    """v9 的速查表必须由 base.action_table() 渲染，不许在 agent.py 手抄第二份。"""
    heading = _V9_ACTION_TABLE.split("\n", 1)[0]
    assert _V9_ACTION_TABLE == action_table(heading=heading)
    # 逐行核：每一行的 opening/follow 都取自 ACTION_V9（不是形似的手抄副本）
    rows = _V9_ACTION_TABLE.split("\n")[1:]
    assert len(rows) == len(ACTION_TABLE_ORDER) == 5
    for row, state in zip(rows, ACTION_TABLE_ORDER, strict=True):
        assert row == (f"- 听出【{STATE_LABELS[state]}】：{ACTION_V9[state]['opening']}；"
                       f"{ACTION_V9[state]['follow']}。"), f"{state} 行与 ACTION_V9 不同源"


def test_v9_table_has_five_states_no_fallback_no_neutral():
    """5 行 5 态；无「未出现以上特殊状态」兜底行、无 [中立]（planS2 改动 3 / §7 V4）。"""
    assert _V9_ACTION_TABLE.count("听出【") == 5
    assert "未出现以上特殊状态" not in VOICE_CALL_INSTRUCTION_V9, "5 选 1 口径下不许留兜底行"
    assert "中立" not in _V9_ACTION_TABLE
    assert set(re.findall(r"听出【(.+?)】", _V9_ACTION_TABLE)) == {
        STATE_LABELS[s] for s in ACTION_TABLE_ORDER
    }
    # 旧的 v8 兜底行仍在 v8 里（字节冻结），不在 v9 里
    assert "未出现以上特殊状态" in VOICE_CALL_INSTRUCTION_V8


def test_v9_dropped_the_dangling_playbook_reference():
    """v7 §一.3「按下方『用户状态应对策略』执行」在 v9 已删 —— 那节不再渲染，留着就是悬空引用。"""
    assert "按下方「用户状态应对策略」执行" in VOICE_CALL_INSTRUCTION_V7
    assert "按下方「用户状态应对策略」执行" not in VOICE_CALL_INSTRUCTION_V9
    # 保留项：声音是判据（§一.1）、双通道应答（§二）、状态优先（§三）、收尾轮（§四）
    for rule in (AGENT_DUAL_RESPONSE_RULE, AGENT_STATE_FIRST_RULE, AGENT_CLOSING_RULE):
        assert rule in VOICE_CALL_INSTRUCTION_V9


# ============ 5. 通用约束瘦身（planS2 改动 4.1 / 4.2，§7 V5）============
@pytest.mark.parametrize("role", sorted(available_roles()))
def test_v9_drops_word_count_constraint(role):
    """「回复控制在…」在 v9 被剔、在 v7 保留（裁判明写不计禁止触发，纯占篇幅）。"""
    domain = get_domain(role)
    has = any(_V9_DROP_CONSTRAINT_SUBSTR in c for c in domain.general_constraints)
    v7 = domain.render_policy_prompt(_AP, variant="v7")
    v9 = domain.render_policy_prompt(_AP, variant="v9")
    assert _V9_DROP_CONSTRAINT_SUBSTR not in v9, f"{role}/v9 不应再出现字数约束"
    if has:
        assert _V9_DROP_CONSTRAINT_SUBSTR in v7, f"{role}/v7 必须保留（字节冻结）"


def test_dedup_keys_match_real_domain_constraints():
    """剔除清单的键必须逐字命中域里真实存在的通用约束。

    这条守的是**静默失效**：键是写死的原文，域文本一旦被改（哪怕加个标点），键就对不上，
    去重会悄悄不再生效，而 prompt 只是"长了几个字"、没有任何报错。
    """
    for role, mapping in _V9_CONSTRAINT_COVERED_BY_RED_LINE.items():
        real = set(get_domain(role).general_constraints)
        unknown = set(mapping) - real
        assert not unknown, (
            f"{role} 的剔除清单里有 {len(unknown)} 个键在域通用约束中已不存在（去重会静默失效）："
            f"{sorted(unknown)}"
        )


def test_constraint_dedup_is_conditional_on_red_line_coverage():
    """红线覆盖才删、不覆盖必须留（planS2 §6 红线 2；实测有 18 条场景属于不覆盖情形）。"""
    constraints = list(get_domain("collection").general_constraints)
    target = "不得承诺减免或延期（无权限）"
    assert target in constraints
    covering = ["RL3 不越权承诺减免延期，一律走申请口径"]
    non_covering = ["RL1 不冒充法院工作人员：只以调解中心身份沟通"]
    assert target not in _v9_filter_constraints("collection", constraints, covering)
    assert target in _v9_filter_constraints("collection", constraints, non_covering)
    # 无红线时一条都不因去重而消失（字数约束仍无条件剔）
    kept = _v9_filter_constraints("collection", constraints, [])
    assert target in kept
    assert not any(_V9_DROP_CONSTRAINT_SUBSTR in c for c in kept)


@pytest.mark.parametrize("role", sorted(available_roles()))
def test_v9_keeps_constraints_not_covered_by_red_lines(role):
    """三域各自"措辞差异有意义、两份都留"的条目必须还在（planS2 §6 红线 2）。"""
    domain = get_domain(role)
    v9 = domain.render_policy_prompt(_AP, variant="v9")
    kept_expected = [c for c in domain.general_constraints
                     if c not in _V9_CONSTRAINT_COVERED_BY_RED_LINE.get(role, {})
                     and _V9_DROP_CONSTRAINT_SUBSTR not in c]
    for c in kept_expected:
        assert f"- {c}" in v9, f"{role}/v9 误删了未被红线覆盖的通用约束：{c}"


def test_v9_filter_does_not_touch_v7_path():
    """过滤器只在 v9 分支生效：v6/v7/v8 的通用约束条数与域里完全一致。"""
    for role in available_roles():
        domain = get_domain(role)
        n = len(domain.general_constraints)
        for variant in ("v6", "v7", "v8"):
            text = domain.render_policy_prompt(_AP, variant=variant)
            block = text.split("# 通用约束\n", 1)[1]
            assert block.count("\n- ") + 1 == n, f"{role}/{variant} 通用约束条数被改动了"


# ================== 6. 版本注册表与裁判契约同步 ==================
def test_agent_prompt_versions_registry():
    """v9/v9_notable 已注册；v8 仍在（供历史复算）；每个版本都能取到通话指令。"""
    for v in ("v6", "v7", "v8", "v9", "v9_notable"):
        assert v in AGENT_PROMPT_VERSIONS
        assert voice_call_instruction(v)
    with pytest.raises(ValueError):
        voice_call_instruction("v10")


@pytest.mark.parametrize("role", sorted(available_roles()))
def test_judge_contract_syncs_with_v9_agent_prompt(role):
    """裁判契约与被测 prompt 同步瘦身，且 v9 仍带三条 v7 会话级规则（同源不脱节）。"""
    domain = get_domain(role)
    scenario = {"agent_policy": _AP, "role": role}
    v9 = render_judge_contract(domain, scenario, variant="v9")
    v7 = render_judge_contract(domain, scenario, variant="v7")
    assert _V9_DROP_CONSTRAINT_SUBSTR not in v9, "裁判契约里也不该再有裁判不判的字数约束"
    assert _V9_DROP_CONSTRAINT_SUBSTR in v7, "v7 契约必须保持原样（历史重评）"
    for rule in (AGENT_DUAL_RESPONSE_RULE, AGENT_STATE_FIRST_RULE, AGENT_CLOSING_RULE):
        assert rule in v9, "v9 契约丢了 v7 会话级规则 —— 分支判断写死 == 'v7' 的老毛病又回来了"
    # 板块同源：契约里的业务流程与收口也是一行一阶段
    assert not re.search(r"^### S\d", v9, flags=re.M)
