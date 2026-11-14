"""域基类：policy 渲染、工具定义、DB schema、任务装载。

每个域（collection/marketing/hotline）实现 `build_domain()` 返回 DomainSpec，
评测运行时按 DomainSpec 装配工具与数据库。
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATE_LABELS: dict[str, str] = {
    "neutral": "中立",
    "cooperative": "配合",
    "displeased": "不满",
    "urgent": "急迫",
    "doubtful": "疑问/不信",
    "hushed": "轻声/不便",
}

# 场景资产仍含部分历史标签；在读取边界归一化，保留原始资产内容。
LEGACY_STATE_MAP: dict[str, str] = {
    "N0_neutral": "neutral",
    "N1_frustrated": "displeased",
    "N2_urgent": "urgent",
    "N3_confused": "doubtful",
    "N4_privacy": "hushed",
    "N5_resistant": "displeased",
    "N6_suspicious": "doubtful",
    "P1_cooperative": "cooperative",
    "P2_relaxed": "cooperative",
    "P3_grateful": "cooperative",
    "P4_interested": "cooperative",
}

_LEGACY_STATE_LABELS: dict[str, str] = {
    "N0_neutral": "N0 中立/谨慎",
    "N1_frustrated": "N1 挫败/升级",
    "N2_urgent": "N2 急迫/不便",
    "N3_confused": "N3 犹豫/困惑",
    "N4_privacy": "N4 隐私受限",
    "N5_resistant": "N5 厌烦/抗拒",
    "N6_suspicious": "N6 怀疑/不信任",
    "P1_cooperative": "P1 配合/信任",
    "P2_relaxed": "P2 轻松/随和",
    "P3_grateful": "P3 感激/满意",
    "P4_interested": "P4 好奇/感兴趣",
}

_LEGACY_PREFIX_MAP: dict[str, str] = {
    "N0": "neutral",
    "N1": "displeased",
    "N2": "urgent",
    "N3": "doubtful",
    "N4": "hushed",
    "N5": "displeased",
    "N6": "doubtful",
    "P1": "cooperative",
    "P2": "cooperative",
    "P3": "cooperative",
    "P4": "cooperative",
}


def normalize_state(key: str | None) -> str | None:
    """将状态码、中文显示名和资产中的历史标签归一为六态英文码。"""
    if not key:
        return None
    k = str(key).strip()
    if k in STATE_LABELS:
        return k
    if k in LEGACY_STATE_MAP:
        return LEGACY_STATE_MAP[k]
    for code, label in _LEGACY_STATE_LABELS.items():
        if k == label or label in k or k in label:
            return LEGACY_STATE_MAP[code]
    for code, label in STATE_LABELS.items():
        if k == label or label in k or k in label:
            return code
    m = re.search(r"([NP]\d)_\w+", k)
    if m:
        prefix = m.group(1)
        for code, new_code in LEGACY_STATE_MAP.items():
            if code.startswith(prefix + "_"):
                return new_code
    m = re.search(r"([NP]\d)(?!\d)", k)
    if m and m.group(1) in _LEGACY_PREFIX_MAP:
        return _LEGACY_PREFIX_MAP[m.group(1)]
    return None


POLICY_SECTION_KEYS = ("business_flow", "red_lines", "knowledge_base", "success_criteria", "speech_style")

_UNSCORED_CONSTRAINT_SUBSTR = "回复控制在"
_CONSTRAINT_COVERED_BY_RED_LINE: dict[str, dict[str, tuple[str, ...]]] = {
    "collection": {
        "不得威胁、辱骂或使用恐吓性语言": ("威胁恐吓", "辱骂"),
        "不得透露欠款信息给第三方": ("透露",),
        "不得复读机式回复（任意连续10字不得与历史完全一致）": ("机械重复",),
        "逾期后果须客观陈述（如征信影响），非威胁": ("据实口径", "给定口径"),
        "不得承诺减免或延期（无权限）": ("越权承诺",),
    },
    "marketing": {
        "不得制造虚假紧迫感（'仅剩最后X个名额'除非属实）": ("虚假紧迫", "虚构紧迫"),
        "不得过度承诺效果": ("权益", "承诺未授权", "不承诺案件结果"),
        "不得在用户明确拒绝后继续施压推销": ("明确拒绝",),
        "不得复读机式回复": ("机械重复",),
    },
    "hotline": {
        "不得敷衍、推诿（'这不归我管'）": ("敷衍", "推诿"),
        "不得复读机式回复": ("机械重复",),
    },
}


def filter_constraints(
    role: str,
    constraints: list[str],
    red_lines: list[str] | None,
) -> list[str]:
    """过滤未评分的字数约束，并在红线覆盖时移除重复约束。

    被测提示词和裁判契约共用此函数；未被红线覆盖的约束继续保留。"""
    covered = _CONSTRAINT_COVERED_BY_RED_LINE.get(role, {})
    rl = red_lines or []
    out: list[str] = []
    for c in constraints:
        if _UNSCORED_CONSTRAINT_SUBSTR in c:
            continue
        keys = covered.get(c)
        if keys and any(k in r for r in rl for k in keys):
            continue
        out.append(c)
    return out


AGENT_VOICE_STATE_RULE = (
    "客户的真实状态常常不写在字面上，而在声音里（语速快慢、音量高低、语气松紧、停顿与犹豫、"
    "句尾上扬或收紧、是否压低嗓音）；判断客户状态以听到的声音为主要依据，字面措辞只是其中一条线索。"
)
AGENT_STATE_FIRST_RULE = (
    "业务流程的下一步与客户当前状态的要求冲突时，以状态应对为先：先接住状态、调整动作，"
    "流程留到后续轮次回补；流程可跳转、可回补，不必逐阶段走完。"
)
AGENT_CLOSING_RULE = (
    "识别到客户想结束通话（明说要挂断/说没别的事了，或声音上只剩极简应答、语气收紧、"
    "明显不愿再谈）时，必须把下一轮作为收尾轮：只做两件事——一句话复述本通已达成的结论或约定，"
    "然后道别；收尾轮不得提出新问题、不得追加推销或劝说、不得重复确认已确认过的信息，"
    "即使流程尚未走完也照此收尾。"
)
AGENT_DUAL_RESPONSE_RULE = (
    "听出客户状态后，你的回应必须同时体现在两个通道上，两者指向同一状态：\n"
    "（一）**文本策略**——说出的内容要针对该状态：不满就先致歉共情再给方案；急迫就压缩表述先给结论；"
    "疑问/不信就先自证身份、给可核实依据；轻声/不便就缩短内容、主动改约；配合就顺势推进坐实要素。\n"
    "（二）**语气转变**——你说话的声音也要跟着变：客户不满时放缓语速、压低音量、把语气放软；"
    "客户急迫时加快节奏、句子变短、去掉铺垫；客户疑问时放慢、加重关键信息、语气笃定；"
    "客户轻声不便时压低音量、简短收束；客户配合时保持平稳积极。\n"
    "只换措辞而声音一成不变，或只改语气而话术照旧，都算没有做到。"
)

STATE_ACTIONS: dict[str, dict[str, str]] = {
    "displeased": {
        "opening": "先道歉，把「抱歉」或「对不起」的意思说出来",
        "follow": "再给一句可执行的下一步",
    },
    "urgent": {
        "opening": "先说「理解您比较着急」的意思",
        "follow": "再直接给结论或时间点",
    },
    "doubtful": {
        "opening": "先说「明白您有疑虑」的意思",
        "follow": "再给一条可核对的依据",
    },
    "hushed": {
        "opening": "先说「您现在不方便的话」的意思",
        "follow": "再提出改个时间或换个方式联系",
    },
    "cooperative": {
        "opening": "先说「好的」的意思",
        "follow": "再一句把事情推到下一步",
    },
}
ACTION_TABLE_ORDER = ("displeased", "urgent", "doubtful", "hushed", "cooperative")
ACTION_TABLE_HEADING = "# 语气应对规范（听出哪一种，就用哪一行的开场）"


def action_table(heading: str | None = None) -> str:
    """渲染五种目标状态的动作表；中立状态不设兜底动作。"""
    lines = [
        f"- 听出【{STATE_LABELS[s]}】：{STATE_ACTIONS[s]['opening']}；{STATE_ACTIONS[s]['follow']}。"
        for s in ACTION_TABLE_ORDER
    ]
    head = ACTION_TABLE_HEADING if heading is None else heading
    return head + "\n" + "\n".join(lines)


JUDGE_ACTION_HEADING = "各状态对应的动作口径（与被测客服 prompt §六 同源；每行的 opening 才是判据）："


def judge_action_table(heading: str | None = None) -> str:
    """裁判视角的「状态→动作」表：opening = 判据，follow = 参考（不单独扣分）。"""
    lines = [
        f"- 听出【{STATE_LABELS[s]}】：判据＝{STATE_ACTIONS[s]['opening']}"
        f"（后续跟进「{STATE_ACTIONS[s]['follow']}」仅作参考，未做不单独扣分）"
        for s in ACTION_TABLE_ORDER
    ]
    head = JUDGE_ACTION_HEADING if heading is None else heading
    return head + "\n" + "\n".join(lines)


def judge_rubric_text(state: str) -> str:
    """返回指定状态的开场评分判据。"""
    code = normalize_state(state) or state
    if code not in STATE_ACTIONS:
        return "按客服规范回应"
    return STATE_ACTIONS[code]["opening"]


_OMITTED_KB_PREFIXES = ("用户类型应对要点",)
_OMITTED_RED_LINE_SUBSTR = "流程顺序推进"


def render_customer_profile(
    domain: DomainSpec | str,
    db_seed: dict[str, Any] | None,
) -> str:
    """将系统侧 db_seed 渲染为双方共享的客户档案。

    用户私有事实仅传给用户模拟器；缺少预期数据时返回空串。"""
    role = domain.role if isinstance(domain, DomainSpec) else str(domain)
    renderer = _PROFILE_RENDERERS.get(role)
    if renderer is None or not db_seed:
        return ""
    return renderer(db_seed)


_PROFILE_TITLE = "# 客户信息档案（系统已预先提供，通话中无需查询、直接作为事实依据）"


def _fmt_num(value: Any) -> str:
    """数值自然化：整数去小数尾巴（3600.0 → 3600），非数值原样转字符串。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _seed_rows(db_seed: dict[str, Any], table: str) -> list[Any]:
    """取预期表的首行（行格式兼容 list/tuple）；缺失或空返回空列表。"""
    rows = db_seed.get(table) or []
    return list(rows[0]) if rows else []


def _profile_collection(db_seed: dict[str, Any]) -> str:
    row = _seed_rows(db_seed, "customers")
    if len(row) < 6:
        return ""
    name, amount, overdue_days, delay_count, promised_date, status = row[:6]
    lines = [
        _PROFILE_TITLE,
        f"客户姓名：{name}",
        f"欠款金额：{_fmt_num(amount)} 元",
        f"逾期天数：{_fmt_num(overdue_days)} 天",
        f"已申请延期次数：{_fmt_num(delay_count)} 次",
        f"上次承诺还款日期：{promised_date}",
        f"当前账户状态：{status}",
        "以上信息通话中可直接使用，向客户提及欠款事实时以本档案为准，不要表述为“正在查询”。",
    ]
    return "\n".join(lines)


def _profile_hotline(db_seed: dict[str, Any]) -> str:
    row = _seed_rows(db_seed, "orders")
    if len(row) < 5:
        return ""
    order_id, product_name, status, eta, issue_note = row[:5]
    note = issue_note or "无备注"
    lines = [
        _PROFILE_TITLE,
        f"订单号：{order_id}",
        f"商品名称：{product_name}",
        f"订单状态：{status}",
        f"预计送达/完成时间（ETA）：{eta}",
        f"订单备注：{note}",
        "以上信息通话中可直接使用，回答订单相关问题时以本档案为准，不要表述为“正在查询”。",
    ]
    return "\n".join(lines)


def _profile_marketing(db_seed: dict[str, Any]) -> str:
    row = _seed_rows(db_seed, "products")
    if len(row) < 5:
        return ""
    product_name, category, price, promotion, description = row[:5]
    lines = [
        _PROFILE_TITLE,
        f"产品名称：{product_name}",
        f"产品类别：{category}",
        f"价格：{_fmt_num(price)} 元",
        f"当前活动：{promotion}",
        f"产品简介：{description}",
        "以上信息通话中可直接使用，介绍产品/价格/活动时以本档案为准，不要表述为“正在查询”。",
    ]
    return "\n".join(lines)


_PROFILE_RENDERERS: dict[str, Any] = {
    "collection": _profile_collection,
    "hotline": _profile_hotline,
    "marketing": _profile_marketing,
}


def render_policy_sections(ap: dict[str, Any]) -> list[str]:
    """按字段存在性渲染业务流程、收口标准、红线及业务知识。

    每阶段仅展示名称和目标；状态优先于流程，声音动作由共享动作表提供。"""
    blocks: list[str] = []
    if ap.get("success_criteria"):
        blocks.append("## 本通电话的收口标准")
        blocks.extend(f"- {c}" for c in ap["success_criteria"])
    if ap.get("business_flow"):
        blocks.append("## 业务流程（推进顺序参考，可跳转与回补；与客户当前状态冲突时以状态应对为先）")
        for st in ap["business_flow"]:
            blocks.append(f"- {st.get('id', '')} {st.get('name', '')}：{st.get('goal', '')}")
    red_lines = [r for r in ap.get("red_lines") or [] if _OMITTED_RED_LINE_SUBSTR not in r]
    if red_lines:
        blocks.append("## 红线（一票否决，任何情况下不得触碰）")
        blocks.extend(f"- {r}" for r in red_lines)
    kb = [k for k in ap.get("knowledge_base") or [] if not k.startswith(_OMITTED_KB_PREFIXES)]
    if kb:
        blocks.append("## 业务知识（回答客户咨询必须依据以下口径，禁止编造）")
        blocks.extend(f"- {k}" for k in kb)
    return blocks


def render_judge_contract(
    domain: DomainSpec,
    scenario: dict[str, Any],
    current_state: str | None = None,
) -> str:
    """渲染与被测客服同源的策略契约及裁判专用关键事件锚点。"""
    ap = scenario.get("agent_policy") or {}
    lines: list[str] = ["# 策略契约（裁判与被测模型同源）"]
    lines.append("## 角色")
    lines.append(ap.get("role_description") or f"{domain.role}客服（{domain.role_type}）")
    if ap.get("communication_style"):
        lines.append("")
        lines.append("## 沟通风格")
        lines.append(str(ap["communication_style"]))
    lines.append("")
    lines.append("## 场景动作约束")
    if ap.get("required_actions"):
        lines.append("必需动作：" + "；".join(ap["required_actions"]))
    if ap.get("forbidden_actions"):
        lines.append("禁止动作：" + "；".join(ap["forbidden_actions"]))
    if ap.get("allowed_actions"):
        lines.append("允许动作：" + "；".join(ap["allowed_actions"]))
    cur = normalize_state(current_state) if current_state else None
    cur_head = (
        f"（当前用户状态：{cur}（{STATE_LABELS[cur]}））"
        if cur and cur in STATE_LABELS
        else (f"（当前用户状态：{cur}）" if cur else "")
    )
    lines.append("")
    state_head = "## 状态应对规则（客服须以客户声音为主要判据识别状态）"
    lines.append(state_head + cur_head)
    for state, rules in domain.policy_rules.items():
        code = normalize_state(state) or state
        tag = f"{code}（{STATE_LABELS[code]}）" if code in STATE_LABELS else str(code)
        mark = " ←当前状态" if cur and code == cur else ""
        lines.append(f"[{tag}]{mark} 必须：{'/'.join(rules.get('required', []))}")
        if rules.get("forbidden"):
            lines.append(f"[{tag}] 禁止：{'/'.join(rules.get('forbidden', []))}")
        if rules.get("allowed"):
            lines.append(f"[{tag}] 可选：{'/'.join(rules.get('allowed', []))}")
    constraints = list(domain.general_constraints)
    constraints = filter_constraints(domain.role, constraints, ap.get("red_lines"))
    constraints += [AGENT_DUAL_RESPONSE_RULE, AGENT_STATE_FIRST_RULE, AGENT_CLOSING_RULE]
    if constraints:
        lines.append("")
        lines.append("## 通用约束（跨状态）")
        lines.extend(f"- {c}" for c in constraints)
    section_lines = render_policy_sections(ap)
    if section_lines:
        lines.append("")
        lines.append("## 业务流程与收口（与被测模型同源）")
        lines.extend(section_lines)
    if scenario.get("expected_key_behavior"):
        lines.append("")
        lines.append("## 关键事件应对锚点（仅裁判可见）")
        lines.append(f"- 关键事件处的预期应对：{scenario['expected_key_behavior']}")
        lines.append("- 说明：收口标准（success_criteria）本轮仅作裁判锚点，不参与自动打分")
    elif scenario.get("stage_at_critical_turn"):
        lines.append("")
        lines.append("## 关键轮流程锚点（仅裁判可见）")
        lines.append(f"- 关键轮所处流程阶段：{scenario['stage_at_critical_turn']}")
        if scenario.get("expected_stage_behavior"):
            lines.append(f"- 该阶段预期行为：{scenario['expected_stage_behavior']}")
        lines.append("- 说明：收口标准（success_criteria）本轮仅作裁判锚点，不参与自动打分")
    lines.append("")
    lines.append("## 声音要求")
    lines.append(ap.get("voice_requirements") or domain.default_voice_requirements or "（无）")
    return "\n".join(lines)


@dataclass
class ToolSpec:
    """可执行服务动作（工具）定义。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema 参数定义
    read_only: bool = True
    impl: Callable[..., str] | None = field(default=None, repr=False)  # (db, **kwargs) -> str


@dataclass
class DBTableSpec:
    """DB 表定义（sqlite 建表语句）。"""

    name: str
    create_sql: str


@dataclass
class DomainSpec:
    """一个客服域的完整定义。"""

    role: str  # collection | marketing | hotline
    role_type: str  # outbound_pressure | outbound_negotiation | inbound_care
    tools: list[ToolSpec] = field(default_factory=list)
    tables: list[DBTableSpec] = field(default_factory=list)
    policy_rules: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    general_constraints: list[str] = field(default_factory=list)
    default_voice_requirements: str = ""

    def render_policy_prompt(
        self,
        agent_policy: dict[str, Any] | None = None,
        turn_budget: dict[str, Any] | None = None,
    ) -> str:
        """渲染客服的业务策略、声音要求和通用约束。

        状态动作由通话指令中的共享动作表提供，soft 预算用于节奏引导。"""
        ap = agent_policy or {}
        lines: list[str] = ["# 角色"]
        lines.append(ap.get("role_description", f"你是{self.role}客服，{self.role_type}。"))
        lines.append("")
        lines.append("# 沟通风格")
        lines.append(ap.get("communication_style", ""))
        section_lines = render_policy_sections(ap)
        if section_lines:
            lines.append("")
            lines.append("# 通话目标与业务规范")
            lines.extend(section_lines)
        lines.append("")
        lines.append("# 声音要求")
        lines.append(ap.get("voice_requirements", self.default_voice_requirements))
        lines.append("")
        lines.append("# 通用约束")
        constraints = self.general_constraints
        constraints = filter_constraints(self.role, constraints, ap.get("red_lines"))
        for c in constraints:
            lines.append(f"- {c}")
        soft = (turn_budget or {}).get("soft")
        if soft:
            lines.append("")
            lines.append(f"本通电话预计约 {soft} 轮内完成主要流程，请合理推进不要拖沓，并为收尾轮留出余量。")
        return "\n".join(lines)

    def init_db(self, seed_data: dict[str, list[tuple]] | None = None) -> sqlite3.Connection:
        """创建内存 sqlite DB 并注入种子数据。"""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        for t in self.tables:
            conn.execute(t.create_sql)
        for table, rows in (seed_data or {}).items():
            if not rows:
                continue
            placeholders = ",".join("?" * len(rows[0]))
            conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
        conn.commit()
        return conn

    def describe_db(self) -> str:
        """供 LLM 看到的 DB 结构说明。"""
        return "\n".join(
            f"- {t.name}: {t.create_sql.split('(', 1)[1].rsplit(')', 1)[0]}" for t in self.tables
        )

    def execute_tool(self, db: sqlite3.Connection, name: str, args: dict[str, Any]) -> str:
        tool = next((t for t in self.tools if t.name == name), None)
        if tool is None:
            return f"ERROR: 未知工具 {name}"
        if tool.impl is None:
            return f"ERROR: 工具 {name} 未实现"
        try:
            return tool.impl(db, **args)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {type(exc).__name__}: {exc}"

    @staticmethod
    def load_tasks(scenarios_path: str | Path) -> list[dict[str, Any]]:
        """读取 data/benchmark/scenarios.jsonl。"""
        path = Path(scenarios_path)
        if not path.exists():
            return []
        tasks = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks
