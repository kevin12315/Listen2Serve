"""催收域（外呼·施压）：提醒还款/缴费，施加适度压力，促成承诺。"""

from __future__ import annotations

import sqlite3

from listen2serve.domains.base import DBTableSpec, DomainSpec, ToolSpec


def _query_debt(db: sqlite3.Connection, customer_name: str) -> str:
    row = db.execute(
        "SELECT customer_name, amount, overdue_days, delay_count, promised_date, status "
        "FROM customers WHERE customer_name = ?",
        (customer_name,),
    ).fetchone()
    if row is None:
        return "未找到该客户记录"
    return (
        f"客户 {row['customer_name']}：欠款 {row['amount']} 元，逾期 {row['overdue_days']} 天，"
        f"已延期 {row['delay_count']} 次，上次承诺还款日 {row['promised_date']}，状态：{row['status']}"
    )


def _register_promise(db: sqlite3.Connection, customer_name: str, amount: float, deadline: str) -> str:
    db.execute(
        "INSERT INTO promises (customer_name, amount, deadline, ts) VALUES (?, ?, ?, datetime('now'))",
        (customer_name, amount, deadline),
    )
    db.commit()
    return f"已登记 {customer_name} 的还款承诺：{amount} 元，截止 {deadline}"


def _register_callback(db: sqlite3.Connection, customer_name: str, time_slot: str) -> str:
    db.execute(
        "INSERT INTO callbacks (customer_name, time_slot, ts) VALUES (?, ?, datetime('now'))",
        (customer_name, time_slot),
    )
    db.commit()
    return f"已约定回电：{customer_name}，{time_slot}"


def build_domain() -> DomainSpec:
    tools = [
        ToolSpec(
            name="query_debt",
            description="查询客户的欠款金额、逾期天数、延期次数与状态",
            parameters={
                "type": "object",
                "properties": {"customer_name": {"type": "string", "description": "客户姓名"}},
                "required": ["customer_name"],
            },
            read_only=True,
            impl=_query_debt,
        ),
        ToolSpec(
            name="register_promise",
            description="登记客户的具体还款承诺（金额+时间）",
            parameters={
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string"},
                    "amount": {"type": "number", "description": "承诺还款金额"},
                    "deadline": {"type": "string", "description": "承诺还款日期"},
                },
                "required": ["customer_name", "amount", "deadline"],
            },
            read_only=False,
            impl=_register_promise,
        ),
        ToolSpec(
            name="register_callback",
            description="登记约定回电时间",
            parameters={
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string"},
                    "time_slot": {"type": "string", "description": "回电时间段"},
                },
                "required": ["customer_name", "time_slot"],
            },
            read_only=False,
            impl=_register_callback,
        ),
    ]

    tables = [
        DBTableSpec(
            name="customers",
            create_sql=(
                "CREATE TABLE customers ("
                "customer_name TEXT, amount REAL, overdue_days INT, "
                "delay_count INT, promised_date TEXT, status TEXT)"
            ),
        ),
        DBTableSpec(
            name="promises",
            create_sql=(
                "CREATE TABLE promises (customer_name TEXT, amount REAL, deadline TEXT, ts TEXT)"
            ),
        ),
        DBTableSpec(
            name="callbacks",
            create_sql="CREATE TABLE callbacks (customer_name TEXT, time_slot TEXT, ts TEXT)",
        ),
    ]

    # 内部方案（v6.0）：新 6 态 policy_rules（策略要点见计划 表）
    # 内部方案（v6.0）：新 6 态 policy_rules（策略要点见计划 表）
    policy_rules = {
        "neutral": {
            "required": ["先确认客户身份", "简明告知来电来意"],
            "forbidden": ["急于施压或推销", "核身前追问欠款/身份敏感信息"],
            "allowed": ["询问方便接听的时间"],
        },
        "cooperative": {
            "required": ["抓住配合势头推进流程", "确认身份与欠款事实", "落实到具体数字与时间，促成还款承诺"],
            "forbidden": ["过度承诺", "承诺减免（无权限）"],
            "allowed": ["提供分期选项", "说明逾期后果（客观）", "复述承诺要点"],
        },
        "displeased": {
            "required": ["先安抚认可感受，再给方案", "客观说明当前欠款状态", "引导到可执行的还款方案"],
            "forbidden": ["继续高压施压", "威胁恐吓", "承诺减免（无权限）"],
            "allowed": ["提供分期/延期选项", "约定回电"],
        },
        "urgent": {
            "required": ["压缩流程，先给结论再补细节", "快速确认核心信息", "给出简短方案"],
            "forbidden": ["强行展开长篇说明", "拖延时间"],
            "allowed": ["提供自助还款渠道说明", "约定方便时间回电"],
        },
        "doubtful": {
            "required": ["先自证身份（工号/机构）", "出示可核实的欠款信息", "放慢节奏耐心解释"],
            "forbidden": ["强行推进业务", "回避核实诉求", "语气轻蔑"],
            "allowed": ["提供官方核实渠道", "约定回电"],
        },
        "hushed": {
            "required": ["简短沟通不展开细节", "主动提出改约时间或极简处理"],
            "forbidden": ["透露欠款细节给在场第三方", "继续询问身份/账号信息"],
            "allowed": ["约定私密时间回电", "改用短信/自助渠道沟通"],
        },
    }

    general_constraints = [
        "不得威胁、辱骂或使用恐吓性语言",
        "不得透露欠款信息给第三方",
        "不得使用'一定''马上'等过度承诺词",
        "不得复读机式回复（任意连续10字不得与历史完全一致）",
        "逾期后果须客观陈述（如征信影响），非威胁",
        "不得承诺减免或延期（无权限）",
        "回复控制在50字以内，语意完整",
    ]

    return DomainSpec(
        role="collection",
        role_type="outbound_pressure",
        tools=tools,
        tables=tables,
        policy_rules=policy_rules,
        general_constraints=general_constraints,
        default_voice_requirements="语速中等，语气坚定但不攻击，施压时音量略提但不喊叫，协定时语速放慢",
    )
