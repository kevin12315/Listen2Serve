"""电话客服域（呼入·安抚）：解决用户问题，安抚情绪，提供方案。"""

from __future__ import annotations

import sqlite3

from listen2serve.domains.base import DBTableSpec, DomainSpec, ToolSpec


def _query_order(db: sqlite3.Connection, order_id: str) -> str:
    row = db.execute(
        "SELECT order_id, product_name, status, eta, issue_note FROM orders WHERE order_id = ?",
        (order_id,),
    ).fetchone()
    if row is None:
        return "未找到该订单"
    return (
        f"订单 {row['order_id']}（{row['product_name']}）：状态 {row['status']}，"
        f"预计 {row['eta']}，备注：{row['issue_note'] or '无'}"
    )


def _query_logistics(db: sqlite3.Connection, order_id: str) -> str:
    rows = db.execute(
        "SELECT ts, event, location FROM logistics WHERE order_id = ? ORDER BY ts",
        (order_id,),
    ).fetchall()
    if not rows:
        return "暂无物流记录"
    return "；".join(f"{r['ts']} {r['event']}（{r['location']}）" for r in rows)


def _register_issue(db: sqlite3.Connection, order_id: str, description: str) -> str:
    db.execute(
        "INSERT INTO issues (order_id, description, ts, status) VALUES (?, ?, datetime('now'), 'open')",
        (order_id, description),
    )
    db.commit()
    return f"已登记问题（订单 {order_id}）：{description}，工单已开启"


def build_domain() -> DomainSpec:
    tools = [
        ToolSpec(
            name="query_order",
            description="查询订单状态、预计时间与备注",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            read_only=True,
            impl=_query_order,
        ),
        ToolSpec(
            name="query_logistics",
            description="查询订单物流轨迹",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            read_only=True,
            impl=_query_logistics,
        ),
        ToolSpec(
            name="register_issue",
            description="登记用户反馈的问题，开启工单",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "description": {"type": "string", "description": "问题描述"},
                },
                "required": ["order_id", "description"],
            },
            read_only=False,
            impl=_register_issue,
        ),
    ]

    tables = [
        DBTableSpec(
            name="orders",
            create_sql=(
                "CREATE TABLE orders (order_id TEXT, product_name TEXT, status TEXT, "
                "eta TEXT, issue_note TEXT)"
            ),
        ),
        DBTableSpec(
            name="logistics",
            create_sql=(
                "CREATE TABLE logistics (order_id TEXT, ts TEXT, event TEXT, location TEXT)"
            ),
        ),
        DBTableSpec(
            name="issues",
            create_sql="CREATE TABLE issues (order_id TEXT, description TEXT, ts TEXT, status TEXT)",
        ),
    ]

    # 内部方案（v6.0）：新 6 态 policy_rules（策略要点见计划 表）
    policy_rules = {
        "neutral": {
            "required": ["先确认用户身份与待处理事项", "简明告知处理方向"],
            "forbidden": ["急于施压或推销", "追问与核实无关的敏感信息"],
            "allowed": ["询问方便接听的时间"],
        },
        "cooperative": {
            "required": ["抓住配合势头确认问题全貌", "给出方案与时间预期", "确认是否解决"],
            "forbidden": ["敷衍了事", "夸大承诺"],
            "allowed": ["提供额外帮助", "邀请评价"],
        },
        "displeased": {
            "required": ["先共情道歉、认可感受", "认真记录诉求", "给出可执行的下一步"],
            "forbidden": ["敷衍推诿", "否定用户感受", "引导投诉升级"],
            "allowed": ["提供补偿方案", "升级工单"],
        },
        "urgent": {
            "required": ["快速定位问题", "先给结论再补细节", "控制时长"],
            "forbidden": ["长篇解释", "让用户等待"],
            "allowed": ["提供自助渠道", "约定回电"],
        },
        "doubtful": {
            "required": ["自证身份（工号）", "出示可验证的订单信息", "放慢节奏分步解释并确认理解"],
            "forbidden": ["使用'一定''保证'等绝对化用语", "否认平台过错", "一次给出过多信息"],
            "allowed": ["提供核实渠道", "举例说明", "升级处理"],
        },
        "hushed": {
            "required": ["简短回应不展开细节", "主动提出改约时间或极简处理"],
            "forbidden": ["强制询问身份证号等敏感信息", "长篇说明"],
            "allowed": ["短信验证码核验", "约定安全时间"],
        },
    }

    general_constraints = [
        "不得敷衍、推诿（'这不归我管'）",
        "不得使用'一定''马上''全力解决'等过度承诺词",
        "不得复读机式回复",
        "不得在回复中传达'系统误判/平台出错'等责任定性",
        "不得建议用户寻求外部渠道（除非政策允许）",
        "回复控制在50字以内，语意完整",
    ]

    return DomainSpec(
        role="hotline",
        role_type="inbound_care",
        tools=tools,
        tables=tables,
        policy_rules=policy_rules,
        general_constraints=general_constraints,
        default_voice_requirements="语速中等偏慢，耐心温和，用户困惑时放慢语速，安抚时语气真诚",
    )
