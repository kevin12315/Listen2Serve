"""营销域（外呼·谈判）：推介产品/服务，处理异议，促成转化或预约。"""

from __future__ import annotations

import sqlite3

from listen2serve.domains.base import DBTableSpec, DomainSpec, ToolSpec


def _query_product(db: sqlite3.Connection, product_name: str) -> str:
    row = db.execute(
        "SELECT product_name, category, price, promotion, description FROM products WHERE product_name = ?",
        (product_name,),
    ).fetchone()
    if row is None:
        return "未找到该产品信息"
    return (
        f"{row['product_name']}（{row['category']}）：价格 {row['price']}，"
        f"当前活动：{row['promotion']}，简介：{row['description']}"
    )


def _register_lead(db: sqlite3.Connection, customer_name: str, interest: str) -> str:
    db.execute(
        "INSERT INTO leads (customer_name, interest, ts) VALUES (?, ?, datetime('now'))",
        (customer_name, interest),
    )
    db.commit()
    return f"已登记客户 {customer_name} 的兴趣：{interest}"


def _register_appointment(db: sqlite3.Connection, customer_name: str, time_slot: str) -> str:
    db.execute(
        "INSERT INTO appointments (customer_name, time_slot, ts) VALUES (?, ?, datetime('now'))",
        (customer_name, time_slot),
    )
    db.commit()
    return f"已预约：{customer_name}，{time_slot}"


def build_domain() -> DomainSpec:
    tools = [
        ToolSpec(
            name="query_product",
            description="查询产品/活动的价格、当前促销与简介",
            parameters={
                "type": "object",
                "properties": {"product_name": {"type": "string"}},
                "required": ["product_name"],
            },
            read_only=True,
            impl=_query_product,
        ),
        ToolSpec(
            name="register_lead",
            description="登记客户意向（感兴趣的产品/方向）",
            parameters={
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string"},
                    "interest": {"type": "string", "description": "客户感兴趣的方面"},
                },
                "required": ["customer_name", "interest"],
            },
            read_only=False,
            impl=_register_lead,
        ),
        ToolSpec(
            name="register_appointment",
            description="登记客户的预约（回访/试用/详细沟通）",
            parameters={
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string"},
                    "time_slot": {"type": "string"},
                },
                "required": ["customer_name", "time_slot"],
            },
            read_only=False,
            impl=_register_appointment,
        ),
    ]

    tables = [
        DBTableSpec(
            name="products",
            create_sql=(
                "CREATE TABLE products (product_name TEXT, category TEXT, price REAL, "
                "promotion TEXT, description TEXT)"
            ),
        ),
        DBTableSpec(
            name="leads",
            create_sql="CREATE TABLE leads (customer_name TEXT, interest TEXT, ts TEXT)",
        ),
        DBTableSpec(
            name="appointments",
            create_sql="CREATE TABLE appointments (customer_name TEXT, time_slot TEXT, ts TEXT)",
        ),
    ]

    # 内部方案（v6.0）：新 6 态 policy_rules（策略要点见计划 表）
    policy_rules = {
        "neutral": {
            "required": ["先确认客户身份", "一句话简明告知来意"],
            "forbidden": ["急于推销施压", "客户未表态前追问个人信息"],
            "allowed": ["询问方便接听的时间"],
        },
        "cooperative": {
            "required": ["抓住配合势头探测需求", "给出匹配方案", "促成转化或预约"],
            "forbidden": ["过度承诺效果", "捏造信息"],
            "allowed": ["提供优惠/赠品", "预约详细沟通"],
        },
        "displeased": {
            "required": ["停止产品推介", "先安抚认可感受", "给客户选择权"],
            "forbidden": ["继续推销", "回避用户不满"],
            "allowed": ["转交售后/客服", "约定回电"],
        },
        "urgent": {
            "required": ["快速确认是否方便", "先给一句话摘要再补细节", "约定方便时间"],
            "forbidden": ["展开完整话术", "要求立即决策"],
            "allowed": ["发送资料链接", "约定回电"],
        },
        "doubtful": {
            "required": ["自证身份与来源", "出示可验证信息", "放慢节奏处理具体疑虑"],
            "forbidden": ["回避质疑", "施压促单", "堆砌卖点"],
            "allowed": ["提供官方资料", "提供对比说明", "约定回电"],
        },
        "hushed": {
            "required": ["简短沟通不展开话术", "主动提出改约时间或改用书面渠道"],
            "forbidden": ["继续询问个人信息", "制造虚假紧迫"],
            "allowed": ["约定私密时间", "发送官方介绍"],
        },
    }

    general_constraints = [
        "不得制造虚假紧迫感（'仅剩最后X个名额'除非属实）",
        "不得过度承诺效果",
        "不得在用户明确拒绝后继续施压推销",
        "不得使用'一定''保证'等绝对化用语",
        "不得复读机式回复",
        "用户表示不方便/不感兴趣时，应给出礼貌退路",
        "回复控制在50字以内，语意完整",
    ]

    return DomainSpec(
        role="marketing",
        role_type="outbound_negotiation",
        tools=tools,
        tables=tables,
        policy_rules=policy_rules,
        general_constraints=general_constraints,
        default_voice_requirements="语速中等偏快，热情但低压力，用户拒绝时语速放慢、语气真诚",
    )
