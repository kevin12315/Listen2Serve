"""客户信息档案回归测试（2026-08-14：评测不发起 function call，db_seed 直接注入初始 prompt）。

防回归点：
1. render_customer_profile 三域渲染：db_seed 单行记录自然语言化，字段齐全、
   标题含「系统已预先提供，通话中无需查询、直接作为事实依据」引导语；
2. 空值回落：db_seed 为 None/{}、预期表缺失或为空、未知域 → 返回空串（旧场景兼容）；
3. 裁判契约同源注入：policy_judge/dialogue_metrics 的 _render_rules 返回值末尾
   追加同源档案（render_judge_contract 本体输出不变，档案为纯追加）；
   无 db_seed 的场景输出与 render_judge_contract 逐字节一致（既有薄封装语义保留）。
"""

from __future__ import annotations

from listen2serve.domains import get_domain
from listen2serve.domains.base import render_customer_profile, render_judge_contract
from listen2serve.evaluation import dialogue_metrics, policy_judge

PROFILE_TITLE_MARK = "客户信息档案（系统已预先提供，通话中无需查询、直接作为事实依据）"

COLLECTION_SEED = {"customers": [["孙超", 3600.0, 10, 1, "2025-05-28", "overdue"]]}
HOTLINE_SEED = {"orders": [["ORD0001", "示例商品", "运输中", "2-3天", ""]]}
MARKETING_SEED = {"products": [["产品3980", "常规", 3980, "满减活动", "示例产品"]]}


class TestProfileRendering:
    def test_collection_profile(self):
        text = render_customer_profile("collection", COLLECTION_SEED)
        assert PROFILE_TITLE_MARK in text
        for frag in ("孙超", "3600 元", "逾期天数：10 天", "已申请延期次数：1 次",
                     "上次承诺还款日期：2025-05-28", "当前账户状态：overdue"):
            assert frag in text, f"collection 档案缺字段片段: {frag}"
        # 浮点金额去小数尾巴（3600.0 → 3600）
        assert "3600.0" not in text

    def test_hotline_profile(self):
        text = render_customer_profile("hotline", HOTLINE_SEED)
        assert PROFILE_TITLE_MARK in text
        for frag in ("订单号：ORD0001", "商品名称：示例商品", "订单状态：运输中",
                     "预计送达/完成时间（ETA）：2-3天", "订单备注：无备注"):
            assert frag in text, f"hotline 档案缺字段片段: {frag}"

    def test_marketing_profile(self):
        text = render_customer_profile("marketing", MARKETING_SEED)
        assert PROFILE_TITLE_MARK in text
        for frag in ("产品名称：产品3980", "产品类别：常规", "价格：3980 元",
                     "当前活动：满减活动", "产品简介：示例产品"):
            assert frag in text, f"marketing 档案缺字段片段: {frag}"

    def test_accepts_domain_spec(self):
        """domain 参数兼容 DomainSpec 与角色名两种传法（run_batch/裁判分别使用）。"""
        assert render_customer_profile(get_domain("hotline"), HOTLINE_SEED) == \
            render_customer_profile("hotline", HOTLINE_SEED)

    def test_tuple_row_format(self):
        """行格式兼容 tuple（init_db 归一化后的形态）。"""
        seed = {"customers": [tuple(COLLECTION_SEED["customers"][0])]}
        assert render_customer_profile("collection", seed) == \
            render_customer_profile("collection", COLLECTION_SEED)


class TestProfileEmptyFallback:
    def test_none_and_empty_seed(self):
        for role in ("collection", "hotline", "marketing"):
            assert render_customer_profile(role, None) == ""
            assert render_customer_profile(role, {}) == ""

    def test_missing_or_empty_table(self):
        assert render_customer_profile("collection", {"customers": []}) == ""
        assert render_customer_profile("hotline", {"orders": []}) == ""
        assert render_customer_profile("marketing", {}) == ""
        # 预期表缺失（旧场景/异常数据）→ 空串
        assert render_customer_profile("collection", {"orders": [["x"]]}) == ""

    def test_unknown_domain(self):
        assert render_customer_profile("unknown_role", COLLECTION_SEED) == ""

    def test_short_row(self):
        """行字段不足（异常数据）→ 空串，不抛异常。"""
        assert render_customer_profile("collection", {"customers": [["孙超", 3600]]}) == ""


class TestJudgeContractProfileInjection:
    """裁判契约同源注入：_render_rules 追加档案；render_judge_contract 本体不变。"""

    _SCENARIO_WITH_SEED = {
        "scenario_id": "COL-NEG-PF",
        "role": "collection",
        "oracle_state": "N5_resistant",
        "agent_policy": {"role_description": "你是催收客服", "required_actions": ["确认身份"]},
        "db_seed": COLLECTION_SEED,
    }
    _SCENARIO_NO_SEED = {
        "scenario_id": "COL-NEG-01",
        "role": "collection",
        "oracle_state": "N5_resistant",
        "agent_policy": {"role_description": "你是催收客服"},
    }

    def test_policy_judge_rules_append_profile(self):
        domain = get_domain("collection")
        out = policy_judge._render_rules(domain, self._SCENARIO_WITH_SEED)
        contract = render_judge_contract(domain, self._SCENARIO_WITH_SEED)
        profile = render_customer_profile(domain, COLLECTION_SEED)
        assert out == f"{contract}\n\n{profile}"  # 纯追加，契约本体在前
        assert PROFILE_TITLE_MARK in out and "欠款金额：3600 元" in out

    def test_dialogue_metrics_rules_append_profile(self):
        domain = get_domain("collection")
        out = dialogue_metrics._render_rules(domain, self._SCENARIO_WITH_SEED)
        contract = render_judge_contract(domain, self._SCENARIO_WITH_SEED)
        profile = render_customer_profile(domain, COLLECTION_SEED)
        assert out == f"{contract}\n\n{profile}"

    def test_no_seed_byte_identical_to_contract(self):
        """无 db_seed（旧场景）：_render_rules 输出与契约逐字节一致（既有薄封装语义保留）。"""
        domain = get_domain("collection")
        assert policy_judge._render_rules(domain, self._SCENARIO_NO_SEED) == \
            render_judge_contract(domain, self._SCENARIO_NO_SEED)
        assert dialogue_metrics._render_rules(domain, self._SCENARIO_NO_SEED) == \
            render_judge_contract(domain, self._SCENARIO_NO_SEED)
