"""用户私有事实、披露时机和提示词边界检查。

对场景数据进行结构校验与提示词渲染测试：
1. 结构完整性：三域字段按角色裁剪（collection 三项私有事实与数值字段 /
   hotline issue_detail / marketing budget_or_concern）；催收拟还金额不超过欠款金额；
   外迁字段的值必须与 db_seed 档案逐项相符；
2. 披露时机 disclosure_plan 与 user_facts 同键；
3. 事实锚渲染含【用户本人掌握的事实】与【披露时机】独立小节，约束措辞
   （来源未提供的事实不得给出具体数字或日期）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import fake_instruction_entry

ROOT = Path(__file__).resolve().parents[1]
SCEN_PATH = ROOT / "data/benchmark/scenarios.jsonl"

COL_DB_SEED = {"customers": [["孙超", 3600, 10, 1, "2025-05-28", "overdue"]]}


@pytest.fixture(scope="module")
def scenarios() -> list[dict]:
    if not SCEN_PATH.exists():
        pytest.skip("scenarios.jsonl 不存在")
    return [json.loads(line) for line in SCEN_PATH.open(encoding="utf-8") if line.strip()]


class TestUserFactsStructure:
    """三域字段裁剪与取值合法性（数据本体校验）。"""

    def test_collection_fields(self, scenarios):
        """催收域包含还款计划、收入说明以及可选的欠款事实。

        身份描述只保留人物背景，数值事实集中在 user_facts。
        """
        from listen2serve.runtime.user_simulator import amount_to_spoken_zh

        base_keys = {"planned_amount", "planned_date", "income_note"}
        migrated_keys = {"debt_amount", "overdue_days"}
        checked = 0
        migrated_bases: set[str] = set()
        for s in scenarios:
            if s["role"] != "collection":
                continue
            uf = s["user_script"].get("user_facts") or {}
            rows = (s.get("db_seed") or {}).get("customers") or []
            amount = list(rows[0])[1] if rows else None
            days = list(rows[0])[2] if rows else None
            if s.get("sub_domain") == "第三方寻人":
                continue  # 非债务人分支无还款底牌
            assert base_keys <= set(uf) <= base_keys | migrated_keys, s["scenario_id"]
            assert (set(uf) & migrated_keys) in (set(), migrated_keys), (
                f"{s['scenario_id']} 外迁字段只来了一半：{sorted(set(uf) & migrated_keys)}"
            )
            assert 100 <= uf["planned_amount"] <= amount, s["scenario_id"]
            if migrated_keys <= set(uf):
                assert uf["debt_amount"] == f"{amount_to_spoken_zh(amount)}元", (
                    f"{s['scenario_id']} 外迁金额与 db_seed 不符：{uf['debt_amount']} vs {amount}"
                )
                assert uf["overdue_days"] == f"逾期 {days} 天", (
                    f"{s['scenario_id']} 外迁天数与 db_seed 不符：{uf['overdue_days']} vs {days}"
                )
                migrated_bases.add(s["base_scenario_id"])
            checked += 1
        assert checked > 0

        # 按 base 去重：三层展开共享同一份 user_facts，直接数行会得到 159。
        # 53 是内部全库 202 base 的口径；发布切片 145 base 里命中的是 34 个（按态分层抽样）
        assert len(migrated_bases) >= 30, f"外迁只覆盖 {len(migrated_bases)} 个 base，少于 F7 命中的 53 个"

    def test_hotline_fields(self, scenarios):
        checked = 0
        for s in scenarios:
            if s["role"] != "hotline":
                continue
            uf = s["user_script"].get("user_facts") or {}
            assert set(uf) <= {"issue_detail"}, s["scenario_id"]
            checked += 1
        assert checked > 0

    def test_marketing_fields(self, scenarios):
        checked = 0
        for s in scenarios:
            if s["role"] != "marketing":
                continue
            uf = s["user_script"].get("user_facts") or {}
            assert set(uf) <= {"budget_or_concern"}, s["scenario_id"]
            checked += 1
        assert checked > 0

    def test_variants_share_facts(self, scenarios):
        """同一基础场景的两条件共享同一 user_facts（骨架共享契约）。"""
        by_base: dict[str, list[dict]] = {}
        for s in scenarios:
            by_base.setdefault(s["base_scenario_id"], []).append(s)
        for base, group in by_base.items():
            facts = {
                json.dumps(g["user_script"].get("user_facts") or {}, ensure_ascii=False, sort_keys=True)
                for g in group
            }
            assert len(facts) == 1, f"{base} 变体 user_facts 不一致"

    def test_disclosure_plan_covers_facts(self, scenarios):
        """disclosure_plan 与 user_facts 同键（有底牌必注披露时机）。"""
        for s in scenarios:
            us = s["user_script"]
            facts = us.get("user_facts") or {}
            dp = us.get("disclosure_plan") or {}
            if facts:
                assert set(dp) == set(facts), s["scenario_id"]
                assert all(str(v).strip() for v in dp.values()), s["scenario_id"]


from listen2serve.runtime.user_simulator import UserSimulator

USER_FACTS = {
    "planned_amount": 1800,
    "planned_date": "最迟下周五",
    "income_note": "店里最近回款慢，要等到月底",
}


def _script(**over) -> dict:
    sc = {
        "identity": "孙超，38 岁上班族，有一笔 3600 元欠款已逾期 10 天",
        "user_facts": dict(USER_FACTS),
        "goal": "配合还款",
        "user_goal": "想还清这笔欠款",
        "mission": ["接听电话确认身份", "弄清来意", "【关键事件】摊牌还款能力", "确认后道别"],
        "key_event": {
            "description": "摊牌还款能力",
            "state": "cooperative",
            "trigger_hint": "客服提出要求后",
            "interrupt": False,
        },
        "disclosure_plan": {
            "planned_amount": "客服问到还款能力前不主动说",
            "planned_date": "客服问到还款能力前不主动说",
            "income_note": "被追问为什么不能全额时才解释",
        },
        "turn_budget": {"soft": 5, "hard": 9},
    }
    sc.update(over)
    return sc


def _build_user_msg(script: dict) -> str:
    sim = UserSimulator(
        script,
        role="collection",
        dynamics="all_positive",
        db_seed=COL_DB_SEED,
        leakage_label="T1",
        user_gender="male",
        instruction_entry=fake_instruction_entry("cooperative"),
    )
    return sim._build_messages(1, "", forced=False)[1]["content"]


class TestFactAnchorRendering:
    def test_user_facts_section_rendered(self):
        user = _build_user_msg(_script())
        # 独立小节，与 db_seed 系统事实分开标注
        assert "【用户本人掌握的事实】\n" in user
        assert "- 拟还金额：一千八百元" in user
        assert "- 拟还金额：1800元" not in user
        assert "- 拟还日期：最迟下周五" in user
        assert "- 收入与资金安排：店里最近回款慢，要等到月底" in user
        assert "customers: 孙超 | 3600" in user  # db_seed 系统事实锚不受影响
        # 披露时机小节（治病灶三）
        assert "【披露时机】" in user
        assert "拟还金额：客服问到还款能力前不主动说" in user
        # 约束措辞：三来源并列 + 未提供事实不得给出具体数字或日期
        assert "【身份】【事实锚】与" in user
        assert "【用户本人掌握的事实】；严禁虚构任何新事实" in user
        assert "不得给出" in user and "具体数字或日期" in user

    def test_legacy_data_without_user_facts(self):
        """无 user_facts 键：回落不崩，事实与披露小节省略，约束措辞仍在。"""
        sc = _script()
        sc.pop("user_facts")
        sc.pop("disclosure_plan")
        user = _build_user_msg(sc)
        assert "【用户本人掌握的事实】\n" not in user
        assert "【披露时机】" not in user
        assert "事实锚定硬约束" in user

    def test_empty_user_facts_omitted(self):
        """空 user_facts：小节省略。"""
        user = _build_user_msg(_script(user_facts={}, disclosure_plan={}))
        assert "【用户本人掌握的事实】\n" not in user
