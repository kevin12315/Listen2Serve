# -*- coding: utf-8 -*-
"""用户侧私拟事实 user_facts 测试（Plan I v6.0 口径）。

v5.5 生成器（gen_scenarios/inject_user_facts）随逐轮脚本管线退役；本文件改为
对 v6.0 数据本体做结构校验 + 骨架 prompt 渲染测试：
1. 结构完整性：三域字段按角色裁剪（collection 三张底牌 + planS3 改动 4 外迁的两个数字字段 /
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
        """催收域 user_facts：三张底牌必在，外加 planS3 改动 4 外迁过来的两个数字字段。

        契约变更（planS3 §4 改动 4，2026-09-06）：F7 的病是「identity 里数字 ≥3 个 ⇒
        常驻背景被演员当台词素材念出来」，修法是把**处境类数字**（欠款金额、逾期天数）
        从 identity 外迁到 user_facts（年龄留在 identity，因为
        `gen_tts_instructions._IDENTITY_RE` 要靠它派生已冻结的 identity_phrase）。
        所以键集从「恰好三个」放宽为「三个底牌 + 至多两个外迁字段」，但**不放松**：
        外迁的两个键必须成对出现，且其值必须与 `db_seed` 档案里的金额/天数逐项相符 ——
        这条一致性此前没有任何测试守，迁移写错一个数字是不会被发现的。
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
            assert (set(uf) & migrated_keys) in (set(), migrated_keys), \
                f"{s['scenario_id']} 外迁字段只来了一半：{sorted(set(uf) & migrated_keys)}"
            assert 100 <= uf["planned_amount"] <= amount, s["scenario_id"]
            if migrated_keys <= set(uf):
                assert uf["debt_amount"] == f"{amount_to_spoken_zh(amount)}元", \
                    f"{s['scenario_id']} 外迁金额与 db_seed 不符：{uf['debt_amount']} vs {amount}"
                assert uf["overdue_days"] == f"逾期 {days} 天", \
                    f"{s['scenario_id']} 外迁天数与 db_seed 不符：{uf['overdue_days']} vs {days}"
                migrated_bases.add(s["base_scenario_id"])
            checked += 1
        assert checked > 0
        # F7 命中的 53 个 base 全部走了外迁（planS3 §3.3 的实测命中数）；少于它说明有 base 漏改。
        # 按 base 去重：三层展开共享同一份 user_facts，直接数行会得到 159。
        # 53 是内部全库 202 base 的口径；发布切片 145 base 里命中的是 34 个（按态分层抽样）
        assert len(migrated_bases) >= 30, \
            f"外迁只覆盖 {len(migrated_bases)} 个 base，少于 F7 命中的 53 个"

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
        """T1/T2/T3 变体共享同一 user_facts（骨架共享契约）。"""
        by_base: dict[str, list[dict]] = {}
        for s in scenarios:
            by_base.setdefault(s["base_scenario_id"], []).append(s)
        for base, group in by_base.items():
            facts = {json.dumps(g["user_script"].get("user_facts") or {},
                                ensure_ascii=False, sort_keys=True) for g in group}
            assert len(facts) == 1, f"{base} 变体 user_facts 不一致"

    def test_disclosure_plan_covers_facts(self, scenarios):
        """Plan I：disclosure_plan 与 user_facts 同键（有底牌必注披露时机）。"""
        for s in scenarios:
            us = s["user_script"]
            facts = us.get("user_facts") or {}
            dp = us.get("disclosure_plan") or {}
            if facts:
                assert set(dp) == set(facts), s["scenario_id"]
                assert all(str(v).strip() for v in dp.values()), s["scenario_id"]


# ---- 运行时渲染（user_simulator，Plan I v6.0 骨架 prompt）----
from listen2serve.runtime.user_simulator import UserSimulator  # noqa: E402

USER_FACTS = {"planned_amount": 1800, "planned_date": "最迟下周五",
              "income_note": "店里最近回款慢，要等到月底"}


def _v6_script(**over) -> dict:
    sc = {
        "identity": "孙超，38 岁上班族，有一笔 3600 元欠款已逾期 10 天",
        "user_facts": dict(USER_FACTS),
        "goal": "配合还款",
        "user_goal": "想还清这笔欠款",
        "mission": ["接听电话确认身份", "弄清来意",
                    "【关键事件】摊牌还款能力", "确认后道别"],
        "key_event": {"description": "摊牌还款能力", "state": "cooperative",
                      "trigger_hint": "客服提出要求后",
                      "interrupt": False},
        "disclosure_plan": {"planned_amount": "客服问到还款能力前不主动说",
                            "planned_date": "客服问到还款能力前不主动说",
                            "income_note": "被追问为什么不能全额时才解释"},
        "turn_budget": {"soft": 5, "hard": 9},
    }
    sc.update(over)
    return sc


def _build_user_msg(script: dict) -> str:
    sim = UserSimulator(script, role="collection", dynamics="all_positive",
                        db_seed=COL_DB_SEED, leakage_label="T1", user_gender="male",
                        instruction_entry=fake_instruction_entry("cooperative"))
    return sim._build_messages(1, "", forced=False)[1]["content"]


class TestFactAnchorRendering:
    def test_user_facts_section_rendered(self):
        user = _build_user_msg(_v6_script())
        # 独立小节，与 db_seed 系统事实分开标注
        assert "【用户本人掌握的事实】\n" in user
        assert "- 拟还金额：一千八百元" in user   # planN §3.3：金额口语中文化，防 LLM 读丢位
        assert "- 拟还金额：1800元" not in user
        assert "- 拟还日期：最迟下周五" in user
        assert "- 收入与资金安排：店里最近回款慢，要等到月底" in user
        assert "customers: 孙超 | 3600" in user  # db_seed 系统事实锚不受影响
        # 披露时机小节（Plan I：治病灶三）
        assert "【披露时机】" in user
        assert "拟还金额：客服问到还款能力前不主动说" in user
        # 约束措辞：三来源并列 + 未提供事实不得给出具体数字或日期
        assert "【身份】【事实锚】与" in user
        assert "【用户本人掌握的事实】；严禁虚构任何新事实" in user
        assert "不得给出" in user and "具体数字或日期" in user

    def test_legacy_data_without_user_facts(self):
        """无 user_facts 键：回落不崩，事实与披露小节省略，约束措辞仍在。"""
        sc = _v6_script()
        sc.pop("user_facts")
        sc.pop("disclosure_plan")
        user = _build_user_msg(sc)
        assert "【用户本人掌握的事实】\n" not in user
        assert "【披露时机】" not in user
        assert "事实锚定硬约束" in user

    def test_empty_user_facts_omitted(self):
        """空 user_facts：小节省略。"""
        user = _build_user_msg(_v6_script(user_facts={}, disclosure_plan={}))
        assert "【用户本人掌握的事实】\n" not in user
