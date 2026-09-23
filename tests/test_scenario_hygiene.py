# -*- coding: utf-8 -*-
"""planS3 §5：演绎侧数据卫生 CI —— 七类缺陷 + 三条新增断言，任一命中即失败。

**实现纪律（planS3 §5，本仓有前科，逐条照办）**

1. **规则与 `scripts/final_plan_sections2.py::_audit()` 同源。** 判据的唯一实现点是
   `src/listen2serve/data/hygiene.py`；`_audit()` 现在只是转发到它（planS3 §5 纪律 1 的
   第二个选项：「把它抽到 src/listen2serve/ 下的一个模块，两边都 import 那一份」）。
   本文件**一行正则都没有**，全部 `from listen2serve.data import hygiene as H` 调用。
   前科：`scripts/validate_tts_contract.py` 的定性词白名单是生成期常量的独立转写副本，
   措辞改了副本没跟上，长期报 208 条假违规（`docs/paper/可质疑点与改进backlog.md`）。
2. **期望值不从被测数据算。** 全部写成绝对值：`== 0` / `== 606` / `== 404` / `== 202`。
   不写「<= 当前命中数」这种跑一次就自毁的闸门（前科：`purify_surface_action.py` 的 GATE
   只存 id→期望裁决、送审文本从盘上现读，第一次 --apply 之后闸门永久判 ❌）。
3. **白名单同源。** T2 冻结面读的是 `data/benchmark/validation/t2_surface_frozen.json`
   —— 那份快照由 `scripts/planS3_freeze_t2.py` 在开工前一次性生成，本测试只读不重算。

**层作用域（2026-09-06 维护者裁定，非本文件自定）**：planS3 §4 改动 3 写「T2 202 条不改，
保留原值供历史复算」，§5 又要求 F1–F7 全部归零，而 F3/F6 按 surface 文本判、T2 行同样命中
（冻结时实测 F3 8 行 / F6 21 行）—— 两条要求在 606 行口径下不可同时满足。裁定：
F3/F6 的断言作用域收到 T1+T3（404 行 == 0），T2 改为逐字节冻结并另立断言 11 守。
F1/F2/F4/F5/F7 仍按 606 行全量 == 0（它们要么只命中 T3，要么判的是 layer-invariant 的
`db_seed`/`identity`，一改三层全清）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from listen2serve.data import hygiene as H
from conftest import N_BASE, N_SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "benchmark" / "scenarios_base.jsonl"

# planS3 §3.3 的实测命中数（改前基线）。写在这里只为报告可读与回归对比，
# **不参与任何断言**（纪律 2：期望值一律是绝对值 0 / 全量）。
_PRE_FIX_HITS = {"F1": 1, "F2": 6, "F3": 24, "F4": 59, "F5": 288, "F6": 85, "F7": 159}
# T2 冻结面在冻结时刻的 F3/F6 命中数（快照里也有，这里写死一份做交叉核对）
_T2_FROZEN_HITS = {"F3": 8, "F6": 21}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    path = ROOT / "data" / "benchmark" / "scenarios.jsonl"
    assert path.exists(), "评测集缺失"
    return [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]


@pytest.fixture(scope="module")
def base_rows() -> list[dict]:
    assert BASE_PATH.exists(), "基础骨架缺失"
    return [json.loads(ln) for ln in BASE_PATH.open(encoding="utf-8") if ln.strip()]


@pytest.fixture(scope="module")
def flags(rows) -> dict[str, list[str]]:
    return H.audit_rows(rows)


def _t1t3(rows) -> list[dict]:
    return [r for r in rows if r.get("leakage_label") in H.REPAIRED_LAYERS]


def _t2(rows) -> list[dict]:
    return [r for r in rows if r.get("leakage_label") == H.FROZEN_LAYER]


class TestSevenDefects:
    """断言 1–7：F1–F7 各自命中数（planS3 §5 表的前七行）。"""

    def test_f1_surface_no_identity_name(self, flags):
        """F1：surface 里不许出现用户本人姓名（改前 1 行 / 1 base）。"""
        assert flags["F1"] == [], f"F1 命中 {flags['F1']}"

    def test_f2_third_party_has_debtor(self, flags):
        """F2：自称第三方时 db_seed 首行不能就是接电话的人（改前 6 行 / 2 base）。"""
        assert flags["F2"] == [], f"F2 命中 {flags['F2']}"

    def test_f3_surface_not_written_style(self, flags, rows):
        """F3：surface 不许书面语/数字堆叠。**作用域 T1+T3**（T2 冻结，见断言 11）。"""
        scope = {r["scenario_id"] for r in _t1t3(rows)}
        hit = [s for s in flags["F3"] if s in scope]
        assert hit == [], f"F3 在 T1/T3 命中 {hit}"

    def test_f4_no_self_contradiction(self, flags):
        """F4：surface 说「不表态」而 description 要「表达/明确」= prompt 自相矛盾（改前 59 行）。"""
        assert flags["F4"] == [], f"F4 命中 {flags['F4']}"

    def test_f5_db_seed_has_truth(self, flags):
        """F5：db_seed 不许是占位符（改前 288 行 / 96 base，业务事实没有真值）。"""
        assert flags["F5"] == [], f"F5 命中 {flags['F5'][:8]}（共 {len(flags['F5'])} 行）"

    def test_f6_form_matches_key_form(self, flags, rows):
        """F6：`key_form=statement` 的 surface 不许是疑问式。**作用域 T1+T3**。"""
        scope = {r["scenario_id"] for r in _t1t3(rows)}
        hit = [s for s in flags["F6"] if s in scope]
        assert hit == [], f"F6 在 T1/T3 命中 {hit}"

    def test_f7_identity_digits_migrated(self, flags):
        """F7：identity 里数字 ≥3 个的要把处境类数字外迁到 user_facts（改前 159 行 / 53 base）。"""
        assert flags["F7"] == [], f"F7 命中 {flags['F7'][:8]}（共 {len(flags['F7'])} 行）"

    def test_no_base_hit_any(self, flags, rows):
        """改前 176/202 base 至少命中一条（planS3 §1）；改后在**修复面**上必须是 0 个 base。

        修复面 = F1/F2/F4/F5/F7 的全部命中 + F3/F6 的 T1/T3 命中。T2 冻结面的 F3/F6
        由断言 11 单独守，不计入这里（否则这条断言永远红，等于没有断言）。
        """
        assert set(flags) == set(H.DEFECT_CODES), "缺陷码集合与 hygiene 不一致"
        t1t3 = {r["scenario_id"] for r in _t1t3(rows)}
        hit: set[str] = set()
        for code in H.DEFECT_CODES:
            sids = flags[code] if code not in H.SURFACE_TEXT_CODES else [
                s for s in flags[code] if s in t1t3
            ]
            hit |= {H.base_id(s) for s in sids}
        assert hit == set(), f"修复面上仍有 {len(hit)} 个 base 命中缺陷：{sorted(hit)[:8]}"
        assert len({H.base_id(r["scenario_id"]) for r in rows}) == N_BASE


class TestNewAssertions:
    """断言 8–10：planS3 §5 表里新增的三条。"""

    def test_8_base_layer_has_no_surface(self, base_rows):
        """断言 8：`scenarios_base.jsonl` 里不许再有 `key_event.surface`（改前 138 条有值）。

        §3.2 的处置：base 层那 138 个 surface 不参与任何实验，是陈旧残留（实测其值与展开层
        T3 逐字相同 134 条、与 T2 相同 4 条 ⇒ 是**过期副本**而不是独立孤儿），删掉并立此断言，
        防止下次又有人往 base 写造成两个真值源。
        """
        holders = H.base_layer_surface_ids(base_rows)
        assert holders == [], f"base 层仍有 {len(holders)} 条带 surface：{holders[:8]}"
        assert len(base_rows) == N_BASE

    def test_9_objective_intent_full_coverage(self, rows):
        """断言 9：606 条全有 `objective` 与 `intent`（改动 2 的新字段，description 保留原值）。"""
        missing = H.objective_intent_missing(rows)
        assert len(rows) == N_SCENARIOS
        assert missing == [], f"{len(missing)} 条缺 objective/intent：{missing[:8]}"

    def test_10_surface_style_all_pass(self, rows):
        """断言 10：所有 T1/T3 的 surface ≤20 字、数字 ≤1、不以「你」开头（404/404）。

        `H.surface_style_violations` 检查的不止这三项，还含书面语黑名单、`key_form` 句式、
        本人姓名、T3 的「不表态」半句与 `_T3_STANCE_MARKERS`（阈值全部来自 planS3 §4 改动 3
        的规范表，实现在 hygiene 里，本测试不抄）。
        """
        scope = _t1t3(rows)
        assert len(scope) == 2 * N_BASE, f"T1/T3 应为 {2 * N_BASE} 行，实为 {len(scope)}"
        bad = H.surface_style_violations(scope)
        assert bad == [], f"{len(bad)} 条 surface 违规：{bad[:5]}"


class TestFrozenAndInvariant:
    """断言 11–14：本次执行中补的护栏（不替代上面十条，只加严）。"""

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "data/benchmark/validation"
             / "t2_surface_frozen.json").exists(),
        reason="发布切片不含 T2 层（多轮主实验只用 T1/T3）⇒ 该冻结基线不在仓内，"
               "显式 skip 而不是静默通过",
    )
    def test_11_t2_surface_byte_frozen(self, rows):
        """断言 11：T2 的 202 条 surface 与开工前快照**逐字节相同**。

        这条与断言 3/6 是一对：F3/F6 的作用域收到 T1/T3，代价是 T2 的 8+21 处缺陷留在盘上，
        那就必须同时保证它**只许原样躺着**——谁去「顺手修一下」T2，历史 run 就再没法用同一份
        刺激复算，而这类改动看起来完全像改进，只有逐字节比对能拦住。
        """
        snap = H.frozen_t2_snapshot()
        assert snap["count"] == N_BASE, "快照条数与 base 数不符"
        drift = H.t2_drift(rows, snap)
        assert drift["id_set_diff"] == [], f"T2 的 id 集合变了：{drift['id_set_diff'][:8]}"
        assert drift["changed"] == [], f"T2 有 {len(drift['changed'])} 条被改动：{drift['changed'][:8]}"
        # 快照里记的冻结面命中数必须与本文件写死的交叉核对值一致（防快照被重生成过）
        assert snap["defect_hit_counts_within_t2_at_freeze"]["F3"] == _T2_FROZEN_HITS["F3"]
        assert snap["defect_hit_counts_within_t2_at_freeze"]["F6"] == _T2_FROZEN_HITS["F6"]

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "data/benchmark/validation"
             / "t2_surface_frozen.json").exists(),
        reason="发布切片不含 T2 层（多轮主实验只用 T1/T3）⇒ 该冻结基线不在仓内，"
               "显式 skip 而不是静默通过",
    )
    def test_11b_t2_frozen_hits_unchanged(self, flags, rows):
        """断言 11b：F3/F6 在 T2 口径下的命中数恒等于冻结基线（8 / 21）。

        与 11 互为冗余校验：11 比文本，这条比判据读数。两者都不为 0 却都通过，
        说明「T2 有缺陷」是**已知且被冻结**的状态，不是新坏掉的。
        """
        t2_ids = {r["scenario_id"] for r in _t2(rows)}
        assert len([s for s in flags["F3"] if s in t2_ids]) == _T2_FROZEN_HITS["F3"]
        assert len([s for s in flags["F6"] if s in t2_ids]) == _T2_FROZEN_HITS["F6"]

    def test_12_base_layer_mirrors_expansion(self, base_rows, rows):
        """断言 12：base 层与展开层除 `surface` 外逐字段相同（含三层之间）。

        不在 planS3 §5 的十条里，是本次执行中补的：实测开工前两层在 db_seed / identity /
        user_facts / key_event(除 surface) / mission / business_context / agent_policy /
        expected_key_behavior / background_card 上 **202/202 全等**。planS3 §2 只写了 base
        「改 db_seed / identity」，但改动 1 要动 `agent_policy.knowledge_base`、改动 2 要动
        `key_event` —— 只改展开层就会造出第二个真值源，正是 §3.2 要消灭的东西。
        """
        drift = H.shared_field_drift(base_rows, rows)
        assert drift == {}, f"base↔展开层字段漂移：{ {k: v[:5] for k, v in drift.items()} }"

    def test_13_kb_family_matches_sub_domain(self, rows):
        """断言 13（F8）：marketing 各 sub_domain 的 kb 家族必须是本域那一套。

        F8 是本次执行发现的计划外缺陷（17 个 v6.5 新增 base 的 kb 挂错业务域模板），
        维护者 2026-09-06 裁定修。家族表同源 import 自 hygiene（纪律 3）。
        """
        bad = H.kb_family_mismatches(rows)
        assert bad == [], f"{len(bad)} 个 base 的 kb 家族与 sub_domain 错配：{bad[:5]}"

    def test_14_product_appears_in_kb(self, rows):
        """断言 14：marketing 的产品名必须同时出现在 db_seed 与 knowledge_base。

        planS3 §4 改动 1 的原话：「同一条产品要同时出现在 db_seed 与 agent_policy.knowledge_base
        （客服才能照口径答）」。db_seed 走 `render_customer_profile` 进【客户信息档案】，
        kb 走 `render_policy_sections` 进【业务口径】，两处缺一处就会出现「档案有、口径无」。
        """
        bad = H.product_kb_missing(rows)
        assert bad == [], f"{len(bad)} 个 marketing base 的产品没进 kb：{bad[:8]}"

    def test_15_background_card_has_no_placeholder(self, rows):
        """断言 15：`background_card` 里不许再有占位商品名（F5 的判据只扫 db_seed，扫不到这里）。

        执行期实测：改完 `db_seed` 之后仍有 **96 个 base** 的卡片写着「产品99 / 示例商品」，
        而 `background_card.consensus` 是经 `run_batch.py:197` 进用户 prompt 的（渲染成「背景：…」）
        ⇒ 只修 db_seed 会让同一条业务事实在一份数据里有两种说法，比占位符更糟。
        判据用 `PLACEHOLDER_IN_PROSE`（比 F5 窄，不含「测试」这个常用词，避免假违规）。
        """
        bad = H.derived_view_placeholders(rows)
        assert bad == [], f"{len(bad)} 个 base 的 background_card 仍有占位名：{bad[:3]}"

    def test_16_background_card_mirrors_db_seed(self, rows):
        """断言 16：`background_card.world_state` 镜像的商品/产品名必须与 `db_seed` 逐字相同。

        守的是「一条业务事实只有一种说法」：`world_state.product(_name)` == `db_seed` 首行名称，
        且 `consensus` / `agent_knows` 里出现的也是同一个名字（不许留着改名前的旧值）。
        """
        bad = H.background_card_db_mismatch(rows)
        assert bad == [], f"{len(bad)} 个 base 的卡片与 db_seed 说法不一致：{bad[:3]}"

    def test_17_t3_surface_has_no_own_state_ban_term(self, rows):
        """断言 17（F13a）：T3 的 `surface` 不许含**本态**的动作/结论禁词。

        词表**同源 import** `data/benchmark/t3_action_ban.json`（planS6 §5 收口的唯一源），
        只取 `ban` 层，作用域只算本态 —— 三点都与 planS6 第一关 `BAN.screen_hits` 的口径一致，
        本断言没有发明任何新阈值。

        为什么这条必须有：`surface` 会被 `_transparency_key_clause_v7` 当成**指令**拼进 T3
        演绎 prompt（「本轮怎么说」），模型逐字 echo。小样实证 `MAR-POS-28-T3` 的 surface
        写「…自己记下了」，产出台词就是「你打这通电话的意思我记下了。」，而 §6.4 的 G3
        只 import `_T3_STANCE_MARKERS`（planS1 的态度词表，**不含**「记下了」）⇒ 全过八道闸、
        被判 `frozen_candidate`。改前本态命中 7 条（全是 cooperative 的「记下了」，
        即 planS6 changelog §9 第 9 条登记并移交给 planS3 的那 7 条），现已返修为 0。
        """
        bad = H.t3_ban_hits(rows, scope="own")
        assert bad == [], f"{len(bad)} 条 T3 surface 含本态禁词：{bad[:5]}"

    def test_18_cross_state_and_prompt_field_ban_hits_not_worse(self, rows):
        """断言 18（F13b/F13c 的**棘轮**，不是归零门）：跨态与 objective/intent 的本态禁词命中数不许变多。

        这三处**都还没修**，因为修它们超出 planS3 的授权，属维护者裁定项（报告 §6.10 第 5/6 项）：

        - `surface` 的**跨态**命中 15 条（displeased 的行里出现 cooperative 的「记下了」11 条、
          doubtful 的「依据」5 条，`COL-ALL-18-T3` 一条占两词）。planS6 changelog §6 第 10 条
          把这个盲区登记为「三关全盲」，并明写「要不要补一道跨态检查 = **维护者裁定**，
          planS6 无权扩门的判据面」⇒ 我也不该单方面判定它。
        - `objective` 本态命中 23 条（cooperative 的「还款方式」12、doubtful 的「核实」10、
          「依据」/「反馈时间」/「工单」各 1），`intent` 本态命中 18 条（urgent 的「立刻」6/
          「马上」6/「尽快」2、doubtful 的「核实」4/「依据」4）。这两个字段是改动 2 新写的，
          同样进 T3 prompt（【关键事件】/这件事你心里怎么想），所以同样是泄漏源；
          但修它们要动 606 行，且会与 `expected_key_behavior` 的锚点措辞（「追问细节/索要依据」）打架。

        所以这里只上**棘轮**：数字变小（有人修了）不报错，变大（有人又写进禁词）就拦住。
        期望值是 2026-09-06 17:2x 的实测快照（裁定 6 净化完 urgent 那 12 条之后），
        **不是**从被测数据现算的（plan §5 纪律 2）。
        """
        cross = H.t3_ban_hits(rows, scope="cross")
        obj = H.field_ban_hits(rows, "objective")
        inte = H.field_ban_hits(rows, "intent")
        assert len(cross) <= 15, f"T3 surface 跨态禁词命中变多了（{len(cross)} > 15）：{cross[:5]}"
        assert len(obj) <= 23, f"T3 objective 本态禁词命中变多了（{len(obj)} > 23）：{obj[:5]}"
        assert len(inte) <= 6, f"T3 intent 本态禁词命中变多了（{len(inte)} > 6）：{inte[:5]}"

    def test_19_t3_positive_example_is_clean(self):
        """断言 19（F14，裁定 7）：注入 T3 演绎 prompt 的**关键轮正例**自己必须干净。

        为什么 CI 要管一句 prompt 里的示范台词：planS1 实测「例词清单会被演绎模型当成推荐模板」
        （填充词起头率 24%→92%），而正例只在关键轮注入 ⇒ **正例的形状就是关键轮台词的形状**。
        改前那条陈述句正例「那就按你说的这个时间来办。」是 cooperative 的接受结论，
        一条占 G7 的 3 个特征串，在 **202/202 条 T3 场景**上撞闸（等于 prompt 在教模型说一句
        闸门必定判死的话）；canonical 小样 v2 的 G7 命中 9 次里 **4 次是它的形状**。
        而 planS1 既有的 `test_planN_pragmatics.py::test_positive_example_by_form_and_level`
        只查 `_T3_STANCE_MARKERS`（那 20 个态度词里没有「那就/就按/按你说的」）⇒ 一路全绿。

        判据四件全部同源 import（`_T3_STANCE_MARKERS` / `t3_action_ban` 的 ban 层**全态** /
        G7+G8 特征串 / 单句结构），本测试不写死任何词。ban 用**全态**而不是数据侧的「只算本态」：
        正例是五个态共用的一条，任何一个态的禁词都不该出现。
        """
        bad = H.t3_positive_example_violations()
        assert bad == [], f"T3 关键轮正例自己违规（会教模型说禁句）：{bad}"

    def test_20_urgent_intent_has_no_own_state_ban_term(self, rows):
        """断言 20（F13c，裁定 6）：**urgent** 态的 T3 `intent` 不许含本态禁词。

        裁定 6 只授权净化 urgent 那一批（改前 12 条 / 14 个词命中：「立刻」6、「马上」6、「尽快」2），
        理由是这批可以**同义替换而语义无损** —— 紧迫感本来就由「时间被卡住 / 耐心快耗尽 /
        正赶着上车」这些处境描述承载，删掉催促副词不丢信息。
        doubtful 的「核实」10 + 「依据」4 **刻意不在本断言里**：`expected_key_behavior` 的锚点
        自己就写着「索要依据」，那是裁判侧的地面真值，改了 `intent` 会与锚点分裂 ⇒ 留待裁定。

        ⚠️ 两处改法上的讲究，逐条写在 `edits/intent_urgent_rework.jsonl` 的 `note` 里：
        ① 「马上要上车」是**字面场景**（人真的要上火车）而不是催促副词，属禁词表的假阳性 ——
           处置是改成同义的「眼看着要上车」，**不是**给判据开例外口子（开口子就等于放宽阈值）；
        ② 写成「可执行**的**安排」而不是「可执行安排」，避免拼出 cooperative 的 ban 词「安排上」。
        """
        bad = H.field_ban_hits(rows, "intent", states=("urgent",))
        assert bad == [], f"{len(bad)} 条 urgent 的 T3 intent 含本态禁词：{bad[:5]}"

    def test_21_trigger_hint_ban_hits_are_registered_not_judged(self, rows):
        """断言 21（F13d 的**棘轮**）：`trigger_hint` 的本态禁词命中数不许变多。

        这是收尾复核时扫出来的**最大一块**：`trigger_hint` 同样进 T3 演绎 prompt
        （渲染成「触发时机：…」），本态命中 **28 条**，比 `objective` 的 23 条还多
        （「还款方式」8 / 「依据」8 / 「凭证」5 / 「不方便」3 / 「工单」3 / 「讲清楚」2 /
        「核实」2 / 其余各 1）。

        **为什么不修**：① 它不在 planS3 的改动面里 —— §2.2 的 0 改动清单里就有
        `key_event.trigger_hint`，planS3 §4 的六项改动没有一项授权碰它；
        ② 它是**三层共享的骨架字段**（`test_smoke.py::test_variants_share_skeleton` 要求
        三层 `key_event` 除 surface 逐字一致），改它会同时改掉**冻结的 T2 层**那一行，
        而 T2 冻结是裁定 1.2 明令不许动的面。
        ⇒ 只登记 + 上棘轮，归零门留给维护者裁定（报告 §6.9 / §10.3）。
        """
        hits = H.trigger_hint_ban_hits(rows)
        assert len(hits) <= 28, f"trigger_hint 本态禁词命中变多了（{len(hits)} > 28）：{hits[:5]}"

    def test_22_surfaces_are_unique_across_bases(self, rows):
        """断言 22（V7 的数据侧防线）：**T3** 的 `surface` 不许在两个 base 之间逐字相同；T1 只上棘轮。

        V7 的原话是「无两条完全相同的 `text`」，查的是**产出的台词**。但台词是 surface 的函数 ——
        两个 base 的 surface 逐字相同、态也相同时，模型可能产出逐字相同的台词。
        全量 290 通实测到了：`MAR-ALL-22-T3` 与 `MAR-POS-28-T3` 都产出
        「你们这通电话的来意我记下了。」，根因是改动 3 给它俩写了同一句 surface。
        ⚠️ 而那时台词**已经冻结**，按红线 5 不许原地改 ⇒ 这个检查必须放在**数据侧、落盘前**，
        不能等到台词生成后才发现。

        **为什么 T3 是归零门、T1 只是棘轮（≤16 组）** —— 这不是放宽，是照全量 290 通的实测分层的：

        | 层 | surface 重复组数（改前） | 实际产出的重复台词 | 台词字数中位 |
        |---|---|---|---|
        | T3 | 8 组 / 19 条 | **1 组**（就是上面那对） | 17 字 |
        | T1 | **16 组** | **0 组** | 43 字 |

        机制很清楚：T3 的台词短且中性，surface 几乎**决定**了台词；T1 的台词长、
        要同时装「态度词 + 业务动作 + 情绪词」，模型会带进本 base 自己的金额/产品/处境，
        所以 surface 相同也不会让台词相同（16 组重复 surface 产出 0 组重复台词，是实测不是推测）。
        ⇒ T3 归零（改前 8 组已按「每组保留一条、其余取材本 base 自己的 `db_seed`/`sub_domain`/
        `user_facts` 改写」修完，`edits/t3_dedup_rework.jsonl` 11 条，逐条带理由）；
        T1 上棘轮（不许变多），并把这 16 组登记为待裁定项 —— 真要改也是 16 组 × 逐条取材，
        在台词已冻结、V7 实测为 0 的当下不值得再动一遍数据（报告 §10.3）。
        """
        dup3 = H.duplicate_surfaces(rows, layers=("T3",))
        assert not dup3, f"T3 有 {len(dup3)} 组跨 base 重复的 surface（实测会让 V7 的台词重复）：{list(dup3.items())[:3]}"
        dup1 = H.duplicate_surfaces(rows, layers=("T1",))
        assert len(dup1) <= 16, f"T1 跨 base 重复的 surface 变多了（{len(dup1)} > 16 组）：{list(dup1.items())[:3]}"


class TestBaselineDocumented:
    """改前基线数字留档（不参与判定，只保证报告里的「改前」有据可查）。"""

    def test_pre_fix_hits_are_documented(self):
        assert sum(_PRE_FIX_HITS.values()) == 622
        assert _PRE_FIX_HITS == {
            "F1": 1, "F2": 6, "F3": 24, "F4": 59, "F5": 288, "F6": 85, "F7": 159,
        }
