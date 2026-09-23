"""内部方案：演绎侧数据卫生规则的**唯一实现点**。

为什么要抽这个模块：F1–F7 这七条判据原先只活在
`scripts/final_plan_sections2.py::_audit()` 里 —— 那是审核页生成器，CI 想复用就只能
`sys.path` 摸进 `scripts/` 或者在测试里抄一份正则。本仓有前科：
`scripts/validate_tts_contract.py` 的定性词白名单是从生成期常量独立转写的一份副本，
生成期措辞改了而副本没跟上，长期报 208 条假违规，把真问题一起淹没
（`内部文档`、内部方案 都点了这条）。
所以规则只写这一份，审核页、CI、落盘前的自检三方都 import 它。

阈值一律取自 内部方案 原文（改动 3 的改写规范表、 的断言表），**不从被测数据反推**：
断言写「命中数 == 0」「== 全量」这种绝对值，不写「<= 当前命中数」（实现纪律 2）。

层的作用域是 2026-09-06 决策记录的结果，不是本模块自己定的：
内部方案 写「T2 202 条不改，保留原值供历史复算」， 又要求 F1–F7 全部归零，
而 F3/F6 是按 surface 文本判的、T2 行同样命中（冻结时实测 F3 8 行 / F6 21 行）——
两条要求在 606 行口径下不可同时满足。裁定：F3/F6 的断言作用域收到 T1+T3（`REPAIRED_LAYERS`），
T2 改为逐字节冻结（快照 `data/benchmark/validation/t2_surface_frozen.json`，
由 `内部脚本` 生成，CI 只读不算）。
F1/F2/F4/F5/F7 不受影响：它们要么只命中 T3，要么判的是 `db_seed`/`identity`
这类 layer-invariant 字段，一改三层全清，仍按 606 行全量口径断言 == 0。
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS_PATH = PROJECT_ROOT / "data" / "benchmark" / "scenarios.jsonl"
SCENARIOS_BASE_PATH = PROJECT_ROOT / "data" / "benchmark" / "scenarios_base.jsonl"
T2_FROZEN_PATH = PROJECT_ROOT / "data" / "benchmark" / "validation" / "t2_surface_frozen.json"

# ---- 内部方案「改写规范」表的阈值（逐行对应，改这里等于改验收口径）----
MAX_SURFACE_CHARS = 20 # 「长度 ≤20 字」
MAX_SURFACE_DIGITS = 1 # 「数字 ≤1 个」
SUBJECT_PREFIX = "你" # 「人称：统一为无主语祈使句」
# 「文体：口语，不许『且曾／及是否／以及是否／并已／系属／及具体／及可』」
# 前五个与 F3 判据同源，后两个（及具体／及可）是改动 3 的文体行额外列的。
WRITTEN_MARKERS: tuple[str, ...] = (
    "且曾", "及是否", "以及是否", "并已", "系属", "及具体", "及可",
)
# 「句式：statement 不许含『是否／能不能／是不是／怎么办／多少／什么时候／吗／呢』」
QUESTION_MARKERS: tuple[str, ...] = (
    "是否", "能不能", "是不是", "怎么办", "多少", "什么时候", "吗", "呢",
)

# 改动 3 只重写 T1/T3；T2 冻结（见模块 docstring 的裁定记录）
REPAIRED_LAYERS: tuple[str, ...] = ("T1", "T3")
FROZEN_LAYER = "T2"
# 只有这两条判据是「按 surface 文本」判的，因此受 T2 冻结影响、断言作用域收到修复面
SURFACE_TEXT_CODES: tuple[str, ...] = ("F3", "F6")

DEFECT_CODES: tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5", "F6", "F7")

# F5 的占位符判据（与 _audit 原实现逐字相同）
_PLACEHOLDER_RE = re.compile(r"示例|产品\d+|测试|占位")
_QUESTION_RE = re.compile("|".join(QUESTION_MARKERS))
_NAME_RE = re.compile(r"^([\u4e00-\u9fa5]{2,4})[，,]")
_THIRD_PARTY_RE = re.compile(r"不是债务人本人|不是本人|老乡|邻居|家属|亲属")
_STANCE_MUTE_RE = re.compile(r"不表态|不评价|不表明")
_DESC_ACTION_RE = re.compile(r"表达|明确|要求|表明|提出")
_WRITTEN_F3_RE = re.compile(r"且曾|及是否|以及是否|并已|系属")


# ---------------------------------------------------------------- 读取与取值


def load_rows(path: Path | str = SCENARIOS_PATH) -> list[dict]:
    """读 jsonl（展开层 606 行 / base 层 202 行都用它）。"""
    p = Path(path)
    return [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]


def load_scenarios() -> list[dict]:
    return load_rows(SCENARIOS_PATH)


def load_base() -> list[dict]:
    return load_rows(SCENARIOS_BASE_PATH)


def base_id(scenario_id: str) -> str:
    """`COL-ALL-01-T3` → `COL-ALL-01`（base 层的 id 本来就没有层后缀）。"""
    return re.sub(r"-T[123]$", "", scenario_id or "")


def user_script(row: dict) -> dict:
    return row.get("user_script") or {}


def key_event(row: dict) -> dict:
    """⚠️ 路径是 `user_script.key_event`，不在顶层（历史坑位登记：
    写成 `row.get("key_event")` 恒为 None，校验闸门会静默空转、assert 全过）。"""
    return user_script(row).get("key_event") or {}


def surface_of(row: dict) -> str:
    return key_event(row).get("surface") or ""


def identity_of(row: dict) -> str:
    return user_script(row).get("identity") or ""


def identity_name(identity: str) -> str | None:
    """`identity` 开头的 2–4 字中文姓名（F1/G5 的判据来源，两处必须同源）。"""
    m = _NAME_RE.match(identity or "")
    return m.group(1) if m else None


# 「句式：statement 不许含『是否／能不能／…』」的**语用补强**（内部方案 小样实测出来的）：
# F6/G2 判的是字面疑问词，但 surface 写成「问X」这种**疑问框架**时，演绎模型照样会产出
# 带问号/带「吗」的台词 —— 小样 20 通里 COL-ALL-04-T1 四次尝试全部撞 G2（47/48/51/53 字，
# 句末问号 + 「行吗」），而它的 surface 是「明说愿意配合，问缴费渠道和到账时间」，
# 字面一个疑问词都没有、F6 判它干净。所以 statement 版的 surface 连「问」这个框架词也不许用
# （「问题」是名词，不算，先剔掉再判）。
INTERROGATIVE_FRAME = ("问",)
_FRAME_EXEMPT = ("问题",)


def _has_interrogative_frame(text: str) -> bool:
    t = text or ""
    for w in _FRAME_EXEMPT:
        t = t.replace(w, "")
    return any(w in t for w in INTERROGATIVE_FRAME)


def digits_in(text: str) -> int:
    """文本里数字串的个数（`ORD0001` 记 1 个，`3600元逾期10天` 记 2 个）。"""
    return len(re.findall(r"\d+", text or ""))


def has_placeholder(value: Any) -> bool:
    """F5 的判据：`db_seed` 里是否还有「示例/产品N/测试/占位」这类没有真值的占位符。"""
    return bool(_PLACEHOLDER_RE.search(json.dumps(value, ensure_ascii=False)))


def key_form(scenario_id: str) -> str:
    """关键轮句式。**运行时计算属性，不是数据字段**：
    定义在 `runtime/user_simulator.py::key_form_for`，按 base 号奇偶配平。
    这里只做转发，绝不复制它的实现，否则层间配平口径会漂移。"""
    from listen2serve.runtime.user_simulator import key_form_for

    return key_form_for(scenario_id)


def t3_stance_markers() -> tuple[str, ...]:
    """T3 忌讳的结论式表述。**import 不要抄**（内部方案 G3 明写）。"""
    from listen2serve.runtime.user_simulator import _T3_STANCE_MARKERS

    return tuple(_T3_STANCE_MARKERS)


def t3_ban_terms(state: str | None = None) -> tuple[str, ...]:
    """T3 的逐态「动作语用」禁词。**唯一源 = `data/benchmark/t3_action_ban.json`，import 不要抄。**

    `state=None` 返回全态合并去重（跨态检查用），否则只返回该态的 `ban` 层。
    **只取 `ban` 层**，理由写在那份 JSON 的 tier 语义里，用错就等于放宽闸门（内部方案 纪律 3）：
    `screen_only` 误报率高到不能参与判定（冲突 C5：「改/换/短/方便」是单字）；
    `rejected` 是逐条判后剔除的（C1：doubtful 的「确认」正是 T3 规定句式的动词，拿它判会把合格台词判死）；
    `candidate` 是 内部方案 提的新增项，**未经决策记录，默认不生效**。

    ⚠ 与 `t3_stance_markers()` 是两张不同的表，且**互不包含**：stance markers 住在
    `runtime/user_simulator.py`（内部方案，20 个态度词，会进 T3 演绎 prompt 当禁用清单），
    ban 词表住在 `data/benchmark/t3_action_ban.json`（内部方案，37 个动作/结论词，
    按 `ban_clause()` 的 docstring **刻意不进演绎 prompt**，因为 内部方案 实测「例词清单被
    演绎模型当成推荐模板」，填充词起头率 24%→92%）。「记下了」在 ban 表里、不在 stance 表里
    ⇒ 只 import stance 表的闸门（的 G3）拦不住它，实测见 内部方案。
    """
    from listen2serve.data import t3_action_ban as BAN

    if state is not None:
        return tuple(BAN.terms(state, ("ban",)))
    seen: list[str] = []
    for s in BAN.states():
        for w in BAN.terms(s, ("ban",)):
            if w not in seen:
                seen.append(w)
    return tuple(seen)


def t3_ban_hits(rows: Iterable[dict], scope: str = "own") -> list[tuple[str, str, list[str]]]:
    """T3 层 `surface` 命中禁词的清单，返回 `[(scenario_id, state, 命中词…)]`。

    `scope`：
      `own` 只算**本态** ban —— 内部方案 第一关（`BAN.screen_hits`）的口径，也是 CI 断言 17 的判据；
      `cross` 只算**他态** ban —— 内部方案 第 10 条点名的结构性盲区（三关全盲），
              它明确写了「要不要补一道跨态检查 = **决策记录**」，故本函数只提供事实、CI 不判定；
      `any` 两者都算。

    为什么 surface 也要过这张表：surface 不是给人看的注释，它经
    `_transparency_key_clause_v7` 被当成**指令**拼进 T3 演绎 prompt（「本轮怎么说」），
    模型会逐字 echo。小样实证：`MAR-POS-28-T3` 的 surface 写「…自己记下了」，
    生成的台词就是「你打这通电话的意思我记下了。」⇒ surface 里的禁词是**泄漏源**，
    只在产出侧拦等于每次靠回炉碰运气。
    """
    if scope not in ("own", "cross", "any"):
        raise ValueError(f"未知 scope {scope}；合法值 own/cross/any")
    from listen2serve.data import t3_action_ban as BAN

    out: list[tuple[str, str, list[str]]] = []
    for r in rows:
        if r.get("leakage_label") != "T3":
            continue
        text = surface_of(r) or ""
        state = str(key_event(r).get("state") or "")
        if state not in BAN.states():
            continue
        hit = [w for w in t3_ban_terms() if w in text]
        if scope == "own":
            hit = [w for w in hit if w in t3_ban_terms(state)]
        elif scope == "cross":
            hit = [w for w in hit if w not in t3_ban_terms(state)]
        if hit:
            out.append((r["scenario_id"], state, hit))
    return out


def field_ban_hits(rows: Iterable[dict], field: str,
                   states: tuple[str, ...] | None = None) -> list[tuple[str, str, list[str]]]:
    """T3 层 `objective` / `intent` 命中**本态** ban 词的清单（口径同 `t3_ban_hits(scope="own")`）。

    `states` 非空时只看这几个态 —— **裁定 6**（2026-09-06 17:1x）只授权净化 **urgent** 那一批
    （「立刻」6 / 「马上」6 / 「尽快」2，同义替换、不碰裁判锚点）；doubtful 的「核实」/「依据」
    暂不动，因为 `expected_key_behavior` 自己就写着「索要依据」，改了会与裁判侧锚点分裂。
    有了这个参数，执行器与 CI 才能**只对已裁定的那一批上归零门**，
    而不是拿一个还没裁定的口径去判死另一半。

    存在理由：这两个字段是 内部方案 新写的，同样会进 T3 演绎 prompt
    （objective → 【关键事件】、intent → 这件事你心里怎么想），所以同样是泄漏源。
    ⚠ 还有一个**更大且不属于 内部方案 改动面**的同类字段：`trigger_hint`（同样进 prompt，
    实测本态命中 **28 条**，比 objective 的 23 还多）。它是三层共享的骨架字段
    （冻结的 T2 层也带着它），且 内部方案 从未授权改它 ⇒ 登记为 **F13d** 待裁定，
    本函数不替它做任何判定。
    """
    if field not in ("objective", "intent"):
        raise ValueError(f"未知字段 {field}；合法值 objective/intent")
    from listen2serve.data import t3_action_ban as BAN

    out: list[tuple[str, str, list[str]]] = []
    for r in rows:
        if r.get("leakage_label") != "T3":
            continue
        ke = key_event(r)
        text = str(ke.get(field) or "")
        state = str(ke.get("state") or "")
        if state not in BAN.states():
            continue
        if states is not None and state not in states:
            continue
        hit = [w for w in t3_ban_terms(state) if w in text]
        if hit:
            out.append((r["scenario_id"], state, hit))
    return out


def trigger_hint_ban_hits(rows: Iterable[dict]) -> list[tuple[str, str, list[str]]]:
    """T3 层 `trigger_hint` 命中**本态** ban 词的清单（F13d，**只提供事实，谁都不许拿它判定**）。

    为什么单列一个函数而不是并进 `field_ban_hits`：这个字段**不在 内部方案 的改动面里**
    （的 0 改动清单里就有 `key_event.trigger_hint`），而且它是三层共享的骨架字段 ——
    改它会同时改掉**冻结的 T2 层**那一行，那是裁定 1.2 明令不许动的面。
    所以它只能登记、不能由任何执行窗口顺手修。实测本态命中 28 条
    （「还款方式」8 / 「依据」8 / 「凭证」5 / 「不方便」3 / 「工单」3 / 「讲清楚」2 /
    「核实」2 / 「改时间」1 / 「没听明白」1 / 「马上」1 / 「尽快」1），
    比 `objective` 的 23 条还多，且**它确实进 T3 演绎 prompt**（渲染成「触发时机：…」）。
    """
    from listen2serve.data import t3_action_ban as BAN

    out: list[tuple[str, str, list[str]]] = []
    for r in rows:
        if r.get("leakage_label") != "T3":
            continue
        ke = key_event(r)
        text = str(ke.get("trigger_hint") or "")
        state = str(ke.get("state") or "")
        if state not in BAN.states():
            continue
        hit = [w for w in t3_ban_terms(state) if w in text]
        if hit:
            out.append((r["scenario_id"], state, hit))
    return out


# ------------------------------------------------- T3 句式级判据（裁定 3 新增，内部方案）
# 为什么住在 hygiene 而不是留在 `内部台词生成脚本`：内部方案 要求
# 「规则同源，不许在执行侧和 CI 各抄一份」。这两组特征串原本只在执行器里，于是**没有任何测试
# 能拿它们去校验演绎 prompt 自己**——F14 就是这么漏的：内部方案 的 T3 陈述句正例
# 「那就按你说的这个时间来办。」一条占 G7 的 3 个特征串，在 202/202 条 T3 场景上撞闸，
# 而 `内部测试用例` 只查
# `_T3_STANCE_MARKERS`（那 20 个态度词里没有「那就/就按/按你说的」）⇒ 全绿。
# 搬进来之后，执行器 import 它、CI 也 import 它，正例才有人守。

T3_ACCEPTANCE_PATTERNS = (
    "那就", "就按", "按这个", "按你们", "照着", "我先", "我就", "听你的", "按你说的",
    "没问题", "可以的", "行，", "行。", "好，", "好。", "成，", "成。",
)
"""G7：承接式结论 / 顺从承诺句式。词表拦不住的那一类 —— 小样 v1 里这几条全过了 G1–G6
却把结论说出口了：「月初钱一到我就转」「行，那就按这个安排处理了」「有什么处理办法我照着来」。
⚠️ 光杆「行，」「好，」「成，」是**故意**收进来的：内部方案 的 T3 prompt 禁用清单写的是
「行吧/算了/好吧/那行」，模型钻空子用「行，」开头（小样 `HOT-NEG-48-T3` 四次尝试中三次），
只能在这里事后拦。已移交 内部方案 按 `_T3_STANCE_MARKERS` 的规矩（新增项追加到尾部）补进 prompt 侧。"""

T3_COMMAND_PATTERNS = (
    "说清楚", "讲清楚", "说明白", "你给我", "直接告诉我", "赶紧", "快点",
    "别绕", "明确一下", "给我个准话", "立刻", "马上给",
)
"""G8：命令 / 催办口气。T3 的 prompt 明写「也不要用命令或催办的口气（『你先说清楚』
『麻烦明确一下』这类）」，而原六道闸没有一条查它 —— 小样 `MAR-ALL-23-T3`
「你们到底是怎么拿到我号码的，说清楚。」就这么过了。"""

T3_GATE_PATTERN_LAYERS = ("T3",)
"""G7/G8 只作用 T3：T1 的设计就是「态度在文本里」，命令口气与承接结论在 T1 是合法表达
（小样 T1 例句「你赶紧把我号码删了，别再打过来了啊」正是 displeased 该有的样子）。"""


def duplicate_surfaces(rows: Iterable[dict],
                       layers: tuple[str, ...] = REPAIRED_LAYERS) -> dict[str, list[str]]:
    """跨 base **逐字相同**的 surface，返回 `{surface: [base_scenario_id…]}`（只留 ≥2 条的组）。

    为什么必须有这条（V7 的教训）：plan 的 V7 要求「无两条完全相同的 `text`」，而 surface 是
    演绎 prompt 里「本轮怎么说」的**指令**——两个 base 的 surface 逐字相同、态也相同时，
    模型很可能产出逐字相同的台词。这不是推测：全量 290 通实测 `MAR-ALL-22-T3` 与
    `MAR-POS-28-T3` 产出了完全相同的「你们这通电话的来意我记下了。」，
    根因就是改动 3 给它俩写了同一句「说这通电话的来意自己记下了」。
    ⇒ **V7 必须在数据侧防**，不能只在产出台词后才发现（那时台词已经冻结，红线 5 不许改）。

    改前实测：T3 层有 **8 组 / 19 条**重复。修法是每组保留一条、其余改写成**本 base 专属**
    的说法（取材该 base 自己的 `db_seed` / `sub_domain` / `user_facts`），
    逐条理由在 `（未发布）`。
    """
    seen: dict[str, list[str]] = {}
    for r in rows:
        if r.get("leakage_label") not in layers:
            continue
        s = surface_of(r) or ""
        if not s:
            continue
        seen.setdefault(s, []).append(str(r.get("base_scenario_id") or r["scenario_id"]))
    return {k: v for k, v in seen.items() if len(v) > 1}


def t3_line_pattern_hits(text: str) -> dict[str, list[str]]:
    """返回一句话命中的 G7/G8 特征串（`{"G7": [...], "G8": [...]}`，空列表 = 未命中）。

    执行器的 `gates()` 与 CI 的断言 19 共用它，避免两侧各写一份 `any(p in text …)`。
    """
    return {
        "G7": [p for p in T3_ACCEPTANCE_PATTERNS if p in (text or "")],
        "G8": [p for p in T3_COMMAND_PATTERNS if p in (text or "")],
    }


def t3_positive_example_violations() -> list[tuple[str, str, list[str]]]:
    """校验注入 T3 演绎 prompt 的**示范句**自己是否干净，返回 `[(来源, 台词, 问题…)]`。

    覆盖两处示范句（它们都是「模型会照抄形状」的文本，性质相同）：
      A. `_key_turn_positive_example(form, "T3")` —— 关键轮正例，问句/陈述句各 1 条；
      B. `_end_call_examples("T3")` —— `end_call=true` 那轮的收尾示范句（多条）。

    为什么 CI 要管 prompt 里的示范台词（F14 / F15b 的教训）：内部方案 实测
    「例词清单会被演绎模型当成推荐模板」（填充词起头率 24%→92%）⇒ **示范句的形状就是产出台词的形状**。
    改前 A 的陈述句版是「那就按你说的这个时间来办。」（cooperative 的接受结论，一条占 G7 的
    3 个特征串，在 **202/202** 条 T3 场景上撞闸）；改前 B 是写死的三条、对三档一律注入，
    其中「那就先这样吧」「行，我知道了，谢谢」**两条都是 G7 形状** —— canonical 小样 v3
    仅剩的 2 条手改队列里，`HOT-NEG-48-T3`（intent 正是「想收尾」）四次尝试全以「行，」开头。
    而 内部方案 既有的 `内部测试用例`
    只查 `_T3_STANCE_MARKERS`（那 20 个态度词里没有「那就/就按/行，」）⇒ 一路全绿。

    查四件事，全部同源 import，不另立词表：
      1. `_T3_STANCE_MARKERS`（内部方案 的态度词表）；
      2. `t3_action_ban` 的 ban 层（内部方案 的动作词表，**全态**：示范句是跨态共用的，
         所以任何一个态的 ban 词都不该出现，比数据侧的「只算本态」更严）；
      3. G7/G8 特征串（裁定 3）；
      4. 结构：单句（无句中句末标点）∧ ≤1 个逗号 ∧ 不以「你」开头。
    """
    import re

    from listen2serve.runtime.user_simulator import (
        _end_call_examples,
        _key_turn_positive_example,
    )

    candidates: list[tuple[str, str]] = []
    for form in ("question", "statement"):
        raw = _key_turn_positive_example(form, "T3")
        m = re.search(r"“(.+?)”", raw)
        candidates.append((f"关键轮正例/{form}", m.group(1) if m else raw))
    # 收尾示范句是一串「“…”“…”」，逐条拆开单独判（合起来判会把它当成一句超长多逗号的话）
    for i, m in enumerate(re.finditer(r"“(.+?)”", _end_call_examples("T3")), 1):
        candidates.append((f"收尾示范句{i}", m.group(1)))

    out: list[tuple[str, str, list[str]]] = []
    for src, line in candidates:
        problems: list[str] = []
        for w in t3_stance_markers():
            if w in line:
                problems.append(f"stance:{w}")
        for w in t3_ban_terms():
            if w in line:
                problems.append(f"ban:{w}")
        for gate, hits in t3_line_pattern_hits(line).items():
            problems += [f"{gate}:{h}" for h in hits]
        body = line.rstrip("。！？；?!;")
        if any(ch in body for ch in "。！？；?!;"):
            problems.append("结构:非单句")
        if line.count("，") + line.count(",") > 1:
            problems.append("结构:逗号>1")
        if line.startswith(SUBJECT_PREFIX):
            problems.append(f"结构:以「{SUBJECT_PREFIX}」开头")
        if problems:
            out.append((src, line, problems))
    return out


# ---------------------------------------------------------------- F1–F7 体检


def audit_rows(rows: Iterable[dict]) -> "OrderedDict[str, list[str]]":
    """按 内部方案 的七条判据扫一遍，返回 `缺陷码 → scenario_id 列表`。

    判据与原 `scripts/final_plan_sections2.py::_audit()` 逐字等价（该函数已改为转发到这里），
    唯一差别是返回的 key 顺序固定为 F1…F7，便于两侧输出对齐。
    """
    flags: "OrderedDict[str, list[str]]" = OrderedDict((c, []) for c in DEFECT_CODES)

    for r in rows:
        ke = key_event(r)
        sid = r["scenario_id"]
        sf = ke.get("surface") or ""
        desc = ke.get("description") or ""
        idt = identity_of(r)
        nm = identity_name(idt)
        kf = key_form(sid)

        # F1 surface 里引用了用户本人姓名
        if nm and nm in sf:
            flags["F1"].append(sid)
        # F2 自称第三方（不是本人/老乡/邻居/家属），但 db_seed 把 TA 登记为债务人
        cust = ((r.get("db_seed") or {}).get("customers") or [[None]])[0]
        if cust and cust[0] == nm and _THIRD_PARTY_RE.search(idt):
            flags["F2"].append(sid)
        # F3 surface 是书面语（数字罗列 / 书面连接词）
        if digits_in(sf) >= 2 or _WRITTEN_F3_RE.search(sf):
            flags["F3"].append(sid)
        # F4 surface 说「不表态」而 description 要求「表达/明确…」→ prompt 自相矛盾
        if _STANCE_MUTE_RE.search(sf) and _DESC_ACTION_RE.search(desc):
            flags["F4"].append(sid)
        # F5 db_seed 是占位符（业务事实无真值）
        if _PLACEHOLDER_RE.search(json.dumps(r.get("db_seed"), ensure_ascii=False)):
            flags["F5"].append(sid)
        # F6 该写陈述句却给了疑问式 surface
        if kf == "statement" and _QUESTION_RE.search(sf):
            flags["F6"].append(sid)
        # F7 identity 里数字 ≥3 个（常驻背景被当台词素材念出来）
        if digits_in(idt) >= 3:
            flags["F7"].append(sid)

    return flags


def audit_files(scenarios_path: Path | str = SCENARIOS_PATH) -> tuple["OrderedDict[str, list[str]]", list[dict], int]:
    """读盘版体检，返回 `(flags, rows, base 数)` —— 与原 `_audit()` 的契约一致。"""
    rows = load_rows(scenarios_path)
    return audit_rows(rows), rows, len({r["base_scenario_id"] for r in rows})


def hits_by_layer(flags: dict[str, list[str]], rows: list[dict]) -> dict[str, dict[str, int]]:
    """每个缺陷码在 T1/T2/T3 各命中多少行（报告 的改前/改后对照要用）。"""
    layer = {r["scenario_id"]: r.get("leakage_label") for r in rows}
    out: dict[str, dict[str, int]] = {}
    for code in DEFECT_CODES:
        c = {"T1": 0, "T2": 0, "T3": 0}
        for sid in flags.get(code, []):
            c[layer.get(sid, "?")] = c.get(layer.get(sid, "?"), 0) + 1
        out[code] = c
    return out


def bases_hit(flags: dict[str, list[str]], codes: Iterable[str] = DEFECT_CODES) -> set[str]:
    """命中给定缺陷码集合中任一条的 base 集合（表末那个 176/202）。"""
    out: set[str] = set()
    for code in codes:
        out |= {base_id(s) for s in flags.get(code, [])}
    return out


# ---------------------------------------------------------------- 断言 8/9/10


def base_layer_surface_ids(base_rows: Iterable[dict]) -> list[str]:
    """断言 8：base 层不得再有 `key_event.surface`（的孤儿字段 / 陈旧副本）。"""
    return [r["scenario_id"] for r in base_rows if surface_of(r)]


def objective_intent_missing(rows: Iterable[dict]) -> list[str]:
    """断言 9：每条展开记录都要有 `objective` 与 `intent`（改动 2 的新字段）。"""
    return [
        r["scenario_id"]
        for r in rows
        if not (key_event(r).get("objective") and key_event(r).get("intent"))
    ]


def surface_style_violations(
    rows: Iterable[dict], layers: Iterable[str] = REPAIRED_LAYERS
) -> list[dict[str, Any]]:
    """断言 10 + 改动 3 的改写规范，逐条给出违反了哪几项。

    检查项（全部来自 内部方案 的规范表，不含本模块自创的口径）：
      len 长度 ≤ MAX_SURFACE_CHARS
      digit 数字 ≤ MAX_SURFACE_DIGITS
      subj 不以「你」开头（统一无主语祈使句）
      style 不含 WRITTEN_MARKERS 里的书面连接词
      form key_form=statement 的不得含 QUESTION_MARKERS
      name 不含 identity 里的本人姓名（F1 的同一判据，改写后必须仍为 0）
      t3 T3 专属：不含「不表态/不评价/不表明」半句，且不含 _T3_STANCE_MARKERS
    """
    want = set(layers)
    markers = t3_stance_markers()
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("leakage_label") not in want:
            continue
        sid = r["scenario_id"]
        sf = surface_of(r)
        bad: list[str] = []
        if not sf:
            bad.append("empty")
        if len(sf) > MAX_SURFACE_CHARS:
            bad.append("len")
        if digits_in(sf) > MAX_SURFACE_DIGITS:
            bad.append("digit")
        if sf.startswith(SUBJECT_PREFIX):
            bad.append("subj")
        if any(m in sf for m in WRITTEN_MARKERS):
            bad.append("style")
        if key_form(sid) == "statement" and _QUESTION_RE.search(sf):
            bad.append("form")
        if key_form(sid) == "statement" and _has_interrogative_frame(sf):
            bad.append("form_frame")
        nm = identity_name(identity_of(r))
        if nm and nm in sf:
            bad.append("name")
        if r.get("leakage_label") == "T3":
            if _STANCE_MUTE_RE.search(sf):
                bad.append("t3_mute")
            if any(m in sf for m in markers):
                bad.append("t3_stance")
        if bad:
            out.append({"scenario_id": sid, "codes": bad, "surface": sf})
    return out


# ------------------------------------------------- F8：kb 家族与 sub_domain 错配
# 内部方案 执行期发现的**计划外缺陷**（2026-09-06 决策记录修）：17 个 marketing base
# （全是 v6.5 扩样新增的 49–68 号）的 `agent_policy.knowledge_base` 挂的是别的业务域的模板 ——
# 例：MAR-ALL-55「正信法务咨询 / 法律咨询线索收集」挂的却是「为蓝领用工平台，为企业匹配
# 普工/技工/服务员等岗位」，MAR-ALL-50「锐途职业培训 / 成人考证」挂的是「对接合作律所提供
# 法律咨询」。它不在 F1–F7 里，但会让改动 1 的「产品条目同时进 db_seed 与 knowledge_base」
# 无法自洽落地（往一份法律口径的 kb 里塞培训课程，客服照口径答就是答错域）。
#
# ⚠ 正确家族**不能按多数派推**：法律咨询线索收集 7 个 base 里错配的占 5 个，多数派恰好是错的
# 那一侧。所以这张表是逐域人工定的（依据：company 名与 ctx 的业务语义），改它要给出理由。
KB_FAMILY_SIG: dict[str, str] = {
    "A": "为蓝领用工平台",
    "B": "面向目标客户推介当期主推产品",
    "C": "对接合作律所提供法律咨询",
}
KB_FAMILY_CORRECT: dict[str, str] = {
    "蓝领招聘": "A",
    "司机撮合": "B",
    "成人考证/职业培训电销": "B",
    "校园招聘": "B",
    "法律咨询线索收集": "C",
}


def kb_family(kb: list[str] | None) -> str | None:
    """按 kb[0] 的特征串判家族；认不出返回 None（hotline/collection 的 kb 不走这套模板）。"""
    if not kb:
        return None
    for fam, sig in KB_FAMILY_SIG.items():
        if sig in kb[0]:
            return fam
    return None


def kb_family_mismatches(rows: Iterable[dict]) -> list[dict[str, Any]]:
    """断言 13：marketing 各 sub_domain 的 kb 家族必须等于 `KB_FAMILY_CORRECT`。"""
    out = []
    seen: set[str] = set()
    for r in rows:
        b = base_id(r["scenario_id"])
        if b in seen or r.get("role") != "marketing":
            continue
        seen.add(b)
        sub = r.get("sub_domain")
        want = KB_FAMILY_CORRECT.get(sub)
        if not want:
            continue
        got = kb_family((r.get("agent_policy") or {}).get("knowledge_base") or [])
        if got != want:
            out.append({"base": b, "sub_domain": sub, "company": r.get("company"),
                        "family": got, "expected": want})
    return out


def product_kb_missing(rows: Iterable[dict]) -> list[str]:
    """断言 14：改动 1 要求「同一条产品同时出现在 db_seed 与 knowledge_base」。

    只查 marketing 的 products 首行产品名是否在 kb 里出现过（客服才照得到同一口径）。
    """
    out = []
    seen: set[str] = set()
    for r in rows:
        b = base_id(r["scenario_id"])
        if b in seen or r.get("role") != "marketing":
            continue
        seen.add(b)
        prods = (r.get("db_seed") or {}).get("products") or []
        if not prods:
            continue
        name = list(prods[0])[0]
        kb = (r.get("agent_policy") or {}).get("knowledge_base") or []
        if not any(name in line for line in kb):
            out.append(b)
    return out


# ------------------------------------------- F5b：db_seed 的派生视图也不许留占位符
# F5 的原判据只扫 `db_seed`，但同一批业务事实在数据里还有**派生视图**：
# `background_card.consensus`（经 `run_batch.py:197` 的 `background_consensus` 进用户 prompt，
# 渲染成「背景：…」）、`background_card.world_state`、`background_card.agent_knows`。
# 内部方案 执行期实测：改完 db_seed 之后仍有 **96 个 base** 的 background_card 写着
# 「产品99 / 示例商品」⇒ 只修 db_seed 会让同一条事实在一份数据里有**两个说法**
# （客服档案说「企业岗位发布体验档」，用户背景说「产品99」），比原来的占位符更糟。
# 所以补这条判据。⚠️ 用词比 F5 窄：**不含「测试」** —— 那是常用词
# （`expected_key_behavior` 里「通过复述测试客服是否能准确接住」是正常表述，改前就有），
# 拿它扫散文只会产假违规，正是本仓登记过的「校验器口径过宽把真问题淹掉」。
PLACEHOLDER_IN_PROSE = re.compile(r"示例商品|示例产品|示例|产品\d+|占位")


def derived_view_placeholders(rows: Iterable[dict]) -> list[dict[str, Any]]:
    """断言 15：`background_card` 里不许再有占位商品名（逐 base 去重报告）。"""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        b = base_id(r["scenario_id"])
        if b in seen:
            continue
        bc = r.get("background_card") or {}
        blob = json.dumps(bc, ensure_ascii=False)
        m = PLACEHOLDER_IN_PROSE.search(blob)
        if m:
            seen.add(b)
            out.append({"base": b, "role": r.get("role"), "hit": m.group(0),
                        "consensus": bc.get("consensus")})
    return out


def background_card_db_mismatch(rows: Iterable[dict]) -> list[dict[str, Any]]:
    """断言 16：`background_card.world_state` 镜像的商品/产品名必须与 `db_seed` 逐字相同。

    守的是「一条业务事实只有一种说法」：`world_state.product` / `.product_name` 是 db_seed 首行
    名称的镜像字段，两者必须逐字相等，否则客服档案与用户背景会各说一套。

    ⚠️ **不检查 `consensus` 里是否逐字出现该产品名**：consensus 是人写的自然语句，允许用简称 ——
    实测 2 条 v6.5 base 就是这么写的（`HOT-POS-56` 的 consensus 写「2019 款朗逸」而 db_seed 是
    「2019款朗逸 1.5L 自动舒适版」；`HOT-ALL-60` 写「2019 款二手 SUV」而 db_seed 是「2019款二手SUV」），
    指的是同一辆车，不是两种说法。卡片里不许留**占位名**这件事由断言 15 守。
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        b = base_id(r["scenario_id"])
        if b in seen:
            continue
        seen.add(b)
        ws = (r.get("background_card") or {}).get("world_state") or {}
        db = r.get("db_seed") or {}
        # 镜像字段 → db_seed 首行的对应列。marketing 有四列被镜像（名称/类别/价格/活动），
        # hotline 只镜像商品名；只比名称会漏掉「产品名换了、活动名还留着占位的满减活动」这类半截同步。
        pairs: list[tuple[str, Any, Any]] = []
        if db.get("products"):
            row = list(db["products"][0])
            pairs = [("world_state.product_name", ws.get("product_name"), row[0]),
                     ("world_state.category", ws.get("category"), row[1]),
                     ("world_state.price", ws.get("price"), row[2]),
                     ("world_state.promotion", ws.get("promotion"), row[3])]
        elif db.get("orders"):
            row = list(db["orders"][0])
            pairs = [("world_state.product", ws.get("product"), row[1])]
        else:
            continue
        for field, got, want in pairs:
            if got is not None and got != want:
                out.append({"base": b, "field": field, "card": got, "db_seed": want})
    return out


def frozen_t2_snapshot() -> dict:
    """读 T2 冻结快照（断言 11 用）。快照由 `内部脚本` 一次性生成，
    CI 只读不重算 —— 期望值从被测数据现算等于没有闸门（实现纪律 2）。"""
    return json.loads(T2_FROZEN_PATH.read_text(encoding="utf-8"))


def t2_drift(rows: Iterable[dict], snapshot: dict | None = None) -> dict[str, Any]:
    """断言 11：T2 层 surface 与冻结快照逐字节比对，返回漂移明细。"""
    snap = snapshot or frozen_t2_snapshot()
    frozen: dict[str, str] = snap["surfaces"]
    cur = {r["scenario_id"]: surface_of(r) for r in rows if r.get("leakage_label") == FROZEN_LAYER}
    changed = sorted(k for k in frozen if frozen[k] != cur.get(k))
    return {
        "frozen_count": len(frozen),
        "current_count": len(cur),
        "id_set_diff": sorted(set(frozen) ^ set(cur)),
        "changed": changed,
        "expected_frozen_hits": {c: len(snap["defect_hits_within_t2_at_freeze"].get(c, []))
                                 for c in snap.get("frozen_scope_codes", SURFACE_TEXT_CODES)},
    }


def shared_field_drift(base_rows: list[dict], rows: list[dict]) -> dict[str, list[str]]:
    """断言 12：base 层与展开层除 `surface` 外必须逐字段相同。

    这条不在 内部方案 的十条里，是本次执行中发现的必要护栏：实测开工前 base 与展开层
    在 db_seed / identity / user_facts / key_event(除 surface) / mission / business_context /
    agent_policy / expected_key_behavior / background_card 上 **202/202 全等**，
    说明 base 层就是展开层去掉 surface 的副本。内部方案 只写了 base「改 db_seed / identity」，
    但改动 1 要动 `agent_policy.knowledge_base`、改动 2 要动 `key_event` ——
    不同步就会造出第二个真值源，正是 要消灭的东西。故把这条不变量机检化。
    """
    exp: dict[str, dict] = {}
    for r in rows:
        exp.setdefault(r["base_scenario_id"], {})[r.get("leakage_label")] = r

    def _canon(v: Any) -> str:
        return json.dumps(v, ensure_ascii=False, sort_keys=True)

    drift: dict[str, list[str]] = {}
    for br in base_rows:
        b = br["scenario_id"]
        layers = exp.get(b) or {}
        t1 = layers.get("T1")
        if not t1:
            drift.setdefault("missing_expansion", []).append(b)
            continue
        pairs: list[tuple[str, Any, Any]] = [
            ("db_seed", br.get("db_seed"), t1.get("db_seed")),
            ("business_context", br.get("business_context"), t1.get("business_context")),
            ("agent_policy", br.get("agent_policy"), t1.get("agent_policy")),
            ("expected_key_behavior", br.get("expected_key_behavior"), t1.get("expected_key_behavior")),
            ("background_card", br.get("background_card"), t1.get("background_card")),
            ("identity", user_script(br).get("identity"), user_script(t1).get("identity")),
            ("user_facts", user_script(br).get("user_facts"), user_script(t1).get("user_facts")),
            ("user_goal", user_script(br).get("user_goal"), user_script(t1).get("user_goal")),
            ("mission", user_script(br).get("mission"), user_script(t1).get("mission")),
            (
                "key_event(-surface)",
                {k: v for k, v in key_event(br).items() if k != "surface"},
                {k: v for k, v in key_event(t1).items() if k != "surface"},
            ),
        ]
        for field, bv, ev in pairs:
            if _canon(bv) != _canon(ev):
                drift.setdefault(field, []).append(b)
        # 展开层三层之间这些字段也必须一致（surface 除外）
        for field in ("identity", "user_facts"):
            vals = {_canon(user_script(layers[L]).get(field)) for L in layers}
            if len(vals) != 1:
                drift.setdefault(f"{field}(层间)", []).append(b)
        vals = {_canon(layers[L].get("db_seed")) for L in layers}
        if len(vals) != 1:
            drift.setdefault("db_seed(层间)", []).append(b)
    return drift
