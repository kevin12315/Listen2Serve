"演绎侧数据卫生规则的**唯一实现点**"

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS_PATH = PROJECT_ROOT / "data" / "benchmark" / "scenarios.jsonl"
SCENARIOS_BASE_PATH = PROJECT_ROOT / "data" / "benchmark" / "scenarios_base.jsonl"

MAX_SURFACE_CHARS = 20  # 「长度 ≤20 字」
MAX_SURFACE_DIGITS = 1  # 「数字 ≤1 个」
SUBJECT_PREFIX = "你"  # 「人称：统一为无主语祈使句」
# 「文体：口语，不许『且曾／及是否／以及是否／并已／系属／及具体／及可』」
# 状态词与中性措辞的语气特征，共用于数据检查与运行时筛查。
WRITTEN_MARKERS: tuple[str, ...] = (
    "且曾",
    "及是否",
    "以及是否",
    "并已",
    "系属",
    "及具体",
    "及可",
)
# 「句式：statement 不许含『是否／能不能／是不是／怎么办／多少／什么时候／吗／呢』」
QUESTION_MARKERS: tuple[str, ...] = (
    "是否",
    "能不能",
    "是不是",
    "怎么办",
    "多少",
    "什么时候",
    "吗",
    "呢",
)

REPAIRED_LAYERS: tuple[str, ...] = ("T1", "T3")

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
    "Key event."
    return user_script(row).get("key_event") or {}


def surface_of(row: dict) -> str:
    return key_event(row).get("surface") or ""


def identity_of(row: dict) -> str:
    return user_script(row).get("identity") or ""


def identity_name(identity: str) -> str | None:
    """`identity` 开头的 2–4 字中文姓名（F1/G5 的判据来源，两处必须同源）。"""
    m = _NAME_RE.match(identity or "")
    return m.group(1) if m else None


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
    "T3 忌讳的结论式表述。**import 不要抄**（ G3 明写）"
    from listen2serve.runtime.user_simulator import NEUTRAL_STANCE_MARKERS

    return tuple(NEUTRAL_STANCE_MARKERS)


def t3_ban_terms(state: str | None = None) -> tuple[str, ...]:
    "T3 的逐态「动作语用」禁词。**唯一源 = `data/benchmark/neutral_wording_ban.json`，import 不要抄。**"
    from listen2serve.data import wording_ban as BAN

    if state is not None:
        return tuple(BAN.terms(state, ("ban",)))
    seen: list[str] = []
    for s in BAN.states():
        for w in BAN.terms(s, ("ban",)):
            if w not in seen:
                seen.append(w)
    return tuple(seen)


def t3_ban_hits(rows: Iterable[dict], scope: str = "own") -> list[tuple[str, str, list[str]]]:
    "T3 层 `surface` 命中禁词的清单，返回 `[(scenario_id, state, 命中词…)]`"
    if scope not in ("own", "cross", "any"):
        raise ValueError(f"未知 scope {scope}；合法值 own/cross/any")
    from listen2serve.data import wording_ban as BAN

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


def field_ban_hits(
    rows: Iterable[dict], field: str, states: tuple[str, ...] | None = None
) -> list[tuple[str, str, list[str]]]:
    'T3 层 `objective` / `intent` 命中**本态** ban 词的清单（口径同 `t3_ban_hits(scope="own")`）'
    if field not in ("objective", "intent"):
        raise ValueError(f"未知字段 {field}；合法值 objective/intent")
    from listen2serve.data import wording_ban as BAN

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
    "T3 层 `trigger_hint` 命中**本态** ban 词的清单（F13d，**只提供事实，谁都不许拿它判定**）"
    from listen2serve.data import wording_ban as BAN

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


T3_ACCEPTANCE_PATTERNS = (
    "那就",
    "就按",
    "按这个",
    "按你们",
    "照着",
    "我先",
    "我就",
    "听你的",
    "按你说的",
    "没问题",
    "可以的",
    "行，",
    "行。",
    "好，",
    "好。",
    "成，",
    "成。",
)
"""Neutral wording must avoid acceptance or commitment cues."""

T3_COMMAND_PATTERNS = (
    "说清楚",
    "讲清楚",
    "说明白",
    "你给我",
    "直接告诉我",
    "赶紧",
    "快点",
    "别绕",
    "明确一下",
    "给我个准话",
    "立刻",
    "马上给",
)
"""G8：命令 / 催办口气。T3 的 prompt 明写「也不要用命令或催办的口气（『你先说清楚』
『麻烦明确一下』这类）」，而原六道闸没有一条查它 —— 小样 `MAR-ALL-23-T3`
「你们到底是怎么拿到我号码的，说清楚。」就这么过了。"""

T3_GATE_PATTERN_LAYERS = ("T3",)
"""G7/G8 只作用 T3：T1 的设计就是「态度在文本里」，命令口气与承接结论在 T1 是合法表达
（小样 T1 例句「你赶紧把我号码删了，别再打过来了啊」正是 displeased 该有的样子）。"""


def duplicate_surfaces(
    rows: Iterable[dict], layers: tuple[str, ...] = REPAIRED_LAYERS
) -> dict[str, list[str]]:
    """返回不同基础场景中逐字相同的表述指令，保留至少出现两次的组。

    相同状态下重复的表述指令可能产生相同台词；此检查在数据侧标记这种复用。
    默认检查中性措辞场景，各场景的业务事实和文本保持原样。
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


# ---------------------------------------------------------------- F1–F7 体检


def audit_rows(rows: Iterable[dict]) -> OrderedDict[str, list[str]]:
    "按  的七条判据扫一遍，返回 `缺陷码 → scenario_id 列表`"
    flags: OrderedDict[str, list[str]] = OrderedDict((c, []) for c in DEFECT_CODES)

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


def audit_files(
    scenarios_path: Path | str = SCENARIOS_PATH,
) -> tuple[OrderedDict[str, list[str]], list[dict], int]:
    """读盘版体检，返回 `(flags, rows, base 数)` —— 与原 `_audit()` 的契约一致。"""
    rows = load_rows(scenarios_path)
    return audit_rows(rows), rows, len({r["base_scenario_id"] for r in rows})


def hits_by_layer(flags: dict[str, list[str]], rows: list[dict]) -> dict[str, dict[str, int]]:
    """按原始措辞标签统计各检查项命中的记录数。"""
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
    "断言 8：base 层不得再有 `key_event.surface`"
    return [r["scenario_id"] for r in base_rows if surface_of(r)]


def objective_intent_missing(rows: Iterable[dict]) -> list[str]:
    """检查每条展开记录是否包含 objective 与 intent。"""
    return [
        r["scenario_id"] for r in rows if not (key_event(r).get("objective") and key_event(r).get("intent"))
    ]


def surface_style_violations(
    rows: Iterable[dict], layers: Iterable[str] = REPAIRED_LAYERS
) -> list[dict[str, Any]]:
    "检查表述字段，并逐条返回违反的规则。"
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
            out.append(
                {"base": b, "sub_domain": sub, "company": r.get("company"), "family": got, "expected": want}
            )
    return out


def product_kb_missing(rows: Iterable[dict]) -> list[str]:
    """检查同一产品在 db_seed 与 knowledge_base 中的引用是否一致。

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
        name = next(iter(prods[0]))
        kb = (r.get("agent_policy") or {}).get("knowledge_base") or []
        if not any(name in line for line in kb):
            out.append(b)
    return out


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
            out.append(
                {"base": b, "role": r.get("role"), "hit": m.group(0), "consensus": bc.get("consensus")}
            )
    return out


def background_card_db_mismatch(rows: Iterable[dict]) -> list[dict[str, Any]]:
    "断言 16：`background_card.world_state` 镜像的商品/产品名必须与 `db_seed` 逐字相同"
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
            pairs = [
                ("world_state.product_name", ws.get("product_name"), row[0]),
                ("world_state.category", ws.get("category"), row[1]),
                ("world_state.price", ws.get("price"), row[2]),
                ("world_state.promotion", ws.get("promotion"), row[3]),
            ]
        elif db.get("orders"):
            row = list(db["orders"][0])
            pairs = [("world_state.product", ws.get("product"), row[1])]
        else:
            continue
        for field, got, want in pairs:
            if got is not None and got != want:
                out.append({"base": b, "field": field, "card": got, "db_seed": want})
    return out


def shared_field_drift(base_rows: list[dict], rows: list[dict]) -> dict[str, list[str]]:
    "断言 12：base 层与展开层除 `surface` 外必须逐字段相同"
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
