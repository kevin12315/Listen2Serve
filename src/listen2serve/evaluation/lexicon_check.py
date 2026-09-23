"""用户话术事后词表校验（内部方案 v6.0：关键轮不再跳过，全轮同口径复核）。

口径（与运行时防线同口径，事后独立复核）：
- 关键轮定位：轨迹 turns 的 is_key_turn（v6.0 运行期产出）；旧 run 无该字段
  时回落场景 measurement.critical_turn 兼容重评；
- T1 关键轮：正向命中校验——须命中 ≥1 个 key_event.state（或旧 oracle）状态
  子表词（validate_level_v2 T1 口径；neutral 态无子表词则不设要求）；
- T2/T3 全部轮（含关键轮）：禁命中任何明确情绪词（validate_level_runtime，
  运行期语境豁免表）；T1 非关键轮同样零命中（v6.0 起非关键轮统一克制中性）；
- 词表**动态** import listen2serve.vocab（单一真相源）；
- 纯事后度量：只产出指标，不拦截、不改写任何 run 产物。
"""

from __future__ import annotations

from typing import Any

# leakage_level（数据字段）→ 层级标签（vocab.validate_level 口径）
_LEVEL_BY_LEAKAGE = {
    "explicit": "T1",
    "implicit_consistent": "T2",
    "prosody_only": "T3",
}


def _scenario_level(scenario: dict[str, Any]) -> str:
    """解析场景透明度层级标签（T1/T2/T3）：优先 leakage_label，回落 leakage_level。"""
    label = str(scenario.get("leakage_label") or "").strip()
    if label in ("T1", "T2", "T3"):
        return label
    return _LEVEL_BY_LEAKAGE.get(str(scenario.get("leakage_level") or ""), "")


def _key_state(scenario: dict[str, Any]) -> str:
    """关键事件要求状态：v6.0 key_event.state；旧数据回落 oracle_state 归一。"""
    ke = (scenario.get("user_script") or {}).get("key_event") or {}
    if ke.get("state"):
        return str(ke["state"])
    from listen2serve.domains.base import normalize_state

    return normalize_state(scenario.get("oracle_state")) or ""


def _key_turn_of(scenario: dict[str, Any], traj: dict[str, Any]) -> int | None:
    """关键轮号：优先轨迹 is_key_turn（唯一），旧 run 回落场景 critical_turn。"""
    key_turns = [t.get("turn") for t in traj.get("turns") or [] if t.get("is_key_turn")]
    if len(key_turns) == 1:
        return key_turns[0]
    if key_turns:
        return None # 非唯一：防御性不判关键轮（评测主链路已跳过该场景）
    if any("is_key_turn" in t for t in traj.get("turns") or []):
        return None # v6 轨迹但 key_turn_missing
    return (scenario.get("measurement") or {}).get("critical_turn")


def check_user_lexicon(scenario: dict[str, Any], traj: dict[str, Any]) -> dict[str, Any]:
    """对单条轨迹的用户全部话术做词表校验（内部方案：关键轮不再跳过）。

    Returns:
        dict：vocab_checked_turns / vocab_violation_turns / vocab_violation_rate（无校验集为 None）
        / vocab_violations（逐条违规明细：turn/state/reason/text/is_key_turn）
    """
    # 调用时动态 import：词表扩容自动生效；validate_level_v2 = T1 关键轮正向契约
    from listen2serve.vocab import validate_level_runtime, validate_level_v2

    level = _scenario_level(scenario)
    critical = _key_turn_of(scenario, traj)
    key_state = _key_state(scenario)
    checked = 0
    violations: list[dict[str, Any]] = []
    for t in traj.get("turns") or []:
        text = (t.get("user_text") or "").strip()
        if not text or level not in ("T1", "T2", "T3"):
            continue
        is_key = t.get("turn") == critical and critical is not None
        if is_key and level == "T1":
            # T1 关键轮：正向命中校验（与运行时防线同口径）
            ok, reason = validate_level_v2(text, "T1", key_state)
        elif level == "T1":
            # T1 非关键轮：克制中性，零命中（v6.0 起纳入校验集）
            ok, reason = validate_level_runtime(text, "T2", t.get("user_state") or "")
        else:
            # T2/T3 全轮：零命中
            ok, reason = validate_level_runtime(text, level, t.get("user_state") or "")
        checked += 1
        if not ok:
            violations.append({
                "turn": t.get("turn"),
                "state": t.get("user_state") or "",
                "is_key_turn": is_key,
                "reason": reason,
                "text": text,
            })
    return {
        "vocab_checked_turns": checked,
        "vocab_violation_turns": len(violations),
        "vocab_violation_rate": round(len(violations) / checked, 4) if checked else None,
        "vocab_violations": violations,
    }
