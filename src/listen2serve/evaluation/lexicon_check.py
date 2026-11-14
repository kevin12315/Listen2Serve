"""Check user wording against the same lexicon contract used during simulation.

Explicit critical turns require a state-specific cue. All neutral turns reject
explicit emotion words with the runtime context exemptions. Checks report
violations without changing saved trajectories.
"""

from __future__ import annotations

from typing import Any

_LEVEL_BY_LEAKAGE = {
    "explicit": "T1",
    "implicit_consistent": "T2",
    "prosody_only": "T3",
}


def _scenario_level(scenario: dict[str, Any]) -> str:
    """Read the immutable source label for the lexicon contract."""
    label = str(scenario.get("leakage_label") or "").strip()
    if label in ("T1", "T2", "T3"):
        return label
    return _LEVEL_BY_LEAKAGE.get(str(scenario.get("leakage_level") or ""), "")


def _key_state(scenario: dict[str, Any]) -> str:
    """Read the expected critical state."""
    ke = (scenario.get("user_script") or {}).get("key_event") or {}
    if ke.get("state"):
        return str(ke["state"])
    from listen2serve.domains.base import normalize_state

    return normalize_state(scenario.get("oracle_state")) or ""


def _key_turn_of(scenario: dict[str, Any], traj: dict[str, Any]) -> int | None:
    """Locate a unique critical turn."""
    key_turns = [t.get("turn") for t in traj.get("turns") or [] if t.get("is_key_turn")]
    if len(key_turns) == 1:
        return key_turns[0]
    if key_turns:
        return None
    if any("is_key_turn" in t for t in traj.get("turns") or []):
        return None
    return (scenario.get("measurement") or {}).get("critical_turn")


def check_user_lexicon(scenario: dict[str, Any], traj: dict[str, Any]) -> dict[str, Any]:
    """Report all user-turn lexicon violations without modifying the trajectory."""

    from listen2serve.vocab import validate_level_runtime, validate_wording

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
            ok, reason = validate_wording(text, "T1", key_state)
        elif level == "T1":
            ok, reason = validate_level_runtime(text, "T2", t.get("user_state") or "")
        else:
            ok, reason = validate_level_runtime(
                text, level, t.get("user_state") or "", is_key_turn=is_key
            )
        checked += 1
        if not ok:
            violations.append(
                {
                    "turn": t.get("turn"),
                    "state": t.get("user_state") or "",
                    "is_key_turn": is_key,
                    "reason": reason,
                    "text": text,
                }
            )
    return {
        "vocab_checked_turns": checked,
        "vocab_violation_turns": len(violations),
        "vocab_violation_rate": round(len(violations) / checked, 4) if checked else None,
        "vocab_violations": violations,
    }
