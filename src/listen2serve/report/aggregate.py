"""Aggregate the released multi-turn metrics by wording, role and dialogue dynamics."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from typing import Any

from listen2serve.conditions import wording_for
from listen2serve.evaluation.incremental import dialogue_is_complete

VOICE_DIM_KEYS = ("role_voice_match", "state_voice_fit", "naturalness")


def aggregate_results(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by role, wording condition and dialogue dynamics."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for v in verdicts:
        key = (v.get("role", ""), wording_for(v), v.get("dynamics", ""), v.get("condition") or "")
        groups[key].append(v)

    rows = []
    for (role, wording, dynamics, condition), items in sorted(groups.items()):
        row = {
            "role": role,
            "wording": wording,
            "condition": condition or None,
            "dynamics": dynamics,
            **metric_block(items),
        }
        rows.append(row)

    return {"summary": metric_block(verdicts), "rows": rows}


def metric_block(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute one consistent metric block for each group."""
    complete = [item for item in items if dialogue_is_complete(item)]
    return {
        "n": len(items),
        "n_complete": len(complete),
        "n_partial": len(items) - len(complete),
        "key_turn_n": sum(item.get("key_turn_pass") is not None for item in items),
        "voice_n": sum(item.get("voice_pass") is not None for item in items),
        "flow_n": sum(isinstance(item.get("flow_rate"), (int, float)) for item in complete),
        "task_n": sum(isinstance(item.get("task_score"), (int, float)) for item in complete),
        "key_turn_pass": _rate_or_none(items, "key_turn_pass"),
        "key_turn_score_mean": _mean_or_none(items, "key_turn_score"),
        "key_behavior_met_rate": _hit_rate(items, "key_behavior_met"),
        "flow_rate_mean": _mean_or_none(complete, "flow_rate"),
        "flow_rate_in_window_mean": _mean_or_none(complete, "flow_rate_in_window"),
        "task_score_mean": _mean_or_none(complete, "task_score"),
        "voice_mean": _voice_detail_mean(items, "mean"),
        "voice_role_match_mean": _voice_detail_mean(items, "role_voice_match"),
        "voice_state_fit_mean": _voice_detail_mean(items, "state_voice_fit"),
        "voice_naturalness_mean": _voice_detail_mean(items, "naturalness"),
        "leakage_score_mean": _mean_or_none(items, "leakage_score"),
        "leakage_state_correct_rate": _rate_or_none(items, "leakage_state_correct"),
        "leakage_stance_marker_mean": _mean_or_none(items, "leakage_stance_markers"),
        "closing_signal_rate": _rate_or_none(complete, "closing_signal"),
        "closing_dangling_rate": _rate_or_none(complete, "closing_dangling"),
        "closing_ok_rate": _rate_or_none(complete, "closing_ok"),
        "voice_pass": _rate_or_none(items, "voice_pass"),
        "vocab_violation_rate": _vocab_violation_rate(items),
    }


def _level_of(v: dict[str, Any]) -> str:
    return wording_for(v)


def aggregate_marginals(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute marginal summaries with the same metric definitions."""
    by_wording = _grouped(verdicts, _level_of)
    scores = [r["leakage_score_mean"] for r in by_wording]
    monotonic = None
    if len(scores) > 1 and all(isinstance(s, (int, float)) for s in scores):
        monotonic = all(a > b for a, b in pairwise(scores))
    return {
        "overall": metric_block(verdicts),
        "by_wording": by_wording,
        "by_role": _grouped(verdicts, lambda v: str(v.get("role") or "")),
        "by_dynamics": _grouped(verdicts, lambda v: str(v.get("dynamics") or "")),
        "by_role_wording": _grouped(verdicts, lambda v: f"{v.get('role') or ''}／{_level_of(v)}"),
        "by_model": _grouped(verdicts, lambda v: str(v.get("model") or "")),
        "cells": aggregate_results(verdicts)["rows"],
        "leakage_monotonic": monotonic,
    }


def _grouped(verdicts: list[dict[str, Any]], keyfn) -> list[dict[str, Any]]:
    """按 keyfn 分组后逐组套 metric_block；组名升序（表格顺序稳定，便于逐版对账）。"""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for v in verdicts:
        buckets[keyfn(v)].append(v)
    return [{"group": k, **metric_block(items)} for k, items in sorted(buckets.items())]


PAIRED_KEYS = ("key_turn_score", "flow_rate", "task_score", "leakage_score")


def aggregate_cross_model(
    verdicts: list[dict[str, Any]],
    baseline: str = "",
) -> dict[str, Any]:
    """Compare models on shared scenario IDs; exclude unmatched scenarios."""
    models = sorted({str(v.get("model") or "") for v in verdicts} - {""})
    base = baseline if baseline in models else (models[0] if models else "")

    mono: dict[str, bool | None] = {}
    for m in models:
        sub = [v for v in verdicts if str(v.get("model") or "") == m]
        mono[m] = aggregate_marginals(sub)["leakage_monotonic"] if sub else None

    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for v in verdicts:
        sid = str(v.get("scenario_id") or "")
        if sid:
            idx[(str(v.get("model") or ""), sid)] = v
    base_sids = [sid for (m, sid) in idx if m == base]

    paired = []
    for m in models:
        if m == base:
            continue
        pairs = [(idx[(base, sid)], idx[(m, sid)]) for sid in base_sids if (m, sid) in idx]
        row: dict[str, Any] = {"model": m, "baseline": base, "n_paired": len(pairs)}
        for key in PAIRED_KEYS:
            diffs = [
                b[key] - a[key]
                for a, b in pairs
                if isinstance(a.get(key), (int, float))
                and isinstance(b.get(key), (int, float))
                and (
                    key not in {"flow_rate", "task_score"}
                    or (dialogue_is_complete(a) and dialogue_is_complete(b))
                )
            ]
            row[f"d_{key}"] = round(sum(diffs) / len(diffs), 4) if diffs else None
            row[f"n_{key}"] = len(diffs)
        wins = sum(1 for a, b in pairs if _kt(b) is not None and _kt(a) is not None and _kt(b) > _kt(a))
        losses = sum(1 for a, b in pairs if _kt(b) is not None and _kt(a) is not None and _kt(b) < _kt(a))
        row["win"], row["loss"] = wins, losses
        row["tie"] = len(pairs) - wins - losses
        paired.append(row)

    return {
        "models": models,
        "baseline": base,
        "by_model": _grouped(verdicts, lambda v: str(v.get("model") or "")),
        "by_model_wording": _grouped(verdicts, lambda v: f"{v.get('model') or ''}／{_level_of(v)}"),
        "monotonic_by_model": mono,
        "paired": paired,
    }


def _kt(v: dict[str, Any]) -> float | None:
    """配对胜负判据取 KeyTurnScore（弦外之音量规，本研究的主结论指标）。"""
    x = v.get("key_turn_score")
    return x if isinstance(x, (int, float)) else None


def _rate_or_none(items: list[dict[str, Any]], key: str) -> float | None:
    """Average boolean observations; return None when no observations exist."""
    vals = [i.get(key) for i in items if i.get(key) is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v) / len(vals), 4)


def _mean_or_none(items: list[dict[str, Any]], key: str) -> float | None:
    """Average numeric observations; return None when no observations exist."""
    vals = [i.get(key) for i in items if isinstance(i.get(key), (int, float))]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _hit_rate(items: list[dict[str, Any]], key: str) -> float | None:
    """Count full or partial critical behavior matches."""
    vals = [i.get(key) for i in items if i.get(key) is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v in ("是", "部分")) / len(vals), 4)


def _voice_detail_mean(items: list[dict[str, Any]], key: str) -> float | None:
    """连续主指标：voice_detail 内字段（mean 或三维单项 1–5 分）的宏平均。
    无 voice_detail（如 text 模态 Voice 跳过）时返回 None（渲染 N/A）。"""
    vals = [(i.get("voice_detail") or {}).get(key) for i in items]
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def _vocab_violation_rate(items: list[dict[str, Any]]) -> float | None:
    """Micro-average violations across all checked user turns."""
    checked = sum(i.get("vocab_checked_turns") or 0 for i in items)
    violations = sum(i.get("vocab_violation_turns") or 0 for i in items)
    if not checked:
        return None
    return round(violations / checked, 4)
