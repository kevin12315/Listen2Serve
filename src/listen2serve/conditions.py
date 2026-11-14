"""Paper terminology at the boundary of the immutable scenario assets."""

from __future__ import annotations

from typing import Any


def wording_for(scenario: dict[str, Any]) -> str:
    """Return the wording condition without rewriting a scenario or its ID."""
    if scenario.get("wording") in ("explicit", "neutral"):
        return scenario["wording"]
    level = scenario.get("leakage_level") or scenario.get("leakage_label")
    mapping = {"explicit": "explicit", "prosody_only": "neutral", "T1": "explicit", "T3": "neutral"}
    if level not in mapping:
        raise ValueError(f"Unsupported wording condition: {level!r}")
    return mapping[level]


def condition_code(wording: str, prosody_arm: str) -> str:
    """Name the three released cue conditions using the paper's notation."""
    try:
        return {("explicit", "state"): "E/E", ("neutral", "state"): "N/E", ("neutral", "neutral"): "N/N"}[
            (wording, prosody_arm)
        ]
    except KeyError:
        raise ValueError(f"Unsupported cue condition: {wording}/{prosody_arm}") from None
