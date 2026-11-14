"""Regression coverage for the approved release order-reference correction."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmark"
APPROVED = {
    "HOT-ALL-49": ("ORD0014", "ORD2307"),
    "HOT-ALL-50": ("ORD0015", "ORD0016"),
    "HOT-ALL-52": ("ORD0018", "ORD0023"),
    "HOT-ALL-53": ("ORD0018", "ORD7302"),
    "HOT-ALL-54": ("ORD0018", "ORD3057"),
    "HOT-ALL-55": ("ORD0018", "ORD5209"),
    "HOT-POS-56": ("ORD0025", "ORD7301"),
    "HOT-POS-57": ("ORD0026", "ORD5207"),
    "HOT-POS-59": ("ORD0028", "ORD8803"),
    "HOT-ALL-60": ("ORD0013", "ORD5286"),
    "HOT-ALL-61": ("ORD0016", "ORD7301"),
    "HOT-ALL-62": ("ORD0021", "ORD8801"),
    "HOT-ALL-63": ("ORD0013", "ORD7205"),
    "HOT-POS-64": ("ORD0025", "ORD8801"),
    "HOT-POS-65": ("ORD0026", "ORD8801"),
    "HOT-POS-67": ("ORD0028", "ORD3057"),
}


def checker_module():
    spec = importlib.util.spec_from_file_location("check_dataset", ROOT / "scripts/check_dataset.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "count", "changed_rows", "original_sha256"),
    [
        (
            "scenarios_base.jsonl",
            145,
            16,
            "7d96b8559bf709751dd106808fb28d067f0fb3abc520019843b8361db327e1f2",
        ),
        (
            "scenarios.jsonl",
            290,
            32,
            "f39c8b06b746ada295940cdf68ec0cfa82c8251a6505d86b3f82d6674173da50",
        ),
    ],
)
def test_only_approved_policy_order_tokens_changed(name, count, changed_rows, original_sha256):
    """Undo exactly the two approved tokens and recover the original file bytes."""
    lines = (BENCH / name).read_text().splitlines(keepends=True)
    assert len(lines) == count
    restored = []
    changes = 0
    for raw in lines:
        row = json.loads(raw)
        base = row.get("base_scenario_id") or row["scenario_id"]
        if base not in APPROVED:
            restored.append(raw)
            continue
        old, canonical = APPROVED[base]
        assert row["db_seed"]["orders"][0][0] == canonical
        current_policy = row["agent_policy"]
        original_policy = copy.deepcopy(current_policy)
        for container, key in (
            (original_policy["knowledge_base"], 0),
            (original_policy["business_flow"][0], "goal"),
        ):
            assert container[key].count(canonical) == 1
            assert old not in container[key]
            container[key] = container[key].replace(canonical, old)
        current_json = json.dumps(current_policy, ensure_ascii=False)
        assert raw.count(current_json) == 1
        restored.append(raw.replace(current_json, json.dumps(original_policy, ensure_ascii=False), 1))
        changes += 1
    assert changes == changed_rows
    assert hashlib.sha256("".join(restored).encode()).hexdigest() == original_sha256


def test_release_revision_preserves_source_metadata_and_updates_fingerprints():
    version = json.loads((BENCH / "version.json").read_text())
    assert version["dataset_version"] == "v6.5"
    assert version["generated_at"] == "2026-09-16T18:57:17+0800"
    assert version["release"]["asset_revision"] == "order-reference-correction-1"
    correction = version["release"]["corrections"][0]
    assert correction["record_count"] == 48
    assert correction["string_replacements"] == 96
    assert set(correction["fields"]) == {
        "agent_policy.knowledge_base[0]",
        "agent_policy.business_flow[0].goal",
    }
    for name in ("scenarios_base", "scenarios"):
        path = BENCH / f"{name}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == version["assets"][name]["sha256"]
        assert len(path.read_text().splitlines()) == version["assets"][name]["count"]


def test_policy_reference_check_accepts_existing_orders_and_ignores_other_numbers():
    checker = checker_module()
    rows = [
        {
            "scenario_id": "example",
            "db_seed": {"orders": [["ORD2307"], ["ORD2308"]]},
            "agent_policy": {
                "knowledge_base": ["订单 ORD2307，费用199元"],
                "business_flow": [{"goal": "核实 ORD2308"}],
            },
        },
        {"scenario_id": "no-order", "agent_policy": {"knowledge_base": ["费用199元"]}},
    ]
    assert checker.policy_order_reference_errors(rows) == []


@pytest.mark.parametrize("field", ["knowledge_base", "business_flow"])
def test_policy_reference_check_finds_chinese_adjacent_orders(field):
    checker = checker_module()
    policy = {"knowledge_base": [], "business_flow": []}
    if field == "knowledge_base":
        policy[field] = ["订单ORD0014相关问题"]
    else:
        policy[field] = [{"goal": "核对订单ORD0014并登记"}]
    row = {"scenario_id": "example", "db_seed": {"orders": [["ORD2307"]]}, "agent_policy": policy}
    assert checker.policy_order_reference_errors([row]) == ["example: ORD0014"]


def test_policy_reference_check_ignores_embedded_latin_tokens():
    checker = checker_module()
    row = {
        "scenario_id": "example",
        "db_seed": {"orders": [["ORD2307"]]},
        "agent_policy": {"knowledge_base": ["XORD0014 ORD0014X 1ORD0014 ORD0014A"]},
    }
    assert checker.policy_order_reference_errors([row]) == []


@pytest.mark.parametrize("field", ["knowledge_base", "business_flow"])
def test_dataset_check_rejects_stale_reference_even_with_matching_fingerprint(
    tmp_path, monkeypatch, capsys, field
):
    checker = checker_module()
    scratch = tmp_path / "benchmark"
    shutil.copytree(BENCH, scratch)
    path = scratch / "scenarios_base.jsonl"
    rows = checker.jl(path)
    row = next(row for row in rows if row["scenario_id"] == "HOT-ALL-49")
    if field == "knowledge_base":
        row["agent_policy"][field][0] = row["agent_policy"][field][0].replace("ORD2307", "ORD0014")
    else:
        goal = row["agent_policy"][field][0]["goal"]
        row["agent_policy"][field][0]["goal"] = goal.replace("ORD2307", "ORD0014")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    version_path = scratch / "version.json"
    version = json.loads(version_path.read_text())
    version["assets"]["scenarios_base"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    version_path.write_text(json.dumps(version))
    monkeypatch.setattr(checker, "BENCH", scratch)
    assert checker.main() == 1
    output = capsys.readouterr().out
    assert "✗ scenarios_base 客服策略显式订单号与订单表一致 HOT-ALL-49: ORD0014" in output
