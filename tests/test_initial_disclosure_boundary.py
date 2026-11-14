"""Protect explicit first-turn disclosure timing without judging later semantics."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import fake_instruction_entry

from listen2serve.runtime.tts_normalize import normalize_tts_text
from listen2serve.runtime.user_simulator import UserSimulator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = next(
    json.loads(line)
    for line in (ROOT / "data/benchmark/scenarios.jsonl").read_text().splitlines()
    if json.loads(line)["scenario_id"] == "HOT-ALL-49-T3"
)
BAD = "我刚提的2019款朗逸，订单ORD2307，报告说无重大事故，结果右前门有钣金喷漆，这怎么回事？"
SAFE = "我想核对订单ORD2307，车况和检测报告不符，麻烦帮我登记。"


def make_sim(*, script=None, prosody_arm="state", db_seed=None):
    script = copy.deepcopy(script or SOURCE["user_script"])
    llm, tts = MagicMock(), MagicMock()
    tts.synthesize.return_value = SimpleNamespace(audio=b"\x00\x00" * 1600, sample_rate=16000)
    sim = UserSimulator(
        script,
        llm_gateway=llm,
        tts_gateway=tts,
        leakage_label="T3",
        scenario_id=SOURCE["scenario_id"],
        db_seed=db_seed,
        prosody_arm=prosody_arm,
        instruction_entry=fake_instruction_entry(script["key_event"]["state"]),
    )
    return sim, llm, tts


def response(text, **extra):
    return SimpleNamespace(
        text=json.dumps(
            {
                "text": text,
                "state": "doubtful",
                "is_key_turn": False,
                "end_call": False,
                "covered": [],
                "interrupt": False,
                **extra,
            },
            ensure_ascii=False,
        ),
        finish_reason="stop",
    )


def test_first_prompt_masks_private_value_and_same_value_in_db_without_mutating_source():
    script = copy.deepcopy(SOURCE["user_script"])
    fact = script["user_facts"]["issue_detail"]
    db = {"orders": [{"order_id": "ORD2307", "private_issue": fact}]}
    before = copy.deepcopy((script, db))
    sim, _, _ = make_sim(script=script, db_seed=db)
    prompt = sim._build_messages(1, "", forced=False)[1]["content"]
    assert fact not in prompt and "右前门有钣金喷漆" not in prompt
    assert "首轮披露锁定" in prompt and "问题情况补充" in prompt
    assert "ORD2307" in prompt and "检测结果与实车不符" in prompt
    assert script == before[0] and db == before[1] and sim.user_script == script


@pytest.mark.parametrize("field", ["planned_amount", "another_private_amount"])
def test_private_numeric_value_does_not_mask_a_larger_amount_or_order_token(field):
    script = copy.deepcopy(SOURCE["user_script"])
    script["user_facts"] = {field: 1800}
    script["disclosure_plan"] = {field: "客服核对具体可还金额后才透露"}
    db = {"account": [{"debt_amount": 18000, "planned_amount": 1800, "order_id": "ORD1800"}]}
    sim, _, _ = make_sim(script=script, db_seed=db)
    prompt = sim._build_messages(1, "", forced=False)[1]["content"]
    assert "'debt_amount': 18000" in prompt and "ORD1800" in prompt
    assert "'planned_amount': （首轮暂不披露）" in prompt
    assert "（首轮暂不披露）0" not in prompt


@pytest.mark.parametrize(
    "text",
    [BAD, "右前门有钣金喷漆，我想问一下这件事该怎么登记。", "右前门钣金喷漆，我想问一下这件事该怎么登记。"],
)
def test_actual_first_turn_leak_and_core_variants_fail_before_tts_after_one_retry(text):
    sim, llm, tts = make_sim()
    llm.chat.return_value = response(text)
    with pytest.raises(RuntimeError, match="硬违规"):
        sim.next_turn()
    assert llm.chat.call_count == 2 and not tts.synthesize.called
    assert sim.stats["retries"] == 1 and sim._turn_count == 0 and not sim.key_completed


def test_first_leak_retries_once_and_synthesizes_only_safe_order_and_general_issue():
    sim, llm, tts = make_sim()
    llm.chat.side_effect = [response(BAD), response(SAFE)]
    assert sim.next_turn().text == SAFE
    assert llm.chat.call_count == 2 and tts.synthesize.call_count == 1
    assert tts.synthesize.call_args.args[0] == normalize_tts_text(SAFE)
    assert "issue_detail" in llm.chat.call_args.args[0][-1]["content"]


def test_normal_first_request_and_mission_order_id_are_not_private_detail():
    sim, llm, tts = make_sim()
    llm.chat.return_value = response(SAFE)
    assert sim.next_turn().text == SAFE
    assert llm.chat.call_count == 1 and tts.synthesize.call_count == 1


def test_source_without_explicit_disclosure_condition_keeps_existing_behavior():
    script = copy.deepcopy(SOURCE["user_script"])
    script.pop("disclosure_plan")
    sim, llm, _ = make_sim(script=script)
    assert script["user_facts"]["issue_detail"] in sim._build_messages(1, "", forced=False)[1]["content"]
    llm.chat.return_value = response(BAD)
    assert sim.next_turn().text == BAD and llm.chat.call_count == 1


def test_current_specific_agent_question_is_not_yet_in_history_but_restores_existing_path():
    sim, llm, _ = make_sim()
    question = "订单ORD2307已经核实，请说一下您提车后发现的具体车况异常。"
    assert not sim._history
    assert (
        sim.user_script["user_facts"]["issue_detail"]
        in sim._build_messages(1, question, forced=False)[1]["content"]
    )
    llm.chat.return_value = response("提车后发现右前门有钣金喷漆，和检测报告不符。")
    assert sim.next_turn(question).text.startswith("提车后发现") and llm.chat.call_count == 1


def test_existing_assistant_history_keeps_original_disclosure_rule_without_claiming_it_was_met():
    sim, _, _ = make_sim()
    sim._history.append({"role": "assistant", "content": "您好。"})
    prompt = sim._build_messages(1, "", forced=False)[1]["content"]
    assert sim.disclosure_plan["issue_detail"] in prompt
    assert "披露条件已满足" not in prompt


def test_other_explicit_source_field_uses_the_same_first_turn_boundary():
    script = copy.deepcopy(SOURCE["user_script"])
    script["user_facts"] = {"repair_note": "发现主板反复掉电，随后设备重启"}
    script["disclosure_plan"] = {"repair_note": "客服核实情况并询问具体表现后才透露"}
    sim, llm, tts = make_sim(script=script)
    llm.chat.return_value = response("主板反复掉电，我想问一下怎么登记这次问题。")
    with pytest.raises(RuntimeError, match="硬违规"):
        sim.next_turn()
    assert llm.chat.call_count == 2 and not tts.synthesize.called


def test_source_optional_you_variant_is_not_a_way_to_reveal_the_same_detail():
    script = copy.deepcopy(SOURCE["user_script"])
    script["user_facts"]["issue_detail"] = "提车后发现右前门钣金喷漆，和检测报告不符"
    sim, llm, tts = make_sim(script=script)
    llm.chat.return_value = response("右前门有钣金喷漆，我想问一下这件事怎么登记。")
    with pytest.raises(RuntimeError, match="硬违规"):
        sim.next_turn()
    assert llm.chat.call_count == 2 and not tts.synthesize.called


def test_neutral_arm_keeps_latent_oracle_and_neutral_delivery_after_safe_retry():
    sim, llm, tts = make_sim(prosody_arm="neutral")
    llm.chat.side_effect = [response(BAD), response(SAFE)]
    first = sim.next_turn()
    assert first.state == "doubtful" and first.tts_control == sim._instruction_entry["non_key"]["neutral"]
    llm.chat.side_effect = None
    llm.chat.return_value = response("这通电话要处理的是不是订单ORD2307，您核对一下？", is_key_turn=True)
    key = sim.next_turn("这笔订单稍后核实，请先确认一下这通电话要处理的订单。")
    assert key.is_key_turn and key.state == sim.key_event["state"] == "doubtful"
    assert key.tts_control == sim._instruction_entry["non_key"]["neutral"]
    assert "[" not in key.text_tts and tts.synthesize.call_count == 2 and llm.chat.call_count == 3
