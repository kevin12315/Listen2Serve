"""Keep neutral prompts and same-turn repair context aligned with source fields."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import fake_instruction_entry

from listen2serve.runtime.user_simulator import UserSimulator

ROOT = Path(__file__).resolve().parents[1]
ROWS = {
    s["scenario_id"]: s
    for s in map(json.loads, (ROOT / "data/benchmark/scenarios.jsonl").read_text().splitlines())
}
AGENT = "明白您有疑虑，您可以登录官方 APP 核对明细账单，那里会列出公摊面积的具体计算方式。"
FIRST = {
    "text": "我哪有空上什么软件查，你们把明细寄给我，我看看再说。",
    "state": "displeased",
    "is_key_turn": False,
    "end_call": False,
    "covered": [],
    "interrupt": False,
}
SAFE = "明细您按原来的方式发我，我核对以后再联系。"


def make_sim(sid="COL-ALL-17-T3", *, prosody_arm="state", script=None):
    row = ROWS[sid]
    script = copy.deepcopy(script or row["user_script"])
    llm, tts = MagicMock(), MagicMock()
    tts.synthesize.return_value = SimpleNamespace(audio=b"\x00\x00" * 1600, sample_rate=16000)
    sim = UserSimulator(
        script,
        llm_gateway=llm,
        tts_gateway=tts,
        role=row["role"],
        dynamics=row["dynamics"],
        scenario_id=sid,
        leakage_label=row["leakage_label"],
        prosody_arm=prosody_arm,
        instruction_entry=fake_instruction_entry(script["key_event"]["state"]),
    )
    return sim, llm, tts


def response(text=SAFE, **extra):
    value = {
        "text": text,
        "state": "displeased",
        "is_key_turn": False,
        "end_call": False,
        "covered": [],
        "interrupt": False,
        **extra,
    }
    return SimpleNamespace(text=json.dumps(value, ensure_ascii=False), finish_reason="stop")


def test_neutral_view_projects_only_marked_key_mission_and_labels_goal_internal():
    sim, _, _ = make_sim()
    before = copy.deepcopy((sim.user_script, sim.mission, sim.mission_entries, sim.key_event))
    messages = sim._build_messages(4, AGENT, forced=False)
    text = messages[1]["content"]
    mission = text.split("【剧情要点】", 1)[1].split("【关键事件】表面业务话题", 1)[0]
    assert f"3. 【关键事件】{sim.key_event['surface']}" in mission
    assert "明确拒绝眼下的还款安排" not in mission
    for number in (1, 2, 4, 5):
        assert f"{number}. {sim.mission[number - 1]}" in mission
    assert f"【内部剧情目标】{sim.user_script['user_goal']}" in text
    assert "不作为用户台词" in text and sim.key_event["trigger_hint"] in text
    assert "JSON state仍为displeased" in text
    assert (sim.user_script, sim.mission, sim.mission_entries, sim.key_event) == before


def test_explicit_first_prompt_remains_byte_identical_to_frozen_baseline():
    sim, _, _ = make_sim("COL-ALL-17-T1")
    messages = sim._build_messages(4, AGENT, forced=False)
    assert (
        hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
        == "7d988bd3da10e6bbba3d86d2588fc6072fc92390564685678f384302b5ca41d1"
    )


def test_neutral_without_source_surface_keeps_existing_goal_and_mission():
    script = copy.deepcopy(ROWS["COL-ALL-17-T3"]["user_script"])
    script["key_event"].pop("surface")
    sim, _, _ = make_sim(script=script)
    text = sim._build_messages(4, AGENT, forced=False)[1]["content"]
    assert "3. 【关键事件】明确拒绝眼下的还款安排" in text
    assert f"【目标】{script['user_goal']}" in text and "【内部剧情目标】" not in text


def test_actual_col17_vocabulary_repair_contains_the_first_json_and_same_turn_context():
    sim, llm, tts = make_sim()
    sim._turn_count = 3
    llm.chat.side_effect = [
        SimpleNamespace(text=json.dumps(FIRST, ensure_ascii=False), finish_reason="stop"),
        response(),
    ]
    result = sim.next_turn(AGENT)
    retried = llm.chat.call_args.args[0]
    assert json.loads(retried[-2]["content"]) == FIRST
    assert "同一用户轮" in retried[-1]["content"] and "客服上文没有变化" in retried[-1]["content"]
    assert "不得借修文跳到下一剧情" in retried[-1]["content"]
    assert AGENT in retried[1]["content"] and result.text == SAFE and not result.is_key_turn
    assert llm.chat.call_count == 2 and tts.synthesize.call_count == 1 and sim.stats["retries"] == 1


def test_prompt_repair_does_not_add_a_new_runtime_key_or_covered_lock():
    sim, llm, _ = make_sim()
    sim._turn_count = 3
    sim._history = [{"role": "assistant", "content": "继续拖欠会产生滞纳金，您明天方便先还一部分吗？"}]
    llm.chat.side_effect = [
        response(**FIRST),
        response("这单的后续联系安排麻烦您再核对一下。", is_key_turn=True, covered=[3]),
    ]
    assert sim.next_turn(AGENT).is_key_turn
    assert llm.chat.call_count == 2


@pytest.mark.parametrize("first_kind", ["invalid_json", "too_short", "invalid_state"])
def test_existing_json_and_hard_protocol_repair_remains_two_calls(first_kind):
    sim, llm, tts = make_sim("COL-ALL-17-T1")
    first = (
        SimpleNamespace(text="not JSON", finish_reason="stop")
        if first_kind == "invalid_json"
        else (response("你好") if first_kind == "too_short" else response(state="invalid-state"))
    )
    llm.chat.side_effect = [first, response()]
    assert sim.next_turn(AGENT).text == SAFE
    repaired = llm.chat.call_args.args[0]
    if first_kind == "invalid_json":
        assert repaired[-2]["content"] == "(上次输出不合格)"
    else:
        assert isinstance(json.loads(repaired[-2]["content"]), dict)
    assert llm.chat.call_count == 2 and tts.synthesize.call_count == 1


def test_forced_budget_can_still_complete_the_key_event_during_existing_repair():
    sim, llm, tts = make_sim()
    sim._turn_count = sim.hard_budget - 3
    llm.chat.side_effect = [
        response(**FIRST),
        response("后面还要联系几次，麻烦您按记录说明一下。", is_key_turn=True, covered=[3]),
    ]
    result = sim.next_turn(AGENT)
    assert result.is_key_turn and result.state == "displeased"
    assert llm.chat.call_count == 2 and tts.synthesize.call_count == 1


def test_neutral_voice_and_latent_key_state_stay_separate_after_repair():
    sim, llm, _ = make_sim(prosody_arm="neutral")
    sim._history = [{"role": "assistant", "content": "继续拖欠会产生滞纳金，您明天方便先还一部分吗？"}]
    llm.chat.side_effect = [
        response("这方案我不接受，眼下这个安排我不同意。", is_key_turn=True),
        response("后面还要联系几次，麻烦您按记录说明一下。", is_key_turn=True),
    ]
    result = sim.next_turn(AGENT)
    assert result.is_key_turn and result.state == "displeased"
    assert result.tts_control == sim._instruction_entry["non_key"]["neutral"]
    assert "[" not in result.text_tts and llm.chat.call_count == 2
