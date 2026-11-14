"""Capture live prompt failures without new inference calls or source changes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import fake_instruction_entry

from listen2serve.runtime.user_simulator import UserSimulator

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    row["scenario_id"]: row
    for row in (
        json.loads(line) for line in (ROOT / "data/benchmark/scenarios.jsonl").read_text().splitlines()
    )
}


def make_sim(sid="HOT-ALL-63-T3"):
    scenario = SCENARIOS[sid]
    llm, tts = MagicMock(), MagicMock()
    tts.synthesize.return_value = SimpleNamespace(audio=b"\x00\x00" * 1600, sample_rate=16000)
    sim = UserSimulator(
        copy.deepcopy(scenario["user_script"]),
        llm_gateway=llm,
        tts_gateway=tts,
        leakage_label=scenario["leakage_level"],
        scenario_id=sid,
        instruction_entry=fake_instruction_entry(scenario["user_script"]["key_event"]["state"]),
    )
    return sim, llm


def reply(text, **overrides):
    return SimpleNamespace(
        text=json.dumps(
            {
                "text": text,
                "state": "urgent",
                "is_key_turn": False,
                "end_call": False,
                "covered": [],
                "interrupt": False,
                **overrides,
            },
            ensure_ascii=False,
        ),
        finish_reason="stop",
    )


def test_missing_address_and_availability_do_not_require_fabrication_to_complete_registration():
    sim, llm = make_sim()
    before = copy.deepcopy(sim.user_script)
    llm.chat.return_value = reply("地址和可用时间我需要再核实一下，先把漏水情况登记清楚。")
    sim.next_turn("请提供上门地址，并说一下今天下午或明天上午哪个时间方便。")
    system = llm.chat.call_args.args[0][0]["content"]
    assert "先把信息给完再道别" not in system
    for constraint in (
        "只能提供场景已明确给定的信息",
        "字段未提供或无法确认",
        "不得为配合登记编造具体值",
        "不得凭空声称系统已有",
    ):
        assert constraint in system
    assert sim.user_script == before and set(before["user_facts"]) == {"issue_detail"}
    assert "幸福小区" not in json.dumps(before, ensure_ascii=False)
    assert llm.chat.call_count == 1


@pytest.mark.parametrize("sid", ["HOT-ALL-63-T3", "COL-POS-26-T3", "MAR-POS-59-T3"])
def test_neutral_key_uses_given_surface_and_objective_but_keeps_intent_internal(sid):
    sim, _ = make_sim(sid)
    before = copy.deepcopy(sim.user_script)
    key = sim.key_event
    user = sim._build_messages(3, "请进一步确认这项业务。", forced=False)[1]["content"]
    block = user[user.index("【关键事件】表面业务话题：") :].split("【透明度约束】", 1)[0]
    assert f"表面业务话题：{key['surface']}" in block
    assert f"业务目标：{key['objective']}" in block
    assert "表面业务话题为唯一措辞依据" in block
    assert "优先于剧情要点、关键事件描述、身份和风格示范" in block
    assert "不要把它们说进text" in block and key["intent"] not in block
    assert f"发生关键事件的那一轮JSON state仍为{key['state']}" in block
    assert sim.user_script == before


def test_hot63_neutral_surface_does_not_expand_into_time_pressure():
    sim, _ = make_sim()
    prompt = sim._build_messages(4, "维修需要二十四小时。", forced=False)[1]["content"]
    block = prompt[prompt.index("【关键事件】表面业务话题：") :].split("【透明度约束】", 1)[0]
    assert "问ORD7205是哪天签收的" in block
    assert "确认今天能否上门与到场时间" in block
    assert "因时间被卡住，催促客服给出处理安排，强调等不起" not in block
    assert "不据此追加时间压力、态度结论、状态自述或施压要求" in block


def test_explicit_key_prompt_and_protocol_are_preserved():
    sim, _ = make_sim("HOT-ALL-63-T1")
    system, user = [m["content"] for m in sim._build_messages(3, "请进一步确认。", forced=False)]
    assert f"【关键事件】{sim.key_event['description']}（要求状态：{sim.key_event['state']}" in user
    assert "【中性关键轮文本边界】" not in user
    for field in ("text", "state", "is_key_turn", "end_call", "covered", "interrupt"):
        assert f'"{field}"' in system


def test_neutral_key_state_and_voice_control_are_preserved():
    sim, llm = make_sim()
    llm.chat.return_value = reply("这个订单是哪天签收的，您帮我核对一下。", is_key_turn=True)
    result = sim.next_turn("请核对一下签收时间。")
    assert result.is_key_turn and result.state == sim.key_event["state"] == "urgent"
    assert result.tts_control == sim._instruction_entry["key"]
    assert llm.chat.call_count == 1


def test_rejected_end_feedback_appears_next_turn_without_polluting_spoken_history():
    sim, llm = make_sim()
    sim._key_turn_no = 1
    llm.chat.return_value = reply("这个订单的漏水情况我已经说明了，您先核对。", end_call=True)
    assert not sim.next_turn("请再确认一下这项记录。").end_call
    assert sim._last_end_call_rejected
    assert "【结束请求反馈】" not in llm.chat.call_args.args[0][1]["content"]
    llm.chat.return_value = reply("好的，您先按刚才的信息核对，今天先到这里。", end_call=True)
    assert sim.next_turn("信息已经登记，稍后与您联系。").end_call
    prompt = llm.chat.call_args.args[0][1]["content"]
    assert "【结束请求反馈】上轮end_call未获准，通话仍在继续" in prompt
    assert "条件性投诉不代表已经收束" in prompt and "不要机械重复上一句" in prompt
    assert not sim._last_end_call_rejected
    assert all("结束请求反馈" not in item["content"] for item in sim._history)
    assert llm.chat.call_count == 2


def test_dry_run_rejection_cannot_modify_next_turn_feedback():
    sim, llm = make_sim()
    sim._key_turn_no = 1
    llm.chat.return_value = reply("这个订单的漏水情况我已经说明了，您先核对。", end_call=True)
    parsed = sim._elicit(2, "请确认这项记录。", forced=False, dry_run=True)
    assert parsed["_end_call_rejected"] and not sim._last_end_call_rejected
    assert "【结束请求反馈】" not in sim._build_messages(2, "请确认这项记录。", forced=False)[1]["content"]
    assert sim._history == []


def test_feedback_does_not_add_a_same_text_hard_ban_or_extra_retry():
    sim, llm = make_sim()
    sim._key_turn_no = 1
    text = "这个订单的漏水情况我已经说明了，您先核对。"
    llm.chat.side_effect = [reply(text, end_call=True), reply(text, end_call=False)]
    assert not sim.next_turn("先确认一下这项记录。").end_call
    assert sim.next_turn("刚才那项记录需要再核对。").text == text
    assert llm.chat.call_count == 2 and not sim._last_end_call_rejected
