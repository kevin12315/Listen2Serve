"""Reject only clear unsupported address assertions before speech synthesis."""

from __future__ import annotations

import pytest
from test_simulator_prompt_boundaries import make_sim, reply


@pytest.mark.parametrize(
    "text",
    [
        "地址是幸福小区三栋二单元，今天下午或者明天上午都行。",
        "地址和电话你们系统里应该有吧，我下单时填过。机器都影响正常用水了，光登记不解决可不行。",
        "地址你们系统里应该有吧？我下单时填过。",
        "我的住址你们系统里已有记录，这项内容不用再问。",
    ],
)
def test_hot_missing_address_rejects_fabricated_value_or_filled_history(text):
    sim, _ = make_sim()
    assert sim._fact_violations(text)


@pytest.mark.parametrize(
    "text",
    [
        "地址我需要再核实一下，稍后按安全方式补充。",
        "地址需要补充，您能说一下如何提供这些信息吗？",
        "你们能不能查到地址？我这边需要先确认一下。",
        "地址你们系统里有没有，能帮我查一下吗？",
        "地址你们系统里应该有吧？我需要核实之后再补充。",
        "地址你们系统里已有记录？我需要先确认一下。",
        "我的地址是在幸福小区吗？这个我需要核实。",
        "你们既然打给我，应该知道我的电话，地址还需要核实。",
        "电话下单时我填过了，地址我需要再核实一下。",
    ],
)
def test_requests_unknown_fallback_and_phone_only_history_are_not_address_claims(text):
    sim, _ = make_sim()
    assert not sim._fact_violations(text)


def test_standalone_outbound_callee_phone_claim_is_outside_guard():
    sim, _ = make_sim()
    assert not sim._fact_violations("你们既然给我打电话，号码你们肯定已经有了。")


def test_explicitly_supplied_address_is_allowed():
    sim, _ = make_sim()
    sim.user_script["user_facts"]["address"] = "幸福小区三栋二单元"
    assert not sim._fact_violations("地址是幸福小区三栋二单元，您先按这个记录。")


def test_explicitly_labelled_address_in_identity_is_allowed():
    sim, _ = make_sim()
    sim.user_script["identity"] += "，地址是幸福小区三栋二单元"
    assert not sim._fact_violations("地址是幸福小区三栋二单元，您先按这个记录。")


def test_unknown_address_field_does_not_authorize_a_value():
    sim, _ = make_sim()
    sim.user_script["user_facts"]["address"] = "待核实"
    assert sim._fact_violations("地址是幸福小区三栋二单元，您先按这个记录。")


def test_existing_single_feedback_retry_accepts_unknown_fallback_before_tts():
    sim, llm = make_sim()
    llm.chat.side_effect = [
        reply("地址和电话你们系统里应该有吧，我下单时填过。"),
        reply("地址和联系电话我需要先核实一下，稍后再补充。"),
    ]
    result = sim.next_turn("请确认您的地址和联系电话。")
    assert result.text == "地址和联系电话我需要先核实一下，稍后再补充。"
    assert llm.chat.call_count == 2 and sim.tts.synthesize.call_count == 1
    assert sim.stats["retries"] == 1 and sim.stats["fact_violations"] == 1


def test_two_bad_address_assertions_fail_without_tts_or_turn_progress():
    sim, llm = make_sim()
    llm.chat.return_value = reply("地址是幸福小区三栋二单元，您按这个先登记。")
    with pytest.raises(RuntimeError, match="硬违规"):
        sim.next_turn("请确认您的地址。")
    assert llm.chat.call_count == 2 and not sim.tts.synthesize.called
    assert sim._turn_count == 0 and sim._history == []
