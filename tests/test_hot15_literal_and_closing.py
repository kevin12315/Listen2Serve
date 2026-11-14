"""Literal business language and callback acknowledgments from the final HOT call."""

from __future__ import annotations

import pytest
from test_simulator_prompt_boundaries import make_sim, reply

from listen2serve.runtime.user_simulator import _has_termination_cue
from listen2serve.vocab import neutral_wording_hits, runtime_emotion_hits


@pytest.mark.parametrize(
    "text",
    [
        "这卡没用几次就出故障，先帮我核对处理记录。",
        "还没用几次就出现问题，按记录核对一下。",
        "请您把这单登记紧急处理。",
        "刚才说的紧急工单，请按这个处理名称登记。",
    ],
)
def test_literal_usage_count_and_named_business_handling_are_runtime_exempt(text):
    assert not runtime_emotion_hits(text)
    assert not neutral_wording_hits(text)


@pytest.mark.parametrize(
    "text,word",
    [
        ("这卡没用，我要换一张。", "没用"),
        ("道歉没用，解释也没用。", "没用"),
        ("这卡没用几次就坏了，道歉也没用。", "没用"),
        ("我很急，先登记紧急处理。", "急"),
        ("我着急，别只说紧急工单。", "急"),
    ],
)
def test_phrase_exemption_does_not_hide_a_separate_emotional_occurrence(text, word):
    assert word in runtime_emotion_hits(text)
    assert word in neutral_wording_hits(text)


@pytest.mark.parametrize(
    "text",
    [
        "行，那我等专员电话，三天内没结果我肯定再来电升级，你们抓紧。",
        "行，那就这么定了，我等专员电话，三天内没结果我肯定升级。",
        "好的，那我等你们回电，先按刚才定下的安排办。",
        "好，那我就等短信通知，按刚才确认的处理安排来。",
        "行，我等你们的消息。",
        "好的，我们等待平台通知。",
    ],
)
def test_callback_or_settled_acknowledgment_can_close_after_key_without_goodbye(text):
    sim, llm = make_sim("HOT-ALL-15-T3")
    sim._key_turn_no = 3
    sim._turn_count = 5
    llm.chat.return_value = reply(text, state="doubtful", end_call=True)
    result = sim.next_turn("已登记处理安排，专员会按约定回电。")
    assert result.end_call and sim.termination_reason == "user_closed"
    assert llm.chat.call_count == 1
    assert sim.next_turn("再见") is None


@pytest.mark.parametrize(
    "text",
    [
        "我等专员电话，具体哪天回电您告诉我吧。",
        "那就这么定了，具体还要登记什么呢？",
        "月底发工资就转，您先帮我把渠道和期限说一下吧。",
        "钱到我就按这个渠道转，先帮我登记一下。",
        "行，那我把话放这儿，三天内没结果我肯定再来电升级，你们最好抓紧点。",
        "我等短信通知，具体什么时候发您告诉我吧。",
        "我等你们的消息，请先把登记编号给我吧。",
        "我不等短信通知，先帮我登记一下。",
        "不是我等短信通知，是您先确认处理结果。",
    ],
)
def test_pending_requests_and_ultimatum_alone_are_not_new_closing_cues(text):
    assert not _has_termination_cue(text)


def test_callback_closing_cue_cannot_bypass_existing_key_gate():
    sim, llm = make_sim("HOT-ALL-15-T3")
    llm.chat.return_value = reply(
        "行，那我等专员电话，就按已经说好的安排来。", state="doubtful", end_call=True
    )
    result = sim.next_turn("请先确认一下本次来电事项。")
    assert not result.end_call and not sim.key_completed and not sim._ended
