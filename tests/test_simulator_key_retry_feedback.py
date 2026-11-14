"""Neutral key retries use the source surface without prescribing a final utterance."""

from __future__ import annotations

import pytest
from test_simulator_prompt_boundaries import make_sim, reply


@pytest.mark.parametrize(
    "scenario,bad,good,state",
    [
        (
            "COL-POS-26-T3",
            "这方案我不接受，眼下这个安排我不同意，改天再谈吧。",
            "我记的这笔已经逾期二十天了，您看一下记录。",
            "displeased",
        ),
        (
            "HOT-ALL-63-T3",
            "二十四小时？我等不了那么久，今天必须给我个准信。",
            "这个订单是哪天签收的，您帮我核对一下。",
            "urgent",
        ),
    ],
)
def test_neutral_key_retry_removes_stance_and_restores_source_surface(scenario, bad, good, state):
    sim, llm = make_sim(scenario)
    llm.chat.side_effect = [
        reply(bad, state=state, is_key_turn=True),
        reply(good, state=state, is_key_turn=True),
    ]
    result = sim.next_turn("请进一步核对这项业务记录。")
    assert llm.chat.call_count == 2 and sim.tts.synthesize.call_count == 1
    assert result.text == good and result.state == state and result.is_key_turn
    note = llm.chat.call_args_list[1].args[0][-1]["content"]
    assert "【中性关键轮重写】" in note
    assert sim.key_event["surface"] in note
    assert "删除状态和立场自述、态度结论" in note
    assert "自然承接客服上文，不要求照抄固定句子" in note
    assert "真实态度保留在JSON state与声音中" in note
    assert "本轮禁止任何明确情绪词" not in note
    assert sim.stats["retries"] == 1 and not result.vocab_violation


def test_non_key_authorized_refusal_closing_does_not_acquire_key_only_retry():
    sim, llm = make_sim("COL-POS-26-T3")
    sim._key_turn_no = 1
    llm.chat.return_value = reply("您改天再联系我，这项安排我现在不接。", state="displeased", end_call=True)
    result = sim.next_turn("这项安排您可以稍后再考虑。")
    assert result.end_call and llm.chat.call_count == 1
    assert sim.stats["retries"] == 0


def test_explicit_condition_non_key_violation_keeps_existing_feedback():
    sim, llm = make_sim("HOT-ALL-63-T1")
    llm.chat.side_effect = [
        reply("我很着急，这件事情什么时候能处理呢。"),
        reply("这个漏水情况具体怎么处理，麻烦您说一下。"),
    ]
    result = sim.next_turn("请先说明一下设备漏水情况。")
    assert result.text == "这个漏水情况具体怎么处理，麻烦您说一下。"
    assert llm.chat.call_count == 2
    note = llm.chat.call_args_list[1].args[0][-1]["content"]
    assert "本轮禁止任何明确情绪词" in note
    assert "【中性关键轮重写】" not in note


def test_repeated_key_declaration_keeps_existing_soft_flag_and_single_retry_limit():
    sim, llm = make_sim("COL-POS-26-T3")
    llm.chat.return_value = reply(
        "这个安排我不接受，您改天再跟我联系吧。", state="displeased", is_key_turn=True
    )
    result = sim.next_turn("请您考虑一下本次还款安排。")
    assert llm.chat.call_count == 2
    assert sim.stats["retries"] == 1 and result.vocab_violation
