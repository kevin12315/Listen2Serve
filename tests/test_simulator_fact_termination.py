"""Bound known fact contradictions and require grounded simulator termination."""

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
ROWS = [json.loads(line) for line in (ROOT / "data/benchmark/scenarios.jsonl").read_text().splitlines()]


def make_sim(script=None):
    script = copy.deepcopy(
        script or next(row["user_script"] for row in ROWS if row["scenario_id"] == "COL-ALL-10-T3")
    )
    llm, tts = MagicMock(), MagicMock()
    tts.synthesize.return_value = SimpleNamespace(audio=b"\x00\x00" * 1600, sample_rate=16000)
    sim = UserSimulator(
        script,
        llm_gateway=llm,
        tts_gateway=tts,
        leakage_label="T3",
        instruction_entry=fake_instruction_entry(script["key_event"]["state"]),
    )
    return sim, llm, tts


def response(text, **extra):
    return SimpleNamespace(
        text=json.dumps(
            {
                "text": text,
                "state": "cooperative",
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


def test_fact_priority_is_explicit_without_mutating_col10_script():
    script = next(row["user_script"] for row in ROWS if row["scenario_id"] == "COL-ALL-10-T3")
    before = copy.deepcopy(script)
    sim, _, _ = make_sim(script)
    prompt = sim._build_messages(1, "请问您目前的还款来源和资金安排如何？", forced=False)[1]["content"]
    assert "工资刚到账，这个月能周转开" in prompt
    assert "等发了工资、月底那两天" in prompt
    assert "说话风格只影响表达" in prompt and "不得照搬风格" in prompt
    assert script == before and sim.user_script == before


def test_salary_contradiction_uses_existing_one_retry_then_accepts_factful_text():
    sim, llm, tts = make_sim()
    llm.chat.side_effect = [
        response("我这边还得等月底发工资，钱到账了再给您转。"),
        response("工资已经到账了，我会按您说的官方渠道转。"),
    ]
    result = sim.next_turn("请问您现在资金周转情况如何？")
    assert result.text == "工资已经到账了，我会按您说的官方渠道转。"
    assert llm.chat.call_count == 2 and tts.synthesize.call_count == 1
    assert sim.stats["fact_violations"] == 1 and sim.stats["retries"] == 1


def test_repeated_known_contradiction_fails_before_tts_or_state_progress():
    sim, llm, tts = make_sim()
    llm.chat.return_value = response("我这边还得等月底发工资，钱到账了再给您转。", is_key_turn=True)
    with pytest.raises(RuntimeError, match="硬违规"):
        sim.next_turn("请按官方渠道处理，您资金目前到账了吗？")
    assert llm.chat.call_count == 2 and not tts.synthesize.called
    assert sim._turn_count == 0 and not sim.key_completed


@pytest.mark.parametrize(
    "text", ["我的工资还没有到账，等钱来了才可以安排。", "我还得等发工资，到时候才能给您转过去。"]
)
def test_explicit_received_income_contradictions_are_rejected(text):
    sim, _, _ = make_sim()
    assert sim._fact_violations(text)


@pytest.mark.parametrize(
    "text",
    [
        "不用等发工资，我已经收到钱了，按原计划转。",
        "您说的是等到月底发工资吗？我想核对一下。",
        "我同事还得等发工资，我这边已经到账了。",
    ],
)
def test_salary_questions_negation_and_other_people_are_not_inferred_as_conflicts(text):
    sim, _, _ = make_sim()
    assert not sim._fact_violations(text)


def test_month_start_plan_rejects_explicit_month_end_commitment_only_when_unambiguous():
    sim, _, _ = make_sim()
    assert sim._fact_violations("我月底把这六千元转进去，您帮我登记好。")
    assert not sim._fact_violations("我月底前会处理，具体还是按月初那天安排。")
    assert not sim._fact_violations("如果月底才转进去，会不会影响原来的安排？")
    sim.user_script["user_facts"]["income_note"] = "这次月初转一部分，月底另外一笔收入再作安排"
    assert not sim._fact_violations("我月底把剩余部分转进去，您帮我登记好。")


def test_other_income_and_non_month_start_plans_are_not_reinterpreted():
    sim, _, _ = make_sim()
    sim.user_script["user_facts"] = {"income_note": "要等发工资", "planned_date": "月底"}
    assert not sim._fact_violations("我还得等月底发工资，到时候再按计划还。")


def test_end_call_without_closing_cue_is_rejected_without_additional_llm_call():
    sim, llm, _ = make_sim()
    sim._key_turn_no = 2
    llm.chat.return_value = response("工资已经到账了，我会按这个渠道转进去。", end_call=True)
    result = sim.next_turn("您还款之后我再核对登记。")
    assert not result.end_call and not sim._ended
    assert sim.stats["end_call_cue_rejected"] == 1
    assert llm.chat.call_count == 1


@pytest.mark.parametrize(
    "text",
    [
        "您把具体的还款方式和期限说一下吧。",
        "您说让我直接挂电话吗？我还没问清楚呢。",
        "您先不要挂电话，渠道我还需要确认一下。",
    ],
)
def test_fake_covered_terminal_beat_does_not_close_open_questions(text):
    script = {
        "identity": "电话用户",
        "mission": ["说明来意", "完成关键事项", "直接挂断电话"],
        "key_event": {"description": "核对业务信息", "state": "cooperative"},
    }
    sim, llm, _ = make_sim(script)
    sim._key_turn_no = 1
    llm.chat.return_value = response(text, end_call=True, covered=[3])
    assert not sim.next_turn("您还有什么需要核对？").end_call
    assert not sim._ended


def test_same_early_key_turn_cannot_fake_scripted_abrupt_completion():
    script = {
        "identity": "电话用户",
        "mission": ["说明来意", "完成关键事项", "直接挂断电话"],
        "key_event": {"description": "核对业务信息", "state": "cooperative"},
    }
    sim, llm, _ = make_sim(script)
    llm.chat.return_value = response(
        "业务内容我刚听到了，价格我还需要核对。", is_key_turn=True, end_call=True, covered=[1, 2, 3]
    )
    result = sim.next_turn("您先听一下业务内容。")
    assert result.is_key_turn and not result.end_call and not sim._ended


def test_explicit_scripted_abrupt_hangup_after_key_is_still_allowed():
    script = {
        "identity": "电话用户",
        "mission": ["说明来意", "完成关键事项", "直接挂断电话"],
        "key_event": {"description": "核对业务信息", "state": "cooperative"},
    }
    sim, llm, _ = make_sim(script)
    sim._key_turn_no = 1
    llm.chat.return_value = response("嗯，业务价格我已经听到了，这边还有事情。", end_call=True, covered=[3])
    assert sim.next_turn("这是业务价格的说明。").end_call
    assert sim._ended and sim.termination_reason == "user_closed"


def test_arbitrary_or_conditional_covered_beat_does_not_create_silent_hangup_permission():
    for beat in ("核对业务价格和期限", "若对方继续施压就直接挂断电话"):
        script = {
            "identity": "电话用户",
            "mission": ["完成关键事项", beat],
            "key_event": {"description": "核对业务信息", "state": "cooperative"},
        }
        sim, llm, _ = make_sim(script)
        sim._key_turn_no = 1
        llm.chat.return_value = response("业务内容我已经听到了，我这边还在看。", end_call=True, covered=[2])
        assert not sim.next_turn("这是业务内容的说明。").end_call


@pytest.mark.parametrize(
    "text",
    [
        "好的，那今天就先这样，谢谢您，再见。",
        "您改天再联系我吧，今天就先到这里。",
        "这项安排我知道了，请您以后不要再打过来。",
    ],
)
def test_genuine_farewell_rescheduling_and_stop_contact_cues_can_end(text):
    sim, llm, _ = make_sim()
    sim._key_turn_no = 2
    llm.chat.return_value = response(text, end_call=True)
    assert sim.next_turn("好的，这项信息已核对。").end_call
    assert sim.next_turn("再见") is None


def test_neutral_cooperative_self_description_uses_existing_retry_budget():
    sim, llm, _ = make_sim()
    sim._history = [{"role": "assistant", "content": "欠款金额六千元，您通过官方APP还款，明天方便还吗？"}]
    llm.chat.side_effect = [
        response("六千元我已经记清楚了，这边我配合。", is_key_turn=True),
        response("六千元这笔我记清楚了，按这个渠道转。", is_key_turn=True),
    ]
    result = sim.next_turn("请您按官方渠道操作。")
    assert result.text == "六千元这笔我记清楚了，按这个渠道转。"
    assert llm.chat.call_count == 2


def test_three_proposed_scenarios_keep_source_facts_and_original_budget():
    for scenario_id in ("HOT-ALL-63-T3", "COL-POS-26-T3", "MAR-POS-59-T3"):
        row = next(s for s in ROWS if s["scenario_id"] == scenario_id)
        before = copy.deepcopy(row)
        sim, _, _ = make_sim(row["user_script"])
        assert sim.user_script == before["user_script"]
        assert sim.hard_budget == before["user_script"]["turn_budget"]["hard"]


def test_other_source_facts_can_authorize_waiting_for_a_future_payday():
    sim, _, _ = make_sim()
    sim.user_script["user_facts"]["planned_date"] = "等下次发工资再转"
    assert not sim._fact_violations("我还得等下次发工资，到时候再给您转过去。")


@pytest.mark.parametrize(
    "text,allowed",
    [
        ("先这样，具体渠道你告诉我吧。", False),
        ("请您把这些信息登记一下，谢谢，再见。", True),
    ],
)
def test_closing_cue_cannot_hide_a_following_open_request(text, allowed):
    sim, llm, _ = make_sim()
    sim._key_turn_no = 2
    llm.chat.return_value = response(text, end_call=True)
    assert sim.next_turn("您这边还有其他事项吗？").end_call is allowed


def test_direct_negation_of_waiting_does_not_contradict_received_income():
    sim, _, _ = make_sim()
    assert not sim._fact_violations("我不等发工资了，这笔现在就可以转。")
    assert not sim._fact_violations("我没等发工资，这笔已经按计划转了。")


def test_resolved_question_word_does_not_block_a_scripted_abrupt_ending():
    script = {
        "identity": "电话用户",
        "mission": ["完成关键事项", "直接挂断电话"],
        "key_event": {"description": "核对业务信息", "state": "cooperative"},
    }
    sim, llm, _ = make_sim(script)
    sim._key_turn_no = 1
    llm.chat.return_value = response("这个渠道怎么操作我已经知道了。", end_call=True, covered=[2])
    assert sim.next_turn("操作方式已经跟您说明了。").end_call


@pytest.mark.parametrize(
    "text",
    [
        "我月底还有工作，转账仍然按原计划办。",
        "我月底还是要工作，那几天没空接电话。",
        "我月底转岗，收入与还款安排没有变化。",
        "我月底交接，这笔还是按原计划处理。",
    ],
)
def test_month_end_non_repayment_verbs_do_not_create_date_conflicts(text):
    sim, _, _ = make_sim()
    assert not sim._fact_violations(text)


@pytest.mark.parametrize(
    "text",
    [
        "我月底还款",
        "我月底还钱",
        "我月底还清",
        "我月底还上",
        "我月底还六千元",
        "我月底再还",
        "我月底把六千元转进去",
        "我月底转账",
        "我月底转款",
        "我月底转过去",
        "我月底支付这笔钱",
        "我月底交钱",
        "我月底缴费",
    ],
)
def test_explicit_month_end_repayment_actions_remain_date_conflicts(text):
    sim, _, _ = make_sim()
    assert sim._fact_violations(text)
