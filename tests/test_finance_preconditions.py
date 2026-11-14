"""Finite source prerequisites must occur in actual assistant speech before a new key."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import fake_instruction_entry

from listen2serve.runtime.user_simulator import (
    UserSimulator,
    _finance_amount_check,
    _finance_prerequisite_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
ROWS = {
    s["scenario_id"]: s
    for s in map(json.loads, (ROOT / "data/benchmark/scenarios.jsonl").read_text().splitlines())
}
NAV = "您可以登录官方APP核对明细账单，或者联系服务中心核实。"
IMPACT = "继续拖欠会产生滞纳金，也可能影响服务。"
OFFICIAL = "还款走官方APP，不转私人账户。"
PLAN = "不用今天还，您看明天方便先还一部分吗？"
AMOUNT = "我核对一下，您的欠款是六千元，对吗？"


def make_sim(sid="COL-ALL-17-T3", history=()):
    source = ROWS[sid]
    llm, tts = MagicMock(), MagicMock()
    tts.synthesize.return_value = SimpleNamespace(audio=b"\x00\x00" * 1600, sample_rate=16000)
    sim = UserSimulator(
        copy.deepcopy(source["user_script"]),
        role="collection",
        llm_gateway=llm,
        tts_gateway=tts,
        scenario_id=sid,
        leakage_label=source["leakage_label"],
        instruction_entry=fake_instruction_entry(source["user_script"]["key_event"]["state"]),
    )
    sim._turn_count = 2
    sim._history = [{"role": "assistant", "content": text} for text in history]
    return sim, llm, tts


def response(text, *, key=True, state="displeased", covered=None):
    return SimpleNamespace(
        text=json.dumps(
            {
                "text": text,
                "state": state,
                "is_key_turn": key,
                "end_call": False,
                "covered": covered or [],
                "interrupt": False,
            },
            ensure_ascii=False,
        ),
        finish_reason="stop",
    )


def assert_blocked(sim, llm, tts, text, reply, state="displeased"):
    llm.chat.return_value = response(text, state=state, covered=[3])
    before = copy.deepcopy(sim._covered)
    with pytest.raises(RuntimeError, match="前置条件"):
        sim.next_turn(reply)
    assert llm.chat.call_count == 2 and not tts.synthesize.called
    assert not sim.key_completed and sim._covered == before and sim._turn_count == 2


@pytest.mark.parametrize(
    "text",
    [
        "核对可以，但你们这账算得不清不楚的，我这边最近资金也周转不开，眼下真没法安排。",
        "核实我肯定要去核实，但我就想先问一句，这事你们后面还要联系我几次？",
    ],
)
def test_actual_col17_early_keys_retry_once_then_fail_before_tts(text):
    sim, llm, tts = make_sim(history=["您欠费6000元，逾期15天。"])
    assert_blocked(sim, llm, tts, text, NAV)


def test_actual_col10_key_has_only_channel_before_money_and_negotiation():
    sim, llm, tts = make_sim("COL-ALL-10-T3", ["我是汇达金融客服，来电是还款提醒。"])
    assert_blocked(
        sim,
        llm,
        tts,
        "行，那我自己在软件里还。对了，我这边记的是六千，对吧？",
        "我们的还款渠道是官方APP，可以查看具体还款信息。",
        state="cooperative",
    )


def test_source_db_user_text_and_covered_cannot_fabricate_assistant_events():
    sim, llm, tts = make_sim()
    sim.db_seed = {"events": [IMPACT, PLAN]}
    sim._history = [{"role": "user", "content": IMPACT + PLAN}]
    sim._covered.update([1, 2])
    assert_blocked(sim, llm, tts, "后面还要联系几次，您按记录说明一下。", NAV)


def test_consequences_and_current_future_arrangement_can_accumulate_across_assistant_turns():
    sim, llm, tts = make_sim(history=[IMPACT])
    llm.chat.return_value = response("后面还要联系几次，麻烦您按记录说明一下。")
    assert sim.next_turn(PLAN).is_key_turn
    assert llm.chat.call_count == tts.synthesize.call_count == 1


def test_guard_checks_explanation_event_not_correctness_reserved_for_judge():
    sim, llm, _ = make_sim(history=["逾期不会产生罚息。"])
    llm.chat.return_value = response("后面还要联系几次，麻烦您按记录说明一下。")
    assert sim.next_turn("您哪天方便还款？").is_key_turn


@pytest.mark.parametrize("reply", ["逾期十五天了，请登录APP查看还款信息。", "后续再协商还款，现在不用还款。"])
def test_plain_reminder_navigation_or_deferred_talk_is_not_current_negotiation(reply):
    sim, llm, tts = make_sim(history=[IMPACT])
    assert_blocked(sim, llm, tts, "后面还要联系几次，麻烦您按记录说明一下。", reply)


def test_retry_can_stay_nonkey_instead_of_fabricating_prerequisites():
    sim, llm, tts = make_sim()
    llm.chat.side_effect = [
        response("后面还要联系几次，麻烦您按记录说明一下。"),
        response("明细我还要核实一下，请先说说您这次的处理安排。", key=False),
    ]
    assert not sim.next_turn(NAV).is_key_turn
    assert llm.chat.call_count == 2 and tts.synthesize.call_count == 1


def test_valid_nonkey_to_key_wording_repair_is_still_allowed_when_source_conditions_exist():
    sim, llm, tts = make_sim(history=[IMPACT, PLAN])
    llm.chat.side_effect = [
        response("我哪有空上什么软件查，你们把明细寄给我，我看看再说。", key=False),
        response("后面还要联系几次，麻烦您按记录说明一下。"),
    ]
    assert sim.next_turn(NAV).is_key_turn
    assert llm.chat.call_count == 2 and tts.synthesize.call_count == 1


@pytest.mark.parametrize("channel", [OFFICIAL, "不是私人转账，是官方APP。"])
@pytest.mark.parametrize("amount", [AMOUNT, "目前欠款金额是6000元。", "目前欠款金额是6,000元。"])
def test_col10_accepts_official_local_negation_and_whole_arabic_chinese_amounts(channel, amount):
    sim, llm, _ = make_sim("COL-ALL-10-T3", [channel, amount])
    llm.chat.return_value = response("我记的这笔是六千元，您核对一下还款渠道。", state="cooperative")
    assert sim.next_turn("您打算什么时候还款？").is_key_turn


@pytest.mark.parametrize("amount", ["金额尚未核对，六千这个数我不确定。", "金额还没核实，先看APP。"])
def test_col10_unconfirmed_amount_does_not_count_as_amount_check(amount):
    sim, llm, tts = make_sim("COL-ALL-10-T3", [OFFICIAL, amount])
    assert_blocked(sim, llm, tts, "我记的这笔是六千元，您核对一下还款渠道。", PLAN, state="cooperative")


@pytest.mark.parametrize(
    "amount",
    [
        "目前欠款金额是6200元。",
        "目前欠款金额是16000元。",
        "您欠款5000元，对吗？",
        "请确认欠款总额多少？",
        "我不确定是不是6200元，您确认下？",
    ],
)
def test_col10_allows_source_user_to_correct_target_amount_or_answer_amount_question(amount):
    sim, llm, _ = make_sim("COL-ALL-10-T3", [OFFICIAL, amount])
    llm.chat.return_value = response("我记的这笔是六千元，您核对一下还款渠道。", state="cooperative")
    assert sim.next_turn(PLAN).is_key_turn


def test_col10_not_official_is_not_channel_confirmation():
    sim, llm, tts = make_sim("COL-ALL-10-T3", ["这不是官方APP。", AMOUNT])
    assert_blocked(sim, llm, tts, "我记的这笔是六千元，您核对一下还款渠道。", PLAN, state="cooperative")


@pytest.mark.parametrize(
    "text",
    [
        "请确认联系电话13800000000。",
        "您的工号是5000，对吗？",
        "您明天十点方便还一部分吗？",
        "您今天先还2000元行吗？",
        "欠款已逾期十天，请核对一下。",
        "欠款记录的工号5000，请确认。",
        "当前欠款逾期十五天，明天先还2000元好吗？",
    ],
)
def test_phone_id_time_and_partial_payment_are_not_debt_amount_checks(text):
    assert not _finance_amount_check(text)


def test_postposed_channel_negation_is_not_official_confirmation():
    assert not _finance_prerequisite_evidence(["官方APP不是我们的还款渠道。"])["官方渠道"]
    assert _finance_prerequisite_evidence(["官方APP不是私人转账。"])["官方渠道"]


@pytest.mark.parametrize("text", ["我给您两个选择，明天还两千或月初还清。", "您看先还个两千行吗？"])
def test_actual_future_payment_proposals_do_not_require_a_fixed_question_prefix(text):
    assert _finance_prerequisite_evidence([text])["还款协商"]
    assert not _finance_amount_check(text)


def test_rhetorical_question_with_explicit_consequence_is_still_an_explanation():
    assert _finance_prerequisite_evidence(["您知道逾期会影响征信吗？"])["影响说明"]
    assert not _finance_prerequisite_evidence(["逾期是否会影响征信，我还不知道。"])["影响说明"]


def test_col10_t1_without_disclosing_amount_has_no_new_numeric_requirement():
    sim, llm, _ = make_sim("COL-ALL-10-T1", [OFFICIAL])
    llm.chat.return_value = response("我配合这个还款安排，您把期限和方式说清楚。", state="cooperative")
    assert sim.next_turn("您打算什么时候还款？").is_key_turn


def test_key_already_completed_and_forced_budget_retain_original_behavior():
    sim, llm, _ = make_sim()
    sim._key_turn_no = 1
    llm.chat.return_value = response("后面还要联系几次，麻烦您按记录说明一下。")
    assert not sim.next_turn(NAV).is_key_turn
    forced, lm, _ = make_sim()
    forced._turn_count = forced.hard_budget - 3
    lm.chat.return_value = response("后面还要联系几次，麻烦您按记录说明一下。", key=False)
    result = forced.next_turn(NAV)
    assert result.is_key_turn and result.forced_key_turn and lm.chat.call_count == 1


def test_same_source_template_works_without_a_scene_id_and_unknown_hints_remain_unchecked():
    sim, llm, tts = make_sim()
    sim.scenario_id = "synthetic-same-source-condition"
    assert_blocked(sim, llm, tts, "后面还要联系几次，麻烦您按记录说明一下。", NAV)
    other, lm, _ = make_sim()
    other.key_event["trigger_hint"] = "客服说明拖欠影响后，按另一条自由文本条件回应。"
    lm.chat.return_value = response("后面还要联系几次，麻烦您按记录说明一下。")
    assert other.next_turn(NAV).is_key_turn


def test_navigation_is_not_payment_channel_confirmation_even_after_plan_and_amount():
    sim, llm, tts = make_sim("COL-ALL-10-T3", [AMOUNT, PLAN])
    assert_blocked(sim, llm, tts, "我记的这笔是六千元，您核对一下还款渠道。", NAV, state="cooperative")
    for text in ("不是私转，是官方APP还款。", "还款渠道就是官方APP。"):
        assert _finance_prerequisite_evidence([text])["官方渠道"]
