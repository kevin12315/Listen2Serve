"""Apply the existing source debt disclosure gate to each supported new key."""

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
ROWS = {
    s["scenario_id"]: s
    for s in map(json.loads, (ROOT / "data/benchmark/scenarios.jsonl").read_text().splitlines())
}
IMPACT = "逾期会产生滞纳金，也可能影响后续服务。"
PLAN = "可以先还一部分，剩余再协商。"
ACTUAL_A2_REPLY = "理解您的难处，郑女士。不过逾期会产生滞纳金，也可能影响后续的服务使用。您看能不能先安排一部分款项，比如今天或明天先缴一部分，缓解一下逾期情况？"
MONEY = "这笔欠款我记得是三万元，您核对一下。"


def make_sim(sid="COL-ALL-04-T3", history=()):
    row = ROWS[sid]
    llm, tts = MagicMock(), MagicMock()
    tts.synthesize.return_value = SimpleNamespace(audio=b"\x00\x00" * 1600, sample_rate=16000)
    sim = UserSimulator(
        copy.deepcopy(row["user_script"]),
        role="collection",
        llm_gateway=llm,
        tts_gateway=tts,
        scenario_id=sid,
        leakage_label=row["leakage_label"],
        instruction_entry=fake_instruction_entry(row["user_script"]["key_event"]["state"]),
    )
    sim._turn_count = 4
    sim._history = [{"role": "assistant", "content": s} for s in history]
    return sim, llm, tts


def response(text=MONEY, key=True):
    return SimpleNamespace(
        text=json.dumps(
            {
                "text": text,
                "state": "cooperative",
                "is_key_turn": key,
                "end_call": False,
                "covered": [3] if key else [],
                "interrupt": False,
            },
            ensure_ascii=False,
        ),
        finish_reason="stop",
    )


def test_actual_col04_a2_partial_offer_does_not_authorize_customer_to_state_total():
    sim, llm, tts = make_sim(history=["我是负责欠费提醒的客服。", "就是这笔欠款，您最近遇到什么困难了吗？"])
    llm.chat.return_value = response()
    before = copy.deepcopy(sim._covered)
    with pytest.raises(RuntimeError, match="客服核对该欠款金额"):
        sim.next_turn(ACTUAL_A2_REPLY)
    assert llm.chat.call_count == 2 and not tts.synthesize.called and not sim.key_completed
    assert sim._covered == before and sim._turn_count == 4


def test_repair_can_ask_for_total_without_disclosing_it_or_claiming_the_key():
    sim, llm, tts = make_sim()
    llm.chat.side_effect = [response(), response("欠款总额麻烦您先说明一下，我再核对。", key=False)]
    assert not sim.next_turn(ACTUAL_A2_REPLY).is_key_turn
    assert llm.chat.call_count == 2 and tts.synthesize.call_count == 1


@pytest.mark.parametrize(
    "assistant", ["欠款金额是两万元，对吗？", "您记得欠款总额是多少？", "我不确定是不是两万元，您确认一下？"]
)
def test_col04_may_correct_wrong_target_total_or_answer_total_question(assistant):
    sim, llm, tts = make_sim(history=[IMPACT, PLAN, assistant])
    llm.chat.return_value = response()
    assert sim.next_turn("这次安排您还想确认什么？").is_key_turn
    assert llm.chat.call_count == tts.synthesize.call_count == 1


def test_source_user_covered_and_later_agent_words_do_not_supply_prior_amount_event():
    sim, _, _ = make_sim(history=[IMPACT, PLAN])
    sim.db_seed = {"customers": [["example", 30000]]}
    sim._history.append({"role": "user", "content": "欠款是三万元。"})
    sim._covered.update([1, 2, 3])
    assert sim._finance_trigger_violations(
        {"text": MONEY, "is_key_turn": True}, "请登录APP查明细。", forced=False
    )


@pytest.mark.parametrize(
    "sid,text",
    [
        ("COL-ALL-17-T3", "这笔我记的是六千元，您核对一下。"),
        ("COL-ALL-10-T3", "这笔我记的是六千元，您核对一下。"),
    ],
)
def test_each_preexisting_supported_rule_uses_original_debt_disclosure(sid, text):
    sim, _, _ = make_sim(sid, history=[IMPACT, PLAN, "还款走官方APP。"])
    assert sim._finance_trigger_violations(
        {"text": text, "is_key_turn": True}, "请看还款信息。", forced=False
    )


def test_t1_no_amount_unknown_hint_forced_completed_and_nonkey_paths_stay_unchanged():
    sim, _, _ = make_sim("COL-ALL-04-T1", history=[IMPACT, PLAN])
    p = {"text": "我愿意配合，请您说清楚缴费渠道。", "is_key_turn": True}
    assert not sim._finance_trigger_violations(p, "请问还需确认什么？", forced=False)
    p = {"text": MONEY, "is_key_turn": True}
    assert not sim._finance_trigger_violations(p, "", forced=True)
    sim._key_turn_no = 2
    assert not sim._finance_trigger_violations(p, "", forced=False)
    sim._key_turn_no = None
    assert not sim._finance_trigger_violations(dict(p, is_key_turn=False), "", forced=False)
    sim.key_event["trigger_hint"] = "另一条自由文本条件。"
    assert not sim._finance_trigger_violations(p, "", forced=False)


@pytest.mark.parametrize("reply", ["今天能不能安排一下还款？", "请登录官方APP看账单。"])
def test_neg45_generic_request_or_navigation_is_not_a_flexible_offer(reply):
    sim, _, _ = make_sim("COL-NEG-45-T3")
    p = {"text": "本期金额和还款路径，麻烦您再说明一下。", "is_key_turn": True}
    assert sim._finance_trigger_violations(p, reply, forced=False)


@pytest.mark.parametrize(
    "plan", [PLAN, "可以帮您申请分期，审核通过后再确认。", "今天先还3000元，其余可以申请分期。"]
)
def test_neg45_offered_plan_needs_no_new_consequence_condition_or_target_correctness(plan):
    sim, llm, tts = make_sim("COL-NEG-45-T3")
    llm.chat.return_value = response("本期金额和还款路径，麻烦您再说明一下。")
    assert sim.next_turn(plan).is_key_turn
    assert llm.chat.call_count == tts.synthesize.call_count == 1


def test_neg45_quoting_debt_still_requires_real_amount_inquiry():
    sim, _, _ = make_sim("COL-NEG-45-T3", history=[PLAN])
    p = {"text": "这笔我记的是九千八百元，您核对一下。", "is_key_turn": True}
    assert sim._finance_trigger_violations(p, "请看缴费明细。", forced=False)
    assert not sim._finance_trigger_violations(p, "欠款九千元对吗？", forced=False)


@pytest.mark.parametrize("assistant", ["本期应还9800元。", "本期应还多少？", "本期金额是九千元，对吗？"])
def test_neg45_current_period_amount_event_allows_source_amount_or_correction(assistant):
    sim, llm, tts = make_sim("COL-NEG-45-T3", history=[PLAN])
    llm.chat.return_value = response("这笔我记的是九千八百元，您核对一下。")
    assert sim.next_turn(assistant).is_key_turn
    assert llm.chat.call_count == tts.synthesize.call_count == 1


@pytest.mark.parametrize(
    "assistant",
    [
        "首期应还6900元。",
        "本期先还6900元。",
        "本期应还请您看APP，工号9800。",
        "本期应还请您看APP，电话9800。",
    ],
)
def test_neg45_partial_payment_or_nonmoney_identifiers_are_not_period_amount_checks(assistant):
    sim, _, _ = make_sim("COL-NEG-45-T3", history=[PLAN])
    assert sim._finance_trigger_violations(
        {"text": "这笔我记的是九千八百元，您核对一下。", "is_key_turn": True},
        assistant,
        forced=False,
    )


@pytest.mark.parametrize("sid", ["COL-ALL-04-T3", "COL-ALL-17-T3", "COL-ALL-10-T3"])
def test_current_period_amount_projection_is_exclusive_to_the_new_source_template(sid):
    sim, _, _ = make_sim(sid, history=[IMPACT, PLAN, "还款走官方APP。"])
    amount = ROWS[sid]["user_script"]["user_facts"]["debt_amount"]
    assert sim._finance_trigger_violations(
        {"text": f"这笔欠款我记的是{amount}，您核对一下。", "is_key_turn": True},
        f"本期应还{amount}。",
        forced=False,
    )
