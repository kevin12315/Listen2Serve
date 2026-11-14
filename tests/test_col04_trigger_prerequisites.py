"""The source's executable-plan event must precede a new non-forced key."""

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
IMPACT = "继续拖欠会产生滞纳金，也可能影响信用记录。"
AMOUNT_CHECK = "请您核对一下，欠款总额是多少？"
ACTUAL_REPLY = "郑女士，理解您店里忙走不开，我就长话短说。您这笔物业费30000元已经逾期7天了，上次承诺的还款日期是5月28日，但至今还没到账。继续拖欠的话，可能会产生滞纳金，甚至影响您的信用记录。您看今天能不能安排一下还款？"
ACTUAL_USER = (
    "我记的也是三万块。这样吧，我下个月十号前后先还九千，剩下的月底回款了就补上。你们这儿怎么交，多久到账？"
)
KEY = "这笔欠款我记得是三万元，您核对一下。"


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
    sim._turn_count = 3
    sim._history = [{"role": "assistant", "content": s} for s in history]
    return sim, llm, tts


def response(text=KEY, key=True):
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


@pytest.mark.parametrize("text", [KEY, ACTUAL_USER])
def test_actual_generic_pressure_retries_once_then_stops_before_tts(text):
    sim, llm, tts = make_sim()
    llm.chat.return_value = response(text)
    before = copy.deepcopy(sim._covered)
    with pytest.raises(RuntimeError, match="前置条件"):
        sim.next_turn(ACTUAL_REPLY)
    assert llm.chat.call_count == 2 and not tts.synthesize.called
    assert not sim.key_completed and sim._turn_count == 3 and sim._covered == before


@pytest.mark.parametrize(
    "reply",
    [
        "今天能不能安排一下还款？",
        "麻烦尽快处理这笔欠费。",
        "您什么时候方便还款？",
        "后续我们再讨论分期方案。",
        "目前不能申请分期，今天先不谈还款方案。",
        "您可以登录官方APP查看缴费明细。",
        "我只是核对，欠款金额为三万元。",
        "您上次还3000元已经到账了吗？",
        "还一部分是不允许的。",
        "先还3000元这个方案不可以。",
    ],
)
def test_generic_request_navigation_or_unoffered_scheme_is_not_an_executable_plan(reply):
    sim, _, _ = make_sim(history=[IMPACT, AMOUNT_CHECK])
    parsed = {"text": KEY, "is_key_turn": True}
    assert sim._finance_trigger_violations(parsed, reply, forced=False)


@pytest.mark.parametrize(
    "plan",
    [
        "可以先还一部分，剩余再协商。",
        "今天先还3000元，其他款项再商量。",
        "我先帮您提交分期申请，审批结果出来再确认。",
        "您可以一次性全额缴费，也可以申请分期。",
        "您先通过官方APP缴纳九千元，期限可再协商。",
        "这笔欠款是两万元，可先还5000元，余款申请分期。",
        "请通过对公渠道先交个两千，剩余部分再讨论。",
        "分期可以申请，但要以审核结果为准。",
        "您可以先缴纳部分，之后再确认剩余安排。",
        "如果暂时一次付不清，可以先按能承受的金额缴纳一部分。",
        "今天通过官方APP付全额。",
        "如果不能全额就先还一部分。",
        "您这次能先还3000元吗？",
    ],
)
def test_source_accepts_a_concrete_payment_option_without_grading_target_correctness(plan):
    sim, llm, tts = make_sim(history=[IMPACT, AMOUNT_CHECK])
    llm.chat.return_value = response()
    assert sim.next_turn(plan).is_key_turn
    assert llm.chat.call_count == 1 and tts.synthesize.call_count == 1


def test_plan_without_explained_consequences_is_not_enough():
    sim, _, _ = make_sim(history=[AMOUNT_CHECK])
    parsed = {"text": KEY, "is_key_turn": True}
    assert sim._finance_trigger_violations(parsed, "可以先还3000元，余款帮您申请分期。", forced=False)


def test_model_can_repair_to_nonkey_without_inventing_an_assistant_offer():
    sim, llm, tts = make_sim()
    llm.chat.side_effect = [response(), response("您先说说有哪些处理选项，我再核实一下。", key=False)]
    assert not sim.next_turn(ACTUAL_REPLY).is_key_turn
    assert llm.chat.call_count == 2 and tts.synthesize.call_count == 1 and not sim.key_completed


def test_assistant_evidence_can_span_turns_but_source_and_user_words_are_not_evidence():
    sim, _, _ = make_sim(history=[IMPACT, AMOUNT_CHECK, "可以先还3000元，余款申请分期。"])
    assert not sim._finance_trigger_violations(
        {"text": KEY, "is_key_turn": True}, "您想了解哪一项？", forced=False
    )
    sim._history = [{"role": "user", "content": IMPACT + AMOUNT_CHECK + "可以先还3000元，余款申请分期。"}]
    sim.db_seed = {"events": [IMPACT, AMOUNT_CHECK, "可以先还3000元，余款申请分期。"]}
    assert sim._finance_trigger_violations({"text": KEY, "is_key_turn": True}, "请核对明细。", forced=False)


def test_t1_has_same_after_condition_while_forced_completed_and_unknown_hints_keep_original_path():
    sim, _, _ = make_sim("COL-ALL-04-T1")
    p = {"text": "我愿意配合这个方案，您说清缴费渠道。", "is_key_turn": True}
    assert sim._finance_trigger_violations(p, ACTUAL_REPLY, forced=False)
    assert not sim._finance_trigger_violations(p, ACTUAL_REPLY, forced=True)
    sim._key_turn_no = 1
    assert not sim._finance_trigger_violations({"text": KEY, "is_key_turn": True}, ACTUAL_REPLY, forced=False)


def test_same_source_hint_works_without_a_runtime_scene_id():
    sim, _, _ = make_sim()
    sim.scenario_id = "synthetic-same-source-hint"
    assert sim._finance_trigger_violations({"text": KEY, "is_key_turn": True}, ACTUAL_REPLY, forced=False)
    sim._key_turn_no = None
    sim.key_event["trigger_hint"] = "另一条自由文本触发条件。"
    assert not sim._finance_trigger_violations({"text": KEY, "is_key_turn": True}, ACTUAL_REPLY, forced=False)
