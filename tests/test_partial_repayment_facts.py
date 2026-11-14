"""Do not invent a balance-clearing commitment from a partial repayment plan."""

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
    for s in (json.loads(line) for line in (ROOT / "data/benchmark/scenarios.jsonl").read_text().splitlines())
}
ACTUAL_KEY = (
    "我记的也是三万块。这样吧，我下个月十号前后先还九千，剩下的月底回款了就补上。你们这儿怎么交，多久到账？"
)
ACTUAL_NEXT = "那就定下个月十号，先还九千，走你们那个对公入口。剩下的月底补齐，你帮我登记上报吧。"
SAFE = "我记的也是三万元，下个月十号前后先还九千，剩下的还要另作安排。"
AGENT = "这笔欠款三万元，继续逾期会产生滞纳金，可以先申请部分还款；您先说本期金额和时间。"


def make_sim(sid="COL-ALL-04-T3", *, prosody_arm="state"):
    row = copy.deepcopy(ROWS[sid])
    llm, tts = MagicMock(), MagicMock()
    tts.synthesize.return_value = SimpleNamespace(audio=b"\x00\x00" * 1600, sample_rate=16000)
    sim = UserSimulator(
        row["user_script"],
        llm_gateway=llm,
        tts_gateway=tts,
        role="collection",
        db_seed=row["db_seed"],
        leakage_label="T3",
        instruction_entry=fake_instruction_entry(row["user_script"]["key_event"]["state"]),
        prosody_arm=prosody_arm,
    )
    return sim, llm, tts


def response(text, *, key=True):
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


@pytest.mark.parametrize(
    "text",
    [
        ACTUAL_KEY,
        ACTUAL_NEXT,
        ACTUAL_KEY.replace("补上。", "补上，"),
        "我下个月十号先还九千，剩余月底全部还清，用哪个渠道？",
        "我下个月十号先还九千，剩余月底全部还清，多久到账？",
        "我月底把全部欠款一次性结清，你帮我登记。",
        "整笔我月底就补齐，您帮我登记一下。",
        "剩余的钱下周全部还清，您把渠道说一下。",
    ],
)
def test_unsupported_balance_completion_is_rejected(text):
    sim, _, _ = make_sim()
    assert sim._fact_violations(text, AGENT)


@pytest.mark.parametrize(
    "text",
    [
        SAFE,
        "下个月十号前后先还九千，剩下的怎么安排？",
        "剩下的月底就要补齐吗？我还需要核实一下。",
        "我不能承诺月底把剩余的补齐，先登记九千。",
        "您说剩下的月底补齐，我还没有确认这个安排。",
        "您刚说“剩下的月底补齐”，我还需要核实。",
        "不是三万一次还清，我先按九千这个数安排。",
        "这九千元我下个月十号前后一次还清。",
        "这部分钱我下个月十号前后全部还清。",
        "首期我下个月十号前后全部还清。",
        "这笔九千块我会全部还完，您按这个登记。",
        "剩下的月底补齐，行不行？我需要先问清楚。",
        "剩下的月底补齐，能不能这样安排？",
        "你让我月底把剩下的补齐，我还需要核实。",
        "如果折扣合适再考虑一次性结清，先说具体方案。",
        "我希望以后还清这笔欠款，现在先登记九千。",
    ],
)
def test_queries_negation_quotes_corrections_partial_payment_and_wishes_pass(text):
    sim, _, _ = make_sim()
    assert not sim._fact_violations(text, AGENT)


def test_target_repeating_the_invention_does_not_authorize_it():
    sim, _, _ = make_sim()
    sim._history = [{"role": "assistant", "content": "您说月底补齐剩余款项，我来记录。"}]
    assert sim._fact_violations(ACTUAL_NEXT, "您刚才说月底把剩余补齐，对吗？")


def test_income_timing_and_generic_goal_do_not_authorize_extra_completion():
    sim, _, _ = make_sim()
    assert "月底" in sim.user_script["user_facts"]["income_note"]
    assert "想还清" in sim.user_script["user_goal"]
    assert sim._fact_violations("剩下的月底回款后全部还清。", AGENT)


def test_source_wish_inside_facts_or_mission_is_not_a_balance_schedule():
    sim, _, _ = make_sim()
    sim.user_script["user_facts"]["repayment_goal"] = "想把剩余欠款早点还清"
    sim.mission.append("想把剩下的欠款都还清")
    assert sim._fact_violations("剩余月底回款后补齐，您帮我登记。", AGENT)


def test_original_full_payment_plan_is_unchanged():
    sim, _, _ = make_sim()
    sim.user_script["user_facts"]["planned_amount"] = 30000
    assert not sim._fact_violations("我下个月十号前后把三万元全部还清。", AGENT)


def test_explicit_source_fact_balance_plan_is_preserved():
    sim, _, _ = make_sim()
    sim.user_script["user_facts"]["remaining_plan"] = "先还九千元，剩余月底回款后补齐。"
    assert not sim._fact_violations("剩余月底回款后补齐，您先登记九千。", AGENT)


def test_existing_pos68_source_mission_authorizes_friday_balance_payment():
    sim, _, _ = make_sim("COL-POS-68-T3")
    assert not sim._fact_violations("我今天先交一千五，剩下的周五发工资后补齐。", AGENT)


def test_existing_pos65_can_wait_for_receipts_without_inventing_full_payment_date():
    sim, _, _ = make_sim("COL-POS-65-T3")
    assert not sim._fact_violations("我先凑一部分，剩余部分要等晚点回款再安排。", AGENT)
    assert sim._fact_violations("我先还六千，剩下的明天全部结清。", AGENT)


@pytest.mark.parametrize(
    ("sid", "text"),
    [
        ("COL-POS-27-T3", "要是折扣合适，我就一次性处理，您先把数字说清。"),
        ("COL-POS-26-T3", "结清后还会追讨剩余款项吗？这个需要说清楚。"),
    ],
)
def test_existing_discount_negotiation_and_balance_query_are_preserved(sid, text):
    sim, _, _ = make_sim(sid)
    assert not sim._fact_violations(text, AGENT)


def test_bad_then_safe_uses_original_single_feedback_and_only_synthesizes_safe():
    sim, llm, tts = make_sim()
    llm.chat.side_effect = [response(ACTUAL_KEY), response(SAFE)]
    result = sim.next_turn(AGENT)
    assert result.text == SAFE and result.is_key_turn
    assert llm.chat.call_count == 2 and tts.synthesize.call_count == 1
    assert SAFE in str(tts.synthesize.call_args) and ACTUAL_KEY not in str(tts.synthesize.call_args)
    assert sim.stats["retries"] == 1 and sim.stats["fact_violations"] == 1


def test_repeated_invention_stops_before_tts_key_or_mission_progress():
    sim, llm, tts = make_sim()
    llm.chat.return_value = response(ACTUAL_KEY)
    with pytest.raises(RuntimeError, match="硬违规"):
        sim.next_turn(AGENT)
    assert llm.chat.call_count == 2 and not tts.synthesize.called
    assert sim._turn_count == 0 and not sim.key_completed and not sim._covered


def test_neutral_audio_arm_keeps_original_latent_state_and_source_objects():
    before = copy.deepcopy(ROWS["COL-ALL-04-T3"])
    sim, llm, tts = make_sim(prosody_arm="neutral")
    llm.chat.side_effect = [response(ACTUAL_NEXT), response(SAFE)]
    result = sim.next_turn(AGENT)
    assert result.state == "cooperative" and result.is_key_turn
    assert sim._assemble_instruct(result.state, True) == sim._instruction_entry["non_key"]["neutral"]
    assert tts.synthesize.call_count == 1 and ROWS["COL-ALL-04-T3"] == before


def test_fact_consistency_does_not_change_forced_key_flags_or_repeat_key_handling():
    sim, llm, _ = make_sim()
    llm.chat.return_value = response(SAFE, key=False)
    parsed = sim._elicit(1, AGENT, forced=True)
    assert parsed["is_key_turn"] and parsed["_forced"] and parsed["state"] == "cooperative"
    sim._key_turn_no = 1
    llm.chat.return_value = response(SAFE)
    assert not sim._elicit(2, AGENT, forced=False)["is_key_turn"]
