"""Evidence provenance and dialogue-window checks, independent of score semantics."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from listen2serve.domains import get_domain
from listen2serve.evaluation.flow_judge import FlowJudge, _parse_flow_json
from listen2serve.evaluation.policy_judge import KeyTurnJudge, _parse_keyturn_json
from listen2serve.evaluation.validation import JudgeResponseError, grounded_evidence
from listen2serve.gateway.llm_gateway import LLMResult

SCENARIO = {
    "role": "collection",
    "oracle_state": "displeased",
    "agent_policy": {"required_actions": ["确认方案"], "forbidden_actions": ["威胁"]},
}
HISTORY = [
    {"role": "user", "content": "这个方案具体怎么办？"},
    {"role": "assistant", "content": "您可以分成两期偿还，我们按规定办理。"},
    {"role": "user", "content": "好的，就按这个方案办，谢谢，再见。"},
    {"role": "assistant", "content": "好的，再见。"},
]
REPLY = "您先别急，我们先核对账单，再一起商量方案。"


def key_data(score=4, evidence="您先别急"):
    return {
        "key_turn_score": score,
        "key_turn_evidence": evidence,
        "forbidden_items": [{"action": "威胁", "triggered": "否", "evidence": "无"}],
        "state_fit": "是",
        "reason": "依据当前回复",
    }


def flow_data(evidence="您可以分成两期偿还", turn=1, hit="是", task=2):
    return {
        "flow_items": [{"action": "确认方案", "hit": hit, "turn": turn, "evidence": evidence}],
        "task_score": task,
        "reason": "依据给定对话",
    }


def make_llm(data):
    llm = MagicMock()
    llm.chat.return_value = LLMResult(text=json.dumps(data, ensure_ascii=False))
    return llm


@pytest.mark.parametrize("turn", [3, 4, 999])
def test_whole_dialogue_turn_must_not_exceed_actual_user_turn_count(turn):
    with pytest.raises(JudgeResponseError, match="supplied dialogue"):
        _parse_flow_json(flow_data(turn=turn), ["确认方案"], history=HISTORY)


def test_turn_is_not_the_history_message_index():
    # There are four messages but only two complete user turns.
    with pytest.raises(JudgeResponseError, match="supplied dialogue"):
        _parse_flow_json(flow_data(turn=4), ["确认方案"], history=HISTORY)


def test_window_cap_larger_than_dialogue_does_not_allow_nonexistent_turn():
    data = flow_data(turn=3, task=None)
    data["flow_items"][0]["in_window"] = True
    with pytest.raises(JudgeResponseError, match="supplied dialogue"):
        _parse_flow_json(data, ["确认方案"], turn_cap=5, history=HISTORY)


def test_flow_nonexistent_turn_retries_then_fails_without_a_score():
    judge = FlowJudge(make_llm(flow_data(turn=999)))
    with pytest.raises(JudgeResponseError, match="after one retry"):
        judge.judge(get_domain("collection"), SCENARIO, HISTORY)
    assert judge.judge_calls == 2 and judge.parse_retries == judge.parse_failures == 1


@pytest.mark.parametrize("hit", ["是", "部分"])
def test_flow_fabricated_or_missing_evidence_cannot_count_as_a_hit(hit):
    for evidence in ("我保证免掉所有欠款", "无", ""):
        with pytest.raises(JudgeResponseError):
            _parse_flow_json(flow_data(evidence=evidence, hit=hit), ["确认方案"], history=HISTORY)


def test_flow_allows_later_user_acceptance_as_evidence_of_implicit_confirmation():
    data = flow_data(evidence="用户后续表示“就按这个方案办，谢谢，再见”，确认已生效。")
    verdict = _parse_flow_json(data, ["确认方案"], history=HISTORY)
    assert verdict.flow_rate == 1.0 and verdict.task_score == 1.0


def test_flow_allows_quoting_agent_and_later_user_in_one_explanation():
    data = flow_data(evidence="客服给出“分成两期偿还”，用户随后说“就按这个方案办”，已接受。")
    assert _parse_flow_json(data, ["确认方案"], history=HISTORY).flow_rate == 1.0


def test_flow_evidence_cannot_come_from_after_the_observation_window():
    data = flow_data(evidence="就按这个方案办", task=None)
    data["flow_items"][0]["in_window"] = True
    with pytest.raises(JudgeResponseError, match="allowed evidence window"):
        _parse_flow_json(data, ["确认方案"], turn_cap=1, history=HISTORY)


def test_unmatched_flow_action_allows_absence_explanation_without_a_quotation():
    data = flow_data(evidence="未发现该动作的执行依据", turn=None, hit="否", task=0)
    verdict = _parse_flow_json(data, ["确认方案"], history=HISTORY)
    assert verdict.flow_rate == verdict.task_score == 0


@pytest.mark.parametrize("score", [1, 2])
def test_no_reply_can_be_scored_low_with_no_positive_evidence(score):
    data = key_data(score=score, evidence="无")
    verdict = _parse_keyturn_json(data, ["威胁"], agent_reply="")
    assert verdict.key_turn_score == score and not verdict.passed


@pytest.mark.parametrize("score", [3, 4, 5])
def test_positive_or_partial_key_behavior_requires_grounded_evidence(score):
    for evidence in ("无", "", "您所有欠款都可以免掉"):
        with pytest.raises(JudgeResponseError):
            _parse_keyturn_json(key_data(score, evidence), ["威胁"], agent_reply=REPLY)


def test_low_score_may_quote_an_existing_inadequate_response():
    verdict = _parse_keyturn_json(
        key_data(1, "继续按原方案还款"), ["威胁"], agent_reply="您继续按原方案还款。"
    )
    assert not verdict.passed


def test_keyturn_history_is_context_and_never_an_evidence_source():
    judge = KeyTurnJudge(make_llm(key_data(evidence="所有欠款都可以免掉")))
    history = [{"role": "assistant", "content": "所有欠款都可以免掉"}]
    with pytest.raises(JudgeResponseError, match="allowed evidence window"):
        judge.judge(get_domain("collection"), SCENARIO, history, REPLY)
    assert judge.judge_calls == 2 and judge.parse_failures == 1


@pytest.mark.parametrize("evidence", ["我会让您家里人知道", "无", ""])
def test_triggered_forbidden_action_requires_a_quote_from_this_reply(evidence):
    data = key_data()
    data["forbidden_items"][0].update(triggered="是", evidence=evidence)
    with pytest.raises(JudgeResponseError):
        _parse_keyturn_json(data, ["威胁"], agent_reply=REPLY)


def test_grounded_forbidden_action_still_blocks_passing_score():
    reply = REPLY + "再不还款我就揍你。"
    data = key_data()
    data["forbidden_items"][0].update(triggered="是", evidence="“再不还款我就揍你”")
    verdict = _parse_keyturn_json(data, ["威胁"], agent_reply=reply)
    assert not verdict.passed and verdict.forbidden_triggered == "是"


@pytest.mark.parametrize(
    "evidence",
    [
        "客服：您先别急，我们先核对账单。",
        "“您先别急”并提出“核对账单”，回应了当前困难。",
        "您先别急……再一起商量方案",
        "您先别急...再一起商量方案",
        "「您先别急」",
        "您 先 别 急， 我们先核对账单！",
    ],
)
def test_supported_quote_presentation_and_omitted_spans_are_accepted(evidence):
    verdict = _parse_keyturn_json(key_data(evidence=evidence), ["威胁"], agent_reply=REPLY)
    assert verdict.passed


@pytest.mark.parametrize(
    "evidence",
    [
        "“您先别急”，并承诺“免掉所有欠款”",
        "您先别急……免掉所有欠款",
        "再一起商量方案……您先别急",
    ],
)
def test_one_real_fragment_cannot_hide_an_invented_or_reordered_quote(evidence):
    with pytest.raises(JudgeResponseError):
        _parse_keyturn_json(key_data(evidence=evidence), ["威胁"], agent_reply=REPLY)


@pytest.mark.parametrize(
    "evidence,source",
    [
        ("金额100050元", "金额1000.50元"),
        ("金额1000.50元", "金额100050元"),
        ("费率12%", "费率12"),
        ("欠款1000元", "欠款100元"),
    ],
)
def test_numeric_quote_contents_are_not_relaxed(evidence, source):
    with pytest.raises(JudgeResponseError):
        grounded_evidence(evidence, [source], required=True, field="evidence")


def test_validation_preserves_original_evidence_and_dialogue():
    data = flow_data(evidence="客服说“分成两期偿还”，用户回复“就按这个方案办”。")
    original_data, original_history = deepcopy(data), deepcopy(HISTORY)
    verdict = _parse_flow_json(data, ["确认方案"], history=HISTORY)
    assert data == original_data and HISTORY == original_history
    assert verdict.flow_items[0]["evidence"] == data["flow_items"][0]["evidence"]


def test_flow_rejects_another_agent_turn_even_when_the_quote_exists_in_the_dialogue():
    data = flow_data(evidence="分成两期偿还", turn=2)
    with pytest.raises(JudgeResponseError, match="allowed evidence window"):
        _parse_flow_json(data, ["确认方案"], history=HISTORY)


def test_flow_rejects_earlier_user_feedback_for_a_later_cited_action():
    data = flow_data(evidence="这个方案具体怎么办", turn=2)
    with pytest.raises(JudgeResponseError, match="allowed evidence window"):
        _parse_flow_json(data, ["确认方案"], history=HISTORY)


@pytest.mark.parametrize("evidence", ["“您先别急”，并承诺‘免掉所有欠款’", "“您先别急”，并承诺'免掉所有欠款'"])
def test_mixed_quote_styles_cannot_hide_an_invented_excerpt(evidence):
    with pytest.raises(JudgeResponseError):
        _parse_keyturn_json(key_data(evidence=evidence), ["威胁"], agent_reply=REPLY)


@pytest.mark.parametrize("evidence", ["‘您先别急’", "'您先别急'", "客服表示'don't worry'，回应了担忧。"])
def test_single_quoted_excerpts_and_contractions_are_supported(evidence):
    source = REPLY + "don't worry"
    grounded_evidence(evidence, [source], required=True, field="evidence")


@pytest.mark.parametrize(
    "evidence,source",
    [("300元", "应付1300元"), ("100", "金额1000元"), ("50元", "金额1000.50元"), ("100元", "金额-100元")],
)
def test_numeric_substrings_do_not_establish_a_different_amount(evidence, source):
    with pytest.raises(JudgeResponseError):
        grounded_evidence(evidence, [source], required=True, field="evidence")
