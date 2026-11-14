"""Offline verdict contracts, scoring definitions, retries and backend routing."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from listen2serve.domains import get_domain
from listen2serve.evaluation.flow_judge import (
    FlowJudge,
    _parse_flow_json,
    _render_history_turn_numbered,
    flow_checklist,
)
from listen2serve.evaluation.policy_judge import KeyTurnJudge, _parse_keyturn_json
from listen2serve.evaluation.validation import JudgeResponseError
from listen2serve.evaluation.voice_judge import VoiceJudge
from listen2serve.gateway.llm_gateway import LLMResult

SCENARIO = {
    "scenario_id": "sample",
    "role": "collection",
    "oracle_state": "displeased",
    "agent_policy": {
        "role_description": "催收客服",
        "required_actions": ["确认身份", "促成还款承诺"],
        "forbidden_actions": ["威胁辱骂", "持续高压"],
    },
    "required_actions_effective": ["确认身份", "促成还款承诺"],
}


KEY_REPLY = "您先别急，我们可以慢慢商量。"
FLOW_HISTORY = [
    {"role": "user", "content": "喂，什么事？"},
    {"role": "assistant", "content": "您好，请问是本人吗？"},
    {"role": "user", "content": "对，是我。"},
    {"role": "assistant", "content": "您看周五还款可以吗？"},
    {"role": "user", "content": "我考虑一下。"},
    {"role": "assistant", "content": "好的，您先考虑。"},
]


def key_data(score=4, evidence="先别急"):
    return {
        "key_turn_score": score,
        "key_turn_evidence": evidence,
        "forbidden_items": [
            {"action": action, "triggered": "否", "evidence": "无"}
            for action in SCENARIO["agent_policy"]["forbidden_actions"]
        ],
        "state_fit": "是",
        "reason": "合理",
    }


def flow_data(task=1):
    return {
        "flow_items": [
            {"action": "确认身份", "hit": "是", "turn": 1, "evidence": "请问是本人吗"},
            {"action": "促成还款承诺", "hit": "部分", "turn": 3, "evidence": "考虑一下"},
        ],
        "task_score": task,
        "reason": "部分推进",
    }


def make_llm(*outputs):
    llm = MagicMock()
    llm.chat.side_effect = [
        LLMResult(text=json.dumps(out, ensure_ascii=False) if isinstance(out, dict) else out)
        for out in outputs
    ]
    return llm


@pytest.mark.parametrize("score,passed", [(1, False), (2, False), (3, False), (4, True), (5, True)])
def test_keyturn_threshold_and_diagnostic_state_fit(score, passed):
    data = key_data(score)
    data["state_fit"] = "过度"
    verdict = _parse_keyturn_json(data, SCENARIO["agent_policy"]["forbidden_actions"], agent_reply=KEY_REPLY)
    assert verdict.passed is passed
    assert verdict.key_turn_score == score


def test_forbidden_action_blocks_otherwise_passing_score():
    data = key_data(5)
    data["forbidden_items"][0]["triggered"] = "是"
    data["forbidden_items"][0]["evidence"] = "再不还我就揍你"
    assert not _parse_keyturn_json(data, agent_reply=KEY_REPLY + "再不还我就揍你").passed


@pytest.mark.parametrize("score", [0, 6, 99, "4", 4.5, True, None])
def test_keyturn_rejects_invalid_score(score):
    with pytest.raises(JudgeResponseError):
        _parse_keyturn_json(key_data(score), agent_reply=KEY_REPLY)


@pytest.mark.parametrize(
    "field", ["key_turn_score", "key_turn_evidence", "forbidden_items", "state_fit", "reason"]
)
def test_keyturn_requires_complete_schema(field):
    data = key_data()
    del data[field]
    with pytest.raises(JudgeResponseError):
        _parse_keyturn_json(data, agent_reply=KEY_REPLY)


@pytest.mark.parametrize(
    "items",
    [
        [],
        ["bad"],
        [{"action": "unknown", "triggered": "否", "evidence": "无"}],
        [{"action": "威胁辱骂", "triggered": "否", "evidence": "无"}] * 2,
    ],
)
def test_keyturn_requires_exact_forbidden_action_coverage(items):
    data = key_data()
    data["forbidden_items"] = items
    llm = make_llm(data, data)
    judge = KeyTurnJudge(llm, "google/gemini-3.8-flash")
    with pytest.raises(JudgeResponseError, match="after one retry"):
        judge.judge(get_domain("collection"), SCENARIO, [], "先别急")
    assert llm.chat.call_count == 2
    assert judge.parse_retries == judge.parse_failures == 1


def test_schema_failure_retries_same_request_then_returns_valid_verdict():
    llm = make_llm({"key_turn_score": 4}, key_data())
    judge = KeyTurnJudge(llm, "google/gemini-3.8-flash")
    verdict = judge.judge(get_domain("collection"), SCENARIO, [], "先别急")
    assert verdict.passed and verdict.parse_retried
    assert llm.chat.call_args_list[0] == llm.chat.call_args_list[1]
    assert judge.parse_failures == 0


def test_keyturn_prompt_keeps_history_as_context_and_declares_no_audio():
    llm = make_llm(key_data(evidence="长话短说"))
    KeyTurnJudge(llm).judge(
        get_domain("collection"), SCENARIO, [{"role": "user", "content": "我不方便"}], "长话短说"
    )
    prompt = llm.chat.call_args.args[0][1]["content"]
    assert "用户: 我不方便" in prompt
    assert "没有音频" in prompt and "仅上下文，非证据" in prompt


def test_flow_formula_and_exact_checklist_source():
    verdict = _parse_flow_json(flow_data(), flow_checklist(SCENARIO), history=FLOW_HISTORY)
    assert verdict.flow_rate == 0.75
    assert verdict.task_score_raw == 1 and verdict.task_score == 0.5
    scenario = deepcopy(SCENARIO)
    scenario["required_actions_effective"] = ["只检查此项"]
    assert flow_checklist(scenario) == ["只检查此项"]


@pytest.mark.parametrize(
    "mutation", ["missing", "reordered", "rewritten", "extra", "bad_hit", "bad_turn", "missing_evidence"]
)
def test_flow_rejects_corrupted_checklist(mutation):
    data = flow_data()
    if mutation == "missing":
        data["flow_items"].pop()
    elif mutation == "reordered":
        data["flow_items"].reverse()
    elif mutation == "rewritten":
        data["flow_items"][0]["action"] = "重写动作"
    elif mutation == "extra":
        data["flow_items"].append(data["flow_items"][0])
    elif mutation == "bad_hit":
        data["flow_items"][0]["hit"] = "true"
    elif mutation == "bad_turn":
        data["flow_items"][0]["turn"] = "1"
    else:
        del data["flow_items"][0]["evidence"]
    llm = make_llm(data, data)
    with pytest.raises(JudgeResponseError):
        FlowJudge(llm).judge(get_domain("collection"), SCENARIO, FLOW_HISTORY)
    assert llm.chat.call_count == 2


@pytest.mark.parametrize("score", [-1, 3, 99, "2", 0.5, True, None])
def test_flow_rejects_invalid_task_score(score):
    with pytest.raises(JudgeResponseError):
        _parse_flow_json(flow_data(score), flow_checklist(SCENARIO), history=FLOW_HISTORY)


def test_windowed_flow_keeps_same_formula_and_does_not_score_terminal_outcome():
    data = flow_data(None)
    for item in data["flow_items"]:
        item["in_window"] = False
    verdict = _parse_flow_json(data, flow_checklist(SCENARIO), turn_cap=3, history=FLOW_HISTORY)
    assert verdict.flow_rate == verdict.flow_rate_in_window == 0.75
    assert verdict.task_score is None and verdict.n_items_in_window == 2
    data["task_score"] = 2
    with pytest.raises(JudgeResponseError):
        _parse_flow_json(data, flow_checklist(SCENARIO), turn_cap=3, history=FLOW_HISTORY)


def test_window_requires_boolean_in_window():
    data = flow_data(None)
    with pytest.raises(JudgeResponseError):
        _parse_flow_json(data, flow_checklist(SCENARIO), turn_cap=3, history=FLOW_HISTORY)


def test_turn_window_keeps_complete_turn_pairs():
    history = [
        {"role": role, "content": f"{role}{turn}"} for turn in range(1, 4) for role in ("user", "assistant")
    ]
    text = _render_history_turn_numbered(history, 2)
    assert "assistant2" in text and "user3" not in text


def voice_data():
    return {"role_voice_match": 4, "state_voice_fit": 4, "naturalness": 4, "reason": "自然"}


@pytest.mark.parametrize(
    "model", ["openai/gpt-4o-audio-preview", "openai/custom-omni", "google/gemini-3.8-flash"]
)
def test_explicit_compatible_platform_never_routes_by_audio_name(tmp_path, model):
    wav = tmp_path / "reply.wav"
    wav.write_bytes(b"wav")
    llm = make_llm(voice_data())
    judge = VoiceJudge(llm, model)
    judge._judge_dashscope_native = MagicMock(side_effect=AssertionError("wrong backend"))
    verdict = judge.judge(wav, "客服", "displeased", "您好")
    assert verdict.passed
    assert llm.chat.call_args.kwargs["model"] == model


def test_dashscope_voice_uses_native_backend_and_same_backend_retry(tmp_path):
    judge = VoiceJudge(MagicMock(), "dashscope/qwen3.5-omni-plus")
    judge._judge_dashscope_native = MagicMock(
        side_effect=[
            ("{}", "dashscope/qwen3.5-omni-plus"),
            (json.dumps(voice_data()), "dashscope/qwen3.5-omni-plus"),
        ]
    )
    verdict = judge.judge(tmp_path / "a.wav", "客服", "displeased", "您好")
    assert verdict.passed and judge.parse_retries == 1
    assert judge._judge_dashscope_native.call_args_list[0] == judge._judge_dashscope_native.call_args_list[1]
    judge.llm.chat.assert_not_called()


@pytest.mark.parametrize("bad", [{}, {"naturalness": 99}, {"naturalness": True}, {"role_voice_match": "4"}])
def test_invalid_voice_verdict_raises_after_one_retry(tmp_path, bad):
    data = voice_data()
    data.update(bad)
    if not bad:
        del data["state_voice_fit"]
    llm = make_llm(data, data)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"wav")
    judge = VoiceJudge(llm, "openai/audio-model")
    with pytest.raises(JudgeResponseError):
        judge.judge(wav, "客服", "displeased", "您好")
    assert llm.chat.call_count == 2
    assert judge.parse_retries == judge.parse_failures == 1


@pytest.mark.parametrize(
    "scores,passed", [((4, 4, 4), True), ((3, 4, 4), True), ((3, 3, 4), False), ((2, 5, 5), False)]
)
def test_voice_pass_formula(tmp_path, scores, passed):
    data = dict(zip(("role_voice_match", "state_voice_fit", "naturalness"), scores), reason="评价")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"wav")
    verdict = VoiceJudge(make_llm(data), "openai/audio-model").judge(wav, "客服", "displeased", "您好")
    assert verdict.passed is passed
