"""Guard observed neutral-key leaks while preserving scripted closing refusals."""

import pytest

from listen2serve.data.wording_ban import explicit_key_state_declaration_hits
from listen2serve.vocab import (
    neutral_wording_hits,
    validate_level_runtime,
    validate_wording,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("我不接受。", ["我不接受"]),
        ("方案我不接受。", ["方案我不接受"]),
        ("这方案我不接受。", ["这方案我不接受"]),
        ("眼下这个安排我不同意，改天再谈吧。", ["眼下这个安排我不同意"]),
        ("我不同意这个安排。", ["我不同意这个安排"]),
        (
            "还一部分你们还要接着追，这方案我不接受。眼下这个安排我不同意，改天再谈吧。",
            ["这方案我不接受", "眼下这个安排我不同意"],
        ),
        (
            "二十四小时？我等不了那么久，机器漏水影响用水，你们今天必须给我个准信儿。",
            ["我等不了那么久"],
        ),
        ("我现在没法等那么长时间。", ["我现在没法等那么长时间"]),
    ],
)
def test_observed_standalone_declarations_are_identified(text, expected):
    assert explicit_key_state_declaration_hits(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "我不接受的话，还有什么办理方式？",
        "如果金额没核实，这个方案我不同意。",
        "要是这个方案我不接受，接下来怎么办？",
        "这个方案我不同意吗？我先核对一下。",
        "你刚说的是‘我不接受’吗？",
        "记录写着‘这个方案，我不同意。’，这个是谁说的？",
        '页面里写了"我等不了那么久"，这是什么意思？',
        "不是我不同意，金额我还需要核对。",
        "不是说我不接受，我先核对一下记录。",
        "如果周五能办，我可以等。",
        "还一部分之后，剩下的金额怎么处理？",
        "我记的是逾期二十天，你帮我核对一下。",
        "那就下周五之后联系，今天先这样。",
        "我不同意的那项费用具体是怎么算的？",
        "我等不了那么久吗？有没有别的处理方式？",
        "",
    ],
)
def test_quotes_questions_conditionals_and_business_text_are_exempt(text):
    assert explicit_key_state_declaration_hits(text) == []


@pytest.mark.parametrize("text", ["这方案我不接受。", "我不同意，改天再联系。", "我等不了那么久。"])
def test_only_neutral_key_turns_enable_the_new_guard(text):
    assert neutral_wording_hits(text) == []
    assert neutral_wording_hits(text, is_key_turn=True)
    assert validate_level_runtime(text, "T3", "displeased")[0]
    assert not validate_level_runtime(text, "T3", "displeased", is_key_turn=True)[0]
    assert validate_level_runtime(text, "T2", "displeased", is_key_turn=True)[0]


def test_prepared_neutral_key_contract_uses_the_same_declaration_guard():
    ok, reason = validate_wording("还款方案我不接受。", "T3", "displeased")
    assert not ok and "明确状态自述" in reason


def test_explicit_key_and_existing_cooperation_guard_keep_their_scope():
    assert validate_wording("我真是失望，我不接受。", "T1", "displeased")[0]
    assert neutral_wording_hits("我配合。") == ["我配合"]
