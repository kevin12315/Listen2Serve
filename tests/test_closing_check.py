"""收尾轮确定性检查（closing_check）与其聚合口径的测试。"""

from __future__ import annotations

from listen2serve.evaluation.closing_check import check_closing
from listen2serve.report.aggregate import metric_block


def _traj(user_text: str, agent_text: str) -> dict:
    return {"turns": [
        {"turn": 1, "user_text": "你好", "agent_text": "您好，请问是本人吗？"},
        {"turn": 2, "user_text": user_text, "agent_text": agent_text},
    ]}


def _traj_end_call(user_text: str, agent_text: str, end_call: bool) -> dict:
    t = _traj(user_text, agent_text)
    t["turns"][-1]["end_call"] = end_call
    return t


def test_end_call_flag_is_authoritative_over_wording():
    """分母取模拟器 end_call 真值：用户没说「再见」也算收尾轮（词表漏检的那一半）。

    用例取自 planP 预飞真实轨迹 COL-ALL-04-T3 末轮：用户确认完就收尾，无任何道别词。
    """
    c = check_closing(_traj_end_call(
        "对，APP上转九千，十号前到账。", "好的，约定下月 10 号前通过 APP 还 9000 元，请您按时缴纳。", True))
    assert c["closing_signal"] is True
    assert c["closing_dangling"] is False
    assert c["closing_farewell"] is False   # 复述到位但缺道别
    assert c["closing_ok"] is False


def test_end_call_false_overrides_incidental_closing_words():
    """反向也认 end_call：用户嘴上「就这样」但并未收尾 → 不进分母，不误判客服。"""
    c = check_closing(_traj_end_call("先就这样试试，那分期怎么申请？", "您需要在 APP 提交申请单。", False))
    assert c["closing_signal"] is False
    assert c["closing_ok"] is None


def test_lexical_fallback_only_when_field_absent():
    """早期 run 无 end_call 字段时才回落词表（同一末句，有/无字段结论不同）。"""
    user, agent = "没别的事就先这样了啊，再见。", "好的，祝您顺利，再见！"
    assert check_closing(_traj(user, agent))["closing_signal"] is True
    assert check_closing(_traj_end_call(user, agent, False))["closing_signal"] is False


def test_no_closing_signal_leaves_judgement_undefined():
    """末轮用户没有词面收尾信号时不判：三项为 None，不进任何分母。"""
    c = check_closing(_traj("我下周五还一千八。", "好的，那您通过 APP 还款就行。"))
    assert c["closing_signal"] is False
    assert c["closing_dangling"] is None
    assert c["closing_farewell"] is None
    assert c["closing_ok"] is None


def test_dangling_close_when_agent_hands_turn_back():
    """用户已明说收尾，客服仍抛回问句 → 悬空收尾（实测最高频的缺陷形态）。"""
    for agent in (
        "好的，我复述一下：下月 10 号前交 9000 元，您确认对吗？",
        "那您计划什么时候还这 15000 元呢？",
        "您说下个月 10 号左右还 9000 元，是通过 APP 缴费吗",
    ):
        c = check_closing(_traj("没别的事就先这样了啊，再见。", agent))
        assert c["closing_signal"] is True
        assert c["closing_dangling"] is True, agent
        assert c["closing_ok"] is False


def test_proper_closing_turn_passes():
    """复述 + 道别、不再提问 → 合格收尾轮。"""
    c = check_closing(_traj("嗯好，那就麻烦你了，没别的事我先挂了啊。", "好的王先生，祝您顺利还款，再见！"))
    assert (c["closing_signal"], c["closing_dangling"], c["closing_farewell"], c["closing_ok"]) == (
        True, False, True, True,
    )


def test_statement_without_farewell_is_not_ok_but_not_dangling():
    """既不提问也不道别：不算悬空，但也不算合格收尾（两项须分别可读）。"""
    c = check_closing(_traj("那先这样吧。", "已记录您的还款承诺，以系统记录为准。"))
    assert c["closing_dangling"] is False
    assert c["closing_farewell"] is False
    assert c["closing_ok"] is False


def test_metric_block_excludes_unsignalled_calls_from_denominator():
    """聚合口径：无收尾信号的场景不进 dangling/ok 分母，signal_rate 独立报告覆盖率。"""
    items = [
        {"closing_signal": True, "closing_dangling": True, "closing_ok": False},
        {"closing_signal": True, "closing_dangling": False, "closing_ok": True},
        {"closing_signal": False, "closing_dangling": None, "closing_ok": None},
        {"closing_signal": False, "closing_dangling": None, "closing_ok": None},
    ]
    m = metric_block(items)
    assert m["closing_signal_rate"] == 0.5
    assert m["closing_dangling_rate"] == 0.5  # 分母 2，而非 4
    assert m["closing_ok_rate"] == 0.5


def test_metric_block_returns_none_for_legacy_verdicts():
    """旧 verdict 无 closing_* 键时返回 None（渲染 N/A），不假性记 0。"""
    m = metric_block([{"key_turn_score": 4}])
    assert m["closing_signal_rate"] is None
    assert m["closing_dangling_rate"] is None
    assert m["closing_ok_rate"] is None
