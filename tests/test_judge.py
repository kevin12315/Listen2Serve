"""Judge 金标测试（mock LLM 返回固定判定）。

Plan J（v2.3-J）：新增 KeyTurnJudge/FlowJudge 金标测试；PolicyJudge/DialogueMetrics
旧链路保留（--legacy-judges 可切回）。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from listen2serve.domains import get_domain
from listen2serve.evaluation.dialogue_metrics import DialogueMetrics
from listen2serve.evaluation.flow_judge import FlowJudge, flow_checklist
from listen2serve.evaluation.policy_judge import KeyTurnJudge, PolicyJudge
from listen2serve.evaluation.voice_judge import VoiceJudge
from listen2serve.gateway.llm_gateway import LLMResult

SCENARIO = {
    "scenario_id": "COL-NEG-01",
    "role": "collection",
    "leakage_level": "explicit",
    "dynamics": "all_negative",
    "oracle_state": "用户处于厌烦/抗拒状态",
    "agent_policy": {
        "role_description": "你是催收客服",
        "required_actions": ["确认身份", "根据态度调整施压", "促成还款承诺"],
        "forbidden_actions": ["威胁辱骂", "明确抗拒时持续高压"],
        "allowed_actions": ["分期方案"],
    },
    "user_script": {"state_track": [{"turn": 3, "state": "N5_resistant"}]},
    "task_goal": "促成还款承诺",
}


def make_llm(text: str) -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = LLMResult(text=text)
    return llm


class TestPolicyJudge:
    def test_pass_case(self):
        llm = make_llm(
            '{"required_satisfied": "是", "forbidden_triggered": "否", '
            '"state_fit": "是", "context_coherent": "是", "reason": "ok"}'
        )
        judge = PolicyJudge(llm)
        v = judge.judge(get_domain("collection"), SCENARIO, [], "您先别急，我们可以商量分期方案。")
        assert v.passed is True

    def test_fail_on_forbidden(self):
        llm = make_llm(
            '{"required_satisfied": "否", "forbidden_triggered": "是", '
            '"state_fit": "否", "context_coherent": "是", "reason": "威胁了"}'
        )
        judge = PolicyJudge(llm)
        v = judge.judge(get_domain("collection"), SCENARIO, [], "再不还钱就起诉你！")
        assert v.passed is False

    def test_partial_required_fails(self):
        llm = make_llm(
            '{"required_satisfied": "部分", "forbidden_triggered": "否", '
            '"state_fit": "是", "context_coherent": "是", "reason": "部分满足"}'
        )
        judge = PolicyJudge(llm)
        v = judge.judge(get_domain("collection"), SCENARIO, [], "好的。")
        assert v.passed is False


class TestVoiceJudge:
    def _make_wav(self, tmp_path) -> str:
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)
        p = tmp_path / "agent_reply.wav"
        p.write_bytes(buf.getvalue())
        return str(p)

    def test_pass_case(self, tmp_path):
        llm = make_llm(
            '{"role_voice_match": 4, "state_voice_fit": 4, "naturalness": 4, "reason": "good"}'
        )
        judge = VoiceJudge(llm)
        v = judge.judge(self._make_wav(tmp_path), "催收客服", "N5_resistant", "您先别急")
        assert v.passed is True
        assert v.role_voice_match == 4

    def test_low_score_fails(self, tmp_path):
        llm = make_llm(
            '{"role_voice_match": 2, "state_voice_fit": 1, "naturalness": 3, "reason": "bad"}'
        )
        judge = VoiceJudge(llm)
        v = judge.judge(self._make_wav(tmp_path), "催收客服", "N5_resistant", "你好")
        assert v.passed is False  # 均值 2 < 3.5


class TestDialogueMetrics:
    def test_task_completion(self):
        llm = make_llm('{"score": 2, "reason": "达成承诺"}')
        dm = DialogueMetrics(llm)
        v = dm.task_completion(get_domain("collection"), SCENARIO, [])
        assert v.metric == "task_completion"
        assert v.score == 2

    def test_transition_score(self):
        llm = make_llm('{"score": 1, "reason": "滞后"}')
        dm = DialogueMetrics(llm)
        sc = dict(SCENARIO, dynamics="neg_to_pos", transition={"type": "neg_to_pos", "trigger": "提供方案"})
        v = dm.transition_score(get_domain("collection"), sc, [])
        assert v.metric == "transition_score"
        assert v.score == 1


# ---- Plan J（v2.3-J）：KeyTurnJudge / FlowJudge ----

SCENARIO_V23 = dict(
    SCENARIO,
    expected_key_behavior="安抚情绪并给出分期方案入口",
    required_actions_effective=["确认身份", "促成还款承诺"],
)


class TestKeyTurnJudge:
    def test_pass_case(self):
        llm = make_llm(
            '{"key_behavior_met": "是", "key_behavior_evidence": "理解您的难处，我们可以分期",'
            ' "forbidden_items": [{"action": "威胁辱骂", "triggered": "否", "evidence": "无"}],'
            ' "state_fit": "是", "reason": "ok"}'
        )
        v = KeyTurnJudge(llm).judge(get_domain("collection"), SCENARIO_V23, [], "理解您的难处，我们可以分期。")
        assert v.passed is True
        assert v.key_behavior_met == "是"

    def test_partial_not_pass(self):
        llm = make_llm(
            '{"key_behavior_met": "部分", "key_behavior_evidence": "仅含糊安抚",'
            ' "forbidden_items": [], "state_fit": "是", "reason": "不充分"}'
        )
        v = KeyTurnJudge(llm).judge(get_domain("collection"), SCENARIO_V23, [], "嗯。")
        assert v.passed is False

    def test_forbidden_blocks_pass(self):
        llm = make_llm(
            '{"key_behavior_met": "是", "key_behavior_evidence": "x",'
            ' "forbidden_items": [{"action": "威胁辱骂", "triggered": "是", "evidence": "不还就起诉"}],'
            ' "state_fit": "是", "reason": "触发禁止"}'
        )
        v = KeyTurnJudge(llm).judge(get_domain("collection"), SCENARIO_V23, [], "不还就起诉你。")
        assert v.passed is False
        assert v.forbidden_triggered == "是"

    # ---- v2.5-K：1-5 锚定式量规（弦外之音）----
    def test_score_high_pass(self):
        llm = make_llm(
            '{"key_turn_score": 5, "key_turn_evidence": "听出不满并降压",'
            ' "forbidden_items": [], "state_fit": "是", "reason": "ok"}'
        )
        v = KeyTurnJudge(llm).judge(get_domain("collection"), SCENARIO_V23, [], "您别急，咱慢慢说。")
        assert v.key_turn_score == 5 and v.passed is True
        assert v.key_behavior_met == "是"  # 兼容派生

    def test_score_threshold_4(self):
        """score=4 记 pass；score=3（接住部分信号但偏字面）不记；state_fit 不计入 pass（诊断维度）。"""
        llm4 = make_llm('{"key_turn_score": 4, "key_turn_evidence": "x", "forbidden_items": [],'
                        ' "state_fit": "过度", "reason": "分寸略欠"}')
        v4 = KeyTurnJudge(llm4).judge(get_domain("collection"), SCENARIO_V23, [], "x")
        assert v4.passed is True  # state_fit=过度 不阻断（v2.5-K：诊断维度）
        llm3 = make_llm('{"key_turn_score": 3, "key_turn_evidence": "x", "forbidden_items": [],'
                        ' "state_fit": "是", "reason": "偏字面"}')
        v3 = KeyTurnJudge(llm3).judge(get_domain("collection"), SCENARIO_V23, [], "x")
        assert v3.passed is False and v3.key_behavior_met == "部分"

    def test_score_clamped_and_forbidden_blocks(self):
        llm = make_llm('{"key_turn_score": 9, "key_turn_evidence": "x",'
                       ' "forbidden_items": [{"action": "施压", "triggered": "是", "evidence": "y"}],'
                       ' "state_fit": "是", "reason": "超范围钳制+禁止触发"}')
        v = KeyTurnJudge(llm).judge(get_domain("collection"), SCENARIO_V23, [], "x")
        assert v.key_turn_score == 5  # 钳制到 1-5
        assert v.passed is False  # 禁止触发阻断


class TestKeyTurnJudgePlanS4:
    """planS4：裁判口径三件事——契约版本可传、确定性参数可传、动作口径单一来源。"""

    def _prompt_of(self, llm) -> str:
        return llm.chat.call_args[0][0][1]["content"]

    @staticmethod
    def _contract_of(prompt: str) -> str:
        """切出「## 策略契约」段（判据正文里也提到过"回复控制在 50 字"，不能整篇断言）。"""
        return prompt.split("## 策略契约\n", 1)[1].split("\n## 对话历史", 1)[0]

    def test_default_kwargs_backward_compatible(self):
        """不传新参数 = 改动前行为：契约走 Settings 缺省版（含双通道规则），seed/thinking 不下发。"""
        llm = make_llm('{"key_turn_score": 4, "forbidden_items": [], "state_fit": "是", "reason": "ok"}')
        v = KeyTurnJudge(llm).judge(get_domain("collection"), SCENARIO_V23, [], "x")
        assert v.passed is True
        kw = llm.chat.call_args.kwargs
        assert kw["seed"] is None and kw["thinking"] is None
        assert kw["temperature"] == 0.0
        assert "两个通道" in self._contract_of(self._prompt_of(llm))  # 缺省版契约仍带双通道规则

    def test_variant_switches_judge_contract(self):
        """契约版本必须能跟着 run 的 agent_prompt 走：v8 无双通道规则、v9 已剔字数约束。

        复算历史 run 时若不传 variant，裁判会拿被测模型根本没见过的约束去判
        （实测同一场景 v7/v8/v9 契约 = 1657/1160/1639 字）。
        """
        contracts = {}
        for variant in ("v8", "v9"):
            llm = make_llm('{"key_turn_score": 4, "forbidden_items": [], "state_fit": "是", "reason": "ok"}')
            KeyTurnJudge(llm).judge(get_domain("collection"), SCENARIO_V23, [], "x", variant=variant)
            contracts[variant] = self._contract_of(self._prompt_of(llm))
        assert "两个通道" not in contracts["v8"] and "两个通道" in contracts["v9"]
        assert "回复控制在" in contracts["v8"] and "回复控制在" not in contracts["v9"]
        assert contracts["v8"] != contracts["v9"]

    def test_seed_and_thinking_passthrough(self):
        llm = make_llm('{"key_turn_score": 5, "forbidden_items": [], "state_fit": "是", "reason": "ok"}')
        KeyTurnJudge(llm).judge(get_domain("collection"), SCENARIO_V23, [], "x",
                                seed=20260903, thinking=False)
        kw = llm.chat.call_args.kwargs
        assert kw["seed"] == 20260903 and kw["thinking"] is False

    def test_history_rendered_and_no_audio_claim(self):
        """planS4 修问题 1/2：history 必须进 prompt；prompt 不得声称有音频。"""
        hist = [{"role": "user", "content": "我这会儿不方便。"},
                {"role": "assistant", "content": "好的，那我先长话短说。"}]
        llm = make_llm('{"key_turn_score": 4, "forbidden_items": [], "state_fit": "是", "reason": "ok"}')
        KeyTurnJudge(llm).judge(get_domain("collection"), SCENARIO_V23, hist, "x")
        p = self._prompt_of(llm)
        assert "用户: 我这会儿不方便。" in p and "客服: 好的，那我先长话短说。" in p
        assert "下面是【客户关键轮的音频】" not in p
        llm2 = make_llm('{"key_turn_score": 4, "forbidden_items": [], "state_fit": "是", "reason": "ok"}')
        KeyTurnJudge(llm2).judge(get_domain("collection"), SCENARIO_V23, [], "x")
        assert "（无）" in self._prompt_of(llm2)

    def test_action_table_single_source(self):
        """v2.7-S：速查由 domains.base 注入，v8 措辞不再出现在裁判 prompt 里。"""
        from listen2serve.domains.base import ACTION_V9, judge_action_table
        from listen2serve.evaluation import prompts as P

        assert P.KEYTURN_JUDGE_VERSION == "v2.7-S"
        assert "{action_cheatsheet}" not in P.KEYTURN_JUDGE_PROMPT
        assert P.KEYTURN_ACTION_CHEATSHEET.strip() in P.KEYTURN_JUDGE_PROMPT
        assert P.KEYTURN_ACTION_CHEATSHEET == P._indent_block(judge_action_table())
        for st, row in ACTION_V9.items():  # 五个态的 opening 全部同源可见
            assert row["opening"] in P.KEYTURN_JUDGE_PROMPT, st
        # 旧两行压缩速查（v8 措辞）必须消失，否则又是「被测按一份做、裁判按另一份判」
        for legacy in ("先自证身份、给可核实依据", "顺势推进坐实要素", "此处为速查"):
            assert legacy not in P.KEYTURN_JUDGE_PROMPT
        # 判据优先级 + 无音频声明（Q3 裁定的处置）
        assert "判据优先级" in P.KEYTURN_JUDGE_PROMPT
        assert "不适用于本判定" in P.KEYTURN_JUDGE_PROMPT
        # follow 只作参考、不单独扣分（ACTION_V9 的定义：opening 才是裁判判据）
        assert "未做不单独扣分" in P.KEYTURN_JUDGE_PROMPT
        assert len(P.KEYTURN_JUDGE_PROMPT_HASH) == 16
        assert P.KEYTURN_ACTION_CHEATSHEET_HASH != P.KEYTURN_JUDGE_PROMPT_HASH

    def test_judge_rubric_text_opening_only(self):
        """副指标 action_met 的单态 rubric：只取 opening，兼容旧中文/旧 11 态码。"""
        from listen2serve.domains.base import ACTION_V9, judge_rubric_text

        assert judge_rubric_text("cooperative") == ACTION_V9["cooperative"]["opening"]
        assert judge_rubric_text("N4_privacy") == ACTION_V9["hushed"]["opening"]
        assert judge_rubric_text("轻声/不便") == ACTION_V9["hushed"]["opening"]
        assert judge_rubric_text("不存在的态") == "按客服规范回应"


class TestFlowJudge:
    def test_checklist_source_priority(self):
        assert flow_checklist(SCENARIO_V23) == ["确认身份", "促成还款承诺"]
        # 无 effective 键回落 agent_policy.required_actions（v6.0 口径）
        assert flow_checklist(SCENARIO) == ["确认身份", "根据态度调整施压", "促成还款承诺"]

    def test_flow_rate_and_task_score(self):
        llm = make_llm(
            '{"flow_items": [{"action": "确认身份", "hit": "是", "turn": 1, "evidence": "请问是孙先生吗"},'
            ' {"action": "促成还款承诺", "hit": "部分", "turn": 4, "evidence": "用户说考虑下"}],'
            ' "task_score": 1, "reason": "部分推进"}'
        )
        v = FlowJudge(llm).judge(get_domain("collection"), SCENARIO_V23,
                                 [{"role": "user", "content": "喂"}, {"role": "assistant", "content": "您好"}])
        assert v.flow_rate == 0.75  # 是=1 + 部分=0.5 → 1.5/2
        assert v.task_score_raw == 1
        assert v.task_score == 0.5  # 0-2 归一 0/0.5/1
        assert [it["action"] for it in v.flow_items] == ["确认身份", "促成还款承诺"]

    def test_checklist_alignment_guard(self):
        """裁判改写/漏条：按固定清单对齐，缺失条目计否。"""
        llm = make_llm('{"flow_items": [{"action": "裁判改写过的动作", "hit": "是", "turn": 2}], "task_score": 2}')
        v = FlowJudge(llm).judge(get_domain("collection"), SCENARIO_V23, [])
        assert v.flow_items[0]["action"] == "确认身份"  # 以清单原文为准
        assert v.flow_items[1]["hit"] == "否"  # 缺失条目计否
        assert v.task_score == 1.0


# ---- 截断观察窗（2026-09-11，为「会话寿命装不下一通完整对话」的端点而加）----

def _flow_data(hits, in_windows, task=2):
    assert len(hits) == len(in_windows)
    return {
        "flow_items": [{"action": f"动作{i+1}", "hit": h, "turn": None if h == "否" else 1,
                        "evidence": "无", "in_window": w}
                       for i, (h, w) in enumerate(zip(hits, in_windows))],
        "task_score": task, "reason": "x",
    }


def test_windowed_flow_rate_uses_in_window_denominator():
    """截断窗内的比率只以「到第 K 轮为止该做的动作」为分母。

    否则「会话被服务端在 91–146 s 掐断」一类的端点，量到的是走到了第几轮，
    而不是能力 —— 清单 p50=5 条、43.9% 属收尾/确认类，跑不到就恒为 0。
    """
    from listen2serve.evaluation.flow_judge import _parse_flow_json

    ck = [f"动作{i}" for i in range(1, 6)]
    d = _flow_data(["是", "部分", "否", "否", "否"], [True, True, True, False, False])
    v = _parse_flow_json(d, ck, turn_cap=3)
    assert v.flow_rate == 0.3, "整通口径的分母仍是全清单（0/None 时才用它，三家既有读数不变）"
    assert v.n_items_in_window == 3 and v.flow_rate_in_window == 0.5
    assert v.task_score is None, "对话没结束 ⇒ 终局分无从判定，必须记 None 而不是 0"
    assert v.turn_cap == 3


def test_whole_call_mode_still_scores_task_and_has_no_window_fields():
    """回归保护：turn_cap 缺省（整通口径）时逐项行为与加窗之前完全一致。"""
    from listen2serve.evaluation.flow_judge import _parse_flow_json

    ck = ["动作1", "动作2"]
    v = _parse_flow_json(_flow_data(["是", "否"], [True, True], task=1), ck)
    assert v.turn_cap is None and v.flow_rate_in_window is None and v.n_items_in_window is None
    assert v.task_score == 0.5 and v.flow_rate == 0.5


def test_hit_items_are_forced_into_window_so_ratio_cannot_be_inflated():
    """裁判若把已命中的条目报成 in_window=false，会把分母缩小、比率抬高 ⇒ 代码侧强制归位。"""
    from listen2serve.evaluation.flow_judge import _parse_flow_json

    ck = ["动作1", "动作2", "动作3"]
    d = _flow_data(["是", "是", "否"], [False, True, False])
    v = _parse_flow_json(d, ck, turn_cap=2)
    assert [it["in_window"] for it in v.flow_items] == [True, True, False]
    assert v.n_items_in_window == 2 and v.flow_rate_in_window == 1.0


def test_turn_cap_cuts_by_turn_not_by_line():
    """按轮截断，不能按下标截 —— 下标截会把某一轮的客服发言切掉半句。"""
    from listen2serve.evaluation.flow_judge import _render_history_turn_numbered

    h = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
         {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
         {"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"}]
    out = _render_history_turn_numbered(h, 2)
    assert "a2" in out and "u3" not in out and "a3" not in out, out
    assert out.count("[第") == 4 and "第2轮] 客服: a2" in out, "第 2 轮必须完整成对"


def test_window_suffix_is_defined_exactly_once():
    """截断窗提示词只许有一份定义。

    Python 允许重复赋值 ⇒ 同名常量写两遍时**后一份静默生效**，而先前那份的措辞会留在
    文件里被当成"当前口径"引用。2026-09-11 就出过一次（编辑工具报"保存失败"实际已写入，
    随后脚本又追加一份），报告引用的措辞与真发给裁判的措辞不是同一份。
    """
    from pathlib import Path
    import listen2serve.evaluation.prompts as P

    src = Path(P.__file__).read_text(encoding="utf-8")
    assert src.count("FLOW_JUDGE_WINDOW_SUFFIX = ") == 1, "窗口提示词出现多份定义"
    # 生效的这份必须与真裁判探针验证过的那份一致（关键约束在、终局分显式置空）
    for must in ("in_window", "task_score", "只包含前 {turn_cap} 轮"):
        assert must in P.FLOW_JUDGE_WINDOW_SUFFIX, must
