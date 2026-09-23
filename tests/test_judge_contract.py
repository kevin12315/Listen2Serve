"""B2 回归测试：统一裁判契约渲染器 + Policy 逐动作分解 + F5 重试 + F1 版本追溯。

防回归点：
1. render_judge_contract 覆盖 agent_policy 全字段 + 域规则（canonical 码）+ general_constraints + 声音要求；
2. policy_judge/dialogue_metrics 的 _render_rules 为薄封装（输出与契约渲染器一致，保留 B1 归一化）；
3. 新格式逐条明细聚合顶层字段且 PolicyPass 判据不变；旧格式 JSON 回落旧判据解析；
4. F5：解析失败重试一次再落默认值（policy/voice/dialogue 三裁判同口径）；
5. F1：judge prompt 版本常量与各 prompt hash 可组装进 judge_meta；
6. Transition 评分标准删除不可判定的「声音适配」子项。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from listen2serve.domains import get_domain
from listen2serve.domains.base import render_judge_contract
from listen2serve.evaluation import dialogue_metrics, policy_judge
from listen2serve.evaluation.dialogue_metrics import DialogueMetrics
from listen2serve.evaluation.policy_judge import PolicyJudge
from listen2serve.evaluation.prompts import (
    DIALOGUE_TRANSITION_JUDGE_PROMPT,
    JUDGE_PROMPT_VERSION,
    POLICY_JUDGE_PROMPT_HASH,
    build_judge_meta,
    prompt_hash,
)
from listen2serve.evaluation.voice_judge import VoiceJudge
from listen2serve.gateway.llm_gateway import LLMResult

SCENARIO = {
    "scenario_id": "COL-NEG-01",
    "role": "collection",
    "oracle_state": "N5_resistant",
    "agent_policy": {
        "role_description": "你是催收客服",
        "communication_style": "坚定、冷静",
        "required_actions": ["确认身份", "促成还款承诺"],
        "forbidden_actions": ["威胁辱骂"],
        "allowed_actions": ["分期方案"],
        "voice_requirements": "语速中等，坚定不攻击",
    },
}


def make_llm(text: str) -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = LLMResult(text=text)
    return llm


class TestRenderJudgeContract:
    def test_full_contract_fields(self):
        domain = get_domain("collection")
        c = render_judge_contract(domain, SCENARIO)
        # agent_policy 全字段
        assert "你是催收客服" in c
        assert "坚定、冷静" in c
        assert "必需动作：确认身份；促成还款承诺" in c
        assert "禁止动作：威胁辱骂" in c
        assert "允许动作：分期方案" in c
        assert "语速中等，坚定不攻击" in c
        # 域规则 canonical 码（Plan I 新 6 态）+ 通用约束 + 声音要求段
        assert "[displeased（不满）]" in c
        assert "## 通用约束（跨状态）" in c
        assert "不得威胁、辱骂或使用恐吓性语言" in c  # collection general_constraints
        assert "## 声音要求" in c

    def test_current_state_marked(self):
        """旧 11 态 oracle 归一化到新 6 态后标注当前状态（旧 run 重评兼容）。"""
        domain = get_domain("collection")
        c = render_judge_contract(domain, SCENARIO, current_state="N5_resistant")
        assert "[displeased（不满）] ←当前状态" in c
        assert "当前用户状态：displeased" in c

    def test_legacy_chinese_state_key_normalizes(self):
        """B1 行为保留：旧格式中文 oracle 状态同样归一后标注。"""
        domain = get_domain("collection")
        c = render_judge_contract(domain, SCENARIO, current_state="N5 厌烦/抗拒（软拒绝）")
        assert "←当前状态" in c
        assert "displeased" in c

    def test_expected_key_behavior_anchor(self):
        """Plan I（v6.0）：expected_key_behavior 锚点渲染为「关键事件应对锚点」（仅裁判可见）；
        旧锚点（stage_at_critical_turn）保留兼容。"""
        domain = get_domain("collection")
        sc = dict(SCENARIO, expected_key_behavior="接住客户出价并落实到数字与时间")
        c = render_judge_contract(domain, sc)
        assert "## 关键事件应对锚点（仅裁判可见）" in c
        assert "接住客户出价并落实到数字与时间" in c
        assert "关键轮所处流程阶段" not in c
        # 旧格式兼容
        legacy = dict(SCENARIO, stage_at_critical_turn="S4 方案协商",
                      expected_stage_behavior="给出可选方案")
        c2 = render_judge_contract(domain, legacy)
        assert "关键轮所处流程阶段：S4 方案协商" in c2

    def test_voice_requirements_domain_fallback(self):
        domain = get_domain("collection")
        sc = dict(SCENARIO, agent_policy={"role_description": "催收"})
        c = render_judge_contract(domain, sc)
        assert domain.default_voice_requirements in c

    def test_thin_wrappers_delegate(self):
        """policy_judge/dialogue_metrics 的 _render_rules 为薄封装（与渲染器输出一致）。"""
        domain = get_domain("collection")
        assert policy_judge._render_rules(domain, SCENARIO) == render_judge_contract(domain, SCENARIO)
        assert dialogue_metrics._render_rules(domain, SCENARIO) == render_judge_contract(domain, SCENARIO)
        assert policy_judge._render_rules(domain, SCENARIO, current_state="P1_cooperative") == (
            render_judge_contract(domain, SCENARIO, current_state="P1_cooperative")
        )


class TestPolicyItemizedVerdict:
    def test_new_format_all_hit_passes(self):
        llm = make_llm(
            '{"required_items": [{"action": "确认身份", "hit": "是", "evidence": "已核实"},'
            '{"action": "促成还款承诺", "hit": "是", "evidence": "约定周五"}],'
            '"forbidden_items": [{"action": "威胁辱骂", "triggered": "否", "evidence": "无"}],'
            '"state_fit": "是", "context_coherent": "是", "required_rate": 1.0, "reason": "ok"}'
        )
        v = PolicyJudge(llm).judge(get_domain("collection"), SCENARIO, [], "已核实身份，周五还款可以吗？")
        assert v.passed is True
        assert v.required_satisfied == "是"
        assert v.forbidden_triggered == "否"
        assert v.required_rate == 1.0
        assert len(v.required_items) == 2 and v.required_items[0]["evidence"] == "已核实"
        assert len(v.forbidden_items) == 1
        assert v.as_dict()["required_items"]  # 明细进增量键

    def test_new_format_partial_fails(self):
        # v5.6：required_rate 以明细聚合为唯一来源；LLM 自报 0.9 与明细（是/部分，
        # 聚合 0.5）矛盾时，采信聚合值 0.5，自报值仅留 raw 对照。
        llm = make_llm(
            '{"required_items": [{"action": "确认身份", "hit": "是", "evidence": "e"},'
            '{"action": "促成还款承诺", "hit": "部分", "evidence": "e"}],'
            '"forbidden_items": [], "state_fit": "是", "context_coherent": "是",'
            '"required_rate": 0.9, "reason": "部分"}'
        )
        v = PolicyJudge(llm).judge(get_domain("collection"), SCENARIO, [], "好的。")
        assert v.required_satisfied == "部分"
        assert v.required_rate == 0.5  # 明细聚合（非自报 0.9）
        assert v.raw.get("required_rate") == 0.9  # 自报值保留在 raw 对照
        assert v.passed is False

    def test_new_format_forbidden_triggered_fails(self):
        llm = make_llm(
            '{"required_items": [{"action": "确认身份", "hit": "是", "evidence": "e"}],'
            '"forbidden_items": [{"action": "威胁辱骂", "triggered": "是", "evidence": "起诉你"}],'
            '"state_fit": "是", "context_coherent": "是", "required_rate": 1.0, "reason": "威胁"}'
        )
        v = PolicyJudge(llm).judge(get_domain("collection"), SCENARIO, [], "不还就起诉你！")
        assert v.forbidden_triggered == "是"
        assert v.passed is False

    def test_legacy_format_fallback(self):
        """旧格式 JSON 回落旧判据解析（存量 mock 兼容）。"""
        llm = make_llm(
            '{"required_satisfied": "是", "forbidden_triggered": "否", '
            '"state_fit": "是", "context_coherent": "是", "reason": "ok"}'
        )
        v = PolicyJudge(llm).judge(get_domain("collection"), SCENARIO, [], "可以商量分期。")
        assert v.passed is True
        assert v.required_items == [] and v.required_rate is None


class TestF5Retry:
    def test_policy_retry_then_success(self):
        llm = MagicMock()
        llm.chat.side_effect = [
            LLMResult(text="这不是 JSON"),
            LLMResult(text='{"required_satisfied": "是", "forbidden_triggered": "否", '
                           '"state_fit": "是", "context_coherent": "是", "reason": "ok"}'),
        ]
        judge = PolicyJudge(llm)
        v = judge.judge(get_domain("collection"), SCENARIO, [], "回复")
        assert v.passed is True
        assert v.parse_retried is True
        assert judge.parse_retries == 1 and judge.parse_failures == 0
        assert llm.chat.call_count == 2

    def test_policy_double_failure_defaults(self):
        llm = MagicMock()
        llm.chat.return_value = LLMResult(text="无法解析")
        judge = PolicyJudge(llm)
        v = judge.judge(get_domain("collection"), SCENARIO, [], "回复")
        assert v.passed is False  # 落默认值（全部否定）
        assert judge.parse_retries == 1 and judge.parse_failures == 1
        assert llm.chat.call_count == 2

    def test_dialogue_retry_then_success(self):
        llm = MagicMock()
        llm.chat.side_effect = [LLMResult(text="乱码"), LLMResult(text='{"score": 2, "reason": "ok"}')]
        dm = DialogueMetrics(llm)
        v = dm.task_completion(get_domain("collection"), SCENARIO, [])
        assert v.score == 2
        assert dm.parse_retries == 1 and dm.parse_failures == 0

    def test_voice_retry_then_success(self, tmp_path):
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)
        wav = tmp_path / "a.wav"
        wav.write_bytes(buf.getvalue())
        llm = MagicMock()
        llm.chat.side_effect = [
            LLMResult(text="抱歉，无法输出"),
            LLMResult(text='{"role_voice_match": 4, "state_voice_fit": 4, "naturalness": 4, "reason": "ok"}'),
        ]
        judge = VoiceJudge(llm)
        v = judge.judge(str(wav), "催收客服", "N5_resistant", "您好")
        assert v.passed is True
        assert judge.parse_retries == 1 and judge.parse_failures == 0


class TestVoiceJudgeContractInjection:
    def test_voice_requirements_injected_into_prompt(self, tmp_path):
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)
        wav = tmp_path / "a.wav"
        wav.write_bytes(buf.getvalue())
        llm = make_llm(
            '{"role_voice_match": 4, "state_voice_fit": 4, "naturalness": 4, "reason": "ok"}'
        )
        judge = VoiceJudge(llm)
        judge.judge(
            str(wav), "催收客服", "N5_resistant", "您好",
            voice_requirements="语速中等，坚定不攻击", communication_style="坚定、冷静",
        )
        prompt_text = llm.chat.call_args[0][0][0]["content"][0]["text"]
        assert "语速中等，坚定不攻击" in prompt_text
        assert "坚定、冷静" in prompt_text
        # 硬编码角色刻板印象已删除
        assert "催收=坚定" not in prompt_text


class TestJudgeMetaAndTransition:
    def test_judge_prompt_version_frozen(self):
        assert JUDGE_PROMPT_VERSION == "v2.5-K"  # Plan K：弦外之音 1-5 量规升档
        assert len(POLICY_JUDGE_PROMPT_HASH) == 16
        # hash 对文本变化敏感（版本追溯有效）
        assert prompt_hash("a") != prompt_hash("b")

    def test_voice_judge_version_independent(self):
        """Plan K 审阅修正：judge_meta.voice.version 独立跟踪，不随全局/keyturn 版本号漂移。"""
        from listen2serve.evaluation.prompts import (
            KEYTURN_JUDGE_VERSION,
            VOICE_JUDGE_VERSION,
        )

        assert VOICE_JUDGE_VERSION not in (JUDGE_PROMPT_VERSION, KEYTURN_JUDGE_VERSION)
        for schema in ("v2.2-I", "v2.3-J"):
            meta = build_judge_meta("m", "v", "d", "task_completion", judge_schema=schema)
            assert meta["voice"]["version"] == VOICE_JUDGE_VERSION

    def test_build_judge_meta_structure(self):
        meta = build_judge_meta("stub-judge-model", "qwen3.5-omni-plus", "stub-judge-model", "task_completion")
        assert meta["policy"]["version"] == JUDGE_PROMPT_VERSION
        assert meta["policy"]["prompt_hash"] == POLICY_JUDGE_PROMPT_HASH
        assert meta["voice"]["model"] == "qwen3.5-omni-plus"
        assert meta["dialogue"]["prompt_hash"]
        assert build_judge_meta(dialogue_metric=None)["dialogue"] is None

    def test_build_judge_meta_profile_injected(self):
        """v5.9 契约内容维度标记：policy/dialogue 携带 profile_injected；
        未传时键缺失（旧版结构兼容，键缺失 = v5.8 断点前的甄别特征）；
        Voice 裁判不注入档案，不携带该标记。"""
        for flag in (True, False):
            meta = build_judge_meta("m", "v", "d", "task_completion", profile_injected=flag)
            assert meta["policy"]["profile_injected"] is flag
            assert meta["dialogue"]["profile_injected"] is flag
            assert "profile_injected" not in meta["voice"]
        meta_tr = build_judge_meta("m", "v", "d", "transition_score", profile_injected=True)
        assert meta_tr["dialogue"]["profile_injected"] is True
        legacy = build_judge_meta("m", "v", "d", "task_completion")
        assert "profile_injected" not in legacy["policy"]
        assert "profile_injected" not in legacy["dialogue"]
        assert build_judge_meta("m", "v", "d", None, profile_injected=True)["dialogue"] is None

    def test_transition_prompt_drops_voice_fit(self):
        """文本裁判无法听声：Transition 评分标准删除「声音适配」子项。"""
        assert "声音适配" not in DIALOGUE_TRANSITION_JUDGE_PROMPT
