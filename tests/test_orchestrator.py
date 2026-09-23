"""编排器单测（tick 级全双工，fake Agent 全流程；Plan I v6.0 骨架驱动模拟器口径）。"""

from __future__ import annotations

import json

import pytest
from conftest import fake_instruction_entry

from listen2serve.gateway.llm_gateway import LLMResult
from listen2serve.gateway.tts_gateway import TTSResult
from listen2serve.runtime.agent import AgentTickResult
from listen2serve.runtime.orchestrator import FullDuplexOrchestrator
from listen2serve.runtime.user_simulator import UserSimulator


def _proto(text: str, **over) -> str:
    """v6.0 输出协议 JSON（中性台词均避开情绪词，T2 口径安全）。"""
    d = {
        "text": text, "state": "neutral", "is_key_turn": False,
        "end_call": False, "covered": [], "interrupt": False,
    }
    d.update(over)
    return json.dumps(d, ensure_ascii=False)


class FakeLLM:
    """按脚本顺序逐次返回协议 JSON（v6.0 模拟器消费）。"""

    def __init__(self, script: list[str] | None = None) -> None:
        # 缺省脚本：前 6 轮中性推进，第 2 轮关键事件，第 5 轮收尾
        self.script = list(script or [
            _proto("喂你好，我就是本人，你说吧，什么事？", covered=[1]),
            _proto("行，我下个月十号前后先还一部分，其余再宽限几天。",
                   state="cooperative", is_key_turn=True, covered=[2, 3],
                   tts={"tone": "商量着来", "rate": "中等", "volume": "正常"}),
            _proto("嗯，那还款的渠道你跟我说一下，怎么操作？", covered=[4]),
            _proto("好，到账时间我知道了，没别的问题了。", covered=[5]),
            _proto("行，那就这么说定了，麻烦你了，再见。", end_call=True, covered=[6]),
            _proto("嗯，好的，先这样吧，挂了。"),
        ])
        self.calls = 0

    def chat(self, *a, **kw):
        self.calls += 1
        if not self.script:
            return LLMResult(text=_proto("嗯，好的，我知道了，先这样吧。"))
        return LLMResult(text=self.script.pop(0), finish_reason="stop")


class FakeTTS:
    def synthesize(self, text, **kw):
        # 非零音频（编排器/FakeAgent 用能量区分语音与静音块）
        return TTSResult(audio=b"\x01\x00" * 1600, sample_rate=16000)


def _v6_scenario(**over) -> dict:
    """最小 v6.0 合成场景（骨架契约：mission/key_event/disclosure_plan/turn_budget）。"""
    sc = {
        "scenario_id": "SYN-V6-01-T2",
        "role": "collection",
        "role_type": "outbound_pressure",
        "dynamics": "all_positive",
        "leakage_label": "T2",
        "leakage_level": "implicit_consistent",
        "dataset_version": "v6.0",
        "oracle_state": "用户处于cooperative（配合）状态，动态类型 all_positive",
        "user_script": {
            "identity": "张三，35 岁上班族，有一笔欠款",
            "user_goal": "核对清楚后给出可行还款安排",
            "user_facts": {"planned_amount": 2000},
            "mission": [
                "接听电话，确认自己是本人",
                "弄清来意与欠款情况",
                "【关键事件】摊牌还款能力：先还一部分其余宽限",
                "确认还款渠道",
                "确认到账时间",
                "达成约定后道别",
            ],
            "key_event": {
                "description": "摊牌还款能力：先还一部分，其余宽限",
                "state": "cooperative",
                "trigger_hint": "客服给出具体还款要求之后",
                "interrupt": False,
            },
            "disclosure_plan": {"planned_amount": "客服问到还款能力前不主动说"},
            "turn_budget": {"soft": 5, "hard": 9},
        },
        "agent_policy": {"role_description": "你是催收客服"},
        "measurement": {"evaluation_point": "Agent 在关键事件轮用户发言后的回复"},
    }
    sc.update(over)
    return sc


class FakeTickAgent:
    """假 Agent（tick 契约）：听到用户语音块后，静默 2 tick 开始回复，
    每轮输出 3 tick 音频 + 转写，最后一 tick 置 response_done。"""

    REPLY_TICKS = 3

    def __init__(self) -> None:
        self.model_spec = "fake/agent"
        self.replies = [
            "您好，请问是张三先生吗？",
            "好的，那您看分期方案可以吗？",
            "明白，我给您登记一下。",
            "好的，感谢您的配合。",
        ]
        self._saw_speech = False
        self._silence_ticks = 0
        self._reply_left = 0
        self._reply_text = ""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def attach_db(self, db) -> None:
        pass

    async def cancel_response(self) -> None:
        pass

    async def run_tick(self, chunk: bytes, tick: int = 0) -> AgentTickResult:
        is_speech = any(chunk)
        r = AgentTickResult(has_activity=True)
        if is_speech:
            self._saw_speech = True
            self._silence_ticks = 0
            return r
        if self._saw_speech and self._reply_left == 0:
            self._silence_ticks += 1
            if self._silence_ticks >= 2:  # server VAD 静默阈值模拟
                self._saw_speech = False
                self._reply_left = self.REPLY_TICKS
                self._reply_text = self.replies.pop(0) if self.replies else "好的。"
            return r
        if self._reply_left > 0:
            idx = self.REPLY_TICKS - self._reply_left
            n = len(self._reply_text)
            r.audio = b"\x02\x00" * 4800  # 200ms @ 24k PCM16
            r.transcript = self._reply_text[idx * n // 3 : (idx + 1) * n // 3]
            self._reply_left -= 1
            if self._reply_left == 0:
                r.transcript = self._reply_text[idx * n // 3 :]
                r.response_done = True
            r.buffer_bytes = 0
        return r


@pytest.mark.asyncio
async def test_orchestrator_tick_full_flow(tmp_path):
    scenario = _v6_scenario()
    agent = FakeTickAgent()
    sim = UserSimulator(scenario["user_script"], FakeLLM(), FakeTTS(), voice="Cherry",
                        role="collection", dynamics="all_positive",
                        leakage_label="T2", scenario_id=scenario["scenario_id"],
                        instruction_entry=fake_instruction_entry("cooperative"))
    orch = FullDuplexOrchestrator(artifact_dir=str(tmp_path), save_audio=True)

    traj = await orch.run_scenario(scenario, agent, sim, max_turns=4)
    assert traj.status == "ok"
    assert traj.termination_reason == "max_turns"
    assert len(traj.traces) == 4
    t1 = traj.traces[0]
    assert t1.user_text
    assert t1.agent_text == "您好，请问是张三先生吗？"
    # Plan I：协议字段随 TurnTrace 落盘（唯一关键轮定位来源）
    assert t1.is_key_turn is False
    assert traj.traces[1].is_key_turn is True
    assert traj.traces[1].user_tts_instruct
    assert sum(1 for t in traj.traces if t.is_key_turn) == 1
    # tick 时间戳：用户先说 → 客服后答
    assert t1.tick_start == 0
    assert t1.user_end_tick >= t1.tick_start
    assert t1.agent_first_audio_tick > t1.user_end_tick
    assert t1.tick_end >= t1.agent_first_audio_tick
    # 音频与 tick 时间线落盘
    assert (tmp_path / "audio" / "turn1_user.wav").exists()
    assert (tmp_path / "audio" / "turn1_agent.wav").exists()
    assert (tmp_path / "ticks.json").exists()
    assert (tmp_path / "both.wav").exists()
    assert (tmp_path / "conversation.wav").exists()
    # 交互指标
    assert traj.interaction["turns"] == 4
    assert traj.interaction["response_rate"] == 1.0
    assert traj.interaction["response_latency_mean_s"] > 0


@pytest.mark.asyncio
async def test_user_closed_termination(tmp_path):
    """end_call 自然收尾：next_turn 返回 None → termination_reason=user_closed。"""
    scenario = _v6_scenario()
    agent = FakeTickAgent()
    sim = UserSimulator(scenario["user_script"], FakeLLM(), FakeTTS(), voice="Cherry",
                        role="collection", dynamics="all_positive",
                        leakage_label="T2", scenario_id=scenario["scenario_id"],
                        instruction_entry=fake_instruction_entry("cooperative"))
    orch = FullDuplexOrchestrator(artifact_dir=str(tmp_path), save_audio=False)
    traj = await orch.run_scenario(scenario, agent, sim, max_turns=8)
    assert traj.status == "ok"
    assert traj.termination_reason == "user_closed"
    assert traj.traces[-1].end_call is True
    assert traj.key_turn_missing is False


@pytest.mark.asyncio
async def test_key_turn_missing_marked(tmp_path):
    """到达硬上限仍无关键轮 → key_turn_missing（评测侧跳过并计数）。"""
    scenario = _v6_scenario()
    agent = FakeTickAgent()
    # 脚本里始终不报关键轮、不收尾
    llm = FakeLLM([
        _proto("喂你好，我就是本人，你说吧，什么事啊？"),
        _proto("嗯，这个情况我再想想，你容我考虑两天。"),
        _proto("我知道了，你先说说具体的流程吧。"),
        _proto("嗯，渠道这块我清楚了，还有别的吗？"),
        _proto("好，时间上我记下了，回头安排。"),
    ])
    sim = UserSimulator(scenario["user_script"], llm, FakeTTS(), voice="Cherry",
                        role="collection", dynamics="all_positive",
                        leakage_label="T2", scenario_id=scenario["scenario_id"],
                        instruction_entry=fake_instruction_entry("cooperative"))
    orch = FullDuplexOrchestrator(artifact_dir=str(tmp_path), save_audio=False)
    traj = await orch.run_scenario(scenario, agent, sim, max_turns=3)
    assert traj.status == "ok"
    assert traj.termination_reason == "max_turns"
    assert traj.key_turn_missing is True


class FakeToolTickAgent:
    """工具轮契约（复现 Qwen Realtime 实测行为）：用户说完后（可选）先说一句 →
    工具调用与空 done 同 tick 到达（工具调用响应无音频即结束）→ 下一 tick 投递结果
    （tools_flushed）→ 延迟若干 tick 后续接语音（新响应的 done 不再发）。"""

    FOLLOWUP_TEXT = "您好，订单运输中，预计后天送达。"

    def __init__(self, pre_speech: bool, followup: bool, delay_ticks: int = 8) -> None:
        self.model_spec = "fake/tool-agent"
        self.pre_speech = pre_speech
        self.followup = followup
        self.delay_ticks = delay_ticks
        self._saw_speech = False
        self._silence = 0
        self._script: list[AgentTickResult] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def cancel_response(self) -> None:
        pass

    def _reply_script(self) -> list[AgentTickResult]:
        s: list[AgentTickResult] = []
        if self.pre_speech:
            for part in ("请稍等，", "我帮您", "查一下。"):
                s.append(AgentTickResult(audio=b"\x02\x00" * 4800, transcript=part))
        tool = AgentTickResult(response_done=True)  # 工具调用响应：空 done 与 function_call 同至
        tool.tool_calls.append({"call_id": "c1", "name": "query_order", "arguments": {}})
        tool.tool_outputs.append({"call_id": "c1", "name": "query_order", "arguments": {}, "output": "{}"})
        s.append(tool)
        s.append(AgentTickResult(tools_flushed=1))  # 下一 tick 投递工具结果
        if self.followup:
            s.extend(AgentTickResult() for _ in range(self.delay_ticks))  # 新响应生成时延
            n = len(self.FOLLOWUP_TEXT)
            for i in range(3):
                hi = (i + 1) * n // 3 if i < 2 else n
                s.append(AgentTickResult(
                    audio=b"\x02\x00" * 4800, transcript=self.FOLLOWUP_TEXT[i * n // 3 : hi],
                ))
        return s

    async def run_tick(self, chunk: bytes, tick: int = 0) -> AgentTickResult:
        if any(chunk):
            self._saw_speech = True
            self._silence = 0
            return AgentTickResult(has_activity=True)
        if self._saw_speech:
            self._silence += 1
            if self._silence >= 2:  # server VAD 静默阈值模拟
                self._saw_speech = False
                self._script = self._reply_script()
            return AgentTickResult(has_activity=True)
        if self._script:
            return self._script.pop(0)
        return AgentTickResult()


@pytest.mark.asyncio
@pytest.mark.parametrize("pre_speech", [False, True])
async def test_tool_turn_late_followup_captured(tmp_path, pre_speech):
    """工具轮续接语音（迟到、done 不重发）必须记入本轮——含先说话再调工具的情形。"""
    scenario = _v6_scenario()
    agent = FakeToolTickAgent(pre_speech=pre_speech, followup=True, delay_ticks=8)
    sim = UserSimulator(scenario["user_script"], FakeLLM(), FakeTTS(), voice="Cherry",
                        role="collection", leakage_label="T2",
                        instruction_entry=fake_instruction_entry("cooperative"))
    orch = FullDuplexOrchestrator(artifact_dir=str(tmp_path), save_audio=False)
    traj = await orch.run_scenario(scenario, agent, sim, max_turns=1)
    assert traj.status == "ok"
    t1 = traj.traces[0]
    assert FakeToolTickAgent.FOLLOWUP_TEXT in t1.agent_text  # 续接语音未丢失/未错归下一轮
    assert t1.tool_calls and t1.tool_calls[0]["name"] == "query_order"
    assert not t1.timed_out
    assert not t1.interrupted


@pytest.mark.asyncio
async def test_tool_turn_no_followup_ends_by_grace(tmp_path):
    """纯工具轮无续接语音：宽限期后正常结束，不拖到 30s 超时。"""
    from listen2serve.runtime.orchestrator import POST_TOOL_GRACE_TICKS

    scenario = _v6_scenario()
    agent = FakeToolTickAgent(pre_speech=False, followup=False)
    sim = UserSimulator(scenario["user_script"], FakeLLM(), FakeTTS(), voice="Cherry",
                        role="collection", leakage_label="T2",
                        instruction_entry=fake_instruction_entry("cooperative"))
    orch = FullDuplexOrchestrator(artifact_dir=str(tmp_path), save_audio=False)
    traj = await orch.run_scenario(scenario, agent, sim, max_turns=1)
    assert traj.status == "ok"
    t1 = traj.traces[0]
    assert not t1.timed_out  # 未走 max_response_ticks 超时
    assert t1.agent_text == ""
    assert t1.agent_first_audio_tick < 0
    # 结束时点在宽限窗附近（非 150 tick 超时兜底）
    assert t1.tick_end - t1.user_end_tick < POST_TOOL_GRACE_TICKS + 15


@pytest.mark.asyncio
async def test_orchestrator_error(tmp_path):
    class BrokenAgent:
        model_spec = "fake/broken"

        async def start(self):
            pass

        async def stop(self):
            pass

        async def cancel_response(self):
            pass

        async def run_tick(self, chunk, tick=0):
            raise RuntimeError("provider died")

    scenario = _v6_scenario()
    sim = UserSimulator(scenario["user_script"], FakeLLM(), FakeTTS(),
                        instruction_entry=fake_instruction_entry("cooperative"))
    orch = FullDuplexOrchestrator(artifact_dir=str(tmp_path), save_audio=False)
    traj = await orch.run_scenario(scenario, BrokenAgent(), sim)
    assert traj.status == "error"
    assert traj.termination_reason == "error"
    assert "provider died" in traj.error


@pytest.mark.asyncio
async def test_trajectory_roundtrip(tmp_path):
    """Trajectory.as_dict ↔ from_dict（checkpoint 续跑依赖；含 Plan I 新字段）。"""
    from listen2serve.runtime.orchestrator import Trajectory, TurnTrace

    traj = Trajectory(scenario_id="S1", model="m", role="collection", status="ok",
                      termination_reason="user_closed", seed=42, key_turn_missing=False)
    traj.traces.append(TurnTrace(turn=1, user_state="neutral", user_text="喂",
                                 agent_text="您好", tick_start=0, tick_end=9,
                                 is_key_turn=True, user_tts_instruct="用…的语气",
                                 covered=[1, 2], forced_key_turn=False))
    d = traj.as_dict()
    back = Trajectory.from_dict(d)
    assert back.scenario_id == "S1"
    assert back.seed == 42
    assert back.termination_reason == "user_closed"
    assert back.traces[0].is_key_turn is True
    assert back.traces[0].covered == [1, 2]
    assert back.as_dict() == d


# ---- 打断来源拆分（interrupt_source：script=关键轮打断 / vad=自然 VAD 截断）----

def _interrupt_scenario() -> dict:
    """key_event.interrupt=true 的合成场景（关键轮以打断方式开口）。"""
    sc = _v6_scenario()
    sc["user_script"]["key_event"]["interrupt"] = True
    return sc


def _interrupt_llm_script() -> FakeLLM:
    """打断场景 LLM 脚本（逐次消费）：
    1) turn1 正常开口；2) agent1 期间 peek 草案（不打断）；3) turn2 正常；
    4) agent2 期间 peek 草案（关键轮+打断=true）→ agent2 被截断；
    5) 打断后 next_turn 重生成关键轮；6+) 推进收尾。"""
    key = dict(state="cooperative", is_key_turn=True, interrupt=True, covered=[2, 3],
               tts={"tone": "着急", "rate": "偏快", "volume": "正常"})
    return FakeLLM([
        _proto("喂你好，我就是本人，你说吧，什么事？", covered=[1]),
        _proto("嗯，你等一下，这个情况我得想想再说。"),          # agent1 peek 草案（不打断）
        _proto("欠款的事我知道了，你容我核对一下细节。", covered=[2]),
        _proto("你等等，我先说，这钱我下月十号前先还一部分。", **key),  # agent2 peek 草案（打断）
        _proto("你听我说，这钱我下月十号前先还一部分，行不行？", **key),  # 打断后重生成
        _proto("嗯，渠道和到账时间你说一下吧，我听着。", covered=[4]),
        _proto("行，那就这么说定了，麻烦你了，再见。", end_call=True, covered=[5, 6]),
        _proto("嗯，好的，先这样吧，挂了。"),
    ])


@pytest.mark.asyncio
async def test_script_interrupt_marked_script_source(tmp_path):
    """关键轮打断：被打断轮标记 interrupt_source=script，且不混入 VAD 截断统计。"""
    scenario = _interrupt_scenario()
    agent = FakeTickAgent()
    sim = UserSimulator(scenario["user_script"], _interrupt_llm_script(), FakeTTS(),
                        voice="Cherry", role="collection", dynamics="all_positive",
                        leakage_label="T2", scenario_id=scenario["scenario_id"],
                        instruction_entry=fake_instruction_entry("cooperative"))
    orch = FullDuplexOrchestrator(artifact_dir=str(tmp_path), save_audio=False)
    traj = await orch.run_scenario(scenario, agent, sim, max_turns=5)
    assert traj.status == "ok"
    t2 = traj.traces[1]  # 关键轮打断 → 第 2 轮客服被截断
    assert t2.interrupted
    assert t2.interrupt_source == "script"
    assert traj.traces[2].is_key_turn is True
    assert not traj.traces[0].interrupted and traj.traces[0].interrupt_source == ""
    assert traj.interaction["script_interrupted_turns"] == 1
    assert traj.interaction["vad_truncated_turns"] == 0
    assert traj.interaction["interrupted_turns"] == 1  # 合计 = 两项之和


class FakeVadTruncateAgent(FakeTickAgent):
    """在用户第 N 次开口首 chunk 上报 was_truncated（模拟自然 VAD 截断未播完缓冲）。"""

    def __init__(self, truncate_burst: int = 2) -> None:
        super().__init__()
        self.truncate_burst = truncate_burst
        self._bursts = 0
        self._speaking = False

    async def run_tick(self, chunk: bytes, tick: int = 0) -> AgentTickResult:
        is_speech = any(chunk)
        if is_speech and not self._speaking:
            self._bursts += 1
            if self._bursts == self.truncate_burst:  # 上一轮客服刚播完用户即接话 → VAD 截断残留缓冲
                self._speaking = True
                self._saw_speech = True
                self._silence_ticks = 0
                return AgentTickResult(has_activity=True, was_truncated=True)
        self._speaking = is_speech
        return await super().run_tick(chunk, tick)


@pytest.mark.asyncio
async def test_vad_truncation_not_counted_as_script(tmp_path):
    """自然 VAD 截断：标记 interrupt_source=vad，不计入脚本打断统计。"""
    scenario = _v6_scenario()  # key_event.interrupt=False
    agent = FakeVadTruncateAgent()
    sim = UserSimulator(scenario["user_script"], FakeLLM(), FakeTTS(), voice="Cherry",
                        role="collection", dynamics="all_positive",
                        leakage_label="T2", scenario_id=scenario["scenario_id"],
                        instruction_entry=fake_instruction_entry("cooperative"))
    orch = FullDuplexOrchestrator(artifact_dir=str(tmp_path), save_audio=False)
    traj = await orch.run_scenario(scenario, agent, sim, max_turns=2)
    assert traj.status == "ok"
    t1 = traj.traces[0]
    assert t1.interrupted
    assert t1.interrupt_source == "vad"
    assert traj.interaction["script_interrupted_turns"] == 0
    assert traj.interaction["vad_truncated_turns"] == 1
    assert traj.interaction["interrupted_turns"] == 1


def test_interaction_metrics_interrupt_split():
    """两类打断并存时指标拆分正确；interrupt_source 随轨迹序列化往返。"""
    from listen2serve.runtime.orchestrator import (
        Trajectory,
        TurnTrace,
        compute_interaction_metrics,
    )

    t1 = TurnTrace(turn=1, user_state="displeased", user_text="a", agent_text="x",
                   tick_start=0, user_end_tick=2, agent_first_audio_tick=4, tick_end=9,
                   interrupted=True, interrupt_source="script")
    t2 = TurnTrace(turn=2, user_state="cooperative", user_text="b", agent_text="y",
                   tick_start=10, user_end_tick=12, agent_first_audio_tick=14, tick_end=19,
                   interrupted=True, interrupt_source="vad")
    t3 = TurnTrace(turn=3, user_state="neutral", user_text="c", agent_text="z",
                   tick_start=20, user_end_tick=22, agent_first_audio_tick=24, tick_end=29)
    m = compute_interaction_metrics([t1, t2, t3], 0.2)
    assert m["interrupted_turns"] == 2  # 合计口径（任意来源）
    assert m["script_interrupted_turns"] == 1
    assert m["vad_truncated_turns"] == 1

    # 新字段随 as_dict ↔ from_dict 往返（checkpoint/旧产物兼容路径）
    traj = Trajectory(scenario_id="S", model="m", role="collection")
    traj.traces.extend([t1, t2, t3])
    back = Trajectory.from_dict(traj.as_dict())
    assert [t.interrupt_source for t in back.traces] == ["script", "vad", ""]
    # 旧产物缺 interrupt_source/is_key_turn 时 from_dict 回落默认值不报错
    legacy = traj.as_dict()
    for t in legacy["turns"]:
        t.pop("interrupt_source")
        t.pop("is_key_turn")
    legacy.pop("key_turn_missing")
    back2 = Trajectory.from_dict(legacy)
    assert all(t.interrupt_source == "" for t in back2.traces)
    assert all(t.interrupted for t in back2.traces[:2])  # interrupted 语义不变
    assert all(t.is_key_turn is False for t in back2.traces)
    assert back2.key_turn_missing is False
