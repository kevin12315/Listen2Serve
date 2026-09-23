"""tick 级全双工编排器（200ms 虚拟时钟，对齐 tau-voice FullDuplexOrchestrator 语义）。

主循环以 tick 为单位推进：
- 每 tick 向 Agent 送入一个 200ms 用户音频块（说话时为语音块，否则为静音块）；
- Agent 侧（server_vad）持续返回本 tick 播出的音频/比例转写/工具调用；
- 轮转由确定性状态机管理：user（用户说话）→ agent（等待+客服说话）→ gap（轮间隙）；
- Agent 轮结束判定：播出缓冲排空 ∧ 连续 idle_end_ticks 静默 ∧ 出过声（工具轮以
  结果投递后的 POST_TOOL_GRACE_TICKS 宽限窗兜底）。不依赖 response.done——
  done 只是单响应边界，一轮可跨多个 response（工具调用响应 + 续接语音响应）；
- 打断（barge-in）：脚本标记打断的轮，在 Agent 开口 interrupt_delay_ticks 后用户
  直接插话，server VAD speech_started 触发适配器截断（丢弃未播出缓冲）；
- 超时：单轮超过 max_response_ticks（连续输入型端点再加 STREAMING_GRACE_TICKS）
  强制取消响应并标记 timed_out；全程超过 max_tick_timeout_s 判定 timeout。

产物：逐轮 TurnTrace（含 tick 时间戳）、tick 时间线（ticks.json）、
对话音频（tick 时间轴合成 both.wav / conversation.wav）、语音交互指标。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from listen2serve.audio.io_utils import split_into_chunks
from listen2serve.runtime.agent import AgentTickResult
from listen2serve.runtime.config import Settings, get_settings
from listen2serve.runtime.errors import EndpointSessionClosed
from listen2serve.runtime.user_simulator import UserSimulator

logger = logging.getLogger(__name__)

USER_SAMPLE_RATE = 16000
AGENT_SAMPLE_RATE = 24000

# 工具结果投递后等待续接语音的宽限窗（25×200ms≈5s）：投递后 provider 会发 response.create
# 另起新响应“查询后说几句话”，宽限期内不结束本轮，以捕获迟到语音
POST_TOOL_GRACE_TICKS = 25

# 要求“连续上行输入”的端点（豆包全双工）在 max_response_ticks 之上追加的等待窗
# （150 tick ≈ 30s，合计 60s）。2026-09-08 真机实测：同一条 1.92s 用户音频，豆包首响在
# 1.7s 与 24.7s 之间跳（重尾，归因在服务端排队），30s 一到就报「Agent 无响应」并丢整通
# （mtLN_doubao 48 通失败里 17 通是这么死的）。刻意不做成 Settings 字段：它跟着**端点类别**
# 走（由 provider.needs_continuous_audio 声明），不是一个会被调参的实验口径。
STREAMING_GRACE_TICKS = 150


@dataclass
class TurnTrace:
    """一轮对话的完整轨迹。"""

    turn: int
    user_state: str
    user_text: str
    # 内部方案：实际送 TTS 的文本（可能带句首情感标签）。评委/ASR/词表一律看
    # user_text；本字段只供排查“听到的与读到的不一致”（标签会吞字，）。
    user_text_tts: str = ""
    user_audio_path: str = ""
    agent_text: str = ""
    agent_audio_path: str = ""
    agent_audio_pcm24_len: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    start_ts: float = 0.0
    end_ts: float = 0.0
    user_send_end_ts: float = 0.0 # 用户音频发送完成时刻（真实时间轴用）
    agent_first_audio_ts: float = 0.0 # 客服首个音频块到达时刻（真实时间轴用）
    tick_start: int = -1 # 本轮用户开口的 tick
    user_end_tick: int = -1 # 用户音频送完的 tick
    agent_first_audio_tick: int = -1 # 客服首个音频块播出的 tick
    tick_end: int = -1 # 本轮结束的 tick
    interrupted: bool = False # 本轮客服发言被用户打断（截断，任意来源）
    interrupt_source: str = "" # 打断来源：script（脚本标注打断）| vad（自然 VAD 截断）| 空（未打断）；
    # 两者并存时 script 优先（脚本打断即刻结束本轮，随后 VAD 截断仍归属该轮）
    timed_out: bool = False # 本轮等待客服响应超时
    # ---- 内部方案（v6.0）：骨架驱动模拟器协议字段（评测唯一关键轮定位来源）----
    is_key_turn: bool = False # 全场有且仅有一个 true（模拟器状态机保证）
    user_key_form: str = "" # 内部方案：关键轮句式（question/statement），仅关键轮非空
    user_tts_instruct: str = "" # 本轮用户侧 TTS instruct（确定性拼装或回落）
    covered: list[int] = field(default_factory=list) # 本轮 LLM 自报完成的 mission 要点序号
    forced_key_turn: bool = False # 预算强制的关键轮（hard-2 兜底，进质量报表）
    vocab_violation: bool = False # 词表防线重试后仍违规（不阻断，仅标记）
    state_mismatch: bool = False # 关键轮 state 与数据不符被覆盖
    end_call: bool = False # 本轮为道别收尾轮（其后模拟器 next_turn 返回 None）

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "user_state": self.user_state,
            "user_text": self.user_text,
            "user_text_tts": self.user_text_tts,
            "user_audio_path": self.user_audio_path,
            "agent_text": self.agent_text,
            "agent_audio_path": self.agent_audio_path,
            "agent_audio_pcm24_len": self.agent_audio_pcm24_len,
            "tool_calls": self.tool_calls,
            "tool_outputs": self.tool_outputs,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "user_send_end_ts": self.user_send_end_ts,
            "agent_first_audio_ts": self.agent_first_audio_ts,
            "tick_start": self.tick_start,
            "user_end_tick": self.user_end_tick,
            "agent_first_audio_tick": self.agent_first_audio_tick,
            "tick_end": self.tick_end,
            "interrupted": self.interrupted,
            "interrupt_source": self.interrupt_source,
            "timed_out": self.timed_out,
            "is_key_turn": self.is_key_turn,
            "user_key_form": self.user_key_form,
            "user_tts_instruct": self.user_tts_instruct,
            "covered": self.covered,
            "forced_key_turn": self.forced_key_turn,
            "vocab_violation": self.vocab_violation,
            "state_mismatch": self.state_mismatch,
            "end_call": self.end_call,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TurnTrace":
        known = {f for f in cls.__dataclass_fields__} # noqa: SLF001
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Trajectory:
    """一次场景运行的完整轨迹。"""

    scenario_id: str
    model: str
    role: str
    leakage_level: str = ""
    traces: list[TurnTrace] = field(default_factory=list)
    status: str = "ok" # ok | error | timeout
    error: str = ""
    termination_reason: str = "" # user_closed | script_exhausted(旧) | max_turns | timeout | error
    # | endpoint_session_closed（端点服务端掐断，见 run_scenario）
    key_turn_missing: bool = False # 内部方案：到达硬上限仍未出现唯一关键轮（评测侧跳过并计数）
    started_at: float = 0.0
    finished_at: float = 0.0
    seed: int | None = None
    interaction: dict[str, Any] = field(default_factory=dict) # 语音交互指标
    usage: dict[str, Any] = field(default_factory=dict) # Realtime 会话 token 用量（审计）
    # 上行流诊断（目前只有豆包：静音保活补发帧数）。放产物而不放日志是因为
    # 这类故障全静默、且仓内默认日志级别是 WARNING —— 没计数就无从自证。
    rt_diag: dict[str, Any] = field(default_factory=dict)
    sim_stats: dict[str, Any] = field(default_factory=dict) # 内部方案：模拟器协议健康度计数
    resumed: bool = False # checkpoint 载入的历史结果（不重复写盘）

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "model": self.model,
            "role": self.role,
            "leakage_level": self.leakage_level,
            "status": self.status,
            "error": self.error,
            "termination_reason": self.termination_reason,
            "key_turn_missing": self.key_turn_missing,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "seed": self.seed,
            "interaction": self.interaction,
            "usage": self.usage,
            "rt_diag": self.rt_diag,
            "sim_stats": self.sim_stats,
            "turns": [t.as_dict() for t in self.traces],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        traj = cls(
            scenario_id=d.get("scenario_id", ""),
            model=d.get("model", ""),
            role=d.get("role", ""),
            leakage_level=d.get("leakage_level", ""),
            status=d.get("status", "ok"),
            error=d.get("error", ""),
            termination_reason=d.get("termination_reason", ""),
            key_turn_missing=bool(d.get("key_turn_missing", False)),
            started_at=d.get("started_at", 0.0),
            finished_at=d.get("finished_at", 0.0),
            seed=d.get("seed"),
            interaction=d.get("interaction") or {},
        )
        traj.usage = d.get("usage") or {}
        traj.sim_stats = d.get("sim_stats") or {}
        traj.traces = [TurnTrace.from_dict(t) for t in d.get("turns", [])]
        return traj


# 跟在句末标点后面的闭合符（引号/括号）：判“说到句末”要先剔掉它们。
_SENTENCE_CLOSERS = "\"'”’)）】」』"


def _sentence_closed(text: str, sentence_end_chars: str) -> bool:
    """本轮已拿到的文本是否以句末标点收尾（即不是一句被截断的话）。

    只看**尾部**而不是“文本里含句号”：模型一口气说几句时，尾部才能区分
    “说到句尾”与“刚说完一句、下一句还在说”。尾部空文本 ⇒ False（还没开口）。
    """
    if not sentence_end_chars:
        return True # 端点未声明标点集 ⇒ 不做语义门，退回纯时间判据
    t = (text or "").strip()
    while t and t[-1] in _SENTENCE_CLOSERS:
        t = t[:-1].rstrip()
    return bool(t) and t[-1] in sentence_end_chars


def _stall_giveup_ticks(agent: Any) -> int:
    """本轮**始终没出声**时提前放弃的 tick 上限；0 = 不启用（沿用通用响应上限）。

    存在的理由是本端点的一种故障形态：连上了、也回了 session.created，但**全程零字节**
    （僵尸会话，实测出现在被掐断后的重试里）。这类会话不会自己活过来，等满默认上限
    （150 + 追加窗 ≈ 60 s）纯属把重试预算烧光。
    取 35 s 而不是更短：实测该端点"用户话音尾→客服首帧"p50=6.4 s 但 **p90=19.5 s**
    （正式批首几通实测），设 20 s 会误杀约一成正常轮。
    """
    return int(getattr(getattr(agent, "provider", None), "stalled_response_ticks", 0) or 0)


def _turn_end_policy(agent: Any, s: Any) -> tuple[bool, int, int, str]:
    """取该端点的判轮静默门：(产出侧计活动, 有标点所需 tick, 无标点所需 tick, 句末标点集)。

    为什么放在 provider 而不是 Settings：它跟着**端点类别**走（同一个全局口径下四个
    端点的节奏完全不同），不是会被调参的实验口径；且与 `needs_continuous_audio`、
    `_retry_backoff_s` 同一族先例。**未声明的端点必须逐字节保持原行为**：
    counts_output=False、两道门都退回 `settings.idle_end_ticks`、标点集为空。
    """
    prov = getattr(agent, "provider", None)
    default = int(s.idle_end_ticks)
    quiet = int(getattr(prov, "turn_end_quiet_ticks", 0) or default)
    return (bool(getattr(prov, "turn_end_counts_output_activity", False)),
            quiet,
            int(getattr(prov, "turn_end_open_ended_quiet_ticks", 0) or quiet),
            str(getattr(prov, "turn_end_sentence_end_chars", "") or ""))


class FullDuplexOrchestrator:
    """tick 级全双工编排器。"""

    def __init__(
        self,
        settings: Settings | None = None,
        artifact_dir: str | None = None,
        save_audio: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.artifact_dir = artifact_dir
        self.save_audio = save_audio
        self.tick_ms = self.settings.tick_ms

    async def run_scenario(
        self,
        scenario: dict[str, Any],
        agent,
        simulator: UserSimulator,
        max_turns: int | None = None,
    ) -> Trajectory:
        """运行一个场景脚本（tick 级全双工）。"""
        s = self.settings
        traj = Trajectory(
            scenario_id=scenario.get("scenario_id", "unknown"),
            model=agent.model_spec,
            role=scenario.get("role", ""),
            leakage_level=scenario.get("leakage_level", ""),
            started_at=time.time(),
            seed=getattr(simulator, "seed", None),
        )
        # 内部方案：硬预算进 max_turns（兜底）；缺省回落 settings.max_turns
        max_turns = max_turns or (scenario.get("user_script") or {}).get(
            "turn_budget", {}).get("hard") or s.max_turns
        tick_s = self.tick_ms / 1000
        chunk_samples = int(USER_SAMPLE_RATE * self.tick_ms / 1000)
        silence = b"\x00\x00" * chunk_samples
        max_total_ticks = int(s.max_tick_timeout_s / tick_s)
        # 单轮无响应上限（连续输入型端点追加宽限窗，见 STREAMING_GRACE_TICKS）
        response_wait_limit = self._response_wait_ticks(agent, s)

        # tick 时间线记录（对话音频合成 + 交互指标）
        tick_user_audio: list[bytes] = [] # 每 tick 用户 16k 块
        tick_agent_audio: list[bytes] = [] # 每 tick 客服 16k 块（重采样+填充）
        tick_records: list[dict[str, Any]] = []

        try:
            await agent.start()

            # ---- 首轮用户发言 ----
            turn_result = await asyncio.to_thread(simulator.next_turn, "", None)
            if turn_result is None:
                traj.termination_reason = (
                    getattr(simulator, "termination_reason", "") or "script_exhausted")
                traj.finished_at = time.time()
                return traj
            trace = self._new_trace(1, turn_result, tick_start=0)
            user_chunks = split_into_chunks(turn_result.pcm16, chunk_samples)
            user_pcm = turn_result.pcm16
            chunk_idx = 0
            agent_pcm = bytearray() # 本轮客服音频（24k PCM16）
            phase = "user" # user | agent | gap
            audio_ever_seen = False # 本轮是否收到过客服音频
            tools_delivered_tick = -1 # 工具结果投递的 tick（工具轮后续语音宽限起点）
            audio_before_tools = False # 工具结果投递时是否已出过声（残留缓冲判别用）
            quiet = 0
            wait_ticks = 0
            gap_left = 0
            interrupt_next = False
            tick = 0
            # 判轮静默门：端点可自己声明（不声明则沿用全局 idle_end_ticks，三家 realtime
            # 端点逐字节不变）。变量在会话内固定，故循环外算一次。
            te_counts_out, te_quiet, te_open_quiet, te_sent_chars = _turn_end_policy(agent, s)
            stall_giveup = _stall_giveup_ticks(agent)

            while tick < max_total_ticks:
                user_speaking = phase == "user" and chunk_idx < len(user_chunks)
                chunk = user_chunks[chunk_idx] if user_speaking else silence
                if user_speaking:
                    chunk_idx += 1

                r: AgentTickResult = await agent.run_tick(chunk, tick)

                # ---- token 熔断（防 Realtime 失控消耗）----
                session_tokens = getattr(agent, "session_usage", {}).get("total_tokens", 0)
                if session_tokens > s.max_session_tokens:
                    raise RuntimeError(
                        f"会话 token 超熔断上限：{session_tokens} > {s.max_session_tokens}"
                        f"（tick={tick}，疑似失控消耗，已终止）"
                    )

                # ---- 记录 ----
                if r.audio:
                    if trace.agent_first_audio_tick < 0:
                        trace.agent_first_audio_tick = tick
                        trace.agent_first_audio_ts = time.time()
                    agent_pcm += r.audio
                trace.agent_text += r.transcript
                trace.tool_calls.extend(r.tool_calls)
                trace.tool_outputs.extend(r.tool_outputs)
                if r.was_truncated:
                    # 截断事件归属于最近一个已完成的客服发言轮（打断发生在新用户轮）；
                    # 仅当该轮客服确实出过声才计打断——否则是“空响应轮 + 用户正常接话
                    # 触发 VAD”的误判（客服没说话，无所谓被打断）
                    target = trace if phase == "agent" else (traj.traces[-1] if traj.traces else trace)
                    if target.agent_first_audio_tick >= 0:
                        target.interrupted = True
                        if not target.interrupt_source: # 已被脚本打断的轮保持 script 归属
                            target.interrupt_source = "vad"
                tick_user_audio.append(chunk)
                tick_agent_audio.append(_agent_chunk_to_16k(r.audio, chunk_samples))
                tick_records.append({
                    "tick": tick,
                    "turn": trace.turn,
                    "phase": phase,
                    "user_speech": user_speaking,
                    "agent_bytes": len(r.audio),
                    "transcript": r.transcript,
                    "speech_started": r.speech_started,
                    "was_truncated": r.was_truncated,
                    "response_done": r.response_done,
                    "done_status": r.done_status,
                    "tool_calls": [tc["name"] for tc in r.tool_calls],
                    # 诊断字段（纯记录，不参与判轮）：出事时靠这三列才能分清
                    # 「端点没说」与「我方没听到」——之前的半句故障就是这么查的，
                    # 当时 ticks.json 里只有播出侧字节，只能靠推断。
                    "buffer_bytes": r.buffer_bytes,
                    "agent_output": r.agent_output,
                    "quiet_ticks": quiet,
                })

                # ---- 状态机 ----
                if phase == "user":
                    if chunk_idx >= len(user_chunks):
                        phase = "agent"
                        trace.user_end_tick = tick
                        trace.user_send_end_ts = time.time()
                        audio_ever_seen = False
                        tools_delivered_tick = -1
                        audio_before_tools = False
                        quiet = 0
                        wait_ticks = 0
                elif phase == "agent":
                    wait_ticks += 1
                    if r.tools_flushed:
                        tools_delivered_tick = tick # 工具结果已投递，模型随后可能续接语音
                        audio_before_tools = audio_ever_seen
                    if r.audio:
                        audio_ever_seen = True
                    if (r.audio or r.tool_calls or r.buffer_bytes > 0
                            or (te_counts_out and r.agent_output)):
                        quiet = 0 # 缓冲非空也算活动：避免音频逐 tick 封顶播出间隙被误判为静默
                        # 声明了 te_counts_out 的端点额外把**下行产出**也算活动：只看已播出
                        # 音频会漏掉「文字/音频已到但未播出」，对产出节奏慢的端点就是假静默。
                    else:
                        quiet += 1

                    turn_complete = False
                    interrupt_next = False
                    agent_started = trace.agent_first_audio_tick >= 0
                    # 轮结束判据（不依赖 response.done：done 只是单响应边界，一轮可跨多个
                    # response——工具调用响应结束后 provider 会 response.create 另起响应续接
                    # 语音）。缓冲排空且连续静默，再按工具投递情况分三类判可否结束：
                    # - 非工具轮：出过声即可（真无响应由 max_response_ticks 超时兜底）；
                    # - 纯工具轮（投递前未出声）：出过声（必为续接语音）∨ 宽限期耗尽（真无续接）；
                    # - 先说话再调工具：投递前残留缓冲的播出与续接语音无法区分，须等满宽限期，
                    # 以免续接语音未到就提前结束（与“迟到音频溢出到下一轮”原缺陷同类）。
                    # 本轮文本若未说到句末标点，走加严的门（宁等不切半句）；未声明该能力的
                    # 端点两个值相同 ⇒ 判据与原来逐字节一致。
                    need_quiet = (te_quiet if _sentence_closed(trace.agent_text, te_sent_chars)
                                  else te_open_quiet)
                    silence_idle = r.buffer_bytes == 0 and quiet >= need_quiet
                    grace_expired = (
                        tools_delivered_tick >= 0
                        and (tick - tools_delivered_tick) >= POST_TOOL_GRACE_TICKS
                    )
                    if tools_delivered_tick < 0:
                        can_end = audio_ever_seen
                    elif audio_before_tools:
                        can_end = grace_expired
                    else:
                        can_end = audio_ever_seen or grace_expired
                    if silence_idle and can_end:
                        turn_complete = True
                    elif (wait_ticks >= response_wait_limit
                          or (stall_giveup and not audio_ever_seen
                              and wait_ticks >= stall_giveup)):
                        # 后者只作用于**从未出声**的轮 ⇒ 正常轮（已出过声）的判据完全不变
                        await agent.cancel_response()
                        trace.timed_out = True
                        turn_complete = True
                    elif (
                        agent_started
                        and (tick - trace.agent_first_audio_tick) >= s.interrupt_delay_ticks
                        and simulator.peek_interrupt(trace.agent_text, trace.tool_calls)
                    ):
                        # 打断轮：不等客服说完，用户直接插话
                        trace.interrupted = True
                        trace.interrupt_source = "script"
                        turn_complete = True
                        interrupt_next = True

                    if turn_complete:
                        self._finalize_trace(trace, user_pcm, bytes(agent_pcm), tick)
                        if trace.timed_out and not trace.agent_text and not agent_pcm:
                            raise RuntimeError(
                                f"Agent 无响应：第 {trace.turn} 轮超时且无任何音频/转写"
                                # 报**实际等了多久**与**是哪道门判的死**：原先固定印
                                # `response_wait_limit`，于是端点自己声明的提前放弃门
                                # （stalled_response_ticks，本端点 35 s）生效后，日志里
                                # 仍写着"已等 60s"——读日志的人会据此判断"提前放弃没生效"，
                                # 而真相是三轮重试各 35 s（合计 122 s 全用在僵尸会话上）。
                                f"（实际等了 {wait_ticks * tick_s:.0f}s，上限 "
                                f"{response_wait_limit * tick_s:.0f}s，"
                                f"提前放弃门={'on' if stall_giveup else 'off'} "
                                f"{stall_giveup * tick_s:.0f}s）"
                            )
                        traj.traces.append(trace)

                        if len(traj.traces) >= max_turns:
                            traj.termination_reason = "max_turns"
                            # 内部方案：到达硬上限仍无唯一关键轮 → 记 key_turn_missing
                            if not any(t.is_key_turn for t in traj.traces):
                                traj.key_turn_missing = True
                            break
                        next_result = await asyncio.to_thread(
                            simulator.next_turn, trace.agent_text, trace.tool_calls
                        )
                        if next_result is None:
                            traj.termination_reason = (
                                getattr(simulator, "termination_reason", "")
                                or "script_exhausted")
                            if not any(t.is_key_turn for t in traj.traces):
                                traj.key_turn_missing = True
                            break
                        trace = self._new_trace(len(traj.traces) + 1, next_result, tick_start=tick + 1)
                        user_chunks = split_into_chunks(next_result.pcm16, chunk_samples)
                        user_pcm = next_result.pcm16
                        chunk_idx = 0
                        agent_pcm = bytearray()
                        audio_ever_seen = False
                        tools_delivered_tick = -1
                        audio_before_tools = False
                        if interrupt_next:
                            phase = "user" # 打断：无轮间隙，立即开口
                        else:
                            phase = "gap"
                            gap_left = s.turn_gap_ticks
                else: # gap
                    gap_left -= 1
                    if gap_left <= 0:
                        phase = "user"

                tick += 1
            else:
                traj.status = "timeout"
                traj.termination_reason = "timeout"
                logger.error("场景 %s 超过总时长上限 %.0fs", traj.scenario_id, s.max_tick_timeout_s)

        except EndpointSessionClosed as exc:
            # 端点服务端单方面掐断会话（某全双工端点实测 91–146s 必发
            # session.closed reason=backend_error，与轮数/输出量无关，我方无参数可延长）。
            # 关键轮若已完整跑完，这通的主读数就已经拿到了 —— 判失败会连带三件坏事：
            # ① 触发重试，而重试必然撞上服务端尚未释放的僵尸会话（连上却全程零字节）；
            # ② 重试的失败产物覆盖掉这一次已经跑出来的音频与 ticks（成功那次的证据丢失）；
            # ③ 该场景最终 0 条可评测轨迹。⇒ 关键轮齐全即留用，并留下可辨识的终止原因，
            # 便于分析侧把「被掐断的通话」单独筛出来（它缺后续轮 ⇒ 全程类指标不可与完整通话并比）。
            traj.error = f"{type(exc).__name__}: {exc}"
            traj.termination_reason = "endpoint_session_closed"
            if self._key_turn_captured(traj, _turn_end_policy(agent, s)[3]):
                traj.status = "ok"
                logger.warning("场景 %s 被端点掐断，但关键轮已齐全 ⇒ 留用: %s",
                               traj.scenario_id, traj.error)
            else:
                traj.status = "error"
                traj.key_turn_missing = True
                logger.error("场景 %s 被端点掐断且关键轮未跑完 ⇒ 判失败: %s",
                             traj.scenario_id, traj.error)
        except Exception as exc: # noqa: BLE001
            traj.status = "error"
            traj.termination_reason = "error"
            traj.error = f"{type(exc).__name__}: {exc}"
            logger.error("场景 %s 运行失败: %s", traj.scenario_id, traj.error)
        finally:
            await agent.stop()
            traj.finished_at = time.time()
            traj.usage = getattr(agent, "session_usage", {}) or {}
            traj.rt_diag = agent.stream_diag() if hasattr(agent, "stream_diag") else {}
            if traj.usage.get("total_tokens"):
                logger.info(
                    "场景 %s Realtime 用量: total=%d (in=%d, out=%d, responses=%d)",
                    traj.scenario_id, traj.usage.get("total_tokens", 0),
                    traj.usage.get("input_tokens", 0), traj.usage.get("output_tokens", 0),
                    traj.usage.get("responses", 0),
                )

        # ---- 产物：tick 时间线 / 对话音频 / 交互指标 ----
        traj.interaction = compute_interaction_metrics(traj.traces, tick_s)
        if self.artifact_dir:
            self._save_ticks(tick_records)
            if self.save_audio and tick_user_audio:
                try:
                    from listen2serve.audio.conversation_audio import write_tick_conversation

                    write_tick_conversation(
                        tick_user_audio, tick_agent_audio, traj.traces,
                        self.artifact_dir, tick_s=tick_s,
                    )
                except Exception as exc: # noqa: BLE001 - 附加产物失败不阻断
                    logger.warning("对话音频合成失败: %s", exc)
        return traj

    # ---- 内部 ----
    @staticmethod
    @staticmethod
    def _key_turn_captured(traj: Trajectory, sentence_chars: str = "") -> bool:
        """关键轮是否已拿到**可用**的被测回复（判据与 cli._build_eval_tasks 的入池条件一致）。

        `sentence_chars` 非空时额外要求"说到句末"：这是给**被端点掐断**的通话准备的。
        2026-09-11 全量批实测：4 通被判"关键轮已齐全"里 2 通的关键轮文本只有「嗯」
        「嗯好」——会话正好在服务端掐断于被测刚开口的瞬间。当时的留用判据只看"非空"，
        于是这种通话既留用（不重试）又被送裁判 ⇒ **把『开口即断』记成『能力差』**。
        被掐断的通话没有"静默=说完"这类时间证据，只能认语言完整性。
        未声明标点集的端点（三家）走 `sentence_chars=""` ⇒ 判据与原来逐字节一致。
        """
        keys = [t for t in traj.traces if getattr(t, "is_key_turn", False)]
        if len(keys) != 1:
            return False
        text = (keys[0].agent_text or "").strip()
        if not text:
            return False
        return _sentence_closed(text, sentence_chars)


    def _new_trace(self, turn: int, turn_result, tick_start: int) -> TurnTrace:
        return TurnTrace(
            turn=turn,
            user_state=turn_result.state,
            user_text=turn_result.text,
            user_text_tts=getattr(turn_result, "text_tts", "") or turn_result.text,
            start_ts=time.time(),
            tick_start=tick_start,
            # 内部方案（v6.0）协议字段（旧模拟器无这些属性时回落默认值）
            is_key_turn=bool(getattr(turn_result, "is_key_turn", False)),
            user_key_form=str(getattr(turn_result, "key_form", "") or ""),
            user_tts_instruct=getattr(turn_result, "tts_control", "") or "",
            covered=list(getattr(turn_result, "covered", []) or []),
            forced_key_turn=bool(getattr(turn_result, "forced_key_turn", False)),
            vocab_violation=bool(getattr(turn_result, "vocab_violation", False)),
            state_mismatch=bool(getattr(turn_result, "state_mismatch", False)),
            end_call=bool(getattr(turn_result, "end_call", False)),
        )
    @staticmethod
    def _response_wait_ticks(agent, s) -> int:
        """单轮等待 Agent 响应的上限 tick 数（按端点类别，见 STREAMING_GRACE_TICKS）。"""
        base = int(s.max_response_ticks)
        return (base + STREAMING_GRACE_TICKS
                if getattr(agent, "needs_continuous_audio", False) else base)

    def _finalize_trace(self, trace: TurnTrace, user_pcm: bytes, agent_pcm: bytes, tick: int) -> None:
        trace.end_ts = time.time()
        trace.tick_end = tick
        trace.agent_audio_pcm24_len = len(agent_pcm)
        if self.save_audio and self.artifact_dir:
            trace.user_audio_path = self._save_audio(
                f"turn{trace.turn}_user.wav", user_pcm, USER_SAMPLE_RATE
            )
            agent_pcm16 = agent_audio_to_pcm16(agent_pcm)
            trace.agent_audio_path = self._save_audio(
                f"turn{trace.turn}_agent.wav", agent_pcm16, USER_SAMPLE_RATE
            )

    def _save_audio(self, filename: str, pcm16: bytes, sample_rate: int) -> str:
        from listen2serve.audio.io_utils import save_wav

        path = Path(self.artifact_dir) / "audio" / filename
        save_wav(path, pcm16, sample_rate)
        return str(path)

    def _save_ticks(self, tick_records: list[dict[str, Any]]) -> None:
        import json

        path = Path(self.artifact_dir) / "ticks.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tick_records, ensure_ascii=False), encoding="utf-8")


def compute_interaction_metrics(traces: list[TurnTrace], tick_s: float) -> dict[str, Any]:
    """语音交互客观指标（tick 时间轴，方案 8.4）。

    - response_latency_mean_s：用户说完到客服首个音频块的平均时延（仿真时间；
      仅统计客服在用户说完后才开口的轮）
    - overlap_start_turns：客服在用户说完前已开口的轮数（全双工重叠，如工具
      结果续接响应跨轮播出），不计入时延均值
    - response_rate：得到客服语音回应的用户轮占比
    - interrupted_turns：被打断的轮数（合计口径，任意来源）；按来源拆分见
      script_interrupted_turns（脚本标注打断）与 vad_truncated_turns（自然 VAD 截断），
      来源互斥故合计 = 两项之和
    - timed_out_turns：超时的轮数
    """
    latencies: list[float] = []
    overlap_starts = 0
    responded = 0
    for t in traces:
        if t.agent_first_audio_tick >= 0:
            responded += 1
            if t.user_end_tick >= 0:
                lat = (t.agent_first_audio_tick - t.user_end_tick) * tick_s
                if lat >= 0:
                    latencies.append(lat)
                else:
                    overlap_starts += 1
    n = len(traces)
    return {
        "turns": n,
        "response_rate": round(responded / n, 4) if n else 0.0,
        "response_latency_mean_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "response_latency_max_s": round(max(latencies), 3) if latencies else None,
        "overlap_start_turns": overlap_starts,
        "interrupted_turns": sum(1 for t in traces if t.interrupted),
        "script_interrupted_turns": sum(1 for t in traces if t.interrupt_source == "script"),
        "vad_truncated_turns": sum(1 for t in traces if t.interrupt_source == "vad"),
        "timed_out_turns": sum(1 for t in traces if t.timed_out),
        "sim_duration_s": round((traces[-1].tick_end + 1) * tick_s, 2) if traces and traces[-1].tick_end >= 0 else 0.0,
    }


def _agent_chunk_to_16k(pcm24k: bytes, chunk_samples_16k: int) -> bytes:
    """客服 tick 音频（24k PCM16）→ 16k 定长块（重采样 + 截断/补零）。"""
    if not pcm24k:
        return b"\x00\x00" * chunk_samples_16k
    import numpy as np

    from listen2serve.audio.io_utils import resample_audio

    arr = np.frombuffer(pcm24k, dtype=np.int16)
    arr = resample_audio(arr, AGENT_SAMPLE_RATE, USER_SAMPLE_RATE)
    if len(arr) >= chunk_samples_16k:
        arr = arr[:chunk_samples_16k]
    else:
        arr = np.concatenate([arr, np.zeros(chunk_samples_16k - len(arr), dtype=np.int16)])
    return arr.astype(np.int16).tobytes()


def agent_audio_to_pcm16(pcm: bytes, src_rate: int = 24000, dst_rate: int = 16000) -> bytes:
    """Qwen Realtime 输出音频 → PCM16 16kHz。

    实测（2026-08-05）：`qwen-audio-3.0-realtime-plus` 的 response.audio.delta 解码后
    实际为 **PCM16（16-bit 小端）24kHz**（tau3 注释中的 PCM24 与实测不符）。
    按 16-bit 解析后语音正常；若数据长度为 3 的倍数且按 16-bit 解析出噪音，
    可切换到 24-bit 路径（保留以兼容其他模型）。
    """
    if not pcm:
        return b""
    if len(pcm) % 3 == 0 and _looks_like_pcm24(pcm):
        pcm = _pcm24_to_pcm16(pcm, src_rate, dst_rate)
        return pcm
    # 16-bit 路径（实测默认）
    pcm16 = pcm
    if src_rate != dst_rate:
        import numpy as np

        from listen2serve.audio.io_utils import resample_audio

        arr = np.frombuffer(pcm16, dtype=np.int16)
        arr = resample_audio(arr, src_rate, dst_rate)
        pcm16 = arr.tobytes()
    return pcm16


def _looks_like_pcm24(pcm: bytes) -> bool:
    """启发式判断是否为 24-bit 数据：三字节列中高字节应变化平缓（语音特征）。

    24-bit 小端采样：低字节（第 0 字节）变化频繁，高字节（第 2 字节）变化平缓；
    16-bit 数据按 3 字节分组时三列统计特征近似（即 24-bit 假设不成立）。
    """
    import numpy as np

    n = len(pcm) // 3
    if n < 100:
        return False
    arr = np.frombuffer(pcm[: n * 3], dtype=np.uint8).reshape(n, 3)
    diff_low = np.abs(np.diff(arr[:, 0].astype(np.int32))).mean()
    diff_high = np.abs(np.diff(arr[:, 2].astype(np.int32))).mean()
    return diff_high < diff_low * 0.5


def _pcm24_to_pcm16(pcm24: bytes, src_rate: int = 24000, dst_rate: int = 16000) -> bytes:
    """PCM24 → PCM16（取高 16 位）+ 重采样到 16kHz。"""
    if not pcm24:
        return b""
    n = len(pcm24) // 3
    out = bytearray(n * 2)
    for i in range(n):
        b0, b1, b2 = pcm24[i * 3], pcm24[i * 3 + 1], pcm24[i * 3 + 2]
        sample = (b2 << 16) | (b1 << 8) | b0
        if sample & 0x800000:
            sample -= 0x1000000
        s16 = sample >> 8
        out[i * 2] = s16 & 0xFF
        out[i * 2 + 1] = (s16 >> 8) & 0xFF
    pcm16 = bytes(out)
    if src_rate != dst_rate:
        import numpy as np

        from listen2serve.audio.io_utils import resample_audio

        arr = np.frombuffer(pcm16, dtype=np.int16)
        arr = resample_audio(arr, src_rate, dst_rate)
        pcm16 = arr.tobytes()
    return pcm16
