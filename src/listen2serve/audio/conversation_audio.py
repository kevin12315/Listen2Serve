"""对话音频合成：对话双方多轮音频 → 单个音频文件（仿真时间模型）。

借鉴 tau-voice（MIT © Sierra）的最终实现形态，**仿真时间与真实时间解耦**：
- 说话时长 = 音频真实播放时长（仿真时间随说话推进）；
- 推理/网络延迟 = **不计入仿真时间**（虚拟时钟在推理时暂停）；
- 轮内/轮间停顿 = 固定自然间隙（模拟真人反应时间），与 tau-voice tick 时间线中
  的协议间隙语义一致（tau-voice 实测客服-用户停顿较短，因其录音不含推理延迟）。

产物：
- `both.wav`：双声道时间轴（左=用户，右=客服），tau-voice 同款（分析用，可重叠）
- `conversation.wav`：单声道时间线（电话听感）
- `segments.txt`：Audacity 标签：秒 → 角色 → 文本
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
AGENT_REACTION_GAP_S = 0.3 # 客服反应间隙（仿真时间：推理不计入，固定自然停顿）
TURN_GAP_S = 0.3 # 轮间间隙


@dataclass
class Segment:
    """一段语音（一轮中的一方），含真实时间信息。"""

    role: str # user | agent
    turn: int
    text: str
    pcm16: np.ndarray
    start_s: float = 0.0 # 在对话时间轴上的起点（秒）


def _load_pcm16(path: str | None) -> np.ndarray:
    """读取 WAV → int16 数组（16k mono，必要时重采样/混单声道）。"""
    if not path or not Path(path).exists():
        return np.array([], dtype=np.int16)
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels()).mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        from listen2serve.audio.io_utils import resample_audio

        data = resample_audio(data, sr, SAMPLE_RATE)
    return data


def _ts(trace, key: str) -> float | None:
    """安全读取轨迹时间戳（兼容对象与 dict）。"""
    v = getattr(trace, key, None) if not isinstance(trace, dict) else trace.get(key)
    return float(v) if v else None


def _build_timeline(traces: list) -> list[Segment]:
    """按仿真时间轴放置语音段（推理不计时）。

    时间轴规则（半双工）：
    - 轮内：用户说话（音频真实时长）→ 客服反应间隙（固定 0.3s，模拟真人反应）→ 客服说话（音频真实时长）
    - 轮间：固定间隙（0.3s）
    """
    segments: list[Segment] = []
    t = 0.0

    for trace in traces:
        user_pcm = _load_pcm16(_ts_audio(trace, "user_audio_path"))
        agent_pcm = _load_pcm16(_ts_audio(trace, "agent_audio_path"))

        # ---- 用户段 ----
        if user_pcm.size:
            text = getattr(trace, "user_text", None) if not isinstance(trace, dict) else trace.get("user_text", "")
            segments.append(Segment("user", _turn(trace), text or "", user_pcm, start_s=t))
            t += len(user_pcm) / SAMPLE_RATE

        # ---- 客服反应间隙（推理延迟不计入仿真时间）----
        if user_pcm.size and agent_pcm.size:
            t += AGENT_REACTION_GAP_S

        # ---- 客服段 ----
        if agent_pcm.size:
            text = getattr(trace, "agent_text", None) if not isinstance(trace, dict) else trace.get("agent_text", "")
            segments.append(Segment("agent", _turn(trace), text or "", agent_pcm, start_s=t))
            t += len(agent_pcm) / SAMPLE_RATE

        # ---- 轮间间隙 ----
        if agent_pcm.size:
            t += TURN_GAP_S

    return segments


def _ts_audio(trace, key: str) -> str | None:
    v = getattr(trace, key, None) if not isinstance(trace, dict) else trace.get(key)
    return str(v) if v else None


def _turn(trace) -> int:
    v = getattr(trace, "turn", None) if not isinstance(trace, dict) else trace.get("turn")
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _write_wav(path: Path, pcm16: np.ndarray, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        if channels == 1:
            wf.writeframes(pcm16.astype(np.int16).tobytes())
        else:
            interleaved = np.empty(pcm16.shape[0] * 2, dtype=np.int16)
            interleaved[0::2] = pcm16[:, 0]
            interleaved[1::2] = pcm16[:, 1]
            wf.writeframes(interleaved.tobytes())


def synthesize_conversation_audio(traces: list, output_dir: str | Path) -> dict[str, Path]:
    """按真实时间轴合成并保存对话音频，返回产物路径映射。"""
    output_dir = Path(output_dir)
    segments = _build_timeline(traces)
    if not segments:
        raise ValueError("无音频片段可合成")

    total_s = max((s.start_s + len(s.pcm16) / SAMPLE_RATE) for s in segments)
    total_n = int(np.ceil(total_s * SAMPLE_RATE))

    # ---- both.wav：双声道时间轴（左=用户，右=客服）----
    both = np.zeros((total_n, 2), dtype=np.int16)
    for s in segments:
        start = int(s.start_s * SAMPLE_RATE)
        end = min(start + len(s.pcm16), total_n)
        ch = 0 if s.role == "user" else 1
        both[start:end, ch] = s.pcm16[: end - start]
    both_path = output_dir / "both.wav"
    _write_wav(both_path, both, channels=2)

    # ---- conversation.wav：单声道时间线（电话听感）----
    conv = np.zeros(total_n, dtype=np.int16)
    for s in segments:
        start = int(s.start_s * SAMPLE_RATE)
        end = min(start + len(s.pcm16), total_n)
        conv[start:end] = s.pcm16[: end - start]
    conv_path = output_dir / "conversation.wav"
    _write_wav(conv_path, conv, channels=1)

    # ---- Audacity 风格标签（秒 → 角色 → 文本）----
    labels_path = output_dir / "segments.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        for s in segments:
            dur = len(s.pcm16) / SAMPLE_RATE
            f.write(f"{s.start_s:.2f}\t{s.start_s + dur:.2f}\t[{s.role}] turn{s.turn}: {s.text}\n")

    return {"both": both_path, "conversation": conv_path, "segments": labels_path}


def write_tick_conversation(
    tick_user_audio: list[bytes],
    tick_agent_audio: list[bytes],
    traces: list,
    output_dir: str | Path,
    tick_s: float = 0.2,
) -> dict[str, Path]:
    """tick 级全双工时间轴合成（tau-voice 同款：无间隙拼接双声道，可重叠）。

    Args:
        tick_user_audio: 每 tick 用户 16k PCM16 定长块（说话/静音）
        tick_agent_audio: 每 tick 客服 16k PCM16 定长块（已重采样+填充）
        traces: TurnTrace 列表（tick 索引用于生成标签）
        tick_s: tick 时长（秒）
    """
    output_dir = Path(output_dir)
    if not tick_user_audio:
        raise ValueError("无 tick 音频可合成")
    user = np.frombuffer(b"".join(tick_user_audio), dtype=np.int16)
    agent = np.frombuffer(b"".join(tick_agent_audio), dtype=np.int16)
    n = max(len(user), len(agent))
    both = np.zeros((n, 2), dtype=np.int16)
    both[: len(user), 0] = user
    both[: len(agent), 1] = agent
    both_path = output_dir / "both.wav"
    _write_wav(both_path, both, channels=2)

    # 单声道：双方混音（打断时真实重叠），裁剪防溢出
    conv = np.clip(
        both[:, 0].astype(np.int32) + both[:, 1].astype(np.int32), -32768, 32767
    ).astype(np.int16)
    conv_path = output_dir / "conversation.wav"
    _write_wav(conv_path, conv, channels=1)

    # ---- Audacity 风格标签（tick 时间轴，秒）----
    labels_path = output_dir / "segments.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        for t in traces:
            tick_start = _attr(t, "tick_start")
            user_end = _attr(t, "user_end_tick")
            if tick_start is not None and tick_start >= 0 and user_end is not None and user_end >= 0:
                f.write(
                    f"{tick_start * tick_s:.2f}\t{(user_end + 1) * tick_s:.2f}\t"
                    f"[user] turn{_attr(t, 'turn')}: {_attr(t, 'user_text') or ''}\n"
                )
            a_start = _attr(t, "agent_first_audio_tick")
            a_end = _attr(t, "tick_end")
            if a_start is not None and a_start >= 0 and a_end is not None and a_end >= 0:
                # 打断标记带来源（script=脚本打断 / vad=自然 VAD 截断）；白名单外/旧轨迹无来源时回落通用标记
                src = _attr(t, "interrupt_source")
                if src not in ("script", "vad"):
                    src = ""
                mark = (f" [interrupted:{src}]" if src else " [interrupted]") if _attr(t, "interrupted") else ""
                f.write(
                    f"{a_start * tick_s:.2f}\t{(a_end + 1) * tick_s:.2f}\t"
                    f"[agent] turn{_attr(t, 'turn')}{mark}: {_attr(t, 'agent_text') or ''}\n"
                )

    return {"both": both_path, "conversation": conv_path, "segments": labels_path}


def _attr(trace, key: str):
    return trace.get(key) if isinstance(trace, dict) else getattr(trace, key, None)
