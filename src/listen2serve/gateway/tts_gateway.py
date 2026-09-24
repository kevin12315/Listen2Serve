"""TTS 网关：dashscope qwen-audio-3.0-tts-plus / CosyVoice 参数切换。

统一输出 16k PCM16（mono），由调用方决定是否写盘为 WAV。
指令控制字段（2026-08-26 修正，官方文档 + 真机双向验证）：SpeechSynthesizer 端点
（Qwen-Audio-TTS / CosyVoice 系）的指令字段是 `input.instruction`（单数）；
`instructions`（复数）是 Qwen3-TTS 系 multimodal-generation 端点的字段名，打到本端点
会被后端**静默忽略**（HTTP 200 且音频与不带指令的基线逐字节相同）。
设计约束：评测通路**不做自动回退**——指定后端失败即抛错、该样本判定为失败；
备选后端通过 `backend` 参数或 `settings.tts_backend` 显式切换：
  - dashscope：qwen-audio-3.0-tts-plus（主要 TTS，情感 instruct + prosody 控制）
    data 或 audio 字段；超时 300s，失败重试一次后仍失败即抛错，无后端间回退）
  - cosyvoice：DashScope CosyVoice v2（克隆音色场景）
"""

from __future__ import annotations

import base64
import io
import logging
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import httpx

from listen2serve.gateway.base import CallRecord, CallSink, MemoryCallSink, prompt_hash
from listen2serve.model_registry import voice_for
from listen2serve.runtime.config import Settings, get_settings

logger = logging.getLogger(__name__)

# seed-tts 火山协议成功码（0=同步成功，3000=服务内部排队后成功）
SEEDTTS_SUCCESS_CODES = (0, 3000, "0", "3000")
SEEDTTS_TIMEOUT_S = 300.0
SEEDTTS_RETRIES = 1 # 失败重试一次（共最多 2 次请求）

# 支持指令控制（input.instruction）的 dashscope 模型前缀：
# Qwen-Audio-TTS 全系（系统音色/复刻音色均可任意指令）、CosyVoice v3 及以上；
# cosyvoice-v2 不在官方支持清单内，实测传 instruction 直接 HTTP 400（Engine 428），
# 故对不支持的模型显式丢弃指令并 warning 留痕，避免整条请求失败。
INSTRUCT_CAPABLE_PREFIXES = ("qwen-audio-3.0-tts-", "cosyvoice-v3")


def supports_instruct(model: str) -> bool:
    """该 TTS 模型是否支持 input.instruction 指令控制。"""
    return any(model.startswith(p) for p in INSTRUCT_CAPABLE_PREFIXES)


@dataclass
class TTSResult:
    audio: bytes # PCM16 16k mono（无 WAV 头）
    sample_rate: int
    latency_ms: float = 0.0
    backend: str = ""
    record: CallRecord | None = field(default=None, repr=False)


class TTSGateway:
    """TTS 网关：按指定后端合成；失败即抛错（无自动回退）。"""

    def __init__(self, settings: Settings | None = None, sink: CallSink | None = None) -> None:
        self.settings = settings or get_settings()
        self.sink = sink or MemoryCallSink()

    # ---- 公开入口 ----
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        style: str | None = None,
        prosody: dict[str, Any] | None = None,
        backend: str | None = None,
    ) -> TTSResult:
        """合成语音。

        Args:
            text: 待合成文本
            voice: 音色 ID（None 用配置默认）
            style: 情感/风格 instruct 描述（如 "用不耐烦的语气"），下发为 input.instruction
            prosody: 韵律参数。**`speed_ratio` 仅部分后端消费**；
                dashscope 路径不下发任何韵律字段（官方本有 rate[0.5,2.0]/
                volume[0,100]/pitch[0.5,2.0]，但评测侧韵律统一由 instruction 文本描述
                承载，不与数值字段叠加）。注意官方 volume 是 0–100 刻度（默认 50）、
                不是倍率，将来若要接线必须先做单位换算。
            backend: 指定后端（dashscope|cosyvoice）；None 用 settings.tts_backend
        """
        bk = backend or self.settings.tts_backend
        if bk == "dashscope":
            return self._synth_dashscope(text, voice, style, prosody)
        if bk == "cosyvoice":
            return self._synth_dashscope(text, voice, style, prosody, force_cosyvoice=True)
        raise ValueError(f"未知 TTS 后端: {bk}（可选 dashscope|cosyvoice）")

    # ---- dashscope（qwen-audio-3.0-tts-plus / CosyVoice）----
    def _synth_dashscope(
        self,
        text: str,
        voice: str | None,
        style: str | None,
        prosody: dict[str, Any] | None,
        force_cosyvoice: bool = False,
    ) -> TTSResult:
        s = self.settings
        model = "cosyvoice-v2" if force_cosyvoice else s.tts_model
        input_payload: dict[str, Any] = {
            "text": text,
            # 音色优先级：显式参数 > .env 临时覆盖 > 模型绑定表（model_registry）
            "voice": voice or s.tts_voice_user or voice_for(model) or "longanhuan_v3.6",
            "format": "wav",
            "sample_rate": s.tts_sample_rate,
        }
        # 指令控制：字段名为 input.instruction（单数，见模块 docstring）
        if style:
            if supports_instruct(model):
                input_payload["instruction"] = style
            else:
                logger.warning("模型 %s 不支持 instruct，已丢弃指令（%d 字）", model, len(style))
        body = {
            "model": model,
            "input": input_payload,
        }
        ph = prompt_hash(text)
        start = time.perf_counter()
        resp = httpx.post(
            s.dashscope_tts_url,
            headers={"Authorization": f"Bearer {s.dashscope_api_key or ''}"},
            json=body,
            timeout=120,
        )
        latency = (time.perf_counter() - start) * 1000
        if resp.status_code != 200:
            detail = resp.text[:500]
            self.sink.append(CallRecord(service="tts", backend="dashscope", model=model,
                                        prompt_hash=ph, latency_ms=latency, error=f"HTTP {resp.status_code}: {detail}"))
            raise RuntimeError(f"dashscope TTS HTTP {resp.status_code}: {detail}")
        data = resp.json()
        if data.get("code") not in (None, 0, "0") and data.get("code") != 200:
            self.sink.append(CallRecord(service="tts", backend="dashscope", model=model,
                                        prompt_hash=ph, latency_ms=latency, error=str(data)[:500]))
            raise RuntimeError(f"dashscope TTS error: {data}")
        # 响应可能直接是 WAV 二进制，也可能在 output.audio 字段
        audio = _extract_audio_bytes(data, resp.content)
        self.sink.append(CallRecord(service="tts", backend="dashscope", model=model,
                                    prompt_hash=ph, latency_ms=latency))
        return TTSResult(audio=_wav_to_pcm16(audio), sample_rate=s.tts_sample_rate,
                         latency_ms=latency, backend="dashscope")



# ---- 工具函数 ----
def _extract_audio_bytes(data: dict[str, Any], raw: bytes) -> bytes:
    """从响应中提取音频字节：兼容 binary 响应、JSON 包装与 OSS URL。"""
    if raw and not data:
        return raw
    if isinstance(data, dict):
        out = data.get("output") or data.get("data") or {}
        if isinstance(out, dict):
            aud = out.get("audio") or {}
            if isinstance(aud, dict):
                if aud.get("data"):
                    return _b64decode_or_raw(aud["data"])
                url = aud.get("url")
                if url:
                    resp = httpx.get(url, timeout=60)
                    if resp.status_code == 200:
                        return resp.content
                    raise RuntimeError(f"音频 URL 下载失败 HTTP {resp.status_code}")
            elif isinstance(aud, str):
                return _b64decode_or_raw(aud)
            url = out.get("audio_url") or out.get("url")
            if url:
                return _b64decode_or_raw(url)
    return raw


def _b64decode_or_raw(value: str) -> bytes:
    try:
        return base64.b64decode(value)
    except Exception: # noqa: BLE001 - 非 base64 则按原始字节
        return value.encode("latin-1")


def _seedtts_wav_to_pcm16(wav_bytes: bytes, target_rate: int) -> bytes:
    """seed-tts 响应 WAV → PCM16；采样率与目标不一致时线性插值重采样。"""
    pcm, rate = _wav_bytes_to_pcm(wav_bytes)
    if rate != target_rate:
        import numpy as np

        from listen2serve.audio.io_utils import resample_audio

        arr = np.frombuffer(pcm, dtype=np.int16)
        pcm = resample_audio(arr, rate, target_rate).tobytes()
    return pcm


def _wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """WAV 字节 → (PCM16 字节, 采样率)；非 WAV 输入按 16k 原始 PCM 处理。"""
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
        return wav_bytes, 16000
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        if wf.getnchannels() > 1:
            import numpy as np

            arr = np.frombuffer(data, dtype=np.int16).reshape(-1, wf.getnchannels())
            data = arr.mean(axis=1).astype(np.int16).tobytes()
        return data, rate


def _wav_to_pcm16(wav_bytes: bytes) -> bytes:
    """WAV → PCM16；若输入已是非 WAV 原始数据则直接返回。"""
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
        return wav_bytes
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getsampwidth() == 2
            return wf.readframes(wf.getnframes())
    except (wave.Error, AssertionError):
        return wav_bytes
