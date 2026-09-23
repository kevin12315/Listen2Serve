"""音频 IO 工具：WAV/PCM 读写、格式转换、声道/采样率处理。"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np


def pcm16_to_wav(pcm: bytes, sample_rate: int = 16000, n_channels: int = 1) -> bytes:
    """PCM16 → WAV 字节流。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def save_wav(path: str | Path, pcm: bytes, sample_rate: int = 16000) -> None:
    """PCM16 写 WAV 文件。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(pcm16_to_wav(pcm, sample_rate))


def load_wav_pcm(path: str | Path) -> tuple[np.ndarray, int]:
    """读 WAV → (int16 数组, 采样率)。"""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels()).mean(axis=1).astype(np.int16)
    return data, sr


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    """PCM16 → float32 [-1, 1]。"""
    arr = np.frombuffer(pcm, dtype=np.int16)
    return arr.astype(np.float32) / 32768.0


def float32_to_pcm(f32: np.ndarray) -> bytes:
    """float32 [-1,1] → PCM16。"""
    arr = np.clip(f32, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def resample_audio(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """线性插值重采样（电话语音用途足够）。"""
    if src_rate == dst_rate:
        return data
    n_out = int(round(len(data) * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, len(data))
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, data).astype(data.dtype)


def split_into_chunks(pcm: bytes, chunk_samples: int) -> list[bytes]:
    """按采样数切块（末块不足时补零）。"""
    n = len(pcm) // 2
    if n == 0:
        return []
    if n % chunk_samples:
        pad = chunk_samples - (n % chunk_samples)
        pcm = pcm + b"\x00\x00" * pad
    return [pcm[i : i + chunk_samples * 2] for i in range(0, len(pcm), chunk_samples * 2)]
