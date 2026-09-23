"""语音复杂度管线：电话信道滤波、背景噪声、丢帧、闷音、vocal tics。

论文主干使用 `control`（干净）+ 可选 `regular`（电话信道 + 噪声 + 丢帧）。
输入/输出均为 int16 数组。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from listen2serve.audio.io_utils import load_wav_pcm

# 电话信道带通（300-3400Hz）
TELEPHONE_BAND = (300.0, 3400.0)


@dataclass
class EffectConfig:
    """音频效果配置。"""

    phone_channel: bool = True # 电话带宽滤波
    noise_snr_db: float | None = 15.0 # None = 不加噪声
    packet_loss_rate: float = 0.0 # 丢帧率（Gilbert-Elliott）
    packet_loss_burst: float = 0.0 # Gilbert-Elliott 突发概率
    packet_ms: int = 20 # 丢帧粒度
    muffled: bool = False # 闷音（低通 2kHz）
    vocal_tics: bool = False # 添加 tic 类杂音
    seed: int = 0


def apply_effects(
    pcm: np.ndarray,
    sample_rate: int,
    cfg: EffectConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """按配置依次施加效果。"""
    cfg = cfg or EffectConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    out = pcm.astype(np.float64)

    if cfg.phone_channel:
        out = _bandpass(out, sample_rate, TELEPHONE_BAND[0], TELEPHONE_BAND[1])
    if cfg.muffled:
        out = _bandpass(out, sample_rate, 0.0, 2000.0)
    if cfg.noise_snr_db is not None:
        out = _add_noise(out, cfg.noise_snr_db, rng)
    if cfg.packet_loss_rate > 0:
        out = _packet_loss(out, sample_rate, cfg, rng)
    if cfg.vocal_tics:
        out = _add_ticks(out, sample_rate, rng)

    return np.clip(out, -32768, 32767).astype(np.int16)


def _bandpass(data: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    nyq = sr / 2.0
    lo = max(lo, 1.0)
    hi = min(hi, nyq * 0.99)
    if lo >= hi:
        return data
    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return signal.sosfilt(sos, data)


def _add_noise(data: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = np.mean(data**2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=data.shape)
    return data + noise


def _packet_loss(
    data: np.ndarray, sr: int, cfg: EffectConfig, rng: np.random.Generator
) -> np.ndarray:
    """Gilbert-Elliott 两状态丢帧模型。"""
    chunk = int(sr * cfg.packet_ms / 1000)
    n_chunks = len(data) // chunk
    if n_chunks == 0:
        return data
    state = False # False=好, True=坏
    p_good_to_bad = cfg.packet_loss_rate
    p_bad_to_good = cfg.packet_loss_burst or max(p_good_to_bad, 0.5)
    out = data.copy()
    for i in range(n_chunks):
        if state:
            out[i * chunk : (i + 1) * chunk] = 0.0
            if rng.random() < p_bad_to_good:
                state = False
        else:
            if rng.random() < p_good_to_bad:
                state = True
                out[i * chunk : (i + 1) * chunk] = 0.0
    return out


def _add_ticks(data: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """添加 2-5 个随机位置短促脉冲（click/impulse 类信道杂音）。

    注意：与 tau-voice 的 "vocal tics"（TTS 文本级 [cough]/[sneeze] 人声）不同，
    这里是信道层面的脉冲噪声；人声 tic 应在 TTS 文本注入实现。
    振幅按 int16 满量程比例（5%-15%）缩放。
    """
    out = data.copy()
    n_ticks = rng.integers(2, 6)
    for _ in range(n_ticks):
        start = rng.integers(0, max(len(data) - sr // 100, 1))
        dur = rng.integers(sr // 200, sr // 100) # 5-10ms
        amp = rng.uniform(0.05, 0.15) * 32767.0 # int16 满量程比例
        tick = amp * np.sin(2 * np.pi * rng.uniform(800, 2000) * np.arange(dur) / sr)
        out[start : start + dur] += tick
    return out


def audio_pipeline_regular(pcm: np.ndarray, sr: int, seed: int = 0) -> np.ndarray:
    """`regular` 条件：电话信道 + SNR15 噪声 + 1% 丢帧。"""
    return apply_effects(
        pcm, sr,
        EffectConfig(phone_channel=True, noise_snr_db=15.0, packet_loss_rate=0.01, seed=seed),
    )


def audio_pipeline_control(pcm: np.ndarray, sr: int) -> np.ndarray:
    """`control` 条件：干净音频（不做任何处理）。"""
    return pcm


def load_and_pipeline(path: str, sr: int = 16000, mode: str = "control", seed: int = 0) -> np.ndarray:
    """加载 WAV 并应用管线（供冒烟/离线合成复用）。"""
    data, src_sr = load_wav_pcm(path)
    if src_sr != sr:
        from listen2serve.audio.io_utils import resample_audio

        data = resample_audio(data, src_sr, sr)
    if mode == "regular":
        return audio_pipeline_regular(data, sr, seed)
    return audio_pipeline_control(data, sr)
