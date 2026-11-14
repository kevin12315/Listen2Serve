"Runtime settings loaded from the repository .env and environment."

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_VOLC_EXHAUSTED: set[str] = set()


def _volc_exhausted_prefixes() -> tuple[str, ...]:
    "Optional exhausted-key prefixes supplied by the operator."
    raw = os.environ.get("VOLC_API_KEY_EXHAUSTED_PREFIXES", "")
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _volc_is_exhausted(key: str) -> bool:
    """Check exhausted-key state and operator-supplied prefixes."""
    if key in _VOLC_EXHAUSTED:
        return True
    return any(key.startswith(p) for p in _volc_exhausted_prefixes())


class Settings(BaseSettings):
    "Runtime credentials, models, and audio timing settings."

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    moonshot_api_key: str | None = None
    moonshot_base_url: str = "https://api.moonshot.cn/v1"

    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    dashscope_api_key: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_tts_url: str = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"

    tts_voice_user: str | None = None
    realtime_voice: str | None = None

    volc_api_key: str | None = None

    volc_api_key2: str | None = None
    volc_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"

    volc_realtime_url: str = "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue"

    llm_model: str = "kimi-k3"

    sim_model: str | None = None

    voice_judge_model: str = "google/gemini-3.8-flash"

    judge_model: str | None = "google/gemini-3.8-flash"

    asr_model: str = "qwen-audio-3.0-asr-flash"

    tts_backend: str = "dashscope"
    tts_model: str = "qwen-audio-3.0-tts-plus"
    tts_sample_rate: int = 16000

    local_base_url: str = "http://localhost:8000/v1"

    tools_enabled: bool = False

    tick_ms: int = Field(default=200, description="编排器虚拟时钟 tick 粒度（对齐 tau-voice 0.20s）")
    concurrency: int = 4
    max_turns: int = 12
    max_tick_timeout_s: float = 600.0
    max_response_ticks: int = Field(
        default=150, description="单轮等待 Agent 完成的最大 tick 数（150×200ms=30s）"
    )

    streaming_response_ticks: int = Field(
        default=300, description="needs_continuous_audio 端点的单轮等待上限 tick 数（300×200ms=60s）"
    )
    idle_end_ticks: int = Field(
        default=3, description="播出缓冲排空后，连续静默多少 tick 判定轮结束（轮结束不依赖 response.done）"
    )
    turn_gap_ticks: int = Field(default=2, description="Agent 轮结束到下一用户轮开始的间隔 tick 数")
    interrupt_delay_ticks: int = Field(default=3, description="打断轮：Agent 开始说话后多少 tick 用户插话")
    max_session_tokens: int = Field(
        default=200_000,
        description="单场景 Realtime 会话 token 熔断上限（response.done usage 累计超限即终止，防失控消耗）",
    )

    def volc_key(self) -> str | None:
        "Return the first configured key that has not been exhausted."
        ks = self.volc_keys()
        if not ks:
            return None
        for k in ks:
            if not _volc_is_exhausted(k):
                return k
        return ks[-1]

    def volc_keys(self) -> list[str]:
        "Configured speech-service keys, in priority order."
        return [k for k in (self.volc_api_key, self.volc_api_key2) if k and k.strip()]

    def volc_mark_exhausted(self, key: str) -> str | None:
        "Mark a key exhausted and return the next available key."
        _VOLC_EXHAUSTED.add(key)
        nxt = self.volc_key()
        return None if (nxt is None or nxt == key) else nxt


@lru_cache
def get_settings() -> Settings:
    return Settings()
