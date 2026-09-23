"""运行配置（pydantic-settings，读取项目根 .env）。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 已耗尽/不可用的火山 key 集合（模块级 ⇒ 不受 get_settings() 的 lru_cache 影响，
# Settings 重新实例化也不丢；跨进程不共享，每个跑批进程独立轮换）。
# 只存 key 本身用于比对，**绝不写进任何日志/报告/产物**（本仓纪律：key 仅从 .env 读取）。
_VOLC_EXHAUSTED: set[str] = set()


def _volc_exhausted_prefixes() -> tuple[str, ...]:
    """启动时**预标**已知不可用的 key，按前缀匹配（env：逗号分隔）。

    key 轮换只在"连接失败之后"才发生；若一把 key 从一开始就不可用，每一通都要先
    白撞若干次重试才切到下一把，整批被拖慢。预标让进程一开始就用可用那把。

    为什么用前缀而不是完整 key：前 6 位是日志既有的安全粒度（见 agent.py 的
    `[:6]`），把完整 key 写进 env 示例/脚本就等于让凭据流出 .env。
    """
    raw = os.environ.get("VOLC_API_KEY_EXHAUSTED_PREFIXES", "")
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _volc_is_exhausted(key: str) -> bool:
    """轮换状态（完整 key 精确匹配）**或**运维预标（前缀匹配）任一命中即视为不可用。"""
    if key in _VOLC_EXHAUSTED:
        return True
    return any(key.startswith(p) for p in _volc_exhausted_prefixes())


class Settings(BaseSettings):
    """全局配置：API 密钥与端点。密钥仅从 .env 读取，不提交仓库。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 通用 OpenAI 兼容后端（裁判 / 演绎 / 文本 LLM 的公共通路）----
    # 任何实现了 OpenAI 协议 /chat/completions 的服务都能填进来（自建 vLLM、
    # Gemini 的 OpenAI 兼容入口、各云厂商网关皆可）。调用时写 `openai/<模型名>`
    # 即可，不必在本仓的绑定表里登记 —— 这是"裁判可替换"的唯一入口。
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    # ---- Google 官方 Gemini API 的 OpenAI 兼容入口（默认裁判通路）----
    # 裁判默认 google/gemini-3.8-flash；文本与音频都走这一条，凭据只需一把 GEMINI_API_KEY。
    # 换任意其他厂商仍可用通用后端：JUDGE_MODEL=openai/<名> + OPENAI_BASE_URL/OPENAI_API_KEY。
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    # ---- DashScope 百炼（被测 Realtime WSS + TTS + ASR）----
    dashscope_api_key: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_tts_url: str = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"

    # ---- 音色（与模型绑定：确定参数写在 model_registry.MODEL_BINDINGS）----
    # 策略：用户侧成年男声，客服侧成年女声。以下两项仅作**临时试验覆盖**（默认 None =
    # 取绑定表 voice_for(模型)）；音色试听 试听选定后应回写绑定表而非长期放 .env。
    tts_voice_user: str | None = None
    realtime_voice: str | None = None

    # ---- 火山引擎 / 豆包（跨厂商全双工基线鉴权位）----
    # 豆包 Seeduplex 3.0 是 JSON 事件协议的全双工端点：鉴权 = 单 X-Api-Key，
    # wire model 固定 1.2.6.1，音频 pcm16k 入 / pcm_s16le 24k 出。
    # 适配器见 runtime/realtime/volc_provider.py（事件归一化为 QwenEvent 后由 QwenTickAdapter 复用）。
    # VOLC_API_KEY 是语音服务的 X-Api-Key（不是 ark chat/completions 键），仅从 .env 读取、绝不入库。
    volc_api_key: str | None = None
    # 第二把火山 key：第一把额度用完后自动切到它（见 volc_keys() / volc_key() / volc_mark_exhausted()）。
    volc_api_key2: str | None = None
    volc_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    # 豆包全双工端点 URL。⚠️ 路径分隔符必须是**斜杠** `duplex/realtime/dialogue`——官方 demo
    # config.py 的下划线版 `realtime_dialogue` 是过期默认值，WS 握手会返回应用级 404，别照抄。
    # 本行与 runtime/realtime/volc_provider.py 的 VOLC_REALTIME_URL 同源，改一处必须改另一处。
    volc_realtime_url: str = "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue"


    # ---- LLM 默认模型（演绎 / Policy / 对话级裁判共用同一 llm_model）----
    # 裸模型名必须收录于 model_registry.MODEL_BINDINGS，未收录直接报错（无回落）；
    # 评测通路无自动回退。
    llm_backend: str = "dashscope" # 保留字段（.env 兼容）；裸名解析已强制按绑定表，不回落本字段
    llm_model: str = "qwen-plus" # 文本 LLM 默认（用户模拟器演绎用）

    # ---- 用户模拟器演绎模型（可与裁判模型不同；None = 跟随 llm_model）----
    # 演绎侧与裁判侧分家：同一型号既生成客户台词又给客服打分会有同源偏好，会系统性
    # 抬高各臂的得分。默认跟随 llm_model（qwen-plus）；如需更强的演绎可用 --sim-model 显式指定。
    # ⚠️ 只改本项、⛔ 不要改 llm_model：llm_model 被 policy_judge / flow_judge /
    # dialogue_metrics / leakage_probe 与 cli.py 共用，改它会把全部裁判一起换掉。
    # 覆盖方式：`.env` 的 SIM_MODEL 或 `run-eval --sim-model`（A/B 对照用）。
    sim_model: str | None = None # None = 跟随 llm_model；A/B 用 --sim-model

    # ---- Voice 裁判（多模态音频）----
    # 默认 google/gemini-3.8-flash（论文口径同型号），经 Google 官方 OpenAI 兼容入口，
    # 音频以 input_audio 形式送进去。
    # 备选通路：`--voice-judge-model openai/<名>` 或 `dashscope/<名>`（支持音频输入的兼容端点）。
    # 无自动回退：换后端必须显式指定，保证失败可归因。
    voice_judge_model: str = "google/gemini-3.8-flash" # 音频裁判（论文口径同型号）
    # ---- 裁判型号（与演绎拆开：同一型号既生成台词又打分会有同源偏好）----
    # None = 跟随 llm_model。换任意 OpenAI 兼容裁判：JUDGE_MODEL=openai/<名>，
    # 并配 OPENAI_BASE_URL / OPENAI_API_KEY。⚠️ 换裁判即换口径，旧 run 重评必须显式传参。
    judge_model: str | None = "google/gemini-3.8-flash" # 论文口径；None = 跟随 llm_model

    # ---- ASR（文本模态评测）----
    asr_model: str = "qwen-audio-3.0-asr-flash"

    # ---- TTS 默认模型（百炼 qwen-audio-3.0-tts-plus）----
    tts_backend: str = "dashscope" # dashscope | cosyvoice（均在百炼）
    tts_model: str = "qwen-audio-3.0-tts-plus"
    tts_sample_rate: int = 16000

    # ---- 本地 vLLM（可选）----
    local_base_url: str = "http://localhost:8000/v1"

    # ---- 工具开关（2026-08-14：评测默认不发起 function call）----
    # 默认 False：被测 Agent 不注册工具、run_batch 不注入 DB；需查询的用户信息
    # （db_seed 单行记录）改由 render_customer_profile 渲染为客户信息档案，直接注入
    # 初始 prompt（被测侧与裁判契约同源注入）。可经 .env TOOLS_ENABLED=true 恢复旧行为
    # （注册工具 + attach_db；机制完整保留，随时可逆）。
    tools_enabled: bool = False

    # ---- 被测 Agent prompt 版本（domains.base.AGENT_PROMPT_VERSIONS）----
    # v9（默认）：v7 + state→动作速查表（瘦身版，5016 字 → 中位 2616 / p95 3013）。
    # v7：「听出声音状态 → 做对服务动作」为任务主体，状态应对优先于流程推进，强制收尾轮。
    # v8：v7 + 速查表（有 variant 透传缺陷，⛔ 不得用于生成 canonical 台词）。
    # v6：旧口径。
    # 生成侧（被测 prompt）与评测侧（render_judge_contract 同源契约）都读本项，
    # 故**重评旧 run 必须显式** `--agent-prompt v7` / `v6` / `v8`（或 AGENT_PROMPT=…），
    # 否则裁判会按被测方当时没收到的规则打分。翻默认值不改变任何已落盘 run 的口径，
    # 只改变「不传参时拿到什么」。
    agent_prompt: str = "v9"

    # ---- 运行参数（tick 级全双工）----
    tick_ms: int = Field(default=200, description="编排器虚拟时钟 tick 粒度（对齐 tau-voice 0.20s）")
    concurrency: int = 4
    max_turns: int = 12 # 内部方案：轮次不预定，场景 turn_budget.hard 优先；缺省 12 兜底（soft 6 + 余量）
    max_tick_timeout_s: float = 600.0
    max_response_ticks: int = Field(default=150, description="单轮等待 Agent 完成的最大 tick 数（150×200ms=30s）")
    # 要求连续上行输入的端点（豆包全双工）专用的单轮等待上限：2×max_response_ticks。
    # 不全局抬：实测同一条 1.92s 用户音频，豆包首响在 1.7s 与 24.7s 之间跳（重尾，归因于服务端
    # 排队而非本地发送）；30s 一到编排器就报「Agent 无响应」并丢弃整通。qwen 侧不声明
    # needs_continuous_audio ⇒ 仍按 30s（既有端点时序口径不变）。
    streaming_response_ticks: int = Field(
        default=300, description="needs_continuous_audio 端点的单轮等待上限 tick 数（300×200ms=60s）")
    idle_end_ticks: int = Field(default=3, description="播出缓冲排空后，连续静默多少 tick 判定轮结束（轮结束不依赖 response.done）")
    turn_gap_ticks: int = Field(default=2, description="Agent 轮结束到下一用户轮开始的间隔 tick 数")
    interrupt_delay_ticks: int = Field(default=3, description="打断轮：Agent 开始说话后多少 tick 用户插话")
    max_session_tokens: int = Field(
        default=200_000,
        description="单场景 Realtime 会话 token 熔断上限（response.done usage 累计超限即终止，防失控消耗）",
    )

    def volc_key(self) -> str | None:
        """火山方舟 / 豆包 API key（跨厂商基线用）—— **当前生效的那一把**。

        多把 key 按额度顺序轮换：返回 `volc_keys()` 里第一把未被 `volc_mark_exhausted()`
        标记的；全部标记过则回落最后一把 —— 让真实错误如实暴露，而不是返回 None
        造成「没配 key」的误判（那会把额度问题伪装成配置问题）。
        """
        ks = self.volc_keys()
        if not ks:
            return None
        for k in ks:
            if not _volc_is_exhausted(k):
                return k
        return ks[-1]

    def volc_keys(self) -> list[str]:
        """全部已配置的火山 key，按优先级排序（先用第一把）。空值剔除，不暴露给日志。"""
        return [k for k in (self.volc_api_key, self.volc_api_key2) if k]

    def volc_mark_exhausted(self, key: str) -> str | None:
        """标记某把 key 不可用（额度耗尽/鉴权失败），返回切换后的 key。

        返回 None = 没有下一把可切（调用方应把原错误如实抛出，不要静默重试到死）。
        进程级状态：`get_settings()` 有 lru_cache，但耗尽集合是**模块级**的，
        故 Settings 重新实例化也不丢；跨进程不共享（每个跑批进程独立轮换，这是想要的行为）。
        """
        _VOLC_EXHAUSTED.add(key)
        nxt = self.volc_key()
        return None if (nxt is None or nxt == key) else nxt


@lru_cache
def get_settings() -> Settings:
    return Settings()
