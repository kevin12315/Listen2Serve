"""OpenAI 兼容 LLM 网关：openai（通用）/ dashscope / local 三档后端切换。

模型名解析（以 model_registry.MODEL_BINDINGS 为准，一个模型只属一个平台）：
- 裸模型名（如 `qwen-plus`）：必须收录于绑定表 → 绑定平台；未收录 → 报错
  （禁止回落默认后端）
- `platform/name`：显式指定；若与绑定表冲突直接抛错（防止同一模型走两个平台）

通用后端的凭据来自 .env 的 OPENAI_API_KEY / OPENAI_BASE_URL。

设计约束：评测通路**不做自动回退**——一条通路失败即抛错、该样本判定为失败；
备选后端只能通过模型参数显式切换（保证审计与可归因）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from listen2serve.gateway.base import CallRecord, CallSink, MemoryCallSink, prompt_hash
from listen2serve.model_registry import resolve_model
from listen2serve.runtime.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ================== 按模型的请求参数适配表 ==================
# 为什么需要：有些托管模型只接受**离散** temperature 值域，传域外值不是被忽略而是
# **直接 400 整批失败**（如某模型只收 {0.0, 0.6, 1.0}，而 user_simulator.py 的演绎调用硬编码 0.7，
# 不改就会整批全灭）。
#
# 适配纪律（两条）：
# ① 取**域内最近值**，不自作主张选别的档；
# ② **改写必须留痕** —— 写进 CallRecord.param_note，并对每个 (模型, 请求值→生效值)
# 组合发一次 warning。⛔ 不许静默 clamp：否则落盘 meta 记的参数与实际发出的不一致，
# 日后没人知道这批数据是在哪个温度下跑的。
# 换用值域受限的模型时在此登记（未登记的模型零行为变化）。
TEMPERATURE_DOMAIN: dict[str, tuple[float, ...]] = {}

# 推理模型的思考档位：某些带长思考的推理模型默认就开思考、且 reasoning_content 计入
# completion_tokens，对演绎这类短任务反而慢数倍、output 花费数倍 —— 在此按模型压到 "low"。
REASONING_EFFORT_BY_MODEL: dict[str, str] = {}

# 已告警过的 (模型, 请求值, 生效值)，避免逐调用刷屏
_WARNED_PARAM_ADAPT: set[tuple[str, float, float]] = set()


def _adapt_temperature(name: str, temperature: float) -> tuple[float, str]:
    """把 temperature 收敛到该模型的合法值域；返回 (生效值, 留痕文本)。

    不在适配表里的模型原样返回、留痕为空串 ⇒ 对未登记模型**零行为变化**。
    """
    domain = TEMPERATURE_DOMAIN.get(name)
    if not domain or temperature in domain:
        return temperature, ""
    effective = min(domain, key=lambda v: abs(v - temperature))
    note = (f"temperature {temperature}→{effective}"
            f"（{name} 只接受 {list(domain)}，见 TEMPERATURE_DOMAIN）")
    key = (name, temperature, effective)
    if key not in _WARNED_PARAM_ADAPT:
        _WARNED_PARAM_ADAPT.add(key)
        logger.warning("LLM 网关参数适配（每个组合只告警一次）：%s", note)
    return effective, note


def _coerce_content(raw: Any, model: str) -> str:
    """把 message.content 归一成 str。

    为什么需要：多数端点返回 str，但**有的模型返回多段 content 列表**（实测
    部分托管模型经 OpenAI 兼容网关返回 list 型 content）。原先 `chat()` 直接透传，
    于是 list 一路流到 `flow_judge._extract_json` 的 `re.search(pattern, text)` ⇒
    `TypeError: expected string or bytes-like object, got 'list'`，
    而且**不是单条失败，是整批 evaluate 硬崩**（实测崩在 8/15，前 8 条只留在 partial 里）。
    归一放在网关层而不是各个 judge 里：judge 有 5 个（policy/keyturn/flow/dialogue/leakage），
    逐个加防御就是 5 份手写副本，正是本仓登记过的「口径散在多份副本里」病。
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for it in raw:
            if isinstance(it, str):
                parts.append(it)
            elif isinstance(it, dict):
                # OpenAI 多段格式：{"type":"text","text":…}；兼容只回 {"text":…} 的变体
                t = it.get("text")
                if isinstance(t, str):
                    parts.append(t)
            else:
                t = getattr(it, "text", None)
                if isinstance(t, str):
                    parts.append(t)
        joined = "".join(parts)
        key = (model, "list_content")
        if key not in _WARNED_PARAM_ADAPT:
            _WARNED_PARAM_ADAPT.add(key) # 复用同一个去重集合，避免逐调用刷屏
            logger.warning("LLM 网关：模型 %s 返回 list 型 content（%d 段），已归一为字符串。"
                           "每个模型只告警一次。", model, len(raw))
        return joined
    return str(raw)


@dataclass
class LLMResult:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "" # 截断守卫（length = 输出被 max_tokens 截断）
    record: CallRecord | None = field(default=None, repr=False)


class LLMGateway:
    """OpenAI 兼容 LLM 网关，带调用审计；失败即抛错（无自动回退）。"""

    def __init__(self, settings: Settings | None = None, sink: CallSink | None = None) -> None:
        self.settings = settings or get_settings()
        self.sink = sink or MemoryCallSink()
        self._clients: dict[str, OpenAI] = {}

    # ---- 客户端懒加载 ----
    def _client(self, backend: str, model_name: str = "") -> OpenAI:
        s = self.settings
        if backend == "openai":
            # 通用 OpenAI 兼容后端：key / base_url 全来自 .env，本仓不预设厂商
            key = s.openai_api_key or "unset"
            if key not in self._clients:
                self._clients[key] = OpenAI(api_key=key, base_url=s.openai_base_url,
                                            timeout=300)
            return self._clients[key]
        if backend == "google":
            # Google 官方 OpenAI 兼容入口：base_url 已含 /v1beta/openai，SDK 自带 /chat/completions
            key = s.gemini_api_key or "unset"
            if key not in self._clients:
                self._clients[key] = OpenAI(api_key=key, base_url=s.gemini_base_url,
                                            timeout=300)
            return self._clients[key]
        if backend in self._clients:
            return self._clients[backend]
        if backend == "dashscope":
            client = OpenAI(api_key=s.dashscope_api_key or "unset", base_url=s.dashscope_base_url, timeout=300)
        elif backend == "local":
            client = OpenAI(api_key="local", base_url=s.local_base_url, timeout=300)
        else:
            raise ValueError(f"未知 LLM 后端: {backend}")
        self._clients[backend] = client
        return client

    def _parse_model(self, model: str | None) -> tuple[str, str]:
        """返回 (backend, model_name)；以模型↔平台绑定表为准，冲突即抛错。"""
        return resolve_model(model, self.settings.llm_model)

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        seed: int | None = None,
        thinking: bool | None = None,
    ) -> LLMResult:
        """一次 chat 调用；失败记录审计后抛出（无自动回退）。

        Args:
            seed: 传给后端的确定性种子（后端支持时生效），用于话术演绎可复现。
            thinking: 显式开关 reasoning/thinking（成本控制：
                qwen3 系经 extra_body.enable_thinking 传递；None = 后端默认）。
        """
        backend, name = self._parse_model(model)
        ph = prompt_hash(str(messages))
        start = time.perf_counter()
        extra: dict[str, Any] = {}
        if seed is not None:
            extra["seed"] = seed
        # 按模型适配请求参数（不在适配表里的模型 ⇒ 生效值 == 请求值、note 为空串）
        eff_temp, param_note = _adapt_temperature(name, temperature)
        extra_body: dict[str, Any] = {}
        if thinking is not None and backend == "dashscope":
            extra_body["enable_thinking"] = bool(thinking)
        effort = REASONING_EFFORT_BY_MODEL.get(name)
        if effort:
            extra_body["reasoning_effort"] = effort
            param_note = (param_note + "; " if param_note else "") + f"reasoning_effort={effort}（网关默认档）"
        if extra_body:
            extra["extra_body"] = extra_body
        try:
            client = self._client(backend, name)
            resp = client.chat.completions.create(
                model=name,
                messages=messages,
                temperature=eff_temp,
                max_tokens=max_tokens,
                **extra,
            )
        except Exception as exc: # noqa: BLE001 - 记录审计后原样抛出
            self.sink.append(CallRecord(
                service="llm", backend=backend, model=name, prompt_hash=ph,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
                param_note=param_note,
            ))
            raise
        latency = (time.perf_counter() - start) * 1000
        # content 归一（见 _coerce_content）：有的模型返回多段 list，直接透传会让下游
        # 所有 re.search/json.loads 型解析炸掉，而且是整批崩不是单条失败。
        text = _coerce_content(resp.choices[0].message.content, name)
        finish_reason = getattr(resp.choices[0], "finish_reason", "") or ""
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        record = CallRecord(
            service="llm", backend=backend, model=name, prompt_hash=ph,
            latency_ms=latency, prompt_tokens=pt, completion_tokens=ct,
            param_note=param_note,
        )
        self.sink.append(record)
        return LLMResult(text=text, model=f"{backend}/{name}", prompt_tokens=pt,
                         completion_tokens=ct, latency_ms=latency,
                         finish_reason=finish_reason, record=record)
