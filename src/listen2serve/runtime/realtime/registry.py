"Realtime Provider 注册表：按名称创建被测模型 Provider"

from __future__ import annotations

from typing import Any

from listen2serve.model_registry import _KNOWN_PLATFORMS, platform_for, voice_for
from listen2serve.runtime.realtime.qwen_provider import QwenRealtimeProvider
from listen2serve.runtime.realtime.volc_provider import (
    VOLC_DEFAULT_VOICE,
    VOLC_MODEL,
    VolcRealtimeProvider,
)


def create_provider(model_spec: str, api_key: str, **kwargs: Any):
    """创建 Provider 实例。

    Args:
        model_spec: 形如 `dashscope/qwen-audio-3.0-realtime-plus` 或裸模型名。
        api_key: 对应平台的 API Key。
    """
    backend, _, model = model_spec.partition("/")
    # ⚠️ 合法后端名单必须取 model_registry._KNOWN_PLATFORMS 这个**单一事实源**。
    # 早先这里写死了一份平台清单，新端点接入时漏登记 ⇒ 带前缀的型号被当成裸名解析：
    # 新平台的 `xxx/Model` 若不在元组里会被当成裸名解析，
    # 服务端回 `Model not found`，看起来像模型不存在。
    # 形状与豆包那次一模一样：多一处清单副本，就多一处会静默走错后端的地方。
    if backend not in _KNOWN_PLATFORMS or not model:
        model = model_spec
        backend = platform_for(model_spec) or "dashscope"

    if backend == "dashscope":
        return QwenRealtimeProvider(
            api_key=api_key,
            model=model,
            # 音色优先级：显式参数/.env 覆盖 > 模型绑定表（model_registry）
            voice=kwargs.get("voice") or voice_for(model) or "longanxiaoxin",
        )
    if backend == "volc":
        # 豆包 Seeduplex 3.0：wire model 固定 1.2.6.1（与友好裸名无关，见 volc_provider）
        return VolcRealtimeProvider(
            api_key=api_key,
            model=VOLC_MODEL,
            voice=kwargs.get("voice") or voice_for(model) or VOLC_DEFAULT_VOICE,
        )
    if backend == "local":
        raise NotImplementedError("本地 vLLM-Omni 待接入")
    raise ValueError(f"未知 Realtime 后端: {backend}")


def default_model_spec() -> str:
    return "dashscope/qwen-audio-3.0-realtime-plus"


def candidate_models() -> list[str]:
    "被测模型候选清单（方案 6.2）"
    return [
        "dashscope/qwen-audio-3.0-realtime-plus",
        "dashscope/qwen-audio-3.0-realtime-flash",
        "dashscope/qwen3.5-omni-plus-realtime",
        "dashscope/qwen3.5-omni-flash-realtime",
    ]
