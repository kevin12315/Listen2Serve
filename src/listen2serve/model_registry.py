"""模型选型绑定表（单一事实来源）：模型 ↔ 平台 ↔ 音色。

原则：
- **一个模型只属于一个平台**：裸模型名按本表解析平台；显式前缀与本表冲突时直接报错
  （宁可失败也不静默走错平台，符合 no-fallback 可归因原则）；
- **裸名未收录 → 报错**：禁止回落任何默认后端（禁止静默降级/换模型）；
  新模型须先在本表登记，或用 `platform/name` 显式指定；
- **音色与模型绑定**：TTS 音色/Realtime 音色是模型的确定参数，写在本表；
  `.env` 的 TTS_VOICE_USER / REALTIME_VOICE 仅作临时试验覆盖（试听选定后应回写本表）。

当前选型（与论文口径一致）：
- 文本 LLM（演绎默认）：qwen-plus@dashscope（百炼）；裸名必须收录于本表，未收录直接报错
  - 三类裁判（关键轮动作 / 输出语音 / 流程与任务）：google/gemini-3.8-flash
（默认，Google 官方 OpenAI 兼容入口）
- TTS / ASR：走百炼（dashscope）
- 被测 Realtime（论文三端点）：qwen-audio-3.0-realtime-plus@dashscope（WSS）、
  qwen3.5-omni-plus-realtime@dashscope，以及跨厂商基线 doubao-seeduplex-3.0@volc；
  模型 ID 与音色均经真机验证（见各条目注）
"""

from __future__ import annotations

import json
from pathlib import Path

# 模型 → {platform, voices?}；voices 按性别绑定（male/female），仅音频类模型有意义。
# 性别均衡策略（2026-08-10）：合成/评测时男女声均匀指定（除非模型只支持一个性别）：
# - 用户侧：性别绑在场景（scenario.user_gender，姓名与声音性别一致）；
# - 客服侧：按场景哈希 50/50 确定性分配。
MODEL_BINDINGS: dict[str, dict] = {
    # ---- 文本 LLM（演绎 / 各文本裁判）：默认公开型号，可整条换成 openai/<名> ----
    "qwen-plus": {"platform": "dashscope"},
    "qwen-max": {"platform": "dashscope"},
    # 论文的三个裁判（关键轮动作 / 输出语音 / 流程与任务）都是这一个型号，经 Google
    # 官方 OpenAI 兼容入口调用（模型 ID 见 https://ai.google.dev/gemini-api/docs）。平台=google ⇒ 只需
    # 一把 GEMINI_API_KEY；换成别的厂商请用 `openai/<名>` 走通用后端。
    "gemini-3.8-flash": {"platform": "google"},

    # ---- Voice 裁判（多模态）----
    "qwen3.5-omni-plus": {"platform": "dashscope"}, # 备选音频裁判（原生多模态端点）

    # ---- TTS（百炼；音色 2026-08-10 实测矩阵验证）----
    "qwen-audio-3.0-tts-plus": {
        "platform": "dashscope",
        # 官方口径：longanlingxin（女）/longanlufeng（男），实测均 ✓；
        # longanhuan_v3.6（女）未见于官方口径但实测可用且唯一通过人工听审 → 女声默认；
        # 若后续服务端收紧导致 400，女声切 longanlingxin（已验证可用，待听审）
        "voices": {"male": "longanlufeng", "female": "longanhuan_v3.6"},
        "voices_alt": {"female": "longanlingxin"}, # 备选（仅记录，不自动切换）
    },

    # ---- ASR（百炼）----
    "qwen-audio-3.0-asr-flash": {"platform": "dashscope"},

    # ---- 被测 Realtime（百炼 WSS）----
    "qwen-audio-3.0-realtime-plus": {
        "platform": "dashscope",
        # 官方音色表 5 个，WSS 会话录音试听后用户选定（2026-08-10）；
        # 其余可用：longanlingxin / longanlingxi / longanxiaoxin（旧默认，已换下）
        "voices": {"male": "longanlufeng", "female": "longanqian"},
    },
    "qwen3.5-omni-plus-realtime": {
        "platform": "dashscope",
        "voices": {"male": "Ethan", "female": "Tina"}, # 官网 API 口径（2026-08-10），真机冒烟验证
    },

    # ---- 跨厂商全双工基线：豆包 Seeduplex 3.0（火山引擎）----
    # 平台 volc；wire model 固定 1.2.6.1（见 runtime/realtime/volc_provider.VOLC_MODEL），
    # 友好裸名仅用于绑定表/清单。协议按官方 demo 恢复，adapter 见 runtime/realtime/volc_provider.py；
    # 端点 URL 的下划线/斜杠路径坑见 runtime/config.py 的 volc_realtime_url 注。
    # 音色仅绑定官方 demo 默认女声；可用男声需按豆包音色表真机验证后再补。
    "doubao-seeduplex-3.0": {
        "platform": "volc",
        "voices": {"female": "zh_female_xiaohe_jupiter_bigtts"},
    },

}

_KNOWN_PLATFORMS = ("google", "openai", "dashscope", "local", "volc")

# 全双工、**没有文字输入通道**且要求连续上行输入的端点族（如豆包 Seeduplex）。
# 单一事实源：run_batch 的重试退避、E-A 的 c0/d 跳格判据都从这里派生，
# 不要再各写一份 `startswith("volc")` 之类的私有清单（接入新的无文字通道端点时
# 正是这种副本漏改导致新端点被当成 qwen 去连、以及重试只退避 2s 而三连撞）。
FULL_DUPLEX_NO_TEXT_INPUT_PLATFORMS = frozenset({"volc"})


def platform_for(model_name: str) -> str | None:
    """模型绑定的平台；未收录返回 None。"""
    entry = MODEL_BINDINGS.get(model_name)
    return entry["platform"] if entry else None


def supported_genders(model_name: str) -> list[str]:
    """模型支持的音色性别列表（无音色绑定返回空）。"""
    entry = MODEL_BINDINGS.get(model_name) or {}
    return sorted((entry.get("voices") or {}).keys())


def voice_for(model_name: str, gender: str | None = None) -> str | None:
    """模型绑定音色。

    gender 指定时：命中返回对应音色；模型不支持该性别则回退到其唯一/任一可用音色
    （均衡策略的例外："除非它只支持一个性别"）；无任何音色绑定返回 None。
    gender=None：返回默认音色（female 优先，其次 male）。
    """
    entry = MODEL_BINDINGS.get(model_name) or {}
    voices: dict[str, str] = entry.get("voices") or {}
    if not voices:
        return None
    if gender and gender in voices:
        return voices[gender]
    return voices.get("female") or voices.get("male")


# ---- 用户侧音色 map（persona 性别×年龄匹配，替代两枚性别默认音色）----
# v6.1 起按数据版本选表：本发布仓只带 v6.1 精选池重建的 user_voice_map_v61.json
# （voice_pool_v61 十音色）。
_ROOT = Path(__file__).resolve().parents[2]
_USER_VOICE_MAP_V61_PATH = _ROOT / "data/benchmark/voices/user_voice_map_v61.json"
_USER_VOICE_MAP: dict | None = None


def _user_voice_map_path() -> Path:
    """发布仓只带一张用户侧音色表（v61 精选池）。"""
    return _USER_VOICE_MAP_V61_PATH


def user_voice_for(scenario_id: str | None, tts_model_name: str,
                   user_gender: str | None = None) -> str | None:
    """用户侧音色优先查 user_voice_map（内部方案 / v5.5）；查不到返回 None，
    由调用方回落 voice_for（性别默认音色，兼容路径）。

    - 键为 base_scenario_id（T1/T2/T3 三变体共享同一音色）：传入展开层
      scenario_id 时自动剥离 -T1/-T2/-T3 后缀；
    - 仅对支持两性音色的 TTS 模型生效（单性别模型如 tts-flash/seed-tts
      无法承载分性别音色库 → 回落绑定表）；
    - 命中时校验 map 与场景性别一致（数据完整性护栏，不一致报错不静默）；
    - map 文件不存在 → None（兼容旧环境）。
    """
    global _USER_VOICE_MAP
    if _USER_VOICE_MAP is None:
        path = _user_voice_map_path()
        if not path.exists():
            return None
        try:
            _USER_VOICE_MAP = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    if not scenario_id:
        return None
    # 仅两性音色模型可用 map（单性别模型回落绑定表，避免性别错配）
    genders = supported_genders(tts_model_name)
    if genders != ["female", "male"]:
        return None
    base_id = str(scenario_id)
    head, _, tail = base_id.rpartition("-")
    if tail in ("T1", "T2", "T3"):
        base_id = head
    entry = (_USER_VOICE_MAP.get("map") or {}).get(base_id)
    if not entry:
        return None
    if user_gender and entry.get("gender") != user_gender:
        raise ValueError(
            f"user_voice_map 性别不一致: {base_id} map={entry.get('gender')!r} "
            f"场景={user_gender!r}（数据完整性护栏，见 内部方案）")
    return entry.get("voice")


def _reset_user_voice_cache() -> None:
    """测试辅助：清空 user_voice_map 缓存（map 文件变更后重新加载）。"""
    global _USER_VOICE_MAP
    _USER_VOICE_MAP = None


def resolve_model(model_spec: str | None, default_model: str) -> tuple[str, str]:
    """解析模型规格 → (backend, model_name)，以绑定表为准（无回落）。

    - `platform/name`：显式前缀；若 name 在绑定表且平台不符 → 抛错（一个模型只属一个平台）
    - 裸 `name`：必须收录于绑定表 → 绑定平台；未收录 → 抛错
      （2026-08-14 LLM 统一：禁止回落默认后端，宁可失败也不静默走错平台/换模型）
    - None/空：用 default_model 递归解析
    """
    spec = model_spec or default_model
    if "/" in spec:
        backend, name = spec.split("/", 1)
        if backend in _KNOWN_PLATFORMS:
            bound = platform_for(name)
            if bound and bound != backend:
                raise ValueError(
                    f"模型平台绑定冲突: {name} 绑定于 {bound}，但显式指定了 {backend}"
                    f"（选型规则：一个模型只对应一个平台，见 model_registry.MODEL_BINDINGS）"
                )
            return backend, name
    bound = platform_for(spec)
    if bound is None:
        raise ValueError(
            f"模型未收录于绑定表: {spec}（no-fallback 原则：裸名必须收录于 "
            f"model_registry.MODEL_BINDINGS，不回落默认后端；新模型请先登记绑定，"
            f"或以 platform/{spec} 显式指定）"
        )
    return bound, spec
