"""模型绑定表的发布契约：一个型号只属一个平台、未登记即报错、裁判可整体换掉。

为什么发布仓要专门留这份测试：型号 ↔ 平台 ↔ 音色这三件事一旦在多处各写一份，
就会出现"同一模型走两个平台"的静默错配（no-fallback 原则就是为了堵它）。同时它是
**内网端点的守门测试** —— 任何 idealab/modelbest 一类的内部别名回到绑定表，这里就红。
"""
from __future__ import annotations

import pytest

from listen2serve.model_registry import (
    MODEL_BINDINGS, _KNOWN_PLATFORMS, platform_for, resolve_model, supported_genders,
    voice_for,
)

# 发布仓允许出现的平台全集（新增平台必须同时改 .env.example 与 docs/adding_an_endpoint.md）
ALLOWED_PLATFORMS = {"google", "openai", "dashscope", "local", "volc"}


class TestModelRegistry:
    def test_every_binding_points_at_a_known_platform(self):
        for name, entry in MODEL_BINDINGS.items():
            assert entry["platform"] in ALLOWED_PLATFORMS, f"{name} → {entry['platform']}"
            assert entry["platform"] in _KNOWN_PLATFORMS, name

    def test_no_internal_platform_left(self):
        """内网代理与未发布端点不得出现在绑定表里（发布闸门）。"""
        assert set(_KNOWN_PLATFORMS) == ALLOWED_PLATFORMS
        banned = ("idealab", "modelbest", "minicpm", "alibaba-inc", "gemini-3.1-pro")
        blob = repr(MODEL_BINDINGS).lower()
        for b in banned:
            assert b not in blob, f"绑定表残留内部端点痕迹：{b}"

    def test_bare_name_resolves_bound_platform(self):
        assert resolve_model("qwen-plus", "x") == ("dashscope", "qwen-plus")
        assert resolve_model("qwen-audio-3.0-realtime-plus", "x") == (
            "dashscope", "qwen-audio-3.0-realtime-plus")
        assert resolve_model("doubao-seeduplex-3.0", "x") == ("volc", "doubao-seeduplex-3.0")

    def test_unknown_bare_model_raises(self):
        with pytest.raises(ValueError, match="未收录"):
            resolve_model("no-such-model", "x")

    def test_generic_openai_escape_hatch(self):
        """`openai/<任意型号>` 不必登记即可解析 —— 换裁判的入口。"""
        assert resolve_model("openai/gemini-2.5-flash", "x") == ("openai", "gemini-2.5-flash")
        assert resolve_model("openai/qwen3-next", "x") == ("openai", "qwen3-next")

    def test_prefix_conflict_raises(self):
        with pytest.raises(ValueError, match="绑定冲突"):
            resolve_model("volc/qwen-plus", "x")

    def test_none_falls_back_to_default_spec(self):
        assert resolve_model(None, "qwen-plus") == ("dashscope", "qwen-plus")

    def test_voice_bound_to_model(self):
        assert voice_for("qwen-audio-3.0-tts-plus", "male") == "longanlufeng"
        assert voice_for("qwen-audio-3.0-tts-plus", "female") == "longanhuan_v3.6"
        assert voice_for("qwen3.5-omni-plus-realtime", "male") == "Ethan"
        assert voice_for("qwen-audio-3.0-realtime-plus", "female") == "longanqian"
        # 不支持该性别时回退到任一可用音色；无音色绑定时 None
        assert voice_for("doubao-seeduplex-3.0", "male") == "zh_female_xiaohe_jupiter_bigtts"
        assert voice_for("qwen-plus", "male") is None

    def test_supported_genders(self):
        assert supported_genders("qwen-audio-3.0-tts-plus") == ["female", "male"]
        assert supported_genders("qwen-plus") == []

    def test_platform_for(self):
        assert platform_for("qwen-plus") == "dashscope"
        assert platform_for("nope") is None


class TestJudgeSwap:
    """裁判/演绎默认可换：judge_model 独立于 llm_model，且默认值是公开型号。"""

    def test_settings_expose_independent_judge(self):
        from listen2serve.runtime.config import Settings

        s = Settings(_env_file=None)
        assert s.llm_model in MODEL_BINDINGS, "默认文本 LLM 必须是公开可解析的裸名"
        assert s.judge_model is None or s.judge_model.split("/")[-1] in MODEL_BINDINGS
        assert s.sim_model is None or s.sim_model.split("/")[-1] in MODEL_BINDINGS
        # 论文口径即默认裁判：三类判定都用同一个公开可调用型号
        assert s.judge_model == "google/gemini-3.8-flash"
        assert s.voice_judge_model == "google/gemini-3.8-flash"

    def test_generic_backend_credentials_exist(self):
        from listen2serve.runtime.config import Settings

        s = Settings(_env_file=None)
        assert s.openai_base_url.startswith("https://")
        assert s.openai_api_key is None or isinstance(s.openai_api_key, str)

    def test_policy_model_prefers_judge_model(self):
        """cli 的裁判型号取值顺序：--judge-model > JUDGE_MODEL > LLM_MODEL。"""
        import inspect

        from listen2serve import cli

        src = inspect.getsource(cli)
        assert "args.judge_model or settings.judge_model or settings.llm_model" in src
