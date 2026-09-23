"""VoicePass 判定：多模态 LLM 直接听关键轮回复音频（三维 1-5 分）。

后端：google/gemini-3.8-flash（默认，Google 官方 OpenAI 兼容入口）→ `openai/<名>`（通用 OpenAI 兼容后端，音频模态）。
VoicePass = 三维均值 ≥3.5 ∧ 无单项 ≤2。
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from listen2serve.evaluation.prompts import VOICE_JUDGE_PROMPT
from listen2serve.gateway.llm_gateway import LLMGateway, LLMResult


@dataclass
class VoiceVerdict:
    """VoicePass 判定结果。"""

    role_voice_match: int = 1
    state_voice_fit: int = 1
    naturalness: int = 1
    reason: str = ""
    passed: bool = False
    backend: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role_voice_match": self.role_voice_match,
            "state_voice_fit": self.state_voice_fit,
            "naturalness": self.naturalness,
            "mean": round((self.role_voice_match + self.state_voice_fit + self.naturalness) / 3, 2),
            "reason": self.reason,
            "passed": self.passed,
            "backend": self.backend,
        }


class VoiceJudge:
    """语音质量裁判（多模态 LLM 听音频）。

    后端选择：
    - `google/<名>` 与 `openai/<名>`：OpenAI 兼容端点 + 标准 input_audio 格式
    - `dashscope/qwen-omni-*`：DashScope 原生 multimodal-generation 端点（base64 音频）
    - 其他：走 LLMGateway 的 input_audio 格式
    """

    def __init__(
        self,
        llm: LLMGateway | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.llm = llm or LLMGateway()
        self.llm_model = llm_model
        # F5 统计（evaluate 结束汇总输出）
        self.parse_retries = 0
        self.parse_failures = 0

    def judge(
        self,
        audio_wav_path: str | Path,
        role_description: str,
        user_state: str,
        agent_text: str,
        voice_requirements: str | None = None,
        communication_style: str | None = None,
    ) -> VoiceVerdict:
        """聆听关键轮回复音频并三维评分。

        B2：声音锚点改为使用注入的 voice_requirements（不再硬编码角色刻板印象），
        communication_style 一并注入角色设定。F5：解析失败重试一次再落默认值。
        """
        prompt = VOICE_JUDGE_PROMPT.format(
            role_description=role_description,
            communication_style=communication_style or "（未指定）",
            voice_requirements=voice_requirements or "（未指定，按角色设定自行判断）",
            user_state=user_state or "（未知）",
            agent_text=agent_text or "（无文本）",
        )
        model = self.llm_model or ""
        backend_name = model.split("/")[-1]
        # F5：解析失败重试一次再落默认值（重走同一后端）
        result_text, used_model = "", ""
        for attempt in range(2):
            if "omni" in backend_name or "audio" in backend_name:
                # DashScope 原生 multimodal 端点（qwen-omni 系）
                result_text, used_model = self._judge_dashscope_native(
                    audio_wav_path, prompt, backend_name
                )
            else:
            # 其余一律走 OpenAI 兼容端点 + 标准 input_audio（google/、openai/ 都在这支）
                result_text, used_model = self._judge_openai_compat(
                    audio_wav_path, prompt, model
                )
            data = _extract_json(result_text)
            if data:
                break
            if attempt == 0: # 首次解析失败 → 记一次重试；重试再失败才落默认值
                self.parse_retries += 1
        if not data:
            self.parse_failures += 1
        verdict = VoiceVerdict(
            role_voice_match=_clamp(data.get("role_voice_match", 1)),
            state_voice_fit=_clamp(data.get("state_voice_fit", 1)),
            naturalness=_clamp(data.get("naturalness", 1)),
            reason=data.get("reason", ""),
            backend=used_model,
            raw=data,
        )
        mean = (verdict.role_voice_match + verdict.state_voice_fit + verdict.naturalness) / 3
        # 规范（方案 8.1）：VoicePass = 三维均值 ≥3.5 ∧ 无单项 ≤2（即单项必须 ≥3）
        verdict.passed = mean >= 3.5 and min(
            verdict.role_voice_match, verdict.state_voice_fit, verdict.naturalness
        ) >= 3
        return verdict

    # ---- 通用 OpenAI 兼容后端（audio_url base64 + 一行紧凑输出）----
    # 带 reasoning 的音频模型会把思考计入 completion tokens，因此要求**紧凑输出格式**
    # （"4,3,5|理由"），但评分上下文（角色设定/用户状态/维度定义）必须完整给出，
    # 否则"角色声音匹配/状态声音适配"两个维度无法判定。

    # ---- DashScope 原生 multimodal（qwen-omni 系列）----
    def _judge_dashscope_native(
        self, audio_wav_path: str | Path, prompt: str, model: str
    ) -> tuple[str, str]:
        import base64

        import httpx

        from listen2serve.runtime.config import get_settings

        s = get_settings()
        audio_b64 = base64.b64encode(Path(audio_wav_path).read_bytes()).decode()
        body = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt},
                            {"audio": f"data:audio/wav;base64,{audio_b64}"},
                        ],
                    }
                ]
            },
        }
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {s.dashscope_api_key or ''}"},
            json=body,
            timeout=180,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope omni 评分失败 HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choices = data.get("output", {}).get("choices", [])
        if not choices:
            return "{}", f"dashscope/{model}"
        content = choices[0].get("message", {}).get("content", [])
        text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return text, f"dashscope/{model}"

    # ---- OpenAI 兼容（Gemini 等，input_audio 格式）----
    def _judge_openai_compat(
        self, audio_wav_path: str | Path, prompt: str, model: str
    ) -> tuple[str, str]:
        import base64

        audio_b64 = base64.b64encode(Path(audio_wav_path).read_bytes()).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            }
        ]
        result: LLMResult = self.llm.chat(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=2048, # reasoning 模型消耗大，预留输出空间
        )
        return result.text, result.model


def _clamp(value: Any, lo: int = 1, hi: int = 5) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 1
    return max(lo, min(hi, v))


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
