"""VoicePass 判定：多模态 LLM 直接听关键轮回复音频（三维 1-5 分）。

默认后端为 google/gemini-3.8-flash；也可显式选择 OpenAI 兼容或 DashScope 音频裁判。
VoicePass = 三维均值 ≥3.5 ∧ 无单项 ≤2。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from listen2serve.evaluation.prompts import VOICE_JUDGE_PROMPT
from listen2serve.evaluation.validation import JudgeResponseError, integer, json_object, string
from listen2serve.gateway.base import CallRecord, model_identity, prompt_hash
from listen2serve.gateway.llm_gateway import LLMGateway, LLMResult
from listen2serve.model_registry import resolve_model


@dataclass
class VoiceVerdict:
    """VoicePass 判定结果。"""

    role_voice_match: int
    state_voice_fit: int
    naturalness: int
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
        """Score the reply audio against the supplied role, state and voice requirements."""
        prompt = VOICE_JUDGE_PROMPT.format(
            role_description=role_description,
            communication_style=communication_style or "（未指定）",
            voice_requirements=voice_requirements or "（未指定，按角色设定自行判断）",
            user_state=user_state or "（未知）",
            agent_text=agent_text or "（无文本）",
        )
        from listen2serve.runtime.config import get_settings

        model = self.llm_model or get_settings().voice_judge_model
        backend, model_name = resolve_model(model, get_settings().voice_judge_model)
        model = f"{backend}/{model_name}"
        for attempt in range(2):
            if backend == "dashscope":
                result_text, used_model = self._judge_dashscope_native(audio_wav_path, prompt, model_name)
            else:
                result_text, used_model = self._judge_openai_compat(audio_wav_path, prompt, model)
            try:
                data = json_object(result_text)
                verdict = VoiceVerdict(
                    role_voice_match=integer(data, "role_voice_match", 1, 5),
                    state_voice_fit=integer(data, "state_voice_fit", 1, 5),
                    naturalness=integer(data, "naturalness", 1, 5),
                    reason=string(data, "reason"),
                    backend=used_model,
                    raw=data,
                )
                break
            except JudgeResponseError as exc:
                if attempt == 0:
                    self.parse_retries += 1
                else:
                    self.parse_failures += 1
                    raise JudgeResponseError(f"Voice judge invalid after one retry: {exc}") from exc
        mean = (verdict.role_voice_match + verdict.state_voice_fit + verdict.naturalness) / 3

        verdict.passed = (
            mean >= 3.5 and min(verdict.role_voice_match, verdict.state_voice_fit, verdict.naturalness) >= 3
        )
        return verdict

    def _judge_dashscope_native(self, audio_wav_path: str | Path, prompt: str, model: str) -> tuple[str, str]:
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
        self.llm.sink.append(
            CallRecord(
                service="llm",
                backend="dashscope",
                model=model,
                prompt_hash=prompt_hash(prompt),
                **model_identity(data, model, response=resp, requested_url=url),
            )
        )
        choices = data.get("output", {}).get("choices", [])
        if not choices:
            return "{}", f"dashscope/{model}"
        content = choices[0].get("message", {}).get("content", [])
        text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return text, f"dashscope/{model}"

    def _judge_openai_compat(self, audio_wav_path: str | Path, prompt: str, model: str) -> tuple[str, str]:
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
            max_tokens=2048,
        )
        return result.text, result.model
