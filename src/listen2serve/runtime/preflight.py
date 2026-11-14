"""Validate required local credentials without contacting any model endpoint."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

from listen2serve.model_registry import resolve_model
from listen2serve.runtime.config import Settings, get_settings

# Alternatives within one tuple satisfy the same service requirement.
CredentialGroup = tuple[str, ...]
_LLM_CREDENTIALS: dict[str, CredentialGroup] = {
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "local": (),
}


def _llm_requirement(model: str | None, default: str) -> CredentialGroup:
    backend, _ = resolve_model(model, default)
    if backend not in _LLM_CREDENTIALS:
        raise ValueError(f"Unsupported LLM backend: {backend}")
    return _LLM_CREDENTIALS[backend]


def _generation_requirements(
    settings: Settings, target_models: Iterable[str], sim_model: str | None
) -> list[CredentialGroup]:
    groups: list[CredentialGroup] = []
    for model in target_models:
        backend, _ = resolve_model(model, settings.llm_model)
        if backend == "volc":
            groups.append(("VOLC_API_KEY", "VOLC_API_KEY2"))
        elif backend == "dashscope":
            groups.append(("DASHSCOPE_API_KEY",))
        else:
            raise ValueError(f"Unsupported Realtime backend: {backend}")
    groups.append(_llm_requirement(sim_model or settings.sim_model, settings.llm_model))
    if settings.tts_backend not in ("dashscope", "cosyvoice"):
        raise ValueError("Unsupported TTS backend")
    # Both supported TTS adapters use DashScope, independently of the target agent.
    groups.append(("DASHSCOPE_API_KEY",))
    return groups


def _evaluation_requirements(
    settings: Settings,
    *,
    judge_model: str | None,
    voice_judge_model: str | None,
    modality: str,
    use_asr: bool,
) -> list[CredentialGroup]:
    if modality not in ("audio", "text"):
        raise ValueError("Unsupported evaluation modality")
    groups = [_llm_requirement(judge_model or settings.judge_model, settings.llm_model)]
    if modality == "audio":
        groups.append(_llm_requirement(voice_judge_model, settings.voice_judge_model))
    if use_asr or modality == "text":
        # The current ASR adapter always uses DashScope's audio endpoint.
        groups.append(("DASHSCOPE_API_KEY",))
    return groups


def _require(settings: Settings, groups: Iterable[CredentialGroup]) -> None:
    missing = []
    for group in dict.fromkeys(groups):
        if group and not any(str(getattr(settings, key.lower(), "") or "").strip() for key in group):
            missing.append(" / ".join(group))
    if missing:
        # Include variable names only: never format Settings or credential values.
        raise ValueError("缺少凭据: " + ", ".join(missing))


def require_generation_credentials(
    settings: Settings, target_models: Iterable[str], sim_model: str | None = None
) -> None:
    """Check target, simulator and TTS credentials before opening a realtime session."""
    _require(settings, _generation_requirements(settings, target_models, sim_model))


def require_evaluation_credentials(
    settings: Settings,
    *,
    judge_model: str | None = None,
    voice_judge_model: str | None = None,
    modality: str = "audio",
    use_asr: bool = False,
) -> None:
    """Saved-trajectory evaluation needs only the selected judges and optional ASR."""
    _require(
        settings,
        _evaluation_requirements(
            settings,
            judge_model=judge_model,
            voice_judge_model=voice_judge_model,
            modality=modality,
            use_asr=use_asr,
        ),
    )


def require_pipeline_credentials(
    settings: Settings,
    target_models: Iterable[str],
    *,
    sim_model: str | None = None,
    judge_model: str | None = None,
    voice_judge_model: str | None = None,
    modality: str = "audio",
    use_asr: bool = False,
) -> None:
    """Check every selected service before the shell pipeline starts generation."""
    groups = _generation_requirements(settings, target_models, sim_model)
    groups.extend(
        _evaluation_requirements(
            settings,
            judge_model=judge_model,
            voice_judge_model=voice_judge_model,
            modality=modality,
            use_asr=use_asr,
        )
    )
    _require(settings, groups)


def main() -> int:
    """Internal shell entrypoint; checks presence only and performs no API requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", action="append", required=True)
    parser.add_argument("--sim-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--voice-judge-model")
    parser.add_argument("--modality", choices=("audio", "text"), default="audio")
    parser.add_argument("--asr", action="store_true")
    args = parser.parse_args()
    try:
        require_pipeline_credentials(
            get_settings(),
            args.target_model,
            sim_model=args.sim_model,
            judge_model=args.judge_model,
            voice_judge_model=args.voice_judge_model,
            modality=args.modality,
            use_asr=args.asr,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
