"""Local credential presence checks precede billed work and preserve offline reuse."""

from __future__ import annotations

import argparse
import asyncio
import json

import pytest

from listen2serve.runtime.config import Settings
from listen2serve.runtime.preflight import (
    require_evaluation_credentials,
    require_generation_credentials,
    require_pipeline_credentials,
)


def settings(**changes) -> Settings:
    # Never inherit developer credentials from the environment or repository .env.
    credentials = {key: None for key in Settings.model_fields if "api_key" in key}
    return Settings(_env_file=None, **(credentials | changes))


def test_generation_reports_all_missing_variables_without_values():
    with pytest.raises(ValueError) as caught:
        require_generation_credentials(settings(), ["qwen3.5-omni-plus-realtime"])
    assert str(caught.value) == "缺少凭据: DASHSCOPE_API_KEY, MOONSHOT_API_KEY"


def test_generation_model_override_uses_only_selected_services():
    require_generation_credentials(
        settings(dashscope_api_key="fake-dashscope", openai_api_key="fake-openai"),
        ["qwen3.5-omni-plus-realtime"],
        sim_model="openai/offline-simulator",
    )
    require_generation_credentials(
        settings(dashscope_api_key="fake-dashscope"),
        ["qwen3.5-omni-plus-realtime"],
        sim_model="local/offline-simulator",
    )


def test_doubao_second_key_is_a_valid_alternative_and_tts_still_needs_dashscope():
    configured = settings(volc_api_key2="fake-volc", moonshot_api_key="fake-moonshot")
    with pytest.raises(ValueError, match="^缺少凭据: DASHSCOPE_API_KEY$"):
        require_generation_credentials(configured, ["doubao-seeduplex-3.0"])
    require_generation_credentials(
        configured.model_copy(update={"dashscope_api_key": "fake-dashscope"}),
        ["doubao-seeduplex-3.0"],
    )


def test_pipeline_checks_all_endpoints_and_later_judges_before_generation():
    configured = settings(dashscope_api_key="fake-dashscope", moonshot_api_key="fake-moonshot")
    with pytest.raises(ValueError) as caught:
        require_pipeline_credentials(
            configured, ["qwen3.5-omni-plus-realtime", "doubao-seeduplex-3.0"], use_asr=True
        )
    assert str(caught.value) == "缺少凭据: VOLC_API_KEY / VOLC_API_KEY2, GEMINI_API_KEY"
    assert "fake-" not in str(caught.value)


def test_evaluation_requires_only_its_selected_backends_and_asr():
    configured = settings(openai_api_key="fake-openai")
    require_evaluation_credentials(
        configured, judge_model="openai/offline", voice_judge_model="openai/offline"
    )
    require_evaluation_credentials(
        configured.model_copy(update={"dashscope_api_key": "fake-dashscope"}),
        judge_model="openai/offline",
        voice_judge_model="google/gemini-3.8-flash",
        modality="text",
    )
    with pytest.raises(ValueError, match="^缺少凭据: DASHSCOPE_API_KEY$"):
        require_evaluation_credentials(
            configured, judge_model="openai/offline", modality="text", use_asr=False
        )
    with pytest.raises(ValueError, match="^缺少凭据: DASHSCOPE_API_KEY$"):
        require_evaluation_credentials(
            configured,
            judge_model="openai/offline",
            voice_judge_model="openai/offline",
            use_asr=True,
        )


def test_empty_or_whitespace_credential_is_missing():
    with pytest.raises(ValueError, match="^缺少凭据: MOONSHOT_API_KEY$"):
        require_generation_credentials(
            settings(moonshot_api_key="  ", dashscope_api_key="fake-dashscope"),
            ["qwen3.5-omni-plus-realtime"],
        )


def test_run_eval_rejects_missing_simulator_before_constructing_runner(tmp_path, monkeypatch):
    from listen2serve import cli
    from listen2serve.runtime import config, run_batch

    monkeypatch.setattr(config, "get_settings", lambda: settings(dashscope_api_key="fake-dashscope"))
    monkeypatch.setattr(cli, "_load_scenarios", lambda *args: [{"scenario_id": "one", "role": "hotline"}])

    def fail_if_constructed(*args, **kwargs):
        pytest.fail("BatchRunner was constructed before validating credentials")

    monkeypatch.setattr(run_batch, "BatchRunner", fail_if_constructed)
    args = argparse.Namespace(
        scenarios="unused",
        limit=1,
        role=None,
        wording=None,
        scenario_ids=None,
        per_role=None,
        model="qwen3.5-omni-plus-realtime",
        sim_model=None,
    )
    with pytest.raises(ValueError, match="^缺少凭据: MOONSHOT_API_KEY$"):
        asyncio.run(cli._run_eval(args))
    assert not (tmp_path / "runs").exists()


def test_independent_evaluate_needs_no_simulator_or_target_credentials(tmp_path, monkeypatch):
    from test_evaluate_incremental import _evaluation_cli_fixture

    from listen2serve import cli
    from listen2serve.runtime import config

    args, _, calls, _ = _evaluation_cli_fixture(tmp_path, monkeypatch)
    args.judge_model = args.voice_judge_model = "openai/offline-judge"
    args.asr = False
    path = next((tmp_path / "runs/case").glob("*/sim_status.json"))
    trajectory = json.loads(path.read_text())
    trajectory["turns"][0]["agent_text"] = "统一转写文本"
    path.write_text(json.dumps(trajectory))
    monkeypatch.setattr(config, "get_settings", lambda: settings(openai_api_key="fake-openai"))
    assert cli._evaluate(args) == 0
    assert not calls


def test_complete_evaluation_cache_needs_no_credentials(tmp_path, monkeypatch):
    from test_evaluate_incremental import _evaluation_cli_fixture

    from listen2serve import cli
    from listen2serve.runtime import config

    args, _, asr_calls, prompts = _evaluation_cli_fixture(tmp_path, monkeypatch)
    assert cli._evaluate(args) == 0
    calls = len(asr_calls)
    prompts.clear()
    monkeypatch.setattr(config, "get_settings", lambda: settings())
    assert cli._evaluate(args) == 0
    assert not prompts and len(asr_calls) == calls


def test_missing_fresh_evaluation_credentials_preserves_existing_cache(tmp_path, monkeypatch):
    from test_evaluate_incremental import _evaluation_cli_fixture

    from listen2serve import cli
    from listen2serve.runtime import config

    args, _, _, _ = _evaluation_cli_fixture(tmp_path, monkeypatch)
    assert cli._evaluate(args) == 0
    partial_path = tmp_path / "reports/case/verdicts_partial_audio.jsonl"
    original = partial_path.read_bytes()
    args.fresh = True
    monkeypatch.setattr(config, "get_settings", lambda: settings())
    with pytest.raises(ValueError, match="缺少凭据"):
        cli._evaluate(args)
    assert partial_path.read_bytes() == original
    assert not list(partial_path.parent.glob("*.stale_*"))


@pytest.mark.parametrize("stored_status", ["partial", "ok"])
def test_partial_evaluation_retains_key_and_voice_but_skips_whole_dialogue(
    tmp_path, monkeypatch, stored_status
):
    from test_evaluate_incremental import _evaluation_cli_fixture

    from listen2serve import cli
    from listen2serve.evaluation.flow_judge import FlowJudge

    args, _, asr_calls, _ = _evaluation_cli_fixture(tmp_path, monkeypatch)
    path = tmp_path / "runs/case/case/sim_status.json"
    if not path.exists():
        path = next((tmp_path / "runs/case").glob("*/sim_status.json"))
    trajectory = json.loads(path.read_text())
    trajectory.update(status=stored_status, termination_reason="endpoint_session_closed")
    trajectory["turns"].append(
        {
            "turn": 2,
            "user_text": "尚未完成",
            "agent_text": "断线片段",
            "is_key_turn": False,
            "endpoint_interrupted": True,
            "agent_audio_path": "missing-tail.wav",
        }
    )
    path.write_text(json.dumps(trajectory))

    def no_whole_dialogue_call(*args, **kwargs):
        pytest.fail("Partial call must not be sent to the whole-dialogue judge")

    monkeypatch.setattr(FlowJudge, "judge", no_whole_dialogue_call)
    assert cli._evaluate(args) == 0
    verdict = json.loads((tmp_path / "reports/case/verdicts_audio.json").read_text())[0]
    assert verdict["trajectory_status"] == "partial" and verdict["dialogue_complete"] is False
    assert verdict["key_turn_pass"] is True and verdict["voice_pass"] is True
    assert verdict["flow_rate"] is None and verdict["task_score"] is None
    assert verdict["flow_detail"] is None and verdict["closing_ok"] is None
    assert len(asr_calls) == 1


def test_partial_interrupted_key_reply_is_rejected_before_judging(tmp_path, monkeypatch):
    from test_evaluate_incremental import _evaluation_cli_fixture

    from listen2serve import cli

    args, _, calls, prompts = _evaluation_cli_fixture(tmp_path, monkeypatch)
    path = next((tmp_path / "runs/case").glob("*/sim_status.json"))
    trajectory = json.loads(path.read_text())
    trajectory.update(status="partial", termination_reason="endpoint_session_closed")
    trajectory["turns"][0]["endpoint_interrupted"] = True
    path.write_text(json.dumps(trajectory))
    with pytest.raises(ValueError, match="关键轮客服回复不完整"):
        cli._evaluate(args)
    assert not calls and not prompts


def test_report_uses_saved_results_without_loading_credentials(tmp_path, monkeypatch):
    import sys

    from listen2serve import cli
    from listen2serve.runtime import config

    monkeypatch.chdir(tmp_path)
    report = tmp_path / "reports/saved"
    report.mkdir(parents=True)
    (report / "verdicts_audio.json").write_text(
        json.dumps(
            [
                {
                    "scenario_id": "saved",
                    "role": "hotline",
                    "wording": "neutral",
                    "dynamics": "all_positive",
                }
            ]
        )
    )

    def no_settings():
        pytest.fail("report must not load credentials")

    monkeypatch.setattr(config, "get_settings", no_settings)
    monkeypatch.setattr(sys, "argv", ["l2s", "report", "--run-id", "saved"])
    assert cli.main() == 0
    assert (report / "report_audio.md").is_file()


def test_doubao_whitespace_first_key_falls_through_to_second_without_trimming_values():
    configured = settings(
        volc_api_key="   ", volc_api_key2="fake-valid",
        dashscope_api_key="fake-dashscope", moonshot_api_key="fake-moonshot",
    )
    require_generation_credentials(configured, ["doubao-seeduplex-3.0"])
    assert configured.volc_keys() == ["fake-valid"]
    assert configured.volc_key() == "fake-valid"
    untrimmed = configured.model_copy(update={"volc_api_key": "  fake-with-padding  "})
    assert untrimmed.volc_keys()[0] == "  fake-with-padding  "
