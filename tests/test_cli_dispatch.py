"""发布版命令派发及移除开发参数后的入口检查。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from listen2serve import cli


def test_run_eval_dispatch(monkeypatch):
    seen = {}

    async def fake_run_eval(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_eval", fake_run_eval)
    monkeypatch.setattr(sys, "argv", ["l2s", "run-eval", "--model", "stub/whatever", "--limit", "1"])
    assert cli.main() == 0
    assert seen["args"].model == "stub/whatever"


def test_evaluate_dispatch(monkeypatch):
    seen = {}

    def fake_evaluate(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli, "_evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["l2s", "evaluate", "--run-id", "smoke_x"])
    assert cli.main() == 0
    assert seen["args"].run_id == "smoke_x"


@pytest.mark.parametrize("flag,value", [("--agent-prompt", "v9"), ("--sim-prompt-schema", "v6.0")])
def test_development_prompt_switches_are_not_public(monkeypatch, flag, value):
    monkeypatch.setattr(sys, "argv", ["l2s", "run-eval", "--model", "stub/whatever", flag, value])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_no_undefined_names_anywhere_in_shipped_code():
    root = Path(cli.__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "F821",
            "--output-format",
            "concise",
            "src",
            "scripts",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, "发现未定义名：\n" + (proc.stdout or proc.stderr)
