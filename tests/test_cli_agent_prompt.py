"""`--agent-prompt` 必须真的走到子命令派发（CLI 残枝防回归）。

防回归点：`--agent-prompt` 是 `scripts/run_experiment.sh` 与论文复现命令的正式入口，
但 `main()` 里 `if args.agent_prompt:` 这个分支曾在清洗未发布的探测脚本后留下两行残枝
（`_sys.argv = [...]` / `return probe_main()`，两个名字都不存在），于是**只要传了这个开关**
就在派发前 NameError —— 开关本身可用、命令必崩，最容易在发布后第一次复现时才暴露。

CI 抓不到它的两个原因，也正是这条测试存在的理由：
* 既有用例都直接调内部函数（`_run_eval` / `_evaluate`），不经过 `main()`；
* `ruff` 的 F821 能静态抓到，但 lint 不在 CI 闸门里（只有本地 `make lint`）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from listen2serve import cli
from listen2serve.domains.base import AGENT_PROMPT_VERSIONS
from listen2serve.runtime.config import get_settings


@pytest.fixture
def prompt_env_restored():
    """AGENT_PROMPT 是进程级环境变量 + Settings 的 lru_cache，两头都得还原。

    不还原会污染同批用例：`get_settings()` 被别处缓存成 v6/v7 时，裁判契约渲染
    版本会跟着悄悄变（口径同 test_judge.py 里"契约版本必须跟着 run 走"那条断言）。
    """
    saved = os.environ.pop("AGENT_PROMPT", None)
    get_settings.cache_clear()
    yield
    if saved is None:
        os.environ.pop("AGENT_PROMPT", None)
    else:
        os.environ["AGENT_PROMPT"] = saved
    get_settings.cache_clear()


@pytest.mark.parametrize("version", sorted(AGENT_PROMPT_VERSIONS))
def test_agent_prompt_flag_reaches_dispatch(monkeypatch, prompt_env_restored, version):
    """传 --agent-prompt：环境变量落地 + Settings 读到 + 走到 run-eval 派发。"""
    seen: dict = {}

    async def fake_run_eval(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_eval", fake_run_eval)
    monkeypatch.setattr(sys, "argv", [
        "l2s", "run-eval", "--model", "stub/whatever", "--limit", "1",
        "--agent-prompt", version,
    ])

    assert cli.main() == 0, "main() 未走到派发就返回/抛错"
    assert os.environ["AGENT_PROMPT"] == version
    assert get_settings().agent_prompt == version, "cache_clear 没生效：Settings 仍是旧版本"
    assert seen["args"].agent_prompt == version


def test_evaluate_subcommand_also_honours_flag(monkeypatch, prompt_env_restored):
    """evaluate 侧同源：裁判契约必须跟着 run 的 agent_prompt 走，否则两侧口径分叉。"""
    seen: dict = {}

    def fake_evaluate(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli, "_evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", [
        "l2s", "evaluate", "--run-id", "smoke_x", "--agent-prompt", "v7",
    ])

    assert cli.main() == 0
    assert os.environ["AGENT_PROMPT"] == "v7"
    assert seen["args"].agent_prompt == "v7"


def test_no_undefined_names_anywhere_in_shipped_code():
    """全仓（src + scripts）不许有未定义引用 —— 把 lint 里"必是 bug"的那一类提成闸门。

    用手写的 AST 扫描判"未定义名"会误报 `except ... as exc`、推导式作用域等绑定，
    所以直接调 ruff 的 F821（`ruff` 已在 dev 依赖里，CI 装 `.[dev]` 时就有）。
    范围故意只取 F821：`make lint` 目前还有 E741/F401/F811/F841 的历史欠账，
    把它们一并塞进测试会让这条"必崩"护栏被风格噪音淹掉。
    """
    import subprocess

    root = Path(cli.__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821",
         "--output-format", "concise", "src", "scripts"],
        cwd=root, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "发现未定义名（调用即 NameError）：\n" + (proc.stdout or proc.stderr)
    )
