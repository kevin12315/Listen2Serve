"""Checks for publishable files; ignored local credentials and runs are allowed."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def publication_paths(root: Path = ROOT) -> list[str]:
    """Include tracked files and new, nonignored release files."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(set(result.stdout.decode().split("\0")) - {""})


def test_local_outputs_and_credentials_are_not_published():
    forbidden = {".env", ".venv", "runs", "reports", "logs", "archive", "internal"}
    assert not [p for p in publication_paths() if p.split("/")[0] in forbidden]


def test_ignored_working_directories_are_allowed(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".env\n.venv/\nruns/\nreports/\n")
    for name in (".venv", "runs", "reports"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "local").write_text("local output")
    (tmp_path / ".env").write_text("MOONSHOT_API_KEY=local-only")
    (tmp_path / "code.py").write_text("x = 1\n")
    assert set(publication_paths(tmp_path)) == {".gitignore", "code.py"}


def test_published_sources_have_no_credentials_or_development_endpoints():
    patterns = (r"idealab", r"modelbest", r"sk-[A-Za-z0-9]{24,}")
    for rel in publication_paths():
        p = ROOT / rel
        if not p.is_file() or rel == "tests/test_release_hygiene.py":
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in patterns:
            assert not re.search(pattern, content), f"{rel}: {pattern}"


def test_public_cli_does_not_expose_development_versions():
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "listen2serve.cli", "run-eval", "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "--wording" in result.stdout
    assert "--agent-prompt" not in result.stdout
    assert "--sim-prompt-schema" not in result.stdout
    assert "--leakage" not in result.stdout


def test_environment_template_has_empty_credentials():
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.endswith(("API_KEY", "API_KEY2")):
            assert not value.strip(), key


def test_release_assets_present():
    for rel in (
        "LICENSE",
        "LICENSE_DATASET",
        "NOTICE",
        "README.md",
        "CITATION.bib",
        "data/benchmark/version.json",
        "data/benchmark/DATASET_CARD.md",
        "data/benchmark/subsets/explicit.json",
        "data/benchmark/subsets/neutral.json",
        "data/benchmark/audio_samples/manifest.jsonl",
        "prompts/MANIFEST.json",
    ):
        assert (ROOT / rel).exists(), rel


ALLOWED_LICENSES = {"Apache-2.0", "CC-BY-NC-4.0", "CC-BY-NC-ND-4.0"}
_SCOPE_ROW = re.compile(r"^\|\s*`(?P<path>[^`]+)`\s*\|\s*(?P<lic>[A-Za-z0-9.\-]+)\s*\|")


def _licence_scope() -> dict[str, str]:
    """解析 LICENSE_DATASET 的「路径 → SPDX」表（全仓许可的唯一口径）。"""
    text = (ROOT / "LICENSE_DATASET").read_text(encoding="utf-8")
    scope = {}
    for line in text.splitlines():
        m = _SCOPE_ROW.match(line.strip())
        if m:
            scope[m.group("path")] = m.group("lic")
    return scope


def test_licence_scope_table_is_wellformed():
    scope = _licence_scope()
    assert scope, "LICENSE_DATASET 的范围表解析不出任何一行（表格格式被改坏了？）"
    unknown = {p: lic for p, lic in scope.items() if lic not in ALLOWED_LICENSES}
    assert not unknown, f"非白名单 SPDX 标识：{unknown}"
    missing = [p for p in scope if not (ROOT / p).exists()]
    assert not missing, f"范围表声明了不存在的路径（口径漂移）：{missing}"


def test_every_data_file_declares_a_licence():
    """`data/benchmark/` 下每一项都必须被范围表覆盖（精确路径或目录前缀）。

    新增数据文件时最容易发生的事：它悄悄继承"看起来最宽松"的那份许可。所以这里不
    列清单，而是拿真实目录树去比对表 —— 漏标即红。
    """
    scope = _licence_scope()
    top = ROOT / "data" / "benchmark"
    uncovered = []
    for p in sorted(top.iterdir()):
        rel = p.relative_to(ROOT).as_posix()
        rel_norm = rel + "/" if p.is_dir() else rel
        if rel_norm in scope:
            continue
        # 目录型条目以 `xxx/` 声明，覆盖其下全部内容
        if any(rel_norm.startswith(d) for d in scope if d.endswith("/")):
            continue
        uncovered.append(rel_norm)
    assert not uncovered, f"这些发布数据没有声明许可：{uncovered}"


def test_audio_demo_is_the_strongest_licence():
    """音频是 demo，许可必须比内容侧更严（ND 在）—— 否则"只是 demo"就成了空话。"""
    scope = _licence_scope()
    assert scope.get("data/benchmark/audio_samples/") == "CC-BY-NC-ND-4.0", (
        "audio_samples 的许可标识变了：demo 定性依赖 ND，改动需同步 README 与数据卡"
    )
    content = [
        lic for path, lic in scope.items() if path.startswith("data/benchmark/") and not path.endswith("/")
    ]
    assert "CC-BY-NC-4.0" in content, "基准内容侧应保留 CC BY-NC 4.0"


def test_readme_licence_table_matches_scope_table():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for label in ("Apache-2.0", "CC BY-NC 4.0", "CC BY-NC-ND 4.0"):
        assert label in readme
    assert _licence_scope()
