"""Plan F：run_manifest 版本指纹单测。

覆盖点：
1. code_fingerprint git 可用分支（commit + dirty 列表）与不可用分支（fallback 聚合哈希）；
2. sanitize_settings 密钥字段零落盘；
3. build_run_manifest 五要素齐全（code/data/prompts/models/cli_args）。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from listen2serve.runtime import manifest as mf
from listen2serve.runtime.config import Settings


def _git_available() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True,
                              check=False).returncode == 0
    except OSError:
        return False


def _git(tmp: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=str(tmp), check=True, capture_output=True,
                   env={"GIT_CONFIG_GLOBAL": "/dev/null", "HOME": str(tmp), "PATH": "/usr/bin:/bin"})


@pytest.mark.skipif(not _git_available(), reason="环境无 git")
def test_code_fingerprint_git_branch(tmp_path: Path) -> None:
    """git 分支：commit 哈希 + 干净/脏文件列表。"""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init")
    fp = mf.code_fingerprint(tmp_path)
    assert fp["git_commit"] and len(fp["git_commit"]) == 40
    assert fp["git_dirty"] == []
    assert fp["fallback_tree_hash"] is None
    # 制造未提交改动 → dirty 列表非空
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    fp2 = mf.code_fingerprint(tmp_path)
    assert "a.py" in fp2["git_dirty"]


def test_code_fingerprint_fallback_branch(tmp_path: Path) -> None:
    """无 git 分支：fallback_tree_hash 稳定且对文件内容敏感。"""
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.py").write_text("v = 1\n", encoding="utf-8")
    (tmp_path / "internal" / "configs").mkdir(parents=True)
    fp1 = mf.code_fingerprint(tmp_path)
    assert fp1["git_commit"] is None
    assert fp1["fallback_tree_hash"] and len(fp1["fallback_tree_hash"]) == 64
    assert mf.code_fingerprint(tmp_path)["fallback_tree_hash"] == fp1["fallback_tree_hash"]
    (src / "m.py").write_text("v = 2\n", encoding="utf-8")
    assert mf.code_fingerprint(tmp_path)["fallback_tree_hash"] != fp1["fallback_tree_hash"]


def test_sanitize_settings_no_secrets() -> None:
    """密钥字段（*_api_key*）一律不出现在 config_snapshot。"""
    settings = Settings(dashscope_api_key="sk-secret-123",
                        openai_api_key="oa-secret-456")
    snapshot = mf.sanitize_settings(settings)
    dumped = json.dumps(snapshot, ensure_ascii=False)
    assert "sk-secret-123" not in dumped
    assert "ik-secret-456" not in dumped
    assert not any("key" in k for k in snapshot)
    # 非密字段保留
    assert snapshot["llm_model"]


def test_build_run_manifest_five_elements(tmp_path: Path) -> None:
    """五要素齐全；数据版本文件缺失时 data=None 不报错。"""
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.py").write_text("v = 1\n", encoding="utf-8")
    settings = Settings(dashscope_api_key="sk-x")
    manifest = mf.build_run_manifest(
        run_id="smoke_test",
        models={"target": "t", "sim": "s", "judge": "j", "voice_judge": "vj", "tts": "x"},
        cli_args={"run_id": "smoke_test"},
        settings=settings,
        root=tmp_path,
    )
    for key in ("code", "data", "prompts", "models", "cli_args", "config_snapshot"):
        assert key in manifest
    assert manifest["run_id"] == "smoke_test"
    assert manifest["data"] is None  # tmp 根无 version.json
    assert manifest["prompts"]["judge_prompt_version"]
    # 落盘/读回往返一致；无 manifest 目录读取返回 None
    path = mf.write_run_manifest(tmp_path / "runs" / "smoke_test", manifest)
    assert mf.load_run_manifest(path.parent) == manifest
    assert mf.load_run_manifest(tmp_path / "runs" / "nope") is None


def test_build_run_manifest_real_root() -> None:
    """真实项目根：prompt/配置资产指纹非空，密钥零泄漏。"""
    settings = Settings()
    manifest = mf.build_run_manifest(
        run_id="t", models={}, cli_args={}, settings=settings,
    )
    prompts = manifest["prompts"]
    assert prompts["sim_prompt_file_sha256"]
    assert prompts["judge_prompts_file_sha256"]
    assert prompts["policy_domains_sha256"]
    assert manifest["data"] and manifest["data"].get("dataset_version")
    dumped = json.dumps(manifest, ensure_ascii=False)
    for secret in (settings.dashscope_api_key, settings.openai_api_key):
        if secret:
            assert secret not in dumped
