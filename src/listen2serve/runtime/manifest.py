"""运行溯源：代码、数据、提示词、实际模型与不含密钥的生效配置。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from listen2serve.runtime.config import PROJECT_ROOT, Settings

MANIFEST_FILENAME = "run_manifest.json"

SIM_PROMPT_FILE = Path("src/listen2serve/runtime/user_simulator.py")
JUDGE_PROMPTS_FILE = Path("src/listen2serve/evaluation/prompts/__init__.py")
POLICY_DOMAINS_DIR = Path("src/listen2serve/domains")
DATA_VERSION_FILE = Path("data/benchmark/version.json")
# git 不可用时仍提供可复核的本地代码指纹。
FALLBACK_CODE_DIRS = ("src",)

# Settings 字段名含以下子串视为密钥/敏感字段，config_snapshot 一律不落盘
_SECRET_MARKERS = ("key", "token", "secret", "password")


def observed_judge(man: dict[str, Any] | None) -> dict[str, Any]:
    """优先展示评测实际观察的裁判；同时显式标识记录中的型号冲突。"""
    ev = (man or {}).get("evaluation") or {}
    models = (man or {}).get("models") or {}
    # Older releases wrote the configured model to an "observed" field.
    # Trust observations only when backed by the explicit requested/response audit schema.
    audited = ev.get("model_identity_schema") == "requested-response-v1"
    obs = ev.get("keyturn_model_observed") if audited else None
    declared = ev.get("judge_model")
    gen_claim = models.get("judge")
    shown = declared or gen_claim
    voice = ev.get("voice_judge_model") or models.get("voice_judge")
    conflict = bool(gen_claim) and bool(shown) and gen_claim != shown
    return {
        "judge": shown,
        "voice_judge": voice,
        "observed": obs,
        "voice_observed": ev.get("voice_model_observed") if audited else None,
        "declared": declared,
        "generation_claim": gen_claim,
        "conflict": conflict,
    }


def sha256_file(path: Path) -> str | None:
    """单文件 sha256（文件缺失返回 None）。"""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_dir(path: Path) -> str | None:
    """目录聚合 sha256：按相对路径排序后逐文件 "路径:哈希" 串联再哈希。

    缺失目录返回 None；__pycache__ 等缓存产物不参与指纹。"""
    if not path.is_dir():
        return None
    entries: list[str] = []
    for f in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rel = f.relative_to(path).as_posix()
        entries.append(f"{rel}:{hashlib.sha256(f.read_bytes()).hexdigest()}")
    digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return digest


def _git_output(root: Path, *cmd: str, rstrip: bool = False) -> str | None:
    """git 命令输出（失败/无 git 仓库返回 None）。

    rstrip=True 时仅去尾部空白：porcelain 行首的状态位空格属有效信息，
    不能整体 strip（否则首行错位）。"""
    try:
        out = subprocess.run(
            ("git", *cmd),
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.rstrip() if rstrip else out.stdout.strip()


def code_fingerprint(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """代码状态指纹：git 可用时取 commit + dirty 文件列表，否则回退目录聚合哈希。"""
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit:
        porcelain = _git_output(root, "status", "--porcelain", rstrip=True)
        dirty = sorted(
            {line[3:].strip().strip('"') for line in (porcelain or "").splitlines() if line.strip()}
        )
        return {
            "git_commit": commit,
            "git_dirty": dirty,
            "fallback_tree_hash": None,
            "source_tree_sha256": sha256_dir(root / "src"),
        }
    # 无 git 时，对发布代码做内容聚合。
    parts = [f"{d}:{sha256_dir(root / d) or ''}" for d in FALLBACK_CODE_DIRS]
    fallback = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return {"git_commit": None, "git_dirty": [], "fallback_tree_hash": fallback}


def sanitize_settings(settings: Settings) -> dict[str, Any]:
    """生效配置快照（非密字段）：字段名命中密钥模式的一律剔除。"""
    return {
        k: v
        for k, v in settings.model_dump().items()
        if k.endswith("_tokens") or not any(m in k.lower() for m in _SECRET_MARKERS)
    }


def build_run_manifest(
    run_id: str,
    models: dict[str, str | None],
    cli_args: dict[str, Any],
    settings: Settings,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """组装 run_manifest.json 五要素：code / data / prompts / models / cli_args。"""
    version_file = root / DATA_VERSION_FILE
    try:
        data_version = json.loads(version_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data_version = None
    return {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code": code_fingerprint(root),
        "data": data_version,
        "prompts": {
            "sim_prompt_file_sha256": sha256_file(root / SIM_PROMPT_FILE),
            "protocol": "multi-turn",
            "judge_prompts_file_sha256": sha256_file(root / JUDGE_PROMPTS_FILE),
            "policy_domains_sha256": sha256_dir(root / POLICY_DOMAINS_DIR),
        },
        "models": models,
        "cli_args": cli_args,
        "config_snapshot": sanitize_settings(settings),
    }


def write_run_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """manifest 落盘（run 目录内 run_manifest.json）。"""
    path = Path(run_dir) / MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    """Read a run manifest; absent or malformed data returns None."""
    path = Path(run_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
