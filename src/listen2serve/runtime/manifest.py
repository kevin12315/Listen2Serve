"""run_manifest.json 生成与读取（内部方案：运行溯源与版本指纹）。

每个 run 目录写入一份 manifest，只看 runs/<run_id>/run_manifest.json 即可回答：
用了哪个代码状态（git commit / fallback tree hash）、哪个数据版本
（data/benchmark/version.json 全文）、哪些 prompt/配置资产指纹、哪些模型与
CLI 参数、以及 .env 覆盖后的最终生效配置快照（密钥一律不落盘）。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from listen2serve.runtime.config import PROJECT_ROOT, Settings

MANIFEST_FILENAME = "run_manifest.json"

# ---- 指纹资产路径（相对项目根）----
SIM_PROMPT_FILE = Path("src/listen2serve/runtime/user_simulator.py")
JUDGE_PROMPTS_FILE = Path("src/listen2serve/evaluation/prompts/__init__.py")
# 发布仓不带数据生成器，故"策略模板库"的指纹改指**服务规范的实现本体**
# （domains/ 里的板块渲染与状态→动作速查表）—— 被测与被评双方共享的规则在此。
POLICY_DOMAINS_DIR = Path("src/listen2serve/domains")
DATA_VERSION_FILE = Path("data/benchmark/version.json")
# git 不可用时的代码指纹聚合范围（与计划约定一致）
FALLBACK_CODE_DIRS = ("src",) # 发布仓只有 src/ 一份代码目录

# Settings 字段名含以下子串视为密钥/敏感字段，config_snapshot 一律不落盘
_SECRET_MARKERS = ("key", "token", "secret", "password")


def observed_judge(man: dict[str, Any] | None) -> dict[str, Any]:
    """这个 run **实际**用了哪个裁判 —— 用于一切对外展示（报告 md / 网页 / 分析）。

    为什么需要它：`models.judge` 是**生成步**写的，那一刻裁判还没被调用过。
    2026-09-07 之前它写的是 `settings.llm_model`（默认 qwen-plus），于是任何
    "用别的裁判评"或"补跑把 manifest 重置"的 run，报告和 dashboard 都会把
    **没参与判定的裁判**展示给评审（实测：verdicts 全是另一型号，
    而 report_audio.md 写 qwen-plus）。现在生成步写 None，真实值由
    `run_experiment.sh` 的②b 回写进 `evaluation`。

    优先级：实测值 > 回写声明值 > 生成期占位。`conflict` 表示三者不一致，
    调用方必须把它显式渲染出来 —— 静默挑一个就是替别人掩盖矛盾。
    """
    ev = (man or {}).get("evaluation") or {}
    models = (man or {}).get("models") or {}
    obs = ev.get("keyturn_model_observed")
    declared = ev.get("judge_model")
    gen_claim = models.get("judge")
    shown = obs or declared or gen_claim
    voice = ev.get("voice_judge_model") or models.get("voice_judge")
    conflict = bool(gen_claim) and bool(shown) and gen_claim != shown
    return {"judge": shown, "voice_judge": voice, "observed": obs,
            "declared": declared, "generation_claim": gen_claim, "conflict": conflict}


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
    for f in sorted(p for p in path.rglob("*") if p.is_file()
                    and "__pycache__" not in p.parts):
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
            ("git", *cmd), cwd=str(root), capture_output=True, text=True, timeout=30,
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
        dirty = sorted({line[3:].strip().strip('"') for line in (porcelain or "").splitlines()
                        if line.strip()})
        return {"git_commit": commit, "git_dirty": dirty, "fallback_tree_hash": None}
    # 无 git：对 src/ + （未发布） 做 sha256 聚合
    parts = [f"{d}:{sha256_dir(root / d) or ''}" for d in FALLBACK_CODE_DIRS]
    fallback = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return {"git_commit": None, "git_dirty": [], "fallback_tree_hash": fallback}


def sanitize_settings(settings: Settings) -> dict[str, Any]:
    """生效配置快照（非密字段）：字段名命中密钥模式的一律剔除。"""
    return {k: v for k, v in settings.model_dump().items()
            if not any(m in k.lower() for m in _SECRET_MARKERS)}


def build_run_manifest(
    run_id: str,
    models: dict[str, str],
    cli_args: dict[str, Any],
    settings: Settings,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """组装 run_manifest.json 五要素：code / data / prompts / models / cli_args。"""
    from listen2serve.evaluation.prompts import JUDGE_PROMPT_VERSION

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
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
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
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    """读取 run 的 manifest；旧 run 无 manifest 或损坏时返回 None。"""
    path = Path(run_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
