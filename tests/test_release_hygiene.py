"""发布闸门：仓库里不许出现内部接口、绝对路径、凭据或未发布的数据资产。

这条测试是"裁判端点已去内网化"的**长期凭证**：改造是一次性的，回潮是无声的
（某次顺手把内部型号写回 model_registry 或 .env.example，CI 也不会红）。
所以把"不许出现什么"写成断言，而不是写进 README 靠人记。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 内网代理、未发布端点、内部台账路径
BANNED_IN_SOURCE = (
    r"alibaba-inc",
    r"idealab",
    r"modelbest",
    r"minicpm|MiniCPM",
    r"gemini-3\.1-",           # 只存在于内部网关的旧别名\n    r"本仓数据目录",              # 绝对路径
    r"(?:^|[^A-Za-z])internal/",  # 内部产物目录
    r"sk-[A-Za-z0-9]{16,}",     # 像真 key 的东西
    r"DASHSCOPE_API_KEY=[A-Za-z0-9]",  # .env.example 里不许带值
)
# 发布范围之外、但容易被顺手带进来的产物
BANNED_PATHS = (
    "internal", "runs", "reports", "archive", "site", "logs", ".venv", "data/audio",
)
# 声明禁词的这两份文件本身必然含那些字面量（扫描器不能扫自己），
# 由 test_banned_directories_absent / test_judge_backend_is_swappable 反向守。
SELF_EXEMPT = {"tests/test_release_hygiene.py", "tests/test_model_registry.py"}

# 内部研发代号（方案编号 / 未发布脚本名 / 清洗前默认模型名等）。这些只该出现在
# 开发期的 tests/ 简写里，**绝不该出现在随论文发布的产物面**（审稿人/复现者会读的文件）。
# 把"不许出现什么"写成断言，是因为清洗时最容易出的事故就是把代号写进了用户可见的
# 字符串/报表标题/数据注记（本次就踩过：_indent_block 的空格串、render_md 的标题）。
BANNED_IN_SHIPPED = (
    r"planS\d", r"planU", r"Plan [A-Z]", r"变更日志", r"内部产物", r"内部阶段",
    r"kimi-k3", r"qwen3\.8-max", r"MODELS_CROSS", r"taskJ", r"probe_endpoint_liveness",
    r"make_canonical_keyturn", r"维护者", r"mainwin", r"step2_", r"purify_surface",
    r"probe_t3", r"voice_audit",
)
# 随论文发布的产物面（顶层目录 + 根文件）；tests/ 允许保留开发期简写，不在其列。
_SHIPPED_TOP = ("src", "scripts", "docs", "configs", "prompts", "data")
_SHIPPED_FILES = {"README.md", ".env.example", "CITATION.bib", "pyproject.toml",
                  "Makefile", "LICENSE", "LICENSE_DATASET"}
_SHIPPED_SUFFIX = {".py", ".md", ".json", ".jsonl", ".sh", ".toml", ".example", ""}


def _source_files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or ".git/" in str(p.relative_to(ROOT)):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.split("/")[0] in BANNED_PATHS or rel in SELF_EXEMPT:
            continue
        yield rel, p


def test_no_internal_endpoint_or_path_in_tracked_sources():
    """全仓（除声明禁词的两份测试）不得出现内部端点/绝对路径/凭据形状。"""
    offenders = []
    for rel, p in _source_files():
        if p.suffix not in {".py", ".md", ".json", ".jsonl", ".sh", ".toml", ".example", ""}:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pat in BANNED_IN_SOURCE:
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(pat, line):
                    offenders.append(f"{rel}:{i}: {pat} → {line.strip()[:80]}")
    assert not offenders, "发现内部痕迹：\n" + "\n".join(offenders[:20])


def test_no_internal_dev_codenames_in_shipped_artifacts():
    """随论文发布的产物面不得出现内部研发代号（方案编号 / 未发布脚本 / 清洗前模型名）。

    作用域故意只覆盖 src/scripts/docs/configs/prompts/data + 根文件，不扫 tests/：
    测试文件是开发面，允许保留 planS/Plan X 之类简写；发布产物一旦漂回代号就红。
    """
    offenders = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or ".git/" in str(p.relative_to(ROOT)):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel not in _SHIPPED_FILES and (
            rel.split("/")[0] not in _SHIPPED_TOP or p.suffix not in _SHIPPED_SUFFIX
        ):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pat in BANNED_IN_SHIPPED:
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(pat, line):
                    offenders.append(f"{rel}:{i}: {pat} → {line.strip()[:80]}")
    assert not offenders, "发布产物里出现内部研发代号：\n" + "\n".join(offenders[:20])


def test_banned_directories_absent():
    present = [d for d in BANNED_PATHS if (ROOT / d).exists()]
    assert not present, f"发布仓不该带这些目录：{present}"


def test_no_dotenv_file():
    assert not (ROOT / ".env").exists(), ".env 属凭据文件，绝不允许入库"


def test_environment_template_has_no_values():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        assert value.strip() == "" or key.strip() in {
            "LLM_MODEL", "OPENAI_BASE_URL", "TTS_MODEL", "ASR_MODEL",
            "VOICE_JUDGE_MODEL", "JUDGE_MODEL", "SIM_MODEL", "AGENT_PROMPT",
            "TOOLS_ENABLED", "TTS_VOICE_USER", "REALTIME_VOICE"}, line
    assert "DASHSCOPE_API_KEY=" in text


def test_judge_backend_is_swappable():
    """裁判必须能整体换成公开可调用端点（发布仓的默认值不得是内部型号）。"""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from listen2serve.model_registry import MODEL_BINDINGS, resolve_model
    from listen2serve.runtime.config import Settings

    s = Settings(_env_file=None)
    assert resolve_model("openai/whatever-judge", "x")[0] == "openai"
    for spec in (s.llm_model, s.judge_model, s.sim_model):
        if spec:
            assert MODEL_BINDINGS.get(spec, {}).get("platform") != "idealab", spec


def test_release_assets_present():
    for rel in ("LICENSE", "LICENSE_DATASET", "README.md", "CITATION.bib",
                "data/benchmark/version.json", "data/benchmark/DATASET_CARD.md",
                "data/benchmark/subsets/balanced145_T1.json",
                "data/benchmark/subsets/balanced145_T3.json",
                "data/benchmark/audio_samples/manifest.jsonl",
                "prompts/MANIFEST.json", "docs/limitations.md"):
        assert (ROOT / rel).exists(), f"缺发布件：{rel}"
