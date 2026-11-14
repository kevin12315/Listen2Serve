"`prompts/*.md` 快照必须与代码常量逐字一致（发布闸门）"

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_dump_module():
    """与导出脚本共用同一份清单（不复制第二份，否则两边各改一半必漂移）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("dump_prompts", ROOT / "scripts" / "dump_prompts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODULES = _load_dump_module().MODULES


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _expected() -> dict[str, str]:
    out: dict[str, str] = {}
    for modname, pat in MODULES.items():
        mod = importlib.import_module(modname)
        for name in sorted(n for n in dir(mod) if re.match(pat, n)):
            value = getattr(mod, name)
            if callable(value):
                continue
            body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
            slug = f"{modname.split('.')[-2 if modname.endswith('prompts') else -1]}_{name}".lower()
            out[f"{slug}.md"] = sha(body)
    return out


def test_manifest_lists_every_snapshot():
    manifest = json.loads((ROOT / "prompts" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert {p.name for p in (ROOT / "prompts").glob("*.md")} == set(manifest["files"])
    assert set(manifest["files"]) == set(_expected()), "快照清单与代码常量不一致，请重跑 dump_prompts.py"


def test_snapshot_bodies_match_code_constants():
    for fname, digest in _expected().items():
        text = (ROOT / "prompts" / fname).read_text(encoding="utf-8")
        body = text.split("```text\n", 1)[1].rsplit("\n```", 1)[0]
        assert sha(body) == digest, f"{fname} 与代码常量不一致（改了一处没改另一处）"


def test_snapshot_keeps_slots_visible():
    """判据里的槽位必须原样保留，否则读者无法知道运行时填了什么。"""
    keyturn = (ROOT / "prompts" / "evaluation_keyturn_judge_prompt.md").read_text(encoding="utf-8")
    for slot in ("policy_rules", "history", "oracle_state", "agent_reply"):
        assert "{" + slot + "}" in keyturn, f"KeyTurn 量规缺槽位 {slot}"
