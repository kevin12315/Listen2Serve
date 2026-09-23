#!/usr/bin/env python3
"""把所有 prompt/量规导出成 `prompts/*.md` 纯文本快照 + `prompts/MANIFEST.json`。

为什么要快照（代码才是真值源，快照会漂移）：论文 承诺公开"模拟与服务提示、评分量规"，
而它们散在 `evaluation/prompts/__init__.py`（27 KB）与 `runtime/agent.py` 的常量里。审稿人
或者想复算分数的同行，不该为了看一份判据去读 Python。快照 + `tests/test_prompts_snapshot.py`
的组合解决这个矛盾：**快照与代码常量不一致就红**，所以快照可信；看的人不用跑代码。

用法：PYTHONPATH=src python scripts/dump_prompts.py
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "prompts"
sys.path.insert(0, str(ROOT / "src"))

# 模块 → 要导出的常量名（正则匹配，避免点名点漏；新增量规会自动进快照）
MODULES = {
    "listen2serve.evaluation.prompts": r"^(KEYTURN|POLICY|FLOW|VOICE|DIALOGUE|LEAKAGE)_.*PROMPT$|"
    r"^KEYTURN_ACTION_CHEATSHEET$",
    "listen2serve.runtime.agent": r"^VOICE_CALL_INSTRUCTION(_V\d[_A-Z]*)?$",
    "listen2serve.domains.base": r"^(ACTION_V9|STATE_ACTION_TABLE|POLICY_SECTION_KEYS)$",
    "listen2serve.runtime.user_simulator": r"^(OUTBOUND_OPENING_POOL|STATES|"
    r"MIN_UTTERANCE_CHARS|MAX_UTTERANCE_CHARS)$",
}
HEADER = """# {name}

- 来源：`{module}` 的常量 `{const}`
- sha256：`{sha}`
- 说明：本文件是**导出快照**，真值源是代码常量；二者不一致由
 `tests/test_prompts_snapshot.py` 判红。运行时填充的槽位以 {{…}} 形式保留在原文里。
"""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for modname, pat in MODULES.items():
        mod = importlib.import_module(modname)
        for name in sorted(n for n in dir(mod) if re.match(pat, n)):
            value = getattr(mod, name)
            if callable(value):
                continue
            body = render(value)
            slug = f"{modname.split('.')[-2 if modname.endswith('prompts') else -1]}_{name}".lower()
            path = OUT / f"{slug}.md"
            path.write_text(HEADER.format(name=name, module=modname, const=name, sha=sha(body))
                            + "\n```text\n" + body + "\n```\n", encoding="utf-8")
            manifest[path.name] = sha(body)
    (OUT / "MANIFEST.json").write_text(json.dumps(
        {"note": "prompt/量规的导出快照（真值源在代码常量里）；由 scripts/dump_prompts.py 生成",
         "files": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"导出 {len(manifest)} 份 prompt/量规快照 → prompts/")
    for f in sorted(manifest):
        print(f" {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
