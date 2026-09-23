"""报告导出：JSON 落盘与按需格式。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_json(data: Any, out_path: str | Path) -> None:
    """导出 JSON（UTF-8、缩进、ensure_ascii=False）。"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def export_jsonl(items: list[Any], out_path: str | Path) -> None:
    """导出 JSONL。"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
