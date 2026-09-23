"""预生成 TTS instruction 表的加载与校验（内部方案 /）。

运行期的韵律指令不再由「LLM 自填 tone/rate/volume + 代码拼模板」产生，而是按
`base_scenario_id × state` 从离线表里查——索引键不含 `-T1/-T2/-T3`，于是「文本透明度
递减而情绪强度恒定」这个不变量由数据结构保证，不依赖代码约定。

本模块只做三件事：按路径加载并缓存表、算文件 sha256（进 run meta 指纹）、逐条做结构
校验。**任何缺失都硬失败**（）：表缺失、base 无条目、条目缺 `non_key[state]` 一律
`raise`，不回落到旧的模板拼装——静默回落正是上一版把「文本平静、声音愤怒」的错配藏了
一整轮的原因。构造期能查出的错就别留到运行期（`UserSimulator.__init__` 即校验条目）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from listen2serve.runtime.tts_tags import STATE_TAG_ALLOWED, allowed_tags_from_entry

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TABLE_PATH = PROJECT_ROOT / "data" / "benchmark" / "tts_instructions.json"


@dataclass(frozen=True)
class InstructionTable:
    """一份表的全部内容与身份（version + sha256 进 run meta，供事后复现核对）。"""

    path: Path
    dataset_version: str
    sha256: str
    entries: dict[str, dict[str, Any]]

    def entry(self, base_scenario_id: str) -> dict[str, Any]:
        """取 base 条目；缺失即数据契约违约，硬失败。"""
        try:
            return self.entries[base_scenario_id]
        except KeyError:
            raise ValueError(
                f"instruction 表缺 base 条目：base={base_scenario_id!r} "
                f"table={self.path}（共 {len(self.entries)} 条）"
            ) from None


_CACHE: dict[Path, InstructionTable] = {}


def load_table(path: str | Path | None = None) -> InstructionTable:
    """加载（并按绝对路径缓存）instruction 表；文件缺失或结构非法即 raise。"""
    p = Path(path) if path else DEFAULT_TABLE_PATH
    p = p.resolve()
    cached = _CACHE.get(p)
    if cached is not None:
        return cached
    if not p.exists():
        raise ValueError(f"instruction 表不存在：{p}（先跑 scripts/gen_tts_instructions.py）")
    raw = p.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"instruction 表不是合法 JSON：{p}（{e}）") from e
    entries = doc.get("instructions")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"instruction 表缺 instructions 段或为空：{p}")
    table = InstructionTable(
        path=p,
        dataset_version=str(doc.get("dataset_version") or ""),
        sha256=hashlib.sha256(raw).hexdigest(),
        entries=entries,
    )
    _CACHE[p] = table
    return table


def validate_entry(base_scenario_id: str, entry: dict[str, Any]) -> None:
    """条目结构校验（构造期调用，把运行期的 KeyError 提前成构造期的 ValueError）。

    只查「查表要用到的字段是否自洽」；文本质量类校验（长度、韵律词、情绪词、禁用词）
    归生成期的 `scripts/validate_tts_contract.py`，不在运行期重复。
    """
    where = f"base={base_scenario_id!r}"
    cand = entry.get("candidate_states")
    if not isinstance(cand, list) or not cand:
        raise ValueError(f"instruction 条目缺 candidate_states：{where}")
    non_key = entry.get("non_key")
    if not isinstance(non_key, dict):
        raise ValueError(f"instruction 条目缺 non_key：{where}")
    missing = [s for s in cand if not str(non_key.get(s) or "").strip()]
    if missing:
        raise ValueError(f"instruction 条目 non_key 缺 state {missing}：{where}")
    if not str(entry.get("key") or "").strip():
        raise ValueError(f"instruction 条目缺关键轮文本 key：{where}")
    if str(entry.get("key_state") or "") not in cand:
        raise ValueError(
            f"instruction 条目 key_state={entry.get('key_state')!r} 不在 candidate_states "
            f"{cand}：{where}"
        )
    # 标签「允许加什么」是全局不变量，必须代码硬管：表里写错一个标签就能静默改变韵律。
    for state, tag in (entry.get("tags") or {}).items():
        if state not in cand:
            raise ValueError(f"instruction 条目 tags 含非候选态 {state!r}：{where}")
        if tag and tag != STATE_TAG_ALLOWED.get(state):
            raise ValueError(
                f"instruction 条目 tags[{state}]={tag!r} 违反一态一标签"
                f"（应为 {STATE_TAG_ALLOWED.get(state)!r}）：{where}"
            )
    key_tag = entry.get("key_tag")
    if key_tag and key_tag != STATE_TAG_ALLOWED.get(str(entry.get("key_state") or "")):
        raise ValueError(f"instruction 条目 key_tag={key_tag!r} 违反一态一标签：{where}")
    declared = entry.get("allowed_tags")
    if declared is not None and list(declared) != allowed_tags_from_entry(entry):
        raise ValueError(
            f"instruction 条目 allowed_tags={declared} 与 tags/key_tag 推导结果 "
            f"{allowed_tags_from_entry(entry)} 不一致：{where}"
        )
