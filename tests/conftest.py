"""测试公用夹具。

Plan L §4.1 起 `UserSimulator` 构造期必须拿到一条 instruction 表条目：合成场景
（`SYN-…`、`T` 之类 scenario_id）不在真实表里，走 `instruction_entry=` 注入口。
这是**唯一**被允许的旁路，且注入的条目同样要过 `validate_entry`——测试不该获得
「查不到就拼一个」的特权，否则表数据坏掉时最先失效的正是这批防回归测试。
"""

from __future__ import annotations

import json
from pathlib import Path

from listen2serve.runtime.tts_tags import allowed_tags_from_entry

# 规模断言锚定 version.json 的 assets.count，不写死条数（口径同 test_smoke 里
# _EXPECTED_VERSION 的做法）。原先各处硬编码 432/144/216，v6.5 扩样到 606/202/303 后
# 13 个用例集体失败，而它们逐行的实质断言全是通过的 —— 报的是"规模变了"而不是"数据坏了"。
# "数据被改却没 bump" 这条护栏仍在：assets 里的 sha256/md5 由
# scripts/bump_dataset_version.py 的读回校验守（改了数据不 bump，指纹就对不上）。
_VERSION_DOC = json.loads(
    (Path(__file__).resolve().parents[1] / "data/benchmark/version.json")
    .read_text(encoding="utf-8"))
DATASET_VERSION: str = _VERSION_DOC["dataset_version"]
N_SCENARIOS: int = _VERSION_DOC["assets"]["scenarios"]["count"]       # 三档展开后的总条数
N_BASE: int = _VERSION_DOC["assets"]["scenarios_base"]["count"]       # 基础场景条数

# 每态一句语气描述（含语速/音量词，满足 _instruct_from_control 的契约判据）
_TONE_BY_STATE = {
    "neutral": "语气淡淡地应答",
    "cooperative": "语气配合、有商量余地",
    "displeased": "语气不耐烦想脱身",
    "urgent": "语气着急催着办",
    "doubtful": "语气半信半疑",
    "hushed": "声音压得很低",
}

IDENTITY_PHRASE = "一位35岁男性欠款客户"


def fake_instruction_entry(
    key_state: str = "cooperative",
    *,
    states: list[str] | tuple[str, ...] | None = None,
    tags: dict[str, str] | None = None,
    key_tag: str | None = None,
) -> dict:
    """合成场景用的 instruction 条目（结构与真实表逐字段同构）。

    默认候选集为 6 态全开——合成场景里 FakeLLM 会自由声明 state，收窄候选集只会让
    测试去测「假 LLM 违约」而不是被测逻辑；需要验证候选集收窄行为的用例显式传 states。
    """
    cand = list(dict.fromkeys(list(states or tuple(_TONE_BY_STATE)) + [key_state]))
    entry: dict = {
        "role": "collection",
        "dynamics": "all_positive",
        "sub_domain": "测试",
        "identity_phrase": IDENTITY_PHRASE,
        "candidate_states": cand,
        "key_state": key_state,
        "tags": dict(tags or {}),
        "key_tag": key_tag,
        "non_key": {
            s: f"{IDENTITY_PHRASE}，{_TONE_BY_STATE.get(s, '语气平稳')}，语速中等，音量正常"
            for s in cand
        },
        "key": f"{IDENTITY_PHRASE}，摊牌时字咬得重、尾音往下压，语速中等，音量正常",
    }
    entry["allowed_tags"] = allowed_tags_from_entry(entry)
    return entry
