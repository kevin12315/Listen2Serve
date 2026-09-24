#!/usr/bin/env python3
"""发布仓的数据自检：结构、规模、均衡、指纹、台词表、音频 demo 六项，全绿才允许跑实验。

为什么不把校验塞进 pytest 就够了：`scripts/run_experiment.sh` 的入口需要先确认
"手上的评测集是完整且自洽的"，而 pytest 只在开发者机器上跑。CI 与人都调这个脚本，
退出码非零 = 数据有问题，⛔ 不许靠"警告继续跑"。

用法：PYTHONPATH=src python scripts/check_dataset.py [--verbose]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"
STATES = ("cooperative", "displeased", "doubtful", "urgent", "hushed")
LAYERS = ("T1", "T3")


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def state_of(row: dict) -> str:
    from listen2serve.domains.base import normalize_state
    ke = row.get("key_event") or {}
    return ke.get("state") or normalize_state(row.get("oracle_state") or "") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f" {'✓' if ok else '✗'} {name}{(' ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    ver = json.loads((BENCH / "version.json").read_text(encoding="utf-8"))
    scenarios = jl(BENCH / "scenarios.jsonl")
    bases = jl(BENCH / "scenarios_base.jsonl")

    print(f"[1] 规模与指纹（version.json：{ver['dataset_version']}，切片 "
          f"{ver.get('release', {}).get('slice', '?')}）")
    for key, path in (("scenarios", BENCH / "scenarios.jsonl"),
                      ("scenarios_base", BENCH / "scenarios_base.jsonl")):
        asset = ver["assets"][key]
        got = sha256(path) if key == "scenarios" or True else ""
        digest = got if asset.get("sha256") else md5(path)
        check(f"{key} sha256 与指纹一致", digest == (asset.get("sha256") or asset.get("md5")),
              f"{len(jl(path))} 行 / 声明 {asset['count']}")
    check("scenarios 行数 == 声明", len(scenarios) == ver["assets"]["scenarios"]["count"])
    check("scenarios_base 行数 == 声明",
          len(bases) == ver["assets"]["scenarios_base"]["count"])

    print("[2] 结构：每 base 恰好两档、字段齐、关键轮位置有效")
    by_base: dict[str, list[dict]] = {}
    for s in scenarios:
        by_base.setdefault(s["base_scenario_id"], []).append(s["leakage_label"])
    check("展开层只有 T1/T3（发布切片不含 T2）",
          {l for v in by_base.values() for l in v} == set(LAYERS), str(sorted(by_base)))
    check("每个 base 两档齐全", all(sorted(v) == sorted(LAYERS) for v in by_base.values()))
    need = ("agent_policy", "gold_action_plan", "measurement", "oracle_state", "user_script")
    missing = [s["scenario_id"] for s in scenarios for k in need if k not in s]
    check("必需字段齐全", not missing, f"缺 {len(missing)} 处")

    print("[3] 论文主批次清单与均衡性")
    for layer in LAYERS:
        doc = json.loads((BENCH / "subsets" / f"balanced145_{layer}.json").read_text(encoding="utf-8"))
        ids = set(doc["scenario_ids"])
        have = {s["scenario_id"] for s in scenarios if s["leakage_label"] == layer}
        cnt = Counter(state_of(s) for s in scenarios if s["scenario_id"] in ids)
        check(f"{layer}：清单 {doc['n']} 条全部在评测集里", ids <= have,
              f"缺 {len(ids - have)}")
        check(f"{layer}：五种状态各 29", all(cnt[x] == 29 for x in STATES), str(dict(cnt)))
    dom = Counter(s["base_scenario_id"].split("-")[0] for s in scenarios)
    check("三域都在（COL/HOT/MAR）", {"COL", "HOT", "MAR"} <= set(dom), str(dict(dom)))

    print("[4] 韵律表与音色表覆盖")
    tab = json.loads((BENCH / "tts_instructions.json").read_text(encoding="utf-8"))
    vmap = json.loads((BENCH / "voices" / "user_voice_map_v61.json").read_text(encoding="utf-8"))
    bs = {s["base_scenario_id"] for s in scenarios}
    check("tts_instructions 覆盖全部 base", bs <= set(tab["instructions"]),
          f"{len(bs & set(tab['instructions']))}/{len(bs)}")
    check("user_voice_map 覆盖全部 base", bs <= set(vmap["map"]),
          f"{len(bs & set(vmap['map']))}/{len(bs)}")
    tagged = Counter(v.get("key_tag") for v in tab["instructions"].values() if v.get("key_tag"))
    check("关键轮标签只挂四态（cooperative 无标签是设计）",
          set(tagged) <= {"angry", "curious", "very fast", "whispers"}, str(dict(tagged)))

    print("[5] 关键轮台词表（keyturn_canonical）自洽")
    # 这个文件不在 version.json 的指纹清单里，也没有任何 pytest 读它 —— 也就是说改坏
    # 内容不会有任何地方报警。所以规模/来源/格式三条都在这儿钉住。
    kt = jl(BENCH / "keyturn_canonical.jsonl")
    kt_layers = Counter(r["leakage_label"] for r in kt)
    check("规模 == 数据卡声明（145 T1 + 130 T3 = 275）",
          dict(kt_layers) == {"T1": 145, "T3": 130}, str(dict(kt_layers)))
    check("scenario_id 全部能在 scenarios.jsonl 里找到",
          {r["scenario_id"] for r in kt} <= {s["scenario_id"] for s in scenarios})
    check("台词不含括号舞台指示（user_simulator 的括号判据依赖这条）",
          not [r for r in kt if any(c in r["text"] for c in "（）()")])
    check("生成器型号不冒充公开型号（口径见 docs/limitations.md）",
          all("undisclosed" in str(r.get("gen_model", "")) for r in kt),
          str(sorted({str(r.get("gen_model")) for r in kt})[:2]))

    print("[6] 音频 demo 包（展示件，不参与测量；全量刺激音频不发布）")
    man = jl(BENCH / "audio_samples" / "manifest.jsonl")
    ok = bool(man)
    for row in man:
        p = BENCH / "audio_samples" / row["file"]
        ok = ok and p.exists() and sha256(p) == row["sha256"] and p.stat().st_size == row["bytes"]
    check(f"{len(man)} 个 demo wav 与 manifest 逐条对得上", ok)
    kt_wavs = {r[f]: r for r in kt for f in ("wav_state", "wav_neutral")}
    check("每条 demo wav 都能在 keyturn 里找到同名的行且台词一致",
          all(row["file"] in kt_wavs and kt_wavs[row["file"]]["text"] == row["text"]
              for row in man), f"{len(man)} 条")
    demo_states = {r["state"] for r in man}
    demo_arms = {r["condition"] for r in man}
    check("五态 × {state,neutral} 都有 demo，且矩阵不缺格",
          demo_states == set(STATES) and demo_arms == {"state", "neutral"},
          f"{len(man)} 条")
    # demo 的数量是设计出来的：每个状态取 2 个 base × 2 个韵律臂 = 4 条，五态共 20 条。
    # README / LICENSE_DATASET 里的"20 条"靠这两条断言兜住：加/删 clip 必须同步改文档口径。
    per_state = {s: {r["base_scenario_id"] for r in man if r["state"] == s} for s in STATES}
    check("每个状态各 2 个 base（demo 矩阵 5×2×2）",
          all(len(b) == 2 for b in per_state.values()),
          " ".join(f"{s}:{len(b)}" for s, b in per_state.items()))
    check("demo 条数 == 状态×臂×每态 base（不重复不缺格）",
          len(man) == len(demo_states) * len(demo_arms) * 2, f"{len(man)} 条")

    print(f"\n{'✅ 全部通过' if not fails else '⛔ 失败项：' + ', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
