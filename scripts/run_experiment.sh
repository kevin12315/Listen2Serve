#!/usr/bin/env bash
# 一轮实验的端到端入口：生成 → 评测 → 报告（三步绑成一条命令，任一步失败立即终止）。
#
# 为什么要有这个脚本：三步分开手跑时最容易漏掉后两步，而漏掉的后果不报错——报告层继续显示
# 上一批次的数字，看起来"跑完了"。所以这里把生成→评测→报告绑成一条命令，并在结尾校验
# "本批次产物齐了 + 口径三要素（数据版本 / 被测 prompt 版本 / 裁判指纹）写进了 manifest"。
#
# 用法：
#   单端点： bash scripts/run_experiment.sh --run-id main_N_E --config configs/main_N_E.json \
#                --model dashscope/qwen3.5-omni-plus-realtime
#   整条件（论文一张表 = config.endpoints 里的三端点各跑一轮）：
#          bash scripts/run_experiment.sh --run-id main_N_E --config configs/main_N_E.json --model all
#   # --config 会带出该条件的文本层/韵律臂/裁判契约版本；显式传参优先于 config。
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONPATH="src:scripts"

RUN_ID="" MODEL="" CONFIG="" PROSODY="" LEAKAGE="" PROMPT="" JUDGE="" LIMIT="" EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --prosody-arm) PROSODY="$2"; shift 2;;
    --agent-prompt) PROMPT="$2"; shift 2;;
    --judge-model) JUDGE="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done
[[ -n "$RUN_ID" && -n "$MODEL" ]] || { echo "必须给 --run-id 与 --model"; exit 2; }

SCENARIOS="data/benchmark/scenarios.jsonl"
if [[ -n "$CONFIG" ]]; then
  [[ -f "$CONFIG" ]] || { echo "找不到 config：$CONFIG"; exit 2; }
  read -r PROSODY LEAKAGE SCHEMA IDS <<<"$($PY - "$CONFIG" <<'PYF'
import json,sys
c=json.load(open(sys.argv[1]))
print(c.get("prosody_arm",""), c.get("leakage",""), c.get("sim_prompt_schema",""),
      c.get("scenario_ids_file",""))
PYF
)"
  [[ -n "$SCHEMA" ]] && EXTRA+=(--sim-prompt-schema "$SCHEMA")
  # 发布仓的 scenarios.jsonl 只含论文主批次 145×2，按档位筛即选中该条件的 145 条；
  # 若你换了更大的人工挑选集，改用 --scenario-ids 逐条点名。
  [[ -n "$IDS" && -f "$IDS" ]] && echo "（config 指定的批次清单：$IDS，本脚本用 --leakage $LEAKAGE 等价筛出）"
fi
ARGS=(--run-id "$RUN_ID" --model "$MODEL" --scenarios "$SCENARIOS")
[[ -n "$LEAKAGE" ]] && ARGS+=(--leakage "$LEAKAGE")
[[ -n "$PROSODY" ]] && ARGS+=(--prosody-arm "$PROSODY")
[[ -n "$PROMPT" ]] && ARGS+=(--agent-prompt "$PROMPT")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
ARGS+=("${EXTRA[@]}")

# 一个端点的一轮：生成→评测→报告→三要素自检，任一步失败立即终止。
run_one() {
  local model="$1" rid="$2"
  local a=(--run-id "$rid" --model "$model" --scenarios "$SCENARIOS")
  [[ -n "$LEAKAGE" ]] && a+=(--leakage "$LEAKAGE")
  [[ -n "$PROSODY" ]] && a+=(--prosody-arm "$PROSODY")
  [[ -n "$PROMPT" ]] && a+=(--agent-prompt "$PROMPT")
  [[ -n "$LIMIT" ]] && a+=(--limit "$LIMIT")
  a+=("${EXTRA[@]}")
  local step; step() { echo; echo "=== [$rid] $1 ==="; shift; "$@"; }
  step 生成 "$PY" -m listen2serve.cli run-eval "${a[@]}"
  step 评测 "$PY" -m listen2serve.cli evaluate --run-id "$rid" ${JUDGE:+--judge-model "$JUDGE"}
  step 报告 "$PY" -m listen2serve.cli report --run-id "$rid"
  # 收尾自检：本批次的三要素（数据版本 / 被测 prompt 版本 / 裁判指纹）必须已落盘，
  # 否则一个月后没人能判断这批数字还能不能和别的批次放一起。
  step 自检 "$PY" - "$rid" <<'PYCHK'
import json, sys
from pathlib import Path
run_id = sys.argv[1]
mani = Path("runs") / run_id / "run_manifest.json"
assert mani.exists(), f"缺 {mani}：生成步没落盘"
m = json.loads(mani.read_text(encoding="utf-8"))
for key_path in (("data", "dataset_version"), ("prompts", "agent_prompt"), ("prompts", "judge_prompts_file_sha256")):
    node = m
    for k in key_path:
        node = node.get(k) if isinstance(node, dict) else None
    assert node, f"manifest 缺 {'.'.join(key_path)}：口径无法追溯"
vd = Path("reports") / run_id / "verdicts_audio.json"
assert vd.exists(), f"缺 {vd}：评测步没落盘"
rows = json.loads(vd.read_text(encoding="utf-8"))
n = len(rows.get("items", rows) if isinstance(rows, dict) else rows)
print(f"✓ {run_id}: dataset/prompt/judge 三要素齐，判定点 {n} 条")
PYCHK
  echo; echo "✅ $rid 完成（产物在 runs/$rid 与 reports/$rid）"
}

if [[ "$MODEL" == "all" ]]; then
  # --model all：按 config.endpoints 数组逐端点各跑一轮（论文一张表 = 三端点 × 145）。
  [[ -n "$CONFIG" ]] || { echo "--model all 需配 --config"; exit 2; }
  mapfile -t ENDPOINTS < <("$PY" - "$CONFIG" <<'PYE'
import json, sys
for e in json.load(open(sys.argv[1])).get("endpoints", []):
    print(e)
PYE
)
  [[ ${#ENDPOINTS[@]} -gt 0 ]] || { echo "config 里没有 endpoints 字段"; exit 2; }
  echo "按 config.endpoints 依次跑 ${#ENDPOINTS[@]} 个端点：${ENDPOINTS[*]}"
  for e in "${ENDPOINTS[@]}"; do run_one "$e" "${RUN_ID}__${e}"; done
else
  run_one "$MODEL" "$RUN_ID"
fi
