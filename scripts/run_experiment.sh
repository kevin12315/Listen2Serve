#!/usr/bin/env bash
# Run generation, scoring and reporting as one checked pipeline (Bash 3.2+).
# Explicit command-line options override the selected JSON configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONPATH="src:scripts${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="" MODEL="" CONFIG="" PROSODY="" WORDING="" IDS="" JUDGE="" VOICE_JUDGE=""
LIMIT="" SCENARIOS="" MODALITY="" ASR="" SIM_OVERRIDE="" EXTRA=() ENDPOINTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id|--model|--config|--prosody-arm|--wording|--scenario-ids|--judge-model|--voice-judge-model|--limit|--scenarios|--modality|--sim-model)
      [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --run-id) RUN_ID="$2";; --model) MODEL="$2";; --config) CONFIG="$2";;
        --prosody-arm) PROSODY="$2";; --wording) WORDING="$2";; --scenario-ids) IDS="$2";;
        --judge-model) JUDGE="$2";; --voice-judge-model) VOICE_JUDGE="$2";;
        --limit) LIMIT="$2";; --scenarios) SCENARIOS="$2";; --modality) MODALITY="$2";;
        --sim-model) SIM_OVERRIDE="$2";;
      esac
      shift 2;;
    --sim-model=*) SIM_OVERRIDE="${1#*=}"; shift;;
    --asr) ASR="true"; shift;;
    --no-asr) ASR="false"; shift;;
    *) EXTRA+=("$1"); shift;;
  esac
done
[[ -n "$RUN_ID" && -n "$MODEL" ]] || { echo "--run-id and --model are required" >&2; exit 2; }

if [[ -n "$CONFIG" ]]; then
  [[ -f "$CONFIG" ]] || { echo "Configuration not found: ${CONFIG}" >&2; exit 2; }
  CONFIG_VALUES=$("$PY" - "$CONFIG" <<'PYCONFIG'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in ("prosody_arm", "wording", "scenarios", "judge_model", "voice_judge_model", "modality"):
    value = config.get(key)
    if value is not None:
        if not isinstance(value, str) or "\n" in value or "\t" in value:
            raise ValueError(f"Invalid configuration value: {key}")
        print(key, value, sep="\t")
if "asr_for_judge_input" in config:
    value = config["asr_for_judge_input"]
    if type(value) is not bool:
        raise ValueError("asr_for_judge_input must be a boolean")
    print("asr", str(value).lower(), sep="\t")
if config.get("scenario_ids_file"):
    path = Path(config["scenario_ids_file"])
    subset = json.loads(path.read_text(encoding="utf-8"))
    ids = subset.get("scenario_ids") if isinstance(subset, dict) else subset
    if not isinstance(ids, list) or not ids or any(not isinstance(s, str) or not s for s in ids):
        raise ValueError(f"Invalid scenario IDs: {path}")
    print("ids", ",".join(ids), sep="\t")
endpoints = config.get("endpoints", [])
if not isinstance(endpoints, list):
    raise ValueError("endpoints must be a list")
for endpoint in endpoints:
    if not isinstance(endpoint, str) or not endpoint or "\n" in endpoint:
        raise ValueError("Invalid endpoint")
    print("endpoint", endpoint, sep="\t")
PYCONFIG
)
  while IFS=$'\t' read -r key value; do
    case "$key" in
      prosody_arm) [[ -n "$PROSODY" ]] || PROSODY="$value";;
      wording) [[ -n "$WORDING" ]] || WORDING="$value";;
      scenarios) [[ -n "$SCENARIOS" ]] || SCENARIOS="$value";;
      judge_model) [[ -n "$JUDGE" ]] || JUDGE="$value";;
      voice_judge_model) [[ -n "$VOICE_JUDGE" ]] || VOICE_JUDGE="$value";;
      modality) [[ -n "$MODALITY" ]] || MODALITY="$value";;
      asr) [[ -n "$ASR" ]] || ASR="$value";;
      ids) [[ -n "$IDS" ]] || IDS="$value";;
      endpoint) ENDPOINTS+=("$value");;
    esac
  done <<< "$CONFIG_VALUES"
fi
SCENARIOS=${SCENARIOS:-data/benchmark/scenarios.jsonl}
MODALITY=${MODALITY:-audio}
ASR=${ASR:-false}

# Validate every selected service, including all endpoints, before any generation.
PREFLIGHT=(--modality "$MODALITY")
[[ -z "$SIM_OVERRIDE" ]] || PREFLIGHT+=(--sim-model "$SIM_OVERRIDE")
[[ -z "$JUDGE" ]] || PREFLIGHT+=(--judge-model "$JUDGE")
[[ -z "$VOICE_JUDGE" ]] || PREFLIGHT+=(--voice-judge-model "$VOICE_JUDGE")
[[ "$ASR" != "true" ]] || PREFLIGHT+=(--asr)
if [[ "$MODEL" == "all" ]]; then
  [[ -n "$CONFIG" && ${#ENDPOINTS[@]} -gt 0 ]] || { echo "--model all requires config.endpoints" >&2; exit 2; }
  for endpoint in "${ENDPOINTS[@]}"; do PREFLIGHT+=(--target-model "$endpoint"); done
else
  PREFLIGHT+=(--target-model "$MODEL")
fi
"$PY" -m listen2serve.runtime.preflight "${PREFLIGHT[@]}"

run_one() {
  local model="$1" rid="$2"
  local a=(--run-id "$rid" --model "$model" --scenarios "$SCENARIOS")
  [[ -z "$WORDING" ]] || a+=(--wording "$WORDING")
  [[ -z "$PROSODY" ]] || a+=(--prosody-arm "$PROSODY")
  [[ -z "$IDS" ]] || a+=(--scenario-ids "$IDS")
  [[ -z "$LIMIT" ]] || a+=(--limit "$LIMIT")
  [[ -z "$SIM_OVERRIDE" ]] || a+=(--sim-model "$SIM_OVERRIDE")
  a+=(${EXTRA[@]+"${EXTRA[@]}"})
  echo "[${rid}] Generate"
  "$PY" -m listen2serve.cli run-eval "${a[@]}"
  local e=(--run-id "$rid" --scenarios "$SCENARIOS" --modality "$MODALITY")
  [[ -z "$JUDGE" ]] || e+=(--judge-model "$JUDGE")
  [[ -z "$VOICE_JUDGE" ]] || e+=(--voice-judge-model "$VOICE_JUDGE")
  [[ "$ASR" != "true" ]] || e+=(--asr)
  echo "[${rid}] Evaluate"
  "$PY" -m listen2serve.cli evaluate "${e[@]}"
  echo "[${rid}] Report"
  "$PY" -m listen2serve.cli report --run-id "$rid" --modality "$MODALITY"
  "$PY" - "$rid" "$MODALITY" <<'PYCHECK'
import json
import sys
from pathlib import Path

run_id, modality = sys.argv[1:]
manifest = json.loads((Path("runs") / run_id / "run_manifest.json").read_text(encoding="utf-8"))
for path in (("data", "dataset_version"), ("prompts", "agent_prompt_sha256"),
             ("prompts", "sim_prompt_sha256"), ("prompts", "judge_prompts_file_sha256"),
             ("evaluation", "judge_model")):
    value = manifest
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    if not value:
        raise ValueError(f"Missing manifest field: {'.'.join(path)}")
verdicts = json.loads((Path("reports") / run_id / f"verdicts_{modality}.json").read_text(encoding="utf-8"))
items = verdicts.get("items", verdicts) if isinstance(verdicts, dict) else verdicts
if not items:
    raise ValueError("No scored results")
partial = sum(item.get("dialogue_complete") is False or item.get("termination_reason") == "endpoint_session_closed"
              for item in items)
print(f"Completed {run_id}: {len(items) - partial} complete, {partial} partial scored scenarios")
PYCHECK
}

if [[ "$MODEL" == "all" ]]; then
  [[ -n "$CONFIG" && ${#ENDPOINTS[@]} -gt 0 ]] || { echo "--model all requires config.endpoints" >&2; exit 2; }
  for endpoint in "${ENDPOINTS[@]}"; do
    run_one "$endpoint" "${RUN_ID}__${endpoint//\//_}"
  done
else
  run_one "$MODEL" "$RUN_ID"
fi
