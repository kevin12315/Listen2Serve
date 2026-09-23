# Listen2Serve

**A controlled benchmark of prosody-conditioned service actions for real-time customer-service voice agents.**

Does a voice agent change *what it does* because of *how the customer said it*?
Listen2Serve pairs each customer state (dissatisfied / doubtful / urgent / hushed / cooperative) with
the service action that the shared policy requires at the moment that state is expressed, and then
turns the two cue channels — wording and voice — on and off independently.

Companion to the paper *"Listen2Serve: Benchmarking Real-Time Customer Service Agents That Listen and Act"*
(ICASSP 2027 submission; arXiv link TBD).

## What this repository contains

| Component | Path | Notes |
|---|---|---|
| Evaluation framework | `src/listen2serve/` | orchestrator, constrained user simulator, TTS/ASR/judge gateways, four metrics |
| Scenario set (main experiment) | `data/benchmark/scenarios.jsonl` | 145 balanced bases × {T1 text-explicit, T3 text-neutral} = 290 scenarios |
| Paper batch definitions | `data/benchmark/subsets/` | the exact 145 `scenario_id` lists behind every number in the paper |
| Prosody recipe | `data/benchmark/tts_instructions.json` | per-turn `instruction` + key-turn TTS tag, keyed by base × state |
| Key-turn target scripts | `data/benchmark/keyturn_canonical.jsonl` | 275 reviewed utterances (145 T1 + 130 T3) |
| Forbidden-action gate | `data/benchmark/t3_action_ban.json` | single source consumed by generator, screener and judge |
| Voice assignment | `data/benchmark/voices/` | per-base user timbre, gender × age band |
| Service / simulation / rubric prompts | `prompts/` | text snapshots; source of truth is the code, drift fails a test |
| Reproduction configs | `configs/` | one file per paper condition (E/E, N/E, N/N) |
| Audio samples only | `data/benchmark/audio_samples/` | 20 clips (5 states × 2 conditions × 2 bases) + `manifest.jsonl` |
| Contract tests | `tests/` | 350 checks: data schema, judge contracts, orchestrator, release hygiene |

**Not included, deliberately** (see [docs/limitations.md](docs/limitations.md)):
per-item verdicts and model outputs, the full stimulus audio, the single-turn / replay /
auditory-probe experiments, and the human-annotation materials. Numbers in the paper were produced
by this pipeline on commercial endpoints; this release ships the instrument, not the recordings.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env          # add DASHSCOPE_API_KEY (tested endpoints + TTS + ASR)
make check-data               # structure / balance / fingerprints / audio samples
make test                     # contract tests, no network access required
make smoke                    # offline end-to-end check of the swappable judge backend

# one full paper condition: run → judge → report, once per endpoint in the config
# (3 endpoints × 145 scenarios); `--model all` loops over configs/*.json `endpoints`.
bash scripts/run_experiment.sh --run-id main_N_E --config configs/main_N_E.json --model all
# or a single endpoint:
bash scripts/run_experiment.sh --run-id main_N_E --config configs/main_N_E.json \
    --model dashscope/qwen3.5-omni-plus-realtime
```

## The measurement

One call per scenario, then four numbers:

| Metric | Question | How it is decided |
|---|---|---|
| **KeyTurnPass** (primary) | did the reply perform the action the state requires, in the key turn | rubric score ≥ 4 **and** no forbidden action in that turn |
| **VoicePass** | was the agent's own voice appropriate (role / state fit / naturalness) | audio judge, 1–5 per axis, mean ≥ 3.5 and every axis ≥ 3 |
| **FlowRate** | were the required flow items covered across the call | 1 / 0.5 / 0 per item, implicit execution counts |
| **TaskScore** | was the business goal reached by the end | 1 / 0.5 / 0 |

Conditions vary two knobs independently: how explicit the **text** is (E vs N) and how explicit the
**voice** is (E vs N). `configs/` pins the three combinations the paper reports.

Reproducibility is by construction: scenario seeds come from `crc32(scenario_id)`, the user
simulator draws utterances from a fixed seed chain, prosody comes from a table (never improvised at
run time), and every run records a `judge_meta` fingerprint — change a rubric and old verdicts are
flagged as stale instead of being silently reused.

## The judge, and swapping it

All three judgements (key-turn action, output voice, flow/task) are made by
`gemini-3.8-flash` through **Google's official OpenAI-compatible endpoint** — the same model the
paper used, now a publicly available one:

```bash
GEMINI_API_KEY=...                 # the default: JUDGE_MODEL / VOICE_JUDGE_MODEL = google/gemini-3.8-flash
```

Any other OpenAI-compatible backend is a two-line change (this is how you ablate judge choice):

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=...
JUDGE_MODEL=openai/gpt-4.1-mini  VOICE_JUDGE_MODEL=openai/gpt-4o-audio-preview
```

Judge choice changes absolute scores, so always report judge model + `judge_meta` fingerprint
alongside a number; the user simulator (`LLM_MODEL`, default `qwen-plus` on Model Studio/Bailian,
which is publicly callable) is deliberately a *different* model from the judge, so generation
style and judging preference are not co-sourced. `make smoke` exercises the whole path
(gateway → rubric → verdict → KeyTurnPass) against a local stub, with no external call.

Nothing in this repository calls an internal or non-public endpoint: that claim is enforced by
`tests/test_release_hygiene.py`, not by a promise.

## Adding an endpoint, a domain, or a state

* new realtime endpoint → [docs/adding_an_endpoint.md](docs/adding_an_endpoint.md) (provider adapter
  + registry entry + liveness gate)
* new service domain / policy section → `src/listen2serve/domains/`, then the scenario generator
  contract in `tests/test_policy_rules.py`
* new customer state → extend `runtime/tts_tags.STATE_TAG_ALLOWED`, the action table in
  `domains/base.py` **and** the rubric, all three at once (`tests/test_judge_contract.py` checks it)

## Repository layout

```
src/listen2serve/{runtime,evaluation,gateway,domains,audio,data,report}/
data/benchmark/{scenarios.jsonl,scenarios_base.jsonl,subsets/,voices/,audio_samples/,version.json}
prompts/         configs/      scripts/     tests/     docs/
```

## License and dataset terms

* Code: **Apache-2.0** ([LICENSE](LICENSE)).
* Benchmark content under `data/` (**CC BY-NC 4.0 semantics**, see [LICENSE_DATASET](LICENSE_DATASET)):
  scenario skeletons are distilled from anonymised production call logs (roles, stages and policy
  rules only — no transcripts), utterances are LLM-generated and human-reviewed, and all audio is
  synthesised. No real customer or agent voice appears anywhere.

## Citation

See [CITATION.bib](CITATION.bib). If you extend or re-mix the scenario set, please keep the
`dataset_version` / asset fingerprints in `data/benchmark/version.json` — they are how we and
everyone else tell incompatible batches apart.
