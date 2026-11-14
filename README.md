# Listen2Serve

A multi-turn benchmark for real-time customer-service voice agents that listen and act.

Listen2Serve connects five customer states to the service actions required in the agent's immediate reply. This release provides the multi-turn evaluation framework, all 145 base scenarios in two wording conditions, service and judging prompts, audio demonstrations, real-call listening examples, and an offline run example.

Companion to *Listen2Serve: Benchmarking Real-Time Customer Service Agents That Listen and Act* (ICASSP submission).

## Install

Use Python 3.12 or newer, and run commands from the repository root. The framework reads the separately licensed benchmark assets from this checkout.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
make check-data
make test
make smoke
```

These checks use local data and simulated API responses. They require no commercial API credentials. `make smoke` opens a temporary loopback HTTP server.

## Configure models

Fill the credentials in `.env` for the services you will use:

| Role | Default model | Credential |
|---|---|---|
| Customer simulator | `kimi-k3` | `MOONSHOT_API_KEY` |
| Customer TTS | `qwen-audio-3.0-tts-plus` | `DASHSCOPE_API_KEY` |
| Uniform response transcription | `qwen-audio-3.0-asr-flash` | `DASHSCOPE_API_KEY` |
| Action, voice, flow and task judges | `google/gemini-3.8-flash` | `GEMINI_API_KEY` |
| Qwen agent under test | model selected below | `DASHSCOPE_API_KEY` |
| Doubao agent under test | `doubao-seeduplex-3.0` | `VOLC_API_KEY` |

The simulator sends the model ID `kimi-k3` to Moonshot's official OpenAI-compatible API. An alternative compatible backend can be selected explicitly with `openai/<model>` and `OPENAI_BASE_URL` / `OPENAI_API_KEY`. Each run records the models actually configured. See [model configuration](docs/model_configuration.md) for endpoint details and the confirmed correspondence between the manuscript TTS name and its official request ID.

See the [real-service acceptance checklist](docs/real_run_checklist.md) for account permissions, all assigned voices, and the staged acceptance procedure.

## Run a multi-turn evaluation

Start with one scenario. Real model runs use your API account and are billed by the selected providers.

```bash
bash scripts/run_experiment.sh --run-id example_N_E \
  --config configs/main_N_E.json \
  --model qwen-audio-3.0-realtime-plus --limit 1
```

Remove `--limit 1` to run all 145 scenarios in the condition. Use `--model all` to run the three endpoints listed in the config; Doubao requires its own speech-service credential.

| Paper condition | Customer wording | Vocal delivery | Config |
|---|---|---|---|
| E/E | explicit | expressive | `configs/main_E_E.json` |
| N/E | neutral | expressive | `configs/main_N_E.json` |
| N/N | neutral | neutral | `configs/main_N_N.json` |

Public options and reports use **explicit / neutral wording** and these paper condition codes. Scenario IDs and source fields retain their original storage labels so that the scenario, voice assignment, and audio demonstration references remain stable. Treat IDs as opaque identifiers; the loader maps their wording metadata to the paper terminology.

The script runs generation, judging, and reporting in order. Results are written to `runs/<run_id>/` and `reports/<run_id>/`. Completed and explicitly partial trajectories, and saved verdicts, can be resumed only when their input identities match. Use a new run ID when changing a model, scenario, prompt, voice, or condition. For deliberate re-judging, use `l2s evaluate --fresh` and a separate `--report-id`.

The three stages can also be run separately; see `l2s --help` and the help for each subcommand. The condition configs enable uniform ASR for the key-turn action judge. ASR or judge failures are surfaced as evaluation errors. A server disconnect after the key reply is captured is labelled `partial`: its key-turn action and voice can be scored, while whole-call FlowRate and TaskScore are left unscored. Reports show completion counts and metric denominators. A response timeout stops the session and preserves its partial audio; it never carries the old reply into the next turn. Timeout records are retained on resume; use a new run ID for a new attempt.

## Outputs and metrics

Each trajectory includes the customer/agent turns, the uniquely identified key turn, audio references, timing, and termination status. Verdicts contain per-item judge results and input fingerprints; the report presents:

| Metric | Definition |
|---|---|
| **KeyTurnPass** | key-turn action score ≥ 4, with no prohibited action in that reply |
| **VoicePass** | mean role/state/naturalness score ≥ 3.5, with every score ≥ 3 |
| **FlowRate** | mean checklist coverage, with full / partial / absent = 1 / 0.5 / 0 |
| **TaskScore** | complete / partial / absent task attainment = 1 / 0.5 / 0 |

See [how a scored example is displayed](docs/scoring_display.md) for the readable report, per-item evidence, audio links and pending-score states.

[examples/offline](examples/offline) illustrates the complete pipeline using simulated responses. The 20 clips in `data/benchmark/audio_samples/` demonstrate customer vocal delivery. The [real conversation examples](examples/real/index.html) contain Mandarin audio, aligned text, and model provenance:

| Service | Condition | Scenario | Turns | Key turn |
|---|---|---|---|---|
| Collection | N/E | `COL-NEG-45-T3` | 8 | 6 |
| Marketing | E/E | `MAR-ALL-12-T1` | 10 | 6 |
| Hotline | N/N | `HOT-ALL-01-T3` | 6 | 5 |

Clone or download the repository, then open `examples/real/index.html` in a browser. GitHub's file view shows the HTML source and does not play the embedded examples. The local checkout also provides the linked JSON status records.

These saved real calls are unscored. The [2026-11-12 judging check](examples/real/evaluation/20261112/README.md) recorded two marketing timeouts followed by HTTP 503, with no successful judge response; its ten original ASR turns were reused. Hotline and collection judging and new ASR were not started. All four metrics remain `null` with zero scored denominators. Earlier failures are recorded separately; example selection uses input and recording fidelity before judging.

## Repository contents

- `src/listen2serve/`: multi-turn controller, simulator, endpoint adapters, judges and reports.
- `data/benchmark/`: 145 base scenarios, 290 wording variants, scenario lists, voice and prosody assets, and audio demonstrations.
- `prompts/`: readable snapshots exported from the runtime constants; tests check their consistency.
- `configs/`: the three multi-turn cue conditions.
- `tests/` and `scripts/`: offline checks and experiment entrypoint.

See [release scope and reproducibility](docs/limitations.md), [the limited order-reference correction](docs/dataset_corrections.md), [adding an endpoint](docs/adding_an_endpoint.md), and [contributing](CONTRIBUTING.md).

## License and citation

Framework code, prompts and tooling use Apache-2.0. Benchmark content uses CC BY-NC 4.0; audio demonstrations use CC BY-NC-ND 4.0. The path-specific license table is in [LICENSE_DATASET](LICENSE_DATASET), and attribution is in [NOTICE](NOTICE).

Citation entries are in [CITATION.bib](CITATION.bib). Record the code revision, dataset fingerprints, models, and judge/input fingerprints with reported results.
