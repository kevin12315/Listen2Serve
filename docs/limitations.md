# Limitations, and what this release does not include

Written so that a reader can judge the numbers without having to trust us.

## Not shipped in this repository (and why)

| Item | Status | Why | Consequence for you |
|---|---|---|---|
| Per-item verdicts, model outputs, run traces | not released | they are measurements of named commercial preview endpoints, and every number is bound to a judge snapshot that will keep moving | you cannot recompute the paper's tables; you can rerun the instrument on any endpoint you have access to |
| Full stimulus audio (550 canonical + 550 duration-matched clips, ~230 MB) | not released | redistribution rights for vendor voices are unclear; volume | the 20 clips in `data/benchmark/audio_samples/` are a **listening demo** and are not read by the measurement path. Stimuli are re-synthesised at run time against your own TTS key from text + tag + voice + instruction; those three tables pin the *composition* of every clip, not the waveform — vendor TTS gives no cross-time byte-identity guarantee, so compare transcript/duration after re-synthesis rather than sha256 |
| Human-annotation materials (blind listening pages, truth pages, protocol, κ script) | not released | they reference reviewer identities and an internal review flow; the agreement measurement is not in the paper yet | the human-validity claim in §3.3 of the paper has no supporting artefact here |
| Single-turn same-script contrast, frozen-replay layer, auditory 2AFC / 5AFC probes | not released | out of scope for the minimal release (main experiment only) | the "fixed-utterance" gain in the paper cannot be reproduced from this repo |
| Scenario generation code and the source call-log skeletons | not released | they touch internal data and internal tooling | you get the generated scenarios, not the generator; the T2 layer and 57 further bases stay internal |

## Model identifiers: what is disclosed and what is not

Every model string that survives into this release is meant to be read as *"a publicly callable
endpoint name"* — i.e. something you can actually point the pipeline at. Where the real identifier
could not be disclosed (unreleased snapshot, or an internal gateway alias), the field is labelled
`undisclosed snapshot` rather than filled in with a nearby public name. Substituting a plausible
public model would be a false provenance claim, and it is checkable: `keyturn_canonical.jsonl`
carries `gen_prompt_hash`, `gen_seed` and `gen_time` per row, so anyone could try to reproduce the
generation and find the named model does not produce those lines.

| Field / site | Value in the release | Truthfulness |
|---|---|---|
| `keyturn_canonical.jsonl` → `gen_model` | `qwen-max (undisclosed snapshot)` | tier-level label, **not** a reproducible snapshot name; an earlier cut of this file said `qwen-plus`, which was wrong and is corrected here |
| `keyturn_canonical.jsonl` → `ctx_agent_model`, `gen_prompt_hash`, `gen_seed`, `gen_time` | as shipped | recorded at generation time, unmodified |
| Judge / simulator / TTS / ASR defaults (`.env.example`, `model_registry.py`, `configs/`) | public endpoint names | these are what the pipeline actually calls; they are yours to change |
| Two removed attributions (a bracket-behaviour note in `runtime/user_simulator.py`, an exempt-table note in `vocab.py`) | "undisclosed snapshot", no model named | the *observations* they cite are still checkable from the shipped data (e.g. 0 of 275 canonical lines contain a bracket); only the model name is withheld |

Consequence you should know about: **the canonical-line generation step is not byte-reproducible
from this repository**, because its generator snapshot is not disclosed. Everything downstream of
the shipped text is reproducible — the text itself, its fingerprints, and every metric given a
reachable endpoint.

## Measurement caveats that survive into the release

* **Judge dependence.** All four metrics come from an LLM judge. The default is the paper's own
  judge (`google/gemini-3.8-flash`, a publicly available model reached through Google's official
  OpenAI-compatible endpoint), so the published numbers are reproducible in that respect — but
  re-judging with any other model moves absolute values, and even re-judging with the same model
  can move 1–5 score bands (see the noise-floor bullet). Always report judge model +
  `judge_meta` fingerprint together with a number.
* **Judge self-consistency sets the noise floor.** Re-judging the same replies flipped 15.3% of the
  1–5 score band but 0.0% of the binary pass — which is exactly why `KeyTurnPass` (binary) is the
  primary metric and why mean-score gaps below ~0.2 points should not be claimed.
* **Prosody manipulation is unequal across states.** Speaking-rate/RMS deltas range from +27.9%
  (`urgent`) to −49.5% RMS (`hushed`) to +0.1% (`cooperative`). A high pass rate for a state with a
  weak stimulus is not evidence of skill.
* **`cooperative` is not a prosody condition.** Its "explicit" and "neutral" renderings are
  near-identical by construction; exclude it from any cross-state prosody statement.
* **The text-neutral layer is not information-free.** A word-only reader still recovers the target
  state ~18% of the time (chance 20% over five states) and abstains in 67–70%, so "text off" means
  "text weakened", not "text removed".
* **Multi-turn differences are whole-chain differences.** Comparing the text-explicit and
  text-neutral multi-turn batches contrasts two complete dialogue trajectories (history diverges),
  not one swapped line. The fixed-utterance contrast is the clean one — and it is not in this release.
* **Endpoint behaviour leaks into task numbers.** Some endpoints refuse scripted business asks by
  design (marketing-domain `TaskScore` drops while `FlowRate` stays high); the two must be read
  together. Endpoints with a short server-side session lifetime can also lose late turns; a batch is
  only comparable if it reached the key turn.
* **Absolute pass rates of the single-turn style settings are prompt-strength dependent** — service
  prompt length and rubric prompt must match on both sides, otherwise the level is depressed by a
  constant. That is why only paired differences are meaningful there.
* **Language and domain.** Everything is Chinese customer-service speech (collection / hotline /
  marketing). Rubrics, forbidden actions and politeness expectations do not transplant.

## Statistical conventions used by the paper

Wilson 95% intervals for proportions; exact McNemar for paired binary outcomes on the same
scenario or utterance; Benjamini–Hochberg over explicitly declared comparison families
(38 main + a separate 14-item family after the fourth endpoint entered the pool — that fourth
endpoint is not part of this release's `configs/`, which carry the three public endpoints only).
q-values move when the family changes even if p does not — quote the family size whenever you
quote a q.
