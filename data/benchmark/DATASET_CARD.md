# Dataset card — Listen2Serve scenario set (main-experiment slice)

| | |
|---|---|
| Version | `v6.5` (content), release slice `main-multiturn-145x2` |
| Licence | see [LICENSE_DATASET](../../LICENSE_DATASET) (research use only, non-commercial) |
| Files | `scenarios.jsonl` 290 · `scenarios_base.jsonl` 145 · `subsets/` 2 · `tts_instructions.json` 145 · `keyturn_canonical.jsonl` 275 · `t3_action_ban.json` · `voices/` 2 · `audio_samples/` 20 wav + `manifest.jsonl` |
| Verify | `python scripts/check_dataset.py` (fingerprints, balance, coverage, audio hashes) |

## What one scenario fixes, and what does not

Fixed offline (identical for every tested model): role and business sub-domain, customer identity and
facts with their disclosure order, the mission beats with the target state per beat, the single key
event and its `oracle_state`, the action the policy requires at the key turn, the forbidden-action
list, the turn budget, and the prosody recipe (per-turn `instruction` text plus the key-turn TTS tag).

Deliberately *not* fixed: the exact wording of each customer turn. A constrained LLM simulator
improvises it per turn (10–60 characters, state drawn from this scenario's candidate set, a lexical
defence line that blocks state words in the neutral layer). The key turn is latched by the state
machine at the first turn satisfying the trigger condition, not at a fixed turn index.

So wording varies between runs of the same scenario; what is held constant across models is the
skeleton, the prosody table and the seed chain (`crc32(scenario_id)`).

## Text layers in this release

| Layer | Paper condition | Text channel |
|---|---|---|
| T1 | E/E | state stated in words (upper-bound control) |
| T3 | N/E and N/N | wording scrubbed to neutral business content, verified by a lexical screener |

The internal set also carries a T2 layer (implicit but consistent hints) used by experiments that are
not part of this release.

## The five customer states

| State | Required opening action | TTS control | Coverage |
|---|---|---|---|
| displeased | apologise first | `[angry]` | 29 bases |
| doubtful | address the concern first | `[curious]` | 29 bases |
| urgent | acknowledge the time pressure first | `[very fast]` | 29 bases |
| hushed | check whether it is a convenient moment | `[whispers]` | 29 bases |
| cooperative | confirm and proceed | natural-language instruction only, no tag | 29 bases |

`cooperative` has no tag on purpose: its acoustic description *is* the neutral baseline, so the two
rendered versions of a cooperative stimulus are near-indistinguishable. Any cross-state prosody
summary must therefore keep it out (see [docs/limitations.md](../../docs/limitations.md)).

## Provenance, and how the audio is rebuilt

Scenario skeletons were extracted from anonymised production call logs of three service lines (debt
collection, hotline support, marketing outbound): role, stage sequence and policy structure only — no
transcript text and no customer data carry over. Business facts (`db_seed`) were written from
scratch. All speech is synthesised through a commercial TTS API; no real person is recorded, and the
personal names, amounts and account numbers in the scenarios are fictional.

Stimulus audio is not redistributed here. `keyturn_canonical.jsonl` (target text) +
`tts_instructions.json` (instruction and tag) + `voices/user_voice_map_v61.json` (timbre) fully
specify it, and the TTS path is byte-deterministic for a given (text, voice, instruction, tag)
quadruple — so the same clips can be regenerated and checked against
`audio_samples/manifest.jsonl` (sha256 and byte size per clip).

## Known limitations of this slice

* The 145 bases are stratified by **state** (29 each), not by gender: within a role × dynamics cell
  the male/female split can be uneven (e.g. hotline/neg_to_pos is 2/8); the global split is 148/142.
* Manipulation strength differs sharply by state (`urgent` +27.9% speaking rate / +28.5% RMS,
  `hushed` −49.5% RMS, `cooperative` +0.1% RMS), so absolute pass rates are partly a property of
  the stimulus, not only of the model.
* The neutral text layer is not information-free: a word-only reader still recovers the state ~18%
  of the time (against 20% chance over five states) and abstains in 67–70% of cases.
* Everything is Chinese-language customer-service speech; prompts, rubrics and forbidden actions are
  Chinese and encode China-specific service norms.
