# Listen2Serve multi-turn scenario set

The released scenario set contains **145 base scenarios** and **290 expanded scenarios**, covering debt collection, hotline support, and marketing. Each of five customer states has 29 base scenarios.

| Asset | Contents |
|---|---|
| `scenarios_base.jsonl` | 145 shared scenario skeletons |
| `scenarios.jsonl` | 145 explicit-wording and 145 neutral-wording scenarios |
| `subsets/explicit.json`, `subsets/neutral.json` | Exact scenario IDs for each wording condition |
| `tts_instructions.json` | Per-state and key-turn vocal-delivery instructions for all bases |
| `voices/user_voice_map.json`, `voices/voice_pool.json` | Voice assignments and their source pool |
| `neutral_wording_ban.json` | Reviewed action-cue screening terms |
| `keyturn_canonical.jsonl` | 275 fixed utterances retained as demonstration provenance |
| `audio_samples/` | 20 customer-turn audio demonstrations and their manifest |
| `version.json` | Dataset identity, counts and asset fingerprints |

## Scenario contract

Every scenario defines the role, customer identity and business facts, disclosure order, mission, key event, target state, required/prohibited service actions, and turn budget. The multi-turn simulator generates customer utterances during the call. The agent's preceding reply drives the next customer turn; the key turn is latched when the scenario event occurs.

The public wording labels are **explicit** and **neutral**. E/E uses explicit wording and expressive speech; N/E uses neutral wording and expressive speech; N/N uses neutral wording and neutral speech. Scenario IDs and original storage fields are stable identifiers. Their raw suffixes are interpreted by the loader and are not public condition names.

| Customer state | Required key-turn action | Vocal control |
|---|---|---|
| Displeased | Apologize | angry |
| Doubtful | Address the concern | curious |
| Urgent | Acknowledge time pressure | very fast |
| Hushed | Check convenience | whispers |
| Cooperative | Affirm | natural-language instruction |

## Provenance and licensing

The released provenance describes scenario skeletons extracted from anonymized service-call structures, with generated dialogue text and fictional business facts. Speech is synthesized with commercial TTS voices. The original asset-generation fields are preserved.

Benchmark content uses CC BY-NC 4.0. Audio demonstrations use CC BY-NC-ND 4.0. See [LICENSE_DATASET](../../LICENSE_DATASET) for the path-specific scopes.

The 20 audio files demonstrate two renderings of the same neutral-wording line: five states × two bases × expressive/neutral delivery. They are individual customer turns; the multi-turn runtime synthesizes its own utterances. Waveform hashes identify the shipped demonstrations, rather than future TTS outputs.

Run `make check-data` to verify counts, fingerprints, coverage, and the audio demonstration matrix. Scenarios, utterances, voice assignments, and prosody recipes are preserved during this framework release cleanup.
