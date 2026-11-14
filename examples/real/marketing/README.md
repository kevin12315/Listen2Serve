# Marketing — E/E listening example

One selected complete live conversation: `MAR-ALL-12-T1`, 10 turns, key turn 6. The controller recorded `ok / user_closed`, with one non-forced critical exchange and no response timeout or endpoint interruption. The scenario's published wording and state delivery are retained.

This local staging candidate awaits the owner's final publication review. Human listening is recorded only within [the review's actual scope](review_status.json). Runtime VAD interruption markers and original waveform content remain visible in the supplied trajectory and unchanged recordings.

- [Full transcript](transcript.md): original endpoint text and real independent ASR are labelled separately.
- [Complete original audio](runs/showcase_20261110_marketing_E_E_MAR12_a1/MAR-ALL-12-T1/conversation.wav), [original stereo audio](runs/showcase_20261110_marketing_E_E_MAR12_a1/MAR-ALL-12-T1/both.wav), and all per-turn WAVs linked in the transcript.
- [Evaluation status](reports/showcase_20261110_marketing_E_E_MAR12_a1/evaluation_status.json), [ASR bindings](runs/showcase_20261110_marketing_E_E_MAR12_a1/MAR-ALL-12-T1/agent_asr.json), [provenance](MANIFEST.json), and SHA256SUMS.

Formal scoring is pending and paused by the owner. All four metrics are null and their evaluated denominators are zero. The retained marketing run has ten real audio-bound ASR results. Its first Gemini KeyTurn request timed out without an HTTP response; the one authorized recovery received HTTP 503. No judge stage succeeded. Failed-call token usage and charges are unknown.

The curated collection is intended to contain three role-condition examples. It is not a complete 3-by-3 role-mode matrix or an overall performance estimate. Selection used script fidelity and technical quality, independently of agent scores.

All audio and segments retain the original bytes, without trimming, normalization, splicing or replacement. Source hashes identify the private originals; public derived hashes identify sanitized metadata. Requested and reported model IDs are separate, and an absent reported model remains null. The public view omits operational request/session/account IDs, private paths, credentials, environment snapshots, raw prompts and signed download URLs. Unexpected private transcript content causes export refusal.

The sanitized run manifest includes a small runtime-settings whitelist and actual per-request LLM/TTS parameters, each bound to its private source-request SHA. Model, temperature, token limit, seed, disabled thinking, voice, sample rate, format and instruction are recorded without copying messages or arbitrary request bodies.

Text, transcription and provenance: CC BY-NC 4.0. All original WAV speech: CC BY-NC-ND 4.0. Framework code is separately Apache-2.0 and is not included. Final repository integration requires matching path-specific LICENSE_DATASET entries.
