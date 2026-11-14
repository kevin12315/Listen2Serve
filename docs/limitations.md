# Release scope and reproducibility

This release provides the multi-turn Listen2Serve instrument: the complete released scenario set, customer simulation, voice-agent interaction, four-metric judging, prompts, and example artifacts. Other experiments in the paper are maintained separately.

## Re-running the instrument

The shared assets fix the customer identity and business facts, disclosure order, mission and key event, required and prohibited actions, turn budget, voice assignment, and vocal-delivery instructions. The simulator generates the wording of each turn during the interaction. The target agent's response drives the next customer turn. For non-forced critical turns, the simulator checks actual agent speech for four specifically supported repayment-trigger hint templates; other free-form hints retain their existing handling and require source-grounded review. When the known debt exceeds the planned payment, the fact check uses source facts and mission beats to reject unsupported definite balance-clearing commitments, retain supplied remainder plans and compare a limited set of literal time anchors. When a supported non-forced critical turn states a debt amount reserved by the source for an agent check, the simulator requires an actual agent amount quotation or question and lets the customer correct a wrong quotation.

Seeds and local input fingerprints make the experiment configuration traceable. Hosted model responses and TTS waveforms can change across requests and provider updates. A new run therefore measures the selected endpoints at that time. Compare runs using their recorded models, condition, input identity, and judge identity.

The framework rejects incompatible cached trajectories and verdicts. Invalid judge output is an evaluation error; it is not scored as customer-service failure. Evidence checks verify quoted text against the supplied reply or dialogue window and bound cited turn numbers; semantic adequacy remains part of the judge rubric and manual inspection. The reports include completion counts and per-metric denominators. A server-disconnected call with a completed key reply is labelled `partial`; its key-turn action and voice can be evaluated, while whole-call flow and task outcomes remain unscored. Retain generation results and error records when comparing coverage.

A response timeout preserves the current audio and trace, cancels the response and closes that session before another customer turn can start. If a complete critical reply was captured earlier, the trajectory is partial; a timeout before or at the critical reply is an error. A `response_timeout` trajectory is terminal for its run, including resume, so it is not silently regenerated or overwritten. Use a new run ID for another attempt. Other configured error retries keep their existing behavior. Evaluation rejects historical timeouts before or at the critical reply before any ASR or judge requests; a timeout after a valid critical reply leaves whole-call metrics unscored.

Realtime server VAD may respond during a pause within synthesized customer speech. The full-call `conversation.wav` and stereo `both.wav` preserve the tick timeline, including overlap and cancelled replies; per-turn endpoint text retains the fragments actually played. Read the full-call audio with `ticks.json` when inspecting this behavior. VAD truncation markers follow the audio item that was played; an unplayed item with unknown ownership is not assigned to the previous turn.

## Assets and examples

The scenario files retain stable IDs and storage labels. Runtime options and reports use the paper's explicit / neutral wording terminology. The data card documents the asset counts and their provenance.

The 20 customer audio clips are listening demonstrations. Multi-turn calls synthesize customer turns at runtime. The fixed-utterance table accompanying those demonstrations is retained as source metadata; it is not consumed by the multi-turn simulator.

The offline example uses simulated endpoint replies and judge responses to demonstrate the runtime output structure. The [current listening examples](../examples/real/index.html) retain three accepted complete calls: collection N/E (8 turns), marketing E/E (10 turns) and hotline N/N (6 turns). Original audio and endpoint text are retained for all three; independent ASR is available for the ten-turn marketing call. The [2026-11-12 scoring check](../examples/real/evaluation/20261112/README.md) made three unsuccessful marketing requests (two timeouts, then HTTP 503), reusing those ASR results. Hotline and collection judging and new ASR were not started. All four metrics remain `null` with zero scored denominators. Earlier failures and original generation-source identities remain separate. Selection checks source fidelity and recording quality before judging, independently of agent scores.

## Interpretation

All scenarios and rubrics are Chinese customer-service interactions. The five states have different vocal-control strengths. The neutral wording condition reduces lexical state cues; the generated dialogue can still contain contextual information. Read key-turn action and voice results alongside whole-call flow and task outcomes.

Model defaults and endpoint references are described in [model_configuration.md](model_configuration.md). The manuscript's TTS name is documented alongside the authors' confirmed official request ID.
