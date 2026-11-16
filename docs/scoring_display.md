# Reading audio examples and scores

The [offline report](../examples/offline/reports/offline/report_audio.md) and
[per-scenario verdict](../examples/offline/reports/offline/verdicts_audio.json)
show the released output format. Their responses and scores are synthetic test
outputs; the WAVs contain test pulses. They illustrate the trajectory, evidence fields, metric aggregation and report links.

## Public example layout

A scored listening example is read in this order:

1. A short case description: role, customer state, condition, scenario ID and complete/partial status.
2. A full-call WAV and aligned transcript, with separate customer and agent WAVs at the marked critical turn.
3. A compact table containing KeyTurnPass, VoicePass, FlowRate and TaskScore, copied from the saved verdict. The saved evaluation status and per-metric denominators accompany the metric values.
4. Links to the generated Markdown report, the original per-item JSON verdict and its model/input provenance. Readers can inspect an action's cited words and turn, or hear the speech associated with a voice score.

The three conversation examples demonstrate full-call audio, aligned turns and critical exchanges. The offline report demonstrates the complete scoring format. Dataset-level results use the benchmark trial set and its own denominators.

## Four main metrics

| Metric | Per-scenario verdict field | Aggregate field | Valid denominator |
|---|---|---|---|
| KeyTurnPass | `key_turn_pass` | `key_turn_pass` | `key_turn_n`: scored critical replies |
| VoicePass | `voice_pass` | `voice_pass` | `voice_n`: scored critical reply audio |
| FlowRate | `flow_rate` | `flow_rate_mean` | `flow_n`: scored complete calls |
| TaskScore | `task_score` | `task_score_mean` | `task_n`: scored complete calls |

KeyTurnPass requires a key-turn score of at least 4 and no triggered prohibited
action. VoicePass requires a three-dimension mean of at least 3.5 and every
dimension at least 3. FlowRate averages the checklist hits (yes = 1, partial = 0.5,
no = 0); TaskScore normalizes the terminal rubric's 0/1/2 to 0/0.5/1.

Read `n_complete` and `n_partial` with these denominators. A disconnected call
with a complete critical reply can contribute key-turn and voice results; its
whole-call flow and task results remain `null`. A scored zero or `false` is a
result; `null` means no valid score was recorded. A failed judge request produces an
evaluation error, rather than a zero for the agent.

## Follow the evidence

In the offline trajectory, turn 2 has `is_key_turn=true`. Its
[user WAV](../examples/offline/runs/offline/HOT-ALL-01-T3/audio/turn2_user.wav)
and [agent WAV](../examples/offline/runs/offline/HOT-ALL-01-T3/audio/turn2_agent.wav)
demonstrate the links used for the critical exchange. For real speech, read the
turn's state and text while listening to its unchanged WAVs.

The verdict's `text_source` identifies ASR or endpoint text. Key-turn evidence is
in `key_turn_detail.key_behavior_evidence` and `forbidden_items`; flow evidence
includes an action, hit, cited turn and excerpt in `flow_items`. `voice_detail`
contains role matching, state fit and naturalness scores with the judge's
reason. `judge_meta` binds the judgment to its models, prompts, configuration,
trajectory, scenario and audio contents. These fields support inspection of the
judgment alongside the audio; a quotation match alone does not establish the
semantic correctness of a score.

## Conversation examples

The [listening page](../examples/real/index.html) presents three complete calls:

| Service | Condition | Scenario | Turns | Key turn |
|---|---|---|---|---|
| Collection | N/E | `COL-NEG-45-T3` | 8 | 6 |
| Marketing | E/E | `MAR-ALL-12-T1` | 10 | 6 |
| Hotline | N/N | `HOT-ALL-01-T3` | 6 | 5 |

Each example includes original full-call and per-turn audio, endpoint text,
state and condition labels, scenario context and generation provenance.
Marketing includes a separately labelled independent ASR transcript.

Generation, judging and reporting can run separately through `l2s run-eval`,
`l2s evaluate` and `l2s report`. For a listening demonstration, select on input
fidelity and recording quality before looking at agent scores, then retain the
selected call's eventual judgment. Benchmark summaries retain the predefined
trial set, completion counts and per-metric denominators.

## Model identity shown with scoring

The report separates the requested judge from the model identifier returned by the provider. A missing returned identifier remains `null`; it is not filled from configuration. Detailed verdict JSON records each judge stage's identities, including multiple snapshots if the service returns them. Saved audio and input identities bind each evaluation to its conversation.
