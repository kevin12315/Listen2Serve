# Real-service acceptance checklist

Use this checklist after the offline checks pass. Configure credentials locally in the repository `.env`; the file is ignored by Git. Keep the paper model IDs and shipped scenario/voice assignments for acceptance runs.

## Credentials and account permissions

| Local setting | Used for | Required account access |
|---|---|---|
| `MOONSHOT_API_KEY` | Customer simulator | Moonshot Chat Completions access to `kimi-k3` |
| `DASHSCOPE_API_KEY` | Customer TTS, uniform ASR, and Qwen agents | `qwen-audio-3.0-tts-plus`, `qwen-audio-3.0-asr-flash`, and the Qwen realtime model selected for the run |
| `GEMINI_API_KEY` | Key-turn action, voice, flow and task judges | `gemini-3.8-flash`, including WAV audio input through the Google OpenAI-compatible endpoint |
| `VOLC_API_KEY` | Doubao agent only | The speech service's full-duplex realtime dialogue API for `doubao-seeduplex-3.0` (wire model `1.2.6.1`) |

The two Qwen target IDs are `qwen-audio-3.0-realtime-plus` and `qwen3.5-omni-plus-realtime`. Running all three targets requires all four credentials. A Qwen-only run requires the first three. `VOLC_API_KEY2` is an optional second speech-service key; no extra OpenAI key is required by the paper defaults.

The current default DashScope HTTP and WSS endpoints are for China (Beijing). Use a key and model permissions for that region. If using another region or a workspace-specific domain, verify the TTS, ASR and realtime endpoints together before starting; changing only the key or TTS URL does not change the other endpoints.

Check that the account can synthesize every assigned customer voice. The 145 base scenarios use these nine voice IDs:

```text
qwen-audio-3.0-tts-plus-longcanzhuyue
qwen-audio-3.0-tts-plus-longfengmeihui
qwen-audio-3.0-tts-plus-longfengyueyao
qwen-audio-3.0-tts-plus-longjuhefeng
qwen-audio-3.0-tts-plus-longlanxuanxian
qwen-audio-3.0-tts-plus-longlinghuiyan
qwen-audio-3.0-tts-plus-longmeixiaolan
qwen-audio-3.0-tts-plus-longyanyuantao
qwen-audio-3.0-tts-plus-longyuzhihe
```

Qwen agent voice assignments are `longanlufeng` / `longanqian` for `qwen-audio-3.0-realtime-plus` and `Ethan` / `Tina` for `qwen3.5-omni-plus-realtime`. Doubao uses `zh_female_xiaohe_jupiter_bigtts`. Confirm service access without replacing the assigned voices for the acceptance run.

References: [Moonshot Chat Completions](https://platform.kimi.com/docs/api/chat), [DashScope TTS and regional keys](https://help.aliyun.com/zh/model-studio/non-realtime-tts-user-guide), [DashScope ASR endpoints](https://help.aliyun.com/en/model-studio/fun-asr-flash-recorded-speech-recognition-http-api), and [Google compatibility and audio input](https://ai.google.dev/gemini-api/docs/openai). Account-specific permissions are verified by a real request.

## Local setup

Create `.env` from `.env.example` if it does not already exist, then fill the keys locally. Preserve an existing `.env`. The default model settings are already in the template. The API machine needs HTTPS and WSS access to the selected official services and access to audio download URLs returned by TTS.

The automatic credential check validates the presence of every required setting before a pipeline run. The following real request checks service access, model permissions, assigned voices and the network route.

Start with one selected scenario and one concurrent call, with scenario retries disabled. To produce a listening candidate without invoking ASR or Gemini, use the generation stage directly:

```bash
l2s run-eval --run-id candidate_NE_hotline_1 \
  --model qwen-audio-3.0-realtime-plus \
  --wording neutral --prosody-arm state \
  --scenario-ids HOT-ALL-63-T3 --concurrency 1 --max-retries 0
```

This stage calls the simulator, customer TTS and selected realtime agent. Each provider may bill its requests. A fresh audio evaluation of one complete call with the default models and `--asr` normally makes four Gemini requests: KeyTurn, Voice, combined Flow/Task, and the auxiliary wording-leakage probe. Uniform ASR makes one request per recorded agent turn. Cache reuse can reduce these counts, while parse or transport retries can add requests. Check candidate generation and input quality before requesting evaluation. The full `scripts/run_experiment.sh` entrypoint also runs judging; reserve it for a scoring run rather than candidate selection.

## Acceptance sequence

1. Run the local credential preflight and offline checks. Review the requested model IDs and service hosts; a successful live response supplies the provider-reported identity when that service exposes one.
2. Generate one complete candidate for each of the three service roles. Check customer facts and disclosure timing, neutral wording, the unique complete key turn, audio continuity and the reason for ending. Listen for natural speech and the assigned vocal condition; N/N keeps customer delivery neutral while retaining the original latent oracle. Select the first suitable candidate; permit at most one new attempt for each preselected scenario, and retain attempt IDs and selection reasons. Repeated defects call for a correction before more synthesis.
3. Review [the scoring display](scoring_display.md) before real judging. Score a retained call with `l2s evaluate --run-id <run_id> --asr --concurrency 1`, then produce its report with `l2s report --run-id <run_id>`. Judging uses the saved trajectory and audio; generating a new call is unnecessary. Keep successful verdicts even when agent scores are low. Retain provider diagnostics with the original audio when resuming an evaluation.
4. Inspect the readable report, per-item evidence, audio binding and successful judge identity. Verify resume reuse for completed verdicts. Retain provider usage records, elapsed time and errors alongside the original outputs.

Three curated calls demonstrate the framework's interaction and recording quality. Formal benchmark measurements use the predefined trial set and report completion counts and per-metric denominators. Wider endpoint/condition checks can be scheduled separately when required by the acceptance scope and account budget.

A partial trajectory preserves a completed key-turn reply after a server disconnect. Its action/voice results can be inspected separately; whole-call flow/task metrics require a completed dialogue. Check the report's completion counts and metric denominators.

## Publishable real-run examples

The existing 20 WAV files in `data/benchmark/audio_samples/` are paired customer-utterance TTS demonstrations. Their manifest records text, model, voice, condition and hashes. The files under `examples/offline/` demonstrate the pipeline format using simulated services and test audio.

The [conversation examples](../examples/real/index.html) provide three complete calls: collection N/E (8 turns), marketing E/E (10 turns) and hotline N/N (6 turns). Each includes original audio, aligned endpoint text, scenario context and generation provenance.

For a real multi-turn example, retain a small complete bundle:

- full conversation audio and an aligned transcript with the key turn marked;
- scenario ID, condition, target/simulator/TTS/judge model IDs and run date;
- completion/termination status, per-item verdicts and the four reported metrics;
- the original run manifest and references linking the audio, transcript and verdicts.

Organize conversation examples around the full-call recording, aligned turns,
critical exchange and original generation metadata. Evaluation output bundles
add the saved verdict and rendered report. Review exported files for credentials
and account-specific metadata before publication.
