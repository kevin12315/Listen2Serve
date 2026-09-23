# Adding a new realtime voice endpoint

The framework makes an "agent-under-test" just a provider adapter plus one binding-table
row — three steps in total. ⚠️ Do not skip step 3 (the liveness gate): this class of
endpoint fails in a way that looks like "it connected but behaves wrong", silently
poisoning a whole batch of numbers.

## 1. Write the adapter

Add `src/listen2serve/runtime/realtime/<vendor>_provider.py` implementing the same tick
interface as `qwen_provider.QwenRealtimeProvider`:

| Must handle | Why (each is a bug we already hit) |
|---|---|
| Upstream audio framing and the `needs_continuous_audio` flag | Full-duplex endpoints end the turn themselves if they don't get continuous input; if the flag is wrong the per-turn wait budget uses the wrong tier (30 s vs 60 s), and an endpoint with a heavy-tailed first response gets mis-read as "no response" and the whole call is dropped |
| Session lifetime | Some endpoints have a hard server-side session cap (observed 91–146 s before a forced disconnect); a disconnect must be attributable by the orchestrator as "endpoint limit", not "the model didn't answer" |
| Whether there is a text input channel | Endpoints without one (audio-only full duplex) cannot run the "text-only" control cells; register them in `FULL_DUPLEX_NO_TEXT_INPUT_PLATFORMS` or those cells get silently scored 0 |
| Event normalisation | Events must be normalised to `QwenEvent` so `QwenTickAdapter` can be reused |

## 2. Register it

* `runtime/realtime/registry.py`: add a backend branch in `create_provider()`.
* `model_registry.py`: add a `bare name → {platform, voices}` row in `MODEL_BINDINGS`.
  The voice **must be verified on a real call** — these endpoints acknowledge any
  model/voice with `session.created` and only surface the error on first inference, so
  roughly half of the voice names copied from docs are wrong.
* `_KNOWN_PLATFORMS`: add the new platform name, otherwise `newplatform/Model` is parsed
  as a bare name and the server replies `Model not found`, which looks like the model is
  missing rather than the platform being unregistered.
* `.env.example`: add the credential key (key name only, ⛔ never a value —
  `tests/test_release_hygiene.py` rejects values there).

If the new endpoint participates in a paper condition, also add it to the `endpoints`
list in the relevant `configs/*.json`, and include it in the statistical comparison
family *before* running the batch (q-values move with family size; adding it after the
fact flips other people's verdicts). `bash scripts/run_experiment.sh --model all` loops
over exactly that `endpoints` list.

## 3. Liveness gate (before any real batch)

Confirm each item with a minimal round-trip; if any fails, keep the endpoint out of the
comparison table:

1. **A1 auth**: a missing key and a wrong key must both be rejected (a "public demo
   gateway" that answers 200 to anything is a trap);
2. **A2 model exists**: a negative control (a deliberately bogus model name) must also
   error truthfully, otherwise the existence check proves nothing;
3. **A3 voice applies**: set the voice explicitly, receive a real audio frame, and check
   F0 / listening impression matches the expected gender;
4. **A4 first-response latency & truncation**: run 5 calls back to back; inspect the
   first-response distribution and whether the server ever "speaks for the user";
5. **A5 session lifetime**: run to the turn-budget ceiling and record whether the session
   is cut short unilaterally.

## 4. Run it

```bash
export VOLC_API_KEY=...           # or whatever credential slot this endpoint family uses
bash scripts/run_experiment.sh --run-id probe_newmodel --limit 2 \
    --model newplatform/Some-Realtime --prosody-arm state
```

`run_experiment.sh` ends with a self-check confirming that `run_manifest.json` carries all
three traceability fields — data version, agent-prompt version and judge fingerprint. If
any is missing, that batch can never be placed alongside other batches later.
