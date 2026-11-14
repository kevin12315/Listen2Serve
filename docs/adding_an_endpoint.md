# Adding a realtime voice endpoint

An agent under test is a provider adapter registered with a model ID and supported voices. Reuse the current tick interface so that customer simulation, judging, and reporting remain shared.

## Adapter contract

Implement a provider in `src/listen2serve/runtime/realtime/` and normalize events to `QwenEvent`. Handle connection, session configuration, upstream audio, response audio/text, cancellation, and disconnection.

Declare whether the endpoint needs continuous audio input, its response-wait policy, and its turn-end behavior. Preserve the controller's distinction between connection failure, response timeout, and an established session closed by the server. Use the actual audio sample rates and frame sizes required by the provider.

## Registration

Register the provider in `runtime/realtime/registry.py`, add its platform and model to `model_registry.py`, and document its credential in `.env.example`. Verify voice IDs against the intended model. Add the endpoint to a condition config only when it can perform the same multi-turn protocol.

## Validation

Add offline event/adapter tests. Before a full batch, run a short credentialed conversation and check authentication errors, applied model/voice, received audio and text, complete turn boundaries, interruption handling, and session termination. Retain the trajectory and runtime manifest when reporting results.

Run identities and judge input hashes must change when the configuration or actual inputs change. Use a new run ID for a new model or protocol configuration.
