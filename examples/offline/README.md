# Offline run example

This directory contains output from a local execution of the multi-turn pipeline using simulated endpoints. It illustrates the trajectory, verdict, and four-metric report formats. Its scores are test-fixture outputs, not measurements of the paper's commercial models.

The full integration test keeps the real scenario loader, simulator constraints, orchestration, judging and reporting, while replacing external model services with deterministic local responses. No API keys or remote calls are involved. Re-run it with `pytest tests/test_release_entrypoint.py`.

The original scenario ID identifies the benchmark scenario used by the example. Referenced scenario content remains subject to the benchmark's license. Audio here, if present, is synthetic test data; the TTS listening demonstrations are in `data/benchmark/audio_samples/`.

The publication metadata retains the original source fingerprints and records that the source tree was dirty. Its development file list is omitted from the public example.
