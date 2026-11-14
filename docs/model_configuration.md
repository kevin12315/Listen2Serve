# Model configuration

Defaults follow the supplied manuscript's customer simulator (`kimi-k3`), multimodal judge (`gemini-3.8-flash`), and the realtime endpoint adapters listed below. The simulator uses Moonshot's official API directly; its identity is recorded as `moonshot/kimi-k3`.

| Service | API / wire model |
|---|---|
| Customer simulation | `https://api.moonshot.cn/v1/chat/completions`, model `kimi-k3` |
| Judging | Google's OpenAI-compatible API, model `gemini-3.8-flash` |
| Qwen agents | DashScope Realtime, models `qwen-audio-3.0-realtime-plus` and `qwen3.5-omni-plus-realtime` |
| Doubao agent | speech-service adapter for `doubao-seeduplex-3.0`; protocol model `1.2.6.1` |
| Customer TTS | existing DashScope speech-synthesis adapter, model `qwen-audio-3.0-tts-plus` |
| Uniform ASR | DashScope ASR, model `qwen-audio-3.0-asr-flash` |

Moonshot's [Chat Completions reference](https://platform.kimi.com/docs/api/chat) documents the `kimi-k3` request model. Credentials use `MOONSHOT_API_KEY`; `MOONSHOT_BASE_URL` is configurable for an authorized compatible deployment. The simulator disables thinking by default. The gateway adapts Kimi temperature to the selected thinking mode and records the requested and effective values in its call audit. The generic `openai/<model>` route remains available with separate credentials and base URL. Explicit model/platform conflicts are rejected.

The authors confirmed on 2026-10-17 that the manuscript name `qwen3-audio-plus` refers to the same TTS model as `qwen-audio-3.0-tts-plus`. The release uses the latter official request ID, listed in [Alibaba Cloud's model documentation](https://help.aliyun.com/zh/model-studio/qwen-audio-3-0-tts-plus), and records that actual ID in each run.

Historical generation fields in the shipped demonstration metadata describe the generation of those assets. They are independent of the multi-turn simulator default and are preserved verbatim.

Endpoint documentation was checked on 2026-10-17. Offline validation uses simulated APIs. The [real listening examples](../examples/real/index.html) contain three accepted complete calls made with `qwen-audio-3.0-realtime-plus`, `kimi-k3` and `qwen-audio-3.0-tts-plus`; each retains its original generation-source identity. Marketing has ten original ASR turns. The [2026-11-12 scoring check](../examples/real/evaluation/20261112/README.md) received no successful judge response: two request timeouts, then HTTP 503. Hotline and collection judging and new ASR were not started. A successful model-metadata lookup is recorded separately and is not a judge response. Service availability remains account-specific.

## Requested and reported identity

Gateway audit records keep `requested_model` separately from nullable `response_model`. The latter contains only a model identifier returned by the service; a binary audio response may not expose one. `requested_host` records the configured destination; `provider_host` is populated from HTTP response metadata when exposed. Both contain hostnames without credentials or URL parameters. Returned aliases or version snapshots are retained and a difference is logged for review. A successful response demonstrates the service request and reported API identity.

The realtime trajectory saves `provider_identity`; the run manifest summarizes target identities and gateway calls under `model_identity.generation`. Doubao records its public model name and the separate wire model. Each scored verdict records judge identities by stage. The report labels requested and reported judge models separately; detailed JSON retains multiple returned versions when present. Older artifacts without response identity remain readable and keep their original evidence.
