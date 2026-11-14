# Limited order-reference correction

`order-reference-correction-1` corrects approved order identifiers in
`agent_policy.knowledge_base[0]` and `agent_policy.business_flow[0].goal` for
16 base scenarios and their 32 expanded explicit/neutral records (48 records,
96 string replacements). The order table's identifier is used for both policy
references. Every other field, phrase, scenario ID, and record order is preserved.

The source `dataset_version` and `generated_at` retain their historical values.
The release revision and updated SHA-256 values in `data/benchmark/version.json`
identify the corrected files. Earlier run manifests retain the fingerprints of
the assets actually used; this correction does not relabel those runs.

| Base scenario | Previous policy order | Corrected policy order |
|---|---|---|
| HOT-ALL-49 | ORD0014 | ORD2307 |
| HOT-ALL-50 | ORD0015 | ORD0016 |
| HOT-ALL-52 | ORD0018 | ORD0023 |
| HOT-ALL-53 | ORD0018 | ORD7302 |
| HOT-ALL-54 | ORD0018 | ORD3057 |
| HOT-ALL-55 | ORD0018 | ORD5209 |
| HOT-POS-56 | ORD0025 | ORD7301 |
| HOT-POS-57 | ORD0026 | ORD5207 |
| HOT-POS-59 | ORD0028 | ORD8803 |
| HOT-ALL-60 | ORD0013 | ORD5286 |
| HOT-ALL-61 | ORD0016 | ORD7301 |
| HOT-ALL-62 | ORD0021 | ORD8801 |
| HOT-ALL-63 | ORD0013 | ORD7205 |
| HOT-POS-64 | ORD0025 | ORD8801 |
| HOT-POS-65 | ORD0026 | ORD8801 |
| HOT-POS-67 | ORD0028 | ORD3057 |
