# Scoring service check · 2026-11-12

All three original examples are accepted for display. Marketing made three new judge requests: two timeouts without an HTTP response, then HTTP 503 UNAVAILABLE. No scoring response succeeded; four metrics remain null with zero denominators. Its ten original ASR turns were replayed each time, with zero new ASR requests. Hotline and collection scoring and their fourteen new ASR turns were not started.

[Summary](summary.json) · [Marketing](marketing.json) · [Hotline](hotline.json) · [Collection](collection.json). Marketing’s earlier timeout and 503 are separately retained in its history. The model metadata GET is a metadata check, not a scoring response.
