# E2v4 — Prospective Ollama structured-output amendment

Status: code complete; actual local-model runs NOT YET EXECUTED by the assistant.

Trigger: Qwen2.5 3B returned malformed Baseline A records for H02 (all eight `citations` keys absent) during a v3 run; this failed run remains unmodified.

Modification: use the Ollama **native** `/api/chat` API with JSON Schema `format` for generation and verification. Preserve exact original strict/weak prompt text and eight versioned source passages, including their E2 post-first ordering. Baseline A's schema requires claim IDs, text and nonempty citations to exact enumerated source IDs; a separate Python validator still enforces consecutive IDs and abstention logic. Verifier B's schema requires a categorical decision for each baseline claim, and `retain_claim_ids` is derived deterministically. Raw API responses, schema, seed, requested temperature, hashes, and validation outcomes are saved without overwrites.

Trade-offs: schema enforcement changes the inference protocol, can change model behavior and does not prove factual support. Original v3 H01 is not pooled with v4. Any v4 results must be analyzed separately. An empty result or invalid output is not counted as a zero error rate. Local structured-output support is confirmed by official Ollama documentation, but the exact model's empirical behavior must be tested on the user's machine.

Evidence status: 8 offline unit tests passed using archival contexts, one fake provider and a mocked Ollama API. No new model experiment was executed while preparing this patch; all prospective H01/H02 outputs remain unobserved.
