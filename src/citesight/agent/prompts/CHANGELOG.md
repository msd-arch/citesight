# Prompt changelog

Every prompt change must reference an eval result once the Phase 5 harness exists.

| Prompt | Version | Date | Change | Eval ref |
|---|---|---|---|---|
| vlm_answer | v1 | 2026-07-03 | Initial: page-grounded answer + per-claim page index & verbatim evidence, strict JSON | pre-harness (Phase 2) |
| vlm_answer | v2 | 2026-07-04 | Brevity limits (claim ≤12 words, evidence ≤15, ≤4 claims): v1 outputs blew a 192-token budget and truncated mid-JSON, collapsing the agent's grounding loop (trace 94be1ff6ffad4d4c) | pre-harness (Phase 3 live run) |
| router | v1 | 2026-07-03 | Initial: query type + entity filters + retrieval budget | pre-harness (Phase 3) |
| verifier | v1 | 2026-07-03 | Initial: per-claim verdicts + region hints + reformulated query | pre-harness (Phase 3) |
| composer | v1 | 2026-07-03 | Initial: final answer from supported claims only | pre-harness (Phase 3) |
| judge | v1 | 2026-07-04 | Initial eval judge: correctness (1/0.5/0) vs reference + per-claim citation support | first harness version (Phase 5 core) |
