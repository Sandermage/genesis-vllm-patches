# `docs/reference/` — long-form historical references

Stable technical references that don't get updated per sprint —
deferred-PR decision logs, cross-rig validation logs, and one-off
setup pages.

## Contents

| File | Purpose | Status |
| --- | --- | --- |
| [`BOTS_SETUP.md`](BOTS_SETUP.md) | One-time setup for GitHub bots (Dependabot, security advisories, etc.). | Evergreen. |
| [`DEFERRED_P50_DEPLOY.md`](DEFERRED_P50_DEPLOY.md) | Research log for the P50 response-cache middleware deployment that was deferred per operator decision 2026-04-27. | Historical reference; revisit trigger documented inside. |
| [`DEFERRED_P87_PR40924.md`](DEFERRED_P87_PR40924.md) | Architectural mismatch analysis for vllm#40924 (`merge_attn_states_kernel` SM-shared-mem). Tracks the "land for free" trigger when upstream merges. | Historical reference. |
| [`LONG_CONTEXT_VALIDATION_20260427.md`](LONG_CONTEXT_VALIDATION_20260427.md) | 256K-context validation run on v748 PROD (TQ k8v4 + MTP K=3 + P82 t=0.3). | Historical bench. |
| [`V758_P75_SUFFIX_DECODING_DEPLOY_VARIANT.md`](V758_P75_SUFFIX_DECODING_DEPLOY_VARIANT.md) | v758 — Suffix Decoding deploy variant. Tested 2026-04-27; inconclusive on non-repeating prompts. | Historical experiment log. |
| [`V759_320K_CONTEXT_EXPANSION_20260427.md`](V759_320K_CONTEXT_EXPANSION_20260427.md) | v759 — 320K context expansion validation (vs 256K v748 baseline). | Historical bench. |

These files are kept verbatim from the original validation runs;
their dates and version numbers reflect the state of the project
at the time of writing. The pin / patch-registry / model names
may not match current PROD — refer to the canonical operator-facing
docs in [`../`](../) for the live state.
