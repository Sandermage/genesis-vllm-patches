# `upstream_refs/` — frozen vLLM upstream snapshots

This directory holds **read-only snapshots** of upstream vLLM source
code used as text-patch anchor references. Files here are not
compiled or imported at runtime; they exist so contributors can
read the exact upstream form a Genesis text-patch is anchored
against without having to check out a specific upstream SHA.

## Contents

### `dev134_*.py` — Wave 8/9 upstream snapshots (commit `dev134`)

| File | Upstream path | Purpose |
| --- | --- | --- |
| `dev134_chunk_scaled_dot_kkt.py` | `vllm/model_executor/layers/fla/ops/...` | GDN chunk-scaled-dot-kkt kernel reference for P39a anchors. |
| `dev134_gdn_linear_attn.py` | `vllm/model_executor/layers/fla/layers/gated_deltanet.py` | GDN linear attention layer — anchor reference for the P67 / PN11 / PN54 patch family. |
| `dev134_linear_attn.py` | `vllm/model_executor/layers/mamba/ops/...` | Mamba2 linear attention helpers. |
| `dev134_triton_turboquant_decode.py` | `vllm/attention/ops/triton_turboquant_decode.py` | TurboQuant decode kernel — P67 / PN116 / PN118 / PN119 anchors. |
| `dev134_turboquant_attn.py` | `vllm/attention/backends/turboquant_attn.py` | TurboQuant attention backend. |

### `gemma4_tracking_2026-05-14.md`

Tracking note for Gemma 4 26B-A4B model class — the next non-Qwen
target for Genesis PROD after the Qwen3.6 stack stabilises. Lists
upstream PRs that we deliberately don't text-port today and the
pin-bump strategy for absorbing them.

### `pr_40792/` — k8v4 GQA head grouping kernel

Working directory for the vllm#40792 backport (PN119, TurboQuant k8v4
GQA head grouping). Holds the upstream test + kernel source at the
moment we forked. `pr_40792_k8v4_gqa_grouping.diff` at the top level
is the diff that derived our backport.

### `pr_40798/` — TurboQuant workspace manager refactor

Working directory for the workspace-manager refactor referenced by
PN118 (vllm#42551). Holds upstream attention / model-runner / TQ
sources + the kernel + the test file. Top-level
`pr_40798_workspace_manager.diff` is the consolidated upstream diff.

## How these are used

- Anchor-drift detector (`scripts/check_upstream_drift.py`) reads
  these to know "what the upstream source looked like when the
  Genesis anchor was last validated".
- Patch authors copy snippets from these files into the `anchor_old`
  field of a new `TextPatcher` instance when proposing a backport.
- The dispatcher's anchor-resolution step does NOT load these files
  at runtime — it operates against the live `site-packages/vllm/...`
  tree. Snapshots are a contributor-facing reference, not a runtime
  dependency.

## Refresh policy

Snapshots are refreshed by hand when:

1. A pin bump (`vllm_pin_required` change) makes the old snapshot
   diverge enough from the live source that contributors can no
   longer use it as a reference.
2. A new patch family lands that anchors against a previously
   un-snapshotted file.

When refreshing, paste the verbatim upstream file at the new SHA
and update the patch's `applies_to` range in `dispatcher/registry.py`.
Snapshots from older SHAs may be kept for one release as a fallback
diffing target — delete them once `audit-no-stale-refs` flags the
mismatch.
