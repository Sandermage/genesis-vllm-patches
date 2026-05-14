# Gemma 4 — upstream PR tracking for next-model integration

Status: **NEXT IN ROADMAP** (operator decision 2026-05-14). Gemma 4 26B-A4B
is the next model class targeted for Genesis PROD after the current
Qwen3.6 stack stabilises on the dev338+gbf0d2dc6d nightly.

This file lists the upstream vLLM PRs that we should NOT text-port today
(too large / too unstable / pre-merge churn) but MUST track for the
pin-bump that absorbs them. When the watched PRs merge upstream, the
operator runs `make audit` to see the `NEWLY-MERGED` summary and decides
whether the existing pin already carries the fix or whether a pin bump
is needed.

## Tracking entries

### vllm#42637 — Mixed-attention KV quantization for Gemma 4

- **Author:** lesj0610
- **Size:** 19 files, +2673 / −345 (feature PR, not a bugfix)
- **Status at sweep:** OPEN
- **Why we track instead of port:**
  - Touches files we already text-patch (`v1/core/kv_cache_interface.py`,
    `single_type_kv_cache_manager.py`, `kv_cache_utils.py`,
    `turboquant_attn.py`, multiple Triton ops). Anchor-collision
    probability with our P83 / P85 / PN95 / PN96 / PN97 stack is
    near-certain.
  - Adds new TQ config schema. Backporting only the runtime hooks
    without the config-loader changes would break the boot path.
  - Author references vllm#40108 as a sibling approach with different
    semantics — pre-merge churn likely.
- **Action when merged:** pin bump, re-run `prove --all`, validate
  Wave 9 27B / 35B presets are bit-identical, then attempt the first
  Gemma 4 boot via a new `qwen3.6-…`-pattern preset (rename when ready).

### vllm#42635 — Gemma4 reasoning parser fixes

- **Author:** mann1x
- **Size:** 1 file, +22 / 0
- **Status at sweep:** OPEN
- **Why we track:** trivial 22-line fix to the reasoning-parser
  half-open-thinking edge case for Gemma 4 specifically. Our Qwen3
  parser path is unaffected. When Gemma 4 lands in PROD we want this
  fix carried; the pre-merge form is small enough that a future PN
  patch could backport it directly if the upstream PR stalls.
- **Action when merged:** absorb via pin bump.
- **Action if upstream stalls and we ship Gemma 4 first:** backport as a
  Genesis text-patch in the `reasoning` family (sibling to PN66
  multi-turn `</think>` leak fix).

### vllm#42559 — Gemma4 frontend empty thought-channel primer

- **Author:** the-david-oy
- **Size:** 2 files, +121 / 0
- **Status at sweep:** OPEN
- **Why we track:** chat-template + APC primer alignment for Gemma 4.
  Affects the historical-assistant-message replay path that currently
  silently drops Gemma 4's `<thought>` boundary. Qwen3 has its own
  primer in `pn91_developer_role_normalizer.py` — Gemma 4 needs the
  equivalent.
- **Action when merged:** absorb via pin bump.
- **Action if we ship Gemma 4 first:** backport as a Genesis text-patch
  in the `serving` family.

## Operator workflow

1. After every pin bump, `make audit` walks the registry and queries the
   GitHub API for each `upstream_pr` field. The summary lists how many
   are now `NEWLY-MERGED`, `STALE-RETIRED`, etc.
2. The three PRs above are *intentionally absent* from the registry as
   active patches — they have no `apply_module`, no env flag, no boot
   hook. This file is the only place they live.
3. When upstream merges any of the three, copy a stub registry entry
   into `dispatcher/registry.py` with `lifecycle: retired` and the
   `superseded_by` field pointing at the upstream commit, OR (if the
   pin we are using does NOT yet carry the merge), build the text-patch
   and wire it into `integrations/<family>/`.

## Cross-references

- The full 2026-05-14 PR sweep analysis lives in `CHANGELOG.md` under
  `[v11.0.0+wave9_dev338_pr_sweep]`.
- The four patches that DID get text-ported in the same sweep are
  P108 / P109 / PN110 / PN111.
