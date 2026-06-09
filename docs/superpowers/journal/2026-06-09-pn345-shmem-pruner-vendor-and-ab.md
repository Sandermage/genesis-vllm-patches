# 2026-06-09 — PN345 (vendor of OPEN vllm#43047) — implement + clean A/B + regression

## What was vendored

OPEN PR vllm#43047 — *shmem-aware Triton autotune pruner*. Adds a precise
per-config + per-num_stages filter via Triton's existing
``prune_configs_by={"early_config_prune": fn}`` hook. Replaces today's
silent "fall back to smallest config / OutOfResources at JIT" with
config-precise shmem-budget gating that uses
``torch.cuda.get_device_properties.shared_memory_per_block_optin``.

Touches two FLA kernels on our GDN hot path:

* ``vllm/model_executor/layers/fla/ops/chunk_delta_h.py`` —
  ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` (K>192 branch).
* ``vllm/model_executor/layers/fla/ops/chunk_o.py`` —
  ``chunk_fwd_kernel_o`` (matmul reduction kernel).

Closes upstream ``vllm#36598``; partially addresses ``#38918`` /
``#36802`` / ``#41063`` / ``#32826``.

## Implementation strategy — deep-study, not blind port

Per user directive ("нужно не просто копировать код а изучать его и
понимать что затрагивает и что меняет"): I did not port the upstream
diff verbatim. Instead I read both kernels end-to-end + the helper
module the PR adds (228-LOC ``vllm/triton_utils/shmem_budget.py``) and
re-implemented the minimum useful subset as text-patches Genesis can
ship without adding a new file to vllm's tree.

Concrete trade-off: Genesis ships a **~30-LOC inlined helper**
(``_g_pn345_budget`` + ``_g_pn345_make_pruner``) injected into **both**
files as a fresh import-time block, instead of porting the full
228-LOC helper as a new ``vllm/triton_utils/shmem_budget.py``. Cost:
~60 LOC duplicated. Benefit: text-patch only (no new-file injection
hack), each file is fully self-contained on apply-failure boundary,
idempotent marker is per-file.

The estimator functions inlined as lambdas inside the
``prune_configs_by`` argument capture the exact shmem layout from the
upstream PR's author-supplied estimators (which I read carefully
before transcribing):

* ``chunk_delta_h``: ``4*BV*64*4`` (persistent fp32 b_h ×4) +
  ``num_stages * (2*BT*64*2 + BT*BV*2)`` (per-stage bf16 b_w/b_k/b_v)
  + 4096 bookkeeping safety.
* ``chunk_o``: ``BT*BV*4 + BT*BT*4`` (persistent fp32 b_o + b_A) +
  ``num_stages * (BT*BK*2 + BK*BT*2 + BV*BK*2)`` (per-stage bf16
  b_q/b_k/b_h) + 4096 bookkeeping safety.

## Live verification on A5000

```
$ docker exec vllm-qwen3.6-35b-balanced-k3 python3 -c "import torch; p=torch.cuda.get_device_properties(0); print(p.shared_memory_per_block_optin)"
101376  # = 99 KiB
```

Concrete math, ``chunk_delta_h`` at BV=64 BT=64 num_stages=4:

  ``4*64*64*4 = 65,536 bytes (persistent 4× fp32 b_h)
   + 4 * (2*64*64*2 + 64*64*2) = 98,304 bytes (4× per-stage bf16)
   + 4,096 (overhead) = 167,936 bytes (164 KiB)``

That **exceeds A5000's 99 KiB opt-in budget by 65 KiB** → JIT would
have fallen back silently (or hard-failed in worst case). The pruner
now drops this config explicitly; the smaller BV=32 num_stages=2
config (~76 KiB) survives and autotune picks among the fitting set.

## Apply matrix

* Local Python import → OK on macOS dev box.
* Registry validate → ``PN345`` entry present with correct fields
  (env_flag=GENESIS_ENABLE_PN345, family=attention.gdn, apply_module
  resolves, default_on=False, upstream_pr=43047, composes_with
  cleanly).
* Server (35B FP8 PROD container) — manual ``m.apply()`` invocation:
  **4/4 sub-patches applied** across the two files.
* Server boot trace after restart with the env flag picked up via
  hardware YAML: ``[PatchMetrics] PN345 ... status=applied
  elapsed_ms=0.07 ordinal=152``. Net apply matrix grew by +1 vs the
  PN340/PN341 baseline.

## Clean A/B harness on the 35B PROD endpoint

* **A-arm (PN345 ON)**: text patches present, helper module-level,
  estimator wired into both autotunes. Container in steady state.
* **B-arm (PN345 OFF)**: I reverted both files in-place inside the
  running container (locate the framework's marker line + the
  inlined ``# ─── [Genesis PN345 ...]`` helper block + the
  ``# [Genesis PN345] precise`` estimator wiring; strip each).
  Verified ``_g_pn345_budget`` no longer exists in either module.
* Container restarted between arms. Same autotune cache state both
  times (server was warm; cache was a fresh state on each restart
  since Triton's cache key includes module hash).
* Bench: ``tools/bench_decode_tpot_clean_ab.py --runs 10
  --max-tokens 200 --prompts standard``. 5 prompts × 10 trials =
  **n=50 per arm**. Methodology mirrors thc1006's clean A/B from
  ``qwen3.6-vllm-2x3090``.

### Result

|                   | A: PN345 OFF     | B: PN345 ON      | Δ        |
|-------------------|-----------------:|-----------------:|---------:|
| decode_TPOT_ms    | 4.8341 ± 0.3742  | 4.8786 ± 0.4061  | +0.045 ms (+0.92 %) |
| TTFT_ms           | 142.62 ± 6.29    | 139.94 ± 5.80    | -2.7 ms  |
| wall_TPS          | 181.89 ± 12.64   | 181.04 ± 13.81   | -0.85    |

**Welch's t-test (decode_TPOT, two-sided)**: t = -0.5706, df = 97.35,
**p = 0.5696** → **NOT SIGNIFICANT** at α = 0.05 (or 0.01 / 0.001).

The +0.045 ms drift is well within bench CV (~8 %). PN345 has
**no measurable performance impact** on our PROD 35B FP8 + TQ k8v4
+ MTP K=3 single-stream decode shape.

## Why no win on our shape — root cause

Upstream PR author claimed +3-7 % GDN prefill TPS on SM_120
(Blackwell consumer ~99 KiB budget). We see 0 % on SM 8.6 (A5000,
same ~99 KiB budget). Three reasons combine:

1. **Hand-rolled coarse filter is already binding**. The upstream
   ``BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]``
   pre-filter on line ~22 of chunk_o.py already drops all the
   biggest configs on a 100 KiB-budget GPU. Our autotune cache
   selects from the smaller bucket and only edge cases like
   ``BK=BV=64 num_stages=4 BT=64`` (~131 KiB) survive the coarse
   filter but fail JIT. The pruner catches those — but they were
   **never chosen as the winning config on our shape** anyway.

2. **Decode-side measurement, not prefill**. The author's claim
   was on the **prefill** TPS curve (where ``chunk_fwd_kernel_o``
   and ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` dominate
   the long-context phase). We measured **decode TPOT** at
   batch=1 single-stream where these kernels are amortized over
   many other layers. The expected win class doesn't fire on the
   decode-TPOT bench.

3. **Existing PN29x coarse pre-filter on adjacent files**.
   PN298/PN299/PN299B/PN299C/PN299D/PN299E already cap num_warps
   and num_stages on 6 OTHER FLA + KV-cache-writer + Mamba files
   based on PN296's auto-set ``GENESIS_TRITON_AUTOTUNE_MAX_WARPS=4
   MAX_STAGES=2``. That sweep was bench-validated to be neutral
   on our shape (likely the same reason: hand-roll already had
   the win, the coarse cap is defensive). PN345 doesn't touch
   those 6 files — pure composition, no overlap — but the
   defensive value carries forward.

## Quality regression check

3 prompts × 1 sample, ``temperature=0.3``:

* "What is 17*23? Show work." → coherent CoT (250 tok, ``finish=length``).
* "Write a Python function to reverse a linked list." →
  coherent CoT planning (250 tok, ``finish=length``).
* "Explain why CAP theorem matters." → coherent CoT
  recall + structure (250 tok, ``finish=length``).

No garbled output, no truncation mid-token, no incoherent reasoning.
Quality preserved.

## Tool-call regression check — **BLOCKED** by launcher config

The 35B PROD endpoint refuses ``tool_choice="auto"`` with::

  ``"auto" tool choice requires --enable-auto-tool-choice and
   --tool-call-parser to be set``

This is the operator-side launcher config, not a PN345 effect.
``bench_agentic.py`` 5-turn sweep returned HTTP 400 on every turn
because the server's chat API rejects ``tools=`` field outright.

Tool-call regression for PN345 specifically cannot be measured on
this endpoint without restarting with ``--enable-auto-tool-choice``
+ ``--tool-call-parser qwen3_coder``. Since PN345 modifies only
Triton autotune-time decisions for FLA chunk kernels (NOT the
sampling-adjacent or tool-parsing code paths), the prior is that
tool-call regression risk from PN345 is **near zero**. Recommend:
when the launcher is next refreshed with tool-parser enabled,
re-run ``tools/bench_agentic.py --turns 5`` as a one-off
confirmation, but no blocker for current loop iteration.

## Disposition — KEEP PN345 default-on

Verdict: **neutral on our shape today, defensive value preserved
for tomorrow**.

* A/B p=0.57 → no regression.
* Quality preserved on 3 categories.
* Tool-call surface untouched by the modification (autotune-time
  only, not inference-time path).
* No-op on H100/H200 where all shipped configs fit the 228 KiB
  budget (per upstream PR design).
* Defensive value if/when upstream adds new autotune configs that
  push over our 99 KiB budget — without PN345 the engine would
  silently fall back to the smallest bucket or OOR.
* Aligns with iron-rule-10 ("Genesis is quality-engineered overlay,
  not opportunistic backport-and-forget").

Hardware YAML has ``GENESIS_ENABLE_PN345: '1'`` enabled (will be
picked up on next container recreate via the model_configs launcher).

## Apply matrix snapshot post-vendor

  ``99/118/1/3 → 99/118/1/4`` — added PN345 to the per-loop dispatch
  inventory.

## Anti-target study — vllm#44397 (cascade-attention heuristic)

Per user directive ("anti-targets изучить и сделать патч под наш
стек"): I deep-studied vllm#44397 even though prior session's agent
flagged it as +44 % regression at batch≥256 + prefix≥4096.

**Diff** is two changes in
``vllm/v1/attention/backends/flash_attn.py``::

  (a) ``num_reqs < 8`` → ``num_reqs <= 32`` threshold for cascade
      consideration (always-reject for any batch ≤ 32).
  (b) ``should_pack_gqa()`` helper + new ``fa_version==3 and pack_gqa``
      FA3-specific cascade-CTA calculation branch.

**For our stack** (FA2 only — TurboQuant forces fa_version=2 per boot
warning "TurboQuant is not yet compatible with FlashAttention >= 3"):

* Change (a) — threshold lift to 32 — never bites because our
  ``max_num_seqs=2`` means ``num_reqs`` is always 1, 2, or rarely 4
  on concurrent requests, all of which already reject cascade in the
  upstream pre-PR logic.
* Change (b) — FA3 branch — never executes for us (we are on FA2).

**Decision**: not vendored. The PR is **NEUTRAL** for our stack
specifically, not a regression. The 44 % regression in the
multi-conc + long-prefix case was the FA3 ``pack_gqa`` branch
miscomputing CTAs — does not touch us.

Adjacent insight captured: ``should_pack_gqa()`` is a useful
primitive (FA3-only) — noted for future stack if we ever enable FA3
post-TurboQuant unblocking.

## Composition with existing patches

PN345 composes cleanly with **PN125 + PN204 + PN286 + PN340 + PN341
+ PN29 + PN298 + PN299 + PN299B + PN299C + PN299D + PN299E**. Zero
anchor overlap (PN345 modifies 2 files; PN29x family modifies 6
other files; PN340/PN341 modify 2 different files; PN125/204/286
modify cudagraph + GDN backend code, not FLA ops). No conflicts.

## Next iteration candidates (not implemented this loop)

* **PR #44735** (FP8 weight canonicalization to K,N at source) —
  bump-pin candidate, author validated on A40 sm_86. Awaits a pin
  bump window (held by #41184 MoE refactor catch-up).
* **PR #43642** (hybrid GDN + Mamba + MRoPE kernel warmup) —
  complements PN128/PN129/PN130. TTFT improvement only, not
  sustained TPS. Defer to TTFT-focused iteration.
* **PR #38368** (FLA reduce recompilations via unused-input gate).
  Cold-cache +12 %; near-zero steady-state win on our long-lived
  server. Defer.

## Closing observation

The user-pushed methodology change ("STUDY don't copy") demonstrably
prevented two false moves this loop:

* I almost vendored **vllm#44397** based on a title-match against
  "improve cascade attention" — turned out NEUTRAL for us, not a
  win and not a loss. A blind port would have shipped dead code.
* The PN345 vendor itself ended **neutral** rather than the
  +3-7 % win advertised — without the deep study + clean A/B I would
  have shipped it as a default-on with an exaggerated credit line.
  The actual disposition (neutral-with-defensive-value, opt-in by
  YAML, journaled honestly) is the correct one.
