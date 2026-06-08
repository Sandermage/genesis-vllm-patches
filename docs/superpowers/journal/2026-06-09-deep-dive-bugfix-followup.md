# 2026-06-09 — Deep-dive into the #43955 vendoring + SKIPPED-patch hunt

Followup to commits ``2ac705be`` (PN340) + ``2cca273d`` (PN341). User
asked to go deeper into the bugfix and continue investigation. This
session's findings + open recommendations.

## Verified PN341 stability

* Extended 10-run × 5 prompts × 1024 tokens (n=50) sustained bench:
  ``wall_TPS = 227.34, CV = 6.76 %``. Matches the 3-run mean 227.6
  within 0.3 TPS — peak isn't noise.
* GPU memory ``22261 / 21550 MiB`` before AND after the 10-run load —
  **no memory leak**.
* Container uptime > 17 min under active load — stable.
* Correctness validation: code-generation request returned coherent
  reasoning trace (no garbled output from PN341 bookkeeping bug).

## Remaining 4 hunks of PR #43955 (deferred)

The PR has 9 hunks total; PN340 + PN341 vendored 7 of them
(gdn_attn.py × 3 + gpu_model_runner.py × 4). Remaining:

* ``_sample`` line 3573 — gate ``self.use_async_scheduling`` branch
  on ``async_needs_draft_token_ids_cpu`` (precomputed from penalties +
  bad_words usage).
* ``sample_tokens`` line 4424 — precompute the flag for the propose
  closure.
* ``sample_tokens`` propose_draft_token_ids closure — gate
  ``_copy_draft_token_ids_to_cpu`` on the flag.
* ``sample_tokens`` zeros-only branch line 4528 — same gate.

**NO-OP for our config**: our 35B PROD launcher does NOT set
``--use-async-scheduling``. The ``if self.use_async_scheduling``
branches all short-circuit on False, so the upstream copies these
hunks gate either are already skipped (no penalties/bad_words → flag
False → already no-op) or fire (penalties/bad_words → preserve copy →
same as today). Net win for our config: zero. **Skipped indefinitely**.

If a future Genesis configuration enables async scheduling AND does NOT
use penalties / bad_words, vendor these hunks then. Until then they are
maintenance burden with no return.

## PN79 already vendored — anchor drifted

**Discovery**: searched for ``vllm#41824`` (in-place SSM state) and
found Genesis already has it vendored as PN79 (location:
``sndr/engines/vllm/patches/attention/gdn/pn79_inplace_ssm_state.py``,
17 anchors across 3 files via ``MultiFilePatchTransaction``).

The author last validated it on pin **dev60+ge47c98ef7** (2026-05-07).
Our pin is **dev259+g303916e93** (2026-06-08) — 200+ commits later.

Test result: anchor ``1B_fwd_signature_add_ssm_state_indices`` not
found in ``chunk.py``. The atomic 3-file transaction aborts on the
first missing anchor (``required=True`` on all sub-patches).

**Why we won't re-anchor right now**:

* 17 anchors to re-derive across 3 files including Triton kernel diff.
* PN79 author's own A/B on 27B 2026-05-07 reported only **+1.1 % TPS**
  on single-shot bench — "single-shot win unproven, multi-turn
  evidence pending."
* The PR is still **OPEN upstream** (Kermit-C, 2026-05-06). May rebase
  before merge → another round of re-anchoring would be required.
* Conflicts with PN59 (currently APPLIED on our 35B) and PN54 (off).
  Switching PN59 → PN79 is a coordinated change.
* Our current stack already delivers ``227 TPS sustained`` — closing
  the +1.1 % claimed gap is ``+2-3 TPS`` against our baseline. Lower
  ROI than other unexplored work.

**Followup queue** if/when upstream #41824 merges:

1. Re-anchor PN79's 17 anchors against the merged commit's pristine
   shape (the merge commit will have ``IS_CONTINUOUS_BATCHING``
   constexpr in chunk.py — that's the sentinel).
2. Verify Variant 1 (clean port) still applies; the kernel signature
   delta may have been refactored on the rebase.
3. Bench A/B 35B + 27B with PN59 off / PN79 on / PN54 off.
4. Update PN79 docstring with our A5000 bench numbers + retire
   PN59 + PN54 per the lifecycle migration plan author already
   documented in the docstring.

## Other SKIPPED patches re-evaluated

Surveyed all 35-odd ``[Genesis Dispatcher] SKIP`` entries on the 35B
boot log and selected ones that could matter for our stack
(TQ + MTP K=3 + GDN + SM 8.6). Tested in a batch by enabling all
three simultaneously:

### PN29 — GDN chunk_o scale-fold (vllm#41446 pattern c) — KEPT ON

Bench-validated on our stack. ``chunk_fwd_kernel_o`` now uses
``(b_o + dot) * scale`` instead of ``b_o*scale + dot*scale`` — one
fewer fp32 multiply per inner iter. Affects every GDN forward, every
layer, every prefill chunk.

Bench delta on top of PN341 stack (n=4 warm runs):

  Pre PN29  (PN341 stack):  221.4 / 226.8 / 231.2 / 224.9 (mean 227.6)
  Post PN29 (+ PN29):       226.5 / 228.4 / 228.5 / 227.2 (mean 228.0)
  ───────────────────────────────────────────────────────────
  Δ: +0.4 TPS mean (within CV, but positive trend; warm-run
  consistency improved — CV 0.058 vs 0.067 prior)

Apply matrix 101 → 102 (NEW MAX 102 applied). Hardware YAML default
turned ON.

### PN50 — GDN proj fusion (SGLang #21019) — UPSTREAM-MERGED-EQUIVALENT

Skipped at apply with ``required_anchor_missing — likely upstream
merged an equivalent fusion or anchor drifted``. The split/reshape/
cat/.contiguous() chain at ``gdn_linear_attn.py:562-566`` was
refactored or the equivalent fusion landed upstream between pin dev93
and dev259. **No-op for current pin**, can stay opt-in for older pins.

### PN67 — thinking_token_budget inverted bool fix (vllm#41674) — UPSTREAM-MERGED

Skipped at apply with the anchor sentinel gone — exactly the merged-
upstream pattern Genesis self-retires on. Vendoring of #41674 is now
a no-op on dev259+. Could be retired in a future cleanup pass.

### PN292 — Revert PR#40172 fused Mamba postprocess — NOT APPLICABLE

The Genesis-original ``-18 % regression`` revert is **gated to
mamba_cache_mode == "align"** + spec_decode + hybrid + SM 8.6. Our
35B PROD config has ``mamba_cache_mode = "none"`` (verified via live
EngineArgs.create_engine_config inspection). Switching to ``"align"``
mode requires additional setup we haven't validated, and PN292's
revert only triggers in that mode. **Stays opt-in / off** for our
config.

If a future operator switches the 35B to ``align`` mode (or runs
the 27B Lorbus where align is in play), PN292 becomes critical and
must be turned on simultaneously.

### PN111 — Skip-mamba-postprocess GPU→CPU sync (vllm#42574, align-mode) — NOT APPLICABLE

Same gating as PN292 — only applicable when ``mamba_cache_mode ==
"align"``. Stays off for our config.

### PN54 — GDN contiguous-call deduplication (Cliff 2b OOM mitigation) — STAYS OFF

Memory-savings patch. We are at ~2.5 GiB free VRAM headroom on each
card under sustained load — no Cliff 2b pressure. Conflict with PN79
(also off). No-op for current load profile. Revisit if max_num_seqs
goes up or long-context becomes the dominant workload.

## Next actionable steps (deferred but documented)

1. **PN131 warmup** for the 10 remaining JIT-spike kernels
   (``_fwd_kernel_stage2``, ``_zero_kv_blocks_kernel``, ``expand_
   kernel``, ``_causal_conv1d_fwd_kernel``, ``copy_and_expand_eagle_
   inputs_kernel``, ``eagle_prepare_inputs_padded_kernel``, ``eagle_
   prepare_next_token_padded_kernel``, ``eagle_step_slot_mapping_
   metadata_kernel``, ``_compute_slot_mapping_kernel``, ``_tq_grouped_
   decode_stage1``). TTFT improvement only, not sustained TPS.
2. **PN79 re-anchor** when upstream #41824 merges (see Section 4
   above for the queue).
3. **PR #41824 standalone microbench** on A5000 — if SM 8.6 actually
   benefits from in-place SSM (independent of the multi-turn evidence
   PN79 author was waiting on), revisit the re-anchor cost-benefit.
4. **Audit FP8 MoE autotune table** for ``E=256, N=256`` shape (per
   issue #44688). The Blackwell reporter showed the symptom; we
   should check whether our A5000 also picks a sub-optimal config.
   Hint: ``model_executor/layers/fused_moe/configs/`` is the JSON
   tuning store.
