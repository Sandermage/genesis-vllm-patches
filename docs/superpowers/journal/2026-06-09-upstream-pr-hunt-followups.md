# 2026-06-09 — Upstream PR hunt findings + followup queue

Captured during the multi-iteration ``/loop`` session that produced commits
``a29e9e49 .. 14f7a971`` (+ ``467971e4``). Agent: "vllm recent PRs hunt"
(2026-06-09, ~3 minutes, 46 tool calls). Anchor for future operators.

## TL;DR

Current pin ``0.22.1rc1.dev259+g303916e93`` is **already at or past 5 of the
6 highest-leverage merged PRs** the agent identified::

  PR #44050 (breakable cudagraph, opt-in env)        ← in pin, env-gated
  PR #44082 (MTP × SWA prefix-cache hits)            ← in pin, active
  PR #44253 (MTP speculator-prefill CG capture)      ← in pin, active
  PR #44700 (split mixed prefill/decode → recurrent) ← in pin, active
                                                       (+24 % B200 claim;
                                                        already firing on
                                                        our Qwen3.6 hybrid)
  PR #42301 (dual-stream GDN input projection)       ← we vendor as PN204
  PR #41824 (in-place SSM state, OPEN)               ← upstream still open

So pin-bump as a strategy yields no incremental gain right now. Real
remaining vendoring candidates are **open** PRs.

## Agent's findings — by category

### A. Merged-and-active in our pin (no work needed)

All five of these are in dev259 because dev259 is 22 commits ahead of
PR #44700's merge commit ``fa27d4e9cf3c8d8a5a143f38c346b27c02b2c2e3``:

* **PR #44700** — *[PERF][Qwen3.5] Split mixed prefill+decode batches:
  route decodes to the recurrent kernel.* Merged 2026-06-06. Author bench
  +24.4 % total TPS on B200; expected +8-12 % on A5000. Verified active:
  ``fused_sigmoid_gating_delta_rule_update`` import + 3 call sites at
  lines 1553/1580/1638 of ``qwen_gdn_linear_attn.py``.
* **PR #44082** — *[Bugfix] Cache the EAGLE/MTP lookahead block in the
  SWA prefix-cache mask.* Merged 2026-06-02. Restores SWA prefix-cache
  hit-rate from 0 → 55 % on MTP × hybrid GDN. Our issue #40124's
  ``turn 1: 0  turn 2: 0`` symptom would have been silently absorbing
  this fix every reboot. Active.
* **PR #44253** — *[Bug Fix][MRV2][Spec Decode] Warmup & capture with
  different attention states for speculator prefill.* Merged 2026-06-03.
  Eliminates IMA-class cudagraph-capture stall during MTP speculator
  prefill. Active.
* **PR #44050** — *[MRV2] Support breakable CUDA graph.* Merged
  2026-05-30. Env-gated via ``VLLM_USE_BREAKABLE_CUDAGRAPH=1``. **Tested
  2026-06-08, REGRESSED OUR TPS BY ~50 %.** This env flag also disables
  ``torch.compile`` entirely (``Equivalent to -cc.mode=none``) — the
  warning we missed before flipping it on. Useful for debugging cudagraph
  capture failures, not for production throughput. DO NOT ENABLE.

### B. Open PRs worth vendoring (ordered by ROI)

1. **PR #43955** — *[Perf] Reduce MTP decode bubbles for Qwen3.5 hybrid
   models* (Nekofish-L, 2026-05-29, REVIEW_REQUIRED). 91+/26- across 2
   files: ``v1/attention/backends/gdn_attn.py`` +
   ``v1/worker/gpu_model_runner.py``. Targets the exact CPU-bubble
   pattern we hit on accepted-token bookkeeping. Same anchor surface as
   PN125/PN128/PN286 — anchor fragility on subsequent pin bumps:
   medium-low. **Highest-leverage open backport candidate.** Expected
   gain on our config: small but consistent decoder-loop win, probably
   +2-4 % TPS at single-stream + bigger at conc>=4.
2. **PR #41824** — *[Kernel] Enable in-place SSM state access for GDN
   chunk prefill* (Kermit-C, 2026-05-06, REVIEW_REQUIRED). 179+/63-
   across 4 files including Triton kernel diffs. Author reports
   +10-13 % RPS on H20; our A5000 should see similar share. **Requires
   PR #42076 (precision fix on SM≥90), but SM 8.6 is unaffected** — we
   can vendor without #42076. Risk: Triton kernel anchors are the most
   fragile patch class. Plan to re-anchor on every pin bump for ~2
   versions.
3. **PR #42301** — *[Model][Perf] Overlap Qwen3.5 GDN input projections
   on dual CUDA streams* (zhangxin81, 2026-05-11, REVIEW_REQUIRED).
   21+/2-, single file. **WE ALREADY VENDOR THIS AS PN204.** Confirmed
   active in production with ``GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ=1
   + GENESIS_PN204_ARM_AFTER_WARMUP=1`` (commit ``5c20b51f``). Multi-conc
   bench shows +19 TPS at conc=2 outside CV.

### C. Open issues matching our symptoms (informational)

* **#40124** (our own tracking issue) — Genesis stack symptom catalog;
  PR #44082 (now merged + in pin) addresses one of the catalog items.
* **#41190** — TP=2 spec-decode on Qwen3.6 hybrid GDN
  ``cudaErrorIllegalAddress`` at ``num_accepted_tokens_event.synchronize()``.
  PR #43909 (Gemma4 MTP TP>1 IMA fix) is the closest analogue — worth
  manual check on our gpu_model_runner.py:1927 region.
* **#42084** — GDN ``mamba_get_block_table_tensor`` torch.gather IOOB
  when prefix caching + ``num_speculative_tokens>=10``. Our PROD K=3
  stays well under threshold. Watch for follow-ups.
* **#44688** — Qwen3.6-35B-A3B-FP8 on RTX PRO 6000 Blackwell: missing
  FP8 MoE config ``E=256, N=256``. Adjacent topology; suggests a global
  MoE autotune table miss that may also affect our A5000 path. Worth a
  manual tune-config audit before next pin bump.

## Concrete followup for next session

1. **Vendor PR #43955** (highest-confidence open backport). Effort:
   ~6 hours including A5000 bench A/B. Anchor count: 2-3 (metadata
   builder rebase + accepted-token bookkeeping site).
2. **Vendor PR #41824** if A5000 microbench shows >5 % standalone gain
   when patched against a dev sandbox. Effort: ~1.5 days; anchor count
   higher (4 files, including Triton kernel diff).
3. **Audit fp8 MoE autotune table coverage** for ``E=256, N=256`` shape
   on our 35B FP8 + 27B INT4. If missing, the workload may be falling
   back to a default-slow path that issue #44688 highlighted on
   Blackwell.
4. **Mamba prefill kernel coverage** (ssd_chunk_scan + ssd_bmm): 4 + 1
   ``num_warps=8`` sites in static Config lists. Different pattern from
   the env-cap approach we used in PN299C/D/E — would need list rewrite
   or runtime autotune wrapper. Low ROI (prefill TTFT only, not
   sustained decode TPS). Deferred.

## Negative results to remember

* ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` regresses our TPS by ~50 %. Disables
  ``torch.compile`` entirely. **Do not enable in production.**
* Bench-validated v2 of PN299E preserves ``num_stages=10`` for the KV
  cache writer — capping it to 2 (the PN296 default) crushes the write-
  pipeline latency hiding for the kernel and loses -8 TPS at conc=1. The
  warps cap alone (4 instead of 16) is the safe + correct win.
* PN300 (universal Triton autotune wrapper) re-tested 2026-06-08 with
  the full session-end patch stack — no measurable TPS gain over
  PN298+PN299+PN299B+PN299C+PN299D+PN299E. Stays opt-in / off-by-default.
