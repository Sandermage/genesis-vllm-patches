# dev491 Patch-Health Audit + Config-Currency Pass — 2026-06-14

LIVE pin: `0.22.1rc1.dev491+g1033ffac2` (PROD container `vllm-qwen3.6-35b-balanced-k3`).
Rollback pin: `0.22.1rc1.dev259+g303916e93` (`:nightly-303916e93`).

## Goal (user)

100% verify every patch is workable on dev491; find any that "really didn't
start" or "address the wrong place" (anchor drift onto the refactored engine);
rewrite for dev491 where needed; make the project + configs current so
everything launches through them and the "missing-patches" class can't recur.

## Method

1. **Inline scout** of the LIVE container: full apply-matrix (`[PatchMetrics]`
   status per patch), 100 `GENESIS_ENABLE_*=1` flags, the enabled-but-not-applied
   cross-reference, the not-dispatched gap, and a `.py` snapshot of the dev491
   engine tree (`/tmp/dev491_vllm`, 2724 files) for ground-truth file reads.
2. **14-agent Workflow** (`dev491-patch-health-audit`, ~1.9M tokens): 8 per-family
   verifiers (each reads OUR patch source AND the dev491 target, verdict per
   patch) → adversarial verify of every claimed failure → not-dispatched triage
   (25 patches) → 4 config-audit dimensions → synthesis.

## HEADLINE RESULT — the patch system is healthy

**0 REAL_FAILURE. 0 NOT_DISPATCHED_BUG.** Every enabled patch is either:

- **APPLIED_OK** — anchor/marker physically present in the live dev491 tree and
  firing (not a masked no-op). Spot-proven: `Genesis PN341` marker at
  `gpu_model_runner.py:906-915`, `PN340` at `gdn_attn.py`, `PN368` at
  `marlin_moe.py` — the Batch A perf stack is live and effective. TQ tunables
  confirmed in the live env (`VLLM_TQ_DECODE_BLOCK_KV=32/NUM_WARPS=8/NUM_STAGES=3`,
  `GENESIS_P67_NUM_WARPS=4`) — no dead-tune.
- **BENIGN_UPSTREAM_FIXED** — anchor absent because dev491 already contains the
  fix/refactor; the patch correctly no-ops; we are NOT exposed. The bug is
  **gone, not moved** — proven by reading the dev491 source:

  | Patch | dev491 evidence the bug is gone |
  |---|---|
  | PN347 | `scaled_mm/marlin.py` refactored — buggy `w_q.shape != (in,out)` transpose guard deleted; layout via explicit `size_k_first` (vllm#44113 closed-unmerged, solved structurally) |
  | P94   | `llm_base_proposer.prepare_next_token_ids_padded` has the native in-place loop (#41043) |
  | PN82  | `gpu_model_runner.py:2360` `is_prefilling[num_reqs:]=False` native (#41873) |
  | PN52  | `prompt_logprob.py:55` fixed `computed_prefill < prompt_lens` (#41411) |
  | PN19  | `gpu_worker.py:228` native `_scoped_allocator_max_split` (#41268) |
  | PN132 | `topk_topp_triton.py:892` native `.contiguous()` (#42739) |
  | PN90  | `gpu_model_runner.py:3640` native `_get_spec_decode_draft_probs` (#40269) |
  | PN22  | `qwen3.py:268` inherits `LocalArgmaxMixin` (#39419) |
  | PN133 | `scheduler.py:1549` native `max(len(generated_token_ids) - num_sampled, 0)` (#42722) |
- **CORRECT_SKIP** — default-off / opt-in / arch-gated (Gemma on a Qwen run) /
  deliberately version-gated. P61c/P64/PN56 skip via the **live** version-gate
  (`GENESIS_ENFORCE_VERSION_RANGE=1` confirmed in the container env) — they would
  CORRUPT dev491's engine-native `Qwen3CoderToolParser` (#45171) if applied, and
  remain load-bearing on the dev259 rollback (their range `<dev491` includes it).

**P68** "not dispatched" is a status-label artifact, not a bug: P68/P69 share one
apply_module + one legacy hook; the hook applied (`serving.py:265-266`) and P68's
runtime mitigation reads its own flag (`long_ctx_tool_adherence.py:309`) — live.

The user's feared class — "enabled patch silently no-ops while the problem still
exists" — **does not occur** anywhere in the load-bearing set.

## The real work was CONFIG INTEGRITY — fixed this session

- **T1.1 — pin-vs-image drift guard** (`scripts/audit_v2_vllm_pin_consistency.py`).
  Root cause of silent pin drift: the existing check compares model pin vs a
  baseline JSON, but every model has `reference_metrics_ref: null`, so it skipped
  vacuously and never noticed model pins declaring dev259 while the a5000-2x
  hardware ships the dev491 image. Added a baseline-independent check: each
  model's `vllm_pin_required` SHA must equal the PROD hardware image SHA, honoring
  an explicit `versions.pin_hold: true` exemption. Now EXIT=1 on real drift.
- **T1.2 — 35B pin bumped dev259→dev491** (validated this session: boot 0 failed,
  streaming tool-calls `{"city":"Paris"}` no XML leak, chat-matrix 254 thinking_off,
  Batch A perf). 27B-tq + gemma-26b given documented `pin_hold` (they run the
  dev491 image but were only validated on dev259 — see follow-ups). Pin audit
  now EXIT=0 with honest state.
- **T1.3 — PN125 version-gate bypass fixed.** PN125's `apply()` checked only
  `_env_enabled()`, so the LEGACY apply loop (which, unlike the spec path, never
  calls `should_apply`) let it report "applied" on dev491 despite its range
  `<0.22.0` — making the hardware YAML's "PN125/PN90 correctly SKIP on dev491"
  claim FALSE. Added the `should_apply("PN125")` self-gate (mirroring PN90).
  Verified: `should_apply("PN125")` returns VERSION-GATE skip on live dev491.
  Benign today (idempotent no-op) but the gate is now a reliable kill-switch.
- **T1.4 — PN133 drift-marker made operand-agnostic.** dev491 merged #42722 but
  emits `max(len(generated_token_ids) - num_sampled, 0)` vs the `- 1` marker, so
  PN133 fell through to a noisy `required_anchor_missing` DRIFT warning instead
  of a clean `upstream_merged` skip. Marker now matches the stable `max(...,0)`
  invariant; verified it self-retires cleanly on the dev491 snapshot.
- **T2 — removed 46 dead retired-flag lines** (P94/PN82/PN52/PN19/PN132 —
  retired + upstream-merged + version-gated-out on BOTH dev491 AND dev259) from
  14 model/profile/compose YAMLs. All parse; 0 left. KEPT the cross-pin gates
  (P61c/P64/PN56/PN347) and version-gated insurance (PN90/PN125) — they are
  load-bearing on the dev259 rollback and the live gate handles the per-pin call.

## Follow-ups (NOT done this session)

### Needs rig time (disrupts live 35B PROD — one rig)
- **27B-int4 TQ k8v4** dev491 re-validation → then bump pin + remove `pin_hold`.
  Smoke+bench+tool-call (qwen3_coder 7/7, TQ decode kernels, PN119 chain).
- **Gemma-4 26B AWQ** dev491 re-validation → bump + remove `pin_hold` (native
  gemma4 parser unaffected by #45171, but the full stack needs its own smoke).
- **dflash variants** (`qwen3.6-{27b,35b}-...-dflash`) pinned dev371: re-verify
  the **PN275** anchor at `config/vllm.py:1703-1715` on dev491 FIRST (if it
  misses that is a separate failure to file), then bump. Currently `pin_hold`.

### Pre-Blackwell latent (NOT a PROD bug on sm_86)
- **PN14 grouped-kernel clamp.** PN14's legacy `_tq_decode_stage1` clamp lands,
  but the grouped kernel `_tq_grouped_decode_stage1`
  (`triton_turboquant_decode.py:421` — the PROD k8v4 GQA decode path via PN119)
  still does an unclamped `Block_table_ptr + bt_base + page_idx` load. Benign on
  sm_86 (the OOB assertion fires only sm_89+/Blackwell) but must be closed BEFORE
  any Blackwell rig. Add a second `required=False` sub-patch mirroring the
  `safe_page_idx = tl.where(kv_mask, page_idx, 0)` legacy fix.

### Audit-coverage hardening (defense-in-depth, optional)
- **Split-env blind spot.** 23 `GENESIS_ENABLE_*` flags + `GENESIS_ENFORCE_VERSION_RANGE`
  live in `hardware.system_env`, which the YAML-consistency matrix
  (`profile.py:1010` reads only `cfg.genesis_env`) cannot see. They DO reach the
  container (verified live), but an operator diffing YAML-vs-launcher would think
  PN296/PN340/PN341/ENFORCE are "not enabled." Fix: scan both `genesis_env` and
  `system_env` for `GENESIS_ENABLE_/GENESIS_ENFORCE_` keys, or relocate the
  hardware patch flags into a merged `genesis_env` so there is one source of truth.
- **Env-only / version-gate-bypass lint.** PN125 was the live instance; PN132/PN347
  share the same shape (apply() never calls should_apply, masked only because
  their anchors are absent). Add a CI lint: every registry entry with a non-empty
  `vllm_version_range` must map to an apply_module whose source calls
  `should_apply()` / `_check_version_gate()`.
- **Tunable-layer drift capture.** The standing drift audit reads only
  `GENESIS_ENABLE_*`. Capture the FULL container env and diff `VLLM_TQ_DECODE_*`,
  `GENESIS_P67_NUM_WARPS`, `GENESIS_P82_THRESHOLD_SINGLE`, `GENESIS_TQ_MAX_MODEL_LEN`,
  etc. against the YAML — the layer where the historical P18b dead-tune happened.

### Cosmetic
- **qwen3_xml comment** (`qwen3.6-35b-a3b-fp8.yaml` capabilities): dev491 maps
  both `qwen3_xml` and `qwen3_coder` → `Qwen3CoderToolParser` (#45171). The
  current `qwen3_xml` value works (streaming tool-calls verified live); the
  "revisit on dev491" comment is now stale and can be updated or switched to
  `qwen3_coder` for clarity (both resolve to the same parser).
