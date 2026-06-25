# SPDX-License-Identifier: Apache-2.0
"""SNDR Core apply — shadow comparison: PatchSpec-driven order vs legacy.

PR38 Day 5 (2026-05-08): before flipping `orchestrator.run()` to use
`PatchSpec.apply_module` directly (Day 6-8), surface differences
between the two apply orders so operators can audit them off-line.

Two sources of order:

  1. **Legacy** — `apply._state.PATCH_REGISTRY` (list of (name, fn)),
     populated by `@register_patch` decorators in `_per_patch_dispatch.py`.
     This is the order Genesis has been running for years.

  2. **Spec-driven** — `dispatcher.iter_patch_specs()` yields a PatchSpec
     per `dispatcher.PATCH_REGISTRY` entry. Order today is registry
     dict-iteration order.

`compare_apply_orders()` returns a structured diff:

  - `legacy_only`: registered fn names with no PatchSpec match
  - `spec_only`: PatchSpec patch_ids with no legacy fn match
  - `legacy_count` / `spec_count`: total per-source counts
  - `coverage_pct`: fraction of spec-driven patches that have an
                    `apply_module` (i.e. could actually run via the
                    new dispatch loop)

CLI:

    python -m sndr.apply.shadow

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("genesis.apply.shadow")


# ─── Known divergent patches (P1-1 audit closure 2026-05-08) ─────────────
#
# Patches intentionally listed only in `dispatcher.PATCH_REGISTRY` and
# not in the legacy `_per_patch_dispatch.py` registry. Two reasons:
#
#   (a) registry-only documentation entries with no `apply_module`
#       — legacy ledger rows for retired/preflight/research patches
#       that don't have a runtime apply path. Adding @register_patch
#       for them would create a dummy dispatcher with no behavior.
#
#   (b) spec-only patches with `apply_module` set — these are the
#       direction the migration is going (registry as single source
#       of truth). Adding them to the legacy parking-lot module would
#       defeat the migration. Listed here so CI gate doesn't false-
#       positive on the intentional gap.
#
# Any new spec_only patch NOT in this set will surface as
# `unexpected_spec_only` and fail `--strict` mode. To add an entry
# here, the patch must either (a) lack an apply_module on purpose, or
# (b) be reviewed and confirmed as a registry-driven-only addition.
KNOWN_SPEC_ONLY_PATCHES: frozenset[str] = frozenset({
    # Category (a): registry-only ledger / preflight rows
    "P102",            # Spec-decode metadata + disagreement tracker (no apply yet)
    "P51",             # TQ-active runtime guard (legacy lifecycle, no on-disk impl)
    "PN60",            # Quant arg vs config.json validator (preflight DX)
    "PN63",            # fp8_e5m2 advisory (gpu_profile recommendation only)
    "PN64",            # Marlin MoE per-SM tuning placeholder for SM 12.0
    # Category (b): spec-only patches with apply_module — registry-
    # driven loop is the canonical path; legacy parking lot is going
    # away (PR38 Day 6-8 migration in progress).
    "P69",             # Long-context tool-format reminder (paired with P68)
    "PN40-classifier", # PN40 sub-D workload classifier middleware
    # Category (c): UNIFIED_CONFIG 2026-05-09+ spec-driven additions
    # — registered through patches/* modules with apply_module set,
    # but no legacy @register_patch entry (canonical path is
    # registry-driven from inception).
    "PN16_V6",         # Streaming <think> truncator middleware (Sprint 4)
    "PN122",  # renamed from SPRINT26_CG_DISPATCH_TRACE 2026-05-14
    # Category (c) continued — Phase 3 (2026-05-21) bucket 3+4
    # spec-driven onboarding. These patches were relocated to their
    # technical-area canonical home (spec_decode/ or attention/
    # turboquant/) AND made spec-only at the same time. Their apply()
    # is invoked by the registry-driven dispatcher loop, not by the
    # legacy @register_patch table. Adding @register_patch for them
    # would re-introduce the parking-lot dependency that PR38 is
    # migrating away from.
    #
    # Bucket 3 (spec_decode drafter routing relocated from gemma4/):
    "G4_71",           # drafter native attn-backend forcing
    "G4_71B",          # drafter sliding-window Triton routing
    "G4_72",           # drafter native KV cache spec
    "G4_73",           # drafter profile-skip
    "G4_74",           # drafter HND layout
    "G4_75",           # drafter head_size=512 Triton route
    "G4_76",           # disable drafter KV sharing
    "G4_78",           # drafter target KV bridge (lifecycle=retired)
    #
    # Bucket 1 (spec_decode probes — diagnostic):
    "PN262",           # FlashAttn drafter trace
    "PN262B",          # KV alloc trace
    #
    # Bucket 4 (TurboQuant cherry-pick overlay loader stack
    # relocated from gemma4/ to attention/turboquant/):
    "G4_19B",          # TQ KV spec integration (case-sensitive
                       # legacy register entry is 'G4_19b ...')
    "G4_19C",          # K,V round-trip attention wrapper
    "G4_31",           # preserve TQ dtype
    "G4_32",           # TQ validation bypass
    "G4_79",           # TQ supports_mm_prefix (Gemma 4 MM 0.22.1 unblock)
    "G4_80",           # fp8_e5m2 KV for weight-only checkpoints (vllm#45040)
    "G4_60A",          # TQ sliding-window spec
    "G4_60B",          # TQ overlay loader (turboquant_attn)
    "G4_60C",          # TQ overlay loader (triton_turboquant_decode)
    "G4_60D",          # TQ overlay loader (triton_turboquant_store)
    "G4_60E",          # KV cache utils overlay
    "G4_60G",          # attention dispatch overlay
    "G4_60H",          # TQ config augment overlay
    "G4_60K",          # arg_utils skip-list plumbing
    "G4_60L",          # TQ backend supports_mm_prefix override
    "G4_61",           # TQ shared workspace
    "G4_62",           # TQ kernel warmup
    "G4_67",           # TQ spec-verify routing
    "G4_68",           # TQ spec CG-downgrade overlay
    "G4_69",           # skip-layers native backend
    #
    # Bucket-6 R3 closure (2026-05-21) — marker-only registry rows:
    # envs read inside the bind-mount overlay, no apply_module by
    # design. Registered to close R3 config-keys catalog gap.
    "G4_70",           # mixed-allocator routing
    "G4_70B",          # mixed-allocator FAIL_FAST
    "G4_70C",          # skip-list plumbing companion
    "PN256",           # raw-K/V continuation inside overlay
    "PN261",           # TQ decode cache layout assert
    "PN271",           # KV contract audit (registry-only)
    "PN274",           # spec-decode KV adapter coordinator (lifecycle=coordinator)
    "PN282",           # spec-decode acceptance metric coordinator (boot-applied
                       # from sndr_core/__init__.py, not via dispatcher;
                       # STAGE-6-HARDENING.2C registration 2026-05-28)
    "PN283",           # prometheus_client multiprocess directory bootstrap
                       # coordinator — sibling of PN282 (same boot pattern,
                       # same SNDR_ENABLE_* canonical env naming); registered
                       # 2026-06-01 to close orphan-flag gap surfaced by
                       # audit_config_keys after chat-K3 profile promotion
    # ── 2026-05-30 session additions — spec-driven from inception ──
    "PN288",           # qwen3_coder tool-call finish_reason override
                       # (§1.3 Phase B+C; serving-layer text-patch
                       # delegating to middleware helper; canonical
                       # registry-driven apply path, no legacy entry)
    "PN289",           # Genesis process-info Prometheus gauge
                       # (§6.H10 enterprise observability; *_info
                       # pattern, no legacy register table presence
                       # by design — emits a gauge, not a runtime
                       # mutation)
    "G4_T1",           # Gemma4 tool-parser PR #42006 vendor marker;
                       # apply_module is the marker stub, actual
                       # vendored file is operator-side bind-mount
    # ── Iteration N+3 (2026-06, commit 1bfbf695) — spec-driven from
    # inception, SYNCED but DEFERRED (intentional; see commit msg):
    "PN353B",          # TQ prefill CG capture safety (vendor of OPEN
                       # vllm#43747); apply_module set, registry-driven
                       # canonical path, no legacy entry by design
    "PN353A",          # TQ builder workspace reserve (vendor of OPEN
                       # vllm#44053); legacy @register_patch wrapper REMOVED
                       # 2026-06-17 (consolidation §2.2.A) — apply() self-gates
                       # via should_apply, so spec-only path is byte-identical
    "PN357",           # remapped greedy draft selection speedup
                       # (vendor of OPEN vllm#43349); same class
    # ── 2026-06-11 50-PR sweep wave 1 — spec-driven from inception
    # (PN370/PN372/PN374/PN375 got legacy parking-lot hooks; these
    # two ride the registry-driven path only, same class as PN353B):
    "PN371",           # deferred ref-pinned encoder-cache eviction
                       # (vendor of CLOSED vllm#45199, Gemma-4 vision
                       # + MTP + async 'Encoder cache miss' fix)
    "PN373",           # parallel_tool_calls explicit null != false
                       # (vendor of OPEN vllm#44955; serving-layer
                       # text patch on tool_calls_utils.py)
    # ── 2026-06-13 50-PR sweep wave 2 — spec-driven from inception
    # (apply_module + own apply(), no legacy @register_patch hook; same
    # class as PN371/PN373). PN377 is the lone wave-2 patch with a
    # legacy parking-lot hook, so it is NOT listed here.
    "P88",             # prefix-cache stats retry de-dup (rewrite of
                       # vllm#45202; metrics-only, KVCacheManager text)
    "PN358",           # FULL cudagraph forward-context refresh
                       # (vendor of OPEN vllm#44868; cuda_graph.py)
    "PN376",           # fp8 modules_to_not_convert substring match
                       # (vendor of OPEN vllm#44628; gated by dispatcher
                       # should_apply before import)
    "PN378",           # recovered-token vocab-pad -inf mask (vendor of
                       # OPEN vllm#45060 kernel half; rejection_sampler)
    "PN379",           # LoadConfig/DefaultModelLoader fail-fast
                       # (vendor of OPEN vllm#45196; atomic 2-file)
    "PN380",           # Qwen3.5/3.6 MTP pre-fused expert loader +
                       # coverage guard (vendor of OPEN vllm#44943)
    "PN381",           # allowed_token_ids spec-decode metadata
                       # hardening (vendor of OPEN vllm#44742)
    "PN382",           # DecodeBenchConnector hybrid per-block KV fill
                       # (vendor of OPEN vllm#45080; bench infra)
    "G4_81",           # TQ multi-query DIRECT decode routing (vllm#45144
                       # blueprint; runtime monkey-patch, no TextPatcher)
    "G4_82",           # TQ prefill SDPA fallback for head_dim>256 (Ampere
                       # FA2 256-cap, vllm#38887; runtime monkey-patch, no
                       # legacy hook — same class as G4_81)
    "PN383",           # KV-offload + MTP segfault gate (vendor of OPEN
                       # vllm#44784; multi-file text patch, no legacy hook)
    # ── 2026-06-13 50-PR sweep BATCH-2 WAVE 1 — five LIVE-bug vendors,
    # spec-driven from inception (apply_module + own apply(), no legacy
    # @register_patch hook; same class as PN383). All opt-in.
    "PN384",           # Eagle/MTP prefix-cache prefill fix (vendor of
                       # OPEN vllm#44986; kv_cache coordinator+manager)
    "PN385",           # forced-named empty-params tool schema ->
                       # JSON object (vendor of OPEN vllm#45290)
    "PN386",           # required-tool streaming brace string-awareness
                       # (vendor of OPEN vllm#45389; tool_parsers/streaming)
    "PN387",           # reject degenerate structured_outputs DoS guard
                       # (vendor of OPEN vllm#45346; serving + edge guard)
    "PN388",           # mamba-block-aligned intermediate prefill split
                       # (vendor of OPEN vllm#45477; scheduler, requires P34)
    # ── 2026-06-13 50-PR sweep BATCH-3 — four more spec-only-by-design
    # vendors (apply_module + own apply(), no legacy hook; same class as
    # PN383-PN388). All opt-in.
    "PN389",           # XGrammar grammar-compilation timeouts (vendor of
                       # OPEN vllm#45390; serving, 3-file overlay)
    "PN390",           # streaming-LSE rejection sampler (vendor of OPEN
                       # vllm#45369; spec_decode Triton kernel rewrite)
    "PN391",           # /health/decode forward-progress watchdog (vendor
                       # of OPEN vllm#45453; observability, 6-file overlay)
    "P89",             # reasoning_tokens in chat usage object (vendor of
                       # OPEN vllm#45471; serving, 2-file overlay)
    "PN392",           # qwen3_coder streaming tool-call coalescing
                       # (dev491 #45171 qwen3_xml->coder remap fix; runtime
                       # class-wrap, no legacy hook)
    # ── 2026-06-14 PR-sweep wave-1 implementation — spec-driven from
    # inception (apply_module + own apply(), no legacy @register_patch
    # hook; applied at legacy boot via _run_spec_only_supplement):
    "PN252",           # M-RoPE prompt_embeds-only DoS fix (vendor of
                       # vllm#45252 / GHSA-33cg-gxv8-3p8g; worker text
                       # patch on gpu_model_runner._init_mrope_positions,
                       # byte-verified dev259+dev491, security default_on)
    "PN517",           # init MemorySnapshot before NCCL (vendor of
                       # vllm#45517; worker text patch on
                       # gpu_worker.init_device, env default-off
                       # observability + asymmetric TP+PP OOM guard)
    # ── 2026-06-17 0.23.1 pin-bump: spec-driven from inception ──────────
    "PN398",           # async spec-decode accepted-counts race (backport of
                       # OPEN vllm#45100; gpu_model_runner + gdn_attn text
                       # patches, is_hybrid + >=0.23.0 gated, default-off
                       # defensive overlay, no legacy @register_patch hook)
    # ── 2026-06-19 dev148 TIER-1 audit: spec-driven from inception ──────
    "PN394",           # qwen3 partial-param value `<` truncation fix
                       # (backport of MERGED vllm#46047; single-line text
                       # patch on parser/qwen3.py, >=0.23.0 gated, default-on
                       # correctness fix, no legacy @register_patch hook)
    "PN399",           # TurboQuant decode-scratch fixed-buffer — fix CUDA IMA
                       # in FULL cudagraph (backport of OPEN vllm#46067; two-file
                       # text patch on turboquant_attn.py + gpu/shutdown.py,
                       # >=0.21.0,<0.24.0 gated, default-OFF experimental belt-
                       # and-suspenders; composes with PN118 / requires PN118,
                       # no legacy @register_patch hook)
    "PN400",           # restore is_sym qzeros guard for symmetric AutoRound/
                       # GPTQ Marlin MoE (backport of MERGED vllm#45656; fixes
                       # the vllm#43409 regression latent in dev148) — spec-
                       # driven from inception (apply_module + own apply(),
                       # no legacy @register_patch hook), default-OFF, pin-
                       # scoped to the pins lacking the native fix)
    # ── 2026-06-23 G4_85 LIVE re-target: spec-driven from inception ─────
    "G4_85",           # TurboMind tensor-core int4 grouped-MoE kernel re-
                       # targeted from the orphaned moe_wna16.MoeWNA16Method to
                       # the LIVE CompressedTensorsWNA16MoEMethod.apply (3-6x vs
                       # CUDA-core moe_wna16; fires only on Marlin-ineligible
                       # int4 MoE, gated on is_moe_model() + G4_84's marlin
                       # detector) — apply_module + own apply() returning
                       # (status, reason), no legacy @register_patch hook,
                       # default-OFF experimental, fail-open to the original)
    # ── 2026-06-25 dev424 TIER-1 backport: spec-driven from inception ───
    "PN401",           # TurboQuant prefill continuation guard — gate the
                       # flash_attn fast path with `not _has_continuation`
                       # so a co-batched continuation (q_len<seq_len) never
                       # drops its cached prefix K/V (backport+improve OPEN
                       # vllm#46461; single-anchor text patch on
                       # turboquant_attn.py _prefill_attention, >=0.23.0,
                       # <0.24.0 gated, default-OFF experimental correctness
                       # fix; conservative None-mirror fall-safe + length-
                       # clamp over the raw PR; composes with P101/PN116/
                       # PN399 on disjoint anchors, no legacy @register_patch
                       # hook)
    "PN402",           # sanitize invalid (-1 / over-vocab) MTP draft token
                       # ids before batch prep on the V1 gpu/model_runner
                       # path so a single bad draft cannot OOB-index the
                       # embedding gather and crash the engine with a CUDA
                       # IMA (backport+improve OPEN vllm#46574; 2-anchor text
                       # patch on gpu/model_runner.py execute_model + method
                       # inject, >=0.23.0,<0.24.0 gated, default-OFF
                       # experimental stability fix; gated on spec_config +
                       # flood-guarded WARNING + sndr_invalid_draft_tokens_
                       # dropped_total counter over the raw PR; composes with
                       # PN378/PN361/PN133, no legacy @register_patch hook)
    "PN518",           # INCConfig hybrid INT4+FP8 AutoRound latent trap-closer
                       # (vendor of OPEN vllm#46322). Injects a detect-and-WARN
                       # maybe_update_config onto INCConfig (dev424 is MISSING
                       # it -> inherits the base no-op), so a hybrid INT4+FP8
                       # auto-round checkpoint's FP8 layers are DIAGNOSED with a
                       # loud boot WARN instead of silently served as
                       # unquantized -> garbage. STRICT NO-OP when no FP8 layers
                       # present (the live 27B keeps linear_attn.in_proj at
                       # bits=16; 35B is fp8 not inc) -> get_quant_method
                       # unperturbed. Single-anchor text patch on inc/inc.py,
                       # >=0.23.0,<0.24.0 gated + autoround quant_format scoped,
                       # default-OFF latent guard, no legacy @register_patch
                       # hook; applied at legacy boot via
                       # _run_spec_only_supplement)
    "PN519",           # SWA/chunked KV-tile loop first_allowed_key base
                       # (backport+improve OPEN vllm#46087, fixes vllm#44575).
                       # compute_tile_loop_bounds returns tile_base + both
                       # triton_unified_attention consumers offset seq_offset so
                       # the SWA loop starts EXACTLY at first_allowed_key (drops
                       # the redundant boundary tile per Gemma4 SWA request +
                       # kills the residue-dependent online-softmax reduction
                       # non-determinism). Atomic 3-file text patch, USE_TD/3D
                       # keep tile_base=0, >=0.23.0,<0.24.0 gated, default-OFF
                       # experimental kernel_perf; no legacy @register_patch
                       # hook; applied at legacy boot via
                       # _run_spec_only_supplement. Runtime-inert on Qwen3.6
                       # (FlashInfer/FA2, head_dim=128).
})


# ─── Order extraction ─────────────────────────────────────────────────────


def _legacy_apply_names() -> list[str]:
    """Return the ordered list of registered apply-function names from
    `_per_patch_dispatch.py` (via `@register_patch`)."""
    # Force-import the parking lot module so the @register_patch
    # decorators run and populate `_state.PATCH_REGISTRY`.
    from sndr.apply import _per_patch_dispatch  # noqa: F401
    from sndr.apply._state import (
        PATCH_REGISTRY as APPLY_REGISTRY,
    )
    return [name for name, _fn in APPLY_REGISTRY]


# Legacy `@register_patch` names look like `"P67 TurboQuant ..."`. Extract
# the leading patch_id token so we can match them against spec patch_ids.
# Examples:
#   "P67 TurboQuant multi-query kernel"          → "P67"
#   "PN14 TQ decode IOOB safe_page_idx clamp"    → "PN14"
#   "P68/P69 long-ctx tool reminder"             → "P68"  (primary)
#   "P5b KV page-size pad-smaller-to-max"        → "P5b"
#   "G4_01 gemma4 Ampere FP8_BLOCK refusal guard" → "G4_01"
#   "G4_19b gemma4 TQ KV spec integration"       → "G4_19B"  (suffix uppercased)
#   "P23_WIRE Marlin FP32_REDUCE env wire"       → "P23_WIRE"  (underscore-suffix taxonomy)
#   "PN118_V2_MD5_WORKSPACE md5+full-file PoC"    → "PN118_V2_MD5_WORKSPACE"
#   "SNDR_EAGLE3_AUX_HIDDEN_001 EAGLE-3 prep"     → "SNDR_EAGLE3_AUX_HIDDEN_001"
# G4_NN[a-z]? prefix added 2026-05-22 (Phase 3A.1) after Phase 3 buckets
# 3/4 onboarded the G4 patch series into the legacy register table.
#
# Underscore-suffix taxonomy added 2026-06-14: a growing convention names
# patch ids with `_SUFFIX` tokens — fix-wires (`P23_WIRE`, `P29_HEAL`),
# kernel-literal tunes (`P18B_TEXT`), md5+full-file PoC scopes
# (`PN118_V2_MD5_WORKSPACE`, `PN79_V2_MD5_CHUNK_DELTA_H`), and `SNDR_`-prefix
# research backports (`SNDR_EAGLE3_AUX_HIDDEN_001`). The `(?:_[A-Za-z0-9]+)*`
# tail + `SNDR_` alternative lift these from the legacy register title. The
# closing `(?=[\s/]|$)` lookahead (replacing the old `\b`) anchors the match
# at a real token boundary — `\b` could not, because `_` is a word char so
# there is no boundary between `P23` and `_WIRE`. This was the exact cause of
# the 8 `legacy_unparseable` rows in `shadow --strict` before this fix. Safe
# by construction: the old `\b` regex already returned None for any name with
# `_` immediately after the id, so none of the 230 previously-parsing names
# carry an underscore tail for the new group to consume (it matches zero
# times for them); only the previously-None names change.
_PATCH_ID_LEAD = re.compile(
    r"^(P[Nn]?\d+[a-zA-Z]?(?:_[A-Za-z0-9]+)*"
    r"|G4_\d+[a-zA-Z]?(?:_[A-Za-z0-9]+)*"
    r"|SNDR_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)"
    r"(?=[\s/]|$)"
)

# UNIFIED_CONFIG 2026-05-10 — non-P/PN style legacy registrations.
# These are sprint/middleware names registered before patch_id taxonomy
# was extended. Map them explicitly to their canonical spec patch_id.
_LEGACY_NAME_TO_PATCH_ID: dict[str, str] = {
    "Sprint 2.6 v2 — CUDA graph dispatch trace wire-in": "PN122",  # renamed from SPRINT26_CG_DISPATCH_TRACE 2026-05-14
    # SNDR_WORKSPACE_001 starts with `SNDR_`, not `P` / `PN`, so the
    # leading-token regex above can't lift the patch id. Explicit map.
    "SNDR_WORKSPACE_001 workspace grow-after-lock graceful fix": "SNDR_WORKSPACE_001",
    # SNDR_MTP_DYNAMIC_K_001 — same situation as SNDR_WORKSPACE_001 above:
    # SNDR_*-prefix patch id (Phase 10.5 sweep 2026-06-01) does not match
    # the leading-token regex so the legacy register name needs an
    # explicit map entry. This is a community-tier backport of vllm#26504
    # (whytem's DynamicProposer adaptive-K MTP) ported to the
    # DraftModelProposer base used by qwen3.6-27B/35B assistant-model
    # MTP — closes shadow --strict's spec_only_unexpected +
    # legacy_unparseable warnings for this entry.
    "SNDR_MTP_DYNAMIC_K_001 adaptive K MTP proposer (vllm#26504 port to DraftModelProposer)": "SNDR_MTP_DYNAMIC_K_001",
    # PN-FP8MOE-KPAD — hyphenated patch id (FP8-core backport of vllm#45703).
    # The leading-token regex above only lifts `P[Nn]\d+...` / `G4_...` /
    # `SNDR_...` shapes; `PN-FP8MOE-KPAD` has a hyphen instead of a digit
    # after `PN`, so it needs an explicit map to bind the @register_patch
    # hook to its registry key (otherwise it shows as legacy_unparseable +
    # spec_only_unexpected in `shadow --strict`).
    "PN-FP8MOE-KPAD FP8 MoE intermediate thread-tile pad (FP8-core backport of OPEN vllm#45703)": "PN-FP8MOE-KPAD",
    # 2026-06-19: PN29 was consolidated into the PN298 registry entry (both
    # patch chunk_o.py at disjoint regions; one apply_module
    # pn29_pn298_chunk_o_consolidated). The legacy boot-log keeps a "PN29 ..."
    # @register_patch label for operator continuity; the leading-token regex
    # would lift "PN29" (no longer a spec id), so map it explicitly to the
    # merged PN298 spec — keeps shadow's legacy_only / spec_only clean.
    "PN29 GDN chunk_o scale-fold (vllm#41446 pattern (c) backport)": "PN298",
    # 2026-06-19: PN369 was consolidated into the P71 registry entry (both
    # patch rejection_sampler.py at disjoint regions; one apply_module
    # p71_pn369_rejection_sampler_consolidated). The legacy boot-log keeps a
    # "PN369 ..." @register_patch label for operator continuity; the
    # leading-token regex would lift "PN369" (no longer a spec id), so map it
    # explicitly to the merged P71 spec — keeps shadow's legacy_only /
    # spec_only clean.
    "PN369 Relaxed acceptance for MTP spec-decode (TRT-LLM-style top-K + delta window)": "P71",
    # 2026-06-20: P59 + PN51 were consolidated into the P61b registry entry,
    # and P61c + PN56 into the P64 entry (each trio patches one parser file at
    # disjoint regions; one apply_module per trio). The legacy boot-log keeps
    # the absorbed ids' @register_patch labels for operator continuity; the
    # leading-token regex would lift "P59"/"PN51"/"P61c"/"PN56" (no longer
    # spec ids), so map each explicitly to its merged primary — keeps shadow's
    # legacy_only / spec_only clean. (P61b and P64 labels still match their own
    # surviving spec ids, so they need no map entry.)
    "P59 Qwen3 reasoning embedded tool_call recovery": "P61b",
    "PN51 Qwen3 streaming `enable_thinking=false` content routing": "P61b",
    "P61c Qwen3Coder deferred-commit (club-3090#72)": "P64",
    "PN56 Qwen3Coder XML parse fallback (vllm#41466)": "P64",
}


def _patch_id_from_legacy_name(name: str) -> Optional[str]:
    # First check explicit map (non-P/PN style names)
    if name in _LEGACY_NAME_TO_PATCH_ID:
        return _LEGACY_NAME_TO_PATCH_ID[name]
    # Then leading P/PN/G4_ regex
    m = _PATCH_ID_LEAD.match(name)
    if not m:
        return None
    raw = m.group(1)
    # Normalize casing.
    if raw.startswith("SNDR_") or raw.startswith("sndr_"):
        # SNDR_-prefix ids are used verbatim in the spec registry (all-caps,
        # underscore-joined). No prefix/suffix casing transform — returning
        # the matched token as-is keeps it identical to the PatchSpec id.
        return raw
    if raw.lower().startswith("pn"):
        # PN-series: uppercase prefix, suffix letter preserved as-is.
        # Registry uses inconsistent suffix case (PN26b/PN96b lowercase
        # vs PN262B uppercase). Both forms are matched by the legacy
        # title's casing, so preserve.
        return "PN" + raw[2:]
    if raw.startswith("G4_") or raw.startswith("g4_"):
        # G4-series: prefix uppercased + suffix letter uppercased to match
        # spec registry shape (G4_19B uppercase suffix — Phase 3A.1, 2026-05-22).
        head = "G4_" + raw[3:]
        return head[:-1] + head[-1].upper() if head[-1].isalpha() else head
    # P-series default — uppercase prefix, suffix letter as-is.
    return "P" + raw[1:]


# ─── Diff result ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ApplyOrderDiff:
    """Structured diff between legacy and spec-driven apply orders."""
    legacy_count: int
    spec_count: int
    legacy_only: list[str] = field(default_factory=list)  # patch_ids in legacy not spec
    spec_only: list[str] = field(default_factory=list)    # patch_ids in spec not legacy (raw)
    spec_only_known: list[str] = field(default_factory=list)  # raw spec_only ∩ KNOWN_SPEC_ONLY
    spec_only_unexpected: list[str] = field(default_factory=list)  # raw spec_only \ KNOWN_SPEC_ONLY
    legacy_unparseable: list[str] = field(default_factory=list)  # legacy names we couldn't match to a patch_id
    # Patch ids that apply TODAY via a legacy @register_patch hook but have
    # NO apply_module — so `SNDR_APPLY_VIA_SPECS=1` (spec-only boot) would
    # SILENTLY DROP them. `legacy_only` does NOT catch this (it compares
    # against ALL spec ids, not apply_module-having ones). Surfaced as an
    # advisory, NOT a strict failure: these are legitimately legacy-only
    # implementations until migrated. Anyone considering flipping the boot
    # to the spec loop must migrate these to apply_module first. (Root-
    # caused 2026-06-14 — P1/P2, P17/P18, P32/P33 bundled hooks.)
    spec_boot_unsafe: list[str] = field(default_factory=list)
    spec_with_apply_module: int = 0
    spec_without_apply_module: int = 0

    @property
    def coverage_pct(self) -> float:
        """Fraction of specs whose apply_module is non-None."""
        if self.spec_count == 0:
            return 0.0
        return self.spec_with_apply_module / self.spec_count

    @property
    def is_clean(self) -> bool:
        """No UNEXPECTED mismatches and every legacy entry maps to a spec.

        P1-1 (audit 2026-05-08): "clean" no longer requires every spec
        to have apply_module — that's a separate coverage metric. It
        also no longer fails on KNOWN_SPEC_ONLY entries (intentional
        registry-driven additions). What remains:

          - no legacy_only (legacy parking lot must not have entries
            missing from registry)
          - no UNEXPECTED spec_only (new spec rows must be either in
            legacy or explicitly added to KNOWN_SPEC_ONLY_PATCHES)
          - no legacy_unparseable (every @register_patch name must
            resolve to a patch_id)
        """
        return (
            not self.legacy_only
            and not self.spec_only_unexpected
            and not self.legacy_unparseable
        )


# ─── Comparison logic ─────────────────────────────────────────────────────


def compare_apply_orders() -> ApplyOrderDiff:
    """Compute the diff between legacy and spec-driven apply orders.

    Pure function — no side effects on either registry. Safe to call
    in shadow mode during a real boot or off-line via the CLI.
    """
    from sndr.dispatcher.spec import iter_patch_specs

    # Legacy: list of names → set of (pid_or_None, name)
    legacy_names = _legacy_apply_names()
    legacy_pids: set[str] = set()
    legacy_unparseable: list[str] = []
    for n in legacy_names:
        pid = _patch_id_from_legacy_name(n)
        if pid is None:
            legacy_unparseable.append(n)
        else:
            legacy_pids.add(pid)

    # Spec-driven: iterate canonical specs
    specs = list(iter_patch_specs())
    spec_pids = {s.patch_id for s in specs}
    spec_with_module = sum(1 for s in specs if s.apply_module is not None)
    specs_without_module = {
        s.patch_id for s in specs if s.apply_module is None
    }

    legacy_only = sorted(legacy_pids - spec_pids)
    spec_only = sorted(spec_pids - legacy_pids)
    # P1-1: split spec_only into known-intentional vs unexpected.
    spec_only_known = sorted(set(spec_only) & KNOWN_SPEC_ONLY_PATCHES)
    spec_only_unexpected = sorted(set(spec_only) - KNOWN_SPEC_ONLY_PATCHES)

    # spec-boot-drop risk: a patch that applies via a legacy hook but whose
    # spec has no apply_module would vanish under SNDR_APPLY_VIA_SPECS=1.
    spec_boot_unsafe = sorted(legacy_pids & specs_without_module)

    return ApplyOrderDiff(
        legacy_count=len(legacy_names),
        spec_count=len(specs),
        legacy_only=legacy_only,
        spec_only=spec_only,
        spec_only_known=spec_only_known,
        spec_only_unexpected=spec_only_unexpected,
        legacy_unparseable=legacy_unparseable,
        spec_boot_unsafe=spec_boot_unsafe,
        spec_with_apply_module=spec_with_module,
        spec_without_apply_module=len(specs) - spec_with_module,
    )


# ─── Human-readable report ────────────────────────────────────────────────


def format_diff(diff: ApplyOrderDiff) -> str:
    """Multi-line human-readable summary of an `ApplyOrderDiff`."""
    lines = [
        "═══════════════════════════════════════════════════════════════",
        "  Genesis apply-loop shadow report  (PR38 Day 5)",
        "═══════════════════════════════════════════════════════════════",
        f"  Legacy apply registrations:  {diff.legacy_count:>4d} "
        "(_per_patch_dispatch.py @register_patch)",
        f"  Spec-driven entries:         {diff.spec_count:>4d} "
        "(dispatcher.PATCH_REGISTRY)",
        f"  Specs with apply_module:     {diff.spec_with_apply_module:>4d}"
        f"  ({diff.coverage_pct:.0%})",
        f"  Specs without apply_module:  {diff.spec_without_apply_module:>4d}",
    ]

    if diff.legacy_only:
        lines.append("")
        lines.append(f"  ⚠ legacy_only ({len(diff.legacy_only)}) — "
                     "registered in _per_patch_dispatch.py but no "
                     "matching dispatcher.PATCH_REGISTRY entry:")
        for pid in diff.legacy_only[:20]:
            lines.append(f"      - {pid}")
        if len(diff.legacy_only) > 20:
            lines.append(f"      ... and {len(diff.legacy_only) - 20} more")

    if diff.spec_only_known:
        lines.append("")
        lines.append(
            f"  ℹ spec_only_known ({len(diff.spec_only_known)}) — "
            "intentionally registry-driven only (P1-1 KNOWN_SPEC_ONLY):"
        )
        for pid in diff.spec_only_known:
            lines.append(f"      - {pid}")
    if diff.spec_only_unexpected:
        lines.append("")
        lines.append(
            f"  ⚠ spec_only_unexpected ({len(diff.spec_only_unexpected)}) — "
            "in dispatcher.PATCH_REGISTRY, no @register_patch, NOT in "
            "KNOWN_SPEC_ONLY_PATCHES allow-list:"
        )
        for pid in diff.spec_only_unexpected[:20]:
            lines.append(f"      - {pid}")
        if len(diff.spec_only_unexpected) > 20:
            lines.append(f"      ... and "
                         f"{len(diff.spec_only_unexpected) - 20} more")

    if diff.legacy_unparseable:
        lines.append("")
        lines.append(
            f"  ⚠ legacy_unparseable ({len(diff.legacy_unparseable)}) — "
            "registered apply names whose patch_id couldn't be parsed:"
        )
        for n in diff.legacy_unparseable[:5]:
            lines.append(f"      - {n!r}")

    if diff.spec_boot_unsafe:
        lines.append("")
        lines.append(
            f"  ⚠ spec_boot_unsafe ({len(diff.spec_boot_unsafe)}) — apply "
            "via a legacy @register_patch hook but NO apply_module, so "
            "SNDR_APPLY_VIA_SPECS=1 would SILENTLY DROP them. Migrate to "
            "apply_module before making the spec loop the default boot:"
        )
        for pid in diff.spec_boot_unsafe:
            lines.append(f"      - {pid}")

    lines.append("")
    if diff.is_clean:
        lines.append(
            "  ✓ CLEAN — no unexpected divergence "
            f"(known spec-only: {len(diff.spec_only_known)})"
        )
    else:
        lines.append("  ⚠ DIVERGENT — see lists above")
    lines.append("═══════════════════════════════════════════════════════════════")
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    """`python -m sndr.apply.shadow` entry point."""
    parser = argparse.ArgumentParser(
        description="Shadow comparison: PatchSpec apply order vs legacy"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if any divergence found",
    )
    args = parser.parse_args(argv)

    diff = compare_apply_orders()
    print(format_diff(diff))
    if args.strict and not diff.is_clean:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
