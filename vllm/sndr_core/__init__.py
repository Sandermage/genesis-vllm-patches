# SPDX-License-Identifier: Apache-2.0
"""SNDR Core — canonical infrastructure for vllm patcher (community tier).

Public-facing brand: "Genesis (powered by SNDR Core)".

This package is the canonical home for:
  - Patcher infrastructure (TextPatcher, MultiFilePatchTransaction, manifest)
  - Upstream backports (community-tier patches; keep "Genesis" marker prefix)
  - Centralized engine paths + env flag registry
  - CLI installer + first-run launcher

Sander-original kernels and advanced features live in `vllm.sndr_engine/`
(commercial tier) — separate pip-installable package.

Brand decision (Sander 2026-05-07, Q2 mixed):
  - Backport patches (from upstream PR vllm/SGLang/llama.cpp) keep
    `Genesis ` marker prefix + `GENESIS_ENABLE_*` env vars. They are
    community-tier work, retain community branding.
  - Sander-original patches use `SNDR ` marker prefix + `SNDR_ENABLE_*`
    env vars. These are the canonical brand for new Sander-IP work.
  - Patcher Layer-1 marker check recognizes BOTH prefixes (text_patch.py).

Migration status (started 2026-05-07, etap 1):
  Stage 1 (CURRENT) — skeleton only. All code still lives in vllm/_genesis/.
  Stages 2-13      — progressive migration; modules move INTO sndr_core/.
  Final            — vllm/_genesis/ becomes thin forward-alias of sndr_core.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from .version import SNDR_CORE_VERSION, GENESIS_VERSION  # noqa: F401
from . import brand  # noqa: F401
from . import env  # noqa: F401
from . import locations  # noqa: F401  (renamed from `paths` 2026-05-11 — audit P-01 P-02)
from . import detection  # noqa: F401  (Stage 4)
# `runtime`, `integrations`, `bundles`, `cli` are loaded lazily via __getattr__.
#
# (1) Torch-less import contract: `runtime.prealloc` imports torch at module
#     top-level. CLI / schema validator / doctor / registry audit / pre-commit
#     must work in environments without torch — eager `from . import runtime`
#     would break them. The legacy `vllm/_genesis/__init__.py` documents the
#     same trap (v7 G-002 fix).
#
# (2) Circular-import contract: `integrations` and `bundles` import back into
#     `vllm._genesis.wiring` (forward shims) — eager init at this point
#     would create a cycle with sndr_core still initializing.
from .env import (  # noqa: F401
    Flags, is_enabled, is_disabled, is_legacy_active, is_meta_flag,
)
from .locations import (  # noqa: F401  (renamed from `paths` 2026-05-11)
    vllm_targets,
    resolve_vllm_file,
    vllm_install_root,
)


def __getattr__(name: str):
    """Lazy-load `runtime`, `integrations`, `bundles`, `cli`.

    `runtime` is lazy to keep torch out of the cold-import path.
    `integrations` / `bundles` / `cli` are lazy to avoid circular imports
    with vllm._genesis.wiring (which loads forward shims on init).

    Back-compat aliases (`patches` → `integrations`, `paths` → `locations`)
    are kept for one major version to ease the transition.
    """
    if name in ("runtime", "integrations", "bundles", "cli"):
        import importlib
        return importlib.import_module(f"vllm.sndr_core.{name}")
    # Back-compat aliases: old `patches`/`paths` names still resolve to the
    # new modules (transition path; can be removed in a future major).
    if name == "patches":
        import importlib
        return importlib.import_module("vllm.sndr_core.integrations")
    if name == "paths":
        import importlib
        return importlib.import_module("vllm.sndr_core.locations")
    raise AttributeError(f"module 'vllm.sndr_core' has no attribute {name!r}")

__version__ = SNDR_CORE_VERSION


# ─── Import-time selective apply for G4_19/G4_19b ─────────────────────
#
# vLLM v1 EngineCore is a separate spawn'd subprocess that doesn't reliably
# call our plugin.register() (vllm.general_plugins discovery is incomplete
# in V1 multiproc). To ensure G4_19/G4_19b monkey-patches apply in BOTH
# parent APIServer AND EngineCore subprocess, we hook them at import-time.
#
# Triggers: vllm.sndr_core is imported by EngineCore via plugin discovery
# OR via vllm_config unpickling that touches sndr_core types. The import
# runs this __init__.py which applies G4_19b BEFORE
# _check_enough_kv_cache_memory is called by vllm.v1.core.kv_cache_utils.
#
# Safety: ONLY runs when explicit env flag is set. Operators who don't
# enable G4_19/G4_19b see zero behavior change vs prior import semantics.
#
# Idempotent: each patch has its own _APPLIED guard so double-apply is a no-op.
def _g4_19_import_time_hook():
    """Apply G4_19/G4_19b/G4_19c/G4_30 at any vllm.sndr_core import (parent + subprocess)."""
    import os as _os
    g19 = _os.environ.get(
        "GENESIS_ENABLE_G4_19_GEMMA4_TURBOQUANT_KV", ""
    ).strip().lower() in ("1", "true", "yes")
    g19b = _os.environ.get(
        "GENESIS_ENABLE_G4_19B_GEMMA4_TQ_KV_SPEC", ""
    ).strip().lower() in ("1", "true", "yes")
    g19c = _os.environ.get(
        "GENESIS_ENABLE_G4_19C_ATTN_WRAP", ""
    ).strip().lower() in ("1", "true", "yes")
    g30 = _os.environ.get(
        "GENESIS_ENABLE_G4_30_TQ_UNBLOCK", ""
    ).strip().lower() in ("1", "true", "yes")
    g31 = _os.environ.get(
        "GENESIS_ENABLE_G4_31_TQ_DTYPE_PRESERVE", ""
    ).strip().lower() in ("1", "true", "yes")
    g32 = _os.environ.get(
        "GENESIS_ENABLE_G4_32_TQ_VALIDATION_BYPASS", ""
    ).strip().lower() in ("1", "true", "yes")
    g43 = _os.environ.get(
        "GENESIS_ENABLE_G4_43_UNBLOCK_TRITON_FORCE", ""
    ).strip().lower() in ("1", "true", "yes")
    g44 = _os.environ.get(
        "GENESIS_ENABLE_G4_44_TQ_HEAD_DIM_512", ""
    ).strip().lower() in ("1", "true", "yes")
    g45 = (
        _os.environ.get("GENESIS_ENABLE_G4_45_UNIFY_DIAG", "").strip().lower()
        in ("1", "true", "yes")
    ) or (
        _os.environ.get("GENESIS_ENABLE_G4_45_UNIFY_FIX", "").strip().lower()
        in ("1", "true", "yes")
    )
    g50 = _os.environ.get(
        "GENESIS_ENABLE_G4_50_NATIVE_TQ", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_60a..k — PR #42637 cherry-pick stack (Mixed-attention TQ for Gemma 4)
    g60a = _os.environ.get(
        "GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC", ""
    ).strip().lower() in ("1", "true", "yes")
    g60e = _os.environ.get(
        "GENESIS_ENABLE_G4_60E_KV_CACHE_UTILS", ""
    ).strip().lower() in ("1", "true", "yes")
    g60g = _os.environ.get(
        "GENESIS_ENABLE_G4_60G_TQ_DISPATCH", ""
    ).strip().lower() in ("1", "true", "yes")
    g60h = _os.environ.get(
        "GENESIS_ENABLE_G4_60H_TQ_CONFIG_AUGMENT", ""
    ).strip().lower() in ("1", "true", "yes")
    g60k = _os.environ.get(
        "GENESIS_ENABLE_G4_60K_TQ_ENGINE_CONFIG", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_61 + G4_62 — PR #40798 (workspace share) + PR #42215 (decode warmup).
    g61 = _os.environ.get(
        "GENESIS_ENABLE_G4_61_TQ_SHARED_WORKSPACE", ""
    ).strip().lower() in ("1", "true", "yes")
    g62 = _os.environ.get(
        "GENESIS_ENABLE_G4_62_TQ_KERNEL_WARMUP", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_60b/c/d — verify bind-mount overlay of PR #42637 source files.
    g60b = _os.environ.get(
        "GENESIS_ENABLE_G4_60B_TQ_ATTN_OVERLAY", ""
    ).strip().lower() in ("1", "true", "yes")
    g60c = _os.environ.get(
        "GENESIS_ENABLE_G4_60C_TQ_DECODE_OVERLAY", ""
    ).strip().lower() in ("1", "true", "yes")
    g60d = _os.environ.get(
        "GENESIS_ENABLE_G4_60D_TQ_STORE_OVERLAY", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_60L — monkey-patch supports_mm_prefix=True on stock
    # TurboQuantBackend (Phase 7.G4.31B.K4-TURBOQUANT-BACKEND-MM-PREFIX).
    # Closes the 'partial multimodal token full attention not
    # supported' rejection for Gemma 4 31B AWQ + TURBOQUANT on the
    # V2 compose path, which doesn't bind-mount the PR42637 overlay.
    g60l = _os.environ.get(
        "GENESIS_ENABLE_G4_60L_TQ_BACKEND_MM_PREFIX", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_67 — backport of upstream PR #40914 (TQ K+1 spec-verify routing).
    g67 = _os.environ.get(
        "GENESIS_ENABLE_G4_67_TQ_SPEC_VERIFY_ROUTE", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_68 — verifier for P65 v2 cudagraph downgrade inlined into the
    # PR #42637 overlay (companion to PN256 raw-K/V continuation).
    g68 = _os.environ.get(
        "GENESIS_ENABLE_G4_68_TQ_SPEC_CG_DOWNGRADE_OVERLAY", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_69 — per-layer native attention backend dispatch for skip-listed
    # TurboQuant layers (unblocks GENESIS_G4_TQ_FORCE_SKIP_LAYERS by
    # rerouting kv_cache_dtype='auto' layers away from forced TURBOQUANT
    # backend selection).
    g69 = _os.environ.get(
        "GENESIS_ENABLE_G4_69_SKIP_LAYERS_NATIVE_BACKEND", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_71 — force FlashAttn backend for Gemma 4 MTP drafter layers
    # (prefix "draft_model.") to prevent TurboQuant impl on native KV cache.
    g71 = _os.environ.get(
        "GENESIS_ENABLE_G4_71_DRAFTER_NATIVE_BACKEND", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_72 — force native FullAttentionSpec/SlidingWindowSpec for drafter
    # layers (G4_71 marker). Companion to G4_71 — fixes spec/impl mismatch
    # that caused FlashAttn unbind(0) ValueError after G4_71 alone (PN261-D).
    g72 = _os.environ.get(
        "GENESIS_ENABLE_G4_72_DRAFTER_NATIVE_SPEC", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN241 — Gemma4MTPAttention.forward finite/norm trace.
    pn241 = _os.environ.get(
        "GENESIS_ENABLE_PN241_MTP_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN248 — Acceptance trace via rejection_sample wrap (Step 4 from
    # corrected plan).
    pn248 = _os.environ.get(
        "GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN282 — Spec-decode acceptance proxy METRIC via rejection_sample
    # wrap (production sibling of PN248's log trace). Increments
    # sndr_spec_decode_accepted_per_call_total{k,profile} +
    # sndr_spec_decode_calls_total{profile} + sndr_spec_decode_max_spec_len
    # on the worker's existing /metrics endpoint. Designed for
    # prod-permanent enable on the structured upstream. Canonical env
    # flag is SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC; legacy alias
    # GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC accepted with
    # deprecation warning. Coexists with PN248 (independent idempotency
    # markers, separate wrap layers).
    pn282 = _os.environ.get(
        "SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC", ""
    ).strip().lower() in ("1", "true", "yes") or _os.environ.get(
        "GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN258 — Oracle acceptance test (record/replay around rejection_sample).
    # First technical step of the post-PN257a cycle: separates "verifier
    # rows 1..K still broken" vs "drafter is the only remaining problem".
    pn258 = _os.environ.get(
        "GENESIS_ENABLE_PN258_ORACLE_ACCEPTANCE", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN262 — FlashAttn drafter KV cache shape/stride trace + fail-fast.
    # PN261-D follow-up: G4_71+G4_72 fixed impl+spec but K=2 still crashed
    # at flash_attn.py:744 kv_cache.unbind(0). Localizes which of 4
    # hypotheses (allocator / bind-view / sharing / global layout)
    # causes the wrong leading axis.
    pn262 = _os.environ.get(
        "GENESIS_ENABLE_PN262_FLASH_ATTN_DRAFTER_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN262-B — KV cache allocator/reshape/proposer-init deep-dive trace.
    # PN262-A nailed the wrong shape to allocator path; PN262-B finds
    # which of (group backend / group spec / proposer per-layer spec
    # lookup) produces the (8192, 8, 256) tensor.
    pn262b = _os.environ.get(
        "GENESIS_ENABLE_PN262B_KV_ALLOC_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_73 — skip drafter.dummy_run during profile_run (PN262-D pragmatic).
    # Unblocks K=2 boot when drafter's profile-time KV cache placeholder
    # is sized by the GROUP's TQ backend instead of FlashAttn (per-layer
    # impl). Runtime path uses initialize_attn_backend sub-grouping by
    # per-layer backend, so runtime should be correct.
    g73 = _os.environ.get(
        "GENESIS_ENABLE_G4_73_DRAFTER_PROFILE_SKIP", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_74 — drafter HND layout enforcement post-reshape (PN263 fix).
    # PN262 with fixed args[4] index revealed drafter kv_cache shape is
    # NHD (num_blocks, 2, ...). FlashAttn expects HND (2, num_blocks, ...).
    # G4_74 wraps _reshape_kv_cache_tensors and, for drafter layers,
    # transposes axes 0 and 1 + .contiguous() before bind_kv_cache.
    g74 = _os.environ.get(
        "GENESIS_ENABLE_G4_74_DRAFTER_HND_LAYOUT", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_75 — per-layer drafter backend split (PN264). Route drafter
    # layers with head_size==threshold (default 512) to TRITON_ATTN
    # instead of FlashAttn (which caps head_size at 256). Triton uses
    # NHD natively, so G4_74 skips Triton-marked drafter layers.
    g75 = _os.environ.get(
        "GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_76 — disable Gemma4Proposer._setup_gemma4_kv_sharing (PN265).
    # Make drafter fully independent (own kv_cache + block_table +
    # slot_mapping) so target's block ids don't OOB drafter's smaller
    # cache. Required after G4_74 broke the physical alias.
    g76 = _os.environ.get(
        "GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN266 — SpecDecodeBaseProposer.propose() input-shape trace
    # (G4_77 design probe — does propose receive prompt context?).
    pn266 = _os.environ.get(
        "GENESIS_ENABLE_PN266_PROPOSE_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN267 — TQ→native K/V bridge feasibility trace (G4_78-0 probe).
    # Captures target[58]/[59] kv_cache shapes + drafter forward args.
    pn267 = _os.environ.get(
        "GENESIS_ENABLE_PN267_KV_BRIDGE_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN268 — Drafter num_blocks origin trace (G4_78-CAP-0 probe).
    # Wraps initialize_kv_cache_tensors and logs every call:
    # kv_cache_config.kv_cache_groups + per-drafter post-bind shape.
    pn268 = _os.environ.get(
        "GENESIS_ENABLE_PN268_DRAFTER_BLOCKS_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN269 — G4_78-A0 block_table accessibility trace.
    # Wraps FlashAttn + Triton attention forward, captures target[58]/[59]
    # block_table/slot_mapping/positions, cross-references drafter[0..3]
    # forward to determine if target's block_table is reachable for KV
    # bridging. Strict prompt-prefill-first design (K=1, single prompt).
    pn269 = _os.environ.get(
        "GENESIS_ENABLE_PN269_A0_BLOCK_TABLE_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_78 — target[58] -> drafter[0..2] K/V bridge (v1: prompt prefill,
    # K=1, no decode). Based on PN269 A0 verdict.
    g78 = _os.environ.get(
        "GENESIS_ENABLE_G4_78_DRAFTER_TARGET_KV_BRIDGE", ""
    ).strip().lower() in ("1", "true", "yes")
    # G4_71b — drafter sliding (head_size=256) -> TRITON_ATTN + kv_cache
    # _dtype='auto' reset. β′-A enabler: clean kv_sharing contract.
    g71b = _os.environ.get(
        "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN270 — drafter K/V projection audit (read-only). Settles whether
    # drafter's k_proj/v_proj exist + differ. Triggered by G4_78 v1
    # finding that drafter K stats == V stats (suspicious tied/missing).
    pn270 = _os.environ.get(
        "GENESIS_ENABLE_PN270_DRAFTER_KV_PROJ_AUDIT", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN271 — SpecDecode KV contract audit (model-agnostic, read-only).
    # Cross-checks attention numerics, RoPE, soft-cap, scale, layout,
    # dtype, quantization between drafter and target layer pairs.
    # Outputs per-pair + overall compatibility verdict.
    pn271 = _os.environ.get(
        "GENESIS_ENABLE_PN271_KV_CONTRACT_AUDIT", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN272 — Gemma 4 MTP drafter input semantics probe (read-only).
    # Inventories pre_projection / post_projection / norm / layer_scalar
    # / embed_tokens / lm_head + per-layer norms, and captures one-shot
    # runtime tensor stats through Gemma4MultiTokenPredictor.forward.
    pn272 = _os.environ.get(
        "GENESIS_ENABLE_PN272_GEMMA4_DRAFTER_INPUT_PROBE", ""
    ).strip().lower() in ("1", "true", "yes")
    if not (
        g19 or g19b or g19c or g30 or g31 or g32 or g43 or g44 or g45 or g50
        or g60a or g60b or g60c or g60d or g60e or g60g or g60h or g60k or g60l
        or g61 or g62 or g67 or g68 or g69 or g71 or g72
        or pn241 or pn248 or pn282 or pn258 or pn262 or pn262b
        or g73 or g74 or g75 or g76 or pn266 or pn267 or pn268 or pn269
        or g78 or pn270 or pn271 or pn272 or g71b
    ):
        return
    try:
        # G4_30/G4_43/G4_44/G4_45/G4_50 relocated to integrations/_retired/g4_upstream_tq_wip/
        # 2026-05-24 (P0.1 M.3b/c) — superseded by upstream PR #42637 (Mixed-attention
        # TQ for Gemma 4). Their env flags still resolve via _retired/ fallback when
        # explicitly enabled; will fully retire after PR #42637 cherry-pick lands.
        # Previous location vllm/sndr_core/sndr_private/g4_upstream_tq_wip/ was an
        # architectural error (packaged core must not contain sndr_private namespace);
        # see hard rule #27 + P0_PROJECT_STRUCTURE_R deliverable.
        if g30 or g43 or g44 or g45 or g50:
            try:
                if g30:
                    from .integrations._retired.g4_upstream_tq_wip import (
                        g4_30_upstream_tq_unblock as _m,
                    )
                    _m.apply()
                if g43:
                    from .integrations._retired.g4_upstream_tq_wip import (
                        g4_43_unblock_forced_triton as _m,
                    )
                    _m.apply()
                if g44:
                    from .integrations._retired.g4_upstream_tq_wip import (
                        g4_44_tq_head_dim_512_prefill as _m,
                    )
                    _m.apply()
                if g45:
                    from .integrations._retired.g4_upstream_tq_wip import (
                        g4_45_unify_page_diag as _m,
                    )
                    _m.apply()
                if g50:
                    from .integrations._retired.g4_upstream_tq_wip import (
                        g4_50_genesis_native_backend as _m,
                    )
                    _m.apply()
            except ImportError:
                # _retired/ module may be absent in slim distributions; silent no-op
                pass
        # G4_31 active in main — preserves turboquant_* dtype from AWQ override.
        if g31:
            from .integrations.attention.turboquant import (
                g4_31_preserve_tq_dtype as _g4_31_mod,
            )
            _g4_31_mod.apply()
        # G4_32 active in main — bypasses TQ validate_configuration.
        if g32:
            from .integrations.attention.turboquant import (
                g4_32_tq_validation_bypass as _g4_32_mod,
            )
            _g4_32_mod.apply()
        if g19b:
            from .integrations.attention.turboquant import (
                g4_19b_tq_kv_spec_integration as _g4_19b_mod,
            )
            _g4_19b_mod.apply()
        if g19:
            # Pre-import gemma4 so Gemma4Config exists for the wrapper
            try:
                import vllm.model_executor.models.gemma4  # noqa: F401
            except ImportError:
                pass
            from .integrations.attention.turboquant import (
                g4_19_turboquant_kv_cache as _g4_19_mod,
            )
            _g4_19_mod.apply()
        if g19c:
            # G4_19c depends on G4_19 having published the config registry,
            # so apply order matters: G4_19 first, G4_19c second.
            try:
                import vllm.model_executor.models.gemma4  # noqa: F401
            except ImportError:
                pass
            from .integrations.attention.turboquant import (
                g4_19c_attention_wrapper as _g4_19c_mod,
            )
            _g4_19c_mod.apply()
        # === G4_60* — PR #42637 cherry-pick stack (apply in dependency order) ===
        # G4_60a (TQSlidingWindowSpec class injection) is prerequisite for
        # G4_60e (kv_cache_utils mixed-route) and G4_60g (dispatch). Apply first.
        if g60a:
            from .integrations.attention.turboquant import (
                g4_60a_tq_sliding_window_spec as _g4_60a_mod,
            )
            _g4_60a_mod.apply()
        # G4_60h augments TurboQuantConfig — required by G4_60k.
        if g60h:
            from .integrations.attention.turboquant import (
                g4_60h_turboquant_config_augment as _g4_60h_mod,
            )
            _g4_60h_mod.apply()
        # G4_60e patches kv_cache_utils — needs G4_60a first.
        if g60e:
            from .integrations.attention.turboquant import (
                g4_60e_kv_cache_utils as _g4_60e_mod,
            )
            _g4_60e_mod.apply()
        # G4_60g patches Attention.get_kv_cache_spec — needs G4_60a first.
        if g60g:
            from .integrations.attention.turboquant import (
                g4_60g_attention_dispatch as _g4_60g_mod,
            )
            _g4_60g_mod.apply()
        # G4_60k wraps EngineArgs.create_engine_config — applies post-build.
        # Order is independent but typically after G4_60h.
        if g60k:
            from .integrations.attention.turboquant import (
                g4_60k_arg_utils as _g4_60k_mod,
            )
            _g4_60k_mod.apply()
        # G4_61 shares TQ decode workspace across layers (PR #40798).
        # Apply before any model forward — patches launcher + capture_model.
        if g61:
            from .integrations.attention.turboquant import (
                g4_61_tq_shared_workspace as _g4_61_mod,
            )
            _g4_61_mod.apply()
        # G4_62 warms up TQ decode kernels before lock_workspace (PR #42215).
        # Apply order: G4_62 after G4_61 so warmup uses shared workspace path.
        if g62:
            from .integrations.attention.turboquant import (
                g4_62_tq_kernel_warmup as _g4_62_mod,
            )
            _g4_62_mod.apply()
        # G4_60b/c/d verify PR #42637 bind-mount overlay activated. These are
        # diagnostic only — they don't perform the mount (that's done in the
        # launch script via docker -v flags). Apply after all monkey-patches
        # so verification reflects the final live state.
        if g60b:
            from .integrations.attention.turboquant import (
                g4_60b_turboquant_attn_overlay_loader as _g4_60b_mod,
            )
            _g4_60b_mod.apply()
        if g60c:
            from .integrations.attention.turboquant import (
                g4_60c_triton_decode_overlay_loader as _g4_60c_mod,
            )
            _g4_60c_mod.apply()
        if g60d:
            from .integrations.attention.turboquant import (
                g4_60d_triton_store_overlay_loader as _g4_60d_mod,
            )
            _g4_60d_mod.apply()
        # G4_60L — supports_mm_prefix=True monkey-patch on stock
        # TurboQuantBackend. Must apply BEFORE Attention.__init__ on
        # any Gemma 4 layer (i.e. before model load). Plugin apply-all
        # runs at engine_core boot, well before model load, so this
        # ordering is satisfied. Idempotent: if G4_60B already
        # confirmed bind-mounted overlay, the override is already
        # present and this no-ops.
        if g60l:
            from .integrations.attention.turboquant import (
                g4_60l_tq_backend_mm_prefix as _g4_60l_mod,
            )
            _g4_60l_mod.apply()
        # G4_67 backports PR #40914 — must apply AFTER G4_60b (overlay
        # verifier) so TurboQuantAttentionImpl exists with PR #42637
        # signatures before we monkey-patch its forward method.
        if g67:
            from .integrations.attention.turboquant import (
                g4_67_tq_spec_verify_routing as _g4_67_mod,
            )
            _g4_67_mod.apply()
        # G4_68 — verifier for inlined P65 v2 cudagraph downgrade in the
        # PR #42637 overlay's TurboQuantMetadataBuilder. Must apply AFTER
        # G4_60b so TurboQuantMetadataBuilder is imported from the
        # overlay. Reports applied/error/skipped; no monkey-patching.
        if g68:
            from .integrations.attention.turboquant import (
                g4_68_tq_spec_cg_downgrade_overlay as _g4_68_mod,
            )
            _g4_68_mod.apply()
        # G4_69 — per-layer native attention backend dispatch for
        # skip-listed TurboQuant layers. Must apply BEFORE the first
        # Attention.__init__ call (i.e., before model load). Plugin
        # registration runs at engine_core boot, ahead of model load,
        # so this ordering is satisfied by being part of the apply-all
        # sweep.
        if g69:
            from .integrations.attention.turboquant import (
                g4_69_skip_layers_native_backend as _g4_69_mod,
            )
            _g4_69_mod.apply()
        # G4_71 — force FlashAttn backend for drafter Attention layers.
        # Must apply BEFORE drafter model loads (i.e. before
        # LLMBaseProposer.load_model runs). Plugin apply-all happens at
        # engine_core boot, well before model load, so order is safe.
        if g71:
            from .integrations.spec_decode import (
                g4_71_drafter_native_attn_backend as _g4_71_mod,
            )
            _g4_71_mod.apply()
        # G4_72 — force native FullAttentionSpec/SlidingWindowSpec for
        # drafter Attention layers (companion to G4_71). MUST apply AFTER
        # G4_60g so its wrap sits on top — non-drafter layers continue
        # through G4_60g's TQ-first dispatch; only drafter (G4_71 marker)
        # is rerouted to native spec.
        if g72:
            from .integrations.spec_decode import (
                g4_72_drafter_native_kv_cache_spec as _g4_72_mod,
            )
            _g4_72_mod.apply()
        # PN241 — finite/norm trace at SpecDecodeBaseProposer
        # boundary (Python orchestration above torch.compile boundary).
        # Logs target_hidden_states (input) + draft_token_ids (output) per
        # propose() call.
        if pn241:
            try:
                import vllm.v1.spec_decode.llm_base_proposer  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn241_mtp_trace as _pn241_mod,
            )
            _pn241_mod.apply()
        # PN248 — Step 4 from corrected plan: per-step acceptance trace via
        # rejection_sample wrap. Logs draft_token_ids, target_argmax, and
        # output_token_ids (accept/reject mask) per call. Direct test for
        # Hypothesis D (cross-quantization verifier loop).
        if pn248:
            try:
                import vllm.v1.sample.rejection_sampler  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn248_acceptance_trace as _pn248_mod,
            )
            _pn248_mod.apply()
        # PN282 — Production sibling of PN248. Wraps the same
        # rejection_sample but emits Prometheus counters instead of a
        # log file. Apply AFTER PN248 so PN248's log writer is the
        # outermost layer; PN282's metric increment runs inside PN248's
        # wrap. Either ordering is functionally safe (both wraps are
        # try/except-guarded) but this order keeps the trace authoritative
        # when both are on. Coexistence enforced by independent
        # idempotency markers on the wrapped callable.
        if pn282:
            try:
                import vllm.v1.sample.rejection_sampler  # noqa: F401
            except ImportError:
                pass
            from .integrations.observability import (
                pn282_spec_decode_acceptance_metric as _pn282_mod,
            )
            _pn282_mod.apply()
        # PN258 — Oracle acceptance test. Record/replay around
        # rejection_sample: pass 1 captures target's greedy continuation
        # to /tmp/genesis_pn258_oracle.txt, pass 2 injects those tokens
        # as the drafter's drafts. If accepted_per_req ~= K under
        # injection: target verifier rows 0..K-1 are all correct and the
        # remaining problem is in the drafter (Outcome A → H8). If
        # accepted_per_req stays 0 under injection: rows 1..K-1 / sampler
        # / position mapping are still broken (Outcome B → fix verifier).
        # Apply AFTER PN248 so the PN258 wrap is on the outside; both
        # logs co-exist.
        if pn258:
            try:
                import vllm.v1.sample.rejection_sampler  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn258_oracle_acceptance as _pn258_mod,
            )
            _pn258_mod.apply()
        # PN262 — FlashAttn drafter KV cache shape/stride trace +
        # fail-fast (PN261-D D-3 localization). Wraps
        # FlashAttentionImpl.forward; only fires for prefix
        # 'draft_model.'. Should apply AFTER G4_71 (which sets the
        # FlashAttn impl on drafter) — order satisfied by being last
        # in the apply sweep.
        if pn262:
            try:
                import vllm.v1.attention.backends.flash_attn  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn262_flash_attn_drafter_trace as _pn262_mod,
            )
            _pn262_mod.apply()
        # PN262-B — KV cache allocator/reshape/proposer-init trace.
        # Wraps GPUModelRunner._reshape_kv_cache_tensors and
        # LLMBaseProposer.initialize_attn_backend with diagnostic logs.
        if pn262b:
            try:
                import vllm.v1.worker.gpu_model_runner  # noqa: F401
                import vllm.v1.spec_decode.llm_base_proposer  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn262b_kv_alloc_trace as _pn262b_mod,
            )
            _pn262b_mod.apply()
        # G4_73 — skip drafter.dummy_run during profile_run.
        # Pragmatic minimal fix for PN262-D: unblocks K=2 boot by skipping
        # drafter's profile-time forward, which would otherwise hit FlashAttn
        # with a TQ-shaped KV cache placeholder. Runtime sub-grouping after
        # initialize_kv_cache(real_config) honors per-layer attn_backend,
        # so runtime should be correct.
        if g73:
            try:
                import vllm.v1.worker.gpu_model_runner  # noqa: F401
                import vllm.v1.spec_decode.llm_base_proposer  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode import (
                g4_73_drafter_profile_skip as _g4_73_mod,
            )
            _g4_73_mod.apply()
        # G4_74 — drafter HND layout enforcement post-reshape (PN263 fix).
        # Must apply AFTER G4_73 (which already wraps GPUModelRunner._dummy_run);
        # G4_74 wraps a separate method (_reshape_kv_cache_tensors) so they
        # compose cleanly. The wrap on _reshape_kv_cache_tensors fires once
        # per initialize_kv_cache call; for each drafter layer in the returned
        # kv_caches dict, if shape is NHD (num_blocks, 2, ...) it transposes
        # axes 0,1 + .contiguous() to give HND (2, num_blocks, ...) before
        # bind_kv_cache stores the tensor in the static forward context.
        if g74:
            try:
                import vllm.v1.worker.gpu_model_runner  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode import (
                g4_74_drafter_hnd_layout as _g4_74_mod,
            )
            _g4_74_mod.apply()
        # G4_75 — drafter head_size==512 -> TRITON_ATTN reroute (PN264).
        # Must apply AFTER G4_71 so that its kwargs["attn_backend"] override
        # preempts G4_71's FlashAttn assignment. Also stamps a marker on
        # the Attention instance so G4_74 skips HND conversion for Triton
        # drafter layers (Triton uses NHD natively).
        if g75:
            from .integrations.spec_decode import (
                g4_75_drafter_head512_triton as _g4_75_mod,
            )
            _g4_75_mod.apply()
        # G4_71b — drafter sliding (head_size=256) -> TRITON_ATTN +
        # kv_cache_dtype='auto' reset. β′-A enabler. Apply AFTER G4_75
        # so both share a uniform Triton routing across all 4 drafter
        # layers when both enabled.
        if g71b:
            from .integrations.spec_decode import (
                g4_71b_drafter_sliding_triton as _g4_71b_mod,
            )
            _g4_71b_mod.apply()
        # G4_76 — disable Gemma4Proposer._setup_gemma4_kv_sharing (PN265).
        # Apply AFTER G4_72 so the spec route is established. Must apply
        # before the proposer's setup is invoked at model load. Drafter
        # becomes fully independent — its own kv_cache_groups entry,
        # block table, and slot_mapping. Required after G4_74 broke the
        # physical alias (otherwise target's block ids overflow drafter's
        # smaller cache → CUDA illegal access on long prompts).
        if g76:
            try:
                import vllm.v1.spec_decode.gemma4  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode import (
                g4_76_disable_drafter_kv_sharing as _g4_76_mod,
            )
            _g4_76_mod.apply()
        # PN266 — propose() input-shape trace (G4_77 design probe).
        if pn266:
            try:
                import vllm.v1.spec_decode.llm_base_proposer  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn266_propose_trace as _pn266_mod,
            )
            _pn266_mod.apply()
        # PN267 — K/V bridge feasibility trace (G4_78-0 probe).
        if pn267:
            try:
                import vllm.v1.worker.gpu_model_runner  # noqa: F401
                import vllm.v1.attention.backends.flash_attn  # noqa: F401
                import vllm.v1.attention.backends.triton_attn  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn267_kv_bridge_trace as _pn267_mod,
            )
            _pn267_mod.apply()
        # PN268 — Drafter num_blocks origin trace (G4_78-CAP-0 probe).
        if pn268:
            try:
                import vllm.v1.worker.gpu_model_runner  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn268_drafter_blocks_origin as _pn268_mod,
            )
            _pn268_mod.apply()
        # PN269 — G4_78-A0 block_table accessibility trace.
        if pn269:
            try:
                import vllm.v1.attention.backends.flash_attn  # noqa: F401
                import vllm.v1.attention.backends.triton_attn  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn269_a0_block_table_trace as _pn269_mod,
            )
            _pn269_mod.apply()
        # G4_78 — target[58] -> drafter[0..2] K/V bridge (v1).
        # Apply AFTER PN269 so that PN269's wrap (if enabled) sits
        # inside G4_78 — bridge runs first, then trace logs the
        # substituted K/V.
        if g78:
            try:
                import vllm.v1.attention.backends.flash_attn  # noqa: F401
                import vllm.v1.attention.backends.triton_attn  # noqa: F401
            except ImportError:
                pass
            from .integrations._retired import (
                g4_78_drafter_target_kv_bridge as _g4_78_mod,
            )
            _g4_78_mod.apply()
        # PN270 — drafter K/V projection audit (read-only).
        if pn270:
            try:
                import vllm.v1.worker.gpu_model_runner  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn270_drafter_kv_proj_audit as _pn270_mod,
            )
            _pn270_mod.apply()
        # PN271 — SpecDecode KV contract audit (model-agnostic).
        if pn271:
            try:
                import vllm.v1.worker.gpu_model_runner  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode import (
                pn271_kv_contract_audit as _pn271_mod,
            )
            _pn271_mod.apply()
        # PN272 — Gemma 4 MTP drafter input semantics probe.
        if pn272:
            try:
                import vllm.v1.worker.gpu_model_runner  # noqa: F401
                import vllm.model_executor.models.gemma4_mtp  # noqa: F401
            except ImportError:
                pass
            from .integrations.spec_decode.probes import (
                pn272_gemma4_drafter_input_probe as _pn272_mod,
            )
            _pn272_mod.apply()
    except Exception:  # noqa: BLE001
        # Never block sndr_core import on G4-TQ apply error
        pass


# PN283 — Bootstrap prometheus_client multiprocess directory BEFORE
# any patch hook runs. PN282's worker-process Counters require the dir
# to exist + be writable by the time prometheus_client opens its first
# value file. Single-process deployments (env unset) → no-op.
# Observability must never block sndr_core import — failures are
# warnings, not errors.
try:
    from .observability.multiproc_bootstrap import setup_prometheus_multiproc_dir
    setup_prometheus_multiproc_dir()
except Exception:  # noqa: BLE001
    pass

_g4_19_import_time_hook()
del _g4_19_import_time_hook


# ─── PN274 SpecDecode safety guard — DEFAULT ON ──────────────────────
#
# Wires safety_guard.evaluate_from_config() into
# EngineArgs.create_engine_config so the spec-decode safety policy
# runs BEFORE workers spawn (saves drafter memory on denial).
#
# Default behavior: guard ON. Escape hatch:
#   SNDR_DISABLE_SPEC_DECODE_SAFETY_GUARD=1
#   (alias GENESIS_DISABLE_SPEC_DECODE_SAFETY_GUARD still works)
#
# Per-pair operator override (still subject to the disable env):
#   SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER=1
#   SNDR_ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN=1
#
# Impact on existing models:
#   Qwen (no Gemma4 hf_config + no MappingProvider match)
#     -> guard ALLOW -> speculative_config unchanged -> no regression
#   Gemma 4 + native KV
#     -> guard ALLOW (EXACT_COPY) -> unchanged
#   Gemma 4 + TQ kv_cache_dtype
#     -> guard DENY by default (FUNCTIONAL_UNVERIFIED) ->
#        speculative_config = None -> boots as TQ-only (safe)
def _pn274_safety_guard_install_time_hook():
    # Use the SNDR/GENESIS-aliased resolver. Both prefixes work; SNDR_
    # wins if both are set; GENESIS_ emits a deprecation warning.
    from .env import get_sndr_env_bool
    if get_sndr_env_bool("DISABLE_SPEC_DECODE_SAFETY_GUARD"):
        return
    try:
        import vllm.engine.arg_utils  # noqa: F401
    except ImportError:
        return
    try:
        from .integrations.spec_decode import pn274_install as _pn274_mod
        _pn274_mod.apply()
    except Exception:  # noqa: BLE001
        # Never block sndr_core import on guard install failure
        pass


_pn274_safety_guard_install_time_hook()
del _pn274_safety_guard_install_time_hook


__all__ = [
    "SNDR_CORE_VERSION",
    "GENESIS_VERSION",
    "brand",
    "env",
    "locations",
    "detection",
    "runtime",
    "integrations",
    "bundles",
    "Flags",
    "is_enabled",
    "is_disabled",
    "is_legacy_active",
    "is_meta_flag",
    "engine_targets",
    "resolve_vllm_file",
    "vllm_install_root",
]
