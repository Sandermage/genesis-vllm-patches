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
    # PN241 — Gemma4MTPAttention.forward finite/norm trace (Codex-designed).
    pn241 = _os.environ.get(
        "GENESIS_ENABLE_PN241_MTP_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN248 — Acceptance trace via rejection_sample wrap (Step 4 from
    # corrected plan).
    pn248 = _os.environ.get(
        "GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    # PN258 — Oracle acceptance test (record/replay around rejection_sample).
    # First technical step of the post-PN257a cycle: separates "verifier
    # rows 1..K still broken" vs "drafter is the only remaining problem".
    pn258 = _os.environ.get(
        "GENESIS_ENABLE_PN258_ORACLE_ACCEPTANCE", ""
    ).strip().lower() in ("1", "true", "yes")
    if not (
        g19 or g19b or g19c or g30 or g31 or g32 or g43 or g44 or g45 or g50
        or g60a or g60b or g60c or g60d or g60e or g60g or g60h or g60k
        or g61 or g62 or g67 or g68 or g69 or g71 or g72
        or pn241 or pn248 or pn258
    ):
        return
    try:
        # G4_30/G4_43/G4_44/G4_45/G4_50 moved to sndr_private/g4_upstream_tq_wip/
        # 2026-05-17 — superseded by upstream PR #42637 (Mixed-attention TQ for
        # Gemma 4). Their env flags still resolve via sndr_private fallback if
        # explicitly enabled; left here as no-ops to maintain back-compat.
        if g30 or g43 or g44 or g45 or g50:
            try:
                if g30:
                    from .sndr_private.g4_upstream_tq_wip import (
                        g4_30_upstream_tq_unblock as _m,
                    )
                    _m.apply()
                if g43:
                    from .sndr_private.g4_upstream_tq_wip import (
                        g4_43_unblock_forced_triton as _m,
                    )
                    _m.apply()
                if g44:
                    from .sndr_private.g4_upstream_tq_wip import (
                        g4_44_tq_head_dim_512_prefill as _m,
                    )
                    _m.apply()
                if g45:
                    from .sndr_private.g4_upstream_tq_wip import (
                        g4_45_unify_page_diag as _m,
                    )
                    _m.apply()
                if g50:
                    from .sndr_private.g4_upstream_tq_wip import (
                        g4_50_genesis_native_backend as _m,
                    )
                    _m.apply()
            except ImportError:
                # sndr_private may be absent in slim distributions; silent no-op
                pass
        # G4_31 active in main — preserves turboquant_* dtype from AWQ override.
        if g31:
            from .integrations.gemma4 import (
                g4_31_preserve_tq_dtype as _g4_31_mod,
            )
            _g4_31_mod.apply()
        # G4_32 active in main — bypasses TQ validate_configuration.
        if g32:
            from .integrations.gemma4 import (
                g4_32_tq_validation_bypass as _g4_32_mod,
            )
            _g4_32_mod.apply()
        if g19b:
            from .integrations.gemma4 import (
                g4_19b_gemma4_tq_kv_spec_integration as _g4_19b_mod,
            )
            _g4_19b_mod.apply()
        if g19:
            # Pre-import gemma4 so Gemma4Config exists for the wrapper
            try:
                import vllm.model_executor.models.gemma4  # noqa: F401
            except ImportError:
                pass
            from .integrations.gemma4 import (
                g4_19_gemma4_turboquant_kv_cache as _g4_19_mod,
            )
            _g4_19_mod.apply()
        if g19c:
            # G4_19c depends on G4_19 having published the config registry,
            # so apply order matters: G4_19 first, G4_19c second.
            try:
                import vllm.model_executor.models.gemma4  # noqa: F401
            except ImportError:
                pass
            from .integrations.gemma4 import (
                g4_19c_attention_wrapper as _g4_19c_mod,
            )
            _g4_19c_mod.apply()
        # === G4_60* — PR #42637 cherry-pick stack (apply in dependency order) ===
        # G4_60a (TQSlidingWindowSpec class injection) is prerequisite for
        # G4_60e (kv_cache_utils mixed-route) and G4_60g (dispatch). Apply first.
        if g60a:
            from .integrations.gemma4 import (
                g4_60a_tq_sliding_window_spec as _g4_60a_mod,
            )
            _g4_60a_mod.apply()
        # G4_60h augments TurboQuantConfig — required by G4_60k.
        if g60h:
            from .integrations.gemma4 import (
                g4_60h_turboquant_config_augment as _g4_60h_mod,
            )
            _g4_60h_mod.apply()
        # G4_60e patches kv_cache_utils — needs G4_60a first.
        if g60e:
            from .integrations.gemma4 import (
                g4_60e_kv_cache_utils as _g4_60e_mod,
            )
            _g4_60e_mod.apply()
        # G4_60g patches Attention.get_kv_cache_spec — needs G4_60a first.
        if g60g:
            from .integrations.gemma4 import (
                g4_60g_attention_dispatch as _g4_60g_mod,
            )
            _g4_60g_mod.apply()
        # G4_60k wraps EngineArgs.create_engine_config — applies post-build.
        # Order is independent but typically after G4_60h.
        if g60k:
            from .integrations.gemma4 import (
                g4_60k_arg_utils as _g4_60k_mod,
            )
            _g4_60k_mod.apply()
        # G4_61 shares TQ decode workspace across layers (PR #40798).
        # Apply before any model forward — patches launcher + capture_model.
        if g61:
            from .integrations.gemma4 import (
                g4_61_tq_shared_workspace as _g4_61_mod,
            )
            _g4_61_mod.apply()
        # G4_62 warms up TQ decode kernels before lock_workspace (PR #42215).
        # Apply order: G4_62 after G4_61 so warmup uses shared workspace path.
        if g62:
            from .integrations.gemma4 import (
                g4_62_tq_kernel_warmup as _g4_62_mod,
            )
            _g4_62_mod.apply()
        # G4_60b/c/d verify PR #42637 bind-mount overlay activated. These are
        # diagnostic only — they don't perform the mount (that's done in the
        # launch script via docker -v flags). Apply after all monkey-patches
        # so verification reflects the final live state.
        if g60b:
            from .integrations.gemma4 import (
                g4_60b_turboquant_attn_overlay_loader as _g4_60b_mod,
            )
            _g4_60b_mod.apply()
        if g60c:
            from .integrations.gemma4 import (
                g4_60c_triton_decode_overlay_loader as _g4_60c_mod,
            )
            _g4_60c_mod.apply()
        if g60d:
            from .integrations.gemma4 import (
                g4_60d_triton_store_overlay_loader as _g4_60d_mod,
            )
            _g4_60d_mod.apply()
        # G4_67 backports PR #40914 — must apply AFTER G4_60b (overlay
        # verifier) so TurboQuantAttentionImpl exists with PR #42637
        # signatures before we monkey-patch its forward method.
        if g67:
            from .integrations.gemma4 import (
                g4_67_tq_spec_verify_routing as _g4_67_mod,
            )
            _g4_67_mod.apply()
        # G4_68 — verifier for inlined P65 v2 cudagraph downgrade in the
        # PR #42637 overlay's TurboQuantMetadataBuilder. Must apply AFTER
        # G4_60b so TurboQuantMetadataBuilder is imported from the
        # overlay. Reports applied/error/skipped; no monkey-patching.
        if g68:
            from .integrations.gemma4 import (
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
            from .integrations.gemma4 import (
                g4_69_skip_layers_native_backend as _g4_69_mod,
            )
            _g4_69_mod.apply()
        # G4_71 — force FlashAttn backend for drafter Attention layers.
        # Must apply BEFORE drafter model loads (i.e. before
        # LLMBaseProposer.load_model runs). Plugin apply-all happens at
        # engine_core boot, well before model load, so order is safe.
        if g71:
            from .integrations.gemma4 import (
                g4_71_drafter_native_attn_backend as _g4_71_mod,
            )
            _g4_71_mod.apply()
        # G4_72 — force native FullAttentionSpec/SlidingWindowSpec for
        # drafter Attention layers (companion to G4_71). MUST apply AFTER
        # G4_60g so its wrap sits on top — non-drafter layers continue
        # through G4_60g's TQ-first dispatch; only drafter (G4_71 marker)
        # is rerouted to native spec.
        if g72:
            from .integrations.gemma4 import (
                g4_72_drafter_native_kv_cache_spec as _g4_72_mod,
            )
            _g4_72_mod.apply()
        # PN241 — Codex-designed finite/norm trace at SpecDecodeBaseProposer
        # boundary (Python orchestration above torch.compile boundary).
        # Logs target_hidden_states (input) + draft_token_ids (output) per
        # propose() call.
        if pn241:
            try:
                import vllm.v1.spec_decode.llm_base_proposer  # noqa: F401
            except ImportError:
                pass
            from .integrations.gemma4 import (
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
            from .integrations.gemma4 import (
                pn248_acceptance_trace as _pn248_mod,
            )
            _pn248_mod.apply()
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
            from .integrations.gemma4 import (
                pn258_oracle_acceptance as _pn258_mod,
            )
            _pn258_mod.apply()
    except Exception:  # noqa: BLE001
        # Never block sndr_core import on G4-TQ apply error
        pass


_g4_19_import_time_hook()
del _g4_19_import_time_hook


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
