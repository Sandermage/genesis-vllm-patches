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
    if not (g19 or g19b or g19c or g30 or g31 or g32 or g43 or g44 or g45):
        return
    try:
        # G4_30 must apply BEFORE Attention.__init__ runs, so do it first.
        # It monkey-patches vllm.v1.attention.backends.turboquant_attn
        # TurboQuantAttentionBackend.supports_mm_prefix → True.
        if g30:
            from .integrations.gemma4 import (
                g4_30_upstream_tq_unblock as _g4_30_mod,
            )
            _g4_30_mod.apply()
        # G4_31 also wraps Attention.__init__ — must apply before model init.
        if g31:
            from .integrations.gemma4 import (
                g4_31_preserve_tq_dtype as _g4_31_mod,
            )
            _g4_31_mod.apply()
        # G4_32 bypasses TQ backend's validate_configuration. Must apply
        # BEFORE get_attn_backend is called, which means before any
        # Attention.__init__ — same timing as G4_30.
        if g32:
            from .integrations.gemma4 import (
                g4_32_tq_validation_bypass as _g4_32_mod,
            )
            _g4_32_mod.apply()
        # G4_43 reverts vllm's forced TRITON_ATTN on Gemma 4. Must apply
        # BEFORE Gemma4Config.verify_and_update_config runs (during boot).
        if g43:
            from .integrations.gemma4 import (
                g4_43_unblock_forced_triton as _g4_43_mod,
            )
            _g4_43_mod.apply()
        # G4_44 patches TQ _prefill_attention to fall back to torch SDPA
        # for head_dim > 256. Must apply BEFORE Attention.__init__
        # creates its backend instance.
        if g44:
            from .integrations.gemma4 import (
                g4_44_tq_head_dim_512_prefill as _g4_44_mod,
            )
            _g4_44_mod.apply()
        if g45:
            from .integrations.gemma4 import (
                g4_45_unify_page_diag as _g4_45_mod,
            )
            _g4_45_mod.apply()
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
