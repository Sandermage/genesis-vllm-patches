# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 28 — GDN core_attn_out prealloc (CRIT-HW-1 correct form).

Architecture
------------
Per master-plan CRIT-HW-1 ("P28 MUST pre-allocate at `__init__`, NEVER
lazy in forward"), this module does TWO things:

  1. **Class-method monkey-patch on `GatedDeltaNet.__init__`**. After the
     original __init__ runs, we call `gdn_core_attn_manager.attach_buffer`
     which allocates `self._genesis_gdn_core_attn_buf` (tensor OR None).
     This runs EAGER, once per module, outside any torch.compile trace —
     so device probes, env reads, dict lookups, logging are all safe.

  2. **Text-patch on `forward_cuda`**. The original `torch.zeros(...)`
     line is replaced with a pure-tensor conditional slice:

         core_attn_out = (
             self._genesis_gdn_core_attn_buf[:num_tokens].zero_()
             if self._genesis_gdn_core_attn_buf is not None
             else torch.zeros(
                 (num_tokens, self.num_v_heads // self.tp_size,
                  self.head_v_dim),
                 dtype=hidden_states.dtype, device=hidden_states.device,
             )
         )

     Both branches are pure tensor ops. The `is not None` guard resolves
     at trace time against a constant module attribute — `torch.dynamo`
     compiles only the selected branch and everything stays in-graph.

Platform compatibility
----------------------
  - NVIDIA CUDA SM ≥ 8.0 with the attribute set → pre-allocated slice.
  - All others (attribute is None) → fall-through `torch.zeros`
    identical to upstream behavior.

Upstream drift detection
------------------------
If `_genesis_gdn_core_attn_buf` already appears in the file OR upstream
lands its own buffer-pool fix, we skip.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""

# Legacy auto-apply note (audit 2026-05-11): registry env_flag
# `GENESIS_LEGACY_P28` is synthetic — flag exists for registry/audit
# coherence but has no runtime effect. Patch applies unconditionally
# via dispatcher's legacy auto-apply path (`is_legacy_active` in
# vllm/sndr_core/dispatcher/decision.py). See registry.py "Legacy
# patches" section (~line 2083) for full context.

from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p28_gdn_core_attn")

GENESIS_P28_MARKER = "Genesis P28 GDN core_attn_out prealloc v7.0"

UPSTREAM_DRIFT_MARKERS = [
    # Self-collision lint (triage plan §6 2026-06-11): former entry
    # "_genesis_gdn_core_attn_buf" is a Genesis-only name baked by our own
    # replacement — false "upstream_merged" skip on residue. Remaining
    # entries are hypothetical-upstream-only names, never emitted by us.
    "gdn_core_attn_out_buffer",
    "gdn_core_attn_prealloc",
]


# Anchor: disambiguates from forward_xpu's identical line via the
# preceding "see discussions in https://github.com/vllm-project/vllm/pull/28182"
# comment (unique to forward_cuda).
_OLD_ALLOC = (
    "        # Note: we should not use torch.empty here like other attention backends,\n"
    "        # see discussions in https://github.com/vllm-project/vllm/pull/28182\n"
    "        core_attn_out = torch.zeros(\n"
    "            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "            dtype=hidden_states.dtype,\n"
    "            device=hidden_states.device,\n"
    "        )"
)

_NEW_ALLOC = (
    "        # Note: we should not use torch.empty here like other attention backends,\n"
    "        # see discussions in https://github.com/vllm-project/vllm/pull/28182\n"
    "        # [Genesis P28] Pre-allocated buffer attached by attach_buffer()\n"
    "        # at module __init__ (see vllm._genesis.kernels.gdn_core_attn_manager).\n"
    "        # Both branches are pure tensor ops — fully torch.dynamo-safe.\n"
    "        # v2 capacity guard (2026-06-10): slicing a 4096-row buffer to\n"
    "        # [:6503] silently returns 4096 rows — Inductor's alias\n"
    "        # regeneration then dies with setStorage-out-of-bounds when a\n"
    "        # prefill chunk exceeds the prealloc (scheduler allows chunks\n"
    "        # up to max_num_batched_tokens; the budget resolver can fall\n"
    "        # back to 4096 when the vllm config context is unset in the\n"
    "        # worker). Eager-alloc fallback when over capacity.\n"
    "        core_attn_out = (\n"
    "            self._genesis_gdn_core_attn_buf[:num_tokens].zero_()\n"
    "            if getattr(self, '_genesis_gdn_core_attn_buf', None) is not None\n"
    "            and self._genesis_gdn_core_attn_buf.shape[0] >= num_tokens\n"
    "            else torch.zeros(\n"
    "                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "                dtype=hidden_states.dtype,\n"
    "                device=hidden_states.device,\n"
    "            )\n"
    "        )"
)


def _make_patcher() -> TextPatcher | None:
    # K.1.R.R fallback (2026-05-29): upstream moved gdn_linear_attn.py
    # into per-model mamba/gdn/{qwen,olmo,kimi}_gdn_linear_attn.py
    # split in dev371 -> nightly-626fa9bb window. P28 anchor text
    # (`core_attn_out = torch.zeros(...)`) is byte-identical in
    # new qwen_gdn_linear_attn.py. Old path stays canonical for
    # dev371 baseline; new path covers 626fa9bb+.
    target = (
        resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
        or resolve_vllm_file(
            "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
        )
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="P28 GDN core_attn_out prealloc",
        target_file=target,
        marker=GENESIS_P28_MARKER,
        sub_patches=[
            TextPatch(
                name="p28_core_attn_out_alloc",
                anchor=_OLD_ALLOC,
                replacement=_NEW_ALLOC,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


# ─── Runtime init wrap ─────────────────────────────────────────────────────
# Wraps `GatedDeltaNet.__init__` so every new instance gets its buffer
# attached after the original init completes. Idempotent.

_INIT_WRAPPED_ATTR = "_genesis_p28_init_wrapped"


# Candidate class names across vLLM versions. Older baselines named the
# class `GatedDeltaNet`; post-2026-04 renamed to `GatedDeltaNetAttention`
# (to reflect the PluggableLayer / MambaBase mixin); 2026-05+ upstream
# split per-model (Qwen / Olmo / Kimi) with shared base `GatedDeltaNetAttention`
# in `mamba.gdn.base`.
# MODEL-SPECIFIC subclasses come FIRST — their __init__ sets per-model
# attrs (num_v_heads, head_v_dim) that attach_buffer reads. If we resolved
# the shared base class (GatedDeltaNetAttention) by mistake, the wrap
# would fire BEFORE the subclass __init__ assigns those attrs.
_CANDIDATE_CLASS_NAMES = (
    "QwenGatedDeltaNetAttention",
    "OlmoGatedDeltaNetAttention",
    "KimiGatedDeltaNetAttention",
    "GatedDeltaNetAttention",  # shared base / legacy single-file fallback
    "GatedDeltaNet",           # pre-2026 name
)

# Candidate module paths, newest first. MUST wrap a class whose
# __init__ actually sets `num_v_heads` / `head_v_dim` on the instance
# (those are checked by attach_buffer). The 2026-05+ split moved those
# attribute assignments into PER-MODEL subclasses (e.g.
# QwenGatedDeltaNetAttention.__init__ sets them; the shared base does
# NOT). Wrapping the base would fire BEFORE the subclass sets attrs,
# so attach_buffer would always log "module missing num_v_heads/head_v_dim".
# Order matters: try concrete subclasses first.
_CANDIDATE_MODULES = (
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn",
    "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn",
    "vllm.model_executor.layers.mamba.gdn.base",          # legacy fallback
    "vllm.model_executor.layers.mamba.gdn_linear_attn",   # pre-2026 single file
)


def _resolve_gdn_class():
    """Import the GDN class. Returns first match across known module
    paths × class names, or None.

    Fix (2026-06-09): pin 0.22.1rc1.dev259+ split GDN into per-model
    files under `mamba.gdn.*`. Previously hardcoded the legacy single-
    file path; the runtime resolver silently returned None so the
    __init__ wrap never installed and `_genesis_gdn_core_attn_buf`
    was never attached. The text-patched `forward_cuda` then fell
    back to eager `torch.empty_like` allocation per call on the GDN
    decode hot path (30 layers × 1 per token).
    """
    import importlib
    last_err = None
    for mod_path in _CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
        for cls_name in _CANDIDATE_CLASS_NAMES:
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                log.info(
                    "[Genesis P28] resolved %s.%s for __init__ wrap",
                    mod_path, cls_name,
                )
                return cls
    log.warning(
        "[Genesis P28] no GDN class found across modules %s × names %s "
        "(last error: %s) — __init__ wrap CANNOT install, decode hot path "
        "will fall back to eager allocation",
        list(_CANDIDATE_MODULES), list(_CANDIDATE_CLASS_NAMES), last_err,
    )
    return None


def _wrap_gdn_init() -> bool:
    """Monkey-patch the GDN class's `__init__`. Return True on success."""
    cls = _resolve_gdn_class()
    if cls is None:
        return False

    if getattr(cls.__init__, _INIT_WRAPPED_ATTR, False):
        return True  # already wrapped (idempotent)

    orig_init = cls.__init__

    from sndr.engines.vllm.kernels_legacy.gdn_core_attn_manager import attach_buffer

    def _genesis_wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            attach_buffer(self)
        except Exception as e:
            log.warning(
                "[Genesis P28] attach_buffer in __init__ failed: %s "
                "(module will fall back to eager alloc on first forward)",
                e,
            )
            if not hasattr(self, "_genesis_gdn_core_attn_buf"):
                self._genesis_gdn_core_attn_buf = None

    setattr(_genesis_wrapped_init, _INIT_WRAPPED_ATTR, True)
    setattr(_genesis_wrapped_init, "_genesis_p28_original_init", orig_init)
    cls.__init__ = _genesis_wrapped_init
    log.info(
        "[Genesis P28] wrapped %s.__init__ to attach "
        "_genesis_gdn_core_attn_buf on each instance",
        cls.__name__,
    )
    return True


def is_applied() -> bool:
    """Verify init wrap is live (used by verify_live_rebinds)."""
    cls = _resolve_gdn_class()
    if cls is None:
        return False
    return getattr(cls.__init__, _INIT_WRAPPED_ATTR, False)


def revert() -> bool:
    """Restore original __init__. Returns True on success."""
    cls = _resolve_gdn_class()
    if cls is None:
        return False
    cur = cls.__init__
    if not getattr(cur, _INIT_WRAPPED_ATTR, False):
        return False
    orig = getattr(cur, "_genesis_p28_original_init", None)
    if orig is None:
        return False
    cls.__init__ = orig
    return True


def apply() -> tuple[str, str]:
    """Apply P28 wiring: warm-up caches + text-patch forward + wrap __init__.

    Never raises.
    """
    # Step 0: warm up the module-level caches (should_apply, env budget)
    # so traced forward paths never have to do device probes or env reads.
    try:
        from sndr.engines.vllm.kernels_legacy.gdn_core_attn_manager import warm_up
        warm_up()
    except Exception as e:
        log.info("[Genesis P28] warm_up failed (non-fatal): %s", e)

    # P53 (v7.9): Hybrid-active dispatch gate. GDN attention only exists
    # on hybrid models (Qwen3-Next, Mamba2 variants). On pure-attention
    # models the text-patch anchor won't even match, but skipping early
    # keeps dispatch logs clean.
    try:
        from sndr.engines.vllm.detection.model_detect import is_hybrid_model, log_skip
        if not is_hybrid_model():
            log_skip("P28 GDN core-attn forward rewire", "pure-attention model (no GDN)")
            return "skipped", "P53 dispatch: model has no hybrid linear-attention layers"
    except Exception as e:
        log.debug("[Genesis P28] model_detect probe failed (proceeding): %s", e)

    # Step 1: text-patch forward_cuda
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", failure.reason if failure else "unknown failure"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    # APPLIED or IDEMPOTENT — proceed to init wrap.

    # Step 2: wrap __init__ so new GDN instances get the buffer attached.
    init_ok = _wrap_gdn_init()
    if result == TextPatchResult.APPLIED:
        reason = "forward_cuda patched + __init__ wrapped" if init_ok \
            else "forward_cuda patched, __init__ wrap skipped"
    else:
        reason = "already applied (idempotent)" if init_ok \
            else "idempotent; init wrap skipped"
    return "applied", reason
