# SPDX-License-Identifier: Apache-2.0
"""Consolidated wiring for PN29 + PN298 — both target the SAME engine file
``model_executor/layers/fla/ops/chunk_o.py`` at DISJOINT regions.

================================================================
WHY THIS MODULE EXISTS (maintainability refactor, 2026-06-19)
================================================================

PN29 (GDN chunk_o scale-fold, vllm#41446 pattern (c)) and PN298 (FLA
chunk_o NUM_WARPS arch-aware prune) historically lived in two separate
wiring modules, each with its own ``TextPatcher`` and its own
``# [Genesis wiring marker: ...]`` line, even though they patch the same
file at non-overlapping anchors:

  - PN29   → ``chunk_fwd_kernel_o`` body (the scale-multiply fold), ~line 137.
  - PN298  → the module-level ``NUM_WARPS = ...`` autotune-config block,
             ~lines 21-22.

This module collapses both into ONE ``TextPatcher`` with ONE shared
marker and TWO sub-patches. The applied OUTPUT for the kernel-code
regions is byte-identical to PN29+PN298 applied separately — the
anchors and replacements are copied VERBATIM from the originals (see
the verbatim blocks below). The only intentional difference vs the
two-module layout is a single shared wiring-marker comment line instead
of two — wiring metadata, not kernel code.

================================================================
PER-FEATURE GATING (byte-equivalent with the original modules)
================================================================

The two features stay INDEPENDENTLY operator-gated, exactly as before:

  - ``pn29_scale_fold``  is applied iff PN29's original gate would have
    passed. PN29 routed through the dispatcher's ``should_apply("PN29")``
    which, for a ``tier=community`` patch with the version-range gate
    OFF (default), reduces to: env ENABLE truthy AND not DISABLE. We
    replicate that with ``env.is_enabled``/``env.is_disabled`` on the
    bare flag ``PN29_GDN_SCALE_FOLD`` (so ``SNDR_*`` aliases + the
    ``GENESIS_DISABLE_PN29_GDN_SCALE_FOLD`` kill-switch keep working).

  - ``pn298_num_warps`` is applied iff PN298's original DIRECT env check
    passed. PN298 did NOT route through ``should_apply`` — it read
    ``GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS`` literally with
    ``.lower() in ("1","true","yes","on")``. We replicate that literal
    check verbatim so the gating is byte-identical (no SNDR_* alias, no
    DISABLE kill-switch — matching the original PN298 behavior).

================================================================
PN296 PRECONDITION — PRESERVED FOR pn298 ONLY
================================================================

PN298's coupling to PN296 (the GPU arch-profile keystone) was NEVER a
runtime apply-gate: the original ``pn298.apply()`` did not consult PN296
state. The coupling lived (a) declaratively as ``requires_patches=
["PN296"]`` in the registry (used only by the audit/topo-sort graph,
NOT by ``should_apply``), and (b) as a runtime check INSIDE the injected
replacement code — ``PN298_NEW`` reads ``get_gpu_arch_profile()`` and
falls back to the upstream ``NUM_WARPS`` expression when the profile is
absent (i.e. when PN296 has not booted the profiler). That runtime
precondition is carried VERBATIM in ``PN298_NEW`` below, so it applies
to the ``pn298_num_warps`` sub-patch ONLY. ``PN29_REPLACEMENT`` contains
no such check, so PN29's scale-fold is NOT transitively gated on PN296.

Because the registry entry-level ``requires_patches=["PN296"]`` would
over-gate PN29 if carried on the merged entry, it is intentionally NOT
present on the merged entry — the PN296 precondition lives here, scoped
to pn298.

================================================================

Authors:
  - PN29:  Sandermage (Sander) Barzov Aleksandr — backport of
           vllm-project/vllm#41446 (zobinHuang) pattern (c).
  - PN298: Sandermage (Sander) Barzov Aleksandr, 2026-06-05.
  - Consolidation: 2026-06-19 (maintainability refactor; runtime-neutral).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn29_pn298_chunk_o_consolidated")


# ─── Shared idempotency marker for the merged patcher ──────────────────
GENESIS_PN29_PN298_MARKER = (
    "Genesis PN29+PN298 chunk_o consolidated "
    "(scale-fold + arch-aware NUM_WARPS) v1"
)

# Original per-feature markers are RE-EXPORTED unchanged so existing
# tests, drift-residue coverage, and operator greps for the old marker
# strings keep resolving against this consolidated module.
GENESIS_PN29_MARKER = (
    "Genesis PN29 GDN chunk_o scale-fold (vllm#41446 pattern c) v7.65"
)
GENESIS_PN298_MARKER = (
    "Genesis PN298 FLA chunk_o NUM_WARPS arch-aware (SM 8.6 prune) v1"
)


# ─── Sub-patch 1: PN29 scale-fold (VERBATIM from pn29_gdn_chunk_o_scale_fold) ──
# Anchor: the exact current upstream line. Indentation: 4 spaces (kernel body).
PN29_ANCHOR = (
    "    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale\n"
)

PN29_REPLACEMENT = (
    "    # [Genesis PN29 vllm#41446 pattern (c) backport]\n"
    "    # Scale-fold: (b_o + dot) * scale instead of b_o*scale + dot*scale.\n"
    "    # One fewer fp32 multiply per inner iteration. Distributive on fp32\n"
    "    # accumulators (drift bounded by 1-2 ULP per element, verified by\n"
    "    # TDD test_pn29_numerical_equivalence_*). Triton compiler does NOT\n"
    "    # auto-fuse across the +/- boundary, so explicit fold = guaranteed\n"
    "    # 1 fewer op per chunk_fwd_kernel_o iteration.\n"
    "    # Source: zobinHuang vllm#41446 GDN MI300X optimization, pattern (c)\n"
    "    # is hardware-agnostic (NVIDIA Triton compatible).\n"
    "    b_o = (b_o + tl.dot(b_A.to(b_v.dtype), b_v)) * scale\n"
)


# ─── Sub-patch 2: PN298 NUM_WARPS (VERBATIM from pn298_fla_chunk_o_arch_warps) ──
PN298_OLD = (
    "BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]\n"
    "NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]\n"
)

PN298_NEW = (
    "BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]\n"
    "# [Genesis PN298 2026-06-05] arch-aware NUM_WARPS — drop num_warps=8\n"
    "# on Ampere SM 8.6 (A5000 100KB shared/SM cannot fit 8-warp configs\n"
    "# with BV=128 — spills registers). is_nvidia_hopper covers SM 9.0+\n"
    "# but Ampere consumer also benefits. A100 (SM 8.0, 164KB) keeps 8.\n"
    "try:\n"
    "    from sndr.detection.gpu_arch_profile import (\n"
    "        get_gpu_arch_profile as _genesis_pn298_get_profile,\n"
    "    )\n"
    "    _genesis_pn298_prof = _genesis_pn298_get_profile()\n"
    "    if _genesis_pn298_prof is not None:\n"
    "        _genesis_pn298_max = _genesis_pn298_prof.max_safe_num_warps\n"
    "        NUM_WARPS = [w for w in [2, 4, 8] if w <= _genesis_pn298_max]\n"
    "    else:\n"
    "        NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]\n"
    "except Exception:\n"
    "    NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]\n"
)


# Bare env-flag names (no GENESIS_ENABLE_/SNDR_ENABLE_ prefix).
_PN29_FLAG = "PN29_GDN_SCALE_FOLD"
_PN298_ENV = "GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS"
_TRUTHY = ("1", "true", "yes", "on")


def _pn29_sub_patch() -> TextPatch:
    return TextPatch(
        name="pn29_scale_fold",
        anchor=PN29_ANCHOR,
        replacement=PN29_REPLACEMENT,
        required=True,
    )


def _pn298_sub_patch() -> TextPatch:
    return TextPatch(
        name="pn298_num_warps",
        anchor=PN298_OLD,
        replacement=PN298_NEW,
        required=True,
    )


def _make_patcher() -> TextPatcher | None:
    """Drift-tool / static entry point: ONE TextPatcher carrying BOTH
    sub-patches UNCONDITIONALLY.

    ``tools/check_upstream_drift.py`` builds the patcher from this
    function and verifies every sub-patch anchor is present-and-unique in
    the pristine tree. Both anchors MUST be declared here regardless of
    runtime env gating so the static drift check covers both.
    """
    target = resolve_vllm_file("model_executor/layers/fla/ops/chunk_o.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN29+PN298 model_executor/layers/fla/ops/chunk_o.py — "
            "scale-fold (chunk_fwd_kernel_o, vllm#41446 pattern c) + "
            "arch-aware NUM_WARPS prune (SM 8.6 100KB shared mem budget)"
        ),
        target_file=str(target),
        marker=GENESIS_PN29_PN298_MARKER,
        sub_patches=[
            _pn29_sub_patch(),
            _pn298_sub_patch(),
        ],
        upstream_drift_markers=[
            # If upstream PR #41446 lands pattern (c), the PN29 anchor line
            # already reads ``(b_o + dot) * scale`` and won't match → no-op.
            "[Genesis PN29",
            # PN298 residue coverage banner (self-collision lint: the
            # internal "_genesis_pn298_max" token is baked by our own
            # replacement, so coverage stays on the banner).
            "[Genesis PN298",
        ],
    )


def _pn29_enabled() -> bool:
    """Replicates PN29's original ``should_apply("PN29")`` gate for a
    ``tier=community`` patch with the version-range enforcement gate OFF
    (the default): env ENABLE truthy AND not explicitly DISABLEd."""
    from sndr.env import is_disabled, is_enabled

    return is_enabled(_PN29_FLAG) and not is_disabled(_PN29_FLAG)


def _pn298_enabled() -> bool:
    """Replicates PN298's original DIRECT env check verbatim. PN298 did
    NOT route through ``should_apply`` — it read the full-prefix env name
    literally (no SNDR_* alias, no DISABLE kill-switch)."""
    return os.environ.get(_PN298_ENV, "").lower() in _TRUTHY


def apply() -> tuple[str, str]:
    """Apply PN29 + PN298 (consolidated) — each sub-patch independently
    operator-gated so the applied set is byte-identical to running the
    two original modules separately."""
    pn29_on = _pn29_enabled()
    pn298_on = _pn298_enabled()

    if not pn29_on and not pn298_on:
        return "skipped", (
            "PN29+PN298 both default OFF — set "
            "GENESIS_ENABLE_PN29_GDN_SCALE_FOLD=1 (scale-fold) and/or "
            "GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS=1 (arch-aware "
            "NUM_WARPS prune) to engage. Each flag independently gates its "
            "own sub-patch."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target = resolve_vllm_file("model_executor/layers/fla/ops/chunk_o.py")
    if target is None:
        return "skipped", "model_executor/layers/fla/ops/chunk_o.py not found"

    sub_patches: list[TextPatch] = []
    if pn29_on:
        sub_patches.append(_pn29_sub_patch())
    if pn298_on:
        sub_patches.append(_pn298_sub_patch())

    patcher = TextPatcher(
        patch_name=(
            "PN29+PN298 model_executor/layers/fla/ops/chunk_o.py — "
            "scale-fold + arch-aware NUM_WARPS prune"
        ),
        target_file=str(target),
        marker=GENESIS_PN29_PN298_MARKER,
        sub_patches=sub_patches,
        upstream_drift_markers=["[Genesis PN29", "[Genesis PN298"],
    )

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", failure.reason if failure else "unknown TextPatch failure"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown TextPatch skip"
    if result == TextPatchResult.IDEMPOTENT:
        return "skipped", (
            "PN29+PN298 consolidated: already applied (marker present)"
        )

    applied = patcher.applied_sub_patches or [sp.name for sp in patcher.sub_patches]
    enabled = []
    if pn29_on:
        enabled.append("PN29 scale-fold")
    if pn298_on:
        enabled.append("PN298 arch-aware NUM_WARPS")
    return "applied", (
        f"PN29+PN298 consolidated installed ({', '.join(enabled)}). "
        f"chunk_o.py: scale-fold uses (b_o + dot) * scale; NUM_WARPS reads "
        f"get_gpu_arch_profile().max_safe_num_warps (PN296-conditioned, "
        f"upstream fallback when profile absent). Sub-patches applied: "
        f"{', '.join(applied)}."
    )


def is_applied() -> bool:
    """Best-effort idempotency probe — True iff the shared marker is
    present in the target file."""
    target = resolve_vllm_file("model_executor/layers/fla/ops/chunk_o.py")
    if target is None:
        return False
    try:
        with open(str(target), "r", encoding="utf-8", errors="ignore") as fh:
            return GENESIS_PN29_PN298_MARKER in fh.read()
    except OSError:
        return False
