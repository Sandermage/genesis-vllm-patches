# SPDX-License-Identifier: Apache-2.0
"""PN346B — coordinator-half clamp (vendor of OPEN PR vllm#45614).

The COORDINATOR half of the SAME fix whose MANAGER half is PN346
==============================================================

Upstream PR vllm#45614 ("[Bugfix][Core] Fix Mamba prefix cache EAGLE
hit", closes vllm#43559) ships TWO halves that together fix the hybrid
Mamba + EAGLE/MTP + ``--enable-prefix-caching`` prefix-cache poison on
our exact PROD shape (Qwen3.6-35B-A3B FP8, MTP K=3, prefix caching on):

  * MANAGER half — ``single_type_kv_cache_manager.py``: walk the
    ``MambaManager`` search boundary back by one block on the EAGLE/MTP
    path so the partially-accepted final SSM state block is never
    matched. **Genesis already vendors this as PN346.**

  * COORDINATOR half — ``v1/core/kv_cache_coordinator.py`` (THIS patch):
    clamp ``curr_hit_length`` so it is **monotonically non-increasing**
    across ``HybridKVCacheCoordinator.find_longest_cache_hit``'s
    fixed-point iteration. **Genesis was MISSING this half.**

Genesis shipped only the manager half (PN346). The coordinator clamp
is the missing sibling — without it, the fixed-point loop can re-grow
``curr_hit_length`` on a verify pass and re-admit the very state block
PN346 walked back, partially defeating the manager-half guard. The two
halves MUST compose and ship together (registry ``composes_with``
cross-references PN346 ⇄ PN346B).

Root cause — the eagle-drop branch can GROW curr_hit_length
=========================================================

Inside ``HybridKVCacheCoordinator.find_longest_cache_hit`` (a while/for
fixed-point loop) each group computes::

    _new_hit_length = len(hit_blocks[0]) * spec.block_size
    if drop_eagle_block:                  # (dev491) / `if use_eagle:` (PROD)
        eagle_verified.add(idx)
    elif _new_hit_length < curr_hit_length:
        # length shrunk; invalidate previous eagle verifications
        eagle_verified.clear()
    curr_hit_length = _new_hit_length     # <-- BUG: naked assignment

The naked assignment unconditionally overwrites ``curr_hit_length``
with ``_new_hit_length``. On the eagle-drop branch ``_new_hit_length``
can be LONGER than the current candidate; the assignment then GROWS
``curr_hit_length`` on a verify pass and re-admits the partially-
accepted final state block — the exact poison PN346 fixes in the
manager. The fix clamps so the value can only stay or shrink::

    curr_hit_length = min(curr_hit_length, _new_hit_length)

This matches the manager-half guard's intent: the hit length is
monotonically non-increasing across the fixed-point iteration.

Pin-agnostic anchor (resolves byte-identically on dev491 AND PROD)
=================================================================

The line IMMEDIATELY ABOVE the clamp diverges by pin:

  * dev491 (0.22.1rc1.dev491+g1033ffac2): ``if drop_eagle_block:``
    (loop unpacks ``(spec, group_ids, manager_cls, use_eagle)``;
    local ``drop_eagle_block = use_eagle and idx not in eagle_verified``).
  * PROD   (0.21.1rc0+g626fa9bba):        ``if use_eagle:``
    (loop unpacks ``(spec, group_ids, manager_cls)``;
    local ``use_eagle = idx in self.eagle_attn_group_indices and
    idx not in eagle_verified``).

So the anchor DELIBERATELY EXCLUDES that ``if ...:`` line. The 4-line
anchor (``elif _new_hit_length < curr_hit_length:`` →
``curr_hit_length = _new_hit_length``) is byte-identical AND
grep-unique (count==1) on BOTH pins. Verified: ``min(curr_hit_length``
is absent on the live dev491 container AND the PROD image → the clamp
is genuinely missing on both.

Why we vendor this OPEN PR (not just wait for upstream merge)
=============================================================

  * It is the missing half of a correctness fix Genesis ALREADY ships
    default-ON (PN346). A half-fix is worse than none: the manager
    guard can be partially undone by the unclamped coordinator re-grow.
  * The fix is a ONE-LINE ``min()`` clamp — surgical, no perf cost on
    the non-EAGLE path, no signature change, no caller change.
  * Same upstream issue (#43559), same PR (#45614) as PN346 — they are
    a unit. Composition + default-ON parity guarantee both land
    together on every boot.

Composition + safety
====================

  * Targets ``v1/core/kv_cache_coordinator.py`` — a DIFFERENT file than
    PN346 (``single_type_kv_cache_manager.py``). Zero anchor overlap;
    the two coexist by construction.
  * No P85-style anchor overlap exists on the coordinator file, so
    PN346B needs no dual-anchor variants.
  * Opt-out-only (default-ON), mirroring PN346: honors
    ``GENESIS_DISABLE_PN346B``, ignores ``GENESIS_ENABLE_PN346B``.
  * Self-skips once #45614 merges upstream via the drift marker
    ``curr_hit_length = min(curr_hit_length, _new_hit_length)`` (the
    exact merged shape) → idempotent / skip, file untouched.

Upstream regression test (lives in the vLLM tree, not Genesis):
``test_hybrid_mamba_eagle_does_not_reuse_lookahead_state`` in
``tests/v1/core/test_prefix_caching.py`` (carried by PR #45614).

Risk: LOW — one-line ``min()`` clamp, only narrows the hit length.
Effort: XS.

Author: Sander Barzov Aleksandr (Sandermage, Ukraine, Odessa).
Vendor target: vllm-project/vllm#45614 (open as of 2026-06-17).
Sibling of PN346 (manager half). Closes vllm#43559 (coordinator half).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger(
    "genesis.wiring.pn346b_mamba_mtp_apc_coordinator_clamp"
)

GENESIS_PN346B_MARKER = (
    "Genesis PN346B vendor of vllm#45614 "
    "(Mamba/GDN + EAGLE/MTP + APC coordinator clamp) v1"
)


# Anchor: the 4-line sequence inside
# HybridKVCacheCoordinator.find_longest_cache_hit right where the loop
# overwrites curr_hit_length. It DELIBERATELY EXCLUDES the line above
# (`if drop_eagle_block:` on dev491 / `if use_eagle:` on PROD) which
# diverges by pin — so this anchor is byte-identical and grep-unique
# (count==1) on BOTH the live dev491 container and the PROD image.
PN346B_ANCHOR_OLD = (
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                curr_hit_length = _new_hit_length\n"
)

PN346B_ANCHOR_NEW = (
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                # [Genesis PN346B vendor of vllm#45614] Coordinator half of the\n"
    "                # Mamba+EAGLE/MTP APC prefix-cache fix. The eagle-drop branch can\n"
    "                # report a _new_hit_length LONGER than the current candidate; the\n"
    "                # naked assignment would then GROW curr_hit_length on a verify pass\n"
    "                # and re-admit the partially-accepted final state block. Clamp so\n"
    "                # the hit length is monotonically non-increasing across the\n"
    "                # fixed-point iteration, matching the manager-half guard (PN346).\n"
    "                curr_hit_length = min(curr_hit_length, _new_hit_length)\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN346B", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _make_patcher() -> Optional[TextPatcher]:
    """Build the coordinator-clamp patcher, or None if target absent."""
    target = resolve_vllm_file("v1/core/kv_cache_coordinator.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN346B vllm/v1/core/kv_cache_coordinator.py — "
            "Mamba/GDN + EAGLE/MTP + APC curr_hit_length min() clamp "
            "(coordinator half of #45614)"
        ),
        target_file=str(target),
        marker=GENESIS_PN346B_MARKER,
        sub_patches=[
            TextPatch(
                name="pn346b_coordinator_curr_hit_length_clamp",
                anchor=PN346B_ANCHOR_OLD,
                replacement=PN346B_ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Genesis sentinel only (our own idempotency marker). The
            # upstream-merge case is handled by ANCHOR-ABSENCE: once #45614
            # lands, the naked `curr_hit_length = _new_hit_length` anchor
            # (PN346B_ANCHOR_OLD) is gone, so the patcher SKIPs cleanly. A bare
            # `curr_hit_length = min(...)` marker would self-collide with this
            # patch's own replacement (caught by tools/lint_drift_markers.py),
            # so it is intentionally NOT used as a drift sentinel.
            "[Genesis PN346B",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN346B — coordinator-half curr_hit_length min() clamp."""
    if _env_disabled():
        return "skipped", "PN346B disabled via GENESIS_DISABLE_PN346B=1"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "v1/core/kv_cache_coordinator.py not found"

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN346B apply raised {e!r}"

    if result == TextPatchResult.FAILED:
        reason = failure.reason if failure else "unknown"
        return "failed", f"PN346B: {reason}"
    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "unknown"
        return "skipped", f"PN346B: {reason}"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN346B already applied (idempotent)"

    return "applied", (
        "PN346B applied: HybridKVCacheCoordinator.find_longest_cache_hit now "
        "clamps curr_hit_length = min(curr_hit_length, _new_hit_length) so the "
        "fixed-point hit length is monotonically non-increasing. Coordinator "
        "half of OPEN PR vllm#45614 (closes #43559); composes with the manager "
        "half PN346 — the two MUST ship together. 1-LOC surgical clamp. No-op "
        "on the non-EAGLE path."
    )


def is_applied() -> bool:
    from pathlib import Path
    target = resolve_vllm_file("v1/core/kv_cache_coordinator.py")
    if target is None:
        return False
    try:
        return GENESIS_PN346B_MARKER in Path(str(target)).read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError):
        return False
