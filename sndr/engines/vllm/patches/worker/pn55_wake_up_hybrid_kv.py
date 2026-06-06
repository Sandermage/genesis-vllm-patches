# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN55 — wake_up crash fix on hybrid KV cache.

PN55v2 (PR38 Day 2, 2026-05-08): unified backport covering both
[vllm-project/vllm#41602](https://github.com/vllm-project/vllm/pull/41602)
(`list[Tensor]` for Mamba) **and**
[vllm-project/vllm#41896](https://github.com/vllm-project/vllm/pull/41896)
(nested `list` / `tuple` / `dict-Mapping` for FP8 KV scales).

Both PRs touch the same wake_up zeroing site and would conflict if
backported as separate patches. PN55v2 collapses them into one
recursive iterator so any nested container shape works.

================================================================
WHAT THIS PATCH DOES
================================================================

In `v1/worker/gpu_model_runner.py`'s wake_up flow, the original loop
naïvely called `.zero_()` on each `kv_caches` element:

    for cache_tensor in kv_caches:
        if cache_tensor is not None:
            cache_tensor.zero_()

This breaks on:

  - **Mamba / DeltaNet hybrid** (PR #41602): `MambaSpec` stores per-layer
    state as `list[Tensor]`, not a single tensor → `AttributeError`.
  - **FP8 KV with nested cache** (PR #41896): future kernel layouts may
    nest as `list[list[Tensor]]`, `tuple[Tensor]`, or
    `Mapping[str, Tensor]` (e.g. block-scaled per-head scratch).

PN55v2 walks the cache structure recursively with a depth-first
iterator and zero-s only real tensors. None / non-tensor sentinels
are silently skipped. The patch is purely additive at the call site —
existing flat `list[Tensor]` paths still work the same.

================================================================
ENV
================================================================

GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV=1 to opt in. Same flag as
v1 — operator intent is unchanged.

Default OFF: defensive backport. Active scripts don't currently
exercise sleep/wake, but external mgmt-API triggers can hit the bug.
Enable in deployment profiles where sleep/wake is part of the
operational pattern.

================================================================
RISK
================================================================

LOW.

  - Same anchor as PN55v1 (literal unchanged buggy loop). Upgrading
    in-place avoids the PN55-vs-PN83 anchor collision PR38 §3.3
    explicitly warned against.
  - Recursive iterator handles list/tuple/dict; None and non-tensor
    sentinels skipped; no .zero_() on objects without the method.
  - Idempotent (marker-protected).

================================================================
STATE
================================================================

PR38 Day 2 (2026-05-08): upgraded from PN55v1 to PN55v2. Same env
flag, registry id PN55. Sister patch PN83 explicitly NOT created
(would conflict on this anchor).

Author: Sandermage backport.
Backport reference: vllm#41602 (kevglynn / Mistral) + vllm#41896
(nested KV cache class). Genesis contribution: idempotent TextPatch,
recursive iterator design, drift markers, integration tests.
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn55_wake_up_hybrid_kv")

# Marker bumped to v2 so an old v1-applied install can be re-patched
# in place — v1 marker presence is no longer treated as idempotent
# (the new replacement has different code).
GENESIS_PN55_MARKER = "Genesis PN55v2 wake_up nested KV (vllm#41602+#41896)"
GENESIS_PN55_V1_MARKER = "Genesis PN55 wake_up hybrid KV (vllm#41602)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor: the original buggy loop. Verbatim — both PR #41602 and
# PR #41896 originate at the same site; the anchor is stable
# regardless of which fix you apply.
ANCHOR_OLD = (
    "        kv_caches = getattr(self, \"kv_caches\", [])\n"
    "        for cache_tensor in kv_caches:\n"
    "            if cache_tensor is not None:\n"
    "                cache_tensor.zero_()"
)


# Replacement: recursive iterator covering Tensor / list / tuple /
# Mapping / None / non-tensor sentinels.
ANCHOR_NEW = (
    "        kv_caches = getattr(self, \"kv_caches\", [])\n"
    "        # [Genesis PN55v2 vllm#41602+#41896] hybrid models (Mamba,\n"
    "        # DeltaNet, future block-scaled FP8 KV) store per-layer state\n"
    "        # as nested list/tuple/Mapping, not a flat list[Tensor]. The\n"
    "        # original .zero_() loop AttributeError'd on the list path\n"
    "        # (#41602) and would break again on Mapping/tuple shapes\n"
    "        # introduced by #41896. This iterator zero-s only real\n"
    "        # tensors and skips None / non-tensor sentinels.\n"
    "        from collections.abc import Mapping as _PN55_Mapping\n"
    "        def _pn55_iter(node):\n"
    "            if node is None:\n"
    "                return\n"
    "            if hasattr(node, \"zero_\") and not isinstance(\n"
    "                node, (list, tuple, _PN55_Mapping)\n"
    "            ):\n"
    "                yield node\n"
    "                return\n"
    "            if isinstance(node, _PN55_Mapping):\n"
    "                for _v in node.values():\n"
    "                    yield from _pn55_iter(_v)\n"
    "                return\n"
    "            if isinstance(node, (list, tuple)):\n"
    "                for _e in node:\n"
    "                    yield from _pn55_iter(_e)\n"
    "                return\n"
    "        for _pn55_t in _pn55_iter(kv_caches):\n"
    "            _pn55_t.zero_()"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN55v2 wake_up nested KV (vllm#41602+#41896)",
        target_file=str(target),
        marker=GENESIS_PN55_MARKER,
        sub_patches=[TextPatch(
            name="pn55_recursive_iterator",
            anchor=ANCHOR_OLD,
            replacement=ANCHOR_NEW,
            required=True,
        )],
        upstream_drift_markers=[
            # Self-marker so re-apply is idempotent.
            "[Genesis PN55v2",
            # Upstream-side: PR #41602 + #41896 both introduce one of
            # these names natively. Detect either to self-retire.
            "_iter_kv_cache_tensors",
            "init_fp8_kv_scales",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN55v2 — recursive zero of nested wake_up KV cache."""
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN55")
    log_decision("PN55", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gpu_model_runner.py not found"

    # Special-case in-place upgrade from v1 → v2: if the v1 marker is
    # present but v2 is not, the file was patched by an older Genesis
    # version. Best to leave alone — operator should rerun apply_all
    # against a pristine pin OR re-deploy. Reporting this explicitly
    # helps diagnose the rare upgrade scenario.
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        return "applied", "PN55v2 already applied (idempotent)"
    if GENESIS_PN55_V1_MARKER in content:
        return (
            "skipped",
            "PN55 v1 marker present from previous boot — file already "
            "patched with the v1 list-only replacement. Re-deploy "
            "from pristine vllm or accept v1 behavior.",
        )

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "PN55v2 applied: nested wake_up KV cache safe"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — likely upstream merged"
    return "failed", failure.reason if failure else "unknown failure"
