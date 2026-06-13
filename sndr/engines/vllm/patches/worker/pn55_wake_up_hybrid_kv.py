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
SCOPE NARROWED 2026-06-11 (preflight residual triage §2)
================================================================

Upstream merged vllm#28783 (2025-11-30): the wake_up zeroing loop now
lives inside ``GPUModelRunner.init_fp8_kv_scales()`` together with
FP8 ``_k_scale``/``_v_scale`` re-init. PN55's remit is therefore the
narrowed post-#28783 corner only: **hybrid (Mamba/GDN nested-KV) +
fp8-KV + wake_up** — the flat ``.zero_()`` loop #28783 ships still
breaks on nested list/tuple/Mapping cache shapes, and #41602/#41896
(the upstream fixes for exactly that) remain OPEN (gh-verified
2026-06-11). ANCHOR_OLD still matches the relocated loop byte-exactly
(count=1 at pristine gpu_model_runner.py:947-950 on pin
0.22.1rc1.dev259+g303916e93). The ``init_fp8_kv_scales`` string was
dropped from ``upstream_drift_markers`` — it name-collides with the
merged #28783 and caused a false "upstream merged" self-retire on
every current pin.

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
iterator and zero-s only real tensors. None sentinels are skipped
silently (expected layout); non-tensor leaves are skipped WITH a
log.warning naming the offending type (2026-06-11 hygiene — an
unexpected cache layout should surface in docker logs, not vanish).
The patch is purely additive at the call site — existing flat
`list[Tensor]` paths still work the same.

================================================================
UPSTREAM REVIEW 2026-06-11 — #44778 AND COMPANION #44779
================================================================

[vllm-project/vllm#44778](https://github.com/vllm-project/vllm/pull/44778)
(terafin, OPEN, reviewed via gh pr view/diff 2026-06-11) is a
downstream backport of #41896: a module-level walker over nested KV
containers plus the same flat-loop swap inside init_fp8_kv_scales.
Functionally a re-implementation of the same fix PN55v2 already
carries (roadmap chunk-5 Theme C) — no re-vendor. Deliberate
divergences kept:

  - upstream raises TypeError on an unexpected leaf; PN55 warn-skips
    (a wedged wake_up on PROD is worse than one stale cache entry);
  - PN55 inlines the walker at the anchor site, so the
    "_iter_kv_cache_tensors" drift marker stays a pure upstream-merge
    signal (tools/lint_drift_markers.py self-collision contract).

Adopted FROM #44778: the exec-patched-text regression-test technique
(its tests/v1/worker/test_gpu_model_runner_fp8_wake_up.py drives the
real patched method CPU-only). Our unit test now applies the real
TextPatcher to a pinned fixture and execs the PATCHED source instead
of hand-mirroring the iterator — a mirrored copy could drift from
ANCHOR_NEW without failing; the exec'd text cannot.

[vllm-project/vllm#44779](https://github.com/vllm-project/vllm/pull/44779)
(same author, OPEN; fixes vllm#44395) gates
EngineCore.resume_scheduler() on `not model_executor.is_sleeping`
after wake_up, so a PARTIAL wake (e.g. tags=["weights"]) no longer
resumes scheduling into released KV memory. Verified on pin
0.22.1rc1.dev259+g303916e93: v1/engine/core.py wake_up() (line 765)
still calls resume_scheduler() unconditionally (line 779), and
is_sleeping() already composes the executor signal (line 783) — the
one-line gate would apply cleanly if ever needed.

VERDICT: NOT a prerequisite for enabling sleep/wake hot-swap on the
2x A5000 24GB rig. Both crash surfaces of #44395 require a partial
wake:

  (a) DP idle ranks running execute_dummy_batch() after a partial
      wake — we run TP=2 / DP=1, so the busy-loop surface does not
      exist on this rig;
  (b) an external request racing the window between a partial wake
      and the follow-up full wake — Genesis hot-swap issues FULL
      wakes only (sleep -> wake_up with no tags), so the window
      never opens.

Defense-in-depth only. Revisit (vendor as a small companion engine
patch) if the mgmt API ever adopts tagged/partial wake sequences
(RLHF-style weight staging). Note its behavior change before
adopting: after the gate, a partial wake leaves the scheduler PAUSED
until a follow-up full wake.

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
  - Recursive iterator handles list/tuple/dict; None skipped silently,
    non-tensor sentinels warn-skipped; no .zero_() on objects without
    the method. The warning uses gpu_model_runner.py's module-level
    `logger` (presence verified on the current pin).
  - Idempotent (marker-protected).

================================================================
STATE
================================================================

PR38 Day 2 (2026-05-08): upgraded from PN55v1 to PN55v2. Same env
flag, registry id PN55. Sister patch PN83 explicitly NOT created
(would conflict on this anchor).

2026-06-11 hygiene pass: replacement text gained the warn-skip branch
for non-tensor leaves. Marker deliberately NOT bumped — Genesis
applies against the pristine image tree at every container boot, so
the only install that keeps the older (silent-skip) v2 text is an
in-place re-apply on an already-patched file, acceptable for a
log-only delta. #44778 added to related_upstream_prs (registry).

Author: Sandermage backport.
Backport reference: vllm#41602 (kevglynn / Mistral) + vllm#41896
(nested KV cache class). Genesis contribution: idempotent TextPatch,
recursive iterator design, drift markers, integration tests.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
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
# Mapping / None / non-tensor sentinels. Relies on the module-level
# `logger = init_logger(__name__)` of gpu_model_runner.py (verified
# present on pin 0.22.1rc1.dev259+g303916e93).
ANCHOR_NEW = (
    "        kv_caches = getattr(self, \"kv_caches\", [])\n"
    "        # [Genesis PN55v2 vllm#41602+#41896] hybrid models (Mamba,\n"
    "        # DeltaNet, future block-scaled FP8 KV) store per-layer state\n"
    "        # as nested list/tuple/Mapping, not a flat list[Tensor]. The\n"
    "        # original .zero_() loop AttributeError'd on the list path\n"
    "        # (#41602) and would break again on Mapping/tuple shapes\n"
    "        # introduced by #41896. This iterator zero-s only real\n"
    "        # tensors, skips None silently, and warn-skips non-tensor\n"
    "        # leaves (upstream #44778 raises TypeError there; a wedged\n"
    "        # wake_up is worse than one stale cache entry on PROD).\n"
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
    "            logger.warning(\n"
    "                \"[Genesis PN55v2] wake_up KV walker skipped \"\n"
    "                \"non-tensor leaf of type %s; entry left \"\n"
    "                \"un-zeroed\", type(node).__name__,\n"
    "            )\n"
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
            # Upstream-side: PR #41896 introduces this helper name
            # natively. Detect it to self-retire.
            #
            # [Preflight triage 2026-06-11 §2] "init_fp8_kv_scales"
            # REMOVED from this list: the name landed in every current
            # pin via merged vllm#28783 (FP8 KV + sleep(level=2)
            # gibberish fix, MERGED 2025-11-30) — gpu_model_runner.py
            # defines init_fp8_kv_scales() natively (pristine line 936
            # on pin 0.22.1rc1.dev259) while #41602 and #41896 are both
            # still OPEN (gh-verified 2026-06-11). Keeping the marker
            # made PN55 false-skip as "upstream merged" on every pin,
            # even though our ANCHOR_OLD (the flat .zero_() loop, now
            # inside init_fp8_kv_scales at pristine 947-950) still
            # matches count=1 and the nested-KV bug is still unfixed.
            "_iter_kv_cache_tensors",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN55v2 — recursive zero of nested wake_up KV cache."""
    from sndr.dispatcher import log_decision, should_apply

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
