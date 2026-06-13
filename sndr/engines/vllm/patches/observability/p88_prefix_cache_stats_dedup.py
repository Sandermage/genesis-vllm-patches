# SPDX-License-Identifier: Apache-2.0
"""P88 — prefix-cache stats retry de-duplication (rewrite of OPEN vllm#45202).

================================================================
PROBLEM (issue #43736)
================================================================

``KVCacheManager.get_computed_blocks`` records the local prefix-cache
query/hit stats at LOOKUP time (pristine
``v1/core/kv_cache_manager.py``)::

    if self.log_stats:
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_new_computed_tokens,
            preempted=request.num_preemptions > 0,
        )

A waiting request whose ``allocate_slots`` then FAILS (no free blocks)
stays in the waiting queue and repeats the lookup on a later scheduler
step — so the stats are counted once PER ATTEMPT. Under KV-pressure
burst retries (the Genesis long-context agent profile) the reported
``prefix_hit_rate`` inflates by tens of percent, poisoning the
P85 / TQ-KV A/B conclusions that read it off ``/metrics``.

================================================================
FIX — Genesis rewrite, NOT the upstream diff
================================================================

Upstream #45202 moves the record into the ~2000-line
``Scheduler.schedule()`` waiting loop. P88 instead keeps BOTH sites
inside ``kv_cache_manager.py`` (P79d-style minimal-anchor convention):

  * the LOOKUP site stashes a single pending record on
    ``self._genesis_p88_pending_stats`` (request_id, num_tokens,
    num_hits, preempted) instead of recording;
  * ``allocate_slots`` COMMITS that record exactly once, right after
    its last failure ``return None`` (so the allocation is guaranteed
    to succeed), gated on a request-id match so a stale stash from a
    different request is never consumed, and clears the slot so a
    second allocate for the same request (running-loop growth) does
    not double-record.

This is also MORE faithful than upstream for our configs: stats record
iff a real lookup happened, so ``enable_caching=False`` / no-lookup
paths record nothing (upstream's scheduler-side record can fire even
when pristine never recorded).

================================================================
SCOPE / SAFETY
================================================================

* Opt-in: ``GENESIS_ENABLE_P88_PREFIX_CACHE_STATS_DEDUP=1``
  (default_on=False in the registry — metrics-only, lands after a
  ``/metrics`` hit-rate sanity check on 35B).
* Fallback-disable when a KV connector is configured
  (``--kv-transfer-config`` / ``--kv-connector`` / LMCache env): with
  a connector driving allocation, the in-process retry de-dup does not
  model the transfer lifecycle, so we skip rather than risk a wrong
  count.
* Drift: if upstream merges #45202 it removes the lookup-site
  ``record(`` call, so the (required) LOOKUP anchor no longer matches
  and the patch self-skips cleanly — no explicit drift marker needed.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
import sys

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.p88_prefix_cache_stats_dedup")

GENESIS_P88_MARKER = (
    "Genesis P88 prefix-cache stats retry de-duplication "
    "(vllm#45202 rewrite — lookup stash + allocate commit)"
)

_TARGET_REL = "v1/core/kv_cache_manager.py"


# ─── Anchors (byte-exact vs pristine 0.22.1rc1.dev259 + the test fake) ────

# LOOKUP site — the record() call inside get_computed_blocks.
P88_LOOKUP_ANCHOR = (
    "        if self.log_stats:\n"
    "            assert self.prefix_cache_stats is not None\n"
    "            self.prefix_cache_stats.record(\n"
    "                num_tokens=request.num_tokens,\n"
    "                num_hits=num_new_computed_tokens,\n"
    "                preempted=request.num_preemptions > 0,\n"
    "            )\n"
)

# The lookup becomes a pure STASH — recording here is the bug.
P88_LOOKUP_REPLACEMENT = (
    "        if self.log_stats:\n"
    "            assert self.prefix_cache_stats is not None\n"
    "            # [Genesis P88] STASH the lookup stats; the commit happens\n"
    "            # in allocate_slots once the allocation is past its last\n"
    "            # failure return. Recording here double-counted failed\n"
    "            # scheduling retries (#43736), inflating prefix_hit_rate\n"
    "            # under KV-pressure bursts (long-ctx agent profile).\n"
    "            self._genesis_p88_pending_stats = (\n"
    "                request.request_id,\n"
    "                request.num_tokens,\n"
    "                num_new_computed_tokens,\n"
    "                request.num_preemptions > 0,\n"
    "            )\n"
)

# COMMIT site — the available-blocks gate (the LAST failure return in
# allocate_slots; verified past the last `return None`).
P88_ALLOC_COMMIT_ANCHOR = (
    "        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks\n"
    "        if num_blocks_to_allocate > available_blocks:\n"
    "            # Cannot allocate new blocks\n"
    "            return None\n"
)

P88_ALLOC_COMMIT_REPLACEMENT = (
    "        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks\n"
    "        if num_blocks_to_allocate > available_blocks:\n"
    "            # Cannot allocate new blocks\n"
    "            return None\n"
    "\n"
    "        # [Genesis P88] commit the pending prefix-cache stats exactly\n"
    "        # once, now that the allocation is past its last failure\n"
    "        # return. Match on request_id so a stale stash from a\n"
    "        # different request is never consumed; clear the slot so a\n"
    "        # second allocate for the same request (running-loop growth)\n"
    "        # does not double-record.\n"
    "        _p88_pending = getattr(self, \"_genesis_p88_pending_stats\", None)\n"
    "        if _p88_pending is not None and _p88_pending[0] == request.request_id:\n"
    "            if self.log_stats and self.prefix_cache_stats is not None:\n"
    "                self.prefix_cache_stats.record(\n"
    "                    num_tokens=_p88_pending[1],\n"
    "                    num_hits=_p88_pending[2],\n"
    "                    preempted=_p88_pending[3],\n"
    "                )\n"
    "            self._genesis_p88_pending_stats = None\n"
)


def _connector_configured() -> str | None:
    """Return a short reason string when a KV connector is configured
    (so P88 should fallback-disable), else None.

    Probes the live launch ``sys.argv`` for ``--kv-transfer-config`` /
    ``--kv-connector`` (space- or ``=``-separated) and the LMCache /
    vLLM connector env vars.
    """
    for tok in list(sys.argv):
        if tok in ("--kv-transfer-config", "--kv-connector"):
            return f"CLI flag {tok}"
        if tok.startswith("--kv-transfer-config=") or tok.startswith(
            "--kv-connector="
        ):
            return f"CLI flag {tok.split('=', 1)[0]}"
    for var in ("VLLM_KV_TRANSFER_CONFIG", "LMCACHE_CONFIG_FILE"):
        if os.environ.get(var):
            return f"env {var}"
    return None


def _make_kv_cache_manager_patcher(
    target_file: str | None = None,
) -> TextPatcher:
    """Build the two-site KVCacheManager patcher.

    ``target_file`` is injectable so the unit tests can run the de-dup
    semantics end-to-end against a synthetic-but-compilable fake; in
    production it resolves through the alternate-root seam.
    """
    if target_file is None:
        resolved = resolve_vllm_file(_TARGET_REL)
        target_file = str(resolved) if resolved is not None else _TARGET_REL
    return TextPatcher(
        patch_name=(
            "P88 v1/core/kv_cache_manager.py — prefix-cache stats retry "
            "de-duplication (vllm#45202 rewrite)"
        ),
        target_file=target_file,
        marker=GENESIS_P88_MARKER,
        sub_patches=[
            TextPatch(
                name="p88_lookup_stash",
                anchor=P88_LOOKUP_ANCHOR,
                replacement=P88_LOOKUP_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="p88_alloc_commit",
                anchor=P88_ALLOC_COMMIT_ANCHOR,
                replacement=P88_ALLOC_COMMIT_REPLACEMENT,
                required=True,
            ),
        ],
        # No explicit upstream_drift_markers: if #45202 merges it removes
        # the required LOOKUP record() anchor, so the patcher self-skips
        # with "anchor not found" — the correct drift behavior, with no
        # self-collision risk (PN369 class).
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    """Apply P88 — prefix-cache stats retry de-dup. Never raises."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P88")
    log_decision("P88", decision, reason)
    if not decision:
        return "skipped", reason

    conn = _connector_configured()
    if conn is not None:
        return "skipped", (
            f"P88 fallback-disabled: KV connector configured ({conn}); the "
            "in-process retry de-dup does not model a connector-driven "
            "allocation lifecycle"
        )

    patcher = _make_kv_cache_manager_patcher()
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "P88 applied: KVCacheManager prefix-cache stats now stash at "
            "lookup and commit once on a successful allocate_slots "
            "(request-id matched, slot cleared). Failed scheduling retries "
            "(#43736) no longer inflate prefix_hit_rate; enable_caching="
            "False / no-lookup paths record nothing (more faithful than "
            "the upstream scheduler-side rewrite)."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    """Return True iff our marker is present in the target file."""
    patcher = _make_kv_cache_manager_patcher()
    try:
        with open(patcher.target_file, encoding="utf-8") as f:
            return patcher.marker in f.read()
    except (OSError, UnicodeDecodeError):
        return False
