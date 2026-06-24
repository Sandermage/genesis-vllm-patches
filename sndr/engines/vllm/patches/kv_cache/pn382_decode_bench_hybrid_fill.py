# SPDX-License-Identifier: Apache-2.0
"""PN382 — override the MERGED-but-weaker vllm#45080 (DecodeBenchConnector
list/tuple KV fill) with a per-block GDN fill + the real group map.

Target: ``distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py``.
On pin 0.23.1rc1.dev301+g04c2a8dea, vllm#45080 has MERGED (the four
anchors re-derived 2026-06-24, byte-verified count==1 on the live dev301
pristine tree). The crash fix is now upstream-native; PN382 keeps the two
Genesis extensions upstream still LACKS (per-block list fill + real group
map). On the pre-merge pin (0.22.1rc1.dev259) PN382 vendored the whole
PR; see git history for that form.

WHY: ``DecodeBenchConnectorWorker._fill_blocks`` originally assumed every
layer's KV cache is a single block-indexed ``torch.Tensor`` and died with
``AttributeError: 'list' object has no attribute 'device'`` on the FIRST
decode batch for hybrid / linear-attention models — Mamba/GDN layers
register a LIST of state tensors. That made decode-TPOT-vs-depth benching
(the DecodeBenchConnector's whole purpose: fill KV with dummy values to
emulate deep prefill) IMPOSSIBLE on our GDN hybrids (Qwen3.6-35B-A3B /
27B). With PN382 the 8K/32K/128K/280K sweep profile (docs/BENCHMARKS.md,
MTP off) runs in minutes.

Upstream #45080 (now merged in dev301) splits the fill: tensors get the
block-row fill via ``_fill_block_tensor``; list/tuple caches get each
state tensor filled IN ITS ENTIRETY via ``_fill_state_tensor``
(``fill_()`` / ``normal_()``) — the WEAKER whole-pool form. PN382 now
REDIRECTS the list/tuple branch to upstream's own per-block
``_fill_block_tensor(state_tensor, block_ids)`` (see Sub-fix 4).

Genesis extensions (roadmap chunk-3 Theme D; iron rule #10 — adapt,
don't blind-copy):

1. PER-BLOCK fill for the list/tuple path. VERIFIED on the pristine
   pin: MambaSpec state tensors ARE block-indexed —
   ``v1/worker/gpu_model_runner.py`` (MambaSpec branch of the KV-cache
   initializer) builds each state tensor with
   ``target_shape = (num_blocks, *shape)``. Upstream's whole-pool fill
   would therefore clobber the recurrent state of every CONCURRENT
   request mid-sweep; PN382 fills only the requested block rows, same
   as the attention path (the upstream PR targets Kimi-Linear where the
   state buffers are per-request, hence its whole-pool shortcut).

2. REAL ``group_idx -> layer_names`` map. Upstream's
   ``register_kv_caches`` maps ALL layers to group 0. On hybrid models
   the scheduler sends per-group block ids
   (``block_ids_per_group``) — with the all-layers-group-0 map the
   Mamba pools get filled with the ATTENTION group's block ids and the
   Mamba group's own ids are silently ignored. PN382 threads the
   ``kv_cache_config`` the connector ctor already receives on this pin
   into the worker and builds the map from
   ``kv_cache_config.kv_cache_groups`` (upstream single-group fallback
   kept for a None config).

SAFETY MODEL
------------
- Opt-in: ``GENESIS_ENABLE_PN382_DECODE_BENCH_HYBRID_FILL=1`` (default
  OFF). Bench-infrastructure only: the DecodeBenchConnector is never in
  a PROD ``--kv-transfer-config``; the patch is inert unless the bench
  profile selects the connector.
- All four anchors required=True — a half-applied fill split would
  silently bench the wrong thing (PN286/PN290 half-apply lesson).
- Drift markers watch the merged form of vllm#45080: the PR's
  ``_fill_state_tensor`` / ``def _fill_block_tensor(`` helper names.
  Our emitted identifiers are ``_pn382_*`` — disjoint by construction
  (tools/lint_drift_markers contract; asserted in tests).
- MTP must be OFF for sweeps: the fill is dummy data, a drafter would
  propose from garbage state and acceptance statistics are meaningless
  (see the sweep profile note in docs/BENCHMARKS.md).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Upstream: https://github.com/vllm-project/vllm/pull/45080 (OPEN at
vendor time, 2026-06-11).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn382_decode_bench_hybrid_fill")

GENESIS_PN382_MARKER = (
    "Genesis PN382 vendor of vllm#45080 (decode-bench hybrid per-block fill) v1"
)

_CONNECTOR_REL = "distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py"

# vllm#45080 MERGED in dev301 as the WEAKER whole-pool form
# (``_fill_state_tensor`` / ``_fill_block_tensor`` are now NATIVE — they
# can no longer be upstream-merge drift markers, they'd auto-skip the
# patch we still need). PN382 now OVERRIDES the merged-but-insufficient
# upstream by redirecting the GDN list/tuple branch to per-block fill.
# The drift marker is the FULL post-fix spelling we'd see only if
# upstream LATER adopts per-block fill on the list path themselves
# (``self._fill_block_tensor(state_tensor, block_ids)``). It is a
# substring of PN382's own emitted replacement (PN399/PN346 self-
# collision class) — harmless because the kernel checks the idempotency
# marker (Layer 2) BEFORE the drift markers (Layer 3), so the drift scan
# never reads PN382's own output on re-apply; on a fresh pin where
# upstream adopted per-block fill, the marker fires and PN382 self-skips
# as genuinely obsolete. Allowlisted in tools/lint_drift_markers.
_DRIFT_MARKERS = (
    "[Genesis PN382",
    "self._fill_block_tensor(state_tensor, block_ids)",
)


# ─── Sub-fix 1: thread kv_cache_config into the worker ────────────────
PN382_WORKER_CTOR_OLD = (
    "        if role == KVConnectorRole.SCHEDULER:\n"
    "            self.connector_scheduler = DecodeBenchConnectorScheduler(vllm_config)\n"
    "        elif role == KVConnectorRole.WORKER:\n"
    "            self.connector_worker = DecodeBenchConnectorWorker(vllm_config)\n"
)
PN382_WORKER_CTOR_NEW = (
    "        if role == KVConnectorRole.SCHEDULER:\n"
    "            self.connector_scheduler = DecodeBenchConnectorScheduler(vllm_config)\n"
    "        elif role == KVConnectorRole.WORKER:\n"
    "            # [Genesis PN382 vendor of vllm#45080] hand the worker the\n"
    "            # KV cache group layout so fills can map group_idx to the\n"
    "            # group's OWN layer names instead of assuming a single\n"
    "            # all-layers group 0 (wrong on hybrid GDN models).\n"
    "            self.connector_worker = DecodeBenchConnectorWorker(\n"
    "                vllm_config, kv_cache_config\n"
    "            )\n"
)

# ─── Sub-fix 2: accept + stash the group layout in the worker ─────────
PN382_WORKER_INIT_OLD = (
    '    def __init__(self, vllm_config: "VllmConfig"):\n'
    "        self.vllm_config = vllm_config\n"
    "        self.block_size = vllm_config.cache_config.block_size\n"
    "\n"
    "        # Get fill parameters from extra config\n"
)
PN382_WORKER_INIT_NEW = (
    "    def __init__(\n"
    "        self,\n"
    '        vllm_config: "VllmConfig",\n'
    '        kv_cache_config: "KVCacheConfig | None" = None,\n'
    "    ):\n"
    "        self.vllm_config = vllm_config\n"
    "        self.block_size = vllm_config.cache_config.block_size\n"
    "        # [Genesis PN382 vendor of vllm#45080] group layout for the\n"
    "        # real group_idx mapping (None keeps the upstream behavior).\n"
    "        self._kv_cache_config = kv_cache_config\n"
    "\n"
    "        # Get fill parameters from extra config\n"
)

# ─── Sub-fix 3: real group_idx -> layer_names map ─────────────────────
PN382_GROUP_MAP_OLD = (
    "        # For simplicity, assume all layers belong to group 0 (standard attention)\n"
    "        # For MLA models with multiple groups, the metadata will handle the mapping\n"
    "        # We just need to fill the blocks specified in the metadata\n"
    "        self.group_to_layers = {0: list(kv_caches.keys())}\n"
)
PN382_GROUP_MAP_NEW = (
    "        # [Genesis PN382 vendor of vllm#45080 + Genesis extension]\n"
    "        # Build the REAL group_idx mapping from the KV cache group\n"
    "        # layout. Upstream maps every layer to group 0, which on\n"
    "        # hybrid models (GDN + full attention, e.g. Qwen3.6) fills\n"
    "        # Mamba state pools with the ATTENTION group's block ids and\n"
    "        # silently ignores the Mamba group's own ids. Fall back to\n"
    "        # the upstream single-group map when no layout was provided.\n"
    "        if self._kv_cache_config is not None:\n"
    "            self.group_to_layers = {\n"
    "                group_idx: list(group.layer_names)\n"
    "                for group_idx, group in enumerate(\n"
    "                    self._kv_cache_config.kv_cache_groups\n"
    "                )\n"
    "            }\n"
    "        else:\n"
    "            self.group_to_layers = {0: list(kv_caches.keys())}\n"
)

# ─── Sub-fix 4: redirect the hybrid list/tuple path to PER-BLOCK fill ──
#
# v2 re-anchor (2026-06-24, pin 0.23.1rc1.dev301+g04c2a8dea): upstream
# vllm#45080 MERGED — dev301 already splits the fill into
# ``_fill_block_tensor`` (per-block, tensors) and ``_fill_state_tensor``
# (WHOLE-POOL, list/tuple GDN/Mamba). Upstream's whole-pool fill is the
# WEAKER form (its target was Kimi-Linear per-request buffers); on our
# pin the GDN/Mamba state tensors are block-indexed
# ``(num_blocks, *shape)`` (verified live in dev301 gpu_model_runner.py
# MambaSpec branch: ``target_shape = (num_blocks, *shape)``), so the
# whole-pool fill would clobber the recurrent state of every CONCURRENT
# request mid-sweep. PN382 redirects the list/tuple branch to upstream's
# OWN per-block helper ``_fill_block_tensor(state_tensor, block_ids)``,
# filling only the requested block rows — same correctness as the
# attention path, no concurrent-state clobber. Anchor matches the merged
# dev301 dispatch (count==1 verified against the live pristine tree).
PN382_FILL_OLD = (
    "            elif isinstance(kv_cache, (list, tuple)) and all(\n"
    "                isinstance(t, torch.Tensor) for t in kv_cache\n"
    "            ):\n"
    "                for state_tensor in kv_cache:\n"
    "                    self._fill_state_tensor(state_tensor)\n"
)
PN382_FILL_NEW = (
    "            elif isinstance(kv_cache, (list, tuple)) and all(\n"
    "                isinstance(t, torch.Tensor) for t in kv_cache\n"
    "            ):\n"
    "                # [Genesis PN382 vendor of vllm#45080 + Genesis extension]\n"
    "                # Hybrid GDN/Mamba state tensors are block-indexed\n"
    "                # (num_blocks, *shape) on this pin, so fill only the\n"
    "                # requested block ROWS via upstream's own per-block\n"
    "                # helper instead of the whole-pool _fill_state_tensor.\n"
    "                # Whole-pool fill would clobber the recurrent state of\n"
    "                # every concurrent request mid-sweep; per-block keeps\n"
    "                # neighbouring requests intact (correctness-critical).\n"
    "                for state_tensor in kv_cache:\n"
    "                    self._fill_block_tensor(state_tensor, block_ids)\n"
)


def build_sub_patches() -> list[TextPatch]:
    """All four anchors required=True — a partial fill split would
    silently bench the wrong thing (PN286/PN290 half-apply lesson)."""
    return [
        TextPatch(
            name="pn382_worker_ctor_kv_cache_config",
            anchor=PN382_WORKER_CTOR_OLD,
            replacement=PN382_WORKER_CTOR_NEW,
            required=True,
        ),
        TextPatch(
            name="pn382_worker_init_kv_cache_config",
            anchor=PN382_WORKER_INIT_OLD,
            replacement=PN382_WORKER_INIT_NEW,
            required=True,
        ),
        TextPatch(
            name="pn382_real_group_map",
            anchor=PN382_GROUP_MAP_OLD,
            replacement=PN382_GROUP_MAP_NEW,
            required=True,
        ),
        TextPatch(
            name="pn382_hybrid_per_block_fill",
            anchor=PN382_FILL_OLD,
            replacement=PN382_FILL_NEW,
            required=True,
        ),
    ]


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_CONNECTOR_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN382 decode_bench_connector.py — hybrid list/tuple KV fill, "
            "per-block (vendor vllm#45080 + Genesis extensions)"
        ),
        target_file=str(target),
        marker=GENESIS_PN382_MARKER,
        sub_patches=build_sub_patches(),
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN382_DECODE_BENCH_HYBRID_FILL", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """Apply PN382 — vendor vllm#45080. Never raises."""
    if not _enabled():
        return "skipped", (
            "PN382 default OFF — set "
            "GENESIS_ENABLE_PN382_DECODE_BENCH_HYBRID_FILL=1 to engage. "
            "Bench infrastructure: DecodeBenchConnector crash fix for "
            "hybrid GDN models + per-block state fill + real group map "
            "(vendor of OPEN PR vllm#45080); enable for the decode-TPOT-"
            "vs-depth sweep profile (MTP off)."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN382: {_CONNECTOR_REL} not resolvable"

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001 — wiring must never raise
        log.warning("[PN382] apply() raised %s — leaving upstream", e)
        return "skipped", f"PN382 raised at apply: {e!r}"

    if result == TextPatchResult.APPLIED:
        subs = ", ".join(patcher.applied_sub_patches)
        return "applied", (
            f"PN382 applied (vendor of OPEN PR vllm#45080): "
            f"DecodeBenchConnector fills hybrid GDN/Mamba list-of-state "
            f"caches per BLOCK ROW (no concurrent-state clobber) and maps "
            f"group_idx to the group's own layers via kv_cache_config "
            f"[{subs}] — unlocks the 8K/32K/128K/280K decode-TPOT-vs-"
            f"depth sweep on Qwen3.6 hybrids."
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN382 already applied (marker present)"
    reason = failure.reason if failure else "unknown"
    detail = f" ({failure.detail})" if failure and failure.detail else ""
    if result == TextPatchResult.FAILED:
        return "failed", f"PN382 failed: {reason}{detail}"
    return "skipped", f"PN382: {reason}{detail}"


def is_applied() -> bool:
    target = resolve_vllm_file(_CONNECTOR_REL)
    if target is None:
        return False
    try:
        return GENESIS_PN382_MARKER in open(str(target), encoding="utf-8").read()
    except (OSError, UnicodeDecodeError):
        return False
