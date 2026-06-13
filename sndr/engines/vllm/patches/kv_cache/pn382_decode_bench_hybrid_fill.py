# SPDX-License-Identifier: Apache-2.0
"""PN382 — vendor OPEN PR vllm#45080 (DecodeBenchConnector list/tuple
KV fill) + two Genesis extensions.

Target: ``distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py``
on pin 0.22.1rc1.dev259+g303916e93 (anchors byte-verified count==1 on
the pristine tree at /private/tmp/candidate_pin_current).

WHY: ``DecodeBenchConnectorWorker._fill_blocks`` assumes every layer's
KV cache is a single block-indexed ``torch.Tensor`` and dies with
``AttributeError: 'list' object has no attribute 'device'`` on the
FIRST decode batch for hybrid / linear-attention models — Mamba/GDN
layers register a LIST of state tensors. That makes decode-TPOT-vs-
depth benching (the DecodeBenchConnector's whole purpose: fill KV with
dummy values to emulate deep prefill) IMPOSSIBLE on our GDN hybrids
(Qwen3.6-35B-A3B / 27B). With PN382 the 8K/32K/128K/280K sweep profile
(docs/BENCHMARKS.md, MTP off) runs in minutes.

Upstream #45080 splits the fill: tensors keep the block-row fill;
list/tuple caches get each state tensor filled IN ITS ENTIRETY
(``fill_()`` / ``normal_()``).

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

# Fires when vllm#45080 merges (the PR introduces these helper names);
# our emitted identifiers are _pn382_* so the markers never match our
# own output (tools/lint_drift_markers contract; asserted in tests).
_DRIFT_MARKERS = (
    "[Genesis PN382",
    "_fill_state_tensor",
    "def _fill_block_tensor(",
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

# ─── Sub-fix 4: tensor-vs-list split with per-block state fill ────────
PN382_FILL_OLD = (
    "            # Convert block_ids to tensor on device\n"
    "            block_ids_tensor = torch.tensor(\n"
    "                block_ids, dtype=torch.long, device=kv_cache.device\n"
    "            )\n"
    "\n"
    "            # Filter invalid block IDs\n"
    "            valid_mask = block_ids_tensor < kv_cache.shape[0]\n"
    "            valid_block_ids = block_ids_tensor[valid_mask]\n"
    "\n"
    "            if len(valid_block_ids) == 0:\n"
    "                continue\n"
    "\n"
    "            # Create fill values - either constant or random\n"
    "            block_shape = kv_cache.shape[1:]\n"
    "            if self.fill_std > 0:\n"
    "                # Random normal sampling\n"
    "                fill_values = torch.normal(\n"
    "                    mean=self.fill_mean,\n"
    "                    std=self.fill_std,\n"
    "                    size=(len(valid_block_ids),) + block_shape,\n"
    "                    dtype=kv_cache.dtype,\n"
    "                    device=kv_cache.device,\n"
    "                )\n"
    "            else:\n"
    "                # Constant fill value\n"
    "                fill_values = torch.full(\n"
    "                    (len(valid_block_ids),) + block_shape,\n"
    "                    self.fill_mean,\n"
    "                    dtype=kv_cache.dtype,\n"
    "                    device=kv_cache.device,\n"
    "                )\n"
    "\n"
    "            # Batch fill operation\n"
    "            kv_cache[valid_block_ids] = fill_values\n"
)
PN382_FILL_NEW = (
    "            # [Genesis PN382 vendor of vllm#45080 + Genesis extension]\n"
    "            # Attention layers store KV as one block-indexed\n"
    "            # torch.Tensor (first dim num_blocks). Hybrid GDN/Mamba\n"
    "            # layers store a LIST of state tensors; on this pin every\n"
    "            # state tensor is ALSO block-indexed (gpu_model_runner\n"
    "            # builds them as (num_blocks, *shape)), so the requested\n"
    "            # block rows are filled PER TENSOR. Upstream #45080 fills\n"
    "            # the whole state pool instead, which would clobber the\n"
    "            # recurrent state of every concurrent request; the\n"
    "            # per-block fill keeps neighbouring requests intact.\n"
    "            # Anything else is skipped with a warning.\n"
    "            if isinstance(kv_cache, torch.Tensor):\n"
    "                _pn382_targets = (kv_cache,)\n"
    "            elif isinstance(kv_cache, (list, tuple)) and kv_cache and all(\n"
    "                isinstance(t, torch.Tensor) for t in kv_cache\n"
    "            ):\n"
    "                _pn382_targets = tuple(kv_cache)\n"
    "            else:\n"
    "                logger.warning_once(\n"
    '                    "DecodeBenchConnector: skipping fill for layer %s "\n'
    '                    "whose KV cache is %s, not a tensor or a non-empty "\n'
    '                    "list/tuple of tensors.",\n'
    "                    layer_name,\n"
    "                    type(kv_cache).__name__,\n"
    "                )\n"
    "                continue\n"
    "\n"
    "            for _pn382_state in _pn382_targets:\n"
    "                # Convert block_ids to tensor on device\n"
    "                block_ids_tensor = torch.tensor(\n"
    "                    block_ids, dtype=torch.long, device=_pn382_state.device\n"
    "                )\n"
    "\n"
    "                # Filter invalid block IDs\n"
    "                valid_mask = block_ids_tensor < _pn382_state.shape[0]\n"
    "                valid_block_ids = block_ids_tensor[valid_mask]\n"
    "\n"
    "                if len(valid_block_ids) == 0:\n"
    "                    continue\n"
    "\n"
    "                # Create fill values - either constant or random\n"
    "                block_shape = _pn382_state.shape[1:]\n"
    "                if self.fill_std > 0:\n"
    "                    # Random normal sampling\n"
    "                    fill_values = torch.normal(\n"
    "                        mean=self.fill_mean,\n"
    "                        std=self.fill_std,\n"
    "                        size=(len(valid_block_ids),) + block_shape,\n"
    "                        dtype=_pn382_state.dtype,\n"
    "                        device=_pn382_state.device,\n"
    "                    )\n"
    "                else:\n"
    "                    # Constant fill value\n"
    "                    fill_values = torch.full(\n"
    "                        (len(valid_block_ids),) + block_shape,\n"
    "                        self.fill_mean,\n"
    "                        dtype=_pn382_state.dtype,\n"
    "                        device=_pn382_state.device,\n"
    "                    )\n"
    "\n"
    "                # Batch fill operation\n"
    "                _pn382_state[valid_block_ids] = fill_values\n"
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
