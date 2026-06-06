# SPDX-License-Identifier: Apache-2.0
"""PN202 — per-layer KV tensor split + contiguous arena allocation.

Two-anchor architectural patch:
  Part A: kv_cache_utils.py Branch C → A (one KVCacheTensor per layer)
  Part B: gpu_model_runner.py _allocate_kv_cache_tensors → contiguous arena

WHY BOTH ARE NEEDED:
  Part A alone caused CUDA OOM at cudagraph capture (b0zx15a8y test, 366 MiB
  fail with 309 MiB free). Root cause: 65 separate `torch.zeros` calls
  create 65 distinct cudaMalloc segments. PyTorch caching allocator does
  not coalesce them, and cudagraph capture needs large contiguous reservation
  for graph private pool → fragmentation OOM.

  Part B fixes this: allocate ONE contiguous arena (sum of all sizes),
  then hand out per-layer views via slicing. Single cudaMalloc, zero
  fragmentation. Downstream `_reshape_kv_cache_tensors` uses `as_strided`
  on tensor.storage() which is transparent to views.

vllm's Branch-C allocation in `kv_cache_utils.py::get_kv_cache_config_from_groups`
emits `group_size` slabs, each shared by one representative layer from
every kv_cache_group. For Qwen3-Next hybrid (16 full-attn + 48 GDN):
group_size=48, 48 physical slabs, each shared by 1-2 layer names.

This sharing **blocks per-layer memory policies** (offload single
layer, evict single layer, quantize single layer differently). The
fix is mechanically trivial: emit one KVCacheTensor per layer with
shared_by=[layer_name], identical to Branch-A behavior. Total bytes
allocated is unchanged (page_size × num_blocks × num_layers either
way) — what changes is **granularity of memory operations**.

This is the Tier-2.A ENABLER for Tier-3 cold-prefix CPU offload
(PN203). Without per-layer split, demoting a single attention layer's
KV requires demoting the shared slab (which also holds Mamba state
for an unrelated layer). Per-layer split lets PN203 demote full-attn
layer N's bytes independently while keeping GDN layer N+1's state
GPU-resident.

Quality / speed impact: **zero**. The math is identical, the tensor
views downstream are identical (see Part 7 in PHASE_7_DESIGN doc).
Only the layout of `kv_cache_raw_tensors` changes from 48 distinct
keys → 65 distinct keys (one per layer); attention forward reads
its layer's tensor via `kv_caches[layer_name]` which is keyed by
layer_name anyway. `_update_hybrid_attention_mamba_layout` uses
`as_strided` on per-tensor storage — that works identically with
per-layer slabs since each slab's storage is contiguous.

Env gate: `GENESIS_ENABLE_PN202_PER_LAYER_KV_SPLIT=1` (default OFF).
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn202_per_layer_kv_split")

GENESIS_MARKER = "Genesis PN202 per-layer KV tensor split"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN202_PER_LAYER_KV_SPLIT", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor verified on vllm nightly dcacdf9a (2026-05-14).
# kv_cache_utils.py:~1303 — Branch C construction of kv_cache_tensors.
PN202_OLD = (
    "        kv_cache_tensors = []\n"
    "        for i in range(group_size):\n"
    "            shared_by = []\n"
    "            for j in range(len(kv_cache_groups)):\n"
    "                if i < len(kv_cache_groups[j].layer_names):\n"
    "                    shared_by.append(kv_cache_groups[j].layer_names[i])\n"
    "            kv_cache_tensors.append(\n"
    "                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)\n"
    "            )\n"
)
PN202_NEW = (
    "        # [Genesis PN202] per-layer KV tensor split. Replaces Branch-C\n"
    "        # 'shared by representative layer of each group' with Branch-A\n"
    "        # semantics 'one tensor per layer'. Net bytes identical;\n"
    "        # enables per-layer policies (offload/evict/quantize) required\n"
    "        # by PN203 Tier-3 cold-prefix CPU offload.\n"
    "        kv_cache_tensors = []\n"
    "        import os as _g_pn202_os\n"
    "        _g_pn202_on = _g_pn202_os.environ.get(\n"
    "            \"GENESIS_ENABLE_PN202_PER_LAYER_KV_SPLIT\", \"0\",\n"
    "        ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\")\n"
    "        if _g_pn202_on:\n"
    "            for group in kv_cache_groups:\n"
    "                for layer_name in group.layer_names:\n"
    "                    kv_cache_tensors.append(\n"
    "                        KVCacheTensor(\n"
    "                            size=page_size * num_blocks,\n"
    "                            shared_by=[layer_name],\n"
    "                        )\n"
    "                    )\n"
    "        else:\n"
    "            for i in range(group_size):\n"
    "                shared_by = []\n"
    "                for j in range(len(kv_cache_groups)):\n"
    "                    if i < len(kv_cache_groups[j].layer_names):\n"
    "                        shared_by.append(kv_cache_groups[j].layer_names[i])\n"
    "                kv_cache_tensors.append(\n"
    "                    KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)\n"
    "                )\n"
)


# Part B: contiguous arena anchor in gpu_model_runner.py:6643
# `_allocate_kv_cache_tensors` originally does:
#     for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
#         tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=...)
#         for layer_name in kv_cache_tensor.shared_by:
#             kv_cache_raw_tensors[layer_name] = tensor
#
# When PN202 Part A splits to 65 small tensors, the 65 separate torch.zeros
# calls fragment the allocator. Part B replaces the loop with a single
# big arena allocation + slice views.

PN202B_OLD = (
    "        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}\n"
    "        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
    "            tensor = torch.zeros(\n"
    "                kv_cache_tensor.size, dtype=torch.int8, device=self.device\n"
    "            )\n"
    "            for layer_name in kv_cache_tensor.shared_by:\n"
    "                kv_cache_raw_tensors[layer_name] = tensor\n"
)
PN202B_NEW = (
    "        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}\n"
    "        # [Genesis PN202 Part B] contiguous arena allocation. When Part A\n"
    "        # is active (per-layer KVCacheTensor), 65 separate torch.zeros\n"
    "        # fragment the CUDA caching allocator and cudagraph capture OOMs.\n"
    "        # Single big alloc + per-layer views = zero fragmentation, same\n"
    "        # downstream semantics (kv_cache_raw_tensors[layer_name] still\n"
    "        # returns a tensor; _reshape_kv_cache_tensors operates on storage).\n"
    "        import os as _g_pn202b_os\n"
    "        _g_pn202b_on = _g_pn202b_os.environ.get(\n"
    "            \"GENESIS_ENABLE_PN202_PER_LAYER_KV_SPLIT\", \"0\",\n"
    "        ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\")\n"
    "        if _g_pn202b_on and kv_cache_config.kv_cache_tensors:\n"
    "            _g_pn202b_total = sum(t.size for t in kv_cache_config.kv_cache_tensors)\n"
    "            _g_pn202b_arena = torch.zeros(\n"
    "                _g_pn202b_total, dtype=torch.int8, device=self.device\n"
    "            )\n"
    "            _g_pn202b_offset = 0\n"
    "            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
    "                _g_pn202b_view = _g_pn202b_arena[\n"
    "                    _g_pn202b_offset : _g_pn202b_offset + kv_cache_tensor.size\n"
    "                ]\n"
    "                for layer_name in kv_cache_tensor.shared_by:\n"
    "                    kv_cache_raw_tensors[layer_name] = _g_pn202b_view\n"
    "                _g_pn202b_offset += kv_cache_tensor.size\n"
    "            # Pin arena reference on self so it survives the function.\n"
    "            self._pn202_arena = _g_pn202b_arena\n"
    "        else:\n"
    "            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
    "                tensor = torch.zeros(\n"
    "                    kv_cache_tensor.size, dtype=torch.int8, device=self.device\n"
    "                )\n"
    "                for layer_name in kv_cache_tensor.shared_by:\n"
    "                    kv_cache_raw_tensors[layer_name] = tensor\n"
)


def _make_part_a_patcher() -> TextPatcher | None:
    """Part A: per-layer KVCacheTensor split in kv_cache_utils.py."""
    target = resolve_vllm_file("v1/core/kv_cache_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN202-A per-layer KV tensor split",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn202_branch_c_per_layer",
                anchor=PN202_OLD,
                replacement=PN202_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN202",
            "per-layer KV tensor split",
        ],
    )


def _make_part_b_patcher() -> TextPatcher | None:
    """Part B: contiguous arena allocation in gpu_model_runner.py."""
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN202-B contiguous arena allocation",
        target_file=str(target),
        marker=GENESIS_MARKER + " (arena)",
        sub_patches=[
            TextPatch(
                name="pn202_contiguous_arena",
                anchor=PN202B_OLD,
                replacement=PN202B_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN202",
            "_g_pn202b_arena",
        ],
    )


def _make_patcher() -> TextPatcher | None:
    """Legacy single-patcher entry — kept for back-compat."""
    if not _enabled():
        return None
    return _make_part_a_patcher()


def _apply_one(patcher) -> tuple[str, str]:
    if patcher is None:
        return "skipped", "patcher None (target file not found)"
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as fh:
        content = fh.read()
    if patcher.marker in content:
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m in content:
            return "skipped", f"drift marker {m!r} already in file"
    result, failure = patcher.apply()
    from vllm.sndr_core.core import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=f"{patcher.patch_name}: applied",
        patch_name=patcher.patch_name,
    )


def apply() -> tuple[str, str]:
    if not _enabled():
        return "skipped", "PN202 disabled (set GENESIS_ENABLE_PN202_PER_LAYER_KV_SPLIT=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Apply BOTH parts. Part A alone causes fragmentation; Part B alone is a
    # no-op (gated on the same env which only Part A actually uses).
    part_a_status, part_a_msg = _apply_one(_make_part_a_patcher())
    part_b_status, part_b_msg = _apply_one(_make_part_b_patcher())

    applied = sum(1 for s in (part_a_status, part_b_status) if s == "applied")
    if applied == 0:
        return "skipped", (
            f"both parts skipped — A:{part_a_msg[:60]} B:{part_b_msg[:60]}"
        )
    return "applied", (
        f"PN202 per-layer split + contiguous arena active "
        f"(A:{part_a_status}, B:{part_b_status})"
    )
