# SPDX-License-Identifier: Apache-2.0
"""PN97 — physical-cap on KV cache tensor allocation (Phase 7 PoC).

Architectural goal: decouple `tensor.shape[0]` (physical GPU slots) from
`pool.num_gpu_blocks` (logical addressable block IDs). This is the
missing Anchor #10 documented in Phase 5 design notes.

Without this patch, GENESIS_PN95_VIRT_ENABLE=1 causes CUDA OOM:
the pool inflates to logical=236 blocks → tensor allocation tries
236*per_block_bytes ≈ 84 GiB on a 24 GiB card → torch.OutOfMemoryError.

PN97 wraps `_allocate_kv_cache_tensors` so each KVCacheTensor.size is
capped to `available_memory * factor` (physical GPU only). Subsequent
attention forward reads against logical block_ids > physical_num_blocks
are out of scope here — that needs PN98 (block_id translation table in
attention kernel surface).

PN97 ALONE does NOT unlock 156K single-user. It is a prerequisite that
prevents the OOM crash when VIRT_ENABLE is set; the actual long-context
support requires the full PN95 demote/promote cycle to keep only the
attention-relevant blocks GPU-resident — coordinated with attention
forward through PN98 (future work).

Env gate: `GENESIS_ENABLE_PN97_TENSOR_PHYSICAL_CAP=1` (default OFF).

Anchor: vllm/v1/worker/gpu_model_runner.py::_allocate_kv_cache_tensors
on the `for kv_cache_tensor in kv_cache_config.kv_cache_tensors:` line.
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn97_tensor_physical_cap")

GENESIS_MARKER = "Genesis PN97 physical-cap KV tensor allocation (Phase 7 PoC)"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN97_TENSOR_PHYSICAL_CAP", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


PN97_OLD = (
    "        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}\n"
    "        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
    "            tensor = torch.zeros(\n"
    "                kv_cache_tensor.size, dtype=torch.int8, device=self.device\n"
    "            )\n"
)
PN97_NEW = (
    "        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}\n"
    "        # [Genesis PN97 Phase 7 PoC] cap each KVCacheTensor.size to\n"
    "        # the physical GPU memory budget. Without this, VIRT_ENABLE=1\n"
    "        # inflates logical num_blocks and vllm tries to allocate\n"
    "        # tensors sized for the inflated count — torch.OutOfMemoryError.\n"
    "        # The cap divides available bytes evenly across distinct\n"
    "        # KVCacheTensor entries (each shared by a layer-group). The\n"
    "        # remainder of the logical addressing surface is owned by\n"
    "        # PN95 L2/L3 — promote-on-miss restores blocks before access.\n"
    "        # NOTE: this is partial Phase 7 — full virtual addressing\n"
    "        # requires PN98 (attention-side block_id translation).\n"
    "        try:\n"
    "            from vllm.sndr_core.cache._pn95_runtime import pn97_physical_cap_bytes as _g_pn97_cap\n"
    "            _pn97_per_tensor_cap = _g_pn97_cap(len(kv_cache_config.kv_cache_tensors))\n"
    "        except Exception:\n"
    "            _pn97_per_tensor_cap = None\n"
    "        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
    "            _alloc_size = kv_cache_tensor.size\n"
    "            if _pn97_per_tensor_cap is not None and _alloc_size > _pn97_per_tensor_cap:\n"
    "                import logging as _g_pn97_log\n"
    "                _g_pn97_log.getLogger(\"genesis.pn97\").info(\n"
    "                    \"[PN97] capping KVCacheTensor %d bytes -> %d bytes (physical limit)\",\n"
    "                    _alloc_size, _pn97_per_tensor_cap,\n"
    "                )\n"
    "                _alloc_size = _pn97_per_tensor_cap\n"
    "            tensor = torch.zeros(\n"
    "                _alloc_size, dtype=torch.int8, device=self.device\n"
    "            )\n"
)


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN97 KV tensor physical-cap (Phase 7 PoC)",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn97_allocate_kv_cache_tensors",
                anchor=PN97_OLD,
                replacement=PN97_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN97",
            "pn97_physical_cap_bytes",
        ],
    )


def apply() -> tuple[str, str]:
    if not _enabled():
        return "skipped", "PN97 disabled (set GENESIS_ENABLE_PN97_TENSOR_PHYSICAL_CAP=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file gpu_model_runner.py not resolvable"
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
        applied_message="PN97 KV tensor physical-cap installed",
        patch_name=patcher.patch_name,
    )
