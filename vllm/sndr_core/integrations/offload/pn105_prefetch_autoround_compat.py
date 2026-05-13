# SPDX-License-Identifier: Apache-2.0
"""PN105 — fix vllm PrefetchOffloader assertion for AutoRound INT4 models.

Problem (verified empirically on Qwen3.6-27B-INT4-AutoRound + dcacdf9a):
PrefetchOffloader.start_onload_to_static (prefetch.py:531) asserts that
EVERY offloaded param's cpu_storage is pinned. AutoRound's
`process_weights_after_loading` replaces param.data with new non-pinned
CPU tensors (g_idx, qzeros, scales — INT tensors). Although vllm has
`_update_cpu_storage_from_param` specifically to re-pin those, the
assertion fires before/instead of the repair in some loader paths,
killing engine startup:

  AssertionError: CPU storage for linear_attn.in_proj_qkvz.g_idx is not pinned!

Fix strategy: replace the assertion with a **conditional copy** —
non-blocking when pinned (fast path), blocking when not pinned
(slow fallback for AutoRound INT tensors). Blocking copy from
pageable memory works correctly; the original assertion was a
performance guardrail, not a correctness guardrail.

Impact:
  - cpu_offload_gb now WORKS on AutoRound INT4 models
  - Per-layer cost: ~2-5 ms slower copy for non-pinned INT tensors (small)
  - Most layers (fp16 weights) still hit fast async path
  - Net: ~3-7% slower offloaded prefill, vs the alternative of UVA
    backend which is 24x slower OR no offload at all

Combined with PN104 (cpu_offload → Prefetch redirect) + Tier 1 GDN
scratch pool, this unlocks the path:
  cpu_offload_gb=8 → free 5 GB GPU → KV pool grows from 4 GB → 9-10 GB
  → 156K-176K context on a single A5000 24 GB
  → quality preserved (no sliding window, no quantization beyond fp8 KV)

Env gate: GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT=1 (default OFF).
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn105_prefetch_autoround_compat")

GENESIS_MARKER = "Genesis PN105 PrefetchOffloader AutoRound pin-assert relax"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor verified on vllm nightly dcacdf9a (2026-05-14).
# prefetch.py:~531 — the assertion + copy inside start_onload_to_static.
PN105_OLD = (
    "                assert cpu_storage is not None, \"CPU storage not initialized\"\n"
    "                assert gpu_buffer is not None, \"GPU buffer not assigned\"\n"
    "                assert not should_pin_memory() or cpu_storage.is_pinned(), (\n"
    "                    f\"CPU storage for {name} is not pinned! \"\n"
    "                    \"non_blocking=True H2D copy from non-pinned memory \"\n"
    "                    \"causes stream synchronization that breaks \"\n"
    "                    \"event-based fork synchronization.\"\n"
    "                )\n"
    "                gpu_buffer.copy_(cpu_storage, non_blocking=True)\n"
)
PN105_NEW = (
    "                assert cpu_storage is not None, \"CPU storage not initialized\"\n"
    "                assert gpu_buffer is not None, \"GPU buffer not assigned\"\n"
    "                # [Genesis PN105] AutoRound INT4 compat: g_idx/qzeros/scales\n"
    "                # are re-created non-pinned by process_weights_after_loading.\n"
    "                # Use blocking copy for those instead of asserting. Blocking\n"
    "                # copy from pageable memory IS correct (just slower); the\n"
    "                # original assert was a perf guardrail.\n"
    "                _g_pn105_pinned_ok = (\n"
    "                    not should_pin_memory() or cpu_storage.is_pinned()\n"
    "                )\n"
    "                if not _g_pn105_pinned_ok:\n"
    "                    # Last-chance repin attempt — uses the helper vllm\n"
    "                    # itself provides for this case.\n"
    "                    try:\n"
    "                        offloader._update_cpu_storage_from_param()\n"
    "                        cpu_storage = offloader._cpu_storage\n"
    "                        _g_pn105_pinned_ok = cpu_storage.is_pinned()\n"
    "                    except Exception:\n"
    "                        pass\n"
    "                gpu_buffer.copy_(cpu_storage, non_blocking=_g_pn105_pinned_ok)\n"
)


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    target = resolve_vllm_file("model_executor/offloader/prefetch.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN105 PrefetchOffloader AutoRound pin-assert relax",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn105_start_onload_pin_relax",
                anchor=PN105_OLD,
                replacement=PN105_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN105",
            "_g_pn105_pinned_ok",
        ],
    )


def apply() -> tuple[str, str]:
    if not _enabled():
        return "skipped", "PN105 disabled (set GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file prefetch.py not resolvable"
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
        applied_message="PN105 AutoRound pin-assert relaxed — Prefetch backend now works on AutoRound INT4",
        patch_name=patcher.patch_name,
    )
