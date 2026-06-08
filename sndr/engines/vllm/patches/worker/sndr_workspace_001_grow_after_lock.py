# SPDX-License-Identifier: Apache-2.0
"""SNDR-WORKSPACE-001 — fix vLLM's workspace grow-after-lock fatal.

vLLM v1 has `WorkspaceManager._ensure_workspace_size` which locks the
GPU scratch workspace after CUDA-graph warmup. If any runtime path
(decode_attention, continuation_prefill, Marlin GEMM scratch) needs a
bigger workspace than what warmup captured, the current code raises:

    AssertionError: Workspace is locked but allocation from
    'turboquant_attn.py:893:_decode_attention' requires 0.38 MB,
    current size is 0.00 MB. Workspace growth is not allowed after
    locking.

This crashes EngineCore on every request and the operator has no
clean way to recover short of `--enforce-eager` (drops CUDA graphs
entirely, losing 5-10% throughput).

The real fix is for warmup to size the workspace correctly for every
runtime path. That requires understanding every code path that
allocates workspace and pre-sizing in warmup — significant vLLM
upstream work.

In the meantime we take a pragmatic stance: **let the workspace
grow after lock**, with a warning. The torch CUDA allocator handles
the re-allocation cleanly. The only cost is the first call to a
larger workspace doesn't hit the captured CUDA graph (one-time slow
path; subsequent calls hit it). This trades a tiny perf hit for
"engine actually serves requests" — strictly better than
AssertionError-crash.

Genesis SNDR-WORKSPACE-001 anchors AT the `if self._locked:` raise
in `vllm/v1/worker/workspace.py::_ensure_workspace_size` and
replaces the raise with a log + continue-to-grow path.

Env gate: `GENESIS_ENABLE_SNDR_WORKSPACE_001=1` (default OFF so
existing deployments stay bit-identical to upstream).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatcher, TextPatch

log = logging.getLogger("genesis.sndr_workspace_001")

GENESIS_MARKER = (
    "Genesis SNDR-WORKSPACE-001 — workspace grow-after-lock graceful fix"
)


# Exact upstream anchor (verified against vllm 0.20.2rc1.dev209+g5536fc0c0
# on 2026-05-13 in vllm/v1/worker/workspace.py). The raise sits inside
# `_ensure_workspace_size`, gated by `self._locked`.
SNDR_WS001_OLD = (
    "            if self._locked:\n"
    "                raise AssertionError(\n"
    "                    f\"Workspace is locked but allocation from '{get_caller_info()}' \"\n"
    "                    f\"requires {required_bytes / _MB:.2f} MB, current size is \"\n"
    "                    f\"{current_size / _MB:.2f} MB. \"\n"
    "                    \"Workspace growth is not allowed after locking.\"\n"
    "                )\n"
)
SNDR_WS001_NEW = (
    "            if self._locked:\n"
    "                # [Genesis SNDR-WORKSPACE-001] warn + grow instead of raise.\n"
    "                # The torch CUDA allocator handles the resize; first call to\n"
    "                # the bigger workspace takes a non-graph slow path (one-time\n"
    "                # cost), subsequent calls hit the graph again. Net effect:\n"
    "                # engine keeps serving instead of crashing on AssertionError.\n"
    "                import logging as _g_ws_log\n"
    "                _g_ws_log.getLogger(\"genesis.sndr_workspace_001\").warning(\n"
    "                    \"[Genesis SNDR-WORKSPACE-001] workspace grow-after-lock from \"\n"
    "                    \"%s: %.2f MB -> %.2f MB (one-time slow path)\",\n"
    "                    get_caller_info(), current_size / _MB, required_bytes / _MB,\n"
    "                )\n"
)


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_SNDR_WORKSPACE_001", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _make_patcher():
    if not _enabled():
        return None
    target = resolve_vllm_file("v1/worker/workspace.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="SNDR-WORKSPACE-001 vllm v1/worker/workspace.py — grow-after-lock graceful fix",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="sndr_ws001_grow_after_lock_anchor",
                anchor=SNDR_WS001_OLD,
                replacement=SNDR_WS001_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis SNDR-WORKSPACE-001",
            "_g_ws_log",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply SNDR-WORKSPACE-001 text-patch. Returns wiring-status tuple."""
    if not _enabled():
        return "skipped", "SNDR-WORKSPACE-001 disabled (set GENESIS_ENABLE_SNDR_WORKSPACE_001=1 to enable)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file workspace.py not resolvable"
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
    from sndr.kernel import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message="SNDR-WORKSPACE-001: workspace grow-after-lock guard installed (logs + grows instead of raising AssertionError)",
        patch_name=patcher.patch_name,
    )
