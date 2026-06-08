# SPDX-License-Identifier: Apache-2.0
"""PN104 — text-patch: redirect cpu_offload_gb -> Prefetch backend.

vllm's `create_offloader` selects UVA when only `cpu_offload_gb > 0`
(via backend="auto" branch). UVA is the slow path — cudaHostGetDevicePointer
+ PCIe-per-MatMul, no GPU caching. PrefetchOffloader is the fast path
(explicit cudaMemcpyAsync + static GPU buffer + lookahead prefetch_step).

PN104 text-patches `vllm/model_executor/offloader/base.py::create_offloader`
to auto-derive prefetch params from cpu_offload_gb when the env flag is
set, so the "auto" backend selector routes to prefetch instead of UVA.

Critically: text-patches modify the .py source file, which means EVERY
process that imports the module (API server, EngineCore worker, draft
workers under spec-decode) sees the patched code. Monkey-patches set
in apply_all only stick in the API server process — `_genesis_pn104_wrapped`
remains False in the spawn'd worker. Worker processes do `import vllm`
fresh and would otherwise get the original UVA path. This was empirically
confirmed today: API server marked PN104 applied, worker still used
UVA (10K request took 224s vs ~9s baseline).

Env gate: `GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT=1`.
Tunable: `GENESIS_PN104_PREFETCH_STEP` (default 2, range 1-8).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn104_offload_backend_redirect")

GENESIS_MARKER = "Genesis PN104 cpu_offload->prefetch backend redirect"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor verified on vllm nightly dcacdf9a (2026-05-13).
# `base.py::create_offloader` auto-resolve branch.
PN104_OLD = (
    "    if backend == \"auto\":\n"
    "        if prefetch.offload_group_size > 0:\n"
    "            backend = \"prefetch\"\n"
    "        elif uva.cpu_offload_gb > 0:\n"
    "            backend = \"uva\"\n"
    "        else:\n"
    "            return NoopOffloader()\n"
)
PN104_NEW = (
    "    # [Genesis PN104] redirect cpu_offload_gb -> prefetch backend.\n"
    "    # vllm's auto-resolve picks UVA when only cpu_offload_gb>0; UVA reads\n"
    "    # weights from pinned host RAM on every GEMM load -> ~24x slower than\n"
    "    # PrefetchOffloader. PN104 auto-derives prefetch params from\n"
    "    # cpu_offload_gb so this exact deployment gets the fast path.\n"
    "    # Env gate: GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT=1.\n"
    "    import os as _g_pn104_os\n"
    "    _g_pn104_on = _g_pn104_os.environ.get(\n"
    "        \"GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT\", \"0\",\n"
    "    ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\")\n"
    "    if (backend == \"auto\" and _g_pn104_on\n"
    "            and prefetch.offload_group_size == 0 and uva.cpu_offload_gb > 0):\n"
    "        try:\n"
    "            _g_pn104_num_layers = 64\n"
    "            _g_pn104_bytes_per_layer = 200 * (1 << 20)\n"
    "            _g_pn104_target = int(uva.cpu_offload_gb * (1 << 30))\n"
    "            _g_pn104_n_off = max(1, _g_pn104_target // _g_pn104_bytes_per_layer)\n"
    "            _g_pn104_n_off = min(_g_pn104_n_off, _g_pn104_num_layers - 2)\n"
    "            prefetch.offload_group_size = max(2, _g_pn104_num_layers // _g_pn104_n_off)\n"
    "            prefetch.offload_num_in_group = 1\n"
    "            _g_pn104_step = int(_g_pn104_os.environ.get(\n"
    "                \"GENESIS_PN104_PREFETCH_STEP\", \"2\"))\n"
    "            prefetch.offload_prefetch_step = max(1, min(8, _g_pn104_step))\n"
    "            backend = \"prefetch\"\n"
    "            import logging as _g_pn104_log\n"
    "            _g_pn104_log.getLogger(\"genesis.pn104\").info(\n"
    "                \"[PN104] cpu_offload_gb=%.1f -> prefetch backend \"\n"
    "                \"(group_size=%d num_in_group=%d prefetch_step=%d)\",\n"
    "                uva.cpu_offload_gb,\n"
    "                prefetch.offload_group_size,\n"
    "                prefetch.offload_num_in_group,\n"
    "                prefetch.offload_prefetch_step,\n"
    "            )\n"
    "        except Exception as _g_pn104_e:\n"
    "            import logging as _g_pn104_log\n"
    "            _g_pn104_log.getLogger(\"genesis.pn104\").warning(\n"
    "                \"[PN104] redirect failed, staying on UVA: %s\", _g_pn104_e,\n"
    "            )\n"
    "    if backend == \"auto\":\n"
    "        if prefetch.offload_group_size > 0:\n"
    "            backend = \"prefetch\"\n"
    "        elif uva.cpu_offload_gb > 0:\n"
    "            backend = \"uva\"\n"
    "        else:\n"
    "            return NoopOffloader()\n"
)


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    target = resolve_vllm_file("model_executor/offloader/base.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN104 cpu_offload -> Prefetch redirect",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn104_create_offloader_auto_redirect",
                anchor=PN104_OLD,
                replacement=PN104_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN104",
            "_g_pn104_on",
        ],
    )


def apply() -> tuple[str, str]:
    if not _enabled():
        return "skipped", "PN104 disabled (set GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file base.py not resolvable"
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
        applied_message="PN104 text-patch applied — cpu_offload now routes to Prefetch backend",
        patch_name=patcher.patch_name,
    )
