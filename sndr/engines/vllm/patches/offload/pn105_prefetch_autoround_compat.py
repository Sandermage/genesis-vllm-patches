# SPDX-License-Identifier: Apache-2.0
"""PN105 v3 — make vllm PrefetchOffloader compatible with AutoRound INT4.

Background (Qwen3.6-27B-INT4-AutoRound + vllm nightly dcacdf9a):
PrefetchOffloader iterates every parameter of an offloaded module and copies
its CPU storage into a rolling GPU StaticBufferPool slot via non_blocking H2D
on a shared copy_stream. Two correctness invariants underpin the design:

  (1) every offloaded cpu_storage MUST be pinned, so non_blocking copies do
      not implicitly synchronise the stream;
  (2) every offloaded slot MUST be refreshed each cycle, because slots are
      reused round-robin and stale data would be read as the layer's weights.

AutoRound INT4 publishes four tensors per linear: weight, g_idx, qzeros,
scales. `process_weights_after_loading` replaces param.data AFTER the initial
offload registration with newly allocated CPU tensors. Some of those tensors
(notably the int32 g_idx and qzeros with a specific strided layout) cannot be
pinned by PyTorch — `pin_memory()` raises or silently fails. That breaks
invariant (1) and the upstream assert kills startup:

  AssertionError: CPU storage for linear_attn.in_proj_qkvz.g_idx is not pinned!

v1 (commit 91a8560) replaced the assert with an inline blocking-copy fallback.
Boot succeeded but at runtime one blocking copy on the shared copy_stream
serialised every following non_blocking copy → ~24× slowdown.

v2 added a bulk pre-pin pass at start_onload entry plus an in-loop "skip if
still not pinned" branch. Boot worked, but skipping a transfer leaves the
StaticBufferPool slot holding stale data from a previous layer's weights —
silent correctness violation.

v3 (this patch) fixes the root cause: AutoRound INT4 metadata tensors are
tiny (g_idx + qzeros + scales for 27B ≈ 30-50 MB total) and there is no
reason to offload them at all. We patch `wrap_modules` to filter those names
out of the offload whitelist, so they stay resident on GPU like normal
fp16 weights. Only large fp16/quant weight tensors enter the prefetch
pipeline, and those CAN be pinned — invariant (1) holds without compromise,
the upstream assert stays in place, and full async transfer speed is
preserved.

Defence in depth: we keep the one-time bulk pre-pin pass at start_onload
entry (cheap, idempotent) to cover any future tensor type that lands on
the non-pinned path; and we keep the loop assert exactly as upstream wrote
it (no skip, no fallback) so any future regression fails loudly instead of
silently corrupting output.

Combined with PN104 (cpu_offload_gb → Prefetch backend) and Tier 1 GDN
scratch pooling (PN106/PN200/PN201), this unlocks:
  cpu_offload_gb=8 → free ~5 GB GPU → KV pool 4 GB → 9-10 GB
  → 156K-176K context on a single A5000 24 GB with full quality
  (no sliding window, no aggressive quantisation beyond fp8/TQ KV).

Env gate: GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT=1 (default OFF).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn105_prefetch_autoround_compat")

GENESIS_MARKER = "Genesis PN105 v3 PrefetchOffloader AutoRound metadata exclusion"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor 1: wrap_modules whitelist construction.
# Exclude AutoRound INT4 metadata suffixes from the offload whitelist so
# those tensors stay resident on the GPU and never enter the Prefetch
# pipeline. Single source-of-truth list; trivial to extend if a future
# quantisation scheme adds new non-pinnable suffixes.
PN105_WRAP_OLD = (
    "                if self.offload_params:\n"
    "                    whitelist = [\n"
    "                        name\n"
    "                        for name, _ in module.named_parameters()\n"
    "                        if any(f\".{p}.\" in f\".{name}.\" for p in self.offload_params)\n"
    "                    ]\n"
    "                else:\n"
    "                    whitelist = [name for name, _ in module.named_parameters()]\n"
)
PN105_WRAP_NEW = (
    "                if self.offload_params:\n"
    "                    whitelist = [\n"
    "                        name\n"
    "                        for name, _ in module.named_parameters()\n"
    "                        if any(f\".{p}.\" in f\".{name}.\" for p in self.offload_params)\n"
    "                    ]\n"
    "                else:\n"
    "                    whitelist = [name for name, _ in module.named_parameters()]\n"
    "                # [Genesis PN105 v3] Exclude AutoRound INT4 metadata\n"
    "                # tensors (g_idx, qzeros, scales) from offload — PyTorch\n"
    "                # cannot pin their specific strided int32 layout, and\n"
    "                # the upstream pipeline requires pinned CPU storage for\n"
    "                # correct async H2D. They are small (~30-50 MB total\n"
    "                # for 27B) so keeping them resident on GPU is free.\n"
    "                import os as _g_pn105_os\n"
    "                if _g_pn105_os.environ.get(\n"
    "                    \"GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT\", \"0\",\n"
    "                ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "                    _g_pn105_blacklist = (\n"
    "                        \".g_idx\", \".qzeros\", \".scales\",\n"
    "                        \".q_scale\", \".kv_scale\", \".weight_scale\",\n"
    "                        \".input_scale\", \".azp\",\n"
    "                    )\n"
    "                    _g_pn105_before = len(whitelist)\n"
    "                    whitelist = [\n"
    "                        _g_pn105_n for _g_pn105_n in whitelist\n"
    "                        if not any(_g_pn105_n.endswith(_g_pn105_s)\n"
    "                                   for _g_pn105_s in _g_pn105_blacklist)\n"
    "                    ]\n"
    "                    _g_pn105_dropped = _g_pn105_before - len(whitelist)\n"
    "                    if _g_pn105_dropped > 0:\n"
    "                        import logging as _g_pn105_log\n"
    "                        _g_pn105_log.getLogger(\"genesis.pn105\").info(\n"
    "                            \"[PN105 v3] module_index=%d: kept %d params on GPU \"\n"
    "                            \"(AutoRound INT4 metadata), offloading %d params\",\n"
    "                            module_index, _g_pn105_dropped, len(whitelist),\n"
    "                        )\n"
)

# Anchor 2: defence-in-depth bulk pre-pin pass at start_onload entry.
# Cheap idempotent guard against future tensor types landing on the
# non-pinned path. With PN105 v3's whitelist filter this is normally a
# no-op; we keep it so a regression manifests as a tiny startup pin-pass
# rather than a hard assert.
PN105_PREPIN_OLD = (
    "        assert self._buffer_pool is not None, \"Buffer pool not assigned\"\n"
    "\n"
    "        # Track if this prefetch is being captured (for _wait_for_layer logic)\n"
    "        self._prefetch_in_capture = torch.cuda.is_current_stream_capturing()\n"
)
PN105_PREPIN_NEW = (
    "        assert self._buffer_pool is not None, \"Buffer pool not assigned\"\n"
    "\n"
    "        # [Genesis PN105 v3] One-time bulk pre-pin pass.\n"
    "        # AutoRound's process_weights_after_loading replaces param.data\n"
    "        # with non-pinned CPU tensors AFTER initial offload registration.\n"
    "        # PN105 v3's whitelist filter already excludes the known\n"
    "        # non-pinnable suffixes — this pass is defence-in-depth for any\n"
    "        # other tensor that may need re-pinning. Runs once per offloader.\n"
    "        if not getattr(self, \"_g_pn105_pre_pinned\", False):\n"
    "            import os as _g_pn105_os\n"
    "            if _g_pn105_os.environ.get(\n"
    "                \"GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT\", \"0\",\n"
    "            ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "                import logging as _g_pn105_log\n"
    "                _g_pn105_lg = _g_pn105_log.getLogger(\"genesis.pn105\")\n"
    "                _g_pn105_repinned = 0\n"
    "                _g_pn105_failed = 0\n"
    "                for _g_pn105_name, _g_pn105_off in self._param_offloaders.items():\n"
    "                    _g_pn105_cs = _g_pn105_off._cpu_storage\n"
    "                    if _g_pn105_cs is None or _g_pn105_cs.is_pinned():\n"
    "                        continue\n"
    "                    try:\n"
    "                        _g_pn105_off._update_cpu_storage_from_param()\n"
    "                        if (_g_pn105_off._cpu_storage is not None\n"
    "                                and _g_pn105_off._cpu_storage.is_pinned()):\n"
    "                            _g_pn105_repinned += 1\n"
    "                        else:\n"
    "                            _g_pn105_failed += 1\n"
    "                    except Exception:\n"
    "                        _g_pn105_failed += 1\n"
    "                if _g_pn105_repinned or _g_pn105_failed:\n"
    "                    _g_pn105_lg.info(\n"
    "                        \"[PN105 v3] one-time bulk pre-pin: repinned=%d, failed=%d\",\n"
    "                        _g_pn105_repinned, _g_pn105_failed,\n"
    "                    )\n"
    "            self._g_pn105_pre_pinned = True\n"
    "\n"
    "        # Track if this prefetch is being captured (for _wait_for_layer logic)\n"
    "        self._prefetch_in_capture = torch.cuda.is_current_stream_capturing()\n"
)


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    target = resolve_vllm_file("model_executor/offloader/prefetch.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN105 v3 PrefetchOffloader AutoRound metadata exclusion",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn105_v3_whitelist_filter",
                anchor=PN105_WRAP_OLD,
                replacement=PN105_WRAP_NEW,
                required=True,
            ),
            TextPatch(
                name="pn105_v3_bulk_pre_pin",
                anchor=PN105_PREPIN_OLD,
                replacement=PN105_PREPIN_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN105",
            "_g_pn105_pre_pinned",
            "_g_pn105_blacklist",
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
    from sndr.kernel import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message="PN105 v3 — AutoRound INT4 metadata kept on GPU, large weights offload via Prefetch",
        patch_name=patcher.patch_name,
    )
