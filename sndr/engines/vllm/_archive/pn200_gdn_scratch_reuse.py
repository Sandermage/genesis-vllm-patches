# SPDX-License-Identifier: Apache-2.0
"""PN200 — GDN outer-forward scratch reuse (Tier 1.B).

Companion to PN106 (FLA chunk-kernel scratch pool). PN106 patches the
FLA inner kernels (chunk_delta_h.py h/v_new, chunk_o.py o). PN200
patches the OUTER GDN linear-attention forward in gdn_linear_attn.py
which allocates a per-step `core_attn_out` tensor that survives the
full layer forward and is the biggest single allocation churn source:

    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

Size for Qwen3.6-27B INT4 at chunk_size=4096, fp16:
  num_tokens=4096 × num_v_heads/tp=32 × head_v_dim=128 × 2 B
  = 32 MiB per call, allocated 48 times per chunked-prefill step

→ 1.5 GiB allocation traffic per step. PyTorch caching allocator
mostly reuses the slabs across calls, but variable `num_tokens` between
chunks fragments — empirically ~300 MiB "reserved but unallocated"
remains after long prefill runs.

PN200 routes core_attn_out through the PN106 named-pool API with
zero=True (the comment at gdn_linear_attn.py:763 explicitly says
`torch.empty` corrupts state — see vllm PR #28182 — so we honor the
zero contract by calling `.zero_()` on the pool slice).

Cost: ~5-15 μs of host-issued memset per call. Recovered from removed
allocation+pool churn — net wash on speed (1-3% TPS variance noise),
~500 MiB - 1 GiB GPU memory reclaimed.

Env gate: `GENESIS_ENABLE_PN200_GDN_SCRATCH_REUSE=1` (default OFF).

Architectural note: this file is the Tier-1.B piece of a three-tier
plan documented in PHASE_7_DESIGN_2026-05-13.md. Tier 1.C is the
empty_cache scheduler hook (PN201). Tier 2/3 (CPU offload of cold
prefix) builds on the per-layer KV split (PN202) + offload manager
(PN203/204) — those use the same PN106 pool API plus PN95 pinned
storage, so reuse is maximized and each tier ships in <300 LoC.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn200_gdn_scratch_reuse")

GENESIS_MARKER = "Genesis PN200 GDN outer-forward scratch pool"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN200_GDN_SCRATCH_REUSE", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor verified on vllm nightly dcacdf9a (2026-05-14).
# gdn_linear_attn.py:765 — CUDA forward path, the one our 27B INT4 hits.
PN200_CORE_ATTN_OLD = (
    "        core_attn_out = torch.zeros(\n"
    "            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "            dtype=hidden_states.dtype,\n"
    "            device=hidden_states.device,\n"
    "        )\n"
)
PN200_CORE_ATTN_NEW = (
    "        # [Genesis PN200] route core_attn_out through PN106 named-pool with\n"
    "        # zero=True. Honors the vllm PR #28182 'must be zeroed' contract\n"
    "        # via explicit .zero_() on the pool slice. Eliminates 32 MiB × 48 layer\n"
    "        # alloc traffic per chunked-prefill step.\n"
    "        core_attn_out = None\n"
    "        try:\n"
    "            import os as _g_pn200_os\n"
    "            if _g_pn200_os.environ.get(\n"
    "                \"GENESIS_ENABLE_PN200_GDN_SCRATCH_REUSE\", \"0\",\n"
    "            ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "                from sndr.cache._pn95_runtime import pn106_get_pooled_buf as _g_pn200_get\n"
    "                core_attn_out = _g_pn200_get(\n"
    "                    \"gdn_core_attn_out\",\n"
    "                    (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "                    hidden_states.dtype,\n"
    "                    hidden_states.device,\n"
    "                    zero=True,\n"
    "                )\n"
    "        except Exception:\n"
    "            core_attn_out = None\n"
    "        if core_attn_out is None:\n"
    "            core_attn_out = torch.zeros(\n"
    "                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "                dtype=hidden_states.dtype,\n"
    "                device=hidden_states.device,\n"
    "            )\n"
)


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    # K.1.R.R.4 (2026-05-29): #41126 split mamba/gdn_linear_attn.py.
    # See P7 / PN50 for the same fallback pattern.
    target = (
        resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
        or resolve_vllm_file(
            "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
        )
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN200 GDN core_attn_out pool",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn200_gdn_core_attn_out",
                anchor=PN200_CORE_ATTN_OLD,
                replacement=PN200_CORE_ATTN_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN200",
            "gdn_core_attn_out",
        ],
    )


def apply() -> tuple[str, str]:
    if not _enabled():
        return "skipped", "PN200 disabled (set GENESIS_ENABLE_PN200_GDN_SCRATCH_REUSE=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file gdn_linear_attn.py not resolvable"
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
        applied_message="PN200 GDN core_attn_out routed through pool — saves ~1 GiB alloc traffic per step",
        patch_name=patcher.patch_name,
    )
