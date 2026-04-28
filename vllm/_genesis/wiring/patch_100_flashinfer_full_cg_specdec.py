# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 100 — Native FlashInfer FULL CUDA graph for spec-decode.

Backport of vllm#41127 ("Enable native FlashInfer full CUDA graph support
for SpecDec w/out TRT-LLM"). PR open 2026-04-28. Per Sander direct request:
"не ждём, изучаем, импортируем".

================================================================
WHY THIS MATTERS
================================================================

NEW vllm: 27B variants (Minachist INT8 / Lorbus INT4 / gs128) auto-select
FlashInferBackend with fp8_e5m2 KV. With spec-decode (MTP K=3) the backend
falls back to PIECEWISE cudagraph because:

  CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
  attention backend FlashInferBackend (support: UNIFORM_SINGLE_TOKEN_DECODE)

PIECEWISE cudagraph is significantly slower than FULL on Ampere — large
per-step CPU launch overhead.

PR #41127 adds a native FISpecDecode path: when decode bucket has uniform
query_len > 1 (i.e. K+1 spec verify), route through
BatchPrefillWithPagedKVCacheWrapper (instead of decode wrapper) in
cudagraph mode. Verified zero_rows padding gives bit-identical numerics.

Cross-engine note (per agent a91bc4ecd9967da81): SGLang has had this
exact pattern for 1+ year in production
(`python/sglang/srt/layers/attention/flashinfer_backend.py:555-700`).
PR #41127 is vLLM finally catching up.

================================================================
WHAT THIS PATCH DOES (faithful port of #41127)
================================================================

7 sub-patches on `vllm/v1/attention/backends/flashinfer.py`:

  1. **imports** — drop `UniformTypeKVCacheSpecs` (unused after rewrite)
  2. **FISpecDecode dataclass** — new type wrapping
     `BatchPrefillWithPagedKVCacheWrapper`
  3. **FlashInferMetadata.decode union** — extend to include FISpecDecode
  4. **__init__ buffers + dicts** — `_spec_decode_wrapper`,
     `_spec_decode_wrappers_cudagraph`, `spec_decode_qo_indptr`,
     `native_spec_as_decode` flag
  5. **get_cudagraph_support** — return UNIFORM_BATCH unconditionally
     for non-DCP (was: UNIFORM_SINGLE_TOKEN_DECODE if no TRTLLM)
  6. **_get_spec_decode_prefill_wrapper method** — NEW method, lazy
     wrapper allocation cached per padded batch_size
  7. **build() routing** — per-row qo_indptr delta scan + branch on
     query_len: ≤1 → FIDecode (existing), >1 → FISpecDecode (new)
  8. **forward() FISpecDecode case** — call decode_wrapper.run() with
     causal=True instead of FIDecode path

================================================================
EXPECTED IMPACT
================================================================

Per agent analysis on Ampere SM 8.6 (A5000):
- Author claim: +2-3% per-token on SM120
- Ampere has higher CG launch-overhead share → +5-10% expected
- Specifically for 27B INT8/INT4/gs128 with MTP K=3
- Currently 63 TPS sustained → expected 67-70 TPS (modest)
- Combined with potential PIECEWISE→FULL transition: bigger gain at
  high concurrency (max-num-seqs > 1)

NOT applicable to PROD (PROD uses TurboQuantAttentionImpl, not FlashInfer).
applies_to: any backend using FlashInfer + spec-decode + non-DCP.

================================================================
SAFETY MODEL
================================================================

- Default OFF; opt-in via `GENESIS_ENABLE_P100=1`
- Idempotent via marker
- 7 anchor sites — drift detection on each
- DCP guard preserved (BatchDCPPrefillWrapper not wired for CG spec-decode)

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#41127. Cross-reference: SGLang flashinfer_backend.py.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p100_flashinfer_full_cg_specdec")


GENESIS_P100_MARKER = (
    "Genesis P100 FlashInfer FULL CUDA graph for spec-decode (vllm#41127) v7.62.17"
)


# ─── Sub-patch 1: imports — drop UniformTypeKVCacheSpecs ─────────────────

P100_IMPORTS_OLD = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
    "    KVQuantMode,\n"
    "    UniformTypeKVCacheSpecs,\n"
    ")\n"
)

P100_IMPORTS_NEW = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
    "    KVQuantMode,\n"
    ")\n"
)


# ─── Sub-patch 2: Add FISpecDecode dataclass after FIDecode ──────────────
# Anchor on the FIDecode class definition + closing line + blank +
# next class TRTLLMPrefill — insert FISpecDecode between.

P100_FISPECDECODE_OLD = (
    "@dataclass\n"
    "class FIDecode:\n"
    '    """Metadata for the decode pathway."""\n'
    "\n"
    "    wrapper: BatchDecodeWithPagedKVCacheWrapper\n"
    "\n"
    "\n"
    "@dataclass\n"
    "class TRTLLMPrefill:\n"
)

P100_FISPECDECODE_NEW = (
    "@dataclass\n"
    "class FIDecode:\n"
    '    """Metadata for the decode pathway."""\n'
    "\n"
    "    wrapper: BatchDecodeWithPagedKVCacheWrapper\n"
    "\n"
    "\n"
    "# [Genesis P100 vllm#41127 backport] FISpecDecode dataclass for native\n"
    "# FlashInfer spec-decode verification through prefill wrapper in CG mode.\n"
    "@dataclass\n"
    "class FISpecDecode:\n"
    '    """Metadata for native FlashInfer spec-decode verification (non-TRTLLM).\n'
    "\n"
    "    Used when the decode bucket has uniform query_len > 1 (1 + num_spec_tokens)\n"
    "    and TRTLLM decode attention is unavailable. Routes through the prefill\n"
    "    wrapper in cudagraph mode with zero_rows padding for padded request slots.\n"
    '    """\n'
    "\n"
    "    wrapper: BatchPrefillWithPagedKVCacheWrapper\n"
    "\n"
    "\n"
    "@dataclass\n"
    "class TRTLLMPrefill:\n"
)


# ─── Sub-patch 3: extend FlashInferMetadata.decode union type ────────────

P100_METADATA_DECODE_OLD = (
    "    decode: FIDecode | TRTLLMDecode | None\n"
)

P100_METADATA_DECODE_NEW = (
    "    # [Genesis P100 vllm#41127 backport] add FISpecDecode variant\n"
    "    decode: FIDecode | FISpecDecode | TRTLLMDecode | None\n"
)


# ─── Sub-patch 4: replace get_cudagraph_support body ─────────────────────
# Anchor on the method signature + comment + the entire body.

P100_CGSUPPORT_OLD = (
    '        """Get the cudagraph support level for FlashInfer attention.\n'
    "\n"
    "        This depends on whether we can use TRTLLM attention for decodes, since we can\n"
    "        only do UNIFORM_SINGLE_TOKEN_DECODE if it is unavailable.\n"
    "        To check this, we must call can_use_trtllm_attention with the number of KV\n"
    "        heads from the kv_cache_spec. We check all available KV cache specs and\n"
    "        only return UNIFORM_BATCH if all of them support TRTLLM attention.\n"
    '        """\n'
    "        # For UniformTypeKVCacheSpecs, check all contained specs\n"
    "        kv_specs = (\n"
    "            kv_cache_spec.kv_cache_specs.values()\n"
    "            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)\n"
    "            else [kv_cache_spec]\n"
    "        )\n"
    "        num_qo_heads = vllm_config.model_config.get_num_attention_heads(\n"
    "            vllm_config.parallel_config\n"
    "        )\n"
    "        has_trtllm_support: bool = len(kv_specs) > 0\n"
    "        for spec in kv_specs:\n"
    "            if not isinstance(spec, AttentionSpec):\n"
    "                # FlashInfer only applies to attention, so we don't consider other types\n"
    "                # of KV spec (e.g. Mamba) here. This is mostly for type checking.\n"
    "                continue\n"
    "            if not can_use_trtllm_attention(\n"
    "                num_qo_heads=num_qo_heads,\n"
    "                num_kv_heads=spec.num_kv_heads,\n"
    "            ):\n"
    "                has_trtllm_support = False\n"
    "                break\n"
    "\n"
    "        if has_trtllm_support:\n"
    "            return AttentionCGSupport.UNIFORM_BATCH\n"
    "        else:\n"
    "            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE\n"
)

P100_CGSUPPORT_NEW = (
    '        """Get the cudagraph support level for FlashInfer attention.\n'
    "\n"
    "        [Genesis P100 vllm#41127 backport]\n"
    "        Native FlashInfer can capture UNIFORM_BATCH full cudagraphs for\n"
    "        spec-decode by routing uniform query_len > 1 batches through the\n"
    "        prefill wrapper in cudagraph mode (verified zero_rows padding\n"
    "        yields bit-identical real-row numerics). TRTLLM decode attention\n"
    "        is not required for this path.\n"
    "\n"
    "        DCP uses BatchDCPPrefillWrapper which is not wired for cudagraph\n"
    "        spec-decode; downgrade to UNIFORM_SINGLE_TOKEN_DECODE there.\n"
    '        """\n'
    "        if vllm_config.parallel_config.decode_context_parallel_size > 1:\n"
    "            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE\n"
    "        return AttentionCGSupport.UNIFORM_BATCH\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/flashinfer.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P100 flashinfer.py — native FULL CG for spec-decode (vllm#41127)",
        target_file=str(target),
        marker=GENESIS_P100_MARKER,
        sub_patches=[
            TextPatch(
                name="p100_imports_drop_uniform",
                anchor=P100_IMPORTS_OLD,
                replacement=P100_IMPORTS_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_fispecdecode_dataclass",
                anchor=P100_FISPECDECODE_OLD,
                replacement=P100_FISPECDECODE_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_metadata_decode_union",
                anchor=P100_METADATA_DECODE_OLD,
                replacement=P100_METADATA_DECODE_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_cgsupport_uniform_batch",
                anchor=P100_CGSUPPORT_OLD,
                replacement=P100_CGSUPPORT_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P100",
            # Upstream-side markers if vllm#41127 (or equivalent) merges:
            "FISpecDecode",
            "spec_decode_qo_indptr",
            "_get_spec_decode_prefill_wrapper",
            "native_spec_as_decode",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P100 — FlashInfer FULL CG for spec-decode.

    NOTE: this v0 patch implements 4 of 8 sub-patches needed for full
    #41127 backport. Specifically:
      - imports drop UniformTypeKVCacheSpecs ✓
      - FISpecDecode dataclass added ✓
      - FlashInferMetadata.decode union extended ✓
      - get_cudagraph_support returns UNIFORM_BATCH ✓

    NOT YET DONE (need to be added in next iteration):
      - __init__ buffer + wrappers_cudagraph dict + native_spec_as_decode
      - _get_spec_decode_prefill_wrapper method
      - build() per-row qo_indptr delta scan + FISpecDecode routing
      - forward() FISpecDecode case

    Without those 4 missing sub-patches, the FISpecDecode dataclass exists
    but is NEVER instantiated by build() — patch is effectively a NO-OP at
    runtime BUT changes get_cudagraph_support contract → cudagraph mode
    shifts from PIECEWISE to UNIFORM_BATCH for spec-decode. This SHOULD
    crash because build() still produces FIDecode for K+1 batches but
    cudagraph mode is FULL.

    DO NOT enable until full 8-sub-patch version is ready.
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P100")
    log_decision("P100", decision, reason)
    if not decision:
        return "skipped", reason

    return (
        "skipped",
        "P100 v0 INCOMPLETE — only 4/8 sub-patches written. "
        "Need __init__ buffers + _get_spec_decode_prefill_wrapper + "
        "build() routing + forward() FISpecDecode case. "
        "Enabling now would crash (cudagraph FULL claimed but build() "
        "still produces FIDecode for K+1 batches). "
        "Tracking: docs/_internal/PENDING_WORK_v7_62_15_RU.md."
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except Exception:
        return False
