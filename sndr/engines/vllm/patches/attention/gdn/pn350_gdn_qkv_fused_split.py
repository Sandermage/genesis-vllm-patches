# SPDX-License-Identifier: Apache-2.0
"""PN350 — text-patch integration of fused GDN Q/K/V split Triton kernel.

The kernel itself lives in
``sndr/engines/vllm/kernels/pn350_gdn_qkv_fused_split.py`` (single-launch
Triton kernel — one program per token row, 1 read + 3 writes).

This file is the text-patch wiring: replace the body of
``Qwen3_GatedDeltaNet.rearrange_mixed_qkv`` in
``vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`` with a
call into our kernel. Caller-visible behaviour is preserved:
  * Same input contract: ``mixed_qkv: [seq_len, q+k+v]``.
  * Same output contract: ``(q, k, v)`` each shaped ``[1, seq_len, heads, head]``,
    contiguous, same dtype.
  * Strict no-regression fallback: any kernel exception routes back to
    the original ``torch.cat`` based split.

Why this composes cleanly with existing GDN patches
===================================================

  * **PN340 / PN341** (MTP decode bubbles): modify ``gdn_attn.py`` +
    ``gpu_model_runner.py`` — different files. No anchor overlap.
  * **PN345** (FLA chunk autotune pruner): modifies
    ``model_executor/layers/fla/ops/chunk_*.py`` — different files.
  * **PN204** (in_proj dual-stream): modifies ``in_proj_qkvz/in_proj_ba``
    in ``forward()`` — upstream of the conv → upstream of this kernel.
    Sequential in data-flow. Composes.
  * **PN54** (post-rearrange contiguous dedup): the original
    ``rearrange_mixed_qkv`` produced contiguous outputs after
    ``torch.cat`` → ``view``. Our kernel ALSO produces contiguous
    outputs (allocated via ``torch.empty(...)`` with contiguous
    stride). PN54's ``.contiguous()`` deduplication remains a no-op
    on PN350-produced tensors. Composes.
  * **PN29** (chunk_o scale-fold): downstream in ``chunk_fwd_kernel_o``
    — different kernel entirely. Composes.
  * **P28** (gdn_core_attn pool): pool for attn output — downstream
    of this kernel. Composes.

Risk + bench expectation
========================

* Author Hopper bench: +2.65 % output tok/s end-to-end on Qwen3.6-35B-A3B.
* On Ampere SM 8.6 the kernel speedup factor (5.7× per-layer split) carries
  because pure memory-bandwidth-bound. Absolute μs savings per layer
  scale with bandwidth (768 GB/s A5000 vs 8 TB/s B200). As a fraction of
  the slower A5000 forward, end-to-end %-gain compresses to ~+1.0-1.5 %
  single-stream wall_TPS. NOT a regression risk: kernel is correctness-
  equivalent + strict no-regression fallback.
* No autotune — single config kernel, immediate launch.

Author: Sander (Sandermage / Aleksandr Barzov), Odessa, Ukraine.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn350_gdn_qkv_fused_split")

GENESIS_PN350_MARKER = (
    "Genesis PN350 fused GDN QKV split Triton kernel "
    "(SGLang#26206 + TRTLLM#12966 convergent algo) v1"
)

_TARGET_REL = "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"


# Anchor: the COMPLETE body of `rearrange_mixed_qkv` from "def rearrange_..."
# through "return query, key, value" — 35 lines verbatim. This block is
# unique in the file (the only method with this exact signature + docstring
# + cat-based split logic).
PN350_ANCHOR_OLD = (
    "    def rearrange_mixed_qkv(self, mixed_qkv):\n"
    "        \"\"\"Split packed qkv into contiguous (1, seq, heads, dim) tensors.\n"
    "\n"
    "        The original code used ``rearrange(x, \"l (h d) -> 1 l h d\", d=...)``\n"
    "        followed by ``.contiguous()`` on each tensor.  This version flattens\n"
    "        all three splits into a single buffer via ``torch.cat`` so that\n"
    "        torch.compile emits one Triton copy kernel instead of three separate\n"
    "        contiguous() calls.\n"
    "        \"\"\"\n"
    "        if mixed_qkv is None:\n"
    "            return None, None, None\n"
    "\n"
    "        seq_len = mixed_qkv.shape[0]\n"
    "        q_dim = self.key_dim // self.tp_size\n"
    "        k_dim = self.key_dim // self.tp_size\n"
    "        v_dim = self.value_dim // self.tp_size\n"
    "\n"
    "        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)\n"
    "\n"
    "        fused = torch.cat(\n"
    "            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0\n"
    "        )\n"
    "\n"
    "        q_size = seq_len * q_dim\n"
    "        k_size = seq_len * k_dim\n"
    "\n"
    "        q_contig = fused[0:q_size]\n"
    "        k_contig = fused[q_size : q_size + k_size]\n"
    "        v_contig = fused[q_size + k_size :]\n"
    "\n"
    "        query = q_contig.view(1, seq_len, -1, self.head_k_dim)\n"
    "        key = k_contig.view(1, seq_len, -1, self.head_k_dim)\n"
    "        value = v_contig.view(1, seq_len, -1, self.head_v_dim)\n"
    "\n"
    "        return query, key, value\n"
)


PN350_ANCHOR_NEW = (
    "    def rearrange_mixed_qkv(self, mixed_qkv):\n"
    "        \"\"\"Split packed qkv into contiguous (1, seq, heads, dim) tensors.\n"
    "\n"
    "        [Genesis PN350 fused GDN QKV split Triton kernel (SGLang#26206 +\n"
    "        TRTLLM#12966 convergent algo) v1] — one Triton kernel does the\n"
    "        full split + reshape in a single launch (1 read + 3 writes per\n"
    "        token row). Replaces the upstream torch.cat-based copy that\n"
    "        introduces a full-buffer memcpy + 4-5 ATen kernel launches.\n"
    "\n"
    "        Strict no-regression fallback: on any kernel exception, fall\n"
    "        back to the upstream cat-based split. Operator can disable via\n"
    "        env GENESIS_DISABLE_PN350=1 (read at call time).\n"
    "        \"\"\"\n"
    "        if mixed_qkv is None:\n"
    "            return None, None, None\n"
    "\n"
    "        seq_len = mixed_qkv.shape[0]\n"
    "        q_dim = self.key_dim // self.tp_size\n"
    "        k_dim = self.key_dim // self.tp_size\n"
    "        v_dim = self.value_dim // self.tp_size\n"
    "\n"
    "        # [PN350] Fused Triton kernel fast path. Falls back on any error.\n"
    "        import os as _g_os\n"
    "        if _g_os.environ.get(\"GENESIS_DISABLE_PN350\", \"\").strip().lower() not in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "            try:\n"
    "                from sndr.engines.vllm.kernels.pn350_gdn_qkv_fused_split import (\n"
    "                    pn350_fused_qkv_split as _pn350_split,\n"
    "                )\n"
    "                num_q = self.key_dim // self.head_k_dim // self.tp_size\n"
    "                num_k = self.key_dim // self.head_k_dim // self.tp_size\n"
    "                num_v = self.value_dim // self.head_v_dim // self.tp_size\n"
    "                return _pn350_split(\n"
    "                    mixed_qkv,\n"
    "                    num_q_heads=num_q, num_k_heads=num_k, num_v_heads=num_v,\n"
    "                    head_q=self.head_k_dim, head_k=self.head_k_dim,\n"
    "                    head_v=self.head_v_dim,\n"
    "                )\n"
    "            except Exception as _g_exc:  # noqa: BLE001\n"
    "                # Strict no-regression: log once, fall back to upstream path.\n"
    "                import logging as _g_log\n"
    "                _g_pn350_logger = _g_log.getLogger(\"genesis.pn350_fallback\")\n"
    "                if not getattr(self.__class__, \"_g_pn350_warned\", False):\n"
    "                    _g_pn350_logger.warning(\n"
    "                        \"PN350 fused QKV split failed (%s); falling back \"\n"
    "                        \"to upstream cat-based split for this layer.\",\n"
    "                        _g_exc,\n"
    "                    )\n"
    "                    self.__class__._g_pn350_warned = True\n"
    "        # ─── Upstream cat-based fallback ───────────────────────────\n"
    "        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)\n"
    "\n"
    "        fused = torch.cat(\n"
    "            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0\n"
    "        )\n"
    "\n"
    "        q_size = seq_len * q_dim\n"
    "        k_size = seq_len * k_dim\n"
    "\n"
    "        q_contig = fused[0:q_size]\n"
    "        k_contig = fused[q_size : q_size + k_size]\n"
    "        v_contig = fused[q_size + k_size :]\n"
    "\n"
    "        query = q_contig.view(1, seq_len, -1, self.head_k_dim)\n"
    "        key = k_contig.view(1, seq_len, -1, self.head_k_dim)\n"
    "        value = v_contig.view(1, seq_len, -1, self.head_v_dim)\n"
    "\n"
    "        return query, key, value\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN350", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    if _env_disabled():
        return "skipped", "PN350 disabled via GENESIS_DISABLE_PN350=1"

    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return "skipped", f"PN350: target file {_TARGET_REL} not found"

    patcher = TextPatcher(
        patch_name="PN350 qwen_gdn_linear_attn.py — fused GDN QKV split (SGLang#26206 + TRTLLM#12966)",
        target_file=str(target),
        marker=GENESIS_PN350_MARKER,
        sub_patches=[
            TextPatch(
                name="pn350_fused_qkv_split_integration",
                anchor=PN350_ANCHOR_OLD,
                replacement=PN350_ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN350",
        ],
    )

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN350 apply raised {e!r}"

    if result == TextPatchResult.FAILED:
        return "failed", f"PN350 FAILED — {failure.reason if failure else 'unknown'}"
    if result == TextPatchResult.SKIPPED:
        return "skipped", f"PN350 skipped — {failure.reason if failure else 'unknown'}"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN350 idempotent (already applied)"

    return "applied", (
        "PN350 applied: Qwen3_GatedDeltaNet.rearrange_mixed_qkv now uses "
        "fused Triton kernel (1 read + 3 writes per token row, single "
        "launch). Replaces torch.cat-based split (1 full-buffer copy + "
        "4-5 ATen launches). Convergent design from SGLang #26206 + "
        "TRT-LLM #12966. Expected: +1-2 % decode TPS on Qwen3.6-35B-A3B "
        "single-stream. Strict no-regression fallback on kernel error. "
        "Composes with PN340+PN341+PN345+PN54+PN29+PN204+P28."
    )


def is_applied() -> bool:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return False
    try:
        return GENESIS_PN350_MARKER in Path(str(target)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
