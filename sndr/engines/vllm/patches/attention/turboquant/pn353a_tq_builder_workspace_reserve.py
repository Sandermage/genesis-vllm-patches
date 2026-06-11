# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN353A — TurboQuant MetadataBuilder workspace reserve.

Backport of OPEN upstream vllm#44053 (`Bot1822`, 2026-06-04):
  ``[Bugfix][V1][TurboQuant] Reserve workspace before CUDA graph capture``
  Supersedes #40798.

================================================================
WHAT THIS PATCH DOES
================================================================

Upstream root-cause: TurboQuant lazily requests larger decode /
continuation-prefill scratch buffers via the V1 ``WorkspaceManager``
the first time a real-size batch reaches the backend. If that first
real-size batch arrives AFTER ``WorkspaceManager.lock()`` (which runs
at end of CUDA-graph capture), it triggers
``AssertionError: workspace locked, cannot grow`` on long-context
requests.

The PR reserves the max decode + continuation-prefill workspace from
``TurboQuantMetadataBuilder.__init__``, which is constructed BEFORE
CUDA-graph capture / workspace lock. After this, the lock snapshot
captures a workspace big enough for any steady-state shape.

================================================================
RELATIONSHIP TO PN118
================================================================

PN118 (backport of vllm#42551) ALSO reserves workspace, but:
  - PN118 reserves in ``TurboQuantAttentionImpl.__init__`` (the
    per-layer kernel impl), using PN118's custom
    ``WorkspaceManager.reserve(...)`` method which sizes
    every ubatch slot.
  - PN118 reserves DECODE scratch only (mid_o / output / lse).
  - PN353A reserves in ``TurboQuantMetadataBuilder.__init__``
    (one per-backend, runs earlier than per-layer Impl init),
    using the STOCK ``get_simultaneous(...)`` call.
  - PN353A reserves BOTH decode scratch AND
    continuation-prefill K/V dequant buffers.

The two compose additively:
  * PN118 covers per-ubatch slot sizing (multi-ubatch DBO support).
  * PN353A covers the additional continuation-prefill K/V buffers
    (1, num_kv_heads, max_model_len_aligned, head_size) × 2 (k+v)
    that PN118 does NOT pre-allocate.

Both reservations route through ``get_simultaneous`` /
``_ensure_workspace_size`` so each call independently grows the
workspace to ``max(prior, new_request)`` — no double-counting, no
conflict on the size dimension. On overlap, the larger reservation
wins which is the correct semantics.

================================================================
ANCHOR
================================================================

Inserts ``self._reserve_workspace()`` call + method definition into
``TurboQuantMetadataBuilder.__init__`` body. Anchor is the existing
3-line ``__init__`` that calls
``self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)``.

================================================================
SAFETY MODEL
================================================================

- LOW RISK. The only behavior change is allocating scratch buffers
  earlier. On models that already pass through this code path,
  ``get_simultaneous`` is idempotent w.r.t. final workspace size,
  so this just front-loads allocation.
- Idempotent via Genesis marker line at file head.
- Drift retreat: if any anchor isn't found, skip cleanly.

================================================================

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original: lesj0610 / Bot1822 — vllm#44053 (OPEN).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn353a_tq_builder_workspace_reserve")

GENESIS_PN353A_MARKER = (
    "Genesis PN353A TQ MetadataBuilder workspace reserve "
    "(backport: vllm#44053) v1"
)


# ─────────────────────────────────────────────────────────────────────
# ANCHOR — TurboQuantMetadataBuilder.__init__ body.
# vllm pin 0.22.1rc1.dev259+g303916e93 has:
#
#     def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
#         super().__init__(kv_cache_spec, layer_names, vllm_config, device)
#         self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
#
#     def build_for_cudagraph_capture(
#
# We inject a call to self._reserve_workspace() right after
# _init_reorder_batch_threshold(), and define the method body before
# build_for_cudagraph_capture.
# ─────────────────────────────────────────────────────────────────────

PN353A_OLD = (
    "    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):\n"
    "        super().__init__(kv_cache_spec, layer_names, vllm_config, device)\n"
    "        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)\n"
    "\n"
    "    def build_for_cudagraph_capture(\n"
)

PN353A_NEW = (
    "    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):\n"
    "        super().__init__(kv_cache_spec, layer_names, vllm_config, device)\n"
    "        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis PN353A — backport of vllm#44053]\n"
    "        # Reserve max TurboQuant decode + continuation-prefill scratch\n"
    "        # BEFORE CUDA-graph capture locks the V1 WorkspaceManager.\n"
    "        # Closes AssertionError: workspace locked, cannot grow on\n"
    "        # long-context requests with TQ + spec-decode + chunked prefill.\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        self._reserve_workspace()\n"
    "\n"
    "    def _reserve_workspace(self) -> None:\n"
    "        \"\"\"[Genesis PN353A] Pre-allocate TQ scratch via WorkspaceManager.\n"
    "\n"
    "        Sized for max steady-state shape so the lock snapshot at end\n"
    "        of CUDA-graph capture captures a workspace big enough for any\n"
    "        real request without further growth.\n"
    "        \"\"\"\n"
    "        if not is_workspace_manager_initialized():\n"
    "            return\n"
    "        # Use stock get_simultaneous which independently grows the\n"
    "        # underlying buffer to max(prior, requested). Composes with\n"
    "        # PN118.reserve() — both target the same WorkspaceManager,\n"
    "        # whichever asks for more bytes wins; no conflict.\n"
    "        import torch as _genesis_pn353a_torch\n"
    "        from vllm.utils.math_utils import round_up as _genesis_pn353a_round_up\n"
    "\n"
    "        scheduler_config = self.vllm_config.scheduler_config\n"
    "        model_config = self.vllm_config.model_config\n"
    "        parallel_config = self.vllm_config.parallel_config\n"
    "\n"
    "        max_num_reqs = scheduler_config.max_num_seqs\n"
    "        try:\n"
    "            num_heads = model_config.get_num_attention_heads(parallel_config)\n"
    "        except Exception:\n"
    "            # Drift safety: if the API changed, skip reservation\n"
    "            # gracefully — engine will fall back to lazy growth.\n"
    "            return\n"
    "        num_kv_heads = self.kv_cache_spec.num_kv_heads\n"
    "        head_size = self.kv_cache_spec.head_size\n"
    "        try:\n"
    "            max_num_splits = (\n"
    "                self.vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph\n"
    "            )\n"
    "        except AttributeError:\n"
    "            return\n"
    "\n"
    "        # Decode scratch — mirrors _decode_attention's get_simultaneous.\n"
    "        current_workspace_manager().get_simultaneous(\n"
    "            (\n"
    "                (max_num_reqs, num_heads, max_num_splits, head_size + 1),\n"
    "                _genesis_pn353a_torch.float32,\n"
    "            ),\n"
    "            ((max_num_reqs, num_heads, head_size), model_config.dtype),\n"
    "            ((max_num_reqs, num_heads), _genesis_pn353a_torch.float32),\n"
    "        )\n"
    "\n"
    "        # Continuation-prefill K/V dequant buffers — only when chunked\n"
    "        # prefill is enabled and batch is large enough to actually use\n"
    "        # the continuation path (gated by _CONTINUATION_DECODE_THRESHOLD).\n"
    "        reserve_cont = (\n"
    "            scheduler_config.enable_chunked_prefill\n"
    "            and scheduler_config.max_num_batched_tokens\n"
    "            > _CONTINUATION_DECODE_THRESHOLD\n"
    "        )\n"
    "        if not reserve_cont:\n"
    "            return\n"
    "        max_cached_len = max(0, model_config.max_model_len - 1)\n"
    "        alloc_len = _genesis_pn353a_round_up(\n"
    "            max_cached_len, self.kv_cache_spec.block_size\n"
    "        )\n"
    "        cache_buf_shape = (1, num_kv_heads, alloc_len, head_size)\n"
    "        current_workspace_manager().get_simultaneous(\n"
    "            (cache_buf_shape, _genesis_pn353a_torch.float16),\n"
    "            (cache_buf_shape, _genesis_pn353a_torch.float16),\n"
    "        )\n"
    "\n"
    "    def build_for_cudagraph_capture(\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN353A turboquant_attn.py — MetadataBuilder workspace reserve "
            "(backport vllm#44053)"
        ),
        target_file=str(target),
        marker=GENESIS_PN353A_MARKER,
        sub_patches=[
            TextPatch(
                name="pn353a_metadata_builder_reserve",
                anchor=PN353A_OLD,
                replacement=PN353A_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN353A",
            # Self-collision lint (triage plan §6 2026-06-11): former entry
            # "def _reserve_workspace" is defined verbatim by our own
            # vllm#44053 backport replacement — it cannot distinguish a
            # real upstream merge from our residue (false "upstream_merged"
            # skip, PN369 class). If upstream lands the same fix, the
            # required anchor misses → Layer 5 skip; preflight deep-diff
            # catches the merge at pin-bump time (iron rule #11).
            # FIXED 2026-06-11 (preflight triage, verified byte-level):
            # "tq_max_kv_splits_for_cuda_graph" removed — it is a
            # PRE-EXISTING pin API name (config/attention.py, read by our
            # own replacement), present in every pin since dev259, so the
            # marker fired unconditionally and PN353A could never apply
            # despite GENESIS_ENABLE_PN353A=1 (self-collision class, same
            # family as the PN369 incident).
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN353A — TQ MetadataBuilder workspace reserve."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN353A")
    log_decision("PN353A", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[PN353A] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    # Drift detection — skip if upstream already shipped equivalent.
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed vllm#44053 or equivalent",
            )

    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    return (
        "applied",
        "PN353A applied: TurboQuantMetadataBuilder.__init__ now reserves "
        "max decode + continuation-prefill scratch via stock "
        "get_simultaneous BEFORE CUDA-graph capture lock. Closes "
        "long-context AssertionError. Composes additively with PN118 "
        "(PN118 covers per-ubatch decode slots via reserve(); PN353A "
        "covers continuation-prefill K/V dequant buffers PN118 misses)."
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
