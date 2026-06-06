# SPDX-License-Identifier: Apache-2.0
"""PN134 — torch.compile fullgraph patch for PyTorch 2.11 (backport vllm#42686).

================================================================
!!! RETIRED 2026-05-15 — BENCH-VALIDATED REGRESSOR !!!
================================================================

DO NOT ENABLE on Qwen3.6-35B-A3B-FP8 / hybrid_gdn_moe.

Bench result (dev371 nightly-bf610c2f, 2× A5000 TP=2):
  baseline       : wall_TPS 211.38, TPOT 4.33 ms, TTFT 85 ms
  with PN134=1   : wall_TPS 158.00, TPOT 5.93 ms, TTFT 196 ms
  delta          : -25.2% TPS, +37% TPOT, +130% TTFT

Root cause: StorageBox.should_realize_on_reuse monkey-patch affects
the ENTIRE Inductor compilation graph (not just the attention path it
was designed for). The size-aware cost model materializes too many
intermediates for hybrid_gdn_moe layout (30 GDN layers + 11 attention
layers + 128 MoE experts), blowing Inductor's compile cache and
forcing recompilation on every batch shape variant.

Module kept on disk for future investigation on dense-attention models
where the cost model may behave correctly. Lifecycle in registry is
"retired" with retired_waiver=True.

================================================================
ORIGINAL PROBLEM STATEMENT (for context)
================================================================

vLLM issue #27828 + pytorch/pytorch#176994: the Inductor
materialization heuristic in PyTorch 2.11 does not realize useful
intermediate tensors that are reused several times. Result:

  - the residual in fused_add_rms_norm is recomputed every time
  - cascade re-computation across the whole model
  - inflated compile cache + slower forward (for torch.compile mode)

Fix landed in PyTorch 2.12 (https://github.com/pytorch/pytorch/pull/176994).
PR #42686 backports a simplified version for 2.11.

Applicable to us?
  - We are ON PyTorch 2.11 → IDEAL match
  - VLLM_USE_AOT_COMPILE=True (our default)
  - torch.compile is active on all 41 model layers
  - Without the fix — potentially redundant recompute ops

================================================================
FIX
================================================================

Monkey-patches `torch._inductor.ir.StorageBox.should_realize_on_reuse`
with a size-aware cost model:

    total_read_bytes = sum read_bytes
    output_bytes = numel * dtype_itemsize

    if total_read_bytes * (users - 1) >= output_bytes * (1 + users):
        return True  # materialize

Meaning: a tensor is realized when reading it n times costs more
than 1x write + n x read. Special cases included:
  - heavy ops (exp, sigmoid) on CPU → realize
  - large inner_fn → realize

PN134 backports via runtime monkey-patch on StorageBox.

================================================================
EXPECTED IMPACT
================================================================

  - Inductor compile cache hit rate ↑
  - First-prefill forward latency: -2..-8 ms (less recompute)
  - Memory pressure ↓ (cached intermediates)
  - Boot time: slight increase (cache warming) — amortizes

================================================================
SAFETY
================================================================

  - Only PyTorch 2.11 (2.12+ has the fix natively, 2.10- is unsupported)
  - Idempotent (flag on the StorageBox class)
  - Auto-skip when torch != 2.11

Author: Sandermage 2026-05-15. Backport vllm#42686 (OPEN).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn134_torch_compile_fullgraph_211")

GENESIS_PN134_MARKER = "Genesis PN134 torch.compile fullgraph 2.11 patch v1 (vllm#42686)"
_ENV_ENABLE = "GENESIS_ENABLE_PN134_TORCH_COMPILE_FULLGRAPH_211"
_ENV_DISABLE = "GENESIS_DISABLE_PN134_TORCH_COMPILE_FULLGRAPH_211"

_APPLIED = False
_ORIGINAL_FN: object = None


def _env_enabled() -> bool:
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    val = os.environ.get(_ENV_ENABLE, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _is_torch_211() -> bool:
    """Check if running PyTorch 2.11."""
    try:
        import torch
        ver = torch.__version__.split("+")[0]
        major, minor = ver.split(".")[:2]
        return int(major) == 2 and int(minor) == 11
    except Exception as e:
        log.warning("[PN134] torch version detection failed: %s", e)
        return False


def _patched_should_realize_on_reuse(self, users: int) -> bool:
    """Size-aware materialization heuristic backport.

    Decides if a reused tensor should be realized (materialized) vs
    inlined for re-computation. Original 2.11 heuristic is too
    conservative for cases like residual in fused_add_rms_norm.
    """
    from torch._inductor import config
    from torch._inductor.ir import Pointwise, Reduction, is_cpu
    from torch._inductor.virtualized import V

    if users > 1 and isinstance(self.data, (Pointwise, Reduction)):
        if is_cpu(self.data):
            opcount = self.data.inner_fn_opcount()
            heavy_ops = ["exp", "sigmoid"]
            if any(x in opcount.used_ops for x in heavy_ops):
                return True
        if self.has_large_inner_fn():
            return True
        # Size-aware cost model
        total_read_bytes = sum(
            V.graph.get_dep_size_hint(dep) for dep in self.get_reads()
        )
        output_bytes = (
            V.graph.sizevars.optimization_hint(self.data.get_numel(), fallback=0)
            * self.data.dtype.itemsize
        )
        if total_read_bytes > 0 and output_bytes > 0:
            return total_read_bytes * (users - 1) >= output_bytes * (1 + users)
        return self.num_reads() > config.realize_reads_threshold
    return False


_ENV_OVERRIDE_REGRESSION = "GENESIS_PN134_FORCE_DESPITE_REGRESSION"


def apply() -> tuple[str, str]:
    """Monkey-patch StorageBox.should_realize_on_reuse for PyTorch 2.11.

    RETIRED 2026-05-15: bench-validated -25% TPS regression on
    hybrid_gdn_moe. The monkey-patch affects the ENTIRE Inductor
    compilation graph, not just attention. Use of this patch requires
    an explicit override env var to make accidental enablement loud.
    """
    global _APPLIED, _ORIGINAL_FN

    if not _env_enabled():
        return "skipped", (
            f"PN134 RETIRED (bench-validated -25% TPS regression on "
            f"hybrid_gdn_moe; set {_ENV_ENABLE}=1 AND "
            f"{_ENV_OVERRIDE_REGRESSION}=1 only if testing on dense "
            f"attention models — see module docstring for bench data)"
        )

    if os.environ.get(_ENV_OVERRIDE_REGRESSION, "").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return "skipped", (
            f"PN134 enabled but {_ENV_OVERRIDE_REGRESSION} not set — "
            f"refusing to apply (PN134 is bench-validated regressor on "
            f"hybrid_gdn_moe, -25% TPS). Set "
            f"{_ENV_OVERRIDE_REGRESSION}=1 ONLY when explicitly "
            f"experimenting on dense-attention models."
        )

    if _APPLIED:
        return "applied", "PN134 already installed (idempotent)"

    if not _is_torch_211():
        try:
            import torch
            return "skipped", (
                f"PN134 only for torch 2.11 (running {torch.__version__}); "
                f"2.12+ has native fix, 2.10- doesn't need it"
            )
        except ImportError:
            return "skipped", "torch not importable"

    try:
        from torch._inductor.ir import StorageBox
    except ImportError as e:
        return "skipped", f"torch._inductor.ir.StorageBox not importable: {e}"

    if hasattr(StorageBox, "_genesis_pn134_wrapped"):
        _APPLIED = True
        return "applied", "PN134 StorageBox already wrapped (idempotent)"

    _ORIGINAL_FN = StorageBox.should_realize_on_reuse
    StorageBox.should_realize_on_reuse = _patched_should_realize_on_reuse
    StorageBox._genesis_pn134_wrapped = True
    _APPLIED = True

    log.info(
        "[PN134] installed: StorageBox.should_realize_on_reuse "
        "now uses a size-aware cost model (backport "
        "pytorch#176994 for torch 2.11). Inductor compilation "
        "should realize shared intermediates more often."
    )
    return "applied", (
        "PN134 installed: torch.compile fullgraph materialization "
        "heuristic patched on PyTorch 2.11 (vllm#42686 backport). "
        "Expected: -2..-8 ms prefill, lower inductor compile cache "
        "miss rate. Auto-skip when torch != 2.11."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_FN
    if not _APPLIED or _ORIGINAL_FN is None:
        return False
    try:
        from torch._inductor.ir import StorageBox
        StorageBox.should_realize_on_reuse = _ORIGINAL_FN  # type: ignore[assignment]
        delattr(StorageBox, "_genesis_pn134_wrapped")
        _APPLIED = False
        return True
    except (ImportError, AttributeError):
        return False
