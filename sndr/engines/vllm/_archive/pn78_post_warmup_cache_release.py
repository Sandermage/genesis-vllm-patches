# SPDX-License-Identifier: Apache-2.0
"""PN78 — [DEPRECATED 2026-05-07] post-warmup empty_cache() wrap.

================================================================
WHY DEPRECATED
================================================================

Investigation 2026-05-07 (Genesis MEMORY_DEEP_PLAN Phase 2.1):
direct read of vllm pin source `vllm/v1/worker/gpu_model_runner.py`
found that `capture_model()` already calls
`torch.accelerator.empty_cache()` TWICE:
  - line 6213: BEFORE CG capture starts (clear pre-capture state)
  - line 6244: AFTER CG capture completes, BEFORE lock_workspace()

This means the cache release that PN78 was designed to add is
already part of upstream behavior in our pin. PN78's wrap would
add a 3rd identical call → no-op.

Additionally, even if the wrap were useful, the apply_all()-time
monkey patch would not reach worker processes. vllm uses
`VLLM_WORKER_MULTIPROC_METHOD=spawn` → workers boot a fresh
Python interpreter that never runs apply_all (which only fires in
the parent process before `exec vllm serve`). To affect workers
the patch would need to be a source-level edit to vllm core (the
pattern used by PN59 in `vllm/model_executor/layers/fla/ops/chunk.py`).

Patch is retained for documentation. `apply()` returns "skipped"
unconditionally with this explanation. No PR / no upstream effort
needed — upstream is already correct.

================================================================
ORIGINAL DESIGN INTENT (kept for reference)
================================================================

After `GPUModelRunner.capture_model()` returns, vLLM has:
  1. Loaded all model weights (Marlin/AutoRound repack temp scratch)
  2. Captured all CUDA graphs (private pool watermark fixed)
  3. Pre-allocated KV cache pool

PyTorch caching allocator does NOT proactively release back to
the OS — only on `cudaMalloc` failure or explicit `empty_cache()`.
Per `c10/cuda/CUDACachingAllocator.cpp:3786` the non-split blocks
are released back to the OS via cudaFree on empty_cache(); split
blocks are NOT released. The intent was to add ONE call after
capture_model() to release ~500 MiB - 1.5 GiB of non-split blocks
from the load+capture phase. Investigation showed upstream already
does this in capture_model itself.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference: PyTorch source `c10/cuda/CUDACachingAllocator.cpp` GC behavior;
vllm `gpu_model_runner.py:6213,6244` already calls empty_cache.
"""
from __future__ import annotations

import logging

log = logging.getLogger("genesis.wiring.pn78_post_warmup_cache_release")


def should_apply() -> bool:
    # Deprecated 2026-05-07 — see module docstring.
    return False


def apply() -> tuple[str, str]:
    """[DEPRECATED 2026-05-07] No-op. Upstream vllm pin already calls
    torch.accelerator.empty_cache() inside GPUModelRunner.capture_model
    (gpu_model_runner.py:6213 and :6244). Wrap would be redundant 3rd
    call. See module docstring for full investigation."""
    return "skipped", (
        "DEPRECATED 2026-05-07: vllm pin already calls "
        "torch.accelerator.empty_cache() inside capture_model "
        "(gpu_model_runner.py:6213 BEFORE capture, :6244 AFTER, before "
        "lock_workspace). PN78 wrap would be redundant 3rd call. Also "
        "monkey-patch wouldn't reach spawn'd workers (only source-level "
        "edits like PN59 propagate). Patch retained for documentation."
    )


def is_applied() -> bool:
    return False


def revert() -> bool:
    return False
