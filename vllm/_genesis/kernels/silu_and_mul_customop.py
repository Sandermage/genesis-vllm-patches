# SPDX-License-Identifier: Apache-2.0
"""PN25 — `silu_and_mul` as torch.library.custom_op (Inductor-safe pool).

Problem (continuation of PN12)
------------------------------
PN12 text-patches `SiluAndMul.forward_cuda` to acquire its
`[M, intermediate_size]` BF16/FP16 transient from `FFNIntermediateCache`
instead of `torch.empty()`. That works in eager mode.

Reported by noonghunna 2026-04-30 in club-3090#16, with confirmation
from VolandBerlioz on a real OpenCode workload (29K sys+tools prefill,
24 GB single 3090): when `custom_ops=["none"]` is the default (which is
typical under V1 `aot_compile_fullgraph`), `SiluAndMul.__call__`
dispatches to `forward_native`, NOT `forward_cuda`. `forward_native`
is

    @staticmethod
    def forward_native(x):
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

torch.compile's Inductor traces this body into a fused kernel and
issues its own `empty_strided_cuda((s18, intermediate_size), fp16)`
for the multiplication output. PN12's pool never gets a chance —
the patched `forward_cuda` method is never reached.

Symptom on a 24 GB 3090 + Lorbus 27B (intermediate=17408) at
`max_num_batched_tokens=4128`: 137.6 MiB allocation, 131.75 MiB free,
OOM. Cliff 1 mech B fires on real workloads while our verify-stress
25K synthetic happens to hit shapes that DO reach eager forward_cuda
and pass.

Genesis stack vulnerability
---------------------------
Same architectural flaw exists in our PN12 — we only patch
`forward_cuda`. Our 27B PROD configs avoid the inductor path because
`--cudagraph-mode=PIECEWISE` + offline-quant INT4 short-circuits the
compile pipeline on this kernel. But long-context + chunked prefill
under a future Inductor-default config could hit it.

Fix design (this module)
------------------------
Register `genesis::silu_and_mul_pooled` as `torch.library.custom_op`
with `device_types=("cuda",)`. Inductor treats custom ops as opaque
nodes — emits a call to the op and does NOT trace through the body.
Inside the op body we run the same eager logic as PN12's patched
forward_cuda: acquire output from `FFNIntermediateCache.acquire_silu_out`
when the [M, d] 2-D shape matches, fall back to `torch.empty` otherwise,
then dispatch to the underlying CUDA `silu_and_mul` kernel.

Companion patch PN25 (`patch_N25_silu_inductor_safe_pool.py`) edits
`SiluAndMul.forward_native` to route through this op when available.
PN12 stays as the eager-path patch on `forward_cuda`. Both can run
simultaneously without conflict — `forward_cuda` is called when
`custom_ops=["+silu_and_mul"]`, `forward_native` is called otherwise.

Composition with PN12
---------------------
PN12 patches `forward_cuda` (eager dispatch).
PN25 patches `forward_native` via opaque op (compile dispatch).

Together: both paths acquire from the same `FFNIntermediateCache`
pool. No state collision — pool is keyed by (intermediate_size, dtype,
device), and forward is strictly sequential in vLLM's schedule.

If only PN12 enabled: eager workloads work, compile path leaks.
If only PN25 enabled: compile workloads work, eager path leaks.
If both enabled: full coverage. Recommended for any inductor-heavy
config (35B FP8 + future MoE; club-3090 long-text/long-vision).

Compat
------
- Requires `torch.library.custom_op` (PyTorch ≥ 2.4, available on
  current vLLM nightly).
- Enabled via `GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE=1`. OFF by
  default.
- Falls back gracefully if torch < 2.4 OR `torch.ops._C.silu_and_mul`
  is missing (CPU-only build).

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa
Cross-engine inspiration:
  - club-3090#16 (noonghunna independent work-in-progress)
  - Genesis P7b `gdn_dual_stream_customop.py` (custom_op template)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

log = logging.getLogger("genesis.silu_and_mul_customop")

_ENV_ENABLE_PN25 = "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE"


def is_pn25_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE_PN25, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def should_apply() -> bool:
    """Platform gate: NVIDIA CUDA + env opt-in."""
    if not is_pn25_enabled():
        return False
    from vllm._genesis.guards import is_nvidia_cuda
    if not is_nvidia_cuda():
        return False
    if not torch.cuda.is_available():
        return False
    return True


_OP_QUALNAME = "genesis::silu_and_mul_pooled"
_op_registered = False


def _silu_and_mul_native_fallback(x: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch fallback when CUDA op `_C.silu_and_mul` is not present.

    Equivalent to `forward_native`. Allocates fresh — no pool benefit,
    but preserves correctness. Hit only on CPU-only builds or in tests.
    """
    import torch.nn.functional as F
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _register_op_once() -> bool:
    """Register `genesis::silu_and_mul_pooled` with torch.library.

    Idempotent. Returns True on success. On any failure (torch too old,
    op already registered by sister module, fake-impl rejection by
    dynamo) returns False and PN25 wiring will fall back to upstream
    `forward_native` body.
    """
    global _op_registered
    if _op_registered:
        return True

    try:
        custom_op = getattr(torch.library, "custom_op", None)
        if custom_op is None:
            log.info(
                "[PN25] torch.library.custom_op not available "
                "(torch<2.4) — falling back to vanilla forward_native"
            )
            return False
    except Exception as e:
        log.info("[PN25] torch.library import failed: %s", e)
        return False

    # Probe the underlying CUDA op once. If absent (CPU-only build,
    # rare), we register a pure-pytorch impl that still routes through
    # the opaque op so Inductor won't inline.
    has_cuda_op = (
        hasattr(torch.ops, "_C") and
        hasattr(torch.ops._C, "silu_and_mul")
    )

    @custom_op(_OP_QUALNAME, mutates_args=(), device_types=("cuda",))
    def _silu_and_mul_pooled(x: torch.Tensor) -> torch.Tensor:
        """Real impl — runs outside dynamo trace (opaque op).

        For 2-D `(M, 2*d)` tensors, acquires output from the shared
        `FFNIntermediateCache` pool. For 3-D `(B, S, 2*d)` we fall
        back to `torch.empty` because the pool is keyed on
        `(num_tokens, intermediate_size)` and 3-D shapes only appear
        in non-prefill paths where the alloc is small enough to not
        matter.
        """
        d = x.shape[-1] // 2

        if has_cuda_op and x.dim() == 2:
            try:
                from vllm._genesis.kernels.ffn_intermediate_cache import (
                    FFNIntermediateCache as _Cache,
                )
                if _Cache.is_production_eligible():
                    out = _Cache.acquire_silu_out(
                        num_tokens=x.shape[0],
                        intermediate_size=d,
                        dtype=x.dtype, device=x.device,
                    )
                    torch.ops._C.silu_and_mul(out, x)
                    return out
            except Exception as e:
                log.debug("[PN25] pool acquire failed, fallback: %s", e)

        if has_cuda_op:
            output_shape = x.shape[:-1] + (d,)
            out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            torch.ops._C.silu_and_mul(out, x)
            return out

        return _silu_and_mul_native_fallback(x)

    @_silu_and_mul_pooled.register_fake
    def _silu_and_mul_pooled_fake(x: torch.Tensor) -> torch.Tensor:
        """Shape-inference impl for dynamo tracing.

        Returns an empty tensor of the correct shape; dynamo never
        executes the body so this is never observed at runtime — only
        used for output shape propagation through the compiled graph.
        """
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)

    _op_registered = True
    log.info("[PN25] registered torch op %s (Inductor-opaque)", _OP_QUALNAME)
    return True


def get_op_callable():
    """Return the registered op callable, or None if registration failed.

    Used by the PN25 wiring patch to populate the replacement body.
    Caller is responsible for graceful degradation on None.
    """
    if not _register_op_once():
        return None
    try:
        return torch.ops.genesis.silu_and_mul_pooled
    except (AttributeError, RuntimeError):
        return None
