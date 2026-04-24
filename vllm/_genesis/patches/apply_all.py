# SPDX-License-Identifier: Apache-2.0
"""Genesis patches orchestrator — applies all enabled patches with defensive guards.

This module replaces the monolithic `patch_genesis_unified.py` orchestration.
It applies each Genesis patch through the 5-layer defensive guard model:

  Layer 1: File exists           → resolve_vllm_file() → skip if None
  Layer 2: Idempotency marker    → grep target file / module attr → skip if already applied
  Layer 3: Upstream merged       → upstream_compat markers → skip if present
  Layer 4: Vendor/chip compat    → is_nvidia_cuda(), is_sm_at_least() → skip on mismatch
  Layer 5: Model/backend arch    → runtime conditional skip where applicable

Each patch reports one of three outcomes:
  - applied:  The patch was wired into the running process.
  - skipped:  Platform/config means this patch is inapplicable (benign).
  - failed:   Something went wrong (missing anchor, import error, etc.).

Usage
-----
From container entrypoint (docker-compose.staging.yml / .yml):

    entrypoint: ["/bin/bash", "-c"]
    command: |
        python3 -m vllm._genesis.patches.apply_all
        exec vllm serve ...

Or standalone for diagnostics:

    $ python3 -m vllm._genesis.patches.apply_all

Exit codes:
  0 — All patches either applied or skipped cleanly (success)
  1 — At least one patch FAILED (anchor miss, unexpected error)
  2 — Setup error (vllm not importable, etc.)

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("genesis.apply_all")


# ═══════════════════════════════════════════════════════════════════════════
#                          ORCHESTRATION STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PatchResult:
    """Outcome of a single patch attempt."""
    name: str
    status: str           # "applied" | "skipped" | "failed"
    reason: str = ""      # short explanation


@dataclass
class PatchStats:
    """Accumulates per-run statistics for reporting."""
    results: list[PatchResult] = field(default_factory=list)

    @property
    def applied(self) -> list[PatchResult]:
        return [r for r in self.results if r.status == "applied"]

    @property
    def skipped(self) -> list[PatchResult]:
        return [r for r in self.results if r.status == "skipped"]

    @property
    def failed(self) -> list[PatchResult]:
        return [r for r in self.results if r.status == "failed"]

    @property
    def applied_count(self) -> int:
        return len(self.applied)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def failed_count(self) -> int:
        return len(self.failed)

    def summary(self) -> dict[str, Any]:
        return {
            "applied": self.applied_count,
            "skipped": self.skipped_count,
            "failed": self.failed_count,
            "details": {
                "applied": [(r.name, r.reason) for r in self.applied],
                "skipped": [(r.name, r.reason) for r in self.skipped],
                "failed": [(r.name, r.reason) for r in self.failed],
            },
        }

    def __str__(self) -> str:
        return (
            f"Results: {self.applied_count} applied, "
            f"{self.skipped_count} skipped, {self.failed_count} failed"
        )


# ═══════════════════════════════════════════════════════════════════════════
#                           PATCH REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

# Each patch function returns a PatchResult describing the outcome.
PATCH_REGISTRY: list[tuple[str, Callable[[], PatchResult]]] = []


def register_patch(name: str):
    """Decorator to register a patch function."""
    def decorator(fn: Callable[[], PatchResult]) -> Callable[[], PatchResult]:
        PATCH_REGISTRY.append((name, fn))
        return fn
    return decorator


def _applied(name: str, reason: str = "") -> PatchResult:
    return PatchResult(name=name, status="applied", reason=reason)


def _skipped(name: str, reason: str) -> PatchResult:
    return PatchResult(name=name, status="skipped", reason=reason)


def _failed(name: str, reason: str) -> PatchResult:
    return PatchResult(name=name, status="failed", reason=reason)


# ═══════════════════════════════════════════════════════════════════════════
#                       PATCH IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

# Module-level state: are we in dry-run or apply mode for this run?
# Set by run(apply=True/False). Dry-run only diagnoses; apply performs the
# actual text-patch / monkey-patch wiring.
_APPLY_MODE: bool = False


def _wiring_text_patch(name: str, wiring_module_name: str) -> PatchResult:
    """Generic helper for dry-run / live dispatch of a text-patch wiring module."""
    try:
        import importlib
        mod = importlib.import_module(f"vllm._genesis.wiring.{wiring_module_name}")
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: wiring ready (pass apply=True to execute)")

    try:
        status, reason = mod.apply()
    except Exception as e:
        return _failed(name, f"wiring raised (should not happen): {e}")

    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P8 KV hybrid reporting (per-token capacity)")
def apply_patch_8_kv_hybrid_reporting() -> PatchResult:
    """Patch 8: KV cache capacity reporting for hybrid (attention+Mamba) models.

    CRITICAL PRIORITY — closes the 3.76× KV-cache gap between v7.0 and the
    prod v5.14.1 monolith on Qwen3.6-35B-A3B. Without this, Mamba groups
    get counted in the per-token divisor, under-reporting capacity by the
    mamba-to-attn group ratio (2× on Qwen3.6, 4× on Nemotron-H-class).

    Mirrors upstream PR #40384 (jhsmith409 + Sandermage co-author). Two
    text-patches across `kv_cache_utils.py` + `scheduler.py`.
    """
    return _wiring_text_patch(
        "P8 KV hybrid reporting (per-token capacity)",
        "patch_8_kv_hybrid_reporting",
    )


@register_patch("P3 TurboQuant BF16->FP8 cast (Ampere fix)")
def apply_patch_3_tq_bf16_cast() -> PatchResult:
    """Patch 3: bf16→fp16→fp8 cast guard for Ampere Triton FP8_E4B15 path.

    Without this, bf16 model weights crash the TurboQuant key-store kernel
    inside Triton's `convert_custom_float8_sm80` (SM<89 only accepts fp16/fp32).
    Platform guard: NVIDIA CUDA SM 8.0+.
    """
    return _wiring_text_patch(
        "P3 TurboQuant BF16->FP8 cast (Ampere fix)",
        "patch_3_tq_bf16_cast",
    )


@register_patch("P6 TurboQuant-aware attention page size")
def apply_patch_6_tq_block_size() -> PatchResult:
    """Patch 6: use TQFullAttentionSpec in platforms/interface.py for hybrid
    alignment — avoids over-sized page calc for TQ packed layout (PR #39931)."""
    return _wiring_text_patch(
        "P6 TurboQuant-aware attention page size",
        "patch_6_tq_block_size_align",
    )


@register_patch("P15 Qwen3 None/null tool arg parser")
def apply_patch_15_qwen3_none_null() -> PatchResult:
    """Patch 15: accept both `null` and `none` in qwen3coder tool parser
    (PR #38996). Critical for Qwen3.6 with `--tool-call-parser qwen3_coder`."""
    return _wiring_text_patch(
        "P15 Qwen3 None/null tool arg parser",
        "patch_15_qwen3_none_null",
    )


@register_patch("P12 Qwen3 <tool_call> implicit reasoning end")
def apply_patch_12_tool_call_reasoning() -> PatchResult:
    """Patch 12: Treat `<tool_call>` as an implicit end-of-reasoning marker.

    Upstream PR #35687 (pending). Qwen3.5/3.6 models sometimes emit
    `<tool_call>` INSIDE `<think>` without closing with `</think>`. Without
    this patch, the whole tool invocation stays trapped as reasoning and
    the serving layer never triggers the tool call.

    Scope: ADDITIVE — adds tool_call token IDs + three serving-layer hook
    methods (is_reasoning_end / is_reasoning_end_streaming /
    extract_content_ids). Does NOT rewrite extract_reasoning body to avoid
    conflict with P27 (BEFORE-THINK). That rewrite is deferred until
    upstream #35687 lands and both can be retired together.

    Platform: vendor-agnostic (pure Python parser).
    Model: Qwen3-family only — NOT applied to DeepSeek-V3 / Kimi / others
    (different reasoning parser).
    """
    return _wiring_text_patch(
        "P12 Qwen3 <tool_call> implicit reasoning end",
        "patch_12_tool_call_reasoning",
    )


@register_patch("P27 Qwen3 BEFORE-THINK fallback")
def apply_patch_27_reasoning_before_think() -> PatchResult:
    """Patch 27: Preserve BEFORE-THINK text as content instead of dropping it.

    Fixes quality regressions (#40699-class) where the Qwen3 reasoning parser
    partitions on `<think>` and discards the text BEFORE it. Pre-reasoning
    scaffolding or summaries emitted by the model are lost in both streaming
    and non-streaming paths.

    Platform compatibility: vendor-agnostic (pure Python parser logic).
    Model compatibility: Qwen3-family only (--reasoning-parser qwen3).
    DeepSeek-V3 and other families use different parsers and are untouched.
    """
    return _wiring_text_patch(
        "P27 Qwen3 BEFORE-THINK fallback",
        "patch_27_reasoning_before_think",
    )


@register_patch("P34 Mamba zero-collapse deadlock guard")
def apply_patch_34_mamba_deadlock_guard() -> PatchResult:
    """Patch 34: Fix permanent scheduling deadlock in hybrid Mamba models
    with multiple large multimodal inputs.

    Mirrors upstream open PR #40757 (fanghao566) / #40709 (anishesg).
    Root cause: `_mamba_block_aligned_split` in scheduler truncates
    `num_new_tokens` to 0 when the gap between two adjacent images is
    smaller than `block_size`; scheduler then loops forever on a
    "0 tokens to process" request.

    CRITICAL for our prod (Qwen3.5-35B-A3B + OpenWebUI multimodal).

    Self-retires when upstream PR #40757 or #40709 merges via
    `upstream_drift_markers = ["aligned = num_new_tokens // block_size * block_size"]`.
    """
    return _wiring_text_patch(
        "P34 Mamba zero-collapse deadlock guard",
        "patch_34_mamba_deadlock_guard",
    )


@register_patch("P29 tool parser IndexError guard")
def apply_patch_29_tool_parser_index_guard() -> PatchResult:
    """Patch 29: Defensive IndexError guard in qwen3coder tool parser.

    Historical bug: `self.streamed_args_for_tool[self.current_tool_index]`
    could raise IndexError when the serving layer processed tools faster
    than the parser tracked them. Baseline v7.0 vLLM already contains
    bounded-index guards at the relevant call sites (lines 609-616, 659-666,
    436-438 of qwen3coder_tool_parser.py). This patch VERIFIES upstream
    acceptance and no-ops if the guards are already in place.

    Scope: the guard we would add is already present in the baseline image
    via upstream PRs. The patch remains registered so that future vLLM
    upgrades where the guard regresses are automatically re-applied.
    """
    name = "P29 tool parser IndexError guard"
    try:
        from vllm._genesis.guards import resolve_vllm_file
    except Exception as e:
        return _failed(name, f"guards import failed: {e}")

    target = resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")
    if target is None:
        return _skipped(name, "qwen3coder_tool_parser.py not found")

    try:
        with open(target) as f:
            content = f.read()
    except Exception as e:
        return _skipped(name, f"read_error: {e}")

    # Upstream-merged detection: all three guarded sites must be present.
    has_streamed_guard = (
        "streamed_args_for_tool out of sync" in content
        and "self.current_tool_index < len(self.streamed_args_for_tool)" in content
    )
    has_positions_guard = (
        "if self.current_tool_index >= len(tool_start_positions)" in content
    )

    if has_streamed_guard and has_positions_guard:
        return _applied(
            name,
            "upstream already contains bounded-index guards (no-op)",
        )

    # Baseline image does not have the guards → we would apply them, but for
    # v7.0 the baseline DOES have them, so this path is unreachable on the
    # supported image. Keep the branch for forward-compat.
    return _skipped(
        name,
        "upstream guards absent; text-patch for this regression path not "
        "shipped in v7.0 (reimplement when upstream regresses)",
    )


@register_patch("P23 Marlin FP32_REDUCE env override")
def apply_patch_23_marlin_fp32_reduce() -> PatchResult:
    """Patch 23: NEW in v7.0. Expose `VLLM_MARLIN_FP32_REDUCE` env var plus
    auto-select (disable on SM<90, keep on SM>=90). Kernel-level helper only
    — does NOT yet wire into Marlin launcher (needs upstream coordination or
    additional text-patch on fused_marlin_moe.py)."""
    name = "P23 Marlin FP32_REDUCE env override"
    try:
        from vllm._genesis.kernels.marlin_fp32_reduce import (
            should_disable_fp32_reduce,
            log_decision,
        )
    except Exception as e:
        return _failed(name, f"kernel import failed: {e}")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: kernel helper ready")

    from vllm._genesis.guards import is_nvidia_cuda
    if not is_nvidia_cuda():
        return _skipped(name, "non-NVIDIA — no Marlin path")

    log_decision()  # writes a structured log line
    disabled = should_disable_fp32_reduce()
    return _applied(
        name,
        f"decision: fp32_reduce disabled={disabled} "
        f"(requires upstream wire into Marlin launcher to take effect)",
    )


@register_patch("P4 TurboQuant hybrid model support")
def apply_patch_4_tq_hybrid() -> PatchResult:
    """Patch 4: Remove TurboQuant NotImplementedError for hybrid models.

    Unblocks Qwen3.6-35B-A3B (hybrid attention+mamba) + turboquant_k8v4, which
    was the blocker of v7.0 integration gate 1 (2026-04-24).

    The fix replaces the unconditional raise at `engine/arg_utils.py:1648-1668`
    with branching that:
      - For non-hybrid: keeps upstream behavior (standard boundary skip).
      - For hybrid: identifies full-attention layers via model config
        conventions (layer_types / layers_block_type / attn_type_list), applies
        TQ only to those. Mamba layers naturally skip KV cache.

    Platform guard: NVIDIA CUDA (upstream TQ is CUDA-only).

    Wiring strategy: TEXT-PATCH at the source file. Must run BEFORE vllm
    imports arg_utils — i.e. invoke via `python3 -m vllm._genesis.patches.
    apply_all` as a pre-step to `vllm serve`. Idempotent; safe on container
    recreate (re-applies on fresh image layer).
    """
    name = "P4 TurboQuant hybrid model support"

    if not _APPLY_MODE:
        # Dry-run: just confirm the wiring module is importable.
        try:
            from vllm._genesis.wiring import patch_4_tq_hybrid
            assert callable(patch_4_tq_hybrid.apply)
        except Exception as e:
            return _failed(name, f"wiring import failed: {e}")
        return _applied(name, "dry-run: wiring ready (pass apply=True to execute)")

    # Real apply path: run the text-patcher.
    try:
        from vllm._genesis.wiring import patch_4_tq_hybrid
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_4_tq_hybrid.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P5 KV cache page size unification")
def apply_patch_5_page_size() -> PatchResult:
    """Patch 5: LCM-padding for KV cache page size unification.

    Unblocks TurboQuant + hybrid models at KV cache init. Without this,
    `unify_kv_cache_spec_page_size()` raises NotImplementedError when
    attention + mamba page sizes are not mutually divisible. Concrete case:
    TQ page=12416 vs DeltaNet mamba page≈12.6MiB — NOT divisible, crash.

    Fix uses `math.lcm` to pad max page UP to nearest multiple of LCM of
    smaller sizes. Overhead <0.1% typical.

    Phase 3 integration test (2026-04-24) hit this AFTER P4 fixed the
    TQ+hybrid validator. P5 is the SECOND-HOP blocker on the path.
    """
    name = "P5 KV cache page size unification"

    if not _APPLY_MODE:
        try:
            from vllm._genesis.wiring import patch_5_page_size
            assert callable(patch_5_page_size.apply)
        except Exception as e:
            return _failed(name, f"wiring import failed: {e}")
        return _applied(name, "dry-run: wiring ready (pass apply=True to execute)")

    try:
        from vllm._genesis.wiring import patch_5_page_size
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_5_page_size.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P5b KV page-size pad-smaller-to-max (env-opt-in)")
def apply_patch_5b_page_size_pad_smaller() -> PatchResult:
    """Patch 5b: pad-SMALLER-to-max KV page-size strategy (alt to P5 v1).

    Frees ~34% per-block VRAM vs P5 v1 LCM-pad-up on Qwen3.6-35B-A3B
    hybrid. Ships env-gated (`GENESIS_ENABLE_P5B=1`) because the
    blast-radius is the KV-cache allocator sizing semantics — operators
    MUST bench GSM8K + long-context regression on VM 100 before
    enabling in prod.

    The precursor attempt (P5 v2) crashed on TurboQuant reshape
    mismatch. P5b adds `real_page_size_bytes` companion + helper
    resolution (`compute_real_page_size_bytes` /
    `clamp_to_real_shape`) in `kernels/page_size_padded.py` so the
    kernel can consult the natural (un-padded) size even when the
    allocator reserves padded blocks.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0 (shared with TurboQuant).
    """
    name = "P5b KV page-size pad-smaller-to-max (env-opt-in)"
    from vllm._genesis.guards import is_nvidia_cuda, is_amd_rocm, is_cpu_only

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no TurboQuant KV layer")
        return _skipped(name, "non-NVIDIA platform")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: env-opt-in scaffold ready")

    try:
        from vllm._genesis.wiring import patch_5b_page_size_pad_smaller
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_5b_page_size_pad_smaller.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P31 MoE router fp32 softmax")
def apply_patch_31_router_softmax() -> PatchResult:
    """Patch 31: Universal fp32 upcast for MoE router softmax.

    Applies to all GPU vendors — pure-torch primitive. CPU is a no-op in
    practice (no benefit), but doesn't fail.

    Wiring strategy: The callable is made available as
    `vllm._genesis.kernels.router_softmax.router_softmax`. At vLLM engine
    init, the Genesis integration layer (loaded lazily via upstream_compat
    hooks) replaces the upstream `torch.softmax(gating_output, dim=-1)`
    call sites with this function.

    For v7.0-dev, we verify the kernel is importable and report readiness.
    The actual monkey-patch binding happens when vLLM's MoE modules import.
    """
    name = "P31 MoE router fp32 softmax"
    from vllm._genesis.guards import is_cpu_only

    if is_cpu_only():
        return _skipped(
            name,
            "CPU-only platform; fp32 upcast has no numerical benefit here",
        )

    try:
        from vllm._genesis.kernels.router_softmax import router_softmax
        assert callable(router_softmax)
    except Exception as e:
        return _failed(name, f"router_softmax import failed: {e}")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: kernel ready (pass apply=True for live wiring)")

    # Live wiring: wrap grouped_topk router (limited scope — only affects
    # grouped-MoE families; Qwen3.6 uses fused-CUDA-kernel softmax that's
    # out of scope for Python-level rebind).
    try:
        from vllm._genesis.wiring import patch_31_router_softmax
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_31_router_softmax.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P22 TurboQuant shared dequant prealloc")
def apply_patch_22_tq_dequant_prealloc() -> PatchResult:
    """Patch 22: Pre-allocate TurboQuant K/V dequant buffers during profile_run.

    Fixes #40420-class OOM at long context: without this patch, dequant buffers
    are allocated lazily inside forward() → invisible to vLLM's memory profiler
    → KV cache over-sized → OOM when a real 234k+ request arrives.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0 (TurboQuant is CUDA-only upstream).

    Wiring strategy: `ensure_turboquant_buffers(impl, layer, device)` is called
    from inside `TurboQuantAttentionImpl._ensure_on_device` via monkey-patch.
    We verify manager is importable and platform-compatible here.
    """
    name = "P22 TurboQuant shared dequant prealloc"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )
    from vllm._genesis.kernels.dequant_buffer import (
        TurboQuantBufferManager, ensure_turboquant_buffers,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported to AMD")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no TurboQuant kernel")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — TurboQuant requires Ampere+")

    if not TurboQuantBufferManager.should_apply():
        return _skipped(name, "platform guard returned False")

    assert callable(ensure_turboquant_buffers)

    if not _APPLY_MODE:
        return _applied(name, "dry-run: kernel ready (pass apply=True for live wiring)")

    # Live wiring: rebind TurboQuantAttentionImpl._ensure_on_device.
    try:
        from vllm._genesis.wiring import patch_22_tq_prealloc
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_22_tq_prealloc.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P26 TurboQuant prefill output prealloc")
def apply_patch_26_prefill_output() -> PatchResult:
    """Patch 26: Pre-allocate the prefill path's output tensor (Opt 4).

    `TurboQuantAttentionImpl._prefill_attention` line 566 does
    `output = torch.zeros(N, Hq, D, ...)` per call + line 575 does a fresh
    `_cu_2 = torch.zeros(2, ..., int32)`. Both are profiler-invisible and
    cost ~1-2% decode TGS on long-context (same root-cause class as
    #40420).

    Fix: text-patch both call-sites onto `TurboQuantBufferManager.
    acquire_prefill_output()` and `.acquire_cu_2()` — pointer-stable
    pools reserved during profile_run. Safety net: both helpers fall
    back to fresh `torch.zeros` on platform-incompatible / budget
    overflow, so correctness is preserved on any platform.

    Platform guard: shared with P22 (NVIDIA CUDA + SM ≥ 8.0 engages the
    pool path; others auto-fallback).
    """
    return _wiring_text_patch(
        "P26 TurboQuant prefill output prealloc",
        "patch_26_prefill_output",
    )


@register_patch("P44 TQ mixed-batch attn_out pool")
def apply_patch_44_tq_mixed_attn_out() -> PatchResult:
    """Patch 44: Pool the mixed decode+prefill `attn_out` zeros.

    Complements P26 which pools the prefill-only path. Mixed-batch
    branch (`turboquant_attn.py:438`) previously did
    `torch.zeros(N, Hq, D, dtype=q.dtype)` per forward → up to 80 MB
    zero-init on 4096 token batches. Pool reuses memory + zeroes
    `[:num_tokens]` slice.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0. Default-on.
    """
    name = "P44 TQ mixed-batch attn_out pool"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )
    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no TurboQuant kernel")
        return _skipped(name, "non-NVIDIA platform")
    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0")
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring import patch_44_tq_mixed_attn_out
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_44_tq_mixed_attn_out.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P46 GDN gating buffer pool")
def apply_patch_46_gdn_gating_buffers() -> PatchResult:
    """Patch 46: Persistent buffers for `fused_gdn_gating`'s `g` +
    `beta_output` outputs.

    The helper is called once per GDN-bearing layer per forward pass
    and allocates two tiny tensors via `torch.empty(...)`. On
    Qwen3.6-35B-A3B (48 GDN layers) at 250 tok/s decode this is
    ~24 000 allocator ops/sec with zero bytes recovered. Replacing
    with a per-shape-key persistent pool eliminates the churn
    completely (no allocator lock contention, no metadata overhead).

    Byte-exact output vs upstream — Triton kernel writes every
    position unconditionally, so allocated-content doesn't matter
    (equivalent to `torch.empty`).

    Platform guard: NVIDIA CUDA + SM ≥ 8.0. Default-on — no env gate.
    """
    name = "P46 GDN gating buffer pool"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — HIP allocator path differs")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no GDN GPU kernel")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — shares P2x platform gate")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")

    try:
        from vllm._genesis.wiring import patch_46_gdn_gating_buffers
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_46_gdn_gating_buffers.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P7b GDN dual-stream via torch.library.custom_op (opt-in)")
def apply_patch_7b_gdn_dual_stream_customop() -> PatchResult:
    """Patch 7b: graph-safe GDN dual-stream parallelism.

    Alternative to P7 (text-patch with `DualStreamDispatcher` raw CUDA
    streams) that works inside `torch.compile(fullgraph=True)` —
    wraps the two in_proj GEMMs as a single `torch.library.custom_op`
    so dynamo sees an opaque node and doesn't try to trace the stream
    operations.

    Expected gain: +5-8% Qwen3-Next decode tok/s (matches P7 eager
    measurement) while being compatible with vLLM's default
    `aot_compile_fullgraph` path (no `--enforce-eager` required).

    Opt-in via `GENESIS_ENABLE_P7B=1`. Mutually exclusive with P7:
    both text-patch the same 2 lines in `gdn_linear_attn.py`. P7b
    detects P7 conflict via anchor mismatch and skips with a clear
    error.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0.
    """
    name = "P7b GDN dual-stream via torch.library.custom_op (opt-in)"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — HIP stream ordering weaker")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no CUDA streams")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — stream parallelism weak")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: env-opt-in scaffold ready")

    try:
        from vllm._genesis.wiring import patch_7b_gdn_dual_stream_customop
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_7b_gdn_dual_stream_customop.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P40 TurboQuant GQA-grouped decode stage1 (opt-in)")
def apply_patch_40_tq_grouped_decode() -> PatchResult:
    """Patch 40: Port upstream PR #40792 GQA-grouped decode stage1 kernel
    for `turboquant_k8v4`.

    Replaces per-head CTA launch (upstream scalar kernel) with
    per-head-group CTA launch (our port). Each CTA handles up to
    BLOCK_H=16 Q heads sharing one KV head → ~4× fewer KV loads,
    2× arithmetic intensity via `tl.dot` on tensor cores.

    Upstream PR body measured +16-27% decode tok/s on Qwen3-32B
    across A100/H100. Our target 2×A5000 (SM 8.6) Qwen3.6-35B-A3B-FP8
    k8v4 should see similar directional gain.

    Opt-in via `GENESIS_ENABLE_P40=1`. Self-retires when upstream PR
    merges (detected by `_tq_grouped_decode_stage1` symbol appearing
    on the upstream module).

    Scope: FP8 keys + 4-bit values only (`turboquant_k8v4`). MSE-key
    presets retain the scalar kernel via dispatcher fallback.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0.
    """
    name = "P40 TurboQuant GQA-grouped decode stage1 (opt-in)"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no Triton GPU kernel")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — Triton tl.dot requires Ampere+")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: rebind ready (pass apply=True for live wiring)")

    try:
        from vllm._genesis.wiring import patch_40_tq_grouped_decode
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_40_tq_grouped_decode.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P39a FLA chunk_scaled_dot_kkt persistent A pool")
def apply_patch_39a_fla_kkt_buffer() -> PatchResult:
    """Patch 39a: Persistent `A` buffer for FLA `chunk_scaled_dot_kkt_fwd`.

    GDN chunked-prefill allocates `A = torch.empty(B, T, H, BT, fp32)`
    per-layer per-chunk call. On Qwen3.6-35B-A3B with 32 GDN-bearing
    layers, B=1 T≤4096 H=16 BT=64 fp32 = 16 MiB × 32 = 512 MiB of
    per-step allocator churn during long-context prefill — profiler-
    invisible (lazy inside forward), saturates at the yaml=0.93
    boundary where 12 MiB allocs fail.

    Rewires `chunk_scaled_dot_kkt_fwd` to use a single shared persistent
    pool via `FlaKktBufferManager.acquire`. Pool is sized to max
    `(B, max_num_batched_tokens, H, BT)` at first call; reused across
    all GDN layers (sequential-forward invariant).

    Applied via module-level symbol swap + caller-module rebind (FLA
    typically does `from .chunk_scaled_dot_kkt import
    chunk_scaled_dot_kkt_fwd` → callers capture the original reference;
    we walk `sys.modules` and fix those too).

    Platform guard: NVIDIA CUDA + SM ≥ 8.0 (shared with rest of P2x).

    Expected win: frees the 12-34 MiB runtime-headroom ceiling that was
    blocking yaml ≥ 0.93 on dev134. Enables yaml=0.93-0.94 range that
    the user requested, at chunk=4096.
    """
    name = "P39a FLA chunk_scaled_dot_kkt persistent A pool"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant/FLA not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no GDN kernel")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — FLA GDN requires Ampere+")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: rebind ready (pass apply=True for live wiring)")

    try:
        from vllm._genesis.wiring import patch_39_fla_kkt_buffer
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_39_fla_kkt_buffer.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P38 TQ _continuation_prefill persistent workspace")
def apply_patch_38_tq_continuation_memory() -> PatchResult:
    """Patch 38: Replace `_continuation_prefill`'s `.contiguous()` + `torch.cat`
    peak-memory pattern with persistent K_full/V_full shared buffers.

    On dev134+ this path allocates 4× ~128 MiB FP16 transients per call at
    deep prefix continuation (Qwen3.6-35B-A3B-FP8, max_model_len 262144,
    k8v4). Together with allocator fragmentation this saturates a 2×A5000
    setup at cached_len ~= 99k and above — reproducible OOM at
    `turboquant_attn.py:776 v_full = torch.cat(...)`.

    This patch REPLACES the entire `_continuation_prefill` method via
    class-level monkey-patch. The replacement:
      * uses 4-D K/V dequant buffers (prealloc'd by P22's updated helper);
      * writes dequant prefix directly into persistent `_tq_k_full_buf` /
        `_tq_v_full_buf` via in-place `.copy_()` — no `.contiguous()` copy;
      * appends the new chunk into the same workspace instead of
        `torch.cat` → zero transient peaks in the forward path.

    Net budget: +516 MiB persistent (profiler-visible → KV sized correctly)
    to eliminate ~500 MiB of transient-with-fragmentation peaks. This makes
    yaml 0.92-0.94 + chunk 4096 stable for 262k single-request on our 2x
    A5000 setup (previously required yaml=0.80 + chunk=2768 workaround).

    Platform guard: NVIDIA CUDA + SM ≥ 8.0 (shared with P22).
    """
    name = "P38 TQ _continuation_prefill persistent workspace"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no TurboQuant kernel")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — TurboQuant requires Ampere+")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: rebind ready (pass apply=True for live wiring)")

    try:
        from vllm._genesis.wiring import patch_38_tq_continuation_memory
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_38_tq_continuation_memory.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P37 MoE intermediate cache pool (opt-in)")
def apply_patch_37_moe_intermediate_cache() -> PatchResult:
    """Patch 37: Shared `intermediate_cache13` / `cache2` across MoE layers.

    Replaces per-call `torch.empty(...)` in `_fused_marlin_moe` with a
    module-level pool. All MoE layers use identical (N, K, num_topk,
    num_shards) config and execute sequentially per step, so one pool
    is safe.

    On Qwen3.6-35B-A3B chunked-prefill M=4096, saves ~553 MiB per
    MoE-layer × N_moe_layers allocator churn per step.

    Opt-in via `GENESIS_ENABLE_P37=1` (new v7.1 feature; enable after
    a successful integration run). Even with gate OFF the manager API
    is registered and usable, so operators can experiment manually.

    `acquire_cache13` / `acquire_cache2` decorated with
    `@torch._dynamo.allow_in_graph` for `aot_compile_fullgraph`
    compatibility.
    """
    return _wiring_text_patch(
        "P37 MoE intermediate cache pool (opt-in)",
        "patch_37_moe_intermediate_cache",
    )


@register_patch("P36 TurboQuant shared decode buffers")
def apply_patch_36_tq_shared_decode_buffers() -> PatchResult:
    """Patch 36: Share `_tq_mid_o_buf` / `_tq_output_buf` / `_tq_lse_buf`
    across all TurboQuant attention layers.

    Mirrors upstream PR #40655 (@bhoomit). For Qwen3-32B (60 layers)
    saves ~16 GiB direct + ~45 GiB allocator fragmentation. For our
    hybrid Qwen3.6-35B-A3B (10 TQ layers) saves ~9 MiB direct; the real
    value is REDUCING allocator slab count at init, which competes with
    weight-load slabs. We observed 50k prefill OOM with only 21 MiB free
    headroom — any freed MiB matters.

    Platform guard: shared with P22 (NVIDIA CUDA + SM ≥ 8.0). Non-NVIDIA
    falls back to upstream per-layer `register_buffer` path inside the
    text-patch replacement.

    Self-retires when upstream PR #40655 (or its alt PR #40748) merges
    via `upstream_drift_markers`.
    """
    return _wiring_text_patch(
        "P36 TurboQuant shared decode buffers",
        "patch_36_tq_shared_decode_buffers",
    )


@register_patch("P32/P33 TurboQuant cu_2 + synth_seq_lens preallocs")
def apply_patch_32_33_tq_bundled_preallocs() -> PatchResult:
    """Patches 32+33: bundled with P22 — second-hop cu_seqlens scratch (P32)
    and synthetic seq_lens device mirror (P33).

    These are profiler-invisible lazy allocations inside TurboQuant's forward
    path that the master plan identifies as contributing a small but
    real (~0.3% TGS) decode regression when left lazy. We pre-allocate them
    in `_ensure_on_device` alongside the P22 K/V dequant buffers.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0 (shared with P22).

    Wiring: the two get_or_create helpers are called inside
    `ensure_turboquant_buffers()`. This entry-point VERIFIES the helpers
    are importable and platform-compatible and logs the decision.
    """
    name = "P32/P33 TurboQuant cu_2 + synth_seq_lens preallocs"

    try:
        from vllm._genesis.kernels.dequant_buffer import (
            TurboQuantBufferManager,
        )
    except Exception as e:
        return _failed(name, f"kernel import failed: {e}")

    if not TurboQuantBufferManager.should_apply():
        return _skipped(name, "platform guard returned False (shared with P22)")

    # Verify helpers are present (catches migration drift on refactor)
    if not callable(getattr(TurboQuantBufferManager, "get_or_create_cu_2", None)):
        return _failed(name, "get_or_create_cu_2 missing")
    if not callable(
        getattr(TurboQuantBufferManager, "get_or_create_synth_seq_lens", None)
    ):
        return _failed(name, "get_or_create_synth_seq_lens missing")

    return _applied(
        name,
        "cu_2 + synth_seq_lens preallocs registered (invoked from "
        "ensure_turboquant_buffers, fires during profile_run)",
    )


@register_patch("P28 GDN core_attn_out prealloc")
def apply_patch_28_gdn_core_attn() -> PatchResult:
    """Patch 28: Pre-allocate `core_attn_out` in GatedDeltaNet.forward_cuda.

    Previous P19 reverted because the buffer was allocated lazily INSIDE
    forward() (profiler-invisible → CUDA graph recaptures → −30% throughput,
    188× stdev). CRIT-HW-1 from master plan: allocation MUST be via a
    profiler-visible path.

    This correct redo uses `GdnCoreAttnManager.acquire_slice()` which
    reserves the max-size buffer on first call (picked up by profile_run
    warmup) and returns a pointer-stable slice on all subsequent calls.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0. Fallback `torch.zeros` preserves
    correctness on incompatible platforms.

    Wiring strategy: TEXT-PATCH on `gdn_linear_attn.py:571-575`.
    """
    name = "P28 GDN core_attn_out prealloc"
    try:
        from vllm._genesis.kernels.gdn_core_attn_manager import (
            GdnCoreAttnManager,
        )
    except Exception as e:
        return _failed(name, f"manager import failed: {e}")

    # Diagnostic: report whether the platform will actually engage the prealloc.
    engaged = GdnCoreAttnManager.should_apply()

    result = _wiring_text_patch(
        name, "patch_28_gdn_core_attn",
    )
    if result.status == "applied":
        note = "" if engaged else (
            " (applied; runtime will fall back to fresh-zeros on this platform)"
        )
        result = _applied(name, (result.reason or "") + note)
    return result


@register_patch("P7 GDN dual-stream in_proj parallelism")
def apply_patch_7_gdn_dual_stream() -> PatchResult:
    """Patch 7: Parallel execution of `in_proj_qkvz` + `in_proj_ba` GEMMs.

    Recovers ~5% decode throughput on Qwen3-Next / Qwen3.6 hybrid models by
    issuing the two independent GEMMs on separate CUDA streams (aux stream).

    Platform guard:
      - NVIDIA CUDA SM ≥ 8.0: true parallelism (measured +8% on A5000)
      - AMD ROCm:             HIP stream attempt; may serialize
      - Intel XPU / CPU:      sequential fallback (safe)

    Wiring strategy: TEXT-PATCH on `gdn_linear_attn.py` — the two
    back-to-back `in_proj_*` calls in forward_cuda are replaced with a
    `DualStreamDispatcher.maybe_parallel(...)` call that chooses parallel
    or sequential execution based on platform.
    """
    name = "P7 GDN dual-stream in_proj parallelism"
    from vllm._genesis.guards import is_cpu_only, is_intel_xpu
    from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher

    # Always initialize the dispatcher (diagnostics) even in dry-run mode.
    parallel_ok = DualStreamDispatcher.init_once()
    if parallel_ok:
        log.info("[Genesis P7] dispatcher ready (parallel path)")
    else:
        log.info("[Genesis P7] dispatcher ready (sequential fallback)")

    if is_cpu_only():
        # Still register wiring in apply mode so a GPU worker spawned from
        # the same install tree sees the patch. But note the zero-benefit.
        note = " — CPU has no stream parallelism, functional fallback only"
    elif is_intel_xpu():
        note = " — XPU falls back to sequential"
    else:
        note = ""

    result = _wiring_text_patch(
        name, "patch_7_gdn_dual_stream",
    )
    if result.status == "applied" and note:
        result = _applied(name, (result.reason or "") + note)
    return result


@register_patch("P17/P18 Marlin MoE per-SM tuning")
def apply_patch_17_18_marlin_tuning() -> PatchResult:
    """Patches 17+18: Per-SM optimal Marlin MoE `block_size_m` selection.

    Upstream heuristic lands on bsm=16 for FP8. On A5000 (SM 8.6) + Qwen3.6
    M≤4, topk=8, E=256, bsm=8 is measured +1.2%. Additional env knobs allow
    manual tuning of num_warps and num_stages.

    Platform guard: NVIDIA CUDA only (Marlin is a CUDA kernel).

    Wiring strategy: `get_optimal_block_size_m()` is consulted by vLLM's
    fused_marlin_moe dispatcher via monkey-patch. Env overrides:
      VLLM_MARLIN_MOE_BLOCK_SIZE_M  → bsm override (8/16/32/48/64)
      VLLM_MARLIN_MOE_NUM_WARPS     → warp count (2/4/8)
      VLLM_MARLIN_MOE_NUM_STAGES    → pipeline stages (1-8)
    """
    name = "P17/P18 Marlin MoE per-SM tuning"
    from vllm._genesis.guards import is_nvidia_cuda, get_compute_capability
    from vllm._genesis.kernels.marlin_tuning import (
        get_optimal_block_size_m,
        get_num_warps_override,
        get_num_stages_override,
    )

    if not is_nvidia_cuda():
        return _skipped(name, "non-NVIDIA — Marlin is CUDA-only")

    cc = get_compute_capability()
    bsm = get_optimal_block_size_m()
    warps = get_num_warps_override()
    stages = get_num_stages_override()

    if bsm is None:
        return _skipped(
            name,
            f"no tuning entry for SM {cc} — upstream heuristic will be used",
        )

    log.info(
        "[Genesis P17/P18] Marlin tuning ready: SM=%s bsm=%d "
        "num_warps=%s num_stages=%s",
        cc, bsm,
        warps if warps is not None else "default",
        stages if stages is not None else "default",
    )
    return _applied(name, f"SM={cc} bsm={bsm}")


@register_patch("P24 fused_moe num_warps/num_stages overlay")
def apply_patch_24_moe_tune() -> PatchResult:
    """Patch 24: Overlay per-SM / env overrides for num_warps + num_stages
    inside `fused_moe.get_default_config()`.

    Upstream hard-codes `num_warps=4` and `num_stages=3 (or 2 on ROCm)` in
    two branches of `get_default_config` (fp8_w8a8 block-quant path + the
    general bf16/fp16/fp8-per-tensor path). After upstream builds the
    config dict we overlay any non-None value from the Genesis helpers
    `get_num_warps_override()` / `get_num_stages_override()` (which resolve
    env first, then a per-SM auto-select table — Ampere A5000 SM 8.6
    maps to warps=4, stages=3 by default).

    Note on Marlin: this patch is a no-op when the engine takes the
    Marlin CUDA-op path (`moe_wna16_marlin_gemm` doesn't accept Triton
    autotune parameters). It's active only when vLLM falls back to the
    Triton fused_moe kernel, which happens on smaller batches and
    Marlin-incompatible quant types.

    Env overrides:
      VLLM_MARLIN_MOE_NUM_WARPS   ∈ {2, 4, 8}
      VLLM_MARLIN_MOE_NUM_STAGES  ∈ {1..8}
    """
    return _wiring_text_patch(
        "P24 fused_moe num_warps/num_stages overlay",
        "patch_24_moe_tune",
    )


@register_patch("P14 block_table tail zero-fill")
def apply_patch_14_block_table_tail_zero() -> PatchResult:
    """Patch 14: Zero the tail of block_table row after append/move.

    Fixes silent divergence from stale block IDs leaking past
    `num_blocks_per_row` when a block_table row slot is reused by a shorter
    request after a longer one (vLLM PR #39591 / issue #39589).

    Platform guard: universal (pure numpy/torch indexing — no vendor deps).

    Wiring strategy (v7.0 step 5): runtime class-method monkey-patch on
    `vllm.v1.worker.block_table.BlockTable.append_row` and `move_row`.
    Wrapped versions call the original then tail-zero with our helper.
    """
    name = "P14 block_table tail zero-fill"

    try:
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail
        assert callable(zero_block_table_tail)
    except Exception as e:
        return _failed(name, f"kernel import failed: {e}")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: kernel ready (pass apply=True for live wiring)")

    try:
        from vllm._genesis.wiring import patch_14_block_table
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_14_block_table.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P18b TurboQuant decode stage1 tune")
def apply_patch_18b_tq_decode_tune() -> PatchResult:
    """Patch 18b: Env-driven TurboQuant decode stage1 kernel tunables.

    Exposes BLOCK_KV / num_warps / num_stages via env vars so non-H100 cards
    (A5000 especially) can re-tune away from H100-shaped defaults.

    Platform guard: NVIDIA CUDA + SM 8.0+ (TurboQuant is CUDA-only).

    Wiring strategy: `resolve_decode_tune()` is consulted by the kernel
    launcher in `triton_turboquant_decode.py` via monkey-patch or text-
    replacement (Triton compile-time params can't be monkey-patched; text
    patcher for those literals).
    """
    name = "P18b TurboQuant decode stage1 tune"
    from vllm._genesis.kernels import tq_decode_tune as t

    if not t.should_apply():
        return _skipped(
            name,
            "non-NVIDIA or pre-Ampere — TurboQuant not applicable",
        )

    # Log and report whether user opted into overrides
    t.log_selected_tune()

    if t.has_any_override():
        bkv, nw, ns = t.resolve_decode_tune()
        return _applied(name, f"env override BLOCK_KV={bkv} warps={nw} stages={ns}")

    return _applied(
        name,
        f"no env override — using upstream defaults "
        f"({t.UPSTREAM_BLOCK_KV}/{t.UPSTREAM_NUM_WARPS}/{t.UPSTREAM_NUM_STAGES})",
    )


@register_patch("P20 TurboQuant continuation-prefill FP16 rotate")
def apply_patch_20_tq_continuation_prefill() -> PatchResult:
    """Patch 20: Halve peak memory of `_continuation_prefill` (fixes #40420).

    Replaces upstream's FP32 rotation + redundant `.contiguous()` with a
    single FP16 matmul + non-contiguous view that torch.cat materializes.

    Platform guard: NVIDIA CUDA + SM 8.0+ (TurboQuant is CUDA-only).

    Wiring strategy: `continuation_prefill_fp16_rotate()` replaces the
    4-step fp32 block in `TurboQuantAttentionImpl._continuation_prefill`
    via monkey-patch.
    """
    name = "P20 TurboQuant continuation-prefill FP16 rotate"
    from vllm._genesis.kernels import tq_continuation_prefill as t

    if not t.should_apply():
        return _skipped(
            name,
            "non-NVIDIA or pre-Ampere — TurboQuant not applicable",
        )

    # Verify helpers importable
    try:
        assert callable(t.continuation_prefill_fp16_rotate)
        assert callable(t.continuation_prefill_k_view_fp8)
        assert callable(t.continuation_prefill_v_view)
        assert callable(t.get_pi_half)
    except Exception as e:
        return _failed(name, f"helper import failed: {e}")

    log.info(
        "[Genesis P20] TQ _continuation_prefill FP16 helpers ready for "
        "TurboQuantAttentionImpl hook"
    )
    return _applied(name, "fp16-rotation helper ready for _continuation_prefill hook")


@register_patch("P1/P2 FP8 kernel dispatcher")
def apply_patch_1_2_fp8_dispatcher() -> PatchResult:
    """Patches 1+2: FP8 kernel path selection (Triton native vs Marlin fallback).

    Upstream `TritonBlockFP8ScaledMMKernel` assumes SM ≥ 8.9. On Ampere
    (SM 8.0/8.6), it silently produces wrong numerics. This dispatcher routes
    Ampere to Marlin fallback and Ada/Hopper/Blackwell to native Triton.

    Platform guard: NVIDIA CUDA only.

    Wiring strategy: `should_skip_triton_fp8()` is consulted by vLLM's FP8
    kernel dispatcher via monkey-patch on `TritonBlockFP8ScaledMMKernel`.
    """
    name = "P1/P2 FP8 kernel dispatcher"
    from vllm._genesis.guards import is_nvidia_cuda, get_compute_capability
    from vllm._genesis.kernels.fp8_dispatcher import (
        requires_marlin_fp8_fallback,
        fp8_triton_kernel_supported,
        log_dispatcher_decision,
    )

    if not is_nvidia_cuda():
        return _skipped(name, "non-NVIDIA — different FP8 path")

    cc = get_compute_capability()
    log_dispatcher_decision()

    if requires_marlin_fp8_fallback():
        return _applied(name, f"SM={cc} → Marlin fallback path selected")

    if fp8_triton_kernel_supported():
        return _applied(name, f"SM={cc} → native Triton FP8 path selected")

    return _skipped(
        name, f"SM={cc} — no FP8 support at all (unexpected on NVIDIA)",
    )


# ═══════════════════════════════════════════════════════════════════════════
#                             MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run(verbose: bool = True, apply: bool = False) -> PatchStats:
    """Apply all registered patches, return statistics.

    Args:
        verbose: If True, log platform summary before applying patches.
        apply:   If True, perform the actual wiring (text-patches on disk +
                 runtime attribute rebinds). If False (default), run in
                 DRY-RUN mode: import kernels, verify platform compat, but
                 do NOT rewrite any files or rebind any attributes. Dry-run
                 is the right default because it's safe from anywhere.

                 apply=True should be passed from:
                   - The vLLM plugin register() entry point (once per process)
                   - The container entrypoint script (for text-patches that
                     must land before `vllm serve` starts)

    Returns:
        PatchStats with counts and details per patch.
    """
    # Configure logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s:%(name)s] %(message)s",
        )

    # Propagate apply mode to patch functions via module-level flag.
    global _APPLY_MODE
    _APPLY_MODE = apply

    stats = PatchStats()

    # Platform diagnostic — helps debugging on unexpected hardware
    try:
        from vllm._genesis.guards import platform_summary
        summary = platform_summary()
        if verbose:
            log.info("Genesis platform: %s",
                     json.dumps(summary, default=str, indent=None))
    except Exception as e:
        log.warning("Platform summary failed: %s", e)

    # PDL misconfig check (vLLM issue #40742). Warn loudly but don't fail —
    # some environments set these globally and other GPUs in the cluster use
    # them. On the local Ampere rank, we just advise unsetting.
    try:
        from vllm._genesis.guards import detect_pdl_env_misconfig
        bad = detect_pdl_env_misconfig()
        if bad:
            log.warning(
                "[Genesis guard] PDL env vars set but this GPU does NOT "
                "support PDL safely: %s. Reference: vLLM issue #40742 "
                "(Inductor autotune + torch.cuda.synchronize() inside CUDA "
                "graph capture → illegal cuda op → engine crash). Consider "
                "unsetting these on this node.",
                bad,
            )
    except Exception as e:
        log.debug("PDL misconfig check failed: %s", e)

    # Banner
    log.info(
        "Genesis Unified Patch v7.0 — Ampere FP8 + TQ + MoE + Hybrid + bugfixes. "
        "Philosophy: МЫ ЧИНИМ, НЕ ЛОМАЕМ."
    )

    # Apply each patch
    for patch_name, patch_fn in PATCH_REGISTRY:
        try:
            result = patch_fn()
            if not isinstance(result, PatchResult):
                # Back-compat: legacy bool return
                result = (
                    _applied(patch_name) if result
                    else _failed(patch_name, "patch_fn returned False")
                )
            stats.results.append(result)
            if result.status == "failed":
                log.error("[Genesis] FAILED: %s — %s",
                          result.name, result.reason)
            elif result.status == "skipped":
                log.info("[Genesis] skipped: %s — %s",
                         result.name, result.reason)
            else:
                log.info("[Genesis] applied: %s — %s",
                         result.name, result.reason)
        except Exception as e:
            stats.results.append(
                _failed(patch_name, f"{type(e).__name__}: {e}")
            )
            log.exception("[Genesis] EXCEPTION in %s", patch_name)

    log.info("Genesis %s", stats)
    return stats


def verify_live_rebinds() -> dict[str, Any]:
    """Post-register verification: confirm runtime rebinds are actually live
    in the current process (TDD discipline from master plan Part 3).

    Returns a dict:
      {
        "P22": {"expected": True, "actual": True, "ok": True},
        "P31": {"expected": True, "actual": True, "ok": True},
        "P14": {"expected": True, "actual": True, "ok": True},
        ...
      }

    Only patches with Python-attribute rebinds are checked. Text-patches
    (P3, P4, P5, P6, P8, P15) modify source files and are verified by the
    diagnostic probes in validate_integration.sh (grep file for markers).

    Usage (end-of-register hook or test):
      from vllm._genesis.patches.apply_all import verify_live_rebinds
      results = verify_live_rebinds()
      for name, r in results.items():
          if not r["ok"]:
              log.warning("[Genesis] rebind %s not live: expected=%s actual=%s",
                          name, r["expected"], r["actual"])
    """
    results: dict[str, dict] = {}

    def _check(patch_id: str, wiring_module: str):
        """Invoke `is_applied()` on the wiring module; record result."""
        try:
            import importlib
            mod = importlib.import_module(f"vllm._genesis.wiring.{wiring_module}")
        except Exception as e:
            results[patch_id] = {
                "expected": True, "actual": False, "ok": False,
                "error": f"import failed: {e}",
            }
            return
        is_applied_fn = getattr(mod, "is_applied", None)
        if is_applied_fn is None or not callable(is_applied_fn):
            results[patch_id] = {
                "expected": True, "actual": None, "ok": True,
                "note": "module has no is_applied() — skipped",
            }
            return
        try:
            actual = bool(is_applied_fn())
        except Exception as e:
            results[patch_id] = {
                "expected": True, "actual": False, "ok": False,
                "error": f"is_applied() raised: {e}",
            }
            return
        results[patch_id] = {
            "expected": True, "actual": actual, "ok": actual,
        }

    # Runtime rebinds (set attrs on live vLLM classes/modules)
    _check("P22", "patch_22_tq_prealloc")
    _check("P31", "patch_31_router_softmax")
    _check("P14", "patch_14_block_table")
    _check("P28", "patch_28_gdn_core_attn")
    # v7.2 / v7.3 additions — both have symmetric `apply/is_applied/revert`
    # trios per patch_38/patch_39 wiring surface contracts.
    _check("P38", "patch_38_tq_continuation_memory")
    _check("P39a", "patch_39_fla_kkt_buffer")

    return results


def main() -> int:
    """CLI entrypoint. Returns exit code.

    CLI default is apply=True because this entrypoint is the one invoked
    from container scripts (pre-vllm-serve) where text-patches MUST land.
    Pass `--dry-run` for diagnosis-only mode.
    Pass `--verify-rebinds` for post-register verification (additional
    verification + non-zero exit code if any rebind not live).
    """
    import sys as _sys
    argv = _sys.argv[1:]
    dry = "--dry-run" in argv
    verify = "--verify-rebinds" in argv

    try:
        stats = run(verbose=True, apply=not dry)
    except Exception as e:
        log.exception("Genesis orchestrator setup error: %s", e)
        return 2

    exit_code = 1 if stats.failed_count > 0 else 0

    if verify:
        log.info("[Genesis] Post-register rebind verification:")
        results = verify_live_rebinds()
        any_failed = False
        for patch_id, r in results.items():
            mark = "✓" if r.get("ok") else "✗"
            extra = r.get("error") or r.get("note") or ""
            log.info(
                "  %s %s expected=%s actual=%s %s",
                mark, patch_id, r.get("expected"), r.get("actual"), extra,
            )
            if not r.get("ok"):
                any_failed = True
        if any_failed:
            exit_code = max(exit_code, 1)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
