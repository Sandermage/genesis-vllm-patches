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

    @property
    def partial_apply_warnings(self) -> list[PatchResult]:
        """Skipped patches whose reason signals a real problem (drift,
        ambiguous anchor, anchor-missing — NOT opt-in-OFF, upstream-merged,
        or platform-mismatch which are all expected).

        Surfaced separately from `skipped_count` so noonghunna's "silent
        skip class" diagnosis (club-3090 discussion #19) is impossible to
        miss in the boot summary. Cliff 8 hardening, v7.65.
        """
        # Reasons that indicate a benign/expected skip
        BENIGN = (
            "opt-in",   # matches "opt-in only", "opt-in:", "opt-in env"
            "default off",
            "upstream_merged",
            "upstream_already",
            "upstream_already_contains",
            "upstream may have absorbed",
            "upstream pr",  # "redundant: upstream PR ..."
            "platform mismatch",
            "platform_skip",
            "config: opt-in",
            "config: opt-out",
            "config: skipped",
            "config: neutral",
            "already applied",
            "marker present",
            "soft_skip",
            "no-op",
            "dry-run",
            "vllm install root not discoverable",
            "target file not resolvable",
            "is_pn",
            "unsupported",
            "not applicable",
            "auto-disabled",
            "auto-skip",
            "deprecated",
            "obsolete",
            "redundant",
            "deferred",
            "incompatible with",  # P7 deferred reason
        )
        warnings = []
        for r in self.skipped:
            reason_lower = (r.reason or "").lower()
            if not any(b.lower() in reason_lower for b in BENIGN):
                warnings.append(r)
        return warnings

    @property
    def partial_apply_warnings_count(self) -> int:
        return len(self.partial_apply_warnings)

    def summary(self) -> dict[str, Any]:
        return {
            "applied": self.applied_count,
            "skipped": self.skipped_count,
            "failed": self.failed_count,
            "partial_apply_warnings": self.partial_apply_warnings_count,
            "details": {
                "applied": [(r.name, r.reason) for r in self.applied],
                "skipped": [(r.name, r.reason) for r in self.skipped],
                "failed": [(r.name, r.reason) for r in self.failed],
                "partial_apply_warnings": [
                    (r.name, r.reason) for r in self.partial_apply_warnings
                ],
            },
        }

    def __str__(self) -> str:
        base = (
            f"Results: {self.applied_count} applied, "
            f"{self.skipped_count} skipped, {self.failed_count} failed"
        )
        warns = self.partial_apply_warnings_count
        if warns:
            base += f", {warns} ⚠️ partial-apply warning(s)"
        return base


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


_WIRING_STEM_INDEX: dict[str, str] | None = None


def _resolve_wiring_module(stem: str) -> str:
    """Resolve a bare wiring filename stem (e.g. 'patch_67_tq_multi_query_kernel')
    to its full dotted module path. Walks `wiring/` recursively so the
    legacy flat layout AND post-Phase-2.1 category subdirs both work
    transparently.
    """
    global _WIRING_STEM_INDEX
    if _WIRING_STEM_INDEX is None:
        from pathlib import Path
        wiring_dir = Path(__file__).resolve().parent.parent / "wiring"
        idx: dict[str, str] = {}
        if wiring_dir.is_dir():
            for f in wiring_dir.rglob("patch_*.py"):
                rel_parts = f.relative_to(
                    wiring_dir.parent.parent.parent
                ).parts
                idx[f.stem] = ".".join(list(rel_parts[:-1]) + [f.stem])
        _WIRING_STEM_INDEX = idx
    # Fallback to flat layout if not in cache (covers a freshly-added file
    # that wasn't there at first-call time).
    return _WIRING_STEM_INDEX.get(
        stem, f"vllm._genesis.wiring.{stem}"
    )


def _wiring_text_patch(name: str, wiring_module_name: str) -> PatchResult:
    """Generic helper for dry-run / live dispatch of a text-patch wiring module."""
    try:
        import importlib
        mod = importlib.import_module(
            _resolve_wiring_module(wiring_module_name)
        )
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
            from vllm._genesis.wiring.legacy import patch_4_tq_hybrid
            assert callable(patch_4_tq_hybrid.apply)
        except Exception as e:
            return _failed(name, f"wiring import failed: {e}")
        return _applied(name, "dry-run: wiring ready (pass apply=True to execute)")

    # Real apply path: run the text-patcher.
    try:
        from vllm._genesis.wiring.legacy import patch_4_tq_hybrid
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
            from vllm._genesis.wiring.legacy import patch_5_page_size
            assert callable(patch_5_page_size.apply)
        except Exception as e:
            return _failed(name, f"wiring import failed: {e}")
        return _applied(name, "dry-run: wiring ready (pass apply=True to execute)")

    try:
        from vllm._genesis.wiring.legacy import patch_5_page_size
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
        from vllm._genesis.wiring.legacy import patch_5b_page_size_pad_smaller
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
        from vllm._genesis.wiring.legacy import patch_31_router_softmax
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
        from vllm._genesis.wiring.legacy import patch_22_tq_prealloc
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


@register_patch("P61b Qwen3 streaming partial-tag overlap guard")
def apply_patch_61b_streaming_overlap() -> PatchResult:
    """Patch 61b: backport slice of vllm#40783 streaming changes.

    Adds defensive overlap guard against partial `<tool_call>` tag fragments
    leaking as reasoning when the tag is being assembled across multiple
    streaming deltas.

    For Qwen3 with proper special-token handling this is a no-op; useful for
    streaming clients with non-Qwen tokenizers or edge cases where the tag
    arrives character-fragmented.

    Status: opt-in via GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1.

    Credit: @ExtReMLapin (vllm#40783).
    """
    name = "P61b Qwen3 streaming partial-tag overlap guard"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.structured_output import patch_61b_qwen3_streaming_overlap_guard
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_61b_qwen3_streaming_overlap_guard.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P62 structured-output spec-decode timing fix")
def apply_patch_62_struct_out_spec_timing() -> PatchResult:
    """Patch 62: backport of upstream PR vllm#36138 (sfbemerk).

    Fixes grammar bypass when `</think>` (or implicit reasoning-end via
    `<tool_call>`) arrives within a speculative-decode token batch. Likely
    candidate for closing residual 30-50% broken tool-call output that
    P60+P60b+P61 doesn't fully resolve.

    Mechanism: old `should_advance()` checks a derived delta that becomes
    empty when speculative tokens are involved → reasoning_end check fails →
    grammar bypass for all post-reasoning tokens → arbitrary XML emission.

    Status: opt-in via GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1.

    Credit:
      - Upstream fix: @sfbemerk (vllm#36138).
      - Original bug: @cicirori (vllm#34650).
    """
    name = "P62 structured-output spec-decode timing fix"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.structured_output import patch_62_structured_output_spec_decode_timing
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_62_structured_output_spec_decode_timing.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P61 Qwen3 multi-tool first-occurrence")
def apply_patch_61_qwen3_multi_tool() -> PatchResult:
    """Patch 61: Backport of upstream PR vllm#40783 minimal slice — fixes
    multi-tool requests where multiple `<tool_call>` blocks were silently
    dropped (parser found LAST occurrence instead of FIRST).

    Status: opt-in via GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1.

    Credit:
      - Upstream fix: @ExtReMLapin (vllm#40783).
    """
    name = "P61 Qwen3 multi-tool first-occurrence"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.structured_output import patch_61_qwen3_multi_tool_first_occurrence
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_61_qwen3_multi_tool_first_occurrence.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P60b GDN+ngram Triton kernel offset")
def apply_patch_60b_gdn_ngram_triton_kernel() -> PatchResult:
    """Patch 60b (P60 Phase 2): backport vllm#40738 Triton kernel portion.

    DEPENDS ON P60 (Phase 1). Apply P60 first; P60b adds the Triton kernel
    offset arithmetic for conv state read/write. Without P60 Phase 1,
    Phase 2 alone won't help (SSM state must be pre-copied first).

    Modifies `_causal_conv1d_fwd_kernel` Triton kernel signature + body
    to apply `conv_state_token_offset = num_accepted - 1` to STEP 1 read
    and STEP 2 write. Also updates `causal_conv1d_fn` Python wrapper +
    GDN call site to pass `num_accepted_tokens` parameter.

    Status: opt-in via GENESIS_ENABLE_P60B_TRITON_KERNEL=1.

    Risk: Triton signature change invalidates JIT cache. Auto-clears
    causal_conv1d cache entries on apply. First spec-decode call triggers
    ~5-10s kernel recompile (profiler-visible spike).

    Combined with P60 Phase 1, expected to push 43% clean → 95%+ clean.

    Credit:
      - Upstream fix: @tdoublep (vllm core team, vllm#40738).
      - Empirical isolation on Genesis: 2026-04-25 blue/green test cycle.
    """
    name = "P60b GDN+ngram Triton kernel offset"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_60b_gdn_ngram_triton_kernel
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_60b_gdn_ngram_triton_kernel.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P60 GDN+ngram state recovery")
def apply_patch_60_gdn_ngram_state_recovery() -> PatchResult:
    """Patch 60 Phase 1 (Python-only): backport vllm#40738 (Thomas Parnell).

    Top candidate root-cause fix for #40831 / our degenerate-output bug after
    P58 (#40768) + P59 (#39055) + ngram_gpu (Path B) all empirically disproven
    2026-04-25 in blue/green tests.

    Bug: hybrid GDN models with ngram speculative decode read SSM state from
    block[0] instead of block[num_accepted-1] after spec acceptance. Manifests
    as token-level corruption (`<<`, `parameter=parameter`, `<argname>`)
    that only appears when both spec-decode AND structured output (tools)
    are active.

    P60 Phase 1: Python-only changes in 3 files (gdn_attn.py + gdn_linear_attn
    + gpu_model_runner.py). Adds `spec_decode_src_indices` metadata field +
    SSM state pre-copy + non-spec passthrough.

    P60 Phase 2 (Triton kernel patch in causal_conv1d.py) DEFERRED — needed
    for full conv-state correctness if Phase 1 doesn't fully fix.

    Status: opt-in via GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1.

    Credit:
      - Upstream fix: @tdoublep (vllm core team, vllm#40738).
      - Bug surface: @noonghunna (#40807, #40831).
      - Empirical isolation on Genesis: 2026-04-25 blue/green test cycle.
    """
    name = "P60 GDN+ngram state recovery"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_60_gdn_ngram_state_recovery
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_60_gdn_ngram_state_recovery.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P63 MTP/Eagle drafter GDN state recovery")
def apply_patch_63_mtp_gdn_state_recovery() -> PatchResult:
    """Patch 63 (Genesis-original): MTP/Eagle drafter forward GDN state recovery.

    Bug class identified by Genesis investigation 2026-04-25 after @noonghunna's
    Probe 9 showed P60+P60b close the ngram path but MTP n=3 still produces
    empty tool calls. Root cause: Eagle/MTP drafter forward goes through
    `build_for_drafting()` which defaults to `self.build()` WITHOUT
    `num_accepted_tokens`, so P60's spec_decode_src_indices recovery never
    fires for the drafter's GDN attention.

    Fix: override GDN's `build_for_drafting` to read cached num_accepted from
    the builder's own buffer (set by the spec branch of the most recent
    main-step build) and pass it through to `build()`. Engages P60's recovery
    logic for the drafter forward path.

    DEPENDS ON P60 being applied. Without P60's `spec_decode_src_indices`
    field + non-spec branch recovery logic, P63 is a no-op.

    Status: opt-in via GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY=1.

    Validation: requires MTP-enabled test rig (Sander's prod uses ngram, so
    we cannot empirically verify on Genesis hardware). Designed for cross-rig
    validation by @noonghunna's Probe 9 setup or upstream maintainers.

    Credit:
      - Bug class identified: Genesis investigation 2026-04-25
      - Pattern adapted from: @tdoublep (vllm#40738) main-model fix
      - Bug surface: @noonghunna Probe 9 (vllm#40831 thread, 2026-04-25)
    """
    name = "P63 MTP/Eagle drafter GDN state recovery"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_63_mtp_gdn_state_recovery
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_63_mtp_gdn_state_recovery.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P64 qwen3coder MTP streaming early-return fix")
def apply_patch_64_qwen3coder_mtp_streaming() -> PatchResult:
    """Patch 64: Backport of vllm-project/vllm#39598 (kotori-yan, OPEN).

    Streaming-only MTP/spec-decode tool-call edge case:
    - Pre-PR `extract_tool_calls_streaming` early-returns after emitting
      parameter fragments. With MTP, a single delta can bundle the LAST
      parameter value AND `</function>` together. The early return skips
      the `</function>` block, leaving prev_tool_call_arr with stale `"{}"`
      and streamed_args_for_tool without closing `}` → empty `tool_calls[]`
      in final chunk.
    - Plus `_should_check_for_unstreamed_tool_arg_tokens` safety-net was
      gated on non-empty `delta_message.tool_calls` — bypassed when the
      final delta carries no tool_calls but tool calls are still in flight.

    Fix scope: streaming code path only. Non-streaming tool calls unaffected.

    Status: opt-in via GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1.
    Recommended for any setup using LibreChat / OpenWebUI / SSE clients
    against MTP-enabled vLLM.

    Credit:
      - Upstream fix: @kotori-yan (vllm#39598).
      - Bug class identified by Genesis MTP test cycle 2026-04-25.
    """
    name = "P64 qwen3coder MTP streaming early-return fix"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.structured_output import patch_64_qwen3coder_mtp_streaming
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_64_qwen3coder_mtp_streaming.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P65 TurboQuant spec-decode cudagraph downgrade")
def apply_patch_65_turboquant_spec_cg_downgrade() -> PatchResult:
    """Patch 65 (Genesis-original): TurboQuant cudagraph downgrade for spec-decode.

    Root cause for noonghunna #40880 (MTP × TurboQuant × FULL cudagraph
    degenerate output) — identified by Genesis investigation 2026-04-25.

    `_prefill_attention` cudagraph capture bypass (and fast path) both pass
    `cu_seqlens_k = query_start_loc`, treating continuation prefill batches
    (q_len < seq_len) as first-chunk prefill. For MTP n=3 spec-verify batches
    (4-token uniform), the captured kernel attends ONLY to the 4 query tokens
    of current chunk, missing the entire ~290-token cached history. Drafter
    runs without context, predictions collapse to high-bias tokens.

    Workaround: downgrade TurboQuant `_cudagraph_support` from UNIFORM_BATCH
    to UNIFORM_SINGLE_TOKEN_DECODE so spec-verify K+1 batches fall to eager
    (correct per-request continuation branch). 1-token decode batches retain
    cudagraph capture.

    Cost: spec-verify batches lose cudagraph speedup. Net throughput should
    land between cudagraph=ON broken (85 TPS) and cudagraph=NONE correct
    (33 TPS). Correctness restored.

    NOT a proper fix — proper fix needs upstream rework of _prefill_attention
    bypass to handle TurboQuant cached KV under cudagraph capture.

    Status: opt-in via GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE=1.

    Credit:
      - Bug surface: @noonghunna (vllm#40880).
      - Root cause analysis: Genesis investigation 2026-04-25.
      - Web research lead: Wasif Basharat (Medium "Overnight Stack" article).
    """
    name = "P65 TurboQuant spec-decode cudagraph downgrade"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_65_turboquant_spec_cg_downgrade
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_65_turboquant_spec_cg_downgrade.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P66 cudagraph_capture_sizes spec-decode divisibility filter")
def apply_patch_66_cudagraph_size_filter() -> PatchResult:
    """Patch 66 (Genesis-original): cudagraph_capture_sizes divisibility filter.

    Mirrors closed/stale upstream PR vllm-project/vllm#23679 + addresses bug
    class identified in vllm-project/vllm#28015.

    When `uniform_decode_query_len > 1` (e.g., MTP n=3 → q_len=4), capture
    sizes NOT divisible by uniform_decode_query_len produce mixed-q_len
    batches at capture time (e.g., size=10 → [4, 4, 2]). The tail request
    with q_len=2 gets misclassified as PREFILL during capture, baking a
    PREFILL branch into the captured "uniform decode" graph. At runtime,
    real decode batches replay that wrong path → degenerate output OR
    illegal memory access.

    Filter: keep only capture sizes divisible by uniform_decode_query_len
    when spec-decode is active. For non-spec-decode setups: no change
    (filter is a no-op when uniform_q_len == 1).

    Benefits:
      - Boot 2-4x faster (fewer captures during warmup)
      - Less peak GPU memory during capture (avoids OOM)
      - No mixed-q_len batches → no prefill branches baked into uniform
        decode captures
      - Reduces blast radius for the bug class

    Status: opt-in via GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1.

    Credit:
      - Mirror of @fhl2000's PR #23679 (closed, stale, never merged)
      - Bug class identified by @ConcurrentLanguage in #28015
      - Brought to attention by Genesis investigation 2026-04-25
        (noonghunna #40880 cross-engine search)
    """
    name = "P66 cudagraph_capture_sizes spec-decode divisibility filter"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_66_cudagraph_size_divisibility_filter
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_66_cudagraph_size_divisibility_filter.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P68/P69 long-context tool-call adherence")
def apply_patch_68_69_long_ctx_tool_adherence() -> PatchResult:
    """Bundle wiring for P68 + P69 long-context tool-call adherence.

    Genesis-original — addresses model-behavior limitation where Qwen3-class
    models lose <tool_call> format adherence at long context (>4K tokens)
    with significant prefix content. Empirically observed:

      prompt chars  | tool_call success
      ─────────────────────────────────
        0-12K       | 3/3 OK
        16K+        | 0/3 FAIL (JSON-text, refusal, hallucination)

    Plain text generation works at same context, so it's NOT engine bug —
    it's structured-output adherence degradation (model-level "lost in
    the middle" + format decay).

    Two complementary mitigations injected at top of create_chat_completion:
      P68: upgrade tool_choice "auto" -> "required" for long-ctx + tools
      P69: append explicit format reminder to last user message

    Both env-flag opt-in. No-op when disabled. Threshold configurable via
    GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS (default 8000 chars ~= 2K tok).

    Status:
      - GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 to engage P68
      - GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1 to engage P69
      - Wiring applies if EITHER is enabled; both can be enabled together

    Credit: Genesis investigation 2026-04-25, ladder test isolation.
    """
    name = "P68/P69 long-context tool-call adherence"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.structured_output import patch_68_69_long_ctx_tool_adherence
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_68_69_long_ctx_tool_adherence.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P70 Auto-strict-ngram (force prompt_lookup_min>=8)")
def apply_patch_70_auto_strict_ngram() -> PatchResult:
    """Patch 70 (Genesis-original): auto-bump ngram prompt_lookup_min>=8.

    Mirror of the empirical breakthrough from vllm#40875: at min<8 ngram
    matches tool-schema fragments and produces degenerate tool-call output.
    At min>=8 acceptance is matched-only and tool-call rate is 100% clean.

    When env GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1, hooks
    SpeculativeConfig.__post_init__ to auto-bump prompt_lookup_min and
    prompt_lookup_max to >=8 when method=="ngram" or "ngram_gpu".

    Affects engine startup only (per-request override is not architecturally
    possible — speculative_config is engine-level).

    Tradeoff: higher min = stricter matching = lower acceptance rate but
    higher correctness. Recommended ON for tool-call workloads, OFF for
    pure plain-text workloads where speed matters more.

    Status: opt-in via GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1.

    Credit: Genesis investigation 2026-04-25, vllm#40875.
    """
    name = "P70 Auto-strict-ngram (force prompt_lookup_min>=8)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_70_auto_strict_ngram
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_70_auto_strict_ngram.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P67 TurboQuant multi-query kernel for spec-decode K+1")
def apply_patch_67_tq_multi_query_kernel() -> PatchResult:
    """Patch 67 (Genesis-original): proper-fix Triton kernel for multi-query
    TurboQuant attention against compressed cache for spec-decode K+1 batches.

    Replaces P65 workaround (cudagraph downgrade for spec-decode → ~30%
    throughput hit) with a Triton kernel that handles compressed KV cache
    DIRECTLY and supports FULL cudagraph capture.

    Reads TurboQuant k8v4 layout in-kernel:
      - FP8 K (e4b15 on Ampere/Ada, e4nv on Hopper+) via tl.float8 bitcast
      - 4-bit V indices unpacked via bit shift
      - FP16 scale + zero loaded as 2-byte pairs
      - Paged block_table lookup per KV position

    Online softmax per (q_token, head) pair. Phase 1 (prior cached, no causal),
    Phase 2 (current chunk K+1, causal mask `q_pos >= k_pos`).

    Cross-arch: pure tl.dot fp16, no FA3/Hopper-specific intrinsics.
    Tested on Ampere SM 8.6 (A5000), should work on SM ≥ 7.5.

    Empirical correctness (Phase 1 + 2 prototype p67_dev/):
      Reference vs kernel: rel_avg ~1% (FP8 + 4-bit quant noise normal)

    Status: opt-in via GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1.
    On any error, falls through to upstream eager continuation branch.

    Once empirically validated end-to-end on Sander's prod rig:
      - Restore P65 to UNIFORM_BATCH (no longer need cudagraph downgrade)
      - Spec-decode batches regain FULL cudagraph speedup
      - Net: P64 + P65v2 + P66 + P67 = correct + fast

    Credit:
      - Bug surface: @noonghunna (vllm#40880)
      - Algorithm: extends @tdoublep #40792 grouped decode pattern
      - References studied: 0xSero/turboquant kernels, FlashInfer, SageAttention
      - Genesis investigation 2026-04-25/26
    """
    name = "P67 TurboQuant multi-query kernel for spec-decode K+1"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_67_tq_multi_query_kernel
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_67_tq_multi_query_kernel.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P71 Block-verify rejection sampler (vllm#40819 + gemini bug-fixes)")
def apply_patch_71_block_verify() -> PatchResult:
    """Patch 71: opt-in backport of vllm-project/vllm#40819 (Z. Golpayegani,
    OPEN draft) implementing Sun et al. 2024 ICLR block verification rule
    (arXiv 2403.10444) for spec-decode rejection sampling.

    Strictly >= per-token rule in expected accepted tokens. Theorem in
    Sun 2024 §4 proves unbiased (same target marginal preserved).

    Backported with TWO critical bug-fixes from gemini-code-assist review:
      - FIX 1: SHARED u per request (PR uses per-position; Sun 2024 requires
        ONE Bernoulli per block)
      - FIX 2: denom==0 → ACCEPT (1.0); PR returned 0.0 which REJECTS perfect
        drafts

    Activation gate (all must hold):
      - GENESIS_ENABLE_P71_BLOCK_VERIFY=1
      - max_spec_len >= 3
      - draft_probs is not None (per-token probs available; ngram has none)
      - not synthetic_mode
      - not all_greedy (block degenerates to per-token at T=0; upstream
        skips this anyway)

    Realistic gain on 35B-A3B + Ampere SM 8.6: +0-3% wall-clock
    (PR's own Qwen3-32B parity bench). Treat as experimental.

    Safety: any kernel error → silent fall-through to upstream per-token
    path. NO output corruption, NO engine impact.

    Status: opt-in, default OFF. Not enabled in v7.42 prod env.
    """
    name = "P71 Block-verify rejection sampler (vllm#40819 + gemini bug-fixes)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_71_block_verify
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_71_block_verify.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P78 TurboQuant .tolist() capture-guard (adapted from noonghunna)")
def apply_patch_78_tolist_capture_guard() -> PatchResult:
    """Patch 78: surgical safety-net for cudagraph capture in
    TurboQuant._prefill_attention. Falls back to flash_attn_varlen_func
    when torch.cuda.is_current_stream_capturing() returns True (capture
    can't tolerate the .tolist() GPU->CPU sync inside the continuation
    branch).

    Composes additively with our P22/P26/P44 prealloc patches: prealloc
    fires on steady-state (eliminates the .tolist() path entirely);
    P78 fires only during cudagraph capture warmup with dynamic shapes
    that pre-empt prealloc. Belt-and-suspenders approach.

    CREDIT: algorithm + anchor strings adapted from noonghunna's
    patch_tolist_cudagraph.py (Apache-2.0):
      https://github.com/noonghunna/qwen36-27b-single-3090

    Status: opt-in via GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=1.
    """
    name = "P78 TurboQuant .tolist() capture-guard (adapted from noonghunna)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.compile_safety import patch_78_tolist_capture_guard
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_78_tolist_capture_guard.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P77 Adaptive ngram K controller (EMA + hysteresis + auto-disable)")
def apply_patch_77_adaptive_ngram_k() -> PatchResult:
    """Patch 77: wraps `NgramProposer.propose()` with adaptive K controller.

    K dynamically chosen from {0, 1, 3, 5} (configurable) based on EMA of
    acceptance over rolling window, with hysteresis to prevent oscillation
    and auto-disable to K=0 (no-spec mode) when accept_rate < 30%.

    Solves the ngram free-form text pathology: vLLM ngram with fixed K=3
    on workload without repeats wastes 4 forward passes per output token
    (acceptance ~10-15%) → effective decode is 4× slower than no-spec.

    With P77 enabled:
      - Free-form text: K auto-drops to 1 then 0 → ~no-spec TPS (~150 tok/s vs current 46)
      - Tool-call: K stays at 3-5 (high acceptance) → no degradation
      - Mid-session workload shift: probe every 100 batches re-tests

    Status: opt-in via GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K=1.

    Algorithm: port of SGLang adaptive_spec_params.py (Apache-2.0) +
    Nightjar arXiv 2512.22420 auto-disable extension.

    Composition:
      - With P75 (suffix): P75 routes to SuffixDecodingProposer instead, P77
        wiring patch is harmless no-op (NgramProposer never instantiated).
      - With P70 (auto-strict-ngram): orthogonal — P70 sets prompt_lookup_min,
        P77 controls K. Stack cleanly.
      - With MTP method: no-op (only NgramProposer is wrapped).
    """
    name = "P77 Adaptive ngram K controller (EMA + hysteresis + auto-disable)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_77_adaptive_ngram_k
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_77_adaptive_ngram_k.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P79b Async × spec-decode proposer-sync backport (vllm#40610)")
def apply_patch_79b_async_proposer_sync() -> PatchResult:
    """Patch 79b: backport of vllm#40610 (OPEN draft, tracked from #40608).

    Wraps GPUModelRunner.sample_tokens() to re-record
    `prepare_inputs_event` AFTER the spec-decode proposer GPU work
    completes (not just after input prep). Fixes async-scheduling ×
    spec-decode race: previously, the next batch's `_update_states`
    could mutate persistent block_table / batch metadata while the
    previous batch's proposer was still reading those tensors on GPU.

    Symptoms (per upstream issue #40608):
    - Nondeterministic instability on async + EAGLE/MTP/ngram_gpu
    - Stale state usage during proposer execution
    - Hard to reproduce — concurrency-sensitive race

    Direct value for Genesis prod (sync ngram): NONE — async path
    not engaged. But protects users on async + spec-decode.

    Status: opt-in via GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC=1.
    """
    name = "P79b Async × spec-decode proposer-sync backport (vllm#40610)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_79b_async_proposer_sync
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_79b_async_proposer_sync.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P79c Stale spec_token_ids cleanup for unscheduled requests (vllm#37629)")
def apply_patch_79c_stale_spec_token_cleanup() -> PatchResult:
    """Patch 79c: backport of vllm#37629 (OPEN, fixes #36906).

    Adds a cleanup pass after the main scheduling loop in
    `Scheduler.schedule()` that clears `spec_token_ids` for any
    running request not present in `num_scheduled_tokens`. Prevents
    stale `-1` placeholder leak into F.embedding() under
    budget-exhausted high-concurrency on async + EAGLE/MTP.

    Trigger: high concurrency exhausting token budget before scheduler
    visits all running requests. Most visible on multimodal models
    (large prefill chunks consume disproportionate budget) but PR's
    regression test proves it's NOT multimodal-specific.

    Direct value for Genesis prod (max_num_seqs=2, sync ngram): NONE.
    Single-user can't exhaust token budget. Useful only for high-concurrency
    multimodal users on async + EAGLE/MTP.

    Status: opt-in via GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP=1.
    """
    name = "P79c Stale spec_token_ids cleanup for unscheduled requests (vllm#37629)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_79c_stale_spec_token_cleanup
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_79c_stale_spec_token_cleanup.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P81 fp8 block-scaled MM low-M decode tuning (vllm#40925)")
def apply_patch_81_fp8_block_scaled_m_le_8() -> PatchResult:
    """Patch 81: backport of vllm#40925 (tonyliu312, OPEN).

    Specializes `w8a8_triton_block_scaled_mm` default config for M<=8
    (single-request decode + MTP K=3 verify):
      - BLOCK_SIZE_M: 64 -> 16  (4x less wasted M-dim)
      - num_stages: 2 -> 3 (non-ROCm only)
    Larger M unchanged. Pre-tuned JSON configs short-circuit before this.

    Direct hit for Genesis prod: Qwen3.6-A3B FP8 + max_num_seqs=2 (M=1
    typical, M=4 for MTP K=3 verify) + no pre-tuned JSON for our
    (N, K, RTX A5000) tuple in configs/.

    Empirical (per upstream PR on GB10 sm_121):
    +23% median decode TPS (5.45 -> 6.73 t/s).

    Status: opt-in via GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1.
    """
    name = "P81 fp8 block-scaled MM low-M decode tuning (vllm#40925)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.kernels import patch_81_fp8_block_scaled_m_le_8
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_81_fp8_block_scaled_m_le_8.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P82 SGLang threshold_single OR-clause acceptance (BIASED — opt-in research)")
def apply_patch_82_sglang_acceptance_threshold() -> PatchResult:
    """Patch 82: backport of SGLang's per-token acceptance OR-clause for
    speculative decoding rejection sampling.

    Adds OR-clause to the per-token rule in `rejection_random_sample_kernel`:
      vanilla:  accepted = draft_prob > 0 AND target_prob/draft_prob >= uniform_prob
      P82:      accepted = vanilla OR target_prob >= GENESIS_P82_THRESHOLD_SINGLE

    Targets the structural ceiling identified in v7.13 strict-ngram analysis:
    `clean_rate ≈ accept_rate^num_spec`. The OR-clause short-circuits when
    target is even moderately confident, decaying the exponent slowly.

    BIASED RULE — loses unbiased-sampling guarantee. Acceptable for
    greedy / low-temperature tool-call workloads (bias is in the right
    direction); risky for high-temperature creative-writing.

    Threshold baked from env GENESIS_P82_THRESHOLD_SINGLE (default 0.3)
    at server start. Changing threshold requires restart.

    Status: opt-in via GENESIS_ENABLE_P82=1. Default OFF. NOT VALIDATED
    on prod yet — must run genesis_quality_harness.py + genesis_bench_v3.py
    blue/green sweep before any deployment decision.
    """
    name = "P82 SGLang threshold_single OR-clause acceptance (BIASED — opt-in research)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_82_sglang_acceptance_threshold
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_82_sglang_acceptance_threshold.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P83 MTP keep-last-cached-block (vllm#38182 mitigation)")
def apply_patch_83_mtp_keep_last_cached_block() -> PatchResult:
    """Patch 83: skip the eagle-style pop() of the last matched cached block
    when GENESIS_ENABLE_P83=1 is set in env.

    Root cause (vllm#38182 by uOnePiece + @Angazenn):
    `vllm/v1/core/single_type_kv_cache_manager.py:447-468` force-pops the
    last matched cached block when `use_eagle=True`. This is intentional
    for true Eagle/Eagle3 drafters (which need pre-materialised hidden
    states from prefill), but MTP gets caught up because
    `config/speculative.py:890-891` returns True for method='mtp' from
    `use_eagle()`. For hybrid Qwen3.6-MoE with P5 LCM-pad, the popped
    block is sized to the Mamba layer requirement (often >1024 tokens),
    so each cache "hit" costs ~1024 recomputed tokens.

    Empirical (this rig, Qwen3.6-35B-A3B-FP8, 2× A5000):
      - cache ON  + default:           ~164 tok/s mean (cache useless)
      - cache OFF (v7.48):              ~213 tok/s mean (+30%)
      - cache ON  + --block-size 16:   ~163 tok/s (P5 LCM overrides)
      - cache ON  + P83 (this patch):  TBD — predicted ~213 tok/s + cache benefit

    Status: opt-in via GENESIS_ENABLE_P83=1. Default OFF.
    MTP-only safe; do NOT enable for true Eagle/Eagle3 — they need the drop.
    """
    name = "P83 MTP keep-last-cached-block (vllm#38182 mitigation)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_83_mtp_keep_last_cached_block
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_83_mtp_keep_last_cached_block.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P84 hash_block_size override (vllm#38182 ACTUAL root cause)")
def apply_patch_84_hash_block_size_override() -> PatchResult:
    """Patch 84: text-patch scheduler.py:234 to read hash_block_size from env
    GENESIS_P84_HASH_BLOCK_SIZE (default: unchanged self.block_size).

    Discovery: Genesis P83 DEBUG instrumentation (2026-04-27) empirically
    demonstrated that find_longest_cache_hit is NEVER called for our hybrid
    Qwen3.6-MoE workload because request_block_hasher returns ZERO hashes
    when block_size > num_tokens. Scheduler.py:234 forces hash_block_size =
    self.block_size, which on hybrid models is LCM-padded up to Mamba state
    size (often >= 2048). For 1424-token requests, num_hashes=0 → cache
    machinery runs with full overhead but produces zero hits.

    The vllm#38182 issue identified the WRONG root cause (the L457 pop);
    Genesis P84 attacks the actual upstream cause (the hash_block_size
    coupling). P83 is kept as opt-in research artifact for the downstream
    symptom; P84 is the real fix.

    Constraint: chosen hash_block_size must divide EVERY KV cache group's
    block_size, otherwise vLLM's own assertion fires at startup
    (kv_cache_coordinator.py:403-405).

    Recommended value: GENESIS_P84_HASH_BLOCK_SIZE=16 (full-attention default).

    Status: opt-in via GENESIS_P84_HASH_BLOCK_SIZE=<int>. Default OFF.
    """
    name = "P84 hash_block_size override (vllm#38182 ACTUAL root cause)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.kv_cache import patch_84_hash_block_size_override
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_84_hash_block_size_override.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch(
    "P100 FlashInfer FULL CUDA graph for spec-decode (vllm#41127 backport v7.62.17)"
)
def apply_patch_100_flashinfer_full_cg_specdec() -> PatchResult:
    """Patch 100: backport of vllm#41127 (FlashInfer FULL CG for spec-decode).

    Per Sander 2026-04-28: 'не ждём, изучаем, импортируем'. 11 sub-patches
    on flashinfer.py. 27B variants (FlashInfer + spec-decode + non-DCP)
    get UNIFORM_BATCH cudagraph instead of PIECEWISE.

    Expected: +5-10% TPS on Ampere SM 8.6.
    NO-OP for PROD (turboquant_attn backend).

    Status: opt-in via GENESIS_ENABLE_P100=1.
    """
    name = "P100 FlashInfer FULL CUDA graph for spec-decode (vllm#41127)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.perf_hotfix import patch_100_flashinfer_full_cg_specdec
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_100_flashinfer_full_cg_specdec.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch(
    "P103 FLA Cliff 2 chunked fwd_h+fwd_o orchestrator (genesis-original v7.62.20)"
)
def apply_patch_103_fla_cliff2_chunked() -> PatchResult:
    """Patch 103: chunked fwd_h+fwd_o for FLA GDN at long context.

    Wraps `vllm.model_executor.layers.fla.ops.chunk.chunk_gated_delta_rule_fwd`
    with a per-sub-T orchestrator that runs fwd_h + fwd_o chained, never
    materializes the full (B, NT, H, V, K) hidden-state tensor.

    Targets the Cliff 2 OOM at ~50-60K single-prompt prefill on 24 GB GPUs
    (qwen36-27b-single-3090#1). Saves ~600 MiB headroom per rank at T=64K.
    No-op for cu_seqlens != None or T <= MAX_T (default 16384).

    Status: opt-in via GENESIS_ENABLE_P103=1.
    """
    name = "P103 FLA Cliff 2 chunked fwd_h+fwd_o orchestrator"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: monkey-patch ready")
    try:
        from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_103_fla_cliff2_chunked.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch(
    "P101 TQ continuation 64-token slicing (vllm#41123 selective v7.62.16)"
)
def apply_patch_101_tq_continuation_slicing() -> PatchResult:
    """Patch 101: SELECTIVE backport of vllm#41123 TQ on hybrid models.

    TAKE: _CONTINUATION_DECODE_THRESHOLD 128→64, _CONTINUATION_DECODE_MAX_CACHED_LEN=32K,
    64-token slicing loop in _prefill_attention.
    SKIP: cudagraph_support downgrade (would hurt PROD), hybrid boundary-skip.

    Expected: +3-12% TPS on PROD long-context.
    Composes with P98/P99 (non-overlapping anchors).
    Status: opt-in via GENESIS_ENABLE_P101=1.
    """
    name = "P101 TQ continuation 64-token slicing (vllm#41123 selective)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.perf_hotfix import patch_101_tq_continuation_slicing
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_101_tq_continuation_slicing.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch(
    "P99 WorkspaceManager memoize get_simultaneous (perf hotfix v7.62.15)"
)
def apply_patch_99_workspace_manager_memoize() -> PatchResult:
    """Patch 99: memoize WorkspaceManager.get_simultaneous().

    Per Sander 2026-04-28 direct request 'if revert gives speedup, look at
    kernel — maybe rewrite'. P99 keeps upstream design but adds memo cache
    by (shapes_and_dtypes, ubatch_id, ws_data_ptr).

    Status: opt-in via GENESIS_ENABLE_P99=1.
    """
    name = "P99 WorkspaceManager memoize get_simultaneous (perf hotfix)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.perf_hotfix import patch_99_workspace_manager_memoize
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_99_workspace_manager_memoize.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch(
    "P98 TQ WorkspaceManager revert (vllm#40941 perf hotfix v7.62.14)"
)
def apply_patch_98_tq_workspace_revert() -> PatchResult:
    """Patch 98: revert WorkspaceManager indirection in turboquant_attn.py.

    Diagnosis 2026-04-28: NEW vllm caused 17% TPS regression on PROD
    (200 → 167 TPS) due to current_workspace_manager().get_simultaneous()
    Python lookup × N layers × per-step in _decode_attention.

    Restores OLD per-layer cached buffer pattern (pre-vllm#40941). Memory
    cost: O(num_layers) extra dequant buffers (~1GB for 64-layer).

    Status: opt-in via GENESIS_ENABLE_P98=1.
    """
    name = "P98 TQ WorkspaceManager revert (vllm#40941 perf hotfix)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.perf_hotfix import patch_98_tq_workspace_revert
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_98_tq_workspace_revert.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch(
    "P94 Spec-decode prepare_next_token_ids_padded zero-alloc (vllm#41043 backport)"
)
def apply_patch_94_spec_decode_zero_alloc() -> PatchResult:
    """Patch 94: backport of vllm#41043 (wangluochao902, OPEN).

    Replaces GPU->CPU .tolist() + list-comprehension + np.array allocation
    chain in `LLMBaseProposer.prepare_next_token_ids_padded` with an
    in-place loop. Algorithmic identity preserved.

    PR author measured P99 TPOT -9.3% on Llama-3.1-8B + Eagle3 TP=4.
    For our MTP K=3 single-stream: expected +2-4% wall TPS + tighter CV.

    Applies to ALL spec methods (Eagle, MTP, ngram, draft model).
    Status: opt-in via GENESIS_ENABLE_P94=1, default OFF.
    """
    name = (
        "P94 Spec-decode prepare_next_token_ids_padded zero-alloc "
        "(vllm#41043 backport)"
    )
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_94_spec_decode_zero_alloc
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_94_spec_decode_zero_alloc.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch(
    "P95 Marlin TP cudagraph cap on Ampere (vllm#40385 backport)"
)
def apply_patch_95_marlin_tp_cudagraph_cap() -> PatchResult:
    """Patch 95: backport of vllm#40385 (OPEN as of 2026-04-28).

    Defensive cap of `max_cudagraph_capture_sizes` to avoid OOM on
    TP>=2 with Marlin kernels on Ampere SM 8.6 (our 2x A5000 PROD).

    [Genesis production-readiness audit fix 2026-04-30]: this hook
    was missing from apply_all.py despite the wiring file existing
    and the PATCH_REGISTRY entry being live since 2026-04-29 — so
    GENESIS_ENABLE_P95=1 silently did nothing. Now wired correctly.

    Status: opt-in via GENESIS_ENABLE_P95=1, default OFF.
    """
    return _wiring_text_patch(
        "P95 Marlin TP cudagraph cap on Ampere (vllm#40385 backport)",
        "patch_95_marlin_tp_cudagraph_cap",
    )


@register_patch(
    "P91 AutoRound row-parallel group cdiv + start-idx fix (vllm#39460 backport)"
)
def apply_patch_91_autoround_row_group_cdiv() -> PatchResult:
    """Patch 91: backport of vllm#39460 (non-MoE portion only).

    Fixes silent dequant corruption when AutoRound INT4/INT8 checkpoints
    have row-parallel layers whose input_size_per_partition is not
    divisible by group_size at TP>=2.

    Two anchored sites in two files:
      - gptq_marlin.py: replace floor-div with cdiv() in two scale-size
        computations + tag scales/qzeros with row_group_size and
        row_input_size_per_partition attrs
      - parameter.py: RowvLLMParameter.load_row_parallel_weight uses
        the group-aware start_idx when the new attrs are present, falls
        back to the original behavior otherwise (no regression for
        layers without quant grouping)

    Hypothesized to address the dominant cause of Lorbus INT4 perf gap
    vs Minachist INT8 on our 2x A5000 deployment.

    Status: opt-in via GENESIS_ENABLE_P91=1. Default OFF.
    """
    name = (
        "P91 AutoRound row-parallel group cdiv + start-idx fix "
        "(vllm#39460 backport)"
    )
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.kernels import patch_91_autoround_row_group_cdiv
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_91_autoround_row_group_cdiv.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P87 Marlin sub-tile output dim pad-on-load (vllm#40361 backport)")
def apply_patch_87_marlin_pad_sub_tile() -> PatchResult:
    """Patch 87: backport of vllm#40361 — MarlinLinearKernel sub-tile
    output dim pad-on-load.

    The Marlin GPTQ/AutoRound kernel requires per-rank out_features to
    be a multiple of GPTQ_MARLIN_MIN_THREAD_N=64. When TP shards a
    weight whose natural out-dim is not tile-aligned (e.g. Qwen3.5
    GatedDeltaNet.in_proj_ba with num_v_heads=64 at TP>=2, or Intel
    Qwen3.6-35B-A3B-int4-AutoRound n=32 shard at TP=2),
    `can_implement` returns False and load fails / falls back to a
    much slower kernel.

    P87 wraps three MarlinLinearKernel methods via class-rebind:
      - can_implement: validates against round_up(n, 64)
      - process_weights_after_loading: zero-pads qweight/scales/qzeros/
        bias along output dim BEFORE the original PWA runs, so all
        downstream repack/permute/zero-point transforms see the padded
        shape consistently
      - apply_weights: pads bias if caller-supplied at orig_n, calls
        the original wrapped method (which now sees padded out-dim
        through c.partition_weight_shape[1]), and slices the extra
        padded columns off the output

    The padded weight columns decode to zero, so marlin_gemm produces
    zero contribution for them — the slice discards both before they
    reach the caller. Runtime cost is zero (padding happens once at
    load). VRAM cost is a few KB per affected layer.

    PR bench: +24% on 2x RTX 3090 SM 8.6 with Intel Qwen3.6-35B-A3B-
    int4-AutoRound TP=2 (137 -> 170 t/s). On our 2x A5000 SM 8.6 the
    same hardware family applies; expected impact depends on whether
    our exact checkpoint shards into sub-tile out-dims.

    Idempotent + drift-aware: skips if `_maybe_pad_n` already exists on
    MarlinLinearKernel (upstream merge detected), or if our wrapper
    sentinel is set (already applied).

    Status: opt-in via GENESIS_ENABLE_P87=1. Default OFF.
    """
    name = "P87 Marlin sub-tile output dim pad-on-load (vllm#40361 backport)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: class-rebind ready")
    try:
        from vllm._genesis.wiring.kernels import patch_87_marlin_pad_sub_tile
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_87_marlin_pad_sub_tile.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("PN8 MTP/draft online-quant propagation (vllm#40849 backport)")
def apply_patch_N8_mtp_draft_online_quant_propagation() -> PatchResult:
    """Patch N8: backport of vllm#40849 (bhoomit) — propagate online
    quantization (e.g. fp8_per_tensor) from target model to spec-decode
    draft model in `get_draft_quant_config()`.

    Currently the draft always loads in BF16 even when the target is
    online-quantized, wasting memory that could feed KV cache. PR #40849
    modifies `vllm/model_executor/models/utils.py::get_draft_quant_config`
    so that, when the draft has no explicit quantization, it inherits
    the target's `OnlineQuantizationConfig` directly. Also adds a
    fallback in the existing draft-quant lookup path that catches
    `ValueError`/`FileNotFoundError` (online-quant methods crash through
    the checkpoint config path because hf_overrides is a callable).

    Empirical (PR author): FP8 target + Eagle3 draft on Qwen3-32B —
    draft model memory 1.45 GiB BF16 → 0.88 GiB FP8 = -40% on draft,
    -0.57 GiB on total worker. Predicates: spec method == 'mtp',
    'qwen3_next_mtp', 'eagle', 'eagle3', 'medusa' AND main model has
    `OnlineQuantizationConfig`.

    Status: opt-in via GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1.
    Default OFF. NO-OP for current Genesis prod (Lorbus/Minachist 27B
    do not run online-quant + external draft); valuable when DFlash /
    Eagle3 / FP8 stacks roll out.
    """
    name = "PN8 MTP/draft online-quant propagation (vllm#40849 backport)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_N8_mtp_draft_online_quant_propagation
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_N8_mtp_draft_online_quant_propagation.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("PN9 independent drafter attention backend (vllm#39930 backport)")
def apply_patch_N9_independent_drafter_attn_backend() -> PatchResult:
    """Patch N9: backport of vllm#39930 (MatthewBonanni, MERGED upstream) —
    allow the spec-decode drafter to use a different attention backend
    than the target model.

    Currently the drafter inherits target's attention backend, which
    breaks for drafters with incompatible requirements (e.g. DFlash
    needs non-causal attention support, which TRITON_ATTN does not
    provide → ValueError on boot). PR #39930 modifies
    `vllm/v1/spec_decode/llm_base_proposer.py::_create_draft_vllm_config`
    to ALWAYS reset the drafter's attention backend (None = auto-select
    independently from target). Unblocks DFlash spike sprint without
    requiring full pin bump (which would drag in #40860 mega-merge risk).

    Genesis backport is minimal — text-patches only the
    `_create_draft_vllm_config` body. Operator chooses the drafter
    backend via env GENESIS_PN9_DRAFTER_BACKEND (e.g. "FLASH_ATTN",
    "FLASHINFER", "TRITON_ATTN"); unset/auto/none → drafter
    auto-selects. We do NOT add the new pydantic field on
    SpeculativeConfig (too invasive at runtime for a frozen dataclass +
    field_validator).

    Predicates: spec_decode active. Patch is a no-op when not.

    Status: opt-in via GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN=1.
    Default OFF.
    """
    name = "PN9 independent drafter attention backend (vllm#39930 backport)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_N9_independent_drafter_attn_backend
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_N9_independent_drafter_attn_backend.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("PN11 GDN a/b contiguity in fix_query_key_value_ordering (vllm#41142 backport)")
def apply_patch_N11_gdn_a_b_contiguous() -> PatchResult:
    """Patch N11: backport of vllm#41142 (Yeuvoir, OPEN as of 2026-04-29) —
    force `.contiguous()` on `b` and `a` tensors after reshape inside
    `GatedDeltaNetAttention.fix_query_key_value_ordering`.

    Fixes upstream issue #41112: the reshape returns a non-contiguous view
    when `num_v_heads == num_k_heads` (np/ng == 1), causing
    `fused_post_conv_prep` Triton kernel to mis-index a/b tensors with
    head-dim stride != 1. Symptom: silent quality drift (no crash).

    For Genesis prod stack (Qwen3.6-27B has np/ng=8, not affected;
    Qwen3.6-35B-A3B has no GDN), this is DEFENSIVE — installs the
    contiguity guard against future model swaps that hit np/ng=1.

    Cost: zero. `.contiguous()` is no-op when tensor is already contiguous.

    Status: opt-in via GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1.
    Default OFF.
    """
    name = "PN11 GDN a/b contiguity (vllm#41142 backport)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.hybrid import patch_N11_gdn_a_b_contiguous
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_N11_gdn_a_b_contiguous.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("PN12 FFN intermediate scratch pool (Cliff 1 fix on TQ3)")
def apply_patch_N12_ffn_intermediate_pool() -> PatchResult:
    """Patch N12: pool transient SiluAndMul output buffers across layers.

    Closes Cliff 1 OOM (138 MiB allocate failed at 122 MiB free) on TQ3
    path that PN8 cannot address (different memory class — transient
    activation peak vs persistent draft footprint).

    Root cause: vllm/model_executor/layers/activation.py:146 SiluAndMul.
    forward_cuda allocates [M, intermediate_size] BF16 transient PER
    LAYER × 64 layers = 4.7-18 GiB allocator churn per forward step on
    Lorbus 27B-int4. Pool single shared buffer per (intermediate_size,
    dtype, device) — pointer-stable, cudagraph-safe.

    Status: opt-in via GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1.
    Default OFF.
    """
    name = "PN12 FFN intermediate scratch pool (Cliff 1 fix)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.hybrid import patch_N12_ffn_intermediate_pool
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_N12_ffn_intermediate_pool.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch(
    "PN25 SiluAndMul.forward_native opaque-op pool "
    "(Cliff 1 mech B compile-path companion to PN12)"
)
def apply_patch_N25_silu_inductor_safe_pool() -> PatchResult:
    """Patch N25: sister-patch to PN12 covering the compile dispatch path.

    PN12 patches `SiluAndMul.forward_cuda` (eager mode); PN25 patches
    `SiluAndMul.forward_native` via a `torch.library.custom_op` so
    torch.compile/Inductor cannot inline the FFN intermediate alloc and
    bypass PN12's pool.

    Reported by noonghunna in club-3090#16 (VolandBerlioz Reddit + ampersandru
    confirmation): on `custom_ops=["none"]` configs (default V1
    aot_compile_fullgraph) `__call__` dispatches to `forward_native`,
    Inductor traces and lowers to `empty_strided_cuda(...)` at line
    `inductor_cache/...py:1208` — completely outside PN12's hot path.

    PN25 registers `genesis::silu_and_mul_pooled` (opaque to Inductor)
    and rewrites `forward_native` to dispatch through it. Inside the
    opaque body, the same `FFNIntermediateCache` pool used by PN12
    serves the [M, intermediate_size] transient. Pool is shared — both
    paths converge on one buffer.

    Status: opt-in via GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE=1.
    Default OFF. Composes with PN12 (recommended pairing for any
    inductor-heavy config). Standalone use covers compile-only paths;
    PN12-only covers eager-only paths.
    """
    name = (
        "PN25 SiluAndMul.forward_native opaque-op pool "
        "(Cliff 1 mech B compile-path)"
    )
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.hybrid import patch_N25_silu_inductor_safe_pool
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_N25_silu_inductor_safe_pool.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("PN13 CUDAGraphWrapper lambda arity (vllm#41235 backport)")
def apply_patch_N13_cuda_graph_lambda_arity() -> PatchResult:
    """Patch N13: backport of vllm#41235 (roikoren755, OPEN as of 2026-04-29) —
    fix CUDAGraphWrapper gc.collect/empty_cache lambda arity.

    Genesis-relevant because our P67/P67b/P78/P85 family uses nested
    @torch.compile callables. When dynamo recompiles inside cudagraph
    capture, gc.collect(generation) fires with a positional arg → 0-arg
    lambda → TypeError → worker dies. Author reports "consistent on GB200
    nightly"; matches Sander's planned R6000 Pro Blackwell upgrade.

    Cost: 2-line text-patch, zero runtime overhead, defensive only.
    Recommend ON for any future Blackwell deployment; intermittent on
    Ampere consumer.

    Status: opt-in via GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY=1.
    Default OFF.
    """
    name = "PN13 CUDAGraphWrapper lambda arity (vllm#41235 backport)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.hybrid import patch_N13_cuda_graph_lambda_arity
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_N13_cuda_graph_lambda_arity.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("PN14 TQ decode IOOB safe_page_idx clamp (vllm#40074 backport)")
def apply_patch_N14_tq_decode_oob_clamp() -> PatchResult:
    """Patch N14: backport of vllm#40074 (devarakondasrikanth, OPEN as of
    2026-04-29) — fix TurboQuant decode kernel index-out-of-bounds.

    `_tq_decode_stage1` in `triton_turboquant_decode.py` uses `page_idx`
    directly in pointer arithmetic. The mask= argument guards the loaded
    VALUE on masked-out lanes but NOT the address computation; on long
    (>32k) sequences the bounds checker fires (originally seen on 4090).

    Fix: `safe_page_idx = tl.where(kv_mask, page_idx, 0)` BEFORE
    Block_table_ptr arithmetic. Zero-cost (one tl.where in registers).

    Genesis hardware-relevance: Ampere sm_86 (A5000) does not see the
    assertion in PROD; Blackwell upgrade path (R6000 Pro Q3 2026) likely
    benefits. Defensive backport — fires whenever P67 dispatch returns
    False or spec-decode is OFF/K=1.

    Status: opt-in via GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1.
    Default OFF. Self-retires via marker `safe_page_idx` when #40074 merges.
    """
    name = "PN14 TQ decode IOOB safe_page_idx clamp (vllm#40074 backport)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.kernels import patch_N14_tq_decode_oob_clamp
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_N14_tq_decode_oob_clamp.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("PN19 Scoped max_split_size_mb during model load (vllm#41268)")
def apply_patch_N19_scoped_max_split() -> PatchResult:
    """Patch N19: backport of vllm#41268 (MatthewBonanni, OPEN
    2026-04-30) — temporarily set max_split_size_mb=20 (PyTorch
    minimum) for the duration of model load to mitigate PyTorch 2.10+
    allocator fragmentation. Restores prior allocator settings on
    exit (or PyTorch's effective default of SIZE_MAX = no limit).

    Cudagraph-safe (load-time only; capture phase uses the restored
    allocator). Self-detects torch lacking
    `_accelerator_setAllocatorSettings` and falls through unchanged.

    Status: opt-in via GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT=1.
    Estimated win: 200-500 MiB on H100 (per PR author); unverified
    on Ampere — measure before relying on it.
    Default OFF.
    """
    return _wiring_text_patch(
        "PN19 Scoped max_split_size_mb during model load (vllm#41268)",
        "patch_N19_scoped_max_split",
    )


@register_patch("PN23 DFlash combine_hidden_states dtype cast (vllm#40334 backport)")
def apply_patch_N23_dflash_combine_hidden_dtype() -> PatchResult:
    """Patch N23: backport of vllm#40334 (ciphernaut, OPEN).

    Six-line defensive cast in Qwen3DFlashModel.combine_hidden_states to
    handle mixed-precision targets (AWQ + non-quantized layers,
    FP8 + BF16 mix). Casts hidden_states to fc.params_dtype before the
    FC layer call. Fixes RuntimeError on mixed-precision DFlash configs.

    Status: opt-in via GENESIS_ENABLE_PN23_DFLASH_DTYPE_FIX=1.
    Default OFF. Auto-no-op once vllm#40334 merges (drift marker).
    """
    return _wiring_text_patch(
        "PN23 DFlash combine_hidden_states dtype cast (vllm#40334 backport)",
        "patch_N23_dflash_combine_hidden_dtype",
    )


@register_patch("PN21 DFlash SWA support partial backport (vllm#40898 backport)")
def apply_patch_N21_dflash_swa_support() -> PatchResult:
    """Patch N21: partial backport of vllm#40898 (jianc99, OPEN).

    Two-file partial: speculators/algos.py preserves SWA config keys
    (layer_types, use_sliding_window, sliding_window, max_window_layers)
    + v1/spec_decode/dflash.py forces causal=True on sliding-window
    layer attention metadata.

    qwen3_dflash.py model class changes NOT backported — 7+ sub-patches
    with multi-line context, fragile. Wait for upstream merge or apply
    manually. Genesis partial preserves config + metadata correctness
    so the upstream merge auto-activates cleanly.

    Composes with PN24 (gpu_model_runner +1 shift). Both can coexist.

    Status: opt-in via GENESIS_ENABLE_PN21_DFLASH_SWA=1.
    Default OFF. Auto-no-op on upstream merge (drift markers).
    """
    return _wiring_text_patch(
        "PN21 DFlash SWA support partial backport (vllm#40898 backport)",
        "patch_N21_dflash_swa_support",
    )


@register_patch("PN22 Local argmax for TP draft (vllm#39419 backport)")
def apply_patch_N22_local_argmax_tp() -> PatchResult:
    """Patch N22: backport of vllm#39419 (EanWang, OPEN).

    Adds get_top_tokens() plumbing to Qwen3 and Qwen3-DFlash model
    classes — enables vocab-parallel argmax per TP rank instead of
    all-gathering full logits. +9.4-30.6% TPS on TP>=2 + draft model
    per PR author measurement.

    LogitsProcessor.get_top_tokens() callsite already in our pin
    (PR #34049 merged). This patch is pure plumbing.

    Llama and Eagle3 parts of upstream PR not backported — Genesis
    does not run those models.

    Status: opt-in via GENESIS_ENABLE_PN22_LOCAL_ARGMAX_TP=1.
    Default OFF. Auto-no-op once vllm#39419 merges.
    """
    return _wiring_text_patch(
        "PN22 Local argmax for TP draft (vllm#39419 backport)",
        "patch_N22_local_argmax_tp",
    )


@register_patch("PN24 DFlash aux layer +1 indexing fix (vllm#40727 backport)")
def apply_patch_N24_dflash_aux_layer_indexing() -> PatchResult:
    """Patch N24: backport of vllm#40727 (benchislett, OPEN).

    One-line semantic fix in `_get_eagle3_aux_layers_from_config` —
    adds `+1` to DFlash's target_layer_ids to convert 0-indexed
    DFlash semantics to 1-indexed Eagle3 aux semantics. Without
    the shift, every aux hidden state was read from the wrong layer.

    Empirical: AL gsm8k 6.18→6.42 per PR author measurement.

    Status: opt-in via GENESIS_ENABLE_PN24_DFLASH_AUX_LAYER_FIX=1.
    Default OFF. Auto-no-op once vllm#40727 merges (drift marker).
    """
    return _wiring_text_patch(
        "PN24 DFlash aux layer +1 indexing fix (vllm#40727 backport)",
        "patch_N24_dflash_aux_layer_indexing",
    )


@register_patch("PN17 FA2 softmax_lse runtime clamp (Issue #11 Cliff 1 mechanism A)")
def apply_patch_N17_fa2_softmax_lse_clamp() -> PatchResult:
    """Patch N17: Genesis-original 2026-04-30 — runtime clamp on FA2
    softmax_lse over-allocation.

    Replaces `max_seqlen_k = attn_metadata.max_seq_len` (which equals
    max_model_len during cudagraph capture per upstream design) with
    a runtime-only clamp to actual chunk max from `seqused_k.max()`.
    Cudagraph capture path falls back to original max_model_len
    behavior for shape stability.

    Closes Cliff 1 mechanism A (FA2 path); widens long-text-no-vision
    safe envelope from ~150K to ~205K. Mechanism B (FFN buffer cliff)
    is OUT OF SCOPE per Genesis Issue #11 dual-mechanism analysis.

    Status: opt-in via GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1.
    Diagnosis credit: noonghunna (cross-rig RTX 3090, Issue #11
    follow-up 2026-04-29).
    Default OFF.
    """
    return _wiring_text_patch(
        "PN17 FA2 softmax_lse runtime clamp (Issue #11 Cliff 1 mechanism A)",
        "patch_N17_fa2_softmax_lse_clamp",
    )


@register_patch("PN16 Lazy-reasoner request hook (per-request enable_thinking)")
def apply_patch_N16_lazy_reasoner() -> PatchResult:
    """Patch N16: Genesis-original 2026-04-29 — per-request decision on
    whether the `<think>` reasoning block adds value.

    Hybrid policy:
      - Respect explicit client `chat_template_kwargs.enable_thinking`
      - For short prompts without tools/schema/reasoning-signals → force
        enable_thinking=False (variant 1)
      - Otherwise allow with optional max-thinking-tokens cap (variant 4
        Phase 2 — stub for now)

    Goal: reduce wasted reasoning tokens + TTFT on trivial prompts
    without retry-induced 2× latency/load.

    Status: opt-in via GENESIS_ENABLE_PN16_LAZY_REASONER=1.
    Threshold: GENESIS_PN16_THRESHOLD_CHARS (default 300).
    Default OFF.
    """
    name = "PN16 Lazy-reasoner request hook (per-request enable_thinking)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.middleware import patch_N16_lazy_reasoner
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_N16_lazy_reasoner.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P86 ngram batch_propose O(N+K) direct-fill (vllm#40876 backport)")
def apply_patch_86_ngram_batch_propose_linear() -> PatchResult:
    """Patch 86: backport of vllm#40876 (aaronagent) — replaces the
    O(N*K) `i in valid_ngram_requests` list-membership scan in
    NgramProposer.batch_propose with an O(N+K) direct-fill loop.

    Original (O(N*K) due to list-membership scan):

        draft_token_ids: list[list[int]] = []
        ...
        for i in range(num_requests):
            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(...)
            else:
                draft_token_ids.append([])

    Patched (O(N+K) direct fill):

        draft_token_ids: list[list[int]] = [[] for _ in range(num_requests)]
        for i in valid_ngram_requests:
            num_drafts = self.valid_ngram_num_drafts[i]
            if num_drafts > 0:
                draft_token_ids[i] = self.valid_ngram_draft[i, :num_drafts].tolist()

    Genesis prod runs max_num_seqs=2 + prompt_lookup_min=8 — at N=2/K=2
    the difference is ns-scale. Real wins are at high-concurrency
    multi-user serving (N=64/K=32 saves ~1952 membership ops/batch).
    Algorithmic improvement, no behavioral change.

    Status: opt-in via GENESIS_ENABLE_P86=1. Default OFF.
    """
    name = "P86 ngram batch_propose O(N+K) direct-fill (vllm#40876 backport)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_86_ngram_batch_propose_linear
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_86_ngram_batch_propose_linear.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P85 Hybrid fine-shadow prefix cache (MambaManager fix for vllm#38182 followup)")
def apply_patch_85_hybrid_fine_shadow_prefix_cache() -> PatchResult:
    """Patch 85: Genesis-original architectural fix for vLLM v1 hybrid
    prefix-cache breakage on Mamba/GDN models.

    Discovery: 6-round empirical investigation + deep code analysis
    identified TWO mismatches that combine to make hybrid prefix-cache
    non-functional:
      (A) MambaManager.cache_blocks early-returns for short prompts
          (num_full_blocks = num_tokens // self.block_size = 0).
      (B) Mamba align-mode pads with null_blocks → 0 entries inserted
          even when num_full_blocks > 0.

    P85 patches MambaManager to:
      1. cache_blocks() also registers `scale_factor = block_size /
         hash_block_size` shadow fine-hash entries pointing to the
         SAME real KVCacheBlock(s).
      2. find_longest_cache_hit() prefers fine-grained scan, with
         eviction-safety: re-derives the coarse hash from current
         request fine hashes and verifies cached_block.block_hash
         matches before returning.

    Memory layout / ref-count untouched (shadows are pure lookup keys).

    Constraints:
      - Requires P84 (GENESIS_ENABLE_P84=1 + GENESIS_P84_HASH_BLOCK_SIZE=N)
        for fine hashes to exist.
      - Architectural limit: cannot help prompts < self.block_size
        (Mamba state genuinely uncached at sub-block boundaries).

    Status: opt-in via GENESIS_ENABLE_P85=1. Default OFF.
    """
    name = "P85 Hybrid fine-shadow prefix cache (MambaManager fix for vllm#38182 followup)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.kv_cache import patch_85_hybrid_fine_shadow_prefix_cache
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_85_hybrid_fine_shadow_prefix_cache.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P75 Auto-enable Suffix Decoding (vllm#25784 Arctic Inference)")
def apply_patch_75_suffix_decoding_enable() -> PatchResult:
    """Patch 75: operator-convenience auto-swap of speculative method from
    "ngram" to "suffix" (Arctic Inference Suffix Decoding) when
    `GENESIS_ENABLE_P75_SUFFIX_DECODING=1`.

    Suffix Decoding (PR #25784, MERGED 2025-11-03, present in our pin) builds
    per-prompt suffix trees with branch-frequency stats and speculates a
    DYNAMIC number of tokens per step (vs ngram's fixed
    num_speculative_tokens). Per arXiv 2411.04975 (NeurIPS 2025): up to 2.8×
    over EAGLE on agentic workloads.

    On our config (Qwen3.6-A3B-FP8 + 2× A5000), expected:
      - Tool-call (heavy repeats): +40-60% TPS over current 75 tok/s strict-ngram
      - Free-form text: +15-25% over current 46 tok/s (suffix tree handles
        short repeats that pure ngram misses with prompt_lookup_min=8)

    Dependency: `pip install arctic-inference` (added to test container
    entrypoint). If missing, P75 logs warning and keeps method=ngram (safe).

    Status: opt-in via GENESIS_ENABLE_P75_SUFFIX_DECODING=1.
    """
    name = "P75 Auto-enable Suffix Decoding (vllm#25784 Arctic Inference)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_75_suffix_decoding_enable
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_75_suffix_decoding_enable.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P74 Auto chunk-clamp via long_prefill_token_threshold (P72 companion)")
def apply_patch_74_chunk_clamp() -> PatchResult:
    """Patch 74: auto-clamp `SchedulerConfig.long_prefill_token_threshold`
    to GENESIS_PREALLOC_TOKEN_BUDGET when user runs with
    `--max-num-batched-tokens > 4096` (typically via P72 unblock).

    Companion safety net to P72: prevents the prefill-chunk-overflow
    regression discovered in v7.42 testing where P28 GDN core_attn_out
    buffer (sized at 4096) was overrun by a 5664-token prefill chunk on
    long-context (180K) requests.

    Mechanism: at SchedulerConfig.__post_init__, if user did not set
    explicit `long_prefill_token_threshold`, AND P74 env enabled, AND
    GENESIS_PREALLOC_TOKEN_BUDGET < max_num_batched_tokens, set
    `long_prefill_token_threshold = budget`. Decode batches still
    consume up to `max_num_batched_tokens` (multi-seq parallelism
    preserved). Only prefill chunks get clamped. Zero VRAM cost.

    Status: opt-in via GENESIS_ENABLE_P74_CHUNK_CLAMP=1.
    Recommended ON whenever GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 AND
    --max-num-batched-tokens > 4096.
    """
    name = "P74 Auto chunk-clamp via long_prefill_token_threshold (P72 companion)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.compile_safety import patch_74_chunk_clamp
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_74_chunk_clamp.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P72 profile_run M cap (unblocks --max-num-batched-tokens>4096 on MoE)")
def apply_patch_72_profile_run_cap() -> PatchResult:
    """Patch 72: workaround for Dynamo fake-tensor mismatch when running with
    `--max-num-batched-tokens > 4096` on MoE models.

    Root cause: profile_run calls `_dummy_run(self.max_num_tokens, is_profile=True)`
    which traces MoE forward with topk_ids shape (M, top_k). For M=8192 + top_k=8,
    `topk_ids.numel() = 65536`. Dynamo specializes 65536 in one trace branch and
    leaves it symbolic (16*s72) in another, then can't reconcile.

    Fix: cap M passed to _dummy_run to GENESIS_PROFILE_RUN_CAP_M (default 4096).
    Memory profile delta < 1MB (negligible vs 35GB model weights). Real runtime
    batches up to 8192 still go through the same compiled graph (Dynamo doesn't
    re-trace; symbolic shape covers both M=4096 and M=8192).

    For our 2-seq MTP K+1=4 interactive workload, real per-step gain is <0.5%.
    The headroom is for prefill chunk size, relevant when ISL > 4096 in
    aggregator multi-turn scenarios.

    Status: opt-in via GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1.

    Tunable knobs:
      - GENESIS_PROFILE_RUN_CAP_M (default 4096) — cap value
      - GENESIS_PROFILE_RUN_CAP_LOG (default 1) — log when cap fires
    """
    name = "P72 profile_run M cap (unblocks --max-num-batched-tokens>4096 on MoE)"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.compile_safety import patch_72_profile_run_cap
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_72_profile_run_cap.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P67b TurboQuant spec-verify forward() routing (FULL CG enable)")
def apply_patch_67b_spec_verify_routing() -> PatchResult:
    """Patch 67b: companion to P67 — adds dispatch branch in TurboQuant
    `forward()` BEFORE prefill/decode classification, intercepting K+1
    spec-verify batches and routing them through the P67 kernel directly.

    Bypasses `_prefill_attention` entirely for K+1 batches → avoids the
    upstream `tolist_cudagraph_fix` bypass crash (`cudaErrorStreamCapture
    Invalidated`) under FULL cudagraph capture. Combined with reverting
    P65 cudagraph downgrade, enables `FULL_AND_PIECEWISE` mode for spec-
    decode → expected +20-30% TPS on top of P67.

    Same env flag as P67: GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1.
    """
    name = "P67b TurboQuant spec-verify forward() routing"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_67b_spec_verify_routing
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_67b_spec_verify_routing.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P59 Qwen3 reasoning embedded tool_call recovery")
def apply_patch_59_qwen3_reasoning_tool_call_recovery() -> PatchResult:
    """Patch 59: Backport of upstream PR vllm#39055 (ZenoAFfectionate, OPEN).

    Empirical candidate for #40831 / our degenerate-output bug after P58
    (#40768 backport) was empirically disproven 2026-04-25 in blue/green test.

    Qwen3.5/3.6 models can emit XML tool_call blocks INSIDE <think>...</think>
    reasoning. The downstream qwen3_coder tool parser only inspects content,
    so embedded tool_calls in reasoning are lost — manifests as empty
    tool_calls OR garbage XML fragments leaking into JSON arguments
    (parameter=city, <<argname>, </parameter, etc.).

    Composes with our existing P12 (Qwen3 tool_call reasoning fix v2):
      - P12 handles the </think>-absent case via implicit tool_call end
      - P59 handles the </think>-present case where tool_call is nested
        inside reasoning

    Status: opt-in via GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY=1.

    Credit:
      - Upstream fix: @ZenoAFfectionate (vllm#39055).
      - Bug surface in our family: @meitalbensinai (Qwen 3.6 30b),
        @epheien (27b + 397b streaming), @jogoossens.
    """
    name = "P59 Qwen3 reasoning embedded tool_call recovery"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.structured_output import patch_59_qwen3_reasoning_tool_call_recovery
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_59_qwen3_reasoning_tool_call_recovery.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P58 async-scheduler -1 placeholder fix")
def apply_patch_58_async_placeholder_fix() -> PatchResult:
    """Patch 58: ROOT-CAUSE fix for vllm-project/vllm#40831 / #40807 / #40756 /
    #37159 — backport of upstream PR vllm#40768 (z1ying, OPEN at time of
    writing).

    Async scheduler shipped `[-1] * num_spec_tokens` as a shared list reference
    every step; worker-side `_prepare_input_ids` overwrite path skips for
    newly-scheduled requests (`prev_positions[i] < 0`) → -1s reach GPU
    embedding lookup → either crash (V100 IMA #37159 / #40756) or garbage
    propagation as degenerate token loop (#40831 / #40807).

    The fix: track placeholder *intent* as a counter on Request, materialize
    `[-1, ...]` only when `request_id in prev_step_scheduled_req_ids` so
    worker-side overwrite is guaranteed to land.

    Touches three files in vllm v1 (request.py + async_scheduler.py +
    scheduler.py). Idempotent + anchor-safe + auto-no-op once #40768 lands
    upstream.

    Status: opt-in via GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1. Independent
    of TurboQuant — bug class affects ALL spec-decode workloads under async
    scheduling. P56 (deprecated routing-layer workaround) and P57 v2
    (buffer-shape workaround) become redundant once P58 closes the actual
    root cause.

    Credit:
      - Upstream fix: @z1ying (vllm#40768).
      - Bug surface in our model family: @SongXiaoMao (#40756), @sweihub
        (#37159), @noonghunna (#40807, #40831).
      - Cross-rig confirmation: independent isolation by @noonghunna
        (Qwen3.6-27B + 3090) and Genesis (Qwen3-Next-35B + 2× A5000).
    """
    name = "P58 async-scheduler -1 placeholder fix"
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_58_async_scheduler_placeholder_fix
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_58_async_scheduler_placeholder_fix.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P57 TQ spec-decode capture-safe buffers")
def apply_patch_57_spec_decode_capture_safe() -> PatchResult:
    """Patch 57: REAL FIX (proof-of-concept) for vllm-project/vllm#40831.

    Addresses the architectural gap surfaced after deep-diving the
    GDN attention pattern at gdn_attn.py:103-115. TurboQuant declares
    `supports_spec_as_decode=False` AND pre-allocates decode buffers at
    `B=max_num_seqs` shape. Spec-decode batches with q_len=1+num_spec
    cannot fit the captured cudagraph's decode shape — buffer addresses
    captured at warmup don't match runtime addresses → token corruption
    visible as `for for`, `age age`, `<function=call`, etc.

    P57 fixes both layers:
      1. `supports_spec_as_decode = True` based on speculative_config
      2. Buffer alloc B = max_num_seqs * (1 + num_speculative_tokens)

    Status: opt-in via GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE=1.
    Experimental — pending server validation that demonstrates clean
    output WITHOUT cudagraph_mode=NONE workaround. If verified, this
    is a candidate upstream PR.

    Credit: bug surface @noonghunna (vllm#40807, #40831 + six-probe
    ladder noonghunna/qwen36-27b-single-3090@de1d1afa). Reference
    implementation pattern: gdn_attn.py:103-115 by vLLM team.
    """
    name = "P57 TQ spec-decode capture-safe buffers"
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
        from vllm._genesis.wiring.spec_decode import patch_57_spec_decode_capture_safe_buffers
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_57_spec_decode_capture_safe_buffers.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P56 TQ spec-decode safe-path guard")
def apply_patch_56_spec_decode_guard() -> PatchResult:
    """Patch 56: Workaround for vllm-project/vllm#40831 — TurboQuant ×
    spec-decode degenerate token loops.

    TurboQuant attention backend declares `supports_spec_as_decode=False`
    at `turboquant_attn.py:192` and lacks a varlen kernel analogous to
    FlashAttention's. Spec-decode batches (q_len > 1) get routed through
    a per-row synthetic-decode fast path that breaks GQA causal semantics
    across draft tokens — symptom: degenerate output loops.

    Tightens the fast-path entry condition from
    `q_len <= _CONTINUATION_DECODE_THRESHOLD` to `q_len == 1`, forcing
    spec-decode batches through `_continuation_prefill` (causal-correct
    `flash_attn_varlen_func` path).

    Status: opt-in (`GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=1`).

    Credit: bug surface @noonghunna (vllm-project/vllm#40807, #40831).
    """
    name = "P56 TQ spec-decode safe-path guard"
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
        from vllm._genesis.wiring.spec_decode import patch_56_spec_decode_decode_path_guard
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_56_spec_decode_decode_path_guard.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


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
        from vllm._genesis.wiring.legacy import patch_44_tq_mixed_attn_out
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
        from vllm._genesis.wiring.legacy import patch_46_gdn_gating_buffers
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
        from vllm._genesis.wiring.legacy import patch_7b_gdn_dual_stream_customop
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
        from vllm._genesis.wiring.legacy import patch_40_tq_grouped_decode
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
        from vllm._genesis.wiring.legacy import patch_39_fla_kkt_buffer
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
        from vllm._genesis.wiring.legacy import patch_38_tq_continuation_memory
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
        from vllm._genesis.wiring.legacy import patch_14_block_table
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

    # GPU profile + per-patch recommendations (suggest-only, never auto-enables)
    try:
        from vllm._genesis.gpu_profile import print_recommendations
        rec_text = print_recommendations(stream=None)
        for line in rec_text.split("\n"):
            log.info(line)
    except Exception as e:
        log.debug("[gpu_profile] recommendation skipped: %s", e)

    # [Phase 5b plugins] Discover + register community plugin patches
    # via setuptools entry-points. OPT-IN: only fires when
    # GENESIS_ALLOW_PLUGINS=1. Default behavior: zero foreign code loaded.
    try:
        from vllm._genesis.compat.plugins import (
            register_plugins as _register_genesis_plugins,
        )
        n_plugins = _register_genesis_plugins()
        if n_plugins > 0:
            log.info(
                "[Genesis plugins] registered %d community patch(es) via "
                "entry-points (lifecycle=community).", n_plugins,
            )
    except Exception as e:
        log.debug("[plugins] discovery skipped: %s", e)

    # [Phase 5c apply_callable] After core patches finish, walk plugins
    # whose env flags are set + call their apply_callable. Plugin failures
    # are isolated (logged, counted, never crash apply_all). Skipped if
    # GENESIS_ALLOW_PLUGINS gate is closed.
    if apply:
        try:
            from vllm._genesis.compat.plugins import apply_all_plugins
            plugin_stats = apply_all_plugins()
            if plugin_stats.get("total", 0) > 0:
                log.info(
                    "[Genesis plugins] apply pass: total=%d applied=%d "
                    "skipped=%d failed=%d",
                    plugin_stats["total"], plugin_stats["applied"],
                    plugin_stats["skipped"], plugin_stats["failed"],
                )
        except Exception as e:
            log.debug("[plugins] apply pass skipped: %s", e)

    # [Phase 5d telemetry] Opt-in anonymized telemetry. Default OFF —
    # only fires when GENESIS_ENABLE_TELEMETRY=1. Even when ON, only
    # saves locally. Network upload is a separate gate
    # (GENESIS_TELEMETRY_UPLOAD=1) and is currently a no-op until the
    # community dashboard is live.
    try:
        from vllm._genesis.compat.telemetry import (
            is_enabled as _telemetry_is_enabled,
            collect_report as _telemetry_collect_report,
            save_report as _telemetry_save_report,
        )
        if _telemetry_is_enabled():
            report = _telemetry_collect_report()
            path = _telemetry_save_report(report)
            if path:
                log.info(
                    "[Genesis telemetry] anonymized report saved → %s "
                    "(no network upload — see telemetry CLI)", path,
                )
    except Exception as e:
        log.debug("[telemetry] save skipped: %s", e)

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
                # 2026-04-28: anchor drift / required_anchor_missing is a
                # latent risk (patch silently not protecting). Surface as
                # WARNING so operators notice in boot logs. Other skip
                # reasons (opt-in, deprecated, redundant) stay at INFO.
                _is_drift = (
                    "required anchor" in result.reason.lower()
                    or "required_anchor_missing" in result.reason.lower()
                    or "anchor not found" in result.reason.lower()
                    or "ambiguous_anchor" in result.reason.lower()
                )
                if _is_drift:
                    log.warning("[Genesis] DRIFT skipped: %s — %s",
                                result.name, result.reason)
                else:
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

    # [Genesis v7.65 / Cliff 8 hardening] Surface partial-apply warnings.
    # Silent anchor-drift / ambiguous-anchor / anchor-missing skips were
    # the class noonghunna flagged in club-3090 discussion #19. Drift
    # detection works correctly, but the user-visible summary previously
    # buried the signal in the same `skipped` count as opt-in OFF. Now
    # warnings are pulled out and logged individually at WARNING level.
    if stats.partial_apply_warnings:
        log.warning(
            "[Genesis] %d partial-apply warning(s) — patch(es) failed to "
            "match expected source pattern. Review below to confirm anchor "
            "drift vs upstream change vs config issue:",
            stats.partial_apply_warnings_count,
        )
        for r in stats.partial_apply_warnings:
            log.warning("[Genesis] ⚠️  %s — %s", r.name, r.reason)

    # [Genesis v7.13] Emit Dispatcher v2 apply matrix as a single readable
    # block. Only matters for patches that route through dispatcher.should_apply
    # (P56-P62 currently); other patches get only the per-line INFO above.
    try:
        from vllm._genesis.dispatcher import log_apply_matrix
        log_apply_matrix()
    except Exception as e:
        log.debug("[Genesis] dispatcher matrix dump failed (non-fatal): %s", e)

    # [Genesis A3/D2] Validate dependencies / conflicts on the actual
    # APPLY set. Static registry validation runs first (cheap, catches
    # typos in requires_patches/conflicts_with refs), then runtime plan
    # check. Issues are logged at ERROR/WARNING level — we do NOT abort
    # boot here because operators may have legitimate reasons for unusual
    # combinations during diagnosis.
    try:
        from vllm._genesis.dispatcher import (
            validate_registry, validate_apply_plan,
            log_validation_issues, get_apply_matrix,
        )
        static_issues = validate_registry()
        if static_issues:
            log_validation_issues(static_issues)
        applied_set = {d["patch_id"] for d in get_apply_matrix() if d["applied"]}
        plan_issues = validate_apply_plan(applied_set)
        log_validation_issues(plan_issues)
    except Exception as e:
        log.debug("[Genesis] dispatcher validator unavailable: %s", e)

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
            mod = importlib.import_module(
                _resolve_wiring_module(wiring_module)
            )
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

    # [v7.13 Dispatcher v2] dump apply matrix at end of boot — single
    # readable summary block instead of grep-ing scattered INFO lines.
    try:
        from vllm._genesis.dispatcher import log_apply_matrix
        log_apply_matrix()
    except Exception as e:
        log.debug("[Genesis] Dispatcher v2 matrix dump unavailable: %s", e)

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
