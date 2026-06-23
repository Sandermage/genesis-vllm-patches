# SPDX-License-Identifier: Apache-2.0
"""Patch PN296 — Genesis GPU Architecture Profile boot-time initializer.

Genesis-original 2026-06-05 — boots the architecture profiler and
auto-sets sane env defaults BEFORE any other Genesis patch runs.

================================================================
WHY
================================================================

Upstream vllm has many code paths that branch on
`current_platform.is_device_capability(N)`. Those branches were designed
when the new arch was the primary target. Older Ampere SM 8.x often
falls through to a generic default that wasn't tuned for our resource
budget — silent slowdowns.

Genesis already has many env vars (e.g. `VLLM_MARLIN_FP32_REDUCE=0`)
that operators must set MANUALLY in launcher to compensate. This patch:

1. Detects the actual GPU at boot via
   `sndr.detection.gpu_arch_profile.get_gpu_arch_profile()`.

2. Logs the full profile (1 line, easy to grep in operator logs).

3. AUTO-SETS env vars that should follow from the detection, IF they
   are not already set by the operator. Operator overrides win.

Auto-set rules (only fire if env not already set):
  - SM 8.x consumer (no FP32 TCs): VLLM_MARLIN_FP32_REDUCE=0
  - SM 9.0+ (has FP32 TCs): VLLM_MARLIN_FP32_REDUCE=1 (default)
  - SM 8.x (100KB shared): GENESIS_TRITON_AUTOTUNE_MAX_WARPS=4
  - SM 9.0+ (228KB shared): GENESIS_TRITON_AUTOTUNE_MAX_WARPS=8
  - Always: GENESIS_GPU_ARCH_DETECTED=<sm_major>.<sm_minor>

Operator override pattern:
  Set the env var in launcher BEFORE this patch fires — Genesis will
  detect it set, leave it untouched, and log "operator override
  preserved" for transparency.

================================================================

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-05.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn296_arch_profile_init")


_APPLIED = False


def _auto_set(env_name: str, value: str, reason: str) -> bool:
    """Set env if not already set. Returns True if changed."""
    existing = os.environ.get(env_name, None)
    if existing is not None and existing != "":
        log.info(
            "[PN296 arch] %s=%r operator override preserved (would have set %r — %s)",
            env_name, existing, value, reason,
        )
        return False
    os.environ[env_name] = value
    log.warning(
        "[PN296 arch] auto-set %s=%s (reason: %s)",
        env_name, value, reason,
    )
    return True


def apply() -> tuple[str, str]:
    """Detect GPU arch + auto-set follow-on env vars."""
    global _APPLIED

    if os.environ.get(
        "GENESIS_ENABLE_PN296_ARCH_PROFILE_INIT", ""
    ).lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            "PN296 default OFF — set "
            "GENESIS_ENABLE_PN296_ARCH_PROFILE_INIT=1 to engage."
        )

    try:
        from sndr.detection.gpu_arch_profile import (
            get_gpu_arch_profile,
        )
    except Exception as e:
        return "failed", f"arch profile module import failed: {e}"

    profile = get_gpu_arch_profile()
    if profile is None:
        return "skipped", "GPU not detected (non-CUDA or detection failed)"

    auto_set_count = 0

    # ─── Marlin FP32 reduce — only beneficial when hardware has FP32 TCs ──
    if not profile.has_fp32_tensor_cores:
        if _auto_set(
            "VLLM_MARLIN_FP32_REDUCE", "0",
            f"SM {profile.sm_string} has no FP32 tensor cores — "
            f"FP32 reduce is pure overhead on this arch (advisory: env is "
            f"inert unless a wire patch consumes it, e.g. P23_WIRE; "
            f"+1.5-3% TGS when wired)",
        ):
            auto_set_count += 1
    else:
        if _auto_set(
            "VLLM_MARLIN_FP32_REDUCE", "1",
            f"SM {profile.sm_string} has FP32 TCs — FP32 reduce maintains "
            f"numerical stability with no perf penalty on this arch",
        ):
            auto_set_count += 1

    # ─── Triton autotune budget hint for downstream patches ───────────
    if _auto_set(
        "GENESIS_TRITON_AUTOTUNE_MAX_WARPS",
        str(profile.max_safe_num_warps),
        f"SM {profile.sm_string} has {profile.shared_mem_kb_per_sm}KB "
        f"shared/SM → num_warps>{profile.max_safe_num_warps} risks spilling",
    ):
        auto_set_count += 1

    if _auto_set(
        "GENESIS_TRITON_AUTOTUNE_MAX_STAGES",
        str(profile.max_safe_num_stages),
        f"SM {profile.sm_string} pipelined-load shared-mem budget",
    ):
        auto_set_count += 1

    # ─── Diagnostic stamp (always set, useful for support) ────────────
    os.environ["GENESIS_GPU_ARCH_DETECTED"] = profile.sm_string
    os.environ["GENESIS_GPU_ARCH_NAME"] = profile.arch_name
    os.environ["GENESIS_GPU_DEVICE_NAME"] = profile.device_name

    # ─── PN286 SM 8.6 gate (informational) ────────────────────────────
    # PN286 already self-detects SM 8.6. We just log it as a sanity check.
    if profile.is_ampere_consumer:
        log.warning(
            "[PN296 arch] SM 8.6 Ampere detected (%s) — PN286 FA layout "
            "revert is the right default for this GPU. Verify "
            "GENESIS_ENABLE_PN286_FA_LAYOUT_REVERT_SM86=1 in launcher.",
            profile.device_name,
        )

    _APPLIED = True
    return "applied", (
        f"PN296 installed: GPU={profile.device_name} arch={profile.arch_name} "
        f"SM={profile.sm_string}. Auto-set {auto_set_count} follow-on env "
        f"vars based on architecture (max_warps={profile.max_safe_num_warps}, "
        f"max_stages={profile.max_safe_num_stages}, "
        f"fp32_reduce={profile.should_use_fp32_reduce}). "
        f"Diagnostic stamps written to env."
    )


def is_applied() -> bool:
    return _APPLIED
