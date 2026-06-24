# SPDX-License-Identifier: Apache-2.0
"""Genesis defensive guards — canonical vendor/chip/model/dependency detection.

Philosophy: we fix, we do not break.
Every helper is fail-safe: returns a safe default (False/None) on any exception.
If detection cannot complete, we SKIP the patch — never crash the engine.

All detection patterns mirror upstream vLLM canonical sources:
  - vllm/platforms/interface.py   (Platform predicates, DeviceCapability)
  - vllm/platforms/cuda.py        (NvmlCudaPlatform.get_device_capability)
  - vllm/platforms/rocm.py        (_GCN_ARCH parsing)

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Optional

log = logging.getLogger("genesis.guards")


# ─── torch.dynamo compatibility ────────────────────────────────────────────
# Genesis guards are eager-only diagnostic helpers (vendor / SM / version
# detection). They are commonly called from kernel paths that vLLM compiles
# with torch.compile / torch.dynamo (Marlin apply_weights, FP8 scaled MM,
# CUDA-graph capture, etc.).
#
# Two issues seen empirically (2026-04-28, with GENESIS_FORCE_MARLIN_W8A16=1
# on Qwen3.6-27B-INT8-AutoRound, Marlin path):
#   1. torch.dynamo IGNORES `@functools.lru_cache` / `@functools.cache`
#      wrappers and traces the underlying function instead → crash inside
#      `current_platform.get_device_capability()` which dynamo can't trace.
#   2. `@torch._dynamo.disable` raises 'Skip calling
#      torch.compiler.disable()d function' when invoked from inside an
#      already-traced compiled region (it cannot fall back to eager mid-trace).
#
# Robust fix: snapshot the platform-derived facts ONCE at module-load time
# (which happens eagerly during plugin register, BEFORE any torch.compile /
# dynamo run), and have all public guards return those captured constants.
# The functions become pure `return _MODULE_CONSTANT` — dynamo can trace
# through them with zero risk of touching unsupported APIs.
#
# A `_refresh()` helper is exposed for tests that monkey-patch
# `current_platform`. Production has no use case for it.
def _detect_platform() -> Optional[Any]:
    try:
        from vllm.platforms import current_platform
        return current_platform
    except Exception as e:
        log.debug("[guards] vllm.platforms.current_platform unavailable: %s", e)
        return None


def _detect_is_cuda(p: Optional[Any]) -> bool:
    try:
        return bool(p is not None and p.is_cuda())
    except Exception:
        return False


def _detect_is_rocm(p: Optional[Any]) -> bool:
    try:
        return bool(p is not None and p.is_rocm())
    except Exception:
        return False


def _detect_is_xpu(p: Optional[Any]) -> bool:
    try:
        return bool(p is not None and p.is_xpu())
    except Exception:
        return False


def _detect_is_cpu(p: Optional[Any]) -> bool:
    try:
        return bool(p is not None and p.is_cpu())
    except Exception:
        return False


def _detect_is_cuda_alike(p: Optional[Any]) -> bool:
    try:
        return bool(p is not None and p.is_cuda_alike())
    except Exception:
        return False


def _detect_compute_capability(
    p: Optional[Any], is_cuda: bool
) -> Optional[tuple[int, int]]:
    if not is_cuda or p is None:
        return None
    try:
        cc = p.get_device_capability()
        if cc is None:
            return None
        return (cc.major, cc.minor)
    except Exception as e:
        log.debug("[guards] get_device_capability failed: %s", e)
        return None


# Module-level captured constants — populated once at import time (eager
# context). Public functions just return these. dynamo / torch.compile can
# trace through `return _MODULE_CONST` trivially without ever touching
# platform internals.
_PLATFORM: Optional[Any] = None
_IS_CUDA: bool = False
_IS_ROCM: bool = False
_IS_XPU: bool = False
_IS_CPU: bool = False
_IS_CUDA_ALIKE: bool = False
_COMPUTE_CAPABILITY: Optional[tuple[int, int]] = None


def _refresh() -> None:
    """Re-snapshot platform facts. Called once at module import; tests can
    re-invoke after monkey-patching `vllm.platforms.current_platform`."""
    global _PLATFORM, _IS_CUDA, _IS_ROCM, _IS_XPU, _IS_CPU, _IS_CUDA_ALIKE
    global _COMPUTE_CAPABILITY
    _PLATFORM = _detect_platform()
    _IS_CUDA = _detect_is_cuda(_PLATFORM)
    _IS_ROCM = _detect_is_rocm(_PLATFORM)
    _IS_XPU = _detect_is_xpu(_PLATFORM)
    _IS_CPU = _detect_is_cpu(_PLATFORM)
    _IS_CUDA_ALIKE = _detect_is_cuda_alike(_PLATFORM)
    _COMPUTE_CAPABILITY = _detect_compute_capability(_PLATFORM, _IS_CUDA)


_refresh()


# ═══════════════════════════════════════════════════════════════════════════
#                          VENDOR / PLATFORM IDENTITY
# ═══════════════════════════════════════════════════════════════════════════

def _current_platform() -> Optional[Any]:
    """Return the snapshotted platform handle (set at module load).

    See module docstring for why this is a constant return now.
    """
    return _PLATFORM


def is_nvidia_cuda() -> bool:
    """True ONLY on NVIDIA CUDA (NOT ROCm).

    Returns the constant snapshot taken at module load. Trace-safe (pure
    `return _IS_CUDA`); see module docstring.
    """
    return _IS_CUDA


def is_amd_rocm() -> bool:
    """True on AMD ROCm. Trace-safe constant return."""
    return _IS_ROCM


def is_intel_xpu() -> bool:
    """True on Intel XPU. Trace-safe constant return."""
    return _IS_XPU


def is_cpu_only() -> bool:
    """True on CPU-only build. Trace-safe constant return."""
    return _IS_CPU


def is_cuda_alike() -> bool:
    """CUDA OR ROCm. Trace-safe constant return.

    ⚠️ TRAP: Do NOT use for NVIDIA-specific patches. Use is_nvidia_cuda().
    """
    return _IS_CUDA_ALIKE


# ═══════════════════════════════════════════════════════════════════════════
#                      NVIDIA COMPUTE CAPABILITY
# ═══════════════════════════════════════════════════════════════════════════

def get_compute_capability() -> Optional[tuple[int, int]]:
    """Return snapshotted (major, minor) compute capability for NVIDIA CUDA.

    Trace-safe constant return. Returns None on non-CUDA / failed detection.
    """
    return _COMPUTE_CAPABILITY


def is_sm_at_least(major: int, minor: int = 0) -> bool:
    """True if SM >= (major, minor). Trace-safe (reads module constant)."""
    cc = _COMPUTE_CAPABILITY
    if cc is None:
        return False
    return cc >= (major, minor)


def is_sm_exactly(major: int, minor: int) -> bool:
    """True if SM is exactly (major, minor). Trace-safe constant read."""
    return _COMPUTE_CAPABILITY == (major, minor)


def is_ampere_datacenter() -> bool:
    """NVIDIA A100 — SM 8.0."""
    return is_sm_exactly(8, 0)


def is_ampere_consumer() -> bool:
    """NVIDIA A5000 / A6000 / RTX 3090 — SM 8.6.

    Genesis prod baseline.
    """
    return is_sm_exactly(8, 6)


def is_ampere_any() -> bool:
    """Any Ampere (SM 8.x except Ada 8.9)."""
    cc = get_compute_capability()
    return cc is not None and cc[0] == 8 and cc[1] < 9


def is_ada_lovelace() -> bool:
    """NVIDIA RTX 4090 / L40 / RTX 6000 Ada — SM 8.9."""
    return is_sm_exactly(8, 9)


def is_hopper() -> bool:
    """NVIDIA H100 / H200 — SM 9.0."""
    return is_sm_exactly(9, 0)


def is_blackwell() -> bool:
    """NVIDIA Blackwell family — SM 10.x (datacenter/pro) OR SM 12.x (consumer).

    NVIDIA splits Blackwell across two SM major versions:
      * sm_10x (10.0/10.1/10.3): B100, B200, GB200, RTX PRO 6000 Blackwell
      * sm_120 (12.0): RTX 5090, 5080, 5070, 5060 (consumer)

    Fix for issue #20 (2026-05-04, club-3090 RTX 5090 user): original
    `cc[0] == 10` missed consumer Blackwell. Now `cc[0] in (10, 12)`.
    """
    cc = get_compute_capability()
    return cc is not None and cc[0] in (10, 12)


def is_blackwell_datacenter() -> bool:
    """NVIDIA Blackwell datacenter/pro (B100/B200/GB200/RTX PRO 6000) — SM 10.x only."""
    cc = get_compute_capability()
    return cc is not None and cc[0] == 10


def is_blackwell_consumer() -> bool:
    """NVIDIA Blackwell consumer (RTX 5090/5080/5070/5060) — SM 12.x only."""
    cc = get_compute_capability()
    return cc is not None and cc[0] == 12


def has_native_fp8() -> bool:
    """True if GPU has native FP8 tensor cores (SM >= 8.9).

    Includes Ada Lovelace (8.9), Hopper (9.0), Blackwell (10.0).
    Ampere (8.6) does NOT have native FP8 — uses emulation.
    """
    return is_sm_at_least(8, 9)


def pdl_support_expected() -> bool:
    """True if platform is expected to support Programmatic Dependent Launch.

    PDL (Programmatic Dependent Launch) is a Hopper+ / Blackwell feature. On
    older GPUs (Ampere / Ada), enabling PDL-related env vars has no effect
    at best; at worst it can trigger vLLM issue #40742 (Inductor autotune
    calls torch.cuda.synchronize() inside CUDA graph capture → illegal
    cuda operation → server crash at startup).

    Returns:
      True on SM >= 9.0 (Hopper, Blackwell, future).
      False on SM < 9.0 (Ampere consumer/datacenter, Ada Lovelace, pre-Ampere).
      False on non-NVIDIA.
    """
    return is_sm_at_least(9, 0)


def detect_pdl_env_misconfig() -> list[str]:
    """Detect user-set PDL env vars that aren't safe on this GPU.

    Reference: vLLM issue #40742 (2026-04-23) — CUDA graph capture crash when
    `TRTLLM_ENABLE_PDL=1` / `TORCHINDUCTOR_ENABLE_PDL=1` is set on GPUs where
    PDL is not fully supported, because Inductor autotune inserts a
    `torch.cuda.synchronize()` inside an active graph capture.

    Returns:
      List of env-var names that are set to truthy values but shouldn't be
      on this platform. Empty list means safe.
    """
    if pdl_support_expected():
        return []

    import os as _os
    misconfigured: list[str] = []
    for var in (
        "TRTLLM_ENABLE_PDL",
        "TORCHINDUCTOR_ENABLE_PDL",
    ):
        val = _os.environ.get(var, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            misconfigured.append(var)
    return misconfigured


# ═══════════════════════════════════════════════════════════════════════════
#                       AMD ROCm ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

@functools.cache
def _gcn_arch() -> str:
    """GCN architecture string (e.g. 'gfx942'). Empty string if not ROCm.

    Canonical source: vllm/platforms/rocm.py:187
    """
    if not is_amd_rocm():
        return ""
    try:
        from vllm.platforms import rocm as _rocm
        return getattr(_rocm, "_GCN_ARCH", "") or ""
    except Exception:
        return ""


def is_rocm_cdna2() -> bool:
    """AMD MI210 / MI250 — gfx90a (CDNA2 datacenter)."""
    return "gfx90a" in _gcn_arch()


def is_rocm_cdna3() -> bool:
    """AMD MI300X / MI325X — gfx942 / gfx950 (CDNA3 datacenter)."""
    arch = _gcn_arch()
    return "gfx942" in arch or "gfx950" in arch


def is_rocm_rdna() -> bool:
    """AMD Radeon RDNA3/4 — gfx11xx / gfx12xx (consumer)."""
    arch = _gcn_arch()
    return "gfx11" in arch or "gfx12" in arch


# ═══════════════════════════════════════════════════════════════════════════
#                   EXTERNAL DEPENDENCY VERSIONS (NEW v7.0)
# ═══════════════════════════════════════════════════════════════════════════

@functools.cache
def get_torch_version() -> Optional[tuple[int, int]]:
    """Returns (major, minor) torch version, or None on failure."""
    try:
        import torch
        parts = torch.__version__.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def is_torch_211_plus() -> bool:
    """True if torch >= 2.11 (required for vLLM v0.20.0+)."""
    v = get_torch_version()
    return v is not None and v >= (2, 11)


def is_torch_212_plus() -> bool:
    """True if torch >= 2.12 (forthcoming late 2026)."""
    v = get_torch_version()
    return v is not None and v >= (2, 12)


@functools.cache
def get_transformers_version() -> Optional[tuple[int, int, int]]:
    """Returns (major, minor, patch) transformers version, or None on failure."""
    try:
        import transformers
        parts = transformers.__version__.split(".")[:3]
        # Handle versions like "5.5.0rc1" by stripping non-digit suffix
        return tuple(int(''.join(c for c in p if c.isdigit())) for p in parts)
    except Exception:
        return None


def is_transformers_v5_plus() -> bool:
    """True if transformers >= 5.0.0 (required for vLLM v0.19.1+)."""
    v = get_transformers_version()
    return v is not None and v[0] >= 5


def is_transformers_v55_plus() -> bool:
    """True if transformers >= 5.5.0 (required for Gemma 4 support)."""
    v = get_transformers_version()
    return v is not None and v >= (5, 5, 0)


@functools.cache
def get_vllm_version_tuple() -> Optional[tuple[int, ...]]:
    """Returns (major, minor, patch) vllm version tuple, or None on failure.

    Example: vllm 0.20.0 -> (0, 20, 0)
    """
    try:
        import vllm
        parts = vllm.__version__.split(".")[:3]
        # Handle versions like "0.19.2rc1.dev8" by taking only leading digits
        result = []
        for p in parts:
            digits = ''.join(c for c in p.split('rc')[0].split('+')[0] if c.isdigit())
            result.append(int(digits) if digits else 0)
        return tuple(result)
    except Exception:
        return None


def is_vllm_020_plus() -> bool:
    """True if vllm >= 0.20.0."""
    v = get_vllm_version_tuple()
    return v is not None and v >= (0, 20, 0)


@functools.cache
def get_vllm_full_version_string() -> Optional[str]:
    """Returns the FULL vllm version string including local-pin suffix.

    Examples:
      "0.20.1rc1.dev16+g7a1eb8ac2"
      "0.20.2rc1.dev9+g01d4d1ad3"

    Returns None if vllm is not importable. This is the canonical pin
    identity used by `assert_vllm_pin_allowed` for protect-against-foot-gun
    enforcement.
    """
    try:
        import vllm
        return getattr(vllm, "__version__", None)
    except Exception:
        return None


# ───────────────────────────────────────────────────────────────────────
# vLLM pin allowlist — known-good pins this Genesis revision validated against.
#
# When a patcher boot encounters an UNKNOWN pin, two policy modes are
# possible (controlled by `GENESIS_VLLM_PIN_POLICY`):
#   - "warn" (default) — log a loud warning, continue
#   - "strict"          — log + sys.exit(2) — refuse to apply patches that
#                         were never validated against this pin
#
# Add new entries via PR review only. Each line should be paired with a
# CHANGELOG entry documenting the validated test surface.
# ───────────────────────────────────────────────────────────────────────
KNOWN_GOOD_VLLM_PINS: tuple[str, ...] = (
    # v7.65 PROD baseline (validated 2026-04-23 → 2026-05-04, 1470 tests)
    "0.20.1rc1.dev16+g7a1eb8ac2",
    # v7.70 pin-bump target (validated 2026-05-04, Test 4 boot+smoke+tool-call clean)
    "0.20.2rc1.dev9+g01d4d1ad3",
    # 2026-05-07 pin-bump candidate — picks up Sander's own merged PR #39931
    # (TurboQuant hybrid uniform quant, supersedes our P4 hybrid bypass) + PR
    # #40961 (cudagraph max_seq_len SWA fix). Anchor-drift audit pending; on
    # boot, observe genesis dispatcher for SKIP-on-anchor-drift entries before
    # promoting to PROD config use.
    "0.20.2rc1.dev60+ge47c98ef7",
    # 2026-05-07 — vllm dev93 (sha 51f22dcfd) — adds CPU GDN attention,
    # Gemma4 MTP support, multiple bugfixes. Validated on 35B PROD live boot:
    # API healthy, smoke 88 TPS gen, tool-call clean. Drift detection auto-
    # skipped P4 / P12 / P26 / PN19 (upstream merged equivalents). 0 patch
    # failures. PN79 4 patchers + Layer 0/4.5 fast-paths verified compatible.
    # Wave 8 canonical bench validated: 27B PROD 132.28 TPS, 35B PROD 232.36 TPS.
    "0.20.2rc1.dev93+g51f22dcfd",
    # 2026-05-11 Phase 2 bump — vllm nightly dev209+g5536fc0c0 (~116 dev-
    # increments / 4 days ahead of dev93). Validated on 27B PROD via
    # parallel throwaway test container (vllm-27b-bump-test, port 8102):
    #   - boot clean (5 min cold compile, API healthy)
    #   - smoke gen + tool-call OK (reasoning_parser=qwen3 splits content
    #     and reasoning into separate fields; 7/7 tool-call positive cases)
    #   - canonical bench (genesis_bench_suite.py --quick --ctx 8k, 5×5):
    #       wall_TPS 131.11 (CV 3.53%), TPOT 7.39 ms (CV 3.66%),
    #       TTFT 97.7 ms (CV 7.44%), 8K context stable
    #     vs Wave 8 dev93 baseline (132.28/7.31/97.1): -0.88% TPS /
    #     +1.10% TPOT / +0.62% TTFT — all within CV bands (net-neutral).
    #   - 0 patches with VERSION: skips (all 13 pin-gated patches in-range)
    #   - 0 new supersession candidates (deep-diff: every merged upstream_pr
    #     in our registry was already in dev93; window dev93→dev209 added
    #     only unrelated upstream commits)
    # Promoted to PROD start-script ~/start_pn95_2xa5000_test.sh by swapping
    # image vllm-genesis-pinned:dev93-2026-05-09 → vllm/vllm-openai:nightly.
    # Backup retained at ~/start_pn95_2xa5000_test.sh.bak.dev93-2026-05-11.
    "0.20.2rc1.dev209+g5536fc0c0",
    # 2026-05-14 — vllm nightly dev338+gbf0d2dc6d (~129 dev-increments /
    # 3 days ahead of dev209). Validated on both PROD models via short
    # smoke test (server up, /v1/chat/completions OK):
    #   - 27B INT4 TQ k8v4: 59 applied / 96 skipped / 0 failed; HTTP 200
    #     ready after 170s; inference 1.97s (1 short prompt).
    #   - 35B-A3B FP8: 57 applied / 98 skipped / 0 failed / 2 partial-apply
    #     warnings (P37 fused_marlin_moe path drift — fixed in same wave
    #     by trying experts/marlin_moe.py first; P103 GDN env not set on
    #     35B profile by design — 35B is dense MoE, no GDN). HTTP 200
    #     ready after ~275s; inference 1.50s, content "60".
    # See CHANGELOG "[v11.0.0+wave9_dev338]" for the upgrade summary.
    "0.20.2rc1.dev338+gbf0d2dc6d",
    # 2026-05-15 — v0.21.0 RELEASE + freshest nightly bf610c2f. The
    # `vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036`
    # image on Docker Hub is the actionable form (pushed 2026-05-15,
    # CUDA-13 native). Commit bf610c2f IS vllm#41674 (thinking_token_budget
    # bool inversion fix — what our PN67 backports). Other notable:
    #   - vllm#42660 Qwen3.5 chat template format fix
    #   - vllm#42604 DSV4-Pro FULL_AND_PIECEWISE (validates PN125 doc note)
    #   - vllm#41869 PD disagg NIXL GDN for Qwen3.5 (active maintenance)
    # Add three entries: v0.21.0 release tag, v0.21.1rc0 git tag form,
    # commit-sha form. PROMOTION_PENDING below until first bench cycle.
    "0.21.0",  # release tag
    "0.21.1rc0",  # git tag form
    "0.21.1rc0+gd735968f6d63",  # canonical dev-pin form (gh tag)
    "0.21.1rc0+gbf610c2f5676",  # docker hub nightly SHA (12-char)
    # Real version string vllm reports when the image
    # `nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036` is running:
    # `0.20.2rc1.dev371+gbf610c2f5`. The build script formats the
    # internal version string as `0.20.2rc1.dev<NNN>+g<SHA[:10]>`
    # rather than tag-based. We include both forms (commit-based +
    # nightly-tag-based) so pin-gate does not warn.
    "0.20.2rc1.dev371+gbf610c2f5",
    # 2026-05-28 — K.1.R pin bump audit target.
    # vllm/vllm-openai:nightly Docker Hub tag pushed 2026-05-28T06:17 UTC,
    # multi-arch manifest digest
    # sha256:674922aae790c2cbf45f4e844098d227b80d40a74bfc7797a444d213a221879f
    # (amd64 sha256:0cc9b1e1a15bcd780c4d730f807770b6c4e7b23d77332d956aa945d01c349c01).
    # Upstream commit 626fa9bba5663a5cf6a870debf031ee344ddb822 (2026-05-28T04:59:34Z),
    # "[BugFix] Fix blocked reasoning parsing with MRV2 (#43808)". Position:
    # 362 commits ahead of bf610c2f5 (our previous canonical dev371),
    # 354 commits ahead of v0.21.1rc0 tag, 13 commits diverged from
    # v0.22.0rc2 (40cf0206, 2026-05-27T16:30), 18 commits behind
    # v0.22.0rc3 (799c3afa, 2026-05-28T07:09 — 2 hours after this commit).
    # Closest annotated ancestor: v0.21.1rc0 (d735968f6d63, 2026-05-15).
    # Genesis PR audit (72 unique PRs in registry, 10 actually merged
    # upstream, only PR #41873 → PN82 merged in window dev371→626fa9bb,
    # case (a) byte-equivalent retire). Anchor drift check: 14/18 OK,
    # 4 files with anchor shifts (config/vllm.py, auto_gptq.py, inc.py,
    # algos.py) — covered by patches' upstream_drift_markers self-skip.
    # See sndr_private/planning/audits/K_1_R_PIN_BUMP_AUDIT_2026-05-28_RU.md.
    # PROMOTED 2026-05-30 (K.1.R.R.8.5): server-side validation complete.
    # 35B FP8 dense MoE bench cycle on 626fa9bb: 195.74 TPS / 6-7 tool-call
    # with P67 enabled (proper prod config). PN286 default_on flip validated
    # (+6.6% TPS on 35B). 27B bench narrowed -9% regression to TQ kernel ×
    # upstream interaction; documented at A11 (stays on dev371 until nsys).
    # Three form aliases: docker tag, setuptools_scm derived, internal-
    # version derived (last form confirmed at K.1.R.R.4 first boot).
    "0.21.1rc0+g626fa9bba5",   # setuptools_scm-derived (closest tag)
    "0.21.1rc0+g626fa9bba566", # setuptools_scm 12-char SHA form
    "nightly-626fa9bba5663a5cf6a870debf031ee344ddb822",  # docker tag form
    "0.20.2rc1.dev733+g626fa9bba5",  # internal-version derived (estimated)
    # 2026-06-05 — K.2 pin bump: jump through v0.22.0 (2026-05-29) +
    # v0.22.1 (2026-06-05) major releases. 285 commits ahead of 626fa9bba.
    # Upstream commit da1daf40bf18e5eaae04f26a80a537c8168a8bc2 (2026-06-05T03:47Z),
    # "[Bugfix] Exclude vision embedder from quantization in Gemma4 Unified (#44571)".
    # Genesis 89 upstream_pr backports cross-referenced with delta — 0 PRs
    # landed in window (no retire candidates). 35B boot smoke PASS on first
    # try (342 applied / 1 cosmetic 'fail' = PN299 idempotent re-detect of
    # already-applied anchors). 27B boot smoke PASS (361/0/471). PROMOTION
    # PENDING pending full bench validation (35B target 220 TPS, 27B 130).
    "0.22.1rc1.dev195+gda1daf40b",
    "0.22.1rc1.dev195+gda1daf40bf18e5eaae04f26a80a537c8168a8bc2",
    "nightly-da1daf40bf18e5eaae04f26a80a537c8168a8bc2",
    # ── PROD pin K.1.R.R.8.5 ratified 2026-06-09 ───────────────────────
    # Image: vllm/vllm-openai:nightly-303916e93 (sha256:bf610c2f5...)
    # PROD container: vllm-qwen3.6-35b-balanced-k3 on <user>@<host>
    # Session 2026-06-09: shipped PN340/341/345/346/347/348/350/361 (text-
    # patches verified via runtime probe) + PN126/128/130 (warmup, runtime
    # log evidence in Worker_TP* logs). PN50 re-anchored against the
    # split_ba helper (vllm#41126 rename). Sustained bench: wall_TPS
    # 218.56 (best in session, +18.56 vs 199 start), TPOT 4.464 ms,
    # Stability CV 0.41 %.
    "0.22.1rc1.dev259+g303916e93",
    "0.22.1rc1.dev259+g303916e93d66",
    "nightly-303916e93",
    "nightly-303916e93d66",
    # ── PROD pin PROMOTION dev491 ratified 2026-06-14 ──────────────────
    # Image: vllm/vllm-openai:nightly-1033ffac2 (0.22.1rc1.dev491+g1033ffac2).
    # The validated streaming-tool-call pin: upstream #45171 deleted
    # qwen3xml_tool_parser.py and remapped qwen3_xml->Qwen3CoderToolParser;
    # Genesis P64/P61c/PN56 are version-capped <dev491 (skip there) and PN392
    # retired, so the engine-native parser handles streaming tool-calls. dev259
    # retained as rollback (nightly-303916e93).
    "0.22.1rc1.dev491+g1033ffac2",
    "0.22.1rc1.dev491+g1033ffac2d66",
    "nightly-1033ffac2",
    # ── PROD pin PROMOTION 0.23.1 ratified 2026-06-17 ─────────────────
    # Image: vllm/vllm-openai:nightly-4c626633159887b0f2c962058c17c78f1434556d
    # (0.23.1rc1.dev101+g4c6266331, commit 4c626633, +128 commits over dev491).
    # FULL-FLEET validation 2026-06-17 — all four models boot with apply
    # failed=0, smoke + tool-call PASS, no regression:
    #   - 35B-A3B FP8 TQ k8v4 + MTP K=3: 210.7 TPS median = 101% of dev491
    #     (208), MTP accept 142%, streaming tool-calls + reasoning clean.
    #   - 27B Lorbus INT4 hybrid GDN+Mamba: coherent + tool-call.
    #   - Gemma4-31B-it AWQ TQ + MTP K=3: coherent (Paris / 6x7=42 /
    #     photosynthesis, no degenerate loop) + get_weather (gemma4 parser).
    #   - DiffusionGemma 26B-A4B FP8 block-diffusion TP=2: coherent + tool-call.
    # MTP root cause fixed: P67 multi-query spec-decode kernel version-cap
    # bumped to <0.24.0 (commit bc75dbfe). PN30 retired (upstream fused
    # postprocess kernel supersedes the DS conv spec-decode path), P29_HEAL
    # capped (target qwen3coder_tool_parser.py deleted by #45588). dev491
    # (nightly-1033ffac2) retained above as the previous/rollback pin.
    "0.23.1rc1.dev101+g4c6266331",
    "0.23.1rc1.dev101+g4c626633159",
    "nightly-4c626633",
    "nightly-4c626633159887b0f2c962058c17c78f1434556d",
    # ── PROD pin PROMOTION dev148 ratified 2026-06-19 ─────────────────
    # Image: vllm/vllm-openai:nightly-b4c80ec0f (0.23.1rc1.dev148+gb4c80ec0f,
    # commit b4c80ec0f, +47 commits over dev101). This is the live rig pin:
    # the 35B PROD runs it and all K=5 + Gemma kv-auto work was validated on
    # it. Session validation 2026-06-19 — 35B-A3B FP8 TQ k8v4 + MTP K=5:
    # 239.7 TPS single-stream / TPOT 3.94 ms (+15.8% vs K=3 207); 27B Lorbus
    # INT4 hybrid GDN+Mamba + MTP K=5: 127.4 TPS / TPOT 7.54 ms (+8.2% vs
    # 117.7); PN119 tensor-core FP8-key decode live (tl.dot=8), P18b revived;
    # PN394 (#46047 qwen3 partial-param `<` truncation) + PN399 (#46067 TQ
    # decode-scratch fixed buffer, IMA-safe FULL cudagraph) promoted to PROD;
    # PN353A (vllm#44053 TQ builder workspace reserve) enabled in Qwen YAMLs;
    # Gemma-4-31B kv-auto chat profile added (70.1 TPS, +69.6% vs TQ chat).
    # dev101 (nightly-4c626633) retained above as the previous/rollback pin
    # per CLAUDE.md ≤2-pin policy.
    "0.23.1rc1.dev148+gb4c80ec0f",                       # setuptools_scm-derived
    "nightly-b4c80ec0f",                                 # docker tag form (short)
    # ── PROD pin PROMOTION dev301 2026-06-24 ──────────────────────────
    # Image: vllm/vllm-openai:nightly-04c2a8dea (0.23.1rc1.dev301+g04c2a8dea,
    # commit 04c2a8dea). PROMOTED 2026-06-24: bump from dev148.
    # Validated: 35B 208 TPS + 31B 94.7 TPS boot+chat+tool-call. The dev301
    # anchor-SOT regen surfaced exactly 5 anchor_drift entries
    # (P85/PN394/PN353A/PN400/PN382). Handled: PN394 + PN400 retired on
    # dev301 (their #46047 / #45656 are IN dev301); PN353A + PN382
    # re-anchored and kept active (PN353A coupled to the PN399 chain; PN382
    # a divergent fork of #45080); P85 OFF. dev148 retained above as
    # previous/rollback per CLAUDE.md ≤2-pin policy.
    "0.23.1rc1.dev301+g04c2a8dea",                       # setuptools_scm-derived
    "nightly-04c2a8dea",                                 # docker tag form (short)
)


# Pins that are in allowlist but not yet bench-validated. Audit gate
# tolerates them but logs a CAVEAT until promoted.
PROMOTION_PENDING_VLLM_PINS: tuple[str, ...] = (
    # NOTE: 626fa9bb variants graduated 2026-05-30 (K.1.R.R.8.5 prod
    # validation). The pre-K.1.R v0.21.0/v0.21.1rc0/d735968 entries below
    # remain pending — they were intermediate stations never bench-cycled.
    "0.21.0",
    "0.21.1rc0",
    "0.21.1rc0+gd735968f6d63",
    "0.21.1rc0+gbf610c2f5676",
    # 0.23.1 (4c626633) GRADUATED to KNOWN_GOOD on 2026-06-17 after full-fleet
    # validation (35B / 27B / Gemma4-31B / DiffusionGemma all apply failed=0 +
    # smoke + tool-call PASS, 35B bench 210.7 TPS = 101% of dev491). See the
    # KNOWN_GOOD_VLLM_PINS "PROD pin PROMOTION 0.23.1" block above.
)


# Validated Genesis patcher commits — short-SHA prefixes that have all
# critical fixes (P67 v7.63.x non-pow-2 GQA split-M, P98 TQ workspace
# revert, etc.) AND have been bench-validated on at least one PROD config.
# Audit rule R-010 uses this to gate 27B INT4 + TQ k8v4 + spec_decode
# configurations. Add new entries via PR review only, paired with a
# CHANGELOG entry naming the bench config + date.
KNOWN_GOOD_GENESIS_PINS: tuple[str, ...] = (
    "991dc1a",   # last patcher-only commit before model_configs work
                 # (PN59 GdnScratchPool wire, validated 35B PROD 192.6 TPS)
    "8264512",   # ModelConfig framework — pure tooling, no patch changes
    "8b4b033",   # 5-layer validation pipeline — pure tooling
    "9c534d4",   # launch script archive cleanup — pure tooling
    "f9576df",   # second archive cleanup pass — pure tooling
                 # (validated 27B INT4 + TQ k8v4: 90.2 TPS, 10/10 tool, 2026-05-05)
)


def is_genesis_pin_validated(pin: Optional[str]) -> bool:
    """Return True if `pin` (short or long SHA) is in KNOWN_GOOD_GENESIS_PINS.

    Matches by prefix in either direction so both 7-char and 40-char SHAs work.
    """
    if not pin:
        return False
    pin = pin.strip().lower()
    return any(
        pin.startswith(good) or good.startswith(pin)
        for good in KNOWN_GOOD_GENESIS_PINS
    )


def assert_vllm_pin_allowed(
    allowlist: tuple[str, ...] = KNOWN_GOOD_VLLM_PINS + PROMOTION_PENDING_VLLM_PINS,
    policy: Optional[str] = None,
) -> tuple[str, str]:
    """Loud check that the running vllm pin is on the allowlist.

    Returns ("ok"|"unknown"|"missing", message). On policy="strict" and
    a non-ok status, raises SystemExit(2) — caller does NOT need to handle.

    Policy resolution order: explicit `policy` arg > env `GENESIS_VLLM_PIN_POLICY`
    > "warn" (default). Set policy="strict" in production start scripts to
    fail fast on accidental pin drift.

    Per Sander 2026-05-04 ("foolproof guard"): never silently apply
    patches against an unvalidated pin. The allowlist must be updated
    explicitly when a new pin is qualified.
    """
    import os as _os

    if policy is None:
        policy = _os.environ.get("GENESIS_VLLM_PIN_POLICY", "warn").strip().lower()
    if policy not in ("warn", "strict"):
        policy = "warn"

    pin = get_vllm_full_version_string()
    if pin is None:
        msg = "vllm not importable — cannot verify pin"
        if policy == "strict":
            print(f"[Genesis pin-gate] STRICT FAIL: {msg}", flush=True)
            import sys as _sys
            _sys.exit(2)
        return "missing", msg

    if pin in allowlist:
        return "ok", f"vllm pin {pin} is on the Genesis allowlist"

    msg = (
        f"vllm pin {pin!r} is NOT on the Genesis known-good list "
        f"({len(allowlist)} entries). Allowed pins: {list(allowlist)}. "
        f"To accept this pin, add it to KNOWN_GOOD_VLLM_PINS (or "
        f"PROMOTION_PENDING_VLLM_PINS for a not-yet-validated candidate) in "
        f"sndr/engines/vllm/detection/guards.py and document the validation in CHANGELOG."
    )
    if policy == "strict":
        print(f"[Genesis pin-gate] STRICT FAIL: {msg}", flush=True)
        import sys as _sys
        _sys.exit(2)
    return "unknown", msg


@functools.cache
def get_flash_attn_major_version() -> Optional[int]:
    """Try to detect FlashAttention version (FA2 / FA3 / FA4).

    Returns major version int or None if not available or detection failed.

    Canonical source: vllm/v1/attention/backends/fa_utils.py:get_flash_attn_version
    """
    try:
        # Try vllm's own helper first
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        v = get_flash_attn_version(head_size=128)
        return int(v) if v else None
    except Exception:
        pass
    try:
        # Fallback: check flash_attn module directly
        import flash_attn
        return int(flash_attn.__version__.split(".")[0])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#                      MODEL ARCHITECTURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def is_model_arch(model_config: Any, arch_name: str) -> bool:
    """Case-insensitive substring match against model_config.architectures.

    Canonical source: vllm/model_executor/models/registry.py resolution logic.

    Examples:
      is_model_arch(cfg, "Qwen3")       # True for Qwen3.5 / Qwen3.6 / Qwen3-Next
      is_model_arch(cfg, "DeepSeekV3")  # True for DeepSeek V3 family
      is_model_arch(cfg, "Llama")       # True for any Llama variant
    """
    if model_config is None:
        return False
    try:
        archs = getattr(model_config, "architectures", None) or []
        needle = arch_name.lower()
        return any(needle in (a or "").lower() for a in archs)
    except Exception:
        return False


def is_qwen3_family(model_config: Any) -> bool:
    """True for Qwen3.5 / Qwen3.6 / Qwen3-Next / Qwen3-Coder family."""
    return is_model_arch(model_config, "Qwen3")


def is_deepseek_v3(model_config: Any) -> bool:
    """True for DeepSeek V3 family (uses MLA attention, distinct from Qwen)."""
    return is_model_arch(model_config, "DeepseekV3") or is_model_arch(model_config, "DeepSeek-V3")


def is_llama_family(model_config: Any) -> bool:
    """True for Llama 3.x / 4.x family."""
    return is_model_arch(model_config, "Llama")


def is_gemma_family(model_config: Any) -> bool:
    """True for Gemma family (uses sliding-window hybrid)."""
    return is_model_arch(model_config, "Gemma")


def is_mixtral_family(model_config: Any) -> bool:
    """True for Mixtral MoE family (different router than Qwen3)."""
    return is_model_arch(model_config, "Mixtral")


# ═══════════════════════════════════════════════════════════════════════════
#                     BACKEND / KERNEL DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def has_turboquant_support(cache_dtype: Optional[str]) -> bool:
    """True if TurboQuant path is active (cache_dtype starts with 'turboquant_').

    Canonical source: vllm/platforms/cuda.py:134 — routing key in CacheConfig.
    """
    return bool(cache_dtype and cache_dtype.startswith("turboquant_"))


def is_marlin_selected(fused_moe_layer: Any) -> bool:
    """Best-effort introspection: is Marlin kernel selected for this MoE layer?

    Returns False (safe default) if detection fails — we prefer to skip
    a Marlin-specific patch than apply it wrong.
    """
    try:
        kernel = getattr(fused_moe_layer, "kernel", None)
        if kernel is None:
            # Try alternate attribute paths
            kernel = getattr(fused_moe_layer, "quant_method", None)
        name = type(kernel).__name__ if kernel else ""
        return "marlin" in name.lower()
    except Exception:
        return False


def is_flash_attn_backend(attn_backend: Any) -> bool:
    """True if FlashAttention backend selected."""
    try:
        name = getattr(attn_backend, "name", "") or type(attn_backend).__name__
        return "flash" in name.lower() and "attn" in name.lower()
    except Exception:
        return False


def is_turboquant_backend(attn_backend: Any) -> bool:
    """True if TurboQuant backend selected."""
    try:
        name = getattr(attn_backend, "name", "") or type(attn_backend).__name__
        return "turboquant" in name.lower()
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#                   PATCH CACHE / SITE MAP CONTROL
# ═══════════════════════════════════════════════════════════════════════════

# Canonical truthy values for Genesis env flags. Matches dispatcher.should_apply
# convention so operators can use the same forms they're used to:
#   GENESIS_NO_PATCH_CACHE=1
#   GENESIS_NO_PATCH_CACHE=true
#   GENESIS_NO_PATCH_CACHE=yes
#   GENESIS_NO_PATCH_CACHE=on
_TRUTHY_VALS = ("1", "true", "yes", "on")


def genesis_no_patch_cache() -> bool:
    """Operator escape hatch — disable ALL Genesis patch-cache layers.

    When set to a truthy value, every cache reader in Genesis MUST fall
    back to the legacy O(N×M) full-scan path. This is the single switch
    operators flip when patch-cache fails / suspected stale / debugging
    new patcher.

    Cache layers that MUST honor this flag (in order of introduction):
      - P1.3 (this fn): self-validating helper, returns False by default
      - P2.1 Anchor offset manifest "Site Map" (planned 2026-05-08+):
        TextPatcherV2 manifest-aware path skipped → legacy full scan
      - P2.2 Persistent file md5 cache (planned 2026-05-08+):
        fast_path_check() returns False → Layer 4 full anchor scan
      - Future: Triton autotuner warm-start, vllm compile cache patch
        fingerprint, CUDA graph capture cache (all P3+ tracks)

    Returns:
        True if operator wants legacy non-cached path; False otherwise.

    Why a helper instead of inline `os.environ.get(...)`:
      - Single canonical truthy parser
      - Single audit log line on observed disable (debugging)
      - Forward-compatible: future opt-out flags can compose here
    """
    val = os.environ.get("GENESIS_NO_PATCH_CACHE", "").strip().lower()
    return val in _TRUTHY_VALS


# ═══════════════════════════════════════════════════════════════════════════
#                      FILE PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════
#
# Stage 3 decision (2026-05-07, revised): vllm_install_root() and
# resolve_vllm_file() stay HERE in guards.py. The earlier plan was to
# move impl into vllm/sndr_core/paths/ but tests monkey-patch the
# attribute on this module (e.g. `guards.vllm_install_root = lambda: td`)
# and `resolve_vllm_file()` calls vllm_install_root() internally — the
# monkey-patch contract requires both functions to share THIS module's
# attribute lookup so test mocks affect both.
#
# sndr.engines.vllm.locations.vllm_install AND sndr.engines.vllm.locations.resolver
# remain as canonical-name re-exports OF the impls below. Import
# `from sndr.engines.vllm.locations import resolve_vllm_file` continues to
# work transparently. Test mocks against `guards.vllm_install_root`
# also propagate (via the re-export forward).

@functools.cache
def vllm_install_root() -> Optional[str]:
    """Returns absolute path to installed vllm package, or None.

    CRITICAL: Replaces hardcoded /usr/local/lib/python3.12/dist-packages/vllm/
    that was in earlier patch versions — those break on:
      - macOS dev environments
      - venv installations
      - Python 3.13+ coming 2027
      - Docker slim/distroless images

    vllm.__file__ is the canonical universal way to locate the package.
    """
    try:
        import vllm
        return os.path.dirname(vllm.__file__)
    except Exception:
        return None


def resolve_vllm_file(relative_path: str) -> Optional[str]:
    """Returns absolute path to file within installed vllm, or None if missing.

    Looks up vllm_install_root() through THIS module's attribute (not a
    bound import) so test mocks via `guards.vllm_install_root = ...`
    propagate through into resolve_vllm_file() calls.

    Example:
        resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
        -> "/path/to/vllm/v1/attention/backends/turboquant_attn.py" or None
    """
    # Use module-level dispatch (not the imported binding) so monkey-
    # patches on `guards.vllm_install_root` are honored by callers of
    # `resolve_vllm_file` too.
    import sys as _sys
    _self = _sys.modules[__name__]
    root = _self.vllm_install_root()
    if root is None:
        return None
    full = os.path.join(root, relative_path)
    return full if os.path.exists(full) else None


# ═══════════════════════════════════════════════════════════════════════════
#                         SUMMARY / DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════

def platform_summary() -> dict[str, Any]:
    """Return full platform diagnostic dict for logging/debugging.

    Useful during patch application to log context:
        log.info("[Genesis] Platform: %s", json.dumps(platform_summary()))
    """
    return {
        "vendor": {
            "is_nvidia_cuda": is_nvidia_cuda(),
            "is_amd_rocm": is_amd_rocm(),
            "is_intel_xpu": is_intel_xpu(),
            "is_cpu_only": is_cpu_only(),
        },
        "nvidia": {
            "compute_capability": get_compute_capability(),
            "is_ampere_datacenter": is_ampere_datacenter(),
            "is_ampere_consumer": is_ampere_consumer(),
            "is_ada_lovelace": is_ada_lovelace(),
            "is_hopper": is_hopper(),
            "is_blackwell": is_blackwell(),
            "has_native_fp8": has_native_fp8(),
        },
        "amd": {
            "gcn_arch": _gcn_arch(),
            "is_cdna2": is_rocm_cdna2(),
            "is_cdna3": is_rocm_cdna3(),
            "is_rdna": is_rocm_rdna(),
        } if is_amd_rocm() else {},
        "versions": {
            "torch": get_torch_version(),
            "transformers": get_transformers_version(),
            "vllm": get_vllm_version_tuple(),
            "flash_attn_major": get_flash_attn_major_version(),
        },
        "paths": {
            "vllm_install_root": vllm_install_root(),
        },
    }
