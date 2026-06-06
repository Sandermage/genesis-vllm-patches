# SPDX-License-Identifier: Apache-2.0
"""PN96 — Persistent Marlin MoE workspace (Path C of Wave 9 35B optimization).

Background — Wave 9 dev209 35B re-bench (2026-05-12) found a -2.82% TPS /
+2.86% TPOT regression vs Sprint 1 dev93 baseline. Configs were
bit-identical; only the vllm pin changed. Root-cause analysis traced the
slowdown to upstream's MoE refactor between dev93..dev209: the Marlin MoE
code itself (`fused_marlin_moe`) is essentially unchanged, but the
modular dispatcher (`mk.FusedMoEExpertsModular`) and surrounding wrappers
add cumulative per-call overhead on the A3B-FP8 path. The 27B path
(hybrid GDN+Mamba INT4) was neutral because it doesn't touch Marlin MoE.

Specific optimization target
-----------------------------
The new dev209 `experts/marlin_moe.py::MarlinExperts.apply()` calls
`fused_marlin_moe(...)` WITHOUT passing the `workspace` parameter. The
fallback path in `_fused_marlin_moe` then allocates fresh per call:

    if workspace is None:
        workspace = marlin_make_workspace_new(hidden_states.device, 4)

`marlin_make_workspace_new` creates a small int32 tensor (Marlin GEMM
scratch). The allocation itself is cheap (~1-2μs per call), but at
~hundreds of MoE calls per generation step on a 35B-A3B model, the
cumulative wall-time matters. The workspace is otherwise stateless and
can be safely reused across all calls on the same device.

Mechanism
---------
This is a RUNTIME monkey-patch (not a text-patch). At apply time we
wrap `MarlinExperts.apply` such that:

  1. The wrapper lazily creates a single workspace tensor per
     (device, multiplier=4) on the wrapped instance (`self._genesis_pn96_ws`).
  2. It monkey-patches `vllm.model_executor.layers.fused_moe.experts.
     marlin_moe.fused_marlin_moe` once to honor a thread-local default
     workspace when the caller passes `workspace=None`.
  3. Subsequent `MarlinExperts.apply` invocations bind the cached
     workspace in the thread-local before delegating to the original
     `apply`.

Composition
-----------
- Safe with P17/P18 (env override of num_warps/num_stages) — different
  optimization vector (kernel config vs workspace lifetime).
- Safe with P22/P38 (TurboQuant workspace) — TQ workspace and Marlin
  workspace are distinct buffers; no aliasing.
- Auto-skips when:
  - `GENESIS_ENABLE_PN96=0` (default ON, but explicit-off available)
  - `current_platform` is not CUDA
  - `experts/marlin_moe.py` module not present (e.g. older vllm pin)
  - Loaded model is a Gemma 4 variant (binary-search 2026-05-15 identified
    PN96B as the patch corrupting Gemma 4 ``max_model_len`` during config
    merge; Gemma 4 31B AWQ-4bit is Dense, not MoE, so this hook would
    never fire usefully on Gemma 4 anyway). Override with
    ``GENESIS_PN96B_FORCE_GEMMA4=1`` for diagnostics only.

Auto-retire
-----------
- When upstream `MarlinExperts.apply` starts passing `workspace=` itself
  (drift marker: `workspace=self._genesis_workspace` or
  `workspace=self.workspace` inside the call); detected via signature
  inspection of the original method.

Env flag
--------
`GENESIS_ENABLE_PN96` (default ON for 35B PROD, no-op when Marlin MoE
absent). Disable via `GENESIS_DISABLE_PN96=1`.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: experimental (lifecycle=experimental, default_on=True for 35B).
"""
from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("genesis.wiring.pn96b_marlin_persistent_workspace")  # renamed from pn96_ 2026-05-14

GENESIS_PN96_MARKER = (
    "Genesis PN96 Marlin persistent workspace v1 (Wave 9 dev209 perf-restore)"
)

# Renamed from PN96 → PN96b 2026-05-14 (collision with kv_cache/PN96
# emergency-demote). Accept legacy GENESIS_ENABLE_PN96 env var for one
# release cycle so operators don't break their existing launch scripts.
_ENV_ENABLE = "GENESIS_ENABLE_PN96B"
_ENV_DISABLE = "GENESIS_DISABLE_PN96B"
_ENV_ENABLE_LEGACY = "GENESIS_ENABLE_PN96"
_ENV_DISABLE_LEGACY = "GENESIS_DISABLE_PN96"

# Diagnostics escape hatch: force PN96B to install on Gemma 4 anyway.
# Default behavior is to skip (Gemma 4 doesn't use Marlin MoE and binary
# search 2026-05-15 traced PN96B as the patch that resets Gemma 4's
# max_model_len during config merge — root cause not yet isolated).
_ENV_FORCE_GEMMA4 = "GENESIS_PN96B_FORCE_GEMMA4"

# Thread-local for default workspace; set by wrapped MarlinExperts.apply,
# read by the patched fused_marlin_moe wrapper.
_TLS = threading.local()


def _env_enabled() -> bool:
    """Default-ON unless explicitly disabled. Mirrors PN35 / PN33 pattern.

    Accepts both new (PN96B) and legacy (PN96) env var names during the
    rename grace period (2026-05-14 — one release cycle).
    """
    for disable_var in (_ENV_DISABLE, _ENV_DISABLE_LEGACY):
        if os.environ.get(disable_var, "").strip().lower() in (
            "1", "true", "yes", "on"
        ):
            return False
    for enable_var in (_ENV_ENABLE, _ENV_ENABLE_LEGACY):
        val = os.environ.get(enable_var, "").strip().lower()
        if val != "":
            return val in ("1", "true", "yes", "on")
    return True  # default ON


# Singletons set by apply() so other helpers can read them.
_ORIGINAL_MARLIN_APPLY = None
_ORIGINAL_FUSED_MARLIN_MOE = None
_APPLY_INSTALLED = False


def _wrapped_fused_marlin_moe(*args, **kwargs):
    """Wrapper around upstream fused_marlin_moe that injects a default
    workspace from the thread-local when the caller passes
    workspace=None (or omits the kwarg)."""
    if kwargs.get("workspace") is None:
        default_ws = getattr(_TLS, "default_workspace", None)
        if default_ws is not None:
            kwargs["workspace"] = default_ws
    return _ORIGINAL_FUSED_MARLIN_MOE(*args, **kwargs)


def _wrapped_marlin_experts_apply(self, *args, **kwargs):
    """Wrapper around MarlinExperts.apply that ensures a persistent
    workspace tensor lives on the instance, then exposes it via
    thread-local to the patched fused_marlin_moe."""
    # Lazy-init workspace on first call (device available from hidden_states)
    if getattr(self, "_genesis_pn96_ws", None) is None:
        hs = kwargs.get("hidden_states")
        if hs is None and len(args) >= 2:
            hs = args[1]  # apply(self, output, hidden_states, ...) — args[1]
        if hs is not None:
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                marlin_make_workspace_new,
            )
            try:
                self._genesis_pn96_ws = marlin_make_workspace_new(hs.device, 4)
            except Exception as e:
                log.debug("[PN96] workspace alloc failed (%s); fallback to per-call", e)
                self._genesis_pn96_ws = False  # sentinel: don't retry

    # Install on TLS for the duration of the original apply call.
    # NOTE: `_genesis_pn96_ws` is a torch.Tensor when initialized; the
    # `tensor not in (None, False)` form invokes Tensor.__eq__ during
    # the `in` membership test which broadcasts a comparison and then
    # `bool(tensor)` raises "Boolean value of Tensor with more than one
    # value is ambiguous". Use identity tests instead.
    prev = getattr(_TLS, "default_workspace", None)
    ws = self._genesis_pn96_ws
    if ws is not None and ws is not False:
        _TLS.default_workspace = ws
    try:
        return _ORIGINAL_MARLIN_APPLY(self, *args, **kwargs)
    finally:
        _TLS.default_workspace = prev


def apply() -> tuple[str, str]:
    """Apply PN96 — install persistent Marlin workspace hooks. Never raises."""
    global _ORIGINAL_MARLIN_APPLY, _ORIGINAL_FUSED_MARLIN_MOE, _APPLY_INSTALLED

    if not _env_enabled():
        return "skipped", (
            "explicit disable: GENESIS_DISABLE_PN96=1 set "
            "(PN96 = persistent Marlin MoE workspace, Wave 9 perf-restore)"
        )

    if _APPLY_INSTALLED:
        return "applied", "PN96 already installed (idempotent)"

    # Gemma 4 arch gate — binary search 2026-05-15 traced PN96B as the
    # patch corrupting Gemma 4's max_model_len at config-merge time.
    # Gemma 4 31B is Dense AWQ-4bit (no MoE), so PN96B has no useful
    # effect there anyway. Skip cleanly unless operator forces install.
    force_gemma4 = os.environ.get(_ENV_FORCE_GEMMA4, "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if not force_gemma4:
        try:
            from vllm.config import get_current_vllm_config
            _cfg = get_current_vllm_config()
        except Exception:
            _cfg = None
        if _cfg is not None:
            try:
                from ..model_compat.gemma4._gemma4_detect import is_gemma4_arch
                if is_gemma4_arch(_cfg):
                    return "skipped", (
                        "Gemma 4 architecture detected — PN96B auto-skips "
                        "(Gemma 4 31B AWQ-4bit is Dense, not MoE; PN96B "
                        "would corrupt max_model_len during config merge). "
                        "Set GENESIS_PN96B_FORCE_GEMMA4=1 to override."
                    )
            except Exception as e:
                log.debug("[PN96B] gemma4 arch probe failed: %s", e)

    # Platform gate — CUDA only, Ampere+ (Marlin MoE requires sm75+)
    try:
        from vllm.platforms import current_platform
        if not current_platform.is_cuda():
            return "skipped", "PN96 targets CUDA — platform skip"
        if not current_platform.has_device_capability((7, 5)):
            return "skipped", "PN96 requires SM 7.5+ (Marlin minimum)"
    except Exception as e:
        return "skipped", f"current_platform unavailable: {e}"

    # Locate the upstream module — gracefully skip if dev93-era layout
    try:
        from vllm.model_executor.layers.fused_moe.experts import (
            marlin_moe as _marlin_moe_mod,
        )
    except ImportError:
        return "skipped", (
            "experts/marlin_moe.py not found — likely pre-refactor "
            "(dev93 era). PN96 only applies to dev209+ modular MoE layout."
        )

    MarlinExperts = getattr(_marlin_moe_mod, "MarlinExperts", None)
    if MarlinExperts is None or not hasattr(MarlinExperts, "apply"):
        return "skipped", "MarlinExperts.apply not found in expected location"

    fused_marlin_moe = getattr(_marlin_moe_mod, "fused_marlin_moe", None)
    if fused_marlin_moe is None:
        return "skipped", "fused_marlin_moe symbol not found"

    # Idempotency: detect prior install via marker attribute on the method
    if getattr(MarlinExperts.apply, "_genesis_pn96_installed", False):
        _APPLY_INSTALLED = True
        return "applied", "MarlinExperts.apply already wrapped (idempotent)"

    # Drift detection: if upstream already passes workspace=, PN96 is no-op
    # (signaled by inspecting `apply`'s source for `workspace=`); we keep
    # the hook in place but it just won't be effective. Logged for audit.
    try:
        import inspect
        src = inspect.getsource(MarlinExperts.apply)
        if "workspace=" in src and "workspace=None" not in src:
            return "skipped", (
                "DRIFT: MarlinExperts.apply already passes workspace= — "
                "upstream likely fixed this. PN96 self-retires; check that "
                "the workspace is actually persistent and remove this patch."
            )
    except Exception:
        pass  # source inspection optional; proceed with install

    # Save originals + install wrappers
    _ORIGINAL_MARLIN_APPLY = MarlinExperts.apply
    _ORIGINAL_FUSED_MARLIN_MOE = fused_marlin_moe

    _wrapped_marlin_experts_apply._genesis_pn96_installed = True
    MarlinExperts.apply = _wrapped_marlin_experts_apply
    _marlin_moe_mod.fused_marlin_moe = _wrapped_fused_marlin_moe
    _APPLY_INSTALLED = True

    return "applied", (
        "PN96 installed: MarlinExperts.apply now caches a persistent "
        "Marlin workspace per instance; fused_marlin_moe honors the "
        "thread-local default when caller passes workspace=None. "
        "Eliminates per-call marlin_make_workspace_new allocation on the "
        "A3B-FP8 MoE hot path. Wave 9 dev209 perf-restore target."
    )


def is_applied() -> bool:
    """True iff the MarlinExperts.apply wrap is live in this process."""
    try:
        from vllm.model_executor.layers.fused_moe.experts import (
            marlin_moe as _marlin_moe_mod,
        )
    except ImportError:
        return False
    return getattr(
        getattr(_marlin_moe_mod, "MarlinExperts", None).apply
        if getattr(_marlin_moe_mod, "MarlinExperts", None) is not None
        else None,
        "_genesis_pn96_installed",
        False,
    )


def revert() -> bool:
    """Restore upstream MarlinExperts.apply + fused_marlin_moe. Returns True
    on successful revert, False if PN96 was never installed."""
    global _ORIGINAL_MARLIN_APPLY, _ORIGINAL_FUSED_MARLIN_MOE, _APPLY_INSTALLED
    if not _APPLY_INSTALLED:
        return False
    try:
        from vllm.model_executor.layers.fused_moe.experts import (
            marlin_moe as _marlin_moe_mod,
        )
        MarlinExperts = getattr(_marlin_moe_mod, "MarlinExperts", None)
        if MarlinExperts is not None and _ORIGINAL_MARLIN_APPLY is not None:
            MarlinExperts.apply = _ORIGINAL_MARLIN_APPLY
        if _ORIGINAL_FUSED_MARLIN_MOE is not None:
            _marlin_moe_mod.fused_marlin_moe = _ORIGINAL_FUSED_MARLIN_MOE
        _APPLY_INSTALLED = False
        return True
    except Exception as e:
        log.warning("[PN96] revert failed: %s", e)
        return False
