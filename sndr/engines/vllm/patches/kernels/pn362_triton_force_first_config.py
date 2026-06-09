# SPDX-License-Identifier: Apache-2.0
"""PN362 — vendor of OPEN PR vllm#42425 (VLLM_TRITON_FORCE_FIRST_CONFIG).

Deep-dive understanding of the upstream PR
==========================================

**Problem.** ``@triton.autotune`` benchmarks every candidate config on
the first launch per ``key``, then caches the winner. Three things make
this painful when *measuring* or *debugging*:

  1. **Autotune is timing-driven, so its winner is non-deterministic.**
     Run-to-run jitter can promote a different ``(BLOCK_M, num_warps,
     ...)`` tuple each container restart. Different tuple → different
     reduction order → numerically different outputs across runs. This
     is the *direct* root cause of the "199 vs 228 wall_TPS regression"
     scare in our 2026-06-09 investigation — Triton autotune picked a
     different config between two server starts, no Genesis code
     change involved.

  2. **Cached results bleed across investigations.** Triton's autotune
     cache is persisted on disk; "just rerun it" doesn't reset state.
     Editing every kernel's ``configs=[...]`` to pin a value or wiping
     caches between runs is fragile — and third-party libs (``fla``,
     ``flashinfer``) ship their own autotuned kernels we don't own.

  3. **No visibility into what was picked.** If autotuning picked a
     different config on a particular GPU/run, you only find out by
     digging into the cache files.

**Upstream mechanism (vllm#42425, Francesco Fusco, IBM)**: a debug-only
env var ``VLLM_TRITON_FORCE_FIRST_CONFIG`` that replaces
``triton.runtime.autotuner.Autotuner.run`` so it walks the candidate
configs in declaration order and uses the first one that does *not*
raise ``OutOfResources`` / ``CompileTimeAssertionFailure`` /
``PTXASError``. The picked index is cached per ``(autotuner, key)`` so
subsequent calls stay deterministic and cheap. One INFO log line per
unique kernel is emitted::

    [triton-autotune-disabled] kernel=chunk_scaled_dot_kkt_fwd_kernel
        configs=27 picked_index=0 picked=BK: 32, num_warps: 2, ...

**Default off** — no behaviour change unless explicitly enabled. The
author cites GDN prefill + MTP non-determinism (PR #40172) as the
motivating use case — *exactly* the path we exercise on Qwen3.6-35B-A3B
hybrid + MTP K=3.

Composition with Genesis PN345 (shmem-aware autotune pruner)
============================================================

PN345 vendors vllm#43047, a *pre-autotune* config filter that drops
configs whose estimated shared-memory footprint exceeds the device
opt-in budget (A5000 ~99 KiB). It runs **before** the Autotuner kicks
off — it is the ``prune_configs_by={"early_config_prune": ...}`` hook.

PN362 monkey-patches **Autotuner.run** itself — the *runtime* method,
not the decorator-time filter. It picks the first surviving config
from the (already PN345-pruned) list.

Therefore PN345 and PN362 **compose cleanly**:

  1. ``@triton.autotune`` decorator invocation builds the full config
     list at module-import time.
  2. **PN345's ``early_config_prune``** drops configs whose
     shmem estimate exceeds the A5000 budget (no OOR at JIT time).
  3. **Triton instantiates the Autotuner** with the surviving configs.
  4. When the kernel is first launched, **PN362's patched
     ``Autotuner.run``** walks the surviving configs in order and
     picks the first one that compiles + launches without errors.

Effect on PN345 invariants:
  * PN345 *removes* dangerous configs → the first surviving config
    is by construction shmem-safe (so PN362 picks it without an OOR
    walk). Win-win: PN345 makes the pick *correct*, PN362 makes the
    pick *deterministic*.
  * Without PN345 (default off), PN362 walks past OOR configs at
    runtime via the ``_invalid_config_errors`` tuple — still correct,
    but slower first call.
  * No anchor conflict: PN345 text-patches FLA kernel source files
    (``chunk_delta_h.py`` + ``chunk_o.py``). PN362 text-patches
    ``env_override.py`` (no overlap).

Genesis vendoring strategy
==========================

Upstream PR adds **4 files** (1 new module, 1 envs.py entry, 1
env_override.py block, 1 test). Per Genesis policy we cannot add new
files to the live vllm install. So we **inline** the entire 107-LOC
``force_first_config.install()`` helper into ``env_override.py`` as a
single block appended after the existing ``_patch_inductor_fallback_
allow_list()`` call (the canonical last line of upstream env_override).

Trade-offs:
  * Pro: pure text-patch, no Genesis file injection.
  * Pro: idempotent marker per file.
  * Pro: zero risk if PR rebases — block is self-contained, anchor
    is the file-end sentinel.
  * Pro: works regardless of vllm.envs registration — we re-read
    ``os.environ`` directly (same gate-text as upstream PR).
  * Con: helper code lives in a non-canonical location until PR
    merges. On merge we detect via the upstream sentinel
    ``vllm/triton_utils/force_first_config.py`` existing and skip.

Single sub-patch (``required=True``) — the env_override.py file is
stable across all our pins (audited 2026-06-09: 33,227 bytes ending
with ``_patch_inductor_fallback_allow_list()``).

Expected use cases
==================

* **BENCH A/B**: enable on both baseline and patched runs so any
  delta is real code/config difference, not autotune jitter.
  Required for the 5-run cross-bench CV<2 % stability target.

* **Determinism debugging**: enable to make GDN prefill / MTP accept
  rate exactly reproducible across container restarts.

* **PROD**: leave off — picking the autotuned winner is faster on the
  steady-state path. Force-first picks the first *valid* config,
  which is often (deliberately, by FLA's declaration order) a
  smaller / safer tile that runs slower than the autotune winner.

Author: Sandermage (Sander) Barzov Aleksandr, Odessa, Ukraine.
Vendor target: vllm-project/vllm#42425 (OPEN as of 2026-06-09).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn362_triton_force_first_config")

GENESIS_PN362_MARKER = (
    "Genesis PN362 vendor of vllm#42425 (VLLM_TRITON_FORCE_FIRST_CONFIG) v1"
)

# Inlined upstream helper. Kept BYTE-FOR-BYTE compatible with the PR's
# ``vllm/triton_utils/force_first_config.py`` (107 LOC) so future audits
# can diff against upstream cleanly. Only differences from upstream:
#   * uses logging.getLogger() instead of vllm.logger.init_logger to
#     avoid pulling vllm.logger this late in env_override.py boot;
#   * gated by direct os.environ check, identical to upstream PR's
#     env_override.py block (no dependency on vllm.envs being loaded).
PN362_HELPER_BLOCK = '''
# ─── [Genesis PN362 vendor of vllm#42425] Triton autotune determinism ─
# Skip Triton autotuning under VLLM_TRITON_FORCE_FIRST_CONFIG.
#
# When the env var is set, triton.runtime.autotuner.Autotuner.run is
# replaced so that, instead of benchmarking every candidate config, it
# walks them in declaration order and uses the first one that does not
# raise OutOfResources / CompileTimeAssertionFailure / PTXASError. The
# picked index is cached per (autotuner, key) so subsequent calls stay
# deterministic. Used to eliminate autotuning variability when
# measuring kernel performance.
#
# Default off. Opt-in via VLLM_TRITON_FORCE_FIRST_CONFIG=1.
# Composes with Genesis PN345 (shmem-aware pre-autotune pruner): PN345
# drops configs that would OOR at JIT time; PN362 picks the first
# surviving one at runtime.
import logging as _g_pn362_logging
import os as _g_pn362_os

_g_pn362_log = _g_pn362_logging.getLogger("genesis.pn362.force_first_config")
_g_pn362_installed = False


def _g_pn362_install():
    global _g_pn362_installed
    if _g_pn362_installed:
        return
    try:
        import importlib as _g_pn362_importlib
        autotuner_mod = _g_pn362_importlib.import_module(
            "triton.runtime.autotuner")
        Autotuner = autotuner_mod.Autotuner
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources, PTXASError
    except Exception as _e:  # noqa: BLE001
        _g_pn362_log.warning(
            "[PN362] Triton not importable, skipping install: %r", _e)
        return

    _invalid = (OutOfResources, CompileTimeAssertionFailure, PTXASError)
    _picked_cache: dict[tuple, int] = {}
    _seen_kernels: set[str] = set()

    def _run_first_valid_config(self, *args, **kwargs):
        if not self.configs:
            return self.fn(*args, **kwargs)

        key_vals = tuple(kwargs[name] for name in self.keys if name in kwargs)
        cache_key = (id(self), key_vals)
        kernel_name = getattr(self.base_fn, "__name__", repr(self.fn))

        cached_idx = _picked_cache.get(cache_key)
        candidate_indices = (
            [cached_idx] if cached_idx is not None
            else list(range(len(self.configs)))
        )

        last_exc: Exception | None = None
        for idx in candidate_indices:
            config = self.configs[idx]
            if config.pre_hook is not None:
                full_nargs = {
                    **dict(zip(self.arg_names, args)),
                    **kwargs,
                    **config.all_kwargs(),
                }
                config.pre_hook(full_nargs)
            # Prefer self.fn.run(...) — kernel-launch entrypoint for both
            # JITFunction and Heuristics. Calling JITFunction(...) raises
            # "Cannot call @triton.jit'd outside of the scope of a
            # kernel". Fall back to plain call only if .run is missing.
            launch = getattr(self.fn, "run", self.fn)
            try:
                result = launch(*args, **kwargs, **config.all_kwargs())
            except _invalid as e:
                last_exc = e
                continue

            if cached_idx is None:
                _picked_cache[cache_key] = idx
                self.best_config = config
                if kernel_name not in _seen_kernels:
                    _seen_kernels.add(kernel_name)
                    _g_pn362_log.info(
                        "[triton-autotune-disabled] kernel=%s configs=%d "
                        "picked_index=%d picked=%s",
                        kernel_name, len(self.configs), idx, config)
            return result

        raise RuntimeError(
            f"[PN362] No valid config for kernel {kernel_name} "
            f"key={key_vals} (tried {len(self.configs)} configs)"
        ) from last_exc

    Autotuner.run = _run_first_valid_config
    _g_pn362_installed = True
    _g_pn362_log.info(
        "[PN362] VLLM_TRITON_FORCE_FIRST_CONFIG=1 — "
        "Autotuner.run replaced; Triton autotune now deterministic.")


if _g_pn362_os.environ.get(
        "VLLM_TRITON_FORCE_FIRST_CONFIG", "0").strip().lower() in (
        "1", "true", "yes", "on"):
    _g_pn362_install()
# ─── /[Genesis PN362] ────────────────────────────────────────────────
'''

# Anchor: the canonical last line of upstream env_override.py.
# Verified 2026-06-09 against pin 0.22.1rc1.dev259+g303916e93 inside
# vllm-qwen3.6-35b-balanced-k3 container: env_override.py ends with
# exactly this call (33,227 bytes). Append the helper after it.
PN362_ENV_OVERRIDE_ANCHOR = "_patch_inductor_fallback_allow_list()\n"
PN362_ENV_OVERRIDE_REPLACEMENT = (
    "_patch_inductor_fallback_allow_list()\n" + PN362_HELPER_BLOCK
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN362", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _upstream_already_present(vllm_root: Path) -> bool:
    """Detect post-merge state: upstream file exists in install."""
    return (vllm_root / "triton_utils" / "force_first_config.py").exists()


def apply() -> tuple[str, str]:
    """Apply PN362 — VLLM_TRITON_FORCE_FIRST_CONFIG into env_override.py."""
    if _env_disabled():
        return "skipped", "PN362 disabled via GENESIS_DISABLE_PN362=1"

    target = resolve_vllm_file("env_override.py")
    if target is None:
        return "skipped", "PN362: env_override.py not found in vllm install"

    target_path = Path(str(target))
    vllm_root = target_path.parent
    if _upstream_already_present(vllm_root):
        return "skipped", (
            "PN362: upstream vllm/triton_utils/force_first_config.py exists "
            "— vllm#42425 merged into pin; no vendoring needed."
        )

    patcher = TextPatcher(
        patch_name="PN362 env_override.py — VLLM_TRITON_FORCE_FIRST_CONFIG",
        target_file=str(target_path),
        marker=GENESIS_PN362_MARKER,
        sub_patches=[
            TextPatch(
                name="pn362_env_override_install_block",
                anchor=PN362_ENV_OVERRIDE_ANCHOR,
                replacement=PN362_ENV_OVERRIDE_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN362",
            "VLLM_TRITON_FORCE_FIRST_CONFIG",  # upstream sentinel if PR merges
            "force_first_config.install",
        ],
    )
    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN362: apply raised {e!r}"

    if result == TextPatchResult.FAILED:
        return "failed", (
            f"PN362: FAILED — {failure.reason if failure else 'unknown'}"
        )
    if result == TextPatchResult.SKIPPED:
        return "skipped", (
            f"PN362: skipped — {failure.reason if failure else 'unknown'}"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN362: idempotent (already applied)"

    return "applied", (
        "PN362 applied: 1 sub-patch into env_override.py — "
        "VLLM_TRITON_FORCE_FIRST_CONFIG=1 now monkey-patches "
        "Triton Autotuner.run to pick first-valid config (kills "
        "autotune variance across container restarts). Vendor of "
        "OPEN PR vllm#42425. Composes with PN345 (shmem-aware "
        "pre-autotune pruner): PN345 drops OOR configs, PN362 "
        "picks first surviving. Default off — opt-in for bench "
        "A/B and determinism debugging; leave off in PROD."
    )


def is_applied() -> bool:
    target = resolve_vllm_file("env_override.py")
    if target is None:
        return False
    try:
        if GENESIS_PN362_MARKER in Path(str(target)).read_text(
                encoding="utf-8"):
            return True
    except (OSError, UnicodeDecodeError):
        return False
    return False
