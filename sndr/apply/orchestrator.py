# SPDX-License-Identifier: Apache-2.0
"""SNDR Core apply — main orchestrator (run + main).

`run(apply=False)` is the entry point invoked by the apply_all CLI and
by the apply_all auto-load triggered by vllm.general_plugins. It:

  1. Imports `_per_patch_dispatch` (which @register_patch's all 95
     functions, populating `_state.PATCH_REGISTRY`).
  2. Invokes each registered function in order.
  3. Returns a PatchStats object summarizing applied/skipped/failed counts.

`main()` is the CLI entrypoint with arg parsing.

Migration history:
  - Original location: vllm/_genesis/patches/apply_all.py (Stage 0).
  - Stage 3 (CURRENT): extracted into apply/orchestrator.py.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

# Import shared state. orchestrator MUTATES `_state._APPLY_MODE` to switch
# between dry-run and apply mode for this run.
from . import _state
from ._state import (
    PatchResult,
    PatchStats,
    _applied,
    _failed,
    _skipped,
)

# Importing _per_patch_dispatch triggers @register_patch decorators on all
# 95 apply_patch_X functions — after this line, _state.PATCH_REGISTRY is
# populated and ready for run() to iterate.
from . import _per_patch_dispatch  # noqa: F401  (side-effect import)

log = logging.getLogger("genesis.apply_all")


# ═══════════════════════════════════════════════════════════════════════════
#                       Dependency / conflict resolver (S2.4)
# ═══════════════════════════════════════════════════════════════════════════


def _is_env_enabled(env_flag: str | None) -> bool:
    """Return True if the env flag is truthy (1/true/yes/on)."""
    if not env_flag:
        return False
    val = os.environ.get(env_flag, "").strip().lower()
    return val in ("1", "true", "yes", "y", "on")


def _strict_dep_mode() -> bool:
    """When ``GENESIS_STRICT_DEPS=1``, hard-block on conflicts via
    SystemExit(2). Default OFF so legacy ops trees keep booting and
    only see the warning surface."""
    return _is_env_enabled("GENESIS_STRICT_DEPS")


def _validate_dependency_graph() -> None:
    """S2.4 audit closure 2026-05-08 — walk env-enabled subset of
    PATCH_REGISTRY and emit warnings/errors per `requires_patches` and
    `conflicts_with` metadata.

    Three tiers of finding:

      • dep_missing: enabled patch X declares ``requires_patches=[Y]``
        but Y's env_flag is not set. Logged as WARNING (X may still
        work if upstream merged Y, but operator should investigate).
      • conflict_active: enabled patches X and Y both declare each
        other in ``conflicts_with``. Logged as ERROR. Hard-block on
        ``GENESIS_STRICT_DEPS=1``.
      • dep_unknown: requires_patches references a patch_id not in the
        registry. Schema drift — logged as ERROR regardless of env.
    """
    from sndr.dispatcher import PATCH_REGISTRY

    # 1. Build set of currently-enabled patches.
    #
    # STRICT OPT-IN (Sander directive 2026-05-17): only env-flag-enabled
    # patches count as "enabled". `default_on=True` is informational under
    # strict mode and does NOT trigger inclusion in the dependency-graph
    # active set. To keep pre-2026-05-17 semantics (default_on auto-apply),
    # set GENESIS_LEGACY_DEFAULT_ON=1.
    import os as _os
    _legacy_mode = _os.environ.get(
        "GENESIS_LEGACY_DEFAULT_ON", ""
    ).strip().lower() in ("1", "true", "yes")
    enabled: set[str] = set()
    for pid, meta in PATCH_REGISTRY.items():
        if _legacy_mode and meta.get("default_on"):
            enabled.add(pid)
            continue
        if _is_env_enabled(meta.get("env_flag")):
            enabled.add(pid)

    if not enabled:
        log.debug("[Genesis dep-graph] no enabled patches — skip")
        return

    log.info(
        "[Genesis dep-graph] checking %d enabled patches against "
        "requires_patches / conflicts_with metadata",
        len(enabled),
    )

    issues = {"dep_missing": [], "conflict_active": [], "dep_unknown": []}

    for pid in enabled:
        meta = PATCH_REGISTRY[pid]
        for req in (meta.get("requires_patches") or []):
            if req not in PATCH_REGISTRY:
                issues["dep_unknown"].append((pid, req))
            elif req not in enabled:
                # req present in registry but not enabled
                # Some "requires" are advisory (req auto-skips upstream-merged) —
                # we still WARN so operators see the wiring expectation.
                issues["dep_missing"].append((pid, req))
        for conflict in (meta.get("conflicts_with") or []):
            if conflict in enabled:
                # Only log once per pair (alphabetical ordering avoids dup)
                if pid < conflict:
                    issues["conflict_active"].append((pid, conflict))

    # Emit findings
    for pid, req in issues["dep_unknown"]:
        log.error(
            "[Genesis dep-graph] %s declares requires_patches=[%s] but %s "
            "is NOT in PATCH_REGISTRY — schema drift",
            pid, req, req,
        )
    for pid, req in issues["dep_missing"]:
        log.warning(
            "[Genesis dep-graph] %s requires %s but %s is not enabled. "
            "Either enable %s or verify upstream merged it (auto-skip path).",
            pid, req, req, req,
        )
    for a, b in issues["conflict_active"]:
        log.error(
            "[Genesis dep-graph] CONFLICT: %s and %s both enabled but "
            "conflicts_with declared. Disable one of them to avoid "
            "undefined behaviour.",
            a, b,
        )

    has_critical = bool(issues["dep_unknown"] or issues["conflict_active"])
    if has_critical and _strict_dep_mode():
        log.error(
            "[Genesis dep-graph] GENESIS_STRICT_DEPS=1 set — REFUSING "
            "to apply with %d unknown deps + %d active conflicts.",
            len(issues["dep_unknown"]), len(issues["conflict_active"]),
        )
        import sys
        sys.exit(2)

    if not has_critical and not issues["dep_missing"]:
        log.info("[Genesis dep-graph] OK — no issues across %d enabled patches",
                 len(enabled))


# ═══════════════════════════════════════════════════════════════════════════
#                                 RUN
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
    _state._APPLY_MODE = apply

    # [Genesis T4.6] Compile-time watchdog — log total apply elapsed.
    # Triton kernel pre-build (e.g. PN26b _build_kernel() at apply()) can
    # take 30-90s on cold cache. >120s is a red flag (autotune regression
    # or stale cache mismatch) — investigate before user requests start.
    _t0_apply = time.perf_counter()

    stats = PatchStats()

    # Platform diagnostic — helps debugging on unexpected hardware
    try:
        from sndr.engines.vllm.detection.guards import platform_summary
        summary = platform_summary()
        if verbose:
            log.info("Genesis platform: %s",
                     json.dumps(summary, default=str, indent=None))
    except Exception as e:
        log.warning("Platform summary failed: %s", e)

    # [Genesis pin-gate] Sander 2026-05-04 — "foolproof guard". Runs in
    # BOTH plugin auto-load (run() called from register()) AND CLI PRE-pass
    # (run() called from main()). Strict mode = sys.exit(2) on unknown pin.
    try:
        from sndr.engines.vllm.detection.guards import (
            assert_vllm_pin_allowed,
            get_vllm_full_version_string,
            KNOWN_GOOD_VLLM_PINS,
        )
        pin = get_vllm_full_version_string() or "unknown"
        log.info("[Genesis pin-gate] running vllm pin = %s", pin)
        log.info(
            "[Genesis pin-gate] allowlist (%d entries): %s",
            len(KNOWN_GOOD_VLLM_PINS), list(KNOWN_GOOD_VLLM_PINS),
        )
        status, message = assert_vllm_pin_allowed()
        if status == "ok":
            log.info("[Genesis pin-gate] OK — %s", message)
        else:
            log.warning("[Genesis pin-gate] %s — %s", status.upper(), message)
    except SystemExit:
        # strict-mode hard-stop already printed; propagate exit
        raise
    except Exception as e:
        log.warning("[Genesis pin-gate] check skipped (error: %s)", e)

    # [Genesis T2.2] FLA TP int32 overflow preflight (vllm#40265 family).
    # Opportunistic check: when operator sets the four shape env vars
    # (GENESIS_FLA_GUARD_TP_SIZE / NUM_HEADS / HEAD_DIM / SEQ_LEN), we
    # compute the FLA/GDN flat-index magnitude and log status. Hard-
    # blocks apply_all only if int64 wraps (theoretically impossible on
    # current hardware; the check is forward-defensive). Otherwise
    # WARN-only (operator decides). NULL when env vars are unset (no
    # overhead).
    try:
        import os as _os
        from sndr.engines.vllm.kernels_legacy.fla_tp_device_index_guard import (
            check_index_overflow,
        )

        _shape_env = {
            "tp_size": _os.environ.get("GENESIS_FLA_GUARD_TP_SIZE"),
            "num_heads": _os.environ.get("GENESIS_FLA_GUARD_NUM_HEADS"),
            "head_dim": _os.environ.get("GENESIS_FLA_GUARD_HEAD_DIM"),
            "seq_len": _os.environ.get("GENESIS_FLA_GUARD_SEQ_LEN"),
        }
        if all(v is not None for v in _shape_env.values()):
            _dtype_bytes = int(
                _os.environ.get("GENESIS_FLA_GUARD_DTYPE_BYTES", "2")
            )
            report = check_index_overflow(
                tp_size=int(_shape_env["tp_size"]),
                num_heads=int(_shape_env["num_heads"]),
                head_dim=int(_shape_env["head_dim"]),
                seq_len=int(_shape_env["seq_len"]),
                dtype_bytes=_dtype_bytes,
            )
            if not report.fits_int64:
                log.error(
                    "[Genesis FLA-guard] int64 overflow on flat indices "
                    "(magnitude=%d). REFUSING apply_all — kernels would "
                    "silently corrupt. Reduce TP, heads, head_dim, or "
                    "max_seq_len.", report.magnitude,
                )
                raise SystemExit(2)
            if not report.fits_int32:
                log.warning(
                    "[Genesis FLA-guard] flat-index space %d exceeds int32 "
                    "range. FLA/GDN kernels using int32 indexing will "
                    "silently corrupt. Margin int64=%.1f%%. Reference: "
                    "vllm#40265 family.",
                    report.magnitude, report.margin_int64_pct,
                )
            elif report.margin_int32_pct > 50.0:
                log.warning(
                    "[Genesis FLA-guard] flat-index margin int32=%.1f%% "
                    "(>50%% of range used) — close to silent-overflow "
                    "boundary. Consider int64 indexing on FLA kernels.",
                    report.margin_int32_pct,
                )
            else:
                log.info(
                    "[Genesis FLA-guard] OK — flat-index magnitude=%d, "
                    "margin int32=%.1f%%.",
                    report.magnitude, report.margin_int32_pct,
                )
    except SystemExit:
        raise
    except Exception as e:
        log.debug("[Genesis FLA-guard] preflight skipped: %s", e)

    # PDL misconfig check (vLLM issue #40742). Warn loudly but don't fail —
    # some environments set these globally and other GPUs in the cluster use
    # them. On the local Ampere rank, we just advise unsetting.
    try:
        from sndr.engines.vllm.detection.guards import detect_pdl_env_misconfig
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

    # [Genesis S2.4] Patch dependency / conflict pre-flight (audit closure
    # 2026-05-08 noonghunna). PATCH_REGISTRY entries declare metadata via
    # `requires_patches` and `conflicts_with`, but until Wave 7 nothing
    # consulted them at apply time — silent contract drift was possible.
    # This block walks the env-enabled subset and emits structured
    # warnings/errors. Hard-block via SystemExit only when
    # GENESIS_STRICT_DEPS=1 (operator opt-in).
    try:
        _validate_dependency_graph()
    except SystemExit:
        raise
    except Exception as e:
        log.debug("[Genesis dep-graph] preflight skipped: %s", e)

    # Banner
    log.info(
        "Genesis Unified Patch v7.0 — Ampere FP8 + TQ + MoE + Hybrid + bugfixes. "
        "Philosophy: we fix, we do not break."
    )

    # Validate _state.PATCH_REGISTRY shape + dependency graph at boot. Issues are
    # logged so operators see drift (e.g. unknown env_flag pattern, missing
    # superseded_by on deprecated patch, requires_patches referencing an
    # unknown ID). ERROR-level issues are surfaced loudly; WARNING are
    # logged at INFO so they don't drown the boot log on a busy registry.
    # The registry IS the contract — silent drift is the failure mode this
    # block was added to catch.
    try:
        from sndr.dispatcher import (
            PATCH_REGISTRY as _GENESIS_DISPATCHER_REGISTRY,
            validate_registry,
        )
        registry_issues = validate_registry()
        for i in registry_issues:
            if i.severity == "ERROR":
                log.error(
                    "[Genesis registry] %s: %s",
                    i.patch_id, i.message,
                )
            elif i.severity == "WARNING":
                log.warning(
                    "[Genesis registry] %s: %s",
                    i.patch_id, i.message,
                )
            else:
                log.info(
                    "[Genesis registry] %s: %s",
                    i.patch_id, i.message,
                )
        if verbose:
            n_err = sum(1 for i in registry_issues if i.severity == "ERROR")
            if n_err == 0:
                log.info(
                    "[Genesis registry] %d dispatcher entries — "
                    "schema-clean, dependency graph consistent.",
                    len(_GENESIS_DISPATCHER_REGISTRY),
                )
            else:
                log.error(
                    "[Genesis registry] %d entries — %d ERROR(s) above. "
                    "Apply will continue but operators must investigate.",
                    len(_GENESIS_DISPATCHER_REGISTRY), n_err,
                )
    except Exception as e:
        log.debug("[Genesis registry] validation skipped: %s", e)

    # GPU profile + per-patch recommendations (suggest-only, never auto-enables)
    try:
        from sndr.runtime.gpu_profile import print_recommendations
        rec_text = print_recommendations(stream=None)
        for line in rec_text.split("\n"):
            log.info(line)
    except Exception as e:
        log.debug("[gpu_profile] recommendation skipped: %s", e)

    # [Phase 5b plugins] Discover + register community plugin patches
    # via setuptools entry-points. OPT-IN: only fires when
    # GENESIS_ALLOW_PLUGINS=1. Default behavior: zero foreign code loaded.
    try:
        from sndr.compat.plugins import (
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

    # G-006 fix (audit 2026-05-02): Phase 5c apply_callable plugin pass
    # was previously HERE (BEFORE core patch loop), contradicting the
    # docstring "After core patches finish, walk plugins". Moved BELOW
    # the core patch loop (just before telemetry) so plugin authors can
    # rely on core patches being already applied — they may text-patch
    # files that core patches have already modified, and need to find
    # the post-modification anchors.

    # [Phase 5d telemetry] Opt-in anonymized telemetry. Default OFF —
    # only fires when GENESIS_ENABLE_TELEMETRY=1. Even when ON, only
    # saves locally. Network upload is a separate gate
    # (GENESIS_TELEMETRY_UPLOAD=1) and is currently a no-op until the
    # community dashboard is live.
    try:
        from sndr.compat.telemetry import (
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

    # ── Stage 8 (2026-05-07): bundle activation (BEFORE per-patch loop).
    # Each bundle's umbrella flag (`SNDR_ENABLE_BUNDLE_*`) triggers
    # atomic apply of 2+ semantically-related patches via
    # MultiFilePatchTransaction. Running BEFORE the per-patch loop:
    #   - Atomic intent honored first (operator chose "activate this
    #     feature group atomically").
    #   - Per-patch loop subsequently sees marker present → IDEMPOTENT,
    #     no double-apply.
    # If a bundle is disabled, this is a single dict lookup + skip;
    # negligible boot-time overhead.
    _run_bundles(stats)

    # PR38 Day 6-8 (2026-05-08): operator can opt into the new
    # registry-driven apply loop via `SNDR_APPLY_VIA_SPECS=1`. The new
    # loop iterates `dispatcher.iter_patch_specs()` and dispatches via
    # `spec.apply_module` instead of the hand-written `apply_patch_X`
    # parking lot. Default OFF until live PROD smoke validates the
    # contract is identical to the legacy loop.
    import os as _os_for_spec_flag
    _use_spec_loop = _os_for_spec_flag.environ.get(
        "SNDR_APPLY_VIA_SPECS", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if _use_spec_loop:
        _run_via_specs(stats)
        log.info("Genesis %s", stats)
        # Skip the legacy loop entirely under spec-driven mode.
        return stats

    # Apply each patch
    for patch_name, patch_fn in _state.PATCH_REGISTRY:
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

    # [Genesis 2026-06-14] Spec-only supplement. The legacy loop above only
    # applies patches with a @register_patch hook; the spec-only patches
    # declare an apply_module but no legacy hook (KNOWN_SPEC_ONLY) and were
    # previously unreachable at boot under the default mode even when an
    # operator set GENESIS_ENABLE_<X>=1. This pass applies the ENABLED ones.
    # The default-boot no-op is guaranteed by STRICT OPT-IN: should_apply
    # gates on the env flag, so nothing applies until explicitly opted in
    # (default_on is informational, not the driver). One spec-only patch
    # (PN252, security) is default_on=True and WILL apply here under
    # GENESIS_LEGACY_DEFAULT_ON=1. On a clean default boot the apply summary
    # is unchanged. See _run_spec_only_supplement for the full rationale.
    _run_spec_only_supplement(stats)

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
        from sndr.dispatcher import log_apply_matrix
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
        from sndr.dispatcher import (
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

    # [Phase 5c apply_callable, G-006 audit fix 2026-05-02] After the
    # core patch loop finishes, walk plugins whose env flags are set
    # and call their apply_callable. Plugin failures are isolated
    # (logged, counted, never crash apply_all). Skipped when
    # GENESIS_ALLOW_PLUGINS gate is closed. Re-runs validate_registry
    # so plugin entries injected at register_plugins() time are
    # included in the boot-time validation pass (G-007 fix).
    if apply:
        try:
            from sndr.compat.plugins import apply_all_plugins
            plugin_stats = apply_all_plugins()
            if plugin_stats.get("total", 0) > 0:
                log.info(
                    "[Genesis plugins] apply pass: total=%d applied=%d "
                    "skipped=%d failed=%d",
                    plugin_stats["total"], plugin_stats["applied"],
                    plugin_stats["skipped"], plugin_stats["failed"],
                )
                # G-007 fix: re-validate registry now that plugin entries
                # were potentially added during register_plugins().
                try:
                    from sndr.dispatcher import validate_registry
                    post_plugin_issues = validate_registry()
                    n_plugin_err = sum(
                        1 for i in post_plugin_issues if i.severity == "ERROR"
                    )
                    if n_plugin_err > 0:
                        log.error(
                            "[Genesis registry] post-plugin validation: "
                            "%d ERROR(s) — operator should investigate",
                            n_plugin_err,
                        )
                        for i in post_plugin_issues:
                            if i.severity == "ERROR":
                                log.error(
                                    "[Genesis registry plugin] %s: %s",
                                    i.patch_id, i.message,
                                )
                except Exception as ve:
                    log.debug(
                        "[Genesis registry] post-plugin validation skipped: %s",
                        ve,
                    )
        except Exception as e:
            log.debug("[plugins] apply pass skipped: %s", e)

    # [Genesis T4.6] Compile-time watchdog post-summary.
    _elapsed = time.perf_counter() - _t0_apply
    if _elapsed > 120:
        log.warning(
            "[Genesis compile-watchdog] apply_all took %.1fs (>120s threshold) — "
            "investigate Triton compile cache state, autotune regression, or "
            "stale .so files. Consider clearing TRITON_CACHE_DIR + retrying.",
            _elapsed,
        )
    elif _elapsed > 60:
        log.info(
            "[Genesis compile-watchdog] apply_all elapsed: %.1fs (warm cache "
            "should be < 30s; first cold-compile boot may take up to 90s)",
            _elapsed,
        )
    else:
        log.info("[Genesis compile-watchdog] apply_all elapsed: %.1fs", _elapsed)
    stats.compile_elapsed_sec = _elapsed

    # ─────────────────────────────────────────────────────────────────
    # [v7.72.2 fix 2026-05-05] Structured boot summary emit point.
    #
    # MUST live in run() (not main()) — vllm's plugin loader calls run()
    # via the genesis_v7 entry point, never main(). Putting the summary
    # only in main() meant it appeared on `python3 -m vllm._genesis.
    # patches.apply_all` CLI runs but NEVER on real production boot.
    # This regression silently shipped between v7.70 and v7.72.2.
    #
    # Falls back to v7.13 apply matrix on any error so boot keeps
    # working. Errors logged at WARN so operators see them (not the old
    # silent debug log that hid the bug).
    # ─────────────────────────────────────────────────────────────────
    try:
        from sndr.dispatcher import log_structured_boot_summary
        log_structured_boot_summary()
    except Exception as e:
        log.warning(
            "[Genesis] structured boot summary unavailable (%s: %s) — "
            "falling back to v7.13 apply matrix. Check "
            "dispatcher.dump_structured_boot_summary().",
            type(e).__name__, e,
        )
        try:
            from sndr.dispatcher import log_apply_matrix
            log_apply_matrix()
        except Exception as e2:
            log.warning(
                "[Genesis] v7.13 apply matrix fallback also unavailable: %s: %s",
                type(e2).__name__, e2,
            )

    # boot-opt §4.1 (2026-06-17): persist the patch-file cache to disk ONCE
    # here, after the whole apply loop + spec-only supplement, instead of
    # ~170 per-patch fsyncs inside record_apply_result (O(N^2)->O(N)). Only on
    # a real apply boot (dry-run never mutates the cache). Failure is swallowed
    # inside flush_file_cache() — never breaks boot.
    if apply:
        try:
            from sndr.engines.vllm.wiring.file_cache import flush_file_cache
            flush_file_cache()
        except Exception as e:  # noqa: BLE001
            log.warning("[Genesis] end-of-boot file_cache flush skipped: %s", e)

    return stats




def _run_via_specs(stats: PatchStats) -> None:
    """PR38 Day 6-8 alternative: iterate `iter_patch_specs()` and call
    `module.apply()` directly via `spec.apply_module`.

    This replaces the hand-written `_per_patch_dispatch.py` parking lot
    with a data-driven dispatch. Each PatchSpec's `apply_module` field
    points at a canonical `sndr.engines.vllm.patches.<family>.<file>`
    module exposing `apply() -> tuple[str, str]`.

    Specs without `apply_module` are skipped with reason "no apply_module
    declared" — they're either legacy P1-P46 stubs (no per-file impl) or
    informational hooks integrated elsewhere (PN26b, P102, etc).

    P0-2 fix (audit 2026-05-08): the dispatcher decision (`should_apply`)
    is consulted BEFORE importing the apply_module. Disabled patches —
    env-flag off, tier ineligible, hardware mismatch, conflicts active —
    no longer import their (potentially torch-heavy) wiring module just
    to be skipped. This makes dry-run a real torch-less diagnostic path:
    every disabled patch records as 'skipped' without touching torch.

    Failures during import / call surface as `_failed(...)` with the
    exception in the reason; same contract as the legacy loop.
    """
    import os as _os_for_topo
    from sndr.dispatcher.spec import iter_patch_specs

    # v11.3.0 BUG #11: opt-in topological sort by `requires_patches`.
    # Default OFF — keeps current behavior (registry-insertion order).
    # When ON, dependencies apply BEFORE dependents, fixing 6 known
    # order violations at the v11.3.0 baseline. See
    # sndr.dispatcher.spec.iter_patch_specs docstring.
    _topo = _os_for_topo.environ.get(
        "SNDR_TOPO_SORT_SPECS", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if _topo:
        log.info(
            "[Genesis spec-driven] SNDR_TOPO_SORT_SPECS=1 — "
            "applying patches in topological order by requires_patches"
        )

    n_applied = 0
    n_skipped = 0
    n_failed = 0
    for spec in iter_patch_specs(topo_sort=_topo):
        # Per-spec gate→import→apply→classify is centralized in
        # `_apply_spec_module` so the spec-only supplement (legacy-boot
        # bridge) shares byte-identical behavior — see that helper's
        # docstring for why a single code path matters here.
        status = _apply_spec_module(spec, stats)
        if status == "applied":
            n_applied += 1
        elif status == "skipped":
            n_skipped += 1
        else:
            n_failed += 1

    log.info(
        "[Genesis spec-loop] %d applied / %d skipped / %d failed "
        "(SNDR_APPLY_VIA_SPECS=1)",
        n_applied, n_skipped, n_failed,
    )


def _apply_spec_module(spec, stats: PatchStats) -> str:
    """Gate, import, and apply a single PatchSpec via its `apply_module`.

    Appends exactly one PatchResult to `stats` and returns its status
    string — one of ``"applied"`` / ``"skipped"`` / ``"failed"``.

    Extracted from `_run_via_specs` (2026-06-14) so the full spec-driven
    boot loop AND the spec-only supplement (`_run_spec_only_supplement`)
    drive patches through one identical sequence. A behavioral drift
    between the two callers would mean a patch applies differently
    depending on which boot mode reached it — precisely the class of bug
    PR38's `shadow --strict` parity gate exists to prevent — so the logic
    lives here once.

    P0-2 contract preserved: the dispatcher decision (`should_apply`) is
    consulted BEFORE importing the (potentially torch-heavy) apply_module,
    so disabled patches record as 'skipped' on a torch-less host without
    importing anything. Import failures naming torch/triton/vllm/flashinfer
    are classified as a clean runtime-gap skip; only structurally-broken
    wiring surfaces as 'failed'.
    """
    import importlib
    from sndr.dispatcher.decision import log_decision, should_apply

    # Use the spec.title as the displayed name (matches what the legacy
    # `@register_patch("...")` decorator passed in).
    display = f"{spec.patch_id} {spec.title}".strip()
    if spec.apply_module is None:
        stats.results.append(_skipped(
            display, "no apply_module declared (informational entry)"
        ))
        return "skipped"

    # P0-2: gate decision FIRST. If the patch is disabled the import of
    # `spec.apply_module` (which may be torch-heavy) is avoided — critical
    # for torch-less dry-run on Mac dev / CI / preflight.
    decision, reason = should_apply(spec.patch_id)
    # Record the decision into the boot apply-matrix. Many spec-only patches
    # never call log_decision() inside their own apply() (or skip before
    # reaching it), so without this the spec-driven path was invisible to
    # get_apply_matrix() — and thus to the apply-plan validator / doctor /
    # telemetry that read it. log_decision() is idempotent by patch_id, so a
    # patch whose apply() also logs is not double-counted.
    log_decision(spec.patch_id, decision, reason)
    if not decision:
        stats.results.append(_skipped(display, reason))
        return "skipped"

    try:
        mod = importlib.import_module(spec.apply_module)
    except ImportError as e:
        # P0-2: distinguish missing-runtime (torch absent on host) from
        # real wiring import errors. The former is a clean skip with a
        # structured reason; only structurally-broken imports surface as
        # failures. Heuristic: ModuleNotFoundError naming
        # `torch` / `triton` / `vllm` / `flashinfer` ⇒ runtime gap.
        msg = str(e).lower()
        runtime_gap_markers = ("torch", "triton", "flashinfer", "vllm")
        if any(m in msg for m in runtime_gap_markers):
            stats.results.append(_skipped(
                display,
                f"runtime not present on this host ({e}) — patch "
                "would apply on a vllm-equipped server",
            ))
            return "skipped"
        stats.results.append(_failed(
            display, f"apply_module import failed: {e}"
        ))
        log.error("[Genesis spec-loop] FAILED import %s: %s",
                  spec.apply_module, e)
        return "failed"
    except Exception as e:
        stats.results.append(_failed(
            display, f"apply_module import raised: {type(e).__name__}: {e}"
        ))
        log.error("[Genesis spec-loop] FAILED import %s: %s",
                  spec.apply_module, e)
        return "failed"
    if not hasattr(mod, "apply"):
        stats.results.append(_failed(
            display, f"{spec.apply_module} missing apply() function"
        ))
        return "failed"
    if not _state._APPLY_MODE:
        stats.results.append(_applied(
            display, "dry-run: apply_module ready"
        ))
        return "applied"
    # v11.4.0 (Phase 6 P3.4 observability parity): wrap mod.apply() in
    # measure_patch_apply() context so PatchMetrics + observability spans
    # fire for the spec-driven path identically to the legacy
    # @register_patch-instrumented path. Without this wrap,
    # GENESIS_OBSERVABILITY=1 boots that use SNDR_APPLY_VIA_SPECS=1 would
    # lose per-patch elapsed_ms + rss_delta_kb telemetry. The context
    # manager is a no-op when observability is off.
    try:
        from sndr.observability.patch_metrics import (
            measure_patch_apply,
        )
        _patch_metric_cm = measure_patch_apply(display)
    except ImportError:
        # Observability stack absent — fall back to a no-op context.
        from contextlib import nullcontext
        _patch_metric_cm = nullcontext()

    try:
        with _patch_metric_cm as _metric:
            status, reason = mod.apply()
            # Mirror the legacy instrumentation behavior — populate metric
            # fields with the result for observability emit.
            if _metric is not None:
                try:
                    _metric.status = status
                    _metric.reason = reason or ""
                except Exception:
                    # Best-effort: older metric implementations may not
                    # expose these attributes.
                    pass
    except Exception as e:
        stats.results.append(_failed(
            display, f"apply() raised: {type(e).__name__}: {e}"
        ))
        log.exception("[Genesis spec-loop] EXCEPTION in %s",
                      spec.patch_id)
        return "failed"
    if status == "applied":
        stats.results.append(_applied(display, reason))
        return "applied"
    elif status == "skipped":
        stats.results.append(_skipped(display, reason))
        return "skipped"
    stats.results.append(_failed(display, reason))
    return "failed"


def _run_spec_only_supplement(stats: PatchStats) -> None:
    """Legacy-boot bridge: apply ENABLED spec-only patches the legacy
    `@register_patch` table doesn't cover.

    Why this exists (root-caused 2026-06-14): the default boot path
    (`SNDR_APPLY_VIA_SPECS` unset) runs the legacy loop over
    `_state.PATCH_REGISTRY` — only patches carrying a hand-written
    `@register_patch` hook apply. 59 patches declare an `apply_module` but
    have NO legacy hook (the KNOWN_SPEC_ONLY set — relocated to their
    canonical technical-area home as part of PR38's migration away from the
    parking lot). Before this supplement they were unreachable at boot in
    the default mode: an operator could set `GENESIS_ENABLE_<X>=1` and the
    patch would silently never apply. That is exactly how the dev491 pin
    bump's streaming-tool-call fix (PN392) appeared inert in live smoke.

    Why not simply switch the default to `_run_via_specs`: it cannot be the
    default yet because three default_on bundled legacy patches — P1/P2
    (FP8 dispatcher), P17/P18 (Marlin MoE tuning), P32/P33 (TurboQuant
    preallocs) — have no `apply_module` and apply only via their bundled
    `@register_patch` hooks. Running the spec loop alone would silently DROP
    them on PROD. This supplement instead augments the legacy loop.

    Safety (no-op by default, no double-apply):
      * Processes ONLY spec-only patch ids — `apply_module` set AND patch id
        absent from the legacy register table — so nothing the legacy loop
        already applied is touched again.
      * Gates each candidate through `should_apply` and skips disabled ones
        SILENTLY (no stats row, no import, no log). The default-boot no-op is
        guaranteed by STRICT OPT-IN: `should_apply` gates on the env flag, so
        a candidate only does work once an operator explicitly opts it in.
        `default_on` is informational metadata, NOT what drives this pass —
        of the spec-only patches exactly one (PN252, a security patch) carries
        default_on=True, and it WILL apply here when
        `GENESIS_LEGACY_DEFAULT_ON=1` flips the default-on cohort on. A clean
        default boot (no env flags, GENESIS_LEGACY_DEFAULT_ON unset) therefore
        adds zero rows and stays byte-identical to pre-supplement behavior.
    """
    from sndr.dispatcher.spec import iter_patch_specs
    from sndr.dispatcher.decision import should_apply
    from sndr.apply.shadow import _patch_id_from_legacy_name

    # Patch ids already covered by the legacy @register_patch loop. Built
    # from the live register table (not a static list) so it tracks future
    # legacy additions automatically — exclude them to prevent double-apply.
    legacy_pids: set[str] = set()
    for name, _fn in _state.PATCH_REGISTRY:
        pid = _patch_id_from_legacy_name(name)
        if pid:
            legacy_pids.add(pid)

    n_applied = 0
    n_failed = 0
    for spec in iter_patch_specs():
        if spec.apply_module is None or spec.patch_id in legacy_pids:
            continue
        # Gate FIRST and skip disabled spec-only patches silently — this is
        # what keeps the default boot's apply summary unchanged (no extra
        # rows for the 59 opt-in patches when none are enabled).
        decision, _reason = should_apply(spec.patch_id)
        if not decision:
            continue
        status = _apply_spec_module(spec, stats)
        last = stats.results[-1]
        if status == "applied":
            n_applied += 1
            log.info("[Genesis spec-only] applied: %s — %s",
                     last.name, last.reason)
        elif status == "failed":
            n_failed += 1
            log.error("[Genesis spec-only] FAILED: %s — %s",
                      last.name, last.reason)
        # 'skipped' here means a runtime-gap on a torch-less host — left at
        # debug-silence; it would apply on the vllm-equipped server.
    if n_applied or n_failed:
        log.info(
            "[Genesis spec-only supplement] %d applied / %d failed — "
            "enabled spec-only patches not in the legacy register table",
            n_applied, n_failed,
        )


def _run_bundles(stats: PatchStats) -> None:
    """Stage 8 (2026-05-07): activate enabled bundles before per-patch loop.

    Iterates the static catalog of bundle modules. For each:
      - Calls bundle.apply() (which internally checks the umbrella flag
        and tier-gate).
      - Wraps the (status, reason) tuple as a PatchResult with name
        prefix "bundle:" so the boot summary distinguishes bundle
        results from per-patch results.
      - Appends to stats.results so reporting picks them up.

    Bundles that are NOT enabled return ("skipped", "disabled") and
    consume ~10μs each — boot overhead is negligible.

    The dry-run / apply distinction is honored by the bundle's
    underlying TextPatcher (Layer 1: read-only checks; Layer 7:
    write+verify only when actually committing). When _state._APPLY_MODE
    is False, bundle still goes through the same code path but the
    underlying TextPatcher.apply() is the one that actually writes.

    Stage 8 design note: bundles run BEFORE the per-patch loop. After
    this function returns, the marker for any applied bundle component
    is already present in the file, so the per-patch loop's TextPatcher
    sees IDEMPOTENT and short-circuits. No double-apply.
    """
    # Static catalog. Adding a new bundle = add 1 line here.
    BUNDLE_MODULES = [
        "tool_parsing_qwen3coder",
        "reasoning_qwen3",
        "attention_gdn_spec",
        "attention_tq_multi_query",
        "spec_decode_async_cleanup",
    ]

    if not _state._APPLY_MODE:
        # Dry-run: bundles still report what WOULD apply, but the
        # underlying TextPatcher won't write. The status the bundle
        # returns is informative.
        log.debug("[Genesis bundles] dry-run mode — bundles report intent only")

    for bundle_name in BUNDLE_MODULES:
        try:
            mod = __import__(
                f"sndr.bundles.{bundle_name}",
                fromlist=["apply"],
            )
            status, reason = mod.apply()
        except Exception as e:
            stats.results.append(_state._failed(
                f"bundle:{bundle_name}",
                f"bundle apply raised {type(e).__name__}: {e}",
            ))
            log.error("[Genesis bundle %s] FAILED: %s", bundle_name, e)
            continue

        result_name = f"bundle:{bundle_name}"
        if status == "applied":
            stats.results.append(_state._applied(result_name, reason))
            log.info("[Genesis bundle %s] applied — %s", bundle_name, reason)
        elif status == "skipped":
            stats.results.append(_state._skipped(result_name, reason))
            # Most bundle skips are "disabled" (operator didn't enable
            # umbrella flag) — don't spam at INFO. Tier-gated and
            # drift-skipped are more interesting → DEBUG suffices.
            log.debug("[Genesis bundle %s] skipped — %s",
                      bundle_name, reason)
        else:  # "failed"
            stats.results.append(_state._failed(result_name, reason))
            log.error("[Genesis bundle %s] FAILED — %s",
                      bundle_name, reason)


def main() -> int:
    """CLI entrypoint. Returns exit code.

    CLI default is apply=True because this entrypoint is the one invoked
    from container scripts (pre-vllm-serve) where text-patches MUST land.
    Pass `--dry-run` for diagnosis-only mode.
    Pass `--verify-rebinds` for post-register verification (additional
    verification + non-zero exit code if any rebind not live).

    Per Sander 2026-05-04: enforce vllm pin allowlist (foolproof guard).
    Set GENESIS_VLLM_PIN_POLICY=strict in production start scripts to
    sys.exit(2) on unknown pin instead of just warning.
    """
    import sys as _sys
    argv = _sys.argv[1:]
    dry = "--dry-run" in argv
    verify = "--verify-rebinds" in argv

    # Pin allowlist gate is now in run() so it triggers on every entry path
    # (CLI + plugin auto-load). No need to duplicate it here.

    try:
        stats = run(verbose=True, apply=not dry)
    except Exception as e:
        log.exception("Genesis orchestrator setup error: %s", e)
        return 2

    # NOTE: structured boot summary already emitted by run() above.
    # (v7.72.2 fix moved the call from main() into run() so the plugin
    # entry point — which only invokes run() — also gets the summary.)

    exit_code = 1 if stats.failed_count > 0 else 0

    if verify:
        log.info("[Genesis] Post-register rebind verification:")
        # Imported here (not at module top) to keep cold-import light.
        from sndr.apply.verify import verify_live_rebinds
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
