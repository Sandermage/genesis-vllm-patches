# SPDX-License-Identifier: Apache-2.0
"""PN283 — vLLM v1 multiprocess Prometheus directory bootstrap.

vLLM v1 already supports prometheus_client multiprocess mode via
``vllm.v1.metrics.prometheus.get_prometheus_registry()``, gated by the
standard ``PROMETHEUS_MULTIPROC_DIR`` env var. vLLM only auto-installs
that env var when ``--api-server-count > 1`` (see
``vllm/entrypoints/cli/serve.py:run_multi_api_server``). Our launchers
use the single-API-server path, so the bridge is never auto-enabled,
which strands PN282's worker-process counters off the API server's
``/metrics`` endpoint (see ``ACCEPTANCE_METRIC_DESIGN_2026-05-20.md``
§16).

PN283 closes that gap by making the operator set
``PROMETHEUS_MULTIPROC_DIR`` in the container launcher; this module is
defense-in-depth on top of that decision.

This module:

  * Reads ``PROMETHEUS_MULTIPROC_DIR`` from the environment
  * If unset → no-op (single-process mode preserved)
  * If set → ensure the directory exists with mode 0700
  * Optional cleanup of stale files **only** when
    ``SNDR_PROMETHEUS_MULTIPROC_CLEAN=1`` (operator must opt in;
    Python never auto-removes files by default)
  * Warn if the directory is non-empty and cleanup is not requested
  * Verify writability with a probe file
  * Emit one INFO log line summarising the outcome
  * Idempotent (subsequent calls return immediately)

The module does **not** touch ``prometheus_client`` itself. The library
inspects ``PROMETHEUS_MULTIPROC_DIR`` at value-allocation time;
prerequisite is only that the directory exists and is writable by the
time the first ``Counter()`` / ``Gauge()`` value is created.

Author: Sandermage; PN283 / 2026-05-20.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.observability.multiproc_bootstrap")

_ENV_DIR = "PROMETHEUS_MULTIPROC_DIR"
_ENV_CLEAN = "SNDR_PROMETHEUS_MULTIPROC_CLEAN"
_PROBE_NAME = ".sndr_writable_check"

_ALREADY_RAN = False


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def setup_prometheus_multiproc_dir() -> tuple[str, str]:
    """Bootstrap the multiproc directory if the env var is set.

    Returns:
        (status, reason). ``status`` is one of ``skipped``, ``applied``,
        ``warned``. ``reason`` is a short human-readable summary.
    """
    global _ALREADY_RAN

    if _ALREADY_RAN:
        return "skipped", "already initialised this process"

    raw = os.environ.get(_ENV_DIR, "").strip()
    if not raw:
        _ALREADY_RAN = True
        return "skipped", f"{_ENV_DIR} unset"

    target = raw

    # Step 1: ensure directory exists
    try:
        os.makedirs(target, mode=0o700, exist_ok=True)
    except OSError as e:
        log.warning(
            "[PN283] could not create %s=%s: %s",
            _ENV_DIR, target, e,
        )
        _ALREADY_RAN = True
        return "warned", f"mkdir failed: {type(e).__name__}: {e}"

    # Step 2: optional cleanup (operator opt-in only)
    clean = _truthy(os.environ.get(_ENV_CLEAN, ""))
    removed = 0
    if clean:
        try:
            for entry in os.listdir(target):
                full = os.path.join(target, entry)
                if os.path.isfile(full):
                    try:
                        os.unlink(full)
                        removed += 1
                    except OSError as e:
                        log.warning(
                            "[PN283] cleanup could not remove %s: %s",
                            full, e,
                        )
            log.info(
                "[PN283] %s=1 — removed %d stale file(s) from %s",
                _ENV_CLEAN, removed, target,
            )
        except OSError as e:
            log.warning(
                "[PN283] cleanup listdir failed for %s: %s",
                target, e,
            )

    # Step 3: non-empty warning if cleanup not requested
    if not clean:
        try:
            entries = os.listdir(target)
            stale = [e for e in entries if not e.startswith(".")]
            if stale:
                log.warning(
                    "[PN283] %s=%s is non-empty (%d entries); stale "
                    "prometheus_client files may inflate counters on "
                    "this boot. Set %s=1 in the launcher to clean on "
                    "container start.",
                    _ENV_DIR, target, len(stale), _ENV_CLEAN,
                )
        except OSError:
            pass

    # Step 4: writability probe — observability must never block boot,
    # so failure is a warning, not an error.
    probe = os.path.join(target, _PROBE_NAME)
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.unlink(probe)
    except OSError as e:
        log.warning(
            "[PN283] %s=%s is not writable: %s — multiproc Counters "
            "will fail silently when prometheus_client attempts to "
            "open value files.",
            _ENV_DIR, target, e,
        )
        _ALREADY_RAN = True
        return "warned", f"not writable: {type(e).__name__}: {e}"

    log.info(
        "[PN283] %s=%s ready (cleanup=%s, removed=%d)",
        _ENV_DIR, target, str(clean).lower(), removed,
    )
    _ALREADY_RAN = True
    if clean:
        return "applied", f"dir ready, cleaned {removed} stale file(s)"
    return "applied", "dir ready"


def is_initialised() -> bool:
    """True if ``setup_prometheus_multiproc_dir`` has been called in
    this process."""
    return _ALREADY_RAN


def _reset_module_state() -> None:
    """Test-only: clear the idempotency latch."""
    global _ALREADY_RAN
    _ALREADY_RAN = False


__all__ = [
    "setup_prometheus_multiproc_dir",
    "is_initialised",
]
