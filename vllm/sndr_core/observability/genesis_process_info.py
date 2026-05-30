# SPDX-License-Identifier: Apache-2.0
"""§6.H10 — Genesis process-info metric (canonical Prometheus pattern).

The plan §6.H10 asks for cross-system Prometheus labels on the
common metric surfaces:

  preset, profile, workload_class, K, backend, patch_hash

Adding these as labels to every vLLM-builtin counter (e.g.
`vllm:num_requests_running`, `vllm:e2e_request_latency_seconds`) would
require modifying vLLM's core metrics code — out of scope and pin-bump
fragile. The canonical alternative used by Grafana / Prometheus best
practice is the *_info* pattern: emit a single-row gauge whose value
is always `1`, labeled with the immutable process metadata. Downstream
queries then JOIN against it:

    rate(vllm:num_requests_running[5m])
      * on(instance) group_left(preset, profile, K, backend, patch_hash)
        genesis_process_info

This module emits ``genesis_process_info`` exactly once at apply()
time. The label values are resolved from:

  preset       — ``$GENESIS_PRESET`` env, fallback "unknown"
  profile      — ``$GENESIS_PROFILE`` env, fallback "unknown"
  workload_class — ``$GENESIS_WORKLOAD_CLASS`` env, fallback "unknown"
                   (operator product card emits this on launch when a
                   preset card.workload_allow is non-empty)
  K            — parsed from ``sys.argv`` ``--speculative-config`` JSON
                 (``num_speculative_tokens`` field); fallback "0"
  backend      — parsed from ``sys.argv`` ``--attention-backend``
                 OR ``$VLLM_ATTENTION_BACKEND``; fallback "default"
  patch_hash   — short SHA of HEAD in ``$GENESIS_REPO`` (= the
                 Genesis patches repo mounted into the container);
                 fallback "uncommitted"
  model        — served-model-name from ``sys.argv``
                 ``--served-model-name``; fallback "unknown"
  pin          — ``vllm.__version__``; fallback "unknown"

Cardinality budget: one row per process. Each label value space is
bounded by operator config (preset+profile+workload_class are
catalog-controlled; K + backend are small enums; patch_hash + pin are
deployment-time-frozen). Total ≤ 1 series per container instance —
canonical *_info* metric semantics, well under Prometheus best
practice.

The counter survives worker forks the same way PN287 / PN288 do —
walk ``REGISTRY._collector_to_names`` to unregister stale collectors
before re-registering. Idempotent across hot reload.

Operator query examples (PromQL):

  # Active configs in fleet
  count by (preset, profile, K, backend) (genesis_process_info)

  # Latency p99 by preset
  histogram_quantile(0.99,
      sum by (preset, le) (
          rate(vllm:e2e_request_latency_seconds_bucket[5m])
        * on(instance) group_left(preset) genesis_process_info
      )
  )

  # Patch-hash drift detection across the fleet
  count by (patch_hash) (genesis_process_info)
  # >1 row = some containers running stale Genesis

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from typing import Any, Optional


log = logging.getLogger("genesis.observability.genesis_process_info")


_LABEL_NAMES = (
    "preset",
    "profile",
    "workload_class",
    "K",
    "backend",
    "patch_hash",
    "model",
    "pin",
)

_METRIC_NAME = "genesis_process_info"

# Lazy state.
_prom_gauge: Any = None
_APPLIED = False


# ─── argv parsing ──────────────────────────────────────────────────────


_SPEC_CONFIG_RE = re.compile(
    r'\\?"num_speculative_tokens\\?"\s*:\s*(\d+)'
)


def _argv_value(flag: str, argv: list[str]) -> Optional[str]:
    """Return the value following ``flag`` in ``argv``.

    Handles both ``--flag value`` and ``--flag=value`` forms. Returns
    None when the flag is absent. Never raises — operator typos must
    not crash the metric setup.
    """
    try:
        for i, arg in enumerate(argv):
            if arg == flag and i + 1 < len(argv):
                return argv[i + 1]
            if arg.startswith(flag + "="):
                return arg[len(flag) + 1:]
    except Exception:
        pass
    return None


def _extract_K(argv: list[str]) -> str:
    """Extract MTP K (``num_speculative_tokens``) from the
    ``--speculative-config`` JSON arg. Returns "0" when the flag is
    absent (no spec-decode) or the JSON can't be parsed."""
    spec = _argv_value("--speculative-config", argv)
    if spec is None:
        return "0"
    try:
        cfg = json.loads(spec)
        if isinstance(cfg, dict):
            k = cfg.get("num_speculative_tokens")
            if isinstance(k, int):
                return str(k)
    except (ValueError, TypeError):
        # Fall through to regex — some shells over-quote the JSON.
        pass
    m = _SPEC_CONFIG_RE.search(spec)
    if m:
        return m.group(1)
    return "0"


def _extract_backend(argv: list[str]) -> str:
    """Extract attention backend from ``--attention-backend`` argv,
    falling back to the ``VLLM_ATTENTION_BACKEND`` env, then "default"."""
    val = _argv_value("--attention-backend", argv)
    if val:
        return val
    env_val = os.environ.get("VLLM_ATTENTION_BACKEND")
    if env_val:
        return env_val
    return "default"


def _extract_model(argv: list[str]) -> str:
    """Extract served-model-name from argv; fall back to the model
    path's basename; fall back to "unknown"."""
    name = _argv_value("--served-model-name", argv)
    if name:
        return name
    model = _argv_value("--model", argv)
    if model:
        # Use the model directory basename as a shorter label.
        return os.path.basename(model.rstrip("/")) or model
    return "unknown"


# ─── git / pin extraction ──────────────────────────────────────────────


def _extract_patch_hash() -> str:
    """Return the short SHA of HEAD in the Genesis patches repo.

    Looks at ``$GENESIS_REPO`` first (the operator-side env that the
    canonical launcher sets). Falls back to ``$GENESIS_PROJECT_ROOT``
    (older name). Returns "uncommitted" if neither is set or `git` is
    not available. Never raises.
    """
    repo = (
        os.environ.get("GENESIS_REPO")
        or os.environ.get("GENESIS_PROJECT_ROOT")
    )
    if not repo or not os.path.isdir(repo):
        return "uncommitted"
    try:
        r = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip() or "uncommitted"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "uncommitted"


def _extract_pin() -> str:
    """Return vllm.__version__ — the live pin the process is running."""
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown") or "unknown"
    except Exception:
        return "unknown"


# ─── Labels assembly ───────────────────────────────────────────────────


def _resolve_labels(argv: Optional[list[str]] = None) -> dict:
    """Assemble the full label dict. argv defaults to ``sys.argv``."""
    if argv is None:
        argv = list(sys.argv)
    return {
        "preset": os.environ.get("GENESIS_PRESET") or "unknown",
        "profile": os.environ.get("GENESIS_PROFILE") or "unknown",
        "workload_class": (
            os.environ.get("GENESIS_WORKLOAD_CLASS") or "unknown"
        ),
        "K": _extract_K(argv),
        "backend": _extract_backend(argv),
        "patch_hash": _extract_patch_hash(),
        "model": _extract_model(argv),
        "pin": _extract_pin(),
    }


# ─── Prometheus setup ──────────────────────────────────────────────────


def _setup_gauge() -> bool:
    """Idempotent registration of the labeled Gauge.

    Mirrors PN287 / PN288 pattern: walk REGISTRY._collector_to_names
    and unregister stale collectors before re-registering. Safe across
    worker spawns + hot reload.
    """
    global _prom_gauge
    try:
        from prometheus_client import REGISTRY, Gauge
    except ImportError:
        return False

    for collector in list(REGISTRY._collector_to_names):
        names = REGISTRY._collector_to_names.get(collector, [])
        if any(n.startswith(_METRIC_NAME) for n in names):
            try:
                REGISTRY.unregister(collector)
            except (KeyError, ValueError):
                pass

    try:
        _prom_gauge = Gauge(
            name=_METRIC_NAME,
            documentation=(
                "Canonical Genesis process-info gauge — always 1 with "
                "labels (preset, profile, workload_class, K, backend, "
                "patch_hash, model, pin). Join other metrics against "
                "this via PromQL `* on(instance) group_left(...)` to "
                "pivot vllm-builtin counters by Genesis operator "
                "metadata. See module docstring for query examples."
            ),
            labelnames=_LABEL_NAMES,
        )
        return True
    except (ValueError, AttributeError) as exc:
        log.warning(
            "[genesis_process_info] failed to register Gauge: %s. "
            "Metric not exposed.", exc,
        )
        return False


def apply(argv: Optional[list[str]] = None) -> tuple[str, str]:
    """Emit the genesis_process_info gauge with current labels.

    Idempotent: re-calling after a successful apply re-resolves labels
    (handy when the operator hot-reloads env) but doesn't re-register
    the metric. Returns the standard (status, reason) tuple shared
    with the Genesis dispatcher.
    """
    global _APPLIED

    if not _setup_gauge():
        return "skipped", (
            "prometheus_client not importable — "
            "genesis_process_info metric will not be exposed. "
            "Install prometheus_client OR ignore on torch-less env."
        )

    labels = _resolve_labels(argv)
    try:
        _prom_gauge.labels(**labels).set(1.0)
        _APPLIED = True
        log.info(
            "[genesis_process_info] registered — "
            "preset=%s profile=%s K=%s backend=%s patch=%s "
            "model=%s pin=%s",
            labels["preset"], labels["profile"], labels["K"],
            labels["backend"], labels["patch_hash"],
            labels["model"], labels["pin"],
        )
        return "applied", (
            f"genesis_process_info emitted; labels: "
            f"preset={labels['preset']!r} profile={labels['profile']!r} "
            f"K={labels['K']!r} backend={labels['backend']!r} "
            f"patch_hash={labels['patch_hash']!r} model={labels['model']!r} "
            f"pin={labels['pin']!r}"
        )
    except Exception as exc:
        return "failed", f"set(1.0) raised: {exc}"


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "apply",
    "is_applied",
    "_resolve_labels",
    "_extract_K",
    "_extract_backend",
    "_extract_model",
    "_extract_patch_hash",
    "_extract_pin",
    "_setup_gauge",
]
