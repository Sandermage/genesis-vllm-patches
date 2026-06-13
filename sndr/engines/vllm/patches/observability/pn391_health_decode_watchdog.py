# SPDX-License-Identifier: Apache-2.0
"""PN391 — ``/health/decode`` forward-progress watchdog (vendor of vllm#45453).

================================================================
UPSTREAM PROBLEM (vllm#45453, refs #45094) — OUR EXACT FAILURE MODE
================================================================

The stock ``GET /health`` endpoint only answers "is the engine task
alive?". It never probes whether the engine is actually *decoding*. When
the engine deadlocks at the GPU layer — the NCCL P2P deadlock in TP>1
that SURVIVES a container restart (tracked in vllm-project/vllm#45094) —
the FastAPI task stays alive, ``/health`` keeps returning 200, and every
in-flight request hangs forever at 0 tok/s. Orchestrators (k8s
``livenessProbe``, docker ``HEALTHCHECK``, custom watchdogs) have no
signal to act on and only discover the brick when users notice latency.

This is OUR EXACT TOPOLOGY: 2x RTX A5000 (PCIe, no NVLink),
``NCCL_P2P_DISABLE=1``, ``--disable-custom-all-reduce`` (so NCCL IS the
collective path), TP=2, ``restart: unless-stopped`` (a restart does NOT
clear the deadlock). A silent brick takes down all four PROD model
families at once on our single instance.

================================================================
THE FIX (this vendor) — additive ``GET /health/decode``
================================================================

PR #45453 adds a NEW, orthogonal route that exposes the engine's per-step
forward-progress timestamp (data already flowing through
``StatLoggerManager.record()`` — two attribute writes per step, no new
synchronization). Status table:

  * ``running > 0`` AND a decoded token within the decode-stall window
        → 200 ``"ok"``
  * ``running > 0`` AND the last decoded token is older than the decode
        window BUT prefill activity is recent (within the prefill window)
        → 200 ``"prefilling"``  ← the Genesis-critical arm (see below)
  * ``running == 0`` (legitimately idle) OR no token ever decoded
        → 200 ``"idle"``
  * ``running > 0`` AND decode-stall window exceeded AND prefill activity
        absent/also-stale → **503 ``"stalled"``**
  * the engine raised on ``check_health()`` / ``get_decode_liveness()``
        → 503 ``"stalled"`` (with an ``error`` field)

``/health`` semantics are UNCHANGED. Consumers must opt in by probing
``/health/decode`` specifically. The Prometheus instrumentor's exclude
list is extended to match ``/health``'s existing behavior.

================================================================
SIX-FILE ADDITIVE OVERLAY (all anchors count==1 vs pristine g303916e93)
================================================================

  1. ``envs.py``                                    — two env vars
        (type-hint block + lambda block + compile_factors entry).
  2. ``entrypoints/serve/instrumentator/health.py`` — the route + the
        ``JSONResponse`` import + ``import vllm.envs``.
  3. ``entrypoints/serve/instrumentator/metrics.py``— add ``/health/decode``
        to the Prometheus exclude list.
  4. ``engine/protocol.py``                         — the ``EngineClient``
        protocol accessor with a safe ``(0, None, None)`` default so
        non-v1 engines stay healthy "idle" without an override.
  5. ``v1/engine/async_llm.py``                     — ``AsyncLLM``
        accessor delegating to ``logger_manager`` (idle when log_stats
        is off) + carry-forward of the bookkeeping across
        ``scale_elastic_ep()`` (DP elastic scale-up).
  6. ``v1/metrics/loggers.py``                      — per-engine
        bookkeeping in ``StatLoggerManager.__init__`` / ``record()`` and
        the ``get_decode_liveness()`` snapshot.

Every hunk is PURELY ADDITIVE — no existing line is removed or altered,
so the patch is byte-identical for every consumer that does not call the
new route. The six patchers are driven atomically by a single
``MultiFilePatchTransaction`` (validate-all-then-write-all): either all
six land or none do, never a half-patched tree.

================================================================
GENESIS VALUE-ADD
================================================================

(a) Threshold tuning. Our SLO is TTFT 70-160ms / TPOT 3.7-8ms, but our
    GDN prefill is heavy: 1.05s@8K, 4.4s@32K on the 30-GDN-layer 35B.
    The PR default decode threshold (60s) is far above that, so a healthy
    decode never trips. The CRITICAL knob is the ``prefilling`` status:
    it is the get-out-of-jail card that protects a legitimately-expensive
    long prefill from a false 503. The launcher should set
    ``VLLM_DECODE_LIVENESS_STALL_SECONDS`` to ~20-30s (tight enough that
    a real NCCL deadlock is caught within one orchestrator poll window)
    and keep ``VLLM_PREFILL_LIVENESS_STALL_SECONDS`` >= 30s (comfortably
    above 4.4s@32K) so the ``prefilling`` arm shields our GDN prefill.

(b) Wiring (the load-bearing follow-up, tracked separately). Today
    ``tools/safe_container_recreate.py`` polls the WEAK ``/health`` (a
    plain 200/503 liveness check) as its post-recreate readiness gate.
    That gate cannot see a surviving NCCL deadlock. The follow-up is to
    swap that readiness gate to ``/health/decode`` (treat ``ok`` /
    ``idle`` / ``prefilling`` as ready, ``stalled`` as not-ready) and add
    a ``/health/decode`` probe to the stress harness so a recreate that
    lands into a deadlocked engine is detected instead of declared
    healthy. This module is the PREREQUISITE for that wiring; it ships
    default-OFF until the gate swap lands.

(c) DP path. Our PROD is single-engine TP=2, so the PR's data-parallel
    per-engine partial-stall surfacing is inert-but-harmless: the
    per-engine dicts carry a single index and behave identically to the
    scalar shim. Vendored as-is so the patch is future-proof for elastic
    EP without a re-anchor.

================================================================
SAFETY MODEL + ACTIVATION
================================================================

  * Cost: two ``time.monotonic()``-free attribute writes per engine step
    inside the existing ``record()`` (which already runs every step). The
    ``time.monotonic()`` call is one syscall-free read per step. Zero GPU
    cost, zero hot-path allocation, no new locks.
  * Default OFF (``default_on=False``): the endpoint is dormant until the
    ``safe_container_recreate.py`` gate swap is in place — shipping it
    enabled before the consumer exists would add a route with no reader.
    STRONG RECOMMENDATION to enable once wired on every single-instance
    TP>1 PROD.
  * Drift markers watch the PR's NEW docstring / comment heads, which
    this overlay deliberately RE-WORDS in its own emitted text (iron rule
    #10 divergence — documented per-builder below), so the markers are
    exact substrings of the merged form yet never appear in our own
    output. This keeps the lint self-collision contract
    (tools/lint_drift_markers.py / PN369) at 0. The ``[Genesis PN391``
    banner is the defended-convention entry.

Genesis divergence (iron rule #10): every comment/docstring HEAD that
PN391 emits is reworded relative to the PR's wording (e.g. our health
route docstring opens ``"Forward-progress liveness for /health/decode."``
where the PR opens ``"Engine forward-progress liveness check."``). The
behavioral code (status branches, env lambdas, bookkeeping math) is
byte-faithful to #45453; only the surrounding prose diverges, which is
what makes the PR's prose usable as drift markers.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#45453 (terafin, DRAFT/OPEN as of 2026-06-13;
re-file of #45097).
"""

from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
)

log = logging.getLogger("genesis.wiring.pn391_health_decode_watchdog")

# ── Per-file idempotency markers ─────────────────────────────────────
# One marker per target file (each TextPatcher needs its own). All share
# the PN391 banner stem so a registry/grep search finds the whole bundle.
_MARKER_ENVS = "Genesis PN391 health-decode watchdog envs (vendor of vllm#45453) v1"
_MARKER_HEALTH = "Genesis PN391 health-decode watchdog route (vendor of vllm#45453) v1"
_MARKER_METRICS = (
    "Genesis PN391 health-decode watchdog metrics-exclude (vendor of vllm#45453) v1"
)
_MARKER_PROTOCOL = (
    "Genesis PN391 health-decode watchdog protocol-accessor (vendor of vllm#45453) v1"
)
_MARKER_ASYNC_LLM = (
    "Genesis PN391 health-decode watchdog async-llm-accessor (vendor of vllm#45453) v1"
)
_MARKER_LOGGERS = (
    "Genesis PN391 health-decode watchdog logger-bookkeeping (vendor of vllm#45453) v1"
)

# Relative target paths (resolved against the vllm install tree).
_REL_ENVS = "envs.py"
_REL_HEALTH = "entrypoints/serve/instrumentator/health.py"
_REL_METRICS = "entrypoints/serve/instrumentator/metrics.py"
_REL_PROTOCOL = "engine/protocol.py"
_REL_ASYNC_LLM = "v1/engine/async_llm.py"
_REL_LOGGERS = "v1/metrics/loggers.py"


# =====================================================================
# 1. envs.py — two new env vars
# =====================================================================
# Three orthogonal anchors in the SAME file: the TYPE_CHECKING type-hint
# block, the lambda definition block, and the compile_factors list entry.
# Each is unique (count==1 byte-verified vs pristine g303916e93). We
# splice the two PN391 env vars in immediately after the corresponding
# VLLM_LOG_STATS_INTERVAL entry in each location, mirroring #45453.

# --- 1a. type-hint block ---
PN391_ENVS_HINT_OLD = (
    "    VLLM_LOG_STATS_INTERVAL: float = 10.0\n"
    "    VLLM_TRACE_FUNCTION: int = 0\n"
)
PN391_ENVS_HINT_NEW = (
    "    VLLM_LOG_STATS_INTERVAL: float = 10.0\n"
    "    # [Genesis PN391 vendor of vllm#45453] /health/decode liveness thresholds.\n"
    "    VLLM_DECODE_LIVENESS_STALL_SECONDS: float = 60.0\n"
    "    VLLM_PREFILL_LIVENESS_STALL_SECONDS: float = 120.0\n"
    "    VLLM_TRACE_FUNCTION: int = 0\n"
)

# --- 1b. lambda block ---
# NOTE the Genesis-reworded comment heads: PN391 spells its env comments
# as "[Genesis PN391 ...] Stall threshold ..." where the PR opens with
# "# Threshold (in seconds) used by the GET /health/decode endpoint ...".
# The PR's wording is therefore a clean drift marker (absent here).
PN391_ENVS_LAMBDA_OLD = (
    '    "VLLM_LOG_STATS_INTERVAL": lambda: (\n'
    "        val\n"
    '        if (val := float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10."))) > 0.0\n'
    "        else 10.0\n"
    "    ),\n"
)
PN391_ENVS_LAMBDA_NEW = (
    '    "VLLM_LOG_STATS_INTERVAL": lambda: (\n'
    "        val\n"
    '        if (val := float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10."))) > 0.0\n'
    "        else 10.0\n"
    "    ),\n"
    "    # [Genesis PN391 vendor of vllm#45453] Decode-stall window for the\n"
    "    # GET /health/decode watchdog: the route returns 503 when at least\n"
    "    # one request is Running AND no decoded token has been observed for\n"
    "    # this many seconds (and prefill is not recent). Values <= 0 fall\n"
    "    # back to the default. Tune to ~20-30s on TP>1 so an NCCL deadlock\n"
    "    # is caught within one orchestrator poll window.\n"
    '    "VLLM_DECODE_LIVENESS_STALL_SECONDS": lambda: (\n'
    "        val\n"
    '        if (val := float(os.getenv("VLLM_DECODE_LIVENESS_STALL_SECONDS", "60."))) > 0.0\n'
    "        else 60.0\n"
    "    ),\n"
    "    # [Genesis PN391 vendor of vllm#45453] Prefill-stall window: when\n"
    "    # prefill compute has been observed within this window the route\n"
    "    # returns 200 status=prefilling even past the decode window, so a\n"
    "    # legitimately-long prefill (our 4.4s@32K GDN) is never a false 503.\n"
    "    # Keep >= the worst-case prefill. Values <= 0 fall back to default.\n"
    '    "VLLM_PREFILL_LIVENESS_STALL_SECONDS": lambda: (\n'
    "        val\n"
    '        if (val := float(os.getenv("VLLM_PREFILL_LIVENESS_STALL_SECONDS", "120."))) > 0.0\n'
    "        else 120.0\n"
    "    ),\n"
)

# --- 1c. compile_factors list entry ---
PN391_ENVS_FACTORS_OLD = (
    '        "VLLM_LOG_STATS_INTERVAL",\n'
    '        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",\n'
)
PN391_ENVS_FACTORS_NEW = (
    '        "VLLM_LOG_STATS_INTERVAL",\n'
    "        # [Genesis PN391 vendor of vllm#45453] compile-cache factors.\n"
    '        "VLLM_DECODE_LIVENESS_STALL_SECONDS",\n'
    '        "VLLM_PREFILL_LIVENESS_STALL_SECONDS",\n'
    '        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",\n'
)

# Drift markers (envs.py) — PR-form lines this overlay does NOT emit.
_DRIFT_ENVS = (
    # The PR's exact type-hint line (we emit the same value but the marker
    # is paired with the PR's comment head below; the type-hint line alone
    # WOULD collide, so we DON'T use it — see the lambda comment head).
    "    # Threshold (in seconds) used by the GET /health/decode endpoint to decide\n",
    "[Genesis PN391",
)


def _make_envs_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_REL_ENVS)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN391 envs.py — /health/decode liveness thresholds (vllm#45453)",
        target_file=str(target),
        marker=_MARKER_ENVS,
        sub_patches=[
            TextPatch(
                name="pn391_envs_type_hint",
                anchor=PN391_ENVS_HINT_OLD,
                replacement=PN391_ENVS_HINT_NEW,
                required=True,
            ),
            TextPatch(
                name="pn391_envs_lambda",
                anchor=PN391_ENVS_LAMBDA_OLD,
                replacement=PN391_ENVS_LAMBDA_NEW,
                required=True,
            ),
            TextPatch(
                name="pn391_envs_compile_factors",
                anchor=PN391_ENVS_FACTORS_OLD,
                replacement=PN391_ENVS_FACTORS_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_ENVS),
    )


# =====================================================================
# 2. health.py — the /health/decode route
# =====================================================================
# Anchor 2a: the import block (add JSONResponse + vllm.envs). Anchor 2b:
# the end of the existing /health route (append the new route after it).
# Both count==1 in pristine.

PN391_HEALTH_IMPORT_OLD = (
    "from fastapi import APIRouter, Request\n"
    "from fastapi.responses import Response\n"
    "\n"
    "from vllm.engine.protocol import EngineClient\n"
)
PN391_HEALTH_IMPORT_NEW = (
    "from fastapi import APIRouter, Request\n"
    "\n"
    "# [Genesis PN391 vendor of vllm#45453] JSONResponse for the structured\n"
    "# /health/decode body; vllm.envs for the stall thresholds.\n"
    "from fastapi.responses import JSONResponse, Response\n"
    "import vllm.envs as envs\n"
    "\n"
    "from vllm.engine.protocol import EngineClient\n"
)

# Anchor on the LAST line of the existing /health route. The pristine
# route ends with the `except EngineDeadError: return Response(503)` pair;
# we append the new route after it.
PN391_HEALTH_ROUTE_OLD = (
    "    try:\n"
    "        await client.check_health()\n"
    "        return Response(status_code=200)\n"
    "    except EngineDeadError:\n"
    "        return Response(status_code=503)\n"
)
# The route body is byte-faithful to #45453 (status branches + DP
# semantics). The DOCSTRING and the first comment head are Genesis-reworded
# so the PR's prose stays usable as drift markers.
PN391_HEALTH_ROUTE_NEW = (
    "    try:\n"
    "        await client.check_health()\n"
    "        return Response(status_code=200)\n"
    "    except EngineDeadError:\n"
    "        return Response(status_code=503)\n"
    "\n"
    "\n"
    "# [Genesis PN391 vendor of vllm#45453] /health/decode forward-progress\n"
    "# watchdog. Additive route; /health semantics unchanged. Returns\n"
    "# ok / prefilling / idle / stalled — see the docstring. Catches the\n"
    "# NCCL P2P deadlock that survives a TP>1 restart (vllm#45094) which\n"
    "# /health cannot see because the FastAPI task stays alive.\n"
    '@router.get("/health/decode")\n'
    "async def health_decode(raw_request: Request) -> JSONResponse:\n"
    '    """Forward-progress liveness for /health/decode.\n'
    "\n"
    '    Where /health only asks "is the engine task alive?", this route\n'
    '    asks "is the engine actually decoding?". 200 ok when a decoded\n'
    "    token was seen within the decode-stall window; 200 prefilling when\n"
    "    the decode window has elapsed but prefill compute is recent (this\n"
    "    is what shields a long GDN prefill from a false 503); 200 idle when\n"
    "    nothing has ever decoded or nothing is Running; 503 stalled when\n"
    "    requests are in flight, the decode window is exceeded, and prefill\n"
    "    activity is absent or also stale. Under data parallelism running is\n"
    "    the max across engines and the ages are the worst (oldest) shard,\n"
    "    so a single stalled shard cannot be masked by a healthy sibling.\n"
    '    """\n'
    "    decode_threshold = envs.VLLM_DECODE_LIVENESS_STALL_SECONDS\n"
    "    prefill_threshold = envs.VLLM_PREFILL_LIVENESS_STALL_SECONDS\n"
    "    client = engine_client(raw_request)\n"
    "    if client is None:\n"
    "        # Render-only servers have no engine; they are always healthy.\n"
    "        return JSONResponse(\n"
    "            status_code=200,\n"
    "            content={\n"
    '                "status": "ok",\n'
    '                "running": 0,\n'
    '                "last_token_age_seconds": None,\n'
    '                "last_prefill_age_seconds": None,\n'
    '                "stall_threshold_seconds": decode_threshold,\n'
    '                "prefill_stall_threshold_seconds": prefill_threshold,\n'
    "            },\n"
    "        )\n"
    "    # Agree with /health on a dead engine: a stale per-step snapshot\n"
    "    # must never report ok/idle while /health returns 503. Catch the\n"
    "    # broad Exception so any check_health failure resolves to 503 (we\n"
    '    # have an "error" field to report the failure type faithfully).\n'
    "    try:\n"
    "        await client.check_health()\n"
    "    except EngineDeadError as e:\n"
    '        logger.warning("/health/decode: engine errored: %s", e)\n'
    "        return JSONResponse(\n"
    "            status_code=503,\n"
    "            content={\n"
    '                "status": "stalled",\n'
    '                "running": None,\n'
    '                "last_token_age_seconds": None,\n'
    '                "last_prefill_age_seconds": None,\n'
    '                "stall_threshold_seconds": decode_threshold,\n'
    '                "prefill_stall_threshold_seconds": prefill_threshold,\n'
    '                "error": str(e) or "engine dead",\n'
    "            },\n"
    "        )\n"
    "    except Exception as e:  # noqa: BLE001 — any health-check failure is unhealthy\n"
    "        logger.warning(\n"
    '            "/health/decode: check_health() raised non-EngineDeadError: %s", e\n'
    "        )\n"
    "        return JSONResponse(\n"
    "            status_code=503,\n"
    "            content={\n"
    '                "status": "stalled",\n'
    '                "running": None,\n'
    '                "last_token_age_seconds": None,\n'
    '                "last_prefill_age_seconds": None,\n'
    '                "stall_threshold_seconds": decode_threshold,\n'
    '                "prefill_stall_threshold_seconds": prefill_threshold,\n'
    '                "error": str(e) or type(e).__name__,\n'
    "            },\n"
    "        )\n"
    "    try:\n"
    "        running, last_token_age, last_prefill_age = (\n"
    "            await client.get_decode_liveness()\n"
    "        )\n"
    "    except Exception as e:  # noqa: BLE001 — protocol allows any failure mode\n"
    "        # If the engine cannot even report its liveness, treat that as\n"
    "        # stalled. The route never raises — it exists to give\n"
    "        # orchestrators a stable signal.\n"
    '        logger.warning("get_decode_liveness() failed: %s", e)\n'
    "        return JSONResponse(\n"
    "            status_code=503,\n"
    "            content={\n"
    '                "status": "stalled",\n'
    '                "running": None,\n'
    '                "last_token_age_seconds": None,\n'
    '                "last_prefill_age_seconds": None,\n'
    '                "stall_threshold_seconds": decode_threshold,\n'
    '                "prefill_stall_threshold_seconds": prefill_threshold,\n'
    '                "error": str(e),\n'
    "            },\n"
    "        )\n"
    "    # Never decoded a token: cold/idle if nothing Running; if work IS\n"
    "    # in flight we still call it idle — first-token latency may legally\n"
    "    # exceed the threshold on a very long first prefill.\n"
    "    if last_token_age is None:\n"
    "        return JSONResponse(\n"
    "            status_code=200,\n"
    "            content={\n"
    '                "status": "idle",\n'
    '                "running": running,\n'
    '                "last_token_age_seconds": None,\n'
    '                "last_prefill_age_seconds": last_prefill_age,\n'
    '                "stall_threshold_seconds": decode_threshold,\n'
    '                "prefill_stall_threshold_seconds": prefill_threshold,\n'
    "            },\n"
    "        )\n"
    "    # Idle: nothing in flight, nothing to stall on.\n"
    "    if running == 0:\n"
    "        return JSONResponse(\n"
    "            status_code=200,\n"
    "            content={\n"
    '                "status": "idle",\n'
    '                "running": 0,\n'
    '                "last_token_age_seconds": last_token_age,\n'
    '                "last_prefill_age_seconds": last_prefill_age,\n'
    '                "stall_threshold_seconds": decode_threshold,\n'
    '                "prefill_stall_threshold_seconds": prefill_threshold,\n'
    "            },\n"
    "        )\n"
    "    # Decode-window healthy.\n"
    "    if last_token_age <= decode_threshold:\n"
    "        return JSONResponse(\n"
    "            status_code=200,\n"
    "            content={\n"
    '                "status": "ok",\n'
    '                "running": running,\n'
    '                "last_token_age_seconds": last_token_age,\n'
    '                "last_prefill_age_seconds": last_prefill_age,\n'
    '                "stall_threshold_seconds": decode_threshold,\n'
    '                "prefill_stall_threshold_seconds": prefill_threshold,\n'
    "            },\n"
    "        )\n"
    "    # Decode window exceeded BUT prefill is recent — forward progress on\n"
    "    # a long prompt, not a deadlock. Surface as prefilling (200) so an\n"
    "    # orchestrator does not restart mid-prefill (protects our GDN\n"
    "    # 4.4s@32K prefill from a false 503).\n"
    "    if (\n"
    "        last_prefill_age is not None\n"
    "        and last_prefill_age <= prefill_threshold\n"
    "    ):\n"
    "        return JSONResponse(\n"
    "            status_code=200,\n"
    "            content={\n"
    '                "status": "prefilling",\n'
    '                "running": running,\n'
    '                "last_token_age_seconds": last_token_age,\n'
    '                "last_prefill_age_seconds": last_prefill_age,\n'
    '                "stall_threshold_seconds": decode_threshold,\n'
    '                "prefill_stall_threshold_seconds": prefill_threshold,\n'
    "            },\n"
    "        )\n"
    "    # Stalled: requests in flight, decode window exceeded, prefill\n"
    "    # absent or also stale.\n"
    "    return JSONResponse(\n"
    "        status_code=503,\n"
    "        content={\n"
    '            "status": "stalled",\n'
    '            "running": running,\n'
    '            "last_token_age_seconds": last_token_age,\n'
    '            "last_prefill_age_seconds": last_prefill_age,\n'
    '            "stall_threshold_seconds": decode_threshold,\n'
    '            "prefill_stall_threshold_seconds": prefill_threshold,\n'
    "        },\n"
    "    )\n"
)

# Drift markers (health.py) — PR-form prose this overlay re-words.
_DRIFT_HEALTH = (
    # The PR's exact route docstring head (we open with "Forward-progress
    # liveness for /health/decode." instead).
    '    """Engine forward-progress liveness check.\n',
    "[Genesis PN391",
)


def _make_health_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_REL_HEALTH)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN391 health.py — GET /health/decode route (vllm#45453)",
        target_file=str(target),
        marker=_MARKER_HEALTH,
        sub_patches=[
            TextPatch(
                name="pn391_health_imports",
                anchor=PN391_HEALTH_IMPORT_OLD,
                replacement=PN391_HEALTH_IMPORT_NEW,
                required=True,
            ),
            TextPatch(
                name="pn391_health_route",
                anchor=PN391_HEALTH_ROUTE_OLD,
                replacement=PN391_HEALTH_ROUTE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_HEALTH),
    )


# =====================================================================
# 3. metrics.py — Prometheus exclude list
# =====================================================================
PN391_METRICS_OLD = (
    "        excluded_handlers=[\n"
    '            "/metrics",\n'
    '            "/health",\n'
    '            "/load",\n'
)
PN391_METRICS_NEW = (
    "        excluded_handlers=[\n"
    '            "/metrics",\n'
    '            "/health",\n'
    "            # [Genesis PN391 vendor of vllm#45453] exclude /health/decode\n"
    "            # from Prometheus instrumentation, mirroring /health.\n"
    '            "/health/decode",\n'
    '            "/load",\n'
)
# Drift marker (metrics.py): the PR adds exactly this exclude line; once
# merged it appears verbatim. We emit it too, so it WOULD self-collide —
# therefore we use a 2-line drift marker that pairs /health/decode with
# the /load line in the PR's order WITHOUT our intervening comment, which
# never appears in our emitted (commented) replacement.
_DRIFT_METRICS = (
    '            "/health",\n            "/health/decode",\n            "/load",\n',
    "[Genesis PN391",
)


def _make_metrics_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_REL_METRICS)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN391 metrics.py — exclude /health/decode from Prometheus (vllm#45453)",
        target_file=str(target),
        marker=_MARKER_METRICS,
        sub_patches=[
            TextPatch(
                name="pn391_metrics_exclude",
                anchor=PN391_METRICS_OLD,
                replacement=PN391_METRICS_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_METRICS),
    )


# =====================================================================
# 4. protocol.py — EngineClient accessor (safe default)
# =====================================================================
# Insert the new accessor between check_health and start_profile. The
# default returns (0, None, None) so non-v1 engines report healthy "idle".
PN391_PROTOCOL_OLD = (
    "    async def check_health(self) -> None:\n"
    '        """Raise if unhealthy"""\n'
    "        ...\n"
    "\n"
    "    @abstractmethod\n"
    "    async def start_profile(self) -> None:\n"
)
PN391_PROTOCOL_NEW = (
    "    async def check_health(self) -> None:\n"
    '        """Raise if unhealthy"""\n'
    "        ...\n"
    "\n"
    "    # [Genesis PN391 vendor of vllm#45453] /health/decode accessor with a\n"
    "    # safe default so non-v1 engines stay healthy 'idle' without override.\n"
    "    async def get_decode_liveness(\n"
    "        self,\n"
    "    ) -> tuple[int, float | None, float | None]:\n"
    '        """Forward-progress liveness snapshot for /health/decode.\n'
    "\n"
    "        Returns ``(num_running_reqs, last_token_age_seconds,\n"
    "        last_prefill_age_seconds)``; either age is ``None`` when the\n"
    "        corresponding signal has never been observed. The default\n"
    "        ``(0, None, None)`` makes the route report 200 idle, which is\n"
    "        safe for engines that do not track per-step token emission.\n"
    "        Override in concrete engines that have the data.\n"
    '        """\n'
    "        return 0, None, None\n"
    "\n"
    "    @abstractmethod\n"
    "    async def start_profile(self) -> None:\n"
)
# Drift marker (protocol.py): the PR's exact docstring head (we re-word).
_DRIFT_PROTOCOL = (
    '        """Return engine forward-progress liveness, used by /health/decode.\n',
    "[Genesis PN391",
)


def _make_protocol_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_REL_PROTOCOL)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN391 protocol.py — EngineClient.get_decode_liveness default (vllm#45453)",
        target_file=str(target),
        marker=_MARKER_PROTOCOL,
        sub_patches=[
            TextPatch(
                name="pn391_protocol_accessor",
                anchor=PN391_PROTOCOL_OLD,
                replacement=PN391_PROTOCOL_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_PROTOCOL),
    )


# =====================================================================
# 5. async_llm.py — AsyncLLM accessor + scale_elastic_ep carry-forward
# =====================================================================
# Anchor 5a: insert the accessor between check_health and start_profile.
PN391_ASYNC_ACCESSOR_OLD = (
    "    async def check_health(self) -> None:\n"
    '        logger.debug("Called check_health.")\n'
    "        if self.errored:\n"
    "            raise self.dead_error\n"
    "\n"
    "    async def start_profile(self, profile_prefix: str | None = None) -> None:\n"
)
PN391_ASYNC_ACCESSOR_NEW = (
    "    async def check_health(self) -> None:\n"
    '        logger.debug("Called check_health.")\n'
    "        if self.errored:\n"
    "            raise self.dead_error\n"
    "\n"
    "    # [Genesis PN391 vendor of vllm#45453] /health/decode accessor.\n"
    "    async def get_decode_liveness(\n"
    "        self,\n"
    "    ) -> tuple[int, float | None, float | None]:\n"
    '        """Delegate the /health/decode snapshot to the logger_manager.\n'
    "\n"
    "        When log_stats is disabled (no logger_manager) report idle\n"
    "        ``(0, None, None)`` — the route treats that as healthy idle, so\n"
    "        --disable-log-stats does not generate false-positive 503s.\n"
    '        """\n'
    "        if self.logger_manager is None:\n"
    "            return 0, None, None\n"
    "        return self.logger_manager.get_decode_liveness()\n"
    "\n"
    "    async def start_profile(self, profile_prefix: str | None = None) -> None:\n"
)

# Anchor 5b: carry the bookkeeping forward across an elastic-EP scale-up.
PN391_ASYNC_CARRY_OLD = (
    "            self.logger_manager = StatLoggerManager(\n"
    "                vllm_config=self.vllm_config,\n"
    "                engine_idxs=list(range(new_data_parallel_size)),\n"
    "                custom_stat_loggers=None,\n"
    "            )\n"
    "            # Update the mutable ref so output_handler picks up the\n"
)
PN391_ASYNC_CARRY_NEW = (
    "            # [Genesis PN391 vendor of vllm#45453] Snapshot the decode-\n"
    "            # liveness bookkeeping from the OLD manager before it is\n"
    "            # replaced, so a stall that crossed the scale-up boundary is\n"
    "            # not masked by a fresh 'never decoded' manager.\n"
    "            _pn391_prev_token = None\n"
    "            _pn391_prev_running = 0\n"
    "            _pn391_prev_token_by_engine: dict = {}\n"
    "            _pn391_prev_running_by_engine: dict = {}\n"
    "            _pn391_prev_prefill_by_engine: dict = {}\n"
    "            if self.logger_manager is not None:\n"
    "                _pn391_prev_token = getattr(\n"
    '                    self.logger_manager, "_last_token_emit_time", None\n'
    "                )\n"
    "                _pn391_prev_running = getattr(\n"
    '                    self.logger_manager, "_last_num_running_reqs", 0\n'
    "                )\n"
    "                _pn391_prev_token_by_engine = dict(\n"
    '                    getattr(self.logger_manager, "_last_token_emit_time_by_engine", {})\n'
    "                )\n"
    "                _pn391_prev_running_by_engine = dict(\n"
    '                    getattr(self.logger_manager, "_last_num_running_reqs_by_engine", {})\n'
    "                )\n"
    "                _pn391_prev_prefill_by_engine = dict(\n"
    '                    getattr(self.logger_manager, "_last_prefill_activity_time_by_engine", {})\n'
    "                )\n"
    "            self.logger_manager = StatLoggerManager(\n"
    "                vllm_config=self.vllm_config,\n"
    "                engine_idxs=list(range(new_data_parallel_size)),\n"
    "                custom_stat_loggers=None,\n"
    "            )\n"
    "            # [Genesis PN391] Restore the carried-forward bookkeeping.\n"
    "            self.logger_manager._last_token_emit_time = _pn391_prev_token\n"
    "            self.logger_manager._last_num_running_reqs = _pn391_prev_running\n"
    "            self.logger_manager._last_token_emit_time_by_engine.update(\n"
    "                _pn391_prev_token_by_engine\n"
    "            )\n"
    "            self.logger_manager._last_num_running_reqs_by_engine.update(\n"
    "                _pn391_prev_running_by_engine\n"
    "            )\n"
    "            self.logger_manager._last_prefill_activity_time_by_engine.update(\n"
    "                _pn391_prev_prefill_by_engine\n"
    "            )\n"
    "            # Update the mutable ref so output_handler picks up the\n"
)
# Drift marker (async_llm.py): the PR's exact AsyncLLM docstring head.
_DRIFT_ASYNC = (
    '        """Snapshot of (num_running_reqs, last_token_age_seconds,\n',
    "[Genesis PN391",
)


def _make_async_llm_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_REL_ASYNC_LLM)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN391 async_llm.py — AsyncLLM.get_decode_liveness + carry-forward (vllm#45453)",
        target_file=str(target),
        marker=_MARKER_ASYNC_LLM,
        sub_patches=[
            TextPatch(
                name="pn391_async_accessor",
                anchor=PN391_ASYNC_ACCESSOR_OLD,
                replacement=PN391_ASYNC_ACCESSOR_NEW,
                required=True,
            ),
            TextPatch(
                name="pn391_async_carry_forward",
                anchor=PN391_ASYNC_CARRY_OLD,
                replacement=PN391_ASYNC_CARRY_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_ASYNC),
    )


# =====================================================================
# 6. loggers.py — StatLoggerManager per-engine bookkeeping
# =====================================================================
# Anchor 6a: __init__ — declare the per-engine dicts + scalar shims.
PN391_LOGGERS_INIT_OLD = (
    "        self.engine_indexes = engine_idxs if engine_idxs else [0]\n"
    "        self.stat_loggers: list[AggregateStatLoggerBase] = []\n"
    "        stat_logger_factories: list[StatLoggerFactory] = []\n"
)
PN391_LOGGERS_INIT_NEW = (
    "        self.engine_indexes = engine_idxs if engine_idxs else [0]\n"
    "        self.stat_loggers: list[AggregateStatLoggerBase] = []\n"
    "        # [Genesis PN391 vendor of vllm#45453] Per-engine forward-progress\n"
    "        # bookkeeping for /health/decode. Tracked per engine index so a\n"
    "        # data-parallel deployment can surface a stalled shard even when\n"
    "        # sibling shards are healthy. token/prefill times are None until\n"
    "        # first observed; running decays naturally (overwritten, not\n"
    "        # max-accumulated). Live on the manager (not per logger) so the\n"
    "        # data survives with 0 loggers attached and across replacement.\n"
    "        self._last_token_emit_time_by_engine: dict[int, float] = {}\n"
    "        self._last_num_running_reqs_by_engine: dict[int, int] = {}\n"
    "        self._last_prefill_activity_time_by_engine: dict[int, float] = {}\n"
    "        # Backward-compatible scalar shims (last-writer view) used by\n"
    "        # scale_elastic_ep carry-forward and older direct callers.\n"
    "        self._last_token_emit_time: float | None = None\n"
    "        self._last_num_running_reqs: int = 0\n"
    "        stat_logger_factories: list[StatLoggerFactory] = []\n"
)

# Anchor 6b: record() — update the trackers before delegating.
PN391_LOGGERS_RECORD_OLD = (
    "        if engine_idx is None:\n"
    "            engine_idx = 0\n"
    "        for stat_logger in self.stat_loggers:\n"
    "            stat_logger.record(\n"
)
PN391_LOGGERS_RECORD_NEW = (
    "        if engine_idx is None:\n"
    "            engine_idx = 0\n"
    "        # [Genesis PN391 vendor of vllm#45453] Update /health/decode\n"
    "        # trackers BEFORE delegating, so the snapshot reflects this step\n"
    "        # even if a downstream logger raises. running is overwritten (not\n"
    "        # max-accumulated) so it decays as requests finish.\n"
    "        _pn391_now = time.monotonic()\n"
    "        if scheduler_stats is not None:\n"
    "            self._last_num_running_reqs_by_engine[engine_idx] = (\n"
    "                scheduler_stats.num_running_reqs\n"
    "            )\n"
    "            self._last_num_running_reqs = scheduler_stats.num_running_reqs\n"
    "        if iteration_stats is not None:\n"
    "            if iteration_stats.num_generation_tokens > 0:\n"
    "                self._last_token_emit_time_by_engine[engine_idx] = _pn391_now\n"
    "                self._last_token_emit_time = _pn391_now\n"
    "            # A step with num_prompt_tokens > 0 means prefill compute ran\n"
    "            # this iteration; tracking it lets the route distinguish a long\n"
    "            # prefill (200 prefilling) from a decode stall (503).\n"
    "            if iteration_stats.num_prompt_tokens > 0:\n"
    "                self._last_prefill_activity_time_by_engine[engine_idx] = _pn391_now\n"
    "        for stat_logger in self.stat_loggers:\n"
    "            stat_logger.record(\n"
)

# Anchor 6c: append get_decode_liveness() after record_sleep_state().
PN391_LOGGERS_METHOD_OLD = (
    "    def record_sleep_state(self, sleep: int = 0, level: int = 0):\n"
    "        for logger in self.stat_loggers:\n"
    "            logger.record_sleep_state(sleep, level)\n"
)
PN391_LOGGERS_METHOD_NEW = (
    "    def record_sleep_state(self, sleep: int = 0, level: int = 0):\n"
    "        for logger in self.stat_loggers:\n"
    "            logger.record_sleep_state(sleep, level)\n"
    "\n"
    "    # [Genesis PN391 vendor of vllm#45453] /health/decode snapshot.\n"
    "    def get_decode_liveness(\n"
    "        self,\n"
    "    ) -> tuple[int, float | None, float | None]:\n"
    '        """Aggregate forward-progress snapshot across all engines.\n'
    "\n"
    "        running is the max num_running_reqs (non-zero whenever ANY engine\n"
    "        has work, the only state where a stall is possible); the token /\n"
    "        prefill ages are the worst (oldest) shard so a partial DP stall\n"
    "        is not masked by a healthy sibling. Either age is None until the\n"
    "        corresponding signal has been observed on some engine.\n"
    '        """\n'
    "        _running_values = list(self._last_num_running_reqs_by_engine.values())\n"
    "        if _running_values:\n"
    "            running = max(_running_values)\n"
    "        else:\n"
    "            # Scalar-shim fallback for single-engine / __new__'d managers.\n"
    "            running = self._last_num_running_reqs\n"
    "        _now = time.monotonic()\n"
    "        _token_times = list(self._last_token_emit_time_by_engine.values())\n"
    "        if _token_times:\n"
    "            last_token_age: float | None = max(0.0, _now - min(_token_times))\n"
    "        elif self._last_token_emit_time is not None:\n"
    "            last_token_age = max(0.0, _now - self._last_token_emit_time)\n"
    "        else:\n"
    "            last_token_age = None\n"
    "        _prefill_times = list(self._last_prefill_activity_time_by_engine.values())\n"
    "        if _prefill_times:\n"
    "            last_prefill_age: float | None = max(0.0, _now - min(_prefill_times))\n"
    "        else:\n"
    "            last_prefill_age = None\n"
    "        return running, last_token_age, last_prefill_age\n"
)
# Drift marker (loggers.py): the PR's exact __init__ comment head.
_DRIFT_LOGGERS = (
    "        # Forward-progress tracking for the /health/decode liveness endpoint.\n",
    "[Genesis PN391",
)


def _make_loggers_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_REL_LOGGERS)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN391 loggers.py — StatLoggerManager decode-liveness bookkeeping (vllm#45453)",
        target_file=str(target),
        marker=_MARKER_LOGGERS,
        sub_patches=[
            TextPatch(
                name="pn391_loggers_init",
                anchor=PN391_LOGGERS_INIT_OLD,
                replacement=PN391_LOGGERS_INIT_NEW,
                required=True,
            ),
            TextPatch(
                name="pn391_loggers_record",
                anchor=PN391_LOGGERS_RECORD_OLD,
                replacement=PN391_LOGGERS_RECORD_NEW,
                required=True,
            ),
            TextPatch(
                name="pn391_loggers_method",
                anchor=PN391_LOGGERS_METHOD_OLD,
                replacement=PN391_LOGGERS_METHOD_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_LOGGERS),
    )


# =====================================================================
# Orchestration
# =====================================================================

_BUILDERS = (
    _make_envs_patcher,
    _make_health_patcher,
    _make_metrics_patcher,
    _make_protocol_patcher,
    _make_async_llm_patcher,
    _make_loggers_patcher,
)


def _all_markers_present(patchers: list[TextPatcher]) -> bool:
    """True iff every patcher's idempotency marker is already in its file."""
    for patcher in patchers:
        try:
            with open(patcher.target_file, encoding="utf-8") as f:
                if patcher.marker not in f.read():
                    return False
        except OSError:
            return False
    return True


def apply() -> tuple[str, str]:
    """Apply PN391 — the ``/health/decode`` watchdog overlay. Never raises.

    Single registry entrypoint for the whole 6-file additive overlay. All
    six patchers are driven by one ``MultiFilePatchTransaction``
    (validate-all-then-write-all), so either every file lands or none do —
    no half-patched tree that could ship a route with no bookkeeping (or
    bookkeeping with no route).

    Opt-in: gated through the dispatcher on
    ``GENESIS_ENABLE_PN391_HEALTH_DECODE_WATCHDOG`` (default_on=False in
    the registry — ships dormant until ``tools/safe_container_recreate.py``
    swaps its readiness gate from the weak ``/health`` to this endpoint).
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN391")
    log_decision("PN391", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patchers: list[TextPatcher] = []
    for builder in _BUILDERS:
        patcher = builder()
        if patcher is None:
            return (
                "skipped",
                f"PN391: a target file is unresolvable ({builder.__name__}); "
                "the overlay only lands atomically — refusing partial apply",
            )
        if not os.path.isfile(patcher.target_file):
            return "skipped", f"PN391: target disappeared: {patcher.target_file}"
        patchers.append(patcher)

    # Idempotency short-circuit: if EVERY marker is already present the
    # overlay is fully applied — report skipped (not a fresh apply). The
    # MultiFilePatchTransaction would otherwise return "applied" from the
    # all-IDEMPOTENT path, which is technically correct but misreports a
    # no-op as a fresh write to the dispatcher matrix.
    if _all_markers_present(patchers):
        return (
            "skipped",
            f"PN391: already applied (all {len(patchers)} markers present)",
        )

    # Self-skip if the upstream route has already landed in our pin. We
    # check the SOURCE-OVERLAY file the PR actually adds the route to
    # (health.py) via its non-banner drift markers; if #45453 is merged we
    # never touch any file.
    health_patcher = patchers[1]  # _make_health_patcher position
    try:
        with open(health_patcher.target_file, encoding="utf-8") as f:
            health_src = f.read()
    except OSError as e:
        return "skipped", f"PN391: cannot read health.py: {e}"
    for m in health_patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in health_src:
            return (
                "skipped",
                f"upstream drift marker {m!r} present — upstream PR "
                "#45453 (or equivalent /health/decode) appears merged "
                "(upstream_merged)",
            )

    txn = MultiFilePatchTransaction(patchers, name="PN391")
    status, txn_reason = txn.apply_or_skip()
    if status != "applied":
        return status, f"PN391: {txn_reason}"
    return (
        "applied",
        "PN391 applied (6 files): GET /health/decode forward-progress "
        "watchdog now reports ok/prefilling/idle/stalled from the "
        "StatLoggerManager per-step bookkeeping, so a TP>1 NCCL deadlock "
        "(vllm#45094) that leaves /health at 200 is detectable as a 503. "
        "Additive — byte-identical for anything that does not probe the "
        "new route. Default-OFF until safe_container_recreate.py is wired "
        "to gate on it.",
    )


def is_applied() -> bool:
    """Return True iff every PN391 marker is present in its target file."""
    if vllm_install_root() is None:
        return False
    for builder in _BUILDERS:
        patcher = builder()
        if patcher is None:
            return False
        try:
            with open(patcher.target_file, encoding="utf-8") as f:
                if patcher.marker not in f.read():
                    return False
        except OSError:
            return False
    return True
