# SPDX-License-Identifier: Apache-2.0
"""PN391 — /health/decode forward-progress watchdog (vendor of vllm#45453).

Contract pinned here (TDD, written before the implementation):

Upstream bug class (vllm#45453, refs #45094): the stock ``/health``
endpoint only asks "is the engine task alive?". When the engine deadlocks
at the GPU layer — notably the NCCL P2P deadlock in TP>1 that survives a
container restart — the FastAPI task stays alive, ``/health`` keeps
returning 200, and every in-flight request hangs forever. Orchestrators
have no signal to act on. This is OUR EXACT topology (2x A5000, PCIe,
``NCCL_P2P_DISABLE=1``, TP=2, ``restart: unless-stopped``).

PN391 vendors the additive ``GET /health/decode`` endpoint plus the
supporting per-step bookkeeping. It is a 6-file additive overlay driven
atomically by a single ``apply()`` via ``MultiFilePatchTransaction``:

  1. ``envs.py``                                   — two new env vars
  2. ``entrypoints/serve/instrumentator/health.py`` — the route
  3. ``entrypoints/serve/instrumentator/metrics.py``— Prometheus exclude
  4. ``engine/protocol.py``                        — protocol accessor
  5. ``v1/engine/async_llm.py``                    — AsyncLLM accessor
  6. ``v1/metrics/loggers.py``                     — StatLoggerManager state

All six are PURELY ADDITIVE (no existing behavior changed). The endpoint
returns ok / prefilling / idle / stalled per the PR's status table. The
``prefilling`` status (HIGH-(a) on the PR) is the Genesis-critical arm:
it protects our 4.4s@32K GDN prefill from a false 503.

Gated default-OFF on
``GENESIS_ENABLE_PN391_HEALTH_DECODE_WATCHDOG`` until wired into
``tools/safe_container_recreate.py`` (which currently polls the weak
``/health``).

Sub-contracts:
  1. Six ``_make_*_patcher()`` builders, each carrying exactly one
     required sub-patch (so the drift-marker lint discovers each).
  2. apply() drives all six atomically and reports "applied".
  3. Each patched file still compiles.
  4. Second apply() is idempotent (every patcher marker short-circuits).
  5. apply() self-skips on the merged form via the health.py drift marker
     (reason: upstream_merged) without touching any file.
  6. Drift markers do not collide with PN391's own replacement text or its
     Layer-6 marker line (tools/lint_drift_markers.py / PN369 contract)
     AND at least one marker per file is an exact substring of the merged
     form.
  7. Opt-in gate: with the dispatcher gate closed, apply() skips without
     touching any target.
  8. Pristine pin invariants (opportunistic): every anchor unique
     (count==1), every non-banner drift marker absent in the pristine
     tree.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Unit tests patch fresh tmp files; the Layer-0 file cache must never
# satisfy apply() from a previous run's state.
os.environ.setdefault("GENESIS_NO_PATCH_CACHE", "1")

from sndr.engines.vllm.patches.observability import (  # noqa: E402
    pn391_health_decode_watchdog as wd,
)

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm")

# Relative paths of the six target files inside the pin tree.
REL_ENVS = "envs.py"
REL_HEALTH = "entrypoints/serve/instrumentator/health.py"
REL_METRICS = "entrypoints/serve/instrumentator/metrics.py"
REL_PROTOCOL = "engine/protocol.py"
REL_ASYNC_LLM = "v1/engine/async_llm.py"
REL_LOGGERS = "v1/metrics/loggers.py"


# ── Fixtures: byte-faithful pin-form slices of each target ───────────

# Each fixture is a minimal but compile-able stand-in carrying the exact
# anchor block PN391 splices into, copied byte-for-byte from pin
# g303916e93. The builder under test must locate its anchor inside.

PIN_ENVS = (
    "# fake envs.py (pin g303916e93 form)\n"
    "import os\n"
    "\n"
    "if TYPE_CHECKING:\n"
    '    VLLM_LOGGING_COLOR: str = "auto"\n'
    "    NO_COLOR: bool = False\n"
    "    VLLM_LOG_STATS_INTERVAL: float = 10.0\n"
    "    VLLM_TRACE_FUNCTION: int = 0\n"
    "    VLLM_USE_FLASHINFER_SAMPLER: bool = True\n"
    "\n"
    "environment_variables = {\n"
    '    "VLLM_LOG_STATS_INTERVAL": lambda: (\n'
    "        val\n"
    '        if (val := float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10."))) > 0.0\n'
    "        else 10.0\n"
    "    ),\n"
    "    # Trace function calls\n"
    "    # If set to 1, vllm will trace function calls\n"
    "    # Useful for debugging\n"
    '    "VLLM_TRACE_FUNCTION": lambda: int(os.getenv("VLLM_TRACE_FUNCTION", "0")),\n'
    "}\n"
    "\n"
    "def compile_factors():\n"
    "    factors = [\n"
    '        "VLLM_LOGGING_COLOR",\n'
    '        "VLLM_LOG_STATS_INTERVAL",\n'
    '        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",\n'
    "    ]\n"
    "    return factors\n"
)

PIN_HEALTH = (
    "# fake health.py (pin g303916e93 form)\n"
    "from fastapi import APIRouter, Request\n"
    "from fastapi.responses import Response\n"
    "\n"
    "from vllm.engine.protocol import EngineClient\n"
    "from vllm.logger import init_logger\n"
    "from vllm.v1.engine.exceptions import EngineDeadError\n"
    "\n"
    "logger = init_logger(__name__)\n"
    "\n"
    "\n"
    "router = APIRouter()\n"
    "\n"
    "\n"
    "def engine_client(request: Request) -> EngineClient:\n"
    "    return request.app.state.engine_client\n"
    "\n"
    "\n"
    '@router.get("/health", response_class=Response)\n'
    "async def health(raw_request: Request) -> Response:\n"
    '    """Health check."""\n'
    "    client = engine_client(raw_request)\n"
    "    if client is None:\n"
    "        # Render-only servers have no engine; they are always healthy.\n"
    "        return Response(status_code=200)\n"
    "    try:\n"
    "        await client.check_health()\n"
    "        return Response(status_code=200)\n"
    "    except EngineDeadError:\n"
    "        return Response(status_code=503)\n"
)

PIN_METRICS = (
    "# fake metrics.py (pin g303916e93 form)\n"
    "def attach_router(app):\n"
    "    Instrumentator(\n"
    "        excluded_handlers=[\n"
    '            "/metrics",\n'
    '            "/health",\n'
    '            "/load",\n'
    '            "/ping",\n'
    '            "/version",\n'
    '            "/server_info",\n'
    "        ],\n"
    "        registry=registry,\n"
    "    )\n"
)

PIN_PROTOCOL = (
    "# fake protocol.py (pin g303916e93 form)\n"
    "class EngineClient:\n"
    "    @abstractmethod\n"
    "    async def check_health(self) -> None:\n"
    '        """Raise if unhealthy"""\n'
    "        ...\n"
    "\n"
    "    @abstractmethod\n"
    "    async def start_profile(self) -> None:\n"
    '        """Start profiling the engine"""\n'
    "        ...\n"
)

PIN_ASYNC_LLM = (
    "# fake async_llm.py (pin g303916e93 form)\n"
    "class AsyncLLM:\n"
    "    async def check_health(self) -> None:\n"
    '        logger.debug("Called check_health.")\n'
    "        if self.errored:\n"
    "            raise self.dead_error\n"
    "\n"
    "    async def start_profile(self, profile_prefix: str | None = None) -> None:\n"
    "        coros = [self.engine_core.profile_async(True, profile_prefix)]\n"
    "        if self.profiler is not None:\n"
    "            coros.append(asyncio.to_thread(self.profiler.start))\n"
    "        await asyncio.gather(*coros)\n"
    "\n"
    "    async def scale_elastic_ep(self, new_data_parallel_size, old_data_parallel_size):\n"
    "        if new_data_parallel_size > old_data_parallel_size and self.log_stats:\n"
    "            self.logger_manager = StatLoggerManager(\n"
    "                vllm_config=self.vllm_config,\n"
    "                engine_idxs=list(range(new_data_parallel_size)),\n"
    "                custom_stat_loggers=None,\n"
    "            )\n"
    "            # Update the mutable ref so output_handler picks up the\n"
    "            # new logger without creating a circular reference via self.\n"
    "            if hasattr(self, '_logger_ref'):\n"
    "                self._logger_ref[0] = self.logger_manager\n"
)

PIN_LOGGERS = (
    "# fake loggers.py (pin g303916e93 form)\n"
    "import time\n"
    "\n"
    "class StatLoggerManager:\n"
    "    def __init__(self, vllm_config, engine_idxs=None, custom_stat_loggers=None):\n"
    "        self.engine_indexes = engine_idxs if engine_idxs else [0]\n"
    "        self.stat_loggers: list[AggregateStatLoggerBase] = []\n"
    "        stat_logger_factories: list[StatLoggerFactory] = []\n"
    "        self._done = True\n"
    "\n"
    "    def record(\n"
    "        self,\n"
    "        scheduler_stats,\n"
    "        iteration_stats,\n"
    "        mm_cache_stats=None,\n"
    "        engine_idx=None,\n"
    "    ):\n"
    "        if engine_idx is None:\n"
    "            engine_idx = 0\n"
    "        for stat_logger in self.stat_loggers:\n"
    "            stat_logger.record(\n"
    "                scheduler_stats,\n"
    "                iteration_stats,\n"
    "                mm_cache_stats=mm_cache_stats,\n"
    "                engine_idx=engine_idx,\n"
    "            )\n"
    "\n"
    "    def record_sleep_state(self, sleep: int = 0, level: int = 0):\n"
    "        for logger in self.stat_loggers:\n"
    "            logger.record_sleep_state(sleep, level)\n"
)


_REL_TO_PIN = {
    REL_ENVS: PIN_ENVS,
    REL_HEALTH: PIN_HEALTH,
    REL_METRICS: PIN_METRICS,
    REL_PROTOCOL: PIN_PROTOCOL,
    REL_ASYNC_LLM: PIN_ASYNC_LLM,
    REL_LOGGERS: PIN_LOGGERS,
}


# ── Helper: install all six tmp targets + open the gate ──────────────


def _install_all(tmp_path, monkeypatch, *, forms=None):
    """Write all six tmp targets and redirect resolve_vllm_file to them.

    ``forms`` optionally overrides specific relative paths with merged-form
    text (used by the self-skip test).
    """
    forms = forms or {}
    written: dict[str, Path] = {}
    for rel, pin_text in _REL_TO_PIN.items():
        # Mirror the pin's nested layout so resolve_vllm_file(rel) maps to a
        # distinct file per relative path.
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(forms.get(rel, pin_text), encoding="utf-8")
        written[rel] = target

    def _resolve(rel):
        t = written.get(rel)
        return str(t) if t is not None else None

    monkeypatch.setattr(wd, "resolve_vllm_file", _resolve)
    monkeypatch.setattr(wd, "vllm_install_root", lambda: str(tmp_path))
    import sndr.dispatcher as dispatcher
    monkeypatch.setattr(
        dispatcher, "should_apply", lambda pid: (True, "test override")
    )
    return written


# ── Builder shape ────────────────────────────────────────────────────


class TestBuilderShape:
    def test_six_builders_exist(self):
        builders = [
            n for n in dir(wd)
            if n.startswith("_make") and n.endswith("patcher")
            and callable(getattr(wd, n))
        ]
        assert len(builders) == 6, builders

    def test_each_builder_has_required_subpatches(self, tmp_path, monkeypatch):
        _install_all(tmp_path, monkeypatch)
        for name in dir(wd):
            if name.startswith("_make") and name.endswith("patcher"):
                patcher = getattr(wd, name)()
                assert patcher is not None, name
                assert patcher.sub_patches, name
                # Every sub-patch in this additive overlay is required —
                # there is no soft-skippable hunk (all six files must land).
                assert all(sp.required for sp in patcher.sub_patches), name

    def test_module_documents_failure_mode_and_env_flag(self):
        doc = wd.__doc__ or ""
        assert "45453" in doc
        assert "NCCL" in doc
        src = Path(wd.__file__).read_text(encoding="utf-8")
        assert "GENESIS_ENABLE_PN391_HEALTH_DECODE_WATCHDOG" in src
        # Genesis value-add wiring note must be present.
        assert "safe_container_recreate" in src


# ── Atomic apply ─────────────────────────────────────────────────────


class TestApply:
    def test_apply_lands_all_six_and_compiles(self, tmp_path, monkeypatch):
        written = _install_all(tmp_path, monkeypatch)
        status, reason = wd.apply()
        assert status == "applied", reason

        envs_out = written[REL_ENVS].read_text(encoding="utf-8")
        assert "VLLM_DECODE_LIVENESS_STALL_SECONDS" in envs_out
        assert "VLLM_PREFILL_LIVENESS_STALL_SECONDS" in envs_out

        health_out = written[REL_HEALTH].read_text(encoding="utf-8")
        assert "/health/decode" in health_out
        assert "health_decode" in health_out
        assert '"status": "prefilling"' in health_out
        assert '"status": "stalled"' in health_out

        metrics_out = written[REL_METRICS].read_text(encoding="utf-8")
        assert metrics_out.count('"/health/decode"') == 1

        proto_out = written[REL_PROTOCOL].read_text(encoding="utf-8")
        assert "get_decode_liveness" in proto_out

        async_out = written[REL_ASYNC_LLM].read_text(encoding="utf-8")
        assert "get_decode_liveness" in async_out

        loggers_out = written[REL_LOGGERS].read_text(encoding="utf-8")
        assert "_last_token_emit_time_by_engine" in loggers_out
        assert "get_decode_liveness" in loggers_out

        # All six still parse.
        for rel, p in written.items():
            compile(p.read_text(encoding="utf-8"), str(p), "exec")

    def test_second_apply_idempotent(self, tmp_path, monkeypatch):
        _install_all(tmp_path, monkeypatch)
        first, first_reason = wd.apply()
        assert first == "applied", first_reason
        second, second_reason = wd.apply()
        assert second == "skipped"
        assert "already applied" in second_reason

    def test_apply_skips_when_gate_closed(self, tmp_path, monkeypatch):
        written = _install_all(tmp_path, monkeypatch)
        import sndr.dispatcher as dispatcher
        monkeypatch.setattr(
            dispatcher, "should_apply", lambda pid: (False, "opt-in: env unset")
        )
        status, _reason = wd.apply()
        assert status == "skipped"
        # No file touched.
        for rel, p in written.items():
            assert p.read_text(encoding="utf-8") == _REL_TO_PIN[rel]

    def test_self_skips_on_merged_health(self, tmp_path, monkeypatch):
        """If the merged /health/decode route is already in health.py the
        whole bundle self-skips without touching any file."""
        merged_health = PIN_HEALTH + (
            "\n\n"
            '@router.get("/health/decode")\n'
            "async def health_decode(raw_request):\n"
            '    """Engine forward-progress liveness check.\n'
            '    upstream merged form."""\n'
            "    return None\n"
        )
        written = _install_all(
            tmp_path, monkeypatch, forms={REL_HEALTH: merged_health}
        )
        status, reason = wd.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower()
        # No file touched (the merged-form health.py included).
        assert written[REL_HEALTH].read_text(encoding="utf-8") == merged_health
        assert written[REL_ENVS].read_text(encoding="utf-8") == PIN_ENVS


# ── Drift-marker self-collision (PN369 contract) ─────────────────────


class TestDriftMarkers:
    def test_markers_not_substring_of_own_emitted_text(
        self, tmp_path, monkeypatch
    ):
        _install_all(tmp_path, monkeypatch)
        for name in dir(wd):
            if not (name.startswith("_make") and name.endswith("patcher")):
                continue
            patcher = getattr(wd, name)()
            marker_line = f"# [Genesis wiring marker: {patcher.marker}]\n"
            assert patcher.upstream_drift_markers, f"{name}: markers required"
            for dm in patcher.upstream_drift_markers:
                if dm.startswith("[Genesis"):
                    continue
                for sp in patcher.sub_patches:
                    assert dm not in sp.replacement, (
                        f"{name}: drift marker {dm!r} collides with "
                        f"{sp.name} replacement (PN369 class)"
                    )
                assert dm not in marker_line

    def test_markers_fire_on_merged_form(self, tmp_path, monkeypatch):
        """Each builder must carry at least one non-banner drift marker
        that is an exact substring of the corresponding merged form."""
        _install_all(tmp_path, monkeypatch)
        merged = {
            wd._make_health_patcher: '    """Engine forward-progress liveness check.\n',
            wd._make_loggers_patcher: (
                "        # Forward-progress tracking for the /health/decode "
                "liveness endpoint.\n"
            ),
            wd._make_protocol_patcher: (
                '        """Return engine forward-progress liveness, used by '
                "/health/decode.\n"
            ),
            wd._make_async_llm_patcher: (
                '        """Snapshot of (num_running_reqs, '
                "last_token_age_seconds,\n"
            ),
            wd._make_envs_patcher: (
                "    # Threshold (in seconds) used by the GET /health/decode "
                "endpoint to decide\n"
            ),
            # The PR's merged exclude list adds /health/decode between
            # /health and /load with NO intervening comment — that exact
            # 3-line run is the metrics drift marker.
            wd._make_metrics_patcher: (
                '            "/metrics",\n'
                '            "/health",\n'
                '            "/health/decode",\n'
                '            "/load",\n'
            ),
        }
        for builder, merged_line in merged.items():
            patcher = builder()
            non_banner = [
                dm for dm in patcher.upstream_drift_markers
                if not dm.startswith("[Genesis")
            ]
            assert non_banner, f"{builder.__name__}: needs upstream-form marker"
            assert any(dm in merged_line for dm in non_banner), (
                f"{builder.__name__}: no drift marker matches merged line"
            )


# ── Pristine pin invariants (opportunistic) ──────────────────────────


@pytest.mark.skipif(
    not (PIN_TREE / REL_HEALTH).is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAgainstPristine:
    def test_every_anchor_unique_and_drift_absent(self, monkeypatch):
        # Resolve builders straight against the pristine tree.
        monkeypatch.setattr(
            wd, "resolve_vllm_file", lambda rel: str(PIN_TREE / rel)
        )
        monkeypatch.setattr(wd, "vllm_install_root", lambda: str(PIN_TREE))
        for name in dir(wd):
            if not (name.startswith("_make") and name.endswith("patcher")):
                continue
            patcher = getattr(wd, name)()
            assert patcher is not None, name
            src = Path(patcher.target_file).read_text(encoding="utf-8")
            for sp in patcher.sub_patches:
                assert src.count(sp.anchor) == 1, (
                    f"{name}/{sp.name}: anchor count != 1"
                )
                # The replacement's net-new text must not already be present.
                assert sp.replacement not in src, f"{name}: already patched?"
            for dm in patcher.upstream_drift_markers:
                if dm.startswith("[Genesis"):
                    continue
                assert dm not in src, f"{name}: drift marker {dm!r} in pristine"
