# SPDX-License-Identifier: Apache-2.0
"""TDD for PN65 — Genesis structured API access log.

Wave 6 v3 architecture (2026-05-09): pure logging-layer reformatter.
No HTTP middleware, no request-path overhead, no streaming concerns.
The ``GenesisAccessLogReformatter`` ``logging.Filter`` intercepts
``uvicorn.access`` INFO records and emits Genesis-API structured
lines via ``genesis.api`` logger.
"""
from __future__ import annotations

import logging

import pytest

from sndr.engines.vllm.patches.middleware.pn65_access_log import (
    GenesisAccessLogReformatter,
    _client_host_from_addr,
    _format_log_line,
    _quiet_paths,
    install_into_app,
)


# ─── Format helpers ─────────────────────────────────────────────────────


class TestQuietPaths:
    def test_default_quiet_paths_includes_health(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN65_QUIET_PATHS", raising=False)
        paths = _quiet_paths()
        assert "/health" in paths
        assert "/metrics" in paths

    def test_env_override_replaces_default(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN65_QUIET_PATHS", "/health,/admin")
        paths = _quiet_paths()
        assert "/health" in paths
        assert "/admin" in paths
        assert "/metrics" not in paths


class TestLogLineFormat:
    def test_simple_get(self):
        line = _format_log_line(
            "GET", "/v1/models", 401, "1.1", "192.0.2.10",
        )
        assert "[Genesis-API]" in line
        assert "401" in line
        assert "GET" in line
        assert "/v1/models" in line
        assert "client=192.0.2.10" in line

    def test_chat_completion_post(self):
        line = _format_log_line(
            "POST", "/v1/chat/completions", 200, "1.1", "192.168.1.35",
        )
        assert "200" in line
        assert "POST" in line
        assert "client=192.168.1.35" in line


class TestClientHostFromAddr:
    def test_ip_port_extracted_to_host(self):
        assert _client_host_from_addr("10.0.0.5:41234") == "10.0.0.5"

    def test_bare_ip_passes_through(self):
        assert _client_host_from_addr("192.0.2.10") == "192.0.2.10"

    def test_empty_returns_question(self):
        assert _client_host_from_addr("") == "?"
        assert _client_host_from_addr(None) == "?"


# ─── GenesisAccessLogReformatter (the entire patch) ─────────────────────


def _make_uvicorn_record(
    client_addr: str = "192.0.2.10:45116",
    method: str = "GET",
    full_path: str = "/v1/models",
    http_version: str = "1.1",
    status_code: int = 200,
    level: int = logging.INFO,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="uvicorn.access",
        level=level,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %s',
        args=(client_addr, method, full_path, http_version, status_code),
        exc_info=None,
    )
    return record


class TestReformatterDecisions:
    def test_non_uvicorn_logger_passes_through(self):
        rf = GenesisAccessLogReformatter(set(), False)
        rec = logging.LogRecord(
            "genesis.api", logging.INFO, "p", 1, "x", (), None,
        )
        assert rf.filter(rec) is True  # not our concern → pass through

    def test_warning_uvicorn_passes_through(self):
        """uvicorn warnings/errors are real diagnostics — keep them."""
        rf = GenesisAccessLogReformatter(set(), False)
        rec = _make_uvicorn_record(level=logging.WARNING, status_code=500)
        assert rf.filter(rec) is True

    def test_quiet_path_dropped_silently(self, caplog):
        rf = GenesisAccessLogReformatter({"/metrics"}, False)
        rec = _make_uvicorn_record(full_path="/metrics/cpu")
        with caplog.at_level(logging.INFO, logger="genesis.api"):
            assert rf.filter(rec) is False
        # Genesis-API line NOT emitted for quiet path
        assert not any("[Genesis-API]" in r.message for r in caplog.records)

    def test_health_default_dropped(self, caplog):
        rf = GenesisAccessLogReformatter({"/health", "/metrics"}, False)
        rec = _make_uvicorn_record(full_path="/health")
        with caplog.at_level(logging.INFO, logger="genesis.api"):
            assert rf.filter(rec) is False
        assert not any("[Genesis-API]" in r.message for r in caplog.records)

    def test_health_with_log_health_emitted(self, caplog):
        rf = GenesisAccessLogReformatter({"/health", "/metrics"}, True)
        rec = _make_uvicorn_record(full_path="/health")
        with caplog.at_level(logging.INFO, logger="genesis.api"):
            result = rf.filter(rec)
        # Reformatted into Genesis-API line and bare line dropped
        assert result is False
        assert any("[Genesis-API]" in r.message for r in caplog.records)

    def test_query_string_stripped_for_quiet_match(self, caplog):
        """Quiet-path matching uses path only, not query string."""
        rf = GenesisAccessLogReformatter({"/metrics"}, False)
        rec = _make_uvicorn_record(full_path="/metrics?detail=true")
        with caplog.at_level(logging.INFO, logger="genesis.api"):
            assert rf.filter(rec) is False
        assert not any("[Genesis-API]" in r.message for r in caplog.records)


class TestReformatterEmitLevels:
    @pytest.fixture
    def rf(self):
        return GenesisAccessLogReformatter(set(), False)

    def test_2xx_emits_info(self, rf, caplog):
        rec = _make_uvicorn_record(status_code=200)
        with caplog.at_level(logging.INFO, logger="genesis.api"):
            rf.filter(rec)
        assert any(
            r.levelno == logging.INFO and "200" in r.message
            for r in caplog.records
        )

    def test_4xx_emits_warning(self, rf, caplog):
        rec = _make_uvicorn_record(status_code=401)
        with caplog.at_level(logging.WARNING, logger="genesis.api"):
            rf.filter(rec)
        assert any(
            r.levelno == logging.WARNING and "401" in r.message
            for r in caplog.records
        )

    def test_5xx_emits_error(self, rf, caplog):
        rec = _make_uvicorn_record(status_code=500)
        # 500 records arrive at INFO level from uvicorn (it's a normal
        # access record); reformatter promotes to ERROR via genesis.api
        with caplog.at_level(logging.ERROR, logger="genesis.api"):
            rf.filter(rec)
        assert any(
            r.levelno == logging.ERROR and "500" in r.message
            for r in caplog.records
        )


class TestReformatterDefensive:
    def test_malformed_args_passes_through(self):
        """If uvicorn ever changes args shape, we let the bare line
        through unchanged instead of crashing."""
        rf = GenesisAccessLogReformatter(set(), False)
        rec = logging.LogRecord(
            "uvicorn.access", logging.INFO, "p", 1,
            "x", ("only-one-arg",), None,
        )
        assert rf.filter(rec) is True

    def test_non_int_status_passes_through(self):
        rf = GenesisAccessLogReformatter(set(), False)
        rec = logging.LogRecord(
            "uvicorn.access", logging.INFO, "p", 1, "x",
            ("ip:port", "GET", "/p", "1.1", "not-a-number"), None,
        )
        assert rf.filter(rec) is True


class TestReformatterStreamingSafe:
    """Wave 6 v3 contract: PN65 must NEVER be on the request hot path.

    Verifies the reformatter is ONLY a logging.Filter — no
    middleware/ASGI behavior on it.
    """

    def test_is_logging_filter_subclass(self):
        assert issubclass(GenesisAccessLogReformatter, logging.Filter)

    def test_does_not_define_call_or_dispatch(self):
        """No ASGI/Starlette dispatch hooks on the reformatter."""
        rf = GenesisAccessLogReformatter(set(), False)
        # Filter exposes .filter(record); should NOT expose the ASGI
        # __call__(scope, receive, send) signature.
        assert not hasattr(rf, "dispatch")  # BaseHTTPMiddleware contract


# ─── install_into_app backward-compat shim ──────────────────────────────


class _FakeApp:
    """Minimal app stub — v3 ignores it but the shim sets a marker."""
    pass


class TestInstallIntoApp:
    def test_first_install_returns_true_and_marks_app(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", "1")
        # Reset module-level install flag so this test sees a fresh state
        from sndr.engines.vllm.patches.middleware import pn65_access_log as p
        p._PN65_REFORMATTER_INSTALLED = False
        # Detach any reformatter left over from prior tests
        uv = logging.getLogger("uvicorn.access")
        for f in list(uv.filters):
            if isinstance(f, p.GenesisAccessLogReformatter):
                uv.removeFilter(f)

        app = _FakeApp()
        installed = install_into_app(app)
        assert installed is True
        assert getattr(app, "__pn65_installed__", False) is True

    def test_idempotent_second_install_returns_false(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", "1")
        from sndr.engines.vllm.patches.middleware import pn65_access_log as p
        p._PN65_REFORMATTER_INSTALLED = False
        uv = logging.getLogger("uvicorn.access")
        for f in list(uv.filters):
            if isinstance(f, p.GenesisAccessLogReformatter):
                uv.removeFilter(f)

        app = _FakeApp()
        install_into_app(app)
        installed_again = install_into_app(app)
        assert installed_again is False


# ─── apply() integration ────────────────────────────────────────────────


class TestApplyFunction:
    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        from sndr.engines.vllm.patches.middleware import pn65_access_log as p
        monkeypatch.delenv("GENESIS_ENABLE_PN65", raising=False)
        status, reason = p.apply()
        assert status == "skipped"

    def test_apply_returns_applied_when_enabled(self, monkeypatch):
        from sndr.engines.vllm.patches.middleware import pn65_access_log as p
        monkeypatch.setenv("GENESIS_ENABLE_PN65", "1")
        # Reset module-level install state so apply() actually installs
        p._PN65_REFORMATTER_INSTALLED = False
        p._PN65_FILTER_INSTALLED = False
        uv = logging.getLogger("uvicorn.access")
        for f in list(uv.filters):
            if isinstance(f, (p.GenesisAccessLogReformatter,
                              p._DropUvicornAccessInfo)):
                uv.removeFilter(f)
        for f in list(logging.getLogger().filters):
            if isinstance(f, p._DropUvicornAccessInfo):
                logging.getLogger().removeFilter(f)
        status, reason = p.apply()
        assert status == "applied"
        assert "v3" in reason or "logging-only" in reason


# ─── G-POST-07: uvicorn.access dedup filter (kept from v1) ──────────────


def _make_record(name: str, level: int, msg: str = "x") -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )


class TestDropUvicornAccessFilter:
    def test_drops_uvicorn_access_info(self):
        from sndr.engines.vllm.patches.middleware.pn65_access_log import (
            _DropUvicornAccessInfo,
        )
        f = _DropUvicornAccessInfo()
        rec = _make_record("uvicorn.access", logging.INFO,
                           '192.0.2.10 - "GET /v1/models" 401')
        assert f.filter(rec) is False

    def test_keeps_uvicorn_access_warning(self):
        from sndr.engines.vllm.patches.middleware.pn65_access_log import (
            _DropUvicornAccessInfo,
        )
        f = _DropUvicornAccessInfo()
        rec = _make_record("uvicorn.access", logging.WARNING, "boom")
        assert f.filter(rec) is True

    def test_keeps_uvicorn_access_error(self):
        from sndr.engines.vllm.patches.middleware.pn65_access_log import (
            _DropUvicornAccessInfo,
        )
        f = _DropUvicornAccessInfo()
        rec = _make_record("uvicorn.access", logging.ERROR, "5xx")
        assert f.filter(rec) is True

    def test_keeps_other_logger_info(self):
        from sndr.engines.vllm.patches.middleware.pn65_access_log import (
            _DropUvicornAccessInfo,
        )
        f = _DropUvicornAccessInfo()
        rec = _make_record("genesis.api", logging.INFO, "[Genesis-API] 200")
        assert f.filter(rec) is True

    def test_keeps_uvicorn_error_logger_info(self):
        from sndr.engines.vllm.patches.middleware.pn65_access_log import (
            _DropUvicornAccessInfo,
        )
        f = _DropUvicornAccessInfo()
        rec = _make_record("uvicorn.error", logging.INFO, "startup")
        assert f.filter(rec) is True


@pytest.fixture
def _reset_pn65_filter_install():
    from sndr.engines.vllm.patches.middleware import pn65_access_log as p

    def _strip():
        for logger in (logging.getLogger(), logging.getLogger("uvicorn.access")):
            for f in list(logger.filters):
                if isinstance(f, (p._DropUvicornAccessInfo,
                                  p.GenesisAccessLogReformatter)):
                    logger.removeFilter(f)
        p._PN65_FILTER_INSTALLED = False
        p._PN65_REFORMATTER_INSTALLED = False

    _strip()
    yield p
    _strip()


class TestSuppressUvicornAccessLogger:
    def test_install_attaches_filter_to_both_loggers(
        self, _reset_pn65_filter_install, monkeypatch,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.delenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", raising=False)
        p._suppress_uvicorn_access_logger()
        root_filters = [
            f for f in logging.getLogger().filters
            if isinstance(f, p._DropUvicornAccessInfo)
        ]
        uvicorn_filters = [
            f for f in logging.getLogger("uvicorn.access").filters
            if isinstance(f, p._DropUvicornAccessInfo)
        ]
        assert len(root_filters) == 1
        assert len(uvicorn_filters) == 1
        assert p._PN65_FILTER_INSTALLED is True

    def test_install_is_idempotent(
        self, _reset_pn65_filter_install, monkeypatch,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.delenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", raising=False)
        p._suppress_uvicorn_access_logger()
        p._suppress_uvicorn_access_logger()
        p._suppress_uvicorn_access_logger()
        root_filters = [
            f for f in logging.getLogger().filters
            if isinstance(f, p._DropUvicornAccessInfo)
        ]
        uvicorn_filters = [
            f for f in logging.getLogger("uvicorn.access").filters
            if isinstance(f, p._DropUvicornAccessInfo)
        ]
        assert len(root_filters) == 1
        assert len(uvicorn_filters) == 1

    def test_keep_env_skips_install(
        self, _reset_pn65_filter_install, monkeypatch,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.setenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", "1")
        p._suppress_uvicorn_access_logger()
        assert p._PN65_FILTER_INSTALLED is False

    @pytest.mark.parametrize("val", ["true", "yes", "Y", "ON", "1"])
    def test_keep_env_truthy_variants_skip_install(
        self, val, _reset_pn65_filter_install, monkeypatch,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.setenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", val)
        p._suppress_uvicorn_access_logger()
        assert p._PN65_FILTER_INSTALLED is False

    def test_keep_env_empty_does_not_skip(
        self, _reset_pn65_filter_install, monkeypatch,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.setenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", "")
        p._suppress_uvicorn_access_logger()
        assert p._PN65_FILTER_INSTALLED is True

    def test_filter_actually_drops_records_after_install(
        self, _reset_pn65_filter_install, monkeypatch, caplog,
    ):
        p = _reset_pn65_filter_install
        monkeypatch.delenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", raising=False)
        p._suppress_uvicorn_access_logger()

        uv = logging.getLogger("uvicorn.access")
        uv.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO, logger="uvicorn.access"):
            uv.info('192.0.2.10:45116 - "GET /v1/models HTTP/1.1" 401')
            uv.warning("503 backend down")

        names_levels = [(r.name, r.levelno, r.getMessage())
                        for r in caplog.records]
        info_records = [
            x for x in names_levels
            if x[0] == "uvicorn.access" and x[1] == logging.INFO
        ]
        warn_records = [
            x for x in names_levels
            if x[0] == "uvicorn.access" and x[1] == logging.WARNING
        ]
        assert info_records == []
        assert len(warn_records) == 1


# ─── deep-audit #5: filter install ORDER (reformatter before suppressor) ───


class TestFilterInstallOrder:
    """Regression for deep-audit #5: the reformatter must install BEFORE the
    suppressor. Both are filters on the uvicorn.access logger, applied in
    install order; the first to return False drops the record before later
    filters run. The suppressor drops every INFO access record, so installing
    it first starved the reformatter and ZERO [Genesis-API] lines were emitted
    for 2xx requests — the patch's entire purpose was dead.
    """

    def _fresh_install(self, monkeypatch):
        from sndr.engines.vllm.patches.middleware import pn65_access_log as p

        monkeypatch.setenv("GENESIS_ENABLE_PN65", "1")
        monkeypatch.delenv("GENESIS_PN65_KEEP_UVICORN_ACCESS", raising=False)
        p._PN65_REFORMATTER_INSTALLED = False
        p._PN65_FILTER_INSTALLED = False
        uv = logging.getLogger("uvicorn.access")
        for f in list(uv.filters):
            if isinstance(f, (p.GenesisAccessLogReformatter,
                              p._DropUvicornAccessInfo)):
                uv.removeFilter(f)
        for f in list(logging.getLogger().filters):
            if isinstance(f, p._DropUvicornAccessInfo):
                logging.getLogger().removeFilter(f)
        status, _ = p.apply()
        assert status == "applied"
        return p, uv

    def test_reformatter_precedes_suppressor(self, monkeypatch):
        p, uv = self._fresh_install(monkeypatch)
        ri = next(i for i, f in enumerate(uv.filters)
                  if isinstance(f, p.GenesisAccessLogReformatter))
        si = next(i for i, f in enumerate(uv.filters)
                  if isinstance(f, p._DropUvicornAccessInfo))
        kinds = [type(f).__name__ for f in uv.filters]
        assert ri < si, f"reformatter must precede suppressor; got {kinds}"

    def test_2xx_record_emits_genesis_api_line(self, monkeypatch):
        p, uv = self._fresh_install(monkeypatch)
        captured: list[str] = []

        class _Cap(logging.Handler):
            def emit(self, r):
                captured.append(r.getMessage())

        gl = logging.getLogger("genesis.api")
        h = _Cap()
        gl.addHandler(h)
        old_level = gl.level
        gl.setLevel(logging.INFO)
        try:
            rec = logging.LogRecord(
                name="uvicorn.access", level=logging.INFO,
                pathname=__file__, lineno=1,
                msg='%s - "%s %s HTTP/%s" %d',
                args=("127.0.0.1:5050", "GET", "/v1/chat/completions",
                      "1.1", 200),
                exc_info=None,
            )
            # Replicate Python's Logger.filter() chain: stop at first False.
            allowed = True
            for f in uv.filters:
                if not f.filter(rec):
                    allowed = False
                    break
        finally:
            gl.removeHandler(h)
            gl.setLevel(old_level)

        # Reformatter ran first, emitted the structured line, then dropped the
        # bare uvicorn record.
        assert allowed is False, "bare uvicorn 2xx line should be dropped"
        assert any("/v1/chat/completions" in m for m in captured), (
            "reformatter must emit a [Genesis-API] line for 2xx — it did not, "
            "the suppressor starved it"
        )
