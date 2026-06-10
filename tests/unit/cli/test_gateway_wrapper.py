# SPDX-License-Identifier: Apache-2.0
"""P1.6 unit tests for `sndr gateway` thin CLI wrapper.

Operator acceptance gates (from the P1.6 GO message):

  G01  `sndr gateway --help` works (argparse-only path, no heavy imports)
  G02  CLI module imports without torch / vllm.v1 / fastapi / uvicorn / httpx
  G03  CLI flags map onto SNDR_GATEWAY_* env vars 1:1
  G04  D2a 6-case smoke is not broken (we don't touch routing — verified
       indirectly: the wrapper just sets env + calls existing main())
  G05  Streaming path D2b is not broken (same: env-only contract)
  G06  No production launcher edits (this is a tests/unit test file only)

The wrapper is intentionally narrow: parse args → set env → call main().
The tests focus on the wrapper's contract, not on gateway behavior.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from sndr.cli.legacy.gateway import (
    _FLAG_TO_ENV,
    _apply_env,
    add_argparser,
    run_gateway,
)


# ─── G01 — --help works ─────────────────────────────────────────────────


class TestHelp:
    def test_help_exits_zero(self):
        """`sndr gateway --help` must succeed without any runtime
        dependency."""
        result = subprocess.run(
            [sys.executable, "-m", "sndr.cli.legacy", "gateway", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"--help failed: exit={result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )
        # Every documented flag should be visible
        for flag in ("--host", "--port", "--default-url", "--structured-url",
                     "--profile", "--health-interval", "--timeout",
                     "--admin-allow-remote", "--log-level"):
            assert flag in result.stdout, f"--help missing: {flag}"


# ─── G02 — no heavy imports at CLI registration ─────────────────────────


class TestNoHeavyImports:
    def test_cli_gateway_module_doesnt_pull_fastapi(self):
        """Importing the wrapper module must NOT drag in fastapi /
        uvicorn / httpx — those should stay lazy. This is what keeps
        `sndr --help` and `sndr gateway --help` fast and dependency-
        free on systems without the gateway runtime deps."""
        # Run in a fresh subprocess so we can isolate sys.modules
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
baseline = set(sys.modules)
import sndr.cli.legacy.gateway
after = set(sys.modules)
heavy = {m for m in (after - baseline)
         if m.split('.')[0] in ('fastapi', 'uvicorn', 'httpx', 'torch')
         or m.startswith('vllm.v1')}
if heavy:
    print('HEAVY:', sorted(heavy))
    sys.exit(1)
print('OK')
"""],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"heavy imports leaked: {result.stdout}\n{result.stderr}"
        )

    def test_cli_init_doesnt_pull_gateway_runtime(self):
        """Importing `sndr.cli.legacy` (the whole CLI package) does
        not eagerly import the gateway's FastAPI runtime."""
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
baseline = set(sys.modules)
import sndr.cli
after = set(sys.modules)
forbidden = {m for m in (after - baseline)
             if m.split('.')[0] in ('fastapi', 'uvicorn')}
if forbidden:
    print('LEAK:', sorted(forbidden))
    sys.exit(1)
print('OK')
"""],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"fastapi/uvicorn leaked: {result.stdout}\n{result.stderr}"
        )


# ─── G03 — arg-to-env mapping ───────────────────────────────────────────


class TestArgToEnvMapping:
    @pytest.fixture(autouse=True)
    def _wipe_env(self, monkeypatch):
        """Each test starts with no SNDR_GATEWAY_* env to avoid
        cross-contamination."""
        for env in list(os.environ):
            if env.startswith("SNDR_GATEWAY_") or env.startswith("GENESIS_GATEWAY_"):
                monkeypatch.delenv(env, raising=False)
        yield

    def test_flag_to_env_map_has_documented_keys(self):
        """The mapping table must include the 8 documented operator-
        facing args. If a flag is added/removed, this gate catches it
        and forces a docstring + test update."""
        expected = {
            "default_url", "structured_url", "host", "port",
            "profile", "health_interval", "timeout", "log_level",
        }
        assert set(_FLAG_TO_ENV) == expected

    def test_default_url_arg_sets_env(self):
        ns = argparse.Namespace(
            default_url="http://example:8101",
            structured_url=None, host=None, port=None,
            profile=None, health_interval=None, timeout=None,
            log_level=None, admin_allow_remote=False,
        )
        _apply_env(ns)
        assert os.environ["SNDR_GATEWAY_DEFAULT_URL"] == "http://example:8101"
        assert "SNDR_GATEWAY_STRUCTURED_URL" not in os.environ

    def test_all_flags_map(self):
        ns = argparse.Namespace(
            default_url="http://a:1",
            structured_url="http://b:2",
            host="1.2.3.4",
            port="9000",
            profile="custom-profile",
            health_interval="3",
            timeout="60",
            log_level="DEBUG",
            admin_allow_remote=False,
        )
        _apply_env(ns)
        assert os.environ["SNDR_GATEWAY_DEFAULT_URL"] == "http://a:1"
        assert os.environ["SNDR_GATEWAY_STRUCTURED_URL"] == "http://b:2"
        assert os.environ["SNDR_GATEWAY_BIND_HOST"] == "1.2.3.4"
        assert os.environ["SNDR_GATEWAY_BIND_PORT"] == "9000"
        assert os.environ["SNDR_GATEWAY_PROFILE"] == "custom-profile"
        assert os.environ["SNDR_GATEWAY_HEALTH_INTERVAL"] == "3"
        assert os.environ["SNDR_GATEWAY_TIMEOUT"] == "60"
        assert os.environ["SNDR_GATEWAY_LOG_LEVEL"] == "DEBUG"

    def test_admin_allow_remote_flag_sets_env_to_1(self):
        ns = argparse.Namespace(
            default_url=None, structured_url=None, host=None, port=None,
            profile=None, health_interval=None, timeout=None,
            log_level=None, admin_allow_remote=True,
        )
        _apply_env(ns)
        assert os.environ["SNDR_GATEWAY_ADMIN_ALLOW_REMOTE"] == "1"

    def test_admin_allow_remote_default_does_not_set_env(self):
        ns = argparse.Namespace(
            default_url=None, structured_url=None, host=None, port=None,
            profile=None, health_interval=None, timeout=None,
            log_level=None, admin_allow_remote=False,
        )
        _apply_env(ns)
        assert "SNDR_GATEWAY_ADMIN_ALLOW_REMOTE" not in os.environ

    def test_unset_args_preserve_existing_env(self, monkeypatch):
        """If operator has SNDR_GATEWAY_BIND_HOST set in the shell and
        doesn't pass --host, the wrapper must not clobber it."""
        monkeypatch.setenv("SNDR_GATEWAY_BIND_HOST", "preserved-value")
        ns = argparse.Namespace(
            default_url=None, structured_url=None, host=None, port=None,
            profile=None, health_interval=None, timeout=None,
            log_level=None, admin_allow_remote=False,
        )
        _apply_env(ns)
        assert os.environ["SNDR_GATEWAY_BIND_HOST"] == "preserved-value"

    def test_cli_arg_overrides_existing_env(self, monkeypatch):
        """CLI is authoritative for this invocation: if both --host and
        the env are set, the CLI value wins."""
        monkeypatch.setenv("SNDR_GATEWAY_BIND_HOST", "old")
        ns = argparse.Namespace(
            default_url=None, structured_url=None, host="new",
            port=None, profile=None, health_interval=None,
            timeout=None, log_level=None, admin_allow_remote=False,
        )
        _apply_env(ns)
        assert os.environ["SNDR_GATEWAY_BIND_HOST"] == "new"


# ─── Handler execution path ─────────────────────────────────────────────


class TestRunGatewayExec:
    @pytest.fixture(autouse=True)
    def _wipe_env(self, monkeypatch):
        for env in list(os.environ):
            if env.startswith("SNDR_GATEWAY_") or env.startswith("GENESIS_GATEWAY_"):
                monkeypatch.delenv(env, raising=False)
        yield

    def test_run_gateway_calls_main(self):
        """run_gateway() lazily imports the gateway main and calls it.
        We patch the import target so this test doesn't actually start
        a uvicorn server."""
        called = {"count": 0}

        def fake_main():
            called["count"] += 1

        fake_module = type(sys)("fake_gateway_app")
        fake_module.main = fake_main
        with patch.dict(
            sys.modules,
            {"sndr.engines.vllm.patches.spec_decode.gateway.app": fake_module},
        ):
            ns = argparse.Namespace(
                default_url="http://a:1", structured_url=None,
                host=None, port=None, profile=None,
                health_interval=None, timeout=None, log_level=None,
                admin_allow_remote=False,
            )
            rc = run_gateway(ns)
        assert rc == 0
        assert called["count"] == 1
        # And the env was set as a side-effect
        assert os.environ["SNDR_GATEWAY_DEFAULT_URL"] == "http://a:1"

    def test_run_gateway_keyboardinterrupt_returns_130(self):
        def fake_main():
            raise KeyboardInterrupt

        fake_module = type(sys)("fake_gateway_app")
        fake_module.main = fake_main
        with patch.dict(
            sys.modules,
            {"sndr.engines.vllm.patches.spec_decode.gateway.app": fake_module},
        ):
            ns = argparse.Namespace(
                default_url=None, structured_url=None,
                host=None, port=None, profile=None,
                health_interval=None, timeout=None, log_level=None,
                admin_allow_remote=False,
            )
            rc = run_gateway(ns)
        assert rc == 130

    def test_run_gateway_missing_deps_returns_3(self):
        """If fastapi / uvicorn aren't installed, we return exit 3 with
        a clear message — distinct from exit 2 (tooling failure)."""
        # Patch the import to raise ImportError
        import importlib
        real_import = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "sndr.engines.vllm.patches.spec_decode.gateway.app":
                raise ImportError("fastapi not installed (synthetic)")
            return real_import(name, *args, **kwargs)

        # We can't patch import directly; force the inner from-import
        # to fail by removing the target module + injecting a sentinel.
        sentinel_name = "sndr.engines.vllm.patches.spec_decode.gateway.app"
        original = sys.modules.pop(sentinel_name, None)
        try:
            # Pre-populate with an empty stub so the `from ... import main`
            # AttributeError-but-actually-ImportError-on-attribute. Better:
            # we put an object that raises on attr access.
            class _Raiser:
                def __getattr__(self, item):
                    raise ImportError(
                        f"synthetic missing dep when accessing {item}"
                    )
            sys.modules[sentinel_name] = _Raiser()

            ns = argparse.Namespace(
                default_url=None, structured_url=None,
                host=None, port=None, profile=None,
                health_interval=None, timeout=None, log_level=None,
                admin_allow_remote=False,
            )
            rc = run_gateway(ns)
            assert rc == 3
        finally:
            if original is not None:
                sys.modules[sentinel_name] = original
            else:
                sys.modules.pop(sentinel_name, None)
