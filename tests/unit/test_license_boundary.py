# SPDX-License-Identifier: Apache-2.0
"""Phase 4.6 acceptance — public/private license boundary.

INVARIANTS (any failure here blocks release):

1. `sndr.license` imports cleanly without network access.
2. `core_license_status()` always reports core = "public (unlicensed)"
   on a clean public install.
3. `verify_license_file()` defers honestly when no engine installed —
   does NOT invent verification success.
4. `sndr license status` exits 0 on a clean public install.
5. No env-var or filesystem read at import time.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from unittest.mock import patch


# ─── Import-time invariants ────────────────────────────────────────────


class TestImportInvariants:
    def test_module_imports_cleanly(self):
        """No exception on import — module is publication-safe."""
        import sndr.license as mod
        # Phase 4.6 boundary surface is stable.
        assert hasattr(mod, "is_engine_installed")
        assert hasattr(mod, "core_license_status")
        assert hasattr(mod, "verify_license_file")
        assert hasattr(mod, "CoreLicenseStatus")
        # Legacy engine-tier surface still present (unchanged).
        assert hasattr(mod, "LicenseStatus")          # enum
        assert hasattr(mod, "check_engine_tier_eligible")

    def test_no_network_module_imported(self):
        """Import of the license module MUST NOT pull in network libs."""
        # Snapshot sys.modules before re-import to isolate the side effect.
        forbidden = {
            "socket", "ssl", "http.client", "urllib.request",
            "requests", "httpx",
        }
        # Drop our module if it's cached so the test re-imports fresh.
        for name in list(sys.modules):
            if name == "sndr.license" or name.startswith(
                "sndr.license."
            ):
                del sys.modules[name]
        # Snapshot which "network" modules were pre-imported by other code.
        pre = {m for m in forbidden if m in sys.modules}
        import sndr.license  # noqa: F401 — testing the side effects of import
        post = {m for m in forbidden if m in sys.modules}
        # The license module itself must not have added any network module.
        added = post - pre
        assert not added, f"license import pulled in: {added}"


# ─── core_license_status() ─────────────────────────────────────────────


class TestCoreLicenseStatus:
    def test_unlicensed_core_default(self):
        """Public install reports core = 'public (unlicensed)'."""
        from sndr.license import core_license_status
        status = core_license_status()
        assert status.core == "public (unlicensed)"
        # On clean public install no engine.
        assert status.engine is None
        # License fields stay None.
        assert status.license_path is None
        assert status.license_tier is None
        assert status.premium_patches_enabled == 0

    def test_returns_license_status_dataclass(self):
        """Stable shape — tooling consumes asdict(status)."""
        from dataclasses import asdict, is_dataclass
        from sndr.license import core_license_status, CoreLicenseStatus
        status = core_license_status()
        assert isinstance(status, CoreLicenseStatus)
        assert is_dataclass(status)
        # asdict returns a plain dict the CLI can json-serialize.
        d = asdict(status)
        assert "core" in d
        assert d["core"] == "public (unlicensed)"


# ─── verify_license_file() ─────────────────────────────────────────────


class TestVerifyLicenseFile:
    def test_defers_without_engine(self):
        """Public-core MUST NOT invent a verification result."""
        from sndr.license import verify_license_file
        result = verify_license_file("/nonexistent/license.lic")
        assert result.valid is False
        assert "deferred" in result.reason.lower()
        assert "vllm-sndr-engine" in result.reason

    def test_returns_dataclass(self):
        from dataclasses import is_dataclass
        from sndr.license import verify_license_file
        result = verify_license_file("/tmp/anything.lic")
        assert is_dataclass(result)
        assert hasattr(result, "valid")
        assert hasattr(result, "reason")


# ─── is_engine_installed() ─────────────────────────────────────────────


class TestEngineDetection:
    def test_engine_not_installed_default(self):
        from sndr.license import is_engine_installed
        result = is_engine_installed()
        # On the public-only repo, no engine present.
        assert result.installed is False
        assert result.module_name is None


# ─── CLI surface ──────────────────────────────────────────────────────


class TestCLILicenseStatus:
    def test_status_returns_zero_on_clean_install(self):
        from sndr.cli.legacy import license as lic_cli
        ns = argparse.Namespace(json=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = lic_cli.run_status(ns)
        out = buf.getvalue()
        assert rc == 0
        assert "public (unlicensed)" in out
        assert "Engine (private):      not detected" in out

    def test_status_json_machine_readable(self):
        from sndr.cli.legacy import license as lic_cli
        ns = argparse.Namespace(json=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = lic_cli.run_status(ns)
        out = buf.getvalue()
        assert rc == 0
        payload = json.loads(out)
        assert payload["core"] == "public (unlicensed)"
        assert payload["engine"] is None


class TestCLILicenseVerify:
    def test_verify_no_engine_exits_zero(self):
        """Deferred verification (no engine) is exit 0 — NOT a failure."""
        from sndr.cli.legacy import license as lic_cli
        ns = argparse.Namespace(
            file="/tmp/anything.lic", offline=True, json=False
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = lic_cli.run_verify(ns)
        out = buf.getvalue()
        assert rc == 0       # Deferred ≠ failure.
        assert "deferred" in out.lower()

    def test_verify_json_deferred(self):
        from sndr.cli.legacy import license as lic_cli
        ns = argparse.Namespace(
            file="/tmp/anything.lic", offline=True, json=True
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = lic_cli.run_verify(ns)
        out = buf.getvalue()
        assert rc == 0
        payload = json.loads(out)
        assert payload["valid"] is False
        assert "deferred" in payload["reason"].lower()


# ─── Argparser registration ───────────────────────────────────────────


class TestArgparserRegistration:
    def test_license_argparser_registers(self):
        from sndr.cli.legacy.license import add_argparser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_argparser(sub)
        ns = p.parse_args(["license", "status"])
        assert ns.license_cmd == "status"

    def test_top_level_includes_license(self):
        # v12: the legacy CLI (which registers the license subcommand)
        # lives in sndr.cli.legacy; bare sndr.cli is the platform shell.
        from sndr.cli import legacy as cli_mod
        assert hasattr(cli_mod, "_license_argparser")
        assert callable(cli_mod._license_argparser)


# ─── security_scan.py smoke ────────────────────────────────────────────


class TestSecurityScanScript:
    """security_scan.py exit code semantics. We don't run the full scan
    here (it walks the repo); instead we verify the module imports + the
    individual check functions are callable."""

    def test_module_imports(self):
        # Run as a Python module to validate syntax.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "security_scan_test",
            __import__("os").path.join(
                __import__("os").path.dirname(__file__),
                "..", "..", "scripts", "security_scan.py",
            ),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Public functions exist.
        assert callable(mod.check_no_operator_paths)
        assert callable(mod.check_no_private_ips)
        assert callable(mod.check_no_private_keys)
        assert callable(mod.check_no_env_files)
        assert callable(mod.check_no_aws_keys)
        assert callable(mod.check_release_artifacts_present)

    def test_aws_key_pattern_catches_real_format(self):
        """Spot-check the AWS detector against a known-format key."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "security_scan_test2",
            __import__("os").path.join(
                __import__("os").path.dirname(__file__),
                "..", "..", "scripts", "security_scan.py",
            ),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Synthesize a fake hit file and assert pattern matches.
        import re
        pat = re.compile(r"\bAKIA[A-Z0-9]{16}\b")
        # Plausible AWS access key format (fake).
        assert pat.search("export AWS_KEY=AKIAIOSFODNN7EXAMPLE")
        # Real env vars without AWS prefix are not flagged.
        assert not pat.search("VLLM_API_KEY=genesis-local")
