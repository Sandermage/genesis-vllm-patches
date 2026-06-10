# SPDX-License-Identifier: Apache-2.0
"""PN283 — tests for the multiprocess Prometheus dir bootstrap.

5 required cases from the build scope:

  1. unset PROMETHEUS_MULTIPROC_DIR  → skipped no-op
  2. set + missing dir               → creates dir, applied
  3. set + existing dir + non-empty  → keeps (warns about staleness),
                                       applied
  4. set + non-writable dir          → warned, no boot block
  5. set + SNDR_PROMETHEUS_MULTIPROC_CLEAN=1
                                     → removes stale files, applied

Plus idempotency: a second call in the same process returns "skipped /
already initialised this process".
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from sndr.observability import multiproc_bootstrap as mp_boot


@pytest.fixture(autouse=True)
def _module_reset(monkeypatch):
    """Wipe singleton state + env between tests."""
    mp_boot._reset_module_state()
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.delenv("SNDR_PROMETHEUS_MULTIPROC_CLEAN", raising=False)
    yield
    mp_boot._reset_module_state()


# ─── Case 1: env unset → no-op ──────────────────────────────────────────


class TestEnvUnset:
    def test_returns_skipped(self):
        status, reason = mp_boot.setup_prometheus_multiproc_dir()
        assert status == "skipped"
        assert "PROMETHEUS_MULTIPROC_DIR unset" in reason

    def test_no_side_effect_on_filesystem(self, tmp_path):
        # Pre-condition: tmp_path empty
        assert list(tmp_path.iterdir()) == []
        mp_boot.setup_prometheus_multiproc_dir()
        # Post-condition: tmp_path still empty (function did NOT touch it)
        assert list(tmp_path.iterdir()) == []

    def test_is_initialised_true_after_skip(self):
        assert mp_boot.is_initialised() is False
        mp_boot.setup_prometheus_multiproc_dir()
        assert mp_boot.is_initialised() is True


# ─── Case 2: env set + dir missing → create ─────────────────────────────


class TestDirMissing:
    def test_creates_missing_dir(self, tmp_path, monkeypatch):
        target = tmp_path / "sndr-prom-mp"
        assert not target.exists()
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))

        status, reason = mp_boot.setup_prometheus_multiproc_dir()

        assert status == "applied"
        assert target.exists()
        assert target.is_dir()
        assert "dir ready" in reason

    def test_dir_mode_is_0700(self, tmp_path, monkeypatch):
        target = tmp_path / "sndr-prom-mp"
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))

        mp_boot.setup_prometheus_multiproc_dir()

        mode = stat.S_IMODE(target.stat().st_mode)
        # Process umask may mask the requested 0o700 down; verify at
        # least owner-only access (no group/other read).
        assert mode & 0o077 == 0, f"unexpected mode {oct(mode)}"


# ─── Case 3: env set + existing non-empty dir → keep + warn ─────────────


class TestDirExistingNonEmpty:
    def test_keeps_files_when_clean_unset(
        self, tmp_path, monkeypatch, caplog,
    ):
        target = tmp_path / "sndr-prom-mp"
        target.mkdir()
        stale = target / "counter_12345.db"
        stale.write_text("stale content")
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))

        import logging
        with caplog.at_level(logging.WARNING):
            status, reason = mp_boot.setup_prometheus_multiproc_dir()

        assert status == "applied"
        # Stale file PRESERVED (no cleanup without explicit opt-in)
        assert stale.exists()
        assert stale.read_text() == "stale content"

        # Warning emitted about stale files
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "non-empty" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "SNDR_PROMETHEUS_MULTIPROC_CLEAN" in warnings[0].getMessage()

    def test_writability_probe_runs_and_cleans_up(
        self, tmp_path, monkeypatch,
    ):
        target = tmp_path / "sndr-prom-mp"
        target.mkdir()
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))

        status, _ = mp_boot.setup_prometheus_multiproc_dir()

        assert status == "applied"
        # Probe file should NOT remain after probe
        assert not (target / ".sndr_writable_check").exists()


# ─── Case 4: env set + non-writable dir → warned ────────────────────────


class TestNonWritable:
    def test_warns_does_not_raise(self, tmp_path, monkeypatch, caplog):
        # Skip on platforms where chmod 0o500 doesn't restrict the
        # running user (e.g. root in some CI). The intent is to verify
        # observability never blocks boot.
        target = tmp_path / "sndr-prom-mp"
        target.mkdir()
        os.chmod(target, 0o500)  # r-x for owner; no write
        if os.access(target, os.W_OK):
            pytest.skip(
                "chmod 0o500 not effective (likely running as root); "
                "test only meaningful for non-root user"
            )

        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))

        import logging
        with caplog.at_level(logging.WARNING):
            status, reason = mp_boot.setup_prometheus_multiproc_dir()

        # Restore mode for tmp_path cleanup
        os.chmod(target, 0o700)

        assert status == "warned"
        assert "not writable" in reason

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "not writable" in r.getMessage()
        ]
        assert len(warnings) >= 1


# ─── Case 5: clean env → removes stale files ────────────────────────────


class TestCleanEnv:
    def test_removes_files_when_clean_env_set(
        self, tmp_path, monkeypatch,
    ):
        target = tmp_path / "sndr-prom-mp"
        target.mkdir()
        for pid in (1234, 5678, 9012):
            (target / f"counter_{pid}.db").write_text(f"stale {pid}")
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))
        monkeypatch.setenv("SNDR_PROMETHEUS_MULTIPROC_CLEAN", "1")

        status, reason = mp_boot.setup_prometheus_multiproc_dir()

        assert status == "applied"
        assert "cleaned" in reason
        # All stale files removed
        remaining = [
            p for p in target.iterdir()
            if not p.name.startswith(".")
        ]
        assert remaining == [], f"unexpected files: {remaining}"

    def test_clean_with_empty_dir_is_idempotent(
        self, tmp_path, monkeypatch,
    ):
        target = tmp_path / "sndr-prom-mp"
        target.mkdir()
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))
        monkeypatch.setenv("SNDR_PROMETHEUS_MULTIPROC_CLEAN", "1")

        status, reason = mp_boot.setup_prometheus_multiproc_dir()
        assert status == "applied"
        assert "cleaned 0 stale" in reason

    def test_clean_does_not_remove_subdirectories(
        self, tmp_path, monkeypatch,
    ):
        target = tmp_path / "sndr-prom-mp"
        target.mkdir()
        (target / "subdir").mkdir()
        (target / "subdir" / "nested.txt").write_text("nested")
        (target / "counter_1.db").write_text("stale")
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))
        monkeypatch.setenv("SNDR_PROMETHEUS_MULTIPROC_CLEAN", "1")

        mp_boot.setup_prometheus_multiproc_dir()

        # File at top level removed; subdirectory preserved
        assert not (target / "counter_1.db").exists()
        assert (target / "subdir" / "nested.txt").exists()


# ─── Idempotency ────────────────────────────────────────────────────────


class TestIdempotency:
    def test_second_call_is_skipped(self, tmp_path, monkeypatch):
        target = tmp_path / "sndr-prom-mp"
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))

        s1, _ = mp_boot.setup_prometheus_multiproc_dir()
        s2, r2 = mp_boot.setup_prometheus_multiproc_dir()

        assert s1 == "applied"
        assert s2 == "skipped"
        assert "already initialised" in r2

    def test_idempotent_after_env_unset_run(self):
        s1, _ = mp_boot.setup_prometheus_multiproc_dir()
        s2, r2 = mp_boot.setup_prometheus_multiproc_dir()

        assert s1 == "skipped"
        assert s2 == "skipped"
        # First-run reason is "unset"; second-run reason is "already
        # initialised"
        assert "already initialised" in r2


# ─── Env-var truthy parsing ─────────────────────────────────────────────


class TestEnvTruthy:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "Y"])
    def test_clean_env_truthy_values(
        self, tmp_path, monkeypatch, value,
    ):
        target = tmp_path / "sndr-prom-mp"
        target.mkdir()
        (target / "stale.db").write_text("x")
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))
        monkeypatch.setenv("SNDR_PROMETHEUS_MULTIPROC_CLEAN", value)

        mp_boot.setup_prometheus_multiproc_dir()

        assert not (target / "stale.db").exists()

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "off"])
    def test_clean_env_falsy_values(
        self, tmp_path, monkeypatch, value,
    ):
        target = tmp_path / "sndr-prom-mp"
        target.mkdir()
        (target / "stale.db").write_text("x")
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(target))
        monkeypatch.setenv("SNDR_PROMETHEUS_MULTIPROC_CLEAN", value)

        mp_boot.setup_prometheus_multiproc_dir()

        # Stale file PRESERVED on falsy / empty clean env
        assert (target / "stale.db").exists()
