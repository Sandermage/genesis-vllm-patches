# SPDX-License-Identifier: Apache-2.0
"""Tests for trust anchor generator script + placeholder detection.

Wave 4.4 — replaces placeholder zero-key in license.py with real
production keypair.
"""
from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_generator_script() -> Path | None:
    """Locate the trust-anchor generator wherever the maintainer keeps it.

    The script is maintainer-only — public clones do not carry it. The
    test module skips entirely when the script is not visible, so the
    public CI surface stays green without revealing the maintainer's
    layout."""
    for rel in ("scripts/generate_trust_anchor.py",):
        path = REPO_ROOT / rel
        if path.is_file():
            return path
    override = REPO_ROOT.parent / "_maintainer_scripts" / "generate_trust_anchor.py"
    if override.is_file():
        return override
    for match in REPO_ROOT.glob("*/scripts/generate_trust_anchor.py"):
        if match.is_file():
            return match
    return None


SCRIPT_PATH = _find_generator_script()

pytestmark = pytest.mark.skipif(
    SCRIPT_PATH is None,
    reason="trust-anchor generator script not present in this checkout "
           "(maintainer-only tool)",
)


@pytest.fixture
def generator_module():
    spec = importlib.util.spec_from_file_location(
        "generate_trust_anchor", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cryptography_available():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        return True
    except ImportError:
        pytest.skip("cryptography not installed")


# ─── Generator output shape ──────────────────────────────────────────────


class TestGenerator:
    def test_generates_valid_keypair(self, cryptography_available, generator_module):
        pub_b64, priv_b64 = generator_module.generate_keypair()
        # Both should be base64url, no padding, 43 chars (32 bytes raw)
        assert len(pub_b64) == 43
        assert len(priv_b64) == 43
        # Decodable
        pub_bytes = base64.urlsafe_b64decode(pub_b64 + "==")
        priv_bytes = base64.urlsafe_b64decode(priv_b64 + "==")
        assert len(pub_bytes) == 32
        assert len(priv_bytes) == 32
        # Non-zero
        assert pub_bytes != b"\x00" * 32
        assert priv_bytes != b"\x00" * 32

    def test_keypair_is_unique_each_call(self, cryptography_available, generator_module):
        p1, _ = generator_module.generate_keypair()
        p2, _ = generator_module.generate_keypair()
        assert p1 != p2

    def test_save_private_key_chmod_0o600(self, cryptography_available, generator_module, tmp_path):
        priv = "x" * 43
        path = tmp_path / "subdir" / "priv.key"
        generator_module.save_private_key(path, priv)
        assert path.is_file()
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_save_refuses_overwrite(self, cryptography_available, generator_module, tmp_path):
        path = tmp_path / "priv.key"
        path.write_text("existing", encoding="utf-8")
        with pytest.raises(SystemExit):
            generator_module.save_private_key(path, "newkey")


# ─── License.py runtime warning ──────────────────────────────────────────


class TestPlaceholderDetection:
    def test_is_placeholder_returns_true_for_zeros(self, monkeypatch):
        """When `_TRUST_ANCHOR_PUBKEY_B64URL` is set to the 32-zero
        placeholder, the detector must fire. This test exercises
        the function semantics — not the live module state (the
        real production key is currently active).
        """
        from sndr import license as L
        zero_b64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        monkeypatch.setattr(L, "_TRUST_ANCHOR_PUBKEY_B64URL", zero_b64)
        assert L._is_placeholder_anchor() is True

    def test_warning_logged_at_module_load(self, caplog, monkeypatch):
        """The warning must fire when the anchor is a placeholder.
        We mock the module constant; the production tree carries
        the real key."""
        from sndr import license as L
        assert hasattr(L, "_maybe_log_placeholder_warning")
        zero_b64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        monkeypatch.setattr(L, "_TRUST_ANCHOR_PUBKEY_B64URL", zero_b64)
        L._TRUST_ANCHOR_PLACEHOLDER_DETECTED = False
        with caplog.at_level("WARNING", logger="genesis.license"):
            L._maybe_log_placeholder_warning()
        assert any(
            "PLACEHOLDER zero-key" in r.message
            for r in caplog.records
        )

    def test_warning_idempotent(self, caplog, monkeypatch):
        """Repeated calls to `_maybe_log_placeholder_warning` must not
        emit new warnings (idempotency). We mock the anchor into
        placeholder mode; otherwise the function emits nothing."""
        from sndr import license as L
        zero_b64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        monkeypatch.setattr(L, "_TRUST_ANCHOR_PUBKEY_B64URL", zero_b64)
        L._TRUST_ANCHOR_PLACEHOLDER_DETECTED = False
        with caplog.at_level("WARNING", logger="genesis.license"):
            L._maybe_log_placeholder_warning()
            n_first = len([r for r in caplog.records if "PLACEHOLDER" in r.message])
        with caplog.at_level("WARNING", logger="genesis.license"):
            L._maybe_log_placeholder_warning()
            n_second = len([r for r in caplog.records if "PLACEHOLDER" in r.message])
        # n_second includes n_first, increment should be 0
        assert n_second == n_first


# ─── update_license_py() ─────────────────────────────────────────────────


class TestUpdateLicenseScript:
    def test_update_license_replaces_pubkey(self, cryptography_available, generator_module, tmp_path, monkeypatch):
        # Mock LICENSE_PY to point at a tmp file
        mock_license = tmp_path / "license.py"
        mock_license.write_text(
            'something\n'
            '_TRUST_ANCHOR_PUBKEY_B64URL = (\n'
            '    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  # placeholder\n'
            ')\n'
            'more code\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(generator_module, "LICENSE_PY", mock_license)
        new_pub = "B" * 43
        ok = generator_module.update_license_py(new_pub)
        assert ok is True
        contents = mock_license.read_text()
        assert new_pub in contents
        # Old placeholder gone
        assert '"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"' not in contents

    def test_update_license_handles_missing_file(self, generator_module, tmp_path, monkeypatch, capsys):
        nonexistent = tmp_path / "no.py"
        monkeypatch.setattr(generator_module, "LICENSE_PY", nonexistent)
        ok = generator_module.update_license_py("B" * 43)
        assert ok is False


# ─── End-to-end via main() ──────────────────────────────────────────────


class TestMainCli:
    """The private key must never reach stdout without an explicit
    opt-in. Default behaviour: `main([])` refuses to generate the
    keypair when no destination for the private half is specified."""

    def test_no_destination_refuses(self, generator_module, capsys):
        """`main([])` without --out and without --print-private exits
        with rc=3 and writes the error to stderr.

        Independent of cryptography — the guard fires before
        generate_keypair() is called."""
        rc = generator_module.main([])
        assert rc == 3
        err = capsys.readouterr().err
        assert "nowhere to go" in err
        assert "--out" in err and "--print-private" in err

    def test_out_only_emits_public_in_stdout(self, cryptography_available, generator_module, capsys, tmp_path):
        """`--out` without `--print-private` → public on stdout, private
        is written to the file ONLY."""
        path = tmp_path / "priv.key"
        rc = generator_module.main(["--out", str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "PUBLIC  KEY" in out
        # The private key must NOT appear in stdout
        assert "PRIVATE KEY" not in out
        # But the file must contain it
        assert path.is_file()
        assert (path.stat().st_mode & 0o777) == 0o600
        # File content: base64url 43 chars (plus newline)
        priv_in_file = path.read_text().strip()
        assert len(priv_in_file) == 43
        # And that string must not appear anywhere in stdout
        assert priv_in_file not in out

    def test_print_private_emits_both(self, cryptography_available, generator_module, capsys):
        """`--print-private` is the explicit opt-in: both public and
        private end up in stdout."""
        rc = generator_module.main(["--print-private"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "PUBLIC  KEY" in out
        assert "PRIVATE KEY" in out

    def test_out_plus_print_private_combination(self, cryptography_available, generator_module, capsys, tmp_path):
        """`--out PATH --print-private` activates both sinks (file +
        stdout)."""
        path = tmp_path / "priv.key"
        rc = generator_module.main(["--out", str(path), "--print-private"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "PRIVATE KEY" in out
        assert path.is_file()

    def test_quiet_with_out_keeps_security_default(self, cryptography_available, generator_module, capsys, tmp_path):
        """`--quiet` does not change exposure rules — it only suppresses
        the explanatory banners."""
        path = tmp_path / "priv.key"
        rc = generator_module.main(["--out", str(path), "--quiet"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "NEXT STEPS" not in out
        assert "PUBLIC" in out
        # The private key still must not leak to stdout
        assert "PRIVATE KEY" not in out
        priv_in_file = path.read_text().strip()
        assert priv_in_file not in out
