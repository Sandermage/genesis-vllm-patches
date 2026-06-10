# SPDX-License-Identifier: Apache-2.0
"""Tests for `plugin_signature` — Wave 4.3."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from sndr.compat.plugin_signature import (
    PluginManifest,
    VerificationResult,
    classify_sandbox,
    verify_file_hashes,
    verify_plugin,
    verify_signature,
)


# ─── Manifest schema ─────────────────────────────────────────────────────


class TestPluginManifest:
    def _valid_dict(self) -> dict:
        return {
            "schema_version": 1,
            "name": "x",
            "version": "0.1.0",
            "files": {"a.py": "0" * 64},
            "patches_modules": ["vllm.something"],
        }

    def test_canonical_construction(self):
        m = PluginManifest.from_dict(self._valid_dict())
        assert m.name == "x"
        assert m.files == {"a.py": "0" * 64}

    def test_rejects_wrong_schema(self):
        d = self._valid_dict()
        d["schema_version"] = 2
        with pytest.raises(ValueError, match="schema_version must be 1"):
            PluginManifest.from_dict(d)

    def test_rejects_missing_required(self):
        d = self._valid_dict()
        del d["files"]
        with pytest.raises(ValueError, match="missing required field"):
            PluginManifest.from_dict(d)

    def test_rejects_bad_hash_format(self):
        d = self._valid_dict()
        d["files"] = {"a.py": "not-hex"}
        with pytest.raises(ValueError, match="not sha256 hex"):
            PluginManifest.from_dict(d)

    def test_canonical_json_deterministic(self):
        d1 = {
            "schema_version": 1, "name": "x", "version": "0.1",
            "files": {"b.py": "0"*64, "a.py": "1"*64},
            "patches_modules": ["m1", "m2"],
        }
        d2 = {
            "patches_modules": ["m1", "m2"],
            "files": {"a.py": "1"*64, "b.py": "0"*64},
            "version": "0.1", "name": "x", "schema_version": 1,
        }
        m1 = PluginManifest.from_dict(d1)
        m2 = PluginManifest.from_dict(d2)
        assert m1.to_canonical_json_unsigned() == m2.to_canonical_json_unsigned()


# ─── File hash verification ──────────────────────────────────────────────


class TestVerifyFileHashes:
    def test_match(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("hello world", encoding="utf-8")
        h = hashlib.sha256(b"hello world").hexdigest()
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {"a.py": h}, "patches_modules": [],
        })
        ok, violations = verify_file_hashes(tmp_path, m)
        assert ok
        assert violations == []

    def test_mismatch(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("real content", encoding="utf-8")
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {"a.py": "0" * 64}, "patches_modules": [],
        })
        ok, violations = verify_file_hashes(tmp_path, m)
        assert not ok
        assert len(violations) == 1
        assert "hash mismatch" in violations[0]

    def test_missing_file(self, tmp_path):
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {"missing.py": "0" * 64}, "patches_modules": [],
        })
        ok, violations = verify_file_hashes(tmp_path, m)
        assert not ok
        assert "missing" in violations[0]


# ─── Sandbox classification ──────────────────────────────────────────────


class TestSandboxClassification:
    def test_safe(self):
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {}, "patches_modules": ["my_module.utils"],
        })
        level, reasons = classify_sandbox(m)
        assert level == "safe"

    def test_moderate(self):
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {}, "patches_modules": ["vllm.entrypoints.openai"],
        })
        level, reasons = classify_sandbox(m)
        assert level == "moderate"

    def test_risky_sndr_core(self):
        # v12: Genesis core moved from `vllm.sndr_core` to the canonical
        # `sndr.` namespace — patching it must stay sandbox-protected.
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {}, "patches_modules": ["sndr.dispatcher"],
        })
        level, reasons = classify_sandbox(m)
        assert level == "risky"
        assert any("protected: sndr." in r for r in reasons)

    def test_risky_v11_shim_namespace_still_protected(self):
        # The archived v11 shim namespace stays protected during the
        # v12.x backward-compat window (commit 6bf9c04c).
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {}, "patches_modules": ["vllm.sndr_core.dispatcher"],
        })
        level, reasons = classify_sandbox(m)
        assert level == "risky"
        assert any("sndr_core" in r for r in reasons)

    def test_risky_kv_cache(self):
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {}, "patches_modules": ["vllm.v1.core.block_pool"],
        })
        level, reasons = classify_sandbox(m)
        assert level == "risky"


# ─── Signature verification ──────────────────────────────────────────────


class TestSignatureVerification:
    def test_unsigned_returns_no_signature(self):
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {}, "patches_modules": [],
        })
        ok, status = verify_signature(m)
        assert not ok
        assert status == "no_signature"

    def test_unsupported_algo(self):
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {}, "patches_modules": [],
            "signature_algo": "rsa", "signature": "abc",
            "signature_key_id": "k",
        })
        ok, status = verify_signature(m)
        assert not ok
        assert "unsupported_algo" in status

    def test_no_trust_anchors(self, monkeypatch, tmp_path):
        # Point trust anchors path at non-existent
        monkeypatch.setenv("HOME", str(tmp_path))
        m = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {}, "patches_modules": [],
            "signature_algo": "ed25519",
            "signature_key_id": "k",
            "signature": "abc",
        })
        ok, status = verify_signature(m)
        assert not ok
        # status could be no_trust_anchors or unknown_key_id
        assert status in ("no_trust_anchors", "unknown_key_id")


class TestSignatureRoundtrip:
    """End-to-end: sign + verify with a real ed25519 key."""

    @pytest.fixture
    def cryptography_available(self):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
            return True
        except ImportError:
            pytest.skip("cryptography not installed")

    def test_full_roundtrip(self, cryptography_available, tmp_path, monkeypatch):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        from cryptography.hazmat.primitives import serialization

        # Generate keypair
        priv = Ed25519PrivateKey.generate()
        pub = priv.public_key()
        pub_bytes = pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        pub_b64 = base64.urlsafe_b64encode(pub_bytes).decode("ascii").rstrip("=")

        # Plant trust anchors
        monkeypatch.setenv("HOME", str(tmp_path))
        anchors_dir = tmp_path / ".sndr"
        anchors_dir.mkdir()
        anchors_file = anchors_dir / "plugin_trust_anchors.json"
        anchors_file.write_text(json.dumps({"test-key": pub_b64}))

        # Build + sign manifest
        unsigned = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {}, "patches_modules": [],
        })
        payload = unsigned.to_canonical_json_unsigned()
        sig = priv.sign(payload)
        sig_b64 = base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")

        signed = PluginManifest.from_dict({
            "schema_version": 1, "name": "x", "version": "0",
            "files": {}, "patches_modules": [],
            "signature_algo": "ed25519",
            "signature_key_id": "test-key",
            "signature": sig_b64,
        })
        ok, status = verify_signature(signed)
        assert ok
        assert status == "verified"


# ─── Top-level verify_plugin() ───────────────────────────────────────────


class TestVerifyPluginEnd2End:
    def _setup_unsigned_safe_plugin(self, tmp_path: Path) -> Path:
        """Create a tiny plugin dir with valid manifest + matching files."""
        plugin = tmp_path / "myplugin"
        plugin.mkdir()
        (plugin / "main.py").write_text("# plugin code", encoding="utf-8")
        h = hashlib.sha256(b"# plugin code").hexdigest()
        manifest = {
            "schema_version": 1,
            "name": "myplugin", "version": "0.1.0",
            "files": {"main.py": h},
            "patches_modules": ["my_external_module.something"],
        }
        (plugin / "genesis-plugin-manifest.json").write_text(json.dumps(manifest))
        return plugin

    def test_unsigned_safe_plugin_passes(self, tmp_path, monkeypatch):
        # No strict mode → unsigned is OK
        monkeypatch.delenv("SNDR_PLUGIN_STRICT_SIGNATURES", raising=False)
        plugin = self._setup_unsigned_safe_plugin(tmp_path)
        result = verify_plugin(plugin)
        assert result.ok
        assert result.level == "safe"

    def test_strict_mode_unsigned_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SNDR_PLUGIN_STRICT_SIGNATURES", "1")
        plugin = self._setup_unsigned_safe_plugin(tmp_path)
        result = verify_plugin(plugin)
        assert not result.ok
        assert "strict mode" in result.summary or "signature" in result.summary

    def test_missing_manifest_warn_only(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SNDR_PLUGIN_STRICT_SIGNATURES", raising=False)
        empty = tmp_path / "empty"
        empty.mkdir()
        result = verify_plugin(empty)
        # Warn-only mode: passes
        assert result.ok
        assert "not found" in result.summary

    def test_missing_manifest_strict_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SNDR_PLUGIN_STRICT_SIGNATURES", "1")
        empty = tmp_path / "empty"
        empty.mkdir()
        result = verify_plugin(empty)
        assert not result.ok

    def test_file_tampering_detected(self, tmp_path, monkeypatch):
        plugin = self._setup_unsigned_safe_plugin(tmp_path)
        # Tamper with main.py AFTER manifest is written
        (plugin / "main.py").write_text("# TAMPERED", encoding="utf-8")
        result = verify_plugin(plugin)
        assert not result.ok
        assert "hash mismatch" in result.summary or "file" in result.summary

    def test_risky_plugin_rejected_without_override(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SNDR_PLUGIN_ALLOW_CORE", raising=False)
        plugin = tmp_path / "risky"
        plugin.mkdir()
        manifest = {
            "schema_version": 1,
            "name": "evil", "version": "0",
            "files": {},
            "patches_modules": ["sndr.dispatcher"],
        }
        (plugin / "genesis-plugin-manifest.json").write_text(json.dumps(manifest))
        result = verify_plugin(plugin)
        assert not result.ok
        assert "protected" in result.summary

    def test_risky_plugin_loaded_with_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SNDR_PLUGIN_ALLOW_CORE", "1")
        monkeypatch.delenv("SNDR_PLUGIN_STRICT_SIGNATURES", raising=False)
        plugin = tmp_path / "risky"
        plugin.mkdir()
        manifest = {
            "schema_version": 1,
            "name": "operator-extension", "version": "0",
            "files": {},
            "patches_modules": ["sndr.dispatcher"],
        }
        (plugin / "genesis-plugin-manifest.json").write_text(json.dumps(manifest))
        result = verify_plugin(plugin)
        # Override allows it to pass even though risky
        assert result.ok
