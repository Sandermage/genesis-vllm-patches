# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.runtime.redact` — DA-018 / report bundle support."""
from __future__ import annotations

import pytest

from sndr.runtime.redact import (
    DEFAULT_RULES,
    RedactRule,
    Redactor,
    redact,
    redact_dict,
)


class TestRedactDefaults:
    def test_ipv4_redacted(self):
        out = redact("connect to 192.168.1.50 for the API")
        assert "192.168.1.50" not in out
        assert "<IP>" in out

    def test_ipv4_inside_url(self):
        out = redact("http://192.168.1.50:8000/v1/models")
        assert "<IP>" in out
        assert "192.168.1.50" not in out

    def test_ssh_target_preserves_user(self):
        out = redact("ssh user@192.168.1.50")
        assert "user" in out
        assert "192.168.1.50" not in out
        assert "<HOSTNAME>" in out

    def test_ssh_target_with_dns_hostname(self):
        out = redact("ssh ops@gpu-server-01.internal.example.com")
        assert "ops@<HOSTNAME>" in out

    def test_bearer_token_redacted(self):
        out = redact("Authorization: Bearer abc123XYZ.def456")
        assert "abc123XYZ.def456" not in out
        assert "<REDACTED>" in out

    def test_env_api_key(self):
        out = redact("GENESIS_API_KEY=secret123 SNDR_ENGINE_LICENSE_KEY=xyz")
        assert "secret123" not in out
        assert "xyz" not in out
        assert "GENESIS_API_KEY=<REDACTED>" in out
        assert "SNDR_ENGINE_LICENSE_KEY=<REDACTED>" in out

    def test_hf_token(self):
        out = redact("HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456")
        assert "hf_abc" not in out
        # Either env-key rule or hf_token rule should mask it.
        assert "REDACTED" in out or "<HF_TOKEN>" in out

    def test_license_token(self):
        # base64url payload + signature
        token = ("eyJjdXN0b21lcl9pZCI6InNhbmRlci10ZXN0IiwiaXNzdWVkX2F0IjoxNzMwMDAwMDAwfQ"
                 ".MEUCIQDxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        out = redact(f"SNDR_ENGINE_LICENSE_KEY={token}")
        # The env-key rule fires first, masking everything after `=`
        assert token not in out
        assert "<REDACTED>" in out or "<LICENSE_TOKEN>" in out

    def test_email_redacted(self):
        out = redact("contact sander.odessa@gmail.com")
        assert "sander.odessa@gmail.com" not in out
        assert "<EMAIL>" in out or "<HOSTNAME>" in out  # ssh_target may catch it

    def test_home_path_user(self):
        out = redact("/home/sander/.sndr/host.yaml")
        assert "/home/sander" not in out
        assert "/home/<USER>" in out

    def test_macos_users_path(self):
        out = redact("/Users/sander/Documents/code")
        assert "/Users/sander" not in out
        assert "/Users/<USER>" in out


class TestRedactPreserves:
    def test_model_path_preserved(self):
        # Public model paths are operationally important context.
        out = redact("/models/Qwen3.6-35B-A3B-FP8")
        assert "Qwen3.6-35B-A3B-FP8" in out
        assert "/models/" in out

    def test_image_tag_preserved(self):
        out = redact("vllm/vllm-openai:nightly")
        assert "vllm/vllm-openai:nightly" in out

    def test_image_digest_preserved(self):
        digest = "sha256:abc123def456"
        out = redact(f"image: {digest}")
        assert digest in out

    def test_patch_id_preserved(self):
        out = redact("PN82 mamba prefill zero rows")
        assert "PN82" in out

    def test_version_constant_preserved(self):
        out = redact("SNDR_CORE_VERSION = 11.0.0")
        assert "11.0.0" in out

    def test_generic_paths_preserved(self):
        out = redact("/var/lib/genesis /opt/models /tmp/foo")
        assert "/var/lib/genesis" in out
        assert "/opt/models" in out
        assert "/tmp/foo" in out


class TestRedactor:
    def test_counts_track_hits(self):
        r = Redactor()
        text = "192.168.1.50 and 10.0.0.1 and another 172.16.0.5"
        r.redact(text)
        assert r.counts["ipv4"] == 3

    def test_counts_accumulate_across_calls(self):
        r = Redactor()
        r.redact("192.168.1.50")
        r.redact("10.0.0.1")
        assert r.counts["ipv4"] == 2

    def test_reset_counts(self):
        r = Redactor()
        r.redact("192.168.1.50")
        assert r.counts["ipv4"] == 1
        r.reset_counts()
        assert "ipv4" not in r.counts

    def test_empty_input(self):
        assert redact("") == ""
        assert redact(None or "") == ""

    def test_no_matches_unchanged(self):
        text = "Genesis vLLM Patches v11.0.0 — patches: 131"
        assert redact(text) == text


class TestRedactDict:
    def test_string_leaf_redacted(self):
        d = {"endpoint": "http://192.168.1.50:8000"}
        out = redact_dict(d)
        assert "192.168.1.50" not in out["endpoint"]

    def test_nested_structure(self):
        d = {
            "config": {
                "ssh": "ssh user@192.168.1.50",
                "model": "/models/Qwen3.6-35B-A3B-FP8",
            },
            "logs": ["192.168.1.50:8000", "/Users/sander/.sndr"],
        }
        out = redact_dict(d)
        assert "user@" in out["config"]["ssh"]
        assert "<HOSTNAME>" in out["config"]["ssh"]
        # Model path preserved
        assert "Qwen3.6-35B-A3B-FP8" in out["config"]["model"]
        # IPs in list redacted
        assert "192.168.1.50" not in out["logs"][0]
        # User path masked
        assert "/Users/<USER>" in out["logs"][1]

    def test_non_string_leaves_preserved(self):
        d = {"count": 42, "ratio": 0.95, "active": True, "list": [1, 2, 3]}
        out = redact_dict(d)
        assert out == d


class TestCustomRules:
    def test_custom_rule_added(self):
        custom = RedactRule(
            name="corp_subnet",
            pattern=__import__("re").compile(r"\b10\.42\.\d+\.\d+\b"),
            replacement="<CORP_IP>",
        )
        out = redact("server at 10.42.5.100", rules=[custom])
        assert "<CORP_IP>" in out
        assert "10.42.5.100" not in out

    def test_default_rules_immutable(self):
        # DEFAULT_RULES is a tuple — not mutable.
        with pytest.raises(AttributeError):
            DEFAULT_RULES.append(None)  # type: ignore[attr-defined]
