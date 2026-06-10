# SPDX-License-Identifier: Apache-2.0
"""Tests for DockerConfig.image_digest + --strict-image launcher gate.

T1.6 (audit closure 2026-05-09 / production roadmap §7.4).

Two layers:

  1. Schema validation — `image_digest` accepts canonical sha256
     digest references and rejects tag-only refs.
  2. Launcher integration — `_verify_image_digest()` enforces the
     pin per --strict-image policy without actually running docker.
     We mock subprocess.run to simulate "match" / "miss" / "missing".
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import pytest

from sndr.model_configs.schema import (
    DockerConfig,
    SchemaError,
)


# ─── DockerConfig.image_digest schema ────────────────────────────────────


class TestDockerConfigImageDigest:
    def test_default_is_none(self):
        d = DockerConfig(
            image="vllm/vllm-openai:nightly",
            container_name="vllm-server",
            port=8000,
        )
        assert d.image_digest is None
        d.validate()  # must not raise

    def test_accepts_canonical_digest(self):
        d = DockerConfig(
            image="vllm/vllm-openai:nightly",
            container_name="vllm-server",
            port=8000,
            image_digest=(
                "vllm/vllm-openai@sha256:"
                "0123456789abcdef0123456789abcdef"
                "0123456789abcdef0123456789abcdef"
            ),
        )
        d.validate()  # must not raise

    def test_rejects_tag_only_ref(self):
        with pytest.raises(SchemaError, match="must include '@sha256:'"):
            DockerConfig(
                image="vllm/vllm-openai:nightly",
                container_name="vllm-server",
                port=8000,
                image_digest="vllm/vllm-openai:v0.10",
            ).validate()

    def test_rejects_non_string(self):
        with pytest.raises(SchemaError, match="must be a string"):
            DockerConfig(
                image="vllm/vllm-openai:nightly",
                container_name="vllm-server",
                port=8000,
                image_digest=12345,  # type: ignore[arg-type]
            ).validate()


# ─── Launcher integration ────────────────────────────────────────────────


def _make_cfg(image: str = "vllm/vllm-openai:nightly",
              digest: str | None = None):
    docker = DockerConfig(
        image=image,
        container_name="vllm-server",
        port=8000,
        image_digest=digest,
    )
    return SimpleNamespace(docker=docker)


class TestVerifyImageDigest:
    def test_off_mode_skips(self, monkeypatch):
        from sndr.cli.legacy.launch import _verify_image_digest
        cfg = _make_cfg(digest="vllm/vllm-openai@sha256:" + "a" * 64)
        # docker shouldn't even be probed
        called = []

        def _fake_run(*args, **kwargs):
            called.append(args)
            raise RuntimeError("should not be called")

        monkeypatch.setattr("subprocess.run", _fake_run)
        assert _verify_image_digest(cfg, "off") == 0
        assert not called

    def test_no_docker_block_returns_zero(self):
        from sndr.cli.legacy.launch import _verify_image_digest
        cfg = SimpleNamespace(docker=None)
        assert _verify_image_digest(cfg, "on") == 0

    def test_auto_mode_no_digest_passes(self):
        from sndr.cli.legacy.launch import _verify_image_digest
        cfg = _make_cfg(digest=None)
        assert _verify_image_digest(cfg, "auto") == 0

    def test_strict_on_no_digest_fails(self, capsys):
        from sndr.cli.legacy.launch import _verify_image_digest
        cfg = _make_cfg(digest=None)
        rc = _verify_image_digest(cfg, "on")
        assert rc == 2
        # Operator-facing error must explain how to fix
        captured = capsys.readouterr()
        assert "image_digest" in captured.out + captured.err

    def test_match_passes(self, monkeypatch, capsys):
        from sndr.cli.legacy.launch import _verify_image_digest
        digest = ("vllm/vllm-openai@sha256:"
                  + "abcdef" * 10 + "abcd")
        cfg = _make_cfg(digest=digest)

        # Mock shutil.which → docker exists
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")
        # Mock subprocess.run to return matching digest list
        import json as _json

        def _fake_run(cmd, **kwargs):
            assert "inspect" in cmd
            return SimpleNamespace(
                returncode=0, stdout=_json.dumps([digest]),
                stderr="",
            )

        monkeypatch.setattr("subprocess.run", _fake_run)
        rc = _verify_image_digest(cfg, "auto")
        assert rc == 0
        out = capsys.readouterr().out
        assert "verified" in out.lower() or "ok" in out.lower()

    def test_mismatch_strict_fails(self, monkeypatch, capsys):
        from sndr.cli.legacy.launch import _verify_image_digest
        expected = "vllm/vllm-openai@sha256:" + "a" * 64
        actual = "vllm/vllm-openai@sha256:" + "b" * 64
        cfg = _make_cfg(digest=expected)

        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")
        import json as _json

        def _fake_run(cmd, **kwargs):
            return SimpleNamespace(
                returncode=0, stdout=_json.dumps([actual]),
                stderr="",
            )

        monkeypatch.setattr("subprocess.run", _fake_run)
        rc = _verify_image_digest(cfg, "auto")
        assert rc == 1
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        # Operator-facing error must include the mismatch banner
        assert "MISMATCH" in combined
        assert expected in combined

    def test_docker_missing_auto_passes(self, monkeypatch):
        from sndr.cli.legacy.launch import _verify_image_digest
        cfg = _make_cfg(digest="vllm/vllm-openai@sha256:" + "a" * 64)
        monkeypatch.setattr("shutil.which", lambda _: None)
        # auto mode — degrade gracefully when docker unavailable
        assert _verify_image_digest(cfg, "auto") == 0

    def test_docker_missing_strict_fails(self, monkeypatch):
        from sndr.cli.legacy.launch import _verify_image_digest
        cfg = _make_cfg(digest="vllm/vllm-openai@sha256:" + "a" * 64)
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert _verify_image_digest(cfg, "on") == 2


# ─── argparse wiring ─────────────────────────────────────────────────────


class TestArgparser:
    def test_strict_image_choices(self):
        from sndr.cli.legacy.launch import add_argparser
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        add_argparser(sub)
        ns = parser.parse_args([
            "launch", "some-key", "--strict-image", "on",
        ])
        assert ns.strict_image == "on"

    def test_strict_image_default_auto(self):
        from sndr.cli.legacy.launch import add_argparser
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        add_argparser(sub)
        ns = parser.parse_args(["launch", "some-key"])
        assert ns.strict_image == "auto"

    def test_strict_image_invalid_rejected(self):
        from sndr.cli.legacy.launch import add_argparser
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        add_argparser(sub)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "launch", "some-key", "--strict-image", "bogus",
            ])
