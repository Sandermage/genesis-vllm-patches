# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_runtime_image_pin.py` — Entry 34."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_runtime_image_pin.py"


def _import():
    name = "_audit_v2_runtime_image_pin_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(p: Path, text: str) -> Path:
    p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return p


_GOOD_DIGEST = ("vllm/vllm-openai@sha256:"
                "9b534fe66daf152e8ceca8a7f8e14c18105aaf6ddabc61eb17730d85b4c7c194")


def _hw_yaml(image="'vllm/vllm-openai:nightly'",
             digest=f"'{_GOOD_DIGEST}'") -> str:
    return textwrap.dedent(f"""
        id: synth
        kind: hardware
        runtime:
          docker:
            image: {image}
            image_digest: {digest}
    """).lstrip("\n")


class TestRegex:
    def test_canonical_matches(self):
        mod = _import()
        assert mod.DIGEST_RE.match(_GOOD_DIGEST)

    def test_short_hash_fails(self):
        mod = _import()
        assert not mod.DIGEST_RE.match("vllm/vllm-openai@sha256:abc123")

    def test_missing_at_sign_fails(self):
        mod = _import()
        assert not mod.DIGEST_RE.match("vllm/vllm-openai-sha256:" + "a"*64)


class TestCheckOneHardware:
    def test_canonical_passes(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml())
        r = mod.check_one_hardware(y)
        assert r.passed is True

    def test_missing_image_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", textwrap.dedent(f"""
            id: synth
            kind: hardware
            runtime:
              docker:
                image_digest: '{_GOOD_DIGEST}'
        """))
        r = mod.check_one_hardware(y)
        assert r.passed is False

    def test_missing_digest_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", """
            id: synth
            kind: hardware
            runtime:
              docker:
                image: 'foo:bar'
        """)
        r = mod.check_one_hardware(y)
        assert r.passed is False

    def test_malformed_digest_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml(digest="'not-a-digest'"))
        r = mod.check_one_hardware(y)
        assert r.passed is False


class TestLiveRepo:
    def test_committed_clean(self):
        mod = _import()
        results = mod.audit_v2_runtime_image_pin()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.hardware_id}: {r.violations}" for r in failed
        )
        assert len(results) == 3


class TestScriptCLI:
    def test_cli_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_json(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "digest_regex" in payload
