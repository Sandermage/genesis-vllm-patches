# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_runtime_pins.py` — P2.6 pin harmonization gates.

Covers:
  * R-PIN-1 bare-tag detection: positive + negative cases via
    `_is_bare_mutable` and a synthetic hardware fixture
  * R-PIN-2 digest presence: missing + wrong-prefix + ok
  * R-PIN-3 render parity: live invocation against the real registry
    (must pass on a clean tree)
  * R-PIN-4 ModelDef migration table: ALLOWED_MODELDEF_PINS set
    membership; unknown pin fails; live registry classified

Tests run on the live repo state — they implicitly verify that the
audit script reports the current tree as clean (post-P2.3). If a future
commit drifts the pins, this test file will fail in CI before the
audit script's exit code does.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_runtime_pins.py"


def _import_audit():
    name = "_audit_v2_runtime_pins_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── R-PIN-1 bare-tag predicate ─────────────────────────────────────────


class TestIsBareMutable:
    @pytest.mark.parametrize("image", [
        "vllm/vllm-openai:nightly",
        "vllm/vllm-openai:latest",
        "vllm/vllm-openai:main",
        "vllm/vllm-openai:stable",
        "vllm/vllm-openai:dev",
        "vllm/vllm-openai",  # no tag at all
    ])
    def test_bare_mutable_detected(self, image):
        mod = _import_audit()
        assert mod._is_bare_mutable(image), (
            f"{image!r} should be detected as a bare mutable tag"
        )

    @pytest.mark.parametrize("image", [
        "vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036",
        "vllm/vllm-openai:nightly-bf0d2dc6d764f7ab1a69504f60a55883ec6d9b39",
        "vllm/vllm-openai:v0.21.0",
        "vllm/vllm-openai:0.20.2rc1",
        ("vllm/vllm-openai@sha256:"
         "7f047b7e625283eee436cfc0c37784064f75422452ed4f9b6fa8c69eae6afe68"),
    ])
    def test_pinned_image_passes(self, image):
        mod = _import_audit()
        assert not mod._is_bare_mutable(image), (
            f"{image!r} should be accepted as explicitly pinned"
        )


# ─── R-PIN-1 / R-PIN-2 on the live hardware tree ───────────────────────


class TestLiveHardwareInvariants:
    def test_r_pin_1_clean(self):
        mod = _import_audit()
        issues = mod.check_r_pin_1_no_mutable_nightly()
        assert issues == [], (
            f"R-PIN-1 must be clean on the live tree (post-P2.3); "
            f"got: {issues}"
        )

    def test_r_pin_2_clean(self):
        mod = _import_audit()
        issues = mod.check_r_pin_2_digest_present()
        assert issues == [], (
            f"R-PIN-2 must be clean on the live tree (post-P2.3); "
            f"got: {issues}"
        )


# ─── R-PIN-1 / R-PIN-2 against synthetic fixtures ───────────────────────


_SYNTHETIC_BARE = """\
schema_version: 2
kind: hardware
id: test-bare-nightly
title: Synthetic bare nightly
maintainer: test
hardware:
  vendor: nvidia
  family: a5000
  n_gpus: 2
  min_vram_per_gpu_mib: 24576
runtime:
  default: docker
  supported: [docker]
  docker:
    image: vllm/vllm-openai:nightly
    image_digest: vllm/vllm-openai@sha256:0000000000000000000000000000000000000000000000000000000000000000
"""

_SYNTHETIC_MISSING_DIGEST = """\
schema_version: 2
kind: hardware
id: test-missing-digest
title: Synthetic missing digest
maintainer: test
hardware:
  vendor: nvidia
  family: a5000
  n_gpus: 2
  min_vram_per_gpu_mib: 24576
runtime:
  default: docker
  supported: [docker]
  docker:
    image: vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036
"""

_SYNTHETIC_WRONG_DIGEST_PREFIX = """\
schema_version: 2
kind: hardware
id: test-wrong-digest
title: Synthetic wrong digest prefix
maintainer: test
hardware:
  vendor: nvidia
  family: a5000
  n_gpus: 2
  min_vram_per_gpu_mib: 24576
runtime:
  default: docker
  supported: [docker]
  docker:
    image: vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036
    image_digest: just-a-sha-no-repo
"""


class TestSyntheticFixtures:
    def test_r_pin_1_flags_bare_synthetic(self, tmp_path, monkeypatch):
        mod = _import_audit()
        fake_hw = tmp_path / "hardware"
        fake_hw.mkdir()
        (fake_hw / "test-bare.yaml").write_text(_SYNTHETIC_BARE)
        monkeypatch.setattr(mod, "HARDWARE_DIR", fake_hw)
        issues = mod.check_r_pin_1_no_mutable_nightly()
        assert len(issues) == 1
        assert "test-bare.yaml" in issues[0]
        assert "bare mutable tag" in issues[0]

    def test_r_pin_2_flags_missing_digest(self, tmp_path, monkeypatch):
        mod = _import_audit()
        fake_hw = tmp_path / "hardware"
        fake_hw.mkdir()
        (fake_hw / "test-missing.yaml").write_text(_SYNTHETIC_MISSING_DIGEST)
        monkeypatch.setattr(mod, "HARDWARE_DIR", fake_hw)
        issues = mod.check_r_pin_2_digest_present()
        assert len(issues) == 1
        assert "image_digest is missing" in issues[0]

    def test_r_pin_2_flags_wrong_digest_prefix(
        self, tmp_path, monkeypatch,
    ):
        mod = _import_audit()
        fake_hw = tmp_path / "hardware"
        fake_hw.mkdir()
        (fake_hw / "test-wrong.yaml").write_text(
            _SYNTHETIC_WRONG_DIGEST_PREFIX
        )
        monkeypatch.setattr(mod, "HARDWARE_DIR", fake_hw)
        issues = mod.check_r_pin_2_digest_present()
        assert len(issues) == 1
        assert "does not begin with" in issues[0]


# ─── R-PIN-3 live render parity ─────────────────────────────────────────


class TestRenderParity:
    def test_r_pin_3_clean_on_live_tree(self):
        mod = _import_audit()
        errors, infos = mod.check_r_pin_3_render_parity()
        assert errors == [], (
            f"R-PIN-3 must be clean on the live tree; got errors: {errors}"
        )
        # Every (profile, hardware) representative should produce one info
        # line confirming the equality.
        assert len(infos) == len(mod.REPRESENTATIVE_RENDERS), (
            f"expected one info per representative render; "
            f"got {len(infos)} for {len(mod.REPRESENTATIVE_RENDERS)} "
            f"renders"
        )
        for info in infos:
            assert "rendered = composed" in info


# ─── R-PIN-4 ModelDef pin migration ────────────────────────────────────


class TestModelDefMigration:
    def test_allowed_pin_set_explicit(self):
        mod = _import_audit()
        # Both currently-live pins must be present in the allowed set.
        assert "0.20.2rc1.dev338+gbf0d2dc6d" in mod.ALLOWED_MODELDEF_PINS
        assert "0.20.2rc1.dev371+gbf610c2f5" in mod.ALLOWED_MODELDEF_PINS

    def test_r_pin_4_clean_on_live_tree(self):
        mod = _import_audit()
        errors, infos = mod.check_r_pin_4_modeldef_migration()
        assert errors == [], (
            f"R-PIN-4 must be clean on the live tree; got: {errors}"
        )
        # Migration table must mention both Gemma and Qwen families
        # in the infos.
        joined = "\n".join(infos)
        assert "gemma" in joined and "qwen" in joined

    def test_r_pin_4_flags_unknown_pin(self, tmp_path, monkeypatch):
        mod = _import_audit()
        fake_models = tmp_path / "model"
        fake_models.mkdir()
        (fake_models / "qwen3.6-fake.yaml").write_text(
            "schema_version: 2\n"
            "kind: model\n"
            "id: fake\n"
            "title: Fake model\n"
            "maintainer: test\n"
            "versions:\n"
            "  vllm_pin_required: 0.99.9rc1.dev999+gunknown\n"
        )
        monkeypatch.setattr(mod, "MODEL_DIR", fake_models)
        errors, _ = mod.check_r_pin_4_modeldef_migration()
        assert len(errors) == 1
        assert "not in the allowed set" in errors[0]

    def test_r_pin_4_flags_missing_pin(self, tmp_path, monkeypatch):
        mod = _import_audit()
        fake_models = tmp_path / "model"
        fake_models.mkdir()
        (fake_models / "gemma-stub.yaml").write_text(
            "schema_version: 2\n"
            "kind: model\n"
            "id: stub\n"
            "title: Stub model\n"
            "maintainer: test\n"
        )
        monkeypatch.setattr(mod, "MODEL_DIR", fake_models)
        errors, _ = mod.check_r_pin_4_modeldef_migration()
        assert len(errors) == 1
        assert "vllm_pin_required is missing" in errors[0]


# ─── CLI driver ─────────────────────────────────────────────────────────


class TestCli:
    def test_default_invocation_exits_clean(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"audit must exit 0 on clean tree; got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "V2 runtime / ModelDef pin harmonization audit" in result.stdout
        assert "All selected rules clean" in result.stdout

    def test_json_output_parseable(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert set(payload.keys()) >= {
            "R-PIN-1", "R-PIN-2", "R-PIN-3", "R-PIN-4", "_summary",
        }
        assert payload["_summary"]["violations_total"] == 0
        for rule in ("R-PIN-1", "R-PIN-2", "R-PIN-3", "R-PIN-4"):
            assert payload[rule]["status"] == "pass"

    def test_single_rule_filter(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--rule", "R-PIN-1", "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "R-PIN-1" in payload
        # Other rules must not be present when one is filtered
        assert "R-PIN-3" not in payload

    def test_verbose_shows_infos(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--verbose"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # Verbose must surface the migration table
        assert "P2.4d candidate" in result.stdout or (
            "→ dev338" in result.stdout
        )
