# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/generate_sbom.py` — Wave 4.2.

Covers the SBOM generator's three output formats (CycloneDX 1.5,
SPDX 2.3, plain text) and the `build_payload()` API surface. Doesn't
re-test transitive deps enumeration — that's stdlib `importlib.metadata`
behavior.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Allow `import generate_sbom` from tests
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def sbom_module():
    """Load the generator module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_sbom", SCRIPTS_DIR / "generate_sbom.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Build payload ───────────────────────────────────────────────────────


class TestBuildPayload:
    def test_payload_keys(self, sbom_module):
        p = sbom_module.build_payload()
        for key in (
            "generated_at", "pyproject", "constraints_txt",
            "installed_distributions", "patch_registry",
            "model_configs", "known_good_vllm_pins",
            "image_allowlist", "genesis_modules",
        ):
            assert key in p, f"missing key: {key}"

    def test_pyproject_loaded(self, sbom_module):
        p = sbom_module.build_payload()
        project = p["pyproject"].get("project", {})
        # v12: project renamed vllm-sndr-core -> sndr-platform.
        assert project.get("name") == "sndr-platform"
        assert "version" in project

    def test_patch_registry_snapshot(self, sbom_module):
        p = sbom_module.build_payload()
        pr = p["patch_registry"]
        assert "total" in pr
        assert pr["total"] >= 100  # we have ~131 patches
        assert "by_tier" in pr
        assert "by_lifecycle" in pr

    def test_known_good_vllm_pins_present(self, sbom_module):
        p = sbom_module.build_payload()
        pins = p["known_good_vllm_pins"]
        assert isinstance(pins, list)
        assert len(pins) >= 1

    def test_image_allowlist_present(self, sbom_module):
        p = sbom_module.build_payload()
        al = p["image_allowlist"]
        assert isinstance(al, list)
        # At least the dev93 entry from Wave 4.1
        assert any(
            (entry.get("vllm_pin") == "0.20.2rc1.dev93+g51f22dcfd")
            for entry in al if "vllm_pin" in entry
        )

    def test_genesis_modules_listed(self, sbom_module):
        p = sbom_module.build_payload()
        mods = p["genesis_modules"]
        # All modules have path + sha256
        for m in mods[:10]:
            assert "path" in m
            assert "sha256" in m
            assert len(m["sha256"]) == 64  # full hex

    def test_generated_at_iso_8601(self, sbom_module):
        import datetime
        p = sbom_module.build_payload()
        # Should be parseable
        ts = p["generated_at"].replace("Z", "+00:00")
        datetime.datetime.fromisoformat(ts)


# ─── CycloneDX emitter ───────────────────────────────────────────────────


class TestCycloneDX:
    def test_emit_produces_valid_json(self, sbom_module, tmp_path):
        payload = sbom_module.build_payload()
        out = tmp_path / "test.cdx.json"
        sbom_module.emit_cyclonedx(payload, out)
        assert out.is_file()
        data = json.loads(out.read_text())
        assert data["bomFormat"] == "CycloneDX"
        assert data["specVersion"] == "1.5"
        assert data["serialNumber"].startswith("urn:uuid:")
        assert "components" in data
        assert "metadata" in data

    def test_includes_genesis_properties(self, sbom_module, tmp_path):
        payload = sbom_module.build_payload()
        out = tmp_path / "test.cdx.json"
        sbom_module.emit_cyclonedx(payload, out)
        data = json.loads(out.read_text())
        prop_names = {p["name"] for p in data["properties"]}
        assert "genesis:patch_registry_total" in prop_names
        assert "genesis:vllm_known_good_pins_count" in prop_names
        assert "genesis:known_good_images_count" in prop_names


# ─── SPDX emitter ────────────────────────────────────────────────────────


class TestSPDX:
    def test_emit_produces_valid_json(self, sbom_module, tmp_path):
        payload = sbom_module.build_payload()
        out = tmp_path / "test.spdx.json"
        sbom_module.emit_spdx(payload, out)
        assert out.is_file()
        data = json.loads(out.read_text())
        assert data["spdxVersion"] == "SPDX-2.3"
        assert data["dataLicense"] == "CC0-1.0"
        assert data["SPDXID"] == "SPDXRef-DOCUMENT"
        assert "packages" in data
        # Genesis package always first (v12 project name: sndr-platform)
        assert data["packages"][0]["name"] == "sndr-platform"

    def test_relationships_link_genesis_to_deps(self, sbom_module, tmp_path):
        payload = sbom_module.build_payload()
        out = tmp_path / "test.spdx.json"
        sbom_module.emit_spdx(payload, out)
        data = json.loads(out.read_text())
        for rel in data["relationships"]:
            assert rel["spdxElementId"] == "SPDXRef-Package-Genesis"
            assert rel["relationshipType"] == "DEPENDS_ON"


# ─── Plain text emitter ──────────────────────────────────────────────────


class TestTextEmitter:
    def test_emit_produces_human_readable(self, sbom_module, tmp_path):
        payload = sbom_module.build_payload()
        out = tmp_path / "test.txt"
        sbom_module.emit_text(payload, out)
        text = out.read_text()
        assert "Genesis SBOM" in text
        assert "## Direct dependencies" in text
        assert "## Patch registry snapshot" in text
        assert "## KNOWN_GOOD_IMAGES" in text


# ─── End-to-end via main() ──────────────────────────────────────────────


class TestMain:
    def test_main_creates_all_three_formats(self, sbom_module, tmp_path):
        out_base = tmp_path / "genesis-sbom"
        rc = sbom_module.main(["--out", str(out_base), "--format", "all"])
        assert rc == 0
        assert (tmp_path / "genesis-sbom.cdx.json").is_file()
        assert (tmp_path / "genesis-sbom.spdx.json").is_file()
        assert (tmp_path / "genesis-sbom.txt").is_file()

    def test_main_single_format(self, sbom_module, tmp_path):
        out_base = tmp_path / "x"
        rc = sbom_module.main(["--out", str(out_base), "--format", "cyclonedx"])
        assert rc == 0
        assert (tmp_path / "x.cdx.json").is_file()
        # Other formats NOT created
        assert not (tmp_path / "x.spdx.json").is_file()
