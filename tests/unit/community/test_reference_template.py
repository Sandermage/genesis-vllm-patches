# SPDX-License-Identifier: Apache-2.0
"""Reference template contract — `plugins/community/_template/` must:

1. Exist (committed reference layout for `sndr community new-patch`).
2. Be SKIPPED by filesystem discovery (parent dir prefix `_` excludes it).
3. Validate cleanly via the loader API (so the example doesn't bit-rot).

This pins the contract that the audit-community gate enforces on
release: an empty `plugins/community/` (no non-underscore authors)
plus a working template = green release-tier gate.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_DIR = REPO_ROOT / "plugins" / "community" / "_template"
TEMPLATE_PATCH = TEMPLATE_DIR / "PN999"


class TestTemplateLayout:
    def test_directory_present(self):
        assert TEMPLATE_DIR.is_dir(), f"missing {TEMPLATE_DIR}"

    def test_readme_present(self):
        assert (TEMPLATE_DIR / "README.md").is_file()

    def test_pn999_skeleton_present(self):
        assert TEMPLATE_PATCH.is_dir()
        assert (TEMPLATE_PATCH / "manifest.yaml").is_file()
        assert (TEMPLATE_PATCH / "patch.py").is_file()
        assert (TEMPLATE_PATCH / "__init__.py").is_file()
        tests_dir = TEMPLATE_PATCH / "tests"
        assert tests_dir.is_dir()
        assert (tests_dir / "test_pn999.py").is_file()


class TestDiscoverySkipsTemplate:
    def test_filesystem_discovery_skips_underscore_dir(self):
        from sndr.community.discovery import discover_filesystem

        root = REPO_ROOT / "plugins" / "community"
        found = list(discover_filesystem(root))
        # Discovery must NOT pull anything from _template/
        for _path, m in found:
            assert "_template" not in m.namespace, (
                f"discovery surfaced _template plugin: {m.namespace}/{m.id}"
            )

    def test_discover_all_skips_template(self):
        from sndr.community.discovery import discover_all

        root = REPO_ROOT / "plugins" / "community"
        found = list(discover_all(root=root))
        for m in found:
            assert "_template" not in m.namespace


class TestTemplateManifestLoadable:
    """The skeleton manifest must remain loadable so the example
    doesn't silently bit-rot when the PatchManifest schema evolves."""

    def test_manifest_yaml_parses(self):
        from sndr.community.manifest import load_manifest

        manifest = load_manifest(TEMPLATE_PATCH / "manifest.yaml")
        assert manifest.id == "PN999"
        assert manifest.publish_state == "draft", (
            "template manifest MUST be publish_state=draft so an accidental "
            "move out from under _template/ stays out of release registry"
        )


class TestReleaseRegistryClean:
    """The release registry — plugins/community/*/<id>/ for any
    non-underscore author — must be empty by default. Operators who
    contribute land their patches via PR, never by being pre-shipped."""

    def test_no_real_authors_in_default_tree(self):
        root = REPO_ROOT / "plugins" / "community"
        if not root.is_dir():
            pytest.skip("plugins/community/ not present in this checkout")
        for child in root.iterdir():
            if child.is_file():
                continue
            assert child.name.startswith("_"), (
                f"unexpected non-underscore author dir in release tree: "
                f"{child.name} — community patches land via PR, not as "
                f"baseline content"
            )
