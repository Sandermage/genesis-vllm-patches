# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.bundles`` — M.6.1 / M.6.4.

The canonical bundle catalog lives here. Drift detection against
``tests/bundles/test_stage7_bundles_smoke.py`` runs via
``test_patches_cli.py::TestBundles::test_bundles_catalog_matches_test_smoke``,
which now reads the catalog directly from this module after M.6.4
removed the ``cli.patches._BUNDLES`` back-compat shim.
"""
from __future__ import annotations

from sndr.product_api.legacy.patches import bundles
from sndr.product_api.legacy.patches.types import BundleSpec


class TestCatalog:
    def test_catalog_has_five_entries(self):
        assert len(bundles.BUNDLES_CATALOG) == 5

    def test_catalog_is_tuple_of_quadruples(self):
        assert isinstance(bundles.BUNDLES_CATALOG, tuple)
        for entry in bundles.BUNDLES_CATALOG:
            assert isinstance(entry, tuple)
            assert len(entry) == 4


class TestListBundles:
    def test_returns_specs(self):
        specs = bundles.list_bundles()
        assert len(specs) == 5
        assert all(isinstance(b, BundleSpec) for b in specs)

    def test_has_apply_is_unprobed(self):
        """List path skips the import probe — ``has_apply`` stays
        ``None`` to keep the call cheap."""
        for spec in bundles.list_bundles():
            assert spec.has_apply is None


class TestExplainBundle:
    def test_known_returns_spec(self):
        spec = bundles.explain_bundle("attention_gdn_spec")
        assert spec is not None
        assert spec.name == "attention_gdn_spec"
        assert spec.umbrella_flag == "BUNDLE_ATTENTION_GDN_SPEC"
        # Imported module is real → ``has_apply`` resolved to True.
        assert spec.has_apply is True

    def test_unknown_returns_none(self):
        assert bundles.explain_bundle("not-a-real-bundle") is None
