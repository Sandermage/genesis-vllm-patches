# SPDX-License-Identifier: Apache-2.0
"""Tests for dashboard-ready Product API overview snapshots."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass

from sndr.model_configs.registry_v2 import (
    list_hardware,
    list_models,
    list_presets,
    list_profiles,
)
from sndr.product_api.legacy.overview import (
    collect_catalog_summary,
    collect_product_overview,
)
from sndr.product_api.legacy.types import CatalogSummary, ProductOverview


def _missing_tool(_tool: str):
    return None


def test_catalog_summary_matches_registry_counts():
    summary = collect_catalog_summary()

    assert isinstance(summary, CatalogSummary)
    assert is_dataclass(summary)
    assert summary.models_count == len(list_models())
    assert summary.hardware_count == len(list_hardware())
    assert summary.profiles_count == len(list_profiles())
    assert summary.presets_count == len(list_presets())
    assert (
        summary.preset_cards_count
        + summary.unannotated_presets_count
        + summary.preset_load_error_count
        == summary.presets_count
    )


def test_catalog_summary_is_dict_serializable():
    payload = asdict(collect_catalog_summary())

    assert isinstance(payload["status_counts"], dict)
    assert isinstance(payload["workload_counts"], dict)
    assert isinstance(payload["family_counts"], dict)
    assert isinstance(payload["default_presets"], tuple)
    assert isinstance(payload["preset_load_errors"], tuple)


def test_product_overview_combines_capabilities_and_catalog():
    overview = collect_product_overview(
        which=_missing_tool,
        engine_installed=False,
    )

    assert isinstance(overview, ProductOverview)
    assert overview.capabilities.platform.engine_installed is False
    assert overview.catalog.presets_count == len(list_presets())
    assert asdict(overview)["capabilities"]["features"]
