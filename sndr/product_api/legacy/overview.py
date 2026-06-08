# SPDX-License-Identifier: Apache-2.0
"""Top-level read-only Product API overview for the future GUI dashboard."""
from __future__ import annotations

import warnings
import shutil
from collections import Counter
from typing import Optional

from .capabilities import WhichFn, collect_capabilities
from .types import CatalogSummary, ProductOverview


def collect_catalog_summary() -> CatalogSummary:
    """Summarize the V2 model/hardware/profile/preset catalog.

    The V2 YAML tree remains the source of truth. This function only loads
    the existing registry surfaces and returns compact counts for GUI
    navigation, filters, and health badges.
    """
    from sndr.model_configs.registry_v2 import (
        list_hardware,
        list_models,
        list_presets,
        list_profiles,
        load_preset_def,
    )

    model_ids = list_models()
    hardware_ids = list_hardware()
    profile_ids = list_profiles()
    preset_ids = list_presets()

    status_counts: Counter[str] = Counter()
    workload_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    default_presets: list[str] = []
    load_errors: list[str] = []
    card_count = 0
    unannotated_count = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for preset_id in preset_ids:
            try:
                preset = load_preset_def(preset_id)
            except Exception as exc:  # pragma: no cover - corpus should load
                load_errors.append(
                    f"{preset_id}: {type(exc).__name__}: {exc}"
                )
                continue

            card = preset.card
            if card is None:
                unannotated_count += 1
                continue

            card_count += 1
            status_counts[card.status] += 1
            for workload in card.workload_allow:
                workload_counts[workload] += 1
            if card.routing_family:
                family_counts[card.routing_family] += 1
            if card.default_for_family:
                default_presets.append(preset_id)

    return CatalogSummary(
        models_count=len(model_ids),
        hardware_count=len(hardware_ids),
        profiles_count=len(profile_ids),
        presets_count=len(preset_ids),
        preset_cards_count=card_count,
        unannotated_presets_count=unannotated_count,
        preset_load_error_count=len(load_errors),
        status_counts=dict(sorted(status_counts.items())),
        workload_counts=dict(sorted(workload_counts.items())),
        family_counts=dict(sorted(family_counts.items())),
        default_presets=tuple(sorted(default_presets)),
        preset_load_errors=tuple(load_errors),
    )


def collect_product_overview(
    *,
    which: WhichFn = shutil.which,
    engine_installed: Optional[bool] = None,
) -> ProductOverview:
    """Return the first dashboard-ready snapshot for GUI clients."""
    return ProductOverview(
        capabilities=collect_capabilities(
            which=which,
            engine_installed=engine_installed,
        ),
        catalog=collect_catalog_summary(),
    )
