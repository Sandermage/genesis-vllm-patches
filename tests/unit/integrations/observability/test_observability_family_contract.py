# SPDX-License-Identifier: Apache-2.0
"""observability family contract — Theme 4 expansion (2026-05-11).

PN122 was registry-tagged family="worker" pre-2026-05-11 audit
(filesystem/category mismatch) — fixed to "observability" in v2 retire
batch. PN391 (batch-3 2026-06-13, vendor of vllm#45453) joins the
contract list; the list is curated, not exhaustive (P88/PN282 carry
their own dedicated module tests and are intentionally not duplicated
here).
"""
from tests.unit.integrations._family_contract_helpers import (
    make_family_contract_class,
    make_family_registry_class,
)

PATCHES = [
    (
        "sndr.engines.vllm.patches.observability.pn122_sprint26_cudagraph_dispatch_trace",
        "PN122",
    ),
    (
        "sndr.engines.vllm.patches.observability.pn391_health_decode_watchdog",
        "PN391",
    ),
]


class TestObservabilityPatchContract(
    make_family_contract_class("observability", PATCHES)
):
    pass


class TestObservabilityFamilyRegistry(
    make_family_registry_class("observability", PATCHES)
):
    pass
