# SPDX-License-Identifier: Apache-2.0
"""compile_safety family contract — Theme 4 expansion (2026-05-11)."""
from tests.unit.integrations._family_contract_helpers import (
    make_family_contract_class,
    make_family_registry_class,
)

PATCHES = [
    ("sndr.engines.vllm._archive.p6_tq_block_size_align", "P6"),
    ("sndr.engines.vllm.patches.compile_safety.p66_cudagraph_size_divisibility_filter", "P66"),
    ("sndr.engines.vllm.patches.compile_safety.p95_marlin_tp_cudagraph_cap", "P95"),
    ("sndr.engines.vllm._archive.pn13_cuda_graph_lambda_arity", "PN13"),
]


class TestCompileSafetyPatchContract(
    make_family_contract_class("compile_safety", PATCHES)
):
    pass


class TestCompileSafetyFamilyRegistry(
    make_family_registry_class("compile_safety", PATCHES)
):
    pass
