# SPDX-License-Identifier: Apache-2.0
"""kv_cache family contract — Theme 4 expansion (2026-05-11).

Phase 3 bucket 2 (2026-05-21): grew to 7 patches after relocation of
G4_06 (kv_proj v_head_size=0) and G4_18 (per-layer KV page-size) from
the gemma4 model bucket — their technical area of influence is KV
cache layout / page sizing, not Gemma-only compatibility.
"""
from tests.unit.integrations._family_contract_helpers import (
    make_family_contract_class,
    make_family_registry_class,
)

PATCHES = [
    ("sndr.engines.vllm.patches.kv_cache.p5_page_size", "P5"),
    ("sndr.engines.vllm.patches.kv_cache.p14_block_table", "P14"),
    ("sndr.engines.vllm._archive.p83_mtp_keep_last_cached_block", "P83"),
    ("sndr.engines.vllm.patches.kv_cache.p85_hybrid_fine_shadow_prefix_cache", "P85"),
    ("sndr.engines.vllm.patches.kv_cache.pn95_tier_aware_cache", "PN95"),
    # Phase 3 bucket 2 (2026-05-21): relocated from gemma4/.
    ("sndr.engines.vllm.patches.kv_cache.g4_06_kv_proj_v_head_size_zero", "G4_06"),
    ("sndr.engines.vllm.patches.kv_cache.g4_18_per_layer_kv_page_size", "G4_18"),
]


class TestKvCachePatchContract(
    make_family_contract_class("kv_cache", PATCHES)
):
    pass


class TestKvCacheFamilyRegistry(
    make_family_registry_class("kv_cache", PATCHES)
):
    pass
