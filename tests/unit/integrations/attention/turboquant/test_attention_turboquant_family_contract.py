# SPDX-License-Identifier: Apache-2.0
"""attention.turboquant family contract — Theme 4 expansion (2026-05-11).

Largest single-family by registry: 23 patches but 4 are legacy registry-
only entries with NO dedicated file (P18b, P20, P32, P51 — pre-dispatcher
era, synthetic GENESIS_LEGACY_* flags applied via legacy auto-apply
path). Contract covers the 19 patches that have files on disk.
"""
from tests.unit.integrations._family_contract_helpers import (
    make_family_contract_class,
    make_family_registry_class,
)

PATCHES = [
    ("vllm.sndr_core.integrations.attention.turboquant.p3_tq_bf16_cast", "P3"),
    ("vllm.sndr_core.integrations.attention.turboquant.p22_tq_prealloc", "P22"),
    ("vllm.sndr_core.integrations.attention.turboquant.p26_prefill_output", "P26"),
    ("vllm.sndr_core.integrations.attention.turboquant.p38_tq_continuation_memory", "P38"),
    ("vllm.sndr_core.integrations.attention.turboquant.p40_tq_grouped_decode", "P40"),
    ("vllm.sndr_core.integrations.attention.turboquant.p44_tq_mixed_attn_out", "P44"),
    ("vllm.sndr_core.integrations.attention.turboquant.p65_turboquant_spec_cg_downgrade", "P65"),
    ("vllm.sndr_core.integrations.attention.turboquant.p67_tq_multi_query_kernel", "P67"),
    ("vllm.sndr_core.integrations.attention.turboquant.p67b_spec_verify_routing", "P67b"),
    ("vllm.sndr_core.integrations.attention.turboquant.p67c_sparse_v", "P67c"),
    ("vllm.sndr_core.integrations.attention.turboquant.p78_tolist_capture_guard", "P78"),
    ("vllm.sndr_core.integrations.attention.turboquant.p98_tq_workspace_revert", "P98"),
    ("vllm.sndr_core.integrations.attention.turboquant.p99_workspace_manager_memoize", "P99"),
    ("vllm.sndr_core.integrations.attention.turboquant.p101_tq_continuation_slicing", "P101"),
    ("vllm.sndr_core.integrations.attention.turboquant.pn14_tq_decode_oob_clamp", "PN14"),
    ("vllm.sndr_core.integrations.attention.turboquant.pn26_sparse_v_kernel", "PN26"),
    ("vllm.sndr_core.integrations.attention.turboquant.pn31_fa_varlen_persistent_out", "PN31"),
    ("vllm.sndr_core.integrations.attention.turboquant.pn34_workspace_lock_runtime_relax", "PN34"),
    ("vllm.sndr_core.integrations.attention.turboquant.pn57_tq_centroids_disk_cache", "PN57"),
    # Phase 3 bucket 4 (2026-05-21): G4_19/G4_19B/G4_19C/G4_31/G4_32/G4_60*/G4_61/G4_62/G4_67/G4_68/G4_69 relocated from gemma4/.
    ("vllm.sndr_core.integrations.attention.turboquant.g4_19_turboquant_kv_cache", "G4_19"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_19b_tq_kv_spec_integration", "G4_19B"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_19c_attention_wrapper", "G4_19C"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_31_preserve_tq_dtype", "G4_31"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_32_tq_validation_bypass", "G4_32"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_60a_tq_sliding_window_spec", "G4_60A"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_60b_turboquant_attn_overlay_loader", "G4_60B"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_60c_triton_decode_overlay_loader", "G4_60C"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_60d_triton_store_overlay_loader", "G4_60D"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_60e_kv_cache_utils", "G4_60E"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_60g_attention_dispatch", "G4_60G"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_60h_turboquant_config_augment", "G4_60H"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_60k_arg_utils", "G4_60K"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_61_tq_shared_workspace", "G4_61"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_62_tq_kernel_warmup", "G4_62"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_67_tq_spec_verify_routing", "G4_67"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_68_tq_spec_cg_downgrade_overlay", "G4_68"),
    ("vllm.sndr_core.integrations.attention.turboquant.g4_69_skip_layers_native_backend", "G4_69"),
]


class TestAttentionTurboquantPatchContract(
    make_family_contract_class("attention.turboquant", PATCHES)
):
    pass


class TestAttentionTurboquantFamilyRegistry(
    make_family_registry_class(
        "attention.turboquant", PATCHES, filesystem_dir="attention/turboquant"
    )
):
    pass
