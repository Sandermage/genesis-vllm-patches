# SPDX-License-Identifier: Apache-2.0
"""Verify relocation shims keep old import paths working.

Plan reference: sndr_private/planning/audits/RELOCATION_DESIGN_2026-05-21_RU.md §4.6
Created: Phase 3 bucket 1 (2026-05-21) — probes relocation from
integrations/gemma4/ to integrations/spec_decode/probes/.

Shim window: one release. Remove this file together with the shim
files themselves once external imports have migrated.

Invariants enforced:
  1. Old import path still resolves to a module (the shim).
  2. New import path resolves to a module (the real implementation).
  3. Old and new modules expose the same canonical attributes
     (apply, is_applied, should_apply when present) by identity.
  4. For registered patches: registry's apply_module points at the
     NEW path, never at the shim.
"""
from __future__ import annotations

import importlib

import pytest

# (old_path, new_path) — extend as later buckets land.
PROBE_RELOCATIONS = [
    # Bucket 1: probes → spec_decode/probes/
    (
        "vllm.sndr_core.integrations.gemma4.pn241_mtp_trace",
        "vllm.sndr_core.integrations.spec_decode.probes.pn241_mtp_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn248_acceptance_trace",
        "vllm.sndr_core.integrations.spec_decode.probes.pn248_acceptance_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn258_oracle_acceptance",
        "vllm.sndr_core.integrations.spec_decode.probes.pn258_oracle_acceptance",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn262_flash_attn_drafter_trace",
        "vllm.sndr_core.integrations.spec_decode.probes.pn262_flash_attn_drafter_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn262b_kv_alloc_trace",
        "vllm.sndr_core.integrations.spec_decode.probes.pn262b_kv_alloc_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn266_propose_trace",
        "vllm.sndr_core.integrations.spec_decode.probes.pn266_propose_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn267_kv_bridge_trace",
        "vllm.sndr_core.integrations.spec_decode.probes.pn267_kv_bridge_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn268_drafter_blocks_origin",
        "vllm.sndr_core.integrations.spec_decode.probes.pn268_drafter_blocks_origin",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn269_a0_block_table_trace",
        "vllm.sndr_core.integrations.spec_decode.probes.pn269_a0_block_table_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn270_drafter_kv_proj_audit",
        "vllm.sndr_core.integrations.spec_decode.probes.pn270_drafter_kv_proj_audit",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn271_spec_decode_kv_contract_audit",
        "vllm.sndr_core.integrations.spec_decode.pn271_kv_contract_audit",
    ),
    # Bucket 2: KV-cache → kv_cache/
    (
        "vllm.sndr_core.integrations.gemma4.g4_06_gemma4_kv_proj_v_head_size_zero",
        "vllm.sndr_core.integrations.kv_cache.g4_06_kv_proj_v_head_size_zero",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_18_gemma4_per_layer_kv_page_size",
        "vllm.sndr_core.integrations.kv_cache.g4_18_per_layer_kv_page_size",
    ),
    # Bucket 3: spec_decode drafter routing → spec_decode/
    (
        "vllm.sndr_core.integrations.gemma4.g4_05_gemma4_dflash_backend_autoselect",
        "vllm.sndr_core.integrations.spec_decode.g4_05_dflash_backend_autoselect",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_71_drafter_native_attn_backend",
        "vllm.sndr_core.integrations.spec_decode.g4_71_drafter_native_attn_backend",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_71b_drafter_sliding_triton",
        "vllm.sndr_core.integrations.spec_decode.g4_71b_drafter_sliding_triton",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_72_drafter_native_kv_cache_spec",
        "vllm.sndr_core.integrations.spec_decode.g4_72_drafter_native_kv_cache_spec",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_73_drafter_profile_skip",
        "vllm.sndr_core.integrations.spec_decode.g4_73_drafter_profile_skip",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_74_drafter_hnd_layout",
        "vllm.sndr_core.integrations.spec_decode.g4_74_drafter_hnd_layout",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_75_drafter_head512_triton",
        "vllm.sndr_core.integrations.spec_decode.g4_75_drafter_head512_triton",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_76_disable_drafter_kv_sharing",
        "vllm.sndr_core.integrations.spec_decode.g4_76_disable_drafter_kv_sharing",
    ),
    # G4_78 retired (Bucket 3, superseded by P1.8 A2 declarative drafter_kv_sharing).
    (
        "vllm.sndr_core.integrations.gemma4.g4_78_drafter_target_kv_bridge",
        "vllm.sndr_core.integrations._retired.g4_78_drafter_target_kv_bridge",
    ),
    # Bucket 4: TurboQuant overlay → attention/turboquant/
    (
        "vllm.sndr_core.integrations.gemma4.g4_19_config_registry",
        "vllm.sndr_core.integrations.attention.turboquant.g4_19_config_registry",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_19_gemma4_turboquant_kv_cache",
        "vllm.sndr_core.integrations.attention.turboquant.g4_19_turboquant_kv_cache",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_19b_gemma4_tq_kv_spec_integration",
        "vllm.sndr_core.integrations.attention.turboquant.g4_19b_tq_kv_spec_integration",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_19c_attention_wrapper",
        "vllm.sndr_core.integrations.attention.turboquant.g4_19c_attention_wrapper",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_31_preserve_tq_dtype",
        "vllm.sndr_core.integrations.attention.turboquant.g4_31_preserve_tq_dtype",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_32_tq_validation_bypass",
        "vllm.sndr_core.integrations.attention.turboquant.g4_32_tq_validation_bypass",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60a_tq_sliding_window_spec",
        "vllm.sndr_core.integrations.attention.turboquant.g4_60a_tq_sliding_window_spec",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60b_turboquant_attn_overlay_loader",
        "vllm.sndr_core.integrations.attention.turboquant.g4_60b_turboquant_attn_overlay_loader",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60c_triton_decode_overlay_loader",
        "vllm.sndr_core.integrations.attention.turboquant.g4_60c_triton_decode_overlay_loader",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60d_triton_store_overlay_loader",
        "vllm.sndr_core.integrations.attention.turboquant.g4_60d_triton_store_overlay_loader",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60e_kv_cache_utils",
        "vllm.sndr_core.integrations.attention.turboquant.g4_60e_kv_cache_utils",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60g_attention_dispatch",
        "vllm.sndr_core.integrations.attention.turboquant.g4_60g_attention_dispatch",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60h_turboquant_config_augment",
        "vllm.sndr_core.integrations.attention.turboquant.g4_60h_turboquant_config_augment",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60k_arg_utils",
        "vllm.sndr_core.integrations.attention.turboquant.g4_60k_arg_utils",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_61_tq_shared_workspace",
        "vllm.sndr_core.integrations.attention.turboquant.g4_61_tq_shared_workspace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_62_tq_kernel_warmup",
        "vllm.sndr_core.integrations.attention.turboquant.g4_62_tq_kernel_warmup",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_67_tq_spec_verify_routing",
        "vllm.sndr_core.integrations.attention.turboquant.g4_67_tq_spec_verify_routing",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_68_tq_spec_cg_downgrade_overlay",
        "vllm.sndr_core.integrations.attention.turboquant.g4_68_tq_spec_cg_downgrade_overlay",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_69_skip_layers_native_backend",
        "vllm.sndr_core.integrations.attention.turboquant.g4_69_skip_layers_native_backend",
    ),
]

CANONICAL_ATTRS = ("apply", "is_applied", "should_apply")

# Registered probe IDs (have PATCH_REGISTRY entries that must point at NEW path).
REGISTERED_AFTER_BUCKET_1 = {
    "PN262": "vllm.sndr_core.integrations.spec_decode.probes.pn262_flash_attn_drafter_trace",
    "PN262B": "vllm.sndr_core.integrations.spec_decode.probes.pn262b_kv_alloc_trace",
}

# Bucket 2: KV-cache patches with PATCH_REGISTRY entries that must point at NEW path.
REGISTERED_AFTER_BUCKET_2 = {
    "G4_06": "vllm.sndr_core.integrations.kv_cache.g4_06_kv_proj_v_head_size_zero",
    "G4_18": "vllm.sndr_core.integrations.kv_cache.g4_18_per_layer_kv_page_size",
}

# Bucket 3: spec_decode drafter-routing patches + G4_78 retired.
# G4_71B is newly-registered (was previously env-only). G4_78 is retired
# and its apply_module points into _retired/.
REGISTERED_AFTER_BUCKET_3 = {
    "G4_05":  "vllm.sndr_core.integrations.spec_decode.g4_05_dflash_backend_autoselect",
    "G4_71":  "vllm.sndr_core.integrations.spec_decode.g4_71_drafter_native_attn_backend",
    "G4_71B": "vllm.sndr_core.integrations.spec_decode.g4_71b_drafter_sliding_triton",
    "G4_72":  "vllm.sndr_core.integrations.spec_decode.g4_72_drafter_native_kv_cache_spec",
    "G4_73":  "vllm.sndr_core.integrations.spec_decode.g4_73_drafter_profile_skip",
    "G4_74":  "vllm.sndr_core.integrations.spec_decode.g4_74_drafter_hnd_layout",
    "G4_75":  "vllm.sndr_core.integrations.spec_decode.g4_75_drafter_head512_triton",
    "G4_76":  "vllm.sndr_core.integrations.spec_decode.g4_76_disable_drafter_kv_sharing",
    "G4_78":  "vllm.sndr_core.integrations._retired.g4_78_drafter_target_kv_bridge",
}

# Bucket 4: TurboQuant patches with PATCH_REGISTRY entries that must point at NEW path.
# G4_19C, G4_31, G4_32 are newly-registered (were previously env-only).
REGISTERED_AFTER_BUCKET_4 = {
    "G4_19":  "vllm.sndr_core.integrations.attention.turboquant.g4_19_turboquant_kv_cache",
    "G4_19B": "vllm.sndr_core.integrations.attention.turboquant.g4_19b_tq_kv_spec_integration",
    "G4_19C": "vllm.sndr_core.integrations.attention.turboquant.g4_19c_attention_wrapper",
    "G4_31":  "vllm.sndr_core.integrations.attention.turboquant.g4_31_preserve_tq_dtype",
    "G4_32":  "vllm.sndr_core.integrations.attention.turboquant.g4_32_tq_validation_bypass",
    "G4_60A": "vllm.sndr_core.integrations.attention.turboquant.g4_60a_tq_sliding_window_spec",
    "G4_60B": "vllm.sndr_core.integrations.attention.turboquant.g4_60b_turboquant_attn_overlay_loader",
    "G4_60C": "vllm.sndr_core.integrations.attention.turboquant.g4_60c_triton_decode_overlay_loader",
    "G4_60D": "vllm.sndr_core.integrations.attention.turboquant.g4_60d_triton_store_overlay_loader",
    "G4_60E": "vllm.sndr_core.integrations.attention.turboquant.g4_60e_kv_cache_utils",
    "G4_60G": "vllm.sndr_core.integrations.attention.turboquant.g4_60g_attention_dispatch",
    "G4_60H": "vllm.sndr_core.integrations.attention.turboquant.g4_60h_turboquant_config_augment",
    "G4_60K": "vllm.sndr_core.integrations.attention.turboquant.g4_60k_arg_utils",
    "G4_61":  "vllm.sndr_core.integrations.attention.turboquant.g4_61_tq_shared_workspace",
    "G4_62":  "vllm.sndr_core.integrations.attention.turboquant.g4_62_tq_kernel_warmup",
    "G4_67":  "vllm.sndr_core.integrations.attention.turboquant.g4_67_tq_spec_verify_routing",
    "G4_68":  "vllm.sndr_core.integrations.attention.turboquant.g4_68_tq_spec_cg_downgrade_overlay",
    "G4_69":  "vllm.sndr_core.integrations.attention.turboquant.g4_69_skip_layers_native_backend",
}

ALL_REGISTERED = {
    **REGISTERED_AFTER_BUCKET_1,
    **REGISTERED_AFTER_BUCKET_2,
    **REGISTERED_AFTER_BUCKET_3,
    **REGISTERED_AFTER_BUCKET_4,
}


@pytest.mark.parametrize("old_path,new_path", PROBE_RELOCATIONS)
def test_old_import_path_resolves(old_path, new_path):
    """Old import path must continue to resolve during the shim window."""
    old_mod = importlib.import_module(old_path)
    new_mod = importlib.import_module(new_path)
    for attr in CANONICAL_ATTRS:
        if hasattr(new_mod, attr):
            assert hasattr(old_mod, attr), (
                f"shim {old_path} is missing attribute {attr!r} "
                f"present on real module {new_path}"
            )
            assert getattr(old_mod, attr) is getattr(new_mod, attr), (
                f"shim drift: {old_path}.{attr} is not the same object "
                f"as {new_path}.{attr}"
            )


@pytest.mark.parametrize("patch_id,expected_path", ALL_REGISTERED.items())
def test_registry_uses_new_path(patch_id, expected_path):
    """Registry's apply_module must point at the new path, not the shim."""
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    spec = PATCH_REGISTRY[patch_id]
    assert spec["apply_module"] == expected_path, (
        f"{patch_id}: registry apply_module={spec['apply_module']!r}, "
        f"expected {expected_path!r}"
    )
