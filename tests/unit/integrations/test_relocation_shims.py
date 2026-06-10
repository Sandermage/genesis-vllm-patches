# SPDX-License-Identifier: Apache-2.0
"""Verify Phase 3 relocation migration window is CLOSED.

History:
  * Phase 3 (2026-05-21) relocated 41 runtime patches + 10 TurboQuant
    kernels from ``integrations/gemma4/`` to their technical-area
    owners (``spec_decode/probes/``, ``spec_decode/``, ``kv_cache/``,
    ``attention/turboquant/``, ``attention/turboquant/kernels/``).
  * During the migration window, 50 thin re-export shims kept the old
    import paths resolving so external consumers (custom scripts,
    notebooks, third-party plugins) could migrate without a hard
    break.
  * Phase 2 of production cleanup (2026-05-21, later) deletes the
    shims after confirming no internal consumer still uses them. The
    migration window is now CLOSED.
  * v12.0 (2026-06) — second platform restructure: every patch moved
    from ``vllm.sndr_core.*`` to the top-level ``sndr.*`` package, so
    the canonical "new path" is now
    ``sndr.engines.vllm.{patches,_archive}.*``. The registry's
    ``apply_module`` is migrated to match (see
    ``scripts/migrate_apply_module_to_canonical.py``). The Phase-3
    ``vllm.sndr_core.integrations.gemma4.*`` shims remain deleted.

This test now enforces the closed state:

  1. Every old Phase-3 ``gemma4`` path raises ``ModuleNotFoundError``.
  2. Every canonical ``sndr.*`` new path resolves to a real module.
  3. The registry's ``apply_module`` for every relocated patch points
     at the canonical new path.

Adding a new shim relocation in the future re-opens the migration
window and switches assertion 1 from "raises" back to "resolves" for
that entry — but please don't. The technical-area-ownership refactor
is intentional; introducing a parallel old-path-alias is regression.
"""
from __future__ import annotations

import importlib

import pytest

# (old_shim_path, new_real_path) for every relocated module. Old must
# raise ModuleNotFoundError; new must still resolve.
PROBE_RELOCATIONS = [
    # Bucket 1: probes → spec_decode/probes/
    (
        "vllm.sndr_core.integrations.gemma4.pn241_mtp_trace",
        "sndr.engines.vllm.patches.spec_decode.probes.pn241_mtp_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn248_acceptance_trace",
        "sndr.engines.vllm.patches.spec_decode.probes.pn248_acceptance_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn258_oracle_acceptance",
        "sndr.engines.vllm.patches.spec_decode.probes.pn258_oracle_acceptance",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn262_flash_attn_drafter_trace",
        "sndr.engines.vllm.patches.spec_decode.probes.pn262_flash_attn_drafter_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn262b_kv_alloc_trace",
        "sndr.engines.vllm.patches.spec_decode.probes.pn262b_kv_alloc_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn266_propose_trace",
        "sndr.engines.vllm.patches.spec_decode.probes.pn266_propose_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn267_kv_bridge_trace",
        "sndr.engines.vllm.patches.spec_decode.probes.pn267_kv_bridge_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn268_drafter_blocks_origin",
        "sndr.engines.vllm.patches.spec_decode.probes.pn268_drafter_blocks_origin",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn269_a0_block_table_trace",
        "sndr.engines.vllm.patches.spec_decode.probes.pn269_a0_block_table_trace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn270_drafter_kv_proj_audit",
        "sndr.engines.vllm.patches.spec_decode.probes.pn270_drafter_kv_proj_audit",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.pn271_spec_decode_kv_contract_audit",
        "sndr.engines.vllm.patches.spec_decode.pn271_kv_contract_audit",
    ),
    # Bucket 2: KV-cache → kv_cache/
    (
        "vllm.sndr_core.integrations.gemma4.g4_06_gemma4_kv_proj_v_head_size_zero",
        "sndr.engines.vllm.patches.kv_cache.g4_06_kv_proj_v_head_size_zero",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_18_gemma4_per_layer_kv_page_size",
        "sndr.engines.vllm.patches.kv_cache.g4_18_per_layer_kv_page_size",
    ),
    # Bucket 3: spec_decode drafter routing → spec_decode/
    # PIN.R-G4_05-RETIRE.1 (2026-05-24): G4_05 retired post-bucket-3 — superseded by
    # vllm#39930; canonical target is now _retired/g4_05_dflash_backend_autoselect.
    (
        "vllm.sndr_core.integrations.gemma4.g4_05_gemma4_dflash_backend_autoselect",
        "sndr.engines.vllm._archive.g4_05_dflash_backend_autoselect",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_71_drafter_native_attn_backend",
        "sndr.engines.vllm.patches.spec_decode.g4_71_drafter_native_attn_backend",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_71b_drafter_sliding_triton",
        "sndr.engines.vllm.patches.spec_decode.g4_71b_drafter_sliding_triton",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_72_drafter_native_kv_cache_spec",
        "sndr.engines.vllm.patches.spec_decode.g4_72_drafter_native_kv_cache_spec",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_73_drafter_profile_skip",
        "sndr.engines.vllm.patches.spec_decode.g4_73_drafter_profile_skip",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_74_drafter_hnd_layout",
        "sndr.engines.vllm.patches.spec_decode.g4_74_drafter_hnd_layout",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_75_drafter_head512_triton",
        "sndr.engines.vllm.patches.spec_decode.g4_75_drafter_head512_triton",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_76_disable_drafter_kv_sharing",
        "sndr.engines.vllm.patches.spec_decode.g4_76_disable_drafter_kv_sharing",
    ),
    # G4_78 was retired during Bucket 3 (superseded by P1.8 A2 declarative
    # drafter_kv_sharing). Its real module lives in _retired/.
    (
        "vllm.sndr_core.integrations.gemma4.g4_78_drafter_target_kv_bridge",
        "sndr.engines.vllm._archive.g4_78_drafter_target_kv_bridge",
    ),
    # Bucket 4: TurboQuant patches → attention/turboquant/
    (
        "vllm.sndr_core.integrations.gemma4.g4_19_config_registry",
        "sndr.engines.vllm.patches.attention.turboquant.g4_19_config_registry",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_19_gemma4_turboquant_kv_cache",
        "sndr.engines.vllm.patches.attention.turboquant.g4_19_turboquant_kv_cache",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_19b_gemma4_tq_kv_spec_integration",
        "sndr.engines.vllm.patches.attention.turboquant.g4_19b_tq_kv_spec_integration",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_19c_attention_wrapper",
        "sndr.engines.vllm.patches.attention.turboquant.g4_19c_attention_wrapper",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_31_preserve_tq_dtype",
        "sndr.engines.vllm.patches.attention.turboquant.g4_31_preserve_tq_dtype",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_32_tq_validation_bypass",
        "sndr.engines.vllm.patches.attention.turboquant.g4_32_tq_validation_bypass",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60a_tq_sliding_window_spec",
        "sndr.engines.vllm.patches.attention.turboquant.g4_60a_tq_sliding_window_spec",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60b_turboquant_attn_overlay_loader",
        "sndr.engines.vllm.patches.attention.turboquant.g4_60b_turboquant_attn_overlay_loader",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60c_triton_decode_overlay_loader",
        "sndr.engines.vllm.patches.attention.turboquant.g4_60c_triton_decode_overlay_loader",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60d_triton_store_overlay_loader",
        "sndr.engines.vllm.patches.attention.turboquant.g4_60d_triton_store_overlay_loader",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60e_kv_cache_utils",
        "sndr.engines.vllm.patches.attention.turboquant.g4_60e_kv_cache_utils",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60g_attention_dispatch",
        "sndr.engines.vllm.patches.attention.turboquant.g4_60g_attention_dispatch",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60h_turboquant_config_augment",
        "sndr.engines.vllm.patches.attention.turboquant.g4_60h_turboquant_config_augment",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_60k_arg_utils",
        "sndr.engines.vllm.patches.attention.turboquant.g4_60k_arg_utils",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_61_tq_shared_workspace",
        "sndr.engines.vllm.patches.attention.turboquant.g4_61_tq_shared_workspace",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_62_tq_kernel_warmup",
        "sndr.engines.vllm.patches.attention.turboquant.g4_62_tq_kernel_warmup",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_67_tq_spec_verify_routing",
        "sndr.engines.vllm.patches.attention.turboquant.g4_67_tq_spec_verify_routing",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_68_tq_spec_cg_downgrade_overlay",
        "sndr.engines.vllm.patches.attention.turboquant.g4_68_tq_spec_cg_downgrade_overlay",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.g4_69_skip_layers_native_backend",
        "sndr.engines.vllm.patches.attention.turboquant.g4_69_skip_layers_native_backend",
    ),
    # Bucket 4 kernels: gemma4/kernels/turboquant/ → attention/turboquant/kernels/
    (
        "vllm.sndr_core.integrations.gemma4.kernels.turboquant.g4_tq_cache",
        "sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_cache",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.kernels.turboquant.g4_tq_codebook",
        "sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_codebook",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.kernels.turboquant.g4_tq_packed_triton",
        "sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packed_triton",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.kernels.turboquant.g4_tq_packed_wht_triton",
        "sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packed_wht_triton",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.kernels.turboquant.g4_tq_packing",
        "sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packing",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.kernels.turboquant.g4_tq_read_triton",
        "sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_read_triton",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.kernels.turboquant.g4_tq_reference",
        "sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_reference",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.kernels.turboquant.g4_tq_rotor",
        "sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_rotor",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.kernels.turboquant.g4_tq_tight_triton",
        "sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_tight_triton",
    ),
    (
        "vllm.sndr_core.integrations.gemma4.kernels.turboquant.g4_tq_write_triton",
        "sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_write_triton",
    ),
]

# Registered patch IDs (have PATCH_REGISTRY entries that must point at NEW path).
REGISTERED_AFTER_BUCKET_1 = {
    "PN262": "sndr.engines.vllm.patches.spec_decode.probes.pn262_flash_attn_drafter_trace",
    "PN262B": "sndr.engines.vllm.patches.spec_decode.probes.pn262b_kv_alloc_trace",
}

REGISTERED_AFTER_BUCKET_2 = {
    "G4_06": "sndr.engines.vllm.patches.kv_cache.g4_06_kv_proj_v_head_size_zero",
    "G4_18": "sndr.engines.vllm.patches.kv_cache.g4_18_per_layer_kv_page_size",
}

REGISTERED_AFTER_BUCKET_3 = {
    "G4_05":  "sndr.engines.vllm._archive.g4_05_dflash_backend_autoselect",
    "G4_71":  "sndr.engines.vllm.patches.spec_decode.g4_71_drafter_native_attn_backend",
    "G4_71B": "sndr.engines.vllm.patches.spec_decode.g4_71b_drafter_sliding_triton",
    "G4_72":  "sndr.engines.vllm.patches.spec_decode.g4_72_drafter_native_kv_cache_spec",
    "G4_73":  "sndr.engines.vllm.patches.spec_decode.g4_73_drafter_profile_skip",
    "G4_74":  "sndr.engines.vllm.patches.spec_decode.g4_74_drafter_hnd_layout",
    "G4_75":  "sndr.engines.vllm.patches.spec_decode.g4_75_drafter_head512_triton",
    "G4_76":  "sndr.engines.vllm.patches.spec_decode.g4_76_disable_drafter_kv_sharing",
    "G4_78":  "sndr.engines.vllm._archive.g4_78_drafter_target_kv_bridge",
}

REGISTERED_AFTER_BUCKET_4 = {
    "G4_19":  "sndr.engines.vllm.patches.attention.turboquant.g4_19_turboquant_kv_cache",
    "G4_19B": "sndr.engines.vllm.patches.attention.turboquant.g4_19b_tq_kv_spec_integration",
    # G4_19C removed from ALL_REGISTERED mapping 2026-05-29 — retired
    # with apply_module=None (torch.compile FakeTensor bug, see registry
    # retired_reason). File preserved on disk for diff against a future
    # opaque-op-wrapped fix candidate.
    "G4_31":  "sndr.engines.vllm.patches.attention.turboquant.g4_31_preserve_tq_dtype",
    "G4_32":  "sndr.engines.vllm.patches.attention.turboquant.g4_32_tq_validation_bypass",
    "G4_60A": "sndr.engines.vllm.patches.attention.turboquant.g4_60a_tq_sliding_window_spec",
    "G4_60B": "sndr.engines.vllm.patches.attention.turboquant.g4_60b_turboquant_attn_overlay_loader",
    "G4_60C": "sndr.engines.vllm.patches.attention.turboquant.g4_60c_triton_decode_overlay_loader",
    "G4_60D": "sndr.engines.vllm.patches.attention.turboquant.g4_60d_triton_store_overlay_loader",
    "G4_60E": "sndr.engines.vllm.patches.attention.turboquant.g4_60e_kv_cache_utils",
    "G4_60G": "sndr.engines.vllm.patches.attention.turboquant.g4_60g_attention_dispatch",
    "G4_60H": "sndr.engines.vllm.patches.attention.turboquant.g4_60h_turboquant_config_augment",
    "G4_60K": "sndr.engines.vllm.patches.attention.turboquant.g4_60k_arg_utils",
    "G4_61":  "sndr.engines.vllm.patches.attention.turboquant.g4_61_tq_shared_workspace",
    "G4_62":  "sndr.engines.vllm.patches.attention.turboquant.g4_62_tq_kernel_warmup",
    "G4_67":  "sndr.engines.vllm.patches.attention.turboquant.g4_67_tq_spec_verify_routing",
    "G4_68":  "sndr.engines.vllm.patches.attention.turboquant.g4_68_tq_spec_cg_downgrade_overlay",
    "G4_69":  "sndr.engines.vllm.patches.attention.turboquant.g4_69_skip_layers_native_backend",
}

ALL_REGISTERED = {
    **REGISTERED_AFTER_BUCKET_1,
    **REGISTERED_AFTER_BUCKET_2,
    **REGISTERED_AFTER_BUCKET_3,
    **REGISTERED_AFTER_BUCKET_4,
}


@pytest.mark.parametrize("old_path,new_path", PROBE_RELOCATIONS)
def test_old_shim_path_no_longer_resolves(old_path, new_path):
    """Migration window CLOSED: the old shim path must NOT import."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old_path)


@pytest.mark.parametrize("old_path,new_path", PROBE_RELOCATIONS)
def test_new_real_path_resolves(old_path, new_path):
    """The new technical-area path must still resolve cleanly."""
    try:
        importlib.import_module(new_path)
    except ImportError as e:
        if "torch" in str(e) or "triton" in str(e):
            pytest.skip(f"{new_path} requires torch/triton: {e}")
        raise


@pytest.mark.parametrize("patch_id,expected_path", ALL_REGISTERED.items())
def test_registry_uses_new_path(patch_id, expected_path):
    """Registry's apply_module must point at the new (real) path."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    spec = PATCH_REGISTRY[patch_id]
    assert spec["apply_module"] == expected_path, (
        f"{patch_id}: registry apply_module={spec['apply_module']!r}, "
        f"expected {expected_path!r}"
    )
