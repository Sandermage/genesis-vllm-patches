# SPDX-License-Identifier: Apache-2.0
"""gemma4 family contract — 21-patch family (2026-05-17).

Covers:
  * Guards: G4_01 (FP8_BLOCK), G4_02 (Marlin K-dim), G4_03 (non-causal
    drafter), G4_12 (FP8 e4nv), G4_13 (asymmetric KV)
  * Vendor backports: G4_04 (AWQ MoE keys), G4_05 (DFlash backend),
    G4_06 (v_head_size=0), G4_18 (per-layer KV WIP)
  * Deep fixes: G4_07 (FP8 double-scale), G4_08 (Marlin K-pad MoE),
    G4_09 (SWA prefill chunker), G4_10 (Ampere non-causal attn)
  * Perf kernels: G4_15 (fused RMSNorm), G4_24 (fused softcap)
  * Compatibility: G4_11 (chat template), G4_14 (tool-call parser),
    G4_16 (FULL_AND_PIECEWISE)
  * Vision: G4_17 (text-only skip), G4_23 (FP16 overflow)
  * Diagnostic: G4_25 (dual-RoPE)
"""
from tests.unit.integrations._family_contract_helpers import (
    make_family_contract_class,
    make_family_registry_class,
)

PATCHES = [
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_01_gemma4_ampere_fp8_block_guard", "G4_01"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_02_gemma4_ampere_marlin_kdim_guard", "G4_02"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_03_gemma4_ampere_non_causal_drafter_guard", "G4_03"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_04_gemma4_awq_moe_keys_remap", "G4_04"),
    # G4_05 relocated to spec_decode/ in Phase 3 bucket 3 (2026-05-21) —
    # see tests/unit/integrations/spec_decode/test_spec_decode_family_contract.py
    # G4_06 relocated to kv_cache/ in Phase 3 bucket 2 (2026-05-21) —
    # see tests/unit/integrations/kv_cache/test_kv_cache_family_contract.py
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_07_gemma4_fp8_block_double_scale_fix", "G4_07"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_08_gemma4_marlin_kdim_pad_fallback", "G4_08"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_09_gemma4_swa_global_prefill_chunker", "G4_09"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_10_gemma4_ampere_non_causal_attn_backend", "G4_10"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_11_gemma4_chat_template_install", "G4_11"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_12_gemma4_fp8_e4nv_ampere_guard", "G4_12"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_13_gemma4_per_token_head_kv_guard", "G4_13"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_14_gemma4_tool_call_parser_pad_token", "G4_14"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_15_gemma4_fused_rmsnorm_route", "G4_15"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_16_gemma4_full_piecewise_cudagraph", "G4_16"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_17_gemma4_vision_tower_text_only_skip", "G4_17"),
    # G4_18 relocated to kv_cache/ in Phase 3 bucket 2 (2026-05-21) —
    # see tests/unit/integrations/kv_cache/test_kv_cache_family_contract.py
    # G4_19, G4_19B, G4_19C, G4_31, G4_32, G4_60a-k, G4_61, G4_62, G4_67,
    # G4_68, G4_69 relocated to attention/turboquant/ in Phase 3 bucket 4
    # (2026-05-21) — see
    # tests/unit/integrations/attention/turboquant/test_attention_turboquant_family_contract.py
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_23_gemma4_vision_fp16_overflow_fix", "G4_23"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_24_gemma4_fused_softcap_route", "G4_24"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_25_gemma4_rope_dual_base_freq_guard", "G4_25"),
    ("sndr.engines.vllm.patches.model_compat.gemma4.g4_26_diffusiongemma_tp_vocab_soft_embed", "G4_26"),
]


class TestGemma4PatchContract(
    make_family_contract_class("gemma4", PATCHES)
):
    pass


class TestGemma4FamilyRegistry(
    make_family_registry_class("gemma4", PATCHES, filesystem_dir="model_compat/gemma4")
):
    pass
