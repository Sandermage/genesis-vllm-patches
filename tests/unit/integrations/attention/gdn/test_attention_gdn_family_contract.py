# SPDX-License-Identifier: Apache-2.0
"""attention.gdn family contract — Theme 4 expansion (2026-05-11).

17 GDN patches — largest single-family contract. Covers all from
classic GDN (P7/P28) through hybrid attention helpers (P39a, P46) to
recent Cliff/streaming fixes (P103, PN59, PN79). Mix of legacy + active.
"""
from tests.unit.integrations._family_contract_helpers import (
    make_family_contract_class,
    make_family_registry_class,
)

PATCHES = [
    ("sndr.engines.vllm.patches.attention.gdn.p7_gdn_dual_stream", "P7"),
    ("sndr.engines.vllm._archive.p7b_gdn_dual_stream_customop", "P7b"),
    ("sndr.engines.vllm.patches.attention.gdn.p28_gdn_core_attn", "P28"),
    ("sndr.engines.vllm.patches.attention.gdn.p39a_fla_kkt_buffer", "P39a"),
    ("sndr.engines.vllm.patches.attention.gdn.p46_gdn_gating_buffers", "P46"),
    ("sndr.engines.vllm.patches.attention.gdn.p60_gdn_ngram_state_recovery", "P60"),
    ("sndr.engines.vllm.patches.attention.gdn.p60b_gdn_ngram_triton_kernel", "P60b"),
    ("sndr.engines.vllm._archive.p63_mtp_gdn_state_recovery", "P63"),
    ("sndr.engines.vllm.patches.attention.gdn.p103_fla_cliff2_chunked", "P103"),
    ("sndr.engines.vllm.patches.attention.gdn.pn11_gdn_a_b_contiguous", "PN11"),
    # PN29 + PN298 consolidated into one module 2026-06-19 (both patch the
    # same engine file chunk_o.py at disjoint regions). The merged registry
    # entry keeps the id "PN298"; PN29's env flag is a recognized alias.
    ("sndr.engines.vllm.patches.attention.gdn.pn29_pn298_chunk_o_consolidated", "PN298"),
    ("sndr.engines.vllm.patches.attention.gdn.pn30_ds_layout_spec_decode_align", "PN30"),
    ("sndr.engines.vllm.patches.attention.gdn.pn32_gdn_chunked_prefill", "PN32"),
    ("sndr.engines.vllm.patches.attention.gdn.pn50_gdn_fused_proj", "PN50"),
    ("sndr.engines.vllm._archive.pn54_gdn_contiguous_dedup", "PN54"),
    ("sndr.engines.vllm.patches.attention.gdn.pn59_streaming_gdn", "PN59"),
    ("sndr.engines.vllm.patches.attention.gdn.pn79_inplace_ssm_state", "PN79"),
]


class TestAttentionGdnPatchContract(
    make_family_contract_class("attention.gdn", PATCHES)
):
    pass


class TestAttentionGdnFamilyRegistry(
    make_family_registry_class(
        "attention.gdn", PATCHES, filesystem_dir="attention/gdn"
    )
):
    pass
