# SPDX-License-Identifier: Apache-2.0
"""kernels family contract — Theme 4 expansion (2026-05-11)."""
from tests.unit.integrations._family_contract_helpers import (
    make_family_contract_class,
    make_family_registry_class,
)

PATCHES = [
    ("sndr.engines.vllm._archive.p36_tq_shared_decode_buffers", "P36"),
    ("sndr.engines.vllm.patches.kernels.p87_marlin_pad_sub_tile", "P87"),
    ("sndr.engines.vllm.patches.kernels.pn12_ffn_intermediate_pool", "PN12"),
    ("sndr.engines.vllm.patches.kernels.pn25_silu_inductor_safe_pool", "PN25"),
    ("sndr.engines.vllm.patches.kernels.pn28_merge_attn_states_nan_guard", "PN28"),
]


class TestKernelsPatchContract(
    make_family_contract_class("kernels", PATCHES)
):
    pass


class TestKernelsFamilyRegistry(
    make_family_registry_class("kernels", PATCHES)
):
    pass
