# SPDX-License-Identifier: Apache-2.0
"""Bundle: GDN spec-decode state recovery (community tier).

Atomic apply of P60 + P60b — the GDN+ngram state-recovery + Triton
kernel offset pair. P60b's docstring explicitly says "should be
enabled together with P60" — this bundle enforces atomic apply so
operators can't accidentally activate one without the other (which
leaves the GDN spec-decode path in an inconsistent state).

  P60  — GDN+ngram state recovery (Phase 1: SSM pre-copy). 6 sub-patches
         touching gdn_attn.py, gdn_linear_attn.py, gpu_model_runner.py.
  P60b — GDN+ngram Triton kernel offset (Phase 2). 9 sub-patches on
         gdn_linear_attn.py.

Patcher factories called by the bundle (5 total):
  P60.{_make_gdn_attn_patcher, _make_gdn_linattn_patcher, _make_gmr_patcher}
  P60b.{_make_kernel_patcher, _make_gdn_caller_patcher}

Tier:    community
Flag:    SNDR_ENABLE_BUNDLE_ATTENTION_GDN_SPEC=1
Targets: v1/attention/backends/gdn_attn.py
         model_executor/layers/mamba/gdn_linear_attn.py
         v1/worker/gpu_model_runner.py
"""
from __future__ import annotations

from sndr.env import Flags

from ._common import run_bundle


def apply() -> tuple[str, str]:
    """Apply GDN spec-decode bundle (P60 + P60b) atomically."""
    from sndr.engines.vllm.patches.attention.gdn import p60_gdn_ngram_state_recovery as _p60

    from sndr.engines.vllm.patches.attention.gdn import p60b_gdn_ngram_triton_kernel as _p60b
    return run_bundle(
        name="attention_gdn_spec",
        umbrella_flag=Flags.BUNDLE_ATTENTION_GDN_SPEC,
        tier="community",
        patcher_factories=[
            _p60._make_gdn_attn_patcher,
            _p60._make_gdn_linattn_patcher,
            _p60._make_gmr_patcher,
            _p60b._make_kernel_patcher,
            _p60b._make_gdn_caller_patcher,
        ],
    )


__all__ = ["apply"]
