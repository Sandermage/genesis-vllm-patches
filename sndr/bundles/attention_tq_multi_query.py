# SPDX-License-Identifier: Apache-2.0
"""Bundle: TurboQuant multi-query kernel + spec-verify routing (community tier).

DA-009 fix (audit 2026-05-08): bundle tier corrected from "engine" to
"community" to match the registry. P67/P67b are community patches
(see registry.py — tier="community"); marking the bundle as engine
created an inconsistency where the umbrella flag was tier-gated even
though the underlying patches were not.

Atomic apply of P67 + P67b — TurboQuant multi-query split-M Triton
kernel for spec-decode K+1 verification. The dispatcher registry's
notes say "P67b reuses P67's env flag intentionally — they're a
coupled pair"; this bundle makes the coupling explicit via atomic
transaction.

  P67  — TQ multi-query split-M Triton kernel (non-pow-2 GQA via
         lane_valid mask, +32% TPS on 35B FP8 + MTP K=3 + spec-decode
         workload).
  P67b — Spec-verify forward() routing (sends spec K+1 to multi-query
         kernel; fall-through to upstream MLA if disabled).

Tier:    COMMUNITY  (matches registry — Apache 2.0)
Flag:    SNDR_ENABLE_BUNDLE_ATTENTION_TQ_MULTI_QUERY=1
Targets: v1/attention/backends/turboquant_attn.py

Conflict awareness (Stage 8 enhancement target): P65 is declared
`conflicts_with: [P67, P67b]` in PATCH_REGISTRY. If operator activates
this bundle WHILE P65 is also enabled, the dispatcher should refuse
both. Today the bundle just commits P67+P67b atomically; conflict
detection is the dispatcher's job at Stage 8.
"""
from __future__ import annotations

from sndr.env import Flags

from ._common import run_bundle


def apply() -> tuple[str, str]:
    """Apply TQ multi-query bundle (P67 + P67b) atomically."""
    from sndr.engines.vllm.patches.attention.turboquant import p67_tq_multi_query_kernel as _p67

    from sndr.engines.vllm.patches.attention.turboquant import p67b_spec_verify_routing as _p67b
    return run_bundle(
        name="attention_tq_multi_query",
        umbrella_flag=Flags.BUNDLE_ATTENTION_TQ_MULTI_QUERY,
        tier="community",   # DA-009: matches registry tier of P67/P67b
        patcher_factories=[
            _p67._make_patcher,
            _p67b._make_patcher,
        ],
    )


__all__ = ["apply"]
