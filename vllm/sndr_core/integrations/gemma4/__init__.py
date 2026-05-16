# SPDX-License-Identifier: Apache-2.0
"""Genesis sndr_core — Gemma 4 family integrations.

All Gemma-4-specific patches live here under the ``G4_NN`` patch-id
convention. Filenames use the ``g4_NN_gemma4_<description>.py`` form
so a grep for ``gemma4`` surfaces every Gemma-4 patch independent of
the numeric id.

Rationale: Gemma 4 has architecture-specific blockers that don't
generalize to other models (hybrid sliding+global attention with
head_dim=256/512, k_eq_v shared KV, p-RoPE for 256K context, MoE
moe_intermediate_size=704 that breaks Marlin K-divisibility, FP8_BLOCK
double-scale bug that's checkpoint-format-specific). Bundling all the
fixes under a single ``gemma4`` family keeps the registry browseable
and the family-contract test self-contained.

Patches under this family:

  Guards (Arc 1 — fail fast on known-broken Ampere combos):
    G4_01  Ampere SM86 + FP8_BLOCK → refuse, point at vllm#39407
    G4_02  Ampere SM86 + MoE K%128≠0 → refuse, point at vllm#40354
    G4_03  Ampere SM86 + non-causal drafter → refuse, point at vllm#40382

  Backports (Arc 1 — vendor in-flight upstream PRs):
    G4_04  AWQ compressed-tensors MoE key remap (vendor vllm#40886)
    G4_05  DFlash drafter backend autoselect (vendor vllm#42069)
    G4_06  k_eq_v V-projection elimination (vendor vllm#41944)
    G4_11  Enhanced chat template install (vendor vllm#42188 fix)

  Deep fixes (Arc 2-4 — solve the root cause):
    G4_07  FP8_BLOCK double-scale fix (closes vllm#39407)
    G4_08  Marlin K-pad Triton MoE fallback (closes vllm#40354 for Gemma 4 26B-A4B)
    G4_09  SWA → global prefill chunker (closes vllm#39914)
    G4_10  Ampere non-causal head_dim=256 attention backend (closes vllm#40382)
    G4_12  TurboQuant KV unblock on Gemma 4 (closes vllm#41403 Gate 6)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
