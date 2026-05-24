# SPDX-License-Identifier: Apache-2.0
"""Genesis sndr_core — Gemma 4 model_compat namespace.

Canonical location after Phase 2.2 of the production cleanup
workstream (2026-05-22): ``integrations/model_compat/gemma4/``. The
previous location ``integrations/gemma4/`` was retired for real
code. It may now contain only the historical PR42637 overlay-path
compatibility shim documented in
``vllm/sndr_core/integrations/gemma4/README.md``.

This package owns ONLY patches whose technical area of influence
is genuinely Gemma-4-specific and cannot be re-homed under a shared
technical bucket without losing meaning. Per the architectural
invariant in
``sndr_private/planning/audits/RELOCATION_DESIGN_2026-05-21_RU.md``
§0.5 Rule 1, real Gemma-owned code belongs here, not in the
historical ``integrations/gemma4/`` shim directory.

Real Gemma-owned patches (16 modules + 2 support files):

  Ampere refusal guards (G4_01/02/03/12/13):
    G4_01  refuse FP8_BLOCK Gemma 4 on Ampere SM86 → vllm#39407
    G4_02  refuse MoE K%128 != 0 on Ampere SM86 → vllm#40354
    G4_03  refuse non-causal drafter on Ampere SM86 → vllm#40382
    G4_12  refuse FP8 e4nv Gemma 4 on Ampere SM86 → vllm#41014
    G4_13  refuse per-token-head KV mismatch (26B-A4B asymmetric)

  Vendor backports (G4_04/11):
    G4_04  AWQ compressed-tensors MoE key remap (vendor vllm#40886)
    G4_11  enhanced chat template install with vllm#42188 tool-id fix

  Deep fixes (G4_07/08/09/10):
    G4_07  FP8_BLOCK double-scale fix (closes vllm#39407 root cause)
    G4_08  Marlin K-pad Triton MoE fallback (closes vllm#40354 for 26B-A4B)
    G4_09  SWA → global prefill chunker (closes vllm#39914)
    G4_10  Ampere non-causal head_dim=256 attention backend (closes vllm#40382)

  Perf kernels (G4_15/24 — partial impl, opt-in):
    G4_15  fused RMSNorm Triton route (PARTIAL; no-op wrapper)
    G4_24  fused FINAL-LOGITS softcap Triton route (PARTIAL)

  Compatibility (G4_14/16):
    G4_14  Gemma 4 tool-call parser pad-token strip (closes vllm#39392)
    G4_16  force FULL_AND_PIECEWISE cudagraph for Gemma 4 dense paths

  Vision-tower (G4_17/23):
    G4_17  vision-tower text-only skip (closes vllm#41565)
    G4_23  vision-tower FP16 overflow guard (closes vllm#40124)

  Diagnostic (G4_25):
    G4_25  dual-RoPE base-freq divergence guard

Support modules (not patches):

  _gemma4_detect.py  — shared detection helpers (is_gemma4_arch,
                       env_truthy, marlin_kdim_supported,
                       detect_fp8_block_format, detect_non_causal_drafter).
                       Imported by Gemma-only patches and by patches
                       relocated to other families that need Gemma
                       arch detection.
  __init__.py        — this file.
  kernels/           — Gemma-only Triton kernel implementations for
                       G4_15 (fused RMSNorm), G4_24 (fused softcap),
                       G4_10 (non-causal attn), G4_08 (k-pad MoE GEMM).

Relocated families (old-path shims were removed during the Phase 3
cleanup; these families now live at their canonical technical-area
owners):

  Spec-decode drafter routing → integrations/spec_decode/:
    G4_05, G4_71, G4_71B, G4_72, G4_73, G4_74, G4_75, G4_76
    G4_78 retired (superseded by P1.8 A2 declarative drafter_kv_sharing)

  KV-cache layout → integrations/kv_cache/:
    G4_06, G4_18

  Probes → integrations/spec_decode/probes/:
    PN241, PN248, PN258, PN262, PN262B, PN266..PN270, PN271 (runtime)

  TurboQuant overlay → integrations/attention/turboquant/:
    G4_19, G4_19B, G4_19C, G4_31, G4_32, G4_60A..K, G4_61, G4_62,
    G4_67, G4_68, G4_69
    Plus kernels: kernels/turboquant/g4_tq_* → attention/turboquant/kernels/

Historical overlay-path shim:
  Phase 2.4 (2026-05-22): PR42637 bind-mount overlay moved to
  integrations/attention/turboquant/overlays/pr42637/. The old
  integrations/gemma4/upstream_overlay_pr42637 path remains only as
  a symlink shim for historical launcher reproducibility. Removing
  that shim requires a dedicated overlay-path retirement phase that
  retires or re-baselines the launcher path and then restores the
  stricter R1 absent-directory rule.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
