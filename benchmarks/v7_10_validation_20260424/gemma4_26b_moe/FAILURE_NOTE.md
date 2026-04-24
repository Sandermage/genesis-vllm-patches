# Gemma 4 26B MoE AWQ — SKIPPED due to vLLM × model compatibility

**Date**: 2026-04-24 23:35 UTC
**Model**: `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (17 GB download)
**vLLM**: `v0.19.2rc1.dev134+gfe9c3d6c5`

## Root cause

`KeyError: 'layers.0.moe.experts.0.down_proj_packed'` during weight load.

The AWQ quantized weights use a per-expert `down_proj_packed` tensor naming
scheme which the current vLLM dev134 Gemma 4 MoE loader does not recognise.

This is NOT a Genesis patch issue. Boot log shows `Genesis Results: 28 applied,
4 skipped, 0 failed` — the patches applied cleanly before the loader crashed.

## What was validated

- P52 (MoE detection) model_detect.py probe correctly identifies Gemma 4 as MoE
  via `text_config.num_experts=128` + `text_config.enable_moe_block=true` —
  both heuristics are present in our v7.10 code (validated by updated
  `_probe_moe` function at [model_detect.py#L68-L148](../../../vllm/_genesis/model_detect.py#L68-L148)).
- Genesis 28-patch apply pipeline ran cleanly on a non-Qwen architecture.

## Follow-up

- Either wait for vLLM to catch up with this AWQ packing, or find an
  alternative Gemma 4 MoE distribution (unquantized / FP8).
- As a substitute for cross-architecture MoE validation, we continue with:
  - Model A (Qwen3-Next-35B-A3B-FP8) — MoE+hybrid+TQ
  - Model B (Qwen3-Next-35B-A3B-AWQ) — MoE+hybrid+AWQ (cross-quantization)
  - Model C (RYS-Qwen3.5-27B-FP8-XL) — dense, cross-family (pending download)
  - fp16kv config — non-TQ dispatch (already validated)

