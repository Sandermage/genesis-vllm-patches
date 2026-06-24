# Pristine vLLM source fixtures

Snapshots of upstream vLLM source files at known pristine pin, used by
build-time anchor validation tests (`test_patcher_anchors_invariants.py`).

## Pin

- PN79 fixtures: vllm `0.22.1rc1.dev259+g303916e93` (re-extracted
  2026-06-10 for the K.2 re-anchor; post-#44700 GDN mixed-batch split.
  Extracted via `docker create` + `docker cp` from image
  `vllm/vllm-openai:nightly-303916e93` — pristine, no Genesis patches)
- PN33+PN35 fixtures: vllm `0.20.2rc1.dev209+g5536fc0c0` (extracted 2026-05-12)
- G4_04 fixture: vllm `0.20.2rc1.dev338+gbf0d2dc6d` (extracted 2026-05-28,
  full SHA `bf0d2dc6d764f7ab1a69504f60a55883ec6d9b39`, sourced from
  GitHub raw at the pinned commit)

2026-06-10 note: upstream split `mamba/gdn_linear_attn.py` into
per-model files under `mamba/gdn/` — PN79 Sub-3/Sub-4 now target
`qwen_gdn_linear_attn.py` / `olmo_gdn_linear_attn.py`. The old
`gdn_linear_attn.py` and `olmo_hybrid.py` fixtures are retained for
historical reference only (no patcher registers against them anymore).

## Files

| File | MD5 | Lines | Purpose |
|---|---|---|---|
| chunk.py | 2949617813535680de692d4c24a7b809 | 245 | FLA chunk_gated_delta_rule wrapper (PN79 Sub-1 target) |
| chunk_delta_h.py | 71b7a5017e8cb4c08617c19f5b5f7d4b | 380 | Triton kernel chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (PN79 Sub-2 target) |
| qwen_gdn_linear_attn.py | 194c57a13156fe2f1105064a483de989 | 1828 | Qwen GDN linear attention — ChunkGatedDeltaRule CustomOp + QwenGatedDeltaNetAttention._forward_core (PN79 Sub-3 target) |
| olmo_gdn_linear_attn.py | 9925d28bfc9f5165ea39faab113b8bb9 | 634 | Olmo GDN linear attention with gather/scatter prefill pattern (PN79 Sub-4 target) |
| gdn_linear_attn.py | 18dc6a9c0b1f615a338b468c11fcb71c | 1234 | LEGACY (pre-gdn/ split) — retained for historical reference |
| olmo_hybrid.py | 63ab5a2d29b29b522693188a8da2e421 | 1179 | LEGACY (pre-gdn/ split) — retained for historical reference |
| gpu_model_runner.py | ac61702177b286d0d7239050fd07cbbc | 7179 | v1 worker GPU model runner (PN33.Sub-1 + PN35.Sub-1 targets) |
| llm_base_proposer.py | b2b5def581d27d4654e66d83b4ff4998 | 1638 | v1 spec-decode LLM base proposer (PN35.Sub-2 target) |
| gemma4.py | ff843002c69f9d2aced01c99325e2b21 | 1721 | Gemma 4 model `load_weights._weight_iterator` (G4_04 Sub-1 target — AWQ MoE keys remap, vendors vllm#40886) |
| single_type_kv_cache_manager.py | 93fe087f893767edb2049647ed335c20 | 1407 | v1 core single-type KV cache manager — MambaManager.cache_blocks + find_longest_cache_hit (P85 Site 1/Site 2 + PN346 targets; extracted 2026-06-11 from /private/tmp/candidate_pin_current, pin 0.23.1rc1.dev301+g04c2a8dea (Site 1 body)) |

## Update procedure

When upgrading the pristine pin (rare — only when an anchor genuinely no
longer matches because upstream refactored the touched region):

1. Pull the new file from a clean docker container:
   ```
   docker cp <container>:/usr/local/lib/python3.12/dist-packages/vllm/<rel_path> ./<filename>
   ```
2. Update the MD5 + line count table above.
3. Update `genesis_pin` in `anchor_manifest.json` (when Site Map P2.1 lands).
4. Re-run anchor invariant tests — they will catch broken anchors.

## What these fixtures DO NOT do

- They are NOT runtime targets. The patcher targets vllm install root via
  `resolve_vllm_file()`. These are TEST fixtures only.
- They DO NOT replace the need for live integration testing. The patcher's
  apply path against real vllm install is still the ground truth — fixtures
  catch drift early in CI but cannot guarantee runtime correctness.
- They are NOT auto-updated. Manual update on pin change.
