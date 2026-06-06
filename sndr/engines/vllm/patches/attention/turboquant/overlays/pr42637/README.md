# PR #42637 overlay files — Genesis project-owned canonical surface

**Source**: https://github.com/vllm-project/vllm/pull/42637
**Source PR HEAD**: `fdeb14981` (2026-05-16)
**Source author**: lesj0610 ("Mixed-attention KV quantization for Gemma 4 models")
**License**: Apache-2.0 (preserved from upstream)

**Status (2026-05-28 onward):** **project-owned canonical overlay.**
These 6 files are vendored in the Genesis tree as the authoritative
runtime surface for Gemma 4 TurboQuant mixed-attention. They are
bind-mounted at container boot via launcher flags. Genesis no longer
treats the upstream PR42637 merge as a precondition for closing the
overlay surface — retirement is an operator decision (switch overlay
strategy, supersede with a different PR, or stock vllm replaces the
surface), not a wait on the original PR. See
`sndr_private/planning/audits/LOCAL_PR42637_CLOSURE_R_2026-05-28_RU.md`
for the policy reframing.

## Что это

Vendored copies of 6 files from upstream PR #42637 for bind-mount
overlay в running vllm container. Эти файлы заменяют соответствующие
dev371 оригиналы в site-packages при container boot, реализуя полный
TurboQuant mixed-attention KV quantization для Gemma 4 (sliding + full
primary + full shared K=V tier dispatch).

| Файл | Target в vllm site-packages | LOC | Назначение |
|---|---|---:|---|
| `turboquant_attn.py` | `vllm/v1/attention/backends/turboquant_attn.py` | 1308 | Main backend с mixed dispatch, workspace helpers, mm_prefix support |
| `triton_turboquant_decode.py` | `vllm/v1/attention/ops/triton_turboquant_decode.py` | 756 | Triton decode kernel с SW + mm_prefix masks |
| `triton_turboquant_store.py` | `vllm/v1/attention/ops/triton_turboquant_store.py` | 447 | Triton store kernels (minor changes) |
| `turboquant_config.py` | `vllm/model_executor/layers/quantization/turboquant/config.py` | 396 | TurboQuantConfig with 4 presets + KV-sharing skip helpers |
| `kv_cache_interface.py` | `vllm/v1/kv_cache_interface.py` | 895 | TQFullAttentionSpec + TQSlidingWindowSpec dataclasses |
| `kv_cache_utils.py` | `vllm/v1/core/kv_cache_utils.py` | 2218 | TQ/native mixed-layout dispatch logic |

Total: 6020 LOC.

## Зачем overlay (а не monkey-patch)

G4_60a/e/g/h/k Genesis monkey-patches предоставляют architectural
foundation (spec dispatch, mixed routing, skip-layer prep), но
TurboQuantAttentionImpl (1308 LOC) too large для monkey-patch — replaces
entire forward path с new dispatch logic, multiple new methods,
restructured workspace reservation. Same для Triton decode kernel
(SW mask + mm_prefix mask intermixed with existing code).

Bind-mount overlay путь:

```bash
docker run \
  -v ${REPO}/vllm/sndr_core/integrations/attention/turboquant/overlays/pr42637/turboquant_attn.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py:ro \
  -v ${REPO}/vllm/sndr_core/integrations/attention/turboquant/overlays/pr42637/triton_turboquant_decode.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_decode.py:ro \
  -v ${REPO}/vllm/sndr_core/integrations/attention/turboquant/overlays/pr42637/triton_turboquant_store.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_store.py:ro \
  ...
```

Genesis launch script `start_g4_60_full_overlay.sh` делает это автоматически.

## Genesis loader G4_60b/c/d

Минимальные patches verify файл присутствует в overlay path и логируют
факт активации. Это позволяет:

  * Determine via env flag whether overlay is being used.
  * Detect mismatch если bind-mount missing.
  * Log overlay activation в boot trace.

См. `g4_60b_turboquant_attn_overlay_loader.py` и siblings.

## Совместимость с dev371

Все 6 файлов основаны на PR #42637 HEAD ``fdeb14981`` который rebased
on top of vllm main за 2 дня до нашего pin `dev371` (2026-05-14). API
surface match'ит:

  * `current_workspace_manager()` / `is_workspace_manager_initialized()`
    — присутствуют в dev371 (PR #40941 merged 2026-04-27)
  * `AttentionSpec.page_size_padded` field — присутствует в dev371
    (added during MLA work)
  * `tq_max_kv_splits_for_cuda_graph` attention_config field —
    присутствует в dev371

Известные risk areas:
  * Triton `tl.float8e4b15` support — verified в dev371 Triton version
  * `attn_groups` schema (nested list[list[AttentionGroup]]) —
    matched в dev371 v1 worker

## Восстановление original

Если overlay вызывает проблему — remove bind-mount flags из launch
script. site-packages вернутся к dev371 оригиналам automatically.

## License

Файлы in this directory © vllm contributors, Apache-2.0.
Genesis loader patches (`g4_60b/c/d_*.py` в parent directory) ©
Sandermage (Sander) Barzov Aleksandr, Apache-2.0.
