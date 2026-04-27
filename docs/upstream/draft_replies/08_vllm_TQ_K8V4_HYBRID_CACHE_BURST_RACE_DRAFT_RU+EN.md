# DRAFT — vllm-project/vllm new issue: TurboQuant k8v4 + hybrid cache + chunked-prefill = IMA under sustained burst

**Status:** DRAFT, NOT POSTED. Awaiting B1/B5/B6 bisect completion + Sander GO.

Per `feedback_github_comment_style.md`:
- HARD rule: never post on vllm without explicit "post it" / "да публикуй"
- HARD rule (2026-04-27): no upstream posts without exact data + retest verification
- Always include English + Russian side-by-side

---

## Draft (English — ready to post)

**Title:** [BUG] TurboQuant `--kv-cache-dtype turboquant_k8v4` + `--enable-prefix-caching --mamba-cache-mode align` + chunked-prefill: `IndexKernel.cu:111` index out of bounds + Xid 43 under sustained burst

Hi maintainers,

(Small disclaimer: I'm from Ukraine and my English is still a work in progress, so I'm using AI to help with translation. Hope it reads okay!)

Posting a clean reproducer for what looks like a race in the interaction between TurboQuant k8v4 KV cache, hybrid prefix-cache (align mode), chunked-prefill, and async scheduling — all together, under sustained burst load.

### TL;DR

`hidden_states[logits_indices]` at [`gpu_model_runner.py:4099`](https://github.com/vllm-project/vllm/blob/HEAD/vllm/v1/worker/gpu_model_runner.py#L4099) crashes with CUDA `IndexKernel.cu:111: index out of bounds` after ~150 requests of sustained burst load (5 concurrent × 30 bursts at `max-num-seqs=2` = 2.5× oversubscription). Repeats consistently, kills both workers, GPU Xid 43 on both A5000s.

### Repro

Hardware: 2× RTX A5000 (Ampere SM 8.6, 24 GB each), TP=2.
Model: `Qwen/Qwen3-Next-Coder-35B-A3B-FP8` (or any hybrid Qwen3.5+ variant).

```bash
docker run --rm --name vllm-test \
  --shm-size=8g --memory=64g -p 8000:8000 --gpus all \
  -v <models>:/models:ro \
  vllm/vllm-openai:nightly \
  --model /models/Qwen3.5-35B-A3B-FP8 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 262144 \
  --kv-cache-dtype turboquant_k8v4 \
  --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align \
  --dtype float16 --disable-custom-all-reduce
```

Bench (any tool that hits 5 concurrent / 30 bursts of `chat/completions`):

```bash
# 5 concurrent x 30 bursts = 150 requests with overlap
python3 -c "
import asyncio, httpx
async def hit():
  async with httpx.AsyncClient() as c:
    return await c.post('http://localhost:8000/v1/chat/completions', json={...})
async def burst(): return await asyncio.gather(*(hit() for _ in range(5)))
async def main(): [await burst() for _ in range(30)]
asyncio.run(main())
"
```

### What we observe

Around burst 11–21 of 30, the engine crashes consistently. Stack with `CUDA_LAUNCH_BLOCKING=1`:

```
File "vllm/v1/worker/gpu_model_runner.py", line 4099, in execute_model
    sample_hidden_states = hidden_states[logits_indices]
                           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: device-side assert triggered
```

PyTorch underlying error:
```
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111:
  Assertion `-sizes[i] <= index && index < sizes[i]
             && "index out of bounds"` failed.
  block: [3,0,0], thread: [0..127,0,0]
```

Then NVIDIA Xid 43 (Reset Channel Verif Error) on both GPUs. Engine dead.

### What we ruled out (5 narrowing runs)

| Run | Variable changed | Result |
|---|---|---|
| Baseline | full config above | crashes |
| `--kv-cache-dtype auto` (no TQ) | only this | **stable** (passes 5/5 + 50/50 + 150/150) |
| Vanilla nightly (no monkey-patches) + `--kv-cache-dtype auto` | both | **stable** |
| `CUDA_LAUNCH_BLOCKING=1` | sync mode | crashes, exact line above |
| <pending> `--enforce-eager` | no CUDA graphs | TBD |

So the trigger is the **combination** of:
- `--kv-cache-dtype turboquant_k8v4`
- `--enable-prefix-caching`
- `--mamba-cache-mode align`
- `--enable-chunked-prefill`
- async scheduling (default in v1)
- sustained burst with `max-num-seqs` oversubscription

`logits_indices = query_start_loc[1:] - 1` ([gpu_model_runner.py:2039](https://github.com/vllm-project/vllm/blob/HEAD/vllm/v1/worker/gpu_model_runner.py#L2039)) is computed from scheduled tokens. The bad-index op suggests `hidden_states.shape[0]` is smaller than `query_start_loc[-1]` under this combo — model produced fewer rows than scheduler expected.

### Possible angles (just hypotheses)

- TQ `_continuation_prefill` concatenates dequant cached K/V with new K/V. If chunk boundaries + cache hits land in a specific way, output rows might miss the partial-prefill last chunk
- Async sample uses `query_start_loc` captured at scheduling time, but TQ may produce different row count than scheduler computed
- `max-num-batched-tokens=8192` with `max-num-seqs=2` allows large mixed batches; under churn the scheduler/TQ row-count invariant might temporarily desync

Happy to run more bisects or test any patch you want me to try.

vLLM commit: `07351e088` (~2026-04-25 nightly).
PyTorch 2.11 + CUDA 13. NVIDIA driver 580.126.09.

Logs / launch script / bench script: <will-link to Genesis repo>

Thanks for everything you do!

— Sander

---

## Черновик (русский — только для проверки тут)

**Тема:** [BUG] TurboQuant `--kv-cache-dtype turboquant_k8v4` + `--enable-prefix-caching --mamba-cache-mode align` + chunked-prefill: `IndexKernel.cu:111` index out of bounds + Xid 43 под sustained burst

Привет мейнтейнерам,

(Маленькая ремарка: я из Украины, мой английский ещё в процессе развития, использую AI для перевода. Надеюсь, читается нормально!)

Публикую чистый репродьюсер на то что выглядит как race в комбинации TurboQuant k8v4 KV-кеш + hybrid prefix-cache (align режим) + chunked-prefill + async scheduling — всё вместе, под sustained burst нагрузкой.

### Кратко

`hidden_states[logits_indices]` в [`gpu_model_runner.py:4099`](https://github.com/vllm-project/vllm/blob/HEAD/vllm/v1/worker/gpu_model_runner.py#L4099) крашится с CUDA `IndexKernel.cu:111: index out of bounds` после ~150 запросов sustained burst (5 параллельных × 30 burst при `max-num-seqs=2` = 2.5× oversubscription). Воспроизводится консистентно, убивает оба worker-а, NVIDIA Xid 43 на обеих A5000.

### Воспроизведение

(см. английский раздел выше — те же команды)

### Что мы наблюдаем

Примерно на burst 11–21 из 30 движок падает. Стек с `CUDA_LAUNCH_BLOCKING=1` точно показывает где. Затем NVIDIA Xid 43 (Reset Channel Verif Error) на обеих GPU. Движок мёртв.

### Что мы исключили (5 narrowing runs)

(см. таблицу выше) Триггер — это **комбинация** TQ k8v4 + cache + chunked + async + sustained burst. Каждый компонент отдельно работает; вместе крашит.

### Возможные направления (только гипотезы)

(см. английский раздел)

Готов прогнать ещё bisect-ы или любой ваш патч на проверку.

vLLM commit: `07351e088` (~2026-04-25 nightly).
PyTorch 2.11 + CUDA 13. NVIDIA driver 580.126.09.

Логи / launch скрипт / bench скрипт: <ссылка на Genesis repo после Sander GO>

Спасибо за всё что вы делаете!

— Sander

---

## Pre-post checklist (before draft becomes a real `gh issue create`)

- [ ] B1 (--enforce-eager) result — adds workaround info if stable
- [ ] B5 (Genesis P67 OFF) — disprove Genesis kernel hook involvement
- [ ] B6 (no external_probe) — disprove Genesis text-edit involvement
- [ ] If any Genesis component is involved, REWRITE issue framing or DON'T POST
- [ ] Push reproducer logs to Genesis public repo (Sander explicit GO)
- [ ] Update GitHub URLs in draft to actual logs URLs
- [ ] Sander reads Russian draft, approves
- [ ] Sander explicit "post it" / "да публикуй"
- [ ] gh issue create with English body
