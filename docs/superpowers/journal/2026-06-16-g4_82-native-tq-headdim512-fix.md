# G4_82 — native TQ head_dim=512 fix + 31B-tq MTP investigation (dev491)

**Date:** 2026-06-16
**Pin:** 0.22.1rc1.dev491+g1033ffac2 (image vllm/vllm-openai:nightly-1033ffac2)
**Model:** Gemma-4-31B-it-AWQ (dense, 62 layers, interleaved sliding head_dim=256 / global head_dim=512), TQ `turboquant_4bit_nc` (MSE 4-bit), TP=2 on 2×A5000.
**Commit:** `ad491baa` feat(turboquant): G4_82 — TQ prefill SDPA fallback for head_dim>256.

---

## TL;DR

- **CORE WIN:** the 31B-tq now runs **natively** on dev491 (no pr42637 overlays) with **coherent** output — the FA2 `head_dim>256` crash is fixed by a single surgical patch **G4_82**. Validated: equivalence test ALL PASS + E1 live boot coherent (`2+2=4`, full sentences).
- **MTP is still broken on dev491** across three independent axes (garbage / OOM / illegal-address). The dev371-era spec-decode stack (G4_67 / G4_71–76 / G4_81) does not port cleanly. MTP is deferred as a dedicated follow-up; the **no-MTP native 31B-tq is the validated baseline**.

---

## 1. Root cause (the crash)

Native vllm `TurboQuantAttentionImpl` routes ALL non-decode-kernel attention compute
through one private method `_flash_attn_varlen`, which calls FA2
`flash_attn_varlen_func` **unconditionally** (turboquant_attn.py:311/322; 3 callers:
:580 first-chunk prefill, :644 per-request, :826 cached continuation).

FA2 caps head_dim at 256 on SM 8.x (Ampere/Ada — no 512 kernel, vllm#38887). The
Gemma-4-31B **global** layers have head_dim=512 → the first user request crashes the
worker: `FlashAttention forward only supports head dimension at most 256`.

With async-scheduling ON, the worker exception is masked as a scheduler
`KeyError: req_id_to_index` (core.py:578 batch-queue desync); `--no-async-scheduling`
unmasks the true error.

The pr42637 overlay had a `_can_use_flash_prefill` + `_sdpa_causal_prefill` fallback,
but that overlay snapshot is **stale on dev491** — sibling overlays
`kv_cache_utils.py` / `single_type_kv_cache_manager.py` lack dev491-added symbols
`get_kv_cache_capacity` / `register_all_kvcache_specs` (boot ImportError). Re-mounting
the overlay set is not viable. **Reframe:** "remove all overlays, use native TQ" was
right EXCEPT that native dev491 TQ is incomplete for head_dim=512 — so port ONLY the
missing fallback.

## 2. The fix — G4_82

Runtime monkey-patch of `TurboQuantAttentionImpl._flash_attn_varlen` (no TextPatcher —
the method body drifts but its signature is stable), dispatching on `self.head_size`:

- `head_size > 256` → per-sequence torch SDPA over the varlen batch (math/efficient
  backend, any head_dim), reproducing FA2's exact `causal=True` masking:
  `is_causal` for q_len==k_len (sites :580/:644); bottom-right offset mask
  `k_pos <= q_pos + (k_len-q_len)` for q_len<k_len continuation (site :826). GQA via
  `enable_gqa`.
- `head_size <= 256` → original FA2 fast path, byte-unchanged, zero added sync.

**Provably FA2-equivalent** (iron rule #11): the wrapper receives the identical q/k/v
FA2 would, and applies FA2's exact mask, so the output matches up to backend rounding.

**Cost:** one `cu_seqlens.tolist()` GPU→CPU sync per call, only on the head_dim>256
prefill path (eager, off the decode hot path). Zero sync for ≤256. Decode is unaffected
(it uses the TQ triton decode kernel, never FA2).

### Validation
- **Equivalence test** (`/tmp/test_g4_82_equiv.py`, run on rig GPU): ALL 5 PASS,
  max_err ~0.008 ≪ 0.03 (bf16 tol) — prefill q==k, continuation q<k, mixed,
  D=512 + D=256.
- **E1 live boot** (native TQ, 0 overlays, G4_82=1, async-off, MTP OFF):
  `2+2`→`4`, `colors`→`Red, yellow, blue`, sentence→ full coherent English. The
  previously-crashing path now serves correctly.
- Registry contract + doctor CLEAN (313 entries).

## 3. The garbage (MTP) investigation

After G4_82 un-crashed the engine, the MTP-K3 config booted+served but produced
**incoherent multilingual token salad** (`2+2`→`'4 a de rest1- জ4...'` — note: starts
with a correct `4` then degrades). A 6-investigator workflow + a live bisect localized it:

| Hypothesis | Verdict |
|---|---|
| FIX-A store `view(-1)→kv_cache` corruption | **REFUTED** (0.08) — byte-neutral; Gemma cache is contiguous (page_size_padded=None) |
| bf16 skip-layer / unified-pool wrong-axis striding | **REFUTED** (0.06) — separate AttentionGroups; would corrupt from token 0 |
| G4_82 SDPA fallback wrong | **REFUTED** — faithful FA2 pass-through (workflow 0.78 + equivalence test) |
| head_dim=512 decode-read geometry | **REFUTED by E1** — pure decode (MTP off) is coherent |
| **MTP subsystem** (drafter / verify routing) | **CONFIRMED by E1** — garbage only with MTP on |

**E1 (MTP off) = COHERENT** is the dispositive result: the pure target decode path —
including the head_dim=512 global-tier TQ cache read — is correct. The garbage is the
MTP subsystem.

### Why MTP is hard on dev491 (3 failure modes)
- **Broken drafter chain** (original launcher): `G4_75=1` but its deps `G4_71=0,
  G4_74=0` (validator ERROR "G4_75 requires G4_74"). Leaves enabled, base disabled.
- **E2 — native drafter** (enable G4_71+72+73+74+75, spec backend FLASH_ATTN, the
  dev371-validated stack): **CUDA OOM** — G4_71 forces the drafter to a **native bf16
  5-dim** KV cache (+9.27 GiB) instead of TQ-compressed; does not fit dev491's memory
  budget. The dev371 native-drafter approach is memory-infeasible here.
- **E3 — pure-TQ drafter + G4_81 cudagraph-safe verify route**: **cudaErrorIllegalAddress**
  — G4_81's synthetic per-token expansion / buffer-holders hit OOB with this drafter's
  KV-sharing on dev491 (the PN255/PN256/PN242 class its own docstring flags).
- **E4 — enforce-eager + native verify (no route patches) + pure-TQ drafter + G4_82**:
  **cudaErrorIllegalAddress** in a Triton kernel (Worker_TP1) during inference. The
  NATIVE verify continuation-decode path ALSO goes out-of-bounds when reading the TQ
  cache for the K+1 batch. This reveals that the original-MTP `G4_67=1` config was
  **masking** the crash by routing verify away from the broken native path — into
  garbage — rather than fixing it.

**Key insight:** G4_82 wraps `_flash_attn_varlen` **globally**, so it already covers the
drafter's own head_dim=512 layer-3 prefill — the native-drafter chain (G4_71+) is NOT
needed for the 512 crash. The drafter can stay TQ-compressed.

**The precise MTP blocker:** the K+1 spec-verify batch reading the TQ cache via the
decode kernel (`triton_turboquant_decode_attention`) goes **out-of-bounds** →
illegal-address. Three routings all fail at the same root:
- native continuation-decode (E4) → OOB illegal-address
- G4_81 direct route (E3) → OOB illegal-address
- G4_67 route (orig) → no crash but garbage (routes around the OOB into a wrong read)

The synthetic per-token expansion (`synth_seq_lens` / `synth_block_table`) that the
verify path builds must index past the allocated TQ cache blocks for this drafter
(KV-sharing) or continuation geometry on dev491. This is the dedicated follow-up.

### Code-level localization (turboquant_attn.py:671-692)
The verify K+1 batch (q_len=4, cached_len=seq_len-4) hits the **continuation-decode
fast path** — exercised ONLY under MTP, never by single-token decode (hence E1 is clean):
```
synth_seq_lens = _arange_cache[cached_len+1 : seq_len+1]          # :676  incremental causal
synth_bt       = block_table[i:i+1].expand(q_len, -1)            # :677  stride-0 view, q_len rows
out = triton_turboquant_decode_attention(q_seq, kv_cache, synth_bt, synth_seq_lens, ...)  # :678
```
The illegal-address is **shared** across routings — native uses `.expand` (stride-0),
G4_81 uses `.repeat_interleave` (contiguous), both OOB (E3/E4) — so it is NOT the
block_table stride. It is in the decode kernel's handling of the B·K1 virtual-row grid
for this drafter's TQ geometry (head_dim 256/512 interleaved + KV-sharing). Candidate:
the kernel's intermediate buffer (`mid_o`) or `BLOCK_SIZE = kv_cache.shape[1]` desyncs
from what the store wrote for the verify row count (G4_81 docstring's "layer buffers
sized for max_num_seqs, B·K1 exceeds that" lesson + synthesis E5's block_size-unification
suspect at kv_cache_utils.py:1044-1066). **Next step:** CUDA_LAUNCH_BLOCKING=1 boot to
get the exact kernel+line, then an offline store→decode pytest at head_dim=512 / num_kv=2
with B·K1 rows vs a PyTorch dequant+SDPA reference to isolate kernel-vs-allocation.

## 4. Verdict & next steps

- **Ship:** G4_82 (committed) + the **no-MTP native 31B-tq** as the validated dev491
  baseline. This is the first time the 31B-tq runs natively + coherently on dev491.
- **Defer:** MTP on the 31B-tq is a dedicated follow-up — the dev371 spec-decode stack
  (drafter backend/layout + verify routing) needs re-validation against dev491's
  memory budget and KV-sharing semantics. Three distinct dev491 bugs to resolve.
- **File regardless** (stale-docstring landmine): native `turboquant_attn.py:171`
  claims `spec.head_size` is `effective_head_size (padded_slot//2)` but `attention.py:597`
  passes the REAL head_size — self-consistent today, seeds a future double-transform.
  (Upstream note — native source, not a Genesis patch.)

## 5. Method notes
- The async-scheduling masking (worker exception → scheduler KeyError) cost several
  boots; `--no-async-scheduling` is essential for diagnosing TQ worker faults.
- A single surgical runtime monkeypatch (G4_82) replaced 7+ stale overlay files —
  the right call once it was clear the overlays were a snapshot behind dev491.
