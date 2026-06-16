# A3 — 31B-tq-mtp dev491 native-TQ unblock: progress + honest assessment

**Date**: 2026-06-16. Full PROD access granted. Goal: boot gemma4-31b-tq-mtp-chat-k3 on dev491 (native TQ,
no pr42637 overlays) — potential 2-3× (club-3090 119-154 TPS vs our 49 no-mtp).

## The change-set (workflow wuajllg09, ground-truth verified) — VALIDATED at the spine
Hand-edited a launcher copy `~/start_gemma4-31b-tq-mtp-chat-k3-dev491.sh`: drop the 8 pr42637 `-v` mounts,
add `G4_79` (mm_prefix gate), native `--kv-cache-dtype-skip-layers 58 59`, zero G4_19/19B/32/60K/60L/68/
60B/C/D/70×3, keep G4_69/60A/60G/61/62/P65/31/71B/75 + the TQ serve args. **The spine works** (confirmed in
boot logs): no ImportError (overlay removal fixed the crash), `G4_79` clears the `supports_mm_prefix`
validity gate, `supports_kv_cache_dtype('turboquant_4bit_nc')→True`, `G4_69` reroutes the skip-layers off
TURBOQUANT to a native backend, `G4_71b` resets the drafter dtype.

## Real dev491-readiness bugs found + FIXED (committed/synced — valuable beyond the 31B)
Each boot peeled exactly one genuine patch-impl bug:
1. **Boot 1 — ImportError** (`get_kv_cache_capacity`): the stale pr42637 `kv_cache_utils.py` overlay → removed
   the mounts. Native dev491 TQ is end-to-end.
2. **Boot 2 — G4_60A `TQSlidingWindowSpec` unpicklable** (local class `apply.<locals>.TQSlidingWindowSpec`
   → AttributeError under TP=2 worker pickling). **FIXED (3d257629)**: set `__module__`/`__qualname__` to
   the exposed `vllm.v1.kv_cache_interface` reference.
3. **Boot 3 — PN282/PN248 `use_fp64_gumbel`** crash: the rig had the STALE wrappers (my A1 forward-proof
   fix 23a28079 was committed locally but never synced). **Synced** the forward-proofed versions to the rig.

## The blocker that stopped boot 3 (deep — native-TQ-layout, not a quick patch)
```
RuntimeError: view size is not compatible with input tensor's size and stride... Use .reshape(...) instead.
  turboquant_attn.py:537 do_kv_cache_update → :726 _store_kv → triton_turboquant_store.py:434
```
`kv_cache.view(-1)` in the native TQ store fails because the cache tensor is **non-contiguous** — an
interaction between the sliding-window-TQ spec (G4_60A/G4_60G), the native store, and the dev491
boundary-union skip layers (native unions 0,1,60,61 on top of 58,59). `.view→.reshape` is NOT a safe fix
(it would write to a copy, silently breaking cache persistence). The dev491 `turboquant_attn.py` is itself
a Genesis-patched tree (P67b markers), so this is native-TQ surgery on a layout contract, likely with more
issues after it. **This is genuine multi-iteration native-TQ-readiness work.**

## Honest assessment + options
The 31B-tq-mtp full dev491 unblock is a DEEP native-TQ-readiness effort. The spine is validated and 3 real
fixes are banked, but the remaining work (native-TQ-store layout + likely more) has diminishing certainty,
and the 31B runs fine on dev259 (pin-held). No clean no-TQ 31B base exists (even the no-mtp baseline uses
the TQ stack); a fresh no-TQ 31B-mtp (bf16 KV) would boot cleaner but at ~16K not 64K (bf16 KV ≈ 4× the TQ
footprint) and is its own multi-boot effort.

**Higher-certainty value available instead:**
- 26B MTP-K3 (+65% thinking/chat) is dev491-ready + validated — the biggest realized speed win; deploy it.
- Quality wins now unblocked by A1: `use_fp64_gumbel` (sampling-tail fidelity), `draft_sample_method=
  probabilistic` (+accept), INT8-per-token-head KV (×8 context) — broadly applicable, lower risk.

**Recommendation**: bank the 3 fixes + the validated change-set; defer the full 31B-tq native unblock to a
focused follow-up (or until native dev491 TQ matures); pivot to the certain wins. Revisit the native-TQ
`.view` issue by extracting `triton_turboquant_store` from the real nightly image (the checkout is mutated)
and fixing the contiguity at the source (the spec page layout), not with a `.view→.reshape` band-aid.

PROD (35B) stayed healthy throughout — restored + verified after each of the 3 boots.
