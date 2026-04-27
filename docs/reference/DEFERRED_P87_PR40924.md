# DEFERRED — PR #40924 merge_attn_states_kernel SM-shared-mem

**Status:** ASSESSED, NOT BACKPORTED (architectural mismatch)
**Date:** 2026-04-27
**Trigger to revisit:** upstream merge → bump pin → land for free

---

## TL;DR

PR #40924 modifies `csrc/attention/merge_attn_states.cu` (CUDA kernel,
~30 lines). It is **not backportable via Genesis text-patch framework**
because:

1. Genesis text-patch operates on Python files in the installed vLLM
   package, not on `.cu` source files.
2. Production vLLM is a wheel/binary install — `merge_attn_states.cu`
   is NOT present at runtime, only the compiled `.so`.
3. Backporting would require rebuilding vLLM from source — multi-hour,
   breaks pin determinism, requires CUDA toolchain on prod node.

---

## Workload-relevance check

`merge_attn_states_kernel` is invoked by hybrid attention paths
(prefix-cache + new-tokens merge via cascade attention).

**Genesis PROD (v748)** runs:
- `enable_prefix_caching=False` (cache OFF)
- MTP K=3 + P82 acceptance
- `--mamba-cache-mode=none`

→ The kernel is NOT on Genesis PROD's hot path.

**v756 align mode** (prefix-cache enabled, hits=2768 on 5K identical):
- This kernel WOULD be invoked. Backport would shave a few µs per
  forward pass, but v756 is already known-faster-decode-blocked under
  load (engine crash under sustained concurrency — separate issue).

---

## Recommended path

1. **Comment on PR #40924** offering to test on A5000 once merged.
   Low effort, signals upstream interest, catches early breakage.
2. **Watch for merge** — once merged, bump the vLLM pin and the
   speedup lands automatically.
3. **No runtime-patch effort.** Even if we wanted to, the cost
   (rebuild + ABI risk + pin instability) far exceeds the benefit
   (a few µs in a kernel that's mostly off our hot path).

---

## Magnitude estimate (for memory)

PR description estimates ~5-10% speedup on cascade-attention forward
pass. Cascade attention itself is ~5-15% of total decode latency on
prefix-cache-heavy workloads. Net Genesis impact: <1% even with
prefix-cache enabled, ~0% with cache off (current PROD).

---

## Pickup checklist (when upstream merges)

- [ ] Bump vLLM pin to merge commit
- [ ] Smoke-test prod compose `up -d` clean boot
- [ ] Run `bench v4` baseline vs new pin
- [ ] If regression → research-agent investigation
- [ ] If improvement → no further action, free win
