# DRAFT — vllm#38903 cross-request data contamination (async + PP>1)

**Status:** Local investigation notes only. NOT a Genesis patch. NOT pushed
upstream. Awaits Sander's explicit approval before any external action.

**Date:** 2026-04-26
**Issue:** [vllm-project/vllm#38903](https://github.com/vllm-project/vllm/issues/38903)
**State (upstream):** OPEN, no PR

---

## Severity

This is the **most serious** of the four async-side bugs we audited
(P79b/P79c/P79d/#38903) because it's a **privacy/security violation**, not
a correctness bug:

> The leaked content includes identifiable information (e.g. local file
> paths containing usernames) that is private to other concurrent users,
> confirming actual data contamination.
> — issue #38903 reporter

Cross-request leakage in a hosted serving stack is a hard-fail compliance
issue. Anyone running multi-tenant vLLM with PP>1 + async on v0.16.0+ is
shipping a vulnerability.

## Trigger conditions

Confirmed positive:
- `--pipeline-parallel-size ≥ 2`
- `--async-scheduling` enabled (default in v0.16.0+)
- Multi-node (2 nodes × 8 GPUs in the report)
- 2+ concurrent users sustained ~30 minutes
- Models tested: Kimi-K2.5, GLM-4.7

Confirmed negative:
- `--no-async-scheduling` (workaround)
- vLLM v0.15.1 or earlier
- Single-node setup

**Genesis exposure: ZERO.** We run TP=2, PP=1, single-node, single-user.
No code path through PP+async is engaged. This bug cannot harm us.

## Why we're not certain enough to push a fix upstream

The issue describes a complex multi-condition trigger and the reporter
says single-node *doesn't* reproduce. This means the bug is at the
multi-node PP boundary — likely in how PP rank-2's async batch state
is reset/serialized across the network. A 1-line "disable async when
PP>1" is **not** the right structural fix:

- Over-broad: would also disable async on single-node PP>1 (which the
  reporter says is NOT affected — so we'd be regressing performance for
  configs that don't have the bug).
- Under-investigated: we haven't traced WHICH state crosses the PP
  boundary uncleared. Without that, a config-level disable is a
  workaround, not a fix.
- Wrong author: I (Genesis) don't have a multi-node test rig and can't
  verify the fix doesn't regress.

The right upstream contribution would be a debug-trace that PINPOINTS
the leaking state structure, then a targeted clear/serialize at the
PP-rank-2 → PP-rank-1 boundary. That requires multi-node hardware
neither of us has.

## Proposed bandaid (DRAFT — not for shipping)

If we DID want to ship a conservative auto-disable, the minimum-risk
location is `vllm/config/vllm.py` in the `async_scheduling is None`
auto-resolution branch. Add an explicit elif arm:

```python
# In vllm/config/vllm.py, inside the `elif self.scheduler_config.async_scheduling is None:`
# block, before the final `else: self.scheduler_config.async_scheduling = True`:

elif (
    self.parallel_config.pipeline_parallel_size > 1
    and self.parallel_config.distributed_executor_backend == "mp"
):
    # Conservative: disable async when PP>1 + mp executor backend.
    # Workaround for cross-request contamination reported in vllm#38903
    # under multi-node PP+async with concurrent users.
    # NOTE: single-node PP>1 was NOT confirmed affected per the report,
    # so this auto-disable is over-broad. Operators on single-node PP>1
    # who want async can override with --async-scheduling.
    logger.warning_once(
        "Disabling async scheduling because pipeline_parallel_size > 1 "
        "with mp executor backend has a known cross-request contamination "
        "issue (vllm#38903). Override with --async-scheduling if your "
        "deployment is single-node and you've verified non-contamination."
    )
    self.scheduler_config.async_scheduling = False
```

**For the explicit-enabled branch** (where `async_scheduling=True` was
set by the operator), an analogous warning-but-allow path:

```python
# In vllm/config/vllm.py, inside `if self.scheduler_config.async_scheduling:` block:

if (
    self.parallel_config.pipeline_parallel_size > 1
    and self.parallel_config.distributed_executor_backend == "mp"
):
    logger.warning_once(
        "Async scheduling with pipeline_parallel_size > 1 and mp executor "
        "has a known cross-request contamination issue on multi-node "
        "deployments (vllm#38903). Verify your deployment is single-node "
        "or use --no-async-scheduling for safety."
    )
```

## My recommendation (vs research-agent recommendations)

**Don't ship this.** Reasons:

1. **No data**: Genesis can't reproduce. The fix is speculative without a
   real reproducer.
2. **Wrong scope**: A config-level disable is a workaround. The community
   needs the actual state-leak point traced (which requires multi-node
   hardware).
3. **Risk of regression**: Disabling async on single-node PP>1 (where the
   bug DOESN'T appear per the reporter) regresses performance for users
   who currently rely on it.

**What to do instead** — three options ranked:

A. **Comment on #38903** with: (i) confirm Genesis cannot reproduce on
   single-node, (ii) point at the auto-disable bandaid as a temp
   workaround for affected users, (iii) request the reporter add
   `--distributed-executor-backend ray` or single-node test data to
   narrow the trigger. This costs nothing and helps.

B. **Wait for a vLLM core member with multi-node access** to investigate.
   This is structurally a PP-runtime bug; the right fix likely lives in
   `vllm/v1/worker/` PP-rank communication code, not `config/vllm.py`.

C. **Skip entirely** — Genesis isn't affected, doesn't run the config,
   has no upstream-ready fix. Spend the time on patches that affect
   our users (P79d already done, P79b for async users with spec-decode).

Recommended: **A**. Polite comment + workaround pointer.

## Comparison with research-agent recommendations

The research agent's report (Section 5 combined scenario) focused on
**throughput optimization** — DynamicProposer, Suffix Decoding tuning,
FULL_AND_PIECEWISE cudagraphs. It correctly identified that PP-disagg
(NVIDIA Dynamo) is **net-negative for our 2× A5000 single-user setup**.

The agent did NOT independently surface #38903 because it filtered for
techniques that improve our throughput, not security bugs in unrelated
code paths. That's the right scoping for the research question Sander
asked. #38903 falls into a separate audit lane: defensive-techniques,
not throughput-techniques.

Section 3 of the report (Defensive Techniques to Ship Regardless)
mentions the embedding-input invariant guard (Cat 5v):
`assert (input_ids >= 0).all()` at embedding entry. **This same guard
would catch #38903** as a side effect — if cross-request contamination
ever produces an invalid embedding index, the assert fires before the
contaminated content reaches `F.embedding()`. That's a much better
defensive layer than a config-level disable, AND it's something Genesis
CAN ship (we use F.embedding too) without depending on multi-node test
infra.

## Action items (if Sander approves)

- [ ] Post comment to #38903 along the lines of recommendation A above
- [ ] Implement the embedding-input invariant guard from Section 5
      Cat 5v of research agent report (catches #38903 + similar contamination
      classes in any deployment, not just PP>1)
- [ ] Skip the config-level bandaid until multi-node repro is available

---

**Bottom line:** #38903 is real and serious, but Genesis is the wrong
shop to ship the fix. We have neither hardware nor reproducer. The
honest play is a polite community comment + the embedding-guard from
the research agent's report (which is a Genesis-shippable, multi-bug
defense layer, not a config workaround).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
