# SPDX-License-Identifier: Apache-2.0
"""Bundle: spec-decode async cleanup trio (community tier).

Atomic apply of P79b + P79c + P79d — the async-spec-decode cleanup
chain. These three patches address related race conditions in the
async-decode path; today they're 3 separate dispatcher entries but
share a common root cause and should fail/succeed together.

  P79b — Async proposer sync (gpu_model_runner.py).
  P79c — Stale spec-token cleanup on preempt (scheduler.py).
  P79d — Preempt async-discard (scheduler.py).

Cross-file (worker + scheduler) — atomic apply guarantees no boot
state where (e.g.) scheduler-side cleanup is active but worker-side
async sync isn't, which can manifest as stale spec tokens accepted
into the wrong request's KV slots.

Tier:    community
Flag:    SNDR_ENABLE_BUNDLE_SPEC_DECODE_ASYNC_CLEANUP=1
Targets: v1/worker/gpu_model_runner.py
         v1/core/sched/scheduler.py
"""
from __future__ import annotations

from vllm.sndr_core.env import Flags

from ._common import run_bundle


def apply() -> tuple[str, str]:
    """Apply spec-decode async cleanup bundle (P79b/c/d) atomically."""
    from sndr.engines.vllm.patches.worker import p79b_async_proposer_sync as _p79b

    from sndr.engines.vllm.patches.scheduler import p79c_stale_spec_token_cleanup as _p79c

    from sndr.engines.vllm.patches.scheduler import p79d_preempt_async_discard as _p79d
    return run_bundle(
        name="spec_decode_async_cleanup",
        umbrella_flag=Flags.BUNDLE_SPEC_DECODE_ASYNC_CLEANUP,
        tier="community",
        patcher_factories=[
            _p79b._make_patcher,
            _p79c._make_patcher,
            _p79d._make_patcher,
        ],
    )


__all__ = ["apply"]
