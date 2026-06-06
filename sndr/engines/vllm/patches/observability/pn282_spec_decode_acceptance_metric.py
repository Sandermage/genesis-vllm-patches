# SPDX-License-Identifier: Apache-2.0
"""PN282 — Spec-decode acceptance proxy metric (rejection_sample wrap).

Sibling of PN248 (acceptance log trace). PN282 wraps the same Python
orchestration function — ``vllm.v1.sample.rejection_sampler.rejection_sample``
— but instead of writing a heavyweight log file it increments three
Prometheus counters / gauge defined in
``vllm.sndr_core.observability.spec_decode_metrics``:

  sndr_spec_decode_accepted_per_call_total{k, profile}
  sndr_spec_decode_calls_total{profile}
  sndr_spec_decode_max_spec_len{profile}

The metric series register into the same ``prometheus_client.REGISTRY``
that vllm already exposes via its built-in /metrics endpoint, so no
new server is started and no gateway aggregation is needed — Grafana /
Prometheus scrape jobs that already pull ``vllm:*`` series pick up the
``sndr_*`` series automatically.

Coexistence with PN248:

  * Idempotency markers are independent (``_genesis_pn282_wrapped`` vs.
    ``_genesis_pn248_wrapped``); each wrap installs at most once.
  * Apply order is non-critical: both wraps fall through to the wrapped
    callable; double-wrap simply stacks layers. PN282 should preferably
    apply BEFORE PN248 so PN248's log-writer is the outer layer and any
    exception in the metric path doesn't lose the trace, but the inverse
    order is also safe (PN248 never raises into PN282 either).
  * No double-counting: each PN282 installation wraps exactly once.

Default OFF — opt in via:

    SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC=1

Legacy alias (warns once):

    GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC=1

Author: Sandermage; PN282 / 2026-05-20.
"""
from __future__ import annotations

import logging

log = logging.getLogger("genesis.observability.pn282_spec_decode_acceptance_metric")

GENESIS_PN282_MARKER = "Genesis PN282 — spec-decode acceptance proxy metric"

_APPLIED = False
_ORIGINAL_REJECTION_SAMPLE = None


def _placeholder_token_id() -> int:
    try:
        from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
        return int(PLACEHOLDER_TOKEN_ID)
    except Exception:  # noqa: BLE001
        return -1


def apply() -> tuple[str, str]:
    """Install the rejection_sample wrap. Idempotent.

    Returns:
        (status, reason) where status in {"applied", "skipped"}.
    """
    global _APPLIED, _ORIGINAL_REJECTION_SAMPLE

    from vllm.sndr_core.observability.spec_decode_metrics import is_enabled

    if _APPLIED:
        return "applied", "PN282 already installed (idempotent)"
    if not is_enabled():
        return "skipped", (
            "PN282 disabled (set SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC=1)"
        )

    try:
        from vllm.v1.sample import rejection_sampler as rs
    except ImportError as e:
        return "skipped", f"rejection_sampler not importable: {e}"

    original = rs.rejection_sample
    if getattr(original, "_genesis_pn282_wrapped", False):
        _APPLIED = True
        return "applied", "PN282 already wrapped (idempotent)"
    _ORIGINAL_REJECTION_SAMPLE = original

    placeholder = _placeholder_token_id()

    def wrapped(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
        synthetic_mode=False,
        synthetic_conditional_rates=None,
    ):
        result = original(
            draft_token_ids,
            num_draft_tokens,
            max_spec_len,
            cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=synthetic_mode,
            synthetic_conditional_rates=synthetic_conditional_rates,
        )

        try:
            # result shape: [B, max_spec_len + 1]
            # row[0] = bonus / recovered token
            # row[1:] = accepted draft IDs or PLACEHOLDER for rejected
            rows = result.detach().cpu().tolist()
            accepted_per_req = [
                sum(1 for x in row[1:] if x != placeholder)
                for row in rows
            ]
            from vllm.sndr_core.observability.spec_decode_metrics import (
                record_acceptance,
            )
            record_acceptance(accepted_per_req, int(max_spec_len))
        except Exception as e:  # noqa: BLE001
            log.debug(
                "[PN282] metric emission suppressed: %s",
                e,
            )

        return result

    wrapped._genesis_pn282_wrapped = True  # type: ignore[attr-defined]
    rs.rejection_sample = wrapped
    _APPLIED = True
    log.info(
        "[PN282] rejection_sample wrapped — sndr_spec_decode_* "
        "metrics live on /metrics."
    )
    return "applied", "PN282 acceptance metric installed"


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Test-only: restore the original rejection_sample.

    Returns True if a wrap was removed, False if no wrap was present.
    """
    global _APPLIED, _ORIGINAL_REJECTION_SAMPLE
    if not _APPLIED or _ORIGINAL_REJECTION_SAMPLE is None:
        return False
    try:
        from vllm.v1.sample import rejection_sampler as rs
        rs.rejection_sample = _ORIGINAL_REJECTION_SAMPLE
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    _ORIGINAL_REJECTION_SAMPLE = None
    return True


__all__ = ["apply", "is_applied", "revert", "GENESIS_PN282_MARKER"]
