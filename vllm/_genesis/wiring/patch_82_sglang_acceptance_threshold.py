# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 82 — SGLang-style threshold_single OR-clause acceptance.

Backport of the per-token acceptance rule from SGLang
(`sgl-kernel/csrc/speculative/speculative_sampling.cuh` ~line 107):

    if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single):
        accept

vs vLLM's vanilla rule (`vllm/v1/sample/rejection_sampler.py:797`):

    accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob

P82 inserts the OR-clause: accept if EITHER vanilla rejection passes OR
the target's confidence in the drafted token meets a threshold. Targets
the structural ceiling `clean_rate ≈ accept_rate^num_spec` identified in
the v7.13 strict-ngram analysis.

================================================================
TRADE-OFF — READ THIS BEFORE ENABLING
================================================================

The threshold rule is **biased** — it loses the unbiased-sampling guarantee
of canonical rejection sampling. SGLang accepts this trade-off explicitly.
For greedy / low-temperature tool-call workloads (our case), the bias
short-circuits in favor of higher-prob target tokens, which is the right
direction. For temperature ≥ 1.0 creative-writing workloads the bias
could compress diversity. WE DO NOT SHIP THIS WITHOUT EMPIRICAL
VALIDATION (`genesis_quality_harness.py` ≥ 30/31 + `genesis_bench_v3.py`
TPS sweep).

================================================================
DESIGN
================================================================

- Text-patch on `vllm/v1/sample/rejection_sampler.py` inside the random
  sampling Triton kernel `rejection_random_sample_kernel`.
- The threshold is baked as a fp32 LITERAL at apply() time from env
  `GENESIS_P82_THRESHOLD_SINGLE` (default 0.3 — SGLang's typical default).
  Changing the threshold requires server restart.
- Greedy path is untouched (greedy already accepts on argmax-match;
  threshold doesn't apply to T=0).
- Synthetic mode is untouched (synthetic acceptance has its own rule).

================================================================
SAFETY MODEL
================================================================

- If env GENESIS_ENABLE_P82 is unset/0 → patch is SKIPPED, source stays
  vanilla. No runtime fall-through path needed.
- If anchor missing (upstream rewrote the line) → SKIPPED with clear
  reason; server boots on vanilla rule.
- Drift markers catch upstream's own threshold patch if/when it lands.

Status: opt-in via `GENESIS_ENABLE_P82=1`. Default OFF.

Tunable knobs
-------------
- `GENESIS_ENABLE_P82` (default unset/0): master switch
- `GENESIS_P82_THRESHOLD_SINGLE` (default 0.3): float in [0.0, 1.0]
  - 0.0 → disables the OR clause (equivalent to OFF, but with overhead)
  - 0.2-0.3 → SGLang typical range, light bias
  - ≥0.5 → aggressive, expect quality regression on diverse outputs

Compatibility
-------------
- All draft methods (ngram, MTP/EAGLE, suffix) — affects only the
  acceptance comparison, not the draft generation.
- Cudagraph: unaffected (rejection sampler runs OUTSIDE the captured graph).
- P71 (block-verify): mutually exclusive in practice — P71 takes the
  block-verify branch BEFORE this point if eligible. P82 fires on the
  per-token fall-through path. Safe to enable both.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference algorithm: SGLang team (sgl-project/sglang).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p82_sglang_acceptance_threshold")


GENESIS_P82_MARKER = "Genesis P82 SGLang-style threshold_single OR-clause v7.53"


# ─── Threshold parsing (with bounds + fallback) ────────────────────────────

_DEFAULT_THRESHOLD = 0.3


def _read_threshold() -> float:
    raw = os.environ.get("GENESIS_P82_THRESHOLD_SINGLE", "").strip()
    if not raw:
        return _DEFAULT_THRESHOLD
    try:
        v = float(raw)
    except ValueError:
        log.warning(
            "[P82] GENESIS_P82_THRESHOLD_SINGLE=%r not parseable as float; using default %.2f",
            raw, _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD
    if not (0.0 <= v <= 1.0):
        log.warning(
            "[P82] threshold %.4f out of [0.0, 1.0]; clamping",
            v,
        )
        v = max(0.0, min(1.0, v))
    return v


# ─── Anchor: 3-line block including upstream NOTE comment for uniqueness ───

P82_OLD = (
    "                # NOTE(woosuk): While the draft probability should never be 0,\n"
    "                # we check it to avoid NaNs. If it happens to be 0, we reject.\n"
    "                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob\n"
)


def _build_replacement(threshold: float) -> str:
    # Bake threshold as a fp32-precision literal (Python repr of float is
    # round-trip safe, sufficient for Triton constexpr coercion).
    threshold_literal = repr(float(threshold))
    return (
        "                # NOTE(woosuk): While the draft probability should never be 0,\n"
        "                # we check it to avoid NaNs. If it happens to be 0, we reject.\n"
        "                # ════════════════════════════════════════════════════════════════\n"
        "                # [Genesis P82 SGLang-style] threshold_single OR-clause acceptance\n"
        "                # accept if EITHER vanilla rejection passes OR target's confidence\n"
        "                # in the drafted token meets the configured threshold. Bias trade-off:\n"
        "                # loses unbiased-sampling guarantee; chosen for low-temp tool-call.\n"
        "                # Threshold baked from env GENESIS_P82_THRESHOLD_SINGLE at server start.\n"
        "                # ════════════════════════════════════════════════════════════════\n"
        "                _genesis_p82_vanilla = (\n"
        "                    draft_prob > 0 and target_prob / draft_prob >= uniform_prob\n"
        "                )\n"
        f"                _genesis_p82_threshold = target_prob >= {threshold_literal}\n"
        "                accepted = _genesis_p82_vanilla or _genesis_p82_threshold\n"
    )


def _make_patcher(threshold: float) -> TextPatcher | None:
    target = resolve_vllm_file("v1/sample/rejection_sampler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P82 v1/sample/rejection_sampler.py — SGLang threshold_single OR-clause "
            f"(threshold={threshold:.4f})"
        ),
        target_file=str(target),
        marker=GENESIS_P82_MARKER,
        sub_patches=[
            TextPatch(
                name="p82_threshold_or_clause",
                anchor=P82_OLD,
                replacement=_build_replacement(threshold),
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P82",
            "_genesis_p82_threshold",
            # Upstream-side markers: if vLLM ever ships its own threshold_single
            # arg in this kernel, we should bow out and let upstream handle it.
            "threshold_single",
            "speculative-accept-threshold",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P82 — SGLang threshold_single OR-clause acceptance."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P82")
    log_decision("P82", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    threshold = _read_threshold()
    if threshold == 0.0:
        # Equivalent to OFF (OR-clause never fires) but with patch overhead;
        # explicitly skip to keep the source vanilla.
        return "skipped", (
            "GENESIS_P82_THRESHOLD_SINGLE=0.0 — OR clause would never fire; "
            "skipping patch to keep source vanilla"
        )

    patcher = _make_patcher(threshold)
    if patcher is None:
        return "skipped", "vllm/v1/sample/rejection_sampler.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P82] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P82" and m in content:
            continue  # our marker; handled above
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix or independent threshold patch",
            )

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return "applied", (
        f"P82 applied: SGLang threshold_single OR-clause installed at threshold={threshold:.4f}. "
        "Activates on random-sample path (greedy / synthetic untouched). "
        "BIASED rule — validate with genesis_quality_harness before prod."
    )
