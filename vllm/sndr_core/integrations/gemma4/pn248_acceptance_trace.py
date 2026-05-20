# SPDX-License-Identifier: Apache-2.0
"""PN248 — Per-step acceptance trace for MTP rejection sampling.

Hooks `vllm.v1.sample.rejection_sampler.rejection_sample` (module-level
function, called from Python orchestration above torch.compile boundary).

Logs per-call:
  - draft_token_ids (what drafter proposed)
  - target_argmax (target's top-1 token per position)
  - num_draft_tokens, max_spec_len
  - output_token_ids (accept/reject mask: accepted draft IDs or
    PLACEHOLDER_TOKEN_ID for rejected; col 0 = bonus/recovered)

Goal: directly test Hypothesis D (cross-quantization verifier loop).

If D is correct: trace will show drafts being ACCEPTED that shouldn't
be, OR target argmax stuck on same token regardless of context.

If D is wrong: drafts being REJECTED properly but next_tok somehow
still stuck — different mechanism.

Activation: GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE=1

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.gemma4.pn248_acceptance_trace")

_ENV = "GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE"
_LOG_PATH = "/tmp/genesis_pn248_acceptance_trace.log"
_CALL_IDX = [0]
_APPLIED = False
_ORIGINAL_REJECTION_SAMPLE = None


def _on() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_REJECTION_SAMPLE

    if _APPLIED:
        return "applied", "PN248 already installed (idempotent)"
    if not _on():
        return "skipped", (
            f"PN248 disabled (set {_ENV}=1 to enable acceptance trace)"
        )

    try:
        from vllm.v1.sample import rejection_sampler as rs
    except ImportError as e:
        return "skipped", f"rejection_sampler not importable: {e}"

    original = rs.rejection_sample
    if getattr(original, "_genesis_pn248_wrapped", False):
        _APPLIED = True
        return "applied", "PN248 already wrapped (idempotent)"
    _ORIGINAL_REJECTION_SAMPLE = original

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
        _CALL_IDX[0] += 1
        call_idx = _CALL_IDX[0]

        # Capture INPUT state (safe — outside compile, no cudagraph)
        try:
            draft_ids_list = draft_token_ids.detach().cpu().tolist()
            target_argmax = target_logits.argmax(dim=-1).detach().cpu().tolist()
            bonus_list = bonus_token_ids.detach().cpu().tolist()
            num_drafts = list(num_draft_tokens) if not isinstance(num_draft_tokens, list) else num_draft_tokens

            with open(_LOG_PATH, "a") as f:
                f.write(
                    f"\n[PN248 call={call_idx}] ENTER "
                    f"max_spec_len={max_spec_len} "
                    f"num_draft_tokens={num_drafts} "
                    f"draft_ids(first 20)={draft_ids_list[:20]} "
                    f"target_argmax(first 20)={target_argmax[:20]} "
                    f"bonus_token_ids={bonus_list}\n"
                )
        except Exception as e:
            try:
                with open(_LOG_PATH, "a") as f:
                    f.write(f"[PN248 call={call_idx}] ENTER err={type(e).__name__}: {e}\n")
            except Exception:
                pass

        # Call original
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

        # Capture OUTPUT state
        try:
            output_list = result.detach().cpu().tolist()
            # Compute accept stats: count non-placeholder per row
            PLACEHOLDER = -1  # PLACEHOLDER_TOKEN_ID
            try:
                from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID as _P
                PLACEHOLDER = _P
            except Exception:
                pass
            stats_per_req = []
            for row in output_list:
                # row[0] = bonus/recovered token
                # row[1:] = accepted drafts or PLACEHOLDER
                accepted = sum(1 for x in row[1:] if x != PLACEHOLDER)
                stats_per_req.append(accepted)

            with open(_LOG_PATH, "a") as f:
                f.write(
                    f"[PN248 call={call_idx}] EXIT  "
                    f"output_token_ids(shape={list(result.shape)})={output_list} "
                    f"accepted_per_req={stats_per_req}\n"
                )
        except Exception as e:
            try:
                with open(_LOG_PATH, "a") as f:
                    f.write(f"[PN248 call={call_idx}] EXIT err={type(e).__name__}: {e}\n")
            except Exception:
                pass

        return result

    wrapped._genesis_pn248_wrapped = True  # type: ignore[attr-defined]
    rs.rejection_sample = wrapped
    _APPLIED = True
    log.info(
        "[PN248] rejection_sample wrapped — per-step acceptance trace "
        "to %s",
        _LOG_PATH,
    )
    return "applied", "PN248 acceptance trace installed"


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_REJECTION_SAMPLE
    if not _APPLIED or _ORIGINAL_REJECTION_SAMPLE is None:
        return False
    try:
        from vllm.v1.sample import rejection_sampler as rs
        rs.rejection_sample = _ORIGINAL_REJECTION_SAMPLE
    except Exception:
        return False
    _APPLIED = False
    _ORIGINAL_REJECTION_SAMPLE = None
    return True
