# SPDX-License-Identifier: Apache-2.0
"""PN258 — Oracle acceptance test for MTP K+1 verify.

Implements the next-cycle entry point from
docs/_internal/MTP_TQ_GEMMA4_NEXT_CYCLE_PN258_H8_NOTES_2026-05-18_RU.md.

Purpose
-------

PN256 + P65 (G4_68 inline) restores target row-0 output coherence but
acceptance remains 0%. Two possibilities:

  Outcome A — Drafter problem only.
    Target verifier rows 0..K-1 are all correct, but the drafter's own
    drafts never match. In that case injecting target-greedy oracle
    tokens as the drafter's drafts will cause every position to accept.
    Next investigation: H8 drafter debug.

  Outcome B — Verifier rows 1..K-1 still broken.
    PN256 may have fixed only row-0; rows 1..K-1 produce wrong argmax
    or the sampler / position mapping / gather is off. Injecting oracle
    tokens will NOT recover acceptance because the rejection sampler
    compares against wrong target argmax for those rows.

This module is the causal test that separates these two cases.

Design
------

Two-pass record/replay around `vllm.v1.sample.rejection_sampler.rejection_sample`:

Pass 1 — RECORD (oracle file does not exist):
  Run the prompt normally with MTP + PN256 + P65. Each call captures
  `target_argmax[r, 0]` (row-0 of each request's target argmax) as the
  "what the model would emit next" oracle. Tokens are appended to
  `/tmp/genesis_pn258_oracle.txt`, one int per line. After Pass 1 the
  file contains the full oracle sequence the target chose during MTP
  generation. Because PN256 fixed row-0, this sequence is the same as
  the TQ-only greedy continuation of the same prompt (verified by the
  PN257a test: output text matches TQ-only baseline).

Pass 2 — REPLAY (oracle file exists):
  Run the same prompt again. At each call, read the next K tokens from
  the oracle file and substitute them into `draft_token_ids` before
  calling the original `rejection_sample`. The drafter's actual drafts
  are discarded for the duration of this run; the oracle "drafter"
  always proposes the target's own greedy continuation.

If the verifier and rejection sampler are wired correctly (Outcome A),
all K injected drafts must accept at each call, and `accepted_per_req`
becomes K (or close to K). If the verifier rows 1..K-1 are broken
(Outcome B), `accepted_per_req` stays 0 or 1 even with the oracle.

Activation: GENESIS_ENABLE_PN258_ORACLE_ACCEPTANCE=1

Switch between RECORD and REPLAY by deleting or keeping the oracle
file `/tmp/genesis_pn258_oracle.txt`.

Logging contract per call (matches the mandatory list in
MTP_TQ_GEMMA4_NEXT_CYCLE_PN258_H8_NOTES_2026-05-18_RU.md section 2):

  + call number
  + mode (RECORD / REPLAY)
  + oracle draft token ids that were used (REPLAY) or recorded (RECORD)
  + draft_token_ids ORIGINAL (drafter's actual proposals before
    substitution)
  + draft_token_ids INJECTED (after substitution, REPLAY only)
  + target rows 0..K-1 argmax (per position)
  + bonus_token_ids
  + output_token_ids and accepted_per_req

This satisfies the "rows 1..K argmax" and "pre/post sampler" trace
requirements.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import List

import torch

log = logging.getLogger("genesis.gemma4.pn258_oracle_acceptance")

_ENV = "GENESIS_ENABLE_PN258_ORACLE_ACCEPTANCE"
_ENV_SELF = "GENESIS_ENABLE_PN258_SELF_ORACLE"
_ORACLE_PATH = "/tmp/genesis_pn258_oracle.txt"
_LOG_PATH = "/tmp/genesis_pn258_oracle_trace.log"
_CALL_IDX = [0]
_APPLIED = False
_ORIGINAL_REJECTION_SAMPLE = None

# In-memory oracle sequence loaded on first REPLAY call. The pointer
# advances by K (per-request num_drafts) per call.
_ORACLE_TOKENS: List[int] = []
_ORACLE_POINTER = [0]
_ORACLE_LOADED = [False]
# Mode is locked at apply() time so RECORD does not flip to REPLAY mid-run
# when the oracle file appears. Operator workflow: clear file -> restart
# (Pass 1 RECORD) -> file populated -> restart (Pass 2 REPLAY).
_MODE_LOCKED = [""]  # "RECORD" or "REPLAY"


def _on() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _self_oracle_on() -> bool:
    """SELF-ORACLE mode: at each call, replace drafts with target_argmax
    of the same target_logits. This isolates the rejection sampler from
    drafter quality entirely. Single-pass, no record/replay needed.

    If accepted_per_req == K under SELF mode: rejection_sample is wired
    correctly AND target rows 0..K-1 argmax is what the sampler compares
    against. Remaining gap (0% accept under normal drafter) IS the
    drafter problem (Outcome A).

    If accepted_per_req < K under SELF mode: rejection_sample / target
    handoff is broken — wrong row indexing, shift, or per-row sampler
    decision. Verifier-side bug remains (Outcome B refinement).
    """
    return os.environ.get(_ENV_SELF, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _is_replay_mode() -> bool:
    """Returns the LOCKED mode for the current process lifetime."""
    return _MODE_LOCKED[0] == "REPLAY"


def _load_oracle_if_needed() -> bool:
    """Load oracle tokens from disk, deduplicating consecutive duplicates.

    Under tensor-parallel TP>1, every worker rank writes the same token
    to the shared trace file in RECORD mode, so each "real" token
    appears once per rank. We dedupe consecutive identical lines to
    recover the unique sequence.
    """
    if _ORACLE_LOADED[0]:
        return True
    try:
        raw_tokens: List[int] = []
        with open(_ORACLE_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_tokens.append(int(line))
                    except ValueError:
                        pass
        # Dedupe consecutive duplicates (TP=N writes N copies of each token).
        prev = None
        for t in raw_tokens:
            if t != prev:
                _ORACLE_TOKENS.append(t)
                prev = t
        _ORACLE_LOADED[0] = True
        log.info(
            "[PN258] oracle loaded: %d raw -> %d unique tokens (TP-dedup)",
            len(raw_tokens),
            len(_ORACLE_TOKENS),
        )
        return True
    except Exception as e:
        log.warning("[PN258] failed to load oracle file: %s", e)
        return False


def _peek_oracle_tokens(k: int) -> List[int]:
    """Peek next k tokens at current pointer WITHOUT advancing.

    Pointer is advanced separately after the rejection_sample call by
    `1 + sum(accepted_per_req)` so the next call's window matches the
    actual cached_len advance.
    """
    if not _load_oracle_if_needed():
        return []
    start = _ORACLE_POINTER[0]
    end = min(start + k, len(_ORACLE_TOKENS))
    return _ORACLE_TOKENS[start:end]


def _advance_oracle_pointer(steps: int) -> None:
    """Advance pointer by `steps` (= 1 + accepted_per_req[0] typically)."""
    _ORACLE_POINTER[0] = min(
        _ORACLE_POINTER[0] + steps, len(_ORACLE_TOKENS)
    )


def _append_oracle_token(tok: int) -> None:
    try:
        with open(_ORACLE_PATH, "a") as f:
            f.write(f"{tok}\n")
    except Exception:
        pass


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_REJECTION_SAMPLE

    if _APPLIED:
        return "applied", "PN258 already installed (idempotent)"
    if not _on():
        return "skipped", (
            f"PN258 disabled (set {_ENV}=1 to enable oracle acceptance "
            f"test). Delete {_ORACLE_PATH} to force RECORD; keep file "
            f"to force REPLAY."
        )

    try:
        from vllm.v1.sample import rejection_sampler as rs
    except ImportError as e:
        return "skipped", f"rejection_sampler not importable: {e}"

    # Lock mode for the entire process lifetime BEFORE wrapping.
    if os.path.isfile(_ORACLE_PATH) and os.path.getsize(_ORACLE_PATH) > 0:
        _MODE_LOCKED[0] = "REPLAY"
    else:
        _MODE_LOCKED[0] = "RECORD"

    original = rs.rejection_sample
    if getattr(original, "_genesis_pn258_wrapped", False):
        _APPLIED = True
        return "applied", "PN258 already wrapped (idempotent)"
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
        replay = _is_replay_mode()
        mode = "REPLAY" if replay else "RECORD"

        # --- INPUT capture: ORIGINAL drafts + target rows 0..K-1 argmax
        try:
            draft_ids_orig = draft_token_ids.detach().cpu().tolist()
            target_argmax_per_row = (
                target_logits.argmax(dim=-1).detach().cpu().tolist()
            )
            bonus_list = bonus_token_ids.detach().cpu().tolist()
            num_drafts = (
                list(num_draft_tokens)
                if not isinstance(num_draft_tokens, list)
                else num_draft_tokens
            )
        except Exception as e:
            draft_ids_orig = []
            target_argmax_per_row = []
            bonus_list = []
            num_drafts = []
            try:
                with open(_LOG_PATH, "a") as f:
                    f.write(
                        f"[PN258 call={call_idx}] INPUT capture err: "
                        f"{type(e).__name__}: {e}\n"
                    )
            except Exception:
                pass

        # --- SELF-ORACLE: substitute drafts with target_argmax_rows
        # verbatim. Each draft d_i := argmax(target_logits[i]). If the
        # sampler is correctly wired and target rows are internally
        # consistent, accepted_per_req must equal K. This is a single-
        # pass version that bypasses the alignment headaches of two-pass
        # RECORD/REPLAY.
        self_oracle = _self_oracle_on()
        self_oracle_injected = False
        if self_oracle:
            try:
                argmax_t = target_logits.argmax(dim=-1)
                if argmax_t.shape[0] == draft_token_ids.shape[0]:
                    draft_token_ids = argmax_t.to(
                        dtype=draft_token_ids.dtype,
                        device=draft_token_ids.device,
                    )
                    self_oracle_injected = True
            except Exception as e:
                try:
                    with open(_LOG_PATH, "a") as f:
                        f.write(
                            f"[PN258 call={call_idx}] SELF-ORACLE err: "
                            f"{type(e).__name__}: {e}\n"
                        )
                except Exception:
                    pass

        # --- REPLAY: substitute drafts with next oracle tokens at the
        # current pointer (do not advance until after the call, so we
        # can re-window by `1 + accepted_per_req` per cached_len step).
        oracle_tokens: List[int] = []
        replay_injected = False
        if replay and not self_oracle:
            try:
                total_drafts = int(draft_token_ids.shape[0])
            except Exception:
                total_drafts = 0
            oracle_tokens = _peek_oracle_tokens(total_drafts)
            if len(oracle_tokens) == total_drafts and total_drafts > 0:
                try:
                    new_drafts = torch.tensor(
                        oracle_tokens,
                        dtype=draft_token_ids.dtype,
                        device=draft_token_ids.device,
                    )
                    draft_token_ids = new_drafts
                    replay_injected = True
                except Exception as e:
                    try:
                        with open(_LOG_PATH, "a") as f:
                            f.write(
                                f"[PN258 call={call_idx}] REPLAY tensor "
                                f"build err: {type(e).__name__}: {e}\n"
                            )
                    except Exception:
                        pass

        # --- INPUT trace
        try:
            draft_ids_injected = draft_token_ids.detach().cpu().tolist()
            effective_mode = "SELF" if self_oracle else mode
            with open(_LOG_PATH, "a") as f:
                f.write(
                    f"\n[PN258 call={call_idx}] {effective_mode} ENTER "
                    f"max_spec_len={max_spec_len} "
                    f"num_drafts={num_drafts} "
                    f"draft_orig={draft_ids_orig[:20]} "
                    f"draft_injected={draft_ids_injected[:20]} "
                    f"target_argmax_rows={target_argmax_per_row[:20]} "
                    f"bonus={bonus_list} "
                    f"self_injected={self_oracle_injected} "
                    f"oracle_pointer={_ORACLE_POINTER[0]}/"
                    f"{len(_ORACLE_TOKENS)}\n"
                )
        except Exception:
            pass

        # --- Call original rejection_sample
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

        # --- OUTPUT trace + RECORD oracle
        try:
            output_list = result.detach().cpu().tolist()
            PLACEHOLDER = -1
            try:
                from vllm.v1.sample.rejection_sampler import (
                    PLACEHOLDER_TOKEN_ID as _P,
                )
                PLACEHOLDER = _P
            except Exception:
                pass

            accepted_per_req: List[int] = []
            for row in output_list:
                accepted = sum(1 for x in row[1:] if x != PLACEHOLDER)
                accepted_per_req.append(accepted)

            exit_mode = "SELF" if self_oracle else mode
            with open(_LOG_PATH, "a") as f:
                f.write(
                    f"[PN258 call={call_idx}] {exit_mode} EXIT  "
                    f"output={output_list} "
                    f"accepted_per_req={accepted_per_req}\n"
                )

            # In RECORD mode, persist row-0 of each request as the
            # oracle's "next correct token". After all rejections, the
            # output's column 0 is the bonus token = target argmax at
            # the bonus position for that request. We record it so the
            # next-step verify gets a sequence-aligned oracle.
            #
            # Future tightening: also record accepted drafts on rows 1..K
            # so RECORD captures multiple tokens per step when acceptance
            # > 0. Initial implementation only records 1 token per call
            # (sufficient for K+1 K=4 with acceptance=0 — typical Pass 1
            # state under the correctness fallback).
            if not replay and output_list:
                for row in output_list:
                    # column 0 = bonus or last-accepted-target-arg-max
                    if row and row[0] is not None and row[0] != PLACEHOLDER:
                        _append_oracle_token(int(row[0]))

            # In REPLAY mode, advance the oracle pointer by
            # (1 + accepted_per_req[0]) — the cached_len advance for this
            # step. If acceptance=0: pointer += 1 (we move forward by 1
            # token, like normal decode). If acceptance=K: pointer += K+1
            # (we accepted K drafts plus the bonus position). This keeps
            # the next call's K-token window aligned with the new
            # cached_len.
            if replay and replay_injected and accepted_per_req:
                advance = 1 + int(accepted_per_req[0])
                _advance_oracle_pointer(advance)
                try:
                    with open(_LOG_PATH, "a") as f:
                        f.write(
                            f"[PN258 call={call_idx}] REPLAY pointer "
                            f"advance=+{advance} -> "
                            f"{_ORACLE_POINTER[0]}/{len(_ORACLE_TOKENS)}\n"
                        )
                except Exception:
                    pass

        except Exception as e:
            try:
                with open(_LOG_PATH, "a") as f:
                    f.write(
                        f"[PN258 call={call_idx}] EXIT err: "
                        f"{type(e).__name__}: {e}\n"
                    )
            except Exception:
                pass

        return result

    wrapped._genesis_pn258_wrapped = True  # type: ignore[attr-defined]
    rs.rejection_sample = wrapped
    _APPLIED = True
    log.info(
        "[PN258] rejection_sample wrapped — oracle acceptance test "
        "mode=%s (locked for process lifetime), trace=%s, oracle=%s",
        _MODE_LOCKED[0],
        _LOG_PATH,
        _ORACLE_PATH,
    )
    return "applied", (
        f"PN258 oracle acceptance trace installed (mode={_MODE_LOCKED[0]} "
        f"— locked at apply, will not flip mid-run). "
        f"Trace -> {_LOG_PATH}; oracle -> {_ORACLE_PATH}."
    )


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


__all__ = [
    "apply",
    "is_applied",
    "revert",
]
