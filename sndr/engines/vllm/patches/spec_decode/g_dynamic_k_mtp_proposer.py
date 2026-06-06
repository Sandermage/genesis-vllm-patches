# SPDX-License-Identifier: Apache-2.0
"""SNDR_MTP_DYNAMIC_K_001 — Genesis-original adaptive K MTP spec-decode proposer.

Vendor of vllm#26504 (DynamicProposer)'s per-seq adaptive K algorithm,
adapted to extend `DraftModelProposer` instead of `EagleProposer` so it
works with the assistant-model MTP path used by qwen3.6 assistant
drafters.

Scope (verified via docker exec MRO probe on dev container 2026-05-31):

  - Effective on: models whose Proposer class inherits from
    `DraftModelProposer` (the generic assistant-model MTP path).
    Confirmed for qwen3.6-27B / qwen3.6-35B assistant-model MTP.
  - NO-OP on: `gemma4-31B` / `gemma4-26B` MTP.
    Reason: `Gemma4Proposer` MRO is
    `[Gemma4Proposer, SpecDecodeBaseProposer, object]` — it does NOT
    inherit from `DraftModelProposer`, so the monkey-patch on
    DraftModelProposer.__init__/.propose has no effect.
    For gemma4 adaptive K, a separate `Gemma4Proposer`-targeted patch
    is needed (future work).
  - Self-gating: there is no runtime guard. The MRO of the actual
    Proposer class self-gates whether the patched DraftModelProposer
    methods reach the inference hot-path. Apply log fires regardless.

Algorithm (1:1 port of PR #26504):

  - Per-seq `SequenceState` tracks `num_spec_tokens` (current K for the
    seq) + `acceptance_rate_history` (deque len 10).
  - On each `propose()` call:
      a. Update states with prev step's `num_accepted_tokens_cpu`.
      b. Drop states for finished req_ids.
      c. Per-seq K adjustment with hysteresis:
           avg_acc >= threshold + 0.05  → K = min(K+1, max_spec_tokens)
           avg_acc <= threshold - 0.05  → K = max(K-1, MIN_SPEC_TOKENS=1)
      d. Set `self.num_speculative_tokens` to max(per_seq_k); call
         `super().propose()` to get the draft tokens at the batch-max
         K; pad to `self.max_spec_tokens` columns so the runner gets a
         fixed-width tensor.

Activation (operator decision):

  GENESIS_ENABLE_SNDR_MTP_DYNAMIC_K_001=1
      Enable the adaptive K wrapper.
  GENESIS_SNDR_MTP_DYNAMIC_K_THRESHOLD=0.7
      Acceptance-rate threshold (PR #26504 default). Hysteresis ±0.05
      applied on top.

The wrapper is INSTALLED at apply() time by monkey-patching
`vllm.v1.spec_decode.draft_model.DraftModelProposer.__init__` and
`.propose` to install the SequenceState state + the propose() override
when the env flag is set. The base class is left intact when the flag
is off — this is a runtime opt-in, not a vLLM API change.

Architecture cross-references:

  - vllm#26504 — original PR (open, needs-rebase as of 2026-05-31).
    The PR extends EagleProposer; this Genesis port extends
    DraftModelProposer instead.
  - tools/upstream_watchlist.yaml — full design sketch + retire
    trigger.
  - docs/_internal/MTP_TQ_GEMMA4_FINAL_SYNTHESIS_*  — backstory on
    why MTP K=4 is suboptimal on chat workloads (the per-seq
    adaptive K answers that empirically).
  - docs/TROUBLESHOOTING.md K-sweep table — empirical evidence that
    K is workload-conditional (this proposer converges to the right
    K per-sequence at runtime instead of requiring a launcher split
    between chat-K=3 and structured-K=4).

Status: EXPERIMENTAL. Default off. Operator A/B benches before
flipping default_on in registry.
"""
from __future__ import annotations

import os
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

# Constants — verbatim from PR #26504
MIN_SPEC_TOKENS = 1
ACCEPTANCE_HISTORY_LEN = 10
ACCEPTANCE_RATE_HYSTERESIS = 0.05
MIN_HISTORY_FOR_ADJUSTMENT = 3

# Genesis marker — preserved across boots for is_applied() detection.
GENESIS_MARKER = "SNDR_MTP_DYNAMIC_K_001 installed: per-seq adaptive K MTP proposer (vllm#26504 port to DraftModelProposer)"


class SequenceState:
    """Per-seq spec-decode K + rolling acceptance window."""

    __slots__ = ("num_spec_tokens", "acceptance_rate_history")

    def __init__(self, initial_spec_tokens: int) -> None:
        self.num_spec_tokens = initial_spec_tokens
        self.acceptance_rate_history: deque[float] = deque(
            maxlen=ACCEPTANCE_HISTORY_LEN
        )


def _env_acceptance_rate_threshold() -> float:
    raw = os.environ.get("GENESIS_SNDR_MTP_DYNAMIC_K_THRESHOLD", "0.7")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.7


def _env_enabled() -> bool:
    raw = os.environ.get("GENESIS_ENABLE_SNDR_MTP_DYNAMIC_K_001", "")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _install_dynamic_k_methods(target_cls: type) -> None:
    """Monkey-patch `target_cls` (DraftModelProposer) with adaptive K state
    + propose() override. Idempotent — re-running is a no-op when the
    marker is already present.
    """
    if getattr(target_cls, "_sndr_dynamic_k_installed", False):
        return

    original_init = target_cls.__init__
    original_propose = target_cls.propose

    def _new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Lazy init for adaptive K state
        self._sndr_seq_states: dict[str, SequenceState] = {}
        self._sndr_last_proposed_k_per_seq: dict[str, int] = {}
        self._sndr_acceptance_rate_threshold = _env_acceptance_rate_threshold()
        self._sndr_max_spec_tokens = self.num_speculative_tokens
        self._sndr_initial_spec_tokens = max(
            MIN_SPEC_TOKENS,
            min(self.num_speculative_tokens, self._sndr_max_spec_tokens),
        )

    def _get_or_create_state(self, req_id: str) -> SequenceState:
        state = self._sndr_seq_states.get(req_id)
        if state is None:
            state = SequenceState(self._sndr_initial_spec_tokens)
            self._sndr_seq_states[req_id] = state
        return state

    def _update_sequence_states(
        self,
        req_ids: Sequence[str | None],
        num_accepted_tokens: Sequence[int],
    ) -> None:
        if not self._sndr_last_proposed_k_per_seq:
            return
        for req_id, num_accepted in zip(req_ids, num_accepted_tokens):
            if req_id is None:
                continue
            num_proposed = self._sndr_last_proposed_k_per_seq.get(req_id)
            if num_proposed is None or num_proposed <= 0:
                continue
            acc = max(int(num_accepted), 0)
            acceptance_rate = (
                (acc / num_proposed) if num_proposed > 0 else 0.0
            )
            state = _get_or_create_state(self, req_id)
            state.acceptance_rate_history.append(acceptance_rate)
        self._sndr_last_proposed_k_per_seq.clear()

    def _cleanup_finished_seqs(
        self, req_ids_in_batch: Sequence[str | None]
    ) -> None:
        if self.runner is None:
            return
        active_req_ids = set(self.runner.requests.keys())
        finished_req_ids = set(self._sndr_seq_states.keys()) - active_req_ids
        for req_id in finished_req_ids:
            del self._sndr_seq_states[req_id]
            self._sndr_last_proposed_k_per_seq.pop(req_id, None)

    def _adjust_and_get_spec_tokens_for_batch(
        self, req_ids: Sequence[str | None]
    ) -> list[int]:
        spec_tokens_for_batch: list[int] = []
        for req_id in req_ids:
            if req_id is None:
                spec_tokens_for_batch.append(MIN_SPEC_TOKENS)
                continue
            state = _get_or_create_state(self, req_id)
            history = state.acceptance_rate_history
            if len(history) < MIN_HISTORY_FOR_ADJUSTMENT:
                spec_tokens_for_batch.append(state.num_spec_tokens)
                continue
            avg_acceptance_rate = float(np.mean(history))
            upper = self._sndr_acceptance_rate_threshold + ACCEPTANCE_RATE_HYSTERESIS
            lower = self._sndr_acceptance_rate_threshold - ACCEPTANCE_RATE_HYSTERESIS
            old_k = state.num_spec_tokens
            new_k = old_k
            if avg_acceptance_rate >= upper:
                new_k = min(old_k + 1, self._sndr_max_spec_tokens)
            elif avg_acceptance_rate <= lower:
                new_k = max(old_k - 1, MIN_SPEC_TOKENS)
            if new_k != old_k:
                state.num_spec_tokens = new_k
            spec_tokens_for_batch.append(state.num_spec_tokens)
        return spec_tokens_for_batch

    @torch.inference_mode()
    def _new_propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: Any = None,
    ) -> torch.Tensor:
        if self.runner is None:
            raise RuntimeError(
                "SNDR_MTP_DYNAMIC_K_001 requires a non-None runner"
            )
        batch_size = next_token_ids.shape[0]
        req_ids = self.runner.input_batch.req_ids[:batch_size]

        # 1) update states from prev step's acceptance
        accepted_tokens = (
            self.runner.input_batch.num_accepted_tokens_cpu[:batch_size].tolist()
        )
        _update_sequence_states(self, req_ids, accepted_tokens)
        _cleanup_finished_seqs(self, req_ids)

        # 2) per-seq K
        per_sequence_k = _adjust_and_get_spec_tokens_for_batch(self, req_ids)
        self._sndr_last_proposed_k_per_seq = {
            req_id: k
            for req_id, k in zip(req_ids, per_sequence_k)
            if req_id is not None
        }

        if len(per_sequence_k) != batch_size:
            fixed_k = [0] * batch_size
            for i in range(min(len(per_sequence_k), batch_size)):
                fixed_k[i] = int(per_sequence_k[i])
            per_sequence_k = fixed_k

        max_k_in_batch = max(per_sequence_k) if per_sequence_k else 0

        # 3) call super().propose() with adjusted num_speculative_tokens
        original_num_tokens = self.num_speculative_tokens
        self.num_speculative_tokens = max_k_in_batch
        try:
            full_draft_token_ids = original_propose(
                self,
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )
        finally:
            self.num_speculative_tokens = original_num_tokens

        # 4) pad with zeros to max_spec_tokens columns
        if full_draft_token_ids.numel() == 0:
            current_width = 0
            full_draft_token_ids = torch.empty(
                (batch_size, 0), dtype=torch.int32, device=self.device
            )
        else:
            current_width = full_draft_token_ids.shape[1]

        if current_width < self._sndr_max_spec_tokens:
            pad_width = self._sndr_max_spec_tokens - current_width
            padding = torch.zeros(
                (batch_size, pad_width), dtype=torch.int32, device=self.device
            )
            full_draft_token_ids = torch.cat(
                [full_draft_token_ids, padding], dim=1
            )
        return full_draft_token_ids

    target_cls.__init__ = _new_init
    target_cls.propose = _new_propose
    target_cls._sndr_dynamic_k_installed = True
    target_cls._sndr_marker = GENESIS_MARKER


def apply() -> tuple[str, str]:
    """Registry-side apply contract.

    Returns (status, message). Status is one of "applied", "skipped",
    "failed". When the env flag is unset (default), returns "skipped"
    so operators can leave the flag at default-off until ready to A/B.
    """
    if not _env_enabled():
        return "skipped", (
            "SNDR_MTP_DYNAMIC_K_001 is OFF by default. Set "
            "GENESIS_ENABLE_SNDR_MTP_DYNAMIC_K_001=1 to enable the "
            "per-seq adaptive K MTP proposer (port of vllm#26504 to "
            "the DraftModelProposer base used by all 4 PROD MTP "
            "models). Optional: GENESIS_SNDR_MTP_DYNAMIC_K_THRESHOLD=<float> "
            "(default 0.7 per PR #26504)."
        )
    try:
        from vllm.v1.spec_decode.draft_model import DraftModelProposer
    except ImportError as e:
        return "failed", f"could not import DraftModelProposer: {e!r}"
    try:
        _install_dynamic_k_methods(DraftModelProposer)
    except Exception as e:
        return "failed", (
            f"SNDR_MTP_DYNAMIC_K_001 installation failed: {e!r}"
        )
    return "applied", (
        f"{GENESIS_MARKER}; threshold={_env_acceptance_rate_threshold()}, "
        f"hysteresis=±{ACCEPTANCE_RATE_HYSTERESIS}, history_len="
        f"{ACCEPTANCE_HISTORY_LEN}, min_history={MIN_HISTORY_FOR_ADJUSTMENT}"
    )


def is_applied() -> bool:
    """True iff the monkey-patch is in place on DraftModelProposer."""
    try:
        from vllm.v1.spec_decode.draft_model import DraftModelProposer
    except ImportError:
        return False
    return getattr(DraftModelProposer, "_sndr_dynamic_k_installed", False)


__all__ = [
    "apply",
    "is_applied",
    "SequenceState",
    "GENESIS_MARKER",
    "MIN_SPEC_TOKENS",
    "ACCEPTANCE_HISTORY_LEN",
    "ACCEPTANCE_RATE_HYSTERESIS",
    "MIN_HISTORY_FOR_ADJUSTMENT",
]
