# SPDX-License-Identifier: Apache-2.0
"""PN266 propose() input trace — G4_77 design probe.

Wraps `SpecDecodeBaseProposer.propose` to log first 8 invocations'
input tensor shapes + sample heads. Stays dormant until the operator
sets `GENESIS_ENABLE_PN266_PROPOSE_TRACE=1`. Resolves the Phase 3
relocation conflict (stash-pop from old `integrations/gemma4/` path
left markers); canonical location is this file itself.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.spec_decode.pn266_propose_trace")

GENESIS_PN266_MARKER = "Genesis PN266 propose() input trace (G4_77 design probe)"

_ENV_ENABLE = "GENESIS_ENABLE_PN266_PROPOSE_TRACE"
_APPLIED = False
_ORIGINAL_PROPOSE = None
_LOG_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_PROPOSE

    if not _env_enabled():
        return "skipped", f"PN266 disabled (set {_ENV_ENABLE}=1)"

    if _APPLIED:
        return "applied", "PN266 already installed"

    log.warning("[PN266] apply() entered")

    try:
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    except Exception as e:  # noqa: BLE001
        log.warning("[PN266] SKIP: SpecDecodeBaseProposer not importable: %s", e)
        return "skipped", f"SpecDecodeBaseProposer not importable: {e!r}"

    original = SpecDecodeBaseProposer.propose
    if getattr(original, "_genesis_pn266_wrapped", False):
        _APPLIED = True
        return "applied", "propose already wrapped"
    _ORIGINAL_PROPOSE = original

    def _wrapped_propose(self, target_token_ids, target_positions,
                        target_hidden_states, next_token_ids,
                        token_indices_to_sample, common_attn_metadata,
                        sampling_metadata, *args, **kwargs):
        _LOG_COUNT[0] += 1
        if _LOG_COUNT[0] <= 8:
            try:
                tti_shape = tuple(target_token_ids.shape)
                tti_head = target_token_ids[:8].tolist() if tti_shape[0] > 0 else []
                tp_shape = tuple(target_positions.shape)
                tp_head = target_positions[:8].tolist() if tp_shape and tp_shape[-1] > 0 else []
                ths_shape = tuple(target_hidden_states.shape)
                nti = next_token_ids.tolist() if hasattr(next_token_ids, "tolist") else next_token_ids
                bs = common_attn_metadata.batch_size() if hasattr(common_attn_metadata, "batch_size") else "?"
                num_actual = getattr(common_attn_metadata, "num_actual_tokens", "?")
                log.warning(
                    "[PN266] propose call #%d: target_token_ids.shape=%s "
                    "head=%s target_positions.shape=%s head=%s "
                    "target_hidden_states.shape=%s next_token_ids=%s "
                    "batch_size=%s num_actual_tokens=%s",
                    _LOG_COUNT[0],
                    tti_shape, tti_head,
                    tp_shape, tp_head,
                    ths_shape, nti, bs, num_actual,
                )
            except Exception as _e:  # noqa: BLE001
                log.warning("[PN266] introspection failed: %s", _e)
        elif _LOG_COUNT[0] == 9:
            log.warning("[PN266] further propose-trace logs suppressed (>8)")

        return original(
            self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, token_indices_to_sample, common_attn_metadata,
            sampling_metadata, *args, **kwargs,
        )

    _wrapped_propose._genesis_pn266_wrapped = True  # type: ignore[attr-defined]
    SpecDecodeBaseProposer.propose = _wrapped_propose  # type: ignore[method-assign]
    _APPLIED = True

    log.warning("[PN266] INSTALLED: SpecDecodeBaseProposer.propose wrapped")
    return "applied", "PN266 installed"


def is_applied() -> bool:
    return _APPLIED


def log_count() -> int:
    return _LOG_COUNT[0]


def revert() -> bool:
    global _APPLIED, _ORIGINAL_PROPOSE
    if not _APPLIED or _ORIGINAL_PROPOSE is None:
        return False
    try:
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        SpecDecodeBaseProposer.propose = _ORIGINAL_PROPOSE  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_PROPOSE = None
    return True


__all__ = ["GENESIS_PN266_MARKER", "apply", "is_applied", "log_count", "revert"]
