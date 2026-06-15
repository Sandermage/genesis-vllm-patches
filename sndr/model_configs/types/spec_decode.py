# SPDX-License-Identifier: Apache-2.0
"""SpecDecodeConfig — speculative-decoding setup for a ModelConfig.

Relocated from ``model_configs/schema.py`` in M.5.1. The class body is
byte-identical to the pre-refactor version; only the import path for
:class:`SchemaError` changed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from ._base import SchemaError


@dataclass
class SpecDecodeConfig:
    """Speculative decoding setup."""
    method: str  # 'mtp' / 'eagle' / 'ngram' / 'dflash'
    num_speculative_tokens: int
    # Path to a separate drafter model. Required for `dflash`/`eagle` (where the
    # drafter is a distinct checkpoint from the target). For `mtp`/`ngram` keep
    # None — vllm uses the target model's own MTP head / its own n-gram cache.
    model: Optional[str] = None

    # Probabilistic draft-probs propagation (vllm dev338+ native).
    #
    # The upstream rejection_sampler now natively supports the
    # `min(1, target_p / draft_p)` accept rule when both fields below are
    # set on the SpeculativeConfig — there is a runtime gate at
    # `LLMBaseProposer._enable_probabilistic_draft_probs` that requires
    # BOTH `rejection_sample_method == "standard"` AND
    # `draft_sample_method == "probabilistic"`. Without those, the
    # drafter falls back to greedy sampling and the verifier gets
    # `None` for draft_probs, so the accept rule degrades to the
    # straight equality check and we lose ~7 % accept_rate / TPS on
    # MTP K=3.
    #
    # This is the upstream-merged replacement for our Genesis PN90
    # backport (which text-patched the rejection_sampler call site
    # itself). PN90 self-retired on dev338 because upstream
    # restructured the anchor — exposing these fields here is the
    # supported migration path.
    #
    # Defaults stay None to keep V1 configs that do not set them
    # bit-identical to their prior behaviour. Operators opt-in via
    # the model YAML's `spec_decode:` block.
    rejection_sample_method: Optional[str] = None  # "standard" | None
    draft_sample_method: Optional[str] = None      # "probabilistic" | None

    # P1.7c (2026-05-20): drafter attention backend — selects the
    # attention kernel the drafter runs with. vLLM v1 SpeculativeConfig
    # accepts an `attention_backend` key; the validated β'-A K=4
    # configuration explicitly sets this to FLASH_ATTN so the drafter
    # routes head_size 256 / 512 layers through G4_71b / G4_75
    # respectively. Without this key the drafter falls back to vLLM's
    # auto-pick (typically TURBOQUANT on a TQ engine) which causes a
    # KV layout / kernel mismatch and breaks acceptance.
    #
    # Known values: FLASH_ATTN | TRITON_ATTN | TURBOQUANT | None.
    # Default None preserves backward compat: pre-P1.7c configs that
    # never set this field render bit-identically to their prior
    # behaviour. Operator opts in via the spec_decode block on
    # ModelDef or via ProfileDef.spec_decode_override.
    attention_backend: Optional[str] = None

    def validate(self) -> None:
        valid_methods = {"mtp", "eagle", "ngram", "dflash"}
        if self.method not in valid_methods:
            raise SchemaError(
                f"SpecDecodeConfig.method must be one of {valid_methods}, "
                f"got '{self.method}'"
            )
        if self.num_speculative_tokens < 1:
            raise SchemaError(
                "SpecDecodeConfig.num_speculative_tokens must be >= 1"
            )
        if self.method in ("dflash", "eagle") and not self.model:
            raise SchemaError(
                f"SpecDecodeConfig.model is required for method='{self.method}' "
                f"(drafter is a separate checkpoint from the target model)"
            )
        # The native probabilistic draft path is documented in
        # vllm.v1.spec_decode.llm_base_proposer at the
        # `_enable_probabilistic_draft_probs` gate. The known-good values
        # at the dev338 pin are listed below; new values become possible
        # if upstream extends the enum without our knowing.
        valid_rejection = {None, "standard"}
        valid_draft_sample = {None, "probabilistic"}
        if self.rejection_sample_method not in valid_rejection:
            raise SchemaError(
                "SpecDecodeConfig.rejection_sample_method must be one of "
                f"{valid_rejection}, got {self.rejection_sample_method!r}"
            )
        if self.draft_sample_method not in valid_draft_sample:
            raise SchemaError(
                "SpecDecodeConfig.draft_sample_method must be one of "
                f"{valid_draft_sample}, got {self.draft_sample_method!r}"
            )
        # P1.7c: attention_backend validates against the known set of
        # vLLM v1 attention backends. None means "do not emit the key"
        # (engine auto-picks).
        valid_attn_backend = {
            None, "FLASH_ATTN", "TRITON_ATTN", "TURBOQUANT",
        }
        if self.attention_backend not in valid_attn_backend:
            raise SchemaError(
                "SpecDecodeConfig.attention_backend must be one of "
                f"{valid_attn_backend}, got {self.attention_backend!r}"
            )

    def to_vllm_arg(self) -> str:
        """Format for --speculative-config flag."""
        d: dict = {
            "method": self.method,
            "num_speculative_tokens": self.num_speculative_tokens,
        }
        if self.model:
            d["model"] = self.model
        # Probabilistic draft-probs path — only emit when both halves of
        # the gate are set, so older vllm pins that lack the upstream
        # native path do not see unknown keys in the config blob.
        if self.rejection_sample_method is not None:
            d["rejection_sample_method"] = self.rejection_sample_method
        if self.draft_sample_method is not None:
            d["draft_sample_method"] = self.draft_sample_method
        # P1.7c: drafter attention backend (FLASH_ATTN, TRITON_ATTN,
        # TURBOQUANT). Emitted only when set; absent key = vLLM auto-pick.
        if self.attention_backend is not None:
            d["attention_backend"] = self.attention_backend
        return json.dumps(d)
