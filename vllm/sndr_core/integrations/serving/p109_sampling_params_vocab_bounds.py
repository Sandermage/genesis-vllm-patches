# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 109 — sampling_params vocab-range validation.

Backport of [vllm#42614](https://github.com/vllm-project/vllm/pull/42614)
by `jperezdealgaba` (OPEN at the time of backport).

================================================================
WHAT THIS PATCH DOES
================================================================

``SamplingParams.verify()`` currently checks ``allowed_token_ids`` and
``logit_bias`` against the model vocabulary but does NOT validate
``stop_token_ids`` or ``logprob_token_ids``. An out-of-vocab id
reaches the V2 Triton sampler ``_bias_kernel`` which performs an
unchecked ``tl.store(logits_ptr + token_idx * logits_stride +
stop_token_ids, ...)`` and triggers a device-side ``illegal memory
access`` or ``device-side assert``. From an operator's perspective
this looks like a hard worker crash on the very first request that
carries a malformed payload — no graceful 4xx response.

The fix is two-part:

1. Add ``_validate_stop_token_ids`` and ``_validate_logprob_token_ids``
   methods on ``SamplingParams`` and call them from ``verify()``. Both
   raise ``VLLMValidationError`` with a clear "out of vocab" message
   so the request bounces with a 400 instead of crashing the worker.
2. As defence-in-depth inside ``_bias_kernel``, mask the offending
   token slot before the store so a future caller bug cannot OOB the
   GPU either. One extra triton ``mask &=`` line, no perf impact.

================================================================
RELEVANCE FOR GENESIS
================================================================

Our public surface (Proxy-AI gateway + LibreChat / OpenWebUI / Cline /
OpenCode) accepts user-controlled ``stop_token_ids`` and
``logprob_token_ids``. Qwen3.6 has a ~152 K vocab, so an out-of-range
id is plausible at the JSON-schema boundary (clients sometimes pass
``stop_token_ids: [-1]`` or arbitrary integers). No existing Genesis
patch hardens this surface — this is a fresh defensive win.

================================================================
SAFETY MODEL
================================================================

- Validation runs once per request inside ``verify()`` — same cost
  envelope as the existing ``_validate_logit_bias`` / ``_validate_
  allowed_token_ids`` calls (one O(N) pass over the user list).
- The triton mask edit is a no-op for in-range ids (most requests).
- Bit-identical for valid inputs; valid clients see no behaviour
  change.
- Drift-marker watches the canonical upstream method name so the
  patch self-skips when the fix lands in our pin.

================================================================

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#42614.
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.core import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)
from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.p109_sampling_params_vocab_bounds")

GENESIS_P109_MARKER = (
    "Genesis P109 sampling_params vocab-range validation (vllm#42614) v11.0.0"
)


# ── Part 1: sampling_params.py ────────────────────────────────────────
# Anchor on the existing call chain in `verify`. Insert two new method
# invocations BEFORE _validate_logits_processors so any vocab-OOB raises
# with a clear message before downstream code touches the value.
P109_VERIFY_OLD = (
    "        self._validate_logprobs(model_config)\n"
    "        self._validate_logit_bias(model_config)\n"
    "        self._validate_logits_processors(model_config)\n"
)

P109_VERIFY_NEW = (
    "        self._validate_logprobs(model_config)\n"
    "        self._validate_logit_bias(model_config)\n"
    "        # [Genesis P109 vllm#42614 backport] vocab-range guard\n"
    "        self._validate_stop_token_ids(model_config)\n"
    "        self._validate_logprob_token_ids(model_config)\n"
    "        self._validate_logits_processors(model_config)\n"
)

# Insert the two new methods immediately after _validate_logit_bias.
# Anchor on the final closing of `_validate_logit_bias` (a unique
# 2-line shape) and emit the same plus the new methods.
P109_METHODS_OLD = (
    '                value=invalid_token_ids,\n'
    '            )\n'
    '\n'
    '    def _validate_logits_processors(self, model_config: ModelConfig) -> None:\n'
)

P109_METHODS_NEW = (
    '                value=invalid_token_ids,\n'
    '            )\n'
    '\n'
    '    # ════════════════════════════════════════════════════════════════\n'
    '    # [Genesis P109 vllm#42614 backport] vocab-range validators —\n'
    '    # forbid out-of-vocab stop_token_ids / logprob_token_ids that\n'
    '    # would OOB the V2 Triton _bias_kernel. Raise VLLMValidationError\n'
    '    # so the request bounces with a 400 instead of crashing the\n'
    '    # worker. Bit-identical for valid inputs.\n'
    '    # ════════════════════════════════════════════════════════════════\n'
    '    def _validate_stop_token_ids(self, model_config: "ModelConfig") -> None:\n'
    '        if not self.stop_token_ids:\n'
    '            return\n'
    '        vocab_size = model_config.get_vocab_size()\n'
    '        invalid_token_ids = [\n'
    '            t for t in self.stop_token_ids\n'
    '            if t < 0 or t >= vocab_size\n'
    '        ]\n'
    '        if invalid_token_ids:\n'
    '            raise VLLMValidationError(\n'
    '                f"stop_token_ids contains out-of-vocab token id(s) "\n'
    '                f"{invalid_token_ids}. Vocabulary size: {vocab_size}",\n'
    '                parameter="stop_token_ids",\n'
    '                value=invalid_token_ids,\n'
    '            )\n'
    '\n'
    '    def _validate_logprob_token_ids(self, model_config: "ModelConfig") -> None:\n'
    '        if not getattr(self, "logprob_token_ids", None):\n'
    '            return\n'
    '        vocab_size = model_config.get_vocab_size()\n'
    '        invalid_token_ids = [\n'
    '            t for t in self.logprob_token_ids\n'
    '            if t < 0 or t >= vocab_size\n'
    '        ]\n'
    '        if invalid_token_ids:\n'
    '            raise VLLMValidationError(\n'
    '                f"logprob_token_ids contains out-of-vocab token id(s) "\n'
    '                f"{invalid_token_ids}. Vocabulary size: {vocab_size}",\n'
    '                parameter="logprob_token_ids",\n'
    '                value=invalid_token_ids,\n'
    '            )\n'
    '\n'
    '    def _validate_logits_processors(self, model_config: ModelConfig) -> None:\n'
)


def _make_patcher_sampling_params() -> TextPatcher | None:
    target = resolve_vllm_file("sampling_params.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P109 sampling_params.py — vocab-range validators (vllm#42614)",
        target_file=str(target),
        marker=GENESIS_P109_MARKER,
        sub_patches=[
            TextPatch(
                name="p109_verify_call_chain",
                anchor=P109_VERIFY_OLD,
                replacement=P109_VERIFY_NEW,
                required=True,
            ),
            TextPatch(
                name="p109_validator_methods",
                anchor=P109_METHODS_OLD,
                replacement=P109_METHODS_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P109",
            "_validate_stop_token_ids",  # upstream-merged form
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P109 — sampling_params vocab-range validation."""
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P109")
    log_decision("P109", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher_sampling_params()
    if patcher is None:
        return "skipped", "vllm/sampling_params.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} present — upstream "
                "PR #42614 (or equivalent fix) appears merged",
            )

    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return (
        "applied",
        "P109 applied: stop_token_ids + logprob_token_ids now validated "
        "against vocab size. Out-of-range ids → 400 with clear message "
        "instead of GPU illegal-memory-access crash. ~0 perf overhead."
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher_sampling_params()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except OSError:
        return False
