# SPDX-License-Identifier: Apache-2.0
"""PN381 companion — port of the vllm#44742 regression test SHAPE
through OUR PN369/P71 rewritten rejection-sampler paths (torch group).

Upstream's regression test (``test_allowed_token_ids_without_output_
token_ids`` in tests/v1/sample/test_rejection_sampler.py, CUDA) builds
the GHSA-8c65-hq7q-r7jm shape — sampling metadata with
``output_token_ids == []``, a non-empty ``allowed_token_ids_mask`` and
active draft-token counts — and asserts ``apply_logits_processors``
masks every draft-expanded row by its REQUEST's mask row (row parity
via ``repeat_interleave(num_draft_tokens)``).

This port runs the same shape ON CPU against the rejection_sampler
module AS GENESIS SHIPS IT: the pristine pin file with the PN369
(relaxed acceptance) and P71 (block-verify) text patches applied,
loaded as a standalone module. Coverage upstream never had (roadmap
chunk-5 synergy note): proves our sampler rewrite preserves the
consumer-side row-parity contract that PN381's producer-side hardening
defends. A consistency twin runs the SAME logits with
``output_token_ids`` POPULATED (the post-PN381 producer state) and
asserts bit-identical masking.

Companion to ``test_pn381_allowed_token_ids_spec_metadata.py``
(torch-less group). This file imports torch at module level and is
auto-skipped on torch-less hosts by the tests/conftest.py AST scan;
run it inside the vLLM container (or any torch+vllm-capable host —
CUDA NOT required):

  python3 -m pytest \
      tests/unit/integrations/spec_decode/test_pn381_sampler_regression_torch.py -v

Requires the pristine pin tree at /private/tmp/candidate_pin_current
(same contract as tools/pin_preflight.py) — skipped when absent.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

# Torch-gated regression test: skip cleanly in a torch-less collection
# environment (CI / dispatcher-only runs) instead of breaking import
# (same importorskip convention as the pn340 torch guard, commit
# b0229923).
torch = pytest.importorskip(
    "torch", reason="requires torch for the rejection-sampler regression run"
)

pytest.importorskip(
    "vllm", reason="requires an installed vllm matching the candidate pin"
)

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm")
PIN_REJECTION_SAMPLER = PIN_TREE / "v1" / "sample" / "rejection_sampler.py"

pytestmark = pytest.mark.skipif(
    not PIN_REJECTION_SAMPLER.is_file(),
    reason="pristine pin tree not present on this machine",
)

DEVICE = torch.device("cpu")
VOCAB_SIZE = 100


class _FakeSampler:
    """Minimal stand-in for vllm.v1.sample.sampler.Sampler — the
    upstream test uses Mock(spec=Sampler) with the same single
    attribute; apply_logits_processors never touches the sampler."""

    logprobs_mode = "raw_logprobs"


@pytest.fixture(scope="module")
def genesis_sampler_module(tmp_path_factory):
    """The pin's rejection_sampler.py with PN369 + P71 applied, loaded
    as a standalone module (exec-patched-text technique)."""
    from sndr.engines.vllm.patches.spec_decode import p71_block_verify as P71
    from sndr.engines.vllm.patches.spec_decode import (
        pn369_relaxed_acceptance as PN369,
    )
    from sndr.kernel import TextPatchResult

    tmp_dir = tmp_path_factory.mktemp("pn381_sampler")
    target = tmp_dir / "rejection_sampler.py"
    target.write_text(
        PIN_REJECTION_SAMPLER.read_text(encoding="utf-8"), encoding="utf-8"
    )

    for module in (PN369, P71):
        original = module.resolve_vllm_file
        module.resolve_vllm_file = lambda rel: str(target)  # type: ignore[assignment]
        try:
            patcher = module._make_patcher()
            assert patcher is not None
            result, failure = patcher.apply()
            assert result == TextPatchResult.APPLIED, (
                module.__name__,
                failure,
            )
        finally:
            module.resolve_vllm_file = original  # type: ignore[assignment]

    patched = target.read_text(encoding="utf-8")
    # Sanity: we are testing the REWRITTEN paths, not pristine upstream.
    assert PN369.GENESIS_PN369_MARKER.split(" v")[0] in patched or (
        "[Genesis PN369" in patched
    )
    assert "[Genesis P71" in patched

    spec = importlib.util.spec_from_file_location(
        "genesis_pn381_patched_rejection_sampler", str(target)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rejection_sampler(genesis_sampler_module):
    return genesis_sampler_module.RejectionSampler(_FakeSampler())


# ─────────────────────────────────────────────────────────────────────
# Helpers — faithful ports of the upstream test fixtures (CPU)
# ─────────────────────────────────────────────────────────────────────


def create_logits_tensor(
    output_token_ids: list[list[int]],
    vocab_size: int = VOCAB_SIZE,
    token_idx_to_override: int | None = None,
) -> torch.Tensor:
    """Port of the upstream helper: logits that argmax to the desired
    tokens, one row per draft token (last token of each row is the
    bonus and gets no logits row)."""
    token_ids = [tokens[:-1] for tokens in output_token_ids]
    num_total_tokens = sum(len(tokens) for tokens in token_ids)
    logits = torch.full((num_total_tokens, vocab_size), -100.0, device=DEVICE)
    start_loc = 0
    for tokens in token_ids:
        for j, token_id in enumerate(tokens):
            logits[start_loc + j, token_id] = 100.0
        start_loc += len(tokens)
    if token_idx_to_override:
        logits[:, token_idx_to_override] = 99.0
    return logits


def create_allowed_token_ids(
    batch_size: int,
    vocab_size: int,
    num_allowed_token_ids: int,
) -> torch.Tensor:
    """Port of tests/v1/sample/utils.create_allowed_token_ids: True
    entries mark DISALLOWED vocab positions (masked_fill to -inf);
    odd-indexed requests carry an all-False row."""
    mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=DEVICE)
    for i in range(batch_size):
        if i % 2 == 1:
            continue
        start = min(i, vocab_size - 1)
        end = min(i + num_allowed_token_ids, vocab_size - 1)
        mask[i, start:end] = True
    return mask


def create_sampling_metadata(
    genesis_sampler_module,
    all_greedy: bool,
    output_token_ids: list[list[int]] | None = None,
    spec_token_ids: list[list[int]] | None = None,
    allowed_token_ids_mask: torch.Tensor | None = None,
):
    """Field-introspective port of the upstream helper — builds the
    pin's SamplingMetadata with the GHSA shape, tolerating optional
    trailing fields drifting across pins."""
    from vllm.v1.sample.logits_processor import LogitsProcessors

    SamplingMetadata = genesis_sampler_module.SamplingMetadata
    kwargs = {
        "temperature": None,
        "all_greedy": all_greedy,
        "all_random": not all_greedy,
        "top_p": None,
        "top_k": None,
        "generators": {},
        "max_num_logprobs": None,
        "no_penalties": True,
        "prompt_token_ids": None,
        "frequency_penalties": torch.tensor([]),
        "presence_penalties": torch.tensor([]),
        "repetition_penalties": torch.tensor([]),
        "output_token_ids": [] if output_token_ids is None else output_token_ids,
        "spec_token_ids": [] if spec_token_ids is None else spec_token_ids,
        "allowed_token_ids_mask": allowed_token_ids_mask,
        "bad_words_token_ids": {},
        "logitsprocs": LogitsProcessors(),
    }
    field_names = {f.name for f in dataclasses.fields(SamplingMetadata)}
    return SamplingMetadata(
        **{k: v for k, v in kwargs.items() if k in field_names}
    )


def create_spec_decode_metadata(genesis_sampler_module, spec_tokens, logits):
    metadata = genesis_sampler_module.SpecDecodeMetadata.make_dummy(
        spec_tokens, device=logits.device
    )
    metadata.target_logits_indices = torch.arange(logits.shape[0])
    metadata.bonus_logits_indices = torch.empty(0, dtype=torch.int32)
    return metadata


def _assert_row_parity_masking(result_logits, mask, num_draft_tokens):
    """The #44742 invariant: every draft-expanded logits row is masked
    by ITS request's mask row (not by output_token_ids-derived rows)."""
    batch_size = len(num_draft_tokens)
    repeat_indices = torch.arange(batch_size).repeat_interleave(
        torch.tensor(num_draft_tokens)
    )
    assert result_logits.shape[0] == int(repeat_indices.shape[0])
    for row_idx in range(result_logits.shape[0]):
        req_idx = int(repeat_indices[row_idx].item())
        disallowed = mask[req_idx]
        assert torch.all(
            result_logits[row_idx, disallowed] == float("-inf")
        ), f"row {row_idx} (req {req_idx}) not masked"
        # Allowed positions must survive untouched.
        assert torch.all(result_logits[row_idx, ~disallowed] > float("-inf"))


# ─────────────────────────────────────────────────────────────────────
# The ported regression test (GHSA-8c65-hq7q-r7jm shape)
# ─────────────────────────────────────────────────────────────────────


def test_allowed_token_ids_without_output_token_ids(
    genesis_sampler_module, rejection_sampler
):
    """Faithful port of upstream #44742's regression test, CPU, against
    the PN369/P71-patched module: empty output_token_ids + non-empty
    allowed_token_ids_mask + active draft-token metadata."""
    spec_tokens = [[1, 2, 10], [10, 5, 3]]
    output_tokens = [[1, 2, 10, 5], [10, 5, 10, 5]]
    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)
    batch_size = len(output_tokens)
    _, vocab_size = logits.size()
    mask = create_allowed_token_ids(
        batch_size=batch_size,
        vocab_size=vocab_size,
        num_allowed_token_ids=5,
    )
    sampling_metadata = create_sampling_metadata(
        genesis_sampler_module,
        all_greedy=True,
        output_token_ids=None,
        spec_token_ids=spec_tokens,
        allowed_token_ids_mask=mask,
    )
    assert sampling_metadata.output_token_ids == []

    spec_decode_metadata = create_spec_decode_metadata(
        genesis_sampler_module, spec_tokens, logits
    )
    assert len(spec_decode_metadata.num_draft_tokens) == batch_size

    result_logits = rejection_sampler.apply_logits_processors(
        logits.clone(),
        sampling_metadata,
        spec_decode_metadata,
    )
    _assert_row_parity_masking(
        result_logits, mask, spec_decode_metadata.num_draft_tokens
    )


def test_allowed_token_ids_with_populated_output_token_ids(
    genesis_sampler_module, rejection_sampler
):
    """Consistency twin: the post-PN381 producer state (output_token_ids
    POPULATED for the same batch) must yield bit-identical masking —
    proving the consumer is invariant to the producer-side hardening."""
    spec_tokens = [[1, 2, 10], [10, 5, 3]]
    output_tokens = [[1, 2, 10, 5], [10, 5, 10, 5]]
    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)
    batch_size = len(output_tokens)
    _, vocab_size = logits.size()
    mask = create_allowed_token_ids(
        batch_size=batch_size,
        vocab_size=vocab_size,
        num_allowed_token_ids=5,
    )
    spec_decode_metadata = create_spec_decode_metadata(
        genesis_sampler_module, spec_tokens, logits
    )

    md_empty = create_sampling_metadata(
        genesis_sampler_module,
        all_greedy=True,
        output_token_ids=None,
        spec_token_ids=spec_tokens,
        allowed_token_ids_mask=mask,
    )
    md_populated = create_sampling_metadata(
        genesis_sampler_module,
        all_greedy=True,
        output_token_ids=[[1, 2], [10, 5]],
        spec_token_ids=spec_tokens,
        allowed_token_ids_mask=mask,
    )

    out_empty = rejection_sampler.apply_logits_processors(
        logits.clone(), md_empty, spec_decode_metadata
    )
    out_populated = rejection_sampler.apply_logits_processors(
        logits.clone(), md_populated, spec_decode_metadata
    )
    assert torch.equal(out_empty, out_populated)
    _assert_row_parity_masking(
        out_populated, mask, spec_decode_metadata.num_draft_tokens
    )


def test_no_mask_leaves_logits_untouched(
    genesis_sampler_module, rejection_sampler
):
    """Fast-path guard: without allowed_token_ids_mask (and with no
    penalties/bad words) apply_logits_processors is the identity."""
    spec_tokens = [[1, 2, 10], [10, 5, 3]]
    output_tokens = [[1, 2, 10, 5], [10, 5, 10, 5]]
    logits = create_logits_tensor(output_tokens)
    sampling_metadata = create_sampling_metadata(
        genesis_sampler_module,
        all_greedy=True,
        spec_token_ids=spec_tokens,
        allowed_token_ids_mask=None,
    )
    spec_decode_metadata = create_spec_decode_metadata(
        genesis_sampler_module, spec_tokens, logits
    )
    result = rejection_sampler.apply_logits_processors(
        logits.clone(), sampling_metadata, spec_decode_metadata
    )
    assert torch.equal(result, logits)


def test_consumer_sizes_by_num_draft_tokens_not_output_token_ids(
    genesis_sampler_module,
):
    """Textual pin-shape guard (#35654): the patched module must still
    derive the request count from metadata.num_draft_tokens — the
    consumer-side half of the GHSA defense PN381 layers on top of."""
    import inspect

    src = inspect.getsource(
        genesis_sampler_module.RejectionSampler.apply_logits_processors
    )
    assert "len(metadata.num_draft_tokens)" in src
    assert "len(output_token_ids)" not in src
