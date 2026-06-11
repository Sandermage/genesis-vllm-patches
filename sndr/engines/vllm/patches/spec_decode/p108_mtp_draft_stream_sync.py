# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 108 — MTP draft-loop stream synchronization.

Backport of [vllm#42603](https://github.com/vllm-project/vllm/pull/42603)
by `z1ying` (OPEN at the time of backport).

================================================================
WHAT THIS PATCH DOES
================================================================

In ``LLMBaseProposer.propose`` the spec-decode driver writes to two
shared CUDA-graph-captured buffers each step::

    self.input_ids[:batch_size]      = input_ids
    self.hidden_states[:batch_size]  = hidden_states

…and *immediately* hands control to attention-metadata construction
and the next draft-model forward. The two stores are async on the
default CUDA stream, but the consumer kernels can dispatch onto a
different stream (FlashInfer reuses captured graphs that recorded
event-based ordering, MTP draft model launches its own stream during
``model.forward``). Under high concurrency or FlashInfer-default
attention this race produces ``cudaErrorIllegalAddress`` because the
attention prologue reads the still-being-written ``input_ids`` buffer.

The fix is one line: synchronize the **current** stream after the
stores but before the next kernel launches. The synchronize is
issued on the recording stream only — not a global
``torch.cuda.synchronize()`` — so the CPU overhead is bounded to the
work already queued on that stream.

PR repro: ``Qwen3.6-27B-FP8`` on RTX 5090 (sm_120) + FlashInfer +
``max_num_seqs >= 8`` reliably IMA'd on the third or fourth multi-turn
request; with this patch the same workload runs >60 minutes clean.
``CUDA_LAUNCH_BLOCKING=1`` was confirmed to mask the race, ruling out
data-corruption causes.

================================================================
RELEVANCE FOR GENESIS
================================================================

Our hot path goes through ``LLMBaseProposer.propose`` on every MTP K=3
step (P67 / P67b TurboQuant verify routes the *target* model, but the
*drafter* still lives here). The same shared-buffer / async-store
pattern is present in our pin (``0.20.2rc1.dev338+gbf0d2dc6d``).

On 27B INT4 + TurboQuant attention backend the race has not (yet) been
observed at ``max_num_seqs=2`` because TQ's verify path issues its own
event-record before the drafter's first attention step. On 35B-A3B-FP8
+ FlashInfer with concurrent traffic the same race class applies and
the bug is upstream-confirmed against Qwen3.6 (validator-1).

We default the patch **ON** for any preset that uses MTP, EAGLE or
DFlash speculation. The added ``stream.synchronize()`` is essentially
free when the producer kernel has already finished by the time the
consumer wants the data (the common case in single-stream PROD); only
under genuine contention does it cost a hand-off, which is exactly the
case where the race fires.

================================================================
SAFETY MODEL
================================================================

- Pure synchronization barrier. No algorithmic change to draft, accept,
  or verify logic. Output sequences are bit-identical to vanilla.
- Idempotent via Genesis marker comment block.
- Drift-marker watches the *upstream-merged* form so the patch
  self-skips once the fix lands in our pin.
- Adds zero VRAM (no new allocations).

================================================================

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#42603.
"""
from __future__ import annotations

import logging
import os

from sndr.kernel import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)
from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.p108_mtp_draft_stream_sync")

GENESIS_P108_MARKER = (
    "Genesis P108 MTP draft-loop stream synchronization (vllm#42603) v2 backend-gated"
)


# Anchor: the three lines right before the multi-modal branch. Whitespace
# matters — these are 12-space indented (inside `propose` → `if not
# self.skip_propose:` block, then a second `else:` arm of an earlier
# conditional, hence the 12-space depth at our pin).
P108_OLD = (
    "            # copy inputs to buffer for cudagraph\n"
    "            self.input_ids[:batch_size] = input_ids\n"
    "            self.hidden_states[:batch_size] = hidden_states\n"
    "            if self.supports_mm_inputs:\n"
)

P108_NEW = (
    "            # copy inputs to buffer for cudagraph\n"
    "            self.input_ids[:batch_size] = input_ids\n"
    "            self.hidden_states[:batch_size] = hidden_states\n"
    "\n"
    "            # ════════════════════════════════════════════════════════════\n"
    "            # [Genesis P108 vllm#42603 backport] Wait for the input_ids /\n"
    "            # hidden_states writes to finish on device before constructing\n"
    "            # attention metadata or launching downstream kernels.\n"
    "            #\n"
    "            # The race observed upstream only fires when the consumer\n"
    "            # kernel runs on a DIFFERENT stream than the producer. In\n"
    "            # vllm that happens specifically with the FlashInfer attention\n"
    "            # backend, which builds its own metadata on a side stream.\n"
    "            # On same-stream backends (TurboQuant, FlashAttention 2/3,\n"
    "            # Triton unified) the next kernel is in-order on the default\n"
    "            # stream and no sync is needed — paying it anyway kills the\n"
    "            # CPU↔GPU pipeline that vllm relies on for spec-decode\n"
    "            # throughput (Genesis 2026-05-14 measurement on 27B INT4 +\n"
    "            # TurboQuant + MTP K=3: −14 % wall TPS when sync is\n"
    "            # unconditional).\n"
    "            #\n"
    "            # Gate the sync on the attention backend (auto-detected once\n"
    "            # and cached). Operator can force-enable via\n"
    "            # GENESIS_P108_FORCE_SYNC=1 (diagnostics / unknown backends)\n"
    "            # or force-disable via GENESIS_P108_FORCE_SYNC=0.\n"
    "            # ════════════════════════════════════════════════════════════\n"
    "            if getattr(self, '_genesis_p108_should_sync', None) is None:\n"
    "                import os as _p108_os\n"
    "                _force = _p108_os.environ.get('GENESIS_P108_FORCE_SYNC', '').lower()\n"
    "                if _force in ('1', 'true', 'yes', 'on'):\n"
    "                    self._genesis_p108_should_sync = True\n"
    "                elif _force in ('0', 'false', 'no', 'off'):\n"
    "                    self._genesis_p108_should_sync = False\n"
    "                else:\n"
    "                    # auto: enable for FlashInfer family, disable otherwise.\n"
    "                    _backend = _p108_os.environ.get('VLLM_ATTENTION_BACKEND', '').upper()\n"
    "                    self._genesis_p108_should_sync = _backend.startswith('FLASHINFER')\n"
    "            if self._genesis_p108_should_sync:\n"
    "                try:\n"
    "                    torch.accelerator.current_stream().synchronize()\n"
    "                except AttributeError:\n"
    "                    # torch < 2.5 → fallback to the CUDA-specific accessor.\n"
    "                    torch.cuda.current_stream().synchronize()\n"
    "\n"
    "            if self.supports_mm_inputs:\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/spec_decode/llm_base_proposer.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P108 v1/spec_decode/llm_base_proposer.py — MTP draft-loop "
            "stream synchronization (vllm#42603)"
        ),
        target_file=str(target),
        marker=GENESIS_P108_MARKER,
        sub_patches=[
            TextPatch(
                name="p108_mtp_draft_stream_sync",
                anchor=P108_OLD,
                replacement=P108_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P108",
            # Self-collision lint (triage plan §6 2026-06-11): former entry
            # "torch.accelerator.current_stream().synchronize()" is the
            # canonical upstream form baked verbatim by our own vllm#42603
            # backport — it cannot distinguish a real upstream merge from
            # our residue (false "upstream_merged" skip, PN369 class).
            # Real-merge detection via required-anchor mismatch (Layer 5)
            # + pin-bump preflight deep-diff.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P108 — MTP draft-loop stream synchronization."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P108")
    log_decision("P108", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/spec_decode/llm_base_proposer.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P108] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} present — upstream "
                "PR #42603 (or equivalent fix) appears merged",
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
        "P108 applied: stream.synchronize() inserted after input_ids / "
        "hidden_states writes in LLMBaseProposer.propose. Closes the "
        "cudaErrorIllegalAddress race observed on FlashInfer + MTP under "
        "concurrency. Bit-identical outputs, ~0 CPU overhead on the "
        "single-stream PROD path."
    )


def is_applied() -> bool:
    """Return True iff our marker is present in the target file."""
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except OSError:
        return False
