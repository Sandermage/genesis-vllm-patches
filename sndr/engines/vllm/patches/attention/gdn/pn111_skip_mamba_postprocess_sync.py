# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN111 — skip the blocking ``num_accepted_tokens``
GPU→CPU sync when ``postprocess_mamba`` is provably a no-op.

Backport of [vllm#42574](https://github.com/vllm-project/vllm/pull/42574)
by `mamingyuan-nv` (OPEN at the time of backport).

================================================================
WHAT THIS PATCH DOES
================================================================

In ``GPUModelRunner._update_states_after_model_execute``, the
``mamba_cache_mode == "align"`` branch performs a per-decode-step
``self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()`` synchronous
copy because the values then feed ``postprocess_mamba``. The copy is
a *blocking* device-to-host transfer that stalls the host before the
draft-model forward can be issued.

``postprocess_mamba`` only mutates state inside the inner conditional

    aligned_new_computed_tokens >= num_tokens_running_state

i.e. only when at least one request just crossed a Mamba block
boundary. The PR adds a CPU-side predicate
``can_skip_mamba_postprocess`` that uses the upper bound
``num_accepted <= n_draft + 1`` to prove that no boundary will be
crossed *this* step. When the predicate fires (the common case), the
host kicks off a non-blocking copy + records
``num_accepted_tokens_event`` — the existing event is already
synchronized in ``_prepare_inputs`` after the draft forward, so the
absorption is free.

Reported: ``+17.4 % TPS`` / ``-13.7 % ITL`` on Nemotron-Super-120B
NVFP4 + MTP=3 on GB300.

================================================================
RELEVANCE FOR GENESIS
================================================================

The PR targets ``hybrid model + MTP K>=1 + align mode``. Our 27B
Qwen3.6 hybrid GDN + MTP K=3 stack is exactly the topology — **but**
our active PROD configs (``prod-qwen3.6-27b-tq-k8v4`` / ``a5000-2x-27b-int4-tq-
k8v4.yaml``) do **not** pass ``--mamba-cache-mode align``. Only the
historical / archived launch scripts (`start_v755/756/757_*.sh`) ran
align mode.

Today the entire ``if self.cache_config.mamba_cache_mode == "align":``
arm is dormant in PROD (the ``else`` branch already does a non-
blocking copy + event-record). PN111 is therefore **default OFF** —
the win materialises only after an operator flips on align mode
(typical reasons: hybrid prefix-cache reuse, multi-turn cache hit on
the SSM section). When that happens the predicate gives roughly the
same single-digit-percent TPS lift our 27B path would otherwise lose
to the blocking sync.

Composes orthogonally with our existing align-aware patches
(PN30 DS-layout fix; P60 / P63 GDN+ngram state recovery; PN82 Mamba
cudagraph prefill-zero). None of them touch the ``num_accepted_tokens
.gpu[:num_reqs].cpu().numpy()`` site, so anchors do not collide.

================================================================
SAFETY MODEL
================================================================

- Bit-identical when ``can_skip_mamba_postprocess`` correctly proves
  the no-op case (proven by the inner conditional of
  ``postprocess_mamba`` — see PR diff).
- Falls back to the original blocking-sync code path whenever the
  predicate returns False (any boundary crossing possible).
- Default OFF; enabled per-preset via ``GENESIS_ENABLE_PN111=1``.
- Idempotent via Genesis marker.
- Drift-marker watches the upstream-merged form of
  ``can_skip_mamba_postprocess`` so the patch self-skips after a
  pin-bump that absorbs the fix.

================================================================

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#42574.
"""
from __future__ import annotations

import logging

from sndr.kernel import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
    TextPatchResult,
)
from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn111_skip_mamba_postprocess_sync")

GENESIS_PN111_MARKER = (
    "Genesis PN111 skip-mamba-postprocess sync (vllm#42574) v11.0.0"
)


# ── Part 1: gpu_model_runner.py — replace the blocking-copy block ─────
# Anchor matches the current pin's literal source. 8-space indent
# (method body inside `GPUModelRunner`).
PN111_RUNNER_OLD = (
    '        if self.cache_config.mamba_cache_mode == "align":\n'
    '            for i, num_tokens in enumerate(\n'
    '                self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()\n'
    '            ):\n'
    '                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens\n'
    '            mamba_utils.postprocess_mamba(\n'
    '                scheduler_output,\n'
    '                self.kv_cache_config,\n'
    '                self.kv_caches,\n'
    '                self.input_batch,\n'
    '                self.mamba_state_idx,\n'
    '                self.compilation_config.static_forward_context,\n'
    '                self.model.get_mamba_state_copy_func(),\n'
    '                self._get_mamba_copy_bufs(),\n'
    '            )\n'
)

PN111_RUNNER_NEW = (
    '        if self.cache_config.mamba_cache_mode == "align":\n'
    '            # ════════════════════════════════════════════════════════\n'
    '            # [Genesis PN111 vllm#42574 backport] Skip the blocking\n'
    '            # GPU->CPU sync of num_accepted_tokens when the downstream\n'
    '            # postprocess_mamba is provably a no-op this step. The\n'
    '            # predicate uses CPU-side state (num_computed_tokens +\n'
    '            # num_scheduled + n_draft) and the upper bound\n'
    '            # num_accepted <= n_draft + 1 to prove no Mamba block\n'
    '            # boundary will be crossed. Fast path: async copy +\n'
    '            # event-record (absorbed by _prepare_inputs sync after\n'
    '            # the draft forward). Slow path: original behaviour.\n'
    '            # ════════════════════════════════════════════════════════\n'
    '            copy_bufs = self._get_mamba_copy_bufs()\n'
    '            _bs = getattr(copy_bufs.mamba_spec, "block_size", None)\n'
    '            if _bs is not None and mamba_utils.can_skip_mamba_postprocess(\n'
    '                scheduler_output,\n'
    '                self.input_batch,\n'
    '                self.requests,\n'
    '                _bs,\n'
    '                num_reqs,\n'
    '            ):\n'
    '                # Async device-to-host; event-record is absorbed by\n'
    '                # the existing event.synchronize() in _prepare_inputs.\n'
    '                self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(\n'
    '                    self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True\n'
    '                )\n'
    '                if getattr(self, "num_accepted_tokens_event", None) is not None:\n'
    '                    self.num_accepted_tokens_event.record()\n'
    '                return\n'
    '            # Slow path: at least one request may cross a Mamba block\n'
    '            # boundary — original blocking-sync behaviour preserved.\n'
    '            for i, num_tokens in enumerate(\n'
    '                self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()\n'
    '            ):\n'
    '                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens\n'
    '            mamba_utils.postprocess_mamba(\n'
    '                scheduler_output,\n'
    '                self.kv_cache_config,\n'
    '                self.kv_caches,\n'
    '                self.input_batch,\n'
    '                self.mamba_state_idx,\n'
    '                self.compilation_config.static_forward_context,\n'
    '                self.model.get_mamba_state_copy_func(),\n'
    '                copy_bufs,\n'
    '            )\n'
)


# ── Part 2: mamba_utils.py — insert the predicate above postprocess_mamba ─
# Anchor on the existing `def postprocess_mamba(` signature line. We
# emit the new function definition plus a blank line plus the original
# signature so the replacement is self-contained.
PN111_UTILS_OLD = (
    'def postprocess_mamba(\n'
)

PN111_UTILS_NEW = (
    '# ════════════════════════════════════════════════════════════════\n'
    '# [Genesis PN111 vllm#42574 backport] CPU-only predicate that\n'
    '# decides whether postprocess_mamba can be skipped this step.\n'
    '# Coupled to the inner conditional of postprocess_mamba below —\n'
    '# keep them in lockstep.\n'
    '# ════════════════════════════════════════════════════════════════\n'
    'def can_skip_mamba_postprocess(\n'
    '    scheduler_output,\n'
    '    input_batch,\n'
    '    requests,\n'
    '    mamba_block_size,\n'
    '    num_reqs,\n'
    '):\n'
    '    """Return True iff `postprocess_mamba` is provably a no-op this step."""\n'
    '    if not mamba_block_size or mamba_block_size <= 0:\n'
    '        return False\n'
    '    num_scheduled = scheduler_output.num_scheduled_tokens\n'
    '    spec_decode = scheduler_output.scheduled_spec_decode_tokens\n'
    '    req_ids = input_batch.req_ids\n'
    '    for i in range(num_reqs):\n'
    '        req_id = req_ids[i]\n'
    '        req_state = requests[req_id]\n'
    '        n_draft = len(spec_decode.get(req_id, ()))\n'
    '        n_running = (\n'
    '            req_state.num_computed_tokens\n'
    '            + num_scheduled[req_id]\n'
    '            - n_draft\n'
    '        )\n'
    '        max_new = n_running + n_draft\n'
    '        if (max_new // mamba_block_size) * mamba_block_size >= n_running:\n'
    '            return False\n'
    '    return True\n'
    '\n'
    '\n'
    'def postprocess_mamba(\n'
)


def _make_patcher_runner() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN111 v1/worker/gpu_model_runner.py — skip-mamba-postprocess "
            "sync (vllm#42574)"
        ),
        target_file=str(target),
        marker=GENESIS_PN111_MARKER + " [runner]",
        sub_patches=[
            TextPatch(
                name="pn111_align_branch_skip_path",
                anchor=PN111_RUNNER_OLD,
                replacement=PN111_RUNNER_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN111",
            "can_skip_mamba_postprocess(",
        ],
    )


def _make_patcher_utils() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/mamba_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN111 v1/worker/mamba_utils.py — can_skip predicate (vllm#42574)",
        target_file=str(target),
        marker=GENESIS_PN111_MARKER + " [utils]",
        sub_patches=[
            TextPatch(
                name="pn111_can_skip_predicate",
                anchor=PN111_UTILS_OLD,
                replacement=PN111_UTILS_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN111",
            "def can_skip_mamba_postprocess(",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN111 — skip the blocking mamba postprocess sync.

    Two-file transaction. If either anchor fails, NEITHER file is
    written (MultiFilePatchTransaction semantics). Default OFF.
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN111")
    log_decision("PN111", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher_runner = _make_patcher_runner()
    patcher_utils = _make_patcher_utils()
    if patcher_runner is None or patcher_utils is None:
        return "skipped", "vllm v1/worker/{gpu_model_runner,mamba_utils}.py not found"

    # Idempotent + upstream-merge short-circuit on either file.
    for patcher in (patcher_runner, patcher_utils):
        try:
            with open(patcher.target_file) as f:
                content = f.read()
        except OSError:
            return "skipped", f"target read failed: {patcher.target_file}"
        if patcher.marker in content:
            log.info("[PN111] marker present on %s — skip", patcher.target_file)
            return "applied", "idempotent (marker present)"
        # Upstream-merged check (other file may not have it yet, that's OK)
        for m in patcher.upstream_drift_markers:
            if m.startswith("[Genesis"):
                continue
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} present in "
                    f"{patcher.target_file} — upstream PR #42574 "
                    "appears merged",
                )

    txn = MultiFilePatchTransaction(
        name="PN111 skip-mamba-postprocess sync (2-file)",
        patchers=[patcher_runner, patcher_utils],
    )
    result, failure = txn.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"PN111 transaction: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            "PN111 transaction: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return (
        "applied",
        "PN111 applied: align-mode mamba postprocess now uses CPU-only "
        "predicate to skip the blocking num_accepted_tokens sync when "
        "no Mamba block boundary will be crossed. ~+15% TPS *if* the "
        "preset enables --mamba-cache-mode align; no-op otherwise."
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    p = _make_patcher_runner()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return p.marker in f.read()
    except OSError:
        return False
