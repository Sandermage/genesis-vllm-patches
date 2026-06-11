# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 71 — block-verify rejection sampling (Sun 2024).

Backport of vllm-project/vllm#40819 (Z. Golpayegani, OPEN draft) implementing
the Sun et al. 2024 ICLR block verification rule (arXiv 2403.10444). Adds
opt-in branch in `rejection_sample()` that routes through our Genesis kernel
(`vllm/_genesis/kernels/block_verify_sampler.py`) when:

  - GENESIS_ENABLE_P71_BLOCK_VERIFY=1 (env opt-in)
  - max_spec_len >= 3 (Sun 2024 algorithm requires γ >= 3 for advantage)
  - draft_probs is not None (per-token probabilities required)
  - not synthetic_mode (synthetic-acceptance overrides verification rule)
  - not all_greedy (block rule degenerates to per-token at T=0)

================================================================
WHY THIS DIFFERS FROM PR #40819 AS-IS
================================================================

We backport with TWO critical bug-fixes from gemini-bot's review of #40819:

1. SHARED u per request (FIX 1):
   PR loads `uniform_prob = tl.load(uniform_probs_ptr + token_idx)` per
   position. Sun 2024 §3.2 requires ONE shared Bernoulli per block.

2. denom==0 → ACCEPT (FIX 2):
   PR returns `h_block = 0.0` when denom==0 (perfect draft match), which
   REJECTS perfect drafts. Must return 1.0 (always accept).

Both fixes are inside `vllm/_genesis/kernels/block_verify_sampler.py`.

================================================================
PN369 RELAXED-ACCEPTANCE THREADING (2026-06-10, marker v7.43)
================================================================

The injected branch now also computes the PN369 relaxed-acceptance mask
(`compute_relaxed_ok_mask` — returns None when
GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE is off) and threads it into
`call_block_verify_sample(relaxed_ok=...)` so the block-verify kernel can
TAIL-EXTEND the Sun-2024 accepted length while the relaxed window holds.
Bit-identical to v7.42 behavior when PN369 is disabled. See
`pn369_relaxed_acceptance.py` for the trade-off banner.

Marker bumped v7.42 -> v7.43: a container fs holding the v7.42 bake will
NOT re-patch in place (anchor already consumed); the operator must reset
the container fs (docker compose down && up -d, NOT stop/start) to pick
up the v7.43 text. On a stale fs P71 apply() reports SKIPPED with an
anchor-drift reason and the old v7.42 text stays active (without PN369
threading) — safe, just stale.

Also in v7.43: the kernel-side A4 precondition was FIXED (it required
len(cu_num_draft_tokens) == batch_size + 1, but upstream passes a cumsum
WITHOUT leading zero, i.e. [batch_size] — verified live at pin
0.22.1rc1.dev259). The old check made EVERY call_block_verify_sample
call raise -> permanent silent fallback to the per-token path. The
documented 27B failure ("length 1 must equal batch_size + 1 = 2") was
this contract bug, not a GQA shape issue.

================================================================
SAFETY MODEL
================================================================

The wiring text-patch wraps the entire block-verify call in try/except:
on any exception (kernel failure, shape mismatch, NaN, missing field), we
silently fall through to the upstream per-token path. Worst case: P71 is
silently no-op for that step. NO engine impact, NO output corruption.

Status: opt-in via `GENESIS_ENABLE_P71_BLOCK_VERIFY=1`. Default OFF.

Tunable knobs
-------------
- `GENESIS_ENABLE_P71_BLOCK_VERIFY` (default unset/0): master switch
- `GENESIS_P71_USE_PYTORCH` (default 0): force PyTorch reference (debug only)

Compatibility
-------------
- Cudagraph: PIECEWISE only (data-dependent loops inside kernel).
  P67b's FULL_AND_PIECEWISE for spec-decode K+1 is unaffected because
  rejection_sample runs OUTSIDE the captured graph (in sampler stage).
- MTP: works (draft_probs populated by MTP draft model)
- ngram: bypassed automatically (draft_probs is None for ngram method)

Risk acknowledgment
-------------------
- PR #40819 is open draft, subject to API change. Pin behavior to PR head SHA
  via marker (regenerate when upstream merges and validate).
- Realistic gain on 35B-A3B + Ampere: +0-3% wall-clock (PR's own Qwen3-32B
  parity bench). Treat as experimental.
- Unbiased: same target marginal as per-token rule (Sun 2024 §4 theorem).
  At T=0 (greedy) we skip P71 entirely (path returns earlier in upstream).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Bug-fixes: gemini-code-assist review on vllm#40819 (cited in kernel file).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p71_block_verify")

# v7.43 (2026-06-10): PN369 relaxed_ok threading added to the injected
# branch. Marker bump forces a fresh bake on clean container fs; stale
# v7.42 bakes SKIP (anchor consumed) and keep the old text — see docstring.
GENESIS_P71_MARKER = "Genesis P71 block-verify rejection sampling vllm#40819 v7.43_pn369_relaxed_tail"


# ─── Sub-patch: inject block-verify branch BEFORE sample_recovered_tokens ───
# Anchor on the comment + start of upstream's sample_recovered_tokens call,
# which is unique inside rejection_sample() at line ~459.

P71_OLD = (
    "    # Compute probability distribution from target logits.\n"
    "    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)\n"
    "    assert target_probs.is_contiguous()\n"
    "\n"
    "    # Sample recovered tokens for each position.\n"
    "    # [num_tokens]\n"
    "    recovered_token_ids = sample_recovered_tokens(\n"
)

P71_NEW = (
    "    # Compute probability distribution from target logits.\n"
    "    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)\n"
    "    assert target_probs.is_contiguous()\n"
    "\n"
    "    # ════════════════════════════════════════════════════════════════\n"
    "    # [Genesis P71 vllm#40819] Block-verify rejection sampling (Sun 2024)\n"
    "    # OPT-IN. Strictly >= per-token rule in expected accepted tokens.\n"
    "    # Bypasses on: greedy / synthetic / no-draft-probs / max_spec_len < 3.\n"
    "    # On any error, silently falls through to upstream per-token path.\n"
    "    # ════════════════════════════════════════════════════════════════\n"
    "    try:\n"
    "        import os as _genesis_p71_os\n"
    "        _genesis_p71_active = _genesis_p71_os.environ.get(\n"
    "            'GENESIS_ENABLE_P71_BLOCK_VERIFY', '').strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "        _genesis_p71_eligible = (\n"
    "            _genesis_p71_active\n"
    "            and max_spec_len >= 3\n"
    "            and draft_probs is not None\n"
    "            and not synthetic_mode\n"
    "        )\n"
    "        if _genesis_p71_eligible:\n"
    "            from sndr.engines.vllm.kernels_legacy.block_verify_sampler import (\n"
    "                call_block_verify_sample as _genesis_p71_call,\n"
    "                compute_relaxed_ok_mask as _genesis_pn369_mask_fn,\n"
    "            )\n"
    "            _genesis_p71_use_pt = _genesis_p71_os.environ.get(\n"
    "                'GENESIS_P71_USE_PYTORCH', '0') == '1'\n"
    "            assert uniform_probs is not None\n"
    "            # [Genesis PN369] relaxed-acceptance mask for the tail\n"
    "            # extension; None when PN369 is disabled at runtime ->\n"
    "            # strict Sun-2024 block rule, bit-identical to v7.42.\n"
    "            _genesis_pn369_relaxed_ok = _genesis_pn369_mask_fn(\n"
    "                target_probs, draft_token_ids\n"
    "            )\n"
    "            _genesis_p71_call(\n"
    "                output_token_ids=output_token_ids,\n"
    "                cu_num_draft_tokens=cu_num_draft_tokens,\n"
    "                draft_token_ids=draft_token_ids,\n"
    "                draft_probs=draft_probs,\n"
    "                target_probs=target_probs,\n"
    "                bonus_token_ids=bonus_token_ids,\n"
    "                uniform_probs=uniform_probs,\n"
    "                is_greedy=is_greedy,\n"
    "                num_draft_tokens=num_draft_tokens,\n"
    "                generators=sampling_metadata.generators,\n"
    "                max_spec_len=max_spec_len,\n"
    "                vocab_size=vocab_size,\n"
    "                use_pytorch=_genesis_p71_use_pt,\n"
    "                relaxed_ok=_genesis_pn369_relaxed_ok,\n"
    "            )\n"
    "            return output_token_ids\n"
    "    except Exception as _genesis_p71_err:\n"
    "        import logging as _genesis_p71_log_mod\n"
    "        _genesis_p71_log = _genesis_p71_log_mod.getLogger('genesis.kernels.p71')\n"
    "        _genesis_p71_log.warning(\n"
    "            '[Genesis P71] block-verify failed (%s: %s); falling back to '\n"
    "            'upstream per-token path. Run is unaffected.',\n"
    "            type(_genesis_p71_err).__name__, _genesis_p71_err,\n"
    "        )\n"
    "\n"
    "    # Sample recovered tokens for each position.\n"
    "    # [num_tokens]\n"
    "    recovered_token_ids = sample_recovered_tokens(\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/sample/rejection_sampler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P71 v1/sample/rejection_sampler.py — block-verify branch (vllm#40819)",
        target_file=str(target),
        marker=GENESIS_P71_MARKER,
        sub_patches=[
            TextPatch(
                name="p71_block_verify_branch",
                anchor=P71_OLD,
                replacement=P71_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P71",
            # Self-collision lint (triage plan §6 2026-06-11): former entry
            # "_genesis_p71_call" was a substring of our own replacement —
            # the exact PN369/P71 false-skip incident class (a sibling
            # baking the string masked P71 as "upstream_merged"). Residue
            # coverage is preserved by the "[Genesis P71" banner above.
            "verify_method",  # if upstream ports their own block-verify
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P71 — block-verify rejection sampling."""
    from sndr.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P71")
    log_decision("P71", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/sample/rejection_sampler.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P71] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P71" and m in content:
            continue  # our marker; handled above
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix or independent block-verify",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return "applied", (
        "P71 applied: block-verify rejection sampling branch installed (v7.43). "
        "Activates when GENESIS_ENABLE_P71_BLOCK_VERIFY=1 + max_spec_len>=3 + "
        "draft_probs available + not synthetic_mode. "
        "Bug-fixes from gemini review: shared u per request, denom==0 → 1.0. "
        "PN369 relaxed tail extension threaded (fires only when "
        "GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE=1; strict block rule otherwise)."
    )
