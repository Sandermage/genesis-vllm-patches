# SPDX-License-Identifier: Apache-2.0
"""Consolidated wiring for P71 + PN369 — both text-patch the SAME engine file
``v1/sample/rejection_sampler.py`` at DISJOINT pristine regions.

================================================================
WHY THIS MODULE EXISTS (maintainability refactor, 2026-06-19)
================================================================

P71 (block-verify rejection sampling, Sun 2024 / vllm#40819) and PN369
(TRT-LLM-style relaxed acceptance) historically lived in two separate
wiring modules, each with its own ``TextPatcher`` and its own marker, even
though they patch the same file at non-overlapping anchors:

  - P71   → injects a block-verify branch BEFORE upstream's
            ``sample_recovered_tokens`` call (~line 471 of the pristine
            ``rejection_sample()`` body). ONE sub-patch.
  - PN369 → THREE disjoint sub-patches further down the file: the random
            kernel SIGNATURE (defaulted trailing constexpr params), the
            OR-compose body immediately before the ``if accepted:`` site,
            and the launch-site mask computation + kernel-arg threading
            (~lines 489-506+ of the pristine file).

This module collapses both into ONE ``TextPatcher`` with ONE shared marker
and FOUR sub-patches (P71's one + PN369's three). The applied OUTPUT for
the kernel-code regions is byte-identical to P71+PN369 applied separately
— anchors and replacements are copied VERBATIM from the originals (see the
verbatim blocks below; a 4-combo md5 harness over the pristine tree proves
every {P71, PN369} flag combination matches). The only intentional
difference vs the two-module layout is a single shared wiring-marker
comment line instead of two — wiring metadata, not kernel code.

================================================================
PER-FEATURE GATING (byte-equivalent with the original modules)
================================================================

The two features stay INDEPENDENTLY operator-gated, exactly as before:

  - The P71 sub-patch is applied iff P71's original ``should_apply("P71")``
    gate would have passed. P71 is a ``tier=community`` opt-in with NO
    version range, so that reduces to: env ENABLE truthy AND not explicitly
    DISABLEd. Replicated by ``_p71_enabled()`` on the bare flag
    ``P71_BLOCK_VERIFY`` (so ``SNDR_*`` aliases + the
    ``GENESIS_DISABLE_P71_BLOCK_VERIFY`` kill-switch keep working).

  - The PN369 sub-patches are applied iff PN369's original
    ``should_apply("PN369")`` gate would have passed. PN369 carried
    ``vllm_version_range=(">=0.22.0", "<0.24.0")`` (lifecycle=research,
    opt-in). The merged P71 entry carries NO version range, and the
    dispatcher's version-only gate (``_check_version_gate``) fires BEFORE
    the env-override branch in ``should_apply`` and is LIVE on the rig
    (``GENESIS_ENFORCE_VERSION_RANGE=1``). So ``_pn369_enabled()`` must
    replicate BOTH PN369's env gate AND its version gate, otherwise the
    PN369 sub-patches would (incorrectly) apply on an out-of-window pin
    where standalone PN369 would have version-skipped. See the version-
    asymmetry guard below.

================================================================

Authors:
  - P71:   Sandermage (Sander) Barzov Aleksandr — backport of
           vllm-project/vllm#40819 (Z. Golpayegani draft) + Sun et al.
           arXiv 2403.10444 + 2 gemini-code-assist review fixes.
  - PN369: Sandermage (Sander) Barzov Aleksandr, 2026-06-10 — adapted from
           TensorRT-LLM relaxed acceptance (NVIDIA).
  - Consolidation: 2026-06-19 (maintainability refactor; runtime-neutral).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p71_pn369_rejection_sampler_consolidated")


# ─── Shared idempotency marker for the merged patcher ──────────────────
# Carry P71's original marker verbatim as the shared marker so a container
# fs already holding the P71 v7.43 bake stays idempotent and operator greps
# for the P71 marker keep resolving.
GENESIS_P71_PN369_MARKER = (
    "Genesis P71 block-verify rejection sampling vllm#40819 "
    "v7.43_pn369_relaxed_tail"
)

# Original per-feature markers are RE-EXPORTED unchanged so existing tests,
# drift-residue coverage, and operator greps for the old marker strings keep
# resolving against this consolidated module.
GENESIS_P71_MARKER = (
    "Genesis P71 block-verify rejection sampling vllm#40819 "
    "v7.43_pn369_relaxed_tail"
)
GENESIS_PN369_MARKER = (
    "Genesis PN369 relaxed acceptance (TRT-style top-K + delta window) v12.0"
)


# ─── PN369's version window (carried here because the merged P71 entry has
# ─── no vllm_version_range — see module docstring + _pn369_enabled).
PN369_VLLM_VERSION_RANGE = (">=0.22.0", "<0.24.0")


# ════════════════════════════════════════════════════════════════════════
# Sub-patch 1: P71 block-verify branch (VERBATIM from p71_block_verify.py)
# ════════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════════
# Sub-patch 2: PN369 kernel signature (VERBATIM from pn369_relaxed_acceptance)
# ════════════════════════════════════════════════════════════════════════
PN369_SIG_OLD = (
    "    uniform_probs_ptr,  # [num_tokens]\n"
    "    is_greedy_ptr,  # [batch_size]\n"
    "    max_spec_len,\n"
    "    vocab_size,\n"
    "    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None\n"
    "    NO_DRAFT_PROBS: tl.constexpr,\n"
    "    SYNTHETIC_MODE: tl.constexpr,\n"
    "):\n"
)

PN369_SIG_NEW = (
    "    uniform_probs_ptr,  # [num_tokens]\n"
    "    is_greedy_ptr,  # [batch_size]\n"
    "    max_spec_len,\n"
    "    vocab_size,\n"
    "    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None\n"
    "    NO_DRAFT_PROBS: tl.constexpr,\n"
    "    SYNTHETIC_MODE: tl.constexpr,\n"
    "    # [Genesis PN369] relaxed-acceptance inputs. Defaults keep any\n"
    "    # other launch site source-compatible (constexpr False prunes the\n"
    "    # relaxed branch entirely -> bit-identical to vanilla).\n"
    "    genesis_pn369_relaxed_ok_ptr=None,  # [num_tokens] int32 or None\n"
    "    GENESIS_PN369_RELAXED: tl.constexpr = False,\n"
    "):\n"
)


# ════════════════════════════════════════════════════════════════════════
# Sub-patch 3: PN369 OR-compose body (VERBATIM from pn369_relaxed_acceptance)
# ════════════════════════════════════════════════════════════════════════
PN369_BODY_OLD = (
    "            if accepted:\n"
    "                token_id = draft_token_id\n"
    "            else:\n"
    "                rejected = True\n"
    "                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)\n"
)

PN369_BODY_NEW = (
    "            # ════════════════════════════════════════════════════════════\n"
    "            # [Genesis PN369] Relaxed acceptance OR-compose: a strictly\n"
    "            # rejected draft token is accepted anyway when it sits inside\n"
    "            # the target's top-K AND within delta of the top-1 probability\n"
    "            # (mask precomputed torch-side at the launch site). Greedy\n"
    "            # requests never reach this body (early return above);\n"
    "            # synthetic mode keeps its own acceptance rule untouched.\n"
    "            # BIASED rule — see PN369 wiring docstring for the trade-off.\n"
    "            # ════════════════════════════════════════════════════════════\n"
    "            if GENESIS_PN369_RELAXED:\n"
    "                if not SYNTHETIC_MODE:\n"
    "                    if not accepted:\n"
    "                        accepted = (\n"
    "                            tl.load(\n"
    "                                genesis_pn369_relaxed_ok_ptr + start_idx + pos\n"
    "                            )\n"
    "                            != 0\n"
    "                        )\n"
    "            if accepted:\n"
    "                token_id = draft_token_id\n"
    "            else:\n"
    "                rejected = True\n"
    "                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)\n"
)


# ════════════════════════════════════════════════════════════════════════
# Sub-patch 4: PN369 launch-site mask (VERBATIM from pn369_relaxed_acceptance)
# ════════════════════════════════════════════════════════════════════════
PN369_LAUNCH_OLD = (
    "    # Rejection sampling for random sampling requests.\n"
    "    assert uniform_probs is not None\n"
    "    rejection_random_sample_kernel[(batch_size,)](\n"
    "        output_token_ids,\n"
    "        cu_num_draft_tokens,\n"
    "        draft_token_ids,\n"
    "        draft_probs,\n"
    "        target_probs,\n"
    "        bonus_token_ids,\n"
    "        recovered_token_ids,\n"
    "        uniform_probs,\n"
    "        is_greedy,\n"
    "        max_spec_len,\n"
    "        vocab_size,\n"
    "        synthetic_conditional_rates,\n"
    "        NO_DRAFT_PROBS=draft_probs is None,\n"
    "        SYNTHETIC_MODE=synthetic_mode,\n"
    "    )\n"
)

PN369_LAUNCH_NEW = (
    "    # ════════════════════════════════════════════════════════════════\n"
    "    # [Genesis PN369] Relaxed acceptance (TRT-LLM-style, adapted).\n"
    "    # Compute the relaxed_ok mask torch-side from the post-processing\n"
    "    # target_probs. The shared helper returns None when the runtime\n"
    "    # env flag is off -> constexpr-pruned kernel, bit-identical to\n"
    "    # vanilla. Synthetic mode keeps its own rule (mask not computed).\n"
    "    # ════════════════════════════════════════════════════════════════\n"
    "    _genesis_pn369_relaxed_ok = None\n"
    "    if not synthetic_mode:\n"
    "        try:\n"
    "            from sndr.engines.vllm.kernels_legacy.block_verify_sampler import (\n"
    "                compute_relaxed_ok_mask as _genesis_pn369_mask_fn,\n"
    "            )\n"
    "            _genesis_pn369_relaxed_ok = _genesis_pn369_mask_fn(\n"
    "                target_probs, draft_token_ids\n"
    "            )\n"
    "        except Exception as _genesis_pn369_err:\n"
    "            import logging as _genesis_pn369_log_mod\n"
    "            _genesis_pn369_log_mod.getLogger('genesis.kernels.pn369').warning(\n"
    "                '[Genesis PN369] relaxed mask computation failed (%s: %s); '\n"
    "                'falling back to strict acceptance for this step.',\n"
    "                type(_genesis_pn369_err).__name__, _genesis_pn369_err,\n"
    "            )\n"
    "            _genesis_pn369_relaxed_ok = None\n"
    "\n"
    "    # Rejection sampling for random sampling requests.\n"
    "    assert uniform_probs is not None\n"
    "    rejection_random_sample_kernel[(batch_size,)](\n"
    "        output_token_ids,\n"
    "        cu_num_draft_tokens,\n"
    "        draft_token_ids,\n"
    "        draft_probs,\n"
    "        target_probs,\n"
    "        bonus_token_ids,\n"
    "        recovered_token_ids,\n"
    "        uniform_probs,\n"
    "        is_greedy,\n"
    "        max_spec_len,\n"
    "        vocab_size,\n"
    "        synthetic_conditional_rates,\n"
    "        NO_DRAFT_PROBS=draft_probs is None,\n"
    "        SYNTHETIC_MODE=synthetic_mode,\n"
    "        genesis_pn369_relaxed_ok_ptr=_genesis_pn369_relaxed_ok,\n"
    "        GENESIS_PN369_RELAXED=_genesis_pn369_relaxed_ok is not None,\n"
    "    )\n"
)


# ─── Bare env-flag names (no GENESIS_ENABLE_/SNDR_ENABLE_ prefix) ──────────
_P71_FLAG = "P71_BLOCK_VERIFY"
_PN369_FLAG = "PN369_RELAXED_ACCEPTANCE"


def _p71_sub_patch() -> TextPatch:
    return TextPatch(
        name="p71_block_verify_branch",
        anchor=P71_OLD,
        replacement=P71_NEW,
        required=True,
    )


def _pn369_sig_sub_patch() -> TextPatch:
    return TextPatch(
        name="pn369_kernel_signature",
        anchor=PN369_SIG_OLD,
        replacement=PN369_SIG_NEW,
        required=True,
    )


def _pn369_body_sub_patch() -> TextPatch:
    return TextPatch(
        name="pn369_kernel_or_compose",
        anchor=PN369_BODY_OLD,
        replacement=PN369_BODY_NEW,
        required=True,
    )


def _pn369_launch_sub_patch() -> TextPatch:
    return TextPatch(
        name="pn369_launch_site_mask",
        anchor=PN369_LAUNCH_OLD,
        replacement=PN369_LAUNCH_NEW,
        required=True,
    )


# PN369's three sub-patches always travel together (signature + body + launch
# are a single feature). P71's branch is inserted BEFORE them so the merged
# apply order matches the original dispatcher order (P71 ran first).
_PN369_DRIFT_MARKERS = [
    # Upstream's own relaxed acceptance landing (all 4 upstream relaxed-
    # acceptance PRs are closed unmerged as of 2026-06-10):
    "relaxed_topk",
    "use_relaxed_acceptance",
    "relax_ratio",
    # PR #41258 lazy-recovery rewrite of the anchor region:
    "_lazy_recovered_token",
    "lazy_recovery",
]


def _make_patcher() -> TextPatcher | None:
    """Drift-tool / static entry point: ONE TextPatcher carrying ALL FOUR
    sub-patches UNCONDITIONALLY.

    ``tools/check_upstream_drift.py`` builds the patcher from this function
    and verifies every sub-patch anchor is present-and-unique in the
    pristine tree. All four anchors MUST be declared here regardless of
    runtime env gating so the static drift check covers both features.
    """
    target = resolve_vllm_file("v1/sample/rejection_sampler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P71+PN369 v1/sample/rejection_sampler.py — block-verify branch "
            "(vllm#40819) + relaxed acceptance (top-K + delta window)"
        ),
        target_file=str(target),
        marker=GENESIS_P71_PN369_MARKER,
        sub_patches=[
            _p71_sub_patch(),
            _pn369_sig_sub_patch(),
            _pn369_body_sub_patch(),
            _pn369_launch_sub_patch(),
        ],
        upstream_drift_markers=[
            "[Genesis P71",
            # Self-collision lint: former entry "_genesis_p71_call" was a
            # substring of our own replacement. Residue coverage preserved by
            # the "[Genesis P71" banner above.
            "verify_method",  # if upstream ports their own block-verify
            *_PN369_DRIFT_MARKERS,
        ],
    )


def _p71_enabled() -> bool:
    """Replicates P71's original ``should_apply("P71")`` gate: a
    ``tier=community`` opt-in with NO version range reduces to env ENABLE
    truthy AND not explicitly DISABLEd."""
    from sndr.env import is_disabled, is_enabled

    return is_enabled(_P71_FLAG) and not is_disabled(_P71_FLAG)


def _pn369_enabled() -> bool:
    """Replicates PN369's original ``should_apply("PN369")`` gate.

    PN369 was opt-in (env ENABLE truthy AND not DISABLEd) AND carried
    ``vllm_version_range=(">=0.22.0", "<0.24.0")``. The merged P71 entry has
    NO version range, and the dispatcher's version-only gate fires BEFORE the
    env-override branch and is LIVE on the rig
    (``GENESIS_ENFORCE_VERSION_RANGE=1``), so this helper must replicate the
    version gate too — otherwise the PN369 sub-patches would apply on an
    out-of-window pin where standalone PN369 would have version-skipped.
    """
    from sndr.env import is_disabled, is_enabled

    if not (is_enabled(_PN369_FLAG) and not is_disabled(_PN369_FLAG)):
        return False
    from sndr.dispatcher.decision import _version_enforcement_on

    if _version_enforcement_on():
        from sndr.compat.version_check import check_version_constraints

        v_ok, _ = check_version_constraints(
            {"vllm_version_range": PN369_VLLM_VERSION_RANGE}
        )
        if not v_ok:
            return False
    return True


def apply() -> tuple[str, str]:
    """Apply P71 + PN369 (consolidated) — each feature independently
    operator-gated so the applied set is byte-identical to running the two
    original modules separately."""
    p71_on = _p71_enabled()
    pn369_on = _pn369_enabled()

    if not p71_on and not pn369_on:
        return "skipped", (
            "P71+PN369 both default OFF — set "
            "GENESIS_ENABLE_P71_BLOCK_VERIFY=1 (block-verify rejection "
            "sampling) and/or GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE=1 "
            "(relaxed acceptance, in-window pins only) to engage. Each flag "
            "independently gates its own sub-patches."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target = resolve_vllm_file("v1/sample/rejection_sampler.py")
    if target is None:
        return "skipped", "vllm/v1/sample/rejection_sampler.py not found"

    if not os.path.isfile(str(target)):
        return "skipped", f"target disappeared: {target}"
    with open(str(target)) as f:
        content = f.read()
    if GENESIS_P71_PN369_MARKER in content:
        log.info("[P71+PN369] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    # Patcher-level upstream-drift markers — only those NOT baked by our own
    # replacements (idempotency is the marker's job, never a drift marker).
    drift_markers: list[str] = []
    if p71_on:
        drift_markers.append("verify_method")
    if pn369_on:
        drift_markers.extend(_PN369_DRIFT_MARKERS)
    for m in drift_markers:
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {target} — upstream may "
                "have absorbed this fix or landed its own implementation",
            )

    sub_patches: list[TextPatch] = []
    if p71_on:
        sub_patches.append(_p71_sub_patch())
    if pn369_on:
        sub_patches.append(_pn369_sig_sub_patch())
        sub_patches.append(_pn369_body_sub_patch())
        sub_patches.append(_pn369_launch_sub_patch())

    patcher = TextPatcher(
        patch_name=(
            "P71+PN369 v1/sample/rejection_sampler.py — block-verify branch "
            "+ relaxed acceptance"
        ),
        target_file=str(target),
        marker=GENESIS_P71_PN369_MARKER,
        sub_patches=sub_patches,
        upstream_drift_markers=["[Genesis P71"],
    )

    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "idempotent (marker present)"

    applied = patcher.applied_sub_patches or [sp.name for sp in patcher.sub_patches]
    enabled = []
    if p71_on:
        enabled.append("P71 block-verify")
    if pn369_on:
        enabled.append("PN369 relaxed acceptance")
    return "applied", (
        f"P71+PN369 consolidated installed ({', '.join(enabled)}). "
        f"P71 activates when GENESIS_ENABLE_P71_BLOCK_VERIFY=1 + "
        f"max_spec_len>=3 + draft_probs available + not synthetic_mode. "
        f"PN369 relaxed acceptance (topk/delta runtime-tuned) fires only "
        f"when GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE=1; greedy/synthetic "
        f"stay strict. Sub-patches applied: {', '.join(applied)}."
    )


def is_applied() -> bool:
    """Best-effort idempotency probe — True iff the shared marker is present
    in the target file."""
    target = resolve_vllm_file("v1/sample/rejection_sampler.py")
    if target is None:
        return False
    try:
        with open(str(target), "r", encoding="utf-8", errors="ignore") as fh:
            return GENESIS_P71_PN369_MARKER in fh.read()
    except OSError:
        return False
