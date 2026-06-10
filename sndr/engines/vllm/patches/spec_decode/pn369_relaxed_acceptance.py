# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN369 — relaxed acceptance for MTP spec-decode.

TRT-LLM-style relaxed acceptance (adapted): accept a draft token that the
strict Leviathan-2022 ratio test would reject IF it lies within the
target's top-K candidates AND its target probability is within ``delta``
of the top-1 probability:

    relaxed_ok[i] = (draft_token[i] in topK(target_probs[i]))
                    and (target_probs[i][draft_token] >= top1 - delta)

    accepted = strict_rule or relaxed_ok[i]          (random path only)

The mask is computed torch-side (vectorized, one topk over the
post-temperature / post-top-k/p target_probs) and shared by BOTH accept
paths:

  1. The upstream per-token Triton kernel
     (``rejection_random_sample_kernel`` in
     ``vllm/v1/sample/rejection_sampler.py``) — patched by THIS module:
     the mask is OR-composed into the accept decision.
  2. The Genesis P71 block-verify path
     (``sndr/engines/vllm/kernels_legacy/block_verify_sampler.py``) —
     after the Sun-2024 block rule fixes ``accepted_len``, a TAIL
     EXTENSION walks forward while ``relaxed_ok`` holds; the recovered
     token is written at the NEW first-rejected position and the bonus
     token fires when the extension reaches full draft length. Threaded
     via the P71 wiring patch (``p71_block_verify.py``).

================================================================
TRADE-OFF — READ THIS BEFORE ENABLING
================================================================

The relaxed rule is **biased** — it deliberately breaks the
distribution-exactness guarantee of canonical rejection sampling (same
trade class as the live P82 threshold patch). The win is on FLAT target
distributions (creative prose; PROD measured ~0.72 accept rate / 191 TPS
vs math content ~0.79 / 253 TPS) where the draft token is plausible but
the ratio test fails. WE DO NOT SHIP THIS WITHOUT EMPIRICAL VALIDATION
(quality harness + canonical bench A/B over the topk x delta grid).

NOTE on defaults: TRT-LLM ships relaxed_topk=10 / relaxed_delta=0.6 —
those values DO NOT transfer here. TRT applies the window to raw
logits-derived probs; our target_probs are post-temperature and
post-top-k/p (``apply_sampling_constraints``), so the distribution is
already truncated and renormalized. Conservative defaults: topk=4,
delta=0.2.

================================================================
THE THREE-OR-CLAUSE ACCEPT STACK (per-token random path)
================================================================

With P82 and PN369 both applied, the per-token accept decision is:

    accepted = strict_ratio_test            (Leviathan 2022, unbiased)
            or target_prob >= P82_threshold (P82, live at 0.3 on 35B)
            or relaxed_ok[pos]              (PN369 top-K + delta window)

Each clause only ADDS accepts — the stack is a strict superset of the
strict rule, and PN369 composes with P82 as a strict superset of accepts.
On the P71 block-verify path the stack is instead:

    accepted_len = block_rule(Sun 2024)     (unbiased, >= per-token)
    accepted_len += tail_extension(relaxed_ok)   (PN369, biased)

GREEDY (temp=0) STAYS STRICT: the greedy kernel keeps exact-argmax
acceptance — tool-call / agentic flows at temp=0 are structurally
unaffected (v1 safety property). Synthetic mode keeps its own rule.

================================================================
DESIGN
================================================================

- Three text sub-patches on ``vllm/v1/sample/rejection_sampler.py``:
  1. ``pn369_kernel_signature`` — adds ``genesis_pn369_relaxed_ok_ptr``
     (defaulted to None) + ``GENESIS_PN369_RELAXED: tl.constexpr = False``
     to the random-sampling kernel. Defaults keep every other launch
     site (none today) source-compatible.
  2. ``pn369_kernel_or_compose`` — OR-injection immediately before the
     ``if accepted:`` site. Anchors AFTER the strict-accept line so it
     composes with P82 (which replaces the strict-accept region) in
     either apply order.
  3. ``pn369_launch_site_mask`` — computes the mask torch-side via the
     shared helper in ``block_verify_sampler.py`` and threads it into
     the kernel launch. The helper returns None when the runtime env
     flag is off -> constexpr False -> kernel bit-identical to vanilla.
- Env values (topk / delta) are read at RUNTIME inside the shared helper
  (cached after first read — no per-step parsing in the hot path), NOT
  baked into the text. The marker therefore does not need to encode
  them; changing topk/delta requires only a container restart (no fs
  reset), unlike P82's baked threshold.

================================================================
SAFETY MODEL
================================================================

- Env GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE unset/0 -> apply() SKIPS,
  source stays vanilla.
- Text applied but runtime flag off (stale container fs after operator
  disables) -> helper returns None -> GENESIS_PN369_RELAXED constexpr
  False -> dead-code-eliminated, bit-identical to vanilla.
- Mask computation wrapped in try/except in the injected text: any
  failure logs once per step and falls back to strict acceptance.
- Anchor missing (upstream rewrote the region) -> SKIPPED with reason.
- Drift markers catch upstream's own relaxed-acceptance landing (all 4
  upstream relaxed-acceptance PRs are closed unmerged as of 2026-06-10)
  and the #41258 lazy-recovery rewrite of the anchor region.

Status: opt-in via ``GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE=1``.
Default OFF.

Tunable knobs
-------------
- ``GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE`` (default unset/0): master
  switch (read at apply time AND at runtime inside the shared helper)
- ``GENESIS_PN369_RELAXED_TOPK`` (default 4, clamped to [1, 32]): size
  of the target top-K window the draft token must fall into
- ``GENESIS_PN369_RELAXED_DELTA`` (default 0.2, clamped to [0.0, 1.0]):
  max allowed gap to the top-1 target probability
  - 0.0 -> only (near-)argmax draft tokens pass the window (near-strict)
  - 0.2 -> conservative default for post-processing probs
  - >=0.5 -> aggressive; expect measurable quality drift

Compatibility
-------------
- P82: OR-composes (strict superset of accepts); disjoint anchors,
  apply-order independent.
- P71 block-verify: composes via tail extension (threaded by the P71
  wiring — see p71_block_verify.py marker v7.43).
- ngram (NO_DRAFT_PROBS): mask depends only on target_probs, so the
  relaxed clause works for ngram drafts too.
- Cudagraph: unaffected (rejection sampler runs OUTSIDE captured graphs).
- Greedy / synthetic paths: untouched.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference algorithm: TensorRT-LLM relaxed acceptance (NVIDIA), adapted
for post-processing target_probs.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatcher,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.pn369_relaxed_acceptance")

GENESIS_PN369_MARKER = (
    "Genesis PN369 relaxed acceptance (TRT-style top-K + delta window) v12.0"
)


# ─── Runtime env parsing (torch-free; shared with block_verify_sampler) ────
#
# block_verify_sampler.py lazily imports these three readers and caches the
# resulting (enabled, topk, delta) tuple at first hot-path use, so the env
# is parsed once per process, not once per decode step.

PN369_DEFAULT_TOPK = 4
PN369_DEFAULT_DELTA = 0.2
PN369_TOPK_MIN = 1
PN369_TOPK_MAX = 32


def is_pn369_runtime_enabled() -> bool:
    """Truthiness of GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE."""
    return os.environ.get(
        "GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def read_relaxed_topk() -> int:
    """GENESIS_PN369_RELAXED_TOPK with default 4, clamped to [1, 32]."""
    raw = os.environ.get("GENESIS_PN369_RELAXED_TOPK", "").strip()
    if not raw:
        return PN369_DEFAULT_TOPK
    try:
        v = int(raw)
    except ValueError:
        log.warning(
            "[PN369] GENESIS_PN369_RELAXED_TOPK=%r not parseable as int; "
            "using default %d", raw, PN369_DEFAULT_TOPK,
        )
        return PN369_DEFAULT_TOPK
    if not (PN369_TOPK_MIN <= v <= PN369_TOPK_MAX):
        log.warning(
            "[PN369] relaxed_topk %d out of [%d, %d]; clamping",
            v, PN369_TOPK_MIN, PN369_TOPK_MAX,
        )
        v = max(PN369_TOPK_MIN, min(PN369_TOPK_MAX, v))
    return v


def read_relaxed_delta() -> float:
    """GENESIS_PN369_RELAXED_DELTA with default 0.2, clamped to [0.0, 1.0]."""
    raw = os.environ.get("GENESIS_PN369_RELAXED_DELTA", "").strip()
    if not raw:
        return PN369_DEFAULT_DELTA
    try:
        v = float(raw)
    except ValueError:
        log.warning(
            "[PN369] GENESIS_PN369_RELAXED_DELTA=%r not parseable as float; "
            "using default %.2f", raw, PN369_DEFAULT_DELTA,
        )
        return PN369_DEFAULT_DELTA
    if not (0.0 <= v <= 1.0):
        log.warning("[PN369] relaxed_delta %.4f out of [0.0, 1.0]; clamping", v)
        v = max(0.0, min(1.0, v))
    return v


# ─── Sub-patch 1: random-kernel signature (defaulted trailing params) ──────
# Anchor on the tail of rejection_random_sample_kernel's signature. The
# leading `uniform_probs_ptr,  # [num_tokens]` line disambiguates from the
# greedy kernel (whose comment differs) and from the recovered-tokens
# kernel (different param set).

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


# ─── Sub-patch 2: OR-compose immediately before the `if accepted:` site ────
# Deliberately anchors AFTER the strict-accept assignment so it does NOT
# overlap P82's replaced region — both patches apply in either order.

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


# ─── Sub-patch 3: launch-site mask computation + threading ─────────────────
# Sits downstream of the P71 injected branch (which early-returns when it
# handles the step), so the mask is computed at most once per step per path.

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


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/sample/rejection_sampler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN369 v1/sample/rejection_sampler.py — relaxed acceptance "
            "(top-K + delta window, runtime-tuned)"
        ),
        target_file=str(target),
        marker=GENESIS_PN369_MARKER,
        sub_patches=[
            TextPatch(
                name="pn369_kernel_signature",
                anchor=PN369_SIG_OLD,
                replacement=PN369_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="pn369_kernel_or_compose",
                anchor=PN369_BODY_OLD,
                replacement=PN369_BODY_NEW,
                required=True,
            ),
            TextPatch(
                name="pn369_launch_site_mask",
                anchor=PN369_LAUNCH_OLD,
                replacement=PN369_LAUNCH_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN369",
            "genesis_pn369_relaxed_ok",
            # Upstream's own relaxed acceptance landing (all 4 upstream
            # relaxed-acceptance PRs are closed unmerged as of 2026-06-10;
            # these markers fire if any of them is resurrected and merged):
            "relaxed_topk",
            "use_relaxed_acceptance",
            "relax_ratio",
            # PR #41258 "Lazy recovery evaluation for spec rejection
            # sampling" rewrites the anchor region (recovered_token_ids
            # threading disappears from the kernel). Match ONLY the new
            # lazy markers — NOT sample_recovered_tokens_kernel (existing
            # base name; the P82 dev93 false-positive lesson applies).
            "_lazy_recovered_token",
            "lazy_recovery",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN369 — relaxed acceptance for MTP spec-decode."""
    from sndr.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN369")
    log_decision("PN369", decision, reason)
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
        log.info("[PN369] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m in ("[Genesis PN369", "genesis_pn369_relaxed_ok") and m in content:
            continue  # our own text; handled by the marker check above
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have landed its own relaxed acceptance or the "
                "lazy-recovery rewrite of the anchor region",
            )

    result, failure = patcher.apply()
    from sndr.kernel import result_to_wiring_status
    applied_msg = (
        "PN369 applied: relaxed acceptance (top-K + delta window) installed "
        "on the per-token random path. Runtime knobs: "
        "GENESIS_PN369_RELAXED_TOPK (default 4), GENESIS_PN369_RELAXED_DELTA "
        "(default 0.2) — read once per process by the shared mask helper. "
        "Greedy / synthetic paths stay strict. BIASED rule — validate with "
        "the quality harness before promoting. P71 block-verify tail "
        "extension is threaded separately (p71_block_verify marker v7.43)."
    )
    return result_to_wiring_status(
        result, failure,
        applied_message=applied_msg,
        patch_name=patcher.patch_name,
    )
