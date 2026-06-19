# SPDX-License-Identifier: Apache-2.0
"""PN369 — relaxed acceptance for MTP spec-decode (CONSOLIDATED 2026-06-19).

PN369 (TRT-LLM-style relaxed acceptance) and P71 (block-verify rejection
sampling) both text-patch ``v1/sample/rejection_sampler.py`` at disjoint
regions. On 2026-06-19 the two wiring modules were merged into one
``TextPatcher`` (one shared marker, four sub-patches) — see
``p71_pn369_rejection_sampler_consolidated.py``. The registry collapsed the
PN369 entry into P71 (PN369's enable flag kept as an
``env_flag_aliases`` entry on the P71 entry).

THIS MODULE is now a THIN COMPATIBILITY SHIM. It keeps two responsibilities:

  1. The RUNTIME env readers (``is_pn369_runtime_enabled`` /
     ``read_relaxed_topk`` / ``read_relaxed_delta`` + their constants) —
     these are imported by
     ``sndr/engines/vllm/kernels_legacy/block_verify_sampler.py`` (the shared
     relaxed-mask helper) and are runtime concerns, NOT text-patch wiring, so
     they live here unchanged.

  2. Re-exports of the PN369 anchor / replacement / marker constants and a
     delegating ``apply()`` — so existing imports, unit tests, and drift-
     residue greps keep resolving. The ``apply()`` delegates to the
     consolidated module (idempotent shared marker), exactly as the legacy
     boot loop's PN369 ``@register_patch`` hook now does.

================================================================
THE RELAXED RULE (unchanged behavior reference)
================================================================

TRT-LLM-style relaxed acceptance (adapted): accept a draft token the strict
Leviathan-2022 ratio test rejects IF it lies within the target's top-K
candidates AND its target probability is within ``delta`` of the top-1:

    relaxed_ok[i] = (draft_token[i] in topK(target_probs[i]))
                    and (target_probs[i][draft_token] >= top1 - delta)

BIASED rule — deliberately breaks distribution-exactness for speed on flat
distributions (same trade class as P82). Default OFF. Greedy (temp=0) and
synthetic paths stay strict. ngram (NO_DRAFT_PROBS) supported (mask depends
only on target_probs). Validate with the quality harness before promoting.

Tunable knobs
-------------
- ``GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE`` (default unset/0): master
  switch (read at apply time AND at runtime inside the shared helper)
- ``GENESIS_PN369_RELAXED_TOPK`` (default 4, clamped to [1, 32])
- ``GENESIS_PN369_RELAXED_DELTA`` (default 0.2, clamped to [0.0, 1.0])

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference algorithm: TensorRT-LLM relaxed acceptance (NVIDIA), adapted for
post-processing target_probs.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root

# Re-export the PN369 anchor / replacement / marker constants from the
# consolidated module so existing imports + drift-residue coverage keep
# resolving against a single source of truth.
from sndr.engines.vllm.patches.spec_decode.p71_pn369_rejection_sampler_consolidated import (  # noqa: E501
    GENESIS_PN369_MARKER,
    PN369_BODY_NEW,
    PN369_BODY_OLD,
    PN369_LAUNCH_NEW,
    PN369_LAUNCH_OLD,
    PN369_SIG_NEW,
    PN369_SIG_OLD,
)
from sndr.engines.vllm.patches.spec_decode.p71_pn369_rejection_sampler_consolidated import (  # noqa: E501
    _pn369_body_sub_patch,
    _pn369_launch_sub_patch,
    _pn369_sig_sub_patch,
    apply as _consolidated_apply,
)
from sndr.kernel import TextPatcher

log = logging.getLogger("genesis.wiring.pn369_relaxed_acceptance")


# Re-exported for fixture-building tests that reach through the module
# namespace (``module.resolve_vllm_file`` / ``module.vllm_install_root``).
__all__ = [
    "GENESIS_PN369_MARKER",
    "PN369_BODY_NEW", "PN369_BODY_OLD",
    "PN369_LAUNCH_NEW", "PN369_LAUNCH_OLD",
    "PN369_SIG_NEW", "PN369_SIG_OLD",
    "PN369_DEFAULT_TOPK", "PN369_DEFAULT_DELTA",
    "PN369_TOPK_MIN", "PN369_TOPK_MAX",
    "is_pn369_runtime_enabled", "read_relaxed_topk", "read_relaxed_delta",
    "resolve_vllm_file", "vllm_install_root",
    "_make_patcher", "apply",
]


# Standalone PN369 patcher builder — RETAINED for the drift tool's fallback
# discovery and for fixture-building tests that apply PN369's three
# sub-patches on their own (e.g. test_pn381_sampler_regression). The sub-
# patches are the same verbatim objects the consolidated module uses, so the
# applied bytes are identical.
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
            _pn369_sig_sub_patch(),
            _pn369_body_sub_patch(),
            _pn369_launch_sub_patch(),
        ],
        upstream_drift_markers=[
            "relaxed_topk",
            "use_relaxed_acceptance",
            "relax_ratio",
            "_lazy_recovered_token",
            "lazy_recovery",
        ],
    )


# ─── Runtime env parsing (torch-free; shared with block_verify_sampler) ────
#
# block_verify_sampler.py lazily imports these three readers and caches the
# resulting (enabled, topk, delta) tuple at first hot-path use, so the env is
# parsed once per process, not once per decode step. They are runtime knobs,
# NOT text-patch wiring — kept here verbatim through the consolidation.

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


def apply() -> tuple[str, str]:
    """Apply PN369 — delegates to the consolidated P71+PN369 wiring module.

    The consolidated module gates each feature by its own env flag (and
    replicates PN369's version gate), and is idempotent under the shared
    marker, so a sibling P71 dispatch that already ran the consolidated
    apply() makes this re-run report IDEMPOTENT. Runtime-neutral.
    """
    return _consolidated_apply()
