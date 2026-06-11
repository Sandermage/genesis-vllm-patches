# SPDX-License-Identifier: Apache-2.0
"""G4_03 — refuse non-causal drafter (Eagle3/DFlash) with Gemma 4 on Ampere.

================================================================
WHAT BREAKS WITHOUT THIS PATCH
================================================================

Gemma 4 has interleaved attention with two different head dimensions:
``head_dim = 256`` for sliding-window layers and ``global_head_dim = 512``
for full-attention layers. The EAGLE-3 and DFlash drafter families
both use **non-causal block-parallel attention** to draft K tokens
in a single pass.

On Ampere consumer GPUs (RTX 3090 / A5000, SM 8.6) every available
attention backend rejects one or the other of those constraints
(vllm#40382):

  | Backend           | head_dim=256 | non-causal       | Verdict     |
  |---|---|---|---|
  | FA2               | sometimes    | causal-only      | unusable    |
  | FA2_DIFFKV        | unsupported  | causal-only      | unusable    |
  | FLASHINFER        | supported    | causal-only      | unusable    |
  | TRITON_ATTN       | supported    | causal-only      | unusable    |
  | FLEX_ATTENTION    | supported    | supported (slow) | net loss    |
  | TREE_ATTN         | unsupported  | causal-only      | unusable    |

FLEX_ATTENTION is technically functional but its overhead exceeds the
draft-acceptance benefit, so EAGLE-3 / DFlash on Ampere produces a
**negative** TPS result vs no spec-decode.

================================================================
THE FIX (this patch — short term)
================================================================

Wrap ``SpecDecodeBaseProposer._create_draft_vllm_config`` (pristine
``vllm/v1/spec_decode/llm_base_proposer.py:1157``) so it refuses at
config build time with a clear error pointing at:

  * upstream bug #40382 (backend matrix)
  * the recommended Ampere drafter (native Gemma 4 MTP assistant —
    ``Gemma4Proposer``, see SPEC-DECODE LAYOUT below)
  * the deep fix (G4_10 — Genesis non-causal head_dim=256 Triton kernel)

The wrapper gates on ``speculative_config.method in ("eagle3", "dflash")``
(via ``detect_non_causal_drafter``), so causal proposers — MTP,
EAGLE-1, draft_model, ngram — sharing the same base class pass through
untouched.

================================================================
SPEC-DECODE LAYOUT IN THE CURRENT PIN (verified 2026-06-11)
================================================================

The candidate pin ``0.22.1rc1.dev259+g303916e93`` (image
``vllm/vllm-openai:nightly-303916e93``, see candidate-tree
PROVENANCE.json) removed the dedicated ``vllm/v1/spec_decode/eagle3.py``
module this guard originally bound to. Verified against the pristine
tree:

  * ``llm_base_proposer.py:59``   — ``class SpecDecodeBaseProposer:``
    with ``self.method = self.speculative_config.method`` (line 71)
    and ``def _create_draft_vllm_config(self) -> VllmConfig:`` (1157).
  * ``eagle.py:10``  — ``class EagleProposer(SpecDecodeBaseProposer):``
    serves methods "eagle"/"eagle3" (and generic "mtp"); it does NOT
    override ``_create_draft_vllm_config`` — the base wrap covers it.
  * ``dflash.py:21`` — ``class DFlashProposer(SpecDecodeBaseProposer):``
    overrides ``_create_draft_vllm_config`` (line 74) but delegates to
    ``super()._create_draft_vllm_config()`` (line 75) — the base wrap
    covers it too.
  * ``gemma4.py:31`` — ``class Gemma4Proposer(SpecDecodeBaseProposer):``
    native Gemma 4 MTP assistant proposer; selected by
    ``SpeculativeConfig.use_gemma4_mtp()`` when ``method == "mtp"`` and
    the draft model_type is ``gemma4_mtp`` (normalized from
    ``gemma4_assistant`` / ``gemma4_unified_assistant``,
    ``config/speculative.py:512-513``). It shares KV cache with the
    target via cross-model KV sharing — causal, Ampere-safe. This is
    the path the refusal message recommends.

Wrapping the base class is rename-proof against future drafter-module
shuffles; the per-class probes on ``eagle.EagleProposer`` /
``dflash.DFlashProposer`` are kept only as fallback.

================================================================
FAIL-LOUD CONTRACT (2026-06-11, preflight triage §6)
================================================================

The original code swallowed drafter-module ImportError at log.debug,
so an ENABLED guard could report "applied" while Eagle3 ran unguarded
(silent hazard caught by the 2026-06-11 preflight residual triage).
New contract when the guard is enabled:

  * primary base-class wrap succeeds            → "applied"
  * base missing, BOTH fallback classes wrapped → "applied" (noted)
  * base missing, ONE fallback class wrapped    → "partial" + warning
  * nothing wrapped                             → "failed"  + warning

"partial"/"failed" propagate through the dispatcher as a failed patch
— never a silent success.

================================================================
DEEP FIX
================================================================

Reach out to G4_10 (``g4_10_gemma4_ampere_non_causal_attn_backend.py``)
which **registers a Triton attention backend** specialized for
head_dim=256 non-causal on Ampere SM 8.6. The kernel implementation
lives at ``kernels/g4_non_causal_attn_triton.py``. Once it ships and
validates, this guard becomes redundant — set
``superseded_by: [G4_10]`` and flip ``default_on=False``.

================================================================
SAFETY MODEL
================================================================

* default_on: True (informational — strict opt-in via env_flag since
  2026-05-17 dispatcher semantics)
* env_flag: GENESIS_ENABLE_G4_03_GEMMA4_NON_CAUSAL_DRAFTER_GUARD
* override: GENESIS_DISABLE_G4_03_GUARD=1 (for G4_10 testing)
* applies_to:
    - architecture: gemma4 (target)
    - hardware: Ampere SM 8.6
    - spec_decode.method in {eagle3, dflash}
* superseded_by: [G4_10] (when Ampere non-causal kernel ships)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/issues/40382 (backend matrix)
  * https://github.com/vllm-project/vllm/pull/41745 (Gemma 4 MTP assistant,
    MERGED — native ``Gemma4Proposer`` in the current pin)
  * https://github.com/vllm-project/vllm/pull/42069 (DFlash backend=None, OPEN)
"""
from __future__ import annotations

import logging

from ._gemma4_detect import (
    detect_non_causal_drafter,
    env_disable,
    env_truthy,
    is_ampere_sm86,
    is_gemma4_arch,
)

log = logging.getLogger("genesis.gemma4.g4_03_non_causal_drafter_guard")

GENESIS_G4_03_MARKER = (
    "Genesis G4_03 gemma4 ampere non-causal drafter guard v2 "
    "(base-proposer wrap, fail-loud bindings; closes operator confusion "
    "from vllm#40382 backend matrix wall)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_03_GEMMA4_NON_CAUSAL_DRAFTER_GUARD"
_ENV_DISABLE = "GENESIS_DISABLE_G4_03_GUARD"

# Human-readable drafter names for the refusal message.
_DRAFTER_DISPLAY = {"eagle3": "EAGLE-3", "dflash": "DFlash"}

_APPLIED = False
# (cls, attr_name, original) triples for exact revert without re-imports.
_WRAPPED_SITES: list[tuple[type, str, object]] = []


def _env_enabled() -> bool:
    if env_disable(_ENV_DISABLE):
        return False
    return env_truthy(_ENV_ENABLE)


def _proposer_targets_gemma4(self) -> bool:
    """Best-effort: detect whether the proposer's target model is Gemma 4."""
    # Common attribute names across proposer flavors; the pristine base
    # __init__ sets self.vllm_config (llm_base_proposer.py:67).
    for attr in ("target_vllm_config", "_target_vllm_config", "vllm_config"):
        cfg = getattr(self, attr, None)
        if cfg is None:
            continue
        # is_gemma4_arch handles vllm_config → model_config → hf_config.
        if is_gemma4_arch(cfg):
            return True
        # Defensive extra probe for model_config objects exposing
        # architectures/model_type directly (no hf_config).
        mc = getattr(cfg, "model_config", None)
        if mc is not None and is_gemma4_arch(
            getattr(mc, "hf_config", None) or mc
        ):
            return True
    return False


def _detect_drafter_method(self) -> str | None:
    """Return "eagle3"/"dflash" when the proposer drafts non-causally.

    Primary source: speculative_config.method. Fallback: the proposer's
    own ``self.method`` mirror (pristine base __init__ line 71:
    ``self.method = self.speculative_config.method``) —
    detect_non_causal_drafter reads ``.method`` off whatever it's given.
    """
    drafter = detect_non_causal_drafter(
        getattr(self, "speculative_config", None)
    )
    if drafter is None:
        drafter = detect_non_causal_drafter(self)
    return drafter


def _make_guarded_create_draft(site: str, original):
    def _genesis_g4_03_guarded_create_draft(self, *args, **kwargs):
        try:
            drafter = _detect_drafter_method(self)
            if (
                drafter is not None
                and is_ampere_sm86()
                and _proposer_targets_gemma4(self)
            ):
                display = _DRAFTER_DISPLAY.get(drafter, drafter)
                # Check G4_10 active (operator opted in to deep fix)
                import os
                if os.environ.get(
                    "GENESIS_ENABLE_G4_10_GEMMA4_AMPERE_NON_CAUSAL_BACKEND", ""
                ).strip() in ("1", "true", "yes", "on"):
                    log.info(
                        "[G4_03] %s drafter on Ampere + Gemma 4 — G4_10 enabled, "
                        "letting drafter through to use Genesis Triton backend",
                        display,
                    )
                else:
                    raise RuntimeError(
                        f"[Genesis G4_03] Refusing {display} drafter on Ampere SM 8.6 "
                        "with Gemma 4 target.\n"
                        "\n"
                        f"{display} uses non-causal block-parallel attention. Gemma 4 has "
                        "head_dim=256 (sliding) and head_dim=512 (global). No Ampere SM 8.6 "
                        "attention backend supports both constraints well — see vllm#40382 "
                        "backend matrix:\n"
                        "  FA2 / FA2_DIFFKV / FLASHINFER / TRITON_ATTN / TREE_ATTN — causal-only.\n"
                        "  FLEX_ATTENTION supports both but is slow enough that "
                        "draft-acceptance gain is wiped out.\n"
                        "\n"
                        "RECOMMENDED — switch to the native Gemma 4 MTP assistant drafter\n"
                        "(Gemma4Proposer, vllm/v1/spec_decode/gemma4.py — in this pin):\n"
                        "  speculative_config:\n"
                        "    method: mtp\n"
                        "    model: /models/Gemma-4-31B-it-assistant\n"
                        "    num_speculative_tokens: 8\n"
                        "The assistant checkpoint must carry model_type gemma4_assistant /\n"
                        "gemma4_unified_assistant (vLLM normalizes it to gemma4_mtp and\n"
                        "routes to Gemma4Proposer). It shares the target's KV cache via\n"
                        "cross-model KV sharing and drafts causally — works on Ampere.\n"
                        "Landed upstream via vllm#41745 (MERGED).\n"
                        "\n"
                        "DEEP FIX — enable Genesis G4_10 Ampere non-causal Triton attention:\n"
                        "  GENESIS_ENABLE_G4_10_GEMMA4_AMPERE_NON_CAUSAL_BACKEND=1\n"
                        "(requires G4_10 implemented — see ``sndr patches list --family gemma4``)\n"
                        "\n"
                        "OVERRIDE — bypass to get the raw backend matrix error:\n"
                        f"  {_ENV_DISABLE}=1\n"
                    )
        except Exception as e:  # noqa: BLE001
            if isinstance(e, RuntimeError) and "[Genesis G4_03]" in str(e):
                raise
            log.warning(
                "[G4_03] detection raised %r at site %s; falling through to upstream",
                e, site,
            )
        return original(self, *args, **kwargs)

    _genesis_g4_03_guarded_create_draft._genesis_g4_03_wrapped = True
    _genesis_g4_03_guarded_create_draft._genesis_g4_03_original = original
    _genesis_g4_03_guarded_create_draft.__wrapped__ = original
    return _genesis_g4_03_guarded_create_draft


def _wrap_class_method(cls: type, site: str) -> bool:
    """Wrap cls._create_draft_vllm_config in place; True when wrapped."""
    method = getattr(cls, "_create_draft_vllm_config", None)
    if method is None:
        return False
    if getattr(method, "_genesis_g4_03_wrapped", False):
        # Already guarded (e.g. inherited from an already-wrapped base).
        return True
    _WRAPPED_SITES.append((cls, "_create_draft_vllm_config", method))
    cls._create_draft_vllm_config = _make_guarded_create_draft(site, method)
    return True


def apply() -> tuple[str, str]:
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"G4_03 disabled (set {_ENV_ENABLE}=1 to refuse Eagle3/DFlash drafters on "
            "Ampere with Gemma 4 — see vllm#40382)"
        )

    if _APPLIED:
        return "applied", "G4_03 already installed (idempotent)"

    # Primary binding: the shared base-class method. Covers every
    # non-causal drafter regardless of module layout — EagleProposer
    # inherits it, DFlashProposer's override delegates to super().
    # The wrapper's method gate keeps causal proposers untouched.
    try:
        from vllm.v1.spec_decode import llm_base_proposer as base_mod
        base_cls = getattr(base_mod, "SpecDecodeBaseProposer", None)
        if base_cls is not None and _wrap_class_method(base_cls, "base"):
            _APPLIED = True
            log.info(
                "[G4_03] installed on SpecDecodeBaseProposer._create_draft_vllm_config "
                "(gated on method in eagle3/dflash). EAGLE-3 / DFlash + Gemma 4 + "
                "Ampere will now raise a clear error.",
            )
            return "applied", (
                "G4_03 installed on SpecDecodeBaseProposer._create_draft_vllm_config "
                "(covers Eagle3 + DFlash; method-gated). "
                f"Override via {_ENV_DISABLE}=1 for G4_10 testing."
            )
        log.warning(
            "[G4_03] llm_base_proposer importable but SpecDecodeBaseProposer/"
            "_create_draft_vllm_config not found — trying per-class fallback",
        )
    except ImportError as e:
        log.warning(
            "[G4_03] vllm.v1.spec_decode.llm_base_proposer not importable (%s) "
            "— trying per-class fallback", e,
        )

    # Fallback: per-class probes. eagle3.py was removed from the pin
    # (verified 2026-06-11) — Eagle3 is served by eagle.EagleProposer;
    # probe legacy class names too in case of future renames.
    fallback_sites: dict[str, bool] = {}
    for mod_name, cls_names, site in (
        ("eagle", ("EagleProposer", "Eagle3Proposer", "Eagle3DraftProposer"), "eagle"),
        ("dflash", ("DFlashProposer",), "dflash"),
    ):
        fallback_sites[site] = False
        try:
            import importlib
            mod = importlib.import_module(f"vllm.v1.spec_decode.{mod_name}")
        except ImportError as e:
            log.warning(
                "[G4_03] fallback module vllm.v1.spec_decode.%s not importable: %s",
                mod_name, e,
            )
            continue
        for cls_name in cls_names:
            cls = getattr(mod, cls_name, None)
            if cls is not None and _wrap_class_method(cls, f"{site}:{cls_name}"):
                fallback_sites[site] = True

    covered = [site for site, ok in fallback_sites.items() if ok]
    missing = [site for site, ok in fallback_sites.items() if not ok]

    if not covered:
        # FAIL LOUD: an enabled guard that binds nothing must never
        # report success — that is exactly the silent hazard the
        # 2026-06-11 preflight triage caught (eagle3 module removal
        # swallowed at log.debug while Eagle3 ran unguarded).
        log.warning(
            "[G4_03] enabled but could not bind any drafter site (base + "
            "eagle + dflash all missing) — Eagle3/DFlash are UNGUARDED on "
            "this pin; failing the patch so the boot log shows it.",
        )
        return "failed", (
            "G4_03 enabled but no binding site found: "
            "SpecDecodeBaseProposer._create_draft_vllm_config missing and "
            "eagle/dflash class probes found nothing. Eagle3/DFlash would "
            "run unguarded — refusing to report success. Re-verify the "
            "spec_decode layout of the current pin."
        )

    if missing:
        # Partial coverage is still a failure mode — report it loudly so
        # the dispatcher marks the patch failed instead of half-guarded.
        log.warning(
            "[G4_03] partial fallback coverage: wrapped %s but %s drafter "
            "class(es) not found — that family is UNGUARDED.",
            ", ".join(covered), ", ".join(missing),
        )
        return "partial", (
            f"G4_03 fallback wrapped {', '.join(covered)} but found no "
            f"{', '.join(missing)} drafter class — that family is unguarded. "
            "Re-verify the spec_decode layout of the current pin."
        )

    _APPLIED = True
    log.info(
        "[G4_03] installed via per-class fallback: wrapped %s. "
        "EAGLE-3 / DFlash + Gemma 4 + Ampere will now raise a clear error.",
        ", ".join(covered),
    )
    return "applied", (
        f"G4_03 installed via per-class fallback ({', '.join(covered)}; "
        "base-class binding unavailable). "
        f"Override via {_ENV_DISABLE}=1 for G4_10 testing."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED
    if not _APPLIED and not _WRAPPED_SITES:
        return False
    reverted = False
    for cls, attr, original in _WRAPPED_SITES:
        setattr(cls, attr, original)
        reverted = True
    _WRAPPED_SITES.clear()
    _APPLIED = False
    return reverted


__all__ = ["GENESIS_G4_03_MARKER", "apply", "is_applied", "revert"]
