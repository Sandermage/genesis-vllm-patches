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

Wrap ``DFlashProposer._create_draft_vllm_config`` and
``EagleProposer._create_draft_vllm_config`` so they refuse at config
build time with a clear error pointing at:

  * upstream bug #40382 (backend matrix)
  * the recommended Ampere drafter (Google MTP assistant via PR #41745)
  * the deep fix (G4_10 — Genesis non-causal head_dim=256 Triton kernel)

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

* default_on: True
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
  * https://github.com/vllm-project/vllm/pull/41745 (MTP assistant, MERGED)
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
    "Genesis G4_03 gemma4 ampere non-causal drafter guard v1 "
    "(closes operator confusion from vllm#40382 backend matrix wall)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_03_GEMMA4_NON_CAUSAL_DRAFTER_GUARD"
_ENV_DISABLE = "GENESIS_DISABLE_G4_03_GUARD"

_APPLIED = False
_ORIGINAL_METHODS: dict[str, object] = {}


def _env_enabled() -> bool:
    if env_disable(_ENV_DISABLE):
        return False
    return env_truthy(_ENV_ENABLE)


def _proposer_targets_gemma4(self) -> bool:
    """Best-effort: detect whether the proposer's target model is Gemma 4."""
    # Common attribute names across proposer flavors
    for attr in ("target_vllm_config", "_target_vllm_config", "vllm_config"):
        cfg = getattr(self, attr, None)
        if cfg is not None:
            mc = getattr(cfg, "model_config", None) or cfg
            if is_gemma4_arch(mc):
                return True
    return False


def _make_guarded_create_draft(method_name: str, original):
    def _genesis_g4_03_guarded_create_draft(self, *args, **kwargs):
        try:
            if (
                is_ampere_sm86()
                and _proposer_targets_gemma4(self)
            ):
                # Check G4_10 active (operator opted in to deep fix)
                import os
                if os.environ.get("GENESIS_ENABLE_G4_10_GEMMA4_AMPERE_NON_CAUSAL_BACKEND", "").strip() in (
                    "1", "true", "yes", "on",
                ):
                    log.info(
                        "[G4_03] %s drafter on Ampere + Gemma 4 — G4_10 enabled, "
                        "letting drafter through to use Genesis Triton backend",
                        method_name,
                    )
                else:
                    raise RuntimeError(
                        f"[Genesis G4_03] Refusing {method_name} drafter on Ampere SM 8.6 "
                        "with Gemma 4 target.\n"
                        "\n"
                        f"{method_name} uses non-causal block-parallel attention. Gemma 4 has "
                        "head_dim=256 (sliding) and head_dim=512 (global). No Ampere SM 8.6 "
                        "attention backend supports both constraints well — see vllm#40382 "
                        "backend matrix:\n"
                        "  FA2 / FA2_DIFFKV / FLASHINFER / TRITON_ATTN / TREE_ATTN — causal-only.\n"
                        "  FLEX_ATTENTION supports both but is slow enough that "
                        "draft-acceptance gain is wiped out.\n"
                        "\n"
                        "RECOMMENDED — switch to Google MTP assistant drafter:\n"
                        "  speculative_config:\n"
                        "    method: mtp\n"
                        "    model: /models/Gemma-4-31B-it-assistant\n"
                        "    num_speculative_tokens: 8\n"
                        "MTP is causal-shared with target — uses TRITON_ATTN, works on Ampere.\n"
                        "vllm#41745 (MERGED) adds Gemma 4 MTP support.\n"
                        "\n"
                        "DEEP FIX — enable Genesis G4_10 Ampere non-causal Triton attention:\n"
                        "  GENESIS_ENABLE_G4_10_GEMMA4_AMPERE_NON_CAUSAL_BACKEND=1\n"
                        "(requires G4_10 implemented — see ``sndr patches list --family gemma4``)\n"
                        "\n"
                        "OVERRIDE — bypass to get the raw backend matrix error:\n"
                        f"  {_ENV_DISABLE}=1\n"
                    )
        except Exception as e:  # noqa: BLE001
            if not isinstance(e, RuntimeError) or "[Genesis G4_03]" not in str(e):
                log.warning(
                    "[G4_03] detection raised %r; falling through to upstream", e,
                )
            else:
                raise
        return original(self, *args, **kwargs)

    _genesis_g4_03_guarded_create_draft._genesis_g4_03_wrapped = True
    _genesis_g4_03_guarded_create_draft._genesis_g4_03_original = original
    _genesis_g4_03_guarded_create_draft.__wrapped__ = original
    return _genesis_g4_03_guarded_create_draft


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_METHODS

    if not _env_enabled():
        return "skipped", (
            f"G4_03 disabled (set {_ENV_ENABLE}=1 to refuse Eagle3/DFlash drafters on "
            "Ampere with Gemma 4 — see vllm#40382)"
        )

    if _APPLIED:
        return "applied", "G4_03 already installed (idempotent)"

    wrapped_count = 0

    # DFlash proposer
    try:
        from vllm.v1.spec_decode import dflash as mod
        cls = getattr(mod, "DFlashProposer", None)
        if cls is not None:
            method = getattr(cls, "_create_draft_vllm_config", None)
            if method is not None and not getattr(method, "_genesis_g4_03_wrapped", False):
                _ORIGINAL_METHODS["dflash_create_draft"] = method
                cls._create_draft_vllm_config = _make_guarded_create_draft("DFlash", method)
                wrapped_count += 1
    except ImportError as e:
        log.debug("vllm.v1.spec_decode.dflash not importable: %s", e)

    # Eagle3 proposer — class layout varies; probe several names
    try:
        from vllm.v1.spec_decode import eagle3 as mod
        for cls_name in ("Eagle3Proposer", "Eagle3DraftProposer", "EagleProposer"):
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            method = getattr(cls, "_create_draft_vllm_config", None)
            if method is not None and not getattr(method, "_genesis_g4_03_wrapped", False):
                _ORIGINAL_METHODS[f"eagle3_{cls_name}_create_draft"] = method
                cls._create_draft_vllm_config = _make_guarded_create_draft("Eagle3", method)
                wrapped_count += 1
    except ImportError as e:
        log.debug("vllm.v1.spec_decode.eagle3 not importable: %s", e)

    if wrapped_count == 0:
        return "skipped", (
            "Could not locate any non-causal drafter class to guard; pin may have "
            "renamed the path or both drafters are absent. G4_03 is no-op."
        )

    _APPLIED = True
    log.info(
        "[G4_03] installed: wrapped %d non-causal drafter class(es). "
        "EAGLE-3 / DFlash + Gemma 4 + Ampere will now raise a clear error.",
        wrapped_count,
    )
    return "applied", (
        f"G4_03 installed: {wrapped_count} non-causal drafter class(es) wrapped. "
        f"Override via {_ENV_DISABLE}=1 for G4_10 testing."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_METHODS
    if not _APPLIED:
        return False
    reverted = False
    try:
        from vllm.v1.spec_decode import dflash as mod
        cls = getattr(mod, "DFlashProposer", None)
        if cls is not None and "dflash_create_draft" in _ORIGINAL_METHODS:
            cls._create_draft_vllm_config = _ORIGINAL_METHODS["dflash_create_draft"]  # type: ignore[assignment]
            reverted = True
    except ImportError:
        pass
    try:
        from vllm.v1.spec_decode import eagle3 as mod
        for key in list(_ORIGINAL_METHODS):
            if key.startswith("eagle3_"):
                cls_name = key[len("eagle3_"):-len("_create_draft")]
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    cls._create_draft_vllm_config = _ORIGINAL_METHODS[key]  # type: ignore[assignment]
                    reverted = True
    except ImportError:
        pass
    if reverted:
        _APPLIED = False
        _ORIGINAL_METHODS.clear()
    return reverted


__all__ = ["GENESIS_G4_03_MARKER", "apply", "is_applied", "revert"]
