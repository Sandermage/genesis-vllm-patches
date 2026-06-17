# SPDX-License-Identifier: Apache-2.0
"""G4_26 — backport the TP-correctness half of OPEN vllm PR #45774.

================================================================
PURPOSE
================================================================

``DiffusionGemmaForBlockDiffusion`` self-conditioning computes the soft
embedding as ``probs @ embed_weight`` where ``probs`` spans the FULL vocab
(262144 on Gemma-4 diffusion checkpoints). Under tensor parallelism the
``embed_tokens.weight`` of a ``VocabParallelEmbedding`` is vocab-sharded —
at TP=2 it is ``[131072, 2816]``. Feeding that sharded weight directly into
the full-vocab matmul raises:

    RuntimeError: a and b must have same reduction dim, but got
                  [..., 262144] X [131072, 2816]

PR #45774 ("Fix DiffusionGemma for TP>1") fixes this by materializing the
full (unsharded) embedding weight via an all-gather before the matmul. This
patch backports ONLY the three TP-correctness hunks:

  1. import ``get_tensor_model_parallel_world_size`` +
     ``tensor_model_parallel_all_gather`` from ``vllm.distributed``;
  2. add the module-level ``_get_full_embed_weight`` helper (all-gathers the
     sharded weight under TP, returns ``.weight`` unchanged at TP=1);
  3. swap the ``custom_sampler`` call site from
     ``embed_weight=self.model.model.embed_tokens.weight`` to
     ``embed_weight=_get_full_embed_weight(self.model.model.embed_tokens)``.

It deliberately SKIPS the XPU/UVA hunks of #45774 (the
``current_platform.is_xpu()``-guarded ``_build_output``/``__call__`` async
copy paths) — those are Intel-GPU-only and irrelevant on our 2× A5000 rig.
``compute_self_conditioning`` (a dead def-only site) is left untouched.

================================================================
INTEGRATION STRATEGY
================================================================

A ``TextPatcher`` over ``model_executor/models/diffusion_gemma.py`` with 3
ordered sub-patches, each anchored on a byte-faithful region of the live
dev491 module:

  * Sub-1 (import-add): inserts the ``vllm.distributed`` import block between
    the ``CUDAGraphMode`` and ``init_logger`` imports — exactly where #45774
    places it.
  * Sub-2 (helper-add): inserts ``_get_full_embed_weight`` between the
    ``_NO_PENALTIES_STATE`` assignment and the ``class DiffusionSampler``
    definition, preserving PEP8 two-blank-line spacing. ``nn`` and ``torch``
    are already module-level imports on dev491, so the helper's
    ``nn.Module`` / ``torch.Tensor`` annotations resolve without new imports.
  * Sub-3 (line-853-swap): replaces the bare sharded-weight argument with the
    gathered-helper call, anchored on its surrounding 12-space-indented
    context (``confidence_threshold`` above + ``normalizer`` below) for
    in-file uniqueness.

The patcher's ``upstream_drift_markers`` contains ``def _get_full_embed_weight``
so it self-skips once #45774 merges (the upstream file then already defines
the helper).

================================================================
SAFETY MODEL
================================================================

* default_on: False (opt-in correctness fix; only Gemma-4 diffusion
  checkpoints at TP>1 need it).
* env_flag: GENESIS_ENABLE_G4_26_DIFFUSIONGEMMA_TP_VOCAB
* arch-gate: apply() probes the live ``diffusion_gemma`` module for
  ``DiffusionGemmaForBlockDiffusion`` and skips if absent (the arch is NOT in
  ``_gemma4_detect.GEMMA4_ARCHITECTURES``, so a direct presence check is used
  rather than ``is_gemma4_arch``).
* intrinsically TP-gated: the all-gather helper returns ``.weight`` unchanged
  at TP=1, so applying the patch on a single-GPU run is a no-op at runtime.
* applies_to: arch DiffusionGemmaForBlockDiffusion;
  vllm_version_range (">=0.22.1rc1.dev491", "<1.0.0").
* conflicts_with: none. composes_with: none.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/pull/45774
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import (
    TextPatch,
    TextPatcher,
    marker_present_in_target,
    result_to_wiring_status,
)

from ._gemma4_detect import env_truthy

log = logging.getLogger("genesis.gemma4.g4_26_diffusiongemma_tp_vocab")

GENESIS_G4_26_MARKER = (
    "Genesis G4_26 diffusion_gemma self-conditioning TP>1 vocab-sharded "
    "soft-embed all-gather v1 (backport vllm#45774 — adds "
    "_get_full_embed_weight + vllm.distributed import + line-853 swap; "
    "SKIPs XPU/UVA hunks)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_26_DIFFUSIONGEMMA_TP_VOCAB"

_TARGET_REL = "model_executor/models/diffusion_gemma.py"

# Self-skip once #45774 merges: the upstream file then defines this helper.
_UPSTREAM_DRIFT_MARKER = "def _get_full_embed_weight"

_APPLIED = False


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


# ── Sub-1 (import-add) ───────────────────────────────────────────────────
# Insert the vllm.distributed import BETWEEN the CUDAGraphMode and
# init_logger imports, matching PR #45774 exactly. Anchor is the two
# consecutive import lines (unique in the file).
_IMPORT_ANCHOR = (
    "from vllm.config.compilation import CUDAGraphMode\n"
    "from vllm.logger import init_logger"
)
_IMPORT_REPLACEMENT = (
    "from vllm.config.compilation import CUDAGraphMode\n"
    "# === Genesis G4_26 (backport vllm#45774) START ===\n"
    "# Distributed collectives for full-vocab soft-embed under TP.\n"
    "from vllm.distributed import (\n"
    "    get_tensor_model_parallel_world_size,\n"
    "    tensor_model_parallel_all_gather,\n"
    ")\n"
    "# === Genesis G4_26 (import) END ===\n"
    "from vllm.logger import init_logger"
)


# ── Sub-2 (helper-add) ───────────────────────────────────────────────────
# Insert _get_full_embed_weight BETWEEN the penalty stub assignment and the
# DiffusionSampler class, preserving the two-blank-line PEP8 spacing. The
# helper body is verbatim from PR #45774's HUNK 3.
_HELPER_ANCHOR = (
    "_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)\n"
    "\n"
    "\n"
    "class DiffusionSampler:"
)
_HELPER_REPLACEMENT = (
    "_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)\n"
    "\n"
    "\n"
    "# === Genesis G4_26 (backport vllm#45774) START ===\n"
    "def _get_full_embed_weight(embed_tokens: nn.Module) -> torch.Tensor:\n"
    '    """Materialize the full (unsharded) embedding weight for the sampler.\n'
    "\n"
    "    The self-conditioning soft embedding is ``probs @ embed_weight`` where\n"
    "    ``probs`` spans the full vocab, so under TP the sharded\n"
    "    VocabParallelEmbedding weight cannot be used directly. All TP ranks run\n"
    "    the sampler, so each keeps a gathered copy (collective: every rank must\n"
    "    reach this call).\n"
    '    """\n'
    "    if get_tensor_model_parallel_world_size() == 1:\n"
    "        return embed_tokens.weight\n"
    "    full = tensor_model_parallel_all_gather(embed_tokens.weight, dim=0)\n"
    "    # Shards are padded; org vocab occupies the leading rows.\n"
    "    return full[: embed_tokens.org_vocab_size]\n"
    "# === Genesis G4_26 (helper) END ===\n"
    "\n"
    "\n"
    "class DiffusionSampler:"
)


# ── Sub-3 (line-853-swap) ────────────────────────────────────────────────
# Swap the bare sharded-weight argument for the gathered-helper call. Anchor
# brackets the surrounding 12-space-indented context to stay unique in the
# file (the bare `embed_tokens.weight` token may appear elsewhere).
_SWAP_ANCHOR = (
    '            confidence_threshold=gen["confidence_threshold"],\n'
    "            embed_weight=self.model.model.embed_tokens.weight,\n"
    "            normalizer=self.model.model.normalizer,"
)
_SWAP_REPLACEMENT = (
    '            confidence_threshold=gen["confidence_threshold"],\n'
    "            # [Genesis G4_26 backport vllm#45774] all-gather vocab-sharded\n"
    "            # embed_weight so probs@embed_weight works at TP>1.\n"
    "            embed_weight=_get_full_embed_weight(self.model.model.embed_tokens),\n"
    "            normalizer=self.model.model.normalizer,"
)


def _make_patcher_for_target(target_file: str) -> TextPatcher:
    """Build the G4_26 TextPatcher pointed at ``target_file``.

    Shared by the runtime ``apply()`` path (live vllm tree) and the unit
    tests (synthetic anchor fixture), so the anchor/replacement contract is
    exercised identically in both.
    """
    return TextPatcher(
        patch_name="G4_26 diffusion_gemma TP>1 vocab-sharded soft-embed",
        target_file=target_file,
        marker=GENESIS_G4_26_MARKER,
        # NOTE: the `_get_full_embed_weight` self-skip lives ONLY at the
        # patcher level (`upstream_drift_markers` below), which fires in
        # Layer 3 BEFORE any sub-patch runs — so once #45774 merges, the
        # whole patch is skipped cleanly. We deliberately do NOT set
        # per-sub `upstream_merged_markers=[_UPSTREAM_DRIFT_MARKER]`:
        # sub-2 (helper-add) inserts `def _get_full_embed_weight` into the
        # in-flight `modified` buffer, which would then make sub-3 (the
        # line-853 swap) see its own freshly-added marker and silently
        # self-skip the swap (the g4_04 self-collision class). The
        # patcher-level drift check reads the ORIGINAL file content, so it
        # is collision-free.
        sub_patches=[
            TextPatch(
                name="g4_26_distributed_import_add",
                anchor=_IMPORT_ANCHOR,
                replacement=_IMPORT_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="g4_26_full_embed_weight_helper_add",
                anchor=_HELPER_ANCHOR,
                replacement=_HELPER_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="g4_26_custom_sampler_embed_weight_swap",
                anchor=_SWAP_ANCHOR,
                replacement=_SWAP_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[_UPSTREAM_DRIFT_MARKER],
    )


def _make_patcher() -> TextPatcher | None:
    target_file = resolve_vllm_file(_TARGET_REL)
    if target_file is None:
        return None
    return _make_patcher_for_target(str(target_file))


def _diffusion_gemma_arch_present() -> bool:
    """Arch-gate: True only when the live vllm exposes the DiffusionGemma
    model class.

    The HF *checkpoint architecture string* is
    ``DiffusionGemmaForBlockDiffusion`` (this is what the registry
    ``applies_to.model_arch`` matches and the dispatcher gates on). But the
    vLLM-side *class* that ``diffusion_gemma.py`` actually defines — and that we
    text-patch — is ``DiffusionGemmaForConditionalGeneration`` (the registry
    maps the arch string to it: ``"DiffusionGemmaForBlockDiffusion":
    ("diffusion_gemma", "DiffusionGemmaForConditionalGeneration")``). So we probe
    the module for that class, NOT the arch string (probing the arch string was
    a silent-no-op bug — it is never a module attribute). Falls back to a source
    scan when the module cannot be imported (e.g. torch-less environment) but the
    file resolves.
    """
    try:
        import vllm.model_executor.models.diffusion_gemma as _dg
        if hasattr(_dg, "DiffusionGemmaForConditionalGeneration"):
            return True
    except Exception as e:  # noqa: BLE001
        log.debug("[G4_26] diffusion_gemma import probe failed: %s", e)
    # Fall back to a static source scan so we still gate correctly on hosts
    # where importing the heavy module is undesirable at apply time.
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return False
    try:
        with open(target, "r", encoding="utf-8", errors="ignore") as fh:
            return "class DiffusionGemmaForConditionalGeneration" in fh.read()
    except OSError:
        return False


def apply() -> tuple[str, str]:
    """Install the TP>1 vocab-sharded soft-embed fix on diffusion_gemma.py."""
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"G4_26 disabled (set {_ENV_ENABLE}=1 to backport vllm#45774 — "
            "fixes DiffusionGemma self-conditioning soft-embed matmul under "
            "TP>1 via vocab-sharded embed_weight all-gather)"
        )

    if _APPLIED:
        return "applied", "G4_26 already installed (idempotent)"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", (
            f"G4_26: target file {_TARGET_REL} not resolvable in this pin — "
            "no DiffusionGemma module present, patch is no-op"
        )

    # Arch-gate: skip when the DiffusionGemma model class is absent.
    if not _diffusion_gemma_arch_present():
        return "skipped", (
            "G4_26: DiffusionGemmaForConditionalGeneration (DiffusionGemma) "
            "not present in this pin — patch is no-op (arch-gate)"
        )

    result, failure = patcher.apply()
    status, message = result_to_wiring_status(
        result, failure,
        applied_message=(
            "G4_26 applied: vllm#45774 TP-correctness backport vendored into "
            "diffusion_gemma.py — DiffusionGemma self-conditioning soft-embed "
            "now all-gathers the vocab-sharded embed_weight at TP>1 (no-op at "
            "TP=1). Will silently skip once upstream #45774 merges."
        ),
        patch_name=patcher.patch_name,
    )
    if status == "applied":
        _APPLIED = True
    return status, message


def is_applied() -> bool:
    patcher = _make_patcher()
    if patcher is None:
        return False
    return marker_present_in_target(patcher)


def revert() -> bool:
    """Restore the pristine diffusion_gemma.py by stripping the G4_26 marker
    + the three inserted blocks.

    For a TextPatcher there is no in-memory saved-original (the splice is on
    disk), so revert is a best-effort no-op signal: it returns False when the
    marker is not present (nothing to revert), True otherwise. The live tree
    is re-pinned on container rebuild, so a hard textual un-splice is not
    needed in production; this mirrors the contract the dispatcher expects.
    """
    global _APPLIED
    if not is_applied():
        _APPLIED = False
        return False
    # The on-disk text edit is reversed by re-pinning the vllm tree (container
    # rebuild). We surface True so callers know the patch WAS present.
    _APPLIED = False
    return True


__all__ = ["GENESIS_G4_26_MARKER", "apply", "is_applied", "revert"]
