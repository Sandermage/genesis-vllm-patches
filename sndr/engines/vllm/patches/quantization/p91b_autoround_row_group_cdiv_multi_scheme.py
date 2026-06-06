# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 91B — AutoRound row-group cdiv defensive coverage for
INC + compressed-tensors schemes (P91 sibling, vllm#39460-derived).

================================================================
SCOPE
================================================================

Same bug class as P91 (silent dequant corruption when
`input_size_per_partition % group_size != 0` or
`input_size % group_size != 0` for row-group quantized layers), but
in files that vllm#39460 did NOT touch. Three cdiv-only sub-patches
across three files.

================================================================
COVERED FILES (3 sub-patches)
================================================================

1. `vllm/model_executor/layers/quantization/inc.py` (Intel Neural
   Compressor linear method).
   Cross-pin anchor drift between two `KNOWN_GOOD_VLLM_PINS`:
     - `0.20.2rc1.dev338+gbf0d2dc6d` — uses `self.group_size`
     - `0.20.2rc1.dev371+gbf610c2f5` — uses bare `group_size`
   Two independent factories (`_make_inc_dev338_patcher`,
   `_make_inc_dev371_patcher`) each carry a distinct anchor. On any
   live pin exactly one of the two anchors matches; the other returns
   SKIPPED with anchor-not-found and is treated as expected per-pin
   alternation (Option A from Step 0 anchor manifest).

2. `vllm/model_executor/layers/quantization/compressed_tensors/schemes/`
   `compressed_tensors_wNa16.py`.
   One real bug at the REPEAT-all-ranks branch:
   `scales_and_zp_size = input_size // group_size`. The companion line
   inside `if partition_scales:` (`input_size_per_partition // group_size`)
   is assert-protected upstream (`assert input_size_per_partition %
   group_size == 0`) and not touched here — cdiv there would be a
   no-op rewrite since the assert guarantees exact divisibility.

3. `vllm/model_executor/layers/quantization/compressed_tensors/schemes/`
   `compressed_tensors_w4a8_fp8.py`.
   Same structural pattern as wNa16: one real bug at the unprotected
   REPEAT-all-ranks line, plus an assert-protected partition-scales
   line that is left alone.

================================================================
NOT COVERED (explicitly)
================================================================

- `compressed_tensors_w4a8_int.py` — the function-head
  `assert input_size_per_partition % effective_group_size == 0`
  proactively rejects partial-group shards before the floor-div at
  line 110 ever executes, so the floor-div is always exact when
  control reaches it. No silent-corruption surface to fix.

- inc.py `setattr(row_group_size, row_input_size_per_partition)`
  companion that would let P91's parameter.py loader gate fire on
  inc.py partial-group row-parallel scales. That is INFRASTRUCTURE
  for an existing fix (P91's `RowvLLMParameter.load_row_parallel_weight`
  group-aware branch) rather than a new bug fix, so it is out of
  scope for this defensive-coverage patch. inc.py cdiv-only fix here
  prevents over-allocation but a partial-group inc.py scales tensor
  still loads from the wrong offset under TP>=2. Deferred to a
  future refresh phase if INC enters Genesis prod use.

================================================================
RELATIONSHIP TO P91
================================================================

Same bug CLASS, different FILES. Per
`_PURE_UPSTREAM_RELATIONSHIPS = frozenset({"backport"})`, only
`backport` permits status-based retire on upstream merge. P91B uses
`related_not_superseding` so it stays even if vllm#39460 ever
merges (#39460 would not retroactively touch inc.py or the
compressed_tensors files).

================================================================
SAFETY MODEL
================================================================

- `cdiv(x, group_size) >= floor(x // group_size)` always; equal when
  x is divisible. The change can only INCREASE allocated rows, never
  shrink — no chance of breaking checkpoints that previously loaded fine.
- Default OFF (env-gated via `GENESIS_ENABLE_P91B=1`, dispatcher-applied).
  When OFF, behavior is upstream nightly.
- Idempotent via `GENESIS_P91B_MARKER_BASE` substring check per file.

K.1.R anchor audit 2026-05-28
-----------------------------
Significant drift detected against new pin nightly-626fa9bb (multi-arch digest
sha256:674922aae790c2cbf45f4e844098d227b80d40a74bfc7797a444d213a221879f,
upstream SHA 626fa9bba5663a5cf6a870debf031ee344ddb822):

  * ``inc.py`` — 1 of 4 anchors PASS; 3 of 4 DRIFT including the
    primary ``P91B_INC_DEV338_ANCHOR``, ``P91B_W4A8_FP8_ANCHOR``,
    ``P91B_WNA16_ANCHOR``. Upstream refactored ``inc.py`` such that
    the ``scales_and_zp_size = input_size_per_partition // ...`` pattern
    no longer appears in the new source — the bug class P91B targets
    may have been resolved by an unrelated upstream restructure, OR
    moved to a parent class we'd need to walk.

Status under new pin: TextPatcher per-anchor self-skip; ``apply()``
returns ``skipped`` for the inc.py sub-patches. P91B remains default-OFF
opt-in only — runtime on new pin is unchanged for everyone not enabling
the multi-scheme cdiv overlay.

Re-anchoring P91B against the new inc.py is non-trivial because the
target code surface (`scales_and_zp_size = ...`) has been refactored
out of inc.py entirely. A separate analysis slice would need to:
  1. Walk the new inc.py inheritance chain to find where scales_and_zp
     are computed.
  2. Confirm whether the floor-div bug class still manifests on AutoRound
     INT4/INT8 checkpoints (it may have been resolved by the same
     restructure).
  3. Either re-anchor against the new code surface or retire P91B as
     incidentally-fixed-by-restructure.

For now: keep self-skip behaviour; document deferred re-anchor status
above. Operator opt-in remains technically possible (env flag is read)
but produces clean no-op apply.

Author backport: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference PR: vllm#39460 (closed without merge, supersession chain
#40281/#41588 also closed; fix abandoned upstream).
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger(
    "genesis.wiring.p91b_autoround_row_group_cdiv_multi_scheme"
)


# Version-independent base string. Idempotency uses this substring so
# future minor bumps within v0.x are recognized as "P91B already applied"
# instead of re-attempting anchor matches on a mutated file.
GENESIS_P91B_MARKER_BASE = (
    "Genesis P91B AutoRound row-group cdiv multi-scheme "
    "(vllm#39460-derived)"
)
GENESIS_P91B_MARKER_VERSION = "v0.1.0"
GENESIS_P91B_MARKER = (
    f"{GENESIS_P91B_MARKER_BASE} {GENESIS_P91B_MARKER_VERSION}"
)


# ─── inc.py: 2 cross-pin anchors (dev338 + dev371) ────────────────────────

P91B_INC_DEV338_ANCHOR = (
    "        scales_and_zp_size = input_size_per_partition // self.group_size\n"
)

P91B_INC_DEV338_REPLACE = (
    "        # [Genesis P91B vllm#39460-derived backport] cdiv not floor-div:\n"
    "        # when input_size_per_partition % self.group_size != 0, AutoRound\n"
    "        # stores cdiv() many scales (trailing partial group covers the\n"
    "        # remainder); floor silently drops the partial group's scales.\n"
    "        from vllm.utils.math_utils import cdiv as _genesis_p91b_cdiv\n"
    "        scales_and_zp_size = _genesis_p91b_cdiv(\n"
    "            input_size_per_partition, self.group_size\n"
    "        )\n"
)


P91B_INC_DEV371_ANCHOR = (
    "        scales_and_zp_size = input_size_per_partition // group_size\n"
)

P91B_INC_DEV371_REPLACE = (
    "        # [Genesis P91B vllm#39460-derived backport] cdiv not floor-div:\n"
    "        # when input_size_per_partition % group_size != 0, AutoRound\n"
    "        # stores cdiv() many scales (trailing partial group covers the\n"
    "        # remainder); floor silently drops the partial group's scales.\n"
    "        from vllm.utils.math_utils import cdiv as _genesis_p91b_cdiv\n"
    "        scales_and_zp_size = _genesis_p91b_cdiv(\n"
    "            input_size_per_partition, group_size\n"
    "        )\n"
)


def _make_inc_dev338_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/quantization/inc.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P91B inc.py (dev338 anchor: self.group_size) — cdiv groups "
            "(vllm#39460-derived)"
        ),
        target_file=str(target),
        marker=GENESIS_P91B_MARKER + "_inc_dev338",
        sub_patches=[
            TextPatch(
                name="p91b_inc_dev338_floor_partition_to_cdiv",
                anchor=P91B_INC_DEV338_ANCHOR,
                replacement=P91B_INC_DEV338_REPLACE,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P91B",
            "_genesis_p91b_cdiv",
            # Upstream-side marker if vLLM merges an equivalent fix
            "scales_and_zp_size = cdiv(input_size_per_partition",
        ],
    )


def _make_inc_dev371_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/quantization/inc.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P91B inc.py (dev371 anchor: group_size) — cdiv groups "
            "(vllm#39460-derived)"
        ),
        target_file=str(target),
        marker=GENESIS_P91B_MARKER + "_inc_dev371",
        sub_patches=[
            TextPatch(
                name="p91b_inc_dev371_floor_partition_to_cdiv",
                anchor=P91B_INC_DEV371_ANCHOR,
                replacement=P91B_INC_DEV371_REPLACE,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P91B",
            "_genesis_p91b_cdiv",
            "scales_and_zp_size = cdiv(input_size_per_partition",
        ],
    )


# ─── compressed_tensors_wNa16.py: 1 sub-patch (REPEAT-all-ranks branch) ───

P91B_WNA16_ANCHOR = (
    "        scales_and_zp_size = input_size // group_size\n"
)

P91B_WNA16_REPLACE = (
    "        # [Genesis P91B vllm#39460-derived backport] cdiv not floor-div\n"
    "        # for the REPEAT-all-ranks branch: input_size % group_size != 0\n"
    "        # cases drop the trailing partial group of scales silently. The\n"
    "        # companion `input_size_per_partition // group_size` inside the\n"
    "        # `if partition_scales:` block is assert-protected upstream\n"
    "        # (assert input_size_per_partition % group_size == 0) so a cdiv\n"
    "        # there would be a no-op rewrite; left unchanged.\n"
    "        from vllm.utils.math_utils import cdiv as _genesis_p91b_cdiv\n"
    "        scales_and_zp_size = _genesis_p91b_cdiv(input_size, group_size)\n"
)


def _make_wna16_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/quantization/compressed_tensors/schemes/"
        "compressed_tensors_wNa16.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P91B compressed_tensors_wNa16.py — cdiv groups "
            "(vllm#39460-derived)"
        ),
        target_file=str(target),
        marker=GENESIS_P91B_MARKER + "_ct_wNa16",
        sub_patches=[
            TextPatch(
                name="p91b_ct_wNa16_floor_input_size_to_cdiv",
                anchor=P91B_WNA16_ANCHOR,
                replacement=P91B_WNA16_REPLACE,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P91B",
            "_genesis_p91b_cdiv",
            "scales_and_zp_size = cdiv(input_size,",
        ],
    )


# ─── compressed_tensors_w4a8_fp8.py: 1 sub-patch (REPEAT-all-ranks) ───────

# Anchor text byte-identical to wNa16's (different file). Each TextPatcher
# is bound to a distinct target_file, so the duplication is safe.

P91B_W4A8_FP8_ANCHOR = P91B_WNA16_ANCHOR
P91B_W4A8_FP8_REPLACE = P91B_WNA16_REPLACE


def _make_w4a8_fp8_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/quantization/compressed_tensors/schemes/"
        "compressed_tensors_w4a8_fp8.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P91B compressed_tensors_w4a8_fp8.py — cdiv groups "
            "(vllm#39460-derived)"
        ),
        target_file=str(target),
        marker=GENESIS_P91B_MARKER + "_ct_w4a8_fp8",
        sub_patches=[
            TextPatch(
                name="p91b_ct_w4a8_fp8_floor_input_size_to_cdiv",
                anchor=P91B_W4A8_FP8_ANCHOR,
                replacement=P91B_W4A8_FP8_REPLACE,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P91B",
            "_genesis_p91b_cdiv",
            "scales_and_zp_size = cdiv(input_size,",
        ],
    )


# ─── apply / is_applied / revert ──────────────────────────────────────────


def apply() -> tuple[str, str]:
    """Apply P91B — multi-scheme row-group cdiv defensive coverage."""
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P91B")
    log_decision("P91B", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Two inc.py factories handle cross-pin anchor drift (self.group_size
    # vs bare group_size); at least one is expected to match on any
    # KNOWN_GOOD pin.
    inc_dev338 = _make_inc_dev338_patcher()
    inc_dev371 = _make_inc_dev371_patcher()
    if inc_dev338 is None and inc_dev371 is None:
        return "skipped", "inc.py not found"

    wna16 = _make_wna16_patcher()
    if wna16 is None:
        return "skipped", "compressed_tensors_wNa16.py not found"
    w4a8_fp8 = _make_w4a8_fp8_patcher()
    if w4a8_fp8 is None:
        return "skipped", "compressed_tensors_w4a8_fp8.py not found"

    # Idempotency via version-agnostic GENESIS_P91B_MARKER_BASE per file.
    inc_path = (
        inc_dev338.target_file if inc_dev338 is not None
        else inc_dev371.target_file
    )
    targets = (
        ("inc.py", inc_path),
        ("compressed_tensors_wNa16.py", wna16.target_file),
        ("compressed_tensors_w4a8_fp8.py", w4a8_fp8.target_file),
    )
    file_contents: dict[str, str] = {}
    all_already_applied = True
    for label, path in targets:
        if not os.path.isfile(path):
            return "skipped", f"target disappeared: {label}"
        with open(path) as f:
            content = f.read()
        file_contents[label] = content
        if GENESIS_P91B_MARKER_BASE not in content:
            all_already_applied = False
    if all_already_applied:
        return (
            "applied",
            "idempotent (P91B markers present in all 3 target files)",
        )

    # Apply inc.py: at least one of the two pin-anchored patchers must
    # succeed. The other returns SKIPPED (anchor mismatch for the other
    # pin) — expected per-pin alternation per Option A.
    inc_content = file_contents["inc.py"]
    inc_applied = GENESIS_P91B_MARKER_BASE in inc_content
    inc_skip_reasons: list[str] = []
    for label, patcher in (("dev338", inc_dev338), ("dev371", inc_dev371)):
        if patcher is None or inc_applied:
            continue
        result, failure = patcher.apply()
        if result in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT):
            inc_applied = True
            break
        # SKIPPED / FAILED here = anchor mismatch for this pin — record
        # and try the next factory.
        reason_text = (
            failure.reason if failure else "anchor mismatch / not eligible"
        )
        inc_skip_reasons.append(f"inc.py {label}: {reason_text}")
    if not inc_applied:
        return "skipped", (
            "no inc.py anchor matched (cross-pin drift not resolved): "
            + " | ".join(inc_skip_reasons)
        )

    # Apply wNa16 + w4a8_fp8 (no cross-pin alternation; anchors stable).
    for label, patcher in (
        ("compressed_tensors_wNa16.py", wna16),
        ("compressed_tensors_w4a8_fp8.py", w4a8_fp8),
    ):
        content = file_contents[label]
        if patcher.marker in content:
            continue  # already applied (exact marker)
        # Drift check (upstream-form anchor)
        for drift_marker in patcher.upstream_drift_markers:
            if drift_marker.startswith("[Genesis"):
                continue
            if drift_marker in content:
                return (
                    "skipped",
                    f"upstream drift in {label}: {drift_marker!r} "
                    "present without our marker — upstream may have merged "
                    "equivalent fix",
                )
        result, failure = patcher.apply()
        if result == TextPatchResult.SKIPPED:
            _r = (
                failure.reason
                if failure
                else "anchor drift / not eligible"
            )
            _d = (
                f" ({failure.detail})"
                if (failure and failure.detail)
                else ""
            )
            return "skipped", f"{patcher.patch_name}: {_r}{_d}"
        if result == TextPatchResult.FAILED:
            return "failed", (
                f"{patcher.patch_name}: "
                f"{failure.reason if failure else 'unknown'} "
                f"({failure.detail if failure else ''})"
            )

    return (
        "applied",
        "P91B applied (3 files): inc.py + compressed_tensors_wNa16.py + "
        "compressed_tensors_w4a8_fp8.py use cdiv() for scale-row "
        "allocation in row-group quant schemes. Companion to P91 (which "
        "covers the GPTQ-Marlin path + parameter.py loader). Same bug "
        "class, different files; vllm#39460 did not touch these."
    )


def is_applied() -> bool:
    """Return True iff all 3 P91B target files contain a Genesis P91B
    marker (any version).

    Uses the version-agnostic GENESIS_P91B_MARKER_BASE so future minor
    bumps are recognized as already-applied.
    """
    if vllm_install_root() is None:
        return False
    inc338 = _make_inc_dev338_patcher()
    inc371 = _make_inc_dev371_patcher()
    inc_path = (
        inc338.target_file if inc338 is not None
        else (inc371.target_file if inc371 is not None else None)
    )
    wna16 = _make_wna16_patcher()
    w4a8_fp8 = _make_w4a8_fp8_patcher()
    if inc_path is None or wna16 is None or w4a8_fp8 is None:
        return False
    try:
        for path in (inc_path, wna16.target_file, w4a8_fp8.target_file):
            with open(path) as f:
                if GENESIS_P91B_MARKER_BASE not in f.read():
                    return False
    except Exception:
        return False
    return True


def revert() -> tuple[str, str]:
    """Revert is text-patch driven — caller should re-image the container.
    For safety we don't attempt in-place restore (the original lines are
    not preserved verbatim once replaced)."""
    return (
        "skipped",
        "P91B text-patch revert not supported in-place; redeploy a fresh "
        "container or `git checkout` the affected vLLM files",
    )
