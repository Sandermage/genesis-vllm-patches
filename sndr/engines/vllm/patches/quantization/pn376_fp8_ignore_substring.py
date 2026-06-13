# SPDX-License-Identifier: Apache-2.0
"""PN376 — FP8 modules_to_not_convert substring match (vendor of vllm#44628).

Upstream bug (vllm#21669): ``Fp8Config.get_quant_method`` calls
``is_layer_skipped`` with the default ``skip_with_substr=False``, which
performs exact ``prefix in ignored_layers`` matching. HuggingFace-style
short ``modules_to_not_convert`` patterns (e.g.
``"linear_attn.in_proj_qkv"`` as exported by llm-compressor / AutoFP8)
never match the fully-qualified runtime prefix
(``"language_model.model.layers.0.linear_attn.in_proj_qkv"``), so the
checkpoint-excluded layer silently loads as FP8 despite lacking its
``weight_scale`` — gibberish output, no exception, no log. The AWQ
family fixed this class with an opt-in substring match (#26909 AWQ,
#27416 AWQ-Marlin, #29774 IPEX-AWQ); #44628 (OPEN 2026-06-11) opts the
FP8 family into the same behaviour. Substring match is a superset of
exact match, so full-path ignore lists keep working.

Genesis impact: the roadmap chunk-5 Theme D study flags Qwen3.6-VL FP8
as broken TODAY on our pin by this exact class, and the fix unblocks
selective-FP8 GDN re-quant experiments for the 35B PROD family.

What is patched (pin 0.22.1rc1.dev259+g303916e93, every anchor
byte-verified count==1 against /private/tmp/candidate_pin_current on
2026-06-11)
------------------------------------------------------------------
CORE (atomic via MultiFilePatchTransaction — all-or-nothing):
  * ``fp8.py`` — BOTH ``is_layer_skipped`` call sites gain
    ``skip_with_substr=True``: the LinearBase site AND the
    RoutedExperts site (the pin has one more call site than #44628's
    base — the PR form is adapted per iron rule #10, not copied).
  * ``utils/quant_utils.py`` — the experts branch loses its
    ``and not skip_with_substr`` gate and matches BOTH containment
    directions (``prefix in layer_name`` keeps the legacy MoE
    parent-in-child convention; ``layer_name in prefix`` honours
    HF-style short patterns under substring mode). Exact-mode
    behaviour is bit-identical to pristine (the added direction is
    gated on ``skip_with_substr``).

PARITY (best-effort, soft-skip on drift — we run none of these
quantization configs; vendored for upstream-parity per the PR diff):
  * ``fbgemm_fp8.py`` — one-liner ``skip_with_substr=True``.
  * ``mxfp4.py`` — one-liner ``skip_with_substr=True``.
  * ``modelopt.py`` — ``is_layer_excluded``'s first check switches to
    substring matching (its own manual substring fallback loop below
    stays untouched).

Atomicity rationale: enabling substring match at the fp8.py call sites
WITHOUT the quant_utils experts-branch fix silently drops the
parent-in-child containment for MoE prefixes (our 35B PROD is
GDN+MoE); landing quant_utils alone changes experts-branch routing for
the AWQ substr callers without delivering the FP8 fix. Hence the
two-phase transaction for the core pair.

Shared-helper blast radius (AWQ): ``awq.py`` (1 call site) and
``awq_marlin.py`` (2 call sites) already pass ``skip_with_substr=True``
on the pin, so the quant_utils experts-branch change is visible to the
Gemma-4 AWQ MoE PROD model when PN376 is ON: (a) experts-prefix lookups
regain parent-in-child containment (a FIX upstream's review demanded);
(b) experts-containing prefixes now consult ONLY experts-containing
ignore entries instead of falling through to the generic substring
else-branch (upstream-accepted narrowing; pinned by a dedicated test).

Validation harness (REQUIRED before any default_on)
----------------------------------------------------
Per-layer quant-scheme log diff on 35B PROD (and on the two Gemma-4
AWQ models — the quant_utils experts branch is shared with the AWQ
substr callers):

  1. Boot the model with PN376 OFF, then ON (one change at a time),
     same pin, same YAML. Capture the per-layer scheme each time::

       docker logs <container> 2>&1 | grep -iE \
         'Unquantized(Linear|FusedMoE)Method|Fp8(Linear|MoE)Method' \
         > /tmp/pn376_{off,on}.layers
       # Supplement (boot logs are quant-method sparse at INFO):
       # VLLM_LOGGING_LEVEL=DEBUG for the capture boots, OR replay
       # checkpoint config offline — exec is_layer_skipped over the
       # model's module names with ignored_layers from the
       # checkpoint's quantization_config (modules_to_not_convert /
       # ignored_layers keys), both modes, and diff the verdicts.

  2. ``diff /tmp/pn376_off.layers /tmp/pn376_on.layers``

  3. EMPTY diff → the checkpoint's ignore list is full-path (or
     empty); PN376 is a no-op for that model; default_on is SAFE for
     it. NON-EMPTY diff → the pristine pin was silently fp8-quantizing
     layers the exporter excluded (#21669 live); every newly-skipped
     layer must be cross-checked against the checkpoint's
     modules_to_not_convert AND a quality bench (canonical
     genesis_bench_suite --quick + tool-call suite, n>=3) must pass
     BEFORE default_on.

Drift / retire triggers
-----------------------
  * quant_utils patcher: drift marker "also honour the substring
    direction" — a byte-substring of #44628's post-image comment
    (gh pr diff 44628, fetched 2026-06-11) and NOT a substring of
    anything PN376 writes (tools/lint_drift_markers.py contract; we
    reworded our in-file comment specifically so the upstream wording
    stays usable as a marker).
  * modelopt patcher: drift marker "Aligns with AWQ family" — same
    contract against #44628's modelopt post-image.
  * fp8/fbgemm/mxfp4 patchers: the merged form necessarily contains
    ``skip_with_substr=True`` — exactly what PN376 emits — so it
    CANNOT serve as a non-defended drift marker; these patchers carry
    only the defended "[Genesis PN376" sentinel and self-skip via
    anchor drift instead (the pristine call without the kwarg stops
    matching once upstream inserts it) — equally safe, less specific
    reason (PN373 precedent).
  * tests/unit/integrations/quantization/
    test_pn376_fp8_ignore_substring.py: TestPristineBugReproduction
    flips to FAILED when an upstream fix lands in a pin — deep-diff
    and retire per iron rule #11.

Gates
-----
  * ``GENESIS_ENABLE_PN376_FP8_IGNORE_SUBSTRING`` — install gate,
    default OFF (opt-in pending the per-layer quant-scheme diff above).
    The dispatcher consults ``should_apply("PN376")`` against the
    registry BEFORE importing this module (data-driven dispatch), so
    ``apply()`` does not duplicate the gate (PN367/PN373 convention).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor target: vllm-project/vllm#44628 (fixes vllm#21669; OPEN
2026-06-11).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
    TextPatchResult,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn376_fp8_ignore_substring")

GENESIS_PN376_MARKER = (
    "Genesis PN376 fp8 modules_to_not_convert substring match "
    "(vendor of vllm#44628) v1"
)

_QUANT_DIR = "model_executor/layers/quantization"

_FP8_REL = f"{_QUANT_DIR}/fp8.py"
_QUANT_UTILS_REL = f"{_QUANT_DIR}/utils/quant_utils.py"
_FBGEMM_REL = f"{_QUANT_DIR}/fbgemm_fp8.py"
_MXFP4_REL = f"{_QUANT_DIR}/mxfp4.py"
_MODELOPT_REL = f"{_QUANT_DIR}/modelopt.py"

# Substrings of #44628's post-image (gh pr diff 44628, 2026-06-11).
# Each is absent from the pristine pin file AND from every replacement
# PN376 emits (TestDriftMarkerHygiene + tools/lint_drift_markers.py).
_QUANT_UTILS_DRIFT_MARKERS = (
    "also honour the substring direction",
)
_MODELOPT_DRIFT_MARKERS = (
    "Aligns with AWQ family",
)
# Defended convention (lint-exempt) — belt for the patchers whose merged
# form collides with our own emitted text (see module docstring).
_DEFENDED_DRIFT_MARKERS = (
    "[Genesis PN376",
)


# ─── fp8.py — BOTH call sites (LinearBase + RoutedExperts) ────────────
# The two pristine call blocks are byte-identical, so each anchor
# carries its distinguishing isinstance line + return line.

FP8_LINEAR_OLD = (
    "        if isinstance(layer, LinearBase):\n"
    "            if is_layer_skipped(\n"
    "                prefix=prefix,\n"
    "                ignored_layers=self.ignored_layers,\n"
    "                fused_mapping=self.packed_modules_mapping,\n"
    "            ):\n"
    "                return UnquantizedLinearMethod()\n"
    "            if not self.is_checkpoint_fp8_serialized:\n"
)

FP8_LINEAR_NEW = (
    "        if isinstance(layer, LinearBase):\n"
    "            # [Genesis PN376 vendor of vllm#44628] match HF-style short\n"
    "            # modules_to_not_convert patterns by substring (AWQ-family\n"
    "            # parity, #26909/#27416). Exact match silently fp8-quantized\n"
    "            # ignored layers whose checkpoint pattern is not the fully\n"
    "            # qualified runtime prefix (issue #21669 gibberish class).\n"
    "            if is_layer_skipped(\n"
    "                prefix=prefix,\n"
    "                ignored_layers=self.ignored_layers,\n"
    "                fused_mapping=self.packed_modules_mapping,\n"
    "                skip_with_substr=True,\n"
    "            ):\n"
    "                return UnquantizedLinearMethod()\n"
    "            if not self.is_checkpoint_fp8_serialized:\n"
)

FP8_MOE_OLD = (
    "        elif isinstance(layer, RoutedExperts):\n"
    "            if is_layer_skipped(\n"
    "                prefix=prefix,\n"
    "                ignored_layers=self.ignored_layers,\n"
    "                fused_mapping=self.packed_modules_mapping,\n"
    "            ):\n"
    "                return UnquantizedFusedMoEMethod(layer.moe_config)\n"
)

FP8_MOE_NEW = (
    "        elif isinstance(layer, RoutedExperts):\n"
    "            # [Genesis PN376 vendor of vllm#44628] substring match for\n"
    "            # MoE expert ignore patterns (second call site of the pin —\n"
    "            # parent-in-child containment handled in quant_utils).\n"
    "            if is_layer_skipped(\n"
    "                prefix=prefix,\n"
    "                ignored_layers=self.ignored_layers,\n"
    "                fused_mapping=self.packed_modules_mapping,\n"
    "                skip_with_substr=True,\n"
    "            ):\n"
    "                return UnquantizedFusedMoEMethod(layer.moe_config)\n"
)


# ─── utils/quant_utils.py — the experts branch ────────────────────────
# Our comment is deliberately worded DIFFERENTLY from #44628's so the
# upstream comment stays usable as a drift marker (PN373 precedent).

QUANT_UTILS_EXPERTS_OLD = (
    '    elif "experts" in prefix and not skip_with_substr:\n'
    "        expert_ignore_layers = filter(\n"
    '            lambda layer_name: "experts" in layer_name, ignored_layers\n'
    "        )\n"
    "        return any(\n"
    "            prefix in layer_name if not skip_with_substr else "
    "layer_name in prefix\n"
    "            for layer_name in expert_ignore_layers\n"
    "        )\n"
)

QUANT_UTILS_EXPERTS_NEW = (
    '    elif "experts" in prefix:\n'
    "        # [Genesis PN376 vendor of vllm#44628] keep the legacy MoE\n"
    "        # convention in BOTH match modes: a child expert listed in\n"
    "        # ignored_layers (model.layers.0.mlp.experts.0.w1) skips its\n"
    "        # parent RoutedExperts prefix (model.layers.0.mlp.experts) via\n"
    "        # parent-in-child containment; under skip_with_substr the\n"
    "        # reverse direction (pattern-in-prefix) additionally matches\n"
    "        # HF-style short patterns. The pristine pin gated this branch\n"
    "        # on `not skip_with_substr`, losing containment for substring\n"
    "        # callers (AWQ family).\n"
    "        expert_ignore_layers = [\n"
    '            layer_name for layer_name in ignored_layers if '
    '"experts" in layer_name\n'
    "        ]\n"
    "        return any(\n"
    "            prefix in layer_name or (skip_with_substr and "
    "layer_name in prefix)\n"
    "            for layer_name in expert_ignore_layers\n"
    "        )\n"
)


# ─── fbgemm_fp8.py — parity one-liner ─────────────────────────────────

FBGEMM_OLD = (
    "            if is_layer_skipped(\n"
    "                prefix=prefix,\n"
    "                ignored_layers=self.ignore_list,\n"
    "                fused_mapping=self.packed_modules_mapping,\n"
    "            ):\n"
    "                return UnquantizedLinearMethod()\n"
)

FBGEMM_NEW = (
    "            if is_layer_skipped(\n"
    "                prefix=prefix,\n"
    "                ignored_layers=self.ignore_list,\n"
    "                fused_mapping=self.packed_modules_mapping,\n"
    "                # [Genesis PN376 vendor of vllm#44628] AWQ-family parity\n"
    "                skip_with_substr=True,\n"
    "            ):\n"
    "                return UnquantizedLinearMethod()\n"
)


# ─── mxfp4.py — parity one-liner ──────────────────────────────────────

MXFP4_OLD = (
    "            if self.ignored_layers and is_layer_skipped(\n"
    "                prefix=prefix,\n"
    "                ignored_layers=self.ignored_layers,\n"
    "                fused_mapping=self.packed_modules_mapping,\n"
    "            ):\n"
    "                return UnquantizedLinearMethod()\n"
)

MXFP4_NEW = (
    "            if self.ignored_layers and is_layer_skipped(\n"
    "                prefix=prefix,\n"
    "                ignored_layers=self.ignored_layers,\n"
    "                fused_mapping=self.packed_modules_mapping,\n"
    "                # [Genesis PN376 vendor of vllm#44628] AWQ-family parity\n"
    "                skip_with_substr=True,\n"
    "            ):\n"
    "                return UnquantizedLinearMethod()\n"
)


# ─── modelopt.py — parity (is_layer_excluded first check) ─────────────
# The manual substring fallback loop below this check stays untouched.

MODELOPT_OLD = (
    "        # First check exact matching with fused layer support\n"
    "        if is_layer_skipped(prefix, self.exclude_modules, "
    "self.packed_modules_mapping):\n"
    "            return True\n"
)

MODELOPT_NEW = (
    "        # [Genesis PN376 vendor of vllm#44628] substring match with\n"
    "        # fused layer support so HF-style short exclude patterns hit\n"
    "        # the fully-qualified runtime prefix (AWQ-family parity).\n"
    "        if is_layer_skipped(\n"
    "            prefix,\n"
    "            self.exclude_modules,\n"
    "            self.packed_modules_mapping,\n"
    "            skip_with_substr=True,\n"
    "        ):\n"
    "            return True\n"
)


# Single source of truth for tests (anchor/replacement pairs per file).
SUB_PATCH_TEXTS: dict[str, list[tuple[str, str]]] = {
    "fp8": [
        (FP8_LINEAR_OLD, FP8_LINEAR_NEW),
        (FP8_MOE_OLD, FP8_MOE_NEW),
    ],
    "quant_utils": [
        (QUANT_UTILS_EXPERTS_OLD, QUANT_UTILS_EXPERTS_NEW),
    ],
    "fbgemm": [
        (FBGEMM_OLD, FBGEMM_NEW),
    ],
    "mxfp4": [
        (MXFP4_OLD, MXFP4_NEW),
    ],
    "modelopt": [
        (MODELOPT_OLD, MODELOPT_NEW),
    ],
}


def _make_quant_utils_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_QUANT_UTILS_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN376 quant_utils.py — experts branch substring-aware "
            "(vendor of vllm#44628)"
        ),
        target_file=str(target),
        marker=GENESIS_PN376_MARKER,
        sub_patches=[
            TextPatch(
                name="pn376_quant_utils_experts_branch",
                anchor=QUANT_UTILS_EXPERTS_OLD,
                replacement=QUANT_UTILS_EXPERTS_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_QUANT_UTILS_DRIFT_MARKERS),
    )


def _make_fp8_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_FP8_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN376 fp8.py — substring opt-in at BOTH is_layer_skipped "
            "call sites (vendor of vllm#44628)"
        ),
        target_file=str(target),
        marker=GENESIS_PN376_MARKER,
        sub_patches=[
            TextPatch(
                name="pn376_fp8_linear_site_substr",
                anchor=FP8_LINEAR_OLD,
                replacement=FP8_LINEAR_NEW,
                required=True,
            ),
            TextPatch(
                name="pn376_fp8_routed_experts_site_substr",
                anchor=FP8_MOE_OLD,
                replacement=FP8_MOE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DEFENDED_DRIFT_MARKERS),
    )


def _make_fbgemm_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_FBGEMM_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN376 fbgemm_fp8.py — substring parity (vllm#44628)",
        target_file=str(target),
        marker=GENESIS_PN376_MARKER,
        sub_patches=[
            TextPatch(
                name="pn376_fbgemm_substr",
                anchor=FBGEMM_OLD,
                replacement=FBGEMM_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DEFENDED_DRIFT_MARKERS),
    )


def _make_mxfp4_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_MXFP4_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN376 mxfp4.py — substring parity (vllm#44628)",
        target_file=str(target),
        marker=GENESIS_PN376_MARKER,
        sub_patches=[
            TextPatch(
                name="pn376_mxfp4_substr",
                anchor=MXFP4_OLD,
                replacement=MXFP4_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DEFENDED_DRIFT_MARKERS),
    )


def _make_modelopt_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_MODELOPT_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN376 modelopt.py — substring parity (vllm#44628)",
        target_file=str(target),
        marker=GENESIS_PN376_MARKER,
        sub_patches=[
            TextPatch(
                name="pn376_modelopt_substr",
                anchor=MODELOPT_OLD,
                replacement=MODELOPT_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_MODELOPT_DRIFT_MARKERS),
    )


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def apply() -> tuple[str, str]:
    """Install the substring-match opt-in. Never raises.

    Env gating (GENESIS_ENABLE_PN376_FP8_IGNORE_SUBSTRING) is enforced
    by the dispatcher via the registry entry before this module is even
    imported.

    Order: CORE pair (quant_utils + fp8) committed atomically via
    MultiFilePatchTransaction first; parity one-liners (fbgemm / mxfp4
    / modelopt) best-effort after — their drift never withholds the
    core FP8 fix.
    """
    qu = _make_quant_utils_patcher()
    if qu is None:
        return "skipped", f"PN376: {_QUANT_UTILS_REL} not resolvable"
    fp8 = _make_fp8_patcher()
    if fp8 is None:
        return "skipped", f"PN376: {_FP8_REL} not resolvable"

    qu_src = _read(qu.target_file)
    if qu_src is None:
        return "skipped", f"PN376: cannot read {qu.target_file}"
    fp8_src = _read(fp8.target_file)
    if fp8_src is None:
        return "skipped", f"PN376: cannot read {fp8.target_file}"

    # Explicit upstream-merged pre-check: the transaction's dry-run
    # would only report a less specific anchor miss (PN371 precedent).
    if GENESIS_PN376_MARKER not in qu_src and any(
        m in qu_src for m in _QUANT_UTILS_DRIFT_MARKERS
    ):
        return "skipped", (
            "PN376: upstream_merged — #44628's substring-aware experts "
            "branch detected in quant_utils.py; native substring match "
            "active, Genesis vendoring obsolete (deep-diff + retire per "
            "iron rule #11)"
        )

    core_was_applied = (
        GENESIS_PN376_MARKER in qu_src and GENESIS_PN376_MARKER in fp8_src
    )

    txn = MultiFilePatchTransaction(
        [qu, fp8], name="PN376 core (quant_utils + fp8)"
    )
    status, reason = txn.apply_or_skip()
    if status == "failed":
        return "failed", f"PN376: {reason}"
    if status == "skipped":
        return "skipped", (
            f"PN376 core withheld atomically (substr opt-in without the "
            f"experts-branch fix — or vice versa — is the partial state "
            f"the transaction prevents): {reason}"
        )

    # Parity one-liners — best-effort; drift logs loudly but never
    # withholds the core fix (we run none of these quant configs).
    parity_notes: list[str] = []
    parity_all_idempotent = True
    for maker, label in (
        (_make_fbgemm_patcher, "fbgemm_fp8"),
        (_make_mxfp4_patcher, "mxfp4"),
        (_make_modelopt_patcher, "modelopt"),
    ):
        patcher = maker()
        if patcher is None:
            parity_notes.append(f"{label}: target missing (soft-skip)")
            continue
        try:
            p_result, p_failure = patcher.apply()
        except Exception as e:  # noqa: BLE001 — parity must not abort core
            log.warning("[PN376] parity patcher %s raised: %r", label, e)
            parity_notes.append(f"{label}: raised {type(e).__name__} (soft)")
            parity_all_idempotent = False
            continue
        _, p_reason = result_to_wiring_status(
            p_result, p_failure,
            applied_message=f"{label} substring parity applied",
            patch_name=f"PN376 {label}",
        )
        if p_result == TextPatchResult.FAILED:
            log.warning("[PN376] parity patcher %s FAILED: %s", label, p_reason)
            parity_notes.append(f"{label}: FAILED ({p_reason}) — core intact")
            parity_all_idempotent = False
        elif p_result == TextPatchResult.SKIPPED:
            log.warning("[PN376] parity patcher %s skipped: %s", label, p_reason)
            parity_notes.append(f"{label}: skipped ({p_reason})")
            parity_all_idempotent = False
        else:
            if p_result != TextPatchResult.IDEMPOTENT:
                parity_all_idempotent = False
            parity_notes.append(f"{label}: {p_reason}")

    detail = " | ".join(parity_notes)

    if core_was_applied and parity_all_idempotent:
        return "skipped", "PN376: already applied (markers present)"

    return "applied", (
        "PN376 applied: FP8 family now matches modules_to_not_convert by "
        "substring — fp8.py BOTH call sites (LinearBase + RoutedExperts) "
        "opt in via skip_with_substr=True and the quant_utils experts "
        "branch keeps parent-in-child containment in substring mode "
        "(vendor of vllm#44628, fixes #21669 silent-gibberish class; "
        "AWQ-family parity #26909/#27416). VALIDATION GATE before any "
        "default_on: per-layer quant-scheme log diff on 35B PROD "
        f"(see module docstring). parity: {detail}"
    )


def is_applied() -> bool:
    """Filesystem-level marker check — True iff BOTH core targets carry
    the PN376 marker. Cheap; used by audit / shadow CLI."""
    for rel in (_QUANT_UTILS_REL, _FP8_REL):
        target = resolve_vllm_file(rel)
        if target is None:
            return False
        content = _read(str(target))
        if content is None or GENESIS_PN376_MARKER not in content:
            return False
    return True
