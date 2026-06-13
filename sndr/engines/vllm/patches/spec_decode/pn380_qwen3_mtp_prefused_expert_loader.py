# SPDX-License-Identifier: Apache-2.0
"""PN380 — vendor of OPEN PR vllm#44943 (Qwen3.5/3.6 MTP pre-fused
expert loader) + Genesis-original draft-weight load-coverage guard.

Deep-dive understanding of the upstream PR
==========================================

**Problem**: ``Qwen3_5MultiTokenPredictor.load_weights()`` has
asymmetric expert-weight handling. ``FusedMoE`` registers its
parameters internally as ``experts.w13_weight`` / ``experts.w2_weight``
(see ``fused_expert_params_mapping`` in the same function), but the
loader only recognizes ``experts.gate_up_proj`` / ``experts.down_proj``
as SOURCE names from the checkpoint. A checkpoint that already stores
the expert tensors under the fused-form names — community AutoRound /
GPTQ INT4 quants of Qwen3.5/3.6 MoE, and manually-renamed checkpoints
working around vllm#36954 — is not detected as a fused-expert weight
and falls through to the fallback branch.

**Failure modes** (verified against the upstream PR description and the
pristine pin source, lines 222-335):

  * MTP quantized (``mtp.layers`` in
    ``quantization_config.block_name_to_quantize``): every expert
    tensor emits ``WARNING ... not found in params_dict, skip loading``
    once, the MTP draft boots with randomly-initialized expert weights,
    and the spec-decode accept rate silently collapses (upstream A/B:
    65.0% -> 41.9%, i.e. -23pp; mean accept length 2.30 -> 1.84 at
    K=3). At MTP K=3 that is roughly a -15-20% decode-TPS hit.
  * MTP unquantized: ``TypeError: FusedMoE.weight_loader() missing 3
    required positional arguments`` — engine crash at startup.

**Why our PROD cares while being unaffected TODAY**: both PROD SKUs
(cyankiwi/Qwen3.6-35B-A3B-FP8, Lorbus/Qwen3.6-27B-int4-AutoRound) use
split-form expert names, verified unaffected in the 50-PR sweep. The
vendor is INSURANCE: it is the prerequisite for the planned INT4
35B-A3B trial (community AutoRound quants of that SKU emit pre-fused
tensors), and it converts a silent-accept-collapse class into a working
load path. Roadmap: chunk-4 Theme 2 (journal
2026-06-11-pr-sweep-50-roadmap.md), "loud startup" validation family
together with the #44837 prefix= AST lint
(tests/unit/lint/test_quantized_linear_prefix.py).

Adaptation per iron rule #10 (adapt, don't blind-copy)
======================================================

The PR head targets a NEWER upstream shape where
``fused_expert_params_mapping`` is built in a loop and the fix appends
an ``alt_ckpt_name`` variant per entry. Our pin
(0.22.1rc1.dev259+g303916e93) carries the older STATIC two-entry list,
so sub-patch 1 appends two static pre-fused entries instead — same
semantics, pin-native shape. Because of that, ``alt_ckpt_name`` is a
safe upstream drift marker: it appears in the file only if the merged
form of vllm#44943 (or its loop-built successor) lands, and never in
our own emitted text (asserted in tests; tools/lint_drift_markers.py
contract).

Three vendored sub-patches (faithful #44943 semantics):

  * Sub-1 (mapping): add ``experts.w13_weight`` / ``experts.w2_weight``
    as alternative checkpoint SOURCE names mapping to the same
    base_layer-aware target params.
  * Sub-2 (detection): extend the fused-expert detection condition so
    pre-fused names route through the fused load path instead of the
    params_dict-miss fallback.
  * Sub-3 (params_dict guard): for (quantized MTP + pre-fused ckpt),
    ``params_dict`` holds ``w13_qweight`` rather than ``w13_weight`` —
    the unguarded lookup inside ``load_fused_expert_weights`` would
    raise ``KeyError``. Skip the fused path AND reset
    ``is_expert_weight = False`` so the outer fallback emits the
    standard params_dict-miss warning (which the coverage guard below
    counts). Also extend the chunk(2) branch to cover
    ``experts.w13_weight`` (pre-fused gate+up come fused on dim -2,
    exactly like ``gate_up_proj``).

Genesis-original extras (NOT in the upstream PR)
================================================

Draft-weight load-coverage guard, P29-style loud-failure conversion
(three more sub-patches on the same function):

  * The engine's strict checkpoint-coverage check
    (``DefaultModelLoader.track_weights_loading``) is gated on
    ``model_config.quantization is None`` — it NEVER runs for our
    FP8/INT4 PROD models, so a partial draft load is silent by design
    upstream.
  * Sub-4 (state): track checkpoint tensors that found no matching
    param (``_pn380_skipped_ckpt``).
  * Sub-5 (count): record each name reaching the params_dict-miss
    fallback, keeping the upstream ``warning_once`` verbatim.
  * Sub-6 (report): after the load loop, ALSO compute expected params
    that received no checkpoint weight (params owned by a module with a
    ``quant_method`` are exempt, mirroring upstream
    ``track_weights_loading`` — quant methods may materialize/rewrite
    params post-load, e.g. attention k/v scales, online quant). On any
    gap emit ONE ``logger.error`` with counts + samples. Never raises —
    the guard makes the failure LOUD, it does not change behavior.

Accept-rate floor companion: ``sndr/extras/tools/genesis_bench_suite.py``
gains a spec-decode accept-rate floor check (WARN below 0.55 on the
bench window) so a partial draft load that slips past boot is caught at
bench time. Same roadmap line item.

Composition + safety
====================

  * **PN348 (vendor of vllm#44644) patches the SAME FILE** — its three
    anchors (embed_tokens predicate, lm_head fallthrough,
    remap_weight_names skip) all live OUTSIDE
    ``Qwen3_5MultiTokenPredictor.load_weights``; PN380's six anchors
    all live INSIDE it. Disjointness and both co-apply orders are
    asserted in
    tests/unit/integrations/spec_decode/test_pn380_qwen3_mtp_prefused_expert_loader.py.
    Cross-module drift-marker hygiene is also asserted: PN380's
    replacements contain neither ``share_backbone_input_output`` nor
    ``[Genesis PN348`` (and PN348's contain no ``alt_ckpt_name``), so
    neither patch can Layer-3 false-skip the other.
  * Composes with PN108 + PN133 + PN290 + PN340 + PN341 + PN370 (MTP
    runtime patches in different files; no anchor overlap).
  * All six sub-patches ``required=True`` — partial application would
    count skips without reporting them (or detect pre-fused names
    without the KeyError guard). On any anchor miss the patcher SKIPs
    cleanly with no file mutation (TextPatcher semantics; PN286/PN290
    half-apply lesson).
  * Behavior on split-form checkpoints (both PROD SKUs today): the new
    mapping entries and detection branches never match, the coverage
    guard sees zero skips and full coverage — patched file is
    behavior-identical to pristine, plus one set() comparison per load
    (load-time only, zero hot-path cost).

Anchors byte-verified count==1 on the pristine tree at
/private/tmp/candidate_pin_current/vllm (pin g303916e93, 2026-06-11).

Runtime verification (post-restart, pre-fused trial checkpoint)
===============================================================

  docker logs <container> 2>&1 | grep -E "Genesis PN380|not found in params_dict"

Expected on a healthy pre-fused load: NO "[Genesis PN380] ... gap"
error and NO params_dict-miss warnings for ``mtp.*`` expert tensors;
accept rate at bench >= floor (genesis_bench_suite.py WARN line).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Upstream: https://github.com/vllm-project/vllm/pull/44943 (OPEN at
vendor time, 2026-06-11).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn380_qwen3_mtp_prefused_expert_loader")

GENESIS_PN380_MARKER = (
    "Genesis PN380 vendor of vllm#44943 (Qwen3.5/3.6 MTP pre-fused expert "
    "loader + load-coverage guard) v1"
)

_TARGET_REL = "model_executor/models/qwen3_5_mtp.py"

# Fires when vllm#44943 merges. The PR-head fix introduces the
# `alt_ckpt_name` variable; our pin-native adaptation emits static
# mapping entries instead, so the marker never matches our own output
# (tools/lint_drift_markers.py contract; asserted in tests).
_DRIFT_MARKERS = (
    "[Genesis PN380",
    "alt_ckpt_name",
)


# ── Sub-1: pre-fused checkpoint source names in the fused mapping ───────
# Anchor: the static two-entry list (pin g303916e93 lines 215-218).
# The PR head builds this list in a loop — adaptation note in the
# module docstring.
PN380_MAPPING_OLD = (
    "        fused_expert_params_mapping = [\n"
    '            (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 0, "w1"),\n'
    '            (f"experts.{base_layer}w2_weight", "experts.down_proj", 0, "w2"),\n'
    "        ]\n"
)
PN380_MAPPING_NEW = (
    "        fused_expert_params_mapping = [\n"
    '            (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 0, "w1"),\n'
    '            (f"experts.{base_layer}w2_weight", "experts.down_proj", 0, "w2"),\n'
    "            # [Genesis PN380 vendor of vllm#44943] pre-fused checkpoints\n"
    "            # (community AutoRound/GPTQ quants of Qwen3.5/3.6 MoE) store\n"
    "            # expert tensors under the fused names directly — same\n"
    "            # target param, alternative checkpoint source name.\n"
    '            (f"experts.{base_layer}w13_weight", "experts.w13_weight", 0, "w1"),\n'
    '            (f"experts.{base_layer}w2_weight", "experts.w2_weight", 0, "w2"),\n'
    "        ]\n"
)


# ── Sub-2: detect pre-fused names as fused-expert weights ───────────────
# Anchor: the detection branch heading the stacked-mapping loop (pin
# lines 226-229). Includes the for-line for uniqueness.
PN380_DETECT_OLD = (
    "            for param_name, weight_name, shard_id in stacked_params_mapping:\n"
    '                if "experts.gate_up_proj" in name or "experts.down_proj" in name:\n'
    "                    is_fused_expert = True\n"
    "                    expert_params_mapping = fused_expert_params_mapping\n"
)
PN380_DETECT_NEW = (
    "            for param_name, weight_name, shard_id in stacked_params_mapping:\n"
    "                # [Genesis PN380 vendor of vllm#44943] also detect\n"
    "                # pre-fused checkpoint names (w13_weight / w2_weight) as\n"
    "                # fused-expert weights so they route through the fused\n"
    "                # load path instead of the params_dict-miss fallback.\n"
    "                if (\n"
    '                    "experts.gate_up_proj" in name\n'
    '                    or "experts.down_proj" in name\n'
    '                    or "experts.w13_weight" in name\n'
    '                    or "experts.w2_weight" in name\n'
    "                ):\n"
    "                    is_fused_expert = True\n"
    "                    expert_params_mapping = fused_expert_params_mapping\n"
)


# ── Sub-3: params_dict guard + pre-fused chunk branch ───────────────────
# Anchor: head of the fused-expert load path (pin lines 261-264).
PN380_FUSED_GUARD_OLD = (
    "                    if is_fused_expert:\n"
    "                        # qwen3.5 no need to transpose\n"
    "                        # loaded_weight = loaded_weight.transpose(-1, -2)\n"
    '                        if "experts.gate_up_proj" in name:\n'
)
PN380_FUSED_GUARD_NEW = (
    "                    if is_fused_expert:\n"
    "                        # qwen3.5 no need to transpose\n"
    "                        # loaded_weight = loaded_weight.transpose(-1, -2)\n"
    "                        # [Genesis PN380 vendor of vllm#44943] guard for\n"
    "                        # (quantized MTP + pre-fused checkpoint):\n"
    "                        # params_dict holds w13_qweight rather than\n"
    "                        # w13_weight, so the lookup inside\n"
    "                        # load_fused_expert_weights would raise KeyError.\n"
    "                        # Reset is_expert_weight so the outer fallback\n"
    "                        # emits the standard params_dict-miss warning\n"
    "                        # (counted by the PN380 coverage guard).\n"
    "                        if name_mapped not in params_dict:\n"
    "                            is_expert_weight = False\n"
    "                            continue\n"
    "                        if (\n"
    '                            "experts.gate_up_proj" in name\n'
    '                            or "experts.w13_weight" in name\n'
    "                        ):\n"
)


# ── Sub-4 (Genesis-original): coverage-guard state ──────────────────────
# Anchor: the loader-state preamble (pin lines 209-211).
PN380_COVERAGE_INIT_OLD = (
    "        params_dict = dict(self.named_parameters())\n"
    "        loaded_params: set[str] = set()\n"
    "        is_fused_expert = False\n"
)
PN380_COVERAGE_INIT_NEW = (
    "        params_dict = dict(self.named_parameters())\n"
    "        loaded_params: set[str] = set()\n"
    "        # [Genesis PN380] draft-weight load-coverage guard state:\n"
    "        # checkpoint tensors that found no matching param.\n"
    "        _pn380_skipped_ckpt: list[str] = []\n"
    "        is_fused_expert = False\n"
)


# ── Sub-5 (Genesis-original): count params_dict-miss fallbacks ──────────
# Anchor: the existing warning_once fallback (pin lines 324-328) — the
# exact symptom line of the #44943 quantized-MTP failure mode. The
# upstream warning is preserved verbatim.
PN380_COVERAGE_SKIP_OLD = (
    "                    if name not in params_dict:\n"
    "                        logger.warning_once(\n"
    '                            f"Parameter {name} not found in params_dict, skip loading"\n'
    "                        )\n"
    "                        continue\n"
)
PN380_COVERAGE_SKIP_NEW = (
    "                    if name not in params_dict:\n"
    "                        # [Genesis PN380] record for the load-coverage\n"
    "                        # report below.\n"
    "                        _pn380_skipped_ckpt.append(name)\n"
    "                        logger.warning_once(\n"
    '                            f"Parameter {name} not found in params_dict, skip loading"\n'
    "                        )\n"
    "                        continue\n"
)


# ── Sub-6 (Genesis-original): one loud report on any coverage gap ───────
# Anchor: tail of load_weights (pin lines 334-335).
PN380_COVERAGE_REPORT_OLD = (
    "            loaded_params.add(name)\n"
    "        return loaded_params\n"
)
PN380_COVERAGE_REPORT_NEW = (
    "            loaded_params.add(name)\n"
    "        # [Genesis PN380] draft-weight load-coverage guard (P29-style\n"
    "        # loud-failure conversion). The engine's strict coverage check\n"
    "        # (DefaultModelLoader.track_weights_loading) is disabled\n"
    "        # whenever model_config.quantization is set, so a quantized\n"
    "        # target's MTP draft that partial-loads boots silently with\n"
    "        # randomly-initialized weights and the spec-decode accept rate\n"
    "        # collapses (~65% -> ~42% in the vllm#44943 repro; roughly\n"
    "        # -15-20% decode TPS at MTP K=3). Convert that silence into\n"
    "        # one loud log.error; never raise, never change behavior.\n"
    "        _pn380_expected = set(params_dict)\n"
    "        for _pn380_mod_name, _pn380_mod in self.named_modules():\n"
    '            if getattr(_pn380_mod, "quant_method", None) is None:\n'
    "                continue\n"
    "            # Mirror DefaultModelLoader.track_weights_loading: params\n"
    "            # owned by a quant method may be materialized or rewritten\n"
    "            # after load (process_weights_after_loading / meta-device\n"
    "            # online quant) — exempt them from the coverage count.\n"
    "            for _pn380_param_name, _ in _pn380_mod.named_parameters():\n"
    "                _pn380_expected.discard(\n"
    '                    f"{_pn380_mod_name}.{_pn380_param_name}"\n'
    "                    if _pn380_mod_name\n"
    "                    else _pn380_param_name\n"
    "                )\n"
    "        _pn380_not_loaded = sorted(_pn380_expected - loaded_params)\n"
    "        if _pn380_skipped_ckpt or _pn380_not_loaded:\n"
    "            logger.error(\n"
    '                "[Genesis PN380] MTP draft-weight load-coverage gap: "\n'
    '                "%d checkpoint tensors found no matching param "\n'
    '                "(sample: %s); %d expected params received no "\n'
    '                "checkpoint weight (sample: %s). Spec-decode accept "\n'
    '                "rate will degrade silently — verify expert tensor "\n'
    '                "naming (vllm#44943) and the quantization ignore "\n'
    '                "list.",\n'
    "                len(_pn380_skipped_ckpt),\n"
    "                _pn380_skipped_ckpt[:4],\n"
    "                len(_pn380_not_loaded),\n"
    "                _pn380_not_loaded[:4],\n"
    "            )\n"
    "        return loaded_params\n"
)


def build_sub_patches() -> list[TextPatch]:
    """All six sub-patches, all ``required=True``.

    Partial application would be worse than none: counting skips
    without the report (or detecting pre-fused names without the
    KeyError guard) ships an inconsistent loader. On any anchor miss
    the whole patcher SKIPs with no file mutation.
    """
    return [
        TextPatch(
            name="pn380_prefused_mapping",
            anchor=PN380_MAPPING_OLD,
            replacement=PN380_MAPPING_NEW,
            required=True,
        ),
        TextPatch(
            name="pn380_prefused_detection",
            anchor=PN380_DETECT_OLD,
            replacement=PN380_DETECT_NEW,
            required=True,
        ),
        TextPatch(
            name="pn380_prefused_params_dict_guard",
            anchor=PN380_FUSED_GUARD_OLD,
            replacement=PN380_FUSED_GUARD_NEW,
            required=True,
        ),
        TextPatch(
            name="pn380_coverage_state",
            anchor=PN380_COVERAGE_INIT_OLD,
            replacement=PN380_COVERAGE_INIT_NEW,
            required=True,
        ),
        TextPatch(
            name="pn380_coverage_count_skips",
            anchor=PN380_COVERAGE_SKIP_OLD,
            replacement=PN380_COVERAGE_SKIP_NEW,
            required=True,
        ),
        TextPatch(
            name="pn380_coverage_report",
            anchor=PN380_COVERAGE_REPORT_OLD,
            replacement=PN380_COVERAGE_REPORT_NEW,
            required=True,
        ),
    ]


def _make_mtp_loader_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN380 models/qwen3_5_mtp.py — pre-fused expert loader + "
            "load-coverage guard (vendor vllm#44943)"
        ),
        target_file=str(target),
        marker=GENESIS_PN380_MARKER,
        sub_patches=build_sub_patches(),
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN380_MTP_PREFUSED_LOADER", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """Apply PN380 — vendor vllm#44943 + coverage guard. Never raises."""
    if not _enabled():
        return "skipped", (
            "PN380 default OFF — set GENESIS_ENABLE_PN380_MTP_PREFUSED_LOADER=1 "
            "to engage. Insurance vendor of OPEN PR vllm#44943 (Qwen3.5/3.6 "
            "MTP pre-fused expert checkpoint loader; prereq for the INT4 "
            "35B-A3B trial) + Genesis draft-weight load-coverage guard."
        )

    patcher = _make_mtp_loader_patcher()
    if patcher is None:
        return "skipped", (
            f"PN380: target file {_TARGET_REL} not found (vllm pin may not "
            "be Qwen3.5/3.6 era). Skipping."
        )

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001 — wiring must never raise
        log.warning("[PN380] apply() raised %s — leaving upstream", e)
        return "skipped", f"PN380 raised at apply: {e!r}"

    if result == TextPatchResult.FAILED:
        return "failed", (
            f"PN380 FAILED — "
            f"{failure.reason if failure else 'unknown anchor mismatch'}"
        )
    if result == TextPatchResult.SKIPPED:
        return "skipped", (
            f"PN380 skipped — {failure.reason if failure else 'unknown'} "
            f"(check for upstream merge of vllm#44943)"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", (
            "PN380 idempotent (already applied). Pre-fused expert loader + "
            "draft-weight load-coverage guard live on qwen3_5_mtp.py."
        )

    n = len(patcher.applied_sub_patches)
    return "applied", (
        f"PN380 applied: {n}/6 sub-patches on qwen3_5_mtp.py — (1-3) vendor "
        f"of OPEN PR vllm#44943: experts.w13_weight/w2_weight pre-fused "
        f"checkpoint names mapped + detected + KeyError-guarded (prevents "
        f"silent -23pp accept-rate collapse on quantized MTP and startup "
        f"TypeError on unquantized MTP for AutoRound/GPTQ community "
        f"checkpoints); (4-6) Genesis draft-weight load-coverage guard — "
        f"one log.error on any checkpoint/param coverage gap (the engine's "
        f"strict check is disabled for quantized models). Split-form "
        f"checkpoints (both PROD SKUs) are behavior-identical. Composes "
        f"with PN348 (same file, disjoint anchors, both orders verified)."
    )


def is_applied() -> bool:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return False
    try:
        return GENESIS_PN380_MARKER in Path(str(target)).read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError):
        return False
