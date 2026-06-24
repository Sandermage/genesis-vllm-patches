#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V2 runtime / ModelDef pin harmonization audit.

After P2.3 unified all hardware YAMLs on a single dev371 runtime image
and a3fa5265 wired the renderer to honor `hw.runtime.docker.image`,
this script locks the invariants in place so the discipline survives
the next pin bump:

  R-PIN-1 — no bare mutable vLLM nightly under
            ``vllm/sndr_core/model_configs/builtin/hardware/*.yaml``.
            Hardware YAMLs must pin an explicit-hash tag (or
            digest-backed tag), never the floating ``:nightly`` /
            ``:latest`` / ``:main`` aliases.

  R-PIN-2 — every hardware YAML that declares ``runtime.docker.image``
            must also declare a non-empty ``runtime.docker.image_digest``
            beginning with ``vllm/vllm-openai@sha256:``. The digest is
            the byte-locked anchor; the tag is for human readability.

  R-PIN-3 — rendered launcher's IMAGE line must equal the composed
            ``cfg.docker.image`` for representative profiles. P2.1
            wired the renderer to read ``hw.runtime.docker.image``
            verbatim, and compose() carries the same value into
            ``cfg.docker.image``. This rule is a regression gate
            against a future refactor that re-introduces a hardcoded
            fallback or a parallel image-resolution path.

  R-PIN-4 — ModelDef pin migration status. Every ModelDef pin must be
            in ``ALLOWED_MODELDEF_PINS`` (a value outside it fails,
            catching stale pins or typos). Reconciled 2026-06-24
            (pin bump dev148 -> dev301): the whole fleet is unified on
            the canonical pin
            (``CANONICAL_PIN_SUBSTRING``, currently dev301). The audit
            classifies each ModelDef into a ``canonical`` /  ``dev338``
            (legacy baseline) / ``dev371`` (legacy sprint era) /
            ``other`` bucket and reports the migration table as info.
            DFlash and placeholder hold semantics now attach to the
            canonical bucket; the dev338 / dev371 buckets are retained
            so a rolled-back tree still classifies correctly.

The script is idempotent and read-only — it never modifies files,
registry, or git state.

Exit codes:

  0 — all selected rules pass (clean tree, allowed migration state).
  1 — at least one violation found.
  2 — audit tooling itself failed (import error, missing dependency).

Usage:

  python3 scripts/audit_v2_runtime_pins.py
  python3 scripts/audit_v2_runtime_pins.py --rule R-PIN-3 --verbose
  python3 scripts/audit_v2_runtime_pins.py --json > /tmp/pin-audit.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Repo layout — resolved relative to this script's location.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure the repo's vllm/ is importable when audit runs from a clean shell.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HARDWARE_DIR = (
    REPO_ROOT / "sndr" / "model_configs" / "builtin" / "hardware"
)
MODEL_DIR = (
    REPO_ROOT / "sndr" / "model_configs" / "builtin" / "model"
)

# ─── Constants ──────────────────────────────────────────────────────────

# Pins the project currently recognizes as legitimate ModelDef values.
# Extend this set when a new pin lands AND a smoke + bench validates it.
ALLOWED_MODELDEF_PINS = frozenset({
    "0.20.2rc1.dev338+gbf0d2dc6d",
    "0.20.2rc1.dev371+gbf610c2f5",
    # K.1.R 2026-05-28 pin bump target — PROMOTED 2026-05-30 (K.1.R.R.8.5):
    # 35B FP8 dense MoE bench 195.74 TPS / 6-7 tool-call on 626fa9bb
    # with P67 enabled; PN286 default_on flip validated +6.6% TPS.
    # Setuptools_scm-derived form (closest annotated ancestor tag
    # v0.21.1rc0 + 12-char SHA).
    "0.21.1rc0+g626fa9bba5",
    # Fleet validation 2026-06-11 pin bump target — PROMOTED: 35B PROD
    # boot 105/0 + chat matrix 250.0/250.0/217.6 TPS + tool-calls verified
    # in BOTH stream and non-stream; 27B full-YAML-stack 109/0, suite
    # 120.9 (+1.2% vs dev371 baseline), tools 7/7; Gemma 26B first boot
    # 0-failed, tools 7/7, TPOT 6.0ms. Receipts: journal
    # 2026-06-11-fleet-validation-on-pin-303916e93.md.
    "0.22.1rc1.dev259+g303916e93",
    "0.21.1rc0+g626fa9bba566",
    # PROD pin PROMOTION dev491 — ratified 2026-06-14 (guards.py:581).
    # Image vllm/vllm-openai:nightly-1033ffac2 (0.22.1rc1.dev491+g1033ffac2).
    # Live-validated 2026-06-15: 35B PROD boots mount-free on the migrated
    # v12 rig (sndr.plugin:register), 112 patches applied / 0 failed, smoke
    # 200; 27B full stack at gmu 0.82 passes all chat-matrix variants.
    "0.22.1rc1.dev491+g1033ffac2",
    "0.22.1rc1.dev491+g1033ffac2d66",
    # ── Canonical PROD pin 0.23.1 (PROMOTED 2026-06-17) ──────────────────
    # Image vllm/vllm-openai:nightly-4c626633159887b0f2c962058c17c78f1434556d
    # (0.23.1rc1.dev101+g4c6266331). Full-fleet validated 2026-06-17: 35B
    # (210.7 TPS = 101% of dev491, MTP K=3, streaming tool-calls), 27B
    # (gen + MTP + qwen3_xml tool-calls), Gemma4-31B, DiffusionGemma — all
    # apply failed=0, smoke + tool-call PASS. See guards.py KNOWN_GOOD +
    # tests/unit/dispatcher/test_pin_gate EXPECTED_PINS.
    "0.23.1rc1.dev101+g4c6266331",
    "0.23.1rc1.dev101+g4c626633159",
    # ── Canonical PROD pin 0.23.1 dev148 (PROMOTED 2026-06-19) ───────────
    # Image vllm/vllm-openai:nightly-b4c80ec0f (0.23.1rc1.dev148+gb4c80ec0f).
    # Live rig pin: the 35B PROD runs it; all MTP K=5 + Gemma kv-auto work
    # validated on it. 35B K=5 239.7 TPS / 27B K=5 127.4 TPS; PN394 (#46047)
    # + PN399 (#46067) promoted to PROD, PN353A (vllm#44053) enabled in Qwen
    # YAMLs. See guards.py KNOWN_GOOD + test_pin_gate EXPECTED_PINS. dev101
    # retained as previous/rollback per CLAUDE.md ≤2-pin policy. The model
    # ModelDefs LEAD the hardware image during this window — the hardware
    # YAML image_digest bumps to the dev148 sha256 once it is captured;
    # until then the two PROD ModelDefs carry a pin_hold waiver (see
    # audit_v2_modeldef_vs_hardware_pin R-MD-HW-2).
    "0.23.1rc1.dev148+gb4c80ec0f",
    # ── Canonical PROD pin 0.23.1 dev301 (PROMOTED 2026-06-24) ───────────
    # Image vllm/vllm-openai:nightly-04c2a8dea (0.23.1rc1.dev301+g04c2a8dea).
    # Pin bump dev148 -> dev301. Smoke: 35B 208 TPS + 31B 94.7 TPS boot+chat+
    # tool-call. The dev301 anchor-SOT regen surfaced 5 anchor_drift
    # (P85/PN394/PN353A/PN400/PN382): PN394+PN400 retired on dev301,
    # PN353A+PN382 kept+re-anchored. See guards.py KNOWN_GOOD + test_pin_gate
    # EXPECTED_PINS. dev148 retained as previous/rollback per CLAUDE.md
    # ≤2-pin policy. The ModelDefs LEAD the hardware image during this
    # window — the hardware YAML image_digest bumps to the dev301 sha256
    # once it is captured; until then the ModelDefs carry a pin_hold waiver
    # (see audit_v2_modeldef_vs_hardware_pin R-MD-HW-2).
    "0.23.1rc1.dev301+g04c2a8dea",
})

# Gemma family ModelDefs are expected to be on dev371 (validated path).
# Qwen family ModelDefs may be on either pin (dev338 baseline OR dev371
# after per-model smoke promotion via P2.4d). Use file-stem prefix to
# classify.
GEMMA_PREFIX = "gemma-"
QWEN_PREFIX = "qwen"

# ─── Canonical pin (fleet-unified) ─────────────────────────────────────
#
# Reconciled 2026-06-24 (pin bump dev148 -> dev301): the P2.4d / P2.DFlash
# / dev371-vs-dev338 sprint is CLOSED. The entire fleet is now unified on a
# single canonical pin (0.23.1rc1.dev301+g04c2a8dea) — all 11 ModelDef
# vllm_pin_required values carry it (verify with `grep -rn
# vllm_pin_required sndr/model_configs/builtin`). dev338 was the original
# migration baseline; dev371 / 626fa9bba5 were the prior-canonical sprint
# era; both are now history, retained only as classifier buckets so the
# audit still reads a (hypothetical) rolled-back tree correctly.
#
# The classifier now recognizes THREE meaningful states:
#   * canonical  — the current fleet-unified pin (substring match below)
#   * dev338     — legacy migration baseline (rollback target)
#   * dev371     — prior-canonical sprint era (legacy)
# DFlash / placeholder hold semantics that used to attach to the dev371
# bucket now attach to the canonical bucket (the holds were LIFTED /
# remain documented, and the rollback-protection still works against the
# canonical pin). On the next pin bump, update CANONICAL_PIN_SUBSTRING
# (and ALLOWED_MODELDEF_PINS) to the new fleet pin.
CANONICAL_PIN_SUBSTRING = "dev301"

# ─── DFlash dev371 hold (P2.DFlash 2026-05-21) ─────────────────────────
#
# Q27-DFlash dev371 re-smoke 2026-05-21 failed at the DFlash drafter's
# `_create_draft_vllm_config()` step with a pydantic VllmConfig
# cross-validation rejection:
#
#   Value error, customized max_cudagraph_capture_size(=8) should be
#   consistent with the max value of cudagraph_capture_sizes(=6)
#
# Site: vllm/v1/spec_decode/dflash.py:74 → llm_base_proposer.py:1152
#       → dataclasses.replace() → pydantic validator (new in dev371).
#
# dev338 did not enforce the cross-validation rule, so the mismatch
# silently coexisted. dev371 rejects it, killing both worker processes
# at engine init. This is an upstream regression, not a renderer / P103
# issue.
#
# Hold lifted 2026-05-21 (M7) after PN275 (DFlash drafter VllmConfig
# max_cgs alignment, commits 40e60ec5 → 387a9a63) shipped its 3-layer
# fix:
#   (a) utils.replace text-patch self-install (workers' replace chain)
#   (b) vllm/config/vllm.py validator waiver (EngineCore direct
#       __post_init__ entry point)
#   (c) in-process setattr wrap (defense-in-depth)
# AND both DFlash variants passed dev371 E2E smoke:
#   * Q27-DFlash — receipt P2_DFLASH_M4_RETRY4_M2f_PASS_2026-05-21_RU.md
#   * Q35-DFlash — receipt P2_DFLASH_M5_RETRY_Q35_PASS_2026-05-21_RU.md
# AND M6 (commit 12d901a5) added `GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN: '1'`
# to BOTH DFlash ModelDef patches matrices, making the compat layer
# a hard prerequisite of each DFlash boot on dev371.
#
# With the hold lifted:
#   * DFlash dev371 promotions are PERMITTED — R-PIN-4 no longer
#     errors on them.
#   * DFlash dev338 entries are treated as ordinary "P2.4d candidate"
#     migration items (no longer "intentional hold"). The actual pin
#     promotion happens in M8 (one single-file commit per ModelDef).
#
# To re-engage the hold (rollback scenario), flip the flag back to
# False; existing tests TestDFlashHoldGate validate both states.

DFLASH_STEM_MARKER = "dflash"   # any model stem containing this is DFlash

DFLASH_DEV371_HOLD_LIFTED = True  # M7 lifted 2026-05-21 after PN275 + Q27/Q35 smokes

DFLASH_HOLD_RECEIPT_PATH = (
    "sndr_private/planning/bench_results/2026-05-21/"
    "P2_Q27_DFLASH_DEV371_RESMOKE_FAIL_2026-05-21_RU.md"
)

DFLASH_HOLD_REASON_SHORT = (
    "dev371 upstream pydantic VllmConfig validator rejects DFlash "
    "draft config (max_cudagraph_capture_size != "
    "max(cudagraph_capture_sizes))"
)

# ─── Placeholder ModelDef hold (P2.Q7B 2026-05-21) ─────────────────────
#
# Some ModelDefs in the registry are reference-only placeholders — they
# declare a `model_path:` pointing at a checkpoint that is not actually
# present on the operator's rig. Q7B-dense is the documented example
# (its own model_path comment says "placeholder dense small-class
# checkpoint"). Smoking such a ModelDef on dev371 is structurally
# impossible: vllm aborts at engine arg parse with an OSError before
# any dev371 code path runs.
#
# These ModelDefs are NOT migration debt — they cannot be promoted in
# either direction until the operator arranges a checkpoint and
# completes a real smoke. R-PIN-4 marks them with an explicit
# "placeholder" annotation rather than the generic "P2.4d candidate"
# tag, so the migration table doesn't pretend they're actionable.

KNOWN_PLACEHOLDER_MODELDEFS = frozenset({
    # ModelDefs intentionally registered as references but without a
    # deployed checkpoint on the validation rig. Add here only when a
    # smoke confirms the checkpoint is missing AND the ModelDef itself
    # self-declares placeholder status. Removing an entry requires the
    # operator to first arrange the checkpoint and re-run the smoke.
    "qwen3.6-7b-dense",   # see P2_Q7B_DENSE_DEV371_SMOKE_NOTRUN_2026-05-21
})

PLACEHOLDER_RECEIPT_PATH = (
    "sndr_private/planning/bench_results/2026-05-21/"
    "P2_Q7B_DENSE_DEV371_SMOKE_NOTRUN_2026-05-21_RU.md"
)

# A mutable image tag is anything ending in `:<word>` with no SHA suffix.
# We allow `:nightly-<sha>` and digest-backed `@sha256:...` references.
_BARE_MUTABLE_TAGS = ("nightly", "latest", "main", "stable", "dev")

# Representative (profile, hardware) combinations exercised by R-PIN-3.
# Picked to cover the three live hardware definitions; if a profile in
# this list is removed from the registry, R-PIN-3 reports a tooling
# error (the operator must update the list, not silently drop coverage).
REPRESENTATIVE_RENDERS = (
    # Canonical-config reorg (2026-06): repointed from the archived
    # gemma4-31b-tq-mtp-structured-k4 to the kept gemma4-31b-tq-default
    # (same gemma4 31B family + a5000-2x hardware; render parity is the
    # image-pin check, which is profile-agnostic).
    ("gemma4-31b-tq-default", "a5000-2x-24gbvram-16cpu-128gbram"),
    ("qwen3.6-35b-balanced", "a5000-2x-24gbvram-16cpu-128gbram"),
    ("tier-aware-3090", "single-3090-24gbvram"),
)


# ─── YAML helpers (regex, no PyYAML dependency) ─────────────────────────

# Each pattern is anchored to start-of-line + leading indent so that it
# only matches top-level fields under the expected parent. This avoids
# false positives from arbitrary comment / string content elsewhere.

_IMAGE_RE = re.compile(
    r"^\s{4}image:\s*(?P<value>\S+)",
    re.MULTILINE,
)
_IMAGE_DIGEST_RE = re.compile(
    r"^\s{4}image_digest:\s*(?P<value>\S+)",
    re.MULTILINE,
)
_VLLM_PIN_REQUIRED_RE = re.compile(
    r"^\s{2}vllm_pin_required:\s*(?P<value>\S+)",
    re.MULTILINE,
)


def _strip_yaml_value(raw: str) -> str:
    """Strip trailing comment + surrounding quotes from a YAML scalar."""
    # YAML inline comment starts with " #" after the value
    if " #" in raw:
        raw = raw.split(" #", 1)[0]
    raw = raw.strip().strip('"').strip("'")
    return raw


def _rel(path: Path) -> str:
    """Best-effort REPO_ROOT-relative display. Falls back to absolute
    path when the file lives outside the tree (e.g. tmp_path in tests).
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_hardware_image(yaml_path: Path) -> str | None:
    """Extract the ``runtime.docker.image`` value from a hardware YAML."""
    src = yaml_path.read_text()
    m = _IMAGE_RE.search(src)
    return _strip_yaml_value(m.group("value")) if m else None


def _read_hardware_image_digest(yaml_path: Path) -> str | None:
    src = yaml_path.read_text()
    m = _IMAGE_DIGEST_RE.search(src)
    return _strip_yaml_value(m.group("value")) if m else None


def _read_model_pin(yaml_path: Path) -> str | None:
    src = yaml_path.read_text()
    m = _VLLM_PIN_REQUIRED_RE.search(src)
    return _strip_yaml_value(m.group("value")) if m else None


# ─── Rule implementations ───────────────────────────────────────────────


def _is_bare_mutable(image: str) -> bool:
    """Return True if image is a mutable floating tag with no SHA suffix.

    Examples:
      bare:        vllm/vllm-openai:nightly
      bare:        vllm/vllm-openai:latest
      not bare:    vllm/vllm-openai:nightly-bf610c2f...
      not bare:    vllm/vllm-openai@sha256:7f047b...
    """
    # Digest reference is never mutable.
    if "@sha256:" in image:
        return False
    # No `:tag` at all (e.g. plain repo name) — treat as bare.
    if ":" not in image:
        return True
    tag = image.rsplit(":", 1)[1]
    return tag in _BARE_MUTABLE_TAGS


def check_r_pin_1_no_mutable_nightly() -> list[str]:
    """R-PIN-1: hardware YAMLs must pin an explicit-hash image."""
    issues: list[str] = []
    for yaml_path in sorted(HARDWARE_DIR.glob("*.yaml")):
        image = _read_hardware_image(yaml_path)
        if image is None:
            # No docker block — skip. R-PIN-2 may report if the YAML
            # declares docker runtime but lacks the image field.
            continue
        if _is_bare_mutable(image):
            issues.append(
                f"{_rel(yaml_path)}: runtime.docker.image "
                f"is a bare mutable tag ({image!r}). Use an explicit-hash "
                f"tag (e.g. vllm/vllm-openai:nightly-<sha>) or "
                f"@sha256:<digest> reference."
            )
    return issues


def check_r_pin_2_digest_present() -> list[str]:
    """R-PIN-2: every hardware YAML with `image:` must also have a
    non-empty `image_digest:` beginning with `vllm/vllm-openai@sha256:`.
    """
    issues: list[str] = []
    for yaml_path in sorted(HARDWARE_DIR.glob("*.yaml")):
        image = _read_hardware_image(yaml_path)
        if image is None:
            continue  # no docker block — out of scope
        digest = _read_hardware_image_digest(yaml_path)
        if not digest:
            issues.append(
                f"{_rel(yaml_path)}: runtime.docker.image "
                f"is declared ({image!r}) but image_digest is missing or "
                f"empty. Add the @sha256: digest for byte-lock."
            )
            continue
        if not digest.startswith("vllm/vllm-openai@sha256:"):
            issues.append(
                f"{_rel(yaml_path)}: image_digest "
                f"({digest!r}) does not begin with "
                f"'vllm/vllm-openai@sha256:'. Digest must be a full "
                f"repo@sha256: reference."
            )
    return issues


def check_r_pin_3_render_parity() -> tuple[list[str], list[str]]:
    """R-PIN-3: rendered IMAGE line equals composed cfg.docker.image
    AND equals hw.runtime.docker.image, for representative profiles."""
    errors: list[str] = []
    infos: list[str] = []

    try:
        from sndr.cli.legacy.profile import render_profile_launcher
        from sndr.model_configs.compose import compose
        from sndr.model_configs.registry_v2 import (
            load_hardware, load_model, load_profile,
        )
    except Exception as e:  # noqa: BLE001
        errors.append(
            f"R-PIN-3 tooling unavailable: import failed "
            f"({type(e).__name__}: {e}). "
            f"The audit cannot verify render parity without sndr_core."
        )
        return errors, infos

    image_line_re = re.compile(r'^IMAGE="([^"]+)"\s*$', re.MULTILINE)

    for profile_id, hardware_id in REPRESENTATIVE_RENDERS:
        try:
            profile = load_profile(profile_id)
            model = load_model(profile.parent_model)
            hardware = load_hardware(hardware_id)
            cfg = compose(model, hardware, profile)
        except Exception as e:  # noqa: BLE001
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): compose() "
                f"failed: {type(e).__name__}: {e}"
            )
            continue

        composed_image = (
            cfg.docker.image if cfg.docker is not None else None
        )
        hw_image = (
            hardware.runtime.docker.image
            if hardware.runtime is not None
            and hardware.runtime.docker is not None
            else None
        )
        if composed_image is None or hw_image is None:
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): "
                f"no docker block in composed cfg / hardware — cannot "
                f"verify image parity."
            )
            continue
        if composed_image != hw_image:
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): "
                f"composed cfg.docker.image={composed_image!r} differs "
                f"from hw.runtime.docker.image={hw_image!r}. compose() "
                f"must carry the hardware image verbatim."
            )

        try:
            script = render_profile_launcher(profile_id, hardware_id)
        except Exception as e:  # noqa: BLE001
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): "
                f"render_profile_launcher() failed: "
                f"{type(e).__name__}: {e}"
            )
            continue

        m = image_line_re.search(script)
        if m is None:
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): "
                f"rendered launcher has no `IMAGE=\"...\"` line."
            )
            continue
        rendered = m.group(1)
        if rendered != composed_image:
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): "
                f"rendered IMAGE={rendered!r} differs from composed "
                f"cfg.docker.image={composed_image!r}. The renderer "
                f"must emit the hardware-pinned image verbatim."
            )
        else:
            infos.append(
                f"{profile_id!r} + {hardware_id!r}: rendered = composed "
                f"= {rendered!r}"
            )

    return errors, infos


def _is_dflash_stem(stem: str) -> bool:
    """Return True if a ModelDef stem belongs to the DFlash spec-decode
    family. Stem-name match keeps the classifier transparent: any future
    DFlash variant added to the registry is auto-classified."""
    return DFLASH_STEM_MARKER in stem.lower()


def check_r_pin_4_modeldef_migration() -> tuple[list[str], list[str]]:
    """R-PIN-4: ModelDef pin migration status with DFlash hold gate.

    Fails on:
      * missing `vllm_pin_required` field
      * value outside ALLOWED_MODELDEF_PINS
      * a DFlash ModelDef promoted to dev371 ONLY while the project-
        wide hold is in effect (DFLASH_DEV371_HOLD_LIFTED = False).
        After M7 lifted the hold to True (2026-05-21), DFlash dev371
        promotions are permitted; this check becomes a no-op.

    Reports as info (NOT fail):
      * per-family migration table across both pins
      * non-DFlash Qwen models on dev338 → marked "P2.4d candidate"
      * DFlash ModelDefs on dev338:
          - when DFLASH_DEV371_HOLD_LIFTED is False → "DFlash hold —
            intentional, NOT a P2.4d candidate" (pre-M7 behavior, kept
            for rollback)
          - when DFLASH_DEV371_HOLD_LIFTED is True → "P2.4d candidate"
            (M7 + onwards; M8 commits move them to dev371)
    """
    errors: list[str] = []
    infos: list[str] = []

    by_family: dict[str, dict[str, list[str]]] = {
        "gemma": {"canonical": [], "dev338": [], "dev371": [], "other": []},
        "qwen": {"canonical": [], "dev338": [], "dev371": [], "other": []},
        "unknown": {"canonical": [], "dev338": [], "dev371": [], "other": []},
    }
    # DFlash ModelDefs by pin-state. `dflash_canonical` = on the current
    # fleet-unified pin (the post-hold, post-M8 resting place); the dev338
    # / dev371 lists are retained for a rolled-back tree.
    dflash_d338: list[str] = []
    dflash_d371: list[str] = []
    dflash_canonical: list[str] = []

    for yaml_path in sorted(MODEL_DIR.glob("*.yaml")):
        pin = _read_model_pin(yaml_path)
        stem = yaml_path.stem
        rel = _rel(yaml_path)
        if pin is None:
            errors.append(
                f"{rel}: vllm_pin_required is missing. Every ModelDef "
                f"must declare a pin."
            )
            continue
        if pin not in ALLOWED_MODELDEF_PINS:
            errors.append(
                f"{rel}: vllm_pin_required={pin!r} is not in the "
                f"allowed set "
                f"{sorted(ALLOWED_MODELDEF_PINS)}. Either add the new "
                f"pin to ALLOWED_MODELDEF_PINS (after smoke + bench "
                f"validation) or correct the typo."
            )
            continue
        # Classify family
        if stem.startswith(GEMMA_PREFIX):
            family = "gemma"
        elif stem.startswith(QWEN_PREFIX):
            family = "qwen"
        else:
            family = "unknown"
        is_dflash = _is_dflash_stem(stem)
        # Reconciled 2026-06-19: the fleet is unified on the canonical pin
        # (CANONICAL_PIN_SUBSTRING). It is classified into a dedicated
        # `canonical` bucket — the post-hold, post-M8 resting state — so
        # DFlash gates, P2.4d migration tracking, and the per-family
        # tables read correctly. The dev371 / 626fa9bba5 sprint-era forms
        # and the dev338 baseline are retained only so a rolled-back tree
        # still classifies. The DFlash promotion that used to land on
        # dev371 now lands on the canonical pin; the hold semantics move
        # with it. When the next pin lands, bump CANONICAL_PIN_SUBSTRING.
        is_canonical_pin = CANONICAL_PIN_SUBSTRING in pin
        if is_canonical_pin:
            by_family[family]["canonical"].append(stem)
            if is_dflash:
                dflash_canonical.append(stem)
        elif "dev338" in pin:
            by_family[family]["dev338"].append(stem)
            if is_dflash:
                dflash_d338.append(stem)
        elif "dev371" in pin or "626fa9bba5" in pin:
            by_family[family]["dev371"].append(stem)
            if is_dflash:
                dflash_d371.append(stem)
        else:
            by_family[family]["other"].append(stem)

        # DFlash hold enforcement — fail if a DFlash ModelDef is promoted
        # to a post-hold pin (dev371 sprint era OR the current canonical
        # pin) WHILE the project-wide hold is re-engaged. Default after M7
        # is hold-lifted, so this is a no-op on the live tree; flipping
        # DFLASH_DEV371_HOLD_LIFTED back to False (rollback) immediately
        # flags every DFlash promotion as a stale claim needing review.
        dflash_on_post_hold_pin = (
            "dev371" in pin or "626fa9bba5" in pin or is_canonical_pin
        )
        if is_dflash and dflash_on_post_hold_pin and not DFLASH_DEV371_HOLD_LIFTED:
            errors.append(
                f"{rel}: DFlash ModelDef promoted to a post-hold pin "
                f"({pin}) while the P2.DFlash hold is active. "
                f"{DFLASH_HOLD_REASON_SHORT}. DFlash ModelDefs must remain "
                f"on the dev338 rollback baseline until either a Genesis "
                f"compatibility patch lands and DFLASH_DEV371_HOLD_LIFTED "
                f"is flipped to True, or upstream relaxes the validator. "
                f"See {DFLASH_HOLD_RECEIPT_PATH}."
            )

    # Infos: per-family migration table.
    #
    # Reconciled 2026-06-19: the fleet-unified `canonical` bucket is the
    # primary state now. DFlash entries on the canonical pin carry the
    # (DFlash) family designation (the hold is lifted). Placeholder
    # ModelDefs on the canonical pin keep the "placeholder ModelDef"
    # annotation (NOT actionable — checkpoint not deployed). Everything
    # else on the canonical pin is "unified" (the P2.4d migration is
    # complete). The dev338 / dev371 paths below only fire on a
    # rolled-back tree.
    for family in ("gemma", "qwen", "unknown"):
        canon = by_family[family]["canonical"]
        d338 = by_family[family]["dev338"]
        d371 = by_family[family]["dev371"]
        other = by_family[family]["other"]
        if not (canon or d338 or d371 or other):
            continue
        infos.append(
            f"{family}: canonical={len(canon)} dev371={len(d371)} "
            f"dev338={len(d338)} other={len(other)}"
        )
        for stem in canon:
            if _is_dflash_stem(stem):
                # Hold lifted (default after M7) — DFlash on the canonical
                # pin is the post-M8 resting state. Tag with the family
                # designation so readers see it.
                infos.append(f"  {stem} → canonical  (DFlash)")
            elif stem in KNOWN_PLACEHOLDER_MODELDEFS:
                infos.append(
                    f"  {stem} → canonical  (placeholder ModelDef — "
                    f"checkpoint not deployed; NOT actionable)"
                )
            else:
                infos.append(f"  {stem} → canonical  (unified)")
        for stem in d371:
            note = ""
            if _is_dflash_stem(stem):
                # Only reachable when DFLASH_DEV371_HOLD_LIFTED is True
                # (else it was an error above), but tag it anyway so
                # readers see the family designation.
                note = "  (DFlash)"
            infos.append(f"  {stem} → dev371{note}")
        for stem in d338:
            if _is_dflash_stem(stem) and not DFLASH_DEV371_HOLD_LIFTED:
                # Hold active — DFlash dev338 entries are intentional
                # holds, NOT migration debt. (Path taken only on
                # rollback; default after M7 is hold-lifted.)
                infos.append(
                    f"  {stem} → dev338  (DFlash hold — intentional, "
                    f"NOT a P2.4d candidate)"
                )
            elif stem in KNOWN_PLACEHOLDER_MODELDEFS:
                infos.append(
                    f"  {stem} → dev338  (placeholder ModelDef — "
                    f"checkpoint not deployed; NOT a P2.4d candidate)"
                )
            else:
                # Includes DFlash dev338 entries when the hold is lifted
                # — they're ordinary P2.4d candidates awaiting M8
                # promotion.
                infos.append(f"  {stem} → dev338  (P2.4d candidate)")

    # Cross-cutting DFlash hold info block (always emitted when DFlash
    # ModelDefs exist, regardless of family classification — operators
    # reading the audit want a single place to see the hold status).
    # Reconciled 2026-06-19: counts the canonical-pin DFlash promotions
    # (the live state) alongside the legacy dev371 / dev338 buckets.
    if dflash_d338 or dflash_d371 or dflash_canonical:
        infos.append("")
        # Post-hold DFlash count = canonical + dev371 (both are post-hold
        # resting places; canonical is the current one, dev371 the prior).
        promoted = len(dflash_canonical) + len(dflash_d371)
        if DFLASH_DEV371_HOLD_LIFTED:
            infos.append(
                f"DFlash hold status: "
                f"DFLASH_DEV371_HOLD_LIFTED={DFLASH_DEV371_HOLD_LIFTED} "
                f"(M7 lifted 2026-05-21); migration candidates on "
                f"dev338={len(dflash_d338)}, "
                f"promoted (canonical+dev371)={promoted}"
            )
            infos.append(
                "DFlash hold lifted — PN275 (M2..M2f, commits "
                "40e60ec5..387a9a63) provides the dev371 compat layer; "
                "M6 commit 12d901a5 wires it into both DFlash ModelDef "
                "patches matrices. Both DFlash variants now ride the "
                "fleet-unified canonical pin."
            )
        else:
            infos.append(
                f"DFlash hold status: "
                f"DFLASH_DEV371_HOLD_LIFTED={DFLASH_DEV371_HOLD_LIFTED}; "
                f"intentional hold on dev338={len(dflash_d338)}, "
                f"promoted (canonical+dev371)={promoted}"
            )
            infos.append(
                f"DFlash hold reason: {DFLASH_HOLD_REASON_SHORT}"
            )
        infos.append(f"DFlash hold receipt: {DFLASH_HOLD_RECEIPT_PATH}")

    # Cross-cutting placeholder hold info block (only when one or more
    # known placeholder ModelDefs are present in the live tree).
    # Reconciled 2026-06-19: scan the canonical bucket too — a placeholder
    # ModelDef (e.g. qwen3.6-7b-dense) now rides the fleet-unified pin
    # while still lacking a deployed checkpoint, so it is "not actionable"
    # regardless of which pin it carries.
    present_placeholders = sorted(
        stem for stems in (by_family["gemma"]["canonical"],
                           by_family["qwen"]["canonical"],
                           by_family["unknown"]["canonical"],
                           by_family["gemma"]["dev338"],
                           by_family["qwen"]["dev338"],
                           by_family["unknown"]["dev338"])
        for stem in stems
        if stem in KNOWN_PLACEHOLDER_MODELDEFS
    )
    if present_placeholders:
        infos.append("")
        infos.append(
            f"Placeholder hold status: "
            f"{len(present_placeholders)} ModelDef(s) registered as "
            f"placeholders (checkpoint not deployed). Promotion blocked "
            f"until operator arranges the checkpoint and a smoke passes."
        )
        for stem in present_placeholders:
            infos.append(f"  {stem}")
        infos.append(f"Placeholder receipt: {PLACEHOLDER_RECEIPT_PATH}")

    return errors, infos


# ─── CLI driver ─────────────────────────────────────────────────────────


RULES = {
    "R-PIN-1": ("no bare mutable vLLM nightly", check_r_pin_1_no_mutable_nightly),
    "R-PIN-2": ("hardware image_digest present", check_r_pin_2_digest_present),
    "R-PIN-3": ("render parity (cfg.docker.image == hw.runtime.docker.image == IMAGE)", None),
    "R-PIN-4": ("ModelDef pin migration status", None),
}


def _run_rule(rule: str) -> tuple[list[str], list[str]]:
    """Return (errors, infos) for one rule."""
    if rule == "R-PIN-3":
        return check_r_pin_3_render_parity()
    if rule == "R-PIN-4":
        return check_r_pin_4_modeldef_migration()
    _, fn = RULES[rule]
    return fn(), []  # type: ignore[misc]


def _emit_text(
    results: dict[str, tuple[list[str], list[str]]],
    verbose: bool,
) -> None:
    print()
    print("╭──────────────────────────────────────────────────────────╮")
    print("│  V2 runtime / ModelDef pin harmonization audit           │")
    print("╰──────────────────────────────────────────────────────────╯")
    print()
    for rule, (errors, infos) in results.items():
        title = RULES[rule][0]
        if errors:
            print(f"  ✗ {rule}  {title}: {len(errors)} violation(s)")
            for issue in errors:
                print(f"      {issue}")
        else:
            print(f"  ✓ {rule}  {title}: clean")
        if verbose and infos:
            print(f"      ── informational ({len(infos)}) ──")
            for info in infos:
                print(f"      ℹ {info}")
    total_err = sum(len(e) for e, _ in results.values())
    total_info = sum(len(i) for _, i in results.values())
    print()
    if total_err:
        print(f"  ✗ {total_err} violation(s) — exit 1")
    else:
        suffix = f" + {total_info} informational" if total_info else ""
        print(f"  ✓ All selected rules clean{suffix} — exit 0")
    print()


def _emit_json(results: dict[str, tuple[list[str], list[str]]]) -> None:
    payload = {
        rule: {
            "title": RULES[rule][0],
            "violations": errors,
            "infos": infos,
            "status": "fail" if errors else "pass",
        }
        for rule, (errors, infos) in results.items()
    }
    payload["_summary"] = {
        "violations_total": sum(len(e) for e, _ in results.values()),
        "infos_total": sum(len(i) for _, i in results.values()),
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument(
        "--rule",
        choices=("all", *RULES.keys()),
        default="all",
        help="which rule to check (default: all).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of human report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="include informational (non-failing) findings in text output.",
    )
    args = parser.parse_args(argv)

    rules = list(RULES.keys()) if args.rule == "all" else [args.rule]

    try:
        results = {rule: _run_rule(rule) for rule in rules}
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"audit_v2_runtime_pins: tooling error: "
            f"{type(e).__name__}: {e}\n"
        )
        return 2

    if args.json:
        _emit_json(results)
    else:
        _emit_text(results, verbose=args.verbose)

    total_errors = sum(len(errors) for errors, _ in results.values())
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
