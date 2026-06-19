# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_runtime_pins.py` — P2.6 pin harmonization gates.

Covers:
  * R-PIN-1 bare-tag detection: positive + negative cases via
    `_is_bare_mutable` and a synthetic hardware fixture
  * R-PIN-2 digest presence: missing + wrong-prefix + ok
  * R-PIN-3 render parity: live invocation against the real registry
    (must pass on a clean tree)
  * R-PIN-4 ModelDef migration table: ALLOWED_MODELDEF_PINS set
    membership; unknown pin fails; live registry classified

Tests run on the live repo state — they implicitly verify that the
audit script reports the current tree as clean (post-P2.3). If a future
commit drifts the pins, this test file will fail in CI before the
audit script's exit code does.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_runtime_pins.py"


def _import_audit():
    name = "_audit_v2_runtime_pins_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── R-PIN-1 bare-tag predicate ─────────────────────────────────────────


class TestIsBareMutable:
    @pytest.mark.parametrize("image", [
        "vllm/vllm-openai:nightly",
        "vllm/vllm-openai:latest",
        "vllm/vllm-openai:main",
        "vllm/vllm-openai:stable",
        "vllm/vllm-openai:dev",
        "vllm/vllm-openai",  # no tag at all
    ])
    def test_bare_mutable_detected(self, image):
        mod = _import_audit()
        assert mod._is_bare_mutable(image), (
            f"{image!r} should be detected as a bare mutable tag"
        )

    @pytest.mark.parametrize("image", [
        "vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036",
        "vllm/vllm-openai:nightly-bf0d2dc6d764f7ab1a69504f60a55883ec6d9b39",
        "vllm/vllm-openai:v0.21.0",
        "vllm/vllm-openai:0.20.2rc1",
        ("vllm/vllm-openai@sha256:"
         "7f047b7e625283eee436cfc0c37784064f75422452ed4f9b6fa8c69eae6afe68"),
    ])
    def test_pinned_image_passes(self, image):
        mod = _import_audit()
        assert not mod._is_bare_mutable(image), (
            f"{image!r} should be accepted as explicitly pinned"
        )


# ─── R-PIN-1 / R-PIN-2 on the live hardware tree ───────────────────────


class TestLiveHardwareInvariants:
    def test_r_pin_1_clean(self):
        mod = _import_audit()
        issues = mod.check_r_pin_1_no_mutable_nightly()
        assert issues == [], (
            f"R-PIN-1 must be clean on the live tree (post-P2.3); "
            f"got: {issues}"
        )

    def test_r_pin_2_clean(self):
        mod = _import_audit()
        issues = mod.check_r_pin_2_digest_present()
        assert issues == [], (
            f"R-PIN-2 must be clean on the live tree (post-P2.3); "
            f"got: {issues}"
        )


# ─── R-PIN-1 / R-PIN-2 against synthetic fixtures ───────────────────────


_SYNTHETIC_BARE = """\
schema_version: 2
kind: hardware
id: test-bare-nightly
title: Synthetic bare nightly
maintainer: test
hardware:
  vendor: nvidia
  family: a5000
  n_gpus: 2
  min_vram_per_gpu_mib: 24576
runtime:
  default: docker
  supported: [docker]
  docker:
    image: vllm/vllm-openai:nightly
    image_digest: vllm/vllm-openai@sha256:0000000000000000000000000000000000000000000000000000000000000000
"""

_SYNTHETIC_MISSING_DIGEST = """\
schema_version: 2
kind: hardware
id: test-missing-digest
title: Synthetic missing digest
maintainer: test
hardware:
  vendor: nvidia
  family: a5000
  n_gpus: 2
  min_vram_per_gpu_mib: 24576
runtime:
  default: docker
  supported: [docker]
  docker:
    image: vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036
"""

_SYNTHETIC_WRONG_DIGEST_PREFIX = """\
schema_version: 2
kind: hardware
id: test-wrong-digest
title: Synthetic wrong digest prefix
maintainer: test
hardware:
  vendor: nvidia
  family: a5000
  n_gpus: 2
  min_vram_per_gpu_mib: 24576
runtime:
  default: docker
  supported: [docker]
  docker:
    image: vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036
    image_digest: just-a-sha-no-repo
"""


class TestSyntheticFixtures:
    def test_r_pin_1_flags_bare_synthetic(self, tmp_path, monkeypatch):
        mod = _import_audit()
        fake_hw = tmp_path / "hardware"
        fake_hw.mkdir()
        (fake_hw / "test-bare.yaml").write_text(_SYNTHETIC_BARE)
        monkeypatch.setattr(mod, "HARDWARE_DIR", fake_hw)
        issues = mod.check_r_pin_1_no_mutable_nightly()
        assert len(issues) == 1
        assert "test-bare.yaml" in issues[0]
        assert "bare mutable tag" in issues[0]

    def test_r_pin_2_flags_missing_digest(self, tmp_path, monkeypatch):
        mod = _import_audit()
        fake_hw = tmp_path / "hardware"
        fake_hw.mkdir()
        (fake_hw / "test-missing.yaml").write_text(_SYNTHETIC_MISSING_DIGEST)
        monkeypatch.setattr(mod, "HARDWARE_DIR", fake_hw)
        issues = mod.check_r_pin_2_digest_present()
        assert len(issues) == 1
        assert "image_digest is missing" in issues[0]

    def test_r_pin_2_flags_wrong_digest_prefix(
        self, tmp_path, monkeypatch,
    ):
        mod = _import_audit()
        fake_hw = tmp_path / "hardware"
        fake_hw.mkdir()
        (fake_hw / "test-wrong.yaml").write_text(
            _SYNTHETIC_WRONG_DIGEST_PREFIX
        )
        monkeypatch.setattr(mod, "HARDWARE_DIR", fake_hw)
        issues = mod.check_r_pin_2_digest_present()
        assert len(issues) == 1
        assert "does not begin with" in issues[0]


# ─── R-PIN-3 live render parity ─────────────────────────────────────────


class TestRenderParity:
    def test_r_pin_3_clean_on_live_tree(self):
        mod = _import_audit()
        errors, infos = mod.check_r_pin_3_render_parity()
        assert errors == [], (
            f"R-PIN-3 must be clean on the live tree; got errors: {errors}"
        )
        # Every (profile, hardware) representative should produce one info
        # line confirming the equality.
        assert len(infos) == len(mod.REPRESENTATIVE_RENDERS), (
            f"expected one info per representative render; "
            f"got {len(infos)} for {len(mod.REPRESENTATIVE_RENDERS)} "
            f"renders"
        )
        for info in infos:
            assert "rendered = composed" in info


# ─── R-PIN-4 ModelDef pin migration ────────────────────────────────────


class TestModelDefMigration:
    def test_allowed_pin_set_explicit(self):
        mod = _import_audit()
        # Both currently-live pins must be present in the allowed set.
        assert "0.20.2rc1.dev338+gbf0d2dc6d" in mod.ALLOWED_MODELDEF_PINS
        assert "0.20.2rc1.dev371+gbf610c2f5" in mod.ALLOWED_MODELDEF_PINS

    def test_r_pin_4_clean_on_live_tree(self):
        mod = _import_audit()
        errors, infos = mod.check_r_pin_4_modeldef_migration()
        assert errors == [], (
            f"R-PIN-4 must be clean on the live tree; got: {errors}"
        )
        # Migration table must mention both Gemma and Qwen families
        # in the infos.
        joined = "\n".join(infos)
        assert "gemma" in joined and "qwen" in joined

    def test_r_pin_4_flags_unknown_pin(self, tmp_path, monkeypatch):
        mod = _import_audit()
        fake_models = tmp_path / "model"
        fake_models.mkdir()
        (fake_models / "qwen3.6-fake.yaml").write_text(
            "schema_version: 2\n"
            "kind: model\n"
            "id: fake\n"
            "title: Fake model\n"
            "maintainer: test\n"
            "versions:\n"
            "  vllm_pin_required: 0.99.9rc1.dev999+gunknown\n"
        )
        monkeypatch.setattr(mod, "MODEL_DIR", fake_models)
        errors, _ = mod.check_r_pin_4_modeldef_migration()
        assert len(errors) == 1
        assert "not in the allowed set" in errors[0]

    def test_r_pin_4_flags_missing_pin(self, tmp_path, monkeypatch):
        mod = _import_audit()
        fake_models = tmp_path / "model"
        fake_models.mkdir()
        (fake_models / "gemma-stub.yaml").write_text(
            "schema_version: 2\n"
            "kind: model\n"
            "id: stub\n"
            "title: Stub model\n"
            "maintainer: test\n"
        )
        monkeypatch.setattr(mod, "MODEL_DIR", fake_models)
        errors, _ = mod.check_r_pin_4_modeldef_migration()
        assert len(errors) == 1
        assert "vllm_pin_required is missing" in errors[0]


# ─── CLI driver ─────────────────────────────────────────────────────────


class TestDFlashHoldGate:
    """P2.DFlash hold gate — R-PIN-4 distinguishes DFlash ModelDefs from
    generic migration candidates.

    Sprint history: the Q27-DFlash dev371 re-smoke 2026-05-21 found that
    upstream dev371 rejects DFlash draft VllmConfig construction at a
    pydantic cross-validator (vllm/config/vllm.py:1703-1715). Genesis
    backported PN275 (M2..M2f, commits 40e60ec5..387a9a63) as a 3-layer
    compatibility patch (utils.replace text-patch self-install +
    validator waiver + in-process setattr wrap). After both DFlash
    variants passed dev371 E2E smoke and M6 wired PN275 into both
    DFlash ModelDef patches matrices, M7 lifted DFLASH_DEV371_HOLD_LIFTED
    to True (commit 43fe70b7) and M8 promoted both DFlash pins to dev371
    (M8-Q27 commit 7e310b25, M8-Q35 closing this sprint).

    R-PIN-4 invariants:

      * NOT mark DFlash dev338 as "P2.4d candidate" — they would be
        intentional holds if any remained (the live tree has none
        post-M8);
      * REJECT (error) any DFlash ModelDef promoted to dev371 while
        DFLASH_DEV371_HOLD_LIFTED is False (rollback protection).

    These tests pin both invariants.
    """

    def test_dflash_classifier_stem_match(self):
        mod = _import_audit()
        assert mod._is_dflash_stem("qwen3.6-27b-dflash") is True
        assert mod._is_dflash_stem("qwen3.6-35b-a3b-fp8-dflash") is True
        assert mod._is_dflash_stem("qwen3.6-27b-int4-autoround-tq-k8v4") is False
        assert mod._is_dflash_stem("qwen3.6-35b-a3b-fp8") is False
        assert mod._is_dflash_stem("gemma-4-31b-it-awq") is False
        # case-insensitive
        assert mod._is_dflash_stem("DFLASH-test") is True

    def test_dflash_hold_lifted_default_true_after_m7(self):
        """DFLASH_DEV371_HOLD_LIFTED defaults to True after M7
        (commit 2026-05-21, after PN275 landed in M2..M2f + both
        Q27-DFlash and Q35-DFlash passed dev371 E2E smoke + M6
        wired PN275 into both DFlash ModelDef patches matrices).

        Re-engaging the hold (rollback) is achieved by flipping the
        constant back to False; TestDFlashHoldGate covers both
        states. Receipt + reason constants stay populated regardless
        of the flag — they document the original defect."""
        mod = _import_audit()
        assert mod.DFLASH_DEV371_HOLD_LIFTED is True
        assert "dflash" in mod.DFLASH_HOLD_RECEIPT_PATH.lower()
        assert "max_cudagraph_capture_size" in mod.DFLASH_HOLD_REASON_SHORT

    def test_live_tree_both_dflash_on_canonical_pin(self):
        """Reconciled 2026-06-19: the P2.DFlash sprint is CLOSED and the
        whole fleet is unified on the canonical pin
        (0.23.1rc1.dev148+gb4c80ec0f). After M8 promoted both DFlash
        variants past the dev371 hold, the subsequent fleet pin bumps
        carried them onto the current canonical pin. BOTH DFlash variants
        now ride the `canonical` bucket; NO DFlash remains on dev338 or
        the legacy dev371 sprint pin in the live tree.

        Sprint chain (history): M1 design → M2..M2f PN275 3-layer fix →
        M5 retry Q35-DFlash dev371 smoke PASS → M6 wired PN275 into both
        DFlash patches matrices (commit 12d901a5) → M7 lifted hold
        (43fe70b7) → M8 promoted both DFlash pins → fleet pin bumps
        (dev371 → 626fa9bb → dev259 → dev491 → dev101 → dev148) landed
        them on the canonical pin."""
        mod = _import_audit()
        errors, infos = mod.check_r_pin_4_modeldef_migration()
        assert errors == [], (
            f"R-PIN-4 must be clean on the live tree; got: {errors}"
        )
        joined = "\n".join(infos)
        # Both DFlash variants must now appear on the canonical pin with
        # the (DFlash) family designation.
        assert (
            "qwen3.6-27b-dflash → canonical  (DFlash)" in joined
        ), f"qwen3.6-27b-dflash must appear on canonical; got:\n{joined}"
        assert (
            "qwen3.6-35b-a3b-fp8-dflash → canonical  (DFlash)" in joined
        ), f"qwen3.6-35b-dflash must appear on canonical; got:\n{joined}"
        # Neither DFlash may remain on dev338 or the legacy dev371 pin —
        # the sprint is closed and the fleet is unified.
        assert (
            "qwen3.6-27b-dflash → dev338" not in joined
        ), "qwen3.6-27b-dflash must NOT remain on dev338"
        assert (
            "qwen3.6-35b-a3b-fp8-dflash → dev338" not in joined
        ), "qwen3.6-35b-dflash must NOT remain on dev338"
        assert (
            "qwen3.6-27b-dflash → dev371" not in joined
        ), "qwen3.6-27b-dflash must NOT remain on the legacy dev371 pin"
        assert (
            "qwen3.6-35b-a3b-fp8-dflash → dev371" not in joined
        ), "qwen3.6-35b-dflash must NOT remain on the legacy dev371 pin"
        # No P2.4d candidate or pre-M7 hold tag should be attached to
        # any DFlash entry anymore.
        assert (
            "dflash → dev338  (p2.4d candidate)" not in joined.lower()
        ), "no DFlash on dev338 → no P2.4d-candidate tag expected"
        assert "DFlash hold —" not in joined, (
            "no DFlash on dev338 → no 'DFlash hold' tag should appear"
        )
        # Cross-cutting block still reflects lifted state.
        assert "DFlash hold status:" in joined
        assert "DFLASH_DEV371_HOLD_LIFTED=True" in joined
        assert "M7 lifted" in joined
        # Both DFlash variants are counted as promoted (canonical+dev371).
        assert "promoted (canonical+dev371)=2" in joined
        # Receipt still cited for the original defect trail.
        assert "DFlash hold receipt:" in joined

    def test_dflash_canonical_promotions_revert_to_violations_when_hold_re_engaged(
        self, monkeypatch,
    ):
        """Rollback safety: setting DFLASH_DEV371_HOLD_LIFTED back to
        False with BOTH DFlash variants now on the canonical (post-hold)
        pin must surface BOTH as R-PIN-4 violations. This is the
        protection the rollback flag provides — if PN275 regresses on a
        future pin, flipping the constant immediately flags every DFlash
        promotion past the dev338 baseline as a stale claim that needs
        review.

        Reconciled 2026-06-19: the DFlash variants moved from the legacy
        dev371 sprint pin onto the fleet-unified canonical pin, so the
        violation now cites the canonical pin (a post-hold pin) rather
        than the dev371 string."""
        mod = _import_audit()
        monkeypatch.setattr(mod, "DFLASH_DEV371_HOLD_LIFTED", False)
        errors, infos = mod.check_r_pin_4_modeldef_migration()
        # Both DFlash promotions must appear as violations on rollback.
        joined_errors = "\n".join(errors)
        assert (
            "qwen3.6-27b-dflash" in joined_errors
            and "post-hold pin" in joined_errors
        ), (
            "rollback must surface Q27-DFlash canonical-pin promotion as "
            f"violation; got: {errors}"
        )
        assert (
            "qwen3.6-35b-a3b-fp8-dflash" in joined_errors
            and "post-hold pin" in joined_errors
        ), (
            "rollback must surface Q35-DFlash canonical-pin promotion as "
            f"violation; got: {errors}"
        )
        # Exactly the two live DFlash variants are flagged — no more.
        assert len(errors) == 2, (
            f"expected exactly 2 DFlash violations on rollback; got: {errors}"
        )
        # And no DFlash remains on dev338, so no "intentional hold"
        # annotation should appear in the info list for any DFlash.
        joined_infos = "\n".join(infos)
        assert "dflash → dev338" not in joined_infos.lower(), (
            "no DFlash on dev338 → no dev338 info line should be "
            "produced by the rollback path"
        )

    def test_non_dflash_non_placeholder_qwen_dev338_uses_p2_4d_tag(
        self, tmp_path, monkeypatch,
    ):
        """Any non-DFlash, non-placeholder Qwen ModelDef on dev338
        must be marked as 'P2.4d candidate'. The DFlash hold and
        placeholder hold logic must NOT bleed into the generic
        migration-candidate family.

        Live tree note: as of 2026-05-21 the only remaining Qwen
        dev338 entries are DFlash holds (2) + Q7B placeholder (1);
        there are no generic P2.4d candidates left. We use a
        synthetic fixture so the test stays meaningful even when the
        live tree has zero candidates."""
        mod = _import_audit()
        fake_models = tmp_path / "model"
        fake_models.mkdir()
        (fake_models / "qwen3.6-99b-synthetic.yaml").write_text(
            "schema_version: 2\n"
            "kind: model\n"
            "id: synthetic\n"
            "title: Synthetic non-DFlash non-placeholder\n"
            "maintainer: test\n"
            "versions:\n"
            "  vllm_pin_required: 0.20.2rc1.dev338+gbf0d2dc6d\n"
        )
        monkeypatch.setattr(mod, "MODEL_DIR", fake_models)
        errors, infos = mod.check_r_pin_4_modeldef_migration()
        assert errors == []
        joined = "\n".join(infos)
        assert (
            "qwen3.6-99b-synthetic → dev338  (P2.4d candidate)" in joined
        )

    def test_dflash_promoted_to_dev371_fails_when_hold_re_engaged(
        self, tmp_path, monkeypatch,
    ):
        """Rollback safety: if an operator re-engages the hold by
        flipping DFLASH_DEV371_HOLD_LIFTED back to False, R-PIN-4
        must FAIL on any DFlash dev371 promotion — preserving the
        pre-M7 protection against accidental promotion.

        Post-M7 the default is True (lifted), so we monkeypatch False
        to exercise the rollback path explicitly."""
        mod = _import_audit()
        monkeypatch.setattr(mod, "DFLASH_DEV371_HOLD_LIFTED", False)
        fake_models = tmp_path / "model"
        fake_models.mkdir()
        (fake_models / "qwen3.6-27b-dflash.yaml").write_text(
            "schema_version: 2\n"
            "kind: model\n"
            "id: dflash\n"
            "title: Fake DFlash\n"
            "maintainer: test\n"
            "versions:\n"
            "  vllm_pin_required: 0.20.2rc1.dev371+gbf610c2f5\n"
        )
        monkeypatch.setattr(mod, "MODEL_DIR", fake_models)
        errors, _ = mod.check_r_pin_4_modeldef_migration()
        # Exactly one error — the DFlash promotion violation
        assert len(errors) == 1
        assert "DFlash" in errors[0]
        assert "dev371" in errors[0]
        assert "hold is active" in errors[0]

    def test_dflash_dev371_allowed_when_hold_lifted(
        self, tmp_path, monkeypatch,
    ):
        """After a future compatibility patch lands and the operator
        flips DFLASH_DEV371_HOLD_LIFTED to True, DFlash dev371 must
        no longer fail (it would still be subject to allowed-pin
        membership checks)."""
        mod = _import_audit()
        fake_models = tmp_path / "model"
        fake_models.mkdir()
        (fake_models / "qwen3.6-27b-dflash.yaml").write_text(
            "schema_version: 2\n"
            "kind: model\n"
            "id: dflash\n"
            "title: Fake DFlash\n"
            "maintainer: test\n"
            "versions:\n"
            "  vllm_pin_required: 0.20.2rc1.dev371+gbf610c2f5\n"
        )
        monkeypatch.setattr(mod, "MODEL_DIR", fake_models)
        monkeypatch.setattr(mod, "DFLASH_DEV371_HOLD_LIFTED", True)
        errors, infos = mod.check_r_pin_4_modeldef_migration()
        assert errors == [], (
            f"With hold lifted, DFlash dev371 must NOT fail; got: {errors}"
        )
        joined = "\n".join(infos)
        # The DFlash designation should still be visible in the info
        assert "(DFlash)" in joined

    def test_dflash_dev338_no_promotion_no_failure(
        self, tmp_path, monkeypatch,
    ):
        """A DFlash ModelDef pinned to dev338 must NOT fail regardless
        of the hold state — dev338 is the safe rollback target."""
        mod = _import_audit()
        fake_models = tmp_path / "model"
        fake_models.mkdir()
        (fake_models / "qwen3.6-27b-dflash.yaml").write_text(
            "schema_version: 2\n"
            "kind: model\n"
            "id: dflash\n"
            "title: Fake DFlash\n"
            "maintainer: test\n"
            "versions:\n"
            "  vllm_pin_required: 0.20.2rc1.dev338+gbf0d2dc6d\n"
        )
        monkeypatch.setattr(mod, "MODEL_DIR", fake_models)
        # Hold active (default) → still clean
        errors, _ = mod.check_r_pin_4_modeldef_migration()
        assert errors == []
        # Hold lifted → still clean
        monkeypatch.setattr(mod, "DFLASH_DEV371_HOLD_LIFTED", True)
        errors, _ = mod.check_r_pin_4_modeldef_migration()
        assert errors == []


class TestPlaceholderHold:
    """P2.Q7B placeholder hold — R-PIN-4 distinguishes ModelDefs that
    cannot be smoked at all (no checkpoint on the rig) from generic
    migration candidates.

    Q7B-dense is the documented example: its `model_path:` comment
    self-declares "placeholder dense small-class checkpoint" and the
    server's `/nfs/genesis/models/` has no `Qwen3.6-7B-*` directory.
    Trying to boot it on dev371 fails at vllm engine-arg parse with
    an `OSError`, long before any dev371 code path runs — see
    P2_Q7B_DENSE_DEV371_SMOKE_NOTRUN_2026-05-21_RU.md.

    These tests pin the audit annotation so a future reader of the
    migration table sees "checkpoint not deployed" instead of
    "P2.4d candidate" (which would imply it's actionable).
    """

    def test_q7b_in_known_placeholder_set(self):
        mod = _import_audit()
        assert "qwen3.6-7b-dense" in mod.KNOWN_PLACEHOLDER_MODELDEFS
        assert mod.PLACEHOLDER_RECEIPT_PATH.endswith(
            "P2_Q7B_DENSE_DEV371_SMOKE_NOTRUN_2026-05-21_RU.md"
        )

    def test_live_tree_q7b_marked_placeholder_not_candidate(self):
        """The live tree must mark Q7B-dense with the placeholder
        annotation, NOT the generic 'P2.4d candidate' / 'unified'
        annotation. Cross-cutting info block must list it.

        Reconciled 2026-06-19: Q7B-dense now rides the fleet-unified
        canonical pin (like every other ModelDef) but still has no
        deployed checkpoint, so the annotation stays 'placeholder
        ModelDef — checkpoint not deployed; NOT actionable'."""
        mod = _import_audit()
        errors, infos = mod.check_r_pin_4_modeldef_migration()
        assert errors == []
        joined = "\n".join(infos)
        assert (
            "qwen3.6-7b-dense → canonical  (placeholder ModelDef" in joined
        )
        # Must NOT be mis-tagged as a generic unified entry or a stale
        # P2.4d migration candidate.
        assert (
            "qwen3.6-7b-dense → canonical  (unified)" not in joined
        )
        assert (
            "qwen3.6-7b-dense → dev338  (P2.4d candidate)" not in joined
        )
        assert "Placeholder hold status:" in joined
        assert "Placeholder receipt:" in joined

    def test_placeholder_set_does_not_collide_with_dflash(self):
        """A ModelDef in KNOWN_PLACEHOLDER_MODELDEFS must not also be
        a DFlash stem. The two hold categories are independent —
        prevent accidental dual-classification in the future."""
        mod = _import_audit()
        for stem in mod.KNOWN_PLACEHOLDER_MODELDEFS:
            assert not mod._is_dflash_stem(stem), (
                f"{stem!r} is in BOTH KNOWN_PLACEHOLDER_MODELDEFS and "
                f"the DFlash stem pattern; pick one classification"
            )

    def test_placeholder_dev338_no_error(self, tmp_path, monkeypatch):
        """A known placeholder ModelDef on dev338 must NOT fail R-PIN-4.
        It's a legitimate hold state, not a violation."""
        mod = _import_audit()
        fake_models = tmp_path / "model"
        fake_models.mkdir()
        (fake_models / "qwen3.6-7b-dense.yaml").write_text(
            "schema_version: 2\n"
            "kind: model\n"
            "id: q7b\n"
            "title: Q7B\n"
            "maintainer: test\n"
            "versions:\n"
            "  vllm_pin_required: 0.20.2rc1.dev338+gbf0d2dc6d\n"
        )
        monkeypatch.setattr(mod, "MODEL_DIR", fake_models)
        errors, infos = mod.check_r_pin_4_modeldef_migration()
        assert errors == []
        joined = "\n".join(infos)
        assert "placeholder ModelDef" in joined

    def test_unknown_stem_on_dev338_still_generic_candidate(
        self, tmp_path, monkeypatch,
    ):
        """A ModelDef NOT in the placeholder set and NOT DFlash must
        still be marked as 'P2.4d candidate'. Negative test ensures
        the placeholder logic doesn't accidentally swallow legitimate
        candidates."""
        mod = _import_audit()
        fake_models = tmp_path / "model"
        fake_models.mkdir()
        (fake_models / "qwen3.6-new-variant.yaml").write_text(
            "schema_version: 2\n"
            "kind: model\n"
            "id: nv\n"
            "title: New Variant\n"
            "maintainer: test\n"
            "versions:\n"
            "  vllm_pin_required: 0.20.2rc1.dev338+gbf0d2dc6d\n"
        )
        monkeypatch.setattr(mod, "MODEL_DIR", fake_models)
        _, infos = mod.check_r_pin_4_modeldef_migration()
        joined = "\n".join(infos)
        assert (
            "qwen3.6-new-variant → dev338  (P2.4d candidate)" in joined
        )


class TestCli:
    def test_default_invocation_exits_clean(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"audit must exit 0 on clean tree; got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "V2 runtime / ModelDef pin harmonization audit" in result.stdout
        assert "All selected rules clean" in result.stdout

    def test_json_output_parseable(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert set(payload.keys()) >= {
            "R-PIN-1", "R-PIN-2", "R-PIN-3", "R-PIN-4", "_summary",
        }
        assert payload["_summary"]["violations_total"] == 0
        for rule in ("R-PIN-1", "R-PIN-2", "R-PIN-3", "R-PIN-4"):
            assert payload[rule]["status"] == "pass"

    def test_single_rule_filter(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--rule", "R-PIN-1", "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "R-PIN-1" in payload
        # Other rules must not be present when one is filtered
        assert "R-PIN-3" not in payload

    def test_verbose_shows_infos(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--verbose"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # Verbose must surface the migration table. Reconciled 2026-06-19:
        # the fleet is unified on the canonical pin, so the table now
        # shows `→ canonical` entries (with the per-family `canonical=N`
        # header) instead of the legacy dev371/dev338 P2.4d-candidate
        # lines. Accept any of the migration-table markers so the test
        # stays meaningful across future pin bumps (a rolled-back tree
        # would still surface the dev338 / P2.4d-candidate forms).
        assert (
            "→ canonical" in result.stdout
            or "canonical=" in result.stdout
            or "P2.4d candidate" in result.stdout
            or "→ dev338" in result.stdout
        )
