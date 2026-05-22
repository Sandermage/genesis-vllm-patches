# SPDX-License-Identifier: Apache-2.0
"""Phase 5.2.E (2026-05-22) — V2 ModelDef ↔ hardware canonical-pin audit.

Tests `scripts/audit_v2_modeldef_vs_hardware_pin.py`:

  1. Live tree passes (regression guard — would have failed before
     5.2.C when Q7B carried dev338 without a `pin_hold` annotation).
  2. Hardware split-rig triggers R-MD-HW-1 violation.
  3. ModelDef pin mismatch without `pin_hold` triggers R-MD-HW-2 ERROR.
  4. ModelDef pin mismatch with non-empty `pin_hold` is waived.
  5. `pin_hold: null` (and synonyms) does NOT count as a hold.
  6. Helpers correctly extract SHA from hardware and model YAML.

The audit is regex-only (no PyYAML in the script), so the tests build
minimal text fixtures that exercise the regex paths.
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import audit_v2_modeldef_vs_hardware_pin as M  # noqa: E402


# ─── Live-tree regression guard ────────────────────────────────────────────


class TestLiveTreeClean:
    def test_run_all_passes_on_live_repo(self):
        results, code = M.run_all()
        assert code == 0, (
            "audit_v2_modeldef_vs_hardware_pin failed on the live tree:\n"
            + "\n".join(
                f"  {r.rule_id}: {len(r.violations)} violation(s)\n    "
                + "\n    ".join(r.violations)
                for r in results
                if not r.passed
            )
        )
        assert all(r.passed for r in results)


# ─── Helper extractors ────────────────────────────────────────────────────


class TestExtractors:
    def test_pin_sha_extracts_short_hex(self):
        assert M._pin_sha("0.20.2rc1.dev371+gbf610c2f5") == "bf610c2f5"
        assert M._pin_sha("0.20.2rc1.dev338+gbf0d2dc6d") == "bf0d2dc6d"

    def test_pin_sha_returns_none_when_no_g_token(self):
        assert M._pin_sha("0.20.2rc1") is None
        assert M._pin_sha("not-a-pin") is None

    def test_strip_yaml_value_drops_inline_comment(self):
        assert M._strip_yaml_value(
            "0.20.2rc1.dev371+gbf610c2f5 # some comment"
        ) == "0.20.2rc1.dev371+gbf610c2f5"

    def test_strip_yaml_value_drops_quotes(self):
        assert M._strip_yaml_value("'value'") == "value"
        assert M._strip_yaml_value('"value"') == "value"


# ─── Hardware self-consistency (R-MD-HW-1) ────────────────────────────────


def _make_hw_yaml(tmp_path: Path, name: str, image: str) -> Path:
    p = tmp_path / name
    p.write_text(dedent(f"""\
        schema_version: 2
        kind: hardware
        id: {name.removesuffix('.yaml')}
        runtime:
          docker:
            image: {image}
            image_digest: vllm/vllm-openai@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    """))
    return p


def test_read_hardware_sha_extracts_nightly_sha(tmp_path):
    p = _make_hw_yaml(
        tmp_path, "hw-a.yaml",
        "vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036",
    )
    assert M._read_hardware_sha(p) == (
        "bf610c2f56764e1b30bc6065f4ceace3d6e59036"
    )


def test_read_hardware_sha_returns_none_for_bare_tag(tmp_path):
    p = _make_hw_yaml(tmp_path, "hw-a.yaml", "vllm/vllm-openai:nightly")
    assert M._read_hardware_sha(p) is None


def test_r_md_hw_1_passes_on_uniform_hardware(tmp_path, monkeypatch):
    _make_hw_yaml(
        tmp_path, "rig-a.yaml",
        "vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036",
    )
    _make_hw_yaml(
        tmp_path, "rig-b.yaml",
        "vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036",
    )
    monkeypatch.setattr(M, "HARDWARE_DIR", tmp_path)
    rr, canonical = M.rule_md_hw_1()
    assert rr.passed
    assert canonical == "bf610c2f56764e1b30bc6065f4ceace3d6e59036"


def test_r_md_hw_1_fails_on_split_rig(tmp_path, monkeypatch):
    _make_hw_yaml(
        tmp_path, "rig-a.yaml",
        "vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036",
    )
    _make_hw_yaml(
        tmp_path, "rig-b.yaml",
        "vllm/vllm-openai:nightly-bf0d2dc6d0000000000000000000000000000000",
    )
    monkeypatch.setattr(M, "HARDWARE_DIR", tmp_path)
    rr, canonical = M.rule_md_hw_1()
    assert not rr.passed
    assert canonical is None
    assert "split-rig" in rr.violations[0]


# ─── ModelDef vs hardware (R-MD-HW-2) ─────────────────────────────────────


CANONICAL_SHA = "bf610c2f56764e1b30bc6065f4ceace3d6e59036"


def _make_model_yaml(
    tmp_path: Path,
    name: str,
    pin: str = "0.20.2rc1.dev371+gbf610c2f5",
    pin_hold: str | None = None,
) -> Path:
    hold_line = (
        f"  pin_hold: {pin_hold}\n" if pin_hold is not None else ""
    )
    p = tmp_path / name
    p.write_text(dedent(f"""\
        schema_version: 2
        kind: model
        id: {name.removesuffix('.yaml')}
        versions:
          genesis_pin_min: v11.0.0
          vllm_pin_required: {pin}
          reference_metrics_ref: null
        {hold_line}""").rstrip() + "\n")
    return p


def test_r_md_hw_2_passes_when_pin_matches(tmp_path, monkeypatch):
    _make_model_yaml(tmp_path, "m1.yaml")
    monkeypatch.setattr(M, "MODEL_DIR", tmp_path)
    rr = M.rule_md_hw_2(CANONICAL_SHA)
    assert rr.passed
    assert "match" in rr.info[0]


def test_r_md_hw_2_errors_on_mismatch_without_hold(tmp_path, monkeypatch):
    _make_model_yaml(
        tmp_path, "drifted.yaml",
        pin="0.20.2rc1.dev338+gbf0d2dc6d",
        pin_hold=None,
    )
    monkeypatch.setattr(M, "MODEL_DIR", tmp_path)
    rr = M.rule_md_hw_2(CANONICAL_SHA)
    assert not rr.passed
    assert any("drifted.yaml" in v for v in rr.violations)
    assert any("pin_hold" in v for v in rr.violations)


def test_r_md_hw_2_waives_mismatch_when_pin_hold_set(tmp_path, monkeypatch):
    _make_model_yaml(
        tmp_path, "held.yaml",
        pin="0.20.2rc1.dev338+gbf0d2dc6d",
        pin_hold='"placeholder checkpoint missing"',
    )
    monkeypatch.setattr(M, "MODEL_DIR", tmp_path)
    rr = M.rule_md_hw_2(CANONICAL_SHA)
    assert rr.passed, rr.violations
    assert any("HOLD" in i for i in rr.info)


@pytest.mark.parametrize("hold_value", ["null", "None", "~", ""])
def test_r_md_hw_2_does_not_treat_empty_hold_as_waiver(
    tmp_path, monkeypatch, hold_value,
):
    _make_model_yaml(
        tmp_path, "explicit-null-hold.yaml",
        pin="0.20.2rc1.dev338+gbf0d2dc6d",
        pin_hold=hold_value,
    )
    monkeypatch.setattr(M, "MODEL_DIR", tmp_path)
    rr = M.rule_md_hw_2(CANONICAL_SHA)
    assert not rr.passed, (
        f"hold_value={hold_value!r} unexpectedly waived the mismatch"
    )


# ─── End-to-end: run_all() returns code 0 on the live tree ────────────────


class TestRunAllExitCode:
    def test_clean_tree_exit_zero(self):
        _, code = M.run_all()
        assert code == 0
