# SPDX-License-Identifier: Apache-2.0
"""Iron rule #11 enforcement — retire/supersession provenance.

Sander 2026-05-11 strategic mandate (codified in skill v2 iron rule #11):
when an upstream PR supersedes a Genesis patch, we either retire it
(byte-equivalent) or update it (we do MORE). Either way the registry
entry must carry explicit provenance:

  - `superseded_by`:    string explaining the supersession (PR + outcome)
  - `vllm_version_range`: PEP 440 spec gating the pin range where the
                           original patch was valid (upper bound = pin
                           where upstream's equivalent landed)

Without these, future maintainers reading the registry have no audit
trail of WHY a patch is retired, on WHICH pins it applies, or what to
verify when bumping further. This test enforces the discipline.

What this test rejects (will fail the suite):
  1. lifecycle="retired" without BOTH superseded_by AND vllm_version_range
  2. superseded_by present without vllm_version_range (except legacy
     auto-apply patches whose synthetic GENESIS_LEGACY_* flag doesn't
     fire pin-gate anyway — those are cosmetically OK to omit)

What this test allows (xfail / known waivers):
  - lifecycle="legacy" patches with superseded_by but no pin-gate
    (synthetic env_flag → pin-gate adds no behavioral effect, just
    documentation; cosmetic gap, low priority)

How to fix a failure:
  - Add `superseded_by`: short string naming the upstream PR + state
    (e.g. `"vllm#41268 (merged 2026-04-30, byte-equivalent on dev209)"`)
  - Add `vllm_version_range`: tuple or string declaring the upper bound
    (e.g. `"<0.20.2rc1.dev93"` for patches superseded before dev93)
  - If the patch is retired but no upstream supersession (e.g. hypothesis
    disproven, or retired pending refactor) — document that in the
    `notes`/`credit` field and add a `_RETIRED_NO_SUPERSEDE_WAIVER`
    constant below for explicit waiver.

Adding a new retired patch?
  - Always set BOTH superseded_by and vllm_version_range, OR
  - Add the patch ID to _RETIRED_NO_SUPERSEDE_WAIVER below with a brief
    reason in the docstring there.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# Patches retired for reasons OTHER than upstream supersession (hypothesis
# disproven, internal refactor pending, etc.). These are explicitly waived
# from the superseded_by requirement. Add new entries with a one-line
# reason in the comment. Do not add casually — the meta-test is meant
# to surface forgotten provenance, not be silenced.
#
# History 2026-05-11: P8 + PN78 promoted out of waiver after tightening —
# both had identifiable upstream supersession in notes (specific commit/
# function names) even without formal PR numbers. Only TRUE no-supersede
# retires remain.
_RETIRED_NO_SUPERSEDE_WAIVER = {
    # Genesis hypothesis disproven 2026-04-25 — MTP/Eagle drafter GDN
    # state recovery turned out unnecessary on our PROD path. Retired
    # without supersession (no upstream equivalent — Genesis-specific
    # hypothesis that didn't pan out).
    "P63": "hypothesis disproven 2026-04-25 (not a backport)",
    # P61 retired in v7.62.5 when P12 was upgraded to FIRST-occurrence
    # logic — P61's LAST-occurrence approach superseded BY OUR OWN P12.
    # Internal supersession; no upstream PR involved.
    "P61": "internal: P12 upgraded to FIRST-occurrence (v7.62.5)",
    # PN108 design conflict — fla recurrent kernel cannot serve single-seq
    # prefill (TOMBSTONED). Retired 2026-05-14 when sync'd with docstring.
    "PN108": "design conflict: fla recurrent kernel cannot serve single-seq prefill",
    # PN134 bench-validated regressor — vllm#42686 backport caused -25%
    # TPS on hybrid_gdn_moe (dev371, 2026-05-15). No upstream supersession
    # — the upstream PR is OPEN, our backport is theoretically correct but
    # behaviorally wrong on this model class. Module kept on disk with
    # double-env-flag guard for future dense-attention experiments.
    "PN134": "bench-validated regressor: -25% TPS on hybrid_gdn_moe (2026-05-15)",
    # G4_78 retired via internal architecture decision documented in
    # backend_plan P1.8 A2 (drafter_kv_sharing=physical). No upstream
    # version boundary applies — the supersessor is an internal
    # spec_decode architecture choice, not a merged vllm PR. Phase 5.3.C
    # (2026-05-22).
    "G4_78": "internal architecture decision (drafter_kv_sharing=physical, P1.8 A2)",
    # G4_19C torch.compile FakeTensor incompatibility found on rig boot
    # 2026-05-29 (gemma4-tq-mtp-structured-k4 container). _g4_19c_
    # roundtrip_tensor custom kernel not wrapped as opaque op; Dynamo
    # raises "Cannot access data pointer of Tensor (FakeTensor)" during
    # fake-tensor tracing through Gemma4Attention.forward. No upstream
    # supersession — fix requires Genesis-side rewrite (wrap kernel via
    # torch.library.custom_op, see P7b reference pattern). Workaround:
    # GENESIS_ENABLE_G4_19C_ATTN_WRAP=0 on launcher. Detailed retired_
    # reason in registry credit.
    "G4_19C": "torch.compile FakeTensor bug 2026-05-29 (kernel not opaque-wrapped)",
}


# Patches whose `superseded_by` value cannot be expressed as a vllm
# version cut, so `vllm_version_range` is structurally inapplicable.
# Two sub-classes:
#   (a) Legacy auto-apply patches (pre-dispatcher era, synthetic
#       GENESIS_LEGACY_* flags) — pin-gate adds no behavioral effect
#       on these because the legacy path bypasses applies_to gating;
#       the wire detector handles their skip via upstream marker
#       detection. P4 / P12 / P26.
#   (b) Patches superseded by an INTERNAL architecture decision
#       (backend plan, internal evolution) rather than an upstream
#       vllm commit. The `superseded_by` value is a free-form string,
#       not a PR / commit / version. G4_78 (Phase 5.3.C 2026-05-22).
_LEGACY_PIN_GATE_WAIVER = {"P4", "P12", "P26", "G4_78"}


def _load_registry_entries() -> dict[str, str]:
    """Read registry.py and return {patch_id: body_text} for every entry."""
    registry_path = (
        Path(__file__).resolve().parents[3]
        / "vllm" / "sndr_core" / "dispatcher" / "registry.py"
    )
    text = registry_path.read_text()
    entries: dict[str, str] = {}
    for m in re.finditer(
        r'    "(\w+)":\s*\{(.*?)^    \},', text, flags=re.M | re.S
    ):
        entries[m.group(1)] = m.group(2)
    return entries


def _has_field(body: str, field: str) -> bool:
    return bool(re.search(rf'"{field}"\s*:', body))


def _lifecycle(body: str) -> str | None:
    m = re.search(r'"lifecycle"\s*:\s*"([^"]+)"', body)
    return m.group(1) if m else None


class TestIronRuleEleven:
    """Enforce iron rule #11 provenance discipline on the registry."""

    def test_retired_patches_have_full_provenance(self):
        """Every lifecycle="retired" patch must have BOTH superseded_by
        AND vllm_version_range, OR be explicitly waived."""
        entries = _load_registry_entries()
        violations: list[str] = []
        for pid, body in entries.items():
            if _lifecycle(body) != "retired":
                continue
            if pid in _RETIRED_NO_SUPERSEDE_WAIVER:
                # Waiver: no supersession (hypothesis disproven, etc.).
                # Still recommend pin-gate but don't require it.
                continue
            has_sb = _has_field(body, "superseded_by")
            has_vvr = _has_field(body, "vllm_version_range")
            if not (has_sb and has_vvr):
                missing = []
                if not has_sb:
                    missing.append("superseded_by")
                if not has_vvr:
                    missing.append("vllm_version_range")
                violations.append(
                    f"  {pid}: missing {missing} (lifecycle=retired)"
                )
        if violations:
            pytest.fail(
                "Iron rule #11: retired patches lack full provenance.\n"
                "Fix: add `superseded_by` + `vllm_version_range` OR add\n"
                "to `_RETIRED_NO_SUPERSEDE_WAIVER` with explanation.\n\n"
                + "\n".join(violations)
            )

    def test_superseded_patches_have_pin_gate(self):
        """Every patch with superseded_by must declare vllm_version_range
        (formalizes the supersession boundary), EXCEPT legacy auto-apply
        patches whose synthetic env_flag bypasses pin-gate anyway."""
        entries = _load_registry_entries()
        violations: list[str] = []
        for pid, body in entries.items():
            if not _has_field(body, "superseded_by"):
                continue
            if pid in _LEGACY_PIN_GATE_WAIVER:
                continue
            if not _has_field(body, "vllm_version_range"):
                violations.append(
                    f"  {pid}: superseded_by present, vllm_version_range "
                    f"missing (lifecycle={_lifecycle(body)!r})"
                )
        if violations:
            pytest.fail(
                "Iron rule #11: patches with superseded_by lack pin-gate.\n"
                "Fix: add `vllm_version_range: '<0.20.2rc1.devN'` OR\n"
                "add to `_LEGACY_PIN_GATE_WAIVER` (legacy auto-apply only).\n\n"
                + "\n".join(violations)
            )

    def test_pn13_specific_provenance(self):
        """PN13 was retired with pin-gate but missing superseded_by note —
        regression sentinel for the audit that flagged it 2026-05-11."""
        entries = _load_registry_entries()
        assert "PN13" in entries
        body = entries["PN13"]
        if _lifecycle(body) == "retired":
            assert _has_field(body, "vllm_version_range"), (
                "PN13: lifecycle=retired without vllm_version_range"
            )
            assert _has_field(body, "superseded_by"), (
                "PN13: lifecycle=retired without superseded_by — add note "
                "naming PR #41235 (merged 2026-04-29, in vllm 0.20.2)"
            )

    def test_waiver_constants_match_registry(self):
        """Waivers should only list patches that actually exist in registry —
        drift detector for waiver lists themselves."""
        entries = _load_registry_entries()
        for pid in _RETIRED_NO_SUPERSEDE_WAIVER:
            assert pid in entries, (
                f"{pid} in _RETIRED_NO_SUPERSEDE_WAIVER but not in PATCH_REGISTRY"
            )
            assert _lifecycle(entries[pid]) == "retired", (
                f"{pid} in waiver but lifecycle != 'retired' "
                f"(actual: {_lifecycle(entries[pid])!r})"
            )
        # Phase 5.3.C (2026-05-22): _LEGACY_PIN_GATE_WAIVER now covers
        # two sub-classes (see waiver definition above):
        #   (a) lifecycle='legacy' auto-apply patches
        #   (b) lifecycle='retired' patches whose superseded_by is an
        #       internal architecture decision (free-form string)
        for pid in _LEGACY_PIN_GATE_WAIVER:
            assert pid in entries, (
                f"{pid} in _LEGACY_PIN_GATE_WAIVER but not in PATCH_REGISTRY"
            )
            lc = _lifecycle(entries[pid])
            assert lc in ("legacy", "retired"), (
                f"{pid} in pin-gate waiver but lifecycle is {lc!r}; "
                f"expected 'legacy' (auto-apply class) or 'retired' "
                f"(internal-architecture supersession class)"
            )
