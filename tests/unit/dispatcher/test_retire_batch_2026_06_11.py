# SPDX-License-Identifier: Apache-2.0
"""Retire batch 2026-06-11 — preflight residual triage §2 hygiene + §3 retires.

Pins the registry/metadata state mandated by
docs/superpowers/journal/2026-06-11-preflight-residual-triage-action-plan.md:

  §3 retire queue (iron-rule-#11, byte-level evidence per patch):
    P7b, PN54, P78, P36, P83, P84, P4, P20, P6 — lifecycle="retired",
    provenance fields present, wiring modules moved to the engine
    `_archive/` corpus (P20 is marker_only — registry-only entry).

  §2 hygiene:
    P26  — superseded_by reworded to PARTIAL (guards against a future
           title-matching retire of a live perf win).
    PN71 — requires_patches=["P27"] (anchor contains P27-injected text).
    PN346 — registry default_on=True (module honors only
           GENESIS_DISABLE_PN346 → effectively default-ON in practice).
    PN353B — P78 dropped from composes_with (P78 retired).
    PN200 — pending-decision note recorded (NOT silently re-anchored);
            decision EXECUTED later the same night: retired, superseded
            by PROD-applied P28 which owns the unique forward_cuda site
            (its anchor textually contains PN200's whole anchor) and
            delivers the same buffer-reuse+zero contract. See
            TestPN200DecisionExecuted.

  §6 upstream_compat:
    P5 Probe 1 — TQFullAttentionSpec probed at its real home
    (vllm.v1.kv_cache_interface), not turboquant.config where the
    hasattr() was always False.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO_ROOT / "sndr" / "dispatcher" / "registry.py"
ARCHIVE_DIR = REPO_ROOT / "sndr" / "engines" / "vllm" / "_archive"

# (patch_id, archived module stem or None for marker_only/registry-only)
RETIRE_BATCH: list[tuple[str, str | None]] = [
    ("P7b", "p7b_gdn_dual_stream_customop"),
    ("PN54", "pn54_gdn_contiguous_dedup"),
    ("P78", "p78_tolist_capture_guard"),
    ("P36", "p36_tq_shared_decode_buffers"),
    ("P83", "p83_mtp_keep_last_cached_block"),
    ("P84", "p84_hash_block_size_override"),
    ("P4", "p4_tq_hybrid"),
    ("P20", None),  # marker_only — no wiring module ever existed
    ("P6", "p6_tq_block_size_align"),
]

# Old (pre-archive) module locations that must be empty after the move.
OLD_LOCATIONS = {
    "p7b_gdn_dual_stream_customop": "patches/attention/gdn",
    "pn54_gdn_contiguous_dedup": "patches/attention/gdn",
    "p78_tolist_capture_guard": "patches/attention/turboquant",
    "p36_tq_shared_decode_buffers": "patches/kernels",
    "p83_mtp_keep_last_cached_block": "patches/kv_cache",
    "p84_hash_block_size_override": "patches/scheduler",
    "p4_tq_hybrid": "patches/scheduler",
    "p6_tq_block_size_align": "patches/compile_safety",
}


def _entry_body(patch_id: str) -> str:
    text = REGISTRY_PATH.read_text()
    m = re.search(
        rf'^    "{patch_id}":\s*\{{(.*?)^    \}},', text, flags=re.M | re.S
    )
    assert m, f"{patch_id}: entry not found in PATCH_REGISTRY"
    return m.group(1)


class TestRetireQueueSection3:
    @pytest.mark.parametrize("patch_id,stem", RETIRE_BATCH)
    def test_lifecycle_retired(self, patch_id, stem):
        body = _entry_body(patch_id)
        m = re.search(r'"lifecycle"\s*:\s*"([^"]+)"', body)
        assert m and m.group(1) == "retired", (
            f"{patch_id}: lifecycle={m.group(1) if m else None!r}, "
            f"expected 'retired' (plan §3)"
        )

    @pytest.mark.parametrize("patch_id,stem", RETIRE_BATCH)
    def test_provenance_fields_present(self, patch_id, stem):
        """Iron rule #11: superseded_by + vllm_version_range both set."""
        body = _entry_body(patch_id)
        assert re.search(r'"superseded_by"\s*:', body), (
            f"{patch_id}: missing superseded_by"
        )
        assert re.search(r'"vllm_version_range"\s*:', body), (
            f"{patch_id}: missing vllm_version_range upper cap"
        )

    @pytest.mark.parametrize(
        "patch_id,stem",
        [(p, s) for p, s in RETIRE_BATCH if s is not None],
    )
    def test_module_archived(self, patch_id, stem):
        """Wiring file moved to _archive/ and apply_module repointed."""
        assert (ARCHIVE_DIR / f"{stem}.py").is_file(), (
            f"{patch_id}: {stem}.py not in {ARCHIVE_DIR}"
        )
        old = (
            REPO_ROOT / "sndr" / "engines" / "vllm"
            / OLD_LOCATIONS[stem] / f"{stem}.py"
        )
        assert not old.exists(), f"{patch_id}: stale copy left at {old}"
        body = _entry_body(patch_id)
        m = re.search(r'"apply_module"\s*:\s*"([^"]+)"', body)
        assert m and m.group(1) == f"sndr.engines.vllm._archive.{stem}", (
            f"{patch_id}: apply_module={m.group(1) if m else None!r} not "
            f"repointed to _archive"
        )

    def test_p83_retire_note_phrasing(self):
        """Plan-mandated phrasing: convergence-interaction cost +
        residual tail-block drop tracked via open #44986."""
        body = _entry_body("P83")
        assert "44986" in body, "P83: retire note must cite open #44986"

    def test_p84_mamba_backoff_caveat(self):
        """Verifier caveat: Mamba back-off precedes both GCD and the
        explicit hash_block_size override."""
        body = _entry_body("P84")
        assert re.search(r"[Mm]amba back-off", body), (
            "P84: retire note must carry the Mamba back-off caveat"
        )

    def test_p85_requires_p84_re_triage_note(self):
        """P84 retire cascades into P85's requires_patches — the fix
        itself is a follow-up, but the note must exist now."""
        body = _entry_body("P85")
        assert re.search(r"re-triage", body, flags=re.I), (
            "P85: missing re-triage note for requires_patches=['P84']"
        )

    def test_pn353b_composes_with_drops_p78(self):
        body = _entry_body("PN353B")
        m = re.search(r'"composes_with"\s*:\s*\[([^\]]*)\]', body)
        assert m, "PN353B: composes_with missing"
        assert '"P78"' not in m.group(1), (
            "PN353B: P78 must be dropped from composes_with (P78 retired)"
        )

class TestPN200DecisionExecuted:
    """§3 pending decision MADE 2026-06-11 (same night): retire
    (option a). Evidence: P28 (PROD-applied, default_on=True legacy
    auto-apply) owns the unique forward_cuda site — its anchor is the
    comment-disambiguated superset that textually CONTAINS PN200's
    entire anchor, and its replacement delivers the same buffer-reuse
    + explicit .zero_() (#28182 zero contract) + torch.zeros fallback.
    PN200's bare anchor is ambiguous on pin 0.22.1rc1.dev259 (3 matches
    pristine: forward_cuda:950 / forward_xpu:991 / forward_cpu:1046;
    2 post-P28 — both non-CUDA paths our 2x A5000 never executes), so
    it can never apply again; a P28-chain variant would only pool-route
    P28's eager fallback branch and reintroduce the in-forward env read
    CRIT-HW-1 forbids."""

    def test_pn200_lifecycle_and_provenance(self):
        body = _entry_body("PN200")
        m = re.search(r'"lifecycle"\s*:\s*"([^"]+)"', body)
        assert m and m.group(1) == "retired", (
            f"PN200: lifecycle={m.group(1) if m else None!r}, expected "
            "'retired' (decision executed 2026-06-11)"
        )
        sb = re.search(r'"superseded_by"\s*:\s*\(?\s*"(.*?)"', body, re.S)
        assert sb and "P28" in sb.group(1), (
            "PN200: superseded_by must name P28 (internal supersession)"
        )
        assert re.search(r'"vllm_version_range"\s*:', body), (
            "PN200: missing vllm_version_range pin-gate cap"
        )

    def test_pn200_module_archived(self):
        stem = "pn200_gdn_scratch_reuse"
        assert (ARCHIVE_DIR / f"{stem}.py").is_file(), (
            f"PN200: {stem}.py not in {ARCHIVE_DIR}"
        )
        old = (
            REPO_ROOT / "sndr" / "engines" / "vllm" / "patches"
            / "streaming" / f"{stem}.py"
        )
        assert not old.exists(), f"PN200: stale copy left at {old}"
        body = _entry_body("PN200")
        m = re.search(r'"apply_module"\s*:\s*"([^"]+)"', body)
        assert m and m.group(1) == f"sndr.engines.vllm._archive.{stem}", (
            f"PN200: apply_module={m.group(1) if m else None!r} not "
            "repointed to _archive"
        )

    def test_pn200_flag_removed_from_launchers(self):
        """Journal corollary: range-capping is NOT retirement while
        launchers still export the flag. Proper retire removes
        GENESIS_ENABLE_PN200_GDN_SCRATCH_REUSE from compose/prod-*.yml,
        the builtin model YAMLs, and the restart helper."""
        flag = "GENESIS_ENABLE_PN200_GDN_SCRATCH_REUSE"
        offenders: list[str] = []
        roots = [
            REPO_ROOT / "compose",
            REPO_ROOT / "sndr" / "model_configs" / "builtin",
            REPO_ROOT / "tools",
        ]
        for root in roots:
            for ext in ("*.yml", "*.yaml", "*.sh"):
                for f in root.rglob(ext):
                    if flag in f.read_text(encoding="utf-8",
                                           errors="ignore"):
                        offenders.append(str(f.relative_to(REPO_ROOT)))
        assert not offenders, (
            f"PN200 retired but {flag} still exported by: {offenders}"
        )


class TestHygieneSection2:
    def test_p26_superseded_by_partial_wording(self):
        body = _entry_body("P26")
        m = re.search(r'"superseded_by"\s*:\s*\(?\s*"(.*?)"', body, re.S)
        assert m and m.group(1).startswith("PARTIAL"), (
            "P26: superseded_by must start with 'PARTIAL' — guards "
            "against future title-matching retire (plan §2)"
        )

    def test_pn71_requires_p27(self):
        body = _entry_body("PN71")
        m = re.search(r'"requires_patches"\s*:\s*\[([^\]]*)\]', body)
        assert m and '"P27"' in m.group(1), (
            "PN71: requires_patches must list P27 (anchor contains "
            "P27-injected comments)"
        )

    def test_pn346_default_on_true(self):
        """Module honors only GENESIS_DISABLE_PN346 → effectively
        default-ON; registry must say so honestly."""
        body = _entry_body("PN346")
        m = re.search(r'"default_on"\s*:\s*(True|False)', body)
        assert m and m.group(1) == "True", (
            "PN346: registry default_on must be True (module is "
            "opt-out-only via GENESIS_DISABLE_PN346)"
        )

    def test_pn204_comment_does_not_call_p7_retired(self):
        """P7 is lifecycle='legacy' (boot-skipped, deferred) — PN204's
        comment must not call it 'retired'."""
        body = _entry_body("PN204")
        assert "retired P7" not in body, (
            "PN204: comment still calls P7 'retired'; P7 is legacy"
        )


class TestSection6ModuleFixes:
    def test_pn55_drift_marker_narrowed(self):
        """'init_fp8_kv_scales' name-collides with merged vllm#28783
        (gh-verified MERGED 2025-11-30) while #41602/#41896 are both
        OPEN — it must not be a PN55 drift marker."""
        src = (
            REPO_ROOT / "sndr" / "engines" / "vllm" / "patches" / "worker"
            / "pn55_wake_up_hybrid_kv.py"
        ).read_text()
        m = re.search(
            r"upstream_drift_markers\s*=\s*\[(.*?)\]", src, flags=re.S
        )
        assert m, "PN55: upstream_drift_markers list not found"
        assert "init_fp8_kv_scales" not in m.group(1), (
            "PN55: 'init_fp8_kv_scales' still in drift markers — "
            "false self-retire on every post-#28783 pin"
        )

    def test_p5_probe_uses_kv_cache_interface(self):
        """Probe 1 must import TQFullAttentionSpec's real home —
        vllm.v1.kv_cache_interface (pristine kv_cache_interface.py:327)
        — not turboquant.config where hasattr() is always False."""
        src = (
            REPO_ROOT / "sndr" / "engines" / "vllm" / "patches"
            / "kv_cache" / "p5_page_size.py"
        ).read_text()
        assert "vllm.v1.kv_cache_interface" in src, (
            "P5: Probe 1 still imports turboquant.config for "
            "TQFullAttentionSpec — the #39931 auto-skip never fires"
        )
