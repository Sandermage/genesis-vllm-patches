# SPDX-License-Identifier: Apache-2.0
"""PN390 — streaming-LSE rejection sampler (vendor of vllm#45369).

Contract pinned here (TDD, written alongside the implementation):
  1. Patcher carries 13 required sub-patches: the import HAS_TRITON add,
     the body softmax→LSE swap, three call-site swaps, the wrapper
     signature, the injected compute_target_lse + target_lse_kernel
     producer, and the two-kernel signature/load rewrites that
     reconstruct probabilities as exp(logit - lse).
  2. KERNEL/SAMPLER ONLY: the module never touches scheduler.py or any
     torch-side prob reader — PN369's torch-side prob read is explicitly
     out of scope (documented in the module).
  3. Deliberate spelling divergence for drift-marker hygiene: our body
     constant is GENESIS_PN390_LSE_BLOCK_SIZE (never the bare
     `BLOCK_SIZE: tl.constexpr = 8192`) and the LSE store is factored
     through a named `lse` intermediate (never the upstream one-liner),
     so #45369's exact lines stay usable as drift markers without
     colliding with our own emitted text (lint_drift_markers contract).
  4. apply() on the pin-form (g303916e93) installs all 13 sub-patches,
     removes the materialized target_probs softmax, and the result
     compiles.
  5. Second apply() is idempotent (marker short-circuit).
  6. apply() on #45369's merged form self-skips via drift markers
     (reason: upstream_merged) without touching the file.
  7. Drift markers do not collide with PN390's own replacement text or
     its Layer-6 marker line (tools/lint_drift_markers.py contract)
     AND at least one marker is an exact substring of the merged form.
  8. Anchors are unique and drift markers absent in the pristine pin
     tree (opportunistic — skipped when the pin tree is not present).
  9. The module documents the live exposure (MTP K=3, vocab 151936
     transient 3.6-14.6 MB), the PN378 line-orthogonality, the PN369
     out-of-scope note, and references GENESIS_ENABLE_PN390_STREAMING_LSE_SAMPLER.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# The lint/preflight tools disable the Layer-0 file cache the same way;
# unit tests patch fresh tmp files, so the cache must never satisfy
# apply() from a previous run's state.
os.environ.setdefault("GENESIS_NO_PATCH_CACHE", "1")

from sndr.engines.vllm.patches.spec_decode import (  # noqa: E402
    pn390_streaming_lse_rejection_sampler as m,
)

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm/v1/sample")
PIN_FILE = PIN_TREE / "rejection_sampler.py"

EXPECTED_SUB_NAMES = {
    "pn390_import_has_triton",
    "pn390_body_drop_softmax",
    "pn390_srt_call_site",
    "pn390_rrs_call_site",
    "pn390_srt_signature",
    "pn390_srt_kernel_launch",
    "pn390_inject_lse_producer",
    "pn390_rrs_kernel_signature",
    "pn390_rrs_kernel_load",
    "pn390_srtk_signature",
    "pn390_srtk_lse_preload",
    "pn390_srtk_nodraft_load",
    "pn390_srtk_draft_load",
}


# ── Helpers ──────────────────────────────────────────────────────────


def _build_merged_form(pin_src: str) -> str:
    """Synthesize #45369's merged form from the pristine source.

    Applies the upstream-form (NOT our Genesis spelling) of the two
    divergent lines the drift markers target, so a self-skip test binds
    the markers to the real merged shape. We only need the two marker
    lines present somewhere in the file for the drift gate to fire.
    """
    merged = pin_src.replace(
        "    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)\n",
        "    BLOCK_SIZE: tl.constexpr = 8192\n"
        "    target_lse = compute_target_lse(target_logits, vocab_size, BLOCK_SIZE)\n",
    )
    # Inject the upstream store one-liner near the producer insertion point
    # so the second drift marker is present in the merged form too.
    merged = merged.replace(
        "    return recovered_token_ids\n",
        "    return recovered_token_ids\n"
        "\n\n"
        "@triton.jit\n"
        "def target_lse_kernel(target_logits_ptr, target_lse_ptr, "
        "vocab_size, BLOCK_SIZE: tl.constexpr):\n"
        "    row = tl.program_id(0)\n"
        "    m = tl.full((), float(\"-inf\"), tl.float32)\n"
        "    s = tl.full((), 0.0, tl.float32)\n"
        "    tl.store(target_lse_ptr + row, m + tl.log(s))\n",
        1,
    )
    return merged


def _install_fake(tmp_path, monkeypatch, sampler_text):
    target = tmp_path / "rejection_sampler.py"
    target.write_text(sampler_text, encoding="utf-8")
    monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: str(target))
    # apply() is dispatcher-gated (opt-in env flag, registry-driven) —
    # force the gate open for unit tests of the patch mechanics.
    import sndr.dispatcher as dispatcher
    monkeypatch.setattr(
        dispatcher, "should_apply", lambda pid: (True, "test override")
    )
    return target


@pytest.fixture
def pin_src():
    if not PIN_FILE.is_file():
        pytest.skip("pristine pin tree not present on this machine")
    return PIN_FILE.read_text(encoding="utf-8")


# ── Patcher shape ────────────────────────────────────────────────────


class TestPatcherShape:
    def test_patcher_has_thirteen_required_subs(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, "placeholder")
        patcher = m._make_patcher()
        assert patcher is not None
        by_name = {sp.name: sp for sp in patcher.sub_patches}
        assert set(by_name) == EXPECTED_SUB_NAMES
        assert all(sp.required for sp in patcher.sub_patches)

    def test_block_size_constant_diverges_from_upstream(self):
        """Drift-marker hygiene: our body constant is Genesis-named and
        the store is a named intermediate, so the upstream forms never
        appear in our emitted replacement text."""
        emitted = m.PN390_BODY_NEW + m.PN390_INJECT_NEW
        assert "GENESIS_PN390_LSE_BLOCK_SIZE = 8192" in m.PN390_BODY_NEW
        assert "    lse = m + tl.log(s)\n" in m.PN390_INJECT_NEW
        # The EXACT drift-marker strings (the form the lint checks) must
        # never appear in our emitted text — that is the self-collision
        # guarantee. Use the canonical _DRIFT_MARKERS so this test binds
        # the same strings tools/lint_drift_markers.py compares.
        for dm in m._DRIFT_MARKERS:
            assert dm not in emitted, (
                f"drift marker {dm!r} self-collides with emitted text"
            )

    def test_no_scheduler_or_torch_prob_reader_touched(self):
        """Sampler/kernel scope only — no scheduler, no PN369 torch-side
        prob reader (PN369 is documented out of scope)."""
        src = Path(m.__file__).read_text(encoding="utf-8")
        assert m._TARGET_REL == "v1/sample/rejection_sampler.py"
        assert "core/sched/scheduler.py" not in src
        # PN369 must be named as out-of-scope, not patched.
        assert "PN369" in (m.__doc__ or "")

    def test_patcher_none_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        assert m._make_patcher() is None

    def test_module_documents_exposure_and_composition(self):
        doc = m.__doc__ or ""
        assert "45369" in doc
        assert "151936" in doc
        assert "MTP" in doc
        assert "PN378" in doc  # line-orthogonality
        assert "PN369" in doc  # out-of-scope note

    def test_module_references_registry_env_flag(self):
        src = Path(m.__file__).read_text(encoding="utf-8")
        assert "GENESIS_ENABLE_PN390_STREAMING_LSE_SAMPLER" in src


# ── Apply semantics ──────────────────────────────────────────────────


class TestApply:
    def test_apply_pin_form_installs_all_subs(
        self, tmp_path, monkeypatch, pin_src
    ):
        target = _install_fake(tmp_path, monkeypatch, pin_src)
        status, reason = m.apply()
        assert status == "applied", reason

        out = target.read_text(encoding="utf-8")
        # Materialized softmax buffer is gone (only comment mentions remain).
        assert "target_probs = target_logits.softmax" not in out
        # The fused producer + kernel are injected.
        assert "def compute_target_lse(" in out
        assert "def target_lse_kernel(" in out
        # The heavy arm reconstructs exp(logit - lse).
        assert "target_prob = tl.exp(target_logit - target_lse)" in out
        # No stray code reference to the removed buffer symbol.
        for line in out.splitlines():
            if "target_probs" in line:
                assert line.lstrip().startswith("#"), (
                    f"non-comment target_probs leftover: {line!r}"
                )
        # File still compiles after all 13 splices.
        compile(out, str(target), "exec")

    def test_second_apply_is_idempotent(self, tmp_path, monkeypatch, pin_src):
        _install_fake(tmp_path, monkeypatch, pin_src)
        first_status, first_reason = m.apply()
        assert first_status == "applied", first_reason
        second_status, second_reason = m.apply()
        assert second_status == "skipped"
        assert "already applied" in second_reason

    def test_self_skips_on_45369_merged_form(
        self, tmp_path, monkeypatch, pin_src
    ):
        merged = _build_merged_form(pin_src)
        target = _install_fake(tmp_path, monkeypatch, merged)
        status, reason = m.apply()
        assert status == "skipped"
        assert "upstream_merged" in reason
        # Self-skip must not modify the merged file.
        assert target.read_text(encoding="utf-8") == merged

    def test_apply_skips_when_gate_closed(self, tmp_path, monkeypatch, pin_src):
        """Opt-in patch (default_on=False): gate closed → no file touch."""
        target = tmp_path / "rejection_sampler.py"
        target.write_text(pin_src, encoding="utf-8")
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: str(target))
        import sndr.dispatcher as dispatcher
        monkeypatch.setattr(
            dispatcher, "should_apply", lambda pid: (False, "opt-in: env unset")
        )
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN390_STREAMING_LSE_SAMPLER", raising=False
        )
        status, _reason = m.apply()
        assert status == "skipped"
        assert target.read_text(encoding="utf-8") == pin_src

    def test_apply_skips_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        import sndr.dispatcher as dispatcher
        monkeypatch.setattr(
            dispatcher, "should_apply", lambda pid: (True, "test override")
        )
        status, _reason = m.apply()
        assert status == "skipped"


# ── Lint contract (tools/lint_drift_markers.py) ──────────────────────


class TestDriftMarkerSelfCollision:
    def test_markers_not_substring_of_own_emitted_text(
        self, tmp_path, monkeypatch
    ):
        _install_fake(tmp_path, monkeypatch, "placeholder")
        patcher = m._make_patcher()
        marker_line = f"# [Genesis wiring marker: {patcher.marker}]\n"
        assert patcher.upstream_drift_markers, "drift markers must exist"
        for dm in patcher.upstream_drift_markers:
            if dm.startswith("[Genesis"):
                continue  # defended convention — exempt
            for sp in patcher.sub_patches:
                assert dm not in sp.replacement, (
                    f"drift marker {dm!r} collides with {sp.name} "
                    "replacement — would false-fire Layer 3 (PN369 class)"
                )
            assert dm not in marker_line

    def test_markers_match_45369_merged_form(self, monkeypatch, pin_src):
        """Markers must actually fire on the real merged form."""
        merged = _build_merged_form(pin_src)
        patcher = m._make_patcher() if PIN_FILE.is_file() else None
        # _make_patcher needs a resolvable target; build directly off the
        # pin file so the markers list is real.
        markers = list(m._DRIFT_MARKERS)
        assert any(dm in merged for dm in markers)


# ── Pristine pin invariants (opportunistic) ──────────────────────────


@pytest.mark.skipif(
    not PIN_FILE.is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAnchorsAgainstPristinePin:
    def test_all_anchors_unique_in_pristine(self):
        src = PIN_FILE.read_text(encoding="utf-8")
        anchors = [
            m.PN390_IMPORT_OLD,
            m.PN390_BODY_OLD,
            m.PN390_SRT_CALL_OLD,
            m.PN390_RRS_CALL_OLD,
            m.PN390_SRT_SIG_OLD,
            m.PN390_SRT_KCALL_OLD,
            m.PN390_INJECT_OLD,
            m.PN390_RRS_SIG_OLD,
            m.PN390_RRS_LOAD_OLD,
            m.PN390_SRTK_SIG_OLD,
            m.PN390_SRTK_PRELOAD_OLD,
            m.PN390_SRTK_NODRAFT_OLD,
            m.PN390_SRTK_DRAFT_OLD,
        ]
        for a in anchors:
            assert src.count(a) == 1, f"anchor not unique (count): {a[:60]!r}"

    def test_drift_markers_absent_in_pristine(self):
        src = PIN_FILE.read_text(encoding="utf-8")
        for dm in m._DRIFT_MARKERS:
            assert dm not in src, f"drift marker present in pristine: {dm!r}"

    def test_replacements_absent_in_pristine(self):
        """Our marker + injected symbols are not in the pristine tree
        (idempotency / dup-application guard)."""
        src = PIN_FILE.read_text(encoding="utf-8")
        assert m.GENESIS_PN390_MARKER not in src
        assert "GENESIS_PN390_LSE_BLOCK_SIZE" not in src
        assert "def compute_target_lse(" not in src

    def test_live_exposure_block_size_8192_in_pin_wrapper(self):
        """The recovered-token wrapper hardcodes BLOCK_SIZE = 8192 and
        MTP K=3 runs the rejection path each decode step. If upstream
        changes the block size, re-derive the transient-MB exposure."""
        src = PIN_FILE.read_text(encoding="utf-8")
        assert "BLOCK_SIZE = 8192" in src
