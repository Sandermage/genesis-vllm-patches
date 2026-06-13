# SPDX-License-Identifier: Apache-2.0
"""TDD for PN381 — vendor of OPEN PR vllm#44742 (allowed_token_ids
metadata hardening for spec-decode), PN67-playbook style.

Upstream context (GHSA-8c65-hq7q-r7jm): when a request sets ONLY
``allowed_token_ids`` (no penalties / bad words / thinking budget),
``InputBatch._make_sampling_metadata`` ships
``output_token_ids == []`` while ``allowed_token_ids_mask`` and the
speculative draft-token counts are non-empty. A consumer that derives
the request count from ``len(output_token_ids)`` then mis-expands the
mask rows (the 0.17.1 worker crash). Our pin's consumer
(``RejectionSampler.apply_logits_processors``) already sizes by
``len(metadata.num_draft_tokens)`` (#35654), so PN381 is producer-side
defense-in-depth: populate ``output_token_ids`` whenever any request
uses ``allowed_token_ids``, keeping the metadata row counts
self-consistent for EVERY consumer — including the PN369/P71 rewritten
rejection-sampler paths.

The vendor is a single anchored condition added to
``needs_output_token_ids`` in ``v1/worker/gpu_input_batch.py`` (the
exact #44742 hunk). Our emitted text deliberately parenthesizes the
clause (``or (not self.no_allowed_token_ids)``) so the drift marker for
the PR's merged form (``or not self.no_allowed_token_ids``) can never
match our own output (tools/lint_drift_markers.py contract).

These tests verify textually (portable embedded fixtures shaped like
pin 0.22.1rc1.dev259+g303916e93) and opportunistically against the
real pristine tree at /private/tmp/candidate_pin_current:
  1. anchor unique on the embedded fixture AND the real pin file
  2. end-to-end TextPatcher apply on tmp copies — APPLIED, ast-valid,
     idempotent on second apply
  3. behavioral contract of the PATCHED expression (torch-free exec):
     allowed_token_ids present  -> output_token_ids populated;
     allowed_token_ids absent   -> output_token_ids stays [] (the
     upstream fast path is preserved)
  4. drift-marker discipline: merged-form marker disjoint from our
     replacement and from the idempotency marker line; absent from the
     pristine pin
  5. module apply() contract — env gate, missing target, drift skip

The CUDA regression test SHAPE from #44742 (allowed_token_ids + draft
tokens through apply_logits_processors) is ported separately in
``test_pn381_sampler_regression_torch.py`` (torch group, auto-skipped
on torch-less hosts, CPU-runnable on the rig).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm")
PIN_INPUT_BATCH = PIN_TREE / "v1" / "worker" / "gpu_input_batch.py"


def _pn381():
    from sndr.engines.vllm.patches.spec_decode import (
        pn381_allowed_token_ids_spec_metadata as M,
    )
    return M


# ─────────────────────────────────────────────────────────────────────
# Portable fixture — pristine-shaped region (pin g303916e93)
# ─────────────────────────────────────────────────────────────────────

# Verbatim pristine `_make_sampling_metadata` needs_output_token_ids
# region (gpu_input_batch.py lines 877-893 at the pin). PN381's anchor
# is the 6-line `needs_output_token_ids = (...)` block; the surrounding
# lines stay attached so the behavioral exec fixture below is
# meaningful.
NEEDS_OUTPUT_REGION_PRISTINE = (
    "        # Only set output_token_ids if required by the current requests'\n"
    "        # sampling parameters.\n"
    "        holder = self.thinking_budget_state_holder\n"
    "        thinking_budget_tracks_reqs = (\n"
    "            holder is not None and holder.has_tracked_requests()\n"
    "        )\n"
    "        needs_output_token_ids = (\n"
    "            not self.no_penalties\n"
    "            or bool(self.bad_words_token_ids)\n"
    "            or self.logitsprocs_need_output_token_ids\n"
    "            or thinking_budget_tracks_reqs\n"
    "        )\n"
    "        output_token_ids = (\n"
    "            cast(list[list[int]], self.req_output_token_ids)\n"
    "            if needs_output_token_ids\n"
    "            else []\n"
    "        )\n"
)


def _fake_pristine_input_batch() -> str:
    """Minimal ast-valid gpu_input_batch.py carrying the PN381 anchor
    region inside a callable ``_make_sampling_metadata`` so the
    behavioral exec test can drive the PATCHED expression with stub
    request states (torch-free)."""
    return (
        "# fake gpu_input_batch.py - pristine-shaped region (pin g303916e93)\n"
        "from typing import cast\n"
        "\n"
        "\n"
        "class InputBatch:\n"
        "    def __init__(\n"
        "        self,\n"
        "        no_penalties=True,\n"
        "        bad_words_token_ids=(),\n"
        "        logitsprocs_need_output_token_ids=False,\n"
        "        thinking_budget_state_holder=None,\n"
        "        no_allowed_token_ids=True,\n"
        "        req_output_token_ids=None,\n"
        "    ):\n"
        "        self.no_penalties = no_penalties\n"
        "        self.bad_words_token_ids = dict(bad_words_token_ids)\n"
        "        self.logitsprocs_need_output_token_ids = (\n"
        "            logitsprocs_need_output_token_ids\n"
        "        )\n"
        "        self.thinking_budget_state_holder = thinking_budget_state_holder\n"
        "        self.no_allowed_token_ids = no_allowed_token_ids\n"
        "        self.req_output_token_ids = req_output_token_ids or []\n"
        "\n"
        "    def _make_sampling_metadata(self):\n"
        + NEEDS_OUTPUT_REGION_PRISTINE
        + "        return output_token_ids\n"
    )


def _patcher_on(tmp_path: Path, content: str, monkeypatch):
    M = _pn381()
    target = tmp_path / "gpu_input_batch.py"
    target.write_text(content, encoding="utf-8")
    monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: str(target))
    patcher = M._make_patcher()
    assert patcher is not None
    return patcher, target


def _exec_patched_class(patched_source: str):
    """Exec the patched fake module and return its InputBatch class."""
    namespace: dict = {}
    exec(compile(patched_source, "<pn381_fixture>", "exec"), namespace)
    return namespace["InputBatch"]


# ─────────────────────────────────────────────────────────────────────
# 1. Anchor — unique on the fixture, replacement faithful to #44742
# ─────────────────────────────────────────────────────────────────────


class TestAnchor:
    def test_anchor_matches_fixture_exactly_once(self):
        M = _pn381()
        assert _fake_pristine_input_batch().count(M.PN381_OLD) == 1

    def test_anchor_is_substring_of_pristine_region(self):
        M = _pn381()
        assert M.PN381_OLD in NEEDS_OUTPUT_REGION_PRISTINE

    def test_replacement_adds_allowed_token_ids_clause(self):
        """Faithful #44742 semantics: the new OR-clause keys off
        ``self.no_allowed_token_ids`` (False when any request set
        allowed_token_ids)."""
        M = _pn381()
        assert "no_allowed_token_ids" in M.PN381_NEW
        # Parenthesized emitted form — disjoint from the PR's merged
        # form by construction (lint_drift_markers contract).
        assert "or (not self.no_allowed_token_ids)" in M.PN381_NEW
        assert "or not self.no_allowed_token_ids" not in M.PN381_NEW

    def test_replacement_preserves_all_pristine_clauses(self):
        """The vendor only ADDS a clause — every upstream condition
        survives byte-identical."""
        M = _pn381()
        for clause in (
            "            not self.no_penalties\n",
            "            or bool(self.bad_words_token_ids)\n",
            "            or self.logitsprocs_need_output_token_ids\n",
            "            or thinking_budget_tracks_reqs\n",
        ):
            assert clause in M.PN381_NEW, clause

    def test_replacement_does_not_resurrect_anchor(self):
        """Sequential-apply safety: the replacement must not contain
        the anchor or a second apply would double-fire."""
        M = _pn381()
        assert M.PN381_OLD not in M.PN381_NEW


# ─────────────────────────────────────────────────────────────────────
# 2. End-to-end TextPatcher apply on tmp copies
# ─────────────────────────────────────────────────────────────────────


class TestEndToEndApply:
    def test_applies_on_pristine_fixture(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        patcher, target = _patcher_on(
            tmp_path, _fake_pristine_input_batch(), monkeypatch
        )
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, failure
        out = target.read_text(encoding="utf-8")
        ast.parse(out)
        assert "[Genesis PN381" in out
        assert "or (not self.no_allowed_token_ids)" in out

    def test_idempotent_on_second_apply(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        M = _pn381()
        patcher, _target = _patcher_on(
            tmp_path, _fake_pristine_input_batch(), monkeypatch
        )
        result, _ = patcher.apply()
        assert result == TextPatchResult.APPLIED
        second = M._make_patcher()
        result2, _ = second.apply()
        assert result2 == TextPatchResult.IDEMPOTENT

    def test_skips_on_merged_form(self, tmp_path, monkeypatch):
        """A file already carrying the PR's merged form (upstream
        landed #44742) must SKIP via the drift marker — never
        double-add the clause."""
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        M = _pn381()
        merged = _fake_pristine_input_batch().replace(
            "            or thinking_budget_tracks_reqs\n",
            "            or thinking_budget_tracks_reqs\n"
            "            or not self.no_allowed_token_ids\n",
            1,
        )
        assert "or not self.no_allowed_token_ids" in merged
        patcher, target = _patcher_on(tmp_path, merged, monkeypatch)
        result, failure = patcher.apply()
        assert result == TextPatchResult.SKIPPED, failure
        # Untouched file — no Genesis text injected next to upstream's.
        assert "[Genesis PN381" not in target.read_text(encoding="utf-8")

    def test_skips_when_anchor_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        patcher, _ = _patcher_on(
            tmp_path, "def unrelated():\n    return 0\n", monkeypatch
        )
        result, _failure = patcher.apply()
        assert result == TextPatchResult.SKIPPED


# ─────────────────────────────────────────────────────────────────────
# 3. Behavioral contract of the patched expression (torch-free exec)
# ─────────────────────────────────────────────────────────────────────


class TestPatchedBehavior:
    """Exec-patched-text technique: drive the REAL patched expression
    with stub request states. This is the producer-side contract of
    GHSA-8c65-hq7q-r7jm, runnable without torch."""

    def _patched_class(self, tmp_path, monkeypatch):
        from sndr.kernel import TextPatchResult

        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        patcher, target = _patcher_on(
            tmp_path, _fake_pristine_input_batch(), monkeypatch
        )
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, failure
        return _exec_patched_class(target.read_text(encoding="utf-8"))

    def test_allowed_token_ids_populates_output_token_ids(
        self, tmp_path, monkeypatch
    ):
        """THE #44742 regression shape, producer side: only
        allowed_token_ids set (no penalties / bad words / thinking)
        must now populate output_token_ids."""
        InputBatch = self._patched_class(tmp_path, monkeypatch)
        batch = InputBatch(
            no_allowed_token_ids=False,
            req_output_token_ids=[[1, 2, 10, 5], [10, 5, 10, 5]],
        )
        assert batch._make_sampling_metadata() == [[1, 2, 10, 5], [10, 5, 10, 5]]

    def test_pristine_behavior_was_the_bug(self, tmp_path, monkeypatch):
        """RED-documenting twin: the UNPATCHED expression ships [] for
        the same state — the exact GHSA inconsistency PN381 removes."""
        InputBatch = _exec_patched_class(_fake_pristine_input_batch())
        batch = InputBatch(
            no_allowed_token_ids=False,
            req_output_token_ids=[[1, 2, 10, 5]],
        )
        assert batch._make_sampling_metadata() == []

    def test_no_allowed_token_ids_keeps_fast_path(self, tmp_path, monkeypatch):
        """No request uses allowed_token_ids -> the upstream
        copy-avoidance fast path survives (output_token_ids stays [])."""
        InputBatch = self._patched_class(tmp_path, monkeypatch)
        batch = InputBatch(
            no_allowed_token_ids=True,
            req_output_token_ids=[[1, 2, 3]],
        )
        assert batch._make_sampling_metadata() == []

    def test_other_clauses_still_populate(self, tmp_path, monkeypatch):
        """The added clause must not perturb the pristine ones."""
        InputBatch = self._patched_class(tmp_path, monkeypatch)
        for kwargs in (
            {"no_penalties": False},
            {"bad_words_token_ids": {0: [[7]]}.items()},
            {"logitsprocs_need_output_token_ids": True},
        ):
            batch = InputBatch(req_output_token_ids=[[9]], **kwargs)
            assert batch._make_sampling_metadata() == [[9]], kwargs


# ─────────────────────────────────────────────────────────────────────
# 4. Self-collision invariants (tools/lint_drift_markers.py contract)
# ─────────────────────────────────────────────────────────────────────


class TestSelfCollision:
    def test_drift_markers_disjoint_from_emitted_text(self):
        M = _pn381()
        marker_line = f"# [Genesis wiring marker: {M.GENESIS_PN381_MARKER}]\n"
        for dm in M._DRIFT_MARKERS:
            if dm.startswith("[Genesis"):
                continue  # defended convention — exempt from the lint
            assert dm not in M.PN381_NEW, dm
            assert dm not in marker_line, dm

    def test_merged_form_marker_present(self):
        """The marker list must carry the PR's merged form so the
        patcher self-skips when #44742 lands at a future pin."""
        M = _pn381()
        assert "or not self.no_allowed_token_ids" in M._DRIFT_MARKERS

    def test_drift_markers_absent_from_pristine_fixture(self):
        M = _pn381()
        src = _fake_pristine_input_batch()
        for dm in M._DRIFT_MARKERS:
            assert dm not in src, dm


# ─────────────────────────────────────────────────────────────────────
# 5. Module apply() contract — env gate, combined statuses
# ─────────────────────────────────────────────────────────────────────


class TestModuleApply:
    def test_skips_when_env_unset(self, monkeypatch):
        M = _pn381()
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN381_ALLOWED_TOKEN_IDS_METADATA", raising=False
        )
        status, detail = M.apply()
        assert status == "skipped"
        assert "GENESIS_ENABLE_PN381_ALLOWED_TOKEN_IDS_METADATA" in detail

    def test_applies_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN381_ALLOWED_TOKEN_IDS_METADATA", "1"
        )
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        M = _pn381()
        target = tmp_path / "gpu_input_batch.py"
        target.write_text(_fake_pristine_input_batch(), encoding="utf-8")
        monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: str(target))
        monkeypatch.setattr(M, "vllm_install_root", lambda: str(tmp_path))
        status, detail = M.apply()
        assert status == "applied", detail
        assert "44742" in detail
        ast.parse(target.read_text(encoding="utf-8"))
        assert M.is_applied()

    def test_skips_when_target_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN381_ALLOWED_TOKEN_IDS_METADATA", "1"
        )
        M = _pn381()
        monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: None)
        monkeypatch.setattr(M, "vllm_install_root", lambda: str(tmp_path))
        status, _ = M.apply()
        assert status == "skipped"

    def test_marker_tracks_upstream_pr(self):
        M = _pn381()
        assert "44742" in M.GENESIS_PN381_MARKER


# ─────────────────────────────────────────────────────────────────────
# 6. Anchors against the real pristine pin (opportunistic)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not PIN_INPUT_BATCH.is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAnchorsAgainstPristinePin:
    def test_anchor_unique_and_markers_absent(self):
        M = _pn381()
        src = PIN_INPUT_BATCH.read_text(encoding="utf-8")
        assert src.count(M.PN381_OLD) == 1
        assert M.PN381_NEW not in src
        for dm in M._DRIFT_MARKERS:
            assert dm not in src, dm

    def test_region_fixture_matches_pin(self):
        """The embedded portable region must stay byte-identical to
        the pin so the behavioral exec tests keep testifying about the
        real file shape."""
        src = PIN_INPUT_BATCH.read_text(encoding="utf-8")
        assert src.count(NEEDS_OUTPUT_REGION_PRISTINE) == 1

    def test_full_file_apply_and_compile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        M = _pn381()
        target = tmp_path / "gpu_input_batch.py"
        target.write_text(
            PIN_INPUT_BATCH.read_text(encoding="utf-8"), encoding="utf-8"
        )
        monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: str(target))
        patcher = M._make_patcher()
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, failure
        out = target.read_text(encoding="utf-8")
        ast.parse(out)
        assert out.count("or (not self.no_allowed_token_ids)") == 1
