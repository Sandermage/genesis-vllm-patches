# SPDX-License-Identifier: Apache-2.0
"""PN133 v2 — MTP scheduler empty-output fix + #45060 observability arm.

Contract pinned here (TDD, written before the v2 implementation; PN133
v1 had no dedicated test file — this closes that gap as part of the
PN378 coordination work, roadmap chunk-3 Theme A):
  1. The replacement keeps the #42722 accounting fix byte-form
     (``max(len(generated_token_ids) - 1, 0)``) AND adds a
     ``logger.error`` arm on the empty-row condition — the invariant
     violation #45060's scheduler half ASSERTS on. We deliberately do
     NOT vendor the assert: PROD must degrade loudly, not crash the
     engine core (an all-NaN forward already produced garbage; killing
     the engine on top of it converts one bad request into an outage).
  2. Anchor/replacement texts are module-level constants (hoisted in
     v2 for testability and pristine-pin byte-verification).
  3. Drift markers carry BOTH upstream forms: #42722's accounting fix
     and #45060's scheduler assert (indented exact line). The #42722
     marker is a substring of our own replacement BY DESIGN — PN133's
     custom apply() defends it with the ``"PN133" not in content``
     guard (and PN133 builds its patcher inline, so the
     lint_drift_markers builder enumeration does not pick it up).
  4. apply() on the pin form installs the log.error arm + accounting
     fix and the file still compiles; re-apply is idempotent; apply on
     either upstream merged form self-skips without touching the file.
  5. The log.error message references the kernel-half coordination
     (PN378 / vllm#45060) so an operator grepping the error lands on
     the patch pair.
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
    pn133_mtp_scheduler_empty_output as m,
)

_ENV = "GENESIS_ENABLE_PN133_MTP_EMPTY_OUTPUT_FIX"

# ── Fake targets ─────────────────────────────────────────────────────
# Pin-form (g303916e93): the update_from_output spec-accounting block
# in v1/core/sched/scheduler.py — byte-faithful copy of the anchor
# region (pristine lines 1409-1427; logger exists at module level in
# the real file, line 62).

PIN_SCHED = (
    "# fake v1/core/sched/scheduler.py (pin g303916e93 form)\n"
    "logger = init_logger(__name__)\n"
    "\n"
    "\n"
    "class Scheduler:\n"
    "    def update_from_output(self, scheduler_output,"
    " model_runner_output):\n"
    "        sampled_token_ids = model_runner_output.sampled_token_ids\n"
    "        spec_decoding_stats = None\n"
    "        for req_id in model_runner_output.req_ids:\n"
    "            request = self.requests.get(req_id)\n"
    "            if request is None or request.is_finished():\n"
    "                continue\n"
    "\n"
    "            req_index = model_runner_output.req_id_to_index[req_id]\n"
    "            generated_token_ids = (\n"
    "                sampled_token_ids[req_index] if sampled_token_ids"
    " else []\n"
    "            )\n"
    "\n"
    "            scheduled_spec_token_ids = (\n"
    "                scheduler_output.scheduled_spec_decode_tokens.get"
    "(req_id)\n"
    "            )\n"
    "            if scheduled_spec_token_ids and generated_token_ids:\n"
    "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
    "                num_accepted = len(generated_token_ids) - 1\n"
    "                num_rejected = num_draft_tokens - num_accepted\n"
    "                if request.num_computed_tokens > 0:\n"
    "                    request.num_computed_tokens -= num_rejected\n"
)

# #45060 scheduler-half merged form — PN133 must self-skip on this
# (the empty row can no longer reach the accounting; the assert owns
# the invariant upstream). Exact hunk from `gh pr diff 45060`.

MERGED_45060_SCHED = PIN_SCHED.replace(
    "            if scheduled_spec_token_ids and generated_token_ids:\n"
    "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
    "                num_accepted = len(generated_token_ids) - 1\n",
    "            if scheduled_spec_token_ids:\n"
    "                # A scheduled-spec request always commits at least"
    " one token,\n"
    "                # since `sample_recovered_tokens_kernel` never"
    " emits an\n"
    "                # out-of-vocab id. Assert the invariant rather than"
    " silently\n"
    "                # skipping accounting, which would leave"
    " `num_computed_tokens`\n"
    "                # too high and could stall the request.\n"
    "                assert generated_token_ids\n"
    "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
    "                num_accepted = len(generated_token_ids) - 1\n",
).replace("(pin g303916e93 form)", "(post-vllm#45060 merged form)")

# #42722 merged form — the PR PN133 backports; self-skip as before.

MERGED_42722_SCHED = PIN_SCHED.replace(
    "            if scheduled_spec_token_ids and generated_token_ids:\n"
    "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
    "                num_accepted = len(generated_token_ids) - 1\n",
    "            if scheduled_spec_token_ids:\n"
    "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
    "                num_accepted = max(len(generated_token_ids) - 1, 0)\n",
).replace("(pin g303916e93 form)", "(post-vllm#42722 merged form)")

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm/v1/core/sched")

ASSERT_LINE = "                assert generated_token_ids\n"


# ── Helpers ──────────────────────────────────────────────────────────


def _install_fake(tmp_path, monkeypatch, sched_text, env="1"):
    target = tmp_path / "scheduler.py"
    target.write_text(sched_text, encoding="utf-8")
    # PN133 imports resolve_vllm_file inside apply() — patch the guards
    # module attribute, not a module-level rebind.
    import sndr.engines.vllm.detection.guards as guards
    monkeypatch.setattr(guards, "resolve_vllm_file", lambda rel: str(target))
    if env is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, env)
    monkeypatch.setattr(m, "_APPLIED", False)
    return target


# ── v2 contract: constants + replacement shape ───────────────────────


class TestV2ReplacementShape:
    def test_anchor_and_replacement_are_module_constants(self):
        assert isinstance(m.PN133_OLD, str)
        assert isinstance(m.PN133_NEW, str)
        assert m.PN133_OLD != m.PN133_NEW

    def test_replacement_emits_log_error_on_empty_row(self):
        """#45060 observability half: empty generated_token_ids with
        scheduled spec tokens must log.error (NOT assert)."""
        assert "if not generated_token_ids:" in m.PN133_NEW
        assert "logger.error(" in m.PN133_NEW

    def test_replacement_does_not_take_the_assert_half(self):
        assert "assert generated_token_ids" not in m.PN133_NEW

    def test_replacement_keeps_42722_accounting_fix(self):
        assert "max(len(generated_token_ids) - 1, 0)" in m.PN133_NEW
        assert "if scheduled_spec_token_ids:" in m.PN133_NEW

    def test_log_error_references_kernel_half_pair(self):
        """Operator greps the error -> must land on the patch pair
        (PN378 kernel mask / vllm#45060)."""
        assert "PN378" in m.PN133_NEW
        assert "45060" in m.PN133_NEW

    def test_drift_markers_carry_both_upstream_forms(self):
        assert "max(len(generated_token_ids) - 1, 0)" in m._DRIFT_MARKERS
        assert ASSERT_LINE in m._DRIFT_MARKERS

    def test_45060_marker_not_in_own_replacement(self):
        """The #45060 marker must never be emitted by PN133 itself —
        only the #42722 marker is self-substring (defended by the
        custom ``"PN133" not in content`` guard in apply())."""
        assert ASSERT_LINE not in m.PN133_NEW

    def test_docstring_documents_45060_coordination(self):
        doc = m.__doc__ or ""
        assert "45060" in doc
        assert "PN378" in doc
        assert "42722" in doc


# ── Apply semantics ──────────────────────────────────────────────────


class TestApply:
    def test_apply_pin_form_installs_log_error_and_accounting(
        self, tmp_path, monkeypatch
    ):
        target = _install_fake(tmp_path, monkeypatch, PIN_SCHED)
        status, reason = m.apply()
        assert status == "applied", reason

        out = target.read_text(encoding="utf-8")
        # Accounting fix (the #42722 half) installed.
        assert "num_accepted = max(len(generated_token_ids) - 1, 0)" in out
        # Observability arm installed BEFORE the accounting lines.
        assert (
            out.index("if not generated_token_ids:")
            < out.index("num_draft_tokens = len(scheduled_spec_token_ids)")
        )
        assert "logger.error(" in out
        # The assert half is NOT vendored.
        assert "assert generated_token_ids" not in out
        # File still compiles after the splice.
        compile(out, str(target), "exec")

    def test_second_apply_is_idempotent(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, PIN_SCHED)
        first_status, first_reason = m.apply()
        assert first_status == "applied", first_reason
        # PN133's custom marker short-circuit reports applied/idempotent
        # (pre-dispatcher-era convention, kept for prod-compose
        # compatibility — all 4 prod composes enable PN133).
        second_status, second_reason = m.apply()
        assert second_status == "applied"
        assert "idempotent" in second_reason

    def test_self_skips_on_45060_scheduler_merged_form(
        self, tmp_path, monkeypatch
    ):
        target = _install_fake(tmp_path, monkeypatch, MERGED_45060_SCHED)
        status, reason = m.apply()
        assert status == "skipped"
        assert "upstream_merged" in reason
        assert target.read_text(encoding="utf-8") == MERGED_45060_SCHED

    def test_self_skips_on_42722_merged_form(self, tmp_path, monkeypatch):
        target = _install_fake(tmp_path, monkeypatch, MERGED_42722_SCHED)
        status, reason = m.apply()
        assert status == "skipped"
        assert "upstream_merged" in reason
        assert target.read_text(encoding="utf-8") == MERGED_42722_SCHED

    def test_own_output_never_false_fires_drift(self, tmp_path, monkeypatch):
        """PN369-class regression guard: after PN133 lands its own
        ``max(len(generated_token_ids) - 1, 0)`` line, a re-apply must
        report idempotent — NOT upstream_merged (the ``"PN133" not in
        content`` guard defends the self-substring marker)."""
        _install_fake(tmp_path, monkeypatch, PIN_SCHED)
        status, reason = m.apply()
        assert status == "applied", reason
        monkeypatch.setattr(m, "_APPLIED", False)  # fresh-boot simulation
        status2, reason2 = m.apply()
        assert (status2, "upstream_merged" in reason2) == ("applied", False)
        assert "idempotent" in reason2

    def test_apply_skips_when_env_unset(self, tmp_path, monkeypatch):
        target = _install_fake(tmp_path, monkeypatch, PIN_SCHED, env=None)
        status, reason = m.apply()
        assert status == "skipped"
        assert _ENV in reason
        assert target.read_text(encoding="utf-8") == PIN_SCHED


# ── Pristine pin invariants (opportunistic) ──────────────────────────


@pytest.mark.skipif(
    not (PIN_TREE / "scheduler.py").is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAnchorsAgainstPristinePin:
    def test_anchor_unique_replacement_and_45060_marker_absent(self):
        src = (PIN_TREE / "scheduler.py").read_text(encoding="utf-8")
        assert src.count(m.PN133_OLD) == 1
        assert m.PN133_NEW not in src
        assert ASSERT_LINE not in src
        assert "max(len(generated_token_ids) - 1, 0)" not in src

    def test_fixture_anchor_region_byte_matches_pristine(self):
        src = (PIN_TREE / "scheduler.py").read_text(encoding="utf-8")
        assert m.PN133_OLD in PIN_SCHED
        assert m.PN133_OLD in src

    def test_logger_and_req_id_in_scope_at_anchor(self):
        """The v2 replacement emits ``logger.error(... req_id ...)`` —
        both names must exist in the pristine file (module-level logger;
        req_id is the loop variable enclosing the anchor)."""
        src = (PIN_TREE / "scheduler.py").read_text(encoding="utf-8")
        assert "logger = init_logger(__name__)" in src
        assert "for req_id in " in src
