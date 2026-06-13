# SPDX-License-Identifier: Apache-2.0
"""#44993 regression suite ported against OUR P62 grammar-advance surface.

Upstream vllm#44993 (OPEN, built on top of vllm#44297) fixes two
failure modes of ``should_advance`` under async scheduling + spec
decode + a Qwen-style reasoning parser:

  Bug 1 — the placeholder-derived delta window
  (``num_computed_tokens - num_output_placeholders``) skips the
  reasoning-end marker whenever some drafts are rejected, so
  ``reasoning_ended`` never flips and the grammar is bypassed.

  Bug 2 — post-marker content tokens produced in the marker step never
  reach the grammar FSM, so the next step's bitmask starts at the
  initial state and json_object responses emit duplicate opening
  tokens (``{{...}``).

Genesis P62 (vendor of vllm#36138, sibling approach) already covers
both bugs with a DIFFERENT mechanism: ``should_advance`` is kept
byte-identical as a compat wrapper, and the scheduler call sites are
rewired to ``update_reasoning_ended`` (Bug 1: exact accepted tokens as
the delta window) + ``identify_constrained_draft_tokens`` (Bug 2: only
the post-marker partition is fed to ``grammar.accept_tokens``). This
suite ports upstream's 4 regression tests onto that surface — per the
2026-06-11 roadmap (#44993 row: "Do NOT vendor code — port ... 4
regression tests against P62").

Test strategy (modeled on test_pn373_parallel_toolcalls_null.py):
  1. Anchor byte-verification against the pristine candidate tree
     (/private/tmp/candidate_pin_current) — count==1 for all 5 anchors.
  2. Exec-technique: P62's replacements are applied to the pristine
     source text, the patched methods are ast-extracted and exec'd, and
     the 4 ported regression tests run against THAT text (not a
     reimplementation). The scheduler test execs the patched
     update_from_output block verbatim.
  3. Pristine bug reproduction (#43388 scenario) — if it starts
     FAILING after a pin bump, upstream merged #44993.
  4. Drift-marker hygiene: the #44297/#44993 watch markers must be
     absent from pristine, absent from P62's own emitted text
     (tools/lint_drift_markers.py contract), and fire on the post-merge
     images captured from `gh pr diff` 2026-06-11.

The 50-trial E2E json_object reproducer from #44993's test plan is
ported separately as tests/integration/
test_json_object_reasoning_boundary_e2e.py (skip-marked for the rig).
"""
from __future__ import annotations

import ast
import itertools
import os
import textwrap
from types import SimpleNamespace

import pytest


PRISTINE_ROOT = "/private/tmp/candidate_pin_current/vllm"
PRISTINE_STRUCT_OUT = os.path.join(
    PRISTINE_ROOT, "v1", "structured_output", "__init__.py"
)
PRISTINE_SCHEDULER = os.path.join(
    PRISTINE_ROOT, "v1", "core", "sched", "scheduler.py"
)

requires_pristine = pytest.mark.skipif(
    not (
        os.path.isfile(PRISTINE_STRUCT_OUT)
        and os.path.isfile(PRISTINE_SCHEDULER)
    ),
    reason="pristine candidate pin tree not extracted on this host",
)

# Reasoning-end marker token id used by upstream #44993's tests
# (mirrors the real Qwen3 </think> detection: end token in the delta).
MARKER = 248069


# ── Post-merge images (gh pr diff, fetched 2026-06-11) ───────────────
#
# Verbatim snippets of the files AS THEY WILL READ once the watched PRs
# merge. Each P62 drift marker must be a substring of one of these (so
# the wedged-apply protection actually fires at the pin bump) while NOT
# being a substring of anything P62 itself writes (self-collision lint).

# vllm#44993 — should_advance grows an optional new_token_ids parameter.
UPSTREAM_44993_STRUCT_OUT_IMAGE = (
    "    def should_advance(\n"
    "        self,\n"
    '        request: "Request",\n'
    "        new_token_ids: list[int] | None = None,\n"
    "    ) -> bool:\n"
)

# vllm#44993 — scheduler call site passes new_token_ids through.
UPSTREAM_44993_SCHEDULER_IMAGE = (
    "            if new_token_ids and "
    "self.structured_output_manager.should_advance(\n"
    "                request, new_token_ids=new_token_ids\n"
    "            ):\n"
)

# vllm#44297 — grammar_bitmask loop rewrite (intra-step bitmask leak).
UPSTREAM_44297_STRUCT_OUT_IMAGE = (
    "                state_advancements = 0\n"
    "                post_reasoning_end_in_window = False\n"
    "                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())\n"
    "                for i, token in enumerate(req_tokens):\n"
)


def _p62():
    from sndr.engines.vllm.patches.serving import (
        p62_structured_output_spec_decode_timing as M,
    )
    return M


def _pristine_struct_out_src() -> str:
    with open(PRISTINE_STRUCT_OUT) as f:
        return f.read()


def _pristine_scheduler_src() -> str:
    with open(PRISTINE_SCHEDULER) as f:
        return f.read()


def _patched_struct_out_src() -> str:
    """Pristine structured_output/__init__.py with BOTH P62 sub-patches
    applied — the exact text TextPatcher would write on the rig."""
    M = _p62()
    src = _pristine_struct_out_src()
    for old, new in (
        (M.GRAMMAR_BITMASK_OLD, M.GRAMMAR_BITMASK_NEW),
        (M.NEW_METHODS_OLD, M.NEW_METHODS_NEW),
    ):
        assert src.count(old) == 1
        src = src.replace(old, new)
    return src


# ── Exec harness ──────────────────────────────────────────────────────


class _StubStructuredOutputOptions:
    """Stands in for vllm's StructuredOutputOptions enum; distinct
    sentinels are all should_advance's STRUCTURAL_TAG check needs."""

    JSON_OBJECT = "JSON_OBJECT"
    STRUCTURAL_TAG = "STRUCTURAL_TAG"


def _extract_method(src: str, class_name: str, method_name: str) -> str:
    """The exact source lines of one method of one class — original
    indentation preserved so the segment can be re-exec'd in a class."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    lines = src.splitlines(keepends=True)
                    return "".join(lines[item.lineno - 1 : item.end_lineno])
    raise AssertionError(f"{class_name}.{method_name} not found")


P62_HELPER_METHODS = (
    "should_advance",
    "update_reasoning_ended",
    "validate_tokens_reasoning_aware",
    "identify_constrained_draft_tokens",
    "_find_reasoning_end_in_tokens",
)


def _manager_from_src(src: str, method_names=P62_HELPER_METHODS):
    """Exec the named StructuredOutputManager methods extracted from
    ``src`` into a fresh class; behavior tests then run against the
    patched TEXT, not a reimplementation."""
    body = "".join(
        _extract_method(src, "StructuredOutputManager", name)
        for name in method_names
    )
    cls_src = "class P62ManagerUnderTest:\n" + body
    ns: dict = {
        "itertools": itertools,
        "TYPE_CHECKING": False,
        "StructuredOutputOptions": _StubStructuredOutputOptions,
    }
    exec(compile(cls_src, "<p62-patched-structured-output>", "exec"), ns)
    cls = ns["P62ManagerUnderTest"]
    mgr = cls.__new__(cls)
    mgr.enable_in_reasoning = False
    mgr.vllm_config = SimpleNamespace(speculative_config=SimpleNamespace())
    mgr._get_reasoner = lambda request: getattr(request, "test_reasoner", None)
    return mgr


class _RecordingReasoner:
    """Mirrors the real Qwen3 parser the way upstream #44993's tests do:
    reasoning ends when MARKER appears in the delta window."""

    def __init__(self, end_token_id: int = MARKER):
        self.end_token_id = end_token_id
        self.calls: list[list[int]] = []

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        delta = list(delta_ids)
        self.calls.append(delta)
        return self.end_token_id in delta


class _RecordingGrammar:
    def __init__(self):
        self.accepted: list[list[int]] = []
        self.validated: list[list[int]] = []

    def accept_tokens(self, req_id, tokens) -> bool:
        self.accepted.append(list(tokens))
        return True

    def validate_tokens(self, tokens):
        self.validated.append(list(tokens))
        return list(tokens)

    def is_terminated(self) -> bool:
        return False


def _request(
    *,
    all_token_ids,
    num_computed_tokens,
    num_output_placeholders,
    reasoner,
    grammar,
    reasoning_ended=False,
):
    structured = SimpleNamespace(
        reasoning_ended=reasoning_ended,
        grammar=grammar,
        structured_output_key=(_StubStructuredOutputOptions.JSON_OBJECT, "{}"),
    )
    return SimpleNamespace(
        request_id="mock_req",
        use_structured_output=True,
        structured_output_request=structured,
        all_token_ids=list(all_token_ids),
        num_computed_tokens=num_computed_tokens,
        num_output_placeholders=num_output_placeholders,
        test_reasoner=reasoner,
        status=None,
        resumable=True,
    )


def _run_patched_sched_block(mgr, request, new_token_ids):
    """Exec P62's SCHED_UPDATE_FROM_OUTPUT_NEW replacement verbatim —
    the patched scheduler text itself, dedented to module level."""
    M = _p62()
    block = textwrap.dedent(M.SCHED_UPDATE_FROM_OUTPUT_NEW)
    ns = {
        "new_token_ids": list(new_token_ids),
        "self": SimpleNamespace(structured_output_manager=mgr),
        "request": request,
        "req_id": request.request_id,
        "logger": SimpleNamespace(warning=lambda *a, **k: None),
        "RequestStatus": SimpleNamespace(FINISHED_ERROR="FINISHED_ERROR"),
        "stopped": False,
    }
    exec(compile(block, "<p62-patched-scheduler-block>", "exec"), ns)
    return ns


# ── 1. Anchor byte-verification (iron rule #11) ───────────────────────


@requires_pristine
class TestAnchorsAgainstPristine:
    def test_struct_out_anchors_count_exactly_one(self):
        M = _p62()
        src = _pristine_struct_out_src()
        assert src.count(M.GRAMMAR_BITMASK_OLD) == 1
        assert src.count(M.NEW_METHODS_OLD) == 1

    def test_scheduler_anchors_count_exactly_one(self):
        M = _p62()
        src = _pristine_scheduler_src()
        assert src.count(M.SCHED_UPDATE_FROM_OUTPUT_OLD) == 1
        assert src.count(M.SCHED_UDTI_OLD) == 1
        assert src.count(M.SCHED_UDTIO_OLD) == 1

    def test_patched_struct_out_is_valid_python(self):
        ast.parse(_patched_struct_out_src())

    def test_patched_scheduler_is_valid_python(self):
        M = _p62()
        src = _pristine_scheduler_src()
        for old, new in (
            (M.SCHED_UPDATE_FROM_OUTPUT_OLD, M.SCHED_UPDATE_FROM_OUTPUT_NEW),
            (M.SCHED_UDTI_OLD, M.SCHED_UDTI_NEW),
            (M.SCHED_UDTIO_OLD, M.SCHED_UDTIO_NEW),
        ):
            src = src.replace(old, new)
        ast.parse(src)


# ── 2. Pristine bug reproduction (#43388 / Bug 1) ─────────────────────


@requires_pristine
class TestPristineBugReproduction:
    """Documents the bug class on the PRISTINE pin. If these start
    FAILING after a pin bump, upstream merged #44993 — re-run the P62
    deep-diff and review its retire/keep stance."""

    def test_placeholder_window_misses_marker_on_pristine(self):
        """#43388 scenario: spec decode K=4, 4 tokens accepted but 1
        placeholder remains. The pristine placeholder math computes
        delta=[271] and never sees the marker — reasoning_ended stays
        False and the grammar is bypassed (Bug 1)."""
        mgr = _manager_from_src(
            _pristine_struct_out_src(), method_names=("should_advance",)
        )
        reasoner = _RecordingReasoner()
        new_token_ids = [9, 198, MARKER, 271]
        request = _request(
            all_token_ids=[1, 2, 3, 4, 5] + new_token_ids,
            num_computed_tokens=9,
            num_output_placeholders=1,
            reasoner=reasoner,
            grammar=_RecordingGrammar(),
        )
        result = mgr.should_advance(request)
        assert result is False
        assert request.structured_output_request.reasoning_ended is False
        assert reasoner.calls == [[271]]


# ── 3. The four #44993 regression tests, ported onto P62's surface ───


@requires_pristine
class TestP62GrammarAdvanceRegression:
    """Upstream test -> P62-surface mapping:

    #1 uses_new_token_ids_when_provided -> update_reasoning_ended
    #2 without_new_token_ids_falls_back -> should_advance kept pristine
    #3 drains_post_marker_into_grammar  -> patched scheduler block
    #4 no_postmarker_skips_grammar      -> patched scheduler block
    """

    def test_1_update_reasoning_ended_sees_full_accepted_delta(self):
        """Bug 1 (upstream test 1): the reasoner sees the exact
        multi-token accepted delta rather than the placeholder-derived
        window, so reasoning_ended flips even with drafts rejected."""
        mgr = _manager_from_src(_patched_struct_out_src())
        reasoner = _RecordingReasoner()
        new_token_ids = [9, 198, MARKER, 271]
        request = _request(
            all_token_ids=[1, 2, 3, 4, 5] + new_token_ids,
            num_computed_tokens=9,
            num_output_placeholders=1,
            reasoner=reasoner,
            grammar=_RecordingGrammar(),
        )
        mgr.update_reasoning_ended(request, new_token_ids)
        assert reasoner.calls[0] == new_token_ids
        assert request.structured_output_request.reasoning_ended is True

    def test_2_should_advance_placeholder_fallback_preserved(self):
        """Backward compat (upstream test 2): P62 keeps should_advance
        as an untouched compat wrapper — callers that were not rewired
        keep the original placeholder-derived delta window."""
        pristine_method = _extract_method(
            _pristine_struct_out_src(),
            "StructuredOutputManager",
            "should_advance",
        )
        patched_method = _extract_method(
            _patched_struct_out_src(),
            "StructuredOutputManager",
            "should_advance",
        )
        assert patched_method == pristine_method

        mgr = _manager_from_src(_patched_struct_out_src())
        reasoner = _RecordingReasoner(end_token_id=-999)  # never ends
        request = _request(
            all_token_ids=[1, 2, 3, 4, 5],
            num_computed_tokens=5,
            num_output_placeholders=2,
            reasoner=reasoner,
            grammar=_RecordingGrammar(),
        )
        result = mgr.should_advance(request)
        # placeholder window: start = 5 - 2 = 3 -> delta = [4, 5]
        assert reasoner.calls == [[4, 5]]
        assert result is False

    def test_3_post_marker_tail_drained_into_grammar(self):
        """Bug 2 (upstream test 3): on the step that ends reasoning the
        patched scheduler block feeds the grammar exactly the
        post-marker portion — neither the reasoning prefix nor the
        marker itself — so the next bitmask reflects the advanced FSM
        and json_object cannot emit a duplicate opening token."""
        mgr = _manager_from_src(_patched_struct_out_src())
        grammar = _RecordingGrammar()
        reasoner = _RecordingReasoner()
        new_token_ids = [9, 198, MARKER, 271, 5005]
        request = _request(
            all_token_ids=[1, 2, 3, 4, 5] + new_token_ids,
            num_computed_tokens=10,
            num_output_placeholders=0,
            reasoner=reasoner,
            grammar=grammar,
        )
        ns = _run_patched_sched_block(mgr, request, new_token_ids)
        assert grammar.accepted == [[271, 5005]]
        assert request.structured_output_request.reasoning_ended is True
        assert ns["stopped"] is False
        assert request.status is None

    def test_4_no_post_marker_tail_skips_grammar_accept(self):
        """Upstream test 4: marker is the LAST accepted token — no
        post-marker tail, so grammar.accept_tokens must not be called;
        reasoning_ended still flips for the next step."""
        mgr = _manager_from_src(_patched_struct_out_src())
        grammar = _RecordingGrammar()
        reasoner = _RecordingReasoner()
        new_token_ids = [9, 198, MARKER]
        request = _request(
            all_token_ids=[1, 2, 3, 4, 5] + new_token_ids,
            num_computed_tokens=8,
            num_output_placeholders=0,
            reasoner=reasoner,
            grammar=grammar,
        )
        _run_patched_sched_block(mgr, request, new_token_ids)
        assert grammar.accepted == []
        assert request.structured_output_request.reasoning_ended is True


# ── 4. Drift-marker watch entries for #44297 / #44993 ────────────────


class TestDriftMarkerWatchEntries:
    """P62's wiring must carry post-image drift markers for BOTH halves
    of the upstream fix pair so a merged #44297/#44993 produces a clean
    upstream-drift SKIP instead of a wedged anchor-mismatch apply."""

    def test_struct_out_patcher_watches_both_prs(self):
        M = _p62()
        markers = set(M.P62_STRUCT_OUT_DRIFT_MARKERS)
        assert M.DRIFT_MARKER_44993_SHOULD_ADVANCE_SIG in markers
        assert M.DRIFT_MARKER_44297_BITMASK_REWRITE in markers

    def test_scheduler_patcher_watches_44993(self):
        M = _p62()
        assert (
            M.DRIFT_MARKER_44993_SCHED_CALLSITE
            in set(M.P62_SCHEDULER_DRIFT_MARKERS)
        )

    @requires_pristine
    def test_patchers_carry_the_markers(self, monkeypatch):
        """Same guards.vllm_install_root redirection seam the lint and
        tools/pin_preflight.py use — builders resolve into the
        candidate tree."""
        import sndr.engines.vllm.detection.guards as guards
        monkeypatch.setattr(guards, "vllm_install_root", lambda: PRISTINE_ROOT)
        M = _p62()
        struct_out = M._make_struct_out_patcher()
        sched = M._make_scheduler_patcher()
        assert struct_out is not None and sched is not None
        assert list(struct_out.upstream_drift_markers) == list(
            M.P62_STRUCT_OUT_DRIFT_MARKERS
        )
        assert list(sched.upstream_drift_markers) == list(
            M.P62_SCHEDULER_DRIFT_MARKERS
        )

    def test_markers_fire_on_post_merge_images(self):
        M = _p62()
        assert (
            M.DRIFT_MARKER_44993_SHOULD_ADVANCE_SIG
            in UPSTREAM_44993_STRUCT_OUT_IMAGE
        )
        assert (
            M.DRIFT_MARKER_44297_BITMASK_REWRITE
            in UPSTREAM_44297_STRUCT_OUT_IMAGE
        )
        assert (
            M.DRIFT_MARKER_44993_SCHED_CALLSITE
            in UPSTREAM_44993_SCHEDULER_IMAGE
        )

    def test_markers_not_in_own_emitted_text(self):
        """tools/lint_drift_markers.py contract (PN369 false-skip
        class): no marker may be a substring of P62's own replacements
        or its idempotency marker lines."""
        M = _p62()
        replacements = (
            M.GRAMMAR_BITMASK_NEW,
            M.NEW_METHODS_NEW,
            M.SCHED_UPDATE_FROM_OUTPUT_NEW,
            M.SCHED_UDTI_NEW,
            M.SCHED_UDTIO_NEW,
        )
        marker_lines = (
            f"# [Genesis wiring marker: {M.GENESIS_P62_MARKER}"
            " :: structured_output]\n",
            f"# [Genesis wiring marker: {M.GENESIS_P62_MARKER}"
            " :: scheduler.py]\n",
        )
        all_markers = list(M.P62_STRUCT_OUT_DRIFT_MARKERS) + list(
            M.P62_SCHEDULER_DRIFT_MARKERS
        )
        assert all_markers, "watch entries must not be empty"
        for marker in all_markers:
            for text in replacements + marker_lines:
                assert marker not in text, (
                    f"drift marker {marker!r} collides with P62's own "
                    "emitted text — lint_drift_markers violation"
                )

    @requires_pristine
    def test_markers_absent_from_pristine(self):
        """A marker already present in the CURRENT pin can never gate
        anything — it would false-fire today."""
        M = _p62()
        struct_src = _pristine_struct_out_src()
        sched_src = _pristine_scheduler_src()
        for marker in M.P62_STRUCT_OUT_DRIFT_MARKERS:
            assert marker not in struct_src
        for marker in M.P62_SCHEDULER_DRIFT_MARKERS:
            assert marker not in sched_src
