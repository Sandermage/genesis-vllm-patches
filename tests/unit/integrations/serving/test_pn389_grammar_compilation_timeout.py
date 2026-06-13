# SPDX-License-Identifier: Apache-2.0
"""PN389 — XGrammar input-validation + grammar-compilation timeouts.

Contract pinned here (TDD, written before the implementation):

Upstream bug class (vllm#45390, 7 GHSA CWE-400): structured-output
compilation runs on the CPU EngineCore loop with NO wall-clock bound, so
a pathological grammar/regex/JSON-schema wedges ALL decode indefinitely —
an instance-wide DoS on our single-instance PROD (async-scheduling
overlap does NOT save us; compilation is pure-CPU off the GPU stream).

PN389 vendors the grammar-timeout core of #45390 across THREE files in
one atomic ``MultiFilePatchTransaction``, gated on
``GENESIS_ENABLE_PN389_GRAMMAR_TIMEOUTS`` (default_on=False):

  (1) v1/structured_output/utils.py — ADDITIVE: ``run_with_timeout``
      (daemon thread + Queue + Semaphore(4)) and ``_check_regex_complexity``
      plus their constants. Our pin g303916e93 has NO compilation timeout
      at all, so these are brand-new symbols.
  (2) v1/structured_output/backend_xgrammar.py — wrap the REGEX compile arm
      and EVERY ``xgr.Grammar.from_*`` validation call (regex / choice /
      json_schema / ebnf / structural_tag) in ``run_with_timeout``.
  (3) envs.py — ADDITIVE: ``VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS``,
      Genesis default 2s (the PR ships 10s, which would blow our
      70-160ms TTFT SLO ~60x before the timeout fires).

Sub-contracts:
  1. Three patchers, with the expected required sub-patches each.
  2. apply() commits all three files atomically (one transaction) and the
     patched files still compile.
  3. Second apply() is idempotent (marker short-circuit -> skipped).
  4. apply() self-skips on the #45390 merged form via drift markers
     (reason: upstream_merged) without touching the files.
  5. Drift markers do not collide with PN389's own replacement text or its
     Layer-6 marker line (tools/lint_drift_markers.py / PN369 contract)
     AND at least one marker is an exact substring of the merged form.
  6. Opt-in gate: with the dispatcher gate closed, apply() skips without
     touching the targets.
  7. GENESIS-SPECIFIC: real gemma4 / qwen3_coder tool-schema-derived regex
     passes ``_check_regex_complexity`` with NO false-positive rejection,
     while a genuinely adversarial pattern IS rejected.
  8. Pristine pin invariants (opportunistic): anchors unique (count==1),
     drift markers absent in the pristine tree.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Unit tests patch fresh tmp files; the Layer-0 file cache must never
# satisfy apply() from a previous run's state.
os.environ.setdefault("GENESIS_NO_PATCH_CACHE", "1")

from sndr.engines.vllm.patches.serving import (  # noqa: E402
    pn389_grammar_compilation_timeout as pn389,
)

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm")


# ── Fixtures: pin-form anchor regions (byte-faithful copies) ─────────

# Pin g303916e93 form of v1/structured_output/utils.py — enough of the
# head to carry both anchors (import block + `CACHE = None` site).
PIN_UTILS = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "from __future__ import annotations\n"
    "\n"
    "import hashlib\n"
    "import importlib.metadata\n"
    "import os\n"
    "import tempfile\n"
    "from typing import TYPE_CHECKING\n"
    "\n"
    "import numpy as np\n"
    "from vllm.logger import init_logger\n"
    "\n"
    "logger = init_logger(__name__)\n"
    "\n"
    "CACHE = None\n"
    "\n"
    "\n"
    "def apply_grammar_bitmask():\n"
    "    pass\n"
)

# Pin g303916e93 form of v1/structured_output/backend_xgrammar.py — the
# import block, the compile_grammar REGEX arm, and the
# validate_xgrammar_grammar body (all five validation arms).
PIN_XGRAMMAR = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "import json\n"
    "\n"
    "import vllm.envs\n"
    "from vllm.v1.structured_output.backend_types import (\n"
    "    StructuredOutputOptions,\n"
    ")\n"
    "from vllm.v1.structured_output.utils import (\n"
    "    choice_as_grammar,\n"
    "    convert_lark_to_ebnf,\n"
    "    grammar_is_likely_lark,\n"
    ")\n"
    "\n"
    "\n"
    "class XgrammarBackend:\n"
    # Byte-faithful copy of the pin g303916e93 compile_grammar method — the
    # full-method anchor PN389 refactors into _compile_ctx /
    # _compile_ctx_inner so the EngineCore DFA build is wall-clock bounded.
    "    def compile_grammar(\n"
    "        self, request_type: StructuredOutputOptions, grammar_spec: str\n"
    "    ) -> StructuredOutputGrammar:\n"
    "        if request_type == StructuredOutputOptions.JSON:\n"
    "            ctx = self.compiler.compile_json_schema(\n"
    "                grammar_spec, any_whitespace=not self.disable_any_whitespace\n"
    "            )\n"
    "        elif request_type == StructuredOutputOptions.JSON_OBJECT:\n"
    "            ctx = self.compiler.compile_json_schema(\n"
    "                '{\"type\": \"object\"}', any_whitespace=not self.disable_any_whitespace\n"
    "            )\n"
    "        elif request_type == StructuredOutputOptions.GRAMMAR:\n"
    "            ctx = self.compiler.compile_grammar(grammar_spec)\n"
    "        elif request_type == StructuredOutputOptions.REGEX:\n"
    "            ctx = self.compiler.compile_regex(grammar_spec)\n"
    "        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:\n"
    "            s_tag = json.loads(grammar_spec)\n"
    '            if "structures" in s_tag:\n'
    "                # Falling back to deprecated method of compiling structural tag\n"
    "                tags = [\n"
    "                    xgr.StructuralTagItem(\n"
    '                        begin=s["begin"],\n'
    '                        schema=json.dumps(s["schema"]),\n'
    '                        end=s["end"],\n'
    "                    )\n"
    '                    for s in s_tag["structures"]\n'
    "                ]\n"
    '                ctx = self.compiler.compile_structural_tag(tags, s_tag["triggers"])\n'
    "            else:\n"
    "                ctx = self.compiler.compile_structural_tag(grammar_spec)\n"
    "        else:\n"
    "            logger.error(\n"
    '                "Validation should have already occurred. Please file an issue."\n'
    "            )\n"
    "            raise ValueError(\n"
    '                f"grammar is not of valid supported types. ({request_type!s})"\n'
    "            )\n"
    "\n"
    "        return XgrammarGrammar(\n"
    "            matcher=xgr.GrammarMatcher(\n"
    "                ctx,\n"
    "                max_rollback_tokens=self.num_speculative_tokens,\n"
    "            ),\n"
    "            vocab_size=self.vocab_size,\n"
    "            ctx=ctx,\n"
    "        )\n"
    "\n"
    "\n"
    "def validate_xgrammar_grammar(sampling_params):\n"
    "    if sampling_params.structured_outputs is None:\n"
    "        return\n"
    "    so_params = sampling_params.structured_outputs\n"
    "\n"
    "    if so_params.regex:\n"
    "        try:\n"
    "            xgr.Grammar.from_regex(so_params.regex)\n"
    "        except Exception as err:\n"
    '            raise ValueError(f"bad regex: {err}") from err\n'
    "\n"
    "    if so_params.choice:\n"
    "        choice_grammar = choice_as_grammar(so_params.choice)\n"
    "        try:\n"
    "            xgr.Grammar.from_ebnf(choice_grammar)\n"
    "        except Exception as err:\n"
    '            raise ValueError(f"bad choice: {err}") from err\n'
    "        return\n"
    "\n"
    "    if so_params.json:\n"
    "        schema = so_params.json\n"
    "        try:\n"
    "            xgr.Grammar.from_json_schema(schema)\n"
    "        except Exception as err:\n"
    '            raise ValueError(f"bad json: {err}") from err\n'
    "        return\n"
    "\n"
    "    if so_params.grammar:\n"
    "        try:\n"
    "            # parse the grammar, but we aren't compiling it.\n"
    "            xgr.Grammar.from_ebnf(so_params.grammar)\n"
    "        except Exception as e:\n"
    '            raise ValueError("Invalid grammar specification.") from e\n'
    "        return\n"
    "\n"
    "    if so_params.structural_tag:\n"
    "        try:\n"
    "            s_tag = json.loads(so_params.structural_tag)\n"
    '            if "structures" in s_tag:\n'
    "                tags = []\n"
    '                xgr.Grammar.from_structural_tag(tags, s_tag["triggers"])\n'
    "            else:\n"
    "                xgr.Grammar.from_structural_tag(so_params.structural_tag)\n"
    "        except Exception as e:\n"
    '            raise ValueError("Invalid structural tag specification.") from e\n'
)

# Pin g303916e93 form of envs.py — the declaration block + the os.getenv
# lambda block carrying VLLM_XGRAMMAR_CACHE_MB.
PIN_ENVS = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "import os\n"
    "\n"
    "if TYPE_CHECKING:\n"
    "    VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE: int = 394 * 1024 * 1024\n"
    "    VLLM_XGRAMMAR_CACHE_MB: int = 0\n"
    "    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256\n"
    "\n"
    "environment_variables = {\n"
    "    # Control the cache sized used by the xgrammar compiler.\n"
    '    "VLLM_XGRAMMAR_CACHE_MB": lambda: int(os.getenv("VLLM_XGRAMMAR_CACHE_MB", "512")),\n'
    "    # Control the threshold for msgspec zero copy.\n"
    '    "VLLM_MSGPACK_ZERO_COPY_THRESHOLD": lambda: int(os.getenv("X", "256")),\n'
    "}\n"
)


def _build_merged(text: str) -> str:
    """Splice one of PN389's drift-marker (PR-form) strings into a copy of
    the pin text so apply() should treat it as upstream-merged."""
    # The envs.py comment head is the most file-local merged signal.
    merged_marker = (
        "    # Maximum time in seconds allowed for grammar/regex "
        "compilation into a\n"
    )
    return text + merged_marker


# ── Helpers ──────────────────────────────────────────────────────────


def _install(tmp_path, monkeypatch, *, utils=PIN_UTILS, xgr=PIN_XGRAMMAR, envs=PIN_ENVS):
    """Install all three PN389 targets under tmp and route resolution."""
    targets = {
        "v1/structured_output/utils.py": (tmp_path / "utils.py", utils),
        "v1/structured_output/backend_xgrammar.py": (tmp_path / "backend_xgrammar.py", xgr),
        "envs.py": (tmp_path / "envs.py", envs),
    }
    for _rel, (path, text) in targets.items():
        path.write_text(text, encoding="utf-8")

    def _resolve(rel):
        entry = targets.get(rel)
        return str(entry[0]) if entry else None

    monkeypatch.setattr(pn389, "resolve_vllm_file", _resolve)
    monkeypatch.setattr(pn389, "vllm_install_root", lambda: str(tmp_path))
    import sndr.dispatcher as dispatcher
    monkeypatch.setattr(
        dispatcher, "should_apply", lambda pid: (True, "test override")
    )
    return {rel: path for rel, (path, _t) in targets.items()}


def _load_helpers(tmp_path, monkeypatch) -> dict:
    """Apply PN389 to a fresh utils.py and exec ONLY the emitted helper
    block (run_with_timeout + _check_regex_complexity + constants) in an
    isolated namespace.

    The full patched module imports ``vllm.logger`` (unavailable in the
    unit env), so we slice from the additive sentinel ``_PN389_T = TypeVar``
    to end-of-file and prepend the stdlib imports the block needs. This
    exercises the REAL emitted helper code, not a re-typed copy.
    """
    paths = _install(tmp_path, monkeypatch)
    status, reason = pn389.apply()
    assert status == "applied", reason
    patched = paths["v1/structured_output/utils.py"].read_text("utf-8")
    start = patched.index("_PN389_T = TypeVar")
    block = patched[start:]
    preamble = (
        "import threading\n"
        "from collections.abc import Callable\n"
        "from queue import Empty, Queue\n"
        "from typing import TypeVar\n"
    )
    ns: dict = {}
    exec(compile(preamble + block, "patched_utils_block.py", "exec"), ns)  # noqa: S102
    return ns


# ── Patcher shape ────────────────────────────────────────────────────


class TestPatcherShape:
    def test_three_patchers_built(self, tmp_path, monkeypatch):
        _install(tmp_path, monkeypatch)
        patchers = pn389._all_patchers()
        assert len(patchers) == 3

    def test_utils_patcher_subpatches(self, tmp_path, monkeypatch):
        _install(tmp_path, monkeypatch)
        p = pn389._make_utils_patcher()
        names = {sp.name for sp in p.sub_patches}
        assert names == {"pn389_utils_imports", "pn389_utils_helpers"}
        assert all(sp.required for sp in p.sub_patches)

    def test_xgrammar_patcher_subpatches(self, tmp_path, monkeypatch):
        _install(tmp_path, monkeypatch)
        p = pn389._make_xgrammar_patcher()
        names = {sp.name for sp in p.sub_patches}
        assert names == {
            "pn389_xgr_imports",
            "pn389_xgr_compile_grammar",
            "pn389_xgr_validate_regex",
            "pn389_xgr_validate_choice",
            "pn389_xgr_validate_json",
            "pn389_xgr_validate_ebnf",
            "pn389_xgr_validate_structural_tag",
        }
        assert all(sp.required for sp in p.sub_patches)

    def test_envs_patcher_subpatches(self, tmp_path, monkeypatch):
        _install(tmp_path, monkeypatch)
        p = pn389._make_envs_patcher()
        names = {sp.name for sp in p.sub_patches}
        assert names == {"pn389_envs_decl", "pn389_envs_lambda"}

    def test_patchers_none_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(pn389, "resolve_vllm_file", lambda rel: None)
        assert pn389._make_utils_patcher() is None
        assert pn389._make_xgrammar_patcher() is None
        assert pn389._make_envs_patcher() is None
        assert pn389._all_patchers() == []

    def test_module_documents_dos_slo_and_env_flag(self):
        doc = pn389.__doc__ or ""
        assert "45390" in doc
        assert "DoS" in doc or "GHSA" in doc
        # The Genesis-specific 2s default must be documented (not the PR's 10s).
        assert "2s" in doc or "TTFT" in doc
        src = Path(pn389.__file__).read_text(encoding="utf-8")
        assert "GENESIS_ENABLE_PN389_GRAMMAR_TIMEOUTS" in src


# ── apply() — atomic three-file commit ───────────────────────────────


class TestApply:
    def test_apply_commits_all_three_files(self, tmp_path, monkeypatch):
        paths = _install(tmp_path, monkeypatch)
        status, reason = pn389.apply()
        assert status == "applied", reason
        assert "3 files" in reason

        utils_out = paths["v1/structured_output/utils.py"].read_text("utf-8")
        xgr_out = paths["v1/structured_output/backend_xgrammar.py"].read_text("utf-8")
        envs_out = paths["envs.py"].read_text("utf-8")

        # utils: helpers + constants landed, file compiles.
        assert "def run_with_timeout(" in utils_out
        assert "def _check_regex_complexity(" in utils_out
        assert "_compilation_semaphore = threading.Semaphore" in utils_out
        assert "MAX_REGEX_NESTING_DEPTH = 20" in utils_out
        compile(utils_out, "utils.py", "exec")

        # xgrammar: helpers imported, all five validation arms wrapped.
        assert "run_with_timeout," in xgr_out
        assert "_check_regex_complexity," in xgr_out
        # Every from_* call now goes through run_with_timeout — the bare
        # direct calls must be gone (each replaced by a wrapped form).
        assert "xgr.Grammar.from_regex,\n" in xgr_out
        assert "xgr.Grammar.from_json_schema,\n" in xgr_out
        assert "xgr.Grammar.from_ebnf,\n" in xgr_out
        assert "xgr.Grammar.from_structural_tag,\n" in xgr_out
        assert xgr_out.count("run_with_timeout(") >= 6
        compile(xgr_out, "backend_xgrammar.py", "exec")

        # envs: new env declared + lambda with Genesis 2s default.
        assert "VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS: int = 2" in envs_out
        assert (
            'os.getenv("VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS", "2")'
            in envs_out
        )
        compile(envs_out, "envs.py", "exec")

    def test_compile_regex_arm_prefiltered(self, tmp_path, monkeypatch):
        paths = _install(tmp_path, monkeypatch)
        status, reason = pn389.apply()
        assert status == "applied", reason
        xgr_out = paths["v1/structured_output/backend_xgrammar.py"].read_text("utf-8")
        # The REGEX compile arm gains a complexity pre-filter before compile.
        assert "_check_regex_complexity(grammar_spec)" in xgr_out

    def test_enginecore_compile_path_is_timeout_bounded(
        self, tmp_path, monkeypatch
    ):
        """CORE FIX (review MAJOR gap): compile_grammar's EngineCore DFA
        build — not just the frontend from_* validation — must run through
        run_with_timeout. A schema that parses fast but compiles
        catastrophically would otherwise wedge the engine unbounded.

        Asserts the PR-style refactor landed: compile_grammar delegates to
        _compile_ctx, which calls run_with_timeout(self._compile_ctx_inner,
        ...), and _compile_ctx_inner holds every compile_* dispatch arm
        (JSON / JSON_OBJECT / GRAMMAR / REGEX / STRUCTURAL_TAG). No bare
        `ctx = self.compiler.compile_*` survives outside the timed inner.
        """
        paths = _install(tmp_path, monkeypatch)
        status, reason = pn389.apply()
        assert status == "applied", reason
        xgr_out = paths["v1/structured_output/backend_xgrammar.py"].read_text("utf-8")

        # The three-method refactor is present.
        assert "def compile_grammar(" in xgr_out
        assert "def _compile_ctx(" in xgr_out
        assert "def _compile_ctx_inner(" in xgr_out

        # compile_grammar delegates to _compile_ctx with the env timeout.
        assert "self._compile_ctx(" in xgr_out
        assert (
            "vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS" in xgr_out
        )

        # _compile_ctx runs the inner dispatch through run_with_timeout
        # with the "Grammar compilation" label (the EngineCore bound).
        ctx_slice = xgr_out[
            xgr_out.index("def _compile_ctx(") : xgr_out.index(
                "def _compile_ctx_inner("
            )
        ]
        assert "run_with_timeout(" in ctx_slice
        assert "self._compile_ctx_inner" in ctx_slice
        assert '"Grammar compilation"' in ctx_slice

        # _compile_ctx_inner carries EVERY compile_* arm — so all five
        # request types' DFA builds are inside the timed thread, and no
        # bare compiler call leaks out into compile_grammar itself.
        inner_slice = xgr_out[xgr_out.index("def _compile_ctx_inner(") :]
        for compile_call in (
            "self.compiler.compile_json_schema(",
            "self.compiler.compile_grammar(",
            "self.compiler.compile_regex(",
            "self.compiler.compile_structural_tag(",
        ):
            assert compile_call in inner_slice, compile_call
        # The REGEX pre-filter lives inside the timed inner, before compile.
        assert "_check_regex_complexity(grammar_spec)" in inner_slice

        # compile_grammar (the wrapper) must NOT call the compiler directly;
        # the only compiler calls are inside _compile_ctx_inner.
        wrapper_slice = xgr_out[
            xgr_out.index("def compile_grammar(") : xgr_out.index(
                "def _compile_ctx("
            )
        ]
        assert "self.compiler.compile_" not in wrapper_slice

        compile(xgr_out, "backend_xgrammar.py", "exec")

    def test_compile_grammar_inner_dispatch_is_bit_identical(
        self, tmp_path, monkeypatch
    ):
        """The refactored _compile_ctx_inner must dispatch to the SAME
        compiler call for each request type as the pin did — bit-identical
        for any compile within budget. Exec the patched class against a
        recording fake compiler and assert each type hits the right call.
        """
        paths = _install(tmp_path, monkeypatch)
        status, reason = pn389.apply()
        assert status == "applied", reason
        xgr_out = paths["v1/structured_output/backend_xgrammar.py"].read_text("utf-8")

        # Slice the XgrammarBackend class body and exec it in isolation with
        # stubs for the symbols it references, so we can drive the real
        # patched _compile_ctx_inner without importing vllm.
        class _RecordingCompiler:
            def __init__(self):
                self.calls = []

            def compile_json_schema(self, spec, any_whitespace=True):
                self.calls.append(("json", spec))
                return ("ctx", "json")

            def compile_grammar(self, spec):
                self.calls.append(("grammar", spec))
                return ("ctx", "grammar")

            def compile_regex(self, spec):
                self.calls.append(("regex", spec))
                return ("ctx", "regex")

            def compile_structural_tag(self, *a):
                self.calls.append(("stag", a))
                return ("ctx", "stag")

        class _Opt:
            JSON = "JSON"
            JSON_OBJECT = "JSON_OBJECT"
            GRAMMAR = "GRAMMAR"
            REGEX = "REGEX"
            STRUCTURAL_TAG = "STRUCTURAL_TAG"

        # Build a minimal namespace and define the patched class against it.
        # Slice the three refactored methods out of the patched fixture
        # (compile_grammar -> _compile_ctx -> _compile_ctx_inner). The
        # fixture follows _compile_ctx_inner with the module-level
        # validate_xgrammar_grammar (dedented), the robust end-boundary.
        method_start = xgr_out.index("    def compile_grammar(")
        method_end = xgr_out.index("\ndef validate_xgrammar_grammar(")
        backend_src = xgr_out[method_start:method_end] + "\n"
        wrapper = (
            "import json\n"
            "class XgrammarBackend:\n"
            "    def __init__(self, compiler):\n"
            "        self.compiler = compiler\n"
            "        self.disable_any_whitespace = False\n"
            "        self.num_speculative_tokens = 0\n"
            "        self.vocab_size = 10\n"
            + backend_src
        )
        ns = {
            "StructuredOutputOptions": _Opt,
            "vllm": type("V", (), {"envs": type("E", (), {
                "VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS": 5
            })}),
            "run_with_timeout": (
                lambda fn, *a, timeout, label: fn(*a)
            ),
            "_check_regex_complexity": lambda p: None,
            "XgrammarGrammar": lambda **kw: kw,
            "xgr": type("X", (), {"GrammarMatcher": lambda *a, **k: None}),
            "logger": type("L", (), {"error": staticmethod(lambda *a: None)}),
        }
        exec(compile(wrapper, "patched_backend.py", "exec"), ns)  # noqa: S102
        comp = _RecordingCompiler()
        backend = ns["XgrammarBackend"](comp)

        backend.compile_grammar(_Opt.JSON, "{}")
        backend.compile_grammar(_Opt.GRAMMAR, "root ::= a")
        backend.compile_grammar(_Opt.REGEX, "abc")
        # Each request type dispatched to its matching compiler call.
        kinds = [c[0] for c in comp.calls]
        assert kinds == ["json", "grammar", "regex"]

    def test_second_apply_idempotent(self, tmp_path, monkeypatch):
        _install(tmp_path, monkeypatch)
        first, first_reason = pn389.apply()
        assert first == "applied", first_reason
        second, second_reason = pn389.apply()
        assert second == "skipped"

    def test_is_applied_true_only_after_all_three(self, tmp_path, monkeypatch):
        paths = _install(tmp_path, monkeypatch)
        assert pn389.is_applied() is False
        status, reason = pn389.apply()
        assert status == "applied", reason
        assert pn389.is_applied() is True


# ── upstream-merge self-skip ─────────────────────────────────────────


class TestUpstreamSelfSkip:
    def test_self_skips_when_envs_merged(self, tmp_path, monkeypatch):
        merged_envs = _build_merged(PIN_ENVS)
        paths = _install(tmp_path, monkeypatch, envs=merged_envs)
        status, reason = pn389.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower()
        # No file was touched (the merged file is unchanged, and the
        # transaction never ran so utils/xgrammar stay pristine too).
        assert paths["envs.py"].read_text("utf-8") == merged_envs
        assert "run_with_timeout" not in (
            paths["v1/structured_output/utils.py"].read_text("utf-8")
        )


# ── drift-marker self-collision (PN369 contract) ─────────────────────


class TestDriftMarkers:
    def test_markers_not_substring_of_own_emitted_text(self, tmp_path, monkeypatch):
        _install(tmp_path, monkeypatch)
        marker_line = f"# [Genesis wiring marker: {pn389.GENESIS_PN389_MARKER}]\n"
        non_banner = [
            dm for dm in pn389._DRIFT_MARKERS if not dm.startswith("[Genesis")
        ]
        assert non_banner, "must carry at least one upstream-form marker"
        for p in pn389._all_patchers():
            for dm in non_banner:
                for sp in p.sub_patches:
                    assert dm not in sp.replacement, (
                        f"drift marker {dm!r} collides with {sp.name} "
                        "replacement — would false-fire (PN369 class)"
                    )
                assert dm not in marker_line

    def test_markers_absent_from_pin_form_fixtures(self):
        non_banner = [
            dm for dm in pn389._DRIFT_MARKERS if not dm.startswith("[Genesis")
        ]
        for dm in non_banner:
            assert dm not in PIN_UTILS
            assert dm not in PIN_XGRAMMAR
            assert dm not in PIN_ENVS


# ── opt-in gate ──────────────────────────────────────────────────────


class TestGate:
    def test_apply_skips_when_gate_closed(self, tmp_path, monkeypatch):
        paths = _install(tmp_path, monkeypatch)
        import sndr.dispatcher as dispatcher
        monkeypatch.setattr(
            dispatcher, "should_apply", lambda pid: (False, "opt-in: env unset")
        )
        status, _reason = pn389.apply()
        assert status == "skipped"
        # Targets untouched.
        assert paths["v1/structured_output/utils.py"].read_text("utf-8") == PIN_UTILS
        assert paths["envs.py"].read_text("utf-8") == PIN_ENVS


# ── GENESIS-SPECIFIC: no false-positive on real tool-schema regex ────


# JSON-schema-derived regexes of the shape outlines/xgrammar build from
# real Genesis tool schemas. Hand-faithful to the structure (one paren
# group per property/value, nested groups for nested objects/arrays).
# These are the patterns that flow through the REGEX arm and through
# _check_regex_complexity; the GENESIS concern is that the naive
# paren-depth counter must NOT reject any of them.

# qwen3_coder-style edit_file tool: path(str), content(str), nested meta
# object {line:int, flags:[bool]}. Realistic agent tool schema.
QWEN3_CODER_EDIT_FILE_REGEX = (
    r'\{[ ]?"path"[ ]?:[ ]?("(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|'
    r'\\u[0-9a-fA-F]{4})*")[ ]?,[ ]?"content"[ ]?:[ ]?'
    r'("(?:[^"\\\x00-\x1f]|\\["\\/bfnrt])*")[ ]?,[ ]?"meta"[ ]?:[ ]?'
    r'(\{[ ]?"line"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))[ ]?,[ ]?'
    r'"flags"[ ]?:[ ]?(\[([ ]?(true|false)([ ]?,[ ]?(true|false))*)?'
    r'[ ]?\])[ ]?\})[ ]?\}'
)

# gemma4-style get_weather tool: location(str), unit(enum), days(int).
GEMMA4_GET_WEATHER_REGEX = (
    r'\{[ ]?"location"[ ]?:[ ]?'
    r'("(?:[^"\\\x00-\x1f]|\\["\\/bfnrt])*")[ ]?,[ ]?'
    r'"unit"[ ]?:[ ]?("celsius"|"fahrenheit")[ ]?,[ ]?'
    r'"days"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))[ ]?\}'
)

# gemma4-style deeply-but-legitimately nested config tool (3 nested
# objects) — still well under the depth-20 bound.
GEMMA4_NESTED_CONFIG_REGEX = (
    r'\{("server":(\{("opts":(\{("tls":(\{("verify":(true|false)\})\})\})\}))\})\}'
)

LEGIT_TOOL_REGEXES = [
    QWEN3_CODER_EDIT_FILE_REGEX,
    GEMMA4_GET_WEATHER_REGEX,
    GEMMA4_NESTED_CONFIG_REGEX,
]


class TestNoFalsePositiveOnRealToolSchemas:
    """The cheap pre-filter must not reject legit JSON-schema-derived regex.

    Loads _check_regex_complexity from the PATCHED utils.py (the real
    emitted code), then feeds it our gemma4 / qwen3_coder tool-schema
    regexes. This is the GENESIS pre-enable gate from the roadmap:
    confirm no false-positive BEFORE turning the timeout reject on.
    """

    def _load_check(self, tmp_path, monkeypatch):
        ns = _load_helpers(tmp_path, monkeypatch)
        return ns["_check_regex_complexity"], ns

    def test_real_tool_schema_regexes_pass(self, tmp_path, monkeypatch):
        check, ns = self._load_check(tmp_path, monkeypatch)
        # Sanity: all our sample regexes nest well under the bound.
        assert ns["MAX_REGEX_NESTING_DEPTH"] == 20
        for rx in LEGIT_TOOL_REGEXES:
            # Must NOT raise — these are legitimate tool-schema regexes.
            check(rx)

    def test_adversarial_pattern_rejected(self, tmp_path, monkeypatch):
        check, ns = self._load_check(tmp_path, monkeypatch)
        # >20 nested groups -> rejected ("too deep").
        deep = "(" * 25 + "a" + ")" * 25
        with pytest.raises(ValueError, match="too deep"):
            check(deep)
        # >10K chars -> rejected ("too long").
        long_pat = "a" * (ns["MAX_REGEX_LENGTH"] + 1)
        with pytest.raises(ValueError, match="too long"):
            check(long_pat)

    def test_pattern_exactly_at_bounds_passes(self, tmp_path, monkeypatch):
        check, ns = self._load_check(tmp_path, monkeypatch)
        depth = ns["MAX_REGEX_NESTING_DEPTH"]
        at_depth = "(" * depth + "a" + ")" * depth
        check(at_depth)  # exactly at the bound — must pass
        at_len = "a" * ns["MAX_REGEX_LENGTH"]
        check(at_len)  # exactly at the bound — must pass


class TestRunWithTimeoutBehaviour:
    """run_with_timeout emitted into the patched utils.py behaves correctly."""

    def _load_run(self, tmp_path, monkeypatch):
        return _load_helpers(tmp_path, monkeypatch)["run_with_timeout"]

    def test_fast_call_returns_value(self, tmp_path, monkeypatch):
        run = self._load_run(tmp_path, monkeypatch)
        assert run(lambda x: x * 2, 21, timeout=5, label="t") == 42

    def test_timeout_raises_value_error(self, tmp_path, monkeypatch):
        import time

        run = self._load_run(tmp_path, monkeypatch)

        def slow():
            time.sleep(3)
            return "never"

        start = time.monotonic()
        with pytest.raises(ValueError, match="timed out"):
            run(slow, timeout=1, label="Grammar compilation")
        # Caller unblocks in ~timeout, not ~fn duration.
        assert time.monotonic() - start < 2.5

    def test_inner_exception_propagates(self, tmp_path, monkeypatch):
        run = self._load_run(tmp_path, monkeypatch)

        def boom():
            raise RuntimeError("inner failure")

        with pytest.raises(RuntimeError, match="inner failure"):
            run(boom, timeout=5, label="t")


# ── Pristine pin invariants (opportunistic) ──────────────────────────


@pytest.mark.skipif(
    not (PIN_TREE / "v1/structured_output/utils.py").is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAgainstPristine:
    def test_utils_anchors_unique_and_helpers_absent(self):
        src = (PIN_TREE / "v1/structured_output/utils.py").read_text("utf-8")
        assert src.count(pn389.PN389_UTILS_IMPORTS_OLD) == 1
        assert src.count(pn389.PN389_UTILS_HELPERS_OLD) == 1
        assert "def run_with_timeout(" not in src
        assert "_check_regex_complexity" not in src

    def test_xgrammar_anchors_unique(self):
        src = (
            PIN_TREE / "v1/structured_output/backend_xgrammar.py"
        ).read_text("utf-8")
        assert src.count(pn389.PN389_XGR_IMPORTS_OLD) == 1
        assert src.count(pn389.PN389_XGR_COMPILE_REGEX_OLD) == 1
        assert src.count(pn389.PN389_XGR_VALIDATE_REGEX_OLD) == 1
        assert src.count(pn389.PN389_XGR_VALIDATE_CHOICE_OLD) == 1
        assert src.count(pn389.PN389_XGR_VALIDATE_JSON_OLD) == 1
        assert src.count(pn389.PN389_XGR_VALIDATE_EBNF_OLD) == 1
        assert src.count(pn389.PN389_XGR_VALIDATE_STAG_OLD) == 1

    def test_envs_anchors_unique_and_env_absent(self):
        src = (PIN_TREE / "envs.py").read_text("utf-8")
        assert src.count(pn389.PN389_ENVS_DECL_OLD) == 1
        assert src.count(pn389.PN389_ENVS_LAMBDA_OLD) == 1
        assert "VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS" not in src

    def test_drift_markers_absent_in_pristine(self):
        for rel in (
            "v1/structured_output/utils.py",
            "v1/structured_output/backend_xgrammar.py",
            "envs.py",
        ):
            src = (PIN_TREE / rel).read_text("utf-8")
            for dm in pn389._DRIFT_MARKERS:
                if dm.startswith("[Genesis"):
                    continue
                assert dm not in src, f"drift marker {dm!r} present in {rel}"

    def test_apply_against_real_pin_copy(self, tmp_path, monkeypatch):
        """End-to-end: copy the real pristine files, apply PN389, assert all
        three compile and the env wiring is present. Exercises the byte-exact
        anchors against the live pin tree (not just the hand fixtures)."""
        import shutil

        rels = {
            "v1/structured_output/utils.py": tmp_path / "utils.py",
            "v1/structured_output/backend_xgrammar.py": tmp_path / "backend_xgrammar.py",
            "envs.py": tmp_path / "envs.py",
        }
        for rel, dst in rels.items():
            shutil.copyfile(PIN_TREE / rel, dst)

        def _resolve(rel):
            entry = rels.get(rel)
            return str(entry) if entry else None

        monkeypatch.setattr(pn389, "resolve_vllm_file", _resolve)
        monkeypatch.setattr(pn389, "vllm_install_root", lambda: str(tmp_path))
        import sndr.dispatcher as dispatcher
        monkeypatch.setattr(
            dispatcher, "should_apply", lambda pid: (True, "test override")
        )

        status, reason = pn389.apply()
        assert status == "applied", reason
        for dst in rels.values():
            out = dst.read_text("utf-8")
            compile(out, str(dst), "exec")
        envs_out = rels["envs.py"].read_text("utf-8")
        assert (
            'os.getenv("VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS", "2")'
            in envs_out
        )
