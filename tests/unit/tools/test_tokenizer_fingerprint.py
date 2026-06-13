# SPDX-License-Identifier: Apache-2.0
"""TDD for tools/tokenizer_fingerprint.py — the tokenizer-fingerprint
pin-bump gate (pr-sweep-50 roadmap chunk 2 Theme E, upstream #45109).

Upstream #45109 is a test-only PR (AWQ expected outputs changed under
the Transformers v5 tokenizer) — nothing to vendor. The actionable
lesson: a silent tokenizer-behavior change across a pin bump produces
output diffs that get misattributed to Genesis patches (iron-rule-#11
class misdirection — hours of patch bisection for a tokenizer drift).

The gate: sha256 over the token-id sequences of a canonical prompt set,
computed in-container against the model's own tokenizer, diffable
PRE-BENCH on every pin bump (PIN_BUMP_PLAYBOOK step 5b). A changed
fingerprint means "re-baseline outputs / check tokenizer_class" BEFORE
blaming patches.

These tests run against a tiny local tokenizer STUB — no transformers,
no model files, no network. The real-tokenizer leg runs in-container on
the rig (the tool lazy-imports transformers only in load_tokenizer).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "tools" / "tokenizer_fingerprint.py"


def _import():
    name = "_tokenizer_fingerprint_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class StubTokenizer:
    """Tiny deterministic local tokenizer stub.

    Maps each character to a stable small int — enough to exercise the
    full hashing/reporting pipeline without transformers. ``shift``
    simulates a tokenizer-behavior change across a pin bump (same text,
    different ids).
    """

    def __init__(self, shift: int = 0):
        self.shift = shift

    def encode(self, text: str) -> list[int]:
        return [(ord(ch) + self.shift) % 50257 for ch in text]


@pytest.fixture
def mod():
    return _import()


# ═══ 1. Hashing primitives ════════════════════════════════════════════


class TestPromptFingerprint:
    def test_deterministic(self, mod):
        ids = [1, 2, 3, 40000]
        assert mod.prompt_fingerprint(ids) == mod.prompt_fingerprint(ids)

    def test_sensitive_to_any_id_change(self, mod):
        assert mod.prompt_fingerprint([1, 2, 3]) != mod.prompt_fingerprint([1, 2, 4])

    def test_sensitive_to_order(self, mod):
        assert mod.prompt_fingerprint([1, 2, 3]) != mod.prompt_fingerprint([3, 2, 1])

    def test_length_prefix_blocks_concat_ambiguity(self, mod):
        # [1, 23] vs [12, 3] must differ even though "1,23" / "12,3"
        # could collide under naive joining without the length prefix.
        assert mod.prompt_fingerprint([1, 23]) != mod.prompt_fingerprint([12, 3])
        # And the classic empty-vs-zero case.
        assert mod.prompt_fingerprint([]) != mod.prompt_fingerprint([0])

    def test_hex_sha256_shape(self, mod):
        h = mod.prompt_fingerprint([5])
        assert len(h) == 64
        assert set(h) <= set("0123456789abcdef")


# ═══ 2. Canonical prompt set ══════════════════════════════════════════


class TestCanonicalPrompts:
    def test_nonempty_and_ids_unique(self, mod):
        prompts = mod.CANONICAL_PROMPTS
        assert len(prompts) >= 10, "canonical set should cover >=10 classes"
        ids = [pid for pid, _ in prompts]
        assert len(ids) == len(set(ids)), "prompt ids must be unique"

    def test_covers_tokenizer_drift_classes(self, mod):
        """The v5-tokenizer drift classes the gate exists for: code,
        JSON/tool payloads, chat markup, whitespace and multilingual
        segmentation."""
        texts = "\n".join(text for _, text in mod.CANONICAL_PROMPTS)
        assert "def " in texts          # code
        assert '{"' in texts            # JSON tool payload
        assert "<|im_start|>" in texts  # chat template markup
        assert "\t" in texts            # whitespace mix
        assert any(ord(c) > 0x2000 for c in texts)  # non-Latin coverage

    def test_prompt_set_id_versioned(self, mod):
        assert mod.CANONICAL_PROMPT_SET_ID.endswith("v1")


# ═══ 3. Report building ═══════════════════════════════════════════════


class TestBuildReport:
    def test_report_shape(self, mod):
        report = mod.build_report(StubTokenizer(), mod.CANONICAL_PROMPTS,
                                  prompt_set_id=mod.CANONICAL_PROMPT_SET_ID,
                                  model_path="/models/stub")
        assert report["prompt_set"] == mod.CANONICAL_PROMPT_SET_ID
        assert report["model_path"] == "/models/stub"
        assert report["tokenizer_class"] == "StubTokenizer"
        assert len(report["per_prompt"]) == len(mod.CANONICAL_PROMPTS)
        for entry in report["per_prompt"]:
            assert set(entry) >= {"id", "sha256", "num_tokens"}
            assert entry["num_tokens"] > 0
        assert len(report["aggregate_sha256"]) == 64

    def test_stable_across_runs(self, mod):
        r1 = mod.build_report(StubTokenizer(), mod.CANONICAL_PROMPTS,
                              prompt_set_id="x", model_path="m")
        r2 = mod.build_report(StubTokenizer(), mod.CANONICAL_PROMPTS,
                              prompt_set_id="x", model_path="m")
        assert r1["aggregate_sha256"] == r2["aggregate_sha256"]
        assert r1["per_prompt"] == r2["per_prompt"]

    def test_tokenizer_change_flips_aggregate(self, mod):
        """The whole point of the gate: a tokenizer-behavior change
        (same prompts, different ids) MUST change the fingerprint."""
        base = mod.build_report(StubTokenizer(shift=0), mod.CANONICAL_PROMPTS,
                                prompt_set_id="x", model_path="m")
        drift = mod.build_report(StubTokenizer(shift=1), mod.CANONICAL_PROMPTS,
                                 prompt_set_id="x", model_path="m")
        assert base["aggregate_sha256"] != drift["aggregate_sha256"]
        assert all(
            b["sha256"] != d["sha256"]
            for b, d in zip(base["per_prompt"], drift["per_prompt"])
        )


# ═══ 4. Prompts-file loading ══════════════════════════════════════════


class TestPromptsFile:
    def test_json_list_of_strings(self, mod, tmp_path):
        f = tmp_path / "prompts.json"
        f.write_text(json.dumps(["alpha", "beta gamma"]))
        prompts = mod.load_prompts_file(str(f))
        assert prompts == [("prompt_001", "alpha"), ("prompt_002", "beta gamma")]

    def test_json_list_of_objects(self, mod, tmp_path):
        f = tmp_path / "prompts.json"
        f.write_text(json.dumps([{"id": "code", "text": "def f(): pass"}]))
        assert mod.load_prompts_file(str(f)) == [("code", "def f(): pass")]

    def test_plain_text_one_per_line(self, mod, tmp_path):
        f = tmp_path / "prompts.txt"
        f.write_text("first prompt\nsecond with literal \\n newline\n\n")
        prompts = mod.load_prompts_file(str(f))
        assert prompts[0] == ("prompt_001", "first prompt")
        # literal \n sequences unescape to real newlines; blank lines skipped
        assert prompts[1][1] == "second with literal \n newline"
        assert len(prompts) == 2


# ═══ 5. CLI — text output, JSON out, compare gate ═════════════════════


def _loader_factory(shift=0):
    def _load(model_path, trust_remote_code=False):
        return StubTokenizer(shift=shift)
    return _load


class TestCLI:
    def test_text_output_diffable(self, mod, capsys):
        rc = mod.main(["--model-path", "/models/stub"],
                      tokenizer_loader=_loader_factory())
        assert rc == 0
        out = capsys.readouterr().out
        assert "AGGREGATE sha256=" in out
        assert "tokenizer_class=StubTokenizer" in out
        # one line per canonical prompt
        for pid, _ in mod.CANONICAL_PROMPTS:
            assert pid in out

    def test_json_out_written(self, mod, tmp_path, capsys):
        out_file = tmp_path / "fp.json"
        rc = mod.main(["--model-path", "/m", "--json-out", str(out_file)],
                      tokenizer_loader=_loader_factory())
        assert rc == 0
        data = json.loads(out_file.read_text())
        assert data["aggregate_sha256"]
        assert data["prompt_set"] == mod.CANONICAL_PROMPT_SET_ID

    def test_compare_match_exit_0(self, mod, tmp_path, capsys):
        baseline = tmp_path / "base.json"
        assert mod.main(["--model-path", "/m", "--json-out", str(baseline)],
                        tokenizer_loader=_loader_factory()) == 0
        rc = mod.main(["--model-path", "/m", "--compare", str(baseline)],
                      tokenizer_loader=_loader_factory())
        assert rc == 0
        assert "MATCH" in capsys.readouterr().out

    def test_compare_drift_exit_1_names_prompts(self, mod, tmp_path, capsys):
        baseline = tmp_path / "base.json"
        assert mod.main(["--model-path", "/m", "--json-out", str(baseline)],
                        tokenizer_loader=_loader_factory(shift=0)) == 0
        rc = mod.main(["--model-path", "/m", "--compare", str(baseline)],
                      tokenizer_loader=_loader_factory(shift=3))
        assert rc == 1
        out = capsys.readouterr().out
        assert "MISMATCH" in out
        # drifted prompts are named so the operator sees WHICH class moved
        assert mod.CANONICAL_PROMPTS[0][0] in out

    def test_compare_different_prompt_set_exit_2(self, mod, tmp_path, capsys):
        baseline = tmp_path / "base.json"
        custom = tmp_path / "custom.json"
        custom.write_text(json.dumps(["only one prompt"]))
        assert mod.main(["--model-path", "/m", "--json-out", str(baseline),
                         "--prompts-file", str(custom)],
                        tokenizer_loader=_loader_factory()) == 0
        rc = mod.main(["--model-path", "/m", "--compare", str(baseline)],
                      tokenizer_loader=_loader_factory())
        assert rc == 2  # different prompt set: comparison is meaningless

    def test_prompts_file_used(self, mod, tmp_path, capsys):
        f = tmp_path / "p.txt"
        f.write_text("custom probe text\n")
        rc = mod.main(["--model-path", "/m", "--prompts-file", str(f)],
                      tokenizer_loader=_loader_factory())
        assert rc == 0
        out = capsys.readouterr().out
        assert "prompt_001" in out
        assert "prompt_set=" in out

    def test_missing_prompts_file_exit_2(self, mod, tmp_path):
        rc = mod.main(["--model-path", "/m",
                       "--prompts-file", str(tmp_path / "absent.txt")],
                      tokenizer_loader=_loader_factory())
        assert rc == 2

    def test_loader_failure_exit_2(self, mod):
        def _boom(model_path, trust_remote_code=False):
            raise RuntimeError("no transformers here")
        rc = mod.main(["--model-path", "/m"], tokenizer_loader=_boom)
        assert rc == 2
