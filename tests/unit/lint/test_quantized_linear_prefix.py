# SPDX-License-Identifier: Apache-2.0
"""AST lint: quantized parallel-Linear constructions must pass prefix=.

Lesson adopted from upstream PR vllm#44837 (issue #44817) — roadmap
chunk-4 Theme 2, "loud startup" validation family together with
PN380's load-coverage guard. We adopt the LESSON, not the code (the
DSV4 files it fixes are not in our engaged set).

The bug class: ``DeepSeekV4MultiTokenPredictorLayer.__init__`` created
``e_proj`` / ``h_proj`` (both ``ReplicatedLinear``) with
``quant_config=quant_config`` but WITHOUT ``prefix=``. The module then
registers under ``prefix=""`` and, when a prefix-matching quant config
(compressed-tensors / AWQ / AutoRound ``inc``) is active,
``get_scheme(layer_name="")`` either raises at startup
(``ValueError: Unable to find matching target for ''``) or — the
nastier variant — matches the WRONG scheme and silently
mis-quantizes the layer. No ignore-regex can fix it from the artifact
side: a pattern matching ``""`` matches every module.

This lint walks the AST of:
  (a) the pristine pin model files our patches anchor (discovered
      dynamically by grepping the patch sources for
      ``model_executor/models/*.py`` targets, so new patch targets are
      enrolled automatically), and
  (b) our vendored overlay copies inside the repo (discovered by the
      vLLM SPDX header),

and flags every call to a parallel-Linear-family class that passes a
non-None ``quant_config`` without an explicit ``prefix=`` keyword.

Allowlist: entries are ``"<basename>::<scope>::<class>"`` where
``<scope>`` is the dotted enclosing class/function chain — stable
across line-number drift. Every entry must carry a justification
comment. As of 2026-06-11 (pin g303916e93) the whole scan surface is
clean, so the allowlist is empty; it exists for verified-safe future
hits (e.g. a constructor that provably reassigns the prefix later).

Scan cost: <1s, pure ast — no torch, no vllm import.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm")
PATCHES_ROOT = REPO_ROOT / "sndr" / "engines" / "vllm" / "patches"

# Parallel-Linear family + LM head — every constructor that takes
# quant_config AND registers itself under a prefix-derived layer name.
QUANT_PREFIX_CLASSES = frozenset({
    "ColumnParallelLinear",
    "RowParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "QKVCrossParallelLinear",
    "ReplicatedLinear",
    "ParallelLMHead",
})

# Verified-safe constructions: "<basename>::<scope>::<class>" plus a
# justification comment above each entry. Keep EMPTY unless a hit has
# been studied per the six-step rule (study/analyze/verify/search/
# compare/change) and proven safe.
ALLOWLIST: frozenset[str] = frozenset()

_TARGET_RE = re.compile(r'"(model_executor/models/[a-z0-9_]+\.py)"')


def _engaged_pristine_files() -> list[Path]:
    """Pristine model files our patch sources anchor — discovered from
    the patch tree so newly-engaged model files enroll automatically."""
    rels: set[str] = set()
    for py in PATCHES_ROOT.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8")
        except OSError:
            continue
        rels.update(_TARGET_RE.findall(text))
    return sorted(PIN_TREE / rel for rel in rels)


def _overlay_copies() -> list[Path]:
    """Vendored full-file vLLM copies inside the repo (carry the vLLM
    SPDX header) — our overlays must satisfy the same invariant."""
    out = []
    for py in (REPO_ROOT / "sndr").rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            head = py.read_text(encoding="utf-8")
        except OSError:
            continue
        if "Copyright contributors to the vLLM project" in head:
            out.append(py)
    return sorted(out)


def find_quantized_linear_without_prefix(path: Path) -> list[str]:
    """All RAW findings in one file as 'basename::scope::Class@line'.

    The ALLOWLIST is applied by callers (``_unallowed``) so the
    hygiene test can detect stale allowlist entries from the raw set.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:  # a broken scan target is itself a failure
        return [f"{path.name}::<syntax error: {e}>"]

    violations: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def _scoped(self, node: ast.AST) -> None:
            self.scope.append(getattr(node, "name", "?"))
            self.generic_visit(node)
            self.scope.pop()

        visit_ClassDef = _scoped
        visit_FunctionDef = _scoped
        visit_AsyncFunctionDef = _scoped

        def visit_Call(self, node: ast.Call) -> None:
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in QUANT_PREFIX_CLASSES:
                kwarg_names = {kw.arg for kw in node.keywords}
                has_splat = None in kwarg_names  # **kwargs — unanalyzable
                qc = next(
                    (kw for kw in node.keywords if kw.arg == "quant_config"),
                    None,
                )
                qc_is_literal_none = (
                    qc is not None
                    and isinstance(qc.value, ast.Constant)
                    and qc.value.value is None
                )
                if (
                    qc is not None
                    and not qc_is_literal_none
                    and "prefix" not in kwarg_names
                    and not has_splat
                ):
                    scope = ".".join(self.scope) or "<module>"
                    violations.append(f"{path.name}::{scope}::{name}@{node.lineno}")
            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


def _unallowed(raw_findings: list[str]) -> list[str]:
    """Raw findings minus allowlisted keys (key = finding without the
    @line suffix, stable across line drift)."""
    return [
        v for v in raw_findings if v.rsplit("@", 1)[0] not in ALLOWLIST
    ]


def _format_report(violations: list[str]) -> str:
    return (
        "quantized parallel-Linear constructed without prefix= "
        "(vllm#44837 class: empty layer_name -> startup ValueError on "
        "compressed-tensors, or silent AWQ/AutoRound mis-quantization):\n  "
        + "\n  ".join(violations)
        + "\nFix: thread prefix=f\"{prefix}.<attr>\" through the "
        "constructor (preferred), or add the entry to ALLOWLIST in "
        "tests/unit/lint/test_quantized_linear_prefix.py WITH a "
        "justification comment after a six-step study."
    )


class TestScanSurface:
    def test_patch_target_discovery_nonempty(self):
        """The dynamic discovery must keep finding our engaged model
        files — an empty set would mean the lint silently checks
        nothing (regex or layout drift)."""
        rels = {p.name for p in _engaged_pristine_files()}
        assert "qwen3_5_mtp.py" in rels, (
            f"discovery lost the PN348/PN380 target; found: {sorted(rels)}"
        )

    def test_overlay_discovery_nonempty(self):
        names = {p.name for p in _overlay_copies()}
        # the TQ overlay family + tool-parser overlays are vendored copies
        assert names, "no vendored vLLM overlay copies discovered"


class TestSelfCheck:
    """The detector must actually detect — guard against a silently
    broken visitor (the lint-that-lints-nothing failure mode)."""

    BAD = (
        "class L:\n"
        "    def __init__(self, quant_config, prefix):\n"
        "        self.e_proj = ReplicatedLinear(\n"
        "            4, 4, bias=False, quant_config=quant_config,\n"
        "        )\n"
    )
    GOOD = BAD.replace(
        "quant_config=quant_config,",
        'quant_config=quant_config, prefix=f"{prefix}.e_proj",',
    )
    NONE_QC = BAD.replace("quant_config=quant_config", "quant_config=None")

    def _scan_text(self, tmp_path, text):
        f = tmp_path / "synthetic_model.py"
        f.write_text(text, encoding="utf-8")
        return find_quantized_linear_without_prefix(f)

    def test_detects_the_44837_shape(self, tmp_path):
        v = self._scan_text(tmp_path, self.BAD)
        assert len(v) == 1
        assert "synthetic_model.py::L.__init__::ReplicatedLinear" in v[0]

    def test_passes_with_prefix(self, tmp_path):
        assert self._scan_text(tmp_path, self.GOOD) == []

    def test_literal_none_quant_config_is_exempt(self, tmp_path):
        """quant_config=None is explicitly unquantized — prefix is
        irrelevant for scheme matching."""
        assert self._scan_text(tmp_path, self.NONE_QC) == []

    def test_kwargs_splat_is_exempt(self, tmp_path):
        text = (
            "def make(**kw):\n"
            "    return ColumnParallelLinear(4, 4, quant_config=qc, **kw)\n"
        )
        assert self._scan_text(tmp_path, text) == []


@pytest.mark.skipif(
    not PIN_TREE.is_dir(),
    reason="pristine pin tree not present on this machine",
)
class TestPristineEngagedModelFiles:
    def test_engaged_pristine_files_clean(self):
        violations: list[str] = []
        scanned = 0
        for path in _engaged_pristine_files():
            if not path.is_file():
                continue  # patch target absent from this pin era
            scanned += 1
            violations.extend(
                _unallowed(find_quantized_linear_without_prefix(path))
            )
        assert scanned > 0, "no pristine scan targets resolved on the pin"
        assert not violations, _format_report(violations)


class TestRepoOverlayCopies:
    def test_overlay_copies_clean(self):
        violations: list[str] = []
        for path in _overlay_copies():
            violations.extend(
                _unallowed(find_quantized_linear_without_prefix(path))
            )
        assert not violations, _format_report(violations)


class TestAllowlistHygiene:
    def test_no_stale_allowlist_entries(self):
        """A stale allowlist entry (file renamed, violation fixed) must
        be pruned — otherwise it can mask a future regression at the
        same key."""
        if not ALLOWLIST:
            pytest.skip("allowlist empty (the desired steady state)")
        targets = list(_overlay_copies())
        if PIN_TREE.is_dir():
            targets += [p for p in _engaged_pristine_files() if p.is_file()]
        live_keys = {
            v.rsplit("@", 1)[0]
            for path in targets
            for v in find_quantized_linear_without_prefix(path)
        }
        stale = ALLOWLIST - live_keys
        assert not stale, (
            f"stale allowlist entries (no matching raw finding): "
            f"{sorted(stale)}"
        )
