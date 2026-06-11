# SPDX-License-Identifier: Apache-2.0
"""G4_T1 overlays — dict-key STRING_DELIM strip (upstream PR #44717).

Upstream issue #44715: ``_parse_gemma4_args()`` strips ``<|"|>``
(STRING_DELIM) from value positions but not from key positions, so any
dict-keyed tool argument the model emits with quoted keys leaks the
sentinel characters into the dict key string (e.g.
``{'<|"|>3<|"|>': 'new text'}`` instead of ``{'3': 'new text'}``).
Upstream PR #44717 (OPEN at vendor time 2026-06-11) adds the same
STRING_DELIM strip to the key position that values already get.

Both Genesis G4_T1 overlay variants vendor the buggy key scanner:

* ``g4_t1_v2_gemma4_tool_parser_pr42237_overlay.py`` (CURRENT, mounted)
* ``g4_t1_gemma4_tool_parser_pr42006_overlay.py``    (LEGACY rollback)

Test strategy: the overlay modules import vllm at module top (not
importable in the dev venv), so the pure parser functions are extracted
via AST from the exact shipped source text and exec'd into an isolated
namespace with a stub logger. This exercises the code that is actually
bind-mounted into the container, not a copy.

Test cases mirror upstream PR #44717's additions to
``tests/tool_parsers/test_gemma4_tool_parser.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PATCH_DIR = REPO_ROOT / "sndr" / "engines" / "vllm" / "patches" / "tool_parsing"

OVERLAYS = {
    "v2_pr42237_current": PATCH_DIR
    / "g4_t1_v2_gemma4_tool_parser_pr42237_overlay.py",
    "v1_pr42006_legacy": PATCH_DIR
    / "g4_t1_gemma4_tool_parser_pr42006_overlay.py",
}

_WANTED_FUNCS = {
    "_parse_gemma4_value",
    "_parse_gemma4_args",
    "_parse_gemma4_array",
}
_WANTED_CONSTS = {"STRING_DELIM", "TOOL_CALL_START", "TOOL_CALL_END"}


class _StubLogger:
    """Minimal logger stand-in for the extracted parser functions."""

    def warning(self, *args, **kwargs) -> None:
        pass

    def debug(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


def _load_parser_namespace(path: Path) -> dict:
    """AST-extract the pure parser functions from an overlay file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED_FUNCS:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & _WANTED_CONSTS:
                selected.append(node)
    found_funcs = {n.name for n in selected if isinstance(n, ast.FunctionDef)}
    assert found_funcs == _WANTED_FUNCS, (
        f"overlay {path.name} drifted: expected funcs {_WANTED_FUNCS}, "
        f"found {found_funcs}"
    )
    module = ast.Module(body=selected, type_ignores=[])
    namespace: dict = {"logger": _StubLogger()}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture(params=sorted(OVERLAYS), ids=sorted(OVERLAYS))
def parse_args(request):
    namespace = _load_parser_namespace(OVERLAYS[request.param])
    return namespace["_parse_gemma4_args"]


def test_string_delim_wrapped_key_is_stripped(parse_args) -> None:
    """Keys wrapped in <|"|>...<|"|> parse to clean strings (PR #44717)."""
    result = parse_args('record_map:{<|"|>3<|"|>:<|"|>new text<|"|>}')
    assert result == {"record_map": {"3": "new text"}}


def test_string_delim_wrapped_key_nested(parse_args) -> None:
    """Nested dict keys with <|"|> wrappers are stripped at every depth."""
    result = parse_args('outer:{<|"|>k1<|"|>:{<|"|>k2<|"|>:<|"|>v<|"|>}}')
    assert result == {"outer": {"k1": {"k2": "v"}}}


def test_bare_keys_still_parse_unchanged(parse_args) -> None:
    """Regression guard: bare-identifier keys (the common case) unchanged."""
    result = parse_args('location:<|"|>Paris<|"|>,count:42')
    assert result == {"location": "Paris", "count": 42}


def test_top_level_string_delim_key_stripped(parse_args) -> None:
    """Top-level (non-nested) quoted key is also stripped."""
    result = parse_args('<|"|>weird key<|"|>:<|"|>v<|"|>')
    assert result == {"weird key": "v"}


def test_lone_delimiter_key_not_over_stripped(parse_args) -> None:
    """A key of exactly one STRING_DELIM must not be sliced negatively."""
    result = parse_args('<|"|>:42')
    assert result == {'<|"|>': 42}
