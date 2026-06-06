# SPDX-License-Identifier: Apache-2.0
"""G4_T1 marker-only stub — apply contract for the registry.

The actual fix is the vendored parser file at one of:

  - ``g4_t1_v2_gemma4_tool_parser_pr42237_overlay.py`` (CURRENT, vendor
    of vllm PR #42237 — hermes-style accumulated-text rescan). This is
    the file the launcher binds into the container as of session
    2026-05-31. v2 dropped the third-party ``regex`` import that v1
    needed, but the marker pattern is preserved for clean separation
    between vendored code and the dispatcher apply contract.
  - ``g4_t1_gemma4_tool_parser_pr42006_overlay.py`` (LEGACY v1, vendor
    of vllm PR #42006 — segment-replay refactor). Kept on disk for
    git-blame + operator rollback path; not bind-mounted by default.

Both files are bind-mounted by the operator's launcher (NOT the Genesis
dispatcher). The dispatcher cannot import either file at boot because
they reference engine-side modules only available inside the container.

This marker module exists so:

  - ``test_patch_apply_contracts.TestApplyModule`` can import the
    apply_module path without pulling in the heavy parser
    dependencies.
  - The dispatcher's ``should_apply()`` and ``apply()`` lifecycle
    work uniformly for every registry row (the vendored overlay
    isn't a Genesis runtime patch — it's an operator-side bind-
    mount — so apply() returns "skipped" with a clear reason).

If the operator follows the launcher pattern documented in the
overlay file's header, the vendored parser takes effect via the
``-v`` docker mount. This marker is only the registry-side contract.
"""
from __future__ import annotations


def apply() -> tuple[str, str]:
    """Marker-only apply — G4_T1 is deployed via operator-side bind-mount.

    Returns ``("skipped", "<reason>")`` so the dispatcher logs the
    correct lifecycle even though no runtime mutation is performed.
    """
    return "skipped", (
        "G4_T1 is operator-side ONLY — the vendored gemma4 tool-parser "
        "is deployed via the launcher's `docker run -v ...:ro` bind-mount. "
        "CURRENT vendor (2026-05-31): vllm PR #42237 (hermes-style "
        "rewrite) at `g4_t1_v2_gemma4_tool_parser_pr42237_overlay.py`. "
        "LEGACY vendor (kept for rollback): vllm PR #42006 (segment-"
        "replay) at `g4_t1_gemma4_tool_parser_pr42006_overlay.py`. "
        "Genesis dispatcher does NOT call into the vendored file directly. "
        "Verify the mount from inside the container: `docker exec <c> "
        "head -3 /usr/local/lib/python3.12/dist-packages/vllm/"
        "tool_parsers/gemma4_tool_parser.py` should show the Genesis "
        "G4_T1 header line."
    )


def is_applied() -> bool:
    """G4_T1 has no in-process state — operator-side mount or absent."""
    return False


__all__ = ["apply", "is_applied"]
