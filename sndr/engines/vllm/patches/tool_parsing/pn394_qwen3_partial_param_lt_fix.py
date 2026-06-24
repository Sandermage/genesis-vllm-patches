# SPDX-License-Identifier: Apache-2.0
"""PN394 — Qwen3 partial-param regex drops argument values containing `<`.

RETIRED 2026-06-24 (pin bump dev148 -> dev301) — superseded by vllm#46047,
which is IN dev301 (the anchor drifted per the dev301 anchor-SOT regen).
Registry lifecycle=retired, capped <0.23.1rc1.dev301. The transform below is
UNCHANGED and still applies on dev148 (the previous/rollback pin, where the
bug is live); on dev301+ the version cap + the post-fix `>(.*)$` drift marker
self-skip it. Do not delete the module while dev148 rollback is possible.

Problem
-------
On the live pin (0.23.1rc1.dev148+gb4c80ec0f) the engine-native Qwen3
XML parser ``vllm/parser/qwen3.py`` builds a partial-parameter regex
that captures the argument VALUE with ``([^<]*)$``:

    _PARTIAL_PARAM_RE = re.compile(
        r"<\\s*parameter\\s*=\\s*([^>]+)>([^<]*)$", re.DOTALL
    )

The value group ``([^<]*)`` stops at the first ``<``. During streaming,
a still-open (``partial=True``) tool-call argument whose value contains
a literal ``<`` — common in code / HTML / math / comparison operators
(``if a < b``, ``<div>``, generics ``List<T>``) — is SILENTLY TRUNCATED
at that ``<``: the model emits ``{"expr": "a < b"}`` but the client
receives ``{"expr": "a "}``. This is a real tool-call correctness bug,
not a cosmetic one — strict agents act on the truncated argument.

Our 27B / 35B production presets run ``--tool-call-parser qwen3_xml``,
which on 0.23.x resolves to this engine parser, so the bug is on the
hot streaming path.

Fix (vllm#46047 — MERGED 2026-06-18, AFTER our dev148 pin)
----------------------------------------------------------
The upstream one-line fix widens the value group from ``([^<]*)`` to
``(.*)`` (``re.DOTALL`` already lets ``.`` span newlines), so the
partial value is captured to end-of-string regardless of any ``<``:

    -_PARTIAL_PARAM_RE = re.compile(r"<\\s*parameter\\s*=\\s*([^>]+)>([^<]*)$", re.DOTALL)
    +_PARTIAL_PARAM_RE = re.compile(r"<\\s*parameter\\s*=\\s*([^>]+)>(.*)$", re.DOTALL)

PN394 is a byte-exact text-patch of that single line. ``required=True``
is correct: the target line EXISTS verbatim on dev148 (byte-verified
against ``vllm/parser/qwen3.py@b4c80ec0f``), so a missing anchor means
the parser drifted and the patch must skip loudly rather than silently
ship a no-op.

Self-skip on a future pin that already carries #46047: the post-fix
spelling ``>(.*)$`` is the ``upstream_drift_markers`` entry. Once a pin
ships the merged form, the marker is present and PN394 auto-skips
(``upstream merged``) before touching the file. The marker is disjoint
from PN394's own emitted text only in the sense that PN394 WRITES the
post-fix form — but the idempotency marker (the ``[Genesis ...]``
comment line) gates re-application first, so the drift check never
mis-fires on our own output. (Confirmed: the drift check runs AFTER the
idempotency-marker short-circuit in ``apply()`` below.)

Verification (2026-06-19, gh against ref b4c80ec0f):
  * Anchor present in dev148 ``vllm/parser/qwen3.py`` — count == 1.
  * Post-fix marker ``>(.*)$`` ABSENT in dev148 (so it does not
    self-skip on the deployed pin).
  * #46047 MERGED 2026-06-18 (touches vllm/parser/qwen3.py +1/-1).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Upstream: https://github.com/vllm-project/vllm/pull/46047 (MERGED).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.env import Flags, is_enabled
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn394_qwen3_partial_param_lt_fix")

GENESIS_PN394_MARKER = (
    "Genesis PN394 qwen3 partial-param value `<` truncation fix (vllm#46047) v1"
)

# Full env var name (for tests / operator docs); the canonical bare flag
# lives in sndr.env.Flags.PN394_QWEN3_PARTIAL_PARAM_LT_FIX.
ENV_FLAG_FULL = "GENESIS_ENABLE_PN394_QWEN3_PARTIAL_PARAM_LT_FIX"

_TARGET_RELPATH = "parser/qwen3.py"

# Pristine dev148 form — value group stops at the first `<`.
ANCHOR_OLD = (
    '_PARTIAL_PARAM_RE = re.compile(r"<\\s*parameter\\s*=\\s*([^>]+)>([^<]*)$", re.DOTALL)\n'
)

# vllm#46047 fix — widen the value group to `(.*)$` (re.DOTALL already
# lets `.` span newlines) so a partial argument value containing `<`
# is no longer truncated at that `<`.
ANCHOR_NEW = (
    "# [Genesis PN394 vllm#46047] widen the partial-param value group from\n"
    "# ([^<]*) to (.*) so a streaming (partial=True) tool-call argument value\n"
    "# containing a literal `<` (code/HTML/math/generics) is not silently\n"
    "# truncated at that `<`. re.DOTALL already spans newlines.\n"
    '_PARTIAL_PARAM_RE = re.compile(r"<\\s*parameter\\s*=\\s*([^>]+)>(.*)$", re.DOTALL)\n'
)

# Post-fix spelling. Present only once a pin carries #46047 -> PN394
# auto-skips. Checked AFTER the idempotency marker, so it never trips on
# PN394's own (post-fix) output.
_UPSTREAM_DRIFT_MARKER = '>(.*)$'


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_RELPATH)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN394 qwen3 partial-param `<` truncation fix (vllm#46047)",
        target_file=target,
        marker=GENESIS_PN394_MARKER,
        sub_patches=[
            TextPatch(
                name="pn394_partial_param_value_group",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[_UPSTREAM_DRIFT_MARKER],
    )


def apply() -> tuple[str, str]:
    """Apply PN394 wiring. Never raises."""
    if not is_enabled(Flags.PN394_QWEN3_PARTIAL_PARAM_LT_FIX, default=True):
        return "skipped", (
            f"PN394 disabled (set {ENV_FLAG_FULL}=0 to opt out of the "
            "qwen3 partial-param `<`-truncation fix; default ON — vllm#46047)"
        )
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"{_TARGET_RELPATH} not found in vllm install"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", (
            "PN394 applied: qwen3 partial-param value group widened to (.*) "
            "— streaming tool-call argument values containing `<` are no "
            "longer truncated (vllm#46047)"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", (
            f"{msg} — likely the pin already carries vllm#46047 (post-fix "
            "regex present) or the qwen3 parser was reshaped"
        )
    return "failed", failure.reason if failure else "unknown failure"


def is_applied() -> bool:
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file, encoding="utf-8") as fh:
            return GENESIS_PN394_MARKER in fh.read()
    except OSError:
        return False


__all__ = [
    "GENESIS_PN394_MARKER",
    "ENV_FLAG_FULL",
    "ANCHOR_OLD",
    "ANCHOR_NEW",
    "apply",
    "is_applied",
]
