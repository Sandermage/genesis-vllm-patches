# SPDX-License-Identifier: Apache-2.0
"""PN401 — TurboQuant prefill continuation guard (backport+improve OPEN vllm#46461).

Genesis backport+improvement of OPEN vllm#46461 ([Bugfix][TurboQuant] guard
the flash_attn prefill fast path against silently dropping cached prefix
K/V on co-batched continuation requests), authored against live dev424
(pin ``0.23.1rc1.dev424+g3f5a1e173``).

Problem (LIVE on our TQ-always-on PROD)
---------------------------------------
``_prefill_attention`` takes a flash_attn fast path on::

    if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
        return self._flash_attn_varlen(..., cu_seqlens_k=attn_metadata.query_start_loc, ...)

The comment claims ``max_query_len == max_seq_len`` means "no request has
prior cached KV". That is INSUFFICIENT. A long first-chunk prefill
(``q_len == seq_len``) can inflate ``max_query_len`` to equal
``max_seq_len`` while the SAME batch step carries one or more shorter
CONTINUATION requests (``q_len < seq_len`` — a non-first chunked-prefill
chunk, OR a prefix-cache hit). For those continuations the fast path
passes ``cu_seqlens_k = query_start_loc`` (the QUERY offsets, not the
full sequence offsets), so flash_attn attends only to the current
chunk's raw K/V and silently DROPS the cached prefix K/V → wrong
attention → hallucination / garbled output on the continued request.

This needs NO prefix caching to fire: ``enable_chunked_prefill: true`` is
on for our TQ k8v4 profiles, so a long prompt is split into chunks and
non-first chunks are continuations. Under multi-concurrency a fresh
full-prompt prefill can land in the same batch step as a mid-chunk
continuation; if the fresh prompt's ``q_len == its seq_len == batch
max_seq_len``, the gate trips and the continuation loses its prefix.
TurboQuant k8v4 is ALWAYS ON on both PROD models (27B + 35B), so this is
a live correctness bug, not gated behind any flag we have OFF.

Fix
---
Compute a host-side continuation check on the CPU-mirror metadata tensors
(``query_start_loc_cpu`` / ``seq_lens_cpu`` — no GPU sync) BEFORE the fast
path and gate it with ``and not _has_continuation``. A continuation
exists iff for any request ``i`` the per-request query length
``qsl[i+1] - qsl[i]`` differs from ``seq_lens[i]`` (i.e. ``q_len <
seq_len``). When any continuation exists, fall through to the existing
per-request branch (which reads cached K/V correctly via the TQ decode
kernel / ``_continuation_prefill``).

OUR version over the raw PR (iron rule #10)
-------------------------------------------
1. **Conservative None-mirror fall-safe.** The PR guards on
   ``query_start_loc_cpu is not None and seq_lens_cpu is not None`` and
   on a None mirror implicitly evaluates ``_has_continuation = False``
   (i.e. TAKES the unsafe fast path). We invert that: a missing CPU
   mirror sets ``_has_continuation = True`` (SKIP the fast path). The
   fast path is only a perf optimization; losing it on the rare
   None-mirror step is cheap versus a silent hallucination. Correctness
   first.
2. **Length-mismatch hardening.** If the CPU tensors are shape-
   inconsistent (``len(qsl) < len(seq_lens) + 1``) — e.g. a builder
   shape drift — we also set ``_has_continuation = True`` and fall to the
   safe path rather than risk an out-of-range slice or a partial check.

Both improvements only ever ADD safety (skip the fast path more often);
they never re-enable the buggy path. On the common all-first-chunk batch
``_has_continuation`` is False and the fast path is preserved unchanged
(no perf regression on the hot path).

Composition
-----------
Disjoint anchors, composes with every TQ sibling:
  * P101  — continuation 64-token slicing (anchors the continuation LOOP
    below the gate; PN401 anchors the GATE above it).
  * PN116 — prefill ``max_seq_len`` fallback (anchors the ``forward``
    dispatch ``prefill_max_seq`` block, a different method).
  * PN399 — TQ decode-scratch fixed buffer (anchors ``_decode_attention``
    + ``__init__`` + module consts, never ``_prefill_attention``'s gate).
PN401 supersedes none. It is NOT folded into P101 (opt-in perf,
``default_on=False``) or PN116 (HW-gated, skips on Hopper) because the
continuation guard is a correctness fix that must be ON on ALL TQ
hardware regardless of either patch's state.

Lifecycle
---------
``default_on=False`` in the registry / ``lifecycle=experimental`` pending
the PROD A/B, but it is a correctness patch and is enabled (``'1'``) on
the live 27B + 35B TQ k8v4 model YAMLs. Self-skips once a pin carries
vllm#46461 natively: the drift marker is the PR's ``_has_continuation``
literal — present once upstream merges, ABSENT in pristine dev424.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Upstream: https://github.com/vllm-project/vllm/pull/46461 (OPEN, backport+improve).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn401_tq_prefill_continuation_guard")

GENESIS_PN401_MARKER = (
    "Genesis PN401 TurboQuant prefill continuation guard (vllm#46461) v1"
)

# Full env var name (for tests / operator docs); the canonical bare flag
# lives in sndr.env.Flags.PN401_TQ_PREFILL_CONTINUATION_GUARD.
ENV_FLAG_FULL = "GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD"

_TQ_RELPATH = "v1/attention/backends/turboquant_attn.py"


# ─────────────────────────────────────────────────────────────────────
# Anchor — the dev424 fast-path gate (3 comment lines + the `if`).
# Byte-exact from pristine dev424 (count == 1, the only flash_attn fast
# path keyed on max_query_len == max_seq_len). The em-dash in "Python
# ints — no GPU sync" is U+2014, byte-verified.
# ─────────────────────────────────────────────────────────────────────

PN401_FASTPATH_OLD = (
    "        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).\n"
    "        # max_query_len == max_seq_len means no request has prior cached KV.\n"
    "        # Both are Python ints — no GPU sync.\n"
    "        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:\n"
)

PN401_FASTPATH_NEW = (
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # [Genesis PN401 — backport+improve of vllm#46461]\n"
    "        # The flash_attn fast path below fires on\n"
    "        # `max_query_len == max_seq_len` and passes\n"
    "        # `cu_seqlens_k = query_start_loc` (the QUERY offsets). That is\n"
    "        # only correct when EVERY request is a first-chunk prefill\n"
    "        # (q_len == seq_len). A long first-chunk prefill can inflate\n"
    "        # max_query_len to equal max_seq_len while the SAME batch carries\n"
    "        # shorter continuation requests (q_len < seq_len — a non-first\n"
    "        # chunked-prefill chunk or a prefix-cache hit). For those the fast\n"
    "        # path would attend only to the current chunk's raw K/V and\n"
    "        # silently DROP the cached prefix K/V -> hallucination. Detect any\n"
    "        # continuation on the CPU-mirror tensors (no GPU sync) and skip the\n"
    "        # fast path when present. Genesis hardening over the raw PR: if a\n"
    "        # CPU mirror is None OR shape-inconsistent we conservatively treat\n"
    "        # this as a continuation (skip the fast path) — a missing mirror\n"
    "        # must never silently re-enable the buggy path (the fast path is\n"
    "        # only a perf optimization; the per-request branch below is\n"
    "        # always correct).\n"
    "        _pn401_qsl_cpu = attn_metadata.query_start_loc_cpu\n"
    "        _pn401_seq_lens_cpu = attn_metadata.seq_lens_cpu\n"
    "        if _pn401_qsl_cpu is None or _pn401_seq_lens_cpu is None:\n"
    "            _has_continuation = True\n"
    "        else:\n"
    "            _pn401_qsl = _pn401_qsl_cpu.tolist()\n"
    "            _pn401_sl = _pn401_seq_lens_cpu.tolist()\n"
    "            _pn401_n = len(_pn401_sl)\n"
    "            if len(_pn401_qsl) < _pn401_n + 1:\n"
    "                _has_continuation = True\n"
    "            else:\n"
    "                _has_continuation = any(\n"
    "                    (_pn401_qsl[i + 1] - _pn401_qsl[i]) != _pn401_sl[i]\n"
    "                    for i in range(_pn401_n)\n"
    "                )\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).\n"
    "        # max_query_len == max_seq_len means no request has prior cached KV.\n"
    "        # Both are Python ints — no GPU sync.\n"
    "        # [Genesis PN401] gated with `not _has_continuation` (vllm#46461).\n"
    "        if (\n"
    "            _HAS_FLASH_ATTN\n"
    "            and attn_metadata.max_query_len == attn_metadata.max_seq_len\n"
    "            and not _has_continuation\n"
    "        ):\n"
)


# Self-skip drift marker — the PR's `_has_continuation` literal is present
# once a pin carries vllm#46461 natively, ABSENT in pristine dev424. It is
# also a substring of OUR own emitted replacement, so the TextPatcher checks
# the idempotency marker (Layer 2) BEFORE the drift markers (Layer 3): the
# drift scan never reads our own output (allowlisted with PN399's markers).
_UPSTREAM_DRIFT_MARKER = "_has_continuation"


def eval_fast_path_taken(attn_metadata, *, has_flash_attn: bool) -> bool:
    """Return whether the PATCHED fast path is taken for ``attn_metadata``.

    This is the exact predicate the patch text inserts, exposed as a pure
    function so the bug-repro unit tests exercise the real guard logic (no
    scaffold — the patch source and this function are the single source of
    truth for the continuation check). A continuation exists iff some
    request's query length differs from its full sequence length. Missing
    or shape-inconsistent CPU mirrors conservatively count as a
    continuation (Genesis fall-safe).
    """
    qsl_cpu = attn_metadata.query_start_loc_cpu
    seq_lens_cpu = attn_metadata.seq_lens_cpu
    if qsl_cpu is None or seq_lens_cpu is None:
        has_continuation = True
    else:
        qsl = qsl_cpu.tolist() if hasattr(qsl_cpu, "tolist") else list(qsl_cpu)
        sl = seq_lens_cpu.tolist() if hasattr(seq_lens_cpu, "tolist") else list(seq_lens_cpu)
        n = len(sl)
        if len(qsl) < n + 1:
            has_continuation = True
        else:
            has_continuation = any(
                (qsl[i + 1] - qsl[i]) != sl[i] for i in range(n)
            )
    return bool(
        has_flash_attn
        and attn_metadata.max_query_len == attn_metadata.max_seq_len
        and not has_continuation
    )


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TQ_RELPATH)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN401 v1/attention/backends/turboquant_attn.py — prefill "
            "continuation guard (vllm#46461)"
        ),
        target_file=str(target),
        marker=GENESIS_PN401_MARKER,
        sub_patches=[
            TextPatch(
                name="pn401_prefill_continuation_guard",
                anchor=PN401_FASTPATH_OLD,
                replacement=PN401_FASTPATH_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN401",
            # `_has_continuation` is the PR's literal AND a substring of our
            # own replacement; idempotency (Layer 2) is checked before the
            # drift scan (Layer 3) so it never self-collides. Listed for the
            # pin-bump preflight deep-diff, allowlisted with PN399's markers.
            _UPSTREAM_DRIFT_MARKER,
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN401 — TurboQuant prefill continuation guard. Never raises.

    Opt-in through the dispatcher on
    ``GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD`` (default_on=False
    in the registry; enabled on the live 27B/35B TQ k8v4 YAMLs as a
    correctness fix).
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN401")
    log_decision("PN401", decision, reason)
    if not decision:
        return "skipped", reason

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN401: target file {_TQ_RELPATH} not found"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN401 applied: TurboQuant `_prefill_attention` flash_attn fast "
            "path is now gated with `not _has_continuation` — a batch that "
            "co-locates a first-chunk prefill (max_query_len == max_seq_len) "
            "with a continuation request (q_len < seq_len) falls through to "
            "the per-request branch that reads cached prefix K/V correctly, "
            "instead of silently dropping it (vllm#46461). Genesis fall-safe: "
            "a None / shape-inconsistent CPU mirror also skips the fast path."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    """Return True iff our marker is present in the target file."""
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file, encoding="utf-8") as f:
            return GENESIS_PN401_MARKER in f.read()
    except (OSError, UnicodeDecodeError):
        return False


__all__ = [
    "GENESIS_PN401_MARKER",
    "ENV_FLAG_FULL",
    "PN401_FASTPATH_OLD",
    "PN401_FASTPATH_NEW",
    "eval_fast_path_taken",
    "apply",
    "is_applied",
]
