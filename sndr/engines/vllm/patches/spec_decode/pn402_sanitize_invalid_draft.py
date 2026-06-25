# SPDX-License-Identifier: Apache-2.0
"""PN402 — sanitize invalid (-1 / over-vocab) draft token ids before batch prep
(backport+improve OPEN vllm#46574).

Genesis backport+improvement of OPEN vllm#46574 ([Bugfix][SpecDecode]
sanitize invalid draft token ids before batch prep), authored against the
NEW V1 ``vllm/v1/worker/gpu/model_runner.py`` path on live dev424
(pin ``0.23.1rc1.dev424+g3f5a1e173``).

Problem (LIVE on our MTP K=5 + FULL_AND_PIECEWISE PROD)
-------------------------------------------------------
A single invalid draft token id reaching
``scheduler_output.scheduled_spec_decode_tokens`` — ``< 0`` (the MTP
proposer's reject-all / padding sentinel) or ``>= vocab_size`` — produces
an out-of-range index in batch prep (embedding / gather OOB) →
``cudaErrorIllegalAddress`` that hard-crashes the WHOLE engine (all TP
ranks). We run MTP K=5 on both PROD models, so a single bad draft = engine
death. Genesis already carries adjacent guards (PN378 recovered-token
vocab-pad mask, PN361 fail-closed on missing probs, PN133 MTP empty-output)
but the ``-1``-in-drafts ingress on the new ``gpu/`` runner path is a
distinct, currently-unguarded hole.

Fix
---
Inject ``_sanitize_scheduled_spec_decode_tokens`` and call it in
``execute_model`` right after ``self.block_tables.apply_staged_writes()``
and BEFORE the ``total_num_scheduled_tokens == 0`` guard. It walks
``scheduled_spec_decode_tokens``: for any request whose drafts contain an
out-of-range id, it DROPS that request's drafts, decrements its
``num_scheduled_tokens`` by ``len(token_ids)`` (floored at 1 — the request
still owns its own real token), recomputes ``total_num_scheduled_tokens``,
and logs a WARNING + bumps a Prometheus counter. The request then runs as a
normal (non-spec) decode this step instead of crashing the engine.

OUR version over the raw PR (iron rule #10)
-------------------------------------------
1. **Gate on spec_config**: when ``self.speculative_config is None`` the
   sanitize is a no-op — the non-spec path pays ZERO cost (no dict walk).
   The PR runs the scan unconditionally.
2. **Flood-guarded WARNING**: the per-step WARNING is rate-limited (a
   bounded counter) so a sustained bad-draft pathology cannot flood PROD
   logs (the P71 per-step log-flood anti-pattern).
3. **Prometheus counter** ``sndr_invalid_draft_tokens_dropped_total`` (the
   ``sndr_`` prefix matches the repo's spec_decode_metrics convention; the
   design's ``genesis_*`` name is renamed for prefix consistency) so the
   silent case is a metric, not just a log line (PN367 "make the silent
   case visible" doctrine). Counter init is lazy + exception-safe: if
   ``prometheus_client`` is unavailable the patch degrades to log-only.

Composition
-----------
Orthogonal stage (ingress sanitize) vs the downstream accept-side guards —
composes with PN378 / PN361 / PN133 (different defect class) and the
accepted-counts races (PN290 / PN370 / PN398). Different file from PN399
(TQ attn). Supersedes none.

Lifecycle
---------
``default_on=False`` / ``lifecycle=experimental`` pending the rig
validation; a cheap correctness guard on the live MTP path — promote to the
MTP YAMLs after a clean boot. Self-skips once a pin carries vllm#46574
natively (drift marker = the PR's ``_sanitize_scheduled_spec_decode_tokens``
literal).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Upstream: https://github.com/vllm-project/vllm/pull/46574 (OPEN, backport+improve).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn402_sanitize_invalid_draft")

GENESIS_PN402_MARKER = (
    "Genesis PN402 sanitize invalid draft token ids (vllm#46574) v1"
)

ENV_FLAG_FULL = "GENESIS_ENABLE_PN402_SANITIZE_INVALID_DRAFT_TOKENS"

_RUNNER_RELPATH = "v1/worker/gpu/model_runner.py"


# ─────────────────────────────────────────────────────────────────────
# Pure helpers — the single source of truth for the sanitize logic. The
# injected method text below MIRRORS these (the unit tests exercise these
# directly so the contract is tested, not a scaffold).
# ─────────────────────────────────────────────────────────────────────


class FloodGuard:
    """Bounded WARNING rate-limiter — emits at most ``cap`` log lines."""

    def __init__(self, cap: int = 20):
        self.cap = cap
        self.emitted = 0

    def allow(self) -> bool:
        if self.emitted < self.cap:
            self.emitted += 1
            return True
        return False


def sanitize(
    scheduler_output,
    vocab_size: int,
    *,
    log_fn=None,
    counter_fn=None,
    flood_guard: FloodGuard | None = None,
) -> int:
    """Drop out-of-range draft tokens from ``scheduler_output`` in place.

    For each request whose ``scheduled_spec_decode_tokens`` contains an id
    ``< 0`` or ``>= vocab_size``: remove that request's drafts, decrement
    its ``num_scheduled_tokens`` by ``len(token_ids)`` (floored at 1), and
    recompute ``total_num_scheduled_tokens``. Returns the number of requests
    whose drafts were dropped (0 = clean fast path, no mutation).

    Mirrors the injected ``_sanitize_scheduled_spec_decode_tokens`` method
    text exactly; exposed for unit testing the contract.
    """
    spec = scheduler_output.scheduled_spec_decode_tokens
    if not spec:
        return 0
    bad_reqs = [
        req_id
        for req_id, ids in spec.items()
        if any((t < 0 or t >= vocab_size) for t in ids)
    ]
    if not bad_reqs:
        return 0
    num_sched = scheduler_output.num_scheduled_tokens
    for req_id in bad_reqs:
        dropped = len(spec.pop(req_id))
        if req_id in num_sched:
            num_sched[req_id] = max(1, num_sched[req_id] - dropped)
        if counter_fn is not None:
            counter_fn(1)
        if log_fn is not None and (flood_guard is None or flood_guard.allow()):
            log_fn(
                f"[Genesis PN402] dropped out-of-range draft tokens for "
                f"request {req_id!r} (vocab_size={vocab_size}); request "
                f"falls back to normal decode this step (vllm#46574)."
            )
    scheduler_output.total_num_scheduled_tokens = sum(num_sched.values())
    return len(bad_reqs)


# ─────────────────────────────────────────────────────────────────────
# Anchor A — method inject. Insert _sanitize_scheduled_spec_decode_tokens
# into the class body immediately BEFORE `execute_model` (anchored on the
# decorator + def, count==1 on dev424). The injected method gates on
# self.speculative_config (Genesis no-op-fast for the non-spec path), uses
# self.vocab_size, and carries a lazy/exception-safe Prometheus counter +
# a flood-guarded WARNING.
# ─────────────────────────────────────────────────────────────────────

PN402_METHOD_OLD = (
    "    @torch.inference_mode()\n"
    "    def execute_model(\n"
)

PN402_METHOD_NEW = (
    "    # ════════════════════════════════════════════════════════════════\n"
    "    # [Genesis PN402 — backport+improve of vllm#46574]\n"
    "    # Sanitize invalid (-1 / over-vocab) draft token ids before batch\n"
    "    # prep. A single out-of-range draft id reaching batch prep produces\n"
    "    # an embedding/gather OOB -> cudaErrorIllegalAddress that hard-crashes\n"
    "    # the whole engine. Drop the offending request's drafts, decrement\n"
    "    # its num_scheduled_tokens (floored at 1), recompute the total, and\n"
    "    # let it run as a normal decode this step. Genesis improvements over\n"
    "    # the raw PR: (1) no-op when speculative_config is None (non-spec\n"
    "    # path pays zero cost); (2) flood-guarded WARNING (no per-step log\n"
    "    # flood); (3) a lazy/exception-safe Prometheus counter so the silent\n"
    "    # case is a metric, not just a log line.\n"
    "    _genesis_pn402_warn_count = 0\n"
    "    _genesis_pn402_warn_cap = 20\n"
    "    _genesis_pn402_counter = None\n"
    "    _genesis_pn402_counter_init = False\n"
    "\n"
    "    def _sanitize_scheduled_spec_decode_tokens(self, scheduler_output):\n"
    "        if self.speculative_config is None:\n"
    "            return scheduler_output\n"
    "        spec = scheduler_output.scheduled_spec_decode_tokens\n"
    "        if not spec:\n"
    "            return scheduler_output\n"
    "        vocab_size = self.vocab_size\n"
    "        bad_reqs = [\n"
    "            req_id\n"
    "            for req_id, ids in spec.items()\n"
    "            if any((t < 0 or t >= vocab_size) for t in ids)\n"
    "        ]\n"
    "        if not bad_reqs:\n"
    "            return scheduler_output\n"
    "        # Lazy, exception-safe Prometheus counter (degrade to log-only).\n"
    "        if not type(self)._genesis_pn402_counter_init:\n"
    "            type(self)._genesis_pn402_counter_init = True\n"
    "            try:\n"
    "                from prometheus_client import Counter\n"
    "\n"
    "                type(self)._genesis_pn402_counter = Counter(\n"
    '                    "sndr_invalid_draft_tokens_dropped_total",\n'
    '                    "Requests whose out-of-range MTP draft tokens were '
    'dropped before batch prep (Genesis PN402, vllm#46574).",\n'
    "                )\n"
    "            except Exception:  # noqa: BLE001\n"
    "                type(self)._genesis_pn402_counter = None\n"
    "        num_sched = scheduler_output.num_scheduled_tokens\n"
    "        for req_id in bad_reqs:\n"
    "            dropped = len(spec.pop(req_id))\n"
    "            if req_id in num_sched:\n"
    "                num_sched[req_id] = max(1, num_sched[req_id] - dropped)\n"
    "            if type(self)._genesis_pn402_counter is not None:\n"
    "                try:\n"
    "                    type(self)._genesis_pn402_counter.inc()\n"
    "                except Exception:  # noqa: BLE001\n"
    "                    pass\n"
    "            if type(self)._genesis_pn402_warn_count < type(self)._genesis_pn402_warn_cap:\n"
    "                type(self)._genesis_pn402_warn_count += 1\n"
    "                logger.warning(\n"
    '                    "[Genesis PN402] dropped out-of-range draft tokens "\n'
    '                    "for request %s (vocab_size=%d); request falls back "\n'
    '                    "to normal decode this step (vllm#46574).",\n'
    "                    req_id,\n"
    "                    vocab_size,\n"
    "                )\n"
    "        scheduler_output.total_num_scheduled_tokens = sum(num_sched.values())\n"
    "        return scheduler_output\n"
    "\n"
    "    @torch.inference_mode()\n"
    "    def execute_model(\n"
)


# ─────────────────────────────────────────────────────────────────────
# Anchor B — call site. Insert the sanitize call in execute_model after
# apply_staged_writes() and BEFORE the `total_num_scheduled_tokens == 0`
# guard (count==1 on dev424).
# ─────────────────────────────────────────────────────────────────────

PN402_CALLSITE_OLD = (
    "            self.block_tables.apply_staged_writes()\n"
    "            if scheduler_output.total_num_scheduled_tokens == 0:\n"
)

PN402_CALLSITE_NEW = (
    "            self.block_tables.apply_staged_writes()\n"
    "            # [Genesis PN402 — backport+improve of vllm#46574] Drop any\n"
    "            # out-of-range MTP draft token ids before batch prep so a\n"
    "            # single -1 / over-vocab draft cannot OOB-index into the\n"
    "            # embedding gather and crash the engine with a CUDA IMA.\n"
    "            scheduler_output = self._sanitize_scheduled_spec_decode_tokens(\n"
    "                scheduler_output\n"
    "            )\n"
    "            if scheduler_output.total_num_scheduled_tokens == 0:\n"
)


# Self-skip drift marker — the PR's helper literal, present once a pin
# carries vllm#46574 natively, ABSENT in pristine dev424. It is also a
# substring of our own emitted replacement, so the TextPatcher checks the
# idempotency marker (Layer 2) BEFORE the drift markers (Layer 3): the
# drift scan never reads our own output (allowlisted alongside PN399's).
_UPSTREAM_DRIFT_MARKER = "_sanitize_scheduled_spec_decode_tokens"


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_RUNNER_RELPATH)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN402 v1/worker/gpu/model_runner.py — sanitize invalid draft "
            "token ids (vllm#46574)"
        ),
        target_file=str(target),
        marker=GENESIS_PN402_MARKER,
        sub_patches=[
            # Method inject FIRST (required), then the call-site (required).
            TextPatch(
                name="pn402_inject_sanitize_method",
                anchor=PN402_METHOD_OLD,
                replacement=PN402_METHOD_NEW,
                required=True,
            ),
            TextPatch(
                name="pn402_insert_sanitize_call",
                anchor=PN402_CALLSITE_OLD,
                replacement=PN402_CALLSITE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN402",
            _UPSTREAM_DRIFT_MARKER,
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN402 — sanitize invalid draft token ids. Never raises.

    Opt-in through the dispatcher on
    ``GENESIS_ENABLE_PN402_SANITIZE_INVALID_DRAFT_TOKENS`` (default_on=False
    in the registry; a cheap correctness guard on the live MTP path —
    promote to the MTP YAMLs after rig validation).
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN402")
    log_decision("PN402", decision, reason)
    if not decision:
        return "skipped", reason

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN402: target file {_RUNNER_RELPATH} not found"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN402 applied: gpu/model_runner.execute_model now sanitizes "
            "scheduled_spec_decode_tokens before batch prep — any request "
            "with an out-of-range MTP draft id (-1 / >= vocab_size) has its "
            "drafts dropped (num_scheduled decremented, floored at 1) and "
            "runs as a normal decode this step, instead of OOB-indexing the "
            "embedding gather and crashing the engine with a CUDA IMA "
            "(vllm#46574). Genesis: no-op when speculative_config is None, "
            "flood-guarded WARNING, sndr_invalid_draft_tokens_dropped_total "
            "Prometheus counter."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file, encoding="utf-8") as f:
            return GENESIS_PN402_MARKER in f.read()
    except (OSError, UnicodeDecodeError):
        return False


__all__ = [
    "GENESIS_PN402_MARKER",
    "ENV_FLAG_FULL",
    "PN402_METHOD_OLD",
    "PN402_METHOD_NEW",
    "PN402_CALLSITE_OLD",
    "PN402_CALLSITE_NEW",
    "FloodGuard",
    "sanitize",
    "apply",
    "is_applied",
]
