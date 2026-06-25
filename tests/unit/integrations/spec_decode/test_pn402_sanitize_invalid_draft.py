# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN402 — sanitize invalid (-1 / over-vocab) draft token ids
before batch prep (backport+improve OPEN vllm#46574).

Genesis backport+improvement of OPEN vllm#46574 ([Bugfix][SpecDecode]
sanitize invalid draft token ids before batch prep), authored against the
NEW V1 ``vllm/v1/worker/gpu/model_runner.py`` path on live dev424.

The bug
-------
A single invalid draft token id (``< 0`` — the MTP proposer's reject-all /
padding sentinel — or ``>= vocab_size``) reaching
``scheduled_spec_decode_tokens`` produces an out-of-range index in batch
prep (embedding / gather OOB) → ``cudaErrorIllegalAddress`` that hard-
crashes the whole engine. We run MTP K=5 + FULL_AND_PIECEWISE on both PROD
models, so a single bad draft = engine death.

The fix
-------
Before ``execute_model`` runs the model, walk
``scheduled_spec_decode_tokens``: for any request whose drafts contain an
out-of-range id, DROP that request's drafts, decrement its
``num_scheduled_tokens`` by ``len(token_ids)`` (floored at 1 — the request
still has its own real token), recompute ``total_num_scheduled_tokens``,
and log a WARNING + bump a Prometheus counter. All-valid → fast-path
identity (no copy, no dict walk cost beyond the cheap scan).

OUR version over the raw PR (iron rule #10)
-------------------------------------------
1. **Gate on spec_config**: the non-spec path pays ZERO cost (the PR runs
   the scan unconditionally). We skip the whole sanitize when
   ``speculative_config is None``.
2. **Flood-guarded WARNING**: rate-limited via a seen-set / once-per-N so a
   sustained bad-draft pathology cannot flood PROD logs (the P71 per-step
   log-flood anti-pattern).
3. **Prometheus counter** ``sndr_invalid_draft_tokens_dropped_total`` so the
   silent case is a metric, not just a log line (PN367 "make the silent
   case visible" doctrine).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.engines.vllm.patches.spec_decode import (
    pn402_sanitize_invalid_draft as pn402,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_LIVE = _FIXTURES / "pn402_live_anchor_region.txt"

_ENV_FLAG = "GENESIS_ENABLE_PN402_SANITIZE_INVALID_DRAFT_TOKENS"


class _SchedOut:
    """Minimal SchedulerOutput stand-in (plain @dataclass-like, mutable)."""

    def __init__(self, spec, num_sched):
        self.scheduled_spec_decode_tokens = spec
        self.num_scheduled_tokens = num_sched
        self.total_num_scheduled_tokens = sum(num_sched.values())


# ─────────────────────────────────────────────────────────────────────
# 1. ANCHOR-PRESENCE on the frozen live-dev424 fixture
# ─────────────────────────────────────────────────────────────────────


def test_anchors_present_and_unique_in_live_region():
    live = _FIXTURE_LIVE.read_text(encoding="utf-8")
    # Call-site anchor (insert the sanitize call).
    assert pn402.PN402_CALLSITE_OLD in live
    assert live.count(pn402.PN402_CALLSITE_OLD) == 1
    # Method-inject anchor (insert the helper before execute_model).
    assert pn402.PN402_METHOD_OLD in live
    assert live.count(pn402.PN402_METHOD_OLD) == 1


def test_pr_literal_helper_absent_in_pristine():
    """The PR's `_sanitize_scheduled_spec_decode_tokens` literal is ABSENT in
    pristine dev424 (pre-merge)."""
    live = _FIXTURE_LIVE.read_text(encoding="utf-8")
    assert "_sanitize_scheduled_spec_decode_tokens" not in live


# ─────────────────────────────────────────────────────────────────────
# 2. CORE SANITIZE LOGIC (the TDD bug-repro — exercise the real helper)
# ─────────────────────────────────────────────────────────────────────


def test_drops_negative_one_draft():
    """REPRO of vllm#46574: a -1 draft is dropped; num decremented (floored
    at 1); total recomputed. Pristine execute_model would index -1 → OOB."""
    so = _SchedOut({"req-0": [-1]}, {"req-0": 2})
    n = pn402.sanitize(so, vocab_size=151936)
    assert n == 1, "one bad draft dropped"
    assert so.scheduled_spec_decode_tokens == {}, "req-0 drafts removed"
    assert so.num_scheduled_tokens == {"req-0": 1}, "num decremented by len([-1])"
    assert so.total_num_scheduled_tokens == 1


def test_drops_over_vocab_draft():
    """An id >= vocab_size is dropped identically."""
    so = _SchedOut({"req-0": [200000]}, {"req-0": 2})
    n = pn402.sanitize(so, vocab_size=151936)
    assert n == 1
    assert so.scheduled_spec_decode_tokens == {}
    assert so.num_scheduled_tokens == {"req-0": 1}
    assert so.total_num_scheduled_tokens == 1


def test_mixed_valid_and_invalid_drops_only_bad_req():
    """A request with all-valid drafts is untouched; only the bad request's
    drafts are dropped."""
    so = _SchedOut(
        {"good": [10, 20, 30], "bad": [5, -1, 7]},
        {"good": 4, "bad": 4},
    )
    n = pn402.sanitize(so, vocab_size=151936)
    assert n == 1
    # good kept, bad dropped.
    assert so.scheduled_spec_decode_tokens == {"good": [10, 20, 30]}
    assert so.num_scheduled_tokens == {"good": 4, "bad": 1}
    assert so.total_num_scheduled_tokens == 5  # 4 + 1


def test_num_scheduled_floored_at_one():
    """If decrementing would drop num below 1, it is floored at 1 (the
    request still owns its real token)."""
    so = _SchedOut({"req-0": [-1, -1, -1]}, {"req-0": 3})
    pn402.sanitize(so, vocab_size=151936)
    # 3 - 3 = 0, floored to 1.
    assert so.num_scheduled_tokens == {"req-0": 1}
    assert so.total_num_scheduled_tokens == 1


def test_all_valid_fast_path_identity():
    """All-valid drafts → the scheduler_output is returned unchanged
    (no mutation, byte-stable dicts)."""
    spec = {"r0": [1, 2, 3], "r1": [4, 5]}
    num = {"r0": 4, "r1": 3}
    so = _SchedOut(dict(spec), dict(num))
    n = pn402.sanitize(so, vocab_size=151936)
    assert n == 0
    assert so.scheduled_spec_decode_tokens == spec
    assert so.num_scheduled_tokens == num
    assert so.total_num_scheduled_tokens == 7


def test_empty_drafts_noop():
    so = _SchedOut({}, {"r0": 1})
    n = pn402.sanitize(so, vocab_size=151936)
    assert n == 0
    assert so.total_num_scheduled_tokens == 1


def test_boundary_vocab_minus_one_is_valid():
    """vocab_size - 1 is the highest VALID id; it must NOT be dropped."""
    so = _SchedOut({"r0": [151935]}, {"r0": 2})
    n = pn402.sanitize(so, vocab_size=151936)
    assert n == 0
    assert so.scheduled_spec_decode_tokens == {"r0": [151935]}


# ─────────────────────────────────────────────────────────────────────
# 3. OUR improvements: no-spec gate, flood guard, counter
# ─────────────────────────────────────────────────────────────────────


def test_flood_guard_caps_warnings_but_counter_counts_all():
    """100 bad-draft steps: the WARNING is rate-capped (<= the cap), but the
    counter reflects every drop."""
    logs = []
    counts = {"n": 0}

    def log_fn(msg):
        logs.append(msg)

    def counter_fn(k):
        counts["n"] += k

    guard = pn402.FloodGuard(cap=5)
    for _ in range(100):
        so = _SchedOut({"r0": [-1]}, {"r0": 2})
        pn402.sanitize(
            so, vocab_size=151936, log_fn=log_fn, counter_fn=counter_fn,
            flood_guard=guard,
        )
    assert counts["n"] == 100, "counter records every dropped-draft event"
    assert len(logs) <= 5, f"WARNING flood-capped at 5, got {len(logs)}"
    assert len(logs) >= 1, "at least one WARNING emitted"


def test_sanitize_returns_drop_count_zero_when_clean():
    so = _SchedOut({"r0": [1, 2]}, {"r0": 3})
    assert pn402.sanitize(so, vocab_size=151936) == 0


# ─────────────────────────────────────────────────────────────────────
# 4. APPLY -> APPLIED on a synthetic file built from the live region
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def env_pn402_on(monkeypatch):
    monkeypatch.setenv(_ENV_FLAG, "1")
    monkeypatch.setenv("SNDR_ENABLE_PN402_SANITIZE_INVALID_DRAFT_TOKENS", "1")
    yield


def _redirect_resolver(monkeypatch, path: Path | None):
    def fake_resolve(rel: str):
        if "model_runner" in rel:
            return str(path) if path is not None else None
        return None
    monkeypatch.setattr(pn402, "resolve_vllm_file", fake_resolve)


@pytest.fixture
def synthetic_target(tmp_path):
    p = tmp_path / "model_runner.py"
    p.write_text(_FIXTURE_LIVE.read_text(encoding="utf-8"), encoding="utf-8")
    return p


def test_apply_applied_markers_present(env_pn402_on, monkeypatch, synthetic_target):
    p = synthetic_target
    _redirect_resolver(monkeypatch, p)
    status, reason = pn402.apply()
    assert status == "applied", reason
    text = p.read_text(encoding="utf-8")
    assert pn402.GENESIS_PN402_MARKER in text
    # The helper method + the call are both injected.
    assert "_sanitize_scheduled_spec_decode_tokens" in text
    # The call is inserted into execute_model after apply_staged_writes.
    assert "self.block_tables.apply_staged_writes()" in text


def test_apply_result_parses(env_pn402_on, monkeypatch, tmp_path):
    """Applying both PN402 anchors to a minimal-but-complete class body must
    produce valid python (the injected method + call are syntactically valid
    in context). Built as a real class so the parse is meaningful (the live
    fixture region is a fragment of a class body)."""
    # A minimal class carrying both anchors verbatim (method-inject point +
    # the call-site inside execute_model).
    klass = (
        "import torch\n"
        "logger = None\n"
        "\n"
        "class _Runner:\n"
        "    def __init__(self):\n"
        "        self.speculative_config = None\n"
        "        self.vocab_size = 151936\n"
        "        self.block_tables = None\n"
        "        self.kv_connector = None\n"
        "\n"
        "    @torch.inference_mode()\n"
        "    def execute_model(\n"
        "        self,\n"
        "        scheduler_output,\n"
        "    ):\n"
        "        if not False:\n"
        "            self.block_tables.apply_staged_writes()\n"
        "            if scheduler_output.total_num_scheduled_tokens == 0:\n"
        "                empty_output = self.kv_connector.no_forward(scheduler_output)\n"
        "                return empty_output\n"
        "        return None\n"
    )
    p = tmp_path / "model_runner.py"
    p.write_text(klass, encoding="utf-8")
    _redirect_resolver(monkeypatch, p)
    status, reason = pn402.apply()
    assert status == "applied", reason
    import ast

    ast.parse(p.read_text(encoding="utf-8"))
    # Both anchors landed.
    text = p.read_text(encoding="utf-8")
    assert "def _sanitize_scheduled_spec_decode_tokens(self, scheduler_output):" in text
    assert "scheduler_output = self._sanitize_scheduled_spec_decode_tokens(" in text


def test_apply_is_idempotent(env_pn402_on, monkeypatch, synthetic_target):
    p = synthetic_target
    _redirect_resolver(monkeypatch, p)
    s1, _ = pn402.apply()
    assert s1 == "applied"
    first = p.read_text(encoding="utf-8")
    s2, r2 = pn402.apply()
    assert s2 == "skipped", r2
    assert "already applied" in r2.lower() or "marker present" in r2.lower()
    assert p.read_text(encoding="utf-8") == first


def test_is_applied_reflects_state(env_pn402_on, monkeypatch, synthetic_target):
    p = synthetic_target
    _redirect_resolver(monkeypatch, p)
    assert pn402.is_applied() is False
    pn402.apply()
    assert pn402.is_applied() is True


def test_disabled_by_default(monkeypatch, synthetic_target):
    monkeypatch.delenv(_ENV_FLAG, raising=False)
    monkeypatch.delenv("SNDR_ENABLE_PN402_SANITIZE_INVALID_DRAFT_TOKENS", raising=False)
    p = synthetic_target
    _redirect_resolver(monkeypatch, p)
    status, reason = pn402.apply()
    assert status == "skipped", reason
    assert pn402.GENESIS_PN402_MARKER not in p.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# 5. REGISTRY contract
# ─────────────────────────────────────────────────────────────────────


def test_registry_pn402_contract():
    from sndr.dispatcher.registry import PATCH_REGISTRY

    assert "PN402" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN402"]
    assert meta["env_flag"] == _ENV_FLAG
    assert meta["family"] == "spec_decode"
    assert meta["category"] == "stability"
    assert meta["lifecycle"] == "experimental"
    assert meta["upstream_pr"] == 46574
    assert meta["applies_to"]["vllm_version_range"] == (">=0.23.0", "<0.24.0")
    composes = set(meta.get("composes_with", []))
    assert {"PN378", "PN361", "PN133"}.issubset(composes)


def test_env_flag_registered():
    from sndr.env import Flags
    assert hasattr(Flags, "PN402_SANITIZE_INVALID_DRAFT_TOKENS")
    assert (
        Flags.PN402_SANITIZE_INVALID_DRAFT_TOKENS
        == "PN402_SANITIZE_INVALID_DRAFT_TOKENS"
    )
