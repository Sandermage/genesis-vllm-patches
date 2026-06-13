# SPDX-License-Identifier: Apache-2.0
"""TDD for Patch 88 — prefix-cache stats retry de-duplication
(rewrite of OPEN vllm#45202, pr-sweep-50 roadmap chunk 1 Theme C,
wave 2).

NAMING NOTE: the roadmap chunk-1 table called this "p86", but P86 is
already taken in PATCH_REGISTRY (ngram batch_propose linear fill,
vllm#40876) — registered as P88 (next free P-number, verified
2026-06-11).

The bug (#43736): ``KVCacheManager.get_computed_blocks`` recorded the
local prefix-cache query/hit stats at LOOKUP time. A waiting request
whose ``allocate_slots`` then fails stays in the waiting queue and
repeats the lookup on a later step — counted once PER ATTEMPT, so the
reported ``prefix_hit_rate`` inflates by tens of percent under
KV-pressure burst retries (Genesis long-ctx agent profile), poisoning
P85/TQ-KV A/B conclusions.

Genesis rewrite (NOT the upstream diff): upstream moves the record
into the 2000-line ``Scheduler.schedule()`` waiting loop. P88 keeps
both sites inside kv_cache_manager.py — the lookup stashes a
single-slot pending record, ``allocate_slots`` commits it once the
allocation is past its last failure return and the request ids match.
Safer (P79d-style minimal-anchor convention) and MORE faithful than
upstream: stats record iff a real lookup happened (upstream's
scheduler-side record also fires for enable_caching=False configs,
where pristine never recorded).

These tests verify anchors against the PRISTINE pin tree, replacement
hygiene, and the de-dup semantics END-TO-END on synthetic-but-
compilable fakes carrying the byte-exact anchors.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.kernel import TextPatchResult

from sndr.engines.vllm.patches.observability.p88_prefix_cache_stats_dedup import (
    GENESIS_P88_MARKER,
    P88_LOOKUP_ANCHOR,
    P88_LOOKUP_REPLACEMENT,
    P88_ALLOC_COMMIT_ANCHOR,
    P88_ALLOC_COMMIT_REPLACEMENT,
    _connector_configured,
    _make_kv_cache_manager_patcher,
)

PRISTINE_ROOT = Path("/private/tmp/candidate_pin_current/vllm")


# ─── synthetic-but-compilable fake carrying the byte-exact anchors ───────
#
# Mirrors the pristine KVCacheManager closely enough that (a) both
# anchors match byte-for-byte at the real indentation depth and (b) the
# patched source still compiles and runs, so the de-dup semantics can
# be exercised end-to-end without torch or a vLLM install.

FAKE_KV_CACHE_MANAGER_PY = '''\
class PrefixCacheStats:
    def __init__(self):
        self.requests = 0
        self.queries = 0
        self.hits = 0
        self.preempted_requests = 0
        self.preempted_queries = 0
        self.preempted_hits = 0

    def record(self, num_tokens, num_hits, preempted):
        if preempted:
            self.preempted_requests += 1
            self.preempted_queries += num_tokens
            self.preempted_hits += num_hits
        else:
            self.requests += 1
            self.queries += num_tokens
            self.hits += num_hits


class FakeCoordinator:
    def __init__(self, hit_tokens=0):
        self.hit_tokens = hit_tokens

    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        return ([], min(self.hit_tokens, max_cache_hit_length))

    def get_num_blocks_to_allocate(self, request_id):
        return 1

    def allocate_new_blocks(self, request_id):
        return ["block"]


class FakeBlockPool:
    def __init__(self, free_blocks=10):
        self.free_blocks = free_blocks

    def get_num_free_blocks(self):
        return self.free_blocks


class KVCacheManager:
    def __init__(self, log_stats=True, enable_caching=True, hit_tokens=0,
                 free_blocks=10):
        self.log_stats = log_stats
        self.enable_caching = enable_caching
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None
        self.coordinator = FakeCoordinator(hit_tokens=hit_tokens)
        self.block_pool = FakeBlockPool(free_blocks=free_blocks)
        self.empty_kv_cache_blocks = ()

    def create_kv_cache_blocks(self, blocks):
        return tuple(blocks)

    def get_computed_blocks(self, request):
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(self, request, num_new_tokens,
                       num_new_computed_tokens=0, reserved_blocks=0):
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request.request_id
        )

        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        if num_blocks_to_allocate > available_blocks:
            # Cannot allocate new blocks
            return None

        new_blocks = self.coordinator.allocate_new_blocks(request.request_id)
        return self.create_kv_cache_blocks(new_blocks)
'''


class FakeRequest:
    _counter = 0

    def __init__(self, num_tokens=64, num_preemptions=0,
                 skip_reading_prefix_cache=False, num_computed_tokens=0):
        FakeRequest._counter += 1
        self.request_id = f"req-{FakeRequest._counter}"
        self.num_tokens = num_tokens
        self.num_preemptions = num_preemptions
        self.skip_reading_prefix_cache = skip_reading_prefix_cache
        self.num_computed_tokens = num_computed_tokens
        self.block_hashes = []


@pytest.fixture
def fake_kv_cache_manager_py(tmp_path):
    p = tmp_path / "kv_cache_manager.py"
    p.write_text(FAKE_KV_CACHE_MANAGER_PY)
    return str(p)


def _apply(fake_kv_cache_manager_py):
    patcher = _make_kv_cache_manager_patcher(
        target_file=fake_kv_cache_manager_py
    )
    result, failure = patcher.apply()
    assert result == TextPatchResult.APPLIED, failure
    return Path(fake_kv_cache_manager_py).read_text()


def _exec_module(src):
    ns = {}
    exec(compile(src, "<patched>", "exec", dont_inherit=True), ns)
    return ns


def _patched_manager(fake_kv_cache_manager_py, **kwargs):
    ns = _exec_module(_apply(fake_kv_cache_manager_py))
    return ns["KVCacheManager"](**kwargs)


# ═══ 1. Anchor contract against the PRISTINE pin tree ════════════════════


@pytest.mark.skipif(
    not (PRISTINE_ROOT / "v1/core/kv_cache_manager.py").is_file(),
    reason="pristine candidate pin tree not present on this machine",
)
def test_pristine_anchors_present_exactly_once():
    kvm = (PRISTINE_ROOT / "v1/core/kv_cache_manager.py").read_text()
    assert kvm.count(P88_LOOKUP_ANCHOR) == 1
    assert kvm.count(P88_ALLOC_COMMIT_ANCHOR) == 1


@pytest.mark.skipif(
    not (PRISTINE_ROOT / "v1/core/kv_cache_manager.py").is_file(),
    reason="pristine candidate pin tree not present on this machine",
)
def test_pristine_commit_point_is_past_last_failure_return():
    """The commit anchor (available-blocks gate) must be the LAST
    `return None` in allocate_slots — recording after it is recording
    on success. If upstream adds a later failure return, the commit
    point must move."""
    kvm = (PRISTINE_ROOT / "v1/core/kv_cache_manager.py").read_text()
    body = kvm.split("def allocate_slots(")[1].split("\n    def ")[0]
    anchor_pos = body.find(
        "if num_blocks_to_allocate > available_blocks:"
    )
    assert anchor_pos != -1
    after_anchor = body[anchor_pos:]
    # Skip the anchor's own `return None`.
    remainder = after_anchor.split("return None", 1)[1]
    assert "return None" not in remainder, (
        "a failure return exists AFTER the P88 commit point — "
        "re-derive the anchor before applying"
    )


# ═══ 2. Replacement hygiene ═══════════════════════════════════════════════


def test_lookup_replacement_does_not_record():
    """The lookup must become a pure stash — recording there is the bug."""
    assert "prefix_cache_stats.record(" not in P88_LOOKUP_REPLACEMENT
    assert "_genesis_p88_pending_stats" in P88_LOOKUP_REPLACEMENT


def test_commit_replacement_records_once_and_clears():
    assert "prefix_cache_stats.record(" in P88_ALLOC_COMMIT_REPLACEMENT
    assert (
        P88_ALLOC_COMMIT_REPLACEMENT.count("self._genesis_p88_pending_stats = None")
        == 1
    )


def test_drift_markers_have_no_self_collision():
    """PN369 false-skip class (mirrors tools/lint_drift_markers.py)."""
    patcher = _make_kv_cache_manager_patcher(target_file="/nonexistent")
    marker_line = f"# [Genesis wiring marker: {patcher.marker}]\n"
    for dm in patcher.upstream_drift_markers:
        if dm.startswith("[Genesis"):
            continue
        assert dm not in marker_line
        for sp in patcher.sub_patches:
            assert dm not in sp.replacement


# ═══ 3. De-dup semantics — end-to-end on the patched executable fake ═════


def test_lookup_then_success_records_exactly_once(fake_kv_cache_manager_py):
    mgr = _patched_manager(fake_kv_cache_manager_py, hit_tokens=16)
    req = FakeRequest(num_tokens=64)
    _, hits = mgr.get_computed_blocks(req)
    assert hits == 16
    # Lookup alone records NOTHING.
    assert mgr.prefix_cache_stats.requests == 0
    assert mgr.allocate_slots(req, 8, num_new_computed_tokens=hits) is not None
    assert mgr.prefix_cache_stats.requests == 1
    assert mgr.prefix_cache_stats.queries == 64
    assert mgr.prefix_cache_stats.hits == 16


def test_failed_allocation_not_counted_then_retry_counts_once(
    fake_kv_cache_manager_py,
):
    """THE regression case (#43736): N failed scheduling attempts must
    contribute zero records; the eventual success records exactly one."""
    mgr = _patched_manager(
        fake_kv_cache_manager_py, hit_tokens=16, free_blocks=0
    )
    req = FakeRequest(num_tokens=64)
    for _ in range(3):  # three failed scheduling attempts
        mgr.get_computed_blocks(req)
        assert mgr.allocate_slots(req, 8) is None
    assert mgr.prefix_cache_stats.requests == 0
    assert mgr.prefix_cache_stats.queries == 0

    mgr.block_pool.free_blocks = 10  # blocks freed -> retry succeeds
    mgr.get_computed_blocks(req)
    assert mgr.allocate_slots(req, 8) is not None
    assert mgr.prefix_cache_stats.requests == 1
    assert mgr.prefix_cache_stats.queries == 64
    assert mgr.prefix_cache_stats.hits == 16


def test_pristine_behavior_double_counts(fake_kv_cache_manager_py):
    """Contrast test documenting the bug on the UNPATCHED fake: the
    retry loop records once per attempt."""
    ns = _exec_module(Path(fake_kv_cache_manager_py).read_text())
    mgr = ns["KVCacheManager"](hit_tokens=16, free_blocks=0)
    req = FakeRequest(num_tokens=64)
    for _ in range(3):
        mgr.get_computed_blocks(req)
        assert mgr.allocate_slots(req, 8) is None
    assert mgr.prefix_cache_stats.requests == 3  # the inflation P88 kills


def test_preempted_request_routes_to_preempted_counters(
    fake_kv_cache_manager_py,
):
    mgr = _patched_manager(fake_kv_cache_manager_py, hit_tokens=8)
    req = FakeRequest(num_tokens=32, num_preemptions=1)
    mgr.get_computed_blocks(req)
    assert mgr.allocate_slots(req, 8) is not None
    assert mgr.prefix_cache_stats.requests == 0
    assert mgr.prefix_cache_stats.preempted_requests == 1
    assert mgr.prefix_cache_stats.preempted_queries == 32
    assert mgr.prefix_cache_stats.preempted_hits == 8


def test_allocate_without_lookup_records_nothing(fake_kv_cache_manager_py):
    """Running-loop allocations (no preceding lookup) must not record —
    pending is empty."""
    mgr = _patched_manager(fake_kv_cache_manager_py)
    req = FakeRequest(num_tokens=64, num_computed_tokens=32)
    assert mgr.allocate_slots(req, 8) is not None
    assert mgr.prefix_cache_stats.requests == 0


def test_pending_not_consumed_by_other_request(fake_kv_cache_manager_py):
    """A stale pending stash for R1 must not be committed by R2's
    allocation (request-id match required)."""
    mgr = _patched_manager(fake_kv_cache_manager_py, hit_tokens=4)
    r1 = FakeRequest(num_tokens=64)
    r2 = FakeRequest(num_tokens=32, num_computed_tokens=16)
    mgr.get_computed_blocks(r1)  # stash for r1
    assert mgr.allocate_slots(r2, 8) is not None  # r2 allocates
    assert mgr.prefix_cache_stats.requests == 0
    # r1's own success still commits its (re-stashed or original) record.
    assert mgr.allocate_slots(r1, 8) is not None
    assert mgr.prefix_cache_stats.requests == 1
    assert mgr.prefix_cache_stats.queries == 64


def test_caching_disabled_records_nothing(fake_kv_cache_manager_py):
    """Pristine parity (and a deliberate improvement over upstream's
    scheduler-side record): no lookup -> no record."""
    mgr = _patched_manager(fake_kv_cache_manager_py, enable_caching=False)
    req = FakeRequest(num_tokens=64)
    blocks, hits = mgr.get_computed_blocks(req)
    assert (blocks, hits) == ((), 0)
    assert mgr.allocate_slots(req, 8) is not None
    assert mgr.prefix_cache_stats.requests == 0


def test_log_stats_off_is_a_noop(fake_kv_cache_manager_py):
    mgr = _patched_manager(fake_kv_cache_manager_py, log_stats=False)
    req = FakeRequest(num_tokens=64)
    mgr.get_computed_blocks(req)
    assert mgr.allocate_slots(req, 8) is not None
    assert mgr.prefix_cache_stats is None


def test_double_success_same_request_records_once(fake_kv_cache_manager_py):
    """The commit clears the pending slot — a second allocate_slots for
    the same request (e.g. running-loop growth) must not re-record."""
    mgr = _patched_manager(fake_kv_cache_manager_py, hit_tokens=4)
    req = FakeRequest(num_tokens=64)
    mgr.get_computed_blocks(req)
    assert mgr.allocate_slots(req, 8) is not None
    assert mgr.allocate_slots(req, 8) is not None
    assert mgr.prefix_cache_stats.requests == 1


# ═══ 4. Connector fallback-disable probe ══════════════════════════════════


def test_connector_probe_clean_env(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["vllm", "serve", "/models/qwen", "--port", "8101"]
    )
    for var in ("VLLM_KV_TRANSFER_CONFIG", "LMCACHE_CONFIG_FILE"):
        monkeypatch.delenv(var, raising=False)
    assert _connector_configured() is None


def test_connector_probe_detects_cli_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["vllm", "serve", "/m", "--kv-transfer-config",
         '{"kv_connector": "LMCacheConnectorV1"}'],
    )
    assert _connector_configured() is not None


def test_connector_probe_detects_env(monkeypatch):
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "/m"])
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "/etc/lmcache.yaml")
    assert _connector_configured() is not None


# ═══ 5. Idempotency ═══════════════════════════════════════════════════════


def test_idempotent_second_apply(fake_kv_cache_manager_py):
    src = _apply(fake_kv_cache_manager_py)
    assert GENESIS_P88_MARKER in src
    patcher = _make_kv_cache_manager_patcher(
        target_file=fake_kv_cache_manager_py
    )
    result, failure = patcher.apply()
    assert result == TextPatchResult.IDEMPOTENT, failure
    assert Path(fake_kv_cache_manager_py).read_text() == src
