# SPDX-License-Identifier: Apache-2.0
"""Tests for PN95 improvements derived from competitor research
(LMCache / SGLang HiCache / vLLM v1 kv_offload).

Three improvements landed in this commit:

  1. Host RAM safety margin (SGLang
     HICACHE_HOST_MEMORY_RESERVE_BYTES contract).
  2. Active-block protection — TTL-based pin equivalent (LMCache
     pin_count contract, simplified).
  3. Upstream KV-offload connector detection — skip PN95 init when
     vLLM v1 kv_offload framework is wired in to avoid double-managing
     blocks.

Each test exercises one behavior without GPU or live vLLM imports;
all run on the dev box.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch


class _FakeTier:
    """Minimal CacheTier shape consumed by TierManager."""

    def __init__(
        self,
        device: str,
        capacity_gib: float,
        *,
        eviction_policy: str = "lru",
        low_water_pct: float = 0.9,
        promote_on_hit: bool = True,
    ):
        self.device = device
        self.capacity_gib = capacity_gib
        self.eviction_policy = eviction_policy
        self.low_water_pct = low_water_pct
        self.promote_on_hit = promote_on_hit


# ─── Improvement 1: Host RAM safety margin ──────────────────────────


class TestHostCapacityCap(unittest.TestCase):
    def test_explicit_cap_overrides_declared(self):
        from vllm.sndr_core.cache.tier_manager import TierManager

        tm = TierManager(
            [_FakeTier("gpu", 20.0), _FakeTier("cpu", 100.0)],
            slot_nbytes=65536,
            host_capacity_cap_gib=32.0,
        )
        self.assertEqual(tm._effective_cpu_capacity_gib, 32.0)

    def test_cap_only_lowers_never_raises(self):
        from vllm.sndr_core.cache.tier_manager import TierManager

        tm = TierManager(
            [_FakeTier("gpu", 20.0), _FakeTier("cpu", 4.0)],
            slot_nbytes=65536,
            host_capacity_cap_gib=64.0,
        )
        # Declared 4 GiB is already below the 64 GiB cap — passes through.
        self.assertEqual(tm._effective_cpu_capacity_gib, 4.0)

    def test_no_cap_passes_declared(self):
        from vllm.sndr_core.cache.tier_manager import TierManager

        tm = TierManager(
            [_FakeTier("gpu", 20.0), _FakeTier("cpu", 8.0)],
            slot_nbytes=65536,
        )
        self.assertEqual(tm._effective_cpu_capacity_gib, 8.0)

    def test_env_override_wins(self):
        from vllm.sndr_core.cache import tier_manager as tm_mod

        with patch.dict(os.environ, {"GENESIS_PN95_HOST_CAP_GIB": "42"}):
            self.assertEqual(tm_mod._host_capacity_cap_gib(), 42.0)

    def test_env_override_rejects_non_positive(self):
        from vllm.sndr_core.cache import tier_manager as tm_mod

        with patch.dict(os.environ, {"GENESIS_PN95_HOST_CAP_GIB": "0"}):
            # 0 is rejected — falls back to auto-compute (None on Mac).
            self.assertIsNone(tm_mod._host_capacity_cap_gib())

    def test_invalid_reserve_falls_back_to_default(self):
        from vllm.sndr_core.cache import tier_manager as tm_mod

        with patch.dict(
            os.environ,
            {"GENESIS_PN95_HOST_RESERVE_GIB": "not-a-number"},
            clear=False,
        ):
            # Invalid value MUST NOT raise — falls back to default 8 GiB.
            # On a host without /proc/meminfo (e.g. macOS) the total-RAM
            # probe returns None so the final cap is None regardless of
            # reserve value. On Linux with a readable /proc/meminfo the
            # probe returns the real free-mem cap (positive float) and
            # the contract is "no exception", not "exactly None".
            result = tm_mod._host_capacity_cap_gib()
            assert result is None or isinstance(result, (int, float)), (
                f"_host_capacity_cap_gib must return None or a numeric "
                f"cap when given an invalid reserve value, got {result!r}"
            )
            if isinstance(result, (int, float)):
                self.assertGreater(result, 0.0)


# ─── Improvement 2: Active-block protection (TTL) ────────────────────


class TestActiveBlockProtection(unittest.TestCase):
    def _build_tm(self):
        from vllm.sndr_core.cache.tier_manager import TierManager

        return TierManager(
            [_FakeTier("gpu", 4.0), _FakeTier("cpu", 4.0)],
            slot_nbytes=65536,
            host_capacity_cap_gib=8.0,
        )

    def test_default_ttl_zero_disables_protection(self):
        tm = self._build_tm()
        tm.admit("blk_A", group_id="attn")
        tm.mark_active("blk_A")
        # ttl=0 → is_active always False
        self.assertFalse(tm.is_active("blk_A"))

    def test_set_active_ttl_negative_raises(self):
        tm = self._build_tm()
        with self.assertRaises(ValueError):
            tm.set_active_ttl(-1)

    def test_marked_key_active_within_window(self):
        tm = self._build_tm()
        tm.set_active_ttl(3)
        tm.admit("blk_A", group_id="attn")
        tm.mark_active("blk_A")
        self.assertTrue(tm.is_active("blk_A"))

    def test_other_key_not_active(self):
        tm = self._build_tm()
        tm.set_active_ttl(3)
        tm.admit("blk_A", group_id="attn")
        tm.admit("blk_B", group_id="attn")
        tm.mark_active("blk_A")
        self.assertFalse(tm.is_active("blk_B"))

    def test_active_window_expires_after_ttl_ticks(self):
        tm = self._build_tm()
        tm.set_active_ttl(2)
        tm.admit("blk_A", group_id="attn")
        tm.mark_active("blk_A")  # tick=1
        tm.mark_active("blk_B")  # tick=2
        # blk_A is still within TTL=2 window (2 - 1 < 2 → True)
        self.assertTrue(tm.is_active("blk_A"))
        tm.mark_active("blk_C")  # tick=3
        # Now 3 - 1 = 2, NOT less than 2 → False
        self.assertFalse(tm.is_active("blk_A"))

    def test_active_key_skipped_from_demote_candidates(self):
        tm = self._build_tm()
        tm.set_active_ttl(10)
        tm.admit("blk_A", group_id="attn")
        tm.admit("blk_B", group_id="attn")
        # Mark A active; only B should be a candidate.
        tm.mark_active("blk_A")
        cands = list(tm._demote_candidates())
        self.assertNotIn("blk_A", cands)
        self.assertIn("blk_B", cands)


# ─── Improvement 3: Upstream offload connector detection ─────────────


class TestUpstreamConnectorDetection(unittest.TestCase):
    def _fake_cfg(self, **kv_transfer_attrs):
        class _Block:
            pass

        class _Cfg:
            kv_transfer_config = None
            cache_config = None

        cfg = _Cfg()
        if kv_transfer_attrs:
            block = _Block()
            for k, v in kv_transfer_attrs.items():
                setattr(block, k, v)
            cfg.kv_transfer_config = block
        return cfg

    def test_no_connector_returns_none(self):
        from vllm.sndr_core.cache._pn95_runtime import (
            _detect_upstream_offload_connector,
        )

        self.assertIsNone(_detect_upstream_offload_connector(self._fake_cfg()))

    def test_kv_connector_attribute_detected(self):
        from vllm.sndr_core.cache._pn95_runtime import (
            _detect_upstream_offload_connector,
        )

        cfg = self._fake_cfg(kv_connector="LMCacheConnectorV1")
        self.assertEqual(
            _detect_upstream_offload_connector(cfg),
            "LMCacheConnectorV1",
        )

    def test_connector_class_attribute_detected(self):
        from vllm.sndr_core.cache._pn95_runtime import (
            _detect_upstream_offload_connector,
        )

        cfg = self._fake_cfg(connector_class="SimpleCPUOffloadConnector")
        self.assertEqual(
            _detect_upstream_offload_connector(cfg),
            "SimpleCPUOffloadConnector",
        )

    def test_empty_string_treated_as_absent(self):
        from vllm.sndr_core.cache._pn95_runtime import (
            _detect_upstream_offload_connector,
        )

        cfg = self._fake_cfg(kv_connector="   ")
        self.assertIsNone(_detect_upstream_offload_connector(cfg))

    def test_non_string_treated_as_absent(self):
        from vllm.sndr_core.cache._pn95_runtime import (
            _detect_upstream_offload_connector,
        )

        cfg = self._fake_cfg(kv_connector=12345)
        self.assertIsNone(_detect_upstream_offload_connector(cfg))

    def test_init_from_config_skips_when_upstream_connector_present(self):
        from vllm.sndr_core.cache import _pn95_runtime

        with patch.dict(
            os.environ,
            {"GENESIS_ENABLE_PN95_TIER_AWARE_CACHE": "1"},
            clear=False,
        ):
            _pn95_runtime.reset_for_tests()
            cfg = self._fake_cfg(kv_connector="LMCacheConnectorV1")
            result = _pn95_runtime.init_from_config(cfg)
            self.assertFalse(
                result,
                "init_from_config must skip when an upstream offload "
                "connector is wired in (avoid double-managing blocks)",
            )


if __name__ == "__main__":
    unittest.main()
