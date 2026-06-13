# SPDX-License-Identifier: Apache-2.0
"""TDD for PN382 — vendor of OPEN PR vllm#45080 (DecodeBenchConnector
list/tuple KV fill) + two Genesis extensions.

Upstream #45080 fixes the AttributeError crash on hybrid / linear-
attention models: ``DecodeBenchConnectorWorker._fill_blocks`` assumed
every layer's KV cache is a single block-indexed ``torch.Tensor`` and
died on ``list.device`` for Mamba/GDN layers whose cache is a LIST of
state tensors. Upstream then fills each state tensor IN ITS ENTIRETY.

Genesis extensions (roadmap chunk-3 Theme D — verified against the
pristine pin tree 0.22.1rc1.dev259+g303916e93):

  1. PER-BLOCK fill for the list/tuple path. On this pin, MambaSpec
     state tensors ARE block-indexed — ``gpu_model_runner.py`` builds
     them with ``target_shape = (num_blocks, *shape)`` (verified in
     the pristine tree, MambaSpec branch of the KV-cache initializer).
     Upstream's whole-pool ``fill_()/normal_()`` would clobber the
     recurrent state of every CONCURRENT request; the Genesis fill
     touches only the requested block rows (same per-row fill as the
     attention path).
  2. REAL group_idx -> layer_names map. Upstream's
     ``register_kv_caches`` maps ALL layers to group 0; on hybrid
     models (full-attn group + Mamba group, e.g. Qwen3.6) that fills
     Mamba state with the ATTENTION group's block ids and ignores the
     Mamba group's own ids. PN382 threads ``kv_cache_config`` (already
     handed to the connector ctor on this pin) into the worker and
     builds the map from ``kv_cache_config.kv_cache_groups``.

These tests verify textually (portable embedded fixtures shaped like
pin 0.22.1rc1.dev259+g303916e93), BEHAVIORALLY (the patched fixture is
exec'd with a fake torch to prove the no-clobber contract), and
opportunistically against the real pristine tree at
/private/tmp/candidate_pin_current.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm")
PIN_CONNECTOR = (
    PIN_TREE
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "decode_bench_connector.py"
)


def _pn382():
    from sndr.engines.vllm.patches.kv_cache import (
        pn382_decode_bench_hybrid_fill as M,
    )
    return M


# ─────────────────────────────────────────────────────────────────────
# Portable fixture — pristine-shaped regions (pin g303916e93)
# ─────────────────────────────────────────────────────────────────────

# Verbatim pristine regions (byte-parity with the pin is asserted in
# TestAgainstPristinePin below).

CTOR_REGION = (
    "        if role == KVConnectorRole.SCHEDULER:\n"
    "            self.connector_scheduler = DecodeBenchConnectorScheduler(vllm_config)\n"
    "        elif role == KVConnectorRole.WORKER:\n"
    "            self.connector_worker = DecodeBenchConnectorWorker(vllm_config)\n"
)

SCHEDULER_INIT_REGION = (
    '    def __init__(self, vllm_config: "VllmConfig"):\n'
    "        self.vllm_config = vllm_config\n"
    "        self.block_size = vllm_config.cache_config.block_size\n"
    "\n"
    "        # Track which requests have already been filled\n"
    "        self._filled_requests: set[str] = set()\n"
)

WORKER_INIT_REGION = (
    '    def __init__(self, vllm_config: "VllmConfig"):\n'
    "        self.vllm_config = vllm_config\n"
    "        self.block_size = vllm_config.cache_config.block_size\n"
    "\n"
    "        # Get fill parameters from extra config\n"
    "        kv_transfer_config = vllm_config.kv_transfer_config\n"
    "        assert kv_transfer_config is not None\n"
    '        self.fill_mean = kv_transfer_config.get_from_extra_config("fill_mean", 0.015)\n'
    '        self.fill_std = kv_transfer_config.get_from_extra_config("fill_std", 0.0)\n'
    "\n"
    "        # Will be populated via register_kv_caches\n"
    "        self.kv_caches: dict[str, torch.Tensor] | None = None\n"
    "\n"
    "        # Mapping from KV cache group index to list of layer names in that group\n"
    "        self.group_to_layers: dict[int, list[str]] | None = None\n"
)

GROUP_MAP_REGION = (
    '    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):\n'
    '        """Store references to the KV cache tensors and build group mapping."""\n'
    "        self.kv_caches = kv_caches\n"
    "\n"
    "        # For simplicity, assume all layers belong to group 0 (standard attention)\n"
    "        # For MLA models with multiple groups, the metadata will handle the mapping\n"
    "        # We just need to fill the blocks specified in the metadata\n"
    "        self.group_to_layers = {0: list(kv_caches.keys())}\n"
    "\n"
    "        logger.debug(\n"
    '            "DecodeBenchConnector: Registered %d KV cache layers",\n'
    "            len(kv_caches),\n"
    "        )\n"
)

FILL_REGION = (
    "            # Convert block_ids to tensor on device\n"
    "            block_ids_tensor = torch.tensor(\n"
    "                block_ids, dtype=torch.long, device=kv_cache.device\n"
    "            )\n"
    "\n"
    "            # Filter invalid block IDs\n"
    "            valid_mask = block_ids_tensor < kv_cache.shape[0]\n"
    "            valid_block_ids = block_ids_tensor[valid_mask]\n"
    "\n"
    "            if len(valid_block_ids) == 0:\n"
    "                continue\n"
    "\n"
    "            # Create fill values - either constant or random\n"
    "            block_shape = kv_cache.shape[1:]\n"
    "            if self.fill_std > 0:\n"
    "                # Random normal sampling\n"
    "                fill_values = torch.normal(\n"
    "                    mean=self.fill_mean,\n"
    "                    std=self.fill_std,\n"
    "                    size=(len(valid_block_ids),) + block_shape,\n"
    "                    dtype=kv_cache.dtype,\n"
    "                    device=kv_cache.device,\n"
    "                )\n"
    "            else:\n"
    "                # Constant fill value\n"
    "                fill_values = torch.full(\n"
    "                    (len(valid_block_ids),) + block_shape,\n"
    "                    self.fill_mean,\n"
    "                    dtype=kv_cache.dtype,\n"
    "                    device=kv_cache.device,\n"
    "                )\n"
    "\n"
    "            # Batch fill operation\n"
    "            kv_cache[valid_block_ids] = fill_values\n"
)


def _fake_pristine_connector() -> str:
    """Minimal exec-able decode_bench_connector.py carrying all four
    PN382 anchor regions verbatim. ``torch`` and the logger are
    resolved from the exec globals so the behavioral tests can inject
    fakes (no real torch on this machine)."""
    return (
        "# fake decode_bench_connector.py - pristine-shaped regions"
        " (pin g303916e93)\n"
        "\n"
        "\n"
        "class KVConnectorRole:\n"
        '    SCHEDULER = "scheduler"\n'
        '    WORKER = "worker"\n'
        "\n"
        "\n"
        "class DecodeBenchConnector:\n"
        "    def __init__(self, vllm_config, role, kv_cache_config):\n"
        "        self.connector_scheduler = None\n"
        "        self.connector_worker = None\n"
        "\n" + CTOR_REGION + "\n"
        "\n"
        "class DecodeBenchConnectorScheduler:\n"
        + SCHEDULER_INIT_REGION
        + "\n"
        "\n"
        "class DecodeBenchConnectorWorker:\n"
        + WORKER_INIT_REGION
        + "\n"
        + GROUP_MAP_REGION
        + "\n"
        "    def _fill_blocks(self, group_idx, block_ids, num_tokens):\n"
        "        if not block_ids:\n"
        "            return\n"
        "\n"
        "        assert self.kv_caches is not None\n"
        "        assert self.group_to_layers is not None\n"
        "\n"
        "        layer_names = self.group_to_layers.get(group_idx, [])\n"
        "\n"
        "        for layer_name in layer_names:\n"
        "            if layer_name not in self.kv_caches:\n"
        "                continue\n"
        "\n"
        "            kv_cache = self.kv_caches[layer_name]\n"
        "\n" + FILL_REGION
    )


# ─────────────────────────────────────────────────────────────────────
# Fake torch — just enough surface for the patched fill code
# ─────────────────────────────────────────────────────────────────────


class FakeIndex:
    def __init__(self, ids):
        self.ids = list(ids)

    def __lt__(self, bound):
        return [i < bound for i in self.ids]

    def __getitem__(self, mask):
        return FakeIndex([i for i, m in zip(self.ids, mask) if m])

    def __len__(self):
        return len(self.ids)


class FakeFill:
    def __init__(self, size, value):
        self.size = size
        self.value = value


class FakeTensor:
    """Block-indexed 2-D buffer over plain lists."""

    def __init__(self, num_blocks, numel, fill=0.0):
        self.data = [[fill] * numel for _ in range(num_blocks)]
        self.shape = (num_blocks, numel)
        self.dtype = "fake_dtype"
        self.device = "fake_device"

    def __setitem__(self, key, value):
        assert isinstance(key, FakeIndex)
        assert isinstance(value, FakeFill)
        for block_id in key.ids:
            self.data[block_id] = [value.value] * self.shape[1]


def _fake_torch():
    t = types.ModuleType("fake_torch")
    t.Tensor = FakeTensor
    t.long = "long"
    t.tensor = lambda ids, dtype=None, device=None: FakeIndex(ids)
    t.full = lambda size, value, dtype=None, device=None: FakeFill(size, value)
    t.normal = (
        lambda mean, std, size=None, dtype=None, device=None: FakeFill(size, mean)
    )
    return t


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def debug(self, *a, **k):
        pass

    def warning(self, msg, *a, **k):
        self.warnings.append(msg % a if a else msg)

    def warning_once(self, msg, *a, **k):
        self.warnings.append(msg % a if a else msg)


def _exec_module(text):
    fake_logger = _FakeLogger()
    namespace = {"torch": _fake_torch(), "logger": fake_logger}
    exec(compile(text, "<pn382-fixture>", "exec"), namespace)
    return namespace, fake_logger


def _fake_vllm_config():
    kv_transfer_config = types.SimpleNamespace(
        get_from_extra_config=lambda key, default: default
    )
    return types.SimpleNamespace(
        cache_config=types.SimpleNamespace(block_size=16),
        kv_transfer_config=kv_transfer_config,
    )


def _fake_kv_cache_config():
    return types.SimpleNamespace(
        kv_cache_groups=[
            types.SimpleNamespace(layer_names=["attn.0", "attn.1"]),
            types.SimpleNamespace(layer_names=["mamba.0"]),
        ]
    )


def _patched_namespace(tmp_path, monkeypatch):
    monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
    from sndr.kernel import TextPatchResult

    M = _pn382()
    target = tmp_path / "decode_bench_connector.py"
    target.write_text(_fake_pristine_connector(), encoding="utf-8")
    monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: str(target))
    patcher = M._make_patcher()
    assert patcher is not None
    result, failure = patcher.apply()
    assert result == TextPatchResult.APPLIED, failure
    out = target.read_text(encoding="utf-8")
    ast.parse(out)
    return _exec_module(out)


# ─────────────────────────────────────────────────────────────────────
# 1. Anchors — unique in the fixture, sub-patch inventory
# ─────────────────────────────────────────────────────────────────────


class TestAnchors:
    def test_four_sub_patches_all_required(self):
        M = _pn382()
        subs = M.build_sub_patches()
        assert len(subs) == 4
        assert all(sp.required for sp in subs)
        names = {sp.name for sp in subs}
        assert names == {
            "pn382_worker_ctor_kv_cache_config",
            "pn382_worker_init_kv_cache_config",
            "pn382_real_group_map",
            "pn382_hybrid_per_block_fill",
        }

    def test_anchors_unique_in_fixture(self):
        M = _pn382()
        src = _fake_pristine_connector()
        for sp in M.build_sub_patches():
            assert src.count(sp.anchor) == 1, sp.name

    def test_replacements_do_not_resurrect_anchors(self):
        M = _pn382()
        subs = M.build_sub_patches()
        for sp in subs:
            for other in subs:
                assert other.anchor not in sp.replacement, (
                    sp.name,
                    other.name,
                )

    def test_worker_init_anchor_does_not_match_scheduler_init(self):
        """Scheduler and Worker share the first four __init__ lines —
        the anchor must include the disambiguating comment line."""
        M = _pn382()
        subs = {sp.name: sp for sp in M.build_sub_patches()}
        anchor = subs["pn382_worker_init_kv_cache_config"].anchor
        assert "# Get fill parameters from extra config" in anchor
        assert anchor not in SCHEDULER_INIT_REGION


# ─────────────────────────────────────────────────────────────────────
# 2. End-to-end apply on the fixture
# ─────────────────────────────────────────────────────────────────────


class TestEndToEndApply:
    def test_applies_all_four_subs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        M = _pn382()
        target = tmp_path / "decode_bench_connector.py"
        target.write_text(_fake_pristine_connector(), encoding="utf-8")
        monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: str(target))
        patcher = M._make_patcher()
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, failure
        assert sorted(patcher.applied_sub_patches) == [
            "pn382_hybrid_per_block_fill",
            "pn382_real_group_map",
            "pn382_worker_ctor_kv_cache_config",
            "pn382_worker_init_kv_cache_config",
        ]
        out = target.read_text(encoding="utf-8")
        ast.parse(out)
        assert "_pn382_targets" in out

    def test_idempotent_on_second_apply(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        M = _pn382()
        target = tmp_path / "decode_bench_connector.py"
        target.write_text(_fake_pristine_connector(), encoding="utf-8")
        monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: str(target))
        result, _ = M._make_patcher().apply()
        assert result == TextPatchResult.APPLIED
        result2, _ = M._make_patcher().apply()
        assert result2 == TextPatchResult.IDEMPOTENT


# ─────────────────────────────────────────────────────────────────────
# 3. Behavioral contract — exec'd patched fixture with fake torch
# ─────────────────────────────────────────────────────────────────────


class TestBehavior:
    def _worker(self, tmp_path, monkeypatch, kv_cache_config=None):
        namespace, fake_logger = _patched_namespace(tmp_path, monkeypatch)
        worker_cls = namespace["DecodeBenchConnectorWorker"]
        worker = (
            worker_cls(_fake_vllm_config(), kv_cache_config)
            if kv_cache_config is not None
            else worker_cls(_fake_vllm_config())
        )
        return worker, fake_logger

    def test_real_group_map_from_kv_cache_config(self, tmp_path, monkeypatch):
        worker, _ = self._worker(
            tmp_path, monkeypatch, kv_cache_config=_fake_kv_cache_config()
        )
        caches = {
            "attn.0": FakeTensor(8, 4),
            "attn.1": FakeTensor(8, 4),
            "mamba.0": [FakeTensor(8, 2), FakeTensor(8, 3)],
        }
        worker.register_kv_caches(caches)
        assert worker.group_to_layers == {
            0: ["attn.0", "attn.1"],
            1: ["mamba.0"],
        }

    def test_upstream_fallback_map_without_config(self, tmp_path, monkeypatch):
        """No kv_cache_config (defensive default) -> upstream behavior."""
        worker, _ = self._worker(tmp_path, monkeypatch)
        caches = {"attn.0": FakeTensor(8, 4)}
        worker.register_kv_caches(caches)
        assert worker.group_to_layers == {0: ["attn.0"]}

    def test_list_state_fill_is_per_block_no_clobber(self, tmp_path, monkeypatch):
        """THE Genesis extension: filling blocks [2, 5] of a Mamba layer
        must leave every other block row untouched (concurrent-request
        state protection — upstream #45080 fills the whole pool)."""
        worker, _ = self._worker(
            tmp_path, monkeypatch, kv_cache_config=_fake_kv_cache_config()
        )
        state_a = FakeTensor(8, 2, fill=7.0)
        state_b = FakeTensor(8, 3, fill=7.0)
        caches = {
            "attn.0": FakeTensor(8, 4),
            "attn.1": FakeTensor(8, 4),
            "mamba.0": [state_a, state_b],
        }
        worker.register_kv_caches(caches)
        worker._fill_blocks(1, [2, 5], num_tokens=32)
        for state in (state_a, state_b):
            for row in (2, 5):
                assert all(v == 0.015 for v in state.data[row]), row
            for row in (0, 1, 3, 4, 6, 7):
                assert all(v == 7.0 for v in state.data[row]), row

    def test_tensor_path_preserved(self, tmp_path, monkeypatch):
        worker, _ = self._worker(
            tmp_path, monkeypatch, kv_cache_config=_fake_kv_cache_config()
        )
        attn0 = FakeTensor(8, 4, fill=3.0)
        caches = {
            "attn.0": attn0,
            "attn.1": FakeTensor(8, 4, fill=3.0),
            "mamba.0": [FakeTensor(8, 2)],
        }
        worker.register_kv_caches(caches)
        worker._fill_blocks(0, [1], num_tokens=16)
        assert all(v == 0.015 for v in attn0.data[1])
        assert all(v == 3.0 for v in attn0.data[0])

    def test_out_of_range_block_ids_filtered(self, tmp_path, monkeypatch):
        worker, _ = self._worker(
            tmp_path, monkeypatch, kv_cache_config=_fake_kv_cache_config()
        )
        state = FakeTensor(4, 2, fill=1.0)
        caches = {
            "attn.0": FakeTensor(4, 4),
            "attn.1": FakeTensor(4, 4),
            "mamba.0": [state],
        }
        worker.register_kv_caches(caches)
        # Block 99 is out of range for a 4-block tensor — must be dropped.
        worker._fill_blocks(1, [1, 99], num_tokens=32)
        assert all(v == 0.015 for v in state.data[1])
        assert all(v == 1.0 for v in state.data[0])

    def test_unknown_cache_type_skipped_with_warning(self, tmp_path, monkeypatch):
        worker, fake_logger = self._worker(
            tmp_path, monkeypatch, kv_cache_config=_fake_kv_cache_config()
        )
        caches = {
            "attn.0": {"weird": "dict"},
            "attn.1": FakeTensor(4, 4),
            "mamba.0": [FakeTensor(4, 2)],
        }
        worker.register_kv_caches(caches)
        worker._fill_blocks(0, [0], num_tokens=16)  # must not raise
        assert any("attn.0" in w for w in fake_logger.warnings)


# ─────────────────────────────────────────────────────────────────────
# 4. Replacement contract — faithful #45080 + Genesis extras
# ─────────────────────────────────────────────────────────────────────


class TestReplacementContract:
    def test_fill_keeps_per_block_indexing_for_states(self):
        """The list/tuple branch must keep the block-row indexing —
        NEVER the whole-pool fill upstream uses."""
        M = _pn382()
        subs = {sp.name: sp for sp in M.build_sub_patches()}
        repl = subs["pn382_hybrid_per_block_fill"].replacement
        assert "isinstance(kv_cache, torch.Tensor)" in repl
        assert "isinstance(kv_cache, (list, tuple))" in repl
        assert "valid_block_ids" in repl
        # Upstream whole-pool calls must not appear.
        assert ".fill_(" not in repl
        assert ".normal_(" not in repl

    def test_group_map_built_from_kv_cache_groups(self):
        M = _pn382()
        subs = {sp.name: sp for sp in M.build_sub_patches()}
        repl = subs["pn382_real_group_map"].replacement
        assert "kv_cache_groups" in repl
        assert "group.layer_names" in repl
        # Defensive fallback to the upstream single-group map.
        assert "{0: list(kv_caches.keys())}" in repl

    def test_ctor_threads_kv_cache_config(self):
        M = _pn382()
        subs = {sp.name: sp for sp in M.build_sub_patches()}
        repl = subs["pn382_worker_ctor_kv_cache_config"].replacement
        assert "DecodeBenchConnectorWorker(" in repl
        assert "kv_cache_config" in repl
        # Scheduler-side ctor stays untouched.
        assert (
            "self.connector_scheduler = DecodeBenchConnectorScheduler(vllm_config)"
            in repl
        )


# ─────────────────────────────────────────────────────────────────────
# 5. Self-collision invariants (tools/lint_drift_markers.py contract)
# ─────────────────────────────────────────────────────────────────────


class TestSelfCollision:
    def test_drift_markers_disjoint_from_emitted_text(self):
        M = _pn382()
        marker_line = f"# [Genesis wiring marker: {M.GENESIS_PN382_MARKER}]\n"
        replacements = [sp.replacement for sp in M.build_sub_patches()]
        for dm in M._DRIFT_MARKERS:
            if dm.startswith("[Genesis"):
                continue  # defended convention — exempt from the lint
            for repl in replacements:
                assert dm not in repl, (dm, repl[:80])
            assert dm not in marker_line

    def test_drift_markers_absent_from_pristine_fixture(self):
        M = _pn382()
        src = _fake_pristine_connector()
        for dm in M._DRIFT_MARKERS:
            assert dm not in src


# ─────────────────────────────────────────────────────────────────────
# 6. Module apply() contract — env gate
# ─────────────────────────────────────────────────────────────────────


class TestModuleApply:
    def test_skips_when_env_unset(self, monkeypatch):
        M = _pn382()
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN382_DECODE_BENCH_HYBRID_FILL", raising=False
        )
        status, detail = M.apply()
        assert status == "skipped"
        assert "GENESIS_ENABLE_PN382_DECODE_BENCH_HYBRID_FILL" in detail

    def test_applies_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN382_DECODE_BENCH_HYBRID_FILL", "1")
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        M = _pn382()
        target = tmp_path / "decode_bench_connector.py"
        target.write_text(_fake_pristine_connector(), encoding="utf-8")
        monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: str(target))
        monkeypatch.setattr(M, "vllm_install_root", lambda: str(tmp_path))
        status, detail = M.apply()
        assert status == "applied", detail
        assert "45080" in detail
        ast.parse(target.read_text(encoding="utf-8"))
        assert M.is_applied()

    def test_skips_when_target_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN382_DECODE_BENCH_HYBRID_FILL", "1")
        M = _pn382()
        monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: None)
        monkeypatch.setattr(M, "vllm_install_root", lambda: str(tmp_path))
        status, _ = M.apply()
        assert status == "skipped"

    def test_marker_tracks_upstream_pr(self):
        M = _pn382()
        assert "45080" in M.GENESIS_PN382_MARKER


# ─────────────────────────────────────────────────────────────────────
# 7. Against the real pristine pin (opportunistic)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not PIN_CONNECTOR.is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAgainstPristinePin:
    def test_fixture_regions_match_pin(self):
        """Embedded portable regions stay byte-identical to the pin."""
        src = PIN_CONNECTOR.read_text(encoding="utf-8")
        for region in (
            CTOR_REGION,
            SCHEDULER_INIT_REGION,
            WORKER_INIT_REGION,
            GROUP_MAP_REGION,
            FILL_REGION,
        ):
            assert src.count(region) == 1

    def test_anchors_unique_and_markers_absent(self):
        M = _pn382()
        src = PIN_CONNECTOR.read_text(encoding="utf-8")
        for sp in M.build_sub_patches():
            assert src.count(sp.anchor) == 1, sp.name
            assert sp.replacement not in src, sp.name
        for dm in M._DRIFT_MARKERS:
            assert dm not in src

    def test_full_file_apply_and_compile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        M = _pn382()
        target = tmp_path / "decode_bench_connector.py"
        target.write_text(
            PIN_CONNECTOR.read_text(encoding="utf-8"), encoding="utf-8"
        )
        monkeypatch.setattr(M, "resolve_vllm_file", lambda rel: str(target))
        patcher = M._make_patcher()
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, failure
        ast.parse(target.read_text(encoding="utf-8"))
