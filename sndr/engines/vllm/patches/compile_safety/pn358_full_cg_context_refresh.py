# SPDX-License-Identifier: Apache-2.0
"""PN358 — FULL cudagraph forward-context refresh (vendor of vllm#44868).

Upstream bug (#44868, OPEN, weicj): during FULL CUDA graph capture, the
graph entry bakes in references to the forward-context tensors that
existed at capture time — attention metadata, slot mappings, ubatch
slices, DP metadata, additional kwargs. On replay, if the live forward
context for the current step carries FRESH tensors (different storage)
while the captured graph still reads the old ones, the replay silently
runs with stale metadata. Symptom class: wrong continuations,
repetitive/degenerate output under speculative decoding, output that
ignores part of the prompt — a serving-quality failure, not a crash.

Verified at our pin g303916e93 (0.22.1rc1.dev259): ``compilation/
cuda_graph.py`` has NO forward-context refresh on the replay path
(``entry.cudagraph.replay()`` is preceded only by the offloader sync);
the PR's helper names are absent (anchor counts byte-verified, each
``== 1``). All metadata classes engaged on our stack
(GDNAttentionMetadata, TurboQuantMetadata, FlashAttentionMetadata,
DPMetadata, UBatchSlices) are dataclasses, so the PR's
dataclass/Mapping/Sequence tree walk covers them.

Why this matters for Genesis PROD (2x A5000, FULL_AND_PIECEWISE via
PN125, MTP K=3, 287-patch overlay): several Genesis patches
(PN340/PN341/PN353 surface) rebuild attention-metadata buffers; any
path that hands a freshly allocated tensor to a step that replays an
already-captured FULL graph hits exactly this silent-corruption class.

Vendoring scope vs upstream #44868 (documented divergence):
  * VENDORED: capture-time recording of the forward-context tensor
    tree on the FULL-mode entry + pre-replay refresh of captured
    tensors from the live context (same three hook sites as the PR).
  * Genesis improvement #1 — data_ptr-pruned copy: upstream copy_()s
    EVERY captured leaf on EVERY replay, including the common case
    where the live tensor still aliases the captured storage (a pure
    self-copy: the graph already reads that storage directly). That
    unconditional walk-and-copy is the PR's 1-3% TPOT cost. PN358
    caches each captured leaf's data_ptr and copies ONLY leaves whose
    live tensor moved storage — the well-behaved persistent-buffer
    path costs a host-side pointer compare per leaf, zero kernel
    launches.
  * Genesis improvement #2 — GENESIS_PN358_MODE=detect: log-only audit
    mode. Hazards (live tensor storage differs from captured) are
    logged WARNING (warn-once per path) and counted, captured tensors
    are NOT mutated. This is the definitive audit of whether the
    Genesis overlay leaks fresh tensors into captured FULL graphs:
    run smoke+bench with detect on 35B/27B; zero hazard lines == the
    overlay is clean; hazard lines name the exact metadata path.
  * Robustness over upstream: shape-mismatched leaves skip the copy
    with a WARNING instead of raising mid-replay (upstream's bare
    ``dst.copy_(src)`` would crash the step); the refresh path never
    raises (self-disables after the first internal error); cyclic
    structures terminate (upstream's refresh walk has no memo);
    tensor-less branches are pruned at capture so a FULL entry whose
    context carries no tensors costs a single ``is not None`` check
    per replay.

Composes with PN353B / PN118 (both target turboquant_attn.py /
workspace.py — no file overlap; the only prior Genesis patch on
cuda_graph.py is retired PN13, now in _archive/). Self-skips when
upstream lands #44868: the drift markers below are exact substrings of
the PR's added code and absent at pin g303916e93.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn358_full_cg_context_refresh")

GENESIS_PN358_MARKER = (
    "Genesis PN358 FULL cudagraph forward-context refresh "
    "(vendor of vllm#44868)"
)

_CUDA_GRAPH_REL = "compilation/cuda_graph.py"

# Drift markers — exact substrings of #44868's added code, verified
# against `gh pr diff 44868` on 2026-06-11. Absent at pin g303916e93
# (both counts == 0, byte-verified) and deliberately NOT substrings of
# our own replacement texts (we name the entry field
# ``genesis_pn358_captured_fc`` and route through ``_g_pn358`` module
# functions, never through the PR's helper names).
_DRIFT_MARKERS = (
    "def _capture_forward_context_tensors() -> dict[str, Any]:",
    "entry.captured_forward_context_tensors",
)

# ── Splice texts (anchors byte-verified at pin g303916e93) ───────────

PN358_ENTRY_OLD = (
    "    # for cudagraph debugging, track the input addresses\n"
    "    # during capture, and check if they are the same during replay\n"
    "    input_addresses: list[int] | None = None\n"
)

PN358_ENTRY_NEW = (
    "    # for cudagraph debugging, track the input addresses\n"
    "    # during capture, and check if they are the same during replay\n"
    "    input_addresses: list[int] | None = None\n"
    "\n"
    "    # [Genesis PN358] forward-context tensor refs recorded at\n"
    "    # FULL-graph capture time; replays refresh stale metadata\n"
    "    # through this tree (vendor of vllm#44868, data_ptr-pruned).\n"
    "    genesis_pn358_captured_fc: Any | None = None\n"
)

PN358_CAPTURE_OLD = (
    "            entry.input_addresses = input_addresses\n"
    "            cudagraph = torch.cuda.CUDAGraph()\n"
)

PN358_CAPTURE_NEW = (
    "            entry.input_addresses = input_addresses\n"
    "            # [Genesis PN358] record forward-context tensor refs at\n"
    "            # FULL capture time so every replay can refresh stale\n"
    "            # metadata before the graph re-reads it (vllm#44868).\n"
    "            if self.runtime_mode == CUDAGraphMode.FULL:\n"
    "                from sndr.engines.vllm.patches.compile_safety import (\n"
    "                    pn358_full_cg_context_refresh as _g_pn358,\n"
    "                )\n"
    "                entry.genesis_pn358_captured_fc = (\n"
    "                    _g_pn358.genesis_pn358_capture(forward_context)\n"
    "                )\n"
    "            cudagraph = torch.cuda.CUDAGraph()\n"
)

PN358_REPLAY_OLD = (
    "        # Sync offloader before replay - ensures any external dependencies\n"
    "        # from pre-capture prefetches are satisfied.\n"
    "        get_offloader().sync_prev_onload()\n"
    "        entry.cudagraph.replay()\n"
)

PN358_REPLAY_NEW = (
    "        # Sync offloader before replay - ensures any external dependencies\n"
    "        # from pre-capture prefetches are satisfied.\n"
    "        get_offloader().sync_prev_onload()\n"
    "        # [Genesis PN358] refresh the captured forward-context tensor\n"
    "        # tree from the live context before FULL replay (vllm#44868).\n"
    "        # data_ptr-pruned: leaves whose live tensor still aliases the\n"
    "        # captured storage are skipped (the graph reads them as-is);\n"
    "        # GENESIS_PN358_MODE=detect logs hazards without mutating.\n"
    "        if entry.genesis_pn358_captured_fc is not None:\n"
    "            from sndr.engines.vllm.patches.compile_safety import (\n"
    "                pn358_full_cg_context_refresh as _g_pn358,\n"
    "            )\n"
    "            _g_pn358.genesis_pn358_refresh(\n"
    "                forward_context, entry.genesis_pn358_captured_fc\n"
    "            )\n"
    "        entry.cudagraph.replay()\n"
)

# ── Runtime: modes, stats, state ─────────────────────────────────────

_ENV_MODE = "GENESIS_PN358_MODE"
MODE_REFRESH = "refresh"
MODE_DETECT = "detect"

# Forward-context fields walked, in upstream #44868's order.
_CONTEXT_FIELDS = (
    "attn_metadata",
    "slot_mapping",
    "batch_descriptor",
    "ubatch_slices",
    "dp_metadata",
    "additional_kwargs",
)

_STAT_KEYS = (
    "captured_leaves",
    "refreshed",
    "pruned",
    "stale_detected",
    "shape_mismatch",
    "structural_mismatch",
    "errors",
)

_stats: dict[str, int] = dict.fromkeys(_STAT_KEYS, 0)
_warned_paths: set[str] = set()
_mode_cache: str | None = None
_disabled = False
_tensor_type: type | None = None


def reset_pn358_state() -> None:
    """Reset cached mode, counters, warn-once memory and the
    self-disable latch. Test hook + operator escape hatch."""
    global _mode_cache, _disabled
    _stats.update(dict.fromkeys(_STAT_KEYS, 0))
    _warned_paths.clear()
    _mode_cache = None
    _disabled = False


def get_stats() -> dict[str, int]:
    """Copy of the runtime counters (observability / tests)."""
    return dict(_stats)


def _is_tensor(obj: Any) -> bool:
    """Lazy torch.Tensor check — keeps this module torch-less at
    import time (collection safety); tests monkeypatch this seam."""
    global _tensor_type
    if _tensor_type is None:
        import torch

        _tensor_type = torch.Tensor
    return isinstance(obj, _tensor_type)


def _resolve_mode() -> str:
    """Resolve GENESIS_PN358_MODE once (cached until reset).

    ``refresh`` (default) — vendored #44868 behavior with the
    data_ptr-pruned copy. ``detect`` — log-only audit, no mutation.
    Unknown values fall back to refresh (fail-safe to correctness)
    with a one-time WARNING."""
    global _mode_cache
    if _mode_cache is not None:
        return _mode_cache
    raw = os.environ.get(_ENV_MODE, "").strip().lower()
    if raw in ("", MODE_REFRESH):
        _mode_cache = MODE_REFRESH
    elif raw == MODE_DETECT:
        _mode_cache = MODE_DETECT
    else:
        log.warning(
            "[Genesis PN358] unknown %s value %r — falling back to "
            "%r (valid: %r, %r)",
            _ENV_MODE, raw, MODE_REFRESH, MODE_REFRESH, MODE_DETECT,
        )
        _mode_cache = MODE_REFRESH
    return _mode_cache


class _CapturedLeaf:
    """One captured forward-context tensor: the tensor reference, its
    capture-time data_ptr (stable — the captured tensor never
    reallocates while the graph entry holds it) and shape, plus the
    human-readable path for detect-mode hazard logs."""

    __slots__ = ("path", "tensor", "data_ptr", "shape")

    def __init__(self, path: str, tensor: Any) -> None:
        self.path = path
        self.tensor = tensor
        self.data_ptr = tensor.data_ptr()
        self.shape = tuple(tensor.shape)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"_CapturedLeaf({self.path!r}, ptr=0x{self.data_ptr:x}, "
            f"shape={self.shape})"
        )


def iter_captured_leaves(node: Any):
    """Yield every _CapturedLeaf in a captured tree (audit/tests)."""
    seen: set[int] = set()

    def _walk(n: Any):
        if n is None:
            return
        if isinstance(n, _CapturedLeaf):
            yield n
            return
        if id(n) in seen:
            return
        seen.add(id(n))
        if isinstance(n, dict):
            for sub in n.values():
                yield from _walk(sub)
        elif isinstance(n, list):
            for sub in n:
                yield from _walk(sub)

    yield from _walk(node)


def _record_tree(obj: Any, path: str, memo: dict[int, Any]) -> Any:
    """Build the captured skeleton: dict/list mirror of ``obj`` with
    _CapturedLeaf at tensor positions and None where a branch holds no
    tensors (pruned — never walked again at replay time). Sequence
    skeletons keep positional None placeholders so the replay-time
    zip pairing stays aligned."""
    if _is_tensor(obj):
        _stats["captured_leaves"] += 1
        return _CapturedLeaf(path, obj)
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        skeleton: dict[str, Any] = {}
        memo[obj_id] = skeleton
        for f in dataclasses.fields(obj):
            sub = _record_tree(
                getattr(obj, f.name), f"{path}.{f.name}", memo
            )
            if sub is not None:
                skeleton[f.name] = sub
        return skeleton if skeleton else None
    if isinstance(obj, Mapping):
        skeleton = {}
        memo[obj_id] = skeleton
        for key, value in obj.items():
            sub = _record_tree(value, f"{path}[{key!r}]", memo)
            if sub is not None:
                skeleton[key] = sub
        return skeleton if skeleton else None
    if isinstance(obj, (list, tuple)):
        items = [
            _record_tree(value, f"{path}[{i}]", memo)
            for i, value in enumerate(obj)
        ]
        if not any(item is not None for item in items):
            return None
        skeleton_list = list(items)
        memo[obj_id] = skeleton_list
        return skeleton_list
    return None


def genesis_pn358_capture(forward_context: Any) -> dict[str, Any] | None:
    """Record the forward-context tensor tree at FULL-graph capture
    time. Returns None when the context holds no tensors (the replay
    hook then costs one ``is not None`` check). Never raises."""
    try:
        memo: dict[int, Any] = {}
        captured: dict[str, Any] = {}
        for name in _CONTEXT_FIELDS:
            sub = _record_tree(
                getattr(forward_context, name, None), name, memo
            )
            if sub is not None:
                captured[name] = sub
        return captured if captured else None
    except Exception as e:  # guard rail: never break graph capture
        _stats["errors"] += 1
        log.warning(
            "[Genesis PN358] forward-context capture failed (%s: %s) — "
            "entry will replay without refresh",
            type(e).__name__, e,
        )
        return None


def _warn_once(path: str, msg: str, *args: Any) -> None:
    if path in _warned_paths:
        return
    _warned_paths.add(path)
    log.warning(msg, *args)


def _refresh_leaf(src: Any, leaf: _CapturedLeaf, mode: str) -> None:
    if not _is_tensor(src):
        _stats["structural_mismatch"] += 1
        _warn_once(
            leaf.path,
            "[Genesis PN358] live forward-context leaf %s is no longer "
            "a tensor (%s) — skipped (structural drift)",
            leaf.path, type(src).__name__,
        )
        return
    if src.data_ptr() == leaf.data_ptr:
        # The live tensor still aliases the captured storage — the
        # graph already reads the fresh values; a copy would be a
        # self-copy kernel launch (upstream #44868's per-replay cost).
        _stats["pruned"] += 1
        return
    if mode == MODE_DETECT:
        _stats["stale_detected"] += 1
        _warn_once(
            leaf.path,
            "[Genesis PN358 detect] stale forward-context leaf %s: "
            "captured ptr=0x%x shape=%s, live ptr=0x%x shape=%s — a "
            "captured FULL graph would replay against stale metadata "
            "(no mutation in detect mode)",
            leaf.path, leaf.data_ptr, leaf.shape,
            src.data_ptr(), tuple(src.shape),
        )
        return
    if tuple(src.shape) != leaf.shape:
        _stats["shape_mismatch"] += 1
        _warn_once(
            leaf.path,
            "[Genesis PN358] live forward-context leaf %s moved storage "
            "AND changed shape (captured %s, live %s) — copy skipped "
            "(upstream copy_ would crash the replay); this signals a "
            "batch-descriptor keying bug upstream of the graph cache",
            leaf.path, leaf.shape, tuple(src.shape),
        )
        return
    leaf.tensor.copy_(src)
    _stats["refreshed"] += 1


def _refresh_tree(
    src: Any, node: Any, mode: str, visited: set[int]
) -> None:
    if node is None:
        return
    if isinstance(node, _CapturedLeaf):
        _refresh_leaf(src, node, mode)
        return
    if id(node) in visited:
        # Cycle/shared-subtree guard — upstream's refresh walk has no
        # memo and would recurse forever on self-referencing kwargs.
        return
    visited.add(id(node))
    if isinstance(node, dict):
        if dataclasses.is_dataclass(src) and not isinstance(src, type):
            for key, sub in node.items():
                _refresh_tree(getattr(src, key, None), sub, mode, visited)
        elif isinstance(src, Mapping):
            for key, sub in node.items():
                if key in src:
                    _refresh_tree(src[key], sub, mode, visited)
                # Missing key: live context dropped the entry —
                # upstream skips silently; we match (the captured
                # tensor keeps its last refreshed values).
        elif src is not None:
            _stats["structural_mismatch"] += 1
        return
    if isinstance(node, list):
        if (
            isinstance(src, Sequence)
            and not isinstance(src, (str, bytes, bytearray))
        ):
            # Length mismatch pairs the common prefix (upstream zip
            # semantics) — extra captured leaves keep their values.
            for src_value, sub in zip(src, node, strict=False):
                _refresh_tree(src_value, sub, mode, visited)
        elif src is not None:
            _stats["structural_mismatch"] += 1


def genesis_pn358_refresh(forward_context: Any, captured: Any) -> None:
    """Refresh (or, in detect mode, audit) the captured forward-context
    tensor tree against the live context before a FULL replay.

    Never raises into the replay path: the first internal error logs a
    WARNING and self-disables the refresh until reset_pn358_state()."""
    global _disabled
    if _disabled or not captured:
        return
    try:
        mode = _resolve_mode()
        visited: set[int] = set()
        for name, node in captured.items():
            _refresh_tree(
                getattr(forward_context, name, None), node, mode, visited
            )
    except Exception as e:  # guard rail: never break replay
        _stats["errors"] += 1
        _disabled = True
        log.warning(
            "[Genesis PN358] refresh failed (%s: %s) — self-disabled "
            "for this process (replays continue un-refreshed, matching "
            "pre-patch behavior); reset via reset_pn358_state()",
            type(e).__name__, e,
        )


# ── Wiring ───────────────────────────────────────────────────────────


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_CUDA_GRAPH_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN358 compilation/cuda_graph.py — FULL-graph forward-context "
            "refresh, data_ptr-pruned + detect mode (vendor of vllm#44868)"
        ),
        target_file=str(target),
        marker=GENESIS_PN358_MARKER,
        sub_patches=[
            # All three required: a partial splice (field without
            # refresh, or refresh without the field) is incoherent —
            # Layer 5 validates every anchor before writing anything.
            TextPatch(
                name="pn358_entry_captured_fc_field",
                anchor=PN358_ENTRY_OLD,
                replacement=PN358_ENTRY_NEW,
                required=True,
            ),
            TextPatch(
                name="pn358_capture_record",
                anchor=PN358_CAPTURE_OLD,
                replacement=PN358_CAPTURE_NEW,
                required=True,
            ),
            TextPatch(
                name="pn358_replay_refresh",
                anchor=PN358_REPLAY_OLD,
                replacement=PN358_REPLAY_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def apply() -> tuple[str, str]:
    """Install the three cuda_graph.py hooks. Never raises."""
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN358: {_CUDA_GRAPH_REL} not resolvable"
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN358 applied: FULL CUDA-graph entries now record their "
            "forward-context tensor tree at capture and refresh it "
            "before every replay (vendor of vllm#44868). Genesis "
            "extras: data_ptr-pruned copy (only leaves whose live "
            "tensor moved storage are copied — kills the PR's "
            "unconditional per-replay copy cost) and "
            "GENESIS_PN358_MODE=detect log-only audit of stale-"
            "metadata hazards."
        ),
        patch_name="PN358 cuda_graph forward-context refresh",
    )
