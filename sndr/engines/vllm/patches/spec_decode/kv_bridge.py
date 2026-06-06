# SPDX-License-Identifier: Apache-2.0
"""kv_bridge — backend-agnostic K/V substitution primitives.

Library code. Does NOT monkey-patch anything. Provides:

  (a) ``gather_kv_at_positions(...)`` — pulls (K, V) slices out of a
      source kv_cache tensor at given logical positions, using the
      source's block_table. Layout-aware.

  (b) ``Bridge`` ABC + concrete implementations:
        ExactBridge        — copy as-is
        LayoutAdapterBridge — HND<->NHD axis swap
        GQARepeatBridge    — repeat_interleave on heads axis
        CompositeBridge    — chain
        UnsupportedBridge  — raises on use
        FunctionallyUnverifiedBridge — wraps an underlying bridge,
                                       requires explicit opt-in env

  (c) ``make_bridge(verdict, hints)`` — factory.

Provenance: extracted from
``integrations/gemma4/g4_78_drafter_target_kv_bridge.py`` (v2)
2026-05-20 per architectural directive (PN273). The Gemma 4 G4_78
patch can be rewritten on top of these primitives in a future
session; the legacy file remains untouched until then.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from .kv_contract import KVContract, Verdict

log = logging.getLogger("genesis.spec_decode.kv_bridge")


# ----------------------- Gather utility -----------------------

def gather_kv_at_positions(
    kv_cache: Any,
    block_table: Any,
    positions: list[int],
    *,
    layout: str,
    block_size: int,
    seq_idx: int = 0,
    torch_module: Any = None,
) -> tuple[Any, Any]:
    """Return (K, V) tensors at logical positions, where K and V have
    shape (num_tokens, num_kv_heads, head_size).

    kv_cache shape conventions:
      HND: (2, num_blocks, block_size, num_kv_heads, head_size)
      NHD: (num_blocks, 2, block_size, num_kv_heads, head_size)

    block_table is (num_seqs, max_blocks). seq_idx selects the row;
    default 0 (batch=1, the common spec-decode case).

    torch_module is the torch namespace. If None, imports torch
    lazily so this module stays torch-less at import time.
    """
    if torch_module is None:
        import torch as torch_module  # local import keeps module torch-less

    if not positions:
        # Construct empty K/V of correct shape
        if layout == "HND":
            K = kv_cache[0, 0:0, 0]  # zero-length slice
            V = kv_cache[1, 0:0, 0]
        else:
            K = kv_cache[0:0, 0, 0]
            V = kv_cache[0:0, 1, 0]
        return K, V

    block_ids: list[int] = []
    offsets: list[int] = []
    for P in positions:
        b_idx = P // block_size
        b_off = P % block_size
        block_ids.append(int(block_table[seq_idx, b_idx].item()))
        offsets.append(b_off)

    bid_t = torch_module.tensor(block_ids, dtype=torch_module.long,
                                device=kv_cache.device)
    off_t = torch_module.tensor(offsets, dtype=torch_module.long,
                                device=kv_cache.device)

    if layout == "HND":
        # kv_cache[0, block, off] = K; kv_cache[1, block, off] = V
        K = kv_cache[0, bid_t, off_t]
        V = kv_cache[1, bid_t, off_t]
    elif layout == "NHD":
        K = kv_cache[bid_t, 0, off_t]
        V = kv_cache[bid_t, 1, off_t]
    else:
        raise ValueError(f"unsupported layout for gather: {layout!r}")

    return K, V


def reconstruct_positions(
    query_start_loc: Any, seq_lens: Any,
) -> list[int]:
    """Reconstruct per-token logical positions from
    attn_metadata.query_start_loc + seq_lens.

    For each seq i: positions are seq_lens[i] - query_len + range(query_len).
    """
    try:
        qsl = query_start_loc.tolist() if hasattr(
            query_start_loc, "tolist") else list(query_start_loc)
        sl = seq_lens.tolist() if hasattr(
            seq_lens, "tolist") else list(seq_lens)
    except Exception as _e:
        log.warning("[kv_bridge] position reconstruction failed: %s", _e)
        return []
    positions: list[int] = []
    for i in range(len(sl)):
        qs = qsl[i]
        qe = qsl[i + 1] if i + 1 < len(qsl) else qsl[-1]
        seq_total = sl[i]
        query_len = qe - qs
        for j in range(query_len):
            positions.append(seq_total - query_len + j)
    return positions


# ----------------------- Bridge abstract -----------------------

class Bridge(ABC):
    """Abstract K/V bridge primitive.

    Sub-classes implement ``adapt`` which takes (raw_K, raw_V) extracted
    from the source cache via ``gather_kv_at_positions`` and returns
    the (K, V) tensors the destination Attention kernel expects.

    All Bridge instances are pure: no in-place modification of inputs.
    """

    name: str = "abstract"

    @abstractmethod
    def adapt(self, K: Any, V: Any, *,
              dst_dtype: Any | None = None) -> tuple[Any, Any]:
        raise NotImplementedError


class ExactBridge(Bridge):
    """No transform. Optional dtype cast at the end."""

    name = "ExactBridge"

    def adapt(self, K, V, *, dst_dtype=None):
        if dst_dtype is not None and K.dtype != dst_dtype:
            K = K.to(dst_dtype)
            V = V.to(dst_dtype)
        return K, V


class LayoutAdapterBridge(Bridge):
    """Source and destination disagree on KV cache layout (HND/NHD).

    The per-token K/V extracted by ``gather_kv_at_positions`` is already
    in (N, num_kv_heads, head_size) form regardless of source layout,
    so the layout adapter is effectively a NO-OP at this stage —
    the layout difference only matters during cache *write* (which is
    handled by the dest backend's own slot_mapping path). This class
    exists so the bridge graph is explicit and audit-friendly.
    """

    name = "LayoutAdapterBridge"

    def __init__(self, src_layout: str, dst_layout: str):
        self.src_layout = src_layout
        self.dst_layout = dst_layout

    def adapt(self, K, V, *, dst_dtype=None):
        if dst_dtype is not None and K.dtype != dst_dtype:
            K = K.to(dst_dtype)
            V = V.to(dst_dtype)
        return K, V


class GQARepeatBridge(Bridge):
    """Source has fewer KV heads than dest by integer factor.

    repeat_interleave on the heads axis (dim=1 of (N, H, D) tensor).
    """

    name = "GQARepeatBridge"

    def __init__(self, repeat: int):
        if repeat <= 1:
            raise ValueError(f"GQARepeatBridge requires repeat>1, got {repeat}")
        self.repeat = repeat

    def adapt(self, K, V, *, dst_dtype=None):
        K_out = K.repeat_interleave(self.repeat, dim=1)
        V_out = V.repeat_interleave(self.repeat, dim=1)
        if dst_dtype is not None and K_out.dtype != dst_dtype:
            K_out = K_out.to(dst_dtype)
            V_out = V_out.to(dst_dtype)
        return K_out, V_out


class CompositeBridge(Bridge):
    """Chain multiple bridges in order."""

    name = "CompositeBridge"

    def __init__(self, stages: list[Bridge]):
        if not stages:
            raise ValueError("CompositeBridge requires >=1 stage")
        self.stages = stages
        self.name = "CompositeBridge[" + "+".join(
            s.name for s in stages) + "]"

    def adapt(self, K, V, *, dst_dtype=None):
        for i, stage in enumerate(self.stages):
            # only last stage applies dtype cast
            dt = dst_dtype if i == len(self.stages) - 1 else None
            K, V = stage.adapt(K, V, dst_dtype=dt)
        return K, V


class UnsupportedBridge(Bridge):
    """Bridge factory returns this when the verdict is UNSUPPORTED.

    Calling .adapt() either passes through (with a warning) or raises,
    depending on ``strict`` flag.
    """

    name = "UnsupportedBridge"

    def __init__(self, reason: str, *, strict: bool = False):
        self.reason = reason
        self.strict = strict

    def adapt(self, K, V, *, dst_dtype=None):
        if self.strict:
            raise RuntimeError(
                f"UnsupportedBridge.adapt called: {self.reason}"
            )
        log.warning(
            "[kv_bridge] UnsupportedBridge passthrough: %s", self.reason,
        )
        return K, V


class FunctionallyUnverifiedBridge(Bridge):
    """Wraps another bridge but requires explicit operator opt-in env
    to actually apply. Without opt-in, .adapt() passes through
    (no-op) and logs a one-shot warning.

    Env: ``SNDR_ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN=1``
    (alias ``GENESIS_ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN`` still works)
    """

    name = "FunctionallyUnverifiedBridge"
    # P1 migration: bare suffix; reader resolves SNDR_/GENESIS_ prefix.
    _OPT_IN_ENV = "ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN"
    _warned: bool = False

    def __init__(self, inner: Bridge, *, reason: str = ""):
        self.inner = inner
        self.reason = reason
        self.name = f"FunctionallyUnverifiedBridge[{inner.name}]"

    def _opted_in(self) -> bool:
        from ...env import get_sndr_env_bool
        return get_sndr_env_bool(self._OPT_IN_ENV)

    def adapt(self, K, V, *, dst_dtype=None):
        if not self._opted_in():
            if not FunctionallyUnverifiedBridge._warned:
                log.warning(
                    "[kv_bridge] FunctionallyUnverifiedBridge passthrough "
                    "(set SNDR_%s=1 to enable inner=%s; reason=%s)",
                    self._OPT_IN_ENV, self.inner.name, self.reason,
                )
                FunctionallyUnverifiedBridge._warned = True
            return K, V
        return self.inner.adapt(K, V, dst_dtype=dst_dtype)


# ----------------------- Factory -----------------------

def make_bridge(
    verdict: Verdict,
    *,
    src: KVContract | None = None,
    dst: KVContract | None = None,
    hints: dict[str, Any] | None = None,
    strict_unsupported: bool = False,
) -> Bridge:
    """Build the right Bridge instance from a verdict + hints.

    hints typically come from ``compare_contracts(...)`` (third return
    value): ``gqa_repeat``, ``src_layout``, ``dst_layout``,
    ``pre_functional_gate_verdict``.

    For verdict == EXACT_COPY -> ExactBridge.
    For verdict == LAYOUT_ADAPTER -> LayoutAdapterBridge.
    For verdict == GQA_REPEAT -> GQARepeatBridge.
    For verdict == COMPOSITE_ADAPTER -> CompositeBridge of needed parts.
    For verdict == DEQUANT_REQUIRED -> NOT YET IMPLEMENTED, returns
        UnsupportedBridge.
    For verdict == ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED ->
        FunctionallyUnverifiedBridge wrapping the inferred inner bridge.
    For verdict == UNSUPPORTED -> UnsupportedBridge (strict or passthrough).
    """
    hints = hints or {}

    if verdict == Verdict.EXACT_COPY:
        return ExactBridge()

    if verdict == Verdict.LAYOUT_ADAPTER:
        return LayoutAdapterBridge(
            src_layout=hints.get("src_layout", "unknown"),
            dst_layout=hints.get("dst_layout", "unknown"),
        )

    if verdict == Verdict.GQA_REPEAT:
        return GQARepeatBridge(repeat=int(hints.get("gqa_repeat", 1)))

    if verdict == Verdict.COMPOSITE_ADAPTER:
        stages: list[Bridge] = []
        if "src_layout" in hints and "dst_layout" in hints and (
                hints["src_layout"] != hints["dst_layout"]):
            stages.append(LayoutAdapterBridge(
                src_layout=hints["src_layout"],
                dst_layout=hints["dst_layout"],
            ))
        repeat = int(hints.get("gqa_repeat", 1) or 1)
        if repeat > 1:
            stages.append(GQARepeatBridge(repeat=repeat))
        if not stages:
            return ExactBridge()
        return CompositeBridge(stages)

    if verdict == Verdict.DEQUANT_REQUIRED:
        return UnsupportedBridge(
            reason="DEQUANT_REQUIRED bridge not implemented yet",
            strict=strict_unsupported,
        )

    if verdict == Verdict.ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED:
        # Build the *underlying* structural bridge from hints, then wrap.
        inner_verdict_str = hints.get("pre_functional_gate_verdict",
                                      "COMPOSITE_ADAPTER")
        try:
            inner_verdict = Verdict(inner_verdict_str)
        except ValueError:
            inner_verdict = Verdict.COMPOSITE_ADAPTER
        # avoid recursion: pass require_functional_gate=False semantics
        inner = make_bridge(
            inner_verdict, src=src, dst=dst, hints=hints,
            strict_unsupported=strict_unsupported,
        )
        return FunctionallyUnverifiedBridge(
            inner,
            reason=(f"verdict downgraded from {inner_verdict_str} until "
                    f"runtime acceptance gate validates this contract pair"),
        )

    # UNSUPPORTED or any unknown
    return UnsupportedBridge(
        reason=f"verdict={verdict.value}",
        strict=strict_unsupported,
    )


__all__ = [
    "Bridge",
    "ExactBridge",
    "LayoutAdapterBridge",
    "GQARepeatBridge",
    "CompositeBridge",
    "UnsupportedBridge",
    "FunctionallyUnverifiedBridge",
    "make_bridge",
    "gather_kv_at_positions",
    "reconstruct_positions",
]
