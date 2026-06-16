# SPDX-License-Identifier: Apache-2.0
"""G4_60a — inject ``TQSlidingWindowSpec`` into ``vllm.v1.kv_cache_interface``.

================================================================
PROBLEM
================================================================

vllm pin ``0.20.2rc1.dev371+gbf610c2f5`` ships ``TQFullAttentionSpec``
(line 311 of ``vllm/v1/kv_cache_interface.py``) but lacks the sibling
``TQSlidingWindowSpec``. As a result, sliding-window attention layers
on a model with ``--kv-cache-dtype turboquant_*`` fall back to plain
``SlidingWindowSpec`` and never report a TQ-compressed ``page_size_bytes``.
vllm then sizes the KV pool by the raw ``head_size × dtype`` formula for
those layers, defeating compression on Gemma 4's sliding tier
(50 of ~64 layers, ``head_dim=256``).

Upstream fix lives in PR #42637 (lesj0610, OPEN as of 2026-05-17):
https://github.com/vllm-project/vllm/pull/42637

This Genesis patch is the first leaf of the PR #42637 cherry-pick stack
(G4_60a through G4_60m). It is purely additive — it adds a new class
and tightens ``TQFullAttentionSpec.merge()``'s isinstance check. No
existing vllm code path changes behaviour until a *caller* actually
constructs a ``TQSlidingWindowSpec`` instance (which happens in G4_60g
when ``Attention.get_kv_cache_spec`` is patched to dispatch).

================================================================
FIX
================================================================

Hook ``vllm.v1.kv_cache_interface`` at module load time and:

  1. Define ``TQSlidingWindowSpec(SlidingWindowSpec)`` as a frozen
     dataclass with ``tq_slot_size: int = 0``, exactly mirroring
     ``TQFullAttentionSpec``'s structure (PR #42637 lines 501-522).

  2. Override ``real_page_size_bytes`` to compute
     ``block_size × num_kv_heads × tq_slot_size`` when ``tq_slot_size > 0``
     (falls through to ``SlidingWindowSpec``'s formula otherwise — safe).

  3. Override ``merge()`` to assert all merged specs are
     ``TQSlidingWindowSpec`` with identical ``tq_slot_size`` — this
     keeps cache-group homogeneity invariants that the KVCacheManager
     relies on.

  4. Tighten ``TQFullAttentionSpec.merge()`` with the matching
     ``isinstance`` assertion that PR #42637 adds at line 329 (without
     the assert, plain ``FullAttentionSpec`` siblings could be merged
     into a TQ group and silently drop their ``tq_slot_size``).

================================================================
SCOPE
================================================================

Active only when ``GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC=1``. When
opted in, it:

  * Adds ``TQSlidingWindowSpec`` symbol on ``vllm.v1.kv_cache_interface``
    module (idempotent).
  * Replaces ``TQFullAttentionSpec.merge`` with the tightened version.

Compatible with every model — only affects code paths that reach
``TQ*Spec.real_page_size_bytes`` or ``TQ*Spec.merge``, which require
opting in via ``--kv-cache-dtype turboquant_*`` *and* one of the
downstream G4_60g/h dispatch patches that constructs the spec class.

================================================================
RISK
================================================================

Adding a class to a frozen module is a runtime mutation; the new
class is exposed at ``vllm.v1.kv_cache_interface.TQSlidingWindowSpec``
but ``isinstance(spec, vllm.v1.kv_cache_interface.SlidingWindowSpec)``
checks elsewhere continue to return True (TQSlidingWindowSpec is a
subclass). No existing call site sees a behaviour change unless it
explicitly constructs ``TQSlidingWindowSpec`` (which only G4_60g does).

The ``merge()`` tightening can produce ``AssertionError`` if a caller
passes a mixed list of ``TQFullAttentionSpec`` and plain
``FullAttentionSpec`` to ``merge()`` — but that mix would already
break the KV cache group invariant (heterogeneous specs cannot share
a page-aligned buffer). Failing loud here is safer than silently
collapsing to non-TQ storage.

================================================================
REFERENCES
================================================================

  * Upstream PR: https://github.com/vllm-project/vllm/pull/42637
  * Upstream commit (TQSlidingWindowSpec): ``fdeb14981`` on branch
    ``fork-gemma4-e4b-tq-clean-20260510`` (PR #42637 HEAD as of
    2026-05-16), file ``vllm/v1/kv_cache_interface.py`` lines 501-522.
  * Related G4 patch: G4_60g (Attention.get_kv_cache_spec dispatch).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.turboquant.g4_60a_tq_sliding_spec")

GENESIS_G4_60A_MARKER = (
    "Genesis G4_60a inject TQSlidingWindowSpec into "
    "vllm.v1.kv_cache_interface (PR #42637 cherry-pick)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC"
_APPLIED = False
_ORIGINAL_TQ_FULL_MERGE = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Inject TQSlidingWindowSpec class + tighten TQFullAttentionSpec.merge.

    Returns:
        Tuple ``(status, message)`` where ``status`` is ``"applied"``,
        ``"skipped"``, or ``"error"``.
    """
    global _APPLIED, _ORIGINAL_TQ_FULL_MERGE

    if not _env_enabled():
        return "skipped", (
            f"G4_60a disabled (set {_ENV_ENABLE}=1 to inject "
            "TQSlidingWindowSpec — PR #42637 cherry-pick prerequisite)"
        )

    if _APPLIED:
        return "applied", "G4_60a already installed (idempotent)"

    try:
        from vllm.v1 import kv_cache_interface as _kci
    except ImportError as e:
        return "skipped", f"vllm.v1.kv_cache_interface not importable: {e}"

    if hasattr(_kci, "TQSlidingWindowSpec"):
        # Either a future pin already merged PR #42637, or another
        # apply() ran. Either way: nothing to do.
        _APPLIED = True
        return "applied", (
            "TQSlidingWindowSpec already present in vllm.v1.kv_cache_interface "
            "(pin may have merged PR #42637 natively)"
        )

    try:
        from dataclasses import dataclass, replace
        from typing import Self
    except ImportError as e:
        return "error", f"stdlib dataclass/typing not importable: {e}"

    # Anchor on existing classes — fail loud if upstream restructured.
    try:
        SlidingWindowSpec = _kci.SlidingWindowSpec
        TQFullAttentionSpec = _kci.TQFullAttentionSpec
    except AttributeError as e:
        return "error", (
            f"required base classes missing on vllm.v1.kv_cache_interface: {e}; "
            "PR #42637 cherry-pick prerequisites violated"
        )

    @dataclass(frozen=True, kw_only=True)
    class TQSlidingWindowSpec(SlidingWindowSpec):  # type: ignore[valid-type, misc]
        """SlidingWindowSpec with TQ-aware page size.

        Cherry-picked from upstream vllm PR #42637 (lesj0610).
        See module docstring for full rationale.
        """

        tq_slot_size: int = 0

        @property
        def real_page_size_bytes(self) -> int:  # type: ignore[override]
            if self.tq_slot_size > 0:
                return self.block_size * self.num_kv_heads * self.tq_slot_size
            return super().real_page_size_bytes

        @classmethod
        def merge(cls, specs: list["Self"]) -> "Self":  # type: ignore[override]
            assert all(isinstance(s, TQSlidingWindowSpec) for s in specs), (
                "TQSlidingWindowSpec can only merge other TQSlidingWindowSpec "
                "layers."
            )
            merged = super().merge(specs)
            assert all(
                s.tq_slot_size == specs[0].tq_slot_size for s in specs
            ), (
                "All TQ sliding-window layers in the same KV cache group must "
                "use the same tq_slot_size."
            )
            return replace(merged, tq_slot_size=specs[0].tq_slot_size)

    # Make this dynamically-defined class PICKLABLE across TP workers. Its
    # qualname is 'apply.<locals>.TQSlidingWindowSpec' (a local), which pickle
    # cannot resolve in a worker process — under TP>1 the KV-cache spec is
    # pickled to each worker, raising AttributeError "Can't get local object
    # 'apply.<locals>.TQSlidingWindowSpec'". Native dev491 has no module-level
    # TQSlidingWindowSpec, so we expose it on vllm.v1.kv_cache_interface (below)
    # and point __module__/__qualname__ there so pickle serialises it by that
    # importable reference. 2026-06-16 (dev491 native-TQ readiness).
    TQSlidingWindowSpec.__module__ = "vllm.v1.kv_cache_interface"
    TQSlidingWindowSpec.__qualname__ = "TQSlidingWindowSpec"

    # Expose on module so other PR #42637 cherry-pick leaves can import it.
    _kci.TQSlidingWindowSpec = TQSlidingWindowSpec

    # Tighten TQFullAttentionSpec.merge with PR #42637's isinstance assert.
    _ORIGINAL_TQ_FULL_MERGE = TQFullAttentionSpec.merge

    @classmethod
    def _tightened_tq_full_merge(cls, specs):
        # PR #42637 line 329: "TQFullAttentionSpec can only merge other
        # TQFullAttentionSpec layers."
        assert all(isinstance(s, TQFullAttentionSpec) for s in specs), (
            "TQFullAttentionSpec can only merge other TQFullAttentionSpec "
            "layers."
        )
        return _ORIGINAL_TQ_FULL_MERGE.__func__(cls, specs)

    _tightened_tq_full_merge._genesis_g4_60a_wrapped = True  # type: ignore[attr-defined]
    TQFullAttentionSpec.merge = _tightened_tq_full_merge

    _APPLIED = True
    log.info(
        "[G4_60a] TQSlidingWindowSpec injected; TQFullAttentionSpec.merge "
        "tightened with isinstance assertion."
    )
    return "applied", (
        "G4_60a installed: TQSlidingWindowSpec available at "
        "vllm.v1.kv_cache_interface.TQSlidingWindowSpec; "
        "TQFullAttentionSpec.merge now rejects non-TQ siblings."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Best-effort revert; not used in production (one-way patch at boot)."""
    global _APPLIED, _ORIGINAL_TQ_FULL_MERGE
    if not _APPLIED or _ORIGINAL_TQ_FULL_MERGE is None:
        return False
    try:
        from vllm.v1 import kv_cache_interface as _kci

        _kci.TQFullAttentionSpec.merge = _ORIGINAL_TQ_FULL_MERGE
        # Leave TQSlidingWindowSpec on module — removing it would break any
        # already-instantiated specs and is not needed for the typical
        # revert use case (test isolation).
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    _ORIGINAL_TQ_FULL_MERGE = None
    return True


__all__ = [
    "GENESIS_G4_60A_MARKER",
    "apply",
    "is_applied",
    "revert",
]
