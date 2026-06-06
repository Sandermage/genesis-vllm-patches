# SPDX-License-Identifier: Apache-2.0
"""Pure-data query layer for ``sndr patches explain <patch_id>`` (M.6.1)."""
from __future__ import annotations

from typing import Optional

from .types import ExplainView


def resolve_patch_id(patch_id: str) -> Optional[str]:
    """Return the canonical-cased registry key, or ``None`` if absent.

    Falls back to case-insensitive lookup so operators can type ``p67``
    or ``pn82``; the registry's actual casing wins.
    """
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY

    if patch_id in PATCH_REGISTRY:
        return patch_id
    for key in PATCH_REGISTRY:
        if key.lower() == patch_id.lower():
            return key
    return None


def suggest_candidates(patch_id: str, *, limit: int = 8) -> list[str]:
    """Closest-prefix candidate suggestions for an unknown ``patch_id``."""
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY

    prefix = patch_id[:2].upper()
    return sorted(k for k in PATCH_REGISTRY if k.startswith(prefix))[:limit]


def explain_patch(patch_id: str) -> Optional[ExplainView]:
    """Return an ``ExplainView`` for the given id, or ``None`` if absent.

    Performs the live ``should_apply`` probe inside a try/except so a
    missing vllm runtime (Mac dev hosts) does not break the API
    contract — ``live_decision`` simply becomes ``None``.
    """
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    from vllm.sndr_core.dispatcher.spec import patch_spec_for

    canonical = resolve_patch_id(patch_id)
    if canonical is None:
        return None
    meta = PATCH_REGISTRY[canonical]
    spec = patch_spec_for(canonical, meta)

    live: Optional[tuple[bool, str]] = None
    live_error: Optional[str] = None
    try:
        from vllm.sndr_core.dispatcher import should_apply

        applied, reason = should_apply(canonical)
        live = (bool(applied), str(reason))
    except Exception as e:
        live = None
        live_error = type(e).__name__

    return ExplainView(
        patch_id=canonical,
        meta=meta,
        spec=spec,
        live_decision=live,
        live_decision_error=live_error,
    )
