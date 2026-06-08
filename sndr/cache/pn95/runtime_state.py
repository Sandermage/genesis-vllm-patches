# SPDX-License-Identifier: Apache-2.0
"""PN95 runtime-state lifecycle — TierManager singleton install / reset.

Owns the four lifecycle functions:

  * ``_detect_upstream_offload_connector`` — coexistence gate
  * ``init_from_config`` — install the TierManager singleton from a
    ModelConfig (idempotent, replace-with-warning)
  * ``tier_manager`` — read accessor for tests and observability
  * ``reset_for_tests`` — drop the singleton + cached state

M.4.2.A scope: function extraction only. The mutable state singletons
(``_TM``, ``_LOCK``, ``_LAST_GROUP_IDS_BY_HASH``) stay defined in
``_pn95_runtime`` because ~36 reader sites across the still-monolithic
file consult the local binding directly. Moving the canonical name
would break that alias the same way it broke ``_PN95_STATS`` in M.4.1
(see commit ``4a583783``).

To preserve byte-identical semantics with the original ``global _TM``
+ assignment pattern, ``init_from_config`` and ``reset_for_tests``
late-import ``_pn95_runtime`` and rebind ``_rt._TM`` via explicit
attribute assignment. This mutates the SAME module attribute the 36
reader sites consult — no aliasing required.

Extracted from ``_pn95_runtime.py`` in M.4.2.A. The legacy module
re-exports each function so:

  * existing tests that do ``rt.reset_for_tests()`` / ``rt.tier_manager()``
    / ``rt.init_from_config(cfg)`` continue to resolve through the shim
  * text-patch anchors that import these names directly are unaffected
  * ``register_kv_caches`` / ``init_mamba_exclusions_from_kv_groups``
    (which stay in ``_pn95_runtime`` for now) keep their ``global _TM``
    declarations harmlessly — they read the local module's ``_TM``
    which is the canonical binding
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .gates import _enabled

log = logging.getLogger("genesis.pn95")


def _detect_upstream_offload_connector(cfg: Any) -> Optional[str]:
    """Return the upstream KV-offload connector class name when one is
    declared on the engine config, else None.

    Mirrors the upstream v1/kv_offload/ framework gate: when an
    OffloadingConnector is wired in, the upstream offload manager
    owns block residency. Running PN95 in parallel would double-manage
    the same blocks and lead to undefined behaviour.

    Probe order:
      1. cfg.kv_transfer_config.kv_connector  (vLLM v0/v1 standard)
      2. cfg.kv_transfer_config.connector_class  (newer attribute)
      3. cfg.cache_config.offload_connector  (PN95 sibling spec)

    A non-empty truthy string is treated as a positive detection; any
    other shape (None, empty, non-string) is treated as not present.
    """
    candidates = (
        ("kv_transfer_config", "kv_connector"),
        ("kv_transfer_config", "connector_class"),
        ("kv_transfer_config", "kv_connector_module_path"),  # PR #40020 surface
        ("cache_config", "offload_connector"),
        ("cache_config", "kv_connector"),                    # older alias
    )
    for top, attr in candidates:
        block = getattr(cfg, top, None)
        if block is None:
            continue
        value = getattr(block, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    # Fallback scan: any *connector* attribute on kv_transfer_config with
    # a truthy-string value indicates an offload framework is wired in.
    # Catches custom user spec names that bypass the well-known attrs above.
    block = getattr(cfg, "kv_transfer_config", None)
    if block is not None:
        try:
            attrs = vars(block)
        except TypeError:
            attrs = {}
        for k, v in attrs.items():
            if "connector" in k.lower() and isinstance(v, str) and v.strip():
                return v
    return None


def init_from_config(cfg: Any) -> bool:
    """Install the TierManager singleton from a ModelConfig.

    Returns True iff a manager was installed; False when:
      - PN95 is disabled via env
      - cfg has no cache_config.tiers
      - upstream KV-offload connector is wired in (avoid double-manage)
      - import error / construction error (logged + swallowed)

    Idempotent: re-calling with the same cfg leaves the singleton
    untouched. Re-calling with a different cfg replaces it (with a
    warning logged).
    """
    if not _enabled():
        return False
    # Late import — `_TM` and `_LOCK` ownership stays in `_pn95_runtime`
    # during M.4.2.A; we mutate the module attribute directly so the
    # 36 reader sites in that file see the canonical binding.
    from sndr.cache import _pn95_runtime as _rt
    with _rt._LOCK:
        # Upstream offload coexistence gate. The upstream v1/kv_offload
        # OffloadingManager assumes exclusive block residency. Letting
        # PN95 also touch / demote blocks would race the upstream
        # prepare_load / complete_store handshake. Detect and skip.
        upstream = _detect_upstream_offload_connector(cfg)
        if upstream is not None:
            log.warning(
                "[PN95] upstream KV offload connector detected (%s) — "
                "skipping PN95 install to avoid double-managing blocks. "
                "Disable the upstream connector or unset "
                "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE to silence this.",
                upstream,
            )
            return False
        try:
            from sndr.cache.tier_manager import make_tier_manager
            new_tm = make_tier_manager(cfg)
        except Exception as e:
            log.warning("[PN95] init_from_config: import/build failed: %s", e)
            return False
        if new_tm is None:
            # 2026-06-04 observability fix: surface why make_tier_manager
            # silently returned None — operators couldn't tell empty tiers
            # apart from missing cfg apart from install actually working.
            cc = getattr(cfg, "cache_config", None)
            log.warning(
                "[PN95] init_from_config: make_tier_manager returned None "
                "(cache_config=%s, tiers=%s) — TierManager NOT installed",
                "missing" if cc is None else "present",
                getattr(cc, "tiers", None) if cc is not None else "n/a",
            )
            return False
        if _rt._TM is not None and _rt._TM is not new_tm:
            log.warning("[PN95] replacing existing TierManager singleton")
        _rt._TM = new_tm
        # 2026-06-04: promote success log info → warning so it survives
        # the VLLM_LOGGING_LEVEL=WARNING filter set in hardware YAML.
        log.warning("[PN95] TierManager installed: %s", _rt._TM.stats())
        return True


def tier_manager() -> Optional[Any]:
    """Accessor for tests + observability."""
    from sndr.cache import _pn95_runtime as _rt
    return _rt._TM


def reset_for_tests() -> None:
    """Drop the singleton + cached state. Used by pytest fixtures."""
    from sndr.cache import _pn95_runtime as _rt
    with _rt._LOCK:
        _rt._TM = None
        _rt._LAST_GROUP_IDS_BY_HASH = {}
