# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard — no `tier="engine"` entries in this
public sndr_core PATCH_REGISTRY.

Iron Rule #1: this repo is `sndr_core` (public, Apache 2.0). Engine-tier
patches live in the separate private `sndr_engine` repo. A
`tier="engine"` entry in this registry is a policy violation AND a
silent-failure bug — the `_check_tier_gate` in
vllm/sndr_core/dispatcher/decision.py requires both:

  - the commercial `vllm.sndr_engine` package present
  - a valid `SNDR_ENGINE_LICENSE_KEY` env or `~/.sndr/license.json`
  - sndr_engine major version matches sndr_core

Without those, the patch silently skips with reason "tier=engine:
vllm.sndr_engine not installed". Operators who enable the
`GENESIS_ENABLE_*` env flag see no apply error in boot logs — just
a `skipped` entry buried in the per-patch chatter. The feature is
DEAD without anyone noticing.

v11.3.0 bug discovered: PN289 (§6.H10 Prometheus process-info gauge)
was registered as `tier="engine"` but its implementation lives at
`vllm/sndr_core/observability/genesis_process_info.py` (public,
Apache 2.0). Fixed to `tier="community"` in the same commit as this
test. Every operator who set `GENESIS_ENABLE_PN289_PROCESS_INFO=1`
to get §6.H10 enterprise observability before this fix got a silent
no-op — no `genesis_process_info` gauge in their Prometheus.

This test pins the boundary so the same class of bug can't return.

Allowlist via `_KNOWN_ENGINE_TIER_ENTRIES` if a future patch genuinely
requires engine-tier (would also need to live in sndr_engine repo,
not here). Empty at v11.3.0.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.3.0+ Iron Rule #1 regression guard.
"""
from __future__ import annotations


# Empty allowlist at v11.3.0. Adding an entry here requires a comment
# explaining why the patch genuinely needs engine-tier gating in the
# public repo (typically NEVER — engine code belongs in sndr_engine).
_KNOWN_ENGINE_TIER_ENTRIES: frozenset[str] = frozenset({
    # No allowlist entries at v11.3.0.
})


def test_no_engine_tier_patches_in_public_registry():
    """Iron Rule #1: no `tier="engine"` entries in this public
    sndr_core PATCH_REGISTRY. Pin the boundary."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    offenders = [
        pid for pid, meta in PATCH_REGISTRY.items()
        if isinstance(meta, dict)
        and meta.get("tier") == "engine"
        and pid not in _KNOWN_ENGINE_TIER_ENTRIES
    ]
    if offenders:
        raise AssertionError(
            f"{len(offenders)} tier='engine' entries in PUBLIC sndr_core "
            f"PATCH_REGISTRY — Iron Rule #1 violation. These patches "
            f"silently skip on every boot via _check_tier_gate without "
            f"the commercial sndr_engine package. Either move the "
            f"implementation to sndr_engine OR change tier to "
            f"'community'.\n\nOffenders: {offenders}"
        )


def test_pn289_is_community_tier_not_engine():
    """Regression guard — specifically pin PN289 to community tier.

    Pre-fix PN289 was tier='engine' but its code is in public
    sndr_core/observability/. Silently disabled §6.H10 Prometheus
    process_info gauge for every operator. The implementation is
    Genesis-original (Apache 2.0) — community tier is correct."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    entry = PATCH_REGISTRY.get("PN289")
    assert entry is not None, "PN289 missing from PATCH_REGISTRY"
    assert entry.get("tier") == "community", (
        f"PN289 tier='{entry.get('tier')}', expected 'community'. "
        f"v11.3.0 bug fix regression guard."
    )


def test_tier_distribution_baseline():
    """Document the current tier distribution. v11.3.0 baseline:
    all 241 entries are community-tier. Any future engine-tier entry
    must be explicitly justified + allowlisted in this test."""
    from collections import Counter
    from sndr.dispatcher.registry import PATCH_REGISTRY
    counts = Counter(
        meta.get("tier", "<missing>")
        for meta in PATCH_REGISTRY.values()
        if isinstance(meta, dict)
    )
    # Pin: zero engine-tier in v11.3.0 baseline
    assert counts.get("engine", 0) == 0, (
        f"Engine-tier patch count {counts.get('engine')}, expected 0 at "
        f"v11.3.0 baseline. Update _KNOWN_ENGINE_TIER_ENTRIES with a "
        f"justification comment if this is intentional."
    )
    # Sanity: at least 200 community entries (registry has 241 in v11.3.0)
    assert counts.get("community", 0) >= 200, (
        f"Community-tier count {counts.get('community')} unexpectedly low"
    )
