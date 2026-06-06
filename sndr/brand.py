# SPDX-License-Identifier: Apache-2.0
"""SNDR Core / Genesis brand strings (single source of truth).

Per Sander decision Q2 (2026-05-07, mixed approach):

    Backport patches from upstream PRs (vllm-project/vllm#NNNNN,
    SGLang#NNNN, llama.cpp#NNNN) keep the `Genesis ` marker prefix
    and `GENESIS_ENABLE_*` env vars. They are community-tier work
    that retains the established community brand.

    Sander-original patches (genesis-original Triton kernels, custom
    DFlash work, Sander-IP innovations) use `SNDR ` marker prefix
    and `SNDR_ENABLE_*` env vars. These are the canonical brand
    going forward for new Sander-IP work.

    Patcher Layer-1 marker check (vllm/_genesis/wiring/text_patch.py
    `_is_already_applied()`) recognizes BOTH prefixes — neither
    supersedes the other. Boot-time apply produces a status report
    showing how many patches use each prefix.

Public-facing brand: "Genesis (powered by SNDR Core)" for community,
"Genesis Pro (powered by SNDR Engine)" for commercial tier.

Why two prefixes coexist:
  Marker strings are baked into installed vllm files (prepended at
  apply time). Renaming `Genesis P15 …` → `SNDR P15 …` retroactively
  would break idempotency on every running container — they would
  re-apply on reboot, triggering re-write + verify cycle. The mixed
  approach keeps the existing baked markers valid AND allows new
  Sander-original work to use canonical SNDR prefix without disrupting
  community-tier deployments.
"""

# ── Public-facing brand ──────────────────────────────────────────────────
PUBLIC_BRAND_COMMUNITY = "Genesis"
PUBLIC_BRAND_PRO = "Genesis Pro"
TAGLINE_COMMUNITY = "Genesis (powered by SNDR Core)"
TAGLINE_PRO = "Genesis Pro (powered by SNDR Engine)"

# ── Internal package names ───────────────────────────────────────────────
PKG_NAME_CORE = "SNDR Core"
PKG_NAME_ENGINE = "SNDR Engine"

# ── Marker prefixes recognized by patcher idempotency check ─────────────
# Order matters for boot-time reporting (legacy first, canonical second).
MARKER_PREFIX_LEGACY = "Genesis "       # backport patches (community-tier)
MARKER_PREFIX_CANONICAL = "SNDR "       # Sander-original (canonical for new work)
RECOGNIZED_MARKER_PREFIXES = (MARKER_PREFIX_LEGACY, MARKER_PREFIX_CANONICAL)

# ── Env flag prefixes ────────────────────────────────────────────────────
# is_enabled() in env.py checks SNDR_* first, then GENESIS_* alias.
ENV_PREFIX_CANONICAL = "SNDR_ENABLE_"
ENV_PREFIX_LEGACY = "GENESIS_ENABLE_"
RECOGNIZED_ENV_PREFIXES = (ENV_PREFIX_CANONICAL, ENV_PREFIX_LEGACY)


def marker_prefix_for(tier: str) -> str:
    """Return the canonical marker prefix for the given tier.

    tier="community" → "Genesis " (backports keep community brand)
    tier="engine"    → "SNDR "    (Sander-original IP, canonical brand)

    Anything else falls back to the canonical SNDR prefix.
    """
    if tier == "community":
        return MARKER_PREFIX_LEGACY
    return MARKER_PREFIX_CANONICAL


__all__ = [
    "PUBLIC_BRAND_COMMUNITY",
    "PUBLIC_BRAND_PRO",
    "TAGLINE_COMMUNITY",
    "TAGLINE_PRO",
    "PKG_NAME_CORE",
    "PKG_NAME_ENGINE",
    "MARKER_PREFIX_LEGACY",
    "MARKER_PREFIX_CANONICAL",
    "RECOGNIZED_MARKER_PREFIXES",
    "ENV_PREFIX_CANONICAL",
    "ENV_PREFIX_LEGACY",
    "RECOGNIZED_ENV_PREFIXES",
    "marker_prefix_for",
]
