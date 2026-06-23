# SPDX-License-Identifier: Apache-2.0
"""Y13 (UNIFIED_CONFIG plan 2026-05-09) — runtime caveats registry.

Caveats are runtime conditions that are bookkeeping-only known issues
(NOT crashes, NOT silent corruption — those go in the dispatcher's
hard-fail path). Operators see them when their host matches a caveat's
condition matrix:

  - Proxmox LXC + kernel 6.17 + uvloop → known async-IO stall
  - WSL2 + GENESIS_FLA_GUARD_NUM_HEADS != 12 → off-by-one in TP layout
  - Single 3090 + max_model_len ≥ 145K + vision encoder → club-3090 #58
    (resolved by Path C v7.73.x — caveat fires only when PN95 OFF)

The registry is **source of truth** here in code. Per-config additions
live in `cache_config.runtime_caveats` (NOT YET WIRED — Tier 4 future
item). The matcher pulls host facts from `sndr.deps.checkers`.

Public API:
  - `KNOWN_CAVEATS` — tuple of `Caveat` instances
  - `match_caveats(facts)` — return list of triggered caveat IDs
  - `get_caveat(caveat_id)` — return the Caveat by id

Matcher uses a small DSL: each match key is a function over facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ─── Caveat dataclass + registry ────────────────────────────────────────


@dataclass(frozen=True)
class Caveat:
    """One known runtime condition + its match predicate.

    `match_fn(facts)` returns True iff the caveat applies. `facts` is
    a dict — typically the output of `sndr.deps.checkers
    .inspect_host()` augmented with operator-supplied keys (config_key,
    enabled_patches, etc.).
    """
    id: str
    severity: str  # 'info' | 'warning' | 'error'
    title: str
    message: str
    docs_url: Optional[str] = None
    # Predicate: callable(facts: dict) -> bool. We keep this as a
    # function (not a YAML expression) for type-safety + IDE support.
    match_fn: Optional[Callable[[dict], bool]] = field(
        default=None, repr=False, compare=False,
    )

    def matches(self, facts: dict) -> bool:
        if self.match_fn is None:
            return False
        try:
            return bool(self.match_fn(facts))
        except Exception:
            return False


# ─── Match-fn helpers ────────────────────────────────────────────────────


def _is_proxmox_lxc(facts: dict) -> bool:
    """Detected via `systemd-detect-virt --container` returning 'lxc'.

    `facts` may carry an explicit `virtualization` key or we fall back
    to checking inventory's os.distro / kernel hints.
    """
    virt = facts.get("virtualization", "")
    if virt:
        return "lxc" in virt.lower()
    # Fallback: check kernel signature commonly seen in PVE
    os_ = facts.get("os", {})
    if isinstance(os_, dict):
        release = os_.get("release", "")
        return "pve" in release.lower()
    return False


def _kernel_at_least(facts: dict, *, major: int, minor: int) -> bool:
    os_ = facts.get("os", {})
    rel = os_.get("release", "") if isinstance(os_, dict) else ""
    parts = rel.split(".")
    try:
        m, n = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return False
    return (m, n) >= (major, minor)


def _has_pn95_disabled(facts: dict) -> bool:
    """True iff PN95 is NOT enabled in the operator's env."""
    env = facts.get("genesis_env", {})
    if isinstance(env, dict):
        return env.get("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "0") != "1"
    return True


def _is_single_24g_gpu(facts: dict) -> bool:
    nv = facts.get("nvidia", {})
    if not isinstance(nv, dict):
        return False
    n = int(nv.get("n_gpus", 0) or 0)
    if n != 1:
        return False
    vrams = nv.get("gpu_total_vram_mib", []) or []
    return any(20_000 <= int(v) <= 26_000 for v in vrams)


# ─── Registry ────────────────────────────────────────────────────────────


KNOWN_CAVEATS: tuple[Caveat, ...] = (
    Caveat(
        id="proxmox_lxc_kernel_617",
        severity="warning",
        title="Proxmox LXC + kernel ≥6.17 known async-IO stall",
        message=(
            "Proxmox LXC containers on kernel 6.17+ have a known async-IO "
            "stall under high concurrency. uvloop in particular triggers "
            "5-10s stalls. Either pin to kernel 6.16 OR run vLLM in a VM "
            "instead of LXC. See feedback_genesis_homelab_inventory.md."
        ),
        match_fn=lambda f: (
            _is_proxmox_lxc(f) and _kernel_at_least(f, major=6, minor=17)
        ),
        docs_url=("https://github.com/Sandermage/genesis-vllm-patches/"
                  "blob/main/docs/RUNTIME_CAVEATS.md#proxmox-lxc-617"),
    ),
    Caveat(
        id="single_3090_long_ctx_vision_no_pn95",
        severity="warning",
        title="Single 24G GPU + long-ctx + vision without PN95 (club-3090 #58)",
        message=(
            "Detected a single 24 GiB GPU (3090/A5000-class). With max_model_len "
            "≥ 145K AND a vision encoder live, you'll OOM after 5-7 chat turns "
            "(club-3090 #58). v7.73.x ships PN95 (tier-aware KV cache + Mamba "
            "SSM exclusion) which solves this for hybrid-GDN models. Set "
            "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1 + declare "
            "cache_config.tiers in the YAML to opt in."
        ),
        match_fn=lambda f: (
            _is_single_24g_gpu(f) and _has_pn95_disabled(f)
        ),
        docs_url=("https://github.com/Sandermage/genesis-vllm-patches/"
                  "blob/main/docs/_internal/research/"
                  "club3090_issue58_long_ctx_vision_oom_2026-05-09.md"),
    ),
    Caveat(
        id="docker_no_nvidia_runtime",
        severity="error",
        title="Docker installed but nvidia-container-toolkit missing",
        message=(
            "Docker daemon is running but the NVIDIA Container Toolkit is "
            "not registered as a runtime. GPUs will not be visible inside "
            "containers. Install `nvidia-container-toolkit` from the NVIDIA "
            "official repo (NOT distro repo, which usually ships an older "
            "version)."
        ),
        match_fn=lambda f: bool(
            f.get("docker", {}).get("installed")
            and f.get("docker", {}).get("daemon_running")
            and not f.get("docker", {}).get("nvidia_runtime_present")
        ),
        docs_url=("https://docs.nvidia.com/datacenter/cloud-native/"
                  "container-toolkit/latest/install-guide.html"),
    ),
    Caveat(
        id="vllm_pin_drift_from_genesis_known_good",
        severity="info",
        title="Running vLLM pin not in KNOWN_GOOD_VLLM_PINS",
        message=(
            "The currently-installed vllm pin is not on the Genesis "
            "KNOWN_GOOD_VLLM_PINS allowlist. Patches may apply cleanly "
            "but anchor drift could surface as runtime SKIPs. "
            "Run `sndr upstream check` for the full picture."
        ),
        match_fn=lambda f: bool(
            f.get("vllm", {}).get("installed")
            and f.get("vllm_pin_in_allowlist", True) is False
        ),
        docs_url=None,
    ),
)


def match_caveats(facts: dict) -> list[Caveat]:
    """Return all caveats whose match_fn fires against `facts`."""
    return [c for c in KNOWN_CAVEATS if c.matches(facts)]


def get_caveat(caveat_id: str) -> Optional[Caveat]:
    """Lookup a caveat by id. Returns None on miss."""
    for c in KNOWN_CAVEATS:
        if c.id == caveat_id:
            return c
    return None


def list_caveat_ids() -> list[str]:
    return [c.id for c in KNOWN_CAVEATS]
