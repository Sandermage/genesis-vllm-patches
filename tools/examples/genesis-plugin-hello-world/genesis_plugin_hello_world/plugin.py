# SPDX-License-Identifier: Apache-2.0
"""Reference Genesis plugin — `get_patch_metadata` + `apply` contract.

This is the canonical reference for how a third-party plugin ships
a patch through the Genesis entry-point system. The patch itself
is a no-op (returns "applied" without touching anything) so the
example focuses purely on the metadata + plumbing contract.

Mirror this structure when building your own plugin. The Genesis
schema validator + plugin discovery pipeline test this file in CI
(`tests/compat/test_plugin_example.py`); if the contract drifts,
those tests catch it before docs become misleading.
"""
from __future__ import annotations


def get_patch_metadata() -> dict:
    """Return the metadata dict that Genesis registers in PATCH_REGISTRY.

    Required fields:
      - patch_id           operator-facing identifier (UPPER_SNAKE_CASE)
      - title              one-line description shown in `genesis explain`
      - env_flag           must start with "GENESIS_"
      - default_on         must be False (good citizenship — opt-in)
      - community_credit   GitHub handle / project name for attribution

    Optional fields used by Genesis (see `docs/PLUGINS.md`):
      - lifecycle          auto-tagged "community" — your value is ignored
      - category           e.g. "spec_decode", "kv_cache", "kernel"
      - credit             text describing what the patch fixes / why
      - applies_to         hardware / model gate (predicate DSL)
      - requires_patches   list of patch_ids this patch depends on
      - conflicts_with     list of patch_ids this patch is incompatible with
      - apply_callable     "module:func" string for Genesis to invoke
                            (defaults to apply() in this module if absent)
    """
    return {
        "patch_id": "HELLO_WORLD",
        "title": "Reference Genesis community plugin (no-op example)",
        "env_flag": "GENESIS_ENABLE_HELLO_WORLD",
        "default_on": False,
        "category": "example",
        "credit": (
            "Documentation example for `docs/PLUGINS.md`. Demonstrates the "
            "minimum metadata + apply contract. No runtime side effects."
        ),
        "community_credit": "@Sandermage / Genesis docs team",
        "apply_callable": "genesis_plugin_hello_world.plugin:apply",
    }


def apply() -> tuple[str, str]:
    """Apply the patch.

    Returns:
        (status, reason) where status is one of:
          - "applied"  — patch took effect
          - "skipped"  — gracefully no-op (e.g. wrong hardware, env disabled)
          - "failed"   — error encountered (operator must investigate)

        `reason` is a short human-readable string that surfaces in boot logs.

    The reference example always returns "applied" with a no-op reason so
    operators can verify the discovery + invocation pipeline end-to-end
    without any production-side risk.
    """
    return "applied", "no-op reference example"
