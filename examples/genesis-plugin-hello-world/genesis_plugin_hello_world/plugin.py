# SPDX-License-Identifier: Apache-2.0
"""Genesis hello-world plugin — minimal reference.

Demonstrates the two halves of the Genesis plugin contract:

  1. `get_patch_metadata()` — entry-point callable, returns a dict
     describing the patch (registry-shape).

  2. `apply()` — actually does the patching when env flag is set.
     Returns (status, reason) tuple, or a bare string treated as
     applied.

Install:
    pip install -e examples/genesis-plugin-hello-world/

Enable (two opt-ins required):
    export GENESIS_ALLOW_PLUGINS=1
    export GENESIS_ENABLE_HELLO_WORLD=1

Verify discovery:
    python3 -m vllm._genesis.compat.cli plugins list

Run:
    # The plugin's apply() is called when env flag is set + Genesis
    # apply_all runs (e.g. via container boot).
"""
from __future__ import annotations


def get_patch_metadata() -> dict:
    """Return the patch metadata dict.

    Genesis enforces:
      - schema validation (catches typos / missing fields)
      - lifecycle force "community" (auto-tagged regardless of input)
      - collision check vs core PATCH_REGISTRY
      - origin stamping (_plugin_origin)

    Returning a list[dict] instead would ship multiple patches per
    package — see the multi-patch docs in PLUGINS.md.
    """
    return {
        "patch_id": "HELLO_WORLD",
        "title": "Hello-world reference plugin (no-op)",
        "env_flag": "GENESIS_ENABLE_HELLO_WORLD",
        "default_on": False,
        "category": "request_middleware",  # arbitrary; pick what fits
        "credit": (
            "Genesis-community reference plugin demonstrating the "
            "entry-point + apply_callable contract. Does nothing useful "
            "in production — it just logs that it ran."
        ),
        "community_credit": "@genesis-community/hello-world v0.1.0",
        # Optional: gate on hardware / model. Empty = applies everywhere.
        "applies_to": {},
        # No deps, no conflicts (no-op plugin)
        "requires_patches": [],
        "conflicts_with": [],
        # The actual code lives in this same module's apply() function.
        "apply_callable": "genesis_plugin_hello_world.plugin:apply",
    }


def apply():
    """Actually apply the patch.

    Return contract:
      - ('applied', '<reason>') — preferred
      - ('skipped', '<reason>') — patch determined it shouldn't apply
      - ('failed',  '<reason>') — tried + hit unrecoverable error
      - bare string             — treated as ('applied', <string>)
      - None                    — treated as applied with generic message
      - any exception           — caught by Genesis, reported as failed

    Genesis isolates failures: if apply() raises, the engine doesn't crash.
    """
    import logging
    log = logging.getLogger("genesis.plugin.hello_world")
    log.info(
        "[hello-world plugin] applied — this is the reference no-op plugin. "
        "Real plugins would patch vllm here (text-patch / setattr / "
        "wrapper / etc)."
    )
    return "applied", (
        "HELLO_WORLD applied — reference plugin executed, no actual "
        "behavior change."
    )
