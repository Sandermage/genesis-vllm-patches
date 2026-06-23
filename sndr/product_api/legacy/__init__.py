# SPDX-License-Identifier: Apache-2.0
"""Genesis product API — pure-Python data layer behind the CLI.

The CLI in :mod:`sndr.cli` is the **operator-facing presentation
layer**. The functions and dataclasses in this package provide the
underlying queries, simulations, and proof-artefact operations as a
side-effect-disciplined Python API that any caller (CLI, tests, SDK
consumers, future GUI / web product layer) can invoke directly.

Design contract:

  * Pure-data inputs / outputs. Returns frozen dataclasses or simple
    dict / list structures. Never prints, never reads ``sys.argv``,
    never short-circuits via ``SystemExit``.
  * Each module mirrors a CLI command surface (e.g. ``product_api.patches.
    listing`` backs ``sndr patches list``).
  * Imports must remain cold-import-safe on hosts without vllm runtime —
    we lazy-import vllm-dependent modules inside function bodies, never
    at module load time.
  * Side effects (filesystem writes, ``os.environ`` overlays) are
    confined to specific named functions and clearly documented.

Phase rollout (per M.6.R design):

  * M.6.1 — read-only query commands (listing, explain, doctor,
    diff-upstream, bundles). NO env mutation, NO artefact writes.
  * M.6.2 — proof / bench-attach / proof-status / release-check
    (artefact writes).
  * M.6.3 — plan (env-overlay) + pn95-status.
  * M.6.4 — drop CLI back-compat shims, finalize boundary.
"""
from __future__ import annotations

__all__: tuple[str, ...] = ()
