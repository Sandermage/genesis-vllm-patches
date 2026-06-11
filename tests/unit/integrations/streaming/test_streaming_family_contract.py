# SPDX-License-Identifier: Apache-2.0
"""streaming family contract — closes the last gap in the per-family
contract coverage (Phase 10.5 D-extension 2026-06-01).

Background
----------

The streaming family hosts four patches that route the chunked-prefill
+ KV-pool + cold-prefix-offload runtime path. Every other top-level
integration family directory under ``vllm/sndr_core/integrations/``
had an associated ``test_<family>_family_contract.py`` pinning the
six per-patch invariants (importable / Genesis marker / apply()
callable / env_flag documented / no top-level torch import / family
field matches the dir) plus the two family-level checks (registry
has every entry on the list + filesystem has every entry from the
registry). Streaming was the lone exception until now.

The 2026-06-01 enterprise audit (commit 9406781f's
Makefile+CONTRIBUTING refresh) surfaced this gap — the "covers 19 of
20 integration family dirs (streaming is the only family without a
contract yet)" note was the audit's record. This file closes that
note. The Makefile / CONTRIBUTING comment can drop the "19 of 20"
qualifier on the next refresh.

Patches enumerated
------------------

  * PN200 — GDN scratch reuse
  * PN201 — scheduler empty-cache wrapper
  * PN202 — per-layer KV split
  * PN203 — cold prefix offload

Each entry maps the canonical apply_module dotted path (registry
``apply_module`` field) to the registry patch_id. The factory
verifies both directions: every patch_id on the list must be in
PATCH_REGISTRY AND the filesystem (``integrations/streaming/``) must
expose every entry on the list. Adding a new ``pn204_*.py`` to the
family dir without adding it here breaks the registry side; removing
an entry here without retiring the patch breaks the filesystem side.
"""
from tests.unit.integrations._family_contract_helpers import (
    make_family_contract_class,
    make_family_registry_class,
)


PATCHES = [
    # PN200 retired 2026-06-11 (superseded by P28 — same buffer-reuse+zero
    # on the unique forward_cuda site); module archived.
    ("sndr.engines.vllm._archive.pn200_gdn_scratch_reuse", "PN200"),
    ("sndr.engines.vllm.patches.streaming.pn201_scheduler_empty_cache", "PN201"),
    ("sndr.engines.vllm.patches.streaming.pn202_per_layer_kv_split", "PN202"),
    ("sndr.engines.vllm.patches.streaming.pn203_cold_prefix_offload", "PN203"),
]


class TestStreamingPatchContract(
    make_family_contract_class("streaming", PATCHES)
):
    """Six-invariant per-patch contract for every streaming patch."""
    pass


class TestStreamingFamilyRegistry(
    make_family_registry_class("streaming", PATCHES)
):
    """Bidirectional list ↔ registry ↔ filesystem coverage check."""
    pass
