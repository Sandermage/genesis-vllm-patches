# SPDX-License-Identifier: Apache-2.0
"""SNDR Core bundles — atomic multi-patch orchestrators.

A "bundle" is a thin orchestrator that composes 2+ semantically-related
patches into an atomic apply via `MultiFilePatchTransaction`. Bundles
are NEW-functionality additions — they don't replace the individual
patch apply() entry points, they offer an alternative.

Why bundles exist (per Sander Q1/Q4 decision 2026-05-07 + consolidation
deep-dive analysis):

  1. **Atomic state** — when a logical feature spans 2+ patches (e.g.
     qwen3coder tool-parser fixes touch tool_parser.py AND serving.py),
     bundles guarantee either ALL apply or NONE — no partial state.

  2. **Reduced dispatch points** — operators activate ONE umbrella env
     flag (`SNDR_ENABLE_BUNDLE_<NAME>=1`) instead of 4-6 individual
     flags. Less typo surface, simpler model_configs.yaml.

  3. **Per-bundle drift detection (Stage 8 enhancement)** — when
     upstream merges PART of a bundle, only that sub-patch no-ops; the
     siblings continue applying. Today's individual-patch drift is
     all-or-nothing within a single patch's sub_patches list.

Activation:
  Each bundle has an umbrella flag like
  `SNDR_ENABLE_BUNDLE_TOOL_PARSING_QWEN3CODER`. Operator sets it OR
  individual sub-patch flags — both work, neither breaks the other
  (bundle's MultiFilePatchTransaction is idempotent via marker check).

Tier semantics:
  A bundle's tier is the MAX of its components' tiers. If any
  component is `tier="engine"`, the whole bundle is engine-tier and
  requires the `vllm.sndr_engine` commercial package.

Stage 7 catalog:
  - tool_parsing_qwen3coder   — P15 + P61c + P64(×2) + PN56  (community)
  - reasoning_qwen3           — P12 + P27 + P59 + P61 + P61b + PN51  (community)
  - attention_gdn_spec        — P60 + P60b  (community)
  - attention_tq_multi_query  — P67 + P67b  (engine — Sander-original kernel)
  - spec_decode_async_cleanup — P79b + P79c + P79d  (community)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from . import attention_gdn_spec        # noqa: F401
from . import attention_tq_multi_query  # noqa: F401
from . import reasoning_qwen3           # noqa: F401
from . import spec_decode_async_cleanup  # noqa: F401
from . import tool_parsing_qwen3coder   # noqa: F401

__all__ = [
    "attention_gdn_spec",
    "attention_tq_multi_query",
    "reasoning_qwen3",
    "spec_decode_async_cleanup",
    "tool_parsing_qwen3coder",
]
