# SPDX-License-Identifier: Apache-2.0
"""Genesis sndr_core — ``model_compat`` technical-area namespace.

Real runtime patches whose primary purpose is making a specific
upstream model family work end-to-end on Genesis-supported hardware
live here, grouped by model family. Each sub-package is itself a small
namespace and does not re-export — concrete wiring lives in
``dispatcher/registry.py`` ``apply_module`` fields.

Current members:
  * ``gemma4/`` — Gemma 4 family (refusal guards, vendor backports,
    deep fixes, perf kernels, compatibility, vision-tower, RoPE
    diagnostics).

Why a separate ``model_compat/`` bucket exists:
  Most Genesis patches live in technical-area buckets
  (``attention/turboquant``, ``spec_decode``, ``kv_cache``, ``moe``,
  ``compile_safety`` etc.) because the patch's mechanism is reusable
  across models. A small minority of patches are genuinely
  model-specific — they would not apply verbatim to a different
  checkpoint that uses the same technical mechanism. Those live here
  under their family. The architectural invariant is documented in
  ``sndr_private/planning/audits/RELOCATION_DESIGN_2026-05-21_RU.md``
  §0.5 Rule 1.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
