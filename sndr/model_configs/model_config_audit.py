# SPDX-License-Identifier: Apache-2.0
"""``audit_model_config(cfg)`` — soft warnings layer.

M.5.3 (2026-05-27): extracted from ``ModelConfig.audit()`` in
``model_configs/schema.py``. The function body is byte-identical to
the pre-refactor method; ``ModelConfig.audit()`` is now a thin
one-line delegation that returns ``audit_model_config(self)``.

Soft warnings cover *risky-but-not-invalid* configurations the
operator can choose to ignore:

  * TQ k8v4 + hybrid GDN model without P98 (vllm#40941 race).
  * ``stable`` lifecycle without reference_metrics — ``verify`` can't
    run without a baseline.
  * CompatibilityMatrix ``discouraged`` rules (severity ≠ ``forbidden``).

Hard validation errors live in :meth:`ModelConfig.validate`; this
module is the operator-visible companion that ``sndr model-config
score`` / dashboards consume.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import COMPATIBILITY_MATRIX

if TYPE_CHECKING:
    from .schema import ModelConfig


def audit_model_config(cfg: "ModelConfig") -> list[str]:
    """Return the list of soft-warning strings for ``cfg``.

    Empty list means the config is clean of advisory issues. The
    operator-facing message wording is preserved byte-for-byte from
    the pre-M.5.3 method so downstream snapshots / dashboards don't
    drift.
    """
    warnings: list[str] = []
    # TQ k8v4 + hybrid GDN model needs P98 (vs vllm#40941 lock).
    # Hybrid GDN models: 27B Lorbus int4, NOT 35B-A3B-FP8 (dense MoE).
    # Detection: PN59_STREAMING_GDN=1 in env is the canonical signal —
    # operator only enables PN59 on hybrid models.
    if cfg.kv_cache_dtype == "turboquant_k8v4":
        pn59_on = cfg.genesis_env.get(
            "GENESIS_ENABLE_PN59_STREAMING_GDN") == "1"
        int4_lorbus = "int4" in cfg.model_path.lower() and \
            "AutoRound" in cfg.model_path
        if (pn59_on or int4_lorbus) and \
                "GENESIS_ENABLE_P98" not in cfg.genesis_env:
            warnings.append(
                "P98 should be enabled for TQ k8v4 + hybrid GDN model "
                "(WorkspaceManager fix vs vllm#40941). "
                "Add GENESIS_ENABLE_P98=1 to genesis_env."
            )
    # Reference metrics expected for stable lifecycle
    if cfg.lifecycle == "stable" and cfg.reference_metrics is None:
        warnings.append(
            "stable lifecycle should have reference_metrics — "
            "operators can't run `verify` without baseline values."
        )
    # S2.5 (2026-05-12): CompatibilityMatrix discouraged rules.
    _, discouraged = COMPATIBILITY_MATRIX.evaluate(cfg)
    for rule, msg in discouraged:
        warnings.append(f"[{rule.id}] {rule.title}: {msg}")
    return warnings
