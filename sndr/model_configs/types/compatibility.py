# SPDX-License-Identifier: Apache-2.0
"""CompatibilityRule + CompatibilityMatrix + predicate helpers + rule registry.

Relocated from ``model_configs/schema.py`` in M.5.1. The 4 rule
definitions (``COMPAT-001`` … ``COMPAT-004``) register themselves into
the module-level :data:`COMPATIBILITY_MATRIX` singleton at import time —
same observable behaviour as the pre-refactor module.

The ``cfg: "ModelConfig"`` forward references keep the import cycle
broken: this module never imports ``ModelConfig`` directly; the
attribute access is resolved lazily at predicate-call time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ._base import SchemaError


# Preserve the historical logger name so any operator filter rules in
# ``logging`` config keep matching after the relocation.
log = logging.getLogger("genesis.model_configs.schema")


@dataclass
class CompatibilityRule:
    """S2.5 declarative compatibility rule.

    Rationale
    ---------
    Previously known incompatibilities were scattered across
    ``ModelConfig.validate()`` and ``audit()`` methods. That works,
    but new operators cannot see at a glance "which option combinations
    are safe". ``CompatibilityMatrix`` collects every rule in one place
    so the CLI / UI can render them as a single table.

    Semantics
    ---------
    Each rule contains:

      • ``id`` — stable identifier (``COMPAT-XXX``).
      • ``severity`` — ``"forbidden"`` (hard error inside validate())
        or ``"discouraged"`` (soft warning inside audit()).
      • ``predicate(cfg) -> bool`` — True if the config falls under
        the rule.
      • ``message`` — human-readable explanation of what is wrong
        and why.
      • ``mitigation`` — what to do to make the config correct.
      • ``references`` — docs / issue links for additional context.

    Does not duplicate existing inline checks — registers NEW
    declarations and supplies an aggregate view for the CLI.
    """
    id: str
    severity: str  # "forbidden" | "discouraged"
    title: str
    message: str
    mitigation: str
    references: list[str] = field(default_factory=list)
    # The predicate is intentionally NOT stored on the dataclass
    # (it cannot be serialised to YAML); ``CompatibilityMatrix``
    # registers it alongside the metadata instead.

    def validate(self) -> None:
        if not self.id:
            raise SchemaError("CompatibilityRule.id required")
        if self.severity not in ("forbidden", "discouraged"):
            raise SchemaError(
                "CompatibilityRule.severity must be 'forbidden' or "
                f"'discouraged' (got '{self.severity}')"
            )
        if not self.title or not self.message or not self.mitigation:
            raise SchemaError(
                "CompatibilityRule requires title, message, mitigation"
            )


class CompatibilityMatrix:
    """S2.5 — registry of known compatibility rules + predicates.

    Usage
    -----

      from vllm.sndr_core.model_configs.schema import COMPATIBILITY_MATRIX
      forbidden, discouraged = COMPATIBILITY_MATRIX.evaluate(cfg)
      for rule, _msg in forbidden:
          # hard error
      for rule, _msg in discouraged:
          # soft warning

    Rules are registered through ``register(rule, predicate)``. The
    predicate receives the whole ModelConfig and returns True when
    the rule applies.

    Immutability: the assumption is one module-level instance
    (``COMPATIBILITY_MATRIX``) with a fixed rule set, known at import
    time. Tests can construct their own instance to isolate state
    (see ``test_compatibility_matrix.py``).
    """

    def __init__(self) -> None:
        self._rules: list[tuple[CompatibilityRule, Any]] = []

    def register(self, rule: CompatibilityRule, predicate) -> None:
        rule.validate()
        if any(r.id == rule.id for r, _ in self._rules):
            raise SchemaError(
                f"CompatibilityMatrix: duplicate rule id '{rule.id}'"
            )
        self._rules.append((rule, predicate))

    def rules(self) -> list[CompatibilityRule]:
        """All registered rules (for CLI rendering)."""
        return [r for r, _ in self._rules]

    def evaluate(
        self, cfg: "ModelConfig",
    ) -> tuple[list[tuple[CompatibilityRule, str]],
               list[tuple[CompatibilityRule, str]]]:
        """Run all predicates against cfg.

        Returns (forbidden_violations, discouraged_violations) — each
        element ``(rule, human_message)``. Caller decides escalation.
        """
        forbidden: list[tuple[CompatibilityRule, str]] = []
        discouraged: list[tuple[CompatibilityRule, str]] = []
        for rule, pred in self._rules:
            try:
                if pred(cfg):
                    bucket = (forbidden if rule.severity == "forbidden"
                              else discouraged)
                    bucket.append((rule, rule.message))
            except Exception as exc:
                # A predicate exception must not bring down the whole
                # validate() — operator sees a warning in the log and
                # can fix the offending rule.
                log.warning(
                    "CompatibilityMatrix rule %s predicate raised %r — "
                    "treating as not-applicable",
                    rule.id, exc,
                )
        return forbidden, discouraged


# ──── Predicate helpers (shared checks for rules) ──────────────────────

def _uses_hybrid_gdn(cfg: "ModelConfig") -> bool:
    """Hybrid GDN indicator — PN59 streaming-GDN env flag is set."""
    return cfg.genesis_env.get("GENESIS_ENABLE_PN59_STREAMING_GDN") == "1"


def _spec_decode_method(cfg: "ModelConfig") -> Optional[str]:
    return cfg.spec_decode.method if cfg.spec_decode else None


def _kv_cache_dtype(cfg: "ModelConfig") -> Optional[str]:
    return cfg.kv_cache_dtype


# ──── Rule declarations ────────────────────────────────────────────────

_COMPAT_DFLASH_ON_QWEN_NEXT = CompatibilityRule(
    id="COMPAT-001",
    severity="forbidden",
    title="DFlash speculative decode on Qwen-next architecture",
    message=(
        "spec_decode.method='dflash' is blocked on the Qwen-next "
        "architecture (upstream Qwen3-next): the MTP head of Qwen-next "
        "models is fused into the main model in a way that prevents "
        "external drafter speculation. See audit P2-2 + vllm#42102 for "
        "details. On other hybrid-GDN models (Qwen3.6-27B Lorbus etc.) "
        "DFlash works with a separate drafter checkpoint."
    ),
    mitigation=(
        "Use method='mtp' (Qwen-next's own MTP head — the intended "
        "path) or 'ngram'. If DFlash is required, switch model_path to "
        "a dense transformer (Qwen3.6-35B-A3B-FP8) or Qwen3.6 hybrid "
        "(27B Lorbus with a separate drafter)."
    ),
    references=["docs/PATCHES.md#PN59", "vllm-project/vllm#42102"],
)


_COMPAT_TQK8V4_ON_HYBRID_GDN_NO_P98 = CompatibilityRule(
    id="COMPAT-002",
    severity="discouraged",
    title="TurboQuant k8v4 on hybrid-GDN without P98 lock",
    message=(
        "kv_cache_dtype='turboquant_k8v4' on a hybrid-GDN model "
        "without explicit P98 (vs vllm#40941 lock) can produce "
        "non-deterministic prefill in long-context. P98 closes a "
        "race condition in the quantised KV write path."
    ),
    mitigation=(
        "Add `GENESIS_ENABLE_P98=1` to genesis_env, OR drop "
        "turboquant_k8v4 for hybrid-GDN configs."
    ),
    references=[
        "docs/PATCHES.md#P98",
        "docs/_internal/research/club3090_issue58_long_ctx_vision_oom_2026-05-09.md",
    ],
)


_COMPAT_NGRAM_ON_TQK8V4_LONG_CTX = CompatibilityRule(
    id="COMPAT-003",
    severity="discouraged",
    title="N-gram spec_decode on TQ k8v4 long-context",
    message=(
        "spec_decode.method='ngram' + kv_cache_dtype='turboquant_k8v4' "
        "+ max_model_len > 131072 was observed in stress tests to drop "
        "acceptance rate from 0.62 to 0.41 after ~10K tokens (cache "
        "thrashing). For long-context, use MTP — it does not depend "
        "on prefix cache."
    ),
    mitigation=(
        "Replace method='ngram' with 'mtp' for max_model_len > 131072. "
        "If ngram is required (workload without an MTP head), reduce "
        "max_model_len to <= 131072."
    ),
    references=["docs/COOKBOOK.md#ngram-vs-mtp"],
)


_COMPAT_DFLASH_REQUIRES_DRAFTER_PATH = CompatibilityRule(
    id="COMPAT-004",
    severity="forbidden",
    title="DFlash without a drafter model",
    message=(
        "spec_decode.method='dflash' requires a separate drafter "
        "checkpoint (the `model` field). Without it vllm fails during "
        "speculative-decoder initialisation. This duplicates "
        "SpecDecodeConfig.validate() but is also checked in the "
        "matrix for global visibility."
    ),
    mitigation=(
        "Set `spec_decode.model: /path/to/dflash-drafter` OR switch "
        "method to 'mtp' (uses the model's own MTP head)."
    ),
    references=["docs/PATCHES.md#dflash"],
)


COMPATIBILITY_MATRIX = CompatibilityMatrix()


def _is_qwen_next(cfg: "ModelConfig") -> bool:
    """Detect Qwen-next architecture by model_path substring.

    Qwen-next (upstream Qwen3-next) — distinct from Qwen3.6 hybrid
    Mamba (Lorbus). Detected purely by path naming convention.
    """
    p = (cfg.model_path or "").lower()
    return "qwen-next" in p or "qwen3-next" in p


COMPATIBILITY_MATRIX.register(
    _COMPAT_DFLASH_ON_QWEN_NEXT,
    lambda c: _spec_decode_method(c) == "dflash" and _is_qwen_next(c),
)
COMPATIBILITY_MATRIX.register(
    _COMPAT_TQK8V4_ON_HYBRID_GDN_NO_P98,
    lambda c: (
        _kv_cache_dtype(c) == "turboquant_k8v4"
        and _uses_hybrid_gdn(c)
        and c.genesis_env.get("GENESIS_ENABLE_P98") != "1"
    ),
)
COMPATIBILITY_MATRIX.register(
    _COMPAT_NGRAM_ON_TQK8V4_LONG_CTX,
    lambda c: (
        _spec_decode_method(c) == "ngram"
        and _kv_cache_dtype(c) == "turboquant_k8v4"
        and c.max_model_len > 131072
    ),
)
COMPATIBILITY_MATRIX.register(
    _COMPAT_DFLASH_REQUIRES_DRAFTER_PATH,
    lambda c: (
        _spec_decode_method(c) == "dflash"
        and c.spec_decode is not None
        and not c.spec_decode.model
    ),
)
