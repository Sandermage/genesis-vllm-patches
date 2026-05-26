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
    """S2.5 декларативное правило совместимости.

    Зачем
    -----
    Раньше известные несовместимости были разбросаны по `validate()` и
    `audit()` методам ModelConfig'а. Это работает, но новые operator'ы не
    могут одним взглядом увидеть, "какие комбинации опций безопасны".
    `CompatibilityMatrix` собирает все правила в одном месте, а UI/CLI
    может рендерить их в виде таблицы.

    Семантика
    ---------
    Каждое правило содержит:

      • `id` — стабильный идентификатор (`COMPAT-XXX`).
      • `severity` — `"forbidden"` (hard error в validate()) или
        `"discouraged"` (soft warning в audit()).
      • `predicate(cfg) → bool` — True если конфиг попадает под правило.
      • `message` — человекочитаемое объяснение, что не так и почему.
      • `mitigation` — что сделать, чтобы стало корректно.
      • `references` — docs / issue links для дополнительного контекста.

    Не дублирует существующие inline checks — добавляет НОВЫЕ декларации
    и предоставляет агрегатный view для CLI.
    """
    id: str
    severity: str  # "forbidden" | "discouraged"
    title: str
    message: str
    mitigation: str
    references: list[str] = field(default_factory=list)
    # predicate хранится не в dataclass поле (нельзя сериализовать в YAML);
    # его регистрирует `CompatibilityMatrix` рядом с метадатой.

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
    """S2.5 — registry известных правил совместимости + predicate'ов.

    Использование
    -------------

      from vllm.sndr_core.model_configs.schema import COMPATIBILITY_MATRIX
      forbidden, discouraged = COMPATIBILITY_MATRIX.evaluate(cfg)
      for rule, _msg in forbidden:
          # hard error
      for rule, _msg in discouraged:
          # soft warning

    Rules добавляются через `register(rule, predicate)`. Predicate
    получает целиком ModelConfig и возвращает True если правило сработало.

    Иммутабельность: предполагается единственный экземпляр модуля
    (`COMPATIBILITY_MATRIX`) с фиксированным набором правил, известным
    на момент загрузки. Тесты могут создавать собственные instance для
    изоляции (см. `test_compatibility_matrix.py`).
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
        """Все зарегистрированные правила (для CLI rendering)."""
        return [r for r, _ in self._rules]

    def evaluate(
        self, cfg: "ModelConfig",
    ) -> tuple[list[tuple[CompatibilityRule, str]],
               list[tuple[CompatibilityRule, str]]]:
        """Прогоняет все predicate'ы по cfg.

        Returns (forbidden_violations, discouraged_violations) — каждый
        элемент `(rule, human_message)`. Caller сам решает escalation.
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
                # Predicate exception не должен ронять validate всего конфига —
                # operator увидит warning в логе и сможет починить правило.
                log.warning(
                    "CompatibilityMatrix rule %s predicate raised %r — "
                    "treating as not-applicable",
                    rule.id, exc,
                )
        return forbidden, discouraged


# ──── Predicate helpers (общие проверки для правил) ────────────────────

def _uses_hybrid_gdn(cfg: "ModelConfig") -> bool:
    """Hybrid GDN признак — PN59 streaming-GDN env установлен."""
    return cfg.genesis_env.get("GENESIS_ENABLE_PN59_STREAMING_GDN") == "1"


def _spec_decode_method(cfg: "ModelConfig") -> Optional[str]:
    return cfg.spec_decode.method if cfg.spec_decode else None


def _kv_cache_dtype(cfg: "ModelConfig") -> Optional[str]:
    return cfg.kv_cache_dtype


# ──── Сами правила ─────────────────────────────────────────────────────

_COMPAT_DFLASH_ON_QWEN_NEXT = CompatibilityRule(
    id="COMPAT-001",
    severity="forbidden",
    title="DFlash speculative decode на Qwen-next архитектуре",
    message=(
        "spec_decode.method='dflash' заблокирован для Qwen-next "
        "архитектуры (upstream Qwen3-next): MTP head Qwen-next модели "
        "fused в main model особым образом, который мешает external "
        "drafter speculation. См. audit P2-2 + vllm#42102 для деталей. "
        "Для других hybrid-GDN моделей (Qwen3.6-27B Lorbus etc.) "
        "DFlash работает с отдельным drafter checkpoint."
    ),
    mitigation=(
        "Используйте method='mtp' (Qwen-next's own MTP head — "
        "intended path) или 'ngram'. Если DFlash обязателен — "
        "переключите на model_path с dense-transformer (Qwen3.6-35B-"
        "A3B-FP8) или Qwen3.6 hybrid (27B Lorbus с separate drafter)."
    ),
    references=["docs/PATCHES.md#PN59", "vllm-project/vllm#42102"],
)


_COMPAT_TQK8V4_ON_HYBRID_GDN_NO_P98 = CompatibilityRule(
    id="COMPAT-002",
    severity="discouraged",
    title="TurboQuant k8v4 на hybrid-GDN без P98 lock",
    message=(
        "kv_cache_dtype='turboquant_k8v4' + hybrid-GDN модель без "
        "явного включения P98 (vs vllm#40941 lock) может выдать "
        "non-deterministic prefill в long-context. P98 закрывает race "
        "condition в quantized KV write path."
    ),
    mitigation=(
        "Добавьте `GENESIS_ENABLE_P98=1` в genesis_env "
        "ИЛИ снимите turboquant_k8v4 для hybrid-GDN configs."
    ),
    references=[
        "docs/PATCHES.md#P98",
        "docs/_internal/research/club3090_issue58_long_ctx_vision_oom_2026-05-09.md",
    ],
)


_COMPAT_NGRAM_ON_TQK8V4_LONG_CTX = CompatibilityRule(
    id="COMPAT-003",
    severity="discouraged",
    title="N-gram spec_decode на TQ k8v4 long-context",
    message=(
        "spec_decode.method='ngram' + kv_cache_dtype='turboquant_k8v4' "
        "+ max_model_len > 131072 показал в стресс-тестах падение "
        "acceptance rate с 0.62 до 0.41 после ~10K tokens (cache "
        "thrashing). Для long-context используйте MTP — он не зависит "
        "от prefix cache."
    ),
    mitigation=(
        "Замените method='ngram' на 'mtp' для max_model_len > 131072. "
        "Если ngram необходим (workload без MTP head), уменьшите "
        "max_model_len ≤ 131072."
    ),
    references=["docs/COOKBOOK.md#ngram-vs-mtp"],
)


_COMPAT_DFLASH_REQUIRES_DRAFTER_PATH = CompatibilityRule(
    id="COMPAT-004",
    severity="forbidden",
    title="DFlash без указания drafter model",
    message=(
        "spec_decode.method='dflash' требует отдельный drafter "
        "checkpoint (поле `model`). Без него vllm падает при инициа­"
        "лизации speculative decoder. Это дублирует SpecDecodeConfig."
        "validate() но проверяется и в matrix для глобальной видимости."
    ),
    mitigation=(
        "Укажите `spec_decode.model: /path/to/dflash-drafter` ИЛИ "
        "смените метод на 'mtp' (использует MTP head самой модели)."
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
