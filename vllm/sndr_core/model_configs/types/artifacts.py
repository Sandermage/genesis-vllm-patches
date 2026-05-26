# SPDX-License-Identifier: Apache-2.0
"""ArtifactModel + ArtifactCache + Artifacts + PatchAttribution.

All four were inline classes in ``model_configs/schema.py`` before
M.5.1. Bodies unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ._base import SchemaError


@dataclass
class ArtifactModel:
    """Y3 (UNIFIED_CONFIG plan 2026-05-09): one model artifact spec.

    Declares a HuggingFace-resolvable model + its local path + verify
    rules. Replaces the old `fetch_models.sh` hardcoded paths and the
    legacy `compat.models.pull` registry-tagged lookup with a typed,
    config-owned spec.

    Fields:
      - hf_id: HuggingFace repo ID (e.g. 'Qwen/Qwen3.6-27B-int4-AutoRound').
      - local_dir: absolute or `${var}` path where weights land.
      - revision: HF revision (commit SHA or tag). Defaults to 'main'.
      - gated: True if the repo requires HF token. Drives token-prompt UX.
      - required_files: glob patterns that MUST exist after pull
        (e.g. ['config.json', '*.safetensors']).
      - min_total_gib: minimum total local size to consider 'pulled OK'.
      - notes: free-form operator notes.
    """
    hf_id: str
    local_dir: str
    revision: str = "main"
    gated: bool = False
    required_files: list[str] = field(default_factory=lambda: ["config.json"])
    min_total_gib: float = 0.0
    notes: str = ""

    def validate(self) -> None:
        if not isinstance(self.hf_id, str) or "/" not in self.hf_id:
            raise SchemaError(
                f"ArtifactModel.hf_id must be 'org/repo' (got {self.hf_id!r})"
            )
        if not isinstance(self.local_dir, str) or not self.local_dir.strip():
            raise SchemaError(
                "ArtifactModel.local_dir must be non-empty string"
            )
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise SchemaError(
                "ArtifactModel.revision must be non-empty string"
            )
        if not isinstance(self.required_files, list):
            raise SchemaError(
                "ArtifactModel.required_files must be list[str]"
            )
        if self.min_total_gib < 0:
            raise SchemaError(
                f"ArtifactModel.min_total_gib must be >= 0 "
                f"(got {self.min_total_gib})"
            )

    def verify(self, *, base_path: Optional[str] = None) -> list[str]:
        """Returns a list of human-readable verification problems.

        Empty list = artifact is present + complete on disk.
        `base_path` overrides `${var}` lookup for tests; production
        callers resolve via host.yaml first.
        """
        from pathlib import Path
        problems: list[str] = []
        local = Path(base_path or self.local_dir).expanduser()
        if not local.is_dir():
            return [f"local_dir does not exist: {local}"]
        # Required files (glob match)
        for pattern in self.required_files:
            matches = list(local.rglob(pattern)) if "*" in pattern else (
                [local / pattern] if (local / pattern).exists() else []
            )
            if not matches:
                problems.append(
                    f"required file {pattern!r} not found under {local}"
                )
        # Min total size
        if self.min_total_gib > 0:
            total = sum(
                f.stat().st_size for f in local.rglob("*")
                if f.is_file()
            )
            total_gib = total / (1 << 30)
            if total_gib < self.min_total_gib:
                problems.append(
                    f"local size {total_gib:.2f} GiB < min_total_gib "
                    f"{self.min_total_gib:.2f} GiB"
                )
        return problems


@dataclass
class ArtifactCache:
    """Y3 (UNIFIED_CONFIG plan 2026-05-09): one cache artifact spec.

    Used for `huggingface_hub`, `triton`, `torch_compile`, `safetensors`
    caches. Drives `sndr deps plan` to know which on-disk caches the
    config expects + lets the launcher mount them when running in
    container mode.

    Fields:
      - kind: 'huggingface_hub' | 'triton' | 'torch_compile' | 'safetensors'
              | 'compile_cache' | 'other'
      - path: absolute or ${var} path to the cache directory.
      - persistent: True if the cache should survive container restarts
        (mount as named volume / host path), False for ephemeral.
      - notes: free-form.
    """
    kind: str
    path: str
    persistent: bool = True
    notes: str = ""

    _VALID_KINDS = (
        "huggingface_hub", "triton", "torch_compile", "compile_cache",
        "safetensors", "other",
    )

    def validate(self) -> None:
        if self.kind not in self._VALID_KINDS:
            raise SchemaError(
                f"ArtifactCache.kind must be one of {self._VALID_KINDS} "
                f"(got {self.kind!r})"
            )
        if not isinstance(self.path, str) or not self.path.strip():
            raise SchemaError("ArtifactCache.path must be non-empty string")


@dataclass
class Artifacts:
    """Y3 (UNIFIED_CONFIG plan 2026-05-09): container for model + cache specs.

    Top-level holder so YAML can express both lists in one block:

        artifacts:
          models:
            - hf_id: Qwen/Qwen3.6-27B-int4-AutoRound
              local_dir: /models/Qwen3.6-27B-int4-AutoRound
              required_files: [config.json, "*.safetensors"]
              min_total_gib: 14.0
          caches:
            - kind: huggingface_hub
              path: ~/.cache/huggingface
              persistent: true
            - kind: triton
              path: ${cache_root}/triton-cache-v11
    """
    models: list[ArtifactModel] = field(default_factory=list)
    caches: list[ArtifactCache] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.models, list):
            raise SchemaError("Artifacts.models must be list[ArtifactModel]")
        if not isinstance(self.caches, list):
            raise SchemaError("Artifacts.caches must be list[ArtifactCache]")
        for m in self.models:
            m.validate()
        for c in self.caches:
            c.validate()


# ─── PatchAttribution — structured rationale for ModelDef.patches ─────
#
# Phase A (2026-05-16) introduced patches_attribution on V2 ModelDef.
# Phase B lifted the dataclass into V1 schema so V1 ModelConfig can
# also carry it through compose() into the runtime pipeline
# (patch_plan resolver, sndr patches plan --explain, sndr compose
# render --policy). M.5.1 (2026-05-27) relocated the dataclass into
# this module; schema_v2.py and schema.py both re-export it for the
# pre-M.5.1 import paths.

_PATCH_ROLES: tuple[str, ...] = (
    "load_bearing",
    "defensive",
    "optional_perf",
    "suspected_regression",
    "no_op",
    "unknown",
)


@dataclass
class PatchAttribution:
    """Why a patch is in the model's canonical set.

    Keyed by registry patch ID (e.g. ``PN204``), not by env-flag name.
    Stored on both ModelDef (authoring) and V1 ModelConfig (after compose)
    so the resolver in `patch_plan.py` can run on the same object the
    rest of the runtime pipeline already consumes.

    Role-conditional required fields (validated at .validate() time):

      load_bearing | suspected_regression  →  ``note`` required
      optional_perf                        →  ``bench_evidence`` required
      defensive | no_op | unknown          →  no auxiliary fields required
    """
    role: str
    note: Optional[str] = None
    bench_evidence: Optional[str] = None
    candidate_when: Optional[dict[str, Any]] = None

    def validate(self, *, key: str) -> None:
        if self.role not in _PATCH_ROLES:
            raise SchemaError(
                f"patches_attribution[{key!r}].role={self.role!r} not in "
                f"{_PATCH_ROLES}"
            )
        if self.role in ("load_bearing", "suspected_regression") and not self.note:
            raise SchemaError(
                f"patches_attribution[{key!r}]: role={self.role!r} requires "
                f"a non-empty `note` (reviewers need to see the rationale)"
            )
        if self.role == "optional_perf" and not self.bench_evidence:
            raise SchemaError(
                f"patches_attribution[{key!r}]: role='optional_perf' requires "
                f"a non-empty `bench_evidence` (perf claims need an anchor)"
            )
        if self.candidate_when is not None and not isinstance(self.candidate_when, dict):
            raise SchemaError(
                f"patches_attribution[{key!r}].candidate_when must be dict | None "
                f"(got {type(self.candidate_when).__name__})"
            )
