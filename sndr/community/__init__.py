# SPDX-License-Identifier: Apache-2.0
"""Community patch SDK (Phase 5, Roadmap V2).

The community SDK lets outside contributors author patches as self-contained
plugins under `plugins/community/<author>/<patch-id>/`. Each plugin ships:

  manifest.yaml      — `PatchManifest` (schema_v2)
  patch.py           — entry-point implementing the patch's `apply` hook
  tests/             — pytest discovery harness

This package exposes four reusable building blocks:

  manifest.py    — load + validate a single manifest
  discovery.py   — walk `plugins/community/**` AND `vllm.community_patches`
                   entry-point group to enumerate manifests
  validator.py   — strict release-tier validation (schema + anchor md5 +
                   conflict detection + tests-required harness)
  scaffold.py    — `sndr community new-patch` generator that drops a
                   working draft plugin tree from a template

Phase 5 contract: validator catches every bad manifest shape. CLI surface
(`sndr community list / validate / scaffold`) ships in cli/community.py.
"""
from __future__ import annotations

from .manifest import load_manifest, list_manifest_paths
from .discovery import discover_all, discover_filesystem, discover_entry_points
from .validator import (
    ValidationIssue,
    ValidationResult,
    validate_manifest,
    validate_directory,
)


__all__ = [
    "load_manifest",
    "list_manifest_paths",
    "discover_all",
    "discover_filesystem",
    "discover_entry_points",
    "ValidationIssue",
    "ValidationResult",
    "validate_manifest",
    "validate_directory",
]
