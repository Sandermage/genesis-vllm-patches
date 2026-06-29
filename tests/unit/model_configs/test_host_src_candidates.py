# SPDX-License-Identifier: Apache-2.0
"""Guard: host.py default mount-source candidates must track the v12
vllm/sndr_core/ -> sndr/ package rename.

Regression for the v12 bug (2026-06-23): _DEFAULT_SNDR_SRC_CANDIDATES still
pointed at the retired .../genesis-vllm-patches/vllm/sndr_core, so the
auto-probed read-only bind-mount source never existed on a clean v12 checkout
and `import sndr` failed in the serving container unless the operator set
sndr_src / SNDR_CORE_SRC by hand. The UNIFIED ROOT BUG fix had retargeted
_DEFAULT_PLUGIN_SRC_CANDIDATES (the repo root) but missed this sibling list.
"""
from __future__ import annotations

from sndr.model_configs import host
from sndr.model_configs.host import (
    _DEFAULT_PLUGIN_SRC_CANDIDATES,
    _DEFAULT_SNDR_SRC_CANDIDATES,
)


def test_no_candidate_references_retired_namespace():
    stale = [c for c in (_DEFAULT_SNDR_SRC_CANDIDATES + _DEFAULT_PLUGIN_SRC_CANDIDATES)
             if "vllm/sndr_core" in c]
    assert not stale, f"retired v11 namespace in mount candidates: {stale}"


def test_sndr_src_candidates_point_at_the_sndr_package_dir():
    # sndr_src is RO-mounted into the container's dist-packages/sndr, so each
    # candidate must end at the `sndr` package dir, not the repo root.
    bad = [c for c in _DEFAULT_SNDR_SRC_CANDIDATES
           if not c.rstrip("/").endswith("/sndr")]
    assert not bad, f"sndr_src candidates must end at the sndr/ package dir: {bad}"


def test_plugin_src_candidates_point_at_the_repo_root():
    # plugin_src is pip-installed editable so the root pyproject's
    # vllm.general_plugins entry-point registers — it must be the repo root,
    # not the sndr/ subdir.
    bad = [c for c in _DEFAULT_PLUGIN_SRC_CANDIDATES
           if c.rstrip("/").endswith("/sndr")]
    assert not bad, f"plugin_src candidates must be the repo root, not sndr/: {bad}"


def test_detect_paths_skips_permission_denied_candidate(monkeypatch):
    # Regression (CI 2026-06-29): a default models candidate like /data/models
    # exists but is permission-denied on a sandboxed CI runner. pathlib's
    # Path.is_dir() PROPAGATES PermissionError (only missing/broken-symlink
    # return False), so the probe crashed instead of skipping the candidate.
    # An inaccessible candidate must be treated as "not usable" and skipped.
    denied = "/denied/models"

    def fake_is_dir(self):
        if str(self) == denied:
            raise PermissionError(13, "Permission denied")
        return False  # nothing else "exists" in this isolated probe

    monkeypatch.setattr(host.Path, "is_dir", fake_is_dir)

    # Must NOT raise; the inaccessible candidate is simply omitted. Empty lists
    # for the other vars isolate the probe to the single denied candidate.
    out = host.detect_paths(
        models_candidates=[denied],
        hf_cache_candidates=[],
        triton_cache_candidates=[],
        compile_cache_candidates=[],
        sndr_src_candidates=[],
        plugin_src_candidates=[],
        cache_root_candidates=[],
    )
    assert "models_dir" not in out
