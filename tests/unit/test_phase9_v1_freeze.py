# SPDX-License-Identifier: Apache-2.0
"""Phase 9 acceptance — V1 freeze.

Contract:

  1. `registry.get(<V1-key>)` emits a `DeprecationWarning` once per key.
  2. The warning carries the V2 migration hint (mentions `sndr hardware list`
     / `sndr model list-v2`).
  3. `GENESIS_DISABLE_V1_DEPRECATION_WARNING=1` silences the warning.
  4. The V1 path STILL WORKS (Phase 9 = warn-only freeze; Phase 10 sunsets).
  5. V2 alias load through `registry_v2.load_alias` does NOT emit any
     V1 deprecation warning (composed configs are V2, not V1).
  6. `audit-no-new-v1` gate passes on the current tree (matches the
     11-entry frozen baseline).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _import_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _reset_warned_set():
    """Clear the one-warning-per-key cache before each test so warning
    behaviour is deterministic regardless of test order."""
    from vllm.sndr_core.model_configs import registry
    registry._V1_DEPRECATION_WARNED.clear()
    # Also clear env override.
    saved = os.environ.pop("GENESIS_DISABLE_V1_DEPRECATION_WARNING", None)
    yield
    if saved is not None:
        os.environ["GENESIS_DISABLE_V1_DEPRECATION_WARNING"] = saved


class TestV1DeprecationWarning:
    def test_first_load_emits_warning(self):
        from vllm.sndr_core.model_configs.registry import get
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            cfg = get("a5000-2x-35b-prod")
            assert cfg is not None
            dep = [w for w in captured
                   if issubclass(w.category, DeprecationWarning)]
            assert len(dep) == 1, (
                f"expected 1 DeprecationWarning, got {len(dep)}"
            )

    def test_warning_message_mentions_v2_migration(self):
        from vllm.sndr_core.model_configs.registry import get
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            get("a5000-2x-35b-prod")
            dep = [w for w in captured
                   if issubclass(w.category, DeprecationWarning)]
            msg = str(dep[0].message)
            assert "deprecated" in msg.lower()
            assert "v2" in msg.lower()
            # Mentions at least one V2 discovery hint.
            assert any(hint in msg for hint in (
                "sndr hardware list",
                "sndr model list-v2",
                "sndr profile list",
                "builtin/presets/",
            )), f"warning lacks V2 migration hint: {msg!r}"

    def test_warning_once_per_key(self):
        """Repeated `get(...)` for the same key warns only once per process."""
        from vllm.sndr_core.model_configs.registry import get
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            for _ in range(5):
                get("a5000-2x-35b-prod")
            dep = [w for w in captured
                   if issubclass(w.category, DeprecationWarning)]
            assert len(dep) == 1, (
                f"warning fired {len(dep)} times — should be 1 per key"
            )

    def test_different_keys_warn_separately(self):
        from vllm.sndr_core.model_configs.registry import get
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            get("a5000-2x-35b-prod")
            get("a5000-2x-27b-int4-tq-k8v4")
            dep = [w for w in captured
                   if issubclass(w.category, DeprecationWarning)]
            assert len(dep) == 2, (
                f"expected 2 warnings (one per distinct V1 key), got {len(dep)}"
            )

    def test_env_var_silences_warning(self):
        from vllm.sndr_core.model_configs.registry import get
        os.environ["GENESIS_DISABLE_V1_DEPRECATION_WARNING"] = "1"
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            cfg = get("a5000-2x-35b-prod")
            assert cfg is not None      # V1 path still works
            dep = [w for w in captured
                   if issubclass(w.category, DeprecationWarning)]
            assert dep == [], "env override did not silence the warning"

    def test_v1_path_still_functional(self):
        """Phase 9 = warn-only freeze. V1 must still serve preflight."""
        from vllm.sndr_core.model_configs.registry import get
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = get("a5000-2x-35b-prod")
        assert cfg is not None
        assert cfg.key == "a5000-2x-35b-prod"
        assert cfg.max_model_len > 0
        assert cfg.hardware.n_gpus > 0


class TestV2AliasNoWarning:
    """V2 aliases compose through `registry_v2.load_alias` — they
    must NOT emit the V1 deprecation warning."""

    def test_v2_alias_silent(self):
        from vllm.sndr_core.model_configs.registry_v2 import load_alias
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            cfg = load_alias("prod-35b")
            assert cfg is not None
            dep = [w for w in captured
                   if issubclass(w.category, DeprecationWarning)]
            assert dep == [], (
                f"V2 alias load emitted V1 deprecation warning: "
                f"{[str(w.message) for w in dep]}"
            )


# ─── audit-no-new-v1 gate ─────────────────────────────────────────────


class TestAuditNoNewV1:
    def test_baseline_matches_current_v1_files(self):
        """Freeze policy: FROZEN_V1_BASELINE must enumerate every
        top-level `builtin/*.yaml`. The baseline is allowed to grow
        with explicit bumps (each bump bumps both the constant in
        `scripts/audit_no_new_v1.py` and the count assertion is here
        implicit — this test catches drift either way)."""
        mod = _import_script("audit_no_new_v1")
        assert mod.FROZEN_V1_BASELINE == mod._current_v1_files(), (
            "FROZEN_V1_BASELINE drifted from the actual top-level "
            "builtin/*.yaml set — bump baseline in "
            "scripts/audit_no_new_v1.py to match (or remove the "
            "extra file)."
        )

    def test_current_v1_files_match_baseline(self):
        mod = _import_script("audit_no_new_v1")
        current = mod._current_v1_files()
        assert current == mod.FROZEN_V1_BASELINE

    def test_script_exits_zero_on_clean_tree(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "audit_no_new_v1.py"),
             "--json"],
            cwd=REPO_ROOT,
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["passed"] is True
        assert payload["added"] == []
        assert payload["removed"] == []


# ─── Roadmap §10 Day 16-20 completion check ───────────────────────────


class TestPhase8bAcceptance:
    """Phase 8b — every V2 alias preflight-resolves through the launcher.

    This is the test-form of the Day 19 V2 acceptance gate. We don't
    actually launch (no GPU on CI), but we DO compose the alias through
    the same path the launcher would use, which is what preflight does
    under the hood.
    """

    @pytest.mark.parametrize("alias", [
        "prod-35b", "prod-27b-tq", "prod-35b-dflash", "long-ctx-27b",
        "qa-27b-tested", "qa-27b-tq-1x", "prod-27b-dflash",
        "experimental-27b-tq-dflash-ab", "example-2x-tier-aware",
        "example-3090-dense-cpu-offload", "example-3090-tier-aware",
    ])
    def test_alias_preflight_path(self, alias):
        from vllm.sndr_core.model_configs.registry_v2 import load_alias
        cfg = load_alias(alias)
        # Compose returned a complete V1 ModelConfig.
        assert cfg is not None
        # Composed key carries the triplet separator. Wave 10 switched the
        # canonical separator from "__" to "--" so V1 ModelConfig kebab-case
        # key regex stays satisfied.
        assert "--" in cfg.key
        assert cfg.hardware.n_gpus >= 1
        assert cfg.max_model_len >= 1
        # Composer attached a docker block from hardware layer.
        assert cfg.docker is not None
        # genesis_env carries the patches matrix.
        assert isinstance(cfg.genesis_env, dict)
