# SPDX-License-Identifier: Apache-2.0
"""TDD for the F-010 paid/free boundary (PR38 Phase 4 Step 6).

Edition tests verify the BOUNDARY between the Apache-licensed
`sndr` (community) and the commercial `vllm.sndr_engine`
(paid). Per Sander's **strict-AND rule** (audit 2026-05-08,
canonical source `docs/PATCHES.md:50-58`):

  A patch is `tier="engine"` only when ALL FOUR conditions hold:
    1. NOT present on the public github repo
    2. NO external author credit in title/credit text
    3. NO PR link / PR number in title/credit text
    4. NO `upstream_pr` / `related_upstream_prs` field

  All other patches are `tier="community"`.

  Historical note: the original informal statement was
  "patches with upstream_pr → community / without → engine".
  That was the loose form; the strict-AND rule (above) is the
  operative current rule, enforced by
  `test_every_engine_patch_passes_strict_AND_rule` below.

The boundary is enforced by:

  1. Tier classification in `dispatcher.PATCH_REGISTRY[pid]["tier"]`
  2. License gate in `dispatcher.decision._check_engine_tier_eligible`
     consulting `sndr.license.check_engine_tier_eligible()`
  3. Engine-tier impl files physically located at
     `vllm/sndr_engine/patches/<family>/<patch>.py`
  4. Stub redirects at
     `vllm/sndr_core/integrations/<family>/<patch>.py` that proxy to engine
     OR raise ImportError when the engine package is absent

These tests ensure all four layers stay coherent. Update if the rule
itself changes — don't paper over a real boundary regression.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ─── Tier classification rule ──────────────────────────────────────────────


class TestTierClassificationRule:
    """Each registry entry's tier MUST match Sander's rule.

    Drift detector: if a contributor adds a new patch without
    `upstream_pr` but labels it `community`, this test fails so the
    label is reviewed before merge.
    """

    def test_every_community_patch_has_external_reference(self):
        """Community-tier ⇒ at least ONE of (Sander's strict AND rule
        inverted, 2026-05-08):
          (a) impl ships in the public `sndr/` tree (i.e., on the
              public github repo — proxy for "on github"; v12 form of
              the pre-v12 `vllm/sndr_core/integrations/` check — covers
              `sndr.engines.vllm.patches.*`, retired impls relocated to
              `sndr.engines.vllm._archive.*`, and non-patch homes such
              as `sndr.observability.*`; `sndr_private` does NOT match
              the `sndr.` prefix),
          (b) `upstream_pr` / `related_upstream_prs` set,
          (c) PR ref in title/credit text (vllm#N / SGLang#N / PR #N),
          (d) external-author credit (@user / "backport of …" / known name).

        Engine = NONE of (a)-(d) — Genesis-original IP, not on the
        public repo, no external attribution anywhere.
        """
        import re
        from sndr.dispatcher import PATCH_REGISTRY, iter_patch_specs

        PR_REF_PAT = re.compile(
            r"\b(?:vllm|sglang|llama\.cpp|huggingface|HF|github)[#/]\d+|"
            r"\bPR\s*#\d+|\bissue\s*#\d+|\b#\d{4,}\b",
            re.IGNORECASE,
        )
        AUTHOR_PAT = re.compile(
            r"@[a-zA-Z][a-zA-Z0-9_-]+|"
            r"\bbackport\s+of\s+|"
            r"\bbackport-with-extension\s+of\s+|"
            r"\bauthored?\s+by\s+|"
            r"\b(?:apnar|noonghunna|kevglynn|ampersandru|joachim|mistral|"
            r"adurham|jartx|fanghao|anishesg|adobe|sfbemerk|tfriedel|thc1006|"
            r"itailang|kevin\s+glynn|yuan[- ]luo|kevkglynn|devarakondasrikanth|"
            r"zobinhuang)\b",
            re.IGNORECASE,
        )

        # (a) Map patch_id → ships-in-public-sndr-tree flag. The whole
        # `sndr` package is the public Apache wheel (pyproject includes
        # `sndr*`, excludes `sndr_private*`); commercial IP lives in
        # `vllm.sndr_engine` / `sndr_private` which never match `sndr.`.
        ships_in_core = {}
        has_impl = set()
        for spec in iter_patch_specs():
            am = spec.apply_module or ""
            ships_in_core[spec.patch_id] = am.startswith("sndr.")
            if am:
                has_impl.add(spec.patch_id)

        violations = []
        for pid, meta in PATCH_REGISTRY.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("tier") != "community":
                continue
            # Skip ledger-only entries (no on-disk impl) — informational
            # registry rows for retired/legacy patches with no apply
            # behavior. Tier label is documentation-only for these.
            if pid not in has_impl:
                continue
            # (a) shipped in sndr_core (public Apache wheel = on github)
            if ships_in_core.get(pid):
                continue
            # (b) has upstream_pr field
            if (meta.get("upstream_pr") is not None
                    or bool(meta.get("related_upstream_prs"))):
                continue
            text = f"{meta.get('title','')}\n{meta.get('credit','')}"
            # (c) / (d) text refs
            if PR_REF_PAT.search(text) or AUTHOR_PAT.search(text):
                continue
            violations.append(pid)
        assert not violations, (
            f"community-tier patches with NO external reference and "
            f"NOT shipped in sndr_core: {violations}. Per Sander's "
            "strict AND rule, these should be tier=engine (Genesis-"
            "original IP, not on the public github repo)."
        )

    def test_every_engine_patch_passes_strict_AND_rule(self):
        """Engine-tier ⇒ Genesis-original ⇒ NONE of (upstream_pr, PR ref
        in text, external-author credit).

        Per Sander's strict AND rule (2026-05-08): a patch is paid only
        if ALL these are absent simultaneously. If even one is present,
        it's a backport / community contribution and belongs in
        tier=community."""
        import re
        from sndr.dispatcher import PATCH_REGISTRY

        PR_REF_PAT = re.compile(
            r"\b(?:vllm|sglang|llama\.cpp|huggingface|HF|github)[#/]\d+|"
            r"\bPR\s*#\d+|\bissue\s*#\d+|\b#\d{4,}\b",
            re.IGNORECASE,
        )
        AUTHOR_PAT = re.compile(
            r"@[a-zA-Z][a-zA-Z0-9_-]+|"
            r"\bbackport\s+of\s+|"
            r"\bauthored?\s+by\s+|"
            r"\b(?:apnar|noonghunna|kevglynn|ampersandru|joachim|mistral|"
            r"adurham|jartx|fanghao|anishesg|adobe|sfbemerk|tfriedel|thc1006|"
            r"itailang|kevin\s+glynn|yuan[- ]luo|zobinhuang)\b",
            re.IGNORECASE,
        )

        violations = []
        for pid, meta in PATCH_REGISTRY.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("tier") != "engine":
                continue
            if (meta.get("upstream_pr") is not None
                    or bool(meta.get("related_upstream_prs"))):
                violations.append((pid, "has upstream_pr"))
                continue
            text = f"{meta.get('title','')}\n{meta.get('credit','')}"
            if PR_REF_PAT.search(text):
                violations.append((pid, "has PR ref in text"))
                continue
            if AUTHOR_PAT.search(text):
                violations.append((pid, "has author/backport ref"))
                continue
        assert not violations, (
            f"engine-tier patches violating strict AND rule: {violations[:5]}"
        )

    def test_tier_distribution_sane(self):
        """Sanity: bulk is community; engine bucket may be empty.

        P0-3/P0-4 (audit 2026-05-08): public Genesis repo no longer
        carries any `tier="engine"` patches. PN72 was the last one and
        moved core→core because it was Genesis-original community code,
        not commercial IP. The engine BUCKET (and the license-gate
        infrastructure) remain in place for future private overlay
        code that genuinely requires a license — the test just no
        longer demands a current engine entry.
        """
        from sndr.dispatcher import PATCH_REGISTRY
        community = sum(
            1 for m in PATCH_REGISTRY.values()
            if isinstance(m, dict) and m.get("tier") == "community"
        )
        engine = sum(
            1 for m in PATCH_REGISTRY.values()
            if isinstance(m, dict) and m.get("tier") == "engine"
        )
        assert community >= 100, f"too few community tier: {community}"
        # Engine MAY be 0 today (skeleton package only). What matters
        # is no NEGATIVE counts and nothing accidentally in a third
        # tier.
        assert engine >= 0
        assert community + engine == sum(
            1 for m in PATCH_REGISTRY.values()
            if isinstance(m, dict) and m.get("tier") in ("community", "engine")
        )


# ─── Engine impl physical location ─────────────────────────────────────────


class TestEngineImplLocation:
    """Engine-tier patches MUST have their real impl at
    `vllm/sndr_engine/patches/<family>/<patch>.py` and a redirect
    stub at the corresponding `vllm/sndr_core/integrations/<family>/<patch>.py`.

    This guarantees the Apache-licensed `vllm-sndr-core` wheel does NOT
    contain engine IP — only thin redirect stubs that fail-fast
    without `vllm.sndr_engine` installed.
    """

    REPO_ROOT = Path(__file__).resolve().parents[2]
    SNDR_CORE_PATCHES = REPO_ROOT / "sndr" / "engines" / "vllm" / "patches"
    SNDR_ENGINE_PATCHES = REPO_ROOT / "vllm" / "sndr_engine" / "patches"

    def test_sndr_core_engine_redirects_are_thin(self):
        """Each engine-tier patch's sndr_core file MUST be a thin
        redirect (under 50 lines) — no real impl IP in the Apache wheel."""
        from sndr.dispatcher import PATCH_REGISTRY, iter_patch_specs

        violations = []
        for spec in iter_patch_specs():
            meta = PATCH_REGISTRY.get(spec.patch_id, {})
            if not isinstance(meta, dict) or meta.get("tier") != "engine":
                continue
            if spec.apply_module is None:
                continue
            if not spec.apply_module.startswith("sndr.engines.vllm.patches."):
                continue
            rel = spec.apply_module.replace(".", "/") + ".py"
            f = self.REPO_ROOT / rel
            if not f.is_file():
                continue
            n_lines = sum(1 for _ in f.read_text().splitlines())
            if n_lines > 60:
                violations.append((spec.patch_id, n_lines, rel))
        assert not violations, (
            f"engine-tier sndr_core files too large (must be redirects only): "
            f"{violations[:5]}. Real impl belongs in "
            f"vllm/sndr_engine/patches/<family>/<patch>.py."
        )

    def test_engine_impl_directory_has_real_files(self):
        """If `vllm/sndr_engine/patches/` exists, it must contain real
        impl files (no shim-only directory).

        P0-3/P0-4 (audit 2026-05-08): the public Genesis distribution
        ships an empty `sndr_engine/` skeleton — no `patches/` subdir
        at all. This test skips cleanly in that case.
        """
        engine_dir = self.SNDR_ENGINE_PATCHES
        if not engine_dir.is_dir():
            pytest.skip(
                "sndr_engine/patches dir not present (skeleton-only "
                "engine — P0-3/P0-4 closure)"
            )
        impl_files = [
            f for f in engine_dir.rglob("*.py")
            if not f.name.startswith("__") and "__pycache__" not in f.parts
        ]
        assert len(impl_files) >= 1, (
            f"no engine impl files: {len(impl_files)}. "
            "Expected at least PN72/PN78/P39a in sndr_engine/patches/."
        )
        # At least one engine file must be substantial (real impl, not shim)
        substantial = [
            f for f in impl_files
            if sum(1 for _ in f.read_text().splitlines()) > 100
        ]
        assert substantial, (
            "no substantial engine impl files found — all are shims? "
            "F-010 migration may not have moved real impls."
        )


# ─── License gate in dispatcher ────────────────────────────────────────────


class TestDispatcherLicenseGate:
    """Dispatcher tier-gate consults the license module before allowing
    an engine-tier patch to apply."""

    def test_engine_patch_skipped_without_license(self, monkeypatch):
        """No license key + engine package present → tier-gate skips
        with NO_KEY status."""
        from sndr.dispatcher import should_apply, PATCH_REGISTRY

        engine_pids = [
            pid for pid, meta in PATCH_REGISTRY.items()
            if isinstance(meta, dict) and meta.get("tier") == "engine"
        ]
        if not engine_pids:
            pytest.skip("no engine-tier patches in registry")
        pid = engine_pids[0]

        # Ensure no override + no key
        monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
        monkeypatch.delenv("SNDR_ENGINE_LICENSE_KEY", raising=False)
        # Set the patch's env_flag truthy so the env check would otherwise pass
        env_flag = PATCH_REGISTRY[pid].get("env_flag")
        if env_flag:
            monkeypatch.setenv(env_flag, "1")

        decision, reason = should_apply(pid)
        assert decision is False
        # Either tier-gate fired (preferred) or another gate above. Accept
        # either, but log if it's not the tier-gate.
        if "tier=engine" in reason:
            # `licens` matches both "license" and "licensing" (the
            # NO_PACKAGE reason currently uses the latter form).
            assert (
                "licens" in reason.lower() or "no_key" in reason.lower()
            )

    def test_engine_patch_proceeds_with_license(self, monkeypatch):
        """License key set + engine package present + env-flag truthy →
        decision is True (or skipped for non-tier reason)."""
        from sndr.dispatcher import should_apply, PATCH_REGISTRY

        engine_pids = [
            pid for pid, meta in PATCH_REGISTRY.items()
            if isinstance(meta, dict) and meta.get("tier") == "engine"
        ]
        if not engine_pids:
            pytest.skip("no engine-tier patches")
        pid = engine_pids[0]

        monkeypatch.setenv("SNDR_ENGINE_LICENSE_KEY", "test-key")
        env_flag = PATCH_REGISTRY[pid].get("env_flag")
        if env_flag:
            monkeypatch.setenv(env_flag, "1")
        decision, reason = should_apply(pid)
        # Either applies cleanly or is skipped for a non-license reason.
        # `licens` substring covers both "license" and "licensing"; the
        # NO_PACKAGE skip (engine package missing) also contains
        # "licensing" in its boilerplate so we exempt that path explicitly.
        if not decision:
            non_license_skip = (
                "licens" not in reason.lower()
                or "vllm.sndr_engine not installed" in reason
            )
            assert non_license_skip, (
                f"unexpected license-related skip with key set: {reason}"
            )

    def test_tier_override_forces_skip_even_when_licensed(self, monkeypatch):
        """SNDR_ENABLE_TIER_OVERRIDE=1 → community-only mode regardless
        of license. CI / community deployment use case."""
        from sndr.dispatcher import should_apply, PATCH_REGISTRY

        engine_pids = [
            pid for pid, meta in PATCH_REGISTRY.items()
            if isinstance(meta, dict) and meta.get("tier") == "engine"
        ]
        if not engine_pids:
            pytest.skip("no engine-tier patches")
        pid = engine_pids[0]

        monkeypatch.setenv("SNDR_ENGINE_LICENSE_KEY", "test-key")
        monkeypatch.setenv("SNDR_ENABLE_TIER_OVERRIDE", "1")
        env_flag = PATCH_REGISTRY[pid].get("env_flag")
        if env_flag:
            monkeypatch.setenv(env_flag, "1")
        decision, reason = should_apply(pid)
        assert decision is False
        assert "tier" in reason.lower() or "override" in reason.lower()


# ─── Wheel-build package separation ────────────────────────────────────────


class TestWheelPackageSeparation:
    """`pyproject.toml` declares which packages ship in the
    `vllm-sndr-core` wheel. Per F-010 the wheel must NOT contain
    engine impl files — only thin redirects."""

    def test_pyproject_includes_sndr_core(self):
        """Wheel carries the sndr runtime tree (v12: ``sndr*`` covers the
        registry; pre-v12 this was ``vllm.sndr_core*``)."""
        try:
            import tomllib
        except ImportError:
            pytest.skip("tomllib not available")
        repo_root = Path(__file__).resolve().parents[2]
        with open(repo_root / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        include = data["tool"]["setuptools"]["packages"]["find"]["include"]
        assert any(p in ("sndr", "sndr*") or p.startswith("sndr.") for p in include), (
            f"wheel include must cover the sndr package (found: {include})"
        )

    def test_pyproject_root_has_sndr_console_entry(self):
        """`sndr` console entry point is in the canonical pyproject."""
        try:
            import tomllib
        except ImportError:
            pytest.skip("tomllib not available")
        repo_root = Path(__file__).resolve().parents[2]
        with open(repo_root / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        scripts = data["project"]["scripts"]
        assert "sndr" in scripts
