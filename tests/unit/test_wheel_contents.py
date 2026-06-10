# SPDX-License-Identifier: Apache-2.0
"""P0.1 M.2 — Black-box wheel delivery contract test.

Defends the *delivery contract* of the packaged wheel, NOT a specific
`pyproject.toml` implementation. Implementation may change (recursive
globs vs MANIFEST.in vs src-layout, etc.); the contract stays:

  1. V2 layered configs reach the wheel — all four subdirs (model/,
     hardware/, profile/, presets/) have ≥1 *.yaml each.
  2. No `sndr_private` path anywhere in the wheel (hard rule #27 —
     packaged core must not contain that namespace).
  3. No `vllm/sndr_engine/*` in the core wheel (engine = separate
     commercial wheel via license gate).
  4. Runtime registry lookups (`list_models / list_hardware /
     list_profiles` + preset aliases) succeed from an installed
     wheel in an isolated venv — catches resource-lookup errors that
     zip inspection alone misses.

Why this matters
----------------
Pre-P0.1 wheel inspection showed 0/4 V2 subdirs packaged → `sndr preset
list` returned empty on pip-installed wheel. The `pyproject.toml`
package-data spec is the current implementation, but future migrations
(MANIFEST.in, src-layout, hatchling, etc.) must keep the contract
intact. This test makes the contract enforceable independent of the
mechanism.

Cost / preconditions
--------------------
The test builds a wheel via `python -m build` and installs it in a
temp venv. Requires `build` + `pip` available. We install with
`--no-deps` to avoid pulling torch/vllm runtime, then add only
`pyyaml` for the V2 loaders.

Skip semantics
--------------
- Skip if `python -m build` is unavailable (slim CI environment).
- Skip if `pyyaml` cannot be installed (offline CI).
The test is informational on those hosts but mandatory on developer
workstations + release CI.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _have_build_module() -> bool:
    """Detect whether `python -m build` is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import build"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _build_wheel(work_dir: Path) -> Path:
    """Build wheel into `work_dir/dist/`. Returns wheel path."""
    dist = work_dir / "dist"
    dist.mkdir(exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
        cwd=REPO_ROOT, check=True, capture_output=True, timeout=300,
    )
    # v12: pyproject `name = "sndr-platform"` → wheel artifact is
    # `sndr_platform-*.whl` (was `vllm_sndr_core-*.whl` through v11).
    wheels = list(dist.glob("sndr_platform-*.whl"))
    assert len(wheels) == 1, (
        f"expected exactly one wheel artifact, found: {wheels}"
    )
    return wheels[0]


@pytest.fixture(scope="module")
def built_wheel() -> Path:
    """Build the wheel once per test module."""
    if not _have_build_module():
        pytest.skip("`python -m build` unavailable")
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel = _build_wheel(Path(tmpdir))
        # Copy to a persistent dir within the test session, KEEPING the
        # original wheel filename — pip rejects renamed wheels with
        # "is not a valid wheel filename" (name-version-pytag-abitag-
        # platform format is mandatory), which silently downgraded the
        # runtime-contract tests to skips.
        persistent_dir = Path(tempfile.mkdtemp(prefix=f"sndr-test-{os.getpid()}-"))
        persistent = persistent_dir / wheel.name
        shutil.copy(wheel, persistent)
        yield persistent
        shutil.rmtree(persistent_dir, ignore_errors=True)


# ─── Zip-level invariants ─────────────────────────────────────────────────


class TestWheelZipContract:
    """File-list invariants the wheel must satisfy."""

    def test_v2_yamls_present_in_all_four_subdirs(self, built_wheel: Path):
        """P0.A wheel packaging gap regression guard.

        Pre-P0.1: package-data spec was non-recursive → V2 YAMLs missing.
        Post-P0.1 M.1: recursive globs cover {model,hardware,profile,
        presets}/*.yaml. This test fires if the spec regresses to the
        non-recursive form (or src-layout migration drops subdirs).
        """
        with zipfile.ZipFile(built_wheel) as z:
            names = z.namelist()
            for subdir in ("model", "hardware", "profile", "presets"):
                yamls = [
                    n for n in names
                    if f"model_configs/builtin/{subdir}/" in n
                    and n.endswith(".yaml")
                ]
                assert yamls, (
                    f"V2 {subdir}/*.yaml absent from wheel; "
                    f"package-data spec regressed (M.1 invariant)"
                )

    def test_gui_bundle_present_in_wheel(self, built_wheel: Path):
        """The built web UI must ship in the wheel so a pip-installed
        daemon can serve the GUI without the source tree.

        Regression guard: v12.x moved the bundle from
        `vllm/sndr_core/product_api/web_static` to
        `sndr/product_api/legacy/web_static`, but the package-data spec
        still pointed at the old (and an empty) path — the wheel shipped
        zero GUI files. This asserts the contract: index.html + ≥1 hashed
        JS asset under the path the daemon resolves from.
        """
        with zipfile.ZipFile(built_wheel) as z:
            names = z.namelist()
            base = "sndr/product_api/legacy/web_static/"
            assert f"{base}index.html" in names, (
                f"GUI index.html absent from wheel; package-data spec for "
                f"the web_static bundle regressed. web_static entries: "
                f"{[n for n in names if 'web_static' in n][:5]}"
            )
            assets = [
                n for n in names
                if n.startswith(f"{base}assets/") and n.endswith(".js")
            ]
            assert assets, (
                f"GUI JS assets absent from wheel under {base}assets/; "
                f"the daemon would serve a blank page"
            )

    def test_no_sndr_private_anywhere_in_wheel(self, built_wheel: Path):
        """Hard rule #27: no `sndr_private` namespace under `vllm/`.

        Defends against accidental re-introduction of the
        `vllm/sndr_core/sndr_private/` namespace removed in P0.1
        M.3d. The top-level `sndr_private/` directory is gitignored
        and never reaches setuptools, so any wheel match is a real
        regression.
        """
        with zipfile.ZipFile(built_wheel) as z:
            names = z.namelist()
            offenders = [n for n in names if "sndr_private" in n]
            assert not offenders, (
                f"sndr_private leaked into wheel: {offenders}; "
                f"hard rule #27 violated"
            )

    def test_no_sndr_engine_in_core_wheel(self, built_wheel: Path):
        """Engine boundary: `vllm/sndr_engine/*` must ship as a
        separate commercial wheel via license gate, never bundled
        into the core wheel. `pyproject.toml` exclude rule defends
        this; the test acts as regression guard.
        """
        with zipfile.ZipFile(built_wheel) as z:
            names = z.namelist()
            offenders = [n for n in names if n.startswith("vllm/sndr_engine")]
            assert not offenders, (
                f"sndr_engine in core wheel: {offenders}; "
                f"engine commercial-wheel boundary violated"
            )


# ─── Runtime delivery contract (catches resource-lookup errors) ───────────


def _have_pip_install_capability(venv_python: Path) -> bool:
    """Probe whether pip in the venv can install a wheel offline."""
    try:
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "--version"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture
def isolated_venv(built_wheel: Path, tmp_path):
    """Create a temp venv, install the wheel + pyyaml only."""
    venv_dir = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True, capture_output=True, timeout=60,
    )
    py = venv_dir / "bin" / "python"
    if sys.platform == "win32":
        py = venv_dir / "Scripts" / "python.exe"

    if not _have_pip_install_capability(py):
        pytest.skip("venv pip unavailable")

    # Install wheel without runtime deps (torch/vllm too heavy).
    # Then add pyyaml separately — required by V2 YAML loader.
    try:
        subprocess.run(
            [str(py), "-m", "pip", "install", "--quiet", "--no-deps",
             str(built_wheel)],
            check=True, capture_output=True, timeout=120,
        )
        subprocess.run(
            [str(py), "-m", "pip", "install", "--quiet", "pyyaml"],
            check=True, capture_output=True, timeout=60,
        )
    except subprocess.CalledProcessError as e:
        pytest.skip(f"pip install failed (offline/network): {e}")

    return py


class TestWheelRuntimeContract:
    """Runtime invariants — catches resource-lookup errors zip-inspect
    misses (e.g. `importlib.resources` can't find files marked
    package-data in `pyproject.toml` if dir structure broke)."""

    def test_v2_listings_visible_from_installed_wheel(
        self, isolated_venv: Path, tmp_path
    ):
        """Smoke from a neutral cwd (so sys.path[0]='' doesn't pick
        up the source tree). Calls the actual V2 registry API and
        asserts non-empty results."""
        script = tmp_path / "smoke.py"
        script.write_text(
            "import sndr.model_configs.registry_v2 as rv2\n"
            "models = rv2.list_models()\n"
            "hws = rv2.list_hardware()\n"
            "profiles = rv2.list_profiles()\n"
            "assert models, f'list_models() empty: {models}'\n"
            "assert hws, f'list_hardware() empty: {hws}'\n"
            "assert profiles, f'list_profiles() empty: {profiles}'\n"
            "alias_dir = rv2._alias_dir()\n"
            "assert alias_dir.exists(), f'alias_dir missing: {alias_dir}'\n"
            "aliases = rv2._list_yaml_ids(alias_dir)\n"
            "assert aliases, f'no preset aliases visible: {alias_dir}'\n"
            "assert any(a.startswith('prod-') for a in aliases), "
            "    f'prod-* presets missing from {aliases}'\n"
            "print(f'V2 delivery OK: {len(models)} models, '\n"
            "      f'{len(hws)} hw, {len(profiles)} profiles, '\n"
            "      f'{len(aliases)} aliases')\n"
        )
        # Run from a neutral cwd to ensure imports come from installed
        # wheel, not the source tree.
        result = subprocess.run(
            [str(isolated_venv), str(script)],
            cwd="/tmp", capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"V2 runtime delivery failed:\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )

    def test_engine_not_importable_from_core(
        self, isolated_venv: Path
    ):
        """`vllm.sndr_engine` must not be discoverable from a clean
        core-only install (no engine wheel present).

        v12: the sndr-platform wheel ships `sndr*` only — the parent
        `vllm` namespace itself is absent from a core-only venv, which
        makes `find_spec` raise ModuleNotFoundError on the parent. That
        is the boundary holding a fortiori, so it is treated as None.
        """
        probe = (
            "import importlib.util as iu\n"
            "try:\n"
            "    spec = iu.find_spec('vllm.sndr_engine')\n"
            "except ModuleNotFoundError:\n"
            "    spec = None  # parent 'vllm' absent — boundary holds\n"
            "assert spec is None, f'engine in core: {spec}'\n"
            "print('engine boundary OK')\n"
        )
        result = subprocess.run(
            [str(isolated_venv), "-c", probe],
            cwd="/tmp", capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, (
            f"engine boundary failed:\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )
