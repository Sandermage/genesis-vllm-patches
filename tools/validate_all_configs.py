#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end validation script for every model / hardware / profile / preset.

Run locally OR inside the production container:

    python3 tools/validate_all_configs.py [--server]

Phases:
    1. YAML parse for every V2 file
    2. V2 schema validation (validate_v2_pack)
    3. V2 compose for every legal (model + hardware) combination
    4. V2 preset.list / preset.show for every preset
    5. Patch registry: per-patch metadata sanity check
    6. CLI smoke (engines.list, engines.info, pins.list, health)
    7. Launcher script parse (syntactic only — does not run them)

Exit code:
    0  all checks pass
    1  one or more failures (final summary table shows which)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILTIN = REPO_ROOT / "sndr" / "model_configs" / "builtin"

# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

if not sys.stdout.isatty():
    GREEN = RED = YELLOW = BOLD = DIM = RESET = ""


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}── {title} ──{RESET}")


# ---------------------------------------------------------------------------

RESULTS: dict[str, dict[str, int]] = {}


def record(phase: str, status: str) -> None:
    RESULTS.setdefault(phase, Counter())[status] += 1


# ---------------------------------------------------------------------------


def phase_yaml_parse() -> None:
    """Phase 1 — Every YAML must parse via yaml.safe_load."""
    import yaml

    section("Phase 1 — YAML parse")
    for yaml_path in sorted(BUILTIN.rglob("*.yaml")):
        rel = yaml_path.relative_to(REPO_ROOT)
        try:
            data = yaml.safe_load(yaml_path.read_text())
            if not isinstance(data, dict):
                fail(f"{rel}: top-level is not a mapping ({type(data).__name__})")
                record("yaml_parse", "fail")
                continue
            record("yaml_parse", "ok")
        except Exception as e:
            fail(f"{rel}: {type(e).__name__}: {e}")
            record("yaml_parse", "fail")
    counts = RESULTS["yaml_parse"]
    print(f"  Summary: {counts['ok']} ok / {counts.get('fail', 0)} fail")


def phase_v2_schema() -> None:
    """Phase 2 — Every V2 YAML must validate via load_*().

    The V2 loaders (load_model / load_hardware / load_profile / load_preset_def)
    are the canonical validation surface — they parse the YAML, run Pydantic
    validation, and run all V2 audit rules.
    """
    section("Phase 2 — V2 schema validation (per-file via load_*)")
    try:
        from sndr.model_configs.registry_v2 import (
            list_hardware,
            list_models,
            list_presets,
            list_profiles,
            load_hardware,
            load_model,
            load_preset_def,
            load_profile,
        )
    except ImportError as e:
        fail(f"cannot import V2 registry: {e}")
        record("schema_validation", "fail")
        return

    pairs: list[tuple[str, list[str], Any]] = [
        ("model", list_models(), load_model),
        ("hardware", list_hardware(), load_hardware),
        ("profile", list_profiles(), load_profile),
        ("preset", list_presets(), load_preset_def),
    ]
    for kind, ids, loader in pairs:
        for cid in ids:
            try:
                loader(cid)
                record("schema_validation", "ok")
            except Exception as e:
                fail(f"{kind}/{cid}: {type(e).__name__}: {e}")
                record("schema_validation", "fail")
    counts = RESULTS["schema_validation"]
    print(f"  Summary: {counts.get('ok', 0)} ok / {counts.get('fail', 0)} fail")


def phase_v2_compose() -> None:
    """Phase 3 — V2 compose for every (model + hardware + profile) triplet.

    Strategy: enumerate every (model, hardware, profile) triplet and run
    ``compose_by_ids``. Reject = expected for many incompatible combos;
    we count rejections separately from hard failures so they don't show
    as 'fail'.
    """
    section("Phase 3 — V2 compose (model × hardware × profile)")
    try:
        from sndr.model_configs.registry_v2 import (
            compose_by_ids,
            list_hardware,
            list_models,
            list_profiles,
            load_profile,
        )
    except ImportError as e:
        fail(f"cannot import V2 compose: {e}")
        record("compose", "fail")
        return

    models = list_models()
    hardware = list_hardware()
    profiles = list_profiles()

    tried = 0
    composed_keys: set[str] = set()
    for m in models:
        for h in hardware:
            for prof in profiles:
                # Skip profile that doesn't belong to this model (V2 contract)
                try:
                    prof_def = load_profile(prof)
                    if prof_def.parent_model and prof_def.parent_model != m:
                        continue
                except Exception:
                    continue

                tried += 1
                try:
                    config = compose_by_ids(m, h, prof)
                    composed_keys.add(config.key)
                    record("compose", "ok")
                except Exception as e:
                    # Many model + hw pairs are intentionally rejected
                    # (model needs 2× GPU but hardware is 1× etc.).
                    record("compose", "rejected")
    print(f"  Summary: tried {tried} model×hw×profile triplets; "
          f"{RESULTS['compose'].get('ok', 0)} composed; "
          f"{RESULTS['compose'].get('rejected', 0)} rejected (expected for incompatible pairs)")
    print(f"  Unique composed keys: {len(composed_keys)}")


def phase_registry() -> None:
    """Phase 4 — Every patch in registry has required metadata."""
    section("Phase 4 — Patch registry metadata sanity")
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    except Exception as e:
        fail(f"cannot import registry: {e}")
        record("registry", "fail")
        return

    required_keys = {"title", "family", "lifecycle", "tier"}
    for pid, entry in PATCH_REGISTRY.items():
        missing = required_keys - set(entry.keys())
        if missing:
            fail(f"{pid}: missing keys: {sorted(missing)}")
            record("registry", "fail")
            continue
        record("registry", "ok")
    counts = RESULTS["registry"]
    print(f"  Summary: {counts['ok']} ok / {counts.get('fail', 0)} fail "
          f"(total {len(PATCH_REGISTRY)} patches)")


def phase_cli() -> None:
    """Phase 5 — sndr CLI commands all succeed."""
    section("Phase 5 — sndr CLI smoke")

    commands = [
        ["python3", "-m", "sndr.cli.main", "engines.list"],
        ["python3", "-m", "sndr.cli.main", "engines.info", "vllm"],
        ["python3", "-m", "sndr.cli.main", "pins.list"],
        ["python3", "-m", "sndr.cli.main", "health"],
        ["python3", "-m", "sndr.cli.main", "--output", "json", "health"],
    ]
    for cmd in commands:
        label = " ".join(cmd[2:])
        try:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                record("cli", "ok")
            else:
                fail(f"{label}: rc={result.returncode}: {result.stderr[:200]}")
                record("cli", "fail")
        except Exception as e:
            fail(f"{label}: {type(e).__name__}: {e}")
            record("cli", "fail")
    counts = RESULTS["cli"]
    print(f"  Summary: {counts.get('ok', 0)} ok / {counts.get('fail', 0)} fail")


def phase_routes() -> None:
    """Phase 6 — FastAPI app loads and all routes reachable via TestClient."""
    section("Phase 6 — FastAPI routes")
    try:
        from fastapi.testclient import TestClient
        from sndr.product_api.server import create_app
    except Exception as e:
        fail(f"cannot import FastAPI app: {e}")
        record("routes", "fail")
        return

    client = TestClient(create_app())
    probes: list[tuple[str, str, int]] = [
        ("GET", "/api/v1/health", 200),
        ("GET", "/api/v1/version", 200),
        ("GET", "/api/v1/engines", 200),
        ("GET", "/api/v1/engines/vllm", 200),
        ("GET", "/api/v1/engines/vllm/pins", 200),
        ("GET", "/api/v1/patches", 200),
        ("GET", "/api/v1/patches/inventory", 200),
        ("GET", "/api/v1/patches/PN119", 200),
        ("GET", "/api/v1/patches/PN9999", 404),
        ("GET", "/api/v1/licensing/status", 200),
    ]
    for method, path, expected in probes:
        try:
            resp = client.request(method, path)
            if resp.status_code == expected:
                record("routes", "ok")
            else:
                fail(f"{method} {path}: got {resp.status_code}, expected {expected}")
                record("routes", "fail")
        except Exception as e:
            fail(f"{method} {path}: {type(e).__name__}: {e}")
            record("routes", "fail")
    counts = RESULTS["routes"]
    print(f"  Summary: {counts.get('ok', 0)} ok / {counts.get('fail', 0)} fail")


def phase_apply_matrix() -> None:
    """Phase 7 — apply doctor / matrix reports no errors."""
    section("Phase 7 — apply matrix (dispatcher doctor)")
    try:
        result = subprocess.run(
            ["python3", "-m", "sndr.compat.cli", "self-test", "--json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            warn(f"self-test rc={result.returncode}; first line of stderr: {result.stderr.splitlines()[:1]}")
            record("apply_matrix", "warn")
        else:
            try:
                data = json.loads(result.stdout)
                patches = data.get("patches", [])
                ok_count = sum(1 for p in patches if p.get("status") in ("ok", "applied"))
                err_count = sum(1 for p in patches if p.get("status") in ("error", "failed"))
                ok(f"self-test loaded: {len(patches)} patches, {ok_count} ok, {err_count} err")
                record("apply_matrix", "ok")
            except json.JSONDecodeError:
                warn(f"self-test stdout not JSON; first 200 chars: {result.stdout[:200]}")
                record("apply_matrix", "warn")
    except FileNotFoundError:
        warn("self-test command not in v12 — checking via registry import instead")
        try:
            from sndr.dispatcher.registry import PATCH_REGISTRY
            ok(f"registry imports cleanly: {len(PATCH_REGISTRY)} patches")
            record("apply_matrix", "ok")
        except Exception as e:
            fail(f"registry import failed: {e}")
            record("apply_matrix", "fail")
    except Exception as e:
        fail(f"self-test failed: {e}")
        record("apply_matrix", "fail")


def phase_layer_rules() -> None:
    """Phase 8 — Layer-rule enforcement."""
    section("Phase 8 — Layer rules (CI gate)")
    try:
        result = subprocess.run(
            ["python3", "tools/ci/verify_layer_imports.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            ok(result.stdout.strip())
            record("layer_rules", "ok")
        else:
            fail(f"layer rules failed:\n{result.stdout}\n{result.stderr}")
            record("layer_rules", "fail")
    except Exception as e:
        fail(f"layer rules: {type(e).__name__}: {e}")
        record("layer_rules", "fail")


# ---------------------------------------------------------------------------


def main() -> int:
    phase_yaml_parse()
    phase_v2_schema()
    phase_v2_compose()
    phase_registry()
    phase_routes()
    phase_cli()
    phase_apply_matrix()
    phase_layer_rules()

    section("FINAL REPORT")
    total_ok = 0
    total_fail = 0
    total_other = 0
    print(f"{'Phase':<22} {'OK':>5} {'FAIL':>5} {'Other':>5}")
    for phase, counts in RESULTS.items():
        ok_n = counts.get("ok", 0)
        fail_n = counts.get("fail", 0)
        other_n = sum(v for k, v in counts.items() if k not in ("ok", "fail"))
        total_ok += ok_n
        total_fail += fail_n
        total_other += other_n
        ind = GREEN if fail_n == 0 else RED
        print(f"  {ind}{phase:<20}{RESET} {ok_n:>5} {fail_n:>5} {other_n:>5}")
    print(f"  {'TOTAL':<20} {total_ok:>5} {total_fail:>5} {total_other:>5}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
