#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V2 runtime / ModelDef pin harmonization audit.

After P2.3 unified all hardware YAMLs on a single dev371 runtime image
and a3fa5265 wired the renderer to honor `hw.runtime.docker.image`,
this script locks the invariants in place so the discipline survives
the next pin bump:

  R-PIN-1 — no bare mutable vLLM nightly under
            ``vllm/sndr_core/model_configs/builtin/hardware/*.yaml``.
            Hardware YAMLs must pin an explicit-hash tag (or
            digest-backed tag), never the floating ``:nightly`` /
            ``:latest`` / ``:main`` aliases.

  R-PIN-2 — every hardware YAML that declares ``runtime.docker.image``
            must also declare a non-empty ``runtime.docker.image_digest``
            beginning with ``vllm/vllm-openai@sha256:``. The digest is
            the byte-locked anchor; the tag is for human readability.

  R-PIN-3 — rendered launcher's IMAGE line must equal the composed
            ``cfg.docker.image`` for representative profiles. P2.1
            wired the renderer to read ``hw.runtime.docker.image``
            verbatim, and compose() carries the same value into
            ``cfg.docker.image``. This rule is a regression gate
            against a future refactor that re-introduces a hardcoded
            fallback or a parallel image-resolution path.

  R-PIN-4 — ModelDef pin migration status. Allowed pin set is
            ``{dev338, dev371}``. Any value outside this set fails
            (catches stale pins or typos). Gemma models are expected
            to be dev371. Qwen models may currently be dev338 OR
            dev371 — the script reports the migration status but does
            NOT fail for Qwen models still on dev338, since P2.4d
            promotes them one at a time with smoke evidence.

The script is idempotent and read-only — it never modifies files,
registry, or git state.

Exit codes:

  0 — all selected rules pass (clean tree, allowed migration state).
  1 — at least one violation found.
  2 — audit tooling itself failed (import error, missing dependency).

Usage:

  python3 scripts/audit_v2_runtime_pins.py
  python3 scripts/audit_v2_runtime_pins.py --rule R-PIN-3 --verbose
  python3 scripts/audit_v2_runtime_pins.py --json > /tmp/pin-audit.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Repo layout — resolved relative to this script's location.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure the repo's vllm/ is importable when audit runs from a clean shell.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HARDWARE_DIR = (
    REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "hardware"
)
MODEL_DIR = (
    REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "model"
)

# ─── Constants ──────────────────────────────────────────────────────────

# Pins the project currently recognizes as legitimate ModelDef values.
# Extend this set when a new pin lands AND a smoke + bench validates it.
ALLOWED_MODELDEF_PINS = frozenset({
    "0.20.2rc1.dev338+gbf0d2dc6d",
    "0.20.2rc1.dev371+gbf610c2f5",
})

# Gemma family ModelDefs are expected to be on dev371 (validated path).
# Qwen family ModelDefs may be on either pin (dev338 baseline OR dev371
# after per-model smoke promotion via P2.4d). Use file-stem prefix to
# classify.
GEMMA_PREFIX = "gemma-"
QWEN_PREFIX = "qwen"

# A mutable image tag is anything ending in `:<word>` with no SHA suffix.
# We allow `:nightly-<sha>` and digest-backed `@sha256:...` references.
_BARE_MUTABLE_TAGS = ("nightly", "latest", "main", "stable", "dev")

# Representative (profile, hardware) combinations exercised by R-PIN-3.
# Picked to cover the three live hardware definitions; if a profile in
# this list is removed from the registry, R-PIN-3 reports a tooling
# error (the operator must update the list, not silently drop coverage).
REPRESENTATIVE_RENDERS = (
    ("gemma4-tq-mtp-structured-k4", "a5000-2x-24gbvram-16cpu-128gbram"),
    ("35b-balanced", "a5000-2x-24gbvram-16cpu-128gbram"),
    ("tier-aware-3090", "single-3090-24gbvram"),
)


# ─── YAML helpers (regex, no PyYAML dependency) ─────────────────────────

# Each pattern is anchored to start-of-line + leading indent so that it
# only matches top-level fields under the expected parent. This avoids
# false positives from arbitrary comment / string content elsewhere.

_IMAGE_RE = re.compile(
    r"^\s{4}image:\s*(?P<value>\S+)",
    re.MULTILINE,
)
_IMAGE_DIGEST_RE = re.compile(
    r"^\s{4}image_digest:\s*(?P<value>\S+)",
    re.MULTILINE,
)
_VLLM_PIN_REQUIRED_RE = re.compile(
    r"^\s{2}vllm_pin_required:\s*(?P<value>\S+)",
    re.MULTILINE,
)


def _strip_yaml_value(raw: str) -> str:
    """Strip trailing comment + surrounding quotes from a YAML scalar."""
    # YAML inline comment starts with " #" after the value
    if " #" in raw:
        raw = raw.split(" #", 1)[0]
    raw = raw.strip().strip('"').strip("'")
    return raw


def _rel(path: Path) -> str:
    """Best-effort REPO_ROOT-relative display. Falls back to absolute
    path when the file lives outside the tree (e.g. tmp_path in tests).
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_hardware_image(yaml_path: Path) -> str | None:
    """Extract the ``runtime.docker.image`` value from a hardware YAML."""
    src = yaml_path.read_text()
    m = _IMAGE_RE.search(src)
    return _strip_yaml_value(m.group("value")) if m else None


def _read_hardware_image_digest(yaml_path: Path) -> str | None:
    src = yaml_path.read_text()
    m = _IMAGE_DIGEST_RE.search(src)
    return _strip_yaml_value(m.group("value")) if m else None


def _read_model_pin(yaml_path: Path) -> str | None:
    src = yaml_path.read_text()
    m = _VLLM_PIN_REQUIRED_RE.search(src)
    return _strip_yaml_value(m.group("value")) if m else None


# ─── Rule implementations ───────────────────────────────────────────────


def _is_bare_mutable(image: str) -> bool:
    """Return True if image is a mutable floating tag with no SHA suffix.

    Examples:
      bare:        vllm/vllm-openai:nightly
      bare:        vllm/vllm-openai:latest
      not bare:    vllm/vllm-openai:nightly-bf610c2f...
      not bare:    vllm/vllm-openai@sha256:7f047b...
    """
    # Digest reference is never mutable.
    if "@sha256:" in image:
        return False
    # No `:tag` at all (e.g. plain repo name) — treat as bare.
    if ":" not in image:
        return True
    tag = image.rsplit(":", 1)[1]
    return tag in _BARE_MUTABLE_TAGS


def check_r_pin_1_no_mutable_nightly() -> list[str]:
    """R-PIN-1: hardware YAMLs must pin an explicit-hash image."""
    issues: list[str] = []
    for yaml_path in sorted(HARDWARE_DIR.glob("*.yaml")):
        image = _read_hardware_image(yaml_path)
        if image is None:
            # No docker block — skip. R-PIN-2 may report if the YAML
            # declares docker runtime but lacks the image field.
            continue
        if _is_bare_mutable(image):
            issues.append(
                f"{_rel(yaml_path)}: runtime.docker.image "
                f"is a bare mutable tag ({image!r}). Use an explicit-hash "
                f"tag (e.g. vllm/vllm-openai:nightly-<sha>) or "
                f"@sha256:<digest> reference."
            )
    return issues


def check_r_pin_2_digest_present() -> list[str]:
    """R-PIN-2: every hardware YAML with `image:` must also have a
    non-empty `image_digest:` beginning with `vllm/vllm-openai@sha256:`.
    """
    issues: list[str] = []
    for yaml_path in sorted(HARDWARE_DIR.glob("*.yaml")):
        image = _read_hardware_image(yaml_path)
        if image is None:
            continue  # no docker block — out of scope
        digest = _read_hardware_image_digest(yaml_path)
        if not digest:
            issues.append(
                f"{_rel(yaml_path)}: runtime.docker.image "
                f"is declared ({image!r}) but image_digest is missing or "
                f"empty. Add the @sha256: digest for byte-lock."
            )
            continue
        if not digest.startswith("vllm/vllm-openai@sha256:"):
            issues.append(
                f"{_rel(yaml_path)}: image_digest "
                f"({digest!r}) does not begin with "
                f"'vllm/vllm-openai@sha256:'. Digest must be a full "
                f"repo@sha256: reference."
            )
    return issues


def check_r_pin_3_render_parity() -> tuple[list[str], list[str]]:
    """R-PIN-3: rendered IMAGE line equals composed cfg.docker.image
    AND equals hw.runtime.docker.image, for representative profiles."""
    errors: list[str] = []
    infos: list[str] = []

    try:
        from vllm.sndr_core.cli.profile import render_profile_launcher
        from vllm.sndr_core.model_configs.compose import compose
        from vllm.sndr_core.model_configs.registry_v2 import (
            load_hardware, load_model, load_profile,
        )
    except Exception as e:  # noqa: BLE001
        errors.append(
            f"R-PIN-3 tooling unavailable: import failed "
            f"({type(e).__name__}: {e}). "
            f"The audit cannot verify render parity without sndr_core."
        )
        return errors, infos

    image_line_re = re.compile(r'^IMAGE="([^"]+)"\s*$', re.MULTILINE)

    for profile_id, hardware_id in REPRESENTATIVE_RENDERS:
        try:
            profile = load_profile(profile_id)
            model = load_model(profile.parent_model)
            hardware = load_hardware(hardware_id)
            cfg = compose(model, hardware, profile)
        except Exception as e:  # noqa: BLE001
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): compose() "
                f"failed: {type(e).__name__}: {e}"
            )
            continue

        composed_image = (
            cfg.docker.image if cfg.docker is not None else None
        )
        hw_image = (
            hardware.runtime.docker.image
            if hardware.runtime is not None
            and hardware.runtime.docker is not None
            else None
        )
        if composed_image is None or hw_image is None:
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): "
                f"no docker block in composed cfg / hardware — cannot "
                f"verify image parity."
            )
            continue
        if composed_image != hw_image:
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): "
                f"composed cfg.docker.image={composed_image!r} differs "
                f"from hw.runtime.docker.image={hw_image!r}. compose() "
                f"must carry the hardware image verbatim."
            )

        try:
            script = render_profile_launcher(profile_id, hardware_id)
        except Exception as e:  # noqa: BLE001
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): "
                f"render_profile_launcher() failed: "
                f"{type(e).__name__}: {e}"
            )
            continue

        m = image_line_re.search(script)
        if m is None:
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): "
                f"rendered launcher has no `IMAGE=\"...\"` line."
            )
            continue
        rendered = m.group(1)
        if rendered != composed_image:
            errors.append(
                f"R-PIN-3 ({profile_id!r}, {hardware_id!r}): "
                f"rendered IMAGE={rendered!r} differs from composed "
                f"cfg.docker.image={composed_image!r}. The renderer "
                f"must emit the hardware-pinned image verbatim."
            )
        else:
            infos.append(
                f"{profile_id!r} + {hardware_id!r}: rendered = composed "
                f"= {rendered!r}"
            )

    return errors, infos


def check_r_pin_4_modeldef_migration() -> tuple[list[str], list[str]]:
    """R-PIN-4: ModelDef pin migration status.

    Fails on:
      * missing `vllm_pin_required` field
      * value outside ALLOWED_MODELDEF_PINS

    Reports as info (NOT fail):
      * per-model migration status across both pins
      * count of Qwen models still on dev338 (next P2.4d candidates)
    """
    errors: list[str] = []
    infos: list[str] = []

    by_family: dict[str, dict[str, list[str]]] = {
        "gemma": {"dev338": [], "dev371": [], "other": []},
        "qwen": {"dev338": [], "dev371": [], "other": []},
        "unknown": {"dev338": [], "dev371": [], "other": []},
    }

    for yaml_path in sorted(MODEL_DIR.glob("*.yaml")):
        pin = _read_model_pin(yaml_path)
        stem = yaml_path.stem
        rel = _rel(yaml_path)
        if pin is None:
            errors.append(
                f"{rel}: vllm_pin_required is missing. Every ModelDef "
                f"must declare a pin."
            )
            continue
        if pin not in ALLOWED_MODELDEF_PINS:
            errors.append(
                f"{rel}: vllm_pin_required={pin!r} is not in the "
                f"allowed set "
                f"{sorted(ALLOWED_MODELDEF_PINS)}. Either add the new "
                f"pin to ALLOWED_MODELDEF_PINS (after smoke + bench "
                f"validation) or correct the typo."
            )
            continue
        # Classify
        if stem.startswith(GEMMA_PREFIX):
            family = "gemma"
        elif stem.startswith(QWEN_PREFIX):
            family = "qwen"
        else:
            family = "unknown"
        if "dev338" in pin:
            by_family[family]["dev338"].append(stem)
        elif "dev371" in pin:
            by_family[family]["dev371"].append(stem)
        else:
            by_family[family]["other"].append(stem)

    # Infos: per-family migration table.
    for family in ("gemma", "qwen", "unknown"):
        d338 = by_family[family]["dev338"]
        d371 = by_family[family]["dev371"]
        other = by_family[family]["other"]
        if not (d338 or d371 or other):
            continue
        infos.append(
            f"{family}: dev371={len(d371)} dev338={len(d338)} "
            f"other={len(other)}"
        )
        for stem in d371:
            infos.append(f"  {stem} → dev371")
        for stem in d338:
            infos.append(f"  {stem} → dev338  (P2.4d candidate)")

    return errors, infos


# ─── CLI driver ─────────────────────────────────────────────────────────


RULES = {
    "R-PIN-1": ("no bare mutable vLLM nightly", check_r_pin_1_no_mutable_nightly),
    "R-PIN-2": ("hardware image_digest present", check_r_pin_2_digest_present),
    "R-PIN-3": ("render parity (cfg.docker.image == hw.runtime.docker.image == IMAGE)", None),
    "R-PIN-4": ("ModelDef pin migration status", None),
}


def _run_rule(rule: str) -> tuple[list[str], list[str]]:
    """Return (errors, infos) for one rule."""
    if rule == "R-PIN-3":
        return check_r_pin_3_render_parity()
    if rule == "R-PIN-4":
        return check_r_pin_4_modeldef_migration()
    _, fn = RULES[rule]
    return fn(), []  # type: ignore[misc]


def _emit_text(
    results: dict[str, tuple[list[str], list[str]]],
    verbose: bool,
) -> None:
    print()
    print("╭──────────────────────────────────────────────────────────╮")
    print("│  V2 runtime / ModelDef pin harmonization audit           │")
    print("╰──────────────────────────────────────────────────────────╯")
    print()
    for rule, (errors, infos) in results.items():
        title = RULES[rule][0]
        if errors:
            print(f"  ✗ {rule}  {title}: {len(errors)} violation(s)")
            for issue in errors:
                print(f"      {issue}")
        else:
            print(f"  ✓ {rule}  {title}: clean")
        if verbose and infos:
            print(f"      ── informational ({len(infos)}) ──")
            for info in infos:
                print(f"      ℹ {info}")
    total_err = sum(len(e) for e, _ in results.values())
    total_info = sum(len(i) for _, i in results.values())
    print()
    if total_err:
        print(f"  ✗ {total_err} violation(s) — exit 1")
    else:
        suffix = f" + {total_info} informational" if total_info else ""
        print(f"  ✓ All selected rules clean{suffix} — exit 0")
    print()


def _emit_json(results: dict[str, tuple[list[str], list[str]]]) -> None:
    payload = {
        rule: {
            "title": RULES[rule][0],
            "violations": errors,
            "infos": infos,
            "status": "fail" if errors else "pass",
        }
        for rule, (errors, infos) in results.items()
    }
    payload["_summary"] = {
        "violations_total": sum(len(e) for e, _ in results.values()),
        "infos_total": sum(len(i) for _, i in results.values()),
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument(
        "--rule",
        choices=("all", *RULES.keys()),
        default="all",
        help="which rule to check (default: all).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of human report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="include informational (non-failing) findings in text output.",
    )
    args = parser.parse_args(argv)

    rules = list(RULES.keys()) if args.rule == "all" else [args.rule]

    try:
        results = {rule: _run_rule(rule) for rule in rules}
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"audit_v2_runtime_pins: tooling error: "
            f"{type(e).__name__}: {e}\n"
        )
        return 2

    if args.json:
        _emit_json(results)
    else:
        _emit_text(results, verbose=args.verbose)

    total_errors = sum(len(errors) for errors, _ in results.values())
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
