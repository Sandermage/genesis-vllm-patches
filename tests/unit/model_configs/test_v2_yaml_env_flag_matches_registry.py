# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard — V2 model/profile/preset/hardware YAMLs
must NOT enable a `GENESIS_ENABLE_*` env flag that is unknown to the
PATCH_REGISTRY.

Why this matters
----------------

The spec-driven and legacy dispatchers both gate patches via the
EXACT `env_flag` string stored in PATCH_REGISTRY. An operator-visible
flag like:

    GENESIS_ENABLE_P109_SAMPLING_PARAMS_VOCAB_RANGE_VALIDATORS: '1'

silently does NOTHING if the registry's `env_flag` for P109 is the
short form `"GENESIS_ENABLE_P109"`. The patch stays skipped on every
boot, the operator sees no error in the apply log — just an unused
env var — and the feature is DEAD in production without anyone
noticing.

v11.4.0 bug class discovered: 4 gemma-4 V2 model YAMLs were enabling
the long-form names for P109 + PN110 added in v11.0.x. Registry had
short form. Every gemma-4 operator who relied on these YAMLs for
sampler-bounds safety + block-pool free-dedup got a silent no-op.

This test pins the boundary so the same class of bug can't return.
Any new `GENESIS_ENABLE_<NAME>` line in any V2 YAML must match an
existing `env_flag` field in PATCH_REGISTRY.

Allowlist via `_KNOWN_NON_REGISTRY_ENV_FLAGS` for genuine non-patch
env vars (system-tuning, debug, etc.) when needed.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.4.0+ CI guard.
"""
from __future__ import annotations

import re
from pathlib import Path


# Allowlist for env vars that look like GENESIS_ENABLE_* but are
# intentional non-registry knobs (system tuning, debug, etc.).
# Empty at v11.4.0 — every GENESIS_ENABLE_* in YAML must map to a
# registry env_flag.
_KNOWN_NON_REGISTRY_ENV_FLAGS: frozenset[str] = frozenset({
    # No allowlist entries at v11.4.0.
})


_REPO_ROOT = Path(__file__).resolve().parents[3]
_V2_YAML_DIRS = [
    _REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "model",
    _REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "profile",
    _REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "preset",
    _REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "hardware",
]


def _registry_env_flags() -> set[str]:
    """Return the set of canonical env_flag strings declared in
    PATCH_REGISTRY."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    flags: set[str] = set()
    for meta in PATCH_REGISTRY.values():
        if not isinstance(meta, dict):
            continue
        ef = meta.get("env_flag")
        if ef:
            flags.add(ef)
    return flags


def _walk_yaml_for_env_flags(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_no, env_flag_name) for every line in
    `path` that looks like `<indent>GENESIS_ENABLE_<NAME>: '<truthy>'`.

    We intentionally scan raw text instead of parsing YAML — env-flag
    lines appear at multiple nesting depths (model.patches,
    profile.patches_delta.enable, preset.system_env, etc.) and the
    schema is heterogeneous. A regex scan keeps the guard simple
    and decoupled from YAML schema evolution.
    """
    if not path.is_file():
        return []
    findings: list[tuple[int, str]] = []
    flag_re = re.compile(
        r"^\s*(GENESIS_ENABLE_[A-Z0-9_]+)\s*:\s*['\"]?([^'\"\s#]+)"
    )
    try:
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            m = flag_re.match(line)
            if not m:
                continue
            name = m.group(1)
            value = m.group(2).strip()
            # Only flag truthy enables — a `'0'` is an explicit disable
            # and harmless even if name doesn't match a registry entry
            # (defensive).
            if value not in ("1", "true", "True"):
                continue
            findings.append((lineno, name))
    except OSError:
        return []
    return findings


def test_no_v2_yaml_enables_unknown_env_flag():
    """Every truthy `GENESIS_ENABLE_<X>` in V2 builtin YAMLs must
    map to a `env_flag` field in PATCH_REGISTRY. Otherwise the line
    is operator-visible noise that does nothing at boot — Class 7
    silent-ignore bug class."""
    registry_flags = _registry_env_flags()
    offenders: list[tuple[str, int, str]] = []
    for ydir in _V2_YAML_DIRS:
        if not ydir.is_dir():
            continue
        for yf in sorted(ydir.rglob("*.yaml")):
            for lineno, flag in _walk_yaml_for_env_flags(yf):
                if flag in registry_flags:
                    continue
                if flag in _KNOWN_NON_REGISTRY_ENV_FLAGS:
                    continue
                rel = yf.relative_to(_REPO_ROOT)
                offenders.append((str(rel), lineno, flag))
    if offenders:
        lines = [
            f"  {rel}:{lineno}: {flag}"
            for rel, lineno, flag in offenders
        ]
        raise AssertionError(
            f"{len(offenders)} V2 YAML line(s) enable an env flag with no "
            f"matching `env_flag` field in PATCH_REGISTRY. The dispatcher "
            f"reads the EXACT env_flag string — long-form aliases are "
            f"silently ignored, leaving the patch disabled in "
            f"production.\n\n"
            f"Either:\n"
            f"  (a) rename the YAML line to match the registry env_flag "
            f"(canonical fix), or\n"
            f"  (b) add the env name to `_KNOWN_NON_REGISTRY_ENV_FLAGS` "
            f"with a comment explaining why it's a non-patch knob.\n\n"
            f"Offenders:\n" + "\n".join(lines)
        )


def test_known_short_form_env_flags_are_used_in_yamls():
    """Sanity check: P109, PN110, P108 — the short-form env flags
    discovered in the v11.4.0 audit — exist in the registry. Pin the
    expected canonical names so a future env_flag rename triggers a
    test failure here, prompting a YAML sweep.
    """
    flags = _registry_env_flags()
    for canonical in (
        "GENESIS_ENABLE_P108",
        "GENESIS_ENABLE_P109",
        "GENESIS_ENABLE_PN110",
    ):
        assert canonical in flags, (
            f"Registry env_flag {canonical!r} missing — either the patch "
            f"was retired (update this test) or the env_flag was renamed "
            f"(sweep V2 YAMLs to match)."
        )
