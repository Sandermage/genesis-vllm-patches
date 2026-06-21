#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the Genesis Site Map anchor offset manifest (LEGACY GLOBAL fallback).

Node 3 of P2.1 design (2026-05-07). One-shot CLI tool — typically run
once per vllm pin upgrade. Output is committed to the repo at
`sndr/manifests/anchor_manifest.json` and serves as the
ground-truth for Phase 3 runtime O(1) anchor lookup.

RELATIONSHIP TO THE PER-PIN ANCHOR SoT (2026-06-21):
This builds the single GLOBAL fallback manifest — a small hand-curated
STABLE-ratchet set (see `_REGISTRY_TARGETS`) read from committed pristine
fixtures, works fully offline (no rig). It is the LAST-RESORT fallback that
`cached_load_manifest` uses when no per-pin manifest matches the live pin.
The AUTHORITATIVE per-pin manifests (`sndr/engines/vllm/pins/<pin>/anchors.json`,
ALL anchor-bearing patches, generated from the real live engine + bare-image
pristine source) are built by the Phase 4 pipeline `scripts/anchor_sot/` →
`make rebuild-pin`. On a pin bump, regenerate the per-pin manifest with
`make rebuild-pin`; this legacy global generator is only touched when the
hand-curated STABLE set itself changes.

Usage:
    # Default: build for currently registered patchers using pristine
    # fixtures from tests/legacy/pristine_fixtures/.
    python3 scripts/build_anchor_manifest.py

    # Specify pins explicitly (otherwise auto-detect)
    python3 scripts/build_anchor_manifest.py \\
        --vllm-pin "0.20.2rc1.dev9+g01d4d1ad3" \\
        --genesis-pin "v7.72.2"

    # Dry-run (print to stdout, don't write file)
    python3 scripts/build_anchor_manifest.py --dry-run

    # Custom output path
    python3 scripts/build_anchor_manifest.py --output /tmp/manifest.json

Exit codes:
    0  success — manifest written and self-verified
    1  no patchers registered (nothing to do)
    2  schema validation failed
    3  verify_against_source failed (anchor doesn't actually exist
       in pristine — patcher metadata broken)
    4  output path unwritable
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Set up logging BEFORE importing Genesis modules (which use logging)
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("genesis.build_manifest")


# Repo root inferred from script location: scripts/<this> → repo/
REPO_ROOT = Path(__file__).resolve().parent.parent

# Make the repo-root `sndr` package importable (v12 layout).
sys.path.insert(0, str(REPO_ROOT))


def _trigger_patcher_registration():
    """Import wiring modules so they have a chance to register their
    TextPatchers via patcher_registry.

    Convention: each wiring module that opts in calls
    `register_text_patcher(patch_id, patcher)` at module-import time
    OR provides a `register_for_manifest()` callable.

    Discovery list (audit 2026-05-12): patches that participate in the
    STABLE-ratchet anchor manifest. Add new entries as patches mature
    enough for STABLE promotion.
    """
    log.info("triggering patcher registration...")
    # P1-4 (audit 2026-05-08): pristine fixtures live at
    # `tests/legacy/pristine_fixtures/` after the v11 _genesis removal.
    pristine = REPO_ROOT / "tests" / "legacy" / "pristine_fixtures"

    _REGISTRY_TARGETS = [
        ("PN79", "sndr.engines.vllm.patches.attention.gdn"
                 ".pn79_inplace_ssm_state"),
        # Added 2026-05-12 (Wave 9 STABLE-prep): both backports verified
        # default_on across Wave 6–9 + dev93/dev209 + upstream PRs OPEN.
        ("PN35", "sndr.engines.vllm.patches.worker"
                 ".pn35_inputs_embeds_optional"),
        ("PN33", "sndr.engines.vllm.patches.worker"
                 ".pn33_spec_decode_warmup_k"),
        # Added 2026-05-28 (STAGE-6-HARDENING.1): G4_04 stable AWQ MoE
        # keys remap for Gemma 4 26B-A4B; pristine gemma4.py extracted
        # at vllm 0.20.2rc1.dev338+gbf0d2dc6d.
        ("G4_04", "sndr.engines.vllm.patches.model_compat.gemma4"
                  ".g4_04_gemma4_awq_moe_keys_remap"),
    ]

    for pid, mod_path in _REGISTRY_TARGETS:
        try:
            mod = __import__(mod_path, fromlist=["register_for_manifest"])
            register_fn = getattr(mod, "register_for_manifest", None)
        except ImportError as e:
            log.warning("%s wiring import failed: %s", pid, e)
            continue
        if register_fn is None:
            log.warning(
                "%s has no register_for_manifest() — skipping", pid,
            )
            continue
        try:
            register_fn(pristine_root=pristine)
        except Exception as e:
            log.error(
                "%s register_for_manifest raised: %s", pid, e, exc_info=True,
            )


def _detect_genesis_pin() -> str:
    """Read Genesis version from `__version__.py`. Falls back to 'unknown'."""
    try:
        from sndr.version import __version__ as gver
        return str(gver)
    except Exception as e:
        log.warning("genesis pin detection failed: %s", e)
        return "unknown"


def _detect_vllm_pin_from_fixture() -> str:
    """Read declared vllm pin from pristine_fixtures/README.md.

    Format expected: a line containing `vllm: ` followed by version.
    Falls back to 'unknown' if README absent or unparseable. Override
    via --vllm-pin CLI arg.
    """
    # P1-4 (audit 2026-05-08): pristine fixtures dir relocated.
    readme = REPO_ROOT / "tests" / "legacy" \
             / "pristine_fixtures" / "README.md"
    if not readme.is_file():
        return "unknown"
    try:
        for line in readme.read_text().splitlines():
            stripped = line.strip()
            # "vllm: `0.20.2rc1.dev9+g01d4d1ad3`"
            if stripped.startswith("vllm:") or stripped.startswith("vllm: "):
                # Extract value between backticks if present
                if "`" in stripped:
                    parts = stripped.split("`")
                    if len(parts) >= 2:
                        return parts[1]
                # Otherwise everything after the colon
                return stripped.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def _build_file_to_inputs(pristine_root: Path) -> dict:
    """Build the file_to_inputs mapping from registered patchers.

    For each registered TextPatcher, derive its rel_path within vllm
    and pair it with the corresponding pristine fixture content.

    This is the bridge between the runtime registry (which has absolute
    target_file paths pointing at vllm install) and the build-time
    manifest (which uses relative paths + pristine fixture content).
    """
    from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
    from sndr.engines.vllm.wiring.patcher_registry import iter_registered_patchers

    file_to_inputs: dict[str, tuple[str, list]] = {}

    for patch_id, patcher in iter_registered_patchers():
        # Derive rel_path: target_file should already be a relative-style
        # path under vllm tree if patcher was constructed for build-mode.
        # For build mode patchers, target_file points at pristine fixture.
        target = Path(patcher.target_file)
        # Heuristic: take the last segment matching vllm/ structure
        # OR use the filename to find pristine fixture
        rel_path = _derive_rel_path(target, pristine_root)
        if rel_path is None:
            log.warning(
                "[%s] cannot derive rel_path from %s — skipping",
                patch_id, target,
            )
            continue

        # Load pristine source: prefer fixture file matching basename,
        # fall back to actual target_file content
        fixture = pristine_root / target.name
        if fixture.is_file():
            try:
                pristine_src = fixture.read_text(encoding="utf-8")
            except Exception as e:
                log.error("[%s] cannot read fixture %s: %s",
                          patch_id, fixture, e)
                continue
        else:
            log.warning("[%s] no fixture for %s — skipping",
                        patch_id, target.name)
            continue

        # Convert TextPatch sub-patches to PatcherManifestInput tuples
        sub_patches = [
            (sp.name, sp.anchor, sp.replacement)
            for sp in patcher.sub_patches
        ]
        inp = PatcherManifestInput(
            patch_id=patch_id, rel_path=rel_path, sub_patches=sub_patches,
        )

        # Multiple patchers can target the same file — accumulate into list
        if rel_path in file_to_inputs:
            existing_src, existing_list = file_to_inputs[rel_path]
            existing_list.append(inp)
        else:
            file_to_inputs[rel_path] = (pristine_src, [inp])

    return file_to_inputs


# Mapping from fixture filename to canonical relative path under vllm tree.
# As more patches register, expand this mapping.
_KNOWN_REL_PATHS = {
    "chunk.py": "model_executor/layers/fla/ops/chunk.py",
    "chunk_delta_h.py": "model_executor/layers/fla/ops/chunk_delta_h.py",
    "gdn_linear_attn.py": "model_executor/layers/mamba/gdn_linear_attn.py",
    "olmo_hybrid.py": "model_executor/models/olmo_hybrid.py",
    # Added 2026-06-10 (PN79 K.2 re-anchor on pin g303916e93): upstream
    # split mamba/gdn_linear_attn.py into per-model files under mamba/gdn/.
    "qwen_gdn_linear_attn.py":
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "olmo_gdn_linear_attn.py":
        "model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py",
    # Added 2026-05-12 (Wave 9 STABLE-prep for PN33 + PN35):
    "gpu_model_runner.py": "v1/worker/gpu_model_runner.py",
    "llm_base_proposer.py": "v1/spec_decode/llm_base_proposer.py",
    # Added 2026-05-28 (STAGE-6-HARDENING.1 for G4_04):
    "gemma4.py": "model_executor/models/gemma4.py",
}


def _derive_rel_path(target: Path, pristine_root: Path) -> Optional[str]:
    """Derive relative path under vllm tree from a patcher's target_file.

    Strategy:
      1. If target points into pristine_root, look up its basename in
         _KNOWN_REL_PATHS.
      2. If target looks like a real vllm install path
         (`.../site-packages/vllm/<rel>`), strip prefix.
      3. Fallback: just basename → _KNOWN_REL_PATHS lookup.
    """
    # Strategy 1: pristine fixture mode
    try:
        if pristine_root in target.parents:
            return _KNOWN_REL_PATHS.get(target.name)
    except Exception:
        pass

    # Strategy 2: vllm install path
    parts = target.parts
    if "vllm" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("vllm")
        # rel = parts after the LAST `vllm` segment
        rel_parts = parts[idx + 1:]
        if rel_parts:
            return "/".join(rel_parts)

    # Strategy 3: basename lookup
    return _KNOWN_REL_PATHS.get(target.name)


def _default_manifest_output() -> Path:
    """Resolve default manifest output path via sndr_paths registry.

    v12 rename (2026-06-04+): canonical home is `sndr/manifests/`
    (was `vllm/sndr_core/manifests/` in v11; that tree is gone).
    """
    try:
        from sndr.engines.vllm.locations.project_paths import manifest_json_path
        return manifest_json_path()
    except Exception:
        # Bootstrap fallback for very-early script invocation.
        return REPO_ROOT / "sndr" / "manifests" / "anchor_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=_default_manifest_output(),
        help="output JSON path (default resolves via sndr_paths.manifest_json_path())"
    )
    parser.add_argument(
        "--vllm-pin", type=str, default=None,
        help="vllm version string (auto-detect from pristine README)"
    )
    parser.add_argument(
        "--genesis-pin", type=str, default=None,
        help="Genesis version (auto-detect from __version__.py)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print manifest to stdout without writing file"
    )
    parser.add_argument(
        "--pristine-root", type=Path,
        default=REPO_ROOT / "tests" / "legacy" / "pristine_fixtures",
        help="path to pristine fixture directory"
    )
    args = parser.parse_args()

    if not args.pristine_root.is_dir():
        log.error("pristine root not found: %s", args.pristine_root)
        return 4

    # Step 1: trigger patcher registration
    _trigger_patcher_registration()

    # Step 2: detect pins
    vllm_pin = args.vllm_pin or _detect_vllm_pin_from_fixture()
    genesis_pin = args.genesis_pin or _detect_genesis_pin()
    log.info("vllm_pin=%s, genesis_pin=%s", vllm_pin, genesis_pin)

    # Step 3: build file_to_inputs
    from sndr.engines.vllm.wiring.patcher_registry import registered_count
    n = registered_count()
    if n == 0:
        log.error("no patchers registered — nothing to build")
        return 1
    log.info("%d patchers registered", n)

    file_to_inputs = _build_file_to_inputs(args.pristine_root)
    if not file_to_inputs:
        log.error("no file inputs derived from registered patchers")
        return 1
    log.info("%d files contribute to manifest", len(file_to_inputs))

    # Step 4: assemble + validate
    from sndr.engines.vllm.wiring.anchor_manifest import (
        assemble_manifest, validate_manifest_schema,
        verify_manifest_against_source, write_manifest_atomic,
    )

    manifest = assemble_manifest(
        vllm_pin=vllm_pin,
        genesis_pin=genesis_pin,
        file_to_inputs=file_to_inputs,
    )

    schema_errors = validate_manifest_schema(manifest)
    if schema_errors:
        log.error("schema validation failed (%d errors):", len(schema_errors))
        for e in schema_errors[:5]:
            log.error("  - %s", e)
        return 2

    # Step 5: self-verify against source
    def _loader(rel: str) -> Optional[str]:
        src_text, _ = file_to_inputs.get(rel, (None, None))
        return src_text

    verify_errors = verify_manifest_against_source(manifest, _loader)
    if verify_errors:
        log.error(
            "verify_against_source failed (%d errors):", len(verify_errors)
        )
        for e in verify_errors[:5]:
            log.error("  - %s", e)
        return 3

    # Step 6: emit
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        log.info("dry-run: manifest printed to stdout, no file written")
        return 0

    try:
        write_manifest_atomic(args.output, manifest)
    except OSError as e:
        log.error("write failed: %s", e)
        return 4

    # Summary
    total_anchors = sum(
        len(p["anchors"])
        for f in manifest["files"].values()
        for p in f["patches"].values()
    )
    log.info(
        "manifest written to %s — %d files / %d patches / %d anchors",
        args.output, len(manifest["files"]),
        sum(len(f["patches"]) for f in manifest["files"].values()),
        total_anchors,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
