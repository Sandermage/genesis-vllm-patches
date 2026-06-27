# SPDX-License-Identifier: Apache-2.0
"""Genesis models pull — `python3 -m sndr.compat.models.pull <key>`.

Downloads a registered model from HuggingFace, verifies it, and
generates a personalized launch script that engages the right Genesis
patches for the user's hardware.

Workflow:
  1. Pre-flight: disk space, HF reachability, optional HF token
  2. Download via huggingface_hub (resume-capable, robust)
  3. Verify (file count + sizes; SHA optional if pinned)
  4. Generate launch script in scripts/launch/start_<key>_<workload>.sh
  5. Print recommended next steps

Usage:
  python3 -m sndr.compat.models.pull qwen3_6_27b_int4_autoround
  python3 -m sndr.compat.models.pull qwen3_6_27b_int4_autoround \
      --models-dir ~/models \
      --workload long_ctx_tool_call \
      --tp 2

Env overrides:
  SNDR_MODELS_DIR          — where to download models
  GENESIS_MODELS_DIR       — legacy alias for SNDR_MODELS_DIR
  HF_TOKEN                 — for gated repos
  HUGGINGFACE_HUB_CACHE    — standard HF lib override

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger("genesis.compat.models.pull")


# ─── Pre-flight ───────────────────────────────────────────────────────────


def _resolve_models_dir(override: str | None = None) -> Path:
    """Resolve where to put downloaded models.

    Precedence:
      1. CLI --models-dir (passed in `override`)
      2. project_paths.models_dir() — single source of truth for
         SNDR_MODELS_DIR / GENESIS_MODELS_DIR + default fallback
      3. HUGGINGFACE_HUB_CACHE env (HF Hub default)
      4. ~/.cache/huggingface/hub (HF default)

    2026-05-11 audit F-013 closure: this helper used to duplicate the
    env-var lookup logic from project_paths.models_dir(). Now delegates
    to the canonical helper, only adding HF Hub-specific fallback for
    pull-time (project_paths default is `/models` for serving — for
    pull-time download into HF cache the fallback chain differs).
    """
    if override:
        return Path(override).expanduser().resolve()
    # Canonical SNDR_MODELS_DIR / GENESIS_MODELS_DIR via project_paths.
    # Only honor it when explicitly set (env present) — the default
    # value for pull-time differs (HF cache, not /models).
    if os.environ.get("SNDR_MODELS_DIR") or os.environ.get("GENESIS_MODELS_DIR"):
        from sndr.engines.vllm.locations.project_paths import models_dir
        return models_dir().expanduser().resolve()
    env_hf = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_hf:
        return Path(env_hf).expanduser().resolve()
    return Path("~/.cache/huggingface/hub").expanduser()


def _check_disk_space(target_dir: Path, needed_gb: float, headroom: float = 1.2) -> tuple[bool, str]:
    """Verify the target dir has enough free space (with `headroom` factor)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    stat = shutil.disk_usage(target_dir)
    free_gb = stat.free / 1e9
    need_with_headroom = needed_gb * headroom
    if free_gb < need_with_headroom:
        return False, (
            f"insufficient disk: {free_gb:.1f} GB free at {target_dir}, "
            f"need ~{need_with_headroom:.1f} GB (model {needed_gb:.1f} GB × {headroom:.1f})"
        )
    return True, f"{free_gb:.1f} GB free at {target_dir} (need ~{need_with_headroom:.1f} GB)"


def _check_hf_reachable() -> tuple[bool, str]:
    """Best-effort connectivity check to huggingface.co."""
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True, "huggingface.co reachable"
    except Exception as e:
        return False, f"huggingface.co not reachable: {e}"


def _check_hf_token_for_gated(model_entry) -> tuple[bool, str]:
    """If the model is gated, verify HF_TOKEN is set."""
    if not model_entry.gated:
        return True, "public repo (no token required)"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return False, (
            "gated model — set HF_TOKEN env var or run `huggingface-cli login` "
            "first. Visit the model card to request access."
        )
    return True, "HF_TOKEN present"


# ─── Download ─────────────────────────────────────────────────────────────


def download_model(
    model_entry,
    models_dir: Path,
    *,
    revision: str | None = None,
    progress: bool = True,
) -> Path:
    """Download via huggingface_hub.snapshot_download. Returns local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Run: "
            "pip install huggingface_hub"
        )

    rev = revision or model_entry.hf_revision
    log.info("Downloading %s (revision=%s) to %s",
             model_entry.hf_id, rev or "latest", models_dir)

    local_path = snapshot_download(
        repo_id=model_entry.hf_id,
        revision=rev,
        cache_dir=str(models_dir),
        # Allow resume on partial downloads
        resume_download=True,
        # Skip files we don't need (tokenizer files we always need;
        # safetensors over .bin where both exist)
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "tf_*", "flax_*"],
        local_files_only=False,
    )
    return Path(local_path)


def _verify_download(local_path: Path, model_entry) -> tuple[bool, str]:
    """Sanity-check that essential files are present + non-empty."""
    if not local_path.is_dir():
        return False, f"download path not a directory: {local_path}"
    safetensors = list(local_path.rglob("*.safetensors"))
    if not safetensors:
        return False, "no .safetensors files found in downloaded path"
    config = list(local_path.glob("config.json"))
    if not config:
        return False, "config.json missing"
    total_gb = sum(p.stat().st_size for p in safetensors) / 1e9
    expected_gb = model_entry.size_gb
    drift = abs(total_gb - expected_gb) / expected_gb if expected_gb else 0.0
    if drift > 0.20:  # 20% tolerance for community quants that re-roll
        return False, (
            f"size drift {drift*100:.0f}% — got {total_gb:.1f} GB, "
            f"expected ~{expected_gb:.1f} GB. Could be a different version."
        )
    return True, f"verified ({len(safetensors)} shards, {total_gb:.1f} GB)"


# ─── Launch script generation ─────────────────────────────────────────────


# ─── B2: pre-download fit verdict ("will model X fit my card?") ─────────────


def _resolve_card_vram_gib(card, fake_gpus):
    """Resolve a per-card VRAM in GiB for the fit gate: --card > --fake-gpus >
    live nvidia-smi probe. Returns (vram_gib, source) or (None, reason).

    Uses the smallest card's MiB at full precision (the binding TP constraint),
    mirroring kv_calc._precise_vram_gib so the pre-download verdict and the
    post-config `sndr kv-calc` verdict agree on the same VRAM basis.
    """
    if card is not None:
        try:
            return float(card), f"card:{card}GB"
        except (TypeError, ValueError):
            return None, "card:invalid"

    def _min_gib(rig):
        gpus = getattr(rig, "gpus", None)
        if not gpus:
            return None
        mib = min(g.vram_mib for g in gpus if g.vram_mib)
        return (mib / 1024.0) if mib else None

    if fake_gpus is not None:
        from sndr.model_configs.preflight_fit import rig_from_fake_spec
        return _min_gib(rig_from_fake_spec(fake_gpus)), "fake"
    try:
        from sndr.model_configs.preflight_fit import RigProbe
        gib = _min_gib(RigProbe().detect())
    except Exception:  # noqa: BLE001 — no nvidia-smi / probe failed
        gib = None
    return (gib, "nvidia-smi") if gib else (None, "no card detected")


def _fit_verdict_for_entry(entry, vram_gib, tp):
    """PASS / TIGHT / FAIL whether ``entry`` fits a ``vram_gib``-per-card rig at
    tensor-parallel size ``tp``, from the entry's declared
    ``min_vram_gb_per_rank`` envelope floor. Returns (verdict, detail).

    This is the ENVELOPE-level pre-download gate — it answers "will this even
    fit before I spend 30 minutes downloading 38 GB?" from data every registry
    entry already carries. The BYTE-level question (exact KV pool, OOM at full
    ctx) is answered post-config by ``sndr kv-calc`` against a V2 preset; this
    gate deliberately stays at the cheaper, always-available envelope tier.

    Thresholds: FAIL when the card is below the declared per-rank floor (vLLM
    won't boot); TIGHT within a 10% band above the floor (boots but little
    headroom for KV growth / fragmentation); PASS otherwise.
    """
    floors = dict(entry.min_vram_gb_per_rank or {})
    if not floors:
        return "UNKNOWN", ("model declares no min_vram_gb_per_rank — cannot "
                           "envelope-fit; run `sndr kv-calc` post-config")
    # Pick the floor for the requested TP, else the smallest available TP's
    # floor (single-card-first: the most demanding per-rank number).
    if tp in floors:
        floor, use_tp = floors[tp], tp
    else:
        use_tp = min(floors)
        floor = floors[use_tp]
    margin = vram_gib - floor
    pct = (margin / floor * 100.0) if floor else 0.0
    basis = (f"needs >={floor:.1f} GiB/rank at TP{use_tp}; "
             f"card has {vram_gib:.1f} GiB ({margin:+.1f} GiB, {pct:+.0f}%)")
    if margin < 0:
        return "FAIL", (
            f"{basis} — below the per-rank floor; vLLM will OOM at boot. "
            f"Use a larger card, raise TP (more cards split the model), or "
            f"pick a smaller quant.")
    if margin < 0.10 * floor:
        return "TIGHT", (
            f"{basis} — clears the floor but with little headroom for KV "
            f"growth / fragmentation. Run `sndr kv-calc` for the byte-level "
            f"OOM check before committing a long-ctx workload.")
    return "PASS", f"{basis} — clears the per-rank floor with headroom"


def _select_config(model_entry, workload: str | None, tp: int | None):
    """Pick a TestedConfig based on workload preference + TP."""
    configs = list(model_entry.tested_configs)
    if not configs:
        return None
    if workload:
        for c in configs:
            if workload in c.name.lower().replace(" ", "_") \
                    or workload in c.name.lower():
                return c
    if tp:
        for c in configs:
            if c.tensor_parallel_size == tp:
                return c
    return configs[0]  # default to first


def generate_launch_script(
    model_entry,
    config,
    local_model_path: Path,
    out_path: Path,
) -> None:
    """Write a bash launch script tailored to (model, config, hardware)."""
    served_name = model_entry.key.replace("_", "-")
    env_lines = []
    for patch_id in config.recommended_genesis_patches:
        # Convert patch_id "P67" → env flag (look up in PATCH_REGISTRY).
        try:
            from sndr.dispatcher import PATCH_REGISTRY
            meta = PATCH_REGISTRY.get(patch_id)
            if meta and meta.get("env_flag"):
                env_lines.append(f"  -e {meta['env_flag']}=1 \\")
        except Exception:
            # Fallback: best-guess name
            env_lines.append(f"  -e GENESIS_ENABLE_{patch_id.upper()}=1 \\")

    spec_json = "null"
    if config.speculative_config:
        import json
        spec_json = json.dumps(config.speculative_config)
    spec_arg = (
        f'  --speculative-config \'{spec_json}\' \\\n'
        if config.speculative_config else ""
    )

    additional = ""
    if config.additional_args:
        additional = "\n".join(f"  {a} \\" for a in config.additional_args) + "\n"

    quirks_block = ""
    if model_entry.quirks:
        quirks_block = (
            "# ───────────────────────────────────────────────────────────\n"
            "# Known quirks for this model:\n"
        )
        for q in model_entry.quirks:
            quirks_block += f"#   - {q}\n"
        quirks_block += "# ───────────────────────────────────────────────────────────\n\n"

    expected_block = ""
    if config.expected:
        e = config.expected
        expected_block = (
            f"# Expected metrics on {e.hardware_class} (captured {e.captured_at}):\n"
            f"#   wall_TPS  ≈ {e.wall_tps_median}  (CV ~5%)\n"
            f"#   TPOT      ≈ {e.decode_tpot_ms} ms\n"
            f"#   TTFT      ≈ {e.ttft_ms} ms\n"
            f"#   VRAM      ≈ {e.vram_gb_per_rank} GB per rank\n"
            f"#   tool-call ≈ {e.tool_call_pass_rate * 100:.0f}% pass rate\n"
            "#\n"
        )

    cache_pref_arg = " --enable-prefix-caching" if config.enable_prefix_caching else ""

    script = f"""#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
# Genesis launch script for {model_entry.key}
# Workload: {config.name}
# vLLM pin: {config.vllm_pin}
# Generated by: python3 -m sndr.compat.models.pull
# ════════════════════════════════════════════════════════════════════════

{quirks_block}{expected_block}set -euo pipefail
docker stop vllm-{served_name} 2>/dev/null || true
docker rm   vllm-{served_name} 2>/dev/null || true

docker run -d \\
  --name vllm-{served_name} \\
  --shm-size=8g --memory=64g -p 8000:8000 --gpus all \\
  --security-opt label=disable --entrypoint /bin/bash \\
  -v {local_model_path}:/models/{model_entry.key}:ro \\
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:ro \\
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_LOGGING_LEVEL=WARNING \\
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \\
{chr(10).join(env_lines)}
  vllm/vllm-openai:nightly -c \\
  "set -e; \\
python3 -m sndr.apply ; \\
exec vllm serve --model /models/{model_entry.key} \\
  --tensor-parallel-size {config.tensor_parallel_size} \\
  --gpu-memory-utilization {config.gpu_memory_utilization} \\
  --max-model-len {config.max_model_len} \\
  --max-num-seqs {config.max_num_seqs} \\
  --max-num-batched-tokens {config.max_num_batched_tokens} \\
  --kv-cache-dtype {config.kv_cache_dtype}{cache_pref_arg} \\
{spec_arg}{additional}\\
  --trust-remote-code --language-model-only \\
  --served-model-name {served_name} \\
  --host 0.0.0.0 --port 8000 --disable-log-stats"

sleep 5
docker logs --tail 5 vllm-{served_name} 2>&1 | sed "s/^/  /"
echo "[{model_entry.key}] container started; tail logs with: docker logs -f vllm-{served_name}"
"""
    out_path.write_text(script)
    out_path.chmod(0o755)


# ─── B4 + Y3 wire-in: pull from cfg.artifacts.models ─────────────────────


def pull_via_artifacts(
    cfg_key: str,
    *,
    models_dir: str | None = None,
    dry_run: bool = False,
) -> int:
    """B4 + Y3 (UNIFIED_CONFIG plan 2026-05-09): pull every model
    declared in `cfg.artifacts.models` instead of going through the
    legacy `compat.models.registry`.

    For each artifact:
      1. Resolve target path (from artifact.local_dir + models_dir override)
      2. If already present and `verify()` returns no problems → skip
      3. Otherwise call `huggingface-cli download <hf_id> --local-dir ...`
      4. Re-verify after pull; report any remaining problems

    Returns 0 on success, 1 if any artifact failed verify after pull,
    2 on bad inputs (unknown cfg_key / no artifacts block).
    """
    try:
        from sndr.model_configs.registry import get
    except ImportError:
        print("ERROR: model_configs registry not importable", file=sys.stderr)
        return 2
    cfg = get(cfg_key)
    # V1 registry is empty post-Phase-10 sunset; resolve V2 aliases (e.g. the
    # llama.cpp GGUF lane `llamacpp-qwen3.6-27b-q4km-1x`) through registry_v2.
    # compose() now carries `cfg.artifacts` onto the composed ModelConfig, so
    # the artifacts pull path works identically for both registries.
    if cfg is None:
        try:
            from sndr.model_configs.registry_v2 import load_alias
            cfg = load_alias(cfg_key)
        except Exception:
            cfg = None
    if cfg is None:
        print(f"ERROR: unknown preset key {cfg_key!r}", file=sys.stderr)
        avail: list[str] = []
        try:
            from sndr.model_configs.registry import list_keys
            avail.extend(list_keys())
        except Exception:
            pass
        try:
            from sndr.model_configs.registry_v2 import list_presets
            avail.extend(list_presets())
        except Exception:
            pass
        if avail:
            print(f"available: {', '.join(sorted(set(avail)))}",
                  file=sys.stderr)
        return 2
    if cfg.artifacts is None or not cfg.artifacts.models:
        print(
            f"ERROR: preset {cfg_key!r} has no `artifacts.models` block. "
            f"Add Y3 artifacts schema to the YAML or use legacy "
            f"`pull <model_key>` mode.",
            file=sys.stderr,
        )
        return 2

    print("=" * 64)
    print(f"Genesis model pull (Y3 artifacts) — preset {cfg_key!r}")
    print("=" * 64)
    print(f"  models declared: {len(cfg.artifacts.models)}")
    if dry_run:
        print("  DRY-RUN — no actual downloads")
    print()

    failed = 0
    for i, art in enumerate(cfg.artifacts.models, 1):
        target = art.local_dir
        if models_dir:
            from pathlib import Path
            target = str(Path(models_dir) / Path(art.local_dir).name)
        kind = getattr(art, "kind", "hf-dir")
        print(f"  [{i}/{len(cfg.artifacts.models)}] {art.hf_id}")
        print(f"    kind:           {kind}")
        if kind == "gguf-file":
            print(f"    filename:       {art.filename}")
        print(f"    revision:       {art.revision}")
        print(f"    local_dir:      {target}")
        print(f"    gated:          {art.gated}")
        if kind != "gguf-file":
            print(f"    required_files: {art.required_files}")

        # Pre-pull verify (skip download if already present)
        problems = art.verify(base_path=target)
        if not problems:
            print("    ✓ already complete — skip pull")
            continue
        print(f"    pre-pull problems: {problems}")

        if dry_run:
            if kind == "gguf-file":
                print(
                    f"    [dry-run] would: hf_hub_download("
                    f"repo_id={art.hf_id!r}, filename={art.filename!r})"
                )
            else:
                print(f"    [dry-run] would: huggingface-cli download {art.hf_id}")
            continue

        # Real pull
        import subprocess
        from pathlib import Path
        Path(target).expanduser().mkdir(parents=True, exist_ok=True)
        if kind == "gguf-file":
            # Single-file GGUF: fetch ONLY the one .gguf via hf_hub_download
            # (snapshot_download would pull the whole repo). The file lands at
            # local_dir/filename so the llama-server `-m` path resolves.
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                print("    ✗ huggingface_hub not installed "
                      "(pip install huggingface_hub)")
                failed += 1
                continue
            try:
                hf_hub_download(
                    repo_id=art.hf_id,
                    filename=art.filename,
                    revision=(art.revision if art.revision else None),
                    local_dir=str(Path(target).expanduser()),
                )
            except Exception as e:  # noqa: BLE001 — surface any fetch failure
                print(f"    ✗ gguf fetch failed: {type(e).__name__}: {e}")
                failed += 1
                continue
            problems = art.verify(base_path=target)
            if problems:
                print(f"    ⚠ post-pull verify problems: {problems}")
                failed += 1
            else:
                print("    ✓ verify OK")
            continue
        cmd = [
            "huggingface-cli", "download", art.hf_id,
            "--local-dir", str(Path(target).expanduser()),
            "--local-dir-use-symlinks", "False",
        ]
        if art.revision and art.revision != "main":
            cmd.extend(["--revision", art.revision])
        print(f"    running: {' '.join(cmd[:4])} ...")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=3600)
            if r.returncode != 0:
                print(f"    ✗ pull failed (rc={r.returncode}): {r.stderr[-200:]}")
                failed += 1
                continue
        except subprocess.TimeoutExpired:
            print("    ✗ pull timed out after 1h")
            failed += 1
            continue
        except FileNotFoundError:
            print("    ✗ huggingface-cli not on PATH (pip install huggingface_hub)")
            failed += 1
            continue

        # Post-pull verify
        problems = art.verify(base_path=target)
        if problems:
            print(f"    ⚠ post-pull verify problems: {problems}")
            failed += 1
        else:
            print("    ✓ verify OK")

    print()
    print("=" * 64)
    if failed == 0:
        print(f"  Genesis model pull: all {len(cfg.artifacts.models)} artifacts OK")
        return 0
    else:
        print(f"  Genesis model pull: {failed} of {len(cfg.artifacts.models)} FAILED")
        return 1


# ─── CLI ─────────────────────────────────────────────────────────────────


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python3 -m sndr.compat.models.pull",
        description="Download a Genesis-supported model from HuggingFace + "
                    "generate a launch script tailored to the chosen workload.",
    )
    p.add_argument("model_key", nargs="?", default=None,
                   help="model key from `genesis list-models` (optional when --config used)")
    p.add_argument("--models-dir", default=None,
                   help="Where to put weights (default: SNDR_MODELS_DIR / "
                        "GENESIS_MODELS_DIR / HUGGINGFACE_HUB_CACHE / ~/.cache/huggingface/hub)")
    p.add_argument("--workload", default=None,
                   help="Workload preference: long_ctx_tool_call / interactive / throughput")
    p.add_argument("--tp", type=int, default=None,
                   help="Tensor parallel size override")
    p.add_argument("--launch-out", default="scripts/launch/",
                   help="Directory to write the generated launch script")
    p.add_argument("--no-launch", action="store_true",
                   help="Skip launch-script generation (just download)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print pre-flight + plan + fit verdict, do not download")
    p.add_argument("--card", default=None, metavar="VRAM_GB",
                   help="Per-card VRAM in GiB for the dry-run fit verdict "
                        "(e.g. --card 24). Default: live nvidia-smi probe.")
    p.add_argument("--fake-gpus", default=None, metavar="SPEC",
                   help="Synthetic rig for the fit verdict, club-3090 style "
                        "'name:vram_mib:cc;...'. Offline; overrides the probe.")
    p.add_argument("--revision", default=None,
                   help="HF revision (commit/tag) override")
    p.add_argument("--hf-id-override", default=None,
                   help="Override the registry's hf_id (e.g. use Lorbus's "
                        "Qwen3.6-27B variant instead of Intel's). Use the "
                        "exact 'org/repo' string accepted by huggingface_hub.")
    # B4 + Y3 wire-in (UNIFIED_CONFIG plan 2026-05-09): alternate mode
    # that pulls models declared in `cfg.artifacts.models` instead of
    # the legacy compat.models.registry. When --config is set, model_key
    # becomes optional (we pull ALL artifacts in the config).
    p.add_argument("--config", default=None,
                   help="B4: model_config preset key (e.g. a5000-2x-35b-prod). "
                        "When set, pulls all models declared in "
                        "cfg.artifacts.models instead of using the legacy "
                        "compat.models.registry. model_key becomes optional.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    # B4 + Y3 wire-in (UNIFIED_CONFIG plan 2026-05-09): when --config is set,
    # delegate to the artifacts-based puller and bypass the legacy
    # compat.models.registry entirely. This is the supported path going
    # forward — the registry path stays for back-compat with old tooling.
    if args.config:
        return pull_via_artifacts(
            cfg_key=args.config,
            models_dir=args.models_dir,
            dry_run=args.dry_run,
        )

    if not args.model_key:
        print("ERROR: must supply either model_key OR --config <preset-key>",
              file=sys.stderr)
        return 2

    from sndr.compat.models.registry import get_model

    entry = get_model(args.model_key)
    if entry is None:
        print(f"unknown model key: {args.model_key!r}", file=sys.stderr)
        print("Run `python3 -m sndr.compat.models.list` to see available models.",
              file=sys.stderr)
        return 2

    # --hf-id-override: replace the registry's hf_id with operator-supplied
    # alternate (e.g. Lorbus/* instead of Intel/*). Recipe metadata
    # (size, quant_format, expected metrics) carries over from the
    # registry entry — operator's responsibility to ensure the override
    # is shape-compatible.
    if args.hf_id_override:
        from dataclasses import replace
        entry = replace(entry, hf_id=args.hf_id_override)
        print(f"[hf-id-override] using {args.hf_id_override!r} "
              f"(registry default: {get_model(args.model_key).hf_id!r})")

    print("=" * 64)
    print(f"Genesis model pull — {entry.title}")
    print("=" * 64)
    print(f"  HF id: {entry.hf_id}")
    print(f"  Size:  {entry.size_gb:.1f} GB")
    print(f"  Quant: {entry.quant_format}")
    print(f"  Status: {entry.status}")

    if entry.status == "PLANNED":
        print("\n[!] This model is PLANNED — not yet validated.")
        print("    Genesis won't auto-block but expect rough edges.")

    models_dir = _resolve_models_dir(args.models_dir)
    print("\n[1/4] Pre-flight checks")

    ok_disk, msg_disk = _check_disk_space(models_dir, entry.size_gb)
    print(f"  disk:    {'✓' if ok_disk else '✗'} {msg_disk}")
    # A dry-run is a pre-download PLAN: surface the ✗ but do not abort, so the
    # operator still reaches the fit verdict below (the whole point of --dry-run
    # is to learn "will this fit?" BEFORE freeing disk / downloading N GB). Same
    # leniency as the network check. A real download still hard-fails here.
    if not ok_disk and not args.dry_run:
        return 3

    ok_net, msg_net = _check_hf_reachable()
    print(f"  network: {'✓' if ok_net else '✗'} {msg_net}")
    if not ok_net and not args.dry_run:
        return 3

    ok_tok, msg_tok = _check_hf_token_for_gated(entry)
    print(f"  token:   {'✓' if ok_tok else '✗'} {msg_tok}")
    # Dry-run leniency (see disk note above): a missing gated-repo token must not
    # hide the fit verdict — the operator can fix the token before the real pull.
    if not ok_tok and not args.dry_run:
        return 3

    if args.dry_run:
        print("\n[dry-run] would download to:", models_dir)
        cfg = _select_config(entry, args.workload, args.tp)
        if cfg:
            print(f"[dry-run] would generate launch script for: {cfg.name}")

        # B2 fit gate: will this even fit my card BEFORE I download N GB?
        fit_tp = args.tp or (cfg.tensor_parallel_size if cfg else 1)
        vram_gib, vram_src = _resolve_card_vram_gib(args.card, args.fake_gpus)
        print("\n[dry-run] fit verdict:")
        if vram_gib is None:
            print(f"    ? UNKNOWN — {vram_src}; pass --card <GB> or --fake-gpus "
                  "to envelope-fit before download")
        else:
            verdict, detail = _fit_verdict_for_entry(entry, vram_gib, fit_tp)
            glyph = {"PASS": "✓", "TIGHT": "!", "FAIL": "✗"}.get(verdict, "?")
            print(f"    {glyph} {verdict} ({vram_src}, TP{fit_tp}) — {detail}")
            print("    (envelope check; run `sndr kv-calc` post-config for the "
                  "byte-level OOM verdict)")
        return 0

    print(f"\n[2/4] Downloading {entry.hf_id}")
    try:
        local_path = download_model(entry, models_dir, revision=args.revision)
    except Exception as e:
        print(f"  ✗ download failed: {e}", file=sys.stderr)
        return 4

    print(f"  ✓ downloaded to {local_path}")

    print("\n[3/4] Verify download")
    ok_v, msg_v = _verify_download(local_path, entry)
    print(f"  {'✓' if ok_v else '⚠'} {msg_v}")
    if not ok_v:
        print("  (download retained in case operator wants to inspect manually)")

    if args.no_launch:
        print("\nSkipping launch-script generation (--no-launch).")
        print(f"\nModel ready at: {local_path}")
        return 0

    print("\n[4/4] Generate launch script")
    cfg = _select_config(entry, args.workload, args.tp)
    if cfg is None:
        print("  ⚠ no tested config available — skipping launch-script generation")
        return 0

    out_dir = Path(args.launch_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    workload_slug = cfg.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_file = out_dir / f"start_{entry.key}_{workload_slug}.sh"
    generate_launch_script(entry, cfg, local_path, out_file)
    print(f"  ✓ {out_file}")

    print()
    print("=" * 64)
    print("Next steps:")
    print(f"  bash {out_file}")
    if entry.quirks:
        print()
        print("Heads-up — known quirks for this model:")
        for q in entry.quirks:
            print(f"  - {q}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
