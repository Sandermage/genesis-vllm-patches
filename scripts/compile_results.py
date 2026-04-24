#!/usr/bin/env python3
"""Genesis v7.10 — aggregate per-model validation results into a master summary.

Reads every `<root>/<model_tag>/summary.md` + raw JSONL files, produces one
markdown table with all 4 models × 7 acceptance criteria. Used as PR description
for v7.10 and as the link we share in upstream comments.

Usage:
    python3 scripts/compile_results.py \
        --root benchmarks/v7_10_validation_20260424/ \
        --out  benchmarks/v7_10_validation_20260424/MASTER_SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MODEL_TAGS = [
    "qwen3_next_fp8",
    "qwen3_next_awq",
    "qwen3_32b_dense",
    "gemma4_26b_moe",
    "fp16kv_non_tq",
]

EXPECTED = {
    # Model A: Qwen3-Next-FP8, kv=turboquant_k8v4 (TQ active → P51 no fire)
    "qwen3_next_fp8":  dict(moe=True,  hybrid=True,  tq=True,  p51_fires=False),
    # Model B: Qwen3-Next-AWQ, STILL uses kv=turboquant_k8v4 (TQ active → P51 no fire)
    # Cross-quantization test: same dispatch as A, different weight quant.
    "qwen3_next_awq":  dict(moe=True,  hybrid=True,  tq=True,  p51_fires=False),
    # Model C: RYS-Qwen3.5-27B-FP8-XL dense, kv=auto, no MoE, no hybrid
    # TQ backend NOT instantiated → ensure_turboquant_buffers never called
    # → P51 firing is informational only (guard only fires if TQ impl exists)
    "qwen3_32b_dense": dict(moe=False, hybrid=False, tq=False, p51_fires=False),
    # Model D: Gemma 4 MoE AWQ — skipped, vLLM×model incompatibility
    "gemma4_26b_moe":  dict(moe=True,  hybrid=False, tq=False, p51_fires=False),
    # BONUS: fp16kv (Qwen3-Next with kv=auto) — MoE + hybrid but non-TQ
    # TQ backend not instantiated on kv=auto → P51 doesn't fire (by design)
    "fp16kv_non_tq":   dict(moe=True,  hybrid=True,  tq=False, p51_fires=False),
}


def _read_boot_summary(model_dir: Path) -> str:
    log = model_dir / "boot.log"
    if not log.exists():
        return "❌ no boot.log"
    for line in log.read_text(errors="replace").splitlines():
        if "Genesis Results:" in line:
            return line.strip().split("Genesis Results:")[-1].strip()
    return "❌ no Genesis Results line"


def _read_dispatch_profile(model_dir: Path) -> dict:
    """Read model_detect profile. Prefer in-engine profile (from boot.log
    `[Genesis v7.9 model_detect] profile resolved: ...` line), fall back
    to dispatch_profile.json which is captured post-boot via docker exec
    (often resolves to conservative `{resolved: false, moe: true, ...}`
    because engine context isn't available outside request handling).

    The boot.log line is authoritative: it's emitted from inside engine
    init where get_current_vllm_config() actually returns the real config.
    """
    # First try boot.log resolved line
    boot = model_dir / "boot.log"
    if boot.exists():
        text = boot.read_text(errors="replace")
        for line in text.splitlines():
            if "[Genesis v7.9 model_detect] profile resolved" in line:
                # Parse:  model_type=X moe=Y hybrid=Z turboquant=W (kv=...)
                out: dict = {"resolved": True, "source": "boot.log"}
                for token in line.split():
                    if "=" in token:
                        k, _, v = token.partition("=")
                        if v.lower() in ("true", "false"):
                            out[k] = (v.lower() == "true")
                        else:
                            out[k] = v.strip("()")
                if "moe" in out and "hybrid" in out and "turboquant" in out:
                    return out
    # Fall back to exec dump
    p = model_dir / "dispatch_profile.json"
    if not p.exists():
        return {}
    try:
        text = p.read_text(errors="replace")
        return json.loads(text)
    except Exception:
        return {}


def _count_jsonl_status(path: Path, needle: str = '"http_status": 200') -> tuple[int, int]:
    """Return (pass, total) for a JSONL of results.

    Handles two formats:
    - smoke.jsonl:  {"run":N,"http_status":"200","has_output":1}
    - genesis_context_sweep.py jsonl:
        {"label":...,"prompt_tokens":182069,"ttft_s":...,"error":""}
      Empty error + positive prompt_tokens == success (200 implicit).
    """
    if not path.exists():
        return (0, 0)
    total = 0
    ok = 0
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            rec = json.loads(line)
        except Exception:
            if needle in line:
                ok += 1
            continue
        # smoke.jsonl format
        st = rec.get("http_status")
        if st == 200 or st == "200":
            ok += 1
            continue
        # genesis_context_sweep format: success = empty error AND positive prompt_tokens
        err = rec.get("error")
        pt = rec.get("prompt_tokens", -1)
        if (err == "" or err is None) and isinstance(pt, int) and pt > 0:
            ok += 1
    return (ok, total)


def _count_smoke(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    total = 0
    ok = 0
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            rec = json.loads(line)
            if str(rec.get("http_status")) == "200" and rec.get("has_output", 0) >= 1:
                ok += 1
        except Exception:
            pass
    return (ok, total)


def _read_memory(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}


def _read_speed_100k(path: Path) -> float | None:
    """Extract decode_tok_s from speed_100k.jsonl, return median across runs.

    Uses median (not mean) to reject prefix-cache-hit outliers where
    comp_tokens=1-3 produces artificial spikes (e.g. 4700 t/s) that
    would distort the mean. Also filter comp_tokens < 8 to drop those
    near-empty outputs entirely.
    """
    if not path.exists():
        return None
    vals = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            v = rec.get("decode_tok_s")
            ct = rec.get("comp_tokens", 0)
            if (isinstance(v, (int, float)) and v > 0
                and isinstance(ct, int) and ct >= 8):
                vals.append(float(v))
        except Exception:
            pass
    if not vals:
        return None
    vals.sort()
    mid = len(vals) // 2
    median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
    return round(median, 2)


def _p51_fires(model_dir: Path) -> int:
    log = model_dir / "boot.log"
    if not log.exists():
        return 0
    return len(re.findall(r"\[P51 TQ-active\]", log.read_text(errors="replace")))


def _p52_skips(model_dir: Path) -> int:
    log = model_dir / "boot.log"
    if not log.exists():
        return 0
    text = log.read_text(errors="replace")
    # P52 skip messages contain "dense model" + one of the MoE patch names
    return len(re.findall(
        r"Genesis v7\.9 dispatch.*skipped.*(no MoE|dense model)", text,
    ))


def _p53_skips(model_dir: Path) -> int:
    log = model_dir / "boot.log"
    if not log.exists():
        return 0
    text = log.read_text(errors="replace")
    return len(re.findall(
        r"Genesis v7\.9 dispatch.*skipped.*(pure-attention|no GDN|no hybrid)", text,
    ))


def _grade(got: bool) -> str:
    return "✅" if got else "❌"


def _model_row(root: Path, tag: str) -> dict:
    d = root / tag
    exp = EXPECTED[tag]
    profile = _read_dispatch_profile(d)

    boot = _read_boot_summary(d)
    smoke_ok, smoke_total = _count_smoke(d / "smoke.jsonl")
    sweep_ok, sweep_total = _count_jsonl_status(d / "context_sweep_full.jsonl")
    stress_ok, stress_total = _count_jsonl_status(d / "stress_probe_m.jsonl")
    speed = _read_speed_100k(d / "speed_100k.jsonl")
    mem = _read_memory(d / "memory_profile.json")

    p51_n = _p51_fires(d)
    p52_n = _p52_skips(d)
    p53_n = _p53_skips(d)

    # Dispatch grade: profile matches EXPECTED + P51 firing pattern matches
    prof_ok = False
    if profile:
        prof_ok = (
            bool(profile.get("moe")) == exp["moe"]
            and bool(profile.get("hybrid")) == exp["hybrid"]
            and bool(profile.get("turboquant")) == exp["tq"]
        )
    p51_ok = (p51_n > 0) == exp["p51_fires"]

    return dict(
        tag=tag,
        boot_summary=boot,
        smoke=f"{smoke_ok}/{smoke_total}",
        smoke_ok=(smoke_total > 0 and smoke_ok == smoke_total),
        sweep=f"{sweep_ok}/{sweep_total}",
        sweep_ok=(sweep_total > 0 and sweep_ok == sweep_total),
        stress=f"{stress_ok}/{stress_total}",
        stress_ok=(stress_total > 0 and stress_ok == stress_total),
        speed=(f"{speed} t/s" if speed else "N/A"),
        mem_delta=mem.get("delta_mib", "N/A"),
        profile_detected=(
            f"moe={profile.get('moe')} hybrid={profile.get('hybrid')} "
            f"tq={profile.get('turboquant')}"
            if profile else "NO PROFILE"
        ),
        profile_ok=prof_ok,
        p51=f"{p51_n} fires (exp: {'yes' if exp['p51_fires'] else 'no'})",
        p51_ok=p51_ok,
        p52=str(p52_n),
        p53=str(p53_n),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Validation root dir")
    ap.add_argument("--out",  required=True, help="Output markdown path")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 1

    rows = []
    for tag in MODEL_TAGS:
        if (root / tag).exists():
            rows.append(_model_row(root, tag))
        else:
            rows.append(dict(tag=tag, boot_summary="⏳ not run yet"))

    out = args.out
    lines: list[str] = []
    lines.append("# Genesis v7.10 — Master Validation Summary")
    lines.append("")
    lines.append("Auto-generated by `scripts/compile_results.py`. One row per model × acceptance criterion.")
    lines.append("")
    lines.append(f"**Root**: `{root}`")
    lines.append("")
    lines.append("## Per-model results")
    lines.append("")
    lines.append(
        "| Model | Boot | Smoke | Sweep | Stress | Speed | Mem Δ | "
        "Profile | P51 | P52 skips | P53 skips |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        if r.get("boot_summary", "").startswith("⏳"):
            lines.append(
                f"| **{r['tag']}** | ⏳ not run | — | — | — | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| **{r['tag']}** "
            f"| {r['boot_summary']} "
            f"| {_grade(r['smoke_ok'])} {r['smoke']} "
            f"| {_grade(r['sweep_ok'])} {r['sweep']} "
            f"| {_grade(r['stress_ok'])} {r['stress']} "
            f"| {r['speed']} "
            f"| {r['mem_delta']} MiB "
            f"| {_grade(r['profile_ok'])} {r['profile_detected']} "
            f"| {_grade(r['p51_ok'])} {r['p51']} "
            f"| {r['p52']} "
            f"| {r['p53']} |"
        )

    lines.append("")
    lines.append("## Acceptance gate")
    lines.append("")
    lines.append(
        "All rows must show ✅ in Smoke / Sweep / Stress / Profile / P51 columns "
        "before v7.10 release is approved. P52/P53 numeric counts are informational — "
        "what matters is whether the dispatch gate fired at the expected layers "
        "(see each model's `apply_all.log`)."
    )
    lines.append("")
    lines.append("## Regression baseline (speed)")
    lines.append("")
    lines.append("| Model | v7.8.5 baseline | v7.10 measured | Delta |")
    lines.append("|---|---|---|---|")
    baselines = {
        "qwen3_next_fp8":  32.0,
        "qwen3_next_awq":  31.8,
        "qwen3_32b_dense": None,
        "gemma4_26b_moe":  None,
    }
    for r in rows:
        b = baselines.get(r["tag"])
        if b is None or r.get("speed", "N/A") == "N/A":
            lines.append(f"| {r['tag']} | — | {r.get('speed','N/A')} | — |")
            continue
        try:
            measured = float(r["speed"].split()[0])
            delta = round((measured - b) / b * 100, 2)
            sign = "+" if delta >= 0 else ""
            ok = "✅" if delta >= -3 else "❌"
            lines.append(
                f"| {r['tag']} | {b} t/s | {r['speed']} | {sign}{delta}% {ok} |"
            )
        except Exception:
            lines.append(f"| {r['tag']} | {b} t/s | {r.get('speed','N/A')} | parse error |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Regenerate with:")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python3 scripts/compile_results.py --root {root} --out {out}")
    lines.append("```")

    Path(out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
