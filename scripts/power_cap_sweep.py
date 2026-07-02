#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Power-cap sweep for a live vLLM engine — find the efficiency knee.

Adapted from the club-3090 power methodology (their cross-class finding: the
knee sits at 60-85% of stock TDP because DECODE is memory-bandwidth-bound, so
GPU-core power caps barely cost decode TPS). This tool measures OUR curve on
OUR hardware instead of trusting anyone's multiplier:

  for each cap: set `nvidia-smi -pl` on every GPU -> run a decode probe against
  the live engine -> record decode TPS + the MEDIAN actual board power sampled
  during generation -> restore the default cap at the end (also on Ctrl-C).

The probe is a SHORT fixed workload — good for the relative shape of the curve
(same methodology at every cap). Iron rule #9 still applies: before promoting a
cap to production, A/B the chosen knee with the canonical genesis_bench_suite.

Requires: passwordless `sudo nvidia-smi -pl` on this host (run rig-side).

Usage:
  python3 scripts/power_cap_sweep.py --caps 230,210,190,170,150,130 \
      --base-url http://127.0.0.1:8102 --api-key genesis-local
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.request

PROBE_PROMPT = (
    "Write a detailed, multi-paragraph explanation of how a transformer language "
    "model generates text token by token, covering attention, the KV cache, and "
    "sampling. Do not use lists; write flowing prose."
)


def _gpus() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [int(x) for x in out.split()]


def _default_limits() -> dict[int, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,power.default_limit", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    limits: dict[int, float] = {}
    for line in out.strip().splitlines():
        idx, lim = line.split(",")
        limits[int(idx)] = float(lim)
    return limits


def _set_cap(gpu: int, watts: float) -> None:
    subprocess.run(
        ["sudo", "-n", "nvidia-smi", "-i", str(gpu), "-pl", str(int(watts))],
        capture_output=True, text=True, check=True,
    )


def _power_draw_total() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return sum(float(x) for x in out.split())


class PowerSampler(threading.Thread):
    """Samples total board power every `period` seconds while running."""

    def __init__(self, period: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.samples: list[float] = []
        self._period = period
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(_power_draw_total())
            except Exception:  # noqa: BLE001 - transient nvidia-smi hiccup
                pass
            self._stop.wait(self._period)

    def stop(self) -> float:
        self._stop.set()
        self.join(timeout=3)
        return statistics.median(self.samples) if self.samples else 0.0


def _probe_decode_tps(base_url: str, api_key: str, model: str,
                      max_tokens: int, runs: int, timeout: float) -> float:
    """Sequential non-streaming completions; decode TPS ~= completion_tokens /
    wall (prefill is negligible against a long generation)."""
    tps: list[float] = []
    for _ in range(runs):
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        dt = time.monotonic() - t0
        tokens = (data.get("usage") or {}).get("completion_tokens") or 0
        if tokens and dt > 0:
            tps.append(tokens / dt)
    return statistics.median(tps) if tps else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Power-cap sweep against a live engine.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8102")
    ap.add_argument("--api-key", default="genesis-local")
    ap.add_argument("--model", default="", help="served model name (auto-detected when empty)")
    ap.add_argument("--caps", default="230,210,190,170,150,130",
                    help="comma-separated watt caps, highest first")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--runs", type=int, default=3, help="probe runs per cap (median)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="", help="write the JSON results here")
    a = ap.parse_args()

    gpus = _gpus()
    defaults = _default_limits()
    caps = [float(c) for c in a.caps.split(",") if c.strip()]

    model = a.model
    if not model:
        req = urllib.request.Request(
            f"{a.base_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {a.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            model = json.load(r)["data"][0]["id"]

    print(f"engine={a.base_url} model={model} gpus={gpus} defaults={defaults}")
    results: list[dict] = []

    def restore() -> None:
        for g in gpus:
            try:
                _set_cap(g, defaults[g])
            except Exception as e:  # noqa: BLE001
                print(f"  WARNING: failed to restore GPU{g} cap: {e}")
        print(f"restored default caps: {defaults}")

    try:
        # Warm-up (compile paths, prefix cache) before any measurement.
        print("warm-up run...")
        _probe_decode_tps(a.base_url, a.api_key, model, a.max_tokens, 1, a.timeout)
        for cap in caps:
            for g in gpus:
                _set_cap(g, cap)
            time.sleep(1.0)  # let the cap engage
            sampler = PowerSampler()
            sampler.start()
            tps = _probe_decode_tps(a.base_url, a.api_key, model, a.max_tokens, a.runs, a.timeout)
            watts = sampler.stop()
            eff = tps / watts if watts else 0.0
            results.append({"cap_w_per_gpu": cap, "decode_tps": round(tps, 2),
                            "board_w_total_median": round(watts, 1),
                            "tps_per_w": round(eff, 4)})
            print(f"cap {int(cap):>3}W/gpu: {tps:7.2f} TPS  |  {watts:6.1f} W total  |  {eff:.4f} TPS/W")
    finally:
        restore()

    if results:
        base = results[0]
        print("\n=== relative to the first (stock) cap ===")
        for r in results:
            dt = (r["decode_tps"] / base["decode_tps"] - 1) * 100 if base["decode_tps"] else 0
            dw = (r["board_w_total_median"] / base["board_w_total_median"] - 1) * 100 if base["board_w_total_median"] else 0
            marker = "  <-- knee candidate" if dt > -2.0 and dw < -10.0 else ""
            print(f"cap {int(r['cap_w_per_gpu']):>3}W: TPS {dt:+6.2f}%  power {dw:+6.2f}%{marker}")
        print("\nknee rule of thumb: deepest power cut with TPS within ~2% of stock.")
        print("Iron rule #9: validate the chosen cap with genesis_bench_suite before promoting.")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"model": model, "gpus": gpus, "defaults": defaults, "results": results}, f, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
