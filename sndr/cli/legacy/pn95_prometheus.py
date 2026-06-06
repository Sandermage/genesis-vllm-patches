"""PN95 Prometheus exporter — standalone HTTP server that exposes the
PN95 stats JSON file as Prometheus text-exposition metrics.

Usage:
    python3 -m vllm.sndr_core.cli.pn95_prometheus
    python3 -m vllm.sndr_core.cli.pn95_prometheus --port 9192 --path /tmp/pn95_stats.json
    python3 -m vllm.sndr_core.cli.pn95_prometheus --once  # one-shot stdout dump

Pulls stats from the JSON dump that the in-process scheduler tick
writes every GENESIS_PN95_STATS_INTERVAL ticks (default ~100). Refreshes
on every /metrics scrape. No torch / vllm imports — runs on any host
that can read the file (sidecar pattern: deploy as a separate container
sharing the stats-file volume with the inference container).

Metrics naming follows Prometheus best practice:
  - `_total` for monotonic counters
  - `_bytes` / `_count` for current-state gauges
  - all metrics share the `pn95_` namespace

No external dependencies required (uses stdlib http.server). The
prometheus_client library would give slightly nicer label semantics
but adds an install dep we don't want for a simple sidecar.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# (metric_name, stats_key, prometheus_type, help_text)
_METRIC_MAP = [
    # Scheduler tick counters
    ("pn95_ticks_total",
     "ticks_total", "counter",
     "Total scheduler ticks observed by PN95"),
    ("pn95_ticks_pressure_check_total",
     "ticks_pressure_check", "counter",
     "Ticks where GPU free-mem dropped below demote threshold"),
    ("pn95_ticks_demote_triggered_total",
     "ticks_demote_triggered", "counter",
     "Ticks where a demote pass actually fired"),
    # Block movement
    ("pn95_blocks_demoted_total",
     "blocks_demoted_total", "counter",
     "Total KV blocks demoted GPU->CPU"),
    ("pn95_blocks_promoted_total",
     "blocks_promoted_total", "counter",
     "Total KV blocks promoted CPU->GPU"),
    # GPU state
    ("pn95_gpu_free_mib",
     "last_free_mib", "gauge",
     "GPU free memory (MiB) at last scheduler tick"),
    # Prefix store (L2 pageable)
    ("pn95_prefix_store_entries",
     "prefix_store_entries", "gauge",
     "Current L2 prefix-store entry count"),
    ("pn95_prefix_store_bytes",
     "prefix_store_bytes_used", "gauge",
     "Current L2 prefix-store bytes resident"),
    ("pn95_prefix_lookups_total",
     "prefix_lookups_total", "counter",
     "promote_on_miss invocations"),
    ("pn95_prefix_lookups_cold_miss_total",
     "prefix_lookups_cold_miss", "counter",
     "Cold misses (no L1/L2/disk hit)"),
    ("pn95_prefix_hit_rate",
     "prefix_hit_rate", "gauge",
     "(lookups - cold_miss) / lookups"),
    # Compression
    ("pn95_compress_raw_bytes_total",
     "compress_raw_bytes_total", "counter",
     "Raw bytes seen by compressor"),
    ("pn95_compress_stored_bytes_total",
     "compress_stored_bytes_total", "counter",
     "Bytes after compression"),
    ("pn95_compress_ratio",
     "compress_ratio", "gauge",
     "raw / stored compression ratio"),
    # Async transfers
    ("pn95_async_demote_total",
     "async_demote_count", "counter",
     "Layer-level async demote ops"),
    ("pn95_async_promote_total",
     "async_promote_count", "counter",
     "Layer-level async promote ops"),
    ("pn95_async_batch_demote_total",
     "async_batch_demote_count", "counter",
     "Batched demote dispatches"),
    ("pn95_async_batch_promote_total",
     "async_batch_promote_count", "counter",
     "Batched promote dispatches"),
    # L1 pinned pool (this session)
    ("pn95_l1_slots_capacity",
     "l1_slots_capacity", "gauge",
     "Total slots in L1 pinned pool"),
    ("pn95_l1_slot_size_bytes",
     "l1_slot_size_bytes", "gauge",
     "Per-slot size in L1 pinned pool"),
    ("pn95_l1_slots_used",
     "l1_slots_used", "gauge",
     "Slots currently holding payloads"),
    ("pn95_l1_bytes_used",
     "l1_bytes_used", "gauge",
     "Total bytes resident in L1"),
    ("pn95_l1_demote_writes_total",
     "l1_demote_writes", "counter",
     "Writes into L1 from demote path"),
    ("pn95_l1_promote_hits_total",
     "l1_promote_hits", "counter",
     "L1 hits served on promote (vs falling back to L2)"),
    ("pn95_l1_full_skips_total",
     "l1_full_skips", "counter",
     "Demote attempts skipped because L1 had no free slot"),
    ("pn95_l1_evictions_total",
     "l1_evictions", "counter",
     "L1 slots evicted (mirrors L2 LRU)"),
    # Disk tier (Tier 3)
    ("pn95_ram_to_disk_spills_total",
     "ram_to_disk_spills_total", "counter",
     "L2 entries spilled to disk"),
    ("pn95_disk_to_ram_promotes_total",
     "disk_to_ram_promotes_total", "counter",
     "Disk entries promoted back to L2"),
    # Prefetch API
    ("pn95_prefetch_calls_total",
     "prefetch_calls", "counter",
     "pn95_prefetch_blocks() invocations"),
    ("pn95_prefetch_block_hashes_total",
     "prefetch_block_hashes", "counter",
     "Block hashes seen by prefetch"),
    ("pn95_prefetch_l2_to_l1_total",
     "prefetch_l2_hits_promoted", "counter",
     "Prefetch L2->L1 warm-ups"),
    ("pn95_prefetch_disk_to_l1_total",
     "prefetch_disk_hits_promoted", "counter",
     "Prefetch disk->L1 warm-ups"),
    ("pn95_prefetch_missing_total",
     "prefetch_missing", "counter",
     "Prefetch block_hashes not found in any tier"),
    # Layer-aware demote
    ("pn95_layer_access_distinct",
     "layer_access_distinct", "gauge",
     "Distinct attention layer names observed in promote"),
    ("pn95_layer_access_total_observations",
     "layer_access_total_observations", "counter",
     "Total per-layer promote-restoration events"),
    # Workspace (#40020 backports)
    ("pn95_super_block_demote_batches_total",
     "super_block_demote_batches", "counter",
     "Grouped demote batches under block_size_factor>1"),
    ("pn95_store_threshold_skips_total",
     "store_threshold_skips", "counter",
     "Blocks skipped by reuse-frequency gate"),
    ("pn95_demote_rollback_total",
     "demote_rollback_count", "counter",
     "Two-phase commit rollbacks on L2 insert failure"),
    ("pn95_stream_pool_batches_total",
     "stream_pool_batches", "counter",
     "v2 stream-pool batch dispatches"),
]


def render_metrics(stats: dict) -> str:
    """Render the loaded stats dict into Prometheus text-exposition format."""
    lines: list[str] = []
    lines.append("# PN95 multi-tier KV cache metrics")
    lines.append(
        f"# stats_file_timestamp {stats.get('timestamp', 0)}"
    )
    for metric, key, kind, help_text in _METRIC_MAP:
        value = stats.get(key)
        if value is None or isinstance(value, str):
            continue  # skip absent or non-numeric (e.g. compress_lib='zstd')
        if isinstance(value, bool):
            value = 1 if value else 0
        lines.append(f"# HELP {metric} {help_text}")
        lines.append(f"# TYPE {metric} {kind}")
        lines.append(f"{metric} {value}")
    # Boolean-state mirror (compress_lib name as label).
    lib = stats.get("compress_lib", "uninit")
    if lib:
        lines.append("# HELP pn95_compress_lib_info Configured compression backend.")
        lines.append("# TYPE pn95_compress_lib_info gauge")
        lines.append(f'pn95_compress_lib_info{{lib="{lib}"}} 1')
    return "\n".join(lines) + "\n"


class _Handler(BaseHTTPRequestHandler):
    stats_path: str = "/tmp/pn95_stats.json"

    # Keep the access log quiet — sidecars are scraped 1-3x/min, no need.
    def log_message(self, format, *args):  # noqa: A003
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(b"ok\n")
            return
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        stats: dict = {}
        try:
            with open(self.stats_path) as f:
                stats = json.load(f)
        except FileNotFoundError:
            stats = {"error": "stats_file_missing"}
        except json.JSONDecodeError:
            stats = {"error": "stats_file_corrupt"}
        body = render_metrics(stats)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pn95-prometheus",
                                description="Serve PN95 stats as Prometheus metrics.")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("PN95_PROM_PORT", "9192")),
                   help="HTTP port to bind (default 9192).")
    p.add_argument("--host", default=os.environ.get("PN95_PROM_HOST", "0.0.0.0"),
                   help="HTTP host to bind (default 0.0.0.0).")
    p.add_argument("--path",
                   default=os.environ.get("GENESIS_PN95_STATS_FILE",
                                          "/tmp/pn95_stats.json"),
                   help="Path to the PN95 stats JSON file.")
    p.add_argument("--once", action="store_true",
                   help="Dump metrics once to stdout and exit (debug).")
    args = p.parse_args(argv)

    _Handler.stats_path = args.path

    if args.once:
        try:
            with open(args.path) as f:
                stats = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"# pn95-prometheus: {e}", file=sys.stderr)
            return 1
        sys.stdout.write(render_metrics(stats))
        return 0

    server = HTTPServer((args.host, args.port), _Handler)
    print(f"[pn95-prometheus] serving on http://{args.host}:{args.port}/metrics "
          f"(reading {args.path})", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
