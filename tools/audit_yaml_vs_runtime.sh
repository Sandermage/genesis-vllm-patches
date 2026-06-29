#!/usr/bin/env bash
# audit_yaml_vs_runtime.sh — drift audit for a model config vs live container env
#
# Compares the EFFECTIVE genesis_env of a builtin model config against the
# actual environment variables of a running vLLM container. Reports drift
# between intended (config) and actual (live) state — in BOTH directions, plus
# value mismatches for keys present in both.
#
# Two config formats are handled transparently:
#   * legacy monolithic YAML — GENESIS_* flags inline under `genesis_env:`;
#   * V2 composed preset — a thin resolver referencing model/hardware/profile;
#     the effective env is composed at launch time, so we resolve it the way the
#     launcher does: `sndr launch --dry-run <key>` (key = preset filename stem).
#   The V2 path is the fix for the silent failure where grepping the thin
#   resolver found 0 flags → false "all-drift" AND an undetectable dangerous
#   direction (config enables X, container lacks it).
#
# Background: drift between configs and start-scripts is a recurring regression
# source on Genesis (2026-05-11 Wave 8 found 11 in one session; PN90 dropped
# from a start-script cost -7% TPS until fixed). This script catches it early.
#
# Usage:
#   ./tools/audit_yaml_vs_runtime.sh <yaml_path> <container_name> [<ssh_host>]
#   ./tools/audit_yaml_vs_runtime.sh --dump-config-keys <yaml_path>   # debug/test
#
# Examples:
#   ./tools/audit_yaml_vs_runtime.sh sndr/model_configs/builtin/presets/prod-qwen3.6-35b-balanced.yaml vllm-qwen3.6-35b-balanced-k3 <user>@<host>
#   ./tools/audit_yaml_vs_runtime.sh --dump-config-keys sndr/model_configs/builtin/presets/prod-qwen3.6-35b-balanced.yaml
#
# Exit codes:
#   0 — no real drift (only intentional disables / experiment-specific extras)
#   1 — real drift (config enables a flag the container lacks, a non-experiment
#       extra, or a value mismatch on a shared key)
#   2 — usage error or file not found

set -e

# --- Resolve a config's EFFECTIVE genesis_env as KEY=VALUE lines -------------
# Legacy monolithic config: GENESIS_ flags inline. V2 composed preset (no inline
# flags): resolve via the launcher's dry-run render — the same composition the
# real launch uses.
resolve_config_kv() {
  local yaml="$1" out="$2"
  grep -E "^[[:space:]]+GENESIS_[A-Z0-9_]+:" "$yaml" 2>/dev/null \
    | sed -E "s/^[[:space:]]+//; s/:[[:space:]]*/=/; s/[[:space:]]*#.*$//; s/[[:space:]]+$//; s/['\"]//g" \
    | sort -u > "$out" || true
  if [ -s "$out" ]; then return 0; fi
  # V2 composed preset → resolve the effective launcher env.
  local key
  key="$(basename "$yaml" .yaml)"
  python3 -m sndr launch --dry-run "$key" 2>/dev/null \
    | grep -oE 'GENESIS_[A-Z0-9_]+=[^ \\"]+' \
    | sort -u > "$out" || true
}

# --- --dump-config-keys mode (no docker/ssh — used by tests + debugging) -----
if [ "${1:-}" = "--dump-config-keys" ]; then
  if [ $# -ne 2 ] || [ ! -f "$2" ]; then
    echo "Usage: $0 --dump-config-keys <yaml_path>" >&2
    exit 2
  fi
  DUMP_TMP=$(mktemp)
  trap 'rm -f "$DUMP_TMP"' EXIT
  resolve_config_kv "$2" "$DUMP_TMP"
  cat "$DUMP_TMP"
  exit 0
fi

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <yaml_path> <container_name> [<ssh_host>]" >&2
  echo "       $0 --dump-config-keys <yaml_path>" >&2
  exit 2
fi

YAML_PATH="$1"
CONTAINER_NAME="$2"
SSH_HOST="${3:-}"

if [ ! -f "$YAML_PATH" ]; then
  echo "ERROR: YAML file not found: $YAML_PATH" >&2
  exit 2
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Effective config env (KEY=VALUE), composing V2 presets when needed.
resolve_config_kv "$YAML_PATH" "$TMPDIR/yaml_kv.txt"
cut -d= -f1 "$TMPDIR/yaml_kv.txt" | sort -u > "$TMPDIR/yaml_keys.txt"

# Live container env (KEY=VALUE + keys).
if [ -n "$SSH_HOST" ]; then
  ssh "$SSH_HOST" "docker inspect $CONTAINER_NAME --format '{{range .Config.Env}}{{println .}}{{end}}'" 2>/dev/null \
    | grep -E '^GENESIS_[A-Z0-9_]+=' | sort -u > "$TMPDIR/live_kv.txt"
else
  docker inspect "$CONTAINER_NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
    | grep -E '^GENESIS_[A-Z0-9_]+=' | sort -u > "$TMPDIR/live_kv.txt"
fi
cut -d= -f1 "$TMPDIR/live_kv.txt" | sort -u > "$TMPDIR/live_keys.txt"

if [ ! -s "$TMPDIR/live_keys.txt" ]; then
  echo "ERROR: Could not get live env from container $CONTAINER_NAME (is it running?)" >&2
  exit 2
fi

YAML_COUNT=$(wc -l < "$TMPDIR/yaml_keys.txt" | tr -d ' ')
LIVE_COUNT=$(wc -l < "$TMPDIR/live_keys.txt" | tr -d ' ')

# Helper: value of a key in a KEY=VALUE file.
val_of() { grep -E "^$1=" "$2" | head -1 | cut -d= -f2-; }

echo "═══════════════════════════════════════════════════════════════════════"
echo " YAML vs Runtime Drift Audit"
echo "  Config:    $YAML_PATH (${YAML_COUNT} effective keys)"
echo "  Container: $CONTAINER_NAME (${LIVE_COUNT} keys) ${SSH_HOST:+via $SSH_HOST}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

if [ "$YAML_COUNT" -eq 0 ]; then
  echo "ERROR: resolved 0 config keys from $YAML_PATH — neither inline genesis_env" >&2
  echo "       nor a launch --dry-run composition produced flags. Check the path." >&2
  exit 2
fi

REAL_DRIFT=0

echo "─── Config keys NOT in live env (config wants, container lacks) ───"
YAML_MINUS_LIVE=$(comm -23 "$TMPDIR/yaml_keys.txt" "$TMPDIR/live_keys.txt")
if [ -z "$YAML_MINUS_LIVE" ]; then
  echo "  (none)"
else
  while read -r key; do
    [ -z "$key" ] && continue
    yaml_val=$(val_of "$key" "$TMPDIR/yaml_kv.txt")
    if [ "$yaml_val" = "0" ]; then
      echo "  $key  [OK — explicit disable in config, default-off in env]"
    else
      echo "  $key = $yaml_val  ⚠ DRIFT: config enables but container lacks it"
      REAL_DRIFT=1
    fi
  done <<< "$YAML_MINUS_LIVE"
fi

echo ""
echo "─── Live env keys NOT in config (container has, config lacks) ───"
LIVE_MINUS_YAML=$(comm -13 "$TMPDIR/yaml_keys.txt" "$TMPDIR/live_keys.txt")
if [ -z "$LIVE_MINUS_YAML" ]; then
  echo "  (none)"
else
  while read -r key; do
    [ -z "$key" ] && continue
    if echo "$key" | grep -qE '^GENESIS_(ENABLE_)?PN95_'; then
      echo "  $key  [INTENTIONAL — PN95 experiment additions, see start_pn95_*.sh]"
    else
      echo "  $key  ⚠ EXTRA: container has env not in the resolved config"
      REAL_DRIFT=1
    fi
  done <<< "$LIVE_MINUS_YAML"
fi

echo ""
echo "─── Value mismatches (key in both, different value) ───"
MISMATCH=0
while read -r key; do
  [ -z "$key" ] && continue
  yv=$(val_of "$key" "$TMPDIR/yaml_kv.txt")
  lv=$(val_of "$key" "$TMPDIR/live_kv.txt")
  if [ "$yv" != "$lv" ]; then
    echo "  $key  ⚠ config='$yv'  live='$lv'"
    MISMATCH=1
    REAL_DRIFT=1
  fi
done <<< "$(comm -12 "$TMPDIR/yaml_keys.txt" "$TMPDIR/live_keys.txt")"
[ "$MISMATCH" -eq 0 ] && echo "  (none)"

echo ""
echo "─── Summary ───"
if [ "$REAL_DRIFT" -eq 0 ]; then
  echo "  ✓ No real drift detected. Config and live env are aligned."
  exit 0
else
  echo "  ✗ Real drift detected. See above for details."
  echo "  Action: reconcile the start-script / restart the container from the"
  echo "          current preset (sndr launch <key>), then re-audit."
  exit 1
fi
