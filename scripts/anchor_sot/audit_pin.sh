#!/usr/bin/env bash
# Phase 4 — audit gate: does the COMMITTED per-pin manifest still match a fresh rig regen?
#
# Regenerates the manifest from the live engine (running-container discovery +
# bare-image pristine source) into a temp area on the rig, pulls it back, and
# compares against the committed manifest ignoring volatile metadata. Non-zero
# exit on genuine drift — this is the R2 drift gate to run on a pin bump / in CI
# with rig access.
#
# Usage: audit_pin.sh <user@host>   (env: CONTAINER IMAGE RIG_REPO)
set -euo pipefail

SSH_HOST="${1:?usage: audit_pin.sh <user@host>}"
RIG_REPO="${RIG_REPO:-/tmp/genesis-consolidated}"
CONTAINER="${CONTAINER:-vllm-qwen3.6-35b-balanced-k3}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== sync anchor-sot scripts + sndr to rig ==="
rsync -a "$HERE/scripts/anchor_sot" "$SSH_HOST:$RIG_REPO/scripts/" >/dev/null
rsync -a "$HERE/sndr" "$SSH_HOST:$RIG_REPO/" >/dev/null

echo "=== regenerate manifest on rig (live engine + pristine) ==="
ssh "$SSH_HOST" "CONTAINER=$CONTAINER IMAGE=$IMAGE REPO=$RIG_REPO bash $RIG_REPO/scripts/anchor_sot/rebuild_pin.sh" >&2

TMP="$(mktemp -d)"
rsync -a "$SSH_HOST:$RIG_REPO/sndr/engines/vllm/pins/" "$TMP/" >/dev/null

rc=0
for d in "$TMP"/*/; do
    pin="$(basename "$d")"
    fresh="$d/anchors.json"
    committed="$HERE/sndr/engines/vllm/pins/$pin/anchors.json"
    [ -f "$fresh" ] || continue
    if [ ! -f "$committed" ]; then
        echo "DRIFT: pin $pin regenerated but not committed — run make rebuild-pin"; rc=1; continue
    fi
    python3 "$HERE/scripts/anchor_sot/compare_manifest.py" "$committed" "$fresh" || rc=1
done
[ "$rc" -eq 0 ] && echo "audit-pin: all committed manifests match a fresh regen"
exit "$rc"
