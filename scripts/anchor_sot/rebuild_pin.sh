#!/usr/bin/env bash
# Phase 4 — regenerate the per-pin anchor manifest end-to-end ON THE RIG.
#
# Runs the proven 2-step pipeline:
#   1. discovery in the RUNNING pinned container  -> canonical anchor set (204)
#   2. pristine source from a BARE same-pin image -> un-patched vLLM source
#   3. classify each anchor against the real pristine source + write
#      sndr/engines/vllm/pins/<pin>/anchors.json (engine schema, round-trip verified)
#
# A bump becomes: boot the new pin, run this, commit the new pins/<pin>/anchors.json.
# Run on the inference rig host (over SSH).  Override CONTAINER/IMAGE/REPO via env.
set -euo pipefail

CONTAINER="${CONTAINER:-vllm-qwen3.6-35b-balanced-k3}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
REPO="${REPO:-/tmp/genesis-consolidated}"
DIST="${DIST:-/usr/local/lib/python3.12/dist-packages/sndr}"
S="$REPO/scripts/anchor_sot"
WORK="$REPO/.anchor_build"
mkdir -p "$WORK"

echo "=== step 1: discovery in running container $CONTAINER ==="
docker exec "$CONTAINER" python3 "$S/discover.py" "$WORK/targets.json"
PIN="$(docker exec "$CONTAINER" python3 -c "import json;print(json.load(open('$WORK/targets.json'))['pin'])")"
GPIN="$(docker exec "$CONTAINER" python3 -c "import json;print(json.load(open('$WORK/targets.json'))['genesis_pin'])")"
echo "    pin=$PIN genesis=$GPIN"

echo "=== step 2: pristine source from bare image $IMAGE ==="
docker run --rm -v "$REPO:$REPO" -v "$REPO/sndr:$DIST:ro" --entrypoint python3 "$IMAGE" \
    "$S/pristine_dump.py" "$WORK/targets.json" "$WORK/pristine.json"

echo "=== step 3: classify + write manifest ==="
docker run --rm -v "$REPO:$REPO" -v "$REPO/sndr:$DIST:ro" --entrypoint python3 "$IMAGE" \
    "$S/build_manifest.py" "$WORK/targets.json" "$WORK/pristine.json" "$REPO" "$PIN" "$GPIN"

echo "=== done — manifest written under $REPO/sndr/engines/vllm/pins/ ==="
