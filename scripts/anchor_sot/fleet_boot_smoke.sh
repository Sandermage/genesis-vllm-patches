#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Phase 4 (dynamic) — FLEET BOOT-SMOKE gate for a candidate vLLM pin.
#
# The static anchor-SoT gates (rebuild_pin / bump_preflight / new_pin_check /
# audit_pin) diff manifests; they cannot catch a *runtime* boot regression. This
# gate does: it serially boots every fleet preset on the CANDIDATE image + the
# repo tree, and per model asserts:
#   * container reaches health 200,
#   * Genesis apply reports failed=0 (no patch crashed),
#   * boot_smoke_probe.py PASSes (coherent generation + streaming tool-call).
#
# It exists because dev672 forcing `disable_chunked_mm_input` for Gemma-4 broke
# boot via G4_09's 2048 chunk clamp (< the new max_tokens_per_mm_item=2496 floor)
# — apply was failed=0, so ONLY a live boot surfaced it. Run this on every bump.
#
# RUNS ON THE RIG (needs docker + the candidate image + /models). It stops the
# live engine for the window and ALWAYS restores it on exit (trap).
#
# Env:
#   IMAGE              candidate image tag (required)  e.g. vllm/vllm-openai:nightly-<sha>
#   REPO               repo tree to mount/install      (default /home/sander/gvp-mainsync)
#   RESTORE_CONTAINER  live engine to stop+restart     (default vllm-35b-dev672)
#   PORT               serve port                      (default 8102)
#   API_KEY            engine api key                  (default genesis-local)
#   MODELS_HOST        host models dir                 (default /nfs/genesis/models)
#   FLEET              space-sep "preset:served[:notool]" entries (required)
#   BOOT_TIMEOUT       per-model health wait seconds   (default 480)
set -uo pipefail

IMAGE="${IMAGE:?set IMAGE to the candidate vllm image tag}"
REPO="${REPO:-/home/sander/gvp-mainsync}"
RESTORE_CONTAINER="${RESTORE_CONTAINER:-vllm-35b-dev672}"
PORT="${PORT:-8102}"
API_KEY="${API_KEY:-genesis-local}"
MODELS_HOST="${MODELS_HOST:-/nfs/genesis/models}"
FLEET="${FLEET:?set FLEET to space-sep preset:served[:notool] entries}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-480}"
SP=/usr/local/lib/python3.12/dist-packages
PROBE="$REPO/scripts/anchor_sot/boot_smoke_probe.py"
NAME=fleet-boot-smoke
RC=0

log(){ echo "[fleet-boot-smoke] $*"; }
restore(){
  docker rm -f "$NAME" >/dev/null 2>&1
  if [ -n "$RESTORE_CONTAINER" ]; then
    log "restoring live engine $RESTORE_CONTAINER"
    docker start "$RESTORE_CONTAINER" >/dev/null 2>&1
    for _ in $(seq 1 80); do
      [ "$(curl -s -o /dev/null -w '%{http_code}' -m3 "http://127.0.0.1:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" 2>/dev/null)" = "200" ] && { log "restored: health 200"; break; }
      sleep 6
    done
  fi
  log "DONE rc=$RC"
}
trap restore EXIT

log "candidate IMAGE=$IMAGE  REPO=$REPO  port=$PORT"
[ -n "$RESTORE_CONTAINER" ] && { log "stopping live engine $RESTORE_CONTAINER (window start)"; docker stop "$RESTORE_CONTAINER" >/dev/null 2>&1; }

for entry in $FLEET; do
  preset="${entry%%:*}"; rest="${entry#*:}"; served="${rest%%:*}"; flags="${rest#*:}"
  [ "$flags" = "$rest" ] && flags=""     # no third field
  skip_tool=""; [ "$flags" = "notool" ] && skip_tool="--skip-toolcall"
  log "===== $preset (served=$served) ====="

  # The dry-run render is a HOST `docker run` launcher (it boots its own
  # container). We must run its IN-CONTAINER `-c` payload directly under OUR
  # candidate image + mounts — wrapping the whole host script inside a container
  # tries docker-in-docker and dies with `docker: command not found` (the bug
  # this gate hit on the dev714 fleet run). Extract payload + inner serve port.
  render=$(cd "$REPO" && python3 -m sndr.cli launch "$preset" --dry-run --port "$PORT" 2>/dev/null)
  if [ -z "$render" ]; then log "  RENDER FAILED for $preset"; RC=1; continue; fi
  extracted=$(printf '%s\n' "$render" | python3 "$REPO/scripts/anchor_sot/_extract_launch_payload.py" 2>/dev/null)
  inner_port="${extracted%%$'\t'*}"; payload="${extracted#*$'\t'}"
  if [ -z "$payload" ] || [ "$payload" = "$extracted" ]; then log "  PAYLOAD EXTRACT FAILED for $preset"; RC=1; continue; fi
  inner_port="${inner_port:-8000}"
  # pip install -e registers the vllm.general_plugins entry-point; the bound sndr
  # mount makes `import sndr` resolve. Then run the rendered in-container payload.
  full="pip install -e $REPO --no-deps --quiet 2>&1 | tail -0; $payload"

  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --entrypoint /bin/bash --gpus all --ipc host --shm-size 67108864 \
    -p "$PORT:$inner_port" \
    -v "$REPO:$REPO" -v "$REPO/sndr:$SP/sndr:ro" -v "$REPO/sndr:$SP/vllm/sndr_core:ro" \
    -v "$MODELS_HOST:/models:ro" \
    "$IMAGE" -c "$full" >/dev/null 2>&1

  ok=no
  for _ in $(seq 1 $((BOOT_TIMEOUT/6))); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m3 "http://127.0.0.1:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" 2>/dev/null)" = "200" ] && { ok=yes; break; }
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ] && { log "  container exited early"; break; }
    sleep 6
  done
  apply=$(docker logs "$NAME" 2>&1 | grep -oE 'applied=[0-9]+ skipped=[0-9]+ failed=[0-9]+' | tail -1)
  failed_n=$(echo "$apply" | grep -oE 'failed=[0-9]+' | grep -oE '[0-9]+')
  log "  boot: health=$ok  apply=[$apply]"
  if [ "$ok" != yes ]; then
    log "  FAIL: did not reach health. diag:"; docker logs "$NAME" 2>&1 | grep -iE 'error|valueerror|assert|traceback|oom' | tail -5 | sed 's/^/    /'
    RC=1
  elif [ "${failed_n:-1}" != "0" ]; then
    log "  FAIL: Genesis apply failed=$failed_n (a patch crashed)"; RC=1
  else
    python3 "$PROBE" --base-url "http://127.0.0.1:$PORT" --api-key "$API_KEY" --model "$served" $skip_tool || RC=1
  fi
  docker stop "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
done

[ "$RC" = 0 ] && log "ALL FLEET MODELS PASSED on $IMAGE" || log "FLEET GATE FAILED (rc=$RC) — do NOT promote this pin until resolved"
exit "$RC"
