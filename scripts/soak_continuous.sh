#!/usr/bin/env bash
#
# Genesis quality-gate — CONTINUOUS soak (Cliff 2b multi-turn VRAM-accretion).
#
# The only probe that surfaces Cliff 2b: GDN multi-turn VRAM accretion under an
# accumulating-context agentic conversation (hermes/openhands/Cline shape). Each
# session is ONE multi-turn coding conversation that ramps to ~22-25K accumulated
# tokens by turn 5 — fresh/reset-each-turn fixtures do NOT trigger this class.
# Read-only against the running deployment; intentionally slow (~10-30 min).
#
# This is the EXTENDED Genesis port of club-3090's scripts/soak-test.sh
# --continuous + scripts/soak-helper.py (github.com/noonghunna/club-3090, MIT).
# Adaptation notes + credit: docs/QUALITY_GATE.md.
#
# Genesis extension — the patch-attribution discipline ("PASS != patches
# load-bearing", club-3090 #140) made executable:
#   --strip-overlays runs the SAME soak twice — once with Genesis overlays ON,
#   once with them OFF (GENESIS_ENABLE=0) — and diffs the two verdicts to decide
#   whether the patch under test (e.g. PN59) was actually LOAD_BEARING, or
#   whether the PASS came for free from topology (TP=2 sidesteps Cliff 2b) or
#   the workload simply not being deep enough. A green soak alone proves the
#   config is stable; it does NOT prove the overlay did the work. This mode
#   proves it.
#
# Verdict (PASS == no failure signal on the sample):
#   - request errors / stream interruptions: 0
#   - silent-empty turns (HTTP 200 + 0 completion tokens): < 50%
#   - max VRAM growth from warm baseline: < SOAK_MAX_GROWTH_MIB (default 200)
#   - decode TPS retention (first vs last sessions): >= 80%
#
# SOAK_CLIFF2B=1 switches to club-3090's VERBATIM Cliff-2b PASS gate (stricter):
#   - silent-empty turns: == 0 (any single one FAILs; counted by completion_tokens)
#   - max VRAM growth from warm baseline: < 200 MiB
#   - decode TPS retention (first-5 vs last-5 turns): >= 98%
#   - request errors: == 0
#   - fixture shape: the 5 sessions x 5 turns ramp (the ONLY shape surfacing 2b)
#
# Usage:
#   scripts/soak_continuous.sh                      # auto-detect endpoint + model
#   scripts/soak_continuous.sh --strip-overlays     # ON vs OFF attribution run
#   scripts/soak_continuous.sh --dry-run            # print the plan, send nothing
#   scripts/soak_continuous.sh --help
#
# Env (optional):
#   URL / MODEL / API_KEY    Endpoint, served model, bearer key.
#   PRESET                   sndr preset (resolves URL/MODEL best-effort).
#   CONTAINER                Container name for VRAM scoping (default none).
#   SOAK_SESSIONS            Sessions (default 5 — the cross-rig cadence).
#   SOAK_TURNS               Turns/session, must be 5 (the ramp shape).
#   SOAK_MAX_GROWTH_MIB      VRAM-growth fail threshold (default 200).
#   SOAK_CLIFF2B             1 = club-3090 verbatim Cliff-2b gate (see above).
#   SOAK_REQ_TIMEOUT_S       Per-request timeout (default 600).
#   ATTR_PATCH               Patch ID under test for attribution (default PN59).
#   ATTR_TP                  Topology TP for attribution (default 1).
#   GENESIS_DISABLE_CMD      Override how overlays are disabled in --strip-overlays
#                            (default: export GENESIS_ENABLE=0). Informational —
#                            the operator must relaunch the stripped config; see
#                            the printed instructions.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER=(python3 -m quality_gate.runner)
export PYTHONPATH="${ROOT_DIR}/tools${PYTHONPATH:+:${PYTHONPATH}}"

DRY_RUN=0
STRIP_OVERLAYS=0
print_usage() {
  sed -n '2,58p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --strip-overlays) STRIP_OVERLAYS=1; shift ;;
    -h|--help) print_usage; exit 0 ;;
    *) echo "soak_continuous.sh: unknown argument: $1" >&2
       echo "  run with --help for usage." >&2
       exit 2 ;;
  esac
done

URL="${URL:-}"
MODEL="${MODEL:-}"
API_KEY="${API_KEY:-genesis-local}"
CONTAINER="${CONTAINER:-none}"
PRESET="${PRESET:-}"
SOAK_SESSIONS="${SOAK_SESSIONS:-5}"
SOAK_TURNS="${SOAK_TURNS:-5}"
SOAK_MAX_GROWTH_MIB="${SOAK_MAX_GROWTH_MIB:-200}"
SOAK_REQ_TIMEOUT_S="${SOAK_REQ_TIMEOUT_S:-600}"
SOAK_CLIFF2B="${SOAK_CLIFF2B:-0}"
ATTR_PATCH="${ATTR_PATCH:-PN59}"
ATTR_TP="${ATTR_TP:-1}"
GENESIS_DISABLE_CMD="${GENESIS_DISABLE_CMD:-export GENESIS_ENABLE=0}"

if [[ "$SOAK_TURNS" -ne 5 ]]; then
  echo "soak_continuous.sh: continuous mode requires SOAK_TURNS=5 (got ${SOAK_TURNS})." >&2
  echo "  The turn shapes are designed to ramp; partial sessions don't reach the target context." >&2
  exit 2
fi

resolve_preset() {
  [[ -z "$PRESET" ]] && return 0
  command -v sndr >/dev/null 2>&1 || return 0
  local rendered port
  rendered="$(sndr launch --dry-run "$PRESET" 2>/dev/null || true)"
  [[ -z "$rendered" ]] && return 0
  if [[ -z "$URL" ]]; then
    port="$(printf '%s\n' "$rendered" | grep -oE '(-p|--port)[ =]+[0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
    [[ -n "$port" ]] && URL="http://localhost:${port}"
  fi
  if [[ -z "$MODEL" ]]; then
    MODEL="$(printf '%s\n' "$rendered" | grep -oE 'served-model-name[ =]+[^ ]+' | awk '{print $2}' | head -1 || true)"
  fi
}
resolve_preset
URL="${URL:-http://localhost:8000}"

C_PASS=$'\033[32m'; C_FAIL=$'\033[31m'; C_WARN=$'\033[33m'; C_OFF=$'\033[0m'
log() { printf '[soak] %s\n' "$*"; }
json_get() { python3 -c "import sys,json; print(json.load(sys.stdin).get('$1',''))"; }

vram_used_mib() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{s+=$1} END {printf "%.0f\n", s}' 2>/dev/null || echo 0
}

if [[ -z "$MODEL" && "$DRY_RUN" == "0" ]]; then
  MODEL="$(curl -sf -m 5 -H "Authorization: Bearer ${API_KEY}" "${URL}/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || true)"
fi
MODEL="${MODEL:-qwen3.6-27b}"

echo "Genesis quality-gate CONTINUOUS soak (Cliff 2b detector)"
echo "  url=${URL}  model=${MODEL}  container=${CONTAINER}"
echo "  sessions=${SOAK_SESSIONS} turns=${SOAK_TURNS} max_growth=${SOAK_MAX_GROWTH_MIB}MiB"
echo "  core=tools/quality_gate (unit-tested)  doc=docs/QUALITY_GATE.md"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] plan (no requests sent):"
  echo "  Each session is ONE ramping multi-turn coding conversation:"
  python3 -c "
import sys
sys.path.insert(0, '${ROOT_DIR}/tools')
from quality_gate import soak
for t in soak.CONTINUOUS_TURNS:
    synth = t['tool_synth']
    synth_s = 'none' if synth is None else f'{synth[1]} ~{synth[2]} chars'
    print(f'    turn {t[\"turn\"]}: max_tokens={t[\"max_tokens\"]:<5} next-tool-result={synth_s}')
print('    => accumulated context reaches ~22-25K tokens by turn 5 (Cliff 2b territory)')
"
  echo ""
  if [[ "$STRIP_OVERLAYS" == "1" ]]; then
    echo "  --strip-overlays: would run the soak TWICE and diff the verdicts:"
    echo "    1. overlays ON  (current launch) -> on-verdict.json"
    echo "    2. overlays OFF (${GENESIS_DISABLE_CMD}; relaunch) -> stripped-verdict.json"
    echo "    3. attribute patch=${ATTR_PATCH} tp=${ATTR_TP} -> LOAD_BEARING / TOPOLOGY_SIDESTEP / NOT_LOAD_BEARING"
    echo "  example attribution verdict (logic only, no engine):"
    on=$(mktemp); strip=$(mktemp)
    printf '{"verdict":"PASS","boot_vram_mib":21000,"max_vram_mib":21100,"growth_mib":100,"growth_limit_mib":200,"sessions_completed":5,"errors":0,"silent_empty":0,"total_turns":25,"tps_retention_pct":99.0,"ttft_ratio":1.0,"p50_decode_tps":110.0,"failures":[],"warnings":[],"exit_code":0}' > "$on"
    printf '{"verdict":"FAIL","boot_vram_mib":21000,"max_vram_mib":23800,"growth_mib":2800,"growth_limit_mib":200,"sessions_completed":2,"errors":1,"silent_empty":0,"total_turns":12,"tps_retention_pct":0.0,"ttft_ratio":0.0,"p50_decode_tps":0.0,"failures":["VRAM grew 2800 MiB"],"warnings":[],"exit_code":1}' > "$strip"
    "${RUNNER[@]}" soak-attribute --on "$on" --stripped "$strip" --patch "$ATTR_PATCH" --tp "$ATTR_TP" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print('   ', d['verdict']); print('   ', d['detail'][:100])"
    rm -f "$on" "$strip"
  fi
  echo ""
  echo "[dry-run] OK — harness wiring is sound. A full live run needs the GPU rig."
  exit 0
fi

# --- run_one_soak: execute a full soak pass, write a verdict JSON to $1 -------
# Returns the soak exit code (0 PASS / 1 FAIL / 2 inconclusive).
run_one_soak() {
  local verdict_out="$1" label="$2"
  local workdir rows boot_vram session turn state req metrics
  workdir="$(mktemp -d)"
  rows="${workdir}/rows.jsonl"
  : > "$rows"

  if ! curl -sf -m 10 -H "Authorization: Bearer ${API_KEY}" "${URL}/v1/models" >/dev/null 2>&1; then
    log "ERROR (${label}): endpoint not reachable at ${URL}/v1/models"
    rm -rf "$workdir"
    return 2
  fi

  log "${label}: running ${SOAK_SESSIONS} sessions x ${SOAK_TURNS} turns against ${URL}"
  boot_vram=0
  local start_epoch
  start_epoch="$(date +%s)"

  for session in $(seq 1 "$SOAK_SESSIONS"); do
    state="${workdir}/state-s${session}.json"
    "${RUNNER[@]}" soak-init --state "$state" --session "$session" >/dev/null
    for turn in $(seq 1 "$SOAK_TURNS"); do
      req="${workdir}/s${session}-t${turn}.req.json"
      metrics="${workdir}/s${session}-t${turn}.metrics.json"
      "${RUNNER[@]}" soak-request --state "$state" --model "$MODEL" --turn "$turn" --req "$req" >/dev/null
      "${RUNNER[@]}" send --url "$URL" --req "$req" --timeout "$SOAK_REQ_TIMEOUT_S" --out "$metrics" >/dev/null
      "${RUNNER[@]}" soak-ingest --state "$state" --metrics "$metrics" --turn "$turn" >/dev/null

      local http status ttft wall comp vram decode_tps err
      http="$(json_get http_code < "$metrics")"
      ttft="$(json_get ttft_ms < "$metrics")"; [[ -z "$ttft" || "$ttft" == "None" ]] && ttft=0
      wall="$(json_get wall_ms < "$metrics")"; [[ -z "$wall" ]] && wall=0
      comp="$(json_get completion_tokens < "$metrics")"; [[ -z "$comp" ]] && comp=0
      err="$(json_get error < "$metrics")"
      vram="$(vram_used_mib)"
      status="${http:-0}"
      # decode_tps = completion / max(wall-ttft, eps); approximate from wall here.
      decode_tps="$(python3 -c "w=${wall}/1000.0; t=${ttft}/1000.0; d=max(w-t,1e-6); print(round(${comp}/d,3) if ${comp}>0 and (w-t)>0.1 else 0.0)")"
      printf '{"session_id":%d,"turn_id":%d,"t_ms":%d,"vram_mib":%d,"ttft_ms":%d,"decode_tps":%s,"status":%d,"error":%s,"completion_tokens":%d}\n' \
        "$session" "$turn" "$wall" "$vram" "$ttft" "$decode_tps" "$status" \
        "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "${err:-}")" "$comp" >> "$rows"
      log "  s${session} t${turn}: status=${status} wall=${wall}ms ttft=${ttft}ms comp=${comp} vram=${vram}MiB"
    done
    # Warm baseline = VRAM after session 1 (prefix cache filled), so cache-fill
    # is not mistaken for accretion.
    if [[ "$boot_vram" -eq 0 ]]; then
      boot_vram="$(vram_used_mib)"
      log "  warm baseline after session 1: ${boot_vram} MiB"
    fi
  done
  [[ "$boot_vram" -eq 0 ]] && boot_vram="$(vram_used_mib)"

  local elapsed timed_out
  elapsed=$(( $(date +%s) - start_epoch ))
  timed_out=0
  log "  ${label}: ${elapsed}s elapsed"

  local cliff2b_flag=()
  [[ "$SOAK_CLIFF2B" == "1" ]] && cliff2b_flag=(--cliff2b)
  set +e
  "${RUNNER[@]}" soak-verdict --rows "$rows" --boot-vram "$boot_vram" \
    --growth-limit "$SOAK_MAX_GROWTH_MIB" --expected-sessions "$SOAK_SESSIONS" \
    --timed-out "$timed_out" "${cliff2b_flag[@]}" --out "$verdict_out" \
    > "${workdir}/verdict.stdout"
  local rc=$?
  set -e
  local v; v="$(json_get verdict < "$verdict_out")"
  local growth; growth="$(json_get growth_mib < "$verdict_out")"
  log "  ${label}: verdict=${v} growth=${growth}MiB exit=${rc}"
  rm -rf "$workdir"
  return "$rc"
}

if [[ "$STRIP_OVERLAYS" != "1" ]]; then
  # Single soak.
  on_verdict="$(mktemp)"
  set +e
  run_one_soak "$on_verdict" "overlays-ON"
  rc=$?
  set -e
  v="$(json_get verdict < "$on_verdict")"
  echo ""
  case "$v" in
    PASS) printf "%sSoak PASS%s — no failure signal on this sample.\n" "$C_PASS" "$C_OFF"
          echo "  NOTE: PASS proves stability, NOT that overlay patches are load-bearing."
          echo "  Re-run with --strip-overlays to attribute (club-3090 #140 discipline)." ;;
    FAIL) printf "%sSoak FAIL%s — see [soak] failures above; likely Cliff 2b (PN59). See docs/QUALITY_GATE.md.\n" "$C_FAIL" "$C_OFF" ;;
    *)    printf "%sSoak INCONCLUSIVE%s.\n" "$C_WARN" "$C_OFF" ;;
  esac
  rm -f "$on_verdict"
  exit "$rc"
fi

# --- --strip-overlays: ON then OFF, then attribute ---------------------------
on_verdict="$(mktemp)"
strip_verdict="$(mktemp)"

log "step 1/3: overlays-ON soak (current launch)"
set +e
run_one_soak "$on_verdict" "overlays-ON"
on_rc=$?
set -e

echo ""
echo "${C_WARN}=== ACTION REQUIRED ===${C_OFF}"
echo "Step 2 needs the SAME config relaunched with Genesis overlays DISABLED, so the"
echo "soak measures the engine WITHOUT the patches under test (${ATTR_PATCH})."
echo "Relaunch the stripped config now, e.g.:"
echo "    ${GENESIS_DISABLE_CMD}"
echo "    sndr launch <same-preset>        # or your launch command, overlays off"
echo "Then press ENTER to run the stripped soak (or Ctrl-C to abort)."
read -r _ || true

log "step 2/3: overlays-STRIPPED soak"
set +e
run_one_soak "$strip_verdict" "overlays-STRIPPED"
strip_rc=$?
set -e
# A FAIL on the stripped run is expected and informative — it is exactly what
# proves the patch is load-bearing. We do not gate on strip_rc; the attribution
# step consumes the stripped verdict. Logged so the exit code is visible.
log "  overlays-STRIPPED soak exit=${strip_rc} (a non-zero here is the load-bearing signal)"

log "step 3/3: attribution (patch=${ATTR_PATCH} tp=${ATTR_TP})"
attr="$(mktemp)"
"${RUNNER[@]}" soak-attribute --on "$on_verdict" --stripped "$strip_verdict" \
  --patch "$ATTR_PATCH" --tp "$ATTR_TP" > "$attr"
attr_verdict="$(json_get verdict < "$attr")"
attr_detail="$(json_get detail < "$attr")"

echo ""
echo "=== PATCH ATTRIBUTION (${ATTR_PATCH}) ==="
case "$attr_verdict" in
  LOAD_BEARING)     printf "%s%s%s — the patch did the work.\n" "$C_PASS" "$attr_verdict" "$C_OFF" ;;
  TOPOLOGY_SIDESTEP|NOT_LOAD_BEARING) printf "%s%s%s\n" "$C_WARN" "$attr_verdict" "$C_OFF" ;;
  *) printf "%s%s%s\n" "$C_FAIL" "$attr_verdict" "$C_OFF" ;;
esac
echo "  ${attr_detail}"
rm -f "$on_verdict" "$strip_verdict" "$attr"

# Exit non-zero if the ON config itself failed; attribution is informational.
if [[ "$on_rc" -ne 0 ]]; then
  exit "$on_rc"
fi
exit 0
