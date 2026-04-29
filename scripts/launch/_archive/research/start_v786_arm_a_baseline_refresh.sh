#!/bin/bash
# Phase 1 — Arm A: refresh v775 PROD baseline measurement
#
# IDENTICAL to start_v775_35b_baseline.sh.
# Purpose: capture today's PROD numbers with the NEW bench_decode_tpot_clean_ab.py
# tool to establish the comparison baseline before Arms B and C.
#
# Run order:
#   1. bash scripts/launch/snapshot_pre_arm.sh arm_a
#   2. ssh sander@192.168.1.10 'bash -s' < scripts/launch/start_v786_arm_a_baseline_refresh.sh
#   3. wait ~3 min for boot, verify /v1/models reachable
#   4. python3 tools/bench_decode_tpot_clean_ab.py --host 192.168.1.10 \
#        --arm-name arm_a_baseline_refresh --runs 25 --prompts standard \
#        --out runs/phase1_arm_a.json
#   5. tool-call quality probe (4 cases) + 30-iteration stress
exec "$(dirname "$0")/start_v775_35b_baseline.sh"
