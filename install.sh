#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  Genesis vLLM Patches — one-command installer
# ──────────────────────────────────────────────────────────────────────
#
#  Usage:
#    curl -sSL https://raw.githubusercontent.com/Sandermage/genesis-vllm-patches/main/install.sh | bash
#
#    curl -sSL .../install.sh | bash -s -- --pin v7.69
#    curl -sSL .../install.sh | bash -s -- --pin dev
#    curl -sSL .../install.sh | bash -s -- --workload long_context -y
#    curl -sSL .../install.sh | bash -s -- --uninstall
#
#  What it does:
#    1. Detects: OS, Python ≥3.10, vllm install, GPU (via nvidia-smi),
#       container vs bare-metal, available disk
#    2. Resolves pin (default 'stable': latest v12+ tag, else 'main';
#       --pin dev = dev tip; --pin <ref> = explicit commit/tag/branch)
#    3. Clones Genesis into ~/.sndr/ (or $SNDR_HOME / $GENESIS_HOME)
#    4. pip install --no-deps -e <repo>  (editable install of the repo
#       ROOT — puts `sndr` on sys.path AND registers the
#       vllm.general_plugins entry point so vLLM auto-loads Genesis in
#       main + every worker; see docs/INSTALL.md step 3)
#    5. Auto-matches a preset for your (gpu × workload) and writes a
#       runnable launch script
#    6. Runs `sndr.compat.cli verify --quick` — 60-second smoke test
#    7. Prints next-step instructions
#
#  Goals (per Sander 2026-05-02):
#    - One paste. Three minutes. Working system.
#    - 0 questions if --workload + -y given (CI-friendly)
#    - At most 1 question (workload) interactive
#    - Clean error messages — rustup/uv style
#    - Idempotent — safe to re-run
#
#  Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Config (overridable via env or flags) ────────────────────────────

GENESIS_REPO="${GENESIS_REPO:-https://github.com/Sandermage/genesis-vllm-patches.git}"
# Primary path: ~/.sndr (new canonical home, post-rebrand). Legacy alias
# GENESIS_HOME still honored for back-compat with v7.x operators — if
# the operator already has $GENESIS_HOME set, prefer it; otherwise use
# $SNDR_HOME (new) and fall back to ~/.sndr.
SNDR_HOME="${SNDR_HOME:-${HOME}/.sndr}"
GENESIS_HOME="${GENESIS_HOME:-${SNDR_HOME}}"
GENESIS_PIN="${GENESIS_PIN:-stable}"     # 'stable' (latest tag) | 'dev' | <commit-or-tag>
# Track whether the operator passed --pin / set GENESIS_PIN explicitly. When
# they did NOT (still the literal 'stable' default and no env override), the
# interactive pin picker may offer the stable/dev/explicit menu; an explicit
# value always wins and skips the menu (scriptable). See pick_pin().
GENESIS_PIN_EXPLICIT="${GENESIS_PIN_EXPLICIT:-0}"
[ -n "${GENESIS_PIN+x}" ] && [ "${GENESIS_PIN}" != "stable" ] && GENESIS_PIN_EXPLICIT=1
GENESIS_WORKLOAD="${GENESIS_WORKLOAD:-}" # one of: long_context, high_throughput, tool_agent, balanced
# Where model weights live on the host (HF cache / models dir). The launcher
# bind-mounts this into the engine container. Empty = ask interactively (or
# fall back to the first auto-detected candidate in non-interactive mode).
GENESIS_MODELS_DIR="${GENESIS_MODELS_DIR:-}"
GENESIS_NON_INTERACTIVE="${GENESIS_NON_INTERACTIVE:-0}"
GENESIS_NO_VERIFY="${GENESIS_NO_VERIFY:-0}"
GENESIS_NO_PLUGIN_INSTALL="${GENESIS_NO_PLUGIN_INSTALL:-0}"
GENESIS_BARE_METAL="${GENESIS_BARE_METAL:-0}"      # 1 = skip Docker hints, point operator at native vllm serve
GENESIS_UNINSTALL=0

PYTHON_BIN="${PYTHON_BIN:-python3}"
# Pip flag overrides — accept both new SNDR_PIP_FLAGS and legacy
# PIP_INSTALL_FLAGS env vars. New name takes priority if set.
SNDR_PIP_FLAGS="${SNDR_PIP_FLAGS:-}"
# If SNDR_PIP_FLAGS is set, it wins; otherwise fall back to legacy
# PIP_INSTALL_FLAGS or the default `--user` (safer than system-wide).
if [ -n "${SNDR_PIP_FLAGS}" ]; then
    PIP_INSTALL_FLAGS="${SNDR_PIP_FLAGS}"
else
    PIP_INSTALL_FLAGS="${PIP_INSTALL_FLAGS:---user}"
fi

# ─── Output helpers (rustup/uv-style) ─────────────────────────────────

# Colors only when stdout is a TTY
if [ -t 1 ]; then
  C_RESET='\033[0m'
  C_BOLD='\033[1m'
  C_RED='\033[31m'
  C_GREEN='\033[32m'
  C_YELLOW='\033[33m'
  C_BLUE='\033[34m'
  C_GRAY='\033[90m'
else
  C_RESET=''; C_BOLD=''; C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_GRAY=''
fi

info()  { printf '%b\n' "${C_BLUE}info${C_RESET}: $*"; }
ok()    { printf '%b\n' "${C_GREEN}  ok${C_RESET}: $*"; }
warn()  { printf '%b\n' "${C_YELLOW}warn${C_RESET}: $*" >&2; }
err()   { printf '%b\n' "${C_RED} err${C_RESET}: $*" >&2; }
step()  { printf '\n%b\n' "${C_BOLD}» $*${C_RESET}"; }
hint()  { printf '%b\n' "${C_GRAY}      $*${C_RESET}"; }

die() { err "$*"; exit 1; }

# ─── Arg parsing ──────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pin) GENESIS_PIN="$2"; GENESIS_PIN_EXPLICIT=1; shift 2 ;;
    --pin=*) GENESIS_PIN="${1#*=}"; GENESIS_PIN_EXPLICIT=1; shift ;;
    --workload) GENESIS_WORKLOAD="$2"; shift 2 ;;
    --workload=*) GENESIS_WORKLOAD="${1#*=}"; shift ;;
    --models-dir) GENESIS_MODELS_DIR="$2"; shift 2 ;;
    --models-dir=*) GENESIS_MODELS_DIR="${1#*=}"; shift ;;
    --home) GENESIS_HOME="$2"; shift 2 ;;
    --home=*) GENESIS_HOME="${1#*=}"; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --no-verify) GENESIS_NO_VERIFY=1; shift ;;
    --no-plugin) GENESIS_NO_PLUGIN_INSTALL=1; shift ;;
    --bare-metal) GENESIS_BARE_METAL=1; shift ;;
    --system) PIP_INSTALL_FLAGS=""; shift ;;
    --uninstall) GENESIS_UNINSTALL=1; shift ;;
    -y|--yes) GENESIS_NON_INTERACTIVE=1; shift ;;
    -h|--help)
      cat <<'HELP_EOF'
Genesis vLLM Patches — one-command installer

Usage:
  install.sh [flags]

Flags:
  --pin <ref>          Genesis ref to install (default: stable)
                       Special values:
                         stable  = latest v12+ stable tag; if none is
                                   published yet, falls back to 'main'
                                   (pre-v12 tags use the retired
                                   vllm/sndr_core/ layout — issue #29)
                         dev     = dev branch tip (mutable)
                       Or any commit/tag/branch
  --workload <name>    One of: balanced, long_context, high_throughput,
                       tool_agent (default: interactive prompt or 'balanced')
  --models-dir <path>  Host directory holding model weights / HF cache; the
                       launcher bind-mounts it into the engine container.
                       Default: interactive prompt (or first auto-detected
                       candidate in non-interactive mode).
  --home <path>        Where to install Genesis (default: ~/.sndr)
  --python <path>      Python interpreter to use (default: python3)
  --no-verify          Skip post-install smoke test
  --no-plugin          Skip the editable `pip install --no-deps -e <repo>`
                       (Genesis still works via PYTHONPATH but won't
                        auto-load in vllm spawn workers)
  --bare-metal         Skip docker-related preset hints; point operator at
                       native `pip install vllm==0.20.x` + `vllm serve`.
                       Auto-enabled if Proxmox VE host is detected (the
                       official vllm/vllm-openai:nightly image has a known
                       uvloop crash on PVE 8.x kernel 6.17.x — see
                       noonghunna/club-3090#49).
  --system             Use system pip (default: --user)
  --uninstall          Remove Genesis and the entry-point plugin
  -y, --yes            Non-interactive (use defaults)
  -h, --help           Show this help

Examples:
  curl -sSL https://raw.githubusercontent.com/Sandermage/genesis-vllm-patches/main/install.sh | bash
  curl -sSL .../install.sh | bash -s -- --pin v7.69 --workload tool_agent -y
  curl -sSL .../install.sh | bash -s -- --uninstall

Env overrides (alternative to flags):
  GENESIS_REPO, GENESIS_HOME, GENESIS_PIN, GENESIS_WORKLOAD,
  GENESIS_MODELS_DIR, GENESIS_NON_INTERACTIVE, GENESIS_NO_VERIFY,
  GENESIS_NO_PLUGIN_INSTALL, PYTHON_BIN, PIP_INSTALL_FLAGS

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
HELP_EOF
      exit 0
      ;;
    *) die "unknown flag: $1 (use --help)" ;;
  esac
done

# ─── Pre-flight: OS, Python, disk ─────────────────────────────────────

preflight() {
  step "Pre-flight checks"

  # OS check
  case "$(uname -s)" in
    Linux) ok "OS: Linux" ;;
    Darwin) warn "OS: macOS — Genesis targets vLLM on Linux/CUDA. Install will set up the package, but vllm serve won't run here." ;;
    *) die "unsupported OS: $(uname -s). Genesis requires Linux (or macOS for development)." ;;
  esac

  # Python check
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    die "$PYTHON_BIN not found. Install Python 3.10+ or pass --python /path/to/python3"
  fi
  PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
  PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
  if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    die "Python $PY_VERSION too old — Genesis requires ≥3.10."
  fi
  ok "Python: $PY_VERSION ($("$PYTHON_BIN" -c 'import sys; print(sys.executable)'))"

  # Required tools
  for tool in git curl; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      die "$tool not found. Install $tool first."
    fi
  done
  ok "git + curl available"

  # Disk space (need ~200 MB for clone + plugin install)
  if command -v df >/dev/null 2>&1; then
    parent="$(dirname "$GENESIS_HOME")"
    [ -d "$parent" ] || mkdir -p "$parent"
    avail_kb=$(df -k "$parent" | awk 'NR==2 {print $4}')
    if [ -n "$avail_kb" ] && [ "$avail_kb" -lt 204800 ]; then
      warn "Less than 200 MB free at $parent — clone may fail."
    fi
  fi
}

# ─── GPU detection (nvidia-smi → gpu_class) ───────────────────────────

# Sets globals: GPU_NAME, N_GPUS, GPU_CLASS_HINT
detect_gpu() {
  step "GPU detection"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found — Genesis can install but presets need GPU info."
    GPU_NAME=""
    N_GPUS=0
    GPU_CLASS_HINT=""
    return
  fi

  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^ *//; s/ *$//' || echo "")
  N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ' || echo "0")

  if [ -z "$GPU_NAME" ]; then
    warn "nvidia-smi reported no GPUs."
    GPU_CLASS_HINT=""
    return
  fi

  ok "GPU: $GPU_NAME × $N_GPUS"

  # Map nvidia-smi name → gpu_profile.GPU_SPECS key (lowercase substring)
  # Mirror of the keys in sndr/compat/gpu_profile.py:GPU_SPECS
  case "$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]')" in
    *"rtx 3060"*) GPU_CLASS_HINT="rtx 3060" ;;
    *"rtx 3070"*) GPU_CLASS_HINT="rtx 3070" ;;
    *"rtx 3080"*) GPU_CLASS_HINT="rtx 3080" ;;
    *"rtx 3090"*) GPU_CLASS_HINT="rtx 3090" ;;
    *"rtx a4000"*) GPU_CLASS_HINT="rtx a4000" ;;
    *"rtx a5000"*) GPU_CLASS_HINT="rtx a5000" ;;
    *"rtx a6000"*) GPU_CLASS_HINT="rtx a6000" ;;
    *"a100"*) GPU_CLASS_HINT="a100" ;;
    # Ada Lovelace consumer (RTX 40-series) — ORDER MATTERS: specific before general
    *"rtx 4060 ti"*) GPU_CLASS_HINT="rtx 4060 ti" ;;
    *"rtx 4060"*) GPU_CLASS_HINT="rtx 4060" ;;
    *"rtx 4070 ti super"*) GPU_CLASS_HINT="rtx 4070 ti super" ;;
    *"rtx 4070 ti"*) GPU_CLASS_HINT="rtx 4070 ti" ;;
    *"rtx 4070 super"*) GPU_CLASS_HINT="rtx 4070 super" ;;
    *"rtx 4070"*) GPU_CLASS_HINT="rtx 4070" ;;
    *"rtx 4080 super"*) GPU_CLASS_HINT="rtx 4080 super" ;;
    *"rtx 4080"*) GPU_CLASS_HINT="rtx 4080" ;;
    *"rtx 4090"*) GPU_CLASS_HINT="rtx 4090" ;;
    *"l40"*) GPU_CLASS_HINT="l40" ;;
    *"rtx 6000 ada"*) GPU_CLASS_HINT="rtx 6000 ada" ;;
    *"h100"*) GPU_CLASS_HINT="h100" ;;
    *"h200"*) GPU_CLASS_HINT="h200" ;;
    *"h20"*) GPU_CLASS_HINT="h20" ;;
    # Blackwell consumer (RTX 50-series, sm_120) — Issue #20 added per noonghunna RTX 5090 user
    *"rtx 5060 ti"*) GPU_CLASS_HINT="rtx 5060 ti" ;;
    *"rtx 5060"*) GPU_CLASS_HINT="rtx 5060" ;;
    *"rtx 5070 ti"*) GPU_CLASS_HINT="rtx 5070 ti" ;;
    *"rtx 5070"*) GPU_CLASS_HINT="rtx 5070" ;;
    *"rtx 5080"*) GPU_CLASS_HINT="rtx 5080" ;;
    *"rtx 5090"*) GPU_CLASS_HINT="rtx 5090" ;;
    *"rtx pro 6000 blackwell max-q"*) GPU_CLASS_HINT="rtx pro 6000 blackwell max-q" ;;
    *"rtx pro 6000 blackwell"*) GPU_CLASS_HINT="rtx pro 6000 blackwell" ;;
    *"rtx pro 4000 blackwell"*) GPU_CLASS_HINT="rtx pro 4000 blackwell" ;;
    *"rtx pro 4500 blackwell"*) GPU_CLASS_HINT="rtx pro 4500 blackwell" ;;
    *"rtx pro 5000 blackwell"*) GPU_CLASS_HINT="rtx pro 5000 blackwell" ;;
    *"b200"*) GPU_CLASS_HINT="b200" ;;
    *)
      warn "GPU '$GPU_NAME' not in Genesis preset matrix — installing without preset."
      GPU_CLASS_HINT=""
      ;;
  esac

  if [ -n "$GPU_CLASS_HINT" ]; then
    hint "matched preset GPU class: $GPU_CLASS_HINT"
  fi
}

# ─── vLLM detection ───────────────────────────────────────────────────

detect_vllm() {
  step "vLLM detection"

  if ! "$PYTHON_BIN" -c 'import vllm' >/dev/null 2>&1; then
    warn "vllm not importable from $PYTHON_BIN — Genesis installs anyway, but you'll need vllm before patches can apply."
    hint "Install vllm: pip install vllm"
    VLLM_VERSION=""
    return
  fi

  VLLM_VERSION=$("$PYTHON_BIN" -c 'import vllm; print(getattr(vllm, "__version__", "?"))' 2>/dev/null || echo "?")
  ok "vllm: $VLLM_VERSION"

  if [ "$VLLM_VERSION" != "?" ] && [[ "$VLLM_VERSION" != *"0.20"* ]]; then
    warn "Genesis is pinned to vllm 0.20.x — your $VLLM_VERSION may have anchor drift."
    hint "See docs/PATCHES.md or run \`sndr doctor\` after install."
  fi
}

# ─── Proxmox VE / container-runtime caveat detection ─────────────────
#
# Background: noonghunna/club-3090#49 (lexhoefsloot 2026-05-04) — the
# official `vllm/vllm-openai:nightly-7a1eb8ac2…` image crashes at boot
# with `RuntimeError: this event loop is already running` (uvloop) on
# Proxmox VE 8.x kernel 6.17.x hosts. The crash happens BEFORE GPU init
# in bare `docker run` — Genesis is not in the picture. Workaround:
# native `pip install vllm==0.20.x` venv on the same host launches
# cleanly past the failure point.
#
# This detector runs after vLLM detection. If we see a PVE host, we WARN
# and auto-enable --bare-metal so the operator gets the venv-path hints
# instead of docker-compose hints in the next-steps printout.

detect_proxmox_runtime() {
  step "Container-runtime caveat probe (Proxmox VE)"

  local on_pve=0
  # Heuristic 1: kernel string contains 'pve' (Proxmox-built kernel)
  if uname -r 2>/dev/null | grep -qE 'pve|proxmox'; then
    on_pve=1
  fi
  # Heuristic 2: /etc/pve directory exists (Proxmox host config)
  if [ -d /etc/pve ]; then
    on_pve=1
  fi

  if [ "$on_pve" = "0" ]; then
    ok "no Proxmox VE markers — docker path should be safe"
    return
  fi

  warn "Proxmox VE host detected (kernel: $(uname -r 2>/dev/null || echo '?'))"
  hint "club-3090#49 (lexhoefsloot 2026-05-04): the official"
  hint "  vllm/vllm-openai:nightly image hits a uvloop"
  hint "  'event loop already running' crash on PVE 8.x + kernel 6.17.x"
  hint "  during \`vllm serve\` (BEFORE Genesis runs — bare docker repros)."
  hint "Workaround validated by lexhoefsloot: native venv ≡ \"pip install"
  hint "  vllm==0.20.1\" + Genesis on the same host runs clean."

  if [ "$GENESIS_BARE_METAL" = "0" ]; then
    GENESIS_BARE_METAL=1
    info "auto-enabled --bare-metal (you can override with --bare-metal=0)"
  else
    ok "--bare-metal already set; no action needed"
  fi
}

# ─── Workload picker (1 question, or env-driven) ──────────────────────

WORKLOAD_OPTIONS=(
  "balanced|Default-safe — chat + occasional long ctx + occasional tools"
  "long_context|Single long prompt (>50K), low concurrency"
  "high_throughput|Many short prompts in parallel, max TPS"
  "tool_agent|IDE coding agents"
)

pick_workload() {
  step "Pick workload"

  # Validate env-provided value if any
  if [ -n "$GENESIS_WORKLOAD" ]; then
    case "$GENESIS_WORKLOAD" in
      balanced|long_context|high_throughput|tool_agent)
        ok "workload: $GENESIS_WORKLOAD (from --workload)"
        return
        ;;
      *)
        die "invalid --workload '$GENESIS_WORKLOAD'. One of: balanced, long_context, high_throughput, tool_agent"
        ;;
    esac
  fi

  # Non-interactive default
  if [ "$GENESIS_NON_INTERACTIVE" = "1" ] || [ ! -t 0 ]; then
    GENESIS_WORKLOAD="balanced"
    ok "workload: balanced (non-interactive default — re-run with --workload to change)"
    return
  fi

  # Interactive prompt
  echo
  echo "Pick the workload Genesis should optimize for:"
  echo
  local i=1
  for entry in "${WORKLOAD_OPTIONS[@]}"; do
    local key="${entry%%|*}"
    local desc="${entry#*|}"
    printf "  %d) %-18s — %s\n" "$i" "$key" "$desc"
    i=$((i+1))
  done
  echo
  while true; do
    read -rp "Choice [1-${#WORKLOAD_OPTIONS[@]}, default 1=balanced]: " pick
    pick="${pick:-1}"
    if [[ "$pick" =~ ^[1-${#WORKLOAD_OPTIONS[@]}]$ ]]; then
      GENESIS_WORKLOAD="${WORKLOAD_OPTIONS[$((pick-1))]%%|*}"
      ok "workload: $GENESIS_WORKLOAD"
      return
    fi
    echo "  invalid — pick 1-${#WORKLOAD_OPTIONS[@]}"
  done
}

# ─── Model-weights location picker ("where do weights go") ────────────
#
# club-3090's setup.sh prompts "where do your GGUF/weights live" before it
# writes the bind-mount into the compose. Same idea here: detect the common
# host weight/cache locations, present a numbered menu (+ "custom path"), and
# export the choice as GENESIS_MODELS_DIR so the launcher can bind-mount it.
# Skipped (with the first auto-detected candidate) in non-interactive mode or
# when --models-dir was passed.

pick_models_dir() {
  step "Pick model-weights location"

  # Explicit choice wins — validate-light and return.
  if [ -n "$GENESIS_MODELS_DIR" ]; then
    if [ -d "$GENESIS_MODELS_DIR" ]; then
      ok "weights dir: $GENESIS_MODELS_DIR (from --models-dir)"
    else
      warn "weights dir $GENESIS_MODELS_DIR does not exist yet — will be created on first pull"
    fi
    return
  fi

  # Auto-detect candidate locations (only those that exist), de-duplicated.
  local candidates=() seen="" c
  for c in \
    "$HF_HOME/hub" \
    "$HOME/.cache/huggingface/hub" \
    "$HOME/models" \
    "/models" \
    "/data/models" \
    "/srv/models"; do
    [ -n "$c" ] || continue
    [ -d "$c" ] || continue
    case "$seen" in *"|$c|"*) continue ;; esac
    seen="${seen}|$c|"
    candidates+=("$c")
  done

  local default_dir="${candidates[0]:-$HOME/.cache/huggingface/hub}"

  # Non-interactive: take the first detected candidate (or the HF default).
  if [ "$GENESIS_NON_INTERACTIVE" = "1" ] || [ ! -t 0 ]; then
    GENESIS_MODELS_DIR="$default_dir"
    ok "weights dir: $GENESIS_MODELS_DIR (non-interactive default)"
    return
  fi

  echo
  echo "Where do your model weights / HF cache live?"
  echo "(the launcher bind-mounts this into the engine container)"
  echo
  local i=1
  for c in "${candidates[@]}"; do
    printf "  %d) %s\n" "$i" "$c"
    i=$((i+1))
  done
  printf "  %d) custom path…\n" "$i"
  echo
  local pick custom
  while true; do
    read -rp "Choice [1-${i}, default 1]: " pick
    pick="${pick:-1}"
    if [ "$pick" = "$i" ]; then
      while true; do
        read -rp "  Enter the weights directory path: " custom
        custom="${custom/#\~/$HOME}"
        if [ -n "$custom" ]; then
          GENESIS_MODELS_DIR="$custom"
          [ -d "$custom" ] || warn "  $custom does not exist yet — will be created on first pull"
          ok "weights dir: $GENESIS_MODELS_DIR"
          return
        fi
        echo "  a path is required"
      done
    fi
    if [[ "$pick" =~ ^[0-9]+$ ]] && [ "$pick" -ge 1 ] && [ "$pick" -lt "$i" ]; then
      GENESIS_MODELS_DIR="${candidates[$((pick-1))]}"
      ok "weights dir: $GENESIS_MODELS_DIR"
      return
    fi
    echo "  invalid — pick 1-${i}"
  done
}

# ─── Pin picker (numbered menu, club-3090 setup.sh style) ─────────────
#
# club-3090's setup.sh asks the operator to pick an engine build from a
# numbered list before writing the compose. We do the analogue for the
# Genesis ref: a 3-way menu (stable / dev / explicit) with SHA/pin guidance,
# so an interactive operator chooses the install ref deliberately instead of
# silently inheriting 'stable'. An explicit --pin / GENESIS_PIN always wins
# and skips this menu (scriptable, CI-safe).

PIN_OPTIONS=(
  "stable|Latest published v12+ stable tag (recommended for production)"
  "dev|Dev-branch tip — newest features, mutable (re-run to update)"
  "explicit|A specific commit SHA / tag / branch you paste"
)

pick_pin() {
  step "Pick Genesis pin"

  # Explicit pin (flag or non-default env) wins — never prompt over it.
  if [ "$GENESIS_PIN_EXPLICIT" = "1" ]; then
    ok "pin: $GENESIS_PIN (explicit — from --pin / GENESIS_PIN)"
    return
  fi

  # Non-interactive default: keep 'stable' (resolve_pin maps it to a tag).
  if [ "$GENESIS_NON_INTERACTIVE" = "1" ] || [ ! -t 0 ]; then
    ok "pin: stable (non-interactive default — re-run with --pin to change)"
    return
  fi

  echo
  echo "Pick the Genesis ref to install:"
  echo
  local i=1 entry key desc
  for entry in "${PIN_OPTIONS[@]}"; do
    key="${entry%%|*}"; desc="${entry#*|}"
    printf "  %d) %-9s — %s\n" "$i" "$key" "$desc"
    i=$((i+1))
  done
  echo
  hint "SHA/pin guidance: 'stable' is reproducible (a published tag); 'dev' is"
  hint "  mutable; for an audited reproducible install paste an explicit commit"
  hint "  SHA (full 40-char) or tag. Genesis verifies the clone carries the"
  hint "  current top-level sndr/ layout (issue #29) after checkout."
  echo
  local pick
  while true; do
    read -rp "Choice [1-${#PIN_OPTIONS[@]}, default 1=stable]: " pick
    pick="${pick:-1}"
    case "$pick" in
      1) GENESIS_PIN="stable"; ok "pin: stable"; return ;;
      2) GENESIS_PIN="dev"; ok "pin: dev"; return ;;
      3)
        local ref=""
        while [ -z "$ref" ]; do
          read -rp "  Paste commit SHA / tag / branch: " ref
          ref="$(printf '%s' "$ref" | tr -d '[:space:]')"
          [ -z "$ref" ] && echo "  a ref is required (or Ctrl-C to abort)"
        done
        GENESIS_PIN="$ref"; GENESIS_PIN_EXPLICIT=1
        ok "pin: $ref (explicit)"
        return
        ;;
      *) echo "  invalid — pick 1-${#PIN_OPTIONS[@]}" ;;
    esac
  done
}

# ─── Resolve pin (stable | dev | <commit/tag>) ────────────────────────

resolve_pin() {
  # Interactive pin menu first (no-op when explicit / non-interactive).
  pick_pin

  step "Resolve Genesis pin"

  case "$GENESIS_PIN" in
    stable)
      # The package was re-rooted from vllm/sndr_core/ to top-level sndr/
      # in v12.0.0 (the "sndr-platform" refactor). Tags published BEFORE
      # that refactor (v7.x, v11.x) carry the retired layout and make the
      # clone sanity gate fatal with "missing sndr/... — wrong pin?"
      # (issue #29). So `stable` only accepts a tag whose major version is
      # >= GENESIS_MIN_STABLE_MAJOR; otherwise it falls back to `main`,
      # which always carries the current sndr/ layout.
      local min_major="${GENESIS_MIN_STABLE_MAJOR:-12}"
      local tags newest=""
      # Pull up to 30 tag names (newest first per GitHub API ordering).
      tags=$(curl -fsSL --max-time 10 \
        "https://api.github.com/repos/Sandermage/genesis-vllm-patches/tags?per_page=30" \
        2>/dev/null | grep '"name":' | sed -E 's/.*"name": *"([^"]+)".*/\1/' || true)
      if [ -z "$tags" ]; then
        warn "Could not query GitHub tags API — falling back to 'main' branch."
        GENESIS_PIN_RESOLVED="main"
      else
        # Pick the first tag whose leading vMAJOR is >= min_major.
        local t major
        while IFS= read -r t; do
          [ -n "$t" ] || continue
          # Extract leading integer after an optional 'v' (e.g. v12.0.0 → 12).
          major=$(printf '%s' "$t" | sed -E 's/^v?([0-9]+).*/\1/')
          case "$major" in
            ''|*[!0-9]*) continue ;;  # not a vN.* tag — skip
          esac
          if [ "$major" -ge "$min_major" ]; then
            newest="$t"
            break
          fi
        done <<EOF
$tags
EOF
        if [ -n "$newest" ]; then
          GENESIS_PIN_RESOLVED="$newest"
          ok "pin: $newest (latest stable v${min_major}+ tag)"
        else
          warn "No v${min_major}+ stable tag published yet — using 'main' (carries the current sndr/ layout)."
          hint "Pre-v${min_major} tags (v7.x/v11.x) use the retired vllm/sndr_core/ layout and would fail the install sanity gate (issue #29)."
          hint "Pin an exact ref with --pin <commit-or-tag> for reproducibility."
          GENESIS_PIN_RESOLVED="main"
        fi
      fi
      ;;
    dev)
      GENESIS_PIN_RESOLVED="dev"
      ok "pin: dev (latest dev branch tip)"
      hint "dev is mutable — for production use --pin <commit> or --pin stable"
      ;;
    *)
      GENESIS_PIN_RESOLVED="$GENESIS_PIN"
      ok "pin: $GENESIS_PIN (explicit ref)"
      ;;
  esac
}

# ─── Clone or update Genesis at GENESIS_HOME ──────────────────────────

clone_genesis() {
  step "Genesis source ($GENESIS_HOME)"

  if [ -d "$GENESIS_HOME/.git" ]; then
    info "found existing clone — updating"
    if ! git -C "$GENESIS_HOME" fetch --tags origin >/dev/null 2>&1; then
      die "git fetch failed in $GENESIS_HOME"
    fi
    if ! git -C "$GENESIS_HOME" checkout --quiet "$GENESIS_PIN_RESOLVED" 2>/dev/null; then
      die "checkout failed for ref '$GENESIS_PIN_RESOLVED' (does it exist on the remote?)"
    fi
    # If on a branch, fast-forward
    if git -C "$GENESIS_HOME" symbolic-ref -q HEAD >/dev/null 2>&1; then
      git -C "$GENESIS_HOME" pull --ff-only --quiet origin "$GENESIS_PIN_RESOLVED" 2>/dev/null || true
    fi
  else
    info "cloning from $GENESIS_REPO"
    if ! git clone --quiet "$GENESIS_REPO" "$GENESIS_HOME"; then
      die "git clone failed"
    fi
    if ! git -C "$GENESIS_HOME" checkout --quiet "$GENESIS_PIN_RESOLVED" 2>/dev/null; then
      die "checkout failed for ref '$GENESIS_PIN_RESOLVED'"
    fi
  fi

  local sha
  sha=$(git -C "$GENESIS_HOME" rev-parse --short HEAD 2>/dev/null || echo "?")
  ok "Genesis at $sha (ref: $GENESIS_PIN_RESOLVED)"

  # Sanity: required canonical files. v12.0.0 re-rooted the package from
  # vllm/sndr_core/ to top-level sndr/ — these paths MUST track the current
  # layout or every fresh clone fatals "missing ... — wrong pin?" (issue
  # #29). Mirror of sndr.cli.legacy.install._REQUIRED_GENESIS_FILES.
  for f in sndr/__init__.py sndr/apply/orchestrator.py sndr/compat/cli.py; do
    if [ ! -f "$GENESIS_HOME/$f" ]; then
      die "Genesis tree at $GENESIS_PIN_RESOLVED is missing $f — wrong pin? (v12 expects the top-level sndr/ layout; pre-v12 tags use the retired vllm/sndr_core/ — try --pin main)"
    fi
  done
}

# ─── Editable install of the repo root (the v12 canonical wiring step) ───
#
# v12.0.0 ("sndr-platform") replaced the v11 symlink-into-site-packages
# model with a plain editable install of the REPO ROOT. That single step:
#   1. puts the top-level `sndr` package on sys.path (so `import sndr`
#      and `python3 -m sndr.apply` work — docs/INSTALL.md step 3), and
#   2. writes the `vllm.general_plugins` entry-point metadata
#      (genesis_v7 = sndr.plugin:register) into site-packages, so vLLM's
#      load_general_plugins() re-applies the runtime monkey-patches IN
#      every engine + worker process.
# `--no-deps` keeps it fast — a vLLM env already has pyyaml/packaging.

install_plugin() {
  if [ "$GENESIS_NO_PLUGIN_INSTALL" = "1" ]; then
    warn "skipping editable install (--no-plugin) — Genesis won't auto-load in vllm serve"
    return
  fi

  step "Editable install of sndr (vllm.general_plugins entry point)"

  if [ ! -f "$GENESIS_HOME/pyproject.toml" ]; then
    warn "pyproject.toml missing in this Genesis tree — skipping editable install"
    return
  fi

  # Use --user unless --system was given. --no-deps per docs/INSTALL.md
  # step 3 (deps already present in a vLLM environment).
  local pip_args="$PIP_INSTALL_FLAGS"
  info "pip install --no-deps $pip_args -e $GENESIS_HOME  (repo root)"
  if ! "$PYTHON_BIN" -m pip install -q --no-deps $pip_args -e "$GENESIS_HOME" 2>&1 | tail -5; then
    warn "editable install failed — Genesis won't import or auto-load in vllm serve"
    hint "Manual: $PYTHON_BIN -m pip install --no-deps -e $GENESIS_HOME"
    return
  fi
  ok "sndr installed (editable, repo root)"

  # Verify `import sndr` resolves from the same interpreter (docs/INSTALL.md
  # step 3 verification).
  if "$PYTHON_BIN" -c 'import sndr' 2>/dev/null; then
    ok "import sndr OK"
  else
    warn "import sndr failed post-install — check the pip output above"
  fi

  # Verify the vllm.general_plugins entry point is registered — THIS is what
  # lets `vllm serve` re-apply runtime monkey-patches in the serving process.
  if "$PYTHON_BIN" -c 'from importlib.metadata import entry_points; eps = entry_points(group="vllm.general_plugins"); vals = [ep.value for ep in eps]; assert any("sndr.plugin" in v for v in vals), vals' 2>/dev/null; then
    ok "vllm.general_plugins → sndr.plugin:register entry point registered"
  else
    warn "entry point not found post-install — verify vLLM picks Genesis up"
  fi
}

# ─── Verify `import sndr` + clean up retired v11 site-packages artifacts ──
#
# v12 wires `sndr` via the editable install in install_plugin() above — no
# PYTHONPATH symlink dance any more. This step just confirms `import sndr`
# resolves from the target interpreter and scrubs any stale v11 symlinks
# (`vllm/_genesis`, `vllm/sndr_core`) left over from an earlier install so
# they cannot shadow the new top-level package.

setup_pythonpath() {
  step "Verify sndr import + scrub retired v11 artifacts"

  # Primary check: the editable install must make `import sndr` work.
  if "$PYTHON_BIN" -c 'import sndr' 2>/dev/null; then
    ok "import sndr OK (editable install on sys.path)"
  elif [ "$GENESIS_NO_PLUGIN_INSTALL" = "1" ]; then
    # No editable install was done (--no-plugin) → fall back to PYTHONPATH.
    warn "sndr not importable and editable install skipped (--no-plugin)"
    hint "Add to your shell rc:  export PYTHONPATH=\"$GENESIS_HOME:\${PYTHONPATH:-}\""
  else
    warn "sndr not importable even after editable install — check pip output above"
    hint "Manual fallback:  export PYTHONPATH=\"$GENESIS_HOME:\${PYTHONPATH:-}\""
  fi

  # Scrub retired v11 artifacts from vLLM's site-packages (harmless if
  # absent). Pre-v11 used vllm/_genesis/, v11 used vllm/sndr_core/ — both
  # are gone in v12 and a stale symlink could shadow the real package.
  local vllm_path
  vllm_path=$("$PYTHON_BIN" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || echo "")
  if [ -n "$vllm_path" ]; then
    if [ -L "$vllm_path/_genesis" ]; then
      rm -f "$vllm_path/_genesis"
      hint "removed retired symlink $vllm_path/_genesis"
    fi
    if [ -L "$vllm_path/sndr_core" ]; then
      rm -f "$vllm_path/sndr_core"
      hint "removed retired symlink $vllm_path/sndr_core"
    fi
  fi
}

# ─── Generate launch script via preset ────────────────────────────────

generate_launch_script() {
  step "Generate launch script"

  if [ -z "$GPU_CLASS_HINT" ] || [ "$N_GPUS" = "0" ]; then
    warn "no GPU detected — skipping launch script generation"
    hint "Pick a preset manually:  sndr preset list"
    return
  fi

  local out_dir="$GENESIS_HOME/launch"
  mkdir -p "$out_dir"
  local out_file="$out_dir/start_${GPU_CLASS_HINT// /_}_${N_GPUS}x_${GENESIS_WORKLOAD}.sh"

  # Surface the chosen weights dir to the preset-match renderer so the
  # generated launch script bind-mounts the right host path (the launcher
  # reads SNDR_MODELS_DIR / GENESIS_MODELS_DIR for the model mount).
  if [ -n "$GENESIS_MODELS_DIR" ]; then
    export GENESIS_MODELS_DIR SNDR_MODELS_DIR="$GENESIS_MODELS_DIR"
  fi

  # Try matching with GENESIS_HOME on PYTHONPATH so the new clone takes
  # precedence (also covers the --no-plugin path where no editable install
  # ran). v12 module path: sndr.compat.cli.
  if PYTHONPATH="$GENESIS_HOME:${PYTHONPATH:-}" "$PYTHON_BIN" -m sndr.compat.cli preset match \
      --gpu "$GPU_CLASS_HINT" \
      --n-gpus "$N_GPUS" \
      --workload "$GENESIS_WORKLOAD" \
      --script > "$out_file" 2>/dev/null; then
    chmod +x "$out_file"
    ok "wrote launch script: $out_file"
  else
    # Fallback to balanced workload
    if PYTHONPATH="$GENESIS_HOME:${PYTHONPATH:-}" "$PYTHON_BIN" -m sndr.compat.cli preset match \
        --gpu "$GPU_CLASS_HINT" \
        --n-gpus "$N_GPUS" \
        --workload balanced \
        --script > "$out_file" 2>/dev/null; then
      chmod +x "$out_file"
      warn "no preset for ($GPU_CLASS_HINT × $N_GPUS × $GENESIS_WORKLOAD); used balanced fallback"
      ok "wrote launch script: $out_file"
    else
      warn "no preset matches your hardware combination — pick manually:"
      hint "  sndr preset list"
      rm -f "$out_file"
      return
    fi
  fi

  LAUNCH_SCRIPT="$out_file"
}

# ─── Verify (smoke test, optional, via `sndr.compat.cli verify`) ───

run_verify() {
  if [ "$GENESIS_NO_VERIFY" = "1" ]; then
    warn "skipping verify (--no-verify)"
    return
  fi

  step "Verify install"

  if ! PYTHONPATH="$GENESIS_HOME:${PYTHONPATH:-}" "$PYTHON_BIN" -m sndr.compat.cli verify --quick 2>&1 | sed 's/^/    /'; then
    warn "verify reported issues — check output above. Genesis is installed but may not be fully functional."
    hint "Diagnose:  sndr doctor   (or: python3 -m sndr.compat.cli doctor)"
    return 0  # Don't fail install on verify warnings
  fi
}

# ─── Print next steps ─────────────────────────────────────────────────

print_next_steps() {
  echo
  printf '%b\n' "${C_GREEN}${C_BOLD}✓ Genesis installed.${C_RESET}"
  echo
  echo "  Location:  $GENESIS_HOME"
  echo "  Pin:       $(git -C "$GENESIS_HOME" rev-parse --short HEAD) ($GENESIS_PIN_RESOLVED)"
  echo "  Plugin:    $([ "$GENESIS_NO_PLUGIN_INSTALL" = "1" ] && echo 'skipped' || echo 'installed (auto-loads in vllm serve)')"
  if [ -n "${GENESIS_MODELS_DIR:-}" ]; then
    echo "  Weights:   $GENESIS_MODELS_DIR"
  fi
  if [ -n "${LAUNCH_SCRIPT:-}" ]; then
    echo "  Launch:    $LAUNCH_SCRIPT"
  fi
  echo
  echo "Next:"
  if [ "$GENESIS_BARE_METAL" = "1" ]; then
    echo "  Bare-metal mode (--bare-metal or auto-enabled by Proxmox detect):"
    echo "      $PYTHON_BIN -m pip install --user vllm==0.20.1   # if not already"
    echo "      genesis verify                                    # full smoke test"
    echo "      vllm serve <model> --tensor-parallel-size <N> ...  # standard vllm CLI"
    echo
    echo "  Generated launch scripts in $GENESIS_HOME/scripts/ are docker-based"
    echo "  by default. To bare-metal-ify any of them, replace the"
    echo "    docker run ... vllm/vllm-openai:nightly ... vllm serve ..."
    echo "  block with a direct \`vllm serve\` invocation using the same flags."
    echo
    if [ -n "${LAUNCH_SCRIPT:-}" ]; then
      echo "  Reference (your detected preset):  $LAUNCH_SCRIPT"
    fi
  elif [ -n "${LAUNCH_SCRIPT:-}" ]; then
    echo "  Edit the launch script (set MODEL_PATH if needed), then:"
    echo "      bash $LAUNCH_SCRIPT"
  else
    echo "  Browse presets and pick one for your rig:"
    echo "      sndr preset list                 # available presets"
    echo "      sndr preset show <alias>         # inspect a preset"
    echo "      sndr launch <alias>              # boot the preset"
  fi
  echo
  echo "Useful commands:"
  echo "  sndr doctor              # full system diagnostic"
  echo "  sndr preset list         # browse production presets"
  echo "  sndr report bundle       # diagnostic bundle for issues"
  # verify / model-config / patches live on the legacy 'genesis' command
  # tree (not yet promoted to the canonical 'sndr' surface).
  echo "  genesis verify           # re-run smoke test"
  echo "  genesis model-config list  # vetted model launch configs"
  echo "  genesis patches doctor   # registry sanity check"
  echo "  sndr --help              # full subcommand list"
  echo
  echo "Docs:    https://github.com/Sandermage/genesis-vllm-patches"
  echo "Issues:  https://github.com/Sandermage/genesis-vllm-patches/issues"
}

# ─── Uninstall ────────────────────────────────────────────────────────

uninstall() {
  step "Uninstall Genesis"

  # Scrub any retired v11 site-packages symlinks (vllm/_genesis,
  # vllm/sndr_core). v12 uses an editable install, not symlinks.
  local vllm_path
  vllm_path=$("$PYTHON_BIN" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || echo "")
  if [ -n "$vllm_path" ]; then
    if [ -L "$vllm_path/_genesis" ]; then
      rm -f "$vllm_path/_genesis"
      ok "removed retired symlink $vllm_path/_genesis"
    fi
    if [ -L "$vllm_path/sndr_core" ]; then
      rm -f "$vllm_path/sndr_core"
      ok "removed retired symlink $vllm_path/sndr_core"
    fi
  fi

  # Uninstall the v12 editable package (sndr-platform) and the legacy
  # back-compat plugin (genesis-vllm-plugin) if either is present.
  for pkg in sndr-platform genesis-vllm-plugin; do
    if "$PYTHON_BIN" -m pip show "$pkg" >/dev/null 2>&1; then
      "$PYTHON_BIN" -m pip uninstall -y -q "$pkg" >/dev/null 2>&1 || true
      ok "uninstalled $pkg"
    fi
  done

  # NOTE: We do NOT delete $GENESIS_HOME automatically — it may contain
  # user-generated launch scripts in $GENESIS_HOME/launch/. Caller can
  # `rm -rf $GENESIS_HOME` if they want a full wipe.
  warn "Genesis source tree at $GENESIS_HOME left in place."
  hint "To fully remove:  rm -rf $GENESIS_HOME"

  warn "Text-patches in vllm/ install were NOT reverted by this script."
  hint "To revert text-patches: pip uninstall vllm && pip install vllm  (re-install clean)"
}

# ─── Main flow ────────────────────────────────────────────────────────

main() {
  echo
  printf '%b\n' "${C_BOLD}Genesis vLLM Patches — installer${C_RESET}"
  printf '%b\n' "${C_GRAY}https://github.com/Sandermage/genesis-vllm-patches${C_RESET}"
  echo

  if [ "$GENESIS_UNINSTALL" = "1" ]; then
    uninstall
    exit 0
  fi

  preflight
  detect_gpu
  detect_vllm
  detect_proxmox_runtime
  pick_workload
  pick_models_dir
  resolve_pin
  clone_genesis
  install_plugin
  setup_pythonpath
  generate_launch_script
  run_verify
  print_next_steps
}

main "$@"
