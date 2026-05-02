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
#    2. Resolves pin (default: latest stable tag; --pin dev = dev tip)
#    3. Clones Genesis into ~/.genesis/ (or $GENESIS_HOME)
#    4. pip install -e <repo>/genesis_vllm_plugin  (so vLLM auto-loads
#       Genesis via vllm.general_plugins entry point in main + workers)
#    5. Auto-matches a preset for your (gpu × workload) and writes a
#       runnable launch script
#    6. Runs `genesis verify` — 60-second smoke test
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
GENESIS_HOME="${GENESIS_HOME:-${HOME}/.genesis}"
GENESIS_PIN="${GENESIS_PIN:-stable}"     # 'stable' (latest tag) | 'dev' | <commit-or-tag>
GENESIS_WORKLOAD="${GENESIS_WORKLOAD:-}" # one of: long_context, high_throughput, tool_agent, balanced
GENESIS_NON_INTERACTIVE="${GENESIS_NON_INTERACTIVE:-0}"
GENESIS_NO_VERIFY="${GENESIS_NO_VERIFY:-0}"
GENESIS_NO_PLUGIN_INSTALL="${GENESIS_NO_PLUGIN_INSTALL:-0}"
GENESIS_UNINSTALL=0

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_INSTALL_FLAGS="${PIP_INSTALL_FLAGS:---user}"  # safer default than system-wide

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
    --pin) GENESIS_PIN="$2"; shift 2 ;;
    --pin=*) GENESIS_PIN="${1#*=}"; shift ;;
    --workload) GENESIS_WORKLOAD="$2"; shift 2 ;;
    --workload=*) GENESIS_WORKLOAD="${1#*=}"; shift ;;
    --home) GENESIS_HOME="$2"; shift 2 ;;
    --home=*) GENESIS_HOME="${1#*=}"; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --no-verify) GENESIS_NO_VERIFY=1; shift ;;
    --no-plugin) GENESIS_NO_PLUGIN_INSTALL=1; shift ;;
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
                         stable  = latest tag
                         dev     = dev branch tip (mutable)
                       Or any commit/tag/branch
  --workload <name>    One of: balanced, long_context, high_throughput,
                       tool_agent (default: interactive prompt or 'balanced')
  --home <path>        Where to install Genesis (default: ~/.genesis)
  --python <path>      Python interpreter to use (default: python3)
  --no-verify          Skip post-install smoke test
  --no-plugin          Skip pip install of genesis_vllm_plugin
                       (Genesis still works via PYTHONPATH but won't
                        auto-load in vllm spawn workers)
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
  GENESIS_NON_INTERACTIVE, GENESIS_NO_VERIFY, GENESIS_NO_PLUGIN_INSTALL,
  PYTHON_BIN, PIP_INSTALL_FLAGS

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
  # Mirror of the keys in vllm/_genesis/gpu_profile.py:GPU_SPECS
  case "$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]')" in
    *"rtx 3060"*) GPU_CLASS_HINT="rtx 3060" ;;
    *"rtx 3070"*) GPU_CLASS_HINT="rtx 3070" ;;
    *"rtx 3080"*) GPU_CLASS_HINT="rtx 3080" ;;
    *"rtx 3090"*) GPU_CLASS_HINT="rtx 3090" ;;
    *"rtx a4000"*) GPU_CLASS_HINT="rtx a4000" ;;
    *"rtx a5000"*) GPU_CLASS_HINT="rtx a5000" ;;
    *"rtx a6000"*) GPU_CLASS_HINT="rtx a6000" ;;
    *"a100"*) GPU_CLASS_HINT="a100" ;;
    *"rtx 4070"*) GPU_CLASS_HINT="rtx 4070" ;;
    *"rtx 4080"*) GPU_CLASS_HINT="rtx 4080" ;;
    *"rtx 4090"*) GPU_CLASS_HINT="rtx 4090" ;;
    *"l40"*) GPU_CLASS_HINT="l40" ;;
    *"rtx 6000 ada"*) GPU_CLASS_HINT="rtx 6000 ada" ;;
    *"h100"*) GPU_CLASS_HINT="h100" ;;
    *"h200"*) GPU_CLASS_HINT="h200" ;;
    *"h20"*) GPU_CLASS_HINT="h20" ;;
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
    hint "See docs/COMPATIBILITY.md or run \`genesis doctor\` after install."
  fi
}

# ─── Workload picker (1 question, or env-driven) ──────────────────────

WORKLOAD_OPTIONS=(
  "balanced|Default-safe — chat + occasional long ctx + occasional tools"
  "long_context|Single long prompt (>50K), low concurrency"
  "high_throughput|Many short prompts in parallel, max TPS"
  "tool_agent|IDE coding agents (Cline / Claude Code / OpenCode)"
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

# ─── Resolve pin (stable | dev | <commit/tag>) ────────────────────────

resolve_pin() {
  step "Resolve Genesis pin"

  case "$GENESIS_PIN" in
    stable)
      # Latest tag matching v*.* (use GitHub API; soft-fail to 'main')
      local tag
      tag=$(curl -fsSL --max-time 10 \
        "https://api.github.com/repos/Sandermage/genesis-vllm-patches/tags?per_page=10" \
        2>/dev/null | grep -m1 '"name":' | sed -E 's/.*"name": *"([^"]+)".*/\1/' || true)
      if [ -n "$tag" ]; then
        GENESIS_PIN_RESOLVED="$tag"
        ok "pin: $tag (latest stable tag)"
      else
        warn "Could not query GitHub tags API — falling back to 'main' branch."
        GENESIS_PIN_RESOLVED="main"
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

  # Sanity: required files
  for f in vllm/_genesis/__init__.py vllm/_genesis/patches/apply_all.py vllm/_genesis/compat/cli.py; do
    if [ ! -f "$GENESIS_HOME/$f" ]; then
      die "Genesis tree at $GENESIS_PIN_RESOLVED is missing $f — wrong pin?"
    fi
  done
}

# ─── Install genesis_vllm_plugin (so vLLM auto-loads us in workers) ───

install_plugin() {
  if [ "$GENESIS_NO_PLUGIN_INSTALL" = "1" ]; then
    warn "skipping plugin install (--no-plugin) — Genesis won't auto-load in vllm serve"
    return
  fi

  step "Install genesis-vllm-plugin (vllm.general_plugins entry point)"

  if [ ! -d "$GENESIS_HOME/genesis_vllm_plugin" ]; then
    warn "genesis_vllm_plugin/ missing in this Genesis tree — skipping"
    return
  fi

  # Use --user unless --system was given
  local pip_args="$PIP_INSTALL_FLAGS"
  info "pip install -e $GENESIS_HOME/genesis_vllm_plugin/  (flags: $pip_args)"
  if ! "$PYTHON_BIN" -m pip install -q $pip_args -e "$GENESIS_HOME/genesis_vllm_plugin/" 2>&1 | tail -5; then
    warn "plugin pip install failed — Genesis still works via PYTHONPATH but won't auto-load in spawn workers"
    hint "Manual: $PYTHON_BIN -m pip install -e $GENESIS_HOME/genesis_vllm_plugin/"
    return
  fi
  ok "genesis-vllm-plugin installed"

  # Verify the entry point is registered
  if "$PYTHON_BIN" -c 'from importlib.metadata import entry_points; eps = entry_points(group="vllm.general_plugins"); names = [ep.name for ep in eps]; assert "genesis_v7" in names, names' 2>/dev/null; then
    ok "vllm.general_plugins → genesis_v7 entry point registered"
  else
    warn "entry point not found post-install — verify vLLM picks Genesis up"
  fi
}

# ─── Add Genesis to PYTHONPATH (so `import vllm._genesis` works) ──────

setup_pythonpath() {
  step "Wire vllm._genesis into PYTHONPATH"

  # Two strategies:
  # 1. If vllm is in a writeable site-packages → symlink/copy _genesis there (most reliable)
  # 2. Otherwise → emit a profile.d hint with PYTHONPATH

  local vllm_path
  vllm_path=$("$PYTHON_BIN" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || echo "")

  if [ -z "$vllm_path" ]; then
    warn "vllm not importable — skipping PYTHONPATH wire"
    return
  fi

  if [ -L "$vllm_path/_genesis" ] || [ -d "$vllm_path/_genesis" ]; then
    # Existing symlink or dir — replace with current
    if [ -L "$vllm_path/_genesis" ]; then
      rm -f "$vllm_path/_genesis"
    fi
  fi

  if [ -w "$vllm_path" ]; then
    ln -sf "$GENESIS_HOME/vllm/_genesis" "$vllm_path/_genesis"
    ok "symlinked $vllm_path/_genesis → $GENESIS_HOME/vllm/_genesis"
  else
    warn "$vllm_path not writeable — set PYTHONPATH manually"
    hint "Add to your shell rc:  export PYTHONPATH=\"$GENESIS_HOME:\${PYTHONPATH:-}\""
  fi
}

# ─── Generate launch script via preset ────────────────────────────────

generate_launch_script() {
  step "Generate launch script"

  if [ -z "$GPU_CLASS_HINT" ] || [ "$N_GPUS" = "0" ]; then
    warn "no GPU detected — skipping launch script generation"
    hint "Pick a preset manually:  python3 -m vllm._genesis.compat.cli preset list"
    return
  fi

  local out_dir="$GENESIS_HOME/launch"
  mkdir -p "$out_dir"
  local out_file="$out_dir/start_${GPU_CLASS_HINT// /_}_${N_GPUS}x_${GENESIS_WORKLOAD}.sh"

  # Try matching with GENESIS_HOME on PYTHONPATH so the new clone takes precedence
  if PYTHONPATH="$GENESIS_HOME:${PYTHONPATH:-}" "$PYTHON_BIN" -m vllm._genesis.compat.cli preset match \
      --gpu "$GPU_CLASS_HINT" \
      --n-gpus "$N_GPUS" \
      --workload "$GENESIS_WORKLOAD" \
      --script > "$out_file" 2>/dev/null; then
    chmod +x "$out_file"
    ok "wrote launch script: $out_file"
  else
    # Fallback to balanced workload
    if PYTHONPATH="$GENESIS_HOME:${PYTHONPATH:-}" "$PYTHON_BIN" -m vllm._genesis.compat.cli preset match \
        --gpu "$GPU_CLASS_HINT" \
        --n-gpus "$N_GPUS" \
        --workload balanced \
        --script > "$out_file" 2>/dev/null; then
      chmod +x "$out_file"
      warn "no preset for ($GPU_CLASS_HINT × $N_GPUS × $GENESIS_WORKLOAD); used balanced fallback"
      ok "wrote launch script: $out_file"
    else
      warn "no preset matches your hardware combination — pick manually:"
      hint "  python3 -m vllm._genesis.compat.cli preset list"
      rm -f "$out_file"
      return
    fi
  fi

  LAUNCH_SCRIPT="$out_file"
}

# ─── Verify (smoke test, optional, requires Day 3 `genesis verify`) ───

run_verify() {
  if [ "$GENESIS_NO_VERIFY" = "1" ]; then
    warn "skipping verify (--no-verify)"
    return
  fi

  step "Verify install"

  if ! PYTHONPATH="$GENESIS_HOME:${PYTHONPATH:-}" "$PYTHON_BIN" -m vllm._genesis.compat.cli verify --quick 2>&1 | sed 's/^/    /'; then
    warn "verify reported issues — check output above. Genesis is installed but may not be fully functional."
    hint "Diagnose:  python3 -m vllm._genesis.compat.cli doctor"
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
  if [ -n "${LAUNCH_SCRIPT:-}" ]; then
    echo "  Launch:    $LAUNCH_SCRIPT"
  fi
  echo
  echo "Next:"
  if [ -n "${LAUNCH_SCRIPT:-}" ]; then
    echo "  Edit the launch script (set MODEL_PATH if needed), then:"
    echo "      bash $LAUNCH_SCRIPT"
  else
    echo "  Browse presets and pick one for your rig:"
    echo "      python3 -m vllm._genesis.compat.cli preset list"
    echo "      python3 -m vllm._genesis.compat.cli preset show <key> --script"
  fi
  echo
  echo "Useful commands:"
  echo "  genesis doctor          # full system diagnostic"
  echo "  genesis preset auto     # auto-pick preset for this rig"
  echo "  genesis verify          # re-run smoke test"
  echo "  genesis explain P103    # per-patch deep-dive"
  echo
  echo "Docs:    https://github.com/Sandermage/genesis-vllm-patches"
  echo "Issues:  https://github.com/Sandermage/genesis-vllm-patches/issues"
}

# ─── Uninstall ────────────────────────────────────────────────────────

uninstall() {
  step "Uninstall Genesis"

  # Remove symlink in vllm site-packages
  local vllm_path
  vllm_path=$("$PYTHON_BIN" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || echo "")
  if [ -n "$vllm_path" ] && [ -L "$vllm_path/_genesis" ]; then
    rm -f "$vllm_path/_genesis"
    ok "removed symlink $vllm_path/_genesis"
  fi

  # Uninstall plugin
  if "$PYTHON_BIN" -m pip show genesis-vllm-plugin >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip uninstall -y -q genesis-vllm-plugin >/dev/null 2>&1 || true
    ok "uninstalled genesis-vllm-plugin"
  fi

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
  pick_workload
  resolve_pin
  clone_genesis
  install_plugin
  setup_pythonpath
  generate_launch_script
  run_verify
  print_next_steps
}

main "$@"
