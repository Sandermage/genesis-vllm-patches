#!/usr/bin/env bash
# Genesis universal launcher — ONE entry point for ALL configs.
#
# Usage:
#   ./scripts/launch.sh <config-key>          # actually launch
#   ./scripts/launch.sh <config-key> --dry    # render to stdout, no exec
#   ./scripts/launch.sh list                  # show all available configs
#
# All config knowledge lives in vllm/_genesis/model_configs/{builtin,
# community,user}/<key>.yaml. To add your own:
#   genesis model-config new my-key --template a5000-2x-35b-prod
#   <edit ~/.genesis/model_configs/my-key.yaml>
#   ./scripts/launch.sh my-key
#
# This replaces the 18 scripts/launch/start_*.sh + bare_metal_*.sh
# duplicates that used to scatter config knowledge.
set -euo pipefail

if [ $# -eq 0 ] || [ "$1" = "list" ]; then
  exec python3 -m sndr.compat.cli model-config list
fi

KEY="$1"
shift

case "${1:-}" in
  --dry|--dry-run)
    exec python3 -m sndr.compat.cli model-config render "$KEY"
    ;;
  --validate)
    exec python3 -m sndr.compat.cli model-config validate "$KEY"
    ;;
  --preflight)
    exec python3 -m sndr.compat.cli model-config preflight "$KEY"
    ;;
  --help|-h)
    cat <<'EOF'
Genesis universal launcher

  ./scripts/launch.sh <key>             launch config (auto-runs preflight)
  ./scripts/launch.sh <key> --dry       render bash script, don't execute
  ./scripts/launch.sh <key> --validate  schema + 16 audit rules
  ./scripts/launch.sh <key> --preflight env check (mounts/GPU/pin)
  ./scripts/launch.sh list              show all configs

After launch, use:
  python3 -m sndr.compat.cli model-config diagnose <key>
  python3 -m sndr.compat.cli model-config verify <key>
EOF
    exit 0
    ;;
  "")
    # Pass-through: actual launch (uses preflight gate, see model_config_cli)
    exec python3 -m sndr.compat.cli model-config launch "$KEY"
    ;;
  *)
    # Pass remaining args through to the launch subcommand
    exec python3 -m sndr.compat.cli model-config launch "$KEY" "$@"
    ;;
esac
