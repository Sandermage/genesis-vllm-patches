#!/usr/bin/env bash
# Install Genesis git hooks.
# Usage: bash scripts/git/install.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ ! -d .git ]; then
    echo "✗ not in a git repo (no .git dir)"
    exit 1
fi

mkdir -p .git/hooks

for hook in pre-commit; do
    src="$REPO_ROOT/scripts/git/$hook"
    dst="$REPO_ROOT/.git/hooks/$hook"
    if [ ! -f "$src" ]; then
        echo "✗ source hook not found: $src"
        exit 1
    fi
    chmod +x "$src"
    if [ -L "$dst" ] || [ -f "$dst" ]; then
        echo "  $hook hook already exists; backing up to $dst.bak"
        mv "$dst" "$dst.bak.$(date +%s)"
    fi
    ln -sf "$src" "$dst"
    echo "✓ installed $hook → $dst (symlink to $src)"
done

echo
echo "Genesis git hooks installed. Skip with: git commit --no-verify"
