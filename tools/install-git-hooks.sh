#!/bin/sh
# Point this clone's Git hooks at the tracked scripts under .githooks/.
set -e

root=$(git rev-parse --show-toplevel)
hooks=$(git rev-parse --git-path hooks)
mkdir -p "$hooks"
ln -sf "$root/.githooks/pre-commit" "$hooks/pre-commit"
chmod +x "$root/.githooks/pre-commit"
echo "installed pre-commit hook -> $hooks/pre-commit"
