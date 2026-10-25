#!/usr/bin/env bash
# Reproducible presubmit for the agate workspace. Run from anywhere:
#
#   ./tools/presubmit.sh              # stages 1-4 (always)
#
# Stages:
#   1. agate test suite, debug   (unit registry + zig fmt gate)
#   2. agate test suite, safe
#   3. sandbox native build      (downstream consumer against the API)
#   4. agate-tools node gates    (gate logic + regression tests)
#
# Notes:
# - agate-tools and the sandbox live OUTSIDE the agate git repo (workspace
#   siblings), so CI runs only the in-repo subset; this script is the full
#   local presubmit.
set -euo pipefail

AGATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$AGATE_DIR")"

step() { printf '\n== %s ==\n' "$*"; }

step "1/4 agate tests (debug)"
(cd "$AGATE_DIR" && zig build test -Doptimize=debug --summary line --test-timeout 180s)

step "2/4 agate tests (safe)"
(cd "$AGATE_DIR" && zig build test -Doptimize=safe --summary line --test-timeout 180s)

step "3/4 sandbox native build"
(cd "$WORKSPACE/sandbox" && zig build -Doptimize=debug --summary line -j4)
(cd "$WORKSPACE/sandbox" && zig build test -Doptimize=debug --summary line -j4)

step "4/4 agate-tools node gates"
(cd "$WORKSPACE" && node --test agate-tools/*.test.mjs)

step "presubmit OK"
