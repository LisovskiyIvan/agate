#!/usr/bin/env bash
# Reproducible presubmit for the agate workspace. Run from anywhere:
#
#   ./tools/presubmit.sh              # stages 1-5 (always)
#   ./tools/presubmit.sh --pixel-gates  # stages 1-5 + the browser pixel gates
#
# Stages:
#   1. agate test suite, debug   (unit registry + zig fmt gate)
#   2. agate test suite, safe
#   3. examples native build     (every consumer executable compiles)
#   4. sandbox native build      (downstream consumer against the API)
#   5. agate-tools node gates    (gate logic + examples manifest gate)
#
# Stage 6 (--pixel-gates) is deliberately separate and explicit: the browser
# WebGPU pixel gates need prebuilt emscripten pages and real headless Chrome.
# `node --test` does NOT replace them, and a native build of an OS backend is
# NOT a live driver check. They require the web build first:
#
#   cd examples && zig build -Dtarget=wasm32-emscripten -Doptimize=ReleaseFast
#
# Notes:
# - agate-tools and the sandbox live OUTSIDE the agate git repo (workspace
#   siblings), so CI runs only the in-repo subset; this script is the full
#   local presubmit.
# - The gate scripts' default --serve-dir assumes agate-tools sits inside
#   agate/. In this workspace it is hoisted one level up, so the pixel stage
#   passes --serve-dir explicitly (override with AGATE_PIXEL_SERVE_DIR).
set -euo pipefail

AGATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$AGATE_DIR")"
PIXEL_SERVE_DIR="${AGATE_PIXEL_SERVE_DIR:-$AGATE_DIR/examples/zig-out/web}"

step() { printf '\n== %s ==\n' "$*"; }

step "1/6 agate tests (debug)"
(cd "$AGATE_DIR" && zig build test -Doptimize=debug --summary line --test-timeout 180s)

step "2/6 agate tests (safe)"
(cd "$AGATE_DIR" && zig build test -Doptimize=safe --summary line --test-timeout 180s)

step "3/6 examples native build"
(cd "$AGATE_DIR/examples" && zig build -Doptimize=debug --summary line -j4)

step "4/6 sandbox native build"
(cd "$WORKSPACE/sandbox" && zig build -Doptimize=debug --summary line -j4)

step "5/6 agate-tools node gates"
(cd "$WORKSPACE" && node --test agate-tools/*.test.mjs)

if [[ "${1:-}" == "--pixel-gates" ]]; then
    step "6/6 pixel gates (headless Chrome on WebGPU)"
    if [[ ! -d "$PIXEL_SERVE_DIR" ]]; then
        echo "error: no prebuilt web pages in $PIXEL_SERVE_DIR" >&2
        echo "build them first: cd examples && zig build -Dtarget=wasm32-emscripten -Doptimize=ReleaseFast" >&2
        exit 1
    fi
    for gate in hdr_browser_gate render_scale_browser_gate taa_browser_gate pbr_variant_gate; do
        echo "== node agate-tools/$gate.mjs =="
        (cd "$WORKSPACE" && node "agate-tools/$gate.mjs" --serve-dir "$PIXEL_SERVE_DIR")
    done
else
    echo
    echo "pixel gates skipped: they are a separate explicit stage (browser WebGPU),"
    echo "run: $0 --pixel-gates"
fi

step "presubmit OK"
