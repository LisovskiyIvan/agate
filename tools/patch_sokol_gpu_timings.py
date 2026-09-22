#!/usr/bin/env python3
"""Patch vendored sokol-gfx with frame-level GPU timings (Metal).

Why
---
Upstream sokol-gfx has no GPU-timestamp mechanism: `sg_query_*` covers
features/limits/resources, and `SOKOL_TRACE_HOOKS` installs CPU-side
begin/end callbacks (trace hooks), not GPU execution time. The engine
profiler therefore could only record CPU-submit times (see
`src/agate/profiler/*`: Shadow/Main/Post are timers around sg submit
calls, explicitly NOT GPU time). This patch adds the smallest possible
true-GPU measurement: one timer per frame on the Metal backend, read
from `MTLCommandBuffer.GPUStartTime/GPUEndTime` after completion.

What it changes
--------------
Three anchored insertions in `vendor/sokol/src/sokol/c/sokol_gfx.h`
(C source, NOT generated — applied in place, documented in
`vendor/sokol/README.agate.md` under "GPU timings patch"):

1. Public declarations after `sg_commit`:
     `sg_agate_set_gpu_timing_enabled(bool)` (default OFF),
     `sg_agate_query_gpu_frame_ms()` (measured ms, or -1).
2. Retain hook in `_sg_mtl_commit`: while enabled, the committed
   command buffer is retained one extra frame (previous one sampled
   then released, so at most one extra buffer is ever alive).
3. Public implementations after `sg_commit`: the query samples the
   retained buffer and returns the last-completed cache. Sampling reads
   `status`; on `MTLCommandBufferStatusCompleted` the cache refreshes
   from `(GPUEndTime-GPUStartTime)*1000`. Sampling the previous buffer
   at replace time (a full frame after its commit) is what makes the
   value converge: a just-committed buffer still reports Committed, not
   Completed. Non-Metal backends (GL/D3D11/dummy/...) always return -1;
   per-pass timers are out of scope.

Disabled path: one predictable branch in `_sg_mtl_commit`, no retained
objects, query short-circuits to -1.

Usage
-----
    python3 tools/patch_sokol_gpu_timings.py            # patch in place
    python3 tools/patch_sokol_gpu_timings.py --check    # report only

The script is idempotent: applied blocks are detected by their
`AGATE GPU TIMINGS` markers and skipped, so it can be re-run after
refreshing `vendor/sokol` from upstream.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

DEFAULT_TARGET = "vendor/sokol/src/sokol/c/sokol_gfx.h"
MARKER = "AGATE GPU TIMINGS"

DECL_ANCHOR = "SOKOL_GFX_API_DECL void sg_commit(void);\n"
DECL_BLOCK = """SOKOL_GFX_API_DECL void sg_commit(void);
// AGATE GPU TIMINGS (vendored patch, see vendor/sokol/README.agate.md):
// frame-level GPU execution time for the Metal backend, off by default.
// sg_agate_set_gpu_timing_enabled(true) retains each committed command
// buffer; sg_agate_query_gpu_frame_ms() returns the last COMPLETED frame's
// (GPUEndTime-GPUStartTime) in ms, or -1 when unavailable/not ready yet.
// Non-Metal backends always return -1. Trace hooks (SOKOL_TRACE_HOOKS) are
// unrelated: they are CPU-side begin/end callbacks, not GPU timestamps.
SOKOL_GFX_API_DECL void sg_agate_set_gpu_timing_enabled(bool enabled);
SOKOL_GFX_API_DECL float sg_agate_query_gpu_frame_ms(void);
"""

COMMIT_ANCHOR = """    // commit the frame's command buffer
    if (_sg.mtl.cmd_buffer) {
        [_sg.mtl.cmd_buffer commit];
    }
"""
COMMIT_BLOCK = """    // commit the frame's command buffer
    if (_sg.mtl.cmd_buffer) {
        [_sg.mtl.cmd_buffer commit];
    }
    // AGATE GPU TIMINGS (vendored patch): while enabled, hold the committed
    // buffer one extra frame so sg_agate_query_gpu_frame_ms() can serve it.
    // Context thread only; the in-flight semaphore handshake is untouched.
    // The previous completed buffer is sampled first (it committed a full
    // frame ago, so it has normally finished on GPU); at most one extra
    // buffer is ever retained.
    if (_sg_agate_gpu_timing_enabled && _sg.mtl.cmd_buffer) {
        _sg_agate_gpu_sample(_sg_agate_gpu_cb);
        if (nil != _sg_agate_gpu_cb) {
            [_sg_agate_gpu_cb release];
        }
        [_sg.mtl.cmd_buffer retain];
        _sg_agate_gpu_cb = _sg.mtl.cmd_buffer;
    }
"""

GLOBALS_ANCHOR = """_SOKOL_PRIVATE void _sg_mtl_commit(void) {
"""
GLOBALS_BLOCK = """// AGATE GPU TIMINGS (vendored patch, see vendor/sokol/README.agate.md):
// file-static state for the frame-level Metal GPU timer. Written on the
// context thread only (commit hook + query poll), so no atomics: the Metal
// runtime publishes GPUStartTime/GPUEndTime once the buffer completes.
#if defined(SOKOL_METAL)
static bool _sg_agate_gpu_timing_enabled = false;
static id<MTLCommandBuffer> _sg_agate_gpu_cb = nil;
static float _sg_agate_gpu_last_ms = -1.0f;
// Refresh the last-completed cache from buf once it has finished on GPU.
// Called for the previous frame's buffer at replace time (a full frame
// after its commit, so it has normally completed) and opportunistically
// for the current buffer on query.
_SOKOL_PRIVATE void _sg_agate_gpu_sample(id<MTLCommandBuffer> buf) {
    if ((nil != buf) && ([buf status] == MTLCommandBufferStatusCompleted)) {
        const CFTimeInterval start = [buf GPUStartTime];
        const CFTimeInterval end = [buf GPUEndTime];
        if ((end > start) && ((end - start) < 10.0)) {
            _sg_agate_gpu_last_ms = (float)((end - start) * 1000.0);
        }
    }
}
#else
static bool _sg_agate_gpu_timing_enabled = false;
#endif
// ---------------------------------------------------------------------------
_SOKOL_PRIVATE void _sg_mtl_commit(void) {
"""

IMPL_ANCHOR = """SOKOL_API_IMPL void sg_commit(void) {
    SOKOL_ASSERT(_sg.valid);
    SOKOL_ASSERT(!_sg.cur_pass.valid);
    SOKOL_ASSERT(!_sg.cur_pass.in_pass);
    _sg_commit();
    _sg_update_stats();
    _sg_notify_commit_listeners();
    _SG_TRACE_NOARGS(commit);
    _sg.frame_index++;
}
"""
IMPL_BLOCK = """SOKOL_API_IMPL void sg_commit(void) {
    SOKOL_ASSERT(_sg.valid);
    SOKOL_ASSERT(!_sg.cur_pass.valid);
    SOKOL_ASSERT(!_sg.cur_pass.in_pass);
    _sg_commit();
    _sg_update_stats();
    _sg_notify_commit_listeners();
    _SG_TRACE_NOARGS(commit);
    _sg.frame_index++;
}
// AGATE GPU TIMINGS (vendored patch, see vendor/sokol/README.agate.md) ---
SOKOL_API_IMPL void sg_agate_set_gpu_timing_enabled(bool enabled) {
    _sg_agate_gpu_timing_enabled = enabled;
    #if defined(SOKOL_METAL)
        if (!enabled) {
            if (nil != _sg_agate_gpu_cb) {
                [_sg_agate_gpu_cb release];
                _sg_agate_gpu_cb = nil;
            }
            _sg_agate_gpu_last_ms = -1.0f;
        }
    #endif
}
SOKOL_API_IMPL float sg_agate_query_gpu_frame_ms(void) {
    #if defined(SOKOL_METAL)
        if (!_sg_agate_gpu_timing_enabled || (nil == _sg_agate_gpu_cb)) {
            return -1.0f;
        }
        _sg_agate_gpu_sample(_sg_agate_gpu_cb);
        // Last-completed semantics: the value lags one frame behind the
        // CPU submit (async GPU execution). -1 until the first completion.
        return _sg_agate_gpu_last_ms;
    #else
        return -1.0f;
    #endif
}
// ---------------------------------------------------------------------------
"""


def patch_text(text: str, anchor: str, block: str, sentinel: str, label: str) -> tuple[str, str]:
    """Insert block at anchor unless this step's sentinel is already present."""
    if sentinel in text:
        return text, f"already applied ({label})"
    idx = text.find(anchor)
    if idx < 0:
        return text, f"anchor missing for {label}"
    return text[:idx] + block + text[idx + len(anchor) :], ""


def patch_file(path: pathlib.Path, write: bool) -> int:
    text = path.read_text()
    steps = (
        (DECL_ANCHOR, DECL_BLOCK, "SOKOL_GFX_API_DECL float sg_agate_query_gpu_frame_ms(void);", "public declarations"),
        (GLOBALS_ANCHOR, GLOBALS_BLOCK, "static id<MTLCommandBuffer> _sg_agate_gpu_cb = nil;", "file-static state"),
        (COMMIT_ANCHOR, COMMIT_BLOCK, "The previous completed buffer is sampled first", "commit retain hook"),
        (IMPL_ANCHOR, IMPL_BLOCK, "SOKOL_API_IMPL float sg_agate_query_gpu_frame_ms(void) {", "public implementations"),
    )
    problems: list[str] = []
    applied = 0
    for anchor, block, sentinel, label in steps:
        text, note = patch_text(text, anchor, block, sentinel, label)
        if note.startswith("already applied"):
            print(f"  skip {label}: already applied")
        elif note:
            problems.append(f"{path}:{label}: {note}")
        else:
            applied += 1
            print(f"  patched {label}")
    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)
        return len(problems)
    if write and applied:
        path.write_text(text)
        print(f"\nwrote {path} ({applied} blocks inserted)")
    elif applied:
        print(f"\n{applied} blocks would be inserted (--check, no write)")
    else:
        print("nothing to patch (already applied?)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    target = root / DEFAULT_TARGET
    if not target.is_file():
        print(f"missing file: {target}", file=sys.stderr)
        return 2
    print(f"== {target.relative_to(root)}")
    return 1 if patch_file(target, write=not args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
