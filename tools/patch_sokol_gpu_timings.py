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
   Completed. Non-Metal backends (GL/D3D11/dummy/...) always return -1
   in v1; per-pass timers are v2 (below).

   v2: per-pass GPU timers + GL timer-query. Metal keeps the v1
   frame-only timer: sokol encodes the whole frame into ONE
   MTLCommandBuffer (`_sg_mtl_begin_pass` creates it on the first pass,
   `_sg_mtl_commit` commits it), so `GPUStartTime/GPUEndTime` can only
   ever span the full frame. True per-encoder GPU times would need
   either one command buffer per pass (splits the submit: extra commits
   plus drawable/present and in-flight-semaphore surgery) or
   `MTLCounterSampleBuffer` timestamp sampling at encoder boundaries
   (device-capability-gated, resolve + barrier overhead). Both were
   rejected for v2: the frame timer above is honest and cheap, per-pass
   on Metal stays a documented follow-up. `sg_agate_gpu_pass_begin/end`
   are linked no-ops on Metal and `sg_agate_query_gpu_pass_ms` is -1.
   GL4.1 (SOKOL_GLCORE, non-Win32) gets REAL per-pass timers: the engine
   brackets each phase with `sg_agate_gpu_pass_begin/end(pass)` (pass
   ids 0=shadow, 1=main, 2=post), which drive a `GL_TIME_ELAPSED` query
   pool (depth 4 per pass, reaped without stalling via
   `GL_QUERY_RESULT_AVAILABLE`, last-completed semantics like v1).
   `sg_agate_query_gpu_frame_ms` on GL returns the sum of the
   last-completed per-pass values (serial GPU execution makes this a
   lower bound of the true frame span: inter-pass bubbles are excluded;
   -1 until the first sample). Win32-GL (the embedded loader has no
   timer entry points), GLES3 (`GL_TIME_ELAPSED_EXT` differs), and the
   remaining backends stay fail-closed at -1.

Disabled path: one predictable branch in `_sg_mtl_commit` (v1) and one
in `_sg_gl_commit` (v2); no retained objects, queries short-circuit
to -1.

Usage
-----
    python3 tools/patch_sokol_gpu_timings.py            # patch in place
    python3 tools/patch_sokol_gpu_timings.py --check    # report only

The script is idempotent: applied blocks are detected by their
`AGATE GPU TIMINGS` markers and skipped, so it can be re-run after
refreshing `vendor/sokol` from upstream. v2 steps apply on top of v1
(a fresh tree gets v1+v2 in a single run; an already-patched tree is
upgraded in place).
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


# --- v2: per-pass GPU timers + GL timer-query ------------------------------

# Public declarations for the engine-driven per-pass timers. Anchored on
# the v1 declaration line so fresh trees (v1 applied just above in the
# same run) and upgraded trees both match.
V2_DECL_ANCHOR = "SOKOL_GFX_API_DECL float sg_agate_query_gpu_frame_ms(void);\n"
V2_DECL_BLOCK = """SOKOL_GFX_API_DECL float sg_agate_query_gpu_frame_ms(void);
// AGATE GPU TIMINGS v2 (per-pass timers, see vendor/sokol/README.agate.md):
// engine-driven phase timers. Pass ids: 0=shadow, 1=main, 2=post.
// The engine brackets each phase with pass_begin/end; query returns the
// last COMPLETED sample in ms, or -1 when unavailable (disabled, not
// ready yet, Metal/unsupported backend, or an invalid pass id).
SOKOL_GFX_API_DECL void sg_agate_gpu_pass_begin(int pass);
SOKOL_GFX_API_DECL void sg_agate_gpu_pass_end(int pass);
SOKOL_GFX_API_DECL float sg_agate_query_gpu_pass_ms(int pass);
// File-static helpers owned by the v2 GL-state block below (real pool on
// GLCORE-non-Win32, no-op stubs elsewhere, bottom stubs when the GL
// region itself is compiled out). Forward-declared here with plain
// `static` (`_SOKOL_PRIVATE` is only defined further down) so the
// backend-independent `sg_agate_set_gpu_timing_enabled` below can mirror
// into the pool on every backend without an implicit declaration.
static void _sg_agate_gl_apply_enabled(bool enabled);
"""

# GL timer-query pool + the per-frame drain hook inside _sg_gl_commit.
# The block carries its own file-static state and is placed BEFORE
# _sg_gl_commit (forward use), so the public entry points implemented
# near sg_commit (v2 query/pass-impl blocks below) can rely on it.
# Non-GL-eligible builds compile the no-op stubs so every public entry
# point always links and stays fail-closed.
GLSTATE_ANCHOR = "_SOKOL_PRIVATE void _sg_gl_commit(void) {\n"
GLSTATE_BLOCK = """// AGATE GPU TIMINGS v2 (per-pass GL timer queries, see vendor/sokol/README.agate.md):
// engine-driven GL_TIME_ELAPSED pools, one per pass (0=shadow, 1=main,
// 2=post). Context thread only (phase brackets + commit drain + query
// poll all run there), so no atomics. Queries retire with a KEEP-frame
// lag and are reaped without stalling (GL_QUERY_RESULT_AVAILABLE);
// the served value is always the last COMPLETED sample, like v1.
#if defined(SOKOL_GLCORE) && !defined(_WIN32)
// <GL/gl.h> on Linux stops at GL 1.x: declare the timer-query entry
// points used below (exported by libGL; macOS <OpenGL/gl3.h> already
// declares them, so this is Linux/Unix-only).
#if (defined(__linux__) || defined(__unix__)) && !defined(__APPLE__)
extern void glGenQueries(GLsizei n, GLuint* ids);
extern void glDeleteQueries(GLsizei n, const GLuint* ids);
extern void glBeginQuery(GLenum target, GLuint id);
extern void glEndQuery(GLenum target);
extern void glGetQueryObjectuiv(GLuint id, GLenum pname, GLuint* params);
extern void glGetQueryObjectui64v(GLuint id, GLenum pname, uint64_t* params);
#endif
#ifndef GL_TIME_ELAPSED
#define GL_TIME_ELAPSED 0x88BF
#endif
#ifndef GL_QUERY_RESULT
#define GL_QUERY_RESULT 0x8866
#endif
#ifndef GL_QUERY_RESULT_AVAILABLE
#define GL_QUERY_RESULT_AVAILABLE 0x8867
#endif
#define _SG_AGATE_GPU_PASSES (3)
#define _SG_AGATE_GPU_QUERY_DEPTH (4)
static bool _sg_agate_gl_enabled = false;
static GLuint _sg_agate_gl_queries[_SG_AGATE_GPU_PASSES][_SG_AGATE_GPU_QUERY_DEPTH] = { { 0 } };
static uint8_t _sg_agate_gl_head[_SG_AGATE_GPU_PASSES] = { 0, 0, 0 };
static uint8_t _sg_agate_gl_tail[_SG_AGATE_GPU_PASSES] = { 0, 0, 0 };
static uint8_t _sg_agate_gl_pending[_SG_AGATE_GPU_PASSES] = { 0, 0, 0 };
static int _sg_agate_gl_active = -1;
static float _sg_agate_gl_last_ms[_SG_AGATE_GPU_PASSES] = { -1.0f, -1.0f, -1.0f };
// Reap retired queries for pass p, oldest first, never stalling: only
// AVAILABLE results are read (ns -> ms with a 10 s sanity clamp).
// Completion for a single target is in order, so the first not-ready
// query stops the drain.
_SOKOL_PRIVATE void _sg_agate_gl_reap(int p) {
    while (_sg_agate_gl_pending[p] > 0) {
        const GLuint q = _sg_agate_gl_queries[p][_sg_agate_gl_tail[p]];
        GLuint avail = 0;
        glGetQueryObjectuiv(q, GL_QUERY_RESULT_AVAILABLE, &avail);
        if (0 == avail) {
            break;
        }
        uint64_t ns = 0;
        glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
        const double ms = (double)ns / 1000000.0;
        if ((ms > 0.0) && (ms < 10000.0)) {
            _sg_agate_gl_last_ms[p] = (float)ms;
        }
        _sg_agate_gl_tail[p] = (uint8_t)((_sg_agate_gl_tail[p] + 1) % _SG_AGATE_GPU_QUERY_DEPTH);
        _sg_agate_gl_pending[p]--;
    }
}
_SOKOL_PRIVATE void _sg_agate_gl_drain_all(void) {
    int p;
    for (p = 0; p < _SG_AGATE_GPU_PASSES; p++) {
        _sg_agate_gl_reap(p);
    }
}
// Close any still-open query (normally a no-op: engine phases are
// strictly sequential, so an open query here means unbalanced calls).
// The closed query joins the retire queue; the slot advances.
_SOKOL_PRIVATE void _sg_agate_gl_close_active(void) {
    if (_sg_agate_gl_active >= 0) {
        const int ap = _sg_agate_gl_active;
        glEndQuery(GL_TIME_ELAPSED);
        _sg_agate_gl_active = -1;
        _sg_agate_gl_pending[ap]++;
        _sg_agate_gl_head[ap] = (uint8_t)((_sg_agate_gl_head[ap] + 1) % _SG_AGATE_GPU_QUERY_DEPTH);
    }
}
// (Re)configure the pool on enable/disable. Disable deletes live queries
// and resets the caches so a later enable starts clean.
_SOKOL_PRIVATE void _sg_agate_gl_apply_enabled(bool enabled) {
    _sg_agate_gl_enabled = enabled;
    if (!enabled) {
        int p;
        for (p = 0; p < _SG_AGATE_GPU_PASSES; p++) {
            GLuint ids[_SG_AGATE_GPU_QUERY_DEPTH];
            int n = 0;
            int i;
            for (i = 0; i < _SG_AGATE_GPU_QUERY_DEPTH; i++) {
                if (_sg_agate_gl_queries[p][i] != 0) {
                    ids[n++] = _sg_agate_gl_queries[p][i];
                    _sg_agate_gl_queries[p][i] = 0;
                }
            }
            if (n > 0) {
                glDeleteQueries((GLsizei)n, ids);
            }
            _sg_agate_gl_head[p] = 0;
            _sg_agate_gl_tail[p] = 0;
            _sg_agate_gl_pending[p] = 0;
            _sg_agate_gl_last_ms[p] = -1.0f;
        }
        _sg_agate_gl_active = -1;
    }
}
#else
// Non-GL-eligible builds (Metal uses the v1 frame timer; Win32-GL has no
// timer entry points in the embedded loader; GLES3/D3D11/WGPU/Vulkan/
// dummy have no support): stubs so the public entry points always link
// and every query stays fail-closed at -1.
_SOKOL_PRIVATE void _sg_agate_gl_apply_enabled(bool enabled) {
    (void)enabled;
}
_SOKOL_PRIVATE void _sg_agate_gl_drain_all(void) {
}
#endif
// ---------------------------------------------------------------------------
_SOKOL_PRIVATE void _sg_gl_commit(void) {
#if defined(SOKOL_GLCORE) && !defined(_WIN32)
    // AGATE GPU TIMINGS v2: reap retired timer queries once per frame.
    // Non-blocking, context thread only; one predictable branch while off.
    if (_sg_agate_gl_enabled) {
        _sg_agate_gl_drain_all();
    }
#endif
"""

# Mirror the public enablement intent into the GL pool. Anchored on the
# first two lines of the v1 set_enabled implementation (present in both
# fresh and upgraded trees once v1 steps ran). NOTE: the v1 flag itself
# is only defined inside the Metal-only region (the v1 globals `#else`
# sits nested under `#if defined(SOKOL_METAL)`), so the assignment is
# guarded back to Metal here; GL builds track `_sg_agate_gl_enabled`.
ENABLE_ANCHOR = """SOKOL_API_IMPL void sg_agate_set_gpu_timing_enabled(bool enabled) {
    _sg_agate_gpu_timing_enabled = enabled;
"""
ENABLE_BLOCK = """SOKOL_API_IMPL void sg_agate_set_gpu_timing_enabled(bool enabled) {
    #if defined(SOKOL_METAL)
        _sg_agate_gpu_timing_enabled = enabled;
    #endif
    // AGATE GPU TIMINGS v2: mirror into the GL timer pool (no-op stub
    // outside GLCORE-non-Win32 builds; disable also deletes live queries).
    _sg_agate_gl_apply_enabled(enabled);
"""

# Extend the v1 frame query with the GL sum-of-passes path. This step
# REPLACES the whole v1 function (anchor = exact v1 text), so the block
# repeats the Metal/else arms verbatim around the new elif arm.
QUERY_ANCHOR = """SOKOL_API_IMPL float sg_agate_query_gpu_frame_ms(void) {
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
"""
QUERY_BLOCK = """SOKOL_API_IMPL float sg_agate_query_gpu_frame_ms(void) {
    #if defined(SOKOL_METAL)
        if (!_sg_agate_gpu_timing_enabled || (nil == _sg_agate_gpu_cb)) {
            return -1.0f;
        }
        _sg_agate_gpu_sample(_sg_agate_gpu_cb);
        // Last-completed semantics: the value lags one frame behind the
        // CPU submit (async GPU execution). -1 until the first completion.
        return _sg_agate_gpu_last_ms;
    #elif defined(SOKOL_GLCORE) && !defined(_WIN32)
        // AGATE GPU TIMINGS v2: GL frame time is the sum of the
        // last-completed per-pass TIME_ELAPSED values (passes with no
        // sample yet contribute 0). GPU work is serial, so this is a
        // lower bound of the true frame span: inter-pass bubbles are
        // excluded. -1 until the first sample. Drains first so the sum
        // is as fresh as the last completed query.
        if (!_sg_agate_gl_enabled) {
            return -1.0f;
        }
        _sg_agate_gl_drain_all();
        {
            float sum_ms = 0.0f;
            bool any = false;
            int p;
            for (p = 0; p < _SG_AGATE_GPU_PASSES; p++) {
                if (_sg_agate_gl_last_ms[p] >= 0.0f) {
                    sum_ms += _sg_agate_gl_last_ms[p];
                    any = true;
                }
            }
            return any ? sum_ms : -1.0f;
        }
    #else
        return -1.0f;
    #endif
}
// AGATE GPU TIMINGS v2 (per-pass entry points, see vendor/sokol/README.agate.md) ---
"""

# Public per-pass entry points, appended after the extended frame query
# above (anchor = its trailing marker). GL drives the query pool; every
# other backend is a linked no-op / -1 (fail-closed).
PASSIMPL_ANCHOR = "// AGATE GPU TIMINGS v2 (per-pass entry points, see vendor/sokol/README.agate.md) ---\n"
PASSIMPL_BLOCK = """#if !defined(_SOKOL_ANY_GL)
// AGATE GPU TIMINGS v2: the GL-state block above lives inside the GL-only
// region, so on Metal/dummy builds (no `_SOKOL_ANY_GL`) the helpers are
// defined here instead. Same fail-closed stubs as the in-region `#else`
// arm (which serves Win32-GL/GLES builds where the region IS compiled).
_SOKOL_PRIVATE void _sg_agate_gl_apply_enabled(bool enabled) {
    (void)enabled;
}
_SOKOL_PRIVATE void _sg_agate_gl_drain_all(void) {
}
#endif
// AGATE GPU TIMINGS v2 (per-pass entry points, see vendor/sokol/README.agate.md) ---
SOKOL_API_IMPL void sg_agate_gpu_pass_begin(int pass) {
    #if defined(SOKOL_GLCORE) && !defined(_WIN32)
        GLuint q;
        if (!_sg_agate_gl_enabled || (pass < 0) || (pass >= _SG_AGATE_GPU_PASSES)) {
            return;
        }
        _sg_agate_gl_close_active();
        _sg_agate_gl_reap(pass);
        if (_sg_agate_gl_pending[pass] >= _SG_AGATE_GPU_QUERY_DEPTH) {
            // Ring full, oldest still in flight: drop this sample, never stall.
            return;
        }
        q = _sg_agate_gl_queries[pass][_sg_agate_gl_head[pass]];
        if (0 == q) {
            glGenQueries(1, &q);
            if (0 == q) {
                return;
            }
            _sg_agate_gl_queries[pass][_sg_agate_gl_head[pass]] = q;
        }
        glBeginQuery(GL_TIME_ELAPSED, q);
        _sg_agate_gl_active = pass;
    #else
        _SOKOL_UNUSED(pass);
    #endif
}
SOKOL_API_IMPL void sg_agate_gpu_pass_end(int pass) {
    #if defined(SOKOL_GLCORE) && !defined(_WIN32)
        if (!_sg_agate_gl_enabled || (pass < 0) || (pass >= _SG_AGATE_GPU_PASSES)) {
            return;
        }
        if (_sg_agate_gl_active == pass) {
            _sg_agate_gl_close_active();
        }
    #else
        _SOKOL_UNUSED(pass);
    #endif
}
SOKOL_API_IMPL float sg_agate_query_gpu_pass_ms(int pass) {
    #if defined(SOKOL_GLCORE) && !defined(_WIN32)
        if (!_sg_agate_gl_enabled || (pass < 0) || (pass >= _SG_AGATE_GPU_PASSES)) {
            return -1.0f;
        }
        _sg_agate_gl_reap(pass);
        // Last-completed semantics, like the v1 frame timer.
        return _sg_agate_gl_last_ms[pass];
    #else
        _SOKOL_UNUSED(pass);
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
        # v2 (apply on top of v1; same idempotency contract via new sentinels).
        (V2_DECL_ANCHOR, V2_DECL_BLOCK, "SOKOL_GFX_API_DECL float sg_agate_query_gpu_pass_ms(int pass);", "v2 per-pass declarations"),
        (GLSTATE_ANCHOR, GLSTATE_BLOCK, "_sg_agate_gl_queries[_SG_AGATE_GPU_PASSES][_SG_AGATE_GPU_QUERY_DEPTH]", "v2 GL timer pool + commit drain"),
        (ENABLE_ANCHOR, ENABLE_BLOCK, "_sg_agate_gl_apply_enabled(enabled);", "v2 enablement mirror into GL pool"),
        (QUERY_ANCHOR, QUERY_BLOCK, "AGATE GPU TIMINGS v2: GL frame time is the sum", "v2 GL-aware frame query"),
        (PASSIMPL_ANCHOR, PASSIMPL_BLOCK, "SOKOL_API_IMPL float sg_agate_query_gpu_pass_ms(int pass) {", "v2 per-pass implementations"),
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
