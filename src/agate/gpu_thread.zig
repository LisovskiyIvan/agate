//! Graphics-context thread marker.
//!
//! `sg.*` calls are only legal on the thread that owns the sokol_gfx
//! context — the sokol-app callback thread. Applications call
//! `markContextThread()` once from their init callback, before spawning the
//! simulation/game thread. Engine paths that may run on a non-context thread
//! (decal expiration inside `Scene.update`, input-driven mesh creation) use
//! `isOnContextThread()` to decide whether a GPU touch can happen inline or
//! must be deferred to the render phase.
//!
//! Unit tests and tools never mark a context thread: every thread then
//! counts as the context thread, which keeps the historical synchronous
//! behavior (GPU handles are inert `id == 0` there anyway).

const std = @import("std");
const builtin = @import("builtin");

/// Written once from the init callback before any other thread starts, then
/// only read. The thread spawn in the apps establishes the happens-before
/// edge for the game thread; a loaded value can therefore never observe a
/// half-written id.
var context_thread_id: ?std.Thread.Id = null;

/// Records the calling thread as the sg-context owner. Call once from the
/// sokol init callback, before spawning the game thread.
pub fn markContextThread() void {
    context_thread_id = std.Thread.getCurrentId();
}

/// True when called on the marked graphics thread, or when no thread has
/// been marked (unit tests, tools, single-threaded mode).
pub fn isOnContextThread() bool {
    const marked = context_thread_id orelse return true;
    return std.Thread.getCurrentId() == marked;
}

/// Assertion for render-phase entry points that must run on the sg-context
/// thread. Kept in Debug and ReleaseSafe so the tripwire also guards the
/// configuration the test suite runs in; compiled out of ReleaseFast/
/// ReleaseSmall where the panic machinery is unwanted.
pub fn assertOnContextThread() void {
    switch (builtin.mode) {
        .Debug, .ReleaseSafe => std.debug.assert(isOnContextThread()),
        .ReleaseFast, .ReleaseSmall => {},
    }
}
