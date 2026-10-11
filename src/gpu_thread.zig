//! Graphics-context thread marker.
//!
//! `sg.*` calls are only legal on the thread that owns the sokol_gfx
//! context. Applications call `markContextThread()` once from their init
//! callback, before spawning the simulation/game thread. Engine paths that
//! may run on a non-context thread use `isOnContextThread()` to decide
//! whether a GPU touch can happen inline or must be deferred.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

/// Written once from the init callback before any other thread starts, then
/// only read. The thread spawn establishes the happens-before edge, so a
/// loaded value never observes a half-written id. `sg.isvalid()` is a plain
/// global SDK flag set by `sg.setup()` before the worker spawns, so reading
/// it from any thread here is safe.
pub var context_thread_id: ?std.Thread.Id = null;

/// Records the calling thread as the sg-context owner. Call once from the
/// sokol init callback, before spawning the game thread.
pub fn markContextThread() void {
    context_thread_id = std.Thread.getCurrentId();
}

/// Test helper to reset context thread state between test cases.
pub fn resetContextThreadForTest() void {
    context_thread_id = null;
}

/// Pure ownership policy, unit-testable without a live Metal context.
/// `live` stands in for `sg.isvalid()`.
pub fn matches(owner: ?std.Thread.Id, caller: std.Thread.Id, live: bool) bool {
    if (owner) |marked| return caller == marked;
    return !live;
}

/// True on the marked owner thread. Unmarked with no live context is true
/// as a CPU-phase decision (headless cleanup runs inline); this is not GPU
/// authorization. Unmarked with a live context is false on every thread.
pub fn isOnContextThread() bool {
    const caller = std.Thread.getCurrentId();
    if (context_thread_id) |marked| return caller == marked;
    return !sg.isvalid();
}

/// Missing/foreign live GPU ownership panics in all build modes, including
/// ReleaseFast/ReleaseSmall.
pub fn assertOnContextThread() void {
    if (!isOnContextThread()) @panic("gpu_thread: sg-context owner thread required");
}

/// VALID-only adoption predicate (Sokol audit fix, single source of truth):
/// a just-created handle is adoptable iff its id is nonzero AND its resource
/// state is VALID. Pool exhaustion is id 0; a nonzero FAILED id (creation
/// validation failure) must be destroyed inline and retried, never adopted —
/// every `writeBufferTransient` below asserts VALID first (sokol_gfx.h
/// `_sg_validate_write_buffer_transient`), so an adopted FAILED handle
/// aborts the next flush in Debug. Pure (no sg.*): headless-testable with
/// the real `sg.ResourceState` values, including a REAL nonzero FAILED.
/// All drain creation sites (particles system/slot, softbody, trail,
/// greased line, compute buffers/views/shader/pipeline) gate through
/// `createdBufferIsValid` below, which is this predicate over the live
/// query — one not-taken branch shape when unarmed, no behavior change on
/// the VALID path.
pub fn shouldAdoptBuffer(id: u32, state: sg.ResourceState) bool {
    return id != 0 and state == .VALID;
}

/// Live adoption check: the predicate above over the real resource state.
/// Context thread + live context only (queries assert without setup).
/// Zero ids short-circuit before the query.
pub fn createdBufferIsValid(buf: sg.Buffer) bool {
    if (buf.id == 0) return false;
    return sg.queryBufferState(buf) == .VALID;
}

/// Live view adoption check (same contract as buffers).
pub fn createdViewIsValid(view: sg.View) bool {
    if (view.id == 0) return false;
    return sg.queryViewState(view) == .VALID;
}

/// Live shader adoption check (same contract as buffers).
pub fn createdShaderIsValid(shd: sg.Shader) bool {
    if (shd.id == 0) return false;
    return sg.queryShaderState(shd) == .VALID;
}

/// Live pipeline adoption check (same contract as buffers).
pub fn createdPipelineIsValid(pip: sg.Pipeline) bool {
    if (pip.id == 0) return false;
    return sg.queryPipelineState(pip) == .VALID;
}
