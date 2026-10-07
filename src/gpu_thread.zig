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
var context_thread_id: ?std.Thread.Id = null;

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
fn matches(owner: ?std.Thread.Id, caller: std.Thread.Id, live: bool) bool {
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

test "unmarked headless is a CPU-phase inline decision, not GPU authorization" {
    const saved = context_thread_id;
    defer context_thread_id = saved;
    context_thread_id = null;

    const me = std.Thread.getCurrentId();
    try std.testing.expect(matches(null, me, false));
    try std.testing.expect(!matches(null, me, true));
    if (!sg.isvalid()) try std.testing.expect(isOnContextThread());
}

test "marked owner passes, foreign fails even headless" {
    const saved = context_thread_id;
    defer context_thread_id = saved;

    markContextThread();
    const owner = context_thread_id.?;
    const me = std.Thread.getCurrentId();
    try std.testing.expect(matches(owner, me, sg.isvalid()));
    try std.testing.expect(isOnContextThread());
    const Probe = struct {
        fn run(out: *bool) void {
            out.* = isOnContextThread();
        }
    };
    var foreign: bool = true;
    const t = try std.Thread.spawn(.{}, Probe.run, .{&foreign});
    t.join();
    try std.testing.expect(!foreign);
}

test "pure policy covers registered, unregistered, and foreign live" {
    const me = std.Thread.getCurrentId();
    const Probe = struct {
        fn run(out: *std.Thread.Id) void {
            out.* = std.Thread.getCurrentId();
        }
    };
    var child: std.Thread.Id = me;
    const t = try std.Thread.spawn(.{}, Probe.run, .{&child});
    t.join();
    try std.testing.expect(matches(me, me, false));
    try std.testing.expect(matches(me, me, true));
    try std.testing.expect(!matches(me, child, false));
    try std.testing.expect(!matches(me, child, true));
    try std.testing.expect(matches(child, child, true));
    try std.testing.expect(matches(null, me, false));
    try std.testing.expect(!matches(null, me, true));
}
