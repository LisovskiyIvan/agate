const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const gpu_thread = @import("gpu_thread.zig");

test "unmarked headless is a CPU-phase inline decision, not GPU authorization" {
    const saved = gpu_thread.context_thread_id;
    defer gpu_thread.context_thread_id = saved;
    gpu_thread.context_thread_id = null;

    const me = std.Thread.getCurrentId();
    try std.testing.expect(gpu_thread.matches(null, me, false));
    try std.testing.expect(!gpu_thread.matches(null, me, true));
    if (!sg.isvalid()) try std.testing.expect(gpu_thread.isOnContextThread());
}

test "marked owner passes, foreign fails even headless" {
    const saved = gpu_thread.context_thread_id;
    defer gpu_thread.context_thread_id = saved;

    gpu_thread.markContextThread();
    const owner = gpu_thread.context_thread_id.?;
    const me = std.Thread.getCurrentId();
    try std.testing.expect(gpu_thread.matches(owner, me, sg.isvalid()));
    try std.testing.expect(gpu_thread.isOnContextThread());
    const Probe = struct {
        fn run(out: *bool) void {
            out.* = gpu_thread.isOnContextThread();
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
    try std.testing.expect(gpu_thread.matches(me, me, false));
    try std.testing.expect(gpu_thread.matches(me, me, true));
    try std.testing.expect(!gpu_thread.matches(me, child, false));
    try std.testing.expect(!gpu_thread.matches(me, child, true));
    try std.testing.expect(gpu_thread.matches(child, child, true));
    try std.testing.expect(gpu_thread.matches(null, me, false));
    try std.testing.expect(!gpu_thread.matches(null, me, true));
}

// Sokol audit fix (1st): VALID-only buffer adoption. Uses the REAL
// `sg.ResourceState` enum values headless (no sg.* calls — the states are
// plain enum tags, so no context is needed): a REAL nonzero FAILED must be
// rejected, while only nonzero VALID adopts. The old `id != 0` check at
// every drain site adopted FAILED (pool exhaustion is id 0 only) and the
// next `writeBufferTransient` asserts VALID (sokol_gfx.h
// `_sg_validate_write_buffer_transient`) — Debug abort. All drain creation
// sites gate through `createdBufferIsValid` (this predicate over the live
// query); the live FAILED synthesis (allocBuffer + failBuffer) and the
// pending-retry/retention/deinit proofs run in the LIVE tests below and in
// the per-leaf *_tests files (they SkipZigTest headless — the sandbox has
// no Metal device, proven by the aborted headless `sg.setup` probe).
test "audit: shouldAdoptBuffer rejects every non-VALID state including REAL FAILED" {
    // The audit's exact failure: nonzero id in FAILED state.
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(7, .FAILED));
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(1, .FAILED));
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(std.math.maxInt(u32), .FAILED));
    // Every other non-VALID state rejects too (INITIAL/ALLOC/INVALID are
    // never adoptable either — the guard is VALID-only, not "not FAILED").
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(7, .INITIAL));
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(7, .ALLOC));
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(7, .UNSEALED));
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(7, .INVALID));
    // Zero id never adopts, even when the state tag reads VALID (a stale
    // tag must not resurrect a dead pool slot).
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(0, .VALID));
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(0, .FAILED));
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(0, .INVALID));
    // Only nonzero VALID adopts (the byte-equivalent valid path).
    try std.testing.expect(gpu_thread.shouldAdoptBuffer(1, .VALID));
    try std.testing.expect(gpu_thread.shouldAdoptBuffer(7, .VALID));
}

// Documents the old bug inline (no sg.*): the pre-fix `id != 0` adoption
// predicate accepts a REAL FAILED handle — this is what every drain site
// did before the fix. Fails if anyone reintroduces id-only adoption.
test "audit: id-only adoption would accept FAILED (the pre-fix bug)" {
    const oldWouldAdopt = struct {
        fn f(id: u32) bool {
            return id != 0;
        }
    }.f;
    try std.testing.expect(oldWouldAdopt(7));
    try std.testing.expect(!gpu_thread.shouldAdoptBuffer(7, .FAILED));
}

// LIVE proof (skips headless): synthesizes a REAL nonzero FAILED via
// allocBuffer + failBuffer, drives the REAL predicate over the REAL live
// query, then proves the failure lifetime (destroy → INVALID, i.e. the
// deinit-count discipline). Callable from the native GPU fixture (same path
// the `LIVE` test below runs); in the headless sandbox it compiles and the
// test skips.
pub fn liveAuditFailedLifecycle(_: std.mem.Allocator) !void {
    gpu_thread.markContextThread();
    defer gpu_thread.resetContextThreadForTest();
    const fb = sg.allocBuffer();
    try std.testing.expect(fb.id != 0);
    sg.failBuffer(fb);
    std.debug.print("audit lifecycle: FAILED id={d} st={s} adopt={}\n", .{
        fb.id, @tagName(sg.queryBufferState(fb)), gpu_thread.createdBufferIsValid(fb),
    });
    try std.testing.expectEqual(sg.ResourceState.FAILED, sg.queryBufferState(fb));
    try std.testing.expect(!gpu_thread.createdBufferIsValid(fb));
    sg.destroyBuffer(fb);
    std.debug.print("audit lifecycle: after destroy st={s}\n", .{@tagName(sg.queryBufferState(fb))});
    try std.testing.expectEqual(sg.ResourceState.INVALID, sg.queryBufferState(fb));
}

test "LIVE audit: REAL FAILED synthesizes, rejects, and destroys to INVALID" {
    if (!sg.isvalid()) return error.SkipZigTest;
    try liveAuditFailedLifecycle(std.testing.allocator);
}
