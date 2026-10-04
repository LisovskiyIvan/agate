const std = @import("std");
const Scene = @import("../scene.zig").Scene;
const Vec3 = @import("math").Vec3;

/// Test-only tracking allocator: wraps a backing allocator, records every
/// live allocation by pointer, and detects cross-domain frees (a free of a
/// pointer this domain never allocated) plus leaks (nonempty live set at
/// test end). All trackers in one test share the same backing, so a foreign
/// free can still be forwarded to its true owner after being recorded.
/// Zero-length frees of unknown pointers are ignored (never foreign):
/// deinits of never-grown lists free an empty/undefined slice, which owns
/// no bytes by definition.
const TrackDomain = struct {
    backing: std.mem.Allocator,
    live: std.AutoHashMap(usize, usize),
    allocated_bytes: usize = 0,
    freed_bytes: usize = 0,
    foreign_frees: usize = 0,
    map_drops: usize = 0,

    fn init(backing: std.mem.Allocator) TrackDomain {
        return .{ .backing = backing, .live = std.AutoHashMap(usize, usize).init(backing) };
    }

    fn deinit(self: *TrackDomain) void {
        self.live.deinit();
    }

    fn allocator(self: *TrackDomain) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn liveCount(self: *const TrackDomain) usize {
        return self.live.count();
    }

    fn trackAlloc(self: *TrackDomain, ptr: [*]u8, len: usize) void {
        self.live.put(@intFromPtr(ptr), len) catch {
            self.map_drops += 1;
            return;
        };
        self.allocated_bytes += len;
    }

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *TrackDomain = @ptrCast(@alignCast(ctx));
        const p = self.backing.rawAlloc(len, alignment, ra) orelse return null;
        self.trackAlloc(p, len);
        return p;
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *TrackDomain = @ptrCast(@alignCast(ctx));
        if (!self.backing.rawResize(memory, alignment, new_len, ra)) return false;
        if (self.live.getPtr(@intFromPtr(memory.ptr))) |slot| {
            if (new_len > slot.*) self.allocated_bytes += new_len - slot.* else self.freed_bytes += slot.* - new_len;
            slot.* = new_len;
        } else if (memory.len > 0) {
            self.foreign_frees += 1;
        }
        return true;
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *TrackDomain = @ptrCast(@alignCast(ctx));
        const res = self.backing.rawRemap(memory, alignment, new_len, ra) orelse return null;
        if (self.live.fetchRemove(@intFromPtr(memory.ptr))) |kv| {
            if (new_len > kv.value) self.allocated_bytes += new_len - kv.value else self.freed_bytes += kv.value - new_len;
            self.live.put(@intFromPtr(res), new_len) catch {
                self.map_drops += 1;
                return res;
            };
        } else if (memory.len > 0) {
            self.foreign_frees += 1;
        }
        return res;
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *TrackDomain = @ptrCast(@alignCast(ctx));
        if (self.live.fetchRemove(@intFromPtr(memory.ptr))) |kv| {
            self.freed_bytes += kv.value;
        } else if (memory.len > 0) {
            self.foreign_frees += 1;
        }
        self.backing.rawFree(memory, alignment, ra);
    }
};

fn expectSameAllocator(a: std.mem.Allocator, b: std.mem.Allocator) !void {
    try std.testing.expect(a.ptr == b.ptr);
    try std.testing.expect(a.vtable == b.vtable);
}

test "allocator domains: nulls resolve to core, explicit domains stick" {
    const t = std.testing;
    // initIntoWithAllocators starts with initAllocatorsInto, and init/initInto
    // delegate to it — so this resolution IS the default-equality rule all
    // init paths share (the full init adds GPU objects, untestable headless).
    var scene: Scene = undefined;
    scene.initAllocatorsInto(.{ .core = t.allocator });
    try expectSameAllocator(t.allocator, scene.allocator);
    try expectSameAllocator(t.allocator, scene.render_allocator);
    try expectSameAllocator(t.allocator, scene.sim_allocator);
    try expectSameAllocator(t.allocator, scene.io_allocator);

    var render_mem = TrackDomain.init(t.allocator);
    defer render_mem.deinit();
    var sim_mem = TrackDomain.init(t.allocator);
    defer sim_mem.deinit();
    var io_mem = TrackDomain.init(t.allocator);
    defer io_mem.deinit();
    scene.initAllocatorsInto(.{
        .core = t.allocator,
        .render = render_mem.allocator(),
        .sim = sim_mem.allocator(),
        .io = io_mem.allocator(),
    });
    try expectSameAllocator(t.allocator, scene.allocator);
    try expectSameAllocator(render_mem.allocator(), scene.render_allocator);
    try expectSameAllocator(sim_mem.allocator(), scene.sim_allocator);
    try expectSameAllocator(io_mem.allocator(), scene.io_allocator);
}

test "allocator domains: io funds UploadQueue and io_runner, frees clean" {
    const t = std.testing;
    const assets_mod = @import("../assets.zig");
    const jobs_mod = @import("../jobs.zig");
    var io_mem = TrackDomain.init(t.allocator);
    defer io_mem.deinit();
    const io = io_mem.allocator();

    // Exact expressions from Scene.initIntoWithAllocators (core.zig).
    var uploads = try assets_mod.UploadQueue.init(io, 2);
    var runner = try jobs_mod.TaskRunner.init(io, 1);

    // Worker-side proof: a task allocating + freeing through io on the
    // runner thread (posted tasks own their context and free it in run).
    const IoProbe = struct {
        alloc: std.mem.Allocator,
        done: *std.atomic.Value(bool),
        fn run(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const probe = self.alloc.alloc(u8, 64) catch unreachable;
            @memset(probe, 0xA5);
            self.alloc.free(probe);
            self.done.store(true, .release);
            self.alloc.destroy(self);
        }
    };
    var done = std.atomic.Value(bool).init(false);
    const ctx = try io.create(IoProbe);
    ctx.* = .{ .alloc = io, .done = &done };
    runner.post(ctx, IoProbe.run);
    var waited_ns: u64 = 0;
    while (!done.load(.acquire)) {
        try t.expect(waited_ns < 5_000_000_000);
        jobs_mod.sleepNs(1_000_000);
        waited_ns += 1_000_000;
    }

    // Exact teardown order of Scene.deinit (uploads first, then io_runner):
    // both free through the stored io allocator.
    uploads.deinit();
    runner.deinit();
    try t.expectEqual(@as(usize, 0), io_mem.liveCount());
    try t.expectEqual(@as(usize, 0), io_mem.foreign_frees);
    try t.expectEqual(@as(usize, 0), io_mem.map_drops);
    try t.expect(io_mem.allocated_bytes > 0);
}

test "allocator domains: sim funds particles and nav, frees clean" {
    const t = std.testing;
    var sim_mem = TrackDomain.init(t.allocator);
    defer sim_mem.deinit();

    const SimWork = struct {
        core: std.mem.Allocator,
        sim: std.mem.Allocator,
        err: ?anyerror = null,
        fn run(self: *@This()) void {
            self.work() catch |e| {
                self.err = e;
            };
        }
        fn work(self: *@This()) !void {
            // Spawned thread = non-context thread, so ParticleSystem.init
            // defers GPU buffers exactly like a game-thread spawn; no sg.*
            // fires anywhere below (create/update/capture are CPU-only,
            // deinit skips zero-id buffers).
            var scene = @import("../testing.zig").testScene(self.core);
            defer scene.lights.deinit(self.core);
            defer scene.cameras.deinit(self.core);
            scene.sim_allocator = self.sim;
            scene.render_allocator = self.core;
            scene.io_allocator = self.core;

            const nm = try scene.createNavMeshGrid(-4, 4, -4, 4, 0, 2, 2, &.{});
            const ag = try scene.createNavAgent(nm, Vec3.new(0, 0, 0));
            // Pathfinder runtime alloc (waypoints) through the stored sim
            // allocator; the open grid connects start to target.
            try t.expect(try ag.setDestination(Vec3.new(3, 0, 3)));
            scene.updateNavAgents(0.016);

            const ps = try scene.createParticleSystem("sim_probe", 8);
            ps.emitOne();
            try scene.updateParticles(0.016);
            // Mirrors the frame_prepare wiring (sim-owned retained frame).
            scene.particles.captureFrame(scene.sim_allocator);

            // Mirrors ParticleLayer.deinit's CPU frees with the lifecycle
            // wiring (same allocator expressions), minus pass.deinit():
            // the pass teardown issues unconditional sg.destroy* and is
            // context-only like the real Scene.deinit, which never runs
            // headless. ps.deinit itself is sg-free here: every handle is
            // zero (deferred creation, CPU mode), all destroys id-guarded.
            for (scene.particles.systems.items) |sys| {
                sys.deinit();
                scene.sim_allocator.destroy(sys);
            }
            scene.particles.systems.deinit(scene.sim_allocator);
            scene.particles.frame.deinit(scene.sim_allocator);
            scene.particles.build_frame.deinit(scene.sim_allocator);
            scene.nav.deinit(scene.sim_allocator);
        }
    };
    var work = SimWork{ .core = t.allocator, .sim = sim_mem.allocator() };
    const thread = try std.Thread.spawn(.{}, SimWork.run, .{&work});
    thread.join();
    if (work.err) |e| return e;
    try t.expectEqual(@as(usize, 0), sim_mem.liveCount());
    try t.expectEqual(@as(usize, 0), sim_mem.foreign_frees);
    try t.expectEqual(@as(usize, 0), sim_mem.map_drops);
    try t.expect(sim_mem.allocated_bytes > 0);
}

test "allocator domains: failing io degrades to null runners" {
    const t = std.testing;
    const assets_mod = @import("../assets.zig");
    const jobs_mod = @import("../jobs.zig");
    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    const f = failing.allocator();
    // Exact expressions from Scene.initIntoWithAllocators (core.zig): both
    // degrade to null and every load takes its synchronous path.
    const uploads = assets_mod.UploadQueue.init(f, 2) catch null;
    try t.expect(uploads == null);
    const io_runner = jobs_mod.TaskRunner.init(f, 1) catch null;
    try t.expect(io_runner == null);
    // Scene.deinit guards both with `if (...)`: the null branches free
    // nothing and deinit succeeds.
}
