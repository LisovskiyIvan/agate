const std = @import("std");
const sokol = @import("sokol");
const jobs = @import("../jobs.zig");
const stats_mod = @import("../scene/stats.zig");
const SceneStats = stats_mod.SceneStats;
const report_queue = @import("report_queue.zig");
const core = @import("core.zig");
const Profiler = core.Profiler;

/// Sleep helper for pacing-sensitive tests (Zig 0.16 has no Thread.sleep).
fn testSleepMs(ms: u64) void {
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    var rem: std.c.timespec = undefined;
    _ = std.c.nanosleep(&ts, &rem);
}

test "Profiler restart preserves monotonic app clock" {
    // Timer is app-owned, initialized once before threads (main.zig init).
    // Standalone unit context: single-threaded explicit setup.
    sokol.time.setup();
    // Let the anchor sit comfortably above call overhead so a clock origin
    // reset (old Profiler.start calling setup) shows up as a deterministic
    // backward step, not timer granularity noise.
    testSleepMs(5);
    const anchor = sokol.time.now();
    try std.testing.expect(anchor > 0);

    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    prof.start();
    const t1 = sokol.time.now();
    // Source clock must not reset on start: still at/after pre-start anchor.
    try std.testing.expect(t1 >= anchor);
    try std.testing.expect(prof.start_time_ticks >= anchor);

    testSleepMs(2);
    prof.stop();
    prof.start();
    const t2 = sokol.time.now();
    try std.testing.expect(t2 >= t1);
    try std.testing.expect(t2 >= anchor);
    try std.testing.expect(prof.start_time_ticks >= t1);
    prof.stop();
}

test "Profiler start/stop never moves app clock backward for concurrent readers" {
    sokol.time.setup();
    testSleepMs(5);

    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    const Ctx = struct {
        ready: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        backward: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
        fn run(c: *@This()) void {
            var last = sokol.time.now();
            c.ready.store(true, .release);
            while (!c.stop.load(.acquire)) {
                const now = sokol.time.now();
                if (now < last) _ = c.backward.fetchAdd(1, .monotonic);
                last = now;
                std.atomic.spinLoopHint();
            }
        }
    };
    var ctx = Ctx{};
    const reader = try std.Thread.spawn(.{}, Ctx.run, .{&ctx});
    // Wait until the reader holds a pre-restart anchor so every restart
    // below races a live read (otherwise a fast restart loop can finish
    // before the thread's first read and miss the window entirely).
    while (!ctx.ready.load(.acquire)) std.atomic.spinLoopHint();
    // Only the context thread touches the Profiler; the worker only reads
    // the shared app clock, mirroring gameLoop's time.now use.
    var i: usize = 0;
    while (i < 200) : (i += 1) {
        prof.start();
        var k: usize = 0;
        while (k < 200) : (k += 1) std.atomic.spinLoopHint();
        prof.stop();
    }
    ctx.stop.store(true, .release);
    reader.join();
    try std.testing.expectEqual(@as(u32, 0), ctx.backward.load(.acquire));
}

test "profiler async enqueue matches sync output; window hold excludes file IO" {
    // Equivalence: the async path (capture + encode + enqueue to io_runner)
    // must produce byte-identical files to the synchronous saveReports.
    // Measurement (smoke-grade, machine noise applies): the old window held
    // the mutex across capture + encode + FILE IO; the new window holds it
    // across capture only (encode is context-owned, IO runs on io_runner).
    // Both timings print below; the reported gate numbers come from here.
    const Mesh = @import("../mesh.zig").Mesh;
    sokol.time.setup();
    const ally = std.testing.allocator;
    const testScene = @import("../testing.zig").testScene;
    var scene = testScene(ally);
    defer {
        @import("../scene/content.zig").deinitMeshes(ally, &scene.meshes);
        scene.profiler.deinit();
    }

    const m = try ally.create(Mesh);
    m.* = @import("../testing.zig").testMesh("BenchCube");
    m.vertex_count = 24;
    m.index_count = 36;
    m.index_type = .UINT16;
    try scene.meshes.append(ally, m);

    scene.profiler.start();
    var stats = SceneStats{};
    stats.draw_calls = 128;
    stats.triangles = 4096;
    var f: u64 = 0;
    while (f < 300) : (f += 1) {
        stats.update_ms = 1.0;
        stats.prepare_ms = 0.5;
        stats.main_ms = 2.0;
        scene.profiler.recordFrame(f, &stats);
    }
    scene.profiler.stop();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const sync_base = try std.fmt.allocPrint(ally, ".zig-cache/tmp/{s}/rep_sync", .{tmp.sub_path});
    defer ally.free(sync_base);
    const async_base = try std.fmt.allocPrint(ally, ".zig-cache/tmp/{s}/rep_async", .{tmp.sub_path});
    defer ally.free(async_base);

    // BEFORE shape: capture + encode + file IO, all inline (what the old
    // bounded window held the mutex across).
    const t_sync0 = jobs.monoNs();
    try scene.profiler.saveReports(&scene, sync_base);
    const sync_ns = jobs.monoNs() -% t_sync0;

    // AFTER shape: capture (the only part left under the window) ...
    const t_cap0 = jobs.monoNs();
    _ = try scene.profiler.captureMemorySnapshot(&scene);
    const cap_ns = jobs.monoNs() -% t_cap0;
    // ... then encode + enqueue with NO lock held and NO file IO inline.
    const runner = try jobs.TaskRunner.init(ally, 1);
    defer runner.deinit();
    var out: [3]?*report_queue.ReportWriteTask = .{ null, null, null };
    const t_enq0 = jobs.monoNs();
    try scene.profiler.enqueueReportWrites(runner, async_base, .all, &out);
    const enq_ns = jobs.monoNs() -% t_enq0;
    // Caller-side poll + deinit (the showcase/cleanup pattern).
    const deadline = jobs.monoNs() + 10_000_000_000;
    var done = false;
    while (!done) {
        if (jobs.monoNs() >= deadline) return error.TestUnexpectedResult;
        done = true;
        for (out) |slot| {
            if (slot) |task| {
                if (!task.isDone()) done = false;
            }
        }
        if (!done) jobs.sleepNs(500_000);
    }
    for (out, 0..) |slot, i| {
        if (slot) |task| {
            try std.testing.expect(task.isSuccess());
            out[i] = null;
            task.deinit();
        }
    }

    // Byte-identical files across the two paths.
    const io = std.Io.Threaded.global_single_threaded.io();
    const exts = [_][]const u8{ ".html", ".md", ".json" };
    for (exts) |ext| {
        const ps = try std.fmt.allocPrint(ally, "{s}{s}", .{ sync_base, ext });
        defer ally.free(ps);
        const pa = try std.fmt.allocPrint(ally, "{s}{s}", .{ async_base, ext });
        defer ally.free(pa);
        const bs = try std.Io.Dir.cwd().readFileAlloc(io, ps, ally, .unlimited);
        defer ally.free(bs);
        const ba = try std.Io.Dir.cwd().readFileAlloc(io, pa, ally, .unlimited);
        defer ally.free(ba);
        try std.testing.expectEqualSlices(u8, bs, ba);
    }

    const to_ms = struct {
        fn ms(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        }
    }.ms;
    std.debug.print(
        "[profile-window] sync capture+encode+IO: {d:.2}ms | capture-only (new window hold): {d:.2}ms | encode+enqueue off-lock: {d:.2}ms (smoke-grade, includes tmpfs IO)\n",
        .{ to_ms(sync_ns), to_ms(cap_ns), to_ms(enq_ns) },
    );
}
