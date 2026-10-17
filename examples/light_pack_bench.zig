//! Headless ReleaseFast benchmark for the per-frame light pack and the
//! clustered-tile CPU build (no GPU context, no window).
//!
//! Why this exists: the audit found ~4 KiB of retired per-pool clustered
//! arrays in `LightRig.FramePack` (written every pack, copied through the
//! frame handoff, read only by tests). This bench prints `@sizeOf(FramePack)`
//! and times the real `LightRig.packFrame` for representative scenes, plus
//! `ClusteredGpuCache.rebuildCpuFromLights` at 1080p/4K. The same source is
//! built against the pre-fix tree in a worktree for an A/B comparison, so it
//! uses only fields/APIs that exist on both sides (`clustered_lights`,
//! `clustered_count`, `packFrame`, `rebuildCpuFromLights`).
//!
//! Scope: CPU only. `scene.lights` is a real `LightRig` from `testScene`;
//! no `sg.*` call is reachable from pack/tile paths. GPU upload/draw cost is
//! intentionally out of scope (see `gpu_phase_bench` / `gpu-timing`).
//!
//! Run (from examples/):
//!   zig build run-light-pack -Doptimize=ReleaseFast -- --reps 7
//!   ./zig-out/bin/light-pack-bench --pack-iters 500000 --tile-iters 400

const std = @import("std");
const z = @import("agate");
const sokol = @import("sokol");

const Vec3 = z.Vec3;
const Mat4 = z.Mat4;
const alloc_global = std.heap.smp_allocator;

const Cfg = struct {
    reps: usize = 5,
    warmup: usize = 1,
    pack_iters: usize = 200_000,
    tile_iters_1080: usize = 400,
    tile_iters_4k: usize = 160,
};

const Result = struct {
    checksum: u64,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    total_ms: f64,
};

fn lessF64(_: void, a: f64, b: f64) bool {
    return a < b;
}

fn median(times: []f64) f64 {
    std.mem.sort(f64, times, {}, lessF64);
    return times[times.len / 2];
}

fn loop(
    comptime C: type,
    ctx: *C,
    comptime run: fn (*C, usize) u64,
    iters: usize,
    cfg: Cfg,
) !Result {
    var checksum: u64 = 0;
    for (0..cfg.warmup) |_| {
        var i: usize = 0;
        while (i < iters) : (i += 1) checksum +%= run(ctx, i);
    }
    const times = try alloc_global.alloc(f64, cfg.reps);
    defer alloc_global.free(times);
    var total: f64 = 0;
    for (0..cfg.reps) |r| {
        const t0 = sokol.time.now();
        var i: usize = 0;
        while (i < iters) : (i += 1) checksum +%= run(ctx, i);
        const ms: f64 = @floatCast(sokol.time.ms(sokol.time.now() -% t0));
        times[r] = ms;
        total += ms;
    }
    var mn = times[0];
    var mx = times[0];
    for (times) |t| {
        mn = @min(mn, t);
        mx = @max(mx, t);
    }
    return .{
        .checksum = checksum,
        .median_ms = median(times),
        .min_ms = mn,
        .max_ms = mx,
        .total_ms = total,
    };
}

fn report(name: []const u8, r: Result, iters: usize, extra: []const u8) void {
    const ns_per_op = r.median_ms * 1e6 / @as(f64, @floatFromInt(iters));
    std.debug.print(
        "case={s}{s} iters={d} checksum={x} median={d:.3}ms min={d:.3}ms max={d:.3}ms total={d:.1}ms ns_per_op={d:.1}\n",
        .{ name, extra, iters, r.checksum, r.median_ms, r.min_ms, r.max_ms, r.total_ms, ns_per_op },
    );
    if (r.total_ms < 100.0) {
        std.debug.print("case={s}{s} INCONCLUSIVE: total {d:.1}ms < 100ms; raise iters/reps\n", .{ name, extra, r.total_ms });
        std.process.exit(2);
    }
}

fn makeScene() z.Scene {
    return z.testing.testScene(alloc_global);
}

fn deinitScene(scene: *z.Scene) void {
    scene.lights.deinit(alloc_global);
    // Headless: no context was marked and sg is invalid, so the cache's
    // teardown only frees CPU scratch (all GPU handles are zero).
    scene.clustered.deinit(alloc_global);
    scene.profiler.deinit();
}

fn addPoints(scene: *z.Scene, count: usize, shadows: bool) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const f: f32 = @floatFromInt(i);
        _ = try scene.lights.createPointLight(alloc_global, "bench_point", .{
            .position = Vec3.new(f * 1.5 - 3.0, 2.0, @mod(f, 3.0) * 2.0 - 2.0),
            .range = 12.0,
            .intensity = 2.0 + f * 0.1,
            .shadow = .{ .cast_shadows = shadows and (i % 2 == 0) },
        });
    }
}

fn addSpots(scene: *z.Scene, count: usize, shadows: bool) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const f: f32 = @floatFromInt(i);
        _ = try scene.lights.createSpotLight(alloc_global, "bench_spot", .{
            .position = Vec3.new(-4.0 + f * 4.0, 6.0, 2.0),
            .direction = Vec3.new(0.0, -1.0, -0.2).normalize(),
            .range = 20.0,
            .intensity = 5.0,
            .inner_angle_deg = 18.0,
            .outer_angle_deg = 34.0,
            .shadow = .{ .cast_shadows = shadows and i == 0 },
        });
    }
}

fn addClustered(scene: *z.Scene, count: usize) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const f: f32 = @floatFromInt(i);
        _ = try scene.lights.addClusteredPointLight(
            Vec3.new(@mod(f, 8.0) - 4.0, 1.5 + @mod(f, 2.0), @floor(f / 8.0) - 3.0),
            .{
                .color = z.Color3.new(0.4 + @mod(f, 2.0) * 0.4, 0.6, 1.0 - @mod(f, 2.0) * 0.4),
                .intensity = 1.5,
                .radius = 6.0,
            },
        );
    }
}

/// Deterministic orbit eye positions; the pack sees a moving camera so the
/// compiler cannot hoist one call out of the loop.
fn makeEyes() [64]Vec3 {
    var out: [64]Vec3 = undefined;
    for (&out, 0..) |*e, i| {
        const f: f32 = @floatFromInt(i);
        const a = f * 0.0981747703; // 2*pi/64
        e.* = Vec3.new(@cos(a) * 7.0, 2.5 + @sin(a * 0.5), @sin(a) * 7.0);
    }
    return out;
}

const PackCtx = struct {
    scene: *z.Scene,
    shadows: bool,
    eyes: *const [64]Vec3,

    fn call(self: *PackCtx, i: usize) u64 {
        const eye = self.eyes[i & 63];
        var p = self.scene.lights.packFrame(eye, self.shadows, 1.0 / 60.0);
        std.mem.doNotOptimizeAway(&p);
        return @as(u64, @as(u32, @bitCast(p.counts[1]))) +% @as(u64, p.clustered_count);
    }
};

const TileCtx = struct {
    scene: *z.Scene,
    view_proj: Mat4,
    w: i32,
    h: i32,

    fn call(self: *TileCtx, _: usize) u64 {
        const lp = &self.scene.light_pack;
        self.scene.clustered.rebuildCpuFromLights(
            alloc_global,
            lp.clustered_lights[0..lp.clustered_count],
            self.view_proj,
            self.w,
            self.h,
            .{ .x = 0, .y = 0, .w = self.w, .h = self.h },
            0,
        ) catch return 0;
        std.mem.doNotOptimizeAway(self.scene.clustered.cpu_indices.items.ptr);
        return @as(u64, @intCast(self.scene.clustered.cpu_indices.items.len)) +%
            @as(u64, self.scene.clustered.tiles_x) * 1000 +%
            @as(u64, self.scene.clustered.tiles_y);
    }
};

fn packCase(name: []const u8, configure: anytype, shadows: bool, cfg: Cfg, eyes: *const [64]Vec3, frame_pack_bytes: usize) !void {
    var scene = makeScene();
    defer deinitScene(&scene);
    try configure(&scene);
    var ctx = PackCtx{ .scene = &scene, .shadows = shadows, .eyes = eyes };
    const r = try loop(PackCtx, &ctx, PackCtx.call, cfg.pack_iters, cfg);
    var extra_buf: [96]u8 = undefined;
    const extra = try std.fmt.bufPrint(&extra_buf, " frame_pack={d}B", .{frame_pack_bytes});
    report(name, r, cfg.pack_iters, extra);
}

fn noLights(_: *z.Scene) !void {}
fn fourPoints(s: *z.Scene) !void {
    try addPoints(s, 4, false);
}
fn fourPointsTwoSpots(s: *z.Scene) !void {
    try addPoints(s, 4, false);
    try addSpots(s, 2, false);
}
fn sixtyFourClustered(s: *z.Scene) !void {
    try addClustered(s, 64);
}
fn allLightsShadows(s: *z.Scene) !void {
    try addPoints(s, 4, true);
    try addSpots(s, 2, true);
    try addClustered(s, 64);
}

fn tileCase(name: []const u8, w: i32, h: i32, iters: usize, cfg: Cfg, frame_pack_bytes: usize) !void {
    var scene = makeScene();
    defer deinitScene(&scene);
    try addClustered(&scene, 64);
    // Pack once (the production render path consumes a staged pack) and keep
    // it in the scene field the tile context reads.
    scene.light_pack = scene.lights.packFrame(Vec3.new(1.0, 2.0, 3.0), true, 1.0 / 60.0);
    var ctx = TileCtx{
        .scene = &scene,
        .view_proj = Mat4.perspective(60.0, @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h)), 0.1, 250.0),
        .w = w,
        .h = h,
    };
    const r = try loop(TileCtx, &ctx, TileCtx.call, iters, cfg);
    var extra_buf: [128]u8 = undefined;
    const extra = try std.fmt.bufPrint(&extra_buf, " target={d}x{d} frame_pack={d}B", .{ w, h, frame_pack_bytes });
    report(name, r, iters, extra);
}

fn parseArgs(args: []const []const u8, cfg: *Cfg) void {
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (i + 1 < args.len) {
            const v = std.fmt.parseInt(usize, args[i + 1], 10) catch continue;
            if (std.mem.eql(u8, a, "--reps")) {
                cfg.reps = std.math.clamp(v, 3, 15);
                i += 1;
            } else if (std.mem.eql(u8, a, "--pack-iters")) {
                cfg.pack_iters = std.math.clamp(v, 1_000, 20_000_000);
                i += 1;
            } else if (std.mem.eql(u8, a, "--tile-iters")) {
                cfg.tile_iters_1080 = std.math.clamp(v, 10, 100_000);
                cfg.tile_iters_4k = std.math.clamp(v / 2, 5, 50_000);
                i += 1;
            } else if (std.mem.eql(u8, a, "--warmup")) {
                cfg.warmup = std.math.clamp(v, 0, 5);
                i += 1;
            }
        }
    }
}

pub fn main(init: std.process.Init) !void {
    sokol.time.setup();
    var cfg = Cfg{};
    var args_buf: [32][]const u8 = undefined;
    var n_args: usize = 0;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc_global);
    defer it.deinit();
    _ = it.next();
    while (it.next()) |a| {
        if (n_args < args_buf.len) {
            args_buf[n_args] = a;
            n_args += 1;
        }
    }
    parseArgs(args_buf[0..n_args], &cfg);

    const eyes = makeEyes();
    std.debug.print(
        "light_pack_bench: reps={d} warmup={d} pack_iters={d} tile_iters_1080={d} tile_iters_4k={d} allocator=smp clock=sokol.time units=ms\n",
        .{ cfg.reps, cfg.warmup, cfg.pack_iters, cfg.tile_iters_1080, cfg.tile_iters_4k },
    );

    // FramePack size is the structural half of the dead-field A/B: the
    // retired per-pool arrays accounted for ~4 KiB.
    var size_scene = makeScene();
    const size_pack = size_scene.lights.packFrame(Vec3.zero, true, 1.0 / 60.0);
    const frame_pack_bytes = @sizeOf(@TypeOf(size_pack));
    std.debug.print("light_pack_bench: frame_pack_bytes={d} cluster_lights_bytes={d}\n", .{
        frame_pack_bytes,
        @sizeOf(@TypeOf(size_pack.clustered_lights)),
    });
    deinitScene(&size_scene);

    try packCase("pack_empty", noLights, false, cfg, &eyes, frame_pack_bytes);
    try packCase("pack_4p", fourPoints, false, cfg, &eyes, frame_pack_bytes);
    try packCase("pack_4p2s", fourPointsTwoSpots, false, cfg, &eyes, frame_pack_bytes);
    try packCase("pack_64cl", sixtyFourClustered, false, cfg, &eyes, frame_pack_bytes);
    try packCase("pack_all_shadows", allLightsShadows, true, cfg, &eyes, frame_pack_bytes);

    try tileCase("tile_1080p_64", 1920, 1080, cfg.tile_iters_1080, cfg, frame_pack_bytes);
    try tileCase("tile_2160p_64", 3840, 2160, cfg.tile_iters_4k, cfg, frame_pack_bytes);

    std.debug.print("light_pack_bench: done\n", .{});
}
