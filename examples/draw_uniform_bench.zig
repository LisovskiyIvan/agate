//! Headless ReleaseFast benchmark for per-draw uniform CPU cost (Track D/P0).
//!
//! Measured subset (pure CPU builders only, no GPU context):
//! per draw — view-variant select (`draw_helpers.frameUniformsForState`),
//! material block build (`draw_uniforms.buildFrameMaterial` from
//! deterministically varied inputs), probe select
//! (`draw_probes.selectProbes` over `--probes` entries at a varied
//! position), plus a minimal DCE guard (`doNotOptimizeAway` on the full
//! outputs — a compiler barrier, not work). Explicitly OUT of scope:
//! `sg.Bindings`/pipeline/`VsParams`/skinned/coat/bindClustered work and
//! the `sg.applyUniforms` upload itself (upload skipping is proven
//! separately in `src/scene/view_cache_tests.zig`). Reported ns/draw is
//! this CPU-prep subset only, not a whole-draw cost.
//!
//! Cases: `cached_empty` (production shape — view variants prebuilt once
//! per view, as `view_render`/`probe_render` do), `fallback` (direct-draw
//! shape — 2096 B view block rebuilt per draw), `cached_probes=N`
//! (overlapping-probe scan stress; N=0 is the common shipped shape).
//! Parity (untimed): cached vs fallback view bytes identical for BOTH
//! receive policies; per-case checksum stable across an untimed pre/post
//! pass; probe control hashes the FULL selection struct.
//!
//! Production allocator note: the path allocates nothing (stack structs +
//! threadlocal fallback); `smp` is reported for protocol parity with
//! `scene_scaling_bench`. Any case totalling <100ms reports INCONCLUSIVE
//! and exits 2 (no claim on that case).
//!
//! Run: `zig build run-draw-uniform -Doptimize=ReleaseFast -- --draws 2000000 --reps 5 --probes 8`
//!   (from examples/; idle machine, medians of --reps).

const std = @import("std");
const z = @import("agate");
const sokol = @import("sokol");

const uniforms_mod = z.testing.draw_uniforms;
const draw_mod = z.testing.draw_helpers;
const probes_mod = z.testing.draw_probes;

const Vec3 = z.Vec3;
const Mat4 = z.Mat4;
const Color3 = z.Color3;

const Cfg = struct {
    draws: usize = 2_000_000,
    reps: usize = 5,
    warmup: usize = 1,
    probes: usize = 8,
};

fn parseArgs(args: []const []const u8) Cfg {
    var c = Cfg{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (i + 1 < args.len and (std.mem.eql(u8, a, "--draws") or std.mem.eql(u8, a, "--reps") or std.mem.eql(u8, a, "--warmup") or std.mem.eql(u8, a, "--probes"))) {
            const v = std.fmt.parseInt(usize, args[i + 1], 10) catch continue;
            if (std.mem.eql(u8, a, "--draws")) c.draws = v;
            if (std.mem.eql(u8, a, "--reps")) c.reps = v;
            if (std.mem.eql(u8, a, "--warmup")) c.warmup = v;
            if (std.mem.eql(u8, a, "--probes")) c.probes = v;
            i += 1;
        }
    }
    if (c.draws < 100_000) c.draws = 100_000;
    if (c.draws > 5_000_000) c.draws = 5_000_000;
    if (c.reps < 3) c.reps = 3;
    if (c.reps > 11) c.reps = 11;
    if (c.warmup > 5) c.warmup = 5;
    if (c.probes > 16) c.probes = 16;
    return c;
}

fn lessF64(_: void, a: f64, b: f64) bool {
    return a < b;
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, lessF64);
    return xs[xs.len / 2];
}

fn buildCtx() uniforms_mod.FrameContext {
    return .{
        .view_proj = Mat4.identity,
        .eye = Vec3.new(1.0, 2.0, 3.0),
        .sun_dir = Vec3.new(0.5, 1.0, 0.3),
        .sun_color = Color3.new(1.0, 0.9, 0.8),
        .sun_intensity = 2.0,
        .directional_dir = .{ .{ 0.5, 1.0, 0.3, 0.0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } },
        .directional_color_int = .{ .{ 1.0, 0.9, 0.8, 2.0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } },
        .cascades = [_]Mat4{Mat4.identity} ** 4,
        .light_counts = .{ 0.0, 0.0, 0.0, 0.0 },
        .point_pos_range = [_][4]f32{.{ 0, 0, 0, 0 }} ** 4,
        .point_color_int = [_][4]f32{.{ 0, 0, 0, 0 }} ** 4,
        .spot_pos_range = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .spot_dir_inner = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .spot_color_outer = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .spot_intensity = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .spot_view_proj = [_]Mat4{Mat4.identity} ** 2,
        .spot_shadow_params = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .point_view_proj = [_]Mat4{Mat4.identity} ** 12,
        .point_shadow_params = [_][4]f32{.{ 0, 0, 0, 0 }} ** 4,
        .area_center_int = .{ .{ 1.0, 2.0, 3.0, 3.0 }, .{ 0, 0, 0, 0 } },
        .area_right = .{ .{ 2.0, 0.0, 0.0, 0.0 }, .{ 0, 0, 0, 0 } },
        .area_up = .{ .{ 0.0, 0.5, 0.0, 0.0 }, .{ 0, 0, 0, 0 } },
        .area_color = .{ .{ 1.0, 0.5, 0.25, 0.0 }, .{ 0, 0, 0, 0 } },
        .clustered_params = .{ 10.0, 6.0, 3.0, 1.0 },
        .clustered_viewport = .{ 640.0, 384.0, 64.0, 0.0 },
    };
}

fn buildShadow(receive: bool) uniforms_mod.ShadowState {
    return .{
        .ground_color = Color3.new(0.2, 0.25, 0.3),
        .enable_shadows = true,
        .mesh_receive_shadows = receive,
        .bias = 0.0012,
        .intensity = 0.75,
        .normal_bias = 0.02,
        .softness = 1.5,
        .debug_cascades = false,
        .splits = .{ 10.0, 26.0, 65.0, 150.0 },
    };
}

const Varied = struct { b0: f32, rough: f32, cutoff: f32, ns: f32, px: f32, py: f32 };

// Deterministic per-draw material scalars (loop-variant so the builder
// cannot hoist; built inline at the call site like the draw path builds
// its literal from record fields — no extra staging copy).
fn variedScalars(i: usize) Varied {
    const t: f32 = @floatFromInt(i % 128);
    return .{
        .b0 = t * 0.007,
        .rough = 0.5 + t * 0.001,
        .cutoff = if ((i % 3) == 0) 0.5 else 0.0,
        .ns = 1.0 + t * 0.0005,
        .px = (t - 64.0) * 0.5,
        .py = @as(f32, @floatFromInt((i / 128) % 16)) * 0.25,
    };
}

fn buildVariedMaterial(v: Varied) uniforms_mod.FrameMaterialUniforms {
    return uniforms_mod.buildFrameMaterial(.{
        .base_color = .{ v.b0, 0.5, 0.25, 1 },
        .pbr_factors = .{ 0.0, v.rough, 1.0, 1.0 },
        .emissive_color = .{ 0, 0, 0, 1 },
        .alpha_cutoff = v.cutoff,
        .normal_scale = v.ns,
        .uv_matrices = [_][4]f32{.{ 1, 0, 0, 1 }} ** 5,
        .uv_offsets = [_][4]f32{.{ 0, 0, 0, 0 }} ** 5,
        .channel_selectors = .{ 0, 1, 2, 0 },
        .clearcoat_factors = .{ 0, 0, 0, 0 },
        .clearcoat_color = .{ 1, 1, 1, 1 },
        .sheen_factors = .{ 0, 0, 0, 0 },
        .sheen_color = .{ 1, 1, 1, 1 },
        .anisotropy_factors = .{ 0, 0, 0, 0 },
        .transmission_factors = .{ 0, 0, 0, 0 },
        .transmission_color = .{ 0, 0, 0, 0 },
        .sss_factors = .{ 0, 0, 0, 0 },
        .sss_color = .{ 0, 0, 0, 0 },
        .clearcoat_uv_matrix = .{ 1, 0, 0, 1 },
        .clearcoat_uv_offset = .{ 0, 0, 0, 0 },
        .sheen_uv_matrix = .{ 1, 0, 0, 1 },
        .sheen_uv_offset = .{ 0, 0, 0, 0 },
        .refraction_factors = .{ 0, 0, 0, 0 },
        .probe_params = .{ 0, 0, 0, 0 },
        .probe2_params = .{ 0, 0, 0, 0 },
        .probe_box = .{ .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } },
    });
}

const CaseResult = struct {
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    total_ms: f64,
    checksum: u64,
};

/// Untimed full-coverage checksum over the exact timed body (view pointee
/// bytes + full material bytes + FULL selection struct). Runs pre/post;
/// equality proves determinism and output stability across reps.
fn checksumCase(
    ctx: *const uniforms_mod.FrameContext,
    shadow: uniforms_mod.ShadowState,
    probe_list: []const probes_mod.ProbeFrameEntry,
    fallback: bool,
    draws: usize,
) u64 {
    var ctx_local = ctx.*;
    if (fallback) {
        ctx_local.uniforms_with_shadows = null;
        ctx_local.uniforms_without_shadows = null;
    }
    var h = std.hash.Wyhash.init(0x9e3779b9);
    for (0..draws) |i| {
        const recv = (i & 1) == 0;
        const s = variedScalars(i);
        const v = draw_mod.frameUniformsForState(shadow, recv, &ctx_local, Mat4.identity, .{ 0, 1, 1, 0 });
        const m = buildVariedMaterial(s);
        const sel = probes_mod.selectProbes(probe_list, Vec3.new(s.px, s.py, 0));
        h.update(std.mem.asBytes(v));
        h.update(std.mem.asBytes(&m));
        h.update(std.mem.asBytes(&sel));
    }
    const digest = h.final();
    std.mem.doNotOptimizeAway(digest);
    return digest;
}

/// Timed case: identical body, hashing replaced by a minimal barrier on
/// the full outputs (forces evaluation, performs no work).
fn runCase(
    ctx: *const uniforms_mod.FrameContext,
    shadow: uniforms_mod.ShadowState,
    probe_list: []const probes_mod.ProbeFrameEntry,
    fallback: bool,
    cfg: Cfg,
    alloc: std.mem.Allocator,
) !CaseResult {
    var ctx_local = ctx.*;
    if (fallback) {
        ctx_local.uniforms_with_shadows = null;
        ctx_local.uniforms_without_shadows = null;
    }
    for (0..cfg.warmup) |_| {
        for (0..cfg.draws) |i| {
            const recv = (i & 1) == 0;
            const s = variedScalars(i);
            const v = draw_mod.frameUniformsForState(shadow, recv, &ctx_local, Mat4.identity, .{ 0, 1, 1, 0 });
            const m = buildVariedMaterial(s);
            const sel = probes_mod.selectProbes(probe_list, Vec3.new(s.px, s.py, 0));
            std.mem.doNotOptimizeAway(v);
            std.mem.doNotOptimizeAway(m);
            std.mem.doNotOptimizeAway(sel);
        }
    }
    const pre = checksumCase(ctx, shadow, probe_list, fallback, cfg.draws);
    const times = try alloc.alloc(f64, cfg.reps);
    defer alloc.free(times);
    var sum: f64 = 0;
    for (0..cfg.reps) |r| {
        const t0 = sokol.time.now();
        for (0..cfg.draws) |i| {
            const recv = (i & 1) == 0;
            const s = variedScalars(i);
            const v = draw_mod.frameUniformsForState(shadow, recv, &ctx_local, Mat4.identity, .{ 0, 1, 1, 0 });
            const m = buildVariedMaterial(s);
            const sel = probes_mod.selectProbes(probe_list, Vec3.new(s.px, s.py, 0));
            std.mem.doNotOptimizeAway(v);
            std.mem.doNotOptimizeAway(m);
            std.mem.doNotOptimizeAway(sel);
        }
        const dt_ms: f64 = sokol.time.ms(sokol.time.now() -% t0);
        times[r] = dt_ms;
        sum += dt_ms;
    }
    const post = checksumCase(ctx, shadow, probe_list, fallback, cfg.draws);
    if (pre != post) std.debug.panic("bench: checksum drift pre={x} post={x} (inputs must be deterministic)", .{ pre, post });
    const med = median(times);
    var mn = times[0];
    var mx = times[0];
    for (times) |t| {
        mn = @min(mn, t);
        mx = @max(mx, t);
    }
    return .{ .median_ms = med, .min_ms = mn, .max_ms = mx, .total_ms = sum, .checksum = pre };
}

fn reportCase(name: []const u8, extra: []const u8, r: CaseResult, draws: usize) void {
    const per_draw_ns = r.median_ms * 1e6 / @as(f64, @floatFromInt(draws));
    std.debug.print("case={s}{s} checksum={x} median={d:.3}ms min={d:.3} max={d:.3} total={d:.1}ms per_draw_subset={d:.1}ns\n", .{
        name, extra, r.checksum, r.median_ms, r.min_ms, r.max_ms, r.total_ms, per_draw_ns,
    });
    if (r.total_ms < 100.0) {
        std.debug.print("case={s}{s} INCONCLUSIVE: total {d:.1}ms < 100ms; raise --draws/--reps\n", .{ name, extra, r.total_ms });
        std.process.exit(2);
    }
}

pub fn main(init: std.process.Init) !void {
    sokol.time.setup();
    const alloc = std.heap.smp_allocator;

    var arg_buf: [16][]const u8 = undefined;
    var n_args: usize = 0;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer it.deinit();
    _ = it.next();
    while (it.next()) |a| {
        if (n_args < arg_buf.len) {
            arg_buf[n_args] = a;
            n_args += 1;
        }
    }
    const cfg = parseArgs(arg_buf[0..n_args]);

    std.debug.print("draw_uniform_bench: draws={d} reps={d} warmup={d} probes={d} allocator=smp clock=sokol.time units=ms\n", .{
        cfg.draws, cfg.reps, cfg.warmup, cfg.probes,
    });
    std.debug.print("draw_uniform_bench: view_bytes={d} mat_bytes={d}\n", .{
        @sizeOf(uniforms_mod.FrameViewUniforms), @sizeOf(uniforms_mod.FrameMaterialUniforms),
    });

    var ctx = buildCtx();
    const shadow = buildShadow(true);
    const u_with = uniforms_mod.buildFrameView(shadow, &ctx, Mat4.identity, .{ 0, 1, 1, 0 });
    const u_no = uniforms_mod.buildFrameView(buildShadow(false), &ctx, Mat4.identity, .{ 0, 1, 1, 0 });
    ctx.uniforms_with_shadows = &u_with;
    ctx.uniforms_without_shadows = &u_no;

    // Parity (untimed, full bytes, BOTH receive policies).
    var ctx_nocache = ctx;
    ctx_nocache.uniforms_with_shadows = null;
    ctx_nocache.uniforms_without_shadows = null;
    for ([2]bool{ true, false }) |recv| {
        const va = draw_mod.frameUniformsForState(shadow, recv, &ctx, Mat4.identity, .{ 0, 1, 1, 0 });
        const vb = draw_mod.frameUniformsForState(shadow, recv, &ctx_nocache, Mat4.identity, .{ 0, 1, 1, 0 });
        if (!std.mem.eql(u8, std.mem.asBytes(va), std.mem.asBytes(vb))) {
            std.debug.panic("bench: cached vs fallback view bytes differ (recv={})", .{recv});
        }
    }

    var probe_store: [16]probes_mod.ProbeFrameEntry = undefined;
    for (&probe_store, 0..) |*e, i| {
        e.* = .{
            .position = Vec3.new(@as(f32, @floatFromInt(i)) * 1.5 - 11.0, 0, 0),
            .radius = 5.0,
            .enabled = true,
            .captured = true,
            .intensity = 0.5,
            .max_probe_lod = 7.0,
            .view = .{ .id = @intCast(31 + i) },
            .sampler = .{ .id = @intCast(101 + i) },
        };
    }
    const probes_active = probe_store[0..cfg.probes];

    reportCase("cached_empty", "", try runCase(&ctx, shadow, &.{}, false, cfg, alloc), cfg.draws);
    const b = try runCase(&ctx, shadow, &.{}, true, cfg, alloc);
    reportCase("fallback", "", b, cfg.draws);
    var probe_extra: [32]u8 = undefined;
    const extra = try std.fmt.bufPrint(&probe_extra, " probes={d}", .{cfg.probes});
    reportCase("cached", extra, try runCase(&ctx, shadow, probes_active, false, cfg, alloc), cfg.draws);
}
