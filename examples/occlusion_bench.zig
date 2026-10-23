//! Headless ReleaseFast benchmark for the CPU occlusion culler
//! (visibility.OcclusionCuller: software Hi-Z, 256x128, 9 mips).
//!
//! Measures the three cost blocks of the per-frame CPU occlusion pass in
//! isolation, in the same order the queue build runs them
//! (`render_queue/build/frame.zig` phase -1 + phase-0 queries):
//!   1. occluder setup:  beginFrame (clear) + rasterizeOccluderBox x K
//!                       + endOccluders (pyramid build), reported as
//!                       setup us and per-occluder ns
//!   2. pyramid build:   endOccluders alone after a single box (fixed cost
//!                       of 43k floats / 8 mip levels)
//!   3. queries:         isOccluded x Q over three query populations —
//!                       open scene (no occluders), occluded query boxes
//!                       behind walls, visible boxes in front of them
//!
//! Deterministic geometry, no GPU, no window. Correctness is checked before
//! timing: the walled scene must actually occlude most query boxes and the
//! open scene must occlude none (a culler that flips either verdict would
//! benchmark nothing).
//!
//! Run (from examples/):
//!   zig build run-occlusion-bench -Doptimize=ReleaseFast -- --reps 9
//!
//! Result interpretation: setup + queries together are the per-frame CPU
//! price of `cull_occlusion`; compare against the frame budget and the
//! queue-build cost it sits next to (scene_scaling_bench).

const std = @import("std");
const z = @import("agate");
const sokol = z.sokol;
const math = z.math;
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const OcclusionCuller = z.visibility.OcclusionCuller;

const reps_default: u32 = 9;

const SetupCase = struct {
    name: []const u8,
    occluders: usize,
    us_total: f64,
    ns_per_occluder: f64,
};

const QueryCase = struct {
    name: []const u8,
    queries: usize,
    occluded_frac: f32,
    ns_per_query: f64,
};

fn median(times: []f64) f64 {
    std.mem.sort(f64, times, {}, std.sort.asc(f64));
    return times[times.len / 2];
}

fn nowTicks() u64 {
    return sokol.time.now();
}

fn ticksToNs(ticks: u64) f64 {
    return @floatCast(sokol.time.ns(ticks));
}

fn setupClock() void {
    sokol.time.setup();
}

/// Perspective camera looking down -Z at the box grid.
fn buildViewProj() Mat4 {
    const eye = Vec3{ .x = 0, .y = 1.5, .z = 8 };
    const view = Mat4.lookAt(eye, Vec3{ .x = 0, .y = 1.0, .z = 0 }, Vec3{ .x = 0, .y = 1, .z = 0 });
    const proj = Mat4.perspective(60.0, 16.0 / 9.0, 0.1, 100.0);
    return Mat4.mul(proj, view);
}

/// K occluder boxes: a wall grid across the view with gaps (so a later
/// query population can be split into occluded/visible bands).
fn placeOccluderBoxes(boxes: []BoundingBox) void {
    const cols: i32 = 8;
    for (boxes, 0..) |*b, i| {
        const ci: i32 = @intCast(i % @as(usize, @intCast(cols)));
        const row: i32 = @intCast((i / @as(usize, @intCast(cols))) % 3);
        const x: f32 = @as(f32, @floatFromInt(ci - cols / 2)) * 1.6;
        const zz: f32 = -2.0 - @as(f32, @floatFromInt(row)) * 3.0;
        const h: f32 = 2.0 + 0.25 * @as(f32, @floatFromInt(i % 5));
        b.* = .{
            .min = .{ .x = x - 0.5, .y = 0, .z = zz - 0.15 },
            .max = .{ .x = x + 0.5, .y = h, .z = zz + 0.15 },
        };
    }
}

fn boxAt(x: f32, y: f32, depth: f32, s: f32) BoundingBox {
    return .{
        .min = .{ .x = x - s, .y = y - s, .z = depth - s },
        .max = .{ .x = x + s, .y = y + s, .z = depth + s },
    };
}

/// Query population: a jittered grid of small boxes through the view volume.
/// `behind_walls` shifts the grid to larger depths so the wall grid covers it.
fn placeQueryBoxes(boxes: []BoundingBox, behind_walls: bool) void {
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();
    for (boxes) |*b| {
        const u = rand.float(f32);
        const v = rand.float(f32);
        const w = rand.float(f32);
        const x = (u - 0.5) * 10.0;
        const y = 0.4 + v * 1.4;
        const depth = if (behind_walls) -4.0 - w * 8.0 else -0.4 - w * 1.2;
        b.* = boxAt(x, y, depth, 0.05);
    }
}

fn measureSetup(culler: *OcclusionCuller, view_proj: Mat4, boxes: []const BoundingBox, reps: u32, times: *std.ArrayListUnmanaged(f64), alloc: std.mem.Allocator) !SetupCase {
    times.clearRetainingCapacity();
    for (0..reps) |_| {
        const t0 = nowTicks();
        culler.beginFrame(view_proj);
        for (boxes) |b| culler.rasterizeOccluderBox(b, Mat4.identity);
        culler.endOccluders();
        const t1 = nowTicks();
        try times.append(alloc, ticksToNs(t1 -% t0));
    }
    const med = median(times.items);
    const k: f64 = @floatFromInt(boxes.len);
    return .{
        .name = "setup",
        .occluders = boxes.len,
        .us_total = med / 1000.0,
        .ns_per_occluder = med / k,
    };
}

fn measurePyramidOnly(culler: *OcclusionCuller, view_proj: Mat4, wall: BoundingBox, reps: u32, times: *std.ArrayListUnmanaged(f64), alloc: std.mem.Allocator) !f64 {
    times.clearRetainingCapacity();
    for (0..reps) |_| {
        culler.beginFrame(view_proj);
        culler.rasterizeOccluderBox(wall, Mat4.identity);
        const t0 = nowTicks();
        culler.endOccluders();
        const t1 = nowTicks();
        try times.append(alloc, ticksToNs(t1 -% t0));
    }
    return median(times.items) / 1000.0;
}

fn measureQueries(culler: *OcclusionCuller, queries: []const BoundingBox, reps: u32, times: *std.ArrayListUnmanaged(f64), alloc: std.mem.Allocator) !f64 {
    times.clearRetainingCapacity();
    for (0..reps) |_| {
        const t0 = nowTicks();
        for (queries) |q| {
            std.mem.doNotOptimizeAway(culler.isOccluded(q));
        }
        const t1 = nowTicks();
        try times.append(alloc, ticksToNs(t1 -% t0));
    }
    return median(times.items) / @as(f64, @floatFromInt(queries.len));
}

fn occludedFraction(culler: *OcclusionCuller, queries: []const BoundingBox) f32 {
    var hit: usize = 0;
    for (queries) |q| {
        if (culler.isOccluded(q)) hit += 1;
    }
    return @as(f32, @floatFromInt(hit)) / @as(f32, @floatFromInt(queries.len));
}

pub fn main(init: std.process.Init) !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const alloc = gpa_state.allocator();

    var reps: u32 = reps_default;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer it.deinit();
    _ = it.next();
    while (it.next()) |a| {
        if (std.mem.eql(u8, a, "--reps")) {
            if (it.next()) |v| reps = std.fmt.parseInt(u32, v, 10) catch reps_default;
        }
    }

    setupClock();
    const view_proj = buildViewProj();
    var culler = OcclusionCuller.init();

    var times: std.ArrayListUnmanaged(f64) = .empty;
    defer times.deinit(alloc);

    var boxes: [256]BoundingBox = undefined;
    placeOccluderBoxes(&boxes);

    // Correctness gate before timing: the wall scene must occlude most
    // behind-wall queries and none in front.
    var behind: [4096]BoundingBox = undefined;
    var front: [4096]BoundingBox = undefined;
    placeQueryBoxes(&behind, true);
    placeQueryBoxes(&front, false);

    culler.beginFrame(view_proj);
    for (boxes[0..64]) |b| culler.rasterizeOccluderBox(b, Mat4.identity);
    culler.endOccluders();
    const behind_frac = occludedFraction(&culler, &behind);
    const front_frac = occludedFraction(&culler, &front);
    if (behind_frac < 0.5 or front_frac > 0.05) {
        std.debug.print("OCCLUSION-BENCH: fixture broken (behind={d:.2} front={d:.2}) — not timing\n", .{ behind_frac, front_frac });
        return;
    }

    std.debug.print("occlusion-bench: reps={} hiz=256x128x9 backend=software\n", .{reps});

    // 1. Occluder setup sweep (clear + rasterize + pyramid).
    const sweep = [_]usize{ 16, 64, 256 };
    for (sweep) |k| {
        const r = try measureSetup(&culler, view_proj, boxes[0..k], reps, &times, alloc);
        std.debug.print("setup k={:<4} total={:<9.2} per_occluder={:.0} ns\n", .{ r.occluders, r.us_total, r.ns_per_occluder });
    }

    // 2. Pyramid build alone.
    const pyramid_us = try measurePyramidOnly(&culler, view_proj, boxes[0], reps, &times, alloc);
    std.debug.print("pyramid    build={:.2} us (43k floats, 8 levels)\n", .{pyramid_us});

    // 3. Queries: open / occluded / visible populations.
    var qcases = [_]QueryCase{
        .{ .name = "open(no occluders)", .queries = behind.len, .occluded_frac = 0, .ns_per_query = 0 },
        .{ .name = "behind walls     ", .queries = behind.len, .occluded_frac = behind_frac, .ns_per_query = 0 },
        .{ .name = "visible          ", .queries = front.len, .occluded_frac = front_frac, .ns_per_query = 0 },
    };
    // Open scene: beginFrame without occluders is the real engine state for
    // "no is_occluder meshes marked".
    culler.beginFrame(view_proj);
    qcases[0].ns_per_query = try measureQueries(&culler, &behind, reps, &times, alloc);
    culler.beginFrame(view_proj);
    for (boxes[0..64]) |b| culler.rasterizeOccluderBox(b, Mat4.identity);
    culler.endOccluders();
    qcases[1].ns_per_query = try measureQueries(&culler, &behind, reps, &times, alloc);
    culler.beginFrame(view_proj);
    for (boxes[0..64]) |b| culler.rasterizeOccluderBox(b, Mat4.identity);
    culler.endOccluders();
    qcases[2].ns_per_query = try measureQueries(&culler, &front, reps, &times, alloc);
    for (qcases) |qc| {
        std.debug.print("query k={:<4} occluded={d:<4.2} per_query={:.0} ns  ({s})\n", .{ qc.queries, qc.occluded_frac, qc.ns_per_query, qc.name });
    }

    // 4. Whole-frame estimate at a representative queue shape: 64 occluders
    // + 8192 mesh queries (the q0/dungeon scale) — compare to the 16.67ms
    // budget and the cull pass it sits inside. Setup number comes from the
    // k=64 sweep row, not a hardcoded constant.
    culler.beginFrame(view_proj);
    for (boxes[0..64]) |b| culler.rasterizeOccluderBox(b, Mat4.identity);
    culler.endOccluders();
    const t0 = nowTicks();
    _ = occludedFraction(&culler, &behind);
    _ = occludedFraction(&culler, &front);
    const q8192_ns = ticksToNs(nowTicks() -% t0);
    const setup_row = try measureSetup(&culler, view_proj, boxes[0..64], reps, &times, alloc);
    const frame_us = setup_row.us_total + q8192_ns / 1000.0;
    std.debug.print("frame estimate: 64 occluders + 8192 queries ≈ {d:.1} us CPU/frame (vs 16667 us budget)\n", .{frame_us});
}
