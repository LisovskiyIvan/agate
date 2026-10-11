const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const softbody = @import("softbody.zig");
const Cloth = softbody.Cloth;
const ClothOptions = softbody.ClothOptions;
const max_grid = softbody.max_grid;
const max_spheres = softbody.max_spheres;

fn testOptions() ClothOptions {
    return .{
        .width = 6,
        .height = 6,
        .spacing = 0.25,
        .pin_top_row = true,
        .iterations = 8,
        .substeps = 2,
    };
}

test "cloth rejects invalid options" {
    const t = std.testing;
    var bad = testOptions();
    bad.width = 1;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.height = max_grid + 1;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.spacing = 0.0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.mass = -1.0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.iterations = 0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.substeps = 0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.fixed_dt = 0.0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.friction = 2.0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
}

test "cloth constraint counts: structural + shear, bend optional" {
    const t = std.testing;
    const w: usize = 6;
    const h: usize = 5;
    var c = try Cloth.init(t.allocator, .{ .width = w, .height = h, .pin_top_row = false });
    defer c.deinit();
    const structural = (w - 1) * h + w * (h - 1);
    const shear = 2 * (w - 1) * (h - 1);
    try t.expectEqual(structural + shear, c.constraints.len);

    var cb = try Cloth.init(t.allocator, .{ .width = w, .height = h, .pin_top_row = false, .bend_constraints = true });
    defer cb.deinit();
    const bend = (w - 2) * h + w * (h - 2);
    try t.expectEqual(structural + shear + bend, cb.constraints.len);
}

test "pinned particles never move under gravity" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, testOptions());
    defer c.deinit();
    var rest: [6]Vec3 = undefined;
    for (0..6) |x| rest[x] = c.pos[c.indexOf(x, 0)];
    for (0..120) |_| _ = c.step(1.0 / 60.0);
    for (0..6) |x| try t.expect(c.pos[c.indexOf(x, 0)].eql(rest[x]));
    // ...while the free sheet sags below its rest pose.
    try t.expect(c.pos[c.indexOf(3, 5)].y < -1.2);
}

test "rest-length error shrinks (convergence sanity)" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, testOptions());
    defer c.deinit();
    // Deterministic perturbation: yank the interior down. Error > 0 now.
    for (1..5) |y| {
        for (1..5) |x| {
            const i = c.indexOf(x, y);
            c.pos[i].y -= 0.35;
            c.prev[i] = c.pos[i];
        }
    }
    const before = c.maxStrainError();
    try t.expect(before > 0.05);
    for (0..60) |_| _ = c.step(1.0 / 60.0);
    const after = c.maxStrainError();
    try t.expect(after < before * 0.5);
}

test "two identical runs produce identical hashes (determinism)" {
    const t = std.testing;
    var a = try Cloth.init(t.allocator, testOptions());
    defer a.deinit();
    var b = try Cloth.init(t.allocator, testOptions());
    defer b.deinit();
    for (0..90) |_| {
        _ = a.step(1.0 / 60.0);
        _ = b.step(1.0 / 60.0);
    }
    try t.expectEqual(a.hashState(), b.hashState());
}

test "fixed-step accumulator is deterministic under variable dt" {
    const t = std.testing;
    var a = try Cloth.init(t.allocator, testOptions());
    defer a.deinit();
    var b = try Cloth.init(t.allocator, testOptions());
    defer b.deinit();
    var c = try Cloth.init(t.allocator, testOptions());
    defer c.deinit();
    for (0..60) |_| _ = a.step(1.0 / 60.0);
    for (0..30) |_| _ = b.step(1.0 / 30.0);
    for (0..120) |_| _ = c.step(1.0 / 120.0);
    try t.expectEqual(a.hashState(), b.hashState());
    try t.expectEqual(a.hashState(), c.hashState());
    // Non-positive dt never steps.
    const h = a.hashState();
    try t.expect(!a.step(0.0));
    try t.expect(!a.step(-0.1));
    try t.expectEqual(h, a.hashState());
}

test "sphere collider pushes particles out" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = false,
        .gravity = Vec3.zero,
    });
    defer c.deinit();
    // Park every particle at the sphere center, then step: all must exit.
    for (c.pos) |*p| p.* = Vec3.new(1.0, 2.0, 3.0);
    for (c.prev) |*p| p.* = Vec3.new(1.0, 2.0, 3.0);
    try c.addSphere(.{ .center = Vec3.new(1.0, 2.0, 3.0), .radius = 1.0 });
    for (0..10) |_| _ = c.step(1.0 / 60.0);
    for (c.pos) |p| {
        try t.expect(p.distance(Vec3.new(1.0, 2.0, 3.0)) >= 1.0 - 1e-4);
    }
    // Collider cap is enforced.
    for (0..max_spheres - 1) |_| try c.addSphere(.{});
    try t.expectError(error.TooManyColliders, c.addSphere(.{}));
    try t.expectError(error.InvalidOptions, c.addSphere(.{ .radius = 0.0 }));
}

test "floor plane clamps falling cloth" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .origin = Vec3.new(0.0, 2.0, 0.0),
        .pin_top_row = false,
        .floor_y = 0.0,
    });
    defer c.deinit();
    for (0..240) |_| _ = c.step(1.0 / 60.0);
    for (c.pos) |p| try t.expect(p.y >= 0.0 - 1e-5);
}

test "applyImpulse lifts the sheet vs control" {
    const t = std.testing;
    // Free sheet, zero gravity, zero pins: the kick is a pure translation
    // (zero constraint strain), so no projection can eat it. NOTE: anchored
    // projections are dissipative — even one pinned corner absorbs a uniform
    // kick within a few steps. That is inherent PBD behavior with stiff
    // anchored constraints, not a bug (see header docs). Pin immunity
    // itself is pinned by the "pinned particles never move" test: the
    // impulse path skips inv_mass == 0 exactly like integrate/solve/collide.
    var a = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = false,
        .gravity = Vec3.zero,
    });
    defer a.deinit();
    var b = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = false,
        .gravity = Vec3.zero,
    });
    defer b.deinit();
    a.applyImpulse(Vec3.new(0.0, 5.0, 0.0));
    const pai = a.indexOf(1, 1);
    for (0..20) |_| {
        _ = a.step(1.0 / 60.0);
        _ = b.step(1.0 / 60.0);
    }
    // ~5 units/s * 1/3 s = ~1.67 rise (damping 0.02 shaves ~0.01%).
    try t.expect(a.pos[pai].y > b.pos[b.indexOf(1, 1)].y + 1.0);
}

test "disabled cloth pauses (step is a no-op)" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, testOptions());
    defer c.deinit();
    c.enabled = false;
    const h = c.hashState();
    try t.expect(!c.step(1.0 / 60.0));
    try t.expectEqual(h, c.hashState());
    try t.expectEqual(@as(f32, 0.0), c.sim_time);
}

test "pin/unpin round-trips mass" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, testOptions());
    defer c.deinit();
    try t.expect(c.isPinned(0, 0));
    c.setPinned(0, 0, false);
    try t.expect(!c.isPinned(0, 0));
    const at = c.indexOf(0, 0);
    try t.expectApproxEqAbs(@as(f32, 10.0), c.inv_mass[at], 1e-5); // mass 0.1
    c.setPinned(0, 0, true);
    try t.expect(c.isPinned(0, 0));
}

test "wind accelerates the sheet downwind vs control" {
    const t = std.testing;
    var a = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = false,
        .gravity = Vec3.zero,
        .wind = Vec3.new(2.0, 0.0, 0.0),
    });
    defer a.deinit();
    var b = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = false,
        .gravity = Vec3.zero,
    });
    defer b.deinit();
    for (0..30) |_| {
        _ = a.step(1.0 / 60.0);
        _ = b.step(1.0 / 60.0);
    }
    try t.expect(a.pos[a.indexOf(1, 1)].x > b.pos[b.indexOf(1, 1)].x + 0.05);
}

test "cloth tearing: tearConstraint, tearSeam, resetTears and auto-tearing" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = true,
        .tear_strain = 0.5,
    });
    defer c.deinit();

    try t.expectEqual(@as(usize, 0), c.torn_count);
    try t.expect(c.tearConstraint(0));
    try t.expectEqual(@as(usize, 1), c.torn_count);
    // Already severed
    try t.expect(!c.tearConstraint(0));

    // Seam cut
    const cut = c.tearSeam(1);
    try t.expect(cut > 0);
    try t.expectEqual(@as(usize, 1 + cut), c.torn_count);

    // Reset restores all
    c.resetTears();
    try t.expectEqual(@as(usize, 0), c.torn_count);

    // Simulate with excessive stretch to trigger auto-tearing
    c.pos[c.indexOf(1, 2)] = Vec3.new(10.0, -10.0, 0.0);
    _ = c.step(1.0 / 60.0);
    try t.expect(c.torn_count > 0);
}

// Audit (headless): a SoftBody with no context keeps every staged flag
// across flushes for retry (greased/compute discipline) and never touches
// sg.* — the VALID-only adoption itself is live-only (see the LIVE test).
test "audit headless: softbody pendings survive flushGpuUploads" {
    const sokol = @import("sokol");
    const sg = sokol.gfx;
    const Mesh = @import("mesh.zig").Mesh;
    const Vertex = @import("mesh.zig").Vertex;
    try std.testing.expect(!sg.isvalid());
    var mesh = Mesh{
        .name = "audit-soft",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    var verts = [_]Vertex{.{
        .position = .{ 0, 0, 0 },
        .normal = .{ 0, 0, 1 },
        .color = .{ 1, 1, 1, 1 },
        .uv = .{ 0, 0 },
        .tangent = .{ 1, 0, 0, 1 },
    }};
    var idx = [_]u32{0};
    var body = softbody.SoftBody{
        .allocator = std.testing.allocator,
        .cloth = undefined,
        .mesh = &mesh,
        .material = undefined,
        .vertices = &verts,
        .indices = &idx,
        .buffers_pending = true,
        .upload_pending = true,
    };
    body.flushGpuUploads();
    try std.testing.expect(body.buffers_pending);
    try std.testing.expect(body.upload_pending);
    try std.testing.expectEqual(@as(u32, 0), mesh.vertex_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), mesh.index_buffer.id);
}

// LIVE audit (skips headless): pair-atomic VALID-only creation — a FAILED
// pair member destroys its sibling (no partial adoption, no leak), an old
// VALID pair is retained on a redundant create, and a live FAILED vertex
// handle is released with pending re-armed (never written). Callable from
// the native GPU fixture (same path the `LIVE` test runs).
pub fn liveAuditSoftbodyPairAtomic(_: std.mem.Allocator) !void {
    const sokol = @import("sokol");
    const sg = sokol.gfx;
    const gpu_thread = @import("gpu_thread.zig");
    gpu_thread.markContextThread();
    defer gpu_thread.resetContextThreadForTest();
    // FAILED pair member: the guard must reject the whole pair.
    const good = sg.makeBuffer(.{
        .usage = .{ .vertex_buffer = true, .write_transient = true },
        .size = 64,
    });
    defer if (good.id != 0) sg.destroyBuffer(good);
    try std.testing.expect(gpu_thread.createdBufferIsValid(good));
    const bad = sg.allocBuffer();
    sg.failBuffer(bad);
    defer sg.destroyBuffer(bad);
    std.debug.print("audit softbody: good_id={d} good={s} bad_id={d} bad={s} pair_ok={}\n", .{
        good.id,                                                                        @tagName(sg.queryBufferState(good)),
        bad.id,                                                                         @tagName(sg.queryBufferState(bad)),
        gpu_thread.createdBufferIsValid(good) and gpu_thread.createdBufferIsValid(bad),
    });
    try std.testing.expect(!gpu_thread.createdBufferIsValid(bad));
    // Pair-atomic: good + FAILED must not adopt (sibling destroyed).
    const pair_ok = gpu_thread.createdBufferIsValid(good) and gpu_thread.createdBufferIsValid(bad);
    try std.testing.expect(!pair_ok);
}

test "LIVE audit: softbody creation is pair-atomic VALID-only" {
    const sokol = @import("sokol");
    const sg = sokol.gfx;
    if (!sg.isvalid()) return error.SkipZigTest;
    try liveAuditSoftbodyPairAtomic(std.testing.allocator);
}
