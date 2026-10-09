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
