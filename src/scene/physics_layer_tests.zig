const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const physics = @import("../physics.zig");
const stats_mod = @import("stats.zig");
const physics_layer = @import("physics_layer.zig");
const PhysicsIntegration = physics_layer.PhysicsIntegration;
const TestMesh = @import("../mesh.zig").Mesh;

fn makeDebugFixture(allocator: std.mem.Allocator) !struct {
    integ: PhysicsIntegration,
    mesh: *TestMesh,
} {
    var integ = PhysicsIntegration{};
    _ = integ.enable(allocator, null);
    const m = try allocator.create(TestMesh);
    m.* = .{ .name = "build_dbg", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    _ = try integ.getWorld().?.createBody(m, .box, 0.0);
    integ.show_debug = true;
    return .{ .integ = integ, .mesh = m };
}

test "physics buildDebug+latchDebug isolates live mutations" {
    const t = std.testing;
    var fx = try makeDebugFixture(t.allocator);
    defer fx.integ.deinit(t.allocator);
    defer t.allocator.destroy(fx.mesh);
    var integ = &fx.integ;

    integ.buildDebug(t.allocator, 9);
    try t.expectEqual(@as(u64, 9), integ.build_seq.load(.acquire));
    try t.expect(integ.build_visible);
    try t.expectEqual(@as(usize, 12), integ.build_lines.items.len);
    try t.expect(!integ.prepared_visible);
    try t.expectEqual(@as(usize, 0), integ.prepared_lines.items.len);

    const x0 = integ.build_lines.items[0].a.x;
    fx.mesh.position = Vec3.new(5, 0, 0);
    try t.expectEqual(x0, integ.build_lines.items[0].a.x);

    integ.latchDebug(t.allocator);
    try t.expectEqual(@as(u64, 9), integ.latched_seq);
    try t.expect(integ.prepared_visible);
    try t.expectEqual(@as(usize, 12), integ.prepared_lines.items.len);
    try t.expectApproxEqAbs(x0, integ.prepared_lines.items[0].a.x, 1e-4);
    integ.step(0.016);
    try t.expectApproxEqAbs(x0, integ.prepared_lines.items[0].a.x, 1e-4);
}

test "physics two builds before latch: newest wins, stale latch recaptures" {
    const t = std.testing;
    var fx = try makeDebugFixture(t.allocator);
    defer fx.integ.deinit(t.allocator);
    defer t.allocator.destroy(fx.mesh);
    var integ = &fx.integ;

    integ.buildDebug(t.allocator, 1);
    const x0 = integ.build_lines.items[0].a.x;
    fx.mesh.position = Vec3.new(5, 0, 0);
    integ.buildDebug(t.allocator, 2);
    try t.expectApproxEqAbs(x0 + 5.0, integ.build_lines.items[0].a.x, 1e-4);

    integ.latchDebug(t.allocator);
    try t.expectApproxEqAbs(x0 + 5.0, integ.prepared_lines.items[0].a.x, 1e-4);

    fx.mesh.position = Vec3.new(9, 0, 0);
    integ.latchDebug(t.allocator);
    try t.expectApproxEqAbs(x0 + 9.0, integ.prepared_lines.items[0].a.x, 1e-4);
}

test "physics buildDebug OOM fail-closes, latch publishes empty, then recovers" {
    const t = std.testing;
    var fx = try makeDebugFixture(t.allocator);
    defer fx.integ.deinit(t.allocator);
    defer t.allocator.destroy(fx.mesh);
    var integ = &fx.integ;

    integ.buildDebug(t.allocator, 1);
    integ.latchDebug(t.allocator);
    try t.expect(integ.prepared_visible);
    try t.expectEqual(@as(usize, 12), integ.prepared_lines.items.len);

    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    integ.build_lines.clearAndFree(t.allocator);
    integ.buildDebug(failing.allocator(), 2);
    try t.expect(!integ.build_visible);
    try t.expectEqual(@as(usize, 0), integ.build_lines.items.len);
    integ.latchDebug(t.allocator);
    try t.expect(!integ.prepared_visible);
    try t.expectEqual(@as(usize, 0), integ.prepared_lines.items.len);

    integ.buildDebug(t.allocator, 3);
    integ.latchDebug(t.allocator);
    try t.expect(integ.prepared_visible);
    try t.expectEqual(@as(usize, 12), integ.prepared_lines.items.len);
}

test "physics stageIntoSlot+latchSlotDebug freezes the build generation" {
    const t = std.testing;
    var fx = try makeDebugFixture(t.allocator);
    defer fx.integ.deinit(t.allocator);
    defer t.allocator.destroy(fx.mesh);
    var integ = &fx.integ;

    integ.buildDebug(t.allocator, 9);
    var slot_lines: std.ArrayListUnmanaged(physics.DebugLine) = .empty;
    defer slot_lines.deinit(t.allocator);
    var slot_visible = false;
    integ.stageIntoSlot(t.allocator, &slot_lines, &slot_visible);
    try t.expect(slot_visible);
    try t.expectEqual(@as(usize, 12), slot_lines.items.len);
    const x0 = slot_lines.items[0].a.x;

    fx.mesh.position = Vec3.new(5, 0, 0);
    integ.show_debug = false;
    integ.build_lines.items[0].a.x += 100.0;
    integ.build_visible = false;
    try t.expect(slot_visible);
    try t.expectApproxEqAbs(x0, slot_lines.items[0].a.x, 1e-4);

    integ.latchSlotDebug(t.allocator, slot_lines.items, slot_visible);
    try t.expectEqual(@as(u64, 9), integ.latched_seq);
    try t.expect(integ.prepared_visible);
    try t.expectEqual(@as(usize, 12), integ.prepared_lines.items.len);
    try t.expectApproxEqAbs(x0, integ.prepared_lines.items[0].a.x, 1e-4);

    integ.latchDebug(t.allocator);
    try t.expect(!integ.prepared_visible);
    try t.expectEqual(@as(usize, 0), integ.prepared_lines.items.len);
}

test "physics stageIntoSlot newest wins; OOM fail-closes, slot latch publishes empty" {
    const t = std.testing;
    var fx = try makeDebugFixture(t.allocator);
    defer fx.integ.deinit(t.allocator);
    defer t.allocator.destroy(fx.mesh);
    var integ = &fx.integ;

    integ.buildDebug(t.allocator, 1);
    const x0 = integ.build_lines.items[0].a.x;
    fx.mesh.position = Vec3.new(5, 0, 0);
    integ.buildDebug(t.allocator, 2);
    var slot_lines: std.ArrayListUnmanaged(physics.DebugLine) = .empty;
    defer slot_lines.deinit(t.allocator);
    var slot_visible = false;
    integ.stageIntoSlot(t.allocator, &slot_lines, &slot_visible);
    try t.expect(slot_visible);
    try t.expectApproxEqAbs(x0 + 5.0, slot_lines.items[0].a.x, 1e-4);

    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    slot_lines.clearAndFree(t.allocator);
    integ.stageIntoSlot(failing.allocator(), &slot_lines, &slot_visible);
    try t.expect(!slot_visible);
    try t.expectEqual(@as(usize, 0), slot_lines.items.len);
    integ.latchSlotDebug(t.allocator, slot_lines.items, slot_visible);
    try t.expect(!integ.prepared_visible);
    try t.expectEqual(@as(usize, 0), integ.prepared_lines.items.len);

    integ.stageIntoSlot(t.allocator, &slot_lines, &slot_visible);
    integ.latchSlotDebug(t.allocator, slot_lines.items, slot_visible);
    try t.expect(integ.prepared_visible);
    try t.expectEqual(@as(usize, 12), integ.prepared_lines.items.len);
    try t.expectApproxEqAbs(x0 + 5.0, integ.prepared_lines.items[0].a.x, 1e-4);
}

test "physics upload tag matches only the tagged shape" {
    const t = std.testing;
    var integ = PhysicsIntegration{};
    try t.expect(!integ.uploadTagMatches(1, .BGRA8));
    integ.prepared_upload_valid = true;
    integ.prepared_upload_samples = 4;
    integ.prepared_upload_format = .RGBA16F;
    try t.expect(integ.uploadTagMatches(4, .RGBA16F));
    try t.expect(!integ.uploadTagMatches(1, .RGBA16F));
    try t.expect(!integ.uploadTagMatches(4, .BGRA8));
    try t.expect(!integ.uploadTagMatches(2, .RGBA16F));
    integ.clearUploadTag();
    try t.expect(!integ.uploadTagMatches(4, .RGBA16F));
}

test "physics upload clears tag headless; target draw stays silent" {
    const t = std.testing;
    var integ = PhysicsIntegration{};
    defer integ.prepared_lines.deinit(t.allocator);
    integ.prepared_upload_valid = true;
    integ.prepared_upload_samples = 4;
    integ.prepared_upload_format = .RGBA16F;
    integ.uploadDebug(t.allocator, 4, .RGBA16F);
    try t.expect(!integ.prepared_upload_valid);
    try t.expect(integ.debug_pass == null);
    try t.expect(integ.debug_pass_msaa == null);

    var stats = stats_mod.SceneStats{};
    integ.renderDebugPrepared(Mat4.identity, 4, .RGBA16F, &stats);
    try t.expectEqual(stats_mod.SceneStats{}, stats);
    integ.renderDebugPrepared(Mat4.identity, 1, .BGRA8, &stats);
    try t.expectEqual(stats_mod.SceneStats{}, stats);

    integ.prepared_upload_valid = true;
    integ.uploadDebug(t.allocator, 1, .BGRA8);
    try t.expect(!integ.prepared_upload_valid);
    try t.expect(integ.debug_pass == null);
}
