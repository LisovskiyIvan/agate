const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const buckets = @import("buckets.zig");
const shadowItemCulled = buckets.shadowItemCulled;
const ShadowPass = @import("core.zig").ShadowPass;

test "shadow item culling skips regular items outside the light frustum" {
    const frustum = math.Frustum.fromViewProjection(Mat4.identity);
    const inside_aabb = math.BoundingBox.init(
        math.Vec3.new(-0.5, -0.5, 0.2),
        math.Vec3.new(0.5, 0.5, 0.8),
    );
    const outside_aabb = math.BoundingBox.init(
        math.Vec3.new(5.0, 5.0, 5.0),
        math.Vec3.new(6.0, 6.0, 6.0),
    );

    var inside = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 1.0,
    };
    try std.testing.expect(!shadowItemCulled(inside, frustum, 0));

    inside.world_aabb = outside_aabb;
    try std.testing.expect(shadowItemCulled(inside, frustum, 0));
}

test "shadow item culling applies batch AABB and instance checks to instanced items" {
    const frustum = math.Frustum.fromViewProjection(Mat4.identity);
    const inside_aabb = math.BoundingBox.init(
        math.Vec3.new(-0.5, -0.5, 0.2),
        math.Vec3.new(0.5, 0.5, 0.8),
    );
    const outside_aabb = math.BoundingBox.init(
        math.Vec3.new(5.0, 5.0, 5.0),
        math.Vec3.new(6.0, 6.0, 6.0),
    );

    var item = ShadowPass.ShadowDrawItem{
        .world_aabb = outside_aabb,
        .max_dim = 1.0,
        .is_instanced = true,
        .visible_instance_count = 4,
        .instance_buffer = .{ .id = 1 },
    };
    try std.testing.expect(shadowItemCulled(item, frustum, 0));

    item.world_aabb = inside_aabb;
    try std.testing.expect(!shadowItemCulled(item, frustum, 0));

    item.visible_instance_count = 0;
    try std.testing.expect(shadowItemCulled(item, frustum, 0));

    item.visible_instance_count = 4;
    item.instance_buffer = .{};
    try std.testing.expect(shadowItemCulled(item, frustum, 0));
}

test "shadow item culling applies far-cascade max_dim policy to all items" {
    const frustum = math.Frustum.fromViewProjection(Mat4.identity);
    const inside_aabb = math.BoundingBox.init(
        math.Vec3.new(-0.5, -0.5, 0.2),
        math.Vec3.new(0.5, 0.5, 0.8),
    );

    const item = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 0.5,
    };
    try std.testing.expect(shadowItemCulled(item, frustum, 3));
    try std.testing.expect(!shadowItemCulled(item, frustum, 0));

    const inst = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 0.5,
        .is_instanced = true,
        .visible_instance_count = 4,
        .instance_buffer = .{ .id = 1 },
    };
    try std.testing.expect(shadowItemCulled(inst, frustum, 3));
    try std.testing.expect(!shadowItemCulled(inst, frustum, 0));
}

test "shadow item culling honors named cascade size thresholds exactly" {
    const frustum = math.Frustum.fromViewProjection(Mat4.identity);
    const inside_aabb = math.BoundingBox.init(
        math.Vec3.new(-0.5, -0.5, 0.2),
        math.Vec3.new(0.5, 0.5, 0.8),
    );

    // Cascade 0 never culls by size, however small the caster.
    const tiny = ShadowPass.ShadowDrawItem{ .world_aabb = inside_aabb, .max_dim = 0.001 };
    try std.testing.expect(!shadowItemCulled(tiny, frustum, 0));
    // Spot/point paths (null cascade) never cull by size either.
    try std.testing.expect(!shadowItemCulled(tiny, frustum, null));

    // Exact policy boundaries (strict less-than, mirrors types.CASCADE_MIN_DIM).
    const cases = [_]struct { c: usize, dim: f32, culled: bool }{
        .{ .c = 1, .dim = 0.11, .culled = true },
        .{ .c = 1, .dim = 0.12, .culled = false },
        .{ .c = 2, .dim = 0.34, .culled = true },
        .{ .c = 2, .dim = 0.35, .culled = false },
        .{ .c = 3, .dim = 0.74, .culled = true },
        .{ .c = 3, .dim = 0.75, .culled = false },
    };
    for (cases) |tc| {
        const item = ShadowPass.ShadowDrawItem{ .world_aabb = inside_aabb, .max_dim = tc.dim };
        try std.testing.expectEqual(tc.culled, shadowItemCulled(item, frustum, tc.c));
    }
}

test "shadow item culling skips invisible items" {
    const frustum = math.Frustum.fromViewProjection(Mat4.identity);
    const inside_aabb = math.BoundingBox.init(
        math.Vec3.new(-0.5, -0.5, 0.2),
        math.Vec3.new(0.5, 0.5, 0.8),
    );

    const hidden = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 1.0,
        .is_visible = false,
    };
    try std.testing.expect(shadowItemCulled(hidden, frustum, 0));

    const hidden_inst = ShadowPass.ShadowDrawItem{
        .world_aabb = inside_aabb,
        .max_dim = 1.0,
        .is_visible = false,
        .is_instanced = true,
        .visible_instance_count = 4,
        .instance_buffer = .{ .id = 1 },
    };
    try std.testing.expect(shadowItemCulled(hidden_inst, frustum, 0));
}
