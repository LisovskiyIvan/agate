const std = @import("std");
const types = @import("types.zig");
const Vertex = types.Vertex;
const GeometryData = types.GeometryData;
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const simplify = @import("simplify.zig");
const simplifyGeometry = simplify.simplifyGeometry;

test "Simplify: multiUV uv1 survives real decimation with split interpolation" {
    const ally = std.testing.allocator;
    // 4x4 grid plane (16 verts, 18 tris) with uv in [0,1] and
    // uv1 = 10*uv + (5,7), so the channels are nonzero and distinct.
    const n: usize = 4;
    var verts = try ally.alloc(Vertex, n * n);
    defer ally.free(verts);
    for (0..n) |iy| {
        for (0..n) |ix| {
            const fx: f32 = @as(f32, @floatFromInt(ix)) / @as(f32, @floatFromInt(n - 1));
            const fy: f32 = @as(f32, @floatFromInt(iy)) / @as(f32, @floatFromInt(n - 1));
            verts[iy * n + ix] = .{
                .position = .{ fx * 2.0 - 1.0, fy * 2.0 - 1.0, 0.0 },
                .normal = .{ 0.0, 0.0, 1.0 },
                .color = .{ 1.0, 1.0, 1.0, 1.0 },
                .uv = .{ fx, fy },
                .uv1 = .{ 10.0 * fx + 5.0, 10.0 * fy + 7.0 },
            };
        }
    }
    var idx: std.ArrayListUnmanaged(u32) = .empty;
    defer idx.deinit(ally);
    for (0..n - 1) |iy| {
        for (0..n - 1) |ix| {
            const a: u32 = @intCast(iy * n + ix);
            const b: u32 = @intCast(iy * n + ix + 1);
            const c: u32 = @intCast((iy + 1) * n + ix);
            const d: u32 = @intCast((iy + 1) * n + ix + 1);
            try idx.appendSlice(ally, &.{ a, b, c, b, d, c });
        }
    }
    var input = GeometryData{
        .vertices = verts,
        .indices = idx.items,
        .bounds = BoundingBox.init(Vec3.new(-1, -1, 0), Vec3.new(1, 1, 0)),
    };
    const initial_tris = input.indices.len / 3;

    var out = try simplifyGeometry(ally, &input, .{
        .target_ratio = 0.5,
        .preserve_border = false,
        .prevent_normal_flips = false,
    });
    defer out.deinit(ally);

    // Real decimation happened.
    try std.testing.expect(out.indices.len / 3 < initial_tris);
    try std.testing.expect(out.indices.len / 3 > 0);

    // Every surviving vertex must keep the split-channel relation
    // uv1 == 10*uv + (5,7): proves uv1 was copied, collapsed with the
    // same t as uv0, and emitted (not zeroed or mixed with uv0).
    var saw_nonzero_uv1 = false;
    for (out.vertices) |v| {
        try std.testing.expectApproxEqAbs(10.0 * v.uv[0] + 5.0, v.uv1[0], 1e-3);
        try std.testing.expectApproxEqAbs(10.0 * v.uv[1] + 7.0, v.uv1[1], 1e-3);
        if (v.uv1[0] != 0.0 or v.uv1[1] != 0.0) saw_nonzero_uv1 = true;
        // Distinct from uv0 (uv in [0,1], uv1 in [5,15]x[7,17]).
        try std.testing.expect(v.uv1[0] > 1.0 and v.uv1[1] > 1.0);
    }
    try std.testing.expect(saw_nonzero_uv1);
}
