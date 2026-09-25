const std = @import("std");
const Vec3 = @import("vec.zig").Vec3;
const Mat4 = @import("mat4.zig").Mat4;
const BoundingBox = @import("bounding_box.zig").BoundingBox;

pub const FrustumPlane = struct {
    normal: Vec3,
    d: f32,

    pub fn init(nx: f32, ny: f32, nz: f32, d: f32) FrustumPlane {
        const len = @sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 0.0) {
            return .{
                .normal = Vec3.new(nx / len, ny / len, nz / len),
                .d = d / len,
            };
        }
        return .{
            .normal = Vec3.new(nx, ny, nz),
            .d = d,
        };
    }
};

pub const Frustum = struct {
    // 6 planes: Left, Right, Bottom, Top, Near, Far
    planes: [6]FrustumPlane,

    /// Extract frustum planes from a column-major View-Projection matrix
    /// using the Gribb-Hartmann algorithm (adapted for [0, 1] clip space depth).
    pub fn fromViewProjection(vp: Mat4) Frustum {
        var f: Frustum = undefined;

        // Left plane: row 3 + row 0
        f.planes[0] = FrustumPlane.init(
            vp.m[3] + vp.m[0],
            vp.m[7] + vp.m[4],
            vp.m[11] + vp.m[8],
            vp.m[15] + vp.m[12],
        );

        // Right plane: row 3 - row 0
        f.planes[1] = FrustumPlane.init(
            vp.m[3] - vp.m[0],
            vp.m[7] - vp.m[4],
            vp.m[11] - vp.m[8],
            vp.m[15] - vp.m[12],
        );

        // Bottom plane: row 3 + row 1
        f.planes[2] = FrustumPlane.init(
            vp.m[3] + vp.m[1],
            vp.m[7] + vp.m[5],
            vp.m[11] + vp.m[9],
            vp.m[15] + vp.m[13],
        );

        // Top plane: row 3 - row 1
        f.planes[3] = FrustumPlane.init(
            vp.m[3] - vp.m[1],
            vp.m[7] - vp.m[5],
            vp.m[11] - vp.m[9],
            vp.m[15] - vp.m[13],
        );

        // Near plane: row 2 (for [0, 1] depth range)
        f.planes[4] = FrustumPlane.init(
            vp.m[2],
            vp.m[6],
            vp.m[10],
            vp.m[14],
        );

        // Far plane: row 3 - row 2 (for [0, 1] depth range)
        f.planes[5] = FrustumPlane.init(
            vp.m[3] - vp.m[2],
            vp.m[7] - vp.m[6],
            vp.m[11] - vp.m[10],
            vp.m[15] - vp.m[14],
        );

        return f;
    }

    /// Tests if an Axis-Aligned Bounding Box intersects or is inside the frustum using SIMD.
    /// Returns true if visible, false if entirely culled.
    pub fn intersectsAABB(self: Frustum, aabb: BoundingBox) bool {
        const c = aabb.center();
        const e = aabb.extents();
        const c_v: @Vector(4, f32) = .{ c.x, c.y, c.z, 1.0 };
        const e_v: @Vector(4, f32) = .{ e.x, e.y, e.z, 0.0 };

        inline for (self.planes) |p| {
            const p_v: @Vector(4, f32) = .{ p.normal.x, p.normal.y, p.normal.z, p.d };
            const abs_n: @Vector(4, f32) = .{ @abs(p.normal.x), @abs(p.normal.y), @abs(p.normal.z), 0.0 };
            const dist = @reduce(.Add, p_v * c_v);
            const radius = @reduce(.Add, abs_n * e_v);

            if (dist < -radius) {
                return false;
            }
        }
        return true;
    }

    /// Tests 4 AABBs simultaneously using SIMD @Vector(4, f32).
    /// Returns a @Vector(4, bool) where true = visible, false = culled.
    pub fn intersectsAABB4(
        self: Frustum,
        c_x: @Vector(4, f32),
        c_y: @Vector(4, f32),
        c_z: @Vector(4, f32),
        e_x: @Vector(4, f32),
        e_y: @Vector(4, f32),
        e_z: @Vector(4, f32),
    ) @Vector(4, bool) {
        var culled_mask: @Vector(4, bool) = @splat(false);

        inline for (self.planes) |p| {
            const nx: @Vector(4, f32) = @splat(p.normal.x);
            const ny: @Vector(4, f32) = @splat(p.normal.y);
            const nz: @Vector(4, f32) = @splat(p.normal.z);
            const pd: @Vector(4, f32) = @splat(p.d);

            const abs_nx: @Vector(4, f32) = @splat(@abs(p.normal.x));
            const abs_ny: @Vector(4, f32) = @splat(@abs(p.normal.y));
            const abs_nz: @Vector(4, f32) = @splat(@abs(p.normal.z));

            const dist = nx * c_x + ny * c_y + nz * c_z + pd;
            const radius = abs_nx * e_x + abs_ny * e_y + abs_nz * e_z;

            culled_mask = culled_mask | (dist < -radius);
        }

        return ~culled_mask;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "FrustumPlane init normalizes plane coefficients" {
    const p = FrustumPlane.init(3.0, 0.0, 4.0, 10.0);
    // Length is 5.0, so normal is (0.6, 0.0, 0.8), d is 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), p.normal.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), p.normal.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), p.normal.z, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), p.d, 1e-6);
}

test "Frustum fromViewProjection and intersectsAABB" {
    // Camera looking down -Z from (0, 0, 5) towards (0, 0, 0)
    const view = Mat4.lookAt(Vec3.new(0, 0, 5), Vec3.new(0, 0, 0), Vec3.up);
    const proj = Mat4.perspective(60.0, 1.0, 0.1, 100.0);
    const vp = proj.mul(view);
    const frustum = Frustum.fromViewProjection(vp);

    // Box at origin (0, 0, 0) is well inside the view frustum
    const box_center = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));
    try std.testing.expect(frustum.intersectsAABB(box_center));

    // Box behind camera (Z = 10) is culled
    const box_behind = BoundingBox.init(Vec3.new(-1, -1, 9), Vec3.new(1, 1, 11));
    try std.testing.expect(!frustum.intersectsAABB(box_behind));

    // Box far to the side (X = 200) is culled
    const box_side = BoundingBox.init(Vec3.new(199, -1, -1), Vec3.new(201, 1, 1));
    try std.testing.expect(!frustum.intersectsAABB(box_side));

    // Box past far plane (Z = -200) is culled
    const box_far = BoundingBox.init(Vec3.new(-1, -1, -205), Vec3.new(1, 1, -195));
    try std.testing.expect(!frustum.intersectsAABB(box_far));
}

test "Frustum intersectsAABB4 matches scalar intersectsAABB" {
    const view = Mat4.lookAt(Vec3.new(0, 0, 10), Vec3.new(0, 0, 0), Vec3.up);
    const proj = Mat4.perspective(60.0, 1.0, 0.5, 50.0);
    const vp = proj.mul(view);
    const frustum = Frustum.fromViewProjection(vp);

    const b0 = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)); // visible
    const b1 = BoundingBox.init(Vec3.new(-1, -1, 15), Vec3.new(1, 1, 20)); // behind
    const b2 = BoundingBox.init(Vec3.new(100, -1, -1), Vec3.new(102, 1, 1)); // side
    const b3 = BoundingBox.init(Vec3.new(0, 0, 2), Vec3.new(1, 1, 3)); // visible

    const c0 = b0.center();
    const e0 = b0.extents();
    const c1 = b1.center();
    const e1 = b1.extents();
    const c2 = b2.center();
    const e2 = b2.extents();
    const c3 = b3.center();
    const e3 = b3.extents();

    const cx: @Vector(4, f32) = .{ c0.x, c1.x, c2.x, c3.x };
    const cy: @Vector(4, f32) = .{ c0.y, c1.y, c2.y, c3.y };
    const cz: @Vector(4, f32) = .{ c0.z, c1.z, c2.z, c3.z };
    const ex: @Vector(4, f32) = .{ e0.x, e1.x, e2.x, e3.x };
    const ey: @Vector(4, f32) = .{ e0.y, e1.y, e2.y, e3.y };
    const ez: @Vector(4, f32) = .{ e0.z, e1.z, e2.z, e3.z };

    const simd_res = frustum.intersectsAABB4(cx, cy, cz, ex, ey, ez);

    try std.testing.expectEqual(frustum.intersectsAABB(b0), simd_res[0]);
    try std.testing.expectEqual(frustum.intersectsAABB(b1), simd_res[1]);
    try std.testing.expectEqual(frustum.intersectsAABB(b2), simd_res[2]);
    try std.testing.expectEqual(frustum.intersectsAABB(b3), simd_res[3]);
}
