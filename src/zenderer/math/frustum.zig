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

    /// Tests if an Axis-Aligned Bounding Box intersects or is inside the frustum.
    /// Returns true if visible, false if entirely culled.
    pub fn intersectsAABB(self: Frustum, aabb: BoundingBox) bool {
        const c = aabb.center();
        const e = aabb.extents();

        for (self.planes) |p| {
            // Distance of center to plane
            const dist = p.normal.dot(c) + p.d;
            // Maximum projection of extents along plane normal
            const radius = @abs(p.normal.x) * e.x +
                @abs(p.normal.y) * e.y +
                @abs(p.normal.z) * e.z;

            // If the box is completely on the negative side of any plane, it's outside!
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
