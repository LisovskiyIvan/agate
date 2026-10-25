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
    plane_vecs: [6]@Vector(4, f32) = undefined,
    abs_normals: [6]@Vector(4, f32) = undefined,

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

        inline for (0..6) |i| {
            const p = f.planes[i];
            f.plane_vecs[i] = .{ p.normal.x, p.normal.y, p.normal.z, p.d };
            f.abs_normals[i] = .{ @abs(p.normal.x), @abs(p.normal.y), @abs(p.normal.z), 0.0 };
        }

        return f;
    }

    /// Tests if an Axis-Aligned Bounding Box intersects or is inside the frustum using SIMD.
    /// Returns true if visible, false if entirely culled.
    pub fn intersectsAABB(self: Frustum, aabb: BoundingBox) bool {
        const c = aabb.center();
        const e = aabb.extents();
        const c_v: @Vector(4, f32) = .{ c.x, c.y, c.z, 1.0 };
        const e_v: @Vector(4, f32) = .{ e.x, e.y, e.z, 0.0 };

        inline for (0..6) |i| {
            const dist = @reduce(.Add, self.plane_vecs[i] * c_v);
            const radius = @reduce(.Add, self.abs_normals[i] * e_v);

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

        inline for (0..6) |i| {
            const pv = self.plane_vecs[i];
            const an = self.abs_normals[i];
            const nx: @Vector(4, f32) = @splat(pv[0]);
            const ny: @Vector(4, f32) = @splat(pv[1]);
            const nz: @Vector(4, f32) = @splat(pv[2]);
            const pd: @Vector(4, f32) = @splat(pv[3]);

            const abs_nx: @Vector(4, f32) = @splat(an[0]);
            const abs_ny: @Vector(4, f32) = @splat(an[1]);
            const abs_nz: @Vector(4, f32) = @splat(an[2]);

            const dist = @mulAdd(@Vector(4, f32), nx, c_x, @mulAdd(@Vector(4, f32), ny, c_y, @mulAdd(@Vector(4, f32), nz, c_z, pd)));
            const radius = @mulAdd(@Vector(4, f32), abs_nx, e_x, @mulAdd(@Vector(4, f32), abs_ny, e_y, abs_nz * e_z));

            culled_mask = culled_mask | (dist < -radius);
        }

        return ~culled_mask;
    }
};
