const std = @import("std");
const Vec3 = @import("vec.zig").Vec3;
const Vec2 = @import("vec.zig").Vec2;
const Quat = @import("quat.zig").Quat;

pub const Mat4 = extern struct {
    // Column-major: m[col * 4 + row]
    m: [16]f32,

    pub const identity = Mat4{ .m = .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    } };

    /// Scalar matrix multiplication (3 nested loops)
    pub fn mulScalar(a: Mat4, b: Mat4) Mat4 {
        var r: Mat4 = undefined;
        for (0..4) |col| {
            for (0..4) |row| {
                var sum: f32 = 0;
                for (0..4) |k| sum += a.m[k * 4 + row] * b.m[col * 4 + k];
                r.m[col * 4 + row] = sum;
            }
        }
        return r;
    }

    /// SIMD matrix multiplication using @Vector(4, f32) (ARM NEON / AVX)
    pub fn mulSimd(a: Mat4, b: Mat4) Mat4 {
        const a0: @Vector(4, f32) = a.m[0..4].*;
        const a1: @Vector(4, f32) = a.m[4..8].*;
        const a2: @Vector(4, f32) = a.m[8..12].*;
        const a3: @Vector(4, f32) = a.m[12..16].*;

        var r: Mat4 = undefined;
        inline for (0..4) |col| {
            const b_col = b.m[col * 4 .. col * 4 + 4];
            const v0: @Vector(4, f32) = @splat(b_col[0]);
            const v1: @Vector(4, f32) = @splat(b_col[1]);
            const v2: @Vector(4, f32) = @splat(b_col[2]);
            const v3: @Vector(4, f32) = @splat(b_col[3]);

            const res = a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3;
            const arr: [4]f32 = res;
            @memcpy(r.m[col * 4 .. col * 4 + 4], &arr);
        }
        return r;
    }

    pub fn mul(a: Mat4, b: Mat4) Mat4 {
        return mulSimd(a, b);
    }

    pub fn perspective(fov_y_deg: f32, aspect: f32, near: f32, far: f32) Mat4 {
        const f = 1.0 / std.math.tan(fov_y_deg * 0.5 * std.math.pi / 180.0);
        return .{ .m = .{
            f / aspect, 0, 0,                           0,
            0,          f, 0,                           0,
            0,          0, far / (near - far),          -1,
            0,          0, (far * near) / (near - far), 0,
        } };
    }

    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) Mat4 {
        const w = right - left;
        const h = top - bottom;
        const d = far - near;
        return .{ .m = .{
            2.0 / w,             0.0,                 0.0,       0.0,
            0.0,                 2.0 / h,             0.0,       0.0,
            0.0,                 0.0,                 -1.0 / d,  0.0,
            -(right + left) / w, -(top + bottom) / h, -near / d, 1.0,
        } };
    }

    pub fn lookAt(eye: Vec3, target: Vec3, up_arg: Vec3) Mat4 {
        const f = target.sub(eye).normalize();
        const s = f.cross(up_arg).normalize();
        const u = s.cross(f);

        var r = identity;
        r.m[0] = s.x;
        r.m[4] = s.y;
        r.m[8] = s.z;
        r.m[12] = -s.dot(eye);

        r.m[1] = u.x;
        r.m[5] = u.y;
        r.m[9] = u.z;
        r.m[13] = -u.dot(eye);

        r.m[2] = -f.x;
        r.m[6] = -f.y;
        r.m[10] = -f.z;
        r.m[14] = f.dot(eye);

        return r;
    }

    pub fn translation(v: Vec3) Mat4 {
        var r = identity;
        r.m[12] = v.x;
        r.m[13] = v.y;
        r.m[14] = v.z;
        return r;
    }

    pub fn scaling(s: Vec3) Mat4 {
        var r = identity;
        r.m[0] = s.x;
        r.m[5] = s.y;
        r.m[10] = s.z;
        return r;
    }

    pub fn rotationX(deg: f32) Mat4 {
        const rad = deg * std.math.pi / 180.0;
        const c = std.math.cos(rad);
        const s = std.math.sin(rad);
        return .{ .m = .{
            1, 0,  0, 0,
            0, c,  s, 0,
            0, -s, c, 0,
            0, 0,  0, 1,
        } };
    }

    pub fn rotationY(deg: f32) Mat4 {
        const rad = deg * std.math.pi / 180.0;
        const c = std.math.cos(rad);
        const s = std.math.sin(rad);
        return .{ .m = .{
            c, 0, -s, 0,
            0, 1, 0,  0,
            s, 0, c,  0,
            0, 0, 0,  1,
        } };
    }

    pub fn rotationZ(deg: f32) Mat4 {
        const rad = deg * std.math.pi / 180.0;
        const c = std.math.cos(rad);
        const s = std.math.sin(rad);
        return .{ .m = .{
            c,  s, 0, 0,
            -s, c, 0, 0,
            0,  0, 1, 0,
            0,  0, 0, 1,
        } };
    }

    /// Analytic Euler->matrix build of T * Rz * Ry * Rx * S (degrees, order
    /// Rz*Ry*Rx — same as Quat.fromEulerDeg). Bit-identical to the old
    /// composed path (translation/rotationX/Y/Z/scaling + mul); kept so by
    /// evaluating every sum in the same order the 0/1 entries of the
    /// elementary rotations force on mulSimd (explicit parens below).
    ///
    /// Derivation (row-major 3x3 view; column-major storage m[col*4+row]):
    ///   Rx = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
    ///   Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
    ///   Rz = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
    ///   Ry*Rx = [[cy, sy*sx, sy*cx],
    ///            [0,  cx,    -sx   ],
    ///            [-sy, cy*sx, cy*cx]]
    ///   R = Rz*(Ry*Rx):
    ///   row0 = cz*row0(RyRx) - sz*row1(RyRx)
    ///        = [cz*cy, cz*(sy*sx) - sz*cx, cz*(sy*cx) + sz*sx]
    ///   row1 = sz*row0(RyRx) + cz*row1(RyRx)
    ///        = [sz*cy, sz*(sy*sx) + cz*cx, sz*(sy*cx) - cz*sx]
    ///   row2 = [-sy, cy*sx, cy*cx]
    /// Scale is applied per column (col j *= scale_j), translation sets the
    /// last column — same as T*(R*S) since T and S are pure diagonal/shift.
    /// The trailing `+ 0.0` on each entry mirrors the composed path's exact
    /// zero dot-product terms: they canonicalize -0 to +0, so the analytic
    /// result matches mulSimd bit-for-bit (x + 0.0 == x for all x but -0).
    /// Two are needed per rotation/scale entry: one on the rotation element
    /// (mulSimd's Rz*(Ry*Rx) trailing zeros) and one after column scaling
    /// (trailing zeros of (R*S)); the translation lanes get one as well
    /// since T*(R*S) accumulates the pos lane onto exact-zero terms.
    pub fn fromRotationTranslationScale(pos: Vec3, rot_deg: Vec3, scale_v: Vec3) Mat4 {
        const rad_x = rot_deg.x * std.math.pi / 180.0;
        const cx = std.math.cos(rad_x);
        const sx = std.math.sin(rad_x);
        const rad_y = rot_deg.y * std.math.pi / 180.0;
        const cy = std.math.cos(rad_y);
        const sy = std.math.sin(rad_y);
        const rad_z = rot_deg.z * std.math.pi / 180.0;
        const cz = std.math.cos(rad_z);
        const sz = std.math.sin(rad_z);

        const r00 = cz * cy + 0.0;
        const r10 = sz * cy + 0.0;
        const r20 = -sy + 0.0;
        const r01 = cz * (sy * sx) - sz * cx + 0.0;
        const r11 = sz * (sy * sx) + cz * cx + 0.0;
        const r21 = cy * sx + 0.0;
        const r02 = cz * (sy * cx) + sz * sx + 0.0;
        const r12 = sz * (sy * cx) - cz * sx + 0.0;
        const r22 = cy * cx + 0.0;

        return .{ .m = .{
            r00 * scale_v.x + 0.0, r10 * scale_v.x + 0.0, r20 * scale_v.x + 0.0, 0.0,
            r01 * scale_v.y + 0.0, r11 * scale_v.y + 0.0, r21 * scale_v.y + 0.0, 0.0,
            r02 * scale_v.z + 0.0, r12 * scale_v.z + 0.0, r22 * scale_v.z + 0.0, 0.0,
            pos.x + 0.0,           pos.y + 0.0,           pos.z + 0.0,           1.0,
        } };
    }

    pub fn fromQuatTranslationScale(pos: Vec3, q_in: Quat, scale_v: Vec3) Mat4 {
        const q = q_in.normalize();
        const xx = q.x * q.x;
        const yy = q.y * q.y;
        const zz = q.z * q.z;
        const xy = q.x * q.y;
        const xz = q.x * q.z;
        const yz = q.y * q.z;
        const wx = q.w * q.x;
        const wy = q.w * q.y;
        const wz = q.w * q.z;

        return .{ .m = .{
            (1.0 - 2.0 * (yy + zz)) * scale_v.x,
            (2.0 * (xy + wz)) * scale_v.x,
            (2.0 * (xz - wy)) * scale_v.x,
            0.0,

            (2.0 * (xy - wz)) * scale_v.y,
            (1.0 - 2.0 * (xx + zz)) * scale_v.y,
            (2.0 * (yz + wx)) * scale_v.y,
            0.0,

            (2.0 * (xz + wy)) * scale_v.z,
            (2.0 * (yz - wx)) * scale_v.z,
            (1.0 - 2.0 * (xx + yy)) * scale_v.z,
            0.0,

            pos.x,
            pos.y,
            pos.z,
            1.0,
        } };
    }

    /// Returns a copy of the matrix with the translation components zeroed out
    pub fn removeTranslation(self: Mat4) Mat4 {
        var r = self;
        r.m[12] = 0.0;
        r.m[13] = 0.0;
        r.m[14] = 0.0;
        return r;
    }

    /// Extracts the translation column as a 3D vector
    pub fn getTranslation(self: Mat4) Vec3 {
        return Vec3.new(self.m[12], self.m[13], self.m[14]);
    }

    /// Computes inverse of 4x4 matrix, returns null if singular (det ≈ 0)
    pub fn invert(self: Mat4) ?Mat4 {
        const a = self.m;
        const s0 = a[0] * a[5] - a[4] * a[1];
        const s1 = a[0] * a[6] - a[4] * a[2];
        const s2 = a[0] * a[7] - a[4] * a[3];
        const s3 = a[1] * a[6] - a[5] * a[2];
        const s4 = a[1] * a[7] - a[5] * a[3];
        const s5 = a[2] * a[7] - a[6] * a[3];

        const c5 = a[10] * a[15] - a[14] * a[11];
        const c4 = a[9] * a[15] - a[13] * a[11];
        const c3 = a[9] * a[14] - a[13] * a[10];
        const c2 = a[8] * a[15] - a[12] * a[11];
        const c1 = a[8] * a[14] - a[12] * a[10];
        const c0 = a[8] * a[13] - a[12] * a[9];

        const det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if (@abs(det) < 1e-8) return null;

        const inv_det = 1.0 / det;
        var r: Mat4 = undefined;

        r.m[0] = (a[5] * c5 - a[6] * c4 + a[7] * c3) * inv_det;
        r.m[1] = (-a[1] * c5 + a[2] * c4 - a[3] * c3) * inv_det;
        r.m[2] = (a[13] * s5 - a[14] * s4 + a[15] * s3) * inv_det;
        r.m[3] = (-a[9] * s5 + a[10] * s4 - a[11] * s3) * inv_det;

        r.m[4] = (-a[4] * c5 + a[6] * c2 - a[7] * c1) * inv_det;
        r.m[5] = (a[0] * c5 - a[2] * c2 + a[3] * c1) * inv_det;
        r.m[6] = (-a[12] * s5 + a[14] * s2 - a[15] * s1) * inv_det;
        r.m[7] = (a[8] * s5 - a[10] * s2 + a[11] * s1) * inv_det;

        r.m[8] = (a[4] * c4 - a[5] * c2 + a[7] * c0) * inv_det;
        r.m[9] = (-a[0] * c4 + a[1] * c2 - a[3] * c0) * inv_det;
        r.m[10] = (a[12] * s4 - a[13] * s2 + a[15] * s0) * inv_det;
        r.m[11] = (-a[8] * s4 + a[9] * s2 - a[11] * s0) * inv_det;

        r.m[12] = (-a[4] * c3 + a[5] * c1 - a[6] * c0) * inv_det;
        r.m[13] = (a[0] * c3 - a[1] * c1 + a[2] * c0) * inv_det;
        r.m[14] = (-a[12] * s3 + a[13] * s1 - a[14] * s0) * inv_det;
        r.m[15] = (a[8] * s3 - a[9] * s1 + a[10] * s0) * inv_det;

        return r;
    }

    /// Transforms a 3D point (w = 1.0) and applies perspective division using SIMD
    pub fn transformPoint(self: Mat4, p: Vec3) Vec3 {
        const col0: @Vector(4, f32) = self.m[0..4].*;
        const col1: @Vector(4, f32) = self.m[4..8].*;
        const col2: @Vector(4, f32) = self.m[8..12].*;
        const col3: @Vector(4, f32) = self.m[12..16].*;
        const v: [4]f32 = col0 * @as(@Vector(4, f32), @splat(p.x)) +
            col1 * @as(@Vector(4, f32), @splat(p.y)) +
            col2 * @as(@Vector(4, f32), @splat(p.z)) +
            col3;

        const w = v[3];
        if (w != 0.0 and w != 1.0) {
            const inv_w = 1.0 / w;
            return Vec3.new(v[0] * inv_w, v[1] * inv_w, v[2] * inv_w);
        }
        return Vec3.new(v[0], v[1], v[2]);
    }

    /// Transforms a 3D direction vector (w = 0.0) without translation using SIMD
    pub fn transformDirection(self: Mat4, d: Vec3) Vec3 {
        const col0: @Vector(4, f32) = self.m[0..4].*;
        const col1: @Vector(4, f32) = self.m[4..8].*;
        const col2: @Vector(4, f32) = self.m[8..12].*;
        const v: [4]f32 = col0 * @as(@Vector(4, f32), @splat(d.x)) +
            col1 * @as(@Vector(4, f32), @splat(d.y)) +
            col2 * @as(@Vector(4, f32), @splat(d.z));

        return Vec3.new(v[0], v[1], v[2]).normalize();
    }

    /// Projects a 3D world space point into 2D screen pixel space (top-left is (0,0)).
    /// Returns null if the point is behind the camera.
    pub fn projectPoint(self: Mat4, p: Vec3, screen_w: f32, screen_h: f32) ?Vec2 {
        const col0: @Vector(4, f32) = self.m[0..4].*;
        const col1: @Vector(4, f32) = self.m[4..8].*;
        const col2: @Vector(4, f32) = self.m[8..12].*;
        const col3: @Vector(4, f32) = self.m[12..16].*;
        const v: [4]f32 = col0 * @as(@Vector(4, f32), @splat(p.x)) +
            col1 * @as(@Vector(4, f32), @splat(p.y)) +
            col2 * @as(@Vector(4, f32), @splat(p.z)) +
            col3;

        const w = v[3];
        if (w <= 0.001) return null;

        const inv_w = 1.0 / w;
        const ndc_x = v[0] * inv_w;
        const ndc_y = v[1] * inv_w;

        const sx = (ndc_x * 0.5 + 0.5) * screen_w;
        const sy = (1.0 - (ndc_y * 0.5 + 0.5)) * screen_h;
        return Vec2.new(sx, sy);
    }
};

test "Mat4 fromQuatTranslationScale" {
    const pos = Vec3.new(1.0, 2.0, 3.0);
    const scale = Vec3.new(2.0, 0.5, 1.5);
    const rot_deg = Vec3.new(30.0, 45.0, 60.0);
    const q = Quat.fromEulerDeg(rot_deg);

    const m_euler = Mat4.fromRotationTranslationScale(pos, rot_deg, scale);
    const m_quat = Mat4.fromQuatTranslationScale(pos, q, scale);

    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(m_euler.m[i], m_quat.m[i], 1e-4);
    }
}

test "Mat4 fromRotationTranslationScale analytic properties" {
    // Cheap deterministic spot checks for the analytic TRS build:
    // translation round-trip, scaled-orthonormal columns (norm == |scale|,
    // pairwise dots == 0), and unit-scale 3x3 determinant == +1.
    const rots = [_]Vec3{
        Vec3.new(0.0, 0.0, 0.0),
        Vec3.new(30.0, 45.0, 60.0),
        Vec3.new(-90.0, 15.0, 180.0),
        Vec3.new(123.0, -67.0, 11.0),
    };
    const pos = Vec3.new(12.0, -34.0, 56.0);
    const scale = Vec3.new(2.0, 0.5, 1.5);
    for (rots) |rot| {
        const m = Mat4.fromRotationTranslationScale(pos, rot, scale);
        try std.testing.expectEqual(pos, m.getTranslation());

        const c0 = Vec3.new(m.m[0], m.m[1], m.m[2]);
        const c1 = Vec3.new(m.m[4], m.m[5], m.m[6]);
        const c2 = Vec3.new(m.m[8], m.m[9], m.m[10]);
        try std.testing.expectApproxEqAbs(@abs(scale.x), c0.length(), 1e-4);
        try std.testing.expectApproxEqAbs(@abs(scale.y), c1.length(), 1e-4);
        try std.testing.expectApproxEqAbs(@abs(scale.z), c2.length(), 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), c0.dot(c1), 1e-3);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), c0.dot(c2), 1e-3);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), c1.dot(c2), 1e-3);

        const u = Mat4.fromRotationTranslationScale(Vec3.zero, rot, Vec3.one);
        const det = u.m[0] * (u.m[5] * u.m[10] - u.m[9] * u.m[6]) -
            u.m[4] * (u.m[1] * u.m[10] - u.m[9] * u.m[2]) +
            u.m[8] * (u.m[1] * u.m[6] - u.m[5] * u.m[2]);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), det, 1e-5);
    }
}

test "Mat4 identity and multiplication parity" {
    const id = Mat4.identity;
    for (0..4) |col| {
        for (0..4) |row| {
            const expected: f32 = if (col == row) 1.0 else 0.0;
            try std.testing.expectEqual(expected, id.m[col * 4 + row]);
        }
    }

    const t = Mat4.translation(Vec3.new(3.0, -5.0, 7.0));
    const s = Mat4.scaling(Vec3.new(2.0, 0.5, 4.0));

    // ID * T == T
    const mul_id = id.mul(t);
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(t.m[i], mul_id.m[i], 1e-6);
    }

    // mulScalar and mulSimd parity
    const mul_scalar = Mat4.mulScalar(t, s);
    const mul_simd = Mat4.mulSimd(t, s);
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(mul_scalar.m[i], mul_simd.m[i], 1e-6);
    }
}

test "Mat4 translation, scaling and rotation" {
    const pos = Vec3.new(12.0, -34.0, 56.0);
    const t = Mat4.translation(pos);

    try std.testing.expectEqual(pos, t.getTranslation());

    const no_trans = t.removeTranslation();
    try std.testing.expectEqual(Vec3.zero, no_trans.getTranslation());

    // Scaling
    const s = Mat4.scaling(Vec3.new(2.0, 3.0, 4.0));
    const scaled_pt = s.transformPoint(Vec3.new(1.0, 1.0, 1.0));
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), scaled_pt.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), scaled_pt.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), scaled_pt.z, 1e-5);

    // Rotation around Z by 90 deg
    const rz = Mat4.rotationZ(90.0);
    const rotated = rz.transformPoint(Vec3.new(1.0, 0.0, 0.0));
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rotated.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rotated.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rotated.z, 1e-5);
}

test "Mat4 invert recovers identity and handles singular matrix" {
    const pos = Vec3.new(10.0, 20.0, 30.0);
    const scale = Vec3.new(2.0, 3.0, 4.0);
    const rot = Vec3.new(15.0, 30.0, 45.0);
    const m = Mat4.fromRotationTranslationScale(pos, rot, scale);

    const inv = m.invert();
    try std.testing.expect(inv != null);

    const prod = m.mul(inv.?);
    for (0..4) |col| {
        for (0..4) |row| {
            const expected: f32 = if (col == row) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, prod.m[col * 4 + row], 1e-4);
        }
    }

    // Singular matrix (zero det) returns null
    var singular: Mat4 = Mat4.identity;
    singular.m[0] = 0;
    singular.m[5] = 0;
    singular.m[10] = 0;
    singular.m[15] = 0;
    try std.testing.expect(singular.invert() == null);
}

test "Mat4 transformPoint, transformDirection and projectPoint" {
    const t = Mat4.translation(Vec3.new(5.0, 10.0, 15.0));

    // transformPoint applies translation
    const pt = t.transformPoint(Vec3.new(1.0, 2.0, 3.0));
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), pt.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), pt.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), pt.z, 1e-5);

    // transformDirection ignores translation
    const dir = t.transformDirection(Vec3.new(0.0, 10.0, 0.0));
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dir.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dir.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dir.z, 1e-5);

    // projectPoint
    const view = Mat4.lookAt(Vec3.new(0, 0, 10), Vec3.new(0, 0, 0), Vec3.up);
    const proj = Mat4.perspective(60.0, 1.0, 0.1, 100.0);
    const vp = proj.mul(view);

    // Point in front of camera at center maps to center of screen (400, 300 for 800x600)
    const screen_pt = vp.projectPoint(Vec3.new(0, 0, 0), 800, 600);
    try std.testing.expect(screen_pt != null);
    try std.testing.expectApproxEqAbs(@as(f32, 400.0), screen_pt.?.x, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), screen_pt.?.y, 1.0);

    // Point behind camera returns null
    const behind = vp.projectPoint(Vec3.new(0, 0, 20), 800, 600);
    try std.testing.expect(behind == null);
}
