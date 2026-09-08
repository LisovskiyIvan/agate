const std = @import("std");
const Vec3 = @import("vec.zig").Vec3;

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
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, far / (near - far), -1,
            0, 0, (far * near) / (near - far), 0,
        } };
    }

    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) Mat4 {
        const w = right - left;
        const h = top - bottom;
        const d = far - near;
        return .{ .m = .{
            2.0 / w, 0.0, 0.0, 0.0,
            0.0, 2.0 / h, 0.0, 0.0,
            0.0, 0.0, -1.0 / d, 0.0,
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

    pub fn fromRotationTranslationScale(pos: Vec3, rot_deg: Vec3, scale_v: Vec3) Mat4 {
        // T * Rz * Ry * Rx * S
        const t = translation(pos);
        const rx = rotationX(rot_deg.x);
        const ry = rotationY(rot_deg.y);
        const rz = rotationZ(rot_deg.z);
        const s = scaling(scale_v);

        const rot = mul(rz, mul(ry, rx));
        return mul(t, mul(rot, s));
    }
};
