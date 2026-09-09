const std = @import("std");
const Vec3 = @import("vec.zig").Vec3;

/// Unit quaternion (x, y, z, w). Matches the R = Rz * Ry * Rx Euler order used
/// by Mat4.fromRotationTranslationScale, so Mesh.rotation (degrees) round-trips.
pub const Quat = struct {
    x: f32 = 0.0,
    y: f32 = 0.0,
    z: f32 = 0.0,
    w: f32 = 1.0,

    pub const identity = Quat{};

    pub fn normalize(q: Quat) Quat {
        const len = @sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        if (len < 1e-9) return .identity;
        return .{ .x = q.x / len, .y = q.y / len, .z = q.z / len, .w = q.w / len };
    }

    /// Hamilton product: applies b first, then a (column-vector convention).
    pub fn mul(a: Quat, b: Quat) Quat {
        return .{
            .x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            .y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            .z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            .w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        };
    }

    /// Euler degrees -> quaternion for R = Rz(z) * Ry(y) * Rx(x).
    pub fn fromEulerDeg(e: Vec3) Quat {
        const hx = e.x * (std.math.pi / 360.0);
        const hy = e.y * (std.math.pi / 360.0);
        const hz = e.z * (std.math.pi / 360.0);
        const qx = Quat{ .x = @sin(hx), .w = @cos(hx) };
        const qy = Quat{ .y = @sin(hy), .w = @cos(hy) };
        const qz = Quat{ .z = @sin(hz), .w = @cos(hz) };
        return Quat.mul(qz, Quat.mul(qy, qx)).normalize();
    }

    /// Quaternion -> Euler degrees for R = Rz(z) * Ry(y) * Rx(x).
    /// Gimbal pole (|pitch| ~ 90deg): rolls into yaw, keeps rendering stable.
    pub fn toEulerDeg(q: Quat) Vec3 {
        const n = q.normalize();
        const x = n.x;
        const y = n.y;
        const z = n.z;
        const w = n.w;

        // R20 = 2(xz - wy) = -sin(pitch)
        const r20 = 2.0 * (x * z - w * y);
        const pitch = std.math.asin(std.math.clamp(-r20, -1.0, 1.0));

        var roll: f32 = 0.0;
        var yaw: f32 = 0.0;
        if (@abs(r20) < 0.9999) {
            // roll = atan2(R21, R22), yaw = atan2(R10, R00)
            roll = std.math.atan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
            yaw = std.math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z));
        } else {
            // Pole: with roll forced to 0, yaw = atan2(-R01, R11).
            yaw = std.math.atan2(-2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z));
        }

        const k = 180.0 / std.math.pi;
        return Vec3.new(roll * k, pitch * k, yaw * k);
    }

    pub fn conjugate(q: Quat) Quat {
        return .{ .x = -q.x, .y = -q.y, .z = -q.z, .w = q.w };
    }

    /// Rotates a 3D vector by this unit quaternion using the Rodrigues formula:
    /// v' = v + 2*w*(q_v x v) + 2*(q_v x (q_v x v))
    pub fn rotateVec(q: Quat, v: Vec3) Vec3 {
        const qv = Vec3.new(q.x, q.y, q.z);
        const t = qv.cross(v).scale(2.0);
        return v.add(t.scale(q.w)).add(qv.cross(t));
    }
};

test "Quat rotateVec and conjugate" {
    const q = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));
    const v = Vec3.new(1.0, 0.0, 0.0);
    const rot = q.rotateVec(v);
    // Rotating (1,0,0) by +90 deg around Y: in right-handed convention, maps to (0, 0, -1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rot.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rot.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), rot.z, 1e-4);

    // Inverse rotation restores original vector
    const inv_rot = q.conjugate().rotateVec(rot);
    try std.testing.expectApproxEqAbs(v.x, inv_rot.x, 1e-4);
    try std.testing.expectApproxEqAbs(v.y, inv_rot.y, 1e-4);
    try std.testing.expectApproxEqAbs(v.z, inv_rot.z, 1e-4);
}
