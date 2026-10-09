const std = @import("std");
const vec = @import("vec.zig");
const Vec2 = vec.Vec2;
const Vec3 = vec.Vec3;
const quat = @import("quat.zig");
const Quat = quat.Quat;
const mat4 = @import("mat4.zig");
const Mat4 = mat4.Mat4;

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
