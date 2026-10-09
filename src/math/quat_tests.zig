const std = @import("std");
const vec = @import("vec.zig");
const Vec3 = vec.Vec3;
const quat_mod = @import("quat.zig");
const Quat = quat_mod.Quat;

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

test "Quat slerp and nlerp" {
    const q0 = Quat.identity;
    const q1 = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));

    const slerp_mid = Quat.slerp(q0, q1, 0.5);
    const euler_mid = slerp_mid.toEulerDeg();
    try std.testing.expectApproxEqAbs(@as(f32, 45.0), euler_mid.y, 1e-2);

    const nlerp_mid = Quat.nlerp(q0, q1, 0.5);
    const euler_nlerp = nlerp_mid.toEulerDeg();
    try std.testing.expectApproxEqAbs(@as(f32, 45.0), euler_nlerp.y, 1e-2);
}
