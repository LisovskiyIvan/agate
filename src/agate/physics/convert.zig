//! Shared Box3D conversion helpers and numeric constants for the physics
//! modules. Extracted verbatim from `physics.zig`; behavior unchanged.
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const c = @import("../c.zig").c;

// Fixed simulation rate Box3D integrates at; step() accumulates frame dt
// towards it (Fix Your Timestep) instead of feeding variable dt to the solver.
pub const step_h: f32 = 1.0 / 60.0;
// Our damping fields are ~per-1/60-step velocity retention factors;
// Box3D damping is per-second. 60x maps one to the other.
pub const damp_scale: f32 = 60.0;
pub const deg2rad: f32 = std.math.pi / 180.0;
pub const rad2deg: f32 = 180.0 / std.math.pi;
// Thin-shape guard: Box3D hulls dislike degenerate half-extents.
pub const min_half_extent: f32 = 0.01;

pub fn toB3Vec(v: Vec3) c.b3Vec3 {
    return .{ .x = v.x, .y = v.y, .z = v.z };
}

pub fn fromB3Vec(v: c.b3Vec3) Vec3 {
    return Vec3.new(v.x, v.y, v.z);
}

pub fn toB3Pos(v: Vec3) c.b3Pos {
    return .{ .x = @floatCast(v.x), .y = @floatCast(v.y), .z = @floatCast(v.z) };
}

pub fn fromB3Pos(p: c.b3Pos) Vec3 {
    return Vec3.new(@floatCast(p.x), @floatCast(p.y), @floatCast(p.z));
}

pub fn toB3Quat(q: Quat) c.b3Quat {
    return .{ .v = .{ .x = q.x, .y = q.y, .z = q.z }, .s = q.w };
}

pub fn fromB3Quat(q: c.b3Quat) Quat {
    return .{ .x = q.v.x, .y = q.v.y, .z = q.v.z, .w = q.s };
}
