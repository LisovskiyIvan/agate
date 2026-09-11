const std = @import("std");

const c = @import("../c.zig").c;
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Mat4 = math.Mat4;
const AnimationInterpolation = @import("../animation/animation.zig").AnimationInterpolation;

/// Converts a glTF node pointer into its node index, or null for foreign pointers.
pub fn gltfNodeIndex(gltf: *c.cgltf_data, node: *c.cgltf_node) ?usize {
    if (gltf.nodes_count == 0) return null;
    const base = @intFromPtr(&gltf.nodes[0]);
    const ptr = @intFromPtr(node);
    if (ptr < base) return null;
    const idx = (ptr - base) / @sizeOf(c.cgltf_node);
    if (idx >= gltf.nodes_count) return null;
    return idx;
}

/// Rotation matrix columns (orthonormal basis) -> unit quaternion.
pub fn quatFromBasis(x: Vec3, y: Vec3, z: Vec3) Quat {
    const trace = x.x + y.y + z.z;
    if (trace > 0.0) {
        const s = 0.5 / @sqrt(trace + 1.0);
        return (Quat{
            .x = (y.z - z.y) * s,
            .y = (z.x - x.z) * s,
            .z = (x.y - y.x) * s,
            .w = 0.25 / s,
        }).normalize();
    }
    if (x.x > y.y and x.x > z.z) {
        const s = 2.0 * @sqrt(1.0 + x.x - y.y - z.z);
        return (Quat{
            .x = 0.25 * s,
            .y = (x.y + y.x) / s,
            .z = (x.z + z.x) / s,
            .w = (y.z - z.y) / s,
        }).normalize();
    }
    if (y.y > z.z) {
        const s = 2.0 * @sqrt(1.0 + y.y - x.x - z.z);
        return (Quat{
            .x = (x.y + y.x) / s,
            .y = 0.25 * s,
            .z = (y.z + z.y) / s,
            .w = (z.x - x.z) / s,
        }).normalize();
    }
    const s = 2.0 * @sqrt(1.0 + z.z - x.x - y.y);
    return (Quat{
        .x = (x.z + z.x) / s,
        .y = (y.z + z.y) / s,
        .z = 0.25 * s,
        .w = (x.y - y.x) / s,
    }).normalize();
}

/// Reads a node's local TRS. Prefers the explicit TRS fields and
/// decomposes the raw matrix when the node uses has_matrix instead.
pub fn nodeLocalTRS(node: *const c.cgltf_node) struct { pos: Vec3, rot: Quat, scale: Vec3 } {
    var pos = Vec3.zero;
    var rot = Quat.identity;
    var scale = Vec3.one;
    if (node.has_translation != 0) {
        pos = Vec3.new(node.translation[0], node.translation[1], node.translation[2]);
    }
    if (node.has_rotation != 0) {
        rot = (Quat{
            .x = node.rotation[0],
            .y = node.rotation[1],
            .z = node.rotation[2],
            .w = node.rotation[3],
        }).normalize();
    }
    if (node.has_scale != 0) {
        scale = Vec3.new(node.scale[0], node.scale[1], node.scale[2]);
    }
    if (node.has_translation == 0 and node.has_rotation == 0 and node.has_scale == 0 and node.has_matrix != 0) {
        const m = Mat4{ .m = node.matrix };
        pos = m.getTranslation();
        const sx = Vec3.new(m.m[0], m.m[1], m.m[2]).length();
        const sy = Vec3.new(m.m[4], m.m[5], m.m[6]).length();
        const sz = Vec3.new(m.m[8], m.m[9], m.m[10]).length();
        if (sx > 1e-9 and sy > 1e-9 and sz > 1e-9) {
            scale = Vec3.new(sx, sy, sz);
            rot = quatFromBasis(
                Vec3.new(m.m[0] / sx, m.m[1] / sx, m.m[2] / sx),
                Vec3.new(m.m[4] / sy, m.m[5] / sy, m.m[6] / sy),
                Vec3.new(m.m[8] / sz, m.m[9] / sz, m.m[10] / sz),
            );
        }
    }
    return .{ .pos = pos, .rot = rot, .scale = scale };
}

/// World matrix of the node's parent (identity for root nodes).
pub fn nodeParentWorld(node: *const c.cgltf_node) Mat4 {
    if (node.parent) |p| {
        var pw: [16]f32 = undefined;
        c.cgltf_node_transform_world(p, &pw);
        return Mat4{ .m = pw };
    }
    return Mat4.identity;
}

/// Reads one animation sampler into owned buffers. Returns null for
/// empty/missing accessors. CUBICSPLINE is preserved as-is; the runtime
/// treats it as (eased) linear, documented in animation.zig.
pub const SamplerData = struct {
    timestamps: []f32,
    outputs: []f32,
    interpolation: AnimationInterpolation,
};

pub fn readSampler(allocator: std.mem.Allocator, samp: *c.cgltf_animation_sampler, stride: usize) !?SamplerData {
    const in_acc = samp.*.input;
    const out_acc = samp.*.output;
    if (in_acc == null or out_acc == null or in_acc.*.count == 0) return null;

    const key_count = in_acc.*.count;
    const timestamps = try allocator.alloc(f32, key_count);
    errdefer allocator.free(timestamps);
    for (0..key_count) |ki| {
        _ = c.cgltf_accessor_read_float(in_acc, ki, &timestamps[ki], 1);
    }
    const outputs = try allocator.alloc(f32, key_count * stride);
    errdefer allocator.free(outputs);
    for (0..key_count) |ki| {
        _ = c.cgltf_accessor_read_float(out_acc, ki, outputs[ki * stride .. ki * stride + stride].ptr, @intCast(stride));
    }

    const interp: AnimationInterpolation = switch (samp.*.interpolation) {
        c.cgltf_interpolation_type_step => .step,
        c.cgltf_interpolation_type_cubic_spline => .cubic_spline,
        else => .linear,
    };
    return .{ .timestamps = timestamps, .outputs = outputs, .interpolation = interp };
}
