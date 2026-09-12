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
/// empty/missing accessors. CUBICSPLINE keeps the full glTF layout: per key
/// (in-tangent, value, out-tangent), i.e. stride*3 floats per keyframe read
/// from accessor elements ki*3+j, matching the spec's 3x output count.
/// LINEAR/STEP keep the historical single-element read bit-for-bit.
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
    const is_cubic = samp.*.interpolation == c.cgltf_interpolation_type_cubic_spline;
    const per_key: usize = if (is_cubic) stride * 3 else stride;
    const outputs = try allocator.alloc(f32, key_count * per_key);
    errdefer allocator.free(outputs);
    // Morph weights are the one glTF track where a SCALAR output accessor
    // holds `stride` elements per keyframe (one per morph target), so a
    // single read per key with element_size = stride would walk the
    // interleaved per-target sequence and alias targets. Read scalar
    // elements individually in that case; vec2/vec3/vec4 tracks keep the
    // fast per-key read path.
    const scalar_multi = out_acc.*.type == c.cgltf_type_scalar and stride > 1;
    for (0..key_count) |ki| {
        if (is_cubic and scalar_multi) {
            for (0..3 * stride) |j| {
                _ = c.cgltf_accessor_read_float(out_acc, ki * 3 * stride + j, outputs[ki * per_key + j ..][0..1].ptr, 1);
            }
        } else if (is_cubic) {
            for (0..3) |j| {
                const dst = outputs[ki * per_key + j * stride .. ki * per_key + (j + 1) * stride];
                _ = c.cgltf_accessor_read_float(out_acc, ki * 3 + j, dst.ptr, @intCast(stride));
            }
        } else if (scalar_multi) {
            for (0..stride) |j| {
                _ = c.cgltf_accessor_read_float(out_acc, ki * stride + j, outputs[ki * stride + j ..][0..1].ptr, 1);
            }
        } else {
            _ = c.cgltf_accessor_read_float(out_acc, ki, outputs[ki * stride .. ki * stride + stride].ptr, @intCast(stride));
        }
    }

    const interp: AnimationInterpolation = switch (samp.*.interpolation) {
        c.cgltf_interpolation_type_step => .step,
        c.cgltf_interpolation_type_cubic_spline => .cubic_spline,
        else => .linear,
    };
    return .{ .timestamps = timestamps, .outputs = outputs, .interpolation = interp };
}

test "readSampler reads interleaved morph weights per target" {
    const allocator = std.testing.allocator;
    var times = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    var weights = [_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25 };

    var time_buf = std.mem.zeroes(c.cgltf_buffer);
    time_buf.data = @ptrCast(&times);
    time_buf.size = @sizeOf(@TypeOf(times));
    var time_view = std.mem.zeroes(c.cgltf_buffer_view);
    time_view.buffer = &time_buf;
    time_view.size = time_buf.size;
    var in_acc = std.mem.zeroes(c.cgltf_accessor);
    in_acc.buffer_view = &time_view;
    in_acc.type = c.cgltf_type_scalar;
    in_acc.component_type = c.cgltf_component_type_r_32f;
    in_acc.count = times.len;

    var weight_buf = std.mem.zeroes(c.cgltf_buffer);
    weight_buf.data = @ptrCast(&weights);
    weight_buf.size = @sizeOf(@TypeOf(weights));
    var weight_view = std.mem.zeroes(c.cgltf_buffer_view);
    weight_view.buffer = &weight_buf;
    weight_view.size = weight_buf.size;
    var out_acc = std.mem.zeroes(c.cgltf_accessor);
    out_acc.buffer_view = &weight_view;
    out_acc.type = c.cgltf_type_scalar;
    out_acc.component_type = c.cgltf_component_type_r_32f;
    out_acc.count = weights.len;

    var samp = std.mem.zeroes(c.cgltf_animation_sampler);
    samp.input = &in_acc;
    samp.output = &out_acc;
    samp.interpolation = c.cgltf_interpolation_type_linear;

    // stride = morph target count: one SCALAR element per target per key.
    const data = (try readSampler(allocator, &samp, 2)).?;
    defer allocator.free(data.timestamps);
    defer allocator.free(data.outputs);
    try std.testing.expectEqual(@as(usize, 4), data.timestamps.len);
    try std.testing.expectEqualSlices(f32, &times, data.timestamps);
    try std.testing.expectEqualSlices(f32, &weights, data.outputs);
}
