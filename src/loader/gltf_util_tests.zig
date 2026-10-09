const std = @import("std");
const c = @import("../c.zig").c;
const gltf_util = @import("gltf_util.zig");
const readSampler = gltf_util.readSampler;

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
    // cgltf fills accessor.stride at parse time; without it every index
    // reads element 0.
    in_acc.stride = @sizeOf(f32);

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
    out_acc.stride = @sizeOf(f32);

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
