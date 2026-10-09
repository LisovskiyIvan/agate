const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const vat = @import("vat.zig");
const VatData = vat.VatData;
const VatBaker = vat.VatBaker;
const VatPlayer = vat.VatPlayer;

test "VAT texel coordination grid and strip layout" {
    const ally = std.testing.allocator;
    const pos = try ally.alloc(f32, 100 * 4);
    defer ally.free(pos);
    @memset(pos, 0);

    const grid_vat = VatData{
        .allocator = ally,
        .vertex_count = 10,
        .frame_count = 5,
        .layout = .grid,
        .texture_width = 10,
        .texture_height = 5,
        .positions = pos,
    };
    const c0 = grid_vat.texelCoord(3, 2);
    try std.testing.expectEqual(@as(u32, 3), c0.x);
    try std.testing.expectEqual(@as(u32, 2), c0.y);
    try std.testing.expectEqual(@as(usize, (2 * 10 + 3) * 4), grid_vat.texelFloatIndex(3, 2));

    const strip_vat = VatData{
        .allocator = ally,
        .vertex_count = 100,
        .frame_count = 10,
        .layout = .strip,
        .texture_width = 40,
        .texture_height = 25,
        .positions = pos,
    };
    // Frame 1, vertex 50 -> flat index 150 -> x = 150 % 40 = 30, y = 150 / 40 = 3
    const c1 = strip_vat.texelCoord(50, 1);
    try std.testing.expectEqual(@as(u32, 30), c1.x);
    try std.testing.expectEqual(@as(u32, 3), c1.y);
}

fn testWaveGenerator(_: void, frame: u32, _: f32, vert_id: u32, out_pos: *Vec3, out_nrm: ?*Vec3) void {
    const v: f32 = @floatFromInt(vert_id);
    const f: f32 = @floatFromInt(frame);
    // Sine wave animated along Y
    out_pos.* = Vec3.new(v, @sin(v * 0.5 + f * 0.2), 0.0);
    if (out_nrm) |n| {
        n.* = Vec3.up;
    }
}

test "VAT bakeProcedural, samplePosition and bounds" {
    const ally = std.testing.allocator;
    var baked = try VatBaker.bakeProcedural(ally, 8, 10, 30.0, true, {}, testWaveGenerator);
    defer baked.deinit();

    try std.testing.expectEqual(@as(u32, 8), baked.vertex_count);
    try std.testing.expectEqual(@as(u32, 10), baked.frame_count);
    try std.testing.expect(baked.normals != null);

    // Vertex 2 at frame 0: y = sin(2 * 0.5 + 0) = sin(1.0)
    const p0 = baked.samplePosition(2, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), p0.x, 1e-5);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), p0.y, 1e-5);

    // Sub-frame interpolated sampling at frame 0.5
    const p_mid = baked.samplePositionInterpolated(2, 0.5);
    const p1 = baked.samplePosition(2, 1);
    const expected_mid_y = (@sin(@as(f32, 1.0)) + p1.y) * 0.5;
    try std.testing.expectApproxEqAbs(expected_mid_y, p_mid.y, 1e-4);

    // Bounds must encompass min and max of all vertices across all frames
    try std.testing.expect(baked.bounding_box.min.x <= 0.0);
    try std.testing.expect(baked.bounding_box.max.x >= 7.0);
    try std.testing.expect(baked.bounding_box.min.y >= -1.0);
    try std.testing.expect(baked.bounding_box.max.y <= 1.0);
}

test "VatPlayer playback and uniform parameter calculation" {
    const ally = std.testing.allocator;
    var baked = try VatBaker.bakeProcedural(ally, 4, 10, 30.0, false, {}, testWaveGenerator);
    defer baked.deinit();

    var player = VatPlayer.init(&baked);
    player.speed = 1.0;

    // At t=0: frame 0 -> 1, frac 0.0
    var params = player.getSampleParams();
    try std.testing.expectEqual(@as(u32, 0), params.frame0);
    try std.testing.expectEqual(@as(u32, 1), params.frame1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), params.lerp_frac, 1e-5);

    // Advance by half a frame: dt = 0.5 / 30.0 seconds
    player.update(0.5 / 30.0);
    params = player.getSampleParams();
    try std.testing.expectEqual(@as(u32, 0), params.frame0);
    try std.testing.expectEqual(@as(u32, 1), params.frame1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), params.lerp_frac, 1e-4);

    // Advance past end of animation: should loop back
    player.seek(10.0 / 30.0 + 0.25 / 30.0); // 10 frames = full cycle + 0.25 frame
    params = player.getSampleParams();
    try std.testing.expectEqual(@as(u32, 0), params.frame0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), params.lerp_frac, 1e-4);
}
