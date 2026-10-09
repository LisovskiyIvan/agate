const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Camera = @import("../camera.zig").Camera;
const shadow_system = @import("shadow_system.zig");
const ShadowSystem = shadow_system.ShadowSystem;
const scene_cascades = @import("cascades.zig");
const scene_uniforms = @import("uniforms.zig");
const Mesh = @import("../mesh.zig").Mesh;

fn matEq(a: Mat4, b: Mat4) bool {
    for (a.m, b.m) |x, y| {
        if (x != y) return false;
    }
    return true;
}

test "computeCascades stores matrices and follows splits" {
    const camera: Camera = .{ .arc_rotate = .{ .alpha = 0.5, .beta = 1.0, .radius = 8.0 } };
    var shadows: ShadowSystem = undefined;
    shadows = .{ .pass = undefined };
    shadows.splits = .{ 10.0, 26.0, 65.0, 150.0 };

    const sun = math.Vec3.new(0.5, 1.0, 0.3).normalize();
    const cascades = shadows.computeCascades(camera, 16.0 / 9.0, sun);
    try std.testing.expectEqual(cascades, shadows.matrices);
    for (cascades) |m| {
        try std.testing.expectEqual(@as(f32, 0.0), m.m[3]);
        try std.testing.expectEqual(@as(f32, 0.0), m.m[7]);
        try std.testing.expectEqual(@as(f32, 0.0), m.m[11]);
        try std.testing.expectEqual(@as(f32, 1.0), m.m[15]);
    }

    const cascades_b = scene_cascades.computeCascades(camera, 16.0 / 9.0, sun, .{ 10.0, 26.0, 65.0, 180.0 });
    try std.testing.expect(matEq(cascades[0], cascades_b[0]));
    try std.testing.expect(matEq(cascades[1], cascades_b[1]));
    try std.testing.expect(matEq(cascades[2], cascades_b[2]));
    try std.testing.expect(!matEq(cascades[3], cascades_b[3]));
}

test "frameUniforms packs shared lighting state verbatim" {
    const uniforms = scene_uniforms;
    const FrameContext = uniforms.FrameContext;

    var shadows: ShadowSystem = .{ .pass = undefined };
    shadows.bias = 0.0012;
    shadows.intensity = 0.75;
    shadows.normal_bias = 0.02;
    shadows.softness = 1.5;
    shadows.debug_cascades = true;
    shadows.splits = .{ 10.0, 26.0, 65.0, 150.0 };
    shadows.pcss_enabled = false;
    shadows.pcss_light_size = 0.02;
    shadows.pcss_blocker_radius = 0.01;

    var mesh_obj: Mesh = undefined;
    mesh_obj.receive_shadows = true;

    const cascades = [_]Mat4{ Mat4.identity, Mat4.identity, Mat4.identity, Mat4.identity };
    const ctx = FrameContext{
        .view_proj = Mat4.identity,
        .eye = math.Vec3.new(1.0, 2.0, 3.0),
        .sun_dir = math.Vec3.new(0.5, 1.0, 0.3),
        .sun_color = Color3.new(1.0, 0.9, 0.8),
        .sun_intensity = 2.0,
        .directional_dir = [_][4]f32{.{ 0.5, 1.0, 0.3, 0.0 }} ++ [_][4]f32{.{ 0, 0, 0, 0 }} ** 3,
        .directional_color_int = [_][4]f32{.{ 1.0, 0.9, 0.8, 2.0 }} ++ [_][4]f32{.{ 0, 0, 0, 0 }} ** 3,
        .cascades = cascades,
        .light_counts = .{ 1.0, 0.0, 0.0, 0.0 },
        .point_pos_range = [_][4]f32{.{ 1, 2, 3, 10 }} ** 4,
        .point_color_int = [_][4]f32{.{ 1, 1, 1, 1 }} ** 4,
        .spot_pos_range = [_][4]f32{.{ 4, 5, 6, 20 }} ** 2,
        .spot_dir_inner = [_][4]f32{.{ 0, -1, 0, 0.9 }} ** 2,
        .spot_color_outer = [_][4]f32{.{ 1, 1, 1, 0.7 }} ** 2,
        .spot_intensity = [_][4]f32{.{ 3, 0, 0, 0 }} ** 2,
        .spot_view_proj = [_]Mat4{ Mat4.identity, Mat4.identity },
        .spot_shadow_params = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .point_view_proj = [_]Mat4{Mat4.identity} ** 12,
        .point_shadow_params = [_][4]f32{.{ 0, 0, 0, 0 }} ** 4,
        .area_center_int = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .area_right = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .area_up = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .area_color = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .clustered_params = .{ 0, 0, 0, 0 },
        .clustered_viewport = .{ 0, 0, 64, 0 },
    };

    var state = shadows.uniformState(Color3.new(0.2, 0.25, 0.3));
    state.mesh_receive_shadows = mesh_obj.receive_shadows;
    const f = uniforms.buildFrameUniforms(state, &ctx);
    try std.testing.expectEqual([4]f32{ 1.0, 2.0, 3.0, 4.0 }, f.eye_pos);
    try std.testing.expectEqual([4]f32{ 0.5, 1.0, 0.3, 2048.0 }, f.light_dir);
    try std.testing.expectEqual([4]f32{ 1.0, 0.9, 0.8, 2.0 }, f.light_color);
    try std.testing.expectEqual([4]f32{ 0.2, 0.25, 0.3, 1.0 }, f.ambient_color);
    try std.testing.expectEqual([4]f32{ 0.0012, 0.75, 0.02, 1.5 }, f.shadow_params);
    try std.testing.expectEqual([4]f32{ 10.0, 26.0, 65.0, 150.0 }, f.shadow_splits);
    try std.testing.expectEqual(cascades, f.cascade_view_proj);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.02, 0.01 }, f.cascade_debug);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0005, 0.01 }, f.light_counts);
    try std.testing.expectEqual(ctx.point_pos_range, f.point_pos_range);
    try std.testing.expectEqual(ctx.spot_intensity, f.spot_intensity);
    try std.testing.expectEqual(ctx.point_shadow_params, f.point_shadow_params);
    try std.testing.expectEqual(ctx.point_view_proj, f.point_view_proj);
    try std.testing.expectEqual(ctx.directional_dir, f.directional_dir);
    try std.testing.expectEqual(ctx.directional_color_int, f.directional_color_int);
    try std.testing.expectEqual(ctx.area_center_int, f.area_center_int);
    try std.testing.expectEqual(ctx.area_right, f.area_right);
    try std.testing.expectEqual(ctx.area_up, f.area_up);
    try std.testing.expectEqual(ctx.area_color, f.area_color);
    try std.testing.expectEqual(ctx.clustered_params, f.clustered_params);
    try std.testing.expectEqual(ctx.clustered_viewport, f.clustered_viewport);

    shadows.enabled = false;
    const off_state = shadows.uniformState(Color3.new(0.2, 0.25, 0.3));
    const off = uniforms.buildFrameUniforms(off_state, &ctx);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.02, 1.5 }, off.shadow_params);
    shadows.enabled = true;
    mesh_obj.receive_shadows = false;
    var skipped_state = shadows.uniformState(Color3.new(0.2, 0.25, 0.3));
    skipped_state.mesh_receive_shadows = mesh_obj.receive_shadows;
    const skipped = uniforms.buildFrameUniforms(skipped_state, &ctx);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.02, 1.5 }, skipped.shadow_params);
}

test "uniformState packs shadow params verbatim" {
    const shadows: ShadowSystem = .{ .pass = undefined };
    const state = shadows.uniformState(Color3.new(0.2, 0.25, 0.3));
    try std.testing.expect(state.enable_shadows);
    try std.testing.expectEqual(@as(f32, 0.0012), state.bias);
    try std.testing.expectEqual(@as(f32, 0.75), state.intensity);
    try std.testing.expectEqual(@as(f32, 0.02), state.normal_bias);
    try std.testing.expectEqual(@as(f32, 1.5), state.softness);
    try std.testing.expectEqual([4]f32{ 10.0, 26.0, 65.0, 150.0 }, state.splits);
    try std.testing.expectEqual(Color3.new(0.2, 0.25, 0.3), state.ground_color);
    try std.testing.expect(!state.pcss_enabled);
    try std.testing.expectEqual(@as(f32, 0.02), state.pcss_light_size);

    const tuned: ShadowSystem = .{
        .pass = undefined,
        .enabled = false,
        .debug_cascades = true,
        .pcss_enabled = true,
    };
    const off = tuned.uniformState(Color3.white);
    try std.testing.expect(!off.enable_shadows);
    try std.testing.expect(off.debug_cascades);
    try std.testing.expect(off.pcss_enabled);
}
