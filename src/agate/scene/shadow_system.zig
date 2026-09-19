const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Color3 = math.Color3;

const Camera = @import("../camera.zig").Camera;
const passes = @import("../passes/mod.zig");
const scene_cascades = @import("cascades.zig");
const scene_uniforms = @import("uniforms.zig");

/// Shadow mapping state for the CSM sun (16-sample Poisson PCF by default,
/// optional PCSS) plus the GPU shadow depth pass. Cascaded matrices are
/// recomputed per frame from the resolved sun direction; the fragment
/// uniform side is exposed via uniformState() so the draw path never pokes
/// at individual bias fields.
pub const ShadowSystem = struct {
    enabled: bool = true,
    bias: f32 = 0.0012,
    normal_bias: f32 = 0.02,
    intensity: f32 = 0.75,
    softness: f32 = 1.5,
    debug_cascades: bool = false,
    // PCSS (percentage-closer soft shadows) for the CSM sun. Disabled keeps
    // the legacy 16x Poisson PCF path bit-identical.
    pcss_enabled: bool = false,
    pcss_light_size: f32 = 0.02,
    pcss_blocker_radius: f32 = 0.01,
    splits: [4]f32 = .{ 10.0, 26.0, 65.0, 150.0 },
    // Last computed cascade view-projections (exposed for debugging/tooling).
    matrices: [4]Mat4 = [_]Mat4{Mat4.identity} ** 4,

    // GPU shadow depth pass (CSM atlas + spot atlas).
    pass: passes.ShadowPass,

    pub fn init(allocator: std.mem.Allocator) ShadowSystem {
        return .{ .pass = passes.ShadowPass.init(allocator) };
    }

    pub fn deinit(self: *ShadowSystem) void {
        self.pass.deinit();
    }

    /// Computes the 4 cascade view-projections for `norm_light_dir` (a
    /// resolveSunDirection output) and caches them in `matrices`.
    pub fn computeCascades(self: *ShadowSystem, camera: Camera, aspect: f32, norm_light_dir: math.Vec3) [4]Mat4 {
        self.matrices = scene_cascades.computeCascades(camera, aspect, norm_light_dir, self.splits);
        return self.matrices;
    }

    /// Scene-level fragment uniform inputs (without the per-mesh
    /// receive_shadows flag, which the draw path patches per item). PCSS
    /// min/max penumbra stay at the ShadowState defaults, matching the
    /// legacy Scene.frameUniforms literals.
    pub fn uniformState(self: *const ShadowSystem, ground_color: Color3) scene_uniforms.ShadowState {
        return .{
            .ground_color = ground_color,
            .enable_shadows = self.enabled,
            .mesh_receive_shadows = true,
            .bias = self.bias,
            .intensity = self.intensity,
            .normal_bias = self.normal_bias,
            .softness = self.softness,
            .debug_cascades = self.debug_cascades,
            .splits = self.splits,
            .pcss_enabled = self.pcss_enabled,
            .pcss_light_size = self.pcss_light_size,
            .pcss_blocker_radius = self.pcss_blocker_radius,
        };
    }
};

test "computeCascades stores matrices and follows splits" {
    const camera: Camera = .{ .arc_rotate = .{ .alpha = 0.5, .beta = 1.0, .radius = 8.0 } };
    var shadows: ShadowSystem = undefined;
    // CPU-only: the GPU pass is never dereferenced here.
    shadows = .{ .pass = undefined };
    shadows.splits = .{ 10.0, 26.0, 65.0, 150.0 };

    const sun = math.Vec3.new(0.5, 1.0, 0.3).normalize();
    const cascades = shadows.computeCascades(camera, 16.0 / 9.0, sun);
    try std.testing.expectEqual(cascades, shadows.matrices);
    // Each cascade is a proper affine matrix (bottom row 0,0,0,1 in
    // column-major storage).
    for (cascades) |m| {
        try std.testing.expectEqual(@as(f32, 0.0), m.m[3]);
        try std.testing.expectEqual(@as(f32, 0.0), m.m[7]);
        try std.testing.expectEqual(@as(f32, 0.0), m.m[11]);
        try std.testing.expectEqual(@as(f32, 1.0), m.m[15]);
    }

    // Split distances flow straight into the solver. The near plane of each
    // cascade chains from the previous split, so only the LAST split can be
    // changed without shifting the earlier cascades: moving it changes just
    // the far cascade.
    const cascades_b = scene_cascades.computeCascades(camera, 16.0 / 9.0, sun, .{ 10.0, 26.0, 65.0, 180.0 });
    try std.testing.expect(matEq(cascades[0], cascades_b[0]));
    try std.testing.expect(matEq(cascades[1], cascades_b[1]));
    try std.testing.expect(matEq(cascades[2], cascades_b[2]));
    try std.testing.expect(!matEq(cascades[3], cascades_b[3]));
}

fn matEq(a: Mat4, b: Mat4) bool {
    for (a.m, b.m) |x, y| {
        if (x != y) return false;
    }
    return true;
}

// Ported from the legacy Scene inline suite: the packed fragment uniforms
// must stay bit-identical to the old per-shader literals.
test "frameUniforms packs shared lighting state verbatim" {
    const uniforms = scene_uniforms;
    const FrameContext = uniforms.FrameContext;
    const Mesh = @import("../mesh.zig").Mesh;

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
    // Point shadow lanes ride through verbatim; zeroed params (the default)
    // keep the shader shadow path gated off.
    try std.testing.expectEqual(ctx.point_shadow_params, f.point_shadow_params);
    try std.testing.expectEqual(ctx.point_view_proj, f.point_view_proj);

    // Disabled shadows (globally or per-mesh) zero the bias/intensity lanes
    // but keep normal bias and softness, exactly like the legacy literals.
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
    // PCSS lanes ride the defaults while disabled.
    try std.testing.expect(!state.pcss_enabled);
    try std.testing.expectEqual(@as(f32, 0.02), state.pcss_light_size);

    // Flipping the flags flows through unchanged.
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
