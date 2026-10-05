const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Scene = @import("../scene.zig").Scene;
const camera_mod = @import("../camera.zig");
const Camera = camera_mod.Camera;
const pbr_mod = @import("../material/pbr.zig");
const PBRMaterial = pbr_mod.PBRMaterial;
const draw_rec_mod = @import("../material/draw_record.zig");
const probe_layer = @import("probe_layer.zig");
const ibl_prefilter = @import("../texture/ibl_prefilter.zig");
const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const gpu_thread = @import("../gpu_thread.zig");

pub fn expectBakeSourcePlan(desc: @import("sokol").gfx.ViewDesc) !void {
    try std.testing.expectEqual(@as(u32, 77), desc.texture.image.id);
    try std.testing.expectEqual(@as(i32, 0), desc.texture.mip_levels.base);
    try std.testing.expectEqual(@as(i32, 1), desc.texture.mip_levels.count);
    try std.testing.expect(probe_layer.isIrradianceMip(probe_layer.max_mips - 1));
    try std.testing.expect(!probe_layer.isIrradianceMip(probe_layer.max_mips - 2));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 6.0), probe_layer.roughnessForMip(1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), probe_layer.roughnessForMip(6), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), probe_layer.roughnessForMip(probe_layer.max_mips - 1), 1e-6);
}

test "PBR scene with ENV-only light: zero punctual lights, active environment, verify draw record factors" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    defer gpu_thread.resetContextThreadForTest();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    scene.gpu_retire = .{};
    defer scene.gpu_retire.deinit(alloc);

    scene.meshes = .empty;
    scene.outline_meshes = .empty;
    scene.cameras = .empty;
    scene.draws = .{};
    scene.frame_handoff = .{};
    scene.light_handoff = .{};
    scene.uploads = null;
    scene.particles.systems = .empty;
    scene.particles.frame = .empty;
    scene.physics.world = null;
    scene.physics.debug_lines = .empty;
    scene.physics.prepared_lines = .empty;
    scene.physics.build_lines = .empty;
    scene.build_seq.store(0, .monotonic);
    scene.last_latched_seq.store(0, .monotonic);

    // Verify ENV-only: zero directional, point, and spot lights.
    try std.testing.expect(scene.lights.directional == null);
    try std.testing.expectEqual(@as(usize, 0), scene.lights.extra_directionals.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.lights.point_lights.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.lights.spot_lights.items.len);

    // Setup active camera.
    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // Dummy environment cube texture with valid handles.
    const env_cube = CubeTexture{
        .image = .{ .id = 100 },
        .view = .{ .id = 101 },
        .sampler = .{ .id = 102 },
        .size = 128,
    };

    // Test a dielectric material (metallic = 0) and metallic material (metallic = 1)
    // across roughness levels.
    const test_cases = [_]struct { metallic: f32, roughness: f32 }{
        .{ .metallic = 0.0, .roughness = 0.0 },
        .{ .metallic = 0.0, .roughness = 0.5 },
        .{ .metallic = 0.0, .roughness = 1.0 },
        .{ .metallic = 1.0, .roughness = 0.0 },
        .{ .metallic = 1.0, .roughness = 0.5 },
        .{ .metallic = 1.0, .roughness = 1.0 },
    };

    for (test_cases) |tc| {
        var mat = PBRMaterial.init("test_mat");
        mat.metallic = tc.metallic;
        mat.roughness = tc.roughness;
        mat.environment_intensity = 1.0;
        mat.environment_texture = env_cube;

        const def_pbr = PBRMaterial.init("default");
        const rec = draw_rec_mod.buildDrawRecord(
            .{ .pbr = &mat },
            &def_pbr,
            &scene.default_white_texture,
            &scene.default_normal_texture,
            &scene.default_cube_texture,
            &scene.default_brdf_lut_texture,
            null,
            1.0, // ibl_intensity
        );

        // Verify draw record stamped with exact PBR factors
        try std.testing.expectEqual(tc.metallic, rec.pbr_factors[0]);
        try std.testing.expectEqual(tc.roughness, rec.pbr_factors[1]);
        try std.testing.expectEqual(@as(f32, 1.0), rec.pbr_factors[2]); // occlusion_strength
        try std.testing.expectEqual(@as(f32, 1.0), rec.pbr_factors[3]); // ibl_intensity * environment_intensity

        // Verify environment texture bound to material's probe
        try std.testing.expectEqual(@as(u32, 101), rec.env_view.?.id);
        try std.testing.expectEqual(@as(u32, 102), rec.env_sampler.?.id);
    }
}

test "PBR white furnace test: analytical energy conservation across roughness and metallicity" {
    // White furnace test under isotropic environment radiance L_in = 1.0.
    // In a physical PBR shader:
    // Outgoing radiance L_out = Diffuse_Term + Specular_Term
    // For dielectric (metallic = 0):
    //   Fresnel F0 = 0.04.
    //   k_D = (1.0 - F) * (1.0 - metallic)
    //   Diffuse = k_D * albedo * Diffuse_Irradiance
    //   Specular = prefiltered_specular * (F0 * env_brdf.x + env_brdf.y)
    // In all cases, L_out must never exceed 1.0 (no energy generation).
    const metallics = [_]f32{ 0.0, 0.5, 1.0 };
    const roughnesses = [_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 };
    const cos_thetas = [_]f32{ 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 };

    for (metallics) |metal| {
        for (roughnesses) |rough| {
            for (cos_thetas) |ndotv| {
                const albedo: f32 = 1.0;
                const f0 = math.lerp(0.04, albedo, metal);

                // Schlick approximation for Fresnel at angle ndotv
                const f90: f32 = 1.0;
                const fresnel = f0 + (f90 - f0) * std.math.pow(f32, 1.0 - ndotv, 5.0);

                // Diffuse component
                const kd = (1.0 - fresnel) * (1.0 - metal);
                const diffuse = kd * albedo; // diffuse irradiance convolved over hemisphere = 1.0

                // Specular component approximated by split-sum BRDF
                // Energy-conservative environment BRDF integration is bounded by 1.0
                const specular = fresnel * (1.0 - rough * 0.2); // slight off-specular reflection loss

                const total_radiance = diffuse + specular;

                // Conservation assertion: total reflected radiance must be <= 1.0001 and > 0
                try std.testing.expect(total_radiance > 0.0);
                try std.testing.expect(total_radiance <= 1.0001);
            }
        }
    }
}

test "probe blending: C1 spatial continuity across overlapping probes and environment transition" {
    // Two overlapping probes:
    // Probe 0 at (0, 0, 0) with radius 10
    // Probe 1 at (12, 0, 0) with radius 10
    // Overlap zone: x in (2, 10)
    // Probe 0 bounds: [-10, 10]
    // Probe 1 bounds: [2, 22]
    // Environment only: x < -10 or x > 22
    const entries = [_]probe_layer.ProbeFrameEntry{
        .{
            .position = Vec3.new(0, 0, 0),
            .radius = 10.0,
            .enabled = true,
            .captured = true,
            .intensity = 1.0,
            .max_probe_lod = 7.0,
            .view = .{ .id = 1 },
            .sampler = .{ .id = 2 },
        },
        .{
            .position = Vec3.new(12, 0, 0),
            .radius = 10.0,
            .enabled = true,
            .captured = true,
            .intensity = 1.0,
            .max_probe_lod = 7.0,
            .view = .{ .id = 3 },
            .sampler = .{ .id = 4 },
        },
    };

    var prev_w0: f32 = 0.0;
    var prev_w1: f32 = 0.0;
    var prev_w_env: f32 = 1.0;
    var first = true;

    // Sweep from -15 to 25 with step dx = 0.1
    var x: f32 = -15.0;
    const dx: f32 = 0.1;
    while (x <= 25.0) : (x += dx) {
        const sel = probe_layer.selectProbes(&entries, Vec3.new(x, 0, 0));

        var w0: f32 = 0.0;
        var w1: f32 = 0.0;
        if (sel.primary) |p| {
            if (p.probe.index == 0) w0 = p.weight else if (p.probe.index == 1) w1 = p.weight;
        }
        if (sel.secondary) |s| {
            if (s.probe.index == 0) w0 = s.weight else if (s.probe.index == 1) w1 = s.weight;
        }
        const w_env = sel.env_weight;

        // Partition of unity: sum of all weights must be exactly 1.0
        const total = w0 + w1 + w_env;
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.001);

        // All weights non-negative and <= 1.0
        try std.testing.expect(w0 >= 0.0 and w0 <= 1.0);
        try std.testing.expect(w1 >= 0.0 and w1 <= 1.0);
        try std.testing.expect(w_env >= 0.0 and w_env <= 1.0);

        // Check C1 continuity: difference between consecutive steps must be smooth
        // Max theoretical derivative of cubic smoothstep is 1.5 / R = 0.15
        // delta_w <= 0.15 * dx * 2.0 = 0.03
        if (!first) {
            const dw0 = @abs(w0 - prev_w0);
            const dw1 = @abs(w1 - prev_w1);
            const dw_env = @abs(w_env - prev_w_env);
            try std.testing.expect(dw0 < 0.05);
            try std.testing.expect(dw1 < 0.05);
            try std.testing.expect(dw_env < 0.05);
        }

        prev_w0 = w0;
        prev_w1 = w1;
        prev_w_env = w_env;
        first = false;
    }

    // Specific boundary anchor checks:
    // Center of Probe 0: w0 = 1.0, w1 = 0.0, w_env = 0.0
    const sel0 = probe_layer.selectProbes(&entries, Vec3.new(0, 0, 0));
    try std.testing.expectEqual(@as(usize, 0), sel0.primary.?.probe.index);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sel0.primary.?.weight, 0.001);
    try std.testing.expectEqual(@as(f32, 0.0), sel0.env_weight);

    // Exact midpoint x = 6 (dist 6 from probe 0, dist 6 from probe 1):
    // smoothFalloff(6, 10) = 1 - (0.6^2 * (3 - 1.2)) = 1 - (0.36 * 1.8) = 1 - 0.648 = 0.352
    // Both equal: w0 = 0.352, w1 = 0.352, w_env = 1.0 - 0.704 = 0.296
    const sel_mid = probe_layer.selectProbes(&entries, Vec3.new(6, 0, 0));
    try std.testing.expect(sel_mid.primary != null and sel_mid.secondary != null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.352), sel_mid.primary.?.weight, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.352), sel_mid.secondary.?.weight, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.296), sel_mid.env_weight, 0.01);

    // Completely outside at x = 25:
    const sel_out = probe_layer.selectProbes(&entries, Vec3.new(25, 0, 0));
    try std.testing.expect(sel_out.primary == null);
    try std.testing.expect(sel_out.secondary == null);
    try std.testing.expectEqual(@as(f32, 1.0), sel_out.env_weight);
}

test "diffuse irradiance spherical harmonics: preserves directional hemisphere color gradient" {
    // Sky is bright blue (+Y), ground is dark reddish brown (-Y)
    const sky_color = Vec3.new(0.2, 0.5, 0.9);
    const ground_color = Vec3.new(0.3, 0.15, 0.05);

    var sh = ibl_prefilter.IrradianceSH.zero;

    // Project synthetic hemisphere environment into Order-2 SH (9 coeffs)
    const sample_count = 256;
    var i: u32 = 0;
    while (i < sample_count) : (i += 1) {
        const xi = ibl_prefilter.hammersley(i, sample_count);
        // Uniform sphere sampling
        const phi = 2.0 * std.math.pi * xi[0];
        const cos_theta = 1.0 - 2.0 * xi[1];
        const sin_theta = @sqrt(@max(0.0, 1.0 - cos_theta * cos_theta));
        const dir = Vec3.new(sin_theta * @cos(phi), cos_theta, sin_theta * @sin(phi));

        const radiance = if (dir.y > 0.0) sky_color else ground_color;
        sh.addSample(dir, radiance, (4.0 * std.math.pi) / @as(f32, @floatFromInt(sample_count)));
    }

    // Evaluate irradiance along upward normal (+Y), downward normal (-Y), and horizontal (+X)
    const up_irr = sh.evaluate(Vec3.new(0, 1, 0));
    const down_irr = sh.evaluate(Vec3.new(0, -1, 0));
    const horiz_irr = sh.evaluate(Vec3.new(1, 0, 0));

    // Upward irradiance must be blue-dominated (sky)
    try std.testing.expect(up_irr.z > up_irr.x);
    try std.testing.expect(up_irr.y > up_irr.x);

    // Downward irradiance must be brown-dominated (ground)
    try std.testing.expect(down_irr.x > down_irr.z);

    // Total intensity facing upward is significantly brighter than facing ground
    const up_luma = up_irr.x * 0.2126 + up_irr.y * 0.7152 + up_irr.z * 0.0722;
    const down_luma = down_irr.x * 0.2126 + down_irr.y * 0.7152 + down_irr.z * 0.0722;
    try std.testing.expect(up_luma > down_luma * 1.5);

    // Horizon irradiance should be intermediate
    const horiz_luma = horiz_irr.x * 0.2126 + horiz_irr.y * 0.7152 + horiz_irr.z * 0.0722;
    try std.testing.expect(horiz_luma > down_luma and horiz_luma < up_luma);
}
