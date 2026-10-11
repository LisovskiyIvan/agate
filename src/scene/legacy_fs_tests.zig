//! Legacy pre-split fs_params wire (2560 bytes): public-API preservation.
//!
//! Runtime `engine_template = true` registrations compiled against the
//! d6c5710 templates keep the old single-block contract (UB 1, whole per
//! draw); static hook templates use the split pair (Entry.frame_split).
//! Pure CPU tests (no GPU calls).

const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const sokol = @import("sokol");
const sg = sokol.gfx;

const uniforms = @import("uniforms.zig");
const sm = @import("../shader_material.zig");

test "legacy wire: block size is the pre-split 2560 bytes" {
    // 2032 view lanes + 520 material lanes + 8 std140 array-alignment gap.
    try std.testing.expectEqual(@as(usize, 2560), @sizeOf(uniforms.LegacyFsParams));
    try std.testing.expect(@sizeOf(uniforms.LegacyFsParams) % 16 == 0);
}

test "legacy wire: field order matches the d6c5710 fs_params schema" {
    // Declaration order is the wire order (extern struct): any reorder
    // breaks shaders compiled against the old templates. This list mirrors
    // the d6c5710 src/shaders/pbr.glsl fs_params member sequence.
    const expected = [_][]const u8{
        "eye_pos",             "light_dir",             "light_color",          "ambient_color",
        "base_color_factor",   "pbr_factors",           "emissive_factor",      "shadow_params",
        "shadow_splits",       "cascade_view_proj",     "cascade_debug",        "light_counts",
        "point_pos_range",     "point_color_int",       "spot_pos_range",       "spot_dir_inner",
        "spot_color_outer",    "spot_intensity",        "spot_view_proj",       "spot_shadow_params",
        "alpha_cutoff",        "normal_scale",          "_pad_legacy_uv",       "uv_matrix",
        "uv_offset",           "channel_selectors",     "clearcoat_factors",    "clearcoat_color",
        "sheen_factors",       "sheen_color",           "point_view_proj",      "point_shadow_params",
        "directional_dir",     "directional_color_int", "probe_params",         "area_center_int",
        "area_right",          "area_up",               "area_color",           "clustered_params",
        "clustered_viewport",  "anisotropy_factors",    "transmission_factors", "transmission_color",
        "sss_factors",         "sss_color",             "hemi_dir_intensity",   "hemi_diffuse",
        "clearcoat_uv_matrix", "clearcoat_uv_offset",   "sheen_uv_matrix",      "sheen_uv_offset",
        "refraction_factors",  "refraction_view_proj",  "refraction_capture",   "probe2_params",
        "probe_box",
    };
    const actual = std.meta.fieldNames(uniforms.LegacyFsParams);
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |w, g| try std.testing.expectEqualStrings(w, g);
    // Hand-derived std140 anchors ( eye@0; shadow pair after the 7 leading
    // vec4s; cascades; scalar pair + 8B gap before the uv arrays; trailing
    // probe box at the block end).
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(uniforms.LegacyFsParams, "eye_pos"));
    try std.testing.expectEqual(@as(usize, 144), @offsetOf(uniforms.LegacyFsParams, "cascade_view_proj"));
    try std.testing.expectEqual(@as(usize, 864), @offsetOf(uniforms.LegacyFsParams, "uv_matrix"));
    try std.testing.expectEqual(@as(usize, 2528), @offsetOf(uniforms.LegacyFsParams, "probe_box"));
}

test "legacy wire: pack preserves view, material, and explicit refraction" {
    const view: uniforms.FrameViewUniforms = .{
        .eye_pos = .{ 1, 2, 3, 4 },
        .light_dir = .{ 0, 1, 0, 2048 },
        .light_color = .{ 1, 1, 1, 2 },
        .ambient_color = .{ 0.2, 0.2, 0.2, 1 },
        .shadow_params = .{ 0.001, 0.8, 0.02, 1.5 },
        .shadow_splits = .{ 10, 26, 65, 150 },
        .cascade_view_proj = [_]Mat4{Mat4.identity} ** 4,
        .cascade_debug = .{ 0, 1, 0.5, 0.01 },
        .light_counts = .{ 1, 0, 0.001, 0.02 },
        .point_pos_range = [_][4]f32{.{ 1, 2, 3, 10 }} ** 4,
        .point_color_int = [_][4]f32{.{ 1, 1, 1, 1 }} ** 4,
        .spot_pos_range = [_][4]f32{.{ 0, 0, 0, 1 }} ** 2,
        .spot_dir_inner = [_][4]f32{.{ 0, 0, 0, 1 }} ** 2,
        .spot_color_outer = [_][4]f32{.{ 0, 0, 0, 1 }} ** 2,
        .spot_intensity = [_][4]f32{.{ 0, 0, 0, 1 }} ** 2,
        .spot_view_proj = [_]Mat4{Mat4.identity} ** 2,
        .spot_shadow_params = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .point_view_proj = [_]Mat4{Mat4.identity} ** 12,
        .point_shadow_params = [_][4]f32{.{ 0, 0, 0, 0 }} ** 4,
        .directional_dir = [_][4]f32{.{ 0, 0, 0, 0 }} ** 4,
        .directional_color_int = [_][4]f32{.{ 0, 0, 0, 0 }} ** 4,
        .area_center_int = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .area_right = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .area_up = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .area_color = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .clustered_params = .{ 0, 0, 0, 0 },
        .clustered_viewport = .{ 0, 0, 64, 0 },
        .hemi_dir_intensity = .{ 0, 1, 0, 1 },
        .hemi_diffuse = .{ 1, 1, 1, 0 },
        // Deliberately live: the pack must override these with the
        // explicit pair (hook legacy uses identity/disabled).
        .refraction_view_proj = Mat4.translation(Vec3.new(9, 9, 9)),
        .refraction_capture = .{ 1, 99, 99, 0 },
        // Legacy wire predates DDGI (zeros: the legacy block has no
        // ddgi lanes and the legacy shaders never sample the atlas).
        .ddgi_grid_origin = .{ 0, 0, 0, 0 },
        .ddgi_grid_counts = .{ 0, 0, 0, 0 },
        .ddgi_grid_step = .{ 0, 0, 0, 0 },
        .ddgi_atlas = .{ 0, 0, 0, 0 },
    };
    const in: uniforms.MaterialUniformInputs = .{
        .base_color = .{ 0.5, 0.5, 0.5, 1 },
        .pbr_factors = .{ 0, 0.5, 1, 1 },
        .emissive_color = .{ 0, 0, 0, 1 },
        .alpha_cutoff = 0.5,
        .normal_scale = 1,
        .uv_matrices = .{.{ 1, 0, 0, 1 }} ** 5,
        .uv_offsets = .{.{ 0, 0, 0, 0 }} ** 5,
        .channel_selectors = .{ 0, 1, 2, 0 },
        .clearcoat_factors = .{ 0, 0.03, 0, 0 },
        .clearcoat_color = .{ 1, 1, 1, 1 },
        .sheen_factors = .{ 0, 0.5, 0, 0 },
        .sheen_color = .{ 1, 1, 1, 1 },
        .anisotropy_factors = .{ 0, 0, 0, 0 },
        .transmission_factors = .{ 0, 0, 0, 0 },
        .transmission_color = .{ 1, 1, 1, 1 },
        .sss_factors = .{ 0, 0, 0, 0 },
        .sss_color = .{ 1, 1, 1, 1 },
        .clearcoat_uv_matrix = .{ 1, 0, 0, 1 },
        .clearcoat_uv_offset = .{ 0, 0, 0, 0 },
        .sheen_uv_matrix = .{ 1, 0, 0, 1 },
        .sheen_uv_offset = .{ 0, 0, 0, 0 },
        .refraction_factors = .{ 0, 0, 1.5, 0 },
        .probe_params = .{ 0, 0, 0, 0 },
        .probe2_params = .{ 0, 0, 0, 0 },
        .probe_box = .{ .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } },
    };
    const leg = uniforms.buildLegacyFsParams(&view, in, Mat4.identity, .{ 0, 1, 1, 0 });
    try std.testing.expectEqual(view.eye_pos, leg.eye_pos);
    try std.testing.expectEqual(view.shadow_params, leg.shadow_params);
    try std.testing.expectEqual(view.cascade_view_proj, leg.cascade_view_proj);
    try std.testing.expectEqual(in.base_color, leg.base_color_factor);
    try std.testing.expectEqual(in.alpha_cutoff, leg.alpha_cutoff);
    try std.testing.expectEqual(in.uv_matrices, leg.uv_matrix);
    try std.testing.expectEqual(view.point_view_proj, leg.point_view_proj);
    // Explicit pair wins over the view's live camera (hook legacy rule).
    try std.testing.expectEqual(Mat4.identity, leg.refraction_view_proj);
    try std.testing.expectEqual([4]f32{ 0, 1, 1, 0 }, leg.refraction_capture);
}

test "legacy contract: runtime true defaults to legacy, static is split" {
    const dummy = struct {
        fn makeShader(_: sg.Backend) sg.Shader {
            return .{};
        }
    };
    // Runtime engine_template=true keeps the pre-split wire by default.
    const legacy_idx = try sm.registerRuntime(.{ .name = "test_legacy_true", .make_shader = dummy.makeShader, .engine_template = true });
    defer sm.runtime_len -= 1;
    const leg_entry = sm.entry(legacy_idx).?;
    try std.testing.expect(leg_entry.engine_template);
    try std.testing.expect(!leg_entry.frame_split);
    try std.testing.expectEqual(@as(u32, 1), leg_entry.fs_ub);
    // Explicit opt-in selects the split wire for newly compiled shaders.
    const split_idx = try sm.registerRuntime(.{ .name = "test_split_true", .make_shader = dummy.makeShader, .engine_template = true, .frame_split = true });
    defer sm.runtime_len -= 1;
    try std.testing.expect(sm.entry(split_idx).?.frame_split);
    // Static hook entries are generated from the split sources.
    const static_idx = sm.indexForName("ramp_wave") orelse return error.TestUnexpectedResult;
    const static_entry = sm.entry(static_idx).?;
    try std.testing.expect(static_entry.engine_template);
    try std.testing.expect(static_entry.frame_split);
    try std.testing.expectEqual(@as(u32, 5), static_entry.fs_mat_ub);
}

test "legacy contract: engine_template=false custom path unaffected" {
    const dummy = struct {
        fn makeShader(_: sg.Backend) sg.Shader {
            return .{};
        }
    };
    const idx = try sm.registerRuntime(.{ .name = "test_legacy_false", .make_shader = dummy.makeShader, .user_ub = 3 });
    defer sm.runtime_len -= 1;
    const e = sm.entry(idx).?;
    try std.testing.expect(!e.engine_template);
    try std.testing.expect(!e.frame_split);
    // User-bytes window intact (v1 budget, default full window).
    try std.testing.expectEqual(@as(u16, 32), e.user_bytes);
    try std.testing.expectEqual(@as(u32, 3), e.user_ub.?);
}
