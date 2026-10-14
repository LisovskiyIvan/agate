const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const velocity_owner = @import("../velocity_pass.zig");
const VelocityPass = velocity_owner.VelocityPass;
const pipelines = @import("pipelines.zig");

test "pipeline leaf fails closed headless without publishing state" {
    // No sg context here: ensure must fail closed before any sg call that
    // could publish state (isvalid probe, then format guards).
    var pass = VelocityPass{};
    try std.testing.expect(!pipelines.ensurePipelinesFor(&pass, .DEFAULT));
    try std.testing.expect(!pipelines.ensurePipelinesFor(&pass, .NONE));
    try std.testing.expect(!pipelines.ensurePipelinesFor(&pass, .DEPTH));
    try std.testing.expectEqual(sg.PixelFormat.DEFAULT, pass.depth_format);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_rigid_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_rigid_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_rigid_u16_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_rigid_u32_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_inst_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_inst_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_inst_u16_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_inst_u32_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_skin_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_skin_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_skin_u16_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_skin_u32_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_cpu.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_gpu.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_rigid_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_rigid_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_rigid_u16_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_rigid_u32_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_inst_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_inst_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_inst_u16_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_inst_u32_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_skin_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_skin_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_skin_u16_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_skin_u32_ds.id);
}

test "pipeline leaf destroy on zero pass resets the format invariant" {
    // Zero handles are guarded (id != 0), so this issues no sg calls
    // headless; it must still reset the depth-format invariant. The stale
    // .DEPTH seed proves the reset: without it this test would fail.
    var pass = VelocityPass{ .depth_format = .DEPTH };
    pipelines.destroyPipelines(&pass);
    try std.testing.expectEqual(sg.PixelFormat.DEFAULT, pass.depth_format);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_rigid_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_rigid_u16_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_inst_u32_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_skin_u16_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_gpu.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_skin_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_skin_u32_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_rigid_u16_ds.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_inst_u32_ds.id);
}

test "sidedness maps to cull mode, touching nothing else" {
    // Single-sided culls back faces (pre-sidedness behavior);
    // double-sided disables culling so backfaces emit coverage.
    try std.testing.expectEqual(sg.CullMode.BACK, pipelines.cullModeFor(false));
    try std.testing.expectEqual(sg.CullMode.NONE, pipelines.cullModeFor(true));

    // applySidedness forces exactly the cull field: every other
    // descriptor field (depth EQUAL vs LESS_EQUAL, writes off, formats,
    // winding, layout) survives untouched on both twins.
    const base = sg.PipelineDesc{
        .depth = .{ .compare = .EQUAL, .write_enabled = false },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    var single = base;
    pipelines.applySidedness(&single, false);
    try std.testing.expectEqual(sg.CullMode.BACK, single.cull_mode);
    try std.testing.expectEqual(base.depth.compare, single.depth.compare);
    try std.testing.expectEqual(base.depth.write_enabled, single.depth.write_enabled);
    try std.testing.expectEqual(base.face_winding, single.face_winding);

    var double = base;
    pipelines.applySidedness(&double, true);
    try std.testing.expectEqual(sg.CullMode.NONE, double.cull_mode);
    try std.testing.expectEqual(base.depth.compare, double.depth.compare);
    try std.testing.expectEqual(base.depth.write_enabled, double.depth.write_enabled);
    try std.testing.expectEqual(base.face_winding, double.face_winding);

    // Same pin for the marker plan (LESS_EQUAL): sidedness never flips
    // the compare or the writes-off invariant.
    var marker = sg.PipelineDesc{
        .depth = .{ .compare = .LESS_EQUAL, .write_enabled = false },
        .cull_mode = .BACK,
    };
    pipelines.applySidedness(&marker, true);
    try std.testing.expectEqual(sg.CullMode.NONE, marker.cull_mode);
    try std.testing.expectEqual(sg.CompareFunc.LESS_EQUAL, marker.depth.compare);
    try std.testing.expect(!marker.depth.write_enabled);
}
