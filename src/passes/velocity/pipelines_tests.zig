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
    try std.testing.expectEqual(@as(u32, 0), pass.pip_inst_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_inst_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_skin_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_skin_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_cpu.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_gpu.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_rigid_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_rigid_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_inst_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_inst_u32.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_skin_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_skin_u32.id);
}

test "pipeline leaf destroy on zero pass resets the format invariant" {
    // Zero handles are guarded (id != 0), so this issues no sg calls
    // headless; it must still reset the depth-format invariant. The stale
    // .DEPTH seed proves the reset: without it this test would fail.
    var pass = VelocityPass{ .depth_format = .DEPTH };
    pipelines.destroyPipelines(&pass);
    try std.testing.expectEqual(sg.PixelFormat.DEFAULT, pass.depth_format);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_rigid_u16.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_gpu.id);
    try std.testing.expectEqual(@as(u32, 0), pass.pip_marker_glass_skin_u32.id);
}
