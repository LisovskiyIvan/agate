//! Thin compute-pass wrapper over sokol-gfx.
//!
//! sokol-gfx (since the late-2025 "views" rework) has no dedicated
//! `sg_begin_compute_pass` / `sg_apply_compute_pipeline` pair: compute is a
//! regular pass started with `sg_begin_pass(&(.{ .compute = true }))`, a
//! regular pipeline created with `sg_pipeline_desc.compute = true`, and is
//! driven with the same apply/bind/dispatch calls as rendering. This module
//! wraps that contract in one place so engine code does not hand-roll pass
//! descriptors, plus documents the platform matrix.
//!
//! Backend support matrix (from the sokol_gfx.h "ON COMPUTE PASSES" section,
//! mirrored by `sg_query_features().compute` at runtime — the runtime query is
//! authoritative, this table is documentation + offline checks):
//!
//! | Platform / backend              | compute | storage buffers |
//! |---------------------------------|---------|-----------------|
//! | macOS / iOS / iOS-sim Metal     | yes     | yes             |
//! | Windows D3D11                   | yes     | yes             |
//! | Windows desktop GL (4.3+)       | yes     | yes             |
//! | Linux desktop GL (4.3+)         | yes     | yes             |
//! | Linux / Android GLES 3.1+       | yes     | yes             |
//! | Web WebGPU                      | yes     | yes             |
//! | macOS desktop GL (capped 4.1)   | no      | no              |
//! | iOS GLES3                       | no      | no              |
//! | Web WebGL2                      | no      | no              |
//! | Vulkan (sokol backend still WIP)| no      | no              |
//!
//! The engine compiles compute shaders for slang glsl430 (desktop GL),
//! metal_macos and hlsl5 — glsl410 cannot express compute, so compute shader
//! modules must use their own sokol-shdc invocation (see build.zig). On any
//! backend where `supported()` returns false, consumers (particles.zig
//! `.compute` mode) must fall back to a render-only simulation path.
const std = @import("std");
const builtin = @import("builtin");
const sokol = @import("sokol");
const sg = sokol.gfx;

/// Recommended default workgroup (local_size_x) used by engine compute
/// shaders. Keep in sync with `layout(local_size_x = ...) in;` in the GLSL.
pub const default_workgroup_size: usize = 64;

/// Runtime capability query. Safe to call before sg.setup() / after
/// sg.shutdown(): sokol then reports a zeroed feature set, i.e. false.
pub fn supported() bool {
    return sg.queryFeatures().compute;
}

/// Static (compile-time documented) counterpart of `supported()` for a known
/// backend. Only `supported()` gates actual usage; this exists for tests and
/// offline reasoning about the matrix above.
pub fn backendSupportsCompute(backend: sg.Backend) bool {
    return switch (backend) {
        .METAL_MACOS, .METAL_IOS, .METAL_SIMULATOR => true,
        .D3D11 => true,
        .WGPU => true,
        // Desktop GL: compute + SSBOs need GL 4.3+; macOS caps out at 4.1.
        .GLCORE => builtin.os.tag != .macos,
        // GLES: needs 3.1+; sokol requests 3.1 on Android/Linux. The runtime
        // query refines this per-context.
        .GLES3 => true,
        // sokol_gfx.h does not list Vulkan among the compute-capable backends.
        .VULKAN, .DUMMY => false,
    };
}

/// Number of workgroups (dispatch groups) needed to cover `items` invocations
/// with a `local_size`-wide workgroup. Ceiling division; 0 items -> 0 groups
/// (a dispatch of 0 groups is legal and a no-op, callers may skip it).
pub fn groupCount(items: usize, local_size: usize) usize {
    std.debug.assert(local_size > 0);
    if (items == 0) return 0;
    return (items + local_size - 1) / local_size;
}

/// Creates a compute pipeline from a compute-stage-only shader.
/// The shader must come from an shdc `@cs` program (see
/// shaders/particle_compute.glsl).
pub fn makePipeline(shader: sg.Shader, label: [:0]const u8) sg.Pipeline {
    return sg.makePipeline(.{
        .compute = true,
        .shader = shader,
        .label = label.ptr,
    });
}

/// Creates the read/write storage-buffer view sokol shaders bind for SSBOs.
/// The buffer must be created with `.usage.storage_buffer = true`.
pub fn makeStorageView(buffer: sg.Buffer, label: [:0]const u8) sg.View {
    return sg.makeView(.{
        .storage_buffer = .{ .buffer = buffer },
        .label = label.ptr,
    });
}

test "compute pass API contract (structure smoke)" {
    // Pins the sokol-gfx surface compute.zig builds on. If a sokol upgrade
    // renames or removes any of these fields, the contract breaks here —
    // without needing a GPU context in the test run.
    try comptime std.testing.expect(@hasDecl(sg, "queryFeatures"));
    try comptime std.testing.expect(@hasField(sg.Pass, "compute"));
    try comptime std.testing.expect(@hasField(sg.Pass, "label"));
    try comptime std.testing.expect(@hasField(sg.PipelineDesc, "compute"));
    try comptime std.testing.expect(@hasField(sg.PipelineDesc, "shader"));
    try comptime std.testing.expect(@hasField(sg.Features, "compute"));
    try comptime std.testing.expect(@hasField(sg.ViewDesc, "storage_buffer"));
    try comptime std.testing.expect(@hasField(sg.Bindings, "views"));
    try comptime std.testing.expect(@hasField(sg.Bindings, "vertex_buffers"));
    // Functions used by the wrapper and by compute consumers.
    try comptime std.testing.expect(@TypeOf(sg.makeView) != void);
    try comptime std.testing.expect(@TypeOf(sg.dispatch) != void);
    try comptime std.testing.expect(@TypeOf(sg.destroyView) != void);
}

test "groupCount is ceiling division" {
    try std.testing.expectEqual(@as(usize, 0), groupCount(0, 64));
    try std.testing.expectEqual(@as(usize, 1), groupCount(1, 64));
    try std.testing.expectEqual(@as(usize, 1), groupCount(64, 64));
    try std.testing.expectEqual(@as(usize, 2), groupCount(65, 64));
    try std.testing.expectEqual(@as(usize, 16), groupCount(1024, 64));
    try std.testing.expectEqual(@as(usize, 17), groupCount(1025, 64));
    // Non-default workgroup sizes stay exact.
    try std.testing.expectEqual(@as(usize, 3), groupCount(97, 40));
}

test "static backend matrix matches the sokol documentation" {
    try std.testing.expect(backendSupportsCompute(.METAL_MACOS));
    try std.testing.expect(backendSupportsCompute(.METAL_IOS));
    try std.testing.expect(backendSupportsCompute(.METAL_SIMULATOR));
    try std.testing.expect(backendSupportsCompute(.D3D11));
    try std.testing.expect(backendSupportsCompute(.WGPU));
    // Desktop GL: 4.3+ everywhere except macOS (capped at 4.1).
    try std.testing.expectEqual(builtin.os.tag != .macos, backendSupportsCompute(.GLCORE));
    try std.testing.expect(backendSupportsCompute(.GLES3));
    try std.testing.expect(!backendSupportsCompute(.VULKAN));
    try std.testing.expect(!backendSupportsCompute(.DUMMY));
}
