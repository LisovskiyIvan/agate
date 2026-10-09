//! Thin compute-pass wrapper over sokol-gfx.
//!
//! sokol-gfx (since the late-2025 "views" rework) has no dedicated
//! `sg_begin_compute_pass` / `sg_apply_compute_pipeline` pair: compute is a
//! regular pass started with `sg_begin_pass(&(.{ .compute = true }))`, a
//! regular pipeline created with `sg_pipeline_desc.compute = true`, and is
//! driven with the same apply/bind/dispatch calls as rendering. This module
//! wraps pipeline creation, storage-buffer view setup, and dispatch group
//! math in one place, plus documents the platform matrix.
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
//! Compute shaders cannot share the default glsl410 slang (410 has no
//! compute): modules must use their own sokol-shdc invocation with a
//! compute-capable slang set (glsl430 / metal_macos / hlsl5). On any
//! backend where `supported()` returns false, a compute pipeline cannot be
//! created — surface that as an explicit error to your users.
const std = @import("std");
const builtin = @import("builtin");
const sokol = @import("sokol");
const sg = sokol.gfx;

/// Recommended default workgroup (local_size_x) used by engine compute
/// shaders. Keep in sync with `layout(local_size_x = ...) in;` in the GLSL.
pub const default_workgroup_size: usize = 64;

/// Runtime capability query. Requires a live sg context: sokol asserts
/// `_sg.valid` in debug builds when queried before setup (this engine only
/// calls it from frame-loop code that runs after sg.setup()).
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
/// The shader must come from an shdc `@cs` program (a compute-stage-only
/// shader module compiled for compute-capable slangs).
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
