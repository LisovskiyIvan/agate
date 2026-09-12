const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const outline_shd = @import("outline_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color4 = math.Color4;
const Mesh = @import("../mesh.zig").Mesh;
const Vertex = @import("../mesh.zig").Vertex;
const InstancedMesh = @import("../mesh.zig").InstancedMesh;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;

// Inverse-hull outline/highlight layer: highlighted meshes are redrawn with
// front-face culling (the inflated "inside out" hull), LESS_EQUAL depth
// test and no depth writes, inside the already-open main pass (same contract
// as DebugPass.render/SkyboxPass.render: the caller must have begun the main
// pass whose depth attachment is reused). The constant screen-space width
// comes from the shader (clip-depth-scaled NDC offset along the projected
// normal); no extra geometry or render targets are needed.
// Skinned and instanced meshes are out of scope: the outline shader has no
// bone/instance path and would draw the wrong pose, so they are skipped.

// Upper bound for the rim width; larger values would swallow small meshes.
pub const max_width_px: f32 = 16.0;

// NDC depth bias pushing the hull toward the camera (reduces rim shimmer at
// grazing angles). Zero by default: the LESS_EQUAL test already hides the
// covered hull interior.
pub const default_depth_bias: f32 = 0.0;

// Viewport used for the px->NDC conversion, set via resize(). Starts at a
// guarded 1x1 so a forgotten resize() cannot divide by zero (the rim would
// just be oversized until the parent wires resize, see report).
var viewport_size: [2]f32 = .{ 1.0, 1.0 };

/// Sanitizes the rim width: non-finite collapses to 0 (no outline),
/// otherwise clamped to [0, max_width_px].
pub fn clampWidthPx(width_px: f32) f32 {
    if (!std.math.isFinite(width_px)) return 0.0;
    return std.math.clamp(width_px, 0.0, max_width_px);
}

/// CPU mirror of the shader px->NDC conversion: per-axis NDC offset of a
/// clamped width on the given viewport. Guards degenerate viewports
/// (never returns Inf/NaN).
pub fn ndcExpandForViewport(width_px: f32, viewport_w: f32, viewport_h: f32) [2]f32 {
    const w = clampWidthPx(width_px);
    const vw = if (std.math.isFinite(viewport_w)) @max(viewport_w, 1.0) else 1.0;
    const vh = if (std.math.isFinite(viewport_h)) @max(viewport_h, 1.0) else 1.0;
    return .{ w * 2.0 / vw, w * 2.0 / vh };
}

/// Offsets a vertex along its normal by scale (CPU-side hull math for
/// tooling/tests; the GPU path does the equivalent in clip space).
/// Non-finite scale leaves the position untouched.
pub fn expandVertex(pos: [3]f32, normal: [3]f32, scale: f32) [3]f32 {
    if (!std.math.isFinite(scale)) return pos;
    return .{
        pos[0] + normal[0] * scale,
        pos[1] + normal[1] * scale,
        pos[2] + normal[2] * scale,
    };
}

/// Packs the shader `params` uniform from the clamped width and the current
/// viewport set via resize().
pub fn outlineParamsFor(width_px: f32) [4]f32 {
    return .{ clampWidthPx(width_px), viewport_size[0], viewport_size[1], default_depth_bias };
}

/// Whether a mesh can take the outline path: visible, has drawable indexed
/// geometry with live GPU buffers, and is a regular (non-skinned,
/// non-instanced) mesh. Pure predicate, no GPU calls.
pub fn shouldOutlineMesh(mesh: *const Mesh) bool {
    if (!mesh.is_visible) return false;
    if (mesh.index_count == 0) return false;
    if (mesh.vertex_buffer.id == 0 or mesh.index_buffer.id == 0) return false;
    if (mesh.skeleton != null) return false;
    if (mesh.instances.items.len > 0) return false;
    return true;
}

/// Fills the inverse-hull state on a pipeline desc (culling, depth, blend,
/// vertex layout). Takes no shader handle so it stays GPU-free and testable;
/// init() assigns the compiled shader before calling it.
pub fn configureOutlineDesc(desc: *sg.PipelineDesc) void {
    desc.depth = .{
        .compare = .LESS_EQUAL,
        .write_enabled = false,
    };
    // Inverse hull: cull front faces, the inflated back faces form the rim.
    desc.cull_mode = .FRONT;
    desc.face_winding = .CCW;
    // Flat color with alpha support (same factors as the transparent twins).
    desc.colors[0].blend = .{
        .enabled = true,
        .src_factor_rgb = .SRC_ALPHA,
        .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
        .src_factor_alpha = .ONE,
        .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
    };
    // Reuses the Mesh vertex buffer directly: same stride, only the
    // position/normal attributes are bound.
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[outline_shd.ATTR_outline_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_normal] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "normal"),
    };
}

pub const OutlinePass = struct {
    pipeline_u16: sg.Pipeline = .{},
    pipeline_u32: sg.Pipeline = .{},

    pub fn init() OutlinePass {
        const shader = sg.makeShader(outline_shd.outlineShaderDesc(sg.queryBackend()));

        var desc_u16 = sg.PipelineDesc{ .shader = shader, .index_type = .UINT16 };
        configureOutlineDesc(&desc_u16);
        desc_u16.index_type = .UINT16;
        const pip_u16 = sg.makePipeline(desc_u16);

        var desc_u32 = sg.PipelineDesc{ .shader = shader, .index_type = .UINT32 };
        configureOutlineDesc(&desc_u32);
        desc_u32.index_type = .UINT32;
        const pip_u32 = sg.makePipeline(desc_u32);

        return .{
            .pipeline_u16 = pip_u16,
            .pipeline_u32 = pip_u32,
        };
    }

    /// Remembers the drawable size for the px->NDC conversion. The parent
    /// (Scene) must forward its framebuffer resizes here.
    pub fn resize(w: i32, h: i32) void {
        viewport_size[0] = @max(1.0, @as(f32, @floatFromInt(w)));
        viewport_size[1] = @max(1.0, @as(f32, @floatFromInt(h)));
    }

    /// Draws the outline rim for `meshes` into the currently open main pass.
    /// Meshes failing shouldOutlineMesh are skipped silently (invisible,
    /// skinned/instanced, or without GPU buffers). Empty list and zero
    /// width are GPU-free no-ops.
    pub fn render(self: *OutlinePass, view_proj: Mat4, camera_pos: Vec3, meshes: []const *Mesh, color: Color4, width_px: f32) void {
        // Reserved for view-dependent effects (distance fade); the width is
        // depth-invariant by construction (clip-scaled NDC offset).
        _ = camera_pos;
        if (meshes.len == 0) return;
        const width = clampWidthPx(width_px);
        if (width <= 0.0) return;
        if (self.pipeline_u16.id == 0 and self.pipeline_u32.id == 0) return;

        for (meshes) |mesh| {
            if (!shouldOutlineMesh(mesh)) continue;
            const model = mesh.getWorldMatrix();
            const mvp = Mat4.mul(view_proj, model);
            const pip = if (mesh.index_type == .UINT32) self.pipeline_u32 else self.pipeline_u16;
            if (pip.id == 0) continue;
            sg.applyPipeline(pip);

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = mesh.vertex_buffer;
            bind.index_buffer = mesh.index_buffer;
            sg.applyBindings(bind);

            const vs_params = outline_shd.VsParams{
                .mvp = mvp,
                .model = model,
                .color = color.toArray(),
                .params = outlineParamsFor(width),
            };
            sg.applyUniforms(outline_shd.UB_vs_params, sg.asRange(&vs_params));

            sg.draw(0, mesh.index_count, 1);
        }
    }

    pub fn deinit(self: *OutlinePass) void {
        sg.destroyPipeline(self.pipeline_u16);
        sg.destroyPipeline(self.pipeline_u32);
        self.* = undefined;
    }
};

test "clampWidthPx clamps to [0, max] and sanitizes non-finite" {
    try std.testing.expectEqual(@as(f32, 0.0), clampWidthPx(-3.0));
    try std.testing.expectEqual(@as(f32, 0.0), clampWidthPx(0.0));
    try std.testing.expectEqual(@as(f32, 2.5), clampWidthPx(2.5));
    try std.testing.expectEqual(max_width_px, clampWidthPx(max_width_px + 100.0));
    try std.testing.expectEqual(@as(f32, 0.0), clampWidthPx(std.math.nan(f32)));
    try std.testing.expectEqual(@as(f32, 0.0), clampWidthPx(std.math.inf(f32)));
}

test "expandVertex offsets position along normal by scale" {
    const pos = [3]f32{ 1.0, 2.0, 3.0 };
    const nrm = [3]f32{ 0.0, 1.0, 0.0 };
    try std.testing.expectEqual([3]f32{ 1.0, 2.5, 3.0 }, expandVertex(pos, nrm, 0.5));
    try std.testing.expectEqual(pos, expandVertex(pos, nrm, 0.0));
    try std.testing.expectEqual([3]f32{ 1.0, 1.0, 3.0 }, expandVertex(pos, nrm, -1.0));
    try std.testing.expectEqual(pos, expandVertex(pos, nrm, std.math.nan(f32)));
    try std.testing.expectEqual(pos, expandVertex(pos, nrm, std.math.inf(f32)));
}

test "ndcExpandForViewport converts px width to NDC without div-by-zero" {
    const e = ndcExpandForViewport(2.0, 800.0, 600.0);
    try std.testing.expectApproxEqAbs(2.0 * 2.0 / 800.0, e[0], 1e-6);
    try std.testing.expectApproxEqAbs(2.0 * 2.0 / 600.0, e[1], 1e-6);
    // Degenerate viewports never produce Inf/NaN.
    for ([2]f32{ 0.0, -10.0, std.math.nan(f32), std.math.inf(f32) }) |bad| {
        const g = ndcExpandForViewport(2.0, bad, bad);
        try std.testing.expect(std.math.isFinite(g[0]));
        try std.testing.expect(std.math.isFinite(g[1]));
    }
    // Width clamping flows through.
    const c = ndcExpandForViewport(max_width_px + 100.0, 800.0, 600.0);
    try std.testing.expectApproxEqAbs(max_width_px * 2.0 / 800.0, c[0], 1e-6);
}

test "configureOutlineDesc sets inverse-hull state" {
    var desc = std.mem.zeroes(sg.PipelineDesc);
    configureOutlineDesc(&desc);
    try std.testing.expect(desc.cull_mode == .FRONT);
    try std.testing.expect(desc.depth.compare == .LESS_EQUAL);
    try std.testing.expect(!desc.depth.write_enabled);
    try std.testing.expect(desc.colors[0].blend.enabled);
    try std.testing.expect(desc.colors[0].blend.src_factor_rgb == .SRC_ALPHA);
    try std.testing.expect(desc.colors[0].blend.dst_factor_rgb == .ONE_MINUS_SRC_ALPHA);
    try std.testing.expectEqual(@sizeOf(Vertex), desc.layout.buffers[0].stride);
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_position].format == .FLOAT3);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "position"))),
        desc.layout.attrs[outline_shd.ATTR_outline_position].offset,
    );
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_normal].format == .FLOAT3);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "normal"))),
        desc.layout.attrs[outline_shd.ATTR_outline_normal].offset,
    );
}

test "shouldOutlineMesh filters invisible, empty, skinned and instanced meshes" {
    var mesh = Mesh{
        .name = "outline-test",
        .vertex_buffer = .{ .id = 1 },
        .index_buffer = .{ .id = 2 },
        .index_count = 36,
    };
    try std.testing.expect(shouldOutlineMesh(&mesh));

    mesh.is_visible = false;
    try std.testing.expect(!shouldOutlineMesh(&mesh));
    mesh.is_visible = true;

    mesh.index_count = 0;
    try std.testing.expect(!shouldOutlineMesh(&mesh));
    mesh.index_count = 36;

    mesh.vertex_buffer = .{};
    try std.testing.expect(!shouldOutlineMesh(&mesh));
    mesh.vertex_buffer = .{ .id = 1 };

    var skel: Skeleton = undefined;
    mesh.skeleton = &skel;
    try std.testing.expect(!shouldOutlineMesh(&mesh));
    mesh.skeleton = null;
    try std.testing.expect(shouldOutlineMesh(&mesh));

    var inst_storage: [1]*InstancedMesh = undefined;
    mesh.instances.items = inst_storage[0..];
    try std.testing.expect(!shouldOutlineMesh(&mesh));
    mesh.instances.items = &.{};
    try std.testing.expect(shouldOutlineMesh(&mesh));
}

test "render with empty mesh list is a GPU-free no-op" {
    var pass = OutlinePass{};
    // Zero pipelines + empty list must return before any sokol call.
    pass.render(Mat4.identity, Vec3.zero, &.{}, Color4.white, 2.0);
    // Zero width is a no-op even with a (skipped) mesh present.
    var mesh = Mesh{
        .name = "outline-test",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    const meshes = [_]*Mesh{&mesh};
    pass.render(Mat4.identity, Vec3.zero, &meshes, Color4.white, 0.0);
}

test "outlineParamsFor packs clamped width and current viewport" {
    OutlinePass.resize(800, 600);
    const p = outlineParamsFor(2.0);
    try std.testing.expectEqual([4]f32{ 2.0, 800.0, 600.0, default_depth_bias }, p);
    const clamped = outlineParamsFor(max_width_px + 100.0);
    try std.testing.expectEqual(max_width_px, clamped[0]);
    // Degenerate resize inputs stay guarded.
    OutlinePass.resize(0, -5);
    const g = outlineParamsFor(2.0);
    try std.testing.expect(g[1] >= 1.0 and g[2] >= 1.0);
    OutlinePass.resize(800, 600);
}
