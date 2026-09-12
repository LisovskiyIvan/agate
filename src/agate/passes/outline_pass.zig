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
const Skeleton = @import("../animation/skeleton.zig").Skeleton;

// Inverse-hull outline/highlight layer: highlighted meshes are redrawn with
// front-face culling (the inflated "inside out" hull), LESS_EQUAL depth
// test and no depth writes, inside the already-open main pass (same contract
// as DebugPass.render/SkyboxPass.render: the caller must have begun the main
// pass whose depth attachment is reused). The constant screen-space width
// comes from the shader (clip-depth-scaled NDC offset along the projected
// normal); no extra geometry or render targets are needed.
// All three mesh flavors are supported: rigid meshes, instanced meshes
// (per-instance model matrix, only the visible instance prefix is drawn)
// and skinned meshes (matrix-palette skinning, matching skinned_pbr.glsl).

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

/// Whether a mesh can take the outline path at all: visible and drawable
/// through indexed geometry with live GPU buffers. Rigid, instanced and
/// skinned meshes all qualify; render() routes each to its pipeline family.
/// Pure predicate, no GPU calls.
pub fn shouldOutlineMesh(mesh: *const Mesh) bool {
    if (!mesh.is_visible) return false;
    if (mesh.index_count == 0) return false;
    if (mesh.vertex_buffer.id == 0 or mesh.index_buffer.id == 0) return false;
    return true;
}

/// Shared inverse-hull pipeline state (culling, depth, blend). Vertex layout
/// is configured per family by the configure*Desc functions below.
fn setInverseHullState(desc: *sg.PipelineDesc) void {
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
}

/// Rigid meshes: reuses the Mesh vertex buffer directly (same stride, only
/// the position/normal attributes are bound).
pub fn configureOutlineDesc(desc: *sg.PipelineDesc) void {
    setInverseHullState(desc);
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

/// Instanced meshes: mesh vertex buffer (position/normal) plus the shared
/// per-instance model-matrix buffer (4 x FLOAT4, PER_INSTANCE), mirroring
/// the instanced/instanced_pbr pipeline layouts.
pub fn configureOutlineInstDesc(desc: *sg.PipelineDesc) void {
    setInverseHullState(desc);
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_normal] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "normal"),
    };
    desc.layout.buffers[1] = .{
        .step_func = .PER_INSTANCE,
        .step_rate = 1,
        .stride = @sizeOf(Mat4),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
    desc.layout.attrs[outline_shd.ATTR_outline_inst_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };
}

/// Skinned meshes: full Mesh.Vertex layout (position/normal/joints/weights)
/// plus the vs_skin bone palette uniform applied per draw.
pub fn configureOutlineSkinnedDesc(desc: *sg.PipelineDesc) void {
    setInverseHullState(desc);
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[outline_shd.ATTR_outline_skinned_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_skinned_normal] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "normal"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_skinned_joints] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "joints"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_skinned_weights] = .{
        .format = .FLOAT4,
        .offset = @offsetOf(Vertex, "weights"),
    };
}

pub const OutlinePass = struct {
    pipeline_u16: sg.Pipeline = .{},
    pipeline_u32: sg.Pipeline = .{},
    pipeline_inst_u16: sg.Pipeline = .{},
    pipeline_inst_u32: sg.Pipeline = .{},
    pipeline_skinned_u16: sg.Pipeline = .{},
    pipeline_skinned_u32: sg.Pipeline = .{},

    pub fn init() OutlinePass {
        // One shader object per program family: the three @program entries in
        // outline.glsl generate separate desc functions, each with its own
        // uniform block table (only the skinned program declares vs_skin).
        const shd_rigid = sg.makeShader(outline_shd.outlineShaderDesc(sg.queryBackend()));
        const shd_inst = sg.makeShader(outline_shd.outlineInstShaderDesc(sg.queryBackend()));
        const shd_skin = sg.makeShader(outline_shd.outlineSkinnedShaderDesc(sg.queryBackend()));

        var desc_u16 = sg.PipelineDesc{ .shader = shd_rigid, .index_type = .UINT16 };
        configureOutlineDesc(&desc_u16);
        const pip_u16 = sg.makePipeline(desc_u16);

        var desc_u32 = sg.PipelineDesc{ .shader = shd_rigid, .index_type = .UINT32 };
        configureOutlineDesc(&desc_u32);
        const pip_u32 = sg.makePipeline(desc_u32);

        var desc_inst_u16 = sg.PipelineDesc{ .shader = shd_inst, .index_type = .UINT16 };
        configureOutlineInstDesc(&desc_inst_u16);
        const pip_inst_u16 = sg.makePipeline(desc_inst_u16);

        var desc_inst_u32 = sg.PipelineDesc{ .shader = shd_inst, .index_type = .UINT32 };
        configureOutlineInstDesc(&desc_inst_u32);
        const pip_inst_u32 = sg.makePipeline(desc_inst_u32);

        var desc_skin_u16 = sg.PipelineDesc{ .shader = shd_skin, .index_type = .UINT16 };
        configureOutlineSkinnedDesc(&desc_skin_u16);
        const pip_skin_u16 = sg.makePipeline(desc_skin_u16);

        var desc_skin_u32 = sg.PipelineDesc{ .shader = shd_skin, .index_type = .UINT32 };
        configureOutlineSkinnedDesc(&desc_skin_u32);
        const pip_skin_u32 = sg.makePipeline(desc_skin_u32);

        return .{
            .pipeline_u16 = pip_u16,
            .pipeline_u32 = pip_u32,
            .pipeline_inst_u16 = pip_inst_u16,
            .pipeline_inst_u32 = pip_inst_u32,
            .pipeline_skinned_u16 = pip_skin_u16,
            .pipeline_skinned_u32 = pip_skin_u32,
        };
    }

    /// Remembers the drawable size for the px->NDC conversion. The parent
    /// (Scene) must forward its framebuffer resizes here.
    pub fn resize(w: i32, h: i32) void {
        viewport_size[0] = @max(1.0, @as(f32, @floatFromInt(w)));
        viewport_size[1] = @max(1.0, @as(f32, @floatFromInt(h)));
    }

    /// Draws the outline rim for `meshes` into the currently open main pass.
    /// Each mesh routes to its pipeline family (rigid / instanced / skinned);
    /// meshes failing shouldOutlineMesh are skipped silently. Empty list and
    /// zero width are GPU-free no-ops.
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
            const skinned = mesh.skeleton != null;
            const instanced = !skinned and mesh.instances.items.len > 0;
            const is_u32 = mesh.index_type == .UINT32;
            const pip = if (skinned)
                (if (is_u32) self.pipeline_skinned_u32 else self.pipeline_skinned_u16)
            else if (instanced)
                (if (is_u32) self.pipeline_inst_u32 else self.pipeline_inst_u16)
            else
                (if (is_u32) self.pipeline_u32 else self.pipeline_u16);
            if (pip.id == 0) continue;

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = mesh.vertex_buffer;
            if (instanced) {
                // Occlusion-culled instances are compacted into the visible
                // prefix; nothing visible means nothing to outline.
                if (mesh.instance_buffer.id == 0 or mesh.visible_instance_count == 0) continue;
                bind.vertex_buffers[1] = mesh.instance_buffer;
            }
            bind.index_buffer = mesh.index_buffer;
            sg.applyPipeline(pip);
            sg.applyBindings(bind);

            const model = mesh.getWorldMatrix();
            const mvp = Mat4.mul(view_proj, model);
            const vs_params = outline_shd.VsParams{
                .mvp = mvp,
                .model = model,
                .color = color.toArray(),
                .params = outlineParamsFor(width),
            };
            sg.applyUniforms(outline_shd.UB_vs_params, sg.asRange(&vs_params));
            if (skinned) {
                const vs_skin = outline_shd.VsSkin{
                    .bones = mesh.skeleton.?.skin_matrices,
                };
                sg.applyUniforms(outline_shd.UB_vs_skin, sg.asRange(&vs_skin));
            }

            const instance_count: u32 = if (instanced) mesh.visible_instance_count else 1;
            sg.draw(0, mesh.index_count, instance_count);
        }
    }

    pub fn deinit(self: *OutlinePass) void {
        sg.destroyPipeline(self.pipeline_u16);
        sg.destroyPipeline(self.pipeline_u32);
        sg.destroyPipeline(self.pipeline_inst_u16);
        sg.destroyPipeline(self.pipeline_inst_u32);
        sg.destroyPipeline(self.pipeline_skinned_u16);
        sg.destroyPipeline(self.pipeline_skinned_u32);
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
    for ([_]f32{ 0.0, -10.0, std.math.nan(f32), std.math.inf(f32) }) |bad| {
        try std.testing.expect(std.math.isFinite(ndcExpandForViewport(2.0, bad, bad)[0]));
        try std.testing.expect(std.math.isFinite(ndcExpandForViewport(2.0, bad, bad)[1]));
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

test "configureOutlineInstDesc binds per-instance matrix buffer" {
    var desc = std.mem.zeroes(sg.PipelineDesc);
    configureOutlineInstDesc(&desc);
    try std.testing.expect(desc.cull_mode == .FRONT);
    try std.testing.expectEqual(@sizeOf(Vertex), desc.layout.buffers[0].stride);
    try std.testing.expectEqual(@sizeOf(Mat4), desc.layout.buffers[1].stride);
    try std.testing.expect(desc.layout.buffers[1].step_func == .PER_INSTANCE);
    inline for (0..4) |i| {
        const attr = switch (i) {
            0 => outline_shd.ATTR_outline_inst_inst_mat0,
            1 => outline_shd.ATTR_outline_inst_inst_mat1,
            2 => outline_shd.ATTR_outline_inst_inst_mat2,
            else => outline_shd.ATTR_outline_inst_inst_mat3,
        };
        try std.testing.expectEqual(@as(i32, 1), desc.layout.attrs[attr].buffer_index);
        try std.testing.expectEqual(@as(i32, @intCast(i * 16)), desc.layout.attrs[attr].offset);
        try std.testing.expect(desc.layout.attrs[attr].format == .FLOAT4);
    }
}

test "configureOutlineSkinnedDesc binds skin attributes" {
    var desc = std.mem.zeroes(sg.PipelineDesc);
    configureOutlineSkinnedDesc(&desc);
    try std.testing.expect(desc.cull_mode == .FRONT);
    try std.testing.expectEqual(@sizeOf(Vertex), desc.layout.buffers[0].stride);
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_skinned_joints].format == .FLOAT4);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "joints"))),
        desc.layout.attrs[outline_shd.ATTR_outline_skinned_joints].offset,
    );
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_skinned_weights].format == .FLOAT4);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "weights"))),
        desc.layout.attrs[outline_shd.ATTR_outline_skinned_weights].offset,
    );
}

test "shouldOutlineMesh filters invisible and empty meshes" {
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

    // Skinned and instanced meshes take dedicated pipeline families, so they
    // stay outline-eligible here (render() routes them).
    var skel: Skeleton = undefined;
    mesh.skeleton = &skel;
    try std.testing.expect(shouldOutlineMesh(&mesh));
    mesh.skeleton = null;
    mesh.instances.items.len = 3;
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
}
