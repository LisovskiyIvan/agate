//! Single-sample depth prepass for the MSAA depth-resolve design
//! (Scene.msaa_depth_prepass, PASS 1.7 in Scene.render).
//!
//! sokol has no depth resolve on any backend (see scene/msaa.zig), so an
//! MSAA main target cannot feed a depth texture to the post chain. When
//! the gate is on, this pass redraws the opaque primary-view geometry
//! with depth-only pipelines (position-only vertex stages, no color
//! attachments — the shadow-map pattern) into a 1x depth texture that
//! SSAO/SSR/DoF/Fog/MotionBlur sample like the legacy 1x main depth.
//!
//! Lifecycle mirrors the lazy MSAA twins (PostFXStack.outline_msaa):
//! created on the first active frame via ensure(), destroyed with the
//! stack or when the gate goes idle again. The off path holds no VRAM
//! and issues no draws (bit-identical to pre-prepass frames).
//!
//! Documented approximations (v1 non-goals):
//! - 1x depth vs MSAA color can differ by a pixel at geometric edges;
//!   depth-driven effects blend across the seam like any 1x depth.
//! - Cutout renders opaque and GPU morphs contribute base positions —
//!   the exact shadow-map precedent (shadow.glsl has neither alpha-test
//!   nor morph stages either: conservative over-occlusion only).
//! - Hook-material (shader_index), transparent, decal, particle, trail,
//!   skybox, debug and secondary-view geometry is skipped: hooks need
//!   their custom vertex stage, transparent/decal meshes write no depth
//!   in the main pass either, and multi-camera stays primary-only (the
//!   PASS 2.85 highlight-mask precedent).
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const depth_shd = @import("msaa_depth_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const mesh_mod = @import("../mesh.zig");
const Vertex = mesh_mod.Vertex;
const scene_render_queue = @import("../scene/render_queue.zig");
const RenderQueues = scene_render_queue.RenderQueues;
const stats_mod = @import("../scene/stats.zig");
const SceneStats = stats_mod.SceneStats;
pub const MsaaDepthPass = struct {
    image: sg.Image = .{},
    att_view: sg.View = .{},
    tex_view: sg.View = .{},
    pip_u16: sg.Pipeline = .{},
    pip_u32: sg.Pipeline = .{},
    inst_pip_u16: sg.Pipeline = .{},
    inst_pip_u32: sg.Pipeline = .{},
    skin_pip_u16: sg.Pipeline = .{},
    skin_pip_u32: sg.Pipeline = .{},
    rigid_shader: sg.Shader = .{},
    inst_shader: sg.Shader = .{},
    skin_shader: sg.Shader = .{},
    width: i32 = 0,
    height: i32 = 0,

    /// Creates the depth-only shaders and pipelines (context thread).
    /// Pipelines mirror the shadow-map descs (depth LESS_EQUAL + write,
    /// colors[0] = NONE, BACK cull, sample_count 1) with one deliberate
    /// difference: NO depth bias — the prepass depth must match the
    /// main-pass depth exactly where rasterization agrees.
    pub fn init() MsaaDepthPass {
        const backend = sg.queryBackend();
        const rigid_shd = sg.makeShader(depth_shd.msaaDepthShaderDesc(backend));
        var rigid_desc = sg.PipelineDesc{
            .shader = rigid_shd,
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        rigid_desc.colors[0].pixel_format = .NONE;
        rigid_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        rigid_desc.layout.attrs[depth_shd.ATTR_msaa_depth_position] = .{
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        const pip_u16 = sg.makePipeline(rigid_desc);
        rigid_desc.index_type = .UINT32;
        const pip_u32 = sg.makePipeline(rigid_desc);

        const inst_shd = sg.makeShader(depth_shd.msaaDepthInstancedShaderDesc(backend));
        var inst_desc = sg.PipelineDesc{
            .shader = inst_shd,
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        inst_desc.colors[0].pixel_format = .NONE;
        inst_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        inst_desc.layout.attrs[depth_shd.ATTR_msaa_depth_instanced_position] = .{
            .buffer_index = 0,
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        inst_desc.layout.buffers[1] = .{
            .step_func = .PER_INSTANCE,
            .step_rate = 1,
            .stride = @sizeOf(Mat4),
        };
        inst_desc.layout.attrs[depth_shd.ATTR_msaa_depth_instanced_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
        inst_desc.layout.attrs[depth_shd.ATTR_msaa_depth_instanced_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
        inst_desc.layout.attrs[depth_shd.ATTR_msaa_depth_instanced_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
        inst_desc.layout.attrs[depth_shd.ATTR_msaa_depth_instanced_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };
        const inst_pip_u16 = sg.makePipeline(inst_desc);
        inst_desc.index_type = .UINT32;
        const inst_pip_u32 = sg.makePipeline(inst_desc);

        const skin_shd = sg.makeShader(depth_shd.msaaDepthSkinnedShaderDesc(backend));
        var skin_desc = sg.PipelineDesc{
            .shader = skin_shd,
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        skin_desc.colors[0].pixel_format = .NONE;
        skin_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        skin_desc.layout.attrs[depth_shd.ATTR_msaa_depth_skinned_position] = .{
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        skin_desc.layout.attrs[depth_shd.ATTR_msaa_depth_skinned_joints] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "joints"),
        };
        skin_desc.layout.attrs[depth_shd.ATTR_msaa_depth_skinned_weights] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "weights"),
        };
        const skin_pip_u16 = sg.makePipeline(skin_desc);
        skin_desc.index_type = .UINT32;
        const skin_pip_u32 = sg.makePipeline(skin_desc);

        return .{
            .pip_u16 = pip_u16,
            .pip_u32 = pip_u32,
            .inst_pip_u16 = inst_pip_u16,
            .inst_pip_u32 = inst_pip_u32,
            .skin_pip_u16 = skin_pip_u16,
            .skin_pip_u32 = skin_pip_u32,
            .rigid_shader = rigid_shd,
            .inst_shader = inst_shd,
            .skin_shader = skin_shd,
        };
    }

    pub fn deinit(self: *MsaaDepthPass) void {
        self.destroyTarget();
        if (self.pip_u16.id != 0) sg.destroyPipeline(self.pip_u16);
        if (self.pip_u32.id != 0) sg.destroyPipeline(self.pip_u32);
        if (self.inst_pip_u16.id != 0) sg.destroyPipeline(self.inst_pip_u16);
        if (self.inst_pip_u32.id != 0) sg.destroyPipeline(self.inst_pip_u32);
        if (self.skin_pip_u16.id != 0) sg.destroyPipeline(self.skin_pip_u16);
        if (self.skin_pip_u32.id != 0) sg.destroyPipeline(self.skin_pip_u32);
        if (self.rigid_shader.id != 0) sg.destroyShader(self.rigid_shader);
        if (self.inst_shader.id != 0) sg.destroyShader(self.inst_shader);
        if (self.skin_shader.id != 0) sg.destroyShader(self.skin_shader);
        self.* = .{};
    }

    fn destroyTarget(self: *MsaaDepthPass) void {
        if (self.image.id != 0) sg.destroyImage(self.image);
        if (self.att_view.id != 0) sg.destroyView(self.att_view);
        if (self.tex_view.id != 0) sg.destroyView(self.tex_view);
        self.image = .{};
        self.att_view = .{};
        self.tex_view = .{};
        self.width = 0;
        self.height = 0;
    }

    /// Ensures the 1x depth target at `width`x`height` (same shape as the
    /// main target, sample_count 1). Returns true when (re)created.
    /// Context thread only; no-op for degenerate sizes.
    pub fn ensure(self: *MsaaDepthPass, width: i32, height: i32) bool {
        if (width <= 0 or height <= 0) return false;
        if (self.width == width and self.height == height and self.image.id != 0) return false;
        self.destroyTarget();
        const img = sg.makeImage(.{
            .usage = .{ .depth_stencil_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = .DEPTH,
            .sample_count = 1,
        });
        self.image = img;
        self.att_view = sg.makeView(.{
            .depth_stencil_attachment = .{ .image = img },
        });
        self.tex_view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        self.width = width;
        self.height = height;
        return true;
    }

    /// Texture view for depth-consuming post slots. Empty until ensure()
    /// succeeds; callers fall back to the main-target view (see
    /// PostFXStack.depthSampleView).
    pub fn depthTexView(self: *const MsaaDepthPass) sg.View {
        return self.tex_view;
    }

    /// Draws the opaque primary-view queues into the 1x depth target.
    /// Fail-closed headless (returns before any sg call when no context
    /// is current); draws nothing when the target was never ensured.
    /// Uniform-only past the binds (no sg.updateBuffer, no upload-meter
    /// records), so renderReuse replays stay upload-free — though the
    /// render path skips the prepass on reuse anyway (the persisted
    /// texture still matches the replayed snapshot).
    pub fn render(
        self: *MsaaDepthPass,
        view_proj: Mat4,
        queues: *const RenderQueues,
        skins: []const [scene_render_queue.MAX_BONES]Mat4,
        stats: *SceneStats,
    ) void {
        if (!sg.isvalid()) return;
        if (self.att_view.id == 0) return;

        var action = sg.PassAction{};
        action.depth = .{
            .load_action = .CLEAR,
            .clear_value = 1.0,
            .store_action = .STORE,
        };
        var pass = sg.Pass{ .action = action };
        pass.attachments.depth_stencil = self.att_view;
        sg.beginPass(pass);

        var draws: u32 = 0;
        for (queues.items.items) |item| {
            if (drawRegularOne(self, view_proj, item, skins)) draws += 1;
        }
        for (queues.opaque_instanced.items) |batch| {
            if (drawInstancedOne(self, view_proj, batch)) draws += 1;
        }

        sg.endPass();
        stats.main_draw_calls += draws;
        stats.draw_calls += draws;
    }

    fn drawRegularOne(
        self: *MsaaDepthPass,
        view_proj: Mat4,
        item: scene_render_queue.RenderMeshItem,
        skins: []const [scene_render_queue.MAX_BONES]Mat4,
    ) bool {
        if (!itemContributesDepth(item.transparent, item.is_decal, item.shader_index != null)) return false;
        if (item.index_count == 0) return false;
        if (item.vertex_buffer.id == 0 or sg.queryBufferState(item.vertex_buffer) != .VALID) return false;
        if (item.index_buffer.id != 0 and sg.queryBufferState(item.index_buffer) != .VALID) return false;

        const mvp = Mat4.mul(view_proj, item.model);
        if (item.is_skinned) {
            // Stale skin index (builder-unreachable): skip the draw
            // instead of uploading another draw's bones (draw.zig
            // precedent: skinAt orelse return).
            const bones = scene_render_queue.skinAt(skins, item.skin_index) orelse return false;
            sg.applyPipeline(if (item.is_u32) self.skin_pip_u32 else self.skin_pip_u16);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = item.vertex_buffer;
            bind.index_buffer = item.index_buffer;
            sg.applyBindings(bind);
            const vs_params = depth_shd.VsParams{ .mvp = mvp };
            sg.applyUniforms(depth_shd.UB_vs_params, sg.asRange(&vs_params));
            const vs_skin = depth_shd.VsSkin{ .bones = bones.* };
            sg.applyUniforms(depth_shd.UB_vs_skin, sg.asRange(&vs_skin));
        } else {
            sg.applyPipeline(if (item.is_u32) self.pip_u32 else self.pip_u16);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = item.vertex_buffer;
            bind.index_buffer = item.index_buffer;
            sg.applyBindings(bind);
            const vs_params = depth_shd.VsParams{ .mvp = mvp };
            sg.applyUniforms(depth_shd.UB_vs_params, sg.asRange(&vs_params));
        }
        sg.draw(item.base_vertex, item.index_count, 1);
        return true;
    }

    fn drawInstancedOne(
        self: *MsaaDepthPass,
        view_proj: Mat4,
        batch: scene_render_queue.RenderInstancedBatch,
    ) bool {
        if (!itemContributesDepth(batch.transparent, batch.is_decal, false)) return false;
        if (batch.visible_instance_count == 0 or batch.instance_buffer.id == 0) return false;
        if (batch.vertex_buffer.id == 0 or sg.queryBufferState(batch.vertex_buffer) != .VALID) return false;
        if (batch.index_buffer.id != 0 and sg.queryBufferState(batch.index_buffer) != .VALID) return false;
        if (sg.queryBufferState(batch.instance_buffer) != .VALID) return false;

        sg.applyPipeline(if (batch.index_type == .UINT32) self.inst_pip_u32 else self.inst_pip_u16);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = batch.vertex_buffer;
        bind.vertex_buffers[1] = batch.instance_buffer;
        bind.index_buffer = batch.index_buffer;
        sg.applyBindings(bind);
        const vs_params = depth_shd.VsInstParams{ .view_proj = view_proj };
        sg.applyUniforms(depth_shd.UB_vs_inst_params, sg.asRange(&vs_params));
        sg.draw(0, batch.index_count, batch.visible_instance_count);
        return true;
    }
};

/// Pure skip-matrix for the prepass (unit-testable without a GPU):
/// transparent and decal meshes write no depth in the main pass either,
/// and hook-material meshes need their custom vertex stage (documented
/// v1 skips in the module docs).
pub fn itemContributesDepth(transparent: bool, is_decal: bool, has_hook: bool) bool {
    return !transparent and !is_decal and !has_hook;
}

const testing = std.testing;

test "itemContributesDepth pins the prepass skip matrix" {
    // Opaque regular geometry: the prepass payload.
    try testing.expect(itemContributesDepth(false, false, false));
    // Transparent/decal write no depth in the main pass: nothing to mirror.
    try testing.expect(!itemContributesDepth(true, false, false));
    try testing.expect(!itemContributesDepth(false, true, false));
    // Hook materials need their custom vertex stage (v1 non-goal).
    try testing.expect(!itemContributesDepth(false, false, true));
    try testing.expect(!itemContributesDepth(true, true, true));
}

test "render is fail-closed without a GPU context" {
    // Headless (no sg context): zeroed pass draws nothing, stats clean.
    var pass = std.mem.zeroes(MsaaDepthPass);
    var queues = RenderQueues{};
    defer queues.deinit(std.testing.allocator);
    var stats = SceneStats{};
    pass.render(Mat4.identity, &queues, &.{}, &stats);
    try testing.expectEqual(@as(u32, 0), stats.draw_calls);
    try testing.expectEqual(@as(u32, 0), stats.main_draw_calls);
    try testing.expect(pass.depthTexView().id == 0);
}
