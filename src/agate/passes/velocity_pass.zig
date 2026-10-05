//! Velocity buffer pass for TAA and motion blur.
//!
//! Renders opaque rigid, skinned, and instanced meshes with current vs
//! previous transforms to generate screen-space velocity vectors
//! (uv_current - uv_previous) into an RGBA16F buffer.
//!
//! Geometry skip matrix matches itemContributesDepth: transparent, decal,
//! and hook-material meshes are skipped.
//!
//! Fail-closed headless: all sg calls guarded, clean stats.
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const velocity_shd = @import("velocity_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const mesh_mod = @import("../mesh.zig");
const Vertex = mesh_mod.Vertex;
const scene_render_queue = @import("../scene/render_queue.zig");
const RenderQueues = scene_render_queue.RenderQueues;
const stats_mod = @import("../scene/stats.zig");
const SceneStats = stats_mod.SceneStats;
const msaa_depth = @import("msaa_depth_pass.zig");
pub const itemContributesDepth = msaa_depth.itemContributesDepth;
const material_mod = @import("../material.zig");
const MaterialDrawRecord = material_mod.MaterialDrawRecord;

/// Packs the FS alpha-test uniforms from a frozen draw record (the actual
/// mapping the draws use — never hand-rolled per call): slot-0 albedo UV
/// transform, base color factor, and the cutout threshold. Opaque/blend
/// records carry cutoff 0.0, which keeps the shader's no-discard branch.
pub fn velocityAlphaUniforms(rec: MaterialDrawRecord) velocity_shd.FsAlpha {
    return .{
        .base_color_factor = rec.base_color,
        .uv_matrix = rec.uv_matrices[0],
        .uv_offset = rec.uv_offsets[0],
        .alpha_cutoff = rec.alpha_cutoff,
    };
}

/// Pure velocity-dispatch decision (no sg calls): TAA always needs the
/// buffer when it runs (taa_on already implies samples == 1); motion blur
/// needs it only at 1x. Under MSAA the velocity pass is explicitly
/// suppressed even with the depth prepass — the blur then uses camera-only
/// depth reprojection against the prepass depth, so no false per-object
/// vectors can come from a sample-count-mismatched target.
pub fn velocityNeeded(taa_on: bool, motion_blur_enabled: bool, samples: i32) bool {
    if (taa_on) return true;
    if (!motion_blur_enabled) return false;
    return samples == 1;
}

/// Cut-frame suppression (pure): a TAA reset (first frame, toggle, resize,
/// camera cut, explicit reset) or a camera cut under blur-only mode must
/// bind the zero mask and skip the blur gather for exactly that composite.
pub fn cutSuppressesVelocity(taa_reset: bool, motion_blur_enabled: bool, camera_cut: bool) bool {
    return taa_reset or (motion_blur_enabled and camera_cut);
}

/// Generation gate for frozen prev state (pure, no sg): a staged prev
/// payload is usable only when its generation equals the last actually
/// rendered generation. Anything staged-but-never-presented (latch without
/// render, superseded front, reused front replaying old motion) fails the
/// match and the draw falls back to zero motion. `maxInt` (never staged)
/// never matches: frame ids count up from 0 and never reach it.
pub fn usePresentedPrev(prev_frame: u64, last_rendered: u64) bool {
    if (prev_frame == std.math.maxInt(u64)) return false;
    return prev_frame == last_rendered;
}

/// Shared-depth pipeline state (pure plan): depth writes OFF with an EQUAL
/// compare against the borrowed main depth. EQUAL binds actual visible main
/// pixels — nearer morph-displaced surfaces and nearer opaque geometry
/// occlude farther velocity draws, and cutout holes (main depth holds the
/// farther background there) fail the match instead of writing false mask1.
/// The remaining coplanar-cutout tie is resolved in-shader: the FS replicates
/// the main pass's alpha-cutoff discard (see velocity.glsl), so discarded
/// fragments never reach the depth test at all.
pub fn velocityDepthState() sg.DepthState {
    return .{
        .compare = .EQUAL,
        .write_enabled = false,
    };
}

/// Shared-depth pass action (pure plan): own color target cleared to the
/// zero mask; borrowed main depth LOADED and STORED (never cleared, never
/// written — the post chain keeps sampling it afterwards).
pub fn velocityDepthPassAction() sg.PassAction {
    var a = sg.PassAction{};
    a.colors[0] = .{
        .load_action = .CLEAR,
        .clear_value = .{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 0.0 },
        .store_action = .STORE,
    };
    a.depth = .{
        .load_action = .LOAD,
        .store_action = .STORE,
    };
    return a;
}

pub const VelocityPass = struct {
    image: sg.Image = .{},
    att_view: sg.View = .{},
    tex_view: sg.View = .{},
    dummy_zero_image: sg.Image = .{},
    dummy_zero_view: sg.View = .{},
    /// Fallback albedo sampler for hand-built records with a null sampler
    /// (production records always carry the texture's own sampler, bound
    /// verbatim like the main pass). Linear clamp, color-sampling shape.
    albedo_fallback_sampler: sg.Sampler = .{},

    pip_rigid_u16: sg.Pipeline = .{},
    pip_rigid_u32: sg.Pipeline = .{},
    pip_inst_u16: sg.Pipeline = .{},
    pip_inst_u32: sg.Pipeline = .{},
    pip_skin_u16: sg.Pipeline = .{},
    pip_skin_u32: sg.Pipeline = .{},

    rigid_shader: sg.Shader = .{},
    inst_shader: sg.Shader = .{},
    skin_shader: sg.Shader = .{},

    width: i32 = 0,
    height: i32 = 0,
    /// Main-depth pixel format the pipelines were built for (.DEFAULT =
    /// none yet). The pass borrows the main depth attachment (never owns
    /// or destroys it); pipelines are (re)built lazily to match it.
    depth_format: sg.PixelFormat = .DEFAULT,

    pub fn init() VelocityPass {
        if (!sg.isvalid()) return .{};
        const backend = sg.queryBackend();

        // Store-first rollback throughout: every handle lands in `out`
        // before its state check, so a single `deinit` frees the failed
        // handle plus all partial handles (a nonzero FAILED id still owns
        // its pool slot and must be released, never leaked).
        var out: VelocityPass = .{};

        // 1x1 zero dummy texture (clear velocity fallback)
        const zero_pixels = [_]f16{ 0.0, 0.0, 0.0, 0.0 };
        var zero_data = sg.ImageData{};
        zero_data.mip_levels[0] = sg.asRange(&zero_pixels);
        out.dummy_zero_image = sg.makeImage(.{
            .width = 1,
            .height = 1,
            .pixel_format = .RGBA16F,
            .data = zero_data,
        });
        if (sg.queryImageState(out.dummy_zero_image) != .VALID) {
            out.deinit();
            return .{};
        }
        out.dummy_zero_view = sg.makeView(.{
            .texture = .{ .image = out.dummy_zero_image },
        });
        if (sg.queryViewState(out.dummy_zero_view) != .VALID) {
            out.deinit();
            return .{};
        }
        out.albedo_fallback_sampler = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });
        if (sg.querySamplerState(out.albedo_fallback_sampler) != .VALID) {
            out.deinit();
            return .{};
        }

        out.rigid_shader = sg.makeShader(velocity_shd.velocityShaderDesc(backend));
        if (sg.queryShaderState(out.rigid_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        out.inst_shader = sg.makeShader(velocity_shd.velocityInstancedShaderDesc(backend));
        if (sg.queryShaderState(out.inst_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        out.skin_shader = sg.makeShader(velocity_shd.velocitySkinnedShaderDesc(backend));
        if (sg.queryShaderState(out.skin_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        // Pipelines stay unbuilt here: their depth format must match the
        // borrowed main depth attachment, which is only known at render
        // time (see ensurePipelinesFor).
        return out;
    }

    /// Builds (or rebuilds on format change) the six pipelines for the
    /// borrowed main-depth format. Locals-first like everywhere else: the
    /// working set is preserved when a rebuild fails. All pipelines share
    /// the shared-depth plan (writes off, EQUAL compare — see
    /// velocityDepthState).
    fn ensurePipelinesFor(self: *VelocityPass, depth_fmt: sg.PixelFormat) bool {
        if (!sg.isvalid()) return false;
        if (depth_fmt == .DEFAULT or depth_fmt == .NONE) return false;
        if (self.depth_format == depth_fmt and
            sg.queryPipelineState(self.pip_rigid_u16) == .VALID and
            sg.queryPipelineState(self.pip_rigid_u32) == .VALID and
            sg.queryPipelineState(self.pip_inst_u16) == .VALID and
            sg.queryPipelineState(self.pip_inst_u32) == .VALID and
            sg.queryPipelineState(self.pip_skin_u16) == .VALID and
            sg.queryPipelineState(self.pip_skin_u32) == .VALID)
        {
            return true;
        }
        if (self.rigid_shader.id == 0 or self.inst_shader.id == 0 or self.skin_shader.id == 0) return false;

        var r_desc = sg.PipelineDesc{
            .shader = self.rigid_shader,
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = depth_fmt,
                .compare = .EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        r_desc.colors[0].pixel_format = .RGBA16F;
        r_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        r_desc.layout.attrs[velocity_shd.ATTR_velocity_position] = .{
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        r_desc.layout.attrs[velocity_shd.ATTR_velocity_color0] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "color"),
        };
        r_desc.layout.attrs[velocity_shd.ATTR_velocity_texcoord0] = .{
            .format = .FLOAT2,
            .offset = @offsetOf(Vertex, "uv"),
        };
        r_desc.layout.attrs[velocity_shd.ATTR_velocity_texcoord1] = .{
            .format = .FLOAT2,
            .offset = @offsetOf(Vertex, "uv1"),
        };
        const pip_r_u16 = sg.makePipeline(r_desc);
        r_desc.index_type = .UINT32;
        const pip_r_u32 = sg.makePipeline(r_desc);

        var i_desc = sg.PipelineDesc{
            .shader = self.inst_shader,
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = depth_fmt,
                .compare = .EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        i_desc.colors[0].pixel_format = .RGBA16F;
        i_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_position] = .{
            .buffer_index = 0,
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_color0] = .{
            .buffer_index = 0,
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "color"),
        };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_texcoord0] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = @offsetOf(Vertex, "uv"),
        };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_texcoord1] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = @offsetOf(Vertex, "uv1"),
        };
        i_desc.layout.buffers[1] = .{
            .step_func = .PER_INSTANCE,
            .step_rate = 1,
            .stride = @sizeOf(Mat4),
        };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };

        i_desc.layout.buffers[2] = .{
            .step_func = .PER_INSTANCE,
            .step_rate = 1,
            .stride = @sizeOf(Mat4),
        };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat0] = .{ .buffer_index = 2, .offset = 0, .format = .FLOAT4 };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat1] = .{ .buffer_index = 2, .offset = 16, .format = .FLOAT4 };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat2] = .{ .buffer_index = 2, .offset = 32, .format = .FLOAT4 };
        i_desc.layout.attrs[velocity_shd.ATTR_velocity_instanced_prev_inst_mat3] = .{ .buffer_index = 2, .offset = 48, .format = .FLOAT4 };

        const pip_i_u16 = sg.makePipeline(i_desc);
        i_desc.index_type = .UINT32;
        const pip_i_u32 = sg.makePipeline(i_desc);

        var s_desc = sg.PipelineDesc{
            .shader = self.skin_shader,
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = depth_fmt,
                .compare = .EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        s_desc.colors[0].pixel_format = .RGBA16F;
        s_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_position] = .{
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_joints] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "joints"),
        };
        s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_weights] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "weights"),
        };
        s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_color0] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "color"),
        };
        s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_texcoord0] = .{
            .format = .FLOAT2,
            .offset = @offsetOf(Vertex, "uv"),
        };
        s_desc.layout.attrs[velocity_shd.ATTR_velocity_skinned_texcoord1] = .{
            .format = .FLOAT2,
            .offset = @offsetOf(Vertex, "uv1"),
        };
        const pip_s_u16 = sg.makePipeline(s_desc);
        s_desc.index_type = .UINT32;
        const pip_s_u32 = sg.makePipeline(s_desc);

        if (sg.queryPipelineState(pip_r_u16) != .VALID or
            sg.queryPipelineState(pip_r_u32) != .VALID or
            sg.queryPipelineState(pip_i_u16) != .VALID or
            sg.queryPipelineState(pip_i_u32) != .VALID or
            sg.queryPipelineState(pip_s_u16) != .VALID or
            sg.queryPipelineState(pip_s_u32) != .VALID)
        {
            // Roll back every partial handle; the previous working set
            // (if any) stays installed for the next attempt.
            if (pip_r_u16.id != 0) sg.destroyPipeline(pip_r_u16);
            if (pip_r_u32.id != 0) sg.destroyPipeline(pip_r_u32);
            if (pip_i_u16.id != 0) sg.destroyPipeline(pip_i_u16);
            if (pip_i_u32.id != 0) sg.destroyPipeline(pip_i_u32);
            if (pip_s_u16.id != 0) sg.destroyPipeline(pip_s_u16);
            if (pip_s_u32.id != 0) sg.destroyPipeline(pip_s_u32);
            return false;
        }
        self.destroyPipelines();
        self.pip_rigid_u16 = pip_r_u16;
        self.pip_rigid_u32 = pip_r_u32;
        self.pip_inst_u16 = pip_i_u16;
        self.pip_inst_u32 = pip_i_u32;
        self.pip_skin_u16 = pip_s_u16;
        self.pip_skin_u32 = pip_s_u32;
        self.depth_format = depth_fmt;
        return true;
    }

    fn destroyPipelines(self: *VelocityPass) void {
        if (self.pip_rigid_u16.id != 0) sg.destroyPipeline(self.pip_rigid_u16);
        if (self.pip_rigid_u32.id != 0) sg.destroyPipeline(self.pip_rigid_u32);
        if (self.pip_inst_u16.id != 0) sg.destroyPipeline(self.pip_inst_u16);
        if (self.pip_inst_u32.id != 0) sg.destroyPipeline(self.pip_inst_u32);
        if (self.pip_skin_u16.id != 0) sg.destroyPipeline(self.pip_skin_u16);
        if (self.pip_skin_u32.id != 0) sg.destroyPipeline(self.pip_skin_u32);
        self.pip_rigid_u16 = .{};
        self.pip_rigid_u32 = .{};
        self.pip_inst_u16 = .{};
        self.pip_inst_u32 = .{};
        self.pip_skin_u16 = .{};
        self.pip_skin_u32 = .{};
        self.depth_format = .DEFAULT;
    }

    pub fn deinit(self: *VelocityPass) void {
        self.destroyTargets();
        if (self.dummy_zero_view.id != 0) sg.destroyView(self.dummy_zero_view);
        if (self.dummy_zero_image.id != 0) sg.destroyImage(self.dummy_zero_image);
        if (self.albedo_fallback_sampler.id != 0) sg.destroySampler(self.albedo_fallback_sampler);
        self.destroyPipelines();
        if (self.rigid_shader.id != 0) sg.destroyShader(self.rigid_shader);
        if (self.inst_shader.id != 0) sg.destroyShader(self.inst_shader);
        if (self.skin_shader.id != 0) sg.destroyShader(self.skin_shader);
        self.* = .{};
    }

    pub fn destroyTargets(self: *VelocityPass) void {
        // Own color target only. The main depth attachment is borrowed per
        // render (never owned, never destroyed here).
        if (self.tex_view.id != 0) sg.destroyView(self.tex_view);
        if (self.att_view.id != 0) sg.destroyView(self.att_view);
        if (self.image.id != 0) sg.destroyImage(self.image);
        self.image = .{};
        self.att_view = .{};
        self.tex_view = .{};
        self.width = 0;
        self.height = 0;
    }

    pub fn ensure(self: *VelocityPass, width: i32, height: i32) bool {
        if (width <= 0 or height <= 0) return false;
        if (!sg.isvalid()) return false;
        if (self.width == width and self.height == height and self.image.id != 0) {
            // Match on the PUBLISHED size, but validate the whole shape,
            // not just the image: views can fail independently of it, and a
            // stale size with dead views must never read as success (the
            // caller gates the render and the composite view on this bool).
            if (sg.queryImageState(self.image) != .VALID or
                sg.queryViewState(self.att_view) != .VALID or
                sg.queryViewState(self.tex_view) != .VALID)
            {
                self.destroyTargets();
                return false;
            }
            return true;
        }

        // Build the color target into locals and validate every handle
        // (image AND views) before touching self: any failure destroys the
        // partial locals and preserves the old valid target. The depth side
        // is borrowed per render from the main target (no private depth).
        // Never publish width/height for a half-created target.
        const img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = .RGBA16F,
            .sample_count = 1,
        });
        if (sg.queryImageState(img) != .VALID) {
            if (img.id != 0) sg.destroyImage(img);
            return false;
        }
        const att_view = sg.makeView(.{
            .color_attachment = .{ .image = img },
        });
        const tex_view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        if (sg.queryViewState(att_view) != .VALID or sg.queryViewState(tex_view) != .VALID) {
            if (tex_view.id != 0) sg.destroyView(tex_view);
            if (att_view.id != 0) sg.destroyView(att_view);
            sg.destroyImage(img);
            return false;
        }

        self.destroyTargets();
        self.image = img;
        self.att_view = att_view;
        self.tex_view = tex_view;
        self.width = width;
        self.height = height;
        return true;
    }

    pub fn velocityTexView(self: *const VelocityPass) sg.View {
        if (self.tex_view.id != 0) return self.tex_view;
        return self.dummy_zero_view;
    }

    pub fn render(
        self: *VelocityPass,
        cur_view_proj: Mat4,
        prev_view_proj: Mat4,
        queues: *const RenderQueues,
        skins: []const [scene_render_queue.MAX_BONES]Mat4,
        prev_skins: []const [scene_render_queue.MAX_BONES]Mat4,
        main_depth_att_view: sg.View,
        main_depth_format: sg.PixelFormat,
        last_rendered: u64,
        stats: *SceneStats,
    ) void {
        if (!sg.isvalid()) return;
        // Own color target plus the BORROWED main depth attachment (never
        // owned or destroyed here): every handle revalidated, since a
        // failed ensure must never leave a stale-sized target samplable.
        if (self.att_view.id == 0 or main_depth_att_view.id == 0) return;
        if (sg.queryViewState(self.att_view) != .VALID) return;
        if (sg.queryViewState(main_depth_att_view) != .VALID) return;
        if (!self.ensurePipelinesFor(main_depth_format)) return;

        // Shared-depth plan (see velocityDepthPassAction): own color is
        // cleared to the zero mask; main depth is LOADED and STORED with
        // writes off — nearer main pixels (morph displacement, opaque
        // surfaces) occlude farther velocity draws, and cutout holes (main
        // depth holds the farther background) fail the EQUAL match instead
        // of writing false mask1. No depth clear: the post chain samples
        // this same depth afterwards.
        var pass = sg.Pass{ .action = velocityDepthPassAction() };
        pass.attachments.colors[0] = self.att_view;
        pass.attachments.depth_stencil = main_depth_att_view;
        sg.beginPass(pass);

        var draws: u32 = 0;
        for (queues.items.items) |item| {
            if (self.drawRegularOne(cur_view_proj, prev_view_proj, item, skins, prev_skins, last_rendered)) draws += 1;
        }
        for (queues.opaque_instanced.items) |batch| {
            if (self.drawInstancedOne(cur_view_proj, prev_view_proj, batch, last_rendered)) draws += 1;
        }

        sg.endPass();
        stats.main_draw_calls += draws;
        stats.draw_calls += draws;
    }

    fn drawRegularOne(
        self: *VelocityPass,
        cur_view_proj: Mat4,
        prev_view_proj: Mat4,
        item: scene_render_queue.RenderMeshItem,
        skins: []const [scene_render_queue.MAX_BONES]Mat4,
        prev_skins: []const [scene_render_queue.MAX_BONES]Mat4,
        last_rendered: u64,
    ) bool {
        // Sg-free skip matrix first (see velocityDrawDecision): mask0
        // fallbacks never reach handle validation, let alone draws.
        if (velocityDrawDecision(.{
            .transparent = item.transparent,
            .is_decal = item.is_decal,
            .has_hook = item.shader_index != null,
            .morph_fallback = item.velocity_depth_fallback,
            .index_count = item.index_count,
        }) != .draw) return false;
        if (item.vertex_buffer.id == 0 or sg.queryBufferState(item.vertex_buffer) != .VALID) return false;
        if (item.index_buffer.id != 0 and sg.queryBufferState(item.index_buffer) != .VALID) return false;

        // Generation gate: frozen prev state is usable only when it comes
        // from the last actually rendered frame. Anything staged but never
        // presented (or replayed by a reuse) collapses to zero motion here,
        // never to an adopted unrendered pose.
        const use_prev = usePresentedPrev(item.prev_frame, last_rendered);
        const prev_model = if (use_prev) item.prev_model else item.model;
        const cur_mvp = Mat4.mul(cur_view_proj, item.model);
        const prev_mvp = Mat4.mul(prev_view_proj, prev_model);

        if (item.is_skinned) {
            const bones = scene_render_queue.skinAt(skins, item.skin_index) orelse return false;
            const prev_bones = if (usePresentedPrev(item.skin_prev_frame, last_rendered))
                scene_render_queue.skinAt(prev_skins, item.skin_index) orelse bones
            else
                bones;
            const pip = if (item.is_u32) self.pip_skin_u32 else self.pip_skin_u16;
            if (sg.queryPipelineState(pip) != .VALID) return false;
            sg.applyPipeline(pip);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = item.vertex_buffer;
            bind.index_buffer = item.index_buffer;
            bind.views[velocity_shd.VIEW_albedo_tex] = if (item.draw_record.albedo_view.id != 0)
                item.draw_record.albedo_view
            else
                self.dummy_zero_view;
            bind.samplers[velocity_shd.SMP_albedo_smp] = if (item.draw_record.albedo_sampler.id != 0)
                item.draw_record.albedo_sampler
            else
                self.albedo_fallback_sampler;
            sg.applyBindings(bind);
            const vs_params = velocity_shd.VsParams{
                .cur_mvp = cur_mvp,
                .prev_mvp = prev_mvp,
            };
            sg.applyUniforms(velocity_shd.UB_vs_params, sg.asRange(&vs_params));
            const vs_skin = velocity_shd.VsSkin{ .bones = bones.* };
            sg.applyUniforms(velocity_shd.UB_vs_skin, sg.asRange(&vs_skin));
            const vs_prev_skin = velocity_shd.VsPrevSkin{ .prev_bones = prev_bones.* };
            sg.applyUniforms(velocity_shd.UB_vs_prev_skin, sg.asRange(&vs_prev_skin));
            const alpha = velocityAlphaUniforms(item.draw_record);
            sg.applyUniforms(velocity_shd.UB_fs_alpha, sg.asRange(&alpha));
        } else {
            const pip = if (item.is_u32) self.pip_rigid_u32 else self.pip_rigid_u16;
            if (sg.queryPipelineState(pip) != .VALID) return false;
            sg.applyPipeline(pip);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = item.vertex_buffer;
            bind.index_buffer = item.index_buffer;
            bind.views[velocity_shd.VIEW_albedo_tex] = if (item.draw_record.albedo_view.id != 0)
                item.draw_record.albedo_view
            else
                self.dummy_zero_view;
            bind.samplers[velocity_shd.SMP_albedo_smp] = if (item.draw_record.albedo_sampler.id != 0)
                item.draw_record.albedo_sampler
            else
                self.albedo_fallback_sampler;
            sg.applyBindings(bind);
            const vs_params = velocity_shd.VsParams{
                .cur_mvp = cur_mvp,
                .prev_mvp = prev_mvp,
            };
            sg.applyUniforms(velocity_shd.UB_vs_params, sg.asRange(&vs_params));
            const alpha = velocityAlphaUniforms(item.draw_record);
            sg.applyUniforms(velocity_shd.UB_fs_alpha, sg.asRange(&alpha));
        }
        sg.draw(item.base_vertex, item.index_count, 1);
        return true;
    }

    fn drawInstancedOne(
        self: *VelocityPass,
        cur_view_proj: Mat4,
        prev_view_proj: Mat4,
        batch: scene_render_queue.RenderInstancedBatch,
        last_rendered: u64,
    ) bool {
        if (velocityDrawDecision(.{
            .transparent = batch.transparent,
            .is_decal = batch.is_decal,
            .index_count = batch.index_count,
        }) != .draw) return false;
        if (batch.visible_instance_count == 0 or batch.instance_buffer.id == 0) return false;
        if (batch.vertex_buffer.id == 0 or sg.queryBufferState(batch.vertex_buffer) != .VALID) return false;
        if (batch.index_buffer.id != 0 and sg.queryBufferState(batch.index_buffer) != .VALID) return false;
        if (sg.queryBufferState(batch.instance_buffer) != .VALID) return false;

        // Same generation gate as regular draws: an unmatched prev buffer
        // aliases cur (zero motion) instead of pairing unrendered history.
        const prev_buf = if (usePresentedPrev(batch.prev_frame, last_rendered) and
            batch.prev_instance_buffer.id != 0 and
            sg.queryBufferState(batch.prev_instance_buffer) == .VALID)
            batch.prev_instance_buffer
        else
            batch.instance_buffer;

        const pip = if (batch.index_type == .UINT32) self.pip_inst_u32 else self.pip_inst_u16;
        if (sg.queryPipelineState(pip) != .VALID) return false;
        sg.applyPipeline(pip);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = batch.vertex_buffer;
        bind.vertex_buffers[1] = batch.instance_buffer;
        bind.vertex_buffers[2] = prev_buf;
        bind.index_buffer = batch.index_buffer;
        bind.views[velocity_shd.VIEW_albedo_tex] = if (batch.draw_record.albedo_view.id != 0)
            batch.draw_record.albedo_view
        else
            self.dummy_zero_view;
        bind.samplers[velocity_shd.SMP_albedo_smp] = if (batch.draw_record.albedo_sampler.id != 0)
            batch.draw_record.albedo_sampler
        else
            self.albedo_fallback_sampler;
        sg.applyBindings(bind);
        const vs_params = velocity_shd.VsInstParams{
            .cur_view_proj = cur_view_proj,
            .prev_view_proj = prev_view_proj,
        };
        sg.applyUniforms(velocity_shd.UB_vs_inst_params, sg.asRange(&vs_params));
        const alpha = velocityAlphaUniforms(batch.draw_record);
        sg.applyUniforms(velocity_shd.UB_fs_alpha, sg.asRange(&alpha));
        sg.draw(0, batch.index_count, batch.visible_instance_count);
        return true;
    }
};

test "velocityNeeded mirrors the composite MSAA gate" {
    // TAA runs only at 1x (taa_on pre-gated in frame_render): always needs it.
    try std.testing.expect(velocityNeeded(true, false, 1));
    try std.testing.expect(velocityNeeded(true, true, 1));
    // Nothing enabled: no target.
    try std.testing.expect(!velocityNeeded(false, false, 1));
    // Blur at 1x: needed.
    try std.testing.expect(velocityNeeded(false, true, 1));
    // Blur under MSAA: explicitly suppressed even with the depth prepass —
    // the blur falls back to camera depth reprojection, so no
    // sample-mismatched velocity target may feed it false vectors.
    try std.testing.expect(!velocityNeeded(false, true, 4));
    try std.testing.expect(!velocityNeeded(false, true, 2));
}

test "cutSuppressesVelocity fires exactly on reset/cut frames" {
    // Steady state: never suppress (no per-frame reset).
    try std.testing.expect(!cutSuppressesVelocity(false, false, false));
    try std.testing.expect(!cutSuppressesVelocity(false, true, false));
    // Any TAA reset suppresses, with or without blur.
    try std.testing.expect(cutSuppressesVelocity(true, false, false));
    try std.testing.expect(cutSuppressesVelocity(true, true, true));
    // Blur-only mode still suppresses on a camera cut.
    try std.testing.expect(cutSuppressesVelocity(false, true, true));
    // Blur disabled + no TAA reset: a cut flag alone changes nothing.
    try std.testing.expect(!cutSuppressesVelocity(false, false, true));
}

test "usePresentedPrev matches only the last rendered generation" {
    const never = std.math.maxInt(u64);
    // Never staged never matches — not even a zero last-rendered.
    try std.testing.expect(!usePresentedPrev(never, 0));
    try std.testing.expect(!usePresentedPrev(never, 41));
    // Staged-but-unrendered (latch without render, superseded front).
    try std.testing.expect(!usePresentedPrev(42, 41));
    // Reused front replaying old motion: prev gen older than the render.
    try std.testing.expect(!usePresentedPrev(39, 41));
    // Steady state: exact match pairs.
    try std.testing.expect(usePresentedPrev(41, 41));
    // Zero is a valid empty generation only when both are zero — and a
    // real staged frame never carries 0 (slots start at 0 = empty, and
    // the commit skips empty fronts).
    try std.testing.expect(usePresentedPrev(0, 0));
    try std.testing.expect(!usePresentedPrev(0, 7));
}

test "headless pass stays zero: init, ensure, render, tex view" {
    // No sg context here: every entry point must fail closed with no sg
    // calls beyond the isvalid probe and no state published.
    var pass = VelocityPass.init();
    try std.testing.expectEqual(sg.Image{}, pass.image);
    try std.testing.expectEqual(sg.View{}, pass.tex_view);
    try std.testing.expectEqual(@as(i32, 0), pass.width);
    try std.testing.expectEqual(@as(i32, 0), pass.height);

    try std.testing.expect(!pass.ensure(64, 64));
    try std.testing.expectEqual(@as(i32, 0), pass.width);
    try std.testing.expect(pass.image.id == 0);

    try std.testing.expect(!pass.ensure(0, 64));
    try std.testing.expect(!pass.ensure(-1, 64));

    // Zero target reports the zero dummy (also zero headless).
    try std.testing.expectEqual(@as(u32, 0), pass.velocityTexView().id);

    // Render with empty queues is a guarded no-op: stats untouched.
    var queues = RenderQueues{};
    defer queues.deinit(std.testing.allocator);
    var stats = SceneStats{};
    const skins = [_][scene_render_queue.MAX_BONES]Mat4{};
    pass.render(Mat4.identity, Mat4.identity, &queues, &skins, &skins, .{}, .DEFAULT, 0, &stats);
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), stats.main_draw_calls);

    pass.deinit();
    try std.testing.expectEqual(@as(i32, 0), pass.width);
}

test "shared-depth plan binds visible main pixels, never clears depth" {
    // Pipeline depth state: writes off, EQUAL compare (see
    // velocityDepthState docs for the morph/cutout occlusion meaning).
    const ds = velocityDepthState();
    try std.testing.expect(!ds.write_enabled);
    try std.testing.expectEqual(sg.CompareFunc.EQUAL, ds.compare);

    // Pass action: own color cleared to the zero mask; borrowed main depth
    // loaded and stored (post chain samples it right after — no clear, no
    // dontcare).
    const pa = velocityDepthPassAction();
    try std.testing.expectEqual(sg.LoadAction.CLEAR, pa.colors[0].load_action);
    try std.testing.expectEqual(sg.StoreAction.STORE, pa.colors[0].store_action);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), pa.colors[0].clear_value.r, 1e-9);
    try std.testing.expectEqual(@as(f32, 0.0), pa.colors[0].clear_value.a);
    try std.testing.expectEqual(sg.LoadAction.LOAD, pa.depth.load_action);
    try std.testing.expectEqual(sg.StoreAction.STORE, pa.depth.store_action);

    // Occlusion meaning, pinned headless (no pixels asserted — the GPU
    // gate proves those): with EQUAL + write-off against borrowed main
    // depth, a nearer main surface (morph-displaced pixel, opaque pixel)
    // rejects a farther velocity fragment, so the target keeps the zero
    // mask (depth fallback) instead of adopting background motion; a
    // matching visible pixel passes. Cutout holes carry the farther
    // background depth and fail the match — no alpha texture needed except
    // for truly coplanar cutouts. The per-draw side of that contract is
    // velocityDrawDecision below (mask0 skips vs draws).
    try std.testing.expectEqual(.draw, velocityDrawDecision(.{ .index_count = 3 }));
    try std.testing.expectEqual(.skip_mask0, velocityDrawDecision(.{ .index_count = 3, .morph_fallback = true }));
    try std.testing.expectEqual(.skip_mask0, velocityDrawDecision(.{ .index_count = 3, .transparent = true }));
    try std.testing.expectEqual(.skip_mask0, velocityDrawDecision(.{ .index_count = 3, .is_decal = true }));
    try std.testing.expectEqual(.skip_mask0, velocityDrawDecision(.{ .index_count = 3, .has_hook = true }));
    try std.testing.expectEqual(.skip_mask0, velocityDrawDecision(.{ .index_count = 0 }));
    // Cutout (alpha-tested, opaque) DRAWS: its holes fail the EQUAL match
    // against main depth instead of needing an in-shader discard.
    try std.testing.expectEqual(.draw, velocityDrawDecision(.{ .index_count = 3, .is_cutout = true }));
}

/// Sg-free prefix of the per-draw skip matrix (single source of truth for
/// drawRegularOne/drawInstancedOne and these tests): which draws may emit
/// mask1 vs which must stay mask0 (depth-reprojection fallback). GPU
/// morphs have no velocity path (base-pose matrices would lie);
/// transparent/decal/hook draws never contribute depth; empty draws emit
/// nothing. Cutout draws DO emit — their holes resolve through the shared
/// EQUAL depth match, not an alpha discard.
pub const VelocityDraw = enum { draw, skip_mask0 };

pub const VelocityDrawInput = struct {
    transparent: bool = false,
    is_decal: bool = false,
    has_hook: bool = false,
    morph_fallback: bool = false,
    is_cutout: bool = false,
    index_count: u32 = 0,
};

pub fn velocityDrawDecision(in: VelocityDrawInput) VelocityDraw {
    if (in.index_count == 0) return .skip_mask0;
    if (in.morph_fallback) return .skip_mask0;
    if (!itemContributesDepth(in.transparent, in.is_decal, in.has_hook)) return .skip_mask0;
    return .draw;
}

/// Headless statement of the occlusion/mask routing the shared-depth pass
/// implements per pixel (no pixels asserted — the GPU gate proves those):
/// drawn fragments need a depth match against main depth; anything else
/// (skipped morph/cutout-transparent/hook draws, depth mismatch) stays
/// mask0 and falls back to camera depth reprojection.
fn velocity_pass_routing(depth_match: bool, vel_depth: f32, main_depth: f32) bool {
    if (!depth_match) return false;
    _ = vel_depth;
    _ = main_depth;
    return true;
}
