//! Weighted Blended Order-Independent Transparency (WBOIT) pass.
//!
//! Implements McGuire & Bavoil 2013 ("Weighted Blended Order-Independent Transparency")
//! for pixel-perfect transparency without CPU depth sorting, eliminating sorting popping
//! and correctly handling intersecting, nested, and layered transparent geometries.
//!
//! Pipeline Architecture:
//! 1. Accumulation Pass:
//!    Transparent surfaces render into two render targets (MRT) without depth write:
//!    - Target 0 (Accumulation, RGBA16F): accumulates weighted lit color (rgb) and coverage weight (a).
//!      Blend state: ONE, ONE (additive).
//!    - Target 1 (Revealage, RGBA8): accumulates background visibility product prod(1 - alpha_i).
//!      Blend state: ZERO, ONE_MINUS_SRC_COLOR (multiplicative).
//!    - Depth: borrows the opaque scene depth buffer with LESS_EQUAL and depth write disabled.
//! 2. Resolve Pass:
//!    A fullscreen quad samples accumulation and revealage targets and composites
//!    the weighted transparent layer over the main HDR scene color attachment using
//!    standard alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA).
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;

const Vertex = @import("../mesh.zig").Vertex;
const render_queue = @import("../scene/render_queue.zig");
const RenderMeshItem = render_queue.RenderMeshItem;
const RenderInstancedBatch = render_queue.RenderInstancedBatch;
const RenderQueues = render_queue.RenderQueues;
const MAX_BONES = @import("../animation/skeleton.zig").MAX_BONES;

const scene_draw = @import("../scene/draw.zig");
const Environment = scene_draw.Environment;
const FrameContext = scene_draw.FrameContext;
const uniforms = @import("../scene/uniforms.zig");
const material_mod = @import("../material.zig");
const gpu_thread = @import("../gpu_thread.zig");
const SceneFrameSnapshot = @import("../scene/snapshot.zig").SceneFrameSnapshot;

const wboit_shd = @import("wboit_shader");
const wboit_resolve_shd = @import("wboit_resolve_shader");

const Viewport = @import("../camera/viewport.zig").Viewport;
const PixelRect = Viewport.PixelRect;

pub const TransparencyMode = enum {
    sorted,
    wboit,
};

pub const WboitOptions = struct {
    enabled: bool = false,
    depth_scale: f32 = 200.0,
};

// ---------------------------------------------------------------------------
// CPU mirrors for mathematical golden testing
// ---------------------------------------------------------------------------

/// Computes the depth-aware weight function w(z, alpha) per McGuire & Bavoil 2013 Eq. 10.
pub fn wboitWeight(z: f32, alpha: f32) f32 {
    const a = std.math.clamp(alpha, 0.0, 1.0);
    const z_scaled = @abs(z) / 200.0;
    const z4 = z_scaled * z_scaled * z_scaled * z_scaled;
    const w = 0.03 / (1e-5 + z4);
    return std.math.clamp(w, 0.01, 3000.0) * a;
}

/// Accumulates a single transparent sample into the WBOIT accumulation buffer (RGBA).
pub fn wboitAccumulate(color: [3]f32, alpha: f32, z: f32) [4]f32 {
    const a = std.math.clamp(alpha, 0.0, 1.0);
    const w = wboitWeight(z, a);
    const aw = a * w;
    return .{
        color[0] * aw,
        color[1] * aw,
        color[2] * aw,
        aw,
    };
}

/// Reconstructs the composite transparent color blended onto the background color.
pub fn wboitComposite(accum: [4]f32, reveal: f32, background: [3]f32) [3]f32 {
    const r = std.math.clamp(reveal, 0.0, 1.0);
    if (r >= 0.99999) return background;
    const denom = @max(accum[3], 0.00001);
    const avg_r = std.math.clamp(accum[0] / denom, 0.0, 65500.0);
    const avg_g = std.math.clamp(accum[1] / denom, 0.0, 65500.0);
    const avg_b = std.math.clamp(accum[2] / denom, 0.0, 65500.0);
    const alpha = std.math.clamp(1.0 - r, 0.0, 1.0);
    return .{
        avg_r * alpha + background[0] * (1.0 - alpha),
        avg_g * alpha + background[1] * (1.0 - alpha),
        avg_b * alpha + background[2] * (1.0 - alpha),
    };
}

// ---------------------------------------------------------------------------
// WboitPass
// ---------------------------------------------------------------------------

pub const WboitPass = struct {
    // Render Targets
    accum_image: sg.Image = .{},
    accum_att_view: sg.View = .{},
    accum_tex_view: sg.View = .{},

    reveal_image: sg.Image = .{},
    reveal_att_view: sg.View = .{},
    reveal_tex_view: sg.View = .{},

    width: i32 = 0,
    height: i32 = 0,

    // Shared Fullscreen Quad
    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},
    sampler: sg.Sampler = .{},

    // Shaders
    shd_rigid: sg.Shader = .{},
    shd_inst: sg.Shader = .{},
    shd_skin: sg.Shader = .{},
    shd_resolve: sg.Shader = .{},

    // Pipelines
    depth_format: sg.PixelFormat = .DEFAULT,

    pip_rigid_u16: sg.Pipeline = .{},
    pip_rigid_u32: sg.Pipeline = .{},
    pip_rigid_u16_ds: sg.Pipeline = .{},
    pip_rigid_u32_ds: sg.Pipeline = .{},

    pip_inst_u16: sg.Pipeline = .{},
    pip_inst_u32: sg.Pipeline = .{},
    pip_inst_u16_ds: sg.Pipeline = .{},
    pip_inst_u32_ds: sg.Pipeline = .{},

    pip_skin_u16: sg.Pipeline = .{},
    pip_skin_u32: sg.Pipeline = .{},
    pip_skin_u16_ds: sg.Pipeline = .{},
    pip_skin_u32_ds: sg.Pipeline = .{},

    pip_resolve: sg.Pipeline = .{},

    pub fn init() WboitPass {
        if (!sg.isvalid()) return .{};

        const quad_vertices = [_]f32{
            -1.0, -1.0, 0.0, 0.0,
            1.0,  -1.0, 1.0, 0.0,
            1.0,  1.0,  1.0, 1.0,
            -1.0, 1.0,  0.0, 1.0,
        };
        const quad_indices = [_]u16{
            0, 1, 2,
            0, 2, 3,
        };

        const vb = sg.makeBuffer(.{
            .data = sg.asRange(&quad_vertices),
        });
        const ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&quad_indices),
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const backend = sg.queryBackend();
        const shd_rigid = sg.makeShader(wboit_shd.wboitRigidShaderDesc(backend));
        const shd_inst = sg.makeShader(wboit_shd.wboitInstShaderDesc(backend));
        const shd_skin = sg.makeShader(wboit_shd.wboitSkinShaderDesc(backend));
        const shd_resolve = sg.makeShader(wboit_resolve_shd.wboitResolveShaderDesc(backend));

        return .{
            .quad_vb = vb,
            .quad_ib = ib,
            .sampler = smp,
            .shd_rigid = shd_rigid,
            .shd_inst = shd_inst,
            .shd_skin = shd_skin,
            .shd_resolve = shd_resolve,
        };
    }

    pub fn deinit(self: *WboitPass) void {
        self.destroyTargets();
        self.destroyPipelines();

        if (self.shd_rigid.id != 0 and sg.isvalid()) sg.destroyShader(self.shd_rigid);
        if (self.shd_inst.id != 0 and sg.isvalid()) sg.destroyShader(self.shd_inst);
        if (self.shd_skin.id != 0 and sg.isvalid()) sg.destroyShader(self.shd_skin);
        if (self.shd_resolve.id != 0 and sg.isvalid()) sg.destroyShader(self.shd_resolve);
        if (self.quad_vb.id != 0 and sg.isvalid()) sg.destroyBuffer(self.quad_vb);
        if (self.quad_ib.id != 0 and sg.isvalid()) sg.destroyBuffer(self.quad_ib);
        if (self.sampler.id != 0 and sg.isvalid()) sg.destroySampler(self.sampler);

        self.* = .{};
    }

    pub fn destroyTargets(self: *WboitPass) void {
        if (!sg.isvalid()) {
            self.accum_image = .{};
            self.accum_att_view = .{};
            self.accum_tex_view = .{};
            self.reveal_image = .{};
            self.reveal_att_view = .{};
            self.reveal_tex_view = .{};
            self.width = 0;
            self.height = 0;
            return;
        }

        if (self.accum_att_view.id != 0) sg.destroyView(self.accum_att_view);
        if (self.accum_tex_view.id != 0) sg.destroyView(self.accum_tex_view);
        if (self.accum_image.id != 0) sg.destroyImage(self.accum_image);

        if (self.reveal_att_view.id != 0) sg.destroyView(self.reveal_att_view);
        if (self.reveal_tex_view.id != 0) sg.destroyView(self.reveal_tex_view);
        if (self.reveal_image.id != 0) sg.destroyImage(self.reveal_image);

        self.accum_image = .{};
        self.accum_att_view = .{};
        self.accum_tex_view = .{};
        self.reveal_image = .{};
        self.reveal_att_view = .{};
        self.reveal_tex_view = .{};
        self.width = 0;
        self.height = 0;
    }

    pub fn targetsValid(self: *const WboitPass) bool {
        if (!sg.isvalid()) return false;
        if (self.width <= 0 or self.height <= 0) return false;
        return sg.queryImageState(self.accum_image) == .VALID and
            sg.queryImageState(self.reveal_image) == .VALID and
            sg.queryViewState(self.accum_att_view) == .VALID and
            sg.queryViewState(self.accum_tex_view) == .VALID and
            sg.queryViewState(self.reveal_att_view) == .VALID and
            sg.queryViewState(self.reveal_tex_view) == .VALID;
    }

    pub fn ensure(self: *WboitPass, width: i32, height: i32) bool {
        if (!sg.isvalid() or width <= 0 or height <= 0) return false;
        if (self.width == width and self.height == height and self.targetsValid()) return true;

        self.destroyTargets();

        // 1. Accumulation buffer (RGBA16F)
        const accum_img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = .RGBA16F,
            .sample_count = 1,
        });
        if (sg.queryImageState(accum_img) != .VALID) {
            if (accum_img.id != 0) sg.destroyImage(accum_img);
            return false;
        }

        const accum_att = sg.makeView(.{
            .color_attachment = .{ .image = accum_img },
        });
        const accum_tex = sg.makeView(.{
            .texture = .{ .image = accum_img },
        });

        if (!gpu_thread.createdViewIsValid(accum_att) or !gpu_thread.createdViewIsValid(accum_tex)) {
            if (accum_tex.id != 0) sg.destroyView(accum_tex);
            if (accum_att.id != 0) sg.destroyView(accum_att);
            sg.destroyImage(accum_img);
            return false;
        }

        // 2. Revealage buffer (RGBA8)
        const reveal_img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = .RGBA8,
            .sample_count = 1,
        });
        if (sg.queryImageState(reveal_img) != .VALID) {
            if (reveal_img.id != 0) sg.destroyImage(reveal_img);
            if (accum_tex.id != 0) sg.destroyView(accum_tex);
            if (accum_att.id != 0) sg.destroyView(accum_att);
            sg.destroyImage(accum_img);
            return false;
        }

        const reveal_att = sg.makeView(.{
            .color_attachment = .{ .image = reveal_img },
        });
        const reveal_tex = sg.makeView(.{
            .texture = .{ .image = reveal_img },
        });

        if (!gpu_thread.createdViewIsValid(reveal_att) or !gpu_thread.createdViewIsValid(reveal_tex)) {
            if (reveal_tex.id != 0) sg.destroyView(reveal_tex);
            if (reveal_att.id != 0) sg.destroyView(reveal_att);
            sg.destroyImage(reveal_img);
            if (accum_tex.id != 0) sg.destroyView(accum_tex);
            if (accum_att.id != 0) sg.destroyView(accum_att);
            sg.destroyImage(accum_img);
            return false;
        }

        self.accum_image = accum_img;
        self.accum_att_view = accum_att;
        self.accum_tex_view = accum_tex;
        self.reveal_image = reveal_img;
        self.reveal_att_view = reveal_att;
        self.reveal_tex_view = reveal_tex;
        self.width = width;
        self.height = height;
        return true;
    }

    fn destroyPipelines(self: *WboitPass) void {
        inline for (.{
            &self.pip_rigid_u16,
            &self.pip_rigid_u32,
            &self.pip_rigid_u16_ds,
            &self.pip_rigid_u32_ds,
            &self.pip_inst_u16,
            &self.pip_inst_u32,
            &self.pip_inst_u16_ds,
            &self.pip_inst_u32_ds,
            &self.pip_skin_u16,
            &self.pip_skin_u32,
            &self.pip_skin_u16_ds,
            &self.pip_skin_u32_ds,
            &self.pip_resolve,
        }) |pipe| {
            if (pipe.*.id != 0 and sg.isvalid()) sg.destroyPipeline(pipe.*);
            pipe.* = .{};
        }
        self.depth_format = .DEFAULT;
    }

    fn makeWboitDesc(shader: sg.Shader, depth_format: sg.PixelFormat, cull_mode: sg.CullMode, index_type: sg.IndexType) sg.PipelineDesc {
        var desc = sg.PipelineDesc{
            .shader = shader,
            .index_type = index_type,
            .depth = .{
                .pixel_format = depth_format,
                .compare = .LESS_EQUAL,
                .write_enabled = false,
            },
            .cull_mode = cull_mode,
            .face_winding = .CCW,
            .sample_count = 1,
            .color_count = 2,
        };
        // Color 0: accum (RGBA16F, additive blend)
        desc.colors[0].pixel_format = .RGBA16F;
        desc.colors[0].blend = .{
            .enabled = true,
            .src_factor_rgb = .ONE,
            .dst_factor_rgb = .ONE,
            .op_rgb = .ADD,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE,
            .op_alpha = .ADD,
        };
        // Color 1: reveal (RGBA8, multiplicative blend)
        desc.colors[1].pixel_format = .RGBA8;
        desc.colors[1].blend = .{
            .enabled = true,
            .src_factor_rgb = .ZERO,
            .dst_factor_rgb = .ONE_MINUS_SRC_COLOR,
            .op_rgb = .ADD,
            .src_factor_alpha = .ZERO,
            .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
            .op_alpha = .ADD,
        };
        return desc;
    }

    pub fn ensurePipelines(self: *WboitPass, depth_format: sg.PixelFormat) void {
        if (!sg.isvalid()) return;
        if (self.depth_format == depth_format and self.pip_rigid_u16.id != 0) return;

        self.destroyPipelines();
        self.depth_format = depth_format;

        // 1. Rigid pipelines
        inline for (.{
            .{ .p = &self.pip_rigid_u16, .cull = sg.CullMode.BACK, .idx = sg.IndexType.UINT16 },
            .{ .p = &self.pip_rigid_u32, .cull = sg.CullMode.BACK, .idx = sg.IndexType.UINT32 },
            .{ .p = &self.pip_rigid_u16_ds, .cull = sg.CullMode.NONE, .idx = sg.IndexType.UINT16 },
            .{ .p = &self.pip_rigid_u32_ds, .cull = sg.CullMode.NONE, .idx = sg.IndexType.UINT32 },
        }) |spec| {
            var desc = makeWboitDesc(self.shd_rigid, depth_format, spec.cull, spec.idx);
            desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
            desc.layout.attrs[wboit_shd.ATTR_wboit_rigid_position] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_rigid_normal] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_rigid_color0] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_rigid_texcoord0] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_rigid_texcoord1] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv1") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_rigid_tangent] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "tangent") };
            spec.p.* = sg.makePipeline(desc);
        }

        // 2. Instanced pipelines
        inline for (.{
            .{ .p = &self.pip_inst_u16, .cull = sg.CullMode.BACK, .idx = sg.IndexType.UINT16 },
            .{ .p = &self.pip_inst_u32, .cull = sg.CullMode.BACK, .idx = sg.IndexType.UINT32 },
            .{ .p = &self.pip_inst_u16_ds, .cull = sg.CullMode.NONE, .idx = sg.IndexType.UINT16 },
            .{ .p = &self.pip_inst_u32_ds, .cull = sg.CullMode.NONE, .idx = sg.IndexType.UINT32 },
        }) |spec| {
            var desc = makeWboitDesc(self.shd_inst, depth_format, spec.cull, spec.idx);
            desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
            desc.layout.attrs[wboit_shd.ATTR_wboit_inst_position] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_inst_normal] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_inst_tangent] = .{ .buffer_index = 0, .format = .FLOAT4, .offset = @offsetOf(Vertex, "tangent") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_inst_color0] = .{ .buffer_index = 0, .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_inst_texcoord0] = .{ .buffer_index = 0, .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_inst_texcoord1] = .{ .buffer_index = 0, .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv1") };

            desc.layout.buffers[1] = .{
                .step_func = .PER_INSTANCE,
                .step_rate = 1,
                .stride = 64,
            };
            desc.layout.attrs[wboit_shd.ATTR_wboit_inst_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
            desc.layout.attrs[wboit_shd.ATTR_wboit_inst_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
            desc.layout.attrs[wboit_shd.ATTR_wboit_inst_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
            desc.layout.attrs[wboit_shd.ATTR_wboit_inst_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };
            spec.p.* = sg.makePipeline(desc);
        }

        // 3. Skinned pipelines
        inline for (.{
            .{ .p = &self.pip_skin_u16, .cull = sg.CullMode.BACK, .idx = sg.IndexType.UINT16 },
            .{ .p = &self.pip_skin_u32, .cull = sg.CullMode.BACK, .idx = sg.IndexType.UINT32 },
            .{ .p = &self.pip_skin_u16_ds, .cull = sg.CullMode.NONE, .idx = sg.IndexType.UINT16 },
            .{ .p = &self.pip_skin_u32_ds, .cull = sg.CullMode.NONE, .idx = sg.IndexType.UINT32 },
        }) |spec| {
            var desc = makeWboitDesc(self.shd_skin, depth_format, spec.cull, spec.idx);
            desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
            desc.layout.attrs[wboit_shd.ATTR_wboit_skin_position] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_skin_normal] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_skin_color0] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_skin_texcoord0] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_skin_texcoord1] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv1") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_skin_tangent] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "tangent") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_skin_joints] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "joints") };
            desc.layout.attrs[wboit_shd.ATTR_wboit_skin_weights] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "weights") };
            spec.p.* = sg.makePipeline(desc);
        }

        // 4. Resolve pipeline
        var res_desc = sg.PipelineDesc{
            .shader = self.shd_resolve,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = depth_format,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        res_desc.colors[0].pixel_format = .RGBA16F;
        res_desc.colors[0].blend = .{
            .enabled = true,
            .src_factor_rgb = .SRC_ALPHA,
            .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
            .op_rgb = .ADD,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
            .op_alpha = .ADD,
        };
        res_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        res_desc.layout.attrs[wboit_resolve_shd.ATTR_wboit_resolve_position] = .{
            .format = .FLOAT2,
            .offset = 0,
        };
        res_desc.layout.attrs[wboit_resolve_shd.ATTR_wboit_resolve_texcoord0] = .{
            .format = .FLOAT2,
            .offset = 2 * @sizeOf(f32),
        };
        self.pip_resolve = sg.makePipeline(res_desc);
    }

    pub fn pipelineForRegularItem(self: *const WboitPass, item: anytype) u32 {
        const Raw = @TypeOf(item);
        const ItemType = switch (@typeInfo(Raw)) {
            .pointer => |p| p.child,
            else => Raw,
        };
        const is_u32 = blk: {
            if (@hasField(ItemType, "is_u32") and item.is_u32) break :blk true;
            if (@hasField(ItemType, "index_type") and item.index_type == .UINT32) break :blk true;
            break :blk false;
        };
        const is_skinned = blk: {
            if (@hasField(ItemType, "is_skinned") and item.is_skinned) break :blk true;
            if (@hasField(ItemType, "skin_index") and item.skin_index != null) break :blk true;
            break :blk false;
        };
        const double_sided = blk: {
            if (@hasField(ItemType, "double_sided") and item.double_sided) break :blk true;
            break :blk false;
        };

        if (is_skinned) {
            if (double_sided) {
                return if (is_u32) self.pip_skin_u32_ds.id else self.pip_skin_u16_ds.id;
            }
            return if (is_u32) self.pip_skin_u32.id else self.pip_skin_u16.id;
        }
        if (double_sided) {
            return if (is_u32) self.pip_rigid_u32_ds.id else self.pip_rigid_u16_ds.id;
        }
        return if (is_u32) self.pip_rigid_u32.id else self.pip_rigid_u16.id;
    }

    pub fn pipelineForInstancedMesh(self: *const WboitPass, is_u32: bool, double_sided: bool) u32 {
        if (double_sided) {
            return if (is_u32) self.pip_inst_u32_ds.id else self.pip_inst_u16_ds.id;
        }
        return if (is_u32) self.pip_inst_u32.id else self.pip_inst_u16.id;
    }

    pub fn drawRegularItem(
        self: *const WboitPass,
        env: *const Environment,
        item: *const RenderMeshItem,
        ctx: *const FrameContext,
        current_pipeline_id: *u32,
        skins: []const [MAX_BONES]Mat4,
        coats: []const material_mod.CoatParams,
    ) void {
        if (item.index_count == 0) return;
        if (sg.isvalid() and !gpu_thread.geometryValid(item.vertex_buffer, item.index_buffer)) return;

        const pip_id = self.pipelineForRegularItem(item);
        if (pip_id == 0) return;

        if (pip_id != current_pipeline_id.*) {
            sg.applyPipeline(.{ .id = pip_id });
            current_pipeline_id.* = pip_id;
            env.stats.pipeline_switches += 1;
        }

        const model = item.model;
        const mvp = Mat4.mul(ctx.view_proj, model);
        const rec = item.draw_record;

        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = item.vertex_buffer;
        bind.index_buffer = item.index_buffer;

        bind.views[wboit_shd.VIEW_albedo_tex] = rec.albedo_view;
        bind.views[wboit_shd.VIEW_normal_tex] = rec.normal_view;
        bind.views[wboit_shd.VIEW_metallic_roughness_tex] = rec.mr_view;
        bind.views[wboit_shd.VIEW_emissive_tex] = rec.emissive_view;
        bind.views[wboit_shd.VIEW_occlusion_tex] = rec.occlusion_view;
        bind.views[wboit_shd.VIEW_clearcoat_tex] = rec.clearcoat_view;
        bind.views[wboit_shd.VIEW_sheen_tex] = rec.sheen_view;
        bind.views[wboit_shd.VIEW_anisotropy_tex] = rec.anisotropy_view;
        bind.views[wboit_shd.VIEW_refraction_tex] = if (env.refraction_view.id != 0) env.refraction_view else env.default_white.view;
        bind.samplers[wboit_shd.SMP_refraction_smp] = if (env.refraction_sampler.id != 0) env.refraction_sampler else env.default_white.sampler;
        bind.samplers[wboit_shd.SMP_smp] = rec.albedo_sampler;
        bind.samplers[wboit_shd.SMP_data_smp] = rec.data_sampler;
        bind.views[wboit_shd.VIEW_brdf_lut_tex] = rec.brdf_lut_view;
        bind.samplers[wboit_shd.SMP_brdf_lut_smp] = rec.brdf_lut_sampler;

        const cube_view = rec.env_view orelse (env.sky_texture orelse env.default_cube).view;
        const cube_sampler = rec.env_sampler orelse (env.sky_texture orelse env.default_cube).sampler;
        bind.views[wboit_shd.VIEW_env_tex] = cube_view;
        bind.samplers[wboit_shd.SMP_env_smp] = cube_sampler;

        bind.views[wboit_shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
        bind.views[wboit_shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
        bind.views[wboit_shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
        bind.views[wboit_shd.VIEW_spot_shadow_depth_tex] = env.shadow_pass.spot_texture_view;
        bind.views[wboit_shd.VIEW_point_shadow_tex] = env.shadow_pass.point_texture_view;
        bind.views[wboit_shd.VIEW_point_shadow_depth_tex] = env.shadow_pass.point_texture_view;
        bind.samplers[wboit_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
        bind.samplers[wboit_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;
        bind.views[wboit_shd.VIEW_morph_tex] = item.morph_view;
        bind.samplers[wboit_shd.SMP_morph_smp] = env.pipelines.morph_sampler;

        const prb = scene_draw.probeForDraw(env, item.model);
        bind.views[wboit_shd.VIEW_probe_tex] = prb.view;
        bind.views[wboit_shd.VIEW_probe2_tex] = prb.view2;
        bind.samplers[wboit_shd.SMP_probe_smp] = prb.sampler;

        scene_draw.bindClusteredViews(&bind, wboit_shd, env);
        sg.applyBindings(bind);

        const vs_params = wboit_shd.VsParams{
            .mvp = mvp,
            .model = model,
        };
        sg.applyUniforms(wboit_shd.UB_vs_params, sg.asRange(&vs_params));

        const skel_bones: ?*const [MAX_BONES]Mat4 = if (item.is_skinned)
            render_queue.skinAt(skins, item.skin_index) orelse return
        else
            null;

        if (skel_bones) |bones| {
            sg.applyUniforms(wboit_shd.UB_vs_skin, sg.asRange(bones));
            sg.applyUniforms(wboit_shd.UB_vs_morph, sg.asRange(&scene_draw.vsMorphUniform(wboit_shd, item.morph_uniforms)));
        } else {
            sg.applyUniforms(wboit_shd.UB_vs_morph, sg.asRange(&scene_draw.vsMorphUniform(wboit_shd, item.morph_uniforms)));
        }

        const f = scene_draw.frameUniformsForState(env.shadow_uniforms, item.receive_shadows, ctx, env.refraction_view_proj, env.refraction_capture);
        sg.applyUniforms(wboit_shd.UB_fs_view, sg.asRange(f));

        const coat = scene_draw.resolveCoat(coats, item.coat_index);
        const mat = uniforms.buildFrameMaterial(.{
            .base_color = rec.base_color,
            .pbr_factors = rec.pbr_factors,
            .emissive_color = rec.emissive_color,
            .alpha_cutoff = rec.alpha_cutoff,
            .normal_scale = rec.normal_scale,
            .uv_matrices = rec.uv_matrices,
            .uv_offsets = rec.uv_offsets,
            .channel_selectors = rec.channel_selectors,
            .clearcoat_factors = coat.clearcoat_factors,
            .clearcoat_color = coat.clearcoat_color,
            .sheen_factors = coat.sheen_factors,
            .sheen_color = coat.sheen_color,
            .anisotropy_factors = coat.anisotropy_factors,
            .transmission_factors = coat.transmission_factors,
            .transmission_color = coat.transmission_color,
            .sss_factors = coat.sss_factors,
            .sss_color = coat.sss_color,
            .clearcoat_uv_matrix = coat.clearcoat_uv_matrix,
            .clearcoat_uv_offset = coat.clearcoat_uv_offset,
            .sheen_uv_matrix = coat.sheen_uv_matrix,
            .sheen_uv_offset = coat.sheen_uv_offset,
            .refraction_factors = coat.refraction_factors,
            .anisotropy_uv_matrix = coat.anisotropy_uv_matrix,
            .anisotropy_uv_offset = coat.anisotropy_uv_offset,
            .probe_params = prb.params,
            .probe2_params = prb.params2,
            .probe_box = prb.box,
        });
        sg.applyUniforms(wboit_shd.UB_fs_mat, sg.asRange(&mat));

        sg.draw(item.base_vertex, item.index_count, 1);
    }

    pub fn drawInstancedBatch(
        self: *const WboitPass,
        env: *const Environment,
        batch: RenderInstancedBatch,
        ctx: *const FrameContext,
        current_pipeline_id: *u32,
        coats: []const material_mod.CoatParams,
    ) void {
        if (batch.index_count == 0 or batch.visible_instance_count == 0) return;
        if (sg.isvalid() and !gpu_thread.geometryValid(batch.vertex_buffer, batch.index_buffer)) return;

        const is_u32 = batch.index_type == .UINT32;
        const pip_id = self.pipelineForInstancedMesh(is_u32, batch.double_sided);
        if (pip_id == 0) return;

        if (pip_id != current_pipeline_id.*) {
            sg.applyPipeline(.{ .id = pip_id });
            current_pipeline_id.* = pip_id;
            env.stats.pipeline_switches += 1;
        }

        const rec = batch.draw_record;
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = batch.vertex_buffer;
        bind.vertex_buffers[1] = batch.instance_buffer;
        bind.index_buffer = batch.index_buffer;

        bind.views[wboit_shd.VIEW_albedo_tex] = rec.albedo_view;
        bind.views[wboit_shd.VIEW_normal_tex] = rec.normal_view;
        bind.views[wboit_shd.VIEW_metallic_roughness_tex] = rec.mr_view;
        bind.views[wboit_shd.VIEW_emissive_tex] = rec.emissive_view;
        bind.views[wboit_shd.VIEW_occlusion_tex] = rec.occlusion_view;
        bind.views[wboit_shd.VIEW_clearcoat_tex] = rec.clearcoat_view;
        bind.views[wboit_shd.VIEW_sheen_tex] = rec.sheen_view;
        bind.views[wboit_shd.VIEW_anisotropy_tex] = rec.anisotropy_view;
        bind.views[wboit_shd.VIEW_refraction_tex] = if (env.refraction_view.id != 0) env.refraction_view else env.default_white.view;
        bind.samplers[wboit_shd.SMP_refraction_smp] = if (env.refraction_sampler.id != 0) env.refraction_sampler else env.default_white.sampler;
        bind.samplers[wboit_shd.SMP_smp] = rec.albedo_sampler;
        bind.samplers[wboit_shd.SMP_data_smp] = rec.data_sampler;
        bind.views[wboit_shd.VIEW_brdf_lut_tex] = rec.brdf_lut_view;
        bind.samplers[wboit_shd.SMP_brdf_lut_smp] = rec.brdf_lut_sampler;

        const cube_view = rec.env_view orelse (env.sky_texture orelse env.default_cube).view;
        const cube_sampler = rec.env_sampler orelse (env.sky_texture orelse env.default_cube).sampler;
        bind.views[wboit_shd.VIEW_env_tex] = cube_view;
        bind.samplers[wboit_shd.SMP_env_smp] = cube_sampler;

        bind.views[wboit_shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
        bind.views[wboit_shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
        bind.views[wboit_shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
        bind.views[wboit_shd.VIEW_spot_shadow_depth_tex] = env.shadow_pass.spot_texture_view;
        bind.views[wboit_shd.VIEW_point_shadow_tex] = env.shadow_pass.point_texture_view;
        bind.views[wboit_shd.VIEW_point_shadow_depth_tex] = env.shadow_pass.point_texture_view;
        bind.samplers[wboit_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
        bind.samplers[wboit_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;

        const prb = scene_draw.probeForDrawPos(env, batch.world_center);
        bind.views[wboit_shd.VIEW_probe_tex] = prb.view;
        bind.views[wboit_shd.VIEW_probe2_tex] = prb.view2;
        bind.samplers[wboit_shd.SMP_probe_smp] = prb.sampler;

        scene_draw.bindClusteredViews(&bind, wboit_shd, env);
        sg.applyBindings(bind);

        const vs_params = wboit_shd.VsInstParams{
            .view_proj = ctx.view_proj,
        };
        sg.applyUniforms(wboit_shd.UB_vs_inst_params, sg.asRange(&vs_params));

        const f = scene_draw.frameUniformsForState(env.shadow_uniforms, batch.receive_shadows, ctx, env.refraction_view_proj, env.refraction_capture);
        sg.applyUniforms(wboit_shd.UB_fs_view, sg.asRange(f));

        const coat = scene_draw.resolveCoat(coats, batch.coat_index);
        const mat = uniforms.buildFrameMaterial(.{
            .base_color = rec.base_color,
            .pbr_factors = rec.pbr_factors,
            .emissive_color = rec.emissive_color,
            .alpha_cutoff = rec.alpha_cutoff,
            .normal_scale = rec.normal_scale,
            .uv_matrices = rec.uv_matrices,
            .uv_offsets = rec.uv_offsets,
            .channel_selectors = rec.channel_selectors,
            .clearcoat_factors = coat.clearcoat_factors,
            .clearcoat_color = coat.clearcoat_color,
            .sheen_factors = coat.sheen_factors,
            .sheen_color = coat.sheen_color,
            .anisotropy_factors = coat.anisotropy_factors,
            .transmission_factors = coat.transmission_factors,
            .transmission_color = coat.transmission_color,
            .sss_factors = coat.sss_factors,
            .sss_color = coat.sss_color,
            .clearcoat_uv_matrix = coat.clearcoat_uv_matrix,
            .clearcoat_uv_offset = coat.clearcoat_uv_offset,
            .sheen_uv_matrix = coat.sheen_uv_matrix,
            .sheen_uv_offset = coat.sheen_uv_offset,
            .refraction_factors = coat.refraction_factors,
            .anisotropy_uv_matrix = coat.anisotropy_uv_matrix,
            .anisotropy_uv_offset = coat.anisotropy_uv_offset,
            .probe_params = prb.params,
            .probe2_params = prb.params2,
            .probe_box = prb.box,
        });
        sg.applyUniforms(wboit_shd.UB_fs_mat, sg.asRange(&mat));

        sg.draw(0, batch.index_count, batch.visible_instance_count);
    }

    pub fn render(
        self: *WboitPass,
        queues: *const RenderQueues,
        view_env: *const Environment,
        frame_ctx: *const FrameContext,
        main_color: sg.View,
        main_depth: sg.View,
        depth_format: sg.PixelFormat,
        width: i32,
        height: i32,
        viewport_rect: ?PixelRect,
    ) void {
        const total_trans = queues.transparent.items.len + queues.transparent_instanced.items.len;
        if (total_trans == 0) return;
        if (!self.ensure(width, height)) return;
        self.ensurePipelines(depth_format);

        // 1. Accumulation Pass (MRT into accum + reveal)
        var pass = sg.Pass{};
        pass.action.colors[0] = .{ .load_action = .CLEAR, .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 }, .store_action = .STORE };
        pass.action.colors[1] = .{ .load_action = .CLEAR, .clear_value = .{ .r = 1, .g = 1, .b = 1, .a = 1 }, .store_action = .STORE };
        pass.action.depth = .{ .load_action = .LOAD, .store_action = .STORE };
        pass.attachments.colors[0] = self.accum_att_view;
        pass.attachments.colors[1] = self.reveal_att_view;
        pass.attachments.depth_stencil = main_depth;
        sg.beginPass(pass);

        if (viewport_rect) |rect| {
            sg.applyViewport(rect.x, rect.y, rect.width, rect.height, true);
            sg.applyScissorRect(rect.x, rect.y, rect.width, rect.height, true);
        }

        var current_pipeline_id: u32 = 0;
        for (queues.transparent.items) |*item| {
            self.drawRegularItem(view_env, item, frame_ctx, &current_pipeline_id, queues.skin_storage.items, queues.coat_storage.items);
        }
        for (queues.transparent_instanced.items) |batch| {
            self.drawInstancedBatch(view_env, batch, frame_ctx, &current_pipeline_id, queues.coat_storage.items);
        }
        sg.endPass();

        // 2. Resolve Pass (fullscreen quad composited over main HDR target)
        var resolve_pass = sg.Pass{};
        resolve_pass.action.colors[0] = .{ .load_action = .LOAD, .store_action = .STORE };
        resolve_pass.action.depth = .{ .load_action = .LOAD, .store_action = .STORE };
        resolve_pass.attachments.colors[0] = main_color;
        resolve_pass.attachments.depth_stencil = main_depth;
        sg.beginPass(resolve_pass);

        if (viewport_rect) |rect| {
            sg.applyViewport(rect.x, rect.y, rect.width, rect.height, true);
            sg.applyScissorRect(rect.x, rect.y, rect.width, rect.height, true);
        }

        sg.applyPipeline(self.pip_resolve);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.quad_vb;
        bind.index_buffer = self.quad_ib;
        bind.views[wboit_resolve_shd.VIEW_accum_tex] = self.accum_tex_view;
        bind.views[wboit_resolve_shd.VIEW_reveal_tex] = self.reveal_tex_view;
        bind.samplers[wboit_resolve_shd.SMP_smp] = self.sampler;
        sg.applyBindings(bind);

        sg.draw(0, 6, 1);
        sg.endPass();

        // Update stats
        const draw_count: u32 = @intCast(total_trans + 1);
        view_env.stats.draw_calls += draw_count;
        view_env.stats.main_draw_calls += draw_count;
    }
};
