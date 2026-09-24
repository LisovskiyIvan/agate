const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const raymarch_shd = @import("volumetric_raymarch_shader");
const blur_shd = @import("volumetric_blur_shader");
const pp = @import("../postprocess.zig");
const glow_mod = @import("glow_pass.zig");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color3 = math.Color3;

// Volumetric light-shaft layer v1: low-resolution sun-CSM raymarch +
// bilateral (depth-aware) separable blur.
//
// Per active frame (parent gates on postprocess.shaftActive, which
// additionally requires shadows_enabled — without a rendered CSM atlas
// there is no real data to march against):
//  1. Raymarch: fullscreen march at half/quarter resolution over the
//     scene depth (reconstructed to world space), one raw CSM tap per
//     step, HG phase + Beer-Lambert accumulation. Outputs unscaled
//     shaft radiance; intensity scaling stays in the postprocess
//     composite (after the highlight block).
//  2. Blur: bilateral H then V over the shaft radiance with the scene
//     depth as the edge gate, so shafts never bleed across silhouettes.
//  3. Composite: the parent feeds the blurred view into the fullscreen
//     postprocess shader, which adds the radiance to linear scene color
//     before exposure/tone mapping (placeholder + zeroed uniforms when
//     inactive, so the off path is bit-identical).
//
// Typical parent (scene/postfx_stack.zig) usage, after the highlight
// stage and before the fullscreen postprocess pass:
//
//   var shaft_view = empty_view; // e.g. the resolved scene view placeholder
//   if (postprocess.shaftActive(post.enabled, post, shadows_enabled)) {
//       shaft_view = self.volumetric_pass.render(.{
//           .depth_view = ...,
//           .shadow_view = ...,
//           .inv_view_proj = ...,
//           .camera_pos = ...,
//           .sun_dir = ...,
//           .sun_color = ...,
//           .splits = ...,
//           .cascades = ...,
//           .shadow_bias = ...,
//           .config = post,
//           .base_w = cur_w,
//           .base_h = cur_h,
//       });
//       self.stats.draw_calls += postprocess.SHAFT_PASS_DRAWS;
//   }
//   self.postprocess_pass.setShaftTexture(shaft_view);
//
// Headless behavior mirrors GlowPass exactly: zero handles fail closed
// (render returns an empty view before any sg.* call), no sg.isvalid
// gates on the fail-closed path. The pass is uniform-only past its input
// binds (applyUniforms, never sg.updateBuffer), so it records nothing
// into gpu_upload_meter and renderReuse replays stay upload-free by
// construction. Target destruction is immediate (same as
// GlowPass.destroyTargets, context thread only): the retire queue stays
// for game-side removals that can happen off-context (probes, ui3d
// panels, meshes), not for pass-owned low-res RTs recreated on resize.
// The targets allocate lazily on the first active render (highlight
// precedent), so post-on with the shaft off holds no shaft VRAM.
pub const VolumetricPass = struct {
    raymarch_image: sg.Image = .{},
    raymarch_att_view: sg.View = .{},
    raymarch_tex_view: sg.View = .{},

    blur_images: [2]sg.Image = [_]sg.Image{.{}} ** 2,
    blur_att_views: [2]sg.View = [_]sg.View{.{}} ** 2,
    blur_tex_views: [2]sg.View = [_]sg.View{.{}} ** 2,

    sampler: sg.Sampler = .{},
    depth_sampler: sg.Sampler = .{},
    raymarch_pipeline: sg.Pipeline = .{},
    blur_pipeline: sg.Pipeline = .{},
    raymarch_shader: sg.Shader = .{},
    blur_shader: sg.Shader = .{},
    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    base_width: i32 = 0,
    base_height: i32 = 0,
    resolution: pp.ShaftResolution = .quarter,

    /// Render-target pixel format, shared convention with
    /// BloomPass/GlowPass (HDR shaft radiance where the backend can
    /// render to it, LDR fallback otherwise). Context thread only
    /// (queries live sokol caps; headless sg aborts on pixelformat
    /// queries).
    pub fn shaftPixelFormat() sg.PixelFormat {
        return glow_mod.GlowPass.glowPixelFormat();
    }

    pub fn shaftBytesPerPixel() usize {
        return glow_mod.GlowPass.glowBytesPerPixel();
    }

    /// VRAM census helper (pure byte math, no GPU calls, headless-safe):
    /// three shaft-resolution targets (raymarch + H/V ping-pong) at
    /// `bytes_per_pixel`. Single source of truth for
    /// profiler/snapshot.zig.
    pub fn targetBytes(base_w: i32, base_h: i32, bytes_per_pixel: usize, res: pp.ShaftResolution) usize {
        const size = pp.shaftTargetSize(base_w, base_h, res);
        const w: usize = @intCast(size.w);
        const h: usize = @intCast(size.h);
        return 3 * w * h * bytes_per_pixel;
    }

    pub fn init() VolumetricPass {
        // Fullscreen Quad (XY, UV)
        const quad_vertices = [_]f32{
            // x,     y,    u,   v
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
        const depth_smp = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const shaft_fmt = shaftPixelFormat();

        const raymarch_shd_handle = sg.makeShader(raymarch_shd.volumetricRaymarchShaderDesc(sg.queryBackend()));
        var raymarch_desc = sg.PipelineDesc{
            .shader = raymarch_shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        raymarch_desc.colors[0].pixel_format = shaft_fmt;
        raymarch_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        raymarch_desc.layout.attrs[raymarch_shd.ATTR_volumetric_raymarch_position] = .{ .format = .FLOAT2, .offset = 0 };
        raymarch_desc.layout.attrs[raymarch_shd.ATTR_volumetric_raymarch_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const raymarch_pip = sg.makePipeline(raymarch_desc);

        const blur_shd_handle = sg.makeShader(blur_shd.volumetricBlurShaderDesc(sg.queryBackend()));
        var blur_desc = sg.PipelineDesc{
            .shader = blur_shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        blur_desc.colors[0].pixel_format = shaft_fmt;
        blur_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        blur_desc.layout.attrs[blur_shd.ATTR_volumetric_blur_position] = .{ .format = .FLOAT2, .offset = 0 };
        blur_desc.layout.attrs[blur_shd.ATTR_volumetric_blur_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const blur_pip = sg.makePipeline(blur_desc);

        return .{
            .sampler = smp,
            .depth_sampler = depth_smp,
            .raymarch_pipeline = raymarch_pip,
            .blur_pipeline = blur_pip,
            .raymarch_shader = raymarch_shd_handle,
            .blur_shader = blur_shd_handle,
            .quad_vb = vb,
            .quad_ib = ib,
        };
    }

    pub fn resize(self: *VolumetricPass, base_w: i32, base_h: i32, res: pp.ShaftResolution) void {
        if (base_w <= 0 or base_h <= 0) return;
        if (self.base_width == base_w and self.base_height == base_h and self.resolution == res) return;

        self.destroyTargets();

        const shaft_fmt = shaftPixelFormat();
        const size = pp.shaftTargetSize(base_w, base_h, res);

        const ray_img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = size.w,
            .height = size.h,
            .pixel_format = shaft_fmt,
            .sample_count = 1,
        });
        self.raymarch_image = ray_img;
        self.raymarch_att_view = sg.makeView(.{
            .color_attachment = .{ .image = ray_img },
        });
        self.raymarch_tex_view = sg.makeView(.{
            .texture = .{ .image = ray_img },
        });

        for (0..2) |i| {
            const blur_img = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = shaft_fmt,
                .sample_count = 1,
            });
            self.blur_images[i] = blur_img;
            self.blur_att_views[i] = sg.makeView(.{
                .color_attachment = .{ .image = blur_img },
            });
            self.blur_tex_views[i] = sg.makeView(.{
                .texture = .{ .image = blur_img },
            });
        }

        self.base_width = base_w;
        self.base_height = base_h;
        self.resolution = res;
    }

    /// Per-frame raymarch inputs. All views are render-owned snapshots
    /// (depth + CSM atlas texture views); matrices and sun state come
    /// from the frame snapshot; config carries the shaft knobs (clamped
    /// inside render, so callers pass the raw frame config).
    pub const RenderArgs = struct {
        depth_view: sg.View = .{},
        shadow_view: sg.View = .{},
        inv_view_proj: Mat4 = Mat4.identity,
        camera_pos: Vec3 = Vec3.zero,
        sun_dir: Vec3 = Vec3.new(0, 1, 0),
        sun_color: Color3 = Color3.white,
        splits: [4]f32 = .{ 0, 0, 0, 0 },
        cascades: [4]Mat4 = [_]Mat4{Mat4.identity} ** 4,
        shadow_bias: f32 = 0.0,
        config: pp.PostProcessOptions = .{},
        base_w: i32 = 0,
        base_h: i32 = 0,
    };

    fn blurStage(
        self: *VolumetricPass,
        src_view: sg.View,
        depth_view: sg.View,
        direction: f32,
        sigma_spatial: f32,
        sigma_edge: f32,
        texel_w: f32,
        texel_h: f32,
        slot: usize,
    ) void {
        var pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        pass.attachments.colors[0] = self.blur_att_views[slot];
        sg.beginPass(pass);

        sg.applyPipeline(self.blur_pipeline);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.quad_vb;
        bind.index_buffer = self.quad_ib;
        bind.views[blur_shd.VIEW_src_tex] = src_view;
        bind.views[blur_shd.VIEW_depth_tex] = depth_view;
        bind.samplers[blur_shd.SMP_smp] = self.sampler;
        bind.samplers[blur_shd.SMP_depth_smp] = self.depth_sampler;
        sg.applyBindings(bind);

        const fs_params = blur_shd.FsParams{
            .texel = .{ texel_w, texel_h, 0.0, 0.0 },
            .params = .{ direction, sigma_spatial, sigma_edge, 0.0 },
        };
        sg.applyUniforms(blur_shd.UB_fs_params, sg.asRange(&fs_params));
        sg.draw(0, 6, 1);
        sg.endPass();
    }

    // March the shaft radiance from `args` and return the bilaterally
    // blurred shaft view (shaft-resolution blur slot 1). Returns an empty
    // view when inactive so the caller can fall back to the placeholder
    // (bit-identical composite).
    pub fn render(self: *VolumetricPass, args: RenderArgs) sg.View {
        if (self.raymarch_pipeline.id == 0 or self.blur_pipeline.id == 0) return .{};
        if (args.depth_view.id == 0) return .{};
        if (args.shadow_view.id == 0) return .{};
        if (args.base_w <= 0 or args.base_h <= 0) return .{};

        const cfg = args.config.clamped();
        // A zeroed bias would speckle grazing march samples against
        // their own surface; floor at a sub-texel epsilon (the forward
        // default is 0.0012, so this only bites degenerate configs).
        const bias = @max(args.shadow_bias, 0.0002);

        self.resize(args.base_w, args.base_h, cfg.shaft_resolution);
        if (self.raymarch_image.id == 0) return .{};

        // Stage 1: raymarch into the raymarch target.
        var pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        pass.attachments.colors[0] = self.raymarch_att_view;
        sg.beginPass(pass);

        sg.applyPipeline(self.raymarch_pipeline);
        var march_bind = sg.Bindings{};
        march_bind.vertex_buffers[0] = self.quad_vb;
        march_bind.index_buffer = self.quad_ib;
        march_bind.views[raymarch_shd.VIEW_depth_tex] = args.depth_view;
        march_bind.views[raymarch_shd.VIEW_shadow_tex] = args.shadow_view;
        march_bind.samplers[raymarch_shd.SMP_depth_smp] = self.depth_sampler;
        sg.applyBindings(march_bind);

        const march_params = raymarch_shd.FsParams{
            .inv_view_proj = args.inv_view_proj,
            .cascade_vp = args.cascades,
            .camera_pos = .{ args.camera_pos.x, args.camera_pos.y, args.camera_pos.z, 0.0 },
            .sun_dir = .{ args.sun_dir.x, args.sun_dir.y, args.sun_dir.z, 0.0 },
            .sun_color = .{ args.sun_color.r, args.sun_color.g, args.sun_color.b, 0.0 },
            .splits = .{ args.splits[0], args.splits[1], args.splits[2], args.splits[3] },
            .march_a = .{
                @floatFromInt(cfg.shaft_steps),
                cfg.shaft_density,
                cfg.shaft_max_distance,
                cfg.shaft_anisotropy,
            },
            .march_b = .{ bias, 0.0, 0.0, 0.0 },
        };
        sg.applyUniforms(raymarch_shd.UB_fs_params, sg.asRange(&march_params));
        sg.draw(0, 6, 1);
        sg.endPass();

        // Stages 2-3: bilateral blur, H into slot 0 then V into slot 1.
        // Intensity scaling stays in the postprocess composite.
        const size = pp.shaftTargetSize(args.base_w, args.base_h, cfg.shaft_resolution);
        const texel_w: f32 = 1.0 / @as(f32, @floatFromInt(size.w));
        const texel_h: f32 = 1.0 / @as(f32, @floatFromInt(size.h));
        self.blurStage(self.raymarch_tex_view, args.depth_view, 0.0, cfg.shaft_blur_sigma, cfg.shaft_edge_sigma, texel_w, texel_h, 0);
        self.blurStage(self.blur_tex_views[0], args.depth_view, 1.0, cfg.shaft_blur_sigma, cfg.shaft_edge_sigma, texel_w, texel_h, 1);

        return self.blur_tex_views[1];
    }

    fn destroyTargets(self: *VolumetricPass) void {
        if (self.raymarch_image.id == 0) return;
        sg.destroyImage(self.raymarch_image);
        sg.destroyView(self.raymarch_att_view);
        sg.destroyView(self.raymarch_tex_view);
        self.raymarch_image = .{};
        self.raymarch_att_view = .{};
        self.raymarch_tex_view = .{};
        for (0..2) |i| {
            sg.destroyImage(self.blur_images[i]);
            sg.destroyView(self.blur_att_views[i]);
            sg.destroyView(self.blur_tex_views[i]);
            self.blur_images[i] = .{};
            self.blur_att_views[i] = .{};
            self.blur_tex_views[i] = .{};
        }
        self.base_width = 0;
        self.base_height = 0;
    }

    pub fn deinit(self: *VolumetricPass) void {
        self.destroyTargets();
        sg.destroySampler(self.sampler);
        sg.destroySampler(self.depth_sampler);
        sg.destroyPipeline(self.raymarch_pipeline);
        sg.destroyPipeline(self.blur_pipeline);
        if (self.raymarch_shader.id != 0) sg.destroyShader(self.raymarch_shader);
        if (self.blur_shader.id != 0) sg.destroyShader(self.blur_shader);
        self.raymarch_shader = .{};
        self.blur_shader = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};

test "volumetric pass fail-closes headless with no state touched" {
    const upload_meter = @import("../gpu_upload_meter.zig");
    // Zero-initialized pass (never init'ed: no sg context headless, same
    // as the GlowPass fail-closed shape) must return an empty view before
    // any sg.* call.
    var pass: VolumetricPass = .{};
    _ = upload_meter.takeAndReset();
    const empty = pass.render(.{});
    try std.testing.expectEqual(@as(u32, 0), empty.id);
    // Missing pipelines, missing depth/shadow views, and degenerate sizes
    // all fail closed the same way (guard order: pipelines, views, size).
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .depth_view = .{ .id = 9 }, .shadow_view = .{ .id = 10 }, .base_w = 1280, .base_h = 720 }).id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .depth_view = .{ .id = 9 }, .base_w = 1280, .base_h = 720 }).id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .depth_view = .{ .id = 9 }, .shadow_view = .{ .id = 10 }, .base_w = 0, .base_h = 720 }).id);
    // Fail-closed render records no GPU uploads and sizes nothing.
    try std.testing.expectEqual(@as(u64, 0), upload_meter.takeAndReset());
    try std.testing.expectEqual(@as(i32, 0), pass.base_width);
    try std.testing.expectEqual(@as(i32, 0), pass.base_height);
    try std.testing.expectEqual(@as(u32, 0), pass.raymarch_image.id);
}

test "volumetric target bytes account three shaft-res targets" {
    // Pure byte math (no sg calls): exact and deterministic headless. The
    // RGBA8 leg (4 Bpp) is pinned here; the RGBA16F leg (8 Bpp) shares the
    // same formula with shaftBytesPerPixel (context-only, like glow).
    try std.testing.expectEqual(@as(usize, 3 * 320 * 180 * 4), VolumetricPass.targetBytes(1280, 720, 4, .quarter));
    try std.testing.expectEqual(@as(usize, 3 * 640 * 360 * 4), VolumetricPass.targetBytes(1280, 720, 4, .half));
    try std.testing.expectEqual(@as(usize, 3 * 320 * 180 * 8), VolumetricPass.targetBytes(1280, 720, 8, .quarter));
    // Quarter is exactly a fourth of half (area scaling).
    try std.testing.expectEqual(VolumetricPass.targetBytes(1280, 720, 4, .half) / 4, VolumetricPass.targetBytes(1280, 720, 4, .quarter));
    // Degenerate sizes clamp to 1x1 targets, never zero.
    try std.testing.expectEqual(@as(usize, 3 * 1 * 1 * 4), VolumetricPass.targetBytes(1, 1, 4, .quarter));
    try std.testing.expectEqual(@as(usize, 3 * 1 * 1 * 4), VolumetricPass.targetBytes(0, 0, 4, .half));
    // Matches the shaft sizing the pass allocates.
    const size = pp.shaftTargetSize(1280, 720, .quarter);
    try std.testing.expectEqual(@as(usize, 3 * @as(usize, @intCast(size.w)) * @as(usize, @intCast(size.h)) * 4), VolumetricPass.targetBytes(1280, 720, 4, .quarter));
}
