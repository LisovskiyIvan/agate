const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const pyr_shd = @import("depth_pyramid_shader");
const pp = @import("../postprocess.zig");

// GPU Hierarchical Depth Pyramid (Hi-Z) Pass.
// Constructs conservative depth downsample mips for hierarchical raymarching (SSR,
// contact shadows) and fast occlusion culling.
// Owns its offscreen targets and fullscreen downsampling pipeline.
pub const DepthPyramidPass = struct {
    mip_images: [pp.DEPTH_PYRAMID_MAX_MIPS]sg.Image = [_]sg.Image{.{}} ** pp.DEPTH_PYRAMID_MAX_MIPS,
    mip_att_views: [pp.DEPTH_PYRAMID_MAX_MIPS]sg.View = [_]sg.View{.{}} ** pp.DEPTH_PYRAMID_MAX_MIPS,
    mip_tex_views: [pp.DEPTH_PYRAMID_MAX_MIPS]sg.View = [_]sg.View{.{}} ** pp.DEPTH_PYRAMID_MAX_MIPS,

    sampler: sg.Sampler = .{},
    pipeline: sg.Pipeline = .{},
    shader: sg.Shader = .{},
    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    base_width: i32 = 0,
    base_height: i32 = 0,
    mip_count: u32 = 0,

    /// Sole-contract render-target pixel format for depth pyramid targets (RGBA16F).
    pub fn pyramidPixelFormat() sg.PixelFormat {
        return .RGBA16F;
    }

    pub fn init() DepthPyramidPass {
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
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const shd_handle = sg.makeShader(pyr_shd.depthPyramidShaderDesc(sg.queryBackend()));
        var pip_desc = sg.PipelineDesc{
            .shader = shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        pip_desc.colors[0].pixel_format = pyramidPixelFormat();
        pip_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        pip_desc.layout.attrs[pyr_shd.ATTR_depth_pyramid_position] = .{ .format = .FLOAT2, .offset = 0 };
        pip_desc.layout.attrs[pyr_shd.ATTR_depth_pyramid_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const pip = sg.makePipeline(pip_desc);

        return .{
            .sampler = smp,
            .pipeline = pip,
            .shader = shd_handle,
            .quad_vb = vb,
            .quad_ib = ib,
        };
    }

    pub fn resize(self: *DepthPyramidPass, width: i32, height: i32) void {
        if (width <= 0 or height <= 0) return;
        if (!sg.isvalid()) return;
        if (self.base_width == width and self.base_height == height) return;

        self.destroyTargets();

        const n = pp.computeMipCount(width, height);
        if (n == 0) return;

        const fmt = pyramidPixelFormat();

        for (0..n) |i| {
            const size = pp.depthPyramidMipSize(width, height, @intCast(i));
            self.mip_images[i] = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = fmt,
                .sample_count = 1,
            });
            if (sg.queryImageState(self.mip_images[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
            self.mip_att_views[i] = sg.makeView(.{
                .color_attachment = .{ .image = self.mip_images[i] },
            });
            if (sg.queryViewState(self.mip_att_views[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
            self.mip_tex_views[i] = sg.makeView(.{
                .texture = .{ .image = self.mip_images[i] },
            });
            if (sg.queryViewState(self.mip_tex_views[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
        }

        self.base_width = width;
        self.base_height = height;
        self.mip_count = n;
    }

    /// Renders the depth pyramid mips from `base_depth_view`.
    /// Level 0 downsamples `base_depth_view` into half-resolution.
    /// Subsequent levels downsample from the previous level.
    /// Returns the Level 0 texture view, or .{} if inactive or unallocated.
    pub fn render(self: *DepthPyramidPass, base_depth_view: sg.View, cur_w: i32, cur_h: i32) sg.View {
        if (base_depth_view.id == 0) return .{};
        if (cur_w <= 0 or cur_h <= 0) return .{};
        if (!sg.isvalid()) return .{};

        if (self.base_width != cur_w or self.base_height != cur_h) {
            self.resize(cur_w, cur_h);
        }

        const n = self.mip_count;
        if (n == 0 or self.mip_tex_views[0].id == 0) return .{};

        var src_view = base_depth_view;
        var src_w: f32 = @floatFromInt(cur_w);
        var src_h: f32 = @floatFromInt(cur_h);

        for (0..n) |i| {
            var pass = sg.Pass{
                .action = .{
                    .colors = [_]sg.ColorAttachmentAction{
                        .{ .load_action = .DONTCARE },
                    } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
                },
            };
            pass.attachments.colors[0] = self.mip_att_views[i];
            sg.beginPass(pass);

            sg.applyPipeline(self.pipeline);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = self.quad_vb;
            bind.index_buffer = self.quad_ib;
            bind.views[pyr_shd.VIEW_depth_tex] = src_view;
            bind.samplers[pyr_shd.SMP_smp] = self.sampler;
            sg.applyBindings(bind);

            const fs_params = pyr_shd.FsParams{
                .src_resolution = .{ src_w, src_h, 1.0 / src_w, 1.0 / src_h },
            };
            sg.applyUniforms(pyr_shd.UB_fs_params, sg.asRange(&fs_params));
            sg.draw(0, 6, 1);
            sg.endPass();

            src_view = self.mip_tex_views[i];
            const size = pp.depthPyramidMipSize(cur_w, cur_h, @intCast(i));
            src_w = @floatFromInt(size.w);
            src_h = @floatFromInt(size.h);
        }

        return self.mip_tex_views[0];
    }

    /// Returns the texture view for a specific mip level in the pyramid, or .{} if unavailable.
    pub fn mipView(self: *const DepthPyramidPass, level: usize) sg.View {
        if (level >= self.mip_count) return .{};
        return self.mip_tex_views[level];
    }

    pub fn deinit(self: *DepthPyramidPass) void {
        self.destroyTargets();
        if (self.pipeline.id != 0) sg.destroyPipeline(self.pipeline);
        if (self.shader.id != 0) sg.destroyShader(self.shader);
        if (self.sampler.id != 0) sg.destroySampler(self.sampler);
        if (self.quad_vb.id != 0) sg.destroyBuffer(self.quad_vb);
        if (self.quad_ib.id != 0) sg.destroyBuffer(self.quad_ib);
        self.pipeline = .{};
        self.shader = .{};
        self.sampler = .{};
        self.quad_vb = .{};
        self.quad_ib = .{};
    }

    fn destroyTargets(self: *DepthPyramidPass) void {
        for (0..pp.DEPTH_PYRAMID_MAX_MIPS) |i| {
            if (self.mip_att_views[i].id != 0) sg.destroyView(self.mip_att_views[i]);
            if (self.mip_tex_views[i].id != 0) sg.destroyView(self.mip_tex_views[i]);
            if (self.mip_images[i].id != 0) sg.destroyImage(self.mip_images[i]);
            self.mip_images[i] = .{};
            self.mip_att_views[i] = .{};
            self.mip_tex_views[i] = .{};
        }
        self.base_width = 0;
        self.base_height = 0;
        self.mip_count = 0;
    }
};

test "depth pyramid pass fail-closes headless with no state touched" {
    var pass = DepthPyramidPass{};
    // Render with empty depth view returns empty view without panic or sg calls
    const res = pass.render(.{}, 1920, 1080);
    try std.testing.expectEqual(@as(u32, 0), res.id);

    // Render with 0 dimensions returns empty view
    const res2 = pass.render(.{ .id = 42 }, 0, 0);
    try std.testing.expectEqual(@as(u32, 0), res2.id);

    // Out of range mip query returns empty view
    const m = pass.mipView(10);
    try std.testing.expectEqual(@as(u32, 0), m.id);

    // deinit on unallocated pass is completely safe
    pass.deinit();
}

test "depth pyramid mip count and halving bounds" {
    try std.testing.expectEqual(@as(u32, 0), pp.computeMipCount(0, 0));
    try std.testing.expectEqual(@as(u32, 8), pp.computeMipCount(1920, 1080));
    try std.testing.expectEqual(@as(u32, 7), pp.computeMipCount(128, 64));

    const s0 = pp.depthPyramidMipSize(800, 600, 0);
    try std.testing.expectEqual(@as(i32, 400), s0.w);
    try std.testing.expectEqual(@as(i32, 300), s0.h);
}
