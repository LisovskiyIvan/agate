const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const update_shd = @import("ddgi_update_shader");
const gi = @import("../gi/ddgi.zig");

// DDGI v1 spike: probe-atlas update pass (fragment-based ray march, no
// compute — D3D11 floor).
//
// Per active frame (parent gates on scene.ddgi.enabled; default off holds
// no atlas VRAM and issues zero draws):
//  1. One fullscreen draw into the RGBA16F atlas with LOAD (content
//     persists across frames for the round-robin slices to converge).
//     First allocation after (re)size uses CLEAR (garbage texels would
//     otherwise seed the sample side with NaNs).
//  2. Each texel of this frame's probe window marches the proxy uniform
//     arrays (miss -> hemi sky, hit -> albedo x sun + ambient); texels
//     outside the window DISCARD so LOAD preserves their values.
//  3. The parent serves atlasTexView() to the forward draws (Environment
//     ddgi_view, sampled through the PBR ddgi chunk) and counts 1 draw.
//
// Typical parent (scene/frame_render.zig) usage, before the main pass:
//
//   var ddgi_view: sg.View = .{};
//   if (scene.ddgi.enabled) {
//       const n = gi.collectProxies(scene.meshes.items, &proxy_buf);
//       const win = gi.budgetWindow(scene.frame_id, probes, scene.ddgi.probes_per_frame);
//       ddgi_view = self.postfx.ddgi_pass.render(.{ ... });
//       self.stats.draw_calls += 1;
//   }
//
// Headless behavior mirrors SsgiPass: zero handles fail closed (render
// returns an empty view before any sg.* call), no sg.isvalid gates on the
// fail-closed path. Uniform-only past the binds (applyUniforms, never
// sg.updateBuffer), so it records nothing into gpu_upload_meter. Target
// destruction is immediate (context thread only). The atlas allocates
// lazily on the first active render, so enabled-off holds no DDGI VRAM.
pub const DdgiPass = struct {
    atlas_image: sg.Image = .{},
    atlas_att_view: sg.View = .{},
    atlas_tex_view: sg.View = .{},

    sampler: sg.Sampler = .{},

    update_pipeline: sg.Pipeline = .{},
    update_shader: sg.Shader = .{},

    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    /// Probe count the atlas was sized for. 0 until the first active
    /// render (lazy allocation, so the off shape holds no DDGI VRAM).
    probes: usize = 0,
    /// First render after (re)size clears; later renders LOAD.
    atlas_cleared: bool = false,

    /// Sole-contract atlas pixel format (always RGBA16F: HDR probe
    /// radiance; same contract as the SSGI pass targets). Pure.
    pub fn ddgiPixelFormat() sg.PixelFormat {
        return .RGBA16F;
    }

    /// VRAM census helper (pure byte math, no GPU calls, headless-safe).
    pub fn targetBytes(probe_count: usize) usize {
        const size = gi.atlasSize(probe_count);
        return @as(usize, size[0]) * size[1] * 8;
    }

    pub fn init() DdgiPass {
        const quad_vertices = [_]f32{
            -1.0, -1.0, 0.0, 0.0,
            1.0,  -1.0, 1.0, 0.0,
            1.0,  1.0,  1.0, 1.0,
            -1.0, 1.0,  0.0, 1.0,
        };
        const quad_indices = [_]u16{ 0, 1, 2, 0, 2, 3 };

        const vb = sg.makeBuffer(.{ .data = sg.asRange(&quad_vertices) });
        const ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&quad_indices),
        });

        // NEAREST + clamp: probe tiles have no guard bands, so bilinear
        // would bleed across tile borders (documented spike limit).
        const smp = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const shd = sg.makeShader(update_shd.ddgiUpdateShaderDesc(sg.queryBackend()));
        var desc = sg.PipelineDesc{
            .shader = shd,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        desc.colors[0].pixel_format = ddgiPixelFormat();
        desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        desc.layout.attrs[update_shd.ATTR_ddgi_update_position] = .{ .format = .FLOAT2, .offset = 0 };
        desc.layout.attrs[update_shd.ATTR_ddgi_update_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const pip = sg.makePipeline(desc);

        return .{
            .sampler = smp,
            .update_pipeline = pip,
            .update_shader = shd,
            .quad_vb = vb,
            .quad_ib = ib,
        };
    }

    /// (Re)allocates the atlas for `probe_count` probes (fixed 64px width,
    /// rows derive from the count). No-op on degenerate or unchanged
    /// counts. Context thread only.
    pub fn resize(self: *DdgiPass, probe_count: usize) void {
        if (probe_count == 0 or probe_count > gi.MAX_PROBES) return;
        if (!sg.isvalid()) return;
        if (self.probes == probe_count and self.atlas_image.id != 0) return;

        self.destroyTargets();

        const size = gi.atlasSize(probe_count);
        self.atlas_image = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = @intCast(size[0]),
            .height = @intCast(size[1]),
            .pixel_format = ddgiPixelFormat(),
            .sample_count = 1,
        });
        if (sg.queryImageState(self.atlas_image) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.atlas_att_view = sg.makeView(.{
            .color_attachment = .{ .image = self.atlas_image },
        });
        if (sg.queryViewState(self.atlas_att_view) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.atlas_tex_view = sg.makeView(.{
            .texture = .{ .image = self.atlas_image },
        });
        if (sg.queryViewState(self.atlas_tex_view) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.probes = probe_count;
        self.atlas_cleared = false;
    }

    pub const RenderArgs = struct {
        opts: gi.DdgiOptions = .{},
        /// Owned probe window for this frame (CPU mirror:
        /// gi.budgetWindow over the clamped probe count).
        window_start: usize = 0,
        window_count: usize = 0,
        /// Direction TO the sun (world, need not be normalized).
        sun_dir: Vec3 = Vec3.new(0, 1, 0),
        sun_color: Color3 = Color3.white,
        sun_intensity: f32 = 1.0,
        /// Hemispheric sky model (scene hemi lanes, same convention as
        /// common/hemi.glsl: groundColor raw, diffuse x intensity ends).
        hemi_dir: Vec3 = Vec3.up,
        hemi_diffuse: Color3 = Color3.white,
        hemi_ground: Color3 = Color3.white,
        hemi_intensity: f32 = 1.0,
        proxies: []const gi.ProxyBox = &.{},
    };

    /// Refreshes this frame's probe window and returns the atlas texture
    /// view (empty when inactive so the caller falls back to the
    /// placeholder — bit-identical composite of the off path). Costs
    /// exactly 1 draw when active.
    pub fn render(self: *DdgiPass, args: RenderArgs) sg.View {
        if (self.update_pipeline.id == 0) return .{};
        if (!args.opts.enabled) return .{};
        const counts = gi.clampedCounts(args.opts.counts);
        const probe_count: usize = @as(usize, counts[0]) * counts[1] * counts[2];
        if (probe_count == 0) return .{};
        if (args.window_count == 0) return .{};

        self.resize(probe_count);
        if (self.probes != probe_count) return .{};
        if (self.atlas_image.id == 0) return .{};

        const cell = gi.cellSize(args.opts.extent, args.opts.counts);
        const lanes = gi.gridLanes(args.opts);

        var proxy_min: [gi.MAX_PROXIES][4]f32 = [_][4]f32{.{ 0, 0, 0, 0 }} ** gi.MAX_PROXIES;
        var proxy_max: [gi.MAX_PROXIES][4]f32 = [_][4]f32{.{ 0, 0, 0, 0 }} ** gi.MAX_PROXIES;
        var proxy_albedo: [gi.MAX_PROXIES][4]f32 = [_][4]f32{.{ 0, 0, 0, 0 }} ** gi.MAX_PROXIES;
        const n: usize = @min(args.proxies.len, gi.MAX_PROXIES);
        for (args.proxies[0..n], 0..) |p, i| {
            proxy_min[i] = .{ p.min[0], p.min[1], p.min[2], 0.0 };
            proxy_max[i] = .{ p.max[0], p.max[1], p.max[2], 0.0 };
            proxy_albedo[i] = .{ p.albedo[0], p.albedo[1], p.albedo[2], 0.0 };
        }

        const sun = Vec3.normalize(args.sun_dir);
        const params = update_shd.FsParams{
            .ddgi_grid_origin = lanes[0],
            .ddgi_grid_counts = lanes[1],
            .ddgi_grid_step = .{ cell[0], cell[1], cell[2], lanes[2][3] },
            .ddgi_atlas = .{ lanes[3][0], lanes[3][1], lanes[3][2], @floatFromInt(n) },
            .ddgi_window = .{
                @floatFromInt(args.window_start),
                @floatFromInt(args.window_count),
                @floatFromInt(probe_count),
                0.0,
            },
            .ddgi_sun_dir = .{ sun.x, sun.y, sun.z, @max(args.sun_intensity, 0.0) },
            .ddgi_sun_color = .{ args.sun_color.r, args.sun_color.g, args.sun_color.b, 0.0 },
            .ddgi_sky = .{ args.hemi_diffuse.r, args.hemi_diffuse.g, args.hemi_diffuse.b, @max(args.hemi_intensity, 0.0) },
            .ddgi_sky_ground = .{ args.hemi_ground.r, args.hemi_ground.g, args.hemi_ground.b, 0.0 },
            .ddgi_sky_dir = .{ args.hemi_dir.x, args.hemi_dir.y, args.hemi_dir.z, 0.0 },
            .ddgi_proxy_min = proxy_min,
            .ddgi_proxy_max = proxy_max,
            .ddgi_proxy_albedo = proxy_albedo,
        };

        const first = !self.atlas_cleared;
        var pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    if (first) .{ .load_action = .CLEAR, .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 1 } } else .{ .load_action = .LOAD },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        pass.attachments.colors[0] = self.atlas_att_view;
        sg.beginPass(pass);

        sg.applyPipeline(self.update_pipeline);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.quad_vb;
        bind.index_buffer = self.quad_ib;
        sg.applyBindings(bind);

        sg.applyUniforms(update_shd.UB_fs_params, sg.asRange(&params));
        sg.draw(0, 6, 1);
        sg.endPass();

        self.atlas_cleared = true;
        return self.atlas_tex_view;
    }

    /// Atlas texture view for the forward draws (.{} when unallocated —
    /// the caller binds the placeholder instead). Pure view read.
    pub fn atlasTexView(self: *const DdgiPass) sg.View {
        if (self.atlas_image.id == 0) return .{};
        return self.atlas_tex_view;
    }

    fn destroyTargets(self: *DdgiPass) void {
        if (self.atlas_att_view.id != 0) sg.destroyView(self.atlas_att_view);
        if (self.atlas_tex_view.id != 0) sg.destroyView(self.atlas_tex_view);
        if (self.atlas_image.id != 0) sg.destroyImage(self.atlas_image);
        self.atlas_image = .{};
        self.atlas_att_view = .{};
        self.atlas_tex_view = .{};
        self.probes = 0;
        self.atlas_cleared = false;
    }

    pub fn deinit(self: *DdgiPass) void {
        self.destroyTargets();
        sg.destroySampler(self.sampler);
        sg.destroyPipeline(self.update_pipeline);
        if (self.update_shader.id != 0) sg.destroyShader(self.update_shader);
        self.update_shader = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};
