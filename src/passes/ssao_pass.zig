const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const ssao_shd = @import("ssao_shader");
const blur_shd = @import("ssao_blur_shader");
const temporal_shd = @import("ssao_temporal_shader");
const ssao_mod = @import("../ssao.zig");
const SSAOOptions = ssao_mod.SSAOOptions;
const Camera = @import("../camera.zig").Camera;

// Screen-space AO: half-res gather (hemisphere kernel or opt-in horizon
// slices) + optional temporal accumulation (opt-in) + full-res bilateral
// upsample.
//
// Per active frame (parent gates on post.enabled + ssao.enabled; MSAA
// suppression clears ssao.enabled so both modes join it for free):
//  1. Gather: fullscreen AO at half of the EFFECTIVE (render-scale) size
//     over the full-res depth. Outputs the AO scalar (1 = unoccluded).
//     `extra_params.y` selects hemisphere (0, default bit-identical) vs
//     horizon slices (1, same tap budget via steps-per-side).
//     `extra_params.z` carries the temporal gather rotation (radians,
//     exactly 0.0 when temporal is off or fail-closed, so the default
//     gather is bit-identical); active non-reset temporal frames add the
//     bounded phase offset so history integrates varying directions.
//     Horizon slices cap at the configured steps per side BEFORE any
//     depth fetch (no wasted taps when steps is 2 or 3).
//  2. Temporal (opt-in via ssao.temporal.enabled, default OFF):
//     reproject the gather output against the velocity buffer
//     (camera-depth fallback without velocity) into the history
//     ping-pong pair (RGBA16F: R = AO, A = linear depth), with variance
//     clipping + depth/normal/reactive-marker rejection. History holds
//     the RAW accumulated signal; OFF skips this stage and the upsample
//     reads the gather directly (bit-identical to pre-temporal).
//     Multi-camera and MSAA fail closed to the gather-direct path with
//     history reset (never blended across views).
//  3. Upsample: bilateral (depth-aware) blur into the full-res target.
//
// Typical parent (scene/postfx_stack.zig) usage, before the fullscreen
// postprocess pass:
//
//   const res = self.ssao_pass.renderWithContext(.{
//       .camera = ...,
//       .aspect = ...,
//       .depth_tex_view = ...,
//       .config = ssao,
//       .cur_w = fx_w,
//       .cur_h = fx_h,
//       .view_proj = ...,
//       .inv_view_proj = ...,
//       .prev_view_proj = ...,
//       .velocity_view = ...,
//       .velocity_valid = ...,
//       .camera_cut = post.taa.camera_cut,
//       .multi_camera = ...,
//       .msaa_active = ...,
//   });
//   draws += 2 + (res.temporal_executed ? 1 : 0);
//
// Concrete public API clients keep the historical render() shape (gather
// + upsample, context-free); the scene orchestrator uses
// renderWithContext for the temporal lanes.
//
// Headless behavior mirrors SsgiPass: zero handles fail closed (render
// returns before any sg.* call). The pass is uniform-only past its input
// binds (applyUniforms, never sg.updateBuffer). Targets allocate lazily
// on the first active render, so post-on with SSAO never enabled holds
// no SSAO VRAM. The temporal pair additionally frees itself on the first
// pass-active render with the flag off, so disabling temporal releases
// its VRAM without waiting for a resize.
pub const SSAOPass = struct {
    ssao_raw_image: sg.Image = .{},
    ssao_raw_att_view: sg.View = .{},
    ssao_raw_tex_view: sg.View = .{},

    ssao_blur_image: sg.Image = .{},
    ssao_blur_att_view: sg.View = .{},
    ssao_blur_tex_view: sg.View = .{},

    temp_a_image: sg.Image = .{},
    temp_a_att_view: sg.View = .{},
    temp_a_tex_view: sg.View = .{},
    temp_b_image: sg.Image = .{},
    temp_b_att_view: sg.View = .{},
    temp_b_tex_view: sg.View = .{},

    /// Temporal history cursor: read/write slots alternate by parity (CPU
    /// mirror: aoTemporalReadIndex/WriteIndex). Incremented per
    /// temporal-active render, including reset frames (the reset output
    /// primes the just-written slot for the next frame).
    temp_frame: u64 = 0,
    /// False until the first temporal-active render completes; the parent
    /// inactivity path clears it, so a mode toggle or disable/enable
    /// cycle always replays the reset frame.
    temp_has_history: bool = false,
    /// Last frame's temporal enable state; a false->true edge is the
    /// toggle reset (mirrors the TAA toggled_on condition).
    temp_enabled_prev: bool = false,
    /// AO mode the history was accumulated under; a hemisphere/horizon
    /// switch with live history replays the reset frame (the two modes
    /// share the history lane format, but their signals must not blend).
    temp_mode: ssao_mod.AoMode = .hemisphere,
    /// Bounded temporal gather-rotation phase (0..PERIOD-1, CPU mirror:
    /// aoTemporalRotation). Advances only after an actual temporal draw
    /// and resets to 0 with history (cut/toggle/resize/mode/projection/
    /// explicit reset, plus the off and fail-closed paths via
    /// noteTemporalInactive), so reset/prime gathers stay unjittered.
    /// Separate from temp_frame: that cursor must keep counting draws
    /// (parent stats + ping-pong parity) across history resets.
    temp_phase: u32 = 0,

    noise_image: sg.Image = .{},
    noise_tex_view: sg.View = .{},
    noise_sampler: sg.Sampler = .{},

    depth_sampler: sg.Sampler = .{},
    blur_sampler: sg.Sampler = .{},
    linear_sampler: sg.Sampler = .{},

    ssao_pipeline: sg.Pipeline = .{},
    ssao_blur_pipeline: sg.Pipeline = .{},
    temporal_pipeline: sg.Pipeline = .{},
    ssao_shader: sg.Shader = .{},
    blur_shader: sg.Shader = .{},
    temporal_shader: sg.Shader = .{},

    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    kernel_samples: [32][4]f32 = undefined,

    width: i32 = 0,
    height: i32 = 0,

    cached_proj: Mat4 = Mat4.identity,
    cached_inv_proj: Mat4 = Mat4.identity,
    last_aspect: f32 = 0.0,
    last_fov: f32 = 0.0,
    last_near: f32 = 0.0,
    last_far: f32 = 0.0,

    /// History pixel format (always RGBA16F: R = AO scalar, A = linear
    /// depth in meters; GB = oct-encoded world normal). The gather/blur targets stay RGBA8
    /// (the historical format); only the opt-in history pair pays the
    /// wider format for depth precision. Pure (no sg calls,
    /// headless-safe).
    pub fn temporalPixelFormat() sg.PixelFormat {
        return .RGBA16F;
    }

    /// True while the history pair is allocated. Pure handle check (no sg
    /// calls, headless-safe); the census gate.
    pub fn temporalSized(self: *const SSAOPass) bool {
        return self.temp_a_image.id != 0;
    }

    pub fn init() SSAOPass {
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

        // 1. Generate 32 hemisphere samples using Fermat spiral (uniform solid angle)
        var kernel: [32][4]f32 = undefined;
        const golden_angle: f32 = 2.0 * std.math.pi * (1.0 - 0.618033988749895);

        for (0..32) |i| {
            const fi = @as(f32, @floatFromInt(i));
            const theta = fi * golden_angle;
            // Distribute z evenly in hemisphere [0.12, 1.0]
            const z = (fi + 0.5) / 32.0 * 0.88 + 0.12;
            const r = @sqrt(@max(0.0, 1.0 - z * z));
            const x = r * @cos(theta);
            const y = r * @sin(theta);

            // Accelerating scale factor towards origin
            const norm_i = (fi + 1.0) / 32.0;
            const scale = std.math.lerp(0.12, 1.0, norm_i * norm_i);

            kernel[i] = .{ x * scale, y * scale, z * scale, 0.0 };
        }

        // 2. Generate 8x8 random rotation noise texture (64 pixels)
        var noise_pixels: [64 * 4]u8 = undefined;
        var prng = std.Random.DefaultPrng.init(1337);
        const rand = prng.random();
        for (0..64) |i| {
            const angle = rand.float(f32) * 2.0 * std.math.pi;
            const nx = @cos(angle);
            const ny = @sin(angle);

            noise_pixels[i * 4 + 0] = @intFromFloat((nx * 0.5 + 0.5) * 255.0);
            noise_pixels[i * 4 + 1] = @intFromFloat((ny * 0.5 + 0.5) * 255.0);
            noise_pixels[i * 4 + 2] = 128;
            noise_pixels[i * 4 + 3] = 255;
        }

        var noise_img_desc = sg.ImageDesc{
            .width = 8,
            .height = 8,
            .pixel_format = .RGBA8,
        };
        noise_img_desc.data.mip_levels[0] = sg.asRange(&noise_pixels);
        const noise_img = sg.makeImage(noise_img_desc);
        const noise_view = sg.makeView(.{
            .texture = .{ .image = noise_img },
        });

        const noise_smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .REPEAT,
            .wrap_v = .REPEAT,
        });

        const depth_smp = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const blur_smp = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const linear_smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        // 3. SSAO pipeline
        const ssao_shd_handle = sg.makeShader(ssao_shd.ssaoShaderDesc(sg.queryBackend()));
        var ssao_pip_desc = sg.PipelineDesc{
            .shader = ssao_shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        ssao_pip_desc.colors[0].pixel_format = .RGBA8;
        ssao_pip_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        ssao_pip_desc.layout.attrs[ssao_shd.ATTR_ssao_position] = .{ .format = .FLOAT2, .offset = 0 };
        ssao_pip_desc.layout.attrs[ssao_shd.ATTR_ssao_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const ssao_pip = sg.makePipeline(ssao_pip_desc);

        // 4. SSAO Blur pipeline
        const blur_shd_handle = sg.makeShader(blur_shd.ssaoBlurShaderDesc(sg.queryBackend()));
        var blur_pip_desc = sg.PipelineDesc{
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
        blur_pip_desc.colors[0].pixel_format = .RGBA8;
        blur_pip_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        blur_pip_desc.layout.attrs[blur_shd.ATTR_ssao_blur_position] = .{ .format = .FLOAT2, .offset = 0 };
        blur_pip_desc.layout.attrs[blur_shd.ATTR_ssao_blur_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const blur_pip = sg.makePipeline(blur_pip_desc);

        // 5. SSAO temporal pipeline (opt-in history; RGBA16F pair)
        const temporal_shd_handle = sg.makeShader(temporal_shd.ssaoTemporalShaderDesc(sg.queryBackend()));
        var temporal_desc = sg.PipelineDesc{
            .shader = temporal_shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        temporal_desc.colors[0].pixel_format = .RGBA16F;
        temporal_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        temporal_desc.layout.attrs[temporal_shd.ATTR_ssao_temporal_position] = .{ .format = .FLOAT2, .offset = 0 };
        temporal_desc.layout.attrs[temporal_shd.ATTR_ssao_temporal_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const temporal_pip = sg.makePipeline(temporal_desc);

        return .{
            .noise_image = noise_img,
            .noise_tex_view = noise_view,
            .noise_sampler = noise_smp,
            .depth_sampler = depth_smp,
            .blur_sampler = blur_smp,
            .linear_sampler = linear_smp,
            .ssao_pipeline = ssao_pip,
            .ssao_blur_pipeline = blur_pip,
            .temporal_pipeline = temporal_pip,
            .ssao_shader = ssao_shd_handle,
            .blur_shader = blur_shd_handle,
            .temporal_shader = temporal_shd_handle,
            .quad_vb = vb,
            .quad_ib = ib,
            .kernel_samples = kernel,
        };
    }

    pub fn resize(self: *SSAOPass, width: i32, height: i32) void {
        if (width <= 0 or height <= 0) return;
        if (self.width == width and self.height == height) return;

        if (self.ssao_raw_image.id != 0) {
            sg.destroyImage(self.ssao_raw_image);
            sg.destroyView(self.ssao_raw_att_view);
            sg.destroyView(self.ssao_raw_tex_view);

            sg.destroyImage(self.ssao_blur_image);
            sg.destroyView(self.ssao_blur_att_view);
            sg.destroyView(self.ssao_blur_tex_view);
        }
        // A resize is a history reset (fresh targets carry no valid
        // history); free the pair so the next temporal-active render
        // replays the reset frame on freshly allocated targets.
        self.destroyTemporal();
        self.temp_frame = 0;
        self.temp_has_history = false;
        self.temp_enabled_prev = false;
        self.temp_phase = 0;

        // Trace at half resolution; the bilateral pass upsamples to full size.
        const half = ssao_mod.aoHalfSize(width, height);
        const raw_img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = half.w,
            .height = half.h,
            .pixel_format = .RGBA8,
            .sample_count = 1,
        });
        const raw_att = sg.makeView(.{
            .color_attachment = .{ .image = raw_img },
        });
        const raw_tex = sg.makeView(.{
            .texture = .{ .image = raw_img },
        });

        // SSAO Blur Target (RGBA8)
        const blur_img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = .RGBA8,
            .sample_count = 1,
        });
        const blur_att = sg.makeView(.{
            .color_attachment = .{ .image = blur_img },
        });
        const blur_tex = sg.makeView(.{
            .texture = .{ .image = blur_img },
        });

        self.width = width;
        self.height = height;
        self.ssao_raw_image = raw_img;
        self.ssao_raw_att_view = raw_att;
        self.ssao_raw_tex_view = raw_tex;
        self.ssao_blur_image = blur_img;
        self.ssao_blur_att_view = blur_att;
        self.ssao_blur_tex_view = blur_tex;
    }

    /// Per-frame AO inputs. Camera/depth/config/cur size are the gather
    /// lanes (historical render() shape); the temporal stage additionally
    /// consumes the frame view-projection set (jittered matrices are
    /// fine — the same ones SSGI reprojects with), the frame velocity
    /// view (render-scale, may be empty: the shader falls back to
    /// camera-depth reprojection), the shared one-frame camera-cut flag
    /// (taa.camera_cut doubles as the AO history reset), and the
    /// fail-closed gates (multi_camera: history must never blend across
    /// views; msaa_active: belt-and-braces behind the
    /// upstream SSAO suppression).
    pub const RenderArgs = struct {
        camera: Camera,
        aspect: f32,
        depth_tex_view: sg.View,
        config: SSAOOptions,
        cur_w: i32,
        cur_h: i32,
        view_proj: Mat4 = Mat4.identity,
        inv_view_proj: Mat4 = Mat4.identity,
        prev_view_proj: Mat4 = Mat4.identity,
        velocity_view: sg.View = .{},
        velocity_valid: bool = false,
        camera_cut: bool = false,
        multi_camera: bool = false,
        msaa_active: bool = false,
    };

    /// Marks the temporal history invalid without touching GPU handles
    /// (pure bool writes, headless-safe). Missing-handle/size early
    /// returns leave even this untouched (fail-closed shape, mirroring
    /// SsgiPass).
    fn noteTemporalInactive(self: *SSAOPass) void {
        self.temp_has_history = false;
        self.temp_enabled_prev = false;
        self.temp_phase = 0;
    }

    /// Public history reset for skipped frames: the orchestrator calls
    /// this when the pass does not run at all (master switch off, SSAO
    /// off, MSAA-suppressed), so a re-enable always replays the reset
    /// frame instead of blending stale history. Frees the pair (one-time
    /// effective) and clears the cursor/mode alongside the flags.
    /// Context thread only; headless-safe (zero handles are never
    /// passed to sg.*).
    pub fn resetTemporalHistory(self: *SSAOPass) void {
        self.destroyTemporal();
        self.temp_frame = 0;
        self.temp_has_history = false;
        self.temp_enabled_prev = false;
        self.temp_mode = .hemisphere;
        self.temp_phase = 0;
    }

    /// Total temporal draws executed since init / the last reset (the
    /// parity cursor; the q0 harness reads it to prove the stage ran).
    /// Pure (no sg calls, headless-safe).
    pub fn temporalDrawCount(self: *const SSAOPass) u64 {
        return self.temp_frame;
    }

    /// What renderWithContext actually executed. The parent counts draws
    /// from this, never from the request alone: an allocation failure or
    /// fail-closed gate must not count a draw that never ran.
    pub const RenderResult = struct {
        temporal_executed: bool = false,
    };

    fn tempReadTexView(self: *const SSAOPass) sg.View {
        return if (ssao_mod.aoTemporalReadIndex(self.temp_frame) == 0) self.temp_a_tex_view else self.temp_b_tex_view;
    }

    fn tempWriteAttView(self: *const SSAOPass) sg.View {
        return if (ssao_mod.aoTemporalWriteIndex(self.temp_frame) == 0) self.temp_a_att_view else self.temp_b_att_view;
    }

    fn tempWriteTexView(self: *const SSAOPass) sg.View {
        return if (ssao_mod.aoTemporalWriteIndex(self.temp_frame) == 0) self.temp_a_tex_view else self.temp_b_tex_view;
    }

    /// Projection changes invalidate depth stored under the old near/far.
    pub fn aoProjectionChanged(
        last_aspect: f32,
        last_fov: f32,
        last_near: f32,
        last_far: f32,
        aspect: f32,
        fov: f32,
        near_z: f32,
        far_z: f32,
    ) bool {
        return aspect != last_aspect or fov != last_fov or near_z != last_near or far_z != last_far;
    }

    /// Velocity-texture bind for the temporal stage (pure handle routing,
    /// headless-safe): the write slot must NEVER be sampled in the same
    /// pass (read/write alias is forbidden even when the shader gates
    /// velocity on temporal_params.y). Prefers the live velocity view;
    /// otherwise the history READ view (a distinct image from the write
    /// slot) when valid; otherwise the raw gather (read-read sharing with
    /// the history slot is legal). The caller passes the same
    /// velocity_live/history_valid bits it uploads, so the bind and the
    /// shader gate cannot disagree.
    pub fn temporalVelocityBind(
        self: *const SSAOPass,
        velocity_view: sg.View,
        velocity_live: bool,
        history_valid: bool,
    ) sg.View {
        if (velocity_live) return velocity_view;
        if (history_valid) return self.tempReadTexView();
        return self.ssao_raw_tex_view;
    }

    /// Allocates the history ping-pong pair at the half-res size.
    /// Returns true when (re)created — the caller's reset trigger for the
    /// frame. No-op (false) when already sized. Context thread only; fails
    /// closed (false) without publishing partial state.
    fn ensureTemporal(self: *SSAOPass, base_w: i32, base_h: i32) bool {
        if (base_w <= 0 or base_h <= 0) return false;
        if (!sg.isvalid()) return false;
        if (self.temporalSized()) return false;
        const size = ssao_mod.aoHalfSize(base_w, base_h);
        if (size.w <= 0 or size.h <= 0) return false;
        const fmt = temporalPixelFormat();
        var imgs: [2]sg.Image = .{ .{}, .{} };
        var atts: [2]sg.View = .{ .{}, .{} };
        var texs: [2]sg.View = .{ .{}, .{} };
        for (0..2) |i| {
            imgs[i] = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = fmt,
                .sample_count = 1,
            });
            if (sg.queryImageState(imgs[i]) != .VALID) {
                if (imgs[i].id != 0) sg.destroyImage(imgs[i]);
                if (i == 1) {
                    sg.destroyView(atts[0]);
                    sg.destroyView(texs[0]);
                    sg.destroyImage(imgs[0]);
                }
                return false;
            }
            atts[i] = sg.makeView(.{ .color_attachment = .{ .image = imgs[i] } });
            texs[i] = sg.makeView(.{ .texture = .{ .image = imgs[i] } });
            if (sg.queryViewState(atts[i]) != .VALID or sg.queryViewState(texs[i]) != .VALID) {
                if (texs[i].id != 0) sg.destroyView(texs[i]);
                if (atts[i].id != 0) sg.destroyView(atts[i]);
                sg.destroyImage(imgs[i]);
                if (i == 1) {
                    sg.destroyView(atts[0]);
                    sg.destroyView(texs[0]);
                    sg.destroyImage(imgs[0]);
                }
                return false;
            }
        }
        self.temp_a_image = imgs[0];
        self.temp_a_att_view = atts[0];
        self.temp_a_tex_view = texs[0];
        self.temp_b_image = imgs[1];
        self.temp_b_att_view = atts[1];
        self.temp_b_tex_view = texs[1];
        return true;
    }

    /// Historical public entry point (gather + bilateral upsample,
    /// context-free): concrete API clients keep this exact signature.
    /// Temporal filtering requires the actual frame matrices supplied to
    /// renderWithContext; this entry point runs spatial AO only.
    pub fn render(
        self: *SSAOPass,
        camera: Camera,
        aspect: f32,
        depth_tex_view: sg.View,
        config: SSAOOptions,
        cur_w: i32,
        cur_h: i32,
    ) void {
        var spatial = config;
        spatial.temporal.enabled = false;
        _ = self.renderWithContext(.{
            .camera = camera,
            .aspect = aspect,
            .depth_tex_view = depth_tex_view,
            .config = spatial,
            .cur_w = cur_w,
            .cur_h = cur_h,
        });
    }

    /// Scene-orchestrated entry point: RenderArgs plus an execution
    /// report (see RenderResult) so the parent counts only draws that
    /// ran.
    pub fn renderWithContext(self: *SSAOPass, args: RenderArgs) RenderResult {
        if (!args.config.enabled) {
            self.noteTemporalInactive();
            return .{};
        }
        const config = args.config.clamped();
        if (!config.enabled) {
            self.noteTemporalInactive();
            return .{};
        }
        const cur_w = args.cur_w;
        const cur_h = args.cur_h;
        if (cur_w <= 0 or cur_h <= 0) return .{};
        if (self.ssao_pipeline.id == 0 or self.ssao_blur_pipeline.id == 0) return .{};
        if (args.depth_tex_view.id == 0) return .{};

        self.resize(cur_w, cur_h);

        const camera = args.camera;
        const aspect = args.aspect;
        const fov = camera.getFovDeg();
        const near_z = camera.getNear();
        const far_z = camera.getFar();
        const proj_changed = SSAOPass.aoProjectionChanged(
            self.last_aspect,
            self.last_fov,
            self.last_near,
            self.last_far,
            aspect,
            fov,
            near_z,
            far_z,
        );
        if (proj_changed) {
            const proj = camera.getProjectionMatrix(aspect);
            const inv = proj.invert() orelse return .{};
            self.cached_proj = proj;
            self.cached_inv_proj = inv;
            self.last_aspect = aspect;
            self.last_fov = fov;
            self.last_near = near_z;
            self.last_far = far_z;
        }

        const proj = self.cached_proj;
        const inv_proj = self.cached_inv_proj;

        // Temporal gating BEFORE the gather: the gather rotation (jitter)
        // must stay 0 unless the temporal draw below will actually
        // execute. ensureTemporal runs here (allocation only, no draws)
        // so a failure fail-closes to the unjittered spatial gather with
        // a truthful result (no history prime, no phase advance).
        var temporal_execute = false;
        var temporal_reset = false;
        var temporal_history_valid = false;
        var temporal_velocity_live = false;
        if (ssao_mod.aoTemporalActive(config)) {
            if (args.multi_camera or args.msaa_active) {
                self.destroyTemporal();
                self.noteTemporalInactive();
            } else if (self.temporal_pipeline.id != 0) {
                const recreated = self.ensureTemporal(cur_w, cur_h);
                if (self.temporalSized()) {
                    temporal_reset = ssao_mod.aoTemporalShouldReset(.{
                        .first_frame = !self.temp_has_history,
                        .toggled_on = !self.temp_enabled_prev,
                        .resized = recreated,
                        .camera_cut = args.camera_cut,
                        .explicit_reset = false,
                    }) or (self.temp_has_history and (self.temp_mode != config.mode or proj_changed));
                    const read_view = self.tempReadTexView();
                    temporal_history_valid = !temporal_reset and read_view.id != 0;
                    temporal_velocity_live = args.velocity_valid and args.velocity_view.id != 0;
                    temporal_execute = true;
                }
            }
        } else {
            // Temporal off: release the pair so disabled temporal holds no
            // VRAM (one-time destroy on the toggle-off frame; the census
            // gate temporalSized() goes quiet with it).
            self.destroyTemporal();
            self.noteTemporalInactive();
        }
        // Reset/prime frames and every fail-closed path gather unjittered;
        // only active non-reset frames rotate (CPU mirror:
        // aoTemporalRotation, mode-aware so the horizon micro-rotates
        // within one slice spacing instead of permuting its star).
        const gather_rotation: f32 = if (temporal_execute and !temporal_reset)
            ssao_mod.aoTemporalRotation(self.temp_phase, config.mode)
        else
            0.0;

        // ---------------------------------------------
        // PASS 1: Generate SSAO into ssao_raw_image
        // ---------------------------------------------
        var ssao_pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        ssao_pass.attachments.colors[0] = self.ssao_raw_att_view;
        sg.beginPass(ssao_pass);

        sg.applyPipeline(self.ssao_pipeline);
        var bind1 = sg.Bindings{};
        bind1.vertex_buffers[0] = self.quad_vb;
        bind1.index_buffer = self.quad_ib;
        bind1.views[ssao_shd.VIEW_depth_tex] = args.depth_tex_view;
        bind1.views[ssao_shd.VIEW_noise_tex] = self.noise_tex_view;
        bind1.samplers[ssao_shd.SMP_smp_depth] = self.depth_sampler;
        bind1.samplers[ssao_shd.SMP_smp_noise] = self.noise_sampler;
        sg.applyBindings(bind1);

        const fs_params = ssao_shd.FsParams{
            .projection = proj,
            .inv_projection = inv_proj,
            .kernel_samples = self.kernel_samples,
            .params = .{ config.radius, config.bias, config.intensity, config.power },
            .resolution = .{
                @floatFromInt(cur_w),
                @floatFromInt(cur_h),
                1.0 / @as(f32, @floatFromInt(cur_w)),
                1.0 / @as(f32, @floatFromInt(cur_h)),
            },
            .extra_params = .{
                @floatFromInt(config.sample_count),
                @floatFromInt(@intFromEnum(config.mode)),
                gather_rotation,
                0.0,
            },
        };
        sg.applyUniforms(ssao_shd.UB_fs_params, sg.asRange(&fs_params));
        sg.draw(0, 6, 1);
        sg.endPass();

        // ---------------------------------------------
        // PASS 1.5 (opt-in): temporal accumulation of the raw AO.
        // OFF (default): the upsample reads the gather directly — the
        // exact pre-temporal frame. ON: reproject + variance-clip +
        // reject into the history write slot; the upsample reads that.
        // Multi-camera/MSAA fail closed to the gather-direct path with
        // history reset (never blended across views).
        // ---------------------------------------------
        var denoise_src = self.ssao_raw_tex_view;
        var result: RenderResult = .{};
        if (temporal_execute) {
            const read_view = self.tempReadTexView();
            const history_valid = temporal_history_valid and read_view.id != 0;
            const velocity_live = temporal_velocity_live;

            var temporal_pass = sg.Pass{
                .action = .{
                    .colors = [_]sg.ColorAttachmentAction{
                        .{ .load_action = .DONTCARE },
                    } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
                },
            };
            temporal_pass.attachments.colors[0] = self.tempWriteAttView();
            sg.beginPass(temporal_pass);

            sg.applyPipeline(self.temporal_pipeline);
            var temporal_bind = sg.Bindings{};
            temporal_bind.vertex_buffers[0] = self.quad_vb;
            temporal_bind.index_buffer = self.quad_ib;
            temporal_bind.views[temporal_shd.VIEW_gather_tex] = self.ssao_raw_tex_view;
            temporal_bind.views[temporal_shd.VIEW_history_tex] = if (history_valid) read_view else self.ssao_raw_tex_view;
            temporal_bind.views[temporal_shd.VIEW_depth_tex] = args.depth_tex_view;
            // Never the write slot (see temporalVelocityBind):
            // read-read sharing is legal, read-write is not.
            temporal_bind.views[temporal_shd.VIEW_velocity_tex] = self.temporalVelocityBind(
                args.velocity_view,
                velocity_live,
                history_valid,
            );
            temporal_bind.samplers[temporal_shd.SMP_smp] = self.linear_sampler;
            temporal_bind.samplers[temporal_shd.SMP_depth_smp] = self.depth_sampler;
            temporal_bind.samplers[temporal_shd.SMP_velocity_smp] = self.depth_sampler;
            sg.applyBindings(temporal_bind);

            const half = ssao_mod.aoHalfSize(cur_w, cur_h);
            const hw: f32 = @floatFromInt(half.w);
            const hh: f32 = @floatFromInt(half.h);
            const fw: f32 = @floatFromInt(cur_w);
            const fh: f32 = @floatFromInt(cur_h);
            const temporal_params = temporal_shd.FsParams{
                .inv_view_proj = args.inv_view_proj,
                .view_proj = args.view_proj,
                .prev_view_proj = args.prev_view_proj,
                .resolution = .{ hw, hh, 1.0 / hw, 1.0 / hh },
                .full_resolution = .{ fw, fh, 1.0 / fw, 1.0 / fh },
                .camera_params = .{ camera.getNear(), camera.getFar(), config.radius, 0.0 },
                .temporal_params = .{
                    if (history_valid) 1.0 else 0.0,
                    if (velocity_live) 1.0 else 0.0,
                    ssao_mod.aoTemporalBlend(config),
                    ssao_mod.AO_TEMPORAL_GAMMA,
                },
            };
            sg.applyUniforms(temporal_shd.UB_fs_params, sg.asRange(&temporal_params));
            sg.draw(0, 6, 1);
            sg.endPass();

            // Capture the just-written slot BEFORE advancing the
            // cursor: the parity helpers flip with temp_frame.
            // Any temporal draw (including a reset prime) counts:
            // the parent stats off temporal_executed, never the
            // request alone. The rotation phase advances the same way
            // (reset primes frame 0, so the next active frame is phase
            // 1); allocation failures and fail-closed gates never reach
            // here, so they neither prime history nor advance the phase.
            denoise_src = self.tempWriteTexView();
            self.temp_frame += 1;
            self.temp_phase = (if (temporal_reset) @as(u32, 1) else self.temp_phase + 1) %
                ssao_mod.AO_TEMPORAL_ROTATION_PERIOD;
            self.temp_has_history = true;
            self.temp_enabled_prev = true;
            self.temp_mode = config.mode;
            result.temporal_executed = true;
        }

        // ---------------------------------------------
        // PASS 2: Bilateral Blur into ssao_blur_image
        // ---------------------------------------------
        var blur_pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        blur_pass.attachments.colors[0] = self.ssao_blur_att_view;
        sg.beginPass(blur_pass);

        sg.applyPipeline(self.ssao_blur_pipeline);
        var bind2 = sg.Bindings{};
        bind2.vertex_buffers[0] = self.quad_vb;
        bind2.index_buffer = self.quad_ib;
        bind2.views[blur_shd.VIEW_ssao_tex] = denoise_src;
        bind2.views[blur_shd.VIEW_depth_tex] = args.depth_tex_view;
        bind2.samplers[blur_shd.SMP_smp] = self.blur_sampler;
        sg.applyBindings(bind2);

        const half = ssao_mod.aoHalfSize(cur_w, cur_h);
        const ao_w: f32 = @floatFromInt(half.w);
        const ao_h: f32 = @floatFromInt(half.h);
        const blur_params = blur_shd.FsParams{
            .resolution = .{
                ao_w,
                ao_h,
                1.0 / ao_w,
                1.0 / ao_h,
            },
            .camera_params = .{
                camera.getNear(),
                camera.getFar(),
                config.radius,
                0.0,
            },
        };
        sg.applyUniforms(blur_shd.UB_fs_params, sg.asRange(&blur_params));
        sg.draw(0, 6, 1);
        sg.endPass();
        return result;
    }

    fn destroyTemporal(self: *SSAOPass) void {
        if (self.temp_a_att_view.id != 0) sg.destroyView(self.temp_a_att_view);
        if (self.temp_a_tex_view.id != 0) sg.destroyView(self.temp_a_tex_view);
        if (self.temp_a_image.id != 0) sg.destroyImage(self.temp_a_image);
        self.temp_a_image = .{};
        self.temp_a_att_view = .{};
        self.temp_a_tex_view = .{};
        if (self.temp_b_att_view.id != 0) sg.destroyView(self.temp_b_att_view);
        if (self.temp_b_tex_view.id != 0) sg.destroyView(self.temp_b_tex_view);
        if (self.temp_b_image.id != 0) sg.destroyImage(self.temp_b_image);
        self.temp_b_image = .{};
        self.temp_b_att_view = .{};
        self.temp_b_tex_view = .{};
    }

    pub fn deinit(self: *SSAOPass) void {
        if (self.ssao_raw_image.id != 0) {
            sg.destroyImage(self.ssao_raw_image);
            sg.destroyView(self.ssao_raw_att_view);
            sg.destroyView(self.ssao_raw_tex_view);

            sg.destroyImage(self.ssao_blur_image);
            sg.destroyView(self.ssao_blur_att_view);
            sg.destroyView(self.ssao_blur_tex_view);
        }
        self.destroyTemporal();
        if (self.noise_image.id != 0) {
            sg.destroyImage(self.noise_image);
            sg.destroyView(self.noise_tex_view);
        }
        sg.destroySampler(self.noise_sampler);
        sg.destroySampler(self.depth_sampler);
        sg.destroySampler(self.blur_sampler);
        sg.destroySampler(self.linear_sampler);
        sg.destroyPipeline(self.ssao_pipeline);
        sg.destroyPipeline(self.ssao_blur_pipeline);
        sg.destroyPipeline(self.temporal_pipeline);
        if (self.ssao_shader.id != 0) sg.destroyShader(self.ssao_shader);
        if (self.blur_shader.id != 0) sg.destroyShader(self.blur_shader);
        if (self.temporal_shader.id != 0) sg.destroyShader(self.temporal_shader);
        self.ssao_shader = .{};
        self.blur_shader = .{};
        self.temporal_shader = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};

// --- Headless regression tests (pure handle routing, no sg.* calls) ---

test "ssao temporal velocity bind never aliases the write slot" {
    var p = SSAOPass{
        .ssao_raw_tex_view = .{ .id = 7 },
        .temp_a_tex_view = .{ .id = 11 },
        .temp_b_tex_view = .{ .id = 12 },
    };
    const vel = sg.View{ .id = 42 };
    // Live velocity always wins (even with valid history).
    try std.testing.expectEqual(vel, p.temporalVelocityBind(vel, true, true));
    try std.testing.expectEqual(vel, p.temporalVelocityBind(vel, true, false));
    // Frame 0: read=A(11), write=B(12) — the fallback is the READ slot.
    const fallback_valid = p.temporalVelocityBind(.{}, false, true);
    try std.testing.expectEqual(@as(u32, 11), fallback_valid.id);
    try std.testing.expect(fallback_valid.id != 12);
    // Reset frame (no valid history): the raw gather placeholder.
    const fallback_prime = p.temporalVelocityBind(.{}, false, false);
    try std.testing.expectEqual(@as(u32, 7), fallback_prime.id);
    try std.testing.expect(fallback_prime.id != 12);
    // Odd frame: read=B(12), write=A(11) — the fallback tracks parity.
    p.temp_frame = 1;
    const fallback_odd = p.temporalVelocityBind(.{}, false, true);
    try std.testing.expectEqual(@as(u32, 12), fallback_odd.id);
    try std.testing.expect(fallback_odd.id != 11);
}

test "ssao resetTemporalHistory clears the cursor headless" {
    var p = SSAOPass{};
    p.temp_frame = 5;
    p.temp_has_history = true;
    p.temp_enabled_prev = true;
    p.temp_mode = .horizon;
    p.temp_phase = 6;
    p.resetTemporalHistory();
    try std.testing.expectEqual(@as(u64, 0), p.temp_frame);
    try std.testing.expect(!p.temp_has_history);
    try std.testing.expect(!p.temp_enabled_prev);
    try std.testing.expectEqual(ssao_mod.AoMode.hemisphere, p.temp_mode);
    try std.testing.expectEqual(@as(u32, 0), p.temp_phase);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao_mod.aoTemporalRotation(p.temp_phase, .hemisphere), 1e-7);
    try std.testing.expect(!p.temporalSized());
    try std.testing.expectEqual(@as(u64, 0), p.temporalDrawCount());
}

test "ssao temporal inactivity resets the rotation phase headless" {
    // Every fail-closed/off path funnels through noteTemporalInactive, so
    // the next active gather reprimes unjittered (phase 0) instead of
    // resuming a stale rotation without history.
    var p = SSAOPass{
        .temp_has_history = true,
        .temp_enabled_prev = true,
        .temp_phase = 5,
    };
    p.noteTemporalInactive();
    try std.testing.expect(!p.temp_has_history);
    try std.testing.expect(!p.temp_enabled_prev);
    try std.testing.expectEqual(@as(u32, 0), p.temp_phase);
    // Fail-closed/off gathers upload exactly 0.0 (CPU mirror of
    // extra_params.z); the default shader arithmetic is untouched.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao_mod.aoTemporalRotation(p.temp_phase, .hemisphere), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao_mod.aoTemporalRotation(p.temp_phase, .horizon), 1e-7);
}

test "ssao projection change forces temporal reset" {
    try std.testing.expect(!SSAOPass.aoProjectionChanged(1.33, 60, 0.1, 100, 1.33, 60, 0.1, 100));
    try std.testing.expect(SSAOPass.aoProjectionChanged(1.33, 60, 0.1, 100, 1.78, 60, 0.1, 100));
    try std.testing.expect(SSAOPass.aoProjectionChanged(1.33, 60, 0.1, 100, 1.33, 45, 0.1, 100));
    try std.testing.expect(SSAOPass.aoProjectionChanged(1.33, 60, 0.1, 100, 1.33, 60, 0.2, 100));
    try std.testing.expect(SSAOPass.aoProjectionChanged(1.33, 60, 0.1, 100, 1.33, 60, 0.1, 200));
    try std.testing.expect(SSAOPass.aoProjectionChanged(0, 0, 0, 0, 1.33, 60, 0.1, 100));
}

test "ssao render result defaults to no temporal draw" {
    const r = SSAOPass.RenderResult{};
    try std.testing.expect(!r.temporal_executed);
}
