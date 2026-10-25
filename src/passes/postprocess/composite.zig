//! Pure composite bindings and uniform packing. GPU resources remain owned by
//! PostProcessPass; the pass is passed as anytype to avoid a leaf-to-owner import.
const sokol = @import("sokol");
const sg = sokol.gfx;
const post_shd = @import("postprocess_shader");
const postprocess = @import("../../postprocess.zig");
const PostProcessOptions = postprocess.PostProcessOptions;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color3 = math.Color3;

pub fn isSrgbBackbuffer(fmt: sg.PixelFormat) bool {
    return fmt == .SRGB8A8 or fmt == .SBGR8A8;
}

pub fn outputParamsFor(backbuffer_fmt: sg.PixelFormat) [4]f32 {
    // The attachment format owns transfer encoding on every backend.
    // Canvas colorSpace does not make an UNORM attachment encode linear
    // shader output; only an sRGB attachment performs that conversion.
    const x: f32 = if (isSrgbBackbuffer(backbuffer_fmt)) 0.0 else 1.0;
    return .{ x, 0.0, 0.0, 0.0 };
}

pub fn buildCompositeBindings(
    self: anytype,
    config: PostProcessOptions,
    ssao_tex: sg.View,
    depth_view: sg.View,
    taa_history_view: sg.View,
    velocity_view: sg.View,
    ssgi_albedo_tex: sg.View,
) sg.Bindings {
    var post_bind = sg.Bindings{};
    post_bind.vertex_buffers[0] = self.postprocess_quad_vb;
    post_bind.index_buffer = self.postprocess_quad_ib;
    post_bind.views[post_shd.VIEW_scene_tex] = self.offscreen_resolve_tex_view;
    post_bind.views[post_shd.VIEW_ssao_tex] = ssao_tex;
    post_bind.views[post_shd.VIEW_depth_tex] = depth_view;
    // SSGI albedo (velocity MRT): bound whenever the parent fed it; the
    // shader samples it only while ssgi_albedo_params.x gates the lane on,
    // so the scene-view placeholder keeps the off path untouched.
    post_bind.views[post_shd.VIEW_ssgi_albedo_tex] = if (ssgi_albedo_tex.id != 0)
        ssgi_albedo_tex
    else
        self.offscreen_resolve_tex_view;
    // Pyramid glow when the parent fed a BloomPass result; otherwise a
    // valid placeholder the shader never samples (params3.z = 0 gates
    // the bloom block off).
    post_bind.views[post_shd.VIEW_bloom_tex] = if (self.bloom_tex_view.id != 0)
        self.bloom_tex_view
    else
        self.offscreen_resolve_tex_view;
    // Glow halo when the parent fed a GlowPass result; otherwise a valid
    // placeholder the shader never samples (glow_params.x = 0 gates the
    // glow block off, so the off path is bit-identical to pre-glow).
    post_bind.views[post_shd.VIEW_glow_tex] = if (self.glow_tex_view.id != 0)
        self.glow_tex_view
    else
        self.offscreen_resolve_tex_view;
    // Highlight halo when the parent fed a HighlightPass result;
    // otherwise a valid placeholder the shader never samples
    // (highlight_params.x = 0 gates the highlight block off, so the
    // off path is bit-identical to pre-highlight).
    post_bind.views[post_shd.VIEW_highlight_tex] = if (self.highlight_tex_view.id != 0)
        self.highlight_tex_view
    else
        self.offscreen_resolve_tex_view;
    // Highlight raw mask (inner-glow minuend) when the parent fed it;
    // otherwise the same harmless placeholder (unread while gated off).
    post_bind.views[post_shd.VIEW_highlight_mask_tex] = if (self.highlight_mask_tex_view.id != 0)
        self.highlight_mask_tex_view
    else
        self.offscreen_resolve_tex_view;
    // Shaft radiance when the parent fed a VolumetricPass result;
    // otherwise a valid placeholder the shader never samples
    // (shaft_params.x = 0 gates the shaft block off, so the off path
    // is bit-identical to pre-shaft).
    post_bind.views[post_shd.VIEW_shaft_tex] = if (self.shaft_tex_view.id != 0)
        self.shaft_tex_view
    else
        self.offscreen_resolve_tex_view;
    // SSGI v2 denoised bleed when the parent fed an SsgiPass result;
    // otherwise a valid placeholder the shader never samples
    // (ssgi_pass_params.x = 0 gates the branch off, so the off path is
    // bit-identical to pre-SSGI-pass).
    post_bind.views[post_shd.VIEW_ssgi_tex] = if (self.ssgi_tex_view.id != 0)
        self.ssgi_tex_view
    else
        self.offscreen_resolve_tex_view;
    // SSR v2 denoised reflection when the parent fed an SsrPass result;
    // otherwise a valid placeholder the shader never samples
    // (ssr_pass_params.x = 0 gates the branch off, so the off path is
    // bit-identical to pre-SSR-pass).
    post_bind.views[post_shd.VIEW_ssr_tex] = if (self.ssr_tex_view.id != 0)
        self.ssr_tex_view
    else
        self.offscreen_resolve_tex_view;
    // LUT when the config carries a live binding; otherwise the resolved
    // scene view as a valid placeholder the shader never samples
    // (lut_params.x = 0 gates the LUT branch off).
    const lut_view: sg.View = if (config.lut.texture) |t| t.view else .{};
    post_bind.views[post_shd.VIEW_lut_tex] = if (lut_view.id != 0)
        lut_view
    else
        self.offscreen_resolve_tex_view;
    // TAA history (bilinear smp in-shader); a valid placeholder the
    // shader never samples while history is invalid or TAA is off
    // (taa_params.x/taa_state.x gate the branch off).
    post_bind.views[post_shd.VIEW_history_tex] = if (taa_history_view.id != 0)
        taa_history_view
    else
        self.offscreen_resolve_tex_view;
    // Screen-space velocity buffer: per-object motion vector + alpha mask;
    // binds default_zero_view when no velocity view was fed so the shader
    // falls back to depth reprojection cleanly.
    post_bind.views[post_shd.VIEW_velocity_tex] = if (velocity_view.id != 0)
        velocity_view
    else
        self.default_zero_view;
    post_bind.samplers[post_shd.SMP_smp] = self.postprocess_sampler;
    post_bind.samplers[post_shd.SMP_depth_smp] = self.depth_sampler;
    post_bind.samplers[post_shd.SMP_velocity_smp] = self.velocity_sampler;
    post_bind.samplers[post_shd.SMP_lut_smp] = self.lut_sampler;
    return post_bind;
}

pub fn packCompositeParams(
    self: anytype,
    config: PostProcessOptions,
    ssao_enabled: bool,
    ssao_intensity: f32,
    cur_w: i32,
    cur_h: i32,
    view_proj: Mat4,
    inv_view_proj: Mat4,
    prev_view_proj: Mat4,
    camera_pos: Vec3,
    sun_dir: Vec3,
    sun_color: Color3,
    near_z: f32,
    far_z: f32,
    taa_history_valid: bool,
    taa_capture_only: bool,
    backbuffer_fmt: sg.PixelFormat,
    // UNJITTERED reprojection (prev_unjittered * inv(curr_unjittered)):
    // reproj_mat drives the TAA history lookup and the motion-blur
    // fallback, and both must live in unjittered screen space — the jitter
    // belongs to the raster only. A jittered reproj displaces the displayed
    // frame by the per-frame jitter delta (still-camera wobble / crawl).
    reproj_mat_unjittered: Mat4,
    // SSGI albedo validity: gates the real-albedo SSGI tint; 0 keeps the
    // lit-color proxy exactly as before.
    ssgi_albedo_valid: bool,
) post_shd.FsParams {
    // Superseded by reproj_mat_unjittered; kept for signature stability.
    _ = prev_view_proj;
    return post_shd.FsParams{
        .params1 = .{
            config.exposure,
            config.bloom.threshold,
            config.bloom.intensity,
            0.0,
        },
        .params2 = .{
            config.vignette.intensity,
            config.vignette.radius,
            config.grade.saturation,
            config.grade.contrast,
        },
        .params3 = .{
            @floatFromInt(@backingInt(config.tonemapping)),
            config.grade.chromatic_aberration,
            if (config.bloom.enabled and self.bloom_tex_view.id != 0) 1.0 else 0.0,
            if (config.vignette.enabled) 1.0 else 0.0,
        },
        .ssao_params = postprocess.ssaoParams(ssao_enabled, ssao_intensity),
        .fxaa_params = postprocess.fxaaParams(config.fxaa.enabled),
        .resolution = .{
            @floatFromInt(cur_w),
            @floatFromInt(cur_h),
            1.0 / @as(f32, @floatFromInt(cur_w)),
            1.0 / @as(f32, @floatFromInt(cur_h)),
        },
        // (near_z, far_z, 0, 0): the SSR step count used to ride in z
        // and now has its own ssr_params2 lane below.
        .camera_params = .{
            near_z,
            far_z,
            0.0,
            0.0,
        },
        .camera_pos = .{
            camera_pos.x,
            camera_pos.y,
            camera_pos.z,
            0.0,
        },
        .sun_dir = .{
            sun_dir.x,
            sun_dir.y,
            sun_dir.z,
            0.0,
        },
        .sun_color = .{
            sun_color.r,
            sun_color.g,
            sun_color.b,
            0.0,
        },
        .fog_params = .{
            if (config.fog.enabled) 1.0 else 0.0,
            config.fog.density,
            config.fog.height_falloff,
            config.fog.start_distance,
        },
        .fog_color = .{
            config.fog.color[0],
            config.fog.color[1],
            config.fog.color[2],
            config.fog.sun_scattering,
        },
        .ssr_params = postprocess.ssrParams(config),
        // (steps, 0, 0, 0); zeros when off OR in v2 pass mode, which keeps
        // the composite identical to the SSR-off path (the inline raymarch
        // branch never runs alongside the pass-sample branch).
        .ssr_params2 = postprocess.ssrStepsParams(config),
        // (pass-sample active 1/0, intensity, 0, 0); zeros unless the
        // parent fed a denoised SsrPass view this frame, which keeps
        // the off path bit-identical.
        .ssr_pass_params = postprocess.ssrPassParams(config, self.ssr_tex_view.id != 0),
        .params5 = .{
            if (config.sharpen.enabled) config.sharpen.intensity else 0.0,
            if (config.grain.enabled) config.grain.intensity else 0.0,
            config.white_balance.temperature,
            config.white_balance.tint,
        },
        .dof_params = .{
            if (config.dof.enabled) 1.0 else 0.0,
            config.dof.focus_distance,
            config.dof.focus_range,
            config.dof.max_blur,
        },
        // (enabled 1/0, intensity, 0, 0); zeros when glow is off, which
        // keeps the composite identical to the pre-glow path.
        .glow_params = postprocess.glowParams(config),
        // (tint rgb, 0); zeros when glow is off (unread while disabled).
        .glow_tint = postprocess.glowTintParams(config),
        // (enabled 1/0, baked global scale 1.0, 0, 0); zeros when the
        // parent fed no highlight view, which keeps the composite
        // identical to the pre-highlight path.
        .highlight_params = postprocess.highlightParams(self.highlight_tex_view.id != 0),
        // (enabled 1/0, intensity, 0, 0); zeros when the shaft is off,
        // which keeps the composite identical to the pre-shaft path.
        .shaft_params = postprocess.shaftParams(config, self.shaft_tex_view.id != 0),
        // (enabled 1/0, intensity, distance, thickness); zeros when off
        .contact_shadow_params = postprocess.contactShadowParams(config),
        // (steps, 0, 0, 0); zeros when off, which keeps the composite
        // identical to the pre-contact-shadow path.
        .contact_shadow_params2 = postprocess.contactShadowStepsParams(config),
        // (enabled 1/0, intensity, radius, steps); zeros when off OR in
        // v2 pass mode, which keeps the composite identical to the
        // pre-SSGI path (the inline gather branch never runs alongside
        // the pass-sample branch).
        .ssgi_params = postprocess.ssgiParams(config),
        // (pass-sample active 1/0, intensity, 0, 0); zeros unless the
        // parent fed a denoised SsgiPass view this frame, which keeps
        // the off path bit-identical.
        .ssgi_pass_params = postprocess.ssgiPassParams(config, self.ssgi_tex_view.id != 0),
        // (enabled 1/0, intensity, contrast, 0); zeros when off
        .local_tonemap_params = postprocess.localTonemapParams(config),
        .grade_shadows = .{
            config.grade.shadows[0],
            config.grade.shadows[1],
            config.grade.shadows[2],
            0.0,
        },
        .grade_midtones = .{
            config.grade.midtones[0],
            config.grade.midtones[1],
            config.grade.midtones[2],
            0.0,
        },
        .grade_highlights = .{
            config.grade.highlights[0],
            config.grade.highlights[1],
            config.grade.highlights[2],
            0.0,
        },
        // (enabled 1/0, intensity, size N, 0); zeros when no LUT, which
        // keeps the composite identical to the pre-LUT path.
        .lut_params = postprocess.lutParams(config),
        .view_proj = view_proj,
        .inv_view_proj = inv_view_proj,
        .reproj_mat = reproj_mat_unjittered,
        .motion_blur_params = .{
            if (config.motion_blur.enabled) 1.0 else 0.0,
            // Former motion_blur_intensity knob (dropped: zero non-default
            // setters): the historical default, hardcoded.
            0.5,
            config.motion_blur.max_blur_px,
            @floatFromInt(config.motion_blur.steps),
        },
        // (enabled 1/0, blend, clamp, sharpness); zeros when TAA is off,
        // which keeps the composite identical to the pre-TAA path.
        .taa_params = postprocess.taaParams(config),
        // (history_valid 1/0, capture_only 1/0, scene_upscale 1/0): the
        // upscale bit gates the composite's Catmull-Rom scene reconstruction
        // while the history/output size exceeds the scene target
        // (render_scale < 1); at native — and everywhere else — it stays 0
        // and the base fetch is the untouched single bilinear sample.
        .taa_state = blk: {
            const tw = if (self.taa_width > 0) self.taa_width else cur_w;
            const th = if (self.taa_height > 0) self.taa_height else cur_h;
            break :blk postprocess.taaState(
                taa_history_valid,
                taa_capture_only,
                postprocess.sceneUpscaleEnabled(tw, th, cur_w, cur_h),
            );
        },
        // (albedo_valid 1/0, 0, 0, 0): gates the real-albedo SSGI tint;
        // 0 keeps the lit-color proxy exactly as before.
        .ssgi_albedo_params = .{ if (ssgi_albedo_valid) 1.0 else 0.0, 0.0, 0.0, 0.0 },
        // (reactive_enabled 1/0, strength, 0, 0); zeros when off, which
        // keeps the composite identical to the pre-reactive path (the
        // shader multiplies a zero boost into the blend).
        .taa_reactive_params = postprocess.taaReactiveParams(config),
        // Output lane (appended last): x = manual display-encode flag
        // iff the backbuffer default is NOT sRGB (UNORM encodes the
        // exact piecewise sRGB once at the end; sRGB targets encode in
        // hardware); y/z/w = 0. The main target is always RGBA16F.
        .output_params = outputParamsFor(backbuffer_fmt),
        // TAA history/output size; at render_scale < 1 the history is native,
        // while `resolution` above stays the low-res scene size.
        .taa_resolution = blk: {
            const tw = if (self.taa_width > 0) self.taa_width else cur_w;
            const th = if (self.taa_height > 0) self.taa_height else cur_h;
            const fw = @as(f32, @floatFromInt(@max(tw, 1)));
            const fh = @as(f32, @floatFromInt(@max(th, 1)));
            break :blk .{ fw, fh, 1.0 / fw, 1.0 / fh };
        },
        .lens_flare_params = postprocess.lensFlareParams(config),
        .lens_flare_params2 = postprocess.lensFlareParams2(config),
    };
}
