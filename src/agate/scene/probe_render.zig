const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;

const blit_probe_shd = @import("probe_mip_shader");
const scene_snapshot = @import("snapshot.zig");
const SceneFrameSnapshot = scene_snapshot.SceneFrameSnapshot;
const scene_draw = @import("draw.zig");
const FrameContext = scene_draw.FrameContext;
const scene_uniforms = @import("uniforms.zig");
const scene_stats = @import("stats.zig");
const SceneStats = scene_stats.SceneStats;
const scene_probes = @import("probe_layer.zig");
const FrameDrawSlot = @import("frame_draws.zig").FrameDrawSlot;

/// Runs at most ONE pending reflection probe capture. Render-
/// local: draws the primary view queues from the probe position into the
/// probe's cube target (mip 0), then runs the GGX/irradiance convolution
/// chain to populate the remaining mips. The main-pass viewport is
/// untouched. When several probes are dirty, the lowest dirty + enabled
/// index captures now and the rest wait for later frames (one capture
/// per frame maximum). The fresh content reaches draws one staged prepare
/// later (the snapshot is staged by the producer build, before `render`
/// captures) — a documented one-frame lag.
pub fn captureDirtyProbes(scene: anytype, snap: *const SceneFrameSnapshot) void {
    const idx = scene.probes.nextDirtyIndex() orelse return;
    if (!sg.isvalid()) return;
    if (!scene.probes.ensureGpu(idx)) return;
    scene.clustered.ensureDummyViews();
    const probe = &scene.probes.probes[idx];
    const draws = scene.preparedDraws();
    // Staged slot snapshot (never the live frame_snapshot): the capture
    // renders the same generation the main pass draws.
    const far = scene_probes.captureFar(probe.radius);

    var face_i: u8 = 0;
    while (face_i < 6) : (face_i += 1) {
        const face: scene_probes.Face = @enumFromInt(face_i);
        renderProbeFace(scene, idx, face, draws, snap, far);
    }
    runProbePrefilter(scene, idx);

    // Viewport/scissor restore: the main pass sets its own per-view
    // rects below, but leave no stale 128px state behind regardless.
    const cur_w = if (snap.screen_w > 0) snap.screen_w else sapp.width();
    const cur_h = if (snap.screen_h > 0) snap.screen_h else sapp.height();
    sg.applyViewport(0, 0, cur_w, cur_h, true);
    sg.applyScissorRect(0, 0, cur_w, cur_h, true);

    scene.probes.notifyCaptured(idx);
}

/// Renders the prepared primary draw list plus the sky into one cube
/// face of probe `index` (mip 0, linear HDR RGBA16F). Capture-local state
/// throughout: scratch stats (discarded — the frame's counters must not 7x),
/// the 1x HDR forward set (`forwardFor(1, RGBA16F)`, resolved BEFORE the
/// pass opens — forwardFor may recreate the twin and must never run
/// mid-pass), and an EMPTY probe pack (capture draws take the environment
/// path — no probe self-sampling or feedback). Shadow maps are the frame's
/// own (captured right after the shadow depth pass). Invalid/FAILED probe
/// views fail closed (no pass is opened).
pub fn renderProbeFace(
    scene: anytype,
    index: usize,
    face: scene_probes.Face,
    draws: *const FrameDrawSlot,
    snap: *const SceneFrameSnapshot,
    far: f32,
) void {
    const probe = &scene.probes.probes[index];
    // Resolve the HDR forward set before touching the pass (see above).
    const fwd = scene.forwardFor(1, .RGBA16F);
    // FAILED-state guard: nonzero ids are not logical validity (retired
    // epoch handles keep ids). Refuse the face instead of drawing into a
    // dead attachment.
    if (probe.gpu.mip_face_views[0][@intFromEnum(face)].id == 0 or
        sg.queryViewState(probe.gpu.mip_face_views[0][@intFromEnum(face)]) != .VALID) return;
    if (probe.gpu.depth_view.id == 0 or sg.queryViewState(probe.gpu.depth_view) != .VALID) return;
    const eye = probe.position;
    const view = scene_probes.faceView(face, eye);
    const proj = Mat4.perspective(90.0, 1.0, scene_probes.capture_near, far);
    const view_proj = Mat4.mul(proj, view);

    var pass_action = sg.PassAction{};
    pass_action.colors[0] = .{
        .load_action = .CLEAR,
        .clear_value = .{
            .r = snap.clear_color.r,
            .g = snap.clear_color.g,
            .b = snap.clear_color.b,
            .a = snap.clear_color.a,
        },
    };
    pass_action.depth = .{
        .load_action = .CLEAR,
        .clear_value = 1.0,
        .store_action = .STORE,
    };
    var pass = sg.Pass{ .action = pass_action };
    pass.attachments.colors[0] = probe.gpu.mip_face_views[0][@intFromEnum(face)];
    pass.attachments.depth_stencil = probe.gpu.depth_view;
    sg.beginPass(pass);
    const res = scene_probes.face_resolution;
    sg.applyViewport(0, 0, res, res, true);
    sg.applyScissorRect(0, 0, res, res, true);

    var scratch = SceneStats{};
    const env = scene_draw.Environment{
        .pipelines = fwd,
        .stats = &scratch,
        .default_white = snap.default_white,
        .default_normal = snap.default_normal,
        .default_cube = snap.default_cube,
        .sky_texture = snap.sky_texture,
        .ibl_intensity = snap.ibl_intensity,
        .probes = &.{},
        .shadow_pass = &scene.shadows.pass,
        .shadow_uniforms = snap.shadow_uniforms,
        .clustered = &scene.clustered,
    };

    var frame_ctx = FrameContext{
        .view_proj = view_proj,
        .eye = eye,
        .sun_dir = snap.sun_dir,
        .sun_color = snap.sun_color,
        .sun_intensity = snap.sun_intensity,
        .directional_dir = snap.light_pack.directional_dir,
        .directional_color_int = snap.light_pack.directional_color_int,
        .cascades = snap.cascades,
        .light_counts = snap.light_pack.counts,
        .point_pos_range = snap.light_pack.point_pos_range,
        .point_color_int = snap.light_pack.point_color_int,
        .spot_pos_range = snap.light_pack.spot_pos_range,
        .spot_dir_inner = snap.light_pack.spot_dir_inner,
        .spot_color_outer = snap.light_pack.spot_color_outer,
        .spot_intensity = snap.light_pack.spot_intensity,
        .spot_view_proj = snap.light_pack.spot_view_proj,
        .spot_shadow_params = snap.light_pack.spot_shadow_params,
        .point_view_proj = snap.light_pack.point_view_proj,
        .point_shadow_params = snap.light_pack.point_shadow_params,
        .area_center_int = snap.light_pack.area_center_int,
        .area_right = snap.light_pack.area_right,
        .area_up = snap.light_pack.area_up,
        .area_color = snap.light_pack.area_color,
        // Probe captures render the legacy lanes only (documented v1
        // scope): zeroed descriptor gates the clustered loop off, and
        // the env cache below binds the dummy views.
        .clustered_params = .{ 0, 0, 0, 0 },
        .clustered_viewport = .{ 0, 0, 64, 0 },
    };

    var shadow_state_with = env.shadow_uniforms;
    shadow_state_with.mesh_receive_shadows = true;
    const u_with = scene_uniforms.buildFrameUniforms(shadow_state_with, &frame_ctx);

    var shadow_state_no = env.shadow_uniforms;
    shadow_state_no.mesh_receive_shadows = false;
    const u_no = scene_uniforms.buildFrameUniforms(shadow_state_no, &frame_ctx);

    frame_ctx.uniforms_with_shadows = &u_with;
    frame_ctx.uniforms_without_shadows = &u_no;

    // Primary view queues only (v1 scope); same draw order as the main
    // pass (opaque, opaque-instanced, unified transparent back-to-front).
    // The transparent sort was built for the main eye — order from the
    // probe eye is approximate (documented).
    var current_pipeline_id: u32 = 0;
    const queues = &draws.primary;
    for (queues.items.items) |item| {
        scene_draw.drawRegularItem(&env, item, &frame_ctx, &current_pipeline_id, queues.skin_storage.items, queues.shader_storage.items, queues.coat_storage.items);
    }
    for (queues.opaque_instanced.items) |batch| {
        scene_draw.drawInstancedBatch(&env, batch, &frame_ctx, &current_pipeline_id, queues.coat_storage.items);
    }
    for (queues.transparent_order.items) |entry| {
        switch (entry.kind) {
            .regular => {
                if (entry.index < queues.transparent.items.len) {
                    scene_draw.drawRegularItem(&env, queues.transparent.items[entry.index], &frame_ctx, &current_pipeline_id, queues.skin_storage.items, queues.shader_storage.items, queues.coat_storage.items);
                }
            },
            .instanced => {
                if (entry.index < queues.transparent_instanced.items.len) {
                    scene_draw.drawInstancedBatch(&env, queues.transparent_instanced.items[entry.index], &frame_ctx, &current_pipeline_id, queues.coat_storage.items);
                }
            },
        }
    }

    // Sky last (depth LESS_EQUAL behind geometry), snapshot switch +
    // exposure; disabled sky leaves the clear color. Translation-free
    // view, like SkyboxPass.render.
    if (snap.sky_enabled) {
        var rot_view = view;
        rot_view.m[12] = 0.0;
        rot_view.m[13] = 0.0;
        rot_view.m[14] = 0.0;
        scene.sky.pass.renderMatrices(rot_view, proj, snap.sky_texture orelse snap.default_cube, snap.sky_exposure);
    }
    sg.endPass();
}

/// GGX/irradiance bake chain for probe `index`. Mip routing (fixed by the
/// mip-contamination plan in `probe_layer`):
/// - Pass A renders scratch mip `m` from the probe's mip-0-only bake view
///   at explicit LOD 0: mode 0 (GGX convolution, `roughnessForMip(m)`) for
///   mips 1..`max_mips - 2`, mode 1 (cosine irradiance, E/pi convention)
///   for the coarsest mip. The full-chain sample view is never a bake
///   source — its coarse mips are stale until Pass B lands them.
/// - Pass B copies scratch mip `m` into probe mip `m` (mode 2, exact mip).
/// Roughness ladder, irradiance convention, and probe lifetime/resize
/// rollback are unchanged. A dead blit pipeline fails closed.
pub fn runProbePrefilter(scene: anytype, index: usize) void {
    if (scene.probes.blit_pipeline.id == 0 or sg.queryPipelineState(scene.probes.blit_pipeline) != .VALID) return;
    if (scene.probes.blit_vb.id == 0 or sg.queryBufferState(scene.probes.blit_vb) != .VALID) return;
    if (scene.probes.blit_ib.id == 0 or sg.queryBufferState(scene.probes.blit_ib) != .VALID) return;
    const probe = &scene.probes.probes[index];
    // Bake-source guard: without the mip-0-only view the convolution would
    // sample stale coarse mips of the full chain. Fail closed like above.
    if (probe.gpu.bake_view.id == 0 or sg.queryViewState(probe.gpu.bake_view) != .VALID) return;
    var mip: u32 = 1;
    while (mip < scene_probes.max_mips) : (mip += 1) {
        const size = scene_probes.mipSize(mip);
        // Pass A: Convolve probe.gpu bake view (mip 0) -> scratch_cube (mip)
        var face_i: u8 = 0;
        while (face_i < 6) : (face_i += 1) {
            var pass = sg.Pass{
                .action = .{
                    .colors = [_]sg.ColorAttachmentAction{
                        .{ .load_action = .DONTCARE },
                    } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
                },
            };
            pass.attachments.colors[0] = scene.probes.scratch_mip_face_views[mip][face_i];
            sg.beginPass(pass);
            sg.applyPipeline(scene.probes.blit_pipeline);
            sg.applyViewport(0, 0, size, size, true);
            sg.applyScissorRect(0, 0, size, size, true);

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = scene.probes.blit_vb;
            bind.index_buffer = scene.probes.blit_ib;
            bind.views[blit_probe_shd.VIEW_src_tex] = probe.gpu.bake_view;
            bind.samplers[blit_probe_shd.SMP_smp] = scene.probes.blit_sampler;
            sg.applyBindings(bind);

            const is_irradiance = scene_probes.isIrradianceMip(mip);
            const roughness = scene_probes.roughnessForMip(mip);
            const mode: f32 = if (is_irradiance) 1.0 else 0.0;
            const fs_params = blit_probe_shd.FsParams{
                .params = .{
                    @floatFromInt(face_i),
                    roughness,
                    mode,
                    @floatFromInt(scene_probes.face_resolution),
                },
            };
            sg.applyUniforms(blit_probe_shd.UB_fs_params, sg.asRange(&fs_params));
            sg.draw(0, 6, 1);
            sg.endPass();
        }

        // Pass B: Copy scratch_cube (mip) -> probe.gpu (mip) (mode 2.0 = blit copy)
        face_i = 0;
        while (face_i < 6) : (face_i += 1) {
            var pass = sg.Pass{
                .action = .{
                    .colors = [_]sg.ColorAttachmentAction{
                        .{ .load_action = .DONTCARE },
                    } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
                },
            };
            pass.attachments.colors[0] = probe.gpu.mip_face_views[mip][face_i];
            sg.beginPass(pass);
            sg.applyPipeline(scene.probes.blit_pipeline);
            sg.applyViewport(0, 0, size, size, true);
            sg.applyScissorRect(0, 0, size, size, true);

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = scene.probes.blit_vb;
            bind.index_buffer = scene.probes.blit_ib;
            bind.views[blit_probe_shd.VIEW_src_tex] = scene.probes.scratch_tex_view;
            bind.samplers[blit_probe_shd.SMP_smp] = scene.probes.blit_sampler;
            sg.applyBindings(bind);

            const fs_params = blit_probe_shd.FsParams{
                .params = .{
                    @floatFromInt(face_i),
                    0.0,
                    2.0,
                    @floatFromInt(mip),
                },
            };
            sg.applyUniforms(blit_probe_shd.UB_fs_params, sg.asRange(&fs_params));
            sg.draw(0, 6, 1);
            sg.endPass();
        }
    }
}
