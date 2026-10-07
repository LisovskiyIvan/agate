const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const postprocess = @import("../postprocess.zig");
const gpu_thread = @import("../gpu_thread.zig");
const gpu_timing = @import("../gpu_timing.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const scene_msaa = @import("msaa.zig");
const scene_draw = @import("draw.zig");

fn msSince(t0: u64) f32 {
    return @floatCast(sokol.time.ms(sokol.time.since(t0)));
}

/// Mesh-vanish diagnostic probe (symptom: ALL meshes disappear from the
/// main view while particles and the skybox keep rendering).
///
/// Logs ONE rate-limited line (first hit, then ≤ once per 600 frames)
/// when the PRIMARY view's opaque draw queues are empty — leftover
/// transparent items or busy secondary views no longer mask a partial
/// vanish (the 2026-09-22 live repro stayed silent under the old
/// "every queue of every view empty" condition). Also logs:
/// - ARMED once per process: liveness proof that this path executed in
///   the very run (closes the "probe never ran" hypothesis);
/// - RECOVERED when the queues refill (what un-stuck it).
///
/// Evidence mapping per line:
/// - live(total/visible/mask_pass/pending_gpu): total=0 → meshes removed
///   from the scene outright; total>0 + visible=0 → mass is_visible-hide
///   (the old probe returned SILENT on visible==0 — that gap is closed);
///   mask_pass=0 → mask cull; pending_gpu=visible → deferred GPU upload
///   pileup.
/// - prim + queues sums: primary-only emptiness vs all-view totals →
///   partial vanish vs full.
/// - build stats total/rendered/culled/occluded + occluders/tris:
///   rendered>0 with empty queues → items dropped AFTER the count
///   (append OOM) or draws issued but not rasterized (screen black with
///   healthy counters); culled==total → frustum/LOD (vp=NaN/Inf poisons
///   the planes); occluded>0 → occlusion culler (user A/B 2026-09-22:
///   SHIFT+O OFF did NOT restore meshes — ruled out as THE cause; a line
///   would re-confirm).
/// - cam mask/enabled/vp + reuse/streak: culling_mask desync,
///   non-finite view_proj, stale reuse-replay.
///
/// Cost when healthy: ~8 integer length loads (primary lens + sums over 8
/// retained view queues) + one flag check; the census loop and all
/// formatting run only when the empty predicate is already true. No
/// allocations, no panics, no state changes beyond the probe's own
/// cursor/flags. Camera-less frames return before this point.
fn probeMeshVanish(scene: anytype, draws: anytype, snap: anytype) void {
    const p_opaque: usize = draws.primary.items.items.len;
    const p_opaque_inst: usize = draws.primary.opaque_instanced.items.len;
    var n_opaque: usize = p_opaque;
    var n_opaque_inst: usize = p_opaque_inst;
    var n_trans: usize = draws.primary.transparent.items.len;
    var n_trans_inst: usize = draws.primary.transparent_instanced.items.len;
    for (0..draws.views.len) |i| {
        const q = &draws.views[i];
        n_opaque += q.items.items.len;
        n_opaque_inst += q.opaque_instanced.items.len;
        n_trans += q.transparent.items.len;
        n_trans_inst += q.transparent_instanced.items.len;
    }
    if (!scene.mesh_vanish_armed) {
        scene.mesh_vanish_armed = true;
        std.log.warn(
            "[mesh-vanish-probe] ARMED frame={} prim(opaque={} opaque_inst={}) sums(opaque={} opaque_inst={} trans={} trans_inst={})",
            .{ snap.frame_id, p_opaque, p_opaque_inst, n_opaque, n_opaque_inst, n_trans, n_trans_inst },
        );
    }
    // Vanish predicate: the PRIMARY view's opaque queues are empty (the
    // user-visible symptom is meshes-gone with skybox/particles/UI alive;
    // leftover transparent items or busy secondary views must not mask a
    // partial vanish). Healthy frames stop here (~8 loads + flag check).
    const vanished = p_opaque == 0 and p_opaque_inst == 0;
    if (!vanished) {
        if (scene.mesh_vanish_active) {
            scene.mesh_vanish_active = false;
            std.log.warn(
                "[mesh-vanish-probe] RECOVERED frame={} queues refilled (opaque={} opaque_inst={} trans={} trans_inst={})",
                .{ snap.frame_id, n_opaque, n_opaque_inst, n_trans, n_trans_inst },
            );
        }
        return;
    }
    // Empty primary opaque queues: census the live meshes. This branch
    // ALWAYS logs now — `visible=0` with `total>0` IS the mass
    // is_visible-hide fingerprint (the old probe returned silent there),
    // `total=0` means meshes were removed from the scene outright.
    const pmask = snap.primary_cam.culling_mask;
    var visible: usize = 0;
    var mask_pass: usize = 0;
    var pending_gpu: usize = 0;
    for (scene.meshes.items) |m| {
        if (m.is_lod_child or !m.is_visible) continue;
        visible += 1;
        if (m.gpu_pending) {
            pending_gpu += 1;
            continue;
        }
        if ((m.layer_mask & pmask) != 0) mask_pass += 1;
    }
    scene.mesh_vanish_active = true;
    // Rate limit: first hit + at most once per 600 frames (wrapping-safe).
    if (scene.mesh_vanish_last_log_frame != 0 and snap.frame_id -% scene.mesh_vanish_last_log_frame < 600) return;
    scene.mesh_vanish_last_log_frame = snap.frame_id;
    var vp_state: []const u8 = "ok";
    for (snap.primary_cam.view_proj.m) |v| {
        if (std.math.isNan(v)) {
            vp_state = "NaN";
            break;
        }
        if (std.math.isInf(v)) vp_state = "Inf";
    }
    const multi = snap.enable_multi_camera and snap.camera_count > 0;
    const cam_enabled = if (multi and snap.active_camera_idx < snap.camera_count)
        snap.cameras[snap.active_camera_idx].enabled
    else
        snap.primary_cam.enabled;
    std.log.warn(
        "[mesh-vanish-probe] VANISH frame={} live(total={} visible={} mask_pass={} pending_gpu={}) prim(opaque={} opaque_inst={}) queues(opaque={} opaque_inst={} trans={} trans_inst={}) build(total={} rendered={} culled={} occluded={} occluders={} occ_tris={}) cam(idx={}/{} multi={} mask=0x{x} enabled={} vp={s}) reuse={} streak={}",
        .{
            snap.frame_id,
            scene.meshes.items.len,
            visible,
            mask_pass,
            pending_gpu,
            p_opaque,
            p_opaque_inst,
            n_opaque,
            n_opaque_inst,
            n_trans,
            n_trans_inst,
            scene.stats.total_meshes,
            scene.stats.rendered_meshes,
            scene.stats.culled_meshes,
            scene.stats.occluded_meshes,
            scene.stats.occluders_count,
            scene.stats.occluder_triangles,
            snap.active_camera_idx,
            snap.camera_count,
            multi,
            snap.primary_cam.culling_mask,
            cam_enabled,
            vp_state,
            scene.rendering_reuse,
            scene.reuse_streak,
        },
    );
}

/// Core scene frame presentation pass.
pub fn render(scene: anytype) void {
    gpu_thread.assertOnContextThread();
    // Presentation consumes a prepared frame only. The host must explicitly
    // build and latch, or choose renderReuse; rendering never reads live state
    // to manufacture a frame when no producer generation is available.
    if (!scene.frame_prepared and !scene.rendering_reuse) return;
    scene.frame_prepared = false;
    // Конец кадра (P3): epoch, начатый в staged prepare, закрывается на ВСЕХ
    // выходах render — включая ранний возврат без камеры ниже. Поэтому
    // epoch — на кадр, а не на камеру/view.
    defer scene.gpu_retire.complete(scene.retire_epoch);

    // P7: consume the published front slot (const payloads only) under a
    // consumer pin: the presenting render holds the lease for the whole
    // draw, so a future concurrent producer could already build the next
    // frame without reclaiming this one. Sequential today (prepare and
    // render never overlap, and the pin is released by the defer below
    // before any later prepare), so the pin is protocol exercise, not a
    // behavior change: back-slot selection still sees exactly the same
    // free set it would without the pin (the front is excluded either
    // way). Unpin is mandatory — the defer covers every return below.
    const pinned_idx = scene.draws.pinFront();
    defer scene.draws.unpin(pinned_idx) catch {};
    // Valid for this render; the next staged prepare invalidates it (the
    // pin only extends CPU-slot reuse exclusion, never GPU consumability
    // — see scene/frame_draws.zig).
    const draws = scene.preparedDraws();
    // Wave 27 slot-owned snapshot: the whole draw below reads the front
    // slot's staged copy — never the live `frame_snapshot` — so a
    // concurrent game-side mutation cannot tear the in-flight frame.
    // Staged verbatim at prepare, so sequential usage is bit-identical.
    const snap = &draws.snapshot;
    if (!snap.has_camera) {
        // Камеры нет — UI/debug-проходов не будет: переносим только
        // prepare-фазу динамики, чтобы счётчик не утёк в следующий кадр.
        scene.stats.updated_bytes_frame = upload_meter.takeAndReset();
        // Кадровый command buffer уже мог быть открыт prepare-фазой
        // (compute-диспетч частиц): commit и здесь — иначе кадр взял
        // in-flight semaphore и никогда его не вернёт, а sg_shutdown
        // ждёт SIG_NUM_INFLIGHT_FRAMES сигналов безусловно и виснет
        // (наблюдалось как редкий зависание на выходе: 1 кадр из ~1245,
        // waits=commits+1 в инструментированном прогоне). commit с nil
        // buffer — no-op; headless (sg не поднят) пропускаем целиком.
        if (sg.isvalid()) sg.commit();
        return;
    }

    // Mesh-vanish probe: log-only, rate-limited (see probeMeshVanish
    // above). Runs on the consumed front slot before any draw.
    probeMeshVanish(scene, draws, snap);

    const cur_w = if (snap.screen_w > 0) snap.screen_w else sapp.width();
    const cur_h = if (snap.screen_h > 0) snap.screen_h else sapp.height();

    var post_config = snap.post_process.forFrame();
    const samples = scene.postfx.prepareMainTargets(&post_config, snap.msaa_sample_count, cur_w, cur_h) catch |err| {
        _ = scene.warn_main_target.warn("HDR main target unavailable ({s}); frame skipped", .{@errorName(err)});
        if (sg.isvalid()) sg.commit();
        return;
    };
    const color_format: sg.PixelFormat = .RGBA16F;
    if (scene.taa_reset_requested.swap(false, .acq_rel)) {
        scene.postfx.taa_explicit_reset = true;
    }
    if (snap.msaa_sample_count > 1 and samples == 1) {
        // Only the runtime format gate can nullify a > 1 request here
        // (clamping lands on a valid count, post-off forces 1 upstream).
        _ = scene.warn_msaa_format.warn(
            "msaa: x{} unavailable for the HDR main target on this backend; running 1x",
            .{snap.msaa_sample_count},
        );
    }

    // TAA sub-pixel jitter (context thread, render-owned): the snapshot
    // view_proj stays UNJITTERED (prepare built queues/culling from it,
    // i.e. the conservative unjittered frustum); the jittered matrix
    // below drives the main-pass draws and the postfx reprojection for
    // this frame so depth/color/history line up. The index is the
    // snapshot frame_id, so a reused frame repeats its jitter instead of
    // advancing history against identical content. Forced off under MSAA
    // (no depth resolve for the velocity term; PostFXStack forces the
    // composite side off the same way).
    const taa_on = post_config.taa_enabled and samples == 1;
    var taa_view_proj = snap.primary_cam.view_proj;
    if (taa_on) {
        const jpx = postprocess.taaJitter(snap.frame_id, post_config.taa_jitter_scale);
        taa_view_proj = postprocess.applyTaaJitterToViewProj(snap.primary_cam.view_proj, jpx, cur_w, cur_h);
    }

    // 1. Directional Light Cascaded Shadow View-Projections
    const cascades = snap.cascades;
    const light_pack = snap.light_pack;

    // ==============================================
    // PASS 1: OFFSCREEN SHADOW DEPTH PASS
    // ==============================================
    if (snap.shadows_enabled) {
        const t_shadow = sokol.time.now();
        // GPU timer bracket (v2, default off): fail-closed no-op while
        // disabled/headless, linked no-op on Metal (frame timer only).
        gpu_timing.beginPass(.shadow);
        const shadow_draws = scene.shadows.pass.renderPreparedFrom(
            &draws.shadow,
            cascades,
            light_pack.spot_shadows[0..light_pack.num_spot_shadows],
            light_pack.point_shadows[0..light_pack.num_point_shadows],
        );
        gpu_timing.endPass(.shadow);
        scene.stats.shadow_draw_calls += shadow_draws;
        scene.stats.draw_calls += shadow_draws;
        scene.stats.shadow_ms = msSince(t_shadow);
    }

    // ==============================================
    // PASS 1.5: REFLECTION-PROBE CAPTURE (at most one dirty probe)
    // ==============================================
    // Runs after the shadow depth pass (captured draws reuse its maps)
    // and before the main pass. renderReuse re-presents the consumed
    // front — including its probe snapshot — and must NOT capture here
    // (checked below); camera-less frames return before this point, so
    // they skip capture too (dirty flags are retained for later).
    if (!scene.rendering_reuse) {
        scene.captureDirtyProbes(snap);
    }

    // ==============================================
    // PASS 1.6: 3D-GUI PANEL CAPTURE (at most one dirty panel)
    // ==============================================
    // Same shape as the probe capture above: on-demand, context thread,
    // skipped by renderReuse (dirty flags are retained for later) and by
    // camera-less frames (which return before this point). With no dirty
    // panels this is one pure-CPU null check — zero sg.* calls.
    if (!scene.rendering_reuse) {
        scene.captureDirtyUi3dPanels();
    }

    // ==============================================
    // PASS 1.7: MSAA DEPTH PREPASS (gated, default off)
    // ==============================================
    // Single-sample depth for the post chain under MSAA (sokol has no
    // depth resolve — see scene/msaa.zig): redraws the opaque
    // primary-view queues with depth-only pipelines into a 1x depth
    // texture BEFORE the main pass. Skipped on renderReuse replays (the
    // persisted texture still matches the replayed snapshot); the gate
    // going idle frees the target so the off shape holds no prepass
    // VRAM. Cost is attributed to the main phase below.
    const t_main = sokol.time.now();
    var env = scene_draw.Environment{
        .pipelines = scene.forwardFor(samples, color_format),
        .stats = &scene.stats,
        .default_white = snap.default_white,
        .default_normal = snap.default_normal,
        .default_cube = snap.default_cube,
        .sky_texture = snap.sky_texture,
        .ibl_intensity = snap.ibl_intensity,
        .probes = snap.probe_pack.entries[0..snap.probe_pack.count],
        .shadow_pass = &scene.shadows.pass,
        .shadow_uniforms = snap.shadow_uniforms,
        .clustered = &scene.clustered,
    };
    @import("refraction.zig").capture(scene, draws, snap, &env);
    gpu_timing.beginPass(.main);
    const depth_prepass = scene_msaa.depthPrepassActive(post_config.enabled, snap.msaa_depth_prepass, samples);
    if (depth_prepass and !scene.rendering_reuse) {
        scene.postfx.renderMsaaDepthPrepass(
            snap.primary_cam.view_proj,
            &draws.primary,
            draws.primary.skin_storage.items,
            snap.primary_cam.viewport,
            cur_w,
            cur_h,
            &scene.stats,
        );
    } else if (!depth_prepass) {
        scene.postfx.destroyMsaaDepth();
    }

    // ==============================================
    // PASS 2: MAIN SCENE RENDER PASS
    // ==============================================
    var main_pass_action = sg.PassAction{};
    main_pass_action.colors[0] = .{
        .load_action = .CLEAR,
        .clear_value = .{
            .r = snap.clear_color.r,
            .g = snap.clear_color.g,
            .b = snap.clear_color.b,
            .a = snap.clear_color.a,
        },
    };
    main_pass_action.depth = .{
        .load_action = .CLEAR,
        .clear_value = 1.0,
        .store_action = .STORE,
    };

    // Linear HDR is the sole scene target; display conversion runs afterwards.
    // (The .main GPU-timer bracket opened above at PASS 1.7, so the
    // prepass cost attributes to the main phase.)
    if (!scene.postfx.beginMainPass(main_pass_action, samples, cur_w, cur_h)) {
        gpu_timing.endPass(.main);
        sg.commit();
        return;
    }

    if (snap.enable_multi_camera and snap.camera_count > 0) {
        const active_idx = snap.active_camera_idx;
        const primary_snap = if (active_idx < snap.camera_count) snap.cameras[active_idx] else snap.primary_cam;
        const primary_rect = primary_snap.viewport.toPixelRect(cur_w, cur_h);
        sg.applyViewport(primary_rect.x, primary_rect.y, primary_rect.width, primary_rect.height, true);
        sg.applyScissorRect(primary_rect.x, primary_rect.y, primary_rect.width, primary_rect.height, true);
        // Only the TAA view (primary) is jittered; secondary views keep
        // their snapshot matrices.
        var primary_jittered = primary_snap;
        if (taa_on) primary_jittered.view_proj = taa_view_proj;
        // Primary view always uploads clustered slot 0 (single-view
        // frames touch only this slot — no VRAM/behavior change there).
        scene.renderSceneView(primary_jittered, &draws.primary, draws.outline_items.items, draws.outline_skins.items, samples, snap, env, 0);

        // Secondary views compact into slots 1+ in loop order (the skip
        // below already compacts enabled non-active views; 1 + up to 7
        // secondaries fit the 8 camera/slot budget exactly). Each view
        // uploads only its own slot, so no buffer is updated twice in
        // this frame (sokol one-update-per-buffer rule).
        var secondary_ordinal: usize = 0;
        for (snap.cameras[0..snap.camera_count], 0..) |entry, i| {
            if (i == active_idx or !entry.enabled) continue;
            secondary_ordinal += 1;
            const rect = entry.viewport.toPixelRect(cur_w, cur_h);
            sg.applyViewport(rect.x, rect.y, rect.width, rect.height, true);
            sg.applyScissorRect(rect.x, rect.y, rect.width, rect.height, true);

            if (entry.clear_viewport) {
                const clr = entry.clear_color orelse snap.clear_color;
                scene.viewport_clear.clear(clr, samples, color_format);
            }

            scene.renderSceneView(entry, &draws.views[i], draws.outline_items.items, draws.outline_skins.items, samples, snap, env, secondary_ordinal);
        }
        // Restore full viewport
        sg.applyViewport(0, 0, cur_w, cur_h, true);
        sg.applyScissorRect(0, 0, cur_w, cur_h, true);
    } else {
        const vp = snap.primary_cam.viewport;
        const rect = vp.toPixelRect(cur_w, cur_h);
        sg.applyViewport(rect.x, rect.y, rect.width, rect.height, true);
        sg.applyScissorRect(rect.x, rect.y, rect.width, rect.height, true);

        var primary_jittered = snap.primary_cam;
        if (taa_on) primary_jittered.view_proj = taa_view_proj;
        scene.renderSceneView(primary_jittered, &draws.primary, draws.outline_items.items, draws.outline_skins.items, samples, snap, env, 0);

        if (rect.width != cur_w or rect.height != cur_h or rect.x != 0 or rect.y != 0) {
            sg.applyViewport(0, 0, cur_w, cur_h, true);
            sg.applyScissorRect(0, 0, cur_w, cur_h, true);
        }
    }

    sg.endPass();
    gpu_timing.endPass(.main);
    scene.stats.main_ms = msSince(t_main);

    // Velocity buffer pass: per-object screen-space motion vectors, exactly
    // when the composite consumes them (TAA at 1x, or 1x motion blur; under
    // MSAA the pass is suppressed and the blur uses camera depth
    // reprojection). The pass borrows the just-rendered single-sample MAIN
    // depth attachment (LOAD, writes off, EQUAL) instead of replaying its
    // own depth: nearer morph-displaced and opaque main pixels occlude
    // farther velocity draws, and cutout holes fail the match. The frozen
    // prev payloads are generation-checked inside against the last rendered
    // frame, so a staged-but-unrendered prev collapses to zero motion here.
    const velocity_needed = @import("../passes/velocity_pass.zig").velocityNeeded(
        taa_on,
        post_config.motion_blur_enabled,
        samples,
    ) and cur_w > 0 and cur_h > 0 and sg.isvalid();
    // Primary view rect (same mapping the main pass drew under, both
    // single- and multi-camera paths): the velocity draws must land on the
    // exact pixels whose main depth they test against.
    const velocity_viewport = if (snap.enable_multi_camera and snap.camera_count > 0)
        (if (snap.active_camera_idx < snap.camera_count) snap.cameras[snap.active_camera_idx].viewport else snap.primary_cam.viewport)
    else
        snap.primary_cam.viewport;
    // Gated on the ensure RESULT (never the published size): a failed
    // ensure must neither render into a stale-sized target nor feed a stale
    // view to the composite.
    var velocity_ok = false;
    if (velocity_needed) {
        const vrect = velocity_viewport.toPixelRect(cur_w, cur_h);
        sg.applyViewport(vrect.x, vrect.y, vrect.width, vrect.height, true);
        sg.applyScissorRect(vrect.x, vrect.y, vrect.width, vrect.height, true);
        defer {
            sg.applyViewport(0, 0, cur_w, cur_h, true);
            sg.applyScissorRect(0, 0, cur_w, cur_h, true);
        }
        if (scene.postfx.velocity_pass.ensure(cur_w, cur_h)) {
            const main_depth_view = scene.postfx.postprocess_pass.offscreen_depth_att_view;
            const main_depth_fmt = scene.postfx.postprocess_pass.depth_format;
            if (main_depth_view.id != 0 and main_depth_fmt != .DEFAULT and main_depth_fmt != .NONE) {
                const prev_vp = if (scene.postfx.has_prev_view_proj) scene.postfx.prev_view_proj else taa_view_proj;
                scene.postfx.velocity_pass.render(
                    taa_view_proj,
                    prev_vp,
                    &draws.primary,
                    draws.primary.skin_storage.items,
                    draws.primary.prev_skin_storage.items,
                    main_depth_view,
                    main_depth_fmt,
                    @as(u64, scene.last_rendered_frame.load(.acquire)),
                    &scene.stats,
                );
                velocity_ok = true;
            }
        }
    }

    // ==============================================
    // PASS 2.5 (SSAO) + 2.75 (bloom) + 2.8 (glow) + 2.85 (highlight) + 3 (composite & UI overlay)
    // ==============================================
    const t_post = sokol.time.now();
    gpu_timing.beginPass(.post);
    // Highlight mask viewport: the primary view's rect (the same viewport
    // the main pass drew under above, both single- and multi-camera
    // paths) — the PASS 2.85 mask maps it onto its half-res target so
    // PIP/sub-viewports composite aligned. Secondary views never feed it.
    const highlight_viewport = if (snap.enable_multi_camera and snap.camera_count > 0)
        (if (snap.active_camera_idx < snap.camera_count) snap.cameras[snap.active_camera_idx].viewport else snap.primary_cam.viewport)
    else
        snap.primary_cam.viewport;
    scene.postfx.renderChain(.{
        .post = post_config,
        .ssao = snap.ssao,
        .camera = snap.primary_cam.camera,
        .aspect = snap.primary_cam.aspect,
        // Jittered when TAA is on (same matrix the color pass drew
        // with); otherwise exactly the snapshot matrix as before.
        .view_proj = taa_view_proj,
        .eye = snap.primary_cam.eye,
        .sun_dir = snap.sun_dir,
        .sun_color = snap.sun_color,
        .default_white_view = snap.default_white.view,
        .main_samples = samples,
        // PASS 1.7 depth-prepass gate (snapshot-carried Scene flag).
        .msaa_depth_prepass = snap.msaa_depth_prepass,
        // P7 staged highlight items from the pinned front slot (never live
        // Scene fields): the render below dereferences no mesh.
        .highlight_items = draws.highlight_items.items,
        // Primary view's viewport for the PASS 2.85 mask (computed above).
        .highlight_viewport = highlight_viewport,
        // PASS 2.9 volumetric shafts: live CSM atlas view (context thread
        // owns it this frame — PASS 1 rendered into it above) plus the
        // snapshot's cascades/splits/bias and the shadows gate. The pass
        // fail-closes on an empty view, so pre-init/headless stays safe.
        .shaft_shadow_view = scene.shadows.pass.texture_view,
        .shaft_cascades = snap.cascades,
        .shaft_splits = snap.shadow_uniforms.splits,
        .shaft_shadow_bias = snap.shadow_uniforms.bias,
        .shadows_enabled = snap.shadows_enabled,
        .ui = if (scene.ui_frame.canvas_present) &scene.ui_frame else null,
        .stats = &scene.stats,
        // Velocity composite gate: only a rendered-this-frame target feeds
        // the TAA/blur resolve; anything else binds the zero fallback.
        .velocity_valid = velocity_ok,
    }, cur_w, cur_h);
    gpu_timing.endPass(.post);

    sg.commit();
    // Actually presented: stamp the rendered slot generation so staged
    // velocity prev payloads can generation-match it on later draws.
    // Reuse restamps the same value (idempotent); skipped presents never
    // reach here (early returns above).
    scene.last_rendered_frame.store(@intCast(draws.frame_id), .release);
    // GPU timing (opt-in, default off): last COMPLETED samples observed
    // here, right after the present commit, while the just-submitted work
    // is still the latest submission. Optional samples distinguish
    // unavailable (null -> duration 0, submit 0) from a valid quantized
    // zero (submit != 0, ms == 0). A skipped shadow phase stores no sample,
    // never a stale one. Each channel keeps its own submission id — pass
    // values are never summed to fabricate a frame duration. Scopes are
    // backend-specific (see gpu_timing.capabilities).
    if (gpu_timing.pollFrameSample()) |sample| {
        scene.stats.gpu_frame_ms = sample.ms;
        scene.stats.gpu_frame_submit = sample.frame_index;
    } else {
        scene.stats.gpu_frame_ms = 0;
        scene.stats.gpu_frame_submit = 0;
    }
    scene.stats.gpu_frame_scope = gpu_timing.capabilities().frame_scope;
    if (scene.stats.gpu_frame_submit == 0) scene.stats.gpu_frame_scope = .none;
    if (snap.shadows_enabled) {
        if (gpu_timing.pollPassSample(.shadow)) |sample| {
            scene.stats.gpu_shadow_ms = sample.ms;
            scene.stats.gpu_shadow_submit = sample.frame_index;
        } else {
            scene.stats.gpu_shadow_ms = 0;
            scene.stats.gpu_shadow_submit = 0;
        }
    } else {
        scene.stats.gpu_shadow_ms = 0;
        scene.stats.gpu_shadow_submit = 0;
    }
    if (gpu_timing.pollPassSample(.main)) |sample| {
        scene.stats.gpu_main_ms = sample.ms;
        scene.stats.gpu_main_submit = sample.frame_index;
    } else {
        scene.stats.gpu_main_ms = 0;
        scene.stats.gpu_main_submit = 0;
    }
    if (gpu_timing.pollPassSample(.post)) |sample| {
        scene.stats.gpu_post_ms = sample.ms;
        scene.stats.gpu_post_submit = sample.frame_index;
    } else {
        scene.stats.gpu_post_ms = 0;
        scene.stats.gpu_post_submit = 0;
    }
    scene.stats.post_ms = msSince(t_post);

    // Перенос динамики в кадровую метрику: staged begin (flush, стейджинг,
    // UI/debug upload'ы) уже накоплена в счётчике со staged begin, сюда
    // добавились только clear-append'ы main-прохода выше. После take
    // счётчик чист для следующего кадра.
    scene.stats.updated_bytes_frame += upload_meter.takeAndReset();

    if (scene.profiler.isRecording() and !scene.rendering_reuse) {
        scene.profiler_frame_seq +%= 1;
        scene.profiler.recordFrame(scene.profiler_frame_seq, &scene.stats);
    }
}

/// Non-blocking render-consumer reuse: re-draws the current front slot
/// without a prepare, for frames where the app skipped the phase-lock
/// acquire (lock contended) instead of stalling the present.
pub fn renderReuse(scene: anytype) void {
    gpu_thread.assertOnContextThread();
    std.debug.assert(!scene.frame_prepared);
    std.debug.assert(scene.draws.slots[scene.draws.front].frame_id != 0);
    scene.reuse_streak += 1;
    const saved_stats = scene.stats;
    scene.rendering_reuse = true;
    defer scene.rendering_reuse = false;
    scene.render();
    // Reuse replays the saved CPU frame: restore the CPU stats, but keep
    // the CURRENT (post-render) GPU samples — the replay's commit may have
    // surfaced newly completed submissions, and restoring the pre-render
    // GPU values would silently drop them for the profiler record below.
    // (recordFrame itself does not run inside render() on this path: the
    // `rendering_reuse` guard skips it there, so the record below is the
    // only one — with restored CPU stats + fresh GPU metadata.)
    const fresh_gpu_frame_ms = scene.stats.gpu_frame_ms;
    const fresh_gpu_frame_submit = scene.stats.gpu_frame_submit;
    const fresh_gpu_frame_scope = scene.stats.gpu_frame_scope;
    const fresh_gpu_shadow_ms = scene.stats.gpu_shadow_ms;
    const fresh_gpu_shadow_submit = scene.stats.gpu_shadow_submit;
    const fresh_gpu_main_ms = scene.stats.gpu_main_ms;
    const fresh_gpu_main_submit = scene.stats.gpu_main_submit;
    const fresh_gpu_post_ms = scene.stats.gpu_post_ms;
    const fresh_gpu_post_submit = scene.stats.gpu_post_submit;
    scene.stats = saved_stats;
    scene.stats.gpu_frame_ms = fresh_gpu_frame_ms;
    scene.stats.gpu_frame_submit = fresh_gpu_frame_submit;
    scene.stats.gpu_frame_scope = fresh_gpu_frame_scope;
    scene.stats.gpu_shadow_ms = fresh_gpu_shadow_ms;
    scene.stats.gpu_shadow_submit = fresh_gpu_shadow_submit;
    scene.stats.gpu_main_ms = fresh_gpu_main_ms;
    scene.stats.gpu_main_submit = fresh_gpu_main_submit;
    scene.stats.gpu_post_ms = fresh_gpu_post_ms;
    scene.stats.gpu_post_submit = fresh_gpu_post_submit;
    if (scene.profiler.isRecording()) {
        scene.profiler_frame_seq +%= 1;
        scene.profiler.recordFrame(scene.profiler_frame_seq, &scene.stats);
    }
}
