const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const postprocess = @import("../postprocess.zig");
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const scene_msaa = @import("msaa.zig");
const scene_draw = @import("draw.zig");

fn msSince(t0: u64) f32 {
    return @floatCast(sokol.time.ms(sokol.time.since(t0)));
}

/// Core scene frame presentation pass.
pub fn render(scene: anytype) void {
    gpu_thread.assertOnContextThread();
    if (!scene.frame_prepared and !scene.rendering_reuse) {
        scene.prepareFrame();
        // Wave-31 counted skip: under lease contention prepare consumes
        // nothing. When a pending frame from an earlier prepare exists
        // (`frame_prepared` kept true by the skip) it is consumed below
        // as usual — still the freshest prepared. With nothing pending
        // (false) there is no retire epoch to complete (the claim sits
        // before it) and no frame to present: drop the present instead
        // of mislabeling the stale front in stats/profiler (latest-wins)
        // and let the next frame retry. A still-open older epoch is
        // closed by the next prepare's `begin`. Sequential behavior is
        // unchanged (prepare never skips there).
        if (!scene.frame_prepared) return;
    }
    scene.frame_prepared = false;
    // Конец кадра (P3): epoch, начатый в prepareFrame, закрывается на ВСЕХ
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
    // Valid for this render; the next prepareFrame invalidates it (the
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

    const cur_w = if (snap.screen_w > 0) snap.screen_w else sapp.width();
    const cur_h = if (snap.screen_h > 0) snap.screen_h else sapp.height();

    // Effective main-target MSAA sample count for this frame
    // (scene/msaa.zig holds the policy and the backend matrix).
    const samples = scene_msaa.effectiveSampleCount(snap.msaa_sample_count, .{
        .post_enabled = snap.post_process.enabled,
        .formats_msaa_capable = scene_msaa.mainTargetFormatsMsaaCapable(),
        .backend = sg.queryBackend(),
    });
    if (snap.post_process.enabled and snap.msaa_sample_count > 1 and samples == 1) {
        // Only the runtime format gate can nullify a > 1 request here
        // (clamping lands on a valid count, post-off forces 1 upstream).
        _ = scene.warn_msaa_format.warn(
            "msaa: x{} requested but the main target formats cannot MSAA on this backend; running 1x",
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
    const taa_on = snap.post_process.enabled and snap.post_process.taa_enabled and samples == 1;
    var taa_view_proj = snap.primary_cam.view_proj;
    if (taa_on) {
        const jpx = postprocess.taaJitter(snap.frame_id, snap.post_process.taa_jitter_scale);
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
        const shadow_draws = scene.shadows.pass.renderPreparedFrom(
            &draws.shadow,
            cascades,
            light_pack.spot_shadows[0..light_pack.num_spot_shadows],
            light_pack.point_shadows[0..light_pack.num_point_shadows],
        );
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

    // Offscreen target when post-processing is on, swapchain otherwise.
    const t_main = sokol.time.now();
    scene.postfx.beginMainPass(main_pass_action, snap.post_process.enabled, samples, cur_w, cur_h);

    // Render-owned draw environment: every fallback below is a snapshot
    // COPY (default textures + sky captured at prepare) — the draw never
    // dereferences game-mutatable Scene.default_*_texture / sky fields,
    // so update may run concurrently with this whole pass. (The
    // default_material pointer is gone: null-material draws were already
    // baked into draw_record at prepare; the draw never needed it.)
    const env = scene_draw.Environment{
        // Pipeline set must match the main target's sample count: the
        // 1x set for the legacy/swapchain path, the MSAA twin otherwise.
        // Render-owned lazy caches (forward_msaa, clear_*, sky/debug/
        // particle/outline MSAA twins, postfx targets, shader-material
        // cache): touched ONLY on this context thread in prepare/render,
        // never by update — safe under overlap given the thread-safe
        // Scene allocator (GPA .thread_safe = true); no prewarm needed
        // just because the fields live in Scene.
        .pipelines = if (samples > 1) scene.ensureForwardMsaa(samples) else &scene.forward,
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
        scene.renderSceneView(primary_jittered, &draws.primary, draws.outline_items.items, draws.outline_skins.items, samples, snap, env);

        for (snap.cameras[0..snap.camera_count], 0..) |entry, i| {
            if (i == active_idx or !entry.enabled) continue;
            const rect = entry.viewport.toPixelRect(cur_w, cur_h);
            sg.applyViewport(rect.x, rect.y, rect.width, rect.height, true);
            sg.applyScissorRect(rect.x, rect.y, rect.width, rect.height, true);

            if (entry.clear_viewport) {
                const clr = entry.clear_color orelse snap.clear_color;
                scene.viewport_clear.clear(clr, samples);
            }

            scene.renderSceneView(entry, &draws.views[i], draws.outline_items.items, draws.outline_skins.items, samples, snap, env);
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
        scene.renderSceneView(primary_jittered, &draws.primary, draws.outline_items.items, draws.outline_skins.items, samples, snap, env);

        if (rect.width != cur_w or rect.height != cur_h or rect.x != 0 or rect.y != 0) {
            sg.applyViewport(0, 0, cur_w, cur_h, true);
            sg.applyScissorRect(0, 0, cur_w, cur_h, true);
        }
    }

    if (!snap.post_process.enabled) {
        // P6: single fullscreen UI draw AFTER the per-view
        // viewport/scissor restoration above (both single- and
        // multi-camera paths restore before this point). Draw site
        // selection reads ONLY the prepared presence flag — never the
        // live canvas — so the snapshot boundary is complete at
        // prepare; the frame itself carries no canvas reference.
        // Counter semantics unchanged (including the historical +1
        // whenever a canvas existed at prepare, even for an empty
        // frame).
        if (scene.ui_frame.canvas_present) {
            scene.ui_frame.drawPrepared();
            scene.stats.post_draw_calls += 1;
            scene.stats.draw_calls += 1;
        }
    }

    sg.endPass();
    scene.stats.main_ms = msSince(t_main);

    // ==============================================
    // PASS 2.5 (SSAO) + 2.75 (bloom) + 2.8 (glow) + 2.85 (highlight) + 3 (composite & UI overlay)
    // ==============================================
    const t_post = sokol.time.now();
    // Highlight mask viewport: the primary view's rect (the same viewport
    // the main pass drew under above, both single- and multi-camera
    // paths) — the PASS 2.85 mask maps it onto its half-res target so
    // PIP/sub-viewports composite aligned. Secondary views never feed it.
    const highlight_viewport = if (snap.enable_multi_camera and snap.camera_count > 0)
        (if (snap.active_camera_idx < snap.camera_count) snap.cameras[snap.active_camera_idx].viewport else snap.primary_cam.viewport)
    else
        snap.primary_cam.viewport;
    scene.postfx.renderChain(.{
        .post = snap.post_process,
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
        // P7 staged highlight items from the pinned front slot (never live
        // Scene fields): the render below dereferences no mesh.
        .highlight_items = draws.highlight_items.items,
        // Primary view's viewport for the PASS 2.85 mask (computed above).
        .highlight_viewport = highlight_viewport,
        .ui = if (scene.ui_frame.canvas_present) &scene.ui_frame else null,
        .stats = &scene.stats,
    }, cur_w, cur_h);

    sg.commit();
    scene.stats.post_ms = msSince(t_post);

    // Перенос динамики в кадровую метрику: prepare-фаза (flush, стейджинг,
    // UI/debug upload'ы) уже накоплена в счётчике с prepareFrame, сюда
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
    scene.stats = saved_stats;
    if (scene.profiler.isRecording()) {
        scene.profiler_frame_seq +%= 1;
        scene.profiler.recordFrame(scene.profiler_frame_seq, &scene.stats);
    }
}
