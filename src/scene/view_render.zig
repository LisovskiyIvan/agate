const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;

const scene_snapshot = @import("snapshot.zig");
const scene_render_queue = @import("render_queue.zig");
const outline_pass = @import("../passes/outline_pass.zig");
const scene_draw = @import("draw.zig");
const scene_clustered = @import("clustered_lights.zig");
const scene_uniforms = @import("uniforms.zig");
const FrameContext = scene_uniforms.FrameContext;

pub fn renderSceneView(
    scene: anytype,
    cam_snap: scene_snapshot.CameraSnapshot,
    queues: *const scene_render_queue.RenderQueues,
    outline_items: []const outline_pass.OutlineDrawItem,
    outline_skins: []const [scene_render_queue.MAX_BONES]Mat4,
    samples: i32,
    snap: *const scene_snapshot.SceneFrameSnapshot,
    env: scene_draw.Environment,
    view_slot: usize,
) void {
    const view_proj = cam_snap.view_proj;
    const eye = cam_snap.eye;
    // This view's clustered buffer slot (0 = primary, 1+ = secondaries):
    // the view rebuilds the tiles and uploads ONLY its own slot, so no
    // buffer is updated twice in one frame (sokol one-update rule).
    const slot = scene_clustered.ClusteredGpuCache.clampSlot(view_slot);
    // Draws bind this view's slot (real views when its upload landed,
    // else the shared dummy); the shadow-uniform inputs below are
    // view-independent, so the copy only affects clustered binding.
    var view_env = env;
    view_env.clustered_slot = slot;

    // Clustered forward lights (wave 30): rebuild the 2D screen tiles
    // for THIS view from the staged snapshot (context thread; never
    // live light state) and upload the storage buffers (metered). The
    // grid lives on window pixels with this view's rect mapping NDC;
    // every view (PIP included) rebuilds from its own staged camera, so
    // no view inherits another view's tile lists. Empty pool (or a
    // failed rebuild/upload) leaves the descriptor zeroed and binds the
    // dummy, so every draw takes the exact legacy path. Probe captures
    // (renderProbeFace) skip this and upload zeroed lanes
    // (legacy lanes only in v1, documented).
    const cl_count = snap.light_pack.clustered_count;
    var cl_params: [4]f32 = .{ 0, 0, 0, 0 };
    var cl_viewport: [4]f32 = .{ 0, 0, 64, 0 };
    if (cl_count > 0 and scene.rendering_reuse) {
        // renderReuse re-presents the SAME staged snapshot the last
        // real frame built tiles from: keep the existing storage
        // buffers and descriptor instead of rebuilding — a replay must
        // stay upload-free (the reuse fixture asserts zero
        // updateBuffer calls). Liveness is per view slot: a view whose
        // slot never uploaded (pure reuse streak) falls back to the
        // dummy/legacy path until the next real frame, while other
        // views keep replaying their own slots.
        if (scene.clustered.isLive(slot)) {
            cl_params = .{
                @floatFromInt(scene.clustered.tiles_x),
                @floatFromInt(scene.clustered.tiles_y),
                @floatFromInt(cl_count),
                1.0,
            };
            cl_viewport = .{
                @floatFromInt(snap.screen_w),
                @floatFromInt(snap.screen_h),
                @floatFromInt(scene_clustered.tile_size_px),
                0.0,
            };
        } else {
            scene.clustered.ensureDummyViews();
        }
    } else if (cl_count > 0) {
        const rect = cam_snap.viewport.toPixelRect(snap.screen_w, snap.screen_h);
        const view_rect = scene_clustered.ViewRect{ .x = rect.x, .y = rect.y, .w = rect.width, .h = rect.height };
        var rebuilt_ok = true;
        // render domain: funds the render-owned clustered CPU scratch
        // (cpu_lights/headers/indices, freed in ClusteredGpuCache.deinit
        // with render). upload below stays core: its allocator funds appends
        // into the core-owned gpu_retire queue on buffer growth.
        scene.clustered.rebuildCpuFromLights(
            scene.render_allocator,
            snap.light_pack.clustered_lights[0..cl_count],
            view_proj,
            snap.screen_w,
            snap.screen_h,
            view_rect,
            slot,
        ) catch {
            rebuilt_ok = false;
        };
        if (rebuilt_ok and scene.clustered.upload(scene.allocator, &scene.gpu_retire, slot)) {
            cl_params = .{
                @floatFromInt(scene.clustered.tiles_x),
                @floatFromInt(scene.clustered.tiles_y),
                @floatFromInt(cl_count),
                if (scene.clustered.isLive(slot)) 1.0 else 0.0,
            };
            cl_viewport = .{
                @floatFromInt(snap.screen_w),
                @floatFromInt(snap.screen_h),
                @floatFromInt(scene_clustered.tile_size_px),
                0.0,
            };
        } else {
            scene.clustered.ensureDummyViews();
        }
    } else {
        scene.clustered.ensureDummyViews();
    }

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
        .clustered_params = cl_params,
        .clustered_viewport = cl_viewport,
    };

    var shadow_state_with = env.shadow_uniforms;
    shadow_state_with.mesh_receive_shadows = true;
    const u_with = scene_uniforms.buildFrameUniforms(shadow_state_with, &frame_ctx);

    var shadow_state_no = env.shadow_uniforms;
    shadow_state_no.mesh_receive_shadows = false;
    const u_no = scene_uniforms.buildFrameUniforms(shadow_state_no, &frame_ctx);

    frame_ctx.uniforms_with_shadows = &u_with;
    frame_ctx.uniforms_without_shadows = &u_no;

    var current_pipeline_id: u32 = 0;

    // Opaque regular meshes first (front-to-back, early-Z).
    for (queues.items.items) |item| {
        if (env.capture_opaque_only and item.draw_record.refractive) continue;
        scene_draw.drawRegularItem(&view_env, item, &frame_ctx, &current_pipeline_id, queues.skin_storage.items, queues.shader_storage.items, queues.coat_storage.items);
    }

    // Opaque instanced meshes.
    for (queues.opaque_instanced.items) |batch| {
        if (env.capture_opaque_only and batch.draw_record.refractive) continue;
        scene_draw.drawInstancedBatch(&view_env, batch, &frame_ctx, &current_pipeline_id, queues.coat_storage.items);
    }

    // Transparent pass: regular items and instanced groups interleaved in
    // one global back-to-front order (each instanced group draws as a
    // single batch at its sorted position; no per-instance sorting).
    for (queues.transparent_order.items) |entry| {
        if (env.capture_opaque_only) break;
        switch (entry.kind) {
            .regular => {
                if (entry.index < queues.transparent.items.len) {
                    scene_draw.drawRegularItem(&view_env, queues.transparent.items[entry.index], &frame_ctx, &current_pipeline_id, queues.skin_storage.items, queues.shader_storage.items, queues.coat_storage.items);
                }
            },
            .instanced => {
                if (entry.index < queues.transparent_instanced.items.len) {
                    scene_draw.drawInstancedBatch(&view_env, queues.transparent_instanced.items[entry.index], &frame_ctx, &current_pipeline_id, queues.coat_storage.items);
                }
            },
        }
    }

    // 3D GUI panels (wave 28, v1): world-space UI quads drawn after the
    // transparent queue (depth-tested, depth-write-off, double-sided —
    // see scene/gui3d_layer.zig). The layer early-outs on a pure CPU
    // count with zero sg.* calls when no panel is drawable, so
    // panel-less frames are bit-identical. Probe face captures never
    // reach this path (renderProbeFace inlines its own draws).
    if (!env.capture_opaque_only) scene.gui3d.drawPanels(scene.allocator, view_proj, samples, env.pipelines.color_format, env.stats);

    // Inverse-hull outline for highlighted meshes (P7: published slot
    // payload, never live Scene fields).
    if (!env.capture_opaque_only) scene.postfx.renderOutlineItems(
        view_proj,
        eye,
        outline_items,
        outline_skins,
        samples,
        env.pipelines.color_format,
        &scene.stats,
        snap.outline_enabled,
        snap.outline_color,
        snap.outline_width_px,
    );

    // Physics debug lines: prepared capture only — no live world,
    // no show_debug read at draw time (update may step the world
    // concurrently). One committed upload (prepare), one draw per view.
    if (!env.capture_opaque_only) scene.physics.renderDebugPrepared(view_proj, samples, env.pipelines.color_format, &scene.stats);

    // Skybox Pass: captured enabled/texture/exposure only. The cube is
    // the snapshot sky texture orelse the snapshot's render-owned
    // default copy — never self.sky.* / self.default_cube_texture live.
    scene.sky.renderPrepared(
        snap.sky_enabled,
        cam_snap.camera,
        cam_snap.aspect,
        snap.sky_texture orelse snap.default_cube,
        snap.sky_exposure,
        samples,
        env.pipelines.color_format,
        &scene.stats,
    );

    // Particle Pass: prepared frame only — no live ParticleSystem reads
    // at draw time (the update side may step systems concurrently).
    if (!env.capture_opaque_only) scene.particles.renderPrepared(cam_snap.camera, cam_snap.aspect, samples, env.pipelines.color_format, &scene.stats);
}
