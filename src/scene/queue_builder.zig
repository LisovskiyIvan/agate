const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;

const mesh = @import("../mesh.zig");
const InstanceSource = mesh.InstanceSource;
const CubeTexture = @import("../texture.zig").CubeTexture;
const SceneStats = @import("stats.zig").SceneStats;
const scene_snapshot = @import("snapshot.zig");
const CameraSnapshot = scene_snapshot.CameraSnapshot;
const SceneFrameSnapshot = scene_snapshot.SceneFrameSnapshot;
const scene_render_queue = @import("render_queue.zig");
const RenderQueues = scene_render_queue.RenderQueues;
const RenderMeshItem = scene_render_queue.RenderMeshItem;
const outline_pass = @import("../passes/outline_pass.zig");
const highlight_pass = @import("../passes/highlight_pass.zig");
const frame_draws = @import("frame_draws.zig");
const FrameDrawSlot = frame_draws.FrameDrawSlot;
const jobs = @import("../jobs.zig");

pub const QueueBuildParams = struct {
    snap: *const SceneFrameSnapshot,
    cache_key: u64,
    stats: *SceneStats,
    eye: Vec3,
    sky_texture: ?CubeTexture,
    ibl_intensity: f32,
    instances_prepared: bool,
    instance_source: InstanceSource,
};

pub fn prepareViewQueues(
    scene: anytype,
    queues: *RenderQueues,
    cam_snap: CameraSnapshot,
    sky_texture: ?CubeTexture,
    ibl_intensity: f32,
    cache_key: u64,
    stats: *SceneStats,
    instances_prepared: bool,
    instance_source: InstanceSource,
) void {
    queues.reset();

    scene_render_queue.buildFrameQueues(.{
        .allocator = scene.allocator,
        .meshes = scene.meshes.items,
        .cache_key = cache_key,
        .instance_source = instance_source,
        .view_proj = cam_snap.view_proj,
        .eye = cam_snap.eye,
        .cull_frustum = scene.enable_frustum_culling,
        .cull_occlusion = scene.enable_occlusion_culling,
        .culling_mask = cam_snap.culling_mask,
        .occlusion_culler = &scene.occlusion_culler,
        .stats = stats,
        .queues = queues,
        .default_white_id = scene.default_white_texture.view.id,
        .default_material = &scene.default_material,
        .default_white = &scene.default_white_texture,
        .default_normal = &scene.default_normal_texture,
        .default_cube = &scene.default_cube_texture,
        .default_brdf_lut = &scene.default_brdf_lut_texture,
        .sky_texture = sky_texture,
        .ibl_intensity = ibl_intensity,
        .default_morph_view = scene.forward.default_morph_view,
        .thread_pool = jobs.global,
        // P5: grown instance buffers retire into the epoch queue; the
        // pre-stage above is definitive for this frame, so view builds
        // never retry staging mid-frame (failure coherence).
        .gpu_retire = &scene.gpu_retire,
        .instances_prepared = instances_prepared,
    });

    std.mem.sort(RenderMeshItem, queues.items.items, {}, scene_render_queue.sortRenderItems);
    // Unified transparent order: regular + instanced groups globally
    // back-to-front by group distance (one entry per batch).
    std.mem.sort(
        scene_render_queue.TransparentDrawEntry,
        queues.transparent_order.items,
        {},
        scene_render_queue.sortTransparentDrawOrder,
    );
}

/// Shared queue/shadow/outline builder: fills a `FrameDrawSlot` from the
/// explicit `params` in one internal function, callable from the
/// single-producer `buildIntoClaimedSlot` (with `&build_snapshot` +
/// `.build_view` + build-unique cache key + `&build_stats`) and from
/// standalone immediate builds (tests/tooling with the staged slot
/// snapshot + `.published` + frame_id + `&self.stats`). Reads cameras
/// ONLY from `params.snap` (never `self.frame_snapshot`/
/// `self.build_snapshot` directly), so the build generation stays frozen. Producer `.build_view`
/// freezes on the snapshot shadow switch alone; standalone keeps the
/// explicit `shadows.enabled` gate.
/// Does NOT reset `back` (the
/// caller reset before consume, as before). sg-free when
/// `instances_prepared=true` (view builds never retry staging mid-frame).
pub fn buildQueuesInto(scene: anytype, back: *FrameDrawSlot, params: QueueBuildParams) void {
    const snap = params.snap;
    // `eye` is informational today: views sort by their own snapshot eye
    // (see prepareViewQueues); the build passes the live eye for future
    // transparent-sort use. The instance CPU staging eye is threaded
    // separately (stageInstancesCpu / InstanceStageContext).
    _ = params.eye;

    // P5: outline capture is unconditional, as before — immediately
    // after the conditional pre-stage above and before conditional
    // shadow/view warming below. With a GPU context instanced items
    // snapshot the resolved staged state (`.published` = this frame's
    // `instance_render`, `.build_view` = provisional `instance_build_view`
    // until the latch patch); regular items are captured before
    // worldMatrixCached warming, exactly like the historical pre-queue
    // capture, so their cached-center behavior is unchanged.
    //
    // Outline identity domain (stage-2B fix): `source_mesh` MUST be the
    // mesh-list index into `self.meshes.items` (patch resolves against
    // that list with uid validation). The outline-list position is a
    // different domain (subset, any order) and MUST NOT be stored. Each
    // outline mesh is resolved to its mesh-list index here; an outline
    // mesh absent from the mesh list gets the OOB sentinel `meshes.len`
    // (deterministic, still emitted so the standalone immediate build — which never patches
    // — keeps shape; the latch patch fail-closes the sentinel via
    // its OOB branch).
    for (scene.outline_meshes.items) |m| {
        if (m.gpu_pending or !m.is_visible or m.index_count == 0) continue;
        var src_idx: u32 = @intCast(scene.meshes.items.len);
        for (scene.meshes.items, 0..) |sm, si| {
            if (sm == m) {
                src_idx = @intCast(si);
                break;
            }
        }
        if (outline_pass.makeOutlineDrawItem(scene.allocator, &back.outline_skins, m, params.cache_key, src_idx, params.instance_source)) |it| {
            back.outline_items.append(scene.allocator, it) catch {
                back.build_stats.build_oom_drops += 1;
            };
        }
    }

    // Highlight layer v1 capture (same phase-locked prepare point as the
    // outline loop above, unconditional like it — zero highlights stage
    // zero items and the whole downstream chain stays bit-identical).
    // Identity domain matches outline (`source_mesh` = mesh-list index,
    // OOB sentinel = meshes.len), but — unlike outline — OOB referents
    // are SKIPPED here: highlights have no latch patch stage
    // (patch_instance_refs only patches instanced batch refs; highlight
    // items stage no instance buffers by design), so emitting them would
    // only feed the mask pass dead handles. destroyMesh drops the entry
    // synchronously, so OOB is defense-in-depth only. Skinned meshes are
    // skipped inside makeHighlightDrawItem (no skin matrix staging v1);
    // instanced meshes stage the template proxy (documented v1 limit).
    for (scene.highlights.entries[0..scene.highlights.count]) |*e| {
        if (e.mesh.gpu_pending or !e.mesh.is_visible or e.mesh.index_count == 0) continue;
        var src_idx: u32 = @intCast(scene.meshes.items.len);
        for (scene.meshes.items, 0..) |sm, si| {
            if (sm == e.mesh) {
                src_idx = @intCast(si);
                break;
            }
        }
        if (src_idx >= scene.meshes.items.len) continue;
        if (highlight_pass.makeHighlightDrawItem(e.mesh, e.options, src_idx)) |it| {
            back.highlight_items.append(scene.allocator, it) catch {
                back.build_stats.build_oom_drops += 1;
            };
        }
    }

    const is_gpu_init = (scene.default_white_texture.view.id != 0);
    if (!is_gpu_init) return;

    // Shadow pass preparation into the back slot (disabled shadows —
    // or no camera — leave the reset-empty payload: coherent, never
    // the front slot's prior bins). Producer `.build_view` freezes on the
    // snapshot switch alone; standalone keeps the explicit live gate.
    if (snap.has_camera and snap.shadows_enabled and (params.instance_source == .build_view or scene.shadows.enabled)) {
        _ = scene.shadows.pass.prepareInto(&back.shadow, scene.meshes.items, params.cache_key, params.instance_source, jobs.global);
    }

    // View queues preparation (params carry the resolved sky/ibl).
    if (snap.has_camera) {
        if (snap.enable_multi_camera and snap.camera_count > 0) {
            const active_idx = snap.active_camera_idx;
            const primary_snap = if (active_idx < snap.camera_count) snap.cameras[active_idx] else snap.primary_cam;
            prepareViewQueues(scene, &back.primary, primary_snap, params.sky_texture, params.ibl_intensity, params.cache_key, params.stats, params.instances_prepared, params.instance_source);
            for (snap.cameras[0..snap.camera_count], 0..) |entry, i| {
                if (i == active_idx or !entry.enabled) continue;
                prepareViewQueues(scene, &back.views[i], entry, params.sky_texture, params.ibl_intensity, params.cache_key, params.stats, params.instances_prepared, params.instance_source);
            }
        } else {
            prepareViewQueues(scene, &back.primary, snap.primary_cam, params.sky_texture, params.ibl_intensity, params.cache_key, params.stats, params.instances_prepared, params.instance_source);
        }
    }
}
