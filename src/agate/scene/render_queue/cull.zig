//! Per-frame culling: world-matrix cache, `FrameCullContext`, plain-mesh
//! cull, material-record build, and queue append. Imports the `items` leaf
//! plus engine modules — never the `render_queue.zig` facade (documented
//! anti-cycle rule). `build` and `instances` import this module; the reverse
//! never happens.
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Mat4 = math.Mat4;
const Frustum = math.Frustum;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../../mesh.zig").Mesh;
const material_mod = @import("../../material.zig");
const Material = material_mod.Material;
const MaterialDrawRecord = material_mod.MaterialDrawRecord;
const StandardMaterial = material_mod.StandardMaterial;
const Texture = @import("../../texture.zig").Texture;
const CubeTexture = @import("../../texture.zig").CubeTexture;
const morph_gpu = @import("../../mesh/morph_gpu.zig");
const visibility = @import("../../visibility/mod.zig");
const jobs = @import("../../jobs.zig");
const stats_mod = @import("../stats.zig");
const SceneStats = stats_mod.SceneStats;
const gpu_retire_mod = @import("../gpu_retire.zig");
const items = @import("items.zig");
const RenderQueues = items.RenderQueues;
const CulledMesh = items.CulledMesh;
const materialIsTransparent = items.materialIsTransparent;
const materialIsDoubleSided = items.materialIsDoubleSided;

/// World matrix computed at most once per cache key (fallback: the frame id
/// on the mesh). Semantics mirror Mesh.getWorldMatrix exactly —
/// including bone attachment, which replaces the parent chain when the
/// host skeleton is live — so every pass sees the same transform. Parent
/// and host chains resolve through the same cache, so hierarchies stay
/// O(depth) total.
///
/// `cache_key` parameterization (stage-2 increment A): the fallback passes
/// `Scene.frame_id`; a future game-side build passes its own key. No new
/// caches, no invalidation change.
pub fn worldMatrixCached(cache_key: u64, mesh: *Mesh) Mat4 {
    if (mesh.cached_frame == cache_key) return mesh.cached_matrix;
    const trs = Mat4.fromRotationTranslationScale(mesh.position, mesh.rotation, mesh.scaling);
    const local = Mat4.mul(trs, mesh.base_matrix);
    var world: Mat4 = undefined;
    if (mesh.attach_bone) |att| {
        if (att.host_mesh.skeleton) |skel| {
            const host_mat = worldMatrixCached(cache_key, att.host_mesh);
            const bone_mat = skel.getBoneWorldMatrix(att.bone_index, host_mat);
            world = Mat4.mul(Mat4.mul(bone_mat, att.offset_matrix), local);
        } else if (mesh.parent) |p| {
            world = Mat4.mul(worldMatrixCached(cache_key, p), local);
        } else {
            world = local;
        }
    } else if (mesh.parent) |p| {
        world = Mat4.mul(worldMatrixCached(cache_key, p), local);
    } else {
        world = local;
    }
    mesh.cached_matrix = world;
    mesh.cached_aabb = mesh.local_bounding_box.transform(world);
    mesh.cached_frame = cache_key;
    return world;
}

pub fn worldAABBCached(cache_key: u64, mesh: *Mesh) BoundingBox {
    if (mesh.cached_frame == cache_key) return mesh.cached_aabb;
    _ = worldMatrixCached(cache_key, mesh);
    return mesh.cached_aabb;
}

/// Everything buildFrameQueues needs from Scene for one frame. Explicit
/// parameters (instead of the legacy `scene: anytype`) keep this module
/// free of scene.zig imports.
pub const FrameCullContext = struct {
    allocator: std.mem.Allocator,
    /// Optional worker pool for the parallel cull pass. Null (or a scene
    /// smaller than `parallel_min_meshes`) keeps the legacy single-threaded
    /// loop — a scheduling detail, never a behavior change: both paths
    /// produce identical queues in identical order.
    thread_pool: ?*jobs.Pool = null,
    /// Scene size at which the parallel cull pays for waking the workers.
    /// Default 128; 0 adapts dynamically to (workerCount + 1) * 32.
    parallel_min_meshes: usize = 128,

    /// View id of the shared 1x1 white fallback (texture-less meshes).
    default_white_id: u32,
    default_material: ?*const StandardMaterial = null,
    default_white: ?*const Texture = null,
    default_normal: ?*const Texture = null,
    default_cube: ?*const CubeTexture = null,
    sky_texture: ?CubeTexture = null,
    ibl_intensity: f32 = 1.0,
    default_morph_view: sg.View = .{},
    meshes: []const *Mesh,
    /// Cache key for worldMatrixCached/worldAABBCached (stage-2 increment A).
    /// Replaces the direct `frame_id` read: the fallback passes
    /// `Scene.frame_id` (identical behavior); a game-side build passes its
    /// own build-unique key `(build_seq | (1<<63))`. No new caches, no
    /// invalidation change.
    cache_key: u64,
    /// Which instance state instanced builders resolve (stage-2 increment B):
    /// `.published` reads `instance_render` (fallback, today's exact
    /// behavior); `.build_view` reads the game-frozen `instance_build_view`
    /// (provisional buffer/count until the latch patch). Threaded through
    /// from `QueueBuildParams.instance_source`; serial + parallel paths both
    /// honor it (the parallel merge tail reuses the same ctx).
    instance_source: @import("../../mesh.zig").InstanceSource = .published,
    view_proj: Mat4,
    eye: Vec3,
    cull_frustum: bool,
    cull_occlusion: bool,
    culling_mask: u32 = 0xFFFFFFFF,
    occlusion_culler: *visibility.OcclusionCuller,
    stats: *SceneStats,
    queues: *RenderQueues,
    /// P5: retire queue for grown instance buffers, threaded into staging by
    /// Scene (null in standalone/test contexts → the old buffer is destroyed
    /// immediately on the context thread; see InstanceStageContext docs).
    gpu_retire: ?*gpu_retire_mod.GpuRetireQueue = null,
    /// P5 revision: true when Scene already pre-staged this frame. The
    /// pre-stage publish is then DEFINITIVE for the frame: view-queue builds
    /// consume the published render state as-is and never retry staging —
    /// a failed pre-stage (scratch OOM, buffer failure) keeps the previous
    /// complete state, and the shadow snapshot taken from it stays coherent
    /// with main/outline even if a retry would have succeeded. Default
    /// false: standalone builds (tests, tooling) stage here, with the
    /// success-gated same-frame retry intact while no snapshot is consumed.
    instances_prepared: bool = false,
};

/// Routes a finished record into the draw queue it belongs to. Transparent
/// regulars also append a unified order entry (index into
/// queues.transparent); instanced transparents append theirs in
/// submitInstancedMesh so both share one mesh-index sequence.
///
/// Перед постановкой заимствования prepare-фазы копируются в render-owned
/// хранилища и item получает только индексы: в очередях не остаётся живых
/// указателей на Mesh/Material/Skeleton. OOM на копии роняет весь item
/// (а не рисует его с живыми матрицами): контракт как у OOM очередей.
///
/// Internal to `render_queue/*` (used by `build`'s serial loop and parallel
/// merge tail); not re-exported by the facade.
pub fn appendRenderItem(ctx: FrameCullContext, culled: CulledMesh) void {
    var item = culled.item;
    if (culled.skin_src) |src| {
        ctx.queues.skin_storage.ensureUnusedCapacity(ctx.allocator, 1) catch return;
        const idx: u32 = @intCast(ctx.queues.skin_storage.items.len);
        ctx.queues.skin_storage.appendAssumeCapacity(src.*);
        item.skin_index = idx;
    }
    if (culled.shader_snap) |snap| {
        ctx.queues.shader_storage.ensureUnusedCapacity(ctx.allocator, 1) catch return;
        const idx: u32 = @intCast(ctx.queues.shader_storage.items.len);
        ctx.queues.shader_storage.appendAssumeCapacity(snap);
        item.shader_index = idx;
    }
    if (culled.coat) |cp| {
        ctx.queues.coat_storage.ensureUnusedCapacity(ctx.allocator, 1) catch return;
        const idx: u32 = @intCast(ctx.queues.coat_storage.items.len);
        ctx.queues.coat_storage.appendAssumeCapacity(cp);
        item.coat_index = idx;
    }
    if (item.transparent) {
        // Reserve both slots up front: if either allocation fails the item
        // is dropped entirely, never an undrawn orphan in one array.
        ctx.queues.transparent.ensureUnusedCapacity(ctx.allocator, 1) catch return;
        ctx.queues.transparent_order.ensureUnusedCapacity(ctx.allocator, 1) catch return;
        const idx: u32 = @intCast(ctx.queues.transparent.items.len);
        const seq: u32 = item.mesh_index;
        ctx.queues.transparent.appendAssumeCapacity(item);
        ctx.queues.transparent_order.appendAssumeCapacity(.{
            .distance_sq = item.distance_sq,
            .seq = seq,
            .kind = .regular,
            .index = idx,
            .is_decal = item.is_decal,
        });
    } else {
        ctx.queues.items.append(ctx.allocator, item) catch {};
    }
}

/// Internal to `render_queue/*` (used by `instances`' submit path); not
/// re-exported by the facade.
pub fn buildMaterialRecord(ctx: FrameCullContext, mat: ?Material) MaterialDrawRecord {
    const dummy_tex = Texture{ .image = .{}, .view = .{ .id = ctx.default_white_id }, .sampler = .{}, .width = 1, .height = 1 };
    const dummy_cube = CubeTexture{ .image = .{}, .view = .{}, .sampler = .{}, .size = 1 };
    const dummy_std = StandardMaterial.init("default");

    const white_tex = ctx.default_white orelse &dummy_tex;
    const norm_tex = ctx.default_normal orelse &dummy_tex;
    const cube_tex = ctx.default_cube orelse &dummy_cube;
    const def_mat = ctx.default_material orelse &dummy_std;

    return material_mod.buildDrawRecord(
        mat,
        def_mat,
        white_tex,
        norm_tex,
        cube_tex,
        ctx.sky_texture,
        ctx.ibl_intensity,
    );
}

/// Plain-mesh cull: LOD pick, world AABB, frustum + occlusion tests, record
/// build. Resolves the fresh world matrix first so LOD picking reads the
/// current-frame AABB (the parallel path pre-warms the same cache — see
/// buildFrameQueuesParallel). Stats go to the caller-supplied counter block
/// so parallel chunks can merge locally.
///
/// Internal to `render_queue/*` (used by `build`'s serial loop and parallel
/// chunks); not re-exported by the facade.
pub fn cullNonInstancedMesh(
    ctx: FrameCullContext,
    frustum: Frustum,
    eye: Vec3,
    mesh: *Mesh,
    stats: *SceneStats,
    mesh_index: usize,
) ?CulledMesh {
    stats.total_meshes += 1;
    if (!mesh.is_visible) return null;

    // Fresh world matrix first: it refreshes mesh.cached_aabb for this
    // frame, so the LOD distance below never reads a stale AABB tagged with
    // a previous cache key (the parallel path pre-warms this cache; the
    // serial path must pick the same LOD at distance thresholds).
    const model = worldMatrixCached(ctx.cache_key, mesh);
    var render_mesh = mesh;
    if (mesh.lod_levels.items.len > 0) {
        const center = if (mesh.cached_aabb.isValid()) mesh.cached_aabb.center() else mesh.position;
        const dist_sq = center.distanceSq(eye);
        const active_lod = mesh.getLODSq(dist_sq);
        if (active_lod) |lod| {
            render_mesh = lod;
        } else {
            // Beyond max distance, culled
            stats.culled_meshes += 1;
            return null;
        }
    }

    // The active LOD child may itself still be awaiting GPU buffers
    // (deferred creation): never queue it for drawing with dead handles.
    if (render_mesh.gpu_pending) return null;

    const world_aabb = if (render_mesh != mesh and render_mesh.local_bounding_box.isValid())
        render_mesh.local_bounding_box.transform(model)
    else
        mesh.cached_aabb;

    if (ctx.cull_frustum and render_mesh.culling_strategy != .always_render) {
        if (!frustum.intersectsAABB(world_aabb)) {
            stats.culled_meshes += 1;
            return null;
        }
    }

    if (ctx.cull_occlusion and !render_mesh.is_occluder and render_mesh.culling_strategy != .always_render) {
        if (ctx.occlusion_culler.isOccluded(world_aabb)) {
            stats.occluded_meshes += 1;
            stats.culled_meshes += 1;
            return null;
        }
    }

    stats.rendered_meshes += 1;

    const mat = render_mesh.material orelse mesh.material;
    const is_pbr = if (mat) |m| (m == .pbr) else false;
    const tex_id: u32 = if (mat) |m|
        if (m.primaryTexture()) |t| t.view.id else ctx.default_white_id
    else
        ctx.default_white_id;

    const draw_rec = buildMaterialRecord(ctx, mat);

    const skin_mat = if (render_mesh.skeleton) |skel| skel.getRenderSkinMatrices() else null;
    const morph_u = if (render_mesh.morph_mode == .gpu)
        morph_gpu.vsUniforms(render_mesh)
    else
        morph_gpu.VsUniforms{
            .weights0 = .{ 0, 0, 0, 0 },
            .weights1 = .{ 0, 0, 0, 0 },
            .params = .{ 0, 1, 1, 0 },
        };
    const morph_v = if (render_mesh.morph_mode == .gpu)
        render_mesh.morph_delta_view
    else
        ctx.default_morph_view;

    const d_sq = world_aabb.center().sub(eye).lengthSq();
    const is_decal = render_mesh.is_decal or mesh.is_decal;
    const transparent = materialIsTransparent(mat) or is_decal;
    // Снимок hook-материала строится здесь же (prepare-фаза, живые данные
    // ещё доступны); в очередь попадёт только после копии в appendRenderItem.
    const dummy_white = Texture{ .image = .{}, .view = .{ .id = ctx.default_white_id }, .sampler = .{}, .width = 1, .height = 1 };
    const shader_snap = material_mod.buildShaderSnapshot(mat, ctx.default_white orelse &dummy_white);
    // Снимок coat-факторов (null для всех draws без включённого lobe —
    // такие draws рисуют из CoatParams.neutral без слота в хранилище).
    const coat = material_mod.coatParamsFor(mat);
    return .{
        .item = .{
            .draw_record = draw_rec,
            .model = model,
            .distance_sq = d_sq,
            .is_pbr = is_pbr,
            .texture_id = tex_id,
            .mesh_index = @intCast(mesh_index),
            .transparent = transparent,
            .double_sided = materialIsDoubleSided(mat) or is_decal,
            .is_decal = is_decal,
            .receive_shadows = render_mesh.receive_shadows,
            .morph_uniforms = morph_u,
            .morph_view = morph_v,
            .vertex_buffer = render_mesh.vertex_buffer,
            .index_buffer = render_mesh.index_buffer,
            .index_count = render_mesh.index_count,
            .index_type = render_mesh.index_type,
            .is_u32 = render_mesh.index_type == .UINT32,
            .is_skinned = render_mesh.skeleton != null,
        },
        .skin_src = skin_mat,
        .shader_snap = shader_snap,
        .coat = coat,
    };
}

test "worldMatrixCached caches per frame and resolves parents" {
    var parent: Mesh = undefined;
    var child: Mesh = undefined;

    // TRS inputs must be fully defined: undefined garbage would poison the
    // composed matrix (the fields have no defaults on a raw Mesh).
    parent.position = Vec3.new(1.0, 0.0, 0.0);
    parent.rotation = Vec3.zero;
    parent.scaling = Vec3.new(1.0, 1.0, 1.0);
    parent.base_matrix = Mat4.identity;
    parent.local_bounding_box = BoundingBox.init(Vec3.zero, Vec3.new(1.0, 1.0, 1.0));
    parent.parent = null;
    parent.attach_bone = null;
    parent.cached_frame = 0;
    child.position = Vec3.new(0.0, 1.0, 0.0);
    child.rotation = Vec3.zero;
    child.scaling = Vec3.new(1.0, 1.0, 1.0);
    child.base_matrix = Mat4.identity;
    child.local_bounding_box = BoundingBox.init(Vec3.zero, Vec3.new(1.0, 1.0, 1.0));
    child.parent = &parent;
    child.attach_bone = null;
    child.cached_frame = 0;

    const world = worldMatrixCached(7, &child);
    // Child = parent TRS composed with local TRS: translation (1, 1, 0).
    try std.testing.expectEqual(@as(f32, 1.0), world.m[12]);
    try std.testing.expectEqual(@as(f32, 1.0), world.m[13]);
    try std.testing.expectEqual(@as(f32, 0.0), world.m[14]);
    // Both nodes are tagged with the frame id now.
    try std.testing.expectEqual(@as(u64, 7), child.cached_frame);
    try std.testing.expectEqual(@as(u64, 7), parent.cached_frame);

    // Same frame: cache hit returns the stored matrix without recomputing.
    const cached = worldMatrixCached(7, &child);
    try std.testing.expectEqual(world, cached);

    // worldAABBCached derives the world-space AABB from the same cache.
    const aabb = worldAABBCached(7, &child);
    try std.testing.expect(aabb.isValid());
}

test "worldMatrixCached honors bone attachment like getWorldMatrix" {
    const ally = std.testing.allocator;
    const Skeleton = @import("../../animation/skeleton.zig").Skeleton;

    // Host: identity TRS with a one-bone skeleton whose bone sits at (2,0,0).
    var host: Mesh = undefined;
    host.position = Vec3.zero;
    host.rotation = Vec3.zero;
    host.scaling = Vec3.new(1.0, 1.0, 1.0);
    host.base_matrix = Mat4.identity;
    host.local_bounding_box = BoundingBox.init(Vec3.zero, Vec3.new(1.0, 1.0, 1.0));
    host.parent = null;
    host.cached_frame = 0;
    host.attach_bone = null;
    host.skeleton = null;

    const skel = try Skeleton.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].model_matrix = Mat4.fromRotationTranslationScale(Vec3.zero, Vec3.zero, Vec3.new(1.0, 1.0, 1.0));
    skel.bones[0].model_matrix.m[12] = 2.0; // bone at x=2
    host.skeleton = skel;

    // Attached mesh: offset (0,1,0) on the bone, local TRS at (0,0,3).
    var attached: Mesh = undefined;
    attached.position = Vec3.new(0.0, 0.0, 3.0);
    attached.rotation = Vec3.zero;
    attached.scaling = Vec3.new(1.0, 1.0, 1.0);
    attached.base_matrix = Mat4.identity;
    attached.local_bounding_box = BoundingBox.init(Vec3.zero, Vec3.new(1.0, 1.0, 1.0));
    attached.parent = null;
    attached.cached_frame = 0;
    attached.skeleton = null;
    attached.attach_bone = .{ .host_mesh = &host, .bone_index = 0, .offset_matrix = Mat4.fromRotationTranslationScale(Vec3.zero, Vec3.zero, Vec3.new(1.0, 1.0, 1.0)) };
    attached.attach_bone.?.offset_matrix.m[13] = 1.0;

    const cached = worldMatrixCached(5, &attached);
    const direct = attached.getWorldMatrix();
    // The cache must agree with the uncached bone-aware path everywhere:
    // bone (x=2) + offset (y=1) + local (z=3).
    try std.testing.expectEqual(direct, cached);
    try std.testing.expectEqual(@as(f32, 2.0), cached.m[12]);
    try std.testing.expectEqual(@as(f32, 1.0), cached.m[13]);
    try std.testing.expectEqual(@as(f32, 3.0), cached.m[14]);
    try std.testing.expectEqual(@as(u64, 5), attached.cached_frame);

    // Dead-skeleton host: falls through to the plain parent-less matrix.
    // Per getWorldMatrix semantics the offset applies only on the bone
    // path, so the fallback is the bare local TRS (z=3).
    host.skeleton = null;
    const fallback = worldMatrixCached(6, &attached);
    try std.testing.expectEqual(@as(f32, 0.0), fallback.m[12]);
    try std.testing.expectEqual(@as(f32, 0.0), fallback.m[13]);
    try std.testing.expectEqual(@as(f32, 3.0), fallback.m[14]);
}
