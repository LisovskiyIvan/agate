const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Mat4 = math.Mat4;
const Frustum = math.Frustum;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../mesh.zig").Mesh;
const material_mod = @import("../material.zig");
const Material = material_mod.Material;
const MaterialDrawRecord = material_mod.MaterialDrawRecord;
const StandardMaterial = material_mod.StandardMaterial;
const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const skeleton_mod = @import("../animation/skeleton.zig");
const MAX_BONES = skeleton_mod.MAX_BONES;
const morph_gpu = @import("../mesh/morph_gpu.zig");
const visibility = @import("../visibility/mod.zig");
const jobs = @import("../jobs.zig");
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

pub const RenderMeshItem = struct {
    mesh: *Mesh,
    model: Mat4,
    distance_sq: f32,
    is_pbr: bool,
    texture_id: u32,
    material: ?Material = null,
    /// Compact, GPU-ready material snapshot (~120-240 B): factors, uv transforms,
    /// cutoff, and texture views/samplers. Render passes consume this directly.
    draw_record: MaterialDrawRecord = .{},
    // True when the mesh material uses .blend alpha mode. Set at queue
    // build time; defaults to false so opaque behavior is unchanged.
    transparent: bool = false,
    // True when the mesh material is double-sided (face culling disabled).
    // Set at queue build time from materialIsDoubleSided; when the flag is
    // stale/missing, pipeline selection re-derives it from mesh.material.
    // Defaults to false so single-sided behavior is unchanged.
    double_sided: bool = false,
    // True when the mesh is a projected decal. Decals route to the transparent
    // queue so they render after all opaque geometry with depth writes disabled,
    // eliminating depth-buffer z-fighting against the underlying surface.
    is_decal: bool = false,
    receive_shadows: bool = true,
    /// Pointer to the published double-buffered skin matrices slot for this mesh's skeleton
    skin_matrices: ?*const [MAX_BONES]Mat4 = null,
    /// Per-mesh morph uniforms (weights and delta dimensions)
    morph_uniforms: morph_gpu.VsUniforms = .{
        .weights0 = .{ 0, 0, 0, 0 },
        .weights1 = .{ 0, 0, 0, 0 },
        .params = .{ 0, 1, 1, 0 },
    },
    /// Delta strip texture view for GPU-mode morphs
    morph_view: sg.View = .{},
};

/// The four draw queues plus the per-frame instance-matrix staging buffer.
/// Cleared and refilled by buildFrameQueues each render(); ownership stays
/// with Scene via a single field.
pub const RenderQueues = struct {
    items: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    // Transparent meshes (material alpha_mode == .blend), sorted strictly
    // back-to-front and drawn after every opaque mesh and instanced mesh.
    transparent: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    opaque_instanced: std.ArrayListUnmanaged(*Mesh) = .empty,
    transparent_instanced: std.ArrayListUnmanaged(*Mesh) = .empty,
    instance_matrices: std.ArrayListUnmanaged(Mat4) = .empty,

    pub fn reset(self: *RenderQueues) void {
        self.items.clearRetainingCapacity();
        self.transparent.clearRetainingCapacity();
        self.opaque_instanced.clearRetainingCapacity();
        self.transparent_instanced.clearRetainingCapacity();
        self.instance_matrices.clearRetainingCapacity();
    }

    pub fn deinit(self: *RenderQueues, allocator: std.mem.Allocator) void {
        self.items.deinit(allocator);
        self.transparent.deinit(allocator);
        self.opaque_instanced.deinit(allocator);
        self.transparent_instanced.deinit(allocator);
        self.instance_matrices.deinit(allocator);
    }
};

pub fn sortRenderItems(_: void, a: RenderMeshItem, b: RenderMeshItem) bool {
    if (a.is_pbr != b.is_pbr) {
        return !a.is_pbr;
    }
    if (a.texture_id != b.texture_id) {
        return a.texture_id < b.texture_id;
    }
    return a.distance_sq < b.distance_sq;
}

// Transparent items sort strictly back-to-front (by squared camera
// distance) so alpha blending composites in the correct order.
// When two coplanar surfaces are at the same distance, decals sort after
// (drawn on top of) non-decals.
pub fn sortTransparentBackToFront(_: void, a: RenderMeshItem, b: RenderMeshItem) bool {
    if (@abs(a.distance_sq - b.distance_sq) < 1e-4) {
        if (a.is_decal != b.is_decal) {
            return !a.is_decal;
        }
    }
    return a.distance_sq > b.distance_sq;
}

// A mesh is transparent when its material opts into .blend alpha mode.
// Meshes without a material render opaque (legacy behavior).
// Cutout (.cutout) is NOT transparent: it renders in the opaque queue
// with depth writes on, discarding only sub-cutoff fragments in-shader.
pub fn materialIsTransparent(mat: ?Material) bool {
    if (mat) |m| return m.isTransparent();
    return false;
}

// A mesh is cutout when its material uses .cutout alpha mode.
// Cutout meshes stay in the opaque queue (see materialIsTransparent).
// Shadow depth passes ignore the alpha test and render full quads
// (documented limitation: cutout holes still cast solid shadows).
pub fn materialIsCutout(mat: ?Material) bool {
    if (mat) |m| return m.isCutout();
    return false;
}

// A mesh is double-sided when its material sets double_sided.
// Such meshes select the cull-off pipeline twins (opaque or blend).
// Meshes without a material render single-sided (legacy behavior).
pub fn materialIsDoubleSided(mat: ?Material) bool {
    if (mat) |m| return m.isDoubleSided();
    return false;
}

// Derives a transparent twin pipeline desc from an opaque base desc:
// same shader/layout/depth test, but SRC_ALPHA/ONE_MINUS_SRC_ALPHA
// blending with depth write disabled. Pure function (no GPU calls).
pub fn blendDescFor(base: sg.PipelineDesc) sg.PipelineDesc {
    var desc = base;
    desc.depth.write_enabled = false;
    desc.colors[0].blend = .{
        .enabled = true,
        .src_factor_rgb = .SRC_ALPHA,
        .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
        .src_factor_alpha = .ONE,
        .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
    };
    return desc;
}

/// World matrix computed at most once per render() call (tagged with the
/// frame id on the mesh). Semantics mirror Mesh.getWorldMatrix exactly —
/// including bone attachment, which replaces the parent chain when the
/// host skeleton is live — so every pass sees the same transform. Parent
/// and host chains resolve through the same cache, so hierarchies stay
/// O(depth) total.
pub fn worldMatrixCached(frame_id: u64, mesh: *Mesh) Mat4 {
    if (mesh.cached_frame == frame_id) return mesh.cached_matrix;
    const trs = Mat4.fromRotationTranslationScale(mesh.position, mesh.rotation, mesh.scaling);
    const local = Mat4.mul(trs, mesh.base_matrix);
    var world: Mat4 = undefined;
    if (mesh.attach_bone) |att| {
        if (att.host_mesh.skeleton) |skel| {
            const host_mat = worldMatrixCached(frame_id, att.host_mesh);
            const bone_mat = skel.getBoneWorldMatrix(att.bone_index, host_mat);
            world = Mat4.mul(Mat4.mul(bone_mat, att.offset_matrix), local);
        } else if (mesh.parent) |p| {
            world = Mat4.mul(worldMatrixCached(frame_id, p), local);
        } else {
            world = local;
        }
    } else if (mesh.parent) |p| {
        world = Mat4.mul(worldMatrixCached(frame_id, p), local);
    } else {
        world = local;
    }
    mesh.cached_matrix = world;
    mesh.cached_aabb = mesh.local_bounding_box.transform(world);
    mesh.cached_frame = frame_id;
    return world;
}

pub fn worldAABBCached(frame_id: u64, mesh: *Mesh) BoundingBox {
    if (mesh.cached_frame == frame_id) return mesh.cached_aabb;
    _ = worldMatrixCached(frame_id, mesh);
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
    parallel_min_meshes: usize = 1024,

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
    frame_id: u64,
    view_proj: Mat4,
    eye: Vec3,
    cull_frustum: bool,
    cull_occlusion: bool,
    culling_mask: u32 = 0xFFFFFFFF,
    occlusion_culler: *visibility.OcclusionCuller,
    stats: *SceneStats,
    queues: *RenderQueues,
};

/// Phase -1 (occluder rasterization) and Phase 0 (frustum/occlusion
/// culling, LOD picking, instance-buffer management, queue fill) of the
/// render frame. Results land in ctx.queues and the stats counters.
///
/// Instance-bearing meshes stay serial in every path: they interleave with
/// sg buffer creation/update, which is single-context. Plain meshes cull
/// data-parallel when a pool is attached and the scene clears
/// `parallel_min_meshes`; the parallel pass produces the same records in
/// the same order as the serial loop (chunks partition the mesh list in
/// order and merge back in chunk order), so the choice between them is a
/// scheduling detail, never a behavior change.
pub fn buildFrameQueues(ctx: FrameCullContext) void {
    const frustum = Frustum.fromViewProjection(ctx.view_proj);
    const eye = ctx.eye;

    // Phase -1: Occlusion Culling setup & occluder rasterization (serial:
    // the Hi-Z rasterizer is stateful).
    if (ctx.cull_occlusion) {
        ctx.occlusion_culler.beginFrame(ctx.view_proj);
        for (ctx.meshes) |m| {
            if (!m.is_lod_child and m.is_visible and m.is_occluder and ((m.layer_mask & ctx.culling_mask) != 0)) {
                const m_world = worldMatrixCached(ctx.frame_id, m);
                ctx.occlusion_culler.rasterizeOccluderMesh(
                    m.cpu_positions,
                    m.cpu_indices,
                    m.local_bounding_box,
                    m_world,
                );
            }
        }
        ctx.occlusion_culler.endOccluders();
        ctx.stats.occluders_count = ctx.occlusion_culler.occluder_count;
        ctx.stats.occluder_triangles = ctx.occlusion_culler.triangles_rasterized;
    }

    // Phase 0: cull + queue fill.
    const pool = ctx.thread_pool;
    if (pool != null and pool.?.workerCount() > 0 and ctx.meshes.len >= ctx.parallel_min_meshes) {
        buildFrameQueuesParallel(ctx, frustum, eye, pool.?) catch {};
        return;
    }

    for (ctx.meshes) |mesh| {
        if (mesh.is_lod_child) continue;
        if ((mesh.layer_mask & ctx.culling_mask) == 0) continue;
        if (mesh.instances.items.len > 0) {
            submitInstancedMesh(ctx, frustum, mesh);
            continue;
        }
        if (cullNonInstancedMesh(ctx, frustum, eye, mesh, ctx.stats)) |item| {
            appendRenderItem(ctx, item);
        }
    }
}

/// Routes a finished record into the draw queue it belongs to.
fn appendRenderItem(ctx: FrameCullContext, item: RenderMeshItem) void {
    if (item.transparent) {
        ctx.queues.transparent.append(ctx.allocator, item) catch {};
    } else {
        ctx.queues.items.append(ctx.allocator, item) catch {};
    }
}

/// Instance-bearing mesh handling, verbatim from the legacy loop: per-frame
/// instance-matrix staging, sg buffer (re)creation/upload, frustum test on
/// the combined AABB, instanced queue fill. Serial-only by design.
fn submitInstancedMesh(ctx: FrameCullContext, frustum: Frustum, mesh: *Mesh) void {
    if (mesh.instance_uploaded_frame != ctx.frame_id) {
        mesh.instance_uploaded_frame = ctx.frame_id;
        ctx.queues.instance_matrices.clearRetainingCapacity();
        var combined_aabb = math.BoundingBox.zero;
        for (mesh.instances.items) |inst| {
            if (!inst.is_visible) continue;
            inst.updateCachedTransforms();
            ctx.queues.instance_matrices.append(ctx.allocator, inst.cached_world_matrix) catch return;
            if (combined_aabb.isValid()) {
                combined_aabb = combined_aabb.merge(inst.cached_bounding_box);
            } else {
                combined_aabb = inst.cached_bounding_box;
            }
        }
        mesh.cached_aabb = combined_aabb;
        const active_count = ctx.queues.instance_matrices.items.len;
        mesh.visible_instance_count = @intCast(active_count);
        if (active_count > 0) {
            if (mesh.instance_buffer.id == 0 or mesh.instance_buffer_capacity < active_count) {
                if (mesh.instance_buffer.id != 0) {
                    sg.destroyBuffer(mesh.instance_buffer);
                }
                const new_cap = @max(active_count, mesh.instance_buffer_capacity * 2);
                mesh.instance_buffer = sg.makeBuffer(.{
                    .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                    .size = new_cap * @sizeOf(Mat4),
                });
                mesh.instance_buffer_capacity = new_cap;
                sg.updateBuffer(mesh.instance_buffer, sg.asRange(ctx.queues.instance_matrices.items[0..active_count]));
                mesh.instance_hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ctx.queues.instance_matrices.items[0..active_count]));
                mesh.instance_uploaded_count = active_count;
            } else {
                const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ctx.queues.instance_matrices.items[0..active_count]));
                if (active_count != mesh.instance_uploaded_count or h != mesh.instance_hash) {
                    sg.updateBuffer(mesh.instance_buffer, sg.asRange(ctx.queues.instance_matrices.items[0..active_count]));
                    mesh.instance_hash = h;
                    mesh.instance_uploaded_count = active_count;
                }
            }
        }
    }

    if (mesh.visible_instance_count > 0 and ((mesh.layer_mask & ctx.culling_mask) != 0)) {
        if (ctx.cull_frustum and mesh.cached_aabb.isValid() and !frustum.intersectsAABB(mesh.cached_aabb)) {
            ctx.stats.culled_meshes += @intCast(mesh.instances.items.len);
            return;
        }
        ctx.stats.total_meshes += @intCast(mesh.instances.items.len);
        ctx.stats.rendered_meshes += mesh.visible_instance_count;
        if (materialIsTransparent(mesh.material)) {
            ctx.queues.transparent_instanced.append(ctx.allocator, mesh) catch {};
        } else {
            ctx.queues.opaque_instanced.append(ctx.allocator, mesh) catch {};
        }
    }
}

/// Plain-mesh cull: LOD pick, world AABB, frustum + occlusion tests, record
/// build. Pure with respect to the mesh (world matrix must already be warm
/// in the parallel path — see buildFrameQueuesParallel). Stats go to the
/// caller-supplied counter block so parallel chunks can merge locally.
fn cullNonInstancedMesh(
    ctx: FrameCullContext,
    frustum: Frustum,
    eye: Vec3,
    mesh: *Mesh,
    stats: *SceneStats,
) ?RenderMeshItem {
    stats.total_meshes += 1;
    if (!mesh.is_visible) return null;

    var render_mesh = mesh;
    if (mesh.lod_levels.items.len > 0) {
        const dist = if (mesh.cached_aabb.isValid()) mesh.cached_aabb.center().distance(eye) else mesh.position.distance(eye);
        const active_lod = mesh.getLOD(dist);
        if (active_lod) |lod| {
            render_mesh = lod;
        } else {
            // Beyond max distance, culled
            stats.culled_meshes += 1;
            return null;
        }
    }

    const model = worldMatrixCached(ctx.frame_id, mesh);
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

    const dummy_tex = Texture{ .image = .{}, .view = .{ .id = ctx.default_white_id }, .sampler = .{}, .width = 1, .height = 1 };
    const dummy_cube = CubeTexture{ .image = .{}, .view = .{}, .sampler = .{}, .size = 1 };
    const dummy_std = StandardMaterial.init("default");

    const white_tex = ctx.default_white orelse &dummy_tex;
    const norm_tex = ctx.default_normal orelse &dummy_tex;
    const cube_tex = ctx.default_cube orelse &dummy_cube;
    const def_mat = ctx.default_material orelse &dummy_std;

    const draw_rec = material_mod.buildDrawRecord(
        mat,
        def_mat,
        white_tex,
        norm_tex,
        cube_tex,
        ctx.sky_texture,
        ctx.ibl_intensity,
    );

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
    return .{
        .mesh = render_mesh,
        .material = mat,
        .draw_record = draw_rec,
        .model = model,
        .distance_sq = d_sq,
        .is_pbr = is_pbr,
        .texture_id = tex_id,
        .transparent = transparent,
        .double_sided = materialIsDoubleSided(mat) or is_decal,
        .is_decal = is_decal,
        .receive_shadows = render_mesh.receive_shadows,
        .skin_matrices = skin_mat,
        .morph_uniforms = morph_u,
        .morph_view = morph_v,
    };
}

/// Parallel cull pass state. Chunks partition `ctx.meshes` into fixed
/// ranges; each chunk collects its records and stats independently (no
/// locks) and the caller merges them in chunk order, reproducing the serial
/// loop's queue order and stat totals exactly.
const ParallelCull = struct {
    ctx: FrameCullContext,
    frustum: Frustum,
    eye: Vec3,
    /// Meshes per chunk (ceil split; the last chunk may be short).
    span: usize,
    records: []std.ArrayListUnmanaged(RenderMeshItem),
    chunk_stats: []SceneStats,

    fn cullChunkRange(pass: *ParallelCull, start: usize, end: usize) void {
        for (start..end) |chunk_id| {
            const lo = chunk_id * pass.span;
            const hi = @min(lo + pass.span, pass.ctx.meshes.len);
            var local = SceneStats{};
            for (pass.ctx.meshes[lo..hi]) |mesh| {
                if (mesh.is_lod_child) continue;
                if ((mesh.layer_mask & pass.ctx.culling_mask) == 0) continue;
                if (cullNonInstancedMesh(pass.ctx, pass.frustum, pass.eye, mesh, &local)) |item| {
                    pass.records[chunk_id].append(pass.ctx.allocator, item) catch {};
                }
            }
            pass.chunk_stats[chunk_id] = local;
        }
    }
};

fn buildFrameQueuesParallel(
    ctx: FrameCullContext,
    frustum: Frustum,
    eye: Vec3,
    pool: *jobs.Pool,
) !void {
    // World matrices cache per mesh with parent-chain recursion; lazy fill
    // from two workers could race on a shared parent's cache. Warm the
    // cache in mesh order first — pure TRS work, O(meshes), and a no-op
    // for meshes already tagged with this frame id (e.g. second camera of
    // a PIP render).
    for (ctx.meshes) |m| _ = worldMatrixCached(ctx.frame_id, m);

    const chunk_count = (pool.workerCount() + 1) * 4;
    const span = (ctx.meshes.len + chunk_count - 1) / chunk_count;

    const records = try ctx.allocator.alloc(std.ArrayListUnmanaged(RenderMeshItem), chunk_count);
    defer ctx.allocator.free(records);
    for (records) |*r| r.* = .empty;
    const chunk_stats = try ctx.allocator.alloc(SceneStats, chunk_count);
    defer ctx.allocator.free(chunk_stats);
    @memset(chunk_stats, .{});

    var pass = ParallelCull{
        .ctx = ctx,
        .frustum = frustum,
        .eye = eye,
        .span = span,
        .records = records,
        .chunk_stats = chunk_stats,
    };
    pool.forkJoin(ParallelCull, &pass, ParallelCull.cullChunkRange, chunk_count);

    // Deterministic merge: chunk order == mesh order, so the queues land
    // exactly as the serial loop would have filled them.
    for (records, chunk_stats) |*list, local| {
        ctx.stats.total_meshes += local.total_meshes;
        ctx.stats.rendered_meshes += local.rendered_meshes;
        ctx.stats.culled_meshes += local.culled_meshes;
        ctx.stats.occluded_meshes += local.occluded_meshes;
        for (list.items) |item| appendRenderItem(ctx, item);
        list.deinit(ctx.allocator);
    }

    // Instance-bearing meshes: serial submission (sg buffer management).
    for (ctx.meshes) |mesh| {
        if (mesh.is_lod_child or mesh.instances.items.len == 0) continue;
        if ((mesh.layer_mask & ctx.culling_mask) == 0) continue;
        submitInstancedMesh(ctx, frustum, mesh);
    }
}

test "cutout stays opaque, blend stays transparent" {
    const material = @import("../material.zig");

    try std.testing.expect(!materialIsTransparent(null));
    try std.testing.expect(!materialIsCutout(null));
    try std.testing.expect(!materialIsDoubleSided(null));

    var std_mat = material.StandardMaterial.init("m");
    var pbr_mat = material.PBRMaterial.init("p");

    // Opaque default: opaque queue, not cutout, single-sided.
    try std.testing.expect(!materialIsTransparent(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(!materialIsCutout(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsCutout(.{ .pbr = &pbr_mat }));

    // Cutout: still NOT in the transparent queue, but flagged cutout.
    std_mat.alpha_mode = .cutout;
    pbr_mat.alpha_mode = .cutout;
    try std.testing.expect(!materialIsTransparent(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(materialIsCutout(.{ .standard = &std_mat }));
    try std.testing.expect(materialIsCutout(.{ .pbr = &pbr_mat }));

    // Blend: transparent queue, never cutout.
    std_mat.alpha_mode = .blend;
    pbr_mat.alpha_mode = .blend;
    try std.testing.expect(materialIsTransparent(.{ .standard = &std_mat }));
    try std.testing.expect(materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(!materialIsCutout(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsCutout(.{ .pbr = &pbr_mat }));

    // Double-sided is orthogonal to the alpha mode.
    try std.testing.expect(!materialIsDoubleSided(.{ .standard = &std_mat }));
    std_mat.double_sided = true;
    pbr_mat.double_sided = true;
    try std.testing.expect(materialIsDoubleSided(.{ .standard = &std_mat }));
    try std.testing.expect(materialIsDoubleSided(.{ .pbr = &pbr_mat }));
}

// ---- Ported from the legacy Scene inline suite. ----

test "transparent classification follows material alpha mode" {
    const material = @import("../material.zig");

    try std.testing.expect(!materialIsTransparent(null));
    var std_mat = material.StandardMaterial.init("s");
    var pbr_mat = material.PBRMaterial.init("p");
    try std.testing.expect(!materialIsTransparent(Material{ .standard = &std_mat }));
    try std.testing.expect(!materialIsTransparent(Material{ .pbr = &pbr_mat }));
    std_mat.alpha_mode = .blend;
    pbr_mat.alpha_mode = .blend;
    try std.testing.expect(materialIsTransparent(Material{ .standard = &std_mat }));
    try std.testing.expect(materialIsTransparent(Material{ .pbr = &pbr_mat }));
}

test "transparent queue sorts strictly back-to-front" {
    var m: Mesh = undefined;
    var items = [_]RenderMeshItem{
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = false, .texture_id = 0, .transparent = true },
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 9.0, .is_pbr = false, .texture_id = 0, .transparent = true },
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 4.0, .is_pbr = true, .texture_id = 1, .transparent = true },
    };
    std.mem.sort(RenderMeshItem, &items, {}, sortTransparentBackToFront);
    try std.testing.expect(items[0].distance_sq == 9.0);
    try std.testing.expect(items[1].distance_sq == 4.0);
    try std.testing.expect(items[2].distance_sq == 1.0);
}

test "opaque sort unchanged: state groups, front-to-back" {
    var m: Mesh = undefined;
    var items = [_]RenderMeshItem{
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 9.0, .is_pbr = false, .texture_id = 2 },
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = false, .texture_id = 2 },
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 5.0, .is_pbr = true, .texture_id = 1 },
        .{ .mesh = &m, .model = Mat4.identity, .distance_sq = 3.0, .is_pbr = false, .texture_id = 1 },
    };
    // Default transparent flag is false, so legacy items sort as before.
    try std.testing.expect(!items[0].transparent);
    std.mem.sort(RenderMeshItem, &items, {}, sortRenderItems);
    // Standard before PBR, texture id ascending, front-to-back within a group.
    try std.testing.expect(!items[0].is_pbr and items[0].texture_id == 1 and items[0].distance_sq == 3.0);
    try std.testing.expect(!items[1].is_pbr and items[1].texture_id == 2 and items[1].distance_sq == 1.0);
    try std.testing.expect(!items[2].is_pbr and items[2].texture_id == 2 and items[2].distance_sq == 9.0);
    try std.testing.expect(items[3].is_pbr and items[3].distance_sq == 5.0);
}

test "blendDescFor enables alpha blending without depth write" {
    const base = sg.PipelineDesc{
        .shader = .{},
        .index_type = .UINT32,
        .depth = .{ .compare = .LESS_EQUAL, .write_enabled = true },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    const blended = blendDescFor(base);
    try std.testing.expect(blended.colors[0].blend.enabled);
    try std.testing.expect(blended.colors[0].blend.src_factor_rgb == .SRC_ALPHA);
    try std.testing.expect(blended.colors[0].blend.dst_factor_rgb == .ONE_MINUS_SRC_ALPHA);
    try std.testing.expect(!blended.depth.write_enabled);
    try std.testing.expect(blended.depth.compare == .LESS_EQUAL);
    try std.testing.expect(blended.index_type == .UINT32);
    try std.testing.expect(blended.cull_mode == .BACK);
    // Pure function: the base desc is left untouched.
    try std.testing.expect(!base.colors[0].blend.enabled);
    try std.testing.expect(base.depth.write_enabled);
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

test "shared LOD mesh preserves entity transforms without mutation" {
    const ally = std.testing.allocator;

    var shared_lod: Mesh = undefined;
    shared_lod.position = Vec3.zero;
    shared_lod.rotation = Vec3.zero;
    shared_lod.scaling = Vec3.new(1.0, 1.0, 1.0);
    shared_lod.base_matrix = Mat4.identity;
    shared_lod.local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));
    shared_lod.cached_aabb = shared_lod.local_bounding_box;
    shared_lod.cached_matrix = Mat4.identity;
    shared_lod.cached_frame = 0;
    shared_lod.parent = null;
    shared_lod.material = null;
    shared_lod.is_visible = true;
    shared_lod.is_decal = false;
    shared_lod.is_occluder = false;
    shared_lod.culling_strategy = .always_render;
    shared_lod.lod_levels = .empty;
    shared_lod.instances = .empty;
    shared_lod.skeleton = null;
    shared_lod.attach_bone = null;
    shared_lod.index_type = .UINT16;

    var mesh1: Mesh = shared_lod;
    mesh1.position = Vec3.new(10.0, 0.0, 0.0);
    mesh1.cached_frame = 0;
    try mesh1.lod_levels.append(ally, .{ .distance = 0.0, .mesh = &shared_lod });
    defer mesh1.lod_levels.deinit(ally);

    var mesh2: Mesh = shared_lod;
    mesh2.position = Vec3.new(20.0, 0.0, 0.0);
    mesh2.cached_frame = 0;
    try mesh2.lod_levels.append(ally, .{ .distance = 0.0, .mesh = &shared_lod });
    defer mesh2.lod_levels.deinit(ally);

    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();

    const meshes = [_]*Mesh{ &mesh1, &mesh2 };
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 42,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    // The shared child LOD mesh MUST NOT be mutated!
    try std.testing.expectEqual(@as(f32, 0.0), shared_lod.position.x);
    try std.testing.expectEqual(@as(usize, 2), queues.items.items.len);

    // Both items render with the shared LOD mesh geometry
    try std.testing.expectEqual(&shared_lod, queues.items.items[0].mesh);
    try std.testing.expectEqual(&shared_lod, queues.items.items[1].mesh);

    // But each entity keeps its own distinct world matrix!
    const m0_x = queues.items.items[0].model.m[12];
    const m1_x = queues.items.items[1].model.m[12];
    try std.testing.expect((m0_x == 10.0 and m1_x == 20.0) or (m0_x == 20.0 and m1_x == 10.0));
}

test "culling_mask filters out meshes with disjoint layer_mask" {
    const ally = std.testing.allocator;

    var mesh1 = Mesh{
        .name = "layer1",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .layer_mask = 0b01,
    };
    var mesh2 = Mesh{
        .name = "layer2",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .layer_mask = 0b10,
    };

    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();

    const meshes = [_]*Mesh{ &mesh1, &mesh2 };

    // Cull with mask 0b01: only mesh1 should be queued
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .culling_mask = 0b01,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(&mesh1, queues.items.items[0].mesh);

    // Cull with mask 0b10: only mesh2 should be queued
    queues.reset();
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .culling_mask = 0b10,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(&mesh2, queues.items.items[0].mesh);
}

test "parallel cull produces serial-identical queues" {
    const ally = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    // Deterministic synthetic scene: mixed layer masks, visibility, and
    // placement so both culling decisions and sort keys vary.
    const count = 3000;
    const meshes = try ally.alloc(Mesh, count);
    defer ally.free(meshes);
    const ptrs = try ally.alloc(*Mesh, count);
    defer ally.free(ptrs);
    for (0..count) |i| {
        meshes[i] = .{
            .name = "m",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 3,
            .layer_mask = if (i % 7 == 0) 0b10 else 0b01,
        };
        if (i % 5 == 0) {
            // Near the origin: survives the identity-projection frustum.
            meshes[i].position = Vec3.new(rand.float(f32) * 0.2, rand.float(f32) * 0.2, rand.float(f32) * 0.2);
        } else {
            meshes[i].position = Vec3.new(rand.float(f32) * 100 - 50, rand.float(f32) * 100 - 50, rand.float(f32) * 100 - 50);
        }
        meshes[i].is_visible = i % 11 != 0;
        ptrs[i] = &meshes[i];
    }

    var culler = visibility.OcclusionCuller.init();
    var stats_a = SceneStats{};
    var queues_a = RenderQueues{};
    defer queues_a.deinit(ally);
    var stats_b = SceneStats{};
    var queues_b = RenderQueues{};
    defer queues_b.deinit(ally);

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    // Serial pass (no pool attached).
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .frame_id = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = true,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_a,
        .queues = &queues_a,
        .default_white_id = 1,
    });

    // Parallel pass (forced past the mesh threshold, 2 workers).
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .frame_id = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = true,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_b,
        .queues = &queues_b,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });

    // Identical stats…
    try std.testing.expectEqual(stats_a.total_meshes, stats_b.total_meshes);
    try std.testing.expectEqual(stats_a.rendered_meshes, stats_b.rendered_meshes);
    try std.testing.expectEqual(stats_a.culled_meshes, stats_b.culled_meshes);
    try std.testing.expectEqual(stats_a.occluded_meshes, stats_b.occluded_meshes);
    // …identical queue lengths…
    try std.testing.expectEqual(queues_a.items.items.len, queues_b.items.items.len);
    try std.testing.expect(stats_a.rendered_meshes > 0);
    try std.testing.expect(stats_a.culled_meshes > 0);
    // …and identical records in identical order (chunk merge order ==
    // serial mesh order, and the same world matrices feed both passes).
    for (queues_a.items.items, queues_b.items.items) |a, b| {
        try std.testing.expectEqual(a.mesh, b.mesh);
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.texture_id, b.texture_id);
        try std.testing.expectEqual(a.transparent, b.transparent);
    }
}

test "worldMatrixCached honors bone attachment like getWorldMatrix" {
    const ally = std.testing.allocator;
    const Skeleton = @import("../animation/skeleton.zig").Skeleton;

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
