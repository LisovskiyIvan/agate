const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Mat4 = math.Mat4;
const Frustum = math.Frustum;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../mesh.zig").Mesh;
const InstancedMesh = @import("../mesh.zig").InstancedMesh;
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
    /// Index of the source mesh in FrameCullContext.meshes. Feeds the
    /// transparent order tie-break (see TransparentDrawEntry.seq) so serial
    /// and parallel paths break exact-distance ties identically.
    mesh_index: u32 = 0,
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

    // Self-contained GPU geometry handles: decouples render execution from *Mesh lifetime
    vertex_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    index_count: u32 = 0,
    index_type: sg.IndexType = .UINT16,
    base_vertex: u32 = 0,
    is_u32: bool = false,
    is_skinned: bool = false,
};

/// Self-contained per-batch payload for instanced meshes. Decouples render execution
/// from *Mesh lifetime and state.
pub const RenderInstancedBatch = struct {
    vertex_buffer: sg.Buffer = .{},
    instance_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    index_count: u32 = 0,
    index_type: sg.IndexType = .UINT16,
    visible_instance_count: u32 = 0,
    is_pbr: bool = false,
    transparent: bool = false,
    double_sided: bool = false,
    is_decal: bool = false,
    receive_shadows: bool = true,
    draw_record: MaterialDrawRecord = .{},
    mesh: ?*Mesh = null,
};

/// One transparent draw in the unified back-to-front pass. Regular items
/// index `queues.transparent`, instanced groups index
/// `queues.transparent_instanced`; both draw as a single batch at their
/// sorted position (no per-instance sorting/OIT). `seq` is the source mesh
/// index (RenderMeshItem.mesh_index) and only breaks exact distance ties
/// deterministically. Mesh order is identical across serial and parallel
/// paths, so the tie-break — and therefore the sorted order — matches
/// exactly between them.
pub const TransparentKind = enum { regular, instanced };
pub const TransparentDrawEntry = struct {
    distance_sq: f32,
    seq: u32,
    kind: TransparentKind,
    index: u32,
    is_decal: bool,
};

/// Reusable scratch for the parallel cull pass (buildFrameQueuesParallel):
/// per-chunk record buffers plus per-chunk stats. Owned per `RenderQueues`
/// (each view queue builds independently, so each keeps its own scratch)
/// and retained across frames: `ensure` grows the outer list and the
/// per-record capacities only when the chunk/span demand exceeds what is
/// already held, `reset` clears lengths for reuse, and `deinit` (via
/// `RenderQueues.deinit`) frees everything at shutdown. Steady-state frames
/// therefore perform zero allocations in the parallel setup path.
pub const ParallelCullScratch = struct {
    records: std.ArrayListUnmanaged(std.ArrayListUnmanaged(RenderMeshItem)) = .empty,
    chunk_stats: std.ArrayListUnmanaged(SceneStats) = .empty,

    /// Fit to hold `chunk_count` chunks of `span` records each. All fallible
    /// growth happens here, before any queue/stats write and before
    /// `forkJoin`, so callers keep the OOM→serial-fallback contract:
    /// failure leaves queues/stats untouched. Growth is atomic:
    /// on error the scratch is freed back to empty (same as the old
    /// per-call cleanup), so a failed setup owns no buffers even if the
    /// caller never runs deinit; the next call simply regrows.
    pub fn ensure(self: *ParallelCullScratch, allocator: std.mem.Allocator, chunk_count: usize, span: usize) !void {
        errdefer {
            self.deinit(allocator);
            self.* = .{};
        }
        try self.records.ensureTotalCapacity(allocator, chunk_count);
        while (self.records.items.len < chunk_count) self.records.appendAssumeCapacity(.empty);
        // Shrinking must release the trailing records' owned buffers before
        // truncating the list, otherwise they become unreachable and leak.
        while (self.records.items.len > chunk_count) {
            var r = self.records.pop().?;
            r.deinit(allocator);
        }
        for (self.records.items) |*r| try r.ensureTotalCapacity(allocator, span);
        try self.chunk_stats.ensureTotalCapacity(allocator, chunk_count);
        while (self.chunk_stats.items.len < chunk_count) self.chunk_stats.appendAssumeCapacity(.{});
        if (self.chunk_stats.items.len > chunk_count) self.chunk_stats.items.len = chunk_count;
    }

    /// Clear per-chunk record lengths and stats for reuse. The outer record
    /// list length is kept — its elements own the retained buffers, and
    /// truncating it would leak them; `ensure` re-fits lengths to the next
    /// call's chunk count.
    pub fn reset(self: *ParallelCullScratch) void {
        for (self.records.items) |*r| r.clearRetainingCapacity();
        self.chunk_stats.clearRetainingCapacity();
    }

    pub fn deinit(self: *ParallelCullScratch, allocator: std.mem.Allocator) void {
        for (self.records.items) |*r| r.deinit(allocator);
        self.records.deinit(allocator);
        self.chunk_stats.deinit(allocator);
    }
};

/// The four draw queues plus the per-frame instance-matrix staging buffer.
/// Cleared and refilled by buildFrameQueues each render(); ownership stays
/// with Scene via a single field.
pub const RenderQueues = struct {
    items: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    // Transparent meshes (material alpha_mode == .blend), sorted strictly
    // back-to-front and drawn after every opaque mesh and instanced mesh.
    transparent: std.ArrayListUnmanaged(RenderMeshItem) = .empty,
    opaque_instanced: std.ArrayListUnmanaged(RenderInstancedBatch) = .empty,
    transparent_instanced: std.ArrayListUnmanaged(RenderInstancedBatch) = .empty,
    instance_matrices: std.ArrayListUnmanaged(Mat4) = .empty,
    // Unified transparent draw order across regular + instanced groups,
    // sorted by sortTransparentDrawOrder in renderSceneView. Retained
    // across frames (clearRetainingCapacity in reset: no per-frame churn).
    transparent_order: std.ArrayListUnmanaged(TransparentDrawEntry) = .empty,
    // Parallel-cull scratch (per-chunk record buffers + stats). Lives with
    // the view queues and is retained across frames (reset + ensure per
    // buildFrameQueuesParallel call, freed once in deinit), so repeated
    // per-view parallel culls reuse the same memory instead of
    // allocating/freeing chunk buffers on every call.
    parallel_scratch: ParallelCullScratch = .{},

    pub fn reset(self: *RenderQueues) void {
        self.items.clearRetainingCapacity();
        self.transparent.clearRetainingCapacity();
        self.opaque_instanced.clearRetainingCapacity();
        self.transparent_instanced.clearRetainingCapacity();
        self.instance_matrices.clearRetainingCapacity();
        self.transparent_order.clearRetainingCapacity();
    }

    pub fn deinit(self: *RenderQueues, allocator: std.mem.Allocator) void {
        self.items.deinit(allocator);
        self.transparent.deinit(allocator);
        self.opaque_instanced.deinit(allocator);
        self.transparent_instanced.deinit(allocator);
        self.instance_matrices.deinit(allocator);
        self.transparent_order.deinit(allocator);
        self.parallel_scratch.deinit(allocator);
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

// Unified transparent order: globally back-to-front by exact group distance
// so regular and instanced transparents interleave correctly. Strict weak
// ordering (exact distance, then decal, then source mesh index — never an
// epsilon band, which would be non-transitive: a≈b and b≈c need not imply
// a≈c). Exact ties (e.g. coplanar surfaces) keep decals after non-decals,
// then mesh order, for a deterministic result identical across paths.
pub fn sortTransparentDrawOrder(_: void, a: TransparentDrawEntry, b: TransparentDrawEntry) bool {
    if (a.distance_sq != b.distance_sq) {
        return a.distance_sq > b.distance_sq;
    }
    if (a.is_decal != b.is_decal) {
        return !a.is_decal;
    }
    return a.seq < b.seq;
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
    const min_meshes = if (ctx.parallel_min_meshes == 0)
        (if (pool) |p| @max(@as(usize, 64), (p.workerCount() + 1) * 32) else 128)
    else
        ctx.parallel_min_meshes;
    if (pool != null and pool.?.workerCount() > 0 and ctx.meshes.len >= min_meshes) {
        const parallel_ran = blk: {
            buildFrameQueuesParallel(ctx, frustum, eye, pool.?) catch break :blk false;
            break :blk true;
        };
        if (parallel_ran) return;
        // Parallel setup OOM: it fails before any queue/stats write (only
        // benign world-matrix cache warming precedes the fallible allocs),
        // so fall through to the serial loop and still draw the frame.
    }

    for (ctx.meshes, 0..) |mesh, mesh_index| {
        if (mesh.is_lod_child) continue;
        if ((mesh.layer_mask & ctx.culling_mask) == 0) continue;
        // Deferred-creation meshes (off-context uploadGeometry) have no GPU
        // buffers until finishGpuUpload runs; drawing them would bind
        // invalid handles, so they stay out of the queues until finished.
        if (mesh.gpu_pending) continue;
        if (mesh.instances.items.len > 0) {
            submitInstancedMesh(ctx, frustum, mesh, mesh_index);
            continue;
        }
        if (cullNonInstancedMesh(ctx, frustum, eye, mesh, ctx.stats, mesh_index)) |item| {
            appendRenderItem(ctx, item);
        }
    }
}

/// Routes a finished record into the draw queue it belongs to. Transparent
/// regulars also append a unified order entry (index into
/// queues.transparent); instanced transparents append theirs in
/// submitInstancedMesh so both share one mesh-index sequence.
fn appendRenderItem(ctx: FrameCullContext, item: RenderMeshItem) void {
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

const ParallelInstanceStage = struct {
    instances: []*InstancedMesh,
    span: usize,
    chunk_aabbs: []BoundingBox,
    chunk_visible_counts: []usize,
    chunk_write_offsets: []usize,
    out_matrices: []Mat4,

    fn updateTransformsAndAABBs(stage: *ParallelInstanceStage, start: usize, end: usize) void {
        for (start..end) |chunk_id| {
            const lo = chunk_id * stage.span;
            const hi = @min(lo + stage.span, stage.instances.len);
            var aabb = BoundingBox.zero;
            var visible_count: usize = 0;
            for (stage.instances[lo..hi]) |inst| {
                if (!inst.is_visible) continue;
                inst.updateCachedTransforms();
                visible_count += 1;
                if (aabb.isValid()) {
                    aabb = aabb.merge(inst.cached_bounding_box);
                } else {
                    aabb = inst.cached_bounding_box;
                }
            }
            stage.chunk_aabbs[chunk_id] = aabb;
            stage.chunk_visible_counts[chunk_id] = visible_count;
        }
    }

    fn scatterMatrices(stage: *ParallelInstanceStage, start: usize, end: usize) void {
        for (start..end) |chunk_id| {
            const lo = chunk_id * stage.span;
            const hi = @min(lo + stage.span, stage.instances.len);
            var dst_idx = stage.chunk_write_offsets[chunk_id];
            for (stage.instances[lo..hi]) |inst| {
                if (!inst.is_visible) continue;
                stage.out_matrices[dst_idx] = inst.cached_world_matrix;
                dst_idx += 1;
            }
        }
    }
};

/// Minimal per-frame input for instance staging, extracted from
/// FrameCullContext. Scene.prepareFrame pre-stages through this before the
/// shadow pass so ShadowPass.prepare snapshots the same frame's
/// cached_aabb / instance_buffer / visible_instance_count instead of the
/// previous frame's. The `instance_uploaded_frame != frame_id` guard keeps
/// staging once per frame, shared by all view queues.
pub const InstanceStageContext = struct {
    allocator: std.mem.Allocator,
    queues: *RenderQueues,
    thread_pool: ?*jobs.Pool,
    frame_id: u64,
    eye: Vec3,
};

/// Per-frame instance-matrix staging for one mesh: per-instance
/// transforms/AABB (parallel at >=256 instances), cached_aabb /
/// visible_instance_count update, transparent back-to-front sort by eye,
/// and sg instance-buffer create/update on the context thread. No-op when
/// already staged this frame.
pub fn stageInstancedMesh(sc: InstanceStageContext, mesh: *Mesh) void {
    // Deferred-creation meshes have no vertex/index buffers yet; staging
    // instance data for them would produce a draw against invalid handles.
    if (mesh.gpu_pending) return;
    if (mesh.instance_uploaded_frame != sc.frame_id) {
        mesh.instance_uploaded_frame = sc.frame_id;
        sc.queues.instance_matrices.clearRetainingCapacity();

        const pool = sc.thread_pool;
        const parallel_min_instances: usize = 256;
        if (pool != null and pool.?.workerCount() > 0 and mesh.instances.items.len >= parallel_min_instances) {
            const p = pool.?;
            const chunk_count = @min((p.workerCount() + 1) * 2, 64);
            const span = (mesh.instances.items.len + chunk_count - 1) / chunk_count;

            var chunk_aabbs_buf: [64]BoundingBox = undefined;
            var chunk_visible_buf: [64]usize = undefined;
            var chunk_offsets_buf: [64]usize = undefined;

            var stage = ParallelInstanceStage{
                .instances = mesh.instances.items,
                .span = span,
                .chunk_aabbs = chunk_aabbs_buf[0..chunk_count],
                .chunk_visible_counts = chunk_visible_buf[0..chunk_count],
                .chunk_write_offsets = chunk_offsets_buf[0..chunk_count],
                .out_matrices = &.{},
            };

            p.forkJoin(ParallelInstanceStage, &stage, ParallelInstanceStage.updateTransformsAndAABBs, chunk_count);

            var combined_aabb = BoundingBox.zero;
            var total_visible: usize = 0;
            for (0..chunk_count) |c| {
                chunk_offsets_buf[c] = total_visible;
                total_visible += chunk_visible_buf[c];
                if (chunk_aabbs_buf[c].isValid()) {
                    if (combined_aabb.isValid()) {
                        combined_aabb = combined_aabb.merge(chunk_aabbs_buf[c]);
                    } else {
                        combined_aabb = chunk_aabbs_buf[c];
                    }
                }
            }
            mesh.cached_aabb = combined_aabb;

            sc.queues.instance_matrices.resize(sc.allocator, total_visible) catch return;
            if (total_visible > 0) {
                stage.out_matrices = sc.queues.instance_matrices.items;
                p.forkJoin(ParallelInstanceStage, &stage, ParallelInstanceStage.scatterMatrices, chunk_count);
            }
        } else {
            var combined_aabb = math.BoundingBox.zero;
            for (mesh.instances.items) |inst| {
                if (!inst.is_visible) continue;
                inst.updateCachedTransforms();
                sc.queues.instance_matrices.append(sc.allocator, inst.cached_world_matrix) catch return;
                if (combined_aabb.isValid()) {
                    combined_aabb = combined_aabb.merge(inst.cached_bounding_box);
                } else {
                    combined_aabb = inst.cached_bounding_box;
                }
            }
            mesh.cached_aabb = combined_aabb;
        }
        const active_count = sc.queues.instance_matrices.items.len;
        mesh.visible_instance_count = @intCast(active_count);

        // Per-instance transparency sorting (OIT):
        // When the instanced mesh is transparent or a decal, sort its instance
        // matrices back-to-front relative to the camera eye. Farthest instances
        // render first, blending nearer instances over them correctly.
        if (active_count > 1 and (materialIsTransparent(mesh.material) or mesh.is_decal)) {
            const SortCtx = struct {
                eye: Vec3,
                pub fn sortFn(c: @This(), a: Mat4, b: Mat4) bool {
                    const pos_a = Vec3.new(a.m[12], a.m[13], a.m[14]);
                    const pos_b = Vec3.new(b.m[12], b.m[13], b.m[14]);
                    const dist_a = pos_a.sub(c.eye).lengthSq();
                    const dist_b = pos_b.sub(c.eye).lengthSq();
                    if (dist_a != dist_b) {
                        return dist_a > dist_b; // back-to-front: farthest first
                    }
                    for (0..16) |i| {
                        if (a.m[i] != b.m[i]) return a.m[i] < b.m[i];
                    }
                    return false;
                }
            };
            std.mem.sort(Mat4, sc.queues.instance_matrices.items[0..active_count], SortCtx{ .eye = sc.eye }, SortCtx.sortFn);
        }

        if (active_count > 0 and sg.isvalid()) {
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
                sg.updateBuffer(mesh.instance_buffer, sg.asRange(sc.queues.instance_matrices.items[0..active_count]));
                mesh.instance_hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sc.queues.instance_matrices.items[0..active_count]));
                mesh.instance_uploaded_count = active_count;
            } else {
                const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sc.queues.instance_matrices.items[0..active_count]));
                if (active_count != mesh.instance_uploaded_count or h != mesh.instance_hash) {
                    sg.updateBuffer(mesh.instance_buffer, sg.asRange(sc.queues.instance_matrices.items[0..active_count]));
                    mesh.instance_hash = h;
                    mesh.instance_uploaded_count = active_count;
                }
            }
        }
    }
}

/// Pre-stage every instance-bearing mesh once per frame. Skips LOD children,
/// GPU-pending meshes, and meshes with no instances; staging itself is
/// guarded per mesh by `instance_uploaded_frame`, so calling this before the
/// view queues and again implicitly via submitInstancedMesh stays once-only.
pub fn stageInstances(sc: InstanceStageContext, meshes: []const *Mesh) void {
    for (meshes) |mesh| {
        if (mesh.is_lod_child) continue;
        if (mesh.gpu_pending) continue;
        if (mesh.instances.items.len == 0) continue;
        stageInstancedMesh(sc, mesh);
    }
}

/// Instance-bearing mesh handling: per-frame instance-matrix staging (parallel
/// when attached to a pool and exceeding parallel_min_instances), sg buffer
/// (re)creation/upload on the context thread, frustum test on the combined AABB,
/// instanced queue fill.
fn submitInstancedMesh(ctx: FrameCullContext, frustum: Frustum, mesh: *Mesh, mesh_index: usize) void {
    // Deferred-creation meshes have no vertex/index buffers yet; staging
    // instance data for them would produce a draw against invalid handles.
    if (mesh.gpu_pending) return;
    // Pre-staged by stageInstances before the shadow pass when running under
    // Scene.prepareFrame; the frame guard makes this a no-op then, while
    // direct callers (tests, parallel merge tail) still stage here.
    stageInstancedMesh(.{
        .allocator = ctx.allocator,
        .queues = ctx.queues,
        .thread_pool = ctx.thread_pool,
        .frame_id = ctx.frame_id,
        .eye = ctx.eye,
    }, mesh);

    if (mesh.visible_instance_count > 0 and ((mesh.layer_mask & ctx.culling_mask) != 0)) {
        if (ctx.cull_frustum and mesh.cached_aabb.isValid() and !frustum.intersectsAABB(mesh.cached_aabb)) {
            ctx.stats.culled_meshes += @intCast(mesh.instances.items.len);
            return;
        }
        ctx.stats.total_meshes += @intCast(mesh.instances.items.len);
        ctx.stats.rendered_meshes += mesh.visible_instance_count;

        const is_pbr = if (mesh.material) |m| (m == .pbr) else false;
        const is_trans = materialIsTransparent(mesh.material) or mesh.is_decal;
        const is_ds = materialIsDoubleSided(mesh.material);
        const draw_rec = buildMaterialRecord(ctx, mesh.material);

        const batch = RenderInstancedBatch{
            .vertex_buffer = mesh.vertex_buffer,
            .instance_buffer = mesh.instance_buffer,
            .index_buffer = mesh.index_buffer,
            .index_count = mesh.index_count,
            .index_type = mesh.index_type,
            .visible_instance_count = mesh.visible_instance_count,
            .is_pbr = is_pbr,
            .transparent = is_trans,
            .double_sided = is_ds,
            .is_decal = mesh.is_decal,
            .receive_shadows = mesh.receive_shadows,
            .draw_record = draw_rec,
            .mesh = mesh,
        };

        if (is_trans) {
            // Reserve both slots up front (see appendRenderItem): the group
            // and its order entry are appended atomically under OOM.
            ctx.queues.transparent_instanced.ensureUnusedCapacity(ctx.allocator, 1) catch return;
            ctx.queues.transparent_order.ensureUnusedCapacity(ctx.allocator, 1) catch return;
            // Group distance key: combined AABB center (batch draws as one;
            // no per-instance sorting). Falls back to the mesh position when
            // the combined AABB is degenerate.
            const center = if (mesh.cached_aabb.isValid()) mesh.cached_aabb.center() else mesh.position;
            const idx: u32 = @intCast(ctx.queues.transparent_instanced.items.len);
            const seq: u32 = @intCast(mesh_index);
            ctx.queues.transparent_instanced.appendAssumeCapacity(batch);
            ctx.queues.transparent_order.appendAssumeCapacity(.{
                .distance_sq = center.sub(ctx.eye).lengthSq(),
                .seq = seq,
                .kind = .instanced,
                .index = idx,
                .is_decal = mesh.is_decal,
            });
        } else {
            ctx.queues.opaque_instanced.append(ctx.allocator, batch) catch {};
        }
    }
}

fn buildMaterialRecord(ctx: FrameCullContext, mat: ?Material) MaterialDrawRecord {
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
fn cullNonInstancedMesh(
    ctx: FrameCullContext,
    frustum: Frustum,
    eye: Vec3,
    mesh: *Mesh,
    stats: *SceneStats,
    mesh_index: usize,
) ?RenderMeshItem {
    stats.total_meshes += 1;
    if (!mesh.is_visible) return null;

    // Fresh world matrix first: it refreshes mesh.cached_aabb for this
    // frame, so the LOD distance below never reads a stale AABB tagged with
    // a previous frame id (the parallel path pre-warms this cache; the
    // serial path must pick the same LOD at distance thresholds).
    const model = worldMatrixCached(ctx.frame_id, mesh);
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
    return .{
        .mesh = render_mesh,
        .material = mat,
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
        .skin_matrices = skin_mat,
        .morph_uniforms = morph_u,
        .morph_view = morph_v,
        .vertex_buffer = render_mesh.vertex_buffer,
        .index_buffer = render_mesh.index_buffer,
        .index_count = render_mesh.index_count,
        .index_type = render_mesh.index_type,
        .is_u32 = render_mesh.index_type == .UINT32,
        .is_skinned = render_mesh.skeleton != null,
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
            // chunk_count can exceed meshes.len (span floors at 1), so tail
            // chunks beyond the mesh list must be skipped: without this,
            // `lo..hi` would have start > end and panic.
            if (lo >= pass.ctx.meshes.len) continue;
            const hi = @min(lo + pass.span, pass.ctx.meshes.len);
            var local = SceneStats{};
            for (pass.ctx.meshes[lo..hi], lo..) |mesh, mesh_index| {
                if (mesh.is_lod_child) continue;
                if ((mesh.layer_mask & pass.ctx.culling_mask) == 0) continue;
                // Deferred-creation meshes have no buffers yet (same skip as
                // the serial loop above).
                if (mesh.gpu_pending) continue;
                // Instance-bearing meshes are staged serially in the merge
                // tail (sg buffer management); skip them here so they are
                // never culled as plain meshes and double-queued.
                if (mesh.instances.items.len > 0) continue;
                if (cullNonInstancedMesh(pass.ctx, pass.frustum, pass.eye, mesh, &local, mesh_index)) |item| {
                    pass.records[chunk_id].appendAssumeCapacity(item);
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

    // Reusable per-view scratch, retained across frames: reset clears the
    // previous call's lengths, ensure grows only when the chunk/span demand
    // exceeds what is already held. ALL fallible growth happens here,
    // before any queue/stats write and before forkJoin, so OOM still fails
    // with queues/stats untouched and the caller falls back to serial.
    const scratch = &ctx.queues.parallel_scratch;
    scratch.reset();
    try scratch.ensure(ctx.allocator, chunk_count, span);
    const records = scratch.records.items[0..chunk_count];
    const chunk_stats = scratch.chunk_stats.items[0..chunk_count];
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

    // Pre-reserve capacity in destination queues to minimize reallocations
    var total_rendered: usize = 0;
    for (chunk_stats) |local| {
        total_rendered += local.rendered_meshes;
    }
    ctx.queues.items.ensureUnusedCapacity(ctx.allocator, total_rendered) catch {};

    // Deterministic merge: chunk order == mesh order, so the queues land
    // exactly as the serial loop would have filled them.
    for (records, chunk_stats) |*list, local| {
        ctx.stats.total_meshes += local.total_meshes;
        ctx.stats.rendered_meshes += local.rendered_meshes;
        ctx.stats.culled_meshes += local.culled_meshes;
        ctx.stats.occluded_meshes += local.occluded_meshes;
        for (list.items) |item| appendRenderItem(ctx, item);
    }

    // Instance-bearing meshes: serial submission (sg buffer management).
    for (ctx.meshes, 0..) |mesh, mesh_index| {
        if (mesh.is_lod_child or mesh.instances.items.len == 0) continue;
        if ((mesh.layer_mask & ctx.culling_mask) == 0) continue;
        submitInstancedMesh(ctx, frustum, mesh, mesh_index);
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

test "parallel cull reuses scratch across calls" {
    const ally = std.testing.allocator;

    // Deterministic scene, forced onto the parallel path; no frustum or
    // occlusion culling so every mesh queues identically each frame.
    const count = 512;
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
        };
        const f: f32 = @floatFromInt(i);
        meshes[i].position = Vec3.new(f * 0.01, 0.0, f * 0.005);
        ptrs[i] = &meshes[i];
    }

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    var culler = visibility.OcclusionCuller.init();
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    const chunk_count = (pool.workerCount() + 1) * 4;
    const span = (count + chunk_count - 1) / chunk_count;

    // First parallel frame.
    var stats_a = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .frame_id = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_a,
        .queues = &queues,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });
    try std.testing.expectEqual(@as(usize, count), queues.items.items.len);
    const snapshot = try ally.alloc(RenderMeshItem, queues.items.items.len);
    defer ally.free(snapshot);
    @memcpy(snapshot, queues.items.items);

    // Scratch grew to the demand: outer capacity covers every chunk and
    // each per-chunk record buffer covers its full span.
    try std.testing.expectEqual(chunk_count, queues.parallel_scratch.records.items.len);
    try std.testing.expect(queues.parallel_scratch.records.capacity >= chunk_count);
    for (queues.parallel_scratch.records.items) |*r| {
        try std.testing.expect(r.capacity >= span);
    }
    const outer_cap = queues.parallel_scratch.records.capacity;
    const inner_caps = try ally.alloc(usize, chunk_count);
    defer ally.free(inner_caps);
    for (queues.parallel_scratch.records.items, 0..) |*r, i| inner_caps[i] = r.capacity;

    // Second parallel frame reuses the same queues (draw queues reset;
    // scratch retained) and must reproduce the first frame exactly.
    queues.reset();
    var stats_b = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .frame_id = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_b,
        .queues = &queues,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });
    try std.testing.expectEqual(stats_a.total_meshes, stats_b.total_meshes);
    try std.testing.expectEqual(stats_a.rendered_meshes, stats_b.rendered_meshes);
    try std.testing.expectEqual(stats_a.culled_meshes, stats_b.culled_meshes);
    try std.testing.expectEqual(stats_a.occluded_meshes, stats_b.occluded_meshes);
    try std.testing.expectEqual(snapshot.len, queues.items.items.len);
    for (snapshot, queues.items.items) |a, b| {
        try std.testing.expectEqual(a.mesh, b.mesh);
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.texture_id, b.texture_id);
        try std.testing.expectEqual(a.transparent, b.transparent);
    }

    // No regrowth on the second call: identical demand reuses the retained
    // buffers (same capacities, leak-checked by the testing allocator via
    // queues.deinit).
    try std.testing.expectEqual(outer_cap, queues.parallel_scratch.records.capacity);
    for (queues.parallel_scratch.records.items, 0..) |*r, i| {
        try std.testing.expectEqual(inner_caps[i], r.capacity);
    }
}

test "stale AABB does not drive LOD selection" {
    const ally = std.testing.allocator;
    const unit_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));

    var lod_far = Mesh{
        .name = "lod_far",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    var base = Mesh{
        .name = "lod_base",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        // Fresh position is beyond the switch distance (far).
        .position = Vec3.new(100, 0, 0),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    try base.addLODLevel(ally, 50.0, &lod_far);
    defer base.lod_levels.deinit(ally);

    // Seed a stale AABB tagged with an older frame: as if the mesh sat at
    // the origin (near, distance 0 < 50) last frame, while its fresh
    // position is far (distance 100 >= 50). Reading it would pick the near
    // LOD (self); the fresh AABB must pick the far child.
    base.cached_aabb = unit_box;
    base.cached_matrix = Mat4.identity;
    base.cached_frame = 0;

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const meshes = [_]*Mesh{&base};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(&lod_far, queues.items.items[0].mesh);
}

test "stale AABB LOD selection matches parallel path" {
    const ally = std.testing.allocator;
    const unit_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));

    var lod_far_s = Mesh{
        .name = "lod_far_s",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    var base_s = Mesh{
        .name = "lod_base_s",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(100, 0, 0),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    try base_s.addLODLevel(ally, 50.0, &lod_far_s);
    defer base_s.lod_levels.deinit(ally);
    base_s.cached_aabb = unit_box;
    base_s.cached_matrix = Mat4.identity;
    base_s.cached_frame = 0;

    var lod_far_p = Mesh{
        .name = "lod_far_p",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    var base_p = Mesh{
        .name = "lod_base_p",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(100, 0, 0),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    try base_p.addLODLevel(ally, 50.0, &lod_far_p);
    defer base_p.lod_levels.deinit(ally);
    base_p.cached_aabb = unit_box;
    base_p.cached_matrix = Mat4.identity;
    base_p.cached_frame = 0;

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    var culler = visibility.OcclusionCuller.init();

    // Serial pass (no pool attached).
    var stats_s = SceneStats{};
    var queues_s = RenderQueues{};
    defer queues_s.deinit(ally);
    const meshes_s = [_]*Mesh{&base_s};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes_s,
        .frame_id = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_s,
        .queues = &queues_s,
        .default_white_id = 1,
    });

    // Parallel pass (forced past the mesh threshold, 2 workers): pre-warms
    // the world-matrix cache, so it always decided from the fresh AABB.
    var stats_p = SceneStats{};
    var queues_p = RenderQueues{};
    defer queues_p.deinit(ally);
    const meshes_p = [_]*Mesh{&base_p};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes_p,
        .frame_id = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_p,
        .queues = &queues_p,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });

    // Both paths must agree on the fresh (far) LOD despite the stale seed.
    try std.testing.expectEqual(@as(usize, 1), queues_s.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), queues_p.items.items.len);
    try std.testing.expectEqual(&lod_far_s, queues_s.items.items[0].mesh);
    try std.testing.expectEqual(&lod_far_p, queues_p.items.items[0].mesh);
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

test "parallel instanced staging produces serial-identical instance matrices" {
    const ally = std.testing.allocator;
    var source_mesh = Mesh{
        .name = "source",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
        .base_matrix = Mat4.identity,
    };

    const count = 300;
    const inst_mem = try ally.alloc(InstancedMesh, count);
    defer ally.free(inst_mem);
    const inst_ptrs = try ally.alloc(*InstancedMesh, count);
    defer ally.free(inst_ptrs);

    for (0..count) |i| {
        const fi: f32 = @floatFromInt(i);
        inst_mem[i] = InstancedMesh{
            .name = "inst",
            .source_mesh = &source_mesh,
            .position = Vec3.new(fi * 0.1, fi * 0.2, fi * 0.3),
            .rotation = Vec3.new(fi * 0.01, fi * 0.02, 0),
            .scaling = Vec3.new(1, 1, 1),
            .is_visible = (i % 7 != 0),
        };
        inst_ptrs[i] = &inst_mem[i];
    }

    var mesh_serial = Mesh{
        .name = "inst_parent_s",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = inst_ptrs, .capacity = count },
    };

    // Serial staging
    var queues_s = RenderQueues{};
    defer queues_s.deinit(ally);
    var stats_s = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const ctx_s = FrameCullContext{
        .allocator = ally,
        .thread_pool = null,
        .default_white_id = 1,
        .meshes = &.{},
        .frame_id = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_s,
        .queues = &queues_s,
    };
    submitInstancedMesh(ctx_s, Frustum.fromViewProjection(Mat4.identity), &mesh_serial, 0);

    // Parallel staging
    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    var mesh_parallel = Mesh{
        .name = "inst_parent_p",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = inst_ptrs, .capacity = count },
    };
    var queues_p = RenderQueues{};
    defer queues_p.deinit(ally);
    var stats_p = SceneStats{};
    const ctx_p = FrameCullContext{
        .allocator = ally,
        .thread_pool = pool,
        .default_white_id = 1,
        .meshes = &.{},
        .frame_id = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_p,
        .queues = &queues_p,
    };
    submitInstancedMesh(ctx_p, Frustum.fromViewProjection(Mat4.identity), &mesh_parallel, 0);

    // Verify bit-identical results
    try std.testing.expectEqual(mesh_serial.cached_aabb, mesh_parallel.cached_aabb);
    try std.testing.expectEqual(queues_s.instance_matrices.items.len, queues_p.instance_matrices.items.len);
    try std.testing.expect(queues_s.instance_matrices.items.len > 0);
    for (queues_s.instance_matrices.items, queues_p.instance_matrices.items) |m_s, m_p| {
        try std.testing.expectEqual(m_s, m_p);
    }
}

test "transparent regular+instanced groups share one back-to-front order" {
    const ally = std.testing.allocator;
    const material = @import("../material.zig");

    var blend_mat = material.StandardMaterial.init("blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .standard = &blend_mat };
    const unit_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5));

    var regular = Mesh{
        .name = "regular",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(0, 0, 10),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
        .material = blend,
    };

    var src_far = Mesh{
        .name = "src_far",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var src_near = Mesh{
        .name = "src_near",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var inst_far = InstancedMesh{
        .name = "far_inst",
        .source_mesh = &src_far,
        .position = Vec3.new(0, 0, 15),
    };
    var inst_near = InstancedMesh{
        .name = "near_inst",
        .source_mesh = &src_near,
        .position = Vec3.new(0, 0, 5),
    };
    var far_ptrs = [_]*InstancedMesh{&inst_far};
    var near_ptrs = [_]*InstancedMesh{&inst_near};
    var far_parent = Mesh{
        .name = "far_group",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = blend,
        .culling_strategy = .always_render,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &far_ptrs, .capacity = 1 },
    };
    var near_parent = Mesh{
        .name = "near_group",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = blend,
        .culling_strategy = .always_render,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &near_ptrs, .capacity = 1 },
    };

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const meshes = [_]*Mesh{ &regular, &far_parent, &near_parent };
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 11,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    // One regular transparent plus two instanced transparent groups share a
    // single order list; per-instance batching is preserved (one entry per
    // group, not per instance).
    try std.testing.expectEqual(@as(usize, 1), queues.transparent.items.len);
    try std.testing.expectEqual(@as(usize, 2), queues.transparent_instanced.items.len);
    try std.testing.expectEqual(@as(usize, 3), queues.transparent_order.items.len);
    try std.testing.expectEqual(@as(u32, 3), stats.rendered_meshes);
    // Material snapshot survives the queue: regular item kept its material.
    try std.testing.expect(queues.transparent.items[0].material != null);
    try std.testing.expect(queues.transparent.items[0].transparent);

    std.mem.sort(TransparentDrawEntry, queues.transparent_order.items, {}, sortTransparentDrawOrder);
    const ordered = queues.transparent_order.items;
    // Global back-to-front: far group (15^2=225), regular (10^2=100), near (5^2=25).
    try std.testing.expectApproxEqAbs(@as(f32, 225.0), ordered[0].distance_sq, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), ordered[1].distance_sq, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), ordered[2].distance_sq, 1e-2);
    try std.testing.expectEqual(TransparentKind.instanced, ordered[0].kind);
    try std.testing.expectEqual(TransparentKind.regular, ordered[1].kind);
    try std.testing.expectEqual(TransparentKind.instanced, ordered[2].kind);
    try std.testing.expectEqual(&far_parent, queues.transparent_instanced.items[ordered[0].index].mesh);
    try std.testing.expectEqual(@as(u32, 0), ordered[1].index);
    try std.testing.expectEqual(&near_parent, queues.transparent_instanced.items[ordered[2].index].mesh);

    // Deterministic tie-break: exactly equal distances keep seq order.
    var ties = [_]TransparentDrawEntry{
        .{ .distance_sq = 4.0, .seq = 7, .kind = .regular, .index = 0, .is_decal = false },
        .{ .distance_sq = 4.0, .seq = 3, .kind = .instanced, .index = 0, .is_decal = false },
    };
    std.mem.sort(TransparentDrawEntry, &ties, {}, sortTransparentDrawOrder);
    try std.testing.expectEqual(@as(u32, 3), ties[0].seq);
    try std.testing.expectEqual(@as(u32, 7), ties[1].seq);

    // Strict weak ordering: near-equal but distinct distances sort by exact
    // distance, never by tie-break. An epsilon band would be non-transitive
    // here (first≈second and second≈third within 1e-4, yet first≉third).
    var near = [_]TransparentDrawEntry{
        .{ .distance_sq = 1.0, .seq = 0, .kind = .regular, .index = 0, .is_decal = false },
        .{ .distance_sq = 1.0 + 5e-5, .seq = 1, .kind = .regular, .index = 1, .is_decal = false },
        .{ .distance_sq = 1.0 + 1e-4, .seq = 2, .kind = .regular, .index = 2, .is_decal = false },
    };
    std.mem.sort(TransparentDrawEntry, &near, {}, sortTransparentDrawOrder);
    try std.testing.expectEqual(@as(u32, 2), near[0].index);
    try std.testing.expectEqual(@as(u32, 1), near[1].index);
    try std.testing.expectEqual(@as(u32, 0), near[2].index);
}

test "parallel cull mixed scene matches serial on all queues" {
    const ally = std.testing.allocator;
    const material = @import("../material.zig");

    var blend_mat = material.StandardMaterial.init("mixed_blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .standard = &blend_mat };
    const unit_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5));

    var opaque_a = Mesh{
        .name = "opaque_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(1, 0, 0),
        .local_bounding_box = unit_box,
    };
    var trans_a = Mesh{
        .name = "trans_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .local_bounding_box = unit_box,
        .material = blend,
    };
    var src_opaque = Mesh{
        .name = "src_opaque",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var inst_op_0 = InstancedMesh{ .name = "op0", .source_mesh = &src_opaque, .position = Vec3.new(2, 0, 0) };
    var inst_op_1 = InstancedMesh{ .name = "op1", .source_mesh = &src_opaque, .position = Vec3.new(3, 0, 0) };
    var op_ptrs = [_]*InstancedMesh{ &inst_op_0, &inst_op_1 };
    var inst_opaque = Mesh{
        .name = "inst_opaque",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &op_ptrs, .capacity = 2 },
    };
    var tie_regular = Mesh{
        .name = "tie_regular",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(7, 0, 0),
        .local_bounding_box = unit_box,
        .material = blend,
    };
    var src_tie = Mesh{
        .name = "src_tie",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    // Same center as tie_regular (7,0,0), same non-decal flag: an exact
    // distance tie between a regular and an instanced transparent.
    var inst_tie = InstancedMesh{ .name = "tie_inst", .source_mesh = &src_tie, .position = Vec3.new(7, 0, 0) };
    var tie_ptrs = [_]*InstancedMesh{&inst_tie};
    var inst_trans_tie = Mesh{
        .name = "inst_trans_tie",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = blend,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &tie_ptrs, .capacity = 1 },
    };
    var decal_a = Mesh{
        .name = "decal_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(3, 0, 0),
        .local_bounding_box = unit_box,
        .is_decal = true,
    };
    var opaque_b = Mesh{
        .name = "opaque_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(9, 0, 0),
        .local_bounding_box = unit_box,
    };
    var src_far = Mesh{
        .name = "src_far",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var inst_far = InstancedMesh{ .name = "far_inst", .source_mesh = &src_far, .position = Vec3.new(15, 0, 0) };
    var far_ptrs = [_]*InstancedMesh{&inst_far};
    var inst_trans_far = Mesh{
        .name = "inst_trans_far",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = blend,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &far_ptrs, .capacity = 1 },
    };

    const meshes = [_]*Mesh{ &opaque_a, &trans_a, &inst_opaque, &tie_regular, &inst_trans_tie, &decal_a, &opaque_b, &inst_trans_far };

    var culler = visibility.OcclusionCuller.init();
    var stats_a = SceneStats{};
    var queues_a = RenderQueues{};
    defer queues_a.deinit(ally);
    var stats_b = SceneStats{};
    var queues_b = RenderQueues{};
    defer queues_b.deinit(ally);

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 21,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_a,
        .queues = &queues_a,
        .default_white_id = 1,
    });
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 22,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_b,
        .queues = &queues_b,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });

    // No double-queueing of instanced meshes: identical stat totals.
    try std.testing.expectEqual(stats_a.total_meshes, stats_b.total_meshes);
    try std.testing.expectEqual(stats_a.rendered_meshes, stats_b.rendered_meshes);
    try std.testing.expectEqual(stats_a.culled_meshes, stats_b.culled_meshes);
    try std.testing.expectEqual(stats_a.occluded_meshes, stats_b.occluded_meshes);
    try std.testing.expect(stats_a.rendered_meshes > 0);

    // Opaque regulars: identical records in identical order.
    try std.testing.expectEqual(queues_a.items.items.len, queues_b.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), queues_a.items.items.len);
    for (queues_a.items.items, queues_b.items.items) |a, b| {
        try std.testing.expectEqual(a.mesh, b.mesh);
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.texture_id, b.texture_id);
        try std.testing.expectEqual(a.transparent, b.transparent);
        try std.testing.expectEqual(a.is_decal, b.is_decal);
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
    }

    // Transparent regulars (incl. the decal): identical records in order.
    try std.testing.expectEqual(queues_a.transparent.items.len, queues_b.transparent.items.len);
    try std.testing.expectEqual(@as(usize, 3), queues_a.transparent.items.len);
    for (queues_a.transparent.items, queues_b.transparent.items) |a, b| {
        try std.testing.expectEqual(a.mesh, b.mesh);
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.is_decal, b.is_decal);
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
    }

    // Instanced groups: submitted once each, in mesh order, on both paths.
    try std.testing.expectEqual(queues_a.opaque_instanced.items.len, queues_b.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(usize, 1), queues_a.opaque_instanced.items.len);
    try std.testing.expectEqual(queues_a.opaque_instanced.items[0].mesh, queues_b.opaque_instanced.items[0].mesh);
    try std.testing.expectEqual(&inst_opaque, queues_a.opaque_instanced.items[0].mesh);
    try std.testing.expectEqual(queues_a.transparent_instanced.items.len, queues_b.transparent_instanced.items.len);
    try std.testing.expectEqual(@as(usize, 2), queues_a.transparent_instanced.items.len);
    for (queues_a.transparent_instanced.items, queues_b.transparent_instanced.items) |a, b| {
        try std.testing.expectEqual(a.mesh, b.mesh);
    }
    try std.testing.expectEqual(&inst_trans_tie, queues_a.transparent_instanced.items[0].mesh);
    try std.testing.expectEqual(&inst_trans_far, queues_a.transparent_instanced.items[1].mesh);

    // Staged instance matrices: identical contents.
    try std.testing.expectEqual(queues_a.instance_matrices.items.len, queues_b.instance_matrices.items.len);
    try std.testing.expect(queues_a.instance_matrices.items.len > 0);
    for (queues_a.instance_matrices.items, queues_b.instance_matrices.items) |a, b| {
        try std.testing.expectEqual(a, b);
    }

    // Unified transparent order: raw insertion differs (parallel merges all
    // regulars before the instanced tail), so compare the sorted order —
    // the contract both paths must honor.
    try std.testing.expectEqual(queues_a.transparent_order.items.len, queues_b.transparent_order.items.len);
    try std.testing.expectEqual(@as(usize, 5), queues_a.transparent_order.items.len);
    std.mem.sort(TransparentDrawEntry, queues_a.transparent_order.items, {}, sortTransparentDrawOrder);
    std.mem.sort(TransparentDrawEntry, queues_b.transparent_order.items, {}, sortTransparentDrawOrder);
    for (queues_a.transparent_order.items, queues_b.transparent_order.items) |a, b| {
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.seq, b.seq);
        try std.testing.expectEqual(a.kind, b.kind);
        try std.testing.expectEqual(a.index, b.index);
        try std.testing.expectEqual(a.is_decal, b.is_decal);
        if (a.kind == .regular) {
            try std.testing.expectEqual(
                queues_a.transparent.items[a.index].mesh,
                queues_b.transparent.items[b.index].mesh,
            );
        } else {
            try std.testing.expectEqual(
                queues_a.transparent_instanced.items[a.index].mesh,
                queues_b.transparent_instanced.items[b.index].mesh,
            );
        }
    }

    // Exact-distance tie (regular mesh 3 + instanced group 4 at 7^2 = 49):
    // the entries are bit-identical distances and mesh-index order wins on
    // both paths.
    const ordered = queues_a.transparent_order.items;
    try std.testing.expect(ordered[1].distance_sq == ordered[2].distance_sq);
    try std.testing.expectEqual(@as(f32, 49.0), ordered[1].distance_sq);
    try std.testing.expectEqual(TransparentKind.regular, ordered[1].kind);
    try std.testing.expectEqual(TransparentKind.instanced, ordered[2].kind);
    try std.testing.expectEqual(@as(u32, 3), ordered[1].seq);
    try std.testing.expectEqual(@as(u32, 4), ordered[2].seq);
}

test "parallel setup OOM fails cleanly without leaking" {
    const ally = std.testing.allocator;

    const count = 4;
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
        };
        ptrs[i] = &meshes[i];
    }

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    // (2 workers + 1) * 4 = 12 per-chunk buffers: fail_index 0 hits the
    // records array itself, 1 the first chunk buffer, 7 a middle one.
    for ([_]usize{ 0, 1, 7 }) |fail_index| {
        var gpa = std.heap.DebugAllocator(.{}){};
        var failing = std.testing.FailingAllocator.init(gpa.allocator(), .{ .fail_index = fail_index });
        var queues = RenderQueues{};
        var stats = SceneStats{};
        var culler = visibility.OcclusionCuller.init();
        const result = buildFrameQueuesParallel(.{
            .allocator = failing.allocator(),
            .meshes = ptrs,
            .frame_id = 100 + fail_index,
            .view_proj = Mat4.identity,
            .eye = Vec3.zero,
            .cull_frustum = false,
            .cull_occlusion = false,
            .occlusion_culler = &culler,
            .stats = &stats,
            .queues = &queues,
            .default_white_id = 1,
        }, Frustum.fromViewProjection(Mat4.identity), Vec3.zero, pool);
        try std.testing.expectError(error.OutOfMemory, result);
        // Setup failed before any queue/stats write: the frame is untouched.
        try std.testing.expectEqual(@as(usize, 0), queues.items.items.len);
        try std.testing.expectEqual(@as(usize, 0), queues.transparent.items.len);
        try std.testing.expectEqual(@as(usize, 0), queues.transparent_order.items.len);
        try std.testing.expectEqual(@as(u32, 0), stats.total_meshes);
        try std.testing.expectEqual(@as(u32, 0), stats.rendered_meshes);
        try std.testing.expectEqual(@as(u32, 0), stats.culled_meshes);
        // The GPA deinit check is the leak proof: every buffer allocated
        // before the failure was freed by the setup cleanup.
        try std.testing.expectEqual(std.heap.Check.ok, gpa.deinit());
    }
}

/// Test allocator that fails the first N allocations, then delegates to
/// backing. Unlike FailingAllocator (which fails persistently at its index),
/// a single failure lets the serial fallback that follows a parallel setup
/// OOM allocate normally.
const FailFirstN = struct {
    backing: std.mem.Allocator,
    failures_left: usize,

    fn allocator(self: *FailFirstN) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *FailFirstN = @ptrCast(@alignCast(ctx));
        if (self.failures_left > 0) {
            self.failures_left -= 1;
            return null;
        }
        return self.backing.rawAlloc(len, alignment, ra);
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *FailFirstN = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory, alignment, new_len, ra);
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *FailFirstN = @ptrCast(@alignCast(ctx));
        return self.backing.rawRemap(memory, alignment, new_len, ra);
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *FailFirstN = @ptrCast(@alignCast(ctx));
        return self.backing.rawFree(memory, alignment, ra);
    }
};

test "parallel setup OOM falls back to serial queues" {
    const ally = std.testing.allocator;
    const material = @import("../material.zig");

    var blend_mat = material.StandardMaterial.init("fallback_blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .standard = &blend_mat };
    const unit_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5));

    var plain_a = Mesh{
        .name = "plain_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(1, 0, 0),
        .local_bounding_box = unit_box,
    };
    var plain_b = Mesh{
        .name = "plain_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .local_bounding_box = unit_box,
        .material = blend,
    };
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var inst = InstancedMesh{ .name = "i", .source_mesh = &src, .position = Vec3.new(2, 0, 0) };
    var inst_ptrs = [_]*InstancedMesh{&inst};
    var inst_parent = Mesh{
        .name = "inst_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &inst_ptrs, .capacity = 1 },
    };
    const meshes = [_]*Mesh{ &plain_a, &plain_b, &inst_parent };

    var culler = visibility.OcclusionCuller.init();
    var stats_a = SceneStats{};
    var queues_a = RenderQueues{};
    defer queues_a.deinit(ally);
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 31,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_a,
        .queues = &queues_a,
        .default_white_id = 1,
    });

    // One failed allocation — the parallel records array — then the serial
    // fallback allocates normally through the same allocator.
    var limited = FailFirstN{ .backing = ally, .failures_left = 1 };
    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();
    var stats_b = SceneStats{};
    var queues_b = RenderQueues{};
    defer queues_b.deinit(limited.allocator());
    buildFrameQueues(.{
        .allocator = limited.allocator(),
        .meshes = &meshes,
        .frame_id = 32,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_b,
        .queues = &queues_b,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });

    // The frame still drew: serial-identical queues, not empty ones.
    try std.testing.expect(stats_b.rendered_meshes > 0);
    try std.testing.expectEqual(stats_a.total_meshes, stats_b.total_meshes);
    try std.testing.expectEqual(stats_a.rendered_meshes, stats_b.rendered_meshes);
    try std.testing.expectEqual(stats_a.culled_meshes, stats_b.culled_meshes);
    try std.testing.expectEqual(queues_a.items.items.len, queues_b.items.items.len);
    try std.testing.expectEqual(queues_a.transparent.items.len, queues_b.transparent.items.len);
    try std.testing.expectEqual(queues_a.opaque_instanced.items.len, queues_b.opaque_instanced.items.len);
    try std.testing.expectEqual(queues_a.transparent_instanced.items.len, queues_b.transparent_instanced.items.len);
    try std.testing.expectEqual(queues_a.transparent_order.items.len, queues_b.transparent_order.items.len);
    for (queues_a.items.items, queues_b.items.items) |a, b| {
        try std.testing.expectEqual(a.mesh, b.mesh);
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
    }
    for (queues_a.transparent.items, queues_b.transparent.items) |a, b| {
        try std.testing.expectEqual(a.mesh, b.mesh);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
    }
    for (queues_a.opaque_instanced.items, queues_b.opaque_instanced.items) |a, b| {
        try std.testing.expectEqual(a.mesh, b.mesh);
    }
    for (queues_a.transparent_order.items, queues_b.transparent_order.items) |a, b| {
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.seq, b.seq);
        try std.testing.expectEqual(a.kind, b.kind);
        try std.testing.expectEqual(a.index, b.index);
    }
    try std.testing.expectEqual(queues_a.instance_matrices.items.len, queues_b.instance_matrices.items.len);
    for (queues_a.instance_matrices.items, queues_b.instance_matrices.items) |a, b| {
        try std.testing.expectEqual(a, b);
    }
}

test "transparent instanced mesh sorts its instance matrices strictly back-to-front" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var mesh = Mesh{
        .name = "inst_sort_test",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 36,
        .index_type = .UINT16,
    };

    var trans_mat = StandardMaterial.init("trans_mat");
    trans_mat.alpha_mode = .blend;
    mesh.material = .{ .standard = &trans_mat };

    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &mesh, .position = Vec3.new(0, 0, 10) };
    var inst1 = InstancedMesh{ .name = "i1", .source_mesh = &mesh, .position = Vec3.new(0, 0, 30) };
    var inst2 = InstancedMesh{ .name = "i2", .source_mesh = &mesh, .position = Vec3.new(0, 0, 20) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1, &inst2 };
    mesh.instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 3 };

    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();

    const meshes = [_]*Mesh{&mesh};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 3), queues.instance_matrices.items.len);
    // Back-to-front: z=30 (farthest), then z=20, then z=10 (nearest)
    try std.testing.expectEqual(@as(f32, 30.0), queues.instance_matrices.items[0].m[14]);
    try std.testing.expectEqual(@as(f32, 20.0), queues.instance_matrices.items[1].m[14]);
    try std.testing.expectEqual(@as(f32, 10.0), queues.instance_matrices.items[2].m[14]);

    // Opaque instanced mesh preserves original instance creation order
    var opaque_mat = StandardMaterial.init("opaque_mat");
    opaque_mat.alpha_mode = .@"opaque";
    mesh.material = .{ .standard = &opaque_mat };

    queues.reset();
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 3), queues.instance_matrices.items.len);
    // Original insertion order: z=10, z=30, z=20
    try std.testing.expectEqual(@as(f32, 10.0), queues.instance_matrices.items[0].m[14]);
    try std.testing.expectEqual(@as(f32, 30.0), queues.instance_matrices.items[1].m[14]);
    try std.testing.expectEqual(@as(f32, 20.0), queues.instance_matrices.items[2].m[14]);
}

test "stageInstances stages visible count and combined AABB" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "stage_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "inst0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "inst1", .source_mesh = &src, .position = Vec3.new(5, 0, 0) };
    var inst2 = InstancedMesh{ .name = "inst2", .source_mesh = &src, .position = Vec3.new(10, 0, 0), .is_visible = false };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1, &inst2 };
    var parent = Mesh{
        .name = "stage_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 3 },
    };
    const meshes = [_]*Mesh{&parent};
    stageInstances(.{
        .allocator = ally,
        .queues = &queues,
        .thread_pool = null,
        .frame_id = 7,
        .eye = Vec3.zero,
    }, &meshes);

    // sg has no context in tests, so the upload half is skipped; the
    // transform/count staging must still run.
    try std.testing.expectEqual(@as(u32, 2), parent.visible_instance_count);
    try std.testing.expect(parent.cached_aabb.isValid());
    // inst0 covers [-1,1], inst1 covers [4,6]: the combined AABB spans both.
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), parent.cached_aabb.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), parent.cached_aabb.max.x, 1e-4);
}

test "stageInstances is idempotent within a frame" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "idem_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "inst0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "inst1", .source_mesh = &src, .position = Vec3.new(5, 0, 0) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1 };
    var parent = Mesh{
        .name = "idem_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 2 },
    };
    const meshes = [_]*Mesh{&parent};
    const sc = InstanceStageContext{
        .allocator = ally,
        .queues = &queues,
        .thread_pool = null,
        .frame_id = 7,
        .eye = Vec3.zero,
    };
    stageInstances(sc, &meshes);
    const first_aabb = parent.cached_aabb;
    const first_count = parent.visible_instance_count;

    // Mutating an instance after staging must not change this frame's
    // snapshot: the frame guard makes the second call a no-op.
    inst0.position = Vec3.new(100, 0, 0);
    stageInstances(sc, &meshes);
    try std.testing.expectEqual(first_count, parent.visible_instance_count);
    try std.testing.expectEqual(first_aabb, parent.cached_aabb);
}

test "stageInstances skips gpu_pending meshes" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "pend_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "inst0", .source_mesh = &src, .position = Vec3.zero };
    var ptrs = [_]*InstancedMesh{&inst0};
    var parent = Mesh{
        .name = "pend_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .gpu_pending = true,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 1 },
    };
    const meshes = [_]*Mesh{&parent};
    stageInstances(.{
        .allocator = ally,
        .queues = &queues,
        .thread_pool = null,
        .frame_id = 7,
        .eye = Vec3.zero,
    }, &meshes);

    try std.testing.expectEqual(@as(u32, 0), parent.visible_instance_count);
}

test "pre-staged instances feed buildFrameQueues batch" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "batch_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "inst0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "inst1", .source_mesh = &src, .position = Vec3.new(5, 0, 0) };
    var inst2 = InstancedMesh{ .name = "inst2", .source_mesh = &src, .position = Vec3.new(10, 0, 0), .is_visible = false };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1, &inst2 };
    var parent = Mesh{
        .name = "batch_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 3 },
    };
    const meshes = [_]*Mesh{&parent};
    stageInstances(.{
        .allocator = ally,
        .queues = &queues,
        .thread_pool = null,
        .frame_id = 9,
        .eye = Vec3.zero,
    }, &meshes);

    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .frame_id = 9,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 1), queues.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 2), queues.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(&parent, queues.opaque_instanced.items[0].mesh);
}
