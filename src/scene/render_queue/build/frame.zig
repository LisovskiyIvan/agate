//! Serial frame-queue entry: `buildFrameQueues` (occluder rasterization,
//! parallel dispatch with OOM fallback to the serial cull loop). Imports the
//! `items`, `cull`, and `instances` leaves plus the `parallel` sibling for
//! the chunked cull pass — never the `build.zig` facade (documented
//! anti-cycle rule). Re-exported through the facade; `equivalence` and
//! `snapshots` import this module for their integration tests.
const std = @import("std");

const math = @import("math");
const Mat4 = math.Mat4;
const Frustum = math.Frustum;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../../../mesh.zig").Mesh;
const InstancedMesh = @import("../../../mesh.zig").InstancedMesh;
const cull = @import("../cull.zig");
const FrameCullContext = cull.FrameCullContext;
const worldMatrixCached = cull.worldMatrixCached;
const appendRenderItem = cull.appendRenderItem;
const cullNonInstancedMesh = cull.cullNonInstancedMesh;
const instances = @import("../instances.zig");
const submitInstancedMesh = instances.submitInstancedMesh;
const parallel = @import("parallel.zig");
const buildFrameQueuesParallel = parallel.buildFrameQueuesParallel;

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

    // Stage-2A identity: assign render uids serially before any parallel
    // read (binning/culling workers only read). Idempotent, no behavior change.
    for (ctx.meshes) |m| _ = m.ensureUid();

    // Phase -1: Occlusion Culling setup & occluder rasterization (serial:
    // the Hi-Z rasterizer is stateful).
    if (ctx.cull_occlusion) {
        ctx.occlusion_culler.beginFrame(ctx.view_proj);
        for (ctx.meshes) |m| {
            if (!m.is_lod_child and m.is_visible and m.is_occluder and ((m.layer_mask & ctx.culling_mask) != 0)) {
                const m_world = worldMatrixCached(ctx.cache_key, m);
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
        if (cullNonInstancedMesh(ctx, frustum, eye, mesh, ctx.stats, mesh_index)) |culled| {
            appendRenderItem(ctx, culled);
        }
    }
}
