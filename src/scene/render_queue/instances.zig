//! Instanced-mesh submit glue: per-frame instance staging plus instanced queue fill.
const std = @import("std");

const math = @import("math");
const Mat4 = math.Mat4;
const Frustum = math.Frustum;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../../mesh.zig").Mesh;
const InstancedMesh = @import("../../mesh.zig").InstancedMesh;
const material_mod = @import("../../material.zig");
const jobs = @import("../../jobs.zig");
const visibility = @import("../../visibility/mod.zig");
const stats_mod = @import("../stats.zig");
const SceneStats = stats_mod.SceneStats;
const instance_staging = @import("../instance_staging.zig");
const items = @import("items.zig");
const RenderQueues = items.RenderQueues;
const RenderInstancedBatch = items.RenderInstancedBatch;
const materialIsTransparent = items.materialIsTransparent;
const materialIsDoubleSided = items.materialIsDoubleSided;
const cull = @import("cull.zig");
const FrameCullContext = cull.FrameCullContext;
const buildMaterialRecord = cull.buildMaterialRecord;

/// Instance-bearing mesh handling: per-frame instance-matrix staging (parallel
/// when attached to a pool and exceeding parallel_min_instances), sg buffer
/// (re)creation/upload on the context thread, frustum test on the combined AABB,
/// instanced queue fill.
///
/// Internal to `render_queue/*` (used by `build`'s serial loop and parallel
/// merge tail); not re-exported by the facade.
pub fn submitInstancedMesh(ctx: FrameCullContext, frustum: Frustum, mesh: *Mesh, mesh_index: usize) void {
    // Deferred-creation meshes have no vertex/index buffers yet; staging
    // instance data for them would produce a draw against invalid handles.
    // Checked here (not only inside staging) so a definitive pre-stage
    // (instances_prepared) still skips them in the view builds.
    if (mesh.gpu_pending) return;
    // Pre-staged by instance_staging.stageInstances before the shadow pass
    // in the producer build; the frame guard makes this a
    // no-op then, while standalone immediate callers (tests, parallel merge tail) still
    // stage here. With instances_prepared the pre-stage publish is
    // definitive: consume it as-is, never retry mid-frame. LOD children
    // stay on the per-mesh guard (pre-stage and shadow both skip them, so
    // the views still share a single guard-stage, as before).
    if (!ctx.instances_prepared or mesh.is_lod_child) {
        instance_staging.stageInstancedMesh(.{
            .allocator = ctx.allocator,
            .instance_matrices = &ctx.queues.instance_matrices,
            .thread_pool = ctx.thread_pool,
            .frame_id = ctx.cache_key,
            .eye = ctx.eye,
            .retire_queue = ctx.gpu_retire,
        }, mesh);
    }

    const staged = mesh.instanceRenderSource(ctx.instance_source).*;
    if (staged.count > 0 and ((mesh.layer_mask & ctx.culling_mask) != 0)) {
        if (ctx.cull_frustum and staged.bounds.isValid() and !frustum.intersectsAABB(staged.bounds)) {
            ctx.stats.culled_meshes += @intCast(mesh.instances.items.len);
            return;
        }
        ctx.stats.total_meshes += @intCast(mesh.instances.items.len);
        ctx.stats.rendered_meshes += staged.count;

        const is_pbr = if (mesh.material) |m| (m == .pbr) else true;
        const is_trans = materialIsTransparent(mesh.material) or mesh.is_decal;
        const is_ds = materialIsDoubleSided(mesh.material);
        const draw_rec = buildMaterialRecord(ctx, mesh.material);

        // Coat slot allocated only for groups with active lobes (otherwise null -> CoatParams.neutral).
        var coat_index: ?u32 = null;
        if (material_mod.coatParamsFor(mesh.material)) |cp| {
            ctx.queues.coat_storage.ensureUnusedCapacity(ctx.allocator, 1) catch {
                ctx.stats.build_oom_drops += 1;
                return;
            };
            coat_index = @intCast(ctx.queues.coat_storage.items.len);
            ctx.queues.coat_storage.appendAssumeCapacity(cp);
        }

        const center = if (staged.bounds.isValid()) staged.bounds.center() else mesh.position;
        const batch = RenderInstancedBatch{
            .vertex_buffer = mesh.vertex_buffer,
            .instance_buffer = staged.buffer,
            .prev_instance_buffer = staged.prev_buffer,
            .prev_frame = staged.prev_frame,
            .index_buffer = mesh.index_buffer,
            .index_count = mesh.index_count,
            .index_type = mesh.index_type,
            .visible_instance_count = staged.count,
            .is_pbr = is_pbr,
            .transparent = is_trans,
            .double_sided = is_ds,
            .is_decal = mesh.is_decal,
            .receive_shadows = mesh.receive_shadows,
            .draw_record = draw_rec,
            .source_uid = mesh.uid,
            .source_mesh = @intCast(mesh_index),
            .coat_index = coat_index,
            .world_center = center,
        };

        if (is_trans) {
            // Reserve both slots up front (see appendRenderItem): the group
            // and its order entry are appended atomically under OOM.
            ctx.queues.transparent_instanced.ensureUnusedCapacity(ctx.allocator, 1) catch {
                ctx.stats.build_oom_drops += 1;
                return;
            };
            ctx.queues.transparent_order.ensureUnusedCapacity(ctx.allocator, 1) catch {
                ctx.stats.build_oom_drops += 1;
                return;
            };
            // Group distance key: combined staged bounds center (batch draws
            // as one; no per-instance sorting). Falls back to the mesh
            // position when the staged bounds are degenerate.
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
            ctx.queues.opaque_instanced.append(ctx.allocator, batch) catch {
                ctx.stats.build_oom_drops += 1;
            };
        }
    }
}
