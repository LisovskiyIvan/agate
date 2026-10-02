//! Instanced-mesh submit glue: per-frame instance staging plus instanced
//! queue fill. Imports `items` (batch payload, material classifiers) and
//! `cull` (`FrameCullContext`, material-record build) plus the existing
//! `instance_staging` producer — never the `render_queue.zig` facade and
//! never `build` (which imports this module for the serial submit and the
//! parallel merge tail; the reverse would be an import cycle). Documented
//! anti-cycle rule.
//!
//! Staging-contract tests (`stageInstances`/P5) live here rather than in
//! `instance_staging.zig`: they exercise staging through the
//! `RenderQueues.instance_matrices` scratch and the published render state
//! this module's submit path consumes, keeping the `render_queue/` split
//! self-contained and `instance_staging.zig` untouched. Tests that drive the
//! full `buildFrameQueues` entry point live in `build` (importing it here
//! would cycle `instances` ↔ `build`).
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
    // when running under Scene.prepareFrame; the frame guard makes this a
    // no-op then, while direct callers (tests, parallel merge tail) still
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

        const is_pbr = if (mesh.material) |m| (m == .pbr) else false;
        const is_trans = materialIsTransparent(mesh.material) or mesh.is_decal;
        const is_ds = materialIsDoubleSided(mesh.material);
        const draw_rec = buildMaterialRecord(ctx, mesh.material);

        // Coat-слот только для групп с включённым lobe (иначе null →
        // CoatParams.neutral на draw-фазе). OOM роняет всю группу, как
        // OOM очередей в appendRenderItem.
        var coat_index: ?u32 = null;
        if (material_mod.coatParamsFor(mesh.material)) |cp| {
            ctx.queues.coat_storage.ensureUnusedCapacity(ctx.allocator, 1) catch return;
            coat_index = @intCast(ctx.queues.coat_storage.items.len);
            ctx.queues.coat_storage.appendAssumeCapacity(cp);
        }

        const batch = RenderInstancedBatch{
            .vertex_buffer = mesh.vertex_buffer,
            .instance_buffer = staged.buffer,
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
        };

        if (is_trans) {
            // Reserve both slots up front (see appendRenderItem): the group
            // and its order entry are appended atomically under OOM.
            ctx.queues.transparent_instanced.ensureUnusedCapacity(ctx.allocator, 1) catch return;
            ctx.queues.transparent_order.ensureUnusedCapacity(ctx.allocator, 1) catch return;
            // Group distance key: combined staged bounds center (batch draws
            // as one; no per-instance sorting). Falls back to the mesh
            // position when the staged bounds are degenerate.
            const center = if (staged.bounds.isValid()) staged.bounds.center() else mesh.position;
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
        .cache_key = 1,
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
        .cache_key = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_p,
        .queues = &queues_p,
    };
    submitInstancedMesh(ctx_p, Frustum.fromViewProjection(Mat4.identity), &mesh_parallel, 0);

    // Verify bit-identical results, including the published render state
    // (bounds/count/frame — not pointer identity, and never a weaker
    // count-only check: 300 instances with every 7th hidden stage 257
    // visible instances, plus the source mesh entry = 258).
    try std.testing.expectEqual(mesh_serial.instance_render.bounds, mesh_parallel.instance_render.bounds);
    try std.testing.expectEqual(@as(u32, 258), mesh_serial.instance_render.count);
    try std.testing.expectEqual(@as(u32, 258), mesh_parallel.instance_render.count);
    try std.testing.expectEqual(@as(u64, 1), mesh_serial.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u64, 2), mesh_parallel.instance_render.staged_frame);
    try std.testing.expectEqual(queues_s.instance_matrices.items.len, queues_p.instance_matrices.items.len);
    try std.testing.expect(queues_s.instance_matrices.items.len > 0);
    for (queues_s.instance_matrices.items, queues_p.instance_matrices.items) |m_s, m_p| {
        try std.testing.expectEqual(m_s, m_p);
    }
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
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 7,
        .eye = Vec3.zero,
    }, &meshes);

    // sg has no context in tests, so the upload half is skipped; the
    // transform/count staging must still run into the published render state.
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expect(parent.instance_render.bounds.isValid());
    // inst0 covers [-1,1], inst1 covers [4,6]: the combined AABB spans both.
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), parent.instance_render.bounds.max.x, 1e-4);
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
    const sc = instance_staging.InstanceStageContext{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 7,
        .eye = Vec3.zero,
    };
    instance_staging.stageInstances(sc, &meshes);
    const first_bounds = parent.instance_render.bounds;
    const first_count = parent.instance_render.count;
    const first_frame = parent.instance_render.staged_frame;

    // Mutating an instance after staging must not change this frame's
    // snapshot: the frame guard makes the second call a no-op.
    inst0.position = Vec3.new(100, 0, 0);
    instance_staging.stageInstances(sc, &meshes);
    try std.testing.expectEqual(first_count, parent.instance_render.count);
    try std.testing.expectEqual(first_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(first_frame, parent.instance_render.staged_frame);
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
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 7,
        .eye = Vec3.zero,
    }, &meshes);

    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), parent.instance_render.staged_frame);
}

// ---- P5 instance staging ownership: регрессия владения и публикации. ----

// Staging is pure: it computes from read-only instance/source TRS,
// base_matrix and local_box and must NOT mutate the InstancedMesh game
// caches (cached_*/dirty/last_*) or the regular Mesh caches — those stay
// functional for picking and game APIs. Sentinels prove non-writes; exact
// staged numbers prove the read path (source base translation included).
test "P5: staging leaves source TRS and game caches untouched" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "pure_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .base_matrix = Mat4.translation(Vec3.new(10, 0, 0)),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
        .cached_matrix = Mat4.translation(Vec3.new(777, 0, 0)),
        .cached_aabb = BoundingBox.init(Vec3.new(777, 777, 777), Vec3.new(778, 778, 778)),
        .cached_frame = 4242,
    };
    var inst = InstancedMesh{
        .name = "pure_inst",
        .source_mesh = &src,
        .position = Vec3.new(3, 4, 5),
        .cached_world_matrix = Mat4.translation(Vec3.new(999, 0, 0)),
        .cached_bounding_box = BoundingBox.init(Vec3.new(999, 999, 999), Vec3.new(1000, 1000, 1000)),
        .last_position = Vec3.new(111, 0, 0),
        .last_rotation = Vec3.new(0, 222, 0),
        .last_scaling = Vec3.new(0, 0, 333),
        .dirty = true,
    };
    var ptrs = [_]*InstancedMesh{&inst};
    var parent = Mesh{
        .name = "pure_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .cached_matrix = Mat4.translation(Vec3.new(555, 0, 0)),
        .cached_aabb = BoundingBox.init(Vec3.new(555, 555, 555), Vec3.new(556, 556, 556)),
        .cached_frame = 8484,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 1 },
    };
    const meshes = [_]*Mesh{&parent};
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 11,
        .eye = Vec3.zero,
    }, &meshes);

    // Published state: count 2 (source mesh entry + one instance), staged
    // frame advanced, bounds read through the source base translation: unit
    // box at (3,4,5)+(10,0,0) = [12,14]x[3,5]x[4,6].
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 11), parent.instance_render.staged_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), parent.instance_render.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), parent.instance_render.bounds.min.y, 1e-4);
    // Scratch carries the source entry first (its own identity transform),
    // then the instance's world translation (13,4,5).
    try std.testing.expectEqual(@as(usize, 2), queues.instance_matrices.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), queues.instance_matrices.items[0].m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 13.0), queues.instance_matrices.items[1].m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), queues.instance_matrices.items[1].m[13], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), queues.instance_matrices.items[1].m[14], 1e-4);

    // Game caches: every sentinel bit-identical, dirty flag untouched.
    try std.testing.expectApproxEqAbs(@as(f32, 999.0), inst.cached_world_matrix.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 999.0), inst.cached_bounding_box.min.x, 1e-6);
    try std.testing.expectEqual(Vec3.new(111, 0, 0), inst.last_position);
    try std.testing.expectEqual(Vec3.new(0, 222, 0), inst.last_rotation);
    try std.testing.expectEqual(Vec3.new(0, 0, 333), inst.last_scaling);
    try std.testing.expect(inst.dirty);
    // Regular mesh caches (source template and instance parent): untouched.
    try std.testing.expectApproxEqAbs(@as(f32, 777.0), src.cached_matrix.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 777.0), src.cached_aabb.min.x, 1e-6);
    try std.testing.expectEqual(@as(u64, 4242), src.cached_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 555.0), parent.cached_matrix.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 555.0), parent.cached_aabb.min.x, 1e-6);
    try std.testing.expectEqual(@as(u64, 8484), parent.cached_frame);
}

// A completed stage stays once-per-frame even when a second view (different
// eye) restages: no recompute, no re-sort, same scratch bytes. The next
// frame picks the mutation up.
test "P5: staged frame survives a second eye; next frame restages" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "eye_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "e0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "e1", .source_mesh = &src, .position = Vec3.new(8, 0, 0) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1 };
    var parent = Mesh{
        .name = "eye_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 2 },
    };
    const meshes = [_]*Mesh{&parent};

    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 21,
        .eye = Vec3.new(-50, 0, 0),
    }, &meshes);
    const first_scratch = try ally.dupe(Mat4, queues.instance_matrices.items);
    defer ally.free(first_scratch);
    const first_bounds = parent.instance_render.bounds;

    // Same frame, other eye, mutated TRS: must be a no-op (shadow + N
    // cameras share the frame's publish; no same-frame retry).
    inst0.position = Vec3.new(100, 0, 0);
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 21,
        .eye = Vec3.new(50, 0, 0),
    }, &meshes);
    try std.testing.expectEqual(@as(u64, 21), parent.instance_render.staged_frame);
    try std.testing.expectEqual(first_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(first_scratch.len, queues.instance_matrices.items.len);
    for (first_scratch, queues.instance_matrices.items) |a, b| {
        try std.testing.expectEqual(a, b);
    }

    // Next frame: the mutation lands (inst0 now covers [99,101]).
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 22,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u64, 22), parent.instance_render.staged_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 101.0), parent.instance_render.bounds.max.x, 1e-4);
}

// Visible-set transitions publish coherent bounds/count every frame, and the
// empty set publishes a consistent empty state (count 0, invalid bounds)
// that recovers when instances return.
test "P5: visible transitions and the empty path stay coherent" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "trans_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "t0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "t1", .source_mesh = &src, .position = Vec3.new(20, 0, 0) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1 };
    var parent = Mesh{
        .name = "trans_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 2 },
    };
    const meshes = [_]*Mesh{&parent};
    const stage = struct {
        fn run(
            allocator: std.mem.Allocator,
            matrices: *std.ArrayListUnmanaged(Mat4),
            frame: u64,
            list: []const *Mesh,
        ) void {
            instance_staging.stageInstances(.{
                .allocator = allocator,
                .instance_matrices = matrices,
                .thread_pool = null,
                .frame_id = frame,
                .eye = Vec3.zero,
            }, list);
        }
    }.run;

    stage(ally, &queues.instance_matrices, 31, &meshes);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 21.0), parent.instance_render.bounds.max.x, 1e-4);

    inst1.is_visible = false;
    stage(ally, &queues.instance_matrices, 32, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 32), parent.instance_render.staged_frame);
    // Only inst0's box remains: a count-only check would miss stale bounds.
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), parent.instance_render.bounds.max.x, 1e-4);

    inst0.is_visible = false;
    // The empty batch needs the source mesh hidden too: Babylon draws the
    // source mesh at its own transform independently of its instances, so a
    // visible source keeps the batch non-empty (one matrix).
    parent.is_visible = false;
    // GPU-side sentinels (no GPU context here, so the upload half is skipped
    // and these must survive untouched): a live GPU probe covers the same
    // preservation against real buffers separately.
    parent.instance_render.buffer = .{ .id = 123 };
    parent.instance_render.capacity = 7;
    parent.instance_render.hash = 0xABCD;
    parent.instance_render.uploaded_count = 5;
    stage(ally, &queues.instance_matrices, 33, &meshes);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 33), parent.instance_render.staged_frame);
    try std.testing.expect(!parent.instance_render.bounds.isValid());
    // The empty publish touches count/bounds/frame only: no mixing of new
    // emptiness with old GPU identity.
    try std.testing.expectEqual(@as(u32, 123), parent.instance_render.buffer.id);
    try std.testing.expectEqual(@as(usize, 7), parent.instance_render.capacity);
    try std.testing.expectEqual(@as(u64, 0xABCD), parent.instance_render.hash);
    try std.testing.expectEqual(@as(usize, 5), parent.instance_render.uploaded_count);

    inst0.is_visible = true;
    inst1.is_visible = true;
    parent.is_visible = true;
    stage(ally, &queues.instance_matrices, 34, &meshes);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expect(parent.instance_render.bounds.isValid());
    try std.testing.expectApproxEqAbs(@as(f32, 21.0), parent.instance_render.bounds.max.x, 1e-4);
}

// Scratch OOM publishes nothing: the previous complete state stays in place
// (no mixed new-bounds/old-count) and the guard stays back so a later call
// with a working allocator retries successfully.
test "P5: scratch OOM keeps the previous publish and allows retry" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "oom_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "o0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var ptrs = [_]*InstancedMesh{&inst0};
    var parent = Mesh{
        .name = "oom_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 1 },
    };
    const meshes = [_]*Mesh{&parent};

    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 41,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);

    // Fresh scratch (no retained capacity) + failing allocator: the first
    // append/resize fails. Mutation is staged for the retry below.
    var bare = RenderQueues{};
    defer bare.deinit(ally);
    inst0.position = Vec3.new(30, 0, 0);
    var failing = std.testing.FailingAllocator.init(ally, .{ .fail_index = 0 });
    instance_staging.stageInstances(.{
        .allocator = failing.allocator(),
        .instance_matrices = &bare.instance_matrices,
        .thread_pool = null,
        .frame_id = 42,
        .eye = Vec3.zero,
    }, &meshes);
    // Previous complete publish intact, guard not advanced.
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 41), parent.instance_render.staged_frame);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), parent.instance_render.bounds.min.x, 1e-4);

    // Retry with a working allocator lands the mutation (box [29,31]).
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &bare.instance_matrices,
        .thread_pool = null,
        .frame_id = 42,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u64, 42), parent.instance_render.staged_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 29.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), parent.instance_render.bounds.max.x, 1e-4);
}

// Without a GPU context there is no upload: hash/uploaded_count must keep
// describing "no upload yet" (zeros), so the dedup gate cannot mistake a
// CPU-only publish for uploaded data when a context appears later.
test "P5: CPU-only staging publishes no upload identity" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "noupload_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "n0", .source_mesh = &src, .position = Vec3.zero };
    var ptrs = [_]*InstancedMesh{&inst0};
    var parent = Mesh{
        .name = "noupload_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 1 },
    };
    const meshes = [_]*Mesh{&parent};
    const ctx = instance_staging.InstanceStageContext{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 51,
        .eye = Vec3.zero,
    };
    instance_staging.stageInstances(ctx, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 51), parent.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u64, 0), parent.instance_render.hash);
    try std.testing.expectEqual(@as(usize, 0), parent.instance_render.uploaded_count);

    // Second identical frame: still no upload identity invented.
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 52,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u64, 0), parent.instance_render.hash);
    try std.testing.expectEqual(@as(usize, 0), parent.instance_render.uploaded_count);
}
