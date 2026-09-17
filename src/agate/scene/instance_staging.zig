const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../mesh.zig").Mesh;
const InstancedMesh = @import("../mesh.zig").InstancedMesh;
const material_mod = @import("../material.zig");
const Material = material_mod.Material;
const jobs = @import("../jobs.zig");
const upload_meter = @import("../gpu_upload_meter.zig");

/// Mirrors render_queue.materialIsTransparent: a mesh is transparent when its
/// material opts into .blend alpha mode. Kept local so this module never
/// imports render_queue.zig (no import cycle with its staging caller).
fn isTransparentMaterial(mat: ?Material) bool {
    if (mat) |m| return m.isTransparent();
    return false;
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
/// FrameCullContext (see render_queue.zig). Scene.prepareFrame pre-stages
/// through this before the shadow pass so ShadowPass.prepare snapshots the
/// same frame's cached_aabb / instance_buffer / visible_instance_count
/// instead of the previous frame's. The `instance_uploaded_frame != frame_id`
/// guard keeps staging once per frame, shared by all view queues.
///
/// Carries the staging scratch list (`RenderQueues.instance_matrices`)
/// directly instead of `*RenderQueues`, so this module never imports
/// render_queue.zig and no import cycle can form.
pub const InstanceStageContext = struct {
    allocator: std.mem.Allocator,
    instance_matrices: *std.ArrayListUnmanaged(Mat4),
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
        sc.instance_matrices.clearRetainingCapacity();

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

            sc.instance_matrices.resize(sc.allocator, total_visible) catch return;
            if (total_visible > 0) {
                stage.out_matrices = sc.instance_matrices.items;
                p.forkJoin(ParallelInstanceStage, &stage, ParallelInstanceStage.scatterMatrices, chunk_count);
            }
        } else {
            var combined_aabb = math.BoundingBox.zero;
            for (mesh.instances.items) |inst| {
                if (!inst.is_visible) continue;
                inst.updateCachedTransforms();
                sc.instance_matrices.append(sc.allocator, inst.cached_world_matrix) catch return;
                if (combined_aabb.isValid()) {
                    combined_aabb = combined_aabb.merge(inst.cached_bounding_box);
                } else {
                    combined_aabb = inst.cached_bounding_box;
                }
            }
            mesh.cached_aabb = combined_aabb;
        }
        const active_count = sc.instance_matrices.items.len;
        mesh.visible_instance_count = @intCast(active_count);

        // Per-instance transparency sorting (OIT):
        // When the instanced mesh is transparent or a decal, sort its instance
        // matrices back-to-front relative to the camera eye. Farthest instances
        // render first, blending nearer instances over them correctly.
        if (active_count > 1 and (isTransparentMaterial(mesh.material) or mesh.is_decal)) {
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
            std.mem.sort(Mat4, sc.instance_matrices.items[0..active_count], SortCtx{ .eye = sc.eye }, SortCtx.sortFn);
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
                sg.updateBuffer(mesh.instance_buffer, sg.asRange(sc.instance_matrices.items[0..active_count]));
                // Учёт динамики: active_count матриц Mat4 (потокобезопасно — счётчик атомарный).
                upload_meter.record(active_count * @sizeOf(Mat4));
                mesh.instance_hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sc.instance_matrices.items[0..active_count]));
                mesh.instance_uploaded_count = active_count;
            } else {
                const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sc.instance_matrices.items[0..active_count]));
                if (active_count != mesh.instance_uploaded_count or h != mesh.instance_hash) {
                    sg.updateBuffer(mesh.instance_buffer, sg.asRange(sc.instance_matrices.items[0..active_count]));
                    // Учёт динамики: только при реальном изменении (dedup по хешу выше).
                    upload_meter.record(active_count * @sizeOf(Mat4));
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
/// view queues and again implicitly via submitInstancedMesh (render_queue.zig)
/// stays once-only.
pub fn stageInstances(sc: InstanceStageContext, meshes: []const *Mesh) void {
    for (meshes) |mesh| {
        if (mesh.is_lod_child) continue;
        if (mesh.gpu_pending) continue;
        if (mesh.instances.items.len == 0) continue;
        stageInstancedMesh(sc, mesh);
    }
}
