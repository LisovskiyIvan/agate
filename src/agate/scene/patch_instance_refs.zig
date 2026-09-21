const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;

const StagedInstanceRecord = @import("../mesh.zig").StagedInstanceRecord;
const scene_render_queue = @import("render_queue.zig");
const FrameDrawSlot = @import("frame_draws.zig").FrameDrawSlot;

fn findStagedRecord(records: []const StagedInstanceRecord, idx: usize) ?*const StagedInstanceRecord {
    for (records) |*rec| {
        const ri: usize = rec.mesh_index;
        if (ri == idx) return rec;
        if (ri > idx) break;
    }
    return null;
}

fn patchBatchList(
    list: *std.ArrayListUnmanaged(scene_render_queue.RenderInstancedBatch),
    records: []const StagedInstanceRecord,
    fid: u64,
    is_gpu: bool,
) void {
    for (list.items) |*b| {
        if (is_gpu and (b.vertex_buffer.id == 0 or sg.queryBufferState(b.vertex_buffer) != .VALID)) {
            b.instance_buffer = .{};
            b.visible_instance_count = 0;
            continue;
        }
        const rec = findStagedRecord(records, b.source_mesh) orelse {
            b.instance_buffer = .{};
            b.visible_instance_count = 0;
            continue;
        };
        if (rec.uid != b.source_uid or rec.staged_frame != fid) {
            b.instance_buffer = .{};
            b.visible_instance_count = 0;
            continue;
        }
        b.instance_buffer = rec.buffer;
        b.visible_instance_count = rec.count;
    }
}

/// Latch patch finalizing game-built provisional handles (stage-2
/// increment B, context side, allocation-free): after
/// `stageInstancesLatch` publishes, every instanced payload entry built
/// with `.build_view` is re-resolved by identity (`source_mesh` index +
/// `source_uid` validation) against the SLOT-OWNED staged records — never
/// against the live mesh list.
pub fn patchInstanceRefs(back: *FrameDrawSlot) void {
    const records = back.staged_instances.items;
    const fid = back.frame_id;
    const is_gpu = sg.isvalid();
    // Queue batches: primary + all views, opaque + transparent.
    const queue_lists = [_]*std.ArrayListUnmanaged(scene_render_queue.RenderInstancedBatch){
        &back.primary.opaque_instanced, &back.primary.transparent_instanced,
    };
    for (queue_lists) |list| patchBatchList(list, records, fid, is_gpu);
    for (&back.views) |*q| {
        patchBatchList(&q.opaque_instanced, records, fid, is_gpu);
        patchBatchList(&q.transparent_instanced, records, fid, is_gpu);
    }
    // Shadow items.
    for (back.shadow.items.items) |*it| {
        if (is_gpu and (it.vertex_buffer.id == 0 or sg.queryBufferState(it.vertex_buffer) != .VALID)) {
            it.is_visible = false;
            it.gpu_pending = true;
            it.instance_buffer = .{};
            it.visible_instance_count = 0;
            it.world_aabb = BoundingBox.zero;
            it.max_dim = 0;
            continue;
        }
        if (!it.is_instanced) continue;
        const rec = findStagedRecord(records, it.source_mesh) orelse {
            it.instance_buffer = .{};
            it.visible_instance_count = 0;
            it.world_aabb = BoundingBox.zero;
            it.max_dim = 0;
            continue;
        };
        if (rec.uid != it.source_uid or rec.staged_frame != fid) {
            it.instance_buffer = .{};
            it.visible_instance_count = 0;
            it.world_aabb = BoundingBox.zero;
            it.max_dim = 0;
            continue;
        }
        it.instance_buffer = rec.buffer;
        it.visible_instance_count = rec.count;
        it.world_aabb = rec.bounds;
        const ext = rec.bounds.extents();
        it.max_dim = @max(ext.x, @max(ext.y, ext.z));
    }
    // Outline items (instanced only).
    for (back.outline_items.items) |*it| {
        if (is_gpu and (it.vertex_buffer.id == 0 or sg.queryBufferState(it.vertex_buffer) != .VALID)) {
            it.instance_buffer = .{};
            it.visible_instance_count = 0;
            it.world_center = Vec3.zero;
            continue;
        }
        if (!it.is_instanced) continue;
        const rec = findStagedRecord(records, it.source_mesh) orelse {
            it.instance_buffer = .{};
            it.visible_instance_count = 0;
            it.world_center = Vec3.zero;
            continue;
        };
        if (rec.uid != it.source_uid or rec.staged_frame != fid) {
            it.instance_buffer = .{};
            it.visible_instance_count = 0;
            it.world_center = rec.mesh_position;
            continue;
        }
        it.instance_buffer = rec.buffer;
        it.visible_instance_count = rec.count;
        it.world_center = if (rec.bounds.isValid()) rec.bounds.center() else rec.mesh_position;
    }
}
