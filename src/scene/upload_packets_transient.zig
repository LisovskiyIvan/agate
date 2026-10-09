//! Transient-buffer rewrite for reuse frames (no staged prepare ran).
//! Section of the `upload_packets` facade (pure move, see
//! `upload_packets.zig` for the full stage/flush/commit contract).

const sokol = @import("sokol");
const sg = sokol.gfx;

/// One write-only re-delivery into a bound transient buffer. The buffer
/// must still exist (retire flush may have destroyed it after the freeze),
/// must carry `write_transient`, and must have payload bytes this frame.
fn rewriteTransientInto(buffer_id: u32, payload: sg.Range) void {
    if (buffer_id == 0 or payload.size == 0) return;
    const buf = sg.Buffer{ .id = buffer_id };
    if (sg.queryBufferState(buf) != .VALID) return;
    if (!sg.queryBufferUsage(buf).write_transient) return;
    sg.writeBufferTransient(.{
        .dst = .{ .buffer = buf },
        .src = .{ .data = payload },
    });
}

/// RE-PRESENTS the staged payloads of a reused front slot into their
/// transient buffers. `renderReuse` redraws the frozen front WITHOUT a
/// staged prepare, so no flush ran in this sokol frame — yet sokol allows
/// at most one `sg_write_buffer_transient` per buffer per frame and
/// VALIDATES that every bound `write_transient` buffer was written this
/// frame (`VALIDATE_DRAW_WRITE_BUFFER_TRANSIENT_MISSING`, which escalated
/// to a hard panic in validation builds once the reuse probe ran).
///
/// Write-only by contract: no upload-meter accounting, no
/// delivered/outcome mutation, no buffer creation, no particle-compute
/// state clear (that clear is semantic — replaying it would wipe
/// simulation state). Contents are the frozen payloads, so the replayed
/// frame shows exactly the geometry the slot froze. Not covered (known,
/// see docs/frame-pipeline.md): pending-mesh creations (their ids were
/// consumed by the commit) and compute-particle state buffers (written on
/// the render-time compute path, which a reuse frame re-runs).
pub fn rewriteTransientWrites(scene: anytype, slot: anytype) void {
    _ = scene;
    if (!sg.isvalid()) return;
    for (slot.morph_uploads.items) |up| {
        const end = up.data_lo + up.count;
        if (end > slot.morph_data.items.len) continue;
        rewriteTransientInto(up.buffer_id, sg.asRange(slot.morph_data.items[up.data_lo..end]));
    }
    for (slot.p_cpu_uploads.items) |up| {
        const end = up.data_lo + up.count;
        if (end > slot.p_cpu_data.items.len) continue;
        rewriteTransientInto(up.buffer_id, sg.asRange(slot.p_cpu_data.items[up.data_lo..end]));
    }
    for (slot.p_gpu_uploads.items) |up| {
        const end = up.data_lo + up.count;
        if (end > slot.p_gpu_data.items.len) continue;
        rewriteTransientInto(up.buffer_id, sg.asRange(slot.p_gpu_data.items[up.data_lo..end]));
    }
    for (slot.trail_uploads.items) |up| {
        const v_end = up.vert_lo + up.vert_count;
        if (v_end > slot.trail_verts.items.len) continue;
        rewriteTransientInto(up.vertex_buffer_id, sg.asRange(slot.trail_verts.items[up.vert_lo..v_end]));
        const i_end = up.index_lo + up.index_count;
        if (i_end > slot.trail_indices.items.len) continue;
        rewriteTransientInto(up.index_buffer_id, sg.asRange(slot.trail_indices.items[up.index_lo..i_end]));
    }
    for (slot.soft_uploads.items) |up| {
        const end = up.data_lo + up.vert_count;
        if (end > slot.soft_data.items.len) continue;
        rewriteTransientInto(up.vertex_buffer_id, sg.asRange(slot.soft_data.items[up.data_lo..end]));
    }
    for (slot.greased_uploads.items) |up| {
        const v_end = up.vert_lo + up.vert_count;
        if (v_end > slot.greased_verts.items.len) continue;
        rewriteTransientInto(up.vertex_buffer_id, sg.asRange(slot.greased_verts.items[up.vert_lo..v_end]));
        const i_end = up.index_lo + up.index_count;
        if (i_end > slot.greased_indices.items.len) continue;
        rewriteTransientInto(up.index_buffer_id, sg.asRange(slot.greased_indices.items[up.index_lo..i_end]));
    }
}
