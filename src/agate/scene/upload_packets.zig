//! Slot-owned dynamic-upload packets (producer freeze-then-latch slice 6,
//! phase 2 lock-free publication).
//!
//! Producer side (`stageUploads`, game/update phase, sg-free): copies every
//! per-frame GPU staging payload into the claimed `FrameDrawSlot` by value
//! (morph vertex bytes, particle CPU/GPU/compute staging bytes, trail verts
//! + indices, softbody verts + indices, greased verts + indices,
//! pending-mesh creation geometry) plus frozen buffer ids / counts / bounds
//! / allocation sizes. Empty-but-dirty owners freeze empty packets so the
//! staged flush observes them exactly like the legacy flush. The producer
//! clears each live dirty flag AT STAGE TIME (when it freezes the packet);
//! a cancelled claim must not consume the upload, so `BuildClaim.cancel`
//! re-arms every flag staged into the dropped slot via
//! `restageDroppedSlot` (game-side, token-validated).
//!
//! Context side (`flushSlotUploads`, beginPrepare with a fresh build):
//! uploads from the slot packets and creates deferred buffers from FROZEN
//! sizes, then records per-packet outcomes IN THE SLOT DESCRIPTORS
//! (`delivered` + created handles + consumed window). THE CONTEXT NEVER
//! WRITES GAME-OWNED STATE on this path: no live flag clears, no live
//! scalar publishes, no live handle installs, no live array frees, no live
//! staging memcpy. No live staging BYTES drive uploads (the compute
//! frozen-window memcpy reinstall is DELETED; upload is direct from packet
//! bytes with frozen counts only) and no live mutable lengths are read
//! (allocation sizes ride frozen in the packets; `capacity` fields are
//! immutable after system init — documented at each freeze site).
//!
//! Game side (`commitSlotResults`, next build, under the game lock, front
//! slot under a read lease): validates each packet outcome by
//! index/token(/uid) and applies it — installs created handles (a replaced
//! non-zero live handle retires through the thread-safe queue; in practice
//! the live id is always zero here), publishes scalars (trail
//! index_count/bounds, pending-mesh vertex_count/morph flag, compute ring
//! advance + dt consume), frees consumed pending arrays, and re-arms flags
//! for every undelivered packet so the next funded build retries. Applied
//! once per published frame (`last_upload_commit_frame` vs
//! `front.frame_id` in the build core — a repeat build without an
//! intervening latch skips, so outcomes are never double-applied).
//!
//! Memory ordering: the context writes outcomes into its CLAIMED slot
//! (private until publish); the publish release edge (`tryPublish` /
//! `releaseHandoffWithSeq` under the lease mutex) pairs with the
//! producer's `pinFrontReader` acquire (same mutex) before the commit
//! reads them. Payload reads/writes on distinct slots need no further
//! locking (single producer; see frame_draws.zig).
//!
//! Deferred morph-delta textures (`morph_upload_pending`, GPU-mode only)
//! ride the pending-mesh packet: the producer packs the RGBA32F delta
//! pixels into `pending_delta_data` at stage time (write-once), the context
//! creates the delta image + view from those frozen bytes alongside the
//! base buffers, and the commit installs everything atomically — the mesh
//! leaves `gpu_pending` fully drawable (no base-pose frame). The legacy
//! `finishGpuUpload` still owns meshes that never went through a staged
//! build (no-build fallback path).
//!
//! The legacy `flushPendingGpuUploads` stays for the no-build fallback
//! (serialized by contract) and is never called on the fresh-build path.
//!
//! Identity: descriptors carry `token` (@intFromPtr of the live owner) +
//! list index (+ uid for meshes). The COMMIT validates token/index (/uid)
//! with a pointer compare first (never dereferencing a stale pointer):
//! mismatch skips fail-closed (previous complete GPU state stands, the
//! created handles retire through the queue, retry next build). Mesh-list
//! or registry mutation between build and latch violates the app contract;
//! the guard keeps it coherent, never corrupt.
//!
//! OOM: any packet that fails to stage is skipped like an OOM-skipped
//! instance segment (no partial packet: data truncated back, descriptor
//! unwritten). The stage-time flag clear is then uncovered — the next
//! build re-freezes from the intact live arrays because the mutation that
//! set the flag is still staged live... EXCEPT a flag cleared with no
//! packet and no newer mutation would be lost. To close that hole the
//! stage clears a flag ONLY when its packet is successfully frozen (clear
//! after append, per owner); an OOM-skipped owner keeps its flag set.
//! Undeliverable packets (creation OOM, headless, unsupported backend)
//! record `delivered = false` and the commit re-arms the flags, so the
//! next funded build retries from the intact live arrays.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const morph_gpu = @import("../mesh/morph_gpu.zig");

/// Producer freeze (game side, sg-free). Copies every dirty staging payload
/// into `slot`; never touches sg.*, never clears live flags, never reads the
/// consumed render snapshot. OOM per owner fail-closes to "no packet".
pub fn stageUploads(scene: anytype, slot: anytype) void {
    const allocator = scene.allocator;
    stageMorphs(scene, slot, allocator);
    stageParticleCpu(scene, slot, allocator);
    stageParticleGpu(scene, slot, allocator);
    stageParticleCompute(scene, slot, allocator);
    stageTrails(scene, slot, allocator);
    stageSoftbodies(scene, slot, allocator);
    stageGreased(scene, slot, allocator);
    stagePendingMeshes(scene, slot, allocator);
}

fn stageMorphs(scene: anytype, slot: anytype, allocator: std.mem.Allocator) void {
    for (scene.meshes.items, 0..) |m, i| {
        if (!m.morph_upload_needed) continue;
        const n = @min(m.morph_base.len, m.morph_staging.len);
        // Empty staging still freezes an empty packet: the legacy flush
        // clears the flag unconditionally, so the staged flush must observe
        // it too instead of leaving a stale flag behind.
        const data_lo = slot.morph_data.items.len;
        if (n > 0) slot.morph_data.appendSlice(allocator, m.morph_staging[0..n]) catch continue;
        slot.morph_uploads.append(allocator, .{
            .token = @intFromPtr(m),
            .uid = m.uid,
            .mesh_index = @intCast(i),
            .buffer_id = m.vertex_buffer.id,
            .count = @intCast(n),
            .data_lo = data_lo,
        }) catch {
            slot.morph_data.items.len = data_lo;
            continue;
        };
        // Flag consumed AT STAGE TIME (phase 2 ownership transfer): the
        // context never writes it. Cleared only on a successfully frozen
        // packet (OOM above keeps the flag for the next build); a
        // cancelled claim re-arms via restageDroppedSlot.
        m.morph_upload_needed = false;
    }
}

fn stageParticleCpu(scene: anytype, slot: anytype, allocator: std.mem.Allocator) void {
    for (scene.particles.systems.items, 0..) |ps, i| {
        if (!ps.instance_dirty) continue;
        const count: usize = ps.active_count;
        if (count > ps.instances.len) continue;
        // Empty-but-dirty systems freeze an empty packet so the staged flush
        // consumes the flag without ever reading live state on the latch.
        const data_lo = slot.p_cpu_data.items.len;
        if (count > 0) slot.p_cpu_data.appendSlice(allocator, ps.instances[0..count]) catch continue;
        slot.p_cpu_uploads.append(allocator, .{
            .token = @intFromPtr(ps),
            .sys_index = @intCast(i),
            .buffer_id = ps.instance_buffer.id,
            .count = @intCast(count),
            .data_lo = data_lo,
            .capacity = ps.capacity,
        }) catch {
            slot.p_cpu_data.items.len = data_lo;
            continue;
        };
        // Consumed at stage time (see stageMorphs). The creation-pending
        // flag is NOT cleared here: the commit clears it when it installs
        // the created buffer and re-arms it (live id still zero) when the
        // outcome is undelivered.
        ps.instance_dirty = false;
    }
}

fn stageParticleGpu(scene: anytype, slot: anytype, allocator: std.mem.Allocator) void {
    for (scene.particles.systems.items, 0..) |ps, i| {
        if (!ps.gpu_dirty and !ps.gpu_flush_pending) continue;
        if (ps.gpu_slots.len == 0) continue;
        // Freeze the exact upload range; a pending flag with an empty window
        // still freezes an empty packet so the latch consumes the flags
        // without live reads.
        const range: []const @TypeOf(ps.gpu_slots[0]) = if (ps.gpu_dirty_wrapped)
            ps.gpu_slots[0..ps.gpu_high_water]
        else if (ps.gpu_dirty and ps.gpu_dirty_end > ps.gpu_dirty_start)
            ps.gpu_slots[ps.gpu_dirty_start..ps.gpu_dirty_end]
        else
            ps.gpu_slots[0..0];
        const data_lo = slot.p_gpu_data.items.len;
        if (range.len > 0) slot.p_gpu_data.appendSlice(allocator, range) catch continue;
        slot.p_gpu_uploads.append(allocator, .{
            .token = @intFromPtr(ps),
            .sys_index = @intCast(i),
            .buffer_id = ps.gpu_slot_buffer.id,
            .count = @intCast(range.len),
            .data_lo = data_lo,
            .capacity = ps.capacity,
        }) catch {
            slot.p_gpu_data.items.len = data_lo;
            continue;
        };
        // Consumed at stage time (see stageMorphs). Creation-pending stays
        // for the commit (see stageParticleCpu).
        ps.gpu_dirty = false;
        ps.gpu_dirty_wrapped = false;
        ps.gpu_flush_pending = false;
    }
}

fn stageParticleCompute(scene: anytype, slot: anytype, allocator: std.mem.Allocator) void {
    for (scene.particles.systems.items, 0..) |ps, i| {
        if (ps.simulation_mode != .compute) continue;
        if (!ps.compute_flush_pending and ps.compute_staged == 0 and !ps.compute_state_clear_pending and !ps.compute_buffers_pending) continue;
        const staged: usize = ps.compute_staged;
        if (staged > ps.compute_staging.len) continue;
        const data_lo = slot.p_compute_data.items.len;
        if (staged > 0) {
            slot.p_compute_data.appendSlice(allocator, ps.compute_staging[0..staged]) catch continue;
        }
        slot.p_compute_uploads.append(allocator, .{
            .token = @intFromPtr(ps),
            .sys_index = @intCast(i),
            .spawn_buffer_id = ps.compute_spawn_buffer.id,
            .staged = staged,
            .stage_base = ps.compute_stage_base,
            .cursor = ps.compute_cursor,
            .high_water = ps.compute_high_water,
            .dt_accum = ps.compute_dt_accum,
            .flush_pending = ps.compute_flush_pending,
            .state_clear_pending = ps.compute_state_clear_pending,
            .buffers_pending = ps.compute_buffers_pending,
            .gravity = .{ ps.gravity.x, ps.gravity.y, ps.gravity.z },
            .drag = ps.drag,
            .sheet_cols = ps.spritesheet_columns,
            .sheet_rows = ps.spritesheet_rows,
            .sheet_loops = ps.spritesheet_loops,
            .data_lo = data_lo,
            .data_count = staged,
            // Frozen capacity (immutable after init) + frozen GPU object
            // ids: the direct staged upload/dispatch addresses these
            // without touching live handles.
            .capacity = ps.capacity,
            .state_buffer_id = ps.compute_state_buffer.id,
            .draw_buffer_id = ps.compute_draw_buffer.id,
            .state_view_id = ps.compute_state_view.id,
            .spawn_view_id = ps.compute_spawn_view.id,
            .draw_view_id = ps.compute_draw_view.id,
            .shader_id = ps.compute_shader.id,
            .pipeline_id = ps.compute_pipeline.id,
        }) catch {
            slot.p_compute_data.items.len = data_lo;
            continue;
        };
        // Consumed at stage time (see stageMorphs): the flush/upload flags
        // for THIS frozen window. The live ring itself is NOT consumed
        // here — the commit advances it by the outcome's consumed counts
        // (guarded), so post-freeze appends are never lost and a failed
        // upload re-freezes the intact window. Creation-pending stays for
        // the commit (see stageParticleCpu).
        ps.compute_flush_pending = false;
        ps.compute_state_clear_pending = false;
    }
}

fn stageTrails(scene: anytype, slot: anytype, allocator: std.mem.Allocator) void {
    for (scene.trails.meshes.items, 0..) |tm, i| {
        if (!tm.gpu_dirty) continue;
        const vc: usize = tm.pending_vertex_count;
        const ic: usize = tm.pending_index_count;
        if (vc > tm.vertices.len or ic > tm.indices.len) continue;
        // Empty-but-dirty freezes an empty packet: the legacy flush clears
        // the flag and publishes index_count/bounds unconditionally.
        const v_lo = slot.trail_verts.items.len;
        const i_lo = slot.trail_indices.items.len;
        if (vc > 0) slot.trail_verts.appendSlice(allocator, tm.vertices[0..vc]) catch continue;
        if (ic > 0) slot.trail_indices.appendSlice(allocator, tm.indices[0..ic]) catch {
            slot.trail_verts.items.len = v_lo;
            continue;
        };
        slot.trail_uploads.append(allocator, .{
            .token = @intFromPtr(tm),
            .trail_index = @intCast(i),
            .vertex_buffer_id = tm.mesh.vertex_buffer.id,
            .index_buffer_id = tm.mesh.index_buffer.id,
            .vert_count = vc,
            .index_count = ic,
            .vert_lo = v_lo,
            .index_lo = i_lo,
            .min_pt = .{ tm.pending_min_pt.x, tm.pending_min_pt.y, tm.pending_min_pt.z },
            .max_pt = .{ tm.pending_max_pt.x, tm.pending_max_pt.y, tm.pending_max_pt.z },
            .buffers_pending = tm.buffers_pending,
            // Frozen allocation lengths (fixed at trail init): creation
            // sizes these, never the live slices.
            .vert_cap = tm.vertices.len,
            .index_cap = tm.indices.len,
        }) catch {
            slot.trail_verts.items.len = v_lo;
            slot.trail_indices.items.len = i_lo;
            continue;
        };
        // Consumed at stage time (see stageMorphs). Creation-pending and
        // the scalar publishes (index_count/bounds) stay for the commit.
        tm.gpu_dirty = false;
    }
}

fn stageSoftbodies(scene: anytype, slot: anytype, allocator: std.mem.Allocator) void {
    for (scene.softbodies.bodies.items, 0..) |b, i| {
        if (!b.upload_pending) continue;
        // Empty verts still freeze an empty packet: the legacy flush clears
        // the flag unconditionally. Indices are frozen alongside the verts
        // (grid topology is fixed at creation, but freezing keeps the staged
        // flush independent of every live array).
        const data_lo = slot.soft_data.items.len;
        if (b.vertices.len > 0) slot.soft_data.appendSlice(allocator, b.vertices) catch continue;
        const i_lo = slot.soft_indices.items.len;
        if (b.indices.len > 0) slot.soft_indices.appendSlice(allocator, b.indices) catch {
            slot.soft_data.items.len = data_lo;
            continue;
        };
        const bb = b.mesh.local_bounding_box;
        slot.soft_uploads.append(allocator, .{
            .token = @intFromPtr(b),
            .body_index = @intCast(i),
            .vertex_buffer_id = b.mesh.vertex_buffer.id,
            .vert_count = b.vertices.len,
            .data_lo = data_lo,
            .index_lo = i_lo,
            .index_count = b.indices.len,
            .min_pt = .{ bb.min.x, bb.min.y, bb.min.z },
            .max_pt = .{ bb.max.x, bb.max.y, bb.max.z },
            .buffers_pending = b.buffers_pending,
            // Frozen vertex allocation length (fixed grid at creation).
            .vert_cap = b.vertices.len,
        }) catch {
            slot.soft_data.items.len = data_lo;
            slot.soft_indices.items.len = i_lo;
            continue;
        };
        // Consumed at stage time (see stageMorphs). Creation-pending stays
        // for the commit (see stageParticleCpu).
        b.upload_pending = false;
    }
}

fn stageGreased(scene: anytype, slot: anytype, allocator: std.mem.Allocator) void {
    for (scene.greased_lines.items, 0..) |gl, i| {
        if (!gl.gpu_dirty) continue;
        // Empty verts still freeze an empty packet so a live context clears
        // the flag exactly like the legacy flush. Indices are frozen
        // whenever a full upload may be needed: the staged full flag, or
        // missing live buffers (the flush will create them, which forces a
        // full index upload there).
        const v_lo = slot.greased_verts.items.len;
        if (gl.vertices.len > 0) slot.greased_verts.appendSlice(allocator, gl.vertices) catch continue;
        const full = gl.gpu_needs_full_upload;
        const need_idx = full or gl.mesh.vertex_buffer.id == 0;
        const i_lo = slot.greased_indices.items.len;
        if (need_idx and gl.indices.len > 0) {
            slot.greased_indices.appendSlice(allocator, gl.indices) catch {
                slot.greased_verts.items.len = v_lo;
                continue;
            };
        }
        slot.greased_uploads.append(allocator, .{
            .token = @intFromPtr(gl),
            .line_index = @intCast(i),
            .vertex_buffer_id = gl.mesh.vertex_buffer.id,
            .index_buffer_id = gl.mesh.index_buffer.id,
            .vert_count = gl.vertices.len,
            .index_count = if (need_idx) gl.indices.len else 0,
            .vert_lo = v_lo,
            .index_lo = i_lo,
            .full_upload = full,
            // Frozen allocation lengths: creation sizes the buffers from
            // these, never the live slices.
            .vert_cap = gl.vertices.len,
            .index_cap = gl.indices.len,
        }) catch {
            slot.greased_verts.items.len = v_lo;
            if (need_idx) slot.greased_indices.items.len = i_lo;
            continue;
        };
        // Consumed at stage time (see stageMorphs). The full-upload flag
        // stays live: the commit clears it only when the full upload
        // landed AND no newer mutation re-dirtied the line meanwhile.
        gl.gpu_dirty = false;
    }
}

fn stagePendingMeshes(scene: anytype, slot: anytype, allocator: std.mem.Allocator) void {
    for (scene.meshes.items, 0..) |m, i| {
        if (!m.gpu_pending) continue;
        // Degenerate geometry still freezes a (possibly empty) packet: the
        // legacy finish attempts creation whenever gpu_pending is set, so
        // the staged flush must observe the same owner instead of skipping
        // it silently.
        //
        // GPU-morph meshes additionally freeze their packed RGBA32F delta
        // pixels (producer-side pack, sg-free) so the context creates the
        // delta image + view from frozen bytes alongside the buffers —
        // write-once (the flag is consumed at stage time below), so the
        // per-frame path stays a pure weights uniform with no base-pose
        // frame. Any freeze failure below keeps BOTH flags for the next
        // build (same OOM contract as every other owner).
        const want_delta = m.morph_upload_pending and m.morph_mode == .gpu and m.morph_targets.len > 0;
        var delta_pixels: []f32 = &.{};
        var delta_size: morph_gpu.TextureSize = .{ .width = 0, .height = 0 };
        if (want_delta) {
            delta_size = morph_gpu.textureSizeFor(m.morph_base.len);
            delta_pixels = morph_gpu.packDeltas(allocator, m.morph_targets, m.morph_base.len, delta_size) catch continue;
        }
        defer if (delta_pixels.len > 0) allocator.free(delta_pixels);
        const v_lo = slot.pending_verts.items.len;
        if (m.pending_vertices.len > 0) slot.pending_verts.appendSlice(allocator, m.pending_vertices) catch continue;
        const i_lo = slot.pending_indices.items.len;
        if (m.cpu_indices.len > 0) slot.pending_indices.appendSlice(allocator, m.cpu_indices) catch {
            slot.pending_verts.items.len = v_lo;
            continue;
        };
        const d_lo = slot.pending_delta_data.items.len;
        if (delta_pixels.len > 0) slot.pending_delta_data.appendSlice(allocator, delta_pixels) catch {
            slot.pending_verts.items.len = v_lo;
            slot.pending_indices.items.len = i_lo;
            continue;
        };
        slot.pending_uploads.append(allocator, .{
            .token = @intFromPtr(m),
            .uid = m.uid,
            .mesh_index = @intCast(i),
            .vert_count = m.pending_vertices.len,
            .index_count = m.cpu_indices.len,
            .vert_lo = v_lo,
            .index_lo = i_lo,
            .index_type_is_u16 = m.index_type == .UINT16,
            .dynamic_update = m.pending_dynamic_update,
            .morph_delta_pending = want_delta,
            .delta_lo = d_lo,
            .delta_count = delta_pixels.len,
            .delta_width = delta_size.width,
            .delta_height = delta_size.height,
        }) catch {
            slot.pending_verts.items.len = v_lo;
            slot.pending_indices.items.len = i_lo;
            slot.pending_delta_data.items.len = d_lo;
            continue;
        };
        // Consumed at stage time (see stageMorphs): the deferred creation
        // will not be attempted again until the commit re-arms it on an
        // undelivered outcome. `pending_dynamic_update` stays live until
        // the commit (which clears it on delivery), so a failed creation
        // still knows to re-arm `morph_upload_needed`. `morph_upload_pending`
        // is consumed here for the same reason (the delta bytes above are
        // the frozen request); the commit re-arms it when the delta
        // texture does not land.
        m.gpu_pending = false;
        if (want_delta) m.morph_upload_pending = false;
    }
}

/// Context-side slot flush (beginPrepare, fresh-build path only). Uploads
/// from `slot` packets and creates deferred buffers from FROZEN sizes,
/// recording per-packet outcomes in the slot descriptors for the
/// game-side commit. THE CONTEXT NEVER WRITES GAME-OWNED STATE HERE: no
/// live flag clears, no live scalar publishes, no live handle installs, no
/// live array frees, no live staging memcpy. No live staging bytes drive
/// uploads (compute uploads direct from packet bytes) and no live mutable
/// state is read at all — only the slot packets, the stable allocator,
/// and context-owned services (retire queue, upload meter, sg).
/// Headless-safe: with no sg context every packet records undelivered and
/// the commit re-arms the flags, only the sg.* calls are skipped. See the
/// module header for the full contract.
pub fn flushSlotUploads(scene: anytype, slot: anytype) void {
    gpu_thread.assertOnContextThread();
    scene.gpu_retire.flush(scene.allocator);
    // Re-flush guard: a cancelled prepare re-latches the same slot without
    // a rebuild, so outcomes from the previous attempt must not linger
    // (an undelivered packet would look delivered). Slot-owned writes into
    // the claimed slot — race-free by the lease protocol.
    resetOutcomes(slot);
    flushPendingCreations(scene, slot);
    flushMorphs(scene, slot);
    flushParticleCpu(scene, slot);
    flushParticleGpu(scene, slot);
    flushParticleCompute(scene, slot);
    flushTrails(scene, slot);
    flushSoftbodies(scene, slot);
    flushGreased(scene, slot);
    if (!scene.flush_in_prepare and sg.isvalid()) sg.commit();
}

/// Clears every packet outcome back to undelivered (see flushSlotUploads).
fn resetOutcomes(slot: anytype) void {
    for (slot.morph_uploads.items) |*up| up.delivered = false;
    for (slot.p_cpu_uploads.items) |*up| {
        up.delivered = false;
        up.created_buffer_id = 0;
    }
    for (slot.p_gpu_uploads.items) |*up| {
        up.delivered = false;
        up.created_buffer_id = 0;
    }
    for (slot.p_compute_uploads.items) |*up| {
        up.delivered = false;
        up.unsupported = false;
        up.consumed_staged = 0;
        up.consumed_dt = 0.0;
        up.created_state_buffer_id = 0;
        up.created_spawn_buffer_id = 0;
        up.created_draw_buffer_id = 0;
        up.created_state_view_id = 0;
        up.created_spawn_view_id = 0;
        up.created_draw_view_id = 0;
        up.created_shader_id = 0;
        up.created_pipeline_id = 0;
    }
    for (slot.trail_uploads.items) |*up| {
        up.delivered = false;
        up.created_vertex_buffer_id = 0;
        up.created_index_buffer_id = 0;
    }
    for (slot.soft_uploads.items) |*up| {
        up.delivered = false;
        up.created_vertex_buffer_id = 0;
        up.created_index_buffer_id = 0;
    }
    for (slot.greased_uploads.items) |*up| {
        up.delivered = false;
        up.full_delivered = false;
        up.created_vertex_buffer_id = 0;
        up.created_index_buffer_id = 0;
    }
    for (slot.pending_uploads.items) |*up| {
        up.delivered = false;
        up.created_vertex_buffer_id = 0;
        up.created_index_buffer_id = 0;
        up.created_delta_image_id = 0;
        up.created_delta_view_id = 0;
    }
}

fn flushPendingCreations(scene: anytype, slot: anytype) void {
    // No live reads, no live writes: creation runs purely from frozen
    // geometry/sizes; the game-side commit installs the handles (guarded)
    // or re-arms `gpu_pending` for retry. Headless records undelivered.
    if (!sg.isvalid()) return;
    for (slot.pending_uploads.items) |*up| {
        const v_end = up.vert_lo + up.vert_count;
        const i_end = up.index_lo + up.index_count;
        if (v_end > slot.pending_verts.items.len or i_end > slot.pending_indices.items.len) continue;
        const verts = slot.pending_verts.items[up.vert_lo..v_end];
        const idx32 = slot.pending_indices.items[up.index_lo..i_end];
        var vbuf: sg.Buffer = .{};
        var ibuf: sg.Buffer = .{};
        if (up.dynamic_update) {
            vbuf = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = verts.len * @sizeOf(@TypeOf(verts[0])),
            });
            if (vbuf.id == 0) continue;
            if (up.index_type_is_u16) {
                const tmp = scene.allocator.alloc(u16, idx32.len) catch {
                    sg.destroyBuffer(vbuf);
                    continue;
                };
                defer scene.allocator.free(tmp);
                for (idx32, 0..) |v, k| tmp[k] = @intCast(v);
                ibuf = sg.makeBuffer(.{ .usage = .{ .index_buffer = true }, .data = sg.asRange(tmp) });
            } else {
                ibuf = sg.makeBuffer(.{ .usage = .{ .index_buffer = true }, .data = sg.asRange(idx32) });
            }
            if (ibuf.id == 0) {
                sg.destroyBuffer(vbuf);
                continue;
            }
            up.created_vertex_buffer_id = vbuf.id;
            up.created_index_buffer_id = ibuf.id;
        } else {
            vbuf = sg.makeBuffer(.{ .data = sg.asRange(verts) });
            if (vbuf.id == 0) continue;
            if (up.index_type_is_u16) {
                const tmp = scene.allocator.alloc(u16, idx32.len) catch {
                    sg.destroyBuffer(vbuf);
                    continue;
                };
                defer scene.allocator.free(tmp);
                for (idx32, 0..) |v, k| tmp[k] = @intCast(v);
                ibuf = sg.makeBuffer(.{ .usage = .{ .index_buffer = true }, .data = sg.asRange(tmp) });
            } else {
                ibuf = sg.makeBuffer(.{ .usage = .{ .index_buffer = true }, .data = sg.asRange(idx32) });
            }
            if (ibuf.id == 0) {
                sg.destroyBuffer(vbuf);
                continue;
            }
            up.created_vertex_buffer_id = vbuf.id;
            up.created_index_buffer_id = ibuf.id;
        }
        // Delivered: the commit publishes vertex_count (when live is zero),
        // re-arms morph_upload_needed for dynamic updates, clears
        // pending_dynamic_update, and frees the consumed live arrays.
        // GPU-morph meshes additionally land their delta texture here, from
        // the frozen packed bytes (never live morph_targets): any failure
        // tears the fresh buffers back down inline (context thread, legal)
        // and records undelivered, so the commit re-arms both flags and the
        // next build retries the whole finish atomically — the mesh stays
        // skipped by the queue meanwhile (no base-pose frame, no panic).
        if (up.morph_delta_pending) {
            const d_end = up.delta_lo + up.delta_count;
            const want: usize = @as(usize, up.delta_width) * up.delta_height * 4;
            if (d_end > slot.pending_delta_data.items.len or up.delta_count != want or want == 0) {
                sg.destroyBuffer(vbuf);
                sg.destroyBuffer(ibuf);
                continue;
            }
            var img_desc = sg.ImageDesc{
                .width = @intCast(up.delta_width),
                .height = @intCast(up.delta_height),
                .pixel_format = .RGBA32F,
            };
            img_desc.data.mip_levels[0] = sg.asRange(slot.pending_delta_data.items[up.delta_lo..d_end]);
            const img = sg.makeImage(img_desc);
            if (img.id == 0) {
                sg.destroyBuffer(vbuf);
                sg.destroyBuffer(ibuf);
                continue;
            }
            const view = sg.makeView(.{ .texture = .{ .image = img } });
            if (view.id == 0) {
                sg.destroyImage(img);
                sg.destroyBuffer(vbuf);
                sg.destroyBuffer(ibuf);
                continue;
            }
            up.created_delta_image_id = img.id;
            up.created_delta_view_id = view.id;
            upload_meter.record(up.delta_count * @sizeOf(f32));
        }
        up.delivered = true;
    }
}

fn flushMorphs(scene: anytype, slot: anytype) void {
    _ = scene;
    // No live reads, no live writes. The frozen target id is uploaded to
    // directly: sokol ids are generation-tagged, so a stale id (owner
    // destroyed between build and latch — an app-contract violation the
    // commit guard also skips) fail-closes in the driver, never aliases a
    // recycled buffer. Morph buffers are updated in place and never
    // replaced, so no foreign-buffer case exists here by construction.
    for (slot.morph_uploads.items) |*up| {
        const end = up.data_lo + up.count;
        if (end > slot.morph_data.items.len) continue;
        if (up.count == 0) {
            up.delivered = true;
            continue;
        }
        if (!sg.isvalid()) continue;
        if (up.buffer_id == 0) continue;
        sg.updateBuffer(.{ .id = up.buffer_id }, sg.asRange(slot.morph_data.items[up.data_lo..end]));
        upload_meter.record(up.count * @sizeOf(@TypeOf(slot.morph_data.items[0])));
        up.delivered = true;
    }
}

fn flushParticleCpu(scene: anytype, slot: anytype) void {
    // No live reads, no live writes. Deferred creation sizes from the
    // FROZEN capacity (immutable after init); the outcome carries the
    // created buffer for the game-side commit, or stays undelivered for
    // retry. Instance buffers are never replaced (updated in place), so
    // the frozen target id is uploaded to directly (see flushMorphs).
    for (slot.p_cpu_uploads.items) |*up| {
        const end = up.data_lo + up.count;
        if (end > slot.p_cpu_data.items.len) continue;
        var target_id = up.buffer_id;
        if (target_id == 0) {
            // Deferred first-use creation (staged by the producer as
            // "no buffer yet"): frozen capacity, never the live field.
            if (!sg.isvalid()) continue;
            if (up.capacity == 0) continue;
            const created = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = up.capacity * @sizeOf(@TypeOf(slot.p_cpu_data.items[0])),
            });
            if (created.id == 0) continue;
            up.created_buffer_id = created.id;
            target_id = created.id;
        }
        if (up.count == 0) {
            up.delivered = true;
            continue;
        }
        if (!sg.isvalid()) continue;
        sg.updateBuffer(.{ .id = target_id }, sg.asRange(slot.p_cpu_data.items[up.data_lo..end]));
        upload_meter.record(up.count * @sizeOf(@TypeOf(slot.p_cpu_data.items[0])));
        up.delivered = true;
    }
    _ = scene;
}

fn flushParticleGpu(scene: anytype, slot: anytype) void {
    // Same shape as flushParticleCpu (see above).
    for (slot.p_gpu_uploads.items) |*up| {
        const end = up.data_lo + up.count;
        if (end > slot.p_gpu_data.items.len) continue;
        var target_id = up.buffer_id;
        if (target_id == 0) {
            if (!sg.isvalid()) continue;
            if (up.capacity == 0) continue;
            const created = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = up.capacity * @sizeOf(@TypeOf(slot.p_gpu_data.items[0])),
            });
            if (created.id == 0) continue;
            up.created_buffer_id = created.id;
            target_id = created.id;
        }
        if (up.count == 0) {
            up.delivered = true;
            continue;
        }
        if (!sg.isvalid()) continue;
        sg.updateBuffer(.{ .id = target_id }, sg.asRange(slot.p_gpu_data.items[up.data_lo..end]));
        upload_meter.record(up.count * @sizeOf(@TypeOf(slot.p_gpu_data.items[0])));
        up.delivered = true;
    }
    _ = scene;
}

fn flushParticleCompute(scene: anytype, slot: anytype) void {
    // Direct staged upload from packet bytes (phase 2): the legacy
    // helper's frozen-window memcpy reinstall into the live staging array
    // is DELETED — no live staging bytes drive uploads, only frozen
    // counts. No live reads, no live writes: creation/dispatch run on
    // frozen ids + frozen capacity, outcomes ride the descriptor for the
    // game-side commit (ring advance, dt consume, handle installs,
    // unsupported latch) or stay undelivered for retry.
    const compute = @import("../compute.zig");
    const pc_shd = @import("particle_compute_shader");
    const workgroup_size = @import("../particles/types.zig").compute_workgroup_size;
    for (slot.p_compute_uploads.items) |*up| {
        const end = up.data_lo + up.data_count;
        if (end > slot.p_compute_data.items.len) continue;
        if (!sg.isvalid()) {
            // Headless: full no-op that records undelivered for retry,
            // exactly like the legacy helper's `if (!sg.isvalid()) return`.
            // The frozen bytes stay in the slot; the next funded build
            // re-freezes from the intact live window.
            continue;
        }
        if (!compute.supported()) {
            // Latch unsupported context-side (mirrors the legacy helper:
            // never a silent fallback, never a retry spin) — published
            // through the outcome, installed by the commit.
            up.unsupported = true;
            continue;
        }
        const cap = up.capacity;
        if (cap == 0) continue;
        // Creation from frozen capacity + frozen request flags. Partial
        // progress is recorded into the outcome even on failure so the
        // commit can install it and the retry completes the rest.
        var state_id = up.state_buffer_id;
        var spawn_id = up.spawn_buffer_id;
        var draw_id = up.draw_buffer_id;
        var state_view = up.state_view_id;
        var spawn_view = up.spawn_view_id;
        var draw_view = up.draw_view_id;
        var shader_id = up.shader_id;
        var pipeline_id = up.pipeline_id;
        if (up.buffers_pending) {
            if (state_id == 0) {
                const zeros = scene.allocator.alloc(u8, cap * @sizeOf(@import("../particles/types.zig").ComputeParticleState)) catch continue;
                defer scene.allocator.free(zeros);
                @memset(zeros, 0);
                const created = sg.makeBuffer(.{
                    .usage = .{ .storage_buffer = true },
                    .data = sg.Range{ .ptr = zeros.ptr, .size = zeros.len },
                });
                if (created.id == 0) continue;
                up.created_state_buffer_id = created.id;
                state_id = created.id;
            }
            if (spawn_id == 0) {
                const created = sg.makeBuffer(.{
                    .usage = .{ .storage_buffer = true, .dynamic_update = true },
                    .size = cap * @sizeOf(@import("../particles/types.zig").GpuParticleSlot),
                });
                if (created.id == 0) continue;
                up.created_spawn_buffer_id = created.id;
                spawn_id = created.id;
            }
            if (draw_id == 0) {
                const created = sg.makeBuffer(.{
                    .usage = .{ .vertex_buffer = true, .storage_buffer = true },
                    .size = cap * @sizeOf(@import("../particles/types.zig").ParticleInstanceData),
                });
                if (created.id == 0) continue;
                up.created_draw_buffer_id = created.id;
                draw_id = created.id;
            }
            if (state_view == 0 and state_id != 0) {
                const created = compute.makeStorageView(.{ .id = state_id }, "compute-particles-state");
                if (created.id == 0) continue;
                up.created_state_view_id = created.id;
                state_view = created.id;
            }
            if (spawn_view == 0 and spawn_id != 0) {
                const created = compute.makeStorageView(.{ .id = spawn_id }, "compute-particles-spawn");
                if (created.id == 0) continue;
                up.created_spawn_view_id = created.id;
                spawn_view = created.id;
            }
            if (draw_view == 0 and draw_id != 0) {
                const created = compute.makeStorageView(.{ .id = draw_id }, "compute-particles-draw");
                if (created.id == 0) continue;
                up.created_draw_view_id = created.id;
                draw_view = created.id;
            }
            if (shader_id == 0) {
                const created = sg.makeShader(pc_shd.particleComputeShaderDesc(sg.queryBackend()));
                if (created.id == 0) continue;
                up.created_shader_id = created.id;
                shader_id = created.id;
            }
            if (pipeline_id == 0 and shader_id != 0) {
                const created = compute.makePipeline(.{ .id = shader_id }, "compute-particles");
                if (created.id == 0) continue;
                up.created_pipeline_id = created.id;
                pipeline_id = created.id;
            }
        }
        if (state_id == 0) continue;
        if (up.state_clear_pending) {
            const zeros = scene.allocator.alloc(u8, cap * @sizeOf(@import("../particles/types.zig").ComputeParticleState)) catch continue;
            defer scene.allocator.free(zeros);
            @memset(zeros, 0);
            sg.updateBuffer(.{ .id = state_id }, sg.Range{ .ptr = zeros.ptr, .size = zeros.len });
            upload_meter.record(zeros.len);
        }
        // Creation incomplete (a make* failed; partial handles ride the
        // outcome for the commit to install): keep everything staged,
        // consume nothing.
        if (pipeline_id == 0) continue;
        const pkt = slot.p_compute_data.items[up.data_lo..end];
        if (up.data_count > 0 and spawn_id != 0) {
            sg.updateBuffer(.{ .id = spawn_id }, sg.asRange(pkt));
            upload_meter.record(up.data_count * @sizeOf(@TypeOf(pkt[0])));
        }
        // Dispatch on staged spawns even without a frame update (a
        // sub-emitter child may hold spawns while its own update did not
        // run; dt 0 then integrates nothing but still applies respawns).
        // Consumes the window/dt only here, via the outcome below.
        if (up.flush_pending or up.data_count > 0) {
            if (up.high_water > 0) {
                const params = pc_shd.CsParams{
                    .dyn = .{ up.dt_accum, up.drag, 0.0, 0.0 },
                    .grav = .{ up.gravity[0], up.gravity[1], up.gravity[2], 0.0 },
                    .sheet = .{
                        @floatFromInt(up.sheet_cols),
                        @floatFromInt(up.sheet_rows),
                        up.sheet_loops,
                        0.0,
                    },
                    .addr = .{
                        @floatFromInt(cap),
                        @floatFromInt(up.stage_base),
                        @floatFromInt(up.data_count),
                        0.0,
                    },
                };
                sg.beginPass(.{ .compute = true, .label = "compute-particles" });
                sg.applyPipeline(.{ .id = pipeline_id });
                var bind = sg.Bindings{};
                bind.views[pc_shd.VIEW_cs_state] = .{ .id = state_view };
                bind.views[pc_shd.VIEW_cs_spawn] = .{ .id = spawn_view };
                bind.views[pc_shd.VIEW_cs_draw] = .{ .id = draw_view };
                sg.applyBindings(bind);
                sg.applyUniforms(pc_shd.UB_cs_params, sg.asRange(&params));
                sg.dispatch(@intCast(compute.groupCount(up.high_water, workgroup_size)), 1, 1);
                sg.endPass();
            }
            up.consumed_staged = up.data_count;
            up.consumed_dt = up.dt_accum;
        }
        up.delivered = true;
    }
}

fn flushTrails(scene: anytype, slot: anytype) void {
    // No live reads, no live writes. Creation sizes from the FROZEN caps
    // (fixed at trail init); uploads target the frozen ids directly (trail
    // buffers are only ever created from zero, never replaced, so no
    // foreign-buffer case exists by construction — same rationale as
    // flushMorphs). Scalars (index_count/bounds) and created handles ride
    // the outcome for the game-side commit.
    _ = scene;
    for (slot.trail_uploads.items) |*up| {
        const v_end = up.vert_lo + up.vert_count;
        const i_end = up.index_lo + up.index_count;
        if (v_end > slot.trail_verts.items.len or i_end > slot.trail_indices.items.len) continue;
        var vertex_id = up.vertex_buffer_id;
        var index_id = up.index_buffer_id;
        if (up.buffers_pending) {
            if (!sg.isvalid()) continue;
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = up.vert_cap * @sizeOf(@TypeOf(slot.trail_verts.items[0])),
            });
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true, .dynamic_update = true },
                .size = up.index_cap * @sizeOf(@TypeOf(slot.trail_indices.items[0])),
            });
            if (vb.id != 0 and ib.id != 0) {
                up.created_vertex_buffer_id = vb.id;
                up.created_index_buffer_id = ib.id;
                vertex_id = vb.id;
                index_id = ib.id;
            } else {
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
                continue;
            }
        }
        if (sg.isvalid()) {
            if (up.vert_count > 0 and vertex_id != 0) {
                sg.updateBuffer(.{ .id = vertex_id }, sg.asRange(slot.trail_verts.items[up.vert_lo..v_end]));
                upload_meter.record(up.vert_count * @sizeOf(@TypeOf(slot.trail_verts.items[0])));
            }
            if (up.index_count > 0 and index_id != 0) {
                sg.updateBuffer(.{ .id = index_id }, sg.asRange(slot.trail_indices.items[up.index_lo..i_end]));
                upload_meter.record(up.index_count * @sizeOf(@TypeOf(slot.trail_indices.items[0])));
            }
            up.delivered = true;
        } else {
            // Headless: an empty packet with no creation still delivers
            // (the commit publishes the zero scalars exactly like the
            // legacy flush, consuming the generation); any payload or
            // creation need records undelivered for retry once a context
            // exists. Uploading to a zero id headless would be a no-op
            // mistaken for delivery, so the zero-id + payload case also
            // stays undelivered.
            const needs_creation = up.buffers_pending;
            const has_payload = up.vert_count > 0 or up.index_count > 0;
            if (!needs_creation and !has_payload) up.delivered = true;
        }
    }
}

fn flushSoftbodies(scene: anytype, slot: anytype) void {
    // No live reads, no live writes. Creation sizes from frozen data (the
    // vertex buffer from the frozen cap, the u32 index buffer from the
    // frozen index bytes); uploads target the frozen id directly (soft
    // buffers are only ever created from zero, never replaced — same
    // rationale as flushMorphs). Outcomes ride the descriptor for the
    // game-side commit.
    _ = scene;
    for (slot.soft_uploads.items) |*up| {
        const end = up.data_lo + up.vert_count;
        if (end > slot.soft_data.items.len) continue;
        const i_end = up.index_lo + up.index_count;
        if (i_end > slot.soft_indices.items.len) continue;
        var vertex_id = up.vertex_buffer_id;
        if (up.buffers_pending) {
            if (!sg.isvalid()) continue;
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = up.vert_cap * @sizeOf(@TypeOf(slot.soft_data.items[0])),
            });
            // Cloth index buffers are always u32 (vertices cap at 4096, so
            // this wastes nothing material and keeps one upload path).
            // Bytes come from the frozen packet, never the live array.
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(slot.soft_indices.items[up.index_lo..i_end]),
            });
            if (vb.id != 0 and ib.id != 0) {
                up.created_vertex_buffer_id = vb.id;
                up.created_index_buffer_id = ib.id;
                vertex_id = vb.id;
            } else {
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
                continue;
            }
        }
        if (sg.isvalid()) {
            if (up.vert_count == 0) {
                up.delivered = true;
                continue;
            }
            if (vertex_id == 0) continue;
            sg.updateBuffer(.{ .id = vertex_id }, sg.asRange(slot.soft_data.items[up.data_lo..end]));
            upload_meter.record(up.vert_count * @sizeOf(@TypeOf(slot.soft_data.items[0])));
            up.delivered = true;
        } else {
            // Headless rule (see flushTrails): empty + no creation still
            // delivers; anything else retries.
            if (!up.buffers_pending and up.vert_count == 0) up.delivered = true;
        }
    }
}

fn flushGreased(scene: anytype, slot: anytype) void {
    // No live reads, no live writes. Creation sizes from the FROZEN caps;
    // uploads target the frozen ids directly (greased buffers are only
    // ever created from zero, never replaced — same rationale as
    // flushMorphs). Outcomes ride the descriptor for the game-side commit.
    _ = scene;
    for (slot.greased_uploads.items) |*up| {
        const v_end = up.vert_lo + up.vert_count;
        if (v_end > slot.greased_verts.items.len) continue;
        const i_end = up.index_lo + up.index_count;
        if (i_end > slot.greased_indices.items.len) continue;
        // Headless rule (see flushTrails): empty + existing targets still
        // delivers (consumes the generation exactly like the legacy
        // `if (!sg.isvalid()) return`-with-retain... no — the legacy path
        // RETAINS the flag headless for retry. Match it: headless always
        // records undelivered; the commit re-arms for retry once a context
        // exists.
        if (!sg.isvalid()) continue;
        var vertex_id = up.vertex_buffer_id;
        var index_id = up.index_buffer_id;
        var just_created = false;
        if (vertex_id == 0) {
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = up.vert_cap * @sizeOf(@TypeOf(slot.greased_verts.items[0])),
            });
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true, .dynamic_update = true },
                .size = up.index_cap * @sizeOf(@TypeOf(slot.greased_indices.items[0])),
            });
            if (vb.id == 0 or ib.id == 0) {
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
                // Creation failed (pool exhaustion): undelivered, the
                // commit re-arms for retry like the legacy path.
                continue;
            }
            up.created_vertex_buffer_id = vb.id;
            up.created_index_buffer_id = ib.id;
            vertex_id = vb.id;
            index_id = ib.id;
            just_created = true;
        }
        if (up.vert_count > 0 and vertex_id != 0) {
            sg.updateBuffer(.{ .id = vertex_id }, sg.asRange(slot.greased_verts.items[up.vert_lo..v_end]));
            upload_meter.record(up.vert_count * @sizeOf(@TypeOf(slot.greased_verts.items[0])));
        }
        // A just-created index buffer is empty: it needs the full index
        // upload even when the frozen full flag was false (the stage
        // froze indices whenever live buffers were missing, precisely for
        // this case).
        if ((up.full_upload or just_created) and up.index_count > 0 and index_id != 0) {
            sg.updateBuffer(.{ .id = index_id }, sg.asRange(slot.greased_indices.items[up.index_lo..i_end]));
            upload_meter.record(up.index_count * @sizeOf(@TypeOf(slot.greased_indices.items[0])));
            up.full_delivered = true;
        } else if (up.full_upload and up.index_count == 0) {
            // Full requested but nothing frozen (degenerate): the full
            // requirement is satisfied vacuously.
            up.full_delivered = true;
        }
        up.delivered = true;
    }
}

/// Game-side commit of staged-upload outcomes (next build, under the game
/// lock, front slot under a read lease — see buildIntoClaimedSlot). For
/// every packet in `front` (the published frame), validates the owner by
/// index/token(/uid) with a pointer compare first — never dereferencing a
/// stale pointer — and either applies the outcome (created-handle
/// installs, scalar publishes, pending-array frees, compute ring advance)
/// or re-arms the owner's flags for retry when the outcome is undelivered.
///
/// Idempotence: the build core calls this once per published frame
/// (`last_upload_commit_frame` vs `front.frame_id`); a repeat build
/// without an intervening latch skips, so outcomes are never
/// double-applied. Re-arming a flag is itself idempotent (boolean set).
/// Created handles install only over a zero live id; a replaced non-zero
/// live handle means someone else created concurrently (defensive only —
/// in practice only the context creates, sequentially), so the created
/// handle retires through the thread-safe queue instead of leaking.
///
/// Front-slot mutation carve-out (documented, benign): the commit zeroes
/// the `created_*` outcome fields on the PUBLISHED front descriptors after
/// installing/retiring them, so a repeat commit is a no-op even without
/// the frame guard. These are outcome-only fields the render path never
/// reads (it consumes the draw payload + snapshot + staged records), and
/// the front slot is read-leased across this call — concurrent render
/// reads of the other fields stay race-free.
///
/// Memory ordering: outcomes were written by the context into its claimed
/// slot; the publish release edge (lease mutex) pairs with the
/// `pinFrontReader` acquire (same mutex) the caller holds across this
/// call — see the module header.
pub fn commitSlotResults(scene: anytype, front: anytype) void {
    commitMorphs(scene, front);
    commitParticleCpu(scene, front);
    commitParticleGpu(scene, front);
    commitParticleCompute(scene, front);
    commitTrails(scene, front);
    commitSoftbodies(scene, front);
    commitGreased(scene, front);
    commitPendingCreations(scene, front);
}

/// Game-side re-arm of a dropped build claim (`BuildClaim.cancel`): every
/// packet staged into `slot` gets its owner flags set again
/// (token/index/uid-validated), so a cancelled claim consumes nothing.
/// Runs under the game lock with full live access — the inverse of the
/// stage-time clear.
pub fn restageDroppedSlot(scene: anytype, slot: anytype) void {
    for (slot.morph_uploads.items) |up| {
        const m = commitMeshAt(scene, up.mesh_index, up.token, up.uid) orelse continue;
        m.morph_upload_needed = true;
    }
    for (slot.p_cpu_uploads.items) |up| {
        const ps = commitParticleAt(scene, up.sys_index, up.token) orelse continue;
        ps.instance_dirty = true;
        if (ps.instance_buffer.id == 0) ps.instance_buffer_pending = true;
    }
    for (slot.p_gpu_uploads.items) |up| {
        const ps = commitParticleAt(scene, up.sys_index, up.token) orelse continue;
        ps.gpu_dirty = true;
        ps.gpu_dirty_wrapped = true;
        ps.gpu_flush_pending = true;
        if (ps.gpu_slot_buffer.id == 0) ps.gpu_slot_buffer_pending = true;
    }
    for (slot.p_compute_uploads.items) |up| {
        const ps = commitParticleAt(scene, up.sys_index, up.token) orelse continue;
        if (ps.compute_staged > 0 or ps.compute_dt_accum != 0.0 or up.flush_pending) ps.compute_flush_pending = true;
        if (up.state_clear_pending) ps.compute_state_clear_pending = true;
        if (ps.compute_state_buffer.id == 0) ps.compute_buffers_pending = true;
    }
    for (slot.trail_uploads.items) |up| {
        const tm = commitTrailAt(scene, up.trail_index, up.token) orelse continue;
        tm.gpu_dirty = true;
        if (tm.mesh.vertex_buffer.id == 0 or tm.mesh.index_buffer.id == 0) tm.buffers_pending = true;
    }
    for (slot.soft_uploads.items) |up| {
        const b = commitSoftAt(scene, up.body_index, up.token) orelse continue;
        b.upload_pending = true;
        if (b.mesh.vertex_buffer.id == 0) b.buffers_pending = true;
    }
    for (slot.greased_uploads.items) |up| {
        const gl = commitGreasedAt(scene, up.line_index, up.token) orelse continue;
        gl.gpu_dirty = true;
    }
    for (slot.pending_uploads.items) |up| {
        const m = commitMeshAt(scene, up.mesh_index, up.token, up.uid) orelse continue;
        m.gpu_pending = true;
        if (up.morph_delta_pending) m.morph_upload_pending = true;
    }
}

/// Retire a created handle the commit cannot install (defensive: replaced
/// live handle). Thread-safe from the game side (spinlock, sg-free); the
/// context destroys it at the next flush.
fn retireCreated(scene: anytype, id: u32) void {
    if (id == 0) return;
    scene.gpu_retire.retireBuffer(scene.allocator, .{ .id = id });
}

fn commitMeshAt(scene: anytype, mesh_index: u32, token: usize, uid: u64) ?@TypeOf(scene.meshes.items[0]) {
    const idx: usize = mesh_index;
    if (idx >= scene.meshes.items.len) return null;
    const m = scene.meshes.items[idx];
    if (@intFromPtr(m) != token) return null;
    if (uid != 0 and m.uid != uid) return null;
    return m;
}

fn commitParticleAt(scene: anytype, sys_index: u32, token: usize) ?@TypeOf(scene.particles.systems.items[0]) {
    const idx: usize = sys_index;
    if (idx >= scene.particles.systems.items.len) return null;
    const ps = scene.particles.systems.items[idx];
    if (@intFromPtr(ps) != token) return null;
    return ps;
}

fn commitTrailAt(scene: anytype, trail_index: u32, token: usize) ?@TypeOf(scene.trails.meshes.items[0]) {
    const idx: usize = trail_index;
    if (idx >= scene.trails.meshes.items.len) return null;
    const tm = scene.trails.meshes.items[idx];
    if (@intFromPtr(tm) != token) return null;
    return tm;
}

fn commitSoftAt(scene: anytype, body_index: u32, token: usize) ?@TypeOf(scene.softbodies.bodies.items[0]) {
    const idx: usize = body_index;
    if (idx >= scene.softbodies.bodies.items.len) return null;
    const b = scene.softbodies.bodies.items[idx];
    if (@intFromPtr(b) != token) return null;
    return b;
}

fn commitGreasedAt(scene: anytype, line_index: u32, token: usize) ?@TypeOf(scene.greased_lines.items[0]) {
    const idx: usize = line_index;
    if (idx >= scene.greased_lines.items.len) return null;
    const gl = scene.greased_lines.items[idx];
    if (@intFromPtr(gl) != token) return null;
    return gl;
}

fn commitMorphs(scene: anytype, front: anytype) void {
    for (front.morph_uploads.items) |up| {
        if (up.delivered) continue;
        const m = commitMeshAt(scene, up.mesh_index, up.token, up.uid) orelse continue;
        m.morph_upload_needed = true;
    }
}

fn commitParticleCpu(scene: anytype, front: anytype) void {
    for (front.p_cpu_uploads.items) |*up| {
        const ps = commitParticleAt(scene, up.sys_index, up.token) orelse {
            retireCreated(scene, up.created_buffer_id);
            up.created_buffer_id = 0;
            continue;
        };
        if (!up.delivered) {
            ps.instance_dirty = true;
            if (ps.instance_buffer.id == 0) ps.instance_buffer_pending = true;
            continue;
        }
        if (up.created_buffer_id != 0) {
            if (ps.instance_buffer.id == 0) {
                ps.instance_buffer = .{ .id = up.created_buffer_id };
                ps.instance_buffer_pending = false;
            } else {
                retireCreated(scene, up.created_buffer_id);
            }
            up.created_buffer_id = 0;
        } else if (ps.instance_buffer.id == 0 and up.count > 0) {
            // Delivered without a target (stale zero id — nothing could
            // have landed): re-arm instead of losing the upload.
            ps.instance_dirty = true;
            ps.instance_buffer_pending = true;
        }
    }
}

fn commitParticleGpu(scene: anytype, front: anytype) void {
    for (front.p_gpu_uploads.items) |*up| {
        const ps = commitParticleAt(scene, up.sys_index, up.token) orelse {
            retireCreated(scene, up.created_buffer_id);
            up.created_buffer_id = 0;
            continue;
        };
        if (!up.delivered) {
            ps.gpu_dirty = true;
            ps.gpu_dirty_wrapped = true;
            ps.gpu_flush_pending = true;
            if (ps.gpu_slot_buffer.id == 0) ps.gpu_slot_buffer_pending = true;
            continue;
        }
        if (up.created_buffer_id != 0) {
            if (ps.gpu_slot_buffer.id == 0) {
                ps.gpu_slot_buffer = .{ .id = up.created_buffer_id };
                ps.gpu_slot_buffer_pending = false;
            } else {
                retireCreated(scene, up.created_buffer_id);
            }
            up.created_buffer_id = 0;
        } else if (ps.gpu_slot_buffer.id == 0 and up.count > 0) {
            ps.gpu_dirty = true;
            ps.gpu_dirty_wrapped = true;
            ps.gpu_flush_pending = true;
            ps.gpu_slot_buffer_pending = true;
        }
    }
}

fn commitParticleCompute(scene: anytype, front: anytype) void {
    for (front.p_compute_uploads.items) |*up| {
        const ps = commitParticleAt(scene, up.sys_index, up.token) orelse {
            retireComputeCreated(scene, up);
            continue;
        };
        if (up.unsupported) {
            // Backend without compute support (mirrors the legacy helper):
            // latch unsupported, drop the pending flags — never a silent
            // fallback, never a retry spin.
            ps.compute_known_unsupported = true;
            ps.compute_buffers_pending = false;
            ps.compute_flush_pending = false;
            retireComputeCreated(scene, up);
            continue;
        }
        if (!up.delivered) {
            if (ps.compute_staged > 0 or ps.compute_dt_accum != 0.0) ps.compute_flush_pending = true;
            if (ps.compute_state_buffer.id == 0) ps.compute_buffers_pending = true;
            installComputeCreated(scene, ps, up);
            continue;
        }
        installComputeCreated(scene, ps, up);
        // Advance the live ring by the consumed window. The guard is a
        // PAIR that cannot alias (see computeWindowMatches): the frozen
        // stage_base locates the live window start, and the byte match
        // proves the oldest live records ARE the dispatched ones. A bare
        // base compare would alias when exactly `cap` post-freeze appends
        // with bulk evictions return the base to the same value with
        // completely replaced contents — the content check closes that
        // hole with zero new state. On mismatch the window is left intact
        // and the re-armed flush_pending (set by the updates that
        // appended) re-freezes it whole on the next build — delayed,
        // never lost.
        const cap = up.capacity;
        if (up.consumed_staged > 0 and cap > 0 and
            ps.compute_stage_base == up.stage_base and
            ps.compute_staged >= up.consumed_staged and
            computeWindowMatches(ps, front, up))
        {
            ps.compute_stage_base = (ps.compute_stage_base + up.consumed_staged) % cap;
            ps.compute_staged -= up.consumed_staged;
        }
        if (up.consumed_dt != 0.0) {
            ps.compute_dt_accum = @max(0.0, ps.compute_dt_accum - up.consumed_dt);
        }
    }
}

/// Content-equality half of the compute ring-advance guard (game side —
/// full live access, no lock-free constraint here). Returns true only if
/// the oldest `consumed_staged` live records starting at the frozen
/// `stage_base` are byte-identical to the dispatched frozen bytes. A
/// base-index compare alone aliases on a full wrap (exactly `cap`
/// post-freeze appends with bulk evictions return the base to the same
/// value with completely replaced contents); identical bytes cannot
/// alias — equal content IS the same window, so advancing it is exact.
/// Any mismatch (eviction, refill, torn length) fails closed: the caller
/// skips the advance and the re-armed flush_pending re-freezes the live
/// window whole on the next build.
fn computeWindowMatches(ps: anytype, front: anytype, up: anytype) bool {
    const c: usize = up.consumed_staged;
    const cap: usize = up.capacity;
    if (c == 0 or cap == 0) return false;
    if (c > up.data_count) return false;
    const end: usize = up.data_lo + c;
    if (end > front.p_compute_data.items.len) return false;
    if (ps.compute_staging.len < cap) return false;
    const frozen = front.p_compute_data.items[up.data_lo..end];
    var i: usize = 0;
    while (i < c) : (i += 1) {
        const live = ps.compute_staging[(up.stage_base + i) % cap];
        if (!std.meta.eql(live, frozen[i])) return false;
    }
    return true;
}

/// Installs created compute GPU objects over zero live ids (game-side
/// commit); retires anything that cannot install. Both the delivered and
/// the undelivered (partial-progress) paths land here: a failed creation
/// records its partial handles in the outcome so the retry completes the
/// rest instead of leaking them.
fn installComputeCreated(scene: anytype, ps: anytype, up: anytype) void {
    if (up.created_state_buffer_id != 0) {
        if (ps.compute_state_buffer.id == 0) {
            ps.compute_state_buffer = .{ .id = up.created_state_buffer_id };
        } else {
            retireCreated(scene, up.created_state_buffer_id);
        }
        up.created_state_buffer_id = 0;
    }
    if (up.created_spawn_buffer_id != 0) {
        if (ps.compute_spawn_buffer.id == 0) {
            ps.compute_spawn_buffer = .{ .id = up.created_spawn_buffer_id };
        } else {
            retireCreated(scene, up.created_spawn_buffer_id);
        }
        up.created_spawn_buffer_id = 0;
    }
    if (up.created_draw_buffer_id != 0) {
        if (ps.compute_draw_buffer.id == 0) {
            ps.compute_draw_buffer = .{ .id = up.created_draw_buffer_id };
        } else {
            retireCreated(scene, up.created_draw_buffer_id);
        }
        up.created_draw_buffer_id = 0;
    }
    if (up.created_state_view_id != 0) {
        if (ps.compute_state_view.id == 0) {
            ps.compute_state_view = .{ .id = up.created_state_view_id };
        } else {
            retireComputeView(up.created_state_view_id);
        }
        up.created_state_view_id = 0;
    }
    if (up.created_spawn_view_id != 0) {
        if (ps.compute_spawn_view.id == 0) {
            ps.compute_spawn_view = .{ .id = up.created_spawn_view_id };
        } else {
            retireComputeView(up.created_spawn_view_id);
        }
        up.created_spawn_view_id = 0;
    }
    if (up.created_draw_view_id != 0) {
        if (ps.compute_draw_view.id == 0) {
            ps.compute_draw_view = .{ .id = up.created_draw_view_id };
        } else {
            retireComputeView(up.created_draw_view_id);
        }
        up.created_draw_view_id = 0;
    }
    if (up.created_shader_id != 0) {
        if (ps.compute_shader.id == 0) {
            ps.compute_shader = .{ .id = up.created_shader_id };
        } else {
            retireComputeShader(up.created_shader_id);
        }
        up.created_shader_id = 0;
    }
    if (up.created_pipeline_id != 0) {
        if (ps.compute_pipeline.id == 0) {
            ps.compute_pipeline = .{ .id = up.created_pipeline_id };
        } else {
            retireComputePipeline(up.created_pipeline_id);
        }
        up.created_pipeline_id = 0;
    }
    // A fully-created set resolves the pending request (mirrors
    // ensureComputeGpu, which clears the flag once every object exists).
    if (ps.compute_state_buffer.id != 0 and ps.compute_spawn_buffer.id != 0 and
        ps.compute_draw_buffer.id != 0 and ps.compute_state_view.id != 0 and
        ps.compute_spawn_view.id != 0 and ps.compute_draw_view.id != 0 and
        ps.compute_shader.id != 0 and ps.compute_pipeline.id != 0)
    {
        ps.compute_buffers_pending = false;
    }
}

/// Views/pipelines/shaders cannot travel through the buffer retire queue
/// (tripwire P6: no new queues) and their destroys are context-thread
/// only. Reaching here means a duplicate creation raced the commit —
/// defensive only (in practice only the context creates, sequentially on
/// one thread, so the live id is always zero when a created id exists).
/// Destroying inline game-side would violate the sg-thread contract, and
/// leaking is bounded (one handle per race); log loudly instead of going
/// silent. If this ever fires, the creation protocol needs a rethink.
fn retireComputeView(id: u32) void {
    std.log.err("upload_packets: duplicate compute view created (id {}), leaking (see installComputeCreated)", .{id});
}

fn retireComputeShader(id: u32) void {
    std.log.err("upload_packets: duplicate compute shader created (id {}), leaking (see installComputeCreated)", .{id});
}

fn retireComputePipeline(id: u32) void {
    std.log.err("upload_packets: duplicate compute pipeline created (id {}), leaking (see installComputeCreated)", .{id});
}

fn retireComputeCreated(scene: anytype, up: anytype) void {
    retireCreated(scene, up.created_state_buffer_id);
    retireCreated(scene, up.created_spawn_buffer_id);
    retireCreated(scene, up.created_draw_buffer_id);
    // Views/shader/pipeline have no retire path: the owner is gone, so
    // these handles (if any) leak — log loudly instead of going silent
    // (same rationale as installComputeCreated).
    if (up.created_state_view_id != 0) retireComputeView(up.created_state_view_id);
    if (up.created_spawn_view_id != 0) retireComputeView(up.created_spawn_view_id);
    if (up.created_draw_view_id != 0) retireComputeView(up.created_draw_view_id);
    if (up.created_shader_id != 0) retireComputeShader(up.created_shader_id);
    if (up.created_pipeline_id != 0) retireComputePipeline(up.created_pipeline_id);
    up.created_state_buffer_id = 0;
    up.created_spawn_buffer_id = 0;
    up.created_draw_buffer_id = 0;
    up.created_state_view_id = 0;
    up.created_spawn_view_id = 0;
    up.created_draw_view_id = 0;
    up.created_shader_id = 0;
    up.created_pipeline_id = 0;
}

fn commitTrails(scene: anytype, front: anytype) void {
    for (front.trail_uploads.items) |*up| {
        const tm = commitTrailAt(scene, up.trail_index, up.token) orelse {
            retireCreated(scene, up.created_vertex_buffer_id);
            retireCreated(scene, up.created_index_buffer_id);
            up.created_vertex_buffer_id = 0;
            up.created_index_buffer_id = 0;
            continue;
        };
        if (!up.delivered) {
            tm.gpu_dirty = true;
            if (tm.mesh.vertex_buffer.id == 0 or tm.mesh.index_buffer.id == 0) tm.buffers_pending = true;
            continue;
        }
        if (up.created_vertex_buffer_id != 0 or up.created_index_buffer_id != 0) {
            if (tm.mesh.vertex_buffer.id == 0 and tm.mesh.index_buffer.id == 0) {
                tm.mesh.vertex_buffer = .{ .id = up.created_vertex_buffer_id };
                tm.mesh.index_buffer = .{ .id = up.created_index_buffer_id };
                tm.buffers_pending = false;
            } else {
                retireCreated(scene, up.created_vertex_buffer_id);
                retireCreated(scene, up.created_index_buffer_id);
            }
            up.created_vertex_buffer_id = 0;
            up.created_index_buffer_id = 0;
        }
        // NOTE: no zero-target re-arm here (unlike particles): trail
        // buffers are only ever created from zero and nothing ever zeroes
        // them back, so a delivered outcome with zero live ids and no
        // created handles is the never-created manual/deferred fixture —
        // the legacy flush likewise skips the upload and still publishes
        // the scalars below.
        // Scalar publish (index_count/bounds), skipped when a newer
        // producer mutation is already pending (live gpu_dirty set after
        // the stage): the fresher generation's own stage+flush will
        // publish — overwriting it here with the older frozen values
        // would clobber one frame of freshness, never coherence.
        if (!tm.gpu_dirty) {
            tm.mesh.index_count = @intCast(up.index_count);
            tm.mesh.local_bounding_box = math.BoundingBox.init(
                math.Vec3.new(up.min_pt[0], up.min_pt[1], up.min_pt[2]),
                math.Vec3.new(up.max_pt[0], up.max_pt[1], up.max_pt[2]),
            );
            tm.mesh.cached_aabb = tm.mesh.local_bounding_box;
        }
    }
}

fn commitSoftbodies(scene: anytype, front: anytype) void {
    for (front.soft_uploads.items) |*up| {
        const b = commitSoftAt(scene, up.body_index, up.token) orelse {
            retireCreated(scene, up.created_vertex_buffer_id);
            retireCreated(scene, up.created_index_buffer_id);
            up.created_vertex_buffer_id = 0;
            up.created_index_buffer_id = 0;
            continue;
        };
        if (!up.delivered) {
            b.upload_pending = true;
            if (b.mesh.vertex_buffer.id == 0) b.buffers_pending = true;
            continue;
        }
        if (up.created_vertex_buffer_id != 0 or up.created_index_buffer_id != 0) {
            if (b.mesh.vertex_buffer.id == 0) {
                b.mesh.vertex_buffer = .{ .id = up.created_vertex_buffer_id };
                b.mesh.index_buffer = .{ .id = up.created_index_buffer_id };
                b.buffers_pending = false;
            } else {
                retireCreated(scene, up.created_vertex_buffer_id);
                retireCreated(scene, up.created_index_buffer_id);
            }
            up.created_vertex_buffer_id = 0;
            up.created_index_buffer_id = 0;
        }
        // NOTE: no zero-target re-arm (see commitTrails): soft buffers are
        // only ever created from zero and nothing ever zeroes them back;
        // the legacy flush likewise skips the upload and consumes the
        // generation. (Soft publishes no scalars.)
    }
}

fn commitGreased(scene: anytype, front: anytype) void {
    for (front.greased_uploads.items) |*up| {
        const gl = commitGreasedAt(scene, up.line_index, up.token) orelse {
            retireCreated(scene, up.created_vertex_buffer_id);
            retireCreated(scene, up.created_index_buffer_id);
            up.created_vertex_buffer_id = 0;
            up.created_index_buffer_id = 0;
            continue;
        };
        if (!up.delivered) {
            gl.gpu_dirty = true;
            continue;
        }
        if (up.created_vertex_buffer_id != 0 or up.created_index_buffer_id != 0) {
            if (gl.mesh.vertex_buffer.id == 0 and gl.mesh.index_buffer.id == 0) {
                gl.mesh.vertex_buffer = .{ .id = up.created_vertex_buffer_id };
                gl.mesh.index_buffer = .{ .id = up.created_index_buffer_id };
            } else {
                retireCreated(scene, up.created_vertex_buffer_id);
                retireCreated(scene, up.created_index_buffer_id);
            }
            up.created_vertex_buffer_id = 0;
            up.created_index_buffer_id = 0;
        } else if (up.vert_count > 0 and gl.mesh.vertex_buffer.id == 0) {
            gl.gpu_dirty = true;
            continue;
        }
        // Full-upload flag: same protocol as trails (producer mutations
        // always set gpu_dirty; the full flag is only ever set at
        // creation/init).
        if (up.full_delivered and !gl.gpu_dirty) gl.gpu_needs_full_upload = false;
    }
}

fn commitPendingCreations(scene: anytype, front: anytype) void {
    for (front.pending_uploads.items) |*up| {
        const m = commitMeshAt(scene, up.mesh_index, up.token, up.uid) orelse {
            retireCreated(scene, up.created_vertex_buffer_id);
            retireCreated(scene, up.created_index_buffer_id);
            up.created_vertex_buffer_id = 0;
            up.created_index_buffer_id = 0;
            // Images/views cannot travel through the buffer retire queue
            // (same defensive-only precedent as installComputeCreated):
            // the owner is gone, so these (if any) leak — log loudly.
            if (up.created_delta_image_id != 0 or up.created_delta_view_id != 0) {
                std.log.err("upload_packets: orphaned morph delta texture created (image {} view {}), leaking (see commitPendingCreations)", .{ up.created_delta_image_id, up.created_delta_view_id });
            }
            up.created_delta_image_id = 0;
            up.created_delta_view_id = 0;
            continue;
        };
        if (!up.delivered) {
            m.gpu_pending = true;
            if (up.morph_delta_pending) m.morph_upload_pending = true;
            continue;
        }
        if (up.created_vertex_buffer_id == 0 or up.created_index_buffer_id == 0) {
            // Delivered without handles (should not happen: the flush only
            // delivers after creating both): re-arm instead of stranding
            // the mesh with cleared flags and no buffers.
            retireCreated(scene, up.created_vertex_buffer_id);
            retireCreated(scene, up.created_index_buffer_id);
            up.created_vertex_buffer_id = 0;
            up.created_index_buffer_id = 0;
            up.created_delta_image_id = 0;
            up.created_delta_view_id = 0;
            m.gpu_pending = true;
            if (up.morph_delta_pending) m.morph_upload_pending = true;
            continue;
        }
        if (m.vertex_buffer.id != 0 or m.index_buffer.id != 0) {
            retireCreated(scene, up.created_vertex_buffer_id);
            retireCreated(scene, up.created_index_buffer_id);
            up.created_vertex_buffer_id = 0;
            up.created_index_buffer_id = 0;
            up.created_delta_image_id = 0;
            up.created_delta_view_id = 0;
            m.gpu_pending = true;
            if (up.morph_delta_pending) m.morph_upload_pending = true;
            continue;
        }
        // Delta texture without handles (should not happen: the flush only
        // delivers a delta request with a valid image + view): retire the
        // buffers and re-arm both flags instead of installing a mesh the
        // queue would panic on (vsUniforms requires the delta view).
        if (up.morph_delta_pending and (up.created_delta_image_id == 0 or up.created_delta_view_id == 0)) {
            retireCreated(scene, up.created_vertex_buffer_id);
            retireCreated(scene, up.created_index_buffer_id);
            up.created_vertex_buffer_id = 0;
            up.created_index_buffer_id = 0;
            up.created_delta_image_id = 0;
            up.created_delta_view_id = 0;
            m.gpu_pending = true;
            m.morph_upload_pending = true;
            continue;
        }
        m.vertex_buffer = .{ .id = up.created_vertex_buffer_id };
        m.index_buffer = .{ .id = up.created_index_buffer_id };
        up.created_vertex_buffer_id = 0;
        up.created_index_buffer_id = 0;
        if (m.vertex_count == 0 and up.vert_count > 0) m.vertex_count = @intCast(up.vert_count);
        // Staged GPU-morph delta texture (see the module header): installed
        // atomically with the base buffers above, so the mesh leaves
        // gpu_pending fully drawable — no base-pose frame, no missing-view
        // panic. The legacy fallback finish still owns meshes that never
        // went through a staged build.
        if (up.morph_delta_pending) {
            if (m.morph_delta_image.id != 0 or m.morph_delta_view.id != 0) {
                // Defensive only (in practice only the context creates, so
                // the live ids are zero here): the live delta texture
                // already exists, so the mesh is complete — drop the
                // created pair loudly (no image/view retire queue, same
                // precedent as installComputeCreated) and finish normally.
                std.log.err("upload_packets: duplicate morph delta texture created (image {} view {}), leaking (see commitPendingCreations)", .{ up.created_delta_image_id, up.created_delta_view_id });
                up.created_delta_image_id = 0;
                up.created_delta_view_id = 0;
                m.morph_upload_pending = false;
            } else {
                m.morph_delta_image = .{ .id = up.created_delta_image_id };
                m.morph_delta_view = .{ .id = up.created_delta_view_id };
                up.created_delta_image_id = 0;
                up.created_delta_view_id = 0;
                m.morph_tex_width = up.delta_width;
                m.morph_tex_height = up.delta_height;
                m.morph_upload_pending = false;
            }
        }
        if (up.dynamic_update) m.morph_upload_needed = true;
        m.pending_dynamic_update = false;
        // The retained CPU mirrors (cpu_positions/cpu_indices) stay — the
        // legacy finish keeps them too; only the consumed pending copy is
        // freed, game-side (the allocator is thread-safe).
        if (m.pending_vertices.len > 0) {
            scene.allocator.free(m.pending_vertices);
            m.pending_vertices = &.{};
        }
    }
}

// --- Focused regression tests (headless, sg-free assertions on packets) ---

test "upload packets: morph freeze survives live mutation, slot reuse retains capacity" {
    const t = std.testing;
    const mesh_types = @import("../mesh/types.zig");
    const frame_draws = @import("frame_draws.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var base = [_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) };
    base[0].position = .{ 1, 2, 3 };
    var staging = [_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) };
    staging[0].position = .{ 7, 8, 9 };
    var mesh = @import("../mesh/mesh.zig").Mesh{
        .name = "m",
        .vertex_buffer = .{ .id = 5 },
        .index_buffer = .{},
        .index_count = 0,
        .morph_base = &base,
        .morph_staging = &staging,
        .morph_upload_needed = true,
    };
    _ = mesh.ensureUid();
    var meshes = [_]*@TypeOf(mesh){&mesh};
    const MeshType = @TypeOf(mesh);
    var no_systems: []*@import("../particles/system.zig").ParticleSystem = &.{};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    const empty_meshes: []*MeshType = &.{};
    _ = empty_meshes;
    var fake_scene = .{
        .allocator = t.allocator,
        .meshes = .{ .items = meshes[0..], .capacity = 1 },
        .particles = .{ .systems = .{ .items = no_systems[0..], .capacity = 0 } },
        .trails = .{ .meshes = .{ .items = no_trails[0..], .capacity = 0 } },
        .softbodies = .{ .bodies = .{ .items = no_bodies[0..], .capacity = 0 } },
        .greased_lines = .{ .items = no_lines[0..], .capacity = 0 },
    };
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.morph_uploads.items.len);
    try t.expectEqual(@as(u32, 5), slot.morph_uploads.items[0].buffer_id);
    try t.expectEqual([3]f32{ 7, 8, 9 }, slot.morph_data.items[0].position);

    // Live mutation after the freeze must not reach the packet.
    staging[0].position = .{ 99, 99, 99 };
    mesh.vertex_buffer = .{ .id = 6 };
    try t.expectEqual([3]f32{ 7, 8, 9 }, slot.morph_data.items[0].position);
    try t.expectEqual(@as(u32, 5), slot.morph_uploads.items[0].buffer_id);

    const cap = slot.morph_data.capacity;
    try t.expect(cap >= 2);
    slot.reset();
    try t.expectEqual(@as(usize, 0), slot.morph_uploads.items.len);
    try t.expectEqual(cap, slot.morph_data.capacity);
    // Reuse after reset: newest wins, no leak (deinit above frees).
    mesh.morph_upload_needed = true;
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.morph_uploads.items.len);
}

test "upload packets: particle cpu freeze survives count/handle mutation" {
    const t = std.testing;
    const sys_mod = @import("../particles/system.zig");
    const frame_draws = @import("frame_draws.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var ps = try sys_mod.makeTestSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
    ps.instance_buffer = .{ .id = 11 };
    ps.active_count = 2;
    ps.instances[0].pos_size[0] = 1.0;
    ps.instances[1].pos_size[0] = 2.0;
    ps.instance_dirty = true;
    var systems = [_]*sys_mod.ParticleSystem{&ps};
    const layer = .{ .systems = .{ .items = systems[0..], .capacity = 1 } };
    const MeshType2 = @import("../mesh/mesh.zig").Mesh;
    var no_meshes2: []*MeshType2 = &.{};
    var no_trails2: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies2: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines2: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var fake_scene = .{ .allocator = t.allocator, .particles = layer, .meshes = .{ .items = no_meshes2[0..], .capacity = 0 }, .trails = .{ .meshes = .{ .items = no_trails2[0..], .capacity = 0 } }, .softbodies = .{ .bodies = .{ .items = no_bodies2[0..], .capacity = 0 } }, .greased_lines = .{ .items = no_lines2[0..], .capacity = 0 } };
    // Stage only the particle path via the shared entry (other lists empty).
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.p_cpu_uploads.items.len);
    try t.expectEqual(@as(u32, 2), slot.p_cpu_uploads.items[0].count);

    ps.active_count = 0;
    ps.instance_buffer = .{ .id = 99 };
    ps.instances[0].pos_size[0] = 42.0;
    try t.expectEqual(@as(u32, 2), slot.p_cpu_uploads.items[0].count);
    try t.expectEqual(@as(u32, 11), slot.p_cpu_uploads.items[0].buffer_id);
    try t.expectEqual(@as(f32, 1.0), slot.p_cpu_data.items[0].pos_size[0]);
}

test "upload packets: trail freeze survives live vertex mutation" {
    const t = std.testing;
    const mesh_types = @import("../mesh/types.zig");
    const frame_draws = @import("frame_draws.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var verts = [_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) };
    verts[0].position = .{ 3, 4, 5 };
    var idx = [_]u16{ 0, 1, 0 };
    var mesh = @import("../mesh/mesh.zig").Mesh{ .name = "tm", .vertex_buffer = .{ .id = 7 }, .index_buffer = .{ .id = 8 }, .index_count = 0 };
    var tm = @import("../mesh/trail.zig").TrailMesh{
        .allocator = t.allocator,
        .scene = undefined,
        .mesh = &mesh,
        .options = .{},
        .vertices = &verts,
        .indices = &idx,
        .gpu_dirty = true,
        .pending_vertex_count = 2,
        .pending_index_count = 3,
        .pending_min_pt = math.Vec3.new(0, 0, 0),
        .pending_max_pt = math.Vec3.new(1, 1, 1),
    };
    var list = [_]*@TypeOf(tm){&tm};
    const MeshType3 = @import("../mesh/mesh.zig").Mesh;
    var no_meshes3: []*MeshType3 = &.{};
    const SysType3 = @import("../particles/system.zig").ParticleSystem;
    var no_systems3: []*SysType3 = &.{};
    var no_bodies3: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines3: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var fake_scene = .{
        .allocator = t.allocator,
        .meshes = .{ .items = no_meshes3[0..], .capacity = 0 },
        .particles = .{ .systems = .{ .items = no_systems3[0..], .capacity = 0 } },
        .trails = .{ .meshes = .{ .items = list[0..], .capacity = 1 } },
        .softbodies = .{ .bodies = .{ .items = no_bodies3[0..], .capacity = 0 } },
        .greased_lines = .{ .items = no_lines3[0..], .capacity = 0 },
    };
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.trail_uploads.items.len);
    verts[0].position = .{ 50, 50, 50 };
    try t.expectEqual([3]f32{ 3, 4, 5 }, slot.trail_verts.items[0].position);
    try t.expectEqual(@as(usize, 2), slot.trail_uploads.items[0].vert_count);
}

test "upload packets: gpu range freeze captures the wrapped prefix" {
    const t = std.testing;
    const sys_mod = @import("../particles/system.zig");
    const slot_types = @import("../particles/types.zig");
    const frame_draws = @import("frame_draws.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var ps = try sys_mod.makeTestSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
    ps.gpu_slots = try t.allocator.alloc(slot_types.GpuParticleSlot, 4);
    ps.gpu_slots[0].spawn_pos_time = .{ 0, 0, 0, 4 };
    ps.gpu_slots[1].spawn_pos_time = .{ 0, 0, 0, 5 };
    ps.gpu_slots[2].spawn_pos_time = .{ 0, 0, 0, 2 };
    ps.gpu_slots[3].spawn_pos_time = .{ 0, 0, 0, 3 };
    ps.gpu_dirty = true;
    ps.gpu_dirty_wrapped = true;
    ps.gpu_high_water = 4;
    ps.gpu_flush_pending = true;
    ps.gpu_slot_buffer = .{ .id = 22 };
    var systems = [_]*sys_mod.ParticleSystem{&ps};
    const layer = .{ .systems = .{ .items = systems[0..], .capacity = 1 } };
    const MeshType = @import("../mesh/mesh.zig").Mesh;
    var no_meshes: []*MeshType = &.{};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var fake_scene = .{ .allocator = t.allocator, .particles = layer, .meshes = .{ .items = no_meshes[0..], .capacity = 0 }, .trails = .{ .meshes = .{ .items = no_trails[0..], .capacity = 0 } }, .softbodies = .{ .bodies = .{ .items = no_bodies[0..], .capacity = 0 } }, .greased_lines = .{ .items = no_lines[0..], .capacity = 0 } };
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.p_gpu_uploads.items.len);
    try t.expectEqual(@as(u32, 4), slot.p_gpu_uploads.items[0].count);

    // Live ring mutation after the freeze must not reach the packet.
    ps.gpu_slots[0].spawn_pos_time = .{ 9, 9, 9, 9 };
    try t.expectEqual([4]f32{ 0, 0, 0, 4 }, slot.p_gpu_data.items[0].spawn_pos_time);
    try t.expectEqual([4]f32{ 0, 0, 0, 3 }, slot.p_gpu_data.items[3].spawn_pos_time);
}

test "upload packets: greased freeze captures verts and indices" {
    const t = std.testing;
    const mesh_types = @import("../mesh/types.zig");
    const frame_draws = @import("frame_draws.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var verts = [_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) };
    verts[1].position = .{ 6, 7, 8 };
    var idx = [_]u32{ 0, 1, 0, 1, 0, 1 };
    var mesh = @import("../mesh/mesh.zig").Mesh{ .name = "gl", .vertex_buffer = .{ .id = 31 }, .index_buffer = .{ .id = 32 }, .index_count = 0 };
    var gl = @import("../mesh/greased_line.zig").GreasedLineMesh{
        .allocator = t.allocator,
        .scene = undefined,
        .mesh = &mesh,
        .options = .{},
        .vertices = &verts,
        .indices = &idx,
        .gpu_dirty = true,
        .gpu_needs_full_upload = true,
    };
    var list = [_]*@TypeOf(gl){&gl};
    const MeshType = @import("../mesh/mesh.zig").Mesh;
    var no_meshes: []*MeshType = &.{};
    const SysType = @import("../particles/system.zig").ParticleSystem;
    var no_systems: []*SysType = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var fake_scene = .{
        .allocator = t.allocator,
        .meshes = .{ .items = no_meshes[0..], .capacity = 0 },
        .particles = .{ .systems = .{ .items = no_systems[0..], .capacity = 0 } },
        .trails = .{ .meshes = .{ .items = no_trails[0..], .capacity = 0 } },
        .softbodies = .{ .bodies = .{ .items = no_bodies[0..], .capacity = 0 } },
        .greased_lines = .{ .items = list[0..], .capacity = 1 },
    };
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.greased_uploads.items.len);
    try t.expect(slot.greased_uploads.items[0].full_upload);
    try t.expectEqual(@as(usize, 6), slot.greased_uploads.items[0].index_count);

    verts[1].position = .{ 50, 50, 50 };
    idx[0] = 99;
    try t.expectEqual([3]f32{ 6, 7, 8 }, slot.greased_verts.items[1].position);
    try t.expectEqual(@as(u32, 0), slot.greased_indices.items[0]);
}

test "upload packets: staged flush re-arms undeliverable buffer-pending particle uploads" {
    // Finding 1: the old flush cleared the dirty flags before the
    // buffer-id checks, so a deferred system's first-frame bytes were
    // skipped after creation and never retried. Phase 2 ownership: the
    // STAGE clears the flags, the headless flush records undelivered
    // without touching live state, and the game-side COMMIT re-arms the
    // flags for retry. The frozen bytes stay intact throughout.
    const t = std.testing;
    gpu_thread.markContextThread();
    const sys_mod = @import("../particles/system.zig");
    const slot_types = @import("../particles/types.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var ps = try sys_mod.makeTestSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
    ps.instance_buffer = .{};
    ps.instance_buffer_pending = true;
    ps.active_count = 2;
    ps.instances[0].pos_size[0] = 1.0;
    ps.instance_dirty = true;
    ps.gpu_slots = try t.allocator.alloc(slot_types.GpuParticleSlot, 4);
    ps.gpu_slots[0].spawn_pos_time = .{ 1, 2, 3, 0 };
    ps.gpu_slots[1].spawn_pos_time = .{ 4, 5, 6, 0 };
    ps.gpu_dirty = true;
    ps.gpu_dirty_start = 0;
    ps.gpu_dirty_end = 2;
    ps.gpu_high_water = 2;
    ps.gpu_slot_buffer = .{};
    ps.gpu_slot_buffer_pending = true;
    ps.gpu_flush_pending = true;
    var systems = [_]*sys_mod.ParticleSystem{&ps};
    const MeshType = @import("../mesh/mesh.zig").Mesh;
    var no_meshes: []*MeshType = &.{};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var retire: retire_mod.GpuRetireQueue = .{};
    defer retire.deinit(t.allocator);
    var fake_scene = .{
        .allocator = t.allocator,
        .meshes = .{ .items = no_meshes[0..], .capacity = 0 },
        .particles = .{ .systems = .{ .items = systems[0..], .capacity = 1 } },
        .trails = .{ .meshes = .{ .items = no_trails[0..], .capacity = 0 } },
        .softbodies = .{ .bodies = .{ .items = no_bodies[0..], .capacity = 0 } },
        .greased_lines = .{ .items = no_lines[0..], .capacity = 0 },
        .gpu_retire = &retire,
        .flush_in_prepare = true,
    };
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.p_cpu_uploads.items.len);
    try t.expectEqual(@as(usize, 1), slot.p_gpu_uploads.items.len);
    // Stage consumed the dirty flags (phase 2 ownership transfer).
    try t.expect(!ps.instance_dirty);
    try t.expect(!ps.gpu_dirty);

    // Headless flush: no context exists, so nothing can be delivered —
    // and the flush must not touch live state itself.
    flushSlotUploads(&fake_scene, &slot);
    try t.expect(!slot.p_cpu_uploads.items[0].delivered);
    try t.expect(!slot.p_gpu_uploads.items[0].delivered);
    try t.expect(!ps.instance_dirty);
    try t.expect(!ps.gpu_dirty);

    // Game-side commit: the undelivered outcomes re-arm every flag for
    // retry, and the frozen bytes are intact for the next freeze.
    commitSlotResults(&fake_scene, &slot);
    try t.expect(ps.instance_dirty);
    try t.expect(ps.instance_buffer_pending);
    try t.expect(ps.gpu_dirty);
    try t.expect(ps.gpu_flush_pending);
    try t.expect(ps.gpu_slot_buffer_pending);
    try t.expectEqual(@as(f32, 1.0), slot.p_cpu_data.items[0].pos_size[0]);
    try t.expectEqual([4]f32{ 1, 2, 3, 0 }, slot.p_gpu_data.items[0].spawn_pos_time);
}

test "upload packets: token mismatch fail-closes, previous state stands" {
    const t = std.testing;
    gpu_thread.markContextThread();
    const mesh_types = @import("../mesh/types.zig");
    const sys_mod = @import("../particles/system.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var base = [_]mesh_types.Vertex{std.mem.zeroes(mesh_types.Vertex)};
    var staging = [_]mesh_types.Vertex{std.mem.zeroes(mesh_types.Vertex)};
    staging[0].position = .{ 7, 8, 9 };
    var mesh = @import("../mesh/mesh.zig").Mesh{
        .name = "m",
        .vertex_buffer = .{ .id = 5 },
        .index_buffer = .{},
        .index_count = 0,
        .morph_base = &base,
        .morph_staging = &staging,
        .morph_upload_needed = true,
    };
    _ = mesh.ensureUid();
    var ps = try sys_mod.makeTestSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
    ps.instance_buffer = .{ .id = 11 };
    ps.active_count = 1;
    ps.instance_dirty = true;
    var meshes = [_]*@TypeOf(mesh){&mesh};
    var systems = [_]*sys_mod.ParticleSystem{&ps};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var retire: retire_mod.GpuRetireQueue = .{};
    defer retire.deinit(t.allocator);
    var fake_scene = .{
        .allocator = t.allocator,
        .meshes = .{ .items = meshes[0..], .capacity = 1 },
        .particles = .{ .systems = .{ .items = systems[0..], .capacity = 1 } },
        .trails = .{ .meshes = .{ .items = no_trails[0..], .capacity = 0 } },
        .softbodies = .{ .bodies = .{ .items = no_bodies[0..], .capacity = 0 } },
        .greased_lines = .{ .items = no_lines[0..], .capacity = 0 },
        .gpu_retire = &retire,
        .flush_in_prepare = true,
    };
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.morph_uploads.items.len);
    try t.expectEqual(@as(usize, 1), slot.p_cpu_uploads.items.len);

    // Stale tokens (owner replaced between build and latch): the flush
    // records outcomes without touching live state, and the commit skips
    // both packets (pointer compare first — never dereferenced). The
    // stage already consumed the old owners' flags; the commit must NOT
    // re-arm flags on the different live owners at those indices, so the
    // replacement owners' state stands exactly as it was.
    slot.morph_uploads.items[0].token +%= 1;
    slot.p_cpu_uploads.items[0].token +%= 1;
    flushSlotUploads(&fake_scene, &slot);
    commitSlotResults(&fake_scene, &slot);
    try t.expect(!mesh.morph_upload_needed);
    try t.expect(!ps.instance_dirty);
    try t.expect(!slot.morph_uploads.items[0].delivered);
    try t.expect(!slot.p_cpu_uploads.items[0].delivered);
}

test "upload packets: lock-free ownership audit — flush writes no live state, commit writes only the documented set" {
    // Phase 2 (stage A.7) write-audit: with the producer exclusion OFF the
    // context path (flushSlotUploads) must perform NO writes to
    // game-owned fields — flags, scalars, handles, arrays alike — for ALL
    // eight owners (morph, particle CPU, particle compute, particle GPU
    // range, trail, greased, softbody, pending-mesh). The game-side commit
    // may then write exactly the documented set (flag re-arms on
    // undelivered outcomes; handle installs + scalar publishes + compute
    // ring advance on delivered ones), never anything else. Sentinel
    // canaries prove both halves: any stray write changes a canary and
    // fails the test. A final negative phase proves the compute
    // content-match guard refuses a same-base/count packet with replaced
    // bytes.
    const t = std.testing;
    gpu_thread.markContextThread();
    const mesh_types = @import("../mesh/types.zig");
    const sys_mod = @import("../particles/system.zig");
    const slot_types = @import("../particles/types.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var base = [_]mesh_types.Vertex{std.mem.zeroes(mesh_types.Vertex)};
    var staging = [_]mesh_types.Vertex{std.mem.zeroes(mesh_types.Vertex)};
    staging[0].position = .{ 7, 8, 9 };
    var mesh = @import("../mesh/mesh.zig").Mesh{
        .name = "m",
        .vertex_buffer = .{ .id = 5 },
        .index_buffer = .{},
        .index_count = 0,
        .morph_base = &base,
        .morph_staging = &staging,
        .morph_upload_needed = true,
    };
    _ = mesh.ensureUid();
    var ps = try sys_mod.makeTestSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
    ps.instance_buffer = .{ .id = 11 };
    ps.active_count = 2;
    ps.instances[0].pos_size[0] = 1.0;
    ps.instance_dirty = true;
    var tverts = [_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) };
    tverts[0].position = .{ 3, 4, 5 };
    var tidx = [_]u16{ 0, 1, 0 };
    var tmesh = @import("../mesh/mesh.zig").Mesh{ .name = "tm", .vertex_buffer = .{ .id = 7 }, .index_buffer = .{ .id = 8 }, .index_count = 0 };
    var tm = @import("../mesh/trail.zig").TrailMesh{
        .allocator = t.allocator,
        .scene = undefined,
        .mesh = &tmesh,
        .options = .{},
        .vertices = &tverts,
        .indices = &tidx,
        .gpu_dirty = true,
        .pending_vertex_count = 2,
        .pending_index_count = 3,
        .pending_min_pt = math.Vec3.new(0, 0, 0),
        .pending_max_pt = math.Vec3.new(1, 1, 1),
    };
    // Compute owner (ring window + dispatch params, all frozen at stage).
    var cps = try sys_mod.makeComputeSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&cps);
    cps.compute_staging = try t.allocator.alloc(slot_types.GpuParticleSlot, 4);
    cps.compute_staging[0].spawn_pos_time = .{ 5, 6, 7, 8 };
    cps.compute_staging[1].spawn_pos_time = .{ 1, 2, 3, 4 };
    cps.compute_staged = 2;
    cps.compute_stage_base = 0;
    cps.compute_cursor = 2;
    cps.compute_high_water = 2;
    cps.compute_dt_accum = 0.016;
    cps.compute_flush_pending = true;
    cps.gravity = math.Vec3.new(0, -1, 0);
    cps.drag = 0.5;
    // GPU-range owner (wrapped upload range, frozen at stage).
    var ps2 = try sys_mod.makeTestSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps2);
    ps2.gpu_slots = try t.allocator.alloc(slot_types.GpuParticleSlot, 4);
    ps2.gpu_slots[0].spawn_pos_time = .{ 11, 12, 13, 14 };
    ps2.gpu_slots[1].spawn_pos_time = .{ 15, 16, 17, 18 };
    ps2.gpu_dirty = true;
    ps2.gpu_dirty_start = 0;
    ps2.gpu_dirty_end = 2;
    ps2.gpu_high_water = 2;
    ps2.gpu_flush_pending = true;
    ps2.gpu_slot_buffer = .{ .id = 22 };
    // Greased-line owner (full-index flag, frozen at stage).
    var gverts = [_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) };
    gverts[1].position = .{ 6, 7, 8 };
    var gidx = [_]u32{ 0, 1, 0, 1, 0, 1 };
    var gmesh = @import("../mesh/mesh.zig").Mesh{ .name = "gl", .vertex_buffer = .{ .id = 31 }, .index_buffer = .{ .id = 32 }, .index_count = 0 };
    var gl2 = @import("../mesh/greased_line.zig").GreasedLineMesh{
        .allocator = t.allocator,
        .scene = undefined,
        .mesh = &gmesh,
        .options = .{},
        .vertices = &gverts,
        .indices = &gidx,
        .gpu_dirty = true,
        .gpu_needs_full_upload = true,
    };
    // Softbody owner (whole-grid verts + u32 indices, frozen at stage).
    // cloth/material are never touched by the stage/flush/commit paths
    // (only upload_pending/vertices/indices/mesh/buffers_pending are).
    var sverts = try t.allocator.dupe(mesh_types.Vertex, &[_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) });
    defer t.allocator.free(sverts);
    sverts[0].position = .{ 2, 3, 4 };
    const sidx = try t.allocator.dupe(u32, &[_]u32{ 0, 1, 2 });
    defer t.allocator.free(sidx);
    var smesh = @import("../mesh/mesh.zig").Mesh{ .name = "sb", .vertex_buffer = .{ .id = 41 }, .index_buffer = .{ .id = 42 }, .index_count = 0 };
    var body = @import("../softbody.zig").SoftBody{
        .allocator = t.allocator,
        .cloth = undefined,
        .mesh = &smesh,
        .material = undefined,
        .vertices = sverts,
        .indices = sidx,
        .buffers_pending = false,
        .upload_pending = true,
    };
    // Deferred-creation owner (pending geometry + dynamic flag).
    // pending_vertices is freed by the delivered commit (guarded defer
    // below reads the LIVE field the commit clears — freeing the local
    // unconditionally would double-free); cpu_indices mirrors are always
    // retained, like the legacy finish.
    var pmesh = @import("../mesh/mesh.zig").Mesh{
        .name = "pm",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .gpu_pending = true,
        .pending_dynamic_update = true,
    };
    var pverts = try t.allocator.dupe(mesh_types.Vertex, &[_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) });
    defer if (pmesh.pending_vertices.len > 0) t.allocator.free(pmesh.pending_vertices);
    pverts[0].position = .{ 8, 8, 8 };
    const pidx = try t.allocator.dupe(u32, &[_]u32{ 0, 1, 2 });
    defer t.allocator.free(pidx);
    pmesh.pending_vertices = pverts;
    pmesh.cpu_indices = pidx;
    _ = pmesh.ensureUid();
    var meshes = [_]*@TypeOf(mesh){ &mesh, &pmesh };
    var systems = [_]*sys_mod.ParticleSystem{ &ps, &cps, &ps2 };
    var trails = [_]*@TypeOf(tm){&tm};
    var bodies = [_]*@TypeOf(body){&body};
    var lines = [_]*@TypeOf(gl2){&gl2};
    var retire: retire_mod.GpuRetireQueue = .{};
    defer retire.deinit(t.allocator);
    var fake_scene = .{
        .allocator = t.allocator,
        .meshes = .{ .items = meshes[0..], .capacity = 2 },
        .particles = .{ .systems = .{ .items = systems[0..], .capacity = 3 } },
        .trails = .{ .meshes = .{ .items = trails[0..], .capacity = 1 } },
        .softbodies = .{ .bodies = .{ .items = bodies[0..], .capacity = 1 } },
        .greased_lines = .{ .items = lines[0..], .capacity = 1 },
        .gpu_retire = &retire,
        .flush_in_prepare = true,
    };
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.morph_uploads.items.len);
    try t.expectEqual(@as(usize, 1), slot.p_cpu_uploads.items.len);
    try t.expectEqual(@as(usize, 1), slot.p_compute_uploads.items.len);
    try t.expectEqual(@as(usize, 1), slot.p_gpu_uploads.items.len);
    try t.expectEqual(@as(usize, 1), slot.trail_uploads.items.len);
    try t.expectEqual(@as(usize, 1), slot.greased_uploads.items.len);
    try t.expectEqual(@as(usize, 1), slot.soft_uploads.items.len);
    try t.expectEqual(@as(usize, 1), slot.pending_uploads.items.len);

    // Clobber every live staging array with canaries AFTER the freeze: if
    // the flush reads any live byte, the outcomes would carry 99s.
    staging[0].position = .{ 99, 99, 99 };
    ps.instances[0].pos_size[0] = 99.0;
    tverts[0].position = .{ 99, 99, 99 };
    cps.compute_staging[0].spawn_pos_time = .{ 99, 99, 99, 99 };
    cps.compute_staging[1].spawn_pos_time = .{ 99, 99, 99, 99 };
    cps.gravity = math.Vec3.new(9, 9, 9);
    ps2.gpu_slots[0].spawn_pos_time = .{ 99, 99, 99, 99 };
    ps2.gpu_slots[1].spawn_pos_time = .{ 99, 99, 99, 99 };
    gverts[1].position = .{ 99, 99, 99 };
    gidx[0] = 99;
    sverts[0].position = .{ 99, 99, 99 };
    pverts[0].position = .{ 99, 99, 99 };
    // Canary scalars/handles the commit must leave alone on undelivered.
    tmesh.index_count = 424242;
    mesh.vertex_count = 434343;

    // Headless flush: must write NOTHING live. Snapshot the full
    // game-owned surface before and compare after.
    const pre_morph_flag = mesh.morph_upload_needed;
    const pre_ps_dirty = ps.instance_dirty;
    const pre_tm_dirty = tm.gpu_dirty;
    flushSlotUploads(&fake_scene, &slot);
    try t.expectEqual(pre_morph_flag, mesh.morph_upload_needed);
    try t.expectEqual(@as(u32, 5), mesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 434343), mesh.vertex_count);
    try t.expectEqual(pre_ps_dirty, ps.instance_dirty);
    try t.expectEqual(@as(u32, 11), ps.instance_buffer.id);
    try t.expectEqual(@as(f32, 99.0), ps.instances[0].pos_size[0]);
    try t.expectEqual(pre_tm_dirty, tm.gpu_dirty);
    try t.expectEqual(@as(u32, 424242), tmesh.index_count);
    try t.expectEqual(@as(u32, 7), tmesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 8), tmesh.index_buffer.id);
    try t.expectEqual([3]f32{ 99, 99, 99 }, tverts[0].position);
    // Compute owner: ring, params, handles all untouched.
    try t.expect(!cps.compute_flush_pending);
    try t.expect(!cps.compute_state_clear_pending);
    try t.expect(!cps.compute_buffers_pending);
    try t.expect(!cps.compute_known_unsupported);
    try t.expectEqual(@as(u64, 0), cps.compute_dispatches);
    try t.expectEqual(@as(usize, 2), cps.compute_staged);
    try t.expectEqual(@as(usize, 0), cps.compute_stage_base);
    try t.expectEqual(@as(usize, 2), cps.compute_cursor);
    try t.expectEqual(@as(usize, 2), cps.compute_high_water);
    try t.expectEqual(@as(f32, 0.016), cps.compute_dt_accum);
    try t.expectEqual(@as(f32, 0.5), cps.drag);
    try t.expectEqual(math.Vec3.new(9, 9, 9), cps.gravity);
    try t.expectEqual([4]f32{ 99, 99, 99, 99 }, cps.compute_staging[0].spawn_pos_time);
    // GPU-range owner: flags, handles, ring untouched.
    try t.expect(!ps2.gpu_dirty);
    try t.expect(!ps2.gpu_dirty_wrapped);
    try t.expect(!ps2.gpu_flush_pending);
    try t.expect(!ps2.gpu_slot_buffer_pending);
    try t.expectEqual(@as(u32, 22), ps2.gpu_slot_buffer.id);
    try t.expectEqual(@as(usize, 2), ps2.gpu_high_water);
    try t.expectEqual([4]f32{ 99, 99, 99, 99 }, ps2.gpu_slots[0].spawn_pos_time);
    // Greased owner: dirty consumed at stage, full flag + handles stay.
    try t.expect(!gl2.gpu_dirty);
    try t.expect(gl2.gpu_needs_full_upload);
    try t.expectEqual(@as(u32, 31), gmesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 32), gmesh.index_buffer.id);
    try t.expectEqual(@as(u32, 99), gidx[0]);
    // Softbody owner: flags + handles untouched.
    try t.expect(!body.upload_pending);
    try t.expect(!body.buffers_pending);
    try t.expectEqual(@as(u32, 41), smesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 42), smesh.index_buffer.id);
    try t.expectEqual([3]f32{ 99, 99, 99 }, sverts[0].position);
    // Pending owner: flags, counts, handles, arrays untouched.
    try t.expect(!pmesh.gpu_pending);
    try t.expect(pmesh.pending_dynamic_update);
    try t.expectEqual(@as(u32, 0), pmesh.vertex_count);
    try t.expect(!pmesh.morph_upload_needed);
    try t.expectEqual(@as(u32, 0), pmesh.vertex_buffer.id);
    try t.expectEqual(@as(usize, 3), pmesh.pending_vertices.len);
    try t.expectEqual(@as(usize, 3), pmesh.cpu_indices.len);
    // Outcomes recorded, frozen bytes pre-clobber (no 99s leaked in).
    try t.expect(!slot.morph_uploads.items[0].delivered);
    try t.expect(!slot.p_cpu_uploads.items[0].delivered);
    try t.expect(!slot.p_compute_uploads.items[0].delivered);
    try t.expect(!slot.p_gpu_uploads.items[0].delivered);
    try t.expect(!slot.trail_uploads.items[0].delivered);
    try t.expect(!slot.greased_uploads.items[0].delivered);
    try t.expect(!slot.soft_uploads.items[0].delivered);
    try t.expect(!slot.pending_uploads.items[0].delivered);
    try t.expectEqual([3]f32{ 7, 8, 9 }, slot.morph_data.items[0].position);
    try t.expectEqual(@as(f32, 1.0), slot.p_cpu_data.items[0].pos_size[0]);
    try t.expectEqual([3]f32{ 3, 4, 5 }, slot.trail_verts.items[0].position);
    try t.expectEqual([4]f32{ 5, 6, 7, 8 }, slot.p_compute_data.items[0].spawn_pos_time);
    try t.expectEqual([4]f32{ 1, 2, 3, 4 }, slot.p_compute_data.items[1].spawn_pos_time);
    try t.expectEqual([4]f32{ 11, 12, 13, 14 }, slot.p_gpu_data.items[0].spawn_pos_time);
    try t.expectEqual([3]f32{ 6, 7, 8 }, slot.greased_verts.items[1].position);
    try t.expectEqual(@as(u32, 0), slot.greased_indices.items[0]);
    try t.expectEqual([3]f32{ 2, 3, 4 }, slot.soft_data.items[0].position);
    try t.expectEqual([3]f32{ 8, 8, 8 }, slot.pending_verts.items[0].position);

    // Game-side commit of undelivered outcomes: EXACTLY the flag re-arms,
    // nothing else (scalars/handles/arrays keep their canaries).
    commitSlotResults(&fake_scene, &slot);
    try t.expect(mesh.morph_upload_needed);
    try t.expect(ps.instance_dirty);
    try t.expect(!ps.instance_buffer_pending);
    try t.expect(tm.gpu_dirty);
    try t.expect(cps.compute_flush_pending);
    try t.expect(cps.compute_buffers_pending);
    try t.expect(ps2.gpu_dirty);
    try t.expect(ps2.gpu_dirty_wrapped);
    try t.expect(ps2.gpu_flush_pending);
    try t.expect(!ps2.gpu_slot_buffer_pending);
    try t.expect(gl2.gpu_dirty);
    try t.expect(gl2.gpu_needs_full_upload);
    try t.expect(body.upload_pending);
    try t.expect(!body.buffers_pending);
    try t.expect(pmesh.gpu_pending);
    try t.expect(pmesh.pending_dynamic_update);
    try t.expectEqual(@as(u32, 5), mesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 434343), mesh.vertex_count);
    try t.expectEqual(@as(u32, 11), ps.instance_buffer.id);
    try t.expectEqual(@as(u32, 424242), tmesh.index_count);
    try t.expectEqual(@as(u32, 7), tmesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 8), tmesh.index_buffer.id);
    try t.expectEqual([3]f32{ 99, 99, 99 }, tverts[0].position);
    try t.expectEqual(@as(usize, 2), cps.compute_staged);
    try t.expectEqual(@as(usize, 0), cps.compute_stage_base);
    try t.expectEqual(@as(f32, 0.016), cps.compute_dt_accum);
    try t.expectEqual(@as(u64, 0), cps.compute_dispatches);
    try t.expectEqual(@as(u32, 22), ps2.gpu_slot_buffer.id);
    try t.expectEqual(@as(u32, 31), gmesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 41), smesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 0), pmesh.vertex_count);
    try t.expectEqual(@as(u32, 0), pmesh.vertex_buffer.id);
    try t.expectEqual(@as(usize, 3), pmesh.pending_vertices.len);

    // Delivered outcomes (simulating the live-context flush): the commit
    // publishes exactly the frozen scalars and clears nothing it should
    // keep. Emulate the next stage's flag consumption first (a real stage
    // always clears the flags before the flush that delivers — a
    // delivered outcome never meets a set flag in production); without
    // that the skip-if-newer-mutation rule would — correctly — hold the
    // older scalars back.
    mesh.morph_upload_needed = false;
    ps.instance_dirty = false;
    tm.gpu_dirty = false;
    cps.compute_flush_pending = false;
    cps.compute_buffers_pending = false;
    ps2.gpu_dirty = false;
    ps2.gpu_dirty_wrapped = false;
    ps2.gpu_flush_pending = false;
    gl2.gpu_dirty = false;
    body.upload_pending = false;
    pmesh.gpu_pending = false;
    // Compute delivered window: restore the live ring to the frozen
    // content (emulating "no post-freeze mutation") so the content-match
    // guard passes and the ring advances exactly by the consumed count.
    cps.compute_staging[0].spawn_pos_time = .{ 5, 6, 7, 8 };
    cps.compute_staging[1].spawn_pos_time = .{ 1, 2, 3, 4 };
    slot.morph_uploads.items[0].delivered = true;
    slot.p_cpu_uploads.items[0].delivered = true;
    slot.p_compute_uploads.items[0].delivered = true;
    slot.p_compute_uploads.items[0].consumed_staged = 2;
    slot.p_compute_uploads.items[0].consumed_dt = 0.016;
    slot.p_gpu_uploads.items[0].delivered = true;
    slot.trail_uploads.items[0].delivered = true;
    slot.greased_uploads.items[0].delivered = true;
    slot.greased_uploads.items[0].full_delivered = true;
    slot.soft_uploads.items[0].delivered = true;
    slot.pending_uploads.items[0].delivered = true;
    slot.pending_uploads.items[0].created_vertex_buffer_id = 61;
    slot.pending_uploads.items[0].created_index_buffer_id = 62;
    commitSlotResults(&fake_scene, &slot);
    try t.expect(!mesh.morph_upload_needed);
    try t.expect(!ps.instance_dirty);
    try t.expect(!tm.gpu_dirty);
    try t.expectEqual(@as(u32, 3), tmesh.index_count);
    try t.expectEqual([3]f32{ 0, 0, 0 }, [3]f32{ tmesh.local_bounding_box.min.x, tmesh.local_bounding_box.min.y, tmesh.local_bounding_box.min.z });
    try t.expectEqual([3]f32{ 1, 1, 1 }, [3]f32{ tmesh.local_bounding_box.max.x, tmesh.local_bounding_box.max.y, tmesh.local_bounding_box.max.z });
    // Compute ring advanced exactly (content matched): base 0->2,
    // staged 2->0, dt consumed to 0.
    try t.expectEqual(@as(usize, 2), cps.compute_stage_base);
    try t.expectEqual(@as(usize, 0), cps.compute_staged);
    try t.expectEqual(@as(f32, 0.0), cps.compute_dt_accum);
    try t.expect(!cps.compute_flush_pending);
    // GPU-range delivered: nothing to install, flags stay cleared.
    try t.expect(!ps2.gpu_dirty);
    try t.expect(!ps2.gpu_flush_pending);
    try t.expectEqual(@as(u32, 22), ps2.gpu_slot_buffer.id);
    // Greased delivered with full: the full flag clears (no newer
    // mutation — dirty stayed clear).
    try t.expect(!gl2.gpu_dirty);
    try t.expect(!gl2.gpu_needs_full_upload);
    try t.expectEqual(@as(u32, 31), gmesh.vertex_buffer.id);
    // Softbody delivered: generation consumed, handles kept.
    try t.expect(!body.upload_pending);
    try t.expectEqual(@as(u32, 41), smesh.vertex_buffer.id);
    // Pending delivered with created handles: installed over the zero
    // live ids, vertex count published, morph re-armed for the dynamic
    // update, dynamic flag cleared, pending copy freed (mirrors retained).
    try t.expectEqual(@as(u32, 61), pmesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 62), pmesh.index_buffer.id);
    try t.expectEqual(@as(u32, 3), pmesh.vertex_count);
    try t.expect(pmesh.morph_upload_needed);
    try t.expect(!pmesh.pending_dynamic_update);
    try t.expect(!pmesh.gpu_pending);
    try t.expectEqual(@as(usize, 0), pmesh.pending_vertices.len);
    try t.expectEqual(@as(usize, 3), pmesh.cpu_indices.len);
    // Live arrays still carry canaries — the commit published the FROZEN
    // values and never read the live arrays.
    try t.expectEqual([3]f32{ 99, 99, 99 }, staging[0].position);
    try t.expectEqual([3]f32{ 99, 99, 99 }, tverts[0].position);

    // Content-guard negative: same base + counts but REPLACED bytes must
    // NOT advance. Neutralize packet[0] first (its outcome was consumed
    // above), then freeze a fresh packet over altered live content,
    // clobber the live window post-freeze, and deliver it.
    slot.p_compute_uploads.items[0].consumed_staged = 0;
    slot.p_compute_uploads.items[0].consumed_dt = 0.0;
    cps.compute_staging[0].spawn_pos_time = .{ 50, 51, 52, 53 };
    cps.compute_staging[1].spawn_pos_time = .{ 60, 61, 62, 63 };
    cps.compute_stage_base = 0;
    cps.compute_staged = 2;
    cps.compute_dt_accum = 0.5;
    cps.compute_flush_pending = true;
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 2), slot.p_compute_uploads.items.len);
    cps.compute_flush_pending = false;
    cps.compute_staging[0].spawn_pos_time = .{ 70, 71, 72, 73 };
    slot.p_compute_uploads.items[1].delivered = true;
    slot.p_compute_uploads.items[1].consumed_staged = 2;
    slot.p_compute_uploads.items[1].consumed_dt = 0.0;
    commitSlotResults(&fake_scene, &slot);
    // Base (0) and counts (2>=2) match, but the bytes differ: no advance,
    // no dt touch, no re-arm (delivered) — the intact live window simply
    // re-freezes on the next build.
    try t.expectEqual(@as(usize, 0), cps.compute_stage_base);
    try t.expectEqual(@as(usize, 2), cps.compute_staged);
    try t.expectEqual(@as(f32, 0.5), cps.compute_dt_accum);
    try t.expect(!cps.compute_flush_pending);
}

test "upload packets: lock-free stress — mutating producer vs unlocked staged cycles converge" {
    // Phase 2 (stage A.7) stress regression with the producer exclusion
    // OFF: one thread loops mutating live staging arrays + flags and
    // staging packets (the exact producer surface), while the context
    // thread runs staged flush/commit cycles with NO lock held anywhere.
    // Asserts: no torn packet consumption (producer writes uniform
    // generation triples — any cross-generation mix in a packet fails),
    // no lost uploads (every stage-clear pairs with a commit re-arm, so
    // headless — where nothing can deliver — all flags converge SET),
    // and the frozen bytes always predate the producer's post-publish
    // clobber (the context never reads live arrays).
    const t = std.testing;
    gpu_thread.markContextThread();
    const mesh_types = @import("../mesh/types.zig");
    const sys_mod = @import("../particles/system.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var staging = [_]mesh_types.Vertex{std.mem.zeroes(mesh_types.Vertex)} ** 2;
    var base = [_]mesh_types.Vertex{std.mem.zeroes(mesh_types.Vertex)} ** 2;
    var mesh = @import("../mesh/mesh.zig").Mesh{
        .name = "m",
        .vertex_buffer = .{ .id = 5 },
        .index_buffer = .{},
        .index_count = 0,
        .morph_base = &base,
        .morph_staging = &staging,
        .morph_upload_needed = true,
    };
    _ = mesh.ensureUid();
    var ps = try sys_mod.makeTestSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
    ps.instance_buffer = .{ .id = 11 };
    var meshes = [_]*@TypeOf(mesh){&mesh};
    var systems = [_]*sys_mod.ParticleSystem{&ps};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var retire: retire_mod.GpuRetireQueue = .{};
    defer retire.deinit(t.allocator);
    var fake_scene = .{
        .allocator = t.allocator,
        .meshes = .{ .items = meshes[0..], .capacity = 1 },
        .particles = .{ .systems = .{ .items = systems[0..], .capacity = 1 } },
        .trails = .{ .meshes = .{ .items = no_trails[0..], .capacity = 0 } },
        .softbodies = .{ .bodies = .{ .items = no_bodies[0..], .capacity = 0 } },
        .greased_lines = .{ .items = no_lines[0..], .capacity = 0 },
        .gpu_retire = &retire,
        .flush_in_prepare = true,
    };

    const total_gens: u32 = 300;
    const Mailbox = struct {
        state: std.atomic.Value(u32) = std.atomic.Value(u32).init(0), // 0 empty, 1 staged
        done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        produced: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    };
    var mailbox = Mailbox{};
    const Ctx = struct {
        scene: *@TypeOf(fake_scene),
        slot: *@TypeOf(slot),
        mesh: *@TypeOf(mesh),
        ps: *sys_mod.ParticleSystem,
        staging: *[2]mesh_types.Vertex,
        box: *Mailbox,
    };
    var ctx = Ctx{
        .scene = &fake_scene,
        .slot = &slot,
        .mesh = &mesh,
        .ps = &ps,
        .staging = &staging,
        .box = &mailbox,
    };
    const Producer = struct {
        fn run(c: *Ctx) void {
            var g: u32 = 1;
            while (g <= total_gens) : (g += 1) {
                // Wait for EMPTY (bounded spin — the consumer always
                // drains; a timeout is a loud failure, never a hang: done
                // is set on EVERY return path below).
                var spins: usize = 0;
                while (c.box.state.load(.acquire) != 0 and spins < 100_000_000) : (spins += 1) {
                    std.atomic.spinLoopHint();
                }
                if (c.box.state.load(.acquire) != 0) {
                    c.box.done.store(true, .release);
                    return;
                }
                const gf: f32 = @floatFromInt(g);
                // Uniform generation triple: any cross-generation mix in
                // the frozen packet is a torn consumption.
                c.staging[0].position = .{ gf, gf, gf };
                c.staging[1].position = .{ gf + 0.5, gf + 0.5, gf + 0.5 };
                c.ps.instances[0].pos_size[0] = gf;
                c.ps.instances[1].pos_size[0] = gf + 0.5;
                c.ps.active_count = 2;
                c.mesh.morph_upload_needed = true;
                c.ps.instance_dirty = true;
                c.slot.reset();
                stageUploads(c.scene, c.slot);
                c.box.produced.store(g, .release);
                c.box.state.store(1, .release);
            }
            c.box.done.store(true, .release);
        }
    };

    var prod_slot: ?std.Thread = try std.Thread.spawn(.{}, Producer.run, .{&ctx});
    errdefer {
        // A failed expect below must neither strand the producer (leaked
        // thread) nor hang the join: signal done, then join. The producer
        // always terminates (bounded spins, done on every path); the
        // deferred slot.deinit above runs after this join (LIFO), so the
        // slot outlives the producer.
        mailbox.done.store(true, .release);
        if (prod_slot) |p| {
            p.join();
            prod_slot = null;
        }
    }
    var consumed: u32 = 0;
    var last_g: f32 = 0;
    var idle_spins: usize = 0;
    while (true) {
        if (mailbox.state.load(.acquire) == 1) {
            idle_spins = 0;
            const g_pub = mailbox.produced.load(.acquire);
            const gf_pub: f32 = @floatFromInt(g_pub);
            // Clobber live staging AFTER publish, BEFORE flush: the
            // producer is idle by protocol (it spins on EMPTY without
            // touching the slot or the arrays), so any canary observed in
            // the packet or outcome bytes below proves the flush read
            // live state instead of the frozen packet — a "flush reads
            // live" regression fails here, loudly.
            staging[0].position = .{ 999, 999, 999 };
            staging[1].position = .{ 999, 999, 999 };
            ps.instances[0].pos_size[0] = 999.0;
            ps.instances[1].pos_size[0] = 999.0;
            // Staged begin equivalent with NO producer exclusion: the
            // producer may be spinning (never touching the slot — the
            // mailbox guarantees it owns nothing while staged).
            flushSlotUploads(&fake_scene, &slot);
            // Headless: every outcome undelivered; the packet content must
            // be EXACTLY the frozen generation (not a range: the packet
            // was staged for produced g_pub BEFORE the 999 clobber above,
            // so any deviation — including a canary — is a torn or
            // live-read packet).
            try t.expectEqual(@as(usize, 1), slot.morph_uploads.items.len);
            try t.expectEqual(@as(usize, 1), slot.p_cpu_uploads.items.len);
            try t.expect(!slot.morph_uploads.items[0].delivered);
            try t.expect(!slot.p_cpu_uploads.items[0].delivered);
            try t.expectEqual([3]f32{ gf_pub, gf_pub, gf_pub }, slot.morph_data.items[0].position);
            try t.expectEqual([3]f32{ gf_pub + 0.5, gf_pub + 0.5, gf_pub + 0.5 }, slot.morph_data.items[1].position);
            try t.expectEqual(gf_pub, slot.p_cpu_data.items[0].pos_size[0]);
            try t.expectEqual(gf_pub + 0.5, slot.p_cpu_data.items[1].pos_size[0]);
            last_g = gf_pub;
            // Game-side commit equivalent (also unlocked — it runs
            // producer-side in production, here serialized by the
            // mailbox): every undelivered outcome re-arms its flag.
            commitSlotResults(&fake_scene, &slot);
            try t.expect(mesh.morph_upload_needed);
            try t.expect(ps.instance_dirty);
            consumed += 1;
            mailbox.state.store(0, .release);
        } else if (mailbox.done.load(.acquire)) {
            break;
        } else {
            // Bounded consumer wait: a wedged producer (done never set)
            // fails loudly with this error instead of hanging CI forever.
            idle_spins += 1;
            if (idle_spins > 100_000_000) return error.LockFreeStressConsumerTimeout;
            std.atomic.spinLoopHint();
        }
    }
    if (prod_slot) |p| {
        p.join();
        prod_slot = null;
    }
    // Drain a final staged packet the producer published before exiting.
    if (mailbox.state.load(.acquire) == 1) {
        flushSlotUploads(&fake_scene, &slot);
        commitSlotResults(&fake_scene, &slot);
        const p0 = slot.morph_data.items[0].position;
        last_g = p0[0];
        consumed += 1;
        mailbox.state.store(0, .release);
    }
    // No lost uploads: every produced generation was consumed exactly
    // once, the last frozen generation is the last produced one, and all
    // flags converge SET (headless: nothing could deliver, everything
    // must be retry-pending).
    try t.expectEqual(total_gens, consumed);
    try t.expectEqual(@as(f32, @floatFromInt(total_gens)), last_g);
    try t.expectEqual(total_gens, mailbox.produced.load(.acquire));
    try t.expect(mesh.morph_upload_needed);
    try t.expect(ps.instance_dirty);
    try t.expectEqual(@as(usize, 1), slot.morph_uploads.items.len);
}

test "upload packets: pending gpu-morph freeze packs delta bytes, survives live mutation" {
    // Write-once GPU-morph creation on the staged path: the producer packs
    // the RGBA32F delta strip (sg-free) and freezes bytes + dims into the
    // pending packet, consuming BOTH flags at stage time. Live mutation
    // after the freeze must not reach the packet.
    const t = std.testing;
    const mesh_types = @import("../mesh/types.zig");
    const mesh_mod = @import("../mesh/mesh.zig");
    const frame_draws = @import("frame_draws.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var pos = [_][3]f32{ .{ 0.5, 0, 0 }, .{ 0, 0.25, 0 } };
    var nrm = [_][3]f32{ .{ 0, 0.1, 0 }, .{ 0, 0, 0.2 } };
    var targets = [_]mesh_types.MorphTarget{.{
        .position_deltas = &pos,
        .normal_deltas = &nrm,
    }};
    var base = [_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) };
    var pverts = [_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) };
    var cidx = [_]u32{ 0, 1, 2 };
    var weights = [_]f32{0} ** 1;
    var mesh = mesh_mod.Mesh{
        .name = "gpu_morph",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .gpu_pending = true,
        .pending_vertices = &pverts,
        .cpu_indices = &cidx,
        .morph_targets = &targets,
        .morph_weights = &weights,
        .morph_base = &base,
        .morph_mode = .gpu,
        .morph_upload_pending = true,
    };
    _ = mesh.ensureUid();
    var meshes = [_]*mesh_mod.Mesh{&mesh};
    const SysType = @import("../particles/system.zig").ParticleSystem;
    var no_systems: []*SysType = &.{};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var fake_scene = .{
        .allocator = t.allocator,
        .meshes = .{ .items = meshes[0..], .capacity = 1 },
        .particles = .{ .systems = .{ .items = no_systems[0..], .capacity = 0 } },
        .trails = .{ .meshes = .{ .items = no_trails[0..], .capacity = 0 } },
        .softbodies = .{ .bodies = .{ .items = no_bodies[0..], .capacity = 0 } },
        .greased_lines = .{ .items = no_lines[0..], .capacity = 0 },
    };
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.pending_uploads.items.len);
    const up = slot.pending_uploads.items[0];
    try t.expect(up.morph_delta_pending);
    const size = morph_gpu.textureSizeFor(2);
    try t.expectEqual(size.width, up.delta_width);
    try t.expectEqual(size.height, up.delta_height);
    try t.expectEqual(@as(usize, size.width) * size.height * 4, up.delta_count);
    // Texel layout mirror (texelIndex * 4 f32): vertex 0 / target 0 /
    // position -> f32 0..3, normal -> 4..7, tangent (absent) -> zeros.
    const px = slot.pending_delta_data.items[up.delta_lo..][0..12];
    try t.expectEqual(pos[0], [3]f32{ px[0], px[1], px[2] });
    try t.expectEqual(@as(f32, 0), px[3]);
    try t.expectEqual(nrm[0], [3]f32{ px[4], px[5], px[6] });
    try t.expectEqual([4]f32{ 0, 0, 0, 0 }, [4]f32{ px[8], px[9], px[10], px[11] });
    // Vertex 1 / target 0 / position texel (1 * 24 + 0) * 4 = f32 96.
    const v1 = slot.pending_delta_data.items[up.delta_lo..][96..100];
    try t.expectEqual(pos[1], [3]f32{ v1[0], v1[1], v1[2] });
    // Both flags consumed at stage time (phase 2 ownership transfer).
    try t.expect(!mesh.gpu_pending);
    try t.expect(!mesh.morph_upload_pending);

    // Live mutation after the freeze must not reach the packet.
    pos[0] = .{ 99, 99, 99 };
    nrm[1] = .{ 99, 99, 99 };
    const px2 = slot.pending_delta_data.items[up.delta_lo..][0..12];
    try t.expectEqual([3]f32{ 0.5, 0, 0 }, [3]f32{ px2[0], px2[1], px2[2] });
}

test "upload packets: pending delta undelivered re-arms both flags, bytes intact" {
    // Headless staged cycle: nothing can deliver, so the game-side commit
    // must re-arm gpu_pending AND morph_upload_pending for retry with the
    // frozen bytes intact — and a cancelled claim must re-arm both too.
    const t = std.testing;
    gpu_thread.markContextThread();
    const mesh_types = @import("../mesh/types.zig");
    const mesh_mod = @import("../mesh/mesh.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var pos = [_][3]f32{ .{ 0.5, 0, 0 }, .{ 0, 0.25, 0 } };
    var targets = [_]mesh_types.MorphTarget{.{
        .position_deltas = &pos,
    }};
    var base = [_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) };
    var pverts = [_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) };
    var cidx = [_]u32{ 0, 1, 2 };
    var mesh = mesh_mod.Mesh{
        .name = "gpu_morph_retry",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .gpu_pending = true,
        .pending_vertices = &pverts,
        .cpu_indices = &cidx,
        .morph_targets = &targets,
        .morph_base = &base,
        .morph_mode = .gpu,
        .morph_upload_pending = true,
    };
    _ = mesh.ensureUid();
    var meshes = [_]*mesh_mod.Mesh{&mesh};
    const SysType = @import("../particles/system.zig").ParticleSystem;
    var no_systems: []*SysType = &.{};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var retire: retire_mod.GpuRetireQueue = .{};
    defer retire.deinit(t.allocator);
    var fake_scene = .{
        .allocator = t.allocator,
        .meshes = .{ .items = meshes[0..], .capacity = 1 },
        .particles = .{ .systems = .{ .items = no_systems[0..], .capacity = 0 } },
        .trails = .{ .meshes = .{ .items = no_trails[0..], .capacity = 0 } },
        .softbodies = .{ .bodies = .{ .items = no_bodies[0..], .capacity = 0 } },
        .greased_lines = .{ .items = no_lines[0..], .capacity = 0 },
        .gpu_retire = &retire,
        .flush_in_prepare = true,
    };
    stageUploads(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), slot.pending_uploads.items.len);
    try t.expect(slot.pending_uploads.items[0].morph_delta_pending);
    try t.expect(!mesh.gpu_pending);
    try t.expect(!mesh.morph_upload_pending);

    // Headless flush: no context, nothing delivered, no live writes.
    flushSlotUploads(&fake_scene, &slot);
    try t.expect(!slot.pending_uploads.items[0].delivered);
    try t.expect(!mesh.gpu_pending);
    try t.expect(!mesh.morph_upload_pending);

    // Game-side commit re-arms both flags; frozen bytes stay intact.
    commitSlotResults(&fake_scene, &slot);
    try t.expect(mesh.gpu_pending);
    try t.expect(mesh.morph_upload_pending);
    const up = slot.pending_uploads.items[0];
    const px = slot.pending_delta_data.items[up.delta_lo..][0..3];
    try t.expectEqual(pos[0], [3]f32{ px[0], px[1], px[2] });

    // A cancelled claim re-arms both flags as well (inverse of stage).
    mesh.gpu_pending = false;
    mesh.morph_upload_pending = false;
    restageDroppedSlot(&fake_scene, &slot);
    try t.expect(mesh.gpu_pending);
    try t.expect(mesh.morph_upload_pending);
}
