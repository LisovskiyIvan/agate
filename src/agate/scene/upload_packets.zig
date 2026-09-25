//! Slot-owned dynamic-upload packets (producer freeze-then-latch slice 6).
//!
//! Producer side (`stageUploads`, game/update phase, sg-free): copies every
//! per-frame GPU staging payload into the claimed `FrameDrawSlot` by value
//! (morph vertex bytes, particle CPU/GPU/compute staging bytes, trail verts
//! + indices, softbody verts + indices, greased verts + indices,
//! pending-mesh creation geometry) plus frozen buffer ids / counts / bounds.
//! Empty-but-dirty owners freeze empty packets so the staged flush consumes
//! flags and publishes zero scalars exactly like the legacy flush. Live
//! dirty flags are NOT cleared here (a cancelled claim must not consume the
//! upload); the staged prepare consumes them in its excluded begin window.
//!
//! Context side (`flushSlotUploads`, beginPrepare with a fresh build):
//! uploads from the slot packets and creates deferred buffers, then
//! publishes the live scalars (index counts, bounds, flag clears) inside
//! the excluded begin window. Accurate contract: no live staging BYTES
//! drive uploads, with two narrow exceptions — the frozen-window reinstall
//! in `flushParticleCompute` (frozen bytes are memcpied into the live
//! staging window under exclusion so the reused legacy helper uploads
//! exactly them) and the deferred-creation sizes (immutable allocation
//! lengths/capacities, never mutable content). Live scalar/flag/buffer-id
//! touches stay under the retained producer exclusion around begin.
//!
//! Deferred morph-delta textures (`morph_upload_pending`) stay on the
//! legacy fallback path and are deliberately NOT frozen (rare,
//! write-once, GPU-mode only): a staged creation draws the base pose for
//! one frame while the delta upload retries on the next fallback flush.
//!
//! The legacy `flushPendingGpuUploads` stays for the no-build fallback
//! (serialized by contract) and is never called on the fresh-build path.
//!
//! Identity: descriptors carry `token` (@intFromPtr of the live owner) +
//! list index (+ uid for meshes). The flush validates token/index (/uid)
//! without dereferencing a stale pointer first: mismatch skips fail-closed
//! (previous complete GPU state stands, retry next build). Mesh-list or
//! registry mutation between build and latch violates the app contract; the
//! guard keeps it coherent, never corrupt.
//!
//! OOM: any packet that fails to stage is skipped like an OOM-skipped
//! instance segment (no partial packet: data truncated back, descriptor
//! unwritten). The live flags stay set, so the next funded build retries;
//! the prepare flush simply has no packet for that owner this frame.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");

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
        // clears the flag unconditionally, so the staged flush must consume
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
        }) catch {
            slot.p_cpu_data.items.len = data_lo;
            continue;
        };
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
        }) catch {
            slot.p_gpu_data.items.len = data_lo;
            continue;
        };
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
        }) catch {
            slot.p_compute_data.items.len = data_lo;
            continue;
        };
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
        }) catch {
            slot.trail_verts.items.len = v_lo;
            slot.trail_indices.items.len = i_lo;
            continue;
        };
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
        }) catch {
            slot.soft_data.items.len = data_lo;
            slot.soft_indices.items.len = i_lo;
            continue;
        };
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
        }) catch {
            slot.greased_verts.items.len = v_lo;
            if (need_idx) slot.greased_indices.items.len = i_lo;
            continue;
        };
    }
}

fn stagePendingMeshes(scene: anytype, slot: anytype, allocator: std.mem.Allocator) void {
    for (scene.meshes.items, 0..) |m, i| {
        if (!m.gpu_pending) continue;
        // Degenerate geometry still freezes a (possibly empty) packet: the
        // legacy finish attempts creation whenever gpu_pending is set, so
        // the staged flush must observe the same owner instead of skipping
        // it silently.
        const v_lo = slot.pending_verts.items.len;
        if (m.pending_vertices.len > 0) slot.pending_verts.appendSlice(allocator, m.pending_vertices) catch continue;
        const i_lo = slot.pending_indices.items.len;
        if (m.cpu_indices.len > 0) slot.pending_indices.appendSlice(allocator, m.cpu_indices) catch {
            slot.pending_verts.items.len = v_lo;
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
        }) catch {
            slot.pending_verts.items.len = v_lo;
            slot.pending_indices.items.len = i_lo;
            continue;
        };
    }
}

/// Context-side slot flush (beginPrepare, fresh-build path only). Uploads
/// from `slot` packets; no live staging BYTES drive uploads, with two
/// narrow exceptions that run under the retained exclusion: the frozen
/// window `memcpy` reinstall in `flushParticleCompute`, and immutable
/// allocation lengths/capacities used for deferred buffer creation. Live
/// scalar publishes (index counts, bounds, flag clears, buffer creations)
/// run here inside the excluded begin window. Headless-safe: with no sg
/// context every flag still clears and every scalar still publishes, only
/// the sg.* calls are skipped. See the module header for the full contract.
pub fn flushSlotUploads(scene: anytype, slot: anytype) void {
    gpu_thread.assertOnContextThread();
    scene.gpu_retire.flush(scene.allocator);
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

fn liveMeshAt(scene: anytype, mesh_index: u32, token: usize, uid: u64) ?@TypeOf(scene.meshes.items[0]) {
    const idx: usize = mesh_index;
    if (idx >= scene.meshes.items.len) return null;
    const m = scene.meshes.items[idx];
    if (@intFromPtr(m) != token) return null;
    if (uid != 0 and m.uid != uid) return null;
    return m;
}

fn flushPendingCreations(scene: anytype, slot: anytype) void {
    if (!sg.isvalid()) return;
    for (slot.pending_uploads.items) |up| {
        const m = liveMeshAt(scene, up.mesh_index, up.token, up.uid) orelse continue;
        if (!m.gpu_pending) continue;
        const v_end = up.vert_lo + up.vert_count;
        const i_end = up.index_lo + up.index_count;
        if (v_end > slot.pending_verts.items.len or i_end > slot.pending_indices.items.len) continue;
        const verts = slot.pending_verts.items[up.vert_lo..v_end];
        const idx32 = slot.pending_indices.items[up.index_lo..i_end];
        if (up.dynamic_update) {
            const vbuf = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = verts.len * @sizeOf(@TypeOf(verts[0])),
            });
            if (vbuf.id == 0) continue;
            var ibuf: sg.Buffer = .{};
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
            m.vertex_buffer = vbuf;
            m.index_buffer = ibuf;
        } else {
            const vbuf = sg.makeBuffer(.{ .data = sg.asRange(verts) });
            if (vbuf.id == 0) continue;
            var ibuf: sg.Buffer = .{};
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
            m.vertex_buffer = vbuf;
            m.index_buffer = ibuf;
        }
        if (m.vertex_count == 0 and verts.len > 0) m.vertex_count = @intCast(verts.len);
        m.gpu_pending = false;
        if (up.dynamic_update) m.morph_upload_needed = true;
        m.pending_dynamic_update = false;
        if (m.pending_vertices.len > 0) {
            scene.allocator.free(m.pending_vertices);
            m.pending_vertices = &.{};
        }
    }
}

fn flushMorphs(scene: anytype, slot: anytype) void {
    for (slot.morph_uploads.items) |up| {
        const m = liveMeshAt(scene, up.mesh_index, up.token, up.uid) orelse continue;
        const end = up.data_lo + up.count;
        if (end > slot.morph_data.items.len) continue;
        m.morph_upload_needed = false;
        if (up.count == 0) continue;
        if (!sg.isvalid()) continue;
        if (up.buffer_id == 0 or m.vertex_buffer.id != up.buffer_id) continue;
        sg.updateBuffer(m.vertex_buffer, sg.asRange(slot.morph_data.items[up.data_lo..end]));
        upload_meter.record(up.count * @sizeOf(@TypeOf(slot.morph_data.items[0])));
    }
}

fn liveParticleAt(scene: anytype, sys_index: u32, token: usize) ?@TypeOf(scene.particles.systems.items[0]) {
    const idx: usize = sys_index;
    if (idx >= scene.particles.systems.items.len) return null;
    const ps = scene.particles.systems.items[idx];
    if (@intFromPtr(ps) != token) return null;
    return ps;
}

fn flushParticleCpu(scene: anytype, slot: anytype) void {
    for (slot.p_cpu_uploads.items) |up| {
        const ps = liveParticleAt(scene, up.sys_index, up.token) orelse continue;
        const end = up.data_lo + up.count;
        if (end > slot.p_cpu_data.items.len) continue;
        // Consume tentatively; re-armed below when bytes cannot be
        // delivered, so a dropped upload always retries instead of going
        // stale (finding 1: legacy creates the deferred buffer first and
        // then uploads in the same flush — the staged path must do the
        // same from the frozen bytes).
        ps.instance_dirty = false;
        if (ps.instance_buffer_pending and sg.isvalid()) {
            ps.instance_buffer = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = ps.capacity * @sizeOf(@TypeOf(slot.p_cpu_data.items[0])),
            });
            if (ps.instance_buffer.id != 0) ps.instance_buffer_pending = false;
        }
        if (up.count == 0) continue;
        if (up.buffer_id != 0 and ps.instance_buffer.id != 0 and ps.instance_buffer.id != up.buffer_id) {
            // Live buffer is not the one staged for (replaced between build
            // and latch — only possible under contract violation): never
            // upload foreign bytes into it; re-arm for a fresh freeze.
            ps.instance_dirty = true;
            continue;
        }
        if (sg.isvalid() and ps.instance_buffer.id != 0) {
            sg.updateBuffer(ps.instance_buffer, sg.asRange(slot.p_cpu_data.items[up.data_lo..end]));
            upload_meter.record(up.count * @sizeOf(@TypeOf(slot.p_cpu_data.items[0])));
            continue;
        }
        // Undeliverable (creation failed/pool-exhausted, or headless with
        // no buffer yet): re-arm for retry. The next funded build
        // re-freezes from the intact live arrays.
        ps.instance_dirty = true;
    }
}

fn flushParticleGpu(scene: anytype, slot: anytype) void {
    for (slot.p_gpu_uploads.items) |up| {
        const ps = liveParticleAt(scene, up.sys_index, up.token) orelse continue;
        const end = up.data_lo + up.count;
        if (end > slot.p_gpu_data.items.len) continue;
        ps.gpu_dirty = false;
        ps.gpu_dirty_wrapped = false;
        ps.gpu_flush_pending = false;
        if (ps.gpu_slot_buffer_pending and sg.isvalid()) {
            ps.gpu_slot_buffer = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = ps.capacity * @sizeOf(@TypeOf(slot.p_gpu_data.items[0])),
            });
            if (ps.gpu_slot_buffer.id != 0) ps.gpu_slot_buffer_pending = false;
        }
        if (up.count == 0) continue;
        if (up.buffer_id != 0 and ps.gpu_slot_buffer.id != 0 and ps.gpu_slot_buffer.id != up.buffer_id) {
            // Same foreign-buffer guard as the CPU path above.
            ps.gpu_dirty = true;
            ps.gpu_dirty_wrapped = true;
            ps.gpu_flush_pending = true;
            continue;
        }
        if (sg.isvalid() and ps.gpu_slot_buffer.id != 0) {
            sg.updateBuffer(ps.gpu_slot_buffer, sg.asRange(slot.p_gpu_data.items[up.data_lo..end]));
            upload_meter.record(up.count * @sizeOf(@TypeOf(slot.p_gpu_data.items[0])));
            continue;
        }
        // Undeliverable: re-arm a full-prefix window (a superset of any
        // dirty range) so the next build re-freezes from intact live slots.
        ps.gpu_dirty = true;
        ps.gpu_dirty_wrapped = true;
        ps.gpu_flush_pending = true;
    }
}

fn flushParticleCompute(scene: anytype, slot: anytype) void {
    const compute_mod = @import("../particles/compute_mode.zig");
    for (slot.p_compute_uploads.items) |up| {
        const ps = liveParticleAt(scene, up.sys_index, up.token) orelse continue;
        const end = up.data_lo + up.data_count;
        if (end > slot.p_compute_data.items.len) continue;
        if (!sg.isvalid()) {
            // Headless: full no-op that retains every live flag for retry,
            // exactly like the legacy helper's `if (!sg.isvalid()) return`.
            // The frozen bytes stay in the slot; the next funded build
            // re-freezes from the intact live window.
            continue;
        }
        // Creation/dispatch stay on the legacy helper but fed with the
        // FROZEN window: the values below are installed over the live ones
        // under the retained exclusion (the producer staged these exact
        // values), including `buffers_pending` so the helper's
        // ensureComputeGpu path is driven by the frozen request, never the
        // live flag. The frozen staging bytes are then memcpied into the
        // live staging window (same exclusion) so the reused helper uploads
        // exactly them — the single documented exception to "no live
        // staging bytes drive uploads".
        ps.compute_staged = up.data_count;
        ps.compute_stage_base = up.stage_base;
        ps.compute_cursor = up.cursor;
        ps.compute_high_water = up.high_water;
        ps.compute_dt_accum = up.dt_accum;
        ps.compute_flush_pending = up.flush_pending;
        ps.compute_state_clear_pending = up.state_clear_pending;
        ps.compute_buffers_pending = up.buffers_pending;
        ps.gravity = math.Vec3.new(up.gravity[0], up.gravity[1], up.gravity[2]);
        ps.drag = up.drag;
        ps.spritesheet_columns = up.sheet_cols;
        ps.spritesheet_rows = up.sheet_rows;
        ps.spritesheet_loops = up.sheet_loops;
        // Install the frozen bytes into the live staging window (same
        // retained exclusion as the scalar installs above) so the reused
        // helper uploads exactly them.
        if (up.data_count > 0) {
            const live_staging = ps.compute_staging;
            const pkt = slot.p_compute_data.items[up.data_lo..end];
            const n = @min(live_staging.len, pkt.len);
            @memcpy(live_staging[0..n], pkt[0..n]);
        }
        compute_mod.flushComputeUploads(ps);
    }
}

fn liveTrailAt(scene: anytype, trail_index: u32, token: usize) ?@TypeOf(scene.trails.meshes.items[0]) {
    const idx: usize = trail_index;
    if (idx >= scene.trails.meshes.items.len) return null;
    const tm = scene.trails.meshes.items[idx];
    if (@intFromPtr(tm) != token) return null;
    return tm;
}

fn flushTrails(scene: anytype, slot: anytype) void {
    for (slot.trail_uploads.items) |up| {
        const tm = liveTrailAt(scene, up.trail_index, up.token) orelse continue;
        const v_end = up.vert_lo + up.vert_count;
        const i_end = up.index_lo + up.index_count;
        if (v_end > slot.trail_verts.items.len or i_end > slot.trail_indices.items.len) continue;
        tm.gpu_dirty = false;
        if (up.buffers_pending and sg.isvalid()) {
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = tm.vertices.len * @sizeOf(@TypeOf(tm.vertices[0])),
            });
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true, .dynamic_update = true },
                .size = tm.indices.len * @sizeOf(@TypeOf(tm.indices[0])),
            });
            if (vb.id != 0 and ib.id != 0) {
                tm.mesh.vertex_buffer = vb;
                tm.mesh.index_buffer = ib;
                tm.buffers_pending = false;
            } else {
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
            }
        }
        if (sg.isvalid()) {
            if (up.vert_count > 0 and tm.mesh.vertex_buffer.id != 0) {
                const target_ok = up.vertex_buffer_id == 0 or tm.mesh.vertex_buffer.id == up.vertex_buffer_id;
                if (target_ok) {
                    sg.updateBuffer(tm.mesh.vertex_buffer, sg.asRange(slot.trail_verts.items[up.vert_lo..v_end]));
                    upload_meter.record(up.vert_count * @sizeOf(@TypeOf(slot.trail_verts.items[0])));
                }
            }
            if (up.index_count > 0 and tm.mesh.index_buffer.id != 0) {
                const target_ok = up.index_buffer_id == 0 or tm.mesh.index_buffer.id == up.index_buffer_id;
                if (target_ok) {
                    sg.updateBuffer(tm.mesh.index_buffer, sg.asRange(slot.trail_indices.items[up.index_lo..i_end]));
                    upload_meter.record(up.index_count * @sizeOf(@TypeOf(slot.trail_indices.items[0])));
                }
            }
        }
        tm.mesh.index_count = @intCast(up.index_count);
        tm.mesh.local_bounding_box = math.BoundingBox.init(
            math.Vec3.new(up.min_pt[0], up.min_pt[1], up.min_pt[2]),
            math.Vec3.new(up.max_pt[0], up.max_pt[1], up.max_pt[2]),
        );
        tm.mesh.cached_aabb = tm.mesh.local_bounding_box;
    }
}

fn liveSoftAt(scene: anytype, body_index: u32, token: usize) ?@TypeOf(scene.softbodies.bodies.items[0]) {
    const idx: usize = body_index;
    if (idx >= scene.softbodies.bodies.items.len) return null;
    const b = scene.softbodies.bodies.items[idx];
    if (@intFromPtr(b) != token) return null;
    return b;
}

fn flushSoftbodies(scene: anytype, slot: anytype) void {
    for (slot.soft_uploads.items) |up| {
        const b = liveSoftAt(scene, up.body_index, up.token) orelse continue;
        const end = up.data_lo + up.vert_count;
        if (end > slot.soft_data.items.len) continue;
        const i_end = up.index_lo + up.index_count;
        if (i_end > slot.soft_indices.items.len) continue;
        b.upload_pending = false;
        if (up.buffers_pending and sg.isvalid()) {
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = up.vert_count * @sizeOf(@TypeOf(slot.soft_data.items[0])),
            });
            // Cloth index buffers are always u32 (vertices cap at 4096, so
            // this wastes nothing material and keeps one upload path).
            // Bytes come from the frozen packet, never the live array.
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(slot.soft_indices.items[up.index_lo..i_end]),
            });
            if (vb.id != 0 and ib.id != 0) {
                b.mesh.vertex_buffer = vb;
                b.mesh.index_buffer = ib;
                b.buffers_pending = false;
                b.upload_pending = true;
            } else {
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
            }
        }
        if (!sg.isvalid()) continue;
        if (up.vert_count == 0) continue;
        if (b.mesh.vertex_buffer.id == 0) continue;
        if (up.vertex_buffer_id != 0 and b.mesh.vertex_buffer.id != up.vertex_buffer_id) continue;
        sg.updateBuffer(b.mesh.vertex_buffer, sg.asRange(slot.soft_data.items[up.data_lo..end]));
        upload_meter.record(up.vert_count * @sizeOf(@TypeOf(slot.soft_data.items[0])));
        // Consume the creation-forced re-upload (mirrors the legacy clear
        // after upload): without this every created buffer would re-upload
        // each frame.
        b.upload_pending = false;
    }
}

fn liveGreasedAt(scene: anytype, line_index: u32, token: usize) ?@TypeOf(scene.greased_lines.items[0]) {
    const idx: usize = line_index;
    if (idx >= scene.greased_lines.items.len) return null;
    const gl = scene.greased_lines.items[idx];
    if (@intFromPtr(gl) != token) return null;
    return gl;
}

fn flushGreased(scene: anytype, slot: anytype) void {
    for (slot.greased_uploads.items) |up| {
        const gl = liveGreasedAt(scene, up.line_index, up.token) orelse continue;
        const v_end = up.vert_lo + up.vert_count;
        if (v_end > slot.greased_verts.items.len) continue;
        const i_end = up.index_lo + up.index_count;
        if (i_end > slot.greased_indices.items.len) continue;
        // Headless: retain the flag for retry, exactly like the legacy
        // `if (!sg.isvalid()) return` (which keeps dirty when no context
        // exists). Flag clears below only on a live context.
        if (!sg.isvalid()) continue;
        gl.gpu_dirty = false;
        var just_created = false;
        if (gl.mesh.vertex_buffer.id == 0) {
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = gl.vertices.len * @sizeOf(@TypeOf(gl.vertices[0])),
            });
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true, .dynamic_update = true },
                .size = gl.indices.len * @sizeOf(@TypeOf(gl.indices[0])),
            });
            if (vb.id == 0 or ib.id == 0) {
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
                // Creation failed (pool exhaustion): re-arm for retry, like
                // the legacy path which returns with dirty still set.
                gl.gpu_dirty = true;
                continue;
            }
            gl.mesh.vertex_buffer = vb;
            gl.mesh.index_buffer = ib;
            gl.gpu_needs_full_upload = true;
            just_created = true;
        }
        if (up.vert_count > 0 and gl.mesh.vertex_buffer.id != 0) {
            if (up.vertex_buffer_id == 0 or gl.mesh.vertex_buffer.id == up.vertex_buffer_id) {
                sg.updateBuffer(gl.mesh.vertex_buffer, sg.asRange(slot.greased_verts.items[up.vert_lo..v_end]));
                upload_meter.record(up.vert_count * @sizeOf(@TypeOf(slot.greased_verts.items[0])));
            }
        }
        // A just-created index buffer is empty: it needs the full index
        // upload even when the frozen full flag was false (the stage
        // froze indices whenever live buffers were missing, precisely for
        // this case).
        if ((up.full_upload or just_created) and up.index_count > 0 and gl.mesh.index_buffer.id != 0) {
            if (up.index_buffer_id == 0 or gl.mesh.index_buffer.id == up.index_buffer_id) {
                sg.updateBuffer(gl.mesh.index_buffer, sg.asRange(slot.greased_indices.items[up.index_lo..i_end]));
                upload_meter.record(up.index_count * @sizeOf(@TypeOf(slot.greased_indices.items[0])));
            }
            gl.gpu_needs_full_upload = false;
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
    // skipped after creation and never retried. Headless there is no
    // context to create buffers with, so the fix must re-arm the flags;
    // this test fails on the old code (flags cleared) and passes on the
    // new code (flags retained for retry).
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

    // Headless: no context exists, so nothing can be delivered — the flags
    // must come back armed for retry, and the frozen bytes must be intact.
    flushSlotUploads(&fake_scene, &slot);
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
    // must skip both packets without touching live state.
    slot.morph_uploads.items[0].token +%= 1;
    slot.p_cpu_uploads.items[0].token +%= 1;
    flushSlotUploads(&fake_scene, &slot);
    try t.expect(mesh.morph_upload_needed);
    try t.expect(ps.instance_dirty);
}
