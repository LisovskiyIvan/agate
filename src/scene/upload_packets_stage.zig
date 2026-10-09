//! Producer-side freeze of slot-owned dynamic-upload packets.
//! Section of the `upload_packets` facade (pure move, see
//! `upload_packets.zig` for the full stage/flush/commit contract).

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
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
        if (m.morph_mode == .gpu) continue;
        if (!m.hasMorphTargets() and !m.morph_upload_needed) continue;
        if (sg.isvalid() and m.vertex_buffer.id != 0 and !sg.queryBufferUsage(m.vertex_buffer).write_transient) continue;
        const n = @min(m.morph_base.len, m.morph_staging.len);
        if (n == 0) {
            if (m.morph_upload_needed) {
                const data_lo = slot.morph_data.items.len;
                slot.morph_uploads.append(allocator, .{
                    .token = @intFromPtr(m),
                    .uid = m.uid,
                    .mesh_index = @intCast(i),
                    .buffer_id = m.vertex_buffer.id,
                    .count = 0,
                    .data_lo = data_lo,
                    .dirty = true,
                }) catch {
                    slot.build_stats.build_oom_drops += 1;
                    continue;
                };
                m.morph_upload_needed = false;
            }
            continue;
        }
        if (m.vertex_buffer.id == 0) continue;
        const is_dirty = m.morph_upload_needed;
        const data_lo = slot.morph_data.items.len;
        slot.morph_data.appendSlice(allocator, m.morph_staging[0..n]) catch {
            slot.build_stats.build_oom_drops += 1;
            continue;
        };
        slot.morph_uploads.append(allocator, .{
            .token = @intFromPtr(m),
            .uid = m.uid,
            .mesh_index = @intCast(i),
            .buffer_id = m.vertex_buffer.id,
            .count = @intCast(n),
            .data_lo = data_lo,
            .dirty = is_dirty,
        }) catch {
            slot.morph_data.items.len = data_lo;
            continue;
        };
        m.morph_upload_needed = false;
    }
}

fn stageParticleCpu(scene: anytype, slot: anytype, allocator: std.mem.Allocator) void {
    for (scene.particles.systems.items, 0..) |ps, i| {
        if (!ps.instance_dirty and ps.active_count == 0) continue;
        const count: usize = ps.active_count;
        if (count > ps.instances.len) continue;
        // Empty-but-dirty systems freeze an empty packet so the staged flush
        // consumes the flag without ever reading live state on the latch.
        const is_dirty = ps.instance_dirty;
        const data_lo = slot.p_cpu_data.items.len;
        if (count > 0) slot.p_cpu_data.appendSlice(allocator, ps.instances[0..count]) catch {
            slot.build_stats.build_oom_drops += 1;
            continue;
        };
        slot.p_cpu_uploads.append(allocator, .{
            .token = @intFromPtr(ps),
            .sys_index = @intCast(i),
            .buffer_id = ps.instance_buffer.id,
            .count = @intCast(count),
            .data_lo = data_lo,
            .capacity = ps.capacity,
            .dirty = is_dirty,
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
        if (!ps.gpu_dirty and !ps.gpu_flush_pending and ps.gpu_high_water == 0) continue;
        if (ps.gpu_slots.len == 0) continue;
        // Freeze the exact upload range; a pending flag with an empty window
        // still freezes an empty packet so the latch consumes the flags
        // without live reads.
        const is_dirty = ps.gpu_dirty or ps.gpu_flush_pending;
        const range: []const @TypeOf(ps.gpu_slots[0]) = ps.gpu_slots[0..ps.gpu_high_water];
        const data_lo = slot.p_gpu_data.items.len;
        if (range.len > 0) slot.p_gpu_data.appendSlice(allocator, range) catch {
            slot.build_stats.build_oom_drops += 1;
            continue;
        };
        slot.p_gpu_uploads.append(allocator, .{
            .token = @intFromPtr(ps),
            .sys_index = @intCast(i),
            .buffer_id = ps.gpu_slot_buffer.id,
            .count = @intCast(range.len),
            .data_lo = data_lo,
            .capacity = ps.capacity,
            .dirty = is_dirty,
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
            slot.p_compute_data.appendSlice(allocator, ps.compute_staging[0..staged]) catch {
                slot.build_stats.build_oom_drops += 1;
                continue;
            };
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
        if (!tm.gpu_dirty and tm.pending_vertex_count == 0) continue;
        const is_dirty = tm.gpu_dirty;
        const vc: usize = tm.pending_vertex_count;
        const ic: usize = tm.pending_index_count;
        if (vc > tm.vertices.len or ic > tm.indices.len) continue;
        // Empty-but-dirty freezes an empty packet: the quiesced drain clears
        // the flag and publishes index_count/bounds unconditionally.
        const v_lo = slot.trail_verts.items.len;
        const i_lo = slot.trail_indices.items.len;
        if (vc > 0) slot.trail_verts.appendSlice(allocator, tm.vertices[0..vc]) catch {
            slot.build_stats.build_oom_drops += 1;
            continue;
        };
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
            .dirty = is_dirty,
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
        if (!b.upload_pending and b.vertices.len == 0) continue;
        const is_dirty = b.upload_pending;
        // Empty verts still freeze an empty packet: the quiesced drain clears
        // the flag unconditionally. Indices are frozen alongside the verts
        // (grid topology is fixed at creation, but freezing keeps the staged
        // flush independent of every live array).
        const data_lo = slot.soft_data.items.len;
        if (b.vertices.len > 0) slot.soft_data.appendSlice(allocator, b.vertices) catch {
            slot.build_stats.build_oom_drops += 1;
            continue;
        };
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
            .dirty = is_dirty,
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
        if (!gl.gpu_dirty and gl.vertices.len == 0) continue;
        const is_dirty = gl.gpu_dirty;
        // Empty verts still freeze an empty packet so a live context clears
        // the flag exactly like the quiesced drain. Indices are frozen
        // whenever a full upload may be needed: the staged full flag, or
        // missing live buffers (the flush will create them, which forces a
        // full index upload there).
        const v_lo = slot.greased_verts.items.len;
        if (gl.vertices.len > 0) slot.greased_verts.appendSlice(allocator, gl.vertices) catch {
            slot.build_stats.build_oom_drops += 1;
            continue;
        };
        const full = gl.gpu_needs_full_upload;
        const i_lo = slot.greased_indices.items.len;
        if (gl.indices.len > 0) {
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
            .index_count = gl.indices.len,
            .vert_lo = v_lo,
            .index_lo = i_lo,
            .full_upload = full,
            // Frozen allocation lengths: creation sizes the buffers from
            // these, never the live slices.
            .vert_cap = gl.vertices.len,
            .index_cap = gl.indices.len,
            .dirty = is_dirty,
        }) catch {
            slot.greased_verts.items.len = v_lo;
            if (gl.indices.len > 0) slot.greased_indices.items.len = i_lo;
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
        // drain's finish attempts creation whenever gpu_pending is set, so
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
            delta_pixels = morph_gpu.packDeltas(allocator, m.morph_targets, m.morph_base.len, delta_size) catch {
                slot.build_stats.build_oom_drops += 1;
                continue;
            };
        }
        defer if (delta_pixels.len > 0) allocator.free(delta_pixels);
        const v_lo = slot.pending_verts.items.len;
        if (m.pending_vertices.len > 0) slot.pending_verts.appendSlice(allocator, m.pending_vertices) catch {
            slot.build_stats.build_oom_drops += 1;
            continue;
        };
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
