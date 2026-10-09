//! Game-side commit of staged-upload outcomes (+ dropped-claim re-arm).
//! Section of the `upload_packets` facade (pure move, see
//! `upload_packets.zig` for the full stage/flush/commit contract).

const std = @import("std");
const math = @import("math");
const gpu_retire = @import("gpu_retire.zig");

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
        if (up.dirty) m.morph_upload_needed = true;
    }
    for (slot.p_cpu_uploads.items) |up| {
        const ps = commitParticleAt(scene, up.sys_index, up.token) orelse continue;
        if (up.dirty) ps.instance_dirty = true;
        if (ps.instance_buffer.id == 0) ps.instance_buffer_pending = true;
    }
    for (slot.p_gpu_uploads.items) |up| {
        const ps = commitParticleAt(scene, up.sys_index, up.token) orelse continue;
        if (up.dirty) {
            ps.gpu_dirty = true;
            ps.gpu_dirty_wrapped = true;
            ps.gpu_flush_pending = true;
        }
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
        if (up.dirty) tm.gpu_dirty = true;
        if (tm.mesh.vertex_buffer.id == 0 or tm.mesh.index_buffer.id == 0) tm.buffers_pending = true;
    }
    for (slot.soft_uploads.items) |up| {
        const b = commitSoftAt(scene, up.body_index, up.token) orelse continue;
        if (up.dirty) b.upload_pending = true;
        if (b.mesh.vertex_buffer.id == 0) b.buffers_pending = true;
    }
    for (slot.greased_uploads.items) |up| {
        const gl = commitGreasedAt(scene, up.line_index, up.token) orelse continue;
        if (up.dirty) gl.gpu_dirty = true;
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
        if (up.dirty) m.morph_upload_needed = true;
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
            if (up.dirty) ps.instance_dirty = true;
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
            if (up.dirty) {
                ps.gpu_dirty = true;
                ps.gpu_dirty_wrapped = true;
                ps.gpu_flush_pending = true;
            }
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
            // Owner gone: the count cannot be applied to any other object
            // (the next token is unrelated), so it is dropped with the
            // packet instead of credited elsewhere.
            up.dispatches = 0;
            continue;
        };
        // Attempt-count transfer (verified token only): every real dispatch
        // the flush issued for this packet lands in the game-owned counter
        // exactly once, BEFORE the delivered/unsupported branches — an
        // earlier attempt counts even when a later retry fails or latches
        // unsupported.
        if (up.dispatches != 0) {
            ps.compute_dispatches +%= up.dispatches;
            up.dispatches = 0;
        }
        if (up.unsupported) {
            // Backend without compute support (mirrors the direct drain):
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
/// commit); every handle that cannot install rides ONE coherent bundle
/// into the epoch retire queue (see GpuRetireQueue.retireComputeBundle) —
/// thread-safe and sg-free game-side, destroyed context-side at the next
/// flush. Both the delivered and the undelivered (partial-progress) paths
/// land here: a failed creation records its partial handles in the outcome
/// so the retry completes the rest instead of leaking them.
///
/// Coherence (no dangling installs): a created view installs only when its
/// live slot is zero AND its backing buffer is live — i.e. the flush
/// stamped the view from frozen ids (which always match live: live buffers
/// only ever transition zero -> set), or from a created buffer that
/// installs alongside it. A view stamped from a LOSING created buffer
/// would reference a retired handle, so it loses with that buffer even
/// when its own live slot is still zero — the same stale-freeze race that
/// makes the buffer lose. Pipelines likewise install only with a live
/// shader that is frozen (= live) or a fellow winner.
fn installComputeCreated(scene: anytype, ps: anytype, up: anytype) void {
    var lost: gpu_retire.ComputeBundle = .{};
    // Buffers and the shader install independently over zero live ids;
    // record which created handles won for the dependent installs below
    // (the outcome fields are read before they are zeroed).
    const made_state_buf = up.created_state_buffer_id != 0;
    var state_buf_wins = false;
    if (made_state_buf) {
        if (ps.compute_state_buffer.id == 0) {
            ps.compute_state_buffer = .{ .id = up.created_state_buffer_id };
            state_buf_wins = true;
        } else if (up.state_clear_pending) {
            lost.state_buffer = ps.compute_state_buffer;
            ps.compute_state_buffer = .{ .id = up.created_state_buffer_id };
            state_buf_wins = true;
        } else {
            lost.state_buffer = .{ .id = up.created_state_buffer_id };
        }
        up.created_state_buffer_id = 0;
    }
    const made_spawn_buf = up.created_spawn_buffer_id != 0;
    var spawn_buf_wins = false;
    if (made_spawn_buf) {
        if (ps.compute_spawn_buffer.id == 0) {
            ps.compute_spawn_buffer = .{ .id = up.created_spawn_buffer_id };
            spawn_buf_wins = true;
        } else {
            lost.spawn_buffer = .{ .id = up.created_spawn_buffer_id };
        }
        up.created_spawn_buffer_id = 0;
    }
    const made_draw_buf = up.created_draw_buffer_id != 0;
    var draw_buf_wins = false;
    if (made_draw_buf) {
        if (ps.compute_draw_buffer.id == 0) {
            ps.compute_draw_buffer = .{ .id = up.created_draw_buffer_id };
            draw_buf_wins = true;
        } else {
            lost.draw_buffer = .{ .id = up.created_draw_buffer_id };
        }
        up.created_draw_buffer_id = 0;
    }
    const made_shader = up.created_shader_id != 0;
    var shader_wins = false;
    if (made_shader) {
        if (ps.compute_shader.id == 0) {
            ps.compute_shader = .{ .id = up.created_shader_id };
            shader_wins = true;
        } else {
            lost.shader = .{ .id = up.created_shader_id };
        }
        up.created_shader_id = 0;
    }
    if (up.created_state_view_id != 0) {
        if ((ps.compute_state_view.id == 0 or up.state_clear_pending) and ps.compute_state_buffer.id != 0 and
            (!made_state_buf or state_buf_wins))
        {
            if (ps.compute_state_view.id != 0) lost.state_view = ps.compute_state_view;
            ps.compute_state_view = .{ .id = up.created_state_view_id };
        } else {
            lost.state_view = .{ .id = up.created_state_view_id };
        }
        up.created_state_view_id = 0;
    }
    if (up.created_spawn_view_id != 0) {
        if (ps.compute_spawn_view.id == 0 and ps.compute_spawn_buffer.id != 0 and
            (!made_spawn_buf or spawn_buf_wins))
        {
            ps.compute_spawn_view = .{ .id = up.created_spawn_view_id };
        } else {
            lost.spawn_view = .{ .id = up.created_spawn_view_id };
        }
        up.created_spawn_view_id = 0;
    }
    if (up.created_draw_view_id != 0) {
        if (ps.compute_draw_view.id == 0 and ps.compute_draw_buffer.id != 0 and
            (!made_draw_buf or draw_buf_wins))
        {
            ps.compute_draw_view = .{ .id = up.created_draw_view_id };
        } else {
            lost.draw_view = .{ .id = up.created_draw_view_id };
        }
        up.created_draw_view_id = 0;
    }
    if (up.created_pipeline_id != 0) {
        if (ps.compute_pipeline.id == 0 and ps.compute_shader.id != 0 and
            (!made_shader or shader_wins))
        {
            ps.compute_pipeline = .{ .id = up.created_pipeline_id };
        } else {
            lost.pipeline = .{ .id = up.created_pipeline_id };
        }
        up.created_pipeline_id = 0;
    }
    if (!lost.isEmpty()) scene.gpu_retire.retireComputeBundle(scene.allocator, lost);
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

fn retireComputeCreated(scene: anytype, up: anytype) void {
    // Owner gone (index/token mismatch): every created handle rides one
    // coherent bundle into the epoch retire queue — buffers exactly as
    // before (same queue, same epoch discipline), views/shader/pipeline
    // now alongside them instead of log-and-leak. Teardown order inside
    // the bundle is dependency-safe (views before buffers, pipeline
    // before shader); zero fields are partial outcomes, skipped.
    var lost: gpu_retire.ComputeBundle = .{};
    if (up.created_state_buffer_id != 0) {
        lost.state_buffer = .{ .id = up.created_state_buffer_id };
        up.created_state_buffer_id = 0;
    }
    if (up.created_spawn_buffer_id != 0) {
        lost.spawn_buffer = .{ .id = up.created_spawn_buffer_id };
        up.created_spawn_buffer_id = 0;
    }
    if (up.created_draw_buffer_id != 0) {
        lost.draw_buffer = .{ .id = up.created_draw_buffer_id };
        up.created_draw_buffer_id = 0;
    }
    if (up.created_state_view_id != 0) {
        lost.state_view = .{ .id = up.created_state_view_id };
        up.created_state_view_id = 0;
    }
    if (up.created_spawn_view_id != 0) {
        lost.spawn_view = .{ .id = up.created_spawn_view_id };
        up.created_spawn_view_id = 0;
    }
    if (up.created_draw_view_id != 0) {
        lost.draw_view = .{ .id = up.created_draw_view_id };
        up.created_draw_view_id = 0;
    }
    if (up.created_shader_id != 0) {
        lost.shader = .{ .id = up.created_shader_id };
        up.created_shader_id = 0;
    }
    if (up.created_pipeline_id != 0) {
        lost.pipeline = .{ .id = up.created_pipeline_id };
        up.created_pipeline_id = 0;
    }
    if (!lost.isEmpty()) scene.gpu_retire.retireComputeBundle(scene.allocator, lost);
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
            if (up.dirty) tm.gpu_dirty = true;
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
        // the quiesced drain likewise skips the upload and still publishes
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
            if (up.dirty) b.upload_pending = true;
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
        // the quiesced drain likewise skips the upload and consumes the
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
            if (up.dirty) gl.gpu_dirty = true;
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
            // Images/views cannot travel through the retire queue (its
            // entries are meshes/buffers/probe/ui3d/compute-bundle kinds):
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
        // panic. The quiesced drain still owns meshes that never went
        // through a staged build.
        if (up.morph_delta_pending) {
            if (m.morph_delta_image.id != 0 or m.morph_delta_view.id != 0) {
                // Defensive only (in practice only the context creates, so
                // the live ids are zero here): the live delta texture
                // already exists, so the mesh is complete — drop the
                // created pair loudly (no image/view retire entry kind)
                // and finish normally.
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
        // drain's finish keeps them too; only the consumed pending copy is
        // freed, game-side (the allocator is thread-safe).
        if (m.pending_vertices.len > 0) {
            scene.allocator.free(m.pending_vertices);
            m.pending_vertices = &.{};
        }
    }
}
