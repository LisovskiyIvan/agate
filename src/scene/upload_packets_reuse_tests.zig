//! Reuse-frame contract tests for the transient rewrite path.
//!
//! Pins the two documented uncovered corners of `rewriteTransientWrites`
//! (see docs/frame-pipeline.md, section on transient buffers in the reuse
//! frame, and the rewrite doc in `upload_packets_transient.zig`):
//!
//! (a) Pending-mesh creation ids: mesh creation ids pending at reuse-frame
//!     time were consumed by the commit, so the rewrite must not resurrect
//!     them (there is no pending-mesh loop in the rewrite at all).
//! (b) Compute-particle state buffers: the render-time compute path re-runs
//!     on a reuse frame, so the rewrite must exclude the state clear
//!     semantically (replaying it would wipe simulation state).
//!
//! Headless testability note: `rewriteTransientWrites` returns early without
//! an sg context, so write emission itself needs a GPU and is NOT pinned
//! here. What IS pinned headless is the contract's observable half: the
//! rewrite never mutates slot descriptors (no `delivered`/outcome writes,
//! no consumed-window accounting, no dispatch counts) and never touches
//! live flags, handles, or arrays — so consumed pending ids stay consumed
//! and a frozen compute state-clear still rides its packet for the
//! render-time path instead of being replayed by the rewrite.
const std = @import("std");
const gpu_thread = @import("../gpu_thread.zig");
const prod = @import("upload_packets.zig");
const stageUploads = prod.stageUploads;
const flushSlotUploads = prod.flushSlotUploads;
const commitSlotResults = prod.commitSlotResults;
const rewriteTransientWrites = prod.rewriteTransientWrites;

test "reuse rewrite: consumed pending-mesh creation ids are not resurrected" {
    const t = std.testing;
    gpu_thread.markContextThread();
    const mesh_types = @import("../mesh/types.zig");
    const mesh_mod = @import("../mesh/mesh.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var pmesh = mesh_mod.Mesh{
        .name = "reuse_pending",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .gpu_pending = true,
        .pending_dynamic_update = false,
    };
    var pverts = try t.allocator.dupe(mesh_types.Vertex, &[_]mesh_types.Vertex{ std.mem.zeroes(mesh_types.Vertex), std.mem.zeroes(mesh_types.Vertex) });
    defer if (pmesh.pending_vertices.len > 0) t.allocator.free(pmesh.pending_vertices);
    pverts[0].position = .{ 8, 8, 8 };
    const pidx = try t.allocator.dupe(u32, &[_]u32{ 0, 1, 2 });
    defer t.allocator.free(pidx);
    pmesh.pending_vertices = pverts;
    pmesh.cpu_indices = pidx;
    _ = pmesh.ensureUid();
    var meshes = [_]*mesh_mod.Mesh{&pmesh};
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
    try t.expect(!pmesh.gpu_pending);

    // Pre-commit rewrite: the frozen creation request is still in flight,
    // so the rewrite must neither deliver it nor re-arm the consumed flag.
    const frozen = slot.pending_uploads.items[0];
    rewriteTransientWrites(&fake_scene, &slot);
    try t.expect(std.meta.eql(frozen, slot.pending_uploads.items[0]));
    try t.expect(!slot.pending_uploads.items[0].delivered);
    try t.expect(!pmesh.gpu_pending);
    try t.expectEqual(@as(u32, 0), pmesh.vertex_buffer.id);

    // Emulate the live-context flush outcome, then commit: the creation ids
    // are consumed (installed over the zero live ids, outcomes zeroed).
    slot.pending_uploads.items[0].delivered = true;
    slot.pending_uploads.items[0].created_vertex_buffer_id = 61;
    slot.pending_uploads.items[0].created_index_buffer_id = 62;
    commitSlotResults(&fake_scene, &slot);
    try t.expectEqual(@as(u32, 61), pmesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 62), pmesh.index_buffer.id);
    try t.expectEqual(@as(u32, 2), pmesh.vertex_count);
    try t.expect(!pmesh.gpu_pending);
    try t.expectEqual(@as(usize, 0), pmesh.pending_vertices.len);
    try t.expectEqual(@as(u32, 0), slot.pending_uploads.items[0].created_vertex_buffer_id);
    try t.expectEqual(@as(u32, 0), slot.pending_uploads.items[0].created_index_buffer_id);

    // Post-commit rewrite (the reuse frame): the consumed ids must stay
    // consumed — no re-emit, no flag re-arm, no outcome mutation.
    const consumed = slot.pending_uploads.items[0];
    rewriteTransientWrites(&fake_scene, &slot);
    try t.expect(std.meta.eql(consumed, slot.pending_uploads.items[0]));
    try t.expectEqual(@as(u32, 61), pmesh.vertex_buffer.id);
    try t.expectEqual(@as(u32, 62), pmesh.index_buffer.id);
    try t.expect(!pmesh.gpu_pending);
}

test "reuse rewrite: compute state clear is excluded, render-time path owns it" {
    const t = std.testing;
    gpu_thread.markContextThread();
    const sys_mod = @import("../particles/system.zig");
    const slot_types = @import("../particles/types.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var ps = try sys_mod.makeComputeSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
    ps.compute_staging = try t.allocator.alloc(slot_types.GpuParticleSlot, 4);
    ps.compute_staging[0].spawn_pos_time = .{ 5, 6, 7, 8 };
    ps.compute_staging[1].spawn_pos_time = .{ 1, 2, 3, 4 };
    ps.compute_staged = 2;
    ps.compute_stage_base = 0;
    ps.compute_cursor = 2;
    ps.compute_high_water = 2;
    ps.compute_dt_accum = 0.016;
    ps.compute_flush_pending = true;
    ps.compute_state_clear_pending = true;
    var systems = [_]*sys_mod.ParticleSystem{&ps};
    var no_meshes: []*@import("../mesh/mesh.zig").Mesh = &.{};
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
    try t.expectEqual(@as(usize, 1), slot.p_compute_uploads.items.len);
    try t.expect(slot.p_compute_uploads.items[0].state_clear_pending);
    // Stage consumed the live request flags; the frozen packet carries them.
    try t.expect(!ps.compute_flush_pending);
    try t.expect(!ps.compute_state_clear_pending);

    // The rewrite replays no compute state: descriptor, ring, dt, and
    // dispatch accounting are all bit-identical afterwards, and the frozen
    // clear still rides the packet for the render-time compute path.
    const frozen = slot.p_compute_uploads.items[0];
    rewriteTransientWrites(&fake_scene, &slot);
    try t.expect(std.meta.eql(frozen, slot.p_compute_uploads.items[0]));
    try t.expect(slot.p_compute_uploads.items[0].state_clear_pending);
    try t.expect(!slot.p_compute_uploads.items[0].delivered);
    try t.expectEqual(@as(u64, 0), slot.p_compute_uploads.items[0].dispatches);
    try t.expectEqual(@as(usize, 2), ps.compute_staged);
    try t.expectEqual(@as(usize, 0), ps.compute_stage_base);
    try t.expectEqual(@as(f32, 0.016), ps.compute_dt_accum);
    try t.expectEqual(@as(u64, 0), ps.compute_dispatches);
    try t.expect(!ps.compute_flush_pending);
    try t.expect(!ps.compute_state_clear_pending);

    // Headless flush: nothing can deliver, nothing live is touched.
    flushSlotUploads(&fake_scene, &slot);
    try t.expect(std.meta.eql(frozen, slot.p_compute_uploads.items[0]));
    try t.expect(!ps.compute_flush_pending);

    // Game-side commit of the undelivered outcome re-arms the window and
    // the creation request, but the live clear flag is NOT re-armed here
    // (unlike the cancelled-claim path in `restageDroppedSlot`, which does
    // re-arm it): headless, the clear request is dropped with the packet
    // while the buffers_pending re-arm carries the retry. A live context
    // would instead have executed the clear in the flush above.
    commitSlotResults(&fake_scene, &slot);
    try t.expect(ps.compute_flush_pending);
    try t.expect(ps.compute_buffers_pending);
    try t.expect(!ps.compute_state_clear_pending);
    try t.expectEqual(@as(usize, 2), ps.compute_staged);
    try t.expectEqual(@as(u64, 0), ps.compute_dispatches);

    // A reuse frame after the commit still replays nothing: outcomes stay
    // undelivered, flags stay as the commit left them.
    const rearmed = slot.p_compute_uploads.items[0];
    rewriteTransientWrites(&fake_scene, &slot);
    try t.expect(std.meta.eql(rearmed, slot.p_compute_uploads.items[0]));
    try t.expect(!slot.p_compute_uploads.items[0].delivered);
    try t.expect(ps.compute_flush_pending);
    try t.expect(ps.compute_buffers_pending);
}
