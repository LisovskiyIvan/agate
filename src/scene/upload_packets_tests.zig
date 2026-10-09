//! Tests for `upload_packets.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const gpu_thread = @import("../gpu_thread.zig");
const morph_gpu = @import("../mesh/morph_gpu.zig");
const prod = @import("upload_packets.zig");
const stageUploads = prod.stageUploads;
const flushSlotUploads = prod.flushSlotUploads;
const commitSlotResults = prod.commitSlotResults;
const restageDroppedSlot = prod.restageDroppedSlot;

// --- Focused regression tests (headless, sg-free assertions on packets) ---

test "upload packets: FAILED-guard sokol surface contract" {
    // Pins the query/destroy surface every VALID-only creation guard above
    // relies on (buffers, image, views, shader, pipeline). Comptime-only:
    // headless runs never call these (every guard sits behind a live
    // makeBuffer on the context thread), so a sokol upgrade that renames
    // or removes one breaks here instead of silently shipping an
    // id==0-only check.
    try comptime std.testing.expect(@TypeOf(sg.queryBufferState) != void);
    try comptime std.testing.expect(@TypeOf(sg.queryImageState) != void);
    try comptime std.testing.expect(@TypeOf(sg.queryViewState) != void);
    try comptime std.testing.expect(@TypeOf(sg.queryShaderState) != void);
    try comptime std.testing.expect(@TypeOf(sg.queryPipelineState) != void);
    try comptime std.testing.expect(@TypeOf(sg.destroyBuffer) != void);
    try comptime std.testing.expect(@TypeOf(sg.destroyImage) != void);
    try comptime std.testing.expect(@TypeOf(sg.destroyView) != void);
    try comptime std.testing.expect(@TypeOf(sg.destroyShader) != void);
    try comptime std.testing.expect(@TypeOf(sg.destroyPipeline) != void);
}

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

test "upload packets: duplicate compute creation retires the loser, winner installs" {
    // Deterministic replay of the stale-freeze race: two in-flight packets
    // for one system (both frozen while live was zero, both flushed into
    // full created sets). The first committed outcome installs; the second
    // finds every live id set, so all eight handles ride ONE coherent
    // retire bundle — no log-and-leak, no dangling installs.
    const t = std.testing;
    gpu_thread.markContextThread();
    const sys_mod = @import("../particles/system.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var ps = try sys_mod.makeComputeSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
    ps.compute_buffers_pending = true;
    var systems = [_]*sys_mod.ParticleSystem{&ps};
    var no_meshes: []*@import("../mesh/mesh.zig").Mesh = &.{};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var retire: retire_mod.GpuRetireQueue = .{};
    // Teardown for fake-id outcomes (no GPU resource behind handles):
    // entries verified below are cleared manually without calling sg.destroy*.
    defer {
        retire.pending.clearRetainingCapacity();
        @memset(&retire.overflow, null);
        retire.overflow_len = 0;
        retire.pending.deinit(t.allocator);
    }
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
    const token = @intFromPtr(&ps);
    try slot.p_compute_uploads.append(t.allocator, .{
        .token = token,
        .sys_index = 0,
        .delivered = true,
        .created_state_buffer_id = 101,
        .created_spawn_buffer_id = 102,
        .created_draw_buffer_id = 103,
        .created_state_view_id = 104,
        .created_spawn_view_id = 105,
        .created_draw_view_id = 106,
        .created_shader_id = 107,
        .created_pipeline_id = 108,
    });
    try slot.p_compute_uploads.append(t.allocator, .{
        .token = token,
        .sys_index = 0,
        .delivered = true,
        .created_state_buffer_id = 201,
        .created_spawn_buffer_id = 202,
        .created_draw_buffer_id = 203,
        .created_state_view_id = 204,
        .created_spawn_view_id = 205,
        .created_draw_view_id = 206,
        .created_shader_id = 207,
        .created_pipeline_id = 208,
    });

    commitSlotResults(&fake_scene, &slot);

    // Winner installs the full set; the pending request resolves.
    try t.expectEqual(@as(u32, 101), ps.compute_state_buffer.id);
    try t.expectEqual(@as(u32, 102), ps.compute_spawn_buffer.id);
    try t.expectEqual(@as(u32, 103), ps.compute_draw_buffer.id);
    try t.expectEqual(@as(u32, 104), ps.compute_state_view.id);
    try t.expectEqual(@as(u32, 105), ps.compute_spawn_view.id);
    try t.expectEqual(@as(u32, 106), ps.compute_draw_view.id);
    try t.expectEqual(@as(u32, 107), ps.compute_shader.id);
    try t.expectEqual(@as(u32, 108), ps.compute_pipeline.id);
    try t.expect(!ps.compute_buffers_pending);
    // Loser retires as exactly one coherent bundle (not eight entries,
    // not a log): buffers, views, shader, pipeline together.
    try t.expectEqual(@as(usize, 1), retire.retainedCount());
    try t.expectEqual(retire_mod.Kind.compute, retire.pending.items[0].kind);
    const lost = retire.pending.items[0].compute;
    try t.expectEqual(@as(u32, 201), lost.state_buffer.id);
    try t.expectEqual(@as(u32, 202), lost.spawn_buffer.id);
    try t.expectEqual(@as(u32, 203), lost.draw_buffer.id);
    try t.expectEqual(@as(u32, 204), lost.state_view.id);
    try t.expectEqual(@as(u32, 205), lost.spawn_view.id);
    try t.expectEqual(@as(u32, 206), lost.draw_view.id);
    try t.expectEqual(@as(u32, 207), lost.shader.id);
    try t.expectEqual(@as(u32, 208), lost.pipeline.id);
    try t.expectEqual(@as(u64, 0), retire.duplicateDropCount());
    try t.expectEqual(@as(u64, 0), retire.cappedDropCount());
    // Outcomes consumed: a repeat commit is a no-op (idempotence carve-out).
    try t.expectEqual(@as(u32, 0), slot.p_compute_uploads.items[0].created_state_buffer_id);
    try t.expectEqual(@as(u32, 0), slot.p_compute_uploads.items[1].created_pipeline_id);
    commitSlotResults(&fake_scene, &slot);
    try t.expectEqual(@as(usize, 1), retire.retainedCount());
    try t.expectEqual(@as(u32, 101), ps.compute_state_buffer.id);
}

test "upload packets: stale compute views lose with their buffers, never dangle" {
    // Partial-race coherence: the first outcome installed buffers only
    // (partial progress); a stale full set commits second. Its buffers
    // lose — and its views MUST lose with them even though their live
    // slots are still zero (they were stamped from the losing buffers and
    // would otherwise dangle). The independent shader still installs, and
    // the pipeline follows its winning shader.
    const t = std.testing;
    gpu_thread.markContextThread();
    const sys_mod = @import("../particles/system.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var ps = try sys_mod.makeComputeSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
    ps.compute_state_buffer = .{ .id = 11 };
    ps.compute_spawn_buffer = .{ .id = 12 };
    ps.compute_draw_buffer = .{ .id = 13 };
    ps.compute_buffers_pending = true;
    var systems = [_]*sys_mod.ParticleSystem{&ps};
    var no_meshes: []*@import("../mesh/mesh.zig").Mesh = &.{};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var retire: retire_mod.GpuRetireQueue = .{};
    // Teardown for fake-id outcomes (no GPU resource behind handles):
    // entries verified below are cleared manually without calling sg.destroy*.
    defer {
        retire.pending.clearRetainingCapacity();
        @memset(&retire.overflow, null);
        retire.overflow_len = 0;
        retire.pending.deinit(t.allocator);
    }
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
    try slot.p_compute_uploads.append(t.allocator, .{
        .token = @intFromPtr(&ps),
        .sys_index = 0,
        .delivered = true,
        .created_state_buffer_id = 21,
        .created_spawn_buffer_id = 22,
        .created_draw_buffer_id = 23,
        .created_state_view_id = 24,
        .created_spawn_view_id = 25,
        .created_draw_view_id = 26,
        .created_shader_id = 27,
        .created_pipeline_id = 28,
    });

    commitSlotResults(&fake_scene, &slot);

    // Buffers stay (first set wins); views must NOT install over the zero
    // live slots — their backing buffers lost.
    try t.expectEqual(@as(u32, 11), ps.compute_state_buffer.id);
    try t.expectEqual(@as(u32, 12), ps.compute_spawn_buffer.id);
    try t.expectEqual(@as(u32, 13), ps.compute_draw_buffer.id);
    try t.expectEqual(@as(u32, 0), ps.compute_state_view.id);
    try t.expectEqual(@as(u32, 0), ps.compute_spawn_view.id);
    try t.expectEqual(@as(u32, 0), ps.compute_draw_view.id);
    // Shader is dependency-free: installs. Pipeline follows its winner.
    try t.expectEqual(@as(u32, 27), ps.compute_shader.id);
    try t.expectEqual(@as(u32, 28), ps.compute_pipeline.id);
    // The set is incomplete (views missing): the request stays pending.
    try t.expect(ps.compute_buffers_pending);
    // Losers retire together: three buffers + three views, no shader or
    // pipeline in the bundle (those installed).
    try t.expectEqual(@as(usize, 1), retire.retainedCount());
    try t.expectEqual(retire_mod.Kind.compute, retire.pending.items[0].kind);
    const lost = retire.pending.items[0].compute;
    try t.expectEqual(@as(u32, 21), lost.state_buffer.id);
    try t.expectEqual(@as(u32, 22), lost.spawn_buffer.id);
    try t.expectEqual(@as(u32, 23), lost.draw_buffer.id);
    try t.expectEqual(@as(u32, 24), lost.state_view.id);
    try t.expectEqual(@as(u32, 25), lost.spawn_view.id);
    try t.expectEqual(@as(u32, 26), lost.draw_view.id);
    try t.expectEqual(@as(u32, 0), lost.shader.id);
    try t.expectEqual(@as(u32, 0), lost.pipeline.id);
}

test "upload packets: owner-gone compute outcome retires whole, live stands" {
    // Stale owner (system removed between stage and commit): every created
    // handle — buffers exactly as before, views/shader/pipeline now too
    // instead of log-and-leak — rides one bundle; live state is untouched.
    const t = std.testing;
    gpu_thread.markContextThread();
    const sys_mod = @import("../particles/system.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var ps = try sys_mod.makeComputeSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
    ps.compute_buffers_pending = true;
    var systems = [_]*sys_mod.ParticleSystem{&ps};
    var no_meshes: []*@import("../mesh/mesh.zig").Mesh = &.{};
    var no_trails: []*@import("../mesh/trail.zig").TrailMesh = &.{};
    var no_bodies: []*@import("../softbody.zig").SoftBody = &.{};
    var no_lines: []*@import("../mesh/greased_line.zig").GreasedLineMesh = &.{};
    var retire: retire_mod.GpuRetireQueue = .{};
    // Teardown for fake-id outcomes (no GPU resource behind handles):
    // entries verified below are cleared manually without calling sg.destroy*.
    defer {
        retire.pending.clearRetainingCapacity();
        @memset(&retire.overflow, null);
        retire.overflow_len = 0;
        retire.pending.deinit(t.allocator);
    }
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
    try slot.p_compute_uploads.append(t.allocator, .{
        .token = @intFromPtr(&ps) +% 1, // owner replaced: pointer mismatch
        .sys_index = 0,
        .delivered = true,
        .created_state_buffer_id = 31,
        .created_spawn_buffer_id = 32,
        .created_draw_buffer_id = 33,
        .created_state_view_id = 34,
        .created_spawn_view_id = 35,
        .created_draw_view_id = 36,
        .created_shader_id = 37,
        .created_pipeline_id = 38,
    });

    commitSlotResults(&fake_scene, &slot);

    // Nothing installed on the unrelated live owner; flags untouched.
    try t.expectEqual(@as(u32, 0), ps.compute_state_buffer.id);
    try t.expectEqual(@as(u32, 0), ps.compute_shader.id);
    try t.expect(ps.compute_buffers_pending);
    // All eight handles queued as one bundle.
    try t.expectEqual(@as(usize, 1), retire.retainedCount());
    try t.expectEqual(retire_mod.Kind.compute, retire.pending.items[0].kind);
    const lost = retire.pending.items[0].compute;
    try t.expectEqual(@as(u32, 31), lost.state_buffer.id);
    try t.expectEqual(@as(u32, 32), lost.spawn_buffer.id);
    try t.expectEqual(@as(u32, 33), lost.draw_buffer.id);
    try t.expectEqual(@as(u32, 34), lost.state_view.id);
    try t.expectEqual(@as(u32, 35), lost.spawn_view.id);
    try t.expectEqual(@as(u32, 36), lost.draw_view.id);
    try t.expectEqual(@as(u32, 37), lost.shader.id);
    try t.expectEqual(@as(u32, 38), lost.pipeline.id);
    try t.expectEqual(@as(u32, 0), slot.p_compute_uploads.items[0].created_shader_id);
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

test "upload packets: compute dispatch count transfers exactly once, survives reflush" {
    // Real-GPU gate regression: the staged flush issues sg.dispatch but the
    // old outcome carried no count, so compute_dispatches stayed 0 while
    // waves ran. The slot-owned `dispatches` attempt counter closes it:
    // the flush increments after each real dispatch, the commit transfers
    // at the verified token exactly once (even on later undelivered
    // retries), a wrong token credits nothing, and a headless re-flush
    // preserves the count without adding to it.
    const t = std.testing;
    gpu_thread.markContextThread();
    const sys_mod = @import("../particles/system.zig");
    const frame_draws = @import("frame_draws.zig");
    const retire_mod = @import("gpu_retire.zig");
    var slot = frame_draws.FrameDrawSlot{};
    defer slot.deinit(t.allocator);

    var ps = try sys_mod.makeComputeSystem(t.allocator, 4);
    defer sys_mod.freeTestSystem(&ps);
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
    const token = @intFromPtr(&ps);
    try slot.p_compute_uploads.append(t.allocator, .{
        .token = token,
        .sys_index = 0,
        .capacity = 4,
        .delivered = true,
        .dispatches = 2,
    });

    // Delivered: count transfers, outcome zeroes (exactly-once).
    commitSlotResults(&fake_scene, &slot);
    try t.expectEqual(@as(u64, 2), ps.compute_dispatches);
    try t.expectEqual(@as(u64, 0), slot.p_compute_uploads.items[0].dispatches);
    // Repeat commit: no extra credit.
    commitSlotResults(&fake_scene, &slot);
    try t.expectEqual(@as(u64, 2), ps.compute_dispatches);

    // Undelivered retry carrying a prior actual attempt: still counts
    // one-shot on top of the earlier transfer.
    slot.p_compute_uploads.items[0].delivered = false;
    slot.p_compute_uploads.items[0].dispatches = 3;
    commitSlotResults(&fake_scene, &slot);
    try t.expectEqual(@as(u64, 5), ps.compute_dispatches);
    try t.expectEqual(@as(u64, 0), slot.p_compute_uploads.items[0].dispatches);

    // Wrong token: no credit to the unrelated live object.
    slot.p_compute_uploads.items[0].token +%= 1;
    slot.p_compute_uploads.items[0].delivered = true;
    slot.p_compute_uploads.items[0].dispatches = 7;
    commitSlotResults(&fake_scene, &slot);
    try t.expectEqual(@as(u64, 5), ps.compute_dispatches);
    try t.expectEqual(@as(u64, 0), slot.p_compute_uploads.items[0].dispatches);

    // Headless re-flush preserves the attempt count without crediting the
    // game counter (no sg context => no dispatch => no fake count).
    slot.p_compute_uploads.items[0].token = token;
    slot.p_compute_uploads.items[0].dispatches = 9;
    slot.p_compute_uploads.items[0].delivered = true;
    flushSlotUploads(&fake_scene, &slot);
    try t.expectEqual(@as(u64, 9), slot.p_compute_uploads.items[0].dispatches);
    try t.expect(!slot.p_compute_uploads.items[0].delivered);
    try t.expectEqual(@as(u64, 5), ps.compute_dispatches);
}
