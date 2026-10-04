const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const camera_mod = @import("../camera.zig");
const Camera = camera_mod.Camera;
const scene_mod = @import("../scene.zig");
const Scene = scene_mod.Scene;
const scene_probes = @import("probe_layer.zig");
const gpu_thread = @import("../gpu_thread.zig");

fn buildForTest(scene: *Scene) !void {
    try std.testing.expect(scene.buildPreparedFrame());
}
fn finishForTest(scene: *Scene) void {
    const claim = scene.beginStagedPrepare() orelse std.debug.panic("{s}", .{"finishForTest: no fresh staged build"});
    scene.finishStagedPrepare(claim);
}
fn stageAndPrepareForTest(scene: *Scene) !void {
    try buildForTest(scene);
    finishForTest(scene);
}

test "reflection probes: add/remove/capacity/dirty bookkeeping" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    // testScene leaves the retire queue undefined: zero it explicitly (the
    // removal path retires through it, even for empty pre-capture targets).
    scene.gpu_retire = .{};
    defer scene.gpu_retire.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 0), scene.reflectionProbeCount());
    try std.testing.expect(scene.getReflectionProbe(0) == null);

    const a = try scene.addReflectionProbe(Vec3.new(1, 0, 0), .{});
    try std.testing.expectEqual(@as(usize, 0), a);
    _ = try scene.addReflectionProbe(Vec3.zero, .{ .enabled = false });
    _ = try scene.addReflectionProbe(Vec3.zero, .{});
    _ = try scene.addReflectionProbe(Vec3.zero, .{});
    try std.testing.expectEqual(@as(usize, 4), scene.reflectionProbeCount());
    // Past the cap: hard error, never a silent clamp or replacement.
    try std.testing.expectError(error.TooManyReflectionProbes, scene.addReflectionProbe(Vec3.zero, .{}));
    try std.testing.expectEqual(@as(usize, 4), scene.reflectionProbeCount());
    // Fresh probes all start dirty.
    try std.testing.expectEqual(@as(usize, 4), scene.probeDirtyCount());

    // Live state is mutable through the accessor; OOB access is null.
    const p = scene.getReflectionProbe(0).?;
    try std.testing.expectEqual(Vec3.new(1, 0, 0), p.position);
    p.intensity = 0.25;
    try std.testing.expectEqual(@as(f32, 0.25), scene.getReflectionProbe(0).?.intensity);
    try std.testing.expect(scene.getReflectionProbe(9) == null);

    // On-demand capture requests are pure bookkeeping (the GPU work runs in
    // render, which needs a real context and is covered there).
    scene.probes.notifyCaptured(0); // simulate one landed capture
    try std.testing.expectEqual(@as(usize, 3), scene.probeDirtyCount());
    scene.captureReflectionProbe(0);
    try std.testing.expectEqual(@as(usize, 4), scene.probeDirtyCount());
    scene.captureDirtyReflectionProbes();
    try std.testing.expectEqual(@as(usize, 4), scene.probeDirtyCount());
    scene.captureReflectionProbe(99); // OOB: no-op.

    // Removal retires the target (here empty: uniform no-op-destroy entry)
    // and shifts higher indices down, order-preserving.
    scene.removeReflectionProbe(0);
    try std.testing.expectEqual(@as(usize, 3), scene.reflectionProbeCount());
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    scene.removeReflectionProbe(99); // OOB: no-op.
    try std.testing.expectEqual(@as(usize, 3), scene.reflectionProbeCount());
    // Drain through the epoch discipline: complete the open epoch, flush.
    scene.gpu_retire.complete(scene.gpu_retire.current());
    scene.gpu_retire.flush(alloc);
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "reflection probe state packs into the frame snapshot" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    scene.gpu_retire = .{};
    defer scene.gpu_retire.deinit(alloc);
    // testScene leaves non-tested subsystems undefined: zero everything
    // the prepare path below can observe (no luck-with-zeroed-stack).
    scene.meshes = .empty;
    scene.outline_meshes = .empty;
    scene.cameras = .empty;
    scene.draws = .{};
    scene.frame_handoff = .{};
    scene.light_handoff = .{};
    scene.uploads = null;
    scene.particles.systems = .empty;
    scene.particles.frame = .empty;
    scene.physics.world = null;
    scene.physics.debug_lines = .empty;
    scene.physics.prepared_lines = .empty;
    scene.physics.build_lines = .empty;
    scene.build_seq.store(0, .monotonic);
    scene.last_latched_seq.store(0, .monotonic);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // No probes: empty pack (every draw takes the legacy path).
    var snap = scene.packFrameSnapshot(1.0, 640, 480);
    try std.testing.expect(snap.has_camera);
    try std.testing.expectEqual(@as(usize, 0), snap.probe_pack.count);

    _ = try scene.addReflectionProbe(Vec3.new(1, 2, 3), .{ .radius = 4.0 });
    _ = try scene.addReflectionProbe(Vec3.zero, .{ .enabled = false });
    snap = scene.packFrameSnapshot(1.0, 640, 480);
    try std.testing.expectEqual(@as(usize, 2), snap.probe_pack.count);
    try std.testing.expectEqual(Vec3.new(1, 2, 3), snap.probe_pack.entries[0].position);
    try std.testing.expectEqual(@as(f32, 4.0), snap.probe_pack.entries[0].radius);
    try std.testing.expect(snap.probe_pack.entries[0].enabled);
    try std.testing.expect(!snap.probe_pack.entries[1].enabled);
    // Nothing captured headlessly: selection skips everything (disabled-
    // probe path leaves the draw state untouched — staged empty).
    try std.testing.expect(!snap.probe_pack.entries[0].captured);
    try std.testing.expect(scene_probes.selectProbe(snap.probe_pack.entries[0..snap.probe_pack.count], Vec3.new(1, 2, 3)) == null);

    // The pack rides the prepare path into the consumed snapshot (the same
    // copy renderReuse later re-presents without capturing): publish, then
    // prepare latches it verbatim.
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 2), scene.frame_snapshot.probe_pack.count);
    try std.testing.expectEqual(Vec3.new(1, 2, 3), scene.frame_snapshot.probe_pack.entries[0].position);

    // The snapshot also round-trips the mailbox by value (newest wins, like
    // the light pack): claim, publish the packed generation, take it back.
    // What renderReuse re-presents is exactly what was packed.
    const slot_i = scene.frame_handoff.claim().?;
    scene.frame_handoff.slot(slot_i).* = snap;
    scene.frame_handoff.publish(slot_i);
    var snap_out: scene_mod.SceneFrameSnapshot = undefined;
    try std.testing.expect(scene.frame_handoff.takeLatest(&snap_out));
    try std.testing.expectEqual(@as(usize, 2), snap_out.probe_pack.count);
    try std.testing.expectEqual(@as(f32, 4.0), snap_out.probe_pack.entries[0].radius);
}
