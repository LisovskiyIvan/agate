const std = @import("std");
const sokol = @import("sokol");
const snapshot_mod = @import("snapshot.zig");
const mainTargetBytes = snapshot_mod.mainTargetBytes;
const taaHistoryBytes = snapshot_mod.taaHistoryBytes;
const bloomPyramidBytes = snapshot_mod.bloomPyramidBytes;
const rtImageBytes = snapshot_mod.rtImageBytes;

test "hdr census byte math" {
    // 1x HDR main: 8 color + 4 depth = 12 B/px.
    try std.testing.expectEqual(@as(usize, 12), mainTargetBytes(1, 1, 1));
    // 4x MSAA: 8*4 + 4*4 color/depth store + one 1x 8 resolve = 56 B/px.
    try std.testing.expectEqual(@as(usize, 56), mainTargetBytes(1, 1, 4));
    // TAA history ping-pong: 2 full-size RGBA16F slots = 16 B/px.
    try std.testing.expectEqual(@as(usize, 16), taaHistoryBytes(1, 1));
    try std.testing.expectEqual(@as(usize, 2 * 4 * 8 * 2), taaHistoryBytes(2, 4));
    // Odd bloom base uses actual floored mip dims, not coarse scaling:
    // base 3x3 -> mip0 1x1, so one level down+up = 2*1*1*8.
    try std.testing.expectEqual(@as(usize, 16), bloomPyramidBytes(3, 3, 1, 8));
    // Single live image, single mip: plain w*h*bpp*samples.
    try std.testing.expectEqual(@as(usize, 640 * 360 * 8), rtImageBytes(640, 360, 1, 1, 8));
    try std.testing.expectEqual(@as(usize, 0), rtImageBytes(0, 8, 1, 1, 8));
}

test "Profiler memory snapshot and file saving" {
    const Profiler = @import("core.zig").Profiler;
    const Mesh = @import("../mesh.zig").Mesh;
    sokol.time.setup();
    const ally = std.testing.allocator;
    const testScene = @import("../testing.zig").testScene;
    var scene = testScene(ally);
    defer {
        @import("../scene/content.zig").deinitMeshes(ally, &scene.meshes);
        scene.profiler.deinit();
    }

    const m = try ally.create(Mesh);
    m.* = @import("../testing.zig").testMesh("TestCube");
    m.vertex_count = 24;
    m.index_count = 36;
    m.index_type = .UINT16;
    try scene.meshes.append(ally, m);

    var prof = Profiler.init(ally);
    defer prof.deinit();

    const snap = try prof.captureMemorySnapshot(&scene);

    try std.testing.expectEqual(@as(usize, 1), snap.mesh_count);
    try std.testing.expectEqual(@as(usize, 1), snap.meshes.len);
    try std.testing.expectEqualStrings("TestCube", snap.meshes[0].name);

    // Test saving reports into an isolated per-run directory (unique,
    // auto-cleaned): never the shared build-cache root, so a concurrent
    // `zig build` or a stale leftover cannot collide with this test.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_base = try std.fmt.allocPrint(ally, ".zig-cache/tmp/{s}/test_profile_out", .{tmp.sub_path});
    defer ally.free(tmp_base);
    try prof.saveReports(&scene, tmp_base);

    const io = std.Io.Threaded.global_single_threaded.io();
    // Verify files were created (cleanup is owned by tmp.cleanup above,
    // so no explicit deletes: the tree vanishes even on failure).
    const h_path = try std.fmt.allocPrint(ally, "{s}.html", .{tmp_base});
    defer ally.free(h_path);
    const m_path = try std.fmt.allocPrint(ally, "{s}.md", .{tmp_base});
    defer ally.free(m_path);
    const j_path = try std.fmt.allocPrint(ally, "{s}.json", .{tmp_base});
    defer ally.free(j_path);
    const h_file = try std.Io.Dir.cwd().openFile(io, h_path, .{});
    h_file.close(io);
    const m_file = try std.Io.Dir.cwd().openFile(io, m_path, .{});
    m_file.close(io);
    const j_file = try std.Io.Dir.cwd().openFile(io, j_path, .{});
    j_file.close(io);
}
