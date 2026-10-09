const std = @import("std");
const math = @import("math");
const BoundingBox = math.BoundingBox;
const types = @import("types.zig");
const Vertex = types.Vertex;
const mesh_mod = @import("mesh.zig");
const Mesh = mesh_mod.Mesh;
const uploadGeometry = mesh_mod.uploadGeometry;

test "finishGpuUpload with pending_dynamic_update stays pending without sg context" {
    const ally = std.testing.allocator;
    var m: Mesh = .{
        .name = "pending_dynamic",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    m.gpu_pending = true;
    m.pending_dynamic_update = true;
    m.pending_vertices = try ally.dupe(Vertex, &[_]Vertex{
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
    });
    m.cpu_indices = try ally.dupe(u32, &[_]u32{ 0, 1, 2 });
    defer m.deinit(ally);
    // No sg context in tests: must stay pending without touching sg.* and
    // keep the retained vertex copy for the later context-thread finish.
    m.finishGpuUpload(ally);
    try std.testing.expect(m.gpu_pending);
    try std.testing.expect(m.pending_dynamic_update);
    try std.testing.expectEqual(@as(usize, 3), m.pending_vertices.len);
    try std.testing.expectEqual(@as(u32, 0), m.vertex_buffer.id);
}

test "finishGpuUpload with morph_upload_pending survives without sg context" {
    const ally = std.testing.allocator;
    var m: Mesh = .{
        .name = "pending_morph",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    m.gpu_pending = true;
    m.morph_upload_pending = true;
    m.pending_vertices = try ally.dupe(Vertex, &[_]Vertex{
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
    });
    m.cpu_indices = try ally.dupe(u32, &[_]u32{ 0, 1, 2 });
    defer m.deinit(ally);
    // No sg context: the delta-texture upload must not run; the flag stays
    // set so the context-thread flush retries it.
    m.finishGpuUpload(ally);
    try std.testing.expect(m.gpu_pending);
    try std.testing.expect(m.morph_upload_pending);
    try std.testing.expectEqual(@as(usize, 3), m.pending_vertices.len);
}

test "uploadGeometry defers without sg context and finish preserves pending" {
    const testScene = @import("../testing.zig").testScene;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var scene = testScene(alloc);
    var verts = [_]Vertex{
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
    };
    var idx = [_]u32{ 0, 1, 2 };
    const m = try uploadGeometry(&scene, "deferred", .{
        .vertices = &verts,
        .indices = &idx,
        .bounds = BoundingBox.zero,
    });
    try std.testing.expect(m.gpu_pending);
    try std.testing.expectEqual(@as(usize, 3), m.pending_vertices.len);
    try std.testing.expectEqual(@as(usize, 3), m.cpu_positions.len);
    try std.testing.expectEqual(@as(usize, 3), m.cpu_indices.len);
    // Still no sg context: the finish attempt must keep everything pending.
    m.finishGpuUpload(alloc);
    try std.testing.expect(m.gpu_pending);
    try std.testing.expectEqual(@as(usize, 3), m.pending_vertices.len);
}

test "stage-2A: mesh uid assigned lazily, stable, never zero" {
    var a: Mesh = .{ .name = "uid_a", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3 };
    var b: Mesh = .{ .name = "uid_b", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3 };
    try std.testing.expectEqual(@as(u64, 0), a.uid);
    const ua1 = a.ensureUid();
    const ua2 = a.ensureUid();
    try std.testing.expect(ua1 != 0);
    try std.testing.expectEqual(ua1, ua2);
    try std.testing.expectEqual(ua1, a.uid);
    const ub = b.ensureUid();
    try std.testing.expect(ub != 0);
    try std.testing.expect(ub != ua1);
}

test "stage-2A: instanceRenderSource returns instance_render" {
    var m: Mesh = .{ .name = "src", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3 };
    m.instance_render.count = 7;
    m.instance_build_view.count = 9;
    const pub_src = m.instanceRenderSource(.published);
    try std.testing.expectEqual(@as(u32, 7), pub_src.count);
    try std.testing.expectEqual(&m.instance_render, pub_src);
    const bv_src = m.instanceRenderSource(.build_view);
    try std.testing.expectEqual(@as(u32, 9), bv_src.count);
    try std.testing.expectEqual(&m.instance_build_view, bv_src);
}

test "Mesh tags operations and query" {
    const alloc = std.testing.allocator;
    var m: Mesh = .{ .name = "orc", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3 };
    defer m.deinit(alloc);

    _ = try m.addTags(alloc, "enemy, orc, melee");
    try std.testing.expect(m.hasTag("enemy"));
    try std.testing.expect(m.hasTag("ORC"));
    try std.testing.expect(!m.hasTag("boss"));

    try std.testing.expect(m.matchesTagQuery("enemy && (orc || goblin)"));
    try std.testing.expect(!m.matchesTagQuery("enemy && boss"));
    try std.testing.expect(m.matchesTagQuery("!boss && melee"));
}
