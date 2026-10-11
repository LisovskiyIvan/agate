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

// Audit (headless): trail drain keeps buffers_pending without a context and
// never adopts anything; greased drain keeps dirty without a context.
test "audit headless: trail buffers_pending survives flush, greased dirty survives" {
    const sokol = @import("sokol");
    const sg = sokol.gfx;
    try std.testing.expect(!sg.isvalid());
    const trail_mod = @import("trail.zig");
    var tmesh = Mesh{ .name = "audit-trail", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    var tverts = [_]Vertex{std.mem.zeroes(Vertex)} ** 4;
    var tidx = [_]u16{0} ** 6;
    var trail = trail_mod.TrailMesh{
        .allocator = std.testing.allocator,
        .scene = undefined,
        .mesh = &tmesh,
        .options = .{},
        .vertices = &tverts,
        .indices = &tidx,
        .buffers_pending = true,
        .pending_min_pt = math.Vec3.zero,
        .pending_max_pt = math.Vec3.zero,
        .gpu_dirty = true,
    };
    trail.flushGpuUploads();
    try std.testing.expect(trail.buffers_pending);
    try std.testing.expect(trail.gpu_dirty);
    try std.testing.expectEqual(@as(u32, 0), tmesh.vertex_buffer.id);

    const greased_mod = @import("greased_line.zig");
    var gmesh = Mesh{ .name = "audit-greased", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    var gverts = [_]Vertex{std.mem.zeroes(Vertex)} ** 4;
    var gidx = [_]u32{0} ** 6;
    var gl = greased_mod.GreasedLineMesh{
        .allocator = std.testing.allocator,
        .scene = undefined,
        .mesh = &gmesh,
        .options = .{},
        .vertices = &gverts,
        .indices = &gidx,
        .gpu_dirty = true,
        .gpu_needs_full_upload = true,
    };
    gl.flushGpuUploads();
    try std.testing.expect(gl.gpu_dirty);
    try std.testing.expectEqual(@as(u32, 0), gmesh.vertex_buffer.id);
}

// LIVE audit (skips headless): trail pair-atomic VALID-only creation and
// FAILED-live release (pending re-armed, payload retained, never written).
// Callable from the native GPU fixture (same path the `LIVE` test runs).
pub fn liveAuditTrailPairAtomic(gpa: std.mem.Allocator) !void {
    const sokol = @import("sokol");
    const sg = sokol.gfx;
    const gpu_thread = @import("../gpu_thread.zig");
    gpu_thread.markContextThread();
    defer gpu_thread.resetContextThreadForTest();
    const trail_mod = @import("trail.zig");
    var tmesh = Mesh{ .name = "live-trail", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    var tverts = [_]Vertex{std.mem.zeroes(Vertex)} ** 4;
    var tidx = [_]u16{ 0, 1, 2, 1, 3, 2 } ** 1;
    var trail = trail_mod.TrailMesh{
        .allocator = gpa,
        .scene = undefined,
        .mesh = &tmesh,
        .options = .{},
        .vertices = &tverts,
        .indices = &tidx,
        .buffers_pending = true,
        .pending_vertex_count = 4,
        .pending_index_count = 6,
        .pending_min_pt = math.Vec3.zero,
        .pending_max_pt = math.Vec3.zero,
        .gpu_dirty = true,
    };
    defer {
        if (tmesh.vertex_buffer.id != 0) sg.destroyBuffer(tmesh.vertex_buffer);
        if (tmesh.index_buffer.id != 0) sg.destroyBuffer(tmesh.index_buffer);
    }
    trail.flushGpuUploads();
    std.debug.print("audit trail: after create pending={} vb_id={d} vb={s} ib_id={d} ib={s} dirty={}\n", .{
        trail.buffers_pending,
        tmesh.vertex_buffer.id,
        @tagName(sg.queryBufferState(tmesh.vertex_buffer)),
        tmesh.index_buffer.id,
        @tagName(sg.queryBufferState(tmesh.index_buffer)),
        trail.gpu_dirty,
    });
    if (!trail.buffers_pending) {
        try std.testing.expect(gpu_thread.createdBufferIsValid(tmesh.vertex_buffer));
        try std.testing.expect(gpu_thread.createdBufferIsValid(tmesh.index_buffer));
    }
    // Pre-fix shape: live FAILED vertex handle. Flush must release the pair,
    // re-arm creation, and keep the payload staged.
    const fb = sg.allocBuffer();
    sg.failBuffer(fb);
    if (tmesh.vertex_buffer.id != 0) sg.destroyBuffer(tmesh.vertex_buffer);
    if (tmesh.index_buffer.id != 0) sg.destroyBuffer(tmesh.index_buffer);
    tmesh.vertex_buffer = fb;
    tmesh.index_buffer = .{};
    trail.buffers_pending = false;
    trail.pending_vertex_count = 4;
    trail.pending_index_count = 0;
    trail.gpu_dirty = true;
    trail.flushGpuUploads();
    std.debug.print("audit trail: after FAILED flush pending={} vb_id={d} counts={d} dirty={}\n", .{
        trail.buffers_pending, tmesh.vertex_buffer.id, trail.pending_vertex_count, trail.gpu_dirty,
    });
    try std.testing.expect(trail.buffers_pending);
    try std.testing.expect(trail.gpu_dirty);
    try std.testing.expectEqual(@as(u32, 0), tmesh.vertex_buffer.id);
    try std.testing.expectEqual(sg.ResourceState.INVALID, sg.queryBufferState(fb));
    try std.testing.expectEqual(@as(usize, 4), trail.pending_vertex_count);
}

test "LIVE audit: trail creation pair-atomic, FAILED-live released" {
    const sokol = @import("sokol");
    const sg = sokol.gfx;
    if (!sg.isvalid()) return error.SkipZigTest;
    try liveAuditTrailPairAtomic(std.testing.allocator);
}

// LIVE audit (skips headless): greased VALID-only creation and FAILED-live
// release. Recovery recreates a VALID pair in the SAME flush and uploads the
// whole payload there (vertex write is unconditional on VALID; the armed
// full upload rides the index write and is consumed) — so a recovered pair
// ends with full_upload=false AND a meter delta of exactly the full payload
// bytes (delivered, not dead). If creation fails, the mesh stays empty and
// the full upload stays armed. Callable from the native GPU fixture (same
// path the `LIVE` test runs).
pub fn liveAuditGreasedValidOnly(gpa: std.mem.Allocator) !void {
    const sokol = @import("sokol");
    const sg = sokol.gfx;
    const gpu_thread = @import("../gpu_thread.zig");
    const upload_meter = @import("../gpu_upload_meter.zig");
    gpu_thread.markContextThread();
    defer gpu_thread.resetContextThreadForTest();
    const greased_mod = @import("greased_line.zig");
    var gmesh = Mesh{ .name = "live-greased", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    var gverts = [_]Vertex{std.mem.zeroes(Vertex)} ** 4;
    var gidx = [_]u32{ 0, 1, 2, 1, 3, 2 } ** 1;
    var gl = greased_mod.GreasedLineMesh{
        .allocator = gpa,
        .scene = undefined,
        .mesh = &gmesh,
        .options = .{},
        .vertices = &gverts,
        .indices = &gidx,
        .gpu_dirty = true,
        .gpu_needs_full_upload = false,
    };
    defer {
        if (gmesh.vertex_buffer.id != 0) sg.destroyBuffer(gmesh.vertex_buffer);
        if (gmesh.index_buffer.id != 0) sg.destroyBuffer(gmesh.index_buffer);
    }
    gl.flushGpuUploads();
    std.debug.print("audit greased: after create vb_id={d} vb={s} ib_id={d} ib={s} full_upload={}\n", .{
        gmesh.vertex_buffer.id,   @tagName(sg.queryBufferState(gmesh.vertex_buffer)),
        gmesh.index_buffer.id,    @tagName(sg.queryBufferState(gmesh.index_buffer)),
        gl.gpu_needs_full_upload,
    });
    if (gmesh.vertex_buffer.id != 0) {
        try std.testing.expect(gpu_thread.createdBufferIsValid(gmesh.vertex_buffer));
        try std.testing.expect(gpu_thread.createdBufferIsValid(gmesh.index_buffer));
    }
    const fb = sg.allocBuffer();
    sg.failBuffer(fb);
    std.debug.print("audit greased: injected FAILED vb id={d} st={s}\n", .{ fb.id, @tagName(sg.queryBufferState(fb)) });
    if (gmesh.vertex_buffer.id != 0) sg.destroyBuffer(gmesh.vertex_buffer);
    if (gmesh.index_buffer.id != 0) sg.destroyBuffer(gmesh.index_buffer);
    gmesh.vertex_buffer = fb;
    gmesh.index_buffer = .{};
    gl.gpu_dirty = true;
    const m0 = upload_meter.peek();
    gl.flushGpuUploads();
    const m1 = upload_meter.peek();
    std.debug.print("audit greased: after FAILED flush vb_id={d} vb={s} ib_id={d} ib={s} full_upload={} dirty={} meter_delta={d}\n", .{
        gmesh.vertex_buffer.id,   @tagName(sg.queryBufferState(gmesh.vertex_buffer)),
        gmesh.index_buffer.id,    @tagName(sg.queryBufferState(gmesh.index_buffer)),
        gl.gpu_needs_full_upload, gl.gpu_dirty,
        m1 - m0,
    });
    // The dead handle is gone in every outcome (destroyed by the cleanup).
    try std.testing.expectEqual(sg.ResourceState.INVALID, sg.queryBufferState(fb));
    if (gmesh.vertex_buffer.id != 0) {
        // Same-flush recovery: both payloads uploaded here — the meter delta
        // is exactly the full vertex + index bytes, which proves delivery;
        // full_upload=false is consumed-by-delivery (same end state as the
        // first successful create above), not dead geometry.
        try std.testing.expect(gpu_thread.createdBufferIsValid(gmesh.vertex_buffer));
        try std.testing.expect(gpu_thread.createdBufferIsValid(gmesh.index_buffer));
        try std.testing.expectEqual(
            4 * @sizeOf(Vertex) + 6 * @sizeOf(u32),
            m1 - m0,
        );
        try std.testing.expect(!gl.gpu_needs_full_upload);
    } else {
        // Creation failed: mesh stays empty, full upload stays armed, and
        // nothing was uploaded by this flush.
        try std.testing.expectEqual(@as(u32, 0), gmesh.index_buffer.id);
        try std.testing.expect(gl.gpu_needs_full_upload);
        try std.testing.expectEqual(m0, m1);
    }
}

test "LIVE audit: greased creation VALID-only, FAILED-live released" {
    const sokol = @import("sokol");
    const sg = sokol.gfx;
    if (!sg.isvalid()) return error.SkipZigTest;
    try liveAuditGreasedValidOnly(std.testing.allocator);
}
