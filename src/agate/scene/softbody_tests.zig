const std = @import("std");
const Scene = @import("../scene.zig").Scene;
const upload_meter = @import("../gpu_upload_meter.zig");
const gpu_thread = @import("../gpu_thread.zig");

fn freeSoftbodyFixture(alloc: std.mem.Allocator, scene: *Scene) void {
    // Bodies first (solver + staging CPU only; meshes stay registered),
    // then meshes, materials, and the retire queue.
    scene.softbodies.deinit(alloc);
    for (scene.meshes.items) |m| {
        m.deinit(alloc);
        alloc.destroy(m);
    }
    scene.meshes.deinit(alloc);
    for (scene.materials.items) |m| alloc.destroy(m);
    scene.materials.deinit(alloc);
    for (scene.pbr_materials.items) |m| alloc.destroy(m);
    scene.pbr_materials.deinit(alloc);
    scene.gpu_retire.deinit(alloc);
    scene.outline_meshes.deinit(alloc);
    scene.lights.deinit(alloc);
}

test "softbody add/get/count/cap/validation/remove errors" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    const b0 = try scene.addSoftBodyCloth("cloth_a", .{ .width = 4, .height = 4 });
    try std.testing.expectEqual(@as(usize, 1), scene.softBodyCount());
    try std.testing.expectEqual(b0, scene.getSoftBody(0).?);
    try std.testing.expectEqual(b0, scene.getSoftBodyByName("cloth_a").?);
    try std.testing.expect(scene.getSoftBody(7) == null);
    try std.testing.expect(scene.getSoftBodyByName("nope") == null);

    // Invalid options never register a body.
    try std.testing.expectError(error.InvalidOptions, scene.addSoftBodyCloth("bad", .{ .width = 1 }));
    try std.testing.expectEqual(@as(usize, 1), scene.softBodyCount());

    _ = try scene.addSoftBodyCloth("cloth_b", .{ .width = 3, .height = 3 });
    _ = try scene.addSoftBodyCloth("cloth_c", .{ .width = 3, .height = 3 });
    _ = try scene.addSoftBodyCloth("cloth_d", .{ .width = 3, .height = 3 });
    try std.testing.expectEqual(@as(usize, 4), scene.softBodyCount());
    try std.testing.expectError(error.TooManySoftBodies, scene.addSoftBodyCloth("cloth_e", .{ .width = 3, .height = 3 }));
    try std.testing.expectError(error.UnknownSoftBody, scene.removeSoftBodyCloth(9));
    try std.testing.expectEqual(@as(usize, 4), scene.softBodyCount());
}

test "softbody mesh coupling: vertex layout matches the solver grid" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    const body = try scene.addSoftBodyCloth("weave", .{ .width = 5, .height = 4, .spacing = 0.5 });
    try std.testing.expectEqual(@as(usize, 20), body.vertices.len);
    try std.testing.expectEqual(@as(u32, 20), body.mesh.vertex_count);
    try std.testing.expectEqual(@as(usize, 4 * 3 * 6), body.indices.len);
    try std.testing.expectEqual(@as(u32, 72), body.mesh.index_count);
    for (body.indices) |idx| try std.testing.expect(idx < 20);
    // Rest pose == solver positions; uv spans the unit square.
    for (body.vertices, body.cloth.pos) |v, p| try std.testing.expectEqual(p.toArray(), v.position);
    try std.testing.expectEqual([2]f32{ 0.0, 0.0 }, body.vertices[0].uv);
    try std.testing.expectEqual([2]f32{ 1.0, 1.0 }, body.vertices[19].uv);
    // Double-sided PBR cloth material; mesh registered in the scene.
    const is_pbr = switch (body.mesh.material.?) {
        .pbr => true,
        else => false,
    };
    try std.testing.expect(is_pbr);
    try std.testing.expect(body.mesh.material.?.pbr.double_sided);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), body.mesh.material.?.pbr.roughness, 0.0001);
    try std.testing.expectEqual(@as(usize, 1), scene.meshes.items.len);
    try std.testing.expect(body.mesh.local_bounding_box.isValid());
}

test "softbody upload flagged once per changed frame, cleared by flush" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);
    _ = upload_meter.takeAndReset();

    const body = try scene.addSoftBodyCloth("flag", .{ .width = 4, .height = 4 });
    try std.testing.expect(!body.upload_pending);
    scene.updateSoftBodies(1.0 / 60.0);
    try std.testing.expect(body.upload_pending);
    // Second changed frame: still exactly one pending flag, never queued.
    scene.updateSoftBodies(1.0 / 60.0);
    try std.testing.expect(body.upload_pending);
    // Staged vertices track the solver.
    for (body.vertices, body.cloth.pos) |v, p| try std.testing.expectEqual(p.toArray(), v.position);
    // Headless flush clears the flag with no sg.* and no meter bytes.
    scene.flushPendingGpuUploads();
    try std.testing.expect(!body.upload_pending);
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());
    // Zero dt steps nothing and flags nothing.
    scene.updateSoftBodies(0.0);
    try std.testing.expect(!body.upload_pending);
    _ = upload_meter.takeAndReset();
}

test "softbody remove retires the mesh (retire-safe, epoch-queued)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    const body = try scene.addSoftBodyCloth("bye", .{ .width = 4, .height = 4 });
    const mesh = body.mesh;
    try scene.removeSoftBodyCloth(0);
    try std.testing.expectEqual(@as(usize, 0), scene.softBodyCount());
    // Mesh unlinked from the registry but alive in the retire queue.
    for (scene.meshes.items) |m| try std.testing.expect(m != mesh);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    // Headless flush completes the teardown (bufferless mesh: no sg.*).
    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "softbody remove from a worker retires without sg.* (update||render)" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    _ = try scene.addSoftBodyCloth("worker", .{ .width = 4, .height = 4 });
    const Job = struct {
        scene: *Scene,
        fn run(j: @This()) void {
            j.scene.removeSoftBodyCloth(0) catch unreachable;
        }
    };
    const t = try std.Thread.spawn(.{}, Job.run, .{Job{ .scene = &scene }});
    t.join();
    try std.testing.expectEqual(@as(usize, 0), scene.softBodyCount());
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    scene.gpu_retire.complete(scene.gpu_retire.current());
    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "softbody destroyMesh drops the bound body (referent cleanup)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    const body = try scene.addSoftBodyCloth("doomed", .{ .width = 4, .height = 4 });
    const mesh = body.mesh;
    scene.destroyMesh(mesh);
    try std.testing.expectEqual(@as(usize, 0), scene.softBodyCount());
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "softbody create OOM rolls back mesh/material/body at every allocation point" {
    // Deterministic failing-allocator sweep over addSoftBodyCloth: every
    // induced OutOfMemory must leave all three registries empty (no
    // registered mesh with a freed name, no dangling material aliasing
    // it, no half-linked body). Success past the last allocation point
    // must register exactly one of each with the material name aliasing
    // the mesh-owned slice. Headless: no sg.* below (buffers stay
    // deferred). Double-free/leak failures surface via the testing
    // allocator + the fixture teardown.
    //
    // Single dimension (fail_index): list growth goes through
    // ensureTotalCapacityPrecise, which falls back to alloc+copy when
    // remap fails, so every growth point is reachable as a raw-alloc
    // failure; a resize_fail_index sweep could never induce OOM here.
    const alloc = std.testing.allocator;
    var saw_induced = false;
    var saw_success = false;
    var n: usize = 0;
    while (n < 64) : (n += 1) {
        var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = n });
        var scene = @import("../testing.zig").testScene(alloc);
        scene.allocator = failing.allocator();
        const res = scene.addSoftBodyCloth("oom_cloth", .{ .width = 4, .height = 4 });
        scene.allocator = alloc;
        if (res) |body| {
            try std.testing.expect(!failing.has_induced_failure);
            saw_success = true;
            try std.testing.expectEqual(@as(usize, 1), scene.softBodyCount());
            try std.testing.expectEqual(@as(usize, 1), scene.meshes.items.len);
            try std.testing.expectEqual(@as(usize, 1), scene.pbr_materials.items.len);
            // Ownership nuance: the material name aliases the mesh-owned
            // slice (freed once via Mesh.deinit/owns_name).
            try std.testing.expect(body.material.name.ptr == body.mesh.name.ptr);
            try std.testing.expect(body.mesh.owns_name);
            freeSoftbodyFixture(alloc, &scene);
        } else |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expect(failing.has_induced_failure);
            saw_induced = true;
            try std.testing.expectEqual(@as(usize, 0), scene.softBodyCount());
            try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
            try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
            freeSoftbodyFixture(alloc, &scene);
        }
    }
    try std.testing.expect(saw_induced);
    try std.testing.expect(saw_success);
}

test "createPBRMaterial append OOM frees the struct (no leak, registry unchanged)" {
    // PBRMaterial.init borrows the name, so creation is exactly two
    // allocations: struct create (#0) then registry-append growth (#1).
    // Index 1 is the previously-leaking path: without the errdefer the
    // struct leaks (DebugAllocator flags it) while the list stays empty.
    const alloc = std.testing.allocator;
    var saw_induced = false;
    var saw_success = false;
    var n: usize = 0;
    while (n < 4) : (n += 1) {
        var scene = @import("../testing.zig").testScene(alloc);
        defer scene.lights.deinit(alloc);
        defer scene.meshes.deinit(alloc);
        defer scene.pbr_materials.deinit(alloc);
        defer scene.gpu_retire.deinit(alloc);
        defer scene.outline_meshes.deinit(alloc);
        errdefer {
            while (scene.pbr_materials.pop()) |m| alloc.destroy(m);
        }
        var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = n });
        scene.allocator = failing.allocator();
        const res = scene.createPBRMaterial("oom_pbr");
        scene.allocator = alloc;
        if (res) |_| {
            try std.testing.expect(!failing.has_induced_failure);
            saw_success = true;
            try std.testing.expectEqual(@as(usize, 1), scene.pbr_materials.items.len);
            alloc.destroy(scene.pbr_materials.pop().?);
        } else |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expect(failing.has_induced_failure);
            saw_induced = true;
            try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
        }
    }
    try std.testing.expect(saw_induced);
    try std.testing.expect(saw_success);
}

test "softbody disabled body pauses: no step, no upload flag" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    const body = try scene.addSoftBodyCloth("paused", .{ .width = 4, .height = 4 });
    body.cloth.enabled = false;
    const h = body.cloth.hashState();
    scene.updateSoftBodies(1.0 / 60.0);
    try std.testing.expectEqual(h, body.cloth.hashState());
    try std.testing.expect(!body.upload_pending);
    body.cloth.enabled = true;
    scene.updateSoftBodies(1.0 / 60.0);
    try std.testing.expect(body.upload_pending);
}
