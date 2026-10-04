const std = @import("std");
const Scene = @import("../scene.zig").Scene;
const Mesh = @import("../mesh.zig").Mesh;
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const BoundingBox = math.BoundingBox;
const Ray = math.Ray;
const gpu_thread = @import("../gpu_thread.zig");
const TrailMesh = @import("../mesh/trail.zig").TrailMesh;
const Vertex = @import("../mesh/types.zig").Vertex;

test "destroyMesh removes the physics body" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.physics.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("body_mesh");
    try scene.meshes.append(alloc, m);

    _ = try scene.createRigidBody(m, .box, 1.0);
    try std.testing.expect(scene.getRigidBody(m) != null);

    scene.destroyMesh(m);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.getWorld().?.bodies.items.len);
}

test "destroyMesh orphans children and detaches bone links" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const parent = try alloc.create(Mesh);
    parent.* = @import("../testing.zig").testMesh("parent");
    try scene.meshes.append(alloc, parent);
    const child = try alloc.create(Mesh);
    child.* = @import("../testing.zig").testMesh("child");
    child.parent = parent;
    try scene.meshes.append(alloc, child);
    const attached = try alloc.create(Mesh);
    attached.* = @import("../testing.zig").testMesh("attached");
    attached.attachToBone(parent, 2);
    try scene.meshes.append(alloc, attached);

    scene.destroyMesh(parent);
    try std.testing.expect(child.parent == null);
    try std.testing.expect(attached.attach_bone == null);
    // No cascade: the orphans stay alive under their own transform.
    try std.testing.expectEqual(@as(usize, 2), scene.meshes.items.len);

    scene.destroyMesh(child);
    scene.destroyMesh(attached);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "destroyMesh removes LOD entries preserving order" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const parent = try alloc.create(Mesh);
    parent.* = @import("../testing.zig").testMesh("lod_parent");
    try scene.meshes.append(alloc, parent);
    const c1 = try alloc.create(Mesh);
    c1.* = @import("../testing.zig").testMesh("lod_c1");
    try scene.meshes.append(alloc, c1);
    const c2 = try alloc.create(Mesh);
    c2.* = @import("../testing.zig").testMesh("lod_c2");
    try scene.meshes.append(alloc, c2);
    const c3 = try alloc.create(Mesh);
    c3.* = @import("../testing.zig").testMesh("lod_c3");
    try scene.meshes.append(alloc, c3);

    try parent.addLODLevel(alloc, 10.0, c1);
    try parent.addLODLevel(alloc, 20.0, c2);
    try parent.addLODLevel(alloc, 30.0, c3);

    scene.destroyMesh(c2);
    try std.testing.expectEqual(@as(usize, 2), parent.lod_levels.items.len);
    try std.testing.expectEqual(@as(f32, 10.0), parent.lod_levels.items[0].distance);
    try std.testing.expect(parent.lod_levels.items[0].mesh.? == c1);
    try std.testing.expectEqual(@as(f32, 30.0), parent.lod_levels.items[1].distance);
    try std.testing.expect(parent.lod_levels.items[1].mesh.? == c3);

    scene.destroyMesh(c1);
    scene.destroyMesh(c3);
    scene.destroyMesh(parent);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "destroyMesh drops decal instances and frees material" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.decals.deinit();
    defer scene.meshes.deinit(alloc);
    defer scene.pbr_materials.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    // Manual instance (no projection/GPU): the manager only stores the
    // mesh/material pair, so removal + material teardown is fully
    // exercisable without projecting any decal geometry.
    const dm = scene.getOrCreateDecalManager(8);
    const dm_mesh = try alloc.create(Mesh);
    dm_mesh.* = @import("../testing.zig").testMesh("decal_mesh");
    try scene.meshes.append(alloc, dm_mesh);
    const mat = try scene.createPBRMaterial("decal_mat");
    try dm.instances.append(alloc, .{
        .mesh = dm_mesh,
        .material = mat,
        .base_color = Color3.white,
        .lifetime = 0.0,
        .fade_duration = 1.0,
    });

    scene.destroyMesh(dm_mesh);
    try std.testing.expectEqual(@as(usize, 0), dm.instances.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "pending destroy overflow drains on flush" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    // Drive the allocation-free spillover directly: the off-context OOM
    // enqueue itself needs fault injection, but the drain path (flush and
    // deinit share it) is covered here with a buffer-free mesh, so no sg.*
    // call is involved. The entry is marked with a completed epoch so flush
    // picks it up — the epoch rule applies to overflow as well.
    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("overflow_mesh");
    const e = scene.gpu_retire.begin();
    scene.gpu_retire.overflow[0] = .{ .kind = .mesh, .mesh = m, .epoch = e };
    scene.gpu_retire.overflow_len = 1;
    scene.gpu_retire.complete(e);

    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "destroyMesh off-context: retention + flush on context thread" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("offctx_mesh");
    try scene.meshes.append(alloc, m);

    // Worker is a non-context thread: destroyMesh must only unlink the mesh
    // (physics/hierarchy/LOD cleaned up) and place it in retention, without sg.*.
    const Job = struct {
        scene: *Scene,
        mesh: *Mesh,
        fn run(j: @This()) void {
            j.scene.destroyMesh(j.mesh);
        }
    };
    const t = try std.Thread.spawn(.{}, Job.run, .{Job{ .scene = &scene, .mesh = m }});
    t.join();
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());

    // End of frame + flush on context thread: retention is freed.
    scene.gpu_retire.complete(scene.gpu_retire.current());
    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "Scene tag queries and tag-filtered raycasting" {
    const alloc = std.testing.allocator;
    var scene: Scene = undefined;
    scene.allocator = alloc;
    scene.meshes = .empty;
    scene.frame_id = 0;
    defer scene.meshes.deinit(alloc);

    var m1 = Mesh{
        .name = "orc_grunt",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(0, 0, 5),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, 4), Vec3.new(1, 1, 6)),
    };
    defer m1.deinit(alloc);
    _ = try m1.addTags(alloc, "enemy, orc, melee");
    try scene.meshes.append(alloc, &m1);

    var m2 = Mesh{
        .name = "orc_boss",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(10, 0, 5),
        .local_bounding_box = BoundingBox.init(Vec3.new(9, -1, 4), Vec3.new(11, 1, 6)),
    };
    defer m2.deinit(alloc);
    _ = try m2.addTags(alloc, "enemy, orc, boss, elite");
    try scene.meshes.append(alloc, &m2);

    var m3 = Mesh{
        .name = "player_hero",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(0, 0, -5),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -6), Vec3.new(1, 1, -4)),
    };
    defer m3.deinit(alloc);
    _ = try m3.addTags(alloc, "player, hero");
    try scene.meshes.append(alloc, &m3);

    // Test getMeshByName
    try std.testing.expectEqual(&m1, scene.getMeshByName("orc_grunt"));
    try std.testing.expectEqual(&m2, scene.getMeshByName("orc_boss"));
    try std.testing.expect(scene.getMeshByName("nonexistent") == null);

    // Test countMeshesByTag
    try std.testing.expectEqual(@as(usize, 2), scene.countMeshesByTag("enemy"));
    try std.testing.expectEqual(@as(usize, 1), scene.countMeshesByTag("boss"));
    try std.testing.expectEqual(@as(usize, 1), scene.countMeshesByTag("player"));
    try std.testing.expectEqual(@as(usize, 0), scene.countMeshesByTag("dragon"));

    // Test getMeshesByTag
    var enemies = try scene.getMeshesByTag(alloc, "enemy");
    defer enemies.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 2), enemies.items.len);

    // Test getMeshesByQuery
    var boss_enemies = try scene.getMeshesByQuery(alloc, "enemy && boss");
    defer boss_enemies.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), boss_enemies.items.len);
    try std.testing.expectEqual(&m2, boss_enemies.items[0]);

    var non_boss = try scene.getMeshesByQuery(alloc, "enemy && !boss");
    defer non_boss.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), non_boss.items.len);
    try std.testing.expectEqual(&m1, non_boss.items[0]);

    // Test findFirstMesh
    try std.testing.expectEqual(&m2, scene.findFirstMeshByTag("boss"));
    try std.testing.expectEqual(&m3, scene.findFirstMeshByQuery("hero"));

    // Test pickWithRayTag: ray looking along +Z at (0, 0, 5) hits m1
    const ray = Ray.new(Vec3.new(0, 0, 0), Vec3.new(0, 0, 1));
    const hit_any_enemy = scene.pickWithRayTag(ray, "enemy");
    try std.testing.expect(hit_any_enemy.hit);
    try std.testing.expectEqual(&m1, hit_any_enemy.picked_mesh.?);

    // Query for boss along the same ray should NOT hit m1 because m1 lacks "boss" tag
    const hit_boss = scene.pickWithRayTag(ray, "boss");
    try std.testing.expect(!hit_boss.hit);

    // Query for hero along the same ray should NOT hit
    const hit_hero = scene.pickWithRayTag(ray, "player");
    try std.testing.expect(!hit_hero.hit);
}

test "renameMesh adopts an owned copy and repeated renames leak nothing" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("helmet_borrowed");
    try scene.meshes.append(alloc, m);

    // Borrowed -> owned: the literal is copied, the flag flips, lookup
    // follows the new name and drops the old one.
    try scene.renameMesh(m, "Damaged Helmet (PBR)");
    try std.testing.expect(m.owns_name);
    try std.testing.expectEqualStrings("Damaged Helmet (PBR)", m.name);
    try std.testing.expect(scene.getMeshByName("Damaged Helmet (PBR)") == m);
    try std.testing.expect(scene.getMeshByName("helmet_borrowed") == null);

    // Owned -> owned: the previous allocation is freed (the testing
    // allocator fails the test on leak) and lookup follows again.
    try scene.renameMesh(m, "Fox Character");
    try std.testing.expect(m.owns_name);
    try std.testing.expectEqualStrings("Fox Character", m.name);
    try std.testing.expect(scene.getMeshByName("Fox Character") == m);
    try std.testing.expect(scene.getMeshByName("Damaged Helmet (PBR)") == null);

    scene.destroyMesh(m);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "renameMesh with aliased input preserves content" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("alias_borrowed");
    try scene.meshes.append(alloc, m);
    try scene.renameMesh(m, "alias_me");
    try std.testing.expect(m.owns_name);

    // Whole-slice alias: copy-before-free keeps the content alive.
    try scene.renameMesh(m, m.name);
    try std.testing.expect(m.owns_name);
    try std.testing.expectEqualStrings("alias_me", m.name);
    try std.testing.expect(scene.getMeshByName("alias_me") == m);

    // Subslice alias into the owned allocation: same guarantee.
    try scene.renameMesh(m, m.name[0..5]);
    try std.testing.expect(m.owns_name);
    try std.testing.expectEqualStrings("alias", m.name);
    try std.testing.expect(scene.getMeshByName("alias") == m);
    try std.testing.expect(scene.getMeshByName("alias_me") == null);

    scene.destroyMesh(m);
}

test "renameMesh OOM keeps the old name and flag" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("old_name");
    try scene.meshes.append(alloc, m);

    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    const saved = scene.allocator;
    scene.allocator = failing.allocator();
    const res = scene.renameMesh(m, "new_name");
    scene.allocator = saved;
    try std.testing.expectError(error.OutOfMemory, res);
    // Atomic failure: old slice and flag untouched, lookup unchanged.
    try std.testing.expect(!m.owns_name);
    try std.testing.expectEqualStrings("old_name", m.name);
    try std.testing.expect(scene.getMeshByName("old_name") == m);
    try std.testing.expect(scene.getMeshByName("new_name") == null);

    scene.destroyMesh(m);
}

test "destroyTrailMesh unlinks the layer and mesh and clears followers" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.trails.deinit(alloc);
    // Failure-safe: the deferred list deinitializers free only backing
    // storage (`trails.deinit` frees remaining trail structs, but
    // `meshes.deinit` does not free mesh structs), so a mid-test `try`
    // failure would leak the still-linked meshes without this drain.
    // Retired entries stay owned by `gpu_retire` and are untouched here.
    errdefer {
        for (scene.meshes.items) |m| {
            m.deinit(alloc);
            alloc.destroy(m);
        }
    }

    const mesh_a = try alloc.create(Mesh);
    mesh_a.* = @import("../testing.zig").testMesh("trail_a_mesh");
    try scene.meshes.append(alloc, mesh_a);
    const ta = try alloc.create(TrailMesh);
    ta.* = .{
        .allocator = alloc,
        .scene = &scene,
        .mesh = mesh_a,
        .options = .{},
        .vertices = try alloc.alloc(Vertex, 4),
        .indices = try alloc.alloc(u16, 6),
    };
    try scene.trails.meshes.append(alloc, ta);

    const mesh_b = try alloc.create(Mesh);
    mesh_b.* = @import("../testing.zig").testMesh("trail_b_mesh");
    try scene.meshes.append(alloc, mesh_b);
    const tb = try alloc.create(TrailMesh);
    tb.* = .{
        .allocator = alloc,
        .scene = &scene,
        .mesh = mesh_b,
        .options = .{},
        .vertices = try alloc.alloc(Vertex, 4),
        .indices = try alloc.alloc(u16, 6),
    };
    tb.target = mesh_a;
    try scene.trails.meshes.append(alloc, tb);

    scene.destroyTrailMesh(ta);
    try std.testing.expectEqual(@as(usize, 1), scene.trails.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 1), scene.meshes.items.len);
    try std.testing.expect(scene.getMeshByName("trail_a_mesh") == null);
    try std.testing.expect(scene.getMeshByName("trail_b_mesh") == mesh_b);
    // The follower no longer points at the freed mesh.
    try std.testing.expect(tb.target == null);

    scene.destroyTrailMesh(tb);
    try std.testing.expectEqual(@as(usize, 0), scene.trails.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "destroyMesh off-context clears trail targets; destroyTrailMesh retires" {
    const alloc = std.testing.allocator;
    // Same convention as the off-context destroyMesh test above: the main
    // thread is the context owner, workers are game-thread stand-ins.
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.trails.deinit(alloc);
    // Failure-safe (see the test above): retired entries stay owned by
    // `gpu_retire`; remaining trail structs go through `trails.deinit`.
    errdefer {
        for (scene.meshes.items) |m| {
            m.deinit(alloc);
            alloc.destroy(m);
        }
    }

    // Follow target + follower trail (manual structs, no GPU handles, so
    // neither the retire nor the flush below touches sg.*).
    const target = try alloc.create(Mesh);
    target.* = @import("../testing.zig").testMesh("offctx_target");
    try scene.meshes.append(alloc, target);
    const follower_mesh = try alloc.create(Mesh);
    follower_mesh.* = @import("../testing.zig").testMesh("offctx_follower_mesh");
    try scene.meshes.append(alloc, follower_mesh);
    const follower = try alloc.create(TrailMesh);
    follower.* = .{
        .allocator = alloc,
        .scene = &scene,
        .mesh = follower_mesh,
        .options = .{},
        .vertices = try alloc.alloc(Vertex, 4),
        .indices = try alloc.alloc(u16, 6),
    };
    follower.target = target;
    try scene.trails.meshes.append(alloc, follower);

    // Ordinary destroyMesh from a worker: unlinks + retires the target and
    // clears the follower — no dangling `target` past the retire.
    const DestroyJob = struct {
        scene: *Scene,
        mesh: *Mesh,
        fn run(j: @This()) void {
            j.scene.destroyMesh(j.mesh);
        }
    };
    const t = try std.Thread.spawn(.{}, DestroyJob.run, .{DestroyJob{ .scene = &scene, .mesh = target }});
    t.join();
    try std.testing.expect(follower.target == null);
    try std.testing.expectEqual(@as(usize, 1), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());

    // Epoch completes on the context thread: the retired target is freed.
    scene.gpu_retire.complete(scene.gpu_retire.current());
    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());

    // destroyTrailMesh from a worker: the trail unlinks + frees inline
    // (CPU-only) while its mesh retires for the next flush.
    const TrailJob = struct {
        scene: *Scene,
        trail: *TrailMesh,
        fn run(j: @This()) void {
            j.scene.destroyTrailMesh(j.trail);
        }
    };
    const t2 = try std.Thread.spawn(.{}, TrailJob.run, .{TrailJob{ .scene = &scene, .trail = follower }});
    t2.join();
    try std.testing.expectEqual(@as(usize, 0), scene.trails.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());

    scene.gpu_retire.complete(scene.gpu_retire.current());
    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}
