//! Staged frame tests (part 3): stage-1 producer build + latch. Split from scene/tests.zig.
const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Ray = math.Ray;
const camera_mod = @import("../camera.zig");
const Camera = camera_mod.Camera;
const Mesh = @import("../mesh.zig").Mesh;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const ShaderMaterial = @import("../material.zig").ShaderMaterial;
const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const UICanvas = @import("../ui.zig").UICanvas;
const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;
const scene_mod = @import("../scene.zig");
const Scene = scene_mod.Scene;
const SceneStats = scene_mod.SceneStats;
const FrameDrawSlot = scene_mod.FrameDrawSlot;
const RenderMeshItem = scene_mod.RenderMeshItem;
const scene_lights = @import("light_rig.zig");
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const postprocess = @import("../postprocess.zig");
const tu = @import("tests_util.zig");
const buildForTest = tu.buildForTest;
const finishForTest = tu.finishForTest;
const stageAndPrepareForTest = tu.stageAndPrepareForTest;
const p7CpuShadowPass = tu.p7CpuShadowPass;
const p7ShadowTotal = tu.p7ShadowTotal;
const p7FindByMeshIndex = tu.p7FindByMeshIndex;
const stage1FillInstances = tu.stage1FillInstances;
const stage1Scene = tu.stage1Scene;
const wave32PushTestSystem = tu.wave32PushTestSystem;
const wave32FreeTestSystems = tu.wave32FreeTestSystems;

test "stage1: worker build + main latch publishes previews; post-build mutation invisible" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 4;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Victim: heap mesh (destroyMesh frees it via the retire queue) with
    // heap instances (Mesh.deinit destroys them at flush).
    const victim = try alloc.create(Mesh);
    victim.* = @import("../testing.zig").testMesh("s1_victim");
    victim.local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));
    for (0..2) |i| {
        const inst = try alloc.create(InstancedMesh);
        inst.* = .{ .name = "v", .source_mesh = victim, .position = Vec3.new(@floatFromInt(i), 0, 0) };
        try victim.instances.append(alloc, inst);
    }
    try scene.meshes.append(alloc, victim);

    // Mark the context on main: the worker below is off-context, so its
    // destroyMesh must take the retire-queue path (never sg.* off-thread).
    gpu_thread.markContextThread();

    // Worker = the game side: build, then destroy the victim inside the
    // build→latch window (sequential in the worker — no data race; the
    // spawn/join edges carry the happens-before). Main only joins.
    const Builder = struct {
        scene: *Scene,
        victim: *Mesh,
        fn run(self: @This()) void {
            std.debug.assert(self.scene.buildPreparedFrame());
            self.scene.destroyMesh(self.victim);
        }
    };
    const t = try std.Thread.spawn(.{}, Builder.run, .{Builder{ .scene = &scene, .victim = victim }});
    t.join();

    // Off-context destroy unlinked the victim into the retire queue; the
    // build recorded previews for both meshes (victim included).
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 1), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    try std.testing.expectEqual(@as(u64, 1), parent.instance_preview.build_seq);
    try std.testing.expectEqual(@as(u32, 5), parent.instance_preview.count);
    try std.testing.expectEqual(@as(u64, 1), victim.instance_preview.build_seq);

    // Live mutation AFTER the build: the published record mirror must
    // reflect the build, not the mutation — and the latch must not have
    // touched the live mesh at all (write-back is a game-side commit now).
    mem[0].position = Vec3.new(100, 0, 0);
    const preview_bounds = parent.instance_preview.bounds;
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), parent.instance_render.staged_frame);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    // Consume the worker's pending FULL build without rebuilding:
    // post-build mutation must not leak into the latched frame.
    finishForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
    // Latch published the mirrors; live meshes still untouched.
    const latched = scene.preparedDraws().staged_instances.items;
    try std.testing.expectEqual(@as(usize, 2), latched.len);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), parent.instance_render.staged_frame);
    // Stage-2B: queues froze at build time (parent + victim = 2 batches);
    // the latch stages whatever the slot owns (no live-list guard remains),
    // so both batches finalize from their records — including the destroyed
    // victim's staged data. The game-side commit below skips the victim by
    // identity instead (no UAF), and the surviving parent lands live.
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws.primary.opaque_instanced.items.len);
    var found_parent: usize = 0;
    var found_victim: usize = 0;
    for (draws.primary.opaque_instanced.items) |b| {
        if (b.source_uid == parent.uid) {
            try std.testing.expectEqual(@as(u32, 5), b.visible_instance_count);
            found_parent += 1;
        } else {
            try std.testing.expectEqual(@as(u32, 3), b.visible_instance_count);
            found_victim += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 1), found_parent);
    try std.testing.expectEqual(@as(usize, 1), found_victim);

    // Commit (next game-side build) applies the published mirrors to live:
    // parent lands, the unlinked victim is skipped by the identity guard.
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 5), parent.instance_render.count);
    try std.testing.expectEqual(preview_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
    // The CPU preview carries the matrix-bytes hash for the GPU dedup gate;
    // headless (no sg context) the GPU half is skipped, so instance_render
    // keeps the stale hash (0) by design — bounds/count still publish.
    try std.testing.expect(parent.instance_preview.hash != 0);
    try std.testing.expectEqual(@as(u64, 0), parent.instance_render.hash);
    // A 100-unit move would have shifted the bounds; the commit kept build
    // time (max.x well under the mutated span).
    try std.testing.expect(parent.instance_render.bounds.max.x < 50.0);
}

test "stage1: two builds before latch, newest wins" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 3;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    try buildForTest(&scene);
    const first_bounds = parent.instance_preview.bounds;
    // Mutate, rebuild: the single preview store is recomputed in place.
    mem[2].position = Vec3.new(40, 0, 0);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u64, 2), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 2), parent.instance_preview.build_seq);
    try std.testing.expect(parent.instance_preview.bounds.max.x > first_bounds.max.x + 10.0);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(u64, 2), scene.last_latched_seq.load(.monotonic));
    // Commit (next game-side build) applies the newest publish to live.
    try buildForTest(&scene);
    try std.testing.expectEqual(parent.instance_preview.bounds, parent.instance_render.bounds);
    try std.testing.expect(parent.instance_render.bounds.max.x > first_bounds.max.x + 10.0);
}

test "stage1: serial producer builds latch identical counts/bounds" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 5;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    mem[4].is_visible = false;
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Serial producer first: publish then build; the outcome lands in the
    // published record mirror.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    const mirror = scene.preparedDraws().staged_instances.items[0];
    try std.testing.expectEqual(@as(u32, 5), mirror.count);
    const latched_bounds = mirror.bounds;
    const latched_hash = mirror.uploaded_hash;

    // Same live state, serial second build: the staged protocol latches
    // identical counts/bounds/hash (only staged_frame advances).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(u64, 2), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(mirror.count, parent.instance_render.count);
    try std.testing.expectEqual(latched_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(latched_hash, parent.instance_render.hash);
}

test "stage1: second staged build republishes without stale mirrors" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 3;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Serial build (frame 1): outcome sits in the slot mirror.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);

    // Serial build (frame 2): the staged build republishes the same live
    // state — frame-1 mirrors must not leak back in later. The frame-2
    // latch outcomes commit at the NEXT build, so live still shows the
    // frame-1 commit here (staged_frame 1, never a stale mirror).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 1), parent.instance_render.staged_frame);

    // Next build commits newest-wins: live keeps the frame-2 state, never
    // regresses to the stale frame-1 mirror.
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 2), parent.instance_render.staged_frame);

    // A fresh latch + commit republishes identically.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 3), parent.instance_render.staged_frame);
}

test "stage1: commit resolves the latest latched front through the lease" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 3;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Generation 1: build + latch (count 3). Live untouched by the latch.
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    const front_f1 = scene.draws.frontIndex();
    try std.testing.expectEqual(@as(u64, 1), scene.draws.slots[front_f1].frame_id);

    // Generation 2: hide one instance, build (commits the F1 mirror), latch.
    mem[0].is_visible = false;
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 1), parent.instance_render.staged_frame);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    const front_f2 = scene.draws.frontIndex();
    try std.testing.expectEqual(@as(u64, 2), scene.draws.slots[front_f2].frame_id);

    // Generation 3: hide another, build — the commit must apply the LATEST
    // latched front (F2, count 2), never a superseded slot's mirror.
    mem[1].is_visible = false;
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 2), parent.instance_render.staged_frame);
    // The lease is balanced: the commit takes no pins and the front the
    // commit resolved is the latest published slot.
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pins_held);
    try std.testing.expectEqual(front_f2, scene.draws.frontIndex());
}

test "stage1: serial same-thread build+latch parity" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 6;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 10);
    mem[1].is_visible = false;
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    try buildForTest(&scene);
    try std.testing.expect(scene.build_slot.load(.monotonic) < scene.draws.slots.len);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);

    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    // Commit (next game-side build) applies the publish to live.
    try buildForTest(&scene);
    try std.testing.expectEqual(parent.instance_preview.count, parent.instance_render.count);
    try std.testing.expectEqual(@as(u32, 6), parent.instance_render.count);
    try std.testing.expectEqual(parent.instance_preview.bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
    try std.testing.expectEqual(scene.frame_id, scene.preparedDraws().frame_id);
    try std.testing.expectEqual(scene.retire_epoch, scene.preparedDraws().retire_epoch);
}

test "stage1: OOM build advances nothing, latch keeps previous, then recovers" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 4;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Prime: funded build + latch publishes the complete state; the next
    // build commits it to the live mesh.
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try buildForTest(&scene);
    const primed_bounds = parent.instance_render.bounds;
    const primed_hash = parent.instance_render.hash;
    try std.testing.expectEqual(@as(u32, 5), parent.instance_render.count);

    // Mutate live, then build unfunded: drop ALL scratch capacity so the
    // segment really allocates, and refuse the first alloc. The preview
    // must not advance (scene seq still does — the latch will skip it).
    mem[0].position = Vec3.new(100, 0, 0);
    for (&scene.draws.slots) |*slot| slot.primary.instance_matrices.clearAndFree(alloc);
    const real_alloc = scene.allocator;
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    scene.allocator = failing.allocator();
    try buildForTest(&scene);
    scene.allocator = real_alloc;
    try std.testing.expectEqual(@as(u64, 3), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 2), parent.instance_preview.build_seq);
    try std.testing.expect(failing.has_induced_failure);

    // Latch: the stale mesh is skipped — previous complete state stands,
    // no partial publish.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(u64, 3), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 5), parent.instance_render.count);
    try std.testing.expectEqual(primed_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(primed_hash, parent.instance_render.hash);

    // Recovery: a funded build + latch + commit publishes the mutated state.
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u64, 5), parent.instance_preview.build_seq);
    try std.testing.expect(parent.instance_render.bounds.max.x > primed_bounds.max.x + 10.0);
}

test "stage1: particle build on worker + latch on main freezes the frame" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.particles.systems.deinit(alloc);
    defer scene.particles.frame.deinit(alloc);
    defer scene.particles.build_frame.deinit(alloc);

    // CPU-only system with fake borrowed handle ids (never deinited via
    // ps.deinit — that would issue sg.destroy* on the fake ids; members are
    // freed manually below).
    const parts = try alloc.alloc(particles.Particle, 4);
    defer alloc.free(parts);
    const insts = try alloc.alloc(particles.ParticleInstanceData, 4);
    defer alloc.free(insts);
    const scratch = try alloc.alloc(u8, 4);
    defer alloc.free(scratch);
    var ps = ParticleSystem{
        .name = "s1",
        .allocator = alloc,
        .particles = parts,
        .instances = insts,
        .alive_scratch = scratch,
        .capacity = 4,
        .instance_buffer = .{ .id = 11 },
        .prng = std.Random.DefaultPrng.init(42),
    };
    ps.active_count = 3;
    try scene.particles.systems.append(alloc, &ps);

    // Worker = the game side: build only. Main only joins, then prepares.
    const Builder = struct {
        scene: *Scene,
        fn run(self: @This()) void {
            std.debug.assert(self.scene.buildPreparedFrame());
        }
    };
    const t = try std.Thread.spawn(.{}, Builder.run, .{Builder{ .scene = &scene }});
    t.join();
    try std.testing.expectEqual(@as(u64, 1), scene.particles.build_seq.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), scene.particles.build_frame.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.particles.frame.items.len);

    // Live mutation after the worker build must not reach the render-owned
    // frame: consume the pending build without rebuilding.
    ps.active_count = 1;
    ps.instance_buffer = .{ .id = 99 };
    finishForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.particles.latched_seq);
    try std.testing.expectEqual(@as(usize, 1), scene.particles.frame.items.len);
    try std.testing.expectEqual(@as(usize, 3), scene.particles.frame.items[0].active_count);
    try std.testing.expectEqual(@as(u32, 11), scene.particles.frame.items[0].instance_buffer.id);
    // renderPrepared keeps reading `frame` only (headless no-op here).
    try std.testing.expectEqual(@as(usize, 1), scene.particles.build_frame.items.len);
    ps.active_count = 4;
    try std.testing.expectEqual(@as(usize, 3), scene.particles.frame.items[0].active_count);
}

test "stage1: compute particle capture borrows the baked buffer; retire takes all five" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.particles.systems.deinit(alloc);
    defer scene.particles.frame.deinit(alloc);
    defer scene.particles.build_frame.deinit(alloc);

    // Compute-mode system with fake borrowed handle ids (never deinited via
    // ps.deinit — that would issue sg.destroy* on the fake ids; members are
    // freed manually below, same idiom as the cpu test above).
    const parts = try alloc.alloc(particles.Particle, 4);
    defer alloc.free(parts);
    const insts = try alloc.alloc(particles.ParticleInstanceData, 4);
    defer alloc.free(insts);
    const scratch = try alloc.alloc(u8, 4);
    defer alloc.free(scratch);
    const staging = try alloc.alloc(particles.GpuParticleSlot, 4);
    defer alloc.free(staging);
    var ps = ParticleSystem{
        .name = "s1_compute",
        .allocator = alloc,
        .particles = parts,
        .instances = insts,
        .alive_scratch = scratch,
        .capacity = 4,
        .instance_buffer = .{ .id = 11 },
        .gpu_slot_buffer = .{ .id = 12 },
        .compute_state_buffer = .{ .id = 21 },
        .compute_spawn_buffer = .{ .id = 22 },
        .compute_draw_buffer = .{ .id = 23 },
        .compute_staging = staging,
        .prng = std.Random.DefaultPrng.init(42),
    };
    ps.simulation_mode = .compute;
    ps.active_count = 3;
    ps.compute_high_water = 3;
    try scene.particles.systems.append(alloc, &ps);

    // Capture borrows the baked draw buffer (not the instance/slot ones) and
    // snapshots the compute mode.
    scene.particles.captureFrame(alloc);
    try std.testing.expectEqual(@as(usize, 1), scene.particles.frame.items.len);
    const draw = scene.particles.frame.items[0];
    try std.testing.expectEqual(particles.SimulationMode.compute, draw.simulation_mode);
    try std.testing.expectEqual(@as(u32, 23), draw.compute_draw_buffer.id);
    try std.testing.expectEqual(@as(u32, 23), draw.drawBuffer().id);

    // Teardown: all five particle buffers retire through the queue from any
    // thread (fake ids: retire into the open epoch like the staged finish would,
    // so the pre-complete flush keeps them without sg.*, then manual cleanup
    // — mirrors the retireBuffer unit-test handling).
    _ = scene.gpu_retire.begin();
    var out = [_]sokol.gfx.Buffer{.{}} ** 8;
    const n = ps.takeGpuBuffersForRetire(&out);
    try std.testing.expectEqual(@as(usize, 5), n);
    for (out[0..n]) |buf| scene.gpu_retire.retireBuffer(alloc, buf);
    try std.testing.expectEqual(@as(usize, 5), scene.gpu_retire.retainedCount());
    try std.testing.expectEqual(@as(u64, 0), scene.gpu_retire.duplicateDropCount());
    scene.gpu_retire.flush(alloc);
    try std.testing.expectEqual(@as(usize, 5), scene.gpu_retire.retainedCount());
    scene.gpu_retire.pending.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
    // Handles zeroed: the layer deinit path (ps.deinit) would now skip every
    // buffer destroy; a second take finds nothing (no double-retire).
    try std.testing.expectEqual(@as(usize, 0), ps.takeGpuBuffersForRetire(&out));
    try std.testing.expectEqual(@as(u32, 0), ps.compute_draw_buffer.id);
}

test "stage1: physics build on worker + latch on main freezes the capture" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.physics.deinit(alloc);

    _ = scene.physics.enable(alloc, null);
    var pmesh = @import("../testing.zig").testMesh("s1_phys");
    _ = try scene.physics.getWorld().?.createBody(&pmesh, .box, 0.0);
    scene.physics.show_debug = true;

    const Builder = struct {
        scene: *Scene,
        fn run(self: @This()) void {
            std.debug.assert(self.scene.buildPreparedFrame());
        }
    };
    const t = try std.Thread.spawn(.{}, Builder.run, .{Builder{ .scene = &scene }});
    t.join();
    try std.testing.expectEqual(@as(u64, 1), scene.physics.build_seq.load(.acquire));
    try std.testing.expect(scene.physics.build_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.build_lines.items.len);
    try std.testing.expect(!scene.physics.prepared_visible);

    // Live mutation after the worker build must not reach the prepared
    // capture: consume the pending build without rebuilding.
    const x0 = scene.physics.build_lines.items[0].a.x;
    pmesh.position = Vec3.new(5, 0, 0);
    finishForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.physics.latched_seq);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
    try std.testing.expectApproxEqAbs(x0, scene.physics.prepared_lines.items[0].a.x, 1e-4);
    scene.physics.step(0.016);
    try std.testing.expectApproxEqAbs(x0, scene.physics.prepared_lines.items[0].a.x, 1e-4);
}

test "stage1: instances cleared between build and latch take the regular path" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 3;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Build while instanced: the preview records 3 (the build never
    // publishes — instance_render stays empty until the staged latch).
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_preview.count);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);

    // Drop all instances before the latch: stage-2B freezes queues at build
    // time, and the latch stages whatever the slot owns (no live emptiness
    // re-check remains) — so this frame still draws the staged 3. The
    // live mesh is untouched by the latch (still empty), and the game-side
    // commit skips the emptied mesh, so the previous complete live state
    // stands. The regular-path switch takes effect on the next build+latch,
    // not this one.
    parent.instances.clearRetainingCapacity();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 4), draws.primary.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(usize, 0), draws.primary.items.items.len);

    // Next build: the commit skips the emptied mesh (live stays empty) and
    // the rebuild reroutes to the regular path; the following latch draws
    // the regular item with no instanced payload.
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    const draws2 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws2.primary.items.items.len);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
}

test "stage1: latch consumes slot records when live previews are cleared" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 3;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Build freezes the slot-owned records; wiping the live previews after
    // that must not matter — the latch reads records + scratch only.
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_preview.count);
    try std.testing.expectEqual(@as(usize, 1), scene.draws.slotAt(scene.build_slot.load(.monotonic)).staged_instances.items.len);
    parent.instance_preview = .{};
    try std.testing.expectEqual(@as(u64, 0), parent.instance_preview.build_seq);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
    // Post-latch state mirrors into the slot record (the patch source) even
    // though the live previews were wiped; live meshes stay untouched until
    // the game-side commit.
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), parent.instance_render.staged_frame);
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 4), draws.primary.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(usize, 1), draws.staged_instances.items.len);
    try std.testing.expectEqual(scene.frame_id, draws.staged_instances.items[0].staged_frame);
    try std.testing.expectEqual(@as(u32, 4), draws.staged_instances.items[0].count);

    // Commit (next game-side build) applies the published mirror to live.
    // (The rebuild recomputes the wiped preview from unchanged live TRS.)
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
    try std.testing.expect(parent.instance_render.bounds.isValid());
    try std.testing.expectEqual(draws.staged_instances.items[0].buffer.id, parent.instance_render.buffer.id);
}

test "stage1: failed latch keeps previous complete state, patch fail-closes" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const mem = try alloc.alloc(InstancedMesh, 4);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, 4);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        // Two live now; grown to four for the failing round below.
        .instances = .{ .items = ptrs[0..2], .capacity = 4 },
    };
    try scene.meshes.append(alloc, &parent);
    try scene.outline_meshes.append(alloc, &parent);

    // Round 1: funded build + latch publishes the complete state; the next
    // build commits it to the live mesh.
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expect(parent.instance_render.bounds.isValid());

    // Distinctive prior values in every preserved field (headless buffers
    // are id 0, so a fake id proves the buffer slot is untouched too).
    parent.instance_render.buffer = .{ .id = 100 };
    parent.instance_render.capacity = 7;
    parent.instance_render.hash = 0xABCD;
    parent.instance_render.uploaded_count = 5;
    const prior = parent.instance_render;

    // Round 2: grow to 4 instances (a live GPU would need growth here),
    // build, then truncate the back-slot scratch before the latch so the
    // record's slice is out of range — the latch fail-closes through the
    // exact same path a GPU-half failure takes (failRecord + continue, live
    // untouched — ST2-C covers the makeBuffer-failure detection itself on a
    // real GPU).
    parent.instances.items = ptrs[0..4];
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 5), parent.instance_preview.count);
    scene.draws.slotAt(scene.build_slot.load(.monotonic)).primary.instance_matrices.clearRetainingCapacity();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);

    // Previous COMPLETE state kept in every field; staged_frame NOT advanced
    // to the current frame (residual readers can tell nothing new published).
    try std.testing.expectEqual(prior.buffer.id, parent.instance_render.buffer.id);
    try std.testing.expectEqual(prior.capacity, parent.instance_render.capacity);
    try std.testing.expectEqual(prior.count, parent.instance_render.count);
    try std.testing.expectEqual(prior.bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(prior.hash, parent.instance_render.hash);
    try std.testing.expectEqual(prior.uploaded_count, parent.instance_render.uploaded_count);
    try std.testing.expectEqual(prior.staged_frame, parent.instance_render.staged_frame);
    try std.testing.expect(parent.instance_render.staged_frame != scene.frame_id);

    // The record carries the not-published state.
    const recs = scene.preparedDraws().staged_instances.items;
    try std.testing.expectEqual(@as(usize, 1), recs.len);
    try std.testing.expectEqual(std.math.maxInt(u64), recs[0].staged_frame);
    try std.testing.expectEqual(@as(u32, 0), recs[0].buffer.id);
    try std.testing.expectEqual(@as(u32, 0), recs[0].count);

    // Payload fail-closed: the frozen batch/shadow/outline entries for the
    // mesh go invisible (count 0, zeroed handles).
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 0), draws.primary.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(u32, 0), draws.primary.opaque_instanced.items[0].instance_buffer.id);
    var shadow_found = false;
    for (draws.shadow.items.items) |it| {
        if (it.is_instanced and it.source_uid == parent.uid) {
            try std.testing.expectEqual(@as(u32, 0), it.visible_instance_count);
            try std.testing.expectEqual(@as(u32, 0), it.instance_buffer.id);
            try std.testing.expect(!it.world_aabb.isValid());
            try std.testing.expectEqual(@as(f32, 0), it.max_dim);
            shadow_found = true;
        }
    }
    try std.testing.expect(shadow_found);
    var outline_found = false;
    for (draws.outline_items.items) |it| {
        if (it.is_instanced and it.source_uid == parent.uid) {
            try std.testing.expectEqual(@as(u32, 0), it.visible_instance_count);
            try std.testing.expectEqual(@as(u32, 0), it.instance_buffer.id);
            outline_found = true;
        }
    }
    try std.testing.expect(outline_found);
    // No retirement happened.
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());

    // Recovery: restore the scratch with a funded rebuild, latch, and commit
    // publishes the grown state.
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 5), parent.instance_render.count);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 5), scene.preparedDraws().primary.opaque_instanced.items[0].visible_instance_count);
}

test "stage1: mesh reorder + post-build add stages through, commit skips, recovers next" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const mem_a = try alloc.alloc(InstancedMesh, 3);
    defer alloc.free(mem_a);
    const ptrs_a = try alloc.alloc(*InstancedMesh, 3);
    defer alloc.free(ptrs_a);
    stage1FillInstances(&src, mem_a, ptrs_a, 0);
    var mesh_a = Mesh{
        .name = "s1_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_a, .capacity = 3 },
    };
    const mem_b = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mem_b);
    const ptrs_b = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(ptrs_b);
    stage1FillInstances(&src, mem_b, ptrs_b, 50);
    var mesh_b = Mesh{
        .name = "s1_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_b, .capacity = 2 },
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_b);

    // Scratch is concatenated [A0 A1 A2 B0 B1]: A.lo=0, B.lo=3.
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(usize, 0), mesh_a.instance_preview.scratch_lo);
    try std.testing.expectEqual(@as(usize, 4), mesh_b.instance_preview.scratch_lo);

    // Reorder (swapRemove unlinks A) and add a fresh mesh C whose preview
    // predates every build (build_seq 0).
    _ = scene.meshes.swapRemove(0);
    const mem_c = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mem_c);
    const ptrs_c = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(ptrs_c);
    stage1FillInstances(&src, mem_c, ptrs_c, 200);
    var mesh_c = Mesh{
        .name = "s1_c",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_c, .capacity = 2 },
    };
    try scene.meshes.append(alloc, &mesh_c);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    // Slot-owned latch: the reorder is invisible here (no live-list read
    // remains) — both records publish from slot data and the patch finalizes
    // the frozen batches from them. Live meshes stay exactly as built
    // (never-staged): the write-back is a game-side commit now.
    try std.testing.expectEqual(@as(u32, 0), mesh_b.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_b.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 0), mesh_a.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_a.instance_render.staged_frame);
    const latched_records = scene.preparedDraws().staged_instances.items;
    try std.testing.expectEqual(@as(usize, 2), latched_records.len);
    for (latched_records) |rec| {
        try std.testing.expectEqual(scene.frame_id, rec.staged_frame);
    }
    // C (never built) is skipped: no publish, no queue batch.
    try std.testing.expectEqual(@as(u64, 0), mesh_c.instance_preview.build_seq);
    try std.testing.expectEqual(@as(u32, 0), mesh_c.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_c.instance_render.staged_frame);
    // Stage-2B: the mesh list MUST NOT be mutated between build and latch
    // (swapRemove + append here); queues froze at build (A+B) and both
    // entries finalize from their records — no OOB, no UAF. The NEXT build's
    // commit runs the identity guard instead: both displaced records skip,
    // so neither live mesh publishes the stale generation.
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws.primary.opaque_instanced.items.len);
    for (draws.primary.opaque_instanced.items) |b| {
        try std.testing.expect(b.visible_instance_count == 4 or b.visible_instance_count == 3);
    }
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 0), mesh_a.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_a.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 0), mesh_b.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_b.instance_render.staged_frame);

    // Recovery: a funded build + latch + commit publishes the live list
    // (B + C).
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 3), mesh_b.instance_render.count);
    try std.testing.expectEqual(@as(u32, 3), mesh_c.instance_render.count);
    try std.testing.expectEqual(scene.frame_id, mesh_b.instance_render.staged_frame);
    const draws_r = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws_r.primary.opaque_instanced.items.len);
    for (draws_r.primary.opaque_instanced.items) |b| {
        try std.testing.expectEqual(@as(u32, 3), b.visible_instance_count);
    }
}

test "stage1: truncated scratch between build and latch is skipped safely" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 2;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Prime: funded build + latch publishes the complete state; the next
    // build commits it to the live mesh.
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try buildForTest(&scene);
    const primed_bounds = parent.instance_render.bounds;
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);

    // Mutate, rebuild, then truncate the scratch before the latch (a
    // contract violation the latch must survive): the out-of-range slice is
    // skipped via the bounds check and the previous state stands.
    mem[0].position = Vec3.new(100, 0, 0);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u64, 3), parent.instance_preview.build_seq);
    scene.draws.slotAt(scene.build_slot.load(.monotonic)).primary.instance_matrices.clearRetainingCapacity();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(u64, 3), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(primed_bounds, parent.instance_render.bounds);

    // Recovery: a funded build + latch + commit publishes the mutated state.
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try buildForTest(&scene);
    try std.testing.expect(parent.instance_render.bounds.max.x > primed_bounds.max.x + 10.0);
}
