//! Shared helpers for the scene/ staged-protocol test files (split from the
//! former monolithic scene/tests.zig). Test-only, NOT a production compat
//! wrapper. No `test` blocks live here; the split files are:
//! tests_frame.zig, tests_slots.zig, tests_stage1.zig, tests_stage2.zig,
//! tests_reuse.zig, tests_waves.zig.
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

// Staged-protocol test helpers (test-only, NOT a production compat wrapper):
// - buildForTest: full producer build (claim.build -> stageUi -> publish),
//   requires success (saturation is a test failure).
// - finishForTest: consumes the pending FULL build via begin/finish; panics
//   when no fresh staged build is pending (hides-nothing: a missing build
//   fails loudly instead of falling back).
// - stageAndPrepareForTest: fresh-build sites only (no pending build):
//   explicit build + begin/finish producer step.
pub fn buildForTest(scene: *Scene) !void {
    try std.testing.expect(scene.buildPreparedFrame());
}
pub fn finishForTest(scene: *Scene) void {
    const claim = scene.beginStagedPrepare() orelse std.debug.panic("{s}", .{"finishForTest: no fresh staged build"});
    scene.finishStagedPrepare(claim);
}
pub fn stageAndPrepareForTest(scene: *Scene) !void {
    try buildForTest(scene);
    finishForTest(scene);
}

// Headless full-path integration runs the staged build+finish with a faked GPU-init
// flag (default_white_texture.view.id != 0) plus a CPU-only shadow pass
// (allocator + empty scratch/payload, zero GPU handles). No sg.* fires on
// the prepare path for plain/skinned/hook meshes: instance staging skips
// non-instanced meshes (and guards the rest with sg.isvalid), shadow
// prepareInto and the view-queue builds are pure CPU snapshots, and UI
// capture is sg-guarded (P6). Render never runs headless.
pub fn p7CpuShadowPass(scene: *Scene, alloc: std.mem.Allocator) void {
    // CPU-only stand-in: prepareInto/binMeshes never touch GPU handles
    // headless (render never runs), so every handle field is safely zero and
    // only allocator + scratch/payload carry state. Full literal — no
    // undefined fields left unread.
    scene.shadows.pass = .{
        .allocator = alloc,
        .image = .{},
        .attachment_view = .{},
        .texture_view = .{},
        .sampler = .{},
        .depth_sampler = .{},
        .spot_image = .{},
        .spot_attachment_view = .{},
        .spot_texture_view = .{},
        .spot_needs_clear = false,
        .point_image = .{},
        .point_attachment_view = .{},
        .point_texture_view = .{},
        .point_needs_clear = false,
        .pipeline_u16 = .{},
        .pipeline_u32 = .{},
        .inst_pipeline_u16 = .{},
        .inst_pipeline_u32 = .{},
        .skinned_pipeline_u16 = .{},
        .skinned_pipeline_u32 = .{},
        .shadow_shader = .{},
        .inst_shader = .{},
        .skinned_shader = .{},
        .binned_meshes = .empty,
        .prepared = .{},
    };
}

pub fn p7ShadowTotal(draws: *const FrameDrawSlot) usize {
    var total: usize = 0;
    for (draws.shadow.bin.counts) |c| total += c;
    return total;
}

pub fn p7FindByMeshIndex(items: []const RenderMeshItem, idx: u32) ?RenderMeshItem {
    for (items) |it| {
        if (it.mesh_index == idx) return it;
    }
    return null;
}

// Headless full-path pattern (mirrors the P7 tests): faked GPU-init flag
// (default_white_texture.view.id != 0) + CPU-only shadow pass + disabled
// culling. sg.isvalid() stays false, so the GPU halves publish bounds/count
// without touching sg.*. Render never runs headless past no-camera.

pub fn stage1FillInstances(src: *Mesh, mem: []@import("../mesh.zig").InstancedMesh, ptrs: []*@import("../mesh.zig").InstancedMesh, x0: f32) void {
    for (mem, 0..) |*inst, i| {
        const fi: f32 = @floatFromInt(i);
        inst.* = .{ .name = "s1", .source_mesh = src, .position = Vec3.new(x0 + fi * 2.0, 0, 0) };
        ptrs[i] = inst;
    }
}

pub fn stage1Scene(alloc: std.mem.Allocator) Scene {
    var scene = @import("../testing.zig").testScene(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.default_white_texture.view.id = 1;
    return scene;
}

// Stage-1 instanced fixture builders (dumb construction, no assertions):
// nearly every stage-1/stage-2B/wave test builds the same camera-bearing
// headless scene, the same unit-box source mesh, and the same N-instance
// parent, so the ~30-line fixture collapses to ~10 lines with the live
// `mem`/`ptrs` names kept (bodies stay bit-identical). Per-test quirks —
// sliced items, custom capacities, blend materials, multi-camera rigs —
// stay inline in the tests.
pub fn stage1Deinit(scene: *Scene, alloc: std.mem.Allocator) void {
    scene.shadows.pass.prepared.deinit(alloc);
    scene.shadows.pass.binned_source.deinit(alloc);
    scene.shadows.pass.binned_meshes.deinit(alloc);
    scene.gpu_retire.deinit(alloc);
    scene.draws.deinit(alloc);
    scene.outline_meshes.deinit(alloc);
    scene.meshes.deinit(alloc);
    scene.cameras.deinit(alloc);
    scene.lights.deinit(alloc);
}

pub fn stage1Prepare(scene: *Scene, alloc: std.mem.Allocator) !void {
    p7CpuShadowPass(scene, alloc);
    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
}

pub fn stage1SrcMesh(name: []const u8) Mesh {
    return .{
        .name = name,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
}

pub fn stage1ParentMesh(name: []const u8, ptrs: []*InstancedMesh, n: usize) Mesh {
    return .{
        .name = name,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
}

pub fn stage1RegularMesh(name: []const u8, position: Vec3) Mesh {
    return .{
        .name = name,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = position,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
}

const InstancedMesh = @import("../mesh.zig").InstancedMesh;

pub const Stage1Instances = struct {
    mem: []InstancedMesh,
    ptrs: []*InstancedMesh,
    pub fn deinit(self: Stage1Instances, alloc: std.mem.Allocator) void {
        alloc.free(self.ptrs);
        alloc.free(self.mem);
    }
};

pub fn stage1Instances(alloc: std.mem.Allocator, src: *Mesh, n: usize, x0: f32) !Stage1Instances {
    const mem = try alloc.alloc(InstancedMesh, n);
    errdefer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    errdefer alloc.free(ptrs);
    stage1FillInstances(src, mem, ptrs, x0);
    return .{ .mem = mem, .ptrs = ptrs };
}

// Minimal testScene teardown (lights/cameras/draws/gpu_retire): wave
// fixtures with extra lists (profiler, ui_frame, meshes, shadows) keep
// those defers inline.
pub fn basicSceneDeinit(scene: *Scene, alloc: std.mem.Allocator) void {
    scene.gpu_retire.deinit(alloc);
    scene.draws.deinit(alloc);
    scene.cameras.deinit(alloc);
    scene.lights.deinit(alloc);
}

// Wave-32 freeze-fixture body: standalone physics body mesh whose debug
// capture the staged build freezes into the slot. The caller owns the
// mesh (`defer alloc.destroy(m)`) and the physics deinit.
pub fn wave32BodyMesh(scene: *Scene, alloc: std.mem.Allocator) !*Mesh {
    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("box");
    _ = try scene.createRigidBody(m, .box, 1.0);
    scene.physics.show_debug = true;
    return m;
}

// CPU-only particle system for scene-level freeze/latch tests: built with
// the shared headless helper (no sg.* anywhere — the init defers buffer
// creation off-context, and here construction itself is plain CPU allocs),
// appended to the layer list by pointer. Teardown is manual (free + list
// deinit): `ParticleLayer.deinit` would run `ps.deinit()` (sg destroys for
// the canary ids below), so scene tests that stage canary handle ids must
// not call it.
pub fn wave32PushTestSystem(scene: *Scene, capacity: usize) !*ParticleSystem {
    const psys_mod = @import("../particles/system.zig");
    const ps = try scene.allocator.create(ParticleSystem);
    errdefer scene.allocator.destroy(ps);
    ps.* = try psys_mod.makeTestSystem(scene.allocator, capacity);
    errdefer psys_mod.freeTestSystem(ps);
    try scene.particles.systems.append(scene.allocator, ps);
    return ps;
}

pub fn wave32FreeTestSystems(scene: *Scene) void {
    const psys_mod = @import("../particles/system.zig");
    for (scene.particles.systems.items) |ps| {
        psys_mod.freeTestSystem(ps);
        scene.allocator.destroy(ps);
    }
    scene.particles.systems.deinit(scene.allocator);
    scene.particles.frame.deinit(scene.allocator);
    scene.particles.build_frame.deinit(scene.allocator);
}

// Upload-packet fake scenes (headless stage/flush/commit tests in
// upload_packets_tests.zig and upload_packets_reuse_tests.zig): dumb
// construction of the duck-typed scene surface stageUploads/flush/commit
// read (allocator + owner lists). Every list defaults to empty and every
// capacity mirrors its items length, exactly like the hand-rolled literals
// these replace. No assertions here; per-test quirks stay inline in tests.
// Two shapes because the flush/commit surface needs more fields than the
// stage surface: stage-only tests use uploadStageScene, tests driving
// flushSlotUploads/commitSlotResults/restageDroppedSlot use uploadFullScene
// (which always sets flush_in_prepare, the only value those tests used).
const TrailMesh = @import("../mesh/trail.zig").TrailMesh;
const SoftBody = @import("../softbody.zig").SoftBody;
const GreasedLineMesh = @import("../mesh/greased_line.zig").GreasedLineMesh;
const GpuRetireQueue = @import("gpu_retire.zig").GpuRetireQueue;

pub const UploadLists = struct {
    meshes: []*Mesh = &.{},
    systems: []*ParticleSystem = &.{},
    trails: []*TrailMesh = &.{},
    bodies: []*SoftBody = &.{},
    lines: []*GreasedLineMesh = &.{},
};

const UploadMeshList = struct { items: []*Mesh, capacity: usize };
const UploadSystemList = struct { items: []*ParticleSystem, capacity: usize };
const UploadTrailList = struct { items: []*TrailMesh, capacity: usize };
const UploadBodyList = struct { items: []*SoftBody, capacity: usize };
const UploadLineList = struct { items: []*GreasedLineMesh, capacity: usize };

pub const UploadStageScene = struct {
    allocator: std.mem.Allocator,
    meshes: UploadMeshList,
    particles: struct { systems: UploadSystemList },
    trails: struct { meshes: UploadTrailList },
    softbodies: struct { bodies: UploadBodyList },
    greased_lines: UploadLineList,
};

pub const UploadFullScene = struct {
    allocator: std.mem.Allocator,
    meshes: UploadMeshList,
    particles: struct { systems: UploadSystemList },
    trails: struct { meshes: UploadTrailList },
    softbodies: struct { bodies: UploadBodyList },
    greased_lines: UploadLineList,
    gpu_retire: *GpuRetireQueue,
    flush_in_prepare: bool,
};

pub fn uploadStageScene(alloc: std.mem.Allocator, lists: UploadLists) UploadStageScene {
    return .{
        .allocator = alloc,
        .meshes = .{ .items = lists.meshes, .capacity = lists.meshes.len },
        .particles = .{ .systems = .{ .items = lists.systems, .capacity = lists.systems.len } },
        .trails = .{ .meshes = .{ .items = lists.trails, .capacity = lists.trails.len } },
        .softbodies = .{ .bodies = .{ .items = lists.bodies, .capacity = lists.bodies.len } },
        .greased_lines = .{ .items = lists.lines, .capacity = lists.lines.len },
    };
}

pub fn uploadFullScene(alloc: std.mem.Allocator, lists: UploadLists, retire: *GpuRetireQueue) UploadFullScene {
    return .{
        .allocator = alloc,
        .meshes = .{ .items = lists.meshes, .capacity = lists.meshes.len },
        .particles = .{ .systems = .{ .items = lists.systems, .capacity = lists.systems.len } },
        .trails = .{ .meshes = .{ .items = lists.trails, .capacity = lists.trails.len } },
        .softbodies = .{ .bodies = .{ .items = lists.bodies, .capacity = lists.bodies.len } },
        .greased_lines = .{ .items = lists.lines, .capacity = lists.lines.len },
        .gpu_retire = retire,
        .flush_in_prepare = true,
    };
}
