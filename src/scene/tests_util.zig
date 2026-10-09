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
