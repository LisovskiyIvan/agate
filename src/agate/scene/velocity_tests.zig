//! Velocity/TAA regression gates (Track C.3).
//!
//! These tests pin ACTUAL queue/history functions and resource mapping —
//! never shader tautologies and never GPU pixels (all headless-safe,
//! CPU-only):
//! - finite-difference rigid motion matches v = omega x r (math contract
//!   the velocity shaders implement per vertex);
//! - TAA jitter determinism: a reused snapshot frame_id repeats its jitter
//!   (history stays aligned; see frame_render);
//! - queue builds never advance velocity prev state; only the
//!   presented-frame commit does (cancel/reuse/multi-view safety);
//! - skeleton presented-skin commit + unpresented zero-motion aliasing;
//! - instance count-change mapping (pair vs whole-batch zero motion);
//! - morph fallback mapping, velocity dispatch truth table, cut/reset
//!   suppression, depth-contribution routing.
//!
//! Rendered-pixel proof (pipelines, history blend, motion-blur gather)
//! needs a GPU run and stays with the coordinator gates.

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;

const Mesh = @import("../mesh.zig").Mesh;
const InstancedMesh = @import("../mesh.zig").InstancedMesh;
const instancePairMode = @import("../mesh.zig").instancePairMode;
const normalizeInstancePingPong = @import("../mesh.zig").normalizeInstancePingPong;
const InstanceRenderState = @import("../mesh.zig").InstanceRenderState;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const postprocess = @import("../postprocess.zig");
const velocity_pass = @import("../passes/velocity_pass.zig");
const render_queue = @import("render_queue.zig");
const cull = @import("render_queue/cull.zig");
const staging = @import("instance_staging.zig");
const gpu_retire = @import("gpu_retire.zig");
const morph_gpu = @import("../mesh/morph_gpu.zig");
const material_mod = @import("../material.zig");
const texture_mod = @import("../texture.zig");
const visibility = @import("../visibility/mod.zig");
const SceneStats = @import("stats.zig").SceneStats;

test "finite-difference rigid motion matches omega x r" {
    // Mesh point at radius R=2 along X, rotating about Y at omega=2 rad/s.
    const omega_y: f32 = 2.0;
    const dt: f32 = 1.0 / 60.0;
    const r = Vec3.new(2.0, 0.0, 0.0);
    const expected = Vec3.new(0.0, 0.0, -omega_y * r.x); // omega x r

    const m_prev = Mat4.rotationY(0.0);
    const m_curr = Mat4.rotationY(std.math.radiansToDegrees(omega_y * dt));
    const fd_vel = m_curr.transformPoint(r).sub(m_prev.transformPoint(r)).scale(1.0 / dt);

    // O(dt) finite-difference error (~0.07 here), not exact equality.
    try std.testing.expectApproxEqAbs(expected.x, fd_vel.x, 0.1);
    try std.testing.expectApproxEqAbs(expected.y, fd_vel.y, 1e-4);
    try std.testing.expectApproxEqAbs(expected.z, fd_vel.z, 0.1);

    // A stationary transform pair has identically zero finite difference.
    const still = m_curr.transformPoint(r).sub(m_curr.transformPoint(r));
    try std.testing.expectEqual(Vec3.zero, still);
}

test "reused snapshot frame repeats its TAA jitter" {
    const base = Mat4.perspective(std.math.degreesToRadians(60.0), 1.0, 0.1, 100.0);
    const j_a = postprocess.taaJitter(7, 1.0);
    const j_a2 = postprocess.taaJitter(7, 1.0);
    try std.testing.expectEqual(j_a, j_a2);
    const vp_a = postprocess.applyTaaJitterToViewProj(base, j_a, 800, 600);
    const vp_a2 = postprocess.applyTaaJitterToViewProj(base, j_a2, 800, 600);
    try std.testing.expectEqual(vp_a, vp_a2);
    // Periodicity is real (period 8): frame 7 and frame 15 share jitter.
    const j_wrap = postprocess.taaJitter(7 + 8, 1.0);
    try std.testing.expectEqual(j_a, j_wrap);
    // Degenerate sizes leave the matrix untouched (fail-closed, no NaN).
    const vp_degen = postprocess.applyTaaJitterToViewProj(base, j_a, 0, 600);
    try std.testing.expectEqual(base, vp_degen);
}

test "queue builds never advance velocity prev; only presented commit does" {
    const ally = std.testing.allocator;
    const unit_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));
    var mesh = Mesh{
        .name = "reuse",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.zero,
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    const meshes = [_]*Mesh{&mesh};
    var queues = render_queue.RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();

    const build = struct {
        fn run(
            a: std.mem.Allocator,
            ms: []const *Mesh,
            q: *render_queue.RenderQueues,
            s: *SceneStats,
            c: *visibility.OcclusionCuller,
            key: u64,
        ) void {
            q.reset();
            render_queue.buildFrameQueues(.{
                .allocator = a,
                .meshes = ms,
                .cache_key = key,
                .view_proj = Mat4.identity,
                .eye = Vec3.zero,
                .cull_frustum = false,
                .cull_occlusion = false,
                .occlusion_culler = c,
                .stats = s,
                .queues = q,
                .default_white_id = 1,
            });
        }
    }.run;

    // First build, never presented: zero motion (prev == model, gen none).
    build(ally, &meshes, &queues, &stats, &culler, 11);
    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), queues.items.items[0].model.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), queues.items.items[0].prev_model.m[12], 1e-6);
    try std.testing.expectEqual(std.math.maxInt(u64), queues.items.items[0].prev_frame);

    // Reuse path (same key → cache hit): byte-identical payload, prev untouched.
    const first = queues.items.items[0];
    build(ally, &meshes, &queues, &stats, &culler, 11);
    try std.testing.expectEqual(first.model, queues.items.items[0].model);
    try std.testing.expectEqual(first.prev_model, queues.items.items[0].prev_model);

    // New attempt with moved TRS but still no commit: model follows, prev
    // still reports zero motion (prev == model, never a stale intermediate).
    mesh.position = Vec3.new(5, 0, 0);
    build(ally, &meshes, &queues, &stats, &culler, 12);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), queues.items.items[0].model.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), queues.items.items[0].prev_model.m[12], 1e-6);

    // Present frame 40 carrying the x=5 payload, then move again: the
    // frozen prev carries generation 40 with it.
    cull.resetPresentedVelocity(&meshes, 40);
    cull.commitPresentedVelocityQueue(&meshes, &queues, 40);
    mesh.position = Vec3.new(8, 0, 0);
    build(ally, &meshes, &queues, &stats, &culler, 13);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), queues.items.items[0].model.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), queues.items.items[0].prev_model.m[12], 1e-6);
    try std.testing.expectEqual(@as(u64, 40), queues.items.items[0].prev_frame);
    // Draw-time gate: usable exactly when 40 was actually rendered.
    try std.testing.expect(velocity_pass.usePresentedPrev(queues.items.items[0].prev_frame, 40));
    try std.testing.expect(!velocity_pass.usePresentedPrev(queues.items.items[0].prev_frame, 39));
}

test "skeleton presented commit freezes prev skin; unpresented aliases cur" {
    const alloc = std.testing.allocator;
    const skel = try Skeleton.init(alloc, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(4, 0, 0);
    skel.update();

    // Two sim updates without a commit: prev stays at the old presentation
    // (here: never presented → aliases current → zero motion).
    skel.bones[0].local_position = Vec3.new(6, 0, 0);
    skel.update();
    try std.testing.expect(skel.getPrevSkinMatrices() == skel.getRenderSkinMatrices());

    // Commit the presented copy: prev freezes, later sim updates can't move it.
    var presented = skel.getRenderSkinMatrices().*;
    skel.commitPresentedSkin(&presented, 21);
    skel.bones[0].local_position = Vec3.new(99, 0, 0);
    skel.update();
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), skel.getPrevSkinMatrices()[0].m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 99.0), skel.getRenderSkinMatrices()[0].m[12], 1e-4);

    // resetToBindPose clears the presentation (next appearance: zero motion).
    skel.resetToBindPose();
    try std.testing.expectEqual(std.math.maxInt(u64), skel.vel_presented_frame);
}

test "staged-but-unrendered prev never pairs: latch, supersede, reuse" { // Latch-without-render: a front staged (commit seeds prev with gen B)
    // but never presented must fail the draw-time match against last
    // rendered A — conservative zero motion, never the unrendered pose.
    try std.testing.expect(!velocity_pass.usePresentedPrev(42, 41));
    // Rendered-A → prepared-B → producer-C-before-B-render: C's items
    // carry prev gen B while last rendered is still A → no pairing.
    try std.testing.expect(!velocity_pass.usePresentedPrev(43, 41));
    // Once B actually renders, its generation pairs.
    try std.testing.expect(velocity_pass.usePresentedPrev(43, 43));
    // Reuse replays front F whose frozen prev gen P predates the render:
    // replaying old object motion indefinitely is forbidden — zero motion.
    try std.testing.expect(!velocity_pass.usePresentedPrev(40, 41));
    // First frame ever: nothing staged, nothing rendered.
    try std.testing.expect(!velocity_pass.usePresentedPrev(std.math.maxInt(u64), 0));
}

test "instance pairing needs same count AND same ordered layout" {
    const layout_a: u64 = 0xA1;
    const layout_b: u64 = 0xB2;
    try std.testing.expectEqual(.pair, instancePairMode(4, 4, layout_a, layout_a));
    try std.testing.expectEqual(.pair, instancePairMode(0, 0, layout_a, layout_a));
    // Count changes: growth, shrink, visibility toggles.
    try std.testing.expectEqual(.zero, instancePairMode(4, 5, layout_a, layout_b));
    try std.testing.expectEqual(.zero, instancePairMode(5, 4, layout_a, layout_b));
    try std.testing.expectEqual(.zero, instancePairMode(0, 3, layout_a, layout_b));
    try std.testing.expectEqual(.zero, instancePairMode(3, 0, layout_a, layout_b));
    // Same count, changed order/membership: substitution and reorder.
    try std.testing.expectEqual(.zero, instancePairMode(4, 4, layout_a, layout_b));
}

test "morph displacement maps to the depth fallback, rigid does not" {
    const idle = morph_gpu.VsUniforms{
        .weights0 = .{ 0, 0, 0, 0 },
        .weights1 = .{ 0, 0, 0, 0 },
        .params = .{ 0, 1, 1, 0 },
    };
    try std.testing.expect(!cull.morphNeedsDepthFallback(idle));
    const armed_zero = morph_gpu.VsUniforms{
        .weights0 = .{ 0, 0, 0, 0 },
        .weights1 = .{ 0, 0, 0, 0 },
        .params = .{ 1, 8, 8, 0 },
    };
    try std.testing.expect(!cull.morphNeedsDepthFallback(armed_zero));
    const displaced = morph_gpu.VsUniforms{
        .weights0 = .{ 0, 0.25, 0, 0 },
        .weights1 = .{ 0, 0, 0, 0 },
        .params = .{ 1, 8, 8, 0 },
    };
    try std.testing.expect(cull.morphNeedsDepthFallback(displaced));
}

test "velocity dispatch matches the composite MSAA gate" {
    try std.testing.expect(velocity_pass.velocityNeeded(true, false, 1));
    try std.testing.expect(!velocity_pass.velocityNeeded(false, false, 1));
    try std.testing.expect(velocity_pass.velocityNeeded(false, true, 1));
    // MSAA (with or without the depth prepass): velocity suppressed, the
    // blur uses camera depth reprojection against the prepass depth.
    try std.testing.expect(!velocity_pass.velocityNeeded(false, true, 4));
    try std.testing.expect(!velocity_pass.velocityNeeded(false, true, 2));
}

test "cut suppression fires exactly on reset/cut frames" {
    try std.testing.expect(!velocity_pass.cutSuppressesVelocity(false, false, false));
    try std.testing.expect(!velocity_pass.cutSuppressesVelocity(false, true, false));
    try std.testing.expect(velocity_pass.cutSuppressesVelocity(true, false, false));
    try std.testing.expect(velocity_pass.cutSuppressesVelocity(true, true, true));
    try std.testing.expect(velocity_pass.cutSuppressesVelocity(false, true, true));
    try std.testing.expect(!velocity_pass.cutSuppressesVelocity(false, false, true));
}

test "velocity skips exactly the non-depth-contributing draws" { // Mirrors the velocity pass skip matrix (itemContributesDepth): opaque
    // rigid draws run; transparent, decal, and hook-material draws fall
    // back to depth reprojection (mask 0).
    try std.testing.expect(velocity_pass.itemContributesDepth(false, false, false));
    try std.testing.expect(!velocity_pass.itemContributesDepth(true, false, false));
    try std.testing.expect(!velocity_pass.itemContributesDepth(false, true, false));
    try std.testing.expect(!velocity_pass.itemContributesDepth(false, false, true));
}

test "velocity alpha uniforms map the frozen record slot-0 lanes exactly" {
    // Distinct ids per lane prove the helper copies the albedo lane (not
    // normal/mr/emissive) and its own sampler (not the data sampler).
    const white = texture_mod.Texture{
        .image = .{},
        .view = .{ .id = 101 },
        .sampler = .{ .id = 102 },
        .width = 1,
        .height = 1,
    };
    const normal = texture_mod.Texture{
        .image = .{},
        .view = .{ .id = 103 },
        .sampler = .{ .id = 104 },
        .width = 1,
        .height = 1,
    };
    const cube = texture_mod.CubeTexture{ .image = .{}, .view = .{}, .sampler = .{}, .size = 1 };
    const brdf = texture_mod.Texture{
        .image = .{},
        .view = .{ .id = 105 },
        .sampler = .{ .id = 106 },
        .width = 1,
        .height = 1,
    };
    const albedo = texture_mod.Texture{
        .image = .{},
        .view = .{ .id = 201 },
        .sampler = .{ .id = 202 },
        .width = 4,
        .height = 4,
    };
    const def_mat = material_mod.PBRMaterial.init("velocity_alpha_default");

    var pbr = material_mod.PBRMaterial.init("velocity_alpha_cutout");
    pbr.alpha_mode = .cutout;
    pbr.alpha_cutoff = 0.37;
    pbr.albedo_color = math.Color3.new(0.1, 0.2, 0.3);
    pbr.alpha = 0.9;
    pbr.albedo_texture = albedo;
    pbr.albedo_uv_transform = .{ .offset = .{ 0.25, 0.5 }, .rotation = 0.0, .scale = .{ 2.0, 3.0 }, .tex_coord = 1 };
    const rec = material_mod.buildDrawRecord(
        .{ .pbr = &pbr },
        &def_mat,
        &white,
        &normal,
        &cube,
        &brdf,
        null,
        1.0,
    );

    // The record itself froze the albedo lane (proves the fixture is live).
    try std.testing.expectEqual(@as(u32, 201), rec.albedo_view.id);
    try std.testing.expectEqual(@as(u32, 202), rec.albedo_sampler.id);
    try std.testing.expectApproxEqAbs(@as(f32, 0.37), rec.alpha_cutoff, 1e-7);

    // The helper maps the record 1:1 — slot-0 UV lanes, full base color,
    // cutoff verbatim. Compared against the RECORD (the draw input), with
    // one independent spot check against the material UV math.
    const alpha = velocity_pass.velocityAlphaUniforms(rec);
    try std.testing.expectEqual(rec.base_color, alpha.base_color_factor);
    try std.testing.expectEqual(rec.uv_matrices[0], alpha.uv_matrix);
    try std.testing.expectEqual(rec.uv_offsets[0], alpha.uv_offset);
    try std.testing.expectEqual(rec.alpha_cutoff, alpha.alpha_cutoff);
    try std.testing.expectEqual(pbr.albedo_uv_transform.matrixRows(), alpha.uv_matrix);
    try std.testing.expectEqual(pbr.albedo_uv_transform.offsetPacked(), alpha.uv_offset);
    // tex_coord=1 selects UV1 in-shader; the packed offset carries it.
    try std.testing.expect(alpha.uv_offset[3] > 0.5);
    // Base color alpha rides the multiply chain (v_color.a * base.a * tex).
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), alpha.base_color_factor[3], 1e-7);

    // Opaque records keep cutoff 0.0: the shader's no-discard branch.
    var opaque_mat = material_mod.PBRMaterial.init("velocity_alpha_opaque");
    opaque_mat.albedo_texture = albedo;
    const rec_opaque = material_mod.buildDrawRecord(
        .{ .pbr = &opaque_mat },
        &def_mat,
        &white,
        &normal,
        &cube,
        &brdf,
        null,
        1.0,
    );
    const alpha_opaque = velocity_pass.velocityAlphaUniforms(rec_opaque);
    try std.testing.expectEqual(@as(f32, 0.0), alpha_opaque.alpha_cutoff);
    try std.testing.expectEqual(@as(u32, 201), rec_opaque.albedo_view.id);

    // Standard family maps its diffuse lane onto the same slot-0 contract.
    var std_mat = material_mod.StandardMaterial.init("velocity_alpha_std");
    std_mat.alpha_mode = .cutout;
    std_mat.alpha_cutoff = 0.42;
    std_mat.diffuse_texture = albedo;
    std_mat.diffuse_uv_transform = .{ .offset = .{ 0.125, 0.75 }, .rotation = 0.0, .scale = .{ 1.0, 1.0 }, .tex_coord = 0 };
    const rec_std = material_mod.buildDrawRecord(
        .{ .standard = &std_mat },
        &def_mat,
        &white,
        &normal,
        &cube,
        &brdf,
        null,
        1.0,
    );
    const alpha_std = velocity_pass.velocityAlphaUniforms(rec_std);
    try std.testing.expectApproxEqAbs(@as(f32, 0.42), alpha_std.alpha_cutoff, 1e-7);
    try std.testing.expectEqual(rec_std.uv_matrices[0], alpha_std.uv_matrix);
    try std.testing.expectEqual(rec_std.uv_offsets[0], alpha_std.uv_offset);
    try std.testing.expectEqual(@as(u32, 201), rec_std.albedo_view.id);
}

test "TAA reset flags recognize cuts and explicit resets" {
    try std.testing.expect(postprocess.taaShouldReset(.{
        .first_frame = false,
        .toggled_on = false,
        .resized = false,
        .camera_cut = true,
        .explicit_reset = false,
    }));
    try std.testing.expect(postprocess.taaShouldReset(.{
        .first_frame = false,
        .toggled_on = false,
        .resized = false,
        .camera_cut = false,
        .explicit_reset = true,
    }));
    try std.testing.expect(!postprocess.taaShouldReset(.{
        .first_frame = false,
        .toggled_on = false,
        .resized = false,
        .camera_cut = false,
        .explicit_reset = false,
    }));
}

test "instance uids are stable, unique, and order the layout hash" {
    const ally = std.testing.allocator;
    var source_mesh = Mesh{
        .name = "source_cube",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 36,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var mem = [_]InstancedMesh{
        .{ .name = "i0", .source_mesh = &source_mesh, .position = Vec3.new(0, 0, 0) },
        .{ .name = "i1", .source_mesh = &source_mesh, .position = Vec3.new(2, 0, 0) },
        .{ .name = "i2", .source_mesh = &source_mesh, .position = Vec3.new(4, 0, 0) },
    };
    var ptrs = [_]*InstancedMesh{ &mem[0], &mem[1], &mem[2] };
    var mesh = Mesh{
        .name = "batched",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 36,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(5, 5, 5)),
        .instances = .{ .items = &ptrs, .capacity = 3 },
    };
    var scratch: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch.deinit(ally);

    const eye = Vec3.zero;
    const first = try staging.stageSegmentCpu(ally, &scratch, 0, null, eye, &mesh);
    const layout0 = first.layout_hash;
    // Uids assigned once, stable across a second staging.
    const uid0 = mem[0].uid;
    try std.testing.expect(uid0 != 0);
    try std.testing.expect(mem[1].uid != 0 and mem[2].uid != 0);
    try std.testing.expect(mem[0].uid != mem[1].uid and mem[1].uid != mem[2].uid);
    scratch.clearRetainingCapacity();
    const second = try staging.stageSegmentCpu(ally, &scratch, 0, null, eye, &mesh);
    try std.testing.expectEqual(uid0, mem[0].uid);
    try std.testing.expectEqual(layout0, second.layout_hash);

    // Same count, one substitution (hide i1, show a newcomer): layout flips.
    mem[1].is_visible = false;
    var mem_new = InstancedMesh{ .name = "i3", .source_mesh = &source_mesh, .position = Vec3.new(2, 0, 0) };
    ptrs[1] = &mem_new;
    mesh.instances = .{ .items = &ptrs, .capacity = 3 };
    scratch.clearRetainingCapacity();
    const subbed = try staging.stageSegmentCpu(ally, &scratch, 0, null, eye, &mesh);
    try std.testing.expect(subbed.layout_hash != layout0);
    try std.testing.expectEqual(.zero, instancePairMode(4, 4, layout0, subbed.layout_hash));

    // Reorder with identical membership: layout flips (no false pairing).
    ptrs[0] = &mem[2];
    ptrs[1] = &mem[0];
    ptrs[2] = &mem[1];
    mem[1].is_visible = true;
    mesh.instances = .{ .items = &ptrs, .capacity = 3 };
    scratch.clearRetainingCapacity();
    const reordered = try staging.stageSegmentCpu(ally, &scratch, 0, null, eye, &mesh);
    try std.testing.expect(reordered.layout_hash != layout0);

    // Growth: layout flips.
    var mem4 = InstancedMesh{ .name = "i4", .source_mesh = &source_mesh, .position = Vec3.new(6, 0, 0) };
    var ptrs4 = [_]*InstancedMesh{ &mem[0], &mem[1], &mem[2], &mem4 };
    mesh.instances = .{ .items = &ptrs4, .capacity = 4 };
    scratch.clearRetainingCapacity();
    const grown = try staging.stageSegmentCpu(ally, &scratch, 0, null, eye, &mesh);
    try std.testing.expect(grown.layout_hash != reordered.layout_hash);

    // Stable identities with changed TRS: layout UNCHANGED (real motion
    // pairs by index), only the matrix hash moves.
    var ptrs5 = [_]*InstancedMesh{ &mem[0], &mem[1], &mem[2] };
    mesh.instances = .{ .items = &ptrs5, .capacity = 3 };
    mem[0].position = Vec3.new(9, 0, 0);
    scratch.clearRetainingCapacity();
    const moved = try staging.stageSegmentCpu(ally, &scratch, 0, null, eye, &mesh);
    try std.testing.expectEqual(layout0, moved.layout_hash);
    try std.testing.expect(moved.hash != first.hash);
    try std.testing.expectEqual(.pair, instancePairMode(4, 4, layout0, moved.layout_hash));
}

test "transparent re-sort changes the layout instead of mispairing" {
    const ally = std.testing.allocator;
    var blend_mat = material_mod.StandardMaterial.init("blend_sort");
    blend_mat.alpha_mode = .blend;

    var source_mesh = Mesh{
        .name = "tsrc",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 36,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var mem = [_]InstancedMesh{
        .{ .name = "t0", .source_mesh = &source_mesh, .position = Vec3.new(0, 0, 0) },
        .{ .name = "t1", .source_mesh = &source_mesh, .position = Vec3.new(10, 0, 0) },
    };
    var ptrs = [_]*InstancedMesh{ &mem[0], &mem[1] };
    var mesh = Mesh{
        .name = "tblend",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 36,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(11, 11, 11)),
        .material = .{ .standard = &blend_mat },
        .instances = .{ .items = &ptrs, .capacity = 2 },
    };
    var scratch: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch.deinit(ally);

    // Eye on the left: farthest-first order is [self?, t1, t0]-ish; the
    // exact permutation does not matter — only that moving the eye flips
    // the layout instead of silently re-pairing positions.
    const left = try staging.stageSegmentCpu(ally, &scratch, 0, null, Vec3.new(-20, 0, 0), &mesh);
    scratch.clearRetainingCapacity();
    const right = try staging.stageSegmentCpu(ally, &scratch, 0, null, Vec3.new(20, 0, 0), &mesh);
    try std.testing.expect(left.layout_hash != right.layout_hash);
    try std.testing.expectEqual(.zero, instancePairMode(3, 3, left.layout_hash, right.layout_hash));
}

test "normalizeInstancePingPong adopts legacy states without losing handles" {
    // Coherent ping-pong: untouched.
    var coherent = InstanceRenderState{
        .buffer = .{ .id = 11 },
        .prev_buffer = .{ .id = 12 },
        .buffers = .{ .{ .id = 11 }, .{ .id = 12 } },
        .active_slot = 0,
    };
    normalizeInstancePingPong(&coherent);
    try std.testing.expectEqual(@as(u32, 11), coherent.buffer.id);
    try std.testing.expectEqual(@as(u32, 12), coherent.prev_buffer.id);
    try std.testing.expectEqual(@as(u8, 0), coherent.active_slot);

    // Legacy: live buffer outside buffers[] → adopted as slot 0, no loss.
    var legacy = InstanceRenderState{
        .buffer = .{ .id = 21 },
        .prev_buffer = .{ .id = 21 },
        .active_slot = 7,
    };
    normalizeInstancePingPong(&legacy);
    try std.testing.expectEqual(@as(u32, 21), legacy.buffers[0].id);
    try std.testing.expectEqual(@as(u32, 0), legacy.buffers[1].id);
    try std.testing.expectEqual(@as(u8, 0), legacy.active_slot);
    try std.testing.expectEqual(@as(u32, 21), legacy.prev_buffer.id);

    // Foreign prev (unknown generation): collapses to zero motion, and the
    // slot index is clamped — the owned handles stay on the books.
    var foreign = InstanceRenderState{
        .buffer = .{ .id = 31 },
        .prev_buffer = .{ .id = 99 },
        .buffers = .{ .{ .id = 31 }, .{ .id = 32 } },
        .active_slot = 5,
    };
    normalizeInstancePingPong(&foreign);
    try std.testing.expectEqual(@as(u8, 1), foreign.active_slot);
    try std.testing.expectEqual(@as(u32, 31), foreign.prev_buffer.id);
    try std.testing.expectEqual(@as(u32, 31), foreign.buffers[0].id);
    try std.testing.expectEqual(@as(u32, 32), foreign.buffers[1].id);

    // Empty state: stays empty, never fabricates handles.
    var empty = InstanceRenderState{};
    normalizeInstancePingPong(&empty);
    try std.testing.expectEqual(@as(u32, 0), empty.buffer.id);
    try std.testing.expectEqual(@as(u32, 0), empty.prev_buffer.id);
}

test "latchCreatedBuffers diffs post against pre, deduped and bounded" {
    const sokol = @import("sokol");
    const sg = sokol.gfx;
    const pre = [_]sg.Buffer{ .{ .id = 1 }, .{ .id = 2 }, .{ .id = 0 }, .{ .id = 0 } };
    var st = InstanceRenderState{
        .buffer = .{ .id = 7 },
        .prev_buffer = .{ .id = 1 },
        .buffers = .{ .{ .id = 7 }, .{ .id = 2 } },
    };
    // Only id 7 is new (1 and 2 were pre-owned); reported once.
    const created = staging.latchCreatedBuffers(pre[0], pre[1], .{ pre[2], pre[3] }, &st);
    try std.testing.expectEqual(@as(u32, 7), created[0].id);
    try std.testing.expectEqual(@as(u32, 0), created[1].id);

    // No change: nothing created.
    var same = InstanceRenderState{
        .buffer = .{ .id = 1 },
        .prev_buffer = .{ .id = 1 },
        .buffers = .{ .{ .id = 1 }, .{} },
    };
    const none = staging.latchCreatedBuffers(pre[0], pre[1], .{ pre[2], pre[3] }, &same);
    try std.testing.expectEqual(@as(u32, 0), none[0].id);
    try std.testing.expectEqual(@as(u32, 0), none[1].id);
}

test "findPrevMatrices resolves prior scratch by generation and uid" {
    var mesh = Mesh{
        .name = "prevlookup",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const uid = mesh.ensureUid();
    try std.testing.expect(uid != 0);

    // Prior slot generation 9: 4 staged matrices, this mesh owns [1..3).
    var prior_mats = [_]Mat4{
        Mat4.translation(Vec3.new(100, 0, 0)),
        Mat4.translation(Vec3.new(1, 0, 0)),
        Mat4.translation(Vec3.new(2, 0, 0)),
        Mat4.translation(Vec3.new(3, 0, 0)),
    };
    var prior_recs = [_]@import("../mesh.zig").StagedInstanceRecord{.{
        .mesh = &mesh,
        .uid = uid,
        .mesh_index = 0,
        .scratch_lo = 1,
        .count = 2,
        .bounds = BoundingBox.zero,
        .hash = 1,
        .layout_hash = 2,
        .uploaded_hash = 3,
        .uploaded_layout = 4,
        .mesh_position = Vec3.zero,
        .staged_frame = 9,
    }};
    const prev_frames = [_]staging.PrevFrameSource{
        .{ .frame_id = 7, .records = &.{}, .scratch = &.{} },
        .{ .frame_id = 9, .records = &prior_recs, .scratch = &prior_mats },
    };

    // Hit: exact slice of the prior slot's scratch (content contract: the
    // pair branch uploads this verbatim as the prev slot).
    const got = staging.findPrevMatrices(&prev_frames, 9, uid, 2).?;
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expect(got.ptr == prior_mats[1..3].ptr);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), got[0].m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), got[1].m[12], 1e-6);

    // Latch without a matching generation (superseded/dropped frame).
    try std.testing.expect(staging.findPrevMatrices(&prev_frames, 8, uid, 2) == null);
    // Unknown uid (destroyed mesh, address reuse).
    try std.testing.expect(staging.findPrevMatrices(&prev_frames, 9, uid + 1, 2) == null);
    // Zero uid never matches (unassigned legacy path).
    try std.testing.expect(staging.findPrevMatrices(&prev_frames, 9, 0, 2) == null);
    // Count mismatch (membership changed: substitution/growth/shrink).
    try std.testing.expect(staging.findPrevMatrices(&prev_frames, 9, uid, 3) == null);
    // Never-staged prior generation.
    try std.testing.expect(staging.findPrevMatrices(&prev_frames, std.math.maxInt(u64), uid, 2) == null);
    // Empty lookup set (standalone immediate path).
    try std.testing.expect(staging.findPrevMatrices(&.{}, 9, uid, 2) == null);

    // Truncated prior scratch: OOB slice fails closed, never partial.
    prior_recs[0].scratch_lo = 3;
    try std.testing.expect(staging.findPrevMatrices(&prev_frames, 9, uid, 2) == null);
}

test "commit abandon retires latch-created handles exactly once" {
    const ally = std.testing.allocator;
    var mesh = Mesh{
        .name = "abandon",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    _ = mesh.ensureUid();
    const meshes = [_]*Mesh{&mesh};

    var q: gpu_retire.GpuRetireQueue = .{};
    defer q.pending.deinit(ally);

    // A latched record whose mesh was destroyed between latch and commit
    // (pointer/uid guard fails): the fresh handles must reach the retire
    // queue instead of stranding, exactly once across repeated commits.
    var records = [_]@import("../mesh.zig").StagedInstanceRecord{.{
        .mesh = &mesh,
        .uid = 0xDEAD,
        .mesh_index = 0,
        .scratch_lo = 0,
        .count = 2,
        .bounds = BoundingBox.zero,
        .hash = 1,
        .layout_hash = 2,
        .uploaded_hash = 3,
        .uploaded_layout = 4,
        .mesh_position = Vec3.zero,
        .buffer = .{ .id = 51 },
        .prev_buffer = .{ .id = 52 },
        .buffers = .{ .{ .id = 51 }, .{ .id = 52 } },
        .capacity = 16,
        .uploaded_count = 2,
        .prev_frame = 9,
        .latch_created = .{ .{ .id = 51 }, .{ .id = 52 } },
        .staged_frame = 9,
    }};
    staging.commitPublishedRecords(&records, &meshes, 9, .{ .allocator = ally, .queue = &q });
    try std.testing.expectEqual(@as(usize, 2), q.pending.items.len);
    try std.testing.expectEqual(@as(u32, 51), q.pending.items[0].buffer.id);
    try std.testing.expectEqual(@as(u32, 52), q.pending.items[1].buffer.id);
    // Mesh untouched (keeps previous complete state).
    try std.testing.expectEqual(@as(u32, 0), mesh.instance_render.buffer.id);
    // Repeat commit: no double retire (set was cleared).
    staging.commitPublishedRecords(&records, &meshes, 9, .{ .allocator = ally, .queue = &q });
    try std.testing.expectEqual(@as(usize, 2), q.pending.items.len);

    // Success path installs and retires nothing.
    records[0].uid = mesh.uid;
    records[0].mesh_index = 0;
    var inst = InstancedMesh{ .name = "k", .source_mesh = &mesh };
    var ptrs = [_]*InstancedMesh{&inst};
    mesh.instances = .{ .items = &ptrs, .capacity = 1 };
    staging.commitPublishedRecords(&records, &meshes, 9, .{ .allocator = ally, .queue = &q });
    try std.testing.expectEqual(@as(usize, 2), q.pending.items.len);
    try std.testing.expectEqual(@as(u32, 51), mesh.instance_render.buffer.id);
    try std.testing.expectEqual(@as(u64, 9), mesh.instance_render.staged_frame);
}
