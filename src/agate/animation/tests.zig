const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Mat4 = math.Mat4;
const Skeleton = @import("skeleton.zig").Skeleton;
const Bone = @import("skeleton.zig").Bone;
const anim = @import("animation.zig");
const AnimationGroup = anim.AnimationGroup;
const AnimationSampler = anim.AnimationSampler;
const AnimationChannel = anim.AnimationChannel;
const AnimationPath = anim.AnimationPath;
const AnimationInterpolation = anim.AnimationInterpolation;
const NodeChannel = anim.NodeChannel;
const NodeTarget = anim.NodeTarget;
const MorphWeightsTarget = anim.MorphWeightsTarget;
const AnimationEvent = anim.AnimationEvent;
const evaluateSkeleton = anim.evaluateSkeleton;
const EasingType = anim.EasingType;

test "AnimationSampler linear interpolation" {
    const times = [_]f32{ 0.0, 1.0, 2.0 };
    const values = [_]f32{ 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 20.0, 40.0, 60.0 };

    const sampler = AnimationSampler{
        .timestamps = &times,
        .outputs = &values,
        .interpolation = .linear,
    };

    const mid = sampler.sampleVec3(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), mid.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), mid.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), mid.z, 1e-4);
}

test "AnimationGroup playRange, speed, and reverse looping" {
    const allocator = std.testing.allocator;
    const times = try allocator.alloc(f32, 2);
    times[0] = 0.0;
    times[1] = 4.0;
    const outputs = try allocator.alloc(f32, 6);
    outputs[0] = 0.0;
    outputs[1] = 0.0;
    outputs[2] = 0.0;
    outputs[3] = 4.0;
    outputs[4] = 8.0;
    outputs[5] = 12.0;

    const channels = try allocator.alloc(AnimationChannel, 1);
    channels[0] = .{
        .bone_index = 0,
        .target_path = .translation,
        .sampler = .{
            .timestamps = times,
            .outputs = outputs,
        },
    };

    const ag = try AnimationGroup.init(allocator, "test_range", channels, 4.0);
    defer ag.deinit();

    // 1. Forward playback in range [1.0 .. 3.0] at 2.0x speed
    ag.playRange(1.0, 3.0, true, 2.0);
    try std.testing.expectEqual(@as(f32, 1.0), ag.current_time);

    ag.update(0.5); // dt=0.5, speed=2.0 -> advance 1.0s -> current_time = 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), ag.current_time, 1e-4);

    ag.update(0.6); // dt=0.6, speed=2.0 -> advance 1.2s -> 3.2s -> wraps to 1.2s
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), ag.current_time, 1e-4);

    // 2. Reverse playback at -1.0x speed
    ag.setSpeed(-1.0);
    ag.update(0.5); // 1.2 - 0.5 = 0.7 -> below 1.0 (range is 2.0) -> wraps to 2.7s
    try std.testing.expectApproxEqAbs(@as(f32, 2.7), ag.current_time, 1e-4);
}

test "evaluateSkeleton blending and additive layer" {
    const allocator = std.testing.allocator;
    const skel = try Skeleton.init(allocator, 1);
    defer skel.deinit();

    skel.bones[0].bind_position = Vec3.zero;
    skel.bones[0].bind_rotation = Quat.identity;
    skel.bones[0].bind_scale = Vec3.one;

    // Clip 1: Walk (pos: 0, 0, 10)
    const times1 = try allocator.alloc(f32, 1);
    times1[0] = 0.0;
    const out1 = try allocator.alloc(f32, 3);
    out1[0] = 0.0;
    out1[1] = 0.0;
    out1[2] = 10.0;
    const ch1 = try allocator.alloc(AnimationChannel, 1);
    ch1[0] = .{ .bone_index = 0, .target_path = .translation, .sampler = .{ .timestamps = times1, .outputs = out1 } };
    const ag1 = try AnimationGroup.init(allocator, "walk", ch1, 1.0);
    defer ag1.deinit();
    ag1.play(true);

    // Clip 2: Run (pos: 0, 0, 20)
    const times2 = try allocator.alloc(f32, 1);
    times2[0] = 0.0;
    const out2 = try allocator.alloc(f32, 3);
    out2[0] = 0.0;
    out2[1] = 0.0;
    out2[2] = 20.0;
    const ch2 = try allocator.alloc(AnimationChannel, 1);
    ch2[0] = .{ .bone_index = 0, .target_path = .translation, .sampler = .{ .timestamps = times2, .outputs = out2 } };
    const ag2 = try AnimationGroup.init(allocator, "run", ch2, 1.0);
    defer ag2.deinit();
    ag2.play(true);

    // Test 50/50 blend: pos should be 15.0
    ag1.setWeight(0.5);
    ag2.setWeight(0.5);
    const base_groups = [_]*AnimationGroup{ ag1, ag2 };
    evaluateSkeleton(skel, &base_groups, &.{});
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), skel.bones[0].local_position.z, 1e-4);

    // Test Additive: Add layer (pos: 0, 5, 0)
    const times3 = try allocator.alloc(f32, 1);
    times3[0] = 0.0;
    const out3 = try allocator.alloc(f32, 3);
    out3[0] = 0.0;
    out3[1] = 5.0;
    out3[2] = 0.0;
    const ch3 = try allocator.alloc(AnimationChannel, 1);
    ch3[0] = .{ .bone_index = 0, .target_path = .translation, .sampler = .{ .timestamps = times3, .outputs = out3 } };
    const ag3 = try AnimationGroup.init(allocator, "jump_add", ch3, 1.0);
    defer ag3.deinit();
    ag3.setAdditive(true);
    ag3.setWeight(1.0);
    ag3.play(true);

    const add_groups = [_]*AnimationGroup{ag3};
    evaluateSkeleton(skel, &base_groups, &add_groups);

    // Blended z should still be 15.0, and y should now be 5.0!
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), skel.bones[0].local_position.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), skel.bones[0].local_position.y, 1e-4);
}

test "AnimationGroup crossFadeTo, fadeIn, and fadeOut" {
    const allocator = std.testing.allocator;

    const times1 = try allocator.alloc(f32, 1);
    times1[0] = 0.0;
    const out1 = try allocator.alloc(f32, 3);
    out1[0] = 0;
    out1[1] = 0;
    out1[2] = 0;
    const ch1 = try allocator.alloc(AnimationChannel, 1);
    ch1[0] = .{ .bone_index = 0, .target_path = .translation, .sampler = .{ .timestamps = times1, .outputs = out1 } };
    const ag1 = try AnimationGroup.init(allocator, "clip1", ch1, 1.0);
    defer ag1.deinit();

    const times2 = try allocator.alloc(f32, 1);
    times2[0] = 0.0;
    const out2 = try allocator.alloc(f32, 3);
    out2[0] = 0;
    out2[1] = 0;
    out2[2] = 0;
    const ch2 = try allocator.alloc(AnimationChannel, 1);
    ch2[0] = .{ .bone_index = 0, .target_path = .translation, .sampler = .{ .timestamps = times2, .outputs = out2 } };
    const ag2 = try AnimationGroup.init(allocator, "clip2", ch2, 1.0);
    defer ag2.deinit();

    ag1.play(true);
    ag1.setWeight(1.0);

    // Cross-fade ag1 -> ag2 over 1.0 second
    ag1.crossFadeTo(ag2, 1.0);

    try std.testing.expect(ag1.is_playing);
    try std.testing.expect(ag2.is_playing);
    try std.testing.expectEqual(@as(f32, 1.0), ag1.weight);
    try std.testing.expectEqual(@as(f32, 0.0), ag2.weight);

    // Advance 0.5s: 50% through cross-fade
    ag1.update(0.5);
    ag2.update(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), ag1.weight, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), ag2.weight, 1e-4);

    // Advance another 0.5s (total 1.0s): cross-fade completed
    ag1.update(0.5);
    ag2.update(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ag1.weight, 1e-4);
    try std.testing.expect(!ag1.is_playing); // ag1 automatically stopped
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ag2.weight, 1e-4);
    try std.testing.expect(ag2.is_playing); // ag2 continues playing at full weight
}

// Stand-in for Mesh in node tests: animation.zig cannot import mesh.zig
// (mesh -> scene -> animation cycle), so tests bind plain Vec3 fields.
const TestNode = struct {
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero, // Euler degrees, matches Mesh.rotation
    scaling: Vec3 = Vec3.one,
};

fn makeNodeGroup(allocator: std.mem.Allocator, name: []const u8, duration: f32) !*AnimationGroup {
    return AnimationGroup.init(allocator, name, &[_]AnimationChannel{}, duration);
}

test "NodeChannel translation applies over time with loop" {
    const allocator = std.testing.allocator;
    var node = TestNode{};

    const times = try allocator.alloc(f32, 2);
    times[0] = 0.0;
    times[1] = 2.0;
    const outputs = try allocator.alloc(f32, 6);
    outputs[0] = 0.0;
    outputs[1] = 0.0;
    outputs[2] = 0.0;
    outputs[3] = 10.0;
    outputs[4] = 0.0;
    outputs[5] = 0.0;

    const ag = try makeNodeGroup(allocator, "node_move", 2.0);
    defer ag.deinit();
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, false);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = times, .outputs = outputs, .interpolation = .linear },
    });

    ag.play(true);
    ag.update(1.0); // t = 1.0 -> halfway
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), node.position.x, 1e-4);

    ag.update(1.5); // t = 2.5 -> wraps to 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), node.position.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), ag.current_time, 1e-4);
}

test "NodeChannel rotation slerp and scale lerp" {
    const allocator = std.testing.allocator;
    var node = TestNode{};

    const q0 = Quat.identity;
    const q1 = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));

    const r_times = try allocator.alloc(f32, 2);
    r_times[0] = 0.0;
    r_times[1] = 1.0;
    const r_out = try allocator.alloc(f32, 8);
    r_out[0] = q0.x;
    r_out[1] = q0.y;
    r_out[2] = q0.z;
    r_out[3] = q0.w;
    r_out[4] = q1.x;
    r_out[5] = q1.y;
    r_out[6] = q1.z;
    r_out[7] = q1.w;

    const s_times = try allocator.alloc(f32, 2);
    s_times[0] = 0.0;
    s_times[1] = 1.0;
    const s_out = try allocator.alloc(f32, 6);
    s_out[0] = 1.0;
    s_out[1] = 1.0;
    s_out[2] = 1.0;
    s_out[3] = 2.0;
    s_out[4] = 2.0;
    s_out[5] = 2.0;

    const ag = try makeNodeGroup(allocator, "node_rot_scale", 1.0);
    defer ag.deinit();
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, false);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .rotation,
        .sampler = .{ .timestamps = r_times, .outputs = r_out, .interpolation = .linear },
    });
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .scale,
        .sampler = .{ .timestamps = s_times, .outputs = s_out, .interpolation = .linear },
    });

    ag.play(true);
    ag.goToFrame(0.5);

    // Slerp halfway between identity and 90 deg Y -> 45 deg Y.
    try std.testing.expectApproxEqAbs(@as(f32, 45.0), node.rotation.y, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), node.rotation.x, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), node.rotation.z, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), node.scaling.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), node.scaling.y, 1e-4);
}

test "NodeChannel STEP holds keyframe while LINEAR blends" {
    const allocator = std.testing.allocator;
    var step_node = TestNode{};
    var linear_node = TestNode{};

    const mk_track = struct {
        fn mk(alloc: std.mem.Allocator) !struct { t: []f32, o: []f32 } {
            const t = try alloc.alloc(f32, 2);
            t[0] = 0.0;
            t[1] = 1.0;
            const o = try alloc.alloc(f32, 6);
            o[0] = 0.0;
            o[1] = 0.0;
            o[2] = 0.0;
            o[3] = 10.0;
            o[4] = 0.0;
            o[5] = 0.0;
            return .{ .t = t, .o = o };
        }
    }.mk;

    const step_track = try mk_track(allocator);
    const ag_step = try makeNodeGroup(allocator, "node_step", 1.0);
    defer ag_step.deinit();
    const step_target = try ag_step.bindNodeTarget(&step_node.position, &step_node.rotation, &step_node.scaling, false);
    try ag_step.addNodeChannel(.{
        .target = step_target,
        .target_path = .translation,
        .sampler = .{ .timestamps = step_track.t, .outputs = step_track.o, .interpolation = .step },
    });

    const lin_track = try mk_track(allocator);
    const ag_lin = try makeNodeGroup(allocator, "node_lin", 1.0);
    defer ag_lin.deinit();
    const lin_target = try ag_lin.bindNodeTarget(&linear_node.position, &linear_node.rotation, &linear_node.scaling, false);
    try ag_lin.addNodeChannel(.{
        .target = lin_target,
        .target_path = .translation,
        // Default easing (.linear) must behave as plain linear interpolation.
        .sampler = .{ .timestamps = lin_track.t, .outputs = lin_track.o, .interpolation = .linear },
    });

    ag_step.play(true);
    ag_lin.play(true);
    ag_step.goToFrame(0.5);
    ag_lin.goToFrame(0.5);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), step_node.position.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), linear_node.position.x, 1e-4);
}

test "NodeChannel weight blends sample with rest pose" {
    const allocator = std.testing.allocator;
    var node = TestNode{
        .position = Vec3.new(0.0, 10.0, 0.0),
    };

    const times = try allocator.alloc(f32, 2);
    times[0] = 0.0;
    times[1] = 1.0;
    const outputs = try allocator.alloc(f32, 6);
    outputs[0] = 0.0;
    outputs[1] = 0.0;
    outputs[2] = 0.0;
    outputs[3] = 0.0;
    outputs[4] = 0.0;
    outputs[5] = 20.0;

    const ag = try makeNodeGroup(allocator, "node_weight", 1.0);
    defer ag.deinit();
    // Rest pose (0, 10, 0) is captured at bind time.
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, false);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = times, .outputs = outputs, .interpolation = .linear },
    });

    ag.play(true);
    ag.setWeight(0.5);
    ag.goToFrame(1.0); // sampled (0, 0, 20), blended with rest (0, 10, 0)

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), node.position.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), node.position.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), node.position.z, 1e-4);

    // Zero weight fully restores the rest pose.
    ag.setWeight(0.0);
    ag.goToFrame(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), node.position.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), node.position.z, 1e-4);
}

test "NodeChannel invalid targets and samplers never crash" {
    const allocator = std.testing.allocator;
    var node = TestNode{
        .position = Vec3.new(1.0, 2.0, 3.0),
    };

    const times = try allocator.alloc(f32, 1);
    times[0] = 0.0;
    const outputs = try allocator.alloc(f32, 3);
    outputs[0] = 9.0;
    outputs[1] = 9.0;
    outputs[2] = 9.0;

    const w_out = try allocator.alloc(f32, 1);
    w_out[0] = 0.5;

    const ag = try makeNodeGroup(allocator, "node_invalid", 1.0);
    defer ag.deinit();
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, false);

    // Out-of-range target index.
    try ag.addNodeChannel(.{
        .target = target + 7,
        .target_path = .translation,
        .sampler = .{ .timestamps = times, .outputs = outputs },
    });
    // Weights path on a plain node is ignored.
    const w_times = try allocator.alloc(f32, 1);
    w_times[0] = 0.0;
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .weights,
        .sampler = .{ .timestamps = w_times, .outputs = w_out },
    });
    // Empty sampler (static slices, never freed).
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = &.{}, .outputs = &.{} },
    });
    // Truncated outputs (1 float for a vec3 track) are skipped.
    const bad_out = try allocator.alloc(f32, 1);
    bad_out[0] = 42.0;
    const bad_times = try allocator.alloc(f32, 1);
    bad_times[0] = 0.0;
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = bad_times, .outputs = bad_out },
    });

    ag.play(true);
    ag.update(0.5);
    ag.goToFrame(0.0);

    var p: ?Vec3 = null;
    var r: ?Quat = null;
    var s: ?Vec3 = null;
    ag.sampleNodeAtTime(target + 99, 0.0, &p, &r, &s);
    try std.testing.expect(p == null and r == null and s == null);

    // Nothing valid applied, transform untouched.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), node.position.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), node.position.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), node.position.z, 1e-4);
}

test "NodeChannel skipped target is parsed but not applied" {
    const allocator = std.testing.allocator;
    var node = TestNode{};

    const times = try allocator.alloc(f32, 1);
    times[0] = 0.0;
    const outputs = try allocator.alloc(f32, 3);
    outputs[0] = 5.0;
    outputs[1] = 6.0;
    outputs[2] = 7.0;

    const ag = try makeNodeGroup(allocator, "node_skip", 1.0);
    defer ag.deinit();
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, true);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = times, .outputs = outputs },
    });

    ag.play(true);
    ag.update(0.5);
    ag.stop(); // skipped targets are not restored either

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), node.position.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), node.position.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), node.position.z, 1e-4);
}

test "NodeChannel easing warps interpolation and stop restores rest" {
    const allocator = std.testing.allocator;
    var node = TestNode{};

    const times = try allocator.alloc(f32, 2);
    times[0] = 0.0;
    times[1] = 1.0;
    const outputs = try allocator.alloc(f32, 6);
    outputs[0] = 0.0;
    outputs[1] = 0.0;
    outputs[2] = 0.0;
    outputs[3] = 10.0;
    outputs[4] = 0.0;
    outputs[5] = 0.0;

    const ag = try makeNodeGroup(allocator, "node_ease", 1.0);
    defer ag.deinit();
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, false);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = times, .outputs = outputs, .interpolation = .linear },
        .easing = .ease_in_quad,
    });

    ag.play(true);
    ag.goToFrame(0.5); // ease_in_quad(0.5) = 0.25 -> x = 2.5
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), node.position.x, 1e-4);

    var p: ?Vec3 = null;
    var r: ?Quat = null;
    var s: ?Vec3 = null;
    ag.sampleNodeAtTime(target, 0.5, &p, &r, &s);
    try std.testing.expect(p != null);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), p.?.x, 1e-4);

    ag.stop();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), node.position.x, 1e-4);
    try std.testing.expect(!ag.is_playing);
}

test "Morph weights channel samples N values per key" {
    const allocator = std.testing.allocator;
    var weights = [_]f32{ 0.0, 0.0 };
    var dirty = false;

    const times = try allocator.alloc(f32, 2);
    times[0] = 0.0;
    times[1] = 1.0;
    const outputs = try allocator.alloc(f32, 4);
    outputs[0] = 0.0;
    outputs[1] = 0.0;
    outputs[2] = 1.0;
    outputs[3] = 0.5;

    const ag = try makeNodeGroup(allocator, "morph_weights", 1.0);
    defer ag.deinit();
    const target = try ag.bindMorphTarget(&weights, &dirty);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .weights,
        .sampler = .{ .timestamps = times, .outputs = outputs, .interpolation = .linear },
    });

    ag.play(true);
    ag.goToFrame(0.5); // linear midpoint between [0,0] and [1,0.5]
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), weights[1], 1e-4);
    try std.testing.expect(dirty);

    // Group weight blends the sample with the rest pose ([0,0] here).
    dirty = false;
    ag.setWeight(0.5);
    ag.goToFrame(1.0); // sampled [1,0.5] -> [0.5,0.25]
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), weights[1], 1e-4);
    try std.testing.expect(dirty);

    // stop() restores rest weights and flags dirty for the scene re-blend.
    dirty = false;
    ag.stop();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), weights[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), weights[1], 1e-6);
    try std.testing.expect(dirty);
}

test "Morph weights channel with truncated outputs is skipped" {
    const allocator = std.testing.allocator;
    var weights = [_]f32{ 0.2, 0.3 };
    var dirty = false;

    // 2 keys but only 2 outputs for 2 targets (needs 4): skipped entirely.
    const bad_times = try allocator.alloc(f32, 2);
    bad_times[0] = 0.0;
    bad_times[1] = 1.0;
    const bad_out = try allocator.alloc(f32, 2);
    bad_out[0] = 0.9;
    bad_out[1] = 0.9;

    const ag = try makeNodeGroup(allocator, "morph_guard", 1.0);
    defer ag.deinit();
    const target = try ag.bindMorphTarget(&weights, &dirty);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .weights,
        .sampler = .{ .timestamps = bad_times, .outputs = bad_out, .interpolation = .linear },
    });

    ag.play(true);
    ag.goToFrame(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), weights[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), weights[1], 1e-6);
    try std.testing.expect(!dirty);
}

// Cubic-spline (Hermite) tests. Per-key layout everywhere below:
// (in-tangent, value, out-tangent); tangents are per-second slopes.

test "Cubic spline vec3 hits keyframe values exactly and clamps" {
    const times = [_]f32{ 0.0, 1.0 };
    // k0: in=(100,0,0) value=(0,0,0) out=(0,0,0);
    // k1: in=(0,0,0) value=(1,2,3) out=(200,0,0).
    const values = [_]f32{
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0, 0.0,
        0.0,   0.0, 0.0, 1.0, 2.0, 3.0, 200.0, 0.0, 0.0,
    };
    const sampler = AnimationSampler{
        .timestamps = &times,
        .outputs = &values,
        .interpolation = .cubic_spline,
    };

    const v0 = sampler.sampleVec3(0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v0.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v0.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v0.z, 1e-6);

    const v1 = sampler.sampleVec3(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), v1.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), v1.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), v1.z, 1e-6);

    // Zero tangents on both sides of the segment -> smoothstep midpoint.
    const mid = sampler.sampleVec3(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mid.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), mid.z, 1e-5);

    const clo = sampler.sampleVec3(-5.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), clo.x, 1e-6);
    const chi = sampler.sampleVec3(99.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), chi.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), chi.z, 1e-6);
}

test "Cubic spline midpoint follows Hermite basis with dt-scaled tangents" {
    const times = [_]f32{ 0.0, 2.0 };
    // x: p0=0, out0=3 (m0 = 3*dt = 6), p1=1, in1=0. At s=0.5:
    // 0.5*0 + 0.125*6 + 0.5*1 - 0.125*0 = 1.25, well above linear 0.5.
    // (Without the dt scaling the answer would be 0.875, not 1.25.)
    const values = [_]f32{
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    const sampler = AnimationSampler{
        .timestamps = &times,
        .outputs = &values,
        .interpolation = .cubic_spline,
    };
    const mid = sampler.sampleVec3(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), mid.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mid.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mid.z, 1e-6);
}

test "Cubic spline rotation is normalized and smooth" {
    const q0 = Quat.identity;
    const q1 = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));
    const times = [_]f32{ 0.0, 1.0 };
    const values = [_]f32{
        0.0, 0.0, 0.0, 0.0, q0.x, q0.y, q0.z, q0.w, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, q1.x, q1.y, q1.z, q1.w, 0.0, 0.0, 0.0, 0.0,
    };
    const sampler = AnimationSampler{
        .timestamps = &times,
        .outputs = &values,
        .interpolation = .cubic_spline,
    };

    const start = sampler.sampleQuat(0.0);
    const dot0 = start.x * q0.x + start.y * q0.y + start.z * q0.z + start.w * q0.w;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @abs(dot0), 1e-5);

    const end = sampler.sampleQuat(1.0);
    const dot1 = end.x * q1.x + end.y * q1.y + end.z * q1.z + end.w * q1.w;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @abs(dot1), 1e-5);

    // Zero tangents: midpoint is normalize((q0+q1)/2), a ~45 deg Y turn.
    const mid = sampler.sampleQuat(0.5);
    const len_sq = mid.x * mid.x + mid.y * mid.y + mid.z * mid.z + mid.w * mid.w;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), len_sq, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 45.0), mid.toEulerDeg().y, 0.5);
}

test "Cubic spline morph weights interpolate per target" {
    const allocator = std.testing.allocator;
    var weights = [_]f32{ 0.0, 0.0 };
    var dirty = false;

    const times = try allocator.alloc(f32, 2);
    times[0] = 0.0;
    times[1] = 1.0;
    // n=2: k0 in=[9,9] val=[0,0] out=[3,0]; k1 in=[0,0] val=[1,0.5] out=[9,9].
    const outputs = try allocator.alloc(f32, 12);
    outputs[0] = 9.0;
    outputs[1] = 9.0;
    outputs[2] = 0.0;
    outputs[3] = 0.0;
    outputs[4] = 3.0;
    outputs[5] = 0.0;
    outputs[6] = 0.0;
    outputs[7] = 0.0;
    outputs[8] = 1.0;
    outputs[9] = 0.5;
    outputs[10] = 9.0;
    outputs[11] = 9.0;

    const ag = try makeNodeGroup(allocator, "morph_cubic", 1.0);
    defer ag.deinit();
    const target = try ag.bindMorphTarget(&weights, &dirty);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .weights,
        .sampler = .{ .timestamps = times, .outputs = outputs, .interpolation = .cubic_spline },
    });

    ag.play(true);
    ag.goToFrame(0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), weights[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), weights[1], 1e-6);

    ag.goToFrame(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), weights[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights[1], 1e-6);

    // Target 0: 0.5*0 + 0.125*3 + 0.5*1 = 0.875; target 1: 0.5*0.5 = 0.25.
    ag.goToFrame(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.875), weights[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), weights[1], 1e-5);
    try std.testing.expect(dirty);
}

test "Truncated cubic sampler never crashes and degrades gracefully" {
    const allocator = std.testing.allocator;
    const times = [_]f32{ 0.0, 1.0 };
    const tiny = [_]f32{ 1.0, 2.0, 3.0 };
    const bad = AnimationSampler{
        .timestamps = &times,
        .outputs = &tiny,
        .interpolation = .cubic_spline,
    };
    // Zero usable frames: vec3 degrades to zero, quat to identity.
    const v = bad.sampleVec3(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v.x, 1e-6);
    const q = bad.sampleQuat(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), q.w, 1e-6);
    var w = [_]f32{ 0.7, 0.8 };
    bad.sampleWeightsInto(0.5, &w, .linear);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), w[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), w[1], 1e-6);

    // Exactly one usable frame: every sample clamps to that key's value.
    const one = [_]f32{ 9.0, 9.0, 9.0, 4.0, 5.0, 6.0, 9.0, 9.0, 9.0 };
    const single = AnimationSampler{
        .timestamps = &times,
        .outputs = &one,
        .interpolation = .cubic_spline,
    };
    const sv = single.sampleVec3(0.75);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), sv.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), sv.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), sv.z, 1e-6);

    // Group path: truncated cubic channels are skipped, never applied.
    var node = TestNode{ .position = Vec3.new(1.0, 2.0, 3.0) };
    const bad_times = try allocator.alloc(f32, 2);
    bad_times[0] = 0.0;
    bad_times[1] = 1.0;
    const bad_out = try allocator.alloc(f32, 3);
    bad_out[0] = 42.0;
    bad_out[1] = 42.0;
    bad_out[2] = 42.0;
    const ag = try makeNodeGroup(allocator, "cubic_guard", 1.0);
    defer ag.deinit();
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, false);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = bad_times, .outputs = bad_out, .interpolation = .cubic_spline },
    });
    ag.play(true);
    ag.goToFrame(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), node.position.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), node.position.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), node.position.z, 1e-6);
}

// Event tests: counts callback invocations alongside drainFiredEvents().
const EventCounter = struct {
    count: usize = 0,
    last_name: []const u8 = "",
    fn handler(ctx: ?*anyopaque, name: []const u8) void {
        const self: *EventCounter = @ptrCast(@alignCast(ctx.?));
        self.count += 1;
        self.last_name = name;
    }
};

test "Animation events fire once on crossing and drain clears" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "ev_once", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{.{ .time = 1.0, .name = "hit" }});
    var counter = EventCounter{};
    ag.on_event = &EventCounter.handler;
    ag.event_context = &counter;

    ag.play(false);
    ag.update(0.6); // t = 0.6, no crossing
    try std.testing.expectEqual(@as(usize, 0), counter.count);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);

    ag.update(0.6); // t = 1.2, crosses 1.0 exactly once
    try std.testing.expectEqual(@as(usize, 1), counter.count);
    try std.testing.expectEqualStrings("hit", counter.last_name);
    const fired = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 1), fired.len);
    try std.testing.expectEqualStrings("hit", fired[0]);

    // Draining clears: a second drain is empty, and later updates refire nothing.
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
    ag.update(0.5); // t = 1.7, no new crossing
    try std.testing.expectEqual(@as(usize, 1), counter.count);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
}

test "Animation events fire across a loop wraparound" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "ev_loop", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{
        .{ .time = 0.25, .name = "wrapped" },
        .{ .time = 1.5, .name = "end" },
        .{ .time = 0.0, .name = "start" },
    });

    ag.play(true);
    // Window (0, 1] crosses "wrapped"@0.25: prev_time starts at from=0, so
    // only the t == 0 event stays below the open lower bound on this pass.
    ag.update(1.0);
    const f0 = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 1), f0.len);
    try std.testing.expectEqualStrings("wrapped", f0[0]);

    ag.update(1.0); // t = 2.0, window (1, 2] fires "end"
    const f1 = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 1), f1.len);
    try std.testing.expectEqualStrings("end", f1[0]);

    // Wraps to t = 1.0: windows (2, 2] (empty) + [0, 1] fire "wrapped" and
    // the zero-boundary "start", each exactly once, in stored order.
    ag.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ag.current_time, 1e-4);
    const f2 = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 2), f2.len);
    try std.testing.expectEqualStrings("wrapped", f2[0]);
    try std.testing.expectEqualStrings("start", f2[1]);
}

test "Animation events fire when playing backward" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "ev_back", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{.{ .time = 1.0, .name = "hit" }});

    ag.play(true);
    ag.setSpeed(-1.0);
    ag.goToFrame(1.5); // seek: must not fire
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);

    ag.update(1.0); // t = 0.5, window [0.5, 1.5) fires "hit" once
    const f1 = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 1), f1.len);
    try std.testing.expectEqualStrings("hit", f1[0]);

    // Wraps to t = 1.5: windows [0, 0.5) + (1.5, 2] miss 1.0, no refire.
    ag.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), ag.current_time, 1e-4);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
}

test "Animation events fire on the duration boundary" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "ev_edge", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{.{ .time = 2.0, .name = "end" }});

    ag.play(false);
    ag.update(2.5); // overshoots to == 2.0, non-loop stops and still fires
    try std.testing.expect(!ag.is_playing);
    const f1 = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 1), f1.len);
    try std.testing.expectEqualStrings("end", f1[0]);

    ag.update(1.0); // stopped: nothing may fire afterwards
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
}

test "stop and fade-out never fire events" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "ev_nofire", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{
        .{ .time = 0.5, .name = "early" },
        .{ .time = 1.0, .name = "late" },
    });

    ag.play(true);
    ag.update(0.4); // t = 0.4, no crossing yet
    ag.stop(); // must not fire and must drop the pending snapshot
    try std.testing.expect(!ag.is_playing);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);

    // A fade-out that completes inside this update stops playback via the
    // early-return path: time would cross both events, but none may fire.
    ag.play(true);
    ag.fadeOut(0.5);
    ag.update(1.5);
    try std.testing.expect(!ag.is_playing);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
}

test "Truncated LINEAR/STEP bone tracks are skipped, preserving pose" {
    const allocator = std.testing.allocator;
    const skel = try Skeleton.init(allocator, 1);
    defer skel.deinit();
    skel.bones[0].bind_position = Vec3.new(1.0, 2.0, 3.0);
    skel.bones[0].bind_rotation = Quat.identity;
    skel.bones[0].bind_scale = Vec3.one;
    skel.bones[0].local_position = Vec3.new(1.0, 2.0, 3.0);
    skel.bones[0].local_rotation = Quat.identity;
    skel.bones[0].local_scale = Vec3.one;

    // 2 keys but only 1 frame of outputs: truncated LINEAR translation.
    const t_times = try allocator.alloc(f32, 2);
    t_times[0] = 0.0;
    t_times[1] = 1.0;
    const t_out = try allocator.alloc(f32, 3);
    t_out[0] = 99.0;
    t_out[1] = 99.0;
    t_out[2] = 99.0;
    // 2 keys but only 1 frame of outputs: truncated STEP rotation.
    const r_times = try allocator.alloc(f32, 2);
    r_times[0] = 0.0;
    r_times[1] = 1.0;
    const r_out = try allocator.alloc(f32, 4);
    r_out[0] = 0.0;
    r_out[1] = 0.0;
    r_out[2] = 0.0;
    r_out[3] = 1.0;

    const channels = try allocator.alloc(AnimationChannel, 2);
    channels[0] = .{
        .bone_index = 0,
        .target_path = .translation,
        .sampler = .{ .timestamps = t_times, .outputs = t_out, .interpolation = .linear },
    };
    channels[1] = .{
        .bone_index = 0,
        .target_path = .rotation,
        .sampler = .{ .timestamps = r_times, .outputs = r_out, .interpolation = .step },
    };

    const ag = try AnimationGroup.init(allocator, "bone_truncated", channels, 1.0);
    defer ag.deinit();
    ag.skeleton = skel;

    // Direct sampling: invalid tracks leave nullable outputs null (all OOB
    // branches: first/mid/last keyframe).
    for ([_]f32{ 0.0, 0.5, 1.0 }) |t| {
        var p: ?Vec3 = null;
        var r: ?Quat = null;
        var s: ?Vec3 = null;
        ag.sampleBoneAtTime(0, t, &p, &r, &s);
        try std.testing.expect(p == null);
        try std.testing.expect(r == null);
        try std.testing.expect(s == null);
    }

    // Direct apply: invalid tracks preserve the pose, never write 99s.
    for ([_]f32{ 0.0, 0.5, 1.0 }) |t| {
        ag.applyAtTime(t);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), skel.bones[0].local_position.x, 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), skel.bones[0].local_position.y, 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 3.0), skel.bones[0].local_position.z, 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), skel.bones[0].local_rotation.w, 1e-6);
    }

    // Cubic fallback preserved: a truncated CUBICSPLINE bone track still
    // reaches the sampler (zero usable frames -> Vec3.zero), not skipped.
    const c_times = try allocator.alloc(f32, 2);
    c_times[0] = 0.0;
    c_times[1] = 1.0;
    const c_out = try allocator.alloc(f32, 3);
    c_out[0] = 42.0;
    c_out[1] = 42.0;
    c_out[2] = 42.0;
    const c_ch = try allocator.alloc(AnimationChannel, 1);
    c_ch[0] = .{
        .bone_index = 0,
        .target_path = .translation,
        .sampler = .{ .timestamps = c_times, .outputs = c_out, .interpolation = .cubic_spline },
    };
    const ag_cubic = try AnimationGroup.init(allocator, "bone_truncated_cubic", c_ch, 1.0);
    defer ag_cubic.deinit();
    ag_cubic.skeleton = skel;
    var cp: ?Vec3 = null;
    var cr: ?Quat = null;
    var cs: ?Vec3 = null;
    ag_cubic.sampleBoneAtTime(0, 0.5, &cp, &cr, &cs);
    try std.testing.expect(cp != null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cp.?.x, 1e-6);
    skel.bones[0].local_position = Vec3.new(1.0, 2.0, 3.0);
    ag_cubic.applyAtTime(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), skel.bones[0].local_position.x, 1e-6);
}

test "Truncated bone tracks fall back to bind pose in skeleton blending" {
    const allocator = std.testing.allocator;
    const skel = try Skeleton.init(allocator, 1);
    defer skel.deinit();
    skel.bones[0].bind_position = Vec3.zero;
    skel.bones[0].bind_rotation = Quat.identity;
    skel.bones[0].bind_scale = Vec3.one;

    // Bad clip: truncated LINEAR translation + truncated STEP rotation.
    const bad_t = try allocator.alloc(f32, 2);
    bad_t[0] = 0.0;
    bad_t[1] = 1.0;
    const bad_t_out = try allocator.alloc(f32, 3);
    bad_t_out[0] = 99.0;
    bad_t_out[1] = 99.0;
    bad_t_out[2] = 99.0;
    const bad_r = try allocator.alloc(f32, 2);
    bad_r[0] = 0.0;
    bad_r[1] = 1.0;
    const bad_r_out = try allocator.alloc(f32, 4);
    bad_r_out[0] = 0.0;
    bad_r_out[1] = 0.0;
    bad_r_out[2] = 0.0;
    bad_r_out[3] = 1.0;
    const bad_ch = try allocator.alloc(AnimationChannel, 2);
    bad_ch[0] = .{
        .bone_index = 0,
        .target_path = .translation,
        .sampler = .{ .timestamps = bad_t, .outputs = bad_t_out, .interpolation = .linear },
    };
    bad_ch[1] = .{
        .bone_index = 0,
        .target_path = .rotation,
        .sampler = .{ .timestamps = bad_r, .outputs = bad_r_out, .interpolation = .step },
    };
    const ag_bad = try AnimationGroup.init(allocator, "bad", bad_ch, 1.0);
    defer ag_bad.deinit();
    ag_bad.play(true);

    // Good clip: valid translation (z=20) + valid 90deg Y rotation.
    const q1 = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));
    const good_t = try allocator.alloc(f32, 1);
    good_t[0] = 0.0;
    const good_t_out = try allocator.alloc(f32, 3);
    good_t_out[0] = 0.0;
    good_t_out[1] = 0.0;
    good_t_out[2] = 20.0;
    const good_r = try allocator.alloc(f32, 1);
    good_r[0] = 0.0;
    const good_r_out = try allocator.alloc(f32, 4);
    good_r_out[0] = q1.x;
    good_r_out[1] = q1.y;
    good_r_out[2] = q1.z;
    good_r_out[3] = q1.w;
    const good_ch = try allocator.alloc(AnimationChannel, 2);
    good_ch[0] = .{
        .bone_index = 0,
        .target_path = .translation,
        .sampler = .{ .timestamps = good_t, .outputs = good_t_out },
    };
    good_ch[1] = .{
        .bone_index = 0,
        .target_path = .rotation,
        .sampler = .{ .timestamps = good_r, .outputs = good_r_out },
    };
    const ag_good = try AnimationGroup.init(allocator, "good", good_ch, 1.0);
    defer ag_good.deinit();
    ag_good.play(true);

    // 50/50 blend: bad side falls back to bind, so translation is the
    // midpoint bind<->good (z=10), rotation is ~45deg Y. A truncated track
    // writing 99s would fail both checks.
    ag_bad.setWeight(0.5);
    ag_good.setWeight(0.5);
    const base_groups = [_]*AnimationGroup{ ag_bad, ag_good };
    evaluateSkeleton(skel, &base_groups, &.{});
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), skel.bones[0].local_position.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 45.0), skel.bones[0].local_rotation.toEulerDeg().y, 0.5);

    // Bad clip alone takes the fast path (single full-weight clip ->
    // applyAtTime), which preserves the current pose when channels yield
    // null. Seed a sentinel so preserve vs. bind-reset is distinguishable.
    skel.bones[0].local_position = Vec3.new(7.0, 8.0, 9.0);
    skel.bones[0].local_rotation = q1;
    ag_bad.setWeight(1.0);
    const single = [_]*AnimationGroup{ag_bad};
    evaluateSkeleton(skel, &single, &.{});
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), skel.bones[0].local_position.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), skel.bones[0].local_position.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), skel.bones[0].local_position.z, 1e-6);
    try std.testing.expectApproxEqAbs(q1.x, skel.bones[0].local_rotation.x, 1e-6);
    try std.testing.expectApproxEqAbs(q1.y, skel.bones[0].local_rotation.y, 1e-6);
    try std.testing.expectApproxEqAbs(q1.z, skel.bones[0].local_rotation.z, 1e-6);
    try std.testing.expectApproxEqAbs(q1.w, skel.bones[0].local_rotation.w, 1e-6);
}
