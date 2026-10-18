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
    const track = try allocVec3Track(allocator, 0.0, 4.0, Vec3.zero, Vec3.new(4.0, 8.0, 12.0));

    const channels = try allocator.alloc(AnimationChannel, 1);
    channels[0] = .{
        .bone_index = 0,
        .target_path = .translation,
        .sampler = .{
            .timestamps = track.t,
            .outputs = track.o,
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
    const ag1 = try makeBoneClip(allocator, "walk", Vec3.new(0.0, 0.0, 10.0));
    defer ag1.deinit();
    ag1.play(true);

    // Clip 2: Run (pos: 0, 0, 20)
    const ag2 = try makeBoneClip(allocator, "run", Vec3.new(0.0, 0.0, 20.0));
    defer ag2.deinit();
    ag2.play(true);

    // Test 50/50 blend: pos should be 15.0
    ag1.setWeight(0.5);
    ag2.setWeight(0.5);
    const base_groups = [_]*AnimationGroup{ ag1, ag2 };
    evaluateSkeleton(skel, &base_groups, &.{});
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), skel.bones[0].local_position.z, 1e-4);

    // Test Additive: Add layer (pos: 0, 5, 0)
    const ag3 = try makeBoneClip(allocator, "jump_add", Vec3.new(0.0, 5.0, 0.0));
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

    const ag1 = try makeBoneClip(allocator, "clip1", Vec3.zero);
    defer ag1.deinit();

    const ag2 = try makeBoneClip(allocator, "clip2", Vec3.zero);
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

// Two-key vec3 track (t0 -> from, t1 -> to); buffers transfer to the
// AnimationGroup that adopts them via addNodeChannel.
fn allocVec3Track(allocator: std.mem.Allocator, t0: f32, t1: f32, from: Vec3, to: Vec3) !struct { t: []f32, o: []f32 } {
    const t = try allocator.alloc(f32, 2);
    errdefer allocator.free(t);
    const o = try allocator.alloc(f32, 6);
    errdefer allocator.free(o);
    t[0] = t0;
    t[1] = t1;
    o[0] = from.x;
    o[1] = from.y;
    o[2] = from.z;
    o[3] = to.x;
    o[4] = to.y;
    o[5] = to.z;
    return .{ .t = t, .o = o };
}

// Morph-weights track copied from literals; buffers transfer to the
// AnimationGroup that adopts them via addNodeChannel.
fn allocWeightsTrack(allocator: std.mem.Allocator, times_vals: []const f32, outs_vals: []const f32) !struct { t: []f32, o: []f32 } {
    const t = try allocator.dupe(f32, times_vals);
    errdefer allocator.free(t);
    const o = try allocator.dupe(f32, outs_vals);
    return .{ .t = t, .o = o };
}

// One-bone translation clip; all buffers transfer to the returned group.
fn makeBoneClip(allocator: std.mem.Allocator, name: []const u8, pos: Vec3) !*AnimationGroup {
    const times = try allocator.alloc(f32, 1);
    errdefer allocator.free(times);
    times[0] = 0.0;
    const outs = try allocator.alloc(f32, 3);
    errdefer allocator.free(outs);
    outs[0] = pos.x;
    outs[1] = pos.y;
    outs[2] = pos.z;
    const channels = try allocator.alloc(AnimationChannel, 1);
    errdefer allocator.free(channels);
    channels[0] = .{ .bone_index = 0, .target_path = .translation, .sampler = .{ .timestamps = times, .outputs = outs } };
    return AnimationGroup.init(allocator, name, channels, 1.0);
}

// One-bone translation + rotation clip (1 key each); all buffers transfer
// to the returned group.
fn makeBoneClipTR(allocator: std.mem.Allocator, name: []const u8, pos: Vec3, rot: Quat) !*AnimationGroup {
    const t_times = try allocator.alloc(f32, 1);
    errdefer allocator.free(t_times);
    t_times[0] = 0.0;
    const t_out = try allocator.alloc(f32, 3);
    errdefer allocator.free(t_out);
    t_out[0] = pos.x;
    t_out[1] = pos.y;
    t_out[2] = pos.z;
    const r_times = try allocator.alloc(f32, 1);
    errdefer allocator.free(r_times);
    r_times[0] = 0.0;
    const r_out = try allocator.alloc(f32, 4);
    errdefer allocator.free(r_out);
    r_out[0] = rot.x;
    r_out[1] = rot.y;
    r_out[2] = rot.z;
    r_out[3] = rot.w;
    const channels = try allocator.alloc(AnimationChannel, 2);
    errdefer allocator.free(channels);
    channels[0] = .{ .bone_index = 0, .target_path = .translation, .sampler = .{ .timestamps = t_times, .outputs = t_out } };
    channels[1] = .{ .bone_index = 0, .target_path = .rotation, .sampler = .{ .timestamps = r_times, .outputs = r_out } };
    return AnimationGroup.init(allocator, name, channels, 1.0);
}

// Node group preloaded with events for the event-firing tests.
fn makeEventGroup(allocator: std.mem.Allocator, name: []const u8, events: []const AnimationEvent) !*AnimationGroup {
    const ag = try makeNodeGroup(allocator, name, 2.0);
    errdefer ag.deinit();
    try ag.setEvents(allocator, events);
    return ag;
}

// Truncated LINEAR translation (2 keys, 1 frame of 99s) + truncated STEP
// rotation (2 keys, 1 identity frame): every sampler must skip these
// tracks, so the shared fixture pins the guard values in one place.
fn allocTruncatedBonePair(allocator: std.mem.Allocator) ![]AnimationChannel {
    const t_times = try allocator.alloc(f32, 2);
    errdefer allocator.free(t_times);
    t_times[0] = 0.0;
    t_times[1] = 1.0;
    const t_out = try allocator.alloc(f32, 3);
    errdefer allocator.free(t_out);
    t_out[0] = 99.0;
    t_out[1] = 99.0;
    t_out[2] = 99.0;
    const r_times = try allocator.alloc(f32, 2);
    errdefer allocator.free(r_times);
    r_times[0] = 0.0;
    r_times[1] = 1.0;
    const r_out = try allocator.alloc(f32, 4);
    errdefer allocator.free(r_out);
    r_out[0] = 0.0;
    r_out[1] = 0.0;
    r_out[2] = 0.0;
    r_out[3] = 1.0;
    const channels = try allocator.alloc(AnimationChannel, 2);
    errdefer allocator.free(channels);
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
    return channels;
}

test "NodeChannel translation applies over time with loop" {
    const allocator = std.testing.allocator;
    var node = TestNode{};

    const track = try allocVec3Track(allocator, 0.0, 2.0, Vec3.zero, Vec3.new(10.0, 0.0, 0.0));

    const ag = try makeNodeGroup(allocator, "node_move", 2.0);
    defer ag.deinit();
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, false);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = track.t, .outputs = track.o, .interpolation = .linear },
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

    const step_track = try allocVec3Track(allocator, 0.0, 1.0, Vec3.zero, Vec3.new(10.0, 0.0, 0.0));
    const ag_step = try makeNodeGroup(allocator, "node_step", 1.0);
    defer ag_step.deinit();
    const step_target = try ag_step.bindNodeTarget(&step_node.position, &step_node.rotation, &step_node.scaling, false);
    try ag_step.addNodeChannel(.{
        .target = step_target,
        .target_path = .translation,
        .sampler = .{ .timestamps = step_track.t, .outputs = step_track.o, .interpolation = .step },
    });

    const lin_track = try allocVec3Track(allocator, 0.0, 1.0, Vec3.zero, Vec3.new(10.0, 0.0, 0.0));
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

    const track = try allocVec3Track(allocator, 0.0, 1.0, Vec3.zero, Vec3.new(0.0, 0.0, 20.0));
    const ag = try makeNodeGroup(allocator, "node_weight", 1.0);
    defer ag.deinit();
    // Rest pose (0, 10, 0) is captured at bind time.
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, false);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = track.t, .outputs = track.o, .interpolation = .linear },
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

    const track = try allocVec3Track(allocator, 0.0, 1.0, Vec3.zero, Vec3.new(10.0, 0.0, 0.0));
    const ag = try makeNodeGroup(allocator, "node_ease", 1.0);
    defer ag.deinit();
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, false);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = track.t, .outputs = track.o, .interpolation = .linear },
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

    const track = try allocWeightsTrack(allocator, &.{ 0.0, 1.0 }, &.{ 0.0, 0.0, 1.0, 0.5 });
    const ag = try makeNodeGroup(allocator, "morph_weights", 1.0);
    defer ag.deinit();
    const target = try ag.bindMorphTarget(&weights, &dirty);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .weights,
        .sampler = .{ .timestamps = track.t, .outputs = track.o, .interpolation = .linear },
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
    const track = try allocWeightsTrack(allocator, &.{ 0.0, 1.0 }, &.{ 0.9, 0.9 });

    const ag = try makeNodeGroup(allocator, "morph_guard", 1.0);
    defer ag.deinit();
    const target = try ag.bindMorphTarget(&weights, &dirty);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .weights,
        .sampler = .{ .timestamps = track.t, .outputs = track.o, .interpolation = .linear },
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

    // Rows: sample time -> expected value. Endpoints hit keyframes exactly
    // (in/out tangents of the far keys are ignored), zero tangents on both
    // sides of the segment give the smoothstep midpoint, and out-of-range
    // times clamp to the edge keys.
    const Case = struct { t: f32, want: Vec3 };
    const cases = [_]Case{
        .{ .t = 0.0, .want = Vec3.new(0.0, 0.0, 0.0) },
        .{ .t = 1.0, .want = Vec3.new(1.0, 2.0, 3.0) },
        .{ .t = 0.5, .want = Vec3.new(0.5, 1.0, 1.5) },
        .{ .t = -5.0, .want = Vec3.new(0.0, 0.0, 0.0) },
        .{ .t = 99.0, .want = Vec3.new(1.0, 2.0, 3.0) },
    };
    for (cases) |c| {
        const v = sampler.sampleVec3(c.t);
        try std.testing.expectApproxEqAbs(c.want.x, v.x, 1e-5);
        try std.testing.expectApproxEqAbs(c.want.y, v.y, 1e-5);
        try std.testing.expectApproxEqAbs(c.want.z, v.z, 1e-5);
    }
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

    // n=2: k0 in=[9,9] val=[0,0] out=[3,0]; k1 in=[0,0] val=[1,0.5] out=[9,9].
    const track = try allocWeightsTrack(allocator, &.{ 0.0, 1.0 }, &.{ 9.0, 9.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.5, 9.0, 9.0 });

    const ag = try makeNodeGroup(allocator, "morph_cubic", 1.0);
    defer ag.deinit();
    const target = try ag.bindMorphTarget(&weights, &dirty);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .weights,
        .sampler = .{ .timestamps = track.t, .outputs = track.o, .interpolation = .cubic_spline },
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

test "Truncated LINEAR/STEP sampler degrades gracefully instead of crashing" {
    const times = [_]f32{ 0.0, 1.0 };
    // 2 keys but only 1 frame of vec3 outputs: every sample clamps to the
    // single usable frame (previously an index-out-of-bounds panic).
    const tiny = [_]f32{ 7.0, 8.0, 9.0 };
    const lin = AnimationSampler{
        .timestamps = &times,
        .outputs = &tiny,
        .interpolation = .linear,
    };
    for ([_]f32{ -1.0, 0.0, 0.5, 1.0, 2.0 }) |t| {
        const v = lin.sampleVec3(t);
        try std.testing.expectApproxEqAbs(@as(f32, 7.0), v.x, 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 8.0), v.y, 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 9.0), v.z, 1e-6);
    }

    // STEP holds the same single usable keyframe.
    const stp = AnimationSampler{
        .timestamps = &times,
        .outputs = &tiny,
        .interpolation = .step,
    };
    const sv = stp.sampleVec3(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), sv.x, 1e-6);

    // 2 keys but only 1 frame of quat outputs: clamps to identity frame.
    const qtiny = [_]f32{ 0.0, 0.0, 0.0, 1.0 };
    const lq = AnimationSampler{
        .timestamps = &times,
        .outputs = &qtiny,
        .interpolation = .linear,
    };
    for ([_]f32{ -1.0, 0.5, 2.0 }) |t| {
        const q = lq.sampleQuat(t);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), q.x, 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), q.w, 1e-6);
    }

    // Zero usable frames: vec3 degrades to zero, quat to identity.
    const empty = AnimationSampler{
        .timestamps = &times,
        .outputs = &.{},
        .interpolation = .linear,
    };
    const ev = empty.sampleVec3(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ev.x, 1e-6);
    const eq = empty.sampleQuat(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), eq.w, 1e-6);
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
    const ag = try makeEventGroup(allocator, "ev_once", &.{.{ .time = 1.0, .name = "hit" }});
    defer ag.deinit();
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
    const ag = try makeEventGroup(allocator, "ev_loop", &.{
        .{ .time = 0.25, .name = "wrapped" },
        .{ .time = 1.5, .name = "end" },
        .{ .time = 0.0, .name = "start" },
    });
    defer ag.deinit();

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
    const ag = try makeEventGroup(allocator, "ev_back", &.{.{ .time = 1.0, .name = "hit" }});
    defer ag.deinit();

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
    const ag = try makeEventGroup(allocator, "ev_edge", &.{.{ .time = 2.0, .name = "end" }});
    defer ag.deinit();

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
    const ag = try makeEventGroup(allocator, "ev_nofire", &.{
        .{ .time = 0.5, .name = "early" },
        .{ .time = 1.0, .name = "late" },
    });
    defer ag.deinit();

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

    // 2 keys but only 1 frame of outputs: truncated LINEAR translation
    // and truncated STEP rotation (shared guard fixture, 99s + identity).
    const channels = try allocTruncatedBonePair(allocator);

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
    const bad_ch = try allocTruncatedBonePair(allocator);
    const ag_bad = try AnimationGroup.init(allocator, "bad", bad_ch, 1.0);
    defer ag_bad.deinit();
    ag_bad.play(true);

    // Good clip: valid translation (z=20) + valid 90deg Y rotation.
    const q1 = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));
    const ag_good = try makeBoneClipTR(allocator, "good", Vec3.new(0.0, 0.0, 20.0), q1);
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

// One track's heap buffers (times + outputs); frees times when outputs fails.
fn invAllocTrack(allocator: std.mem.Allocator, n_out: usize) !struct { t: []f32, o: []f32 } {
    const t = try allocator.alloc(f32, 2);
    errdefer allocator.free(t);
    const o = try allocator.alloc(f32, n_out);
    return .{ .t = t, .o = o };
}

// One bone clip driving every bone (translation + rotation, 2 linear keys).
// Buffers are per-channel heap so AnimationGroup.deinit owns them.
fn invMakeBoneClip(allocator: std.mem.Allocator, skel: *Skeleton, seed: f32, weight: f32, additive: bool) !*AnimationGroup {
    const n_bones = skel.bones.len;
    const channels = try allocator.alloc(AnimationChannel, n_bones * 2);
    var filled: usize = 0;
    errdefer {
        for (channels[0..filled]) |ch| {
            allocator.free(ch.sampler.timestamps);
            allocator.free(ch.sampler.outputs);
        }
        allocator.free(channels);
    }
    for (0..n_bones) |b| {
        const fb: f32 = @floatFromInt(b);
        const tr = try invAllocTrack(allocator, 6);
        tr.t[0] = 0.0;
        tr.t[1] = 1.0;
        tr.o[0] = fb + seed;
        tr.o[1] = seed;
        tr.o[2] = fb * seed;
        tr.o[3] = fb + seed + 1.0;
        tr.o[4] = seed + 1.0;
        tr.o[5] = fb * seed + 1.0;
        const rr = try invAllocTrack(allocator, 8);
        rr.t[0] = 0.0;
        rr.t[1] = 1.0;
        const q = Quat.fromEulerDeg(Vec3.new(seed * 10.0 + fb * 5.0, seed * 3.0, 0.0));
        rr.o[0] = 0.0;
        rr.o[1] = 0.0;
        rr.o[2] = 0.0;
        rr.o[3] = 1.0;
        rr.o[4] = q.x;
        rr.o[5] = q.y;
        rr.o[6] = q.z;
        rr.o[7] = q.w;
        channels[b * 2 + 0] = .{ .bone_index = b, .target_path = .translation, .sampler = .{ .timestamps = tr.t, .outputs = tr.o, .interpolation = .linear } };
        channels[b * 2 + 1] = .{ .bone_index = b, .target_path = .rotation, .sampler = .{ .timestamps = rr.t, .outputs = rr.o, .interpolation = .linear } };
        filled += 2;
    }
    const ag = try AnimationGroup.init(allocator, "inv_clip", channels, 1.0);
    ag.skeleton = skel;
    ag.play(true);
    ag.setWeight(weight);
    ag.setAdditive(additive);
    return ag;
}

const InvRig = struct {
    skels: []*Skeleton,
    groups: std.ArrayListUnmanaged(*AnimationGroup),
    nodes: [8]TestNode,

    fn deinit(r: *InvRig, allocator: std.mem.Allocator) void {
        for (r.groups.items) |ag| ag.deinit();
        r.groups.deinit(allocator);
        for (r.skels) |sk| sk.deinit();
        allocator.free(r.skels);
    }
};

// 8 skeletons x 4 bones covering every eval path: fast path (0,1),
// two-clip blend (2,3), base + additive (4), fast path WITH node channels
// (5: serial-exclusion path), blend WITH node channels (6), inactive (7).
//
// `out` must stay at a stable address for the rig's lifetime: node targets
// hold raw pointers into out.nodes (same discipline as Mesh bindings).
fn invBuildRig(allocator: std.mem.Allocator, out: *InvRig) !void {
    out.* = InvRig{
        .skels = try allocator.alloc(*Skeleton, 8),
        .groups = .empty,
        .nodes = @as([8]TestNode, @splat(.{})),
    };
    errdefer allocator.free(out.skels);
    var built_skel: usize = 0;
    errdefer {
        for (out.groups.items) |ag| ag.deinit();
        out.groups.deinit(allocator);
        for (out.skels[0..built_skel]) |sk| sk.deinit();
    }
    for (0..8) |s| {
        const skel = try Skeleton.init(allocator, 4);
        out.skels[s] = skel;
        built_skel += 1;
        for (0..4) |b| {
            const fb: f32 = @floatFromInt(b);
            skel.bones[b].bind_position = Vec3.new(fb, @as(f32, @floatFromInt(s)), 0);
            skel.bones[b].bind_rotation = Quat.identity;
            skel.bones[b].bind_scale = Vec3.one;
            if (b > 0) skel.bones[b].parent_index = b - 1;
        }
        skel.resetToBindPose();
        const fs: f32 = @floatFromInt(s);
        switch (s) {
            0, 1 => {
                try out.groups.append(allocator, try invMakeBoneClip(allocator, skel, fs, 1.0, false));
            },
            2, 3 => {
                try out.groups.append(allocator, try invMakeBoneClip(allocator, skel, fs, 0.5, false));
                try out.groups.append(allocator, try invMakeBoneClip(allocator, skel, fs + 10.0, 0.5, false));
            },
            4 => {
                try out.groups.append(allocator, try invMakeBoneClip(allocator, skel, fs, 0.6, false));
                try out.groups.append(allocator, try invMakeBoneClip(allocator, skel, fs + 20.0, 0.4, true));
            },
            5 => {
                const ag = try invMakeBoneClip(allocator, skel, fs, 1.0, false);
                try out.groups.append(allocator, ag);
                const target = try ag.bindNodeTarget(&out.nodes[s].position, &out.nodes[s].rotation, &out.nodes[s].scaling, false);
                const tr = try invAllocTrack(allocator, 6);
                tr.t[0] = 0.0;
                tr.t[1] = 1.0;
                tr.o[0] = 0.0;
                tr.o[1] = 0.0;
                tr.o[2] = 0.0;
                tr.o[3] = 5.0;
                tr.o[4] = 0.0;
                tr.o[5] = 0.0;
                try ag.addNodeChannel(.{ .target = target, .target_path = .translation, .sampler = .{ .timestamps = tr.t, .outputs = tr.o, .interpolation = .linear } });
            },
            6 => {
                const ag_a = try invMakeBoneClip(allocator, skel, fs, 0.5, false);
                try out.groups.append(allocator, ag_a);
                const ag_b = try invMakeBoneClip(allocator, skel, fs + 30.0, 0.5, false);
                try out.groups.append(allocator, ag_b);
                const target = try ag_b.bindNodeTarget(&out.nodes[s].position, &out.nodes[s].rotation, &out.nodes[s].scaling, false);
                const tr = try invAllocTrack(allocator, 6);
                tr.t[0] = 0.0;
                tr.t[1] = 1.0;
                tr.o[0] = 0.0;
                tr.o[1] = 0.0;
                tr.o[2] = 0.0;
                tr.o[3] = 0.0;
                tr.o[4] = 7.0;
                tr.o[5] = 0.0;
                try ag_b.addNodeChannel(.{ .target = target, .target_path = .translation, .sampler = .{ .timestamps = tr.t, .outputs = tr.o, .interpolation = .linear } });
            },
            else => {},
        }
    }
}

test "updateAnimations serial vs pool is bit-identical (node writers stay serial)" {
    const jobs = @import("../jobs.zig");
    const runtime = @import("../scene/animation_runtime.zig");
    const allocator = std.testing.allocator;

    // The rig (8 skeletons) must clear the parallel threshold.
    try std.testing.expect(runtime.min_skeletons_for_workers <= 8);

    var serial: InvRig = undefined;
    try invBuildRig(allocator, &serial);
    defer serial.deinit(allocator);
    var parallel: InvRig = undefined;
    try invBuildRig(allocator, &parallel);
    defer parallel.deinit(allocator);

    const pool = try jobs.Pool.init(allocator, 2);
    defer pool.deinit();

    // Identical tick sequences from identical start states; the only
    // difference is the execution path (null global = inline, pool = forkJoin).
    for (0..3) |_| runtime.updateAnimations(serial.groups.items, serial.skels, &.{}, 1.0 / 60.0);
    jobs.global = pool;
    defer jobs.global = null;
    for (0..3) |_| runtime.updateAnimations(parallel.groups.items, parallel.skels, &.{}, 1.0 / 60.0);

    for (serial.skels, parallel.skels, 0..) |sk_s, sk_p, i| {
        for (sk_s.bones, sk_p.bones) |b_s, b_p| {
            try std.testing.expectEqual(b_s.local_position, b_p.local_position);
            try std.testing.expectEqual(b_s.local_rotation, b_p.local_rotation);
            try std.testing.expectEqual(b_s.local_scale, b_p.local_scale);
            try std.testing.expectEqual(b_s.model_matrix, b_p.model_matrix);
        }
        for (sk_s.skin_matrices, sk_p.skin_matrices) |m_s, m_p| {
            try std.testing.expectEqual(m_s, m_p);
        }
        try std.testing.expectEqual(sk_s.render_slot.load(.monotonic), sk_p.render_slot.load(.monotonic));
        // Node targets bound by the exclusion-path skeletons (5, 6) must
        // match too: update() wrote them serially on both paths, and the
        // parallel path correctly skipped the duplicate re-apply.
        try std.testing.expectEqual(serial.nodes[i].position, parallel.nodes[i].position);
        try std.testing.expectEqual(serial.nodes[i].rotation, parallel.nodes[i].rotation);
        try std.testing.expectEqual(serial.nodes[i].scaling, parallel.nodes[i].scaling);
    }
}

test "fadeOut completion resets skeleton to bind pose (single clip)" {
    const runtime = @import("../scene/animation_runtime.zig");
    const allocator = std.testing.allocator;
    const skel = try Skeleton.init(allocator, 1);
    defer skel.deinit();
    skel.bones[0].bind_position = Vec3.zero;
    skel.bones[0].bind_rotation = Quat.identity;
    skel.bones[0].bind_scale = Vec3.one;
    skel.resetToBindPose();

    const ag = try makeBoneClip(allocator, "fade Solo", Vec3.new(0.0, 0.0, 10.0));
    defer ag.deinit();
    ag.skeleton = skel;
    ag.play(true);

    var groups = [_]*AnimationGroup{ag};
    var skels = [_]*Skeleton{skel};
    // Drive one frame so the skeleton leaves bind (fast path writes z=10).
    runtime.updateAnimations(&groups, &skels, &.{}, 1.0 / 60.0);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), skel.bones[0].local_position.z, 1e-4);

    // Fade out fully: group stops, skeleton must return to bind like stop().
    // Without the group.zig reset, collectActive finds zero active clips,
    // eval returns early, and the z=10 pose sticks.
    ag.fadeOut(0.5);
    runtime.updateAnimations(&groups, &skels, &.{}, 0.25);
    try std.testing.expect(ag.is_playing);
    runtime.updateAnimations(&groups, &skels, &.{}, 0.25);
    try std.testing.expect(!ag.is_playing);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), skel.bones[0].local_position.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), skel.bones[0].local_position.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), skel.bones[0].local_position.z, 1e-5);
}

test "crossFade source reset does not disturb remaining target" {
    const runtime = @import("../scene/animation_runtime.zig");
    const allocator = std.testing.allocator;
    const skel = try Skeleton.init(allocator, 1);
    defer skel.deinit();
    skel.bones[0].bind_position = Vec3.zero;
    skel.bones[0].bind_rotation = Quat.identity;
    skel.bones[0].bind_scale = Vec3.one;
    skel.resetToBindPose();

    const src = try makeBoneClip(allocator, "fade_src", Vec3.new(0.0, 0.0, 10.0));
    defer src.deinit();
    src.skeleton = skel;
    src.play(true);
    src.setWeight(1.0);

    const dst = try makeBoneClip(allocator, "fade_dst", Vec3.new(0.0, 0.0, 20.0));
    defer dst.deinit();
    dst.skeleton = skel;

    var groups = [_]*AnimationGroup{ src, dst };
    var skels = [_]*Skeleton{skel};
    src.crossFadeTo(dst, 1.0);
    runtime.updateAnimations(&groups, &skels, &.{}, 0.5);
    runtime.updateAnimations(&groups, &skels, &.{}, 0.5);

    // Source stopped at weight 0; target alone drives the skeleton at full
    // weight. The source's bind reset must not leak into the result.
    try std.testing.expect(!src.is_playing);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), src.weight, 1e-4);
    try std.testing.expect(dst.is_playing);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst.weight, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), skel.bones[0].local_position.z, 1e-4);
}

test "play reseats finished endpoint cursors and replays terminal events" {
    const allocator = std.testing.allocator;
    // Forward one-shot with a terminal event. Steps stay below the range so
    // the advance>=range fire-all shortcut cannot mask an empty window.
    const ag = try makeNodeGroup(allocator, "play_replay_fwd", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{.{ .time = 2.0, .name = "end" }});
    ag.play(false);
    ag.update(1.0);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
    ag.update(1.1); // overshoots to, clamps, stops, fires "end"
    try std.testing.expect(!ag.is_playing);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), ag.current_time, 1e-6);
    {
        const f = ag.drainFiredEvents();
        try std.testing.expectEqual(@as(usize, 1), f.len);
        try std.testing.expectEqualStrings("end", f[0]);
    }
    ag.play(false); // cursor at/past to (forward) must reseat to from
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ag.current_time, 1e-6);
    try std.testing.expect(ag.is_playing);
    ag.update(1.0);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
    ag.update(1.1);
    try std.testing.expect(!ag.is_playing);
    {
        const f = ag.drainFiredEvents();
        try std.testing.expectEqual(@as(usize, 1), f.len);
        try std.testing.expectEqualStrings("end", f[0]);
    }

    // Reverse one-shot with a terminal event at from.
    const agr = try makeNodeGroup(allocator, "play_replay_rev", 2.0);
    defer agr.deinit();
    try agr.setEvents(allocator, &.{.{ .time = 0.0, .name = "start" }});
    agr.setSpeed(-1.0);
    agr.play(false); // initial reverse: 0 is at/below from, must start at to
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), agr.current_time, 1e-6);
    agr.update(1.0);
    try std.testing.expectEqual(@as(usize, 0), agr.drainFiredEvents().len);
    agr.update(1.1); // overshoots from, clamps, stops, fires "start"
    try std.testing.expect(!agr.is_playing);
    {
        const f = agr.drainFiredEvents();
        try std.testing.expectEqual(@as(usize, 1), f.len);
        try std.testing.expectEqualStrings("start", f[0]);
    }
    agr.play(false); // cursor at/below from (reverse) must reseat to to
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), agr.current_time, 1e-6);
    agr.update(1.0);
    try std.testing.expectEqual(@as(usize, 0), agr.drainFiredEvents().len);
    agr.update(1.1);
    {
        const f = agr.drainFiredEvents();
        try std.testing.expectEqual(@as(usize, 1), f.len);
        try std.testing.expectEqualStrings("start", f[0]);
    }
}

test "play preserves mid-range pause and seek cursors" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "play_resume", 2.0);
    defer ag.deinit();
    ag.play(true);
    ag.update(0.5);
    ag.pause();
    const paused = ag.current_time;
    try std.testing.expect(paused > 0.0 and paused < 2.0);
    ag.play(true); // in-range resume, must not reseat
    try std.testing.expectApproxEqAbs(paused, ag.current_time, 1e-6);
    ag.goToFrame(1.25);
    ag.play(true);
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), ag.current_time, 1e-6);
    // Reverse in-range resume is preserved too.
    ag.setSpeed(-1.0);
    ag.goToFrame(1.5);
    ag.play(true);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), ag.current_time, 1e-6);
}
