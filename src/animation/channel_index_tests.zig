const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Skeleton = @import("skeleton.zig").Skeleton;
const MAX_BONES = @import("skeleton.zig").MAX_BONES;
const anim = @import("animation.zig");
const AnimationGroup = anim.AnimationGroup;
const AnimationChannel = anim.AnimationChannel;
const AnimationPath = anim.AnimationPath;
const AnimationInterpolation = anim.AnimationInterpolation;
const NodeChannel = anim.NodeChannel;
const evaluateSkeleton = anim.evaluateSkeleton;
const evaluateSkeletonPoseOnly = @import("eval.zig").evaluateSkeletonPoseOnly;

// Parity proof for the per-bone channel route (group.zig): the routed path
// must visit the same channels in the same order with the same guards as
// the legacy scan, so every scenario below asserts exact equality between
// a group sampled before update() (legacy) and after update() (routed).

fn dupF(allocator: std.mem.Allocator, vals: []const f32) ![]f32 {
    const s = try allocator.alloc(f32, vals.len);
    @memcpy(s, vals);
    return s;
}

fn mkCh(
    allocator: std.mem.Allocator,
    bone: usize,
    path: AnimationPath,
    times: []const f32,
    outs: []const f32,
    interp: AnimationInterpolation,
) !AnimationChannel {
    const t = try dupF(allocator, times);
    errdefer allocator.free(t);
    return .{
        .bone_index = bone,
        .target_path = path,
        .sampler = .{
            .timestamps = t,
            .outputs = try dupF(allocator, outs),
            .interpolation = interp,
        },
    };
}

fn freeCh(allocator: std.mem.Allocator, ch: *AnimationChannel) void {
    allocator.free(ch.sampler.timestamps);
    allocator.free(ch.sampler.outputs);
}

const Sample = struct { p: ?Vec3, r: ?Quat, s: ?Vec3 };

fn grab(ag: *const AnimationGroup, bone: usize, t: f32) Sample {
    var out: Sample = .{ .p = null, .r = null, .s = null };
    ag.sampleBoneAtTime(bone, t, &out.p, &out.r, &out.s);
    return out;
}

fn grabIndexed(ag: *const AnimationGroup, bone: usize, t: f32) Sample {
    var out: Sample = .{ .p = null, .r = null, .s = null };
    ag.sampleBoneIndexed(bone, t, &out.p, &out.r, &out.s);
    return out;
}

fn expectSampleEqual(a: Sample, b: Sample) !void {
    try expectOptVecEqual(a.p, b.p);
    try expectOptQuatEqual(a.r, b.r);
    try expectOptVecEqual(a.s, b.s);
}

fn expectOptVecEqual(a: ?Vec3, b: ?Vec3) !void {
    if (a == null or b == null) {
        try std.testing.expect(a == null and b == null);
        return;
    }
    try std.testing.expectEqual(a.?.x, b.?.x);
    try std.testing.expectEqual(a.?.y, b.?.y);
    try std.testing.expectEqual(a.?.z, b.?.z);
}

fn expectOptQuatEqual(a: ?Quat, b: ?Quat) !void {
    if (a == null or b == null) {
        try std.testing.expect(a == null and b == null);
        return;
    }
    try std.testing.expectEqual(a.?.x, b.?.x);
    try std.testing.expectEqual(a.?.y, b.?.y);
    try std.testing.expectEqual(a.?.z, b.?.z);
    try std.testing.expectEqual(a.?.w, b.?.w);
}

fn expectSkelEqual(a: *Skeleton, b: *Skeleton) !void {
    try std.testing.expectEqual(a.bones.len, b.bones.len);
    for (a.bones, b.bones) |ba, bb| {
        try std.testing.expectEqual(ba.local_position.x, bb.local_position.x);
        try std.testing.expectEqual(ba.local_position.y, bb.local_position.y);
        try std.testing.expectEqual(ba.local_position.z, bb.local_position.z);
        try std.testing.expectEqual(ba.local_rotation.x, bb.local_rotation.x);
        try std.testing.expectEqual(ba.local_rotation.y, bb.local_rotation.y);
        try std.testing.expectEqual(ba.local_rotation.z, bb.local_rotation.z);
        try std.testing.expectEqual(ba.local_rotation.w, bb.local_rotation.w);
        try std.testing.expectEqual(ba.local_scale.x, bb.local_scale.x);
        try std.testing.expectEqual(ba.local_scale.y, bb.local_scale.y);
        try std.testing.expectEqual(ba.local_scale.z, bb.local_scale.z);
    }
}

fn bindDefaults(skel: *Skeleton) void {
    for (skel.bones) |*bn| {
        bn.bind_position = Vec3.zero;
        bn.bind_rotation = Quat.identity;
        bn.bind_scale = Vec3.one;
    }
}

// A 90-degree pitch quaternion (gimbal pole for the euler round-trip).
fn pitch90() Quat {
    return Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));
}

test "channel index: routed vs legacy sample parity" {
    const allocator = std.testing.allocator;
    const skel = try Skeleton.init(allocator, 2);
    defer skel.deinit();
    bindDefaults(skel);

    const q90 = pitch90();
    const channels = try allocator.alloc(AnimationChannel, 7);
    var filled: usize = 0;
    errdefer {
        for (channels[0..filled]) |*ch| freeCh(allocator, ch);
        allocator.free(channels);
    }
    channels[0] = try mkCh(allocator, 0, .translation, &.{ 0.0, 1.0 }, &.{ 0, 0, 0, 10, 0, 0 }, .linear);
    filled += 1;
    // Duplicate path: LAST VALID wins (z must read 99, not 10).
    channels[1] = try mkCh(allocator, 0, .translation, &.{0.0}, &.{ 0, 0, 99 }, .step);
    filled += 1;
    // Truncated LINEAR scale: skipped, output stays null.
    channels[2] = try mkCh(allocator, 0, .scale, &.{ 0.0, 1.0 }, &.{ 1, 1, 1 }, .linear);
    filled += 1;
    // Zero quat: normalize() falls back to identity.
    channels[3] = try mkCh(allocator, 0, .rotation, &.{0.0}, &.{ 0, 0, 0, 0 }, .linear);
    filled += 1;
    // Truncated CUBICSPLINE: not guarded, sampler fallback applies.
    channels[4] = try mkCh(allocator, 0, .rotation, &.{ 0.0, 1.0 }, &.{ 0, 0, 0, 1 }, .cubic_spline);
    filled += 1;
    // Gimbal-pole rotation on bone 1 (valid step track).
    channels[5] = try mkCh(allocator, 1, .rotation, &.{0.0}, &.{ q90.x, q90.y, q90.z, q90.w }, .step);
    filled += 1;
    channels[6] = try mkCh(allocator, 1, .translation, &.{ 0.0, 1.0 }, &.{ 1, 2, 3, 4, 5, 6 }, .linear);
    filled += 1;

    const ag = try AnimationGroup.init(allocator, "route_parity", channels, 1.0);
    defer ag.deinit();
    ag.skeleton = skel;

    try std.testing.expect(!ag.hasValidChannelIndex());
    var legacy: [3]Sample = undefined;
    const times = [_]f32{ 0.0, 0.37, 1.0 };
    for (times, 0..) |t, i| legacy[i] = grab(ag, 0, t);
    const legacy_b1 = grab(ag, 1, 0.5);
    // Duplicate LAST-valid-wins visible on the legacy path already.
    try std.testing.expectEqual(@as(f32, 99.0), legacy[0].p.?.z);
    try std.testing.expect(legacy[0].s == null);
    // Explicit-legacy entry point agrees with the dispatch pre-index.
    for (times, 0..) |t, i| {
        var p: ?Vec3 = null;
        var r: ?Quat = null;
        var s: ?Vec3 = null;
        ag.sampleBoneAtTimeLegacy(0, t, &p, &r, &s);
        try expectSampleEqual(legacy[i], .{ .p = p, .r = r, .s = s });
    }

    ag.play(true);
    ag.update(0.2);
    try std.testing.expect(ag.hasValidChannelIndex());
    for (times, 0..) |t, i| try expectSampleEqual(legacy[i], grabIndexed(ag, 0, t));
    try expectSampleEqual(legacy_b1, grabIndexed(ag, 1, 0.5));
    // Public dispatch stays legacy (always current) after update too.
    for (times, 0..) |t, i| try expectSampleEqual(legacy[i], grab(ag, 0, t));
}

test "channel index: oversized bone falls back to legacy scan" {
    const allocator = std.testing.allocator;
    try std.testing.expect(MAX_BONES <= 70);
    const skel = try Skeleton.init(allocator, 72);
    defer skel.deinit();
    bindDefaults(skel);

    const channels = try allocator.alloc(AnimationChannel, 2);
    var filled: usize = 0;
    errdefer {
        for (channels[0..filled]) |*ch| freeCh(allocator, ch);
        allocator.free(channels);
    }
    channels[0] = try mkCh(allocator, 70, .translation, &.{ 0.0, 1.0 }, &.{ 7, 8, 9, 1, 2, 3 }, .linear);
    filled += 1;
    channels[1] = try mkCh(allocator, 3, .translation, &.{0.0}, &.{ 5, 5, 5 }, .step);
    filled += 1;

    const ag = try AnimationGroup.init(allocator, "route_overflow", channels, 1.0);
    defer ag.deinit();
    ag.skeleton = skel;

    const before = grab(ag, 70, 0.5);
    try std.testing.expectEqual(@as(f32, 4.0), before.p.?.x);
    ag.play(true);
    ag.update(0.1);
    try std.testing.expect(ag.hasValidChannelIndex());
    try expectSampleEqual(before, grab(ag, 70, 0.5));
    // Trusted leaf on the oversized bone uses its internal legacy fallback.
    try expectSampleEqual(before, grabIndexed(ag, 70, 0.5));
    // Routed bone below the cap on the same group still matches legacy.
    var p: ?Vec3 = null;
    var r: ?Quat = null;
    var s: ?Vec3 = null;
    ag.sampleBoneAtTimeLegacy(3, 0.0, &p, &r, &s);
    try expectSampleEqual(.{ .p = p, .r = r, .s = s }, grabIndexed(ag, 3, 0.0));
}

test "channel index: slice replacement rebuilds, never stale" {
    const allocator = std.testing.allocator;
    const skel = try Skeleton.init(allocator, 1);
    defer skel.deinit();
    bindDefaults(skel);

    const ch_old = try allocator.alloc(AnimationChannel, 1);
    ch_old[0] = try mkCh(allocator, 0, .translation, &.{0.0}, &.{ 1, 1, 1 }, .step);
    const ag = try AnimationGroup.init(allocator, "route_replace", ch_old, 1.0);
    defer ag.deinit();
    ag.skeleton = skel;
    ag.play(true);
    ag.update(0.1);
    try std.testing.expect(ag.hasValidChannelIndex());
    try std.testing.expectEqual(@as(f32, 1.0), (grab(ag, 0, 0.0)).p.?.x);

    // Replace the slice (old buffers freed by hand; group takes the new one).
    // Same bone sequence, so the route stays valid AND must serve the new
    // values immediately — even before the next update().
    freeCh(allocator, &ch_old[0]);
    allocator.free(ch_old);
    const ch_new = try allocator.alloc(AnimationChannel, 1);
    var new_filled: usize = 0;
    errdefer {
        for (ch_new[0..new_filled]) |*ch| freeCh(allocator, ch);
        allocator.free(ch_new);
    }
    ch_new[0] = try mkCh(allocator, 0, .translation, &.{0.0}, &.{ 42, 43, 44 }, .step);
    new_filled += 1;
    ag.channels = ch_new;
    try std.testing.expect(ag.hasValidChannelIndex());
    try std.testing.expectEqual(@as(f32, 42.0), (grab(ag, 0, 0.0)).p.?.x);
    try std.testing.expectEqual(@as(f32, 42.0), (grabIndexed(ag, 0, 0.0)).p.?.x);
    ag.update(0.1);
    try std.testing.expect(ag.hasValidChannelIndex());
    try std.testing.expectEqual(@as(f32, 42.0), (grabIndexed(ag, 0, 0.0)).p.?.x);
    // A replacement with a different bone sequence invalidates instead.
    freeCh(allocator, &ch_new[0]);
    allocator.free(ch_new);
    const ch_new2 = try allocator.alloc(AnimationChannel, 1);
    var new2_filled: usize = 0;
    errdefer {
        for (ch_new2[0..new2_filled]) |*ch| freeCh(allocator, ch);
        allocator.free(ch_new2);
    }
    ch_new2[0] = try mkCh(allocator, 0, .translation, &.{0.0}, &.{ 7, 7, 7 }, .step);
    new2_filled += 1;
    ag.channels = ch_new2;
    try std.testing.expect(ag.hasValidChannelIndex());
    ag.channels[0].bone_index = 99; // same ptr/len in-place move
    try std.testing.expect(!ag.hasValidChannelIndex());
    try std.testing.expect((grab(ag, 0, 0.0)).p == null);
    try std.testing.expectEqual(@as(f32, 7.0), (grab(ag, 99, 0.0)).p.?.x);
}

fn makeCrossfadeClip(allocator: std.mem.Allocator, n_bones: usize, seed: f32) !*AnimationGroup {
    const channels = try allocator.alloc(AnimationChannel, n_bones * 3);
    var filled: usize = 0;
    errdefer {
        for (channels[0..filled]) |*ch| {
            allocator.free(ch.sampler.timestamps);
            allocator.free(ch.sampler.outputs);
        }
        allocator.free(channels);
    }
    for (0..n_bones) |b| {
        const fb: f32 = @floatFromInt(b);
        channels[b * 3 + 0] = try mkCh(allocator, b, .translation, &.{ 0.0, 1.0 }, &.{ fb + seed, seed, fb * seed, fb + seed + 1, seed + 1, fb * seed + 1 }, .linear);
        filled += 1;
        const q = Quat.fromEulerDeg(Vec3.new(fb * 10.0 + seed, seed * 5.0, 0.0));
        channels[b * 3 + 1] = try mkCh(allocator, b, .rotation, &.{ 0.0, 1.0 }, &.{ q.x, q.y, q.z, q.w, q.x, q.y, q.z, q.w }, .linear);
        filled += 1;
        channels[b * 3 + 2] = try mkCh(allocator, b, .scale, &.{0.0}, &.{ 1, 1, 1 }, .step);
        filled += 1;
    }
    return AnimationGroup.init(allocator, "xfade", channels, 1.0);
}

test "channel index: 2-clip crossfade evaluate parity" {
    const allocator = std.testing.allocator;
    const n_bones = 4;
    const skel_l = try Skeleton.init(allocator, n_bones);
    defer skel_l.deinit();
    const skel_r = try Skeleton.init(allocator, n_bones);
    defer skel_r.deinit();
    bindDefaults(skel_l);
    bindDefaults(skel_r);

    const a_l = try makeCrossfadeClip(allocator, n_bones, 1.0);
    defer a_l.deinit();
    const b_l = try makeCrossfadeClip(allocator, n_bones, 7.0);
    defer b_l.deinit();
    const a_r = try makeCrossfadeClip(allocator, n_bones, 1.0);
    defer a_r.deinit();
    const b_r = try makeCrossfadeClip(allocator, n_bones, 7.0);
    defer b_r.deinit();

    for ([_]*AnimationGroup{ a_l, b_l, a_r, b_r }) |ag| ag.play(true);
    a_l.setWeight(0.35);
    b_l.setWeight(0.65);
    a_r.setWeight(0.35);
    b_r.setWeight(0.65);
    // Legacy rig: never updated, route stays invalid.
    try std.testing.expect(!a_l.hasValidChannelIndex());
    // Indexed rig: update builds the route (then reseat time).
    a_r.update(0.2);
    b_r.update(0.2);
    try std.testing.expect(a_r.hasValidChannelIndex());
    try std.testing.expect(b_r.hasValidChannelIndex());
    a_r.current_time = a_l.current_time;
    b_r.current_time = b_l.current_time;

    const base_l = [_]*AnimationGroup{ a_l, b_l };
    const base_r = [_]*AnimationGroup{ a_r, b_r };
    evaluateSkeleton(skel_l, &base_l, &.{});
    evaluateSkeleton(skel_r, &base_r, &.{});
    try expectSkelEqual(skel_l, skel_r);
}

test "channel index: 3-clip plus additive evaluate parity, pose-only match" {
    const allocator = std.testing.allocator;
    const n_bones = 6;
    const skel_l = try Skeleton.init(allocator, n_bones);
    defer skel_l.deinit();
    const skel_r = try Skeleton.init(allocator, n_bones);
    defer skel_r.deinit();
    const skel_p = try Skeleton.init(allocator, n_bones);
    defer skel_p.deinit();
    bindDefaults(skel_l);
    bindDefaults(skel_r);
    bindDefaults(skel_p);

    var base_l: [3]*AnimationGroup = undefined;
    var base_r: [3]*AnimationGroup = undefined;
    const seeds = [_]f32{ 1.0, 7.0, 13.0 };
    for (seeds, 0..) |sd, i| {
        base_l[i] = try makeCrossfadeClip(allocator, n_bones, sd);
        errdefer base_l[i].deinit();
        base_r[i] = try makeCrossfadeClip(allocator, n_bones, sd);
        errdefer base_r[i].deinit();
    }
    defer for (base_l) |ag| ag.deinit();
    defer for (base_r) |ag| ag.deinit();
    const add_l = try makeCrossfadeClip(allocator, n_bones, 29.0);
    defer add_l.deinit();
    const add_r = try makeCrossfadeClip(allocator, n_bones, 29.0);
    defer add_r.deinit();

    const weights = [_]f32{ 0.5, 0.3, 0.2 };
    for (base_l, base_r, weights) |l, r, w| {
        l.play(true);
        r.play(true);
        l.setWeight(w);
        r.setWeight(w);
    }
    add_l.play(true);
    add_r.play(true);
    add_l.setAdditive(true);
    add_r.setAdditive(true);
    add_l.setWeight(0.4);
    add_r.setWeight(0.4);

    for (base_r) |ag| ag.update(0.15);
    add_r.update(0.15);
    for (base_r, base_l) |r, l| r.current_time = l.current_time;
    add_r.current_time = add_l.current_time;

    const add_l_s = [_]*AnimationGroup{add_l};
    const add_r_s = [_]*AnimationGroup{add_r};
    evaluateSkeleton(skel_l, &base_l, &add_l_s);
    evaluateSkeleton(skel_r, &base_r, &add_r_s);
    try expectSkelEqual(skel_l, skel_r);
    // Pose-only path over the indexed rig matches the full eval exactly.
    evaluateSkeletonPoseOnly(skel_p, &base_r, &add_r_s);
    try expectSkelEqual(skel_r, skel_p);
}

test "channel index: node rotation euler sane, update allocates once" {
    const allocator = std.testing.allocator;
    // --- euler sanity (normalize cleanup must not change values) ---
    var pos = Vec3.zero;
    var rot = Vec3.zero;
    var scl = Vec3.one;
    const ch0 = try allocator.alloc(AnimationChannel, 0);
    const ag = try AnimationGroup.init(allocator, "node_euler", ch0, 1.0);
    defer ag.deinit();
    const target = try ag.bindNodeTarget(&pos, &rot, &scl, false);
    const q0 = Quat.identity;
    const q90 = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));
    const nt = try dupF(allocator, &.{ 0.0, 1.0 });
    const no = try dupF(allocator, &.{ q0.x, q0.y, q0.z, q0.w, q90.x, q90.y, q90.z, q90.w });
    const node_ch = NodeChannel{
        .target = target,
        .target_path = .rotation,
        .sampler = .{ .timestamps = nt, .outputs = no, .interpolation = .linear },
    };
    errdefer {
        allocator.free(node_ch.sampler.timestamps);
        allocator.free(node_ch.sampler.outputs);
    }
    try ag.addNodeChannel(node_ch);
    ag.play(true);
    ag.goToFrame(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rot.x, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 45.0), rot.y, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rot.z, 0.5);
    ag.goToFrame(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), rot.y, 0.5);

    // Degenerate input (F2): a zero-quat track normalizes to identity, so
    // toEulerDeg must yield a stable zero euler, never NaN — with or without
    // the removed outer normalize (toEulerDeg normalizes unconditionally).
    var zpos = Vec3.zero;
    var zrot = Vec3.zero;
    var zscl = Vec3.one;
    const zch = try allocator.alloc(AnimationChannel, 0);
    const zg = try AnimationGroup.init(allocator, "node_euler_zero", zch, 1.0);
    defer zg.deinit();
    const ztarget = try zg.bindNodeTarget(&zpos, &zrot, &zscl, false);
    const zt = try dupF(allocator, &.{ 0.0, 1.0 });
    errdefer allocator.free(zt);
    const zo = try dupF(allocator, &.{ 0, 0, 0, 0, 0, 0, 0, 0 });
    errdefer allocator.free(zo);
    try zg.addNodeChannel(.{
        .target = ztarget,
        .target_path = .rotation,
        .sampler = .{ .timestamps = zt, .outputs = zo, .interpolation = .linear },
    });
    zg.play(true);
    zg.goToFrame(0.7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), zrot.x, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), zrot.y, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), zrot.z, 0.5);

    // --- warmup churn: route build allocates once, then nothing ---
    const skel = try Skeleton.init(allocator, 2);
    defer skel.deinit();
    bindDefaults(skel);
    const ch2 = try allocator.alloc(AnimationChannel, 4);
    var filled: usize = 0;
    errdefer {
        for (ch2[0..filled]) |*ch| freeCh(allocator, ch);
        allocator.free(ch2);
    }
    ch2[0] = try mkCh(allocator, 0, .translation, &.{0.0}, &.{ 1, 2, 3 }, .step);
    filled += 1;
    ch2[1] = try mkCh(allocator, 1, .translation, &.{0.0}, &.{ 4, 5, 6 }, .step);
    filled += 1;
    ch2[2] = try mkCh(allocator, 0, .rotation, &.{0.0}, &.{ 0, 0, 0, 1 }, .step);
    filled += 1;
    ch2[3] = try mkCh(allocator, 1, .rotation, &.{0.0}, &.{ 0, 0, 0, 1 }, .step);
    filled += 1;
    const ag2 = try AnimationGroup.init(allocator, "churn", ch2, 1.0);
    defer ag2.deinit();
    ag2.skeleton = skel;
    ag2.play(true);

    var counter = CountAlloc{ .parent = allocator };
    const callocator = counter.allocator();
    // Swap allocator only for the update-driven route build; group-owned
    // buffers stay on the test allocator so deinit stays balanced.
    const saved = ag2.allocator;
    ag2.allocator = callocator;
    ag2.update(0.1);
    const first_allocs = counter.allocs;
    ag2.update(0.1);
    ag2.update(0.1);
    ag2.update(0.1);
    ag2.allocator = saved;
    try std.testing.expectEqual(first_allocs, counter.allocs);
    // Route build allocates exactly the route + bone-sequence buffers, then
    // steady-state updates allocate nothing.
    try std.testing.expect(first_allocs <= 2);
    // Route buffer frees with the group: no leak (test allocator checks).
    ag2.allocator = saved;
}

const CountAlloc = struct {
    parent: std.mem.Allocator,
    allocs: usize = 0,

    fn allocFn(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountAlloc = @ptrCast(@alignCast(ctx));
        self.allocs += 1;
        return self.parent.vtable.alloc(self.parent.ptr, len, ptr_align, ret_addr);
    }
    fn resizeFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CountAlloc = @ptrCast(@alignCast(ctx));
        return self.parent.vtable.resize(self.parent.ptr, buf, buf_align, new_len, ret_addr);
    }
    fn remapFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CountAlloc = @ptrCast(@alignCast(ctx));
        return self.parent.vtable.remap(self.parent.ptr, buf, buf_align, new_len, ret_addr);
    }
    fn freeFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountAlloc = @ptrCast(@alignCast(ctx));
        return self.parent.vtable.free(self.parent.ptr, buf, buf_align, ret_addr);
    }

    fn allocator(self: *CountAlloc) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }
};

fn makeTwoStepGroup(
    allocator: std.mem.Allocator,
    name: []const u8,
    b0: usize,
    v0: [3]f32,
    b1: usize,
    v1: [3]f32,
) !*AnimationGroup {
    const channels = try allocator.alloc(AnimationChannel, 2);
    var filled: usize = 0;
    errdefer {
        for (channels[0..filled]) |*ch| freeCh(allocator, ch);
        allocator.free(channels);
    }
    channels[0] = try mkCh(allocator, b0, .translation, &.{0.0}, &v0, .step);
    filled += 1;
    channels[1] = try mkCh(allocator, b1, .translation, &.{0.0}, &v1, .step);
    filled += 1;
    return AnimationGroup.init(allocator, name, channels, 1.0);
}

// F1: same-ptr/len in-place bone move. The moved group is updated BEFORE
// the mutation (route built), then read and evaluated WITHOUT re-update:
// every public read must serve the live channels, and the eval must match
// an exact-legacy twin carrying the same final content.
fn checkInPlaceBoneMove(allocator: std.mem.Allocator, from: usize, to: usize) !void {
    const skel = try Skeleton.init(allocator, 8);
    defer skel.deinit();
    const skel2 = try Skeleton.init(allocator, 8);
    defer skel2.deinit();
    bindDefaults(skel);
    bindDefaults(skel2);

    const ag = try makeTwoStepGroup(allocator, "move", from, .{ 5, 5, 5 }, 0, .{ 1, 1, 1 });
    defer ag.deinit();
    ag.skeleton = skel;
    ag.play(true);
    ag.update(0.1);
    try std.testing.expect(ag.hasValidChannelIndex());

    const bl = try makeTwoStepGroup(allocator, "move_legacy", from, .{ 5, 5, 5 }, 0, .{ 1, 1, 1 });
    defer bl.deinit();
    bl.skeleton = skel2;
    bl.play(true);

    // Identical same-ptr/len mutation on both (twin never updates: legacy).
    ag.channels[0].bone_index = to;
    bl.channels[0].bone_index = to;
    try std.testing.expect(!ag.hasValidChannelIndex());
    for ([_]usize{ 0, 2, 5 }) |b| {
        for ([_]f32{ 0.0, 0.7 }) |t| try expectSampleEqual(grab(bl, b, t), grab(ag, b, t));
    }

    // Full 2-clip + pose-only evals without re-update match the twin.
    const c2 = try makeTwoStepGroup(allocator, "move_c2", 3, .{ 9, 9, 9 }, 4, .{ 8, 8, 8 });
    defer c2.deinit();
    const c2l = try makeTwoStepGroup(allocator, "move_c2l", 3, .{ 9, 9, 9 }, 4, .{ 8, 8, 8 });
    defer c2l.deinit();
    for ([_]*AnimationGroup{ ag, bl, c2, c2l }) |g| {
        g.is_playing = true;
        g.current_time = 0.0;
        g.setWeight(0.5);
    }
    const base_m = [_]*AnimationGroup{ ag, c2 };
    const base_l = [_]*AnimationGroup{ bl, c2l };
    evaluateSkeleton(skel, &base_m, &.{});
    evaluateSkeleton(skel2, &base_l, &.{});
    try expectSkelEqual(skel, skel2);
    evaluateSkeletonPoseOnly(skel, &base_m, &.{});
    try expectSkelEqual(skel, skel2);

    // Re-update rebuilds: indexed results rejoin the twin exactly.
    ag.update(0.1);
    try std.testing.expect(ag.hasValidChannelIndex());
    ag.current_time = 0.0;
    for ([_]usize{ 0, 2, 5 }) |b| {
        try expectSampleEqual(grab(bl, b, 0.3), grabIndexed(ag, b, 0.3));
    }
    evaluateSkeleton(skel, &base_m, &.{});
    try expectSkelEqual(skel, skel2);
}

test "channel index: in-place bone move into indexed bone" {
    try checkInPlaceBoneMove(std.testing.allocator, 5, 2);
}

test "channel index: in-place bone move out of indexed bone" {
    try checkInPlaceBoneMove(std.testing.allocator, 2, 5);
}

test "channel index: in-place cross-bone swap" {
    const allocator = std.testing.allocator;
    const skel = try Skeleton.init(allocator, 8);
    defer skel.deinit();
    const skel2 = try Skeleton.init(allocator, 8);
    defer skel2.deinit();
    bindDefaults(skel);
    bindDefaults(skel2);

    const ag = try makeTwoStepGroup(allocator, "swap", 0, .{ 1, 1, 1 }, 1, .{ 2, 2, 2 });
    defer ag.deinit();
    ag.skeleton = skel;
    ag.play(true);
    ag.update(0.1);
    try std.testing.expect(ag.hasValidChannelIndex());

    const bl = try makeTwoStepGroup(allocator, "swap_legacy", 0, .{ 1, 1, 1 }, 1, .{ 2, 2, 2 });
    defer bl.deinit();
    bl.skeleton = skel2;
    bl.play(true);

    std.mem.swap(usize, &ag.channels[0].bone_index, &ag.channels[1].bone_index);
    std.mem.swap(usize, &bl.channels[0].bone_index, &bl.channels[1].bone_index);
    try std.testing.expect(!ag.hasValidChannelIndex());
    // Same ptr/len, swapped bones: bone 0 now reads the (2,2,2) values.
    try std.testing.expectEqual(@as(f32, 2.0), (grab(ag, 0, 0.0)).p.?.x);
    try std.testing.expectEqual(@as(f32, 1.0), (grab(ag, 1, 0.0)).p.?.x);
    for ([_]usize{ 0, 1 }) |b| try expectSampleEqual(grab(bl, b, 0.4), grab(ag, b, 0.4));

    // Single partial-weight clip (1-clip blend path) without re-update.
    for ([_]*AnimationGroup{ ag, bl }) |g| {
        g.is_playing = true;
        g.current_time = 0.0;
        g.setWeight(0.5);
    }
    const single_m = [_]*AnimationGroup{ag};
    const single_l = [_]*AnimationGroup{bl};
    evaluateSkeleton(skel, &single_m, &.{});
    evaluateSkeleton(skel2, &single_l, &.{});
    try expectSkelEqual(skel, skel2);

    ag.update(0.1);
    try std.testing.expect(ag.hasValidChannelIndex());
    ag.current_time = 0.0;
    for ([_]usize{ 0, 1 }) |b| try expectSampleEqual(grab(bl, b, 0.4), grabIndexed(ag, b, 0.4));
    evaluateSkeleton(skel, &single_m, &.{});
    try expectSkelEqual(skel, skel2);
}

test "channel index: in-place path and sampler replacement stays live" {
    const allocator = std.testing.allocator;
    const skel = try Skeleton.init(allocator, 1);
    defer skel.deinit();
    const skel2 = try Skeleton.init(allocator, 1);
    defer skel2.deinit();
    bindDefaults(skel);
    bindDefaults(skel2);

    const mkPair = struct {
        fn f(a: std.mem.Allocator, name: []const u8) !*AnimationGroup {
            const channels = try a.alloc(AnimationChannel, 2);
            var filled: usize = 0;
            errdefer {
                for (channels[0..filled]) |*ch| freeCh(a, ch);
                a.free(channels);
            }
            channels[0] = try mkCh(a, 0, .translation, &.{ 0.0, 1.0 }, &.{ 0, 0, 0, 10, 20, 30 }, .linear);
            filled += 1;
            channels[1] = try mkCh(a, 0, .rotation, &.{ 0.0, 1.0 }, &.{ 0, 0, 0, 1, 0, 0, 0, 1 }, .linear);
            filled += 1;
            return AnimationGroup.init(a, name, channels, 1.0);
        }
    }.f;

    const ag = try mkPair(allocator, "live");
    defer ag.deinit();
    ag.skeleton = skel;
    ag.play(true);
    ag.update(0.1);
    try std.testing.expect(ag.hasValidChannelIndex());

    const bl = try mkPair(allocator, "live_legacy");
    defer bl.deinit();
    bl.skeleton = skel2;
    bl.play(true);

    // Same bones, swapped target paths (same ptr/len): the route stays
    // valid, but sampling must follow the live paths immediately.
    std.mem.swap(AnimationPath, &ag.channels[0].target_path, &ag.channels[1].target_path);
    std.mem.swap(AnimationPath, &bl.channels[0].target_path, &bl.channels[1].target_path);
    try std.testing.expect(ag.hasValidChannelIndex());
    for ([_]f32{ 0.0, 0.3, 1.0 }) |t| try expectSampleEqual(grab(bl, 0, t), grabIndexed(ag, 0, t));

    // Sampler float edits are impossible by type (outputs is []const f32);
    // slice replacement (mutable header) reads live through the valid route.
    const new_out = try dupF(allocator, &.{ 7, 8, 9, 11, 12, 13 });
    errdefer allocator.free(new_out);
    const new_out_l = try dupF(allocator, &.{ 7, 8, 9, 11, 12, 13 });
    errdefer allocator.free(new_out_l);
    allocator.free(ag.channels[0].sampler.outputs);
    allocator.free(bl.channels[0].sampler.outputs);
    ag.channels[0].sampler.outputs = new_out;
    bl.channels[0].sampler.outputs = new_out_l;
    // Bones unchanged: still valid, new sampler honored without update.
    try std.testing.expect(ag.hasValidChannelIndex());
    for ([_]f32{ 0.0, 0.3, 1.0 }) |t| try expectSampleEqual(grab(bl, 0, t), grabIndexed(ag, 0, t));

    // Partial-weight eval through the valid route matches the legacy twin.
    for ([_]*AnimationGroup{ ag, bl }) |g| {
        g.is_playing = true;
        g.current_time = 0.0;
        g.setWeight(0.5);
    }
    const single_m = [_]*AnimationGroup{ag};
    const single_l = [_]*AnimationGroup{bl};
    evaluateSkeleton(skel, &single_m, &.{});
    evaluateSkeleton(skel2, &single_l, &.{});
    try expectSkelEqual(skel, skel2);
}

test "channel index: same-bone reorder stays valid and exact" {
    // Whole-struct swap with identical (bone, path): the bone sequence is
    // unchanged so the route stays valid — and stays CORRECT, because the
    // route stores positions while the leaf reads live contents: LAST-wins
    // follows the new order exactly like the legacy scan.
    const allocator = std.testing.allocator;
    const skel = try Skeleton.init(allocator, 1);
    defer skel.deinit();
    const skel2 = try Skeleton.init(allocator, 1);
    defer skel2.deinit();
    bindDefaults(skel);
    bindDefaults(skel2);

    const ag = try makeTwoStepGroup(allocator, "reorder", 0, .{ 1, 1, 1 }, 0, .{ 2, 2, 2 });
    defer ag.deinit();
    ag.skeleton = skel;
    ag.play(true);
    ag.update(0.1);
    const bl = try makeTwoStepGroup(allocator, "reorder_legacy", 0, .{ 1, 1, 1 }, 0, .{ 2, 2, 2 });
    defer bl.deinit();
    bl.skeleton = skel2;
    bl.play(true);

    try std.testing.expectEqual(@as(f32, 2.0), (grabIndexed(ag, 0, 0.0)).p.?.x);
    std.mem.swap(AnimationChannel, &ag.channels[0], &ag.channels[1]);
    std.mem.swap(AnimationChannel, &bl.channels[0], &bl.channels[1]);
    try std.testing.expect(ag.hasValidChannelIndex());
    try std.testing.expectEqual(@as(f32, 1.0), (grabIndexed(ag, 0, 0.0)).p.?.x);
    try expectSampleEqual(grab(bl, 0, 0.2), grabIndexed(ag, 0, 0.2));

    for ([_]*AnimationGroup{ ag, bl }) |g| {
        g.is_playing = true;
        g.current_time = 0.0;
        g.setWeight(0.5);
    }
    const single_m = [_]*AnimationGroup{ag};
    const single_l = [_]*AnimationGroup{bl};
    evaluateSkeleton(skel, &single_m, &.{});
    evaluateSkeleton(skel2, &single_l, &.{});
    try expectSkelEqual(skel, skel2);
}
