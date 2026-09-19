const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Skeleton = @import("skeleton.zig").Skeleton;
const Bone = @import("skeleton.zig").Bone;
const AnimationGroup = @import("group.zig").AnimationGroup;
const AnimationChannel = @import("channels.zig").AnimationChannel;

/// How translation tracks are handled when retargeting between skeletons.
pub const TranslationMode = enum {
    /// Copy translation outputs verbatim.
    keep,
    /// Scale translation outputs by the target/source bone rest-length ratio
    /// when both bones have a rest length, otherwise keep verbatim.
    scale_by_bone_length,
    /// Drop translation channels entirely.
    drop,
};

/// Explicit source-bone-name → target-bone-name override for
/// `retargetAnimationGroup`. Lets clips transfer between rigs that share no
/// bone names (e.g. Fox `b_Hip_01` → CesiumMan `Skeleton_torso_joint_1`).
pub const BoneMap = struct {
    source: []const u8,
    target: []const u8,
};

/// Knobs for `retargetAnimationGroup`. All fields defaulted.
pub const RetargetOptions = struct {
    /// When true, only rotation channels are kept; translation, scale and
    /// weights channels are dropped regardless of `translation_mode`.
    rotation_only: bool = false,
    translation_mode: TranslationMode = .scale_by_bone_length,
    /// Explicit per-bone mapping, matched by source bone name. Checked
    /// before the default name lookup; unmatched bones keep the existing
    /// behavior (same-name lookup, then same-index fallback for unnamed
    /// bones). Empty by default (pure name mapping).
    bone_map: []const BoneMap = &.{},
};

/// Rest length of a bone: the bind-pose offset length from its parent joint.
/// Used as the translation scaling reference. Bones authored at the origin
/// (zero bind offset, e.g. root joints) have no rest length.
pub fn boneRestLength(bone: Bone) f32 {
    return bone.bind_position.length();
}

/// Translation scale factor for one mapped bone pair: target rest length
/// divided by source rest length. Returns 1.0 (keep) when either bone has
/// no rest length.
pub fn translationScaleFactor(source_bone: Bone, target_bone: Bone) f32 {
    const src = boneRestLength(source_bone);
    const dst = boneRestLength(target_bone);
    if (src <= 1e-6 or dst <= 1e-6) return 1.0;
    return dst / src;
}

/// Retargets a skeleton `AnimationGroup` clip from `source_skeleton` onto
/// `target_skeleton`, returning a new group owned by the caller (deinit to
/// free). The returned group's `skeleton` is bound to `target_skeleton` and
/// its `duration`/`from`/`to` are copied from the source group.
///
/// Mapping: per source bone, in order: (1) explicit `options.bone_map`
/// entry matched by source bone name → target bone of that name; (2) target
/// bone with the same name via `findBoneIndex`; (3) same bone index when
/// the source bone is unnamed (existing fallback). Named bones with no
/// target match are skipped. Out-of-range source bone indexes with no name
/// are resolved by index fallback the same way, otherwise skipped.
///
/// Channel policy: rotations are copied verbatim; translations follow
/// `options.translation_mode` (`scale_by_bone_length` multiplies every
/// translation output, tangents included, by `translationScaleFactor`);
/// scale (and weights) channels are copied verbatim. With
/// `options.rotation_only`, only rotation channels are kept.
///
/// Unmapped bones are skipped with a single `std.log.warn` per call
/// summarizing the skipped channel count; playback itself (`applyAtTime`,
/// `evaluateSkeleton`) never logs and never allocates.
///
/// MVP limitations: skeleton channels only. Node/morph channels, timeline
/// events and node bindings are not copied.
pub fn retargetAnimationGroup(
    allocator: std.mem.Allocator,
    source_group: *const AnimationGroup,
    source_skeleton: *const Skeleton,
    target_skeleton: *const Skeleton,
    options: RetargetOptions,
) !*AnimationGroup {
    var out_channels = std.ArrayList(AnimationChannel).empty;
    defer {
        // Freed on success via toOwnedSlice; on error, release partial work.
        if (out_channels.items.len == 0) out_channels.deinit(allocator);
    }
    errdefer {
        for (out_channels.items) |ch| {
            if (ch.sampler.timestamps.len > 0) allocator.free(ch.sampler.timestamps);
            if (ch.sampler.outputs.len > 0) allocator.free(ch.sampler.outputs);
        }
        out_channels.deinit(allocator);
    }

    var skipped_unmapped: usize = 0;

    for (source_group.channels) |src_ch| {
        const keep_path: bool = switch (src_ch.target_path) {
            .rotation => true,
            .translation => !options.rotation_only and options.translation_mode != .drop,
            .scale, .weights => !options.rotation_only,
        };
        if (!keep_path) continue;

        // Resolve the target bone: explicit map entry by source name
        // first, then same-name lookup, otherwise index fallback for
        // unnamed bones.
        var target_index: ?usize = null;
        var source_bone: ?Bone = null;
        if (src_ch.bone_index < source_skeleton.bones.len) {
            const src_bone = source_skeleton.bones[src_ch.bone_index];
            source_bone = src_bone;
            if (src_bone.name.len > 0) {
                for (options.bone_map) |entry| {
                    if (std.mem.eql(u8, entry.source, src_bone.name)) {
                        target_index = target_skeleton.findBoneIndex(entry.target);
                        break;
                    }
                }
                if (target_index == null) {
                    target_index = target_skeleton.findBoneIndex(src_bone.name);
                }
            } else if (src_ch.bone_index < target_skeleton.bones.len) {
                target_index = src_ch.bone_index;
            }
        } else if (src_ch.bone_index < target_skeleton.bones.len) {
            target_index = src_ch.bone_index;
        }

        const tgt_idx = target_index orelse {
            skipped_unmapped += 1;
            continue;
        };

        const timestamps = try allocator.dupe(f32, src_ch.sampler.timestamps);
        errdefer if (timestamps.len > 0) allocator.free(timestamps);
        const outputs = try allocator.dupe(f32, src_ch.sampler.outputs);
        errdefer if (outputs.len > 0) allocator.free(outputs);

        if (src_ch.target_path == .translation and options.translation_mode == .scale_by_bone_length) {
            var factor: f32 = 1.0;
            if (source_bone) |sb| {
                if (tgt_idx < target_skeleton.bones.len) {
                    factor = translationScaleFactor(sb, target_skeleton.bones[tgt_idx]);
                }
            }
            if (factor != 1.0) {
                for (outputs) |*v| v.* *= factor;
            }
        }

        try out_channels.append(allocator, .{
            .bone_index = tgt_idx,
            .target_path = src_ch.target_path,
            .sampler = .{
                .timestamps = timestamps,
                .outputs = outputs,
                .interpolation = src_ch.sampler.interpolation,
            },
        });
    }

    if (skipped_unmapped > 0) {
        std.log.warn("retargetAnimationGroup: skipped {d} channel(s) with no target bone for group '{s}'", .{ skipped_unmapped, source_group.name });
    }

    const owned = try out_channels.toOwnedSlice(allocator);
    errdefer {
        for (owned) |ch| {
            if (ch.sampler.timestamps.len > 0) allocator.free(ch.sampler.timestamps);
            if (ch.sampler.outputs.len > 0) allocator.free(ch.sampler.outputs);
        }
        if (owned.len > 0) allocator.free(owned);
    }

    const retargeted = try AnimationGroup.init(allocator, source_group.name, owned, source_group.duration);
    retargeted.skeleton = @constCast(target_skeleton);
    retargeted.from = source_group.from;
    retargeted.to = source_group.to;
    return retargeted;
}

const testing = std.testing;

fn makeTestSkeleton(allocator: std.mem.Allocator, names: []const []const u8) !*Skeleton {
    const skel = try Skeleton.init(allocator, names.len);
    errdefer skel.deinit();
    for (names, 0..) |nm, i| {
        skel.bones[i].name = try allocator.dupe(u8, nm);
    }
    return skel;
}

fn addTestChannel(
    allocator: std.mem.Allocator,
    channels: *std.ArrayList(AnimationChannel),
    bone_index: usize,
    path: @import("sampler.zig").AnimationPath,
    times: []const f32,
    outputs: []const f32,
) !void {
    try channels.append(allocator, .{
        .bone_index = bone_index,
        .target_path = path,
        .sampler = .{
            .timestamps = try allocator.dupe(f32, times),
            .outputs = try allocator.dupe(f32, outputs),
            .interpolation = .linear,
        },
    });
}

fn quatDot(a: Quat, b: Quat) f32 {
    return @abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
}

test "retarget maps same names in different order" {
    const allocator = testing.allocator;
    const src = try makeTestSkeleton(allocator, &.{ "hips", "spine", "head" });
    defer src.deinit();
    const dst = try makeTestSkeleton(allocator, &.{ "head", "hips", "spine" });
    defer dst.deinit();

    const q_hips = Quat.fromEulerDeg(Vec3.new(0.0, 30.0, 0.0));
    const q_spine = Quat.fromEulerDeg(Vec3.new(0.0, 60.0, 0.0));
    const q_head = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));

    var list = std.ArrayList(AnimationChannel).empty;
    defer {
        for (list.items) |ch| {
            allocator.free(ch.sampler.timestamps);
            allocator.free(ch.sampler.outputs);
        }
        list.deinit(allocator);
    }
    try addTestChannel(allocator, &list, 0, .rotation, &.{0.0}, &.{ q_hips.x, q_hips.y, q_hips.z, q_hips.w });
    try addTestChannel(allocator, &list, 1, .rotation, &.{0.0}, &.{ q_spine.x, q_spine.y, q_spine.z, q_spine.w });
    try addTestChannel(allocator, &list, 2, .rotation, &.{0.0}, &.{ q_head.x, q_head.y, q_head.z, q_head.w });

    const owned = try list.toOwnedSlice(allocator);
    const src_group = try AnimationGroup.init(allocator, "order", owned, 1.0);
    defer src_group.deinit();

    const retargeted = try retargetAnimationGroup(allocator, src_group, src, dst, .{});
    defer retargeted.deinit();

    try testing.expectEqual(@as(usize, 3), retargeted.channels.len);
    // Target order is head=0, hips=1, spine=2.
    for (retargeted.channels) |ch| {
        var p: ?Vec3 = null;
        var r: ?Quat = null;
        var s: ?Vec3 = null;
        retargeted.sampleBoneAtTime(ch.bone_index, 0.0, &p, &r, &s);
        const name = dst.bones[ch.bone_index].name;
        if (std.mem.eql(u8, name, "hips")) {
            try testing.expect(quatDot(r.?, q_hips) > 0.9999);
        } else if (std.mem.eql(u8, name, "spine")) {
            try testing.expect(quatDot(r.?, q_spine) > 0.9999);
        } else if (std.mem.eql(u8, name, "head")) {
            try testing.expect(quatDot(r.?, q_head) > 0.9999);
        } else {
            return error.TestUnexpectedResult;
        }
    }
}

test "retarget skips renamed and missing bones" {
    const allocator = testing.allocator;
    const src = try makeTestSkeleton(allocator, &.{ "a", "b", "gone" });
    defer src.deinit();
    const dst = try makeTestSkeleton(allocator, &.{ "a", "b" });
    defer dst.deinit();

    var list = std.ArrayList(AnimationChannel).empty;
    defer {
        for (list.items) |ch| {
            allocator.free(ch.sampler.timestamps);
            allocator.free(ch.sampler.outputs);
        }
        list.deinit(allocator);
    }
    try addTestChannel(allocator, &list, 0, .rotation, &.{0.0}, &.{ 0.0, 0.0, 0.0, 1.0 });
    try addTestChannel(allocator, &list, 1, .rotation, &.{0.0}, &.{ 0.0, 0.0, 0.0, 1.0 });
    try addTestChannel(allocator, &list, 2, .rotation, &.{0.0}, &.{ 0.0, 0.0, 0.0, 1.0 });

    const owned = try list.toOwnedSlice(allocator);
    const src_group = try AnimationGroup.init(allocator, "missing", owned, 1.0);
    defer src_group.deinit();

    const retargeted = try retargetAnimationGroup(allocator, src_group, src, dst, .{});
    defer retargeted.deinit();

    try testing.expectEqual(@as(usize, 2), retargeted.channels.len);
    for (retargeted.channels) |ch| {
        try testing.expect(ch.bone_index < dst.bones.len);
        try testing.expect(!std.mem.eql(u8, dst.bones[ch.bone_index].name, "gone"));
    }
}

test "retarget rotation-only drops translations" {
    const allocator = testing.allocator;
    const src = try makeTestSkeleton(allocator, &.{"root"});
    defer src.deinit();
    const dst = try makeTestSkeleton(allocator, &.{"root"});
    defer dst.deinit();

    var list = std.ArrayList(AnimationChannel).empty;
    defer {
        for (list.items) |ch| {
            allocator.free(ch.sampler.timestamps);
            allocator.free(ch.sampler.outputs);
        }
        list.deinit(allocator);
    }
    try addTestChannel(allocator, &list, 0, .rotation, &.{0.0}, &.{ 0.0, 0.0, 0.0, 1.0 });
    try addTestChannel(allocator, &list, 0, .translation, &.{0.0}, &.{ 1.0, 2.0, 3.0 });
    try addTestChannel(allocator, &list, 0, .scale, &.{0.0}, &.{ 1.0, 1.0, 1.0 });

    const owned = try list.toOwnedSlice(allocator);
    const src_group = try AnimationGroup.init(allocator, "rot_only", owned, 1.0);
    defer src_group.deinit();

    const retargeted = try retargetAnimationGroup(allocator, src_group, src, dst, .{ .rotation_only = true });
    defer retargeted.deinit();

    try testing.expectEqual(@as(usize, 1), retargeted.channels.len);
    try testing.expectEqual(@import("sampler.zig").AnimationPath.rotation, retargeted.channels[0].target_path);
}

test "retarget scale_by_bone_length scales translations by rest ratio" {
    const allocator = testing.allocator;
    const src = try makeTestSkeleton(allocator, &.{"limb"});
    defer src.deinit();
    const dst = try makeTestSkeleton(allocator, &.{"limb"});
    defer dst.deinit();
    src.bones[0].bind_position = Vec3.new(1.0, 0.0, 0.0);
    dst.bones[0].bind_position = Vec3.new(2.0, 0.0, 0.0);

    var list = std.ArrayList(AnimationChannel).empty;
    defer {
        for (list.items) |ch| {
            allocator.free(ch.sampler.timestamps);
            allocator.free(ch.sampler.outputs);
        }
        list.deinit(allocator);
    }
    try addTestChannel(allocator, &list, 0, .translation, &.{ 0.0, 1.0 }, &.{ 1.0, 0.0, 0.0, 3.0, 0.0, 0.0 });
    try addTestChannel(allocator, &list, 0, .scale, &.{0.0}, &.{ 2.0, 2.0, 2.0 });

    const owned = try list.toOwnedSlice(allocator);
    const src_group = try AnimationGroup.init(allocator, "scale_t", owned, 1.0);
    defer src_group.deinit();

    const scaled = try retargetAnimationGroup(allocator, src_group, src, dst, .{ .translation_mode = .scale_by_bone_length });
    defer scaled.deinit();
    try testing.expectEqual(@as(usize, 2), scaled.channels.len);
    for (scaled.channels) |ch| {
        if (ch.target_path == .translation) {
            try testing.expectApproxEqAbs(@as(f32, 2.0), ch.sampler.outputs[0], 1e-5);
            try testing.expectApproxEqAbs(@as(f32, 6.0), ch.sampler.outputs[3], 1e-5);
        } else {
            // Scale channels are kept as-is, never length-scaled.
            try testing.expectApproxEqAbs(@as(f32, 2.0), ch.sampler.outputs[0], 1e-6);
        }
    }

    const kept = try retargetAnimationGroup(allocator, src_group, src, dst, .{ .translation_mode = .keep });
    defer kept.deinit();
    for (kept.channels) |ch| {
        if (ch.target_path == .translation) {
            try testing.expectApproxEqAbs(@as(f32, 1.0), ch.sampler.outputs[0], 1e-6);
            try testing.expectApproxEqAbs(@as(f32, 3.0), ch.sampler.outputs[3], 1e-6);
        }
    }

    const dropped = try retargetAnimationGroup(allocator, src_group, src, dst, .{ .translation_mode = .drop });
    defer dropped.deinit();
    try testing.expectEqual(@as(usize, 1), dropped.channels.len);
    try testing.expectEqual(@import("sampler.zig").AnimationPath.scale, dropped.channels[0].target_path);

    // Zero rest length on either side keeps translations verbatim.
    dst.bones[0].bind_position = Vec3.zero;
    const zero = try retargetAnimationGroup(allocator, src_group, src, dst, .{});
    defer zero.deinit();
    for (zero.channels) |ch| {
        if (ch.target_path == .translation) {
            try testing.expectApproxEqAbs(@as(f32, 1.0), ch.sampler.outputs[0], 1e-6);
        }
    }
}

test "retargeted playback matches source rotations on matching bones" {
    const allocator = testing.allocator;
    const src = try makeTestSkeleton(allocator, &.{ "hips", "arm" });
    defer src.deinit();
    const dst = try makeTestSkeleton(allocator, &.{ "arm", "hips", "extra" });
    defer dst.deinit();

    const q0 = Quat.identity;
    const q1 = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));

    var list = std.ArrayList(AnimationChannel).empty;
    defer {
        for (list.items) |ch| {
            allocator.free(ch.sampler.timestamps);
            allocator.free(ch.sampler.outputs);
        }
        list.deinit(allocator);
    }
    try addTestChannel(allocator, &list, 0, .rotation, &.{ 0.0, 1.0 }, &.{ q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w });
    try addTestChannel(allocator, &list, 1, .rotation, &.{ 0.0, 1.0 }, &.{ q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w });

    const owned = try list.toOwnedSlice(allocator);
    const src_group = try AnimationGroup.init(allocator, "play", owned, 1.0);
    defer src_group.deinit();
    src_group.skeleton = src;

    const retargeted = try retargetAnimationGroup(allocator, src_group, src, dst, .{});
    defer retargeted.deinit();
    try testing.expect(retargeted.skeleton == dst);

    for ([_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 }) |t| {
        var sp: ?Vec3 = null;
        var sr: ?Quat = null;
        var ss: ?Vec3 = null;
        src_group.sampleBoneAtTime(0, t, &sp, &sr, &ss);
        var dp: ?Vec3 = null;
        var dr: ?Quat = null;
        var ds: ?Vec3 = null;
        const dst_hips = dst.findBoneIndex("hips").?;
        retargeted.sampleBoneAtTime(dst_hips, t, &dp, &dr, &ds);
        try testing.expect(sr != null and dr != null);
        try testing.expect(quatDot(sr.?, dr.?) > 0.9999);

        // Full evaluation path: single full-weight clip drives the skeleton.
        src_group.current_time = t;
        src_group.weight = 1.0;
        const src_base = [_]*AnimationGroup{src_group};
        @import("eval.zig").evaluateSkeleton(src, &src_base, &.{});
        retargeted.current_time = t;
        retargeted.weight = 1.0;
        const dst_base = [_]*AnimationGroup{retargeted};
        @import("eval.zig").evaluateSkeleton(dst, &dst_base, &.{});
        const src_idx: usize = 0;
        const got = dst.bones[dst_hips].local_rotation;
        const want = src.bones[src_idx].local_rotation;
        try testing.expect(quatDot(got, want) > 0.9999);
    }
}

test "retarget falls back to index for unnamed bones" {
    const allocator = testing.allocator;
    const src = try Skeleton.init(allocator, 1);
    defer src.deinit();
    const dst = try Skeleton.init(allocator, 2);
    defer dst.deinit();
    dst.bones[1].name = try allocator.dupe(u8, "named");

    var list = std.ArrayList(AnimationChannel).empty;
    defer {
        for (list.items) |ch| {
            allocator.free(ch.sampler.timestamps);
            allocator.free(ch.sampler.outputs);
        }
        list.deinit(allocator);
    }
    try addTestChannel(allocator, &list, 0, .rotation, &.{0.0}, &.{ 0.0, 0.0, 0.0, 1.0 });

    const owned = try list.toOwnedSlice(allocator);
    const src_group = try AnimationGroup.init(allocator, "unnamed", owned, 1.0);
    defer src_group.deinit();

    const retargeted = try retargetAnimationGroup(allocator, src_group, src, dst, .{});
    defer retargeted.deinit();
    try testing.expectEqual(@as(usize, 1), retargeted.channels.len);
    try testing.expectEqual(@as(usize, 0), retargeted.channels[0].bone_index);
}

test "retarget explicit bone_map transfers rotations across disjoint names" {
    const allocator = testing.allocator;
    // Source and target rigs share NO bone names; without a map nothing
    // resolves, with a map every channel lands on its anatomical peer.
    const src = try makeTestSkeleton(allocator, &.{ "src_hip", "src_arm", "src_tail" });
    defer src.deinit();
    const dst = try makeTestSkeleton(allocator, &.{ "dst_hip", "dst_arm" });
    defer dst.deinit();

    const q_rest = Quat.identity;
    const q_pose = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));

    var list = std.ArrayList(AnimationChannel).empty;
    defer {
        for (list.items) |ch| {
            allocator.free(ch.sampler.timestamps);
            allocator.free(ch.sampler.outputs);
        }
        list.deinit(allocator);
    }
    const times = [_]f32{ 0.0, 1.0 };
    const outputs = [_]f32{ q_rest.x, q_rest.y, q_rest.z, q_rest.w, q_pose.x, q_pose.y, q_pose.z, q_pose.w };
    try addTestChannel(allocator, &list, 0, .rotation, &times, &outputs);
    try addTestChannel(allocator, &list, 1, .rotation, &times, &outputs);
    try addTestChannel(allocator, &list, 2, .rotation, &times, &outputs);

    const owned = try list.toOwnedSlice(allocator);
    const src_group = try AnimationGroup.init(allocator, "disjoint", owned, 1.0);
    defer src_group.deinit();
    src_group.skeleton = src;

    // No map: disjoint names resolve nothing.
    const unmapped = try retargetAnimationGroup(allocator, src_group, src, dst, .{});
    defer unmapped.deinit();
    try testing.expectEqual(@as(usize, 0), unmapped.channels.len);

    // Explicit map: hip+arm transfer, tail (no peer) is honestly skipped.
    const map = [_]BoneMap{
        .{ .source = "src_hip", .target = "dst_hip" },
        .{ .source = "src_arm", .target = "dst_arm" },
    };
    const retargeted = try retargetAnimationGroup(allocator, src_group, src, dst, .{ .bone_map = &map });
    defer retargeted.deinit();
    try testing.expectEqual(@as(usize, 2), retargeted.channels.len);

    const dst_hip = dst.findBoneIndex("dst_hip").?;
    const dst_arm = dst.findBoneIndex("dst_arm").?;
    for (retargeted.channels) |ch| {
        try testing.expect(ch.bone_index == dst_hip or ch.bone_index == dst_arm);

        // Sampled rotation matches the source pose at t=1 and is genuinely
        // off rest (real motion, not bind pose).
        var sp: ?Vec3 = null;
        var sr: ?Quat = null;
        var ss: ?Vec3 = null;
        src_group.sampleBoneAtTime(if (ch.bone_index == dst_hip) 0 else 1, 1.0, &sp, &sr, &ss);
        var dp: ?Vec3 = null;
        var dr: ?Quat = null;
        var ds: ?Vec3 = null;
        retargeted.sampleBoneAtTime(ch.bone_index, 1.0, &dp, &dr, &ds);
        try testing.expect(sr != null and dr != null);
        try testing.expect(quatDot(sr.?, dr.?) > 0.9999);
        try testing.expect(quatDot(dr.?, Quat.identity) < 0.999);

        // Full evaluation path: the mapped local rotation leaves rest.
        retargeted.current_time = 1.0;
        retargeted.weight = 1.0;
        const dst_base = [_]*AnimationGroup{retargeted};
        @import("eval.zig").evaluateSkeleton(dst, &dst_base, &.{});
        try testing.expect(quatDot(dst.bones[ch.bone_index].local_rotation, Quat.identity) < 0.999);
    }
}
