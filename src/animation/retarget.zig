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
