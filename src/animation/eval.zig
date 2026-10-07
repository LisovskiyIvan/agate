const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Skeleton = @import("skeleton.zig").Skeleton;
const AnimationGroup = @import("group.zig").AnimationGroup;

pub fn evaluateSkeleton(skel: *Skeleton, active_base: []const *AnimationGroup, active_additive: []const *AnimationGroup) void {
    evaluateSkeletonInner(skel, active_base, active_additive, false);
}

/// Parallel-safe evaluation: bit-identical bone results to
/// evaluateSkeleton, but never touches node/morph targets. The serial
/// update() pass already applied every playing group's node tracks at the
/// same current_time/weight, so the fast-path node re-apply inside
/// applyAtTime would only rewrite identical values — skipping it keeps mesh
/// state identical while removing the one cross-skeleton shared write
/// (two groups can bind the same mesh transform, where "last group wins"
/// ordering would otherwise become schedule-dependent).
///
/// Precondition: no concurrent update() on the same groups (times/weights
/// are read but never written here). Skeletons whose active groups own
/// node channels must use evaluateSkeleton on the serial path instead —
/// see scene/animation_runtime.zig.
pub fn evaluateSkeletonPoseOnly(skel: *Skeleton, active_base: []const *AnimationGroup, active_additive: []const *AnimationGroup) void {
    evaluateSkeletonInner(skel, active_base, active_additive, true);
}

fn evaluateSkeletonInner(skel: *Skeleton, active_base: []const *AnimationGroup, active_additive: []const *AnimationGroup, comptime pose_only: bool) void {
    if (active_base.len == 0 and active_additive.len == 0) return;

    // Fast path: exactly 1 active base clip with full weight and no additive layers
    if (active_base.len == 1 and active_additive.len == 0 and active_base[0].weight >= 0.999) {
        if (pose_only) {
            active_base[0].applySkeletonAtTime(active_base[0].current_time);
        } else {
            active_base[0].applyAtTime(active_base[0].current_time);
        }
        return;
    }

    var total_base_w: f32 = 0.0;
    for (active_base) |ag| {
        total_base_w += ag.weight;
    }

    // Hoist clip weight normalization outside the per-bone loop
    const two_clip_alpha: f32 = if (active_base.len == 2) blk: {
        const sum_w = active_base[0].weight + active_base[1].weight;
        break :blk if (sum_w > 0.0001) active_base[1].weight / sum_w else 0.5;
    } else 0.5;

    var norm_weights_buf: [16]f32 = undefined;
    if (active_base.len > 2 and total_base_w > 0.0001) {
        const inv_w = 1.0 / total_base_w;
        for (active_base, 0..) |ag, i| {
            if (i < 16) norm_weights_buf[i] = ag.weight * inv_w;
        }
    }

    for (skel.bones, 0..) |*bone, b_idx| {
        var blended_pos = bone.bind_position;
        var blended_rot = bone.bind_rotation;
        var blended_scale = bone.bind_scale;

        if (total_base_w > 0.0001) {
            if (active_base.len == 1) {
                const ag = active_base[0];
                var p: ?Vec3 = null;
                var r: ?Quat = null;
                var s: ?Vec3 = null;
                ag.sampleBoneAtTime(b_idx, ag.current_time, &p, &r, &s);
                const bp = p orelse bone.bind_position;
                const br = r orelse bone.bind_rotation;
                const bs = s orelse bone.bind_scale;

                if (ag.weight >= 0.999) {
                    blended_pos = bp;
                    blended_rot = br;
                    blended_scale = bs;
                } else {
                    blended_pos = Vec3.lerp(bone.bind_position, bp, ag.weight);
                    blended_rot = Quat.slerp(bone.bind_rotation, br, ag.weight);
                    blended_scale = Vec3.lerp(bone.bind_scale, bs, ag.weight);
                }
            } else if (active_base.len == 2) {
                const ag0 = active_base[0];
                const ag1 = active_base[1];
                var p0: ?Vec3 = null;
                var r0: ?Quat = null;
                var s0: ?Vec3 = null;
                var p1: ?Vec3 = null;
                var r1: ?Quat = null;
                var s1: ?Vec3 = null;
                ag0.sampleBoneAtTime(b_idx, ag0.current_time, &p0, &r0, &s0);
                ag1.sampleBoneAtTime(b_idx, ag1.current_time, &p1, &r1, &s1);

                const bp0 = p0 orelse bone.bind_position;
                const br0 = r0 orelse bone.bind_rotation;
                const bs0 = s0 orelse bone.bind_scale;

                const bp1 = p1 orelse bone.bind_position;
                const br1 = r1 orelse bone.bind_rotation;
                const bs1 = s1 orelse bone.bind_scale;

                blended_pos = Vec3.lerp(bp0, bp1, two_clip_alpha);
                blended_rot = Quat.slerp(br0, br1, two_clip_alpha);
                blended_scale = Vec3.lerp(bs0, bs1, two_clip_alpha);
            } else {
                var acc_p = Vec3.zero;
                var acc_s = Vec3.zero;
                var acc_q = Quat{ .x = 0, .y = 0, .z = 0, .w = 0 };
                var ref_q: ?Quat = null;

                for (active_base, 0..) |ag, ag_idx| {
                    const norm_w = if (ag_idx < 16) norm_weights_buf[ag_idx] else ag.weight / total_base_w;
                    var p: ?Vec3 = null;
                    var r: ?Quat = null;
                    var s: ?Vec3 = null;
                    ag.sampleBoneAtTime(b_idx, ag.current_time, &p, &r, &s);
                    const bp = p orelse bone.bind_position;
                    const br = r orelse bone.bind_rotation;
                    const bs = s orelse bone.bind_scale;

                    acc_p = acc_p.add(bp.scale(norm_w));
                    acc_s = acc_s.add(bs.scale(norm_w));

                    var cur_r = br;
                    if (ref_q) |rq| {
                        const dot = cur_r.x * rq.x + cur_r.y * rq.y + cur_r.z * rq.z + cur_r.w * rq.w;
                        if (dot < 0.0) {
                            cur_r = .{ .x = -cur_r.x, .y = -cur_r.y, .z = -cur_r.z, .w = -cur_r.w };
                        }
                    } else {
                        ref_q = cur_r;
                    }
                    acc_q.x += cur_r.x * norm_w;
                    acc_q.y += cur_r.y * norm_w;
                    acc_q.z += cur_r.z * norm_w;
                    acc_q.w += cur_r.w * norm_w;
                }
                blended_pos = acc_p;
                blended_scale = acc_s;
                blended_rot = acc_q.normalize();
            }
        }

        // Apply additive layers on top of blended pose
        for (active_additive) |ag| {
            if (ag.weight <= 0.0001) continue;
            var p: ?Vec3 = null;
            var r: ?Quat = null;
            var s: ?Vec3 = null;
            ag.sampleBoneAtTime(b_idx, ag.current_time, &p, &r, &s);

            if (p) |add_p| {
                const delta_p = add_p.sub(bone.bind_position);
                blended_pos = blended_pos.add(delta_p.scale(ag.weight));
            }
            if (s) |add_s| {
                const delta_s = add_s.sub(bone.bind_scale);
                blended_scale = blended_scale.add(delta_s.scale(ag.weight));
            }
            if (r) |add_r| {
                const delta_q = bone.bind_rotation.conjugate().mul(add_r).normalize();
                const weighted_delta_q = Quat.slerp(Quat.identity, delta_q, ag.weight);
                blended_rot = blended_rot.mul(weighted_delta_q).normalize();
            }
        }

        bone.local_position = blended_pos;
        bone.local_rotation = blended_rot;
        bone.local_scale = blended_scale;
    }

    skel.update();
}
