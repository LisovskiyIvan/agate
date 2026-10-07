const std = @import("std");

const jobs = @import("../jobs.zig");
const AnimationGroup = @import("../animation/animation.zig").AnimationGroup;
const evaluateSkeleton = @import("../animation/animation.zig").evaluateSkeleton;
const evaluateSkeletonPoseOnly = @import("../animation/eval.zig").evaluateSkeletonPoseOnly;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const Mesh = @import("../mesh.zig").Mesh;

/// At or above this many skeletons the per-skeleton eval loop runs on the
/// jobs pool; below it everything stays inline. Deliberately far below
/// jobs.Pool.min_len_for_workers (4096): one skeleton is up to 64 bones of
/// sampler binary search + quaternion slerp plus a full hierarchy resolve
/// (~130us for a 64-bone / 3-clip skeleton in benchmark 9), so the forkJoin
/// overhead (~dozens of us) is already paid off at N=2 (measured 1.84x).
/// The threshold sits at 4 for margin: half-size skeletons at N=4 still
/// carry the work of N=2 full-size ones, and N=8 already measures 3.75x.
pub const min_skeletons_for_workers: usize = 4;

/// Advances skeletal animation for one frame: per-group timeline updates
/// (serial: advances time, fires events/callbacks, applies node tracks),
/// then per-skeleton blending of active base + additive clips (parallel at
/// or above min_skeletons_for_workers, see evalSkeletons), then CPU
/// morph-target blending on flagged meshes (applyMorphs early-outs when
/// clean). Called before the render queue is built in Scene.render.
pub fn updateAnimations(
    animation_groups: []const *AnimationGroup,
    skeletons: []const *Skeleton,
    meshes: []const *Mesh,
    dt: f32,
) void {
    for (animation_groups) |ag| {
        ag.update(dt);
    }

    evalSkeletons(animation_groups, skeletons);

    // CPU morph targets: weights tracks only flag meshes dirty above;
    // blend base + deltas here once per frame, before the render queue is
    // built in render().
    for (meshes) |mesh| {
        if (mesh.hasMorphTargets()) mesh.applyMorphs();
    }
}

const ActiveLists = struct {
    base: []*AnimationGroup,
    add: []*AnimationGroup,
};

/// Per-skeleton active-list scan (16-clips-per-side cap, same as before).
/// Read-only over groups: safe to run in workers once update() finished.
fn collectActive(
    animation_groups: []const *AnimationGroup,
    skel: *Skeleton,
    base_buf: *[16]*AnimationGroup,
    add_buf: *[16]*AnimationGroup,
) ActiveLists {
    var base_count: usize = 0;
    var add_count: usize = 0;
    for (animation_groups) |ag| {
        if (ag.skeleton == skel and ag.is_playing and ag.weight > 0.0001) {
            if (ag.is_additive) {
                if (add_count < add_buf.len) {
                    add_buf[add_count] = ag;
                    add_count += 1;
                }
            } else {
                if (base_count < base_buf.len) {
                    base_buf[base_count] = ag;
                    base_count += 1;
                }
            }
        }
    }
    return .{ .base = base_buf[0..base_count], .add = add_buf[0..add_count] };
}

/// Today's exact serial eval for one skeleton (bit-identical path).
fn evalOne(animation_groups: []const *AnimationGroup, skel: *Skeleton) void {
    var active_base: [16]*AnimationGroup = undefined;
    var active_add: [16]*AnimationGroup = undefined;
    const active = collectActive(animation_groups, skel, &active_base, &active_add);
    evaluateSkeleton(skel, active.base, active.add);
}

/// Hidden shared state check: true when any active group driving `skel`
/// owns node channels. Those write through to shared mesh transforms /
/// morph weights (documented "last updated group wins"), so the skeleton
/// is NOT index-disjoint and stays on the serial path with the full eval
/// (including the fast-path node re-apply, in skeleton order).
fn skelHasNodeWriters(animation_groups: []const *AnimationGroup, skel: *Skeleton) bool {
    for (animation_groups) |ag| {
        if (ag.skeleton == skel and ag.is_playing and ag.weight > 0.0001) {
            if (ag.node_channels.len > 0) return true;
        }
    }
    return false;
}

const EvalCtx = struct {
    groups: []const *AnimationGroup,
    skeletons: []const *Skeleton,

    /// One worker range: disjoint skeletons only. Node-writing skeletons
    /// were already evaluated serially above and are skipped here; the
    /// rest run the pose-only eval (no allocation, no sg.*, same op order
    /// as serial, hence bit-identical for any scheduling).
    fn run(c: *EvalCtx, start: usize, end: usize) void {
        for (c.skeletons[start..end]) |skel| {
            if (skelHasNodeWriters(c.groups, skel)) continue;
            var active_base: [16]*AnimationGroup = undefined;
            var active_add: [16]*AnimationGroup = undefined;
            const active = collectActive(c.groups, skel, &active_base, &active_add);
            evaluateSkeletonPoseOnly(skel, active.base, active.add);
        }
    }
};

fn evalSkeletons(animation_groups: []const *AnimationGroup, skeletons: []const *Skeleton) void {
    const pool = jobs.global;
    if (pool == null or skeletons.len < min_skeletons_for_workers) {
        for (skeletons) |skel| evalOne(animation_groups, skel);
        return;
    }
    // Serial-first: node-writing skeletons keep today's exact behavior.
    // Eligible skeletons never write shared state in either design, so
    // running these first (in skeleton order) cannot change any outcome.
    for (skeletons) |skel| {
        if (skelHasNodeWriters(animation_groups, skel)) evalOne(animation_groups, skel);
    }
    var ctx = EvalCtx{ .groups = animation_groups, .skeletons = skeletons };
    pool.?.forkJoin(EvalCtx, &ctx, EvalCtx.run, skeletons.len);
}
