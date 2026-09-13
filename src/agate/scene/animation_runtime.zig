const std = @import("std");

const AnimationGroup = @import("../animation/animation.zig").AnimationGroup;
const evaluateSkeleton = @import("../animation/animation.zig").evaluateSkeleton;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const Mesh = @import("../mesh.zig").Mesh;

/// Advances skeletal animation for one frame: per-group timeline updates,
/// then per-skeleton blending of active base + additive clips, then CPU
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

    for (skeletons) |skel| {
        var active_base: [16]*AnimationGroup = undefined;
        var base_count: usize = 0;
        var active_add: [16]*AnimationGroup = undefined;
        var add_count: usize = 0;

        for (animation_groups) |ag| {
            if (ag.skeleton == skel and ag.is_playing and ag.weight > 0.0001) {
                if (ag.is_additive) {
                    if (add_count < active_add.len) {
                        active_add[add_count] = ag;
                        add_count += 1;
                    }
                } else {
                    if (base_count < active_base.len) {
                        active_base[base_count] = ag;
                        base_count += 1;
                    }
                }
            }
        }

        evaluateSkeleton(skel, active_base[0..base_count], active_add[0..add_count]);
    }

    // CPU morph targets: weights tracks only flag meshes dirty above;
    // blend base + deltas here once per frame, before the render queue is
    // built in render().
    for (meshes) |mesh| {
        if (mesh.hasMorphTargets()) mesh.applyMorphs();
    }
}
