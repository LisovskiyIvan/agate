const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const easing_mod = @import("easing.zig");
pub const EasingType = easing_mod.EasingType;

const sampler_mod = @import("sampler.zig");
pub const AnimationPath = sampler_mod.AnimationPath;
pub const AnimationInterpolation = sampler_mod.AnimationInterpolation;
pub const AnimationEvent = sampler_mod.AnimationEvent;
pub const AnimationSampler = sampler_mod.AnimationSampler;

pub const AnimationChannel = struct {
    bone_index: usize,
    target_path: AnimationPath,
    sampler: AnimationSampler,
};

/// Live binding between a node animation track and a mesh transform.
/// Raw Vec3 pointers (instead of a Mesh pointer) keep this module free of
/// import cycles: mesh.zig -> scene.zig -> animation.zig. The loader binds
/// &mesh.position / &mesh.rotation / &mesh.scale here; unit tests bind plain
/// Vec3 fields of a local struct. rotation_euler uses degrees to match Mesh.
/// Pointers must stay valid for the group lifetime (both are scene-owned).
pub const NodeTarget = struct {
    position: *Vec3,
    rotation_euler: *Vec3,
    scaling: *Vec3,
    rest_position: Vec3 = Vec3.zero,
    rest_rotation: Quat = Quat.identity,
    rest_scale: Vec3 = Vec3.one,
    /// When true the channel is parsed but never applied. The loader sets it
    /// when the target glTF node is a skeleton joint of a skinned mesh: the
    /// skeleton path already drives that transform, applying the node track
    /// on top would double-apply it.
    skip: bool = false,
};

/// One glTF node animation channel (target node, not a skeleton joint).
/// translation/rotation/scale targets index AnimationGroup.node_targets;
/// weights (morph) targets index AnimationGroup.morph_targets and write the
/// bound mesh weights via bindMorphTarget (mesh.zig owns the blend).
pub const NodeChannel = struct {
    target: usize,
    target_path: AnimationPath,
    sampler: AnimationSampler,
    easing: EasingType = .linear,
};

/// Live binding between a node weights track and a mesh's morph weights.
/// Raw slice/pointer (instead of a Mesh pointer) keeps this module free of
/// import cycles: mesh.zig -> scene.zig -> animation.zig. The loader binds
/// mesh.morph_weights and &mesh.morph_dirty; unit tests bind plain arrays.
/// rest_weights is an owned snapshot for group-weight blending and stop()
/// restore; the group frees it in deinit. All three must stay valid for the
/// group lifetime (weights/dirty are scene-owned).
pub const MorphWeightsTarget = struct {
    weights: []f32 = &.{},
    rest_weights: []f32 = &.{},
    dirty: ?*bool = null,
};

/// Guards sampler reads: true when timestamps exist and outputs holds at
/// least one full frame per keyframe for the given path. For weights a frame
/// is weight_count values (the bound mesh's morph target count), not 1.
/// CUBICSPLINE tracks store (in-tangent, value, out-tangent) per key, so they
/// need 3x the floats. Invalid channels are skipped by the node applier
/// instead of crashing on out-of-bounds access.
pub fn samplerHasFrames(sampler: AnimationSampler, path: AnimationPath, weight_count: usize) bool {
    if (sampler.timestamps.len == 0) return false;
    const stride: usize = switch (path) {
        .translation, .scale => 3,
        .rotation => 4,
        .weights => @max(weight_count, 1),
    };
    const mult: usize = if (sampler.interpolation == .cubic_spline) 3 else 1;
    return sampler.outputs.len >= sampler.timestamps.len * stride * mult;
}
