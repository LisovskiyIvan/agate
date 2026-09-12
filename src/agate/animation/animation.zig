const std = @import("std");
const easing_mod = @import("easing.zig");
pub const EasingType = easing_mod.EasingType;

pub const sampler = @import("sampler.zig");
pub const AnimationPath = sampler.AnimationPath;
pub const AnimationInterpolation = sampler.AnimationInterpolation;
pub const AnimationEvent = sampler.AnimationEvent;
pub const AnimationSampler = sampler.AnimationSampler;

pub const channels = @import("channels.zig");
pub const AnimationChannel = channels.AnimationChannel;
pub const NodeTarget = channels.NodeTarget;
pub const NodeChannel = channels.NodeChannel;
pub const MorphWeightsTarget = channels.MorphWeightsTarget;

pub const group = @import("group.zig");
pub const AnimationGroup = group.AnimationGroup;

pub const eval = @import("eval.zig");
pub const evaluateSkeleton = eval.evaluateSkeleton;

test {
    _ = @import("tests.zig");
}
