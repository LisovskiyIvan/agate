const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Skeleton = @import("skeleton.zig").Skeleton;
const easing_mod = @import("easing.zig");
pub const EasingType = easing_mod.EasingType;

pub const AnimationPath = enum {
    translation,
    rotation,
    scale,
    weights,
};

pub const AnimationInterpolation = enum {
    linear,
    step,
    cubic_spline,
};

pub const AnimationSampler = struct {
    timestamps: []const f32,
    outputs: []const f32,
    interpolation: AnimationInterpolation = .linear,

    pub fn sampleVec3(self: AnimationSampler, time: f32) Vec3 {
        return self.sampleVec3Eased(time, .linear);
    }

    /// Samples a vec3 track, warping the intra-keyframe factor with an easing
    /// curve. STEP ignores easing. CUBICSPLINE has no tangent support here and
    /// falls back to (eased) linear interpolation; it never crashes.
    pub fn sampleVec3Eased(self: AnimationSampler, time: f32, easing: EasingType) Vec3 {
        if (self.timestamps.len == 0) return Vec3.zero;
        if (time <= self.timestamps[0]) {
            return Vec3.new(self.outputs[0], self.outputs[1], self.outputs[2]);
        }
        const last_idx = self.timestamps.len - 1;
        if (time >= self.timestamps[last_idx]) {
            const base = last_idx * 3;
            return Vec3.new(self.outputs[base], self.outputs[base + 1], self.outputs[base + 2]);
        }

        const idx = self.findKeyframeIndex(time);
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const raw = if (t1 > t0) (time - t0) / (t1 - t0) else 0.0;

        const base0 = idx * 3;
        const v0 = Vec3.new(self.outputs[base0], self.outputs[base0 + 1], self.outputs[base0 + 2]);

        if (self.interpolation == .step) return v0;

        const factor = easing_mod.evaluate(easing, raw);
        const base1 = (idx + 1) * 3;
        const v1 = Vec3.new(self.outputs[base1], self.outputs[base1 + 1], self.outputs[base1 + 2]);
        return Vec3.lerp(v0, v1, factor);
    }

    pub fn sampleQuat(self: AnimationSampler, time: f32) Quat {
        return self.sampleQuatEased(time, .linear);
    }

    /// Quaternion variant of sampleVec3Eased: slerp factor is eased, STEP
    /// ignores easing, CUBICSPLINE falls back to (eased) linear blending.
    pub fn sampleQuatEased(self: AnimationSampler, time: f32, easing: EasingType) Quat {
        if (self.timestamps.len == 0) return Quat.identity;
        if (time <= self.timestamps[0]) {
            const q = Quat{
                .x = self.outputs[0],
                .y = self.outputs[1],
                .z = self.outputs[2],
                .w = self.outputs[3],
            };
            return q.normalize();
        }
        const last_idx = self.timestamps.len - 1;
        if (time >= self.timestamps[last_idx]) {
            const base = last_idx * 4;
            const q = Quat{
                .x = self.outputs[base],
                .y = self.outputs[base + 1],
                .z = self.outputs[base + 2],
                .w = self.outputs[base + 3],
            };
            return q.normalize();
        }

        const idx = self.findKeyframeIndex(time);
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const raw = if (t1 > t0) (time - t0) / (t1 - t0) else 0.0;

        const base0 = idx * 4;
        const q0 = Quat{
            .x = self.outputs[base0],
            .y = self.outputs[base0 + 1],
            .z = self.outputs[base0 + 2],
            .w = self.outputs[base0 + 3],
        };

        if (self.interpolation == .step) return q0.normalize();

        const factor = easing_mod.evaluate(easing, raw);
        const base1 = (idx + 1) * 4;
        const q1 = Quat{
            .x = self.outputs[base1],
            .y = self.outputs[base1 + 1],
            .z = self.outputs[base1 + 2],
            .w = self.outputs[base1 + 3],
        };
        return Quat.slerp(q0, q1, factor);
    }

    /// Samples a weights (morph) track with out.len values per keyframe.
    /// Per-component mirror of sampleVec3Eased: STEP holds the keyframe,
    /// other modes lerp with the eased factor; CUBICSPLINE falls back to
    /// (eased) linear. Undersized outputs or missing keys leave out untouched.
    pub fn sampleWeightsInto(self: AnimationSampler, time: f32, out: []f32, easing: EasingType) void {
        const n = out.len;
        if (n == 0 or self.timestamps.len == 0) return;
        const frames = self.outputs.len / n;
        if (frames == 0) return;
        const keys = @min(self.timestamps.len, frames);
        if (keys == 0) return;
        if (time <= self.timestamps[0]) {
            @memcpy(out, self.outputs[0..n]);
            return;
        }
        const last = keys - 1;
        if (time >= self.timestamps[last]) {
            @memcpy(out, self.outputs[last * n .. last * n + n]);
            return;
        }
        var idx = self.findKeyframeIndex(time);
        if (idx >= last) idx = last - 1;
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const raw = if (t1 > t0) (time - t0) / (t1 - t0) else 0.0;

        const base0 = idx * n;
        if (self.interpolation == .step) {
            @memcpy(out, self.outputs[base0 .. base0 + n]);
            return;
        }
        const factor = easing_mod.evaluate(easing, raw);
        const base1 = (idx + 1) * n;
        for (0..n) |j| {
            out[j] = self.outputs[base0 + j] + (self.outputs[base1 + j] - self.outputs[base0 + j]) * factor;
        }
    }

    fn findKeyframeIndex(self: AnimationSampler, time: f32) usize {
        var low: usize = 0;
        var high: usize = self.timestamps.len - 1;
        while (low < high - 1) {
            const mid = low + (high - low) / 2;
            if (self.timestamps[mid] <= time) {
                low = mid;
            } else {
                high = mid;
            }
        }
        return low;
    }
};

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
/// Invalid channels are skipped by the node applier instead of crashing on
/// out-of-bounds access.
fn samplerHasFrames(sampler: AnimationSampler, path: AnimationPath, weight_count: usize) bool {
    if (sampler.timestamps.len == 0) return false;
    const stride: usize = switch (path) {
        .translation, .scale => 3,
        .rotation => 4,
        .weights => @max(weight_count, 1),
    };
    return sampler.outputs.len >= sampler.timestamps.len * stride;
}

pub const AnimationGroup = struct {
    allocator: std.mem.Allocator,
    name: []const u8 = "",
    channels: []AnimationChannel,
    skeleton: ?*Skeleton = null,
    duration: f32 = 0.0,
    /// Plain-node (non-skeleton) tracks. update() applies them directly to the
    /// bound mesh transforms, so Scene.updateAnimations needs no changes.
    /// Limitation: node tracks are NOT cross-group blended. Every playing
    /// group blends its own sample with the captured rest pose by its own
    /// weight; when several groups drive the same mesh, the last updated
    /// group wins.
    node_channels: []NodeChannel = &.{},
    node_targets: []NodeTarget = &.{},
    /// Morph-weight bindings for weights channels. NodeChannel.target indexes
    /// this list when target_path == .weights (TRS paths index node_targets).
    morph_targets: []MorphWeightsTarget = &.{},

    from: f32 = 0.0,
    to: f32 = 0.0,
    current_time: f32 = 0.0,
    speed_ratio: f32 = 1.0,
    weight: f32 = 1.0,
    is_additive: bool = false,
    is_playing: bool = false,
    loop: bool = true,

    // Weight transition / Cross-fading
    fade_start_weight: f32 = 1.0,
    fade_target_weight: f32 = 1.0,
    fade_duration: f32 = 0.0,
    fade_timer: f32 = 0.0,
    stop_on_fade_out: bool = true,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, channels: []AnimationChannel, duration: f32) !*AnimationGroup {
        const ag = try allocator.create(AnimationGroup);
        ag.* = .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .channels = channels,
            .duration = duration,
            .from = 0.0,
            .to = duration,
        };
        return ag;
    }

    pub fn deinit(self: *AnimationGroup) void {
        for (self.channels) |ch| {
            if (ch.sampler.timestamps.len > 0) self.allocator.free(ch.sampler.timestamps);
            if (ch.sampler.outputs.len > 0) self.allocator.free(ch.sampler.outputs);
        }
        if (self.channels.len > 0) self.allocator.free(self.channels);
        for (self.node_channels) |ch| {
            if (ch.sampler.timestamps.len > 0) self.allocator.free(ch.sampler.timestamps);
            if (ch.sampler.outputs.len > 0) self.allocator.free(ch.sampler.outputs);
        }
        if (self.node_channels.len > 0) self.allocator.free(self.node_channels);
        if (self.node_targets.len > 0) self.allocator.free(self.node_targets);
        for (self.morph_targets) |mt| {
            if (mt.rest_weights.len > 0) self.allocator.free(mt.rest_weights);
        }
        if (self.morph_targets.len > 0) self.allocator.free(self.morph_targets);
        if (self.name.len > 0) {
            self.allocator.free(self.name);
        }
        self.allocator.destroy(self);
    }

    pub fn play(self: *AnimationGroup, loop: bool) void {
        self.from = 0.0;
        self.to = self.duration;
        self.loop = loop;
        self.is_playing = true;
        self.fade_duration = 0.0;
    }

    pub fn playRange(self: *AnimationGroup, from: f32, to: f32, loop: bool, speed: ?f32) void {
        self.from = std.math.clamp(from, 0.0, self.duration);
        self.to = std.math.clamp(to, self.from, self.duration);
        if (self.to <= self.from) self.to = self.duration;
        self.loop = loop;
        self.fade_duration = 0.0;
        if (speed) |s| self.speed_ratio = s;
        if (self.speed_ratio >= 0.0) {
            if (self.current_time < self.from or self.current_time > self.to) {
                self.current_time = self.from;
            }
        } else {
            if (self.current_time < self.from or self.current_time > self.to) {
                self.current_time = self.to;
            }
        }
        self.is_playing = true;
    }

    pub fn setSpeed(self: *AnimationGroup, speed: f32) void {
        self.speed_ratio = speed;
    }

    pub fn setWeight(self: *AnimationGroup, w: f32) void {
        self.weight = std.math.clamp(w, 0.0, 1.0);
        self.fade_duration = 0.0;
    }

    pub fn setAdditive(self: *AnimationGroup, additive: bool) void {
        self.is_additive = additive;
    }

    /// Fades the animation weight towards target_weight over duration seconds.
    pub fn fadeTo(self: *AnimationGroup, target_weight: f32, duration: f32, stop_if_zero: bool) void {
        self.fade_start_weight = self.weight;
        self.fade_target_weight = std.math.clamp(target_weight, 0.0, 1.0);
        self.fade_duration = @max(duration, 0.0001);
        self.fade_timer = 0.0;
        self.stop_on_fade_out = stop_if_zero;
        if (!self.is_playing and self.fade_target_weight > 0.0) {
            self.is_playing = true;
        }
    }

    /// Smoothly fades in this animation to weight 1.0 over duration seconds.
    pub fn fadeIn(self: *AnimationGroup, duration: f32) void {
        if (!self.is_playing) {
            self.weight = 0.0;
            self.is_playing = true;
        }
        self.fadeTo(1.0, duration, false);
    }

    /// Smoothly fades out this animation to weight 0.0 over duration seconds, then stops it.
    pub fn fadeOut(self: *AnimationGroup, duration: f32) void {
        self.fadeTo(0.0, duration, true);
    }

    /// Smoothly cross-fades from this animation to target over duration seconds.
    pub fn crossFadeTo(self: *AnimationGroup, target: *AnimationGroup, duration: f32) void {
        if (self == target) return;
        self.fadeOut(duration);
        target.fadeIn(duration);
    }

    pub fn pause(self: *AnimationGroup) void {
        self.is_playing = false;
    }

    pub fn stop(self: *AnimationGroup) void {
        self.is_playing = false;
        self.current_time = self.from;
        self.fade_duration = 0.0;
        if (self.skeleton) |skel| {
            skel.resetToBindPose();
        }
        self.restoreNodeRestPose();
        self.restoreMorphRestWeights();
    }

    pub fn goToFrame(self: *AnimationGroup, time: f32) void {
        self.current_time = std.math.clamp(time, self.from, self.to);
        self.applyAtTime(self.current_time);
    }

    pub fn update(self: *AnimationGroup, dt: f32) void {
        if (!self.is_playing) return;

        // Process weight fading
        if (self.fade_duration > 0.00001) {
            self.fade_timer += dt;
            const t = std.math.clamp(self.fade_timer / self.fade_duration, 0.0, 1.0);
            self.weight = self.fade_start_weight + (self.fade_target_weight - self.fade_start_weight) * t;

            if (self.fade_timer >= self.fade_duration) {
                self.weight = self.fade_target_weight;
                self.fade_duration = 0.0;
                self.fade_timer = 0.0;
                if (self.weight <= 0.0001 and self.stop_on_fade_out) {
                    self.is_playing = false;
                    self.current_time = self.from;
                    // Weight is zero so this restores the rest pose.
                    self.applyNodesAtTime(self.current_time);
                    return;
                }
            }
        }
        const range = self.to - self.from;
        if (range <= 0.0) {
            // Zero-length clips (e.g. a single keyframe) still drive nodes.
            self.applyNodesAtTime(self.current_time);
            return;
        }

        self.current_time += dt * self.speed_ratio;

        if (self.speed_ratio >= 0.0) {
            if (self.current_time > self.to) {
                if (self.loop) {
                    self.current_time = self.from + @mod(self.current_time - self.from, range);
                } else {
                    self.current_time = self.to;
                    self.is_playing = false;
                }
            }
        } else {
            if (self.current_time < self.from) {
                if (self.loop) {
                    self.current_time = self.to - @mod(self.from - self.current_time, range);
                } else {
                    self.current_time = self.from;
                    self.is_playing = false;
                }
            }
        }

        self.applyNodesAtTime(self.current_time);
    }

    /// Binds a mesh transform to this group and snapshots its current TRS as
    /// the rest pose used for weight blending. Returns the target index for
    /// NodeChannel.target. The pointed-to Vec3s must outlive the group.
    pub fn bindNodeTarget(self: *AnimationGroup, position: *Vec3, rotation_euler: *Vec3, scaling: *Vec3, skip: bool) !usize {
        const idx = self.node_targets.len;
        const grown = try self.allocator.alloc(NodeTarget, idx + 1);
        if (idx > 0) {
            @memcpy(grown[0..idx], self.node_targets);
            self.allocator.free(self.node_targets);
        }
        grown[idx] = .{
            .position = position,
            .rotation_euler = rotation_euler,
            .scaling = scaling,
            .rest_position = position.*,
            .rest_rotation = Quat.fromEulerDeg(rotation_euler.*),
            .rest_scale = scaling.*,
            .skip = skip,
        };
        self.node_targets = grown;
        return idx;
    }

    /// Binds a mesh morph-weights slice to this group and snapshots its
    /// current values as the rest pose used for weight blending and stop()
    /// restore. Returns the morph target index for weights NodeChannels.
    /// The slice and dirty flag must outlive the group (both scene-owned).
    pub fn bindMorphTarget(self: *AnimationGroup, weights: []f32, dirty: *bool) !usize {
        const rest = try self.allocator.dupe(f32, weights);
        errdefer self.allocator.free(rest);
        const idx = self.morph_targets.len;
        const grown = try self.allocator.alloc(MorphWeightsTarget, idx + 1);
        if (idx > 0) {
            @memcpy(grown[0..idx], self.morph_targets);
            self.allocator.free(self.morph_targets);
        }
        grown[idx] = .{
            .weights = weights,
            .rest_weights = rest,
            .dirty = dirty,
        };
        self.morph_targets = grown;
        return idx;
    }

    /// Appends a node channel. The group takes ownership of the sampler
    /// buffers and frees them in deinit.
    pub fn addNodeChannel(self: *AnimationGroup, channel: NodeChannel) !void {
        const old_len = self.node_channels.len;
        const grown = try self.allocator.alloc(NodeChannel, old_len + 1);
        if (old_len > 0) {
            @memcpy(grown[0..old_len], self.node_channels);
            self.allocator.free(self.node_channels);
        }
        grown[old_len] = channel;
        self.node_channels = grown;
    }

    /// Re-snapshots the rest pose of all bound targets from their current
    /// live values. Useful when the mesh transform changed after binding.
    pub fn captureNodeRestPoses(self: *AnimationGroup) void {
        for (self.node_targets) |*t| {
            t.rest_position = t.position.*;
            t.rest_rotation = Quat.fromEulerDeg(t.rotation_euler.*);
            t.rest_scale = t.scaling.*;
        }
    }

    /// Restores every bound morph target to its rest weights and flags it
    /// dirty so the next scene update re-blends the mesh.
    pub fn restoreMorphRestWeights(self: *AnimationGroup) void {
        for (self.morph_targets) |*mt| {
            const n = @min(mt.weights.len, mt.rest_weights.len);
            @memcpy(mt.weights[0..n], mt.rest_weights[0..n]);
            if (mt.dirty) |d| d.* = true;
        }
    }

    /// Restores every bound (non-skipped) target to its rest pose.
    pub fn restoreNodeRestPose(self: *AnimationGroup) void {
        for (self.node_targets) |*t| {
            if (t.skip) continue;
            t.position.* = t.rest_position;
            t.rotation_euler.* = t.rest_rotation.normalize().toEulerDeg();
            t.scaling.* = t.rest_scale;
        }
    }

    /// Samples all node channels at the given time and writes the result into
    /// the bound transforms, blended with the rest pose by the group weight:
    /// lerp for translation/scale, slerp (via quaternion round-trip) for
    /// rotation, lerp with rest weights for morph weights (clamped to
    /// [0, 1], mesh flagged dirty for the scene blend). Called automatically
    /// by update() and applyAtTime().
    pub fn applyNodesAtTime(self: *AnimationGroup, time: f32) void {
        if (self.node_channels.len == 0) return;
        const w = std.math.clamp(self.weight, 0.0, 1.0);
        for (self.node_channels) |ch| {
            // Weights channels index morph_targets, not node_targets, and
            // write per-mesh state (each mesh owns its weights slice).
            if (ch.target_path == .weights) {
                if (ch.target >= self.morph_targets.len) continue;
                const mt = &self.morph_targets[ch.target];
                if (mt.weights.len == 0) continue;
                if (!samplerHasFrames(ch.sampler, ch.target_path, mt.weights.len)) continue;
                ch.sampler.sampleWeightsInto(time, mt.weights, ch.easing);
                const rn = @min(mt.weights.len, mt.rest_weights.len);
                for (mt.weights, 0..) |*slot, j| {
                    var v = slot.*;
                    if (w < 0.999 and j < rn) {
                        v = mt.rest_weights[j] + (v - mt.rest_weights[j]) * w;
                    }
                    slot.* = std.math.clamp(v, 0.0, 1.0);
                }
                if (mt.dirty) |d| d.* = true;
                continue;
            }
            if (ch.target >= self.node_targets.len) continue;
            if (!samplerHasFrames(ch.sampler, ch.target_path, 0)) continue;
            const t = &self.node_targets[ch.target];
            if (t.skip) continue;
            switch (ch.target_path) {
                .translation => {
                    const sampled = ch.sampler.sampleVec3Eased(time, ch.easing);
                    t.position.* = if (w >= 0.999) sampled else Vec3.lerp(t.rest_position, sampled, w);
                },
                .scale => {
                    const sampled = ch.sampler.sampleVec3Eased(time, ch.easing);
                    t.scaling.* = if (w >= 0.999) sampled else Vec3.lerp(t.rest_scale, sampled, w);
                },
                .rotation => {
                    const sampled = ch.sampler.sampleQuatEased(time, ch.easing);
                    if (w >= 0.999) {
                        t.rotation_euler.* = sampled.normalize().toEulerDeg();
                    } else {
                        const blended = Quat.slerp(t.rest_rotation, sampled, w);
                        t.rotation_euler.* = blended.normalize().toEulerDeg();
                    }
                },
                // Handled above; kept for exhaustiveness.
                .weights => {},
            }
        }
    }

    pub fn sampleBoneAtTime(self: *const AnimationGroup, bone_idx: usize, time: f32, out_pos: *?Vec3, out_rot: *?Quat, out_scale: *?Vec3) void {
        for (self.channels) |ch| {
            if (ch.bone_index == bone_idx) {
                switch (ch.target_path) {
                    .translation => out_pos.* = ch.sampler.sampleVec3(time),
                    .rotation => out_rot.* = ch.sampler.sampleQuat(time),
                    .scale => out_scale.* = ch.sampler.sampleVec3(time),
                    .weights => {},
                }
            }
        }
    }

    pub fn applyAtTime(self: *AnimationGroup, time: f32) void {
        if (self.skeleton) |skel| {
            for (self.channels) |ch| {
                if (ch.bone_index >= skel.bones.len) continue;
                const bone = &skel.bones[ch.bone_index];
                switch (ch.target_path) {
                    .translation => bone.local_position = ch.sampler.sampleVec3(time),
                    .rotation => bone.local_rotation = ch.sampler.sampleQuat(time),
                    .scale => bone.local_scale = ch.sampler.sampleVec3(time),
                    .weights => {},
                }
            }

            skel.update();
        }

        self.applyNodesAtTime(time);
    }

    /// Node counterpart of sampleBoneAtTime: samples every channel driving
    /// target_idx at the given time (with easing). Missing paths stay null.
    /// Out-of-range targets and invalid samplers yield nulls, never a crash.
    pub fn sampleNodeAtTime(self: *const AnimationGroup, target_idx: usize, time: f32, out_pos: *?Vec3, out_rot: *?Quat, out_scale: *?Vec3) void {
        if (target_idx >= self.node_targets.len) return;
        if (self.node_targets[target_idx].skip) return;
        for (self.node_channels) |ch| {
            if (ch.target != target_idx) continue;
            if (!samplerHasFrames(ch.sampler, ch.target_path, 0)) continue;
            switch (ch.target_path) {
                .translation => out_pos.* = ch.sampler.sampleVec3Eased(time, ch.easing),
                .rotation => out_rot.* = ch.sampler.sampleQuatEased(time, ch.easing),
                .scale => out_scale.* = ch.sampler.sampleVec3Eased(time, ch.easing),
                .weights => {},
            }
        }
    }
};

/// Evaluates blended base animations and applies additive animation layers onto the target skeleton.
pub fn evaluateSkeleton(skel: *Skeleton, active_base: []const *AnimationGroup, active_additive: []const *AnimationGroup) void {
    if (active_base.len == 0 and active_additive.len == 0) return;

    // Fast path: exactly 1 active base clip with full weight and no additive layers
    if (active_base.len == 1 and active_additive.len == 0 and active_base[0].weight >= 0.999) {
        active_base[0].applyAtTime(active_base[0].current_time);
        return;
    }

    var total_base_w: f32 = 0.0;
    for (active_base) |ag| {
        total_base_w += ag.weight;
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

                const sum_w = ag0.weight + ag1.weight;
                const alpha = if (sum_w > 0.0001) ag1.weight / sum_w else 0.5;
                blended_pos = Vec3.lerp(bp0, bp1, alpha);
                blended_rot = Quat.slerp(br0, br1, alpha);
                blended_scale = Vec3.lerp(bs0, bs1, alpha);
            } else {
                var acc_p = Vec3.zero;
                var acc_s = Vec3.zero;
                var acc_q = Quat{ .x = 0, .y = 0, .z = 0, .w = 0 };
                var ref_q: ?Quat = null;

                for (active_base) |ag| {
                    const norm_w = ag.weight / total_base_w;
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
