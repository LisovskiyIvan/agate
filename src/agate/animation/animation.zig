const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Skeleton = @import("skeleton.zig").Skeleton;

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
        const factor = if (t1 > t0) (time - t0) / (t1 - t0) else 0.0;

        const base0 = idx * 3;
        const v0 = Vec3.new(self.outputs[base0], self.outputs[base0 + 1], self.outputs[base0 + 2]);

        if (self.interpolation == .step) return v0;

        const base1 = (idx + 1) * 3;
        const v1 = Vec3.new(self.outputs[base1], self.outputs[base1 + 1], self.outputs[base1 + 2]);
        return Vec3.lerp(v0, v1, factor);
    }

    pub fn sampleQuat(self: AnimationSampler, time: f32) Quat {
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
        const factor = if (t1 > t0) (time - t0) / (t1 - t0) else 0.0;

        const base0 = idx * 4;
        const q0 = Quat{
            .x = self.outputs[base0],
            .y = self.outputs[base0 + 1],
            .z = self.outputs[base0 + 2],
            .w = self.outputs[base0 + 3],
        };

        if (self.interpolation == .step) return q0.normalize();

        const base1 = (idx + 1) * 4;
        const q1 = Quat{
            .x = self.outputs[base1],
            .y = self.outputs[base1 + 1],
            .z = self.outputs[base1 + 2],
            .w = self.outputs[base1 + 3],
        };
        return Quat.slerp(q0, q1, factor);
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

pub const AnimationGroup = struct {
    allocator: std.mem.Allocator,
    name: []const u8 = "",
    channels: []AnimationChannel,
    skeleton: ?*Skeleton = null,
    duration: f32 = 0.0,

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
            self.allocator.free(ch.sampler.timestamps);
            self.allocator.free(ch.sampler.outputs);
        }
        self.allocator.free(self.channels);
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
                    return;
                }
            }
        }
        const range = self.to - self.from;
        if (range <= 0.0) return;

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
        const skel = self.skeleton orelse return;

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
    outputs[0] = 0.0; outputs[1] = 0.0; outputs[2] = 0.0;
    outputs[3] = 4.0; outputs[4] = 8.0; outputs[5] = 12.0;

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
    const times1 = try allocator.alloc(f32, 1); times1[0] = 0.0;
    const out1 = try allocator.alloc(f32, 3); out1[0] = 0.0; out1[1] = 0.0; out1[2] = 10.0;
    const ch1 = try allocator.alloc(AnimationChannel, 1);
    ch1[0] = .{ .bone_index = 0, .target_path = .translation, .sampler = .{ .timestamps = times1, .outputs = out1 } };
    const ag1 = try AnimationGroup.init(allocator, "walk", ch1, 1.0);
    defer ag1.deinit();
    ag1.play(true);

    // Clip 2: Run (pos: 0, 0, 20)
    const times2 = try allocator.alloc(f32, 1); times2[0] = 0.0;
    const out2 = try allocator.alloc(f32, 3); out2[0] = 0.0; out2[1] = 0.0; out2[2] = 20.0;
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
    const times3 = try allocator.alloc(f32, 1); times3[0] = 0.0;
    const out3 = try allocator.alloc(f32, 3); out3[0] = 0.0; out3[1] = 5.0; out3[2] = 0.0;
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

    const times1 = try allocator.alloc(f32, 1); times1[0] = 0.0;
    const out1 = try allocator.alloc(f32, 3); out1[0] = 0; out1[1] = 0; out1[2] = 0;
    const ch1 = try allocator.alloc(AnimationChannel, 1);
    ch1[0] = .{ .bone_index = 0, .target_path = .translation, .sampler = .{ .timestamps = times1, .outputs = out1 } };
    const ag1 = try AnimationGroup.init(allocator, "clip1", ch1, 1.0);
    defer ag1.deinit();

    const times2 = try allocator.alloc(f32, 1); times2[0] = 0.0;
    const out2 = try allocator.alloc(f32, 3); out2[0] = 0; out2[1] = 0; out2[2] = 0;
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
