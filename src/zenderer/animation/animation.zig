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

    current_time: f32 = 0.0,
    speed_ratio: f32 = 1.0,
    is_playing: bool = false,
    loop: bool = true,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, channels: []AnimationChannel, duration: f32) !*AnimationGroup {
        const ag = try allocator.create(AnimationGroup);
        ag.* = .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .channels = channels,
            .duration = duration,
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
        self.loop = loop;
        self.is_playing = true;
    }

    pub fn pause(self: *AnimationGroup) void {
        self.is_playing = false;
    }

    pub fn stop(self: *AnimationGroup) void {
        self.is_playing = false;
        self.current_time = 0.0;
        if (self.skeleton) |skel| {
            skel.resetToBindPose();
        }
    }

    pub fn goToFrame(self: *AnimationGroup, time: f32) void {
        self.current_time = std.math.clamp(time, 0.0, self.duration);
        self.applyAtTime(self.current_time);
    }

    pub fn update(self: *AnimationGroup, dt: f32) void {
        if (!self.is_playing or self.duration <= 0.0) return;

        self.current_time += dt * self.speed_ratio;
        if (self.current_time > self.duration) {
            if (self.loop) {
                self.current_time = @mod(self.current_time, self.duration);
            } else {
                self.current_time = self.duration;
                self.is_playing = false;
            }
        } else if (self.current_time < 0.0) {
            if (self.loop) {
                self.current_time = self.duration - @mod(-self.current_time, self.duration);
            } else {
                self.current_time = 0.0;
                self.is_playing = false;
            }
        }

        self.applyAtTime(self.current_time);
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
