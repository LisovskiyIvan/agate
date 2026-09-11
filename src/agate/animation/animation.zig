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

/// Scalar cubic Hermite basis: interpolates p0 -> p1 over s in [0, 1] with
/// endpoint slopes m0 (at p0) and m1 (at p1). Callers pre-scale glTF
/// per-second tangents by the keyframe interval: m = tangent * dt.
/// Exact on the endpoints: hermite(p0, m0, p1, m1, 0) == p0, (... , 1) == p1.
fn hermiteScalar(p0: f32, m0: f32, p1: f32, m1: f32, s: f32) f32 {
    const s2 = s * s;
    const s3 = s2 * s;
    return (2.0 * s3 - 3.0 * s2 + 1.0) * p0 +
        (s3 - 2.0 * s2 + s) * m0 +
        (-2.0 * s3 + 3.0 * s2) * p1 +
        (s3 - s2) * m1;
}

/// A named timestamp marker on an animation timeline. Owned by the group via
/// setEvents (names are duplicated); the slice passed to setEvents is only
/// borrowed. Events outside the group's [from, to] range never fire.
pub const AnimationEvent = struct {
    time: f32,
    name: []const u8,
};

pub const AnimationSampler = struct {
    timestamps: []const f32,
    outputs: []const f32,
    interpolation: AnimationInterpolation = .linear,

    /// Cubic (glTF CUBICSPLINE) layout: outputs hold stride*3 floats per key
    /// in (in-tangent, value, out-tangent) order, so a valid track needs at
    /// least keys * stride * 3 floats. Returns the number of keyframes that
    /// can actually be sampled (timestamps capped by the outputs capacity).
    /// LINEAR/STEP tracks do not use this helper.
    fn cubicFrames(self: AnimationSampler, stride: usize) usize {
        if (self.timestamps.len == 0 or stride == 0) return 0;
        return @min(self.timestamps.len, self.outputs.len / (stride * 3));
    }

    fn cubicValue3(self: AnimationSampler, key: usize) Vec3 {
        const base = (key * 3 + 1) * 3;
        return Vec3.new(self.outputs[base], self.outputs[base + 1], self.outputs[base + 2]);
    }

    fn cubicQuatValue(self: AnimationSampler, key: usize) Quat {
        const base = (key * 3 + 1) * 4;
        return (Quat{
            .x = self.outputs[base],
            .y = self.outputs[base + 1],
            .z = self.outputs[base + 2],
            .w = self.outputs[base + 3],
        }).normalize();
    }

    /// True cubic-spline (Hermite) evaluation for a vec3 track. Tangents are
    /// per-second slopes scaled by the keyframe interval dt, per the glTF
    /// spec. Easing is ignored here on purpose: the Hermite curve already
    /// defines the intra-keyframe shape, easing only warps the LINEAR/STEP
    /// factor. Short/truncated buffers never crash: sampling degrades to the
    /// closest usable keyframe value (or zero when nothing is usable).
    fn sampleVec3Cubic(self: AnimationSampler, time: f32) Vec3 {
        const usable = self.cubicFrames(3);
        if (usable == 0) return Vec3.zero;
        if (usable == 1 or time <= self.timestamps[0]) return self.cubicValue3(0);
        if (time >= self.timestamps[usable - 1]) return self.cubicValue3(usable - 1);
        var idx = self.findKeyframeIndex(time);
        if (idx >= usable - 1) idx = usable - 2;
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const dt = t1 - t0;
        if (dt <= 0.0) return self.cubicValue3(idx);
        const s = std.math.clamp((time - t0) / dt, 0.0, 1.0);
        const k0 = idx * 9;
        const k1 = (idx + 1) * 9;
        return Vec3.new(
            hermiteScalar(self.outputs[k0 + 3], self.outputs[k0 + 6] * dt, self.outputs[k1 + 3], self.outputs[k1] * dt, s),
            hermiteScalar(self.outputs[k0 + 4], self.outputs[k0 + 7] * dt, self.outputs[k1 + 4], self.outputs[k1 + 1] * dt, s),
            hermiteScalar(self.outputs[k0 + 5], self.outputs[k0 + 8] * dt, self.outputs[k1 + 5], self.outputs[k1 + 2] * dt, s),
        );
    }

    /// Quaternion cubic-spline: component-wise Hermite on the vec4 track,
    /// then normalized (glTF-correct). Antipodal keyframes (dot < 0) take the
    /// short path by negating the next value and its in-tangent together,
    /// which preserves the derivative direction.
    fn sampleQuatCubic(self: AnimationSampler, time: f32) Quat {
        const usable = self.cubicFrames(4);
        if (usable == 0) return Quat.identity;
        if (usable == 1 or time <= self.timestamps[0]) return self.cubicQuatValue(0);
        if (time >= self.timestamps[usable - 1]) return self.cubicQuatValue(usable - 1);
        var idx = self.findKeyframeIndex(time);
        if (idx >= usable - 1) idx = usable - 2;
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const dt = t1 - t0;
        if (dt <= 0.0) return self.cubicQuatValue(idx);
        const s = std.math.clamp((time - t0) / dt, 0.0, 1.0);
        const k0 = idx * 12;
        const k1 = (idx + 1) * 12;
        const dot = self.outputs[k0 + 4] * self.outputs[k1 + 4] +
            self.outputs[k0 + 5] * self.outputs[k1 + 5] +
            self.outputs[k0 + 6] * self.outputs[k1 + 6] +
            self.outputs[k0 + 7] * self.outputs[k1 + 7];
        const flip: f32 = if (dot < 0.0) -1.0 else 1.0;
        return (Quat{
            .x = hermiteScalar(self.outputs[k0 + 4], self.outputs[k0 + 8] * dt, flip * self.outputs[k1 + 4], flip * self.outputs[k1] * dt, s),
            .y = hermiteScalar(self.outputs[k0 + 5], self.outputs[k0 + 9] * dt, flip * self.outputs[k1 + 5], flip * self.outputs[k1 + 1] * dt, s),
            .z = hermiteScalar(self.outputs[k0 + 6], self.outputs[k0 + 10] * dt, flip * self.outputs[k1 + 6], flip * self.outputs[k1 + 2] * dt, s),
            .w = hermiteScalar(self.outputs[k0 + 7], self.outputs[k0 + 11] * dt, flip * self.outputs[k1 + 7], flip * self.outputs[k1 + 3] * dt, s),
        }).normalize();
    }

    pub fn sampleVec3(self: AnimationSampler, time: f32) Vec3 {
        return self.sampleVec3Eased(time, .linear);
    }

    /// Samples a vec3 track, warping the intra-keyframe factor with an easing
    /// curve. STEP ignores easing. CUBICSPLINE evaluates the true Hermite
    /// spline (easing intentionally ignored: it only warps the linear
    /// factor, never the Hermite shape).
    pub fn sampleVec3Eased(self: AnimationSampler, time: f32, easing: EasingType) Vec3 {
        if (self.timestamps.len == 0) return Vec3.zero;
        if (self.interpolation == .cubic_spline) return self.sampleVec3Cubic(time);
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
    /// ignores easing, CUBICSPLINE evaluates component-wise Hermite plus
    /// normalize (easing ignored, same rule as for vec3).
    pub fn sampleQuatEased(self: AnimationSampler, time: f32, easing: EasingType) Quat {
        if (self.timestamps.len == 0) return Quat.identity;
        if (self.interpolation == .cubic_spline) return self.sampleQuatCubic(time);
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
    /// LINEAR lerps with the eased factor, CUBICSPLINE runs Hermite per
    /// target (easing ignored). Undersized outputs or missing keys leave out
    /// untouched; truncated cubic buffers clamp to the closest usable value.
    pub fn sampleWeightsInto(self: AnimationSampler, time: f32, out: []f32, easing: EasingType) void {
        const n = out.len;
        if (n == 0 or self.timestamps.len == 0) return;
        if (self.interpolation == .cubic_spline) {
            self.sampleWeightsCubic(time, out);
            return;
        }
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

    /// Per-target Hermite mirror of sampleVec3Cubic: each keyframe holds
    /// (in-tangent, value, out-tangent) per morph target, i.e. 3*n floats.
    fn sampleWeightsCubic(self: AnimationSampler, time: f32, out: []f32) void {
        const n = out.len;
        const usable = self.cubicFrames(n);
        if (usable == 0) return;
        if (usable == 1 or time <= self.timestamps[0]) {
            @memcpy(out, self.outputs[n .. n + n]);
            return;
        }
        if (time >= self.timestamps[usable - 1]) {
            const base = ((usable - 1) * 3 + 1) * n;
            @memcpy(out, self.outputs[base .. base + n]);
            return;
        }
        var idx = self.findKeyframeIndex(time);
        if (idx >= usable - 1) idx = usable - 2;
        const t0 = self.timestamps[idx];
        const t1 = self.timestamps[idx + 1];
        const dt = t1 - t0;
        if (dt <= 0.0) {
            const base = (idx * 3 + 1) * n;
            @memcpy(out, self.outputs[base .. base + n]);
            return;
        }
        const s = std.math.clamp((time - t0) / dt, 0.0, 1.0);
        const k0 = idx * n * 3;
        const k1 = (idx + 1) * n * 3;
        for (0..n) |j| {
            out[j] = hermiteScalar(
                self.outputs[k0 + n + j],
                self.outputs[k0 + 2 * n + j] * dt,
                self.outputs[k1 + n + j],
                self.outputs[k1 + j] * dt,
                s,
            );
        }
    }

    fn findKeyframeIndex(self: AnimationSampler, time: f32) usize {
        if (self.timestamps.len < 2) return 0;
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
/// CUBICSPLINE tracks store (in-tangent, value, out-tangent) per key, so they
/// need 3x the floats. Invalid channels are skipped by the node applier
/// instead of crashing on out-of-bounds access.
fn samplerHasFrames(sampler: AnimationSampler, path: AnimationPath, weight_count: usize) bool {
    if (sampler.timestamps.len == 0) return false;
    const stride: usize = switch (path) {
        .translation, .scale => 3,
        .rotation => 4,
        .weights => @max(weight_count, 1),
    };
    const mult: usize = if (sampler.interpolation == .cubic_spline) 3 else 1;
    return sampler.outputs.len >= sampler.timestamps.len * stride * mult;
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

    /// Timeline event markers (owned: setEvents duplicates every name).
    /// Fired only by update() when playback time advances past an event;
    /// seeks (goToFrame/applyAtTime), stop() and fade-out completion never
    /// fire. Forward playback fires events in (prev, curr], wrapping around
    /// the loop as (prev, to] + [from, curr]; backward playback mirrors it
    /// as [curr, prev) and [from, prev) + (curr, to]. A single update that
    /// spans the whole range fires each in-range event exactly once.
    events: []AnimationEvent = &.{},
    /// Snapshot of the names fired by the most recent update() that advanced
    /// time. Borrowed views into events names; read via drainFiredEvents().
    /// Capacity always equals events.len (each event fires at most once per
    /// update), so update() itself never allocates.
    fired_names: [][]const u8 = &.{},
    fired_len: usize = 0,
    /// Optional synchronous callback invoked once per fired event, after the
    /// pose for the new time has been applied.
    on_event: ?*const fn (ctx: ?*anyopaque, name: []const u8) void = null,
    event_context: ?*anyopaque = null,

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
        for (self.events) |ev| {
            if (ev.name.len > 0) self.allocator.free(ev.name);
        }
        if (self.events.len > 0) self.allocator.free(self.events);
        if (self.fired_names.len > 0) self.allocator.free(self.fired_names);
        if (self.name.len > 0) {
            self.allocator.free(self.name);
        }
        self.allocator.destroy(self);
    }

    /// Replaces the group's timeline events. The input slice is borrowed:
    /// every name is duplicated with the given allocator (normally the
    /// group's own allocator) and owned by the group afterwards. Any
    /// previously fired-but-undrained snapshot is discarded.
    pub fn setEvents(self: *AnimationGroup, allocator: std.mem.Allocator, events: []const AnimationEvent) !void {
        const new_events = try allocator.alloc(AnimationEvent, events.len);
        var done: usize = 0;
        errdefer {
            for (new_events[0..done]) |ev| {
                if (ev.name.len > 0) allocator.free(ev.name);
            }
            if (new_events.len > 0) allocator.free(new_events);
        }
        for (events, 0..) |ev, i| {
            new_events[i] = .{ .time = ev.time, .name = try allocator.dupe(u8, ev.name) };
            done += 1;
        }
        const new_fired = try allocator.alloc([]const u8, events.len);
        errdefer {
            if (new_fired.len > 0) allocator.free(new_fired);
        }

        for (self.events) |ev| {
            if (ev.name.len > 0) self.allocator.free(ev.name);
        }
        if (self.events.len > 0) self.allocator.free(self.events);
        if (self.fired_names.len > 0) self.allocator.free(self.fired_names);
        self.events = new_events;
        self.fired_names = new_fired;
        self.fired_len = 0;
    }

    /// Returns the event names fired by the most recent update() that
    /// advanced time, then clears the queue (a second call without an
    /// intervening firing update returns an empty slice). The slice is only
    /// valid until the next update()/setEvents()/deinit().
    pub fn drainFiredEvents(self: *AnimationGroup) []const []const u8 {
        defer self.fired_len = 0;
        return self.fired_names[0..self.fired_len];
    }

    fn pushFired(self: *AnimationGroup, name: []const u8) void {
        if (self.fired_len < self.fired_names.len) {
            self.fired_names[self.fired_len] = name;
            self.fired_len += 1;
        }
    }

    /// Fires in-range events with the given endpoint inclusivity. Events
    /// outside the group's [from, to] range never fire. Each matching event
    /// fires at most once per call (the caller splits wraparound windows).
    fn fireInRange(self: *AnimationGroup, lo: f32, hi: f32, lo_inclusive: bool, hi_inclusive: bool) void {
        for (self.events) |ev| {
            if (ev.time < self.from or ev.time > self.to) continue;
            const ok_lo = if (lo_inclusive) ev.time >= lo else ev.time > lo;
            const ok_hi = if (hi_inclusive) ev.time <= hi else ev.time < hi;
            if (ok_lo and ok_hi) self.pushFired(ev.name);
        }
    }

    /// Collects the events crossed when advancing from prev_time by advance
    /// seconds into self.current_time (already wrapped/clamped by update()).
    /// Resets the fired snapshot first, so it always reflects this update.
    fn collectEvents(self: *AnimationGroup, prev_time: f32, advance: f32) void {
        self.fired_len = 0;
        if (self.events.len == 0 or advance == 0.0) return;
        const range = self.to - self.from; // > 0, checked by update()
        if (self.speed_ratio >= 0.0) {
            if (advance >= range) {
                self.fireInRange(self.from, self.to, true, true);
                return;
            }
            const curr = self.current_time;
            if (curr >= prev_time) {
                self.fireInRange(prev_time, curr, false, true);
            } else {
                self.fireInRange(prev_time, self.to, false, true);
                self.fireInRange(self.from, curr, true, true);
            }
        } else {
            if (-advance >= range) {
                self.fireInRange(self.from, self.to, true, true);
                return;
            }
            const curr = self.current_time;
            if (curr <= prev_time) {
                self.fireInRange(curr, prev_time, true, false);
            } else {
                self.fireInRange(self.from, prev_time, true, false);
                self.fireInRange(curr, self.to, false, true);
            }
        }
    }

    fn fireEventCallbacks(self: *AnimationGroup) void {
        if (self.on_event) |cb| {
            for (self.fired_names[0..self.fired_len]) |nm| cb(self.event_context, nm);
        }
    }

    pub fn play(self: *AnimationGroup, loop: bool) void {
        self.from = 0.0;
        self.to = self.duration;
        self.loop = loop;
        self.is_playing = true;
        self.fade_duration = 0.0;
        self.fired_len = 0;
    }

    pub fn playRange(self: *AnimationGroup, from: f32, to: f32, loop: bool, speed: ?f32) void {
        self.from = std.math.clamp(from, 0.0, self.duration);
        self.to = std.math.clamp(to, self.from, self.duration);
        if (self.to <= self.from) self.to = self.duration;
        self.loop = loop;
        self.fade_duration = 0.0;
        self.fired_len = 0;
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
        self.fired_len = 0;
        if (self.skeleton) |skel| {
            skel.resetToBindPose();
        }
        self.restoreNodeRestPose();
        self.restoreMorphRestWeights();
    }

    pub fn goToFrame(self: *AnimationGroup, time: f32) void {
        self.current_time = std.math.clamp(time, self.from, self.to);
        // A seek never fires events; it also discards any pending snapshot.
        self.fired_len = 0;
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

        const prev_time = self.current_time;
        const advance = dt * self.speed_ratio;
        self.current_time += advance;

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

        self.collectEvents(prev_time, advance);
        self.applyNodesAtTime(self.current_time);
        self.fireEventCallbacks();
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

// Cubic-spline (Hermite) tests. Per-key layout everywhere below:
// (in-tangent, value, out-tangent); tangents are per-second slopes.

test "Cubic spline vec3 hits keyframe values exactly and clamps" {
    const times = [_]f32{ 0.0, 1.0 };
    // k0: in=(100,0,0) value=(0,0,0) out=(0,0,0);
    // k1: in=(0,0,0) value=(1,2,3) out=(200,0,0).
    const values = [_]f32{
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0, 0.0,
        0.0,   0.0, 0.0, 1.0, 2.0, 3.0, 200.0, 0.0, 0.0,
    };
    const sampler = AnimationSampler{
        .timestamps = &times,
        .outputs = &values,
        .interpolation = .cubic_spline,
    };

    const v0 = sampler.sampleVec3(0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v0.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v0.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v0.z, 1e-6);

    const v1 = sampler.sampleVec3(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), v1.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), v1.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), v1.z, 1e-6);

    // Zero tangents on both sides of the segment -> smoothstep midpoint.
    const mid = sampler.sampleVec3(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mid.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), mid.z, 1e-5);

    const clo = sampler.sampleVec3(-5.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), clo.x, 1e-6);
    const chi = sampler.sampleVec3(99.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), chi.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), chi.z, 1e-6);
}

test "Cubic spline midpoint follows Hermite basis with dt-scaled tangents" {
    const times = [_]f32{ 0.0, 2.0 };
    // x: p0=0, out0=3 (m0 = 3*dt = 6), p1=1, in1=0. At s=0.5:
    // 0.5*0 + 0.125*6 + 0.5*1 - 0.125*0 = 1.25, well above linear 0.5.
    // (Without the dt scaling the answer would be 0.875, not 1.25.)
    const values = [_]f32{
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    const sampler = AnimationSampler{
        .timestamps = &times,
        .outputs = &values,
        .interpolation = .cubic_spline,
    };
    const mid = sampler.sampleVec3(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), mid.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mid.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mid.z, 1e-6);
}

test "Cubic spline rotation is normalized and smooth" {
    const q0 = Quat.identity;
    const q1 = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));
    const times = [_]f32{ 0.0, 1.0 };
    const values = [_]f32{
        0.0, 0.0, 0.0, 0.0, q0.x, q0.y, q0.z, q0.w, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, q1.x, q1.y, q1.z, q1.w, 0.0, 0.0, 0.0, 0.0,
    };
    const sampler = AnimationSampler{
        .timestamps = &times,
        .outputs = &values,
        .interpolation = .cubic_spline,
    };

    const start = sampler.sampleQuat(0.0);
    const dot0 = start.x * q0.x + start.y * q0.y + start.z * q0.z + start.w * q0.w;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @abs(dot0), 1e-5);

    const end = sampler.sampleQuat(1.0);
    const dot1 = end.x * q1.x + end.y * q1.y + end.z * q1.z + end.w * q1.w;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @abs(dot1), 1e-5);

    // Zero tangents: midpoint is normalize((q0+q1)/2), a ~45 deg Y turn.
    const mid = sampler.sampleQuat(0.5);
    const len_sq = mid.x * mid.x + mid.y * mid.y + mid.z * mid.z + mid.w * mid.w;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), len_sq, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 45.0), mid.toEulerDeg().y, 0.5);
}

test "Cubic spline morph weights interpolate per target" {
    const allocator = std.testing.allocator;
    var weights = [_]f32{ 0.0, 0.0 };
    var dirty = false;

    const times = try allocator.alloc(f32, 2);
    times[0] = 0.0;
    times[1] = 1.0;
    // n=2: k0 in=[9,9] val=[0,0] out=[3,0]; k1 in=[0,0] val=[1,0.5] out=[9,9].
    const outputs = try allocator.alloc(f32, 12);
    outputs[0] = 9.0;
    outputs[1] = 9.0;
    outputs[2] = 0.0;
    outputs[3] = 0.0;
    outputs[4] = 3.0;
    outputs[5] = 0.0;
    outputs[6] = 0.0;
    outputs[7] = 0.0;
    outputs[8] = 1.0;
    outputs[9] = 0.5;
    outputs[10] = 9.0;
    outputs[11] = 9.0;

    const ag = try makeNodeGroup(allocator, "morph_cubic", 1.0);
    defer ag.deinit();
    const target = try ag.bindMorphTarget(&weights, &dirty);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .weights,
        .sampler = .{ .timestamps = times, .outputs = outputs, .interpolation = .cubic_spline },
    });

    ag.play(true);
    ag.goToFrame(0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), weights[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), weights[1], 1e-6);

    ag.goToFrame(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), weights[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights[1], 1e-6);

    // Target 0: 0.5*0 + 0.125*3 + 0.5*1 = 0.875; target 1: 0.5*0.5 = 0.25.
    ag.goToFrame(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.875), weights[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), weights[1], 1e-5);
    try std.testing.expect(dirty);
}

test "Truncated cubic sampler never crashes and degrades gracefully" {
    const allocator = std.testing.allocator;
    const times = [_]f32{ 0.0, 1.0 };
    const tiny = [_]f32{ 1.0, 2.0, 3.0 };
    const bad = AnimationSampler{
        .timestamps = &times,
        .outputs = &tiny,
        .interpolation = .cubic_spline,
    };
    // Zero usable frames: vec3 degrades to zero, quat to identity.
    const v = bad.sampleVec3(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v.x, 1e-6);
    const q = bad.sampleQuat(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), q.w, 1e-6);
    var w = [_]f32{ 0.7, 0.8 };
    bad.sampleWeightsInto(0.5, &w, .linear);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), w[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), w[1], 1e-6);

    // Exactly one usable frame: every sample clamps to that key's value.
    const one = [_]f32{ 9.0, 9.0, 9.0, 4.0, 5.0, 6.0, 9.0, 9.0, 9.0 };
    const single = AnimationSampler{
        .timestamps = &times,
        .outputs = &one,
        .interpolation = .cubic_spline,
    };
    const sv = single.sampleVec3(0.75);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), sv.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), sv.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), sv.z, 1e-6);

    // Group path: truncated cubic channels are skipped, never applied.
    var node = TestNode{ .position = Vec3.new(1.0, 2.0, 3.0) };
    const bad_times = try allocator.alloc(f32, 2);
    bad_times[0] = 0.0;
    bad_times[1] = 1.0;
    const bad_out = try allocator.alloc(f32, 3);
    bad_out[0] = 42.0;
    bad_out[1] = 42.0;
    bad_out[2] = 42.0;
    const ag = try makeNodeGroup(allocator, "cubic_guard", 1.0);
    defer ag.deinit();
    const target = try ag.bindNodeTarget(&node.position, &node.rotation, &node.scaling, false);
    try ag.addNodeChannel(.{
        .target = target,
        .target_path = .translation,
        .sampler = .{ .timestamps = bad_times, .outputs = bad_out, .interpolation = .cubic_spline },
    });
    ag.play(true);
    ag.goToFrame(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), node.position.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), node.position.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), node.position.z, 1e-6);
}

// Event tests: counts callback invocations alongside drainFiredEvents().
const EventCounter = struct {
    count: usize = 0,
    last_name: []const u8 = "",
    fn handler(ctx: ?*anyopaque, name: []const u8) void {
        const self: *EventCounter = @ptrCast(@alignCast(ctx.?));
        self.count += 1;
        self.last_name = name;
    }
};

test "Animation events fire once on crossing and drain clears" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "ev_once", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{.{ .time = 1.0, .name = "hit" }});
    var counter = EventCounter{};
    ag.on_event = &EventCounter.handler;
    ag.event_context = &counter;

    ag.play(false);
    ag.update(0.6); // t = 0.6, no crossing
    try std.testing.expectEqual(@as(usize, 0), counter.count);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);

    ag.update(0.6); // t = 1.2, crosses 1.0 exactly once
    try std.testing.expectEqual(@as(usize, 1), counter.count);
    try std.testing.expectEqualStrings("hit", counter.last_name);
    const fired = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 1), fired.len);
    try std.testing.expectEqualStrings("hit", fired[0]);

    // Draining clears: a second drain is empty, and later updates refire nothing.
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
    ag.update(0.5); // t = 1.7, no new crossing
    try std.testing.expectEqual(@as(usize, 1), counter.count);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
}

test "Animation events fire across a loop wraparound" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "ev_loop", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{
        .{ .time = 0.25, .name = "wrapped" },
        .{ .time = 1.5, .name = "end" },
        .{ .time = 0.0, .name = "start" },
    });

    ag.play(true);
    ag.update(1.0); // t = 1.0, crosses nothing
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);

    ag.update(1.0); // t = 2.0, window (1, 2] fires "end"
    const f1 = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 1), f1.len);
    try std.testing.expectEqualStrings("end", f1[0]);

    // Wraps to t = 1.0: windows (2, 2] (empty) + [0, 1] fire "wrapped" and
    // the zero-boundary "start", each exactly once, in stored order.
    ag.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ag.current_time, 1e-4);
    const f2 = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 2), f2.len);
    try std.testing.expectEqualStrings("wrapped", f2[0]);
    try std.testing.expectEqualStrings("start", f2[1]);
}

test "Animation events fire when playing backward" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "ev_back", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{.{ .time = 1.0, .name = "hit" }});

    ag.play(true);
    ag.setSpeed(-1.0);
    ag.goToFrame(1.5); // seek: must not fire
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);

    ag.update(1.0); // t = 0.5, window [0.5, 1.5) fires "hit" once
    const f1 = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 1), f1.len);
    try std.testing.expectEqualStrings("hit", f1[0]);

    // Wraps to t = 1.5: windows [0, 0.5) + (1.5, 2] miss 1.0, no refire.
    ag.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), ag.current_time, 1e-4);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
}

test "Animation events fire on the duration boundary" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "ev_edge", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{.{ .time = 2.0, .name = "end" }});

    ag.play(false);
    ag.update(2.5); // overshoots to == 2.0, non-loop stops and still fires
    try std.testing.expect(!ag.is_playing);
    const f1 = ag.drainFiredEvents();
    try std.testing.expectEqual(@as(usize, 1), f1.len);
    try std.testing.expectEqualStrings("end", f1[0]);

    ag.update(1.0); // stopped: nothing may fire afterwards
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
}

test "stop and fade-out never fire events" {
    const allocator = std.testing.allocator;
    const ag = try makeNodeGroup(allocator, "ev_nofire", 2.0);
    defer ag.deinit();
    try ag.setEvents(allocator, &.{
        .{ .time = 0.5, .name = "early" },
        .{ .time = 1.0, .name = "late" },
    });

    ag.play(true);
    ag.update(0.4); // t = 0.4, no crossing yet
    ag.stop(); // must not fire and must drop the pending snapshot
    try std.testing.expect(!ag.is_playing);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);

    // A fade-out that completes inside this update stops playback via the
    // early-return path: time would cross both events, but none may fire.
    ag.play(true);
    ag.fadeOut(0.5);
    ag.update(1.5);
    try std.testing.expect(!ag.is_playing);
    try std.testing.expectEqual(@as(usize, 0), ag.drainFiredEvents().len);
}
