const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Skeleton = @import("skeleton.zig").Skeleton;
const MAX_BONES = @import("skeleton.zig").MAX_BONES;
const easing_mod = @import("easing.zig");

const sampler_mod = @import("sampler.zig");
pub const AnimationEvent = sampler_mod.AnimationEvent;

const channels_mod = @import("channels.zig");
pub const AnimationChannel = channels_mod.AnimationChannel;
pub const NodeTarget = channels_mod.NodeTarget;
pub const NodeChannel = channels_mod.NodeChannel;
pub const MorphWeightsTarget = channels_mod.MorphWeightsTarget;
const samplerHasFrames = channels_mod.samplerHasFrames;

pub const AnimationGroup = struct {
    allocator: std.mem.Allocator,
    name: []const u8 = "",
    channels: []AnimationChannel,
    skeleton: ?*Skeleton = null,
    duration: f32 = 0.0,
    /// Per-bone routing over `channels`, built serially by update().
    /// `chan_route` packs channel indices grouped by bone_index (< MAX_BONES)
    /// in original channel order; `chan_head`/`chan_count` delimit each
    /// bone's run. Channels with bone_index >= MAX_BONES are not routed and
    /// keep the legacy scan (see sampleBoneIndexed). Rebuilds allocate only
    /// when the channel set actually changes, so steady-state frames
    /// allocate nothing.
    chan_route: []u32 = &.{},
    chan_head: [MAX_BONES]u32 = [_]u32{0} ** MAX_BONES,
    chan_count: [MAX_BONES]u32 = [_]u32{0} ** MAX_BONES,
    /// Exact bone_index per channel position at route-build time. The route
    /// depends only on the bone sequence (bucket assignment) and channel
    /// order (duplicate LAST-wins): target_path, interpolation and sampler
    /// contents are read live per sample, so only bone changes can stale it.
    /// validateChannelRoute() compares this against the live channels —
    /// an exact O(C) check, not a hash — so same-ptr/len in-place edits
    /// (bone moves, cross-bone swaps, slice replacement) fail validation
    /// and fall back to the exact legacy scan. No caller ever needs to
    /// invalidate: currency is re-proven, never assumed.
    chan_route_bones: []usize = &.{},
    chan_route_valid: bool = false,
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
        errdefer allocator.destroy(ag);
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
        if (self.chan_route.len > 0) self.allocator.free(self.chan_route);
        if (self.chan_route_bones.len > 0) self.allocator.free(self.chan_route_bones);
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
        // Finished one-shots replay: reseat only an endpoint cursor, so a
        // mid-range pause()/goToFrame() cursor keeps resuming. Forward
        // restarts when at/past to, reverse when at/below from (mirrors the
        // playRange out-of-range seating, but play() always owns [0, to]).
        if (self.speed_ratio >= 0.0) {
            if (self.current_time < self.from or self.current_time >= self.to) {
                self.current_time = self.from;
            }
        } else {
            if (self.current_time <= self.from or self.current_time > self.to) {
                self.current_time = self.to;
            }
        }
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

        // Serial-only: (re)build the per-bone channel route before any
        // concurrent pose evaluation can read it (see sampleBoneAtTime).
        // update() callers are serial by contract — animation_runtime runs
        // this loop before the parallel eval, and it never runs in workers.
        self.refreshChannelIndex();

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
                    // Skeleton channels are blended later in
                    // scene/animation_runtime (collectActive skips stopped /
                    // zero-weight groups, eval returns on zero active clips),
                    // so without this the skeleton would retain its last pose
                    // while nodes/morphs below restore to rest. Mirror stop().
                    if (self.skeleton) |skel| {
                        skel.resetToBindPose();
                    }
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
                    // sampleQuatEased and slerp already return unit quats,
                    // and toEulerDeg normalizes internally: no extra
                    // normalize pass needed (it only burned a sqrt).
                    if (w >= 0.999) {
                        t.rotation_euler.* = sampled.toEulerDeg();
                    } else {
                        const blended = Quat.slerp(t.rest_rotation, sampled, w);
                        t.rotation_euler.* = blended.toEulerDeg();
                    }
                },
                // Handled above; kept for exhaustiveness.
                .weights => {},
            }
        }
    }

    /// True while the per-bone channel route matches the live channels
    /// (exact bone-sequence check, see validateChannelRoute). Introspection
    /// for tests and diagnostics; evaluation consults it once per group.
    pub fn hasValidChannelIndex(self: *const AnimationGroup) bool {
        return self.validateChannelRoute();
    }

    /// Exact O(C) route-currency check: stored bone sequence vs live
    /// channels. Read-only (no allocation, no mutation): safe to call from
    /// anywhere, including once per group per evaluation before taking the
    /// indexed path. This is a full comparison, not a hash — same-ptr/len
    /// in-place edits are always detected.
    pub fn validateChannelRoute(self: *const AnimationGroup) bool {
        if (!self.chan_route_valid) return false;
        if (self.chan_route_bones.len != self.channels.len) return false;
        for (self.chan_route_bones, 0..) |want, i| {
            if (want != self.channels[i].bone_index) return false;
        }
        return true;
    }

    /// Serial-only route maintenance; see the chan_route field docs. Keeps
    /// the current route when the bone sequence still matches (O(C) reads,
    /// no writes, no allocs), otherwise rebuilds exactly. OOM leaves the
    /// route invalid and every sampler keeps the exact legacy scan —
    /// behavior never changes, only speed. Must never run concurrently
    /// (no group mutation during parallel pose evaluation); update() is
    /// the caller.
    fn refreshChannelIndex(self: *AnimationGroup) void {
        if (self.chan_route_valid and self.validateChannelRoute()) return;
        self.chan_route_valid = false;
        self.chan_head = [_]u32{0} ** MAX_BONES;
        self.chan_count = [_]u32{0} ** MAX_BONES;
        // u32 indices below are provably in range only past this guard.
        if (self.channels.len > std.math.maxInt(u32)) return;
        var routable: usize = 0;
        for (self.channels) |ch| {
            if (ch.bone_index < MAX_BONES) {
                self.chan_count[ch.bone_index] += 1;
                routable += 1;
            }
        }
        // Exact-size (re)allocation: rebuilds run only when the channel set
        // actually changed, so steady-state frames allocate nothing.
        if (self.chan_route.len > 0) self.allocator.free(self.chan_route);
        if (self.chan_route_bones.len > 0) self.allocator.free(self.chan_route_bones);
        self.chan_route = &.{};
        self.chan_route_bones = &.{};
        if (routable > 0) {
            self.chan_route = self.allocator.alloc(u32, routable) catch return;
        }
        if (self.channels.len > 0) {
            self.chan_route_bones = self.allocator.alloc(usize, self.channels.len) catch {
                // Keep the pair consistent: a half-built route is invalid.
                if (self.chan_route.len > 0) self.allocator.free(self.chan_route);
                self.chan_route = &.{};
                return;
            };
        }
        var cursor: [MAX_BONES]u32 = undefined;
        var off: u32 = 0;
        for (0..MAX_BONES) |b| {
            self.chan_head[b] = off;
            cursor[b] = off;
            off += self.chan_count[b];
        }
        for (self.channels, 0..) |ch, i| {
            self.chan_route_bones[i] = ch.bone_index;
            if (ch.bone_index < MAX_BONES) {
                const b = ch.bone_index;
                self.chan_route[cursor[b]] = @intCast(i);
                cursor[b] += 1;
            }
        }
        self.chan_route_valid = true;
    }

    /// One channel's contribution to a bone sample: malformed/truncated
    /// LINEAR/STEP tracks are skipped (outputs stay null), CUBICSPLINE keeps
    /// its sampler fallback. Shared verbatim by the legacy scan and the
    /// routed path so both visit identical channels with identical guards.
    inline fn sampleChannelBone(ch: *const AnimationChannel, time: f32, out_pos: *?Vec3, out_rot: *?Quat, out_scale: *?Vec3) void {
        if (ch.sampler.interpolation != .cubic_spline and !samplerHasFrames(ch.sampler, ch.target_path, 0)) return;
        switch (ch.target_path) {
            .translation => out_pos.* = ch.sampler.sampleVec3(time),
            .rotation => out_rot.* = ch.sampler.sampleQuat(time),
            .scale => out_scale.* = ch.sampler.sampleVec3(time),
            .weights => {},
        }
    }

    /// Legacy scan over the live channels: always current, before or after
    /// any update(), under any mutation. This is the behavior-preserving
    /// default for direct callers; the indexed path is taken only inside
    /// evaluated skeletons after once-per-eval validation (see
    /// sampleBoneIndexed), so public sampling honors current channels
    /// immediately, even before the next update().
    pub fn sampleBoneAtTime(self: *const AnimationGroup, bone_idx: usize, time: f32, out_pos: *?Vec3, out_rot: *?Quat, out_scale: *?Vec3) void {
        self.sampleBoneAtTimeLegacy(bone_idx, time, out_pos, out_rot, out_scale);
    }

    /// Trusted indexed sampler: the bone's routed run, no currency check.
    /// Call only when validateChannelRoute() is true and no channel
    /// mutation intervenes — evaluateSkeletonInner upholds this with a
    /// once-per-group-per-eval validation and no group writes during the
    /// eval. Oversized bones fall back to the legacy scan internally
    /// (single branch, always correct), so evaluated skeletons past
    /// MAX_BONES stay exact.
    pub fn sampleBoneIndexed(self: *const AnimationGroup, bone_idx: usize, time: f32, out_pos: *?Vec3, out_rot: *?Quat, out_scale: *?Vec3) void {
        if (bone_idx >= MAX_BONES) {
            self.sampleBoneAtTimeLegacy(bone_idx, time, out_pos, out_rot, out_scale);
            return;
        }
        // Bounds hold exactly when the route is current (trusted caller):
        // head+count <= routable <= chan_route.len by rebuild construction.
        const h: usize = self.chan_head[bone_idx];
        const e: usize = h + self.chan_count[bone_idx];
        for (self.chan_route[h..e]) |ci| {
            sampleChannelBone(&self.channels[ci], time, out_pos, out_rot, out_scale);
        }
    }

    /// Explicit legacy scan: every channel in order, bone filter inline.
    /// Bit-identical to the pre-index sampler; kept for direct callers that
    /// need index-independent behavior and as the parity baseline in tests.
    pub fn sampleBoneAtTimeLegacy(self: *const AnimationGroup, bone_idx: usize, time: f32, out_pos: *?Vec3, out_rot: *?Quat, out_scale: *?Vec3) void {
        for (self.channels) |ch| {
            if (ch.bone_index != bone_idx) continue;
            sampleChannelBone(&ch, time, out_pos, out_rot, out_scale);
        }
    }

    pub fn applyAtTime(self: *AnimationGroup, time: f32) void {
        self.applySkeletonAtTime(time);
        self.applyNodesAtTime(time);
    }

    /// Skeleton-pose half of applyAtTime: writes bone local TRS from the
    /// group's skeleton channels and recomputes model/skin matrices. Never
    /// touches node/morph targets, so calls for distinct skeletons are
    /// index-disjoint (see evaluateSkeletonPoseOnly in eval.zig).
    pub fn applySkeletonAtTime(self: *AnimationGroup, time: f32) void {
        if (self.skeleton) |skel| {
            for (self.channels) |ch| {
                if (ch.bone_index >= skel.bones.len) continue;
                // Skip malformed/truncated LINEAR/STEP tracks: preserve current pose.
                // CUBICSPLINE keeps its sampler fallback (usable frame/zero), so it is not guarded.
                if (ch.sampler.interpolation != .cubic_spline and !samplerHasFrames(ch.sampler, ch.target_path, 0)) continue;
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
