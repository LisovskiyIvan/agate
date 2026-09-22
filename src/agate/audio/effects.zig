//! Bus DSP effects: biquad filters, reverb units and occlusion state.
//! Split out of `audio.zig` (facade).
//!
//! This module owns the per-bus effect registry: `setBusFilter*`/
//! `clearBusFilter`/`getBusFilter` (biquad lowpass/highpass/bandpass), the
//! pooled reverb units (`setBusReverb`/`clearBusReverb`/`getBusReverb`/
//! `isBusReverbEnabled` plus the underwater/muffled/telephone/cave/room
//! presets), and the occlusion state (`setBusOcclusion*`/
//! `clearBusOcclusion`/`updateBusOcclusion`, `evaluateOcclusion`/
//! `updateBusOcclusionWithRaycast` wiring the `occlusion.zig` raycast
//! evaluator to a bus). The registry (`buses.zig`) reaches these through
//! direct sibling imports when creating/configuring/destroying a bus.
//!
//! Anti-cycle rule (same as `profiler/*`, `particles/*`): every function
//! takes the engine as `anytype` (a `*AudioEngine` from `engine.zig` in
//! practice) and this module never imports `engine.zig` or the `audio.zig`
//! facade back. Effect parameter types come from the pre-existing `dsp.zig`
//! and `occlusion.zig` leaves.

const std = @import("std");
const math = @import("math");
const dsp = @import("dsp.zig");
const occlusion_mod = @import("occlusion.zig");
const types = @import("types.zig");

const Vec3 = math.Vec3;
const BusId = types.BusId;
const BiquadFilterType = dsp.BiquadFilterType;
const BusFilterConfig = dsp.BusFilterConfig;
const BusReverbConfig = dsp.BusReverbConfig;
const AudioOcclusionConfig = occlusion_mod.AudioOcclusionConfig;
const RaycastFn = occlusion_mod.RaycastFn;
const evaluateRaycastOcclusion = occlusion_mod.evaluateRaycastOcclusion;

pub fn setBusFilter(self: anytype, bus: BusId, filter_type: BiquadFilterType, cutoff: f32, q: f32) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    self.bus_filter_type[id].store(@intFromEnum(filter_type), .release);
    self.bus_filter_cutoff[id].store(@bitCast(cutoff), .release);
    self.bus_filter_q[id].store(@bitCast(q), .release);
    self.bus_filters[id].setParams(filter_type, cutoff, q, self.sample_rate);
}

pub fn setBusFilterCutoff(self: anytype, bus: BusId, cutoff: f32) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    self.bus_filter_cutoff[id].store(@bitCast(cutoff), .release);
    const f_type: BiquadFilterType = @enumFromInt(self.bus_filter_type[id].load(.acquire));
    const q: f32 = @bitCast(self.bus_filter_q[id].load(.acquire));
    self.bus_filters[id].setParams(f_type, cutoff, q, self.sample_rate);
}

pub fn setBusFilterQ(self: anytype, bus: BusId, q: f32) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    self.bus_filter_q[id].store(@bitCast(q), .release);
    const f_type: BiquadFilterType = @enumFromInt(self.bus_filter_type[id].load(.acquire));
    const cutoff: f32 = @bitCast(self.bus_filter_cutoff[id].load(.acquire));
    self.bus_filters[id].setParams(f_type, cutoff, q, self.sample_rate);
}

pub fn clearBusFilter(self: anytype, bus: BusId) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    self.bus_filter_type[id].store(@intFromEnum(BiquadFilterType.none), .release);
    self.bus_filters[id].setParams(.none, 1000.0, 0.7071, self.sample_rate);
    self.bus_filters[id].resetState();
}

pub fn getBusFilter(self: anytype, bus: BusId) BusFilterConfig {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return .{};
    return .{
        .filter_type = @enumFromInt(self.bus_filter_type[id].load(.acquire)),
        .cutoff = @bitCast(self.bus_filter_cutoff[id].load(.acquire)),
        .q = @bitCast(self.bus_filter_q[id].load(.acquire)),
    };
}

pub fn setBusReverb(self: anytype, bus: BusId, config: BusReverbConfig) bool {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return false;
    if (!self.bus_active[id].load(.acquire)) return false;

    const current_slot = self.bus_reverb_slot[id].load(.acquire);
    if (current_slot < types.max_reverbs) {
        self.reverbs[current_slot].setConfig(config);
        self.reverbs[current_slot].active = true;
        return true;
    }

    // Allocate free reverb unit from pool
    for (0..types.max_reverbs) |slot| {
        if (self.reverb_bus_owner[slot].load(.acquire) == 0xFF) {
            self.reverb_bus_owner[slot].store(@intCast(id), .release);
            self.bus_reverb_slot[id].store(@intCast(slot), .release);
            self.reverbs[slot].setSampleRate(self.sample_rate);
            self.reverbs[slot].setConfig(config);
            self.reverbs[slot].active = true;
            return true;
        }
    }
    return false;
}

pub fn clearBusReverb(self: anytype, bus: BusId) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    const slot = self.bus_reverb_slot[id].load(.acquire);
    if (slot < types.max_reverbs) {
        self.reverbs[slot].active = false;
        self.reverbs[slot].clear();
        self.reverb_bus_owner[slot].store(0xFF, .release);
        self.bus_reverb_slot[id].store(0xFF, .release);
    }
}

pub fn getBusReverb(self: anytype, bus: BusId) ?BusReverbConfig {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return null;
    const slot = self.bus_reverb_slot[id].load(.acquire);
    if (slot >= types.max_reverbs) return null;
    return self.reverbs[slot].config;
}

pub fn isBusReverbEnabled(self: anytype, bus: BusId) bool {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return false;
    return self.bus_reverb_slot[id].load(.acquire) < types.max_reverbs;
}

pub fn setBusUnderwater(self: anytype, bus: BusId) void {
    setBusFilter(self, bus, .lowpass, 600.0, 1.0);
}

pub fn setBusMuffled(self: anytype, bus: BusId) void {
    setBusFilter(self, bus, .lowpass, 1200.0, 0.7071);
}

pub fn setBusTelephone(self: anytype, bus: BusId) void {
    setBusFilter(self, bus, .bandpass, 1800.0, 1.5);
}

pub fn setBusCaveReverb(self: anytype, bus: BusId) bool {
    return setBusReverb(self, bus, .{ .room_size = 0.85, .damping = 0.25, .wet = 0.5, .dry = 0.7 });
}

pub fn setBusRoomReverb(self: anytype, bus: BusId) bool {
    return setBusReverb(self, bus, .{ .room_size = 0.45, .damping = 0.5, .wet = 0.25, .dry = 0.85 });
}

pub fn setBusOcclusion(self: anytype, bus: BusId, occlusion: f32) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    const val = std.math.clamp(occlusion, 0.0, 1.0);
    self.bus_occlusion[id].store(@bitCast(val), .release);
}

pub fn getBusOcclusion(self: anytype, bus: BusId) f32 {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return 0.0;
    return @bitCast(self.bus_occlusion[id].load(.acquire));
}

pub fn setBusOcclusionConfig(self: anytype, bus: BusId, config: AudioOcclusionConfig) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    self.bus_occlusion_config[id] = config;
}

pub fn getBusOcclusionConfig(self: anytype, bus: BusId) AudioOcclusionConfig {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return .{};
    return self.bus_occlusion_config[id];
}

pub fn clearBusOcclusion(self: anytype, bus: BusId) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    self.bus_occlusion[id].store(@bitCast(@as(f32, 0.0)), .release);
    self.bus_occlusion_filters[id].resetState();
}

pub fn updateBusOcclusion(self: anytype, bus: BusId, target_occlusion: f32, dt: f32) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    const target = std.math.clamp(target_occlusion, 0.0, 1.0);
    const smooth_time = self.bus_occlusion_config[id].smooth_time;
    const cur = getBusOcclusion(self, bus);
    if (smooth_time <= 1e-4 or dt <= 1e-4) {
        setBusOcclusion(self, bus, target);
        return;
    }
    const alpha = 1.0 - @exp(-dt / @max(smooth_time, 0.001));
    const new_val = cur + (target - cur) * std.math.clamp(alpha, 0.0, 1.0);
    setBusOcclusion(self, bus, new_val);
}

pub fn evaluateOcclusion(
    self: anytype,
    emitter_pos: Vec3,
    config: AudioOcclusionConfig,
    raycast_fn: RaycastFn,
    user_data: ?*anyopaque,
) f32 {
    return evaluateRaycastOcclusion(self.listener_pos, emitter_pos, config, raycast_fn, user_data);
}

pub fn updateBusOcclusionWithRaycast(
    self: anytype,
    bus: BusId,
    emitter_pos: Vec3,
    dt: f32,
    raycast_fn: RaycastFn,
    user_data: ?*anyopaque,
) f32 {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return 0.0;
    const cfg = self.bus_occlusion_config[id];
    const raw = evaluateOcclusion(self, emitter_pos, cfg, raycast_fn, user_data);
    updateBusOcclusion(self, bus, raw, dt);
    return getBusOcclusion(self, bus);
}
