//! Bus registry and routing DAG: create/configure/destroy/find + cascade.
//! Split out of `audio.zig` (facade).
//!
//! This module owns bus lifecycle (`createBus`, `createSpatialBus`,
//! `createNonSpatialBus`, `configureBus`, `destroyBus`, `findBus`,
//! `findOrCreateBus`), identity (`isBusActive`, `getBusCount`,
//! `isBusSpatial`/`setBusSpatial`, `getBusName`/`setBusName`), hierarchy
//! (`setBusParent`/`getBusParent`), spatial parameters
//! (`setBusAttenuation`/`getBusAttenuation`, `setBusDopplerFactor`/
//! `getBusDopplerFactor`), and the volume/mute cascade
//! (`setBusVolume`/`getBusVolume`/`getBusEffectiveVolume`, `setBusMuted`/
//! `isBusMuted`/`toggleBusMuted`, `stopBus`). Effect wiring on
//! create/configure/destroy reaches the `effects.zig` sibling directly.
//!
//! Anti-cycle rule (same as `profiler/*`, `particles/*`): every function
//! takes the engine as `anytype` (a `*AudioEngine` from `engine.zig` in
//! practice) and this module never imports `engine.zig` or the `audio.zig`
//! facade back. Capacity comes from the engine-resident `getBusCapacity`
//! through `anytype`; everything else is fields or the `effects` sibling.

const std = @import("std");
const types = @import("types.zig");
const effects = @import("effects.zig");

const BusId = types.BusId;
const BusConfig = types.BusConfig;
const BusAttenuation = types.BusAttenuation;
const AttenuationModel = types.AttenuationModel;

// Local aliases so the moved bodies stay byte-identical.
const setBusFilter = effects.setBusFilter;
const setBusReverb = effects.setBusReverb;
const clearBusFilter = effects.clearBusFilter;
const clearBusReverb = effects.clearBusReverb;
const clearBusOcclusion = effects.clearBusOcclusion;
const setBusOcclusion = effects.setBusOcclusion;
const setBusOcclusionConfig = effects.setBusOcclusionConfig;

pub fn createBus(self: anytype, config: BusConfig) ?BusId {
    var slot: ?usize = null;
    const cap = self.getBusCapacity();
    for (0..cap) |i| {
        if (!self.bus_active[i].load(.acquire)) {
            slot = i;
            break;
        }
    }
    const idx = slot orelse return null;
    setBusNameRaw(self, idx, config.name);
    self.bus_spatial[idx].store(config.spatial, .release);
    self.bus_volumes[idx].store(@bitCast(std.math.clamp(config.volume, 0.0, 2.0)), .release);
    self.bus_muted[idx].store(config.muted, .release);
    const parent_id: u8 = if (config.parent) |p| @intFromEnum(p) else 0xFF;
    self.bus_parent[idx].store(parent_id, .release);
    self.bus_model[idx].store(@intFromEnum(config.attenuation_model), .release);
    self.bus_min_dist[idx].store(@bitCast(@max(0.001, config.min_distance)), .release);
    self.bus_max_dist[idx].store(@bitCast(@max(0.002, config.max_distance)), .release);
    self.bus_rolloff[idx].store(@bitCast(@max(0.0, config.rolloff)), .release);
    self.bus_doppler[idx].store(@bitCast(std.math.clamp(config.doppler_factor, 0.0, 5.0)), .release);
    self.bus_active[idx].store(true, .release);
    const idx_bus = @as(BusId, @enumFromInt(@as(u8, @intCast(idx))));
    if (config.filter.filter_type != .none) {
        setBusFilter(self, idx_bus, config.filter.filter_type, config.filter.cutoff, config.filter.q);
    }
    if (config.reverb) |rev_cfg| {
        _ = setBusReverb(self, idx_bus, rev_cfg);
    }
    return idx_bus;
}

pub fn createSpatialBus(self: anytype, name: []const u8, config: ?BusConfig) ?BusId {
    var cfg = config orelse BusConfig{};
    cfg.name = name;
    cfg.spatial = true;
    return createBus(self, cfg);
}

pub fn createNonSpatialBus(self: anytype, name: []const u8, volume: f32) ?BusId {
    return createBus(self, .{
        .name = name,
        .spatial = false,
        .volume = volume,
    });
}

pub fn configureBus(self: anytype, bus: BusId, config: BusConfig) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    if (config.name.len > 0) {
        setBusNameRaw(self, id, config.name);
    }
    self.bus_spatial[id].store(config.spatial, .release);
    self.bus_volumes[id].store(@bitCast(std.math.clamp(config.volume, 0.0, 2.0)), .release);
    self.bus_muted[id].store(config.muted, .release);
    const parent_id: u8 = if (config.parent) |p| @intFromEnum(p) else 0xFF;
    self.bus_parent[id].store(parent_id, .release);
    self.bus_model[id].store(@intFromEnum(config.attenuation_model), .release);
    self.bus_min_dist[id].store(@bitCast(@max(0.001, config.min_distance)), .release);
    self.bus_max_dist[id].store(@bitCast(@max(0.002, config.max_distance)), .release);
    self.bus_rolloff[id].store(@bitCast(@max(0.0, config.rolloff)), .release);
    self.bus_doppler[id].store(@bitCast(std.math.clamp(config.doppler_factor, 0.0, 5.0)), .release);
    if (config.filter.filter_type != .none) {
        setBusFilter(self, bus, config.filter.filter_type, config.filter.cutoff, config.filter.q);
    }
    if (config.reverb) |rev_cfg| {
        _ = setBusReverb(self, bus, rev_cfg);
    }
    setBusOcclusionConfig(self, bus, config.occlusion_config);
    setBusOcclusion(self, bus, config.occlusion);
}

pub fn destroyBus(self: anytype, bus: BusId) void {
    const id = @intFromEnum(bus);
    const cap = self.getBusCapacity();
    if (id >= cap) return;
    if (!self.bus_active[id].load(.acquire)) return;
    stopBus(self, bus);
    clearBusFilter(self, bus);
    clearBusReverb(self, bus);
    clearBusOcclusion(self, bus);
    const my_parent = self.bus_parent[id].load(.acquire);
    for (0..cap) |i| {
        if (self.bus_active[i].load(.acquire) and self.bus_parent[i].load(.acquire) == id) {
            self.bus_parent[i].store(my_parent, .release);
        }
    }
    self.bus_active[id].store(false, .release);
    self.bus_name_lens[id] = 0;
}

pub fn findBus(self: anytype, name: []const u8) ?BusId {
    const cap = self.getBusCapacity();
    for (0..cap) |i| {
        if (self.bus_active[i].load(.acquire)) {
            const len = self.bus_name_lens[i];
            if (std.mem.eql(u8, self.bus_names[i][0..len], name)) {
                return @enumFromInt(@as(u8, @intCast(i)));
            }
        }
    }
    return null;
}

pub fn findOrCreateBus(self: anytype, name: []const u8, config: BusConfig) ?BusId {
    if (findBus(self, name)) |b| return b;
    var cfg = config;
    cfg.name = name;
    return createBus(self, cfg);
}

pub fn isBusActive(self: anytype, bus: BusId) bool {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return false;
    return self.bus_active[id].load(.acquire);
}

pub fn getBusCount(self: anytype) usize {
    var count: usize = 0;
    const cap = self.getBusCapacity();
    for (0..cap) |i| {
        if (self.bus_active[i].load(.acquire)) count += 1;
    }
    return count;
}

pub fn isBusSpatial(self: anytype, bus: BusId) bool {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return false;
    return self.bus_spatial[id].load(.acquire);
}

pub fn setBusSpatial(self: anytype, bus: BusId, spatial: bool) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    self.bus_spatial[id].store(spatial, .release);
}

pub fn getBusName(self: anytype, bus: BusId) []const u8 {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return "";
    const len = self.bus_name_lens[id];
    return self.bus_names[id][0..len];
}

pub fn setBusName(self: anytype, bus: BusId, name: []const u8) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    setBusNameRaw(self, id, name);
}

fn setBusNameRaw(self: anytype, idx: usize, name: []const u8) void {
    if (idx >= self.getBusCapacity()) return;
    const len = @min(name.len, 31);
    @memcpy(self.bus_names[idx][0..len], name[0..len]);
    self.bus_name_lens[idx] = @intCast(len);
}

pub fn setBusParent(self: anytype, bus: BusId, parent: ?BusId) void {
    const id = @intFromEnum(bus);
    const cap = self.getBusCapacity();
    if (id >= cap) return;
    if (parent) |p| {
        if (@intFromEnum(p) >= cap) return;
    }
    const p_val: u8 = if (parent) |p| @intFromEnum(p) else 0xFF;
    self.bus_parent[id].store(p_val, .release);
}

pub fn getBusParent(self: anytype, bus: BusId) ?BusId {
    const id = @intFromEnum(bus);
    const cap = self.getBusCapacity();
    if (id >= cap) return null;
    const p = self.bus_parent[id].load(.acquire);
    if (p >= cap) return null;
    return @enumFromInt(p);
}

pub fn setBusAttenuation(self: anytype, bus: BusId, model: AttenuationModel, min_dist: f32, max_dist: f32, rolloff: f32) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    self.bus_model[id].store(@intFromEnum(model), .release);
    self.bus_min_dist[id].store(@bitCast(@max(0.001, min_dist)), .release);
    self.bus_max_dist[id].store(@bitCast(@max(0.002, max_dist)), .release);
    self.bus_rolloff[id].store(@bitCast(@max(0.0, rolloff)), .release);
}

pub fn getBusAttenuation(self: anytype, bus: BusId) BusAttenuation {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return .{ .model = .inverse, .min_distance = 1.0, .max_distance = 30.0, .rolloff = 1.0 };
    return .{
        .model = @enumFromInt(self.bus_model[id].load(.acquire)),
        .min_distance = @bitCast(self.bus_min_dist[id].load(.acquire)),
        .max_distance = @bitCast(self.bus_max_dist[id].load(.acquire)),
        .rolloff = @bitCast(self.bus_rolloff[id].load(.acquire)),
    };
}

pub fn setBusDopplerFactor(self: anytype, bus: BusId, factor: f32) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    self.bus_doppler[id].store(@bitCast(std.math.clamp(factor, 0.0, 5.0)), .release);
}

pub fn getBusDopplerFactor(self: anytype, bus: BusId) f32 {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return 1.0;
    return @bitCast(self.bus_doppler[id].load(.acquire));
}

pub fn setBusVolume(self: anytype, bus: BusId, vol: f32) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    const v = std.math.clamp(vol, 0.0, 2.0);
    self.bus_volumes[id].store(@bitCast(v), .release);
}

pub fn getBusVolume(self: anytype, bus: BusId) f32 {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return 0.0;
    return @bitCast(self.bus_volumes[id].load(.acquire));
}

pub fn getBusEffectiveVolume(self: anytype, bus_opt: ?BusId) f32 {
    const bus = bus_opt orelse return 1.0;
    const b_id = @intFromEnum(bus);
    const cap = self.getBusCapacity();
    if (b_id >= cap) return 0.0;
    if (!self.bus_active[b_id].load(.acquire)) return 0.0;

    var current = bus;
    var vol: f32 = 1.0;
    var hops: usize = 0;
    while (hops < 16) : (hops += 1) {
        const id = @intFromEnum(current);
        if (id >= cap) break;
        if (!self.bus_active[id].load(.acquire)) return 0.0;
        if (self.bus_muted[id].load(.acquire)) return 0.0;
        const cur_vol: f32 = @bitCast(self.bus_volumes[id].load(.acquire));
        vol *= cur_vol;

        const p_id = self.bus_parent[id].load(.acquire);
        if (p_id >= cap or p_id == id) break;
        current = @enumFromInt(p_id);
    }
    return vol;
}

pub fn setBusMuted(self: anytype, bus: BusId, muted_val: bool) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    self.bus_muted[id].store(muted_val, .release);
}

pub fn isBusMuted(self: anytype, bus: BusId) bool {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return true;
    return self.bus_muted[id].load(.acquire);
}

pub fn toggleBusMuted(self: anytype, bus: BusId) void {
    const id = @intFromEnum(bus);
    if (id >= self.getBusCapacity()) return;
    var cur = self.bus_muted[id].load(.monotonic);
    while (self.bus_muted[id].cmpxchgWeak(cur, !cur, .acq_rel, .monotonic)) |next| {
        cur = next;
    }
}

pub fn stopBus(self: anytype, bus: BusId) void {
    for (&self.voices) |*v| {
        if (v.bus) |b| {
            if (b == bus) {
                v.active = false;
            }
        }
    }
}
