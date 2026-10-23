//! Shared audio vocabulary: bus handles, configs and voice layouts.
//! Split out of `audio.zig` (facade).
//!
//! This module owns every structural type the engine and its siblings name
//! without owning behavior: the `BusId` handle (+ `AudioBus` alias,
//! capacity constants, `invalid_bus`), `AudioConfig`/`AudioEngineConfig`,
//! the attenuation vocabulary (`AttenuationModel`, `BusAttenuation`,
//! `BusConfig`), the procedural `VoiceKind`/`Voice` slot layout, and the
//! `AudioEngine` sizing limits (`max_voices`, `max_distance`,
//! `max_commands`, `max_reverbs`, `chunk_frames`/`chunk_samples`,
//! `max_streams`). `AudioEngine` re-exports the limits as nested constants
//! (`AudioEngine.max_voices`, ...) so the historical paths keep working.
//!
//! Anti-cycle rule (same as `particles/types.zig`, `profiler/types.zig`):
//! this module imports only the pre-existing DSP/clip leaves (`dsp.zig`,
//! `occlusion.zig`, `clip.zig`) for field types and never the `audio.zig`
//! facade or the `engine.zig` owner back. `audio.zig` re-exports everything
//! here under its historical paths.

const std = @import("std");
const dsp = @import("dsp.zig");
const occlusion_mod = @import("occlusion.zig");
const clip_mod = @import("clip.zig");

const BusFilterConfig = dsp.BusFilterConfig;
const BusReverbConfig = dsp.BusReverbConfig;
const AudioOcclusionConfig = occlusion_mod.AudioOcclusionConfig;
const AudioClip = clip_mod.AudioClip;

pub const BusId = enum(u8) {
    _,

    pub const invalid: BusId = @fromBackingInt(0xFF);

    pub fn id(self: BusId) u8 {
        return @backingInt(self);
    }

    pub fn fromInt(val: u8) BusId {
        return @fromBackingInt(val);
    }
};

pub const AudioBus = BusId;
pub const default_max_buses: usize = 32;
pub const max_bus_capacity: usize = 128;
pub const max_buses: usize = default_max_buses;
pub const invalid_bus: BusId = BusId.invalid;

pub const AudioConfig = struct {
    max_buses: usize = default_max_buses,
    master_volume: f32 = 0.8,
    muted: bool = false,
};
pub const AudioEngineConfig = AudioConfig;

pub const AttenuationModel = enum(u8) {
    linear = 0,
    inverse = 1,
    exponential = 2,
};

pub const BusAttenuation = struct {
    model: AttenuationModel = .inverse,
    min_distance: f32 = 1.0,
    max_distance: f32 = 30.0,
    rolloff: f32 = 1.0,
};

pub const BusConfig = struct {
    name: []const u8 = "",
    spatial: bool = false,
    volume: f32 = 1.0,
    muted: bool = false,
    parent: ?BusId = null,
    attenuation_model: AttenuationModel = .inverse,
    min_distance: f32 = 1.0,
    max_distance: f32 = 30.0,
    rolloff: f32 = 1.0,
    doppler_factor: f32 = 1.0,
    filter: BusFilterConfig = .{},
    reverb: ?BusReverbConfig = null,
    occlusion: f32 = 0.0,
    occlusion_config: AudioOcclusionConfig = .{},
};

/// Voice slot sizing limits, hoisted out of `AudioEngine` so the sibling
/// leaves (`commands.zig`, `mixer.zig`) can size their scratch storage
/// without importing the `engine.zig` owner back. `AudioEngine` re-exports
/// each of these as a nested constant under its historical name.
pub const max_voices = 24;
pub const max_distance = 30.0;
pub const max_commands = 64;
pub const max_reverbs: usize = 4;
pub const chunk_frames: usize = 64;
pub const chunk_samples: usize = chunk_frames * 2;
pub const max_streams: usize = 32;

pub const VoiceKind = enum {
    noise_burst,
    thump,
    blip,
    sample,
};

pub const Voice = struct {
    /// Audio-thread-owned lifecycle flag, published atomically so main-thread
    /// `stopBus`/`stopAll` do not race the mixer.
    active: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    kind: VoiceKind = .thump,
    /// Packed bus id; 0xFF means "no bus". Atomic for the same reason.
    bus: std.atomic.Value(u8) = std.atomic.Value(u8).init(0xFF),
    t: f32 = 0.0,
    duration: f32 = 0.2,
    freq: f32 = 110.0,
    freq_end: f32 = 55.0,
    volume: f32 = 0.5,
    pan: f32 = 0.0,
    cutoff: f32 = 5000.0,
    cutoff_end: f32 = 500.0,
    seed: u32 = 1,
    lp: f32 = 0.0,
    lp_r: f32 = 0.0,
    phase: f32 = 0.0,
    // Sample playback state. `clip` is read-only and must outlive the voice.
    clip: ?*const AudioClip = null,
    sample_pos: f64 = 0.0,
    sample_step: f64 = 1.0,
    loop: bool = false,

    pub fn isActive(self: *const Voice) bool {
        return self.active.load(.acquire);
    }

    pub fn setActive(self: *Voice, on: bool) void {
        self.active.store(on, .release);
    }

    pub fn setBusId(self: *Voice, bus: ?BusId) void {
        self.bus.store(if (bus) |b| b.id() else 0xFF, .release);
    }

    pub fn busId(self: *const Voice) ?BusId {
        const raw = self.bus.load(.acquire);
        return if (raw == 0xFF) null else BusId.fromInt(raw);
    }
};
