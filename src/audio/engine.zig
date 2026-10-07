//! AudioEngine owner: type, lifecycle and thin forwarders.
//! Split out of `audio.zig` (facade).
//!
//! This module owns the `AudioEngine` type: the fields, the sizing limits
//! (re-exported from `types.zig` as nested constants so `AudioEngine.max_voices`
//! and every other historical path keep working), the option/result types
//! (re-exported from their home leaves), the trivial lifecycle
//! (`init`/`configure`/`setBusCapacity`/`getBusCapacity`, `start`/
//! `shutdown`, listener, master/mute, `stopAll`) and thin forwarders into
//! the siblings below, so every call site keeps working unchanged (same
//! pattern as `profiler/core.zig` and `particles/system.zig` — the type
//! cannot span files in Zig, so cross-file methods live as free functions
//! taking the engine as `anytype` and are reached through same-name
//! forwarders here):
//!
//! - `types.zig` — bus handles, configs, voice layout, sizing limits.
//! - `commands.zig` — voice trigger queue (`Command`, SPSC ring, slot steal).
//! - `playback.zig` — `play`/`playClip`, one-shot helpers, spatialize math.
//! - `buses.zig` — bus registry and routing DAG.
//! - `effects.zig` — bus DSP filters, reverb pool, occlusion state.
//! - `mixer.zig` — `renderFrames` + synth kernels (0-alloc, lock-free).
//! - `streams.zig` — stream slots, sound/music playback API.
//!
//! Anti-cycle rule (same as `profiler/`, `particles/`): siblings take the
//! engine as `anytype` and never import this module or the `audio.zig`
//! facade back; this module passes `self` straight through. `audio.zig`
//! re-exports `AudioEngine` under its historical path. Cross-leaf helpers
//! (`commands.pushCommand`/`drainCommands`, `buses.isBusSpatial`/…,
//! `effects.setBusFilter`/…, `playback.playClip`, …) are `pub` in their
//! home module for the sibling that needs them but are deliberately NOT
//! re-exported by the facade, so the public surface is identical to the
//! pre-split file.

const std = @import("std");
const sokol = @import("sokol");
const saudio = sokol.audio;
const slog = sokol.log;
const math = @import("math");
const Vec3 = math.Vec3;

const types = @import("types.zig");
const commands = @import("commands.zig");
const playback = @import("playback.zig");
const buses = @import("buses.zig");
const effects = @import("effects.zig");
const mixer = @import("mixer.zig");
const streams = @import("streams.zig");

const clip_mod = @import("clip.zig");
const AudioClip = clip_mod.AudioClip;

const dsp = @import("dsp.zig");
const BiquadFilterType = dsp.BiquadFilterType;
const BusFilterConfig = dsp.BusFilterConfig;
const BusReverbConfig = dsp.BusReverbConfig;
const BiquadFilter = dsp.BiquadFilter;
const ReverbProcessor = dsp.ReverbProcessor;

const occlusion_mod = @import("occlusion.zig");
const AudioOcclusionConfig = occlusion_mod.AudioOcclusionConfig;
const RaycastFn = occlusion_mod.RaycastFn;
const evaluateRaycastOcclusion = occlusion_mod.evaluateRaycastOcclusion;
const AudioOcclusionTracker = occlusion_mod.AudioOcclusionTracker;
const AudioEmitter = occlusion_mod.AudioEmitter;

const stream_mod = @import("stream.zig");
const StreamFormat = stream_mod.StreamFormat;
const StreamState = stream_mod.StreamState;
const StreamError = stream_mod.StreamError;
const StreamOptions = stream_mod.StreamOptions;
const PlaySoundOptions = stream_mod.PlaySoundOptions;
const AudioStream = stream_mod.AudioStream;

const BusId = types.BusId;
const AudioBus = types.AudioBus;
const default_max_buses = types.default_max_buses;
const max_bus_capacity = types.max_bus_capacity;
const max_buses = types.max_buses;
const invalid_bus = types.invalid_bus;
const AudioConfig = types.AudioConfig;
const AudioEngineConfig = types.AudioEngineConfig;
const AttenuationModel = types.AttenuationModel;
const BusAttenuation = types.BusAttenuation;
const BusConfig = types.BusConfig;
const VoiceKind = types.VoiceKind;
const Voice = types.Voice;
const Command = commands.Command;

pub const AudioEngine = struct {
    pub const max_voices = types.max_voices;
    pub const max_distance = types.max_distance;
    pub const max_commands = types.max_commands;
    pub const bus_limit = types.max_bus_capacity;
    pub const max_reverbs: usize = types.max_reverbs;
    pub const chunk_frames: usize = types.chunk_frames;
    pub const chunk_samples: usize = types.chunk_samples;
    pub const max_streams: usize = types.max_streams;

    pub const PlayOptions = playback.PlayOptions;
    pub const SpatializeResult = playback.SpatializeResult;
    pub const ClipPlayOptions = playback.ClipPlayOptions;
    pub const MusicPlayOptions = streams.MusicPlayOptions;

    // Lock-free single-producer single-consumer ring buffer for voice triggers.
    cmd_ring: [max_commands]Command = undefined,
    cmd_head: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    cmd_tail: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    voices: [max_voices]Voice = [_]Voice{Voice{}} ** max_voices,
    master_volume: std.atomic.Value(u32) = std.atomic.Value(u32).init(@bitCast(@as(f32, 0.8))),

    // Bus registry (up to max_bus_capacity = 128, configured at init, defaults to 32)
    bus_capacity: u8 = default_max_buses,
    bus_names: [max_bus_capacity][32]u8 = [_][32]u8{[_]u8{0} ** 32} ** max_bus_capacity,
    bus_name_lens: [max_bus_capacity]u8 = [_]u8{0} ** max_bus_capacity,
    bus_active: [max_bus_capacity]std.atomic.Value(bool) = [_]std.atomic.Value(bool){std.atomic.Value(bool).init(false)} ** max_bus_capacity,
    bus_spatial: [max_bus_capacity]std.atomic.Value(bool) = [_]std.atomic.Value(bool){std.atomic.Value(bool).init(false)} ** max_bus_capacity,
    bus_volumes: [max_bus_capacity]std.atomic.Value(u32) = [_]std.atomic.Value(u32){std.atomic.Value(u32).init(@bitCast(@as(f32, 1.0)))} ** max_bus_capacity,
    bus_muted: [max_bus_capacity]std.atomic.Value(bool) = [_]std.atomic.Value(bool){std.atomic.Value(bool).init(false)} ** max_bus_capacity,
    bus_parent: [max_bus_capacity]std.atomic.Value(u8) = [_]std.atomic.Value(u8){std.atomic.Value(u8).init(0xFF)} ** max_bus_capacity,
    bus_model: [max_bus_capacity]std.atomic.Value(u8) = [_]std.atomic.Value(u8){std.atomic.Value(u8).init(@intFromEnum(AttenuationModel.inverse))} ** max_bus_capacity,
    bus_min_dist: [max_bus_capacity]std.atomic.Value(u32) = [_]std.atomic.Value(u32){std.atomic.Value(u32).init(@bitCast(@as(f32, 1.0)))} ** max_bus_capacity,
    bus_max_dist: [max_bus_capacity]std.atomic.Value(u32) = [_]std.atomic.Value(u32){std.atomic.Value(u32).init(@bitCast(@as(f32, 30.0)))} ** max_bus_capacity,
    bus_rolloff: [max_bus_capacity]std.atomic.Value(u32) = [_]std.atomic.Value(u32){std.atomic.Value(u32).init(@bitCast(@as(f32, 1.0)))} ** max_bus_capacity,
    bus_doppler: [max_bus_capacity]std.atomic.Value(u32) = [_]std.atomic.Value(u32){std.atomic.Value(u32).init(@bitCast(@as(f32, 1.0)))} ** max_bus_capacity,

    // DSP effects (filters and reverb units)
    bus_filter_type: [max_bus_capacity]std.atomic.Value(u8) = [_]std.atomic.Value(u8){std.atomic.Value(u8).init(@intFromEnum(BiquadFilterType.none))} ** max_bus_capacity,
    bus_filter_cutoff: [max_bus_capacity]std.atomic.Value(u32) = [_]std.atomic.Value(u32){std.atomic.Value(u32).init(@bitCast(@as(f32, 1000.0)))} ** max_bus_capacity,
    bus_filter_q: [max_bus_capacity]std.atomic.Value(u32) = [_]std.atomic.Value(u32){std.atomic.Value(u32).init(@bitCast(@as(f32, 0.7071)))} ** max_bus_capacity,
    bus_filters: [max_bus_capacity]dsp.BiquadFilter = [_]dsp.BiquadFilter{dsp.BiquadFilter{}} ** max_bus_capacity,

    reverbs: [max_reverbs]dsp.ReverbProcessor = [_]dsp.ReverbProcessor{dsp.ReverbProcessor.init(44100.0)} ** max_reverbs,
    bus_reverb_slot: [max_bus_capacity]std.atomic.Value(u8) = [_]std.atomic.Value(u8){std.atomic.Value(u8).init(0xFF)} ** max_bus_capacity,
    reverb_bus_owner: [max_reverbs]std.atomic.Value(u8) = [_]std.atomic.Value(u8){std.atomic.Value(u8).init(0xFF)} ** max_reverbs,

    // Occlusion state & filters (geometry attenuation and muffling)
    bus_occlusion: [max_bus_capacity]std.atomic.Value(u32) = [_]std.atomic.Value(u32){std.atomic.Value(u32).init(@bitCast(@as(f32, 0.0)))} ** max_bus_capacity,
    bus_occlusion_config: [max_bus_capacity]AudioOcclusionConfig = [_]AudioOcclusionConfig{AudioOcclusionConfig{}} ** max_bus_capacity,
    bus_occlusion_filters: [max_bus_capacity]dsp.BiquadFilter = [_]dsp.BiquadFilter{dsp.BiquadFilter{}} ** max_bus_capacity,

    // Block-based scratch chunk buffers
    bus_chunks: [max_bus_capacity][chunk_samples]f32 = [_][chunk_samples]f32{[_]f32{0.0} ** chunk_samples} ** max_bus_capacity,

    muted: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    listener_pos: Vec3 = Vec3.zero,
    listener_right: Vec3 = Vec3.new(1.0, 0.0, 0.0),
    listener_vel: Vec3 = Vec3.zero,
    started: bool = false,
    sample_rate: f32 = 44100.0,
    next_seed: u32 = 0x12345678,

    // Audio streams (background music, ambient loops, dialogue)
    streams: [max_streams]?*AudioStream = [_]?*AudioStream{null} ** max_streams,
    music_stream: ?*AudioStream = null,
    music_fade_stream: ?*AudioStream = null,

    pub fn init(config: AudioConfig) AudioEngine {
        var eng = AudioEngine{};
        eng.configure(config);
        return eng;
    }

    pub fn configure(self: *AudioEngine, config: AudioConfig) void {
        self.setBusCapacity(config.max_buses);
        self.setMasterVolume(config.master_volume);
        self.setMuted(config.muted);
    }

    pub fn setBusCapacity(self: *AudioEngine, cap: usize) void {
        self.bus_capacity = @intCast(std.math.clamp(cap, 1, max_bus_capacity));
    }

    pub fn getBusCapacity(self: *const AudioEngine) usize {
        return self.bus_capacity;
    }

    /// Opens the audio device (stereo). On failure the engine stays silent
    /// and `play()` calls keep queueing cheaply.
    pub fn start(self: *AudioEngine) void {
        if (self.started) return;
        saudio.setup(.{
            .stream_userdata_cb = streamCallback,
            .user_data = self,
            .num_channels = 2,
            .logger = .{ .func = slog.func },
        });
        if (saudio.isvalid()) {
            self.sample_rate = @floatFromInt(saudio.sampleRate());
            for (&self.reverbs) |*r| r.setSampleRate(self.sample_rate);
            self.started = true;
        }
    }

    /// Stops the audio backend and frees every stream still registered.
    /// Streams live in engine slots from registerStream until
    /// unregisterStream, so a stream the caller never destroyed (e.g. a
    /// looping song kept for the app's lifetime) is freed here — without
    /// this, shutdown only stopped it and its decoder/ring buffers leaked.
    /// Caller-held stream pointers are invalid after shutdown.
    pub fn shutdown(self: *AudioEngine) void {
        if (!self.started) return;
        saudio.shutdown();
        self.started = false;
        for (&self.streams) |*slot| {
            if (slot.*) |s| {
                s.stop();
                slot.* = null;
                s.deinit();
            }
        }
        self.music_stream = null;
        self.music_fade_stream = null;
    }

    /// Lock-free attenuation/pan/doppler from a listener snapshot
    /// (see playback.zig).
    pub fn spatializeWith(
        lpos: Vec3,
        lright: Vec3,
        lvel: Vec3,
        position: ?Vec3,
        volume: f32,
        min_dist: f32,
        max_dist: f32,
        rolloff: f32,
        model: AttenuationModel,
        velocity: ?Vec3,
    ) SpatializeResult {
        return playback.spatializeWith(lpos, lright, lvel, position, volume, min_dist, max_dist, rolloff, model, velocity);
    }

    pub fn play(self: *AudioEngine, params: PlayOptions) void {
        playback.play(self, params);
    }

    /// Impact thump, louder and deeper with speed (m/s) (see playback.zig).
    pub fn playImpact(self: *AudioEngine, position: Vec3, speed: f32) void {
        playback.playImpact(self, position, speed);
    }

    pub fn playImpactOn(self: *AudioEngine, bus: ?BusId, position: Vec3, speed: f32) void {
        playback.playImpactOn(self, bus, position, speed);
    }

    /// Explosion noise burst; size scales duration (see playback.zig).
    pub fn playExplosion(self: *AudioEngine, position: Vec3, size: f32) void {
        playback.playExplosion(self, position, size);
    }

    pub fn playExplosionOn(self: *AudioEngine, bus: ?BusId, position: Vec3, size: f32) void {
        playback.playExplosionOn(self, bus, position, size);
    }

    /// Short non-positional blip (UI, jumps, snaps) (see playback.zig).
    pub fn playBlip(self: *AudioEngine, freq: f32) void {
        playback.playBlip(self, freq);
    }

    pub fn playBlipOn(self: *AudioEngine, bus: ?BusId, freq: f32) void {
        playback.playBlipOn(self, bus, freq);
    }

    /// Plays a decoded WAV clip on a mixer voice (see playback.zig).
    pub fn playClip(self: *AudioEngine, clip: *const AudioClip, options: ClipPlayOptions) void {
        playback.playClip(self, clip, options);
    }

    pub fn updateListener(self: *AudioEngine, pos: Vec3, right: Vec3) void {
        const r = if (right.length() > 1e-4) right.normalize() else Vec3.new(1.0, 0.0, 0.0);
        self.listener_pos = pos;
        self.listener_right = r;
    }

    pub fn updateListenerWithVelocity(self: *AudioEngine, pos: Vec3, right: Vec3, vel: Vec3) void {
        const r = if (right.length() > 1e-4) right.normalize() else Vec3.new(1.0, 0.0, 0.0);
        self.listener_pos = pos;
        self.listener_right = r;
        self.listener_vel = vel;
    }

    pub fn createBus(self: *AudioEngine, config: BusConfig) ?BusId {
        return buses.createBus(self, config);
    }

    pub fn createSpatialBus(self: *AudioEngine, name: []const u8, config: ?BusConfig) ?BusId {
        return buses.createSpatialBus(self, name, config);
    }

    pub fn createNonSpatialBus(self: *AudioEngine, name: []const u8, volume: f32) ?BusId {
        return buses.createNonSpatialBus(self, name, volume);
    }

    pub fn configureBus(self: *AudioEngine, bus: BusId, config: BusConfig) void {
        buses.configureBus(self, bus, config);
    }

    pub fn destroyBus(self: *AudioEngine, bus: BusId) void {
        buses.destroyBus(self, bus);
    }

    pub fn findBus(self: *const AudioEngine, name: []const u8) ?BusId {
        return buses.findBus(self, name);
    }

    pub fn findOrCreateBus(self: *AudioEngine, name: []const u8, config: BusConfig) ?BusId {
        return buses.findOrCreateBus(self, name, config);
    }

    pub fn isBusActive(self: *const AudioEngine, bus: BusId) bool {
        return buses.isBusActive(self, bus);
    }

    pub fn getBusCount(self: *const AudioEngine) usize {
        return buses.getBusCount(self);
    }

    pub fn isBusSpatial(self: *const AudioEngine, bus: BusId) bool {
        return buses.isBusSpatial(self, bus);
    }

    pub fn setBusSpatial(self: *AudioEngine, bus: BusId, spatial: bool) void {
        buses.setBusSpatial(self, bus, spatial);
    }

    pub fn getBusName(self: *const AudioEngine, bus: BusId) []const u8 {
        return buses.getBusName(self, bus);
    }

    pub fn setBusName(self: *AudioEngine, bus: BusId, name: []const u8) void {
        buses.setBusName(self, bus, name);
    }

    pub fn setBusParent(self: *AudioEngine, bus: BusId, parent: ?BusId) void {
        buses.setBusParent(self, bus, parent);
    }

    pub fn getBusParent(self: *const AudioEngine, bus: BusId) ?BusId {
        return buses.getBusParent(self, bus);
    }

    pub fn setBusAttenuation(self: *AudioEngine, bus: BusId, model: AttenuationModel, min_dist: f32, max_dist: f32, rolloff: f32) void {
        buses.setBusAttenuation(self, bus, model, min_dist, max_dist, rolloff);
    }

    pub fn getBusAttenuation(self: *const AudioEngine, bus: BusId) BusAttenuation {
        return buses.getBusAttenuation(self, bus);
    }

    pub fn setBusDopplerFactor(self: *AudioEngine, bus: BusId, factor: f32) void {
        buses.setBusDopplerFactor(self, bus, factor);
    }

    pub fn getBusDopplerFactor(self: *const AudioEngine, bus: BusId) f32 {
        return buses.getBusDopplerFactor(self, bus);
    }

    pub fn setBusVolume(self: *AudioEngine, bus: BusId, vol: f32) void {
        buses.setBusVolume(self, bus, vol);
    }

    pub fn getBusVolume(self: *const AudioEngine, bus: BusId) f32 {
        return buses.getBusVolume(self, bus);
    }

    pub fn getBusEffectiveVolume(self: *const AudioEngine, bus_opt: ?BusId) f32 {
        return buses.getBusEffectiveVolume(self, bus_opt);
    }

    pub fn setBusMuted(self: *AudioEngine, bus: BusId, muted_val: bool) void {
        buses.setBusMuted(self, bus, muted_val);
    }

    pub fn isBusMuted(self: *const AudioEngine, bus: BusId) bool {
        return buses.isBusMuted(self, bus);
    }

    pub fn toggleBusMuted(self: *AudioEngine, bus: BusId) void {
        buses.toggleBusMuted(self, bus);
    }

    pub fn stopBus(self: *AudioEngine, bus: BusId) void {
        buses.stopBus(self, bus);
    }

    // --- Bus DSP Effect Methods (Biquad Filter & Reverb) ---

    pub fn setBusFilter(self: *AudioEngine, bus: BusId, filter_type: BiquadFilterType, cutoff: f32, q: f32) void {
        effects.setBusFilter(self, bus, filter_type, cutoff, q);
    }

    pub fn setBusFilterCutoff(self: *AudioEngine, bus: BusId, cutoff: f32) void {
        effects.setBusFilterCutoff(self, bus, cutoff);
    }

    pub fn setBusFilterQ(self: *AudioEngine, bus: BusId, q: f32) void {
        effects.setBusFilterQ(self, bus, q);
    }

    pub fn clearBusFilter(self: *AudioEngine, bus: BusId) void {
        effects.clearBusFilter(self, bus);
    }

    pub fn getBusFilter(self: *const AudioEngine, bus: BusId) BusFilterConfig {
        return effects.getBusFilter(self, bus);
    }

    pub fn setBusReverb(self: *AudioEngine, bus: BusId, config: BusReverbConfig) bool {
        return effects.setBusReverb(self, bus, config);
    }

    pub fn clearBusReverb(self: *AudioEngine, bus: BusId) void {
        effects.clearBusReverb(self, bus);
    }

    pub fn getBusReverb(self: *const AudioEngine, bus: BusId) ?BusReverbConfig {
        return effects.getBusReverb(self, bus);
    }

    pub fn isBusReverbEnabled(self: *const AudioEngine, bus: BusId) bool {
        return effects.isBusReverbEnabled(self, bus);
    }

    pub fn setBusUnderwater(self: *AudioEngine, bus: BusId) void {
        effects.setBusUnderwater(self, bus);
    }

    pub fn setBusMuffled(self: *AudioEngine, bus: BusId) void {
        effects.setBusMuffled(self, bus);
    }

    pub fn setBusTelephone(self: *AudioEngine, bus: BusId) void {
        effects.setBusTelephone(self, bus);
    }

    pub fn setBusCaveReverb(self: *AudioEngine, bus: BusId) bool {
        return effects.setBusCaveReverb(self, bus);
    }

    pub fn setBusRoomReverb(self: *AudioEngine, bus: BusId) bool {
        return effects.setBusRoomReverb(self, bus);
    }

    pub fn setBusOcclusion(self: *AudioEngine, bus: BusId, occlusion: f32) void {
        effects.setBusOcclusion(self, bus, occlusion);
    }

    pub fn getBusOcclusion(self: *const AudioEngine, bus: BusId) f32 {
        return effects.getBusOcclusion(self, bus);
    }

    pub fn setBusOcclusionConfig(self: *AudioEngine, bus: BusId, config: AudioOcclusionConfig) void {
        effects.setBusOcclusionConfig(self, bus, config);
    }

    pub fn getBusOcclusionConfig(self: *const AudioEngine, bus: BusId) AudioOcclusionConfig {
        return effects.getBusOcclusionConfig(self, bus);
    }

    pub fn clearBusOcclusion(self: *AudioEngine, bus: BusId) void {
        effects.clearBusOcclusion(self, bus);
    }

    pub fn updateBusOcclusion(self: *AudioEngine, bus: BusId, target_occlusion: f32, dt: f32) void {
        effects.updateBusOcclusion(self, bus, target_occlusion, dt);
    }

    pub fn evaluateOcclusion(
        self: *const AudioEngine,
        emitter_pos: Vec3,
        config: AudioOcclusionConfig,
        raycast_fn: RaycastFn,
        user_data: ?*anyopaque,
    ) f32 {
        return effects.evaluateOcclusion(self, emitter_pos, config, raycast_fn, user_data);
    }

    pub fn updateBusOcclusionWithRaycast(
        self: *AudioEngine,
        bus: BusId,
        emitter_pos: Vec3,
        dt: f32,
        raycast_fn: RaycastFn,
        user_data: ?*anyopaque,
    ) f32 {
        return effects.updateBusOcclusionWithRaycast(self, bus, emitter_pos, dt, raycast_fn, user_data);
    }

    pub fn stopAll(self: *AudioEngine) void {
        for (&self.voices) |*v| {
            v.active = false;
        }
    }

    // --- Audio Stream & Music Management ---

    pub fn registerStream(self: *AudioEngine, stream: *AudioStream) !void {
        return streams.registerStream(self, stream);
    }

    pub fn unregisterStream(self: *AudioEngine, stream: *AudioStream) void {
        streams.unregisterStream(self, stream);
    }

    pub fn createStreamFromFile(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        path: []const u8,
        options: StreamOptions,
    ) !*AudioStream {
        return streams.createStreamFromFile(self, allocator, path, options);
    }

    pub fn createStreamFromMemory(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        options: StreamOptions,
    ) !*AudioStream {
        return streams.createStreamFromMemory(self, allocator, bytes, format, options);
    }

    pub fn destroyStream(self: *AudioEngine, stream: *AudioStream) void {
        streams.destroyStream(self, stream);
    }

    pub fn updateStreams(self: *AudioEngine, dt: f32) void {
        streams.updateStreams(self, dt);
    }

    // ========================================================================
    // Sound Playback & Streaming API
    // ========================================================================

    /// Plays a sound from a file path (OGG, MP3, WAV) (see streams.zig).
    pub fn playSound(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        path: []const u8,
        options: PlaySoundOptions,
    ) !*AudioStream {
        return streams.playSound(self, allocator, path, options);
    }

    /// Plays a sound from an in-memory byte slice (OGG, MP3, WAV)
    /// (see streams.zig).
    pub fn playSoundFromMemory(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        options: PlaySoundOptions,
    ) !*AudioStream {
        return streams.playSoundFromMemory(self, allocator, bytes, format, options);
    }

    /// Fire-and-forget sound playback (see streams.zig).
    pub fn playSoundOnce(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        path: []const u8,
        options: PlaySoundOptions,
    ) !void {
        return streams.playSoundOnce(self, allocator, path, options);
    }

    /// Fire-and-forget in-memory sound playback (see streams.zig).
    pub fn playSoundOnceFromMemory(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        options: PlaySoundOptions,
    ) !void {
        return streams.playSoundOnceFromMemory(self, allocator, bytes, format, options);
    }

    /// Stops a sound stream immediately, or over `fade_duration` seconds
    /// (see streams.zig).
    pub fn stopSound(self: *AudioEngine, stream: *AudioStream, fade_duration: f32) void {
        streams.stopSound(self, stream, fade_duration);
    }

    /// Pauses an active sound stream (see streams.zig).
    pub fn pauseSound(self: *AudioEngine, stream: *AudioStream) void {
        streams.pauseSound(self, stream);
    }

    /// Resumes a paused sound stream (see streams.zig).
    pub fn resumeSound(self: *AudioEngine, stream: *AudioStream) void {
        streams.resumeSound(self, stream);
    }

    /// Sets the volume of an active sound stream (see streams.zig).
    pub fn setSoundVolume(self: *AudioEngine, stream: *AudioStream, vol: f32) void {
        streams.setSoundVolume(self, stream, vol);
    }

    /// Stops all currently active sound streams (see streams.zig).
    pub fn stopAllSounds(self: *AudioEngine, fade_duration: f32) void {
        streams.stopAllSounds(self, fade_duration);
    }

    /// Crossfades from an existing sound stream to a new sound file
    /// (see streams.zig).
    pub fn crossfadeSound(
        self: *AudioEngine,
        old_stream: ?*AudioStream,
        allocator: std.mem.Allocator,
        path: []const u8,
        fade_duration: f32,
        options: PlaySoundOptions,
    ) !*AudioStream {
        return streams.crossfadeSound(self, old_stream, allocator, path, fade_duration, options);
    }

    /// Crossfades from an existing sound stream to a new in-memory sound
    /// (see streams.zig).
    pub fn crossfadeSoundFromMemory(
        self: *AudioEngine,
        old_stream: ?*AudioStream,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        fade_duration: f32,
        options: PlaySoundOptions,
    ) !*AudioStream {
        return streams.crossfadeSoundFromMemory(self, old_stream, allocator, bytes, format, fade_duration, options);
    }

    // ========================================================================
    // Dedicated Music Slot Convenience Helpers
    // ========================================================================

    /// Convenience helper for playing background music / looping track
    /// (see streams.zig).
    pub fn playMusic(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        path: []const u8,
        options: PlaySoundOptions,
    ) !*AudioStream {
        return streams.playMusic(self, allocator, path, options);
    }

    pub fn playMusicFromMemory(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        options: PlaySoundOptions,
    ) !*AudioStream {
        return streams.playMusicFromMemory(self, allocator, bytes, format, options);
    }

    pub fn crossfadeMusic(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        path: []const u8,
        fade_duration: f32,
        options: PlaySoundOptions,
    ) !*AudioStream {
        return streams.crossfadeMusic(self, allocator, path, fade_duration, options);
    }

    pub fn stopMusic(self: *AudioEngine, fade_duration: f32) void {
        streams.stopMusic(self, fade_duration);
    }

    pub fn pauseMusic(self: *AudioEngine) void {
        streams.pauseMusic(self);
    }

    pub fn resumeMusic(self: *AudioEngine) void {
        streams.resumeMusic(self);
    }

    pub fn setMusicVolume(self: *AudioEngine, vol: f32) void {
        streams.setMusicVolume(self, vol);
    }

    pub fn getMusicStream(self: *const AudioEngine) ?*AudioStream {
        return streams.getMusicStream(self);
    }

    pub fn playMusicClip(self: *AudioEngine, clip: *const AudioClip, options: MusicPlayOptions) void {
        streams.playMusicClip(self, clip, options);
    }

    pub fn setMuted(self: *AudioEngine, muted_val: bool) void {
        self.muted.store(muted_val, .release);
    }

    pub fn toggleMuted(self: *AudioEngine) void {
        var cur = self.muted.load(.monotonic);
        while (self.muted.cmpxchgWeak(cur, !cur, .acq_rel, .monotonic)) |next| {
            cur = next;
        }
    }

    pub fn isMuted(self: *const AudioEngine) bool {
        return self.muted.load(.acquire);
    }

    pub fn setMasterVolume(self: *AudioEngine, vol: f32) void {
        const v = std.math.clamp(vol, 0.0, 2.0);
        self.master_volume.store(@bitCast(v), .release);
    }

    pub fn getMasterVolume(self: *const AudioEngine) f32 {
        return @bitCast(self.master_volume.load(.acquire));
    }

    /// Mixes stereo-interleaved frames and advances voice state
    /// (see mixer.zig).
    pub fn renderFrames(self: *AudioEngine, buffer: []f32) void {
        mixer.renderFrames(self, buffer);
    }

    fn streamCallback(buffer: [*c]f32, num_frames: i32, num_channels: i32, user_data: ?*anyopaque) callconv(.c) void {
        const self: *AudioEngine = @ptrCast(@alignCast(user_data orelse return));
        if (num_channels != 2) return;
        const n: usize = @intCast(num_frames);
        self.renderFrames(buffer[0 .. n * 2]);
    }
};
