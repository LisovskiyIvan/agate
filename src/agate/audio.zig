const std = @import("std");
const builtin = @import("builtin");
const sokol = @import("sokol");
const saudio = sokol.audio;
const slog = sokol.log;
const math = @import("math");
const Vec3 = math.Vec3;

/// Procedural sound effects mixed on the sokol.audio stream thread.
/// No audio assets needed: explosions, impacts and blips are synthesized.
/// `play()` is thread-safe and cheap; the mixer holds a short spinlock per
/// callback. `renderFrames` is the mix core and is unit-testable without a
/// device.
const clip_mod = @import("audio/clip.zig");
pub const AudioClip = clip_mod.AudioClip;

const dsp = @import("audio/dsp.zig");
pub const BiquadFilterType = dsp.BiquadFilterType;
pub const BusFilterConfig = dsp.BusFilterConfig;
pub const BusReverbConfig = dsp.BusReverbConfig;
pub const BiquadFilter = dsp.BiquadFilter;
pub const ReverbProcessor = dsp.ReverbProcessor;

const occlusion_mod = @import("audio/occlusion.zig");
pub const AudioOcclusionConfig = occlusion_mod.AudioOcclusionConfig;
pub const RaycastFn = occlusion_mod.RaycastFn;
pub const evaluateRaycastOcclusion = occlusion_mod.evaluateRaycastOcclusion;
pub const AudioOcclusionTracker = occlusion_mod.AudioOcclusionTracker;
pub const AudioEmitter = occlusion_mod.AudioEmitter;

const stream_mod = @import("audio/stream.zig");
pub const StreamFormat = stream_mod.StreamFormat;
pub const StreamState = stream_mod.StreamState;
pub const StreamError = stream_mod.StreamError;
pub const StreamOptions = stream_mod.StreamOptions;
pub const PlaySoundOptions = stream_mod.PlaySoundOptions;
pub const AudioStream = stream_mod.AudioStream;

pub const BusId = enum(u8) {
    _,

    pub const invalid: BusId = @enumFromInt(0xFF);

    pub fn id(self: BusId) u8 {
        return @intFromEnum(self);
    }

    pub fn fromInt(val: u8) BusId {
        return @enumFromInt(val);
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

pub const AudioEngine = struct {
    pub const max_voices = 24;
    pub const max_distance = 30.0;
    pub const max_commands = 64;
    pub const bus_limit = max_bus_capacity;
    pub const max_reverbs: usize = 4;
    pub const chunk_frames: usize = 64;
    pub const chunk_samples: usize = chunk_frames * 2;
    pub const max_streams: usize = 32;

    const Command = union(enum) {
        voice: struct {
            kind: VoiceKind,
            bus: ?BusId = null,
            volume: f32,
            pan: f32,
            duration: f32,
            freq: f32,
            freq_end: f32,
            cutoff: f32,
            cutoff_end: f32,
            seed: u32,
        },
        clip: struct {
            clip: *const AudioClip,
            bus: ?BusId = null,
            volume: f32,
            pan: f32,
            duration: f32,
            sample_step: f64,
            loop: bool,
            seed: u32,
            cutoff: f32 = 20000.0,
        },
    };

    /// Per-buffer render state for one voice, built once in `renderFrames`
    /// so the inner sample loops do no sqrt, no voice scan and no pan math.
    const ActiveVoice = struct {
        v: *Voice,
        kind: VoiceKind,
        gl: f32, // volume * master * left pan gain
        gr: f32, // volume * master * right pan gain
        duration: f32,
        inv_duration: f32, // 1 / max(duration, 1e-6)
        freq0: f32, // synth only: ramp slopes per second of voice time
        freq_slope: f32,
        cutoff0: f32,
        cutoff_slope: f32,
        clip: ?[]const f32 = null, // sample only: interleaved stereo frames
        frames: usize = 0,
        step: f64 = 1.0, // clip_rate / engine_rate * rate
        loop: bool = false,
        step_one: bool = false, // step == 1 from an integer pos: copy, no lerp
    };

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

    pub fn shutdown(self: *AudioEngine) void {
        if (!self.started) return;
        saudio.shutdown();
        self.started = false;
        for (&self.streams) |*slot| {
            if (slot.*) |s| s.stop();
        }
    }

    /// One-shot procedural voice trigger: kind + spatial position + the
    /// envelope/filter sweep the synthesizer applies. `play` is the only entry.
    pub const PlayOptions = struct {
        kind: VoiceKind = .thump,
        bus: ?BusId = null,
        pan: f32 = 0.0,
        position: ?Vec3 = null, // null = non-positional
        min_distance: ?f32 = null, // null = inherit from bus
        max_distance: ?f32 = null, // null = inherit from bus
        rolloff: ?f32 = null, // null = inherit from bus
        attenuation_model: ?AttenuationModel = null, // null = inherit from bus
        doppler_factor: ?f32 = null, // null = inherit from bus
        velocity: ?Vec3 = null, // emitter velocity for Doppler
        volume: f32 = 0.5,
        duration: f32 = 0.2,
        freq: f32 = 110.0,
        freq_end: f32 = 55.0,
        cutoff: f32 = 5000.0,
        cutoff_end: f32 = 500.0,
        pitch: f32 = 1.0,
        pitch_randomness: f32 = 0.0,
        occlusion: f32 = 0.0,
        occlusion_config: AudioOcclusionConfig = .{},
    };

    /// Pushes an audio trigger command to the lock-free SPSC queue.
    fn pushCommand(self: *AudioEngine, cmd: Command) bool {
        const head = self.cmd_head.load(.monotonic);
        const tail = self.cmd_tail.load(.acquire);
        if (head -% tail >= max_commands) {
            return false;
        }
        self.cmd_ring[head % max_commands] = cmd;
        self.cmd_head.store(head +% 1, .release);
        return true;
    }

    /// Drains pending commands from the lock-free SPSC queue into voices.
    fn drainCommands(self: *AudioEngine) void {
        const head = self.cmd_head.load(.acquire);
        var tail = self.cmd_tail.load(.monotonic);
        while (tail != head) {
            const cmd = self.cmd_ring[tail % max_commands];
            self.applyCommand(cmd);
            tail +%= 1;
        }
        self.cmd_tail.store(tail, .release);
    }

    fn applyCommand(self: *AudioEngine, cmd: Command) void {
        const v = self.acquireSlot() orelse return;
        switch (cmd) {
            .voice => |p| {
                v.* = .{
                    .active = true,
                    .kind = p.kind,
                    .bus = p.bus,
                    .volume = p.volume,
                    .pan = p.pan,
                    .duration = p.duration,
                    .freq = p.freq,
                    .freq_end = p.freq_end,
                    .cutoff = p.cutoff,
                    .cutoff_end = p.cutoff_end,
                    .seed = p.seed,
                };
            },
            .clip => |c| {
                v.* = .{
                    .active = true,
                    .kind = .sample,
                    .bus = c.bus,
                    .volume = c.volume,
                    .pan = c.pan,
                    .duration = c.duration,
                    .cutoff = c.cutoff,
                    .cutoff_end = c.cutoff,
                    .lp = 0.0,
                    .lp_r = 0.0,
                    .clip = c.clip,
                    .sample_pos = 0.0,
                    .sample_step = c.sample_step,
                    .loop = c.loop,
                    .seed = c.seed,
                };
            },
        }
    }

    /// Picks a free voice, or steals the one closest to finishing.
    fn acquireSlot(self: *AudioEngine) ?*Voice {
        for (&self.voices) |*v| {
            if (!v.active) return v;
        }
        // Steal the voice closest to finishing.
        var best: ?*Voice = null;
        var best_k: f32 = -1.0;
        for (&self.voices) |*v| {
            const k = v.t / @max(v.duration, 1e-6);
            if (k > best_k) {
                best_k = k;
                best = v;
            }
        }
        return best;
    }

    pub const SpatializeResult = struct {
        vol: f32,
        pan: f32,
        doppler: f32,
    };

    /// Shared distance attenuation + pan for `play` and `playClip`.
    fn spatialize(self: *const AudioEngine, position: ?Vec3, volume: f32) struct { vol: f32, pan: f32 } {
        const res = spatializeWith(self.listener_pos, self.listener_right, self.listener_vel, position, volume, 1.0, max_distance, 1.0, .linear, null);
        return .{ .vol = res.vol, .pan = res.pan };
    }

    /// Lock-free attenuation/pan/doppler from a listener snapshot.
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
        var vol = volume;
        var pan: f32 = 0.0;
        var doppler: f32 = 1.0;
        if (position) |p| {
            const to = p.sub(lpos);
            const dist = to.length();
            const min_d = @max(0.001, min_dist);
            const max_d = @max(min_d + 0.001, max_dist);
            const r_off = @max(0.0, rolloff);

            if (dist <= min_d) {
                // inside inner full-volume radius
            } else if (dist >= max_d) {
                vol = 0.0;
            } else {
                const delta_d = dist - min_d;
                const range_d = max_d - min_d;
                switch (model) {
                    .linear => {
                        const factor = @max(0.0, 1.0 - delta_d / range_d);
                        vol *= if (r_off != 1.0) std.math.pow(f32, factor, r_off) else factor;
                    },
                    .inverse => {
                        const factor = min_d / (min_d + r_off * delta_d);
                        vol *= factor;
                    },
                    .exponential => {
                        const factor = std.math.exp(-r_off * delta_d / range_d);
                        vol *= factor;
                    },
                }
            }

            if (dist > 1e-4) {
                const dir = to.scale(1.0 / dist);
                pan = std.math.clamp(dir.dot(lright), -1.0, 1.0);

                if (velocity) |vel| {
                    const SPEED_OF_SOUND: f32 = 343.0;
                    const v_l = lvel.dot(dir);
                    const v_e = vel.dot(dir);
                    const num = SPEED_OF_SOUND + v_l;
                    const den = SPEED_OF_SOUND + v_e;
                    if (num > 1.0 and den > 1.0) {
                        doppler = std.math.clamp(num / den, 0.25, 4.0);
                    }
                }
            }
        }
        return .{ .vol = vol, .pan = pan, .doppler = doppler };
    }

    pub fn play(self: *AudioEngine, params: PlayOptions) void {
        const is_spatial = if (params.bus) |b| self.isBusSpatial(b) else (params.position != null);
        var sp_vol: f32 = params.volume;
        var sp_pan: f32 = params.pan;
        var sp_doppler: f32 = 1.0;

        if (is_spatial and params.position != null) {
            var min_dist: f32 = 1.0;
            var max_dist: f32 = 30.0;
            var rolloff: f32 = 1.0;
            var model: AttenuationModel = .inverse;
            var dop_scale: f32 = 1.0;

            if (params.bus) |b| {
                const bus_attn = self.getBusAttenuation(b);
                min_dist = params.min_distance orelse bus_attn.min_distance;
                max_dist = params.max_distance orelse bus_attn.max_distance;
                rolloff = params.rolloff orelse bus_attn.rolloff;
                model = params.attenuation_model orelse bus_attn.model;
                dop_scale = params.doppler_factor orelse self.getBusDopplerFactor(b);
            } else {
                min_dist = params.min_distance orelse 1.0;
                max_dist = params.max_distance orelse 30.0;
                rolloff = params.rolloff orelse 1.0;
                model = params.attenuation_model orelse .inverse;
                dop_scale = params.doppler_factor orelse 1.0;
            }

            const sp = spatializeWith(
                self.listener_pos,
                self.listener_right,
                self.listener_vel,
                params.position,
                params.volume,
                min_dist,
                max_dist,
                rolloff,
                model,
                params.velocity,
            );
            sp_vol = sp.vol;
            sp_pan = sp.pan;
            if (dop_scale > 0.0) {
                sp_doppler = 1.0 + (sp.doppler - 1.0) * dop_scale;
            }
        }

        var eff_cutoff = params.cutoff;
        var eff_cutoff_end = params.cutoff_end;
        if (params.occlusion > 0.0) {
            const occ = std.math.clamp(params.occlusion, 0.0, 1.0);
            sp_vol *= std.math.lerp(1.0, params.occlusion_config.min_volume, occ);
            const occ_cutoff = std.math.lerp(params.occlusion_config.max_cutoff, params.occlusion_config.min_cutoff, occ);
            eff_cutoff = @min(eff_cutoff, occ_cutoff);
            eff_cutoff_end = @min(eff_cutoff_end, occ_cutoff);
        }
        if (sp_vol <= 0.001) return;

        self.next_seed +%= 1;
        const seed = self.next_seed *% 2654435761 +% 97;

        var pitch_factor = params.pitch * sp_doppler;
        if (params.pitch_randomness > 0.0) {
            const r = @as(f32, @floatFromInt(seed & 0xFFFF)) / 65535.0;
            const rnd_offset = (r * 2.0 - 1.0) * params.pitch_randomness;
            pitch_factor *= @max(0.1, 1.0 + rnd_offset);
        }
        pitch_factor = std.math.clamp(pitch_factor, 0.1, 10.0);

        const duration = @max(params.duration / pitch_factor, 0.01);
        const cmd = Command{
            .voice = .{
                .kind = params.kind,
                .bus = params.bus,
                .volume = sp_vol,
                .pan = sp_pan,
                .duration = duration,
                .freq = params.freq * pitch_factor,
                .freq_end = params.freq_end * pitch_factor,
                .cutoff = eff_cutoff,
                .cutoff_end = eff_cutoff_end,
                .seed = seed,
            },
        };

        _ = self.pushCommand(cmd);
        if (!self.started) {
            self.drainCommands();
        }
    }

    /// Impact thump, louder and deeper with speed (m/s).
    pub fn playImpact(self: *AudioEngine, position: Vec3, speed: f32) void {
        self.playImpactOn(null, position, speed);
    }

    pub fn playImpactOn(self: *AudioEngine, bus: ?BusId, position: Vec3, speed: f32) void {
        const s = std.math.clamp(speed, 0.0, 15.0);
        self.play(.{
            .kind = .thump,
            .bus = bus,
            .position = position,
            .volume = 0.12 + 0.05 * s,
            .duration = 0.12 + 0.012 * s,
            .freq = 130.0 - 4.0 * s,
            .freq_end = 55.0,
        });
    }

    /// Explosion noise burst; size scales duration.
    pub fn playExplosion(self: *AudioEngine, position: Vec3, size: f32) void {
        self.playExplosionOn(null, position, size);
    }

    pub fn playExplosionOn(self: *AudioEngine, bus: ?BusId, position: Vec3, size: f32) void {
        self.play(.{
            .kind = .noise_burst,
            .bus = bus,
            .position = position,
            .volume = 0.85,
            .duration = 0.5 + 0.3 * size,
            .cutoff = 6000.0,
            .cutoff_end = 300.0,
        });
    }

    /// Short non-positional blip (UI, jumps, snaps).
    pub fn playBlip(self: *AudioEngine, freq: f32) void {
        self.playBlipOn(null, freq);
    }

    pub fn playBlipOn(self: *AudioEngine, bus: ?BusId, freq: f32) void {
        self.play(.{
            .kind = .blip,
            .bus = bus,
            .volume = 0.25,
            .duration = 0.09,
            .freq = freq,
            .freq_end = freq,
        });
    }

    pub const ClipPlayOptions = struct {
        bus: ?BusId = null,
        pan: f32 = 0.0,
        position: ?Vec3 = null, // null = non-positional
        min_distance: ?f32 = null, // null = inherit from bus
        max_distance: ?f32 = null, // null = inherit from bus
        rolloff: ?f32 = null, // null = inherit from bus
        attenuation_model: ?AttenuationModel = null, // null = inherit from bus
        doppler_factor: ?f32 = null, // null = inherit from bus
        velocity: ?Vec3 = null,
        volume: f32 = 1.0,
        loop: bool = false,
        rate: f32 = 1.0, // playback speed multiplier
        pitch_randomness: f32 = 0.0,
        occlusion: f32 = 0.0,
        occlusion_config: AudioOcclusionConfig = .{},
    };

    /// Plays a decoded WAV clip on a mixer voice. Reuses the same
    /// voice-stealing and attenuation/pan as `play()`. The clip is only
    /// read (never mutated), so sharing it with the audio thread is safe,
    /// but the clip MUST outlive any voice using it: do not free or
    /// mutate it while it may still be playing.
    pub fn playClip(self: *AudioEngine, clip: *const AudioClip, options: ClipPlayOptions) void {
        if (clip.frames == 0 or clip.samples.len < clip.frames * 2) return;
        if (clip.sample_rate <= 0.0) return;
        if (!std.math.isFinite(options.rate) or options.rate <= 1e-6) return;

        const engine_rate = self.sample_rate;
        if (engine_rate <= 0.0) return;

        const is_spatial = if (options.bus) |b| self.isBusSpatial(b) else (options.position != null);
        var sp_vol: f32 = options.volume;
        var sp_pan: f32 = options.pan;
        var sp_doppler: f32 = 1.0;

        if (is_spatial and options.position != null) {
            var min_dist: f32 = 1.0;
            var max_dist: f32 = 30.0;
            var rolloff: f32 = 1.0;
            var model: AttenuationModel = .inverse;
            var dop_scale: f32 = 1.0;

            if (options.bus) |b| {
                const bus_attn = self.getBusAttenuation(b);
                min_dist = options.min_distance orelse bus_attn.min_distance;
                max_dist = options.max_distance orelse bus_attn.max_distance;
                rolloff = options.rolloff orelse bus_attn.rolloff;
                model = options.attenuation_model orelse bus_attn.model;
                dop_scale = options.doppler_factor orelse self.getBusDopplerFactor(b);
            } else {
                min_dist = options.min_distance orelse 1.0;
                max_dist = options.max_distance orelse 30.0;
                rolloff = options.rolloff orelse 1.0;
                model = options.attenuation_model orelse .inverse;
                dop_scale = options.doppler_factor orelse 1.0;
            }

            const sp = spatializeWith(
                self.listener_pos,
                self.listener_right,
                self.listener_vel,
                options.position,
                options.volume,
                min_dist,
                max_dist,
                rolloff,
                model,
                options.velocity,
            );
            sp_vol = sp.vol;
            sp_pan = sp.pan;
            if (dop_scale > 0.0) {
                sp_doppler = 1.0 + (sp.doppler - 1.0) * dop_scale;
            }
        }

        var eff_cutoff: f32 = 20000.0;
        if (options.occlusion > 0.0) {
            const occ = std.math.clamp(options.occlusion, 0.0, 1.0);
            sp_vol *= std.math.lerp(1.0, options.occlusion_config.min_volume, occ);
            eff_cutoff = std.math.lerp(options.occlusion_config.max_cutoff, options.occlusion_config.min_cutoff, occ);
        }
        if (sp_vol <= 0.001) return;

        self.next_seed +%= 1;
        const seed = self.next_seed *% 2654435761 +% 97;

        var eff_rate = options.rate * sp_doppler;
        if (options.pitch_randomness > 0.0) {
            const r = @as(f32, @floatFromInt(seed & 0xFFFF)) / 65535.0;
            const rnd_offset = (r * 2.0 - 1.0) * options.pitch_randomness;
            eff_rate *= @max(0.1, 1.0 + rnd_offset);
        }
        eff_rate = std.math.clamp(eff_rate, 0.05, 20.0);

        const rate_f64: f64 = @floatCast(eff_rate);
        const step: f64 = @as(f64, @floatCast(clip.sample_rate / engine_rate)) * rate_f64;
        if (!std.math.isFinite(step) or step <= 0.0) return;
        const clip_dur: f64 = @as(f64, @floatFromInt(clip.frames)) / @as(f64, @floatCast(clip.sample_rate)) / rate_f64;
        const duration = @max(@as(f32, @floatCast(clip_dur)), 0.01);

        const cmd = Command{
            .clip = .{
                .clip = clip,
                .bus = options.bus,
                .volume = sp_vol,
                .pan = sp_pan,
                .duration = duration,
                .sample_step = step,
                .loop = options.loop,
                .seed = seed,
                .cutoff = eff_cutoff,
            },
        };

        _ = self.pushCommand(cmd);
        if (!self.started) {
            self.drainCommands();
        }
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
        var slot: ?usize = null;
        const cap = self.getBusCapacity();
        for (0..cap) |i| {
            if (!self.bus_active[i].load(.acquire)) {
                slot = i;
                break;
            }
        }
        const idx = slot orelse return null;
        self.setBusNameRaw(idx, config.name);
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
            self.setBusFilter(idx_bus, config.filter.filter_type, config.filter.cutoff, config.filter.q);
        }
        if (config.reverb) |rev_cfg| {
            _ = self.setBusReverb(idx_bus, rev_cfg);
        }
        return idx_bus;
    }

    pub fn createSpatialBus(self: *AudioEngine, name: []const u8, config: ?BusConfig) ?BusId {
        var cfg = config orelse BusConfig{};
        cfg.name = name;
        cfg.spatial = true;
        return self.createBus(cfg);
    }

    pub fn createNonSpatialBus(self: *AudioEngine, name: []const u8, volume: f32) ?BusId {
        return self.createBus(.{
            .name = name,
            .spatial = false,
            .volume = volume,
        });
    }

    pub fn configureBus(self: *AudioEngine, bus: BusId, config: BusConfig) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        if (config.name.len > 0) {
            self.setBusNameRaw(id, config.name);
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
            self.setBusFilter(bus, config.filter.filter_type, config.filter.cutoff, config.filter.q);
        }
        if (config.reverb) |rev_cfg| {
            _ = self.setBusReverb(bus, rev_cfg);
        }
        self.setBusOcclusionConfig(bus, config.occlusion_config);
        self.setBusOcclusion(bus, config.occlusion);
    }

    pub fn destroyBus(self: *AudioEngine, bus: BusId) void {
        const id = @intFromEnum(bus);
        const cap = self.getBusCapacity();
        if (id >= cap) return;
        if (!self.bus_active[id].load(.acquire)) return;
        self.stopBus(bus);
        self.clearBusFilter(bus);
        self.clearBusReverb(bus);
        self.clearBusOcclusion(bus);
        const my_parent = self.bus_parent[id].load(.acquire);
        for (0..cap) |i| {
            if (self.bus_active[i].load(.acquire) and self.bus_parent[i].load(.acquire) == id) {
                self.bus_parent[i].store(my_parent, .release);
            }
        }
        self.bus_active[id].store(false, .release);
        self.bus_name_lens[id] = 0;
    }

    pub fn findBus(self: *const AudioEngine, name: []const u8) ?BusId {
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

    pub fn findOrCreateBus(self: *AudioEngine, name: []const u8, config: BusConfig) ?BusId {
        if (self.findBus(name)) |b| return b;
        var cfg = config;
        cfg.name = name;
        return self.createBus(cfg);
    }

    pub fn isBusActive(self: *const AudioEngine, bus: BusId) bool {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return false;
        return self.bus_active[id].load(.acquire);
    }

    pub fn getBusCount(self: *const AudioEngine) usize {
        var count: usize = 0;
        const cap = self.getBusCapacity();
        for (0..cap) |i| {
            if (self.bus_active[i].load(.acquire)) count += 1;
        }
        return count;
    }

    pub fn isBusSpatial(self: *const AudioEngine, bus: BusId) bool {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return false;
        return self.bus_spatial[id].load(.acquire);
    }

    pub fn setBusSpatial(self: *AudioEngine, bus: BusId, spatial: bool) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.bus_spatial[id].store(spatial, .release);
    }

    pub fn getBusName(self: *const AudioEngine, bus: BusId) []const u8 {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return "";
        const len = self.bus_name_lens[id];
        return self.bus_names[id][0..len];
    }

    pub fn setBusName(self: *AudioEngine, bus: BusId, name: []const u8) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.setBusNameRaw(id, name);
    }

    fn setBusNameRaw(self: *AudioEngine, idx: usize, name: []const u8) void {
        if (idx >= self.getBusCapacity()) return;
        const len = @min(name.len, 31);
        @memcpy(self.bus_names[idx][0..len], name[0..len]);
        self.bus_name_lens[idx] = @intCast(len);
    }

    pub fn setBusParent(self: *AudioEngine, bus: BusId, parent: ?BusId) void {
        const id = @intFromEnum(bus);
        const cap = self.getBusCapacity();
        if (id >= cap) return;
        if (parent) |p| {
            if (@intFromEnum(p) >= cap) return;
        }
        const p_val: u8 = if (parent) |p| @intFromEnum(p) else 0xFF;
        self.bus_parent[id].store(p_val, .release);
    }

    pub fn getBusParent(self: *const AudioEngine, bus: BusId) ?BusId {
        const id = @intFromEnum(bus);
        const cap = self.getBusCapacity();
        if (id >= cap) return null;
        const p = self.bus_parent[id].load(.acquire);
        if (p >= cap) return null;
        return @enumFromInt(p);
    }

    pub fn setBusAttenuation(self: *AudioEngine, bus: BusId, model: AttenuationModel, min_dist: f32, max_dist: f32, rolloff: f32) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.bus_model[id].store(@intFromEnum(model), .release);
        self.bus_min_dist[id].store(@bitCast(@max(0.001, min_dist)), .release);
        self.bus_max_dist[id].store(@bitCast(@max(0.002, max_dist)), .release);
        self.bus_rolloff[id].store(@bitCast(@max(0.0, rolloff)), .release);
    }

    pub fn getBusAttenuation(self: *const AudioEngine, bus: BusId) BusAttenuation {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return .{ .model = .inverse, .min_distance = 1.0, .max_distance = 30.0, .rolloff = 1.0 };
        return .{
            .model = @enumFromInt(self.bus_model[id].load(.acquire)),
            .min_distance = @bitCast(self.bus_min_dist[id].load(.acquire)),
            .max_distance = @bitCast(self.bus_max_dist[id].load(.acquire)),
            .rolloff = @bitCast(self.bus_rolloff[id].load(.acquire)),
        };
    }

    pub fn setBusDopplerFactor(self: *AudioEngine, bus: BusId, factor: f32) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.bus_doppler[id].store(@bitCast(std.math.clamp(factor, 0.0, 5.0)), .release);
    }

    pub fn getBusDopplerFactor(self: *const AudioEngine, bus: BusId) f32 {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return 1.0;
        return @bitCast(self.bus_doppler[id].load(.acquire));
    }

    pub fn setBusVolume(self: *AudioEngine, bus: BusId, vol: f32) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        const v = std.math.clamp(vol, 0.0, 2.0);
        self.bus_volumes[id].store(@bitCast(v), .release);
    }

    pub fn getBusVolume(self: *const AudioEngine, bus: BusId) f32 {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return 0.0;
        return @bitCast(self.bus_volumes[id].load(.acquire));
    }

    pub fn getBusEffectiveVolume(self: *const AudioEngine, bus_opt: ?BusId) f32 {
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

    pub fn setBusMuted(self: *AudioEngine, bus: BusId, muted_val: bool) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.bus_muted[id].store(muted_val, .release);
    }

    pub fn isBusMuted(self: *const AudioEngine, bus: BusId) bool {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return true;
        return self.bus_muted[id].load(.acquire);
    }

    pub fn toggleBusMuted(self: *AudioEngine, bus: BusId) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        var cur = self.bus_muted[id].load(.monotonic);
        while (self.bus_muted[id].cmpxchgWeak(cur, !cur, .acq_rel, .monotonic)) |next| {
            cur = next;
        }
    }

    pub fn stopBus(self: *AudioEngine, bus: BusId) void {
        for (&self.voices) |*v| {
            if (v.bus) |b| {
                if (b == bus) {
                    v.active = false;
                }
            }
        }
    }

    // --- Bus DSP Effect Methods (Biquad Filter & Reverb) ---

    pub fn setBusFilter(self: *AudioEngine, bus: BusId, filter_type: BiquadFilterType, cutoff: f32, q: f32) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.bus_filter_type[id].store(@intFromEnum(filter_type), .release);
        self.bus_filter_cutoff[id].store(@bitCast(cutoff), .release);
        self.bus_filter_q[id].store(@bitCast(q), .release);
        self.bus_filters[id].setParams(filter_type, cutoff, q, self.sample_rate);
    }

    pub fn setBusFilterCutoff(self: *AudioEngine, bus: BusId, cutoff: f32) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.bus_filter_cutoff[id].store(@bitCast(cutoff), .release);
        const f_type: BiquadFilterType = @enumFromInt(self.bus_filter_type[id].load(.acquire));
        const q: f32 = @bitCast(self.bus_filter_q[id].load(.acquire));
        self.bus_filters[id].setParams(f_type, cutoff, q, self.sample_rate);
    }

    pub fn setBusFilterQ(self: *AudioEngine, bus: BusId, q: f32) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.bus_filter_q[id].store(@bitCast(q), .release);
        const f_type: BiquadFilterType = @enumFromInt(self.bus_filter_type[id].load(.acquire));
        const cutoff: f32 = @bitCast(self.bus_filter_cutoff[id].load(.acquire));
        self.bus_filters[id].setParams(f_type, cutoff, q, self.sample_rate);
    }

    pub fn clearBusFilter(self: *AudioEngine, bus: BusId) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.bus_filter_type[id].store(@intFromEnum(BiquadFilterType.none), .release);
        self.bus_filters[id].setParams(.none, 1000.0, 0.7071, self.sample_rate);
        self.bus_filters[id].resetState();
    }

    pub fn getBusFilter(self: *const AudioEngine, bus: BusId) BusFilterConfig {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return .{};
        return .{
            .filter_type = @enumFromInt(self.bus_filter_type[id].load(.acquire)),
            .cutoff = @bitCast(self.bus_filter_cutoff[id].load(.acquire)),
            .q = @bitCast(self.bus_filter_q[id].load(.acquire)),
        };
    }

    pub fn setBusReverb(self: *AudioEngine, bus: BusId, config: BusReverbConfig) bool {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return false;
        if (!self.bus_active[id].load(.acquire)) return false;

        const current_slot = self.bus_reverb_slot[id].load(.acquire);
        if (current_slot < max_reverbs) {
            self.reverbs[current_slot].setConfig(config);
            self.reverbs[current_slot].active = true;
            return true;
        }

        // Allocate free reverb unit from pool
        for (0..max_reverbs) |slot| {
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

    pub fn clearBusReverb(self: *AudioEngine, bus: BusId) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        const slot = self.bus_reverb_slot[id].load(.acquire);
        if (slot < max_reverbs) {
            self.reverbs[slot].active = false;
            self.reverbs[slot].clear();
            self.reverb_bus_owner[slot].store(0xFF, .release);
            self.bus_reverb_slot[id].store(0xFF, .release);
        }
    }

    pub fn getBusReverb(self: *const AudioEngine, bus: BusId) ?BusReverbConfig {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return null;
        const slot = self.bus_reverb_slot[id].load(.acquire);
        if (slot >= max_reverbs) return null;
        return self.reverbs[slot].config;
    }

    pub fn isBusReverbEnabled(self: *const AudioEngine, bus: BusId) bool {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return false;
        return self.bus_reverb_slot[id].load(.acquire) < max_reverbs;
    }

    pub fn setBusUnderwater(self: *AudioEngine, bus: BusId) void {
        self.setBusFilter(bus, .lowpass, 600.0, 1.0);
    }

    pub fn setBusMuffled(self: *AudioEngine, bus: BusId) void {
        self.setBusFilter(bus, .lowpass, 1200.0, 0.7071);
    }

    pub fn setBusTelephone(self: *AudioEngine, bus: BusId) void {
        self.setBusFilter(bus, .bandpass, 1800.0, 1.5);
    }

    pub fn setBusCaveReverb(self: *AudioEngine, bus: BusId) bool {
        return self.setBusReverb(bus, .{ .room_size = 0.85, .damping = 0.25, .wet = 0.5, .dry = 0.7 });
    }

    pub fn setBusRoomReverb(self: *AudioEngine, bus: BusId) bool {
        return self.setBusReverb(bus, .{ .room_size = 0.45, .damping = 0.5, .wet = 0.25, .dry = 0.85 });
    }

    pub fn setBusOcclusion(self: *AudioEngine, bus: BusId, occlusion: f32) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        const val = std.math.clamp(occlusion, 0.0, 1.0);
        self.bus_occlusion[id].store(@bitCast(val), .release);
    }

    pub fn getBusOcclusion(self: *const AudioEngine, bus: BusId) f32 {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return 0.0;
        return @bitCast(self.bus_occlusion[id].load(.acquire));
    }

    pub fn setBusOcclusionConfig(self: *AudioEngine, bus: BusId, config: AudioOcclusionConfig) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.bus_occlusion_config[id] = config;
    }

    pub fn getBusOcclusionConfig(self: *const AudioEngine, bus: BusId) AudioOcclusionConfig {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return .{};
        return self.bus_occlusion_config[id];
    }

    pub fn clearBusOcclusion(self: *AudioEngine, bus: BusId) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        self.bus_occlusion[id].store(@bitCast(@as(f32, 0.0)), .release);
        self.bus_occlusion_filters[id].resetState();
    }

    pub fn updateBusOcclusion(self: *AudioEngine, bus: BusId, target_occlusion: f32, dt: f32) void {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return;
        const target = std.math.clamp(target_occlusion, 0.0, 1.0);
        const smooth_time = self.bus_occlusion_config[id].smooth_time;
        const cur = self.getBusOcclusion(bus);
        if (smooth_time <= 1e-4 or dt <= 1e-4) {
            self.setBusOcclusion(bus, target);
            return;
        }
        const alpha = 1.0 - @exp(-dt / @max(smooth_time, 0.001));
        const new_val = cur + (target - cur) * std.math.clamp(alpha, 0.0, 1.0);
        self.setBusOcclusion(bus, new_val);
    }

    pub fn evaluateOcclusion(
        self: *const AudioEngine,
        emitter_pos: Vec3,
        config: AudioOcclusionConfig,
        raycast_fn: RaycastFn,
        user_data: ?*anyopaque,
    ) f32 {
        return evaluateRaycastOcclusion(self.listener_pos, emitter_pos, config, raycast_fn, user_data);
    }

    pub fn updateBusOcclusionWithRaycast(
        self: *AudioEngine,
        bus: BusId,
        emitter_pos: Vec3,
        dt: f32,
        raycast_fn: RaycastFn,
        user_data: ?*anyopaque,
    ) f32 {
        const id = @intFromEnum(bus);
        if (id >= self.getBusCapacity()) return 0.0;
        const cfg = self.bus_occlusion_config[id];
        const raw = self.evaluateOcclusion(emitter_pos, cfg, raycast_fn, user_data);
        self.updateBusOcclusion(bus, raw, dt);
        return self.getBusOcclusion(bus);
    }

    pub fn stopAll(self: *AudioEngine) void {
        for (&self.voices) |*v| {
            v.active = false;
        }
    }

    // --- Audio Stream & Music Management ---

    pub fn registerStream(self: *AudioEngine, stream: *AudioStream) !void {
        for (&self.streams) |*slot| {
            if (slot.* == null) {
                slot.* = stream;
                return;
            }
        }
        return error.StreamLimitReached;
    }

    pub fn unregisterStream(self: *AudioEngine, stream: *AudioStream) void {
        for (&self.streams) |*slot| {
            if (slot.* == stream) {
                slot.* = null;
                break;
            }
        }
        if (self.music_stream == stream) self.music_stream = null;
        if (self.music_fade_stream == stream) self.music_fade_stream = null;
    }

    pub fn createStreamFromFile(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        path: []const u8,
        options: StreamOptions,
    ) !*AudioStream {
        const stream = try AudioStream.openFile(allocator, path, self.sample_rate, options);
        errdefer stream.deinit();
        try self.registerStream(stream);
        return stream;
    }

    pub fn createStreamFromMemory(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        options: StreamOptions,
    ) !*AudioStream {
        const stream = try AudioStream.openMemory(allocator, bytes, format, self.sample_rate, options);
        errdefer stream.deinit();
        try self.registerStream(stream);
        return stream;
    }

    pub fn destroyStream(self: *AudioEngine, stream: *AudioStream) void {
        self.unregisterStream(stream);
        stream.deinit();
    }

    pub fn updateStreams(self: *AudioEngine, dt: f32) void {
        for (&self.streams) |*slot| {
            if (slot.*) |s| {
                s.update(dt);
                if (s.isStopped() and (s.auto_destroy or s.stop_on_fade_out)) {
                    if (self.music_stream == s) self.music_stream = null;
                    if (self.music_fade_stream == s) self.music_fade_stream = null;
                    slot.* = null;
                    s.deinit();
                }
            }
        }
        if (self.music_fade_stream) |fade_s| {
            if (fade_s.isStopped()) {
                self.destroyStream(fade_s);
                self.music_fade_stream = null;
            }
        }
    }

    // ========================================================================
    // Sound Playback & Streaming API
    // ========================================================================

    /// Plays a sound from a file path (OGG, MP3, WAV).
    /// Streams audio in background with real-time safety.
    pub fn playSound(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        path: []const u8,
        options: PlaySoundOptions,
    ) !*AudioStream {
        return self.createStreamFromFile(allocator, path, options);
    }

    /// Plays a sound from an in-memory byte slice (OGG, MP3, WAV).
    pub fn playSoundFromMemory(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        options: PlaySoundOptions,
    ) !*AudioStream {
        return self.createStreamFromMemory(allocator, bytes, format, options);
    }

    /// Fire-and-forget sound playback: automatically destroys and frees the stream
    /// when playback finishes or is stopped.
    pub fn playSoundOnce(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        path: []const u8,
        options: PlaySoundOptions,
    ) !void {
        var opt = options;
        opt.loop = false;
        opt.auto_destroy = true;
        _ = try self.playSound(allocator, path, opt);
    }

    /// Fire-and-forget in-memory sound playback.
    pub fn playSoundOnceFromMemory(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        options: PlaySoundOptions,
    ) !void {
        var opt = options;
        opt.loop = false;
        opt.auto_destroy = true;
        _ = try self.playSoundFromMemory(allocator, bytes, format, opt);
    }

    /// Stops a sound stream immediately, or over `fade_duration` seconds.
    pub fn stopSound(self: *AudioEngine, stream: *AudioStream, fade_duration: f32) void {
        if (fade_duration > 0.0) {
            stream.fadeTo(0.0, fade_duration, true);
        } else {
            self.destroyStream(stream);
        }
    }

    /// Pauses an active sound stream.
    pub fn pauseSound(self: *AudioEngine, stream: *AudioStream) void {
        _ = self;
        stream.pause();
    }

    /// Resumes a paused sound stream.
    pub fn resumeSound(self: *AudioEngine, stream: *AudioStream) void {
        _ = self;
        stream.unpause();
    }

    /// Sets the volume of an active sound stream.
    pub fn setSoundVolume(self: *AudioEngine, stream: *AudioStream, vol: f32) void {
        _ = self;
        stream.setVolume(vol);
    }

    /// Stops all currently active sound streams.
    pub fn stopAllSounds(self: *AudioEngine, fade_duration: f32) void {
        for (&self.streams) |maybe_s| {
            if (maybe_s) |s| {
                if (fade_duration > 0.0) {
                    s.fadeTo(0.0, fade_duration, true);
                } else {
                    self.destroyStream(s);
                }
            }
        }
    }

    /// Crossfades from an existing sound stream to a new sound file.
    pub fn crossfadeSound(
        self: *AudioEngine,
        old_stream: ?*AudioStream,
        allocator: std.mem.Allocator,
        path: []const u8,
        fade_duration: f32,
        options: PlaySoundOptions,
    ) !*AudioStream {
        if (old_stream) |s| {
            s.fadeTo(0.0, fade_duration, true);
        }

        var opt = options;
        const target_vol = opt.volume;
        opt.volume = 0.0;
        const new_s = try self.createStreamFromFile(allocator, path, opt);
        new_s.fadeTo(target_vol, fade_duration, false);
        return new_s;
    }

    /// Crossfades from an existing sound stream to a new in-memory sound.
    pub fn crossfadeSoundFromMemory(
        self: *AudioEngine,
        old_stream: ?*AudioStream,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        fade_duration: f32,
        options: PlaySoundOptions,
    ) !*AudioStream {
        if (old_stream) |s| {
            s.fadeTo(0.0, fade_duration, true);
        }

        var opt = options;
        const target_vol = opt.volume;
        opt.volume = 0.0;
        const new_s = try self.createStreamFromMemory(allocator, bytes, format, opt);
        new_s.fadeTo(target_vol, fade_duration, false);
        return new_s;
    }

    // ========================================================================
    // Dedicated Music Slot Convenience Helpers
    // ========================================================================

    /// Convenience helper for playing background music / looping track.
    pub fn playMusic(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        path: []const u8,
        options: PlaySoundOptions,
    ) !*AudioStream {
        if (self.music_stream) |old_s| {
            self.destroyStream(old_s);
            self.music_stream = null;
        }
        var opt = options;
        opt.loop = true;
        const s = try self.playSound(allocator, path, opt);
        self.music_stream = s;
        return s;
    }

    pub fn playMusicFromMemory(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        options: PlaySoundOptions,
    ) !*AudioStream {
        if (self.music_stream) |old_s| {
            self.destroyStream(old_s);
            self.music_stream = null;
        }
        var opt = options;
        opt.loop = true;
        const s = try self.playSoundFromMemory(allocator, bytes, format, opt);
        self.music_stream = s;
        return s;
    }

    pub fn crossfadeMusic(
        self: *AudioEngine,
        allocator: std.mem.Allocator,
        path: []const u8,
        fade_duration: f32,
        options: PlaySoundOptions,
    ) !*AudioStream {
        const old_s = self.music_stream;
        self.music_stream = null;
        var opt = options;
        opt.loop = true;
        const new_s = try self.crossfadeSound(old_s, allocator, path, fade_duration, opt);
        self.music_stream = new_s;
        return new_s;
    }

    pub fn stopMusic(self: *AudioEngine, fade_duration: f32) void {
        if (self.music_stream) |s| {
            self.stopSound(s, fade_duration);
            self.music_stream = null;
        }
    }

    pub fn pauseMusic(self: *AudioEngine) void {
        if (self.music_stream) |s| s.pause();
    }

    pub fn resumeMusic(self: *AudioEngine) void {
        if (self.music_stream) |s| s.unpause();
    }

    pub fn setMusicVolume(self: *AudioEngine, vol: f32) void {
        if (self.music_stream) |s| s.setVolume(vol);
    }

    pub fn getMusicStream(self: *const AudioEngine) ?*AudioStream {
        return self.music_stream;
    }

    pub const MusicPlayOptions = struct {
        bus: ?BusId = null,
        volume: f32 = 1.0,
        loop: bool = true,
        rate: f32 = 1.0,
    };

    pub fn playMusicClip(self: *AudioEngine, clip: *const AudioClip, options: MusicPlayOptions) void {
        if (options.bus) |b| {
            self.stopBus(b);
        }
        self.playClip(clip, .{
            .bus = options.bus,
            .volume = options.volume,
            .loop = options.loop,
            .rate = options.rate,
            .position = null,
        });
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

    /// Mixes stereo-interleaved frames and advances voice state. Called by
    /// the stream callback and directly by tests.
    pub fn renderFrames(self: *AudioEngine, buffer: []f32) void {
        self.drainCommands();
        std.debug.assert(buffer.len % 2 == 0);
        const total: usize = buffer.len / 2;
        if (total == 0) return;
        const dt: f32 = 1.0 / self.sample_rate;
        const master: f32 = if (self.isMuted()) 0.0 else self.getMasterVolume();
        const cap = self.getBusCapacity();

        // 1. Update Biquad filter parameters if changed atomically
        for (0..cap) |i| {
            if (self.bus_active[i].load(.acquire)) {
                const f_type: BiquadFilterType = @enumFromInt(self.bus_filter_type[i].load(.acquire));
                const f_cut: f32 = @bitCast(self.bus_filter_cutoff[i].load(.acquire));
                const f_q: f32 = @bitCast(self.bus_filter_q[i].load(.acquire));
                if (self.bus_filters[i].filter_type != f_type or
                    self.bus_filters[i].cutoff != f_cut or
                    self.bus_filters[i].q != f_q or
                    self.bus_filters[i].last_sample_rate != self.sample_rate)
                {
                    self.bus_filters[i].setParams(f_type, f_cut, f_q, self.sample_rate);
                }
            }
        }

        // 2. Build topological evaluation order (leaves first, parents after children)
        var active_buses: [max_bus_capacity]u8 = undefined;
        var active_count: usize = 0;
        var heights: [max_bus_capacity]u8 = [_]u8{0} ** max_bus_capacity;
        var parents: [max_bus_capacity]u8 = undefined;

        for (0..cap) |i| {
            if (self.bus_active[i].load(.acquire)) {
                active_buses[active_count] = @intCast(i);
                active_count += 1;
                parents[i] = self.bus_parent[i].load(.acquire);
            }
        }

        if (active_count > 1) {
            var pass: usize = 0;
            while (pass < 8) : (pass += 1) {
                var changed = false;
                for (active_buses[0..active_count]) |b| {
                    const p = parents[b];
                    if (p < cap and p != b and self.bus_active[p].load(.acquire)) {
                        if (heights[p] <= heights[b]) {
                            heights[p] = heights[b] + 1;
                            changed = true;
                        }
                    }
                }
                if (!changed) break;
            }

            var i: usize = 1;
            while (i < active_count) : (i += 1) {
                const key = active_buses[i];
                const key_h = heights[key];
                var j: usize = i;
                while (j > 0 and heights[active_buses[j - 1]] > key_h) : (j -= 1) {
                    active_buses[j] = active_buses[j - 1];
                }
                active_buses[j] = key;
            }
        }

        // 3. Collect active voices
        var list: [max_voices]ActiveVoice = undefined;
        var voice_count: usize = 0;
        for (&self.voices) |*v| {
            if (!v.active) continue;
            const pan = std.math.clamp(v.pan, -1.0, 1.0);
            const g = v.volume;
            const dur = @max(v.duration, 1e-6);
            var a = ActiveVoice{
                .v = v,
                .kind = v.kind,
                .gl = g * @sqrt(0.5 * (1.0 - pan)),
                .gr = g * @sqrt(0.5 * (1.0 + pan)),
                .duration = v.duration,
                .inv_duration = 1.0 / dur,
                .freq0 = v.freq,
                .freq_slope = (v.freq_end - v.freq) / dur,
                .cutoff0 = v.cutoff,
                .cutoff_slope = (v.cutoff_end - v.cutoff) / dur,
                .step = v.sample_step,
                .loop = v.loop,
            };
            if (v.kind == .sample) {
                const clip = v.clip orelse {
                    v.active = false;
                    continue;
                };
                if (clip.frames == 0 or clip.sample_rate <= 0.0 or clip.samples.len < clip.frames * 2) {
                    v.active = false;
                    continue;
                }
                if (!std.math.isFinite(v.sample_step) or v.sample_step <= 0.0) {
                    v.active = false;
                    continue;
                }
                var pos = v.sample_pos;
                if (!std.math.isFinite(pos)) {
                    v.active = false;
                    continue;
                }
                const n: f64 = @floatFromInt(clip.frames);
                if (pos < 0.0 or pos >= n) {
                    if (v.loop) {
                        pos = @mod(pos, n);
                        if (!(pos >= 0.0)) pos += n;
                        v.sample_pos = pos;
                    } else {
                        v.active = false;
                        continue;
                    }
                }
                a.clip = clip.samples[0 .. clip.frames * 2];
                a.frames = clip.frames;
                a.step_one = v.sample_step == 1.0 and pos == @floor(pos);
                list[voice_count] = a;
            } else {
                list[voice_count] = a;
            }
            voice_count += 1;
        }

        // Fast-path: if no voices, no active reverbs, and no active streams, silence buffer
        var any_reverb = false;
        for (&self.reverbs) |*r| {
            if (r.active) {
                any_reverb = true;
                break;
            }
        }
        var any_stream = false;
        for (&self.streams) |*maybe_s| {
            if (maybe_s.*) |s| {
                if (s.state.load(.acquire) == .playing) {
                    any_stream = true;
                    break;
                }
            }
        }
        if (voice_count == 0 and !any_reverb and !any_stream) {
            @memset(buffer, 0.0);
            return;
        }

        // 4. Chunked block rendering (64 frames = 128 samples per block)
        var offset: usize = 0;
        var master_chunk: [chunk_samples]f32 = undefined;
        var bus_has_audio: [max_bus_capacity]bool = undefined;

        while (offset < buffer.len) {
            const remaining = buffer.len - offset;
            const cur_samples = @min(remaining, chunk_samples);

            @memset(master_chunk[0..cur_samples], 0.0);
            for (active_buses[0..active_count]) |b| {
                @memset(self.bus_chunks[b][0..cur_samples], 0.0);
                bus_has_audio[b] = false;
            }

            for (list[0..voice_count]) |*a| {
                if (!a.v.active) continue;
                const target = if (a.v.bus) |b_id| blk: {
                    const idx = @intFromEnum(b_id);
                    if (idx < cap and self.bus_active[idx].load(.acquire)) {
                        bus_has_audio[idx] = true;
                        break :blk self.bus_chunks[idx][0..cur_samples];
                    }
                    break :blk master_chunk[0..cur_samples];
                } else master_chunk[0..cur_samples];

                switch (a.kind) {
                    .noise_burst => renderNoise(a.*, dt, target),
                    .thump => renderTone(true, a.*, dt, target),
                    .blip => renderTone(false, a.*, dt, target),
                    .sample => renderSample(a.*, dt, target),
                }
            }

            // Render active audio streams
            for (&self.streams) |*maybe_s| {
                const s = maybe_s.* orelse continue;
                if (s.state.load(.acquire) != .playing) continue;
                const bus_opt = s.getBus();
                const target = if (bus_opt) |b_id| blk: {
                    const idx = @intFromEnum(b_id);
                    if (idx < cap and self.bus_active[idx].load(.acquire)) {
                        bus_has_audio[idx] = true;
                        break :blk self.bus_chunks[idx][0..cur_samples];
                    }
                    break :blk master_chunk[0..cur_samples];
                } else master_chunk[0..cur_samples];

                _ = s.renderToBuffer(target);
            }

            // Process buses in leaf-to-root topological order
            for (active_buses[0..active_count]) |b| {
                const r_slot = self.bus_reverb_slot[b].load(.acquire);
                const has_rev = (r_slot < max_reverbs and self.reverbs[r_slot].active);
                const occ: f32 = @bitCast(self.bus_occlusion[b].load(.acquire));

                if (!bus_has_audio[b] and !has_rev) continue;

                if (self.bus_filters[b].filter_type != .none) {
                    self.bus_filters[b].processBuffer(self.bus_chunks[b][0..cur_samples]);
                }

                if (occ > 0.001) {
                    const o_cfg = self.bus_occlusion_config[b];
                    const occ_cutoff = std.math.lerp(o_cfg.max_cutoff, o_cfg.min_cutoff, occ);
                    self.bus_occlusion_filters[b].setParams(.lowpass, occ_cutoff, 0.7071, self.sample_rate);
                    self.bus_occlusion_filters[b].processBuffer(self.bus_chunks[b][0..cur_samples]);
                }

                if (has_rev) {
                    self.reverbs[r_slot].processBuffer(self.bus_chunks[b][0..cur_samples]);
                }

                if (self.bus_muted[b].load(.acquire)) {
                    @memset(self.bus_chunks[b][0..cur_samples], 0.0);
                } else {
                    var vol: f32 = @bitCast(self.bus_volumes[b].load(.acquire));
                    if (occ > 0.001) {
                        vol *= std.math.lerp(1.0, self.bus_occlusion_config[b].min_volume, occ);
                    }
                    if (vol != 1.0) {
                        for (self.bus_chunks[b][0..cur_samples]) |*s| {
                            s.* *= vol;
                        }
                    }
                }

                const p = parents[b];
                if (p < cap and p != b and self.bus_active[p].load(.acquire)) {
                    for (0..cur_samples) |k| {
                        self.bus_chunks[p][k] += self.bus_chunks[b][k];
                    }
                    bus_has_audio[p] = true;
                } else {
                    for (0..cur_samples) |k| {
                        master_chunk[k] += self.bus_chunks[b][k];
                    }
                }
            }

            if (master == 0.0) {
                @memset(buffer[offset .. offset + cur_samples], 0.0);
            } else {
                var k: usize = 0;
                while (k < cur_samples) : (k += 2) {
                    const l = master_chunk[k] * master;
                    const r = master_chunk[k + 1] * master;
                    buffer[offset + k] = l / (1.0 + @abs(l));
                    buffer[offset + k + 1] = r / (1.0 + @abs(r));
                }
            }

            offset += cur_samples;
        }
    }

    /// Noise burst core: per-sample LCG noise + one-pole lowpass. The
    /// cutoff ramp slope and filter constant are hoisted; `alpha` stays
    /// per-sample since the filter state integrates every sample.
    inline fn renderNoise(a: ActiveVoice, dt: f32, buffer: []f32) void {
        const v = a.v;
        const omega: f32 = 2.0 * std.math.pi * dt; // hoisted filter constant
        var t = v.t;
        var seed = v.seed;
        var lp = v.lp;
        const total: usize = buffer.len / 2;
        var j: usize = 0;
        while (j < total) : (j += 1) {
            t += dt;
            if (t >= a.duration) break;
            const k = t * a.inv_duration;
            const env = (1.0 - k) * (1.0 - k);
            const cutoff = @max(a.cutoff0 + a.cutoff_slope * t, 30.0);
            const alpha = 1.0 - @exp(-omega * cutoff);
            seed = seed *% 1664525 +% 1013904223;
            const u: f32 = @floatFromInt(seed >> 9); // 0..2^23-1
            const n = u * (2.0 / 8388608.0) - 1.0;
            lp += alpha * (n - lp);
            const s = lp * 1.6 * env;
            buffer[j * 2] += s * a.gl;
            buffer[j * 2 + 1] += s * a.gr;
        }
        v.t = t;
        v.seed = seed;
        v.lp = lp;
        if (t >= a.duration) v.active = false;
    }

    /// Oscillator core: phase and envelope stay per-sample (they *are* the
    /// signal); only the freq ramp slope is hoisted to one multiply-add.
    /// `thump_click` is comptime so `.blip` pays no branch for the click.
    inline fn renderTone(comptime thump_click: bool, a: ActiveVoice, dt: f32, buffer: []f32) void {
        const v = a.v;
        const two_pi: f32 = 2.0 * std.math.pi;
        var t = v.t;
        var phase = v.phase;
        const total: usize = buffer.len / 2;
        var j: usize = 0;
        while (j < total) : (j += 1) {
            t += dt;
            if (t >= a.duration) break;
            const k = t * a.inv_duration;
            const env = (1.0 - k) * (1.0 - k);
            const freq = @max(a.freq0 + a.freq_slope * t, 20.0);
            phase += freq * dt;
            const s = @sin(phase * two_pi);
            const click: f32 = if (thump_click and t < 0.008) 0.5 * (1.0 - t / 0.008) else 0.0;
            const o = (s * 0.8 + click) * env;
            buffer[j * 2] += o * a.gl;
            buffer[j * 2 + 1] += o * a.gr;
        }
        v.t = t;
        v.phase = phase;
        if (t >= a.duration) v.active = false;
    }

    /// Sample playback core. Clip validity, null/empty checks and the
    /// initial position wrap happen once in `renderFrames`; here only the
    /// position advance can end/wrap the voice, with unchanged semantics:
    /// the last frame stays audible, then a non-looping voice deactivates.
    inline fn renderSample(a: ActiveVoice, dt: f32, buffer: []f32) void {
        const v = a.v;
        const s = a.clip.?; // validated during setup
        const n: f64 = @floatFromInt(a.frames);
        var pos = v.sample_pos;
        var t = v.t;
        var lp_l = v.lp;
        var lp_r = v.lp_r;
        const filter_active = a.cutoff0 < 19000.0;
        const omega: f32 = 2.0 * std.math.pi * dt;
        const alpha: f32 = if (filter_active) 1.0 - @exp(-omega * @max(a.cutoff0, 40.0)) else 1.0;
        const total: usize = buffer.len / 2;
        var j: usize = 0;
        if (a.step_one) {
            // Fast path: 1:1 step from an integer position, direct copy.
            while (j < total) : (j += 1) {
                t += dt;
                if (a.loop and a.duration > 0.0 and t >= a.duration) t = @mod(t, a.duration);
                if (pos >= n) {
                    if (a.loop) {
                        pos = @mod(pos, n);
                    } else {
                        v.sample_pos = pos;
                        v.t = t;
                        v.lp = lp_l;
                        v.lp_r = lp_r;
                        v.active = false;
                        return;
                    }
                }
                const idx: usize = @intFromFloat(pos);
                var smp_l = s[idx * 2];
                var smp_r = s[idx * 2 + 1];
                if (filter_active) {
                    lp_l += alpha * (smp_l - lp_l);
                    lp_r += alpha * (smp_r - lp_r);
                    if (@abs(lp_l) < 1e-15) lp_l = 0.0;
                    if (@abs(lp_r) < 1e-15) lp_r = 0.0;
                    smp_l = lp_l;
                    smp_r = lp_r;
                }
                buffer[j * 2] += smp_l * a.gl;
                buffer[j * 2 + 1] += smp_r * a.gr;
                pos += 1.0;
            }
        } else {
            // Resampling path: linear interpolation between frames.
            while (j < total) : (j += 1) {
                t += dt;
                if (a.loop and a.duration > 0.0 and t >= a.duration) t = @mod(t, a.duration);
                if (pos >= n) {
                    if (a.loop) {
                        pos = @mod(pos, n);
                    } else {
                        v.sample_pos = pos;
                        v.t = t;
                        v.lp = lp_l;
                        v.lp_r = lp_r;
                        v.active = false;
                        return;
                    }
                }
                const fl = @floor(pos);
                const idx0: usize = @intFromFloat(fl);
                const frac: f32 = @floatCast(pos - fl);
                const x = @min(idx0, a.frames - 1);
                const y = @min(x + 1, a.frames - 1);
                var smp_l = s[x * 2] + (s[y * 2] - s[x * 2]) * frac;
                var smp_r = s[x * 2 + 1] + (s[y * 2 + 1] - s[x * 2 + 1]) * frac;
                if (filter_active) {
                    lp_l += alpha * (smp_l - lp_l);
                    lp_r += alpha * (smp_r - lp_r);
                    if (@abs(lp_l) < 1e-15) lp_l = 0.0;
                    if (@abs(lp_r) < 1e-15) lp_r = 0.0;
                    smp_l = lp_l;
                    smp_r = lp_r;
                }
                buffer[j * 2] += smp_l * a.gl;
                buffer[j * 2 + 1] += smp_r * a.gr;
                pos += a.step;
            }
        }
        v.sample_pos = pos;
        v.t = t;
        v.lp = lp_l;
        v.lp_r = lp_r;
    }

    fn streamCallback(buffer: [*c]f32, num_frames: i32, num_channels: i32, user_data: ?*anyopaque) callconv(.c) void {
        const self: *AudioEngine = @ptrCast(@alignCast(user_data orelse return));
        if (num_channels != 2) return;
        const n: usize = @intCast(num_frames);
        self.renderFrames(buffer[0 .. n * 2]);
    }
};

pub const VoiceKind = enum {
    noise_burst,
    thump,
    blip,
    sample,
};

pub const Voice = struct {
    active: bool = false,
    kind: VoiceKind = .thump,
    bus: ?BusId = null,
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
};

test {
    _ = @import("audio/tests.zig");
}
