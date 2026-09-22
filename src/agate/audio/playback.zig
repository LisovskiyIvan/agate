//! Procedural voice triggers + spatial playback entry points.
//! Split out of `audio.zig` (facade).
//!
//! This module owns the synth/sampling trigger API: `PlayOptions`/`play`
//! (kind + spatial position + envelope/filter sweep + pitch/doppler), the
//! one-shot helpers (`playImpact`/`playImpactOn`, `playExplosion`/
//! `playExplosionOn`, `playBlip`/`playBlipOn`), `ClipPlayOptions`/`playClip`
//! (decoded-clip voices sharing the same stealing/attenuation path), and
//! the shared distance/pan/doppler math (`SpatializeResult`,
//! `spatializeWith`). `AudioEngine` re-exports the option/result types as
//! nested constants (`AudioEngine.PlayOptions`, ...) and forwards every
//! method, so call sites keep working unchanged.
//!
//! Anti-cycle rule (same as `profiler/*`, `particles/*`): every function
//! takes the engine as `anytype` (a `*AudioEngine` from `engine.zig` in
//! practice) and this module never imports `engine.zig` or the `audio.zig`
//! facade back. Bus queries reach the `buses.zig` sibling directly and
//! trigger enqueue reaches `commands.zig` directly (documented in the
//! facade); engine-resident state (listener, seed, sample rate) is read
//! through `anytype` fields. `spatializeWith` is pure (no `self` at all).

const std = @import("std");
const math = @import("math");
const clip_mod = @import("clip.zig");
const occlusion_mod = @import("occlusion.zig");
const types = @import("types.zig");
const commands = @import("commands.zig");
const buses = @import("buses.zig");

const Vec3 = math.Vec3;
const BusId = types.BusId;
const VoiceKind = types.VoiceKind;
const AudioClip = clip_mod.AudioClip;
const AttenuationModel = types.AttenuationModel;
const AudioOcclusionConfig = occlusion_mod.AudioOcclusionConfig;
const max_distance = types.max_distance;

// Local aliases so the moved bodies stay byte-identical.
const isBusSpatial = buses.isBusSpatial;
const getBusAttenuation = buses.getBusAttenuation;
const getBusDopplerFactor = buses.getBusDopplerFactor;
const pushCommand = commands.pushCommand;
const drainCommands = commands.drainCommands;

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

pub const SpatializeResult = struct {
    vol: f32,
    pan: f32,
    doppler: f32,
};

/// Shared distance attenuation + pan for `play` and `playClip`.
/// Dead (no callers) before the split; moved verbatim so the file's
/// content is preserved exactly.
fn spatialize(self: anytype, position: ?Vec3, volume: f32) struct { vol: f32, pan: f32 } {
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

pub fn play(self: anytype, params: PlayOptions) void {
    const is_spatial = if (params.bus) |b| isBusSpatial(self, b) else (params.position != null);
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
            const bus_attn = getBusAttenuation(self, b);
            min_dist = params.min_distance orelse bus_attn.min_distance;
            max_dist = params.max_distance orelse bus_attn.max_distance;
            rolloff = params.rolloff orelse bus_attn.rolloff;
            model = params.attenuation_model orelse bus_attn.model;
            dop_scale = params.doppler_factor orelse getBusDopplerFactor(self, b);
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
    const cmd = commands.Command{
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

    _ = pushCommand(self, cmd);
    if (!self.started) {
        drainCommands(self);
    }
}

/// Impact thump, louder and deeper with speed (m/s).
pub fn playImpact(self: anytype, position: Vec3, speed: f32) void {
    playImpactOn(self, null, position, speed);
}

pub fn playImpactOn(self: anytype, bus: ?BusId, position: Vec3, speed: f32) void {
    const s = std.math.clamp(speed, 0.0, 15.0);
    play(self, .{
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
pub fn playExplosion(self: anytype, position: Vec3, size: f32) void {
    playExplosionOn(self, null, position, size);
}

pub fn playExplosionOn(self: anytype, bus: ?BusId, position: Vec3, size: f32) void {
    play(self, .{
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
pub fn playBlip(self: anytype, freq: f32) void {
    playBlipOn(self, null, freq);
}

pub fn playBlipOn(self: anytype, bus: ?BusId, freq: f32) void {
    play(self, .{
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
pub fn playClip(self: anytype, clip: *const AudioClip, options: ClipPlayOptions) void {
    if (clip.frames == 0 or clip.samples.len < clip.frames * 2) return;
    if (clip.sample_rate <= 0.0) return;
    if (!std.math.isFinite(options.rate) or options.rate <= 1e-6) return;

    const engine_rate = self.sample_rate;
    if (engine_rate <= 0.0) return;

    const is_spatial = if (options.bus) |b| isBusSpatial(self, b) else (options.position != null);
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
            const bus_attn = getBusAttenuation(self, b);
            min_dist = options.min_distance orelse bus_attn.min_distance;
            max_dist = options.max_distance orelse bus_attn.max_distance;
            rolloff = options.rolloff orelse bus_attn.rolloff;
            model = options.attenuation_model orelse bus_attn.model;
            dop_scale = options.doppler_factor orelse getBusDopplerFactor(self, b);
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

    const cmd = commands.Command{
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

    _ = pushCommand(self, cmd);
    if (!self.started) {
        drainCommands(self);
    }
}
