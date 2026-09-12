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

pub const AudioEngine = struct {
    pub const max_voices = 24;
    pub const max_distance = 30.0;
    pub const max_commands = 64;

    const Command = union(enum) {
        voice: struct {
            kind: VoiceKind,
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
            volume: f32,
            pan: f32,
            duration: f32,
            sample_step: f64,
            loop: bool,
            seed: u32,
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
    muted: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    listener_pos: Vec3 = Vec3.zero,
    listener_right: Vec3 = Vec3.new(1.0, 0.0, 0.0),
    started: bool = false,
    sample_rate: f32 = 44100.0,
    next_seed: u32 = 0x12345678,

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
            self.started = true;
        }
    }

    pub fn shutdown(self: *AudioEngine) void {
        if (!self.started) return;
        saudio.shutdown();
        self.started = false;
    }

    pub const PlayParams = struct {
        kind: VoiceKind = .thump,
        position: ?Vec3 = null, // null = non-positional
        volume: f32 = 0.5,
        duration: f32 = 0.2,
        freq: f32 = 110.0,
        freq_end: f32 = 55.0,
        cutoff: f32 = 5000.0,
        cutoff_end: f32 = 500.0,
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
                    .volume = c.volume,
                    .pan = c.pan,
                    .duration = c.duration,
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

    /// Shared distance attenuation + pan for `play` and `playClip`.
    fn spatialize(self: *const AudioEngine, position: ?Vec3, volume: f32) struct { vol: f32, pan: f32 } {
        return spatializeWith(self.listener_pos, self.listener_right, position, volume);
    }

    /// Lock-free attenuation/pan from a listener snapshot.
    fn spatializeWith(lpos: Vec3, lright: Vec3, position: ?Vec3, volume: f32) struct { vol: f32, pan: f32 } {
        var vol = volume;
        var pan: f32 = 0.0;
        if (position) |p| {
            const to = p.sub(lpos);
            const dist = to.length();
            vol *= @max(0.0, 1.0 - dist / max_distance);
            if (dist > 1e-4) {
                pan = std.math.clamp(to.scale(1.0 / dist).dot(lright), -1.0, 1.0);
            }
        }
        return .{ .vol = vol, .pan = pan };
    }

    pub fn play(self: *AudioEngine, params: PlayParams) void {
        const sp = spatializeWith(self.listener_pos, self.listener_right, params.position, params.volume);
        if (sp.vol <= 0.001) return;
        const duration = @max(params.duration, 0.01);

        self.next_seed +%= 1;
        const seed = self.next_seed *% 2654435761 +% 97;
        const cmd = Command{
            .voice = .{
                .kind = params.kind,
                .volume = sp.vol,
                .pan = sp.pan,
                .duration = duration,
                .freq = params.freq,
                .freq_end = params.freq_end,
                .cutoff = params.cutoff,
                .cutoff_end = params.cutoff_end,
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
        const s = std.math.clamp(speed, 0.0, 15.0);
        self.play(.{
            .kind = .thump,
            .position = position,
            .volume = 0.12 + 0.05 * s,
            .duration = 0.12 + 0.012 * s,
            .freq = 130.0 - 4.0 * s,
            .freq_end = 55.0,
        });
    }

    /// Explosion noise burst; size scales duration.
    pub fn playExplosion(self: *AudioEngine, position: Vec3, size: f32) void {
        self.play(.{
            .kind = .noise_burst,
            .position = position,
            .volume = 0.85,
            .duration = 0.5 + 0.3 * size,
            .cutoff = 6000.0,
            .cutoff_end = 300.0,
        });
    }

    /// Short non-positional blip (UI, jumps, snaps).
    pub fn playBlip(self: *AudioEngine, freq: f32) void {
        self.play(.{
            .kind = .blip,
            .volume = 0.25,
            .duration = 0.09,
            .freq = freq,
            .freq_end = freq,
        });
    }

    pub const ClipPlayOptions = struct {
        position: ?Vec3 = null, // null = non-positional
        volume: f32 = 1.0,
        loop: bool = false,
        rate: f32 = 1.0, // playback speed multiplier
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
        const sp = spatializeWith(self.listener_pos, self.listener_right, options.position, options.volume);
        if (sp.vol <= 0.001) return;
        const rate_f64: f64 = @floatCast(options.rate);
        const step: f64 = @as(f64, @floatCast(clip.sample_rate / engine_rate)) * rate_f64;
        if (!std.math.isFinite(step) or step <= 0.0) return;
        const clip_dur: f64 = @as(f64, @floatFromInt(clip.frames)) / @as(f64, @floatCast(clip.sample_rate)) / rate_f64;
        const duration = @max(@as(f32, @floatCast(clip_dur)), 0.01);

        self.next_seed +%= 1;
        const seed = self.next_seed *% 2654435761 +% 97;
        const cmd = Command{
            .clip = .{
                .clip = clip,
                .volume = sp.vol,
                .pan = sp.pan,
                .duration = duration,
                .sample_step = step,
                .loop = options.loop,
                .seed = seed,
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

    pub fn setMuted(self: *AudioEngine, muted: bool) void {
        self.muted.store(muted, .release);
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
        self.master_volume.store(@bitCast(vol), .release);
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

        // Per-callback setup: collect the active voices once with all
        // constant-per-buffer data precomputed (pan gains, envelope and
        // filter slopes, clip step/loop), so the per-sample inner loops do
        // no sqrt, no voice scan and no pan math. Sample voices that are
        // already invalid are deactivated here, not per sample.
        var list: [max_voices]ActiveVoice = undefined;
        var count: usize = 0;
        for (&self.voices) |*v| {
            if (!v.active) continue;
            const pan = std.math.clamp(v.pan, -1.0, 1.0);
            const g = v.volume * master;
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
                // Exact 1:1 step from an integer position needs no
                // interpolation, and the position stays integral all buffer.
                a.step_one = v.sample_step == 1.0 and pos == @floor(pos);
                list[count] = a;
            } else {
                list[count] = a;
            }
            count += 1;
        }
        if (count == 0) {
            for (buffer) |*x| x.* = 0.0;
            return;
        }

        // Voices outer, samples inner: the kind dispatch happens once per
        // voice per buffer and each inner loop only touches its own state.
        for (buffer) |*x| x.* = 0.0;
        for (list[0..count]) |a| {
            switch (a.kind) {
                .noise_burst => renderNoise(a, dt, buffer),
                .thump => renderTone(true, a, dt, buffer),
                .blip => renderTone(false, a, dt, buffer),
                .sample => renderSample(a, dt, buffer),
            }
        }
        // Soft clip into the output buffer.
        var f: usize = 0;
        while (f < buffer.len) : (f += 2) {
            buffer[f] = buffer[f] / (1.0 + @abs(buffer[f]));
            buffer[f + 1] = buffer[f + 1] / (1.0 + @abs(buffer[f + 1]));
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
                        v.active = false;
                        return;
                    }
                }
                const idx: usize = @intFromFloat(pos);
                buffer[j * 2] += s[idx * 2] * a.gl;
                buffer[j * 2 + 1] += s[idx * 2 + 1] * a.gr;
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
                        v.active = false;
                        return;
                    }
                }
                const fl = @floor(pos);
                const idx0: usize = @intFromFloat(fl);
                const frac: f32 = @floatCast(pos - fl);
                const x = @min(idx0, a.frames - 1);
                const y = @min(x + 1, a.frames - 1);
                const l = s[x * 2] + (s[y * 2] - s[x * 2]) * frac;
                const r = s[x * 2 + 1] + (s[y * 2 + 1] - s[x * 2 + 1]) * frac;
                buffer[j * 2] += l * a.gl;
                buffer[j * 2 + 1] += r * a.gr;
                pos += a.step;
            }
        }
        v.sample_pos = pos;
        v.t = t;
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
