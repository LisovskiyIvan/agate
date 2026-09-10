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
pub const AudioEngine = struct {
    pub const max_voices = 24;
    pub const max_distance = 30.0;

    const SpinLock = struct {
        locked: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

        fn acquire(self: *SpinLock) void {
            while (self.locked.cmpxchgStrong(false, true, .acquire, .monotonic) != null) {
                std.atomic.spinLoopHint();
            }
        }

        fn release(self: *SpinLock) void {
            self.locked.store(false, .release);
        }
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

    mutex: SpinLock = .{},
    voices: [max_voices]Voice = [_]Voice{Voice{}} ** max_voices,
    master_volume: f32 = 0.8,
    muted: bool = false,
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

    /// Picks a free voice, or steals the one closest to finishing.
    /// Callers must hold `mutex`.
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

    /// Lock-free attenuation/pan from a listener snapshot, so callers can
    /// compute playback parameters before acquiring the mixer lock.
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
        // Attenuation/pan read only the listener snapshot, so they are
        // computed before taking the lock; the critical section then only
        // claims the slot and writes it.
        const sp = spatializeWith(self.listener_pos, self.listener_right, params.position, params.volume);
        if (sp.vol <= 0.001) return;
        const duration = @max(params.duration, 0.01);

        self.mutex.acquire();
        defer self.mutex.release();

        const v = self.acquireSlot() orelse return;

        self.next_seed +%= 1;
        v.* = .{
            .active = true,
            .kind = params.kind,
            .volume = sp.vol,
            .pan = sp.pan,
            .duration = duration,
            .freq = params.freq,
            .freq_end = params.freq_end,
            .cutoff = params.cutoff,
            .cutoff_end = params.cutoff_end,
            .seed = self.next_seed *% 2654435761 +% 97,
        };
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
        // Capture playback parameters before taking the lock so the
        // critical section only claims the slot and writes it.
        const engine_rate = self.sample_rate;
        if (engine_rate <= 0.0) return;
        const sp = spatializeWith(self.listener_pos, self.listener_right, options.position, options.volume);
        if (sp.vol <= 0.001) return;
        const rate_f64: f64 = @floatCast(options.rate);
        const step: f64 = @as(f64, @floatCast(clip.sample_rate / engine_rate)) * rate_f64;
        if (!std.math.isFinite(step) or step <= 0.0) return;
        const clip_dur: f64 = @as(f64, @floatFromInt(clip.frames)) / @as(f64, @floatCast(clip.sample_rate)) / rate_f64;
        const duration = @max(@as(f32, @floatCast(clip_dur)), 0.01);

        self.mutex.acquire();
        defer self.mutex.release();

        const v = self.acquireSlot() orelse return;

        self.next_seed +%= 1;
        v.* = .{
            .active = true,
            .kind = .sample,
            .volume = sp.vol,
            .pan = sp.pan,
            .duration = duration,
            .clip = clip,
            .sample_pos = 0.0,
            .sample_step = step,
            .loop = options.loop,
            .seed = self.next_seed *% 2654435761 +% 97,
        };
    }

    pub fn updateListener(self: *AudioEngine, pos: Vec3, right: Vec3) void {
        // Normalize outside the lock; only the stores need the mutex.
        const r = if (right.length() > 1e-4) right.normalize() else Vec3.new(1.0, 0.0, 0.0);
        self.mutex.acquire();
        defer self.mutex.release();
        self.listener_pos = pos;
        self.listener_right = r;
    }

    pub fn setMuted(self: *AudioEngine, muted: bool) void {
        self.mutex.acquire();
        defer self.mutex.release();
        self.muted = muted;
    }

    pub fn toggleMuted(self: *AudioEngine) void {
        self.mutex.acquire();
        defer self.mutex.release();
        self.muted = !self.muted;
    }

    /// Mixes stereo-interleaved frames and advances voice state. Called by
    /// the stream callback and directly by tests.
    pub fn renderFrames(self: *AudioEngine, buffer: []f32) void {
        std.debug.assert(buffer.len % 2 == 0);
        const total: usize = buffer.len / 2;
        if (total == 0) return;
        const dt: f32 = 1.0 / self.sample_rate;
        const master: f32 = if (self.muted) 0.0 else self.master_volume;

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
        self.mutex.acquire();
        defer self.mutex.release();
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

/// Decoded WAV clip: interleaved stereo f32 at the clip's own sample rate.
/// Read-only after load, so the mixer thread can safely read it while a
/// voice plays it. The clip must outlive any voice using it.
pub const AudioClip = struct {
    samples: []f32 = &.{},
    sample_rate: f32 = 44100.0,
    frames: usize = 0,

    pub fn frameCount(self: *const AudioClip) usize {
        return self.frames;
    }

    pub fn deinit(self: *AudioClip, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
        self.* = .{};
    }

    pub fn fromWavFile(allocator: std.mem.Allocator, path: []const u8) !AudioClip {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const raw = try file.readToEndAlloc(allocator, 256 * 1024 * 1024);
        defer allocator.free(raw);
        return fromWavMemory(allocator, raw);
    }

    /// Decodes RIFF/WAVE PCM (8-bit unsigned, 16/24/32-bit signed int) and
    /// IEEE float32 (format tag 3), mono or stereo, any sample rate.
    /// Output is interleaved stereo f32 at the clip's own rate; playback
    /// resamples via `AudioEngine.playClip`. Returns error.InvalidWav for
    /// malformed data and error.UnsupportedWavFormat for valid WAVs we
    /// cannot play (channel count, bit depth, format tag).
    pub fn fromWavMemory(allocator: std.mem.Allocator, bytes: []const u8) !AudioClip {
        if (bytes.len < 12) return error.InvalidWav;
        if (!std.mem.eql(u8, bytes[0..4], "RIFF")) return error.InvalidWav;
        if (!std.mem.eql(u8, bytes[8..12], "WAVE")) return error.InvalidWav;

        const Fmt = struct {
            audio_format: u16,
            channels: u16,
            sample_rate: u32,
            bits: u16,
        };
        var fmt: ?Fmt = null;
        var data: ?[]const u8 = null;

        // Chunks may appear in any order; each is padded to an even size.
        var off: usize = 12;
        while (off + 8 <= bytes.len) {
            const id = bytes[off..][0..4];
            const size: usize = std.mem.readInt(u32, bytes[off + 4 ..][0..4], .little);
            const start = off + 8;
            if (size > bytes.len - start) return error.InvalidWav; // truncated chunk
            const chunk = bytes[start..][0..size];
            if (std.mem.eql(u8, id, "fmt ")) {
                if (chunk.len < 16) return error.InvalidWav;
                fmt = .{
                    .audio_format = std.mem.readInt(u16, chunk[0..][0..2], .little),
                    .channels = std.mem.readInt(u16, chunk[2..][0..2], .little),
                    .sample_rate = std.mem.readInt(u32, chunk[4..][0..4], .little),
                    .bits = std.mem.readInt(u16, chunk[14..][0..2], .little),
                };
            } else if (std.mem.eql(u8, id, "data")) {
                if (data == null) data = chunk;
            }
            off = start + size + (size & 1);
        }

        const f = fmt orelse return error.InvalidWav;
        const d = data orelse return error.InvalidWav;
        if (f.sample_rate == 0) return error.InvalidWav;
        if (f.channels != 1 and f.channels != 2) return error.UnsupportedWavFormat;
        const is_pcm = f.audio_format == 1 and (f.bits == 8 or f.bits == 16 or f.bits == 24 or f.bits == 32);
        const is_f32 = f.audio_format == 3 and f.bits == 32;
        if (!is_pcm and !is_f32) return error.UnsupportedWavFormat;

        const bytes_per_sample: usize = f.bits / 8;
        const block_align: usize = @as(usize, f.channels) * bytes_per_sample;
        if (d.len < block_align) return error.InvalidWav;
        const frames = d.len / block_align;
        if (frames == 0) return error.InvalidWav;

        const out = try allocator.alloc(f32, frames * 2);
        errdefer allocator.free(out);
        const channels: usize = f.channels;
        if (builtin.cpu.arch.endian() == .little) {
            // Fast paths for the common cases. WAV data is little-endian,
            // so on LE hosts values can be read directly, and float32
            // stereo already matches the output layout exactly. Anything
            // unaligned falls through to the scalar `readInt` loop below.
            if (is_f32 and channels == 2) {
                @memcpy(std.mem.sliceAsBytes(out), d[0 .. frames * 8]);
                return .{
                    .samples = out,
                    .sample_rate = @floatFromInt(f.sample_rate),
                    .frames = frames,
                };
            }
            if (is_f32 and channels == 1 and @intFromPtr(d.ptr) % 4 == 0) {
                // Mono float32: duplicate to stereo, vectorized loads.
                const raw: []align(4) const u8 = @alignCast(d[0 .. frames * 4]);
                const src: []align(4) const f32 = std.mem.bytesAsSlice(f32, raw);
                const V = 8;
                var i: usize = 0;
                while (i + V <= frames) : (i += V) {
                    const v: @Vector(V, f32) = src[i..][0..V].*;
                    inline for (0..V) |k| {
                        out[(i + k) * 2] = v[k];
                        out[(i + k) * 2 + 1] = v[k];
                    }
                }
                while (i < frames) : (i += 1) {
                    out[i * 2] = src[i];
                    out[i * 2 + 1] = src[i];
                }
                return .{
                    .samples = out,
                    .sample_rate = @floatFromInt(f.sample_rate),
                    .frames = frames,
                };
            }
            if (is_pcm and f.bits == 16 and @intFromPtr(d.ptr) % 2 == 0) {
                // 16-bit PCM: typed loads + vector int->float conversion.
                const raw: []align(2) const u8 = @alignCast(d[0 .. frames * block_align]);
                const src = std.mem.bytesAsSlice(i16, raw);
                const scale: f32 = 1.0 / 32768.0;
                const V = 8;
                if (channels == 2) {
                    const total = frames * 2;
                    var i: usize = 0;
                    while (i + V <= total) : (i += V) {
                        const vi: @Vector(V, i16) = src[i..][0..V].*;
                        const vf: @Vector(V, f32) = @floatFromInt(vi);
                        out[i..][0..V].* = vf * @as(@Vector(V, f32), @splat(scale));
                    }
                    while (i < total) : (i += 1) {
                        out[i] = @as(f32, @floatFromInt(src[i])) * scale;
                    }
                } else {
                    var i: usize = 0;
                    while (i + V <= frames) : (i += V) {
                        const vi: @Vector(V, i16) = src[i..][0..V].*;
                        const vf: @Vector(V, f32) = @floatFromInt(vi);
                        const vs = vf * @as(@Vector(V, f32), @splat(scale));
                        inline for (0..V) |k| {
                            out[(i + k) * 2] = vs[k];
                            out[(i + k) * 2 + 1] = vs[k];
                        }
                    }
                    while (i < frames) : (i += 1) {
                        const v = @as(f32, @floatFromInt(src[i])) * scale;
                        out[i * 2] = v;
                        out[i * 2 + 1] = v;
                    }
                }
                return .{
                    .samples = out,
                    .sample_rate = @floatFromInt(f.sample_rate),
                    .frames = frames,
                };
            }
        }
        // Scalar fallback: 8/24/32-bit int, unaligned or big-endian data.
        var i: usize = 0;
        while (i < frames) : (i += 1) {
            var c: usize = 0;
            var vals: [2]f32 = .{ 0.0, 0.0 };
            while (c < channels) : (c += 1) {
                const b = d[i * block_align + c * bytes_per_sample ..][0..bytes_per_sample];
                var v: f32 = 0.0;
                if (f.audio_format == 3) {
                    v = @bitCast(std.mem.readInt(u32, b[0..4], .little));
                } else switch (f.bits) {
                    8 => v = (@as(f32, @floatFromInt(b[0])) - 128.0) / 128.0,
                    16 => v = @as(f32, @floatFromInt(std.mem.readInt(i16, b[0..2], .little))) / 32768.0,
                    24 => {
                        const u: u32 = @as(u32, b[0]) | (@as(u32, b[1]) << 8) | (@as(u32, b[2]) << 16);
                        const s: i32 = @as(i32, @bitCast(u << 8)) >> 8;
                        v = @as(f32, @floatFromInt(s)) / 8388608.0;
                    },
                    else => v = @as(f32, @floatFromInt(std.mem.readInt(i32, b[0..4], .little))) / 2147483648.0,
                }
                vals[c] = v;
            }
            if (channels == 1) vals[1] = vals[0];
            out[i * 2] = vals[0];
            out[i * 2 + 1] = vals[1];
        }
        return .{
            .samples = out,
            .sample_rate = @floatFromInt(f.sample_rate),
            .frames = frames,
        };
    }
};

fn peakAbs(buf: []const f32, from_frame: usize, to_frame: usize) f32 {
    var peak: f32 = 0.0;
    var f = from_frame;
    while (f < to_frame) : (f += 1) {
        peak = @max(peak, @abs(buf[f * 2]));
        peak = @max(peak, @abs(buf[f * 2 + 1]));
    }
    return peak;
}

fn channelEnergy(buf: []const f32, channel: usize) f32 {
    var e: f32 = 0.0;
    var f = channel;
    while (f < buf.len) : (f += 2) {
        e += buf[f] * buf[f];
    }
    return e;
}

test "AudioEngine thump sounds then decays to silence" {
    var eng = AudioEngine{};
    eng.play(.{ .kind = .thump, .volume = 0.8, .duration = 0.2, .freq = 120.0, .freq_end = 60.0 });

    var buf: [16758]f32 = [_]f32{0.0} ** 16758; // 0.19 s stereo at 44100 Hz
    eng.renderFrames(&buf);

    const head = peakAbs(&buf, 0, 440);
    const tail = peakAbs(&buf, 7900, 8379);
    try std.testing.expect(head > 0.05);
    try std.testing.expect(tail < head * 0.5);

    // Past the duration every voice is done: pure silence.
    var rest: [4410]f32 = [_]f32{0.0} ** 4410; // +0.05 s -> t in [0.19, 0.24]
    eng.renderFrames(&rest);
    try std.testing.expect(peakAbs(&rest, 1500, 2205) < 1e-6);
}

test "AudioEngine voice pool caps polyphony by stealing" {
    var eng = AudioEngine{};
    var i: usize = 0;
    while (i < AudioEngine.max_voices + 6) : (i += 1) {
        eng.play(.{ .kind = .blip, .volume = 0.5, .duration = 1.0, .freq = 440.0, .freq_end = 440.0 });
    }
    var active: usize = 0;
    for (&eng.voices) |*v| {
        if (v.active) active += 1;
    }
    try std.testing.expectEqual(AudioEngine.max_voices, active);
}

test "AudioEngine positional pan follows the listener" {
    var eng = AudioEngine{};
    eng.updateListener(Vec3.zero, Vec3.new(1.0, 0.0, 0.0));

    eng.play(.{ .kind = .blip, .position = Vec3.new(10.0, 0.0, 0.0), .volume = 0.8, .duration = 0.05, .freq = 440.0, .freq_end = 440.0 });
    var right_buf: [4410]f32 = [_]f32{0.0} ** 4410;
    eng.renderFrames(&right_buf);
    try std.testing.expect(channelEnergy(&right_buf, 1) > channelEnergy(&right_buf, 0) * 2.0);

    eng.play(.{ .kind = .blip, .position = Vec3.new(-10.0, 0.0, 0.0), .volume = 0.8, .duration = 0.05, .freq = 440.0, .freq_end = 440.0 });
    var left_buf: [4410]f32 = [_]f32{0.0} ** 4410;
    eng.renderFrames(&left_buf);
    try std.testing.expect(channelEnergy(&left_buf, 0) > channelEnergy(&left_buf, 1) * 2.0);

    // Beyond max distance: silence.
    eng.play(.{ .kind = .blip, .position = Vec3.new(100.0, 0.0, 0.0), .volume = 0.8, .duration = 0.05, .freq = 440.0, .freq_end = 440.0 });
    var far_buf: [4410]f32 = [_]f32{0.0} ** 4410;
    eng.renderFrames(&far_buf);
    try std.testing.expect(peakAbs(&far_buf, 0, 2205) < 1e-6);
}

// --- WAV clip test helpers ---

fn writeFmtChunk(buf: []u8, off: usize, audio_format: u16, channels: u16, sample_rate: u32, bits: u16) usize {
    @memcpy(buf[off..][0..4], "fmt ");
    std.mem.writeInt(u32, buf[off + 4 ..][0..4], 16, .little);
    std.mem.writeInt(u16, buf[off + 8 ..][0..2], audio_format, .little);
    std.mem.writeInt(u16, buf[off + 10 ..][0..2], channels, .little);
    std.mem.writeInt(u32, buf[off + 12 ..][0..4], sample_rate, .little);
    const block_align: u32 = @as(u32, channels) * @as(u32, bits) / 8;
    std.mem.writeInt(u32, buf[off + 16 ..][0..4], sample_rate * block_align, .little);
    std.mem.writeInt(u16, buf[off + 20 ..][0..2], @intCast(block_align), .little);
    std.mem.writeInt(u16, buf[off + 22 ..][0..2], bits, .little);
    return off + 24;
}

fn writeDataChunk(buf: []u8, off: usize, raw_data: []const u8) usize {
    @memcpy(buf[off..][0..4], "data");
    std.mem.writeInt(u32, buf[off + 4 ..][0..4], @intCast(raw_data.len), .little);
    @memcpy(buf[off + 8 ..][0..raw_data.len], raw_data);
    var end = off + 8 + raw_data.len;
    if (raw_data.len & 1 == 1) {
        buf[end] = 0;
        end += 1;
    }
    return end;
}

fn buildWav(
    allocator: std.mem.Allocator,
    fmt_first: bool,
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
    raw_data: []const u8,
) ![]u8 {
    const pad: usize = raw_data.len & 1;
    const total = 12 + (8 + 16) + (8 + raw_data.len + pad);
    const buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);
    @memcpy(buf[0..4], "RIFF");
    std.mem.writeInt(u32, buf[4..8], @intCast(total - 8), .little);
    @memcpy(buf[8..12], "WAVE");
    var off: usize = 12;
    if (fmt_first) {
        off = writeFmtChunk(buf, off, audio_format, channels, sample_rate, bits);
        off = writeDataChunk(buf, off, raw_data);
    } else {
        off = writeDataChunk(buf, off, raw_data);
        off = writeFmtChunk(buf, off, audio_format, channels, sample_rate, bits);
    }
    std.debug.assert(off == total);
    return buf;
}

fn encodeI16(allocator: std.mem.Allocator, pcm: []const i16) ![]u8 {
    const raw = try allocator.alloc(u8, pcm.len * 2);
    errdefer allocator.free(raw);
    for (pcm, 0..) |s, i| std.mem.writeInt(i16, raw[i * 2 ..][0..2], s, .little);
    return raw;
}

fn sineI16(allocator: std.mem.Allocator, frames: usize, sample_rate: u32, freq: f32, amp: f32) ![]i16 {
    const pcm = try allocator.alloc(i16, frames);
    errdefer allocator.free(pcm);
    for (pcm, 0..) |*s, i| {
        const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(sample_rate));
        s.* = @intFromFloat(@sin(t * freq * 2.0 * std.math.pi) * amp);
    }
    return pcm;
}

test "AudioClip decodes 16-bit mono WAV" {
    const alloc = std.testing.allocator;
    const pcm = try sineI16(alloc, 4410, 44100, 440.0, 20000.0);
    defer alloc.free(pcm);
    const raw = try encodeI16(alloc, pcm);
    defer alloc.free(raw);
    const wav = try buildWav(alloc, true, 1, 1, 44100, 16, raw);
    defer alloc.free(wav);

    var clip = try AudioClip.fromWavMemory(alloc, wav);
    defer clip.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4410), clip.frames);
    try std.testing.expectEqual(@as(usize, 4410), clip.frameCount());
    try std.testing.expectEqual(@as(f32, 44100.0), clip.sample_rate);
    for (pcm, 0..) |s, i| {
        const want = @as(f32, @floatFromInt(s)) / 32768.0;
        try std.testing.expectApproxEqAbs(want, clip.samples[i * 2], 1e-4);
        try std.testing.expectApproxEqAbs(want, clip.samples[i * 2 + 1], 1e-4); // mono duplicated
    }
}

test "AudioClip decodes 16-bit stereo WAV" {
    const alloc = std.testing.allocator;
    const left = try sineI16(alloc, 1000, 44100, 440.0, 20000.0);
    defer alloc.free(left);
    const right = try sineI16(alloc, 1000, 44100, 880.0, 10000.0);
    defer alloc.free(right);
    const raw = try alloc.alloc(u8, 1000 * 4);
    defer alloc.free(raw);
    for (0..1000) |i| {
        std.mem.writeInt(i16, raw[i * 4 ..][0..2], left[i], .little);
        std.mem.writeInt(i16, raw[i * 4 + 2 ..][0..2], right[i], .little);
    }
    const wav = try buildWav(alloc, true, 1, 2, 44100, 16, raw);
    defer alloc.free(wav);

    var clip = try AudioClip.fromWavMemory(alloc, wav);
    defer clip.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1000), clip.frames);
    for (0..1000) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(left[i])) / 32768.0, clip.samples[i * 2], 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(right[i])) / 32768.0, clip.samples[i * 2 + 1], 1e-4);
    }
}

test "AudioClip decodes 24-bit WAV" {
    const alloc = std.testing.allocator;
    // Stereo: left = max positive, right = min negative, then silence.
    const raw = try alloc.alloc(u8, 2 * 6);
    defer alloc.free(raw);
    raw[0] = 0xFF;
    raw[1] = 0xFF;
    raw[2] = 0x7F; // 0x7FFFFF
    raw[3] = 0x00;
    raw[4] = 0x00;
    raw[5] = 0x80; // 0x800000
    @memset(raw[6..], 0);
    const wav = try buildWav(alloc, true, 1, 2, 48000, 24, raw);
    defer alloc.free(wav);

    var clip = try AudioClip.fromWavMemory(alloc, wav);
    defer clip.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 2), clip.frames);
    try std.testing.expectEqual(@as(f32, 48000.0), clip.sample_rate);
    try std.testing.expectApproxEqAbs(@as(f32, 8388607.0) / 8388608.0, clip.samples[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -8388608.0) / 8388608.0, clip.samples[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), clip.samples[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), clip.samples[3], 1e-6);
}

test "AudioClip decodes float32 WAV" {
    const alloc = std.testing.allocator;
    const vals = [_]f32{ 0.5, -0.25, 1.0, -1.0 };
    const raw = try alloc.alloc(u8, vals.len * 4);
    defer alloc.free(raw);
    for (vals, 0..) |v, i| std.mem.writeInt(u32, raw[i * 4 ..][0..4], @bitCast(v), .little);
    const wav = try buildWav(alloc, true, 3, 2, 44100, 32, raw);
    defer alloc.free(wav);

    var clip = try AudioClip.fromWavMemory(alloc, wav);
    defer clip.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 2), clip.frames);
    for (vals, 0..) |v, i| try std.testing.expectApproxEqAbs(v, clip.samples[i], 1e-6);
}

test "AudioClip handles unordered chunks, pad byte, and 8-bit samples" {
    const alloc = std.testing.allocator;
    // 8-bit unsigned mono, odd length forces a pad byte before the fmt chunk.
    const raw = [_]u8{ 0, 64, 128, 192, 255 };
    const wav = try buildWav(alloc, false, 1, 1, 22050, 8, &raw);
    defer alloc.free(wav);

    var clip = try AudioClip.fromWavMemory(alloc, wav);
    defer clip.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 5), clip.frames);
    try std.testing.expectEqual(@as(f32, 22050.0), clip.sample_rate);
    for (raw, 0..) |b, i| {
        const want = (@as(f32, @floatFromInt(b)) - 128.0) / 128.0;
        try std.testing.expectApproxEqAbs(want, clip.samples[i * 2], 1e-6);
        try std.testing.expectApproxEqAbs(want, clip.samples[i * 2 + 1], 1e-6);
    }
}

test "AudioEngine playClip renders audible audio and loops past the end" {
    const alloc = std.testing.allocator;
    const pcm = try sineI16(alloc, 4410, 44100, 440.0, 20000.0); // 0.1 s
    defer alloc.free(pcm);
    const raw = try encodeI16(alloc, pcm);
    defer alloc.free(raw);
    const wav = try buildWav(alloc, true, 1, 1, 44100, 16, raw);
    defer alloc.free(wav);
    var clip = try AudioClip.fromWavMemory(alloc, wav);
    defer clip.deinit(alloc);

    // One-shot: audible head, silence after the clip ends.
    {
        var eng = AudioEngine{};
        eng.playClip(&clip, .{});
        var buf: [13230]f32 = [_]f32{0.0} ** 13230; // 0.15 s, clip is 0.1 s
        eng.renderFrames(&buf);
        try std.testing.expect(channelEnergy(&buf, 0) + channelEnergy(&buf, 1) > 10.0);
        try std.testing.expect(peakAbs(&buf, 5513, 6615) < 1e-6);
    }

    // Looping: still producing energy well past the clip duration.
    {
        var eng = AudioEngine{};
        eng.playClip(&clip, .{ .loop = true });
        var buf: [26460]f32 = [_]f32{0.0} ** 26460; // 0.3 s = 3x clip length
        eng.renderFrames(&buf);
        var tail: f32 = 0.0;
        var f: usize = 11025;
        while (f < 13230) : (f += 1) {
            tail += buf[f * 2] * buf[f * 2] + buf[f * 2 + 1] * buf[f * 2 + 1];
        }
        try std.testing.expect(tail > 10.0);
    }
}

test "AudioEngine playClip resamples clips with other sample rates" {
    const alloc = std.testing.allocator;
    const pcm = try sineI16(alloc, 2205, 22050, 440.0, 20000.0); // 0.1 s at 22050 Hz
    defer alloc.free(pcm);
    const raw = try encodeI16(alloc, pcm);
    defer alloc.free(raw);
    const wav = try buildWav(alloc, true, 1, 1, 22050, 16, raw);
    defer alloc.free(wav);
    var clip = try AudioClip.fromWavMemory(alloc, wav);
    defer clip.deinit(alloc);

    var eng = AudioEngine{}; // 44100 Hz engine plays the 22050 Hz clip upsampled
    eng.playClip(&clip, .{});
    var buf: [4410]f32 = [_]f32{0.0} ** 4410; // 0.05 s
    eng.renderFrames(&buf);
    try std.testing.expect(channelEnergy(&buf, 0) + channelEnergy(&buf, 1) > 5.0);
}

test "AudioEngine playClip step-one fast path matches interpolated path" {
    const alloc = std.testing.allocator;
    // Constant mono float32 clip: every output frame must equal the same
    // soft-clipped constant, whether the 1:1 fast path or the resampling
    // (linear interpolation) path renders it.
    const vals = [_]f32{0.5} ** 8;
    const raw = try alloc.alloc(u8, vals.len * 4);
    defer alloc.free(raw);
    for (vals, 0..) |v, i| std.mem.writeInt(u32, raw[i * 4 ..][0..4], @bitCast(v), .little);
    const wav = try buildWav(alloc, true, 3, 1, 44100, 32, raw);
    defer alloc.free(wav);
    var clip = try AudioClip.fromWavMemory(alloc, wav);
    defer clip.deinit(alloc);

    const g: f32 = 1.0 * 0.8 * @sqrt(0.5); // volume * master * center pan gain
    const want: f32 = (0.5 * g) / (1.0 + @abs(0.5 * g));

    // step == 1.0 fast path (same-rate clip, rate 1).
    {
        var eng = AudioEngine{};
        eng.playClip(&clip, .{ .loop = true });
        var buf: [4410]f32 = [_]f32{0.0} ** 4410;
        eng.renderFrames(&buf);
        for (0..2205) |j| {
            try std.testing.expectApproxEqAbs(want, buf[j * 2], 1e-6);
            try std.testing.expectApproxEqAbs(want, buf[j * 2 + 1], 1e-6);
        }
    }

    // Resampling path (rate 0.5 -> step 0.5, linear interpolation).
    {
        var eng = AudioEngine{};
        eng.playClip(&clip, .{ .loop = true, .rate = 0.5 });
        var buf: [4410]f32 = [_]f32{0.0} ** 4410;
        eng.renderFrames(&buf);
        for (0..2205) |j| {
            try std.testing.expectApproxEqAbs(want, buf[j * 2], 1e-6);
            try std.testing.expectApproxEqAbs(want, buf[j * 2 + 1], 1e-6);
        }
    }
}

test "AudioEngine playClip far position and zero volume yield silence" {
    const alloc = std.testing.allocator;
    const pcm = try sineI16(alloc, 4410, 44100, 440.0, 20000.0);
    defer alloc.free(pcm);
    const raw = try encodeI16(alloc, pcm);
    defer alloc.free(raw);
    const wav = try buildWav(alloc, true, 1, 1, 44100, 16, raw);
    defer alloc.free(wav);
    var clip = try AudioClip.fromWavMemory(alloc, wav);
    defer clip.deinit(alloc);

    var eng = AudioEngine{};
    eng.playClip(&clip, .{ .position = Vec3.new(100.0, 0.0, 0.0), .volume = 1.0 });
    var far_buf: [4410]f32 = [_]f32{0.0} ** 4410;
    eng.renderFrames(&far_buf);
    try std.testing.expect(peakAbs(&far_buf, 0, 2205) < 1e-6);

    eng.playClip(&clip, .{ .volume = 0.0 });
    var quiet_buf: [4410]f32 = [_]f32{0.0} ** 4410;
    eng.renderFrames(&quiet_buf);
    try std.testing.expect(peakAbs(&quiet_buf, 0, 2205) < 1e-6);
}

test "AudioClip rejects invalid and unsupported WAV data" {
    const alloc = std.testing.allocator;
    // Bad magic.
    try std.testing.expectError(error.InvalidWav, AudioClip.fromWavMemory(alloc, "NOTRxxxxWAVE"));
    // Too short for a header.
    try std.testing.expectError(error.InvalidWav, AudioClip.fromWavMemory(alloc, "RIFF"));
    // Truncated chunk.
    const pcm = try sineI16(alloc, 100, 44100, 440.0, 10000.0);
    defer alloc.free(pcm);
    const raw = try encodeI16(alloc, pcm);
    defer alloc.free(raw);
    const wav = try buildWav(alloc, true, 1, 1, 44100, 16, raw);
    defer alloc.free(wav);
    try std.testing.expectError(error.InvalidWav, AudioClip.fromWavMemory(alloc, wav[0 .. wav.len / 2]));
    // Header only, no data chunk.
    try std.testing.expectError(error.InvalidWav, AudioClip.fromWavMemory(alloc, wav[0..36]));
    // Unsupported channel count.
    const raw3 = try alloc.alloc(u8, 12);
    defer alloc.free(raw3);
    @memset(raw3, 0);
    const wav3 = try buildWav(alloc, true, 1, 3, 44100, 16, raw3);
    defer alloc.free(wav3);
    try std.testing.expectError(error.UnsupportedWavFormat, AudioClip.fromWavMemory(alloc, wav3));
    // Unsupported format tag.
    const wav7 = try buildWav(alloc, true, 7, 1, 44100, 8, raw3[0..8]);
    defer alloc.free(wav7);
    try std.testing.expectError(error.UnsupportedWavFormat, AudioClip.fromWavMemory(alloc, wav7));
}
