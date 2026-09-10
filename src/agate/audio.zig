const std = @import("std");
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

    pub fn play(self: *AudioEngine, params: PlayParams) void {
        self.mutex.acquire();
        defer self.mutex.release();

        var slot: ?*Voice = null;
        for (&self.voices) |*v| {
            if (!v.active) {
                slot = v;
                break;
            }
        }
        if (slot == null) {
            // Steal the voice closest to finishing.
            var best_k: f32 = -1.0;
            for (&self.voices) |*v| {
                const k = v.t / @max(v.duration, 1e-6);
                if (k > best_k) {
                    best_k = k;
                    slot = v;
                }
            }
        }
        const v = slot orelse return;

        var vol = params.volume;
        var pan: f32 = 0.0;
        if (params.position) |p| {
            const to = p.sub(self.listener_pos);
            const dist = to.length();
            vol *= @max(0.0, 1.0 - dist / max_distance);
            if (dist > 1e-4) {
                pan = std.math.clamp(to.scale(1.0 / dist).dot(self.listener_right), -1.0, 1.0);
            }
        }
        if (vol <= 0.001) return;

        self.next_seed +%= 1;
        v.* = .{
            .active = true,
            .kind = params.kind,
            .volume = vol,
            .pan = pan,
            .duration = @max(params.duration, 0.01),
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

    pub fn updateListener(self: *AudioEngine, pos: Vec3, right: Vec3) void {
        self.mutex.acquire();
        defer self.mutex.release();
        self.listener_pos = pos;
        self.listener_right = if (right.length() > 1e-4) right.normalize() else Vec3.new(1.0, 0.0, 0.0);
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
        const dt = 1.0 / self.sample_rate;
        const master = if (self.muted) 0.0 else self.master_volume;
        var f: usize = 0;
        while (f < buffer.len) : (f += 2) {
            var l: f32 = 0.0;
            var r: f32 = 0.0;
            for (&self.voices) |*v| {
                if (!v.active) continue;
                v.t += dt;
                if (v.t >= v.duration) {
                    v.active = false;
                    continue;
                }
                const s = renderVoice(v, dt);
                const g = s * master;
                l += g * @sqrt(0.5 * (1.0 - v.pan));
                r += g * @sqrt(0.5 * (1.0 + v.pan));
            }
            // Soft clip into the output buffer.
            buffer[f] = l / (1.0 + @abs(l));
            buffer[f + 1] = r / (1.0 + @abs(r));
        }
    }

    fn renderVoice(v: *Voice, dt: f32) f32 {
        const k = v.t / v.duration;
        const env = (1.0 - k) * (1.0 - k);
        switch (v.kind) {
            .noise_burst => {
                v.seed = v.seed *% 1664525 +% 1013904223;
                const u: f32 = @floatFromInt(v.seed >> 9); // 0..2^23-1
                const n = u * (2.0 / 8388608.0) - 1.0;
                const cutoff = @max(v.cutoff + (v.cutoff_end - v.cutoff) * k, 30.0);
                const alpha = 1.0 - @exp(-2.0 * std.math.pi * cutoff * dt);
                v.lp += alpha * (n - v.lp);
                return v.lp * 1.6 * env * v.volume;
            },
            .thump, .blip => {
                const freq = @max(v.freq + (v.freq_end - v.freq) * k, 20.0);
                v.phase += freq * dt;
                const s = @sin(v.phase * 2.0 * std.math.pi);
                const click: f32 = if (v.kind == .thump and v.t < 0.008) 0.5 * (1.0 - v.t / 0.008) else 0.0;
                return (s * 0.8 + click) * env * v.volume;
            },
        }
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
