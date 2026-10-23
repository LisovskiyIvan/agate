const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const audio = @import("../audio.zig");
const streams_mod = @import("streams.zig");
const AudioEngine = audio.AudioEngine;
const AudioClip = audio.AudioClip;
const VoiceKind = audio.VoiceKind;
const Voice = audio.Voice;
const BusId = audio.BusId;
const AudioBus = audio.AudioBus;
const BiquadFilterType = audio.BiquadFilterType;
const BusFilterConfig = audio.BusFilterConfig;
const BusReverbConfig = audio.BusReverbConfig;
const Mesh = @import("../mesh.zig").Mesh;
const physics = @import("../physics.zig");
const PhysicsWorld = physics.PhysicsWorld;

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

    var buf: [16758]f32 = @as([16758]f32, @splat(0.0)); // 0.19 s stereo at 44100 Hz
    eng.renderFrames(&buf);

    const head = peakAbs(&buf, 0, 440);
    const tail = peakAbs(&buf, 7900, 8379);
    try std.testing.expect(head > 0.05);
    try std.testing.expect(tail < head * 0.5);

    // Past the duration every voice is done: pure silence.
    var rest: [4410]f32 = @as([4410]f32, @splat(0.0)); // +0.05 s -> t in [0.19, 0.24]
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
        if (v.isActive()) active += 1;
    }
    try std.testing.expectEqual(AudioEngine.max_voices, active);
}

test "AudioEngine positional pan follows the listener" {
    var eng = AudioEngine{};
    eng.updateListener(Vec3.zero, Vec3.new(1.0, 0.0, 0.0));

    eng.play(.{ .kind = .blip, .position = Vec3.new(10.0, 0.0, 0.0), .volume = 0.8, .duration = 0.05, .freq = 440.0, .freq_end = 440.0 });
    var right_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&right_buf);
    try std.testing.expect(channelEnergy(&right_buf, 1) > channelEnergy(&right_buf, 0) * 2.0);

    eng.play(.{ .kind = .blip, .position = Vec3.new(-10.0, 0.0, 0.0), .volume = 0.8, .duration = 0.05, .freq = 440.0, .freq_end = 440.0 });
    var left_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&left_buf);
    try std.testing.expect(channelEnergy(&left_buf, 0) > channelEnergy(&left_buf, 1) * 2.0);

    // Beyond max distance: silence.
    eng.play(.{ .kind = .blip, .position = Vec3.new(100.0, 0.0, 0.0), .volume = 0.8, .duration = 0.05, .freq = 440.0, .freq_end = 440.0 });
    var far_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
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

// Bundled 16-bit mono sine fixture (pcm -> raw bytes -> WAV container):
// the mono playClip/streaming/rejection tests all build the same sine
// through the same three steps, so the triple alloc/defer collapses to
// one builder + one defer. Stereo/interleaved constructions stay inline.
const MonoSineWav = struct {
    pcm: []i16,
    raw: []u8,
    wav: []u8,
    fn deinit(self: MonoSineWav, allocator: std.mem.Allocator) void {
        allocator.free(self.wav);
        allocator.free(self.raw);
        allocator.free(self.pcm);
    }
};

fn monoSineWav(allocator: std.mem.Allocator, frames: usize, sample_rate: u32, freq: f32, amp: f32) !MonoSineWav {
    const pcm = try sineI16(allocator, frames, sample_rate, freq, amp);
    errdefer allocator.free(pcm);
    const raw = try encodeI16(allocator, pcm);
    errdefer allocator.free(raw);
    const wav = try buildWav(allocator, true, 1, 1, sample_rate, 16, raw);
    errdefer allocator.free(wav);
    return .{ .pcm = pcm, .raw = raw, .wav = wav };
}

test "AudioClip decodes 16-bit mono WAV" {
    const alloc = std.testing.allocator;
    const built = try monoSineWav(alloc, 4410, 44100, 440.0, 20000.0);
    defer built.deinit(alloc);
    const pcm = built.pcm;
    const wav = built.wav;

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
    const built = try monoSineWav(alloc, 4410, 44100, 440.0, 20000.0); // 0.1 s
    defer built.deinit(alloc);
    var clip = try AudioClip.fromWavMemory(alloc, built.wav);
    defer clip.deinit(alloc);

    // One-shot: audible head, silence after the clip ends.
    {
        var eng = AudioEngine{};
        eng.playClip(&clip, .{});
        var buf: [13230]f32 = @as([13230]f32, @splat(0.0)); // 0.15 s, clip is 0.1 s
        eng.renderFrames(&buf);
        try std.testing.expect(channelEnergy(&buf, 0) + channelEnergy(&buf, 1) > 10.0);
        try std.testing.expect(peakAbs(&buf, 5513, 6615) < 1e-6);
    }

    // Looping: still producing energy well past the clip duration.
    {
        var eng = AudioEngine{};
        eng.playClip(&clip, .{ .loop = true });
        var buf: [26460]f32 = @as([26460]f32, @splat(0.0)); // 0.3 s = 3x clip length
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
    const built = try monoSineWav(alloc, 2205, 22050, 440.0, 20000.0); // 0.1 s at 22050 Hz
    defer built.deinit(alloc);
    var clip = try AudioClip.fromWavMemory(alloc, built.wav);
    defer clip.deinit(alloc);

    var eng = AudioEngine{}; // 44100 Hz engine plays the 22050 Hz clip upsampled
    eng.playClip(&clip, .{});
    var buf: [4410]f32 = @as([4410]f32, @splat(0.0)); // 0.05 s
    eng.renderFrames(&buf);
    try std.testing.expect(channelEnergy(&buf, 0) + channelEnergy(&buf, 1) > 5.0);
}

test "AudioEngine playClip step-one fast path matches interpolated path" {
    const alloc = std.testing.allocator;
    // Constant mono float32 clip: every output frame must equal the same
    // soft-clipped constant, whether the 1:1 fast path or the resampling
    // (linear interpolation) path renders it.
    const vals = @as([8]f32, @splat(0.5));
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
        var buf: [4410]f32 = @as([4410]f32, @splat(0.0));
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
        var buf: [4410]f32 = @as([4410]f32, @splat(0.0));
        eng.renderFrames(&buf);
        for (0..2205) |j| {
            try std.testing.expectApproxEqAbs(want, buf[j * 2], 1e-6);
            try std.testing.expectApproxEqAbs(want, buf[j * 2 + 1], 1e-6);
        }
    }
}

test "AudioEngine playClip far position and zero volume yield silence" {
    const alloc = std.testing.allocator;
    const built = try monoSineWav(alloc, 4410, 44100, 440.0, 20000.0);
    defer built.deinit(alloc);
    var clip = try AudioClip.fromWavMemory(alloc, built.wav);
    defer clip.deinit(alloc);

    var eng = AudioEngine{};
    eng.playClip(&clip, .{ .position = Vec3.new(100.0, 0.0, 0.0), .volume = 1.0 });
    var far_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&far_buf);
    try std.testing.expect(peakAbs(&far_buf, 0, 2205) < 1e-6);

    eng.playClip(&clip, .{ .volume = 0.0 });
    var quiet_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
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
    const built = try monoSineWav(alloc, 100, 44100, 440.0, 10000.0);
    defer built.deinit(alloc);
    const wav = built.wav;
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

fn encodeI24(allocator: std.mem.Allocator, samples: []const i32) ![]u8 {
    const raw = try allocator.alloc(u8, samples.len * 3);
    errdefer allocator.free(raw);
    for (samples, 0..) |s, i| {
        const u: u32 = @bitCast(s);
        raw[i * 3] = @truncate(u);
        raw[i * 3 + 1] = @truncate(u >> 8);
        raw[i * 3 + 2] = @truncate(u >> 16);
    }
    return raw;
}

fn encodeI32(allocator: std.mem.Allocator, pcm: []const i32) ![]u8 {
    const raw = try allocator.alloc(u8, pcm.len * 4);
    errdefer allocator.free(raw);
    for (pcm, 0..) |s, i| std.mem.writeInt(i32, raw[i * 4 ..][0..4], s, .little);
    return raw;
}

test "AudioClip decodes 8-bit stereo WAV via fast path" {
    const alloc = std.testing.allocator;
    const frames = 260; // crosses the 16-sample vector width with a tail
    const raw = try alloc.alloc(u8, frames * 2);
    defer alloc.free(raw);
    for (0..frames) |i| {
        raw[i * 2] = @intCast((i * 7 + 3) & 0xFF); // left ramp over the full range
        raw[i * 2 + 1] = @intCast(255 - ((i * 13 + 5) & 0xFF)); // right, different pattern
    }
    // Extreme values on the first frames: silence bounds and center.
    raw[0] = 0;
    raw[1] = 255;
    raw[2] = 128;
    raw[3] = 128;
    raw[4] = 1;
    raw[5] = 254;
    raw[6] = 127;
    raw[7] = 129;
    const wav = try buildWav(alloc, true, 1, 2, 44100, 8, raw);
    defer alloc.free(wav);

    var clip = try AudioClip.fromWavMemory(alloc, wav);
    defer clip.deinit(alloc);
    try std.testing.expectEqual(@as(usize, frames), clip.frames);
    for (0..frames) |i| {
        const want_l = (@as(f32, @floatFromInt(raw[i * 2])) - 128.0) / 128.0;
        const want_r = (@as(f32, @floatFromInt(raw[i * 2 + 1])) - 128.0) / 128.0;
        try std.testing.expectApproxEqAbs(want_l, clip.samples[i * 2], 1e-6);
        try std.testing.expectApproxEqAbs(want_r, clip.samples[i * 2 + 1], 1e-6);
    }
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), clip.samples[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 127.0) / 128.0, clip.samples[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), clip.samples[2], 1e-6);
}

test "AudioClip decodes 24-bit mono and stereo WAV via chunked path" {
    const alloc = std.testing.allocator;
    // Mono: 19 frames = two 8-frame groups + 3-sample tail. Edges first,
    // then a ramp crossing zero to check sign extension mid-buffer.
    const mono_vals = [_]i32{
        0x7FFFFF, -0x800000, 0,        -1,        1,
        -9000,    -7000,     -5000,    -3000,     -1000,
        1000,     3000,      5000,     7000,      9000,
        0x123456, -0x123456, 0x000001, -0x000001,
    };
    const mono_raw = try encodeI24(alloc, &mono_vals);
    defer alloc.free(mono_raw);
    const mono_wav = try buildWav(alloc, true, 1, 1, 48000, 24, mono_raw);
    defer alloc.free(mono_wav);

    var mono = try AudioClip.fromWavMemory(alloc, mono_wav);
    defer mono.deinit(alloc);
    try std.testing.expectEqual(@as(usize, mono_vals.len), mono.frames);
    for (mono_vals, 0..) |s, i| {
        const want = @as(f32, @floatFromInt(s)) / 8388608.0;
        try std.testing.expectApproxEqAbs(want, mono.samples[i * 2], 1e-6);
        try std.testing.expectApproxEqAbs(want, mono.samples[i * 2 + 1], 1e-6); // mono duplicated
    }
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), mono.samples[1 * 2], 1e-6); // 0x800000

    // Stereo: distinct channels with negative values on both sides.
    const stereo_vals = [_]i32{
        0x7FFFFF,  -0x800000,
        -0x400000, 0x400000,
        0,         -1,
        123456,    -654321,
        -0x000100, 0x000100,
        0x555555,  -0x555555,
        -42,       42,
        0x7FFFFE,  -0x7FFFFF,
        999,       -999,
        -0x123456, 0x123456,
    };
    const stereo_raw = try encodeI24(alloc, &stereo_vals);
    defer alloc.free(stereo_raw);
    const stereo_wav = try buildWav(alloc, true, 1, 2, 44100, 24, stereo_raw);
    defer alloc.free(stereo_wav);

    var stereo = try AudioClip.fromWavMemory(alloc, stereo_wav);
    defer stereo.deinit(alloc);
    try std.testing.expectEqual(@as(usize, stereo_vals.len / 2), stereo.frames);
    for (0..stereo.frames) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(stereo_vals[i * 2])) / 8388608.0, stereo.samples[i * 2], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(stereo_vals[i * 2 + 1])) / 8388608.0, stereo.samples[i * 2 + 1], 1e-6);
    }
}

test "AudioClip decodes 32-bit mono and stereo WAV" {
    const alloc = std.testing.allocator;
    const min = std.math.minInt(i32);
    const max = std.math.maxInt(i32);
    // Mono: 9 frames = one 8-sample group + 1-sample tail.
    const mono_vals = [_]i32{ min, max, 0, -1, 1, -123456789, 123456789, -1 << 30, 1 << 30 };
    const mono_raw = try encodeI32(alloc, &mono_vals);
    defer alloc.free(mono_raw);
    const mono_wav = try buildWav(alloc, true, 1, 1, 44100, 32, mono_raw);
    defer alloc.free(mono_wav);

    var mono = try AudioClip.fromWavMemory(alloc, mono_wav);
    defer mono.deinit(alloc);
    try std.testing.expectEqual(@as(usize, mono_vals.len), mono.frames);
    for (mono_vals, 0..) |s, i| {
        const want = @as(f32, @floatFromInt(s)) / 2147483648.0;
        try std.testing.expectApproxEqAbs(want, mono.samples[i * 2], 1e-6);
        try std.testing.expectApproxEqAbs(want, mono.samples[i * 2 + 1], 1e-6);
    }
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), mono.samples[0], 1e-6); // INT32_MIN

    // Stereo: ramp over several vector groups plus extremes.
    const frames = 25;
    const stereo_vals = try alloc.alloc(i32, frames * 2);
    defer alloc.free(stereo_vals);
    stereo_vals[0] = min;
    stereo_vals[1] = max;
    for (2..stereo_vals.len) |i| {
        const k: i32 = @intCast(i);
        stereo_vals[i] = if (i & 1 == 0) k * 1234567 - 9999999 else -(k * 7654321 - 8888888);
    }
    const stereo_raw = try encodeI32(alloc, stereo_vals);
    defer alloc.free(stereo_raw);
    const stereo_wav = try buildWav(alloc, true, 1, 2, 48000, 32, stereo_raw);
    defer alloc.free(stereo_wav);

    var stereo = try AudioClip.fromWavMemory(alloc, stereo_wav);
    defer stereo.deinit(alloc);
    try std.testing.expectEqual(@as(usize, frames), stereo.frames);
    for (stereo_vals, 0..) |s, i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(s)) / 2147483648.0, stereo.samples[i], 1e-6);
    }
}

// --- MP3 / OGG clip tests ---
//
// Fixture provenance: `tone_440_880.mp3` / `tone_440_880.ogg` (src/audio/
// fixtures/) are 0.06 s stereo sine tones (left 440 Hz, right 880 Hz, amp 0.5,
// 44100 Hz, 2646 source samples). Both are committed so the decode results
// below are fully deterministic.
//   mp3: ffmpeg -f lavfi -i "aevalsrc=0.5*sin(2*PI*440*t)|0.5*sin(2*PI*880*t):s=44100:d=0.06" -c:a libmp3lame -b:a 96k tone_440_880.mp3
//   ogg: libvorbis 1.3.7 (vorbis_encode_init_vbr, quality 0) via a one-shot
//        generator over the same samples; ffmpeg's bundled vorbis encoder is
//        experimental and produced silence, and homebrew ffmpeg has no
//        libvorbis, hence the direct libvorbis encode.
// Reference decode (vendored decoders, C probe): mp3 = 2646 frames @44100 Hz
// stereo (dr_mp3 honors the LAME gapless tag), ogg = 2646 frames per channel
// @44100 Hz stereo, per-channel RMS ~0.35.

fn clipRms(clip: *const AudioClip) f64 {
    if (clip.samples.len == 0) return 0.0;
    var sum: f64 = 0.0;
    for (clip.samples) |s| sum += @as(f64, s) * s;
    return @sqrt(sum / @as(f64, @floatFromInt(clip.samples.len)));
}

fn expectStereoTone(clip: *const AudioClip, want_frames: usize) !void {
    try std.testing.expectEqual(@as(f32, 44100.0), clip.sample_rate);
    try std.testing.expectEqual(want_frames, clip.frames);
    try std.testing.expectEqual(want_frames * 2, clip.samples.len);
    // Audible tone on both channels, channels distinct (440 vs 880 Hz).
    try std.testing.expect(clipRms(clip) > 0.2);
    var differ: usize = 0;
    for (0..want_frames) |i| {
        if (@abs(clip.samples[i * 2] - clip.samples[i * 2 + 1]) > 0.01) differ += 1;
    }
    try std.testing.expect(differ > want_frames / 4);
}

test "AudioClip decodes MP3 and OGG Vorbis from memory with expected duration, channels and RMS" {
    const alloc = std.testing.allocator;
    const Case = struct {
        name: []const u8,
        bytes: []const u8,
        decode: @TypeOf(AudioClip.fromMp3Memory),
    };
    const cases = [_]Case{
        .{ .name = "mp3", .bytes = @embedFile("fixtures/tone_440_880.mp3"), .decode = AudioClip.fromMp3Memory },
        .{ .name = "ogg", .bytes = @embedFile("fixtures/tone_440_880.ogg"), .decode = AudioClip.fromOggMemory },
    };
    inline for (cases) |case| {
        var clip = try case.decode(alloc, case.bytes);
        defer clip.deinit(alloc);
        try expectStereoTone(&clip, 2646);
    }
}

test "AudioClip MP3 and OGG file decodes match the memory decodes exactly" {
    const alloc = std.testing.allocator;
    const tio = std.testing.io;
    const cwd = std.Io.Dir.cwd();

    const mp3_bytes = @embedFile("fixtures/tone_440_880.mp3");
    const ogg_bytes = @embedFile("fixtures/tone_440_880.ogg");
    var mp3_mem = try AudioClip.fromMp3Memory(alloc, mp3_bytes);
    defer mp3_mem.deinit(alloc);
    var ogg_mem = try AudioClip.fromOggMemory(alloc, ogg_bytes);
    defer ogg_mem.deinit(alloc);

    // Isolated per-run directory: fixed CWD-relative fixture names would
    // collide with a concurrent run of the same suite.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const mp3_path = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/agate_mp3_fixture_test.mp3", .{tmp.sub_path});
    defer alloc.free(mp3_path);
    const ogg_path = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/agate_ogg_fixture_test.ogg", .{tmp.sub_path});
    defer alloc.free(ogg_path);

    if (cwd.writeFile(tio, .{ .sub_path = mp3_path, .data = mp3_bytes })) {
        var mp3_file = try AudioClip.fromMp3File(alloc, mp3_path);
        defer mp3_file.deinit(alloc);
        try std.testing.expectEqual(mp3_mem.frames, mp3_file.frames);
        try std.testing.expectEqual(mp3_mem.sample_rate, mp3_file.sample_rate);
        try std.testing.expectEqualSlices(f32, mp3_mem.samples, mp3_file.samples);
    } else |_| {}

    if (cwd.writeFile(tio, .{ .sub_path = ogg_path, .data = ogg_bytes })) {
        var ogg_file = try AudioClip.fromOggFile(alloc, ogg_path);
        defer ogg_file.deinit(alloc);
        try std.testing.expectEqual(ogg_mem.frames, ogg_file.frames);
        try std.testing.expectEqual(ogg_mem.sample_rate, ogg_file.sample_rate);
        try std.testing.expectEqualSlices(f32, ogg_mem.samples, ogg_file.samples);
    } else |_| {}
}

test "AudioClip rejects garbage MP3 and OGG data with clean errors" {
    const alloc = std.testing.allocator;
    // Empty input.
    try std.testing.expectError(error.InvalidMp3, AudioClip.fromMp3Memory(alloc, ""));
    try std.testing.expectError(error.InvalidOgg, AudioClip.fromOggMemory(alloc, ""));
    // No MP3 frame sync (no 0xFF byte) / no Ogg page magic: deterministic failures.
    const garbage = "This is definitely not an MP3 file";
    try std.testing.expectError(error.InvalidMp3, AudioClip.fromMp3Memory(alloc, garbage));
    try std.testing.expectError(error.InvalidOgg, AudioClip.fromOggMemory(alloc, garbage));
    // All-zero buffer: no sync, no pages.
    try std.testing.expectError(error.InvalidMp3, AudioClip.fromMp3Memory(alloc, &@as([512]u8, @splat(0))));
    // MP3 bytes carry no "OggS" magic, so the vorbis open must fail cleanly.
    try std.testing.expectError(error.InvalidOgg, AudioClip.fromOggMemory(alloc, @embedFile("fixtures/tone_440_880.mp3")));
}

test "AudioClip truncated MP3 and OGG data fail cleanly or decode fewer frames" {
    const alloc = std.testing.allocator;
    const mp3_bytes = @embedFile("fixtures/tone_440_880.mp3");
    const ogg_bytes = @embedFile("fixtures/tone_440_880.ogg");

    var full_mp3 = try AudioClip.fromMp3Memory(alloc, mp3_bytes);
    defer full_mp3.deinit(alloc);
    var full_ogg = try AudioClip.fromOggMemory(alloc, ogg_bytes);
    defer full_ogg.deinit(alloc);

    // MP3 cut to 10 bytes cannot hold a whole frame (>= ~100 bytes at this
    // bitrate): dr_mp3 finds no valid frame header, so this is a clean error.
    try std.testing.expectError(error.InvalidMp3, AudioClip.fromMp3Memory(alloc, mp3_bytes[0..10]));

    // Ogg pages carry CRCs: a half-truncated stream fails at open time.
    try std.testing.expectError(error.InvalidOgg, AudioClip.fromOggMemory(alloc, ogg_bytes[0 .. ogg_bytes.len / 2]));

    // MP3 truncated mid-payload may legitimately decode the frames it holds;
    // whatever happens, it must never crash or exceed the full decode.
    if (AudioClip.fromMp3Memory(alloc, mp3_bytes[0 .. mp3_bytes.len / 2])) |half_in| {
        var half = half_in;
        defer half.deinit(alloc);
        try std.testing.expect(half.frames > 0 and half.frames < full_mp3.frames);
        try std.testing.expectEqual(@as(f32, 44100.0), half.sample_rate);
    } else |err| {
        try std.testing.expectEqual(error.InvalidMp3, err);
    }
}

test "AudioClip fromMp3File and fromOggFile surface missing files" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.FileNotFound, AudioClip.fromMp3File(alloc, "agate_missing_file.mp3"));
    try std.testing.expectError(error.FileNotFound, AudioClip.fromOggFile(alloc, "agate_missing_file.ogg"));
}

test "AudioClip fromWavFile streams PCM in chunks" {
    const alloc = std.testing.allocator;
    const tio = std.testing.io;
    const cwd = std.Io.Dir.cwd();

    // 20000 stereo i16 frames = 160KB PCM: spans several 64KB chunks with a
    // partial tail, exercising the chunk loop boundaries.
    const frames = 20000;
    const left = try sineI16(alloc, frames, 44100, 440.0, 20000.0);
    defer alloc.free(left);
    const right = try sineI16(alloc, frames, 44100, 880.0, 10000.0);
    defer alloc.free(right);
    const raw = try alloc.alloc(u8, frames * 4);
    defer alloc.free(raw);
    for (0..frames) |i| {
        std.mem.writeInt(i16, raw[i * 4 ..][0..2], left[i], .little);
        std.mem.writeInt(i16, raw[i * 4 + 2 ..][0..2], right[i], .little);
    }
    const wav = try buildWav(alloc, true, 1, 2, 44100, 16, raw);
    defer alloc.free(wav);

    var mem = try AudioClip.fromWavMemory(alloc, wav);
    defer mem.deinit(alloc);
    try std.testing.expectEqual(@as(usize, frames), mem.frames);

    // Filesystem round-trip when a writable tmp file is available; otherwise
    // the conversions are verified through the memory path above and below.
    // Paths live in an isolated per-run directory so concurrent suite runs
    // never share fixture names.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_name = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/agate_audio_stream_test.wav", .{tmp.sub_path});
    defer alloc.free(tmp_name);
    const trunc_name = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/agate_audio_stream_test_trunc.wav", .{tmp.sub_path});
    defer alloc.free(trunc_name);
    const bad_name = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/agate_audio_stream_test_bad.wav", .{tmp.sub_path});
    defer alloc.free(bad_name);
    const u8name = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/agate_audio_stream_test_u8.wav", .{tmp.sub_path});
    defer alloc.free(u8name);
    var from_file: ?AudioClip = null;
    if (cwd.writeFile(tio, .{ .sub_path = tmp_name, .data = wav })) {
        from_file = try AudioClip.fromWavFile(alloc, tmp_name);

        // Truncated file must fail like the memory path.
        try cwd.writeFile(tio, .{ .sub_path = trunc_name, .data = wav[0 .. wav.len / 2] });
        try std.testing.expectError(error.InvalidWav, AudioClip.fromWavFile(alloc, trunc_name));

        // Unsupported format must fail like the memory path.
        const bad_raw = try alloc.alloc(u8, 12);
        defer alloc.free(bad_raw);
        @memset(bad_raw, 0);
        const bad_wav = try buildWav(alloc, true, 1, 3, 44100, 16, bad_raw);
        defer alloc.free(bad_wav);
        try cwd.writeFile(tio, .{ .sub_path = bad_name, .data = bad_wav });
        try std.testing.expectError(error.UnsupportedWavFormat, AudioClip.fromWavFile(alloc, bad_name));

        // Data-before-fmt chunk order with an odd-size pad byte.
        const u8raw = [_]u8{ 0, 64, 128, 192, 255 };
        const u8wav = try buildWav(alloc, false, 1, 1, 22050, 8, &u8raw);
        defer alloc.free(u8wav);
        try cwd.writeFile(tio, .{ .sub_path = u8name, .data = u8wav });
        var u8file = try AudioClip.fromWavFile(alloc, u8name);
        defer u8file.deinit(alloc);
        var u8mem = try AudioClip.fromWavMemory(alloc, u8wav);
        defer u8mem.deinit(alloc);
        try std.testing.expectEqual(u8mem.frames, u8file.frames);
        try std.testing.expectEqualSlices(f32, u8mem.samples, u8file.samples);
    } else |_| {}
    defer {
        if (from_file) |*c| c.deinit(alloc);
    }

    if (from_file) |*fc| {
        try std.testing.expectEqual(mem.frames, fc.frames);
        try std.testing.expectEqual(mem.sample_rate, fc.sample_rate);
        try std.testing.expectEqualSlices(f32, mem.samples, fc.samples);
    }
}

test "AudioEngine audio buses volume, mute, and stop isolation" {
    var eng = AudioEngine{};
    const sfx = eng.createSpatialBus("sfx", .{}) orelse return error.BusCreationFailed;
    const music = eng.createNonSpatialBus("music", 1.0) orelse return error.BusCreationFailed;

    try std.testing.expectEqual(@as(f32, 1.0), eng.getBusVolume(sfx));
    try std.testing.expectEqual(false, eng.isBusMuted(sfx));

    // Play on sfx and check output
    eng.play(.{ .kind = .thump, .bus = sfx, .volume = 0.8, .duration = 0.1, .freq = 120.0, .freq_end = 60.0 });
    var buf_sfx: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf_sfx);
    const e_sfx = channelEnergy(&buf_sfx, 0);
    try std.testing.expect(e_sfx > 0.01);

    // Mute sfx, play again -> should produce pure silence
    eng.setBusMuted(sfx, true);
    try std.testing.expect(eng.isBusMuted(sfx));
    eng.play(.{ .kind = .thump, .bus = sfx, .volume = 0.8, .duration = 0.1, .freq = 120.0, .freq_end = 60.0 });
    var buf_muted: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf_muted);
    try std.testing.expect(peakAbs(&buf_muted, 0, 2205) < 1e-6);

    // music is unmuted, so music should still play while sfx is muted
    eng.play(.{ .kind = .thump, .bus = music, .volume = 0.8, .duration = 0.1, .freq = 120.0, .freq_end = 60.0 });
    var buf_music: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf_music);
    try std.testing.expect(channelEnergy(&buf_music, 0) > 0.01);

    // Unmute sfx, reduce volume to 0.2 -> energy should be noticeably lower
    eng.setBusMuted(sfx, false);
    eng.setBusVolume(sfx, 0.2);
    try std.testing.expectEqual(@as(f32, 0.2), eng.getBusVolume(sfx));
    eng.play(.{ .kind = .thump, .bus = sfx, .volume = 0.8, .duration = 0.1, .freq = 120.0, .freq_end = 60.0 });
    var buf_half: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf_half);
    const e_half = channelEnergy(&buf_half, 0);
    try std.testing.expect(e_half < e_sfx * 0.2);

    // stopBus clears active voices on that bus
    eng.play(.{ .kind = .blip, .bus = music, .duration = 1.0, .freq = 440.0, .freq_end = 440.0 });
    eng.renderFrames(&buf_half); // process command into active voice
    var active_music: usize = 0;
    for (&eng.voices) |*v| {
        if (v.isActive() and v.busId() == music) active_music += 1;
    }
    try std.testing.expect(active_music > 0);
    eng.stopBus(music);
    active_music = 0;
    for (&eng.voices) |*v| {
        if (v.isActive() and v.busId() == music) active_music += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), active_music);
}

test "AudioEngine attenuation models and Doppler shift calculation" {
    const lpos = Vec3.zero;
    const lright = Vec3.new(1.0, 0.0, 0.0);
    const lvel = Vec3.zero;

    // 1. Inside min_dist (0..2.0): full volume
    const inside = AudioEngine.spatializeWith(lpos, lright, lvel, Vec3.new(0.0, 1.0, 0.0), 1.0, 2.0, 20.0, 1.0, .linear, null);
    try std.testing.expectEqual(@as(f32, 1.0), inside.vol);

    // 2. Beyond max_dist: zero volume
    const outside = AudioEngine.spatializeWith(lpos, lright, lvel, Vec3.new(0.0, 25.0, 0.0), 1.0, 2.0, 20.0, 1.0, .linear, null);
    try std.testing.expectEqual(@as(f32, 0.0), outside.vol);

    // 3. Linear vs Inverse vs Exponential at midpoint (dist = 11.0, range = 18.0)
    const mid_pos = Vec3.new(0.0, 11.0, 0.0);
    const lin = AudioEngine.spatializeWith(lpos, lright, lvel, mid_pos, 1.0, 2.0, 20.0, 1.0, .linear, null);
    const inv = AudioEngine.spatializeWith(lpos, lright, lvel, mid_pos, 1.0, 2.0, 20.0, 1.0, .inverse, null);
    const exp = AudioEngine.spatializeWith(lpos, lright, lvel, mid_pos, 1.0, 2.0, 20.0, 1.0, .exponential, null);
    try std.testing.expect(lin.vol > 0.4 and lin.vol < 0.6); // 1.0 - 9/18 = 0.5
    try std.testing.expect(inv.vol > 0.15 and inv.vol < 0.25); // 2 / (2 + 9) = 2/11 ~= 0.18
    try std.testing.expect(exp.vol > 0.55 and exp.vol < 0.65); // exp(-9/18) = exp(-0.5) ~= 0.606

    // 4. Doppler effect: source approaching listener (velocity towards listener)
    const approach_vel = Vec3.new(0.0, -34.3, 0.0); // emitter at (0, 10, 0) moving towards (0, 0, 0)
    const dop_app = AudioEngine.spatializeWith(lpos, lright, lvel, Vec3.new(0.0, 10.0, 0.0), 1.0, 1.0, 30.0, 1.0, .linear, approach_vel);
    try std.testing.expect(dop_app.doppler > 1.08); // pitch shifted up!

    // Source receding from listener (velocity away from listener)
    const recede_vel = Vec3.new(0.0, 34.3, 0.0);
    const dop_rec = AudioEngine.spatializeWith(lpos, lright, lvel, Vec3.new(0.0, 10.0, 0.0), 1.0, 1.0, 30.0, 1.0, .linear, recede_vel);
    try std.testing.expect(dop_rec.doppler < 0.95); // pitch shifted down!
}

test "AudioEngine spatial vs non-spatial bus behavior and runtime toggling" {
    var eng = AudioEngine{};
    eng.updateListener(Vec3.zero, Vec3.new(1.0, 0.0, 0.0));

    // Create a dynamic spatial bus (max distance 30)
    const sp_bus = eng.createSpatialBus("sp_effects", .{
        .max_distance = 30.0,
    }) orelse return error.BusCreationFailed;

    // Create a dynamic non-spatial bus
    const flat_bus = eng.createNonSpatialBus("flat_cues", 1.0) orelse return error.BusCreationFailed;

    // 1. Play sound at (0, 100, 0) on spatial bus -> beyond max_distance, MUST be silent!
    eng.play(.{
        .kind = .thump,
        .bus = sp_bus,
        .position = Vec3.new(0.0, 100.0, 0.0),
        .volume = 0.8,
        .duration = 0.1,
    });
    var buf_silent: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf_silent);
    try std.testing.expect(peakAbs(&buf_silent, 0, 2205) < 1e-6);

    // 2. Play identical sound with identical position (0, 100, 0) on non-spatial bus -> MUST be audible!
    eng.play(.{
        .kind = .thump,
        .bus = flat_bus,
        .position = Vec3.new(0.0, 100.0, 0.0), // distance is ignored by non-spatial bus!
        .volume = 0.8,
        .duration = 0.1,
    });
    var buf_audible: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf_audible);
    try std.testing.expect(channelEnergy(&buf_audible, 0) > 0.01);

    // 3. Non-spatial bus direct stereo panning (pan = -1.0 -> left speaker only)
    eng.stopAll();
    eng.play(.{
        .kind = .thump,
        .bus = flat_bus,
        .pan = -1.0,
        .volume = 0.8,
        .duration = 0.1,
    });
    var buf_left: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf_left);
    try std.testing.expect(channelEnergy(&buf_left, 0) > 0.01);
    try std.testing.expect(channelEnergy(&buf_left, 1) < 1e-6);

    // 4. Runtime toggling: toggle sp_bus to non-spatial via setBusSpatial(false)!
    eng.stopAll();
    eng.setBusSpatial(sp_bus, false);
    try std.testing.expect(!eng.isBusSpatial(sp_bus));

    // Play again at 100m on sp_bus -> now that it is non-spatial, it MUST be audible!
    eng.play(.{
        .kind = .thump,
        .bus = sp_bus,
        .position = Vec3.new(0.0, 100.0, 0.0),
        .volume = 0.8,
        .duration = 0.1,
    });
    var buf_toggled: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf_toggled);
    try std.testing.expect(channelEnergy(&buf_toggled, 0) > 0.01);

    // Toggle back to spatial -> distant sound is silent again!
    eng.stopAll();
    eng.setBusSpatial(sp_bus, true);
    try std.testing.expect(eng.isBusSpatial(sp_bus));
    eng.play(.{
        .kind = .thump,
        .bus = sp_bus,
        .position = Vec3.new(0.0, 100.0, 0.0),
        .volume = 0.8,
        .duration = 0.1,
    });
    var buf_back: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf_back);
    try std.testing.expect(peakAbs(&buf_back, 0, 2205) < 1e-6);
}

test "AudioEngine bus hierarchy and effective volume/mute propagation" {
    var eng = AudioEngine{};
    // Create root bus
    const root = eng.createNonSpatialBus("root", 0.8) orelse return error.BusCreationFailed;
    // Create parent bus routing into root
    const parent = eng.createNonSpatialBus("combat_mix", 0.5) orelse return error.BusCreationFailed;
    eng.setBusParent(parent, root);
    // Create child bus routing into parent
    const child = eng.createNonSpatialBus("gunshots", 0.5) orelse return error.BusCreationFailed;
    eng.setBusParent(child, parent);

    try std.testing.expectEqual(parent, eng.getBusParent(child).?);

    // root local = 0.8, parent local = 0.5, child local = 0.5.
    // child effective = 0.5 * 0.5 * 0.8 = 0.20
    const eff_child = eng.getBusEffectiveVolume(child);
    try std.testing.expectApproxEqAbs(@as(f32, 0.20), eff_child, 1e-4);

    // Muting parent mutes child effectively
    eng.setBusMuted(parent, true);
    try std.testing.expect(eng.isBusMuted(parent));
    try std.testing.expect(!eng.isBusMuted(child)); // child local flag is false
    try std.testing.expectEqual(@as(f32, 0.0), eng.getBusEffectiveVolume(child)); // but effective volume is 0!

    // Rendering on child while parent is muted produces silence
    eng.play(.{ .kind = .thump, .bus = child, .volume = 0.8, .duration = 0.1 });
    var buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf);
    try std.testing.expect(peakAbs(&buf, 0, 2205) < 1e-6);

    // Unmute parent, child becomes audible again
    eng.setBusMuted(parent, false);
    try std.testing.expectApproxEqAbs(@as(f32, 0.20), eng.getBusEffectiveVolume(child), 1e-4);
    eng.play(.{ .kind = .thump, .bus = child, .volume = 0.8, .duration = 0.1 });
    eng.renderFrames(&buf);
    try std.testing.expect(channelEnergy(&buf, 0) > 0.001);

    // Destroying parent re-parents child to parent's parent (root)
    eng.destroyBus(parent);
    try std.testing.expectEqual(root, eng.getBusParent(child).?);
    // Now child routes directly to root: 0.5 * 0.8 = 0.40
    try std.testing.expectApproxEqAbs(@as(f32, 0.40), eng.getBusEffectiveVolume(child), 1e-4);
}

test "AudioEngine configurable bus capacity at initialization" {
    // 1. Default initialization has max_buses = 32
    var def_eng = AudioEngine{};
    try std.testing.expectEqual(@as(usize, 32), def_eng.getBusCapacity());

    // 2. Custom small capacity: max_buses = 8
    var small_eng = AudioEngine.init(.{ .max_buses = 8 });
    try std.testing.expectEqual(@as(usize, 8), small_eng.getBusCapacity());
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        var name_buf: [16]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "bus_{d}", .{i}) catch "b";
        const b = small_eng.createNonSpatialBus(name, 1.0);
        try std.testing.expect(b != null);
    }
    // 9th bus creation MUST fail (returns null) because capacity is 8!
    try std.testing.expect(small_eng.createNonSpatialBus("overflow", 1.0) == null);
    try std.testing.expectEqual(@as(usize, 8), small_eng.getBusCount());

    // 3. Custom large capacity: max_buses = 64 (well above default 32)
    var large_eng = AudioEngine.init(.{ .max_buses = 64 });
    try std.testing.expectEqual(@as(usize, 64), large_eng.getBusCapacity());
    var created_ids: [48]BusId = undefined;
    for (0..48) |idx| {
        var name_buf: [16]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "wide_{d}", .{idx}) catch "w";
        const b = large_eng.createSpatialBus(name, .{ .volume = 0.8 });
        try std.testing.expect(b != null);
        created_ids[idx] = b.?;
    }
    try std.testing.expectEqual(@as(usize, 48), large_eng.getBusCount());

    // Test parent-child relationship on buses above 32 (e.g. 45 -> 40)
    large_eng.setBusParent(created_ids[45], created_ids[40]);
    try std.testing.expectEqual(created_ids[40], large_eng.getBusParent(created_ids[45]).?);
    large_eng.setBusMuted(created_ids[40], true);
    try std.testing.expectEqual(@as(f32, 0.0), large_eng.getBusEffectiveVolume(created_ids[45]));

    // 4. Clamping behavior
    var clamped_eng = AudioEngine.init(.{ .max_buses = 500 });
    try std.testing.expectEqual(@as(usize, 128), clamped_eng.getBusCapacity());
}

test "AudioEngine bus filter lowpass attenuates high frequency sounds" {
    var eng = AudioEngine{};
    const raw_bus = eng.createNonSpatialBus("raw", 1.0) orelse return error.BusCreationFailed;
    const lp_bus = eng.createNonSpatialBus("lp", 1.0) orelse return error.BusCreationFailed;

    // Apply lowpass filter at 300 Hz to lp_bus
    eng.setBusFilter(lp_bus, .lowpass, 300.0, 0.7071);
    const cfg = eng.getBusFilter(lp_bus);
    try std.testing.expectEqual(BiquadFilterType.lowpass, cfg.filter_type);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), cfg.cutoff, 0.1);

    // Play high-frequency 3000 Hz tone on raw_bus
    eng.play(.{
        .kind = .blip,
        .bus = raw_bus,
        .volume = 0.8,
        .duration = 0.05,
        .freq = 3000.0,
        .freq_end = 3000.0,
    });
    var raw_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&raw_buf);
    const raw_energy = channelEnergy(&raw_buf, 0);
    try std.testing.expect(raw_energy > 0.1);

    // Now play the same 3000 Hz tone on lp_bus
    eng.play(.{
        .kind = .blip,
        .bus = lp_bus,
        .volume = 0.8,
        .duration = 0.05,
        .freq = 3000.0,
        .freq_end = 3000.0,
    });
    var lp_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&lp_buf);
    const lp_energy = channelEnergy(&lp_buf, 0);

    // 3000 Hz is 10x cutoff (300 Hz), 2nd order Butterworth drops ~40 dB => energy should be < 10% of raw
    try std.testing.expect(lp_energy < raw_energy * 0.1);

    // Open filter cutoff to 5000 Hz
    eng.setBusFilterCutoff(lp_bus, 5000.0);
    eng.play(.{
        .kind = .blip,
        .bus = lp_bus,
        .volume = 0.8,
        .duration = 0.05,
        .freq = 3000.0,
        .freq_end = 3000.0,
    });
    var open_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&open_buf);
    const open_energy = channelEnergy(&open_buf, 0);
    try std.testing.expect(open_energy > lp_energy * 5.0);

    // Clear filter
    eng.clearBusFilter(lp_bus);
    try std.testing.expectEqual(BiquadFilterType.none, eng.getBusFilter(lp_bus).filter_type);
}

test "AudioEngine bus reverb generates reverberation tail after voice completes" {
    var eng = AudioEngine{};
    const dry_bus = eng.createNonSpatialBus("dry", 1.0) orelse return error.BusCreationFailed;
    const rev_bus = eng.createNonSpatialBus("reverb", 1.0) orelse return error.BusCreationFailed;

    try std.testing.expect(!eng.isBusReverbEnabled(rev_bus));
    try std.testing.expect(eng.setBusReverb(rev_bus, .{
        .room_size = 0.8,
        .damping = 0.2,
        .wet = 0.6,
        .dry = 0.8,
    }));
    try std.testing.expect(eng.isBusReverbEnabled(rev_bus));

    // Play short 0.02s impulse on dry bus
    eng.play(.{
        .kind = .blip,
        .bus = dry_bus,
        .volume = 0.8,
        .duration = 0.02,
        .freq = 440.0,
        .freq_end = 440.0,
    });
    var initial_buf: [4410]f32 = @as([4410]f32, @splat(0.0)); // 0.05s -> voice finishes here
    eng.renderFrames(&initial_buf);

    // Next buffer (t in [0.05, 0.10]): dry bus should be completely silent
    var dry_tail_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&dry_tail_buf);
    try std.testing.expect(peakAbs(&dry_tail_buf, 0, 2205) < 1e-6);

    // Now play short 0.02s impulse on reverb bus
    eng.play(.{
        .kind = .blip,
        .bus = rev_bus,
        .volume = 0.8,
        .duration = 0.02,
        .freq = 440.0,
        .freq_end = 440.0,
    });
    eng.renderFrames(&initial_buf); // voice finishes

    // Next buffer: reverb tail should still be ringing out audibly!
    var rev_tail_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&rev_tail_buf);
    const rev_peak = peakAbs(&rev_tail_buf, 0, 2205);
    try std.testing.expect(rev_peak > 1e-4);

    // Clear reverb releases slot
    eng.clearBusReverb(rev_bus);
    try std.testing.expect(!eng.isBusReverbEnabled(rev_bus));
}

test "AudioEngine DAG hierarchy mixes child filtered audio into parent reverb" {
    var eng = AudioEngine{};
    const parent = eng.createNonSpatialBus("parent_hall", 1.0) orelse return error.BusCreationFailed;
    const child = eng.createNonSpatialBus("child_synth", 1.0) orelse return error.BusCreationFailed;
    eng.setBusParent(child, parent);

    // Parent has cave reverb
    _ = eng.setBusCaveReverb(parent);
    // Child has underwater lowpass filter
    eng.setBusUnderwater(child);

    // Play audio on child bus
    eng.play(.{
        .kind = .blip,
        .bus = child,
        .volume = 0.8,
        .duration = 0.02,
        .freq = 440.0,
        .freq_end = 440.0,
    });

    var buf1: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&buf1);
    try std.testing.expect(channelEnergy(&buf1, 0) > 0.01);

    // Tail buffer after voice has ended: parent reverb should be ringing
    var tail: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&tail);
    try std.testing.expect(peakAbs(&tail, 0, 2205) > 1e-4);
}

test "AudioEngine play procedural voice with occlusion attenuates energy and cutoff" {
    var eng = AudioEngine{};
    // 1. Play unoccluded tone
    eng.play(.{
        .kind = .blip,
        .volume = 0.8,
        .duration = 0.05,
        .freq = 4000.0,
        .freq_end = 4000.0,
        .cutoff = 5000.0,
        .occlusion = 0.0,
    });
    var unocc_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&unocc_buf);
    const unocc_energy = channelEnergy(&unocc_buf, 0);
    try std.testing.expect(unocc_energy > 0.05);

    // 2. Play fully occluded tone (occlusion = 1.0)
    eng.play(.{
        .kind = .blip,
        .volume = 0.8,
        .duration = 0.05,
        .freq = 4000.0,
        .freq_end = 4000.0,
        .cutoff = 5000.0,
        .occlusion = 1.0,
        .occlusion_config = .{
            .min_volume = 0.2,
            .min_cutoff = 400.0,
            .max_cutoff = 20000.0,
        },
    });
    var occ_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&occ_buf);
    const occ_energy = channelEnergy(&occ_buf, 0);

    // Occluded sound has min_volume = 0.2 (energy ~ 0.04) plus cutoff dropped to 400 Hz
    // so energy should be a tiny fraction of unoccluded
    try std.testing.expect(occ_energy < unocc_energy * 0.1);
}

test "AudioEngine playClip with occlusion attenuates volume and lowpass filters WAV audio" {
    const alloc = std.testing.allocator;
    // Generate 3000 Hz sine wave (2205 frames = 0.05s at 44.1 kHz, 4410 stereo samples)
    const built = try monoSineWav(alloc, 2205, 44100, 3000.0, 20000.0);
    defer built.deinit(alloc);

    var clip = try AudioClip.fromWavMemory(alloc, built.wav);
    defer clip.deinit(alloc);

    var eng = AudioEngine{};

    // Play unoccluded clip
    eng.playClip(&clip, .{
        .volume = 0.8,
        .occlusion = 0.0,
    });
    var unocc_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&unocc_buf);
    const unocc_energy = channelEnergy(&unocc_buf, 0);
    try std.testing.expect(unocc_energy > 0.05);

    // Play occluded clip with fresh engine (min_cutoff = 500 Hz, while audio is 3000 Hz)
    var eng_occ = AudioEngine{};
    eng_occ.playClip(&clip, .{
        .volume = 0.8,
        .occlusion = 1.0,
        .occlusion_config = .{
            .min_volume = 0.25,
            .min_cutoff = 500.0,
            .max_cutoff = 20000.0,
        },
    });
    var occ_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng_occ.renderFrames(&occ_buf);
    const occ_energy = channelEnergy(&occ_buf, 0);

    // Volume scaled to 0.25 (-12 dB) and 3000 Hz filtered by 500 Hz one-pole filter => energy << unoccluded
    try std.testing.expect(occ_energy < unocc_energy * 0.1);
}

test "AudioEngine bus occlusion sets lowpass cutoff and attenuates volume" {
    var eng = AudioEngine{};
    const b = eng.createNonSpatialBus("monitored_bus", 1.0) orelse return error.BusCreationFailed;

    try std.testing.expectEqual(@as(f32, 0.0), eng.getBusOcclusion(b));

    eng.setBusOcclusionConfig(b, .{
        .min_volume = 0.2,
        .min_cutoff = 400.0,
        .max_cutoff = 20000.0,
    });
    eng.setBusOcclusion(b, 0.9);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), eng.getBusOcclusion(b), 1e-4);

    // Play tone on occluded bus
    eng.play(.{
        .kind = .blip,
        .bus = b,
        .volume = 0.8,
        .duration = 0.05,
        .freq = 2500.0,
        .freq_end = 2500.0,
    });
    var occ_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&occ_buf);
    const occ_energy = channelEnergy(&occ_buf, 0);

    // Clear occlusion and play same tone
    eng.clearBusOcclusion(b);
    try std.testing.expectEqual(@as(f32, 0.0), eng.getBusOcclusion(b));

    eng.play(.{
        .kind = .blip,
        .bus = b,
        .volume = 0.8,
        .duration = 0.05,
        .freq = 2500.0,
        .freq_end = 2500.0,
    });
    var unocc_buf: [4410]f32 = @as([4410]f32, @splat(0.0));
    eng.renderFrames(&unocc_buf);
    const unocc_energy = channelEnergy(&unocc_buf, 0);

    try std.testing.expect(occ_energy < unocc_energy * 0.15);
}

test "AudioEngine bus occlusion smooth temporal update" {
    var eng = AudioEngine{};
    const b = eng.createNonSpatialBus("smooth_bus", 1.0) orelse return error.BusCreationFailed;
    eng.setBusOcclusionConfig(b, .{ .smooth_time = 0.10 });

    // Step 0.05s towards target 1.0: should interpolate smoothly without reaching 1.0 immediately
    eng.updateBusOcclusion(b, 1.0, 0.05);
    const step1 = eng.getBusOcclusion(b);
    try std.testing.expect(step1 > 0.2 and step1 < 0.6);

    // After several seconds: converges to 1.0
    eng.updateBusOcclusion(b, 1.0, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), eng.getBusOcclusion(b), 1e-3);
}

test "PhysicsWorld audio occlusion raycast integration" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    // Create a wall mesh & body at x = 5.0
    var m = Mesh{
        .name = "wall",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(5.0, 0.0, 0.0),
    };
    m.local_bounding_box = math.BoundingBox.init(Vec3.new(-0.2, -5.0, -5.0), Vec3.new(0.2, 5.0, 5.0));
    _ = try pw.createBody(&m, .box, 0.0); // static body

    const listener = Vec3.new(0.0, 0.0, 0.0);
    const emitter_behind = Vec3.new(10.0, 0.0, 0.0);
    const emitter_clear = Vec3.new(2.0, 0.0, 0.0);

    // Emitter behind wall is occluded (1.0)
    const occ_behind = pw.evaluateAudioOcclusion(listener, emitter_behind, .{});
    try std.testing.expectEqual(@as(f32, 1.0), occ_behind);

    // Emitter in front of wall is clear (0.0)
    const occ_clear = pw.evaluateAudioOcclusion(listener, emitter_clear, .{});
    try std.testing.expectEqual(@as(f32, 0.0), occ_clear);
}

test "AudioStream compressed streaming from memory with pause, resume, and stop" {
    const alloc = std.testing.allocator;
    // One row per compressed codec: the full open/render/pause/resume/stop
    // lifecycle from the old OGG-only test now runs for both, so the MP3
    // path gains the lifecycle assertions (same AudioStream API).
    const Case = struct {
        name: []const u8,
        bytes: []const u8,
        format: audio.StreamFormat,
        volume: f32,
    };
    const cases = [_]Case{
        .{ .name = "ogg", .bytes = @embedFile("fixtures/tone_440_880.ogg"), .format = .ogg, .volume = 0.75 },
        .{ .name = "mp3", .bytes = @embedFile("fixtures/tone_440_880.mp3"), .format = .mp3, .volume = 0.9 },
    };
    for (cases) |case| {
        var stream = try audio.AudioStream.openMemory(alloc, case.bytes, case.format, 44100.0, .{
            .volume = case.volume,
            .pan = 0.0,
            .loop = true,
            .buffer_frames = 4096,
        });
        defer stream.deinit();

        try std.testing.expect(stream.isPlaying());
        try std.testing.expect(!stream.isPaused());
        try std.testing.expect(!stream.isStopped());
        try std.testing.expectApproxEqAbs(case.volume, stream.getVolume(), 1e-4);
        try std.testing.expect(stream.getDurationSeconds() > 0.0);

        var buf: [2048]f32 = @as([2048]f32, @splat(0.0));
        const rendered = stream.renderToBuffer(&buf);
        try std.testing.expectEqual(@as(usize, 1024), rendered);
        try std.testing.expect(channelEnergy(&buf, 0) > 0.01);

        // Pause
        stream.pause();
        try std.testing.expect(stream.isPaused());
        try std.testing.expect(!stream.isPlaying());

        // Resume
        stream.unpause();
        try std.testing.expect(stream.isPlaying());

        // Refill and render again
        stream.update(0.05);
        @memset(&buf, 0.0);
        const rendered2 = stream.renderToBuffer(&buf);
        try std.testing.expect(rendered2 > 0);
        try std.testing.expect(channelEnergy(&buf, 0) > 0.01);

        // Stop
        stream.stop();
        try std.testing.expect(stream.isStopped());
        try std.testing.expectEqual(@as(f32, 0.0), stream.getPositionSeconds());
    }
}

test "AudioStream MP3 memory and file refills, seek and rewind match decoded PCM" {
    const alloc = std.testing.allocator;
    const bytes = @embedFile("fixtures/tone_440_880.mp3");
    var clip = try AudioClip.fromMp3Memory(alloc, bytes);
    defer clip.deinit(alloc);
    try std.testing.expect(clip.frames > 1024);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/stream.mp3", .{tmp.sub_path});
    defer alloc.free(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = bytes });

    for ([_]bool{ false, true }) |from_file| {
        const options = audio.StreamOptions{ .buffer_frames = 1024 };
        const stream = if (from_file)
            try audio.AudioStream.openFile(alloc, path, clip.sample_rate, options)
        else
            try audio.AudioStream.openMemory(alloc, bytes, .mp3, clip.sample_rate, options);
        defer stream.deinit();

        var buf: [1024]f32 = undefined;
        var frame: usize = 0;
        while (frame < clip.frames) {
            stream.update(0.0);
            @memset(&buf, 0.0);
            const got = stream.renderToBuffer(&buf);
            try std.testing.expect(got > 0);
            try std.testing.expect(frame + got <= clip.frames);
            try std.testing.expectEqualSlices(f32, clip.samples[frame * 2 .. (frame + got) * 2], buf[0 .. got * 2]);
            frame += got;
        }

        const seek_frame = clip.frames / 2;
        stream.seekToFrame(seek_frame);
        stream.play();
        @memset(&buf, 0.0);
        const got_seek = stream.renderToBuffer(&buf);
        try std.testing.expectEqual(@as(usize, buf.len / 2), got_seek);
        try std.testing.expectEqualSlices(f32, clip.samples[seek_frame * 2 .. (seek_frame + got_seek) * 2], &buf);

        stream.stop();
        stream.play();
        stream.refill();
        @memset(&buf, 0.0);
        const got_rewind = stream.renderToBuffer(&buf);
        try std.testing.expectEqual(@as(usize, buf.len / 2), got_rewind);
        try std.testing.expectEqualSlices(f32, clip.samples[0..buf.len], &buf);
    }
}

fn openMp3StreamUnderOom(alloc: std.mem.Allocator, path: ?[]const u8) !void {
    const options = audio.StreamOptions{ .buffer_frames = 1024 };
    const stream = if (path) |p|
        try audio.AudioStream.openFile(alloc, p, 44100.0, options)
    else
        try audio.AudioStream.openMemory(alloc, @embedFile("fixtures/tone_440_880.mp3"), .mp3, 44100.0, options);
    defer stream.deinit();
}

test "AudioStream MP3 creation rolls back every allocation and invalid input" {
    const alloc = std.testing.allocator;
    try std.testing.checkAllAllocationFailures(alloc, openMp3StreamUnderOom, .{null});

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/stream.mp3", .{tmp.sub_path});
    defer alloc.free(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = @embedFile("fixtures/tone_440_880.mp3") });
    try std.testing.checkAllAllocationFailures(alloc, openMp3StreamUnderOom, .{path});

    const garbage = "not an MP3 stream";
    try std.testing.expectError(error.InvalidMp3, audio.AudioStream.openMemory(alloc, garbage, .mp3, 44100.0, .{}));
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = garbage });
    try std.testing.expectError(error.InvalidMp3, audio.AudioStream.openFile(alloc, path, 44100.0, .{}));
}

test "AudioStream WAV streaming from memory and auto-stop when loop is false" {
    const alloc = std.testing.allocator;
    // 2205 frames of 440 Hz sine = 0.05s
    const built = try monoSineWav(alloc, 2205, 44100, 440.0, 20000.0);
    defer built.deinit(alloc);
    const wav = built.wav;

    var stream = try audio.AudioStream.openMemory(alloc, wav, .wav, 44100.0, .{
        .volume = 0.8,
        .loop = false,
        .buffer_frames = 1024,
    });
    defer stream.deinit();

    try std.testing.expect(stream.isPlaying());

    var total_rendered: usize = 0;
    var buf: [1024]f32 = @as([1024]f32, @splat(0.0));
    while (!stream.isStopped()) {
        stream.update(0.02);
        @memset(&buf, 0.0);
        const got = stream.renderToBuffer(&buf);
        if (got == 0) break;
        total_rendered += got;
        if (total_rendered > 5000) break; // safety breaker
    }

    try std.testing.expectEqual(@as(usize, 2205), total_rendered);
    try std.testing.expect(stream.isStopped());
}

test "AudioStream seamless looping renders continuously past track length" {
    const alloc = std.testing.allocator;
    const ogg_bytes = @embedFile("fixtures/tone_440_880.ogg");

    var stream = try audio.AudioStream.openMemory(alloc, ogg_bytes, .ogg, 44100.0, .{
        .loop = true,
        .buffer_frames = 2048,
    });
    defer stream.deinit();

    var buf: [1024]f32 = @as([1024]f32, @splat(0.0));
    // Track is 2646 frames. Render 4 x 512 = 2048 frames, then refill and render 2048 more = 4096 frames (> 2646)
    var total_frames: usize = 0;
    for (0..8) |_| {
        stream.update(0.02);
        @memset(&buf, 0.0);
        const got = stream.renderToBuffer(&buf);
        try std.testing.expect(got > 0);
        try std.testing.expect(channelEnergy(&buf, 0) > 0.005);
        total_frames += got;
    }

    try std.testing.expect(total_frames > 2646);
    try std.testing.expect(stream.isPlaying());
}

test "AudioStream volume fadeTo smoothly interpolates and stops on fade out" {
    const alloc = std.testing.allocator;
    const ogg_bytes = @embedFile("fixtures/tone_440_880.ogg");

    var stream = try audio.AudioStream.openMemory(alloc, ogg_bytes, .ogg, 44100.0, .{
        .volume = 1.0,
        .loop = true,
        .buffer_frames = 2048,
    });
    defer stream.deinit();

    stream.fadeTo(0.0, 0.1, true); // fade to 0 in 0.1s and stop

    // Halfway through fade (0.05s)
    stream.update(0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), stream.getVolume(), 0.05);
    try std.testing.expect(stream.isPlaying());

    // Complete fade (0.06s more = 0.11s total)
    stream.update(0.06);
    try std.testing.expect(stream.getVolume() <= 0.001);
    try std.testing.expect(stream.isStopped());
}

test "AudioEngine playSoundOnce auto-destroys on stream completion" {
    const alloc = std.testing.allocator;
    const built = try monoSineWav(alloc, 50, 44100, 440.0, 20000.0);
    defer built.deinit(alloc);
    const wav = built.wav;

    var eng = AudioEngine{};

    try eng.playSoundOnceFromMemory(alloc, wav, .wav, .{
        .buffer_frames = 512,
    });

    // Verify stream was registered
    var active_stream: ?*audio.AudioStream = null;
    for (0..AudioEngine.max_streams) |i| {
        if (streams_mod.getSlot(&eng, i)) |s| {
            active_stream = s;
            break;
        }
    }
    try std.testing.expect(active_stream != null);
    try std.testing.expect(active_stream.?.auto_destroy);

    // Render enough frames to consume 50 frames and hit EOF
    var out_buf: [256]f32 = @as([256]f32, @splat(0.0));
    eng.renderFrames(&out_buf);

    // Main thread updates streams: stream hit EOF and stopped, should auto-destroy
    eng.updateStreams(0.01);

    // Stream slot should now be null
    var has_any = false;
    for (0..AudioEngine.max_streams) |i| {
        if (streams_mod.getSlot(&eng, i) != null) {
            has_any = true;
            break;
        }
    }
    try std.testing.expect(!has_any);
}

test "AudioEngine crossfadeSound and stopAllSounds" {
    const alloc = std.testing.allocator;
    const ogg_bytes = @embedFile("fixtures/tone_440_880.ogg");

    var eng = AudioEngine{};

    const s1 = try eng.playSoundFromMemory(alloc, ogg_bytes, .ogg, .{
        .volume = 0.8,
        .loop = true,
        .buffer_frames = 4096,
    });

    const s2 = try eng.crossfadeSoundFromMemory(s1, alloc, ogg_bytes, .ogg, 0.05, .{
        .volume = 0.9,
        .loop = true,
        .buffer_frames = 4096,
    });

    // Advance fade past 0.05s
    eng.updateStreams(0.06);

    // s1 should have stopped and auto-cleaned up, s2 should be at full volume
    var has_s1 = false;
    for (0..AudioEngine.max_streams) |i| {
        if (streams_mod.getSlot(&eng, i) == s1) has_s1 = true;
    }
    try std.testing.expect(!has_s1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), s2.getVolume(), 0.01);

    // Now test stopAllSounds with fade-out
    eng.stopAllSounds(0.05);
    eng.updateStreams(0.06);

    // All streams should be cleaned up
    var count: usize = 0;
    for (0..AudioEngine.max_streams) |i| {
        if (streams_mod.getSlot(&eng, i) != null) count += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), count);
}

// --- perf/audio-tag-query characterization: noise cutoff-constant hoist ---
//
// Test-only legacy reference: the pre-hoist per-sample `renderNoise` loop
// (cutoff + `alpha` recomputed every sample) fused with the production
// master-gain + soft-clip output stage. Engine output through the public
// `play`/`renderFrames` API must match it sample-for-sample, proving the
// constant-cutoff hoist changes no signal, seed/LP/time progression,
// expiry, or clamping.
fn legacyNoiseBurst(
    out: []f32,
    frames: usize,
    sample_rate: f32,
    cutoff0: f32,
    cutoff_end: f32,
    duration: f32,
    seed0: u32,
    volume: f32,
    pan: f32,
    master: f32,
) void {
    std.debug.assert(out.len >= frames * 2);
    // NOTE: every input arrives as a runtime parameter, so all float
    // math below evaluates at runtime exactly like the mixer kernel.
    // Folding any of these from literals would let comptime arithmetic
    // round differently (1-ulp) and break the bit-exact comparison.
    const dt: f32 = 1.0 / sample_rate;
    const omega: f32 = 2.0 * std.math.pi * dt;
    const dur = @max(duration, 1e-6);
    const inv_duration: f32 = 1.0 / dur;
    const slope: f32 = (cutoff_end - cutoff0) / dur;
    const gl: f32 = volume * @sqrt(0.5 * (1.0 - pan));
    const gr: f32 = volume * @sqrt(0.5 * (1.0 + pan));
    var t: f32 = 0.0;
    var seed: u32 = seed0;
    var lp: f32 = 0.0;
    var j: usize = 0;
    while (j < frames) : (j += 1) {
        var vl: f32 = 0.0;
        var vr: f32 = 0.0;
        t += dt;
        if (t < duration) {
            const k = t * inv_duration;
            const env = (1.0 - k) * (1.0 - k);
            const cutoff = @max(cutoff0 + slope * t, 30.0);
            const alpha = 1.0 - @exp(-omega * cutoff);
            seed = seed *% 1664525 +% 1013904223;
            const u: f32 = @floatFromInt(seed >> 9);
            const n = u * (2.0 / 8388608.0) - 1.0;
            lp += alpha * (n - lp);
            const s = lp * 1.6 * env;
            vl = s * gl;
            vr = s * gr;
        }
        const l = vl * master;
        const r = vr * master;
        out[j * 2] = l / (1.0 + @abs(l));
        out[j * 2 + 1] = r / (1.0 + @abs(r));
    }
}

/// First `play()` seed on a fresh engine: `next_seed` starts at
/// 0x12345678 and `play` bumps it before deriving the voice seed.
fn firstPlaySeed() u32 {
    var ns: u32 = 0x12345678;
    ns +%= 1;
    return ns *% 2654435761 +% 97;
}

test "noise static cutoff matches legacy reference exactly above and below the clamp" {
    // One row per clamp side: 5000 Hz exercises the normal path, 10 Hz the
    // 30 Hz floor. Both must match the per-sample legacy loop bit-exactly.
    const cutoffs = [_]f32{ 5000.0, 10.0 };
    for (cutoffs) |cutoff| {
        var eng = AudioEngine{};
        eng.play(.{ .kind = .noise_burst, .volume = 0.9, .duration = 0.05, .cutoff = cutoff, .cutoff_end = cutoff });
        var buf: [8820]f32 = @as([8820]f32, @splat(0.0)); // 4410 frames = 0.1 s; voice lasts 0.05 s
        eng.renderFrames(&buf);

        var want: [8820]f32 = undefined;
        legacyNoiseBurst(&want, 4410, 44100.0, cutoff, cutoff, 0.05, firstPlaySeed(), 0.9, 0.0, 0.8);
        try std.testing.expectEqualSlices(f32, &want, &buf);
        try std.testing.expect(channelEnergy(&buf, 0) > 0.01);
    }
}

test "noise ramped cutoff matches legacy reference and expires to silence" {
    var eng = AudioEngine{};
    eng.play(.{ .kind = .noise_burst, .volume = 0.9, .duration = 0.05, .cutoff = 6000.0, .cutoff_end = 300.0 });
    var buf: [8820]f32 = @as([8820]f32, @splat(0.0));
    eng.renderFrames(&buf);

    var want: [8820]f32 = undefined;
    legacyNoiseBurst(&want, 4410, 44100.0, 6000.0, 300.0, 0.05, firstPlaySeed(), 0.9, 0.0, 0.8);
    try std.testing.expectEqualSlices(f32, &want, &buf);

    // Past the duration the voice is done: exact silence and slot released.
    for (buf[4410..]) |s| try std.testing.expectEqual(@as(f32, 0.0), s);
    try std.testing.expect(!eng.voices[0].isActive());
}

test "noise split-buffer rendering matches single render" {
    const opts = AudioEngine.PlayOptions{
        .kind = .noise_burst,
        .volume = 0.9,
        .duration = 0.08,
        .cutoff = 2000.0,
        .cutoff_end = 2000.0,
    };
    var e1 = AudioEngine{};
    e1.play(opts);
    var full: [8820]f32 = @as([8820]f32, @splat(0.0));
    e1.renderFrames(&full);

    // Split at 2000 frames (not a 64-frame chunk multiple) to stress
    // voice-state carry (t/seed/LP) across render calls and chunk edges.
    var e2 = AudioEngine{};
    e2.play(opts);
    var part1: [4000]f32 = @as([4000]f32, @splat(0.0));
    e2.renderFrames(&part1);
    var part2: [4820]f32 = @as([4820]f32, @splat(0.0));
    e2.renderFrames(&part2);

    try std.testing.expectEqualSlices(f32, full[0..4000], &part1);
    try std.testing.expectEqualSlices(f32, full[4000..], &part2);
}

test "AudioEngine sanitizes non-finite voice parameters" {
    var eng = AudioEngine{};
    eng.play(.{ .kind = .blip, .volume = std.math.nan(f32), .duration = 0.2 });
    eng.play(.{ .kind = .blip, .volume = 0.5, .duration = std.math.inf(f32) });
    eng.play(.{ .kind = .blip, .volume = 0.5, .duration = 0.2, .freq = std.math.nan(f32) });
    eng.playExplosion(Vec3.zero, std.math.inf(f32));

    var buf: [256]f32 = @as([256]f32, @splat(0.0));
    eng.renderFrames(&buf);
    for (buf) |s| try std.testing.expect(std.math.isFinite(s));

    for (&eng.voices) |*v| {
        if (v.isActive()) {
            try std.testing.expect(std.math.isFinite(v.duration));
            try std.testing.expect(v.duration <= 2.0);
        }
    }
}

test "AudioEngine renderFrames ignores odd and tiny buffers" {
    var eng = AudioEngine{};
    eng.play(.{ .kind = .blip, .duration = 0.2 });
    var odd: [17]f32 = @as([17]f32, @splat(0.0));
    eng.renderFrames(&odd);
    var one: [1]f32 = @as([1]f32, @splat(0.0));
    eng.renderFrames(&one);
}

test "AudioStream seek rejects non-finite input and clamps the frame" {
    const alloc = std.testing.allocator;
    const ogg_bytes = @embedFile("fixtures/tone_440_880.ogg");
    var eng = AudioEngine{};
    const s = try eng.playSoundFromMemory(alloc, ogg_bytes, .ogg, .{ .buffer_frames = 4096 });
    s.seekToSeconds(std.math.inf(f32));
    s.seekToSeconds(-1.0);
    s.seekToFrame(std.math.maxInt(usize));
    try std.testing.expect(std.math.isFinite(s.getPositionSeconds()));
    try std.testing.expect(s.getPositionSeconds() <= s.getDurationSeconds() + 0.01);
    eng.destroyStream(s);
}

test "AudioStream destroy defers free until the audio thread acks" {
    const alloc = std.testing.allocator;
    const ogg_bytes = @embedFile("fixtures/tone_440_880.ogg");

    var eng = AudioEngine{};
    // Pretend the backend is live without spawning a real audio thread: the
    // retire handshake is exactly what the callback would drive.
    eng.started = true;
    defer eng.started = false;

    const s = try eng.playSoundFromMemory(alloc, ogg_bytes, .ogg, .{ .buffer_frames = 4096 });
    eng.destroyStream(s);

    // Slot is cleared synchronously...
    var still_slotted = false;
    for (0..AudioEngine.max_streams) |i| {
        if (streams_mod.getSlot(&eng, i) == s) still_slotted = true;
    }
    try std.testing.expect(!still_slotted);

    // ...but the object is not freed until the audio callback acknowledges
    // (the testing allocator fails the test if either step is skipped).
    var buf: [64]f32 = @as([64]f32, @splat(0.0));
    eng.renderFrames(&buf);
    eng.updateStreams(0.0);
}

test "AudioEngine shutdown frees streams registered before start" {
    const alloc = std.testing.allocator;
    const ogg_bytes = @embedFile("fixtures/tone_440_880.ogg");

    var eng = AudioEngine{};
    _ = try eng.playSoundFromMemory(alloc, ogg_bytes, .ogg, .{ .buffer_frames = 4096 });
    var found = false;
    for (0..AudioEngine.max_streams) |i| {
        if (streams_mod.getSlot(&eng, i) != null) found = true;
    }
    try std.testing.expect(found);

    // Never started, but shutdown must still release the decoder/ring.
    eng.shutdown();
    for (0..AudioEngine.max_streams) |i| {
        try std.testing.expect(streams_mod.getSlot(&eng, i) == null);
    }
}
