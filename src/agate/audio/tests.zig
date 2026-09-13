const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const audio = @import("../audio.zig");
const AudioEngine = audio.AudioEngine;
const AudioClip = audio.AudioClip;
const VoiceKind = audio.VoiceKind;
const Voice = audio.Voice;

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
// Fixture provenance: `tone_440_880.mp3` / `tone_440_880.ogg` (src/agate/audio/
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

test "AudioClip decodes MP3 from memory with expected duration, channels and RMS" {
    const alloc = std.testing.allocator;
    const bytes = @embedFile("fixtures/tone_440_880.mp3");
    var clip = try AudioClip.fromMp3Memory(alloc, bytes);
    defer clip.deinit(alloc);
    try expectStereoTone(&clip, 2646);
}

test "AudioClip decodes OGG Vorbis from memory with expected duration, channels and RMS" {
    const alloc = std.testing.allocator;
    const bytes = @embedFile("fixtures/tone_440_880.ogg");
    var clip = try AudioClip.fromOggMemory(alloc, bytes);
    defer clip.deinit(alloc);
    try expectStereoTone(&clip, 2646);
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

    if (cwd.writeFile(tio, .{ .sub_path = "agate_mp3_fixture_test.mp3", .data = mp3_bytes })) {
        defer cwd.deleteFile(tio, "agate_mp3_fixture_test.mp3") catch {};
        var mp3_file = try AudioClip.fromMp3File(alloc, "agate_mp3_fixture_test.mp3");
        defer mp3_file.deinit(alloc);
        try std.testing.expectEqual(mp3_mem.frames, mp3_file.frames);
        try std.testing.expectEqual(mp3_mem.sample_rate, mp3_file.sample_rate);
        try std.testing.expectEqualSlices(f32, mp3_mem.samples, mp3_file.samples);
    } else |_| {}

    if (cwd.writeFile(tio, .{ .sub_path = "agate_ogg_fixture_test.ogg", .data = ogg_bytes })) {
        defer cwd.deleteFile(tio, "agate_ogg_fixture_test.ogg") catch {};
        var ogg_file = try AudioClip.fromOggFile(alloc, "agate_ogg_fixture_test.ogg");
        defer ogg_file.deinit(alloc);
        try std.testing.expectEqual(ogg_mem.frames, ogg_file.frames);
        try std.testing.expectEqual(ogg_mem.sample_rate, ogg_file.sample_rate);
        try std.testing.expectEqualSlices(f32, ogg_mem.samples, ogg_file.samples);
    } else |_| {}
}

test "AudioClip plays a decoded MP3 clip through the mixer" {
    const alloc = std.testing.allocator;
    var clip = try AudioClip.fromMp3Memory(alloc, @embedFile("fixtures/tone_440_880.mp3"));
    defer clip.deinit(alloc);

    var eng = AudioEngine{};
    eng.playClip(&clip, .{});
    // Clip is ~0.06 s; render 0.15 s and expect energy only in the head.
    var buf: [13230]f32 = [_]f32{0.0} ** 13230;
    eng.renderFrames(&buf);
    try std.testing.expect(channelEnergy(&buf, 0) + channelEnergy(&buf, 1) > 10.0);
    try std.testing.expect(peakAbs(&buf, 5513, 6615) < 1e-6);
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
    try std.testing.expectError(error.InvalidMp3, AudioClip.fromMp3Memory(alloc, &[_]u8{0} ** 512));
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
    const tmp_name = "agate_audio_stream_test.wav";
    var from_file: ?AudioClip = null;
    if (cwd.writeFile(tio, .{ .sub_path = tmp_name, .data = wav })) {
        defer cwd.deleteFile(tio, tmp_name) catch {};
        from_file = try AudioClip.fromWavFile(alloc, tmp_name);

        // Truncated file must fail like the memory path.
        const trunc_name = "agate_audio_stream_test_trunc.wav";
        try cwd.writeFile(tio, .{ .sub_path = trunc_name, .data = wav[0 .. wav.len / 2] });
        defer cwd.deleteFile(tio, trunc_name) catch {};
        try std.testing.expectError(error.InvalidWav, AudioClip.fromWavFile(alloc, trunc_name));

        // Unsupported format must fail like the memory path.
        const bad_raw = try alloc.alloc(u8, 12);
        defer alloc.free(bad_raw);
        @memset(bad_raw, 0);
        const bad_wav = try buildWav(alloc, true, 1, 3, 44100, 16, bad_raw);
        defer alloc.free(bad_wav);
        const bad_name = "agate_audio_stream_test_bad.wav";
        try cwd.writeFile(tio, .{ .sub_path = bad_name, .data = bad_wav });
        defer cwd.deleteFile(tio, bad_name) catch {};
        try std.testing.expectError(error.UnsupportedWavFormat, AudioClip.fromWavFile(alloc, bad_name));

        // Data-before-fmt chunk order with an odd-size pad byte.
        const u8raw = [_]u8{ 0, 64, 128, 192, 255 };
        const u8wav = try buildWav(alloc, false, 1, 1, 22050, 8, &u8raw);
        defer alloc.free(u8wav);
        const u8name = "agate_audio_stream_test_u8.wav";
        try cwd.writeFile(tio, .{ .sub_path = u8name, .data = u8wav });
        defer cwd.deleteFile(tio, u8name) catch {};
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
