const std = @import("std");
const dsp = @import("dsp.zig");
const BiquadFilter = dsp.BiquadFilter;
const ReverbProcessor = dsp.ReverbProcessor;

test "BiquadFilter lowpass attenuates high frequencies" {
    var f = BiquadFilter{};
    f.setParams(.lowpass, 500.0, 0.7071, 44100.0);

    // Test with high frequency tone (5000 Hz)
    var high_buf: [1000]f32 = undefined;
    for (0..500) |i| {
        const t = @as(f32, @floatFromInt(i)) / 44100.0;
        const val = @sin(2.0 * std.math.pi * 5000.0 * t);
        high_buf[i * 2] = val;
        high_buf[i * 2 + 1] = val;
    }
    f.processBuffer(&high_buf);

    var high_energy: f32 = 0.0;
    for (high_buf[600..1000]) |x| high_energy += x * x;

    // Reset and test with low frequency tone (100 Hz)
    f.resetState();
    f.setParams(.lowpass, 500.0, 0.7071, 44100.0);

    var low_buf: [1000]f32 = undefined;
    for (0..500) |i| {
        const t = @as(f32, @floatFromInt(i)) / 44100.0;
        const val = @sin(2.0 * std.math.pi * 100.0 * t);
        low_buf[i * 2] = val;
        low_buf[i * 2 + 1] = val;
    }
    f.processBuffer(&low_buf);

    var low_energy: f32 = 0.0;
    for (low_buf[600..1000]) |x| low_energy += x * x;

    // Low frequency should pass through with significantly higher energy than high frequency
    try std.testing.expect(low_energy > high_energy * 20.0);
}

test "BiquadFilter highpass attenuates low frequencies" {
    var f = BiquadFilter{};
    f.setParams(.highpass, 2000.0, 0.7071, 44100.0);

    // Test with low frequency tone (100 Hz)
    var low_buf: [1000]f32 = undefined;
    for (0..500) |i| {
        const t = @as(f32, @floatFromInt(i)) / 44100.0;
        const val = @sin(2.0 * std.math.pi * 100.0 * t);
        low_buf[i * 2] = val;
        low_buf[i * 2 + 1] = val;
    }
    f.processBuffer(&low_buf);

    var low_energy: f32 = 0.0;
    for (low_buf[600..1000]) |x| low_energy += x * x;

    // Test with high frequency tone (4000 Hz)
    f.resetState();
    f.setParams(.highpass, 2000.0, 0.7071, 44100.0);

    var high_buf: [1000]f32 = undefined;
    for (0..500) |i| {
        const t = @as(f32, @floatFromInt(i)) / 44100.0;
        const val = @sin(2.0 * std.math.pi * 4000.0 * t);
        high_buf[i * 2] = val;
        high_buf[i * 2 + 1] = val;
    }
    f.processBuffer(&high_buf);

    var high_energy: f32 = 0.0;
    for (high_buf[600..1000]) |x| high_energy += x * x;

    try std.testing.expect(high_energy > low_energy * 20.0);
}

test "BiquadFilter bandpass passes center and attenuates edges" {
    var f = BiquadFilter{};
    f.setParams(.bandpass, 1000.0, 2.0, 44100.0);

    // Center tone (1000 Hz)
    var mid_buf: [1000]f32 = undefined;
    for (0..500) |i| {
        const t = @as(f32, @floatFromInt(i)) / 44100.0;
        const val = @sin(2.0 * std.math.pi * 1000.0 * t);
        mid_buf[i * 2] = val;
        mid_buf[i * 2 + 1] = val;
    }
    f.processBuffer(&mid_buf);

    var mid_energy: f32 = 0.0;
    for (mid_buf[600..1000]) |x| mid_energy += x * x;

    // Edge tone (100 Hz)
    f.resetState();
    f.setParams(.bandpass, 1000.0, 2.0, 44100.0);

    var edge_buf: [1000]f32 = undefined;
    for (0..500) |i| {
        const t = @as(f32, @floatFromInt(i)) / 44100.0;
        const val = @sin(2.0 * std.math.pi * 100.0 * t);
        edge_buf[i * 2] = val;
        edge_buf[i * 2 + 1] = val;
    }
    f.processBuffer(&edge_buf);

    var edge_energy: f32 = 0.0;
    for (edge_buf[600..1000]) |x| edge_energy += x * x;

    try std.testing.expect(mid_energy > edge_energy * 10.0);
}

test "ReverbProcessor impulse produces decaying tail" {
    var rev = ReverbProcessor.init(44100.0);
    rev.active = true;
    rev.setConfig(.{
        .room_size = 0.8,
        .damping = 0.2,
        .wet = 1.0,
        .dry = 0.0,
    });

    // 1 sample impulse followed by zeroes
    var buf1: [256]f32 = [_]f32{0.0} ** 256;
    buf1[0] = 1.0;
    buf1[1] = 1.0;
    rev.processBuffer(&buf1);

    // After many samples, reverb tail should still be ringing
    var tail_buf: [4096]f32 = [_]f32{0.0} ** 4096;
    rev.processBuffer(&tail_buf);

    var tail_energy: f32 = 0.0;
    for (tail_buf) |x| tail_energy += x * x;

    try std.testing.expect(tail_energy > 0.001);

    // Clearing reverb should silence it completely
    rev.clear();
    var silent_buf: [256]f32 = [_]f32{0.0} ** 256;
    rev.processBuffer(&silent_buf);
    var silent_energy: f32 = 0.0;
    for (silent_buf) |x| silent_energy += x * x;
    try std.testing.expectEqual(@as(f32, 0.0), silent_energy);
}
