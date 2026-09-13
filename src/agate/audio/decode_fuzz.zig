//! Fuzz/robustness targets for the audio decoders (audio/decode.zig +
//! AudioClip.fromWavMemory): WAV (hand-rolled chunk parser), MP3 (vendored
//! dr_mp3) and Ogg Vorbis (vendored stb_vorbis) on arbitrary bytes.
//!
//! Invariant: any input yields an error or a DecodedAudio/AudioClip whose
//! samples the test frees — no panic, no UB, no leaks. Zig-side buffers are
//! std.testing.allocator-backed; the C decoders' internal allocations are the
//! vendors' responsibility (crash-safety is what we assert).
//! Run notes in build.zig.

const std = @import("std");
const decode = @import("decode.zig");
const AudioClip = @import("clip.zig").AudioClip;
const fzg = @import("../testing.zig");

const mp3_seed = @embedFile("fixtures/tone_440_880.mp3");
const ogg_seed = @embedFile("fixtures/tone_440_880.ogg");

/// Minimal valid 16-bit mono WAV: RIFF/WAVE + one 16-byte fmt chunk + a
/// 4-byte data chunk with two frames. Built at comptime.
const wav_seed: [48]u8 = blk: {
    var buf: [48]u8 = @splat(0);
    @memcpy(buf[0..4], "RIFF");
    std.mem.writeInt(u32, buf[4..8], 40, .little);
    @memcpy(buf[8..12], "WAVE");
    @memcpy(buf[12..16], "fmt ");
    std.mem.writeInt(u32, buf[16..20], 16, .little);
    std.mem.writeInt(u16, buf[20..22], 1, .little); // PCM
    std.mem.writeInt(u16, buf[22..24], 1, .little); // mono
    std.mem.writeInt(u32, buf[24..28], 8000, .little); // sample rate
    std.mem.writeInt(u32, buf[28..32], 16000, .little); // byte rate
    std.mem.writeInt(u16, buf[32..34], 2, .little); // block align
    std.mem.writeInt(u16, buf[34..36], 16, .little); // bits
    @memcpy(buf[36..40], "data");
    std.mem.writeInt(u32, buf[40..44], 4, .little);
    std.mem.writeInt(i16, buf[44..46], 1000, .little);
    std.mem.writeInt(i16, buf[46..48], -1000, .little);
    break :blk buf;
};

const wav_corpus = fzg.join(
    &[_][]const u8{&wav_seed},
    fzg.join(
        // Cuts through the RIFF header, fmt chunk and data chunk.
        fzg.truncations(&wav_seed, &.{ 1, 4, 12, 16, 20, 36, 40, 44, 47 }),
        fzg.join(
            fzg.flips(&wav_seed, &.{ 0, 4, 8, 12, 20, 22, 24, 34, 40 }),
            &[_][]const u8{
                "",
                "RIFF",
                "RIFF\xff\xff\xff\x7fWAVE",
                "RIFF\x00\x00\x00\x00WAVE",
                // fmt-last ordering (chunk loop must handle either order).
                "RIFF\x1c\x00\x00\x00WAVE" ++ "data\x02\x00\x00\x00\x00\x00" ++ "fmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00",
            },
        ),
    ),
);

fn testWavOne(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var generated: [8192]u8 = undefined;
    const input = fzg.fuzzInput(smith, &generated);

    var clip = AudioClip.fromWavMemory(std.testing.allocator, input) catch return;
    clip.deinit(std.testing.allocator);
}

test "fuzz: wav decode survives arbitrary bytes" {
    try std.testing.fuzz({}, testWavOne, .{ .corpus = wav_corpus });
}

const mp3_corpus = fzg.join(
    &[_][]const u8{mp3_seed},
    fzg.join(
        fzg.truncations(mp3_seed, &.{ 1, 4, 64, 128, 512, mp3_seed.len / 2, mp3_seed.len - 1 }),
        fzg.join(
            fzg.flips(mp3_seed, &.{ 0, 1, 4, 32, 128, 512 }),
            &[_][]const u8{ "", "\xff\xfb", "\xff\xff\xff\xff" },
        ),
    ),
);

fn testMp3One(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var generated: [16384]u8 = undefined;
    const input = fzg.fuzzInput(smith, &generated);

    const audio = decode.decodeMp3Memory(std.testing.allocator, input) catch return;
    std.testing.allocator.free(audio.samples);
}

test "fuzz: mp3 decode survives arbitrary bytes" {
    try std.testing.fuzz({}, testMp3One, .{ .corpus = mp3_corpus });
}

const ogg_corpus = fzg.join(
    &[_][]const u8{ogg_seed},
    fzg.join(
        fzg.truncations(ogg_seed, &.{ 1, 4, 27, 64, 256, ogg_seed.len / 2, ogg_seed.len - 1 }),
        fzg.join(
            // 0..4 = capture pattern, 4 = version, 28.. = segment table.
            fzg.flips(ogg_seed, &.{ 0, 1, 4, 28, 64, 256 }),
            &[_][]const u8{ "", "OggS", "OggS\x00" ++ "\x00" ** 60 },
        ),
    ),
);

fn testOggOne(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var generated: [16384]u8 = undefined;
    const input = fzg.fuzzInput(smith, &generated);

    const audio = decode.decodeOggMemory(std.testing.allocator, input) catch return;
    std.testing.allocator.free(audio.samples);
}

test "fuzz: ogg decode survives arbitrary bytes" {
    try std.testing.fuzz({}, testOggOne, .{ .corpus = ogg_corpus });
}
