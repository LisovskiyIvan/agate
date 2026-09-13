const std = @import("std");

/// Declarations-only C imports for the vendored audio decoders
/// (src/agate/c/dr_mp3.h, src/agate/c/stb_vorbis.c; see src/agate/c/LICENSES.md).
/// This lives in its own `@cImport`, separate from the shared `c.zig`, so the
/// audio module depends only on these two headers. The implementations are
/// compiled exactly once through `src/agate/c/c_impl.c`; STB_VORBIS_HEADER_ONLY
/// keeps stb_vorbis declarations-only here because the file is both header and
/// implementation.
pub const c = @cImport({
    @cInclude("dr_mp3.h");
    @cDefine("STB_VORBIS_HEADER_ONLY", "1");
    @cInclude("stb_vorbis.c");
    @cUndef("STB_VORBIS_HEADER_ONLY");
});

/// Decoded audio in the same layout `AudioClip` uses: interleaved stereo f32
/// at the source's own sample rate (mono is duplicated to both channels).
pub const DecodedAudio = struct {
    samples: []f32,
    sample_rate: f32,
    frames: usize,
};

pub const DecodeError = error{
    OutOfMemory,
    /// Malformed, truncated or unrecognizable MP3 stream (nothing decodable).
    InvalidMp3,
    /// Malformed, truncated or unrecognizable Ogg Vorbis stream.
    InvalidOgg,
    /// Valid stream we cannot play: more than 2 channels.
    UnsupportedOggFormat,
    /// Decoded payload exceeds the safety cap (`max_decoded_frames`).
    StreamTooLong,
};

/// dr_mp3 reads whole frames (1152 samples each); 4096 frames per read keeps
/// the scratch buffer at 32KB while amortizing the call overhead.
const mp3_chunk_frames: usize = 4096;

/// Safety cap mirroring the WAV payload cap idea: bounds how much a hostile
/// or corrupt stream can make us allocate (compressed audio expands ~10:1,
/// so this does not correspond to a file-size limit).
const max_decoded_frames: usize = 64 * 1024 * 1024;

/// Decodes a whole MP3 from memory. Compressed input is not buffered beyond
/// the caller's slice; peak memory is the f32 output plus a 32KB scratch.
pub fn decodeMp3Memory(allocator: std.mem.Allocator, bytes: []const u8) DecodeError!DecodedAudio {
    if (bytes.len == 0) return error.InvalidMp3;

    var mp3 = std.mem.zeroes(c.drmp3);
    if (c.drmp3_init_memory(&mp3, @ptrCast(bytes.ptr), bytes.len, null) == 0) {
        return error.InvalidMp3;
    }
    defer c.drmp3_uninit(&mp3);

    return decodeMp3Stream(allocator, &mp3);
}

/// Decodes an MP3 from a file. dr_mp3 reads the file incrementally through its
/// own stdio, so peak memory is the f32 output only — the compressed file is
/// never fully buffered (mirrors `AudioClip.fromWavFile` semantics). Filesystem
/// errors (e.g. FileNotFound) surface from the upfront open check; dr_mp3
/// would otherwise flatten them into "not an MP3".
pub fn decodeMp3File(allocator: std.mem.Allocator, path: []const u8) !DecodedAudio {
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    file.close(io);

    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    var mp3 = std.mem.zeroes(c.drmp3);
    if (c.drmp3_init_file(&mp3, path_z.ptr, null) == 0) {
        return error.InvalidMp3;
    }
    defer c.drmp3_uninit(&mp3);

    return decodeMp3Stream(allocator, &mp3);
}

/// Shared read loop for an initialized drmp3 decoder. MP3 is frame-based, so
/// a truncated stream decodes the frames it contains instead of failing; a
/// stream that yields no frames at all is reported as InvalidMp3.
fn decodeMp3Stream(allocator: std.mem.Allocator, mp3: *c.drmp3) DecodeError!DecodedAudio {
    const channels: usize = mp3.channels;
    const rate: u32 = mp3.sampleRate;
    if (channels == 0 or channels > 2 or rate == 0) return error.InvalidMp3;

    const tmp = try allocator.alloc(f32, mp3_chunk_frames * 2);
    defer allocator.free(tmp);

    var samples: std.ArrayListUnmanaged(f32) = .empty;
    errdefer samples.deinit(allocator);
    var frames: usize = 0;
    while (true) {
        const got: usize = @intCast(c.drmp3_read_pcm_frames_f32(mp3, mp3_chunk_frames, tmp.ptr));
        if (got == 0) break;
        frames += got;
        if (frames > max_decoded_frames) return error.StreamTooLong;
        try appendStereo(allocator, &samples, tmp[0 .. got * channels], channels);
    }
    if (frames == 0) return error.InvalidMp3;

    return .{
        .samples = try samples.toOwnedSlice(allocator),
        .sample_rate = @floatFromInt(rate),
        .frames = frames,
    };
}

/// Decodes a whole Ogg Vorbis stream from memory. Unlike MP3, vorbis pages
/// carry checksums, so truncation is detected and reported instead of decoded
/// through.
pub fn decodeOggMemory(allocator: std.mem.Allocator, bytes: []const u8) DecodeError!DecodedAudio {
    if (bytes.len == 0) return error.InvalidOgg;
    // stb_vorbis takes an `int` length; a buffer that large cannot be decoded
    // within the frame cap anyway.
    if (bytes.len > std.math.maxInt(c_int)) return error.StreamTooLong;

    var err: c_int = 0;
    const v = c.stb_vorbis_open_memory(@ptrCast(bytes.ptr), @intCast(bytes.len), &err, null) orelse return error.InvalidOgg;
    defer c.stb_vorbis_close(v);

    return decodeOggStream(allocator, v, err);
}

/// Decodes an Ogg Vorbis file. stb_vorbis pulls data from its own stdio handle
/// incrementally; peak memory is the f32 output only. Filesystem errors
/// surface from the upfront open check (same as `decodeMp3File`).
pub fn decodeOggFile(allocator: std.mem.Allocator, path: []const u8) !DecodedAudio {
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    file.close(io);

    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    var err: c_int = 0;
    const v = c.stb_vorbis_open_filename(path_z.ptr, &err, null) orelse return error.InvalidOgg;
    defer c.stb_vorbis_close(v);

    return decodeOggStream(allocator, v, err);
}

/// Shared read loop for an opened stb_vorbis decoder. `stb_vorbis_get_frame_float`
/// decodes one frame at a time into the decoder's internal channel planes, so
/// we interleave into the growing output per frame.
fn decodeOggStream(allocator: std.mem.Allocator, v: *c.stb_vorbis, open_err: c_int) DecodeError!DecodedAudio {
    // open_memory sets the error out-param on failure paths even when it
    // returns a decoder for partial headers, so both checks are needed.
    if (open_err != c.VORBIS__no_error) return error.InvalidOgg;

    const info = c.stb_vorbis_get_info(v);
    const channels: usize = @intCast(info.channels);
    const rate: u32 = info.sample_rate;
    if (channels == 0) return error.InvalidOgg;
    if (channels > 2) return error.UnsupportedOggFormat;
    if (rate == 0) return error.InvalidOgg;

    var outs: [*c][*c]f32 = undefined;
    var samples: std.ArrayListUnmanaged(f32) = .empty;
    errdefer samples.deinit(allocator);
    var frames: usize = 0;
    while (true) {
        var nch: c_int = @intCast(channels);
        const got: usize = @intCast(c.stb_vorbis_get_frame_float(v, &nch, &outs));
        if (got == 0) break;
        frames += got;
        if (frames > max_decoded_frames) return error.StreamTooLong;

        const dst = try samples.addManyAsSlice(allocator, got * 2);
        if (channels == 1) {
            for (0..got) |i| {
                const x = outs[0][i];
                dst[i * 2] = x;
                dst[i * 2 + 1] = x;
            }
        } else {
            for (0..got) |i| {
                dst[i * 2] = outs[0][i];
                dst[i * 2 + 1] = outs[1][i];
            }
        }
    }
    if (frames == 0) return error.InvalidOgg;

    return .{
        .samples = try samples.toOwnedSlice(allocator),
        .sample_rate = @floatFromInt(rate),
        .frames = frames,
    };
}

/// Grows the interleaved stereo output by one decoded chunk: mono duplicates
/// into both channels, stereo copies as-is (same rules as the WAV decoders).
fn appendStereo(
    allocator: std.mem.Allocator,
    list: *std.ArrayListUnmanaged(f32),
    interleaved: []const f32,
    channels: usize,
) std.mem.Allocator.Error!void {
    const frames = interleaved.len / channels;
    const dst = try list.addManyAsSlice(allocator, frames * 2);
    if (channels == 1) {
        for (0..frames) |i| {
            const v = interleaved[i];
            dst[i * 2] = v;
            dst[i * 2 + 1] = v;
        }
    } else {
        @memcpy(dst, interleaved[0 .. frames * 2]);
    }
}
