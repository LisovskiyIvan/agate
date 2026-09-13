const std = @import("std");
const builtin = @import("builtin");
const decode = @import("decode.zig");

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

    /// Decodes a whole MP3 file from memory via the vendored dr_mp3 decoder
    /// (src/agate/c/dr_mp3.h). Output follows the WAV rules: interleaved
    /// stereo f32 (mono duplicated) at the clip's own rate; playback
    /// resamples via `AudioEngine.playClip`. MP3 is frame-based, so a
    /// truncated stream yields the frames it contains; a stream with no
    /// decodable frames is error.InvalidMp3.
    pub fn fromMp3Memory(allocator: std.mem.Allocator, bytes: []const u8) decode.DecodeError!AudioClip {
        const d = try decode.decodeMp3Memory(allocator, bytes);
        return .{ .samples = d.samples, .sample_rate = d.sample_rate, .frames = d.frames };
    }

    /// Streaming MP3 file decode: dr_mp3 reads the file incrementally, so
    /// peak memory is the f32 output (the compressed file is never fully
    /// buffered). Filesystem errors surface before decoding; content errors
    /// match `fromMp3Memory`.
    pub fn fromMp3File(allocator: std.mem.Allocator, path: []const u8) !AudioClip {
        const d = try decode.decodeMp3File(allocator, path);
        return .{ .samples = d.samples, .sample_rate = d.sample_rate, .frames = d.frames };
    }

    /// Decodes an Ogg Vorbis stream from memory via the vendored stb_vorbis
    /// decoder (src/agate/c/stb_vorbis.c). Output follows the WAV rules; a
    /// valid stream with more than 2 channels is error.UnsupportedOggFormat
    /// (mirroring UnsupportedWavFormat), anything malformed is
    /// error.InvalidOgg.
    pub fn fromOggMemory(allocator: std.mem.Allocator, bytes: []const u8) decode.DecodeError!AudioClip {
        const d = try decode.decodeOggMemory(allocator, bytes);
        return .{ .samples = d.samples, .sample_rate = d.sample_rate, .frames = d.frames };
    }

    /// Streaming Ogg Vorbis file decode; same semantics as `fromOggMemory`
    /// with incremental input reads.
    pub fn fromOggFile(allocator: std.mem.Allocator, path: []const u8) !AudioClip {
        const d = try decode.decodeOggFile(allocator, path);
        return .{ .samples = d.samples, .sample_rate = d.sample_rate, .frames = d.frames };
    }

    /// Streaming file decode: parses the RIFF header and chunk table with
    /// positional reads, then decodes the PCM payload in ~64KB windows
    /// straight into the output buffer. Peak memory is the f32 output plus
    /// one chunk — the file itself is never fully buffered, unlike the old
    /// `readToEndAlloc` path. Truncated or malformed files report the same
    /// `InvalidWav` / `UnsupportedWavFormat` errors as `fromWavMemory`.
    /// NOTE: the `(allocator, path)` signature is public API and stays as
    /// is; the single-threaded global Io is used internally for that reason.
    pub fn fromWavFile(allocator: std.mem.Allocator, path: []const u8) !AudioClip {
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);

        const file_len = try file.length(io);
        if (file_len < 12) return error.InvalidWav;
        var header: [12]u8 = undefined;
        if (try file.readPositionalAll(io, &header, 0) < 12) return error.InvalidWav;
        if (!std.mem.eql(u8, header[0..4], "RIFF")) return error.InvalidWav;
        if (!std.mem.eql(u8, header[8..12], "WAVE")) return error.InvalidWav;

        // Chunk scan with positional reads only. Unknown chunks are skipped
        // (with their pad byte); the first data chunk wins and the last fmt
        // chunk wins, exactly like `fromWavMemory` below.
        var fields: ?WavFmtFields = null;
        var data_off: u64 = 0;
        var data_size: u64 = 0;
        var have_data = false;
        var off: u64 = 12;
        while (true) {
            if (off + 8 > file_len) break; // done, or trailing bytes hold no header
            var chunk_hdr: [8]u8 = undefined;
            if (try file.readPositionalAll(io, &chunk_hdr, off) < 8) return error.InvalidWav;
            const size: u64 = std.mem.readInt(u32, chunk_hdr[4..8], .little);
            const start = off + 8;
            const end = start + size;
            if (end < start or end > file_len) return error.InvalidWav; // truncated chunk
            if (std.mem.eql(u8, chunk_hdr[0..4], "fmt ")) {
                if (size < 16) return error.InvalidWav;
                var fbuf: [16]u8 = undefined;
                if (try file.readPositionalAll(io, &fbuf, start) < 16) return error.InvalidWav;
                fields = .{
                    .audio_format = std.mem.readInt(u16, fbuf[0..2], .little),
                    .channels = std.mem.readInt(u16, fbuf[2..4], .little),
                    .sample_rate = std.mem.readInt(u32, fbuf[4..8], .little),
                    .bits = std.mem.readInt(u16, fbuf[14..16], .little),
                };
            } else if (std.mem.eql(u8, chunk_hdr[0..4], "data")) {
                if (!have_data) {
                    data_off = start;
                    data_size = size;
                    have_data = true;
                }
            }
            off = end + (size & 1);
        }

        const f = fields orelse return error.InvalidWav;
        if (!have_data) return error.InvalidWav;
        const spec = try wavSpecFromFields(f);
        // Preserve the historical 256MB whole-file cap as a payload cap so
        // a corrupt size field cannot drive a wild allocation.
        if (data_size > max_wav_data_bytes) return error.StreamTooLong;
        if (data_size < spec.block_align) return error.InvalidWav;
        const frames: usize = @intCast(data_size / @as(u64, spec.block_align));
        if (frames == 0) return error.InvalidWav;

        const out = try allocator.alloc(f32, frames * 2);
        errdefer allocator.free(out);

        const tmp = try allocator.alloc(u8, wav_io_chunk_bytes);
        defer allocator.free(tmp);
        // Whole frames per iteration; block_align <= 8, so at least one
        // frame always fits into the 64KB window.
        const frames_per_iter = tmp.len / spec.block_align;
        var done: usize = 0;
        var done_bytes: u64 = 0;
        while (done < frames) {
            const want_frames = @min(frames - done, frames_per_iter);
            const want_bytes = want_frames * spec.block_align;
            if (try file.readPositionalAll(io, tmp[0..want_bytes], data_off + done_bytes) < want_bytes)
                return error.InvalidWav; // file shrank mid-decode
            decodePcmFrames(out[done * 2 ..][0 .. want_frames * 2], tmp[0..want_bytes], spec);
            done += want_frames;
            done_bytes += @as(u64, @intCast(want_bytes));
        }
        return .{
            .samples = out,
            .sample_rate = @floatFromInt(spec.sample_rate),
            .frames = frames,
        };
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

        var fmt: ?WavFmtFields = null;
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
        const spec = try wavSpecFromFields(f);
        if (d.len < spec.block_align) return error.InvalidWav;
        const frames = d.len / spec.block_align;
        if (frames == 0) return error.InvalidWav;

        const out = try allocator.alloc(f32, frames * 2);
        errdefer allocator.free(out);
        decodePcmFrames(out, d[0 .. frames * spec.block_align], spec);
        return .{
            .samples = out,
            .sample_rate = @floatFromInt(spec.sample_rate),
            .frames = frames,
        };
    }

    /// PCM window for the streaming `fromWavFile` decode: whole frames only.
    const wav_io_chunk_bytes: usize = 64 * 1024;
    /// Payload cap preserving the historical 256MB whole-file limit.
    const max_wav_data_bytes: u64 = 256 * 1024 * 1024;

    /// Raw format fields as stored in the `fmt ` chunk, before validation.
    const WavFmtFields = struct {
        audio_format: u16,
        channels: u16,
        sample_rate: u32,
        bits: u16,
    };

    /// Validated decode parameters shared by the memory and streaming paths.
    /// Error mapping matches the historical `fromWavMemory` behavior:
    /// malformed header values are `InvalidWav`, valid WAVs we cannot play
    /// are `UnsupportedWavFormat`.
    const WavSpec = struct {
        channels: usize, // 1 or 2
        sample_rate: u32, // != 0
        bits: u16, // 8/16/24/32, consistent with is_float
        is_float: bool, // IEEE float32 (format tag 3)
        bytes_per_sample: usize,
        block_align: usize, // channels * bytes_per_sample
    };

    fn wavSpecFromFields(f: WavFmtFields) !WavSpec {
        if (f.sample_rate == 0) return error.InvalidWav;
        if (f.channels != 1 and f.channels != 2) return error.UnsupportedWavFormat;
        const is_pcm = f.audio_format == 1 and (f.bits == 8 or f.bits == 16 or f.bits == 24 or f.bits == 32);
        const is_f32 = f.audio_format == 3 and f.bits == 32;
        if (!is_pcm and !is_f32) return error.UnsupportedWavFormat;
        const bytes_per_sample: usize = f.bits / 8;
        return .{
            .channels = f.channels,
            .sample_rate = f.sample_rate,
            .bits = f.bits,
            .is_float = is_f32,
            .bytes_per_sample = bytes_per_sample,
            .block_align = @as(usize, f.channels) * bytes_per_sample,
        };
    }

    /// Single dispatch for a PCM run: no per-sample branches. `out` holds
    /// exactly `frames * 2` f32s, `src` at least `frames * block_align`
    /// bytes. Used once per whole buffer by `fromWavMemory` and once per
    /// 64KB chunk by `fromWavFile`.
    fn decodePcmFrames(out: []f32, src: []const u8, spec: WavSpec) void {
        if (out.len == 0) return;
        if (spec.is_float) {
            decodeF32(out, src, spec);
        } else switch (spec.bits) {
            8 => decodeU8(out, src, spec),
            16 => decodeI16(out, src, spec),
            24 => decodeI24(out, src, spec),
            else => decodeI32(out, src, spec),
        }
    }

    /// 8-bit unsigned PCM: value 128 is silence. Byte loads need no
    /// alignment and byte order is trivial, so this path runs on any host
    /// endianness with vector int->float conversion and no branches.
    fn decodeU8(out: []f32, src: []const u8, spec: WavSpec) void {
        const V = 16;
        const offset: @Vector(V, f32) = @splat(128.0);
        const scale: @Vector(V, f32) = @splat(1.0 / 128.0);
        if (spec.channels == 2) {
            const n = @min(out.len, src.len);
            var i: usize = 0;
            while (i + V <= n) : (i += V) {
                const vb: @Vector(V, u8) = src[i..][0..V].*;
                const vf: @Vector(V, f32) = @floatFromInt(vb);
                out[i..][0..V].* = (vf - offset) * scale;
            }
            while (i < n) : (i += 1) out[i] = u8ToF32(src[i]);
        } else {
            const frames = @min(out.len / 2, src.len);
            var i: usize = 0;
            while (i + V <= frames) : (i += V) {
                const vb: @Vector(V, u8) = src[i..][0..V].*;
                const vf: @Vector(V, f32) = @floatFromInt(vb);
                const vs = (vf - offset) * scale;
                inline for (0..V) |k| {
                    out[(i + k) * 2] = vs[k];
                    out[(i + k) * 2 + 1] = vs[k];
                }
            }
            while (i < frames) : (i += 1) {
                const v = u8ToF32(src[i]);
                out[i * 2] = v;
                out[i * 2 + 1] = v;
            }
        }
    }

    /// 16-bit signed PCM, little-endian. Typed loads + vector conversion on
    /// LE hosts when the source is 2-byte aligned; otherwise the scalar
    /// little-endian fallback (also used on big-endian hosts).
    fn decodeI16(out: []f32, src: []const u8, spec: WavSpec) void {
        const frames = @min(out.len / 2, src.len / spec.block_align);
        if (builtin.cpu.arch.endian() == .little and @intFromPtr(src.ptr) % 2 == 0) {
            const raw: []align(2) const u8 = @alignCast(src[0 .. frames * spec.block_align]);
            const pcm = std.mem.bytesAsSlice(i16, raw);
            const scale: f32 = 1.0 / 32768.0;
            const V = 8;
            if (spec.channels == 2) {
                const total = frames * 2;
                var i: usize = 0;
                while (i + V <= total) : (i += V) {
                    const vi: @Vector(V, i16) = pcm[i..][0..V].*;
                    const vf: @Vector(V, f32) = @floatFromInt(vi);
                    out[i..][0..V].* = vf * @as(@Vector(V, f32), @splat(scale));
                }
                while (i < total) : (i += 1) {
                    out[i] = @as(f32, @floatFromInt(pcm[i])) * scale;
                }
            } else {
                var i: usize = 0;
                while (i + V <= frames) : (i += V) {
                    const vi: @Vector(V, i16) = pcm[i..][0..V].*;
                    const vf: @Vector(V, f32) = @floatFromInt(vi);
                    const vs = vf * @as(@Vector(V, f32), @splat(scale));
                    inline for (0..V) |k| {
                        out[(i + k) * 2] = vs[k];
                        out[(i + k) * 2 + 1] = vs[k];
                    }
                }
                while (i < frames) : (i += 1) {
                    const v = @as(f32, @floatFromInt(pcm[i])) * scale;
                    out[i * 2] = v;
                    out[i * 2 + 1] = v;
                }
            }
            return;
        }
        decodeScalarFrames(out, src, spec);
    }

    /// 24-bit signed PCM, little-endian. Samples are 3 bytes with no natural
    /// alignment, so groups of 8 frames are assembled with explicit
    /// little-endian loads (correct on any host endianness, with sign
    /// extension matching the old scalar code) and converted with vector
    /// int->float; no branches on sample values.
    fn decodeI24(out: []f32, src: []const u8, spec: WavSpec) void {
        const V = 8;
        const scale: f32 = 1.0 / 8388608.0;
        const frames = @min(out.len / 2, src.len / spec.block_align);
        if (spec.channels == 2) {
            var i: usize = 0;
            while (i + V <= frames) : (i += V) {
                var l: [V]i32 = undefined;
                var r: [V]i32 = undefined;
                for (0..V) |k| {
                    const b = (i + k) * 6;
                    l[k] = i24ToI32(src[b], src[b + 1], src[b + 2]);
                    r[k] = i24ToI32(src[b + 3], src[b + 4], src[b + 5]);
                }
                const lv: @Vector(V, i32) = l;
                const rv: @Vector(V, i32) = r;
                const ls = @as(@Vector(V, f32), @floatFromInt(lv)) * @as(@Vector(V, f32), @splat(scale));
                const rs = @as(@Vector(V, f32), @floatFromInt(rv)) * @as(@Vector(V, f32), @splat(scale));
                inline for (0..V) |k| {
                    out[(i + k) * 2] = ls[k];
                    out[(i + k) * 2 + 1] = rs[k];
                }
            }
            while (i < frames) : (i += 1) {
                out[i * 2] = @as(f32, @floatFromInt(i24ToI32(src[i * 6], src[i * 6 + 1], src[i * 6 + 2]))) * scale;
                out[i * 2 + 1] = @as(f32, @floatFromInt(i24ToI32(src[i * 6 + 3], src[i * 6 + 4], src[i * 6 + 5]))) * scale;
            }
        } else {
            var i: usize = 0;
            while (i + V <= frames) : (i += V) {
                var m: [V]i32 = undefined;
                for (0..V) |k| {
                    const b = (i + k) * 3;
                    m[k] = i24ToI32(src[b], src[b + 1], src[b + 2]);
                }
                const mv: @Vector(V, i32) = m;
                const ms = @as(@Vector(V, f32), @floatFromInt(mv)) * @as(@Vector(V, f32), @splat(scale));
                inline for (0..V) |k| {
                    out[(i + k) * 2] = ms[k];
                    out[(i + k) * 2 + 1] = ms[k];
                }
            }
            while (i < frames) : (i += 1) {
                const v = @as(f32, @floatFromInt(i24ToI32(src[i * 3], src[i * 3 + 1], src[i * 3 + 2]))) * scale;
                out[i * 2] = v;
                out[i * 2 + 1] = v;
            }
        }
    }

    /// 32-bit signed PCM, little-endian. Typed loads + vector conversion on
    /// LE hosts when 4-byte aligned; scalar little-endian fallback otherwise.
    fn decodeI32(out: []f32, src: []const u8, spec: WavSpec) void {
        const frames = @min(out.len / 2, src.len / spec.block_align);
        if (builtin.cpu.arch.endian() == .little and @intFromPtr(src.ptr) % 4 == 0) {
            const raw: []align(4) const u8 = @alignCast(src[0 .. frames * spec.block_align]);
            const pcm = std.mem.bytesAsSlice(i32, raw);
            const scale: f32 = 1.0 / 2147483648.0;
            const V = 8;
            if (spec.channels == 2) {
                const total = frames * 2;
                var i: usize = 0;
                while (i + V <= total) : (i += V) {
                    const vi: @Vector(V, i32) = pcm[i..][0..V].*;
                    const vf: @Vector(V, f32) = @floatFromInt(vi);
                    out[i..][0..V].* = vf * @as(@Vector(V, f32), @splat(scale));
                }
                while (i < total) : (i += 1) {
                    out[i] = @as(f32, @floatFromInt(pcm[i])) * scale;
                }
            } else {
                var i: usize = 0;
                while (i + V <= frames) : (i += V) {
                    const vi: @Vector(V, i32) = pcm[i..][0..V].*;
                    const vf: @Vector(V, f32) = @floatFromInt(vi);
                    const vs = vf * @as(@Vector(V, f32), @splat(scale));
                    inline for (0..V) |k| {
                        out[(i + k) * 2] = vs[k];
                        out[(i + k) * 2 + 1] = vs[k];
                    }
                }
                while (i < frames) : (i += 1) {
                    const v = @as(f32, @floatFromInt(pcm[i])) * scale;
                    out[i * 2] = v;
                    out[i * 2 + 1] = v;
                }
            }
            return;
        }
        decodeScalarFrames(out, src, spec);
    }

    /// IEEE float32: stereo already matches the output layout (memcpy); mono
    /// duplicates with vectorized loads. Unaligned mono falls back to scalar
    /// little-endian loads (also used on big-endian hosts).
    fn decodeF32(out: []f32, src: []const u8, spec: WavSpec) void {
        const frames = @min(out.len / 2, src.len / spec.block_align);
        if (spec.channels == 2) {
            @memcpy(std.mem.sliceAsBytes(out[0 .. frames * 2]), src[0 .. frames * 8]);
            return;
        }
        if (builtin.cpu.arch.endian() == .little and @intFromPtr(src.ptr) % 4 == 0) {
            const raw: []align(4) const u8 = @alignCast(src[0 .. frames * 4]);
            const pcm = std.mem.bytesAsSlice(f32, raw);
            const V = 8;
            var i: usize = 0;
            while (i + V <= frames) : (i += V) {
                const v: @Vector(V, f32) = pcm[i..][0..V].*;
                inline for (0..V) |k| {
                    out[(i + k) * 2] = v[k];
                    out[(i + k) * 2 + 1] = v[k];
                }
            }
            while (i < frames) : (i += 1) {
                out[i * 2] = pcm[i];
                out[i * 2 + 1] = pcm[i];
            }
            return;
        }
        decodeScalarFrames(out, src, spec);
    }

    /// Scalar fallback for unaligned or big-endian sources. Bit-exact
    /// reference semantics for every format; the vector paths above must
    /// match it (all scale factors are exact powers of two).
    fn decodeScalarFrames(out: []f32, src: []const u8, spec: WavSpec) void {
        const frames = @min(out.len / 2, src.len / spec.block_align);
        const channels = spec.channels;
        const bps = spec.bytes_per_sample;
        var i: usize = 0;
        while (i < frames) : (i += 1) {
            var c: usize = 0;
            var vals: [2]f32 = .{ 0.0, 0.0 };
            while (c < channels) : (c += 1) {
                const b = src[i * spec.block_align + c * bps ..][0..bps];
                var v: f32 = 0.0;
                if (spec.is_float) {
                    v = @bitCast(std.mem.readInt(u32, b[0..4], .little));
                } else switch (spec.bits) {
                    8 => v = u8ToF32(b[0]),
                    16 => v = @as(f32, @floatFromInt(std.mem.readInt(i16, b[0..2], .little))) / 32768.0,
                    24 => v = @as(f32, @floatFromInt(i24ToI32(b[0], b[1], b[2]))) / 8388608.0,
                    else => v = @as(f32, @floatFromInt(std.mem.readInt(i32, b[0..4], .little))) / 2147483648.0,
                }
                vals[c] = v;
            }
            if (channels == 1) vals[1] = vals[0];
            out[i * 2] = vals[0];
            out[i * 2 + 1] = vals[1];
        }
    }

    inline fn u8ToF32(b: u8) f32 {
        return (@as(f32, @floatFromInt(b)) - 128.0) / 128.0;
    }

    inline fn i24ToI32(b0: u8, b1: u8, b2: u8) i32 {
        const u: u32 = @as(u32, b0) | (@as(u32, b1) << 8) | (@as(u32, b2) << 16);
        return @as(i32, @bitCast(u << 8)) >> 8;
    }
};
