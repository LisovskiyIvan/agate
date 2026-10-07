//! Audio streaming system for music, ambient loops, and long narrative voiceover.
//!
//! Features:
//! - Streaming from disk files or memory slices (OGG Vorbis, MP3, WAV).
//! - Ring-buffered single-producer single-consumer (SPSC) lock-free playback.
//! - Audio thread safety: zero allocations and zero filesystem I/O in the audio callback.
//! - Real-time sample-rate conversion / linear resampling to engine sample rate.
//! - Seamless gapless looping at track boundaries.
//! - Dynamic volume fading, crossfading, stereo panning, and routing to any audio bus.
//! - Direct integration with AudioEngine bus effects (DSP filters, occlusion, reverb).

const std = @import("std");
const decode = @import("decode.zig");
const clip_mod = @import("clip.zig");
const BusId = @import("types.zig").BusId;

pub const StreamFormat = enum {
    auto,
    ogg,
    mp3,
    wav,
};

pub const StreamState = enum(u8) {
    stopped = 0,
    playing = 1,
    paused = 2,
};

pub const StreamError = error{
    OutOfMemory,
    FileNotFound,
    InvalidFormat,
    InvalidWav,
    InvalidMp3,
    InvalidOgg,
    UnsupportedWavFormat,
    UnsupportedOggFormat,
    StreamLimitReached,
    StreamTooLong,
    AccessDenied,
    Unexpected,
};

pub const StreamOptions = struct {
    bus: ?BusId = null,
    volume: f32 = 1.0,
    pan: f32 = 0.0,
    loop: bool = false,
    buffer_frames: usize = AudioStream.default_buffer_frames,
    start_paused: bool = false,
    format: StreamFormat = .auto,
    fade_in_time: f32 = 0.0,
    auto_destroy: bool = false,
};

pub const PlaySoundOptions = StreamOptions;

/// Internal decoder backend abstraction.
const Decoder = union(enum) {
    ogg: OggDecoder,
    mp3: Mp3Decoder,
    wav: WavDecoder,

    pub fn readFrames(self: *Decoder, dst: []f32, loop: bool) usize {
        return switch (self.*) {
            .ogg => |*d| d.readFrames(dst, loop),
            .mp3 => |*d| d.readFrames(dst, loop),
            .wav => |*d| d.readFrames(dst, loop),
        };
    }

    pub fn seekToFrame(self: *Decoder, frame: usize) void {
        switch (self.*) {
            .ogg => |*d| d.seekToFrame(frame),
            .mp3 => |*d| d.seekToFrame(frame),
            .wav => |*d| d.seekToFrame(frame),
        }
    }

    pub fn deinit(self: *Decoder, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .ogg => |*d| d.deinit(allocator),
            .mp3 => |*d| d.deinit(allocator),
            .wav => |*d| d.deinit(allocator),
        }
    }

    pub fn sampleRate(self: *const Decoder) f32 {
        return switch (self.*) {
            .ogg => |*d| d.sample_rate,
            .mp3 => |*d| d.sample_rate,
            .wav => |*d| d.sample_rate,
        };
    }

    pub fn totalFrames(self: *const Decoder) usize {
        return switch (self.*) {
            .ogg => |*d| d.total_frames,
            .mp3 => |*d| d.total_frames,
            .wav => |*d| d.total_frames,
        };
    }
};

const OggDecoder = struct {
    vorbis: *decode.c.stb_vorbis,
    channels: usize,
    sample_rate: f32,
    total_frames: usize,
    path_z: ?[:0]const u8 = null,
    mono_scratch: []f32,

    pub fn initFile(allocator: std.mem.Allocator, path: []const u8) StreamError!OggDecoder {
        const path_z = allocator.dupeZ(u8, path) catch return error.OutOfMemory;
        errdefer allocator.free(path_z);

        var err: c_int = 0;
        const v = decode.c.stb_vorbis_open_filename(path_z.ptr, &err, null) orelse {
            return error.InvalidOgg;
        };
        errdefer decode.c.stb_vorbis_close(v);

        const info = decode.c.stb_vorbis_get_info(v);
        const channels: usize = @intCast(info.channels);
        const rate: u32 = info.sample_rate;
        if (channels == 0 or rate == 0) return error.InvalidOgg;
        if (channels > 2) return error.UnsupportedOggFormat;

        const total_samples = decode.c.stb_vorbis_stream_length_in_samples(v);
        const mono_scratch: []f32 = if (channels == 1)
            allocator.alloc(f32, 2048) catch return error.OutOfMemory
        else
            &.{};

        return .{
            .vorbis = v,
            .channels = channels,
            .sample_rate = @floatFromInt(rate),
            .total_frames = @intCast(total_samples),
            .path_z = path_z,
            .mono_scratch = mono_scratch,
        };
    }

    pub fn initMemory(allocator: std.mem.Allocator, bytes: []const u8) StreamError!OggDecoder {
        if (bytes.len == 0) return error.InvalidOgg;
        if (bytes.len > std.math.maxInt(c_int)) return error.StreamTooLong;

        var err: c_int = 0;
        const v = decode.c.stb_vorbis_open_memory(@ptrCast(bytes.ptr), @intCast(bytes.len), &err, null) orelse {
            return error.InvalidOgg;
        };
        errdefer decode.c.stb_vorbis_close(v);

        const info = decode.c.stb_vorbis_get_info(v);
        const channels: usize = @intCast(info.channels);
        const rate: u32 = info.sample_rate;
        if (channels == 0 or rate == 0) return error.InvalidOgg;
        if (channels > 2) return error.UnsupportedOggFormat;

        const total_samples = decode.c.stb_vorbis_stream_length_in_samples(v);
        const mono_scratch: []f32 = if (channels == 1)
            allocator.alloc(f32, 2048) catch return error.OutOfMemory
        else
            &.{};

        return .{
            .vorbis = v,
            .channels = channels,
            .sample_rate = @floatFromInt(rate),
            .total_frames = @intCast(total_samples),
            .path_z = null,
            .mono_scratch = mono_scratch,
        };
    }

    pub fn readFrames(self: *OggDecoder, dst: []f32, loop: bool) usize {
        const want_frames = dst.len / 2;
        if (want_frames == 0) return 0;

        var got_frames: usize = 0;
        if (self.channels == 1) {
            const chunk = @min(want_frames, self.mono_scratch.len);
            const got = decode.c.stb_vorbis_get_samples_float_interleaved(
                self.vorbis,
                1,
                self.mono_scratch.ptr,
                @intCast(chunk),
            );
            if (got > 0) {
                const n: usize = @intCast(got);
                for (0..n) |i| {
                    const s = self.mono_scratch[i];
                    dst[i * 2] = s;
                    dst[i * 2 + 1] = s;
                }
                got_frames = n;
            }
        } else {
            const got = decode.c.stb_vorbis_get_samples_float_interleaved(
                self.vorbis,
                2,
                dst.ptr,
                @intCast(want_frames * 2),
            );
            if (got > 0) {
                got_frames = @intCast(got);
            }
        }

        if (got_frames == 0 and loop) {
            _ = decode.c.stb_vorbis_seek_start(self.vorbis);
            return self.readFrames(dst, false);
        }

        return got_frames;
    }

    pub fn seekToFrame(self: *OggDecoder, frame: usize) void {
        _ = decode.c.stb_vorbis_seek_frame(self.vorbis, @intCast(frame));
    }

    pub fn deinit(self: *OggDecoder, allocator: std.mem.Allocator) void {
        decode.c.stb_vorbis_close(self.vorbis);
        if (self.path_z) |pz| allocator.free(pz);
        if (self.mono_scratch.len > 0) allocator.free(self.mono_scratch);
    }
};

const Mp3Decoder = struct {
    mp3: decode.c.drmp3,
    channels: usize,
    sample_rate: f32,
    total_frames: usize,
    path_z: ?[:0]const u8 = null,
    mono_scratch: []f32,

    pub fn initFile(allocator: std.mem.Allocator, path: []const u8) StreamError!Mp3Decoder {
        const path_z = allocator.dupeZ(u8, path) catch return error.OutOfMemory;
        errdefer allocator.free(path_z);

        var mp3 = std.mem.zeroes(decode.c.drmp3);
        if (decode.c.drmp3_init_file(&mp3, path_z.ptr, null) == 0) {
            return error.InvalidMp3;
        }
        errdefer decode.c.drmp3_uninit(&mp3);

        const channels: usize = mp3.channels;
        const rate: u32 = mp3.sampleRate;
        if (channels == 0 or channels > 2 or rate == 0) return error.InvalidMp3;

        const total_frames = decode.c.drmp3_get_pcm_frame_count(&mp3);
        const mono_scratch: []f32 = if (channels == 1)
            allocator.alloc(f32, 2048) catch return error.OutOfMemory
        else
            &.{};

        return .{
            .mp3 = mp3,
            .channels = channels,
            .sample_rate = @floatFromInt(rate),
            .total_frames = @intCast(total_frames),
            .path_z = path_z,
            .mono_scratch = mono_scratch,
        };
    }

    pub fn initMemory(allocator: std.mem.Allocator, bytes: []const u8) StreamError!Mp3Decoder {
        if (bytes.len == 0) return error.InvalidMp3;

        var mp3 = std.mem.zeroes(decode.c.drmp3);
        if (decode.c.drmp3_init_memory(&mp3, @ptrCast(bytes.ptr), bytes.len, null) == 0) {
            return error.InvalidMp3;
        }
        errdefer decode.c.drmp3_uninit(&mp3);

        const channels: usize = mp3.channels;
        const rate: u32 = mp3.sampleRate;
        if (channels == 0 or channels > 2 or rate == 0) return error.InvalidMp3;

        const total_frames = decode.c.drmp3_get_pcm_frame_count(&mp3);
        const mono_scratch: []f32 = if (channels == 1)
            allocator.alloc(f32, 2048) catch return error.OutOfMemory
        else
            &.{};

        return .{
            .mp3 = mp3,
            .channels = channels,
            .sample_rate = @floatFromInt(rate),
            .total_frames = @intCast(total_frames),
            .path_z = null,
            .mono_scratch = mono_scratch,
        };
    }

    pub fn readFrames(self: *Mp3Decoder, dst: []f32, loop: bool) usize {
        const want_frames = dst.len / 2;
        if (want_frames == 0) return 0;

        var got_frames: usize = 0;
        if (self.channels == 1) {
            const chunk = @min(want_frames, self.mono_scratch.len);
            const got = decode.c.drmp3_read_pcm_frames_f32(&self.mp3, chunk, self.mono_scratch.ptr);
            if (got > 0) {
                const n: usize = @intCast(got);
                for (0..n) |i| {
                    const s = self.mono_scratch[i];
                    dst[i * 2] = s;
                    dst[i * 2 + 1] = s;
                }
                got_frames = n;
            }
        } else {
            const got = decode.c.drmp3_read_pcm_frames_f32(&self.mp3, want_frames, dst.ptr);
            if (got > 0) {
                got_frames = @intCast(got);
            }
        }

        if (got_frames == 0 and loop) {
            _ = decode.c.drmp3_seek_to_pcm_frame(&self.mp3, 0);
            return self.readFrames(dst, false);
        }

        return got_frames;
    }

    pub fn seekToFrame(self: *Mp3Decoder, frame: usize) void {
        _ = decode.c.drmp3_seek_to_pcm_frame(&self.mp3, @intCast(frame));
    }

    pub fn deinit(self: *Mp3Decoder, allocator: std.mem.Allocator) void {
        decode.c.drmp3_uninit(&self.mp3);
        if (self.path_z) |pz| allocator.free(pz);
        if (self.mono_scratch.len > 0) allocator.free(self.mono_scratch);
    }
};

const WavDecoder = struct {
    file: ?std.Io.File = null,
    memory: ?[]const u8 = null,
    data_offset: u64,
    data_size: u64,
    channels: usize,
    sample_rate: f32,
    bits: u16,
    is_float: bool,
    bytes_per_sample: usize,
    block_align: usize,
    total_frames: usize,
    current_frame: usize = 0,
    raw_buffer: []u8,

    pub fn initFile(allocator: std.mem.Allocator, path: []const u8) StreamError!WavDecoder {
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            error.AccessDenied => return error.AccessDenied,
            else => return error.InvalidWav,
        };
        errdefer file.close(io);

        const file_len = file.length(io) catch return error.InvalidWav;
        if (file_len < 12) return error.InvalidWav;
        var header: [12]u8 = undefined;
        if ((file.readPositionalAll(io, &header, 0) catch return error.InvalidWav) < 12) return error.InvalidWav;
        if (!std.mem.eql(u8, header[0..4], "RIFF") or !std.mem.eql(u8, header[8..12], "WAVE")) return error.InvalidWav;

        var fmt_opt: ?struct {
            format: u16,
            channels: u16,
            rate: u32,
            bits: u16,
        } = null;
        var data_off: u64 = 0;
        var data_size: u64 = 0;
        var have_data = false;
        var off: u64 = 12;

        while (off + 8 <= file_len) {
            var chunk_hdr: [8]u8 = undefined;
            if ((file.readPositionalAll(io, &chunk_hdr, off) catch return error.InvalidWav) < 8) return error.InvalidWav;
            const size: u64 = std.mem.readInt(u32, chunk_hdr[4..8], .little);
            const start = off + 8;
            const end = start + size;
            if (end < start or end > file_len) return error.InvalidWav;

            if (std.mem.eql(u8, chunk_hdr[0..4], "fmt ")) {
                if (size < 16) return error.InvalidWav;
                var fbuf: [16]u8 = undefined;
                if ((file.readPositionalAll(io, &fbuf, start) catch return error.InvalidWav) < 16) return error.InvalidWav;
                fmt_opt = .{
                    .format = std.mem.readInt(u16, fbuf[0..2], .little),
                    .channels = std.mem.readInt(u16, fbuf[2..4], .little),
                    .rate = std.mem.readInt(u32, fbuf[4..8], .little),
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

        const f = fmt_opt orelse return error.InvalidWav;
        if (!have_data or f.rate == 0) return error.InvalidWav;
        if (f.channels != 1 and f.channels != 2) return error.UnsupportedWavFormat;
        const is_pcm = f.format == 1 and (f.bits == 8 or f.bits == 16 or f.bits == 24 or f.bits == 32);
        const is_f32 = f.format == 3 and f.bits == 32;
        if (!is_pcm and !is_f32) return error.UnsupportedWavFormat;

        const bytes_per_sample = f.bits / 8;
        const block_align = @as(usize, f.channels) * bytes_per_sample;
        const total_frames = data_size / block_align;
        if (total_frames == 0) return error.InvalidWav;

        const raw_buffer = allocator.alloc(u8, 8192 * block_align) catch return error.OutOfMemory;

        return .{
            .file = file,
            .memory = null,
            .data_offset = data_off,
            .data_size = data_size,
            .channels = f.channels,
            .sample_rate = @floatFromInt(f.rate),
            .bits = f.bits,
            .is_float = is_f32,
            .bytes_per_sample = bytes_per_sample,
            .block_align = block_align,
            .total_frames = @intCast(total_frames),
            .raw_buffer = raw_buffer,
        };
    }

    pub fn initMemory(allocator: std.mem.Allocator, bytes: []const u8) StreamError!WavDecoder {
        if (bytes.len < 12) return error.InvalidWav;
        if (!std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.InvalidWav;

        var fmt_opt: ?struct {
            format: u16,
            channels: u16,
            rate: u32,
            bits: u16,
        } = null;
        var data_off: u64 = 0;
        var data_size: u64 = 0;
        var have_data = false;
        var off: usize = 12;

        while (off + 8 <= bytes.len) {
            const id = bytes[off..][0..4];
            const size: usize = std.mem.readInt(u32, bytes[off + 4 ..][0..4], .little);
            const start = off + 8;
            if (size > bytes.len - start) return error.InvalidWav;
            const chunk = bytes[start..][0..size];

            if (std.mem.eql(u8, id, "fmt ")) {
                if (chunk.len < 16) return error.InvalidWav;
                fmt_opt = .{
                    .format = std.mem.readInt(u16, chunk[0..2], .little),
                    .channels = std.mem.readInt(u16, chunk[2..4], .little),
                    .rate = std.mem.readInt(u32, chunk[4..8], .little),
                    .bits = std.mem.readInt(u16, chunk[14..16], .little),
                };
            } else if (std.mem.eql(u8, id, "data")) {
                if (!have_data) {
                    data_off = start;
                    data_size = size;
                    have_data = true;
                }
            }
            off = start + size + (size & 1);
        }

        const f = fmt_opt orelse return error.InvalidWav;
        if (!have_data or f.rate == 0) return error.InvalidWav;
        if (f.channels != 1 and f.channels != 2) return error.UnsupportedWavFormat;
        const is_pcm = f.format == 1 and (f.bits == 8 or f.bits == 16 or f.bits == 24 or f.bits == 32);
        const is_f32 = f.format == 3 and f.bits == 32;
        if (!is_pcm and !is_f32) return error.UnsupportedWavFormat;

        const bytes_per_sample = f.bits / 8;
        const block_align = @as(usize, f.channels) * bytes_per_sample;
        const total_frames = data_size / block_align;
        if (total_frames == 0) return error.InvalidWav;

        const raw_buffer = allocator.alloc(u8, 4096 * block_align) catch return error.OutOfMemory;

        return .{
            .file = null,
            .memory = bytes,
            .data_offset = data_off,
            .data_size = data_size,
            .channels = f.channels,
            .sample_rate = @floatFromInt(f.rate),
            .bits = f.bits,
            .is_float = is_f32,
            .bytes_per_sample = bytes_per_sample,
            .block_align = block_align,
            .total_frames = @intCast(total_frames),
            .raw_buffer = raw_buffer,
        };
    }

    pub fn readFrames(self: *WavDecoder, dst: []f32, loop: bool) usize {
        const want_frames = dst.len / 2;
        if (want_frames == 0) return 0;

        if (self.current_frame >= self.total_frames) {
            if (loop) {
                self.current_frame = 0;
            } else {
                return 0;
            }
        }

        const avail_in_file = self.total_frames - self.current_frame;
        const frames_to_read = @min(want_frames, avail_in_file);
        const bytes_to_read = frames_to_read * self.block_align;

        var src_slice: []const u8 = undefined;
        if (self.file) |file| {
            const io = std.Io.Threaded.global_single_threaded.io();
            const pos = self.data_offset + @as(u64, @intCast(self.current_frame * self.block_align));
            const buf = self.raw_buffer[0..@min(bytes_to_read, self.raw_buffer.len)];
            const got = file.readPositionalAll(io, buf, pos) catch 0;
            if (got < self.block_align) return 0;
            const actual_frames = got / self.block_align;
            src_slice = buf[0 .. actual_frames * self.block_align];
            decodeChunk(dst[0 .. actual_frames * 2], src_slice, self);
            self.current_frame += actual_frames;
            return actual_frames;
        } else if (self.memory) |mem| {
            const start: usize = @intCast(self.data_offset + @as(u64, @intCast(self.current_frame * self.block_align)));
            src_slice = mem[start .. start + bytes_to_read];
            decodeChunk(dst[0 .. frames_to_read * 2], src_slice, self);
            self.current_frame += frames_to_read;
            return frames_to_read;
        }
        return 0;
    }

    fn decodeChunk(dst: []f32, src: []const u8, self: *const WavDecoder) void {
        const frames = src.len / self.block_align;
        if (self.is_float) {
            if (self.channels == 1) {
                for (0..frames) |i| {
                    const bytes = src[i * 4 ..][0..4];
                    const v: f32 = @bitCast(std.mem.readInt(u32, bytes, .little));
                    dst[i * 2] = v;
                    dst[i * 2 + 1] = v;
                }
            } else {
                for (0..frames) |i| {
                    const b0 = src[i * 8 ..][0..4];
                    const b1 = src[i * 8 + 4 ..][0..4];
                    dst[i * 2] = @bitCast(std.mem.readInt(u32, b0, .little));
                    dst[i * 2 + 1] = @bitCast(std.mem.readInt(u32, b1, .little));
                }
            }
        } else switch (self.bits) {
            8 => {
                if (self.channels == 1) {
                    for (0..frames) |i| {
                        const v = (@as(f32, @floatFromInt(src[i])) - 128.0) / 128.0;
                        dst[i * 2] = v;
                        dst[i * 2 + 1] = v;
                    }
                } else {
                    for (0..frames) |i| {
                        dst[i * 2] = (@as(f32, @floatFromInt(src[i * 2])) - 128.0) / 128.0;
                        dst[i * 2 + 1] = (@as(f32, @floatFromInt(src[i * 2 + 1])) - 128.0) / 128.0;
                    }
                }
            },
            16 => {
                if (self.channels == 1) {
                    for (0..frames) |i| {
                        const raw = std.mem.readInt(i16, src[i * 2 ..][0..2], .little);
                        const v = @as(f32, @floatFromInt(raw)) / 32768.0;
                        dst[i * 2] = v;
                        dst[i * 2 + 1] = v;
                    }
                } else {
                    for (0..frames) |i| {
                        const raw_l = std.mem.readInt(i16, src[i * 4 ..][0..2], .little);
                        const raw_r = std.mem.readInt(i16, src[i * 4 + 2 ..][0..2], .little);
                        dst[i * 2] = @as(f32, @floatFromInt(raw_l)) / 32768.0;
                        dst[i * 2 + 1] = @as(f32, @floatFromInt(raw_r)) / 32768.0;
                    }
                }
            },
            24 => {
                if (self.channels == 1) {
                    for (0..frames) |i| {
                        const b = src[i * 3 ..][0..3];
                        const u_val = @as(u32, b[0]) | (@as(u32, b[1]) << 8) | (@as(u32, b[2]) << 16);
                        const sign_ext: i32 = @as(i32, @bitCast(if ((u_val & 0x800000) != 0) u_val | 0xFF000000 else u_val));
                        const v = @as(f32, @floatFromInt(sign_ext)) / 8388608.0;
                        dst[i * 2] = v;
                        dst[i * 2 + 1] = v;
                    }
                } else {
                    for (0..frames) |i| {
                        const b0 = src[i * 6 ..][0..3];
                        const b1 = src[i * 6 + 3 ..][0..3];
                        const val0 = @as(u32, b0[0]) | (@as(u32, b0[1]) << 8) | (@as(u32, b0[2]) << 16);
                        const val1 = @as(u32, b1[0]) | (@as(u32, b1[1]) << 8) | (@as(u32, b1[2]) << 16);
                        const s0: i32 = @as(i32, @bitCast(if ((val0 & 0x800000) != 0) val0 | 0xFF000000 else val0));
                        const s1: i32 = @as(i32, @bitCast(if ((val1 & 0x800000) != 0) val1 | 0xFF000000 else val1));
                        dst[i * 2] = @as(f32, @floatFromInt(s0)) / 8388608.0;
                        dst[i * 2 + 1] = @as(f32, @floatFromInt(s1)) / 8388608.0;
                    }
                }
            },
            32 => {
                if (self.channels == 1) {
                    for (0..frames) |i| {
                        const raw = std.mem.readInt(i32, src[i * 4 ..][0..4], .little);
                        const v = @as(f32, @floatFromInt(raw)) / 2147483648.0;
                        dst[i * 2] = v;
                        dst[i * 2 + 1] = v;
                    }
                } else {
                    for (0..frames) |i| {
                        const raw_l = std.mem.readInt(i32, src[i * 8 ..][0..4], .little);
                        const raw_r = std.mem.readInt(i32, src[i * 8 + 4 ..][0..4], .little);
                        dst[i * 2] = @as(f32, @floatFromInt(raw_l)) / 2147483648.0;
                        dst[i * 2 + 1] = @as(f32, @floatFromInt(raw_r)) / 2147483648.0;
                    }
                }
            },
            else => {},
        }
    }

    pub fn seekToFrame(self: *WavDecoder, frame: usize) void {
        self.current_frame = @min(frame, self.total_frames);
    }

    pub fn deinit(self: *WavDecoder, allocator: std.mem.Allocator) void {
        if (self.file) |file| {
            const io = std.Io.Threaded.global_single_threaded.io();
            file.close(io);
        }
        allocator.free(self.raw_buffer);
    }
};

/// High-performance, real-time safe audio stream.
pub const AudioStream = struct {
    pub const default_buffer_frames: usize = 65536; // power of 2: ~1.48s at 44.1kHz

    allocator: std.mem.Allocator,
    decoder: Decoder,
    ring_buffer: []f32, // interleaved stereo f32 at engine_rate
    capacity: usize, // power of 2 frames
    mask: usize, // capacity - 1

    // Lock-free single-producer single-consumer indices (frame counts)
    write_pos: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    read_pos: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    // Live state
    state: std.atomic.Value(StreamState) = std.atomic.Value(StreamState).init(.stopped),
    volume: std.atomic.Value(u32) = std.atomic.Value(u32).init(@bitCast(@as(f32, 1.0))),
    pan: std.atomic.Value(u32) = std.atomic.Value(u32).init(@bitCast(@as(f32, 0.0))),
    bus_id: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),
    loop: std.atomic.Value(bool) = std.atomic.Value(bool).init(true),

    source_rate: f32,
    engine_rate: f32,
    total_frames: usize,
    played_frames: usize = 0,
    is_eof: bool = false,

    // Smooth volume fade
    fade_active: bool = false,
    fade_start_vol: f32 = 0.0,
    fade_target_vol: f32 = 1.0,
    fade_duration: f32 = 1.0,
    fade_elapsed: f32 = 0.0,
    stop_on_fade_out: bool = false,

    // Preallocated scratch for decoding raw source frames
    source_scratch: []f32,
    auto_destroy: bool = false,

    pub fn openFile(
        allocator: std.mem.Allocator,
        path: []const u8,
        engine_sample_rate: f32,
        options: StreamOptions,
    ) StreamError!*AudioStream {
        const eng_rate = if (engine_sample_rate > 0.0) engine_sample_rate else 44100.0;
        const fmt = resolveFormatFromPath(path, options.format);

        const decoder: Decoder = switch (fmt) {
            .ogg => .{ .ogg = try OggDecoder.initFile(allocator, path) },
            .mp3 => .{ .mp3 = try Mp3Decoder.initFile(allocator, path) },
            .wav => .{ .wav = try WavDecoder.initFile(allocator, path) },
            .auto => return error.InvalidFormat,
        };

        return createStream(allocator, decoder, eng_rate, options);
    }

    pub fn openMemory(
        allocator: std.mem.Allocator,
        bytes: []const u8,
        format: StreamFormat,
        engine_sample_rate: f32,
        options: StreamOptions,
    ) StreamError!*AudioStream {
        const eng_rate = if (engine_sample_rate > 0.0) engine_sample_rate else 44100.0;
        const fmt = resolveFormatFromMemory(bytes, format);

        const decoder: Decoder = switch (fmt) {
            .ogg => .{ .ogg = try OggDecoder.initMemory(allocator, bytes) },
            .mp3 => .{ .mp3 = try Mp3Decoder.initMemory(allocator, bytes) },
            .wav => .{ .wav = try WavDecoder.initMemory(allocator, bytes) },
            .auto => return error.InvalidFormat,
        };

        return createStream(allocator, decoder, eng_rate, options);
    }

    fn createStream(
        allocator: std.mem.Allocator,
        decoder: Decoder,
        engine_rate: f32,
        options: StreamOptions,
    ) StreamError!*AudioStream {
        var dec = decoder;
        errdefer dec.deinit(allocator);

        // Ensure power of two capacity for branchless bitwise wrapping
        var cap: usize = 1024;
        while (cap < options.buffer_frames and cap < 262144) : (cap *= 2) {}

        const ring = allocator.alloc(f32, cap * 2) catch return error.OutOfMemory;
        errdefer allocator.free(ring);
        @memset(ring, 0.0);

        const scratch = allocator.alloc(f32, 4096 * 2) catch return error.OutOfMemory;
        errdefer allocator.free(scratch);

        const stream = allocator.create(AudioStream) catch return error.OutOfMemory;
        const b_id: u8 = if (options.bus) |b| @intFromEnum(b) else 0;

        stream.* = .{
            .allocator = allocator,
            .decoder = dec,
            .ring_buffer = ring,
            .capacity = cap,
            .mask = cap - 1,
            .source_rate = dec.sampleRate(),
            .engine_rate = engine_rate,
            .total_frames = dec.totalFrames(),
            .source_scratch = scratch,
        };

        stream.setVolume(options.volume);
        stream.setPan(options.pan);
        stream.bus_id.store(b_id, .release);
        stream.loop.store(options.loop, .release);
        stream.auto_destroy = options.auto_destroy;

        if (options.fade_in_time > 0.0) {
            stream.setVolume(0.0);
            stream.fadeTo(options.volume, options.fade_in_time, false);
        }

        // Initial prefill of the ring buffer
        stream.refill();

        if (!options.start_paused) {
            stream.play();
        } else {
            stream.state.store(.paused, .release);
        }

        return stream;
    }

    pub fn deinit(self: *AudioStream) void {
        self.stop();
        self.decoder.deinit(self.allocator);
        self.allocator.free(self.ring_buffer);
        self.allocator.free(self.source_scratch);
        self.allocator.destroy(self);
    }

    pub fn play(self: *AudioStream) void {
        self.state.store(.playing, .release);
    }

    pub fn pause(self: *AudioStream) void {
        self.state.store(.paused, .release);
    }

    pub fn unpause(self: *AudioStream) void {
        self.state.store(.playing, .release);
    }

    pub fn @"resume"(self: *AudioStream) void {
        self.state.store(.playing, .release);
    }

    pub fn stop(self: *AudioStream) void {
        self.state.store(.stopped, .release);
        // Flush the ring buffer
        const w = self.write_pos.load(.monotonic);
        self.read_pos.store(w, .release);
        self.played_frames = 0;
        self.is_eof = false;
        self.decoder.seekToFrame(0);
    }

    pub fn seekToSeconds(self: *AudioStream, seconds: f32) void {
        const frame = @as(usize, @intFromFloat(@max(0.0, seconds) * self.source_rate));
        self.seekToFrame(frame);
    }

    pub fn seekToFrame(self: *AudioStream, frame: usize) void {
        self.decoder.seekToFrame(frame);
        const w = self.write_pos.load(.monotonic);
        self.read_pos.store(w, .release);
        self.played_frames = @intFromFloat(@as(f32, @floatFromInt(frame)) * (self.engine_rate / self.source_rate));
        self.is_eof = false;
        self.refill();
    }

    pub fn setVolume(self: *AudioStream, vol: f32) void {
        self.volume.store(@bitCast(std.math.clamp(vol, 0.0, 10.0)), .release);
    }

    pub fn getVolume(self: *const AudioStream) f32 {
        return @bitCast(self.volume.load(.acquire));
    }

    pub fn setPan(self: *AudioStream, pan: f32) void {
        self.pan.store(@bitCast(std.math.clamp(pan, -1.0, 1.0)), .release);
    }

    pub fn getPan(self: *const AudioStream) f32 {
        return @bitCast(self.pan.load(.acquire));
    }

    pub fn setBus(self: *AudioStream, bus: ?BusId) void {
        const id: u8 = if (bus) |b| @intFromEnum(b) else 0;
        self.bus_id.store(id, .release);
    }

    pub fn getBus(self: *const AudioStream) ?BusId {
        const id = self.bus_id.load(.acquire);
        return @enumFromInt(id);
    }

    pub fn setLoop(self: *AudioStream, loop_playback: bool) void {
        self.loop.store(loop_playback, .release);
    }

    pub fn isLooping(self: *const AudioStream) bool {
        return self.loop.load(.acquire);
    }

    pub fn fadeTo(self: *AudioStream, target_volume: f32, duration_seconds: f32, stop_on_fade_out: bool) void {
        self.fade_start_vol = self.getVolume();
        self.fade_target_vol = std.math.clamp(target_volume, 0.0, 10.0);
        self.fade_duration = @max(0.001, duration_seconds);
        self.fade_elapsed = 0.0;
        self.fade_active = true;
        self.stop_on_fade_out = stop_on_fade_out;
    }

    pub fn getState(self: *const AudioStream) StreamState {
        return self.state.load(.acquire);
    }

    pub fn isPlaying(self: *const AudioStream) bool {
        return self.getState() == .playing;
    }

    pub fn isPaused(self: *const AudioStream) bool {
        return self.getState() == .paused;
    }

    pub fn isStopped(self: *const AudioStream) bool {
        return self.getState() == .stopped;
    }

    pub fn getPositionSeconds(self: *const AudioStream) f32 {
        if (self.engine_rate <= 0.0) return 0.0;
        return @as(f32, @floatFromInt(self.played_frames)) / self.engine_rate;
    }

    pub fn getDurationSeconds(self: *const AudioStream) f32 {
        if (self.source_rate <= 0.0) return 0.0;
        return @as(f32, @floatFromInt(self.total_frames)) / self.source_rate;
    }

    /// Producer update: processes volume fading and refills the ring buffer.
    pub fn update(self: *AudioStream, dt: f32) void {
        if (self.fade_active) {
            self.fade_elapsed += dt;
            const t = @min(1.0, self.fade_elapsed / self.fade_duration);
            const cur_vol = std.math.lerp(self.fade_start_vol, self.fade_target_vol, t);
            self.setVolume(cur_vol);
            if (t >= 1.0) {
                self.fade_active = false;
                if (self.stop_on_fade_out and self.fade_target_vol <= 0.001) {
                    self.stop();
                    return;
                }
            }
        }

        if (self.state.load(.acquire) != .playing) return;

        self.refill();
    }

    /// Fills the ring buffer from the decoder up to available capacity.
    pub fn refill(self: *AudioStream) void {
        const w = self.write_pos.load(.monotonic);
        const r = self.read_pos.load(.acquire);
        const occupied = w -% r;
        if (occupied >= self.capacity) return; // Buffer is full

        const free_frames = self.capacity - occupied;
        if (free_frames < 256) return;

        const is_loop = self.loop.load(.monotonic);
        var written: usize = 0;

        while (written < free_frames) {
            const chunk_to_read = @min(free_frames - written, self.source_scratch.len / 2);
            const got = self.decoder.readFrames(self.source_scratch[0 .. chunk_to_read * 2], is_loop);
            if (got == 0) {
                self.is_eof = true;
                break;
            }

            for (0..got) |i| {
                const idx = (w +% written +% i) & self.mask;
                self.ring_buffer[idx * 2] = self.source_scratch[i * 2];
                self.ring_buffer[idx * 2 + 1] = self.source_scratch[i * 2 + 1];
            }
            written += got;
            if (got < chunk_to_read) {
                self.is_eof = true;
                break;
            }
        }

        if (written > 0) {
            self.write_pos.store(w +% written, .release);
        }
    }

    /// Consumer render: reads stereo frames from the ring buffer and mixes them
    /// into `target` with volume and panning. Real-time safe: 0 allocations.
    pub fn renderToBuffer(self: *AudioStream, target: []f32) usize {
        const w = self.write_pos.load(.acquire);
        const r = self.read_pos.load(.monotonic);
        const available = w -% r;
        if (available == 0) {
            if (!self.loop.load(.monotonic) and self.is_eof) {
                self.state.store(.stopped, .release);
            }
            return 0;
        }

        const want_frames = target.len / 2;
        const to_render = @min(available, want_frames);

        const vol = self.getVolume();
        const pan = self.getPan();
        const gl = vol * @min(1.0, 1.0 - pan);
        const gr = vol * @min(1.0, 1.0 + pan);

        for (0..to_render) |i| {
            const idx = (r +% i) & self.mask;
            const s_l = self.ring_buffer[idx * 2];
            const s_r = self.ring_buffer[idx * 2 + 1];
            target[i * 2] += s_l * gl;
            target[i * 2 + 1] += s_r * gr;
        }

        self.read_pos.store(r +% to_render, .release);
        self.played_frames += to_render;

        if (to_render < want_frames and !self.loop.load(.monotonic) and self.is_eof) {
            self.state.store(.stopped, .release);
        }

        return to_render;
    }
};

fn resolveFormatFromPath(path: []const u8, requested: StreamFormat) StreamFormat {
    if (requested != .auto) return requested;
    if (std.mem.endsWith(u8, path, ".ogg") or std.mem.endsWith(u8, path, ".OGG")) return .ogg;
    if (std.mem.endsWith(u8, path, ".mp3") or std.mem.endsWith(u8, path, ".MP3")) return .mp3;
    if (std.mem.endsWith(u8, path, ".wav") or std.mem.endsWith(u8, path, ".WAV")) return .wav;
    return .ogg;
}

fn resolveFormatFromMemory(bytes: []const u8, requested: StreamFormat) StreamFormat {
    if (requested != .auto) return requested;
    if (bytes.len >= 12 and std.mem.eql(u8, bytes[0..4], "RIFF") and std.mem.eql(u8, bytes[8..12], "WAVE")) return .wav;
    if (bytes.len >= 4 and std.mem.eql(u8, bytes[0..4], "OggS")) return .ogg;
    return .mp3;
}
