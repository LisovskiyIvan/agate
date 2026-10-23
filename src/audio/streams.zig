//! Stream registry + sound/music playback API.
//! Split out of `audio.zig` (facade).
//!
//! This module owns the engine-side stream slots (`registerStream`,
//! `unregisterStream`, `createStreamFromFile`/`createStreamFromMemory`,
//! `destroyStream`, `updateStreams` with auto-destroy reaping), the
//! fire-and-forget sound API (`playSound`, `playSoundFromMemory`,
//! `playSoundOnce`, `playSoundOnceFromMemory`, `stopSound`, `pauseSound`,
//! `resumeSound`, `setSoundVolume`, `stopAllSounds`, `crossfadeSound`,
//! `crossfadeSoundFromMemory`), and the dedicated music slot
//! (`playMusic`, `playMusicFromMemory`, `crossfadeMusic`, `stopMusic`,
//! `pauseMusic`, `resumeMusic`, `setMusicVolume`, `getMusicStream`).
//! Decoding/streaming itself lives
//! in the pre-existing `stream.zig` leaf; this module only manages slots.
//!
//! Anti-cycle rule (same as `profiler/*`, `particles/*`): every function
//! takes the engine as `anytype` (a `*AudioEngine` from `engine.zig` in
//! practice) and this module never imports `engine.zig` or the `audio.zig`
//! facade back; everything else is fields or the `stream.zig` leaf.

const std = @import("std");
const clip_mod = @import("clip.zig");
const stream_mod = @import("stream.zig");
const types = @import("types.zig");

const BusId = types.BusId;
const AudioClip = clip_mod.AudioClip;
const StreamFormat = stream_mod.StreamFormat;
const StreamOptions = stream_mod.StreamOptions;
const AudioStream = stream_mod.AudioStream;

/// Depth of the retire request/ack rings. Comfortably above `max_streams`
/// (32): a frame would have to retire more than this many streams between
/// two audio callbacks to overflow.
pub const retire_capacity: usize = 128;

/// Minimal SPSC ring of retired stream pointers. Producer/consumer are
/// fixed (game thread pushes requests, audio thread pops and pushes acks).
pub fn RetireRing(comptime capacity: usize) type {
    return struct {
        buf: [capacity]usize = @splat(0),
        head: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        tail: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

        pub fn push(self: *@This(), stream: *AudioStream) bool {
            const h = self.head.load(.monotonic);
            const t = self.tail.load(.acquire);
            if (h -% t >= capacity) return false;
            self.buf[h % capacity] = @intFromPtr(stream);
            self.head.store(h +% 1, .release);
            return true;
        }

        pub fn pop(self: *@This()) ?*AudioStream {
            const t = self.tail.load(.monotonic);
            const h = self.head.load(.acquire);
            if (t == h) return null;
            const raw = self.buf[t % capacity];
            self.tail.store(t +% 1, .release);
            return @ptrFromInt(raw);
        }

        pub fn len(self: *const @This()) usize {
            return self.head.load(.monotonic) -| self.tail.load(.monotonic);
        }
    };
}

pub inline fn getSlot(self: anytype, i: usize) ?*AudioStream {
    const raw = self.streams[i].load(.acquire);
    return if (raw == 0) null else @ptrFromInt(raw);
}

pub inline fn setSlot(self: anytype, i: usize, stream: ?*AudioStream) void {
    self.streams[i].store(if (stream) |s| @intFromPtr(s) else 0, .release);
}

pub fn registerStream(self: anytype, stream: *AudioStream) !void {
    for (0..types.max_streams) |i| {
        if (getSlot(self, i) == null) {
            setSlot(self, i, stream);
            return;
        }
    }
    return error.StreamLimitReached;
}

pub fn unregisterStream(self: anytype, stream: *AudioStream) void {
    for (0..types.max_streams) |i| {
        if (getSlot(self, i) == stream) {
            setSlot(self, i, null);
            break;
        }
    }
    if (self.music_stream == stream) self.music_stream = null;
    if (self.music_fade_stream == stream) self.music_fade_stream = null;
}

pub fn createStreamFromFile(
    self: anytype,
    allocator: std.mem.Allocator,
    path: []const u8,
    options: StreamOptions,
) !*AudioStream {
    const stream = try AudioStream.openFile(allocator, path, self.sample_rate, options);
    errdefer stream.deinit();
    try registerStream(self, stream);
    return stream;
}

pub fn createStreamFromMemory(
    self: anytype,
    allocator: std.mem.Allocator,
    bytes: []const u8,
    format: StreamFormat,
    options: StreamOptions,
) !*AudioStream {
    const stream = try AudioStream.openMemory(allocator, bytes, format, self.sample_rate, options);
    errdefer stream.deinit();
    try registerStream(self, stream);
    return stream;
}

/// Removes the stream from its slot and hands its lifetime to the audio
/// thread. `deinit` runs on the game thread only after the audio callback
/// has acknowledged that no in-flight render can still reference it.
pub fn retireStream(self: anytype, stream: *AudioStream) void {
    unregisterStream(self, stream);
    if (!self.started) {
        stream.deinit();
        return;
    }
    if (!self.retire_req.push(stream)) enqueueRetireOverflow(self, stream);
}

fn enqueueRetireOverflow(self: anytype, stream: *AudioStream) void {
    for (&self.retire_overflow) |*slot| {
        if (slot.* == null) {
            slot.* = stream;
            return;
        }
    }
    // Ring + overflow are sized above the live-stream limit; failing here
    // would mean losing the only ownership record, so leak instead of UAF.
    std.debug.print("audio: stream retire queue overflow; leaking one stream\n", .{});
}

fn flushRetireOverflow(self: anytype) void {
    for (&self.retire_overflow) |*slot| {
        const s = slot.* orelse continue;
        if (self.retire_req.push(s)) slot.* = null;
    }
}

/// Game thread: frees streams the audio thread already stopped referencing.
pub fn drainRetireAcks(self: anytype) void {
    while (self.retire_ack.pop()) |s| s.deinit();
}

/// Audio thread, at the top of renderFrames: acknowledge every pending
/// retire request. After this returns, the previous callback (and therefore
/// any in-flight `renderToBuffer`) has completed.
pub fn processRetireRequests(self: anytype) void {
    while (self.retire_ack.len() < retire_capacity) {
        const s = self.retire_req.pop() orelse break;
        if (!self.retire_ack.push(s)) break;
    }
}

pub fn destroyStream(self: anytype, stream: *AudioStream) void {
    retireStream(self, stream);
}

pub fn updateStreams(self: anytype, dt: f32) void {
    drainRetireAcks(self);
    flushRetireOverflow(self);
    for (0..types.max_streams) |i| {
        const s = getSlot(self, i) orelse continue;
        s.update(dt);
        if (s.isStopped() and (s.auto_destroy or s.stop_on_fade_out)) {
            retireStream(self, s);
        }
    }
    if (self.music_fade_stream) |fade_s| {
        if (fade_s.isStopped()) {
            destroyStream(self, fade_s);
            self.music_fade_stream = null;
        }
    }
}

// ========================================================================
// Sound Playback & Streaming API
// ========================================================================

/// Plays a sound from a file path (OGG, MP3, WAV).
/// Streams audio in background with real-time safety.
pub fn playSound(
    self: anytype,
    allocator: std.mem.Allocator,
    path: []const u8,
    options: StreamOptions,
) !*AudioStream {
    return createStreamFromFile(self, allocator, path, options);
}

/// Plays a sound from an in-memory byte slice (OGG, MP3, WAV).
pub fn playSoundFromMemory(
    self: anytype,
    allocator: std.mem.Allocator,
    bytes: []const u8,
    format: StreamFormat,
    options: StreamOptions,
) !*AudioStream {
    return createStreamFromMemory(self, allocator, bytes, format, options);
}

/// Fire-and-forget sound playback: automatically destroys and frees the stream
/// when playback finishes or is stopped.
pub fn playSoundOnce(
    self: anytype,
    allocator: std.mem.Allocator,
    path: []const u8,
    options: StreamOptions,
) !void {
    var opt = options;
    opt.loop = false;
    opt.auto_destroy = true;
    _ = try playSound(self, allocator, path, opt);
}

/// Fire-and-forget in-memory sound playback.
pub fn playSoundOnceFromMemory(
    self: anytype,
    allocator: std.mem.Allocator,
    bytes: []const u8,
    format: StreamFormat,
    options: StreamOptions,
) !void {
    var opt = options;
    opt.loop = false;
    opt.auto_destroy = true;
    _ = try playSoundFromMemory(self, allocator, bytes, format, opt);
}

/// Stops a sound stream immediately, or over `fade_duration` seconds.
pub fn stopSound(self: anytype, stream: *AudioStream, fade_duration: f32) void {
    if (fade_duration > 0.0) {
        stream.fadeTo(0.0, fade_duration, true);
    } else {
        destroyStream(self, stream);
    }
}

/// Pauses an active sound stream.
pub fn pauseSound(self: anytype, stream: *AudioStream) void {
    _ = self;
    stream.pause();
}

/// Resumes a paused sound stream.
pub fn resumeSound(self: anytype, stream: *AudioStream) void {
    _ = self;
    stream.unpause();
}

/// Sets the volume of an active sound stream.
pub fn setSoundVolume(self: anytype, stream: *AudioStream, vol: f32) void {
    _ = self;
    stream.setVolume(vol);
}

/// Stops all currently active sound streams.
pub fn stopAllSounds(self: anytype, fade_duration: f32) void {
    for (0..types.max_streams) |i| {
        const s = getSlot(self, i) orelse continue;
        if (fade_duration > 0.0) {
            s.fadeTo(0.0, fade_duration, true);
        } else {
            destroyStream(self, s);
        }
    }
}

/// Crossfades from an existing sound stream to a new sound file.
pub fn crossfadeSound(
    self: anytype,
    old_stream: ?*AudioStream,
    allocator: std.mem.Allocator,
    path: []const u8,
    fade_duration: f32,
    options: StreamOptions,
) !*AudioStream {
    if (old_stream) |s| {
        s.fadeTo(0.0, fade_duration, true);
    }

    var opt = options;
    const target_vol = opt.volume;
    opt.volume = 0.0;
    const new_s = try createStreamFromFile(self, allocator, path, opt);
    new_s.fadeTo(target_vol, fade_duration, false);
    return new_s;
}

/// Crossfades from an existing sound stream to a new in-memory sound.
pub fn crossfadeSoundFromMemory(
    self: anytype,
    old_stream: ?*AudioStream,
    allocator: std.mem.Allocator,
    bytes: []const u8,
    format: StreamFormat,
    fade_duration: f32,
    options: StreamOptions,
) !*AudioStream {
    if (old_stream) |s| {
        s.fadeTo(0.0, fade_duration, true);
    }

    var opt = options;
    const target_vol = opt.volume;
    opt.volume = 0.0;
    const new_s = try createStreamFromMemory(self, allocator, bytes, format, opt);
    new_s.fadeTo(target_vol, fade_duration, false);
    return new_s;
}

// ========================================================================
// Dedicated Music Slot Convenience Helpers
// ========================================================================

/// Convenience helper for playing background music / looping track.
pub fn playMusic(
    self: anytype,
    allocator: std.mem.Allocator,
    path: []const u8,
    options: StreamOptions,
) !*AudioStream {
    if (self.music_stream) |old_s| {
        destroyStream(self, old_s);
        self.music_stream = null;
    }
    var opt = options;
    opt.loop = true;
    const s = try playSound(self, allocator, path, opt);
    self.music_stream = s;
    return s;
}

pub fn playMusicFromMemory(
    self: anytype,
    allocator: std.mem.Allocator,
    bytes: []const u8,
    format: StreamFormat,
    options: StreamOptions,
) !*AudioStream {
    if (self.music_stream) |old_s| {
        destroyStream(self, old_s);
        self.music_stream = null;
    }
    var opt = options;
    opt.loop = true;
    const s = try playSoundFromMemory(self, allocator, bytes, format, opt);
    self.music_stream = s;
    return s;
}

pub fn crossfadeMusic(
    self: anytype,
    allocator: std.mem.Allocator,
    path: []const u8,
    fade_duration: f32,
    options: StreamOptions,
) !*AudioStream {
    const old_s = self.music_stream;
    self.music_stream = null;
    var opt = options;
    opt.loop = true;
    const new_s = try crossfadeSound(self, old_s, allocator, path, fade_duration, opt);
    self.music_stream = new_s;
    return new_s;
}

pub fn stopMusic(self: anytype, fade_duration: f32) void {
    if (self.music_stream) |s| {
        stopSound(self, s, fade_duration);
        self.music_stream = null;
    }
}

pub fn pauseMusic(self: anytype) void {
    if (self.music_stream) |s| s.pause();
}

pub fn resumeMusic(self: anytype) void {
    if (self.music_stream) |s| s.unpause();
}

pub fn setMusicVolume(self: anytype, vol: f32) void {
    if (self.music_stream) |s| s.setVolume(vol);
}

pub fn getMusicStream(self: anytype) ?*AudioStream {
    return self.music_stream;
}
