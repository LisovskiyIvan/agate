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
//! `pauseMusic`, `resumeMusic`, `setMusicVolume`, `getMusicStream`,
//! `MusicPlayOptions`, `playMusicClip`). Decoding/streaming itself lives
//! in the pre-existing `stream.zig` leaf; this module only manages slots.
//!
//! Anti-cycle rule (same as `profiler/*`, `particles/*`): every function
//! takes the engine as `anytype` (a `*AudioEngine` from `engine.zig` in
//! practice) and this module never imports `engine.zig` or the `audio.zig`
//! facade back. `playMusicClip` reaches the `buses.zig` (`stopBus`) and
//! `playback.zig` (`playClip`) siblings directly (documented in the
//! facade); everything else is fields or the `stream.zig` leaf.

const std = @import("std");
const clip_mod = @import("clip.zig");
const stream_mod = @import("stream.zig");
const types = @import("types.zig");
const buses = @import("buses.zig");
const playback = @import("playback.zig");

const BusId = types.BusId;
const AudioClip = clip_mod.AudioClip;
const StreamFormat = stream_mod.StreamFormat;
const StreamOptions = stream_mod.StreamOptions;
const PlaySoundOptions = stream_mod.PlaySoundOptions;
const AudioStream = stream_mod.AudioStream;

// Local aliases so the moved bodies stay byte-identical.
const stopBus = buses.stopBus;
const playClip = playback.playClip;

pub fn registerStream(self: anytype, stream: *AudioStream) !void {
    for (&self.streams) |*slot| {
        if (slot.* == null) {
            slot.* = stream;
            return;
        }
    }
    return error.StreamLimitReached;
}

pub fn unregisterStream(self: anytype, stream: *AudioStream) void {
    for (&self.streams) |*slot| {
        if (slot.* == stream) {
            slot.* = null;
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

pub fn destroyStream(self: anytype, stream: *AudioStream) void {
    unregisterStream(self, stream);
    stream.deinit();
}

pub fn updateStreams(self: anytype, dt: f32) void {
    for (&self.streams) |*slot| {
        if (slot.*) |s| {
            s.update(dt);
            if (s.isStopped() and (s.auto_destroy or s.stop_on_fade_out)) {
                if (self.music_stream == s) self.music_stream = null;
                if (self.music_fade_stream == s) self.music_fade_stream = null;
                slot.* = null;
                s.deinit();
            }
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
    options: PlaySoundOptions,
) !*AudioStream {
    return createStreamFromFile(self, allocator, path, options);
}

/// Plays a sound from an in-memory byte slice (OGG, MP3, WAV).
pub fn playSoundFromMemory(
    self: anytype,
    allocator: std.mem.Allocator,
    bytes: []const u8,
    format: StreamFormat,
    options: PlaySoundOptions,
) !*AudioStream {
    return createStreamFromMemory(self, allocator, bytes, format, options);
}

/// Fire-and-forget sound playback: automatically destroys and frees the stream
/// when playback finishes or is stopped.
pub fn playSoundOnce(
    self: anytype,
    allocator: std.mem.Allocator,
    path: []const u8,
    options: PlaySoundOptions,
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
    options: PlaySoundOptions,
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
    for (&self.streams) |maybe_s| {
        if (maybe_s) |s| {
            if (fade_duration > 0.0) {
                s.fadeTo(0.0, fade_duration, true);
            } else {
                destroyStream(self, s);
            }
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
    options: PlaySoundOptions,
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
    options: PlaySoundOptions,
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
    options: PlaySoundOptions,
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
    options: PlaySoundOptions,
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
    options: PlaySoundOptions,
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

pub const MusicPlayOptions = struct {
    bus: ?BusId = null,
    volume: f32 = 1.0,
    loop: bool = true,
    rate: f32 = 1.0,
};

pub fn playMusicClip(self: anytype, clip: *const AudioClip, options: MusicPlayOptions) void {
    if (options.bus) |b| {
        stopBus(self, b);
    }
    playClip(self, clip, .{
        .bus = options.bus,
        .volume = options.volume,
        .loop = options.loop,
        .rate = options.rate,
        .position = null,
    });
}
