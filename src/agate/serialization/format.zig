//! Binary format primitives for the AGSC scene snapshot.
//!
//! Header/version constants, hard caps for untrusted input, the little-endian
//! Writer/Reader pair, the comptime options-field codec, and the
//! PostProcessOptions persistence contract (field list IS the byte order).
//! No scene knowledge here; see writer.zig (scene -> bytes) and reader.zig
//! (bytes -> scene). Layout documented in full on serializeAlloc/writer side;
//! summary: magic[4]="AGSC", version u32 (2..3), mesh/light/camera/render/
//! postprocess/game-property sections, all integers LE, f32 as IEEE754 LE
//! bits, bool as u8 0/1, string as u32 length + raw bytes, no padding.

const std = @import("std");
const PostProcessOptions = @import("../postprocess.zig").PostProcessOptions;

pub const MAGIC: [4]u8 = .{ 'A', 'G', 'S', 'C' };
/// v3: adds mesh stable entity `id` (u64) and `parent_name` (string) for hierarchy
/// persistence, plus custom game key-value properties table at file tail.
/// Backwards-compatible: v2 files parse cleanly without parent/game-properties.
pub const VERSION: u32 = 3;

/// Hard caps for untrusted input. Counts above max_entries and strings above
/// max_string_bytes report TooLarge instead of driving wild allocations.
pub const MAX_ENTRIES: u32 = 1_000_000;
pub const MAX_STRING_BYTES: u32 = 8 * 1024 * 1024;
pub const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;

pub const DecodeError = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
    TooLarge,
};

pub const Writer = struct {
    alloc: std.mem.Allocator,
    buf: std.ArrayListUnmanaged(u8) = .empty,

    pub fn bytes(self: *Writer, data: []const u8) !void {
        try self.buf.appendSlice(self.alloc, data);
    }

    pub fn byte(self: *Writer, v: u8) !void {
        try self.buf.append(self.alloc, v);
    }

    pub fn u32le(self: *Writer, v: u32) !void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, v, .little);
        try self.bytes(&b);
    }

    pub fn u64le(self: *Writer, v: u64) !void {
        var b: [8]u8 = undefined;
        std.mem.writeInt(u64, &b, v, .little);
        try self.bytes(&b);
    }

    pub fn f32le(self: *Writer, v: f32) !void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, @as(u32, @bitCast(v)), .little);
        try self.bytes(&b);
    }

    pub fn bool8(self: *Writer, v: bool) !void {
        try self.byte(if (v) 1 else 0);
    }

    pub fn vec3(self: *Writer, v: [3]f32) !void {
        try self.f32le(v[0]);
        try self.f32le(v[1]);
        try self.f32le(v[2]);
    }

    pub fn str(self: *Writer, s: []const u8) !void {
        const len = std.math.cast(u32, s.len) orelse return error.TooLarge;
        try self.u32le(len);
        try self.bytes(s);
    }
};

pub const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    pub fn readU8(self: *Reader) DecodeError!u8 {
        if (self.pos >= self.bytes.len) return error.Truncated;
        const v = self.bytes[self.pos];
        self.pos += 1;
        return v;
    }

    pub fn readU32(self: *Reader) DecodeError!u32 {
        const raw = try self.readRaw(4);
        return std.mem.readInt(u32, raw[0..4], .little);
    }

    pub fn readU64(self: *Reader) DecodeError!u64 {
        const raw = try self.readRaw(8);
        return std.mem.readInt(u64, raw[0..8], .little);
    }

    pub fn readF32(self: *Reader) DecodeError!f32 {
        const raw = try self.readRaw(4);
        return @as(f32, @bitCast(std.mem.readInt(u32, raw[0..4], .little)));
    }

    pub fn readBool(self: *Reader) DecodeError!bool {
        return switch (try self.readU8()) {
            0 => false,
            1 => true,
            else => error.Truncated,
        };
    }

    pub fn readRaw(self: *Reader, n: u32) DecodeError![]const u8 {
        const len = std.math.cast(usize, n) orelse return error.TooLarge;
        const end = std.math.add(usize, self.pos, len) catch return error.TooLarge;
        if (end > self.bytes.len) return error.Truncated;
        const out = self.bytes[self.pos..end];
        self.pos = end;
        return out;
    }

    /// Validated element count for a variable-length list.
    pub fn readCount(self: *Reader) DecodeError!u32 {
        const n = try self.readU32();
        if (n > MAX_ENTRIES) return error.TooLarge;
        return n;
    }

    pub fn readString(self: *Reader, allocator: std.mem.Allocator) (DecodeError || std.mem.Allocator.Error)![]u8 {
        const n = try self.readU32();
        if (n > MAX_STRING_BYTES) return error.TooLarge;
        const raw = try self.readRaw(n);
        return allocator.dupe(u8, raw);
    }

    pub fn readVec3(self: *Reader) DecodeError![3]f32 {
        return .{ try self.readF32(), try self.readF32(), try self.readF32() };
    }
};

/// Comptime field codec for plain options structs. The explicit
/// `persisted` name list IS the on-disk contract: list order == byte
/// order, and fields absent from the list stay session-local instead of
/// silently changing the format. Field types dispatch at comptime:
/// bool -> bool8, f32 -> f32le, u32 -> u32le, [3]f32 -> vec3,
/// enum -> u32le(@intFromEnum). The enum mapping is derived from the
/// enum declaration itself, so a name<->int mismatch between the writer
/// and the reader is impossible by construction.
pub fn writeField(w: *Writer, value: anytype) !void {
    const T = @TypeOf(value);
    if (T == bool) return w.bool8(value);
    if (T == f32) return w.f32le(value);
    if (T == u32) return w.u32le(value);
    if (T == [3]f32) return w.vec3(value);
    if (@typeInfo(T) == .@"enum") return w.u32le(@intFromEnum(value));
    @compileError("serialization: unsupported options field type " ++ @typeName(T));
}

pub fn readFieldAs(comptime T: type, r: *Reader) DecodeError!T {
    if (T == bool) return r.readBool();
    if (T == f32) return r.readF32();
    if (T == u32) return r.readU32();
    if (T == [3]f32) return r.readVec3();
    if (@typeInfo(T) == .@"enum") {
        const raw = try r.readU32();
        inline for (@typeInfo(T).@"enum".fields) |f| {
            if (f.value == raw) return @enumFromInt(f.value);
        }
        return error.Truncated;
    }
    @compileError("serialization: unsupported options field type " ++ @typeName(T));
}

pub fn writeOptions(w: *Writer, options: anytype, comptime persisted: []const []const u8) !void {
    inline for (persisted) |name| {
        try writeField(w, @field(options, name));
    }
}

pub fn readOptions(comptime T: type, r: *Reader, comptime persisted: []const []const u8) DecodeError!T {
    var out: T = .{};
    inline for (persisted) |name| {
        @field(out, name) = try readFieldAs(@TypeOf(@field(out, name)), r);
    }
    return out;
}

/// PostProcessOptions fields persisted in save states, in byte order.
/// Anything not listed here (transient knobs like bloom_pyramid, dof_*,
/// grade_*) is rebuilt from defaults on load.
pub const postprocess_persisted = [_][]const u8{
    "enabled",              "exposure",           "tonemapping",
    "bloom_enabled",        "bloom_threshold",    "bloom_intensity",
    "bloom_radius",         "vignette_enabled",   "vignette_intensity",
    "vignette_radius",      "saturation",         "contrast",
    "chromatic_aberration", "fxaa_enabled",       "fog_enabled",
    "fog_density",          "fog_height_falloff", "fog_start_distance",
    "fog_color",            "fog_sun_scattering", "ssr_enabled",
    "ssr_intensity",        "ssr_max_distance",   "ssr_thickness",
    "sharpen_enabled",      "sharpen_amount",     "grain_enabled",
    "grain_intensity",      "temperature",        "tint",
};

pub fn writePostProcess(w: *Writer, pp: *const PostProcessOptions) !void {
    try writeOptions(w, pp.*, &postprocess_persisted);
}

pub fn readPostProcess(r: *Reader) DecodeError!PostProcessOptions {
    return readOptions(PostProcessOptions, r, &postprocess_persisted);
}
