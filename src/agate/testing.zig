//! Shared test fixtures and helpers (test-only; never compiled into the
//! library facade — the generated tests.zig imports it like any other
//! module).
//!
//! Contents:
//! - CPU-only `Scene`/`Mesh` fixtures for tests that exercise subsystems
//!   touching only the allocator (capture/restore, material loading, ...).
//!   These replace the per-file `testScene`/`testMesh` copies: the fields
//!   left `undefined` are GPU-backed objects the tested code never
//!   dereferences, and `std.mem.zeroes` cannot build them in Zig 0.16
//!   (non-nullable pointer fields).
//! - Comptime corpus mutators for the `std.testing.fuzz` smoke passes
//!   (see loader/obj_fuzz.zig and siblings): valid seed + truncations +
//!   byte flips + garbage, all derived at compile time.

const std = @import("std");
const Scene = @import("scene.zig").Scene;
const Mesh = @import("mesh.zig").Mesh;

/// Minimal CPU-only Scene for subsystem tests (serialization capture/restore,
/// material loading): only `.allocator` is dereferenced. The default textures
/// and render passes are GPU-backed and stay `undefined`.
pub fn testScene(alloc: std.mem.Allocator) Scene {
    return .{
        .allocator = alloc,
        .default_white_texture = undefined,
        .default_normal_texture = undefined,
        .default_cube_texture = undefined,
        .lights = .{},
        .shadows = .{ .pass = undefined },
        .sky = .{ .pass = undefined },
        .postfx = .{
            .postprocess_pass = undefined,
            .ssao_pass = undefined,
            .bloom_pass = undefined,
            .outline_pass = undefined,
        },
        .forward = .{},
        .particles = .{ .pass = undefined },
    };
}

/// Buffer-free Mesh for tests that only touch name/transform/visibility/
/// material fields (serialization): sokol buffers stay empty.
pub fn testMesh(name: []const u8) Mesh {
    return .{
        .name = name,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
}

// ---------------------------------------------------------------------------
// Comptime fuzz corpus builders: each turns a valid seed into the standard
// malformed family (truncate at a power-of-two-ish spread + single-byte flips
// at structural offsets + garbage with a valid magic prefix). Entries are
// comptime-known, so the whole corpus is a plain array literal.
// ---------------------------------------------------------------------------

/// A corpus entry: comptime bytes with a stable name for failure reports.
pub const Corpus = []const []const u8;

/// `seed` truncated at each of `offsets` (clamped to seed length).
pub fn truncations(comptime seed: []const u8, comptime offsets: []const usize) Corpus {
    comptime var out: Corpus = &.{};
    inline for (offsets) |raw_off| {
        const off = @min(raw_off, seed.len);
        out = out ++ &[_][]const u8{seed[0..off]};
    }
    return out;
}

/// `seed` with one byte XORed at each of `offsets` (clamped; empty seed wins).
pub fn flips(comptime seed: []const u8, comptime offsets: []const usize) Corpus {
    comptime var out: Corpus = &.{};
    inline for (offsets) |raw_off| {
        const off = @min(raw_off, if (seed.len == 0) 0 else seed.len - 1);
        var copy: [seed.len]u8 = seed[0..].*;
        if (seed.len > 0) copy[off] ^= 0xff;
        out = out ++ &[_][]const u8{&copy};
    }
    return out;
}

/// `magic ++ garbage`: header-valid garbage, the class most likely to slip
/// past early validation and reach the payload decoder.
pub fn magicGarbage(comptime magic: []const u8, comptime garbage: []const u8) Corpus {
    return &[_][]const u8{magic ++ garbage};
}

/// Concatenates corpus groups into one.
pub fn join(a: Corpus, b: Corpus) Corpus {
    return a ++ b;
}
