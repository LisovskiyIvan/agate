//! Fuzz/robustness target for the AGSC snapshot deserializer
//! (serialization.deserializeAlloc) — the entry point that consumes untrusted
//! bytes (loaded scene files).
//!
//! Invariant: any input yields an error or a fully owned SceneState that
//! deinit() releases — no panic, no UB, no leaks, no double frees
//! (std.testing.allocator backs every iteration).
//!
//! The seed is a real snapshot (empty state, serialized by the engine itself),
//! serialized at test time so the corpus always matches the CURRENT format
//! version; truncations and byte flips then attack counts, string lengths and
//! float payloads. Run notes in build.zig.

const std = @import("std");
const serialization = @import("serialization.zig");
const fzg = @import("testing.zig");

fn testOne(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var generated: [8192]u8 = undefined;
    const input = fzg.fuzzInput(smith, &generated);

    var state = serialization.deserializeAlloc(std.testing.allocator, input) catch return;
    state.deinit(std.testing.allocator);
}

test "fuzz: deserializeAlloc survives arbitrary bytes" {
    const alloc = std.testing.allocator;

    // Valid seed: an empty-state snapshot in the current format.
    const empty_state = serialization.SceneState{};
    const seed = try serialization.serializeAlloc(alloc, &empty_state);
    defer alloc.free(seed);

    var corpus: std.ArrayListUnmanaged([]const u8) = .empty;
    defer {
        for (corpus.items) |entry| alloc.free(entry);
        corpus.deinit(alloc);
    }
    // Every entry is an owned copy — seed included: the loop below frees all
    // entries, and the seed's own defer frees it a second time otherwise.
    try corpus.append(alloc, try alloc.dupe(u8, seed));
    // Truncations: cut inside the magic, the version field and right after it.
    for ([_]usize{ 1, 2, 4, 8, seed.len - 1 }) |off| {
        const n = @min(off, seed.len);
        try corpus.append(alloc, try alloc.dupe(u8, seed[0..n]));
    }
    // Flips: magic, version, mesh_count, first length field.
    for ([_]usize{ 0, 2, 4, 6, 8, 10 }) |off| {
        if (off >= seed.len) continue;
        const copy = try alloc.dupe(u8, seed);
        copy[off] ^= 0xff;
        try corpus.append(alloc, copy);
    }
    // Header-valid garbage: right magic and version, hostile counts.
    const garbage = try alloc.alloc(u8, 32);
    defer alloc.free(garbage);
    @memset(garbage, 0xff);
    @memcpy(garbage[0..4], "AGSC");
    std.mem.writeInt(u32, garbage[4..8], serialization.VERSION, .little);
    try corpus.append(alloc, try alloc.dupe(u8, garbage));

    try std.testing.fuzz({}, testOne, .{ .corpus = corpus.items });
}
