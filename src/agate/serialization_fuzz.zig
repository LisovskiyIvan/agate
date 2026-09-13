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

// ---------------------------------------------------------------------------
// Allocation-failure, two contracts:
// 1. capture/serialize/deserialize are strict: every allocation fails in
//    turn, the call must return OOM and free everything it already took.
// 2. restore is best-effort BY DESIGN (names degrade via `catch null`,
//    ownership transfers into the scene), so it cannot pass the strict
//    checker (OOM is swallowed, transferred memory outlives the call).
//    For it we only assert crash-safety: a spread of fail indices runs the
//    full pipeline through an arena (which also cleans up whatever restore
//    managed to allocate).
// ---------------------------------------------------------------------------

fn snapshotAllocDense(alloc: std.mem.Allocator) !void {
    var scene = fzg.testScene(alloc);
    var mesh = fzg.testMesh("alloc-fail");
    mesh.position = (fzg.vec3)(1, 2, 3);
    try scene.meshes.append(alloc, &mesh);
    defer scene.meshes.deinit(alloc);

    var state = try serialization.capture(alloc, &scene);
    defer state.deinit(alloc);

    const bytes = try serialization.serializeAlloc(alloc, &state);
    defer alloc.free(bytes);

    var parsed = try serialization.deserializeAlloc(alloc, bytes);
    defer parsed.deinit(alloc);
}

test "alloc-failure: capture/serialize/deserialize free everything on OOM" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, snapshotAllocDense, .{});
}

fn snapshotAllocCount() usize {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    snapshotAllocDense(failing.allocator()) catch {};
    return failing.alloc_index;
}

test "restore survives allocation failures without crashing" {
    const total = snapshotAllocCount();
    for ([_]usize{ 0, total / 3, 2 * total / 3, total -| 1 }) |fail_index| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();

        var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = fail_index });
        const scoped = failing.allocator();

        var scene = fzg.testScene(scoped);
        var mesh = fzg.testMesh("alloc-fail");
        scene.meshes.append(scoped, &mesh) catch continue;

        var state = serialization.capture(scoped, &scene) catch continue;
        const bytes = serialization.serializeAlloc(scoped, &state) catch {
            state.deinit(scoped);
            continue;
        };
        var parsed = serialization.deserializeAlloc(scoped, bytes) catch {
            state.deinit(scoped);
            continue;
        };
        serialization.restore(&scene, &parsed);

        parsed.deinit(scoped);
        state.deinit(scoped);
    }
}
