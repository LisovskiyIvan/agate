//! Tests for `assets.zig` (moved from `assets.zig` inline blocks).
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const jobs = @import("jobs.zig");
const gpu_thread = @import("gpu_thread.zig");
const testing = std.testing;
const assets = @import("assets.zig");
const ktx2 = assets.ktx2;
const TextureState = assets.TextureState;
const PendingTexture = assets.PendingTexture;
const uploadBudgetExhausted = assets.uploadBudgetExhausted;
const UploadQueue = assets.UploadQueue;
const Texture = @import("texture.zig").Texture;

fn waitForState(p: *PendingTexture, comptime states: []const TextureState) bool {
    // Bounded spin: the states under test are reached in microseconds.
    var i: usize = 0;
    while (i < 50_000_000) : (i += 1) {
        const st = p.state.load(.acquire);
        for (states) |want| {
            if (st == want) return true;
        }
        std.atomic.spinLoopHint();
    }
    return false;
}

/// Repo-relative font bitmap probe (same locations as the PNG test above).
fn findFontPng() ?[]const u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const candidates = [_][]const u8{
        "src/assets/font_sdf.png",
        "agate/src/assets/font_sdf.png",
        "../agate/src/assets/font_sdf.png",
        "src/agate/assets/font_sdf.png",
        "agate/src/agate/assets/font_sdf.png",
        "../agate/src/agate/assets/font_sdf.png",
    };
    for (candidates) |candidate| {
        const f = std.Io.Dir.cwd().openFile(io, candidate, .{}) catch continue;
        f.close(io);
        return candidate;
    }
    return null;
}

test "requestFile reports missing files through the worker" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    const p = try queue.requestFile("/definitely/missing/image.png", .{}, .{});
    try testing.expect(waitForState(p, &.{.failed}));
    try testing.expect(p.err != null);

    // Failed slots never upload and never take.
    try testing.expectEqual(@as(usize, 0), queue.drain());
    try testing.expect(p.take() == null);

    queue.release(p);
}

test "real PNG decodes to ready off-thread; drain reports nothing without sg" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    // Repo-relative font bitmap: real decode work on the worker thread.
    // (.ready is the terminal state reachable without a GPU context; the
    // sg upload path itself is exercised by every app boot via fromRaw.)
    // @src().file yields only the basename under this build, so probe the
    // known repo-relative locations for the test's cwd.
    const font_candidates = [_][]const u8{
        "src/assets/font_sdf.png",
        "agate/src/assets/font_sdf.png",
        "../agate/src/assets/font_sdf.png",
        "src/agate/assets/font_sdf.png",
        "agate/src/agate/assets/font_sdf.png",
        "../agate/src/agate/assets/font_sdf.png",
    };
    // Same Io pattern as Texture.decodeFile (Zig 0.16 removed std.fs.cwd).
    const io = std.Io.Threaded.global_single_threaded.io();
    var path: []const u8 = "";
    for (font_candidates) |candidate| {
        const f = std.Io.Dir.cwd().openFile(io, candidate, .{}) catch continue;
        f.close(io);
        path = candidate;
        break;
    }
    try testing.expect(path.len > 0);

    const p = try queue.requestFile(path, .{}, .{ .gen_mipmaps = true });
    try testing.expect(waitForState(p, &.{ .ready, .failed }));
    try testing.expectEqual(TextureState.ready, p.state.load(.acquire));
    try testing.expect(p.raw.width > 0);
    try testing.expect(p.raw.levels[0] != null);

    // No sg context in tests: drain cannot run here; the ready decode is
    // torn down by queue deinit.
}

test "drain is a no-op on an empty queue" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();
    try testing.expectEqual(@as(usize, 0), queue.drain());
}

test "shutdown with in-flight decodes frees without use-after-free" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 2);
    // Missing files: fast worker tasks still in flight when deinit runs.
    // DebugAllocator + join-on-shutdown catch any racing write-after-free.
    for (0..16) |i| {
        var buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "/missing/{d}.png", .{i});
        _ = try queue.requestFile(path, .{}, .{});
    }
    queue.deinit();
}

test "many parallel loads all report terminal states" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 2);
    defer queue.deinit();

    var slots: [8]*PendingTexture = undefined;
    for (&slots) |*slot| {
        slot.* = try queue.requestFile("/definitely/missing/image.png", .{}, .{});
    }
    for (slots) |p| {
        try testing.expect(waitForState(p, &.{.failed}));
    }
    for (slots) |p| queue.release(p);
    try testing.expectEqual(@as(usize, 0), queue.drain());
}

test "drainBudget paces ready slots across calls without an sg context" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    const path = findFontPng();
    try testing.expect(path != null);

    var slots: [3]*PendingTexture = undefined;
    for (&slots) |*slot| {
        slot.* = try queue.requestFile(path.?, .{}, .{ .gen_mipmaps = true });
    }
    for (slots) |p| {
        try testing.expect(waitForState(p, &.{.ready}));
    }

    // No sg context in tests: bounded and unbounded drains upload nothing
    // and leave every .ready slot intact for a later real drain.
    try testing.expectEqual(@as(usize, 0), queue.drainBudget(1));
    try testing.expectEqual(@as(usize, 0), queue.drainBudget(2));
    try testing.expectEqual(@as(usize, 0), queue.drain());
    const no_result = queue.drainCounted(1);
    try testing.expectEqual(@as(usize, 0), no_result.count);
    try testing.expectEqual(@as(u64, 0), no_result.bytes);
    for (slots) |p| {
        try testing.expectEqual(TextureState.ready, p.state.load(.acquire));
    }

    // Budget pacing at the selection level (the fromRaw upload itself needs
    // a real sg context and is exercised by every app boot): collect at most
    // `max` ready slots per call. Each simulated upload — raw freed plus
    // `.uploaded` stored, exactly what drainCounted does after fromRaw —
    // shrinks the next collection. 3 ready, budget 1 -> 1, then 1, then 1.
    var batch: [3]*PendingTexture = undefined;

    try testing.expectEqual(@as(usize, 1), queue.collectReady(batch[0..1]));
    try testing.expect(batch[0] == slots[0]);
    batch[0].raw.deinit(a);
    batch[0].state.store(.uploaded, .release);

    try testing.expectEqual(@as(usize, 1), queue.collectReady(batch[0..1]));
    try testing.expect(batch[0] == slots[1]);
    batch[0].raw.deinit(a);
    batch[0].state.store(.uploaded, .release);

    // A budget larger than the remainder collects just the remainder.
    try testing.expectEqual(@as(usize, 1), queue.collectReady(batch[0..2]));
    try testing.expect(batch[0] == slots[2]);
    batch[0].raw.deinit(a);
    batch[0].state.store(.uploaded, .release);

    try testing.expectEqual(@as(usize, 0), queue.collectReady(batch[0..1]));
    try testing.expectEqual(@as(usize, 0), queue.collectReady(batch[0..3]));
}

test "unbounded collect takes every ready slot and skips the rest" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    const path = findFontPng();
    try testing.expect(path != null);

    var slots: [3]*PendingTexture = undefined;
    for (&slots) |*slot| {
        slot.* = try queue.requestFile(path.?, .{}, .{ .gen_mipmaps = true });
    }
    const failed = try queue.requestFile("/definitely/missing/image.png", .{}, .{});
    for (slots) |p| {
        try testing.expect(waitForState(p, &.{.ready}));
    }
    try testing.expect(waitForState(failed, &.{.failed}));

    var batch: [3]*PendingTexture = undefined;

    // Zero budget collects nothing even with ready slots pending.
    try testing.expectEqual(@as(usize, 0), queue.collectReady(batch[0..0]));

    // A full-width collection takes every ready slot, never the failed one.
    const n = queue.collectReady(batch[0..3]);
    try testing.expectEqual(@as(usize, 3), n);
    for (batch[0..n]) |p| {
        try testing.expect(p != failed);
        try testing.expectEqual(TextureState.ready, p.state.load(.acquire));
    }

    // Mirror drainCounted's post-upload teardown so deinit frees cleanly.
    for (batch[0..n]) |p| {
        p.raw.deinit(a);
        p.state.store(.uploaded, .release);
    }
    queue.release(failed);
}

test "UploadQueue.getOrRequestFile deduplicates in-flight and uploaded textures" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    const path = findFontPng();
    try testing.expect(path != null);

    const p1 = try queue.getOrRequestFile(path.?, .{}, .{});
    const p2 = try queue.getOrRequestFile(path.?, .{}, .{});
    try testing.expectEqual(p1, p2);

    const found = queue.findFile(path.?);
    try testing.expectEqual(p1, found);

    try testing.expect(waitForState(p1, &.{.ready}));

    // Simulating upload to .uploaded state
    p1.raw.deinit(a);
    p1.state.store(.uploaded, .release);

    // After upload, requesting the same path again still returns the same slot
    const p3 = try queue.getOrRequestFile(path.?, .{}, .{});
    try testing.expectEqual(p1, p3);
}

test "requestMemory failure frees input before publishing failed" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    const garbage = try a.dupe(u8, "this is definitely not a valid image payload!!!");
    const p = queue.requestMemory(garbage, .{}, .{}) catch |e| {
        a.free(garbage);
        return e;
    };
    try testing.expect(waitForState(p, &.{.failed}));
    // Input bytes must be freed/nulled before the terminal store, while the
    // slot is still alive (release destroys it).
    try testing.expect(p.memory == null);
    queue.release(p);
    try testing.expectEqual(@as(usize, 0), queue.drain());
}

test "requestMemory routes block KTX2 to the block upload path" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    // Minimal BC7 4x4 container: 80-byte header+index, one 24-byte level
    // entry, one 16-byte block payload. (Full spec-shaped fixtures live in
    // ktx2.zig; this only exercises the queue routing decision.)
    var file: [80 + 24 + 16]u8 = [_]u8{0} ** (80 + 24 + 16);
    @memcpy(file[0..12], &[12]u8{ 0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A });
    std.mem.writeInt(u32, file[12..16], 145, .little); // vkFormat BC7_UNORM_BLOCK
    std.mem.writeInt(u32, file[16..20], 1, .little); // typeSize
    std.mem.writeInt(u32, file[20..24], 4, .little); // width
    std.mem.writeInt(u32, file[24..28], 4, .little); // height
    // depth/layers zero, scheme NONE zero.
    std.mem.writeInt(u32, file[36..40], 1, .little); // faceCount
    std.mem.writeInt(u32, file[40..44], 1, .little); // levelCount
    std.mem.writeInt(u64, file[80..88], 104, .little); // byteOffset
    std.mem.writeInt(u64, file[88..96], 16, .little); // byteLength
    std.mem.writeInt(u64, file[96..104], 16, .little); // uncompressedByteLength
    for (file[104..], 0..) |*b, i| b.* = @intCast(i);

    const owned = try a.dupe(u8, &file);
    const p = try queue.requestMemory(owned, .{}, .{});
    try testing.expect(waitForState(p, &.{.ready}));
    try testing.expect(p.memory == null); // input freed before publish
    try testing.expect(p.block_raw != null);
    try testing.expectEqual(ktx2.BlockFormat.bc7_unorm, p.block_raw.?.format);
    try testing.expectEqual(@as(u32, 4), p.block_raw.?.width);
    try testing.expectEqual(@as(usize, 16), p.block_raw.?.totalBytes());
    // The RGBA8 side stays empty for block files (no double decode).
    try testing.expectEqual(@as(u32, 0), p.raw.num_levels);
    // No sg context in tests: the .ready block decode is torn down by
    // queue deinit (which must free block_raw without leaking).

    // A supercompressed block file fails decode with the ktx2 reason and
    // never populates either pixel side.
    var zstd_file = file;
    std.mem.writeInt(u32, zstd_file[44..48], 1, .little); // BasisLZ scheme
    const zstd_owned = try a.dupe(u8, &zstd_file);
    const q = try queue.requestMemory(zstd_owned, .{}, .{});
    try testing.expect(waitForState(q, &.{.failed}));
    try testing.expect(q.err != null);
    try testing.expect(q.block_raw == null);
    try testing.expectEqual(@as(u32, 0), q.raw.num_levels);
    queue.release(q);
}

test "requestMemory decodes font PNG and frees input before publish" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    const path = findFontPng();
    try testing.expect(path != null);

    // Same Io pattern as Texture.decodeFile (Zig 0.16 removed std.fs.cwd).
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, path.?, .{});
    defer file.close(io);
    const file_size = try file.length(io);
    const bytes = try a.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
    const read = try file.readPositionalAll(io, bytes, 0);
    try testing.expectEqual(bytes.len, read);

    const p = queue.requestMemory(bytes, .{}, .{ .gen_mipmaps = true }) catch |e| {
        a.free(bytes);
        return e;
    };
    try testing.expect(waitForState(p, &.{.ready}));
    try testing.expect(p.memory == null);
    try testing.expect(p.raw.width > 0);
    // No sg context in tests: the .ready decode is torn down by queue deinit.
}

test "uploadBudgetExhausted gates byte pacing without an sg context" {
    // Null budget never exhausts, even with uploads already tallied.
    try testing.expect(!uploadBudgetExhausted(0, 0, null));
    try testing.expect(!uploadBudgetExhausted(4, 64 * 1024 * 1024, null));
    // First-upload (overshoot) rule: count == 0 never stops, so a lone
    // texture bigger than the whole budget still makes progress.
    try testing.expect(!uploadBudgetExhausted(0, 0, 8 * 1024 * 1024));
    try testing.expect(!uploadBudgetExhausted(0, 100 * 1024 * 1024, 8 * 1024 * 1024));
    // Below budget continues once progress exists.
    try testing.expect(!uploadBudgetExhausted(1, 1024, 8 * 1024 * 1024));
    try testing.expect(!uploadBudgetExhausted(2, 8 * 1024 * 1024 - 1, 8 * 1024 * 1024));
    // At budget and above stop.
    try testing.expect(uploadBudgetExhausted(1, 8 * 1024 * 1024, 8 * 1024 * 1024));
    try testing.expect(uploadBudgetExhausted(1, 21 * 1024 * 1024, 8 * 1024 * 1024));
    try testing.expect(uploadBudgetExhausted(3, 9 * 1024 * 1024, 8 * 1024 * 1024));
    // Zero budget still uploads exactly one texture per call, then stops.
    try testing.expect(!uploadBudgetExhausted(0, 0, 0));
    try testing.expect(uploadBudgetExhausted(1, 1, 0));
}

test "byte budgeting paces ready slots across calls without an sg context" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    const path = findFontPng();
    try testing.expect(path != null);

    var slots: [3]*PendingTexture = undefined;
    for (&slots) |*slot| {
        slot.* = try queue.requestFile(path.?, .{}, .{ .gen_mipmaps = true });
    }
    for (slots) |p| {
        try testing.expect(waitForState(p, &.{.ready}));
    }

    // No sg context: the byte-aware drain is the same safe no-op as the
    // count-only one and leaves every .ready slot intact (null byte budget
    // behaves exactly like drainCounted).
    const noop = queue.drainCountedBudget(1, 1024);
    try testing.expectEqual(@as(usize, 0), noop.count);
    try testing.expectEqual(@as(u64, 0), noop.bytes);
    const noop_null = queue.drainCountedBudget(null, null);
    try testing.expectEqual(@as(usize, 0), noop_null.count);
    try testing.expectEqual(@as(u64, 0), noop_null.bytes);
    for (slots) |p| {
        try testing.expectEqual(TextureState.ready, p.state.load(.acquire));
    }

    // Mocked byte sizes: real uploads sum raw mip level lengths while the
    // slot is being uploaded, so measure the same way here.
    var sizes: [3]u64 = undefined;
    for (slots, 0..) |p, i| {
        var bytes: u64 = 0;
        for (p.raw.levels[0..p.raw.num_levels]) |level| {
            if (level) |level_buf| bytes += level_buf.len;
        }
        try testing.expect(bytes > 0);
        sizes[i] = bytes;
    }

    // Budget for exactly one texture: after the mandatory first upload the
    // tally hits the budget, so the simulated frame stops with two leftover
    // .ready slots — mirroring drainCountedBudget's between-upload check.
    const budget: u64 = sizes[0];
    var batch: [1]*PendingTexture = undefined;

    var frame_count: usize = 0;
    var frame_bytes: u64 = 0;
    try testing.expectEqual(@as(usize, 1), queue.collectReady(batch[0..1]));
    try testing.expect(batch[0] == slots[0]);
    frame_bytes += sizes[0];
    frame_count += 1;
    batch[0].raw.deinit(a);
    batch[0].state.store(.uploaded, .release);
    // Budget reached: the next slot must wait for the following call even
    // though it is already collected-ready.
    try testing.expect(uploadBudgetExhausted(frame_count, frame_bytes, budget));
    for (slots[1..]) |p| {
        try testing.expectEqual(TextureState.ready, p.state.load(.acquire));
    }

    // Next call picks up where the previous one stopped: the leftovers are
    // still queued as .ready (overshoot rule restarts the count at zero).
    frame_count = 0;
    frame_bytes = 0;
    try testing.expect(!uploadBudgetExhausted(frame_count, frame_bytes, budget));
    try testing.expectEqual(@as(usize, 1), queue.collectReady(batch[0..1]));
    try testing.expect(batch[0] == slots[1]);
    batch[0].raw.deinit(a);
    batch[0].state.store(.uploaded, .release);
    frame_count += 1;
    frame_bytes += sizes[1];

    // A single texture bigger than the whole remaining budget still uploads
    // (overshoot): the stop decision only applies after progress.
    try testing.expect(uploadBudgetExhausted(frame_count, frame_bytes, sizes[1]));
    try testing.expectEqual(TextureState.ready, slots[2].state.load(.acquire));
    try testing.expectEqual(@as(usize, 1), queue.collectReady(batch[0..1]));
    try testing.expect(batch[0] == slots[2]);
    batch[0].raw.deinit(a);
    batch[0].state.store(.uploaded, .release);

    try testing.expectEqual(@as(usize, 0), queue.collectReady(batch[0..1]));
}

test "requestMemory routes real Basis payloads to the block path off-thread" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    // Real toktx UASTC fixture (see src/agate/ktx2_fixtures/README.md): the
    // worker transcodes to the default .bc7 target without touching sg.
    const fx = @embedFile("ktx2_fixtures/fx_uastc_rgba_mip.ktx2");
    const owned = try a.dupe(u8, fx);
    const p = try queue.requestMemory(owned, .{}, .{});
    try testing.expect(waitForState(p, &.{.ready}));
    try testing.expect(p.memory == null); // input freed before publish
    try testing.expect(p.block_raw != null);
    try testing.expectEqual(ktx2.BlockFormat.bc7_srgb, p.block_raw.?.format);
    try testing.expectEqual(@as(u32, 16), p.block_raw.?.width);
    try testing.expectEqual(@as(u32, 5), p.block_raw.?.num_levels);
    try testing.expectEqual(@as(usize, 256 + 64 + 16 + 16 + 16), p.block_raw.?.totalBytes());
    // The RGBA8 side stays empty for transcoded files (no double decode).
    try testing.expectEqual(@as(u32, 0), p.raw.num_levels);
    // No sg context in tests: the .ready transcode is torn down by queue
    // deinit (which must free block_raw without leaking).

    // The RGBA32 fallback target decodes to the .rgba side instead.
    const owned2 = try a.dupe(u8, fx);
    const q = try queue.requestMemory(owned2, .{}, .{ .basis_target = .rgba32 });
    try testing.expect(waitForState(q, &.{.ready}));
    try testing.expect(q.block_raw == null);
    try testing.expectEqual(@as(u32, 5), q.raw.num_levels);
    try testing.expectEqual(@as(usize, 1024), q.raw.levels[0].?.len);
}

// Headless stand-in for the context drain's publish half (`finishUpload`
// minus the sg upload, which needs a GPU context and is therefore NOT
// covered here): frees the decoded pixels and publishes `.uploaded` with a
// caller-provided texture, leaving game-owned target slots untouched.
fn publishHeadless(p: *PendingTexture, tex: Texture) void {
    p.raw.deinit(testing.allocator);
    p.texture = tex;
    p.state.store(.uploaded, .release);
}

test "drain publishes without touching game slots; game commit patches exactly once" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    const path = findFontPng();
    try testing.expect(path != null);

    const p = try queue.requestFile(path.?, .{}, .{ .gen_mipmaps = true });
    try testing.expect(waitForState(p, &.{.ready}));

    // A game-thread registration racing the in-flight prepare.
    var slot: ?Texture = null;
    p.addTarget(&slot);

    // Headless drain is a no-op: must neither upload nor patch game state.
    try testing.expectEqual(@as(usize, 0), queue.drain());
    try testing.expectEqual(TextureState.ready, p.state.load(.acquire));
    try testing.expect(slot == null);

    // Context publishes (sg upload untestable headless); the slot is still
    // null — the drain side never writes game-owned memory.
    publishHeadless(p, std.mem.zeroes(Texture));
    try testing.expect(slot == null);

    // Game-side commit patches exactly once, then goes quiescent.
    try testing.expectEqual(@as(usize, 1), queue.commitUploadedTargets());
    try testing.expect(slot != null);
    try testing.expectEqual(@as(usize, 0), queue.commitUploadedTargets());
    try testing.expect(slot != null);

    // Late registration after publication patches inline, dedup preserved.
    var late: ?Texture = null;
    p.addTarget(&late);
    try testing.expect(late != null);
    try testing.expectEqual(@as(usize, 0), queue.commitUploadedTargets());

    // take()/release() pairing is unchanged (commit leaves take semantics).
    try testing.expect(p.take() != null);
    queue.release(p);
}

test "failed uploads drop game targets and keep slots null" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    const garbage = try a.dupe(u8, "this is definitely not a valid image payload!!!");
    const p = queue.requestMemory(garbage, .{}, .{}) catch |e| {
        a.free(garbage);
        return e;
    };
    var slot: ?Texture = null;
    p.addTarget(&slot);
    try testing.expect(waitForState(p, &.{.failed}));

    // Failed commit drops the target without patching; slot stays null
    // (default-white), and the slot remains release-ready.
    try testing.expectEqual(@as(usize, 0), queue.commitUploadedTargets());
    try testing.expect(slot == null);
    try testing.expect(p.take() == null);
    queue.release(p);
}

const targetAdderCtx = struct {
    p: *PendingTexture,
    slots: []?Texture,
    go: *std.atomic.Value(bool),
};

fn targetAdder(ctx: *targetAdderCtx) void {
    while (!ctx.go.load(.acquire)) std.atomic.spinLoopHint();
    for (ctx.slots) |*slot| ctx.p.addTarget(slot);
}

const commitRacerCtx = struct {
    q: *UploadQueue,
    sum: *std.atomic.Value(usize),
};

fn commitRacer(ctx: *commitRacerCtx) void {
    const n = ctx.q.commitUploadedTargets();
    _ = ctx.sum.fetchAdd(n, .monotonic);
}

test "concurrent addTarget vs publish/commit loses no target and patches once" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 1);
    defer queue.deinit();

    const path = findFontPng();
    try testing.expect(path != null);

    const p = try queue.requestFile(path.?, .{}, .{ .gen_mipmaps = true });
    try testing.expect(waitForState(p, &.{.ready}));
    // Test-only reclaim: the worker never touches a `.ready` slot again
    // (module contract), so the test may restage it as in-flight to force
    // registration/publish interleaving deterministically.
    p.raw.deinit(a);
    p.state.store(.decoding, .release);

    // Four game-side registrars (e.g. concurrent GLB loads sharing one
    // image) racing the context publish below.
    const threads_n = 4;
    const per_thread = 16;
    var all_slots: [threads_n * per_thread]?Texture = [_]?Texture{null} ** (threads_n * per_thread);
    var go = std.atomic.Value(bool).init(false);
    var adders: [threads_n]std.Thread = undefined;
    var actxs: [threads_n]targetAdderCtx = undefined;
    for (0..threads_n) |i| {
        actxs[i] = .{
            .p = p,
            .slots = all_slots[i * per_thread ..][0..per_thread],
            .go = &go,
        };
        adders[i] = try std.Thread.spawn(.{}, targetAdder, .{&actxs[i]});
    }
    go.store(true, .release);
    // Context-side publish lands mid-registration: early targets ride the
    // pending list, late ones take the `.uploaded` fast path inline.
    // Either way each slot is patched exactly once.
    p.texture = std.mem.zeroes(Texture);
    p.state.store(.uploaded, .release);
    for (&adders) |*t| t.join();

    try testing.expectEqual(TextureState.uploaded, p.state.load(.acquire));
    _ = queue.commitUploadedTargets();
    for (all_slots) |slot| try testing.expect(slot != null);
    try testing.expectEqual(@as(usize, 0), queue.commitUploadedTargets());

    // Zero-handle stand-in texture: drop without sg deinit (headless).
    try testing.expect(p.take() != null);
    queue.release(p);

    // Wave 2: racing COMMITS on one published slot patch exactly once in
    // total (swap-empties-the-list under the per-slot mutex).
    const q2 = try queue.requestFile(path.?, .{}, .{});
    try testing.expect(waitForState(q2, &.{.ready}));
    q2.raw.deinit(a);
    var wave2: [32]?Texture = [_]?Texture{null} ** 32;
    for (&wave2) |*slot| q2.addTarget(slot);
    q2.texture = std.mem.zeroes(Texture);
    q2.state.store(.uploaded, .release);

    var total = std.atomic.Value(usize).init(0);
    var racers: [4]std.Thread = undefined;
    var rctxs: [4]commitRacerCtx = undefined;
    for (0..4) |i| {
        rctxs[i] = .{ .q = &queue, .sum = &total };
        racers[i] = try std.Thread.spawn(.{}, commitRacer, .{&rctxs[i]});
    }
    for (&racers) |*t| t.join();
    try testing.expectEqual(@as(usize, 32), total.load(.monotonic));
    for (wave2) |slot| try testing.expect(slot != null);

    try testing.expect(q2.take() != null);
    queue.release(q2);
}
