//! Async texture pipeline (stage 2 of REFACTOR.md).
//!
//! Decode runs off-thread on a `jobs.TaskRunner` (`Texture.decodeFile` is
//! documented GPU-free and thread-safe); the `sg.*` upload happens later on
//! the thread that calls `drain`/`drainBudget` — usually main — via
//! `Texture.fromRaw`.
//!
//! Ownership model:
//!   - `requestFile` allocates a `PendingTexture` and posts the decode.
//!   - `drainBudget` uploads at most `max` finished decodes per call
//!     (`drain` is the unbounded wrapper); an optional `target` slot gets
//!     the live texture pointer patched in — e.g. `&pbr.albedo_texture.?`,
//!     so draws pick the real texture up on the next frame with no further
//!     wiring. Leftover `.ready` slots ride to later frames.
//!   - `take` moves the GPU texture out once uploaded.
//!   - `release` frees a finished slot (after `take`, or when failed).
//!
//! Locking/ownership contract (what `drainCounted` relies on):
//!   - The queue is serialized by external phase ownership, not by thread
//!     identity: in the threaded apps the game and render phases never
//!     overlap (`phase_mutex`), so `requestFile`/`requestMemory` +
//!     `PendingTexture.addTarget` (posted from either side — the off-context
//!     GLB loader does it from the game thread) and
//!     `drainCounted`/`release`/`deinit` never run concurrently. Posting and
//!     draining from two threads at once is out of contract for every op
//!     except the worker's decode.
//!   - The spinlock therefore only ever contends with list ops from the
//!     current phase plus the worker's atomic state stores.
//!   - The worker touches a slot only while it is `.decoding` and publishes
//!     `.ready`/`.failed` via the state release store; after that it never
//!     touches the slot again. So once a slot is collected as `.ready`, the
//!     unlocked upload phase (fromRaw, raw deinit, target patching, state
//!     store) races with nothing and needs no re-lock.
//!   - Slots are freed only by `release` (contract: `.failed`/`.taken` only)
//!     and `deinit` (runner joined first), both serialized by the same phase
//!     ownership, so a collected pointer stays alive through the unlocked
//!     phase.
//!
//! State publication is ordered by the atomic `state` release/acquire pair:
//! the worker writes results before publishing, consumers read after.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const jobs = @import("jobs.zig");
const gpu_thread = @import("gpu_thread.zig");
const Texture = @import("texture.zig").Texture;

pub const TextureState = enum(u8) {
    /// Worker is reading/decoding the file.
    decoding,
    /// Pixels (and mips) decoded; waiting for the main-thread upload.
    ready,
    /// GPU image/view/sampler created; `texture` is live, `raw` freed.
    uploaded,
    /// Decode failed; `err` holds the reason (file missing, bad format...).
    failed,
    /// `texture` was moved out via `take`; the slot is release-ready.
    taken,
};

pub const PendingTexture = struct {
    state: std.atomic.Value(TextureState) = std.atomic.Value(TextureState).init(.decoding),
    /// Set by the worker before publishing `.failed`; read after observing
    /// that state. The state store is the release barrier.
    err: ?anyerror = null,
    /// Owned source: `path` = file on disk; `memory` = owned copy of an
    /// embedded payload (glTF buffer views die with the cgltf data, so the
    /// bytes are copied at request time and freed right after decode).
    path: []u8 = &.{},
    memory: ?[]u8 = null,
    options: Texture.Options = .{},
    decode_opts: Texture.DecodeOptions = .{},
    /// Material slots to patch on upload (e.g. `&pbr.albedo_texture`).
    /// Several materials may share one image. Registration happens on the
    /// posting thread before the next `drain`; both run on the sg thread,
    /// so no lock is needed here.
    targets: std.ArrayListUnmanaged(*?Texture) = .empty,
    allocator: std.mem.Allocator = undefined,
    raw: Texture.RawTexture = .{},
    texture: ?Texture = null,

    pub fn addTarget(self: *PendingTexture, slot: *?Texture) void {
        self.targets.append(self.allocator, slot) catch {};
    }

    fn decode(self: *PendingTexture) void {
        if (self.memory) |bytes| {
            defer {
                self.allocator.free(bytes);
                self.memory = null;
            }
            if (Texture.decodeMemory(self.allocator, bytes, self.decode_opts)) |raw| {
                self.raw = raw;
                self.state.store(.ready, .release);
            } else |e| {
                self.err = e;
                self.state.store(.failed, .release);
            }
            return;
        }
        const raw = Texture.decodeFile(self.allocator, self.path, self.decode_opts) catch |e| {
            self.err = e;
            self.state.store(.failed, .release);
            return;
        };
        self.raw = raw;
        self.state.store(.ready, .release);
    }

    /// Moves the uploaded GPU texture out. Returns null while the load is
    /// not uploaded yet (or was already taken / failed).
    pub fn take(self: *PendingTexture) ?Texture {
        if (self.state.load(.acquire) != .uploaded) return null;
        self.state.store(.taken, .release);
        const t = self.texture.?;
        self.texture = null;
        return t;
    }

    fn deinitResources(self: *PendingTexture) void {
        if (self.path.len > 0) self.allocator.free(self.path);
        if (self.memory) |m| self.allocator.free(m);
        self.targets.deinit(self.allocator);
    }
};

/// std.atomic.Mutex is a spinlock with tryLock only; block by spinning.
/// Critical sections here are appends/removes of one pointer.
fn lockSpin(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) std.atomic.spinLoopHint();
}

pub const UploadQueue = struct {
    allocator: std.mem.Allocator,
    runner: *jobs.TaskRunner,
    /// Spinlock: critical sections are pointer-sized appends/removes at
    /// asset-scale rates (dozens per load, never per frame). sokol 0.16
    /// std.Thread ships no blocking Mutex (see jobs.ParkingLot note).
    mutex: std.atomic.Mutex = .unlocked,
    pending: std.ArrayListUnmanaged(*PendingTexture) = .empty,

    /// Creates the queue plus its decode worker pool. `task_threads` of 1-2
    /// is plenty: single-image decode does not split further (stb/ktx2 are
    /// monolithic), so parallelism across images is the only axis.
    pub fn init(allocator: std.mem.Allocator, task_threads: usize) !UploadQueue {
        return .{
            .allocator = allocator,
            .runner = try jobs.TaskRunner.init(allocator, task_threads),
        };
    }

    /// Shuts the runner down first: join-on-shutdown guarantees every
    /// decode task has finished writing before the slots are freed.
    pub fn deinit(self: *UploadQueue) void {
        self.runner.deinit();
        for (self.pending.items) |p| {
            switch (p.state.load(.acquire)) {
                .uploaded => if (p.texture) |*t| t.deinit(),
                .ready => p.raw.deinit(self.allocator),
                else => {},
            }
            p.deinitResources();
            self.allocator.destroy(p);
        }
        self.pending.deinit(self.allocator);
    }

    /// Starts an async load of an image file. The returned slot stays valid
    /// until `release` (or queue `deinit`); register targets via
    /// `PendingTexture.addTarget` before the next `drain`.
    pub fn requestFile(
        self: *UploadQueue,
        path: []const u8,
        options: Texture.Options,
        decode_opts: Texture.DecodeOptions,
    ) !*PendingTexture {
        const p = try self.allocator.create(PendingTexture);
        errdefer self.allocator.destroy(p);
        p.* = .{
            .path = try self.allocator.dupe(u8, path),
            .options = options,
            .decode_opts = decode_opts,
            .allocator = self.allocator,
        };
        errdefer self.allocator.free(p.path);

        lockSpin(&self.mutex);
        self.pending.append(self.allocator, p) catch |e| {
            self.mutex.unlock();
            return e;
        };
        self.mutex.unlock();

        self.runner.post(p, decodeTask);
        return p;
    }

    /// Starts an async decode of an owned embedded payload. `bytes`
    /// ownership transfers to the queue (freed right after decode).
    pub fn requestMemory(
        self: *UploadQueue,
        bytes: []u8,
        options: Texture.Options,
        decode_opts: Texture.DecodeOptions,
    ) !*PendingTexture {
        const p = try self.allocator.create(PendingTexture);
        errdefer self.allocator.destroy(p);
        p.* = .{
            .memory = bytes,
            .options = options,
            .decode_opts = decode_opts,
            .allocator = self.allocator,
        };

        lockSpin(&self.mutex);
        self.pending.append(self.allocator, p) catch |e| {
            self.mutex.unlock();
            return e;
        };
        self.mutex.unlock();

        self.runner.post(p, decodeTask);
        return p;
    }

    fn decodeTask(ctx: *anyopaque) void {
        const p: *PendingTexture = @ptrCast(@alignCast(ctx));
        p.decode();
    }

    /// Per-call upload tally: how many textures reached the GPU and how
    /// many decoded bytes were handed to sg (sum of mip level bytes).
    pub const DrainResult = struct {
        count: usize = 0,
        bytes: u64 = 0,
    };

    /// Ready-slot batch capacity per lock acquisition. Normal frames drain
    /// at most `Scene.upload_budget_per_frame`; asset-scale unbounded drains
    /// loop over chunks, so the spinlock is never held across GPU work and
    /// the batch itself is a stack array (no per-frame allocation).
    const drain_chunk: usize = 64;

    /// Collects up to `out.len` `.ready` slots into `out`. The spinlock is
    /// held only for this scan; GPU work and slot patching happen after
    /// release (see module contract).
    fn collectReady(self: *UploadQueue, out: []*PendingTexture) usize {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        var n: usize = 0;
        for (self.pending.items) |p| {
            if (n == out.len) break;
            if (p.state.load(.acquire) != .ready) continue;
            out[n] = p;
            n += 1;
        }
        return n;
    }

    /// Uploads at most `max` finished decodes (null = every finished decode),
    /// in chunks so the spinlock is never held across GPU work. Must run
    /// under phase ownership (sg-context phase; see module contract).
    /// Returns per-call count + bytes. Without an sg context (unit tests,
    /// tools) this is a safe no-op that leaves `.ready` slots intact for a
    /// later real drain.
    pub fn drainCounted(self: *UploadQueue, max: ?usize) DrainResult {
        gpu_thread.assertOnContextThread();
        if (!sg.isvalid()) return .{};
        var res: DrainResult = .{};
        var buf: [drain_chunk]*PendingTexture = undefined;
        while (max == null or res.count < max.?) {
            const want = if (max) |m| @min(m - res.count, drain_chunk) else drain_chunk;
            if (want == 0) break;
            const n = self.collectReady(buf[0..want]);
            if (n == 0) break;
            for (buf[0..n]) |p| {
                // Defensive only: per the module contract no other thread can
                // advance a collected `.ready` slot (worker never leaves it;
                // release/deinit are serialized by phase ownership).
                if (p.state.load(.acquire) != .ready) continue;
                var bytes: u64 = 0;
                for (p.raw.levels[0..p.raw.num_levels]) |level| {
                    if (level) |level_buf| bytes += level_buf.len;
                }
                p.texture = Texture.fromRaw(&p.raw, p.options);
                p.raw.deinit(p.allocator);
                for (p.targets.items) |slot| slot.* = p.texture.?;
                p.targets.deinit(p.allocator);
                p.targets = .empty;
                p.state.store(.uploaded, .release);
                res.count += 1;
                res.bytes += bytes;
            }
            // Fewer ready slots than requested: the queue is exhausted for
            // this call (a defensive skip above does not change that).
            if (n < want) break;
        }
        return res;
    }

    /// Uploads at most `max` finished decodes. Must run on the sg-context
    /// thread. Returns how many textures were uploaded this call; leftover
    /// `.ready` slots upload on subsequent calls.
    pub fn drainBudget(self: *UploadQueue, max: usize) usize {
        return self.drainCounted(max).count;
    }

    /// Uploads every finished decode. Must run on the sg-context thread.
    /// Returns how many textures were uploaded this call. Unbounded wrapper
    /// around `drainCounted` (kept for compatibility/tests).
    pub fn drain(self: *UploadQueue) usize {
        return self.drainCounted(null).count;
    }

    /// Frees a finished slot. Valid once the state is `.failed` or `.taken`
    /// (an `.uploaded` texture must be `take`n first — otherwise its GPU
    /// resources would leak).
    pub fn release(self: *UploadQueue, p: *PendingTexture) void {
        const st = p.state.load(.acquire);
        std.debug.assert(st == .failed or st == .taken);
        lockSpin(&self.mutex);
        for (self.pending.items, 0..) |item, i| {
            if (item == p) {
                _ = self.pending.orderedRemove(i);
                break;
            }
        }
        self.mutex.unlock();
        p.deinitResources();
        self.allocator.destroy(p);
    }
};

// --- tests ---

const testing = std.testing;

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

/// Repo-relative font bitmap probe (same locations as the PNG test above).
fn findFontPng() ?[]const u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const candidates = [_][]const u8{
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
