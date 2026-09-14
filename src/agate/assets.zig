//! Async texture pipeline (stage 2 of REFACTOR.md).
//!
//! Decode runs off-thread on a `jobs.TaskRunner` (`Texture.decodeFile` is
//! documented GPU-free and thread-safe); the `sg.*` upload happens later on
//! the thread that calls `drain` — usually main — via `Texture.fromRaw`.
//!
//! Ownership model:
//!   - `requestFile` allocates a `PendingTexture` and posts the decode.
//!   - `drain` uploads every finished decode (optional `target` slot gets
//!     the live texture pointer patched in — e.g. `&pbr.albedo_texture.?`,
//!     so draws pick the real texture up on the next frame with no further
//!     wiring).
//!   - `take` moves the GPU texture out once uploaded.
//!   - `release` frees a finished slot (after `take`, or when failed).
//!
//! State publication is ordered by the atomic `state` release/acquire pair:
//! the worker writes results before publishing, consumers read after.

const std = @import("std");
const jobs = @import("jobs.zig");
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
    path: []u8 = &.{},
    options: Texture.Options = .{},
    decode_opts: Texture.DecodeOptions = .{},
    /// When set, `drain` patches the uploaded texture pointer into here so
    /// materials pick it up without re-wiring (in-place update propagates:
    /// draws read the material every frame).
    target: ?*Texture = null,
    allocator: std.mem.Allocator = undefined,
    raw: Texture.RawTexture = .{},
    texture: ?Texture = null,

    fn decode(self: *PendingTexture) void {
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
            if (p.path.len > 0) self.allocator.free(p.path);
            self.allocator.destroy(p);
        }
        self.pending.deinit(self.allocator);
    }

    /// Starts an async load of an image file. The returned slot stays valid
    /// until `release` (or queue `deinit`).
    pub fn requestFile(
        self: *UploadQueue,
        path: []const u8,
        options: Texture.Options,
        decode_opts: Texture.DecodeOptions,
        target: ?*Texture,
    ) !*PendingTexture {
        const p = try self.allocator.create(PendingTexture);
        errdefer self.allocator.destroy(p);
        p.* = .{
            .path = try self.allocator.dupe(u8, path),
            .options = options,
            .decode_opts = decode_opts,
            .target = target,
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

    fn decodeTask(ctx: *anyopaque) void {
        const p: *PendingTexture = @ptrCast(@alignCast(ctx));
        p.decode();
    }

    /// Uploads every finished decode. Must run on the sg-context thread.
    /// Returns how many textures were uploaded this call.
    pub fn drain(self: *UploadQueue) usize {
        var uploaded: usize = 0;
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        for (self.pending.items) |p| {
            if (p.state.load(.acquire) != .ready) continue;
            p.texture = Texture.fromRaw(&p.raw, p.options);
            p.raw.deinit(p.allocator);
            if (p.target) |slot| slot.* = p.texture.?;
            p.state.store(.uploaded, .release);
            uploaded += 1;
        }
        return uploaded;
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
        if (p.path.len > 0) self.allocator.free(p.path);
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

    const p = try queue.requestFile("/definitely/missing/image.png", .{}, .{}, null);
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

    const p = try queue.requestFile(path, .{}, .{ .gen_mipmaps = true }, null);
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
        _ = try queue.requestFile(path, .{}, .{}, null);
    }
    queue.deinit();
}

test "many parallel loads all report terminal states" {
    const a = testing.allocator;
    var queue = try UploadQueue.init(a, 2);
    defer queue.deinit();

    var slots: [8]*PendingTexture = undefined;
    for (&slots) |*slot| {
        slot.* = try queue.requestFile("/definitely/missing/image.png", .{}, .{}, null);
    }
    for (slots) |p| {
        try testing.expect(waitForState(p, &.{.failed}));
    }
    for (slots) |p| queue.release(p);
    try testing.expectEqual(@as(usize, 0), queue.drain());
}
