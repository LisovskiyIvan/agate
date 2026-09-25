//! Async texture pipeline (stage 2 of REFACTOR.md).
//!
//! Decode runs off-thread on a `jobs.TaskRunner` (`Texture.decodeFile` /
//! `decodeImageFile` are documented GPU-free and thread-safe); the `sg.*`
//! upload happens later on the thread that calls `drain`/`drainBudget` —
//! usually main — via `Texture.fromRaw` (RGBA8) or `Texture.fromRawBlock`
//! (block-compressed KTX2/DDS, gated by backend support).
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
pub const ktx2 = @import("ktx2.zig");
pub const asset_manager = @import("asset_manager.zig");
pub const AssetManager = asset_manager.AssetManager;
pub const AssetTask = asset_manager.AssetTask;
pub const AssetCache = asset_manager.AssetCache;
pub const TaskState = asset_manager.TaskState;
pub const TaskType = asset_manager.TaskType;

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
    /// Block-compressed decode (KTX2/DDS BC/ASTC): owned per-level slices for
    /// Texture.fromRawBlock. Exactly one of `raw` / `block_raw` is populated
    /// per slot (an empty RawTexture has num_levels 0; block uses null).
    block_raw: ?ktx2.RawBlockTexture = null,
    texture: ?Texture = null,

    pub fn addTarget(self: *PendingTexture, slot: *?Texture) void {
        if (self.state.load(.acquire) == .uploaded and self.texture != null) {
            slot.* = self.texture;
            return;
        }
        self.targets.append(self.allocator, slot) catch {};
    }

    fn decode(self: *PendingTexture) void {
        if (self.memory) |bytes| {
            // Invariant: no field writes after the terminal release store —
            // a consumer observing .ready/.failed may release() and destroy
            // the slot, so free/null inputs and assign raw/err first.
            // decodeImageMemory routes block KTX2 to the block path and
            // everything else to the unchanged RGBA8 decode.
            const result = Texture.decodeImageMemory(self.allocator, bytes, self.decode_opts);
            self.allocator.free(bytes);
            self.memory = null;
            if (result) |img| {
                switch (img) {
                    .rgba => |raw| self.raw = raw,
                    .block => |raw| self.block_raw = raw,
                }
                self.state.store(.ready, .release);
            } else |e| {
                self.err = e;
                self.state.store(.failed, .release);
            }
            return;
        }
        // File branch upholds the same invariant: raw/err are assigned
        // before the terminal store, with no field writes after it.
        // decodeImageFile applies the same block/RGBA8 routing to files.
        const img = Texture.decodeImageFile(self.allocator, self.path, self.decode_opts) catch |e| {
            self.err = e;
            self.state.store(.failed, .release);
            return;
        };
        switch (img) {
            .rgba => |raw| self.raw = raw,
            .block => |raw| self.block_raw = raw,
        }
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

/// Byte-budget stop decision for texture-upload drains. Pure (no sg), so it
/// is unit-testable. Overshoot rule: the first texture of a call always
/// uploads (`uploaded_count == 0` never stops) — a single texture bigger
/// than the whole budget must still make progress, or it would never
/// upload. Once at least one texture has uploaded, reaching `max_bytes`
/// (`uploaded_bytes >= budget`) stops the drain; leftover `.ready` slots
/// ride to the next call. A null budget never exhausts (count-only drain).
pub fn uploadBudgetExhausted(uploaded_count: usize, uploaded_bytes: u64, max_bytes: ?u64) bool {
    const budget = max_bytes orelse return false;
    if (uploaded_count == 0) return false;
    return uploaded_bytes >= budget;
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
                .ready => {
                    p.raw.deinit(self.allocator);
                    if (p.block_raw) |*b| b.deinit(self.allocator);
                },
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

    /// Looks for an existing pending or uploaded texture for `path`.
    /// Returns the matching slot if it is still valid (not failed or taken).
    pub fn findFile(self: *UploadQueue, path: []const u8) ?*PendingTexture {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        for (self.pending.items) |p| {
            if (p.path.len > 0 and std.mem.eql(u8, p.path, path)) {
                const s = p.state.load(.acquire);
                if (s != .failed and s != .taken) return p;
            }
        }
        return null;
    }

    /// Deduplicating asset pipeline request: reuses an in-flight or uploaded
    /// texture for the same path instead of issuing redundant decodes and GPU uploads.
    pub fn getOrRequestFile(
        self: *UploadQueue,
        path: []const u8,
        options: Texture.Options,
        decode_opts: Texture.DecodeOptions,
    ) !*PendingTexture {
        if (self.findFile(path)) |existing| {
            return existing;
        }
        return self.requestFile(path, options, decode_opts);
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
    pub fn collectReady(self: *UploadQueue, out: []*PendingTexture) usize {
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
        return self.drainCountedBudget(max, null);
    }

    /// Shared post-upload bookkeeping for the RGBA8 and block pixel paths
    /// (runs on the sg-context thread, right after the fromRaw/fromRawBlock
    /// call): patches material targets, publishes .uploaded, tallies the
    /// per-call count + the exact bytes handed to sg.
    fn finishUpload(p: *PendingTexture, bytes: u64, res: *DrainResult) void {
        for (p.targets.items) |slot| slot.* = p.texture.?;
        p.targets.deinit(p.allocator);
        p.targets = .empty;
        p.state.store(.uploaded, .release);
        res.count += 1;
        res.bytes += bytes;
    }

    /// Byte-aware drain: like `drainCounted` but additionally stops once
    /// accumulated uploaded bytes reach `max_bytes` (null = no byte limit,
    /// identical to `drainCounted`). The budget is checked between slot
    /// uploads via `uploadBudgetExhausted`, never after queuing work beyond
    /// it: each collected chunk uploads only its affordable prefix and the
    /// rest stays `.ready` for the next call (collection itself mutates no
    /// slot state). Overshoot rule: at least ONE texture uploads per call
    /// even when it alone exceeds the budget, so huge textures still make
    /// progress. Without an sg context this is a safe no-op, same as
    /// `drainCounted`.
    pub fn drainCountedBudget(self: *UploadQueue, max_count: ?usize, max_bytes: ?u64) DrainResult {
        gpu_thread.assertOnContextThread();
        if (!sg.isvalid()) return .{};
        var res: DrainResult = .{};
        var buf: [drain_chunk]*PendingTexture = undefined;
        while (max_count == null or res.count < max_count.?) {
            // Byte budget stops the drain before collecting more work; the
            // first upload of the call is always allowed (see helper).
            if (uploadBudgetExhausted(res.count, res.bytes, max_bytes)) break;
            const want = if (max_count) |m| @min(m - res.count, drain_chunk) else drain_chunk;
            if (want == 0) break;
            const n = self.collectReady(buf[0..want]);
            if (n == 0) break;
            var stopped_by_budget = false;
            for (buf[0..n]) |p| {
                // Re-check between uploads: bytes are only known while a
                // slot is being uploaded, so the next slot starts only when
                // the tally so far is still under budget.
                if (uploadBudgetExhausted(res.count, res.bytes, max_bytes)) {
                    stopped_by_budget = true;
                    break;
                }
                // Defensive only: per the module contract no other thread can
                // advance a collected `.ready` slot (worker never leaves it;
                // release/deinit are serialized by phase ownership).
                if (p.state.load(.acquire) != .ready) continue;
                if (p.block_raw) |*br| {
                    // Block path: the backend gate can still fail here (no
                    // CPU fallback exists), so the slot reports .failed with
                    // the reason instead of uploading. Bytes are tallied
                    // only on success — exactly what reaches sg.
                    const bytes: u64 = br.totalBytes();
                    const tex = Texture.fromRawBlock(br, p.options) catch |e| {
                        br.deinit(p.allocator);
                        p.block_raw = null;
                        p.err = e;
                        p.state.store(.failed, .release);
                        continue;
                    };
                    p.texture = tex;
                    br.deinit(p.allocator);
                    p.block_raw = null;
                    finishUpload(p, bytes, &res);
                    continue;
                }
                var bytes: u64 = 0;
                for (p.raw.levels[0..p.raw.num_levels]) |level| {
                    if (level) |level_buf| bytes += level_buf.len;
                }
                p.texture = Texture.fromRaw(&p.raw, p.options);
                p.raw.deinit(p.allocator);
                finishUpload(p, bytes, &res);
            }
            // Budget stopped us mid-chunk: remaining collected slots were
            // never touched and stay `.ready` for the next call.
            if (stopped_by_budget) break;
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
