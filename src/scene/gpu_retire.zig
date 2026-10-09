//! GPU handle lifetime epochs: unified mechanism for deferred destruction of GPU resources.
//!
//! Ownership model:
//! - WRITERS (any thread): `retireMesh`, `retireBuffer`, `retireProbeTarget`,
//!   `retireUi3dTarget`, `retireComputeBundle`. Acquires spinlock, timestamps with
//!   current epoch and appends entry. Never calls `sg.*` or frees memory on writer thread.
//! - READERS/DESTROYS (GPU context thread only): `flush` and `deinit`.
//!   Invokes `Mesh.deinit`, `sg.destroy*`, and `allocator.destroy`.
//!
//! Epoch rules:
//! - `begin` opens a new frame epoch and closes the previous unclosed epoch.
//! - `complete(e)` closes epoch `e`.
//! - Entries retired in epoch E are destroyed on the first `flush` after `complete(E)`
//!   (`entry.epoch <= lastCompleted()`). Entries from uncompleted epochs wait.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const Mesh = @import("../mesh.zig").Mesh;
const gpu_thread = @import("../gpu_thread.zig");
const probe_layer = @import("probe_layer.zig");
const gui3d_layer = @import("gui3d_layer.zig");

/// Render frame counter. 0 means no frame yet; counter starts at 1.
pub const Epoch = u64;

/// Capacity of zero-allocation spillover buffer on OOM in retireMesh.
const overflow_cap: usize = 8;

/// Single deferred resource entry with retirement epoch.
const Entry = struct {
    kind: Kind,
    mesh: ?*Mesh = null,
    buffer: sg.Buffer = .{},
    probe: probe_layer.ProbeGpu = .{},
    ui3d: gui3d_layer.Ui3dTarget = .{},
    compute: ComputeBundle = .{},
    epoch: Epoch,
};

pub const Kind = enum { mesh, buffer, probe, ui3d, compute };

/// Staged compute creation outcome bundle to be retired safely in dependency order.
pub const ComputeBundle = struct {
    state_buffer: sg.Buffer = .{},
    spawn_buffer: sg.Buffer = .{},
    draw_buffer: sg.Buffer = .{},
    state_view: sg.View = .{},
    spawn_view: sg.View = .{},
    draw_view: sg.View = .{},
    shader: sg.Shader = .{},
    pipeline: sg.Pipeline = .{},

    /// Returns true if the bundle has no allocated handles.
    pub fn isEmpty(self: ComputeBundle) bool {
        return self.state_buffer.id == 0 and self.spawn_buffer.id == 0 and
            self.draw_buffer.id == 0 and self.state_view.id == 0 and
            self.spawn_view.id == 0 and self.draw_view.id == 0 and
            self.shader.id == 0 and self.pipeline.id == 0;
    }

    /// Destroys all bundle resources in dependency order on the GPU context thread.
    pub fn deinit(self: *ComputeBundle) void {
        if (self.state_view.id != 0) sg.destroyView(self.state_view);
        if (self.spawn_view.id != 0) sg.destroyView(self.spawn_view);
        if (self.draw_view.id != 0) sg.destroyView(self.draw_view);
        if (self.pipeline.id != 0) sg.destroyPipeline(self.pipeline);
        if (self.shader.id != 0) sg.destroyShader(self.shader);
        if (self.state_buffer.id != 0) sg.destroyBuffer(self.state_buffer);
        if (self.spawn_buffer.id != 0) sg.destroyBuffer(self.spawn_buffer);
        if (self.draw_buffer.id != 0) sg.destroyBuffer(self.draw_buffer);
    }
};

fn lockSpin(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) std.atomic.spinLoopHint();
}

pub const GpuRetireQueue = struct {
    const Self = @This();

    mutex: std.atomic.Mutex = .unlocked,
    current_epoch: Epoch = 0,
    completed_epoch: Epoch = 0,
    pending: std.ArrayListUnmanaged(Entry) = .empty,
    overflow: [overflow_cap]?Entry = [_]?Entry{null} ** overflow_cap,
    overflow_len: usize = 0,
    /// Bound on jointly retained entries (`pending.items.len + overflow_len`).
    pending_cap: usize = 8192,
    /// Entries dropped by the cap (observable; each also logs).
    capped_drops: u64 = 0,
    /// Duplicate retires skipped (same mesh pointer / buffer id already queued).
    duplicate_drops: u64 = 0,

    /// Opens a new frame epoch and closes previous unclosed epoch. Context thread only.
    pub fn begin(self: *Self) Epoch {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        if (self.current_epoch > self.completed_epoch) {
            self.completed_epoch = self.current_epoch;
        }
        self.current_epoch +%= 1;
        return self.current_epoch;
    }

    /// Closes epoch e. Context thread only.
    pub fn complete(self: *Self, e: Epoch) void {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        std.debug.assert(e <= self.current_epoch);
        if (e > self.completed_epoch) {
            self.completed_epoch = e;
        }
    }

    /// Returns the current (open) epoch.
    pub fn current(self: *Self) Epoch {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.current_epoch;
    }

    /// Returns the last completed epoch.
    pub fn lastCompleted(self: *Self) Epoch {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.completed_epoch;
    }

    /// Retires a mesh for deferred destruction on the GPU context thread.
    pub fn retireMesh(self: *Self, allocator: std.mem.Allocator, mesh: *Mesh) void {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .kind = .mesh, .mesh = mesh, .epoch = self.current_epoch };
        if (self.containsLocked(entry)) {
            self.duplicate_drops += 1;
            return;
        }
        if (!self.admitsLocked()) {
            self.capped_drops += 1;
            std.log.err("scene: retire queue cap {d} reached, leaking mesh '{s}' (bound skip-streak destroys; see pending_cap)", .{ self.pending_cap, mesh.name });
            return;
        }
        self.pending.append(allocator, entry) catch {
            // Overflow slot requires no allocation, preserving thread affinity.
            if (self.overflow_len < self.overflow.len) {
                self.overflow[self.overflow_len] = entry;
                self.overflow_len += 1;
            } else {
                // Queues exhausted under pathological OOM: log and leak rather than mutating sg off-thread.
                std.log.err("scene: destroy queues exhausted, leaking mesh '{s}'", .{mesh.name});
            }
        };
    }

    /// Retires an instance buffer for deferred destruction on the GPU context thread.
    pub fn retireBuffer(self: *Self, allocator: std.mem.Allocator, buf: sg.Buffer) void {
        if (buf.id == 0) return;
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .kind = .buffer, .buffer = buf, .epoch = self.current_epoch };
        if (self.containsLocked(entry)) {
            self.duplicate_drops += 1;
            return;
        }
        if (!self.admitsLocked()) {
            self.capped_drops += 1;
            std.log.err("scene: retire queue cap {d} reached, leaking instance buffer (id {})", .{ self.pending_cap, buf.id });
            return;
        }
        self.pending.append(allocator, entry) catch {
            if (self.overflow_len < self.overflow.len) {
                self.overflow[self.overflow_len] = entry;
                self.overflow_len += 1;
            } else {
                std.log.err("scene: destroy queues exhausted, leaking instance buffer (id {})", .{buf.id});
            }
        };
    }

    /// Retires a reflection probe target for deferred destruction on the GPU context thread.
    pub fn retireProbeTarget(self: *Self, allocator: std.mem.Allocator, target: probe_layer.ProbeGpu) void {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .kind = .probe, .probe = target, .epoch = self.current_epoch };
        if (self.containsLocked(entry)) {
            self.duplicate_drops += 1;
            return;
        }
        if (!self.admitsLocked()) {
            self.capped_drops += 1;
            std.log.err("scene: retire queue cap {d} reached, leaking probe target (image id {})", .{ self.pending_cap, target.image.id });
            return;
        }
        self.pending.append(allocator, entry) catch {
            if (self.overflow_len < self.overflow.len) {
                self.overflow[self.overflow_len] = entry;
                self.overflow_len += 1;
            } else {
                std.log.err("scene: destroy queues exhausted, leaking probe target (image id {})", .{target.image.id});
            }
        };
    }

    /// Retires a 3D UI panel target for deferred destruction on the GPU context thread.
    pub fn retireUi3dTarget(self: *Self, allocator: std.mem.Allocator, target: gui3d_layer.Ui3dTarget) void {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .kind = .ui3d, .ui3d = target, .epoch = self.current_epoch };
        if (self.containsLocked(entry)) {
            self.duplicate_drops += 1;
            return;
        }
        if (!self.admitsLocked()) {
            self.capped_drops += 1;
            std.log.err("scene: retire queue cap {d} reached, leaking ui3d target (image id {})", .{ self.pending_cap, target.image.id });
            return;
        }
        self.pending.append(allocator, entry) catch {
            if (self.overflow_len < self.overflow.len) {
                self.overflow[self.overflow_len] = entry;
                self.overflow_len += 1;
            } else {
                std.log.err("scene: destroy queues exhausted, leaking ui3d target (image id {})", .{target.image.id});
            }
        };
    }

    /// Retires a discarded compute bundle for deferred destruction on the GPU context thread.
    pub fn retireComputeBundle(self: *Self, allocator: std.mem.Allocator, bundle: ComputeBundle) void {
        if (bundle.isEmpty()) return;
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        const entry = Entry{ .kind = .compute, .compute = bundle, .epoch = self.current_epoch };
        if (self.containsLocked(entry)) {
            self.duplicate_drops += 1;
            return;
        }
        if (!self.admitsLocked()) {
            self.capped_drops += 1;
            std.log.err("scene: retire queue cap {d} reached, leaking compute bundle (state buffer id {})", .{ self.pending_cap, bundle.state_buffer.id });
            return;
        }
        self.pending.append(allocator, entry) catch {
            if (self.overflow_len < self.overflow.len) {
                self.overflow[self.overflow_len] = entry;
                self.overflow_len += 1;
            } else {
                std.log.err("scene: destroy queues exhausted, leaking compute bundle (state buffer id {})", .{bundle.state_buffer.id});
            }
        };
    }

    /// Destroys all entries with epoch <= completed_epoch. Context thread only.
    pub fn flush(self: *Self, allocator: std.mem.Allocator) void {
        gpu_thread.assertOnContextThread();
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        self.drainLocked(allocator, self.completed_epoch);
    }

    /// Flushes all remaining entries unconditionally and releases queues. Context thread only.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        gpu_thread.assertOnContextThread();
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        self.drainLocked(allocator, std.math.maxInt(Epoch));
        self.pending.deinit(allocator);
        @memset(&self.overflow, null);
        self.overflow_len = 0;
    }

    /// Returns the number of entries waiting to be destroyed (pending + overflow).
    pub fn retainedCount(self: *Self) usize {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.pending.items.len + self.overflow_len;
    }

    /// Returns the number of entries dropped by pending_cap.
    pub fn cappedDropCount(self: *Self) u64 {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.capped_drops;
    }

    /// Returns the number of duplicate retire calls ignored.
    pub fn duplicateDropCount(self: *Self) u64 {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.duplicate_drops;
    }

    /// Tests if one more entry can be admitted under pending_cap.
    pub fn admitsOneMore(self: *Self) bool {
        lockSpin(&self.mutex);
        defer self.mutex.unlock();
        return self.admitsLocked();
    }

    /// Internal: checks capacity under mutex.
    fn admitsLocked(self: *const Self) bool {
        return self.pending.items.len + self.overflow_len < self.pending_cap;
    }

    /// Internal: checks if an identical handle is already queued under mutex.
    fn containsLocked(self: *const Self, entry: Entry) bool {
        for (self.pending.items) |e| {
            if (sameHandle(e, entry)) return true;
        }
        for (self.overflow[0..self.overflow_len]) |slot| {
            if (slot) |e| {
                if (sameHandle(e, entry)) return true;
            }
        }
        return false;
    }

    /// Handle equality comparison across retire kinds.
    fn sameHandle(a: Entry, b: Entry) bool {
        if (a.kind != b.kind) return false;
        return switch (a.kind) {
            .mesh => a.mesh == b.mesh,
            .buffer => a.buffer.id == b.buffer.id,
            .probe => a.probe.image.id == b.probe.image.id,
            .ui3d => a.ui3d.image.id == b.ui3d.image.id,
            .compute => std.meta.eql(a.compute, b.compute),
        };
    }

    /// Internal: destroys entries with epoch <= done under mutex.
    fn drainLocked(self: *Self, allocator: std.mem.Allocator, done: Epoch) void {
        var kept: usize = 0;
        for (self.pending.items) |entry| {
            if (entry.epoch <= done) {
                switch (entry.kind) {
                    .mesh => {
                        entry.mesh.?.deinit(allocator);
                        allocator.destroy(entry.mesh.?);
                    },
                    .buffer => sg.destroyBuffer(entry.buffer),
                    .probe => {
                        var target = entry.probe;
                        target.deinit();
                    },
                    .ui3d => {
                        var target = entry.ui3d;
                        target.deinit();
                    },
                    .compute => {
                        var bundle = entry.compute;
                        bundle.deinit();
                    },
                }
            } else {
                self.pending.items[kept] = entry;
                kept += 1;
            }
        }
        self.pending.items.len = kept;
        var okept: usize = 0;
        for (self.overflow[0..self.overflow_len]) |slot| {
            if (slot) |entry| {
                if (entry.epoch <= done) {
                    switch (entry.kind) {
                        .mesh => {
                            entry.mesh.?.deinit(allocator);
                            allocator.destroy(entry.mesh.?);
                        },
                        .buffer => sg.destroyBuffer(entry.buffer),
                        .probe => {
                            var target = entry.probe;
                            target.deinit();
                        },
                        .ui3d => {
                            var target = entry.ui3d;
                            target.deinit();
                        },
                        .compute => {
                            var bundle = entry.compute;
                            bundle.deinit();
                        },
                    }
                    continue;
                }
                self.overflow[okept] = entry;
                okept += 1;
            }
        }
        @memset(self.overflow[okept..], null);
        self.overflow_len = okept;
    }
};

// GPU-retire regression tests live in `gpu_retire_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).
