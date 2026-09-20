//! P7 triple-buffered prepared draw payload: three retained owning queue slots
//! covering the prepared mesh draw lists — PRIMARY + ALL PIP view queues
//! (with their skin/shader side stores), outline items+skins, and prepared
//! shadow items+skins+bin ranges. Scope is mesh draws only (trail meshes
//! ride these same queues — Trail.update stages CPU-side and the prepare
//! flush uploads before the queue build bakes the values), plus the
//! game-side UI CPU packet staging (`ui_vertices`/`ui_indices` + `ui_packet`
//! header, CPU geometry only — the committed P6 `UiFrame` itself stays
//! single-owned outside the slots); particles, physics-debug lines, and sky
//! carry their own prepared frames/payloads — never these slots.
//!
//! Ownership / lifecycle block:
//! - Owns only CPU-side queue storage (ArrayList buffers, skin/shader copies,
//!   sort-order entries). GPU handles inside items (buffers/views/samplers/
//!   pipelines) are BORROWED under the phase mutex / P3 epoch discipline —
//!   this buffer never destroys them, never duplicates GPU buffers, and never
//!   changes the P3 retire algorithm. CPU slot retention is NOT a GPU
//!   lifetime pin: a retained slot may reference handles a GpuRetire flush
//!   already destroyed.
//! - Sequential phase contract: ONE pending frame, no concurrent
//!   prepare/render (same context thread). prepare builds a BACK slot in
//!   place (reset first), then publishes with a single index flip — never a
//!   shallow ArrayList copy (that would double-free), never a per-frame deep
//!   clone. Render reads only the published FRONT slot through const
//!   payloads. Update CAN overlap render (actual update||render boundary):
//!   the mailbox producer (update) vs consumer (prepare) stay excluded under
//!   phase_mutex instead — phase ownership no longer spans prepare+render.
//! - GPU consumability of a published slot ends at the consuming render's
//!   return or — when no render consumes it — at the START of the next
//!   prepareFrame: a repeated prepare discards the pending frame BEFORE
//!   GpuRetire.begin/flush (Scene.prepareFrame clears its consumable flag
//!   first), and that flush may tear down the borrowed handles the old
//!   front references. It also ends at Scene.deinit. The new publish at
//!   the end of prepare re-associates frame_id/retire_epoch.
//! - CPU storage outlives consumability (retained capacity, reused as a
//!   future back scratch), but that carries NO snapshot-lifetime or pinning
//!   guarantee: contents may be reset and rebuilt by any prepare. Getter
//!   pointers (Scene.preparedDraws and any slice taken from it) are therefore
//!   consumable only while frame_prepared is set or during the render call
//!   consuming the frame — never across a prepare boundary — UNLESS the slot
//!   is held under an explicit pin (see the lease protocol below), which
//!   excludes exactly that slot from every future build target and publish.
//!
//! Rotation (3 slots): `front` is the published consumable slot; the build
//! scratch is the first unpinned, unclaimed slot after `front`
//! (`(front + k) % SLOT_COUNT`, k = 1..). Flipping `front` IS the publish —
//! the lists themselves never move. With no pins held this cycles
//! 0 -> 1 -> 2 -> 0, so every prepare hands the producer a slot the consumer
//! is not reading: one slot deeper than the old 2-slot flip, which is what
//! lets a presenting consumer keep its frame while the producer already
//! builds the next one.
//!
//! Consumer pin/lease protocol (the prerequisite for a future true
//! concurrent update/prepare rotation):
//! - The render/present path PINs the slot it is presenting (`pin` /
//!   `pinFront`) and MUST unpin it when done (`unpin`). While pinned, a slot
//!   is never handed out as a build target (`backIndex` / `claimBack` skip
//!   it) and can never become a publish target (`publish` asserts it,
//!   `tryPublish` refuses it with `error.PinnedSlot` and counts the refusal).
//! - The concurrent producer path is `claimBack` (reserve a free slot for
//!   writing; null = saturated, skip the frame — the documented latest-wins
//!   drop, never a block) -> fill the payload -> `tryPublish` (hand it to
//!   the consumer). `claimBack` also marks the slot WRITING, and `pin`
//!   refuses a writing slot (`error.SlotBusy`): claim-vs-pin concurrency
//!   therefore always resolves exactly one way — either the pin wins and the
//!   claim looks elsewhere, or the claim wins and the pin retries on a newer
//!   front. A pinned slot is never written by the producer, a claimed slot
//!   is never pinned by the consumer: payload reads/writes on distinct slots
//!   need no further locking (single producer; index words are serialized by
//!   the internal mutex).
//! - Missed unpin is NEVER a silent wedge: a leaked pin only shrinks the
//!   free set — claims skip it, publishes refuse it, saturation degrades to
//!   the counted skip (`saturation_skips`, `publish_refusals`,
//!   `pin_denials`, `unpin_denials` are all observable). The debug guard is
//!   `deinit`, which asserts no pins are still held. `unpin` is mandatory and
//!   `unpin`-without-`pin` is an error, never a no-op.
//! - Threading split (explicit, no silent gap): the lease INDEX words
//!   (`front`, `pinned`, `writing`, counters) are serialized by the internal
//!   mutex in `claimBack` / `tryPublish` / `cancelClaim` / `pin` /
//!   `pinFront` / `unpin` / `frontIndex` / `isPinned` / `pinsHeld`. The legacy
//!   trio (`backIndex` / `backSlot` / `publish`) plus direct `slots[i]`
//!   payload access are SINGLE-THREADED ONLY (the sequential
//!   build/prepare/render path): they read `pinned`/`writing` without the
//!   mutex and must never run concurrently with lease activity. Concurrent
//!   producers/consumers must use the claim/pin API exclusively.
//! - Do not copy a FrameDraws (it owns a Mutex); Scene holds the single
//!   instance by value.
//!
//! Honest scope note — what this is NOT: the phase-mutex removal (true
//! concurrent update+prepare) is still NOT done and is not claimed. This
//! file makes the 3-slot rotation and the pin/lease primitives real and
//! tested, and wave 27 made the frame snapshot slot-owned (`FrameDrawSlot.
//! snapshot`: prepare/render/reuse/UI-latch read the staged slot copy, never
//! the live `Scene.frame_snapshot` — that closes the snapshot-tearing
//! obstacle), but update-vs-prepare exclusion (phase_mutex) is still
//! required, because the producer build and the prepare latch still touch
//! LIVE state outside the slot-owned data:
//! - `Scene.buildPreparedFrame` reads live meshes (TRS, materials, culling
//!   flags), the live canvas (stageUiPacket), live particle/physics systems,
//!   and live camera/light/sky state (packFrameSnapshot), and writes live
//!   per-mesh previews/build_views;
//! - the prepare latch writes back live `mesh.instance_render` (guarded
//!   write-back), reads the live mesh list for the identity guard, and the
//!   UI latch resolves handles/pipeline/font/dims from the live canvas;
//! - the frame mailbox (`frame_handoff`) producer/consumer pair is still
//!   phase-excluded (a true concurrent producer would need the claim/pin API
//!   here instead of the sequential backIndex/publish path).
//! What IS already slot-owned (and therefore needs no lock once the mutex
//! goes): the frame snapshot, the staged instance records, the UI packet
//! lists+header, and the prepare-gated GPU uploads (P3 epochs + upload
//! meter). Removing the phase mutex means moving the remaining live reads
//! above under the same freeze-then-latch shape (or an equivalent mailbox)
//! — the pin/lease here only covers the variable-length draw payload plus
//! the staged snapshot, deliberately nothing else.

const std = @import("std");
const render_queue = @import("render_queue.zig");
const snapshot_mod = @import("snapshot.zig");
const retire_mod = @import("gpu_retire.zig");
const ui_frame_mod = @import("ui_frame.zig");
const outline_pass = @import("../passes/outline_pass.zig");
const shadow_pass = @import("../passes/shadow_pass.zig");
const mesh_mod = @import("../mesh.zig");
const ui_mod = @import("../ui.zig");

pub const RenderQueues = render_queue.RenderQueues;
pub const SkinStorage = render_queue.SkinStorage;
pub const OutlineDrawItem = outline_pass.OutlineDrawItem;
pub const PreparedShadowDraws = shadow_pass.ShadowPass.PreparedShadowDraws;
pub const Epoch = retire_mod.Epoch;

/// Retained prepared-frame slot count. 3 (not 2): the consumer can present
/// one pinned slot while the producer builds the next and the newest
/// published frame waits in between — a 2-slot flip has no room for all
/// three at once.
pub const SLOT_COUNT: usize = 3;

/// Lease-protocol errors. All are fail-closed and counted; none wedge.
pub const LeaseError = error{
    /// Slot index out of range.
    InvalidSlot,
    /// `pin` on an already-pinned slot.
    AlreadyPinned,
    /// `pin` on a slot the producer currently holds claimed for writing.
    /// Retry on a newer front (latest-wins): the claim will publish soon.
    SlotBusy,
    /// `unpin` on a slot that is not pinned.
    NotPinned,
    /// `tryPublish` targeting a pinned slot (a would-be overwrite of a
    /// presented frame). State unchanged; counted in `publish_refusals`.
    PinnedSlot,
    /// `tryPublish` of a slot that was never claimed via `claimBack`.
    NotClaimed,
};

/// One coherent prepared frame: the prepared mesh draw lists (view queues,
// outline, shadow) the render phase consumes. Built whole into a BACK
// slot, then published by index flip; render touches it only through const
// references while it is the consumable front (see header), or under an
// explicit pin (which extends consumability past the next publish for
// exactly that slot).
//
// Stage-2 increment B payload identity: every instanced batch / shadow item /
// outline item carries `source_uid` (stable `Mesh.uid`) + `source_mesh`
// (mesh-list index at build time). Game-built (`.build_view`) payloads hold
// provisional `instance_buffer`/`visible_instance_count` (plus shadow
// `world_aabb`/`max_dim`, outline `world_center`) until the latch
// `patchInstanceRefs` finalizes them from the slot-owned `staged_instances`
// records (fail-closed zero on record-missing/uid mismatch or stale publish);
// fallback (`.published`) payloads are final at build time.
pub const StagedInstanceRecord = mesh_mod.StagedInstanceRecord;

pub const FrameDrawSlot = struct {
    primary: RenderQueues = .{},
    views: [snapshot_mod.MAX_CAMERAS]RenderQueues = [_]RenderQueues{.{}} ** snapshot_mod.MAX_CAMERAS,
    outline_items: std.ArrayListUnmanaged(OutlineDrawItem) = .empty,
    outline_skins: SkinStorage = .empty,
    shadow: PreparedShadowDraws = .{},
    /// Slot-owned staged instance records (lock-free-publication slice 1):
    /// one per instance-bearing mesh with a fresh preview, frozen by
    /// `Scene.buildPreparedFrame` (`freezeStagedRecords`) and consumed by
    /// the prepare latch (`stageInstancesLatch` + `patchInstanceRefs`)
    /// instead of live `Mesh.instance_preview` reads. Appended in mesh-list
    /// order (strictly increasing `mesh_index`); reset retains capacity, so
    /// the latch/patch allocate nothing. Record `buffer` copies are borrowed
    /// read handles (never destroyed/retired through the record).
    staged_instances: std.ArrayListUnmanaged(StagedInstanceRecord) = .empty,
    /// Slot-owned UI CPU packet (lock-free-publication slice 2, b): the
    /// game side (`Scene.stageUiPacket`) records live canvas CPU geometry
    /// into these back-slot lists and stamps `ui_packet`; the prepare latch
    /// consumes them into `Scene.ui_frame` instead of reading the live
    /// canvas lists. Plain CPU data (UIVertex/u16 — no GPU handles); reset
    /// retains capacity and clears the header, so the latch/patch allocate
    /// nothing and a stale packet can never resurface.
    ui_vertices: std.ArrayListUnmanaged(ui_mod.UIVertex) = .empty,
    ui_indices: std.ArrayListUnmanaged(u16) = .empty,
    ui_packet: ui_frame_mod.UiPacketState = .{},
    /// Slot-owned staged frame snapshot (wave 27, lock-free prerequisite):
    /// the frame-level camera/light/pass state the prepared payload was
    /// built against, frozen by value at build time. `Scene.buildPreparedFrame`
    /// (game side) stages `build_snapshot` here; the fallback prepare path
    /// stages `frame_snapshot` here after the takeLatest-else-pack latch.
    /// `prepareFrame`/`render`/`renderReuse`/the UI latch all read THIS copy
    /// (prepare reads the slot it is consuming, render/reuse read the front
    /// slot's copy) — never the live `Scene.frame_snapshot`, so a concurrent
    /// game-side mutation cannot tear the in-flight frame.
    ///
    /// Plain struct copy (`slot.snapshot = snap`), never pointers into game
    /// state: `Camera.name` slices alias the live camera names, but the draw
    /// path never dereferences names (projection matrices are precomputed in
    /// the snapshot); GPU handles (`sky_texture`, default copies, probe
    /// views/samplers) are borrowed VALUES under the P3 epoch discipline
    /// (same as every other handle in this slot — never destroyed/retired
    /// through the slot). Fixed-size (no allocation, no deinit); `reset`
    /// clears it so a skipped path can never resurface a prior frame, and
    /// `cpuBytes` deliberately excludes it (fixed scalar, like
    /// `frame_id`/`retire_epoch` — the census counts retained list capacity).
    snapshot: snapshot_mod.SceneFrameSnapshot = .{},
    /// Scene.frame_id that built this slot.
    frame_id: u64 = 0,
    /// GpuRetire epoch opened by the prepareFrame that built this slot.
    retire_epoch: Epoch = 0,

    /// Clear lengths for reuse, retaining all capacity. Covers EVERY list —
    /// including disabled views and disabled shadow bins — so a skipped path
    /// can never resurface another slot's prior frame.
    pub fn reset(self: *FrameDrawSlot) void {
        self.primary.reset();
        for (&self.views) |*q| q.reset();
        self.outline_items.clearRetainingCapacity();
        self.outline_skins.clearRetainingCapacity();
        self.shadow.reset();
        self.staged_instances.clearRetainingCapacity();
        self.ui_vertices.clearRetainingCapacity();
        self.ui_indices.clearRetainingCapacity();
        self.ui_packet = .{};
        self.snapshot = .{};
        self.frame_id = 0;
        self.retire_epoch = 0;
    }

    pub fn deinit(self: *FrameDrawSlot, allocator: std.mem.Allocator) void {
        self.primary.deinit(allocator);
        for (&self.views) |*q| q.deinit(allocator);
        self.outline_items.deinit(allocator);
        self.outline_skins.deinit(allocator);
        self.shadow.deinit(allocator);
        self.staged_instances.deinit(allocator);
        self.ui_vertices.deinit(allocator);
        self.ui_indices.deinit(allocator);
    }

    /// Retained CPU bytes held by this slot (retained capacities × element
    /// sizes across every owned list, including the parallel-cull scratch
    /// and the staged/UI lists). Census helper for the profiler memory
    /// snapshot: capacities, not lengths, because retention is the cost.
    /// The fixed-size staged `snapshot` is deliberately excluded (a constant
    /// scalar like `frame_id`/`retire_epoch`, not retained list capacity).
    pub fn cpuBytes(self: *const FrameDrawSlot) usize {
        var b: usize = 0;
        b += queuesBytes(&self.primary);
        for (&self.views) |*q| b += queuesBytes(q);
        b += listBytes(self.outline_items);
        b += listBytes(self.outline_skins);
        b += listBytes(self.shadow.items);
        b += listBytes(self.shadow.skins);
        b += listBytes(self.staged_instances);
        b += listBytes(self.ui_vertices);
        b += listBytes(self.ui_indices);
        return b;
    }
};

fn listBytes(list: anytype) usize {
    const Child = @typeInfo(@TypeOf(list.items)).pointer.child;
    return list.capacity * @sizeOf(Child);
}

/// Lease spinlock: `std.atomic.Mutex` is tryLock-only (same shape as
/// assets.zig `lockSpin` — this std.Thread ships no blocking Mutex), so
/// block by spinning. Critical sections are a few index words each.
fn lockLease(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) std.atomic.spinLoopHint();
}

fn queuesBytes(q: *const RenderQueues) usize {
    var b: usize = 0;
    b += listBytes(q.items);
    b += listBytes(q.transparent);
    b += listBytes(q.opaque_instanced);
    b += listBytes(q.transparent_instanced);
    b += listBytes(q.instance_matrices);
    b += listBytes(q.transparent_order);
    b += listBytes(q.skin_storage);
    b += listBytes(q.shader_storage);
    b += listBytes(q.coat_storage);
    b += listBytes(q.parallel_scratch.records);
    for (q.parallel_scratch.records.items) |*r| b += listBytes(r.*);
    b += listBytes(q.parallel_scratch.chunk_stats);
    return b;
}

/// The retained slots. `front` is the published consumable slot; the build
/// scratch is the first slot after `front` (modulo SLOT_COUNT) that is
/// neither pinned nor claimed. Flipping `front` IS the publish — the
/// lists themselves never move. See the header for the lease protocol and
/// the threading split (legacy trio single-threaded-only vs claim/pin API).
pub const FrameDraws = struct {
    slots: [SLOT_COUNT]FrameDrawSlot = .{ .{}, .{}, .{} },
    front: usize = 0,
    /// Consumer pins: pinned slots are never build targets and never publish
    /// targets. Set only via `pin`/`pinFront`, cleared only via `unpin`.
    pinned: [SLOT_COUNT]bool = .{false} ** SLOT_COUNT,
    /// Producer claims: slots currently being filled between `claimBack`
    /// and `tryPublish`/`cancelClaim`. `pin` refuses these (`SlotBusy`).
    writing: [SLOT_COUNT]bool = .{false} ** SLOT_COUNT,
    /// Serializes the lease index words (`front`, `pinned`, `writing` and
    /// every counter below): a spinlock (`std.atomic.Mutex` is tryLock-only
    /// in this std, so `lockLease` spins — same shape as assets.zig; the
    /// critical sections are a few index words each). Payload bytes need no
    /// lock: the protocol keeps producer writes and consumer reads on
    /// distinct slots.
    mutex: std.atomic.Mutex = .unlocked,
    /// Pins currently held (lease count, not a boolean: exactly one
    /// pin→unpin pair per hold; double-pin is an error, never a refcount).
    pins_held: usize = 0,
    /// Observability counters (all under `mutex`): lease misuse and
    /// saturation are counted, never silent and never wedging.
    total_pins: u64 = 0,
    pin_denials: u64 = 0,
    unpin_denials: u64 = 0,
    publish_refusals: u64 = 0,
    saturation_skips: u64 = 0,

    /// SINGLE-THREADED ONLY (see header): the back index for the sequential
    /// build/prepare path. First slot after `front` that is neither pinned
    /// nor claimed; asserts one exists (the sequential path never holds pins
    /// or claims across the build, so with 3 slots one is always free).
    pub fn backIndex(self: *const FrameDraws) usize {
        var k: usize = 1;
        while (k < SLOT_COUNT) : (k += 1) {
            const idx = (self.front + k) % SLOT_COUNT;
            if (!self.pinned[idx] and !self.writing[idx]) return idx;
        }
        unreachable; // sequential path holds no pins/claims: a slot is free
    }

    /// SINGLE-THREADED ONLY (see header): the build scratch slot.
    pub fn backSlot(self: *FrameDraws) *FrameDrawSlot {
        return &self.slots[self.backIndex()];
    }

    /// Slot accessor by index (no locking; the caller must own the slot via
    /// the protocol: producer owns claimed slots, consumer owns pinned ones,
    /// the sequential path owns back/front between publish boundaries).
    pub fn slotAt(self: *FrameDraws, idx: usize) *FrameDrawSlot {
        std.debug.assert(idx < SLOT_COUNT);
        return &self.slots[idx];
    }

    pub fn slotAtConst(self: *const FrameDraws, idx: usize) *const FrameDrawSlot {
        std.debug.assert(idx < SLOT_COUNT);
        return &self.slots[idx];
    }

    /// SINGLE-THREADED ONLY (see header): publish the sequentially built
    /// back slot. Must be exactly the current back index and unpinned.
    pub fn publish(self: *FrameDraws, back_idx: usize) void {
        std.debug.assert(back_idx == self.backIndex());
        std.debug.assert(!self.pinned[back_idx]);
        self.front = back_idx;
    }

    /// Concurrent-producer claim: reserve a free slot for writing. Returns
    /// null when every non-front slot is pinned or claimed (consumer
    /// lagging): the producer SKIPS the frame instead of blocking (the
    /// documented latest-wins saturation behavior) and the skip is counted.
    /// The claim marks the slot WRITING until `tryPublish`/`cancelClaim`.
    pub fn claimBack(self: *FrameDraws) ?usize {
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        var k: usize = 1;
        while (k < SLOT_COUNT) : (k += 1) {
            const idx = (self.front + k) % SLOT_COUNT;
            if (!self.pinned[idx] and !self.writing[idx]) {
                self.writing[idx] = true;
                return idx;
            }
        }
        self.saturation_skips += 1;
        return null;
    }

    /// Concurrent-producer publish: hand a claimed slot to the consumer.
    /// Refuses (with counting, state unchanged) a pinned target or a slot
    /// that was never claimed. Clears WRITING and flips `front` on success.
    pub fn tryPublish(self: *FrameDraws, back_idx: usize) LeaseError!void {
        if (back_idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (!self.writing[back_idx]) return LeaseError.NotClaimed;
        if (self.pinned[back_idx]) {
            self.publish_refusals += 1;
            return LeaseError.PinnedSlot;
        }
        std.debug.assert(back_idx != self.front);
        self.writing[back_idx] = false;
        self.front = back_idx;
    }

    /// Release a claim without publishing (producer drops the build, e.g. a
    /// mid-fill abort). Always legal on a claimed slot; a no-op error
    /// (`NotClaimed`) otherwise. Exists so a dropped build can never wedge
    /// the rotation by leaking WRITING.
    pub fn cancelClaim(self: *FrameDraws, back_idx: usize) LeaseError!void {
        if (back_idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (!self.writing[back_idx]) return LeaseError.NotClaimed;
        self.writing[back_idx] = false;
    }

    /// Consumer pin: hold `idx` for presentation. The slot must be a valid,
    /// unpinned, unclaimed slot; a pinned slot stays readable across any
    /// number of later publishes until `unpin`. Double-pin and pinning a
    /// slot under construction are errors (counted), never silent.
    pub fn pin(self: *FrameDraws, idx: usize) LeaseError!void {
        if (idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (self.pinned[idx]) {
            self.pin_denials += 1;
            return LeaseError.AlreadyPinned;
        }
        if (self.writing[idx]) {
            self.pin_denials += 1;
            return LeaseError.SlotBusy;
        }
        self.pinned[idx] = true;
        self.pins_held += 1;
        self.total_pins += 1;
    }

    /// Consumer pin of the current front (the render/present path). Returns
    /// the pinned index; the caller reads `slots[idx]` and must `unpin` it.
    /// Single-threaded render use: the front is never pinned or claimed
    /// there (asserted in debug).
    pub fn pinFront(self: *FrameDraws) usize {
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        const idx = self.front;
        std.debug.assert(!self.pinned[idx]);
        std.debug.assert(!self.writing[idx]);
        self.pinned[idx] = true;
        self.pins_held += 1;
        self.total_pins += 1;
        return idx;
    }

    /// Consumer unpin: release a held slot back to the rotation. MANDATORY
    /// after every successful `pin`/`pinFront`; unpinning a slot that is not
    /// pinned is an error (counted), never a no-op.
    pub fn unpin(self: *FrameDraws, idx: usize) LeaseError!void {
        if (idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (!self.pinned[idx]) {
            self.unpin_denials += 1;
            return LeaseError.NotPinned;
        }
        self.pinned[idx] = false;
        self.pins_held -= 1;
    }

    /// Locked front read for concurrent consumers (freshest publish index;
    /// `pin` it before reading the payload).
    pub fn frontIndex(self: *FrameDraws) usize {
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        return self.front;
    }

    pub fn isPinned(self: *FrameDraws, idx: usize) bool {
        if (idx >= SLOT_COUNT) return false;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        return self.pinned[idx];
    }

    pub fn pinsHeld(self: *FrameDraws) usize {
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        return self.pins_held;
    }

    /// Retained CPU bytes across ALL slots (census helper for the profiler
    /// memory snapshot). Single-threaded census use (prepare/render
    /// boundary); capacities are read without the mutex.
    pub fn cpuBytes(self: *const FrameDraws) usize {
        var b: usize = 0;
        for (&self.slots) |*s| b += s.cpuBytes();
        return b;
    }

    pub fn deinit(self: *FrameDraws, allocator: std.mem.Allocator) void {
        // Debug guard for the mandatory-unpin contract: a leaked pin at
        // shutdown is a programming error (in release the slots still free
        // normally — a missed unpin never wedges, it only shrinks rotation).
        std.debug.assert(self.pins_held == 0);
        for (&self.slots) |*s| s.deinit(allocator);
    }
};

// --- wave-26 tests: 3-slot rotation + pin/lease protocol (CPU-only). ---

const testing = std.testing;

test "wave26: rotation cycles 0-1-2-0 without stalling, back never equals front" {
    var draws = FrameDraws{};
    try testing.expectEqual(SLOT_COUNT, draws.slots.len);
    try testing.expectEqual(@as(usize, 0), draws.front);

    // Six sequential publish round-trips (legacy single-threaded path):
    // each build targets the back, each publish flips to it.
    var expect_front: usize = 0;
    var round: usize = 0;
    while (round < 6) : (round += 1) {
        const back = draws.backIndex();
        try testing.expect(back != draws.front);
        try testing.expectEqual((expect_front + 1) % SLOT_COUNT, back);
        draws.slotAt(back).frame_id = round + 1;
        draws.publish(back);
        expect_front = back;
        try testing.expectEqual(expect_front, draws.front);
        // The published front carries the build's frame; the other slots
        // keep their own (no cross-slot copy, no wipe).
        try testing.expectEqual(@as(u64, round + 1), draws.slotAtConst(draws.front).frame_id);
    }
    // Full cycle proof: after 6 publishes from 0 the front is back at 0.
    try testing.expectEqual(@as(usize, 0), draws.front);
}

test "wave26: claimBack skips the front and pinned slots, publish flips" {
    var draws = FrameDraws{};
    // No pins: the claim lands on the back index (capture it BEFORE the
    // claim — claiming marks the slot WRITING, which backIndex then skips).
    const want0 = draws.backIndex();
    const c0 = draws.claimBack().?;
    try testing.expectEqual(want0, c0);
    draws.slotAt(c0).frame_id = 7;
    try draws.tryPublish(c0);
    try testing.expectEqual(c0, draws.front);

    // Pin the front (a presenting consumer): the next claim must avoid both
    // the front and the pin, and the next publish must flip to it.
    const f = draws.front;
    try draws.pin(f);
    const c1 = draws.claimBack().?;
    try testing.expect(c1 != f);
    try testing.expect(!draws.isPinned(c1));
    draws.slotAt(c1).frame_id = 42;
    try draws.tryPublish(c1);
    try testing.expectEqual(c1, draws.front);
    try testing.expectEqual(@as(u64, 42), draws.slotAtConst(draws.front).frame_id);
    // The old pinned front kept its own frame (presenting consumer undisturbed).
    try testing.expect(draws.isPinned(f));
    try testing.expectEqual(@as(u64, 7), draws.slotAtConst(f).frame_id);
    try draws.unpin(f);
    try testing.expectEqual(@as(usize, 0), draws.pinsHeld());
}

test "wave26: pin/unpin contract — double pin, unpin without pin, counters" {
    var draws = FrameDraws{};

    try testing.expectEqual(@as(usize, 0), draws.pinsHeld());
    try draws.pin(0);
    try testing.expect(draws.isPinned(0));
    try testing.expectEqual(@as(usize, 1), draws.pinsHeld());

    // Double pin: error + denial counted, hold count unchanged.
    try testing.expectError(LeaseError.AlreadyPinned, draws.pin(0));
    try testing.expectEqual(@as(usize, 1), draws.pinsHeld());
    try testing.expectEqual(@as(u64, 1), draws.pin_denials);

    // Invalid slot pins/unpins: errors, no state change.
    try testing.expectError(LeaseError.InvalidSlot, draws.pin(SLOT_COUNT));
    try testing.expectError(LeaseError.InvalidSlot, draws.unpin(SLOT_COUNT));

    // Unpin without pin: error + denial counted.
    try testing.expectError(LeaseError.NotPinned, draws.unpin(1));
    try testing.expectEqual(@as(u64, 1), draws.unpin_denials);

    try draws.unpin(0);
    try testing.expect(!draws.isPinned(0));
    try testing.expectEqual(@as(usize, 0), draws.pinsHeld());
    // Unpin twice: the second is without-pin again.
    try testing.expectError(LeaseError.NotPinned, draws.unpin(0));
    try testing.expectEqual(@as(u64, 2), draws.unpin_denials);

    try testing.expectEqual(@as(u64, 1), draws.total_pins);
}

test "wave26: tryPublish refuses pinned targets, cancelClaim releases writes" {
    var draws = FrameDraws{};

    // Publish of a never-claimed slot: error, front unchanged.
    const f0 = draws.front;
    try testing.expectError(LeaseError.NotClaimed, draws.tryPublish((f0 + 1) % SLOT_COUNT));
    try testing.expectEqual(f0, draws.front);

    // Claim, then pin the CLAIMED slot is refused (SlotBusy); publish while
    // pinned-after-unclaim... first: claim then pin attempt fails.
    const c = draws.claimBack().?;
    try testing.expectError(LeaseError.SlotBusy, draws.pin(c));
    // Cancel the claim: the slot is free again and pinnable.
    try draws.cancelClaim(c);
    try testing.expectError(LeaseError.NotClaimed, draws.cancelClaim(c));
    try draws.pin(c);
    try draws.unpin(c);

    // PinnedSlot refusal (white-box): correct API use can never produce a
    // writing+pinned slot (claim skips pins, pin refuses writing), so the
    // guard below is defense-in-depth — a presented frame is never
    // overwritten even under an unforeseen interleaving. Forge the state by
    // hand (same-file test, private access) and prove the refusal is
    // fail-closed and counted.
    const d = draws.claimBack().?;
    lockLease(&draws.mutex);
    draws.pinned[d] = true;
    draws.mutex.unlock();
    const front_before = draws.front;
    try testing.expectError(LeaseError.PinnedSlot, draws.tryPublish(d));
    try testing.expectEqual(@as(u64, 1), draws.publish_refusals);
    try testing.expectEqual(front_before, draws.front);
    lockLease(&draws.mutex);
    draws.pinned[d] = false;
    draws.mutex.unlock();
    try draws.cancelClaim(d);
    try testing.expectEqual(@as(usize, 0), draws.pinsHeld());
}

test "wave26: saturation with pins held degrades to counted skip, never wedge" {
    var draws = FrameDraws{};

    // front=0 pinned, the other two slots claimed as WRITING (producer
    // mid-fill on both): {front pinned, two writing} leaves nothing free,
    // so the THIRD claim must return null (counted skip), not block.
    try draws.pin(draws.front);
    const w1 = draws.claimBack().?;
    try testing.expect(!draws.isPinned(w1));
    const w2 = draws.claimBack().?;
    try testing.expect(w2 != w1 and w2 != draws.front);
    try testing.expect(draws.claimBack() == null);
    try testing.expectEqual(@as(u64, 1), draws.saturation_skips);
    // Still not wedged: cancel both claims + unpin restores the rotation.
    try draws.cancelClaim(w1);
    try draws.cancelClaim(w2);
    try draws.unpin(draws.front);
    const again = draws.claimBack().?;
    try draws.tryPublish(again);
    try testing.expectEqual(again, draws.front);
    try testing.expectEqual(@as(usize, 0), draws.pinsHeld());
}

// Concurrent stress: single producer claims/builds/publishes while a
// consumer holds pins and reads. Protocol guarantees under test: no torn
// payload reads (frame_id/retire_epoch canary pair always consistent), no
// deadlock (both threads finish), saturation degrades to skip (producer
// counts skips only when the consumer deliberately over-pins).
test "wave26: concurrent producer vs pinned consumer — no tears, no deadlock" {
    var draws = FrameDraws{};
    const total_publishes: u64 = 5000;

    const Ctx = struct {
        draws: *FrameDraws,
        total: u64,
        published: u64 = 0,
        skipped: u64 = 0,
        stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        // Consumer-side observations (consumer thread writes; test thread
        // reads after join).
        reads: u64 = 0,
    };

    const Producer = struct {
        fn run(c: *Ctx) void {
            var seq: u64 = 1;
            while (seq <= c.total) {
                const idx = c.draws.claimBack() orelse {
                    c.skipped += 1;
                    std.atomic.spinLoopHint();
                    continue;
                };
                // Fill BEFORE publish (single producer; the claimed slot is
                // unpinned by construction, the consumer never reads it).
                // Canary pair: retire_epoch is the bitwise inverse of
                // frame_id — any torn concurrent read observes a mismatch.
                c.draws.slotAt(idx).frame_id = seq;
                c.draws.slotAt(idx).retire_epoch = ~seq;
                c.draws.tryPublish(idx) catch |e| switch (e) {
                    // A pin landing between claim and publish is impossible
                    // (pin refuses writing slots); any error here is a bug.
                    else => unreachable,
                };
                seq += 1;
            }
            c.published = c.total;
            c.stop.store(true, .release);
        }
    };

    const Consumer = struct {
        fn run(c: *Ctx) void {
            // Phase 1 (hold one pin, churn reads): pin the CURRENT front
            // with latest-wins retry (a stale front may already be claimed
            // for writing — pin then refuses with SlotBusy and we re-read).
            // With exactly one pin held the producer must make progress with
            // ZERO skips — 3 slots always leave a free one.
            var held: usize = 0;
            while (true) {
                const f = c.draws.frontIndex();
                c.draws.pin(f) catch |e| switch (e) {
                    LeaseError.AlreadyPinned, LeaseError.SlotBusy => continue,
                    else => unreachable,
                };
                held = f;
                break;
            }
            var spins: usize = 0;
            while (spins < 20000) : (spins += 1) {
                const s = c.draws.slotAtConst(held);
                const fid = s.frame_id;
                const canary = s.retire_epoch;
                // The pinned slot is never written by the producer: the
                // pair is always the initial (0,0) or one fully published
                // (seq,~seq) pair — never a mix.
                if (!((fid == 0 and canary == 0) or canary == ~fid)) unreachable;
                c.reads += 1;
                std.atomic.spinLoopHint();
            }
            c.draws.unpin(held) catch unreachable;

            // Phase 2 (pin/unpin churn on the live front): every read must
            // still be canary-consistent; SlotBusy pins just retry. A freshly
            // pinned front may still be unpublished (0,0) when the producer
            // is slow to start — that is consistent, not torn.
            while (!c.stop.load(.acquire)) {
                const idx = c.draws.frontIndex();
                c.draws.pin(idx) catch |e| switch (e) {
                    LeaseError.AlreadyPinned, LeaseError.SlotBusy => continue,
                    else => unreachable,
                };
                const s = c.draws.slotAtConst(idx);
                const fid = s.frame_id;
                const canary = s.retire_epoch;
                if (!((fid == 0 and canary == 0) or canary == ~fid)) unreachable;
                c.reads += 1;
                c.draws.unpin(idx) catch unreachable;
            }
        }
    };

    var ctx = Ctx{ .draws = &draws, .total = total_publishes };
    const prod = try std.Thread.spawn(.{}, Producer.run, .{&ctx});
    const cons = try std.Thread.spawn(.{}, Consumer.run, .{&ctx});
    prod.join();
    cons.join();

    try testing.expectEqual(total_publishes, ctx.published);
    // Phase 1 held exactly one pin: the producer always had a free slot, so
    // saturation skips must be zero (phase 2 holds at most one pin with one
    // claim outstanding — 3 slots never saturate there either).
    try testing.expectEqual(@as(u64, 0), ctx.skipped);
    try testing.expectEqual(@as(u64, 0), draws.saturation_skips);
    try testing.expect(ctx.reads > 0);
    try testing.expectEqual(@as(usize, 0), draws.pinsHeld());
    try testing.expectEqual(@as(u64, 0), draws.publish_refusals);
    // Tail: every slot holds a consistent canary pair (no slot was ever
    // published half-written); never-published slots are still (0,0).
    for (0..SLOT_COUNT) |i| {
        const s = draws.slotAtConst(i);
        const fid = s.frame_id;
        const canary = s.retire_epoch;
        try testing.expect((fid == 0 and canary == 0) or canary == ~fid);
    }
    draws.deinit(testing.allocator);
}
