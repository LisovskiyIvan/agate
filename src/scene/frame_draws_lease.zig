//! Slot rotation + consumer pin/lease protocol: the retained slots,
//! `SLOT_COUNT` / `LeaseError` / `HandoffClaim`, and the `FrameDraws`
//! claim/publish/handoff/pin API with its observability counters.
//! Section of the `frame_draws` facade (pure move; see `frame_draws.zig`
//! for the full ownership / rotation / lease contract).

const std = @import("std");
const slot_mod = @import("frame_draws_slot.zig");

/// Slot payload type owned by `frame_draws_slot.zig` (private alias so the
/// `FrameDraws.slots` field below stays byte-identical).
const FrameDrawSlot = slot_mod.FrameDrawSlot;

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

pub const HandoffClaim = struct {
    slot: usize,
    seq: usize,
    has_scene_build: bool,
};

/// Lease spinlock: `std.atomic.Mutex` is tryLock-only (same shape as
/// assets.zig `lockSpin` — this std.Thread ships no blocking Mutex), so
/// block by spinning. Critical sections are a few index words each.
fn lockLease(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) std.atomic.spinLoopHint();
}

/// The retained slots. `front` is the published consumable slot; the build
/// scratch is the first slot after `front` (modulo SLOT_COUNT) that is
/// neither pinned nor claimed. Flipping `front` IS the publish — the
/// lists themselves never move. See the header for the lease protocol and
/// the threading split (sequential path: single-threaded-only; concurrent
/// paths: claim/pin API only).
pub const FrameDraws = struct {
    slots: [SLOT_COUNT]FrameDrawSlot = .{ .{}, .{}, .{} },
    front: usize = 0,
    /// Consumer pins: pinned slots are never build targets and never publish
    /// targets. Set only via `pin`/`pinFront`, cleared only via `unpin`.
    pinned: [SLOT_COUNT]bool = @splat(false),
    /// Shared read leases used by producer commit while the context may also
    /// be presenting the current front. Multiple readers are safe; every
    /// writer/rotation path treats a nonzero count as pinned.
    read_pins: [SLOT_COUNT]usize = @splat(0),
    /// Producer claims: slots currently being filled between `claimBack`
    /// and `tryPublish`/`cancelClaim`. `pin` refuses these (`SlotBusy`).
    writing: [SLOT_COUNT]bool = @splat(false),
    /// Pending handoff slot: the slot published by `releaseHandoff` waiting
    /// for the prepare latch. `claimBack` avoids this slot when another free
    /// slot exists, preventing the game thread from reclaiming the in-flight
    /// handoff and degrading prepare to a skip under high tick rates.
    handoff: ?usize = null,
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

    /// Concurrent-producer claim: reserve a free slot for writing. Returns
    /// null when every non-front slot is pinned or claimed (consumer
    /// lagging): the producer SKIPS the frame instead of blocking (the
    /// documented latest-wins saturation behavior) and the skip is counted.
    /// The claim marks the slot WRITING until `tryPublish`/`cancelClaim`.
    /// Two-pass selection: prefers any slot that is neither pinned, nor
    /// writing, nor the pending handoff slot awaiting prepare latch.
    pub fn claimBack(self: *FrameDraws) ?usize {
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        // Pass 1: find a slot that is NOT pinned, NOT writing, and NOT the pending handoff.
        var k: usize = 1;
        while (k < SLOT_COUNT) : (k += 1) {
            const idx = (self.front + k) % SLOT_COUNT;
            if (!self.pinned[idx] and self.read_pins[idx] == 0 and !self.writing[idx] and (self.handoff == null or self.handoff.? != idx)) {
                self.writing[idx] = true;
                return idx;
            }
        }
        // Pass 2: if all other non-pinned slots are busy, claim the pending handoff
        // slot (superseding the unconsumed frame, latest-wins).
        k = 1;
        while (k < SLOT_COUNT) : (k += 1) {
            const idx = (self.front + k) % SLOT_COUNT;
            if (!self.pinned[idx] and self.read_pins[idx] == 0 and !self.writing[idx]) {
                self.writing[idx] = true;
                if (self.handoff == idx) self.handoff = null;
                return idx;
            }
        }
        self.saturation_skips += 1;
        return null;
    }

    /// Locked specific-slot claim (prepare-latch side): reserve
    /// exactly `idx` for writing — the handoff slot a concurrent game build
    /// published via `releaseHandoff` (`Scene.build_slot`), which a blind
    /// `claimBack` is not guaranteed to return once game||prepare overlap.
    /// Fails closed and counted, state unchanged: `PinnedSlot` (counted in
    /// `publish_refusals` — a presented frame is never overwritten) when the
    /// slot is pinned, `SlotBusy` (counted in `saturation_skips` — skip the
    /// frame, latest-wins) when another producer holds it for writing,
    /// `InvalidSlot` when out of range. The pinned check runs first: correct
    /// API use can never produce writing+pinned (claim skips pins, pin
    /// refuses writing), so either order is defense-in-depth and the
    /// presented frame wins ties.
    pub fn claimSlot(self: *FrameDraws, idx: usize) LeaseError!void {
        if (idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (self.pinned[idx] or self.read_pins[idx] != 0) {
            self.publish_refusals += 1;
            return LeaseError.PinnedSlot;
        }
        if (self.writing[idx]) {
            self.saturation_skips += 1;
            return LeaseError.SlotBusy;
        }
        self.writing[idx] = true;
        if (self.handoff == idx) self.handoff = null;
    }

    /// Atomically reads the producer's published `(build_slot, build_seq)`
    /// pair and claims that exact handoff under the same lease mutex used by
    /// `releaseHandoffWithSeq`. This prevents the consumer from pairing an
    /// old sequence with a newer slot when the producer publishes between
    /// separate atomic loads. `null` means no generation is pending; lease
    /// failures are counted and returned.
    pub fn claimLatestHandoff(
        self: *FrameDraws,
        build_slot: *const std.atomic.Value(usize),
        build_seq: *const std.atomic.Value(usize),
        last_latched_seq: usize,
        require_scene_build: bool,
    ) LeaseError!?HandoffClaim {
        lockLease(&self.mutex);
        defer self.mutex.unlock();

        const seq = build_seq.load(.monotonic);
        if (seq == last_latched_seq) return null;
        const idx = build_slot.load(.monotonic);
        if (idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        // Lease words FIRST, payload second: every payload writer holds
        // `writing` (or the publish mutex edge that clears it) under this
        // same mutex, so once we observe an unclaimed, unpinned slot here
        // the payload reads below are race-free. `claimBack` pass 2 may
        // supersede this very handoff slot (reset + rebuild under its own
        // claim) — that window is exactly `writing == true` for us.
        if (self.pinned[idx] or self.read_pins[idx] != 0) {
            self.publish_refusals += 1;
            return LeaseError.PinnedSlot;
        }
        if (self.writing[idx]) {
            self.saturation_skips += 1;
            return LeaseError.SlotBusy;
        }
        // Pair validation: publication stamped the slot under this same
        // mutex, so the slot stamp is the authoritative marker that this
        // exact (slot, seq) pair is still the pending handoff. The `handoff`
        // field itself is NOT checked here: claimSlot/cancelClaim legitimately
        // clear it as a reservation marker (e.g. a contention holder probing
        // the slot), and the pending build must stay claimable after that
        // reservation is released. A stamp mismatch means the slot was
        // reclaimed and reset between publish and this claim — fail closed
        // as a counted skip rather than consuming the wrong payload.
        if (self.slots[idx].build_seq != seq) {
            self.saturation_skips += 1;
            return LeaseError.SlotBusy;
        }
        const has_scene_build = self.slots[idx].has_scene_build;
        if (require_scene_build and !has_scene_build) return null;
        self.writing[idx] = true;
        self.handoff = null;
        return HandoffClaim{ .slot = idx, .seq = seq, .has_scene_build = has_scene_build };
    }

    /// Concurrent-producer publish: hand a claimed slot to the consumer.
    /// Refuses (with counting, state unchanged) a pinned target or a slot
    /// that was never claimed. Clears WRITING and flips `front` on success.
    pub fn tryPublish(self: *FrameDraws, back_idx: usize) LeaseError!void {
        if (back_idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (!self.writing[back_idx]) return LeaseError.NotClaimed;
        if (self.pinned[back_idx] or self.read_pins[back_idx] != 0) {
            self.publish_refusals += 1;
            return LeaseError.PinnedSlot;
        }
        std.debug.assert(back_idx != self.front);
        self.writing[back_idx] = false;
        self.front = back_idx;
        if (self.handoff == back_idx) self.handoff = null;
    }

    /// Game-side handoff release (concurrent-build primitive): hand
    /// a claimed slot to the prepare latch WITHOUT flipping `front` (the
    /// front flip stays context-owned in the staged finish). Clears WRITING so
    /// the slot is a normal rotation member again; the payload is kept for
    /// prepare, which consumes it via the `build_slot`/`build_seq` handoff
    /// and flips `front` itself at latch time. Refuses (counted, state
    /// unchanged) a pinned target or a never-claimed slot, exactly like
    /// `tryPublish` — a presented frame is never overwritten even under an
    /// unforeseen interleaving (`pin` refuses writing slots, so correct API
    /// use can never produce writing+pinned; the guard is defense-in-depth).
    pub fn releaseHandoff(self: *FrameDraws, back_idx: usize) LeaseError!void {
        if (back_idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (!self.writing[back_idx]) return LeaseError.NotClaimed;
        if (self.pinned[back_idx] or self.read_pins[back_idx] != 0) {
            self.publish_refusals += 1;
            return LeaseError.PinnedSlot;
        }
        self.writing[back_idx] = false;
        self.handoff = back_idx;
    }

    /// Atomically releases WRITING and registers handoff while storing the
    /// atomic build_slot and build_seq under the lease mutex, guaranteeing
    /// consumers never see a torn pair or a SlotBusy refusal on a freshly
    /// published frame.
    pub fn releaseHandoffWithSeq(
        self: *FrameDraws,
        back_idx: usize,
        seq: usize,
        build_slot: *std.atomic.Value(usize),
        build_seq: *std.atomic.Value(usize),
    ) LeaseError!void {
        if (back_idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (!self.writing[back_idx]) return LeaseError.NotClaimed;
        if (self.pinned[back_idx] or self.read_pins[back_idx] != 0) {
            self.publish_refusals += 1;
            return LeaseError.PinnedSlot;
        }
        self.writing[back_idx] = false;
        self.handoff = back_idx;
        build_slot.store(back_idx, .release);
        build_seq.store(seq, .release);
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
        if (self.handoff == back_idx) self.handoff = null;
    }

    /// Releases a prepare-held handoff claim after cancellation. Restore it
    /// only if no newer producer generation replaced the published pair
    /// while prepare was in flight; newest-wins remains intact otherwise.
    pub fn cancelHandoffClaim(
        self: *FrameDraws,
        back_idx: usize,
        seq: usize,
        build_slot: *const std.atomic.Value(usize),
        build_seq: *const std.atomic.Value(usize),
    ) LeaseError!void {
        if (back_idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (!self.writing[back_idx]) return LeaseError.NotClaimed;
        self.writing[back_idx] = false;
        if (build_seq.load(.monotonic) == seq and build_slot.load(.monotonic) == back_idx) {
            self.handoff = back_idx;
        }
    }

    /// Consumer pin: hold `idx` for presentation. The slot must be a valid,
    /// unpinned, unclaimed slot; a pinned slot stays readable across any
    /// number of later publishes until `unpin`. Double-pin and pinning a
    /// slot under construction are errors (counted), never silent.
    pub fn pin(self: *FrameDraws, idx: usize) LeaseError!void {
        if (idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (self.pinned[idx] or self.read_pins[idx] != 0) {
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

    /// Shared read lease on the current front. Unlike the presentation pin,
    /// this may coexist with another reader (including render) because the
    /// slot is immutable while published. It prevents the slot from being
    /// reclaimed if a concurrent prepare advances `front` during the read.
    pub fn pinFrontReader(self: *FrameDraws) usize {
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        const idx = self.front;
        std.debug.assert(!self.writing[idx]);
        self.read_pins[idx] += 1;
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

    pub fn unpinReader(self: *FrameDraws, idx: usize) LeaseError!void {
        if (idx >= SLOT_COUNT) return LeaseError.InvalidSlot;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        if (self.read_pins[idx] == 0) {
            self.unpin_denials += 1;
            return LeaseError.NotPinned;
        }
        self.read_pins[idx] -= 1;
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

    pub fn isReadPinned(self: *FrameDraws, idx: usize) bool {
        if (idx >= SLOT_COUNT) return false;
        lockLease(&self.mutex);
        defer self.mutex.unlock();
        return self.read_pins[idx] != 0;
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
