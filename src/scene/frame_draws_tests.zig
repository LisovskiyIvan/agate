//! Tests for `frame_draws.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const draws_mod = @import("frame_draws.zig");
const FrameDraws = draws_mod.FrameDraws;
const SLOT_COUNT = draws_mod.SLOT_COUNT;
const LeaseError = draws_mod.LeaseError;

// --- wave-26 tests: 3-slot rotation + pin/lease protocol (CPU-only). ---

const testing = std.testing;

test "wave26: rotation cycles 0-1-2-0 without stalling, back never equals front" {
    var draws = FrameDraws{};
    try testing.expectEqual(SLOT_COUNT, draws.slots.len);
    try testing.expectEqual(@as(usize, 0), draws.front);

    // Six sequential publish round-trips (claim/tryPublish path):
    // each build targets a claimed back, each publish flips to it.
    var expect_front: usize = 0;
    var round: usize = 0;
    while (round < 6) : (round += 1) {
        const back = draws.claimBack().?;
        try testing.expect(back != draws.front);
        try testing.expectEqual((expect_front + 1) % SLOT_COUNT, back);
        draws.slotAt(back).frame_id = round + 1;
        try draws.tryPublish(back);
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
    // No pins: the claim lands off the front.
    const c0 = draws.claimBack().?;
    try testing.expect(c0 != draws.front);
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

test "wave39: shared front reader coexists with render pin and blocks slot reuse" {
    var draws = FrameDraws{};
    const front = draws.pinFrontReader();
    try testing.expectEqual(draws.front, front);
    try testing.expect(draws.isReadPinned(front));
    try testing.expect(!draws.isPinned(front));

    // Rendering is a second immutable reader of the same published slot.
    try testing.expectEqual(front, draws.pinFront());
    try testing.expect(draws.isPinned(front));

    // The producer/context may advance the front, but neither lease allows
    // the old slot to be reclaimed while commit/render still reads it.
    const next = draws.claimBack().?;
    draws.slotAt(next).frame_id = 1;
    try draws.tryPublish(next);
    const third = draws.claimBack().?;
    draws.slotAt(third).frame_id = 2;
    try draws.tryPublish(third);
    try testing.expect(draws.front != front);
    try testing.expectError(LeaseError.PinnedSlot, draws.claimSlot(front));

    try draws.unpin(front);
    try testing.expect(!draws.isPinned(front));
    try testing.expect(draws.isReadPinned(front));
    try testing.expectError(LeaseError.PinnedSlot, draws.claimSlot(front));
    try draws.unpinReader(front);
    try testing.expect(!draws.isReadPinned(front));
    try draws.claimSlot(front);
    try draws.cancelClaim(front);
    try testing.expectEqual(@as(usize, 0), draws.pinsHeld());
}

test "wave39: handoff slot and generation are claimed as one counted pair" {
    var draws = FrameDraws{};
    var build_slot = std.atomic.Value(usize).init(0);
    var build_seq = std.atomic.Value(usize).init(0);

    const slot = draws.claimBack().?;
    draws.slotAt(slot).build_seq = 7;
    draws.slotAt(slot).has_scene_build = true;
    try draws.releaseHandoffWithSeq(slot, 7, &build_slot, &build_seq);
    const claimed = (try draws.claimLatestHandoff(&build_slot, &build_seq, 0, true)).?;
    try testing.expectEqual(slot, claimed.slot);
    try testing.expectEqual(@as(usize, 7), claimed.seq);
    try testing.expect(claimed.has_scene_build);
    try draws.tryPublish(claimed.slot);
    try testing.expect((try draws.claimLatestHandoff(&build_slot, &build_seq, 7, true)) == null);

    const stale_slot = draws.claimBack().?;
    draws.slotAt(stale_slot).build_seq = 8;
    try draws.releaseHandoffWithSeq(stale_slot, 9, &build_slot, &build_seq);
    try testing.expectError(LeaseError.SlotBusy, draws.claimLatestHandoff(&build_slot, &build_seq, 7, false));
    try testing.expectEqual(@as(u64, 1), draws.saturation_skips);
    // The fail-closed mismatch did not consume or wedge the handoff.
    try draws.claimSlot(stale_slot);
    try draws.cancelClaim(stale_slot);
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
    // hand (direct field access; single-threaded here, so no lease lock is
    // needed — tryPublish takes it itself) and prove the refusal is
    // fail-closed and counted.
    const d = draws.claimBack().?;
    draws.pinned[d] = true;
    const front_before = draws.front;
    try testing.expectError(LeaseError.PinnedSlot, draws.tryPublish(d));
    try testing.expectEqual(@as(u64, 1), draws.publish_refusals);
    try testing.expectEqual(front_before, draws.front);
    draws.pinned[d] = false;
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

// --- wave-29 tests: concurrent-build primitive proof (CPU-only). ---
//
// Producer claims -> builds (frame generation + canary pair + a slot-owned
// snapshot word, all stamped from one seq) -> publishes while a consumer
// pins the front as a prepare-equivalent read (pin -> validate -> unpin).
// Invariants under test: fresh front deliveries never decrease (publications
// are strictly increasing by single-producer construction; the front only
// ever flips to a newer publish), every pinned read is canary-consistent
// (no torn payload, including the slot-owned snapshot word), an uncongested
// rotation never saturates (one held pin leaves a free slot), no deadlock
// over 20000 publishes, and the tail front holds exactly the last publish.
test "wave29: concurrent claim/build/publish vs pin/prepare-read — increasing ids, no tears, no deadlock" {
    var draws = FrameDraws{};
    const total_publishes: u64 = 20000;

    const Ctx = struct {
        draws: *FrameDraws,
        total: u64,
        skipped: u64 = 0,
        stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        reads: u64 = 0,
        max_seen: u64 = 0,
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
                // Build BEFORE publish (owns the claimed slot; the consumer
                // never reads it): one generation stamps the frame id, the
                // inverse canary, and the slot-owned snapshot word.
                const s = c.draws.slotAt(idx);
                s.frame_id = seq;
                s.retire_epoch = ~seq;
                s.snapshot.frame_id = seq;
                c.draws.tryPublish(idx) catch |e| switch (e) {
                    // A pin landing between claim and publish is impossible
                    // (pin refuses writing slots); any error here is a bug.
                    else => unreachable,
                };
                seq += 1;
            }
            c.stop.store(true, .release);
        }
    };

    const Consumer = struct {
        fn run(c: *Ctx) void {
            while (!c.stop.load(.acquire)) {
                const f = c.draws.frontIndex();
                c.draws.pin(f) catch |e| switch (e) {
                    LeaseError.AlreadyPinned, LeaseError.SlotBusy => continue,
                    else => unreachable,
                };
                // Stale pin (front moved between frontIndex and pin): only
                // the canary check applies — monotonicity is a property of
                // fresh front deliveries, and holding an older pinned slot
                // is a legal explicit hold, never a resurfacing.
                const fresh = c.draws.frontIndex() == f;
                const s = c.draws.slotAtConst(f);
                const fid = s.frame_id;
                const canary = s.retire_epoch;
                const snap_id = s.snapshot.frame_id;
                if (!((fid == 0 and canary == 0 and snap_id == 0) or
                    (canary == ~fid and snap_id == fid))) unreachable;
                if (fresh and fid != 0) {
                    if (fid < c.max_seen) unreachable;
                    if (fid > c.max_seen) c.max_seen = fid;
                }
                c.reads += 1;
                c.draws.unpin(f) catch unreachable;
            }
        }
    };

    var ctx = Ctx{ .draws = &draws, .total = total_publishes };
    const prod = try std.Thread.spawn(.{}, Producer.run, .{&ctx});
    const cons = try std.Thread.spawn(.{}, Consumer.run, .{&ctx});
    prod.join();
    cons.join();

    // Uncongested rotation (at most one pin held at a time): the producer
    // always had a free slot — zero skips, all counted.
    try testing.expectEqual(@as(u64, 0), ctx.skipped);
    try testing.expectEqual(@as(u64, 0), draws.saturation_skips);
    try testing.expect(ctx.reads > 0);
    try testing.expectEqual(@as(usize, 0), draws.pinsHeld());
    try testing.expectEqual(@as(u64, 0), draws.publish_refusals);
    // The tail front holds exactly the last publish (strictly-increasing
    // publications end to end: nothing newer-or-older may surface).
    const tail = draws.frontIndex();
    try draws.pin(tail);
    const ts = draws.slotAtConst(tail);
    try testing.expectEqual(total_publishes, ts.frame_id);
    try testing.expectEqual(~total_publishes, ts.retire_epoch);
    try testing.expectEqual(total_publishes, ts.snapshot.frame_id);
    // max_seen is a liveness witness (the consumer observed real fresh
    // deliveries, never the future): it usually equals total, but the last
    // publishes may land after the consumer's final iteration — the tail
    // pin above is the deterministic end-to-end proof.
    try testing.expect(ctx.max_seen > 0);
    try testing.expect(ctx.max_seen <= total_publishes);
    try draws.unpin(tail);
    // Tail: every slot canary-consistent (triple word, snapshot included).
    for (0..SLOT_COUNT) |i| {
        const s = draws.slotAtConst(i);
        const fid = s.frame_id;
        try testing.expect((fid == 0 and s.retire_epoch == 0 and s.snapshot.frame_id == 0) or
            (s.retire_epoch == ~fid and s.snapshot.frame_id == fid));
    }
    draws.deinit(testing.allocator);
}

// Saturation path: when the consumer holds pins on every slot, the producer
// must observe counted skips (claimBack null) and keep spinning — never
// block, never wedge — then drain to completion once pins release.
test "wave29: consumer-held pins force the producer to skip, never stall" {
    var draws = FrameDraws{};
    const total_publishes: u64 = 3000;

    const Ctx = struct {
        draws: *FrameDraws,
        total: u64,
        skipped: u64 = 0,
        stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
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
                const s = c.draws.slotAt(idx);
                s.frame_id = seq;
                s.retire_epoch = ~seq;
                s.snapshot.frame_id = seq;
                c.draws.tryPublish(idx) catch |e| switch (e) {
                    else => unreachable,
                };
                seq += 1;
                // Slowed publish (same precedent as the handoff 3-slot drain
                // test's slowed consumer): gives the consumer thread time to
                // accumulate pins on every slot so the saturation path is
                // really exercised instead of lapped.
                var spin: usize = 0;
                while (spin < 2000) : (spin += 1) std.atomic.spinLoopHint();
            }
            c.stop.store(true, .release);
        }
    };

    const Consumer = struct {
        fn run(c: *Ctx) void {
            while (!c.stop.load(.acquire)) {
                // Accumulate a pin on every new front without releasing:
                // consecutive fronts always differ, so after two pins the
                // next publish must land on the third slot — pinnable too —
                // and the rotation saturates deterministically.
                const f = c.draws.frontIndex();
                c.draws.pin(f) catch |e| switch (e) {
                    LeaseError.AlreadyPinned, LeaseError.SlotBusy => {},
                    else => unreachable,
                };
                if (c.draws.pinsHeld() >= SLOT_COUNT) {
                    // Full hold: the producer must be skipping now. Hold
                    // briefly so skips accumulate, then release everything
                    // and let it drain.
                    var spin: usize = 0;
                    while (spin < 20000) : (spin += 1) std.atomic.spinLoopHint();
                    for (0..SLOT_COUNT) |i| {
                        if (c.draws.isPinned(i)) c.draws.unpin(i) catch unreachable;
                    }
                }
                // Every pinned slot stays canary-consistent under the hold.
                for (0..SLOT_COUNT) |i| {
                    if (!c.draws.isPinned(i)) continue;
                    const s = c.draws.slotAtConst(i);
                    const fid = s.frame_id;
                    if (!((fid == 0 and s.retire_epoch == 0 and s.snapshot.frame_id == 0) or
                        (s.retire_epoch == ~fid and s.snapshot.frame_id == fid))) unreachable;
                    c.reads += 1;
                }
            }
            for (0..SLOT_COUNT) |i| {
                if (c.draws.isPinned(i)) c.draws.unpin(i) catch unreachable;
            }
        }
    };

    var ctx = Ctx{ .draws = &draws, .total = total_publishes };
    const prod = try std.Thread.spawn(.{}, Producer.run, .{&ctx});
    const cons = try std.Thread.spawn(.{}, Consumer.run, .{&ctx});
    prod.join();
    cons.join();

    // Saturation really happened (skips observed AND counted one-for-one),
    // yet the producer still completed every publish: skip, never stall.
    try testing.expect(ctx.skipped > 0);
    try testing.expectEqual(ctx.skipped, draws.saturation_skips);
    try testing.expect(ctx.reads > 0);
    try testing.expectEqual(@as(usize, 0), draws.pinsHeld());
    try testing.expectEqual(@as(u64, 0), draws.publish_refusals);
    const tail = draws.frontIndex();
    try draws.pin(tail);
    try testing.expectEqual(total_publishes, draws.slotAtConst(tail).frame_id);
    try draws.unpin(tail);
    draws.deinit(testing.allocator);
}

// --- wave-31 tests: prepare-side specific claim (`claimSlot`). ---

// `claimSlot` reserves exactly the handoff slot: success marks WRITING and
// composes with `tryPublish`; a pinned target refuses with `PinnedSlot`
// (counted in `publish_refusals`, presented frame never overwritten); an
// already-claimed target refuses with `SlotBusy` (counted in
// `saturation_skips`, skip-the-frame); out-of-range is `InvalidSlot`. State
// is unchanged on every refusal.
test "wave31: claimSlot reserves the handoff slot, refusals are fail-closed and counted" {
    var draws = FrameDraws{};

    // Success: the slot is marked WRITING and publishes normally.
    try draws.claimSlot(1);
    draws.slotAt(1).frame_id = 11;
    try draws.tryPublish(1);
    try testing.expectEqual(@as(usize, 1), draws.front);
    try testing.expectEqual(@as(u64, 11), draws.slotAtConst(1).frame_id);

    // Already claimed (a concurrent producer mid-fill): SlotBusy + counted.
    const c = draws.claimBack().?;
    try testing.expectError(LeaseError.SlotBusy, draws.claimSlot(c));
    try testing.expectEqual(@as(u64, 1), draws.saturation_skips);
    // Release then re-claim the same slot: now it succeeds.
    try draws.cancelClaim(c);
    try draws.claimSlot(c);
    try draws.cancelClaim(c);

    // Pinned (a presenting consumer): PinnedSlot + counted, front unchanged.
    const f = draws.front;
    try draws.pin(f);
    const front_before = draws.front;
    try testing.expectError(LeaseError.PinnedSlot, draws.claimSlot(f));
    try testing.expectEqual(@as(u64, 1), draws.publish_refusals);
    try testing.expectEqual(front_before, draws.front);
    try draws.unpin(f);

    // Out of range: InvalidSlot, no counter moves.
    try testing.expectError(LeaseError.InvalidSlot, draws.claimSlot(SLOT_COUNT));
    try testing.expectEqual(@as(u64, 1), draws.saturation_skips);
    try testing.expectEqual(@as(u64, 1), draws.publish_refusals);

    // Clean teardown: no pins held, nothing left WRITING (a claim without a
    // matching release would wedge the rotation — the wave-31 prepare audit
    // requires every claim to pair with publish/cancel at every exit).
    try testing.expectEqual(@as(usize, 0), draws.pinsHeld());
    for (0..SLOT_COUNT) |i| {
        try testing.expectError(LeaseError.NotClaimed, draws.cancelClaim(i));
    }
    draws.deinit(testing.allocator);
}

test "wave42: handoff slot is protected from claimBack when free slot exists" {
    var draws = FrameDraws{};

    // Initial state: front is 0.
    try testing.expectEqual(@as(usize, 0), draws.front);
    try testing.expectEqual(@as(?usize, null), draws.handoff);

    // Producer claims back: gets slot 1.
    const slot1 = draws.claimBack().?;
    try testing.expectEqual(@as(usize, 1), slot1);

    // Producer hands off slot 1 to consumer.
    try draws.releaseHandoff(slot1);
    try testing.expectEqual(@as(?usize, 1), draws.handoff);

    // Producer immediately claims next slot BEFORE consumer consumes slot 1:
    // MUST NOT reclaim slot 1; MUST claim slot 2 instead!
    const slot2 = draws.claimBack().?;
    try testing.expectEqual(@as(usize, 2), slot2);
    try testing.expect(slot2 != slot1);

    // Consumer latches slot 1 while producer is filling slot 2:
    // Slot 1 is NOT writing (producer is in slot 2), so claimSlot(1) SUCCEEDS without contention!
    try draws.claimSlot(slot1);
    try testing.expectEqual(@as(?usize, null), draws.handoff);

    // Consumer publishes slot 1 (front flips to 1).
    try draws.tryPublish(slot1);
    try testing.expectEqual(@as(usize, 1), draws.front);

    // Producer hands off slot 2.
    try draws.releaseHandoff(slot2);
    try testing.expectEqual(@as(?usize, 2), draws.handoff);

    // Producer claims next slot: front is 1, (1+1)%3 = 2 is handoff, so claimBack avoids 2 and claims 0!
    const slot0 = draws.claimBack().?;
    try testing.expectEqual(@as(usize, 0), slot0);

    // Consumer latches slot 2.
    try draws.claimSlot(slot2);
    try draws.tryPublish(slot2);
    try testing.expectEqual(@as(usize, 2), draws.front);

    // Teardown.
    try draws.cancelClaim(slot0);
    draws.deinit(testing.allocator);
}
