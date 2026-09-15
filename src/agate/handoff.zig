//! Latest-wins frame-state handoff (stage 3 groundwork, see REFACTOR.md).
//!
//! A publisher (the simulation side) fills one of N payload slots and
//! publishes it; a single consumer (the render side) takes the newest
//! complete payload and releases it. Older unclaimed payloads are dropped
//! — that is the point: a slow consumer never queues up stale frames, it
//! always renders the freshest complete state, and a fast publisher never
//! blocks (it just reuses free slots or skips).
//!
//! Single-producer, single-consumer. The publisher is the game/update
//! thread, the consumer the render thread. Freshness ordering comes from a
//! global publish counter stamped at publish time; with two publishers the
//! counter is assigned before the meta store lands, so "newest" would be
//! ill-defined — which is why the CAS in `claim` guards the slot against
//! accidental double-claiming but the type does not support concurrent
//! publishers.
//!
//! Lock-free: each slot carries `seq << 2 | state` in one atomic word
//! (free / writing / published). `takeLatest` copies out the highest-
//! sequence published slot and releases every published slot, so stale
//! frames never resurface.
//!
//! Copy-out semantics: `takeLatest(dest)` memcpys the payload — keep `T` a
//! plain data struct (no pointers into itself). Payload reads happen while
//! the slot is still `published`, so a publisher cannot race the read: it
//! can only reclaim the slot after release.

const std = @import("std");

const STATE_FREE: u64 = 0;
const STATE_WRITING: u64 = 1;
const STATE_PUBLISHED: u64 = 2;

pub fn Handoff(comptime T: type, comptime slot_count: usize) type {
    comptime std.debug.assert(slot_count >= 2); // double buffer minimum
    comptime std.debug.assert((slot_count & (slot_count - 1)) == 0); // power of two
    return struct {
        const Self = @This();

        slots: [slot_count]T = undefined,
        /// Global publish counter: sequence numbers must be unique and
        /// monotonic across ALL slots, otherwise takeLatest cannot tell
        /// which published frame is newer.
        publish_seq: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        meta: [slot_count]std.atomic.Value(u64) = blk: {
            var m: [slot_count]std.atomic.Value(u64) = undefined;
            for (&m) |*v| v.* = std.atomic.Value(u64).init(STATE_FREE);
            break :blk m;
        },

        /// Reserves a free slot for writing. Returns null when every slot
        /// is published or being written (consumer lagging) — publishers
        /// skip the frame instead of blocking.
        pub fn claim(self: *Self) ?usize {
            for (0..slot_count) |i| {
                const m = self.meta[i].load(.monotonic);
                if (m & 3 != STATE_FREE) continue;
                if (self.meta[i].cmpxchgWeak(m, (m & ~@as(u64, 3)) | STATE_WRITING, .acquire, .monotonic) == null) {
                    return i;
                }
            }
            return null;
        }

        /// Access for filling a claimed slot. Valid until `publish`.
        pub fn slot(self: *Self, i: usize) *T {
            return &self.slots[i];
        }

        /// Publishes a claimed slot (release: payload writes happen-before
        /// the consumer's acquire read). Sequence comes from a global
        /// counter, so "newest" is well defined across slots.
        pub fn publish(self: *Self, i: usize) void {
            const seq = self.publish_seq.fetchAdd(1, .monotonic);
            self.meta[i].store(seq << 2 | STATE_PUBLISHED, .release);
        }

        /// Drops every published slot WITHOUT copying. Publisher-side drain
        /// for mailbox saturation: the publisher could not claim a free
        /// slot, so it discards the stale published frames and retries the
        /// claim to publish the newest payload. Only safe while the consumer
        /// is excluded (the apps hold phase_mutex across update and render);
        /// with a concurrent consumer a takeLatest could race the release and
        /// either miss or re-deliver a frame. The type's single-consumer
        /// contract otherwise stays.
        pub fn releasePublished(self: *Self) void {
            for (0..slot_count) |j| {
                const m = self.meta[j].load(.monotonic);
                if (m & 3 == STATE_PUBLISHED) {
                    // Keep the seq (same as takeLatest): free slots are
                    // reclaimed with the same seq, publish stamps a fresh
                    // global one, so ordering stays monotonic.
                    _ = self.meta[j].cmpxchgStrong(m, m & ~@as(u64, 3), .release, .monotonic);
                }
            }
        }

        /// Copies the newest published payload into `out` and releases all
        /// published slots. Returns false when nothing new is available.
        pub fn takeLatest(self: *Self, out: *T) bool {
            var best: ?usize = null;
            var best_seq: u64 = 0;
            for (0..slot_count) |i| {
                const m = self.meta[i].load(.acquire);
                if (m & 3 != STATE_PUBLISHED) continue;
                const seq = m >> 2;
                if (best == null or seq > best_seq) {
                    best = i;
                    best_seq = seq;
                }
            }
            const i = best orelse return false;
            out.* = self.slots[i]; // copy while still published (safe)
            // Release every published slot: stale frames must not
            // resurface on a later take.
            for (0..slot_count) |j| {
                const m = self.meta[j].load(.monotonic);
                if (m & 3 == STATE_PUBLISHED) {
                    // Keep the seq: free slots are reclaimed with the same
                    // seq, publish stamps a fresh global one, so ordering
                    // stays monotonic.
                    _ = self.meta[j].cmpxchgStrong(m, m & ~@as(u64, 3), .release, .monotonic);
                }
            }
            return true;
        }
    };
}

// --- tests ---

const testing = std.testing;

const Frame = struct {
    seq: u64,
    checksum: u64, // seq * 0x9E3779B97F4A7C15 — validates an untorn copy
};

fn fill(frame: *Frame, seq: u64) void {
    frame.* = .{ .seq = seq, .checksum = seq *% 0x9E3779B97F4A7C15 };
}

test "Handoff delivers the latest and drops stale frames" {
    var h = Handoff(Frame, 2){};

    // Nothing published yet.
    var out: Frame = undefined;
    try testing.expect(!h.takeLatest(&out));

    // Publish seq 1, then seq 2; the stale seq 1 must never surface.
    for (1..3) |seq| {
        const i = h.claim().?;
        fill(h.slot(i), seq);
        h.publish(i);
    }
    try testing.expect(h.takeLatest(&out));
    try testing.expectEqual(@as(u64, 2), out.seq);
    // Both published slots released: nothing new afterwards.
    try testing.expect(!h.takeLatest(&out));
}

test "Handoff publisher drains stale slots so the newest wins" {
    var h = Handoff(Frame, 2){};
    for (1..3) |seq| {
        const i = h.claim().?;
        fill(h.slot(i), seq);
        h.publish(i);
    }
    // Saturated: claim fails. Drain stale published slots, then the newest
    // payload claims and publishes.
    try testing.expect(h.claim() == null);
    h.releasePublished();
    const i = h.claim().?;
    fill(h.slot(i), 3);
    h.publish(i);
    var out: Frame = undefined;
    try testing.expect(h.takeLatest(&out));
    try testing.expectEqual(@as(u64, 3), out.seq);
    try testing.expect(!h.takeLatest(&out));
}

test "Handoff publisher skips instead of blocking when slots run out" {
    var h = Handoff(Frame, 2){};
    const a = h.claim().?;
    fill(h.slot(a), 1);
    h.publish(a);
    const b = h.claim().?;
    fill(h.slot(b), 2);
    h.publish(b);
    // Both slots published (consumer slow): claim must fail, not block.
    try testing.expect(h.claim() == null);
    var out: Frame = undefined;
    try testing.expect(h.takeLatest(&out));
    try testing.expectEqual(@as(u64, 2), out.seq);
    // Released: publishing works again.
    const c = h.claim().?;
    fill(h.slot(c), 3);
    h.publish(c);
    try testing.expect(h.takeLatest(&out));
    try testing.expectEqual(@as(u64, 3), out.seq);
}

test "Handoff survives a publisher and consumer across threads" {
    var h = Handoff(Frame, 4){};
    var stop = std.atomic.Value(bool).init(false);

    const Pub = struct {
        fn run(hp: *Handoff(Frame, 4), stop_flag: *std.atomic.Value(bool)) void {
            var seq: u64 = 1;
            while (!stop_flag.load(.monotonic)) {
                const i = hp.claim() orelse {
                    std.atomic.spinLoopHint();
                    continue;
                };
                fill(hp.slot(i), seq);
                hp.publish(i);
                seq += 1;
            }
        }
    };
    const p = try std.Thread.spawn(.{}, Pub.run, .{ &h, &stop });

    // Consumer: every delivered frame must be internally consistent
    // (untorn) with a strictly increasing sequence; drops are fine
    // (latest-wins), resurfacing stale frames is not.
    var last_seq: u64 = 0;
    var out: Frame = undefined;
    var deliveries: usize = 0;
    while (deliveries < 5000) {
        if (h.takeLatest(&out)) {
            if (out.seq <= last_seq) return error.StaleFrameResurfaced;
            try testing.expectEqual(out.seq *% 0x9E3779B97F4A7C15, out.checksum);
            last_seq = out.seq;
            deliveries += 1;
        } else {
            std.atomic.spinLoopHint();
        }
    }
    stop.store(true, .release);
    p.join();

    // Drain whatever is left; invariants still hold.
    while (h.takeLatest(&out)) {
        if (out.seq <= last_seq) return error.StaleFrameResurfaced;
        try testing.expectEqual(out.seq *% 0x9E3779B97F4A7C15, out.checksum);
        last_seq = out.seq;
    }
}
