const std = @import("std");
const testing = std.testing;
const Handoff = @import("handoff.zig").Handoff;

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

// Stress: 3-slot rotation with a RACING publisher drain. The producer
// drains stale published slots when saturated (the real `releasePublished`
// use from updateLights/publishFrameSnapshot) while the consumer runs
// `takeLatest` concurrently with NO exclusion — the exact race the old
// "assumed exclusion" contract forbade. Every delivery must still be untorn
// (seqlock validation) and strictly increasing (floor); saturation must
// resolve to fresh claims, never to a hang.
test "Handoff 3-slot claim/publish/drain races takeLatest across threads" {
    var h = Handoff(Frame, 3){};
    var stop = std.atomic.Value(bool).init(false);
    // Publisher-side counters (single producer thread writes; the test
    // thread reads after join — release/acquire pairing below).
    var published_total: u64 = 0;
    var drain_count: u64 = 0;

    const Ctx = struct {
        hp: *Handoff(Frame, 3),
        stop_flag: *std.atomic.Value(bool),
        published: *u64,
        drains: *u64,
    };
    const Pub = struct {
        fn run(c: Ctx) void {
            var seq: u64 = 1;
            // Bounded run: the assertion below needs a finite producer.
            while (seq <= 20000) {
                const i = c.hp.claim() orelse {
                    // Saturated: drain stale published frames (the
                    // publisher-side saturation path) and retry — exactly
                    // what the apps do under consumer lag.
                    c.hp.releasePublished();
                    c.drains.* += 1;
                    continue;
                };
                fill(c.hp.slot(i), seq);
                c.hp.publish(i);
                seq += 1;
            }
            c.published.* = seq - 1;
            c.stop_flag.store(true, .release);
        }
    };
    const ctx = Ctx{ .hp = &h, .stop_flag = &stop, .published = &published_total, .drains = &drain_count };
    const p = try std.Thread.spawn(.{}, Pub.run, .{ctx});

    // Consumer on the test thread: strictly increasing untorn deliveries.
    // A racing drain may reclaim a slot mid-copy — the seqlock retry must
    // hide it (no torn checksum may ever surface). The consumer is
    // deliberately slowed (a few thousand spin hints per take) so the tight
    // producer deterministically laps it and the drain path above is really
    // exercised — without the slowdown a fast consumer could keep 3 slots
    // drained and `drain_count` would stay zero.
    var last_seq: u64 = 0;
    var out: Frame = undefined;
    var deliveries: usize = 0;
    while (!stop.load(.acquire)) {
        if (h.takeLatest(&out)) {
            if (out.seq <= last_seq) return error.StaleFrameResurfaced;
            try testing.expectEqual(out.seq *% 0x9E3779B97F4A7C15, out.checksum);
            last_seq = out.seq;
            deliveries += 1;
            var spin: usize = 0;
            while (spin < 3000) : (spin += 1) std.atomic.spinLoopHint();
        } else {
            std.atomic.spinLoopHint();
        }
    }
    p.join();

    // Progress: the consumer must have observed frames (a livelocked
    // seqlock retry would hang above instead of failing here).
    try testing.expect(deliveries > 0);
    // Drain the tail; invariants hold to the end.
    while (h.takeLatest(&out)) {
        if (out.seq <= last_seq) return error.StaleFrameResurfaced;
        try testing.expectEqual(out.seq *% 0x9E3779B97F4A7C15, out.checksum);
        last_seq = out.seq;
        deliveries += 1;
    }
    // Nothing newer than what the producer published may ever surface, and
    // the last delivery cannot exceed the published total.
    try testing.expect(last_seq <= published_total);
    try testing.expectEqual(@as(u64, 20000), published_total);
    // Saturation really happened (3 slots churn under a spinning consumer):
    // the drain path above was exercised, not skipped.
    try testing.expect(drain_count > 0);
}
