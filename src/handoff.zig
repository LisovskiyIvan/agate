//! Latest-wins frame-state handoff (stage 3 groundwork, see REFACTOR.md).
//!
//! A publisher (the simulation side) fills one of N payload slots and
//! publishes it; a single consumer (the render side) takes the newest
//! complete payload and releases it. Older unclaimed payloads are dropped
//! — that is the point: a slow consumer never queues up stale frames, it
//! always renders the freshest complete state, and a fast publisher never
//! blocks (it just reuses free slots or skips).
//!
//! Single-producer, single-consumer for PAYLOAD writes/reads, but the slot
//! reclamation path is safe under a concurrent drain: the producer may call
//! `releasePublished` (drop stale published frames so the newest payload can
//! claim a slot) while the consumer is inside `takeLatest`. No external
//! exclusion is assumed between those two calls anymore.
//!
//! Lock-free: each slot carries `seq << 2 | state` in one atomic word
//! (free / writing / published). `takeLatest` copies out the highest-
//! sequence published slot and releases every published slot, so stale
//! frames never resurface.
//!
//! Copy-out semantics: `takeLatest(dest)` memcpys the payload — keep `T` a
//! plain data struct (no pointers into itself). The copy is seqlock-
//! validated: the slot word is loaded (acquire) before the copy and
//! re-loaded (acquire) after; on any change the scan restarts, so a payload
//! torn by a concurrent reclaim + overwrite is always discarded and never
//! delivered. A reclaimed slot can never present its old word again: every
//! publish stamps a fresh global sequence (monotonic fetchAdd), and both
//! intermediate transitions (published→free, free→writing) change the state
//! bits — so word equality across the copy proves no reclaim completed
//! inside it. A torn mid-copy read is therefore unobservable (the bytes are
//! dropped and re-copied), which is the standard seqlock trade: brief
//! logical retry instead of blocking the publisher.
//!
//! Stale-delivery guard: the consumer remembers the newest sequence it ever
//! delivered (`floor`). A slot whose sequence is not newer than the floor is
//! released, never delivered. This closes the cross-location visibility
//! race: with a single producer, publish stores land in sequence order, but
//! acquire loads of DIFFERENT slots carry no mutual ordering — a scan may
//! observe publish N+1 before publish N becomes visible. Without the floor,
//! a later scan could then deliver N after N+1 already went out. With the
//! floor, any such late-visible frame compares stale and is dropped, so
//! deliveries are strictly increasing across threads (drops are inherent to
//! latest-wins and are not errors).
//!
//! Memory-ordering contract, per operation:
//! - `claim`: filter load monotonic (state bits only); the free→writing CAS
//!   uses acquire on success. The acquire pairs with the consumer's release
//!   of that slot, ordering our payload overwrite after the consumer's
//!   copy-out read — a slot is never reused before its release is observed.
//! - `publish`: payload writes happen-before the release store of the
//!   published word; the consumer's acquire load of that word then
//!   synchronizes the copy-out read. The sequence fetchAdd itself is
//!   monotonic (uniqueness only — single producer, no ordering needed).
//! - `takeLatest`: scan loads are acquire (prompt, synchronized payload
//!   reads); the post-copy validation load is acquire for the same reason.
//!   The release-all loop uses monotonic loads (a missed observation only
//!   defers a release to the next call; the floor keeps delivery correct
//!   regardless) with release CAS on success.
//! - `releasePublished`: monotonic filter loads (a missed racing publish
//!   only makes the drain incomplete — the publisher then skips the frame,
//!   which is the documented saturation behavior, never corruption) with
//!   release CAS on success (pairs with the consumer's acquire scan and
//!   with a later claim's acquire CAS).
//!
//! What this does NOT do (explicit non-goal, no silent gap): slot COUNT
//! rotation stays at 2 for the frame mailboxes. A third slot would only
//! lower saturation frequency; it cannot remove the drain-vs-take race, and
//! with update-vs-prepare exclusion still holding there is no concurrent
//! consumer that could use a deeper rotation. A true 3-slot concurrent
//! rotation (producer builds slot C while the consumer draws slot A, slot B
//! in flight) additionally needs a consumer-side pin/lease protocol for the
//! variable-length payloads and removal of the phase mutex — that protocol
//! does not exist yet, so adding the slot now would be retention without a
//! reader. The mailbox layer is ready for it (slot_count is a comptime
//! parameter; the stress test below already exercises 3 slots).

const std = @import("std");

const SeqInt = usize;
const STATE_FREE: SeqInt = 0;
const STATE_WRITING: SeqInt = 1;
const STATE_PUBLISHED: SeqInt = 2;

pub fn Handoff(comptime T: type, comptime slot_count: usize) type {
    comptime std.debug.assert(slot_count >= 2); // double buffer minimum
    // NOTE: no power-of-two requirement — slot selection is a linear scan,
    // never an index mask, so any count >= 2 (including 3-slot rotations)
    // is correct by construction.
    return struct {
        const Self = @This();

        slots: [slot_count]T = undefined,
        /// Global publish counter: sequence numbers must be unique and
        /// monotonic across ALL slots, otherwise takeLatest cannot tell
        /// which published frame is newer.
        publish_seq: std.atomic.Value(SeqInt) = std.atomic.Value(SeqInt).init(0),
        meta: [slot_count]std.atomic.Value(SeqInt) = blk: {
            var m: [slot_count]std.atomic.Value(SeqInt) = undefined;
            for (&m) |*v| v.* = std.atomic.Value(SeqInt).init(STATE_FREE);
            break :blk m;
        },
        /// Newest sequence ever delivered by `takeLatest` (null = nothing
        /// delivered yet). Consumer-side only: written after a validated
        /// copy, read to suppress late-visible stale frames (see header).
        /// Plain field — only the single consumer touches it.
        floor: ?SeqInt = null,

        /// Reserves a free slot for writing. Returns null when every slot
        /// is published or being written (consumer lagging) — publishers
        /// skip the frame instead of blocking. The success CAS is acquire
        /// (see header): our overwrite orders after the consumer's release.
        pub fn claim(self: *Self) ?usize {
            for (0..slot_count) |i| {
                const m = self.meta[i].load(.monotonic);
                if (m & 3 != STATE_FREE) continue;
                if (self.meta[i].cmpxchgWeak(m, (m & ~@as(SeqInt, 3)) | STATE_WRITING, .acquire, .monotonic) == null) {
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
        /// counter, so "newest" is well defined across slots. Single
        /// producer only: with two publishers the counter assignment and
        /// the meta store could land out of order and "newest" would be
        /// ill-defined (the CAS in `claim` guards against accidental
        /// double-claiming, not against concurrent publishers).
        pub fn publish(self: *Self, i: usize) void {
            const seq = self.publish_seq.fetchAdd(1, .monotonic);
            self.meta[i].store(seq << 2 | STATE_PUBLISHED, .release);
        }

        /// Drops every published slot WITHOUT copying. Publisher-side drain
        /// for mailbox saturation: the publisher could not claim a free
        /// slot, so it discards the stale published frames and retries the
        /// claim to publish the newest payload. Safe under a concurrent
        /// `takeLatest`: reclamation is CAS-based (a slot the consumer
        /// already released fails the CAS, never double-frees), and a slot
        /// reclaimed mid-copy changes its word, so the consumer's seqlock
        /// validation discards the torn copy and rescans (see header). The
        /// drain may miss a racing publish (monotonic filter load) — then
        /// the retry claim still fails and the publisher skips the frame,
        /// which is the documented saturation behavior.
        pub fn releasePublished(self: *Self) void {
            for (0..slot_count) |j| {
                const m = self.meta[j].load(.monotonic);
                if (m & 3 == STATE_PUBLISHED) {
                    // Keep the seq (same as takeLatest): free slots are
                    // reclaimed with the same seq, publish stamps a fresh
                    // global one, so ordering stays monotonic.
                    _ = self.meta[j].cmpxchgStrong(m, m & ~@as(SeqInt, 3), .release, .monotonic);
                }
            }
        }

        /// Copies the newest published payload into `out` and releases all
        /// published slots. Returns false when nothing new is available.
        /// The copy is seqlock-validated (see header): a concurrent
        /// `releasePublished` + reclaim + overwrite changes the slot word,
        /// which restarts the scan instead of delivering torn bytes.
        /// Deliveries are strictly increasing: anything at or below the
        /// floor is released, never delivered.
        pub fn takeLatest(self: *Self, out: *T) bool {
            while (true) {
                var best: ?usize = null;
                var best_seq: SeqInt = 0;
                var best_word: SeqInt = 0;
                for (0..slot_count) |i| {
                    const m = self.meta[i].load(.acquire);
                    if (m & 3 != STATE_PUBLISHED) continue;
                    const seq = m >> 2;
                    if (best == null or seq > best_seq) {
                        best = i;
                        best_seq = seq;
                        best_word = m;
                    }
                }
                const i = best orelse return false;
                out.* = self.slots[i]; // copy while published; validated below
                // Validate BEFORE releasing: the slot must still carry the
                // exact word we copied under. Any reclaim cycle (drain CAS,
                // claim CAS, republish store) changes it — discard and
                // rescan for the newer frame instead of delivering bytes
                // that may have torn mid-copy.
                if (self.meta[i].load(.acquire) != best_word) continue;
                // Stale-visibility guard (see header): a frame that only
                // became visible after a newer one was already delivered is
                // dropped, never resurfaced.
                if (self.floor) |f| {
                    if (best_seq <= f) {
                        self.releaseScanned(f);
                        return false;
                    }
                }
                self.floor = best_seq;
                // Release every published slot at or below the delivered
                // sequence: stale frames must not resurface on a later take.
                // Slots published NEWER concurrently (seq > best_seq) are
                // left alone for the next take — releasing them here would
                // drop a frame that was never copied.
                self.releaseScanned(best_seq);
                return true;
            }
        }

        /// Releases every currently-published slot with `seq <= max_seq`,
        /// keeping sequences (free slots are reclaimed with the same seq;
        /// publish stamps a fresh global one, so ordering stays monotonic).
        /// Shared by `takeLatest` and the stale path above; NOT the
        /// publisher drain (that one is `releasePublished` — same mechanics,
        /// explicit publisher intent). The CAS is on the exact observed
        /// word, so a slot reclaimed + republished with a newer sequence in
        /// between fails the CAS and survives for the next take.
        fn releaseScanned(self: *Self, max_seq: SeqInt) void {
            for (0..slot_count) |j| {
                const m = self.meta[j].load(.monotonic);
                if (m & 3 != STATE_PUBLISHED) continue;
                if ((m >> 2) > max_seq) continue;
                _ = self.meta[j].cmpxchgStrong(m, m & ~@as(SeqInt, 3), .release, .monotonic);
            }
        }
    };
}
