//! Triple-buffered prepared draw payload: three retained owning queue slots
//! covering the prepared mesh draw lists — PRIMARY + ALL PIP view queues
//! (with their skin/shader side stores), outline items+skins, and prepared
//! shadow items+skins+bin ranges. Scope is mesh draws only (trail meshes
//! ride these same queues — Trail.update stages CPU-side and the prepare
//! flush uploads before the queue build bakes the values), plus the
//! game-side UI CPU packet staging (`ui_vertices`/`ui_indices` + `ui_packet`
//! header, CPU geometry only — the committed `UiFrame` itself stays
//! single-owned outside the slots); particles, physics-debug lines, and sky
//! carry their own prepared frames/payloads — never these slots.
//!
//! Ownership / lifecycle block:
//! - Owns only CPU-side queue storage (ArrayList buffers, skin/shader copies,
//!   sort-order entries). GPU handles inside items (buffers/views/samplers/
//!   pipelines) are BORROWED under the phase mutex / retire-epoch discipline —
//!   this buffer never destroys them, never duplicates GPU buffers, and never
//!   changes the retire algorithm. CPU slot retention is NOT a GPU
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
//!   prepare: a repeated begin discards the pending frame BEFORE
//!   GpuRetire.begin/flush (the staged begin clears its consumable flag
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
//! is not reading, allowing a presenting consumer to keep its frame while the
//! producer already builds the next one.
//!
//! Consumer pin/lease protocol (the prerequisite for a future true
//! concurrent update/prepare rotation):
//! - The render/present path PINs the slot it is presenting (`pin` /
//!   `pinFront`) and MUST unpin it when done (`unpin`). While pinned, a slot
//!   is never handed out as a build target (`claimBack` skips
//!   it) and can never become a publish target (
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
//!   mutex in `claimBack` / `claimSlot` / `claimLatestHandoff` /
//!   `cancelHandoffClaim` / `tryPublish` / `cancelClaim` / `pin` /
//!   `pinFront` / `pinFrontReader` / `unpin` / `unpinReader` / `frontIndex` /
//!   `isPinned` / `isReadPinned` / `pinsHeld`.
//!   Direct `slots[i]` payload access is SINGLE-THREADED ONLY: the holder
//!   must own the slot via the claim/pin protocol. Concurrent
//!   producers/consumers must use the claim/pin API exclusively.
//! - Do not copy a FrameDraws (it owns a Mutex); Scene holds the single
//!   instance by value.
//!
//! Scope note: true concurrent update+prepare (phase-mutex removal) is
//! still NOT done and is not claimed. This file provides the 3-slot
//! rotation and the pin/lease primitives. Slot-owned: the frame snapshot
//! (`FrameDrawSlot.snapshot`: prepare/render/reuse/UI-latch read the staged
//! slot copy, never the live `Scene.frame_snapshot`), and the remaining
//! prepare-latch live touches are closed: the
//! instance latch stages GPU purely from the slot-owned `staged_instances`
//! records + scratch (outcomes mirrored into the records for the patch, no
//! live mesh reads, no live mesh writes, not even a `record.mesh` compare)
//! with the `instance_render` write-back moved game-side as a commit of
//! published results at the next build (O(1) identity guard included), and
//! the UI latch consumes staged packet handles (pipeline/font/buffers,
//! frozen by `stageUiPacket`) instead of live-canvas reads. What still
//! needs the phase mutex:
//! - build-side live reads: `Scene.buildPreparedFrame` reads live meshes
//!   (TRS, materials, culling flags, `instance_render` as the prior state
//!   for the record freeze + `instance_build_view`), the live canvas
//!   (`BuildClaim.stageUi` geometry + handle stamps), live particle/physics
//!   systems, and live camera/light/sky state (`packFrameSnapshot`), and
//!   writes live per-mesh previews/build_views — plus the commit itself
//!   reads the live mesh list (game-side, ordered after publish, never
//!   concurrent with the context; the committed slot is resolved through
//!   the lease — no plain `front` word read on the game side). The game
//!   must finish mutating before building, under exclusion.
//! - the frame mailbox (`frame_handoff`) producer/consumer pair is still
//!   phase-excluded (a true concurrent producer would need the claim/pin API
//!   here instead of the sequential claim/publish path).
//! The pin/lease here covers the variable-length draw payload plus the
//! staged snapshot, stats, and particle/physics captures, deliberately
//! nothing else. Removing the phase mutex means moving the remaining
//! live reads above under the same freeze-then-latch shape (or an
//! equivalent mailbox).
//!
//! Concurrent-build primitive: the game side can `claimBack` a slot, fill it (`Scene.BuildClaim.build`
//! runs the real build core into the claimed slot), and hand it to prepare
//! via `releaseHandoff` (no front flip — the flip stays context-owned in
//! the staged finish via `tryPublish`) or drop it via `cancelClaim`.
//! Payload reads/writes on distinct slots need no locking (single
//! producer; the lease mutex pairs payload writes-before-release with
//! reads-after-pin: the producer's slot writes happen-before the mutex
//! release in `tryPublish`/`releaseHandoff`, the consumer's mutex acquire
//! in `pin`/`pinFront`/`frontIndex` synchronizes the subsequent payload
//! reads).
//! - The handoff words (`build_seq`/`last_latched_seq`) plus
//!   `build_slot` are `std.atomic.Value` on `Scene` — release on
//!   publish, acquire on latch/claim-consume (see the field docs in
//!   scene.zig). The mutex still guards the payload the words order.
//! - use `Scene.tryClaimBuildSlot` + `BuildClaim.build`/`stageUi` +
//!   `publish`/`cancel` on the game thread; never touch raw `slots[i]`
//!   writes concurrently.
//! - Staged prepare resolves its slot through the locked
//!   `claimLatestHandoff` claim — held for the whole prepare and released
//!   at every exit (`tryPublish` on success, `cancelHandoffClaim` on
//!   failure/cancel).
//!   A concurrent game claim therefore never targets the slot prepare is
//!   consuming (it skips the WRITING slot or saturates, counted), `pin`
//!   refuses that slot (`SlotBusy`, counted), and prepare never resets a
//!   game-held slot. Prepare's own publish flip stays context-owned
//!   (via the locked `tryPublish`); a missed slot degrades to
//!   a counted skip, never a wedge.
//! - Particle/physics freeze-then-latch: the
//!   particle capture and the physics-debug capture are slot payloads
//!   (`FrameDrawSlot.particle_draws`, `physics_lines`/`physics_visible`):
//!   the game build freezes a plain copy into the claimed slot right after the shared build capture
//!   and the prepare latch consumes the SLOT copy, never the shared
//!   `build_frame`/`build_lines` staging stores — so a game-thread
//!   producer build colliding with the context-thread consumer latch
//!   can no longer deliver a torn record for one frame. The shared
//!   stores stay (direct layer tests + tooling compat) but the adopted
//!   path never reads them. Staged-wins on OOM (fail-closes to
//!   coherent-empty).
//! - Epochs stay context-owned (`begin`/`complete`/`flush` only in
//!   prepare/render): the build path must never gain epoch calls (tested).
//! - remaining live touches (meshes/canvas/cameras/lights, the
//!   `packFrameSnapshot` live reads, the
//!   `frame_handoff` producer/consumer exclusion) move under
//!   freeze-then-latch before the mutex can go; `ui_canvas` mutation
//!   must additionally quiesce before prepare (the latch reads upload
//!   identity from the live canvas even on the staged path).
//! Layout: `frame_draws_packets.zig` (frozen per-owner packet descriptors),
//! `frame_draws_slot.zig` (retained slot + lifecycle), `frame_draws_lease.zig`
//! (rotation + pin/lease protocol). This file is a facade re-exporting the
//! public API so callers are unchanged.

const packets_mod = @import("frame_draws_packets.zig");
const slot_mod = @import("frame_draws_slot.zig");
const lease_mod = @import("frame_draws_lease.zig");

pub const SLOT_COUNT = lease_mod.SLOT_COUNT;
pub const LeaseError = lease_mod.LeaseError;
pub const FrameDrawSlot = slot_mod.FrameDrawSlot;
pub const FrameDraws = lease_mod.FrameDraws;

// Lease-protocol regression tests live in `frame_draws_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).
