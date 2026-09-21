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
//!   mutex in `claimBack` / `claimSlot` / `tryPublish` / `cancelClaim` /
//!   `pin` / `pinFront` / `unpin` / `frontIndex` / `isPinned` / `pinsHeld`.
//!   The legacy trio (`backIndex` / `backSlot` / `publish`) plus direct
//!   `slots[i]` payload access are SINGLE-THREADED ONLY (the sequential
//!   legacy `stageUiPacket` path and the wave-26 rotation tests):
//!   `prepareFrame` itself resolves its working slot through the locked
//!   claim API since wave 31. The legacy trio reads `pinned`/`writing`
//!   without the mutex and must never run concurrently with lease activity.
//!   Concurrent producers/consumers must use the claim/pin API exclusively.
//! - Do not copy a FrameDraws (it owns a Mutex); Scene holds the single
//!   instance by value.
//!
//! Honest scope note — what this is NOT: the phase-mutex removal (true
//! concurrent update+prepare) is still NOT done and is not claimed. This
//! file makes the 3-slot rotation and the pin/lease primitives real and
//! tested, wave 27 made the frame snapshot slot-owned (`FrameDrawSlot.
//! snapshot`: prepare/render/reuse/UI-latch read the staged slot copy, never
//! the live `Scene.frame_snapshot` — that closes the snapshot-tearing
//! obstacle), and wave 28 closed the last prepare-latch live touches: the
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
//!   (`stageUiPacket` geometry + handle stamps), live particle/physics
//!   systems, and live camera/light/sky state (`packFrameSnapshot`), and
//!   writes live per-mesh previews/build_views — plus the commit itself
//!   reads the live mesh list (game-side, ordered after publish, never
//!   concurrent with the context). This is the app-side ordering problem:
//!   the game must finish mutating before building, under exclusion.
//! - the inline fallback paths (no fresh build): context-side `prepareFrame`
//!   stages instances and captures UI/particles/physics-debug straight from
//!   live state and writes live `instance_render` — unchanged legacy
//!   behavior for apps that never call `buildPreparedFrame`/`stageUiPacket`.
//! - the frame mailbox (`frame_handoff`) producer/consumer pair is still
//!   phase-excluded (a true concurrent producer would need the claim/pin API
//!   here instead of the sequential backIndex/publish path).
//! What IS already slot-owned (and therefore needs no lock once the mutex
//! goes): the frame snapshot, the staged instance records (+ their latched
//! outcomes), the UI packet lists + header + handles, the staged build stats
//! (`FrameDrawSlot.build_stats`, frozen by the game build and merged by the
//! prepare latch), the frozen particle capture (`particle_draws`) and the
//! frozen physics-debug capture (`physics_lines`/`physics_visible`) — both
//! frozen by the game build and consumed by the prepare latch instead of
//! the shared staging stores — and the prepare-gated GPU uploads (P3 epochs
//! + upload meter). Removing the phase mutex means moving the remaining
//! live reads above under the same freeze-then-latch shape (or an
//! equivalent mailbox) — the pin/lease here covers the variable-length
//! draw payload plus the staged snapshot, stats, and particle/physics
//! captures, deliberately nothing else.
//!
//! Wave 29 (concurrent-build ENGINE primitive — proof, not adoption): the
//! game side can now `claimBack` a slot, fill it (`Scene.BuildClaim.build`
//! runs the real build core into the claimed slot), and hand it to prepare
//! via `releaseHandoff` (no front flip — the flip stays context-owned in
//! `prepareFrame`) or drop it via `cancelClaim`. Payload reads/writes on
//! distinct slots need no locking (single producer; the lease mutex pairs
//! payload writes-before-release with reads-after-pin: the producer's slot
//! writes happen-before the mutex release in `tryPublish`/`releaseHandoff`,
//! the consumer's mutex acquire in `pin`/`pinFront`/`frontIndex`
//! synchronizes the subsequent payload reads). The wave-29 stress test
//! below proves slot-payload concurrency (claim/fill/publish vs pin/verify,
//! increasing published ids, canary consistency, skip-on-saturation, no
//! deadlock). The phase mutex between app update/build and prepare/render
//! is STILL HELD by the apps — this slice changes no app-facing flow
//! defaults and removes no mutex. App-side adoption checklist (NEXT wave,
//! not this one — wave 30 closed the seq-words item, the rest is app-side
//! flow):
//! - DONE (wave 30): the four handoff seq words (`build_seq`/
//!   `last_latched_seq`, `ui_packet_seq`/`last_latched_ui_seq`) plus
//!   `build_slot` are `std.atomic.Value` on `Scene` — release on
//!   publish/stage, acquire on latch/claim-consume, monotonic for the
//!   single-producer reserve and the context-side latch stamps (see the
//!   field docs in scene.zig). Sequential behavior is bit-identical and the
//!   wave-30 handoff-edge test proves the release/acquire pairing across
//!   threads. The mutex still guards the payload the words order; what
//!   REMAINS before it can go:
//! - stop calling bare `buildPreparedFrame`/`stageUiPacket` across threads:
//!   use `Scene.tryClaimBuildSlot` + `BuildClaim.build`/`stageUi` +
//!   `publish`/`cancel` on the game thread; never touch `backIndex`/
//!   `backSlot`/raw `slots[i]` writes concurrently (those stay
//!   single-threaded-only, as does legacy `stageUiPacket`, which must not
//!   run concurrently with prepare).
//! - DONE (wave 31): `prepareFrame`'s back resolution is a locked claim —
//!   `claimBack` on the fallback path, `claimSlot(build_slot)` on the latch
//!   path — held for the whole prepare and released at every exit
//!   (`tryPublish` on success, `cancelClaim`/early return on contention).
//!   A concurrent game claim therefore never targets the slot prepare is
//!   consuming (it skips the WRITING slot or saturates, counted), `pin`
//!   refuses that slot (`SlotBusy`, counted), and prepare never resets a
//!   game-held slot. Prepare's own publish flip stays context-owned exactly
//!   as before (now via the locked `tryPublish`); a missed slot degrades to
//!   a counted skip, never a wedge. What REMAINS for adoption is app-side
//!   flow (the first bullet) + canvas quiesce + freeze-then-latch:
//! - DONE (wave 32): particle/physics freeze-then-latch — the
//!   particle capture and the physics-debug capture are slot payloads
//!   (`FrameDrawSlot.particle_draws`, `physics_lines`/`physics_visible`,
//!   lock-free-publication slices 4/5): the game build freezes a plain
//!   copy into the claimed slot right after the shared build capture
//!   and the prepare latch consumes the SLOT copy, never the shared
//!   `build_frame`/`build_lines` staging stores — so a game-thread
//!   producer build colliding with the context-thread consumer latch
//!   can no longer deliver a torn record for one frame. The shared
//!   stores stay (direct layer tests + tooling compat) but the adopted
//!   path never reads them. Staged-wins on OOM (fail-closes to
//!   coherent-empty, same precedent as the snapshot/stats slices).
//!   What REMAINS for adoption is app-side flow (the first bullet) +
//!   canvas quiesce + the remaining live touches below:
//! - epochs stay context-owned (`begin`/`complete`/`flush` only in
//!   prepare/render): the build path must never gain epoch calls (tested).
//! - remaining live touches (meshes/canvas/cameras/lights, the
//!   `packFrameSnapshot` live reads, the inline fallback paths, the
//!   `frame_handoff` producer/consumer exclusion) move under
//!   freeze-then-latch before the mutex can go; `ui_canvas` mutation
//!   must additionally quiesce before prepare (the latch reads upload
//!   identity from the live canvas even on the staged path).

const std = @import("std");
const render_queue = @import("render_queue.zig");
const snapshot_mod = @import("snapshot.zig");
const stats_mod = @import("stats.zig");
const retire_mod = @import("gpu_retire.zig");
const ui_frame_mod = @import("ui_frame.zig");
const outline_pass = @import("../passes/outline_pass.zig");
const particle_pass = @import("../passes/particle_pass.zig");
const physics_types = @import("../physics/types.zig");
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
    /// instead of any live mesh reads — then by the game-side commit at the
    /// next build, which applies the latched outcomes to the live meshes.
    /// Appended in mesh-list
    /// order (strictly increasing `mesh_index`); reset retains capacity, so
    /// the latch/patch allocate nothing. Record `buffer` copies are borrowed
    /// read handles (never destroyed/retired through the record).
    staged_instances: std.ArrayListUnmanaged(StagedInstanceRecord) = .empty,
    /// Slot-owned UI CPU packet (lock-free-publication slice 2, b): the
    /// game side (`Scene.stageUiPacket`) records live canvas CPU geometry
    /// into these back-slot lists and stamps `ui_packet` (presence + staged
    /// draw handles); the prepare latch consumes them into `Scene.ui_frame`
    /// instead of reading the live canvas lists or handles. Plain CPU data
    /// (UIVertex/u16 + borrowed handle values — never destroyed/retired
    /// through the packet); reset
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
    /// Slot-owned staged build stats (wave 31 second slice,
    /// lock-free-publication slice 3): the game-side queue build
    /// accumulates into the live `Scene.build_stats` accumulator and
    /// `Scene.buildIntoClaimedSlot` freezes a plain copy here at build
    /// time; the prepare latch merges THIS copy into `Scene.stats`
    /// instead of reading the shared field — so a concurrent game-side
    /// accumulation cannot race the context-side merge. Plain struct
    /// copy (counter fields only — `mergeFrom` never touches the
    /// context-owned timing/upload fields); `reset` zeroes it so a
    /// reused slot can never resurface a prior frame's stats.
    build_stats: stats_mod.SceneStats = .{},
    /// Slot-owned frozen particle capture (wave 32, freeze-then-latch
    /// slice 4): the game-side build (`Scene.buildIntoClaimedSlot`,
    /// via `ParticleLayer.stageIntoSlot`) freezes the just-captured
    /// build frame here by value right after `buildCapture`; the
    /// prepare latch (`ParticleLayer.latchSlotFrame`) consumes THIS
    /// copy into the render-owned frame instead of the shared
    /// `build_frame` — so a game-thread build colliding with the
    /// context-side latch cannot deliver a torn record. Plain values
    /// only (borrowed handle ids, never destroyed/retired through the
    /// slot); `reset` clears the list (retaining capacity) so a reused
    /// slot can never resurface a prior frame's draws. Staged-wins on
    /// OOM (fail-closes to coherent-empty, same as the build frame).
    particle_draws: std.ArrayListUnmanaged(particle_pass.ParticlePass.ParticleDraw) = .empty,
    /// Slot-owned frozen physics debug capture (wave 32,
    /// freeze-then-latch slice 5): the game-side build (via
    /// `PhysicsIntegration.stageIntoSlot`) freezes the just-captured
    /// build lines + visibility here right after `buildDebug`; the
    /// prepare latch (`PhysicsIntegration.latchSlotDebug`) consumes
    /// THESE into the render-owned capture instead of the shared
    /// `build_lines`/`build_visible`. Same plain-value, staged-wins,
    /// coherent-empty-on-OOM contract as `particle_draws` above.
    physics_lines: std.ArrayListUnmanaged(physics_types.DebugLine) = .empty,
    /// Frozen visibility for the slot's physics debug capture (see
    /// `physics_lines`); `reset` clears it alongside the list.
    physics_visible: bool = false,
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
        self.build_stats = .{};
        self.particle_draws.clearRetainingCapacity();
        self.physics_lines.clearRetainingCapacity();
        self.physics_visible = false;
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
        self.particle_draws.deinit(allocator);
        self.physics_lines.deinit(allocator);
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
        b += listBytes(self.particle_draws);
        b += listBytes(self.physics_lines);
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
    /// legacy stage path and the rotation tests. First slot after `front`
    /// that is neither pinned nor claimed; asserts one exists (the
    /// sequential path never holds pins or claims across the call, so with
    /// 3 slots one is always free). `prepareFrame` no longer uses this
    /// (wave 31: locked claim); concurrent callers must use `claimBack`.
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
    /// Rotation-test helper since wave 31 (`prepareFrame` publishes via the
    /// locked `tryPublish` instead).
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

    /// Locked specific-slot claim (wave 31, prepare-latch side): reserve
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
        if (self.pinned[idx]) {
            self.publish_refusals += 1;
            return LeaseError.PinnedSlot;
        }
        if (self.writing[idx]) {
            self.saturation_skips += 1;
            return LeaseError.SlotBusy;
        }
        self.writing[idx] = true;
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

    /// Game-side handoff release (wave 29 concurrent-build primitive): hand
    /// a claimed slot to the prepare latch WITHOUT flipping `front` (the
    /// front flip stays context-owned in `prepareFrame`). Clears WRITING so
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
        if (self.pinned[back_idx]) {
            self.publish_refusals += 1;
            return LeaseError.PinnedSlot;
        }
        self.writing[back_idx] = false;
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
