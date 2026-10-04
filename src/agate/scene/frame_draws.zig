//! Triple-buffered prepared draw payload: three retained owning queue slots
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
//! is not reading: one slot deeper than the old 2-slot flip, which is what
//! lets a presenting consumer keep its frame while the producer already
//! builds the next one.
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

const std = @import("std");
const render_queue = @import("render_queue.zig");
const snapshot_mod = @import("snapshot.zig");
const stats_mod = @import("stats.zig");
const retire_mod = @import("gpu_retire.zig");
const ui_frame_mod = @import("ui_frame.zig");
const outline_pass = @import("../passes/outline_pass.zig");
const highlight_pass = @import("../passes/highlight_pass.zig");
const particle_pass = @import("../passes/particle_pass.zig");
const particle_types = @import("../particles/types.zig");
const physics_types = @import("../physics/types.zig");
const shadow_pass = @import("../passes/shadow_pass.zig");
const mesh_mod = @import("../mesh.zig");
const ui_mod = @import("../ui.zig");

pub const RenderQueues = render_queue.RenderQueues;
pub const SkinStorage = render_queue.SkinStorage;
pub const OutlineDrawItem = outline_pass.OutlineDrawItem;
pub const HighlightDrawItem = highlight_pass.HighlightDrawItem;
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

pub const HandoffClaim = struct {
    slot: usize,
    seq: usize,
    has_scene_build: bool,
};

/// One coherent prepared frame: the prepared mesh draw lists (view queues,
// outline, shadow) the render phase consumes. Built whole into a BACK
// slot, then published by index flip; render touches it only through const
// references while it is the consumable front (see header), or under an
// explicit pin (which extends consumability past the next publish for
// exactly that slot).
//
// Payload identity: every instanced batch / shadow item /
// outline item carries `source_uid` (stable `Mesh.uid`) + `source_mesh`
// (mesh-list index at build time). Game-built (`.build_view`) payloads hold
// provisional `instance_buffer`/`visible_instance_count` (plus shadow
// `world_aabb`/`max_dim`, outline `world_center`) until the latch
// `patchInstanceRefs` finalizes them from the slot-owned `staged_instances`
// records (fail-closed zero on record-missing/uid mismatch or stale publish);
// fallback (`.published`) payloads are final at build time.
pub const StagedInstanceRecord = mesh_mod.StagedInstanceRecord;

/// Slot-owned dynamic-upload packets (producer freeze-then-latch):
/// every per-frame GPU staging payload the prepare flush uploads is frozen
/// here by value on the producer side (`buildIntoClaimedSlot` via
/// `upload_packets.stageUploads`) and consumed by the staged prepare
/// (`upload_packets.flushSlotUploads`) instead of any live mutable array.
/// All descriptors are plain integers (tokens/ids/counts/offsets); the byte
/// payloads live in the sibling flattened lists below. Buffer ids are
/// borrowed values under the P3 epoch discipline (never destroyed/retired
/// through the packet). `reset` clears lengths retaining capacity; `deinit`
/// frees; `cpuBytes` counts retained capacities.
pub const MorphUpload = struct {
    token: usize = 0,
    uid: u64 = 0,
    mesh_index: u32 = 0,
    buffer_id: u32 = 0,
    count: u32 = 0,
    data_lo: usize = 0,
    /// Lock-free publication outcome (phase 2, context-written): the staged
    /// flush sets true once the frozen bytes landed in `buffer_id` (or the
    /// packet was empty). The producer cleared `morph_upload_needed` at
    /// stage time; a false outcome means the game-side commit must re-arm
    /// it for retry. Never written by the producer after staging.
    delivered: bool = false,
};
pub const ParticleCpuUpload = struct {
    token: usize = 0,
    sys_index: u32 = 0,
    buffer_id: u32 = 0,
    count: u32 = 0,
    data_lo: usize = 0,
    /// Frozen allocation capacity (immutable after system init): deferred
    /// creation sizes the buffer from this, never the live field.
    capacity: usize = 0,
    /// Deferred-creation outcome (context-written): the created buffer, or
    /// zero when no creation was needed/possible. Installed by the
    /// game-side commit (which retires a replaced non-zero handle, though
    /// in practice the live id is always zero here — only the context
    /// creates, and both paths run on the same thread).
    created_buffer_id: u32 = 0,
    /// Lock-free publication outcome (context-written, see MorphUpload).
    delivered: bool = false,
};
pub const ParticleGpuUpload = struct {
    token: usize = 0,
    sys_index: u32 = 0,
    buffer_id: u32 = 0,
    count: u32 = 0,
    data_lo: usize = 0,
    /// Frozen allocation capacity (see ParticleCpuUpload).
    capacity: usize = 0,
    /// Deferred-creation outcome (context-written, see ParticleCpuUpload).
    created_buffer_id: u32 = 0,
    /// Lock-free publication outcome (context-written, see MorphUpload).
    delivered: bool = false,
};
pub const ParticleComputeUpload = struct {
    token: usize = 0,
    sys_index: u32 = 0,
    spawn_buffer_id: u32 = 0,
    staged: usize = 0,
    stage_base: usize = 0,
    cursor: usize = 0,
    high_water: usize = 0,
    dt_accum: f32 = 0.0,
    flush_pending: bool = false,
    state_clear_pending: bool = false,
    buffers_pending: bool = false,
    gravity: [3]f32 = .{ 0, 0, 0 },
    drag: f32 = 0.0,
    sheet_cols: u32 = 1,
    sheet_rows: u32 = 1,
    sheet_loops: f32 = 1.0,
    data_lo: usize = 0,
    data_count: usize = 0,
    /// Frozen allocation capacity (immutable after init): creation and the
    /// dispatch addr field use this, never the live field.
    capacity: usize = 0,
    /// Frozen GPU object ids (staged alongside the bytes): the direct
    /// staged upload/dispatch addresses these without touching live
    /// handles. Zero means "not yet created".
    state_buffer_id: u32 = 0,
    draw_buffer_id: u32 = 0,
    state_view_id: u32 = 0,
    spawn_view_id: u32 = 0,
    draw_view_id: u32 = 0,
    shader_id: u32 = 0,
    pipeline_id: u32 = 0,
    /// Deferred-creation outcomes (context-written): created GPU objects,
    /// zero when no creation was needed. Installed by the game-side
    /// commit (replaced non-zero live handles retire through the queue —
    /// defensive only: in practice the live ids are zero whenever these
    /// are set, because only the context creates).
    created_state_buffer_id: u32 = 0,
    created_spawn_buffer_id: u32 = 0,
    created_draw_buffer_id: u32 = 0,
    created_state_view_id: u32 = 0,
    created_spawn_view_id: u32 = 0,
    created_draw_view_id: u32 = 0,
    created_shader_id: u32 = 0,
    created_pipeline_id: u32 = 0,
    /// Consumed window (context-written): how many staged records and how
    /// much dt the dispatch consumed. The game-side commit advances the
    /// live ring by exactly this (guarded by a stage_base match, so a
    /// post-freeze eviction skips the advance instead of corrupting it)
    /// and subtracts the dt clamped at zero.
    consumed_staged: usize = 0,
    consumed_dt: f32 = 0.0,
    /// Backend without compute support latched context-side
    /// (context-written): the game-side commit publishes it into
    /// `compute_known_unsupported` and drops the pending flags, mirroring
    /// the legacy helper — never a silent fallback, never a retry spin.
    unsupported: bool = false,
    /// Actual dispatches issued by the staged flush for this packet
    /// (context-written, `+= 1` after each real `sg.dispatch`): the
    /// game-side commit transfers the count into the owner's
    /// `compute_dispatches` exactly once (then zeroes it here), so the
    /// real-GPU gate observes every dispatch even when the packet's
    /// upload later fails and retries. An attempt counter, not a delivery
    /// outcome: `resetOutcomes` deliberately leaves it (a cancelled
    /// prepare re-flushes the same slot and must not lose the earlier
    /// attempt). Headless flushes never dispatch, so this stays 0 there.
    dispatches: u64 = 0,
    /// Lock-free publication outcome (context-written, see MorphUpload).
    delivered: bool = false,
};
pub const TrailUpload = struct {
    token: usize = 0,
    trail_index: u32 = 0,
    vertex_buffer_id: u32 = 0,
    index_buffer_id: u32 = 0,
    vert_count: usize = 0,
    index_count: usize = 0,
    vert_lo: usize = 0,
    index_lo: usize = 0,
    min_pt: [3]f32 = .{ 0, 0, 0 },
    max_pt: [3]f32 = .{ 0, 0, 0 },
    buffers_pending: bool = false,
    /// Frozen allocation lengths (fixed at trail init): deferred creation
    /// sizes the buffers from these, never the live slices.
    vert_cap: usize = 0,
    index_cap: usize = 0,
    /// Deferred-creation outcomes (context-written, see ParticleCpuUpload).
    created_vertex_buffer_id: u32 = 0,
    created_index_buffer_id: u32 = 0,
    /// Lock-free publication outcome (context-written, see MorphUpload).
    /// The frozen index_count/bounds are published by the game-side
    /// commit; the render path reads only the baked queue payload.
    delivered: bool = false,
};
pub const SoftUpload = struct {
    token: usize = 0,
    body_index: u32 = 0,
    vertex_buffer_id: u32 = 0,
    vert_count: usize = 0,
    data_lo: usize = 0,
    index_lo: usize = 0,
    index_count: usize = 0,
    min_pt: [3]f32 = .{ 0, 0, 0 },
    max_pt: [3]f32 = .{ 0, 0, 0 },
    buffers_pending: bool = false,
    /// Frozen vertex allocation length (fixed grid at body creation):
    /// deferred creation sizes the vertex buffer from this.
    vert_cap: usize = 0,
    /// Deferred-creation outcomes (context-written, see ParticleCpuUpload).
    created_vertex_buffer_id: u32 = 0,
    created_index_buffer_id: u32 = 0,
    /// Lock-free publication outcome (context-written, see MorphUpload).
    delivered: bool = false,
};
pub const GreasedUpload = struct {
    token: usize = 0,
    line_index: u32 = 0,
    vertex_buffer_id: u32 = 0,
    index_buffer_id: u32 = 0,
    vert_count: usize = 0,
    index_count: usize = 0,
    vert_lo: usize = 0,
    index_lo: usize = 0,
    full_upload: bool = false,
    /// Frozen allocation lengths: deferred creation sizes the buffers
    /// from these, never the live slices.
    vert_cap: usize = 0,
    index_cap: usize = 0,
    /// Deferred-creation outcomes (context-written, see ParticleCpuUpload).
    created_vertex_buffer_id: u32 = 0,
    created_index_buffer_id: u32 = 0,
    /// Full-index-upload delivery (context-written, same protocol as
    /// TrailUpload.full_delivered).
    full_delivered: bool = false,
    /// Lock-free publication outcome (context-written, see MorphUpload).
    delivered: bool = false,
};
pub const PendingMeshUpload = struct {
    token: usize = 0,
    uid: u64 = 0,
    mesh_index: u32 = 0,
    vert_count: usize = 0,
    index_count: usize = 0,
    vert_lo: usize = 0,
    index_lo: usize = 0,
    index_type_is_u16: bool = true,
    dynamic_update: bool = false,
    /// Frozen GPU-morph delta request (write-once, GPU-mode only): the mesh
    /// carried `morph_upload_pending` at stage time, so the producer packed
    /// its RGBA32F delta pixels into `pending_delta_data` below (frozen
    /// dims + f32 range). The context creates the delta image + view from
    /// those frozen bytes alongside the vertex/index buffers, and the
    /// game-side commit installs everything atomically — the per-frame CPU
    /// blend stays gone on the GPU path with no base-pose frame. False for
    /// every non-morph and CPU-morph mesh (no bytes frozen, legacy paths
    /// untouched).
    morph_delta_pending: bool = false,
    delta_lo: usize = 0,
    delta_count: usize = 0,
    delta_width: u32 = 0,
    delta_height: u32 = 0,
    /// Deferred-creation outcomes for the delta texture (context-written,
    /// see ParticleCpuUpload): installed by the game-side commit over zero
    /// live handles together with the buffers above.
    created_delta_image_id: u32 = 0,
    created_delta_view_id: u32 = 0,
    /// Deferred-creation outcomes (context-written, see ParticleCpuUpload).
    created_vertex_buffer_id: u32 = 0,
    created_index_buffer_id: u32 = 0,
    /// Lock-free publication outcome (context-written, see MorphUpload).
    /// The game-side commit installs the handles, publishes
    /// `vertex_count` (when live is zero), re-arms `morph_upload_needed`
    /// for dynamic updates, and frees the consumed live pending arrays.
    delivered: bool = false,
};

pub const FrameDrawSlot = struct {
    primary: RenderQueues = .{},
    views: [snapshot_mod.MAX_CAMERAS]RenderQueues = [_]RenderQueues{.{}} ** snapshot_mod.MAX_CAMERAS,
    outline_items: std.ArrayListUnmanaged(OutlineDrawItem) = .empty,
    outline_skins: SkinStorage = .empty,
    /// Staged per-mesh highlight items (highlight layer v1): render-owned
    /// snapshots (model, handles, frozen options) built by
    /// `buildQueuesInto` from `Scene.highlights`, consumed by the PASS 2.85
    /// mask/blur stage in `PostFXStack.renderChain`. No skin store: v1
    /// stages no skin matrices (skinned meshes are skipped at capture).
    /// Borrowed GPU handles under the P3 epoch discipline, same as
    /// outline_items above.
    highlight_items: std.ArrayListUnmanaged(HighlightDrawItem) = .empty,
    shadow: PreparedShadowDraws = .{},
    /// Slot-owned staged instance records:
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
    /// Slot-owned UI CPU packet: the
    /// game side (`BuildClaim.stageUi`) records live canvas CPU geometry
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
    /// Slot-owned staged frame snapshot:
    /// the frame-level camera/light/pass state the prepared payload was
    /// built against, frozen by value at build time. `Scene.buildPreparedFrame`
    /// (game side) stages `build_snapshot` here; the fallback prepare path
    /// stages `frame_snapshot` here after the takeLatest-else-pack latch.
    /// `beginStagedPrepare`/`render`/`renderReuse`/the UI latch all read THIS copy
    /// (prepare reads the slot it is consuming, render/reuse read the front
    /// slot's copy) — never the live `Scene.frame_snapshot`, so a concurrent
    /// game-side mutation cannot tear the in-flight frame.
    ///
    /// Plain struct copy (`slot.snapshot = snap`), never pointers into game
    /// state: `Camera.name` slices alias the live camera names, but the draw
    /// path never dereferences names (projection matrices are precomputed in
    /// the snapshot); GPU handles (`sky_texture`, default copies, probe
    /// views/samplers) are borrowed VALUES under the retire-epoch discipline
    /// (same as every other handle in this slot — never destroyed/retired
    /// through the slot). Fixed-size (no allocation, no deinit); `reset`
    /// clears it so a skipped path can never resurface a prior frame, and
    /// `cpuBytes` deliberately excludes it (fixed scalar, like
    /// `frame_id`/`retire_epoch` — the census counts retained list capacity).
    snapshot: snapshot_mod.SceneFrameSnapshot = .{},
    /// Slot-owned build counters. The producer writes directly into this
    /// claimed slot while building; prepare merges the same immutable copy
    /// after the publish edge. No shared producer-side accumulator crosses
    /// the handoff. `reset` zeroes this field so a reused slot cannot
    /// resurface stale counters.
    build_stats: stats_mod.SceneStats = .{},
    /// Producer generation that built this slot. Prepare latches this exact
    /// generation even if a newer build publishes while the slot-only latch
    /// is finishing.
    build_seq: usize = 0,
    /// True only when the producer built the 3D/frame payload, not a
    /// UI-packet-only claim. Staged-only prepare must not consume stale 3D
    /// data as if it belonged to a UI-only handoff.
    has_scene_build: bool = false,
    /// Slot-owned frozen particle capture: the game-side build
    /// (`Scene.buildIntoClaimedSlot`,
    /// via `ParticleLayer.stageIntoSlot`) freezes the just-captured
    /// build frame here by value right after `buildCapture`; the
    /// prepare latch (`ParticleLayer.latchSlotFrame`) consumes THIS
    /// copy into the render-owned frame instead of the shared
    /// `build_frame` — so a game-thread build colliding with the
    /// context-side latch cannot deliver a torn record. Plain values
    /// only (borrowed handle ids, never destroyed/retired through the
    /// slot); `reset` clears the list (retaining capacity) so a reused
    /// slot can never resurface a prior frame's draws. Staged-wins on
    /// OOM (fail-closes to coherent-empty).
    particle_draws: std.ArrayListUnmanaged(particle_pass.ParticlePass.ParticleDraw) = .empty,
    /// Slot-owned frozen physics debug capture: the game-side build (via
    /// `PhysicsIntegration.stageIntoSlot`) freezes the just-captured
    /// build lines + visibility here right after `buildDebug`; the
    /// prepare latch (`PhysicsIntegration.latchSlotDebug`) consumes
    /// THESE into the render-owned capture instead of the shared
    /// `build_lines`/`build_visible`. Same plain-value, staged-wins,
    /// coherent-empty-on-OOM contract as `particle_draws`.
    physics_lines: std.ArrayListUnmanaged(physics_types.DebugLine) = .empty,
    /// Frozen visibility for the slot's physics debug capture (see
    /// `physics_lines`); `reset` clears it alongside the list.
    physics_visible: bool = false,
    /// Slot-owned dynamic-upload packets (see the packet structs
    /// above): frozen producer-side by `upload_packets.stageUploads`,
    /// consumed context-side by `upload_packets.flushSlotUploads` on the
    /// fresh-build path. Descriptors + flattened byte stores; reset retains
    /// capacity, deinit frees, cpuBytes counts capacities.
    morph_uploads: std.ArrayListUnmanaged(MorphUpload) = .empty,
    morph_data: std.ArrayListUnmanaged(mesh_mod.Vertex) = .empty,
    p_cpu_uploads: std.ArrayListUnmanaged(ParticleCpuUpload) = .empty,
    p_cpu_data: std.ArrayListUnmanaged(particle_types.ParticleInstanceData) = .empty,
    p_gpu_uploads: std.ArrayListUnmanaged(ParticleGpuUpload) = .empty,
    p_gpu_data: std.ArrayListUnmanaged(particle_types.GpuParticleSlot) = .empty,
    p_compute_uploads: std.ArrayListUnmanaged(ParticleComputeUpload) = .empty,
    p_compute_data: std.ArrayListUnmanaged(particle_types.GpuParticleSlot) = .empty,
    trail_uploads: std.ArrayListUnmanaged(TrailUpload) = .empty,
    trail_verts: std.ArrayListUnmanaged(mesh_mod.Vertex) = .empty,
    trail_indices: std.ArrayListUnmanaged(u16) = .empty,
    soft_uploads: std.ArrayListUnmanaged(SoftUpload) = .empty,
    soft_data: std.ArrayListUnmanaged(mesh_mod.Vertex) = .empty,
    soft_indices: std.ArrayListUnmanaged(u32) = .empty,
    greased_uploads: std.ArrayListUnmanaged(GreasedUpload) = .empty,
    greased_verts: std.ArrayListUnmanaged(mesh_mod.Vertex) = .empty,
    greased_indices: std.ArrayListUnmanaged(u32) = .empty,
    pending_uploads: std.ArrayListUnmanaged(PendingMeshUpload) = .empty,
    pending_verts: std.ArrayListUnmanaged(mesh_mod.Vertex) = .empty,
    pending_indices: std.ArrayListUnmanaged(u32) = .empty,
    /// Frozen GPU-morph delta pixels (PendingMeshUpload delta
    /// fields): packed RGBA32F texels (4 f32 per texel, see
    /// mesh/morph_gpu.zig) for pending GPU-morph meshes only. Same
    /// freeze-then-latch contract as every other byte store here.
    pending_delta_data: std.ArrayListUnmanaged(f32) = .empty,
    /// Slot-owned frozen host bytes: the
    /// producer (`BuildClaim.stageHostBytes`) copies small host-owned
    /// payloads here (picked-name bytes, memory-summary tallies) while it
    /// holds the claim; the context (`PrepareClaim.host_bytes`) reads the
    /// frozen copy instead of live host state. Plain bytes — the app owns
    /// the encoding. `reset` clears (retaining capacity) so a reused slot
    /// can never resurface a prior frame; `deinit` frees; `cpuBytes`
    /// counts the retained capacity.
    host_bytes: std.ArrayListUnmanaged(u8) = .empty,
    /// Scene.frame_id that built this slot.
    frame_id: u64 = 0,
    /// GpuRetire epoch opened by the staged begin that built this slot.
    retire_epoch: Epoch = 0,

    /// Clear lengths for reuse, retaining all capacity. Covers EVERY list —
    /// including disabled views and disabled shadow bins — so a skipped path
    /// can never resurface another slot's prior frame.
    pub fn reset(self: *FrameDrawSlot) void {
        self.primary.reset();
        for (&self.views) |*q| q.reset();
        self.outline_items.clearRetainingCapacity();
        self.outline_skins.clearRetainingCapacity();
        self.highlight_items.clearRetainingCapacity();
        self.shadow.reset();
        self.staged_instances.clearRetainingCapacity();
        self.ui_vertices.clearRetainingCapacity();
        self.ui_indices.clearRetainingCapacity();
        self.ui_packet = .{};
        self.snapshot = .{};
        self.build_stats = .{};
        self.build_seq = 0;
        self.has_scene_build = false;
        self.particle_draws.clearRetainingCapacity();
        self.physics_lines.clearRetainingCapacity();
        self.physics_visible = false;
        self.morph_uploads.clearRetainingCapacity();
        self.morph_data.clearRetainingCapacity();
        self.p_cpu_uploads.clearRetainingCapacity();
        self.p_cpu_data.clearRetainingCapacity();
        self.p_gpu_uploads.clearRetainingCapacity();
        self.p_gpu_data.clearRetainingCapacity();
        self.p_compute_uploads.clearRetainingCapacity();
        self.p_compute_data.clearRetainingCapacity();
        self.trail_uploads.clearRetainingCapacity();
        self.trail_verts.clearRetainingCapacity();
        self.trail_indices.clearRetainingCapacity();
        self.soft_uploads.clearRetainingCapacity();
        self.soft_data.clearRetainingCapacity();
        self.soft_indices.clearRetainingCapacity();
        self.greased_uploads.clearRetainingCapacity();
        self.greased_verts.clearRetainingCapacity();
        self.greased_indices.clearRetainingCapacity();
        self.pending_uploads.clearRetainingCapacity();
        self.pending_verts.clearRetainingCapacity();
        self.pending_indices.clearRetainingCapacity();
        self.pending_delta_data.clearRetainingCapacity();
        self.host_bytes.clearRetainingCapacity();
        self.frame_id = 0;
        self.retire_epoch = 0;
    }

    pub fn deinit(self: *FrameDrawSlot, allocator: std.mem.Allocator) void {
        self.primary.deinit(allocator);
        for (&self.views) |*q| q.deinit(allocator);
        self.outline_items.deinit(allocator);
        self.outline_skins.deinit(allocator);
        self.highlight_items.deinit(allocator);
        self.shadow.deinit(allocator);
        self.staged_instances.deinit(allocator);
        self.ui_vertices.deinit(allocator);
        self.ui_indices.deinit(allocator);
        self.particle_draws.deinit(allocator);
        self.physics_lines.deinit(allocator);
        self.morph_uploads.deinit(allocator);
        self.morph_data.deinit(allocator);
        self.p_cpu_uploads.deinit(allocator);
        self.p_cpu_data.deinit(allocator);
        self.p_gpu_uploads.deinit(allocator);
        self.p_gpu_data.deinit(allocator);
        self.p_compute_uploads.deinit(allocator);
        self.p_compute_data.deinit(allocator);
        self.trail_uploads.deinit(allocator);
        self.trail_verts.deinit(allocator);
        self.trail_indices.deinit(allocator);
        self.soft_uploads.deinit(allocator);
        self.soft_data.deinit(allocator);
        self.soft_indices.deinit(allocator);
        self.greased_uploads.deinit(allocator);
        self.greased_verts.deinit(allocator);
        self.greased_indices.deinit(allocator);
        self.pending_uploads.deinit(allocator);
        self.pending_verts.deinit(allocator);
        self.pending_indices.deinit(allocator);
        self.pending_delta_data.deinit(allocator);
        self.host_bytes.deinit(allocator);
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
        b += listBytes(self.highlight_items);
        b += listBytes(self.shadow.items);
        b += listBytes(self.shadow.skins);
        b += listBytes(self.staged_instances);
        b += listBytes(self.ui_vertices);
        b += listBytes(self.ui_indices);
        b += listBytes(self.particle_draws);
        b += listBytes(self.physics_lines);
        b += listBytes(self.morph_uploads);
        b += listBytes(self.morph_data);
        b += listBytes(self.p_cpu_uploads);
        b += listBytes(self.p_cpu_data);
        b += listBytes(self.p_gpu_uploads);
        b += listBytes(self.p_gpu_data);
        b += listBytes(self.p_compute_uploads);
        b += listBytes(self.p_compute_data);
        b += listBytes(self.trail_uploads);
        b += listBytes(self.trail_verts);
        b += listBytes(self.trail_indices);
        b += listBytes(self.soft_uploads);
        b += listBytes(self.soft_data);
        b += listBytes(self.soft_indices);
        b += listBytes(self.greased_uploads);
        b += listBytes(self.greased_verts);
        b += listBytes(self.greased_indices);
        b += listBytes(self.pending_uploads);
        b += listBytes(self.pending_verts);
        b += listBytes(self.pending_indices);
        b += listBytes(self.pending_delta_data);
        b += listBytes(self.host_bytes);
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
/// the threading split (sequential path: single-threaded-only; concurrent
/// paths: claim/pin API only).
pub const FrameDraws = struct {
    slots: [SLOT_COUNT]FrameDrawSlot = .{ .{}, .{}, .{} },
    front: usize = 0,
    /// Consumer pins: pinned slots are never build targets and never publish
    /// targets. Set only via `pin`/`pinFront`, cleared only via `unpin`.
    pinned: [SLOT_COUNT]bool = .{false} ** SLOT_COUNT,
    /// Shared read leases used by producer commit while the context may also
    /// be presenting the current front. Multiple readers are safe; every
    /// writer/rotation path treats a nonzero count as pinned.
    read_pins: [SLOT_COUNT]usize = .{0} ** SLOT_COUNT,
    /// Producer claims: slots currently being filled between `claimBack`
    /// and `tryPublish`/`cancelClaim`. `pin` refuses these (`SlotBusy`).
    writing: [SLOT_COUNT]bool = .{false} ** SLOT_COUNT,
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

// Lease-protocol regression tests live in `frame_draws_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).

test {
    _ = @import("frame_draws_tests.zig");
}
