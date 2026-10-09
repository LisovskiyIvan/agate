//! Scene frame orchestration: queue/view builds, snapshot publish,
//! UI packet staging, prepared-draw accessors, the producer-build
//! claim core, prepare/update/flush/render entries and reuse guards.
//! Split out of `scene.zig` (facade). One-line delegations to the
//! existing frame_* leaves moved here unchanged.
//!
/// Anti-cycle rule (same as `audio/*`, `profiler/*`): every function takes
/// the scene as `anytype` (a `*Scene` from `core.zig` in practice) and this
/// module never imports `core.zig` or the `scene.zig` facade back.
/// Cross-leaf helpers consumed here are `pub` in their home module but are
/// deliberately NOT re-exported by the facade.
const sokol = @import("sokol");
const sapp = sokol.app;
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const particles = @import("../particles.zig");
const UpdateError = particles.UpdateError;
const gpu_thread = @import("../gpu_thread.zig");
const scene_render_queue = @import("render_queue.zig");
const scene_snapshot = @import("snapshot.zig");
const SceneFrameSnapshot = scene_snapshot.SceneFrameSnapshot;
const CameraSnapshot = scene_snapshot.CameraSnapshot;
const scene_frame_draws = @import("frame_draws.zig");
const FrameDrawSlot = scene_frame_draws.FrameDrawSlot;
const scene_queue_builder = @import("queue_builder.zig");
const QueueBuildParams = scene_queue_builder.QueueBuildParams;
const outline_pass = @import("../passes/outline_pass.zig");
const scene_draw = @import("draw.zig");
const scene_ui_capture = @import("ui_capture.zig");
const scene_patch_instances = @import("patch_instance_refs.zig");
const scene_frame_build = @import("frame_build.zig");
const scene_frame_prepare = @import("frame_prepare.zig");
const scene_frame_render = @import("frame_render.zig");
const scene_view_render = @import("view_render.zig");
const scene_probe_render = @import("probe_render.zig");
const scene_gui3d = @import("gui3d_layer.zig");
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;
const SceneStats = @import("stats.zig").SceneStats;
const CubeTexture = @import("../texture.zig").CubeTexture;

/// Stages the update-phase wall time measured by the app around
/// Scene.update. Update-side write (game thread, under update-vs-prepare
/// phase ownership); the staged begin transfers the last tick into
/// stats.update_ms. This is the ONLY update-side timing write — direct
/// `scene.stats.*` writes from the update thread are forbidden (stats is
/// context-owned; render reads it concurrently with update).
pub fn recordUpdateTime(self: anytype, ms: f32) void {
    self.pending_update_ms.store(ms, .release);
}

pub fn recordPhysicsTime(self: anytype, ms: f32) void {
    self.pending_physics_ms.store(ms, .release);
}

/// Shared queue/shadow/outline build parameters (stage-2 increment B):
/// one internal function `buildQueuesInto` fills a `FrameDrawSlot` from
/// these plus the explicit `snap` cameras (never `Scene.frame_snapshot`
/// nor the slot's staged copy: the build must not read the consumed
/// render snapshot). Fallback passes the staged slot snapshot with
/// `frame_id` as `cache_key`, `&self.stats`, the snapshot primary eye,
/// snapshot primary eye, the resolved snapshot sky/ibl, `true`, and
/// `.published` (today's exact behavior). The game-side build passes
/// `&build_snapshot` with the per-attempt build-unique key
/// (`build_cache_seq | (1<<63)` under the producer high-bit namespace —
/// fresh on every build call, including cancelled/repeated ones — never
/// the handoff `build_seq`, which a cancelled build leaves uncommitted
/// for the next build to reuse), the claimed slot's `&back.build_stats`, the frozen snapshot eye,
/// the exact snapshot sky/ibl, `true`, and `.build_view`
/// (provisional buffer/count until the latch patch).
/// Payload identity invariant: every instanced batch / shadow item /
/// outline item carries `source_uid` (stable `Mesh.uid`) + `source_mesh`
/// (mesh-list index at build time); the latch validates uid before
/// finalizing provisional handles (fail-closed zero).
/// `eye` is currently informational (views sort by their own snapshot
/// eye; the build passes the live eye for future transparent-sort use
/// and for the instance CPU staging eye, which is threaded separately).
/// `stats` stays a direct pointer: the game-side build passes the claimed
/// slot's counter accumulator.
// internal, used by scene tests
pub fn prepareViewQueues(
    self: anytype,
    queues: *scene_render_queue.RenderQueues,
    cam_snap: scene_snapshot.CameraSnapshot,
    sky_texture: ?CubeTexture,
    ibl_intensity: f32,
    cache_key: u64,
    stats: *SceneStats,
    instances_prepared: bool,
    instance_source: mesh_mod.InstanceSource,
) void {
    scene_queue_builder.prepareViewQueues(self, queues, cam_snap, sky_texture, ibl_intensity, cache_key, stats, instances_prepared, instance_source);
}

// internal, used by scene tests
pub fn buildQueuesInto(self: anytype, back: *FrameDrawSlot, params: QueueBuildParams) void {
    scene_queue_builder.buildQueuesInto(self, back, params);
}

pub fn renderSceneView(
    self: anytype,
    cam_snap: scene_snapshot.CameraSnapshot,
    queues: *const scene_render_queue.RenderQueues,
    outline_items: []const outline_pass.OutlineDrawItem,
    outline_skins: []const [scene_render_queue.MAX_BONES]Mat4,
    samples: i32,
    snap: *const scene_snapshot.SceneFrameSnapshot,
    env: scene_draw.Environment,
    view_slot: usize,
) void {
    scene_view_render.renderSceneView(self, cam_snap, queues, outline_items, outline_skins, samples, snap, env, view_slot);
}

/// Runs at most one pending reflection-probe capture (wave 25, v1).
/// Context thread only; called from `render` between the shadow depth
/// pass and the main pass, never from `renderReuse` (which re-presents
/// the consumed front, probe snapshot included). Takes the staged slot
/// snapshot from the caller (the front slot's copy `render` already
/// reads) — never the live `frame_snapshot`.
///
/// What one capture does, in order: ensure the probe's GPU target (fail
/// closed when headless or on creation failure — the probe stays dirty
/// and is retried on a later frame), render the six cube faces from the
/// probe position (prepared PRIMARY draw list: opaque + opaque-instanced
/// + transparent, plus the sky; no PIP views, outline, particles,
/// physics-debug, or UI in v1), then run the box-prefilter blit chain
/// over the mip levels. No `UploadQueue`/texture-streaming interaction
/// and no `sg.updateBuffer` traffic on this path — only draws into
/// probe-owned targets — so the single-upload-per-frame discipline is
/// untouched. When several probes are dirty, the lowest dirty + enabled
/// index captures now and the rest wait for later frames (one capture
/// per frame maximum). The fresh content reaches draws one staged prepare
/// later (the snapshot is staged by the build, before `render`
/// captures) — a documented one-frame lag.
pub fn captureDirtyProbes(self: anytype, snap: *const SceneFrameSnapshot) void {
    scene_probe_render.captureDirtyProbes(self, snap);
}

/// Runs at most `max_captures_per_frame` (1) pending 3D-GUI panel
/// captures (wave 28, v1). Lowest dirty + enabled index first; the rest
/// wait for later frames. Fail-closed like the probe path (headless or
/// creation failure keeps the panel dirty for retry). The layer owns the
/// upload + offscreen RT pass; this only budgets the count.
pub fn captureDirtyUi3dPanels(self: anytype) void {
    var n: usize = 0;
    while (n < scene_gui3d.max_captures_per_frame) : (n += 1) {
        const idx = self.gui3d.nextDirtyIndex() orelse return;
        if (!self.gui3d.capturePanel(self.allocator, &self.gpu_retire, idx)) return;
    }
}

/// Packs the current camera, light, shadow, and environment state into an immutable
/// frame snapshot that can be published to the render thread.
pub fn packFrameSnapshot(self: anytype, aspect: f32, cur_w: i32, cur_h: i32) scene_snapshot.SceneFrameSnapshot {
    return scene_snapshot.packFrameSnapshot(self, aspect, cur_w, cur_h);
}

/// Publishes a complete frame snapshot through the lock-free mailbox.
/// When the mailbox is saturated (consumer lagging, both slots
/// published), stale published slots are drained first so the NEWEST
/// snapshot wins — otherwise the build's takeLatest would resurface
/// an older published frame over the newer tick.
pub fn publishFrameSnapshot(self: anytype, aspect: f32, cur_w: i32, cur_h: i32) void {
    scene_snapshot.publishFrameSnapshot(self, aspect, cur_w, cur_h);
}

/// Game-side UI CPU packet staging core behind `BuildClaim.stageUi`:
/// the stage body targeted at `slot`.
pub fn stageUiPacketInto(self: anytype, slot: usize) void {
    scene_ui_capture.stageUiPacketInto(self, slot);
}

/// P6 UI handoff: latches the claimed slot's staged UI packet into the
/// render-owned frame and uploads at this prepare/context boundary.
pub fn captureUiFrame(self: anytype, snap: *const SceneFrameSnapshot, back: *FrameDrawSlot) void {
    scene_ui_capture.captureUiFrame(self, snap, back);
}

/// P7 published consumable draw payload: the prepared mesh draw lists
/// (PRIMARY + ALL PIP view queues with their skin/shader side stores,
/// outline items+skins, prepared shadow items+skins+bin ranges), with the
/// frame_id/retire_epoch that built them, PLUS the staged frame snapshot
/// (wave 27: render/reuse read the slot's copy, never live state). The
/// ONLY low-level draw accessor — render and P5/P7 tests read through
/// here, never raw fields.
/// Scope is mesh draws only: UI (P6 ui_frame), physics-debug lines
/// (prepared_lines + committed DebugPass upload), sky params + default
/// copies (staged slot snapshot), and particles (prepared frame) are
/// separate payloads — none of them reads live subsystems at draw time. Trail
/// meshes ride these same queues: Trail.update stages CPU-side, the
/// prepare flush uploads, and the queue build bakes the values.
///
/// Borrow rules: every GPU handle inside is BORROWED (phase mutex / P3
/// epochs, no second GPU copies); CPU slot retention is NOT a GPU
/// lifetime pin. GPU consumability ends at the consuming render's return
/// or — when no render consumes the frame — at the START of the next
/// staged prepare (a repeated begin discards the pending frame before
/// GpuRetire.begin/flush, and that flush may tear down its borrowed
/// handles); it also ends at deinit. Retained CPU storage may be reused
/// as back scratch by any prepare, so the returned pointer (and any slice
/// taken from it) is consumable only while frame_prepared is set or
/// during the render call consuming this frame (including the inner
/// render of `renderReuse`, which re-draws the already-consumed front
/// without a prepare) — never across a prepare boundary — UNLESS the
/// caller holds a consumer pin on the slot (`FrameDraws.pin`), which
/// render itself does for the whole draw (see render). Render
/// completes the frame epoch on all returns (no-camera too); the reuse
/// re-run re-completes the same epoch, which is idempotent (no-op).
/// One pending frame, no concurrent prepare/render — but update
/// CAN overlap render (update-vs-prepare stay excluded instead).
pub fn preparedDraws(self: anytype) *const FrameDrawSlot {
    return &self.draws.slots[self.draws.front];
}

/// Latch patch finalizing game-built provisional handles (stage-2
/// increment B, context side, allocation-free): after
/// `stageInstancesLatch` publishes, every instanced payload entry built
/// with `.build_view` is re-resolved by identity (`source_mesh` index +
/// `source_uid` validation) against the SLOT-OWNED staged records — never
/// against the live mesh list.
///
/// Payload identity invariant: `source_uid` is the stable `Mesh.uid`,
/// `source_mesh` the mesh-list index at build time. Provisional vs
/// finalized: at build time `instance_buffer`/`visible_instance_count`
/// (plus shadow `world_aabb`/`max_dim`, outline `world_center`) came from
/// the provisional `instance_build_view` (frozen count/bounds + old
/// handle); here they are finalized from the post-latch RECORD mirror
/// (which the game-side commit applies verbatim to `instance_render`,
/// so the values are identical to reading live state — without the
/// race).
///
/// Records are appended in mesh-list order (strictly increasing
/// `mesh_index`), so the lookup below is a linear scan with early exit.
/// A mesh-list mutation between build and latch can therefore NOT slip
/// a stale entry through the COMMIT (the game-side guard skips displaced
/// meshes there); the patch itself resolves purely from the records —
/// a record the latch published finalizes, a record the latch
/// fail-closed (`staged_frame != frame_id`) zeroes. The live mesh list
/// is validated at commit time, not patch time — the patch performs
/// zero live reads (outline stale fallbacks use the record-frozen
/// `mesh_position`, not live `mesh.position`).
///
/// Per entry (only `is_instanced` shadow/outline items and all instanced
/// batches; regular items have no provisional handle and are skipped):
/// - find the record with `mesh_index == source_mesh`; missing (mesh
///   list shrank, or the mesh never froze a record) → fail-closed zero.
///   Outline center falls back to `Vec3.zero` (no record to read).
/// - `record.uid != source_uid` (reorder changed the list) →
///   fail-closed zero; outline center falls back to the record-frozen
///   `mesh_position` (the mesh is known via the record).
/// - `record.staged_frame != frame_id` (latch skipped/failed this
///   record) → fail-closed zero with the same outline fallback.
/// - else copy `buffer`/`count` (+ shadow `world_aabb` from
///   `record.bounds` with `max_dim` recomputed from extents exactly as
///   `prepareInto` did, outline `world_center` from `record.bounds`
///   center or the frozen `mesh_position` when invalid).
/// - Transparent order entries referencing zeroed batches need no distance
///   change: the draw skips `count==0` batches, so order is harmless.
/// - Culling/inclusion stay frozen at build time (bounds/model/distance
///   are NOT repatched): a live TRS mutation between build and latch
///   never alters this frame's sets, only the next build sees it.
pub fn patchInstanceRefs(self: anytype, back: *FrameDrawSlot) void {
    _ = self;
    scene_patch_instances.patchInstanceRefs(back);
}

/// Stage-2 producer build (game/update phase, CPU-only, sg-free): the sole
/// Stage-2 producer build (game/update phase, CPU-only, sg-free): the sole
/// producer convenience over the claim flow (`tryClaimBuildSlot` + `build` +
/// `stageUi` + `publish` on the SAME path). Returns true when a fully built
/// frame published, false when every non-front slot was pinned/claimed
/// (counted latest-wins skip — the context reuses the last front).
/// The producer freezes only: instance scratch + previews + staged records +
/// particle/physics captures + upload packets + snapshot + queue payload.
/// The context latch consumes the staged slot copy unconditionally.
pub fn buildPreparedFrame(self: anytype) bool {
    var claim = self.tryClaimBuildSlot() orelse return false;
    claim.build();
    claim.stageUi();
    claim.publish();
    return true;
}

/// Shared build core behind `buildPreparedFrame` and `BuildClaim.build`:
/// the exact historical build body targeted at the claimed `slot` under
/// the reserved generation `seq` (preview stamps, record freeze). Commits
/// NOTHING global: `build_seq`/`build_slot` are stamped by
/// `BuildClaim.publish`, so a cancelled claim leaves no handoff behind.
/// The world-cache key is per-attempt (`build_cache_seq | (1<<63)`,
/// bumped on every call) rather than per-`seq`, so cancelled/repeated
/// builds never alias cache entries with the next real build.
///
/// EPOCH DISCIPLINE (wave 29): this core must never call
/// `GpuRetire.begin`/`complete`/`flush` and never passes a `retire_queue`
/// anywhere — epochs stay context-owned (prepare/render own the
/// begin/complete pairing); enforced by test.
pub fn buildIntoClaimedSlot(self: anytype, slot: usize, seq: usize) void {
    scene_frame_build.buildIntoClaimedSlot(self, slot, seq);
}

pub const PrepareClaim = scene_frame_prepare.PrepareClaim;

/// Begins staged preparation: claims the latest fully built producer frame
/// and holds its slot for the whole prepare. Returns null when no fresh
/// producer frame is ready — never a live read.
///
/// Token contract: every successful begin MUST be paired with exactly one
/// `finishStagedPrepare` or `cancelStagedPrepare` on the same thread, on
/// every path including errors. A violated pairing (stale/lost/double
/// token) is logged and the active claim is released defensively — the
/// frame is lost, but the pipeline keeps running instead of wedging.
/// A `render` between begin and finish sees `frame_prepared == false` and
/// drops the present, so keep the pair adjacent around the unlock window.
/// Only the context thread may call any of the three.
pub fn beginStagedPrepare(self: anytype) ?PrepareClaim {
    return scene_frame_prepare.beginPrepare(self);
}

/// Completes a claim from `beginStagedPrepare`. The producer lock may be
/// released before this call: a staged claim consumes only slot-owned and
/// context-owned state. Consumes the token (one-shot; a second call asserts).
pub fn finishStagedPrepare(self: anytype, claim: PrepareClaim) void {
    scene_frame_prepare.finishPrepare(self, claim);
}

/// Releases a claim from `beginStagedPrepare` without publishing: drops the
/// slot lease (restoring the handoff when no newer producer generation
/// superseded it) and closes the begin-side retire epoch. Consumes the
/// token (one-shot; a second call asserts).
pub fn cancelStagedPrepare(self: anytype, claim: PrepareClaim) void {
    scene_frame_prepare.cancelPrepare(self, claim);
}

/// Stage 3, slice 2: the game-side update entry point. Everything the
/// simulation advances per frame, in one call, in the canonical order
/// (camera -> lights -> physics -> animations -> soft bodies ->
/// particles -> decals);
/// render() then consumes the published frame values (light_pack,
/// staged slot snapshot, prepared draws/UI/debug/sky/particles) without
/// simulating anything itself.
///
/// Runs on the game side and MAY overlap render (update||render): it must
/// touch ONLY update-owned state (live cameras/lights/world/meshes/
/// materials/canvas/particles/trails/nav + the mailboxes +
/// pending_update_ms). It must NEVER touch stats/Profiler/render-owned
/// caches or prepared payloads, and never call sg.* (frozen upload packets
/// flush on the context thread during staged prepare). A single producer
/// owns these mutations and freezes them before publishing its claim.
///
/// Deliberately NOT included: `updateTrails` and `updateNavAgents` —
/// both require real-seconds dt (the 60fps-normalized dt breaks their
/// SI tuning), so apps drive them explicitly with their own time base.
/// Those explicit CPU mutators run on the same producer before its build
/// (prepare/render see only their staged/uploaded results) — the fact
/// that `Scene.update` skips them is a dt-base distinction, not a
/// thread-ownership one: no render path traces into their live state.
///
/// Stage 1: after this (and any explicit mutators), the app may call
/// `buildPreparedFrame` on the game side to freeze the CPU payloads and UI
/// packet the context-side prepare will consume.
/// Without a fresh build the staged begin returns null and the context
/// reuses the last front or skips the present.
pub fn update(self: anytype, dt: f32) particles.UpdateError!void {
    self.updateCamera(dt);
    self.updateLights(dt);
    self.updatePhysics(dt);
    self.updateAnimations(dt);
    self.updateSoftBodies(dt);
    try self.updateParticles(dt);
    self.updateDecals(dt);

    if (self.post_process.auto_exposure_enabled) {
        const est_lum = self.estimateSceneLuminance();
        _ = self.updateAutoExposure(est_lum, dt);
        self.post_process.auto_exposure_camera_cut = false;
    }

    const cur_w = sapp.width();
    const cur_h = sapp.height();
    const aspect = if (cur_h > 0) @as(f32, @floatFromInt(cur_w)) / @as(f32, @floatFromInt(cur_h)) else 1.0;
    self.publishFrameSnapshot(aspect, cur_w, cur_h);
}

/// Stage 3: uploads the GPU buffers that the update phase staged
/// (particle instances, soft-body cloth vertices, trail geometry).
/// Quiesced-context drain only: the producer must be stopped/excluded while
/// this scans live arrays. Normal frames use immutable slot upload packets,
/// never this routine; the update phase remains free of sg.* calls.
pub fn flushPendingGpuUploads(self: anytype) void {
    gpu_thread.assertOnContextThread();
    // Deferred off-context destroys first: unlinking already happened in
    // destroyMesh, this completes the GPU teardown (deinit + free) for
    // entries whose epoch already completed (see GpuRetireQueue.flush).
    self.gpu_retire.flush(self.allocator);
    // Deferred off-context creations (uploadGeometry): finish the vertex/
    // index buffers before queue building can reference them. Plain scan
    // over meshes — the loop below already visits every mesh, so the
    // pending check adds no traversal, just one branch per mesh.
    for (self.meshes.items) |m| m.finishGpuUpload(self.allocator);
    for (self.particles.systems.items) |ps| ps.flushGpuUploads();
    for (self.softbodies.bodies.items) |b| b.flushGpuUploads();
    for (self.trails.meshes.items) |tm| tm.flushGpuUploads();
    for (self.greased_lines.items) |gl| gl.flushGpuUploads();
    for (self.meshes.items) |m| m.flushGpuUploads();
    // Standalone completion (quiesced-context drains — NOT the staged
    // begin path, whose frame a render commits): the
    // compute-particle dispatch above can open the frame's command
    // buffer, and a buffer that is never committed keeps its in-flight
    // semaphore forever — sg_shutdown waits NUM_INFLIGHT_FRAMES signals
    // unconditionally and hangs at exit (observed: waits=commits+1 in
    // an instrumented soak). Nil-buffer commit is a no-op; headless
    // (no sg.setup) skips.
    if (!self.flush_in_prepare and sg.isvalid()) sg.commit();
}

/// Render entry: draws the frame published by the staged prepare (context
/// thread, SEQUENTIAL with prepare — never concurrent; update MAY run
/// concurrently on the game side). Reads ONLY render-owned captures
/// (prepared draws + the staged slot snapshot incl. sky/default copies,
/// ui_frame, debug capture + committed upload, particle prepared frame)
/// plus BORROWED GPU handles under P3 epochs. No global phase lock is
/// taken here: the app unlocks update-vs-prepare ownership BEFORE calling
/// render (see main), and prepare already ran. Takes no sg.* outside the
/// context thread (asserted). Every subsystem above is snapshot-driven;
/// no live subsystem reads remain on this path.
///
/// Non-blocking consumer: when the app skipped the phase-lock acquire it
/// calls `renderReuse` instead, which re-draws the already-consumed front
/// through this same function with `rendering_reuse` set — the staged
/// begin below is skipped and the profiler tail is
/// suppressed while the stats are still accumulating inside; the
/// `renderReuse` wrapper records the re-presented frame after restoring
/// them. Every presented frame is recorded once, including reuses.
pub fn render(self: anytype) void {
    scene_frame_render.render(self);
}

/// True once a prepare published a frame (front slot `frame_id != 0`).
/// The app blocks once for the first prepare while this is false instead
/// of reusing: before the first successful prepare nothing is consumable,
/// and reuse must not prepare without phase ownership (the game side may
/// be mid-mutation).
pub fn hasConsumableFrame(self: anytype) bool {
    return self.draws.slots[self.draws.front].frame_id != 0;
}

/// Current reuse streak: consecutive `renderReuse` presents since the
/// last successful prepare (0 right after any prepare). Context thread
/// only. Together with `pendingRetires()` this bounds what a skip streak
/// can hide: new uploads stay pending, retires stay queued (capped).
pub fn reuseStreak(self: anytype) u64 {
    return self.reuse_streak;
}

/// Prepares that consumed a staged UI packet as the geometry source
/// (`capturePacket`). The staged-absence clear does not count.
pub fn uiPacketLatchedCount(self: anytype) u64 {
    return self.ui_packet_latched;
}

/// Retire entries currently awaiting the next successful prepare's
/// leading flush (queue + overflow spillover). Grows with
/// streak × destroy-rate, bounded by `GpuRetireQueue.pending_cap`.
pub fn pendingRetires(self: anytype) usize {
    return self.gpu_retire.retainedCount();
}

/// Non-blocking render-consumer reuse: re-draws the current front slot
/// without a prepare, for frames where the app skipped the phase-lock
/// acquire (lock contended) instead of stalling the present.
///
/// Takes no lease claim (nothing is built, nothing publishes): reuse
/// only re-presents the pinned front through the inner `render` — the
/// claim protocol is untouched, and a concurrent game claim can proceed
/// against any other slot while the present holds its pin.
///
/// Reuse contract: the caller reuses only when `hasConsumableFrame()`
/// is true (debug assert below enforces it: front slot `frame_id != 0`,
/// i.e. at least one prepare published a frame). The app skipped prepare
/// because the phase lock was busy, so `frame_prepared` is false and the
/// front slot holds the last consumed frame. Calling with a pending
/// prepared frame (`frame_prepared` true) is a contract violation
/// (debug assert) — consume pending frames with `render`, never
/// `renderReuse`.
///
/// No GpuRetire begin/flush runs here (no prepare): pending retire
/// entries (queue + overflow[8]) stay queued until the next successful
/// prepare's leading flush, which destroys only entries whose epoch
/// already completed — the reused frame's borrowed handles stay valid
/// precisely because no prepare ran. The inner render's
/// `complete(retire_epoch)` re-run is idempotent (same epoch: no-op), as
/// is the `frame_prepared = false` store; no other render step is
/// prepare-frame-epoch dependent.
///
/// Skip-streak retention (bounded, observable): every skipped prepare
/// defers flush AND pending-upload completion, so `GpuRetireQueue.pending`
/// grows with skip-streak × destroy-rate until the next successful
/// prepare drains it. The growth is CAPPED (`pending_cap`, default 8192 —
/// steady state holds a handful; anything past the cap is a counted +
/// logged drop, never silent) and OBSERVABLE (`reuseStreak()` counts the
/// streak, `pendingRetires()`/`cappedDropCount()` the pileup and drops).
/// Previously this was bounded only by the phase-mutex coupling (every
/// frame ran prepare+flush); with reuse the bound is the cap plus the
/// app's contract — do not streak reuse indefinitely — documented here,
/// not wished away. Borrowed handles stay valid throughout the streak
/// (no flush ran), and new meshes stay `gpu_pending` (invisible: queue
/// builds skip them until a successful prepare completes their uploads)
/// — staleness visible as missing objects, never corruption.
///
/// Stats are saved and restored around the inner render because the frame
/// was already recorded; the upload meter is still drained by the inner
/// render and its tally discarded with the restored stats.
///
/// Recording: every presented frame is recorded once, including reuses.
/// After the restore above, while still recording, the seq is bumped and
/// the re-shown frame recorded with the consumed frame's metrics
/// (identical draws; wall pacing/timestamps are real). The inner render's
/// own tail stays suppressed via `rendering_reuse`, so no double record.
pub fn renderReuse(self: anytype) void {
    scene_frame_render.renderReuse(self);
}
