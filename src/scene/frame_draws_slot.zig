//! Retained prepared-frame slot: one coherent prepared frame (mesh draw
//! lists, outline/highlight/shadow payloads, staged records, UI packet,
//! snapshot, frozen captures, upload packets) plus its lifecycle
//! (`reset` / `deinit` / `cpuBytes`).
//! Section of the `frame_draws` facade (pure move; see `frame_draws.zig`
//! for the full ownership / rotation / lease contract).

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
const packets_mod = @import("frame_draws_packets.zig");

pub const RenderQueues = render_queue.RenderQueues;
pub const SkinStorage = render_queue.SkinStorage;
pub const OutlineDrawItem = outline_pass.OutlineDrawItem;
pub const HighlightDrawItem = highlight_pass.HighlightDrawItem;
pub const PreparedShadowDraws = shadow_pass.ShadowPass.PreparedShadowDraws;
pub const Epoch = retire_mod.Epoch;

/// Canonical packet descriptors live in `frame_draws_packets.zig`; these
/// aliases keep the slot field types unqualified (pure move, bodies unchanged).
pub const StagedInstanceRecord = packets_mod.StagedInstanceRecord;
pub const MorphUpload = packets_mod.MorphUpload;
pub const ParticleCpuUpload = packets_mod.ParticleCpuUpload;
pub const ParticleGpuUpload = packets_mod.ParticleGpuUpload;
pub const ParticleComputeUpload = packets_mod.ParticleComputeUpload;
pub const TrailUpload = packets_mod.TrailUpload;
pub const SoftUpload = packets_mod.SoftUpload;
pub const GreasedUpload = packets_mod.GreasedUpload;
pub const PendingMeshUpload = packets_mod.PendingMeshUpload;

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
    /// Borrowed GPU handles under the retire-epoch discipline, same as
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
