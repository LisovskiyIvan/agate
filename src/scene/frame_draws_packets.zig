//! Frozen per-owner upload-packet descriptors for `FrameDrawSlot`
//! (morph / particle CPU-GPU-compute / trail / softbody / greased /
//! pending-mesh packets plus the staged instance record alias).
//! Section of the `frame_draws` facade (pure move; see `frame_draws.zig`
//! for the full ownership / freeze-then-latch / lease contract).

const mesh_mod = @import("../mesh.zig");

/// Staged per-mesh instance record (slot-owned; stored in
/// `FrameDrawSlot.staged_instances`, frozen by `freezeStagedRecords`).
pub const StagedInstanceRecord = mesh_mod.StagedInstanceRecord;

/// Slot-owned dynamic-upload packets (producer freeze-then-latch):
/// every per-frame GPU staging payload the prepare flush uploads is frozen
/// here by value on the producer side (`buildIntoClaimedSlot` via
/// `upload_packets.stageUploads`) and consumed by the staged prepare
/// (`upload_packets.flushSlotUploads`) instead of any live mutable array.
/// All descriptors are plain integers (tokens/ids/counts/offsets); the byte
/// payloads live in the sibling flattened lists below. Buffer ids are
/// borrowed values under the retire-epoch discipline (never destroyed/retired
/// through the packet). `reset` clears lengths retaining capacity; `deinit`
/// frees; `cpuBytes` counts retained capacities.
pub const MorphUpload = struct {
    token: usize = 0,
    uid: u64 = 0,
    mesh_index: u32 = 0,
    buffer_id: u32 = 0,
    count: u32 = 0,
    data_lo: usize = 0,
    dirty: bool = false,
    /// Lock-free publication outcome (phase 2, context-written): the staged
    /// flush sets true once the frozen bytes landed in `buffer_id` (or the
    /// packet was empty). The producer cleared `morph_upload_needed` at
    /// stage time; a false outcome means the game-side commit must re-arm
    /// it for retry. Never written by the producer after staging.
    delivered: bool = false,
};
pub const ParticleCpuUpload = struct {
    token: usize = 0,
    uid: u64 = 0,
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
    dirty: bool = false,
};
pub const ParticleGpuUpload = struct {
    token: usize = 0,
    uid: u64 = 0,
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
    dirty: bool = false,
};
pub const ParticleComputeUpload = struct {
    token: usize = 0,
    uid: u64 = 0,
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
    /// the direct drain — never a silent fallback, never a retry spin.
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
    uid: u64 = 0,
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
    dirty: bool = false,
};
pub const SoftUpload = struct {
    token: usize = 0,
    uid: u64 = 0,
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
    dirty: bool = false,
};
pub const GreasedUpload = struct {
    token: usize = 0,
    uid: u64 = 0,
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
    dirty: bool = false,
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
    /// every non-morph and CPU-morph mesh (no bytes frozen; non-GPU-morph
    /// paths untouched).
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
