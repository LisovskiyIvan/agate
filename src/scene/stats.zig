// Per-frame render statistics. Lives in its own leaf module so the scene
// subsystems (render queue, draw helpers) can share a *SceneStats without
// importing scene.zig (no import cycles); Scene re-exports the type under
// its historical name.
//
// Frame-metric fields (uploaded_*, *_ms) are published by Scene at render
// start and measured around the phase boundaries: upload counters are
// staged by the staged begin (texture drain tally), prepare_ms is written by the
// app around the staged begin (same context thread as render), update_ms arrives
// via Scene.recordUpdateTime -> pending_update_ms -> staged-begin transfer (see
// ownership below), shadow/main/post_ms are timed inside Scene.render. All
// default to zero, so `SceneStats{}` and `self.stats = .{}` stay valid resets.
//
// Ownership during concurrent update and render:
// - update_ms is written ONLY by the GPU context thread: staged begin transfers
//   the latest pending_update_ms tick recorded from the update thread. Direct
//   writes to scene.stats.update_ms from the update thread are prohibited.
// - prepare_ms is written by the app on the context thread; uploaded_*,
//   shadow/main/post_ms and all counters are written by the context thread.
//   updated_bytes_frame tracks dynamic buffer updates via gpu_upload_meter.
// - Profiler is entirely context/render-owned.
// Fields use phase-based synchronization and do not require atomic overhead.
//
// Stage 1 producer build (`Scene.buildPreparedFrame`, game/update side)
// intentionally writes NO stats here — not even counters: stats stay
// context-owned (prepare/render), and the build's `build_seq` counters live
// on Scene/the subsystems, never in this struct. A concurrent update must
// never observe a half-written stats word while render reads it.
//
// GPU timing (opt-in via gpu_timing): render reads optional
// `pollFrameSample`/`pollPassSample` AFTER sg.commit and stores each
// completed measurement's duration (floats below, names kept for
// compatibility) PLUS its completed-submission id (`*_submit`, 0 = absent)
// and the backend frame scope (`gpu_frame_scope`). Availability is the
// submit id, not the duration: a valid quantized zero (submit != 0,
// ms == 0) is a real measurement. Scopes are backend-specific
// (Metal command-buffer time, WebGPU first-to-last native-pass span,
// GL same-frame phase sum; a phase bracket may span multiple real
// render/compute passes) and MUST NOT be described as interchangeable
// full-frame values, nor mixed across submission ids to fabricate a frame
// duration — each channel keeps its own last-completed sample.

const std = @import("std");
const gpu_timing = @import("../gpu_timing.zig");
pub const SceneStats = struct {
    total_meshes: u32 = 0,
    rendered_meshes: u32 = 0,
    culled_meshes: u32 = 0,
    occluded_meshes: u32 = 0,
    occluders_count: u32 = 0,
    occluder_triangles: u32 = 0,
    draw_calls: u32 = 0,
    shadow_draw_calls: u32 = 0,
    main_draw_calls: u32 = 0,
    post_draw_calls: u32 = 0,
    triangles: u32 = 0,
    pipeline_switches: u32 = 0,
    /// Render items/batches/upload packets dropped by the queue build or the
    /// staged upload flush because an allocation failed (fail-closed OOM:
    /// the item vanishes coherently, never half-appended). Merged from the
    /// game-side `build_stats` like the mesh counters above, so a frame that
    /// silently lost draws is diagnosable instead of invisible.
    build_oom_drops: u32 = 0,
    /// Textures uploaded by the staged begin's bounded drain this frame.
    uploaded_textures_frame: u32 = 0,
    /// GPU bytes uploaded this frame from UploadQueue textures (subject to 8 MiB budget).
    uploaded_bytes_frame: u64 = 0,
    /// Bytes of dynamic GPU buffer updates this frame (instances, morphs, particles, UI).
    updated_bytes_frame: u64 = 0,
    /// Wall-clock phase timings in milliseconds (see header for writers).
    update_ms: f32 = 0,
    physics_ms: f32 = 0,
    prepare_ms: f32 = 0,
    shadow_ms: f32 = 0,
    main_ms: f32 = 0,
    post_ms: f32 = 0,
    /// Last COMPLETED GPU frame time in ms (gpu_timing.pollFrameSample,
    /// read in render right after sg.commit; 0 when no completed sample is
    /// available — which is also what a valid quantized zero reads as, so
    /// availability is `gpu_frame_submit != 0`, never the duration).
    /// Async GPU execution: the sample belongs to an earlier completed
    /// submission, observed here after commit (no fixed one-frame lag is
    /// promised). Context-owned like the phase timings above: written by
    /// render, never merged from game-side builds.
    gpu_frame_ms: f32 = 0,
    /// Completed-submission id for `gpu_frame_ms` (gpu_timing.Sample
    /// frame_index; 0 = absent/unavailable). Preserved alongside the
    /// duration so profiler summaries and traces can dedup repeated async
    /// polls of the same submission and count valid zeros.
    gpu_frame_submit: u32 = 0,
    /// Backend scope of the frame measurement (gpu_timing.capabilities at
    /// commit time; .none when unavailable). Backend-specific, never a
    /// cross-backend interchangeable full-frame value.
    gpu_frame_scope: gpu_timing.FrameScope = .none,
    /// Last COMPLETED per-pass GPU times in ms (gpu_timing.pollPassSample
    /// per phase, read in render right after sg.commit alongside
    /// `gpu_frame_ms`). A skipped phase stores no sample (duration 0,
    /// submit 0), never a stale one. Same ownership and async semantics
    /// as `gpu_frame_ms`; like it, never merged from game-side builds
    /// (see `mergeFrom` below). Phase brackets may span multiple real
    /// render/compute passes; pass values from different submission ids
    /// must never be summed to fabricate a frame duration.
    gpu_shadow_ms: f32 = 0,
    gpu_main_ms: f32 = 0,
    gpu_post_ms: f32 = 0,
    /// Completed-submission ids for the per-pass measurements above
    /// (0 = absent: unsupported, disabled, or the phase was skipped).
    gpu_shadow_submit: u32 = 0,
    gpu_main_submit: u32 = 0,
    gpu_post_submit: u32 = 0,

    /// Deferred merge of a game-side queue build (stage-2 increment B): adds
    /// the counter fields `buildFrameQueues` produces into the context-owned
    /// latch stats, then the caller resets the build stats to `.{}`.
    /// Per-field mapping (see render_queue.zig writes):
    /// - total_meshes/rendered_meshes/culled_meshes/occluded_meshes: `+=`
    ///   (queue builds accumulate per mesh/view; parallel chunks merge the
    ///   same way).
    /// - occluders_count/occluder_triangles: `=` (queue builds ASSIGN from
    ///   the occlusion culler per view, last view wins; the merge mirrors
    ///   that assignment — with `self` starting from the latch reset the two
    ///   forms coincide for one build, and `=` keeps multi-view overwrite
    ///   semantics instead of summing).
    /// - build_oom_drops: `+=` (items/batches/packets dropped on allocation
    ///   failure by the build and the staged upload flush; fail-closed
    ///   drops stay observable).
    /// Timing/upload/size fields (update_ms/prepare_ms/shadow_ms/main_ms/
    /// post_ms, gpu_frame_ms/gpu_shadow_ms/gpu_main_ms/gpu_post_ms plus
    /// gpu_frame_submit/gpu_shadow_submit/gpu_main_submit/gpu_post_submit
    /// and gpu_frame_scope, uploaded_*/updated_bytes_frame,
    /// draw_calls/triangles/etc.)
    /// are NEVER merged: they are context-owned (prepare/render), and the
    /// game-side build must not observe or disturb them.
    pub fn mergeFrom(self: *SceneStats, other: *const SceneStats) void {
        self.total_meshes += other.total_meshes;
        self.rendered_meshes += other.rendered_meshes;
        self.culled_meshes += other.culled_meshes;
        self.occluded_meshes += other.occluded_meshes;
        self.occluders_count = other.occluders_count;
        self.occluder_triangles = other.occluder_triangles;
        self.build_oom_drops += other.build_oom_drops;
    }
};
