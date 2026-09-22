// Per-frame render statistics. Lives in its own leaf module so the scene
// subsystems (render queue, draw helpers) can share a *SceneStats without
// importing scene.zig (no import cycles); Scene re-exports the type under
// its historical name.
//
// Frame-metric fields (uploaded_*, *_ms) are published by Scene at render
// start and measured around the phase boundaries: upload counters are
// staged by prepareFrame (texture drain tally), prepare_ms is written by the
// app around prepareFrame (same context thread as render), update_ms arrives
// via Scene.recordUpdateTime -> pending_update_ms -> prepare transfer (see
// ownership below), shadow/main/post_ms are timed inside Scene.render. All
// default to zero, so `SceneStats{}` and `self.stats = .{}` stay valid resets.
//
// Владение при actual update||render (update CAN overlap render; prepare and
// render stay SEQUENTIAL on the context thread, next prepare NEVER runs
// concurrently with render):
// - update_ms пишет ТОЛЬКО context-поток: prepareFrame переносит в stats
//   последний тик pending_update_ms, который игровой поток сложил через
//   Scene.recordUpdateTime (фазовый мьютекс update-vs-prepare; это staged
//   f32, НЕ stats). Прямая запись scene.stats.update_ms с игрового потока
//   ЗАПРЕЩЕНА: stats читает render (Profiler.recordFrame в конце
//   Scene.render) конкурентно с update — поле обязано быть context-owned.
//   prepareFrame сбрасывает stats, сохраняя перенесённый update_ms и
//   app-записанный prepare_ms (handoff через pending/keep).
// - prepare_ms пишет app на context-потоке (frame() вокруг prepareFrame —
//   тот же поток, что render); uploaded_*, shadow/main/post_ms и все
//   счётчики пишет context-поток (prepareFrame/render); updated_bytes_frame
//   идёт через атомарный gpu_upload_meter (воркеры стейджинга пишут record(),
//   render забирает takeAndReset()). UI-байты с P6 записываются в prepare
//   (capture-upload в prepareFrame), а не в render — сумма за кадр та же,
//   меняется только prepare-vs-render атрибуция.
// - Profiler целиком render-owned: recordFrame вызывается только в конце
//   render, start/stop/reset/saveReports — с context-потока между
//   submissions (окно/тулы; F8-колбэк того же потока, конкурентного render
//   там нет), captureMemorySnapshot — с context-потока под фазовым мьютексом
//   (читает живые регистры, update исключён). Воркеры/игровой поток к
//   Profiler не прикасаются никогда.
// Поля остаются обычными (не атомарными): синхронизация фазовая
// (phase_mutex update-vs-prepare + границы prepare/render), а не поточечная;
// update_ms/статистика и pending тик — РАЗНЫЕ поля, поэтому update||render
// не делят ни одного слова памяти.
//
// Stage 1 producer build (`Scene.buildPreparedFrame`, game/update side)
// intentionally writes NO stats here — not even counters: stats stay
// context-owned (prepare/render), and the build's `build_seq` counters live
// on Scene/the subsystems, never in this struct. A concurrent update must
// never observe a half-written stats word while render reads it.
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
    /// Textures uploaded by prepareFrame's bounded drain this frame.
    uploaded_textures_frame: u32 = 0,
    /// GPU bytes uploaded this frame: ТОЛЬКО текстуры из UploadQueue
    /// (drainCountedBudget, бюджет upload_byte_budget_per_frame = 8 MiB).
    /// Динамические обновления буферов через sg.updateBuffer/appendBuffer
    /// (инстансы, морфы, частицы, трейлы, UI, debug-линии) учитываются
    /// отдельно в updated_bytes_frame и на троттлинг не влияют.
    uploaded_bytes_frame: u64 = 0,
    /// Байты динамических обновлений GPU-буферов за кадр (uncounted-budget):
    /// сумма диапазонов всех фактических sg.updateBuffer/appendBuffer,
    /// накопленная через gpu_upload_meter (сброс в prepareFrame, перенос
    /// в render перед Profiler.recordFrame).
    updated_bytes_frame: u64 = 0,
    /// Wall-clock phase timings in milliseconds (see header for writers).
    update_ms: f32 = 0,
    prepare_ms: f32 = 0,
    shadow_ms: f32 = 0,
    main_ms: f32 = 0,
    post_ms: f32 = 0,
    /// Last COMPLETED GPU frame time in ms (gpu_timing.pollFrameMs, set in
    /// render right after sg.commit; Metal-only via the vendored sokol
    /// patch, 0 when disabled/headless/unsupported). Lags one frame behind
    /// the CPU submit (async GPU execution). Context-owned like the phase
    /// timings above: written by render, never merged from game-side builds.
    gpu_frame_ms: f32 = 0,
    /// Last COMPLETED per-pass GPU times in ms (gpu_timing.pollPassMs,
    /// set in render right after sg.commit alongside `gpu_frame_ms`).
    /// Real `GL_TIME_ELAPSED` samples on GL4.1; always 0 on Metal (the
    /// frame timer is the only Metal GPU number — one command buffer per
    /// frame) and 0 when disabled/headless/unsupported. Same ownership
    /// and lag semantics as `gpu_frame_ms`; like it, never merged from
    /// game-side builds (see `mergeFrom` below).
    gpu_shadow_ms: f32 = 0,
    gpu_main_ms: f32 = 0,
    gpu_post_ms: f32 = 0,

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
    /// Timing/upload/size fields (update_ms/prepare_ms/shadow_ms/main_ms/
    /// post_ms, gpu_frame_ms/gpu_shadow_ms/gpu_main_ms/gpu_post_ms,
    /// uploaded_*/updated_bytes_frame, draw_calls/triangles/etc.)
    /// are NEVER merged: they are context-owned (prepare/render), and the
    /// game-side build must not observe or disturb them.
    pub fn mergeFrom(self: *SceneStats, other: *const SceneStats) void {
        self.total_meshes += other.total_meshes;
        self.rendered_meshes += other.rendered_meshes;
        self.culled_meshes += other.culled_meshes;
        self.occluded_meshes += other.occluded_meshes;
        self.occluders_count = other.occluders_count;
        self.occluder_triangles = other.occluder_triangles;
    }
};
