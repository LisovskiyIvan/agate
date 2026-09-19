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
};
