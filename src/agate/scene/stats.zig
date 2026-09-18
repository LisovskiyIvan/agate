// Per-frame render statistics. Lives in its own leaf module so the scene
// subsystems (render queue, draw helpers) can share a *SceneStats without
// importing scene.zig (no import cycles); Scene re-exports the type under
// its historical name.
//
// Frame-metric fields (uploaded_*, *_ms) are published by Scene at render
// start and measured around the phase boundaries: upload counters are
// staged by prepareFrame (texture drain tally), update_ms/prepare_ms are
// written by the app around Scene.update/prepareFrame (see agate main),
// shadow/main/post_ms are timed inside Scene.render. All default to zero,
// so `SceneStats{}` and `self.stats = .{}` stay valid resets.
//
// Владение при будущем parallel update/render (P2):
// - update_ms пишет игровой поток (app вокруг Scene.update), читает
//   render-поток (Profiler.recordFrame в конце Scene.render). Гонки сегодня
//   нет: обе фазы держит phase_mutex (frame() в main), а сброс prepareFrame
//   сохраняет оба значения (handoff через keep_update_ms/keep_prepare_ms).
// - prepare_ms, uploaded_*, shadow/main/post_ms и все счётчики пишет
//   context-поток (prepareFrame/render); updated_bytes_frame идёт через
//   атомарный gpu_upload_meter (воркеры стейджинга пишут record(),
//   render забирает takeAndReset()). UI-байты с P6 записываются в prepare
//   (capture-upload в prepareFrame), а не в render — сумма за кадр та же,
//   меняется только prepare-vs-render атрибуция.
// - Profiler целиком render-owned: recordFrame вызывается только в конце
//   render, captureMemorySnapshot/summarize/analyze — с render-потока или
//   тулов. Игровой поток к Profiler не прикасается.
// Поля остаются обычными (не атомарными): синхронизация фазовая
// (phase_mutex + границы prepare/render), а не поточечная.
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
