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
    /// GPU bytes uploaded this frame. Currently tallies texture uploads
    /// (sum of decoded mip bytes handed to sg); per-frame sg.updateBuffer
    /// ranges (instance/morph/particle/trail buffers) are not yet tallied —
    /// those flush paths live in subsystems outside this change.
    uploaded_bytes_frame: u64 = 0,
    /// Wall-clock phase timings in milliseconds (see header for writers).
    update_ms: f32 = 0,
    prepare_ms: f32 = 0,
    shadow_ms: f32 = 0,
    main_ms: f32 = 0,
    post_ms: f32 = 0,
};
