// Per-frame render statistics. Lives in its own leaf module so the scene
// subsystems (render queue, draw helpers) can share a *SceneStats without
// importing scene.zig (no import cycles); Scene re-exports the type under
// its historical name.
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
};
