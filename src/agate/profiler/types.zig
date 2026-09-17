const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

// Pure profiler data types. Leaf module: imports nothing of ours
// (only std + sokol), so report.zig and profiler.zig can both
// depend on it without creating an import cycle.

/// Record of a single frame's timing and counters.
///
/// `dt_s` / `frame_interval_ms` / `fps` describe the observed wall-clock
/// interval between consecutive `recordFrame` calls (real frame pacing).
/// `total_frame_ms` and the phase fields are CPU-submit times measured with
/// CPU timers around submit calls — NOT GPU execution time and NOT wall time.
pub const FrameRecord = struct {
    frame_index: u64 = 0,
    timestamp_us: u64 = 0,
    dt_s: f32 = 0,
    /// Observed wall-clock FPS (= 1/dt_s when dt_s is valid).
    fps: f32 = 0,
    /// Wall-clock frame interval in ms (= dt_s * 1000). Real pacing.
    frame_interval_ms: f32 = 0,
    /// Sum of CPU phase timings (update+prepare+shadow+main+post).
    /// CPU-submit time, not wall time and not GPU time.
    total_frame_ms: f32 = 0,

    // Phase breakdown (ms)
    update_ms: f32 = 0,
    prepare_ms: f32 = 0,
    shadow_ms: f32 = 0,
    main_ms: f32 = 0,
    post_ms: f32 = 0,

    // Rendering counters
    draw_calls: u32 = 0,
    triangles: u32 = 0,
    pipeline_switches: u32 = 0,
    rendered_meshes: u32 = 0,
    culled_objects: u32 = 0,
    uploaded_textures: u32 = 0,
    uploaded_bytes: usize = 0,
};

/// Information about a single texture in GPU memory.
pub const TextureMemoryRecord = struct {
    name: []const u8,
    width: u32,
    height: u32,
    num_mips: u32,
    format: sg.PixelFormat,
    is_cube: bool,
    gpu_bytes: usize,
};

/// Information about a single mesh in CPU/GPU memory.
pub const MeshMemoryRecord = struct {
    name: []const u8,
    vertex_count: u32,
    index_count: u32,
    index_type: sg.IndexType,
    gpu_bytes: usize,
    cpu_bytes: usize,
};

/// Information about an offscreen render target.
pub const RenderTargetRecord = struct {
    name: []const u8,
    width: u32,
    height: u32,
    format: sg.PixelFormat,
    samples: u32,
    gpu_bytes: usize,
};

/// Complete snapshot of engine memory (CPU object mirrors & GPU VRAM).
pub const MemorySnapshot = struct {
    timestamp_us: u64 = 0,

    // Aggregate VRAM
    total_gpu_vram_bytes: usize = 0,
    textures_vram_bytes: usize = 0,
    meshes_vram_bytes: usize = 0,
    render_targets_vram_bytes: usize = 0,

    // Aggregate CPU memory
    total_cpu_mesh_bytes: usize = 0,

    // Scene inventory
    mesh_count: usize = 0,
    material_count: usize = 0,
    pbr_material_count: usize = 0,
    light_count: usize = 0,
    camera_count: usize = 0,
    particle_system_count: usize = 0,

    // Itemized tables
    textures: []TextureMemoryRecord = &.{},
    meshes: []MeshMemoryRecord = &.{},
    render_targets: []RenderTargetRecord = &.{},

    pub fn deinit(self: *MemorySnapshot, allocator: std.mem.Allocator) void {
        for (self.textures) |tex| {
            allocator.free(tex.name);
        }
        if (self.textures.len > 0) allocator.free(self.textures);

        for (self.meshes) |m| {
            allocator.free(m.name);
        }
        if (self.meshes.len > 0) allocator.free(self.meshes);

        for (self.render_targets) |rt| {
            allocator.free(rt.name);
        }
        if (self.render_targets.len > 0) allocator.free(self.render_targets);

        self.* = .{};
    }
};

/// Aggregate statistical summary of a recorded profiling session.
///
/// Fields named `avg/p50/p95/p99/min/max_frame_ms`, `avg_fps`,
/// `fps_1pct/01pct_low`, `total_time_ms` and `hitches_over_*ms` are derived
/// from `total_frame_ms` (the CPU-submit sum) — NOT wall time.
/// Fields prefixed with `observed_` / `interval_` / `avg_interval` / `p*_interval`
/// / `max_interval` are derived from the wall-clock frame intervals
/// (`frame_interval_ms`) and describe real frame pacing.
pub const SessionSummary = struct {
    frame_count: usize = 0,
    /// Sum of CPU-submit frame times (not wall duration).
    total_time_ms: f64 = 0,

    // FPS metrics derived from the CPU-submit sum (not wall pacing).
    avg_fps: f32 = 0,
    fps_1pct_low: f32 = 0,
    fps_01pct_low: f32 = 0,

    // CPU-submit frame times (ms), not wall intervals.
    min_frame_ms: f32 = 0,
    avg_frame_ms: f32 = 0,
    max_frame_ms: f32 = 0,
    p50_frame_ms: f32 = 0,
    p95_frame_ms: f32 = 0,
    p99_frame_ms: f32 = 0,

    // Phase averages (ms)
    avg_update_ms: f32 = 0,
    avg_prepare_ms: f32 = 0,
    avg_shadow_ms: f32 = 0,
    avg_main_ms: f32 = 0,
    avg_post_ms: f32 = 0,

    // Counter stats
    avg_draw_calls: u32 = 0,
    max_draw_calls: u32 = 0,
    avg_triangles: u32 = 0,
    max_triangles: u32 = 0,
    avg_pipeline_switches: u32 = 0,
    total_uploaded_bytes: usize = 0,

    // Hitches / dropped frames derived from the CPU-submit sum (not wall pacing).
    hitches_over_16ms: u32 = 0, // CPU-submit time > 16.67ms
    hitches_over_33ms: u32 = 0, // CPU-submit time > 33.33ms
    hitches_over_50ms: u32 = 0, // CPU-submit time > 50ms

    // Wall-clock frame pacing derived from frame_interval_ms (real pacing).
    // Frames with dt_s <= 0 are skipped; all zeros when no valid intervals.
    avg_interval_ms: f32 = 0,
    p50_interval_ms: f32 = 0,
    p99_interval_ms: f32 = 0,
    max_interval_ms: f32 = 0,
    observed_avg_fps: f32 = 0,
    observed_fps_1pct_low: f32 = 0,
    observed_fps_01pct_low: f32 = 0,
    interval_hitches_over_16ms: u32 = 0, // Wall interval > 16.67ms (< 60 FPS)
    interval_hitches_over_33ms: u32 = 0, // Wall interval > 33.33ms (< 30 FPS)
    interval_hitches_over_50ms: u32 = 0, // Wall interval > 50ms (< 20 FPS)
};

pub const DiagnosticSeverity = enum {
    good,
    info,
    warning,
    critical,
};

/// An automated diagnostic finding pointing out bottlenecks and recommendations.
pub const DiagnosticFinding = struct {
    severity: DiagnosticSeverity,
    title: []const u8,
    details: []const u8,
    recommendation: []const u8,

    pub fn deinit(self: *DiagnosticFinding, allocator: std.mem.Allocator) void {
        allocator.free(self.title);
        allocator.free(self.details);
        allocator.free(self.recommendation);
    }
};

/// Attribution of the dominant CPU-submit phase in a frame (not GPU time).
pub const PhaseCulprit = struct {
    name: []const u8,
    ms: f32,
    percent: f32,
};

pub fn dominantPhase(record: FrameRecord) PhaseCulprit {
    const total = @max(0.0001, record.total_frame_ms);
    var max_name: []const u8 = "Main Pass";
    var max_ms: f32 = record.main_ms;

    if (record.update_ms > max_ms) {
        max_ms = record.update_ms;
        max_name = "Update";
    }
    if (record.prepare_ms > max_ms) {
        max_ms = record.prepare_ms;
        max_name = "Prepare";
    }
    if (record.shadow_ms > max_ms) {
        max_ms = record.shadow_ms;
        max_name = "Shadow Pass";
    }
    if (record.post_ms > max_ms) {
        max_ms = record.post_ms;
        max_name = "PostFX";
    }
    return .{
        .name = max_name,
        .ms = max_ms,
        .percent = (max_ms / total) * 100.0,
    };
}
