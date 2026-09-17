const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const stats_mod = @import("scene/stats.zig");
const SceneStats = stats_mod.SceneStats;
const texture_mod = @import("texture.zig");
const Texture = texture_mod.Texture;
const CubeTexture = texture_mod.CubeTexture;
const pixelFormatBytes = texture_mod.pixelFormatBytes;
const mesh_mod = @import("mesh.zig");
const Mesh = mesh_mod.Mesh;
const scene_mod = @import("scene.zig");
const Scene = scene_mod.Scene;
const SHADOW_ATLAS_SIZE = @import("passes/shadow_pass.zig").SHADOW_ATLAS_SIZE;
const SPOT_SHADOW_MAP_WIDTH = @import("passes/shadow_pass.zig").SPOT_SHADOW_MAP_WIDTH;
const SPOT_SHADOW_MAP_HEIGHT = @import("passes/shadow_pass.zig").SPOT_SHADOW_MAP_HEIGHT;

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

/// Flight recorder and profiler for performance and memory diagnostics.
pub const Profiler = struct {
    allocator: std.mem.Allocator,
    is_recording: bool = false,
    frames: std.ArrayListUnmanaged(FrameRecord) = .empty,
    max_frames: usize = 3600, // 60 seconds @ 60 FPS default
    start_time_ticks: u64 = 0,
    last_frame_ticks: u64 = 0,
    last_memory_snapshot: ?MemorySnapshot = null,

    pub fn init(allocator: std.mem.Allocator) Profiler {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Profiler) void {
        self.frames.deinit(self.allocator);
        if (self.last_memory_snapshot) |*snap| {
            snap.deinit(self.allocator);
            self.last_memory_snapshot = null;
        }
        self.* = undefined;
    }

    /// Sets the maximum number of frames retained in the recording ring buffer.
    pub fn setMaxFrames(self: *Profiler, max: usize) void {
        self.max_frames = @max(1, max);
    }

    /// Begins recording frames.
    pub fn start(self: *Profiler) void {
        sokol.time.setup();
        self.is_recording = true;
        self.start_time_ticks = sokol.time.now();
        self.last_frame_ticks = self.start_time_ticks;
    }

    /// Stops recording frames.
    pub fn stop(self: *Profiler) void {
        self.is_recording = false;
    }

    /// Clears recorded frame history while maintaining configuration.
    pub fn reset(self: *Profiler) void {
        self.frames.clearRetainingCapacity();
        self.start_time_ticks = 0;
        self.last_frame_ticks = 0;
        if (self.last_memory_snapshot) |*old| old.deinit(self.allocator);
        self.last_memory_snapshot = null;
    }

    /// Returns true if currently recording frames.
    pub fn isRecording(self: *const Profiler) bool {
        return self.is_recording;
    }

    /// Records metrics for the completed frame. Called at the end of Scene.render().
    pub fn recordFrame(self: *Profiler, frame_id: u64, stats: *const SceneStats) void {
        if (!self.is_recording) return;

        const now = sokol.time.now();
        const dt_s: f32 = if (self.last_frame_ticks > 0)
            @floatCast(sokol.time.sec(sokol.time.diff(now, self.last_frame_ticks)))
        else
            0.0166;
        self.last_frame_ticks = now;

        const total_ms = stats.update_ms + stats.prepare_ms + stats.shadow_ms + stats.main_ms + stats.post_ms;
        // Observed wall-clock pacing: fps derives from the real frame interval.
        // total_ms is the CPU-submit sum (timers around sg submit calls), not wall/GPU time.
        const frame_interval_ms: f32 = dt_s * 1000.0;
        const fps: f32 = if (dt_s > 0.0001) 1.0 / dt_s else (if (total_ms > 0.001) 1000.0 / total_ms else 60.0);
        const rel_us: u64 = if (self.start_time_ticks > 0)
            @intFromFloat(sokol.time.us(sokol.time.diff(now, self.start_time_ticks)))
        else
            0;

        const rec: FrameRecord = .{
            .frame_index = frame_id,
            .timestamp_us = rel_us,
            .dt_s = dt_s,
            .fps = fps,
            .frame_interval_ms = frame_interval_ms,
            .total_frame_ms = total_ms,
            .update_ms = stats.update_ms,
            .prepare_ms = stats.prepare_ms,
            .shadow_ms = stats.shadow_ms,
            .main_ms = stats.main_ms,
            .post_ms = stats.post_ms,
            .draw_calls = stats.draw_calls,
            .triangles = stats.triangles,
            .pipeline_switches = stats.pipeline_switches,
            .rendered_meshes = stats.rendered_meshes,
            .culled_objects = stats.culled_meshes + stats.occluded_meshes,
            .uploaded_textures = stats.uploaded_textures_frame,
            .uploaded_bytes = @intCast(stats.uploaded_bytes_frame),
        };

        if (self.frames.items.len >= self.max_frames) {
            _ = self.frames.orderedRemove(0);
        }
        self.frames.append(self.allocator, rec) catch return;
    }

    fn isCleanAscii(str: []const u8) bool {
        if (str.len == 0) return false;
        for (str) |c| {
            if (!std.ascii.isPrint(c)) return false;
        }
        return true;
    }

    /// Captures a complete snapshot of CPU memory and GPU VRAM allocations in the scene.
    pub fn captureMemorySnapshot(self: *Profiler, scene: *const Scene) !*const MemorySnapshot {
        const dir_count: usize = if (scene.lights.directional != null) 1 else 0;
        var snap: MemorySnapshot = .{
            .timestamp_us = if (self.start_time_ticks > 0)
                @intFromFloat(sokol.time.us(sokol.time.diff(sokol.time.now(), self.start_time_ticks)))
            else
                0,
            .mesh_count = scene.meshes.items.len,
            .material_count = scene.materials.items.len,
            .pbr_material_count = scene.pbr_materials.items.len,
            .light_count = scene.lights.point_lights.items.len + scene.lights.spot_lights.items.len + dir_count + 1,
            .camera_count = scene.cameras.items.len,
            .particle_system_count = scene.particles.systems.items.len,
        };

        // 1. Textures
        var textures_list = std.ArrayListUnmanaged(TextureMemoryRecord).empty;
        errdefer {
            for (textures_list.items) |t| self.allocator.free(t.name);
            textures_list.deinit(self.allocator);
        }

        // Keep track of visited GPU image IDs so shared textures aren't double-counted
        var seen_images = std.ArrayListUnmanaged(u32).empty;
        defer seen_images.deinit(self.allocator);

        // Helper to record a 2D texture
        const addTexture = struct {
            fn run(
                allocator: std.mem.Allocator,
                list: *std.ArrayListUnmanaged(TextureMemoryRecord),
                seen: *std.ArrayListUnmanaged(u32),
                tex: ?Texture,
                name: []const u8,
                vram_sum: *usize,
            ) !void {
                if (tex) |t| {
                    if (!sg.isvalid() or t.image.id == 0) return;
                    for (seen.items) |id| {
                        if (id == t.image.id) return;
                    }
                    try seen.append(allocator, t.image.id);

                    const bytes = t.getGpuMemoryBytes();
                    vram_sum.* += bytes;
                    const owned_name = try allocator.dupe(u8, name);
                    try list.append(allocator, .{
                        .name = owned_name,
                        .width = t.width,
                        .height = t.height,
                        .num_mips = t.num_mipmaps,
                        .format = t.format,
                        .is_cube = false,
                        .gpu_bytes = bytes,
                    });
                }
            }
        }.run;

        // Helper to record a Cube texture
        const addCubeTexture = struct {
            fn run(
                allocator: std.mem.Allocator,
                list: *std.ArrayListUnmanaged(TextureMemoryRecord),
                seen: *std.ArrayListUnmanaged(u32),
                cube: ?CubeTexture,
                name: []const u8,
                vram_sum: *usize,
            ) !void {
                if (cube) |c| {
                    if (!sg.isvalid() or c.image.id == 0) return;
                    for (seen.items) |id| {
                        if (id == c.image.id) return;
                    }
                    try seen.append(allocator, c.image.id);

                    const bytes = c.getGpuMemoryBytes();
                    vram_sum.* += bytes;
                    const owned_name = try allocator.dupe(u8, name);
                    try list.append(allocator, .{
                        .name = owned_name,
                        .width = c.size,
                        .height = c.size,
                        .num_mips = c.num_mipmaps,
                        .format = c.format,
                        .is_cube = true,
                        .gpu_bytes = bytes,
                    });
                }
            }
        }.run;

        // Default built-in textures
        try addTexture(self.allocator, &textures_list, &seen_images, scene.default_white_texture, "[builtin] White 1x1", &snap.textures_vram_bytes);
        try addTexture(self.allocator, &textures_list, &seen_images, scene.default_normal_texture, "[builtin] Normal 1x1", &snap.textures_vram_bytes);
        try addCubeTexture(self.allocator, &textures_list, &seen_images, scene.default_cube_texture, "[builtin] Cube 1x1", &snap.textures_vram_bytes);

        // Skybox texture
        if (scene.sky.texture) |sky_cube| {
            try addCubeTexture(self.allocator, &textures_list, &seen_images, sky_cube, "Skybox Environment Cubemap", &snap.textures_vram_bytes);
        }

        // Standard materials textures
        for (scene.materials.items, 0..) |mat, idx| {
            var name_buf: [64]u8 = undefined;
            const base_name = if (isCleanAscii(mat.name)) mat.name else std.fmt.bufPrint(&name_buf, "material_{d}", .{idx}) catch "material";
            if (mat.diffuse_texture) |t| {
                const name = try std.fmt.allocPrint(self.allocator, "{s} (diffuse)", .{base_name});
                defer self.allocator.free(name);
                try addTexture(self.allocator, &textures_list, &seen_images, t, name, &snap.textures_vram_bytes);
            }
        }

        // PBR materials textures
        for (scene.pbr_materials.items, 0..) |mat, idx| {
            var name_buf: [64]u8 = undefined;
            const base_name = if (isCleanAscii(mat.name)) mat.name else std.fmt.bufPrint(&name_buf, "pbr_material_{d}", .{idx}) catch "pbr_material";
            if (mat.albedo_texture) |t| {
                const name = try std.fmt.allocPrint(self.allocator, "{s} (albedo)", .{base_name});
                defer self.allocator.free(name);
                try addTexture(self.allocator, &textures_list, &seen_images, t, name, &snap.textures_vram_bytes);
            }
            if (mat.normal_texture) |t| {
                const name = try std.fmt.allocPrint(self.allocator, "{s} (normal)", .{base_name});
                defer self.allocator.free(name);
                try addTexture(self.allocator, &textures_list, &seen_images, t, name, &snap.textures_vram_bytes);
            }
            if (mat.metallic_roughness_texture) |t| {
                const name = try std.fmt.allocPrint(self.allocator, "{s} (metallic-roughness)", .{base_name});
                defer self.allocator.free(name);
                try addTexture(self.allocator, &textures_list, &seen_images, t, name, &snap.textures_vram_bytes);
            }
            if (mat.emissive_texture) |t| {
                const name = try std.fmt.allocPrint(self.allocator, "{s} (emissive)", .{base_name});
                defer self.allocator.free(name);
                try addTexture(self.allocator, &textures_list, &seen_images, t, name, &snap.textures_vram_bytes);
            }
            if (mat.occlusion_texture) |t| {
                const name = try std.fmt.allocPrint(self.allocator, "{s} (occlusion)", .{base_name});
                defer self.allocator.free(name);
                try addTexture(self.allocator, &textures_list, &seen_images, t, name, &snap.textures_vram_bytes);
            }
            if (mat.environment_texture) |c| {
                const name = try std.fmt.allocPrint(self.allocator, "{s} (env cubemap)", .{base_name});
                defer self.allocator.free(name);
                try addCubeTexture(self.allocator, &textures_list, &seen_images, c, name, &snap.textures_vram_bytes);
            }
        }
        snap.textures = try textures_list.toOwnedSlice(self.allocator);

        // 2. Meshes
        var meshes_list = std.ArrayListUnmanaged(MeshMemoryRecord).empty;
        errdefer {
            for (meshes_list.items) |m| self.allocator.free(m.name);
            meshes_list.deinit(self.allocator);
        }

        for (scene.meshes.items, 0..) |mesh, idx| {
            const gpu_b = mesh.getGpuMemoryBytes();
            const cpu_b = mesh.getCpuMemoryBytes();
            snap.meshes_vram_bytes += gpu_b;
            snap.total_cpu_mesh_bytes += cpu_b;

            var name_buf: [64]u8 = undefined;
            const mesh_name = if (isCleanAscii(mesh.name)) mesh.name else std.fmt.bufPrint(&name_buf, "mesh_{d}", .{idx}) catch "mesh";
            const owned_name = try self.allocator.dupe(u8, mesh_name);
            try meshes_list.append(self.allocator, .{
                .name = owned_name,
                .vertex_count = mesh.vertex_count,
                .index_count = mesh.index_count,
                .index_type = mesh.index_type,
                .gpu_bytes = gpu_b,
                .cpu_bytes = cpu_b,
            });
        }
        snap.meshes = try meshes_list.toOwnedSlice(self.allocator);

        // 3. Render Targets (only when valid GPU context exists)
        var rt_list = std.ArrayListUnmanaged(RenderTargetRecord).empty;
        errdefer {
            for (rt_list.items) |rt| self.allocator.free(rt.name);
            rt_list.deinit(self.allocator);
        }

        if (sg.isvalid()) {
            const pp = &scene.postfx.postprocess_pass;
            if (pp.width > 0 and pp.height > 0) {
                const w: u32 = @intCast(pp.width);
                const h: u32 = @intCast(pp.height);
                const samples: u32 = @intCast(@max(1, pp.sample_count));
                const color_bpp: usize = 4;
                const depth_bpp: usize = 4;

                // Offscreen Color
                const color_bytes = @as(usize, w) * h * color_bpp * samples;
                snap.render_targets_vram_bytes += color_bytes;
                try rt_list.append(self.allocator, .{
                    .name = try self.allocator.dupe(u8, "Main Offscreen Color Target"),
                    .width = w,
                    .height = h,
                    .format = .RGBA8,
                    .samples = samples,
                    .gpu_bytes = color_bytes,
                });

                // Offscreen Depth
                const depth_bytes = @as(usize, w) * h * depth_bpp * samples;
                snap.render_targets_vram_bytes += depth_bytes;
                try rt_list.append(self.allocator, .{
                    .name = try self.allocator.dupe(u8, "Main Offscreen Depth Target"),
                    .width = w,
                    .height = h,
                    .format = .DEPTH,
                    .samples = samples,
                    .gpu_bytes = depth_bytes,
                });

                // Resolve Color if MSAA
                if (samples > 1) {
                    const resolve_bytes = @as(usize, w) * h * color_bpp;
                    snap.render_targets_vram_bytes += resolve_bytes;
                    try rt_list.append(self.allocator, .{
                        .name = try self.allocator.dupe(u8, "MSAA Resolve Target"),
                        .width = w,
                        .height = h,
                        .format = .RGBA8,
                        .samples = 1,
                        .gpu_bytes = resolve_bytes,
                    });
                }
            }

            // SSAO Target
            const ssao = &scene.postfx.ssao_pass;
            if (ssao.width > 0 and ssao.height > 0) {
                const ssao_bytes = @as(usize, @intCast(ssao.width)) * @as(usize, @intCast(ssao.height)) * 4 * 2; // raw + blur
                snap.render_targets_vram_bytes += ssao_bytes;
                try rt_list.append(self.allocator, .{
                    .name = try self.allocator.dupe(u8, "SSAO Targets (Raw + Blur)"),
                    .width = @intCast(ssao.width),
                    .height = @intCast(ssao.height),
                    .format = .RGBA8,
                    .samples = 1,
                    .gpu_bytes = ssao_bytes,
                });
            }

            // Shadow Atlas
            const shadow_atlas_bytes = @as(usize, SHADOW_ATLAS_SIZE) * SHADOW_ATLAS_SIZE * 4;
            snap.render_targets_vram_bytes += shadow_atlas_bytes;
            try rt_list.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, "CSM Directional Shadow Atlas"),
                .width = SHADOW_ATLAS_SIZE,
                .height = SHADOW_ATLAS_SIZE,
                .format = .DEPTH,
                .samples = 1,
                .gpu_bytes = shadow_atlas_bytes,
            });

            // Spot Shadow Atlas
            const spot_bytes = @as(usize, SPOT_SHADOW_MAP_WIDTH) * SPOT_SHADOW_MAP_HEIGHT * 4;
            snap.render_targets_vram_bytes += spot_bytes;
            try rt_list.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, "Spot Light Shadow Map"),
                .width = SPOT_SHADOW_MAP_WIDTH,
                .height = SPOT_SHADOW_MAP_HEIGHT,
                .format = .DEPTH,
                .samples = 1,
                .gpu_bytes = spot_bytes,
            });
        }

        snap.render_targets = try rt_list.toOwnedSlice(self.allocator);
        snap.total_gpu_vram_bytes = snap.textures_vram_bytes + snap.meshes_vram_bytes + snap.render_targets_vram_bytes;

        // Store as last snapshot
        if (self.last_memory_snapshot) |*old| old.deinit(self.allocator);
        self.last_memory_snapshot = snap;

        return &self.last_memory_snapshot.?;
    }

    /// Computes statistical summary across all recorded frames.
    /// CPU-submit fields come from `total_frame_ms`; pacing fields come from
    /// wall-clock `frame_interval_ms` (frames with dt_s <= 0 are skipped).
    pub fn summarize(self: *const Profiler) SessionSummary {
        const n = self.frames.items.len;
        if (n == 0) return .{};

        var sum_time: f64 = 0;
        var sum_update: f64 = 0;
        var sum_prepare: f64 = 0;
        var sum_shadow: f64 = 0;
        var sum_main: f64 = 0;
        var sum_post: f64 = 0;

        var sum_draw_calls: u64 = 0;
        var max_draw_calls: u32 = 0;
        var sum_triangles: u64 = 0;
        var max_triangles: u32 = 0;
        var sum_switches: u64 = 0;
        var sum_uploaded_bytes: usize = 0;

        var hitches_16: u32 = 0;
        var hitches_33: u32 = 0;
        var hitches_50: u32 = 0;

        // Temporary slice to sort frame times for percentiles
        var times = self.allocator.alloc(f32, n) catch return .{};
        defer self.allocator.free(times);

        // Wall-clock intervals for real frame pacing (subset of frames).
        var intervals = self.allocator.alloc(f32, n) catch return .{};
        defer self.allocator.free(intervals);
        var interval_count: usize = 0;
        var sum_interval: f64 = 0;
        var max_interval: f32 = 0;
        var interval_hitches_16: u32 = 0;
        var interval_hitches_33: u32 = 0;
        var interval_hitches_50: u32 = 0;

        var min_ms: f32 = std.math.floatMax(f32);
        var max_ms: f32 = 0;

        for (self.frames.items, 0..) |frame, i| {
            const ms = frame.total_frame_ms;
            times[i] = ms;
            sum_time += ms;
            min_ms = @min(min_ms, ms);
            max_ms = @max(max_ms, ms);

            if (frame.dt_s > 0) {
                const iv_ms = frame.frame_interval_ms;
                intervals[interval_count] = iv_ms;
                interval_count += 1;
                sum_interval += iv_ms;
                max_interval = @max(max_interval, iv_ms);
                if (iv_ms > 50.0) interval_hitches_50 += 1;
                if (iv_ms > 33.33) interval_hitches_33 += 1;
                if (iv_ms > 16.67) interval_hitches_16 += 1;
            }

            sum_update += frame.update_ms;
            sum_prepare += frame.prepare_ms;
            sum_shadow += frame.shadow_ms;
            sum_main += frame.main_ms;
            sum_post += frame.post_ms;

            sum_draw_calls += frame.draw_calls;
            max_draw_calls = @max(max_draw_calls, frame.draw_calls);
            sum_triangles += frame.triangles;
            max_triangles = @max(max_triangles, frame.triangles);
            sum_switches += frame.pipeline_switches;
            sum_uploaded_bytes += frame.uploaded_bytes;

            if (ms > 50.0) {
                hitches_50 += 1;
            }
            if (ms > 33.33) {
                hitches_33 += 1;
            }
            if (ms > 16.67) {
                hitches_16 += 1;
            }
        }

        std.mem.sort(f32, times, {}, struct {
            fn lessThan(_: void, a: f32, b: f32) bool {
                return a < b;
            }
        }.lessThan);

        const avg_ms: f32 = @floatCast(sum_time / @as(f64, @floatFromInt(n)));
        const avg_fps: f32 = if (avg_ms > 0.001) 1000.0 / avg_ms else 0;

        // Percentiles
        const p50_idx = @min(n - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(n)) * 0.50)));
        const p95_idx = @min(n - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(n)) * 0.95)));
        const p99_idx = @min(n - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(n)) * 0.99)));

        // 1% low: average of worst 1% frames
        const count_1pct = @max(1, n / 100);
        var sum_1pct: f64 = 0;
        for (times[n - count_1pct ..]) |t| sum_1pct += t;
        const avg_1pct_ms: f32 = @floatCast(sum_1pct / @as(f64, @floatFromInt(count_1pct)));
        const fps_1pct_low: f32 = if (avg_1pct_ms > 0.001) 1000.0 / avg_1pct_ms else 0;

        // 0.1% low: average of worst 0.1% frames
        const count_01pct = @max(1, n / 1000);
        var sum_01pct: f64 = 0;
        for (times[n - count_01pct ..]) |t| sum_01pct += t;
        const avg_01pct_ms: f32 = @floatCast(sum_01pct / @as(f64, @floatFromInt(count_01pct)));
        const fps_01pct_low: f32 = if (avg_01pct_ms > 0.001) 1000.0 / avg_01pct_ms else 0;

        // Wall-clock pacing from observed frame intervals.
        var avg_interval_ms: f32 = 0;
        var p50_interval_ms: f32 = 0;
        var p99_interval_ms: f32 = 0;
        var observed_avg_fps: f32 = 0;
        var observed_fps_1pct_low: f32 = 0;
        var observed_fps_01pct_low: f32 = 0;
        if (interval_count > 0) {
            const valid = intervals[0..interval_count];
            std.mem.sort(f32, valid, {}, struct {
                fn lessThan(_: void, a: f32, b: f32) bool {
                    return a < b;
                }
            }.lessThan);
            const m = interval_count;
            avg_interval_ms = @floatCast(sum_interval / @as(f64, @floatFromInt(m)));
            observed_avg_fps = if (avg_interval_ms > 0.001) 1000.0 / avg_interval_ms else 0;
            const p50_iv_idx = @min(m - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(m)) * 0.50)));
            const p99_iv_idx = @min(m - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(m)) * 0.99)));
            p50_interval_ms = valid[p50_iv_idx];
            p99_interval_ms = valid[p99_iv_idx];

            const count_iv_1pct = @max(1, m / 100);
            var sum_iv_1pct: f64 = 0;
            for (valid[m - count_iv_1pct ..]) |t| sum_iv_1pct += t;
            const avg_iv_1pct_ms: f32 = @floatCast(sum_iv_1pct / @as(f64, @floatFromInt(count_iv_1pct)));
            observed_fps_1pct_low = if (avg_iv_1pct_ms > 0.001) 1000.0 / avg_iv_1pct_ms else 0;

            const count_iv_01pct = @max(1, m / 1000);
            var sum_iv_01pct: f64 = 0;
            for (valid[m - count_iv_01pct ..]) |t| sum_iv_01pct += t;
            const avg_iv_01pct_ms: f32 = @floatCast(sum_iv_01pct / @as(f64, @floatFromInt(count_iv_01pct)));
            observed_fps_01pct_low = if (avg_iv_01pct_ms > 0.001) 1000.0 / avg_iv_01pct_ms else 0;
        }

        const nf = @as(f64, @floatFromInt(n));
        return .{
            .frame_count = n,
            .total_time_ms = sum_time,
            .avg_fps = avg_fps,
            .fps_1pct_low = fps_1pct_low,
            .fps_01pct_low = fps_01pct_low,
            .min_frame_ms = min_ms,
            .avg_frame_ms = avg_ms,
            .max_frame_ms = max_ms,
            .p50_frame_ms = times[p50_idx],
            .p95_frame_ms = times[p95_idx],
            .p99_frame_ms = times[p99_idx],
            .avg_update_ms = @floatCast(sum_update / nf),
            .avg_prepare_ms = @floatCast(sum_prepare / nf),
            .avg_shadow_ms = @floatCast(sum_shadow / nf),
            .avg_main_ms = @floatCast(sum_main / nf),
            .avg_post_ms = @floatCast(sum_post / nf),
            .avg_draw_calls = @intCast(sum_draw_calls / n),
            .max_draw_calls = max_draw_calls,
            .avg_triangles = @intCast(sum_triangles / n),
            .max_triangles = max_triangles,
            .avg_pipeline_switches = @intCast(sum_switches / n),
            .total_uploaded_bytes = sum_uploaded_bytes,
            .hitches_over_16ms = hitches_16,
            .hitches_over_33ms = hitches_33,
            .hitches_over_50ms = hitches_50,
            .avg_interval_ms = avg_interval_ms,
            .p50_interval_ms = p50_interval_ms,
            .p99_interval_ms = p99_interval_ms,
            .max_interval_ms = max_interval,
            .observed_avg_fps = observed_avg_fps,
            .observed_fps_1pct_low = observed_fps_1pct_low,
            .observed_fps_01pct_low = observed_fps_01pct_low,
            .interval_hitches_over_16ms = interval_hitches_16,
            .interval_hitches_over_33ms = interval_hitches_33,
            .interval_hitches_over_50ms = interval_hitches_50,
        };
    }

    /// Runs automated bottleneck diagnostics on recorded metrics and memory.
    pub fn analyze(self: *const Profiler, memory: ?*const MemorySnapshot, allocator: std.mem.Allocator) ![]DiagnosticFinding {
        var findings = std.ArrayListUnmanaged(DiagnosticFinding).empty;
        errdefer {
            for (findings.items) |*f| f.deinit(allocator);
            findings.deinit(allocator);
        }

        const summary = self.summarize();

        // 1. Frame Pacing / Target FPS Check (observed wall-clock intervals).
        // Phase attribution in section 2 stays CPU-submit based.
        if (summary.frame_count >= 5 and summary.max_interval_ms > 0) {
            if (summary.observed_avg_fps >= 58.0 and summary.p99_interval_ms <= 18.0 and summary.interval_hitches_over_33ms == 0) {
                try findings.append(allocator, .{
                    .severity = .good,
                    .title = try allocator.dupe(u8, "Стабильный кадровый темп 60+ FPS"),
                    .details = try std.fmt.allocPrint(allocator, "Наблюдаемый средний FPS (wall-интервал): {d:.1}, P99 интервала: {d:.1} мс, средний CPU-submit: {d:.2} мс. Просадок ниже 30 FPS не зафиксировано.", .{ summary.observed_avg_fps, summary.p99_interval_ms, summary.avg_frame_ms }),
                    .recommendation = try allocator.dupe(u8, "Производительность соответствует целевому бюджету времени кадра (16.6 мс)."),
                });
            } else if (summary.interval_hitches_over_33ms > 0) {
                const sev: DiagnosticSeverity = if (summary.interval_hitches_over_33ms > summary.frame_count / 10 or summary.max_interval_ms > 50.0) .critical else .warning;
                try findings.append(allocator, .{
                    .severity = sev,
                    .title = try allocator.dupe(u8, "Просадки кадровой частоты ниже 30 FPS"),
                    .details = try std.fmt.allocPrint(allocator, "Зафиксировано {d} wall-интервалов > 33.3 мс (худший интервал: {d:.1} мс, 1% Low (wall): {d:.1} FPS). Пиковый CPU-submit кадра: {d:.1} мс.", .{ summary.interval_hitches_over_33ms, summary.max_interval_ms, summary.observed_fps_1pct_low, summary.max_frame_ms }),
                    .recommendation = try allocator.dupe(u8, "Изучите таблицу Spike Frames (время CPU-submit) для определения виновной фазы и исключите блокирующие операции на главном потоке."),
                });
            } else if (summary.interval_hitches_over_16ms > 0) {
                try findings.append(allocator, .{
                    .severity = .info,
                    .title = try allocator.dupe(u8, "Периодические просадки ниже 60 FPS"),
                    .details = try std.fmt.allocPrint(allocator, "Зафиксировано {d} wall-интервалов длительностью от 16.7 до 33.3 мс.", .{summary.interval_hitches_over_16ms}),
                    .recommendation = try allocator.dupe(u8, "Оптимизируйте наиболее тяжелые CPU-фазы (Main pass, Shadows) для достижения чистых 60 FPS."),
                });
            }
        }

        // 2. Worst Phase Attribution on Spikes (CPU-submit times, not GPU time)
        if (self.frames.items.len > 0) {
            var worst_frame = self.frames.items[0];
            for (self.frames.items) |f| {
                if (f.total_frame_ms > worst_frame.total_frame_ms) worst_frame = f;
            }

            if (worst_frame.total_frame_ms > 20.0) {
                const culprit = dominantPhase(worst_frame);
                if (std.mem.eql(u8, culprit.name, "Main Pass") and culprit.percent > 45.0) {
                    try findings.append(allocator, .{
                        .severity = .warning,
                        .title = try allocator.dupe(u8, "Узкое горлышко: Main Render Pass"),
                        .details = try std.fmt.allocPrint(allocator, "В пиковом кадре #{d} (CPU-submit {d:.1} мс) фаза Main Pass (CPU submit) заняла {d:.1} мс ({d:.1}% всего кадра).", .{ worst_frame.frame_index, worst_frame.total_frame_ms, culprit.ms, culprit.percent }),
                        .recommendation = try allocator.dupe(u8, "Сократите количество вызовов отрисовки через InstancedMesh, объедините меши с одинаковыми материалами и включите Occlusion Culling."),
                    });
                } else if (std.mem.eql(u8, culprit.name, "Shadow Pass") and culprit.percent > 35.0) {
                    try findings.append(allocator, .{
                        .severity = .warning,
                        .title = try allocator.dupe(u8, "Узкое горлышко: CSM Shadow Pass"),
                        .details = try std.fmt.allocPrint(allocator, "В пиковом кадре #{d} рендеринг теней (CPU submit) занял {d:.1} мс ({d:.1}% кадра).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                        .recommendation = try allocator.dupe(u8, "Отключите cast_shadows для мелких мешей, уменьшите дистанцию теневых каскадов или отключите тени для точечных источников."),
                    });
                } else if (std.mem.eql(u8, culprit.name, "Update") and culprit.percent > 40.0) {
                    try findings.append(allocator, .{
                        .severity = .warning,
                        .title = try allocator.dupe(u8, "Узкое горлышко: CPU Update / Скрипты"),
                        .details = try std.fmt.allocPrint(allocator, "В пиковом кадре #{d} обновление логики заняло {d:.1} мс ({d:.1}% CPU-submit кадра).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                        .recommendation = try allocator.dupe(u8, "Оптимизируйте анимации, физическую симуляцию или перенесите тяжелые расчеты в фоновый пул jobs.TaskRunner."),
                    });
                } else if (std.mem.eql(u8, culprit.name, "PostFX") and culprit.percent > 40.0) {
                    try findings.append(allocator, .{
                        .severity = .warning,
                        .title = try allocator.dupe(u8, "Узкое горлышко: PostFX Stack"),
                        .details = try std.fmt.allocPrint(allocator, "В пиковом кадре #{d} пост-обработка (CPU submit) заняла {d:.1} мс ({d:.1}% кадра).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                        .recommendation = try allocator.dupe(u8, "Проверьте настройки SSAO (уменьшите sample count), снизьте bloom pyramid mips или уменьшите разрешение буфера."),
                    });
                }

                if (worst_frame.uploaded_bytes > 4 * 1024 * 1024) {
                    try findings.append(allocator, .{
                        .severity = .warning,
                        .title = try allocator.dupe(u8, "Задержка из-за загрузки текстур на GPU"),
                        .details = try std.fmt.allocPrint(allocator, "В кадре #{d} произошла загрузка {d:.2} МБ текстур на видеокарту.", .{ worst_frame.frame_index, @as(f32, @floatFromInt(worst_frame.uploaded_bytes)) / (1024.0 * 1024.0) }),
                        .recommendation = try allocator.dupe(u8, "Используйте асинхронную очередь Scene.uploads (UploadQueue) с лимитом байт на кадр (max_bytes_per_frame)."),
                    });
                }
            }
        }

        // 3. Draw Calls & Pipeline Switches
        if (summary.frame_count > 0) {
            if (summary.avg_draw_calls > 1000) {
                try findings.append(allocator, .{
                    .severity = .critical,
                    .title = try allocator.dupe(u8, "Критическое количество Draw Calls (> 1000)"),
                    .details = try std.fmt.allocPrint(allocator, "В среднем {d} draw calls за кадр (максимум {d}).", .{ summary.avg_draw_calls, summary.max_draw_calls }),
                    .recommendation = try allocator.dupe(u8, "Используйте InstancedMesh для повторяющихся объектов, объединяйте статические меши и проверьте фрустум-куллинг."),
                });
            } else if (summary.avg_draw_calls > 400) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Повышенное количество Draw Calls (> 400)"),
                    .details = try std.fmt.allocPrint(allocator, "В среднем {d} draw calls за кадр (максимум {d}).", .{ summary.avg_draw_calls, summary.max_draw_calls }),
                    .recommendation = try allocator.dupe(u8, "Рекомендуется батчинг и инстансинг для снижения нагрузки на CPU драйвера."),
                });
            }

            if (summary.avg_pipeline_switches > 120) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Частые переключения шейдеров и пайплайнов"),
                    .details = try std.fmt.allocPrint(allocator, "В среднем {d} переключений пайплайнов за кадр.", .{summary.avg_pipeline_switches}),
                    .recommendation = try allocator.dupe(u8, "Сортируйте меши по материалам и шейдерам перед отрисовкой для минимизации смены состояний GPU."),
                });
            }
        }

        // 4. Memory & Assets Checks (if snapshot available)
        if (memory) |mem| {
            // High total VRAM check
            if (mem.total_gpu_vram_bytes > 512 * 1024 * 1024) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Высокое потребление видеопамяти (> 512 МБ)"),
                    .details = try std.fmt.allocPrint(allocator, "Суммарно VRAM: {d:.1} МБ (текстуры: {d:.1} МБ, меши: {d:.1} МБ, буферы кадров: {d:.1} МБ).", .{
                        @as(f32, @floatFromInt(mem.total_gpu_vram_bytes)) / (1024.0 * 1024.0),
                        @as(f32, @floatFromInt(mem.textures_vram_bytes)) / (1024.0 * 1024.0),
                        @as(f32, @floatFromInt(mem.meshes_vram_bytes)) / (1024.0 * 1024.0),
                        @as(f32, @floatFromInt(mem.render_targets_vram_bytes)) / (1024.0 * 1024.0),
                    }),
                    .recommendation = try allocator.dupe(u8, "Примените KTX2 GPU-сжатие текстур (Basis Universal / BCn / ETC2) для снижения объема памяти на 70-80%."),
                });
            }

            // Heavy uncompressed textures
            for (mem.textures) |tex| {
                if (tex.width >= 2048 and tex.height >= 2048 and tex.gpu_bytes >= 16 * 1024 * 1024 and tex.format == .RGBA8) {
                    try findings.append(allocator, .{
                        .severity = .warning,
                        .title = try allocator.dupe(u8, "Тяжелая несжатая 2K+ текстура"),
                        .details = try std.fmt.allocPrint(allocator, "Текстура '{s}' ({d}x{d}, RGBA8) занимает {d:.1} МБ видеопамяти.", .{ tex.name, tex.width, tex.height, @as(f32, @floatFromInt(tex.gpu_bytes)) / (1024.0 * 1024.0) }),
                        .recommendation = try allocator.dupe(u8, "Сконвертируйте текстуру в формат KTX2 с блочным сжатием (BC7 / BC1 / ETC1S) или уменьшите разрешение до 1024x1024."),
                    });
                    break;
                }
            }

            // Missing mipmaps
            for (mem.textures) |tex| {
                if (tex.width >= 512 and tex.height >= 512 and tex.num_mips <= 1 and !tex.is_cube) {
                    try findings.append(allocator, .{
                        .severity = .warning,
                        .title = try allocator.dupe(u8, "Отсутствуют mip-уровни у большой текстуры"),
                        .details = try std.fmt.allocPrint(allocator, "Текстура '{s}' ({d}x{d}) загружена без цепочки mipmaps.", .{ tex.name, tex.width, tex.height }),
                        .recommendation = try allocator.dupe(u8, "Включите генерацию mipmaps (options.mipmaps = true) для устранения мерцания при удалении и повышения попаданий в кэш GPU."),
                    });
                    break;
                }
            }

            // Inefficient 32-bit indices
            var wasteful_u32_count: usize = 0;
            for (mem.meshes) |m| {
                if (m.index_type == .UINT32 and m.vertex_count > 0 and m.vertex_count <= 65535) {
                    wasteful_u32_count += 1;
                }
            }
            if (wasteful_u32_count > 0) {
                try findings.append(allocator, .{
                    .severity = .info,
                    .title = try allocator.dupe(u8, "Неоптимальные 32-битные индексные буферы"),
                    .details = try std.fmt.allocPrint(allocator, "{d} мешей используют UINT32 индексы при менее чем 65536 вершинах.", .{wasteful_u32_count}),
                    .recommendation = try allocator.dupe(u8, "UINT16 индексы экономят 50% памяти индексов и снижают нагрузку на шину памяти GPU."),
                });
            }

            // Heavy dense meshes
            for (mem.meshes) |m| {
                if (m.vertex_count > 60000) {
                    try findings.append(allocator, .{
                        .severity = .warning,
                        .title = try allocator.dupe(u8, "Высокая плотность вершин в меше (> 60k)"),
                        .details = try std.fmt.allocPrint(allocator, "Меш '{s}' содержит {d} вершин и {d} индексов ({d:.1} МБ VRAM).", .{ m.name, m.vertex_count, m.index_count, @as(f32, @floatFromInt(m.gpu_bytes)) / (1024.0 * 1024.0) }),
                        .recommendation = try allocator.dupe(u8, "Используйте meshopt simplification для генерации уровней детализации (LOD)."),
                    });
                    break;
                }
            }
        }

        // Default good condition if no warnings or criticals were produced
        var has_issues = false;
        for (findings.items) |f| {
            if (f.severity == .warning or f.severity == .critical) {
                has_issues = true;
                break;
            }
        }
        if (!has_issues and findings.items.len == 0) {
            try findings.append(allocator, .{
                .severity = .good,
                .title = try allocator.dupe(u8, "Все основные параметры в норме"),
                .details = try allocator.dupe(u8, "Критических задержек, перерасхода памяти или чрезмерного количества вызовов отрисовки не обнаружено."),
                .recommendation = try allocator.dupe(u8, "Текущая конфигурация сцены работает оптимально."),
            });
        }

        return findings.toOwnedSlice(allocator);
    }

    /// Helper to format byte values (e.g. "12.4 MB").
    pub fn formatBytes(allocator: std.mem.Allocator, bytes: usize) ![]u8 {
        if (bytes < 1024) {
            return std.fmt.allocPrint(allocator, "{d} B", .{bytes});
        } else if (bytes < 1024 * 1024) {
            return std.fmt.allocPrint(allocator, "{d:.1} KB", .{@as(f32, @floatFromInt(bytes)) / 1024.0});
        } else if (bytes < 1024 * 1024 * 1024) {
            return std.fmt.allocPrint(allocator, "{d:.2} MB", .{@as(f32, @floatFromInt(bytes)) / (1024.0 * 1024.0)});
        } else {
            return std.fmt.allocPrint(allocator, "{d:.2} GB", .{@as(f32, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0)});
        }
    }

    /// Generates a standalone, beautiful, dark-themed HTML report.
    pub fn generateReportHtml(self: *const Profiler, scene: ?*const Scene, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        // Ensure memory snapshot is captured if scene is provided
        const memory_ptr: ?*const MemorySnapshot = if (scene) |sc| blk: {
            var mut_prof = @constCast(self);
            break :blk mut_prof.captureMemorySnapshot(sc) catch null;
        } else (if (self.last_memory_snapshot) |*s| s else null);

        const summary = self.summarize();
        const findings = try self.analyze(memory_ptr, allocator);
        defer {
            for (findings) |*f| @constCast(f).deinit(allocator);
            allocator.free(findings);
        }

        // HTML Header & Embedded Dark Styles
        try buf.appendSlice(allocator,
            \\<!DOCTYPE html>
            \\<html lang="ru">
            \\<head>
            \\<meta charset="UTF-8">
            \\<meta name="viewport" content="width=device-width, initial-scale=1.0">
            \\<title>Agate Engine - Profile & Memory Report</title>
            \\<style>
            \\  :root {
            \\    --bg-main: #0b0f19;
            \\    --bg-card: #151e2e;
            \\    --bg-card-hover: #1c283c;
            \\    --border: #243247;
            \\    --border-subtle: #1a2536;
            \\    --text-main: #f1f5f9;
            \\    --text-muted: #94a3b8;
            \\    --color-update: #3b82f6;
            \\    --color-prepare: #06b6d4;
            \\    --color-shadow: #8b5cf6;
            \\    --color-main: #10b981;
            \\    --color-post: #f59e0b;
            \\    --color-good: #22c55e;
            \\    --color-warn: #eab308;
            \\    --color-crit: #ef4444;
            \\    --color-info: #38bdf8;
            \\  }
            \\  * { box-sizing: border-box; margin: 0; padding: 0; }
            \\  body {
            \\    background: var(--bg-main);
            \\    color: var(--text-main);
            \\    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            \\    line-height: 1.5;
            \\    padding: 24px;
            \\  }
            \\  .container { max-width: 1400px; margin: 0 auto; }
            \\  header {
            \\    display: flex;
            \\    justify-content: space-between;
            \\    align-items: center;
            \\    border-bottom: 1px solid var(--border);
            \\    padding-bottom: 20px;
            \\    margin-bottom: 24px;
            \\  }
            \\  .logo-title { display: flex; align-items: center; gap: 14px; }
            \\  .badge-logo {
            \\    background: linear-gradient(135deg, #6366f1, #a855f7);
            \\    padding: 6px 14px;
            \\    border-radius: 8px;
            \\    font-weight: 800;
            \\    font-size: 16px;
            \\    color: #fff;
            \\    letter-spacing: 1px;
            \\  }
            \\  h1 { font-size: 24px; font-weight: 700; color: #fff; }
            \\  .meta-sub { color: var(--text-muted); font-size: 13px; margin-top: 4px; }
            \\  .kpi-grid {
            \\    display: grid;
            \\    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            \\    gap: 16px;
            \\    margin-bottom: 28px;
            \\  }
            \\  .kpi-card {
            \\    background: var(--bg-card);
            \\    border: 1px solid var(--border);
            \\    border-radius: 12px;
            \\    padding: 16px;
            \\    transition: transform 0.15s ease, border-color 0.15s ease;
            \\  }
            \\  .kpi-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
            \\  .kpi-label { font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }
            \\  .kpi-val { font-size: 26px; font-weight: 800; margin-top: 6px; color: #fff; }
            \\  .kpi-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
            \\  .section-title {
            \\    font-size: 18px;
            \\    font-weight: 700;
            \\    margin-bottom: 16px;
            \\    display: flex;
            \\    align-items: center;
            \\    gap: 10px;
            \\  }
            \\  .diag-list { display: flex; flex-direction: column; gap: 12px; margin-bottom: 32px; }
            \\  .diag-card {
            \\    background: var(--bg-card);
            \\    border: 1px solid var(--border);
            \\    border-left: 5px solid var(--border);
            \\    border-radius: 10px;
            \\    padding: 16px 20px;
            \\  }
            \\  .diag-card.good { border-left-color: var(--color-good); }
            \\  .diag-card.info { border-left-color: var(--color-info); }
            \\  .diag-card.warning { border-left-color: var(--color-warn); }
            \\  .diag-card.critical { border-left-color: var(--color-crit); }
            \\  .diag-header { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
            \\  .diag-badge {
            \\    font-size: 11px;
            \\    font-weight: 700;
            \\    text-transform: uppercase;
            \\    padding: 3px 8px;
            \\    border-radius: 4px;
            \\  }
            \\  .diag-badge.good { background: #052e16; color: #86efac; border: 1px solid #16a34a; }
            \\  .diag-badge.info { background: #082f49; color: #7dd3fc; border: 1px solid #0284c7; }
            \\  .diag-badge.warning { background: #422006; color: #fde68a; border: 1px solid #d97706; }
            \\  .diag-badge.critical { background: #450a0a; color: #fca5a5; border: 1px solid #dc2626; }
            \\  .diag-title { font-size: 15px; font-weight: 600; color: #fff; }
            \\  .diag-desc { font-size: 13px; color: var(--text-muted); margin-bottom: 8px; }
            \\  .diag-rec {
            \\    font-size: 13px;
            \\    color: #a7f3d0;
            \\    background: #064e3b33;
            \\    border: 1px solid #065f46;
            \\    border-radius: 6px;
            \\    padding: 8px 12px;
            \\  }
            \\  .chart-box {
            \\    background: var(--bg-card);
            \\    border: 1px solid var(--border);
            \\    border-radius: 12px;
            \\    padding: 20px;
            \\    margin-bottom: 32px;
            \\  }
            \\  .chart-legend {
            \\    display: flex;
            \\    flex-wrap: wrap;
            \\    gap: 18px;
            \\    font-size: 12px;
            \\    margin-top: 14px;
            \\    justify-content: center;
            \\  }
            \\  .legend-item { display: flex; align-items: center; gap: 6px; color: var(--text-muted); }
            \\  .legend-dot { width: 12px; height: 12px; border-radius: 3px; }
            \\  .timeline-svg { width: 100%; height: 260px; overflow: visible; }
            \\  .card-table {
            \\    background: var(--bg-card);
            \\    border: 1px solid var(--border);
            \\    border-radius: 12px;
            \\    overflow: hidden;
            \\    margin-bottom: 32px;
            \\  }
            \\  table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
            \\  th {
            \\    background: #111927;
            \\    color: var(--text-muted);
            \\    font-weight: 600;
            \\    text-transform: uppercase;
            \\    font-size: 11px;
            \\    letter-spacing: 0.5px;
            \\    padding: 12px 16px;
            \\    border-bottom: 1px solid var(--border);
            \\  }
            \\  td { padding: 12px 16px; border-bottom: 1px solid var(--border-subtle); color: #cbd5e1; }
            \\  tr:last-child td { border-bottom: none; }
            \\  tr:hover td { background: var(--bg-card-hover); }
            \\  .num { text-align: right; font-variant-numeric: tabular-nums; }
            \\  .vram-bar {
            \\    display: flex;
            \\    height: 18px;
            \\    border-radius: 6px;
            \\    overflow: hidden;
            \\    background: #0f172a;
            \\    margin: 12px 0 20px 0;
            \\    border: 1px solid var(--border);
            \\  }
            \\  .vram-seg { height: 100%; transition: width 0.3s ease; }
            \\  footer {
            \\    text-align: center;
            \\    color: var(--text-muted);
            \\    font-size: 12px;
            \\    padding-top: 24px;
            \\    border-top: 1px solid var(--border);
            \\  }
            \\</style>
            \\</head>
            \\<body>
            \\<div class="container">
            \\<header>
            \\  <div class="logo-title">
            \\    <div class="badge-logo">AGATE</div>
            \\    <div>
            \\      <h1>Performance & Memory Profile Report</h1>
            \\      <div class="meta-sub">Engine Flight Recorder & Snapshot Inspector</div>
            \\    </div>
            \\  </div>
            \\  <div style="text-align: right;">
        );

        // Header metadata: wall-clock pacing is primary; CPU-submit sum is separate.
        const total_sec = summary.total_time_ms / 1000.0;
        const meta_str = try std.fmt.allocPrint(allocator,
            \\<div style="font-size: 14px; font-weight: 700; color: #fff;">{d} Кадров ({d:.2} с CPU-submit)</div>
            \\<div class="meta-sub">Avg {d:.1} FPS (wall) | P99 интервала: {d:.1} мс | CPU-submit avg: {d:.2} мс</div>
            \\</div></header>
        , .{ summary.frame_count, total_sec, summary.observed_avg_fps, summary.p99_interval_ms, summary.avg_frame_ms });
        defer allocator.free(meta_str);
        try buf.appendSlice(allocator, meta_str);

        // KPI Cards Grid
        const vram_total_str = if (memory_ptr) |m| try formatBytes(allocator, m.total_gpu_vram_bytes) else try allocator.dupe(u8, "N/A");
        defer allocator.free(vram_total_str);
        const cpu_mesh_str = if (memory_ptr) |m| try formatBytes(allocator, m.total_cpu_mesh_bytes) else try allocator.dupe(u8, "N/A");
        defer allocator.free(cpu_mesh_str);

        const kpi_html = try std.fmt.allocPrint(allocator,
            \\<div class="kpi-grid">
            \\  <div class="kpi-card">
            \\    <div class="kpi-label">Average FPS (wall)</div>
            \\    <div class="kpi-val" style="color: {s};">{d:.1}</div>
            \\    <div class="kpi-sub">Avg interval: {d:.2} ms | CPU-submit avg: {d:.2} ms</div>
            \\  </div>
            \\  <div class="kpi-card">
            \\    <div class="kpi-label">1% Low FPS (wall)</div>
            \\    <div class="kpi-val" style="color: {s};">{d:.1}</div>
            \\    <div class="kpi-sub">0.1% low (wall): {d:.1} FPS</div>
            \\  </div>
            \\  <div class="kpi-card">
            \\    <div class="kpi-label">P99 Frame Interval (wall)</div>
            \\    <div class="kpi-val">{d:.1} <span style="font-size: 14px; font-weight: normal; color: #94a3b8;">ms</span></div>
            \\    <div class="kpi-sub">P50 interval: {d:.1} ms | CPU-submit P99: {d:.1} ms</div>
            \\  </div>
            \\  <div class="kpi-card">
            \\    <div class="kpi-label">Average Draw Calls</div>
            \\    <div class="kpi-val">{d}</div>
            \\    <div class="kpi-sub">Max: {d} | Avg Tris: {d}</div>
            \\  </div>
            \\  <div class="kpi-card">
            \\    <div class="kpi-label">GPU VRAM Total</div>
            \\    <div class="kpi-val" style="color: #38bdf8;">{s}</div>
            \\    <div class="kpi-sub">Textures + Meshes + RTs</div>
            \\  </div>
            \\  <div class="kpi-card">
            \\    <div class="kpi-label">CPU Mesh Memory</div>
            \\    <div class="kpi-val" style="color: #a855f7;">{s}</div>
            \\    <div class="kpi-sub">{d} Meshes in Scene</div>
            \\  </div>
            \\</div>
        , .{
            if (summary.observed_avg_fps >= 55.0) "#22c55e" else if (summary.observed_avg_fps >= 30.0) "#eab308" else "#ef4444",
            summary.observed_avg_fps,
            summary.avg_interval_ms,
            summary.avg_frame_ms,
            if (summary.observed_fps_1pct_low >= 45.0) "#22c55e" else if (summary.observed_fps_1pct_low >= 25.0) "#eab308" else "#ef4444",
            summary.observed_fps_1pct_low,
            summary.observed_fps_01pct_low,
            summary.p99_interval_ms,
            summary.p50_interval_ms,
            summary.p99_frame_ms,
            summary.avg_draw_calls,
            summary.max_draw_calls,
            summary.avg_triangles,
            vram_total_str,
            cpu_mesh_str,
            if (memory_ptr) |m| m.mesh_count else 0,
        });
        defer allocator.free(kpi_html);
        try buf.appendSlice(allocator, kpi_html);

        // Section: "Что не так / Автоматическая диагностика"
        try buf.appendSlice(allocator,
            \\<div class="section-title">
            \\  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#eab308" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
            \\  Что не так / Автоматическая диагностика
            \\</div>
            \\<div class="diag-list">
        );

        for (findings) |finding| {
            const sev_class = switch (finding.severity) {
                .good => "good",
                .info => "info",
                .warning => "warning",
                .critical => "critical",
            };
            const sev_label = switch (finding.severity) {
                .good => "НОРМА",
                .info => "ИНФО",
                .warning => "ВНИМАНИЕ",
                .critical => "КРИТИЧНО",
            };
            const f_html = try std.fmt.allocPrint(allocator,
                \\  <div class="diag-card {s}">
                \\    <div class="diag-header">
                \\      <span class="diag-badge {s}">{s}</span>
                \\      <span class="diag-title">{s}</span>
                \\    </div>
                \\    <div class="diag-desc">{s}</div>
                \\    <div class="diag-rec"><strong>Рекомендация:</strong> {s}</div>
                \\  </div>
            , .{ sev_class, sev_class, sev_label, finding.title, finding.details, finding.recommendation });
            defer allocator.free(f_html);
            try buf.appendSlice(allocator, f_html);
        }
        try buf.appendSlice(allocator, "</div>\n");

        // Section: Interactive Stacked Frame Timeline (SVG)
        try buf.appendSlice(allocator,
            \\<div class="section-title">
            \\  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            \\  График времени кадров по фазам (Stacked Frame Timeline)
            \\</div>
            \\<div class="chart-box">
            \\  <svg viewBox="0 0 1000 240" class="timeline-svg" preserveAspectRatio="none">
        );

        const chart_h: f32 = 190.0;
        const chart_w: f32 = 930.0;
        const chart_y_bottom: f32 = 210.0;
        const chart_x_start: f32 = 50.0;
        const max_time = @max(35.0, summary.max_frame_ms * 1.15);
        const y_scale = chart_h / max_time;

        // Horizontal guidelines: 16.6ms (60 FPS) and 33.3ms (30 FPS)
        const y_16 = chart_y_bottom - (16.67 * y_scale);
        const line16_str = try std.fmt.allocPrint(allocator,
            \\<line x1="{d:.1}" y1="{d:.1}" x2="{d:.1}" y2="{d:.1}" stroke="#22c55e" stroke-dasharray="4" stroke-width="1.5" />
            \\<text x="5" y="{d:.1}" fill="#22c55e" font-size="11" font-weight="600">16.6ms (60 FPS)</text>
        , .{ chart_x_start, y_16, chart_x_start + chart_w, y_16, y_16 + 4.0 });
        defer allocator.free(line16_str);
        try buf.appendSlice(allocator, line16_str);

        if (max_time >= 33.33) {
            const y_33 = chart_y_bottom - (33.33 * y_scale);
            const line33_str = try std.fmt.allocPrint(allocator,
                \\<line x1="{d:.1}" y1="{d:.1}" x2="{d:.1}" y2="{d:.1}" stroke="#f97316" stroke-dasharray="4" stroke-width="1.5" />
                \\<text x="5" y="{d:.1}" fill="#f97316" font-size="11" font-weight="600">33.3ms (30 FPS)</text>
            , .{ chart_x_start, y_33, chart_x_start + chart_w, y_33, y_33 + 4.0 });
            defer allocator.free(line33_str);
            try buf.appendSlice(allocator, line33_str);
        }

        // Draw stacked bars for each frame
        const frame_count = self.frames.items.len;
        if (frame_count > 0) {
            const bar_step = chart_w / @as(f32, @floatFromInt(frame_count));
            const bar_w = @max(1.0, bar_step * 0.85);

            for (self.frames.items, 0..) |f, i| {
                const x = chart_x_start + @as(f32, @floatFromInt(i)) * bar_step;

                // Stack from bottom up: update -> prepare -> shadow -> main -> post
                const h_update = f.update_ms * y_scale;
                const h_prepare = f.prepare_ms * y_scale;
                const h_shadow = f.shadow_ms * y_scale;
                const h_main = f.main_ms * y_scale;
                const h_post = f.post_ms * y_scale;

                var cur_y = chart_y_bottom;

                // Group with native tooltip (stacked bars show CPU-submit phases, not GPU time)
                const tooltip_open = try std.fmt.allocPrint(allocator,
                    \\<g><title>Кадр #{d}: {d:.2} мс CPU-submit (FPS wall: {d:.1}, интервал wall: {d:.2} мс)&#10;Update: {d:.2} мс&#10;Prepare: {d:.2} мс&#10;Shadow (CPU submit): {d:.2} мс&#10;Main (CPU submit): {d:.2} мс&#10;Post (CPU submit): {d:.2} мс&#10;Draw calls: {d} | Tris: {d}</title>
                , .{ f.frame_index, f.total_frame_ms, f.fps, f.frame_interval_ms, f.update_ms, f.prepare_ms, f.shadow_ms, f.main_ms, f.post_ms, f.draw_calls, f.triangles });
                defer allocator.free(tooltip_open);
                try buf.appendSlice(allocator, tooltip_open);

                // Update rect
                if (h_update > 0.1) {
                    cur_y -= h_update;
                    const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#3b82f6\" />\n", .{ x, cur_y, bar_w, h_update });
                    defer allocator.free(r);
                    try buf.appendSlice(allocator, r);
                }

                // Prepare rect
                if (h_prepare > 0.1) {
                    cur_y -= h_prepare;
                    const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#06b6d4\" />\n", .{ x, cur_y, bar_w, h_prepare });
                    defer allocator.free(r);
                    try buf.appendSlice(allocator, r);
                }

                // Shadow rect
                if (h_shadow > 0.1) {
                    cur_y -= h_shadow;
                    const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#8b5cf6\" />\n", .{ x, cur_y, bar_w, h_shadow });
                    defer allocator.free(r);
                    try buf.appendSlice(allocator, r);
                }

                // Main pass rect
                if (h_main > 0.1) {
                    cur_y -= h_main;
                    const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#10b981\" />\n", .{ x, cur_y, bar_w, h_main });
                    defer allocator.free(r);
                    try buf.appendSlice(allocator, r);
                }

                // PostFX rect
                if (h_post > 0.1) {
                    cur_y -= h_post;
                    const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#f59e0b\" />\n", .{ x, cur_y, bar_w, h_post });
                    defer allocator.free(r);
                    try buf.appendSlice(allocator, r);
                }

                try buf.appendSlice(allocator, "</g>\n");
            }
        }

        try buf.appendSlice(allocator,
            \\  </svg>
            \\  <div class="chart-legend">
            \\    <div class="legend-item"><div class="legend-dot" style="background: #3b82f6;"></div>Update</div>
            \\    <div class="legend-item"><div class="legend-dot" style="background: #06b6d4;"></div>Prepare</div>
            \\    <div class="legend-item"><div class="legend-dot" style="background: #8b5cf6;"></div>Shadow Pass</div>
            \\    <div class="legend-item"><div class="legend-dot" style="background: #10b981;"></div>Main Pass</div>
            \\    <div class="legend-item"><div class="legend-dot" style="background: #f59e0b;"></div>PostFX</div>
            \\  </div>
            \\</div>
        );

        // Section: Top Spike Frames Table (ranked by CPU-submit time, not GPU/wall time)
        try buf.appendSlice(allocator,
            \\<div class="section-title">
            \\  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            \\  Топ пиковых кадров (Spike Frames, время CPU-submit)
            \\</div>
            \\<div class="card-table">
            \\<table>
            \\  <thead>
            \\    <tr>
            \\      <th>Ранг</th>
            \\      <th>Кадр #</th>
            \\      <th class="num">Время CPU-submit</th>
            \\      <th class="num">FPS (wall)</th>
            \\      <th>Главная причина (CPU-фаза)</th>
            \\      <th class="num">Draw Calls</th>
            \\      <th class="num">Треугольники</th>
            \\      <th class="num">Pipeline Switches</th>
            \\      <th class="num">Uploads (КБ)</th>
            \\    </tr>
            \\  </thead>
            \\  <tbody>
        );

        // Sort frames by total_frame_ms (CPU-submit sum) descending
        if (self.frames.items.len > 0) {
            const spike_count = @min(10, self.frames.items.len);
            const sorted_frames = try allocator.dupe(FrameRecord, self.frames.items);
            defer allocator.free(sorted_frames);

            std.mem.sort(FrameRecord, sorted_frames, {}, struct {
                fn lessThan(_: void, a: FrameRecord, b: FrameRecord) bool {
                    return a.total_frame_ms > b.total_frame_ms;
                }
            }.lessThan);

            for (sorted_frames[0..spike_count], 1..) |sf, rank| {
                const culprit = dominantPhase(sf);
                const upload_kb: f32 = @as(f32, @floatFromInt(sf.uploaded_bytes)) / 1024.0;
                const row = try std.fmt.allocPrint(allocator,
                    \\    <tr>
                    \\      <td><strong>#{d}</strong></td>
                    \\      <td>Кадр {d}</td>
                    \\      <td class="num"><span style="color: {s}; font-weight: 700;">{d:.2} мс</span></td>
                    \\      <td class="num">{d:.1}</td>
                    \\      <td><span style="color: #fff; font-weight: 600;">{s}</span> <span style="color: var(--text-muted);">({d:.1} мс, {d:.0}%)</span></td>
                    \\      <td class="num">{d}</td>
                    \\      <td class="num">{d}</td>
                    \\      <td class="num">{d}</td>
                    \\      <td class="num">{d:.1}</td>
                    \\    </tr>
                , .{
                    rank,
                    sf.frame_index,
                    if (sf.total_frame_ms > 33.33) "#ef4444" else if (sf.total_frame_ms > 16.67) "#eab308" else "#22c55e",
                    sf.total_frame_ms,
                    sf.fps,
                    culprit.name,
                    culprit.ms,
                    culprit.percent,
                    sf.draw_calls,
                    sf.triangles,
                    sf.pipeline_switches,
                    upload_kb,
                });
                defer allocator.free(row);
                try buf.appendSlice(allocator, row);
            }
        }
        try buf.appendSlice(allocator, "  </tbody>\n</table>\n</div>\n");

        // Section: Memory & VRAM Breakdown
        if (memory_ptr) |mem| {
            try buf.appendSlice(allocator,
                \\<div class="section-title">
                \\  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
                \\  Распределение видеопамяти (VRAM Breakdown)
                \\</div>
            );

            // VRAM segmented progress bar
            const total_vram = @max(1, mem.total_gpu_vram_bytes);
            const tex_pct = (@as(f32, @floatFromInt(mem.textures_vram_bytes)) / @as(f32, @floatFromInt(total_vram))) * 100.0;
            const mesh_pct = (@as(f32, @floatFromInt(mem.meshes_vram_bytes)) / @as(f32, @floatFromInt(total_vram))) * 100.0;
            const rt_pct = (@as(f32, @floatFromInt(mem.render_targets_vram_bytes)) / @as(f32, @floatFromInt(total_vram))) * 100.0;

            const tex_vram_str = try formatBytes(allocator, mem.textures_vram_bytes);
            defer allocator.free(tex_vram_str);
            const mesh_vram_str = try formatBytes(allocator, mem.meshes_vram_bytes);
            defer allocator.free(mesh_vram_str);
            const rt_vram_str = try formatBytes(allocator, mem.render_targets_vram_bytes);
            defer allocator.free(rt_vram_str);

            const vram_bar_html = try std.fmt.allocPrint(allocator,
                \\<div class="vram-bar">
                \\  <div class="vram-seg" style="width: {d:.1}%; background: #38bdf8;" title="Текстуры: {s} ({d:.1}%)"></div>
                \\  <div class="vram-seg" style="width: {d:.1}%; background: #a855f7;" title="Меши: {s} ({d:.1}%)"></div>
                \\  <div class="vram-seg" style="width: {d:.1}%; background: #f59e0b;" title="Render Targets: {s} ({d:.1}%)"></div>
                \\</div>
                \\<div class="chart-legend" style="margin-bottom: 24px;">
                \\  <div class="legend-item"><div class="legend-dot" style="background: #38bdf8;"></div>Текстуры: {s} ({d:.1}%)</div>
                \\  <div class="legend-item"><div class="legend-dot" style="background: #a855f7;"></div>Меши (VBO/IBO): {s} ({d:.1}%)</div>
                \\  <div class="legend-item"><div class="legend-dot" style="background: #f59e0b;"></div>Буферы кадров (RT): {s} ({d:.1}%)</div>
                \\</div>
            , .{ tex_pct, tex_vram_str, tex_pct, mesh_pct, mesh_vram_str, mesh_pct, rt_pct, rt_vram_str, rt_pct, tex_vram_str, tex_pct, mesh_vram_str, mesh_pct, rt_vram_str, rt_pct });
            defer allocator.free(vram_bar_html);
            try buf.appendSlice(allocator, vram_bar_html);

            // Top Textures Table
            try buf.appendSlice(allocator,
                \\<div class="section-title" style="font-size: 16px;">Текстуры (отсортированы по размеру VRAM)</div>
                \\<div class="card-table">
                \\<table>
                \\  <thead>
                \\    <tr>
                \\      <th>Название / Назначение</th>
                \\      <th>Разрешение</th>
                \\      <th>Тип</th>
                \\      <th>Mips</th>
                \\      <th>Формат</th>
                \\      <th class="num">VRAM</th>
                \\    </tr>
                \\  </thead>
                \\  <tbody>
            );

            const sorted_textures = try allocator.dupe(TextureMemoryRecord, mem.textures);
            defer allocator.free(sorted_textures);
            std.mem.sort(TextureMemoryRecord, sorted_textures, {}, struct {
                fn lessThan(_: void, a: TextureMemoryRecord, b: TextureMemoryRecord) bool {
                    return a.gpu_bytes > b.gpu_bytes;
                }
            }.lessThan);

            for (sorted_textures) |tex| {
                const tex_sz = try formatBytes(allocator, tex.gpu_bytes);
                defer allocator.free(tex_sz);
                const row = try std.fmt.allocPrint(allocator,
                    \\    <tr>
                    \\      <td><strong style="color: #fff;">{s}</strong></td>
                    \\      <td>{d}x{d}</td>
                    \\      <td>{s}</td>
                    \\      <td>{d}</td>
                    \\      <td>{s}</td>
                    \\      <td class="num"><strong>{s}</strong></td>
                    \\    </tr>
                , .{ tex.name, tex.width, tex.height, if (tex.is_cube) "Cubemap" else "2D Texture", tex.num_mips, @tagName(tex.format), tex_sz });
                defer allocator.free(row);
                try buf.appendSlice(allocator, row);
            }
            try buf.appendSlice(allocator, "  </tbody>\n</table>\n</div>\n");

            // Top Meshes Table
            try buf.appendSlice(allocator,
                \\<div class="section-title" style="font-size: 16px;">Меши (геометрия сцены)</div>
                \\<div class="card-table">
                \\<table>
                \\  <thead>
                \\    <tr>
                \\      <th>Имя меша</th>
                \\      <th class="num">Вершины</th>
                \\      <th class="num">Индексы</th>
                \\      <th>Тип индексов</th>
                \\      <th class="num">VRAM Буферы</th>
                \\      <th class="num">CPU Память</th>
                \\    </tr>
                \\  </thead>
                \\  <tbody>
            );

            const sorted_meshes = try allocator.dupe(MeshMemoryRecord, mem.meshes);
            defer allocator.free(sorted_meshes);
            std.mem.sort(MeshMemoryRecord, sorted_meshes, {}, struct {
                fn lessThan(_: void, a: MeshMemoryRecord, b: MeshMemoryRecord) bool {
                    return a.gpu_bytes > b.gpu_bytes;
                }
            }.lessThan);

            for (sorted_meshes) |m| {
                const gpu_sz = try formatBytes(allocator, m.gpu_bytes);
                defer allocator.free(gpu_sz);
                const cpu_sz = try formatBytes(allocator, m.cpu_bytes);
                defer allocator.free(cpu_sz);

                const row = try std.fmt.allocPrint(allocator,
                    \\    <tr>
                    \\      <td><strong style="color: #fff;">{s}</strong></td>
                    \\      <td class="num">{d}</td>
                    \\      <td class="num">{d}</td>
                    \\      <td>{s}</td>
                    \\      <td class="num"><strong>{s}</strong></td>
                    \\      <td class="num" style="color: var(--text-muted);">{s}</td>
                    \\    </tr>
                , .{ m.name, m.vertex_count, m.index_count, @tagName(m.index_type), gpu_sz, cpu_sz });
                defer allocator.free(row);
                try buf.appendSlice(allocator, row);
            }
            try buf.appendSlice(allocator, "  </tbody>\n</table>\n</div>\n");

            // Render Targets Table
            try buf.appendSlice(allocator,
                \\<div class="section-title" style="font-size: 16px;">Таргеты рендера (Offscreen Targets)</div>
                \\<div class="card-table">
                \\<table>
                \\  <thead>
                \\    <tr>
                \\      <th>Название таргета</th>
                \\      <th>Разрешение</th>
                \\      <th>MSAA Сэмплы</th>
                \\      <th>Формат</th>
                \\      <th class="num">VRAM</th>
                \\    </tr>
                \\  </thead>
                \\  <tbody>
            );

            for (mem.render_targets) |rt| {
                const rt_sz = try formatBytes(allocator, rt.gpu_bytes);
                defer allocator.free(rt_sz);
                const row = try std.fmt.allocPrint(allocator,
                    \\    <tr>
                    \\      <td><strong style="color: #fff;">{s}</strong></td>
                    \\      <td>{d}x{d}</td>
                    \\      <td>{d}x</td>
                    \\      <td>{s}</td>
                    \\      <td class="num"><strong>{s}</strong></td>
                    \\    </tr>
                , .{ rt.name, rt.width, rt.height, rt.samples, @tagName(rt.format), rt_sz });
                defer allocator.free(row);
                try buf.appendSlice(allocator, row);
            }
            try buf.appendSlice(allocator, "  </tbody>\n</table>\n</div>\n");
        }

        // Footer
        try buf.appendSlice(allocator,
            \\<footer>
            \\  Сгенерировано встроенным модулем профилирования Agate Engine &bull; Совместимо с Chrome Trace & Perfetto
            \\</footer>
            \\</div>
            \\</body>
            \\</html>
        );

        return buf.toOwnedSlice(allocator);
    }

    /// Generates a comprehensive Markdown report.
    pub fn generateReportMd(self: *const Profiler, scene: ?*const Scene, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        const memory_ptr: ?*const MemorySnapshot = if (scene) |sc| blk: {
            var mut_prof = @constCast(self);
            break :blk mut_prof.captureMemorySnapshot(sc) catch null;
        } else (if (self.last_memory_snapshot) |*s| s else null);

        const summary = self.summarize();
        const findings = try self.analyze(memory_ptr, allocator);
        defer {
            for (findings) |*f| @constCast(f).deinit(allocator);
            allocator.free(findings);
        }

        // Header & Overview
        try buf.appendSlice(allocator,
            \\# Agate Engine - Отчет о производительности и памяти
            \\
            \\## 1. Сводка сессии (Session Overview)
            \\
        );

        const vram_total_str = if (memory_ptr) |m| try formatBytes(allocator, m.total_gpu_vram_bytes) else try allocator.dupe(u8, "N/A");
        defer allocator.free(vram_total_str);
        const cpu_mesh_str = if (memory_ptr) |m| try formatBytes(allocator, m.total_cpu_mesh_bytes) else try allocator.dupe(u8, "N/A");
        defer allocator.free(cpu_mesh_str);

        const overview_table = try std.fmt.allocPrint(allocator,
            \\| Метрика | Значение | Метрика | Значение |
            \\| :--- | :--- | :--- | :--- |
            \\| **Всего кадров** | {d} | **Длительность (сумма CPU-submit)** | {d:.2} с |
            \\| **Средний FPS (wall, интервал)** | {d:.1} FPS | **1% Low FPS (wall)** | {d:.1} FPS |
            \\| **Средний интервал (wall)** | {d:.2} мс | **P99 интервал (wall)** | {d:.2} мс |
            \\| **Средний CPU-submit** | {d:.2} мс | **Макс. CPU-submit** | {d:.2} мс |
            \\| **CPU-submit P50 / P95 / P99** | {d:.2} / {d:.2} / {d:.2} мс | **Просадки интервала > 33.3 мс (wall)** | {d} кадров (CPU-submit > 33.3: {d}) |
            \\| **Средний Draw Calls** | {d} | **Макс. Draw Calls** | {d} |
            \\| **Средний треугольников** | {d} | **Макс. треугольников** | {d} |
            \\| **Суммарный VRAM** | {s} | **CPU геометрия** | {s} |
            \\
            \\### Фазы кадра в среднем (CPU-submit, не GPU-время)
            \\- **Update (CPU логика/анимация):** {d:.2} мс
            \\- **Prepare (подготовка очередей):** {d:.2} мс
            \\- **Shadow Pass (CPU submit, не GPU):** {d:.2} мс
            \\- **Main Pass (CPU submit, не GPU):** {d:.2} мс
            \\- **PostFX (CPU submit, не GPU):** {d:.2} мс
            \\
        , .{
            summary.frame_count,
            summary.total_time_ms / 1000.0,
            summary.observed_avg_fps,
            summary.observed_fps_1pct_low,
            summary.avg_interval_ms,
            summary.p99_interval_ms,
            summary.avg_frame_ms,
            summary.max_frame_ms,
            summary.p50_frame_ms,
            summary.p95_frame_ms,
            summary.p99_frame_ms,
            summary.interval_hitches_over_33ms,
            summary.hitches_over_33ms,
            summary.avg_draw_calls,
            summary.max_draw_calls,
            summary.avg_triangles,
            summary.max_triangles,
            vram_total_str,
            cpu_mesh_str,
            summary.avg_update_ms,
            summary.avg_prepare_ms,
            summary.avg_shadow_ms,
            summary.avg_main_ms,
            summary.avg_post_ms,
        });
        defer allocator.free(overview_table);
        try buf.appendSlice(allocator, overview_table);

        // Section: "Что не так"
        try buf.appendSlice(allocator,
            \\## 2. Что не так / Автоматическая диагностика узких мест
            \\
        );

        for (findings) |finding| {
            const badge = switch (finding.severity) {
                .good => "🟢 [НОРМА]",
                .info => "ℹ️ [ИНФО]",
                .warning => "⚠️ [ВНИМАНИЕ]",
                .critical => "🚨 [КРИТИЧНО]",
            };
            const f_md = try std.fmt.allocPrint(allocator,
                \\### {s} {s}
                \\- **Описание:** {s}
                \\- **Рекомендация:** {s}
                \\
            , .{ badge, finding.title, finding.details, finding.recommendation });
            defer allocator.free(f_md);
            try buf.appendSlice(allocator, f_md);
        }

        // Section: Spike Frames (ranked by CPU-submit time, not GPU/wall time)
        try buf.appendSlice(allocator,
            \\## 3. Топ пиковых кадров (Spike Frames, время CPU-submit)
            \\
            \\| Ранг | Кадр # | Время CPU-submit (мс) | FPS (wall) | Главная причина (CPU-фаза) | Draw Calls | Треугольники | Uploads |
            \\| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
        );

        if (self.frames.items.len > 0) {
            const spike_count = @min(10, self.frames.items.len);
            const sorted_frames = try allocator.dupe(FrameRecord, self.frames.items);
            defer allocator.free(sorted_frames);

            std.mem.sort(FrameRecord, sorted_frames, {}, struct {
                fn lessThan(_: void, a: FrameRecord, b: FrameRecord) bool {
                    return a.total_frame_ms > b.total_frame_ms;
                }
            }.lessThan);

            for (sorted_frames[0..spike_count], 1..) |sf, rank| {
                const culprit = dominantPhase(sf);
                const upload_kb: f32 = @as(f32, @floatFromInt(sf.uploaded_bytes)) / 1024.0;
                const row = try std.fmt.allocPrint(allocator,
                    \\| #{d} | {d} | **{d:.2} мс** | {d:.1} | {s} ({d:.1} мс, {d:.0}%) | {d} | {d} | {d:.1} KB |
                , .{
                    rank,
                    sf.frame_index,
                    sf.total_frame_ms,
                    sf.fps,
                    culprit.name,
                    culprit.ms,
                    culprit.percent,
                    sf.draw_calls,
                    sf.triangles,
                    upload_kb,
                });
                defer allocator.free(row);
                try buf.appendSlice(allocator, row);
                try buf.appendSlice(allocator, "\n");
            }
        }
        try buf.appendSlice(allocator, "\n");

        // Section: Memory Breakdown
        if (memory_ptr) |mem| {
            try buf.appendSlice(allocator,
                \\## 4. Использование памяти и VRAM
                \\
            );

            const tex_vram_str = try formatBytes(allocator, mem.textures_vram_bytes);
            defer allocator.free(tex_vram_str);
            const mesh_vram_str = try formatBytes(allocator, mem.meshes_vram_bytes);
            defer allocator.free(mesh_vram_str);
            const rt_vram_str = try formatBytes(allocator, mem.render_targets_vram_bytes);
            defer allocator.free(rt_vram_str);

            const mem_summary = try std.fmt.allocPrint(allocator,
                \\- **Текстуры:** {s}
                \\- **Меши (VBO / IBO):** {s}
                \\- **Буферы кадра (Render Targets):** {s}
                \\- **Суммарно VRAM:** {s}
                \\- **CPU геометрия:** {s}
                \\
                \\### Топ тяжелых текстур
                \\| Название | Разрешение | Тип | Mips | Формат | VRAM |
                \\| :--- | :--- | :--- | :--- | :--- | :--- |
            , .{ tex_vram_str, mesh_vram_str, rt_vram_str, vram_total_str, cpu_mesh_str });
            defer allocator.free(mem_summary);
            try buf.appendSlice(allocator, mem_summary);
            try buf.appendSlice(allocator, "\n");

            const sorted_textures = try allocator.dupe(TextureMemoryRecord, mem.textures);
            defer allocator.free(sorted_textures);
            std.mem.sort(TextureMemoryRecord, sorted_textures, {}, struct {
                fn lessThan(_: void, a: TextureMemoryRecord, b: TextureMemoryRecord) bool {
                    return a.gpu_bytes > b.gpu_bytes;
                }
            }.lessThan);

            const top_tex_n = @min(10, sorted_textures.len);
            for (sorted_textures[0..top_tex_n]) |tex| {
                const tex_sz = try formatBytes(allocator, tex.gpu_bytes);
                defer allocator.free(tex_sz);
                const row = try std.fmt.allocPrint(allocator,
                    \\| {s} | {d}x{d} | {s} | {d} | {s} | **{s}** |
                , .{ tex.name, tex.width, tex.height, if (tex.is_cube) "Cubemap" else "2D", tex.num_mips, @tagName(tex.format), tex_sz });
                defer allocator.free(row);
                try buf.appendSlice(allocator, row);
                try buf.appendSlice(allocator, "\n");
            }
            try buf.appendSlice(allocator, "\n");

            // Top Meshes
            try buf.appendSlice(allocator,
                \\### Топ мешей по памяти
                \\| Меш | Вершины | Индексы | Формат | VRAM | CPU |
                \\| :--- | :--- | :--- | :--- | :--- | :--- |
            );
            try buf.appendSlice(allocator, "\n");

            const sorted_meshes = try allocator.dupe(MeshMemoryRecord, mem.meshes);
            defer allocator.free(sorted_meshes);
            std.mem.sort(MeshMemoryRecord, sorted_meshes, {}, struct {
                fn lessThan(_: void, a: MeshMemoryRecord, b: MeshMemoryRecord) bool {
                    return a.gpu_bytes > b.gpu_bytes;
                }
            }.lessThan);

            const top_mesh_n = @min(10, sorted_meshes.len);
            for (sorted_meshes[0..top_mesh_n]) |m| {
                const gpu_sz = try formatBytes(allocator, m.gpu_bytes);
                defer allocator.free(gpu_sz);
                const cpu_sz = try formatBytes(allocator, m.cpu_bytes);
                defer allocator.free(cpu_sz);
                const row = try std.fmt.allocPrint(allocator,
                    \\| {s} | {d} | {d} | {s} | **{s}** | {s} |
                , .{ m.name, m.vertex_count, m.index_count, @tagName(m.index_type), gpu_sz, cpu_sz });
                defer allocator.free(row);
                try buf.appendSlice(allocator, row);
                try buf.appendSlice(allocator, "\n");
            }
            try buf.appendSlice(allocator, "\n");
        }

        return buf.toOwnedSlice(allocator);
    }

    /// Generates Chrome Trace Event JSON for chrome://tracing and ui.perfetto.dev.
    pub fn generateTraceJson(self: *const Profiler, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\n  \"traceEvents\": [\n");

        for (self.frames.items, 0..) |f, i| {
            const is_last_frame = (i + 1 == self.frames.items.len);
            const ts = f.timestamp_us;
            const dur_us: u64 = @intFromFloat(f.total_frame_ms * 1000.0);

            // Complete event for whole frame. dur covers the CPU-submit span;
            // fps/interval are observed wall-clock pacing.
            const frame_event = try std.fmt.allocPrint(allocator,
                \\    {{"name": "Frame #{d} (CPU submit)", "cat": "frame", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1, "args": {{"fps_wall": {d:.1}, "frame_interval_ms": {d:.3}, "cpu_submit_ms": {d:.3}, "draw_calls": {d}, "triangles": {d}, "switches": {d}}}}},
            , .{ f.frame_index, ts, dur_us, f.fps, f.frame_interval_ms, f.total_frame_ms, f.draw_calls, f.triangles, f.pipeline_switches });
            defer allocator.free(frame_event);
            try buf.appendSlice(allocator, frame_event);
            try buf.appendSlice(allocator, "\n");

            // Sub-phase slices inside the frame (CPU-submit times, not GPU execution).
            // All phases use "cpu": Shadow/Main/Post measure CPU timers around
            // sg submit calls, never GPU timestamps.
            var cur_ts = ts;
            const u_us: u64 = @intFromFloat(f.update_ms * 1000.0);
            const p_us: u64 = @intFromFloat(f.prepare_ms * 1000.0);
            const s_us: u64 = @intFromFloat(f.shadow_ms * 1000.0);
            const m_us: u64 = @intFromFloat(f.main_ms * 1000.0);
            const post_us: u64 = @intFromFloat(f.post_ms * 1000.0);

            // Update
            const u_ev = try std.fmt.allocPrint(allocator,
                \\    {{"name": "Update", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}},
            , .{ cur_ts, u_us });
            defer allocator.free(u_ev);
            try buf.appendSlice(allocator, u_ev);
            try buf.appendSlice(allocator, "\n");
            cur_ts += u_us;

            // Prepare
            const p_ev = try std.fmt.allocPrint(allocator,
                \\    {{"name": "Prepare", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}},
            , .{ cur_ts, p_us });
            defer allocator.free(p_ev);
            try buf.appendSlice(allocator, p_ev);
            try buf.appendSlice(allocator, "\n");
            cur_ts += p_us;

            // Shadow
            const s_ev = try std.fmt.allocPrint(allocator,
                \\    {{"name": "Shadow Pass (CPU submit)", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}},
            , .{ cur_ts, s_us });
            defer allocator.free(s_ev);
            try buf.appendSlice(allocator, s_ev);
            try buf.appendSlice(allocator, "\n");
            cur_ts += s_us;

            // Main
            const m_ev = try std.fmt.allocPrint(allocator,
                \\    {{"name": "Main Pass (CPU submit)", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}},
            , .{ cur_ts, m_us });
            defer allocator.free(m_ev);
            try buf.appendSlice(allocator, m_ev);
            try buf.appendSlice(allocator, "\n");
            cur_ts += m_us;

            // PostFX
            const post_comma = if (is_last_frame) "" else ",";
            const post_ev = try std.fmt.allocPrint(allocator,
                \\    {{"name": "PostFX (CPU submit)", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}}{s}
            , .{ cur_ts, post_us, post_comma });
            defer allocator.free(post_ev);
            try buf.appendSlice(allocator, post_ev);
            try buf.appendSlice(allocator, "\n");
        }

        try buf.appendSlice(allocator, "  ],\n  \"displayTimeUnit\": \"ms\"\n}\n");
        return buf.toOwnedSlice(allocator);
    }

    /// Saves an interactive HTML report to `path`.
    pub fn saveReportHtml(self: *const Profiler, scene: ?*const Scene, path: []const u8) !void {
        const html = try self.generateReportHtml(scene, self.allocator);
        defer self.allocator.free(html);
        const io = std.Io.Threaded.global_single_threaded.io();
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = html });
    }

    /// Saves a Markdown report to `path`.
    pub fn saveReportMd(self: *const Profiler, scene: ?*const Scene, path: []const u8) !void {
        const md = try self.generateReportMd(scene, self.allocator);
        defer self.allocator.free(md);
        const io = std.Io.Threaded.global_single_threaded.io();
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = md });
    }

    /// Saves Chrome Trace Event JSON to `path`.
    pub fn saveTraceJson(self: *const Profiler, path: []const u8) !void {
        const json = try self.generateTraceJson(self.allocator);
        defer self.allocator.free(json);
        const io = std.Io.Threaded.global_single_threaded.io();
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }

    /// Saves all reports (HTML, Markdown, and Chrome Trace JSON) to `<base_path>.html`,
    /// `<base_path>.md`, and `<base_path>.json`.
    pub fn saveReports(self: *const Profiler, scene: ?*const Scene, base_path: []const u8) !void {
        if (scene) |sc| {
            var mut_prof = @constCast(self);
            _ = try mut_prof.captureMemorySnapshot(sc);
        }
        const html_path = try std.fmt.allocPrint(self.allocator, "{s}.html", .{base_path});
        defer self.allocator.free(html_path);
        try self.saveReportHtml(null, html_path);

        const md_path = try std.fmt.allocPrint(self.allocator, "{s}.md", .{base_path});
        defer self.allocator.free(md_path);
        try self.saveReportMd(null, md_path);

        const json_path = try std.fmt.allocPrint(self.allocator, "{s}.json", .{base_path});
        defer self.allocator.free(json_path);
        try self.saveTraceJson(json_path);
    }
};

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

/// Sleep helper for pacing-sensitive tests (Zig 0.16 has no Thread.sleep).
fn testSleepMs(ms: u64) void {
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    var rem: std.c.timespec = undefined;
    _ = std.c.nanosleep(&ts, &rem);
}

test "Profiler start, recordFrame, and summarize" {
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    try std.testing.expect(!prof.isRecording());
    prof.start();
    try std.testing.expect(prof.isRecording());

    // Record synthetic frames
    var stats: SceneStats = .{
        .update_ms = 2.0,
        .prepare_ms = 1.0,
        .shadow_ms = 3.0,
        .main_ms = 8.0,
        .post_ms = 2.0,
        .draw_calls = 50,
        .triangles = 10000,
        .pipeline_switches = 5,
    };

    prof.recordFrame(1, &stats);
    stats.main_ms = 12.0;
    prof.recordFrame(2, &stats);
    stats.main_ms = 25.0; // spike frame
    prof.recordFrame(3, &stats);

    prof.stop();
    try std.testing.expect(!prof.isRecording());

    const summary = prof.summarize();
    try std.testing.expectEqual(@as(usize, 3), summary.frame_count);
    try std.testing.expect(summary.avg_frame_ms > 15.0);
    try std.testing.expect(summary.max_frame_ms >= 33.0);
    try std.testing.expectEqual(@as(u32, 50), summary.avg_draw_calls);

    // Analyze
    const findings = try prof.analyze(null, ally);
    defer {
        for (findings) |*f| @constCast(f).deinit(ally);
        ally.free(findings);
    }
    try std.testing.expect(findings.len > 0);
}

test "Profiler report HTML, MD, and JSON generation" {
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    prof.start();
    var stats: SceneStats = .{
        .update_ms = 1.5,
        .prepare_ms = 0.5,
        .shadow_ms = 2.0,
        .main_ms = 6.0,
        .post_ms = 1.0,
        .draw_calls = 25,
        .triangles = 5000,
        .pipeline_switches = 2,
    };
    prof.recordFrame(1, &stats);
    prof.recordFrame(2, &stats);
    prof.stop();

    // HTML
    const html = try prof.generateReportHtml(null, ally);
    defer ally.free(html);
    try std.testing.expect(std.mem.indexOf(u8, html, "<!DOCTYPE html>") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "Performance & Memory Profile Report") != null);

    // MD
    const md = try prof.generateReportMd(null, ally);
    defer ally.free(md);
    try std.testing.expect(std.mem.indexOf(u8, md, "# Agate Engine - Отчет") != null);

    // JSON
    const json = try prof.generateTraceJson(ally);
    defer ally.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"traceEvents\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"Frame #1 (CPU submit)\"") != null);
    // Submit times must never be presented as GPU time.
    try std.testing.expect(std.mem.indexOf(u8, json, "\"cat\": \"gpu\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"cat\": \"cpu\"") != null);

    // Reports must label CPU-submit vs wall-clock pacing explicitly.
    try std.testing.expect(std.mem.indexOf(u8, html, "wall") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "CPU-submit") != null);
    try std.testing.expect(std.mem.indexOf(u8, md, "wall") != null);
    try std.testing.expect(std.mem.indexOf(u8, md, "CPU-submit") != null);
}

test "Profiler memory snapshot and file saving" {
    const ally = std.testing.allocator;
    const testScene = @import("testing.zig").testScene;
    var scene = testScene(ally);
    defer {
        @import("scene/content.zig").deinitMeshes(ally, &scene.meshes);
        scene.profiler.deinit();
    }

    const m = try ally.create(Mesh);
    m.* = @import("testing.zig").testMesh("TestCube");
    m.vertex_count = 24;
    m.index_count = 36;
    m.index_type = .UINT16;
    try scene.meshes.append(ally, m);

    var prof = Profiler.init(ally);
    defer prof.deinit();

    const snap = try prof.captureMemorySnapshot(&scene);

    try std.testing.expectEqual(@as(usize, 1), snap.mesh_count);
    try std.testing.expectEqual(@as(usize, 1), snap.meshes.len);
    try std.testing.expectEqualStrings("TestCube", snap.meshes[0].name);

    // Test saving reports to temporary path
    const tmp_base = ".zig-cache/test_profile_out";
    try prof.saveReports(&scene, tmp_base);

    const io = std.Io.Threaded.global_single_threaded.io();
    // Verify files were created
    const h_file = try std.Io.Dir.cwd().openFile(io, tmp_base ++ ".html", .{});
    h_file.close(io);
    const m_file = try std.Io.Dir.cwd().openFile(io, tmp_base ++ ".md", .{});
    m_file.close(io);
    const j_file = try std.Io.Dir.cwd().openFile(io, tmp_base ++ ".json", .{});
    j_file.close(io);

    // Clean up temporary files
    try std.Io.Dir.cwd().deleteFile(io, tmp_base ++ ".html");
    try std.Io.Dir.cwd().deleteFile(io, tmp_base ++ ".md");
    try std.Io.Dir.cwd().deleteFile(io, tmp_base ++ ".json");
}

test "Profiler analyzer detects bottlenecks" {
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    prof.start();
    // Simulate high draw calls and severe hitch. Pacing findings now come from
    // real wall-clock intervals, so sleep ~40ms between records to register
    // genuine interval hitches (~0.4s total for 10 frames).
    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 0.5,
        .shadow_ms = 1.0,
        .main_ms = 45.0, // huge hitch in main pass
        .post_ms = 1.0,
        .draw_calls = 1200, // critical draw call count
        .triangles = 200000,
        .pipeline_switches = 150,
    };
    for (0..10) |i| {
        if (i > 0) testSleepMs(40);
        prof.recordFrame(i, &stats);
    }
    prof.stop();

    const summary = prof.summarize();
    // Pacing hitches must be observed on the wall clock, not just CPU sums.
    try std.testing.expect(summary.interval_hitches_over_33ms > 0);

    const findings = try prof.analyze(null, ally);
    defer {
        for (findings) |*f| @constCast(f).deinit(ally);
        ally.free(findings);
    }

    var found_draw_calls = false;
    var found_main_pass = false;
    var found_hitch = false;

    for (findings) |f| {
        if (std.mem.indexOf(u8, f.title, "Draw Calls") != null) found_draw_calls = true;
        if (std.mem.indexOf(u8, f.title, "Main Render Pass") != null) found_main_pass = true;
        if (std.mem.indexOf(u8, f.title, "Просадки") != null) found_hitch = true;
    }

    try std.testing.expect(found_draw_calls);
    try std.testing.expect(found_main_pass);
    try std.testing.expect(found_hitch);
}

test "Profiler summarize uses frame intervals for pacing" {
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    prof.start();
    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 0.5,
        .shadow_ms = 1.0,
        .main_ms = 6.0,
        .post_ms = 1.0,
        .draw_calls = 10,
        .triangles = 1000,
        .pipeline_switches = 1,
    };
    const n_frames = 6;
    for (0..n_frames) |i| {
        // Keep every interval comfortably above the 0.1ms fps-validity gate
        // so fps == 1/dt_s holds exactly for every record.
        testSleepMs(2);
        prof.recordFrame(i, &stats);
    }
    prof.stop();

    try std.testing.expectEqual(n_frames, prof.frames.items.len);

    // Each record's wall-clock fields derive from its own dt_s.
    for (prof.frames.items) |rec| {
        try std.testing.expect(rec.dt_s > 0);
        try std.testing.expectEqual(rec.dt_s * 1000.0, rec.frame_interval_ms);
        try std.testing.expect(rec.dt_s > 0.0001);
        try std.testing.expectEqual(@as(f32, 1.0) / rec.dt_s, rec.fps);
    }

    // Recompute pacing directly from the recorded dt_s values
    // (exact comparison, no wall-clock thresholds).
    const summary = prof.summarize();

    // CPU-submit average is untouched by pacing: 1+0.5+1+6+1 == 9.5.
    try std.testing.expectEqual(@as(f32, 9.5), summary.avg_frame_ms);

    var expected = try ally.alloc(f32, n_frames);
    defer ally.free(expected);
    var m: usize = 0;
    var sum: f64 = 0;
    var max_iv: f32 = 0;
    var eh16: u32 = 0;
    var eh33: u32 = 0;
    var eh50: u32 = 0;
    for (prof.frames.items) |rec| {
        if (rec.dt_s > 0) {
            const iv = rec.frame_interval_ms;
            expected[m] = iv;
            m += 1;
            sum += iv;
            max_iv = @max(max_iv, iv);
            if (iv > 50.0) eh50 += 1;
            if (iv > 33.33) eh33 += 1;
            if (iv > 16.67) eh16 += 1;
        }
    }
    try std.testing.expect(m > 0);
    const valid = expected[0..m];
    std.mem.sort(f32, valid, {}, struct {
        fn lessThan(_: void, a: f32, b: f32) bool {
            return a < b;
        }
    }.lessThan);

    const exp_avg: f32 = @floatCast(sum / @as(f64, @floatFromInt(m)));
    const exp_fps: f32 = if (exp_avg > 0.001) 1000.0 / exp_avg else 0;
    const p50_idx = @min(m - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(m)) * 0.50)));
    const p99_idx = @min(m - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(m)) * 0.99)));
    const c1 = @max(1, m / 100);
    var s1: f64 = 0;
    for (valid[m - c1 ..]) |t| s1 += t;
    const a1: f32 = @floatCast(s1 / @as(f64, @floatFromInt(c1)));
    const exp_1pct: f32 = if (a1 > 0.001) 1000.0 / a1 else 0;
    const c01 = @max(1, m / 1000);
    var s01: f64 = 0;
    for (valid[m - c01 ..]) |t| s01 += t;
    const a01: f32 = @floatCast(s01 / @as(f64, @floatFromInt(c01)));
    const exp_01pct: f32 = if (a01 > 0.001) 1000.0 / a01 else 0;

    try std.testing.expectEqual(exp_avg, summary.avg_interval_ms);
    try std.testing.expectEqual(exp_fps, summary.observed_avg_fps);
    try std.testing.expectEqual(valid[p50_idx], summary.p50_interval_ms);
    try std.testing.expectEqual(valid[p99_idx], summary.p99_interval_ms);
    try std.testing.expectEqual(max_iv, summary.max_interval_ms);
    try std.testing.expectEqual(exp_1pct, summary.observed_fps_1pct_low);
    try std.testing.expectEqual(exp_01pct, summary.observed_fps_01pct_low);
    try std.testing.expectEqual(eh16, summary.interval_hitches_over_16ms);
    try std.testing.expectEqual(eh33, summary.interval_hitches_over_33ms);
    try std.testing.expectEqual(eh50, summary.interval_hitches_over_50ms);
}
