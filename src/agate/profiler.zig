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

const types = @import("profiler/types.zig");
const report = @import("profiler/report.zig");

// Re-exported so `@import("profiler.zig").FrameRecord` and every existing
// caller keep working; the data types now live in profiler/types.zig.
pub const FrameRecord = types.FrameRecord;
pub const TextureMemoryRecord = types.TextureMemoryRecord;
pub const MeshMemoryRecord = types.MeshMemoryRecord;
pub const RenderTargetRecord = types.RenderTargetRecord;
pub const MemorySnapshot = types.MemorySnapshot;
pub const SessionSummary = types.SessionSummary;
pub const DiagnosticSeverity = types.DiagnosticSeverity;
pub const DiagnosticFinding = types.DiagnosticFinding;
pub const PhaseCulprit = types.PhaseCulprit;
pub const dominantPhase = types.dominantPhase;

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
            .updated_bytes = @intCast(stats.updated_bytes_frame),
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
        var sum_updated_bytes: usize = 0;

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
            sum_updated_bytes += frame.updated_bytes;

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
            .total_updated_bytes = sum_updated_bytes,
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
                        .title = try allocator.dupe(u8, "Задержка из-за стриминга текстур на GPU"),
                        .details = try std.fmt.allocPrint(allocator, "В кадре #{d} стрим текстур (UploadQueue) записал {d:.2} МБ на видеокарту.", .{ worst_frame.frame_index, @as(f32, @floatFromInt(worst_frame.uploaded_bytes)) / (1024.0 * 1024.0) }),
                        .recommendation = try allocator.dupe(u8, "Используйте асинхронную очередь Scene.uploads (UploadQueue) с лимитом байт на кадр (max_bytes_per_frame)."),
                    });
                }

                // Порог 16 MiB = 2x текстурного бюджета: легитимные массовые
                // изменения (полная перезаливка инстансов/морфов) дают единицы
                // МБ (100k инстансов x 64 Б = 6.4 МБ), а UI/debug/трейлы —
                // десятки-сотни КБ. Выше — патологическая перезапись динамики.
                if (worst_frame.updated_bytes > 16 * 1024 * 1024) {
                    try findings.append(allocator, .{
                        .severity = .warning,
                        .title = try allocator.dupe(u8, "Массовые динамические обновления GPU-буферов"),
                        .details = try std.fmt.allocPrint(allocator, "В кадре #{d} динамические буферы (sg.updateBuffer: инстансы, морфы, частицы) записали {d:.2} МБ.", .{ worst_frame.frame_index, @as(f32, @floatFromInt(worst_frame.updated_bytes)) / (1024.0 * 1024.0) }),
                        .recommendation = try allocator.dupe(u8, "Проверьте частоту полных перезаливок instance-буферов (dedup по instance_hash уже пропускает неизменные), вес CPU-морфов и число активных частиц."),
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
        return report.formatBytes(allocator, bytes);
    }

    /// Generates a standalone, beautiful, dark-themed HTML report.
    pub fn generateReportHtml(self: *const Profiler, scene: ?*const Scene, allocator: std.mem.Allocator) ![]u8 {
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

        return report.generateReportHtml(self.frames.items, summary, findings, memory_ptr, allocator);
    }

    /// Generates a comprehensive Markdown report.
    pub fn generateReportMd(self: *const Profiler, scene: ?*const Scene, allocator: std.mem.Allocator) ![]u8 {
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

        return report.generateReportMd(self.frames.items, summary, findings, memory_ptr, allocator);
    }

    /// Generates Chrome Trace Event JSON for chrome://tracing and ui.perfetto.dev.
    pub fn generateTraceJson(self: *const Profiler, allocator: std.mem.Allocator) ![]u8 {
        return report.generateTraceJson(self.frames.items, allocator);
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

test "Profiler переносит динамику буферов отдельно от текстур" {
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();
    prof.start();
    defer prof.stop();

    // Кадр 1: только текстуры; кадр 2: только динамика — метрики не смешиваются.
    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 1.0,
        .shadow_ms = 1.0,
        .main_ms = 5.0,
        .post_ms = 1.0,
        .uploaded_bytes_frame = 1024,
        .updated_bytes_frame = 0,
    };
    prof.recordFrame(1, &stats);
    stats.uploaded_bytes_frame = 0;
    stats.updated_bytes_frame = 2048;
    prof.recordFrame(2, &stats);

    try std.testing.expectEqual(@as(usize, 1024), prof.frames.items[0].uploaded_bytes);
    try std.testing.expectEqual(@as(usize, 0), prof.frames.items[0].updated_bytes);
    try std.testing.expectEqual(@as(usize, 0), prof.frames.items[1].uploaded_bytes);
    try std.testing.expectEqual(@as(usize, 2048), prof.frames.items[1].updated_bytes);

    const summary = prof.summarize();
    try std.testing.expectEqual(@as(usize, 1024), summary.total_uploaded_bytes);
    try std.testing.expectEqual(@as(usize, 2048), summary.total_updated_bytes);
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
