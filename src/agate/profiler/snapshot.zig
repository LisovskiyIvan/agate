//! Profiler memory snapshots. Split out of `profiler.zig` (facade).
//!
//! `captureMemorySnapshot` takes the profiler as `anytype` (a `*Profiler`
//! from `core.zig` in practice) so this module never imports `core.zig` or
//! the facade back — same discipline as `ui/*` taking a generic canvas.
//! `core.zig` owns the `Profiler` type and forwards here. `Scene` comes
//! straight from `../scene.zig`, like `serialization/writer.zig` does.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const types = @import("types.zig");
const scene_mod = @import("../scene.zig");
const texture_mod = @import("../texture.zig");
const probe_layer = @import("../scene/probe_layer.zig");
const gui3d_layer = @import("../scene/gui3d_layer.zig");
const shadow_pass = @import("../passes/shadow_pass.zig");
const glow_pass_mod = @import("../passes/glow_pass.zig");
const highlight_pass_mod = @import("../passes/highlight_pass.zig");

const Scene = scene_mod.Scene;
const Texture = texture_mod.Texture;
const CubeTexture = texture_mod.CubeTexture;
const SHADOW_ATLAS_SIZE = shadow_pass.SHADOW_ATLAS_SIZE;
const SPOT_SHADOW_MAP_WIDTH = shadow_pass.SPOT_SHADOW_MAP_WIDTH;
const SPOT_SHADOW_MAP_HEIGHT = shadow_pass.SPOT_SHADOW_MAP_HEIGHT;
const MemorySnapshot = types.MemorySnapshot;
const TextureMemoryRecord = types.TextureMemoryRecord;
const MeshMemoryRecord = types.MeshMemoryRecord;
const RenderTargetRecord = types.RenderTargetRecord;

fn isCleanAscii(str: []const u8) bool {
    if (str.len == 0) return false;
    for (str) |c| {
        if (!std.ascii.isPrint(c)) return false;
    }
    return true;
}

/// Captures a complete snapshot of CPU memory and GPU VRAM allocations in the scene.
pub fn captureMemorySnapshot(self: anytype, scene: *const Scene) !*const MemorySnapshot {
    const dir_count: usize = scene.lights.directionalCount();
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

        // Glow layer targets (extract + H/V ping-pong, half res): pass-owned
        // like the bloom pyramid (pure byte math via GlowPass.targetBytes, no
        // GPU calls; a sized-out pass contributes nothing). The bloom pyramid
        // itself has no census entry (predates the census); glow registers
        // from day one.
        const glow = &scene.postfx.glow_pass;
        if (glow.base_width > 0 and glow.base_height > 0) {
            const glow_bytes = glow_pass_mod.GlowPass.targetBytes(glow.base_width, glow.base_height, glow_pass_mod.GlowPass.glowBytesPerPixel());
            snap.render_targets_vram_bytes += glow_bytes;
            try rt_list.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, "Glow Targets (Extract + Blur Ping-Pong)"),
                .width = @intCast(glow.base_width),
                .height = @intCast(glow.base_height),
                .format = .RGBA8,
                .samples = 1,
                .gpu_bytes = glow_bytes,
            });
        }

        // Highlight layer targets (mask + H/V ping-pong, half res):
        // pass-owned like the glow layer above (pure byte math via
        // HighlightPass.targetBytes — the same three-target shape, no GPU
        // calls; an unsized pass contributes nothing). The targets are
        // LAZY (allocated on the first active render, never by
        // resizeAll/beginMainPass), so this gate keeps the census honest:
        // zero-highlight frames report no highlight VRAM.
        const hl = &scene.postfx.highlight_pass;
        if (hl.base_width > 0 and hl.base_height > 0) {
            const hl_bytes = highlight_pass_mod.HighlightPass.targetBytes(hl.base_width, hl.base_height, glow_pass_mod.GlowPass.glowBytesPerPixel());
            snap.render_targets_vram_bytes += hl_bytes;
            try rt_list.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, "Highlight Targets (Mask + Blur Ping-Pong)"),
                .width = @intCast(hl.base_width),
                .height = @intCast(hl.base_height),
                .format = .RGBA8,
                .samples = 1,
                .gpu_bytes = hl_bytes,
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

        // Reflection-probe cube targets (wave 25): one mipmapped color cube
        // plus its depth target per captured probe. Same render-target
        // convention as the passes above (pure byte math, no GPU calls;
        // uncaptured probes own no target and contribute nothing).
        for (0..scene.probes.count) |i| {
            const probe = &scene.probes.probes[i];
            if (probe.gpu.image.id == 0) continue;
            const probe_bytes = probe_layer.targetBytes();
            snap.render_targets_vram_bytes += probe_bytes;
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "Reflection Probe {d} (cube+mips+depth)", .{i}) catch "Reflection Probe";
            try rt_list.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, name),
                .width = @intCast(probe_layer.face_resolution),
                .height = @intCast(probe_layer.face_resolution),
                .format = .RGBA8,
                .samples = 1,
                .gpu_bytes = probe_bytes,
            });
        }

        // 3D-GUI panel targets (wave 28): one RGBA8 color RT per panel with
        // a live target (probe precedent: pure byte math, no GPU calls;
        // panels without a target contribute nothing).
        for (0..scene.gui3d.panelCount()) |i| {
            const panel = &scene.gui3d.panels[i];
            if (panel.gpu.target.image.id == 0) continue;
            const panel_bytes = gui3d_layer.targetBytes(panel.canvas_width, panel.canvas_height);
            snap.render_targets_vram_bytes += panel_bytes;
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "3D GUI Panel {d} (color)", .{i}) catch "3D GUI Panel";
            try rt_list.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, name),
                .width = @intCast(panel.canvas_width),
                .height = @intCast(panel.canvas_height),
                .format = .RGBA8,
                .samples = 1,
                .gpu_bytes = panel_bytes,
            });
        }
    }

    snap.render_targets = try rt_list.toOwnedSlice(self.allocator);
    snap.total_gpu_vram_bytes = snap.textures_vram_bytes + snap.meshes_vram_bytes + snap.render_targets_vram_bytes;

    // P7 prepared-draw slots: retained CPU capacity across ALL rotation
    // slots (same convention as the mesh CPU census above — capacities, not
    // lengths, because retention is the cost). One more slot of prepared
    // data appears here exactly like the existing ones.
    snap.prepared_draws_cpu_bytes = scene.draws.cpuBytes();
    snap.prepared_draws_slots = scene.draws.slots.len;

    // Store as last snapshot
    if (self.last_memory_snapshot) |*old| old.deinit(self.allocator);
    self.last_memory_snapshot = snap;

    return &self.last_memory_snapshot.?;
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "Profiler memory snapshot and file saving" {
    const Profiler = @import("core.zig").Profiler;
    const Mesh = @import("../mesh.zig").Mesh;
    sokol.time.setup();
    const ally = std.testing.allocator;
    const testScene = @import("../testing.zig").testScene;
    var scene = testScene(ally);
    defer {
        @import("../scene/content.zig").deinitMeshes(ally, &scene.meshes);
        scene.profiler.deinit();
    }

    const m = try ally.create(Mesh);
    m.* = @import("../testing.zig").testMesh("TestCube");
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
