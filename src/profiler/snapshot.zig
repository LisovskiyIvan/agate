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
const volumetric_pass_mod = @import("../passes/volumetric_pass.zig");
const bloom_mod = @import("../postprocess/bloom.zig");

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

/// Estimated bytes for the HDR main target shape (pure, no sg calls):
/// RGBA16F color (8 B/px) + depth (4 B/px), both scaled by samples, plus
/// one 1x RGBA16F resolve copy under MSAA.
pub fn mainTargetBytes(width: u32, height: u32, samples: u32) usize {
    const px: usize = @as(usize, width) * @as(usize, height);
    const s: usize = @as(usize, @max(1, samples));
    var total: usize = px * 8 * s + px * 4 * s;
    if (s > 1) total += px * 8;
    return total;
}

/// Estimated bytes for the TAA history ping-pong (pure): 2 full-size
/// RGBA16F slots (8 B/px each).
pub fn taaHistoryBytes(width: u32, height: u32) usize {
    return @as(usize, width) * @as(usize, height) * 8 * 2;
}

/// Estimated bytes for the bloom pyramid (pure): down + up chains over the
/// actual per-mip dims (bloomMipSize floors odd sizes, never a coarse
/// base-size scaling).
pub fn bloomPyramidBytes(base_w: i32, base_h: i32, mip_count: usize, bytes_per_pixel: usize) usize {
    var total: usize = 0;
    for (0..mip_count) |i| {
        const s = bloom_mod.bloomMipSize(base_w, base_h, @intCast(i));
        total += @as(usize, @intCast(@max(1, s.w))) * @as(usize, @intCast(@max(1, s.h))) * bytes_per_pixel;
    }
    return total * 2; // down + up
}

/// Estimated bytes for one live render-target image desc (pure): the full
/// mip chain at `bpp`, scaled by the sample count. Single-level targets
/// (the bloom pyramid shape) collapse to w*h*bpp*samples.
pub fn rtImageBytes(width: i32, height: i32, num_mipmaps: i32, sample_count: i32, bpp: usize) usize {
    if (width <= 0 or height <= 0) return 0;
    var w: usize = @intCast(width);
    var h: usize = @intCast(height);
    const s: usize = @as(usize, @intCast(@max(1, sample_count)));
    const levels: usize = @intCast(@max(1, num_mipmaps));
    var total: usize = 0;
    var i: usize = 0;
    while (i < levels) : (i += 1) {
        total += w * h * bpp * s;
        if (w == 1 and h == 1) break;
        w = @max(1, w / 2);
        h = @max(1, h / 2);
    }
    return total;
}

/// Leak-safe render-target append: the duped name frees on append OOM.
fn appendRt(
    allocator: std.mem.Allocator,
    list: *std.ArrayListUnmanaged(RenderTargetRecord),
    vram_sum: *usize,
    name: []const u8,
    width: u32,
    height: u32,
    format: sg.PixelFormat,
    samples: u32,
    bytes: usize,
) !void {
    const owned = try allocator.dupe(u8, name);
    errdefer allocator.free(owned);
    try list.append(allocator, .{
        .name = owned,
        .width = width,
        .height = height,
        .format = format,
        .samples = samples,
        .gpu_bytes = bytes,
    });
    vram_sum.* += bytes;
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
        .material_count = scene.pbr_materials.items.len,
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
            const color_bpp: usize = 8; // RGBA16F sole HDR contract
            const depth_bpp: usize = 4;

            // Offscreen Color (HDR)
            const color_bytes = @as(usize, w) * h * color_bpp * samples;
            try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "Main Offscreen Color Target (HDR)", w, h, .RGBA16F, samples, color_bytes);

            // Offscreen Depth
            const depth_bytes = @as(usize, w) * h * depth_bpp * samples;
            try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "Main Offscreen Depth Target", w, h, .DEPTH, samples, depth_bytes);

            // Resolve Color if MSAA
            if (samples > 1) {
                const resolve_bytes = @as(usize, w) * h * color_bpp;
                try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "MSAA Resolve Target (HDR)", w, h, .RGBA16F, 1, resolve_bytes);
            }

            // TAA history ping-pong: only when both slots are really VALID
            // (state query, not id-only: FAILED-but-nonzero counts nothing).
            if (pp.taaAvailable()) {
                const tw: u32 = @intCast(pp.taa_width);
                const th: u32 = @intCast(pp.taa_height);
                const hist_bytes = taaHistoryBytes(tw, th);
                try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "TAA History (Ping-Pong 2x HDR)", tw, th, .RGBA16F, 1, hist_bytes);
            }
        }

        // Bloom pyramid: sum the ACTUAL allocated down+up images (VALID
        // only, desc dims/mips/samples at 8 B/px) into one named 16F entry.
        // Never a free-allocation or configured-mip estimate: unallocated
        // slots and FAILED handles contribute nothing.
        {
            const bloom = &scene.postfx.bloom_pass;
            if (bloom.base_width > 0 and bloom.base_height > 0) {
                var bloom_bytes: usize = 0;
                for (bloom.down_images) |img| {
                    if (sg.queryImageState(img) != .VALID) continue;
                    const d = sg.queryImageDesc(img);
                    bloom_bytes += rtImageBytes(d.width, d.height, d.num_mipmaps, d.sample_count, 8);
                }
                for (bloom.up_images) |img| {
                    if (sg.queryImageState(img) != .VALID) continue;
                    const d = sg.queryImageDesc(img);
                    bloom_bytes += rtImageBytes(d.width, d.height, d.num_mipmaps, d.sample_count, 8);
                }
                if (bloom_bytes > 0) {
                    try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "Bloom Targets (Down+Up Pyramid HDR)", @intCast(bloom.base_width), @intCast(bloom.base_height), .RGBA16F, 1, bloom_bytes);
                }
            }
        }

        // SSAO Target
        const ssao = &scene.postfx.ssao_pass;
        if (ssao.width > 0 and ssao.height > 0) {
            const ssao_bytes = @as(usize, @intCast(ssao.width)) * @as(usize, @intCast(ssao.height)) * 4 * 2; // raw + blur
            try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "SSAO Targets (Raw + Blur)", @intCast(ssao.width), @intCast(ssao.height), .RGBA8, 1, ssao_bytes);
        }

        // Glow layer targets (extract + H/V ping-pong, half res): pass-owned
        // like the bloom pyramid (pure byte math via GlowPass.targetBytes, no
        // GPU calls; a sized-out pass contributes nothing). The bloom pyramid
        // itself has no census entry (predates the census); glow registers
        // from day one.
        const glow = &scene.postfx.glow_pass;
        if (glow.base_width > 0 and glow.base_height > 0) {
            const glow_bytes = glow_pass_mod.GlowPass.targetBytes(glow.base_width, glow.base_height, glow_pass_mod.GlowPass.glowBytesPerPixel());
            try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "Glow Targets (Extract + Blur Ping-Pong)", @intCast(glow.base_width), @intCast(glow.base_height), .RGBA16F, 1, glow_bytes);
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
            try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "Highlight Targets (Mask + Blur Ping-Pong)", @intCast(hl.base_width), @intCast(hl.base_height), .RGBA16F, 1, hl_bytes);
        }

        // Volumetric shaft targets (raymarch + bilateral H/V ping-pong,
        // half/quarter res): pass-owned like the layers above (pure byte
        // math via VolumetricPass.targetBytes, no GPU calls; an unsized
        // pass contributes nothing). The targets are LAZY (allocated on
        // the first active render, never by resizeAll/beginMainPass), so
        // this gate keeps the census honest: shaft-off frames report no
        // shaft VRAM.
        const shaft = &scene.postfx.volumetric_pass;
        if (shaft.base_width > 0 and shaft.base_height > 0) {
            const shaft_bytes = volumetric_pass_mod.VolumetricPass.targetBytes(shaft.base_width, shaft.base_height, volumetric_pass_mod.VolumetricPass.shaftBytesPerPixel(), shaft.resolution);
            try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "Volumetric Shaft Targets (Raymarch + Blur Ping-Pong)", @intCast(shaft.base_width), @intCast(shaft.base_height), .RGBA16F, 1, shaft_bytes);
        }

        // Shadow Atlas
        const shadow_atlas_bytes = @as(usize, SHADOW_ATLAS_SIZE) * SHADOW_ATLAS_SIZE * 4;
        try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "CSM Directional Shadow Atlas", SHADOW_ATLAS_SIZE, SHADOW_ATLAS_SIZE, .DEPTH, 1, shadow_atlas_bytes);

        // Spot Shadow Atlas
        const spot_bytes = @as(usize, SPOT_SHADOW_MAP_WIDTH) * SPOT_SHADOW_MAP_HEIGHT * 4;
        try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "Spot Light Shadow Map", SPOT_SHADOW_MAP_WIDTH, SPOT_SHADOW_MAP_HEIGHT, .DEPTH, 1, spot_bytes);

        // Reflection-probe cube targets: one mipmapped color cube
        // plus its depth target per captured probe. Same render-target
        // convention as the passes above (pure byte math, no GPU calls;
        // uncaptured probes own no target and contribute nothing).
        for (0..scene.probes.count) |i| {
            const probe = &scene.probes.probes[i];
            if (sg.queryImageState(probe.gpu.image) != .VALID) continue;
            const probe_bytes = probe_layer.targetBytes();
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "Reflection Probe {d} (cube+mips+depth)", .{i}) catch "Reflection Probe";
            try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, name, @intCast(probe_layer.face_resolution), @intCast(probe_layer.face_resolution), .RGBA16F, 1, probe_bytes);
        }

        // 3D-GUI panel targets: one RGBA8 color RT per panel with
        // a live target (probe precedent: pure byte math, no GPU calls;
        // panels without a target contribute nothing).
        for (0..scene.gui3d.panelCount()) |i| {
            const panel = &scene.gui3d.panels[i];
            if (sg.queryImageState(panel.gpu.target.image) != .VALID) continue;
            const panel_bytes = gui3d_layer.targetBytes(panel.canvas_width, panel.canvas_height);
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "3D GUI Panel {d} (color)", .{i}) catch "3D GUI Panel";
            try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, name, @intCast(panel.canvas_width), @intCast(panel.canvas_height), .RGBA8, 1, panel_bytes);
        }

        // Refraction background target (half-res HDR + depth): only when
        // the target is really allocated (isValid + VALID color image).
        {
            const rt = &scene.refraction.target;
            if (rt.isValid() and sg.queryImageState(rt.color_image) == .VALID) {
                const refr_bytes = rt.estimatedBytes();
                if (refr_bytes > 0) {
                    const rs: u32 = @intCast(@max(1, rt.sample_count));
                    try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "Refraction Background Target (Half-Res HDR + Depth)", rt.width, rt.height, .RGBA16F, rs, refr_bytes);
                }
            }
        }

        // MSAA depth prepass target (1x depth, 4 B/px): only when really
        // allocated (VALID image, no other new resources).
        if (scene.postfx.msaa_depth) |*md| {
            if (md.width > 0 and md.height > 0 and sg.queryImageState(md.image) == .VALID) {
                const mw: u32 = @intCast(md.width);
                const mh: u32 = @intCast(md.height);
                const md_bytes = @as(usize, mw) * mh * 4;
                try appendRt(self.allocator, &rt_list, &snap.render_targets_vram_bytes, "MSAA Depth Prepass Target", mw, mh, .DEPTH, 1, md_bytes);
            }
        }
    }

    snap.render_targets = try rt_list.toOwnedSlice(self.allocator);
    snap.total_gpu_vram_bytes = snap.textures_vram_bytes + snap.meshes_vram_bytes + snap.render_targets_vram_bytes;

    // Prepared-draw slots: retained CPU capacity across ALL rotation
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
