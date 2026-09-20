//! Cube texture type and converters. Split out of `texture.zig` (facade).
//!
//! This module owns `SkyboxOptions` and the `CubeTexture` type: faces,
//! equirectangular (LDR + HDR) converters, the HDR cube mip chain, and the
//! procedural skybox. `Texture` references below resolve to
//! `core.zig` (same call sites as before the split).
//!
//! Anti-cycle rule (same as `camera/`, `profiler/`, `ui/`): this module
//! never imports the `texture.zig` facade back.
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const c = @import("../c.zig").c;
const core = @import("core.zig");
const Texture = core.Texture;
const mip = @import("mip.zig");

/// Gradient + sun parameters for `CubeTexture.createProceduralSkybox`.
pub const SkyboxOptions = struct {
    top_color: [4]u8 = .{ 35, 75, 155, 255 }, // Deep blue zenith
    horizon_color: [4]u8 = .{ 175, 195, 220, 255 }, // Atmospheric horizon
    bottom_color: [4]u8 = .{ 45, 42, 40, 255 }, // Ground nadir
    sun_direction: [3]f32 = .{ 0.4, 0.7, 0.5 }, // Matching scene light
    sun_color: [4]u8 = .{ 255, 250, 220, 255 },
    sun_intensity: f32 = 2.5,
    size: u32 = 128,
};

pub const CubeTexture = struct {
    image: sg.Image,
    view: sg.View,
    sampler: sg.Sampler,
    size: u32,
    num_mipmaps: u32 = 1,
    /// Pixel format of the GPU cube image. Defaults keep LDR call sites
    /// unchanged; HDR uploaders set RGBA16F.
    format: sg.PixelFormat = .RGBA8,
    /// True for float (HDR) cubes: no gamma correction, linear sampling.
    is_hdr: bool = false,

    pub fn deinit(self: *CubeTexture) void {
        sg.destroyView(self.view);
        sg.destroyImage(self.image);
        sg.destroySampler(self.sampler);
    }

    /// Total estimated GPU memory in bytes for this cube texture (all 6 faces, including mips).
    pub fn getGpuMemoryBytes(self: *const CubeTexture) usize {
        if (self.image.id == 0) return 0;
        const bpp = mip.pixelFormatBytes(self.format);
        var total: usize = 0;
        var s = @max(1, self.size);
        const mips = @max(1, self.num_mipmaps);
        for (0..mips) |_| {
            total += @as(usize, s) * @as(usize, s) * bpp * 6;
            if (s == 1) break;
            s = @max(1, s / 2);
        }
        return total;
    }

    pub fn createDefault1x1(color: [4]u8) CubeTexture {
        var raw: [24]u8 = undefined;
        for (0..6) |face| {
            raw[face * 4 + 0] = color[0];
            raw[face * 4 + 1] = color[1];
            raw[face * 4 + 2] = color[2];
            raw[face * 4 + 3] = color[3];
        }

        var img_desc = sg.ImageDesc{
            .type = .CUBE,
            .width = 1,
            .height = 1,
            .num_slices = 6,
            .num_mipmaps = 1,
            .pixel_format = .RGBA8,
            .sample_count = 1,
        };
        img_desc.data.mip_levels[0] = sg.asRange(&raw);

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .size = 1,
            .num_mipmaps = 1,
        };
    }

    pub fn initRawFaces(allocator: std.mem.Allocator, size: u32, faces: [6][]const u8, generate_mips: bool) !CubeTexture {
        const face_bytes = try mip.checkedFaceBytes(size);
        for (faces) |face| {
            if (face.len != face_bytes) return error.InvalidFaceBufferSize;
        }

        // Concatenate 6 faces for mip 0
        const mip0_total_bytes = std.math.mul(usize, face_bytes, 6) catch return error.ImageTooLarge;
        const mip0_buffer = try allocator.alloc(u8, mip0_total_bytes);
        defer allocator.free(mip0_buffer);

        for (faces, 0..) |face, i| {
            const dst_offset = i * face_bytes;
            @memcpy(mip0_buffer[dst_offset .. dst_offset + face_bytes], face);
        }

        var num_mips: u32 = 1;
        if (generate_mips and size > 1) {
            // Same chain length as the 2D path; identical to the old
            // divide-by-2 loop for powers of two, NPOT-safe otherwise.
            num_mips = Texture.mipLevelCount(size, size);
        }
        if (num_mips > 16) num_mips = 16;

        var img_desc = sg.ImageDesc{
            .type = .CUBE,
            .width = @intCast(size),
            .height = @intCast(size),
            .num_slices = 6,
            .num_mipmaps = @intCast(num_mips),
            .pixel_format = .RGBA8,
            .sample_count = 1,
        };
        img_desc.data.mip_levels[0] = sg.asRange(mip0_buffer);

        // Generate downsampled mipmaps
        var mip_buffers: [16]?[]u8 = @splat(null);
        defer {
            for (1..num_mips) |m| {
                if (mip_buffers[m]) |buf| allocator.free(buf);
            }
        }

        var prev_size = size;
        var prev_buf = mip0_buffer;
        for (1..num_mips) |m| {
            // @max(1, ...) keeps NPOT chains alive (e.g. 3 -> 1) instead of
            // hitting a zero-sized level; POT chains are unchanged.
            const cur_size: u32 = @max(1, prev_size / 2);
            const cur_face_checked = try mip.checkedFaceBytes(cur_size);
            const cur_total_bytes = std.math.mul(usize, cur_face_checked, 6) catch return error.ImageTooLarge;
            const cur_buf = try allocator.alloc(u8, cur_total_bytes);
            mip_buffers[m] = cur_buf;

            const prev_face_bytes: usize = @as(usize, prev_size) * @as(usize, prev_size) * 4;
            const cur_face_bytes: usize = cur_face_checked;
            for (0..6) |face| {
                const src_face = prev_buf[face * prev_face_bytes .. (face + 1) * prev_face_bytes];
                const dst_face = cur_buf[face * cur_face_bytes .. (face + 1) * cur_face_bytes];
                // Shared NPOT-safe box filter; bit-identical to the old
                // inline loop when prev == cur*2 (all POT levels).
                mip.boxDownsampleU8(src_face, prev_size, prev_size, dst_face, cur_size, cur_size);
            }

            img_desc.data.mip_levels[m] = sg.asRange(cur_buf);
            prev_size = cur_size;
            prev_buf = cur_buf;
        }

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .mipmap_filter = if (num_mips > 1) .LINEAR else .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .size = size,
            .num_mipmaps = num_mips,
        };
    }

    pub fn createProceduralSkybox(allocator: std.mem.Allocator, config: SkyboxOptions) !CubeTexture {
        const size = config.size;
        const face_bytes = try mip.checkedFaceBytes(size);

        var face_slices: [6][]u8 = undefined;
        var allocated_faces: usize = 0;
        defer for (face_slices[0..allocated_faces]) |s| allocator.free(s);
        for (0..6) |i| {
            face_slices[i] = try allocator.alloc(u8, face_bytes);
            allocated_faces += 1;
        }

        // Normalize sun direction
        const s_len = @sqrt(config.sun_direction[0] * config.sun_direction[0] +
            config.sun_direction[1] * config.sun_direction[1] +
            config.sun_direction[2] * config.sun_direction[2]);
        const sun_dir: [3]f32 = if (s_len > 0.0001) .{
            config.sun_direction[0] / s_len,
            config.sun_direction[1] / s_len,
            config.sun_direction[2] / s_len,
        } else .{ 0.0, 1.0, 0.0 };

        for (0..6) |face| {
            const buf = face_slices[face];
            var py: u32 = 0;
            while (py < size) : (py += 1) {
                const v = 2.0 * (@as(f32, @floatFromInt(py)) + 0.5) / @as(f32, @floatFromInt(size)) - 1.0;
                var px: u32 = 0;
                while (px < size) : (px += 1) {
                    const u = 2.0 * (@as(f32, @floatFromInt(px)) + 0.5) / @as(f32, @floatFromInt(size)) - 1.0;

                    const dir: [3]f32 = switch (face) {
                        0 => .{ 1.0, -v, -u }, // +X
                        1 => .{ -1.0, -v, u }, // -X
                        2 => .{ u, 1.0, v }, // +Y
                        3 => .{ u, -1.0, -v }, // -Y
                        4 => .{ u, -v, 1.0 }, // +Z
                        5 => .{ -u, -v, -1.0 }, // -Z
                        else => unreachable,
                    };

                    const d_len = @sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
                    const d: [3]f32 = .{ dir[0] / d_len, dir[1] / d_len, dir[2] / d_len };

                    // Elevation gradient (d[1] is Y)
                    var color: [3]f32 = undefined;
                    if (d[1] >= 0.0) {
                        const t = std.math.pow(f32, d[1], 0.65);
                        for (0..3) |c_idx| {
                            const horiz = @as(f32, @floatFromInt(config.horizon_color[c_idx])) / 255.0;
                            const top = @as(f32, @floatFromInt(config.top_color[c_idx])) / 255.0;
                            color[c_idx] = (1.0 - t) * horiz + t * top;
                        }
                    } else {
                        const t = std.math.pow(f32, -d[1], 0.7);
                        for (0..3) |c_idx| {
                            const horiz = @as(f32, @floatFromInt(config.horizon_color[c_idx])) / 255.0;
                            const bot = @as(f32, @floatFromInt(config.bottom_color[c_idx])) / 255.0;
                            color[c_idx] = (1.0 - t) * horiz + t * bot;
                        }
                    }

                    // Sun contribution
                    const sun_dot = d[0] * sun_dir[0] + d[1] * sun_dir[1] + d[2] * sun_dir[2];
                    if (sun_dot > 0.0) {
                        const sun_disk = std.math.pow(f32, sun_dot, 512.0) * 3.5;
                        const sun_halo = std.math.pow(f32, sun_dot, 16.0) * 0.45;
                        const total_sun = (sun_disk + sun_halo) * config.sun_intensity;

                        for (0..3) |c_idx| {
                            const sc = @as(f32, @floatFromInt(config.sun_color[c_idx])) / 255.0;
                            color[c_idx] += total_sun * sc;
                        }
                    }

                    const out_idx: usize = (@as(usize, py) * @as(usize, size) + @as(usize, px)) * 4;
                    inline for (0..3) |c_idx| {
                        const byte_val = std.math.clamp(color[c_idx] * 255.0, 0.0, 255.0);
                        buf[out_idx + c_idx] = @intFromFloat(byte_val);
                    }
                    buf[out_idx + 3] = 255;
                }
            }
        }

        const const_faces = [6][]const u8{
            face_slices[0],
            face_slices[1],
            face_slices[2],
            face_slices[3],
            face_slices[4],
            face_slices[5],
        };

        return initRawFaces(allocator, size, const_faces, true);
    }

    pub fn fromFiles(allocator: std.mem.Allocator, face_paths: [6][]const u8) !CubeTexture {
        var face_data: [6][*c]u8 = undefined;
        // Single ownership scheme: `loaded` counts fully validated faces.
        // The defer frees exactly those on every exit — error returns after
        // a `try` (e.g. dupeZ OOM) or decode failure, and the success path
        // after initRawFaces has copied the pixels for GPU upload.
        var loaded: usize = 0;
        defer for (face_data[0..loaded]) |d| c.stbi_image_free(d);
        var size: u32 = 0;

        for (face_paths, 0..) |path, i| {
            const path_z = try allocator.dupeZ(u8, path);
            defer allocator.free(path_z);

            var w: c_int = 0;
            var h: c_int = 0;
            var comp: c_int = 0;
            const data = c.stbi_load(path_z.ptr, &w, &h, &comp, 4);
            if (data == null) {
                return error.ImageDecodeFailed;
            }
            if (w <= 0 or h <= 0) {
                c.stbi_image_free(data);
                return error.InvalidDimensions;
            }
            if (w != h) {
                c.stbi_image_free(data);
                return error.CubeFaceMustBeSquare;
            }
            if (i == 0) {
                size = @intCast(w);
            } else if (@as(u32, @intCast(w)) != size) {
                c.stbi_image_free(data);
                return error.CubeFacesMustHaveEqualSize;
            }
            face_data[i] = data;
            loaded += 1;
        }

        const face_bytes = try mip.checkedFaceBytes(size);
        var const_faces: [6][]const u8 = undefined;
        for (0..6) |i| {
            const_faces[i] = face_data[i][0..face_bytes];
        }

        return initRawFaces(allocator, size, const_faces, true);
    }

    pub fn fromEquirectangular(allocator: std.mem.Allocator, panorama_bytes: []const u8, face_size: u32) !CubeTexture {
        // Same guard as the HDR converter: zero face size would divide by
        // zero in cubeTexelDirection and produce zero-byte faces.
        if (face_size == 0) return error.InvalidDimensions;
        var pw: c_int = 0;
        var ph: c_int = 0;
        var comp: c_int = 0;
        const p_data = c.stbi_load_from_memory(
            panorama_bytes.ptr,
            @intCast(panorama_bytes.len),
            &pw,
            &ph,
            &comp,
            4,
        );
        if (p_data == null) return error.ImageDecodeFailed;
        defer c.stbi_image_free(p_data);
        if (pw <= 0 or ph <= 0) return error.InvalidDimensions;

        const pano_w: u32 = @intCast(pw);
        const pano_h: u32 = @intCast(ph);
        const face_bytes = try mip.checkedFaceBytes(face_size);

        var face_slices: [6][]u8 = undefined;
        var allocated_faces: usize = 0;
        defer for (face_slices[0..allocated_faces]) |s| allocator.free(s);
        for (0..6) |i| {
            face_slices[i] = try allocator.alloc(u8, face_bytes);
            allocated_faces += 1;
        }

        for (0..6) |face| {
            const buf = face_slices[face];
            var py: u32 = 0;
            while (py < face_size) : (py += 1) {
                var px: u32 = 0;
                while (px < face_size) : (px += 1) {
                    const d = cubeTexelDirection(face, px, py, face_size);
                    const t = panoramaTexel(d, pano_w, pano_h);
                    const p_idx: usize = (@as(usize, t.y) * @as(usize, pano_w) + @as(usize, t.x)) * 4;
                    const out_idx: usize = (@as(usize, py) * @as(usize, face_size) + @as(usize, px)) * 4;

                    @memcpy(buf[out_idx .. out_idx + 4], p_data[p_idx .. p_idx + 4]);
                }
            }
        }

        const const_faces = [6][]const u8{
            face_slices[0],
            face_slices[1],
            face_slices[2],
            face_slices[3],
            face_slices[4],
            face_slices[5],
        };

        return initRawFaces(allocator, face_size, const_faces, true);
    }

    pub fn fromEquirectangularFile(allocator: std.mem.Allocator, file_path: []const u8, face_size: u32) !CubeTexture {
        if (face_size == 0) return error.InvalidDimensions;
        // Zig 0.16 removed std.fs.cwd(); read through the global single-threaded
        // Io so the public (allocator, path) signature stays unchanged.
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
        defer file.close(io);

        const file_size = try file.length(io);
        const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
        defer allocator.free(bytes);

        const read = try file.readPositionalAll(io, bytes, 0);
        if (read < bytes.len) return error.ImageDecodeFailed;
        return fromEquirectangular(allocator, bytes, face_size);
    }

    /// Normalized cube-face direction for texel (px, py). Shared by the LDR
    /// and HDR equirectangular converters so both sample the same texels.
    fn cubeTexelDirection(face: usize, px: u32, py: u32, size: u32) [3]f32 {
        const size_f: f32 = @floatFromInt(size);
        const u = 2.0 * (@as(f32, @floatFromInt(px)) + 0.5) / size_f - 1.0;
        const v = 2.0 * (@as(f32, @floatFromInt(py)) + 0.5) / size_f - 1.0;

        const dir: [3]f32 = switch (face) {
            0 => .{ 1.0, -v, -u }, // +X
            1 => .{ -1.0, -v, u }, // -X
            2 => .{ u, 1.0, v }, // +Y
            3 => .{ u, -1.0, -v }, // -Y
            4 => .{ u, -v, 1.0 }, // +Z
            5 => .{ -u, -v, -1.0 }, // -Z
            else => unreachable,
        };

        const len = @sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        return .{ dir[0] / len, dir[1] / len, dir[2] / len };
    }

    /// Nearest panorama texel for a normalized direction (spherical mapping).
    /// Shared by the LDR and HDR equirectangular converters.
    fn panoramaTexel(d: [3]f32, pano_w: u32, pano_h: u32) struct { x: u32, y: u32 } {
        // Spherical coordinates
        const phi = std.math.atan2(d[2], d[0]); // longitude [-pi, pi]
        const theta = std.math.asin(std.math.clamp(d[1], -1.0, 1.0)); // latitude [-pi/2, pi/2]

        const u_pano = (phi / (2.0 * std.math.pi)) + 0.5;
        const v_pano = 0.5 - (theta / std.math.pi);

        const sx_f = std.math.clamp(u_pano * @as(f32, @floatFromInt(pano_w)), 0.0, @as(f32, @floatFromInt(pano_w - 1)));
        const sy_f = std.math.clamp(v_pano * @as(f32, @floatFromInt(pano_h)), 0.0, @as(f32, @floatFromInt(pano_h - 1)));

        return .{ .x = @intFromFloat(sx_f), .y = @intFromFloat(sy_f) };
    }

    /// GPU-free half-float cube faces converted from an HDR panorama.
    /// Owns six size*size*4 half-float bit patterns; free with deinit.
    /// Single mip level: `initRawFacesHdr` builds the f16 chain from this
    /// level-0 data via `buildRawFacesHdr`.
    pub const RawHdrCube = struct {
        size: u32 = 0,
        faces: [6]?[]u16 = @splat(null),

        pub fn deinit(self: *RawHdrCube, allocator: std.mem.Allocator) void {
            for (&self.faces) |*face| {
                if (face.*) |buf| allocator.free(buf);
                face.* = null;
            }
            self.size = 0;
        }
    };

    /// Converts an f32 RGBA equirectangular panorama to six half-float cube
    /// faces using the same texel math as `fromEquirectangular`. GPU-free and
    /// safe for worker threads; pair with `initRawFacesHdr`. Nearest sampling,
    /// no gamma correction. Rejects zero dimensions and short buffers.
    pub fn convertEquirectangularHDR(
        allocator: std.mem.Allocator,
        width: u32,
        height: u32,
        rgba: []const f32,
        size: u32,
    ) !RawHdrCube {
        if (size == 0 or width == 0 or height == 0) return error.InvalidDimensions;
        const pano_pixels = std.math.mul(usize, width, height) catch return error.ImageTooLarge;
        const pano_channels = std.math.mul(usize, pano_pixels, 4) catch return error.ImageTooLarge;
        if (rgba.len != pano_channels) return error.InvalidPanoramaSize;

        const face_pixels = std.math.mul(usize, std.math.mul(usize, size, size) catch return error.ImageTooLarge, 4) catch return error.ImageTooLarge;

        var raw = RawHdrCube{ .size = size };
        errdefer raw.deinit(allocator);

        for (0..6) |face| {
            const out = try allocator.alloc(u16, face_pixels);
            raw.faces[face] = out;
            var py: u32 = 0;
            while (py < size) : (py += 1) {
                var px: u32 = 0;
                while (px < size) : (px += 1) {
                    const d = cubeTexelDirection(face, px, py, size);
                    const t = panoramaTexel(d, width, height);
                    const p_idx: usize = (@as(usize, t.y) * @as(usize, width) + @as(usize, t.x)) * 4;
                    const out_idx: usize = (@as(usize, py) * @as(usize, size) + @as(usize, px)) * 4;
                    for (0..4) |ch| {
                        out[out_idx + ch] = Texture.floatToHalfBits(rgba[p_idx + ch]);
                    }
                }
            }
        }
        return raw;
    }

    /// GPU-free f16 mip chain for a cube (six faces concatenated per level,
    /// face layout identical to the LDR path). Level 0 is a deep copy of
    /// `faces`; each following level halves the face size via
    /// `boxDownsampleF16`. Same chain length as the 2D path. Used by
    /// `initRawFacesHdr` and tests.
    pub const RawHdrCubeMips = struct {
        size: u32 = 0,
        num_levels: u32 = 0,
        /// levels[m] holds 6 * size_m * size_m * 4 half-float patterns.
        levels: [16]?[]u16 = @splat(null),

        pub fn deinit(self: *RawHdrCubeMips, allocator: std.mem.Allocator) void {
            for (self.levels[0..self.num_levels]) |level| {
                if (level) |buf| allocator.free(buf);
            }
            self.* = .{};
        }
    };

    pub fn buildRawFacesHdr(allocator: std.mem.Allocator, size: u32, faces: [6][]const u16) !RawHdrCubeMips {
        if (size == 0) return error.InvalidDimensions;
        const face_texels = std.math.mul(usize, std.math.mul(usize, size, size) catch return error.ImageTooLarge, 4) catch return error.ImageTooLarge;
        for (faces) |face| {
            if (face.len != face_texels) return error.InvalidFaceBufferSize;
        }

        var chain = RawHdrCubeMips{ .size = size, .num_levels = 1 };
        errdefer chain.deinit(allocator);

        const mip0 = try allocator.alloc(u16, std.math.mul(usize, 6, face_texels) catch return error.ImageTooLarge);
        for (faces, 0..) |face, i| {
            @memcpy(mip0[i * face_texels .. (i + 1) * face_texels], face);
        }
        chain.levels[0] = mip0;

        var prev_size = size;
        const levels = Texture.mipLevelCount(size, size);
        for (1..levels) |m| {
            const cur_size: u32 = @max(1, prev_size / 2);
            const prev_texels: usize = @as(usize, prev_size) * prev_size * 4;
            const cur_texels: usize = @as(usize, cur_size) * cur_size * 4;
            const cur = try allocator.alloc(u16, std.math.mul(usize, 6, cur_texels) catch return error.ImageTooLarge);
            chain.levels[m] = cur;
            chain.num_levels = @intCast(m + 1);

            const prev = chain.levels[m - 1].?;
            for (0..6) |face| {
                mip.boxDownsampleF16(
                    prev[face * prev_texels .. (face + 1) * prev_texels],
                    prev_size,
                    prev_size,
                    cur[face * cur_texels .. (face + 1) * cur_texels],
                    cur_size,
                    cur_size,
                );
            }
            prev_size = cur_size;
        }
        return chain;
    }

    /// Uploads six half-float RGBA faces as an RGBA16F cube WITH a full
    /// CPU-generated mip chain (the PBR IBL path samples
    /// `textureLod(lod <= 7)`, which needs the levels). Main thread only.
    /// Each face must hold size*size*4 half patterns.
    /// Linear min/mag/mip, clamp on all axes, no gamma correction.
    pub fn initRawFacesHdr(allocator: std.mem.Allocator, size: u32, faces: [6][]const u16) !CubeTexture {
        var chain = try buildRawFacesHdr(allocator, size, faces);
        defer chain.deinit(allocator);

        var img_desc = sg.ImageDesc{
            .type = .CUBE,
            .width = @intCast(size),
            .height = @intCast(size),
            .num_slices = 6,
            .num_mipmaps = @intCast(chain.num_levels),
            .pixel_format = .RGBA16F,
            .sample_count = 1,
        };
        for (0..chain.num_levels) |m| {
            if (chain.levels[m]) |level| {
                img_desc.data.mip_levels[m] = sg.asRange(level);
            }
        }

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .mipmap_filter = if (chain.num_levels > 1) .LINEAR else .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .size = size,
            .num_mipmaps = chain.num_levels,
            .format = .RGBA16F,
            .is_hdr = true,
        };
    }

    /// Converts an in-memory f32 RGBA equirectangular panorama to an RGBA16F
    /// cube for IBL/skybox. Main thread only (GPU upload); use
    /// convertEquirectangularHDR alone to stay GPU-free (worker threads).
    pub fn fromEquirectangularHDR(
        allocator: std.mem.Allocator,
        width: u32,
        height: u32,
        rgba: []const f32,
        size: u32,
    ) !CubeTexture {
        var raw = try convertEquirectangularHDR(allocator, width, height, rgba, size);
        defer raw.deinit(allocator);
        var faces: [6][]const u16 = undefined;
        for (0..6) |i| faces[i] = raw.faces[i].?;
        return initRawFacesHdr(allocator, size, faces);
    }

    /// GPU-free pre-built LDR cube levels (six faces concatenated per level,
    /// face order +X,-X,+Y,-Y,+Z,-Z, level 0 largest). Produced by
    /// ktx2.decodeCube; uploaded with `initRawFacesMips`. Free with deinit.
    pub const RawCubeMips = struct {
        size: u32 = 0,
        num_levels: u32 = 0,
        /// levels[m] holds 6 * size_m * size_m * 4 RGBA8 bytes.
        levels: [16]?[]u8 = @splat(null),

        pub fn deinit(self: *RawCubeMips, allocator: std.mem.Allocator) void {
            for (self.levels[0..self.num_levels]) |level| {
                if (level) |buf| allocator.free(buf);
            }
            self.* = .{};
        }
    };

    /// Uploads pre-built LDR cube levels as an RGBA8 cube WITH a full mip
    /// chain as authored (unlike `initRawFaces`, which generates the chain
    /// from level 0). Main thread only. Sampler: linear min/mag/mip, clamp
    /// on all axes — the env/skybox convention of every cube path.
    pub fn initRawFacesMips(raw: *const RawCubeMips) CubeTexture {
        var img_desc = sg.ImageDesc{
            .type = .CUBE,
            .width = @intCast(raw.size),
            .height = @intCast(raw.size),
            .num_slices = 6,
            .num_mipmaps = @intCast(raw.num_levels),
            .pixel_format = .RGBA8,
            .sample_count = 1,
        };
        for (0..raw.num_levels) |m| {
            if (raw.levels[m]) |level| {
                img_desc.data.mip_levels[m] = sg.asRange(level);
            }
        }

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .mipmap_filter = if (raw.num_levels > 1) .LINEAR else .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .size = raw.size,
            .num_mipmaps = raw.num_levels,
        };
    }
};

test "convertEquirectangularHDR maps gradient faces without NaN" {
    const allocator = std.testing.allocator;
    const pano_w: u32 = 4;
    const pano_h: u32 = 2;
    const face_size: u32 = 4;

    var pano: [4 * 2 * 4]f32 = undefined;
    for (0..pano_h) |y| {
        for (0..pano_w) |x| {
            const o = (y * pano_w + x) * 4;
            pano[o + 0] = @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(pano_w - 1));
            pano[o + 1] = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(pano_h - 1));
            pano[o + 2] = 0.5;
            pano[o + 3] = 1.0;
        }
    }

    var cube = try CubeTexture.convertEquirectangularHDR(allocator, pano_w, pano_h, &pano, face_size);
    defer cube.deinit(allocator);

    try std.testing.expectEqual(face_size, cube.size);
    for (cube.faces) |maybe_face| {
        const face = maybe_face.?;
        try std.testing.expectEqual(@as(usize, face_size * face_size * 4), face.len);
        for (face) |bits| {
            const v = Texture.halfBitsToFloat(bits);
            try std.testing.expect(!std.math.isNan(v));
            try std.testing.expect(std.math.isFinite(v));
        }
    }
}

test "convertEquirectangularHDR validates dimensions and buffer size" {
    const allocator = std.testing.allocator;
    var pano: [2 * 1 * 4]f32 = [_]f32{0.0} ** (2 * 1 * 4);

    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.convertEquirectangularHDR(allocator, 0, 1, &pano, 2),
    );
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.convertEquirectangularHDR(allocator, 2, 1, &pano, 0),
    );
    try std.testing.expectError(
        error.InvalidPanoramaSize,
        CubeTexture.convertEquirectangularHDR(allocator, 2, 1, pano[0..4], 2),
    );
}

test "fromEquirectangularFile validates input and missing files" {
    const allocator = std.testing.allocator;

    // face_size = 0 fails before any file IO.
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.fromEquirectangularFile(allocator, "definitely/missing/file.png", 0),
    );
    // Missing file must fail cleanly (this also keeps the std.Io read path
    // analyzed, so future Zig API churn is caught by `zig build test`).
    try std.testing.expectError(
        error.FileNotFound,
        CubeTexture.fromEquirectangularFile(allocator, "definitely/missing/file.png", 4),
    );
}

// ---------------------------------------------------------------------------
// Golden test: HDR cube f16 mip chain (feeds textureLod in the IBL path)
// ---------------------------------------------------------------------------
test "buildRawFacesHdr builds the f16 mip chain with averaged values" {
    const allocator = std.testing.allocator;

    // One 2x2 face with R={1, 3, 5, 7}, G=B=0, A=1: level 1 must average
    // R to 4.0 exactly (f16-representable).
    var face: [2 * 2 * 4]u16 = undefined;
    const r_values = [_]f32{ 1, 3, 5, 7 };
    for (r_values, 0..) |rv, i| {
        face[i * 4 + 0] = Texture.floatToHalfBits(rv);
        face[i * 4 + 1] = 0;
        face[i * 4 + 2] = 0;
        face[i * 4 + 3] = Texture.floatToHalfBits(1.0);
    }
    var faces: [6][]const u16 = undefined;
    for (&faces) |*f| f.* = &face;

    var chain = try CubeTexture.buildRawFacesHdr(allocator, 2, faces);
    defer chain.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), chain.num_levels);
    try std.testing.expectEqual(@as(u32, 2), chain.size);

    const l1 = chain.levels[1].?;
    try std.testing.expectEqual(@as(usize, 6 * 1 * 1 * 4), l1.len);
    // Face 0 averaged R = 4.0; every face inherits the same average.
    for (0..6) |f| {
        const o = f * 4;
        try std.testing.expectEqual(@as(f32, 4.0), Texture.halfBitsToFloat(l1[o + 0]));
        try std.testing.expectEqual(@as(f32, 0.0), Texture.halfBitsToFloat(l1[o + 1]));
        try std.testing.expectEqual(@as(f32, 1.0), Texture.halfBitsToFloat(l1[o + 3]));
    }
}

test "buildRawFacesHdr validates size and face buffers" {
    const allocator = std.testing.allocator;
    const face: [2 * 2 * 4]u16 = @splat(0);
    var faces: [6][]const u16 = @splat(&face);

    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.buildRawFacesHdr(allocator, 0, faces),
    );
    faces[5] = face[0..2];
    try std.testing.expectError(
        error.InvalidFaceBufferSize,
        CubeTexture.buildRawFacesHdr(allocator, 2, faces),
    );
}

test "initRawFaces validates dimensions without GPU upload" {
    const allocator = std.testing.allocator;
    const empty_faces: [6][]const u8 = @splat(&.{});
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.initRawFaces(allocator, 0, empty_faces, false),
    );
    // Huge size fails on checked arithmetic before face-length checks.
    try std.testing.expectError(
        error.ImageTooLarge,
        CubeTexture.initRawFaces(allocator, 100000, empty_faces, false),
    );
    // Small size with short faces fails on length validation (no GPU).
    var one_pixel: [4]u8 = .{ 1, 2, 3, 255 };
    const bad_faces: [6][]const u8 = .{
        one_pixel[0..2], &one_pixel, &one_pixel, &one_pixel, &one_pixel, &one_pixel,
    };
    try std.testing.expectError(
        error.InvalidFaceBufferSize,
        CubeTexture.initRawFaces(allocator, 1, bad_faces, false),
    );
}

test "createProceduralSkybox rejects empty size" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.createProceduralSkybox(allocator, .{ .size = 0 }),
    );
}

test "fromFiles reports missing faces without freeing uninitialized memory" {
    const allocator = std.testing.allocator;
    // First face missing: loaded == 0, the sized defer frees nothing.
    // C stbi allocations are not tracked by std.testing.allocator; this
    // only asserts the error contract, not leak accounting.
    const missing: [6][]const u8 = @splat("definitely/missing/face.png");
    try std.testing.expectError(
        error.ImageDecodeFailed,
        CubeTexture.fromFiles(allocator, missing),
    );
}

test "fromEquirectangular rejects huge faces before allocation" {
    const allocator = std.testing.allocator;
    // 1x1 RGBA PNG through the in-file fixture builder (no GPU involved).
    const scanlines = [_]u8{ 0, 10, 20, 30, 255 };
    const png = try core.TestPng.build(allocator, 1, 1, 8, 6, null, &scanlines);
    defer allocator.free(png);
    try std.testing.expectError(
        error.ImageTooLarge,
        CubeTexture.fromEquirectangular(allocator, png, 100000),
    );
}
