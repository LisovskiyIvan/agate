const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const c = @import("c.zig").c;

pub const Texture = struct {
    image: sg.Image,
    view: sg.View,
    sampler: sg.Sampler,
    width: u32,
    height: u32,

    pub const Options = struct {
        min_filter: sg.Filter = .LINEAR,
        mag_filter: sg.Filter = .LINEAR,
        wrap_u: sg.Wrap = .REPEAT,
        wrap_v: sg.Wrap = .REPEAT,
        /// Signed distance fields must not be box-downsampled: the mip chain
        /// dilutes thin strokes and the shader edge drifts. Disable for fonts.
        mipmaps: bool = true,
    };

    pub fn initRaw(width: u32, height: u32, rgba_pixels: []const u8, options: Options) Texture {
        var img_desc = sg.ImageDesc{
            .width = @intCast(width),
            .height = @intCast(height),
            .pixel_format = .RGBA8,
        };
        img_desc.data.mip_levels[0] = sg.asRange(rgba_pixels);

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });

        const smp = sg.makeSampler(.{
            .min_filter = options.min_filter,
            .mag_filter = options.mag_filter,
            .wrap_u = options.wrap_u,
            .wrap_v = options.wrap_v,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .width = width,
            .height = height,
        };
    }
    /// Box-filter downsample of one RGBA8 level. Dims floor at 1, source coords
    /// clamp at edges (handles NPOT).
    fn downsampleLevel(src: []const u8, src_w: u32, src_h: u32, dst: []u8, dst_w: u32, dst_h: u32) void {
        @setRuntimeSafety(false);
        if (src_w == dst_w * 2 and src_h == dst_h * 2) {
            const src_stride: usize = @as(usize, src_w) * 4;
            const dst_stride: usize = @as(usize, dst_w) * 4;
            var y: usize = 0;
            while (y < dst_h) : (y += 1) {
                const r0 = src[(y * 2) * src_stride ..];
                const r1 = src[(y * 2 + 1) * src_stride ..];
                const out_row = dst[y * dst_stride ..];
                var x: usize = 0;
                while (x < dst_w) : (x += 1) {
                    const si = x * 8;
                    const di = x * 4;
                    inline for (0..4) |ch| {
                        const sum: u32 = @as(u32, r0[si + ch]) +
                            @as(u32, r0[si + 4 + ch]) +
                            @as(u32, r1[si + ch]) +
                            @as(u32, r1[si + 4 + ch]);
                        out_row[di + ch] = @intCast((sum + 2) >> 2);
                    }
                }
            }
            return;
        }

        var y: u32 = 0;
        while (y < dst_h) : (y += 1) {
            const sy0 = @min(y * 2, src_h - 1);
            const sy1 = @min(y * 2 + 1, src_h - 1);
            var x: u32 = 0;
            while (x < dst_w) : (x += 1) {
                const sx0 = @min(x * 2, src_w - 1);
                const sx1 = @min(x * 2 + 1, src_w - 1);
                const q00 = (sy0 * src_w + sx0) * 4;
                const q10 = (sy0 * src_w + sx1) * 4;
                const q01 = (sy1 * src_w + sx0) * 4;
                const q11 = (sy1 * src_w + sx1) * 4;
                const o = (y * dst_w + x) * 4;
                inline for (0..4) |ch| {
                    const sum: u32 = @as(u32, src[q00 + ch]) + @as(u32, src[q10 + ch]) + @as(u32, src[q01 + ch]) + @as(u32, src[q11 + ch]);
                    dst[o + ch] = @intCast((sum + 2) >> 2);
                }
            }
        }
    }

    pub fn mipLevelCount(width: u32, height: u32) u32 {
        var levels: u32 = 1;
        var w = width;
        var h = height;
        while ((w > 1 or h > 1) and levels < 16) {
            w = @max(1, w / 2);
            h = @max(1, h / 2);
            levels += 1;
        }
        return levels;
    }

    /// initRaw plus a full CPU mipmap chain. Load-time cost only; minification
    /// becomes a trilinear fetch instead of cache-thrashing level 0.
    pub fn initRawMipped(allocator: std.mem.Allocator, width: u32, height: u32, rgba_pixels: []const u8, options: Options) !Texture {
        const levels = mipLevelCount(width, height);
        if (levels == 1) return initRaw(width, height, rgba_pixels, options);

        var img_desc = sg.ImageDesc{
            .width = @intCast(width),
            .height = @intCast(height),
            .pixel_format = .RGBA8,
            .num_mipmaps = @intCast(levels),
        };
        img_desc.data.mip_levels[0] = sg.asRange(rgba_pixels);

        var mip_bufs: [16]?[]u8 = @splat(null);
        defer {
            for (1..levels) |m| {
                if (mip_bufs[m]) |buf| allocator.free(buf);
            }
        }

        var prev_w = width;
        var prev_h = height;
        var prev: []const u8 = rgba_pixels;
        for (1..levels) |m| {
            const cur_w: u32 = @max(1, prev_w / 2);
            const cur_h: u32 = @max(1, prev_h / 2);
            const cur = try allocator.alloc(u8, @as(usize, cur_w) * cur_h * 4);
            mip_bufs[m] = cur;
            downsampleLevel(prev, prev_w, prev_h, cur, cur_w, cur_h);
            img_desc.data.mip_levels[m] = sg.asRange(cur);
            prev_w = cur_w;
            prev_h = cur_h;
            prev = cur;
        }

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = options.min_filter,
            .mag_filter = options.mag_filter,
            .mipmap_filter = .LINEAR,
            .wrap_u = options.wrap_u,
            .wrap_v = options.wrap_v,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .width = width,
            .height = height,
        };
    }

    pub fn createWhite1x1() Texture {
        const white = [_]u8{ 255, 255, 255, 255 };
        return initRaw(1, 1, &white, .{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
        });
    }

    pub fn createBlack1x1() Texture {
        const black = [_]u8{ 0, 0, 0, 255 };
        return initRaw(1, 1, &black, .{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
        });
    }

    pub fn createFlatNormal1x1() Texture {
        const flat_normal = [_]u8{ 128, 128, 255, 255 };
        return initRaw(1, 1, &flat_normal, .{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
        });
    }

    pub fn createCheckerboard(
        allocator: std.mem.Allocator,
        width: u32,
        height: u32,
        cell_size: u32,
        color1: [4]u8,
        color2: [4]u8,
    ) !Texture {
        const pixel_count = width * height;
        const buffer = try allocator.alloc(u8, pixel_count * 4);
        defer allocator.free(buffer);

        var y: u32 = 0;
        while (y < height) : (y += 1) {
            var x: u32 = 0;
            while (x < width) : (x += 1) {
                const is_even = (((x / cell_size) + (y / cell_size)) % 2) == 0;
                const c_val = if (is_even) color1 else color2;
                const idx = (y * width + x) * 4;
                buffer[idx + 0] = c_val[0];
                buffer[idx + 1] = c_val[1];
                buffer[idx + 2] = c_val[2];
                buffer[idx + 3] = c_val[3];
            }
        }

        return initRaw(width, height, buffer, .{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
        });
    }

    pub fn createDefaultParticleDot32() Texture {
        var buf: [32 * 32 * 4]u8 = undefined;
        const size: usize = 32;
        const center: f32 = 15.5;
        const radius: f32 = 15.5;

        var y: usize = 0;
        while (y < size) : (y += 1) {
            var x: usize = 0;
            while (x < size) : (x += 1) {
                const dx = @as(f32, @floatFromInt(x)) - center;
                const dy = @as(f32, @floatFromInt(y)) - center;
                const dist = @sqrt(dx * dx + dy * dy);
                const norm_dist = @min(1.0, dist / radius);
                const alpha_f = (1.0 - norm_dist) * (1.0 - norm_dist);
                const alpha: u8 = @intFromFloat(std.math.clamp(alpha_f * 255.0, 0.0, 255.0));

                const idx = (y * size + x) * 4;
                buf[idx + 0] = 255;
                buf[idx + 1] = 255;
                buf[idx + 2] = 255;
                buf[idx + 3] = alpha;
            }
        }

        return initRaw(32, 32, &buf, .{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });
    }

    pub fn createParticleDot(allocator: std.mem.Allocator, size: u32) !Texture {
        const pixel_count = size * size;
        const buffer = try allocator.alloc(u8, pixel_count * 4);
        defer allocator.free(buffer);

        const center: f32 = @as(f32, @floatFromInt(size)) * 0.5 - 0.5;
        const radius: f32 = @as(f32, @floatFromInt(size)) * 0.5;

        var y: u32 = 0;
        while (y < size) : (y += 1) {
            var x: u32 = 0;
            while (x < size) : (x += 1) {
                const dx = @as(f32, @floatFromInt(x)) - center;
                const dy = @as(f32, @floatFromInt(y)) - center;
                const dist = @sqrt(dx * dx + dy * dy);
                const norm_dist = @min(1.0, dist / radius);
                const alpha_f = (1.0 - norm_dist) * (1.0 - norm_dist);
                const alpha: u8 = @intFromFloat(std.math.clamp(alpha_f * 255.0, 0.0, 255.0));

                const idx = (y * size + x) * 4;
                buffer[idx + 0] = 255;
                buffer[idx + 1] = 255;
                buffer[idx + 2] = 255;
                buffer[idx + 3] = alpha;
            }
        }

        return initRaw(size, size, buffer, .{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });
    }

    pub fn fromMemory(allocator: std.mem.Allocator, bytes: []const u8, options: Options) !Texture {
        var w: c_int = 0;
        var h: c_int = 0;
        var channels_in_file: c_int = 0;

        const desired_channels = 4; // Always load as RGBA
        const t0 = sokol.time.now();
        const data = c.stbi_load_from_memory(
            bytes.ptr,
            @intCast(bytes.len),
            &w,
            &h,
            &channels_in_file,
            desired_channels,
        );
        const t1 = sokol.time.now();

        if (data == null) {
            return error.ImageDecodeFailed;
        }
        defer c.stbi_image_free(data);

        const width: u32 = @intCast(w);
        const height: u32 = @intCast(h);
        const size_bytes: usize = @as(usize, width) * @as(usize, height) * 4;
        const slice = data[0..size_bytes];

        const res = if (options.mipmaps)
            try initRawMipped(allocator, width, height, slice, options)
        else
            initRaw(width, height, slice, options);
        const t2 = sokol.time.now();
        std.debug.print("  [TEX {d}x{d}] stbi: {d:0.1} ms | mipgen: {d:0.1} ms | total: {d:0.1} ms\n", .{
            width,                                  height,
            sokol.time.ms(sokol.time.diff(t1, t0)), sokol.time.ms(sokol.time.diff(t2, t1)),
            sokol.time.ms(sokol.time.diff(t2, t0)),
        });
        return res;
    }

    pub fn fromFile(allocator: std.mem.Allocator, file_path: []const u8, options: Options) !Texture {
        const path_z = try allocator.dupeZ(u8, file_path);
        defer allocator.free(path_z);

        var w: c_int = 0;
        var h: c_int = 0;
        var channels_in_file: c_int = 0;

        const data = c.stbi_load(
            path_z.ptr,
            &w,
            &h,
            &channels_in_file,
            4,
        );
        if (data == null) {
            return error.ImageDecodeFailed;
        }
        defer c.stbi_image_free(data);

        const width: u32 = @intCast(w);
        const height: u32 = @intCast(h);
        const size_bytes: usize = @as(usize, width) * @as(usize, height) * 4;
        const slice = data[0..size_bytes];

        if (!options.mipmaps) {
            return initRaw(width, height, slice, options);
        }
        return initRawMipped(allocator, width, height, slice, options);
    }

    pub fn deinit(self: *Texture) void {
        sg.destroyView(self.view);
        sg.destroyImage(self.image);
        sg.destroySampler(self.sampler);
    }
};

pub const SkyboxConfig = struct {
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

    pub fn deinit(self: *CubeTexture) void {
        sg.destroyView(self.view);
        sg.destroyImage(self.image);
        sg.destroySampler(self.sampler);
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
        const face_bytes = size * size * 4;
        for (faces) |face| {
            if (face.len != face_bytes) return error.InvalidFaceBufferSize;
        }

        // Concatenate 6 faces for mip 0
        const mip0_total_bytes = 6 * face_bytes;
        const mip0_buffer = try allocator.alloc(u8, mip0_total_bytes);
        defer allocator.free(mip0_buffer);

        for (faces, 0..) |face, i| {
            const dst_offset = i * face_bytes;
            @memcpy(mip0_buffer[dst_offset .. dst_offset + face_bytes], face);
        }

        var num_mips: u32 = 1;
        if (generate_mips and size > 1) {
            var s = size;
            while (s > 1) : (s /= 2) {
                num_mips += 1;
            }
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
            const cur_size = prev_size / 2;
            const cur_total_bytes = 6 * cur_size * cur_size * 4;
            const cur_buf = try allocator.alloc(u8, cur_total_bytes);
            mip_buffers[m] = cur_buf;

            for (0..6) |face| {
                const src_face_offset = face * prev_size * prev_size * 4;
                const dst_face_offset = face * cur_size * cur_size * 4;

                var dy: u32 = 0;
                while (dy < cur_size) : (dy += 1) {
                    const sy = dy * 2;
                    var dx: u32 = 0;
                    while (dx < cur_size) : (dx += 1) {
                        const sx = dx * 2;
                        const idx00 = src_face_offset + (sy * prev_size + sx) * 4;
                        const idx10 = src_face_offset + (sy * prev_size + sx + 1) * 4;
                        const idx01 = src_face_offset + ((sy + 1) * prev_size + sx) * 4;
                        const idx11 = src_face_offset + ((sy + 1) * prev_size + sx + 1) * 4;

                        const out_i = dst_face_offset + (dy * cur_size + dx) * 4;
                        inline for (0..4) |c_idx| {
                            const sum: u32 = @as(u32, prev_buf[idx00 + c_idx]) +
                                @as(u32, prev_buf[idx10 + c_idx]) +
                                @as(u32, prev_buf[idx01 + c_idx]) +
                                @as(u32, prev_buf[idx11 + c_idx]);
                            cur_buf[out_i + c_idx] = @intCast((sum + 2) / 4);
                        }
                    }
                }
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

    pub fn createProceduralSkybox(allocator: std.mem.Allocator, config: SkyboxConfig) !CubeTexture {
        const size = config.size;
        const face_bytes = size * size * 4;

        var face_slices: [6][]u8 = undefined;
        for (0..6) |i| {
            face_slices[i] = try allocator.alloc(u8, face_bytes);
        }
        defer {
            for (0..6) |i| allocator.free(face_slices[i]);
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

                    const out_idx = (py * size + px) * 4;
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
        var size: u32 = 0;

        for (face_paths, 0..) |path, i| {
            const path_z = try allocator.dupeZ(u8, path);
            defer allocator.free(path_z);

            var w: c_int = 0;
            var h: c_int = 0;
            var comp: c_int = 0;
            const data = c.stbi_load(path_z.ptr, &w, &h, &comp, 4);
            if (data == null) {
                // Free previously loaded
                for (0..i) |prev| c.stbi_image_free(face_data[prev]);
                return error.ImageDecodeFailed;
            }
            if (w != h) {
                for (0..i + 1) |prev| c.stbi_image_free(face_data[prev]);
                return error.CubeFaceMustBeSquare;
            }
            if (i == 0) {
                size = @intCast(w);
            } else if (@as(u32, @intCast(w)) != size) {
                for (0..i + 1) |prev| c.stbi_image_free(face_data[prev]);
                return error.CubeFacesMustHaveEqualSize;
            }
            face_data[i] = data;
        }
        defer {
            for (0..6) |i| c.stbi_image_free(face_data[i]);
        }

        const face_bytes = size * size * 4;
        var const_faces: [6][]const u8 = undefined;
        for (0..6) |i| {
            const_faces[i] = face_data[i][0..face_bytes];
        }

        return initRawFaces(allocator, size, const_faces, true);
    }

    pub fn fromEquirectangular(allocator: std.mem.Allocator, panorama_bytes: []const u8, face_size: u32) !CubeTexture {
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

        const pano_w: u32 = @intCast(pw);
        const pano_h: u32 = @intCast(ph);
        const face_bytes = face_size * face_size * 4;

        var face_slices: [6][]u8 = undefined;
        for (0..6) |i| {
            face_slices[i] = try allocator.alloc(u8, face_bytes);
        }
        defer {
            for (0..6) |i| allocator.free(face_slices[i]);
        }

        for (0..6) |face| {
            const buf = face_slices[face];
            var py: u32 = 0;
            while (py < face_size) : (py += 1) {
                const v = 2.0 * (@as(f32, @floatFromInt(py)) + 0.5) / @as(f32, @floatFromInt(face_size)) - 1.0;
                var px: u32 = 0;
                while (px < face_size) : (px += 1) {
                    const u = 2.0 * (@as(f32, @floatFromInt(px)) + 0.5) / @as(f32, @floatFromInt(face_size)) - 1.0;

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
                    const d: [3]f32 = .{ dir[0] / len, dir[1] / len, dir[2] / len };

                    // Spherical coordinates
                    const phi = std.math.atan2(d[2], d[0]); // longitude [-pi, pi]
                    const theta = std.math.asin(std.math.clamp(d[1], -1.0, 1.0)); // latitude [-pi/2, pi/2]

                    const u_pano = (phi / (2.0 * std.math.pi)) + 0.5;
                    const v_pano = 0.5 - (theta / std.math.pi);

                    const sx_f = std.math.clamp(u_pano * @as(f32, @floatFromInt(pano_w)), 0.0, @as(f32, @floatFromInt(pano_w - 1)));
                    const sy_f = std.math.clamp(v_pano * @as(f32, @floatFromInt(pano_h)), 0.0, @as(f32, @floatFromInt(pano_h - 1)));

                    const sx: u32 = @intFromFloat(sx_f);
                    const sy: u32 = @intFromFloat(sy_f);

                    const p_idx = (sy * pano_w + sx) * 4;
                    const out_idx = (py * face_size + px) * 4;

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
        const file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        const file_size = (try file.stat()).size;
        const bytes = try allocator.alloc(u8, file_size);
        defer allocator.free(bytes);

        _ = try file.readAll(bytes);
        return fromEquirectangular(allocator, bytes, face_size);
    }
};

test "mipLevelCount covers powers of two and minimums" {
    try std.testing.expectEqual(@as(u32, 1), Texture.mipLevelCount(1, 1));
    try std.testing.expectEqual(@as(u32, 2), Texture.mipLevelCount(2, 2));
    try std.testing.expectEqual(@as(u32, 10), Texture.mipLevelCount(512, 512));
    try std.testing.expectEqual(@as(u32, 3), Texture.mipLevelCount(5, 3));
}

test "downsampleLevel handles odd dimensions and averages correctly" {
    var src: [3 * 5 * 4]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @intCast(i % 251);
    var dst: [1 * 2 * 4]u8 = undefined;
    Texture.downsampleLevel(&src, 3, 5, &dst, 1, 2);
    // Top-left texel averages src bytes {0,4,12,16}: (0+4+12+16+2)/4 = 8.
    try std.testing.expectEqual(@as(u8, 8), dst[0]);
}
