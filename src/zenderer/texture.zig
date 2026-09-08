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

    pub fn fromMemory(bytes: []const u8, options: Options) !Texture {
        var w: c_int = 0;
        var h: c_int = 0;
        var channels_in_file: c_int = 0;

        const desired_channels = 4; // Always load as RGBA
        const data = c.stbi_load_from_memory(
            bytes.ptr,
            @intCast(bytes.len),
            &w,
            &h,
            &channels_in_file,
            desired_channels,
        );

        if (data == null) {
            return error.ImageDecodeFailed;
        }
        defer c.stbi_image_free(data);

        const width: u32 = @intCast(w);
        const height: u32 = @intCast(h);
        const size_bytes: usize = @as(usize, width) * @as(usize, height) * 4;
        const slice = data[0..size_bytes];

        return initRaw(width, height, slice, options);
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

        return initRaw(width, height, slice, options);
    }

    pub fn deinit(self: *Texture) void {
        sg.destroyView(self.view);
        sg.destroyImage(self.image);
        sg.destroySampler(self.sampler);
    }
};
