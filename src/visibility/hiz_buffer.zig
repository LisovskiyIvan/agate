const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;

/// Hierarchical Z-Buffer (Hi-Z) for ultra-fast, conservative CPU software occlusion culling.
/// Resolution: 256x128 at Level 0, downsampled into 9 mip levels down to 1x1.
/// Total memory footprint is only ~175 KB, fitting entirely within CPU L2 cache.
pub const HiZBuffer = struct {
    pub const WIDTH: u32 = 256;
    pub const HEIGHT: u32 = 128;
    pub const NUM_MIPS: usize = 9;

    // Level 0: 256x128 = 32768
    // Level 1: 128x64  = 8192
    // Level 2: 64x32   = 2048
    // Level 3: 32x16   = 512
    // Level 4: 16x8    = 128
    // Level 5: 8x4     = 32
    // Level 6: 4x2     = 8
    // Level 7: 2x1     = 2
    // Level 8: 1x1     = 1
    // Total: 43691 floats (~174.76 KB)
    pub const TOTAL_FLOATS: usize = 43691;

    pub const MipInfo = struct {
        offset: usize,
        width: u32,
        height: u32,
    };

    pub const MIP_TABLE: [NUM_MIPS]MipInfo = blk: {
        var table: [NUM_MIPS]MipInfo = undefined;
        var offset: usize = 0;
        var w: u32 = WIDTH;
        var h: u32 = HEIGHT;
        var i: usize = 0;
        while (i < NUM_MIPS) : (i += 1) {
            table[i] = .{
                .offset = offset,
                .width = w,
                .height = h,
            };
            offset += @as(usize, w) * @as(usize, h);
            w = @max(w / 2, 1);
            h = @max(h / 2, 1);
        }
        break :blk table;
    };

    data: [TOTAL_FLOATS]f32,

    pub fn init() HiZBuffer {
        var buf: HiZBuffer = undefined;
        buf.clear(1.0);
        return buf;
    }

    /// Clears Level 0 to `depth` (typically 1.0, the far plane in 0..1 depth range).
    pub fn clear(self: *HiZBuffer, depth: f32) void {
        const l0_len = @as(usize, WIDTH) * @as(usize, HEIGHT);
        @memset(self.data[0..l0_len], depth);
    }

    /// Writes depth into Level 0 using standard depth-test (closer is smaller depth).
    pub inline fn writePixel(self: *HiZBuffer, x: u32, y: u32, depth: f32) void {
        if (x >= WIDTH or y >= HEIGHT) return;
        const idx = y * WIDTH + x;
        if (depth < self.data[idx]) {
            self.data[idx] = depth;
        }
    }

    /// Samples a specific pixel from any mip level.
    pub inline fn sampleLevel(self: *const HiZBuffer, level: usize, x: u32, y: u32) f32 {
        const info = MIP_TABLE[level];
        const cl_x = @min(x, info.width - 1);
        const cl_y = @min(y, info.height - 1);
        return self.data[info.offset + cl_y * info.width + cl_x];
    }

    /// Builds the conservative Hi-Z pyramid.
    /// Each texel in level N+1 stores the MAXIMUM depth of its 2x2 children in level N.
    /// Because closer is smaller Z, the maximum is the FURTHEST occluder depth in that footprint.
    /// An occludee is only hidden if its nearest depth is GREATER than this maximum,
    /// guaranteeing mathematically ZERO false-positive culls.
    pub fn buildPyramid(self: *HiZBuffer) void {
        var lvl: usize = 0;
        while (lvl + 1 < NUM_MIPS) : (lvl += 1) {
            const src = MIP_TABLE[lvl];
            const dst = MIP_TABLE[lvl + 1];

            // SIMD fast-path for levels where dst.width >= 4 and src.height == dst.height * 2
            if (dst.width >= 4 and src.height == dst.height * 2) {
                var y: u32 = 0;
                while (y < dst.height) : (y += 1) {
                    const row0 = self.data[src.offset + (y * 2) * src.width ..];
                    const row1 = self.data[src.offset + (y * 2 + 1) * src.width ..];
                    const dst_row = self.data[dst.offset + y * dst.width ..];

                    var x: u32 = 0;
                    while (x < dst.width) : (x += 4) {
                        const r0: @Vector(8, f32) = row0[x * 2 ..][0..8].*;
                        const r1: @Vector(8, f32) = row1[x * 2 ..][0..8].*;
                        const m = @max(r0, r1);
                        const evens: @Vector(4, f32) = @shuffle(f32, m, undefined, @Vector(4, i32){ 0, 2, 4, 6 });
                        const odds: @Vector(4, f32) = @shuffle(f32, m, undefined, @Vector(4, i32){ 1, 3, 5, 7 });
                        dst_row[x..][0..4].* = @max(evens, odds);
                    }
                }
            } else {
                var y: u32 = 0;
                while (y < dst.height) : (y += 1) {
                    const src_y0 = y * 2;
                    const src_y1 = @min(src_y0 + 1, src.height - 1);

                    var x: u32 = 0;
                    while (x < dst.width) : (x += 1) {
                        const src_x0 = x * 2;
                        const src_x1 = @min(src_x0 + 1, src.width - 1);

                        const d00 = self.data[src.offset + src_y0 * src.width + src_x0];
                        const d10 = self.data[src.offset + src_y0 * src.width + src_x1];
                        const d01 = self.data[src.offset + src_y1 * src.width + src_x0];
                        const d11 = self.data[src.offset + src_y1 * src.width + src_x1];

                        const m0 = @max(d00, d10);
                        const m1 = @max(d01, d11);
                        self.data[dst.offset + y * dst.width + x] = @max(m0, m1);
                    }
                }
            }
        }
    }

    /// Tests if a world-space Axis-Aligned Bounding Box (AABB) is fully occluded.
    /// Returns:
    ///   - true: 100% guaranteed occluded (hidden behind occluders) -> CULL IT!
    ///   - false: potentially visible -> RENDER IT!
    pub fn testAABB(self: *const HiZBuffer, view_proj: Mat4, aabb: BoundingBox) bool {
        if (!aabb.isValid()) return false;

        // SIMD 4-wide corner projection: corners 0..3 (min.x) and corners 4..7 (max.x)
        const min_x_4: @Vector(4, f32) = @splat(aabb.min.x);
        const max_x_4: @Vector(4, f32) = @splat(aabb.max.x);
        const y_4: @Vector(4, f32) = .{ aabb.min.y, aabb.min.y, aabb.max.y, aabb.max.y };
        const z_4: @Vector(4, f32) = .{ aabb.min.z, aabb.max.z, aabb.min.z, aabb.max.z };

        const m0: @Vector(4, f32) = @splat(view_proj.m[0]);
        const m1: @Vector(4, f32) = @splat(view_proj.m[1]);
        const m2: @Vector(4, f32) = @splat(view_proj.m[2]);
        const m3: @Vector(4, f32) = @splat(view_proj.m[3]);
        const m4: @Vector(4, f32) = @splat(view_proj.m[4]);
        const m5: @Vector(4, f32) = @splat(view_proj.m[5]);
        const m6: @Vector(4, f32) = @splat(view_proj.m[6]);
        const m7: @Vector(4, f32) = @splat(view_proj.m[7]);
        const m8: @Vector(4, f32) = @splat(view_proj.m[8]);
        const m9: @Vector(4, f32) = @splat(view_proj.m[9]);
        const m10: @Vector(4, f32) = @splat(view_proj.m[10]);
        const m11: @Vector(4, f32) = @splat(view_proj.m[11]);
        const m12: @Vector(4, f32) = @splat(view_proj.m[12]);
        const m13: @Vector(4, f32) = @splat(view_proj.m[13]);
        const m14: @Vector(4, f32) = @splat(view_proj.m[14]);
        const m15: @Vector(4, f32) = @splat(view_proj.m[15]);

        const V4 = @Vector(4, f32);
        const w_03 = @mulAdd(V4, m3, min_x_4, @mulAdd(V4, m7, y_4, @mulAdd(V4, m11, z_4, m15)));
        const w_47 = @mulAdd(V4, m3, max_x_4, @mulAdd(V4, m7, y_4, @mulAdd(V4, m11, z_4, m15)));

        // If any corner is behind the camera or on near plane, cannot cull conservatively
        if (@reduce(.Min, w_03) <= 0.001 or @reduce(.Min, w_47) <= 0.001) return false;

        const inv_w_03 = @as(V4, @splat(1.0)) / w_03;
        const inv_w_47 = @as(V4, @splat(1.0)) / w_47;

        const x_03 = @mulAdd(V4, m0, min_x_4, @mulAdd(V4, m4, y_4, @mulAdd(V4, m8, z_4, m12)));
        const y_03 = @mulAdd(V4, m1, min_x_4, @mulAdd(V4, m5, y_4, @mulAdd(V4, m9, z_4, m13)));
        const z_03 = @mulAdd(V4, m2, min_x_4, @mulAdd(V4, m6, y_4, @mulAdd(V4, m10, z_4, m14)));

        const x_47 = @mulAdd(V4, m0, max_x_4, @mulAdd(V4, m4, y_4, @mulAdd(V4, m8, z_4, m12)));
        const y_47 = @mulAdd(V4, m1, max_x_4, @mulAdd(V4, m5, y_4, @mulAdd(V4, m9, z_4, m13)));
        const z_47 = @mulAdd(V4, m2, max_x_4, @mulAdd(V4, m6, y_4, @mulAdd(V4, m10, z_4, m14)));

        const ndc_x_03 = x_03 * inv_w_03;
        const ndc_y_03 = y_03 * inv_w_03;
        const ndc_z_03 = z_03 * inv_w_03;

        const ndc_x_47 = x_47 * inv_w_47;
        const ndc_y_47 = y_47 * inv_w_47;
        const ndc_z_47 = z_47 * inv_w_47;

        const w_f: f32 = @floatFromInt(WIDTH - 1);
        const h_f: f32 = @floatFromInt(HEIGHT - 1);
        const w_f_4: @Vector(4, f32) = @splat(w_f);
        const h_f_4: @Vector(4, f32) = @splat(h_f);

        const sx_03 = (ndc_x_03 * @as(@Vector(4, f32), @splat(0.5)) + @as(@Vector(4, f32), @splat(0.5))) * w_f_4;
        const sy_03 = (@as(@Vector(4, f32), @splat(1.0)) - (ndc_y_03 * @as(@Vector(4, f32), @splat(0.5)) + @as(@Vector(4, f32), @splat(0.5)))) * h_f_4;

        const sx_47 = (ndc_x_47 * @as(@Vector(4, f32), @splat(0.5)) + @as(@Vector(4, f32), @splat(0.5))) * w_f_4;
        const sy_47 = (@as(@Vector(4, f32), @splat(1.0)) - (ndc_y_47 * @as(@Vector(4, f32), @splat(0.5)) + @as(@Vector(4, f32), @splat(0.5)))) * h_f_4;

        const min_screen_x = @min(@reduce(.Min, sx_03), @reduce(.Min, sx_47));
        const max_screen_x = @max(@reduce(.Max, sx_03), @reduce(.Max, sx_47));
        const min_screen_y = @min(@reduce(.Min, sy_03), @reduce(.Min, sy_47));
        const max_screen_y = @max(@reduce(.Max, sy_03), @reduce(.Max, sy_47));
        const min_ndc_z = @min(@reduce(.Min, ndc_z_03), @reduce(.Min, ndc_z_47));

        // Outside screen horizontally or vertically
        if (max_screen_x < 0.0 or min_screen_x > w_f or
            max_screen_y < 0.0 or min_screen_y > h_f)
        {
            return false;
        }

        const cl_min_x: u32 = @intFromFloat(@max(0.0, @floor(min_screen_x)));
        const cl_max_x: u32 = @intFromFloat(@min(w_f, @ceil(max_screen_x)));
        const cl_min_y: u32 = @intFromFloat(@max(0.0, @floor(min_screen_y)));
        const cl_max_y: u32 = @intFromFloat(@min(h_f, @ceil(max_screen_y)));

        const dx = @as(f32, @floatFromInt(cl_max_x - cl_min_x));
        const dy = @as(f32, @floatFromInt(cl_max_y - cl_min_y));
        const max_dim = @max(dx, dy);

        // Select mip level using fast integer @clz instead of libm log2(f32)
        var mip_level: usize = 0;
        if (max_dim > 1.0) {
            const max_dim_u: u32 = @intFromFloat(max_dim);
            const raw_mip: usize = if (max_dim_u > 1) 31 - @clz(max_dim_u) else 0;
            mip_level = @min(raw_mip, NUM_MIPS - 1);
        }

        const info = MIP_TABLE[mip_level];
        const shift: u5 = @intCast(mip_level);
        const tex_u0 = @min(cl_min_x >> shift, info.width - 1);
        const tex_u1 = @min(cl_max_x >> shift, info.width - 1);
        const tex_v0 = @min(cl_min_y >> shift, info.height - 1);
        const tex_v1 = @min(cl_max_y >> shift, info.height - 1);

        var max_occluder_depth: f32 = 0.0;
        var row_offset = info.offset + tex_v0 * info.width;
        var v = tex_v0;
        while (v <= tex_v1) : (v += 1) {
            var u = tex_u0;
            while (u <= tex_u1) : (u += 1) {
                const d = self.data[row_offset + u];
                max_occluder_depth = @max(max_occluder_depth, d);
            }
            row_offset += info.width;
        }

        // If the closest point of the occludee is FURTHER than the furthest occluder
        // in the covered region, the object is 100% occluded!
        return (min_ndc_z > max_occluder_depth);
    }
};
