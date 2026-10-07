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

                    // Conservative downsample: max of 4 children
                    const m0 = @max(d00, d10);
                    const m1 = @max(d01, d11);
                    const max_depth = @max(m0, m1);

                    self.data[dst.offset + y * dst.width + x] = max_depth;
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

        const corners = aabb.corners();
        var min_screen_x: f32 = std.math.inf(f32);
        var max_screen_x: f32 = -std.math.inf(f32);
        var min_screen_y: f32 = std.math.inf(f32);
        var max_screen_y: f32 = -std.math.inf(f32);
        var min_ndc_z: f32 = std.math.inf(f32);

        // Project all 8 corners
        inline for (0..8) |i| {
            const p = corners[i];
            const x = view_proj.m[0] * p.x + view_proj.m[4] * p.y + view_proj.m[8] * p.z + view_proj.m[12];
            const y = view_proj.m[1] * p.x + view_proj.m[5] * p.y + view_proj.m[9] * p.z + view_proj.m[13];
            const z = view_proj.m[2] * p.x + view_proj.m[6] * p.y + view_proj.m[10] * p.z + view_proj.m[14];
            const w = view_proj.m[3] * p.x + view_proj.m[7] * p.y + view_proj.m[11] * p.z + view_proj.m[15];

            // If any corner is behind the camera or on near plane, bounding box intersects camera near plane:
            // Must conservatively consider visible (cannot cull).
            if (w <= 0.001) return false;

            const inv_w = 1.0 / w;
            const ndc_x = x * inv_w;
            const ndc_y = y * inv_w;
            const ndc_z = z * inv_w;

            const sx = (ndc_x * 0.5 + 0.5) * @as(f32, @floatFromInt(WIDTH - 1));
            const sy = (1.0 - (ndc_y * 0.5 + 0.5)) * @as(f32, @floatFromInt(HEIGHT - 1));

            min_screen_x = @min(min_screen_x, sx);
            max_screen_x = @max(max_screen_x, sx);
            min_screen_y = @min(min_screen_y, sy);
            max_screen_y = @max(max_screen_y, sy);
            min_ndc_z = @min(min_ndc_z, ndc_z);
        }

        // Outside screen horizontally or vertically
        if (max_screen_x < 0.0 or min_screen_x > @as(f32, @floatFromInt(WIDTH - 1)) or
            max_screen_y < 0.0 or min_screen_y > @as(f32, @floatFromInt(HEIGHT - 1)))
        {
            return false;
        }

        const cl_min_x: u32 = @intFromFloat(@max(0.0, @floor(min_screen_x)));
        const cl_max_x: u32 = @intFromFloat(@min(@as(f32, @floatFromInt(WIDTH - 1)), @ceil(max_screen_x)));
        const cl_min_y: u32 = @intFromFloat(@max(0.0, @floor(min_screen_y)));
        const cl_max_y: u32 = @intFromFloat(@min(@as(f32, @floatFromInt(HEIGHT - 1)), @ceil(max_screen_y)));

        const dx = @as(f32, @floatFromInt(cl_max_x - cl_min_x));
        const dy = @as(f32, @floatFromInt(cl_max_y - cl_min_y));
        const max_dim = @max(dx, dy);

        // Select mip level where footprint fits within 2x2 or 3x3 texels
        var mip_level: usize = 0;
        if (max_dim > 1.0) {
            const log2_dim = std.math.log2(max_dim);
            const raw_mip: usize = @intFromFloat(@floor(log2_dim));
            mip_level = @min(raw_mip, NUM_MIPS - 1);
        }

        const info = MIP_TABLE[mip_level];
        const tex_u0 = @min(cl_min_x >> @intCast(mip_level), info.width - 1);
        const tex_u1 = @min(cl_max_x >> @intCast(mip_level), info.width - 1);
        const tex_v0 = @min(cl_min_y >> @intCast(mip_level), info.height - 1);
        const tex_v1 = @min(cl_max_y >> @intCast(mip_level), info.height - 1);

        var max_occluder_depth: f32 = 0.0;
        var v = tex_v0;
        while (v <= tex_v1) : (v += 1) {
            var u = tex_u0;
            while (u <= tex_u1) : (u += 1) {
                const d = self.data[info.offset + v * info.width + u];
                max_occluder_depth = @max(max_occluder_depth, d);
            }
        }

        // If the closest point of the occludee is FURTHER than the furthest occluder
        // in the covered region, the object is 100% occluded!
        return (min_ndc_z > max_occluder_depth);
    }
};
