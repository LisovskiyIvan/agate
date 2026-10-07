const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const HiZBuffer = @import("hiz_buffer.zig").HiZBuffer;

pub const SoftwareRasterizer = struct {
    pub const ClipVertex = struct {
        x: f32,
        y: f32,
        z: f32,
        w: f32,
    };

    inline fn transformToClip(m: Mat4, p: Vec3) ClipVertex {
        return .{
            .x = m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12],
            .y = m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13],
            .z = m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14],
            .w = m.m[3] * p.x + m.m[7] * p.y + m.m[11] * p.z + m.m[15],
        };
    }

    inline fn lerpClip(a: ClipVertex, b: ClipVertex, t: f32) ClipVertex {
        return .{
            .x = a.x + (b.x - a.x) * t,
            .y = a.y + (b.y - a.y) * t,
            .z = a.z + (b.z - a.z) * t,
            .w = a.w + (b.w - a.w) * t,
        };
    }

    /// Rasterizes a single 3D triangle given world-space vertices.
    /// Handles near-plane clipping, perspective divide, and conservative depth writing.
    pub fn rasterizeTriangle(
        hiz: *HiZBuffer,
        view_proj: Mat4,
        p0: Vec3,
        p1: Vec3,
        p2: Vec3,
        cull_backface: bool,
    ) u32 {
        const c0 = transformToClip(view_proj, p0);
        const c1 = transformToClip(view_proj, p1);
        const c2 = transformToClip(view_proj, p2);

        const NEAR_W: f32 = 0.001;
        const in0 = c0.w >= NEAR_W;
        const in1 = c1.w >= NEAR_W;
        const in2 = c2.w >= NEAR_W;

        const count: u32 = (@as(u32, @intFromBool(in0)) + @as(u32, @intFromBool(in1)) + @as(u32, @intFromBool(in2)));

        if (count == 0) {
            // Triangle is entirely behind camera
            return 0;
        } else if (count == 3) {
            // Triangle is entirely in front of near plane
            rasterizeClippedTriangle(hiz, c0, c1, c2, cull_backface);
            return 1;
        } else if (count == 1) {
            // One vertex in front, two behind -> clips to 1 triangle
            var v_in: ClipVertex = undefined;
            var v_out0: ClipVertex = undefined;
            var v_out1: ClipVertex = undefined;

            if (in0) {
                v_in = c0;
                v_out0 = c1;
                v_out1 = c2;
            } else if (in1) {
                v_in = c1;
                v_out0 = c2;
                v_out1 = c0;
            } else {
                v_in = c2;
                v_out0 = c0;
                v_out1 = c1;
            }

            const t0 = (NEAR_W - v_in.w) / (v_out0.w - v_in.w);
            const t1 = (NEAR_W - v_in.w) / (v_out1.w - v_in.w);
            const new0 = lerpClip(v_in, v_out0, t0);
            const new1 = lerpClip(v_in, v_out1, t1);

            rasterizeClippedTriangle(hiz, v_in, new0, new1, cull_backface);
            return 1;
        } else {
            // Two vertices in front, one behind -> clips to a quad (2 triangles)
            var v_in0: ClipVertex = undefined;
            var v_in1: ClipVertex = undefined;
            var v_out: ClipVertex = undefined;

            if (!in0) {
                v_out = c0;
                v_in0 = c1;
                v_in1 = c2;
            } else if (!in1) {
                v_out = c1;
                v_in0 = c2;
                v_in1 = c0;
            } else {
                v_out = c2;
                v_in0 = c0;
                v_in1 = c1;
            }

            const t0 = (NEAR_W - v_in0.w) / (v_out.w - v_in0.w);
            const t1 = (NEAR_W - v_in1.w) / (v_out.w - v_in1.w);
            const new0 = lerpClip(v_in0, v_out, t0);
            const new1 = lerpClip(v_in1, v_out, t1);

            rasterizeClippedTriangle(hiz, v_in0, v_in1, new0, cull_backface);
            rasterizeClippedTriangle(hiz, v_in1, new1, new0, cull_backface);
            return 2;
        }
    }

    fn rasterizeClippedTriangle(
        hiz: *HiZBuffer,
        c0: ClipVertex,
        c1: ClipVertex,
        c2: ClipVertex,
        cull_backface: bool,
    ) void {
        const inv_w0 = 1.0 / c0.w;
        const inv_w1 = 1.0 / c1.w;
        const inv_w2 = 1.0 / c2.w;

        const ndc0_x = c0.x * inv_w0;
        const ndc0_y = c0.y * inv_w0;
        const ndc0_z = c0.z * inv_w0;

        const ndc1_x = c1.x * inv_w1;
        const ndc1_y = c1.y * inv_w1;
        const ndc1_z = c1.z * inv_w1;

        const ndc2_x = c2.x * inv_w2;
        const ndc2_y = c2.y * inv_w2;
        const ndc2_z = c2.z * inv_w2;

        const w_f = @as(f32, @floatFromInt(HiZBuffer.WIDTH - 1));
        const h_f = @as(f32, @floatFromInt(HiZBuffer.HEIGHT - 1));

        const sx0 = (ndc0_x * 0.5 + 0.5) * w_f;
        const sy0 = (1.0 - (ndc0_y * 0.5 + 0.5)) * h_f;

        const sx1 = (ndc1_x * 0.5 + 0.5) * w_f;
        const sy1 = (1.0 - (ndc1_y * 0.5 + 0.5)) * h_f;

        const sx2 = (ndc2_x * 0.5 + 0.5) * w_f;
        const sy2 = (1.0 - (ndc2_y * 0.5 + 0.5)) * h_f;

        // 2D cross product for winding / area
        const det = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);

        // Optional backface culling
        if (cull_backface and det <= 0.0) return;
        if (@abs(det) < 0.0001) return; // Degenerate / collinear

        const inv_det = 1.0 / det;

        const min_x_f = @max(0.0, @floor(@min(sx0, @min(sx1, sx2))));
        const max_x_f = @min(w_f, @ceil(@max(sx0, @max(sx1, sx2))));
        const min_y_f = @max(0.0, @floor(@min(sy0, @min(sy1, sy2))));
        const max_y_f = @min(h_f, @ceil(@max(sy0, @max(sy1, sy2))));

        if (min_x_f > max_x_f or min_y_f > max_y_f) return;

        const min_x: u32 = @intFromFloat(min_x_f);
        const max_x: u32 = @intFromFloat(max_x_f);
        const min_y: u32 = @intFromFloat(min_y_f);
        const max_y: u32 = @intFromFloat(max_y_f);

        var y: u32 = min_y;
        while (y <= max_y) : (y += 1) {
            const fy = @as(f32, @floatFromInt(y)) + 0.5;
            var x: u32 = min_x;
            while (x <= max_x) : (x += 1) {
                const fx = @as(f32, @floatFromInt(x)) + 0.5;

                // Barycentric coordinates
                const w0 = ((sx1 - fx) * (sy2 - fy) - (sx2 - fx) * (sy1 - fy)) * inv_det;
                const w1 = ((sx2 - fx) * (sy0 - fy) - (sx0 - fx) * (sy2 - fy)) * inv_det;
                const w2 = 1.0 - w0 - w1;

                // Conservative sign check (handling both clockwise and counter-clockwise)
                const inside = if (det > 0.0)
                    (w0 >= -0.0001 and w1 >= -0.0001 and w2 >= -0.0001)
                else
                    (w0 <= 0.0001 and w1 <= 0.0001 and w2 <= 0.0001);

                if (inside) {
                    const z = w0 * ndc0_z + w1 * ndc1_z + w2 * ndc2_z;
                    // Clamp to valid depth range
                    const cl_z = @max(0.0, @min(1.0, z));
                    hiz.writePixel(x, y, cl_z);
                }
            }
        }
    }

    /// Rasterizes an oriented bounding box (12 triangles) into the Hi-Z buffer.
    pub fn rasterizeBox(
        hiz: *HiZBuffer,
        view_proj: Mat4,
        aabb: BoundingBox,
        world_mat: Mat4,
    ) u32 {
        if (!aabb.isValid()) return 0;
        const local_corners = aabb.corners();
        var world_corners: [8]Vec3 = undefined;

        inline for (0..8) |i| {
            world_corners[i] = world_mat.transformPoint(local_corners[i]);
        }

        // Standard 12 triangles of a box (6 quad faces)
        const box_indices = [36]usize{
            // Front (+Z)
            4, 5, 6, 4, 6, 7,
            // Back (-Z)
            1, 0, 3, 1, 3, 2,
            // Left (-X)
            0, 4, 7, 0, 7, 3,
            // Right (+X)
            5, 1, 2, 5, 2, 6,
            // Top (+Y)
            7, 6, 2, 7, 2, 3,
            // Bottom (-Y)
            0, 1, 5, 0, 5, 4,
        };

        var count: u32 = 0;
        var i: usize = 0;
        while (i < 36) : (i += 3) {
            const p0 = world_corners[box_indices[i + 0]];
            const p1 = world_corners[box_indices[i + 1]];
            const p2 = world_corners[box_indices[i + 2]];
            count += rasterizeTriangle(hiz, view_proj, p0, p1, p2, true);
        }
        return count;
    }

    /// Rasterizes indexed triangles into the Hi-Z buffer.
    pub fn rasterizeTriangles(
        hiz: *HiZBuffer,
        view_proj: Mat4,
        positions: []const Vec3,
        indices: []const u32,
        world_mat: Mat4,
    ) u32 {
        if (positions.len < 3 or indices.len < 3) return 0;

        var count: u32 = 0;
        var i: usize = 0;
        while (i + 2 < indices.len) : (i += 3) {
            const idx0 = indices[i + 0];
            const idx1 = indices[i + 1];
            const idx2 = indices[i + 2];

            if (idx0 >= positions.len or idx1 >= positions.len or idx2 >= positions.len) continue;

            const p0_loc = positions[idx0];
            const p1_loc = positions[idx1];
            const p2_loc = positions[idx2];

            const p0 = world_mat.transformPoint(p0_loc);
            const p1 = world_mat.transformPoint(p1_loc);
            const p2 = world_mat.transformPoint(p2_loc);

            count += rasterizeTriangle(hiz, view_proj, p0, p1, p2, false);
        }
        return count;
    }
};
