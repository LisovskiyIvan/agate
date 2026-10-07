//! Vertex Animation Texture (VAT) subsystem for Agate.
//!
//! VAT bakes complex vertex animations (skeletal meshes, cloth simulations,
//! morph sequences, procedural deformations) into 2D float textures.
//!
//! Layout:
//!   - Position Texture: RGBA32F (or RGBA16F), where:
//!       X = vertex index
//!       Y = animation frame index
//!       Texel (X, Y) = vec4(position.x, position.y, position.z, 1.0)
//!   - Normal Texture (optional): RGBA16F / RGBA32F, where:
//!       Texel (X, Y) = vec4(normal.x, normal.y, normal.z, 0.0)
//!
//! Runtime:
//!   Vertex shaders sample texels at (vertex_id, frame_time) and interpolate
//!   between adjacent frames. Enables rendering thousands of animated instanced
//!   characters in a single draw call with zero CPU animation/skinning overhead.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;

const types = @import("types.zig");
const SkinJointWeight = types.SkinJointWeight;
const Mesh = @import("mesh.zig").Mesh;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const AnimationGroup = @import("../animation/group.zig").AnimationGroup;

pub const VatLayout = enum {
    /// Width = vertex count, Height = frame count (direct 2D indexing)
    grid,
    /// 1D linear texel strip wrapped at texture_max_width (for meshes with > 4096 vertices)
    strip,
};

pub const VatConfig = struct {
    fps: f32 = 30.0,
    include_normals: bool = true,
    /// Wrap width when vertex count exceeds max width
    texture_max_width: u32 = 4096,
};

pub const VatSampleParams = struct {
    frame0: u32,
    frame1: u32,
    lerp_frac: f32,
    total_frames: f32,

    pub fn toVec4(self: VatSampleParams) [4]f32 {
        return .{
            @floatFromInt(self.frame0),
            @floatFromInt(self.frame1),
            self.lerp_frac,
            self.total_frames,
        };
    }
};

pub const VatData = struct {
    allocator: std.mem.Allocator,
    vertex_count: u32,
    frame_count: u32,
    fps: f32 = 30.0,
    layout: VatLayout = .grid,
    texture_width: u32,
    texture_height: u32,

    // CPU raw float buffers (RGBA: 4 floats per vertex/frame)
    positions: []f32,
    normals: ?[]f32 = null,

    // GPU resources
    position_image: sg.Image = .{},
    position_view: sg.View = .{},
    normal_image: sg.Image = .{},
    normal_view: sg.View = .{},

    // Bounds across the entire animation
    bounding_box: BoundingBox = BoundingBox.zero,
    frame_aabbs: []BoundingBox = &.{},

    pub fn deinit(self: *VatData) void {
        if (self.position_image.id != 0) {
            sg.destroyImage(self.position_image);
            self.position_image = .{};
        }
        if (self.normal_image.id != 0) {
            sg.destroyImage(self.normal_image);
            self.normal_image = .{};
        }
        self.allocator.free(self.positions);
        if (self.normals) |n| {
            self.allocator.free(n);
            self.normals = null;
        }
        if (self.frame_aabbs.len > 0) {
            self.allocator.free(self.frame_aabbs);
            self.frame_aabbs = &.{};
        }
    }

    /// Computes texel coordinates (X, Y) in pixel space for a vertex and frame.
    pub fn texelCoord(self: VatData, vertex_idx: usize, frame_idx: usize) struct { x: u32, y: u32 } {
        switch (self.layout) {
            .grid => {
                return .{
                    .x = @intCast(vertex_idx),
                    .y = @intCast(frame_idx),
                };
            },
            .strip => {
                const flat_idx: u64 = @as(u64, frame_idx) * self.vertex_count + vertex_idx;
                const w: u64 = self.texture_width;
                return .{
                    .x = @intCast(flat_idx % w),
                    .y = @intCast(flat_idx / w),
                };
            },
        }
    }

    /// Flat index into the 4-float texel slice `positions` / `normals`.
    pub fn texelFloatIndex(self: VatData, vertex_idx: usize, frame_idx: usize) usize {
        const c = self.texelCoord(vertex_idx, frame_idx);
        return (@as(usize, c.y) * self.texture_width + c.x) * 4;
    }

    /// Samples the vertex position for a discrete frame.
    pub fn samplePosition(self: VatData, vertex_idx: usize, frame_idx: usize) Vec3 {
        if (vertex_idx >= self.vertex_count) return Vec3.zero;
        const f = @min(frame_idx, self.frame_count - 1);
        const idx = self.texelFloatIndex(vertex_idx, f);
        return Vec3.new(self.positions[idx], self.positions[idx + 1], self.positions[idx + 2]);
    }

    /// Samples the vertex position with sub-frame linear interpolation.
    pub fn samplePositionInterpolated(self: VatData, vertex_idx: usize, frame_time: f32) Vec3 {
        if (self.frame_count == 0 or self.vertex_count == 0) return Vec3.zero;
        if (!std.math.isFinite(frame_time)) return Vec3.zero;
        const total: f32 = @floatFromInt(self.frame_count);
        const wrapped = @mod(frame_time, total);
        const f0_idx: usize = @intFromFloat(@floor(wrapped));
        const f1_idx: usize = (f0_idx + 1) % self.frame_count;
        const frac = wrapped - @floor(wrapped);

        const p0 = self.samplePosition(vertex_idx, f0_idx);
        const p1 = self.samplePosition(vertex_idx, f1_idx);
        return p0.lerp(p1, frac);
    }

    /// Samples the vertex normal for a discrete frame.
    pub fn sampleNormal(self: VatData, vertex_idx: usize, frame_idx: usize) Vec3 {
        const norms = self.normals orelse return Vec3.up;
        if (vertex_idx >= self.vertex_count) return Vec3.up;
        const f = @min(frame_idx, self.frame_count - 1);
        const idx = self.texelFloatIndex(vertex_idx, f);
        return Vec3.new(norms[idx], norms[idx + 1], norms[idx + 2]);
    }

    /// Samples the vertex normal with sub-frame interpolation and normalization.
    pub fn sampleNormalInterpolated(self: VatData, vertex_idx: usize, frame_time: f32) Vec3 {
        if (self.normals == null or self.frame_count == 0) return Vec3.up;
        if (!std.math.isFinite(frame_time)) return Vec3.up;
        const total: f32 = @floatFromInt(self.frame_count);
        const wrapped = @mod(frame_time, total);
        const f0_idx: usize = @intFromFloat(@floor(wrapped));
        const f1_idx: usize = (f0_idx + 1) % self.frame_count;
        const frac = wrapped - @floor(wrapped);

        const n0 = self.sampleNormal(vertex_idx, f0_idx);
        const n1 = self.sampleNormal(vertex_idx, f1_idx);
        const n = n0.lerp(n1, frac);
        return if (n.length() > 0.0001) n.normalize() else Vec3.up;
    }

    /// Uploads CPU float buffers into Sokol GPU textures (RGBA32F).
    pub fn uploadTextures(self: *VatData) !void {
        var pos_img_desc = sg.ImageDesc{
            .width = @intCast(self.texture_width),
            .height = @intCast(self.texture_height),
            .pixel_format = .RGBA32F,
            .usage = .{ .immutable = true },
        };
        pos_img_desc.data.mip_levels[0] = sg.asRange(self.positions);
        self.position_image = sg.makeImage(pos_img_desc);
        if (self.position_image.id == 0) return error.GpuImageCreationFailed;

        self.position_view = sg.makeView(.{
            .texture = .{ .image = self.position_image },
        });

        if (self.normals) |norms| {
            var norm_img_desc = sg.ImageDesc{
                .width = @intCast(self.texture_width),
                .height = @intCast(self.texture_height),
                .pixel_format = .RGBA32F,
                .usage = .{ .immutable = true },
            };
            norm_img_desc.data.mip_levels[0] = sg.asRange(norms);
            self.normal_image = sg.makeImage(norm_img_desc);
            if (self.normal_image.id != 0) {
                self.normal_view = sg.makeView(.{
                    .texture = .{ .image = self.normal_image },
                });
            }
        }
    }

    /// Calculates overall bounding box and per-frame bounding boxes.
    pub fn calculateBounds(self: *VatData) !void {
        if (self.frame_aabbs.len != self.frame_count) {
            if (self.frame_aabbs.len > 0) self.allocator.free(self.frame_aabbs);
            self.frame_aabbs = try self.allocator.alloc(BoundingBox, self.frame_count);
        }

        var overall = BoundingBox.zero;
        var first = true;

        for (0..self.frame_count) |f| {
            var frame_box = BoundingBox.zero;
            var frame_first = true;
            for (0..self.vertex_count) |v| {
                const pos = self.samplePosition(v, f);
                if (frame_first) {
                    frame_box.min = pos;
                    frame_box.max = pos;
                    frame_first = false;
                } else {
                    frame_box.min = Vec3.new(@min(frame_box.min.x, pos.x), @min(frame_box.min.y, pos.y), @min(frame_box.min.z, pos.z));
                    frame_box.max = Vec3.new(@max(frame_box.max.x, pos.x), @max(frame_box.max.y, pos.y), @max(frame_box.max.z, pos.z));
                }
            }
            self.frame_aabbs[f] = frame_box;
            if (first) {
                overall = frame_box;
                first = false;
            } else {
                overall.min = Vec3.new(@min(overall.min.x, frame_box.min.x), @min(overall.min.y, frame_box.min.y), @min(overall.min.z, frame_box.min.z));
                overall.max = Vec3.new(@max(overall.max.x, frame_box.max.x), @max(overall.max.y, frame_box.max.y), @max(overall.max.z, frame_box.max.z));
            }
        }
        self.bounding_box = overall;
    }
};

pub const VatBaker = struct {
    /// Bakes a skeletal animation into VAT textures using CPU software skinning.
    pub fn bakeSkeletal(
        allocator: std.mem.Allocator,
        mesh: *const Mesh,
        skeleton: *Skeleton,
        anim_group: *AnimationGroup,
        config: VatConfig,
    ) !VatData {
        const vertex_count: u32 = @intCast(mesh.cpu_positions.len);
        if (vertex_count == 0) return error.MeshHasNoCpuPositions;

        const fps = @max(1.0, config.fps);
        const duration = anim_group.duration();
        const frame_count: u32 = @max(1, @as(u32, @intFromFloat(@ceil(duration * fps))));

        const layout: VatLayout = if (vertex_count <= config.texture_max_width) .grid else .strip;
        const tex_w: u32 = switch (layout) {
            .grid => vertex_count,
            .strip => config.texture_max_width,
        };
        const tex_h: u32 = switch (layout) {
            .grid => frame_count,
            .strip => @intCast((@as(u64, frame_count) * vertex_count + tex_w - 1) / tex_w),
        };

        const total_texels: usize = @as(usize, tex_w) * tex_h;
        const pos_buffer = try allocator.alloc(f32, total_texels * 4);
        errdefer allocator.free(pos_buffer);
        @memset(pos_buffer, 0);

        var norm_buffer: ?[]f32 = null;
        if (config.include_normals) {
            norm_buffer = try allocator.alloc(f32, total_texels * 4);
            @memset(norm_buffer.?, 0);
        }
        errdefer if (norm_buffer) |nb| allocator.free(nb);

        var vat = VatData{
            .allocator = allocator,
            .vertex_count = vertex_count,
            .frame_count = frame_count,
            .fps = fps,
            .layout = layout,
            .texture_width = tex_w,
            .texture_height = tex_h,
            .positions = pos_buffer,
            .normals = norm_buffer,
        };

        const dt: f32 = 1.0 / fps;
        const has_skin = mesh.cpu_skin.len >= vertex_count;

        for (0..frame_count) |f| {
            const time: f32 = @as(f32, @floatFromInt(f)) * dt;
            anim_group.seek(time);
            anim_group.apply(skeleton);
            skeleton.update();

            for (0..vertex_count) |v| {
                const bind_pos = mesh.cpu_positions[v];
                var skinned_pos = bind_pos;

                if (has_skin) {
                    const skin = mesh.cpu_skin[v];
                    var accum = Vec3.zero;
                    for (0..4) |j_idx| {
                        const weight = skin.weights[j_idx];
                        if (weight <= 0.0001) continue;
                        const bone_idx: usize = @intFromFloat(skin.joints[j_idx]);
                        if (bone_idx < Skeleton.MAX_BONES) {
                            const bone_mat = skeleton.skin_matrices[bone_idx];
                            const transformed = bone_mat.transformPoint(bind_pos);
                            accum = accum.add(transformed.scale(weight));
                        }
                    }
                    if (accum.lengthSq() > 0.00001) {
                        skinned_pos = accum;
                    }
                }

                const out_idx = vat.texelFloatIndex(v, f);
                pos_buffer[out_idx + 0] = skinned_pos.x;
                pos_buffer[out_idx + 1] = skinned_pos.y;
                pos_buffer[out_idx + 2] = skinned_pos.z;
                pos_buffer[out_idx + 3] = 1.0;

                if (norm_buffer) |nb| {
                    nb[out_idx + 0] = 0.0;
                    nb[out_idx + 1] = 1.0;
                    nb[out_idx + 2] = 0.0;
                    nb[out_idx + 3] = 0.0;
                }
            }
        }

        try vat.calculateBounds();
        return vat;
    }

    /// Bakes procedural deformations into VAT textures.
    /// `generator` signature: `fn (frame: u32, time: f32, vertex_idx: u32, out_pos: *Vec3, out_nrm: ?*Vec3) void`
    pub fn bakeProcedural(
        allocator: std.mem.Allocator,
        vertex_count: u32,
        frame_count: u32,
        fps: f32,
        include_normals: bool,
        generator_ctx: anytype,
        comptime generator_fn: anytype,
    ) !VatData {
        if (vertex_count == 0 or frame_count == 0) return error.InvalidDimensions;

        const layout: VatLayout = if (vertex_count <= 4096) .grid else .strip;
        const tex_w: u32 = switch (layout) {
            .grid => vertex_count,
            .strip => 4096,
        };
        const tex_h: u32 = switch (layout) {
            .grid => frame_count,
            .strip => @intCast((@as(u64, frame_count) * vertex_count + tex_w - 1) / tex_w),
        };

        const total_texels: usize = @as(usize, tex_w) * tex_h;
        const pos_buffer = try allocator.alloc(f32, total_texels * 4);
        errdefer allocator.free(pos_buffer);
        @memset(pos_buffer, 0);

        var norm_buffer: ?[]f32 = null;
        if (include_normals) {
            norm_buffer = try allocator.alloc(f32, total_texels * 4);
            @memset(norm_buffer.?, 0);
        }
        errdefer if (norm_buffer) |nb| allocator.free(nb);

        var vat = VatData{
            .allocator = allocator,
            .vertex_count = vertex_count,
            .frame_count = frame_count,
            .fps = fps,
            .layout = layout,
            .texture_width = tex_w,
            .texture_height = tex_h,
            .positions = pos_buffer,
            .normals = norm_buffer,
        };

        const dt = 1.0 / fps;
        for (0..frame_count) |f| {
            const time = @as(f32, @floatFromInt(f)) * dt;
            for (0..vertex_count) |v| {
                var p = Vec3.zero;
                var n = Vec3.up;
                generator_fn(generator_ctx, @intCast(f), time, @intCast(v), &p, if (include_normals) &n else null);

                const out_idx = vat.texelFloatIndex(v, f);
                pos_buffer[out_idx + 0] = p.x;
                pos_buffer[out_idx + 1] = p.y;
                pos_buffer[out_idx + 2] = p.z;
                pos_buffer[out_idx + 3] = 1.0;

                if (norm_buffer) |nb| {
                    nb[out_idx + 0] = n.x;
                    nb[out_idx + 1] = n.y;
                    nb[out_idx + 2] = n.z;
                    nb[out_idx + 3] = 0.0;
                }
            }
        }

        try vat.calculateBounds();
        return vat;
    }
};

pub const VatPlayer = struct {
    vat: *const VatData,
    time: f32 = 0.0,
    speed: f32 = 1.0,
    loop: bool = true,

    pub fn init(vat: *const VatData) VatPlayer {
        return .{ .vat = vat };
    }

    pub fn update(self: *VatPlayer, dt: f32) void {
        self.time += dt * self.speed;
        // Guard against degenerate VatData (frame_count=0 / fps=0): the
        // division would yield 0/0=NaN, and `NaN <= 0` is false, so the
        // early-out below would NOT fire — @mod(time, NaN) would then
        // poison `time` forever (NaN mesh transforms downstream). Zero or
        // non-finite config means "nothing to play": clamp to zero.
        if (self.vat.frame_count == 0 or !(self.vat.fps > 0.0)) {
            self.time = 0.0;
            return;
        }
        const total_duration = @as(f32, @floatFromInt(self.vat.frame_count)) / self.vat.fps;
        if (!(total_duration > 0.0)) {
            self.time = 0.0;
            return;
        }

        if (self.loop) {
            self.time = @mod(self.time, total_duration);
        } else {
            self.time = std.math.clamp(self.time, 0.0, total_duration);
        }
    }

    pub fn seek(self: *VatPlayer, time_seconds: f32) void {
        if (self.vat.frame_count == 0 or !(self.vat.fps > 0.0)) return;
        const total_duration = @as(f32, @floatFromInt(self.vat.frame_count)) / self.vat.fps;
        if (!(total_duration > 0.0)) return;
        if (self.loop) {
            self.time = @mod(time_seconds, total_duration);
        } else {
            self.time = std.math.clamp(time_seconds, 0.0, total_duration);
        }
    }

    /// Computes discrete frames and sub-frame fraction for shader uniform sampling.
    pub fn getSampleParams(self: VatPlayer) VatSampleParams {
        const total_frames = self.vat.frame_count;
        if (total_frames == 0) return .{ .frame0 = 0, .frame1 = 0, .lerp_frac = 0.0, .total_frames = 1.0 };

        const frame_time = self.time * self.vat.fps;
        const total_f: f32 = @floatFromInt(total_frames);
        const wrapped = @mod(frame_time, total_f);
        const f0 = @as(u32, @intFromFloat(@floor(wrapped)));
        const f1 = (f0 + 1) % total_frames;
        const frac = wrapped - @floor(wrapped);

        return .{
            .frame0 = f0,
            .frame1 = f1,
            .lerp_frac = frac,
            .total_frames = total_f,
        };
    }

    pub fn currentFrame(self: VatPlayer) u32 {
        return self.getSampleParams().frame0;
    }
};

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "VAT texel coordination grid and strip layout" {
    const ally = std.testing.allocator;
    const pos = try ally.alloc(f32, 100 * 4);
    defer ally.free(pos);
    @memset(pos, 0);

    const grid_vat = VatData{
        .allocator = ally,
        .vertex_count = 10,
        .frame_count = 5,
        .layout = .grid,
        .texture_width = 10,
        .texture_height = 5,
        .positions = pos,
    };
    const c0 = grid_vat.texelCoord(3, 2);
    try std.testing.expectEqual(@as(u32, 3), c0.x);
    try std.testing.expectEqual(@as(u32, 2), c0.y);
    try std.testing.expectEqual(@as(usize, (2 * 10 + 3) * 4), grid_vat.texelFloatIndex(3, 2));

    const strip_vat = VatData{
        .allocator = ally,
        .vertex_count = 100,
        .frame_count = 10,
        .layout = .strip,
        .texture_width = 40,
        .texture_height = 25,
        .positions = pos,
    };
    // Frame 1, vertex 50 -> flat index 150 -> x = 150 % 40 = 30, y = 150 / 40 = 3
    const c1 = strip_vat.texelCoord(50, 1);
    try std.testing.expectEqual(@as(u32, 30), c1.x);
    try std.testing.expectEqual(@as(u32, 3), c1.y);
}

fn testWaveGenerator(_: void, frame: u32, _: f32, vert_id: u32, out_pos: *Vec3, out_nrm: ?*Vec3) void {
    const v: f32 = @floatFromInt(vert_id);
    const f: f32 = @floatFromInt(frame);
    // Sine wave animated along Y
    out_pos.* = Vec3.new(v, @sin(v * 0.5 + f * 0.2), 0.0);
    if (out_nrm) |n| {
        n.* = Vec3.up;
    }
}

test "VAT bakeProcedural, samplePosition and bounds" {
    const ally = std.testing.allocator;
    var vat = try VatBaker.bakeProcedural(ally, 8, 10, 30.0, true, {}, testWaveGenerator);
    defer vat.deinit();

    try std.testing.expectEqual(@as(u32, 8), vat.vertex_count);
    try std.testing.expectEqual(@as(u32, 10), vat.frame_count);
    try std.testing.expect(vat.normals != null);

    // Vertex 2 at frame 0: y = sin(2 * 0.5 + 0) = sin(1.0)
    const p0 = vat.samplePosition(2, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), p0.x, 1e-5);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), p0.y, 1e-5);

    // Sub-frame interpolated sampling at frame 0.5
    const p_mid = vat.samplePositionInterpolated(2, 0.5);
    const p1 = vat.samplePosition(2, 1);
    const expected_mid_y = (@sin(@as(f32, 1.0)) + p1.y) * 0.5;
    try std.testing.expectApproxEqAbs(expected_mid_y, p_mid.y, 1e-4);

    // Bounds must encompass min and max of all vertices across all frames
    try std.testing.expect(vat.bounding_box.min.x <= 0.0);
    try std.testing.expect(vat.bounding_box.max.x >= 7.0);
    try std.testing.expect(vat.bounding_box.min.y >= -1.0);
    try std.testing.expect(vat.bounding_box.max.y <= 1.0);
}

test "VatPlayer playback and uniform parameter calculation" {
    const ally = std.testing.allocator;
    var vat = try VatBaker.bakeProcedural(ally, 4, 10, 30.0, false, {}, testWaveGenerator);
    defer vat.deinit();

    var player = VatPlayer.init(&vat);
    player.speed = 1.0;

    // At t=0: frame 0 -> 1, frac 0.0
    var params = player.getSampleParams();
    try std.testing.expectEqual(@as(u32, 0), params.frame0);
    try std.testing.expectEqual(@as(u32, 1), params.frame1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), params.lerp_frac, 1e-5);

    // Advance by half a frame: dt = 0.5 / 30.0 seconds
    player.update(0.5 / 30.0);
    params = player.getSampleParams();
    try std.testing.expectEqual(@as(u32, 0), params.frame0);
    try std.testing.expectEqual(@as(u32, 1), params.frame1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), params.lerp_frac, 1e-4);

    // Advance past end of animation: should loop back
    player.seek(10.0 / 30.0 + 0.25 / 30.0); // 10 frames = full cycle + 0.25 frame
    params = player.getSampleParams();
    try std.testing.expectEqual(@as(u32, 0), params.frame0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), params.lerp_frac, 1e-4);
}
