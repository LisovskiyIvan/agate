const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const debug_shd = @import("debug_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const DebugLine = @import("../physics.zig").DebugLine;

// Interleaved line-list vertex: position + unpacked linear-space RGBA.
// RGBA8 packing (UBYTE4N) would halve vertex bandwidth, but debug overlays
// are a few thousand verts at most, while float4 keeps the CPU pack path
// trivial (no quantize/unpack, no endianness care) and matches the sokol
// FLOAT4 vertex format 1:1. Physics DebugLine already carries [3]f32 color.
pub const Vertex = extern struct {
    position: [3]f32,
    color: [4]f32,
};

pub const initial_capacity_lines: usize = 4096;
pub const max_capacity_lines: usize = 65536;

comptime {
    std.debug.assert(@sizeOf(Vertex) == 7 * @sizeOf(f32));
    std.debug.assert(@offsetOf(Vertex, "position") == 0);
    std.debug.assert(@offsetOf(Vertex, "color") == 3 * @sizeOf(f32));
}

/// Non-finite components (NaN/Inf from broken physics transforms) collapse
/// the whole line batch into garbage on some drivers, so sanitize on pack.
pub fn sanitizePositionComponent(v: f32) f32 {
    return if (std.math.isFinite(v)) v else 0.0;
}

pub fn sanitizeColorComponent(v: f32) f32 {
    if (!std.math.isFinite(v)) return 0.0;
    return std.math.clamp(v, 0.0, 1.0);
}

pub fn verticesForLineCount(line_count: usize) usize {
    return line_count * 2;
}

/// Doubling growth for the GPU-side line capacity (in lines, not verts).
pub fn grownCapacity(current_lines: usize, needed_lines: usize) usize {
    var cap = @max(current_lines, 1);
    while (cap < needed_lines) cap *= 2;
    return @min(cap, max_capacity_lines);
}

pub fn packDebugLine(line: DebugLine, out: *[2]Vertex) void {
    const c = [4]f32{
        sanitizeColorComponent(line.color[0]),
        sanitizeColorComponent(line.color[1]),
        sanitizeColorComponent(line.color[2]),
        1.0,
    };
    out.* = .{
        .{
            .position = .{
                sanitizePositionComponent(line.a.x),
                sanitizePositionComponent(line.a.y),
                sanitizePositionComponent(line.a.z),
            },
            .color = c,
        },
        .{
            .position = .{
                sanitizePositionComponent(line.b.x),
                sanitizePositionComponent(line.b.y),
                sanitizePositionComponent(line.b.z),
            },
            .color = c,
        },
    };
}

pub const DebugPass = struct {
    allocator: std.mem.Allocator,
    pipeline: sg.Pipeline = .{},
    vertex_buffer: sg.Buffer = .{},
    staging: std.ArrayListUnmanaged(Vertex) = .empty,
    capacity_lines: usize = 0,
    /// Main-target sample count the pipeline was built for (scene/msaa.zig).
    sample_count: i32 = 1,

    pub fn init(allocator: std.mem.Allocator) !DebugPass {
        return initSampled(allocator, 1);
    }

    /// Same pass at a different main-target sample count: sokol requires
    /// pipeline.sample_count to match the attachments of the pass it draws
    /// into.
    pub fn initSampled(allocator: std.mem.Allocator, sample_count: i32) !DebugPass {
        var staging: std.ArrayListUnmanaged(Vertex) = .empty;
        try staging.ensureTotalCapacity(allocator, verticesForLineCount(initial_capacity_lines));

        const vb = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .dynamic_update = true },
            .size = verticesForLineCount(initial_capacity_lines) * @sizeOf(Vertex),
        });

        var pip_desc = sg.PipelineDesc{
            .shader = sg.makeShader(debug_shd.debugShaderDesc(sg.queryBackend())),
            .index_type = .NONE,
            .primitive_type = .LINES,
            .depth = .{
                .compare = .LESS_EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = sample_count,
        };
        pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        pip_desc.layout.attrs[debug_shd.ATTR_debug_position] = .{
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        pip_desc.layout.attrs[debug_shd.ATTR_debug_color0] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "color"),
        };
        const pip = sg.makePipeline(pip_desc);

        return .{
            .allocator = allocator,
            .pipeline = pip,
            .vertex_buffer = vb,
            .staging = staging,
            .capacity_lines = initial_capacity_lines,
            .sample_count = sample_count,
        };
    }

    fn ensureCapacity(self: *DebugPass, needed_lines: usize) void {
        if (needed_lines <= self.capacity_lines) return;
        const new_cap = grownCapacity(self.capacity_lines, needed_lines);
        if (new_cap <= self.capacity_lines) return; // already at max; caller clamps
        self.staging.ensureTotalCapacity(self.allocator, verticesForLineCount(new_cap)) catch return;
        const new_vb = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .dynamic_update = true },
            .size = verticesForLineCount(new_cap) * @sizeOf(Vertex),
        });
        if (new_vb.id == 0) return;
        sg.destroyBuffer(self.vertex_buffer);
        self.vertex_buffer = new_vb;
        self.capacity_lines = new_cap;
    }

    /// Draws `lines` into the currently open pass (same contract as
    /// SkyboxPass.render: caller must have begun the main pass, whose depth
    /// attachment is reused for the LESS_EQUAL test). No depth view param
    /// needed — the pass reads the already-bound depth buffer.
    /// Overflow beyond max_capacity_lines is clamped (oldest lines win:
    /// prefix is drawn, tail dropped).
    pub fn render(self: *DebugPass, view_proj: Mat4, lines: []const DebugLine) void {
        if (lines.len == 0) return;
        if (self.pipeline.id == 0 or self.vertex_buffer.id == 0) return;
        self.ensureCapacity(lines.len);
        const drawable = lines[0..@min(lines.len, self.capacity_lines)];

        self.staging.clearRetainingCapacity();
        for (drawable) |line| {
            var pair: [2]Vertex = undefined;
            packDebugLine(line, &pair);
            self.staging.appendSliceAssumeCapacity(&pair);
        }
        if (self.staging.items.len == 0) return;

        sg.updateBuffer(self.vertex_buffer, sg.asRange(self.staging.items));
        sg.applyPipeline(self.pipeline);

        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.vertex_buffer;
        sg.applyBindings(bind);

        const vs_params = debug_shd.VsParams{
            .mvp = view_proj,
        };
        sg.applyUniforms(debug_shd.UB_vs_params, sg.asRange(&vs_params));

        sg.draw(0, @intCast(self.staging.items.len), 1);
    }

    pub fn deinit(self: *DebugPass) void {
        sg.destroyPipeline(self.pipeline);
        sg.destroyBuffer(self.vertex_buffer);
        self.staging.deinit(self.allocator);
        self.* = undefined;
    }
};

test "debug Vertex layout is tightly packed pos + rgba" {
    try std.testing.expectEqual(@as(usize, 28), @sizeOf(Vertex));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Vertex, "position"));
    try std.testing.expectEqual(@as(usize, 12), @offsetOf(Vertex, "color"));
}

test "packDebugLine emits two verts with opaque color" {
    const line = DebugLine{
        .a = .{ .x = 1.0, .y = 2.0, .z = 3.0 },
        .b = .{ .x = 4.0, .y = 5.0, .z = 6.0 },
        .color = .{ 0.1, 0.9, 0.3 },
    };
    var pair: [2]Vertex = undefined;
    packDebugLine(line, &pair);
    try std.testing.expectEqual([3]f32{ 1.0, 2.0, 3.0 }, pair[0].position);
    try std.testing.expectEqual([3]f32{ 4.0, 5.0, 6.0 }, pair[1].position);
    try std.testing.expectEqual([4]f32{ 0.1, 0.9, 0.3, 1.0 }, pair[0].color);
    try std.testing.expectEqual([4]f32{ 0.1, 0.9, 0.3, 1.0 }, pair[1].color);
    try std.testing.expectEqual(@as(usize, 2), verticesForLineCount(1));
    try std.testing.expectEqual(@as(usize, 0), verticesForLineCount(0));
}

test "packDebugLine sanitizes non-finite and out-of-range inputs" {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const line = DebugLine{
        .a = .{ .x = nan, .y = 1.0, .z = inf },
        .b = .{ .x = 0.0, .y = -inf, .z = 2.0 },
        .color = .{ 2.0, -1.0, nan },
    };
    var pair: [2]Vertex = undefined;
    packDebugLine(line, &pair);
    try std.testing.expectEqual([3]f32{ 0.0, 1.0, 0.0 }, pair[0].position);
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 2.0 }, pair[1].position);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 1.0 }, pair[0].color);
}

test "grownCapacity doubles and clamps at max" {
    try std.testing.expectEqual(@as(usize, 4096), grownCapacity(4096, 100));
    try std.testing.expectEqual(@as(usize, 4096), grownCapacity(4096, 4096));
    try std.testing.expectEqual(@as(usize, 8192), grownCapacity(4096, 4097));
    try std.testing.expectEqual(@as(usize, 16384), grownCapacity(4096, 16383));
    try std.testing.expectEqual(max_capacity_lines, grownCapacity(4096, max_capacity_lines * 4));
}
