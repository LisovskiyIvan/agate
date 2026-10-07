const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const debug_shd = @import("debug_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const DebugLine = @import("../physics.zig").DebugLine;
const upload_meter = @import("../gpu_upload_meter.zig");

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
    /// Main-target sample count the pipeline was built for.
    sample_count: i32 = 1,
    /// Main-target color format the pipeline was built for.
    color_format: sg.PixelFormat = .RGBA16F,
    shader: sg.Shader = .{},
    /// Vertices committed by the last upload(), drawn by drawPrepared().
    /// Upload-free draws read this, never the caller's slice — one upload
    /// serves all PIP views of the frame.
    prepared_verts: usize = 0,
    /// Same-sokol-frame guard (P6 UiFrame policy, pass-owned variant): the
    /// commit watermark of the last successful upload. A repeated prepare
    /// before the next sg.commit keeps the committed upload — sokol spends a
    /// single updateBuffer per buffer per frame. Headless (no commits exist)
    /// the window never opens, so repeated prepares always restage newest.
    upload_commit: u32 = 0,
    upload_armed: bool = false,

    /// Same pass for an explicit target shape (sample count + color
    /// format): each main-target shape needs its own pipeline variant.
    pub fn init(allocator: std.mem.Allocator, sample_count: i32, color_format: sg.PixelFormat) !DebugPass {
        var staging: std.ArrayListUnmanaged(Vertex) = .empty;
        try staging.ensureTotalCapacity(allocator, verticesForLineCount(initial_capacity_lines));

        const vb = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .write_transient = true },
            .size = verticesForLineCount(initial_capacity_lines) * @sizeOf(Vertex),
        });

        const shd = sg.makeShader(debug_shd.debugShaderDesc(sg.queryBackend()));
        var pip_desc = sg.PipelineDesc{
            .shader = shd,
            .index_type = .NONE,
            .primitive_type = .LINES,
            .depth = .{
                .compare = .LESS_EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = sample_count,
        };
        pip_desc.colors[0].pixel_format = color_format;
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
            .color_format = color_format,
            .shader = shd,
        };
    }

    fn ensureCapacity(self: *DebugPass, needed_lines: usize) void {
        if (needed_lines <= self.capacity_lines) return;
        const new_cap = grownCapacity(self.capacity_lines, needed_lines);
        if (new_cap <= self.capacity_lines) return; // already at max; caller clamps
        self.staging.ensureTotalCapacity(self.allocator, verticesForLineCount(new_cap)) catch return;
        if (!sg.isvalid()) {
            // Headless: CPU staging grows, but there is no GPU buffer to
            // replace (sg.makeBuffer traps without a context) — capacity
            // tracks the staging so clamping stays exact; draws no-op.
            self.capacity_lines = new_cap;
            return;
        }
        const new_vb = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .write_transient = true },
            .size = verticesForLineCount(new_cap) * @sizeOf(Vertex),
        });
        // A failed makeBuffer may hand out a nonzero FAILED id (pool
        // exhaustion is id == 0 only): validity is the state query, same as
        // the P5/P6 growth paths. Only the failed replacement is destroyed;
        // the current buffer and capacity stay, and the caller clamps to
        // them. No retire queue: prepare runs sequentially on the context
        // thread after the previous draw, so the replaced buffer is
        // already consumed.
        if (new_vb.id == 0 or sg.queryBufferState(new_vb) != .VALID) {
            if (new_vb.id != 0) sg.destroyBuffer(new_vb);
            return;
        }
        sg.destroyBuffer(self.vertex_buffer);
        self.vertex_buffer = new_vb;
        self.capacity_lines = new_cap;
    }

    /// True when this sokol frame's single `updateBuffer` is already spent on
    /// the line buffer. Read-only SDK metadata (`queryStats`, no GPU writes),
    /// `isvalid`-gated and headless-safe: without a context no commit can
    /// exist, so the window never opens and repeated prepares always restage
    /// the newest lines (CPU-only, no sg.* fires headless anyway).
    fn isUploadOpen(self: *const DebugPass) bool {
        if (!self.upload_armed) return false;
        if (!sg.isvalid()) return false;
        return sg.queryStats().prev_frame.frame_index == self.upload_commit;
    }

    fn markUploaded(self: *DebugPass) void {
        self.upload_commit = if (sg.isvalid()) sg.queryStats().prev_frame.frame_index else 0;
        self.upload_armed = true;
    }

    /// Stages `lines` into the GPU line buffer: pack (sanitize + clamp) +
    /// the frame's single `sg.updateBuffer`. Call ONCE per prepare; every PIP
    /// view of the frame then draws from the same upload via drawPrepared()
    /// with no per-view re-upload.
    ///
    /// Overflow beyond max_capacity_lines is clamped (oldest lines win:
    /// prefix is drawn, tail dropped) — same as render().
    /// Headless-safe: packing + staging are pure CPU, every `sg.*` sits
    /// behind `sg.isvalid()` (no meter bytes headless either).
    /// Returns true when drawable data is staged (live: on the GPU).
    pub fn upload(self: *DebugPass, lines: []const DebugLine) bool {
        if (lines.len == 0) {
            self.prepared_verts = 0;
            return false;
        }
        if (self.pipeline.id == 0 or self.vertex_buffer.id == 0) {
            self.prepared_verts = 0;
            return false;
        }
        // Repeat prepare inside one sokol frame: keep the committed upload
        // (first-wins, P6 policy); the next frame's prepare restages newest.
        if (self.isUploadOpen()) return self.prepared_verts > 0;
        self.ensureCapacity(lines.len);
        const drawable = lines[0..@min(lines.len, self.capacity_lines)];

        self.staging.clearRetainingCapacity();
        for (drawable) |line| {
            var pair: [2]Vertex = undefined;
            packDebugLine(line, &pair);
            self.staging.appendSliceAssumeCapacity(&pair);
        }
        if (self.staging.items.len == 0) {
            self.prepared_verts = 0;
            return false;
        }
        self.prepared_verts = self.staging.items.len;
        if (!sg.isvalid()) return true; // headless: staged, no GPU calls

        sg.writeBufferTransient(.{
            .dst = .{ .buffer = self.vertex_buffer },
            .src = .{ .data = sg.asRange(self.staging.items) },
        });
        // Учёт динамики: все staged debug-вершины кадра.
        upload_meter.record(self.staging.items.len * @sizeOf(Vertex));
        self.markUploaded();
        return true;
    }

    /// Upload-free draw of the last uploaded lines into the currently open
    /// pass (same contract as SkyboxPass.render: caller must have begun the
    /// main pass, whose depth attachment is reused for the LESS_EQUAL test).
    /// Reads ONLY the committed upload + the view projection — never the
    /// physics world, never the caller's line slice. Headless-safe no-op.
    /// Returns true only when the draw was actually issued (consumable
    /// upload + live context); counters belong to the caller and must be
    /// gated on this.
    pub fn drawPrepared(self: *DebugPass, view_proj: Mat4) bool {
        if (!sg.isvalid()) return false;
        if (self.prepared_verts == 0) return false;
        if (self.pipeline.id == 0 or self.vertex_buffer.id == 0) return false;
        if (self.staging.items.len == 0) return false;
        const count = @min(self.prepared_verts, self.staging.items.len);
        if (count == 0) return false;
        sg.applyPipeline(self.pipeline);

        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.vertex_buffer;
        sg.applyBindings(bind);

        const vs_params = debug_shd.VsParams{
            .mvp = view_proj,
        };
        sg.applyUniforms(debug_shd.UB_vs_params, sg.asRange(&vs_params));

        sg.draw(0, @intCast(count), 1);
        return true;
    }

    /// Draws `lines` into the currently open pass (same contract as
    /// SkyboxPass.render: caller must have begun the main pass, whose depth
    /// attachment is reused for the LESS_EQUAL test). No depth view param
    /// needed — the pass reads the already-bound depth buffer.
    /// Overflow beyond max_capacity_lines is clamped (oldest lines win:
    /// prefix is drawn, tail dropped).
    /// Single-view path (upload + draw); multi-view callers upload
    /// once and drawPrepared per view instead.
    pub fn render(self: *DebugPass, view_proj: Mat4, lines: []const DebugLine) void {
        if (!self.upload(lines)) return;
        _ = self.drawPrepared(view_proj);
    }

    pub fn deinit(self: *DebugPass) void {
        sg.destroyPipeline(self.pipeline);
        if (self.shader.id != 0) sg.destroyShader(self.shader);
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

test "upload stages once headless, drawPrepared is a safe no-op" {
    const t = std.testing;
    _ = upload_meter.takeAndReset();
    // Fake handles, no sokol context: upload packs CPU-side only (no sg.*,
    // no meter bytes); drawPrepared must never trap headless.
    var pass = DebugPass{
        .allocator = t.allocator,
        .pipeline = .{ .id = 1 },
        .vertex_buffer = .{ .id = 2 },
        .capacity_lines = 1,
    };
    defer pass.staging.deinit(t.allocator);
    try pass.staging.ensureTotalCapacity(t.allocator, verticesForLineCount(4));

    const lines = [_]DebugLine{
        .{ .a = .{ .x = 0, .y = 0, .z = 0 }, .b = .{ .x = 1, .y = 0, .z = 0 }, .color = .{ 1, 0, 0 } },
        .{ .a = .{ .x = 0, .y = 1, .z = 0 }, .b = .{ .x = 0, .y = 2, .z = 0 }, .color = .{ 0, 1, 0 } },
    };
    // Headless growth (2 lines > capacity 1): CPU staging grows with no
    // sg.makeBuffer (which traps without a context); clamping stays exact.
    try t.expect(pass.upload(&lines));
    try t.expectEqual(@as(usize, 2), pass.capacity_lines);
    try t.expectEqual(@as(usize, 4), pass.prepared_verts);
    try t.expectEqual(@as(usize, 4), pass.staging.items.len);
    try t.expectEqual(@as(u64, 0), upload_meter.peek());

    // Headless the same-frame window never opens (no commits exist): a
    // repeated upload restages the newest lines instead of first-wins.
    const moved = [_]DebugLine{
        .{ .a = .{ .x = 9, .y = 0, .z = 0 }, .b = .{ .x = 10, .y = 0, .z = 0 }, .color = .{ 0, 0, 1 } },
    };
    try t.expect(pass.upload(&moved));
    try t.expectEqual(@as(usize, 2), pass.prepared_verts);
    try t.expectEqual(@as(f32, 9.0), pass.staging.items[0].position[0]);

    // Empty input fail-closes the prepared count; draws stay safe no-ops.
    try t.expect(!pass.upload(&.{}));
    try t.expectEqual(@as(usize, 0), pass.prepared_verts);
    try t.expect(!pass.drawPrepared(Mat4.identity));
    try t.expectEqual(@as(u64, 0), upload_meter.peek());

    // Zero handles fail close without touching staging.
    var dead = DebugPass{ .allocator = t.allocator };
    defer dead.staging.deinit(t.allocator);
    try t.expect(!dead.upload(&lines));
    try t.expectEqual(@as(usize, 0), dead.prepared_verts);
}

test "upload clamps at max capacity, oldest lines win" {
    const t = std.testing;
    var pass = DebugPass{
        .allocator = t.allocator,
        .pipeline = .{ .id = 1 },
        .vertex_buffer = .{ .id = 2 },
        .capacity_lines = max_capacity_lines,
    };
    defer pass.staging.deinit(t.allocator);
    try pass.staging.ensureTotalCapacity(t.allocator, verticesForLineCount(max_capacity_lines));

    // max + 2 lines: growth is capped, so the prefix draws, tail drops —
    // same clamp as render().
    const n = max_capacity_lines + 2;
    const lines = try t.allocator.alloc(DebugLine, n);
    defer t.allocator.free(lines);
    for (lines, 0..) |*ln, i| {
        const x: f32 = @floatFromInt(i);
        ln.* = .{
            .a = .{ .x = x, .y = 0, .z = 0 },
            .b = .{ .x = x + 0.5, .y = 0, .z = 0 },
            .color = .{ 1, 0, 0 },
        };
    }
    try t.expect(pass.upload(lines));
    try t.expectEqual(verticesForLineCount(max_capacity_lines), pass.prepared_verts);
    try t.expectEqual(@as(f32, 0.0), pass.staging.items[0].position[0]);
    const last: f32 = @floatFromInt(max_capacity_lines - 1);
    try t.expectEqual(last, pass.staging.items[pass.staging.items.len - 2].position[0]);
    try t.expectEqual(last + 0.5, pass.staging.items[pass.staging.items.len - 1].position[0]);
}
