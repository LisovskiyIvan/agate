const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Mesh = @import("../mesh.zig").Mesh;
const Vertex = @import("../mesh.zig").Vertex;
const camera_mod = @import("../camera.zig");
const Viewport = camera_mod.Viewport;
const HighlightOptions = @import("../scene/highlight_layer.zig").HighlightOptions;
const glow_mod = @import("glow_pass.zig");
const outline_shd = @import("outline_shader");

const hp = @import("highlight_pass.zig");
const HighlightPass = hp.HighlightPass;
const HighlightDrawItem = hp.HighlightDrawItem;
const makeHighlightDrawItem = hp.makeHighlightDrawItem;
const highlightMaskColor = hp.highlightMaskColor;
const highlightFrameSigma = hp.highlightFrameSigma;
const highlightMaskViewport = hp.highlightMaskViewport;
const configureHighlightMaskDesc = hp.configureHighlightMaskDesc;

test "highlight pass fail-closes headless with no state touched" {
    const upload_meter = @import("../gpu_upload_meter.zig");
    // Zero-initialized pass (never init'ed: no sg context headless, same
    // as the GlowPass fail-closed shape) must return empty before any
    // sg.* call — disabled highlights touch nothing.
    var pass: HighlightPass = .{};
    _ = upload_meter.takeAndReset();
    const full = Viewport.PixelRect{ .x = 0, .y = 0, .width = 1280, .height = 720 };
    const empty = pass.render(Mat4.identity, &.{}, 1280, 720, full);
    try std.testing.expectEqual(@as(u32, 0), empty.view.id);
    try std.testing.expectEqual(@as(u32, 0), empty.mask_view.id);
    try std.testing.expectEqual(@as(u32, 0), empty.mask_draws);
    try std.testing.expectEqual(@as(u32, 0), empty.mask_tris);
    // Empty items, degenerate size, and missing pipelines all fail closed
    // the same way (guard order: pipelines first, then items, then size).
    const item = HighlightDrawItem{ .index_count = 3 };
    const no_pipe = pass.render(Mat4.identity, &[_]HighlightDrawItem{item}, 1280, 720, full);
    try std.testing.expectEqual(@as(u32, 0), no_pipe.view.id);
    try std.testing.expectEqual(@as(u32, 0), no_pipe.mask_view.id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(Mat4.identity, &[_]HighlightDrawItem{item}, 0, 720, full).view.id);
    // Fail-closed render records no GPU uploads (uniform-only past the
    // mask binds: replay in renderReuse stays upload-free).
    try std.testing.expectEqual(@as(u64, 0), upload_meter.takeAndReset());
    // Base size untouched: no resize happened.
    try std.testing.expectEqual(@as(i32, 0), pass.base_width);
    try std.testing.expectEqual(@as(i32, 0), pass.base_height);
}

test "highlight target bytes share the three-half-res-target shape" {
    // Same formula as GlowPass (mask + H/V ping-pong): exact and
    // deterministic headless, pinned through the shared helper.
    try std.testing.expectEqual(glow_mod.GlowPass.targetBytes(1280, 720, 4), HighlightPass.targetBytes(1280, 720, 4));
    try std.testing.expectEqual(@as(usize, 3 * 640 * 360 * 4), HighlightPass.targetBytes(1280, 720, 4));
    try std.testing.expectEqual(@as(usize, 3 * 640 * 360 * 8), HighlightPass.targetBytes(1280, 720, 8));
    try std.testing.expectEqual(@as(usize, 3 * 1 * 1 * 4), HighlightPass.targetBytes(0, 0, 4));
}

test "makeHighlightDrawItem stages a render-owned snapshot" {
    var mesh = Mesh{
        .name = "hl_stage",
        .vertex_buffer = .{ .id = 11 },
        .index_buffer = .{ .id = 12 },
        .index_count = 36,
        .position = Vec3.new(4, 0, 0),
    };
    const opts = HighlightOptions{ .color = .{ 0.2, 0.4, 0.6, 1.0 }, .blur = 6.0, .intensity = 0.8 };
    const it = makeHighlightDrawItem(&mesh, opts, 5) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 5), it.source_mesh);
    try std.testing.expect(it.source_uid != 0);
    try std.testing.expectEqual(mesh.uid, it.source_uid);
    try std.testing.expectEqual(opts.color, it.color);
    try std.testing.expect(!it.is_u32);

    // The staged model survives live TRS mutation (no live reads at draw).
    mesh.position = Vec3.new(99, 99, 99);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), it.model.m[12], 1e-4);

    // Fail-closed filters (outline pre-filter precedent). Dead buffer
    // handles stage fine — the mask draw skips them (renderMask guards).
    mesh.is_visible = false;
    try std.testing.expect(makeHighlightDrawItem(&mesh, opts, 5) == null);
    mesh.is_visible = true;
    mesh.gpu_pending = true;
    try std.testing.expect(makeHighlightDrawItem(&mesh, opts, 5) == null);
    mesh.gpu_pending = false;
    mesh.index_count = 0;
    try std.testing.expect(makeHighlightDrawItem(&mesh, opts, 5) == null);
    mesh.index_count = 36;
    mesh.vertex_buffer = .{};
    const dead = makeHighlightDrawItem(&mesh, opts, 5) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 0), dead.vertex_buffer.id);
}

test "makeHighlightDrawItem skips skinned meshes, proxies instanced ones" {
    const Skeleton = @import("../animation/skeleton.zig").Skeleton;
    const ally = std.testing.allocator;
    const skel = try Skeleton.init(ally, 1);
    defer skel.deinit();

    // Skinned: fail-closed skip (v1 stages no skin matrices — never a
    // bind-pose draw).
    var skinned = Mesh{
        .name = "hl_skinned",
        .vertex_buffer = .{ .id = 1 },
        .index_buffer = .{ .id = 2 },
        .index_count = 3,
        .skeleton = skel,
    };
    try std.testing.expect(makeHighlightDrawItem(&skinned, .{}, 0) == null);

    // Instanced: the template proxy stages (mesh world matrix, single
    // draw — no per-instance matrices, documented v1 limit).
    var instanced = Mesh{
        .name = "hl_instanced",
        .vertex_buffer = .{ .id = 3 },
        .index_buffer = .{ .id = 4 },
        .index_count = 6,
    };
    instanced.instances.items.len = 3;
    const it = makeHighlightDrawItem(&instanced, .{}, 1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 6), it.index_count);
}

test "highlight mask viewport maps the primary rect onto the half-res target" {
    // Fullscreen 1280x720: the full 640x360 mask target (fullscreen
    // mapping — identical pixels to the unmapped mask).
    const full = highlightMaskViewport(.{ .x = 0, .y = 0, .width = 1280, .height = 720 }, 1280, 720);
    try std.testing.expectEqual(@as(i32, 0), full.x);
    try std.testing.expectEqual(@as(i32, 0), full.y);
    try std.testing.expectEqual(@as(i32, 640), full.width);
    try std.testing.expectEqual(@as(i32, 360), full.height);

    // Right-half PIP (640,0,640x720): the right half of the mask target.
    const pip = highlightMaskViewport(.{ .x = 640, .y = 0, .width = 640, .height = 720 }, 1280, 720);
    try std.testing.expectEqual(@as(i32, 320), pip.x);
    try std.testing.expectEqual(@as(i32, 0), pip.y);
    try std.testing.expectEqual(@as(i32, 320), pip.width);
    try std.testing.expectEqual(@as(i32, 360), pip.height);

    // Quarter viewport scales both axes.
    const q = highlightMaskViewport(.{ .x = 100, .y = 50, .width = 400, .height = 300 }, 1280, 720);
    try std.testing.expectEqual(@as(i32, 50), q.x);
    try std.testing.expectEqual(@as(i32, 25), q.y);
    try std.testing.expectEqual(@as(i32, 200), q.width);
    try std.testing.expectEqual(@as(i32, 150), q.height);

    // Degenerate sizes never produce a zero viewport (1px floor) and
    // degenerate bases fall back to the full target rect.
    const tiny = highlightMaskViewport(.{ .x = 0, .y = 0, .width = 1, .height = 1 }, 1280, 720);
    try std.testing.expectEqual(@as(i32, 1), tiny.width);
    try std.testing.expectEqual(@as(i32, 1), tiny.height);
    const degenerate = highlightMaskViewport(.{ .x = 0, .y = 0, .width = 1280, .height = 720 }, 0, 720);
    try std.testing.expectEqual(@as(i32, 1), degenerate.width);
    try std.testing.expectEqual(@as(i32, 360), degenerate.height);
}

test "highlight mask color folds intensity, frame sigma takes the max" {
    const item = HighlightDrawItem{ .color = .{ 0.5, 0.25, 1.0, 0.5 }, .intensity = 0.8 };
    try std.testing.expectEqual([4]f32{ 0.4, 0.2, 0.8, 0.5 }, highlightMaskColor(item));

    const items = [_]HighlightDrawItem{
        .{ .blur = 2.0 },
        .{ .blur = 6.0 },
        .{ .blur = 4.0 },
    };
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), highlightFrameSigma(&items), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), highlightFrameSigma(&.{}), 1e-6);
}

test "configureHighlightMaskDesc sets opaque silhouette-fill state" {
    var desc = std.mem.zeroes(sg.PipelineDesc);
    configureHighlightMaskDesc(&desc);
    // No culling (silhouette-exact for open geometry), no depth (additive,
    // never depth-tested), opaque overwrite (last overlap wins the mask).
    try std.testing.expect(desc.cull_mode == .NONE);
    try std.testing.expect(desc.depth.pixel_format == .NONE);
    try std.testing.expect(!desc.depth.write_enabled);
    try std.testing.expect(!desc.colors[0].blend.enabled);
    try std.testing.expectEqual(@sizeOf(Vertex), desc.layout.buffers[0].stride);
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_position].format == .FLOAT3);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "position"))),
        desc.layout.attrs[outline_shd.ATTR_outline_position].offset,
    );
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_normal].format == .FLOAT3);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "normal"))),
        desc.layout.attrs[outline_shd.ATTR_outline_normal].offset,
    );
}
