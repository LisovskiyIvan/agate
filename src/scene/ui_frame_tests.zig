const std = @import("std");
const math = @import("math");
const Color4 = math.Color4;
const ui = @import("../ui.zig");
const UICanvas = ui.UICanvas;
const UIVertex = ui.UIVertex;
const ui_frame = @import("ui_frame.zig");
const UiFrame = ui_frame.UiFrame;
const UploadResult = ui_frame.UploadResult;
const UiPacketHandles = ui_frame.UiPacketHandles;
const upload_meter = @import("../gpu_upload_meter.zig");
const TestTexture = @import("../texture.zig").Texture;
const ttf_mod = @import("../ttf.zig");

/// Canvas without GPU state for capture tests: draw calls only push quads
/// into the CPU-side lists (same shape as ui.zig's own test helper, which
/// is file-private).
fn testCanvas(alloc: std.mem.Allocator) UICanvas {
    return .{ .allocator = alloc, .font_texture = std.mem.zeroes(TestTexture) };
}

fn freeTestCanvas(canvas: *UICanvas) void {
    canvas.vertices.deinit(canvas.allocator);
    canvas.indices.deinit(canvas.allocator);
}

test "P6: capture copies geometry and survives source mutation/deallocation" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    // Fake borrowed handles: capture must copy the VALUES, never deref.
    canvas.pipeline = .{ .id = 7 };
    canvas.font_texture.view = .{ .id = 11 };
    canvas.font_texture.sampler = .{ .id = 13 };
    canvas.vertex_buffer = .{ .id = 21 };
    canvas.index_buffer = .{ .id = 23 };

    canvas.drawRect(10, 20, 30, 40, Color4.white);
    canvas.drawText("hi", 0, 0, 16.0, Color4.white);
    canvas.drawLine(0, 0, 5, 5, 2.0, Color4.white);
    const want_verts = canvas.vertices.items.len;
    const want_idx = canvas.indices.items.len;
    try t.expect(want_verts > 0 and want_idx > 0);

    var frame = UiFrame{};
    defer frame.deinit(t.allocator);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(frame.has_capture);
    try t.expect(frame.needs_upload);
    try t.expect(!frame.gpu_ready);
    try t.expectEqual(want_verts, frame.vertices.items.len);
    try t.expectEqual(want_idx, frame.indices.items.len);
    try t.expectEqualSlices(UIVertex, canvas.vertices.items, frame.vertices.items);
    try t.expectEqualSlices(u16, canvas.indices.items, frame.indices.items);
    // Packet mirrors the source: clamped counts, dims, borrowed handle IDs.
    try t.expectEqual(UICanvas.clampedVertCount(want_verts), frame.vert_count);
    try t.expectEqual(want_idx, frame.index_count);
    try t.expectEqual(@as(f32, 800.0), frame.screen_w);
    try t.expectEqual(@as(f32, 600.0), frame.screen_h);
    try t.expectEqual(@as(u32, 7), frame.pipeline.id);
    try t.expectEqual(@as(u32, 11), frame.font_view.id);
    try t.expectEqual(@as(u32, 13), frame.font_sampler.id);
    try t.expectEqual(@as(u32, 21), frame.vertex_buffer.id);
    try t.expectEqual(@as(u32, 23), frame.index_buffer.id);
    // Text quads carry SDF mode + glyph UVs, solid quads mode 0.
    try t.expectEqualSlices(f32, &.{ 0.0, 0.0, 0.0, 0.0 }, &frame.vertices.items[0].mode_params);
    try t.expectEqual(@as(f32, 1.0), frame.vertices.items[4].mode_params[0]);

    // Mutate the source past recognition: frame owns its copies.
    canvas.begin();
    canvas.drawRect(1, 2, 3, 4, Color4.white);
    try t.expectEqual(want_verts, frame.vertices.items.len);
    try t.expectEqual(want_idx, frame.indices.items.len);
    try t.expectEqual(@as(f32, 10.0), frame.vertices.items[0].position[0]);

    // Deallocate the source entirely: frame stays intact.
    freeTestCanvas(&canvas);
    try t.expectEqual(want_verts, frame.vertices.items.len);
    try t.expectEqual(@as(f32, 10.0), frame.vertices.items[0].position[0]);
    try t.expectEqual(@as(u32, 7), frame.pipeline.id);
}

test "P6: next capture publishes the newest frame" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    var frame = UiFrame{};
    defer frame.deinit(t.allocator);

    canvas.drawRect(0, 0, 10, 10, Color4.white);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    const first_verts = frame.vertices.items.len;
    try t.expect(first_verts > 0);

    canvas.begin();
    canvas.drawRect(0, 0, 10, 10, Color4.white);
    canvas.drawRect(20, 20, 10, 10, Color4.white);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expectEqual(first_verts * 2, frame.vertices.items.len);
    try t.expectEqual(@as(f32, 20.0), frame.vertices.items[4].position[0]);
}

test "P6: clear/removal/zero-dims/empty paths yield coherent empty frames" {
    const t = std.testing;
    _ = upload_meter.takeAndReset();
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    var frame = UiFrame{};
    defer frame.deinit(t.allocator);

    // Empty canvas → empty frame (matches legacy early-out).
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(!frame.has_capture);
    try t.expect(!frame.needs_upload);

    canvas.drawRect(0, 0, 10, 10, Color4.white);
    // Zero dims → empty frame, no upload/draw.
    frame.capture(t.allocator, &canvas, 0.0, 600.0);
    try t.expect(!frame.has_capture);
    frame.capture(t.allocator, &canvas, 800.0, 0.0);
    try t.expect(!frame.has_capture);
    frame.capture(t.allocator, &canvas, -1.0, 600.0);
    try t.expect(!frame.has_capture);

    // Content then removal (begin with no draws) → empty, no stale packet.
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(frame.has_capture);
    canvas.begin();
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(!frame.has_capture);
    try t.expectEqual(@as(usize, 0), frame.vertices.items.len);
    try t.expectEqual(@as(usize, 0), frame.indices.items.len);
    try t.expectEqual(@as(usize, 0), frame.vert_count);

    // Explicit clear + empty upload idempotence.
    frame.clearEmpty();
    try t.expectEqual(UploadResult.nothing_to_do, frame.upload(&canvas, .{ .allocator = t.allocator }));
    // Draw of an empty/never-uploaded frame: safe no-op, 0 meter bytes.
    frame.drawPrepared();
    try t.expectEqual(@as(u64, 0), upload_meter.peek());
    frame.drawPrepared();
    frame.drawPrepared();
    try t.expectEqual(@as(u64, 0), upload_meter.peek());
}

test "P6: retained capacity is reused across captures" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    var frame = UiFrame{};
    defer frame.deinit(t.allocator);

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        canvas.drawRect(0, 0, 10, 10, Color4.white);
    }
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    const big_cap_v = frame.vertices.capacity;
    const big_cap_i = frame.indices.capacity;
    try t.expect(big_cap_v >= 400 and big_cap_i >= 600);

    canvas.begin();
    canvas.drawRect(0, 0, 10, 10, Color4.white);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expectEqual(@as(usize, 4), frame.vertices.items.len);
    try t.expectEqual(@as(usize, 6), frame.indices.items.len);
    // No shrink: capacity retained for the next big frame.
    try t.expectEqual(big_cap_v, frame.vertices.capacity);
    try t.expectEqual(big_cap_i, frame.indices.capacity);
}

test "P6: allocator failure on either reserve gives coherent empty, then recovers" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    canvas.drawRect(0, 0, 10, 10, Color4.white);

    var frame = UiFrame{};
    defer frame.deinit(t.allocator);

    // Fail the vertices reserve (first allocation of the capture).
    var failing_v = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    frame.capture(failing_v.allocator(), &canvas, 800.0, 600.0);
    try t.expect(!frame.has_capture);
    try t.expectEqual(@as(usize, 0), frame.vertices.items.len);
    try t.expectEqual(@as(usize, 0), frame.indices.items.len);

    // Fail the indices reserve (second allocation): no partial vertex half.
    var failing_i = t.FailingAllocator.init(t.allocator, .{ .fail_index = 1 });
    frame.capture(failing_i.allocator(), &canvas, 800.0, 600.0);
    try t.expect(!frame.has_capture);
    try t.expectEqual(@as(usize, 0), frame.vertices.items.len);
    try t.expectEqual(@as(usize, 0), frame.indices.items.len);
    try t.expect(!frame.needs_upload);

    // Recovery: a funded capture publishes the full frame.
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(frame.has_capture);
    try t.expectEqual(canvas.vertices.items.len, frame.vertices.items.len);
    try t.expectEqual(canvas.indices.items.len, frame.indices.items.len);
}

test "P6: byte math stays in usize past 1366 vertices, u16 clamp holds" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    // 500 rects = 2000 verts: past the 1366-vertex u16-byte overflow tripwire.
    var i: usize = 0;
    while (i < 500) : (i += 1) {
        canvas.drawRect(0, 0, 10, 10, Color4.white);
    }
    var frame = UiFrame{};
    defer frame.deinit(t.allocator);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expectEqual(@as(usize, 2000), frame.vert_count);
    try t.expectEqual(@as(usize, 3000), frame.index_count);
    const want = @as(usize, 2000) * @sizeOf(UIVertex) + @as(usize, 3000) * @sizeOf(u16);
    try t.expectEqual(want, UICanvas.batchUploadBytes(frame.vert_count, frame.index_count));
    // The u16 clamp itself (unreachable via the silently-capped public draw
    // API, but the packet invariant must hold for any length).
    try t.expectEqual(@as(usize, std.math.maxInt(u16)), UICanvas.clampedVertCount(70000));
    try t.expectEqual(@as(usize, 2000), UICanvas.clampedVertCount(2000));
    // Growth formula mirrors the legacy path.
    try t.expectEqual(@as(usize, 65536), UICanvas.grownCapacity(32768, 40000));
    try t.expectEqual(@as(usize, 65536), UICanvas.grownCapacity(32768, 65536));
}

test "P6: staged packet meeting an open window defers; unstaged recapture fail-closes" {
    const t = std.testing;
    _ = upload_meter.takeAndReset();
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    var frame = UiFrame{};
    defer frame.deinit(t.allocator);

    // Stage A with a closed window (headless: no context, stays staged).
    canvas.drawRect(0, 0, 10, 10, Color4.white);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(frame.has_capture and frame.needs_upload);
    try t.expectEqual(UploadResult.no_context, frame.upload(&canvas, .{ .allocator = t.allocator }));
    try t.expect(frame.needs_upload);

    // Another writer uploads (window opens) with NO recapture in between:
    // the staged packet is still ours-or-unknown, so the upload defers
    // rather than resubmitting — retry after a commit.
    canvas.ui_upload_armed = true;
    try t.expectEqual(UploadResult.deferred_open_window, frame.upload(&canvas, .{ .allocator = t.allocator }));
    try t.expect(frame.needs_upload);
    try t.expectEqual(@as(u64, 0), upload_meter.peek());

    // But a recapture in that open window cannot prove its bytes are on the
    // GPU (staged A was never committed, and the writer may have
    // overwritten them): fail close to coherent-empty, no partial half.
    canvas.begin();
    canvas.drawRect(50, 60, 70, 80, Color4.white);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(!frame.has_capture);
    try t.expect(!frame.needs_upload);
    try t.expect(!frame.gpu_ready);
    try t.expectEqual(@as(usize, 0), frame.vertices.items.len);
    try t.expectEqual(@as(usize, 0), frame.vert_count);
    try t.expectEqual(UploadResult.nothing_to_do, frame.upload(&canvas, .{ .allocator = t.allocator }));
    try t.expectEqual(@as(u64, 0), upload_meter.peek());

    // Commit boundary (headless override clears the arm): the next capture
    // publishes the newest canvas fresh, with no auto-upload promise.
    canvas.ui_upload_armed = false;
    frame.capture(t.allocator, &canvas, 1024.0, 768.0);
    try t.expect(frame.has_capture and frame.needs_upload);
    try t.expectEqual(@as(f32, 50.0), frame.vertices.items[0].position[0]);
    try t.expectEqual(@as(f32, 1024.0), frame.screen_w);
    try t.expectEqual(@as(u64, 0), upload_meter.peek());
}

test "P6: another writer invalidates the committed frame (fail-close, then recover)" {
    const t = std.testing;
    _ = upload_meter.takeAndReset();
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    var frame = UiFrame{};
    defer frame.deinit(t.allocator);

    // Commit frame A through the frame path (simulated headless success:
    // stage, then assign the committed identity as a live upload would —
    // sequence + installed IDs, window armed).
    canvas.pipeline = .{ .id = 7 };
    canvas.vertex_buffer = .{ .id = 21 };
    canvas.index_buffer = .{ .id = 23 };
    canvas.drawRect(0, 0, 10, 10, Color4.white);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(frame.has_capture and frame.needs_upload);
    canvas.markUiUploaded();
    frame.upload_seq = canvas.ui_upload_seq;
    frame.gpu_ready = true;
    frame.needs_upload = false;
    try t.expect(canvas.isUploadOpen());

    // Exact same upload, double capture: whole A unchanged.
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(frame.has_capture and frame.gpu_ready);
    try t.expectEqual(@as(usize, 4), frame.vertices.items.len);
    try t.expectEqual(@as(f32, 0.0), frame.vertices.items[0].position[0]);
    try t.expectEqual(@as(usize, 4), frame.vert_count);

    // Empty recapture under the same valid upload: first-wins preserves A
    // (the window check precedes the empty check).
    canvas.begin();
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(frame.has_capture and frame.gpu_ready);
    try t.expectEqual(@as(usize, 4), frame.vertices.items.len);
    try t.expectEqual(@as(usize, 4), frame.vert_count);

    // Another writer overwrites the same buffers (IDs equal, sequence
    // bumped): identity mismatch → fail close to coherent-empty, never a
    // draw of the writer's bytes under A's counts.
    canvas.drawRect(50, 60, 70, 80, Color4.white);
    canvas.markUiUploaded();
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(!frame.has_capture);
    try t.expect(!frame.gpu_ready);
    try t.expect(!frame.needs_upload);
    try t.expectEqual(@as(u64, 0), frame.upload_seq);
    try t.expectEqual(@as(usize, 0), frame.vertices.items.len);
    try t.expectEqual(@as(usize, 0), frame.vert_count);
    try t.expectEqual(@as(u64, 0), upload_meter.peek());

    // Recommit, then a growth-swap by another writer (buffer IDs replaced):
    // same fail-close — destroyed handles are never drawn.
    canvas.ui_upload_armed = false; // simulated commit boundary
    canvas.begin();
    canvas.drawRect(0, 0, 10, 10, Color4.white);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(frame.has_capture);
    canvas.markUiUploaded();
    frame.upload_seq = canvas.ui_upload_seq;
    frame.gpu_ready = true;
    frame.needs_upload = false;
    canvas.vertex_buffer = .{ .id = 31 };
    canvas.index_buffer = .{ .id = 33 };
    canvas.markUiUploaded();
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(!frame.has_capture);
    try t.expectEqual(@as(usize, 0), frame.vertices.items.len);

    // Later capture after a real commit recovers with the newest canvas.
    canvas.ui_upload_armed = false;
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(frame.has_capture);
    try t.expectEqual(@as(u32, 31), frame.vertex_buffer.id);
    try t.expectEqual(@as(u32, 33), frame.index_buffer.id);
    try t.expectEqual(@as(u64, 0), upload_meter.peek());
}

test "P6: idempotent upload never resubmits without a new capture" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    var frame = UiFrame{};
    defer frame.deinit(t.allocator);

    // Empty frame: nothing to do, no sg touch.
    try t.expectEqual(UploadResult.nothing_to_do, frame.upload(&canvas, .{ .allocator = t.allocator }));

    canvas.drawRect(0, 0, 10, 10, Color4.white);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    // First call stages (headless: no_context); simulate the live outcome
    // by clearing the pending flag as a successful upload would, then prove
    // the repeat call is a no-op.
    _ = frame.upload(&canvas, .{ .allocator = t.allocator });
    frame.needs_upload = false;
    frame.gpu_ready = true;
    try t.expectEqual(UploadResult.nothing_to_do, frame.upload(&canvas, .{ .allocator = t.allocator }));
}

test "P6: legacy canvas render is a safe headless no-op and honors the window" {
    const t = std.testing;
    _ = upload_meter.takeAndReset();
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    canvas.drawRect(0, 0, 10, 10, Color4.white);

    // No sokol context: safe no-op, 0 bytes (previously an assert trap).
    canvas.render(800.0, 600.0);
    try t.expectEqual(@as(u64, 0), upload_meter.peek());
    // Empty/dims guards unchanged.
    canvas.render(0.0, 600.0);
    canvas.begin();
    canvas.render(800.0, 600.0);
    try t.expectEqual(@as(u64, 0), upload_meter.peek());

    // Armed window (a frame upload spent this sokol frame's update): legacy
    // render fails close — no upload, no draw against foreign GPU data.
    // Headless the armed flag alone decides (no commits exist).
    canvas.drawRect(0, 0, 10, 10, Color4.white);
    canvas.ui_upload_armed = true;
    canvas.render(800.0, 600.0);
    try t.expectEqual(@as(u64, 0), upload_meter.peek());
    // Never armed (fresh canvas): the window is closed by construction.
    canvas.ui_upload_armed = false;
    try t.expect(!canvas.isUploadOpen());
}

test "slice2b: capturePacket matches capture and ignores later canvas mutation" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    canvas.pipeline = .{ .id = 7 };
    canvas.font_texture.view = .{ .id = 11 };
    canvas.font_texture.sampler = .{ .id = 13 };
    canvas.vertex_buffer = .{ .id = 21 };
    canvas.index_buffer = .{ .id = 23 };

    canvas.drawRect(10, 20, 30, 40, Color4.white);
    canvas.drawText("pkt", 0, 0, 16.0, Color4.white);

    // Reference: legacy capture straight from the live lists.
    var legacy = UiFrame{};
    defer legacy.deinit(t.allocator);
    legacy.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(legacy.has_capture);

    // Packet capture from staged copies of the same bytes: identical frame.
    const staged_v = try t.allocator.dupe(UIVertex, canvas.vertices.items);
    defer t.allocator.free(staged_v);
    const staged_i = try t.allocator.dupe(u16, canvas.indices.items);
    defer t.allocator.free(staged_i);
    const staged_handles = UiPacketHandles{
        .pipeline = canvas.pipeline,
        .font_view = canvas.activeFontView(),
        .font_sampler = canvas.activeFontSampler(),
        .vertex_buffer = canvas.vertex_buffer,
        .index_buffer = canvas.index_buffer,
    };
    var packet = UiFrame{};
    defer packet.deinit(t.allocator);
    packet.capturePacket(t.allocator, &canvas, staged_v, staged_i, 800.0, 600.0, staged_handles);
    try t.expect(packet.has_capture);
    try t.expectEqualSlices(UIVertex, legacy.vertices.items, packet.vertices.items);
    try t.expectEqualSlices(u16, legacy.indices.items, packet.indices.items);
    try t.expectEqual(legacy.vert_count, packet.vert_count);
    try t.expectEqual(legacy.index_count, packet.index_count);
    try t.expectEqual(legacy.screen_w, packet.screen_w);
    try t.expectEqual(legacy.pipeline.id, packet.pipeline.id);
    try t.expectEqual(legacy.font_view.id, packet.font_view.id);
    try t.expectEqual(legacy.vertex_buffer.id, packet.vertex_buffer.id);

    // Mutate the canvas past recognition: a packet capture from the STAGED
    // bytes still yields the staged frame (staged wins, nothing live leaks).
    canvas.begin();
    canvas.drawRect(1, 2, 3, 4, Color4.white);
    var late = UiFrame{};
    defer late.deinit(t.allocator);
    late.capturePacket(t.allocator, &canvas, staged_v, staged_i, 800.0, 600.0, staged_handles);
    try t.expect(late.has_capture);
    try t.expectEqualSlices(UIVertex, legacy.vertices.items, late.vertices.items);
    try t.expectEqualSlices(u16, legacy.indices.items, late.indices.items);

    // Guards mirror capture: empty slices and bad dims fail closed.
    var empty = UiFrame{};
    defer empty.deinit(t.allocator);
    empty.capturePacket(t.allocator, &canvas, &.{}, &.{}, 800.0, 600.0, staged_handles);
    try t.expect(!empty.has_capture);
    empty.capturePacket(t.allocator, &canvas, staged_v, staged_i, 0.0, 600.0, staged_handles);
    try t.expect(!empty.has_capture);
}

test "slice2b: staged handles win over post-stage canvas handle mutation" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    canvas.pipeline = .{ .id = 7 };
    canvas.font_texture.view = .{ .id = 11 };
    canvas.font_texture.sampler = .{ .id = 13 };
    canvas.vertex_buffer = .{ .id = 21 };
    canvas.index_buffer = .{ .id = 23 };
    canvas.drawRect(10, 20, 30, 40, Color4.white);

    // Freeze the stage-time handles (what Scene.stageUiPacket stamps).
    const staged_handles = UiPacketHandles{
        .pipeline = canvas.pipeline,
        .font_view = canvas.activeFontView(),
        .font_sampler = canvas.activeFontSampler(),
        .vertex_buffer = canvas.vertex_buffer,
        .index_buffer = canvas.index_buffer,
    };
    const staged_v = try t.allocator.dupe(UIVertex, canvas.vertices.items);
    defer t.allocator.free(staged_v);
    const staged_i = try t.allocator.dupe(u16, canvas.indices.items);
    defer t.allocator.free(staged_i);

    // Mutate every live handle past recognition between stage and latch.
    canvas.pipeline = .{ .id = 70 };
    canvas.font_texture.view = .{ .id = 110 };
    canvas.font_texture.sampler = .{ .id = 130 };
    canvas.vertex_buffer = .{ .id = 210 };
    canvas.index_buffer = .{ .id = 230 };

    // The latch consumes the STAGED handles: no live handle leaks in.
    var frame = UiFrame{};
    defer frame.deinit(t.allocator);
    frame.capturePacket(t.allocator, &canvas, staged_v, staged_i, 800.0, 600.0, staged_handles);
    try t.expect(frame.has_capture);
    try t.expectEqual(@as(u32, 7), frame.pipeline.id);
    try t.expectEqual(@as(u32, 11), frame.font_view.id);
    try t.expectEqual(@as(u32, 13), frame.font_sampler.id);
    try t.expectEqual(@as(u32, 21), frame.vertex_buffer.id);
    try t.expectEqual(@as(u32, 23), frame.index_buffer.id);
}

test "P6: capture binds the TTF atlas view when a font is set" {
    const t = std.testing;
    const file = try ttf_mod.buildFixture(t.allocator, .{});
    defer t.allocator.free(file);
    var font = try ttf_mod.TtfFont.init(t.allocator, file, 20.0, &.{'A'});
    defer font.deinit();

    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    canvas.font_texture.view = .{ .id = 11 };
    canvas.font_texture.sampler = .{ .id = 13 };
    canvas.drawText("A", 0, 0, 16.0, Color4.white);

    // No font: legacy atlas view.
    var legacy = UiFrame{};
    defer legacy.deinit(t.allocator);
    legacy.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(legacy.has_capture);
    try t.expectEqual(@as(u32, 11), legacy.font_view.id);

    // Font set with an uploaded atlas: capture takes the TTF view.
    canvas.ttf_font = &font;
    var uploaded = std.mem.zeroes(TestTexture);
    uploaded.view = .{ .id = 42 };
    uploaded.sampler = .{ .id = 43 };
    canvas.ttf_texture = uploaded;
    var frame = UiFrame{};
    defer frame.deinit(t.allocator);
    frame.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(frame.has_capture);
    try t.expectEqual(@as(u32, 42), frame.font_view.id);
    try t.expectEqual(@as(u32, 43), frame.font_sampler.id);

    // Font set but headless (no upload): falls back to the legacy view,
    // vertices still carry the TTF coverage mode.
    canvas.ttf_texture = null;
    canvas.begin();
    canvas.drawText("A", 0, 0, 16.0, Color4.white);
    try t.expectApproxEqAbs(@as(f32, 3.0), canvas.vertices.items[0].mode_params[0], 1e-6);
    var headless = UiFrame{};
    defer headless.deinit(t.allocator);
    headless.capture(t.allocator, &canvas, 800.0, 600.0);
    try t.expect(headless.has_capture);
    try t.expectEqual(@as(u32, 11), headless.font_view.id);
}
