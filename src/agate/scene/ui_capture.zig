const sokol = @import("sokol");
const sapp = sokol.app;
const scene_snapshot = @import("snapshot.zig");
const SceneFrameSnapshot = scene_snapshot.SceneFrameSnapshot;
const scene_frame_draws = @import("frame_draws.zig");
const FrameDrawSlot = scene_frame_draws.FrameDrawSlot;

/// Shared UI stage core behind `BuildClaim.stageUi`: stages the live canvas
/// CPU geometry into the claimed `slot` (`ui_vertices`/`ui_indices` +
/// `ui_packet` header) for the staged prepare latch to consume. Game side,
/// sg-free: CPU list copies into retained slot capacity + plain handle
/// stamps. Call AFTER `build()` (the build resets the slot). Newest wins;
/// OOM fail-closes to coherent-empty with `canvas_present` preserved (the
/// canvas existed at stage time).
pub fn stageUiPacketInto(scene: anytype, slot: usize) void {
    // Deliberately NO gpu_thread assert: game side, sg-free by design.
    const back = &scene.draws.slots[slot];
    const canvas = if (scene.ui_canvas) |*c| c else {
        back.ui_vertices.clearRetainingCapacity();
        back.ui_indices.clearRetainingCapacity();
        back.ui_packet = .{ .valid = true, .canvas_present = false };
        return;
    };
    back.ui_vertices.clearRetainingCapacity();
    back.ui_indices.clearRetainingCapacity();
    back.ui_vertices.appendSlice(scene.allocator, canvas.vertices.items) catch {
        back.ui_vertices.clearRetainingCapacity();
        back.ui_indices.clearRetainingCapacity();
        back.ui_packet = .{ .valid = false, .canvas_present = true };
        return;
    };
    back.ui_indices.appendSlice(scene.allocator, canvas.indices.items) catch {
        back.ui_vertices.clearRetainingCapacity();
        back.ui_indices.clearRetainingCapacity();
        back.ui_packet = .{ .valid = false, .canvas_present = true };
        return;
    };
    back.ui_packet = .{
        .valid = true,
        .canvas_present = true,
        .handles = .{
            .pipeline = canvas.pipeline,
            .font_view = canvas.activeFontView(),
            .font_sampler = canvas.activeFontSampler(),
            .vertex_buffer = canvas.vertex_buffer,
            .index_buffer = canvas.index_buffer,
        },
        .upload_seq = canvas.ui_upload_seq,
    };
}

/// Staged UI latch: consumes the claimed slot's staged packet into the
/// render-owned frame. Never reads live canvas geometry — a missing/invalid
/// packet or a missing canvas fail-closes to coherent-empty. Canvas GPU
/// metadata and buffers stay context-owned and are used only through the
/// staged `capturePacket`/`upload(canvas)` identity (first-wins lifecycle).
/// Screen size comes from the staged snapshot (sapp dims fallback when 0).
/// No-camera frames fail closed to coherent-empty with presence preserved.
pub fn captureUiFrame(scene: anytype, snap: *const SceneFrameSnapshot, back: *FrameDrawSlot) void {
    if (!back.ui_packet.valid) {
        scene.ui_frame.clearEmpty();
        scene.ui_frame.canvas_present = back.ui_packet.canvas_present;
        return;
    }
    if (!back.ui_packet.canvas_present) {
        scene.ui_frame.clearEmpty();
        scene.ui_frame.canvas_present = false;
        return;
    }
    const canvas = if (scene.ui_canvas) |*c| c else {
        scene.ui_frame.clearEmpty();
        scene.ui_frame.canvas_present = false;
        return;
    };
    scene.ui_frame.canvas_present = true;
    if (!snap.has_camera) {
        scene.ui_frame.clearEmpty();
        scene.ui_frame.canvas_present = true;
        return;
    }
    const w = if (snap.screen_w > 0) snap.screen_w else sapp.width();
    const h = if (snap.screen_h > 0) snap.screen_h else sapp.height();
    scene.ui_frame.capturePacket(
        scene.allocator,
        canvas,
        back.ui_vertices.items,
        back.ui_indices.items,
        @floatFromInt(w),
        @floatFromInt(h),
        back.ui_packet.handles,
    );
    scene.ui_packet_latched +%= 1;
    _ = scene.ui_frame.upload(canvas, .{
        .allocator = scene.allocator,
        .retire_queue = &scene.gpu_retire,
    });
}
