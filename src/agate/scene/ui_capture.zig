const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const scene_snapshot = @import("snapshot.zig");
const SceneFrameSnapshot = scene_snapshot.SceneFrameSnapshot;
const scene_frame_draws = @import("frame_draws.zig");
const FrameDrawSlot = scene_frame_draws.FrameDrawSlot;

/// Shared UI stage core behind `stageUiPacket` and `BuildClaim.stageUi`:
/// the exact historical stage body targeted at `slot`.
pub fn stageUiPacketInto(scene: anytype, slot: usize) void {
    // Deliberately NO gpu_thread assert: game side, sg-free by design.
    const back = &scene.draws.slots[slot];
    const canvas = if (scene.ui_canvas) |*c| c else {
        back.ui_vertices.clearRetainingCapacity();
        back.ui_indices.clearRetainingCapacity();
        back.ui_packet = .{ .valid = true, .canvas_present = false };
        // Release: the slot packet bytes + header above are staged before
        // the generation the latch acquire-reads.
        _ = scene.ui_packet_seq.fetchAdd(1, .release);
        return;
    };
    back.ui_vertices.clearRetainingCapacity();
    back.ui_indices.clearRetainingCapacity();
    back.ui_vertices.appendSlice(scene.allocator, canvas.vertices.items) catch {
        back.ui_vertices.clearRetainingCapacity();
        back.ui_indices.clearRetainingCapacity();
        back.ui_packet = .{ .valid = false };
        // Release: staged-before-bump (see above).
        _ = scene.ui_packet_seq.fetchAdd(1, .release);
        return;
    };
    back.ui_indices.appendSlice(scene.allocator, canvas.indices.items) catch {
        back.ui_vertices.clearRetainingCapacity();
        back.ui_indices.clearRetainingCapacity();
        back.ui_packet = .{ .valid = false };
        // Release: staged-before-bump (see above).
        _ = scene.ui_packet_seq.fetchAdd(1, .release);
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
    // Release: the slot packet bytes + header above are staged before
    // the generation the latch acquire-reads.
    _ = scene.ui_packet_seq.fetchAdd(1, .release);
}

/// P6 UI handoff: captures CPU geometry + draw parameters out of the
/// live canvas into the render-owned frame and uploads at this
/// prepare/context boundary (phase mutex held, context thread). Runs
/// after the UI build (sandbox builds under the same mutex before
/// prepareFrame) and after the frame snapshot above is final, so the
/// captured screen size matches the selected scene snapshot (same
/// fallback as render: snapshot dims when positive, live sapp dims
/// otherwise — a zero placeholder snapshot never hides a valid GPU
/// frame). Takes the STAGED slot snapshot from the caller (the slot
/// prepare is consuming) — never the live `frame_snapshot` — so the
/// latched dims track the prepared frame's generation. Snapshots canvas
/// presence separately from nonempty/gpu_ready
/// so the historical phantom+1 counters survive empty frames. Missing
/// canvas or camera-less frames fail close to coherent-empty — never a
/// stale prior overlay, never an unsafe upload/draw.
///
/// Takes the claimed back slot prepare holds (wave 31) — never
/// `backSlot()`: the held claim marks the slot WRITING, which the
/// unlocked helper would skip, returning a different slot than the one
/// prepare latches.
pub fn captureUiFrame(scene: anytype, snap: *const SceneFrameSnapshot, back: *FrameDrawSlot) void {
    // Slot-owned UI packet first (game-side staging, slice 2 b): when
    // the game staged a fresh packet, the latch consumes it instead of
    // reading the live canvas lists. The seq is stamped consumed on ALL
    // fresh-packet paths — including the fallbacks — so a stale packet
    // is never re-latched by a later prepare. Acquire: pairs with the
    // stage's release-bump, so the slot packet bytes are visible here.
    const staged_seq = scene.ui_packet_seq.load(.acquire);
    if (staged_seq != scene.last_latched_ui_seq.load(.monotonic)) {
        // Context-side stamp only (the producer never touches this word).
        scene.last_latched_ui_seq.store(staged_seq, .monotonic);
        if (back.ui_packet.valid) {
            if (back.ui_packet.canvas_present) {
                if (scene.ui_canvas) |*canvas| {
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
                    return;
                }
                // Staged presence but the canvas vanished before the
                // latch: fall through to the legacy path, whose
                // missing-canvas branch fail-closes identically.
            } else {
                // Staged absence of canvas: same clear as legacy.
                scene.ui_frame.clearEmpty();
                scene.ui_frame.canvas_present = false;
                return;
            }
        }
        // Fresh seq but invalid packet (stage OOM, or a later
        // buildPreparedFrame reset wiped the slot — stage UI after the
        // build when both are used): the legacy canvas path below still
        // holds the content, so degrade to it instead of clearing.
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
    scene.ui_frame.capture(
        scene.allocator,
        canvas,
        @floatFromInt(w),
        @floatFromInt(h),
    );
    _ = scene.ui_frame.upload(canvas, .{
        .allocator = scene.allocator,
        .retire_queue = &scene.gpu_retire,
    });
}
