const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const velocity_pass = @import("velocity_pass.zig");
const VelocityPass = velocity_pass.VelocityPass;
const velocityNeeded = velocity_pass.velocityNeeded;
const cutSuppressesVelocity = velocity_pass.cutSuppressesVelocity;
const usePresentedPrev = velocity_pass.usePresentedPrev;
const velocityDepthState = velocity_pass.velocityDepthState;
const velocityDepthPassAction = velocity_pass.velocityDepthPassAction;
const velocityDrawDecision = velocity_pass.velocityDrawDecision;
const scene_render_queue = @import("../scene/render_queue.zig");
const RenderQueues = scene_render_queue.RenderQueues;
const stats_mod = @import("../scene/stats.zig");
const SceneStats = stats_mod.SceneStats;

test "velocityNeeded mirrors the composite MSAA gate" {
    // TAA runs only at 1x (taa_on pre-gated in frame_render): always needs it.
    try std.testing.expect(velocityNeeded(true, false, 1));
    try std.testing.expect(velocityNeeded(true, true, 1));
    // Nothing enabled: no target.
    try std.testing.expect(!velocityNeeded(false, false, 1));
    // Blur at 1x: needed.
    try std.testing.expect(velocityNeeded(false, true, 1));
    // Blur under MSAA: explicitly suppressed even with the depth prepass —
    // the blur falls back to camera depth reprojection, so no
    // sample-mismatched velocity target may feed it false vectors.
    try std.testing.expect(!velocityNeeded(false, true, 4));
    try std.testing.expect(!velocityNeeded(false, true, 2));
}

test "cutSuppressesVelocity fires exactly on reset/cut frames" {
    // Steady state: never suppress (no per-frame reset).
    try std.testing.expect(!cutSuppressesVelocity(false, false, false));
    try std.testing.expect(!cutSuppressesVelocity(false, true, false));
    // Any TAA reset suppresses, with or without blur.
    try std.testing.expect(cutSuppressesVelocity(true, false, false));
    try std.testing.expect(cutSuppressesVelocity(true, true, true));
    // Blur-only mode still suppresses on a camera cut.
    try std.testing.expect(cutSuppressesVelocity(false, true, true));
    // Blur disabled + no TAA reset: a cut flag alone changes nothing.
    try std.testing.expect(!cutSuppressesVelocity(false, false, true));
}

test "usePresentedPrev matches only the last rendered generation" {
    const never = std.math.maxInt(u64);
    // Never staged never matches — not even a zero last-rendered.
    try std.testing.expect(!usePresentedPrev(never, 0));
    try std.testing.expect(!usePresentedPrev(never, 41));
    // Staged-but-unrendered (latch without render, superseded front).
    try std.testing.expect(!usePresentedPrev(42, 41));
    // Reused front replaying old motion: prev gen older than the render.
    try std.testing.expect(!usePresentedPrev(39, 41));
    // Steady state: exact match pairs.
    try std.testing.expect(usePresentedPrev(41, 41));
    // Zero is a valid empty generation only when both are zero — and a
    // real staged frame never carries 0 (slots start at 0 = empty, and
    // the commit skips empty fronts).
    try std.testing.expect(usePresentedPrev(0, 0));
    try std.testing.expect(!usePresentedPrev(0, 7));
}

test "headless pass stays zero: init, ensure, render, tex view" {
    // No sg context here: every entry point must fail closed with no sg
    // calls beyond the isvalid probe and no state published.
    var pass = VelocityPass.init();
    try std.testing.expectEqual(sg.Image{}, pass.image);
    try std.testing.expectEqual(sg.View{}, pass.tex_view);
    try std.testing.expectEqual(@as(i32, 0), pass.width);
    try std.testing.expectEqual(@as(i32, 0), pass.height);

    try std.testing.expect(!pass.ensure(64, 64));
    try std.testing.expectEqual(@as(i32, 0), pass.width);
    try std.testing.expect(pass.image.id == 0);

    try std.testing.expect(!pass.ensure(0, 64));
    try std.testing.expect(!pass.ensure(-1, 64));

    // Zero target reports the zero dummy (also zero headless).
    try std.testing.expectEqual(@as(u32, 0), pass.velocityTexView().id);

    // Render with empty queues is a guarded no-op: stats untouched.
    var queues = RenderQueues{};
    defer queues.deinit(std.testing.allocator);
    var stats = SceneStats{};
    const skins = [_][scene_render_queue.MAX_BONES]Mat4{};
    pass.render(Mat4.identity, Mat4.identity, &queues, &skins, &skins, .{}, .DEFAULT, 0, &stats);
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), stats.main_draw_calls);

    pass.deinit();
    try std.testing.expectEqual(@as(i32, 0), pass.width);
}

test "shared-depth plan binds visible main pixels, never clears depth" {
    // Pipeline depth state: writes off, EQUAL compare (see
    // velocityDepthState docs for the morph/cutout occlusion meaning).
    const ds = velocityDepthState();
    try std.testing.expect(!ds.write_enabled);
    try std.testing.expectEqual(sg.CompareFunc.EQUAL, ds.compare);

    // Pass action: own color cleared to the zero mask; borrowed main depth
    // loaded and stored (post chain samples it right after — no clear, no
    // dontcare).
    const pa = velocityDepthPassAction();
    try std.testing.expectEqual(sg.LoadAction.CLEAR, pa.colors[0].load_action);
    try std.testing.expectEqual(sg.StoreAction.STORE, pa.colors[0].store_action);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), pa.colors[0].clear_value.r, 1e-9);
    try std.testing.expectEqual(@as(f32, 0.0), pa.colors[0].clear_value.a);
    try std.testing.expectEqual(sg.LoadAction.LOAD, pa.depth.load_action);
    try std.testing.expectEqual(sg.StoreAction.STORE, pa.depth.store_action);

    // Occlusion meaning, pinned headless (no pixels asserted — the GPU
    // gate proves those): with EQUAL + write-off against borrowed main
    // depth, a nearer main surface (morph-displaced pixel, opaque pixel)
    // rejects a farther velocity fragment, so the target keeps the zero
    // mask (depth fallback) instead of adopting background motion; a
    // matching visible pixel passes. Cutout holes carry the farther
    // background depth and fail the match — no alpha texture needed except
    // for truly coplanar cutouts. The per-draw side of that contract is
    // velocityDrawDecision below (mask0 skips vs draws).
    try std.testing.expectEqual(.draw, velocityDrawDecision(.{ .index_count = 3 }));
    try std.testing.expectEqual(.skip_mask0, velocityDrawDecision(.{ .index_count = 3, .morph_fallback = true }));
    try std.testing.expectEqual(.skip_mask0, velocityDrawDecision(.{ .index_count = 3, .transparent = true }));
    try std.testing.expectEqual(.skip_mask0, velocityDrawDecision(.{ .index_count = 3, .is_decal = true }));
    try std.testing.expectEqual(.skip_mask0, velocityDrawDecision(.{ .index_count = 3, .has_hook = true }));
    try std.testing.expectEqual(.skip_mask0, velocityDrawDecision(.{ .index_count = 0 }));
    // Cutout (alpha-tested, opaque) DRAWS: its holes fail the EQUAL match
    // against main depth instead of needing an in-shader discard.
    try std.testing.expectEqual(.draw, velocityDrawDecision(.{ .index_count = 3, .is_cutout = true }));
}
