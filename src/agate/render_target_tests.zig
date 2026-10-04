//! Tests for `render_target.zig`.
const std = @import("std");
const sg = @import("sokol").gfx;
const math = @import("math");
const Color4 = math.Color4;
const clustered_lights = @import("scene/clustered_lights.zig");
const prod = @import("render_target.zig");
const validateDimensions = prod.validateDimensions;
const max_target_dimension = prod.max_target_dimension;
const snappedSamples = prod.snappedSamples;
const needsResolve = prod.needsResolve;
const isSrgbFormat = prod.isSrgbFormat;
const isDepthFormat = prod.isDepthFormat;
const estimatedBytesFor = prod.estimatedBytesFor;
const defaultColorFormat = prod.defaultColorFormat;
const RenderTargetDesc = prod.RenderTargetDesc;
const RenderTarget = prod.RenderTarget;
const queryCapabilities = prod.queryCapabilities;
const Capabilities = prod.Capabilities;

const testing = std.testing;

test "validateDimensions rejects empty and overflowing sizes" {
    try testing.expectError(error.InvalidDimensions, validateDimensions(0, 64));
    try testing.expectError(error.InvalidDimensions, validateDimensions(64, 0));
    try testing.expectError(error.InvalidDimensions, validateDimensions(0, 0));
    // 70000^2 overflows u32: old wrap-to-small hazard, now ImageTooLarge.
    try testing.expectError(error.ImageTooLarge, validateDimensions(70000, 70000));
    try testing.expectError(error.ImageTooLarge, validateDimensions(max_target_dimension + 1, 4));
    try validateDimensions(1, 1);
    try validateDimensions(4096, 2160);
    try validateDimensions(max_target_dimension, max_target_dimension);
}

test "snappedSamples follows the engine MSAA policy per backend" {
    try testing.expectEqual(@as(i32, 1), snappedSamples(0, .DUMMY));
    try testing.expectEqual(@as(i32, 1), snappedSamples(1, .DUMMY));
    try testing.expectEqual(@as(i32, 2), snappedSamples(3, .DUMMY));
    try testing.expectEqual(@as(i32, 8), snappedSamples(8, .DUMMY));
    // Real backends cap at 4x (same table as the main target).
    try testing.expectEqual(@as(i32, 4), snappedSamples(8, .METAL_MACOS));
    try testing.expectEqual(@as(i32, 4), snappedSamples(99, .VULKAN));
    try testing.expectEqual(@as(i32, 2), snappedSamples(2, .GLCORE));
}

test "needsResolve follows the sokol resolve contract" {
    try testing.expect(!needsResolve(1));
    try testing.expect(needsResolve(2));
    try testing.expect(needsResolve(4));
}

test "isSrgbFormat flags exactly the hardware-sRGB variants" {
    try testing.expect(isSrgbFormat(.SRGB8A8));
    try testing.expect(isSrgbFormat(.SBGR8A8));
    try testing.expect(isSrgbFormat(.BC7_SRGBA));
    try testing.expect(isSrgbFormat(.BC3_SRGBA));
    try testing.expect(isSrgbFormat(.ETC2_SRGB8A8));
    try testing.expect(isSrgbFormat(.ASTC_4x4_SRGBA));
    try testing.expect(!isSrgbFormat(.RGBA8));
    try testing.expect(!isSrgbFormat(.BGRA8));
    try testing.expect(!isSrgbFormat(.RGBA16F));
    try testing.expect(!isSrgbFormat(.DEPTH));
    try testing.expect(!isSrgbFormat(.DEFAULT));
}

test "estimatedBytesFor accounts color, MSAA store, resolve, and depth" {
    // 4x4 RGBA8, 1x, no depth: 16 px * 4 B.
    try testing.expectEqual(@as(usize, 64), estimatedBytesFor(4, 4, .RGBA8, .NONE, 1));
    // With DEPTH (4 B/px): +64.
    try testing.expectEqual(@as(usize, 128), estimatedBytesFor(4, 4, .RGBA8, .DEPTH, 1));
    // 4x MSAA: color store x4 + one 1x resolve copy + depth store x4.
    try testing.expectEqual(@as(usize, 64 * 4 + 64 + 64 * 4), estimatedBytesFor(4, 4, .RGBA8, .DEPTH, 4));
    // Zero-size target estimates zero.
    try testing.expectEqual(@as(usize, 0), estimatedBytesFor(0, 0, .RGBA8, .NONE, 1));
}

test "defaultColorFormat is pure linear HDR" {
    // No sg calls: the HDR default holds headless (no context needed).
    try testing.expectEqual(sg.PixelFormat.RGBA16F, defaultColorFormat());
    // .DEFAULT in the descriptor resolves to that same HDR format.
    const d = RenderTargetDesc{ .color_format = .DEFAULT };
    const resolved = if (d.color_format != .DEFAULT) d.color_format else defaultColorFormat();
    try testing.expectEqual(sg.PixelFormat.RGBA16F, resolved);
}

test "zero target is invalid, estimates zero, and exposes empty views" {
    var t = RenderTarget{};
    try testing.expect(!t.isValid());
    try testing.expect(!t.hasDepth());
    try testing.expectEqual(@as(usize, 0), t.estimatedBytes());
    try testing.expectEqual(@as(u32, 0), t.sampleView().id);
    try testing.expectEqual(@as(u32, 0), t.sampleSampler().id);
    try testing.expectEqual(@as(u32, 0), t.depthSampleView().id);
    // Headless deinit is a safe no-op reset (never touches sg).
    t.deinit();
    try testing.expect(!t.isValid());
}

test "create fails closed headless with NoContext and allocates nothing" {
    try testing.expectError(error.NoContext, RenderTarget.create(.{ .width = 64, .height = 64 }));
    try testing.expectError(error.NoContext, queryCapabilities(.RGBA8, .DEPTH));
    // Dimension validation runs before the context gate.
    try testing.expectError(error.InvalidDimensions, RenderTarget.create(.{ .width = 0, .height = 64 }));
}

test "begin/clear/resize fail closed headless; resize refuses open passes" {
    var t = RenderTarget{};
    try testing.expect(!t.begin(Color4.new(0, 0, 0, 1), 1.0));
    try testing.expect(!t.clear(Color4.new(0, 0, 0, 1), 1.0));
    try testing.expect(!t.resize(64, 64));
    try testing.expect(!t.isValid());
    try testing.expect(!t.isCapturing());
    // A stuck-open pass refuses rebuilds (and would trip the deinit
    // tripwire): fail closed, content conceptually intact.
    t.pass_open = true;
    try testing.expect(t.isCapturing());
    try testing.expect(!t.resize(64, 64));
    try testing.expect(!t.begin(Color4.new(0, 0, 0, 1), 1.0));
    // Sampling views stay empty while the pass is open (no in-flight
    // self-sampling); the borrow keeps dims but drops handles.
    try testing.expectEqual(@as(u32, 0), t.sampleView().id);
    try testing.expectEqual(@as(u32, 0), t.sampleSampler().id);
    try testing.expectEqual(@as(u32, 0), t.depthSampleView().id);
    try testing.expectEqual(@as(u32, 0), t.asTexture().image.id);
    t.pass_open = false;
    // End without begin is a safe no-op (never a bare sg.endPass).
    t.end();
    try testing.expect(!t.isCapturing());
}

test "failed resize retains the old target verbatim" {
    // CPU-only: a populated (never-created-headless) value pins the
    // rollback contract — invalid dims fail before anything is touched.
    var t = RenderTarget{
        .width = 128,
        .height = 64,
        .sample_count = 1,
        .color_format = .RGBA8,
        .depth_format = .DEPTH,
        .color_image = .{ .id = 11 },
        .color_att_view = .{ .id = 12 },
        .color_tex_view = .{ .id = 13 },
        .depth_image = .{ .id = 15 },
        .depth_att_view = .{ .id = 16 },
        .depth_tex_view = .{ .id = 17 },
        .sampler = .{ .id = 14 },
        .valid = true,
    };
    try testing.expect(!t.resize(0, 64));
    try testing.expect(!t.resize(1_000_000, 64));
    // Untouched: same dims, same handles, still valid, no capture latched.
    try testing.expectEqual(@as(u32, 128), t.width);
    try testing.expectEqual(@as(u32, 64), t.height);
    try testing.expectEqual(@as(u32, 11), t.color_image.id);
    try testing.expectEqual(@as(u32, 13), t.color_tex_view.id);
    try testing.expect(t.isValid());
    try testing.expect(t.hasDepth());
    try testing.expect(!t.isCapturing());
}

test "renderPrimaryView enforces its gates as errors, headless" {
    // Pure gates first (no scene dereference before them): an invalid
    // target and an out-of-range slot are refused without a context, so a
    // real (but GPU-less) scene is safe to hand in — this pins the order.
    const alloc = testing.allocator;
    var scene = @import("testing.zig").testScene(alloc);
    defer {
        scene.lights.deinit(alloc);
        scene.cameras.deinit(alloc);
        scene.draws.deinit(alloc);
        scene.gpu_retire.deinit(alloc);
        scene.profiler.deinit();
    }
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);

    var invalid = RenderTarget{};
    try testing.expectError(error.InvalidTarget, invalid.renderPrimaryView(&scene, .{}));
    try testing.expectError(
        error.InvalidTarget,
        invalid.renderPrimaryView(&scene, .{ .view_slot = clustered_lights.MAX_VIEW_SLOTS }),
    );

    // A populated HDR value passes the pure gates and fails at the context
    // gate headless (live: format/consumable/camera gates follow).
    var headless = RenderTarget{
        .width = 64,
        .height = 64,
        .sample_count = 1,
        .color_format = .RGBA16F,
        .depth_format = .DEPTH,
        .color_image = .{ .id = 21 },
        .color_att_view = .{ .id = 22 },
        .color_tex_view = .{ .id = 23 },
        .depth_image = .{ .id = 25 },
        .depth_att_view = .{ .id = 26 },
        .depth_tex_view = .{ .id = 27 },
        .sampler = .{ .id = 24 },
        .valid = true,
    };
    try testing.expectError(error.NoContext, headless.renderPrimaryView(&scene, .{}));
    // Out-of-range slots are refused before the context gate (pure check).
    try testing.expectError(
        error.InvalidViewSlot,
        headless.renderPrimaryView(&scene, .{ .view_slot = clustered_lights.MAX_VIEW_SLOTS }),
    );
    try testing.expectError(
        error.InvalidViewSlot,
        headless.renderPrimaryView(&scene, .{ .view_slot = std.math.maxInt(usize) }),
    );
}

test "capabilities predicates stay pure and conservative" {
    const no_gpu = Capabilities{};
    try testing.expect(!no_gpu.supportsTarget(.RGBA8, .DEPTH));
    try testing.expect(!no_gpu.supportsTarget(.NONE, .NONE));
    try testing.expect(no_gpu.supportsSamples(1));
    try testing.expect(!no_gpu.supportsSamples(4));
    try testing.expect(isDepthFormat(.DEPTH));
    try testing.expect(isDepthFormat(.DEPTH_STENCIL));
    try testing.expect(!isDepthFormat(.RGBA8));
    try testing.expect(!isDepthFormat(.NONE));
    const hw = Capabilities{
        .backend = .METAL_MACOS,
        .color_sample = true,
        .color_filter = true,
        .color_render = true,
        .color_blend = true,
        .color_msaa = true,
        .depth_render = true,
        .depth_msaa = true,
        .max_image_2d = 16384,
        .max_samples = 4,
    };
    try testing.expect(hw.supportsTarget(.RGBA8, .DEPTH));
    try testing.expect(hw.supportsTarget(.RGBA8, .NONE));
    // Capture needs exactly the HDR scene format with full blend caps.
    try testing.expect(hw.supportsCapture(.RGBA16F, .DEPTH));
    try testing.expect(!hw.supportsCapture(.RGBA8, .DEPTH));
    try testing.expect(!hw.supportsCapture(.RGBA32F, .DEPTH));
    try testing.expect(!hw.supportsCapture(.RGBA16F, .NONE));
    var no_blend = hw;
    no_blend.color_blend = false;
    try testing.expect(!no_blend.supportsCapture(.RGBA16F, .DEPTH));
    // Creation stays blend-agnostic (opaque/clear-only targets allowed).
    try testing.expect(no_blend.supportsTarget(.RGBA16F, .DEPTH));
    // A non-depth format is never a valid depth attachment; unresolved
    // sentinels are never valid targets.
    try testing.expect(!hw.supportsTarget(.RGBA8, .RGBA8));
    try testing.expect(!hw.supportsTarget(.DEFAULT, .DEPTH));
    try testing.expect(hw.supportsSamples(4));
    try testing.expect(!hw.supportsSamples(8));
    var no_msaa = hw;
    no_msaa.color_msaa = false;
    try testing.expect(!no_msaa.supportsSamples(2));
    // 1x never needs MSAA support.
    try testing.expect(no_msaa.supportsSamples(1));
}

test "asTexture borrows handles without taking ownership" {
    // CPU-only: a manually populated value (no sg calls) pins the mapping
    // the sampling passes will rely on — same ids, same size, single level.
    const t = RenderTarget{
        .width = 128,
        .height = 64,
        .sample_count = 1,
        .color_format = .RGBA8,
        .depth_format = .DEPTH,
        .color_image = .{ .id = 11 },
        .color_att_view = .{ .id = 12 },
        .color_tex_view = .{ .id = 13 },
        .depth_image = .{ .id = 15 },
        .sampler = .{ .id = 14 },
        .valid = true,
    };
    try testing.expect(t.isValid());
    try testing.expect(t.hasDepth());
    var tex = t.asTexture();
    try testing.expect(!tex.owns_handles);
    tex.deinit(); // safe even with nonzero handles and no sg context
    try testing.expectEqual(@as(u32, 11), t.color_image.id);
    try testing.expectEqual(@as(u32, 11), tex.image.id);
    try testing.expectEqual(@as(u32, 13), tex.view.id);
    try testing.expectEqual(@as(u32, 14), tex.sampler.id);
    try testing.expectEqual(@as(u32, 128), tex.width);
    try testing.expectEqual(@as(u32, 64), tex.height);
    try testing.expectEqual(@as(u32, 1), tex.num_mipmaps);
    try testing.expectEqual(sg.PixelFormat.RGBA8, tex.format);
    try testing.expect(!tex.is_hdr);
    // HDR targets borrow as HDR (linear radiance, not display-referred).
    const h = RenderTarget{ .color_format = .RGBA16F, .valid = true };
    try testing.expect(h.asTexture().is_hdr);
}
test "default descriptor is a depth-carrying 1x target" {
    const d = RenderTargetDesc{};
    try testing.expectEqual(@as(u32, 256), d.width);
    try testing.expectEqual(@as(u32, 256), d.height);
    try testing.expectEqual(@as(i32, 1), d.sample_count);
    try testing.expect(d.depth_enabled);
    try testing.expect(!needsResolve(d.sample_count));
}

test "capture feedback detects saved material and particle texture views" {
    const t = RenderTarget{ .color_tex_view = .{ .id = 31 }, .depth_tex_view = .{ .id = 32 } };
    const Record = @import("material/draw_record.zig").MaterialDrawRecord;
    try testing.expect(!t.recordSamplesSelf(Record{}));
    try testing.expect(t.recordSamplesSelf(Record{ .albedo_view = .{ .id = 31 } }));
    try testing.expect(t.recordSamplesSelf(Record{ .normal_view = .{ .id = 32 } }));
    const Particle = @import("scene/particle_layer.zig").ParticleDraw;
    try testing.expect(t.recordSamplesSelf(Particle{ .texture_view = .{ .id = 31 } }));
}

