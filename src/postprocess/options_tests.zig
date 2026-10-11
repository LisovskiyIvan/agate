//! Tests for `postprocess/options.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const prod = @import("options.zig");
const ssgi = @import("ssgi.zig");
const PostProcessOptions = prod.PostProcessOptions;
const TonemappingType = prod.TonemappingType;

test "config clamped sanitizes new fields" {
    var cfg = PostProcessOptions{
        .exposure = -2.0,
        .dof = .{
            .focus_distance = -4.0,
            .focus_range = -1.0,
            .max_blur = -3.0,
        },
        .grade = .{
            .shadows = .{ 2.0, -2.0, 0.5 },
        },
    };
    const out = cfg.clamped();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.exposure, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.dof.focus_distance, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.dof.focus_range, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.dof.max_blur, 1e-6);
    try std.testing.expectEqual([3]f32{ 1.0, -1.0, 0.5 }, out.grade.shadows);
}

test "ssr and motion blur clamp quality options" {
    var custom = PostProcessOptions{
        .ssr = .{
            .steps = 1,
        },
        .motion_blur = .{
            .steps = 100,
        },
    };
    const c = custom.clamped();
    try std.testing.expectEqual(@as(u32, 4), c.ssr.steps);
    try std.testing.expectEqual(@as(u32, 32), c.motion_blur.steps);
}

test "manual exposure is finite and bounded for every output frame" {
    const defaults = PostProcessOptions{};
    try std.testing.expectEqual(defaults, defaults.clamped());
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |value| {
        const cfg = (PostProcessOptions{
            .exposure = value,
        }).clamped();
        try std.testing.expectEqual(@as(f32, 1), cfg.exposure);
    }
    try std.testing.expectEqual(@as(f32, 0), (PostProcessOptions{
        .exposure = -2,
    }).clamped().exposure);
    try std.testing.expectEqual(@as(f32, 65504), (PostProcessOptions{
        .exposure = 1e20,
    }).clamped().exposure);
}

test "effect master preserves exposure and tone curve without mutating authored config" {
    const config = PostProcessOptions{
        .exposure = 2,
        .tonemapping = .reinhard,
        .taa = .{
            .enabled = true,
        },
        .glow = .{
            .enabled = true,
        },
    };
    const frame = config.forFrame();
    try std.testing.expectEqual(@as(f32, 2), frame.exposure);
    try std.testing.expectEqual(TonemappingType.reinhard, frame.tonemapping);
    try std.testing.expect(!frame.bloom.enabled and !frame.glow.enabled and !frame.taa.enabled and !frame.fxaa.enabled and !frame.fog.enabled and !frame.ssr.enabled and !frame.contact_shadows.enabled and !frame.local_tonemapping.enabled);
    try std.testing.expectEqual(@as(f32, 1), frame.grade.saturation);
    try std.testing.expectEqual(@as(f32, 1), frame.grade.contrast);
    try std.testing.expect(config.taa.enabled and config.glow.enabled);
    const enabled = PostProcessOptions{
        .enabled = true,
        .taa = .{
            .enabled = true,
        },
    };
    try std.testing.expectEqual(enabled.clamped(), enabled.forFrame());
}

test "contact shadows options clamp and validate correctly" {
    var custom = PostProcessOptions{
        .contact_shadows = .{
            .intensity = 2.5,
            .distance = -0.5,
            .thickness = -0.1,
            .steps = 100,
        },
    };
    const c = custom.clamped();
    try std.testing.expectEqual(@as(f32, 1.0), c.contact_shadows.intensity);
    try std.testing.expectEqual(@as(f32, 0.01), c.contact_shadows.distance);
    try std.testing.expectEqual(@as(f32, 0.001), c.contact_shadows.thickness);
    try std.testing.expectEqual(@as(u32, 32), c.contact_shadows.steps);

    // Non-finite values fallback gracefully
    const nan_cfg = (PostProcessOptions{
        .contact_shadows = .{
            .intensity = std.math.nan(f32),
            .distance = std.math.inf(f32),
            .thickness = -std.math.inf(f32),
        },
    }).clamped();
    try std.testing.expectEqual(@as(f32, 0.5), nan_cfg.contact_shadows.intensity);
    try std.testing.expectEqual(@as(f32, 0.3), nan_cfg.contact_shadows.distance);
    try std.testing.expectEqual(@as(f32, 0.05), nan_cfg.contact_shadows.thickness);
}

test "local tonemapping options clamp and validate correctly" {
    var custom = PostProcessOptions{
        .local_tonemapping = .{
            .intensity = 3.0,
        },
    };
    const c = custom.clamped();
    try std.testing.expectEqual(@as(f32, 1.0), c.local_tonemapping.intensity);

    const nan_cfg = (PostProcessOptions{
        .local_tonemapping = .{
            .intensity = std.math.nan(f32),
        },
    }).clamped();
    try std.testing.expectEqual(@as(f32, 0.5), nan_cfg.local_tonemapping.intensity);
}

test "render scale clamps to [min, 1] with a finite default" {
    var over = PostProcessOptions{
        .render_scale = 1.5,
    };
    try std.testing.expectEqual(@as(f32, 1.0), over.clamped().render_scale);
    var under = PostProcessOptions{
        .render_scale = 0.1,
    };
    try std.testing.expectEqual(prod.min_render_scale, under.clamped().render_scale);
    var nan = PostProcessOptions{
        .render_scale = std.math.nan(f32),
    };
    try std.testing.expectEqual(@as(f32, 1.0), nan.clamped().render_scale);
    var zero = PostProcessOptions{
        .render_scale = 0.0,
    };
    try std.testing.expectEqual(prod.min_render_scale, zero.clamped().render_scale);
    // Default (fresh struct) is native: the pre-scale path stays bit-identical.
    try std.testing.expectEqual(@as(f32, 1.0), (PostProcessOptions{}).render_scale);
}

test "scaledRenderSize rounds, keeps aspect, and never collapses a dimension" {
    // Native identity at scale 1.0 (byte-identical to the pre-scale path).
    const native = prod.scaledRenderSize(1920, 1080, 1.0);
    try std.testing.expectEqual(@as(i32, 1920), native.w);
    try std.testing.expectEqual(@as(i32, 1080), native.h);

    // Nearest rounding, aspect error under half a source pixel.
    const half = prod.scaledRenderSize(1920, 1080, 0.5);
    try std.testing.expectEqual(@as(i32, 960), half.w);
    try std.testing.expectEqual(@as(i32, 540), half.h);
    const twothirds = prod.scaledRenderSize(1920, 1080, 0.66);
    try std.testing.expectEqual(@as(i32, 1267), twothirds.w); // 1267.2 -> 1267
    try std.testing.expectEqual(@as(i32, 713), twothirds.h); // 712.8 -> 713
    try std.testing.expectApproxEqAbs(1920.0 / 1080.0, @as(f64, @floatFromInt(twothirds.w)) / @as(f64, @floatFromInt(twothirds.h)), 0.002);

    // Defensive clamps: scale above 1 falls back to native, and each
    // dimension stays >= 1.
    try std.testing.expectEqual(@as(i32, 1920), prod.scaledRenderSize(1920, 1080, 4.0).w);
    try std.testing.expectEqual(@as(i32, 1080), prod.scaledRenderSize(1920, 1080, 4.0).h);
    const tiny = prod.scaledRenderSize(4, 4, prod.min_render_scale);
    try std.testing.expectEqual(@as(i32, 1), tiny.w);
    try std.testing.expectEqual(@as(i32, 1), tiny.h);
    // Degenerate window passes through unchanged (callers reject <= 0).
    const zero = prod.scaledRenderSize(0, 1080, 0.5);
    try std.testing.expectEqual(@as(i32, 0), zero.w);
    try std.testing.expectEqual(@as(i32, 1080), zero.h);
}

test "ssgi clamps intensity, radius and steps" {
    var cfg = PostProcessOptions{
        .ssgi = .{
            .enabled = true,
            .intensity = 2.0,
            .radius = 50.0,
            .steps = 99,
        },
    };
    var c = cfg.clamped();
    try std.testing.expectEqual(@as(f32, 1.0), c.ssgi.intensity);
    try std.testing.expectEqual(ssgi.SSGI_RADIUS_MAX, c.ssgi.radius);
    try std.testing.expectEqual(ssgi.SSGI_STEPS_MAX, c.ssgi.steps);
    cfg = .{ .ssgi = .{ .enabled = true, .intensity = -1.0, .radius = 0.0, .steps = 1 } };
    c = cfg.clamped();
    try std.testing.expectEqual(@as(f32, 0.0), c.ssgi.intensity);
    try std.testing.expectEqual(ssgi.SSGI_RADIUS_MIN, c.ssgi.radius);
    try std.testing.expectEqual(ssgi.SSGI_STEPS_MIN, c.ssgi.steps);
    // master switch off zeroes the family (forFrame path); default stays off.
    var dead = PostProcessOptions{
        .ssgi = .{
            .enabled = true,
            .intensity = 0.8,
        },
    };
    dead.enabled = false;
    try std.testing.expect(!dead.forFrame().ssgi.enabled);
    try std.testing.expect(!(PostProcessOptions{}).ssgi.enabled);
    // Temporal blend clamps to [0, 1]; the flag defaults off and survives
    // forFrame like the mode enum (render-owned, not an artistic effect).
    var tb = PostProcessOptions{
        .ssgi = .{
            .temporal = .{
                .blend = 2.0,
            },
        },
    };
    try std.testing.expectEqual(@as(f32, 1.0), tb.clamped().ssgi.temporal.blend);
    tb.ssgi.temporal.blend = -1.0;
    try std.testing.expectEqual(@as(f32, 0.0), tb.clamped().ssgi.temporal.blend);
    try std.testing.expect(!(PostProcessOptions{}).ssgi.temporal.enabled);
    var tframe = PostProcessOptions{
        .ssgi = .{
            .temporal = .{
                .enabled = true,
            },
        },
    };
    tframe.enabled = false;
    try std.testing.expect(tframe.forFrame().ssgi.temporal.enabled);
}

test "table-driven clamps map non-finite guarded floats to spec defaults" {
    const nan = std.math.nan(f32);
    // Guarded lanes fall back to their defaults.
    try std.testing.expectEqual(@as(f32, 0.5), (PostProcessOptions{
        .ssgi = .{
            .intensity = nan,
        },
    }).clamped().ssgi.intensity);
    try std.testing.expectEqual(@as(f32, 1.5), (PostProcessOptions{
        .ssgi = .{
            .radius = nan,
        },
    }).clamped().ssgi.radius);
    try std.testing.expectEqual(@as(f32, 0.5), (PostProcessOptions{
        .contact_shadows = .{
            .intensity = nan,
        },
    }).clamped().contact_shadows.intensity);
    try std.testing.expectEqual(@as(f32, 0.3), (PostProcessOptions{
        .contact_shadows = .{
            .distance = nan,
        },
    }).clamped().contact_shadows.distance);
    try std.testing.expectEqual(@as(f32, 2.0), (PostProcessOptions{
        .bloom = .{
            .radius = nan,
        },
    }).clamped().bloom.radius);
    try std.testing.expectEqual(@as(f32, 3.0), (PostProcessOptions{
        .auto_exposure = .{
            .speed_up = nan,
        },
    }).clamped().auto_exposure.speed_up);
    // Order-dependent leftovers still floor at the clamped neighbor.
    const dep = (PostProcessOptions{
        .auto_exposure = .{
            .min = 5.0,
            .max = 2.0,
        },
    }).clamped();
    try std.testing.expectEqual(@as(f32, 5.0), dep.auto_exposure.max);
    // Raw lanes keep the historical no-guard pass-through: NaN sorts low
    // (@max -> min), +Inf clamps to the bound.
    try std.testing.expectEqual(@as(f32, 0.0), (PostProcessOptions{
        .bloom = .{
            .threshold = nan,
        },
    }).clamped().bloom.threshold);
    try std.testing.expectEqual(@as(f32, 0.0), (PostProcessOptions{
        .glow = .{
            .intensity = nan,
        },
    }).clamped().glow.intensity);
    try std.testing.expectEqual(@as(f32, 1.0), (PostProcessOptions{
        .lut = .{
            .intensity = nan,
        },
    }).clamped().lut.intensity);
    try std.testing.expectEqual(@as(f32, 128.0), (PostProcessOptions{
        .motion_blur = .{
            .max_blur_px = std.math.inf(f32),
        },
    }).clamped().motion_blur.max_blur_px);
    try std.testing.expectEqual(@as(f32, 0.9), (PostProcessOptions{
        .shafts = .{
            .anisotropy = std.math.inf(f32),
        },
    }).clamped().shafts.anisotropy);
}

test "forFrame clears effect flags but keeps render-owned state" {
    const cfg = PostProcessOptions{
        .depth_pyramid_enabled = true,
        .render_scale = 0.5,
        .auto_exposure = .{
            .enabled = true,
        },
        .taa = .{
            .camera_cut = true,
        },
        .fxaa = .{
            .enabled = true,
        },
        .shafts = .{
            .enabled = true,
        },
    };
    const frame = cfg.forFrame();
    try std.testing.expect(!frame.fxaa.enabled and !frame.shafts.enabled);
    // Render-owned state (not artistic effects) survives the master switch.
    try std.testing.expect(frame.auto_exposure.enabled and frame.depth_pyramid_enabled and frame.taa.camera_cut);
    try std.testing.expectEqual(@as(f32, 0.5), frame.render_scale);
}
