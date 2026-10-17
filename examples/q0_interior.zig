//! Q0 calibration interior (fill-bound A/B workload, graphics-roadmap Q0/Q4).
//!
//! A Cornell-style room — floor, three walls, ceiling, a tall colored box,
//! a white sphere and a mirror sphere — lit by a warm ceiling area panel
//! plus a sun through the open side (shadows + volumetric shafts). The
//! fullscreen post chain runs at maximum per-pixel weight on purpose:
//! SSR (32 steps, depth pyramid), DoF, shafts, a 7-mip bloom pyramid,
//! local tonemapping and TAA. That makes the frame FILL-BOUND — the class
//! of workload the hdr-showcase A/B could not separate from harness noise
//! (MEASUREMENTS.md wave 3: that scene is draw-bound at 102 draws).
//!
//! Frame flow is the serial staged path (build -> begin -> finish ->
//! render) exactly like the other examples; the shared three-phase
//! render-scale bench (`gpu_phase_bench.zig`) drives the A/B:
//!
//!   AGATE_Q0_FRAMES=540 AGATE_Q0_TIMINGS=1 ./zig-out/bin/q0-interior
//!   (phases 1.0 -> 0.75 -> 0.66 x 180 frames; median is canonical)
//!
//! Cost attribution uses the drift-balanced interleaved A/B
//! (`AGATE_Q0_AB=matrix`, every pair in one process/window; see
//! `docs/post-ab.md`). Interactive runs show a feature-isolation HUD and
//! hotkeys (shadows, sun/point/spot/area, PCSS, IBL, SSAO/SSGI/SSR, TAA,
//! tonemap, fog, DoF, shafts, particles, glass); `AGATE_Q0_DISABLE` applies the
//! same switches for reproducible runs — see `docs/q0-isolation.md`.
//! No licensed assets yet (Q0 proper calls for them):
//! the checker floor and solid albedo panels carry the sharpness/ghosting
//! story for now. Run: `AGATE_Q0_FRAMES=540 zig build run-q0` (or the env
//! knobs above).

const std = @import("std");
const builtin = @import("builtin");
const z = @import("agate");
const sokol = @import("sokol");
const sapp = sokol.app;
const slog = sokol.log;
const sg = sokol.gfx;
const timing = z.gpu_timing;
const bench_mod = @import("gpu_phase_bench.zig");
const env_mod = @import("env.zig");
const envFlag = env_mod.envFlag;
const envF32 = env_mod.envF32;

var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
var scene: z.Scene = undefined;

// Feature-isolation handles: the lights/meshes/UI are owned by the scene;
// these are stable pointers captured at setup for the runtime toggles.
var sun_light: ?*z.DirectionalLight = null;
var point_light: ?*z.PointLight = null;
var spot_light: ?*z.SpotLight = null;
var area_light: ?*z.AreaLight = null;
var dust_system: ?*z.ParticleSystem = null;
var glass_pane: ?*z.Mesh = null;
var ui_canvas: ?*z.UICanvas = null;
var panel_enabled: bool = false;

var bench: bench_mod.Bench = .{};
/// Frame + phase channels share one block schedule. Frame alone is too
/// noisy for sub-millisecond effects (vsync/present and whole-frame drift
/// swamp them); post/main brackets are the phase the effect actually
/// changes. Each channel keeps its own completed-submission dedupe.
var ab: bench_mod.AbBench = .{};
var ab_post: bench_mod.AbBench = .{};
var ab_main: bench_mod.AbBench = .{};

/// Interleaved post/lighting A/B pairs (opt-in via AGATE_Q0_AB=<pair>).
/// Each pair alternates its scene-default arm A with the reduced arm B in
/// `AbBench`'s drift-balanced A B B A B A A B blocks, so the frame-time
/// delta is measured inside one process against the same target.
///
/// Cost-attribution pairs: ssgi, ssao, bloom, fxaa, taa, fog, shadows, pcss,
/// dof, shafts, local_tonemap, ssr_off. Mode pair `ssr` compares inline vs
/// half-res. `pcss` toggles local point/spot source_radius (0.02 -> 0); the
/// sun PCSS lane is left at the scene baseline so the arm isolates
/// local-shadow cost.
const AbPair = enum {
    ssgi,
    ssr,
    ssr_off,
    ssao,
    bloom,
    fxaa,
    taa,
    fog,
    shadows,
    pcss,
    dof,
    shafts,
    local_tonemap,

    fn parse(raw: []const u8) ?AbPair {
        inline for (@typeInfo(AbPair).@"enum".fields) |f| {
            if (std.mem.eql(u8, raw, f.name)) return @enumFromInt(f.value);
        }
        if (std.mem.eql(u8, raw, "shadow")) return .shadows;
        return null;
    }
};

var ab_mode: bool = false;
var ab_pair: AbPair = .ssgi;
/// Matrix mode (`AGATE_Q0_AB=matrix`) runs every pair in one process and
/// one window, back to back, so GPU clock/focus state cannot masquerade as
/// an effect delta. Each pair keeps its own drift-balanced AbBench.
const ab_matrix_pairs = [_]AbPair{
    .ssao,
    .ssgi,
    .ssr_off,
    .bloom,
    .fxaa,
    .taa,
    .shadows,
    .pcss,
    .dof,
    .shafts,
    .local_tonemap,
    .fog,
};
var ab_matrix: bool = false;
var ab_matrix_idx: usize = 0;
var ab_pair_frame: u32 = 0;
var ab_pair_limit: u32 = 0;
var ab_pair_done: bool = false;
var ab_block_frames: u32 = 60;
/// Frame index inside the CURRENT pair (matrix mode) or the global frame
/// (single-pair mode). AbBench samples and block positions must use this,
/// never the global frame, or pair 2+ would look like a mid-block switch.
var ab_record_frame: u32 = 0;

fn abActive(pair: AbPair) bool {
    return ab_mode and ab_pair == pair;
}

fn abArmName(pair: AbPair, arm: bench_mod.AbArm) []const u8 {
    return switch (pair) {
        .ssgi => if (arm == .A) "ssgi_on" else "ssgi_off",
        .ssr => if (arm == .A) "ssr_inline" else "ssr_halfres",
        .ssr_off => if (arm == .A) "ssr_on" else "ssr_off",
        .ssao => if (arm == .A) "ssao_on" else "ssao_off",
        .bloom => if (arm == .A) "bloom_on" else "bloom_off",
        .fxaa => if (arm == .A) "fxaa_on" else "fxaa_off",
        .taa => if (arm == .A) "taa_on" else "taa_off",
        .fog => if (arm == .A) "fog_on" else "fog_off",
        .shadows => if (arm == .A) "shadows_on" else "shadows_off",
        .pcss => if (arm == .A) "local_pcss_on" else "local_pcss_off",
        .dof => if (arm == .A) "dof_on" else "dof_off",
        .shafts => if (arm == .A) "shafts_on" else "shafts_off",
        .local_tonemap => if (arm == .A) "local_tonemap_on" else "local_tonemap_off",
    };
}

/// Applies one arm on the game/context boundary, before update+render.
/// Toggles are deliberately cheap scene/option writes (no target rebuild):
/// AO/SSGI/SSR targets keep their sized allocations across switches, and
/// shadows rebuild from the staged snapshot on the next frame.
fn applyAbArm(pair: AbPair, arm: bench_mod.AbArm) void {
    const a = arm == .A;
    switch (pair) {
        .ssgi => scene.post_process.ssgi.enabled = a,
        .ssr => scene.post_process.ssr.mode = if (a) .inline_ else .half_res_pass,
        .ssr_off => scene.post_process.ssr.enabled = a,
        .ssao => scene.ssao.enabled = a,
        .bloom => scene.post_process.bloom.enabled = a,
        .fxaa => scene.post_process.fxaa.enabled = a,
        .taa => scene.post_process.taa.enabled = a,
        .fog => scene.post_process.fog.enabled = a,
        .shadows => scene.shadows.enabled = a,
        .pcss => {
            // Local PCSS only; the sun keeps the scene baseline.
            const radius: f32 = if (a) 0.02 else 0.0;
            for (scene.lights.point_lights.items) |pl| pl.shadow.source_radius = radius;
            for (scene.lights.spot_lights.items) |sl| sl.shadow.source_radius = radius;
        },
        .dof => scene.post_process.dof.enabled = a,
        .shafts => scene.post_process.shafts.enabled = a,
        .local_tonemap => scene.post_process.local_tonemapping.enabled = a,
    }
}

/// Matrix-mode baseline: the heavy q0 chain (AO horizon+temporal, SSGI
/// half+temporal, SSR half, bloom, TAA, FXAA, shadows, DoF, shafts, local
/// tonemap, local PCSS). Every pair starts from this same state; without
/// the reset, a previous pair's disabled effect would leak into the next
/// pair's arm A and turn the matrix into a cumulative test.
fn resetAbBaseline() void {
    scene.ssao.enabled = true;
    scene.ssao.mode = .horizon;
    scene.ssao.temporal.enabled = true;

    scene.post_process.ssgi.enabled = true;
    scene.post_process.ssgi.mode = .half_res_pass;
    scene.post_process.ssgi.temporal.enabled = true;

    scene.post_process.ssr.enabled = true;
    scene.post_process.ssr.mode = .half_res_pass;
    scene.post_process.bloom.enabled = true;
    scene.post_process.fxaa.enabled = true;
    scene.post_process.taa.enabled = true;
    scene.post_process.fog.enabled = false;
    scene.post_process.dof.enabled = true;
    scene.post_process.shafts.enabled = true;
    scene.post_process.local_tonemapping.enabled = true;
    scene.shadows.enabled = true;

    for (scene.lights.point_lights.items) |pl| pl.shadow.source_radius = 0.02;
    for (scene.lights.spot_lights.items) |sl| sl.shadow.source_radius = 0.02;
    scene.resetTaa();
}

fn beginAbPair(index: usize) void {
    ab_matrix_idx = index;
    ab = .{};
    ab_post = .{};
    ab_main = .{};
    ab.block_frames = ab_block_frames;
    ab_post.block_frames = ab_block_frames;
    ab_main.block_frames = ab_block_frames;
    ab_pair_limit = ab_block_frames * (8 + ab.settle_blocks);
    ab_pair_frame = 0;
    ab_pair_done = false;
    ab_pair = ab_matrix_pairs[index];
    resetAbBaseline();
    applyAbArm(ab_pair, .A);
    std.debug.print("q0-interior ab matrix: pair={s} index={}/{} block_frames={} limit={}\n", .{
        @tagName(ab_pair), index + 1, ab_matrix_pairs.len, ab_block_frames, ab_pair_limit,
    });
}

fn printAbResult(tag: []const u8, b: *bench_mod.AbBench, pair: AbPair) void {
    const s = b.finalSummary(abArmName(pair, .A), abArmName(pair, .B));
    b.printBlocks();
    bench_mod.AbBench.printSummary(s);
    std.debug.print("AB_{s}_RESULT pair={s} delta_median_ms={d:.3} noise_floor_ms={d:.3} verdict={s} n_a={} n_b={} dropped={} unstable={} median_a={d:.3} median_b={d:.3}\n", .{
        tag,                 @tagName(pair), s.delta_median_ms, s.noise_floor_ms,
        @tagName(s.verdict), s.n_a,          s.n_b,             s.dropped,
        s.unstable,          s.median_a,     s.median_b,
    });
}

fn finishAbPair() void {
    if (ab_pair_done) return;
    printAbResult("FRAME", &ab, ab_pair);
    printAbResult("POST", &ab_post, ab_pair);
    printAbResult("MAIN", &ab_main, ab_pair);
    ab_pair_done = true;
}

/// Frame-30 proof that requested AO history actually executed.
var ao_temporal_requested: bool = false;
var frame_limit: u32 = 0;
var frames: u32 = 0;
var log_errors: u32 = 0;
var sokol_live_allocations: usize = 0;
var failures: u32 = 0;
var render_scale: f32 = 1.0;
var camera: z.ArcRotateCamera = undefined;
var still_camera: bool = false;

// ---- feature isolation -----------------------------------------------------

/// One runtime-toggleable lighting/post feature. The panel and hotkeys use
/// this list; `AGATE_Q0_DISABLE=shadows,point,pcss` applies the same switches
/// non-interactively for reproducible bug reports. `R` restores the exact
/// post-env baseline (including features that were already off at startup).
const Feature = enum {
    shadows,
    sun,
    point,
    spot,
    area,
    hemi,
    ibl,
    pcss,
    sun_pcss,
    ssao,
    ssgi,
    ssr,
    taa,
    bloom,
    local_tonemap,
    fog,
    dof,
    shafts,
    particles,
    glass,
};

const feature_count = @typeInfo(Feature).@"enum".fields.len;

const FeatureInfo = struct {
    key: sapp.Keycode,
    key_char: u8,
    label: []const u8,
    env: []const u8,
};

const feature_info = [feature_count]FeatureInfo{
    .{ .key = ._1, .key_char = '1', .label = "SHADOWS", .env = "shadows" },
    .{ .key = ._2, .key_char = '2', .label = "SUN", .env = "sun" },
    .{ .key = ._3, .key_char = '3', .label = "POINT", .env = "point" },
    .{ .key = ._4, .key_char = '4', .label = "SPOT", .env = "spot" },
    .{ .key = ._5, .key_char = '5', .label = "AREA", .env = "area" },
    .{ .key = ._6, .key_char = '6', .label = "HEMI", .env = "hemi" },
    .{ .key = ._7, .key_char = '7', .label = "IBL", .env = "ibl" },
    .{ .key = ._8, .key_char = '8', .label = "LOCAL PCSS", .env = "pcss" },
    .{ .key = .U, .key_char = 'U', .label = "SUN PCSS", .env = "sun_pcss" },
    .{ .key = ._9, .key_char = '9', .label = "SSAO", .env = "ssao" },
    .{ .key = ._0, .key_char = '0', .label = "SSGI", .env = "ssgi" },
    .{ .key = .S, .key_char = 'S', .label = "SSR", .env = "ssr" },
    .{ .key = .T, .key_char = 'T', .label = "TAA", .env = "taa" },
    .{ .key = .B, .key_char = 'B', .label = "BLOOM", .env = "bloom" },
    .{ .key = .L, .key_char = 'L', .label = "LOCAL TONE", .env = "local_tonemap" },
    .{ .key = .F, .key_char = 'F', .label = "FOG", .env = "fog" },
    .{ .key = .D, .key_char = 'D', .label = "DOF", .env = "dof" },
    .{ .key = .H, .key_char = 'H', .label = "SHAFTS", .env = "shafts" },
    .{ .key = .P, .key_char = 'P', .label = "PARTICLES", .env = "particles" },
    .{ .key = .G, .key_char = 'G', .label = "GLASS", .env = "glass" },
};

const FeatureBaseline = struct {
    shadows: bool = true,
    sun: bool = true,
    point: bool = true,
    spot: bool = true,
    area: bool = true,
    hemi_intensity: f32 = 0,
    ibl_intensity: f32 = 1,
    point_radius: f32 = 0,
    spot_radius: f32 = 0,
    sun_pcss: bool = false,
    ssao: bool = true,
    ssgi: bool = true,
    ssr: bool = true,
    taa: bool = true,
    bloom: bool = true,
    local_tonemap: bool = true,
    fog: bool = false,
    dof: bool = true,
    shafts: bool = true,
    particles: bool = false,
    glass: bool = false,
};

var feature_baseline: FeatureBaseline = .{};
var feature_on: [feature_count]bool = .{true} ** feature_count;
var baseline_on: [feature_count]bool = .{true} ** feature_count;

fn featureIndexByName(name: []const u8) ?usize {
    inline for (feature_info, 0..) |info, i| {
        if (std.mem.eql(u8, name, info.env)) return i;
    }
    return null;
}

fn captureFeatureBaseline() void {
    feature_baseline = .{
        .shadows = scene.shadows.enabled,
        .sun = if (sun_light) |l| l.is_enabled else false,
        .point = if (point_light) |l| l.is_enabled else false,
        .spot = if (spot_light) |l| l.is_enabled else false,
        .area = if (area_light) |l| l.is_enabled else false,
        .hemi_intensity = scene.lights.hemi.intensity,
        .ibl_intensity = scene.sky.ibl_intensity,
        .point_radius = if (point_light) |l| l.shadow.source_radius else 0,
        .spot_radius = if (spot_light) |l| l.shadow.source_radius else 0,
        .sun_pcss = scene.shadows.pcss_enabled,
        .ssao = scene.ssao.enabled,
        .ssgi = scene.post_process.ssgi.enabled,
        .ssr = scene.post_process.ssr.enabled,
        .taa = scene.post_process.taa.enabled,
        .bloom = scene.post_process.bloom.enabled,
        .local_tonemap = scene.post_process.local_tonemapping.enabled,
        .fog = scene.post_process.fog.enabled,
        .dof = scene.post_process.dof.enabled,
        .shafts = scene.post_process.shafts.enabled,
        .particles = dust_system != null,
        .glass = if (glass_pane) |p| p.is_visible else false,
    };
    baseline_on = .{false} ** feature_count;
    baseline_on[@intFromEnum(Feature.shadows)] = feature_baseline.shadows;
    baseline_on[@intFromEnum(Feature.sun)] = feature_baseline.sun;
    baseline_on[@intFromEnum(Feature.point)] = feature_baseline.point;
    baseline_on[@intFromEnum(Feature.spot)] = feature_baseline.spot;
    baseline_on[@intFromEnum(Feature.area)] = feature_baseline.area;
    baseline_on[@intFromEnum(Feature.hemi)] = feature_baseline.hemi_intensity > 0.0001;
    baseline_on[@intFromEnum(Feature.ibl)] = feature_baseline.ibl_intensity > 0.0001;
    baseline_on[@intFromEnum(Feature.pcss)] = feature_baseline.point_radius > 0.0001 or feature_baseline.spot_radius > 0.0001;
    baseline_on[@intFromEnum(Feature.sun_pcss)] = feature_baseline.sun_pcss;
    baseline_on[@intFromEnum(Feature.ssao)] = feature_baseline.ssao;
    baseline_on[@intFromEnum(Feature.ssgi)] = feature_baseline.ssgi;
    baseline_on[@intFromEnum(Feature.ssr)] = feature_baseline.ssr;
    baseline_on[@intFromEnum(Feature.taa)] = feature_baseline.taa;
    baseline_on[@intFromEnum(Feature.bloom)] = feature_baseline.bloom;
    baseline_on[@intFromEnum(Feature.local_tonemap)] = feature_baseline.local_tonemap;
    baseline_on[@intFromEnum(Feature.fog)] = feature_baseline.fog;
    baseline_on[@intFromEnum(Feature.dof)] = feature_baseline.dof;
    baseline_on[@intFromEnum(Feature.shafts)] = feature_baseline.shafts;
    baseline_on[@intFromEnum(Feature.particles)] = feature_baseline.particles;
    baseline_on[@intFromEnum(Feature.glass)] = feature_baseline.glass;
    feature_on = baseline_on;
}

fn applyFeature(index: usize, on: bool) void {
    const f: Feature = @enumFromInt(index);
    switch (f) {
        .shadows => scene.shadows.enabled = on,
        .sun => {
            if (sun_light) |l| l.is_enabled = on;
        },
        .point => {
            if (point_light) |l| l.is_enabled = on;
        },
        .spot => {
            if (spot_light) |l| l.is_enabled = on;
        },
        .area => {
            if (area_light) |l| l.is_enabled = on;
        },
        .hemi => {
            const base = if (feature_baseline.hemi_intensity > 0.0001) feature_baseline.hemi_intensity else 0.18;
            scene.lights.hemi.intensity = if (on) base else 0;
        },
        .ibl => {
            const base = if (feature_baseline.ibl_intensity > 0.0001) feature_baseline.ibl_intensity else 1.0;
            scene.sky.ibl_intensity = if (on) base else 0;
        },
        .pcss => {
            // Local casters only; the sun PCSS lane stays at its env baseline.
            const pr: f32 = if (on) (if (feature_baseline.point_radius > 0.0001) feature_baseline.point_radius else 0.02) else 0;
            const sr: f32 = if (on) (if (feature_baseline.spot_radius > 0.0001) feature_baseline.spot_radius else 0.02) else 0;
            if (point_light) |l| l.shadow.source_radius = pr;
            if (spot_light) |l| l.shadow.source_radius = sr;
        },
        .sun_pcss => scene.shadows.pcss_enabled = on,
        .ssao => scene.ssao.enabled = on,
        .ssgi => scene.post_process.ssgi.enabled = on,
        .ssr => scene.post_process.ssr.enabled = on,
        .taa => scene.post_process.taa.enabled = on,
        .bloom => scene.post_process.bloom.enabled = on,
        .local_tonemap => scene.post_process.local_tonemapping.enabled = on,
        .fog => scene.post_process.fog.enabled = on,
        .dof => scene.post_process.dof.enabled = on,
        .shafts => scene.post_process.shafts.enabled = on,
        .particles => {
            if (dust_system) |d| {
                if (on) d.start() else d.stop();
            }
        },
        .glass => {
            if (glass_pane) |p| p.is_visible = on;
        },
    }
}

fn restoreFeatureBaseline() void {
    scene.shadows.enabled = feature_baseline.shadows;
    if (sun_light) |l| l.is_enabled = feature_baseline.sun;
    if (point_light) |l| l.is_enabled = feature_baseline.point;
    if (spot_light) |l| l.is_enabled = feature_baseline.spot;
    if (area_light) |l| l.is_enabled = feature_baseline.area;
    scene.lights.hemi.intensity = feature_baseline.hemi_intensity;
    scene.sky.ibl_intensity = feature_baseline.ibl_intensity;
    if (point_light) |l| l.shadow.source_radius = feature_baseline.point_radius;
    if (spot_light) |l| l.shadow.source_radius = feature_baseline.spot_radius;
    scene.shadows.pcss_enabled = feature_baseline.sun_pcss;
    scene.ssao.enabled = feature_baseline.ssao;
    scene.post_process.ssgi.enabled = feature_baseline.ssgi;
    scene.post_process.ssr.enabled = feature_baseline.ssr;
    scene.post_process.taa.enabled = feature_baseline.taa;
    scene.post_process.bloom.enabled = feature_baseline.bloom;
    scene.post_process.local_tonemapping.enabled = feature_baseline.local_tonemap;
    scene.post_process.fog.enabled = feature_baseline.fog;
    scene.post_process.dof.enabled = feature_baseline.dof;
    scene.post_process.shafts.enabled = feature_baseline.shafts;
    if (dust_system) |d| {
        if (feature_baseline.particles) d.start() else d.stop();
    }
    if (glass_pane) |p| p.is_visible = feature_baseline.glass;
    feature_on = baseline_on;
    scene.resetTaa();
    updateWindowTitle();
}

fn setFeature(index: usize, on: bool) void {
    feature_on[index] = on;
    applyFeature(index, on);
    scene.resetTaa();
    std.debug.print("q0-feature: {s}={s}\n", .{ feature_info[index].env, if (on) "on" else "off" });
    updateWindowTitle();
}

fn resetFeatures() void {
    restoreFeatureBaseline();
    std.debug.print("q0-feature: reset to baseline\n", .{});
}

fn printFeatureState() void {
    std.debug.print("q0-feature state:", .{});
    for (feature_info, 0..) |info, i| {
        std.debug.print(" {s}={}", .{ info.env, @intFromBool(feature_on[i]) });
    }
    std.debug.print(" hud={} baseline_captured=true\n", .{panel_enabled});
}

fn printFeatureHelp() void {
    std.debug.print(
        "q0-feature: keys 1 shadows 2 sun 3 point 4 spot 5 area 6 hemi 7 ibl 8 local-pcss U sun-pcss 9 ssao 0 ssgi S ssr T taa B bloom L local-tone F fog D dof H shafts P particles G glass; R reset O state F1 HUD; env AGATE_Q0_DISABLE=shadows,point,pcss (comma/space) AGATE_Q0_PANEL=1 AGATE_Q0_NO_PANEL=1\n",
        .{},
    );
}

fn applyDisableList() void {
    const raw = std.c.getenv("AGATE_Q0_DISABLE") orelse return;
    var it = std.mem.tokenizeAny(u8, std.mem.span(raw), ", \t");
    var changed: usize = 0;
    while (it.next()) |name| {
        if (featureIndexByName(name)) |i| {
            feature_on[i] = false;
            applyFeature(i, false);
            changed += 1;
        } else {
            std.debug.print("q0-feature: unknown AGATE_Q0_DISABLE entry '{s}'\n", .{name});
        }
    }
    if (changed > 0) {
        scene.resetTaa();
        updateWindowTitle();
        std.debug.print("q0-feature: disabled {} feature(s) from env\n", .{changed});
    }
}

fn appendTitle(buf: []u8, len: *usize, text: []const u8) void {
    const room = buf.len - len.*;
    const n = @min(room, text.len);
    @memcpy(buf[len.*..][0..n], text[0..n]);
    len.* += n;
}

fn updateWindowTitle() void {
    var buf: [512]u8 = undefined;
    var n: usize = 0;
    appendTitle(&buf, &n, "Agate Q0 isolation");
    var off: usize = 0;
    for (feature_on) |on| {
        if (!on) off += 1;
    }
    if (off == 0) {
        appendTitle(&buf, &n, " | all features ON");
    } else {
        appendTitle(&buf, &n, " | OFF:");
        for (feature_info, 0..) |info, i| {
            if (!feature_on[i]) {
                appendTitle(&buf, &n, " ");
                appendTitle(&buf, &n, info.env);
            }
        }
    }
    if (!panel_enabled) appendTitle(&buf, &n, " | HUD off (F1)");
    var title: [513:0]u8 = undefined;
    @memcpy(title[0..n], buf[0..n]);
    title[n] = 0;
    sapp.setWindowTitle(title[0..n :0]);
}

fn handleFeatureKey(key: sapp.Keycode) void {
    if (key == .F1) {
        panel_enabled = !panel_enabled;
        std.debug.print("q0-feature: HUD {s}\n", .{if (panel_enabled) "on" else "off"});
        updateWindowTitle();
        return;
    }
    if (key == .R) {
        resetFeatures();
        return;
    }
    if (key == .O) {
        printFeatureState();
        return;
    }
    inline for (feature_info, 0..) |info, i| {
        if (key == info.key) {
            setFeature(i, !feature_on[i]);
            return;
        }
    }
}

fn drawFeaturePanel() void {
    const ui = ui_canvas orelse return;
    ui.begin();
    if (!panel_enabled) return;
    ui.drawRect(8, 8, 520, 218, z.Color4.new(0.015, 0.02, 0.03, 0.74));
    ui.drawRectOutline(8, 8, 520, 218, 1, z.Color4.new(0.35, 0.55, 0.7, 0.55));
    ui.drawTextWithOutline("Q0 FEATURE ISOLATION", 16, 14, 13, z.Color4.new(0.55, 0.85, 1.0, 1), 0.25);
    ui.drawTextWithOutline("1-8 lights/shadows  U sun PCSS  9 AO  0 GI  S SSR  T TAA  B bloom", 16, 31, 9.5, z.Color4.new(0.72, 0.82, 0.92, 1), 0.2);
    ui.drawTextWithOutline("L localTone  F fog  D DoF  H shafts  P particles  G glass", 16, 43, 9.5, z.Color4.new(0.72, 0.82, 0.92, 1), 0.2);
    for (feature_info, 0..) |info, i| {
        const col = i / 10;
        const row = i % 10;
        const x: f32 = 16 + @as(f32, @floatFromInt(col)) * 252;
        const y: f32 = 60 + @as(f32, @floatFromInt(row)) * 14;
        var label_buf: [48]u8 = undefined;
        const label = std.fmt.bufPrint(&label_buf, "[{c}] {s}", .{ info.key_char, info.label }) catch info.label;
        const on = feature_on[i];
        ui.drawTextWithOutline(label, x, y, 10, if (on) z.Color4.new(0.72, 0.9, 0.78, 1) else z.Color4.new(0.55, 0.58, 0.62, 1), 0.2);
        ui.drawTextWithOutline(if (on) "ON" else "OFF", x + 176, y, 10, if (on) z.Color4.new(0.45, 1.0, 0.6, 1) else z.Color4.new(1.0, 0.55, 0.45, 1), 0.2);
    }
    ui.drawTextWithOutline("R reset  O state  F1 hide HUD", 16, 208, 9.5, z.Color4.new(0.6, 0.72, 0.85, 1), 0.2);
}

extern "c" fn malloc(size: usize) ?*anyopaque;
extern "c" fn free(ptr: ?*anyopaque) void;

fn sokolAlloc(size: usize, _: ?*anyopaque) callconv(.c) ?*anyopaque {
    const ptr = malloc(size) orelse return null;
    sokol_live_allocations += 1;
    return ptr;
}

fn sokolFree(ptr: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
    if (ptr != null) {
        if (sokol_live_allocations == 0) fail("sokol free without ownership", .{}) else sokol_live_allocations -= 1;
    }
    free(ptr);
}

fn allocator() std.mem.Allocator {
    return if (builtin.cpu.arch.isWasm()) std.heap.c_allocator else gpa.allocator();
}

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    std.debug.print("q0-interior FAIL: " ++ fmt ++ "\n", args);
}

fn envU32(name: [*:0]const u8, fallback: u32) u32 {
    const value = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseInt(u32, std.mem.span(value), 10) catch fallback;
}

/// Fixed-frame SSR wiring proof: at frame 30 (post-render) the half-res
/// mode must own a live pass sized to the render target, while inline/off
/// must hold no pass VRAM. Prints mode + target + post draws either way;
/// mismatches fail the run. Headless-safe (pure handle/size reads).
fn ssrProof() void {
    const fx_w = scene.postfx.renderWidth();
    const fx_h = scene.postfx.renderHeight();
    const draws = scene.stats.post_draw_calls;
    if (z.postprocess.ssrPassActive(scene.post_process)) {
        if (scene.postfx.ssr_pass.width != fx_w or scene.postfx.ssr_pass.height != fx_h) {
            fail("ssr proof: pass target {}x{} != render target {}x{}", .{ scene.postfx.ssr_pass.width, scene.postfx.ssr_pass.height, fx_w, fx_h });
        }
        if (scene.postfx.ssr_pass.blur_tex_view.id == 0) fail("ssr proof: pass active but denoised view dead", .{});
        std.debug.print("q0-interior ssr-proof: mode=half_res_pass target={}x{} post_draws={}\n", .{ fx_w, fx_h, draws });
    } else {
        // Inline/off owns no pass VRAM — except inside the SSR A/B, where
        // inline arms reuse the targets the half-res arms sized (same
        // EFFECTIVE size, so resize is a no-op and nothing reallocates).
        if (!abActive(.ssr) and scene.postfx.ssr_pass.width != 0) {
            fail("ssr proof: mode={s} but pass targets allocated", .{@tagName(scene.post_process.ssr.mode)});
        }
        std.debug.print("q0-interior ssr-proof: mode={s} no-pass post_draws={}\n", .{ @tagName(scene.post_process.ssr.mode), draws });
    }
}

fn qLogger(tag: [*c]const u8, level: u32, item: u32, msg: [*c]const u8, line: u32, file: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    if (level <= 1) {
        log_errors += 1;
        fail("sokol log: {s}", .{msg});
    }
    slog.func(tag, level, item, msg, line, file, user_data);
}

// ---- scene -----------------------------------------------------------------

fn pbr(name: []const u8, albedo: [3]f32, metallic: f32, roughness: f32) *z.PBRMaterial {
    const m = scene.createPBRMaterial(name) catch @panic("q0: material failed");
    m.albedo_color = z.Color3.new(albedo[0], albedo[1], albedo[2]);
    m.metallic = metallic;
    m.roughness = roughness;
    return m;
}

fn wall(name: []const u8, w: f32, h: f32) *z.Mesh {
    const m = z.MeshBuilder.createPlane(&scene, name, .{ .width = w, .height = h, .subdivisions_x = 2, .subdivisions_y = 2 }) catch @panic("q0: wall failed");
    m.cast_shadows = true;
    m.receive_shadows = true;
    return m;
}

fn setupScene() void {
    scene.active_camera = .{ .arc_rotate = camera };
    scene.clear_color = z.Color4.new(0.02, 0.02, 0.025, 1.0);

    _ = scene.createHemisphericLight("hemi", .{
        .direction = z.Vec3.new(0.1, 1.0, 0.2),
        .diffuse = z.Color3.white,
        .ground_color = z.Color3.new(0.12, 0.1, 0.09),
        .intensity = if (envFlag("AGATE_Q0_NO_HEMI")) 0.0 else 0.18,
    });

    // Sun through the open front side: drives CSM + volumetric shafts.
    sun_light = scene.createDirectionalLight("sun", .{
        .direction = z.Vec3.new(0.25, 1.2, 1.0),
        .diffuse = z.Color3.new(1.0, 0.92, 0.8),
        .intensity = if (envFlag("AGATE_Q0_NO_SUN")) 0.0 else 2.6,
    }) catch @panic("q0: sun failed");
    scene.shadows.enabled = !envFlag("AGATE_Q0_NO_SHADOWS");

    // Analytic physical sky (opt-in, atmosphere consistency arc):
    // AGATE_Q0_PHYS_SKY=1 derives a consistent sun color/intensity,
    // hemispheric ambient and fog tint from elevation + turbidity instead
    // of the hand-tuned manual values above (which stay the default).
    // AGATE_Q0_SKY_TURBIDITY sets haze; AGATE_Q0_SUN_ELEV/AGATE_Q0_SUN_AZIM
    // (degrees) optionally move the sun for time-of-day iteration.
    if (envFlag("AGATE_Q0_PHYS_SKY")) {
        scene.phys_sky.enabled = true;
        scene.phys_sky.dome_enabled = true;
        scene.phys_sky.turbidity = envF32("AGATE_Q0_SKY_TURBIDITY", 2.6);
        if (std.c.getenv("AGATE_Q0_SUN_ELEV") != null) scene.phys_sky.sun_elevation_deg = envF32("AGATE_Q0_SUN_ELEV", 49.0);
        if (std.c.getenv("AGATE_Q0_SUN_AZIM") != null) scene.phys_sky.sun_azimuth_deg = envF32("AGATE_Q0_SUN_AZIM", 0.0);
        // Sky v2 (opt-in inside the analytic dome): AGATE_Q0_CLOUDS=1 adds
        // the 2D FBM dome layer (coverage/scale/wind tunable); AGATE_Q0_NIGHT=1
        // forces full night (stars + moon) for day iteration — otherwise
        // night fades in automatically once the sun drops below ~+2 deg.
        if (envFlag("AGATE_Q0_CLOUDS")) {
            scene.phys_sky.clouds.enabled = true;
            scene.phys_sky.clouds.coverage = envF32("AGATE_Q0_CLOUD_COVERAGE", 0.45);
            scene.phys_sky.clouds.scale = envF32("AGATE_Q0_CLOUD_SCALE", 3.0);
            scene.phys_sky.clouds.wind_speed = envF32("AGATE_Q0_WIND", 0.02);
            scene.phys_sky.clouds.wind_dir_deg = envF32("AGATE_Q0_WIND_DIR", 45.0);
        }
        if (envFlag("AGATE_Q0_NIGHT")) scene.phys_sky.night.enabled = true;
    }

    // Spot PCSS calibration light: a small shadow-casting spot over the
    // content cluster. source_radius > 0 runs the spot PCSS path (blocker
    // search + variable penumbra); AGATE_Q0_NO_SPOT_PCSS=1 falls back to
    // the legacy fixed PCF bit-identically (0.0 disables).
    const spot_pcss_off = envFlag("AGATE_Q0_NO_SPOT_PCSS");
    spot_light = scene.createSpotLight("q0_key_spot", .{
        .position = z.Vec3.new(0.6, 3.6, 0.6),
        .direction = z.Vec3.new(-0.2, -1.0, -0.4),
        .color = z.Color3.new(1.0, 0.85, 0.7),
        .intensity = if (envFlag("AGATE_Q0_NO_SPOT")) 0.0 else 6.0,
        .range = 12.0,
        .inner_angle_deg = 18.0,
        .outer_angle_deg = 32.0,
        .shadow = .{
            .cast_shadows = !envFlag("AGATE_Q0_NO_SHADOWS"),
            .source_radius = if (spot_pcss_off) 0.0 else 0.02,
        },
    }) catch @panic("q0: spot failed");

    // Point PCSS calibration light: a small shadow-casting point near the
    // content cluster. source_radius > 0 runs the border-correct point PCSS
    // path (world-space disk taps, per-tap face projection + cross-face
    // normalization, then blocker search + variable penumbra);
    // AGATE_Q0_NO_POINT_PCSS=1 falls back to the legacy
    // fixed PCF bit-identically (0.0 disables).
    const point_pcss_off = envFlag("AGATE_Q0_NO_POINT_PCSS");
    point_light = scene.createPointLight("q0_key_point", .{
        .position = z.Vec3.new(-1.2, 2.6, 1.2),
        .color = z.Color3.new(1.0, 0.8, 0.6),
        .intensity = if (envFlag("AGATE_Q0_NO_POINT")) 0.0 else 8.0,
        .range = 10.0,
        .shadow = .{
            .cast_shadows = !envFlag("AGATE_Q0_NO_SHADOWS"),
            .source_radius = if (point_pcss_off) 0.0 else 0.02,
        },
    }) catch @panic("q0: point failed");

    // Warm ceiling area panel + emissive quad (bloom picks the emitter up).
    area_light = scene.addAreaLight("area_panel", .{
        .center = z.Vec3.new(0.0, 3.85, -0.8),
        .right = z.Vec3.new(1.4, 0.0, 0.0),
        .up = z.Vec3.new(0.0, 0.0, 1.0),
        .color = z.Color3.new(1.0, 0.72, 0.45),
        .intensity = 9.0,
        .is_enabled = !envFlag("AGATE_Q0_NO_AREA"),
    }) catch null;
    const panel = z.MeshBuilder.createPlane(&scene, "area_panel_quad", .{ .width = 2.8, .height = 2.0 }) catch @panic("q0: panel failed");
    panel.position = z.Vec3.new(0.0, 3.9, -0.8);
    panel.rotation = z.Vec3.new(90.0, 0.0, 0.0);
    panel.cast_shadows = false;
    const panel_mat = pbr("area_panel_pbr", .{ 1.0, 0.8, 0.55 }, 0.0, 0.9);
    panel_mat.emissive_color = z.Color3.new(2.4, 1.4, 0.6);
    panel_mat.double_sided = true;
    panel.setPBRMaterial(panel_mat);

    // Room shell: floor + back/left/right walls + ceiling (open front).
    const floor = z.MeshBuilder.createGround(&scene, "floor", .{ .width = 6.0, .height = 6.0, .subdivisions = 8 }) catch @panic("q0: floor failed");
    const checker = z.Texture.createCheckerboard(allocator(), 512, 512, 32, .{ 26, 26, 30, 255 }, .{ 208, 203, 196, 255 }) catch @panic("q0: checker failed");
    const floor_mat = pbr("floor_pbr", .{ 1, 1, 1 }, 0.0, 0.55);
    floor_mat.albedo_texture = checker;
    floor_mat.roughness = 0.5;
    floor.setPBRMaterial(floor_mat);

    const back = wall("wall_back", 6.0, 4.0);
    back.position = z.Vec3.new(0.0, 2.0, -3.0);
    back.setPBRMaterial(pbr("wall_back_pbr", .{ 0.62, 0.64, 0.66 }, 0.0, 0.85));
    const left = wall("wall_left", 6.0, 4.0);
    left.position = z.Vec3.new(-3.0, 2.0, 0.0);
    left.rotation.y = 90.0;
    left.setPBRMaterial(pbr("wall_left_pbr", .{ 0.72, 0.36, 0.30 }, 0.0, 0.85));
    const right = wall("wall_right", 6.0, 4.0);
    right.position = z.Vec3.new(3.0, 2.0, 0.0);
    right.rotation.y = -90.0;
    right.setPBRMaterial(pbr("wall_right_pbr", .{ 0.34, 0.40, 0.55 }, 0.0, 0.85));
    const ceiling = wall("ceiling", 6.0, 6.0);
    ceiling.position = z.Vec3.new(0.0, 4.0, 0.0);
    ceiling.rotation.x = -90.0;
    ceiling.cast_shadows = true;
    const ceiling_mat = pbr("ceiling_pbr", .{ 0.9, 0.88, 0.85 }, 0.0, 0.9);
    ceiling_mat.double_sided = true;
    ceiling.setPBRMaterial(ceiling_mat);

    // Content: tall red box, white sphere, mirror sphere (SSR showpiece).
    const tall = z.MeshBuilder.createBox(&scene, "tall_box", .{ .width = 1.5, .height = 2.4, .depth = 1.5 }) catch @panic("q0: box failed");
    tall.position = z.Vec3.new(-1.7, 1.2, -1.3);
    tall.setPBRMaterial(pbr("tall_pbr", .{ 0.78, 0.22, 0.18 }, 0.0, 0.7));
    const ball = z.MeshBuilder.createSphere(&scene, "white_sphere", .{ .diameter = 1.6, .segments = 48 }) catch @panic("q0: sphere failed");
    ball.position = z.Vec3.new(1.5, 0.8, -1.1);
    ball.setPBRMaterial(pbr("ball_pbr", .{ 0.95, 0.93, 0.9 }, 0.0, 0.45));
    const mirror = z.MeshBuilder.createSphere(&scene, "mirror_sphere", .{ .diameter = 1.1, .segments = 48 }) catch @panic("q0: mirror failed");
    mirror.position = z.Vec3.new(0.1, 0.55, 1.0);
    const mirror_mat = pbr("mirror_pbr", .{ 0.98, 0.98, 0.98 }, 1.0, 0.04);
    mirror.setPBRMaterial(mirror_mat);

    // Dust motes in the sun shaft (TAA reactive-marker live path, C.3 v2):
    // a slow alpha-blended CPU emitter whose billboards write NO depth and
    // carry NO velocity, so their pixels stamp the velocity-pass reactive
    // marker when taa_reactive is on. AGATE_Q0_NO_PARTICLES=1 isolates the
    // particle-free baseline bit-identically (no draws, no markers).
    if (!envFlag("AGATE_Q0_NO_PARTICLES")) {
        dust_system = scene.createParticleSystem("q0_dust", 256) catch @panic("q0: dust failed");
        const dust = dust_system.?;
        dust.emitter_position = z.Vec3.new(0.0, 1.8, 0.2);
        dust.emitter_box_min = z.Vec3.new(-1.4, -0.9, -1.2);
        dust.emitter_box_max = z.Vec3.new(1.4, 0.9, 1.2);
        dust.emit_rate = 24.0;
        dust.lifetime_min = 2.0;
        dust.lifetime_max = 3.5;
        dust.speed_min = 0.05;
        dust.speed_max = 0.22;
        dust.size_start = 0.06;
        dust.size_end = 0.02;
        dust.color_start = z.Color4.new(1.0, 0.9, 0.75, 0.35);
        dust.color_end = z.Color4.new(1.0, 0.85, 0.65, 0.0);
        dust.blend_mode = .alpha_blend;
        dust.start();
    }

    // Tinted glass pane (TAA reactive-marker glass half, C.3 v2): a blended
    // PBR quad standing in the open-front view axis, so the orbiting camera
    // sweeps it across the background every frame — its pixels carry the
    // background depth+velocity with no signal of their own, and only the
    // velocity-pass glass marker can shed their history. AGATE_Q0_NO_GLASS=1
    // isolates the glass-free baseline bit-identically (no draw, and the
    // marker loop over the empty transparent queue issues nothing).
    if (!envFlag("AGATE_Q0_NO_GLASS")) {
        const pane = z.MeshBuilder.createPlane(&scene, "q0_glass_pane", .{ .width = 1.4, .height = 2.0 }) catch @panic("q0: glass failed");
        glass_pane = pane;
        pane.position = z.Vec3.new(-0.5, 1.5, 0.6);
        pane.rotation = z.Vec3.new(0.0, -18.0, 0.0);
        pane.cast_shadows = false;
        const glass_mat = pbr("q0_glass_pbr", .{ 0.65, 0.85, 0.95 }, 0.0, 0.08);
        glass_mat.alpha_mode = .blend;
        glass_mat.alpha = 0.35;
        glass_mat.double_sided = true;
        pane.setPBRMaterial(glass_mat);
    }

    // Fill-bound by design: the heaviest legal per-pixel chain.
    scene.post_process.enabled = true;
    scene.post_process.exposure = 1.1;
    scene.post_process.tonemapping = .agx;
    scene.post_process.bloom.enabled = !envFlag("AGATE_Q0_NO_BLOOM");
    scene.post_process.bloom.threshold = 1.0;
    scene.post_process.bloom.intensity = 0.55;
    scene.post_process.bloom.mips = 7;
    scene.post_process.ssr.enabled = !envFlag("AGATE_Q0_NO_SSR");
    scene.post_process.ssr.intensity = 0.6;
    scene.post_process.ssr.steps = 32;
    // SSR v2 half-res pass (opt-in): AGATE_Q0_SSR_PASS=1 runs the
    // dedicated half-res trace + bilateral denoise (the composite
    // reconstructs full res with a depth-aware 2x2 gather) instead of the
    // inline fullscreen raymarch (same steps); default stays v1 inline.
    // The SSR A/B (AGATE_Q0_AB=ssr) owns the mode per block, so it starts
    // at the arm-A baseline here.
    if (envFlag("AGATE_Q0_SSR_PASS")) scene.post_process.ssr.mode = .half_res_pass;
    if (envFlag("AGATE_Q0_NO_SSR")) scene.post_process.ssr.enabled = false;
    if (abActive(.ssr)) scene.post_process.ssr.mode = .inline_;
    scene.post_process.depth_pyramid_enabled = true;
    scene.post_process.dof.enabled = !envFlag("AGATE_Q0_NO_DOF");
    scene.post_process.dof.focus_distance = 4.6;
    scene.post_process.dof.focus_range = 2.2;
    scene.post_process.dof.max_blur = 5.0;
    // Lighting isolation: shafts off leaves the actual sunlit opening
    // footprint intact; DoF off + a still camera make shadow A/Bs readable.
    scene.post_process.shafts.enabled = !envFlag("AGATE_Q0_NO_SHAFTS");
    scene.post_process.shafts.intensity = 0.35;
    scene.post_process.shafts.steps = 32;
    scene.post_process.local_tonemapping.enabled = !envFlag("AGATE_Q0_NO_LOCAL_TONEMAP");
    // SSGI v1 color bleed: the calibration scene carries it by default —
    // AGATE_Q0_NO_SSGI=1 isolates the pre-SSGI baseline.
    scene.post_process.ssgi.enabled = !envFlag("AGATE_Q0_NO_SSGI");
    scene.post_process.ssgi.intensity = 0.55;
    scene.post_process.ssgi.radius = 1.6;
    scene.post_process.ssgi.steps = 16;
    // SSGI v2 half-res pass (opt-in): AGATE_Q0_SSGI_PASS=1 runs the
    // dedicated gather + bilateral denoise instead of the inline
    // composite gather (same steps); default stays v1 inline.
    if (envFlag("AGATE_Q0_SSGI_PASS")) scene.post_process.ssgi.mode = .half_res_pass;
    // SSGI v2.1 temporal accumulation (opt-in): AGATE_Q0_SSGI_TEMPORAL=1
    // enables history accumulation inside the half-res pass; requires
    // AGATE_Q0_SSGI_PASS=1 (inline mode warns once and ignores it).
    if (envFlag("AGATE_Q0_SSGI_TEMPORAL")) {
        if (scene.post_process.ssgi.mode != .half_res_pass) {
            std.debug.print("q0-interior: AGATE_Q0_SSGI_TEMPORAL=1 needs AGATE_Q0_SSGI_PASS=1; temporal ignored\n", .{});
        }
        scene.post_process.ssgi.temporal.enabled = true;
    }
    if (std.c.getenv("AGATE_Q0_AO_MODE")) |mode| {
        if (std.mem.eql(u8, std.mem.span(mode), "horizon")) scene.ssao.mode = .horizon;
    }
    if (envFlag("AGATE_Q0_AO_TEMPORAL")) {
        scene.ssao.temporal.enabled = true;
    }
    if (envFlag("AGATE_Q0_NO_AO")) scene.ssao.enabled = false;
    ao_temporal_requested = scene.ssao.enabled and scene.ssao.temporal.enabled;
    std.debug.print("q0-interior: ssao mode={s} temporal={} radius={d:.2} samples={}\n", .{
        @tagName(scene.ssao.mode), scene.ssao.temporal.enabled, scene.ssao.radius, scene.ssao.sample_count,
    });
    // Without TAA, AO history exercises camera-only reprojection.
    scene.post_process.taa.enabled = !envFlag("AGATE_Q0_NO_TAA");
    // TAA reactive treatment for velocity-less content (particles + glass
    // via the v2 velocity-alpha marker, v1 heuristic underneath for the
    // documented gaps): on in the calibration scene —
    // AGATE_Q0_NO_REACTIVE=1 isolates the pre-reactive baseline.
    scene.post_process.taa.reactive.enabled = !envFlag("AGATE_Q0_NO_REACTIVE");
    scene.post_process.fog.enabled = false;
    // Single-scale runs keep the env selection via overrideScale; full
    // 3-phase A/B runs start at the bench schedule's phase 0 (which a
    // reversed order sets to 0.66 — the env default must not clobber it).
    if (frame_limit < bench.scales.len * bench.phase_frames) bench.overrideScale(render_scale);
    scene.post_process.render_scale = bench.currentScale();
}

// ---- sokol app --------------------------------------------------------------

export fn init() callconv(.c) void {
    z.gpu_thread.markContextThread();
    sokol.time.setup();
    sg.setup(.{
        .environment = sokol.glue.environment(),
        .allocator = .{ .alloc_fn = sokolAlloc, .free_fn = sokolFree },
        .logger = .{ .func = qLogger },
        .pipeline_pool_size = 256,
        .shader_pool_size = 128,
        .view_pool_size = 256,
    });
    timing.setEnabled(envFlag("AGATE_Q0_TIMINGS") or envFlag("AGATE_GPU_TIMINGS"));
    camera = z.ArcRotateCamera.init("q0_cam", .{
        .alpha = envF32("AGATE_Q0_CAMERA_ALPHA", -std.math.pi / 2.0),
        .beta = std.math.pi / 2.75,
        .radius = 5.4,
        .target = z.Vec3.new(0.0, 1.2, -0.4),
    });
    std.debug.print("q0-interior: setup backend={s}\n", .{@tagName(sg.queryBackend())});
    scene = z.Scene.init(allocator());
    setupScene();
    captureFeatureBaseline();
    if (panel_enabled) {
        ui_canvas = scene.createUI() catch |err| blk: {
            std.debug.print("q0-feature: HUD unavailable ({s}); hotkeys still work\n", .{@errorName(err)});
            panel_enabled = false;
            break :blk null;
        };
    }
    applyDisableList();
    updateWindowTitle();
    printFeatureState();
    printFeatureHelp();
    if (ab_mode and ab_matrix) beginAbPair(0);
}

export fn frame() callconv(.c) void {
    frames += 1;
    const f = frames;
    const gate = frame_limit > 0;
    defer if (gate and f >= frame_limit) sapp.quit();
    if (gate) {
        if (ab_mode) {
            if (ab_matrix and ab_pair_frame >= ab_pair_limit) {
                finishAbPair();
                if (ab_matrix_idx + 1 >= ab_matrix_pairs.len) {
                    sapp.quit();
                    return;
                }
                beginAbPair(ab_matrix_idx + 1);
            }
            if (ab_matrix) ab_pair_frame += 1;
            const block_f = if (ab_matrix) ab_pair_frame else f;
            const block_limit = if (ab_matrix) ab_pair_limit else frame_limit;
            ab_record_frame = block_f;
            if (ab.beginBlock(block_f, block_limit)) |ev| {
                _ = ab_post.beginBlock(block_f, block_limit);
                _ = ab_main.beginBlock(block_f, block_limit);
                applyAbArm(ab_pair, ev.arm);
                if (ev.index > 0) scene.resetTaa();
                // draws = previous frame's draw calls (stats reset at the
                // next staged begin): the delta is the effect's draw/pass
                // footprint, not a timing claim.
                std.debug.print("q0-interior ab: pair={s} block={} arm={s} ssgi={} ssao={} bloom={} taa={} fxaa={} fog={} shadows={} ssr={s} draws={}\n", .{
                    @tagName(ab_pair),
                    ev.index,
                    @tagName(ev.arm),
                    scene.post_process.ssgi.enabled,
                    scene.ssao.enabled,
                    scene.post_process.bloom.enabled,
                    scene.post_process.taa.enabled,
                    scene.post_process.fxaa.enabled,
                    scene.post_process.fog.enabled,
                    scene.shadows.enabled,
                    @tagName(scene.post_process.ssr.mode),
                    scene.stats.draw_calls,
                });
            }
        } else {
            if (bench.beginPhase(f, frame_limit)) |finished| bench_mod.Bench.printPhase(finished);
            if (bench.phase_started) {
                scene.post_process.render_scale = bench.currentScale();
                scene.resetTaa();
            }
        }
    }
    // Deterministic motion under the gate: same jitter/animation frame for
    // cross-run compare; slow drift keeps TAA/SSR/DoF genuinely working.
    const dt: f32 = if (gate) 1.0 / 60.0 else @floatCast(sapp.frameDuration());
    if (!still_camera) {
        camera.alpha += 0.045 * dt;
        camera.beta = std.math.pi / 2.75 + @sin(@as(f32, @floatFromInt(f)) * 0.008) * 0.12;
    }
    scene.active_camera = .{ .arc_rotate = camera };

    // Sky clock for cloud wind advection (deterministic under the gate:
    // fixed dt accumulates identically every run; live mode uses wall dt).
    if (scene.phys_sky.enabled and (scene.phys_sky.clouds.enabled or scene.phys_sky.night.enabled)) {
        scene.phys_sky.time_sec += dt;
    }

    scene.update(dt) catch |err| fail("update: {s}", .{@errorName(err)});
    const w: i32 = @max(2, sapp.width());
    const h: i32 = @max(2, sapp.height());
    scene.publishFrameSnapshot(@as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h)), w, h);
    drawFeaturePanel();
    if (!scene.buildPreparedFrame()) {
        fail("producer claim unavailable at frame {}", .{f});
        return;
    }
    const claim = scene.beginStagedPrepare() orelse {
        fail("staged claim unavailable at frame {}", .{f});
        return;
    };
    scene.finishStagedPrepare(claim);
    scene.render();
    // GPU sample poll must live HERE: render writes stats.gpu_frame_* at
    // its end and the next staged begin resets `stats`.
    if (timing.isEnabled()) {
        if (ab_mode) {
            ab.recordSample(ab_record_frame, scene.stats.gpu_frame_ms, scene.stats.gpu_frame_submit);
            ab_post.recordSample(ab_record_frame, scene.stats.gpu_post_ms, scene.stats.gpu_post_submit);
            ab_main.recordSample(ab_record_frame, scene.stats.gpu_main_ms, scene.stats.gpu_main_submit);
        } else bench.recordSample(scene.stats.gpu_frame_ms, scene.stats.gpu_frame_submit);
    }
    // Every rendered frame, not just printed boundaries: a window-manager
    // resize must invalidate phase medians as well as interleaved blocks.
    if (ab_mode) {
        ab.noteTargetSize(scene.postfx.renderWidth(), scene.postfx.renderHeight());
        ab_post.noteTargetSize(scene.postfx.renderWidth(), scene.postfx.renderHeight());
        ab_main.noteTargetSize(scene.postfx.renderWidth(), scene.postfx.renderHeight());
    } else {
        bench.noteTargetSize(scene.postfx.renderWidth(), scene.postfx.renderHeight());
    }
    if (f == 2 or bench.phase_started or ab.block_started) {
        std.debug.print("q0-interior: render target {}x{} scale={d:.2} swapchain {}x{}\n", .{
            scene.postfx.renderWidth(),      scene.postfx.renderHeight(),
            scene.post_process.render_scale, sapp.width(),
            sapp.height(),
        });
    }
    if (f == 30) {
        if (scene.stats.draw_calls == 0) fail("no draw calls at frame 30", .{});
        const ao_sized = scene.postfx.ssao_pass.temporalSized();
        const ao_draws = scene.postfx.ssao_pass.temporalDrawCount();
        std.debug.print("q0-interior: frame30 draws={} post_draws={} tris={} ao_temp_sized={} ao_temp_draws={}\n", .{ scene.stats.draw_calls, scene.stats.post_draw_calls, scene.stats.triangles, ao_sized, ao_draws });
        if (ao_temporal_requested and (ao_draws == 0 or !ao_sized)) fail("AO temporal requested but never executed (draws={} sized={})", .{ ao_draws, ao_sized });
        ssrProof();
    }
}

export fn cleanup() callconv(.c) void {
    if (timing.isEnabled()) {
        if (ab_mode) {
            if (ab_matrix) {
                finishAbPair();
            } else {
                std.debug.print("q0-interior ab pair={s}\n", .{@tagName(ab_pair)});
                printAbResult("FRAME", &ab, ab_pair);
                printAbResult("POST", &ab_post, ab_pair);
                printAbResult("MAIN", &ab_main, ab_pair);
            }
        } else bench_mod.Bench.printPhase(bench.report(bench.index));
    }
    scene.deinit();
    sg.shutdown();
    if (sokol_live_allocations != 0) fail("sokol allocations after shutdown: {}", .{sokol_live_allocations});
    if (log_errors != 0) fail("sokol log errors: {}", .{log_errors});
    if (!builtin.cpu.arch.isWasm() and gpa.deinit() == .leak) fail("scene allocator leaked", .{});
    std.debug.print("q0-interior: teardown log_errors={} sokol_live_allocations={}\n", .{ log_errors, sokol_live_allocations });
    std.debug.print("q0-interior: verdict={s} failures={} frames={}\n", .{ if (failures == 0) "PASS" else "FAIL", failures, frames });
    if (failures != 0 and !builtin.cpu.arch.isWasm()) std.process.exit(1);
}

export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    bench.noteEvent(ev.*);
    ab.noteEvent(ev.*);
    ab_post.noteEvent(ev.*);
    ab_main.noteEvent(ev.*);
    if (ev.*.type == .KEY_DOWN and !ev.*.key_repeat) {
        if (ev.*.key_code == .ESCAPE) {
            sapp.quit();
            return;
        }
        handleFeatureKey(ev.*.key_code);
    }
}

pub fn startApp() void {
    frame_limit = envU32("AGATE_Q0_FRAMES", 0);
    still_camera = envFlag("AGATE_Q0_STILL");
    render_scale = envF32("AGATE_Q0_RENDER_SCALE", 1.0);
    bench.phase_frames = @max(30, envU32("AGATE_Q0_PHASE_FRAMES", 180));
    if (envFlag("AGATE_Q0_PHASES_REVERSED")) bench.scales = .{ 0.66, 0.75, 1.0 };
    // Opt-in interleaved A/B: AGATE_Q0_AB=<pair> alternates the scene
    // default (arm A) with the reduced/mode-B arm in drift-balanced blocks
    // (default 2 settle blocks + 8 measured blocks x 60 frames = 600
    // frames when AGATE_Q0_FRAMES is unset). Pairs: ssgi, ssao, bloom,
    // fxaa, taa, fog, shadows, pcss, dof, shafts, local_tonemap, ssr_off,
    // ssr. AGATE_Q0_AB=matrix runs every pair in one process/window
    // (the only comparable cross-pair ranking; per-pair A/B stays
    // interleaved). AGATE_Q0_AB_BLOCK overrides the block size (clamped to
    // the bench scratch). The phase bench is bypassed in this mode;
    // everything else keeps its meaning.
    if (std.c.getenv("AGATE_Q0_AB")) |sel| {
        const raw = std.mem.span(sel);
        if (std.mem.eql(u8, raw, "matrix")) {
            ab_mode = true;
            ab_matrix = true;
        } else if (AbPair.parse(raw)) |pair| {
            ab_mode = true;
            ab_pair = pair;
        } else {
            std.debug.print("q0-interior: unknown AGATE_Q0_AB={s}; A/B disabled\n", .{raw});
        }
    }
    if (ab_mode) {
        ab_block_frames = @min(@max(envU32("AGATE_Q0_AB_BLOCK", 60), 16), bench_mod.ab_max_block_frames);
        ab.block_frames = ab_block_frames;
        ab_post.block_frames = ab_block_frames;
        ab_main.block_frames = ab_block_frames;
        if (ab_matrix) {
            ab_pair_limit = ab_block_frames * (8 + ab.settle_blocks);
            if (frame_limit == 0) frame_limit = ab_pair_limit * @as(u32, @intCast(ab_matrix_pairs.len));
        } else if (frame_limit == 0) {
            frame_limit = ab_block_frames * (8 + ab.settle_blocks);
        }
    }
    render_scale = @min(@max(render_scale, z.postprocess.min_render_scale), 1.0);
    // Interactive q0 starts with the feature HUD; finite/gate runs stay
    // panel-free so benchmarks and screenshots keep their old frame shape.
    panel_enabled = frame_limit == 0;
    if (envFlag("AGATE_Q0_PANEL")) panel_enabled = true;
    if (envFlag("AGATE_Q0_NO_PANEL")) panel_enabled = false;
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "Agate Q0 calibration interior (fill-bound)",
        // Small enough to fit any desktop work area: a window the WM has
        // to resize mid-run poisons phase medians (target-size mixing).
        .width = 720,
        .height = 540,
        .sample_count = 1,
        .logger = .{ .func = qLogger },
    });
}

pub fn main() void {
    startApp();
}
