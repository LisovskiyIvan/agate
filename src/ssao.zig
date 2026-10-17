const std = @import("std");

/// Ambient occlusion execution mode. `hemisphere` is the historical
/// Fermat-spiral kernel gather (default, bit-identical); `horizon` is the
/// opt-in slice-based horizon search (GTAO class): the same depth-derived
/// normal, radius, bias and power knobs, but occlusion comes from
/// per-slice horizon maxima (angular coverage) instead of per-sample
/// counting, which tightens corners and reduces halo at the same tap
/// budget. An enum like ssgi.mode: no clamp row, survives clamped()
/// untouched (the enabled flag still gates both lanes).
pub const AoMode = enum(u8) {
    hemisphere = 0,
    horizon = 1,
};

/// Temporal filtering for the SSAO raw signal (opt-in). Default OFF: the
/// pass runs gather -> bilateral upsample exactly as before
/// (bit-identical). Meaningful for both modes; the history holds the RAW
/// half-res AO scalar plus the current linear depth for the depth gate.
pub const AoTemporalOptions = struct {
    enabled: bool = false,
    /// History weight for the temporal blend (1 = full history). 0.85 is
    /// the conservative default (less trailing than the 0.9 TAA/SSGI
    /// compromise on the higher-frequency AO signal); the documented
    /// sweet spot is [0.8, 0.9]. Clamped to [0, 1].
    blend: f32 = AO_TEMPORAL_BLEND_DEFAULT,
};

/// Screen-space ambient occlusion pass settings. Lives flat on `Scene.ssao`
/// (tooling reads/writes it directly); consumed by the SSAO + blur passes.
pub const SSAOOptions = struct {
    enabled: bool = true,
    radius: f32 = 0.5,
    bias: f32 = 0.035,
    intensity: f32 = 1.1,
    power: f32 = 1.2,
    sample_count: u32 = 24,
    mode: AoMode = .hemisphere,
    temporal: AoTemporalOptions = .{},

    pub fn clamped(self: SSAOOptions) SSAOOptions {
        var out = self;
        out.radius = @max(self.radius, 0.0);
        out.bias = @max(self.bias, 0.0);
        out.intensity = @max(self.intensity, 0.0);
        out.power = @max(self.power, 0.0);
        out.sample_count = std.math.clamp(self.sample_count, 4, 32);
        out.temporal.blend = std.math.clamp(self.temporal.blend, 0.0, 1.0);
        return out;
    }
};

/// Temporal activity: the master enable plus the opt-in flag. MSAA
/// suppression clears ssao.enabled upstream, so temporal joins the
/// existing suppression with no extra plumbing.
pub fn aoTemporalActive(config: SSAOOptions) bool {
    return config.enabled and config.temporal.enabled;
}

/// History blend weight in [0, 1] after clamped().
pub fn aoTemporalBlend(config: SSAOOptions) f32 {
    return std.math.clamp(config.temporal.blend, 0.0, 1.0);
}

/// Half-res raw/temporal target size for a full EFFECTIVE (render-scale)
/// base size: ceil-half per axis, mirroring the SSGI sizing rule. Pure
/// (no GPU), so the pass and the tests share one rule; odd sizes round
/// up, never to zero from a positive base.
pub const AoHalfSize = struct { w: i32, h: i32 };

pub fn aoHalfSize(base_w: i32, base_h: i32) AoHalfSize {
    if (base_w <= 0 or base_h <= 0) return .{ .w = base_w, .h = base_h };
    return .{
        .w = @divTrunc(base_w, 2) + @mod(base_w, 2),
        .h = @divTrunc(base_h, 2) + @mod(base_h, 2),
    };
}

/// Reset conditions mirror taaShouldReset/ssgiTemporalShouldReset:
/// first frame, toggle-on, resize/recreate, camera cut, explicit reset.
/// The camera-cut bit rides the shared taa_camera_cut snapshot flag.
pub const AoReset = struct {
    first_frame: bool = false,
    toggled_on: bool = false,
    resized: bool = false,
    camera_cut: bool = false,
    explicit_reset: bool = false,
};

pub fn aoTemporalShouldReset(r: AoReset) bool {
    return r.first_frame or r.toggled_on or r.resized or r.camera_cut or r.explicit_reset;
}

/// History ping-pong slots driven by SSAOPass.temp_frame (same parity
/// shape as the TAA/SSGI helpers so tests pin the alternation).
pub fn aoTemporalReadIndex(frame: u64) u8 {
    return @intCast(frame & 1);
}

pub fn aoTemporalWriteIndex(frame: u64) u8 {
    return 1 - aoTemporalReadIndex(frame);
}

/// History UV from a velocity tap: the velocity buffer stores
/// uv_curr - uv_prev (see velocity.glsl), so prev = uv - vel. Pure.
pub fn aoTemporalHistoryUv(uv: [2]f32, velocity_xy: [2]f32) [2]f32 {
    return .{ uv[0] - velocity_xy[0], uv[1] - velocity_xy[1] };
}

// --- AO temporal rejection gates (opt-in history path) ---
//
// The temporal stage sits between the half-res gather and the bilateral
// upsample: gather -> temporal -> denoise. History holds the RAW AO
// scalar (R), the oct-encoded CURRENT world normal (G,B) for the normal
// gate, and the current linear depth (A) for priming. The depth gate
// compares history against the EXPECTED previous-camera depth (current
// world pos projected with prevVP), never directly against the current
// depth — a translating camera must not reject static-geometry history.

/// Variance-clip tightness: history must survive mean +/- gamma * stddev
/// of the current 3x3 raw-AO neighborhood (TAA uses 1.25 on radiance;
/// SSGI temporal uses 1.0 on bleed; AO matches SSGI at 1.0).
pub const AO_TEMPORAL_GAMMA: f32 = 1.0;

/// Normal-consistency gate: history survives only when the current surface
/// normal and the normal reconstructed at the reprojected UV agree to this
/// cosine (about 25 degrees). Below it the pixel reprojected across a
/// geometric edge and the fresh gather wins.
pub const AO_TEMPORAL_NORMAL_MIN_DOT: f32 = 0.9;

/// History blend default (conservative vs the 0.9 TAA/SSGI compromise:
/// the AO signal carries higher spatial frequency, so trailing shows
/// sooner).
pub const AO_TEMPORAL_BLEND_DEFAULT: f32 = 0.85;

/// Reactive-marker band over the velocity alpha: particle/glass coverage
/// the velocity pass stamps with exactly 0.25 (see velocity.glsl
/// fs_marker). Local copies of the TAA band (ssao.zig must stay free of
/// the postprocess import cycle; ssao_tests.zig cross-pins the values).
pub const AO_TEMPORAL_MARKER_LO: f32 = 0.15;
pub const AO_TEMPORAL_MARKER_HI: f32 = 0.35;

/// Horizon-search slice count for `horizon` mode (4 slices x 2 sides;
/// steps-per-side derive from sample_count so the tap budget matches the
/// hemisphere path). Shared by the shader and the CPU mirror.
pub const AO_HORIZON_SLICES: u32 = 4;

/// Steps per slice side for a sample_count: total taps ~= sample_count
/// (2 taps per slice-step), clamped to [2, 4] so quality scales with the
/// existing knob instead of adding a new one.
pub fn aoHorizonStepsPerSide(sample_count: u32) u32 {
    const s = sample_count / (2 * AO_HORIZON_SLICES);
    return std.math.clamp(s, 2, 4);
}

/// Temporal gather rotation period (frames): the opt-in history integrates
/// varying sample directions instead of the same IGN angle every frame.
/// Frame 0 (phase 0, and every history reset) is exactly 0.0 so the
/// reset/prime gather matches the unjittered spatial path. Later phases
/// are mode-aware: hemisphere steps evenly by 2pi/PERIOD (full-circle
/// coverage), while horizon steps by (pi/4)/PERIOD — a full 2pi/PERIOD
/// step would only permute the horizon's 8-direction slice star (4 slices
/// spaced pi/4 x 2 sides), gathering the SAME taps every frame, so the
/// horizon micro-rotates WITHIN one slice spacing instead (pi/32 at
/// PERIOD 8: 8 frames x 8 directions = 64 stratified angles). Uploaded
/// via the historical unused extra_params.z lane: 0.0 whenever temporal
/// is off or fail-closed (multi-camera/MSAA/allocation failure), so the
/// default shader arithmetic is bit-identical. Pure.
pub const AO_TEMPORAL_ROTATION_PERIOD: u32 = 8;

/// Gather rotation in radians for a temporal phase under a mode: phase 0
/// (and every multiple of the period) is exactly 0.0 in both modes;
/// otherwise an even multiple of the mode step (hemisphere 2pi/PERIOD,
/// horizon (pi/4)/PERIOD). The caller uploads 0.0 on reset/off/fail-closed
/// and only this value on active non-reset frames; the phase cursor advances
/// only after an actual temporal draw and resets with history (cut,
/// toggle, resize, mode, projection, explicit reset). Pure.
pub fn aoTemporalRotation(phase: u32, mode: AoMode) f32 {
    const m = phase % AO_TEMPORAL_ROTATION_PERIOD;
    if (m == 0) return 0.0;
    const step: f32 = switch (mode) {
        .hemisphere => 2.0 * std.math.pi / @as(f32, @floatFromInt(AO_TEMPORAL_ROTATION_PERIOD)),
        .horizon => (std.math.pi / 4.0) / @as(f32, @floatFromInt(AO_TEMPORAL_ROTATION_PERIOD)),
    };
    return @as(f32, @floatFromInt(m)) * step;
}

/// Reactive-marker classification over the velocity alpha (same band the
/// TAA/SSGI resolves test; non-finite input is never a marker). Pure.
pub fn aoTemporalIsMarker(alpha: f32) bool {
    if (!std.math.isFinite(alpha)) return false;
    return alpha >= AO_TEMPORAL_MARKER_LO and alpha <= AO_TEMPORAL_MARKER_HI;
}

/// Depth rejection gate (linearized depths, meters): history survives only
/// when the surface the previous camera saw at the reprojected UV is
/// within the same bilateral gate the denoise uses (max(0.001,
/// radius * 0.2)) of the EXPECTED depth — the current world position
/// projected with the previous view-projection (see aoReproject).
/// `expected_lin` is that expectation linearized with the CURRENT
/// near/far; `hist_lin` is history alpha. Comparing history to the
/// expectation (not to the current depth) keeps static-geometry history
/// alive under camera translation; silhouette crossings still reject.
/// Pure.
pub fn aoTemporalDepthOk(expected_lin: f32, hist_lin: f32, radius: f32) bool {
    if (!std.math.isFinite(expected_lin) or !std.math.isFinite(hist_lin)) return false;
    return @abs(expected_lin - hist_lin) < @max(0.001, radius * 0.2);
}

/// Normal rejection gate: history survives only when the current normal
/// and the PREVIOUS normal decoded from history (see aoOctDecode) agree
/// within AO_TEMPORAL_NORMAL_MIN_DOT (degenerate zero normals reject:
/// there is no surface to agree on). Pure.
pub fn aoTemporalNormalOk(cur_n: [3]f32, hist_n: [3]f32) bool {
    const dot = cur_n[0] * hist_n[0] + cur_n[1] * hist_n[1] + cur_n[2] * hist_n[2];
    if (!std.math.isFinite(dot)) return false;
    return dot >= AO_TEMPORAL_NORMAL_MIN_DOT;
}

/// Octahedral encode: unit world normal -> [0,1]^2 for the history G,B
/// lanes. Same convention as the AO oct codec (local copy: ssao.zig
/// must stay free of the mesh-pulling import; ssao_tests.zig cross-pins
/// the values). Degenerate input encodes to the (0.5, 0.5) sentinel,
/// which decodes to +Z (see aoOctDecode); the depth gate is the primary
/// guard where that fires. Matches aoOctEncode in ssao_temporal.glsl.
pub fn aoOctEncode(d: [3]f32) [2]f32 {
    const sum: f32 = @abs(d[0]) + @abs(d[1]) + @abs(d[2]);
    const inv: f32 = if (sum > 1e-9) 1.0 / sum else 0.0;
    var p: [2]f32 = .{ d[0] * inv, d[1] * inv };
    if (d[2] < 0.0) {
        const ox = (1.0 - @abs(p[1])) * (if (p[0] >= 0.0) @as(f32, 1.0) else @as(f32, -1.0));
        const oy = (1.0 - @abs(p[0])) * (if (p[1] >= 0.0) @as(f32, 1.0) else @as(f32, -1.0));
        p = .{ ox, oy };
    }
    return .{ p[0] * 0.5 + 0.5, p[1] * 0.5 + 0.5 };
}

/// Octahedral decode: history G,B -> unit world normal. Same convention
/// as aoOctDecode (local copy, round-trip tested). The (0.5,
/// 0.5) sentinel decodes to +Z. Matches aoOctDecode in
/// ssao_temporal.glsl.
pub fn aoOctDecode(o: [2]f32) [3]f32 {
    const p: [2]f32 = .{ o[0] * 2.0 - 1.0, o[1] * 2.0 - 1.0 };
    var d: [3]f32 = .{ p[0], p[1], 1.0 - @abs(p[0]) - @abs(p[1]) };
    if (d[2] < 0.0) {
        const ox = (1.0 - @abs(d[1])) * (if (d[0] >= 0.0) @as(f32, 1.0) else @as(f32, -1.0));
        const oy = (1.0 - @abs(d[0])) * (if (d[1] >= 0.0) @as(f32, 1.0) else @as(f32, -1.0));
        d[0] = ox;
        d[1] = oy;
    }
    const len: f32 = @sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (len < 1e-9) return .{ 0.0, 0.0, 1.0 };
    return .{ d[0] / len, d[1] / len, d[2] / len };
}

/// Previous-clip w epsilon for aoReproject (GLSL mirror: the
/// `prev_clip.w <= 0.0001` early-out in ssao_temporal.glsl).
pub const AO_REPROJECT_W_EPS: f32 = 0.0001;

/// Reproject a world position into the previous frame: prev_clip =
/// prevVP * (x,y,z,1), then the history UV (same NDC->UV mapping as the
/// shader and velocity.glsl) plus the EXPECTED previous raw depth
/// (prev_clip.z / prev_clip.w, for the aoTemporalDepthOk comparison).
/// Returns null when the point sits behind the previous camera
/// (w <= eps): history is unusable there and the caller primes instead.
/// Column-major 4x4 (math.Mat4 layout); dependency-free so ssao.zig
/// stays std-only. Pure.
pub const AoReproject = struct { uv: [2]f32, raw_depth: f32 };

pub fn aoReproject(prev_vp_m16: [16]f32, world: [3]f32) ?AoReproject {
    const m = prev_vp_m16;
    // Column-major: clip = m[0]*x + m[4]*y + m[8]*z + m[12] (rows 0..3).
    const cx = m[0] * world[0] + m[4] * world[1] + m[8] * world[2] + m[12];
    const cy = m[1] * world[0] + m[5] * world[1] + m[9] * world[2] + m[13];
    const cz = m[2] * world[0] + m[6] * world[1] + m[10] * world[2] + m[14];
    const cw = m[3] * world[0] + m[7] * world[1] + m[11] * world[2] + m[15];
    if (cw <= AO_REPROJECT_W_EPS) return null;
    const inv_w = 1.0 / cw;
    return .{
        .uv = .{ cx * inv_w * 0.5 + 0.5, 0.5 - cy * inv_w * 0.5 },
        .raw_depth = cz * inv_w,
    };
}

/// 1D variance bounds over center + 8 raw-AO neighbors (scalar version of
/// the SSGI/TAA 3x3 estimator): [mu - gamma*sigma, mu + gamma*sigma]
/// intersected with the min/max box. Pure.
pub const AoBounds = struct { min: f32, max: f32 };

pub fn aoTemporalVarianceBounds(center: f32, neighbors: [8]f32, gamma: f32) AoBounds {
    var mn = center;
    var mx = center;
    var sum = center;
    var sum_sq = center * center;
    for (neighbors) |n| {
        mn = @min(mn, n);
        mx = @max(mx, n);
        sum += n;
        sum_sq += n * n;
    }
    const g = @max(gamma, 0.0);
    const mu = sum / 9.0;
    const variance = @max(sum_sq / 9.0 - mu * mu, 0.0);
    const sigma = @sqrt(variance);
    var lo = @max(mn, mu - g * sigma);
    var hi = @min(mx, mu + g * sigma);
    if (lo > hi) {
        lo = mn;
        hi = mx;
    }
    return .{ .min = lo, .max = hi };
}

/// History acceptance weight in [0, 1] from the variance-box divergence
/// (same 1/(1+4d^2) falloff as the TAA/SSGI resolves). Pure.
pub fn aoTemporalRejectionWeight(history: f32, box_min: f32, box_max: f32) f32 {
    const lo = @min(box_min, box_max);
    const hi = @max(box_min, box_max);
    const span = @max(hi - lo, 1e-4);
    var dist: f32 = 0.0;
    if (history < lo) {
        dist = (lo - history) / span;
    } else if (history > hi) {
        dist = (history - hi) / span;
    }
    return 1.0 / (1.0 + dist * dist * 4.0);
}

/// Full temporal resolve mirror of the ssao_temporal.glsl main: variance
/// clip (full strength) then mix(current, clamped, blend * accept) where
/// accept folds the variance divergence, the depth/normal gates, and the
/// reactive-marker shed (markers take exactly zero history). Reset frames
/// (history_valid=false) return current. Pure.
pub fn aoTemporalResolve(
    current: f32,
    neighbors: [8]f32,
    history: f32,
    history_valid: bool,
    blend: f32,
    gamma: f32,
    depth_ok: bool,
    normal_ok: bool,
    is_marker: bool,
) f32 {
    if (!history_valid) return current;
    const box = aoTemporalVarianceBounds(current, neighbors, gamma);
    const reject = aoTemporalRejectionWeight(history, box.min, box.max);
    var accept = reject;
    if (!depth_ok or !normal_ok) accept = 0.0;
    if (is_marker) accept = 0.0;
    const b = std.math.clamp(blend, 0.0, 1.0) * accept;
    const lo = @min(box.min, box.max);
    const hi = @max(box.min, box.max);
    const hc = std.math.clamp(history, lo, hi);
    return current + (hc - current) * b;
}

// --- AO geometry mirrors (deterministic golden tests) ---
//
// Both modes share the depth-derived normal, radius, horizon offset
// (0.15 ~ 9 degrees above the tangent plane) and power curve; they differ
// in the blocked-visibility estimator: weighted hemisphere taps versus
// per-direction horizon maxima.

/// Horizon elevation offset above the tangent plane before a tap counts:
/// hemisphere uses a binary gate, horizon search subtracts the offset.
pub const AO_HORIZON_OFFSET: f32 = 0.15;

/// Near-field exclusion (meters): taps closer than this are the shading
/// pixel's own triangle (GLSL mirror: `dist < 0.03` skip).
pub const AO_NEAR_EXCLUSION: f32 = 0.03;

/// Kernel tap layout (CPU mirror of the Fermat-spiral build in
/// passes/ssao_pass.zig SSAOPass.init and the ssao.glsl gather):
/// 32 taps, tap k carries the unit hemisphere direction with
/// z = (k + 0.5) / 32 * 0.88 + 0.12 scaled toward the origin.
pub const AO_KERNEL_SIZE: u32 = 32;

/// Cosine sampling weight of kernel tap k: the z of the NORMALIZED
/// kernel vector (normalizing removes the accelerating scale, leaving
/// the unit-direction cosine). Analytic form of the kernel build, so
/// the CPU mirror and the shader's normalize() agree exactly. Pure.
pub fn aoKernelCos(kernel_idx: u32) f32 {
    const k: f32 = @floatFromInt(kernel_idx);
    return (k + 0.5) / 32.0 * 0.88 + 0.12;
}

/// Stratified tap selection for a reduced budget N (CPU mirror of the
/// ssao.glsl gather index): midpoint stratification
/// idx = ((2*i+1)*32)/(2*N) over the 32-tap kernel. N=32 maps to
/// idx=i (bit-identical to the historical first-N loop); smaller
/// budgets spread across the whole kernel (radius AND angle) instead
/// of clustering on the grazing/short-ray prefix. Integer truncation
/// matches the GLSL integer division exactly. Pure.
pub fn aoKernelIndex(tap: u32, sample_count: u32) u32 {
    const n = std.math.clamp(sample_count, 4, 32);
    const t = @min(tap, n - 1);
    return ((2 * t + 1) * AO_KERNEL_SIZE) / (2 * n);
}

/// Visibility denominator for a budget N: the sum of the sampling
/// weights of the ACTUAL taps the gather executes (never an arbitrary
/// constant). The mean weight is ~0.56 at every budget, so the
/// estimator below keeps finite ambient where the historical 2.8/32
/// divisor (11.43x mean amplification) drove ordinary corners black.
/// Pure.
pub fn aoHemisphereDenom(sample_count: u32) f32 {
    const n = std.math.clamp(sample_count, 4, 32);
    var sum: f32 = 0.0;
    for (0..n) |i| sum += aoKernelCos(aoKernelIndex(@intCast(i), n));
    return sum;
}

/// View-space normal from centered-difference gradients with the
/// scale-relative degeneracy guard shared with
/// postprocess/depth_normal.zig (squared sine of the gradient angle:
/// len_sq > max(scale * 1e-8, 1e-30), never an absolute world-unit
/// cutoff). Returns null when the gradients are degenerate — the
/// caller must return unoccluded (AO 1), NOT invent a camera-facing
/// fallback: the historical normalize(-pos) painted a fictitious
/// occlusion response onto degenerate pixels. Pure.
pub fn aoNormalFromGradients(dx: [3]f32, dy: [3]f32) ?[3]f32 {
    const cx = dx[1] * dy[2] - dx[2] * dy[1];
    const cy = dx[2] * dy[0] - dx[0] * dy[2];
    const cz = dx[0] * dy[1] - dx[1] * dy[0];
    const len_sq = cx * cx + cy * cy + cz * cz;
    const scale = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]) *
        (dy[0] * dy[0] + dy[1] * dy[1] + dy[2] * dy[2]);
    if (len_sq <= @max(scale * 1e-8, 1e-30)) return null;
    const inv = 1.0 / @sqrt(len_sq);
    return .{ cx * inv, cy * inv, cz * inv };
}

/// One hemisphere tap contribution under the cosine-weighted
/// visibility estimator: the tap's SAMPLING weight (aoKernelCos of its
/// stratified index) times smooth linear distance attenuation, gated
/// by the existing local self-rejection (near exclusion, radius) and
/// the horizon positive condition (n_dot above AO_HORIZON_OFFSET).
/// `occluded` is the shader's view-space depth comparison
/// (sample_frag.z >= sample_view.z + bias). The horizon gate stays
/// BINARY here: n_dot is the geometric cosine of the OCCLUDER
/// direction while the weight is the cosine of the SAMPLE direction,
/// and multiplying the two distributions would double-count the
/// cosine. Flat surfaces never pass the depth comparison, so they
/// still resolve to exactly unoccluded. Pure.
pub fn aoHemisphereTap(n_dot: f32, dist: f32, radius: f32, occluded: bool, weight: f32) f32 {
    if (!occluded) return 0.0;
    if (radius <= 0.0) return 0.0;
    if (weight <= 0.0) return 0.0;
    if (dist < AO_NEAR_EXCLUSION or dist >= radius) return 0.0;
    if (!std.math.isFinite(n_dot)) return 0.0;
    if (n_dot - AO_HORIZON_OFFSET <= 0.0) return 0.0;
    return weight * (1.0 - dist / radius);
}

/// Hemisphere resolve: the cosine-weighted blocked fraction
/// (summed taps over aoHemisphereDenom for the same budget), then the
/// power curve. Mirrors the ssao.glsl tail. A perpendicular floor/wall
/// blocks roughly half the cosine-weighted hemisphere, so it keeps
/// ~0.5 visibility instead of going black; only true near-full
/// enclosure darkens deeply. Pure.
pub fn aoHemisphereResolve(occlusion_sum: f32, sample_count: u32, power: f32) f32 {
    const denom = aoHemisphereDenom(sample_count);
    if (denom <= 0.0) return 1.0;
    const norm_occ = std.math.clamp(occlusion_sum / denom, 0.0, 1.0);
    const p = @max(power, 0.0);
    return std.math.clamp(std.math.pow(f32, 1.0 - norm_occ, p), 0.0, 1.0);
}

/// One horizon-slice resolve: the per-side maxima (each the max over its
/// taps of elevation above AO_HORIZON_OFFSET times falloff) average to the
/// slice coverage. Max-filtering per direction is the horizon-search
/// distinction: a single high occluder sets the horizon, additional
/// samples behind it add nothing. Pure.
pub fn aoHorizonSliceResolve(max_side_a: f32, max_side_b: f32) f32 {
    return (std.math.clamp(max_side_a, 0.0, 1.0) + std.math.clamp(max_side_b, 0.0, 1.0)) * 0.5;
}

/// Horizon resolve: average slice coverage over AO_HORIZON_SLICES, then
/// the same power curve as the hemisphere tail. Pure.
pub fn aoHorizonResolve(slices: [AO_HORIZON_SLICES]f32, power: f32) f32 {
    var sum: f32 = 0.0;
    for (slices) |s| sum += std.math.clamp(s, 0.0, 1.0);
    const avg = std.math.clamp(sum / @as(f32, @floatFromInt(AO_HORIZON_SLICES)), 0.0, 1.0);
    const p = @max(power, 0.0);
    return std.math.clamp(std.math.pow(f32, 1.0 - avg, p), 0.0, 1.0);
}
