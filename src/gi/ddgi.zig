//! DDGI v1 spike: probe-grid diffuse GI over a proxy AABB scene.
//!
//! Pure math + options for the opt-in DDGI slice (roadmap Q3 scenario 3,
//! ray source = proxy scene, no BVH/SDF). No GPU calls: the fragment-based
//! update pass (passes/ddgi_pass.zig + shaders/ddgi_update.glsl) and the
//! PBR sample chunk (shaders/common/ddgi_sample.glsl) implement these
//! same formulas in GLSL; this module is the CPU mirror the house tests
//! pin down.
//!
//! Model (v1 limits, see docs/gi.md): static world-space probe grid over
//! a Scene-provided volume. Each probe stores 8x8 outgoing-radiance
//! samples (one texel = one octahedral ray direction) in an RGBA16F
//! atlas. Update marches proxy AABBs per ray: miss -> hemispheric sky
//! ambient, hit -> proxy albedo x direct sun (secondary visibility march
//! vs proxies only, no shadow-map reads) + ambient. Single bounce only.
//! Sampling is one tap per probe along the surface normal, trilinearly
//! blended (diffuse proxy, not a true hemisphere integral).
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const Mesh = @import("../mesh.zig").Mesh;

/// Hard cap on proxy boxes per update (uniform array sizing, shader loop
/// bound). Past the cap extra proxies are dropped in collection order;
/// the count reaching the shader is exact, never silently wrapped.
pub const MAX_PROXIES: usize = 64;
/// Probe tile edge in texels (8x8 rays per probe).
pub const PROBE_TILE: u32 = 8;
/// Tiles per atlas row (fixed: keeps probe->UV addressing branchless).
pub const TILES_PER_ROW: u32 = 8;
/// Atlas width in px (TILES_PER_ROW * PROBE_TILE). Height derives from
/// the probe count: rows = ceil(probes / TILES_PER_ROW).
pub const ATLAS_WIDTH: u32 = TILES_PER_ROW * PROBE_TILE;
/// Hard cap on probes per grid (atlas VRAM + update cost bound).
pub const MAX_PROBES: usize = 256;
/// Per-axis probe count clamp (each axis).
pub const MAX_AXIS_PROBES: u32 = 8;
/// Ray hits closer than this (m) are ignored: self-hit guard for probes
/// resting on/inside proxy surfaces.
pub const SELF_HIT_MIN_T: f32 = 0.05;
/// Degenerate proxy guard: per-axis extent below this is dropped.
pub const MIN_PROXY_EXTENT: f32 = 1e-4;

/// One proxy box: world AABB + linear diffuse albedo.
pub const ProxyBox = struct {
    min: [3]f32,
    max: [3]f32,
    albedo: [3]f32,
};

/// Scene-provided volume + density. Default grid is inert (disabled flag
/// lives in DdgiOptions): volume defaults only shape the q0 calibration.
pub const DdgiOptions = struct {
    enabled: bool = false,
    /// Grid origin (min corner, world).
    origin: [3]f32 = .{ -3.0, 0.0, -3.0 },
    /// Grid extent (world size from origin).
    extent: [3]f32 = .{ 6.0, 3.0, 6.0 },
    /// Probe counts per axis (clamped to [1, MAX_AXIS_PROBES], product
    /// capped at MAX_PROBES by clampedCounts).
    counts: [3]u32 = .{ 5, 3, 5 },
    /// Sample-side multiplier (atlas values stay raw; intensity stays
    /// live-tunable like the SSGI composite scaling).
    intensity: f32 = 1.0,
    /// Probes refreshed per frame (round-robin window); 0 = whole grid.
    probes_per_frame: u32 = 0,

    pub fn clampedIntensity(self: DdgiOptions) f32 {
        return std.math.clamp(self.intensity, 0.0, 4.0);
    }
};

/// Clamp per-axis counts to [1, MAX_AXIS_PROBES], then shrink the largest
/// axis until the product fits MAX_PROBES. Deterministic; pure.
pub fn clampedCounts(counts: [3]u32) [3]u32 {
    var c: [3]u32 = .{
        std.math.clamp(counts[0], 1, MAX_AXIS_PROBES),
        std.math.clamp(counts[1], 1, MAX_AXIS_PROBES),
        std.math.clamp(counts[2], 1, MAX_AXIS_PROBES),
    };
    while (c[0] * c[1] * c[2] > MAX_PROBES) {
        if (c[0] >= c[1] and c[0] >= c[2] and c[0] > 1) {
            c[0] -= 1;
        } else if (c[1] >= c[2] and c[1] > 1) {
            c[1] -= 1;
        } else if (c[2] > 1) {
            c[2] -= 1;
        } else break;
    }
    return c;
}

pub fn probeCount(counts: [3]u32) usize {
    const c = clampedCounts(counts);
    return @as(usize, c[0]) * c[1] * c[2];
}

/// Cell size = extent / counts (counts already clamped by the caller via
/// clampedCounts; zero counts guarded to 1).
pub fn cellSize(extent: [3]f32, counts: [3]u32) [3]f32 {
    const c = clampedCounts(counts);
    return .{
        extent[0] / @max(1.0, @as(f32, @floatFromInt(c[0]))),
        extent[1] / @max(1.0, @as(f32, @floatFromInt(c[1]))),
        extent[2] / @max(1.0, @as(f32, @floatFromInt(c[2]))),
    };
}

/// World position of a probe: cell centers, linear index x-fastest
/// (idx = x + nx*(y + ny*z)). Matches the update shader addressing.
pub fn probePosition(origin: [3]f32, cell: [3]f32, counts: [3]u32, idx: usize) [3]f32 {
    const c = clampedCounts(counts);
    const nx: usize = c[0];
    const ny: usize = c[1];
    const x = idx % nx;
    const y = (idx / nx) % ny;
    const z = idx / (nx * ny);
    return .{
        origin[0] + (@as(f32, @floatFromInt(x)) + 0.5) * cell[0],
        origin[1] + (@as(f32, @floatFromInt(y)) + 0.5) * cell[1],
        origin[2] + (@as(f32, @floatFromInt(z)) + 0.5) * cell[2],
    };
}

/// Atlas pixel size for a probe count (width fixed, rows ceil).
pub fn atlasSize(probes: usize) [2]u32 {
    const rows: u32 = @intCast((probes + TILES_PER_ROW - 1) / TILES_PER_ROW);
    return .{ ATLAS_WIDTH, @max(1, rows) * PROBE_TILE };
}

/// Tile origin (px) of a probe tile. Probes past the grid end are never
/// addressed (update window clamps to the probe count).
pub fn atlasTileOrigin(probe_idx: usize) [2]u32 {
    const tx: u32 = @intCast(@as(u64, probe_idx) % TILES_PER_ROW);
    const ty: u32 = @intCast(@as(u64, probe_idx) / TILES_PER_ROW);
    return .{ tx * PROBE_TILE, ty * PROBE_TILE };
}

/// Octahedral encode: unit dir -> [0,1]^2. Matches ddgi_sample.glsl.
pub fn octEncode(d: [3]f32) [2]f32 {
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

/// Octahedral decode: [0,1]^2 -> unit dir. Matches the update shader.
pub fn octDecode(o: [2]f32) [3]f32 {
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

/// Ray direction of atlas texel (tx,ty in [0,PROBE_TILE)): texel center
/// through octDecode. The update shader derives the same direction from
/// gl_FragCoord by construction (floor to texel, +0.5, /8).
pub fn probeRayDir(tx: u32, ty: u32) [3]f32 {
    const cx = @as(f32, @floatFromInt(@min(tx, PROBE_TILE - 1)));
    const cy = @as(f32, @floatFromInt(@min(ty, PROBE_TILE - 1)));
    const t = @as(f32, @floatFromInt(PROBE_TILE));
    return octDecode(.{ (cx + 0.5) / t, (cy + 0.5) / t });
}

/// Texel-center UV of one atlas texel (NEAREST sampling, no guard bands:
/// bilinear would bleed across tile borders; documented spike limit).
pub fn atlasTexelUv(probe_idx: usize, tx: u32, ty: u32, atlas_w: u32, atlas_h: u32) [2]f32 {
    const to = atlasTileOrigin(probe_idx);
    const fx = @as(f32, @floatFromInt(to[0] + @min(tx, PROBE_TILE - 1))) + 0.5;
    const fy = @as(f32, @floatFromInt(to[1] + @min(ty, PROBE_TILE - 1))) + 0.5;
    return .{ fx / @as(f32, @floatFromInt(atlas_w)), fy / @as(f32, @floatFromInt(atlas_h)) };
}

/// Sample-side UV: direction -> oct -> texel -> atlas UV. CPU mirror of
/// ddgi_sample.glsl ddgiProbeUv.
pub fn sampleUv(probe_idx: usize, d: [3]f32, atlas_w: u32, atlas_h: u32) [2]f32 {
    const o = octEncode(d);
    const t = @as(f32, @floatFromInt(PROBE_TILE));
    const fx = std.math.clamp(@floor(o[0] * t), 0.0, t - 1.0);
    const fy = std.math.clamp(@floor(o[1] * t), 0.0, t - 1.0);
    return atlasTexelUv(probe_idx, @intFromFloat(fx), @intFromFloat(fy), atlas_w, atlas_h);
}

/// Accepts a raw world AABB as a proxy: finite lanes, strict min<=max
/// per axis, sane magnitude. Thin slabs (zero extent on one axis: walls,
/// floors, panels) are legitimate occluders — the slab march resolves
/// them exactly — so only all-degenerate (point), inverted, NaN or huge
/// boxes are dropped. Garbage bounds would otherwise poison every ray.
pub fn sanitizeProxy(min: [3]f32, max: [3]f32, albedo: [3]f32) ?ProxyBox {
    var extents: [3]f32 = .{ 0, 0, 0 };
    for (0..3) |a| {
        if (!std.math.isFinite(min[a]) or !std.math.isFinite(max[a])) return null;
        if (max[a] < min[a]) return null;
        if (@abs(min[a]) > 1e5 or @abs(max[a]) > 1e5) return null;
        extents[a] = max[a] - min[a];
    }
    if (extents[0] < MIN_PROXY_EXTENT and extents[1] < MIN_PROXY_EXTENT and extents[2] < MIN_PROXY_EXTENT) return null;
    for (0..3) |a| {
        if (!std.math.isFinite(albedo[a])) return null;
    }
    return .{ .min = min, .max = max, .albedo = .{
        std.math.clamp(albedo[0], 0.0, 1.0),
        std.math.clamp(albedo[1], 0.0, 1.0),
        std.math.clamp(albedo[2], 0.0, 1.0),
    } };
}

/// Removes near-duplicate proxies in place (O(n^2), n <= MAX_PROXIES):
/// same min/max within 1e-6 keep the first entry. Returns the kept count.
pub fn dedupeProxies(proxies: []ProxyBox) usize {
    var kept: usize = 0;
    for (proxies) |p| {
        var dup = false;
        for (proxies[0..kept]) |q| {
            var same = true;
            for (0..3) |a| {
                if (@abs(p.min[a] - q.min[a]) > 1e-6 or @abs(p.max[a] - q.max[a]) > 1e-6) {
                    same = false;
                    break;
                }
            }
            if (same) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            proxies[kept] = p;
            kept += 1;
        }
    }
    return kept;
}

/// Round-robin update window for amortized refresh: contiguous
/// [start, start+count) probe range for `frame_id`. per_frame == 0
/// updates the whole grid every frame.
pub const BudgetWindow = struct {
    start: usize,
    count: usize,
};

pub fn budgetWindow(frame_id: u64, probe_count: usize, per_frame: u32) BudgetWindow {
    if (probe_count == 0) return .{ .start = 0, .count = 0 };
    if (per_frame == 0) return .{ .start = 0, .count = probe_count };
    const b: usize = @min(@as(usize, per_frame), probe_count);
    const cycle: usize = (probe_count + b - 1) / b;
    const start: usize = (@as(usize, @intCast(frame_id % @as(u64, cycle)))) * b;
    return .{ .start = start, .count = @min(b, probe_count - start) };
}

/// Frames for one full-grid refresh under the budget (documented latency).
pub fn convergeFrames(probe_count: usize, per_frame: u32) usize {
    if (probe_count == 0) return 0;
    if (per_frame == 0) return 1;
    const b: usize = @min(@as(usize, per_frame), probe_count);
    return (probe_count + b - 1) / b;
}

/// CPU mirror of the update-shader slab test: nearest positive t, or
/// null on miss. inv_dir must be the component-wise reciprocal (guarded
/// by the caller against zero lanes, same as the shader's 1e-9 floor).
pub fn rayAabbT(origin: [3]f32, inv_dir: [3]f32, min: [3]f32, max: [3]f32) ?f32 {
    var tmin: f32 = 0.0;
    var tmax: f32 = std.math.floatMax(f32);
    for (0..3) |a| {
        var t0: f32 = (min[a] - origin[a]) * inv_dir[a];
        var t1: f32 = (max[a] - origin[a]) * inv_dir[a];
        if (t0 > t1) {
            const t = t0;
            t0 = t1;
            t1 = t;
        }
        tmin = @max(tmin, t0);
        tmax = @min(tmax, t1);
        if (tmin > tmax) return null;
    }
    if (tmax < 0.0) return null;
    return tmin;
}

/// The four view-global uniform lanes the sample path uploads
/// (frame_view.glsl ddgi_grid_origin/counts/step/atlas, appended last):
/// [origin.xyz, enabled], [counts.xyz, intensity],
/// [cell.xyz, tile_texels], [atlas_w, atlas_h, tiles_per_row, 0].
/// Zeros when disabled (bit-identical off path). Pure; tested.
pub fn gridLanes(opts: DdgiOptions) [4][4]f32 {
    if (!opts.enabled) return .{ .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } };
    const c = clampedCounts(opts.counts);
    const cell = cellSize(opts.extent, opts.counts);
    const probes = @as(usize, c[0]) * c[1] * c[2];
    const size = atlasSize(probes);
    return .{
        .{ opts.origin[0], opts.origin[1], opts.origin[2], 1.0 },
        .{ @floatFromInt(c[0]), @floatFromInt(c[1]), @floatFromInt(c[2]), opts.clampedIntensity() },
        .{ cell[0], cell[1], cell[2], @floatFromInt(PROBE_TILE) },
        .{ @floatFromInt(size[0]), @floatFromInt(size[1]), @floatFromInt(TILES_PER_ROW), 0.0 },
    };
}

/// Proxy albedo for a mesh: PBR base color, white otherwise (shader
/// materials carry procedural color; unlit/transparent never reach here).
pub fn proxyAlbedoFor(mesh: *const Mesh) [3]f32 {
    if (mesh.material) |m| {
        if (m == .pbr) {
            const c: Color3 = m.pbr.albedo_color;
            return .{
                std.math.clamp(c.r, 0.0, 1.0),
                std.math.clamp(c.g, 0.0, 1.0),
                std.math.clamp(c.b, 0.0, 1.0),
            };
        }
    }
    return .{ 1.0, 1.0, 1.0 };
}

/// Collects proxy boxes from scene meshes into `out` (cap = out.len,
/// collection order = mesh order), then dedupes in place. Skips hidden,
/// decal, transparent, LOD-child and geometry-less meshes; instanced
/// batches are v1-explicitly excluded (documented in docs/gi.md).
/// Particles/decals/transparents never enter: particles are not meshes,
/// transparents/decals are filtered here. Returns the kept count.
pub fn collectProxies(meshes: []const *Mesh, out: []ProxyBox) usize {
    var n: usize = 0;
    for (meshes) |mesh| {
        if (n >= out.len) break;
        if (!mesh.is_visible) continue;
        if (mesh.is_decal) continue;
        if (mesh.is_lod_child) continue;
        if (mesh.material) |m| {
            if (m.isTransparent()) continue;
        }
        const wb = mesh.getWorldBoundingBox();
        if (!wb.isValid()) continue;
        const lo: [3]f32 = .{ wb.min.x, wb.min.y, wb.min.z };
        const hi: [3]f32 = .{ wb.max.x, wb.max.y, wb.max.z };
        if (sanitizeProxy(lo, hi, proxyAlbedoFor(mesh))) |p| {
            out[n] = p;
            n += 1;
        }
    }
    return dedupeProxies(out[0..n]);
}

// ---------------------------------------------------------------------------
// House tests (pure math; the GLSL mirrors are pinned by value here).
// ---------------------------------------------------------------------------

fn expectApprox(a: f32, b: f32, tol: f32) !void {
    try std.testing.expect(@abs(a - b) <= tol);
}

test "ddgi counts clamp to axes and product cap" {
    try std.testing.expectEqual([3]u32{ 1, 5, 8 }, clampedCounts(.{ 0, 5, 99 }));
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, clampedCounts(.{ 0, 0, 0 }));
    const big = clampedCounts(.{ 8, 8, 8 });
    for (big) |a| try std.testing.expect(a >= 1 and a <= MAX_AXIS_PROBES);
    try std.testing.expect(big[0] * big[1] * big[2] <= MAX_PROBES);
    try std.testing.expectEqual([3]u32{ 5, 3, 5 }, clampedCounts(.{ 5, 3, 5 }));
    try std.testing.expectEqual(@as(usize, 75), probeCount(.{ 5, 3, 5 }));
}

test "ddgi probe positions are cell centers, x-fastest" {
    const origin = [3]f32{ 0, 0, 0 };
    const cell = [3]f32{ 1, 1, 1 };
    const p0 = probePosition(origin, cell, .{ 1, 1, 1 }, 0);
    try std.testing.expectEqual([3]f32{ 0.5, 0.5, 0.5 }, p0);
    const pa = probePosition(origin, cell, .{ 2, 1, 1 }, 0);
    const pb = probePosition(origin, cell, .{ 2, 1, 1 }, 1);
    try std.testing.expectEqual([3]f32{ 0.5, 0.5, 0.5 }, pa);
    try std.testing.expectEqual([3]f32{ 1.5, 0.5, 0.5 }, pb);
    const pc = probePosition(origin, cell, .{ 2, 2, 2 }, 6);
    try std.testing.expectEqual([3]f32{ 0.5, 1.5, 1.5 }, pc);
}

test "ddgi octahedral roundtrip" {
    const dirs = [_][3]f32{
        .{ 1, 0, 0 },  .{ 0, 1, 0 },     .{ 0, 0, 1 },
        .{ -1, 0, 0 }, .{ 0, -1, 0 },    .{ 0, 0, -1 },
        .{ 1, 1, 1 },  .{ -1, 1, -0.5 }, .{ 0.3, -0.8, 0.5 },
    };
    for (dirs) |d| {
        const len: f32 = @sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        const n = [3]f32{ d[0] / len, d[1] / len, d[2] / len };
        const back = octDecode(octEncode(n));
        try expectApprox(back[0], n[0], 1e-5);
        try expectApprox(back[1], n[1], 1e-5);
        try expectApprox(back[2], n[2], 1e-5);
        const o = octEncode(n);
        try std.testing.expect(o[0] >= 0.0 and o[0] <= 1.0 and o[1] >= 0.0 and o[1] <= 1.0);
    }
}

test "ddgi probe ray set covers the sphere" {
    var up = false;
    var down = false;
    var prev = [3]f32{ 0, 0, 0 };
    var n: u32 = 0;
    for (0..PROBE_TILE) |y| {
        for (0..PROBE_TILE) |x| {
            const d = probeRayDir(@intCast(x), @intCast(y));
            const len: f32 = @sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            try expectApprox(len, 1.0, 1e-5);
            if (d[2] > 0.0) up = true;
            if (d[2] < 0.0) down = true;
            if (n > 0) {
                const same = d[0] == prev[0] and d[1] == prev[1] and d[2] == prev[2];
                try std.testing.expect(!same);
            }
            prev = d;
            n += 1;
        }
    }
    try std.testing.expectEqual(@as(u32, 64), n);
    try std.testing.expect(up and down);
    // Deterministic: same texel, same direction.
    try std.testing.expectEqual(probeRayDir(3, 5), probeRayDir(3, 5));
}

test "ddgi atlas addressing" {
    try std.testing.expectEqual([2]u32{ 64, 8 }, atlasSize(1));
    try std.testing.expectEqual([2]u32{ 64, 8 }, atlasSize(8));
    try std.testing.expectEqual([2]u32{ 64, 16 }, atlasSize(9));
    try std.testing.expectEqual([2]u32{ 0, 0 }, atlasTileOrigin(0));
    try std.testing.expectEqual([2]u32{ 56, 0 }, atlasTileOrigin(7));
    try std.testing.expectEqual([2]u32{ 0, 8 }, atlasTileOrigin(8));
    const uv = atlasTexelUv(0, 0, 0, 64, 8);
    try expectApprox(uv[0], 0.5 / 64.0, 1e-6);
    try expectApprox(uv[1], 0.5 / 8.0, 1e-6);
}

test "ddgi sample uv roundtrips through the ray set" {
    // Every ray direction must sample back into its own texel: this pins
    // the CPU/GLSL encode agreement the whole sample path relies on.
    const size = atlasSize(75);
    for (0..PROBE_TILE) |y| {
        for (0..PROBE_TILE) |x| {
            const tx: u32 = @intCast(x);
            const ty: u32 = @intCast(y);
            const uv = sampleUv(3, probeRayDir(tx, ty), size[0], size[1]);
            const to = atlasTileOrigin(3);
            const px = uv[0] * @as(f32, @floatFromInt(size[0])) - @as(f32, @floatFromInt(to[0]));
            const py = uv[1] * @as(f32, @floatFromInt(size[1])) - @as(f32, @floatFromInt(to[1]));
            try std.testing.expectEqual(tx, @as(u32, @intFromFloat(@floor(px))));
            try std.testing.expectEqual(ty, @as(u32, @intFromFloat(@floor(py))));
        }
    }
}

test "ddgi probe blend weights" {
    const origin = [3]f32{ 0, 0, 0 };
    const cell = [3]f32{ 1, 1, 1 };
    const counts = [3]u32{ 2, 2, 2 };
    // Probe center: full weight on that probe.
    const b0 = probeBlend(.{ 0.5, 0.5, 0.5 }, origin, cell, counts);
    try std.testing.expectEqual(@as(usize, 0), b0.indices[0]);
    try expectApprox(b0.weights[0], 1.0, 1e-6);
    var sum: f32 = 0;
    for (b0.weights) |w| sum += w;
    try expectApprox(sum, 1.0, 1e-6);
    // Cell center: uniform 1/8 over all corners.
    const b1 = probeBlend(.{ 1.0, 1.0, 1.0 }, origin, cell, counts);
    sum = 0;
    for (b1.weights) |w| {
        try expectApprox(w, 0.125, 1e-6);
        sum += w;
    }
    try expectApprox(sum, 1.0, 1e-6);
    // Out of volume clamps: weights still sum to 1, indices in range.
    const b2 = probeBlend(.{ -5.0, 99.0, 0.5 }, origin, cell, counts);
    sum = 0;
    for (b2.indices, b2.weights) |idx, w| {
        try std.testing.expect(idx < 8);
        sum += w;
    }
    try expectApprox(sum, 1.0, 1e-5);
    // Degenerate single-probe axis: no blend, index 0.
    const b3 = probeBlend(.{ 3.0, 3.0, 3.0 }, origin, cell, .{ 1, 1, 1 });
    for (b3.indices) |idx| try std.testing.expectEqual(@as(usize, 0), idx);
    try expectApprox(b3.weights[0], 1.0, 1e-6);
}

test "ddgi sanitize drops bad boxes and clamps albedo" {
    try std.testing.expect(sanitizeProxy(.{ 0, 0, 0 }, .{ 1, 1, 1 }, .{ 0.5, 0.5, 0.5 }) != null);
    try std.testing.expect(sanitizeProxy(.{ 1, 0, 0 }, .{ 0, 1, 1 }, .{ 1, 1, 1 }) == null);
    // Thin slabs (walls/floors: zero extent on one axis) are valid proxies.
    try std.testing.expect(sanitizeProxy(.{ 0, 0, 0 }, .{ 0, 1, 1 }, .{ 1, 1, 1 }) != null);
    // Points (degenerate on all axes) are dropped.
    try std.testing.expect(sanitizeProxy(.{ 1, 1, 1 }, .{ 1, 1, 1 }, .{ 1, 1, 1 }) == null);
    try std.testing.expect(sanitizeProxy(.{ 0, 0, 0 }, .{ 1, 1, 1 }, .{ 2, -1, 0.5 }) != null);
    const cl = sanitizeProxy(.{ 0, 0, 0 }, .{ 1, 1, 1 }, .{ 2, -1, 0.5 }).?;
    try std.testing.expectEqual([3]f32{ 1, 0, 0.5 }, cl.albedo);
    const nan: f32 = std.math.nan(f32);
    try std.testing.expect(sanitizeProxy(.{ nan, 0, 0 }, .{ 1, 1, 1 }, .{ 1, 1, 1 }) == null);
    try std.testing.expect(sanitizeProxy(.{ 0, 0, 0 }, .{ 1, 1, 1 }, .{ nan, 1, 1 }) == null);
}

test "ddgi dedupe collapses copies in order" {
    var ps = [_]ProxyBox{
        .{ .min = .{ 0, 0, 0 }, .max = .{ 1, 1, 1 }, .albedo = .{ 1, 0, 0 } },
        .{ .min = .{ 0, 0, 0 }, .max = .{ 1, 1, 1 }, .albedo = .{ 0, 1, 0 } },
        .{ .min = .{ 2, 2, 2 }, .max = .{ 3, 3, 3 }, .albedo = .{ 0, 0, 1 } },
    };
    try std.testing.expectEqual(@as(usize, 2), dedupeProxies(&ps));
    try std.testing.expectEqual([3]f32{ 1, 0, 0 }, ps[0].albedo);
    try std.testing.expectEqual([3]f32{ 0, 0, 1 }, ps[1].albedo);
}

test "ddgi budget window round-robins" {
    try std.testing.expectEqual(BudgetWindow{ .start = 0, .count = 75 }, budgetWindow(0, 75, 0));
    try std.testing.expectEqual(BudgetWindow{ .start = 0, .count = 0 }, budgetWindow(0, 0, 8));
    const w0 = budgetWindow(0, 75, 32);
    try std.testing.expectEqual(BudgetWindow{ .start = 0, .count = 32 }, w0);
    const w1 = budgetWindow(1, 75, 32);
    try std.testing.expectEqual(BudgetWindow{ .start = 32, .count = 32 }, w1);
    const w2 = budgetWindow(2, 75, 32);
    try std.testing.expectEqual(BudgetWindow{ .start = 64, .count = 11 }, w2);
    const w3 = budgetWindow(3, 75, 32);
    try std.testing.expectEqual(BudgetWindow{ .start = 0, .count = 32 }, w3);
    try std.testing.expectEqual(@as(usize, 1), convergeFrames(75, 0));
    try std.testing.expectEqual(@as(usize, 3), convergeFrames(75, 32));
    try std.testing.expectEqual(@as(usize, 75), convergeFrames(75, 1));
}

test "ddgi slab march hits, misses and inside origins" {
    const inv = [3]f32{ 1, 1, 1 };
    const hit = rayAabbT(.{ 0, 0, -2 }, inv, .{ -1, -1, -1 }, .{ 1, 1, 1 });
    try std.testing.expect(hit != null);
    try expectApprox(hit.?, 1.0, 1e-6);
    try std.testing.expect(rayAabbT(.{ 5, 5, -2 }, inv, .{ -1, -1, -1 }, .{ 1, 1, 1 }) == null);
    // Origin inside: t = 0 (the update shader then applies the
    // SELF_HIT_MIN_T guard; the raw march reports contact).
    const inside = rayAabbT(.{ 0, 0, 0 }, inv, .{ -1, -1, -1 }, .{ 1, 1, 1 });
    try std.testing.expect(inside != null);
    try expectApprox(inside.?, 0.0, 1e-6);
    // Box fully behind the ray (origin above, direction up): miss.
    try std.testing.expect(rayAabbT(.{ 0, 0, 2 }, [3]f32{ 1, 1, 1 }, .{ -1, -1, -1 }, .{ 1, 1, 1 }) == null);
}

test "ddgi grid lanes are zero when disabled" {
    const off = gridLanes(.{});
    for (off) |lane| try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, lane);
    const on = gridLanes(.{
        .enabled = true,
        .origin = .{ -3, 0, -3 },
        .extent = .{ 6, 3, 6 },
        .counts = .{ 5, 3, 5 },
        .intensity = 2.0,
    });
    try std.testing.expectEqual([4]f32{ -3, 0, -3, 1 }, on[0]);
    try std.testing.expectEqual([4]f32{ 5, 3, 5, 2 }, on[1]);
    try std.testing.expectEqual([4]f32{ 1.2, 1.0, 1.2, 8 }, on[2]);
    try std.testing.expectEqual([4]f32{ 64, 80, 8, 0 }, on[3]);
}

test "ddgi collect skips hidden, decal, transparent and invalid" {
    const PBRMaterial = @import("../material.zig").PBRMaterial;
    const Material = @import("../material.zig").Material;
    var red = PBRMaterial.init("red");
    red.albedo_color = Color3.new(1, 0, 0);
    var glass = PBRMaterial.init("glass");
    glass.alpha_mode = .blend;
    var meshes = [_]Mesh{
        .{ .name = "wall", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3, .local_bounding_box = .{ .min = .{ .x = 0, .y = 0, .z = 0 }, .max = .{ .x = 1, .y = 1, .z = 1 } }, .material = Material{ .pbr = &red } },
        .{ .name = "wall_dup", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3, .local_bounding_box = .{ .min = .{ .x = 0, .y = 0, .z = 0 }, .max = .{ .x = 1, .y = 1, .z = 1 } }, .material = Material{ .pbr = &red } },
        .{ .name = "hidden", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3, .is_visible = false, .local_bounding_box = .{ .min = .{ .x = 5, .y = 5, .z = 5 }, .max = .{ .x = 6, .y = 6, .z = 6 } } },
        .{ .name = "decal", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3, .is_decal = true, .local_bounding_box = .{ .min = .{ .x = 7, .y = 7, .z = 7 }, .max = .{ .x = 8, .y = 8, .z = 8 } } },
        .{ .name = "glass", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3, .local_bounding_box = .{ .min = .{ .x = 9, .y = 9, .z = 9 }, .max = .{ .x = 10, .y = 10, .z = 10 } }, .material = Material{ .pbr = &glass } },
        .{ .name = "flat", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 3, .local_bounding_box = .{ .min = .{ .x = 11, .y = 11, .z = 11 }, .max = .{ .x = 11, .y = 11, .z = 11 } } },
    };
    var ptrs = [_]*Mesh{ &meshes[0], &meshes[1], &meshes[2], &meshes[3], &meshes[4], &meshes[5] };
    var out = [_]ProxyBox{.{ .min = .{ 0, 0, 0 }, .max = .{ 0, 0, 0 }, .albedo = .{ 0, 0, 0 } }} ** 8;
    // wall + wall_dup dedupe to one, everything else filtered.
    try std.testing.expectEqual(@as(usize, 1), collectProxies(&ptrs, &out));
    try std.testing.expectEqual([3]f32{ 0, 0, 0 }, out[0].min);
    try std.testing.expectEqual([3]f32{ 1, 0, 0 }, out[0].albedo);
}

/// Trilinear blend of the 8 surrounding probes for a world position.
/// Out-of-volume positions clamp to the border probes (no extrapolation).
/// Matches ddgi_sample.glsl ddgiBlend.
pub const ProbeBlend = struct {
    indices: [8]usize,
    weights: [8]f32,
};

pub fn probeBlend(pos: [3]f32, origin: [3]f32, cell: [3]f32, counts: [3]u32) ProbeBlend {
    const c = clampedCounts(counts);
    const nx: usize = c[0];
    const ny: usize = c[1];
    const nz: usize = c[2];
    var base: [3]usize = .{ 0, 0, 0 };
    var frac: [3]f32 = .{ 0, 0, 0 };
    const o = [3]f32{ origin[0], origin[1], origin[2] };
    const cl = [3]f32{ cell[0], cell[1], cell[2] };
    const ns = [3]usize{ nx, ny, nz };
    inline for (0..3) |a| {
        const n = ns[a];
        if (n <= 1) {
            base[a] = 0;
            frac[a] = 0.0;
        } else {
            const g: f32 = (pos[a] - o[a]) / @max(cl[a], 1e-9) - 0.5;
            const gc: f32 = std.math.clamp(g, 0.0, @as(f32, @floatFromInt(n - 1)));
            // Lower corner, clamped so base+1 stays in range.
            base[a] = @min(@as(usize, @intFromFloat(gc)), n - 2);
            frac[a] = std.math.clamp(gc - @as(f32, @floatFromInt(base[a])), 0.0, 1.0);
        }
    }
    // Linearize 8 corners x-fastest.
    var out = ProbeBlend{ .indices = [_]usize{0} ** 8, .weights = [_]f32{0} ** 8 };
    for (0..8) |k| {
        const dx: usize = k & 1;
        const dy: usize = (k >> 1) & 1;
        const dz: usize = (k >> 2) & 1;
        const ix = @min(base[0] + dx, nx - 1);
        const iy = @min(base[1] + dy, ny - 1);
        const iz = @min(base[2] + dz, nz - 1);
        out.indices[k] = ix + nx * (iy + ny * iz);
        const wx: f32 = if (dx == 0) 1.0 - frac[0] else frac[0];
        const wy: f32 = if (dy == 0) 1.0 - frac[1] else frac[1];
        const wz: f32 = if (dz == 0) 1.0 - frac[2] else frac[2];
        out.weights[k] = wx * wy * wz;
    }
    return out;
}
