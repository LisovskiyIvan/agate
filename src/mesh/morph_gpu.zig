const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const types = @import("types.zig");
const MorphTarget = types.MorphTarget;
pub const MAX_MORPH_TARGETS = types.MAX_MORPH_TARGETS;
const Mesh = @import("mesh.zig").Mesh;

// GPU morph blending (opt-in via Mesh.morph_mode == .gpu).
//
// Deltas live in one RGBA32F texture laid out as a linear strip of texels:
// per vertex texels_per_vertex = target_count * 3 texels, texel index
//   vertex_index * (target_count * 3) + target * 3 + slot,
// slot order position, normal, tangent; texel.xyz carries the delta, .w is
// padding (zeros). Absent attributes/targets stay zero. The vertex shader
// loops up to target_count (passed via morph_params.w, cap 64) and samples
// texel centers with a nonfiltering sampler; the strip wraps at
// TEXTURE_MAX_WIDTH so NPOT widths and huge meshes stay inside every
// backend's texture size limits.

pub const MAX_TEXELS_PER_VERTEX: u32 = MAX_MORPH_TARGETS * 3;
pub const TEXELS_PER_VERTEX: u32 = MAX_TEXELS_PER_VERTEX;

/// Strip wrap point. 4096 texels keeps the largest dimension at or below
/// the common minimum MAX_TEXTURE_SIZE across GL/Metal/D3D11 backends.
pub const TEXTURE_MAX_WIDTH: u32 = 4096;

pub const MorphSlot = enum(u32) { position = 0, normal = 1, tangent = 2 };

pub const TextureSize = struct { width: u32, height: u32 };

pub fn texelsPerVertex(target_count: usize) u32 {
    const t: u64 = @as(u64, @min(target_count, MAX_MORPH_TARGETS));
    const res: u64 = t * 3;
    return @intCast(res);
}

/// Strip dimensions covering vertex_count vertices for target_count targets.
/// At least 1x1 so the texture exists even for degenerate meshes (default bind target).
pub fn textureSizeFor(vertex_count: usize, target_count: usize) TextureSize {
    const tpv: u64 = texelsPerVertex(target_count);
    const texels: u64 = @as(u64, vertex_count) * tpv;
    const width: u64 = @min(@as(u64, TEXTURE_MAX_WIDTH), @max(texels, 1));
    const height: u64 = @max((texels + width - 1) / width, 1);
    return .{ .width = @intCast(width), .height = @intCast(height) };
}

/// Texel index for one vertex/target/attribute slot (matches the GLSL).
pub fn texelIndex(vertex_index: usize, target: usize, slot: MorphSlot, target_count: usize) u32 {
    const tpv = target_count * 3;
    return @intCast(vertex_index * tpv + target * 3 + @intFromEnum(slot));
}

/// Packs morph target deltas into RGBA32F texels (4 floats per texel,
/// texel.xyz = delta, .w = 0). Everything absent stays zero. Returns a
/// texel_count * 4 slice; caller frees.
pub fn packDeltas(
    allocator: std.mem.Allocator,
    targets: []const MorphTarget,
    vertex_count: usize,
    size: TextureSize,
    target_count: usize,
) ![]f32 {
    const texel_count: usize = @as(usize, size.width) * size.height;
    const pixels = try allocator.alloc(f32, texel_count * 4);
    errdefer allocator.free(pixels);
    @memset(pixels, 0);
    const texels: [][4]f32 = @ptrCast(pixels);

    const actual_targets = @min(@min(targets.len, target_count), MAX_MORPH_TARGETS);
    for (0..actual_targets) |t| {
        const mt = &targets[t];
        for (0..vertex_count) |v| {
            const ti = texelIndex(v, t, .position, target_count);
            if (ti + 2 >= texel_count) break;
            if (v < mt.position_deltas.len) {
                const d = mt.position_deltas[v];
                texels[ti] = .{ d[0], d[1], d[2], 0 };
            }
            if (v < mt.normal_deltas.len) {
                const d = mt.normal_deltas[v];
                texels[ti + 1] = .{ d[0], d[1], d[2], 0 };
            }
            if (v < mt.tangent_deltas.len) {
                const d = mt.tangent_deltas[v];
                texels[ti + 2] = .{ d[0], d[1], d[2], 0 };
            }
        }
    }
    return pixels;
}

pub const BlendedDeltas = struct {
    position: [3]f32,
    normal: [3]f32,
    tangent: [3]f32,
};

/// Pure Zig mirror of the GLSL applyMorphDeltas (standard/pbr/skinned_pbr
/// vs): same texel index math, same zero-weight skip, same accumulation
/// order. Lets tests pin the shader contract without a GPU.
pub fn blendDeltas(
    texels: []const [4]f32,
    weights: []const f32,
    vertex_index: usize,
    target_count: usize,
) BlendedDeltas {
    var out: BlendedDeltas = .{
        .position = .{ 0, 0, 0 },
        .normal = .{ 0, 0, 0 },
        .tangent = .{ 0, 0, 0 },
    };
    const n = @min(@min(weights.len, target_count), MAX_MORPH_TARGETS);
    for (0..n) |t| {
        const w = weights[t];
        if (w == 0.0) continue;
        const idx = texelIndex(vertex_index, t, .position, target_count);
        if (idx + 2 >= texels.len) break;
        const wx: @Vector(4, f32) = @splat(w);
        const dp = wx * texels[idx];
        const dn = wx * texels[idx + 1];
        const dt = wx * texels[idx + 2];
        out.position[0] += dp[0];
        out.position[1] += dp[1];
        out.position[2] += dp[2];
        out.normal[0] += dn[0];
        out.normal[1] += dn[1];
        out.normal[2] += dn[2];
        out.tangent[0] += dt[0];
        out.tangent[1] += dt[1];
        out.tangent[2] += dt[2];
    }
    return out;
}

pub const PackedWeights = [16][4]f32;

/// Splits up to 64 weights into 16 vec4 lanes of the vs_morph
/// uniform block. Missing entries stay 0.
pub fn packWeights(weights: []const f32) PackedWeights {
    std.debug.assert(weights.len <= MAX_MORPH_TARGETS);
    var pw: PackedWeights = @splat(.{ 0, 0, 0, 0 });
    const n = @min(weights.len, MAX_MORPH_TARGETS);
    for (0..n) |i| {
        pw[i >> 2][i & 3] = weights[i];
    }
    return pw;
}

/// Per-draw vs_morph uniform values for one mesh. Active only when the
/// mesh is in GPU mode; everything else binds the disabled default (zero
/// weights on a 1x1 zero texture). A `.gpu` mesh without an uploaded delta
/// texture is a wiring bug and panics here — drawing the base pose silently
/// would hide it.
pub const VsUniforms = struct {
    weights: [16][4]f32,
    /// x: enabled (0/1), y: tex width, z: tex height, w: target count.
    params: [4]f32,
};

pub fn vsUniforms(mesh: *const Mesh) VsUniforms {
    if (mesh.morph_mode != .gpu) {
        return .{
            .weights = @splat(.{ 0, 0, 0, 0 }),
            .params = .{ 0, 1, 1, 0 },
        };
    }
    if (mesh.morph_delta_view.id == 0) {
        std.debug.panic(
            "mesh '{s}': morph_mode == .gpu requires morph_gpu.uploadMorphDeltas() before drawing (no silent base-pose fallback)",
            .{mesh.name},
        );
    }
    const pw = packWeights(mesh.morph_weights);
    const target_count = @min(mesh.morph_targets.len, MAX_MORPH_TARGETS);
    return .{
        .weights = pw,
        .params = .{
            1,
            @floatFromInt(mesh.morph_tex_width),
            @floatFromInt(mesh.morph_tex_height),
            @floatFromInt(target_count),
        },
    };
}

/// Backend capability gate for GPU morph blending (compute-style explicit
/// check, no silent fallback): true only with a live sg context whose
/// backend reports RGBA32F sample support. The morph shaders sample the
/// delta strip with NEAREST through an `unfilterable_float` /
/// `nonfiltering` pair (the ozz-skin precedent in sokol's docs), so only
/// the sample bit gates — filtering is never required. Headless (unit
/// tests, id == 0) reports false.
pub fn supported() bool {
    if (!sg.isvalid()) return false;
    return sg.queryPixelformat(.RGBA32F).sample;
}

/// Packs the mesh's morph targets into an RGBA32F image and stores the
/// resulting view on the mesh (replacing any previous delta texture).
/// Requires RGBA32F support; failure propagates loudly (no silent CPU
/// fallback: the caller already built a static vertex buffer).
pub fn uploadMorphDeltas(mesh: *Mesh, allocator: std.mem.Allocator) !void {
    const target_count = @min(mesh.morph_targets.len, MAX_MORPH_TARGETS);
    const size = textureSizeFor(mesh.morph_base.len, target_count);
    const pixels = try packDeltas(allocator, mesh.morph_targets, mesh.morph_base.len, size, target_count);
    defer allocator.free(pixels);

    var img_desc = sg.ImageDesc{
        .width = @intCast(size.width),
        .height = @intCast(size.height),
        .pixel_format = .RGBA32F,
    };
    img_desc.data.mip_levels[0] = sg.asRange(pixels);
    const img = sg.makeImage(img_desc);
    const view = sg.makeView(.{ .texture = .{ .image = img } });
    if (img.id == 0 or view.id == 0) {
        // Pool exhaustion or missing RGBA32F support: never publish dead
        // handles (vsUniforms panics on an invalid view) and destroy any
        // partial handle here instead of leaking it. The caller decides
        // between a hard load error (on-context) and a deferred retry.
        if (view.id != 0) sg.destroyView(view);
        if (img.id != 0) sg.destroyImage(img);
        return error.MorphDeltaTextureUnavailable;
    }

    destroyDeltaResources(mesh);
    mesh.morph_delta_image = img;
    mesh.morph_delta_view = view;
    mesh.morph_tex_width = size.width;
    mesh.morph_tex_height = size.height;
}

/// Destroys the mesh's delta image and view (GPU calls). Both Mesh.deinit
/// and finishGpuUpload tear the resources down inline for the retry path.
fn destroyDeltaResources(mesh: *Mesh) void {
    if (mesh.morph_delta_view.id != 0) sg.destroyView(mesh.morph_delta_view);
    if (mesh.morph_delta_image.id != 0) sg.destroyImage(mesh.morph_delta_image);
    mesh.morph_delta_image = .{};
    mesh.morph_delta_view = .{};
    mesh.morph_tex_width = 0;
    mesh.morph_tex_height = 0;
}
