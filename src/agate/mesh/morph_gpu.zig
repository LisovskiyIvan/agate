const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const types = @import("types.zig");
const MorphTarget = types.MorphTarget;
const MAX_MORPH_TARGETS = types.MAX_MORPH_TARGETS;
const Mesh = @import("mesh.zig").Mesh;

// GPU morph blending (opt-in via Mesh.morph_mode == .gpu).
//
// Deltas live in one RGBA32F texture laid out as a linear strip of texels:
// per vertex TEXELS_PER_VERTEX texels, texel index
//   vertex_index * TEXELS_PER_VERTEX + target * 3 + slot,
// slot order position, normal, tangent; texel.xyz carries the delta, .w is
// padding (zeros). Absent attributes/targets stay zero, so the vertex
// shader can accumulate all 8 targets unconditionally (zero weights and
// zero texels are both no-ops). The vertex shader mirrors blendDeltas
// below and samples texel centers with a nonfiltering sampler; the strip
// wraps at TEXTURE_MAX_WIDTH so NPOT widths and huge meshes stay inside
// every backend's texture size limits.

/// One RGBA32F texel per attribute slot of every target:
/// 8 targets x (position, normal, tangent) = 24 texels per vertex.
pub const TEXELS_PER_VERTEX: u32 = MAX_MORPH_TARGETS * 3;

/// Strip wrap point. 4096 texels keeps the largest dimension at or below
/// the common minimum MAX_TEXTURE_SIZE across GL/Metal/D3D11 backends.
pub const TEXTURE_MAX_WIDTH: u32 = 4096;

pub const MorphSlot = enum(u32) { position = 0, normal = 1, tangent = 2 };

pub const TextureSize = struct { width: u32, height: u32 };

/// Strip dimensions covering vertex_count vertices. At least 1x1 so the
/// texture exists even for degenerate meshes (default bind target).
pub fn textureSizeFor(vertex_count: usize) TextureSize {
    const texels: u64 = @as(u64, vertex_count) * TEXELS_PER_VERTEX;
    const width: u64 = @min(@as(u64, TEXTURE_MAX_WIDTH), @max(texels, 1));
    const height: u64 = @max((texels + width - 1) / width, 1);
    return .{ .width = @intCast(width), .height = @intCast(height) };
}

/// Texel index for one vertex/target/attribute slot (matches the GLSL).
pub fn texelIndex(vertex_index: usize, target: usize, slot: MorphSlot) u32 {
    return @intCast(vertex_index * TEXELS_PER_VERTEX + target * 3 + @intFromEnum(slot));
}

/// Packs morph target deltas into RGBA32F texels (4 floats per texel,
/// texel.xyz = delta, .w = 0). Everything absent stays zero. Returns a
/// texel_count * 4 slice; caller frees.
pub fn packDeltas(
    allocator: std.mem.Allocator,
    targets: []const MorphTarget,
    vertex_count: usize,
    size: TextureSize,
) ![]f32 {
    const texel_count: usize = @as(usize, size.width) * size.height;
    const pixels = try allocator.alloc(f32, texel_count * 4);
    errdefer allocator.free(pixels);
    @memset(pixels, 0);
    const texels: [][4]f32 = @ptrCast(pixels);

    for (targets, 0..) |mt, t| {
        if (t >= MAX_MORPH_TARGETS) break;
        for (0..vertex_count) |v| {
            const ti = texelIndex(v, t, .position);
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
    weights: [MAX_MORPH_TARGETS]f32,
    vertex_index: usize,
) BlendedDeltas {
    var out: BlendedDeltas = .{
        .position = .{ 0, 0, 0 },
        .normal = .{ 0, 0, 0 },
        .tangent = .{ 0, 0, 0 },
    };
    for (0..MAX_MORPH_TARGETS) |t| {
        const w = weights[t];
        if (w == 0.0) continue;
        const idx = texelIndex(vertex_index, t, .position);
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

pub const PackedWeights = struct {
    w0: [4]f32,
    w1: [4]f32,
};

/// Splits the up-to-8 weights into the two vec4 lanes of the vs_morph
/// uniform block (targets 0..3 -> w0, 4..7 -> w1). Missing entries stay 0.
/// The 8-target cap is a hard invariant (mesh loader clamps to
/// MAX_MORPH_TARGETS); a longer slice is a caller bug -> panic.
pub fn packWeights(weights: []const f32) PackedWeights {
    std.debug.assert(weights.len <= MAX_MORPH_TARGETS);
    var pw: PackedWeights = .{ .w0 = .{ 0, 0, 0, 0 }, .w1 = .{ 0, 0, 0, 0 } };
    for (weights, 0..) |w, i| {
        switch (i) {
            0...3 => pw.w0[i] = w,
            4...7 => pw.w1[i - 4] = w,
            // Assert above proves i < 8; ReleaseFast drops the branch.
            else => unreachable,
        }
    }
    return pw;
}

/// Per-draw vs_morph uniform values for one mesh. Active only when the
/// mesh is in GPU mode; everything else binds the disabled default (zero
/// weights on a 1x1 zero texture). A `.gpu` mesh without an uploaded delta
/// texture is a wiring bug and panics here — drawing the base pose silently
/// would hide it.
pub const VsUniforms = struct {
    weights0: [4]f32,
    weights1: [4]f32,
    /// x: enabled (0/1), y: tex width, z: tex height, w: unused.
    params: [4]f32,
};

pub fn vsUniforms(mesh: *const Mesh) VsUniforms {
    if (mesh.morph_mode != .gpu) {
        return .{
            .weights0 = .{ 0, 0, 0, 0 },
            .weights1 = .{ 0, 0, 0, 0 },
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
    return .{
        .weights0 = pw.w0,
        .weights1 = pw.w1,
        .params = .{
            1,
            @floatFromInt(mesh.morph_tex_width),
            @floatFromInt(mesh.morph_tex_height),
            0,
        },
    };
}

/// Packs the mesh's morph targets into an RGBA32F image and stores the
/// resulting view on the mesh (replacing any previous delta texture).
/// Requires RGBA32F support; failure propagates loudly (no silent CPU
/// fallback: the caller already built a static vertex buffer).
pub fn uploadMorphDeltas(mesh: *Mesh, allocator: std.mem.Allocator) !void {
    const size = textureSizeFor(mesh.morph_base.len);
    const pixels = try packDeltas(allocator, mesh.morph_targets, mesh.morph_base.len, size);
    defer allocator.free(pixels);

    var img_desc = sg.ImageDesc{
        .width = @intCast(size.width),
        .height = @intCast(size.height),
        .pixel_format = .RGBA32F,
    };
    img_desc.data.mip_levels[0] = sg.asRange(pixels);
    const img = sg.makeImage(img_desc);
    const view = sg.makeView(.{ .texture = .{ .image = img } });

    destroyDeltaResources(mesh);
    mesh.morph_delta_image = img;
    mesh.morph_delta_view = view;
    mesh.morph_tex_width = size.width;
    mesh.morph_tex_height = size.height;
}

/// Destroys the mesh's delta image and view (GPU calls). Mesh.deinit
/// performs the same teardown inline to avoid importing this module back.
fn destroyDeltaResources(mesh: *Mesh) void {
    if (mesh.morph_delta_view.id != 0) sg.destroyView(mesh.morph_delta_view);
    if (mesh.morph_delta_image.id != 0) sg.destroyImage(mesh.morph_delta_image);
    mesh.morph_delta_image = .{};
    mesh.morph_delta_view = .{};
    mesh.morph_tex_width = 0;
    mesh.morph_tex_height = 0;
}
