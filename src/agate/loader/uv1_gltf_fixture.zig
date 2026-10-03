//! CPU-only glTF JSON fixtures for UV1 (TEXCOORD_1) validation tests.
//!
//! Builds a minimal one-triangle scene (positions + UV0 + UV1) with an
//! embedded base64 `data:` buffer, then parses it through `cgltf_parse` +
//! `cgltf_load_buffers` exactly like `SceneLoader.appendGlbOptions` does.
//! No GPU, no files: images are URI-only (`dummy.png`, never decoded) so
//! the texture views exist for coordinate validation without touching the
//! image pipeline.
//!
//! Triangle data (shared by every case):
//! - positions: (0,0,0) (1,0,0) (0,1,0)
//! - UV0: (0,0) (1,0) (0,1)
//! - UV1: (0.25,0.75) (0.5,0.5) (1,0) — distinct from UV0 on purpose.
//!
//! The default material selects UV1 three independent ways: the base color
//! view overrides `texCoord: 0` to 1 via `KHR_texture_transform`, the
//! clearcoat view selects 1 directly, and the sheen view overrides 0 to 1
//! with its own offset. Cases tweak one axis each.

const std = @import("std");

const c = @import("../c.zig").c;

pub const Case = enum {
    /// Textured material wants UV1; mesh provides TEXCOORD_0 + TEXCOORD_1.
    valid_override,
    /// Same material, but the primitive only has TEXCOORD_0.
    missing_uv1,
    /// Base `KHR_texture_transform` overrides to texCoord 2 (unsupported).
    override_2,
    /// TEXCOORD_1 accessor is VEC3 instead of VEC2.
    uv1_vec3,
    /// TEXCOORD_1 accessor count (2) mismatches POSITION count (3).
    uv1_count_mismatch,
};

const positions: [9]f32 = .{ 0, 0, 0, 1, 0, 0, 0, 1, 0 };
const uv0: [6]f32 = .{ 0, 0, 1, 0, 0, 1 };
const uv1: [6]f32 = .{ 0.25, 0.75, 0.5, 0.5, 1, 0 };

fn writeF32Le(dst: *[4]u8, v: f32) void {
    std.mem.writeInt(u32, dst, @bitCast(v), .little);
}

/// Material whose three textured slots all resolve to coordinate set 1
/// through independent selections (base override, clearcoat direct,
/// sheen override with its own offset).
const valid_material_json =
    \\{"pbrMetallicRoughness":{"baseColorTexture":{"index":0,"texCoord":0,"extensions":{"KHR_texture_transform":{"texCoord":1}}}},
    \\"extensions":{"KHR_materials_clearcoat":{"clearcoatTexture":{"index":0,"texCoord":1}},
    \\"KHR_materials_sheen":{"sheenColorTexture":{"index":0,"texCoord":0,"extensions":{"KHR_texture_transform":{"texCoord":1,"offset":[0.5,0.25]}}}}}}
;

/// Same shape, but the base override points at the unsupported set 2.
const override2_material_json =
    \\{"pbrMetallicRoughness":{"baseColorTexture":{"index":0,"texCoord":0,"extensions":{"KHR_texture_transform":{"texCoord":2}}}},
    \\"extensions":{"KHR_materials_clearcoat":{"clearcoatTexture":{"index":0,"texCoord":1}},
    \\"KHR_materials_sheen":{"sheenColorTexture":{"index":0,"texCoord":0,"extensions":{"KHR_texture_transform":{"texCoord":1,"offset":[0.5,0.25]}}}}}}
;

/// Serializes one fixture case to an in-memory `.gltf` JSON document.
/// Caller owns the returned slice.
pub fn buildJson(allocator: std.mem.Allocator, case: Case) ![]u8 {
    var raw: [96]u8 = undefined;
    var len: usize = 0;
    for (positions) |v| {
        writeF32Le(raw[len..][0..4], v);
        len += 4;
    }
    for (uv0) |v| {
        writeF32Le(raw[len..][0..4], v);
        len += 4;
    }
    const uv1_offset = len;
    if (case == .uv1_vec3) {
        // VEC3 needs 36 bytes; pad z with 0.
        var i: usize = 0;
        while (i < uv1.len) : (i += 2) {
            writeF32Le(raw[len..][0..4], uv1[i]);
            len += 4;
            writeF32Le(raw[len..][0..4], uv1[i + 1]);
            len += 4;
            writeF32Le(raw[len..][0..4], 0);
            len += 4;
        }
    } else {
        for (uv1) |v| {
            writeF32Le(raw[len..][0..4], v);
            len += 4;
        }
    }
    const uv1_view_len = len - uv1_offset;

    // 96 raw bytes max -> 128 base64 chars; shorter cases leave slack.
    var b64buf: [128]u8 = undefined;
    const b64 = std.base64.standard.Encoder.encode(&b64buf, raw[0..len]);

    const uv1_type: []const u8 = if (case == .uv1_vec3) "VEC3" else "VEC2";
    const uv1_count: usize = if (case == .uv1_count_mismatch) 2 else 3;
    const material_json: []const u8 = if (case == .override_2) override2_material_json else valid_material_json;
    const attributes: []const u8 = if (case == .missing_uv1)
        \\"POSITION":0,"TEXCOORD_0":1
    else
        \\"POSITION":0,"TEXCOORD_0":1,"TEXCOORD_1":2
    ;

    return std.fmt.allocPrint(allocator,
        \\{{"asset":{{"version":"2.0"}},"extensionsUsed":["KHR_texture_transform","KHR_materials_clearcoat","KHR_materials_sheen"],
        \\"buffers":[{{"byteLength":{d},"uri":"data:application/octet-stream;base64,{s}"}}],
        \\"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36,"target":34962}},{{"buffer":0,"byteOffset":36,"byteLength":24,"target":34962}},{{"buffer":0,"byteOffset":{d},"byteLength":{d},"target":34962}}],
        \\"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},{{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"}},{{"bufferView":2,"componentType":5126,"count":{d},"type":"{s}"}}],
        \\"images":[{{"uri":"dummy.png"}}],"textures":[{{"source":0}}],
        \\"materials":[{s}],
        \\"meshes":[{{"primitives":[{{"attributes":{{{s}}},"material":0}}]}}],
        \\"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}],"scene":0}}
    , .{
        len,
        b64,
        uv1_offset,
        uv1_view_len,
        uv1_count,
        uv1_type,
        material_json,
        attributes,
    });
}

/// Parses `json` (from `buildJson`) and loads its embedded buffers.
/// Mirrors the `SceneLoader.appendGlbOptions` parse prefix. Caller must
/// call `c.cgltf_free` on success; `json` must outlive the call only.
pub fn parseLoaded(json: []const u8) !*c.cgltf_data {
    var options = std.mem.zeroes(c.cgltf_options);
    var data: ?*c.cgltf_data = null;
    const parse_res = c.cgltf_parse(&options, json.ptr, json.len, &data);
    const loaded = data orelse return error.GltfParseFailed;
    if (parse_res != c.cgltf_result_success) {
        c.cgltf_free(loaded);
        return error.GltfParseFailed;
    }
    errdefer c.cgltf_free(loaded);
    if (c.cgltf_load_buffers(&options, loaded, null) != c.cgltf_result_success) {
        return error.GltfLoadBuffersFailed;
    }
    return loaded;
}
