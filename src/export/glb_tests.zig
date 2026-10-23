const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mesh = @import("../mesh.zig").Mesh;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const c = @import("../c.zig").c;
const glb = @import("glb.zig");
const writeGlbMeshAlloc = glb.writeGlbMeshAlloc;
const GLB_MAGIC = glb.GLB_MAGIC;

test "GLB exporter creates valid glTF 2.0 binary and round-trips via cgltf" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var positions = [_]Vec3{
        Vec3.new(-1, -1, -1),
        Vec3.new(1, -1, -1),
        Vec3.new(1, 1, -1),
        Vec3.new(-1, 1, -1),
        Vec3.new(-1, -1, 1),
        Vec3.new(1, -1, 1),
        Vec3.new(1, 1, 1),
        Vec3.new(-1, 1, 1),
    };
    var indices = [_]u32{
        0, 1, 2, 0, 2, 3,
        5, 4, 7, 5, 7, 6,
        4, 0, 3, 4, 3, 7,
        1, 5, 6, 1, 6, 2,
        3, 2, 6, 3, 6, 7,
        4, 5, 1, 4, 1, 0,
    };
    var cube = Mesh{
        .name = "TestCube",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = indices.len,
        .cpu_positions = &positions,
        .cpu_indices = &indices,
    };

    var mat = PBRMaterial{
        .name = "CubeMat",
        .albedo_color = math.Color3.new(0.8, 0.2, 0.1),
        .metallic = 0.5,
        .roughness = 0.25,
    };
    cube.material = .{ .pbr = &mat };

    const glb_bytes = try writeGlbMeshAlloc(allocator, &cube, .{
        .emit_materials = true,
        .apply_world_transform = false,
    });
    defer allocator.free(glb_bytes);

    // 1. Verify container layout
    try testing.expect(glb_bytes.len >= 28);
    const magic = std.mem.readInt(u32, glb_bytes[0..4], .little);
    try testing.expectEqual(GLB_MAGIC, magic);
    const version = std.mem.readInt(u32, glb_bytes[4..8], .little);
    try testing.expectEqual(@as(u32, 2), version);
    const total_len = std.mem.readInt(u32, glb_bytes[8..12], .little);
    try testing.expectEqual(@as(u32, @intCast(glb_bytes.len)), total_len);

    // 2. Parse using cgltf
    var options = std.mem.zeroes(c.cgltf_options);
    var data: ?*c.cgltf_data = null;
    const parse_res = c.cgltf_parse(&options, glb_bytes.ptr, glb_bytes.len, &data);
    try testing.expect(parse_res == c.cgltf_result_success);
    defer c.cgltf_free(data);

    const gltf_data = data.?;
    const load_res = c.cgltf_load_buffers(&options, gltf_data, null);
    try testing.expect(load_res == c.cgltf_result_success);

    // 3. Verify scene & mesh counts
    try testing.expectEqual(@as(usize, 1), gltf_data.scenes_count);
    try testing.expectEqual(@as(usize, 1), gltf_data.nodes_count);
    try testing.expectEqual(@as(usize, 1), gltf_data.meshes_count);

    const mesh = &gltf_data.meshes[0];
    try testing.expect(std.mem.eql(u8, std.mem.span(mesh.name), "TestCube"));
    try testing.expectEqual(@as(usize, 1), mesh.primitives_count);

    const prim = &mesh.primitives[0];
    try testing.expect(prim.indices != null);
    try testing.expectEqual(cube.cpu_indices.len, prim.indices.*.count);

    // Verify position attribute count matches cube positions
    var found_pos = false;
    for (0..prim.attributes_count) |ai| {
        const attr = &prim.attributes[ai];
        if (attr.type == c.cgltf_attribute_type_position) {
            found_pos = true;
            try testing.expectEqual(cube.cpu_positions.len, attr.data.*.count);
        }
    }
    try testing.expect(found_pos);

    // Verify material
    try testing.expect(prim.material != null);
    const parsed_mat = prim.material.?;
    try testing.expect(std.mem.eql(u8, std.mem.span(parsed_mat.*.name), "CubeMat"));
    try testing.expect(parsed_mat.*.has_pbr_metallic_roughness != 0);
    const pbr = parsed_mat.*.pbr_metallic_roughness;
    try testing.expectApproxEqAbs(@as(f32, 0.8), pbr.base_color_factor[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.2), pbr.base_color_factor[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.1), pbr.base_color_factor[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.5), pbr.metallic_factor, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.25), pbr.roughness_factor, 1e-4);
}

fn glbJsonChunk(bytes: []const u8) []const u8 {
    const json_len = std.mem.readInt(u32, bytes[12..16], .little);
    return bytes[20 .. 20 + json_len];
}

fn glbBinPayload(bytes: []const u8) []const u8 {
    const json_len = std.mem.readInt(u32, bytes[12..16], .little);
    const bin_start = 20 + json_len;
    const bin_len = std.mem.readInt(u32, bytes[bin_start..][0..4], .little);
    return bytes[bin_start + 8 .. bin_start + 8 + bin_len];
}

// Index accessor count + index buffer-view range from the JSON chunk
// (the exporter always writes the index accessor/view first).
fn glbIndexMeta(allocator: std.mem.Allocator, bytes: []const u8) !struct { count: i64, offset: i64, length: i64 } {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, glbJsonChunk(bytes), .{});
    defer parsed.deinit();
    const accessors = parsed.value.object.get("accessors").?.array;
    const views = parsed.value.object.get("bufferViews").?.array;
    return .{
        .count = accessors.items[0].object.get("count").?.integer,
        .offset = views.items[0].object.get("byteOffset").?.integer,
        .length = views.items[0].object.get("byteLength").?.integer,
    };
}

test "GLB exporter drops out-of-range triangles and trailing tails" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var positions = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(1, 0, 0),
        Vec3.new(0, 1, 0),
    };

    // Index 3 is out of range for 3 vertices: the triangle is dropped
    // (same contract as the OBJ/STL exporters), so no index >= 3 reaches
    // the BIN chunk (repro: export wrote vertex index 3).
    var bad_idx = [_]u32{ 0, 1, 3 };
    var bad = Mesh{
        .name = "BadTri",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = bad_idx.len,
        .cpu_positions = &positions,
        .cpu_indices = &bad_idx,
    };
    const bad_bytes = try writeGlbMeshAlloc(allocator, &bad, .{
        .emit_materials = false,
        .apply_world_transform = false,
    });
    defer allocator.free(bad_bytes);
    const bad_meta = try glbIndexMeta(allocator, bad_bytes);
    try testing.expectEqual(@as(i64, 0), bad_meta.count);
    try testing.expectEqual(@as(i64, 0), bad_meta.length);

    // Trailing index tail (< 3) is ignored: [0,1,2,1] exports one triangle.
    var tail_idx = [_]u32{ 0, 1, 2, 1 };
    var tail = Mesh{
        .name = "TailTri",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = tail_idx.len,
        .cpu_positions = &positions,
        .cpu_indices = &tail_idx,
    };
    const tail_bytes = try writeGlbMeshAlloc(allocator, &tail, .{
        .emit_materials = false,
        .apply_world_transform = false,
    });
    defer allocator.free(tail_bytes);
    const tail_meta = try glbIndexMeta(allocator, tail_bytes);
    try testing.expectEqual(@as(i64, 3), tail_meta.count);
    // Every stored index is in range (3 verts -> u16 path at BIN start).
    const bin = glbBinPayload(tail_bytes);
    const idx_bytes = bin[@intCast(tail_meta.offset)..@intCast(tail_meta.offset + tail_meta.length)];
    try testing.expectEqual(@as(usize, 6), idx_bytes.len);
    for ([3]usize{ 0, 1, 2 }, 0..) |want, k| {
        const got = std.mem.readInt(u16, idx_bytes[k * 2 ..][0..2], .little);
        try testing.expectEqual(@as(u16, @intCast(want)), got);
        try testing.expect(got < 3);
    }
}
