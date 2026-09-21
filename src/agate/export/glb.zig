//! Mesh -> glTF 2.0 Binary (.glb) exporter (CPU geometry only, no Scene dependency).
//!
//! Exports `Mesh.cpu_positions` / `Mesh.cpu_indices` as standard binary glTF 2.0 (.glb).
//! Meshes without CPU geometry (empty `cpu_positions`) are skipped.
//!
//! Standard glTF 2.0 container:
//! - 12-byte GLB header (magic "glTF", version 2, total length)
//! - JSON chunk (chunkType 0x4E4F534A, padded with spaces to 4-byte alignment)
//! - BIN chunk (chunkType 0x004E4942, padded with zeros to 4-byte alignment)
//!
//! Vertex attributes:
//! - POSITION: VEC3 FLOAT (componentType 5126), includes required min/max bounds
//! - NORMAL: VEC3 FLOAT (componentType 5126), smooth averaged face normals
//! - Indices: SCALAR UNSIGNED_SHORT (5123) or UNSIGNED_INT (5125)
//!
//! Materials:
//! - With `emit_materials`, Standard and PBR materials export with baseColorFactor,
//!   metallicFactor, and roughnessFactor.
//!
//! Transforms:
//! - With `apply_world_transform == true`, vertex positions are pre-transformed
//!   by `mesh.getWorldMatrix()`.

const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;

const Mesh = @import("../mesh.zig").Mesh;
const Material = @import("../material.zig").Material;
const c = @import("../c.zig").c;

pub const GlbExportOptions = struct {
    visible_only: bool = false,
    apply_world_transform: bool = false,
    emit_materials: bool = true,
};

const GLB_MAGIC: u32 = 0x46546C67; // "glTF"
const GLB_VERSION: u32 = 2;
const CHUNK_TYPE_JSON: u32 = 0x4E4F534A; // "JSON"
const CHUNK_TYPE_BIN: u32 = 0x004E4942; // "BIN\0"

const COMPONENT_UNSIGNED_SHORT: u32 = 5123;
const COMPONENT_UNSIGNED_INT: u32 = 5125;
const COMPONENT_FLOAT: u32 = 5126;

const TARGET_ARRAY_BUFFER: u32 = 34962;
const TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;

fn shouldExport(mesh: *const Mesh, options: GlbExportOptions) bool {
    if (options.visible_only and !mesh.is_visible) return false;
    if (mesh.cpu_positions.len == 0) return false;
    return true;
}

fn writeJsonString(buf: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, str: []const u8) !void {
    try buf.append(allocator, '"');
    for (str) |ch| {
        switch (ch) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => {
                if (ch < 32) {
                    try buf.print(allocator, "\\u{x:0>4}", .{ch});
                } else {
                    try buf.append(allocator, ch);
                }
            },
        }
    }
    try buf.append(allocator, '"');
}

const ExportMaterial = struct {
    name: []const u8,
    base_color: [4]f32,
    metallic: f32,
    roughness: f32,
};

fn extractMaterial(mat_opt: ?Material) ?ExportMaterial {
    const m = mat_opt orelse return null;
    return switch (m) {
        .pbr => |p| .{
            .name = p.name,
            .base_color = .{ p.albedo_color.r, p.albedo_color.g, p.albedo_color.b, p.alpha },
            .metallic = p.metallic,
            .roughness = p.roughness,
        },
        .standard => |s| .{
            .name = s.name,
            .base_color = .{ s.diffuse_color.r, s.diffuse_color.g, s.diffuse_color.b, s.alpha },
            .metallic = 0.0,
            .roughness = 0.5,
        },
        else => null,
    };
}

const BufferViewInfo = struct {
    byte_offset: usize,
    byte_length: usize,
    target: u32,
};

const AccessorInfo = struct {
    buffer_view: usize,
    component_type: u32,
    count: usize,
    type_str: []const u8,
    min_bounds: ?Vec3 = null,
    max_bounds: ?Vec3 = null,
};

const PrimitiveInfo = struct {
    pos_accessor: usize,
    norm_accessor: usize,
    index_accessor: usize,
    material_index: ?usize,
};

const MeshInfo = struct {
    name: []const u8,
    primitive: PrimitiveInfo,
};

/// Computes smooth averaged vertex normals from positions and triangle indices.
fn computeNormals(
    allocator: std.mem.Allocator,
    positions: []const Vec3,
    indices: []const u32,
) ![]Vec3 {
    const normals = try allocator.alloc(Vec3, positions.len);
    @memset(normals, Vec3.zero);

    var i: usize = 0;
    while (i + 2 < indices.len) : (i += 3) {
        const ia = indices[i];
        const ib = indices[i + 1];
        const ic = indices[i + 2];
        if (ia >= positions.len or ib >= positions.len or ic >= positions.len) continue;

        const p0 = positions[ia];
        const p1 = positions[ib];
        const p2 = positions[ic];

        const fn_norm = p1.sub(p0).cross(p2.sub(p0));
        normals[ia] = normals[ia].add(fn_norm);
        normals[ib] = normals[ib].add(fn_norm);
        normals[ic] = normals[ic].add(fn_norm);
    }

    for (normals) |*n| {
        if (n.lengthSq() > 1e-12) {
            n.* = n.normalize();
        } else {
            n.* = Vec3.up;
        }
    }
    return normals;
}

/// Exports a list of meshes into a self-contained glTF 2.0 binary (.glb) buffer.
/// Returned slice is owned by the caller (free with `allocator`).
pub fn writeGlbAlloc(
    allocator: std.mem.Allocator,
    meshes: []const *Mesh,
    options: GlbExportOptions,
) ![]u8 {
    var bin_data: std.ArrayListUnmanaged(u8) = .empty;
    defer bin_data.deinit(allocator);

    var buffer_views: std.ArrayListUnmanaged(BufferViewInfo) = .empty;
    defer buffer_views.deinit(allocator);

    var accessors: std.ArrayListUnmanaged(AccessorInfo) = .empty;
    defer accessors.deinit(allocator);

    var export_meshes: std.ArrayListUnmanaged(MeshInfo) = .empty;
    defer export_meshes.deinit(allocator);

    var materials: std.ArrayListUnmanaged(ExportMaterial) = .empty;
    defer materials.deinit(allocator);

    for (meshes) |mesh| {
        if (!shouldExport(mesh, options)) continue;

        // 1. Resolve positions & compute bounds
        const world: ?Mat4 = if (options.apply_world_transform) mesh.getWorldMatrix() else null;
        const positions = try allocator.alloc(Vec3, mesh.cpu_positions.len);
        defer allocator.free(positions);

        var min_pos = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
        var max_pos = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));

        for (mesh.cpu_positions, 0..) |p, i| {
            const pt = if (world) |w| w.transformPoint(p) else p;
            positions[i] = pt;
            min_pos.x = @min(min_pos.x, pt.x);
            min_pos.y = @min(min_pos.y, pt.y);
            min_pos.z = @min(min_pos.z, pt.z);
            max_pos.x = @max(max_pos.x, pt.x);
            max_pos.y = @max(max_pos.y, pt.y);
            max_pos.z = @max(max_pos.z, pt.z);
        }

        if (positions.len == 0) {
            min_pos = Vec3.zero;
            max_pos = Vec3.zero;
        }

        // 2. Resolve indices
        var owned_indices: ?[]u32 = null;
        defer if (owned_indices) |idx| allocator.free(idx);

        const indices: []const u32 = if (mesh.cpu_indices.len > 0)
            mesh.cpu_indices
        else blk: {
            const seq = try allocator.alloc(u32, positions.len);
            for (seq, 0..) |*v, idx| {
                v.* = @intCast(idx);
            }
            owned_indices = seq;
            break :blk seq;
        };

        // 3. Compute normals
        const normals = try computeNormals(allocator, positions, indices);
        defer allocator.free(normals);

        // 4. Resolve Material
        var mat_idx: ?usize = null;
        if (options.emit_materials and mesh.material != null) {
            if (extractMaterial(mesh.material)) |mat| {
                // Find existing matching material or append
                var found: ?usize = null;
                for (materials.items, 0..) |existing, mi| {
                    if (std.mem.eql(u8, existing.name, mat.name)) {
                        found = mi;
                        break;
                    }
                }
                if (found) |mi| {
                    mat_idx = mi;
                } else {
                    mat_idx = materials.items.len;
                    try materials.append(allocator, mat);
                }
            }
        }

        // 5. Append Indices to BIN
        // Align bin offset to 4 bytes
        while (bin_data.items.len % 4 != 0) {
            try bin_data.append(allocator, 0);
        }
        const idx_bv_offset = bin_data.items.len;
        const use_u16 = (positions.len <= 65536);
        const idx_component = if (use_u16) COMPONENT_UNSIGNED_SHORT else COMPONENT_UNSIGNED_INT;

        if (use_u16) {
            for (indices) |idx| {
                const u: u16 = @intCast(idx);
                const bytes = std.mem.asBytes(&u);
                try bin_data.appendSlice(allocator, bytes);
            }
        } else {
            for (indices) |idx| {
                const bytes = std.mem.asBytes(&idx);
                try bin_data.appendSlice(allocator, bytes);
            }
        }
        const idx_bv_len = bin_data.items.len - idx_bv_offset;
        const idx_bv_idx = buffer_views.items.len;
        try buffer_views.append(allocator, .{
            .byte_offset = idx_bv_offset,
            .byte_length = idx_bv_len,
            .target = TARGET_ELEMENT_ARRAY_BUFFER,
        });

        const idx_acc_idx = accessors.items.len;
        try accessors.append(allocator, .{
            .buffer_view = idx_bv_idx,
            .component_type = idx_component,
            .count = indices.len,
            .type_str = "SCALAR",
        });

        // 6. Append Positions to BIN
        while (bin_data.items.len % 4 != 0) {
            try bin_data.append(allocator, 0);
        }
        const pos_bv_offset = bin_data.items.len;
        for (positions) |p| {
            const arr = [3]f32{ p.x, p.y, p.z };
            try bin_data.appendSlice(allocator, std.mem.sliceAsBytes(&arr));
        }
        const pos_bv_len = bin_data.items.len - pos_bv_offset;
        const pos_bv_idx = buffer_views.items.len;
        try buffer_views.append(allocator, .{
            .byte_offset = pos_bv_offset,
            .byte_length = pos_bv_len,
            .target = TARGET_ARRAY_BUFFER,
        });

        const pos_acc_idx = accessors.items.len;
        try accessors.append(allocator, .{
            .buffer_view = pos_bv_idx,
            .component_type = COMPONENT_FLOAT,
            .count = positions.len,
            .type_str = "VEC3",
            .min_bounds = min_pos,
            .max_bounds = max_pos,
        });

        // 7. Append Normals to BIN
        while (bin_data.items.len % 4 != 0) {
            try bin_data.append(allocator, 0);
        }
        const norm_bv_offset = bin_data.items.len;
        for (normals) |n| {
            const arr = [3]f32{ n.x, n.y, n.z };
            try bin_data.appendSlice(allocator, std.mem.sliceAsBytes(&arr));
        }
        const norm_bv_len = bin_data.items.len - norm_bv_offset;
        const norm_bv_idx = buffer_views.items.len;
        try buffer_views.append(allocator, .{
            .byte_offset = norm_bv_offset,
            .byte_length = norm_bv_len,
            .target = TARGET_ARRAY_BUFFER,
        });

        const norm_acc_idx = accessors.items.len;
        try accessors.append(allocator, .{
            .buffer_view = norm_bv_idx,
            .component_type = COMPONENT_FLOAT,
            .count = normals.len,
            .type_str = "VEC3",
        });

        try export_meshes.append(allocator, .{
            .name = mesh.name,
            .primitive = .{
                .pos_accessor = pos_acc_idx,
                .norm_accessor = norm_acc_idx,
                .index_accessor = idx_acc_idx,
                .material_index = mat_idx,
            },
        });
    }

    // Pad BIN data to 4-byte boundary
    while (bin_data.items.len % 4 != 0) {
        try bin_data.append(allocator, 0);
    }

    // Build JSON metadata
    var json_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer json_buf.deinit(allocator);
    try json_buf.appendSlice(allocator, "{\"asset\":{\"generator\":\"Agate Engine\",\"version\":\"2.0\"},\"scene\":0,\"scenes\":[{\"nodes\":[");
    for (0..export_meshes.items.len) |ni| {
        if (ni > 0) try json_buf.append(allocator, ',');
        try json_buf.print(allocator, "{d}", .{ni});
    }
    try json_buf.appendSlice(allocator, "]}],\"nodes\":[");
    for (export_meshes.items, 0..) |em, ni| {
        if (ni > 0) try json_buf.append(allocator, ',');
        try json_buf.appendSlice(allocator, "{\"mesh\":");
        try json_buf.print(allocator, "{d},\"name\":", .{ni});
        try writeJsonString(&json_buf, allocator, em.name);
        try json_buf.append(allocator, '}');
    }
    try json_buf.appendSlice(allocator, "],\"meshes\":[");
    for (export_meshes.items, 0..) |em, mi| {
        if (mi > 0) try json_buf.append(allocator, ',');
        try json_buf.appendSlice(allocator, "{\"name\":");
        try writeJsonString(&json_buf, allocator, em.name);
        try json_buf.appendSlice(allocator, ",\"primitives\":[{\"attributes\":{\"POSITION\":");
        try json_buf.print(allocator, "{d},\"NORMAL\":{d}", .{ em.primitive.pos_accessor, em.primitive.norm_accessor });
        try json_buf.appendSlice(allocator, "},\"indices\":");
        try json_buf.print(allocator, "{d}", .{em.primitive.index_accessor});
        if (em.primitive.material_index) |mati| {
            try json_buf.appendSlice(allocator, ",\"material\":");
            try json_buf.print(allocator, "{d}", .{mati});
        }
        try json_buf.appendSlice(allocator, "}]}");
    }
    try json_buf.appendSlice(allocator, "]");

    if (materials.items.len > 0) {
        try json_buf.appendSlice(allocator, ",\"materials\":[");
        for (materials.items, 0..) |mat, mati| {
            if (mati > 0) try json_buf.append(allocator, ',');
            try json_buf.appendSlice(allocator, "{\"name\":");
            try writeJsonString(&json_buf, allocator, mat.name);
            try json_buf.appendSlice(allocator, ",\"pbrMetallicRoughness\":{\"baseColorFactor\":[");
            try json_buf.print(allocator, "{d:.5},{d:.5},{d:.5},{d:.5}", .{
                mat.base_color[0], mat.base_color[1], mat.base_color[2], mat.base_color[3],
            });
            try json_buf.print(allocator, "],\"metallicFactor\":{d:.4},\"roughnessFactor\":{d:.4}", .{
                mat.metallic, mat.roughness,
            });
            try json_buf.appendSlice(allocator, "}}");
        }
        try json_buf.appendSlice(allocator, "]");
    }

    try json_buf.appendSlice(allocator, ",\"accessors\":[");
    for (accessors.items, 0..) |acc, ai| {
        if (ai > 0) try json_buf.append(allocator, ',');
        try json_buf.appendSlice(allocator, "{\"bufferView\":");
        try json_buf.print(allocator, "{d},\"componentType\":{d},\"count\":{d},\"type\":", .{
            acc.buffer_view, acc.component_type, acc.count,
        });
        try writeJsonString(&json_buf, allocator, acc.type_str);
        if (acc.min_bounds) |mn| {
            const mx = acc.max_bounds.?;
            try json_buf.print(allocator, ",\"max\":[{d:.5},{d:.5},{d:.5}],\"min\":[{d:.5},{d:.5},{d:.5}]", .{
                mx.x, mx.y, mx.z, mn.x, mn.y, mn.z,
            });
        }
        try json_buf.append(allocator, '}');
    }
    try json_buf.appendSlice(allocator, "],\"bufferViews\":[");
    for (buffer_views.items, 0..) |bv, bvi| {
        if (bvi > 0) try json_buf.append(allocator, ',');
        try json_buf.appendSlice(allocator, "{\"buffer\":0,\"byteLength\":");
        try json_buf.print(allocator, "{d},\"byteOffset\":{d},\"target\":{d}}}", .{
            bv.byte_length, bv.byte_offset, bv.target,
        });
    }
    try json_buf.appendSlice(allocator, "],\"buffers\":[{\"byteLength\":");
    try json_buf.print(allocator, "{d}", .{bin_data.items.len});
    try json_buf.appendSlice(allocator, "}]}");

    // Pad JSON chunk with spaces to 4-byte boundary
    while (json_buf.items.len % 4 != 0) {
        try json_buf.append(allocator, ' ');
    }

    // Compute total GLB size
    const json_chunk_len: u32 = @intCast(json_buf.items.len);
    const bin_chunk_len: u32 = @intCast(bin_data.items.len);
    const total_length: u32 = 12 + 8 + json_chunk_len + 8 + bin_chunk_len;

    var out = try allocator.alloc(u8, total_length);
    errdefer allocator.free(out);

    // 1. Header (12 bytes)
    std.mem.writeInt(u32, out[0..4], GLB_MAGIC, .little);
    std.mem.writeInt(u32, out[4..8], GLB_VERSION, .little);
    std.mem.writeInt(u32, out[8..12], total_length, .little);

    // 2. Chunk 0: JSON (8 bytes header + payload)
    std.mem.writeInt(u32, out[12..16], json_chunk_len, .little);
    std.mem.writeInt(u32, out[16..20], CHUNK_TYPE_JSON, .little);
    @memcpy(out[20 .. 20 + json_chunk_len], json_buf.items);

    // 3. Chunk 1: BIN (8 bytes header + payload)
    const bin_start = 20 + json_chunk_len;
    std.mem.writeInt(u32, out[bin_start..][0..4], bin_chunk_len, .little);
    std.mem.writeInt(u32, out[bin_start + 4 ..][0..4], CHUNK_TYPE_BIN, .little);
    @memcpy(out[bin_start + 8 .. bin_start + 8 + bin_chunk_len], bin_data.items);

    return out;
}

/// Convenience function to export a single mesh to a GLB buffer.
pub fn writeGlbMeshAlloc(
    allocator: std.mem.Allocator,
    mesh: *Mesh,
    options: GlbExportOptions,
) ![]u8 {
    const list = [_]*Mesh{mesh};
    return writeGlbAlloc(allocator, &list, options);
}

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

    const PBRMaterial = @import("../material.zig").PBRMaterial;
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
