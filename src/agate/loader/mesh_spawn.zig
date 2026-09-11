const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const c = @import("../c.zig").c;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;

const Scene = @import("../scene.zig").Scene;
const Mesh = @import("../mesh.zig").Mesh;
const Vertex = @import("../mesh.zig").Vertex;
const computeTangents = @import("../mesh.zig").computeTangents;
const Material = @import("../material.zig").Material;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;

pub fn spawnMeshes(
    scene: *Scene,
    gltf: *c.cgltf_data,
    materials: []const ?Material,
    skeletons: []?*Skeleton,
    spawned_meshes: *std.ArrayList(*Mesh),
    node_mesh_start: []usize,
    node_mesh_count: []usize,
) !void {
    if (gltf.nodes_count > 0) {
        for (0..gltf.nodes_count) |node_idx| {
            const node = &gltf.nodes[node_idx];
            if (node.mesh == null) continue;
            const src_mesh = node.mesh.?;
            const mesh_name = if (node.name != null)
                std.mem.span(node.name)
            else if (src_mesh.*.name != null)
                std.mem.span(src_mesh.*.name)
            else
                "glb_mesh";

            var world_mat: [16]f32 = undefined;
            c.cgltf_node_transform_world(node, &world_mat);
            const base_matrix = Mat4{ .m = world_mat };
            const range_start = spawned_meshes.items.len;

            var node_skeleton: ?*Skeleton = null;
            if (node.skin) |n_skin| {
                for (0..gltf.skins_count) |si| {
                    if (&gltf.skins[si] == n_skin) {
                        node_skeleton = skeletons[si];
                        break;
                    }
                }
            }

            for (0..src_mesh.*.primitives_count) |prim_idx| {
                const prim: *const c.cgltf_primitive = @ptrCast(&src_mesh.*.primitives[prim_idx]);
                if (try parsePrimitive(scene, gltf, prim, mesh_name, base_matrix, materials, node_skeleton)) |mesh_obj| {
                    try scene.meshes.append(scene.allocator, mesh_obj);
                    try spawned_meshes.append(scene.allocator, mesh_obj);
                }
            }
            node_mesh_start[node_idx] = range_start;
            node_mesh_count[node_idx] = spawned_meshes.items.len - range_start;
        }
    }

    // Fallback for files without node hierarchy
    if (spawned_meshes.items.len == 0) {
        for (0..gltf.meshes_count) |mesh_idx| {
            const src_mesh = &gltf.meshes[mesh_idx];
            const mesh_name = if (src_mesh.name != null)
                std.mem.span(src_mesh.name)
            else
                "glb_mesh";

            for (0..src_mesh.primitives_count) |prim_idx| {
                const prim: *const c.cgltf_primitive = @ptrCast(&src_mesh.primitives[prim_idx]);
                if (try parsePrimitive(scene, gltf, prim, mesh_name, Mat4.identity, materials, null)) |mesh_obj| {
                    try scene.meshes.append(scene.allocator, mesh_obj);
                    try spawned_meshes.append(scene.allocator, mesh_obj);
                }
            }
        }
    }
}

pub fn parsePrimitive(
    scene: *Scene,
    gltf: *c.cgltf_data,
    prim: *const c.cgltf_primitive,
    mesh_name: []const u8,
    base_matrix: Mat4,
    materials: []const ?Material,
    skeleton: ?*Skeleton,
) !?*Mesh {
    if (prim.type != c.cgltf_primitive_type_triangles) return null;

    var pos_accessor: ?*c.cgltf_accessor = null;
    var norm_accessor: ?*c.cgltf_accessor = null;
    var col_accessor: ?*c.cgltf_accessor = null;
    var uv_accessor: ?*c.cgltf_accessor = null;
    var tan_accessor: ?*c.cgltf_accessor = null;
    var joints_accessor: ?*c.cgltf_accessor = null;
    var weights_accessor: ?*c.cgltf_accessor = null;

    for (0..prim.attributes_count) |attr_idx| {
        const attr = &prim.attributes[attr_idx];
        switch (attr.type) {
            c.cgltf_attribute_type_position => pos_accessor = attr.data,
            c.cgltf_attribute_type_normal => norm_accessor = attr.data,
            c.cgltf_attribute_type_color => col_accessor = attr.data,
            c.cgltf_attribute_type_tangent => tan_accessor = attr.data,
            c.cgltf_attribute_type_texcoord => {
                if (attr.index == 0) uv_accessor = attr.data;
            },
            c.cgltf_attribute_type_joints => {
                if (attr.index == 0) joints_accessor = attr.data;
            },
            c.cgltf_attribute_type_weights => {
                if (attr.index == 0) weights_accessor = attr.data;
            },
            else => {},
        }
    }

    if (pos_accessor == null) return null;

    const vert_count = pos_accessor.?.count;
    const vertices = try scene.allocator.alloc(Vertex, vert_count);
    defer scene.allocator.free(vertices);

    for (0..vert_count) |i| {
        var p: [3]f32 = .{ 0, 0, 0 };
        _ = c.cgltf_accessor_read_float(pos_accessor.?, i, &p, 3);

        var n: [3]f32 = .{ 0, 1, 0 };
        if (norm_accessor) |na| {
            _ = c.cgltf_accessor_read_float(na, i, &n, 3);
        }

        var col: [4]f32 = .{ 1, 1, 1, 1 };
        if (col_accessor) |ca| {
            _ = c.cgltf_accessor_read_float(ca, i, &col, 4);
        }

        var uv: [2]f32 = .{ 0, 0 };
        if (uv_accessor) |ua| {
            _ = c.cgltf_accessor_read_float(ua, i, &uv, 2);
        }

        var tan: [4]f32 = .{ 1, 0, 0, 1 };
        if (tan_accessor) |ta| {
            _ = c.cgltf_accessor_read_float(ta, i, &tan, 4);
        }

        var j_val: [4]f32 = .{ 0, 0, 0, 0 };
        if (joints_accessor) |ja| {
            _ = c.cgltf_accessor_read_float(ja, i, &j_val, 4);
        }

        var w_val: [4]f32 = .{ 1, 0, 0, 0 };
        if (weights_accessor) |wa| {
            _ = c.cgltf_accessor_read_float(wa, i, &w_val, 4);
        }

        vertices[i] = .{
            .position = p,
            .normal = n,
            .color = col,
            .uv = uv,
            .tangent = tan,
            .joints = j_val,
            .weights = w_val,
        };
    }

    var index_count: u32 = 0;
    var index_type: sg.IndexType = .UINT16;
    var ibuf: sg.Buffer = .{};

    if (prim.indices) |ind_accessor| {
        index_count = @intCast(ind_accessor.*.count);

        if (ind_accessor.*.count > 65535 or vert_count > 65535) {
            index_type = .UINT32;
            const indices = try scene.allocator.alloc(u32, ind_accessor.*.count);
            defer scene.allocator.free(indices);

            for (0..ind_accessor.*.count) |i| {
                indices[i] = @intCast(c.cgltf_accessor_read_index(ind_accessor, i));
            }

            if (tan_accessor == null) {
                computeTangents(vertices, indices, null);
            }

            ibuf = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(indices),
            });
        } else {
            index_type = .UINT16;
            const indices = try scene.allocator.alloc(u16, ind_accessor.*.count);
            defer scene.allocator.free(indices);

            for (0..ind_accessor.*.count) |i| {
                indices[i] = @intCast(c.cgltf_accessor_read_index(ind_accessor, i));
            }

            if (tan_accessor == null) {
                computeTangents(vertices, null, indices);
            }

            ibuf = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(indices),
            });
        }
    } else {
        index_count = @intCast(vert_count);
        if (vert_count > 65535) {
            index_type = .UINT32;
            const indices = try scene.allocator.alloc(u32, vert_count);
            defer scene.allocator.free(indices);
            for (0..vert_count) |i| {
                indices[i] = @intCast(i);
            }
            if (tan_accessor == null) {
                computeTangents(vertices, indices, null);
            }
            ibuf = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(indices),
            });
        } else {
            index_type = .UINT16;
            const indices = try scene.allocator.alloc(u16, vert_count);
            defer scene.allocator.free(indices);
            for (0..vert_count) |i| {
                indices[i] = @intCast(i);
            }
            if (tan_accessor == null) {
                computeTangents(vertices, null, indices);
            }
            ibuf = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(indices),
            });
        }
    }

    const vbuf = sg.makeBuffer(.{
        .data = sg.asRange(vertices),
    });

    var local_box = BoundingBox.zero;
    if (pos_accessor) |pos_acc| {
        if (pos_acc.*.has_min != 0 and pos_acc.*.has_max != 0) {
            local_box = BoundingBox.init(
                Vec3.new(pos_acc.*.min[0], pos_acc.*.min[1], pos_acc.*.min[2]),
                Vec3.new(pos_acc.*.max[0], pos_acc.*.max[1], pos_acc.*.max[2]),
            );
        } else if (vertices.len > 0) {
            var min_p = Vec3.new(vertices[0].position[0], vertices[0].position[1], vertices[0].position[2]);
            var max_p = min_p;
            for (vertices) |v| {
                min_p.x = @min(min_p.x, v.position[0]);
                min_p.y = @min(min_p.y, v.position[1]);
                min_p.z = @min(min_p.z, v.position[2]);
                max_p.x = @max(max_p.x, v.position[0]);
                max_p.y = @max(max_p.y, v.position[1]);
                max_p.z = @max(max_p.z, v.position[2]);
            }
            local_box = BoundingBox.init(min_p, max_p);
        }
    }

    const owned_name = try scene.allocator.dupe(u8, mesh_name);
    const mesh_obj = try scene.allocator.create(Mesh);
    mesh_obj.* = .{
        .name = owned_name,
        .owns_name = true,
        .vertex_buffer = vbuf,
        .index_buffer = ibuf,
        .index_count = index_count,
        .index_type = index_type,
        .skeleton = skeleton,
        .base_matrix = base_matrix,
        .local_bounding_box = local_box,
    };

    if (prim.material) |pm| {
        for (0..gltf.materials_count) |mat_i| {
            if (&gltf.materials[mat_i] == pm) {
                mesh_obj.material = materials[mat_i];
                break;
            }
        }
    }

    // Retain CPU geometry so physics colliders (convex hull / triangle mesh)
    // can be built from loaded models.
    const cpu_positions = try scene.allocator.alloc(Vec3, vert_count);
    for (vertices, 0..) |v, i| {
        cpu_positions[i] = Vec3.new(v.position[0], v.position[1], v.position[2]);
    }
    const cpu_indices = try scene.allocator.alloc(u32, index_count);
    for (0..index_count) |i| {
        cpu_indices[i] = if (prim.indices) |ind_accessor|
            @intCast(c.cgltf_accessor_read_index(ind_accessor, i))
        else
            @intCast(i);
    }
    mesh_obj.cpu_positions = cpu_positions;
    mesh_obj.cpu_indices = cpu_indices;

    return mesh_obj;
}
