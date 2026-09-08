const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const c = @import("../c.zig").c;
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;

const Scene = @import("../scene.zig").Scene;
const Mesh = @import("../mesh.zig").Mesh;
const Vertex = @import("../mesh.zig").Vertex;
const StandardMaterial = @import("../material.zig").StandardMaterial;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const Material = @import("../material.zig").Material;
const Texture = @import("../texture.zig").Texture;

pub const SceneLoader = struct {
    pub fn appendGlb(scene: *Scene, file_path: []const u8) ![]*Mesh {
        const path_z = try scene.allocator.dupeZ(u8, file_path);
        defer scene.allocator.free(path_z);

        var options = std.mem.zeroes(c.cgltf_options);
        var data: ?*c.cgltf_data = null;

        const parse_res = c.cgltf_parse_file(&options, path_z.ptr, &data);
        if (parse_res != c.cgltf_result_success or data == null) {
            return error.GltfParseFailed;
        }
        defer c.cgltf_free(data);

        // Load binary buffers (in GLB they are inside the buffer itself)
        const load_buf_res = c.cgltf_load_buffers(&options, data, path_z.ptr);
        if (load_buf_res != c.cgltf_result_success) {
            return error.GltfLoadBuffersFailed;
        }

        const gltf = data.?;

        // 1. Parse materials
        var materials = try scene.allocator.alloc(?Material, gltf.materials_count);
        defer scene.allocator.free(materials);
        @memset(materials, null);

        for (0..gltf.materials_count) |i| {
            const src_mat = &gltf.materials[i];
            const mat_name = if (src_mat.name != null)
                std.mem.span(src_mat.name)
            else
                "glb_material";

            if (src_mat.has_pbr_metallic_roughness != 0) {
                const pbr_mat = try scene.createPBRMaterial(mat_name);
                const pbr = &src_mat.pbr_metallic_roughness;
                pbr_mat.albedo_color = Color3.new(
                    pbr.base_color_factor[0],
                    pbr.base_color_factor[1],
                    pbr.base_color_factor[2],
                );
                pbr_mat.alpha = pbr.base_color_factor[3];
                pbr_mat.metallic = pbr.metallic_factor;
                pbr_mat.roughness = pbr.roughness_factor;

                if (pbr.base_color_texture.texture != null) {
                    const tex = pbr.base_color_texture.texture;
                    if (tex.*.image != null) {
                        const img = tex.*.image;
                        if (img.*.buffer_view != null) {
                            const bv = img.*.buffer_view.?;
                            const raw_buf = @as([*]const u8, @ptrCast(bv.*.buffer.*.data));
                            const img_data = (raw_buf + bv.*.offset)[0..bv.*.size];

                            if (Texture.fromMemory(img_data, .{})) |loaded_tex| {
                                pbr_mat.albedo_texture = loaded_tex;
                            } else |_| {}
                        }
                    }
                }
                materials[i] = .{ .pbr = pbr_mat };
            } else {
                const std_mat = try scene.createStandardMaterial(mat_name);
                materials[i] = .{ .standard = std_mat };
            }
        }

        // 2. Parse meshes and primitives (preserving glTF node transforms)
        var spawned_meshes = std.ArrayList(*Mesh).empty;

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

                for (0..src_mesh.*.primitives_count) |prim_idx| {
                    const prim: *const c.cgltf_primitive = @ptrCast(&src_mesh.*.primitives[prim_idx]);
                    if (try parsePrimitive(scene, gltf, prim, mesh_name, base_matrix, materials)) |mesh_obj| {
                        try scene.meshes.append(scene.allocator, mesh_obj);
                        try spawned_meshes.append(scene.allocator, mesh_obj);
                    }
                }
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
                    if (try parsePrimitive(scene, gltf, prim, mesh_name, Mat4.identity, materials)) |mesh_obj| {
                        try scene.meshes.append(scene.allocator, mesh_obj);
                        try spawned_meshes.append(scene.allocator, mesh_obj);
                    }
                }
            }
        }

        return spawned_meshes.toOwnedSlice(scene.allocator);
    }

    fn parsePrimitive(
        scene: *Scene,
        gltf: *c.cgltf_data,
        prim: *const c.cgltf_primitive,
        mesh_name: []const u8,
        base_matrix: Mat4,
        materials: []const ?Material,
    ) !?*Mesh {
        if (prim.type != c.cgltf_primitive_type_triangles) return null;

        var pos_accessor: ?*c.cgltf_accessor = null;
        var norm_accessor: ?*c.cgltf_accessor = null;
        var col_accessor: ?*c.cgltf_accessor = null;
        var uv_accessor: ?*c.cgltf_accessor = null;

        for (0..prim.attributes_count) |attr_idx| {
            const attr = &prim.attributes[attr_idx];
            switch (attr.type) {
                c.cgltf_attribute_type_position => pos_accessor = attr.data,
                c.cgltf_attribute_type_normal => norm_accessor = attr.data,
                c.cgltf_attribute_type_color => col_accessor = attr.data,
                c.cgltf_attribute_type_texcoord => {
                    if (attr.index == 0) uv_accessor = attr.data;
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

            vertices[i] = .{
                .position = p,
                .normal = n,
                .color = col,
                .uv = uv,
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

        const mesh_obj = try scene.allocator.create(Mesh);
        mesh_obj.* = .{
            .name = mesh_name,
            .vertex_buffer = vbuf,
            .index_buffer = ibuf,
            .index_count = index_count,
            .index_type = index_type,
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

        return mesh_obj;
