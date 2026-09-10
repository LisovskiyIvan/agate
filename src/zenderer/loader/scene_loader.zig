const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const c = @import("../c.zig").c;
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Color3 = math.Color3;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;

const Scene = @import("../scene.zig").Scene;
const Mesh = @import("../mesh.zig").Mesh;
const Vertex = @import("../mesh.zig").Vertex;
const computeTangents = @import("../mesh.zig").computeTangents;
const StandardMaterial = @import("../material.zig").StandardMaterial;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const Material = @import("../material.zig").Material;
const Texture = @import("../texture.zig").Texture;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const AnimationGroup = @import("../animation/animation.zig").AnimationGroup;
const AnimationChannel = @import("../animation/animation.zig").AnimationChannel;
const AnimationSampler = @import("../animation/animation.zig").AnimationSampler;
const AnimationPath = @import("../animation/animation.zig").AnimationPath;
const AnimationInterpolation = @import("../animation/animation.zig").AnimationInterpolation;

pub const SceneLoader = struct {
    fn loadTextureFromView(
        scene: *Scene,
        gltf: *c.cgltf_data,
        image_cache: []?Texture,
        view: [*c]const c.cgltf_texture_view,
        base_dir: ?[]const u8,
    ) ?Texture {
        if (view == null) return null;
        if (view.*.texture == null) return null;
        const tex = view.*.texture.?;
        if (tex.*.image == null) return null;
        const img = tex.*.image.?;

        var img_idx: ?usize = null;
        for (0..gltf.images_count) |im_i| {
            if (&gltf.images[im_i] == img) {
                img_idx = im_i;
                break;
            }
        }

        if (img_idx) |idx| {
            if (image_cache[idx]) |cached| {
                return cached;
            }
        }

        var tex_options: Texture.Options = .{};
        if (tex.*.sampler) |smp| {
            switch (smp.*.wrap_s) {
                33071 => tex_options.wrap_u = .CLAMP_TO_EDGE,
                33648 => tex_options.wrap_u = .MIRRORED_REPEAT,
                10497 => tex_options.wrap_u = .REPEAT,
                else => {},
            }
            switch (smp.*.wrap_t) {
                33071 => tex_options.wrap_v = .CLAMP_TO_EDGE,
                33648 => tex_options.wrap_v = .MIRRORED_REPEAT,
                10497 => tex_options.wrap_v = .REPEAT,
                else => {},
            }
            switch (smp.*.mag_filter) {
                9728 => tex_options.mag_filter = .NEAREST,
                9729 => tex_options.mag_filter = .LINEAR,
                else => {},
            }
            switch (smp.*.min_filter) {
                9728, 9984, 9986 => tex_options.min_filter = .NEAREST,
                9729, 9985, 9987 => tex_options.min_filter = .LINEAR,
                else => {},
            }
        }

        // 1. Embedded buffer view (typical in GLB or embedded GLTF)
        if (img.*.buffer_view) |bv| {
            if (bv.*.buffer != null and bv.*.buffer.*.data != null) {
                const raw_buf = @as([*]const u8, @ptrCast(bv.*.buffer.*.data));
                const img_data = (raw_buf + bv.*.offset)[0..bv.*.size];
                if (Texture.fromMemory(scene.allocator, img_data, tex_options)) |loaded| {
                    if (img_idx) |idx| {
                        image_cache[idx] = loaded;
                    }
                    return loaded;
                } else |_| {}
            }
        }

        // 2. External URI (typical in standard GLTF with external textures)
        if (img.*.uri) |uri_c| {
            const uri = std.mem.span(uri_c);
            if (base_dir) |dir| {
                const full_path = std.fs.path.join(scene.allocator, &.{ dir, uri }) catch null;
                if (full_path) |fp| {
                    defer scene.allocator.free(fp);
                    if (Texture.fromFile(scene.allocator, fp, tex_options)) |loaded| {
                        if (img_idx) |idx| {
                            image_cache[idx] = loaded;
                        }
                        return loaded;
                    } else |_| {}
                }
            } else {
                if (Texture.fromFile(scene.allocator, uri, tex_options)) |loaded| {
                    if (img_idx) |idx| {
                        image_cache[idx] = loaded;
                    }
                    return loaded;
                } else |_| {}
            }
        }

        return null;
    }

    pub const appendGltf = appendGlb;

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

        // Load binary buffers (in GLB they are inside the buffer itself, in GLTF from .bin on disk)
        const load_buf_res = c.cgltf_load_buffers(&options, data, path_z.ptr);
        if (load_buf_res != c.cgltf_result_success) {
            return error.GltfLoadBuffersFailed;
        }

        const gltf = data.?;
        const base_dir = std.fs.path.dirname(file_path);

        // 1. Parse materials
        var materials = try scene.allocator.alloc(?Material, gltf.materials_count);
        defer scene.allocator.free(materials);
        @memset(materials, null);

        const image_cache = try scene.allocator.alloc(?Texture, gltf.images_count);
        defer scene.allocator.free(image_cache);
        @memset(image_cache, null);

        for (0..gltf.materials_count) |i| {
            const src_mat = &gltf.materials[i];
            const mat_name = if (src_mat.name != null)
                std.mem.span(src_mat.name)
            else
                "glb_material";

            const pbr_mat = try scene.createPBRMaterial(mat_name);

            if (src_mat.has_pbr_metallic_roughness != 0) {
                const pbr = &src_mat.pbr_metallic_roughness;
                pbr_mat.albedo_color = Color3.new(
                    pbr.base_color_factor[0],
                    pbr.base_color_factor[1],
                    pbr.base_color_factor[2],
                );
                pbr_mat.alpha = pbr.base_color_factor[3];
                pbr_mat.metallic = pbr.metallic_factor;
                pbr_mat.roughness = pbr.roughness_factor;

                pbr_mat.albedo_texture = loadTextureFromView(scene, gltf, image_cache, &pbr.base_color_texture, base_dir);
                pbr_mat.metallic_roughness_texture = loadTextureFromView(scene, gltf, image_cache, &pbr.metallic_roughness_texture, base_dir);
            }

            pbr_mat.normal_texture = loadTextureFromView(scene, gltf, image_cache, &src_mat.normal_texture, base_dir);
            pbr_mat.occlusion_texture = loadTextureFromView(scene, gltf, image_cache, &src_mat.occlusion_texture, base_dir);
            pbr_mat.occlusion_strength = src_mat.occlusion_texture.scale;

            pbr_mat.emissive_texture = loadTextureFromView(scene, gltf, image_cache, &src_mat.emissive_texture, base_dir);
            if (pbr_mat.emissive_texture != null and
                src_mat.emissive_factor[0] == 0.0 and
                src_mat.emissive_factor[1] == 0.0 and
                src_mat.emissive_factor[2] == 0.0)
            {
                pbr_mat.emissive_color = Color3.white;
            } else {
                pbr_mat.emissive_color = Color3.new(
                    src_mat.emissive_factor[0],
                    src_mat.emissive_factor[1],
                    src_mat.emissive_factor[2],
                );
            }

            materials[i] = .{ .pbr = pbr_mat };
        }

        // 2. Parse skeletons/skins
        var skeletons = try scene.allocator.alloc(?*Skeleton, gltf.skins_count);
        defer scene.allocator.free(skeletons);
        @memset(skeletons, null);

        for (0..gltf.skins_count) |skin_idx| {
            const s = &gltf.skins[skin_idx];
            const skel = try Skeleton.init(scene.allocator, s.joints_count);
            if (s.name != null) {
                skel.name = try scene.allocator.dupe(u8, std.mem.span(s.name));
            }

            // Inverse bind matrices
            if (s.inverse_bind_matrices) |ibm_acc| {
                for (0..s.joints_count) |ji| {
                    var m_floats: [16]f32 = undefined;
                    _ = c.cgltf_accessor_read_float(ibm_acc, ji, &m_floats, 16);
                    skel.bones[ji].inverse_bind_matrix = Mat4{ .m = m_floats };
                }
            }

            // Find the mesh node that uses this skin to compute root_transform
            var mesh_node: ?*c.cgltf_node = null;
            for (0..gltf.nodes_count) |ni| {
                const n = &gltf.nodes[ni];
                if (n.skin == s and n.mesh != null) {
                    mesh_node = n;
                    break;
                }
            }

            var inv_mesh_w = Mat4.identity;
            if (mesh_node) |mn| {
                var mn_w: [16]f32 = undefined;
                c.cgltf_node_transform_world(mn, &mn_w);
                const mn_mat = Mat4{ .m = mn_w };
                if (mn_mat.invert()) |inv| {
                    inv_mesh_w = inv;
                }
            }

            for (0..s.joints_count) |ji| {
                const j_node = s.joints[ji] orelse continue;
                const b = &skel.bones[ji];
                if (j_node[0].name != null) {
                    b.name = try scene.allocator.dupe(u8, std.mem.span(j_node[0].name));
                }

                // Parent within skin.joints
                if (j_node[0].parent) |parent_node| {
                    for (0..s.joints_count) |pi| {
                        if (s.joints[pi] == parent_node) {
                            b.parent_index = pi;
                            break;
                        }
                    }
                }

                // If root joint (parent_index == null), compute root_transform
                if (b.parent_index == null) {
                    var p_world = Mat4.identity;
                    if (j_node[0].parent) |parent_node| {
                        var pw_floats: [16]f32 = undefined;
                        c.cgltf_node_transform_world(parent_node, &pw_floats);
                        p_world = Mat4{ .m = pw_floats };
                    }
                    skel.root_transform = inv_mesh_w.mul(p_world);
                }

                // Initial local TRS
                if (j_node[0].has_translation != 0) {
                    b.local_position = Vec3.new(j_node[0].translation[0], j_node[0].translation[1], j_node[0].translation[2]);
                }
                if (j_node[0].has_rotation != 0) {
                    b.local_rotation = (Quat{
                        .x = j_node[0].rotation[0],
                        .y = j_node[0].rotation[1],
                        .z = j_node[0].rotation[2],
                        .w = j_node[0].rotation[3],
                    }).normalize();
                }
                if (j_node[0].has_scale != 0) {
                    b.local_scale = Vec3.new(j_node[0].scale[0], j_node[0].scale[1], j_node[0].scale[2]);
                }
                b.bind_position = b.local_position;
                b.bind_rotation = b.local_rotation;
                b.bind_scale = b.local_scale;
            }

            skel.update();
            skeletons[skin_idx] = skel;
            try scene.skeletons.append(scene.allocator, skel);
        }

        // 3. Parse meshes and primitives (preserving glTF node transforms)
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

        // 4. Parse animations
        for (0..gltf.animations_count) |anim_idx| {
            const src_anim = &gltf.animations[anim_idx];
            const anim_name = if (src_anim.name != null)
                try scene.allocator.dupe(u8, std.mem.span(src_anim.name))
            else
                try std.fmt.allocPrint(scene.allocator, "anim_{d}", .{anim_idx});

            // Find which skeleton this animation targets
            var target_skel: ?*Skeleton = null;
            var target_skin_idx: ?usize = null;
            for (0..src_anim.channels_count) |ci| {
                const ch = &src_anim.channels[ci];
                if (ch.target_node == null) continue;
                for (0..gltf.skins_count) |si| {
                    const s = &gltf.skins[si];
                    for (0..s.joints_count) |ji| {
                        if (s.joints[ji] == ch.target_node) {
                            target_skel = skeletons[si];
                            target_skin_idx = si;
                            break;
                        }
                    }
                    if (target_skel != null) break;
                }
                if (target_skel != null) break;
            }

            if (target_skel == null and skeletons.len > 0) {
                target_skel = skeletons[0];
                target_skin_idx = 0;
            }

            const skel = target_skel orelse {
                scene.allocator.free(anim_name);
                continue;
            };
            const skin_ref = &gltf.skins[target_skin_idx.?];

            var channels_list = std.ArrayList(AnimationChannel).empty;
            var max_duration: f32 = 0.0;

            for (0..src_anim.channels_count) |ci| {
                const ch = &src_anim.channels[ci];
                if (ch.target_node == null or ch.sampler == null) continue;

                var bone_idx: ?usize = null;
                for (0..skin_ref.joints_count) |ji| {
                    if (skin_ref.joints[ji] == ch.target_node) {
                        bone_idx = ji;
                        break;
                    }
                }
                const b_idx = bone_idx orelse continue;

                const path_type: AnimationPath = switch (ch.target_path) {
                    c.cgltf_animation_path_type_translation => .translation,
                    c.cgltf_animation_path_type_rotation => .rotation,
                    c.cgltf_animation_path_type_scale => .scale,
                    c.cgltf_animation_path_type_weights => .weights,
                    else => continue,
                };

                const samp = ch.sampler.?;
                const in_acc = samp.*.input;
                const out_acc = samp.*.output;
                if (in_acc == null or out_acc == null or in_acc.*.count == 0) continue;

                const key_count = in_acc.*.count;
                const timestamps = try scene.allocator.alloc(f32, key_count);
                for (0..key_count) |ki| {
                    _ = c.cgltf_accessor_read_float(in_acc, ki, &timestamps[ki], 1);
                }
                if (timestamps[key_count - 1] > max_duration) {
                    max_duration = timestamps[key_count - 1];
                }

                const stride: usize = switch (path_type) {
                    .translation, .scale => 3,
                    .rotation => 4,
                    .weights => 1,
                };
                const outputs = try scene.allocator.alloc(f32, key_count * stride);
                for (0..key_count) |ki| {
                    _ = c.cgltf_accessor_read_float(out_acc, ki, outputs[ki * stride .. ki * stride + stride].ptr, @intCast(stride));
                }

                const interp: AnimationInterpolation = switch (samp.*.interpolation) {
                    c.cgltf_interpolation_type_step => .step,
                    c.cgltf_interpolation_type_cubic_spline => .cubic_spline,
                    else => .linear,
                };

                try channels_list.append(scene.allocator, .{
                    .bone_index = b_idx,
                    .target_path = path_type,
                    .sampler = .{
                        .timestamps = timestamps,
                        .outputs = outputs,
                        .interpolation = interp,
                    },
                });
            }

            const ag = try AnimationGroup.init(scene.allocator, anim_name, try channels_list.toOwnedSlice(scene.allocator), max_duration);
            scene.allocator.free(anim_name);
            ag.skeleton = skel;
            try scene.animation_groups.append(scene.allocator, ag);
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

        return mesh_obj;
    }
};
