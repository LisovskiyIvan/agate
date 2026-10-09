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
const SkinJointWeight = @import("../mesh.zig").types.SkinJointWeight;
const MorphTarget = @import("../mesh.zig").MorphTarget;
const MAX_MORPH_TARGETS = @import("../mesh.zig").MAX_MORPH_TARGETS;
const computeNormals = @import("../mesh.zig").computeNormals;
const computeTangentsForUv = @import("../mesh.zig").tangents.computeTangentsForUv;
const Material = @import("../material.zig").Material;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const MorphMode = @import("../mesh.zig").MorphMode;
const morph_gpu = @import("../mesh.zig").morph_gpu;
const gpu_thread = @import("../gpu_thread.zig");
const materials_mod = @import("materials.zig");

/// Reject invalid/missing UV1 before publishing any imported materials.
pub fn validateTextureCoordinates(gltf: *const c.cgltf_data) !void {
    try materials_mod.validateTextureCoordinates(gltf);
    for (0..gltf.meshes_count) |mi| {
        const mesh = &gltf.meshes[mi];
        for (0..mesh.primitives_count) |pi| {
            const prim = &mesh.primitives[pi];
            if (prim.type != c.cgltf_primitive_type_triangles) continue;
            var pos: ?*c.cgltf_accessor = null;
            var uv1: ?*c.cgltf_accessor = null;
            for (0..prim.attributes_count) |ai| {
                const attr = prim.attributes[ai];
                if (attr.type == c.cgltf_attribute_type_position) pos = attr.data;
                if (attr.type == c.cgltf_attribute_type_texcoord and attr.index == 1) uv1 = attr.data;
            }
            if (pos == null) continue;
            if (uv1) |uv| {
                if (uv.type != c.cgltf_type_vec2 or uv.count != pos.?.count) return error.InvalidTextureCoordinate;
            }
            if (prim.material) |mat| {
                for (materials_mod.textureViewsForMaterial(mat)) |view| {
                    if (view.*.texture != null and try materials_mod.textureCoordFromView(view) == 1 and uv1 == null)
                        return error.MissingTextureCoordinate;
                }
            }
        }
    }
}

pub fn spawnMeshes(
    scene: *Scene,
    gltf: *c.cgltf_data,
    materials: []const ?Material,
    skeletons: []?*Skeleton,
    spawned_meshes: *std.ArrayList(*Mesh),
    node_mesh_start: []usize,
    node_mesh_count: []usize,
    morph_mode: MorphMode,
) !void {
    // Off-context spawns are safe: mesh buffers are deferred (gpu_pending +
    // pending_vertices, finished by Scene.flushPendingGpuUploads), and the
    // texture paths assert the graphics thread only when they would create
    // sg objects inline (sync texture mode). Async texture mode queues the
    // decode/upload through Scene.uploads instead, so a GLB load with
    // `.async_textures = true` is legal from the game thread.
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
                if (try parsePrimitive(scene, gltf, prim, src_mesh, node, mesh_name, base_matrix, materials, node_skeleton, morph_mode)) |mesh_obj| {
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
            const src_mesh: *const c.cgltf_mesh = @ptrCast(&gltf.meshes[mesh_idx]);
            const mesh_name = if (src_mesh.name != null)
                std.mem.span(src_mesh.name)
            else
                "glb_mesh";

            for (0..src_mesh.primitives_count) |prim_idx| {
                const prim: *const c.cgltf_primitive = @ptrCast(&src_mesh.primitives[prim_idx]);
                if (try parsePrimitive(scene, gltf, prim, src_mesh, null, mesh_name, Mat4.identity, materials, null, morph_mode)) |mesh_obj| {
                    try scene.meshes.append(scene.allocator, mesh_obj);
                    try spawned_meshes.append(scene.allocator, mesh_obj);
                }
            }
        }
    }
}

/// Reads one glTF morph target (POSITION/NORMAL/TANGENT delta attributes)
/// into owned slices. Missing attributes stay empty (no delta); tangents are
/// often absent. Short accessors read as zeros past their end.
fn parseMorphTarget(allocator: std.mem.Allocator, src: *const c.cgltf_morph_target, vert_count: usize) !MorphTarget {
    var pos_accessor: ?*c.cgltf_accessor = null;
    var norm_accessor: ?*c.cgltf_accessor = null;
    var tan_accessor: ?*c.cgltf_accessor = null;

    for (0..src.attributes_count) |attr_idx| {
        const attr = &src.attributes[attr_idx];
        switch (attr.type) {
            c.cgltf_attribute_type_position => pos_accessor = attr.data,
            c.cgltf_attribute_type_normal => norm_accessor = attr.data,
            c.cgltf_attribute_type_tangent => tan_accessor = attr.data,
            else => {},
        }
    }

    var mt: MorphTarget = .{};
    errdefer freeMorphTarget(allocator, &mt);

    // glTF morph tangents are vec3 (xyz); the base tangent w is preserved.
    if (pos_accessor) |pa| {
        const deltas = try allocator.alloc([3]f32, vert_count);
        for (0..vert_count) |i| {
            var d: [3]f32 = .{ 0, 0, 0 };
            _ = c.cgltf_accessor_read_float(pa, i, &d, 3);
            deltas[i] = d;
        }
        mt.position_deltas = deltas;
    }
    if (norm_accessor) |na| {
        const deltas = try allocator.alloc([3]f32, vert_count);
        for (0..vert_count) |i| {
            var d: [3]f32 = .{ 0, 0, 0 };
            _ = c.cgltf_accessor_read_float(na, i, &d, 3);
            deltas[i] = d;
        }
        mt.normal_deltas = deltas;
    }
    if (tan_accessor) |ta| {
        const deltas = try allocator.alloc([3]f32, vert_count);
        for (0..vert_count) |i| {
            var d: [3]f32 = .{ 0, 0, 0 };
            _ = c.cgltf_accessor_read_float(ta, i, &d, 3);
            deltas[i] = d;
        }
        mt.tangent_deltas = deltas;
    }
    return mt;
}

fn freeMorphTarget(allocator: std.mem.Allocator, mt: *MorphTarget) void {
    if (mt.position_deltas.len > 0) allocator.free(mt.position_deltas);
    if (mt.normal_deltas.len > 0) allocator.free(mt.normal_deltas);
    if (mt.tangent_deltas.len > 0) allocator.free(mt.tangent_deltas);
    mt.* = .{};
}

pub fn parsePrimitive(
    scene: *Scene,
    gltf: *c.cgltf_data,
    prim: *const c.cgltf_primitive,
    src_mesh: *const c.cgltf_mesh,
    node: ?*const c.cgltf_node,
    mesh_name: []const u8,
    base_matrix: Mat4,
    materials: []const ?Material,
    skeleton: ?*Skeleton,
    morph_mode: MorphMode,
) !?*Mesh {
    if (prim.type != c.cgltf_primitive_type_triangles) return null;

    // Off-context (game-thread) loads must not touch sg.*: buffer and delta
    // texture creation below is skipped and the mesh is finished by
    // Mesh.finishGpuUpload on the context thread instead.
    const off_context = !gpu_thread.isOnContextThread();

    var pos_accessor: ?*c.cgltf_accessor = null;
    var norm_accessor: ?*c.cgltf_accessor = null;
    var col_accessor: ?*c.cgltf_accessor = null;
    var uv_accessor: ?*c.cgltf_accessor = null;
    var uv1_accessor: ?*c.cgltf_accessor = null;
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
                if (attr.index == 1) uv1_accessor = attr.data;
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

    // A referenced UV1 set must exist. Silently substituting UV0 renders a
    // valid-looking but wrong material; reject before allocating GPU state.
    if (prim.material) |mat| {
        for (materials_mod.textureViewsForMaterial(mat)) |view| {
            if (view.*.texture != null and try materials_mod.textureCoordFromView(view) == 1 and uv1_accessor == null)
                return error.MissingTextureCoordinate;
        }
    }
    const tangent_uv: u1 = if (prim.material) |mat|
        (if (mat.*.normal_texture.texture != null) try materials_mod.textureCoordFromView(&mat.*.normal_texture) else 0)
    else
        0;

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
        const uv1 = try readUv1(uv1_accessor, vert_count, i);

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
            .uv1 = uv1,
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

            if (norm_accessor == null) {
                computeNormals(vertices, indices, null);
            }
            if (tan_accessor == null) {
                computeTangentsForUv(vertices, indices, null, tangent_uv);
            }

            ibuf = if (off_context) .{} else sg.makeBuffer(.{
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

            if (norm_accessor == null) {
                computeNormals(vertices, null, indices);
            }
            if (tan_accessor == null) {
                computeTangentsForUv(vertices, null, indices, tangent_uv);
            }

            ibuf = if (off_context) .{} else sg.makeBuffer(.{
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
            if (norm_accessor == null) {
                computeNormals(vertices, indices, null);
            }
            if (tan_accessor == null) {
                computeTangentsForUv(vertices, indices, null, tangent_uv);
            }
            ibuf = if (off_context) .{} else sg.makeBuffer(.{
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
            if (norm_accessor == null) {
                computeNormals(vertices, null, indices);
            }
            if (tan_accessor == null) {
                computeTangentsForUv(vertices, null, indices, tangent_uv);
            }
            ibuf = if (off_context) .{} else sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(indices),
            });
        }
    }

    // Morph targets blend either on CPU or GPU:
    // - CPU mode (default): buffers created with initial data and no dynamic
    //   flag are immutable in sokol, so morph meshes get an empty updatable
    //   buffer; the first frame's applyMorphs() (before render) fills it
    //   with base + default weights.
    // - GPU mode: the vertex buffer stays the static base pose; deltas live
    //   in the RGBA32F delta texture and the vertex shader blends them, so
    //   the buffer is created filled with base data.
    const has_morph = prim.targets_count > 0;
    // Explicit result type: the deferred branch is an empty handle, and
    // without the annotation Zig cannot unify `.{}` with `sg.Buffer` here
    // (visible only when a consumer actually instantiates spawnMeshes).
    const vbuf: sg.Buffer = if (off_context)
        .{}
    else if (!has_morph or morph_mode == .gpu)
        sg.makeBuffer(.{ .data = sg.asRange(vertices) })
    else blk: {
        const vb = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .write_transient = true },
            .size = vertices.len * @sizeOf(Vertex),
        });
        if (vb.id != 0 and vertices.len > 0 and sg.queryBufferState(vb) == .VALID) {
            sg.writeBufferTransient(.{
                .dst = .{ .buffer = vb },
                .src = .{ .data = sg.asRange(vertices) },
            });
        }
        break :blk vb;
    };

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

    // Allocate the struct first: if the name dupe fails, destroying the bare
    // struct cannot leak the name (the reverse order would).
    const mesh_obj = try scene.allocator.create(Mesh);
    const owned_name = scene.allocator.dupe(u8, mesh_name) catch |err| {
        scene.allocator.destroy(mesh_obj);
        return err;
    };
    mesh_obj.* = .{
        .name = owned_name,
        .owns_name = true,
        .vertex_buffer = vbuf,
        .index_buffer = ibuf,
        .vertex_count = @intCast(vertices.len),
        .index_count = index_count,
        .index_type = index_type,
        .skeleton = skeleton,
        .base_matrix = base_matrix,
        .local_bounding_box = local_box,
        .gpu_pending = off_context,
        .pending_dynamic_update = off_context and has_morph and morph_mode == .cpu,
    };

    // Adopt the mesh's owned allocations before any fallible step below.
    // Buffers are id-guarded (empty on the deferred path); the morph-target
    // list has its own errdefer that clears the field first, so this outer
    // cleanup can mirror Mesh.deinit for everything else without double-free.
    errdefer {
        if (mesh_obj.vertex_buffer.id != 0) sg.destroyBuffer(mesh_obj.vertex_buffer);
        if (mesh_obj.index_buffer.id != 0) sg.destroyBuffer(mesh_obj.index_buffer);
        if (mesh_obj.morph_delta_view.id != 0) sg.destroyView(mesh_obj.morph_delta_view);
        if (mesh_obj.morph_delta_image.id != 0) sg.destroyImage(mesh_obj.morph_delta_image);
        for (mesh_obj.morph_targets) |*mt| freeMorphTarget(scene.allocator, mt);
        if (mesh_obj.morph_targets.len > 0) scene.allocator.free(mesh_obj.morph_targets);
        if (mesh_obj.morph_weights.len > 0) scene.allocator.free(mesh_obj.morph_weights);
        if (mesh_obj.morph_base.len > 0) scene.allocator.free(mesh_obj.morph_base);
        if (mesh_obj.morph_staging.len > 0) scene.allocator.free(mesh_obj.morph_staging);
        if (mesh_obj.cpu_positions.len > 0) scene.allocator.free(mesh_obj.cpu_positions);
        if (mesh_obj.cpu_indices.len > 0) scene.allocator.free(mesh_obj.cpu_indices);
        if (mesh_obj.cpu_skin.len > 0) scene.allocator.free(mesh_obj.cpu_skin);
        if (mesh_obj.pending_vertices.len > 0) scene.allocator.free(mesh_obj.pending_vertices);
        if (mesh_obj.owns_name and mesh_obj.name.len > 0) scene.allocator.free(mesh_obj.name);
        scene.allocator.destroy(mesh_obj);
    }

    if (off_context) {
        // No sg.* above: retain the vertices so Mesh.finishGpuUpload can
        // create the buffers on the context thread. cpu_positions/cpu_indices
        // below stay the finish-step index source, exactly as on-context.
        mesh_obj.pending_vertices = try scene.allocator.dupe(Vertex, vertices);
    }

    // Morph targets (blend shapes): per-vertex POSITION/NORMAL/TANGENT
    // deltas. Only the first MAX_MORPH_TARGETS are kept; extras are dropped
    // (documented engine limit). usize ranges: vertex counts above u16 need
    // no special handling; absent tangent deltas stay empty slices.
    if (prim.targets_count > 0 and prim.targets != null) {
        const want: usize = @min(prim.targets_count, MAX_MORPH_TARGETS);
        const list = try scene.allocator.alloc(MorphTarget, want);
        for (list) |*mt| mt.* = .{};
        var done: usize = 0;
        errdefer {
            for (list[0..done]) |*mt| freeMorphTarget(scene.allocator, mt);
            scene.allocator.free(list);
            // The outer mesh errdefer also scans morph_targets: clear the
            // field so a failure here can never double-free the list.
            mesh_obj.morph_targets = &.{};
        }
        for (0..want) |ti| {
            const target_ptr: *const c.cgltf_morph_target = @ptrCast(&prim.targets[ti]);
            list[ti] = try parseMorphTarget(scene.allocator, target_ptr, vert_count);
            done += 1;
        }
        mesh_obj.morph_targets = list;

        // Default morph weights: mesh.weights first, node.weights overrides
        // (glTF: the instantiated node's weights win). cgltf exposes no
        // per-primitive weights; missing entries stay 0.
        const weights = try scene.allocator.alloc(f32, want);
        for (weights) |*wgt| wgt.* = 0.0;
        if (src_mesh.weights != null) {
            const n = @min(src_mesh.weights_count, want);
            for (0..n) |i| weights[i] = src_mesh.weights[i];
        }
        if (node) |nd| {
            if (nd.weights != null) {
                const n = @min(nd.weights_count, want);
                for (0..n) |i| weights[i] = nd.weights[i];
            }
        }
        mesh_obj.morph_weights = weights;
    }

    mesh_obj.morph_mode = morph_mode;

    if (mesh_obj.hasMorphTargets()) {
        try mesh_obj.retainMorphBase(scene.allocator, vertices);
        if (morph_mode == .gpu) {
            // Pack target deltas into the RGBA32F delta texture. The vertex
            // buffer is already the static base pose; applyMorphs() no-ops
            // in this mode, so no first-frame upload is needed. Fails loudly
            // when RGBA32F is unavailable (no silent CPU fallback: the
            // buffer is already immutable).
            if (off_context) {
                // No sg.* off-context: Mesh.finishGpuUpload uploads the
                // delta texture on the context thread and retries on failure.
                mesh_obj.morph_upload_pending = true;
            } else {
                try morph_gpu.uploadMorphDeltas(mesh_obj, scene.allocator);
            }
        } else {
            // The GPU buffer is empty for CPU-morph meshes until the first
            // applyMorphs(); mark dirty so frame 1 uploads base + default
            // weights before render.
            mesh_obj.morph_dirty = true;
        }
    }

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

    if (skeleton != null or joints_accessor != null) {
        const cpu_skin = try scene.allocator.alloc(SkinJointWeight, vert_count);
        for (vertices, 0..) |v, i| {
            cpu_skin[i] = .{
                .joints = v.joints,
                .weights = v.weights,
            };
        }
        mesh_obj.cpu_skin = cpu_skin;
    }

    return mesh_obj;
}

pub fn readUv1(accessor: ?*c.cgltf_accessor, count: usize, index: usize) ![2]f32 {
    var uv: [2]f32 = .{ 0, 0 };
    if (accessor) |a| {
        if (a.count != count or a.type != c.cgltf_type_vec2 or index >= count or
            c.cgltf_accessor_read_float(a, index, &uv, 2) == 0) return error.InvalidTextureCoordinate;
    }
    return uv;
}
