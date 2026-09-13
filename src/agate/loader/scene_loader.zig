const std = @import("std");

const c = @import("../c.zig").c;

const Scene = @import("../scene.zig").Scene;
const Mesh = @import("../mesh.zig").Mesh;
const Material = @import("../material.zig").Material;
const Texture = @import("../texture.zig").Texture;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;

const materials_mod = @import("materials.zig");
const skins_mod = @import("skins.zig");
const mesh_spawn_mod = @import("mesh_spawn.zig");
const animations_mod = @import("animations.zig");
const lights_mod = @import("lights.zig");
const MorphMode = @import("../mesh.zig").MorphMode;
const math = @import("math");
const Mat4 = math.Mat4;

pub const SceneLoader = struct {
    pub const LoadOptions = struct {
        /// Morph blending mode for meshes spawned from this file.
        /// .cpu (default) keeps the historical behavior: applyMorphs()
        /// rewrites a dynamic vertex buffer. .gpu keeps a static base-pose
        /// vertex buffer and the vertex shader blends from an RGBA32F delta
        /// texture (forward standard/pbr/skinned paths only; fails loudly
        /// when RGBA32F is unavailable).
        morph_mode: MorphMode = .cpu,
    };

    /// Parses a glTF scene and spawns its content into `scene`. Both
    /// `.glb` (binary) and `.gltf` (JSON + external .bin) files work.
    /// Returns the spawned meshes in node order.
    pub fn appendGlb(scene: *Scene, file_path: []const u8) ![]*Mesh {
        return appendGlbOptions(scene, file_path, .{});
    }

    pub fn appendGlbOptions(scene: *Scene, file_path: []const u8, load_options: LoadOptions) ![]*Mesh {
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

        // EXT_meshopt_compression: decode compressed buffer views in place.
        // No-op (and bit-identical behaviour) for files without the extension;
        // see loader/meshopt.zig for the supported modes and limitations.
        const decode_res = c.agate_cgltf_decode_meshopt(&options, data);
        if (decode_res != c.cgltf_result_success) {
            return error.GltfMeshoptDecodeFailed;
        }

        const gltf = data.?;
        const base_dir = std.fs.path.dirname(file_path);

        // 1. Parse materials
        const materials = try scene.allocator.alloc(?Material, gltf.materials_count);
        defer scene.allocator.free(materials);
        @memset(materials, null);

        const image_cache = try scene.allocator.alloc(?Texture, gltf.images_count);
        defer scene.allocator.free(image_cache);
        @memset(image_cache, null);

        // Decode every image up front on worker threads; the sg.Image is built
        // later in loadTextureFromView because sokol_gfx is not thread-safe.
        const decoded = try scene.allocator.alloc(?Texture.RawTexture, gltf.images_count);
        @memset(decoded, null);
        defer {
            for (decoded) |*d| {
                if (d.*) |*raw| raw.deinit(scene.allocator);
            }
            scene.allocator.free(decoded);
        }
        materials_mod.decodeImagesInParallel(scene, gltf, decoded, base_dir);
        try materials_mod.loadMaterials(scene, gltf, base_dir, materials, image_cache, decoded);

        // 2. Parse skeletons/skins
        const skeletons = try scene.allocator.alloc(?*Skeleton, gltf.skins_count);
        defer scene.allocator.free(skeletons);
        @memset(skeletons, null);
        try skins_mod.loadSkins(scene, gltf, skeletons);

        // Marks every glTF node used as a skeleton joint. Node tracks that
        // target such joints of skinned meshes are skipped (the skeleton
        // path already drives them).
        const is_joint_node = try skins_mod.buildJointMask(scene.allocator, gltf);
        defer scene.allocator.free(is_joint_node);

        // 3. Parse meshes and primitives (preserving glTF node transforms)
        var spawned_meshes = std.ArrayList(*Mesh).empty;

        // Maps each glTF node to the contiguous range of meshes spawned from
        // it, so node animation channels can find their target meshes below.
        const node_mesh_start = try scene.allocator.alloc(usize, gltf.nodes_count);
        defer scene.allocator.free(node_mesh_start);
        const node_mesh_count = try scene.allocator.alloc(usize, gltf.nodes_count);
        defer scene.allocator.free(node_mesh_count);
        @memset(node_mesh_start, 0);
        @memset(node_mesh_count, 0);

        try mesh_spawn_mod.spawnMeshes(scene, gltf, materials, skeletons, &spawned_meshes, node_mesh_start, node_mesh_count, load_options.morph_mode);

        // 4. Parse animations (skeleton tracks + plain node tracks).
        // Group names come from gltf animation.name so UI code can enumerate
        // them via scene.animation_groups.
        try animations_mod.loadAnimations(scene, gltf, skeletons, &spawned_meshes, node_mesh_start, node_mesh_count, is_joint_node);

        // 5. Parse punctual lights (KHR_lights_punctual) and cameras.
        // No scene-level transform exists at this level, so identity is used.
        try lights_mod.loadLights(scene, gltf, Mat4.identity);
        _ = try lights_mod.loadCameras(scene, gltf, Mat4.identity);

        return spawned_meshes.toOwnedSlice(scene.allocator);
    }
};
