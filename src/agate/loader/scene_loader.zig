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
        /// Stage 2: decode images on the UploadQueue workers instead of
        /// blocking this call. Materials start with null texture slots
        /// (default-white fallback renders) and the real textures patch in
        /// via scene.render()'s drain. Requires scene.uploads (present in
        /// every Scene.init scene); silently falls back to synchronous
        /// decoding when the queue is unavailable.
        async_textures: bool = false,
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

        // Check required extensions against supported set per glTF 2.0 specification:
        // Client implementations must not parse/render assets requiring unsupported extensions.
        if (data.?.extensions_required_count > 0 and data.?.extensions_required != null) {
            for (0..data.?.extensions_required_count) |ext_i| {
                const ext_ptr = data.?.extensions_required[ext_i];
                if (ext_ptr == null) continue;
                const ext_name = std.mem.span(ext_ptr);
                if (!isExtensionSupported(ext_name)) {
                    return error.UnsupportedGltfExtension;
                }
            }
        }

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

        const image_cache = try scene.allocator.alloc(?Texture, gltf.images_count * 2);
        defer scene.allocator.free(image_cache);
        @memset(image_cache, null);

        const decoded = try scene.allocator.alloc(?Texture.RawTexture, gltf.images_count);
        @memset(decoded, null);
        defer {
            for (decoded) |*d| {
                if (d.*) |*raw| raw.deinit(scene.allocator);
            }
            scene.allocator.free(decoded);
        }

        // Stage 2: in async mode the pre-decode pass is skipped entirely —
        // images decode on the scene's UploadQueue after this call returns,
        // and loadMaterials registers material slots as patch targets.
        // Sync mode keeps the fork-join predecode (single hitch, no pop-in).
        const async_textures = load_options.async_textures and scene.uploads != null;
        var actx: ?materials_mod.AsyncTexCtx = if (async_textures)
            materials_mod.AsyncTexCtx.init(scene, gltf, base_dir, &scene.uploads.?)
        else
            null;
        defer if (actx) |*a| a.deinit();

        if (!async_textures) {
            materials_mod.decodeImagesInParallel(scene, gltf, decoded, base_dir);
        }
        try materials_mod.loadMaterials(scene, gltf, base_dir, materials, image_cache, decoded, if (actx) |*a| a else null);

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

pub fn isExtensionSupported(name: []const u8) bool {
    const supported = [_][]const u8{
        "EXT_meshopt_compression",
        "KHR_lights_punctual",
        "KHR_texture_transform",
        "KHR_materials_unlit",
    };
    for (supported) |s| {
        if (std.mem.eql(u8, s, name)) return true;
    }
    return false;
}

test "isExtensionSupported accepts engine extensions and rejects unsupported" {
    try std.testing.expect(isExtensionSupported("EXT_meshopt_compression"));
    try std.testing.expect(isExtensionSupported("KHR_lights_punctual"));
    try std.testing.expect(isExtensionSupported("KHR_texture_transform"));
    try std.testing.expect(isExtensionSupported("KHR_materials_unlit"));

    // Unsupported extensions that must be rejected when required:
    try std.testing.expect(!isExtensionSupported("KHR_draco_mesh_compression"));
    try std.testing.expect(!isExtensionSupported("KHR_materials_volume"));
    try std.testing.expect(!isExtensionSupported("UNKNOWN_extension"));
}
