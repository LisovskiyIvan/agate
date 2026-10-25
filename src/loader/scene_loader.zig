const std = @import("std");
const sokol = @import("sokol");

const c = @import("../c.zig").c;

const Scene = @import("../scene.zig").Scene;
const Mesh = @import("../mesh.zig").Mesh;
const Material = @import("../material.zig").Material;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;

const materials_mod = @import("materials.zig");
const skins_mod = @import("skins.zig");
const mesh_spawn_mod = @import("mesh_spawn.zig");
const animations_mod = @import("animations.zig");
const lights_mod = @import("lights.zig");
const colliders_mod = @import("colliders.zig");
const physics = @import("../physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const RigidBody = physics.RigidBody;
const MorphMode = @import("../mesh.zig").MorphMode;
const math = @import("math");
const Mat4 = math.Mat4;
const gpu_thread = @import("../gpu_thread.zig");

pub const LoadTimings = struct {
    /// Caller-owned per-stage wall times in milliseconds. The loader only
    /// reads the clock when `LoadOptions.timings` is non-null (default null
    /// = zero overhead, no logging). Clock: the app-owned sokol.time
    /// timeline (same as `scene.elapsedMsSince` / the profiler): the host
    /// must have called `sokol.time.setup()` once at startup (main.zig
    /// does); the loader never calls setup itself (that would reset the
    /// global origin for concurrent readers).
    parse_ms: f64 = 0,
    buffers_ms: f64 = 0,
    meshopt_ms: f64 = 0,
    /// Parallel stb/KTX2/DDS/Basis CPU decode (`decodeImagesInParallel`).
    textures_ms: f64 = 0,
    /// GPU upload + material wiring (`loadMaterials`, main thread).
    materials_ms: f64 = 0,
    /// Skins + joint mask + mesh spawn + animations + lights/cameras.
    geometry_ms: f64 = 0,
    total_ms: f64 = 0,
};

/// Milliseconds elapsed since a `sokol.time.now()` tick. Same expression
/// as `scene.elapsedMsSince`; only called on the timings opt-in path.
fn msSinceTicks(t0: u64) f64 {
    return @floatCast(sokol.time.ms(sokol.time.now() -% t0));
}

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
        /// decoding when the queue is unavailable. A load running off the
        /// graphics thread forces this mode when the queue exists: sync
        /// texture creation touches sg.* inline and would fail loudly.
        async_textures: bool = false,
        /// Sampler anisotropy (1..16) for every texture this load creates.
        /// null keeps the engine default, which is Babylon's
        /// `Texture.DEFAULT_ANISOTROPIC_FILTERING_LEVEL` = 4 — the value the
        /// Babylon glTF loader leaves untouched. Pass 1 to opt out of
        /// anisotropic filtering (matches Babylon at
        /// `anisotropicFilteringLevel = 1`), or 16 for the maximum.
        /// The authored glTF sampler still wins on wrap/filter; this only
        /// replaces the anisotropy, and the LINEAR-filter clamp in
        /// `Texture.effectiveAnisotropy` still applies.
        max_anisotropy: ?u32 = null,
        /// Optional caller-owned stage timings (parse/buffers/meshopt/
        /// texture decode/materials upload/geometry/total, milliseconds).
        /// Null (default) disables all timing reads: every existing call
        /// keeps working unchanged and pays no clock cost.
        timings: ?*LoadTimings = null,
        /// Optional physics world to spawn colliders into automatically.
        /// If provided (or if null and auto_colliders is true, falls back to scene.physics.getWorld()),
        /// nodes specifying collider data (via extras, OMI_physics_body / KHR_physics_rigid_bodies,
        /// or naming conventions) will have RigidBody instances created and registered.
        physics_world: ?*PhysicsWorld = null,
        /// If true, colliders detected from glTF are automatically instantiated when a
        /// physics world is available.
        auto_colliders: bool = true,
    };

    /// Loads a glTF scene (`.glb` or `.gltf`) and spawns its content into
    /// `scene`. Returns a `GltfAsset`: the spawned meshes (node order) plus
    /// ownership of everything the loader created for them (materials and
    /// their textures). See `GltfAsset.deinit` for the unload contract.
    pub fn loadGlb(scene: *Scene, file_path: []const u8) !*GltfAsset {
        return loadGlbOptions(scene, file_path, .{});
    }

    /// Ownership record of one `loadGlb`/`loadGlbOptions` load: the spawned
    /// meshes plus everything the loader created for them (the load's PBR
    /// materials and their textures).
    ///
    /// Unload contract (`deinit`):
    /// - Destroys the meshes, deinitializes the material textures (each
    ///   distinct GPU handle set exactly once — one glTF image backs several
    ///   slots/materials as copies of the same handles), removes the
    ///   materials from the scene registry and frees this struct.
    /// - The asset registers itself with the scene: `Scene.deinit` unloads
    ///   every asset still alive, so forgetting `deinit` never leaks GPU
    ///   memory past the scene.
    /// - The materials belong to the asset: do not keep or re-assign them
    ///   past `deinit`. Context thread only when it will destroy nonzero
    ///   handles (the same rule as `Scene.deinit`), and after any async
    ///   texture commit patched the slots.
    /// - `Scene.destroyMesh` frees a single mesh only; it never touches
    ///   materials/textures — this struct is the unload path for loader
    ///   content.
    pub const GltfAsset = struct {
        scene: *Scene,
        meshes: []*Mesh,
        materials: []*PBRMaterial,
        bodies: []*RigidBody = &.{},
        physics_world: ?*PhysicsWorld = null,

        /// Explicitly spawn colliders for this asset into `pw` if not already spawned.
        pub fn spawnColliders(self: *GltfAsset, pw: *PhysicsWorld) !void {
            if (self.bodies.len > 0) return;
            var list = std.ArrayList(*RigidBody).empty;
            errdefer {
                for (list.items) |b| pw.removeBody(b);
                list.deinit(self.scene.allocator);
            }
            for (self.meshes) |mesh| {
                if (colliders_mod.parseNameCollider(mesh.name)) |spec| {
                    if (spec.is_collision_proxy) {
                        mesh.is_visible = false;
                        mesh.cast_shadows = false;
                        mesh.receive_shadows = false;
                    }
                    const safe_mass = if (spec.collider_type == .mesh) 0.0 else spec.mass;
                    const body = try pw.createBodyWith(mesh, spec.collider_type, safe_mass, .{
                        .is_sensor = spec.is_sensor,
                        .filter = spec.filter,
                    });
                    body.friction = spec.friction;
                    body.restitution = spec.restitution;
                    try list.append(self.scene.allocator, body);
                }
            }
            self.bodies = try list.toOwnedSlice(self.scene.allocator);
            self.physics_world = pw;
        }

        pub fn deinit(self: *GltfAsset) void {
            const scene = self.scene;
            scene.unregisterLoadedAsset(self);
            if (self.physics_world) |pw| {
                for (self.bodies) |b| pw.removeBody(b);
            }
            if (self.bodies.len > 0) {
                scene.allocator.free(self.bodies);
            }
            // Pass 1: textures. Read-only on the materials (later dedupe
            // compares against earlier slot values), each distinct handle
            // set destroyed exactly once.
            for (self.materials, 0..) |mat, mi| {
                const slots = textureSlots(mat);
                for (slots, 0..) |slot, si| {
                    const t = slot.* orelse continue;
                    if (assetTextureReleased(self.materials, mi, si, t)) continue;
                    var owned = t;
                    owned.deinit();
                }
                if (mat.environment_texture) |cube| {
                    if (!assetCubeReleased(self.materials, mi, cube)) {
                        var owned = cube;
                        owned.deinit();
                    }
                }
            }
            // Pass 2: the loader-created materials themselves.
            for (self.materials) |mat| {
                scene.destroyPBRMaterial(mat);
            }
            // Pass 3: meshes and this struct.
            for (self.meshes) |mesh| {
                scene.destroyMesh(mesh);
            }
            scene.allocator.free(self.meshes);
            scene.allocator.free(self.materials);
            scene.allocator.destroy(self);
        }
    };

    /// Material texture slots in canonical release order.
    fn textureSlots(mat: *PBRMaterial) [7]*?Texture {
        return .{
            &mat.albedo_texture,
            &mat.normal_texture,
            &mat.metallic_roughness_texture,
            &mat.emissive_texture,
            &mat.occlusion_texture,
            &mat.clearcoat.mask_texture,
            &mat.sheen.color_texture,
        };
    }

    fn sameHandles(a: Texture, b: Texture) bool {
        return a.image.id == b.image.id and a.view.id == b.view.id and a.sampler.id == b.sampler.id;
    }

    /// True when an earlier slot of the material set already carried the
    /// same GPU handles: one glTF image backs several material slots as
    /// copies of the same image/view/sampler, and every handle set must be
    /// destroyed exactly once. "Earlier" = earlier slots of this material,
    /// then all slots of earlier materials.
    fn assetTextureReleased(materials: []const *PBRMaterial, mat_index: usize, slot_index: usize, t: Texture) bool {
        const cur_slots = textureSlots(materials[mat_index]);
        for (cur_slots[0..slot_index]) |earlier| {
            if (earlier.*) |e| {
                if (sameHandles(e, t)) return true;
            }
        }
        for (materials[0..mat_index]) |m| {
            for (textureSlots(m)) |earlier| {
                if (earlier.*) |e| {
                    if (sameHandles(e, t)) return true;
                }
            }
        }
        return false;
    }

    /// True when an earlier material of the set already carried the same
    /// environment-cube handles.
    fn assetCubeReleased(materials: []const *PBRMaterial, mat_index: usize, cube: CubeTexture) bool {
        for (materials[0..mat_index]) |m| {
            if (m.environment_texture) |e| {
                if (e.image.id == cube.image.id and e.view.id == cube.view.id and e.sampler.id == cube.sampler.id) {
                    return true;
                }
            }
        }
        return false;
    }

    /// Unwinds the loader-created materials and their textures recorded so
    /// far (zero-alloc: we are already on an error path; hostile inputs
    /// with more than 64 distinct materials leave the tail to Scene.deinit).
    fn unwindMaterials(scene: *Scene, materials: []?Material) void {
        var mats: [64]*PBRMaterial = undefined;
        var count: usize = 0;
        for (materials) |m| {
            const mat = m orelse continue;
            const pbr_mat = switch (mat) {
                .pbr => |p| p,
                .shader_material => continue,
            };
            var seen = false;
            for (mats[0..count]) |p| {
                if (p == pbr_mat) {
                    seen = true;
                    break;
                }
            }
            if (!seen and count < mats.len) {
                mats[count] = pbr_mat;
                count += 1;
            }
        }
        for (mats[0..count], 0..) |mat, mi| {
            const slots = textureSlots(mat);
            for (slots, 0..) |slot, si| {
                const t = slot.* orelse continue;
                if (!assetTextureReleased(mats[0..count], mi, si, t)) {
                    var owned = t;
                    owned.deinit();
                }
                slot.* = null;
            }
            if (mat.environment_texture) |*cube| {
                if (!assetCubeReleased(mats[0..count], mi, cube.*)) {
                    var owned = cube.*;
                    owned.deinit();
                }
                mat.environment_texture = null;
            }
            scene.destroyPBRMaterial(mat);
        }
    }

    /// Unwinds exactly the committed pieces of a failed load (null pieces
    /// are skipped): destroys the meshes of both the owned slice and the
    /// spawn list, frees every owned slice, then the materials/textures.
    fn unwindLoad(
        scene: *Scene,
        materials: []?Material,
        spawned_meshes: *std.ArrayList(*Mesh),
        owned_mats: ?[]*PBRMaterial,
        meshes: ?[]*Mesh,
        bodies: ?[]*RigidBody,
        physics_world: ?*PhysicsWorld,
    ) void {
        if (bodies) |bs| {
            if (physics_world) |pw| {
                for (bs) |b| pw.removeBody(b);
            }
            scene.allocator.free(bs);
        }
        if (meshes) |ms| {
            for (ms) |m| scene.destroyMesh(m);
            scene.allocator.free(ms);
        }
        for (spawned_meshes.items) |m| scene.destroyMesh(m);
        spawned_meshes.deinit(scene.allocator);
        if (owned_mats) |om| scene.allocator.free(om);
        unwindMaterials(scene, materials);
    }

    pub fn loadGlbOptions(scene: *Scene, file_path: []const u8, load_options: LoadOptions) !*GltfAsset {
        const t_total: u64 = if (load_options.timings != null) sokol.time.now() else 0;
        if (load_options.timings) |t| t.* = .{};
        defer if (load_options.timings) |t| {
            t.total_ms = msSinceTicks(t_total);
        };
        const path_z = try scene.allocator.allocSentinel(u8, file_path.len, 0);
        @memcpy(path_z, file_path);
        defer scene.allocator.free(path_z);

        var options = std.mem.zeroes(c.cgltf_options);
        var data: ?*c.cgltf_data = null;

        const t_parse: u64 = if (load_options.timings != null) sokol.time.now() else 0;
        const parse_res = c.cgltf_parse_file(&options, path_z.ptr, &data);
        if (load_options.timings) |t| t.parse_ms = msSinceTicks(t_parse);
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
        const t_buffers: u64 = if (load_options.timings != null) sokol.time.now() else 0;
        const load_buf_res = c.cgltf_load_buffers(&options, data, path_z.ptr);
        if (load_options.timings) |t| t.buffers_ms = msSinceTicks(t_buffers);
        if (load_buf_res != c.cgltf_result_success) {
            return error.GltfLoadBuffersFailed;
        }

        // EXT_meshopt_compression: decode compressed buffer views in place.
        // No-op (and bit-identical behaviour) for files without the extension;
        // see loader/meshopt.zig for the supported modes and limitations.
        const t_meshopt: u64 = if (load_options.timings != null) sokol.time.now() else 0;
        const decode_res = c.agate_cgltf_decode_meshopt(&options, data);
        if (load_options.timings) |t| t.meshopt_ms = msSinceTicks(t_meshopt);
        if (decode_res != c.cgltf_result_success) {
            return error.GltfMeshoptDecodeFailed;
        }

        const gltf = data.?;
        // Bound per-array counts before allocating: a tiny hostile JSON can
        // otherwise request hundreds of MB of `?Texture`/nodes.
        const max_items: usize = 1 << 20;
        if (gltf.materials_count > max_items or gltf.images_count > max_items or
            gltf.skins_count > max_items or gltf.nodes_count > max_items or
            gltf.meshes_count > max_items)
        {
            return error.SceneTooLarge;
        }
        try mesh_spawn_mod.validateTextureCoordinates(gltf);
        const base_dir = std.fs.path.dirname(file_path);

        // 1. Parse materials
        const materials = try scene.allocator.alloc(?Material, gltf.materials_count);
        defer scene.allocator.free(materials);
        @memset(materials, null);

        const image_cache = try scene.allocator.alloc(?Texture, gltf.images_count * 2);
        defer scene.allocator.free(image_cache);
        @memset(image_cache, null);

        const decoded = try scene.allocator.alloc(?Texture.DecodedImage, gltf.images_count);
        @memset(decoded, null);
        defer {
            for (decoded) |*d| {
                if (d.*) |*img| img.deinit(scene.allocator);
            }
            scene.allocator.free(decoded);
        }

        // Stage 2: in async mode the pre-decode pass is skipped entirely —
        // images decode on the scene's UploadQueue after this call returns,
        // and loadMaterials registers material slots as patch targets.
        // Sync mode keeps the fork-join predecode (single hitch, no pop-in).
        // Off-context loads are forced into async mode when the queue exists:
        // sync texture creation touches sg.* inline and would fail loudly.
        const off_context = !gpu_thread.isOnContextThread();
        const async_textures = (load_options.async_textures or off_context) and scene.uploads != null;
        var actx: ?materials_mod.AsyncTexCtx = if (async_textures)
            materials_mod.AsyncTexCtx.init(scene, gltf, base_dir, &scene.uploads.?)
        else
            null;
        if (actx) |*a| a.max_anisotropy = load_options.max_anisotropy;
        defer if (actx) |*a| a.deinit();

        if (!async_textures) {
            const t_tex: u64 = if (load_options.timings != null) sokol.time.now() else 0;
            materials_mod.decodeImagesInParallel(scene, gltf, decoded, base_dir);
            if (load_options.timings) |t| t.textures_ms = msSinceTicks(t_tex);
        }
        const t_mat: u64 = if (load_options.timings != null) sokol.time.now() else 0;
        materials_mod.loadMaterials(scene, gltf, base_dir, materials, image_cache, decoded, if (actx) |*a| a else null, load_options.max_anisotropy) catch |e| {
            // Error path: record time-to-failure so a failed upload stage
            // stays diagnosable instead of reporting a silent 0.
            if (load_options.timings) |t| t.materials_ms = msSinceTicks(t_mat);
            return e;
        };
        if (load_options.timings) |t| t.materials_ms = msSinceTicks(t_mat);

        const t_geo: u64 = if (load_options.timings != null) sokol.time.now() else 0;
        // Confined to errors inside the geometry section: a plain
        // function-scope errdefer would also fire for later failures and
        // clobber the success value written below.
        var geo_done = false;
        errdefer if (!geo_done) {
            if (load_options.timings) |t| t.geometry_ms = msSinceTicks(t_geo);
        };

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

        // From here a failure must unwind the already-created pieces
        // instead of stranding them until Scene.deinit. finishLoad takes
        // that duty over once it starts (the `cleaned` flag keeps the two
        // unwinds mutually exclusive).
        var cleaned = false;
        errdefer if (!cleaned) {
            unwindLoad(scene, materials, &spawned_meshes, null, null, null, null);
        };

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

        // 6. Parse physics colliders (extensions, extras, proxy naming)
        const pw_target = load_options.physics_world orelse scene.physics.getWorld();
        var created_bodies = std.ArrayList(*RigidBody).empty;
        errdefer {
            if (pw_target) |pw| {
                for (created_bodies.items) |b| pw.removeBody(b);
            }
            created_bodies.deinit(scene.allocator);
        }
        try colliders_mod.loadColliders(
            scene,
            gltf,
            spawned_meshes.items,
            node_mesh_start,
            node_mesh_count,
            pw_target,
            load_options.auto_colliders,
            &created_bodies,
        );

        if (load_options.timings) |t| {
            t.geometry_ms = msSinceTicks(t_geo);
        }
        geo_done = true;

        const owned_bodies = try created_bodies.toOwnedSlice(scene.allocator);
        errdefer {
            if (pw_target) |pw| {
                for (owned_bodies) |b| pw.removeBody(b);
            }
            scene.allocator.free(owned_bodies);
        }

        // Ownership: everything the loader created for this load belongs to
        // the returned asset (GltfAsset.deinit / Scene.deinit unload it).
        const asset = finishLoad(scene, materials, &spawned_meshes, owned_bodies, pw_target) catch |e| {
            // finishLoad already unwound every piece it had committed.
            cleaned = true;
            return e;
        };
        cleaned = true;
        return asset;
    }

    /// Assembles the GltfAsset from the load's pieces. Each failure rung
    /// unwinds exactly the pieces committed so far, exactly once; on
    /// success the pieces belong to the asset.
    fn finishLoad(
        scene: *Scene,
        materials: []?Material,
        spawned_meshes: *std.ArrayList(*Mesh),
        bodies: []*RigidBody,
        physics_world: ?*PhysicsWorld,
    ) !*GltfAsset {
        var count: usize = 0;
        for (materials) |m| {
            if (m) |mat| switch (mat) {
                .pbr => count += 1,
                .shader_material => {},
            };
        }
        const owned_mats = scene.allocator.alloc(*PBRMaterial, count) catch |e| {
            unwindLoad(scene, materials, spawned_meshes, null, null, bodies, physics_world);
            return e;
        };
        {
            var mi: usize = 0;
            for (materials) |m| {
                if (m) |mat| switch (mat) {
                    .pbr => |p| {
                        owned_mats[mi] = p;
                        mi += 1;
                    },
                    .shader_material => {},
                };
            }
        }
        const meshes = spawned_meshes.toOwnedSlice(scene.allocator) catch |e| {
            unwindLoad(scene, materials, spawned_meshes, owned_mats, null, bodies, physics_world);
            return e;
        };
        const asset = scene.allocator.create(GltfAsset) catch |e| {
            unwindLoad(scene, materials, spawned_meshes, owned_mats, meshes, bodies, physics_world);
            return e;
        };
        asset.* = .{
            .scene = scene,
            .meshes = meshes,
            .materials = owned_mats,
            .bodies = bodies,
            .physics_world = physics_world,
        };
        // Registration failure degrades, it does not fail the load: the
        // asset is fully valid, the caller just owns the deinit outright.
        scene.registerLoadedAsset(asset, deinitAssetThunk) catch {
            std.log.warn("agate SceneLoader: asset registration failed; caller must deinit this asset", .{});
        };
        return asset;
    }

    fn deinitAssetThunk(ctx: *anyopaque) void {
        const self: *GltfAsset = @ptrCast(@alignCast(ctx));
        self.deinit();
    }
};

pub fn isExtensionSupported(name: []const u8) bool {
    const supported = [_][]const u8{
        "EXT_meshopt_compression",
        "KHR_mesh_quantization",
        "KHR_lights_punctual",
        "KHR_texture_transform",
        "KHR_materials_unlit",
        "KHR_materials_clearcoat",
        "KHR_materials_sheen",
        "KHR_materials_transmission",
        "KHR_materials_ior",
        // KTX2/Basisu textures: the material loader resolves
        // KHR_texture_basisu images (materials.textureImage prefers
        // basisu_image when source is absent) and the KTX2 reader decodes
        // both the uncompressed and the block-compressed subsets, so a
        // file that requires this extension stays loadable. Gated on that
        // path existing — without it this entry must go.
        "KHR_texture_basisu",
        "OMI_physics_body",
        "KHR_physics_rigid_bodies",
    };
    for (supported) |s| {
        if (std.mem.eql(u8, s, name)) return true;
    }
    return false;
}
