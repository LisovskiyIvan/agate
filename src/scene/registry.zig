//! Scene content-registry API: material/mesh create/destroy, mesh search
//! by name/tag/query. Split out of `scene.zig` (facade). `destroyMesh`
//! keeps its cross-layer referent cleanup (outline/highlights/soft
//! bodies/physics/hierarchy/LOD/animation/decals/trail targets) plus the
//! epoch-retire off-context branch.
//!
/// Anti-cycle rule (same as `audio/*`, `profiler/*`): every function takes
/// the scene as `anytype` (a `*Scene` from `core.zig` in practice) and this
/// module never imports `core.zig` or the `scene.zig` facade back.
/// Cross-leaf helpers consumed here are `pub` in their home module but are
/// deliberately NOT re-exported by the facade.
const std = @import("std");
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const ShaderMaterial = @import("../material.zig").ShaderMaterial;
const TagQuery = @import("../tags.zig").TagQuery;
const gpu_thread = @import("../gpu_thread.zig");
const trail_mod = @import("../mesh/trail.zig");
const TrailMesh = trail_mod.TrailMesh;

// ---- Content registries: materials & meshes. ----
//
// Ownership summary (see agate/API.md for the user-facing version):
// - Meshes/materials are Scene-owned: creation appends to a registry list,
//   destruction unlinks + frees through the matching destroy below.
// - Mesh GPU buffers die on the context thread only: `destroyMesh` destroys
//   inline on-context, otherwise unlinks now and retires into `gpu_retire`
//   for the next render-start flush (or `deinit`, which drains everything).
// - Mesh names are plain `[]const u8` slices with an `owns_name` flag (Zig
//   has no field privacy; the flag is convention, not enforcement):
//   borrowed (`owns_name == false`, e.g. string literals or builder inputs)
//   or Scene-allocator-owned (`owns_name == true`, freed in `Mesh.deinit`).
//   Mutate names only through `renameMesh` below.
// - Particle systems are Scene-owned until `deinit` (context-thread only):
//   there is deliberately NO `destroyParticleSystem` — `ParticleSystem.deinit`
//   issues `sg.destroy*` inline (buffers, compute views/pipelines, owned
//   textures), which is illegal off-context, and the retire queue has no
//   particle entry kind. Removing one mid-life would also need to scrub
//   sub-emitter back-references and the prepared/build frames that borrow
//   its handle ids by value. Create systems sparingly and reuse them.

pub fn createPBRMaterial(self: anytype, name: []const u8) !*PBRMaterial {
    const mat = try self.allocator.create(PBRMaterial);
    errdefer self.allocator.destroy(mat);
    mat.* = PBRMaterial.init(name);
    try self.pbr_materials.append(self.allocator, mat);
    return mat;
}

// Creates a custom-shader material bound to a registered shader (by
// name — build.zig `user_shader_materials` or registerRuntime). Returns
// null when the shader name is not registered.
pub fn createShaderMaterial(self: anytype, name: []const u8, shader_name: []const u8) ?*ShaderMaterial {
    const mat = self.allocator.create(ShaderMaterial) catch return null;
    mat.* = ShaderMaterial.initForShader(shader_name, name) orelse {
        self.allocator.destroy(mat);
        return null;
    };
    self.shader_materials.append(self.allocator, mat) catch {
        self.allocator.destroy(mat);
        return null;
    };
    return mat;
}

/// Unlinks `mat` from the registry and frees the CPU material. Safe under
/// update||render WITHOUT any GPU retire: prepared draw records carry
/// GPU handle VALUES (views/samplers), never CPU material refs, and
/// materials own no GPU objects needing deinit — the in-flight frame's
/// baked copies stay valid. (No audit-driven retire-all-materials/textures
/// queue: that would be a false-positive fix for a non-issue.)
pub fn destroyPBRMaterial(self: anytype, mat: *PBRMaterial) void {
    for (self.pbr_materials.items, 0..) |m, i| {
        if (m == mat) {
            _ = self.pbr_materials.swapRemove(i);
            break;
        }
    }
    self.allocator.destroy(mat);
}

/// Loader-owned asset registration (see SceneLoader.GltfAsset). The scene
/// stores only an opaque pointer + destructor so this module (and the scene
/// facade) never imports the loader — the anti-cycle rule. `Scene.deinit`
/// runs every remaining destructor, so a caller who forgets
/// `GltfAsset.deinit` still cannot leak GPU memory past the scene.
pub const RegisteredAsset = struct {
    ctx: *anyopaque,
    deinit_fn: *const fn (ctx: *anyopaque) void,
};

pub fn registerLoadedAsset(self: anytype, ctx: *anyopaque, deinit_fn: *const fn (ctx: *anyopaque) void) !void {
    try self.loaded_assets.append(self.allocator, .{ .ctx = ctx, .deinit_fn = deinit_fn });
}

pub fn unregisterLoadedAsset(self: anytype, ctx: *anyopaque) void {
    for (self.loaded_assets.items, 0..) |a, i| {
        if (a.ctx == ctx) {
            _ = self.loaded_assets.swapRemove(i);
            return;
        }
    }
}

/// Runs every remaining asset destructor (each normally unregisters itself;
/// entries are popped first so a destructor that forgets cannot loop).
pub fn deinitLoadedAssets(self: anytype) void {
    while (self.loaded_assets.items.len > 0) {
        const a = self.loaded_assets.items[self.loaded_assets.items.len - 1];
        self.loaded_assets.items.len -= 1;
        a.deinit_fn(a.ctx);
    }
    self.loaded_assets.deinit(self.allocator);
    self.loaded_assets = .empty;
}

pub fn removeMesh(self: anytype, mesh: *Mesh) bool {
    for (self.meshes.items, 0..) |m, i| {
        if (m == mesh) {
            _ = self.meshes.swapRemove(i);
            return true;
        }
    }
    return false;
}

pub fn destroyMesh(self: anytype, mesh: *Mesh) void {
    // NOT safe inside `PhysicsWorld.step`: the body unlink below defers to
    // end-of-step, but the mesh storage is freed synchronously here — a
    // body still pointing at it for the rest of the step would read freed
    // memory in push/pull. Poll physics events after `step` and destroy
    // then; never from a callback running inside the step window.
    //
    // Frees the mesh (GPU buffers + CPU mirrors) only. Materials and
    // textures are NOT owned by the mesh: loader content must be unloaded
    // through `SceneLoader.releaseGlb`, which frees the loader-created
    // materials and their textures exactly once per load.
    _ = self.removeMesh(mesh);
    for (self.outline_meshes.items, 0..) |m, i| {
        if (m == mesh) {
            _ = self.outline_meshes.swapRemove(i);
            break;
        }
    }
    // Highlights: drop entries bound to the destroyed mesh (same
    // referent cleanup as outline_meshes above; the entry owns no GPU,
    // so both the sync and the off-context epoch-retired branches are
    // covered synchronously here — already-staged items fail closed on
    // their borrowed handles at draw time).
    self.highlights.removeForMesh(mesh);
    // Referent cleanup: neutralize every cross-mesh reference to `mesh`
    // before its storage is freed. Runs before the sync/deferred branch
    // below so both paths are covered. Never cascade-destroys: orphaned
    // meshes stay alive under their own transform.
    // Soft bodies: drop the cloth body bound to this mesh, if any (frees
    // the solver side only; the mesh itself proceeds below as usual).
    self.softbodies.removeForMesh(self.allocator, mesh);
    // Physics: drop the rigid body bound to this mesh, if any.
    if (self.physics.getWorld()) |pw| {
        if (pw.findBody(mesh)) |body| pw.removeBody(body);
    }
    // Hierarchy: orphan children and detach bone attachments hosted by
    // the destroyed mesh (this also covers decal meshes parented to a
    // destroyed target via createDecal).
    for (self.meshes.items) |child| {
        if (child.parent == mesh) child.parent = null;
        if (child.attach_bone) |ab| {
            if (ab.host_mesh == mesh) child.detachFromBone();
        }
    }
    // LOD: order-preserving removal keeps the distance-sorted band
    // order; the parent renders its own geometry for the freed band
    // instead of holding a dangling pointer.
    for (self.meshes.items) |other| {
        var i: usize = 0;
        while (i < other.lod_levels.items.len) {
            if (other.lod_levels.items[i].mesh == mesh) {
                _ = other.lod_levels.orderedRemove(i);
            } else {
                i += 1;
            }
        }
    }
    // Animation groups: morph targets bind slices INTO the mesh (weights
    // slice + the dirty flag), unlike node transforms whose pointers are
    // scene-owned. Tombstone matching targets instead of removing them so
    // NodeChannel.target indices stay valid (applyNodesAtTime skips empty
    // weight slices) and free the group-owned rest snapshot.
    for (self.animation_groups.items) |ag| {
        for (ag.morph_targets) |*mt| {
            const binds_mesh = mt.dirty == &mesh.morph_dirty or
                (mt.weights.len > 0 and mt.weights.ptr == mesh.morph_weights.ptr);
            if (!binds_mesh) continue;
            // rest_weights belongs to the group's allocator, not the
            // scene's; today they are the same instance.
            if (mt.rest_weights.len > 0) ag.allocator.free(mt.rest_weights);
            mt.weights = &.{};
            mt.rest_weights = &.{};
            mt.dirty = null;
        }
    }
    // Decals: drop manager instances of this mesh and free their
    // material so neither dangles nor leaks. Safe against DecalManager
    // iteration: update/destroyOldest/clear remove the instance BEFORE
    // calling destroyMesh, so this scan only mutates the list on
    // user-initiated destroys where no manager iteration is in flight.
    if (self.decals.manager) |*dm| {
        var i: usize = 0;
        while (i < dm.instances.items.len) {
            if (dm.instances.items[i].mesh == mesh) {
                const removed = dm.instances.orderedRemove(i);
                self.destroyPBRMaterial(removed.material);
            } else {
                i += 1;
            }
        }
    }
    // Trails: clear follow targets bound to the destroyed mesh so
    // followers never read retired/freed storage in `TrailMesh.update`.
    // Runs before the sync/deferred branch below so both paths are covered
    // (same placement as the highlight scrub above); `destroyTrailMesh`
    // relies on this instead of scrubbing itself.
    for (self.trails.meshes.items) |other| {
        if (other.target == mesh) other.target = null;
    }
    // Decal expiration calls this from Scene.update on the game thread,
    // where sg.destroyBuffer is illegal: unlink now, destroy the GPU
    // resources at the next render-start flush on the context thread
    // (epoch retention: entry waits for the current frame to complete).
    if (!gpu_thread.isOnContextThread()) {
        self.gpu_retire.retireMesh(self.allocator, mesh);
        return;
    }
    mesh.deinit(self.allocator);
    self.allocator.destroy(mesh);
}

/// Renames a Scene-owned mesh, taking ownership of an internal copy of
/// `new_name` (the caller's slice is borrowed, never retained).
///
/// Semantics:
/// - The copy is allocated BEFORE the old name is freed, so an aliased
///   input (`renameMesh(m, m.name)`, or a subslice of it) is safe: the new
///   copy lands first, then the old allocation drops.
/// - Failure is atomic: on `OutOfMemory` the mesh keeps its old name and
///   `owns_name` flag untouched.
/// - After success `mesh.owns_name` is always true (even for an empty
///   name, which holds no allocation and frees nothing in `Mesh.deinit`).
/// - The mesh keeps its registry slot: only the name changes, so a
///   subsequent `getMeshByName` resolves the new name and no longer the old
///   one (unless another mesh still carries it).
/// - Game-thread safe (pure CPU: one dupe + one free, no `sg.*`). Callers
///   must pass a mesh owned by this scene; membership is not re-checked.
pub fn renameMesh(self: anytype, mesh: *Mesh, new_name: []const u8) !void {
    const owned = try self.allocator.dupe(u8, new_name);
    if (mesh.owns_name and mesh.name.len > 0) {
        self.allocator.free(mesh.name);
    }
    mesh.name = owned;
    mesh.owns_name = true;
}

/// Destroys a Scene-owned trail: unlinks it from the trail layer, destroys
/// its linked scene mesh through `destroyMesh` (same epoch-retire contract:
/// inline on the context thread, unlinked + retired off-context for the
/// next render-start flush; `destroyMesh` also clears other trails targeting
/// that mesh), then frees the trail's CPU staging (`TrailMesh.deinit` is
/// CPU-only: nodes, vertex/index mirrors) and the struct itself.
///
/// Game-thread safe via the `destroyMesh` retire path. Callers must pass a
/// trail owned by this scene; membership is not re-checked beyond the
/// unlink scan.
pub fn destroyTrailMesh(self: anytype, trail: *TrailMesh) void {
    for (self.trails.meshes.items, 0..) |t, i| {
        if (t == trail) {
            _ = self.trails.meshes.swapRemove(i);
            break;
        }
    }
    const mesh = trail.mesh;
    destroyMesh(self, mesh);
    trail.deinit();
    self.allocator.destroy(trail);
}

// ---- Mesh search, tags & queries ----

pub fn getMeshByName(self: anytype, name: []const u8) ?*Mesh {
    for (self.meshes.items) |m| {
        if (std.mem.eql(u8, m.name, name)) return m;
    }
    return null;
}

/// Returns a list of all scene meshes that have the specified tag (case-insensitive).
pub fn getMeshesByTag(self: anytype, allocator: std.mem.Allocator, tag_str: []const u8) !std.ArrayListUnmanaged(*Mesh) {
    var list = std.ArrayListUnmanaged(*Mesh).empty;
    errdefer list.deinit(allocator);
    for (self.meshes.items) |m| {
        if (m.hasTag(tag_str)) {
            try list.append(allocator, m);
        }
    }
    return list;
}

/// Returns a list of all scene meshes matching the given boolean tag query expression (e.g. "enemy & (boss | elite)").
pub fn getMeshesByQuery(self: anytype, allocator: std.mem.Allocator, query_str: []const u8) !std.ArrayListUnmanaged(*Mesh) {
    var list = std.ArrayListUnmanaged(*Mesh).empty;
    errdefer list.deinit(allocator);
    var q = try TagQuery.parse(allocator, query_str);
    defer q.deinit();

    for (self.meshes.items) |m| {
        if (m.tags.matches(&q)) {
            try list.append(allocator, m);
        }
    }
    return list;
}

/// Counts how many scene meshes have the specified tag.
pub fn countMeshesByTag(self: anytype, tag_str: []const u8) usize {
    var n: usize = 0;
    for (self.meshes.items) |m| {
        if (m.hasTag(tag_str)) n += 1;
    }
    return n;
}

/// Counts how many scene meshes match the given boolean tag query expression.
pub fn countMeshesByQuery(self: anytype, query_str: []const u8) usize {
    var n: usize = 0;
    for (self.meshes.items) |m| {
        if (m.matchesTagQuery(query_str)) n += 1;
    }
    return n;
}

/// Finds the first scene mesh with the specified tag, or null if none found.
pub fn findFirstMeshByTag(self: anytype, tag_str: []const u8) ?*Mesh {
    for (self.meshes.items) |m| {
        if (m.hasTag(tag_str)) return m;
    }
    return null;
}

/// Finds the first scene mesh matching the boolean tag query expression, or null if none found.
pub fn findFirstMeshByQuery(self: anytype, query_str: []const u8) ?*Mesh {
    for (self.meshes.items) |m| {
        if (m.matchesTagQuery(query_str)) return m;
    }
    return null;
}
