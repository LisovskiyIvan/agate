//! Scene content-registry API: material/mesh create/destroy, mesh search
//! by name/tag/query. Split out of `scene.zig` (facade). `destroyMesh`
//! keeps its cross-layer referent cleanup (outline/highlights/soft
//! bodies/physics/hierarchy/LOD/animation/decals) plus the epoch-retire
//! off-context branch.
//!
/// Anti-cycle rule (same as `audio/*`, `profiler/*`): every function takes
/// the scene as `anytype` (a `*Scene` from `core.zig` in practice) and this
/// module never imports `core.zig` or the `scene.zig` facade back.
/// Cross-leaf helpers consumed here are `pub` in their home module but are
/// deliberately NOT re-exported by the facade.
const std = @import("std");
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;
const StandardMaterial = @import("../material.zig").StandardMaterial;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const ShaderMaterial = @import("../material.zig").ShaderMaterial;
const TagQuery = @import("../tags.zig").TagQuery;
const gpu_thread = @import("../gpu_thread.zig");

// ---- Content registries: materials & meshes. ----

pub fn createStandardMaterial(self: anytype, name: []const u8) !*StandardMaterial {
    const mat = try self.allocator.create(StandardMaterial);
    mat.* = StandardMaterial.init(name);
    try self.materials.append(self.allocator, mat);
    return mat;
}

pub fn createPBRMaterial(self: anytype, name: []const u8) !*PBRMaterial {
    const mat = try self.allocator.create(PBRMaterial);
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
    // Decal expiration calls this from Scene.update on the game thread,
    // where sg.destroyBuffer is illegal: unlink now, destroy the GPU
    // resources at the next render-start flush on the context thread
    // (epoch-ретенция: запись ждёт завершения текущего кадра).
    if (!gpu_thread.isOnContextThread()) {
        self.gpu_retire.retireMesh(self.allocator, mesh);
        return;
    }
    mesh.deinit(self.allocator);
    self.allocator.destroy(mesh);
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
