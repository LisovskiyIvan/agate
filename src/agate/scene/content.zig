const std = @import("std");

const Mesh = @import("../mesh.zig").Mesh;
const StandardMaterial = @import("../material.zig").StandardMaterial;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const ShaderMaterial = @import("../material.zig").ShaderMaterial;
const AnimationGroup = @import("../animation/animation.zig").AnimationGroup;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;

/// Teardown helpers for the Scene content registries (meshes, materials,
/// animation data). The registries themselves stay flat Scene fields —
/// loaders, sandbox and serialization iterate them directly — but the
/// destruction rules (texture dedup, ownership) live here.
/// Destroys every mesh (GPU buffers + CPU mirrors) and the list.
pub fn deinitMeshes(allocator: std.mem.Allocator, meshes: *std.ArrayListUnmanaged(*Mesh)) void {
    for (meshes.items) |m| {
        m.deinit(allocator);
        allocator.destroy(m);
    }
    meshes.deinit(allocator);
}

/// Destroys standard materials with their diffuse textures and the list.
pub fn deinitMaterials(allocator: std.mem.Allocator, materials: *std.ArrayListUnmanaged(*StandardMaterial)) void {
    for (materials.items) |mat| {
        if (mat.diffuse_texture) |*t| {
            t.deinit();
        }
        allocator.destroy(mat);
    }
    materials.deinit(allocator);
}

/// Destroys shader materials (texture ownership mirrors deinitMaterials) and
/// the list.
pub fn deinitShaderMaterials(allocator: std.mem.Allocator, materials: *std.ArrayListUnmanaged(*ShaderMaterial)) void {
    for (materials.items) |mat| {
        if (mat.texture) |*t| {
            t.deinit();
        }
        allocator.destroy(mat);
    }
    materials.deinit(allocator);
}

/// Destroys PBR materials. The same texture view can be shared across
/// material slots, so views are destroyed exactly once (dedup by view id).
pub fn deinitPbrMaterials(allocator: std.mem.Allocator, pbr_materials: *std.ArrayListUnmanaged(*PBRMaterial)) void {
    var destroyed_views = std.AutoHashMap(u32, void).init(allocator);
    defer destroyed_views.deinit();

    for (pbr_materials.items) |mat| {
        inline for (.{ "albedo_texture", "normal_texture", "metallic_roughness_texture", "emissive_texture", "occlusion_texture" }) |field| {
            if (@field(mat, field)) |*t| {
                if (t.view.id != 0 and !destroyed_views.contains(t.view.id)) {
                    destroyed_views.put(t.view.id, {}) catch {};
                    t.deinit();
                }
            }
        }
        allocator.destroy(mat);
    }
    pbr_materials.deinit(allocator);
}

/// Deinitializes animation groups and skeletons (CPU skinning data).
pub fn deinitAnimations(
    allocator: std.mem.Allocator,
    animation_groups: *std.ArrayListUnmanaged(*AnimationGroup),
    skeletons: *std.ArrayListUnmanaged(*Skeleton),
) void {
    for (animation_groups.items) |ag| {
        ag.deinit();
    }
    animation_groups.deinit(allocator);

    for (skeletons.items) |skel| {
        skel.deinit();
    }
    skeletons.deinit(allocator);
}
