const std = @import("std");

const Mesh = @import("../mesh.zig").Mesh;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const ShaderMaterial = @import("../material.zig").ShaderMaterial;
const Texture = @import("../texture.zig").Texture;
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

/// Destroys PBR materials. The same texture (identical image/view/sampler
/// handle set) can be shared across material slots and materials, so each
/// distinct handle set is destroyed exactly once (same dedup rule as
/// SceneLoader.GltfAsset.deinit). Covers every texture-bearing slot:
/// albedo/normal/metallic-roughness/emissive/occlusion plus clearcoat mask,
/// sheen color and the environment cube.
pub fn deinitPbrMaterials(allocator: std.mem.Allocator, pbr_materials: *std.ArrayListUnmanaged(*PBRMaterial)) void {
    var destroyed = std.AutoHashMap(TextureHandleSet, void).init(allocator);
    defer destroyed.deinit();

    for (pbr_materials.items) |mat| {
        inline for (.{ "albedo_texture", "normal_texture", "metallic_roughness_texture", "emissive_texture", "occlusion_texture" }) |field| {
            if (@field(mat, field)) |*t| {
                destroyTextureOnce(&destroyed, t);
            }
        }
        if (mat.clearcoat.mask_texture) |*t| destroyTextureOnce(&destroyed, t);
        if (mat.sheen.color_texture) |*t| destroyTextureOnce(&destroyed, t);
        if (mat.environment_texture) |*cube| {
            const key = TextureHandleSet{ .image = cube.image.id, .view = cube.view.id, .sampler = cube.sampler.id };
            if ((cube.image.id != 0 or cube.view.id != 0 or cube.sampler.id != 0) and !destroyed.contains(key)) {
                destroyed.put(key, {}) catch {};
                cube.deinit();
            }
        }
        allocator.destroy(mat);
    }
    pbr_materials.deinit(allocator);
}

/// Dedup key: one texture's full GPU handle set.
const TextureHandleSet = struct { image: u32, view: u32, sampler: u32 };

/// Destroys `t` unless its handle set was already destroyed in this pass.
fn destroyTextureOnce(destroyed: *std.AutoHashMap(TextureHandleSet, void), t: *Texture) void {
    const key = TextureHandleSet{ .image = t.image.id, .view = t.view.id, .sampler = t.sampler.id };
    if ((t.image.id != 0 or t.view.id != 0 or t.sampler.id != 0) and !destroyed.contains(key)) {
        destroyed.put(key, {}) catch {};
        t.deinit();
    }
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
