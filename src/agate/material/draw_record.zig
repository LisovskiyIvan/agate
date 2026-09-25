const sokol = @import("sokol");
const sg = sokol.gfx;
const std = @import("std");
const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const shader_material = @import("../shader_material.zig");

const types = @import("types.zig");
const StandardMaterial = @import("standard.zig").StandardMaterial;
const PBRMaterial = @import("pbr.zig").PBRMaterial;
const ShaderMaterial = @import("shader_mat.zig").ShaderMaterial;
const union_mod = @import("union.zig");
const Material = union_mod.Material;

/// Compact, GPU-ready draw record containing factors, UV transforms,
/// alpha cutoff, and texture handles. Built at queue-build time; the render passes
/// draw from this record without inspecting mutable material state on the mesh.
pub const MaterialDrawRecord = struct {
    // Texture views & samplers (or defaults)
    albedo_view: sg.View = .{},
    albedo_sampler: sg.Sampler = .{},
    normal_view: sg.View = .{},
    mr_view: sg.View = .{},
    emissive_view: sg.View = .{},
    occlusion_view: sg.View = .{},
    /// PBR layer masks (v1): clearcoat R-mask + sheen rgb-tint, staged like
    /// every other map (white fallback = identity when unset). Sampled
    /// through the shared data_smp, so no extra sampler lane is needed.
    clearcoat_view: sg.View = .{},
    sheen_view: sg.View = .{},
    data_sampler: sg.Sampler = .{},
    env_view: ?sg.View = null,
    env_sampler: ?sg.Sampler = null,

    // Factors & params
    base_color: [4]f32 = .{ 1, 1, 1, 1 },
    pbr_factors: [4]f32 = .{ 0, 0.5, 1.0, 1.0 }, // metallic, roughness, occlusion_strength, env_intensity
    emissive_color: [4]f32 = .{ 0, 0, 0, 1 },
    normal_scale: f32 = 1.0,
    alpha_cutoff: f32 = 0.0,

    // UV transforms & channel selectors
    uv_matrices: [5][4]f32 = @splat(.{ 1, 0, 0, 1 }),
    uv_offsets: [5][4]f32 = @splat(.{ 0, 0, 0, 0 }),
    channel_selectors: [4]f32 = .{ 0, 1, 2, 0 },

    // Standard material diffuse UV matrix / offset
    standard_uv_matrix: [4]f32 = .{ 1, 0, 0, 1 },
    standard_uv_offset: [4]f32 = .{ 0, 0, 0, 0 },
};

/// Render-owned копия изменяемых CPU-данных hook-материала: draw-путь читает
/// только этот снимок, живой ShaderMaterial (tint/uniforms/texture/entry)
/// во время отрисовки не трогается. Резолюция entry_index через глобальный
/// реестр остаётся заимствованием (как GPU-хендлы под фазовым мьютексом P3).
/// Хранится в side-таблице очередей (только для shader-draws), чтобы не
/// раздувать каждую запись фиксированной ценой uniform-блока.
pub const ShaderDrawSnapshot = struct {
    entry_index: u32 = shader_material.invalid_index,
    tint: [4]f32 = .{ 1, 1, 1, 1 },
    tex_view: sg.View = .{},
    tex_sampler: sg.Sampler = .{},
    /// Frozen second-texture handles (runtime sources only; the draw path
    /// binds them to view/sampler slot 1).
    tex1_view: sg.View = .{},
    tex1_sampler: sg.Sampler = .{},
    uniforms: shader_material.UniformStorage = .{.{ 0, 0, 0, 0 }} ** shader_material.merge.user_slot_count,
    /// Собственный double_sided материала (без decal-форсинга item: раньше draw
    /// читал sm.double_sided напрямую, поведение сохранено точь-в-точь).
    double_sided: bool = false,
};

/// Строит ShaderDrawSnapshot из живого материала (только prepare-фаза).
/// Null для всех не-shader материалов — их draw-пути снимок не используют.
pub fn buildShaderSnapshot(mat: ?Material, default_white: *const Texture) ?ShaderDrawSnapshot {
    const m = mat orelse return null;
    if (m != .shader_material) return null;
    const sm = m.shader_material;
    const tex = sm.texture orelse default_white.*;
    const tex1 = sm.texture1 orelse default_white.*;
    return .{
        .entry_index = sm.entry_index,
        .tint = sm.getTintColor4(),
        .tex_view = tex.view,
        .tex_sampler = tex.sampler,
        .tex1_view = tex1.view,
        .tex1_sampler = tex1.sampler,
        .uniforms = sm.uniforms,
        .double_sided = sm.double_sided,
    };
}

pub fn buildDrawRecord(
    mat: ?Material,
    default_material: *const StandardMaterial,
    default_white: *const Texture,
    default_normal: *const Texture,
    default_cube: *const CubeTexture,
    sky_texture: ?CubeTexture,
    ibl_intensity: f32,
) MaterialDrawRecord {
    var rec = MaterialDrawRecord{};
    if (mat) |m| {
        switch (m) {
            .pbr => |p| {
                const albedo_tex = p.albedo_texture orelse default_white.*;
                const normal_tex = p.normal_texture orelse default_normal.*;
                const mr_tex = p.metallic_roughness_texture orelse default_white.*;
                const emissive_tex = p.emissive_texture orelse default_white.*;
                const occlusion_tex = p.occlusion_texture orelse default_white.*;
                const clearcoat_tex = p.clearcoat.mask_texture orelse default_white.*;
                const sheen_tex = p.sheen.color_texture orelse default_white.*;

                rec.albedo_view = albedo_tex.view;
                rec.albedo_sampler = albedo_tex.sampler;
                rec.normal_view = normal_tex.view;
                rec.mr_view = mr_tex.view;
                rec.emissive_view = emissive_tex.view;
                rec.occlusion_view = occlusion_tex.view;
                rec.clearcoat_view = clearcoat_tex.view;
                rec.sheen_view = sheen_tex.view;

                const data_tex = p.normal_texture orelse p.metallic_roughness_texture orelse p.occlusion_texture orelse p.emissive_texture orelse albedo_tex;
                rec.data_sampler = data_tex.sampler;

                if (p.environment_texture) |env_t| {
                    rec.env_view = env_t.view;
                    rec.env_sampler = env_t.sampler;
                } else if (sky_texture) |st| {
                    rec.env_view = st.view;
                    rec.env_sampler = st.sampler;
                } else {
                    rec.env_view = default_cube.view;
                    rec.env_sampler = default_cube.sampler;
                }

                rec.base_color = p.getAlbedoColor4();
                rec.pbr_factors = .{
                    p.metallic,
                    p.roughness,
                    p.occlusion_strength,
                    ibl_intensity * p.environment_intensity,
                };
                rec.emissive_color = .{ p.emissive_color.r, p.emissive_color.g, p.emissive_color.b, 1.0 };
                rec.normal_scale = p.normal_scale;
                rec.alpha_cutoff = if (p.alpha_mode == .cutout) p.alpha_cutoff else 0.0;

                rec.uv_matrices[0] = p.albedo_uv_transform.matrixRows();
                rec.uv_matrices[1] = p.normal_uv_transform.matrixRows();
                rec.uv_matrices[2] = p.metallic_roughness_uv_transform.matrixRows();
                rec.uv_matrices[3] = p.emissive_uv_transform.matrixRows();
                rec.uv_matrices[4] = p.occlusion_uv_transform.matrixRows();

                rec.uv_offsets[0] = p.albedo_uv_transform.offsetPacked();
                if (p.unlit) rec.uv_offsets[0][2] = 1.0;
                rec.uv_offsets[1] = p.normal_uv_transform.offsetPacked();
                rec.uv_offsets[2] = p.metallic_roughness_uv_transform.offsetPacked();
                rec.uv_offsets[3] = p.emissive_uv_transform.offsetPacked();
                rec.uv_offsets[4] = p.occlusion_uv_transform.offsetPacked();

                rec.channel_selectors = .{
                    p.occlusion_channel.selector(),
                    p.roughness_channel.selector(),
                    p.metallic_channel.selector(),
                    0,
                };
            },
            .standard => |s| {
                const tex = s.diffuse_texture orelse default_white.*;
                rec.albedo_view = tex.view;
                rec.albedo_sampler = tex.sampler;
                rec.base_color = s.getDiffuseColor4();
                rec.alpha_cutoff = if (s.alpha_mode == .cutout) s.alpha_cutoff else 0.0;
                rec.standard_uv_matrix = s.diffuse_uv_transform.matrixRows();
                rec.standard_uv_offset = s.diffuse_uv_transform.offsetPacked();
                if (s.unlit) rec.standard_uv_offset[2] = 1.0;
            },
            .shader_material => |sm| {
                const tex = sm.texture orelse default_white.*;
                rec.albedo_view = tex.view;
                rec.albedo_sampler = tex.sampler;
                rec.base_color = sm.getTintColor4();
                rec.alpha_cutoff = if (sm.alpha_mode == .cutout) sm.alpha_cutoff else 0.0;
            },
        }
    } else {
        const tex = default_material.diffuse_texture orelse default_white.*;
        rec.albedo_view = tex.view;
        rec.albedo_sampler = tex.sampler;
        rec.base_color = default_material.getDiffuseColor4();
        rec.alpha_cutoff = if (default_material.alpha_mode == .cutout) default_material.alpha_cutoff else 0.0;
        rec.standard_uv_matrix = default_material.diffuse_uv_transform.matrixRows();
        rec.standard_uv_offset = default_material.diffuse_uv_transform.offsetPacked();
    }
    return rec;
}
