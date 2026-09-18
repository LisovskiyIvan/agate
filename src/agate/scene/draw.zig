const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const shd = @import("shader");
const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_shd = @import("instanced_shader");
const inst_pbr_shd = @import("instanced_pbr_shader");

const math = @import("math");
const Mat4 = math.Mat4;

const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const material_mod = @import("../material.zig");
const StandardMaterial = material_mod.StandardMaterial;
const PBRMaterial = material_mod.PBRMaterial;
const Material = material_mod.Material;
const passes = @import("../passes/mod.zig");

const render_queue = @import("render_queue.zig");
const RenderMeshItem = render_queue.RenderMeshItem;
const RenderInstancedBatch = render_queue.RenderInstancedBatch;
const morph_gpu = @import("../mesh/morph_gpu.zig");
const MAX_BONES = @import("../animation/skeleton.zig").MAX_BONES;
const uniforms = @import("uniforms.zig");
const forward_pipelines = @import("forward_pipelines.zig");
const ForwardPipelines = forward_pipelines.ForwardPipelines;
const shader_material = @import("../shader_material.zig");
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

pub const FrameContext = uniforms.FrameContext;

/// Per-frame render state the draw helpers need, passed explicitly (this
/// module must not import scene.zig). Scene fills it once per frame from the
/// consumed frame snapshot; every value below is a render-owned COPY — the
/// draw never dereferences game-mutatable Scene fields, so update may run
/// concurrently with render. The only mutations are the lazy shader-material
/// pipeline cache inside `pipelines` (render thread only) and `stats`
/// (context-owned counters).
pub const Environment = struct {
    // Mutable: shader-material pipelines are created lazily on first use
    // (ShaderMaterialCache.getOrCreate, render thread only); everything else
    // is a read-only snapshot copy.
    pipelines: *ForwardPipelines,
    stats: *SceneStats,
    // Fallback textures for meshes/materials without their own textures:
    // render-owned COPIES from the frame snapshot (Scene.default_* captured
    // at prepare), never the live Scene fields.
    default_white: Texture,
    default_normal: Texture,
    default_cube: CubeTexture,
    // Environment IBL cubemap: a material-level probe overrides the skybox,
    // which in turn falls back to default_cube. Snapshot copy as well.
    sky_texture: ?CubeTexture,
    ibl_intensity: f32,
    // Shadow map views/samplers from the CSM/spot atlas.
    shadow_pass: *const passes.ShadowPass,
    // Scene-level fragment uniform inputs; mesh.receive_shadows is patched
    // per draw before building the shader uniforms.
    shadow_uniforms: uniforms.ShadowState,
};

// Draws one regular (non-instanced) queue item: pipeline select, bind,
// uniforms, draw. Shared by the opaque pass and the transparent pass
// (pipeline choice follows item.transparent; double-sided items additionally
// resolve to the cull-off twins via pipelines.forRegularItem).
// Cutout items ride the opaque pass; their alpha_cutoff uniform enables the
// in-shader discard. Updates stats.
//
// P4: читает только render-owned данные item/draw_record плюс срезы хранилищ
// той же очереди (skins/shaders уже после всех реаллокаций prepare-фазы).
// Живые Mesh/Material/Skeleton здесь недоступны по построению.
pub fn drawRegularItem(
    env: *const Environment,
    item: RenderMeshItem,
    ctx: *const FrameContext,
    current_pipeline_id: *u32,
    skins: []const [MAX_BONES]Mat4,
    shaders: []const material_mod.ShaderDrawSnapshot,
) void {
    const model = item.model;
    const mvp = Mat4.mul(ctx.view_proj, model);
    const rec = item.draw_record;

    // Shader materials take their own path: pipeline from the lazy
    // ShaderMaterialCache, bindings/uniforms per the registration's base
    // template contract (identical layouts — see drawShaderMaterialItem).
    if (item.shader_index) |s_idx| {
        if (s_idx < shaders.len) {
            return drawShaderMaterialItem(env, item, ctx, current_pipeline_id, shaders[s_idx], mvp);
        }
        return;
    }

    const pip_id = env.pipelines.forRegularItem(item);

    if (pip_id == 0) return;
    if (pip_id != current_pipeline_id.*) {
        sg.applyPipeline(.{ .id = pip_id });
        current_pipeline_id.* = pip_id;
        env.stats.pipeline_switches += 1;
    }

    var bind = sg.Bindings{};
    bind.vertex_buffers[0] = item.vertex_buffer;
    bind.index_buffer = item.index_buffer;

    const morph_view = item.morph_view;
    const morph_uniforms = item.morph_uniforms;

    if (item.is_pbr) {
        bind.views[pbr_shd.VIEW_albedo_tex] = rec.albedo_view;
        bind.views[pbr_shd.VIEW_normal_tex] = rec.normal_view;
        bind.views[pbr_shd.VIEW_metallic_roughness_tex] = rec.mr_view;
        bind.views[pbr_shd.VIEW_emissive_tex] = rec.emissive_view;
        bind.views[pbr_shd.VIEW_occlusion_tex] = rec.occlusion_view;
        bind.samplers[pbr_shd.SMP_smp] = rec.albedo_sampler;
        bind.samplers[pbr_shd.SMP_data_smp] = rec.data_sampler;

        // Environment IBL Cubemap & Shadow Depth Map
        const cube_view = rec.env_view orelse (env.sky_texture orelse env.default_cube).view;
        const cube_sampler = rec.env_sampler orelse (env.sky_texture orelse env.default_cube).sampler;
        bind.views[pbr_shd.VIEW_env_tex] = cube_view;
        bind.samplers[pbr_shd.SMP_env_smp] = cube_sampler;

        bind.views[pbr_shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
        bind.views[pbr_shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
        bind.views[pbr_shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
        bind.samplers[pbr_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
        bind.samplers[pbr_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;

        bind.views[pbr_shd.VIEW_morph_tex] = morph_view;
        bind.samplers[pbr_shd.SMP_morph_smp] = env.pipelines.morph_sampler;

        sg.applyBindings(bind);

        const vs_params = pbr_shd.VsParams{
            .mvp = mvp,
            .model = model,
        };
        sg.applyUniforms(pbr_shd.UB_vs_params, sg.asRange(&vs_params));

        const skel_bones: ?*const [MAX_BONES]Mat4 = if (item.is_skinned)
            // Битый индекс skinned-записи (билдером недостижимо): пропуск draw
            // вместо аплоада stale-униформы чужого draw.
            render_queue.skinAt(skins, item.skin_index) orelse return
        else
            null;
        if (skel_bones) |bones| {
            const vs_skin = skinned_pbr_shd.VsSkin{
                .bones = bones.*,
            };
            sg.applyUniforms(skinned_pbr_shd.UB_vs_skin, sg.asRange(&vs_skin));
            sg.applyUniforms(skinned_pbr_shd.UB_vs_morph, sg.asRange(&vsMorphUniform(skinned_pbr_shd, morph_uniforms)));
        } else {
            sg.applyUniforms(pbr_shd.UB_vs_morph, sg.asRange(&vsMorphUniform(pbr_shd, morph_uniforms)));
        }

        const f = frameUniformsForState(env.shadow_uniforms, item.receive_shadows, ctx);
        const fs_params = pbr_shd.FsParams{
            .eye_pos = f.eye_pos,
            .light_dir = f.light_dir,
            .light_color = f.light_color,
            .ambient_color = f.ambient_color,
            .base_color_factor = rec.base_color,
            .pbr_factors = rec.pbr_factors,
            .emissive_factor = rec.emissive_color,
            .alpha_cutoff = rec.alpha_cutoff,
            .normal_scale = rec.normal_scale,
            .uv_matrix = rec.uv_matrices,
            .uv_offset = rec.uv_offsets,
            .channel_selectors = rec.channel_selectors,
            .shadow_params = f.shadow_params,
            .shadow_splits = f.shadow_splits,
            .cascade_view_proj = f.cascade_view_proj,
            .cascade_debug = f.cascade_debug,
            .light_counts = f.light_counts,
            .point_pos_range = f.point_pos_range,
            .point_color_int = f.point_color_int,
            .spot_pos_range = f.spot_pos_range,
            .spot_dir_inner = f.spot_dir_inner,
            .spot_color_outer = f.spot_color_outer,
            .spot_intensity = f.spot_intensity,
            .spot_view_proj = f.spot_view_proj,
            .spot_shadow_params = f.spot_shadow_params,
        };
        if (skel_bones != null) {
            sg.applyUniforms(skinned_pbr_shd.UB_fs_params, sg.asRange(&fs_params));
        } else {
            sg.applyUniforms(pbr_shd.UB_fs_params, sg.asRange(&fs_params));
        }
    } else {
        bind.views[shd.VIEW_diffuse_tex] = rec.albedo_view;
        bind.samplers[shd.SMP_smp] = rec.albedo_sampler;

        bind.views[shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
        bind.views[shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
        bind.views[shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
        bind.samplers[shd.SMP_shadow_smp] = env.shadow_pass.sampler;
        bind.samplers[shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;

        bind.views[shd.VIEW_morph_tex] = morph_view;
        bind.samplers[shd.SMP_morph_smp] = env.pipelines.morph_sampler;

        sg.applyBindings(bind);

        const vs_params = shd.VsParams{
            .mvp = mvp,
            .model = model,
        };
        sg.applyUniforms(shd.UB_vs_params, sg.asRange(&vs_params));
        sg.applyUniforms(shd.UB_vs_morph, sg.asRange(&vsMorphUniform(shd, morph_uniforms)));

        const f = frameUniformsForState(env.shadow_uniforms, item.receive_shadows, ctx);
        const fs_params = shd.FsParams{
            .eye_pos = f.eye_pos,
            .light_dir = f.light_dir,
            .light_color = f.light_color,
            .ambient_color = f.ambient_color,
            .diffuse_color = rec.base_color,
            .alpha_cutoff = rec.alpha_cutoff,
            .uv_matrix = rec.standard_uv_matrix,
            .uv_offset = rec.standard_uv_offset,
            .shadow_params = f.shadow_params,
            .shadow_splits = f.shadow_splits,
            .cascade_view_proj = f.cascade_view_proj,
            .cascade_debug = f.cascade_debug,
            .light_counts = f.light_counts,
            .point_pos_range = f.point_pos_range,
            .point_color_int = f.point_color_int,
            .spot_pos_range = f.spot_pos_range,
            .spot_dir_inner = f.spot_dir_inner,
            .spot_color_outer = f.spot_color_outer,
            .spot_intensity = f.spot_intensity,
            .spot_view_proj = f.spot_view_proj,
            .spot_shadow_params = f.spot_shadow_params,
        };
        sg.applyUniforms(shd.UB_fs_params, sg.asRange(&fs_params));
    }

    sg.draw(item.base_vertex, item.index_count, 1);
    env.stats.main_draw_calls += 1;
    env.stats.draw_calls += 1;
    env.stats.triangles += item.index_count / 3;
}

// Draws one regular queue item whose material is a .shader_material.
//
// The pipeline comes from the lazy ShaderMaterialCache (keyed by the
// registration key); bindings and uniforms reuse the engine module structs
// (shd.* / pbr_shd.*) because hook materials are compiled from the unmodified
// engine templates — the uniform/texture contract is layout-identical, only
// the UB slot handles are read from the registration entry.
//
// Documented limitations: skinned meshes skip drawing (hook materials are
// compiled from the non-skinned templates); runtime-registered sources with
// engine_template == false get only vertex/index binds, the material texture
// at view slot 0 and (optionally) the user uniform block.
fn drawShaderMaterialItem(
    env: *const Environment,
    item: RenderMeshItem,
    ctx: *const FrameContext,
    current_pipeline_id: *u32,
    snap: material_mod.ShaderDrawSnapshot,
    mvp: Mat4,
) void {
    if (item.is_skinned) return; // see limitation note above

    const entry = shader_material.entry(snap.entry_index) orelse return;
    const set = env.pipelines.shader_materials.getOrCreate(entry.key) orelse return;

    const is_u32 = item.is_u32;
    // P4: sidedness hook-материала — из снимка (sm.double_sided), без
    // decal-форсинга item.double_sided: поведение как до P4.
    const pip_id = set.pipelineFor(item.transparent, is_u32, snap.double_sided);
    if (pip_id == 0) return;
    if (pip_id != current_pipeline_id.*) {
        sg.applyPipeline(.{ .id = pip_id });
        current_pipeline_id.* = pip_id;
        env.stats.pipeline_switches += 1;
    }

    var bind = sg.Bindings{};
    bind.vertex_buffers[0] = item.vertex_buffer;
    bind.index_buffer = item.index_buffer;

    const f = frameUniformsForState(env.shadow_uniforms, item.receive_shadows, ctx);
    const alpha_cutoff = item.draw_record.alpha_cutoff;

    if (entry.engine_template) {
        const morph_view = item.morph_view;
        const morph_uniforms = item.morph_uniforms;
        if (entry.base == .pbr) {
            // PBR-base hook material: full PBR lighting with engine defaults
            // for the maps the material does not override. Текстура — из
            // prepare-снимка (дефолт уже подставлен при построении).
            bind.views[pbr_shd.VIEW_albedo_tex] = snap.tex_view;
            bind.views[pbr_shd.VIEW_normal_tex] = env.default_normal.view;
            bind.views[pbr_shd.VIEW_metallic_roughness_tex] = env.default_white.view;
            bind.views[pbr_shd.VIEW_emissive_tex] = env.default_white.view;
            bind.views[pbr_shd.VIEW_occlusion_tex] = env.default_white.view;
            bind.samplers[pbr_shd.SMP_smp] = snap.tex_sampler;
            // Hook materials have no per-slot data textures; the flat normal
            // default's sampler keeps the data_smp contract satisfied.
            bind.samplers[pbr_shd.SMP_data_smp] = env.default_normal.sampler;
            const cube = env.sky_texture orelse env.default_cube;
            bind.views[pbr_shd.VIEW_env_tex] = cube.view;
            bind.samplers[pbr_shd.SMP_env_smp] = cube.sampler;
            bind.views[pbr_shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
            bind.views[pbr_shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
            bind.views[pbr_shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
            bind.samplers[pbr_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
            bind.samplers[pbr_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;
            bind.views[pbr_shd.VIEW_morph_tex] = morph_view;
            bind.samplers[pbr_shd.SMP_morph_smp] = env.pipelines.morph_sampler;
            sg.applyBindings(bind);

            const vs_params = pbr_shd.VsParams{ .mvp = mvp, .model = item.model };
            sg.applyUniforms(entry.vs_ub, sg.asRange(&vs_params));
            sg.applyUniforms(pbr_shd.UB_vs_morph, sg.asRange(&vsMorphUniform(pbr_shd, morph_uniforms)));

            const fs_params = pbr_shd.FsParams{
                .eye_pos = f.eye_pos,
                .light_dir = f.light_dir,
                .light_color = f.light_color,
                .ambient_color = f.ambient_color,
                .base_color_factor = snap.tint,
                .pbr_factors = .{ 0.0, 0.5, 1.0, env.ibl_intensity },
                .emissive_factor = .{ 0, 0, 0, 1 },
                .alpha_cutoff = alpha_cutoff,
                .normal_scale = 1.0,
                // Hook materials carry no per-slot maps: identity UVs and
                // the glTF channel conventions.
                .uv_matrix = identityUvMatrices(),
                .uv_offset = identityUvOffsets(),
                .channel_selectors = pbrChannelSelectors(null),
                .shadow_params = f.shadow_params,
                .shadow_splits = f.shadow_splits,
                .cascade_view_proj = f.cascade_view_proj,
                .cascade_debug = f.cascade_debug,
                .light_counts = f.light_counts,
                .point_pos_range = f.point_pos_range,
                .point_color_int = f.point_color_int,
                .spot_pos_range = f.spot_pos_range,
                .spot_dir_inner = f.spot_dir_inner,
                .spot_color_outer = f.spot_color_outer,
                .spot_intensity = f.spot_intensity,
                .spot_view_proj = f.spot_view_proj,
                .spot_shadow_params = f.spot_shadow_params,
            };
            sg.applyUniforms(entry.fs_ub, sg.asRange(&fs_params));
        } else {
            // Standard-base hook material (текстура из снимка).
            bind.views[shd.VIEW_diffuse_tex] = snap.tex_view;
            bind.samplers[shd.SMP_smp] = snap.tex_sampler;
            bind.views[shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
            bind.views[shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
            bind.views[shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
            bind.samplers[shd.SMP_shadow_smp] = env.shadow_pass.sampler;
            bind.samplers[shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;
            bind.views[shd.VIEW_morph_tex] = morph_view;
            bind.samplers[shd.SMP_morph_smp] = env.pipelines.morph_sampler;
            sg.applyBindings(bind);

            const vs_params = shd.VsParams{ .mvp = mvp, .model = item.model };
            sg.applyUniforms(entry.vs_ub, sg.asRange(&vs_params));
            sg.applyUniforms(shd.UB_vs_morph, sg.asRange(&vsMorphUniform(shd, morph_uniforms)));

            const fs_params = shd.FsParams{
                .eye_pos = f.eye_pos,
                .light_dir = f.light_dir,
                .light_color = f.light_color,
                .ambient_color = f.ambient_color,
                .diffuse_color = snap.tint,
                .alpha_cutoff = alpha_cutoff,
                // Hook materials have no UV transform: identity.
                .uv_matrix = material_mod.UvTransform.identity.matrixRows(),
                .uv_offset = material_mod.UvTransform.identity.offsetPacked(),
                .shadow_params = f.shadow_params,
                .shadow_splits = f.shadow_splits,
                .cascade_view_proj = f.cascade_view_proj,
                .cascade_debug = f.cascade_debug,
                .light_counts = f.light_counts,
                .point_pos_range = f.point_pos_range,
                .point_color_int = f.point_color_int,
                .spot_pos_range = f.spot_pos_range,
                .spot_dir_inner = f.spot_dir_inner,
                .spot_color_outer = f.spot_color_outer,
                .spot_intensity = f.spot_intensity,
                .spot_view_proj = f.spot_view_proj,
                .spot_shadow_params = f.spot_shadow_params,
            };
            sg.applyUniforms(entry.fs_ub, sg.asRange(&fs_params));
        }
    } else {
        // Runtime-registered custom source: vertex/index binds plus the
        // material texture at view slot 0 (sokol tolerates binding slots the
        // shader does not declare; shaders that declare nothing get white).
        // Contract: UB 0 carries {mat4 mvp, mat4 model} like every forward
        // shader (runtime sources must declare it, see registerRuntime docs).
        // Текстура view slot 0 — из снимка.
        bind.views[0] = snap.tex_view;
        bind.samplers[0] = snap.tex_sampler;
        sg.applyBindings(bind);

        const vs_params = shd.VsParams{ .mvp = mvp, .model = item.model };
        sg.applyUniforms(entry.vs_ub, sg.asRange(&vs_params));
    }

    // User uniform block (declarative param table -> packed 128 bytes);
    // materials with vertex-stage params carry the same payload on the
    // fs-stage and vs-stage blocks.
    if (entry.user_ub) |ub| {
        sg.applyUniforms(ub, sg.asRange(&snap.uniforms));
    }
    if (entry.vs_user_ub) |ub| {
        sg.applyUniforms(ub, sg.asRange(&snap.uniforms));
    }

    sg.draw(item.base_vertex, item.index_count, 1);
    env.stats.main_draw_calls += 1;
    env.stats.draw_calls += 1;
    env.stats.triangles += item.index_count / 3;
}

threadlocal var fallback_uniforms: uniforms.FrameUniforms = undefined;

fn frameUniformsForState(shadow_uniforms: uniforms.ShadowState, mesh_receive_shadows: bool, ctx: *const FrameContext) *const uniforms.FrameUniforms {
    if (mesh_receive_shadows) {
        if (ctx.uniforms_with_shadows) |u| return u;
    } else {
        if (ctx.uniforms_without_shadows) |u| return u;
    }
    var state = shadow_uniforms;
    state.mesh_receive_shadows = mesh_receive_shadows;
    fallback_uniforms = uniforms.buildFrameUniforms(state, ctx);
    return &fallback_uniforms;
}

// Builds the module-specific VsMorph struct (identical layout in all three
// forward shader modules) from the packed per-draw values.
fn vsMorphUniform(comptime module: anytype, u: morph_gpu.VsUniforms) module.VsMorph {
    return .{
        .morph_weights0 = u.weights0,
        .morph_weights1 = u.weights1,
        .morph_params = u.params,
    };
}

// ---------------------------------------------------------------------------
// Per-slot UV transforms (KHR_texture_transform) and channel selection
// packing (wave/ktx2). Identity/default inputs produce no-op uniforms, so
// materials without these features upload exactly the same values the
// shaders' defaults would sample with.
// ---------------------------------------------------------------------------

const uv_identity_matrix: [4]f32 = .{ 1, 0, 0, 1 };
const uv_identity_offset: [4]f32 = .{ 0, 0, 0, 0 };

fn identityUvMatrices() [5][4]f32 {
    return @splat(uv_identity_matrix);
}

fn identityUvOffsets() [5][4]f32 {
    return @splat(uv_identity_offset);
}

/// PBR slot order (must match the pbr.glsl texture bindings):
/// 0 albedo, 1 normal, 2 metallic-roughness, 3 emissive, 4 occlusion.
/// Null material = all identity.
fn pbrUvMatrices(pbr_mat: ?*const PBRMaterial) [5][4]f32 {
    var rows: [5][4]f32 = identityUvMatrices();
    if (pbr_mat) |p| {
        rows[0] = p.albedo_uv_transform.matrixRows();
        rows[1] = p.normal_uv_transform.matrixRows();
        rows[2] = p.metallic_roughness_uv_transform.matrixRows();
        rows[3] = p.emissive_uv_transform.matrixRows();
        rows[4] = p.occlusion_uv_transform.matrixRows();
    }
    return rows;
}

fn pbrUvOffsets(pbr_mat: ?*const PBRMaterial) [5][4]f32 {
    var offs: [5][4]f32 = identityUvOffsets();
    if (pbr_mat) |p| {
        offs[0] = p.albedo_uv_transform.offsetPacked();
        offs[1] = p.normal_uv_transform.offsetPacked();
        offs[2] = p.metallic_roughness_uv_transform.offsetPacked();
        offs[3] = p.emissive_uv_transform.offsetPacked();
        offs[4] = p.occlusion_uv_transform.offsetPacked();
    }
    return offs;
}

/// Lane indices for occlusion/roughness/metallic; null material = glTF
/// conventions (R, G, B).
fn pbrChannelSelectors(pbr_mat: ?*const PBRMaterial) [4]f32 {
    if (pbr_mat) |p| {
        return .{
            p.occlusion_channel.selector(),
            p.roughness_channel.selector(),
            p.metallic_channel.selector(),
            0,
        };
    }
    return .{ 0, 1, 2, 0 };
}

// Pipeline flags for one instanced group. Mirrors the regular-decal rule
// (see cullNonInstancedMesh): is_decal forces the transparent blend
// pipeline and the cull-off twin, so an instanced decal with an opaque
// material draws exactly like a regular decal. Pure (no GPU calls).
pub fn instancedDrawFlags(material: ?Material, is_decal: bool) struct { transparent: bool, double_sided: bool } {
    return .{
        .transparent = render_queue.materialIsTransparent(material) or is_decal,
        .double_sided = render_queue.materialIsDoubleSided(material) or is_decal,
    };
}

// Draws one instanced mesh with the currently visible instance buffer.
// Transparent instanced meshes use the blend twin pipeline and are drawn
// as-is (no per-instance back-to-front sort); the caller draws them
// after all opaque geometry. Double-sided materials select the cull-off
// twins (opaque or blend) when the pipeline set provides them. Updates stats.
// NOTE: GPU morphs and instancing do not combine in this version: the
// instanced shader families have no vs_morph block and this path never
// binds morph resources, so a morph mesh placed in an instanced queue
// renders its base pose. CPU mode (the default) is unaffected: instanced
// meshes share the already-blended dynamic vertex buffer.
pub fn drawInstancedBatch(env: *const Environment, batch: RenderInstancedBatch, ctx: *const FrameContext, current_pipeline_id: *u32) void {
    if (batch.visible_instance_count == 0 or batch.instance_buffer.id == 0) return;

    const is_pbr = batch.is_pbr;
    const is_u32 = batch.index_type == .UINT32;
    // Double-sided instanced meshes use the cull-off twins when the set
    // provides them; otherwise the regular pipelines (legacy behavior).
    const pip_id = env.pipelines.forInstancedMesh(is_pbr, batch.transparent, is_u32, batch.double_sided);
    if (pip_id == 0) return;
    if (pip_id != current_pipeline_id.*) {
        sg.applyPipeline(.{ .id = pip_id });
        current_pipeline_id.* = pip_id;
        env.stats.pipeline_switches += 1;
    }

    var bind = sg.Bindings{};
    bind.vertex_buffers[0] = batch.vertex_buffer;
    bind.vertex_buffers[1] = batch.instance_buffer;
    bind.index_buffer = batch.index_buffer;

    const rec = batch.draw_record;
    if (is_pbr) {
        bind.views[inst_pbr_shd.VIEW_albedo_tex] = rec.albedo_view;
        bind.views[inst_pbr_shd.VIEW_normal_tex] = rec.normal_view;
        bind.views[inst_pbr_shd.VIEW_metallic_roughness_tex] = rec.mr_view;
        bind.views[inst_pbr_shd.VIEW_emissive_tex] = rec.emissive_view;
        bind.views[inst_pbr_shd.VIEW_occlusion_tex] = rec.occlusion_view;
        bind.samplers[inst_pbr_shd.SMP_smp] = rec.albedo_sampler;
        bind.samplers[inst_pbr_shd.SMP_data_smp] = rec.data_sampler;

        // Environment IBL Cubemap & Shadow Depth Map
        const cube_view = rec.env_view orelse (if (env.sky_texture) |s| s.view else env.default_cube.view);
        const cube_sampler = rec.env_sampler orelse (if (env.sky_texture) |s| s.sampler else env.default_cube.sampler);
        bind.views[inst_pbr_shd.VIEW_env_tex] = cube_view;
        bind.samplers[inst_pbr_shd.SMP_env_smp] = cube_sampler;

        bind.views[inst_pbr_shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
        bind.views[inst_pbr_shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
        bind.views[inst_pbr_shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
        bind.samplers[inst_pbr_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
        bind.samplers[inst_pbr_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;

        sg.applyBindings(bind);

        const inst_vs = inst_pbr_shd.VsParams{
            .view_proj = ctx.view_proj,
        };
        sg.applyUniforms(inst_pbr_shd.UB_vs_params, sg.asRange(&inst_vs));

        const f = frameUniformsForState(env.shadow_uniforms, batch.receive_shadows, ctx);
        const inst_fs = inst_pbr_shd.FsParams{
            .eye_pos = f.eye_pos,
            .light_dir = f.light_dir,
            .light_color = f.light_color,
            .ambient_color = f.ambient_color,
            .base_color_factor = rec.base_color,
            .pbr_factors = rec.pbr_factors,
            .emissive_factor = rec.emissive_color,
            .shadow_params = f.shadow_params,
            .shadow_splits = f.shadow_splits,
            .cascade_view_proj = f.cascade_view_proj,
            .cascade_debug = f.cascade_debug,
            .light_counts = f.light_counts,
            .point_pos_range = f.point_pos_range,
            .point_color_int = f.point_color_int,
            .spot_pos_range = f.spot_pos_range,
            .spot_dir_inner = f.spot_dir_inner,
            .spot_color_outer = f.spot_color_outer,
            .spot_intensity = f.spot_intensity,
            .spot_view_proj = f.spot_view_proj,
            .spot_shadow_params = f.spot_shadow_params,
            .alpha_cutoff = rec.alpha_cutoff,
            .normal_scale = rec.normal_scale,
            .uv_matrix = rec.uv_matrices,
            .uv_offset = rec.uv_offsets,
            .channel_selectors = rec.channel_selectors,
        };
        sg.applyUniforms(inst_pbr_shd.UB_fs_params, sg.asRange(&inst_fs));
    } else {
        bind.views[inst_shd.VIEW_diffuse_tex] = rec.albedo_view;
        bind.samplers[inst_shd.SMP_smp] = rec.albedo_sampler;

        bind.views[inst_shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
        bind.views[inst_shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
        bind.views[inst_shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
        bind.samplers[inst_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
        bind.samplers[inst_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;

        sg.applyBindings(bind);

        const inst_vs = inst_shd.VsParams{
            .view_proj = ctx.view_proj,
        };
        sg.applyUniforms(inst_shd.UB_vs_params, sg.asRange(&inst_vs));

        const f = frameUniformsForState(env.shadow_uniforms, batch.receive_shadows, ctx);
        const inst_fs = inst_shd.FsParams{
            .eye_pos = f.eye_pos,
            .light_dir = f.light_dir,
            .light_color = f.light_color,
            .ambient_color = f.ambient_color,
            .diffuse_color = rec.base_color,
            .alpha_cutoff = rec.alpha_cutoff,
            .uv_matrix = rec.standard_uv_matrix,
            .uv_offset = rec.standard_uv_offset,
            .shadow_params = f.shadow_params,
            .shadow_splits = f.shadow_splits,
            .cascade_view_proj = f.cascade_view_proj,
            .cascade_debug = f.cascade_debug,
            .light_counts = f.light_counts,
            .point_pos_range = f.point_pos_range,
            .point_color_int = f.point_color_int,
            .spot_pos_range = f.spot_pos_range,
            .spot_dir_inner = f.spot_dir_inner,
            .spot_color_outer = f.spot_color_outer,
            .spot_intensity = f.spot_intensity,
            .spot_view_proj = f.spot_view_proj,
            .spot_shadow_params = f.spot_shadow_params,
        };
        sg.applyUniforms(inst_shd.UB_fs_params, sg.asRange(&inst_fs));
    }

    sg.draw(0, batch.index_count, batch.visible_instance_count);
    env.stats.main_draw_calls += 1;
    env.stats.draw_calls += 1;
    env.stats.triangles += (batch.index_count / 3) * batch.visible_instance_count;
}

// ---------------------------------------------------------------------------
// Tests: the shader-facing contract of the wave/ktx2 uniforms. The generated
// modules are build artifacts of sokol-shdc, so these comptime checks fail
// at test time (GPU-free) if the .glsl templates and the packing helpers
// ever drift apart.
// ---------------------------------------------------------------------------

test "forward shader FsParams carry the appended uv/channel uniforms" {
    // PBR family: 5 slots + channel selectors, appended after normal_scale.
    comptime {
        for ([_]type{ pbr_shd.FsParams, skinned_pbr_shd.FsParams, inst_pbr_shd.FsParams }) |P| {
            if (!@hasField(P, "uv_matrix")) @compileError("FsParams missing uv_matrix");
            if (!@hasField(P, "uv_offset")) @compileError("FsParams missing uv_offset");
            if (!@hasField(P, "channel_selectors")) @compileError("FsParams missing channel_selectors");
        }
    }
    // Standard family: one diffuse slot.
    comptime {
        for ([_]type{ shd.FsParams, inst_shd.FsParams }) |P| {
            if (!@hasField(P, "uv_matrix")) @compileError("FsParams missing uv_matrix");
            if (!@hasField(P, "uv_offset")) @compileError("FsParams missing uv_offset");
        }
    }
}

test "pbr uniform packing defaults are identity and glTF conventions" {
    const mats = pbrUvMatrices(null);
    const offs = pbrUvOffsets(null);
    for (0..5) |slot| {
        try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, &mats[slot]);
        try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, &offs[slot]);
    }
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 0 }, &pbrChannelSelectors(null));

    var mat = PBRMaterial.init("m");
    mat.albedo_uv_transform = .{ .offset = .{ 0.5, 0 }, .scale = .{ 2, 2 } };
    mat.occlusion_channel = .a;
    const packed_mats = pbrUvMatrices(&mat);
    const packed_offs = pbrUvOffsets(&mat);
    try std.testing.expectEqualSlices(f32, &.{ 2, 0, 0, 2 }, &packed_mats[0]);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0, 0, 0 }, &packed_offs[0]);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, &packed_mats[1]); // normal untouched
    try std.testing.expectEqualSlices(f32, &.{ 3, 1, 2, 0 }, &pbrChannelSelectors(&mat));
}

test "instanced decal forces transparent + double-sided like regular decals" {
    var opaque_mat = StandardMaterial.init("opaque");
    const solid_mat: Material = .{ .standard = &opaque_mat };
    // No material, no decal: opaque single-sided (legacy behavior).
    const plain = instancedDrawFlags(null, false);
    try std.testing.expect(!plain.transparent and !plain.double_sided);
    // Opaque material, no decal: stays opaque single-sided.
    const solid = instancedDrawFlags(solid_mat, false);
    try std.testing.expect(!solid.transparent and !solid.double_sided);
    // The regression: opaque material + is_decal must draw as transparent
    // double-sided, matching cullNonInstancedMesh for regular decals.
    const decal = instancedDrawFlags(solid_mat, true);
    try std.testing.expect(decal.transparent and decal.double_sided);
    // Blend material is transparent without the decal flag; double-sided
    // still follows the material alone here.
    opaque_mat.alpha_mode = .blend;
    const blended = instancedDrawFlags(solid_mat, false);
    try std.testing.expect(blended.transparent and !blended.double_sided);
}
