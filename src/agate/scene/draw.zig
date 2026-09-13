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
const StandardMaterial = @import("../material.zig").StandardMaterial;
const passes = @import("../passes/mod.zig");

const render_queue = @import("render_queue.zig");
const RenderMeshItem = render_queue.RenderMeshItem;
const Mesh = @import("../mesh.zig").Mesh;
const morph_gpu = @import("../mesh/morph_gpu.zig");
const uniforms = @import("uniforms.zig");
const forward_pipelines = @import("forward_pipelines.zig");
const ForwardPipelines = forward_pipelines.ForwardPipelines;
const shader_material = @import("../shader_material.zig");
const ShaderMaterial = @import("../material.zig").ShaderMaterial;
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

pub const FrameContext = uniforms.FrameContext;

/// Per-frame render state the draw helpers need, passed explicitly (this
/// module must not import scene.zig). Scene fills it once per frame; the
/// values are read-only snapshots except stats and the per-mesh
/// receive_shadows patch on shadow_uniforms.
pub const Environment = struct {
    // Mutable: shader-material pipelines are created lazily on first use
    // (ShaderMaterialCache.getOrCreate); everything else is read-only.
    pipelines: *ForwardPipelines,
    stats: *SceneStats,
    // Fallbacks for meshes/materials without their own textures.
    default_material: *const StandardMaterial,
    default_white: *const Texture,
    default_normal: *const Texture,
    default_cube: *const CubeTexture,
    // Environment IBL cubemap: a material-level probe overrides the skybox,
    // which in turn falls back to default_cube.
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
pub fn drawRegularItem(env: Environment, item: RenderMeshItem, ctx: *const FrameContext, current_pipeline_id: *u32) void {
    const mesh = item.mesh;
    const model = item.model;
    const mvp = Mat4.mul(ctx.view_proj, model);

    // Shader materials take their own path: pipeline from the lazy
    // ShaderMaterialCache, bindings/uniforms per the registration's base
    // template contract (identical layouts — see drawShaderMaterialItem).
    if (mesh.material) |mat| {
        if (mat == .shader_material) {
            return drawShaderMaterialItem(env, item, ctx, current_pipeline_id, mat.shader_material, mvp);
        }
    }

    const pip_id = env.pipelines.forRegularItem(item);

    if (pip_id == 0) return;
    if (pip_id != current_pipeline_id.*) {
        sg.applyPipeline(.{ .id = pip_id });
        current_pipeline_id.* = pip_id;
        env.stats.pipeline_switches += 1;
    }

    var bind = sg.Bindings{};
    bind.vertex_buffers[0] = mesh.vertex_buffer;
    bind.index_buffer = mesh.index_buffer;

    const morph_bind = morphBindFor(env, mesh);

    if (item.is_pbr) {
        const pbr_mat = if (mesh.material) |m| m.pbr else null;
        const albedo_tex = if (pbr_mat) |p| (p.albedo_texture orelse env.default_white.*) else env.default_white.*;
        const normal_tex = if (pbr_mat) |p| (p.normal_texture orelse env.default_normal.*) else env.default_normal.*;
        const mr_tex = if (pbr_mat) |p| (p.metallic_roughness_texture orelse env.default_white.*) else env.default_white.*;
        const emissive_tex = if (pbr_mat) |p| (p.emissive_texture orelse env.default_white.*) else env.default_white.*;
        const occlusion_tex = if (pbr_mat) |p| (p.occlusion_texture orelse env.default_white.*) else env.default_white.*;

        bind.views[pbr_shd.VIEW_albedo_tex] = albedo_tex.view;
        bind.views[pbr_shd.VIEW_normal_tex] = normal_tex.view;
        bind.views[pbr_shd.VIEW_metallic_roughness_tex] = mr_tex.view;
        bind.views[pbr_shd.VIEW_emissive_tex] = emissive_tex.view;
        bind.views[pbr_shd.VIEW_occlusion_tex] = occlusion_tex.view;
        bind.samplers[pbr_shd.SMP_smp] = albedo_tex.sampler;
        // Data slots sample with their own sampler class: the first present
        // data texture wins (normal maps care most about wrap/filter);
        // falls back to the albedo sampler when only a color map exists.
        const data_sampler_tex = if (pbr_mat) |p|
            (p.normal_texture orelse p.metallic_roughness_texture orelse p.occlusion_texture orelse p.emissive_texture orelse albedo_tex)
        else
            albedo_tex;
        bind.samplers[pbr_shd.SMP_data_smp] = data_sampler_tex.sampler;

        // Environment IBL Cubemap & Shadow Depth Map: a material-level
        // environment texture (e.g. an HDR probe) overrides the skybox,
        // which in turn falls back to default_cube.
        const cube = (if (pbr_mat) |p| p.environment_texture else null) orelse env.sky_texture orelse env.default_cube.*;
        bind.views[pbr_shd.VIEW_env_tex] = cube.view;
        bind.samplers[pbr_shd.SMP_env_smp] = cube.sampler;

        bind.views[pbr_shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
        bind.views[pbr_shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
        bind.views[pbr_shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
        bind.samplers[pbr_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
        bind.samplers[pbr_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;

        bind.views[pbr_shd.VIEW_morph_tex] = morph_bind.view;
        bind.samplers[pbr_shd.SMP_morph_smp] = env.pipelines.morph_sampler;

        sg.applyBindings(bind);

        const vs_params = pbr_shd.VsParams{
            .mvp = mvp,
            .model = model,
        };
        sg.applyUniforms(pbr_shd.UB_vs_params, sg.asRange(&vs_params));

        if (mesh.skeleton) |skel| {
            const vs_skin = skinned_pbr_shd.VsSkin{
                .bones = skel.skin_matrices,
            };
            sg.applyUniforms(skinned_pbr_shd.UB_vs_skin, sg.asRange(&vs_skin));
            sg.applyUniforms(skinned_pbr_shd.UB_vs_morph, sg.asRange(&vsMorphUniform(skinned_pbr_shd, morph_bind.uniforms)));
        } else {
            sg.applyUniforms(pbr_shd.UB_vs_morph, sg.asRange(&vsMorphUniform(pbr_shd, morph_bind.uniforms)));
        }

        const mat_albedo = if (pbr_mat) |p| p.getAlbedoColor4() else [4]f32{ 1, 1, 1, 1 };
        const metallic = if (pbr_mat) |p| p.metallic else 0.0;
        const roughness = if (pbr_mat) |p| p.roughness else 0.5;
        const env_intensity = if (pbr_mat) |p| env.ibl_intensity * p.environment_intensity else env.ibl_intensity;
        const emissive_col = if (pbr_mat) |p| [4]f32{ p.emissive_color.r, p.emissive_color.g, p.emissive_color.b, 1.0 } else [4]f32{ 0, 0, 0, 1 };
        const occlusion_strength = if (pbr_mat) |p| p.occlusion_strength else 1.0;

        const f = frameUniformsFor(env, mesh, ctx);
        const fs_params = pbr_shd.FsParams{
            .eye_pos = f.eye_pos,
            .light_dir = f.light_dir,
            .light_color = f.light_color,
            .ambient_color = f.ambient_color,
            .base_color_factor = mat_albedo,
            .pbr_factors = .{ metallic, roughness, occlusion_strength, env_intensity },
            .emissive_factor = emissive_col,
            .alpha_cutoff = uniforms.alphaCutoffFor(mesh.material),
            .normal_scale = if (pbr_mat) |p| p.normal_scale else 1.0,
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
        if (mesh.skeleton != null) {
            sg.applyUniforms(skinned_pbr_shd.UB_fs_params, sg.asRange(&fs_params));
        } else {
            sg.applyUniforms(pbr_shd.UB_fs_params, sg.asRange(&fs_params));
        }
    } else {
        const std_mat = if (mesh.material) |m| m.standard else env.default_material;
        const tex = if (std_mat.diffuse_texture) |t| t else env.default_white.*;

        bind.views[shd.VIEW_diffuse_tex] = tex.view;
        bind.samplers[shd.SMP_smp] = tex.sampler;

        bind.views[shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
        bind.views[shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
        bind.views[shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
        bind.samplers[shd.SMP_shadow_smp] = env.shadow_pass.sampler;
        bind.samplers[shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;

        bind.views[shd.VIEW_morph_tex] = morph_bind.view;
        bind.samplers[shd.SMP_morph_smp] = env.pipelines.morph_sampler;

        sg.applyBindings(bind);

        const vs_params = shd.VsParams{
            .mvp = mvp,
            .model = model,
        };
        sg.applyUniforms(shd.UB_vs_params, sg.asRange(&vs_params));
        sg.applyUniforms(shd.UB_vs_morph, sg.asRange(&vsMorphUniform(shd, morph_bind.uniforms)));

        const f = frameUniformsFor(env, mesh, ctx);
        const fs_params = shd.FsParams{
            .eye_pos = f.eye_pos,
            .light_dir = f.light_dir,
            .light_color = f.light_color,
            .ambient_color = f.ambient_color,
            .diffuse_color = std_mat.getDiffuseColor4(),
            .alpha_cutoff = uniforms.alphaCutoffFor(mesh.material),
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

    sg.draw(0, mesh.index_count, 1);
    env.stats.main_draw_calls += 1;
    env.stats.draw_calls += 1;
    env.stats.triangles += mesh.index_count / 3;
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
    env: Environment,
    item: RenderMeshItem,
    ctx: *const FrameContext,
    current_pipeline_id: *u32,
    sm: *const ShaderMaterial,
    mvp: Mat4,
) void {
    const mesh = item.mesh;
    if (mesh.skeleton != null) return; // see limitation note above

    const entry = shader_material.entry(sm.entry_index) orelse return;
    const set = env.pipelines.shader_materials.getOrCreate(entry.key) orelse return;

    const is_u32 = mesh.index_type == .UINT32;
    const pip_id = set.pipelineFor(item.transparent, is_u32, sm.double_sided);
    if (pip_id == 0) return;
    if (pip_id != current_pipeline_id.*) {
        sg.applyPipeline(.{ .id = pip_id });
        current_pipeline_id.* = pip_id;
        env.stats.pipeline_switches += 1;
    }

    var bind = sg.Bindings{};
    bind.vertex_buffers[0] = mesh.vertex_buffer;
    bind.index_buffer = mesh.index_buffer;

    const f = frameUniformsFor(env, mesh, ctx);
    const alpha_cutoff = uniforms.alphaCutoffFor(mesh.material);

    if (entry.engine_template) {
        const morph_bind = morphBindFor(env, mesh);
        if (entry.base == .pbr) {
            // PBR-base hook material: full PBR lighting with engine defaults
            // for the maps the material does not override.
            const albedo_tex = sm.texture orelse env.default_white.*;
            bind.views[pbr_shd.VIEW_albedo_tex] = albedo_tex.view;
            bind.views[pbr_shd.VIEW_normal_tex] = env.default_normal.view;
            bind.views[pbr_shd.VIEW_metallic_roughness_tex] = env.default_white.view;
            bind.views[pbr_shd.VIEW_emissive_tex] = env.default_white.view;
            bind.views[pbr_shd.VIEW_occlusion_tex] = env.default_white.view;
            bind.samplers[pbr_shd.SMP_smp] = albedo_tex.sampler;
            // Hook materials have no per-slot data textures; the flat normal
            // default's sampler keeps the data_smp contract satisfied.
            bind.samplers[pbr_shd.SMP_data_smp] = env.default_normal.sampler;
            const cube = env.sky_texture orelse env.default_cube.*;
            bind.views[pbr_shd.VIEW_env_tex] = cube.view;
            bind.samplers[pbr_shd.SMP_env_smp] = cube.sampler;
            bind.views[pbr_shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
            bind.views[pbr_shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
            bind.views[pbr_shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
            bind.samplers[pbr_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
            bind.samplers[pbr_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;
            bind.views[pbr_shd.VIEW_morph_tex] = morph_bind.view;
            bind.samplers[pbr_shd.SMP_morph_smp] = env.pipelines.morph_sampler;
            sg.applyBindings(bind);

            const vs_params = pbr_shd.VsParams{ .mvp = mvp, .model = item.model };
            sg.applyUniforms(entry.vs_ub, sg.asRange(&vs_params));
            sg.applyUniforms(pbr_shd.UB_vs_morph, sg.asRange(&vsMorphUniform(pbr_shd, morph_bind.uniforms)));

            const fs_params = pbr_shd.FsParams{
                .eye_pos = f.eye_pos,
                .light_dir = f.light_dir,
                .light_color = f.light_color,
                .ambient_color = f.ambient_color,
                .base_color_factor = sm.getTintColor4(),
                .pbr_factors = .{ 0.0, 0.5, 1.0, env.ibl_intensity },
                .emissive_factor = .{ 0, 0, 0, 1 },
                .alpha_cutoff = alpha_cutoff,
                .normal_scale = 1.0,
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
            // Standard-base hook material.
            const tex = sm.texture orelse env.default_white.*;
            bind.views[shd.VIEW_diffuse_tex] = tex.view;
            bind.samplers[shd.SMP_smp] = tex.sampler;
            bind.views[shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
            bind.views[shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
            bind.views[shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
            bind.samplers[shd.SMP_shadow_smp] = env.shadow_pass.sampler;
            bind.samplers[shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;
            bind.views[shd.VIEW_morph_tex] = morph_bind.view;
            bind.samplers[shd.SMP_morph_smp] = env.pipelines.morph_sampler;
            sg.applyBindings(bind);

            const vs_params = shd.VsParams{ .mvp = mvp, .model = item.model };
            sg.applyUniforms(entry.vs_ub, sg.asRange(&vs_params));
            sg.applyUniforms(shd.UB_vs_morph, sg.asRange(&vsMorphUniform(shd, morph_bind.uniforms)));

            const fs_params = shd.FsParams{
                .eye_pos = f.eye_pos,
                .light_dir = f.light_dir,
                .light_color = f.light_color,
                .ambient_color = f.ambient_color,
                .diffuse_color = sm.getTintColor4(),
                .alpha_cutoff = alpha_cutoff,
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
        const tex = sm.texture orelse env.default_white.*;
        bind.views[0] = tex.view;
        bind.samplers[0] = tex.sampler;
        sg.applyBindings(bind);

        const vs_params = shd.VsParams{ .mvp = mvp, .model = item.model };
        sg.applyUniforms(entry.vs_ub, sg.asRange(&vs_params));
    }

    // User uniform block (declarative param table -> packed 128 bytes);
    // materials with vertex-stage params carry the same payload on the
    // fs-stage and vs-stage blocks.
    if (entry.user_ub) |ub| {
        sg.applyUniforms(ub, sg.asRange(&sm.uniforms));
    }
    if (entry.vs_user_ub) |ub| {
        sg.applyUniforms(ub, sg.asRange(&sm.uniforms));
    }

    sg.draw(0, mesh.index_count, 1);
    env.stats.main_draw_calls += 1;
    env.stats.draw_calls += 1;
    env.stats.triangles += mesh.index_count / 3;
}

// Packs the shared fragment uniforms for one mesh: the scene-level state is
// copied and the per-mesh receive_shadows flag patched in.
fn frameUniformsFor(env: Environment, mesh: *const @import("../mesh.zig").Mesh, ctx: *const FrameContext) uniforms.FrameUniforms {
    var state = env.shadow_uniforms;
    state.mesh_receive_shadows = mesh.receive_shadows;
    return uniforms.buildFrameUniforms(state, ctx);
}

// GPU-morph bind resources for one regular draw. Every draw must bind the
// morph view and apply the vs_morph uniforms: sokol binding/uniform state
// persists across draws, so a skipped draw would inherit the previous
// mesh's delta texture and weights. CPU-morph and non-morph meshes bind
// the pipeline-owned 1x1 zero texture with the enable flag off, keeping
// the shader's fetch loop a no-op.
fn morphBindFor(env: Environment, mesh: *const Mesh) struct { view: sg.View, uniforms: morph_gpu.VsUniforms } {
    const gpu = mesh.morph_mode == .gpu and mesh.morph_delta_view.id != 0;
    return .{
        .view = if (gpu) mesh.morph_delta_view else env.pipelines.default_morph_view,
        .uniforms = morph_gpu.vsUniforms(mesh),
    };
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
pub fn drawInstancedMesh(env: Environment, mesh: *Mesh, ctx: *const FrameContext, current_pipeline_id: *u32) void {
    if (mesh.instances.items.len == 0) return;
    if (mesh.visible_instance_count == 0 or mesh.instance_buffer.id == 0) return;

    const is_pbr = if (mesh.material) |m| (m == .pbr) else false;
    const transparent = render_queue.materialIsTransparent(mesh.material);
    const is_u32 = mesh.index_type == .UINT32;
    // Double-sided instanced meshes use the cull-off twins when the set
    // provides them; otherwise the regular pipelines (legacy behavior).
    const pip_id = env.pipelines.forInstancedMesh(is_pbr, transparent, is_u32, render_queue.materialIsDoubleSided(mesh.material));
    if (pip_id == 0) return;
    if (pip_id != current_pipeline_id.*) {
        sg.applyPipeline(.{ .id = pip_id });
        current_pipeline_id.* = pip_id;
        env.stats.pipeline_switches += 1;
    }

    var bind = sg.Bindings{};
    bind.vertex_buffers[0] = mesh.vertex_buffer;
    bind.vertex_buffers[1] = mesh.instance_buffer;
    bind.index_buffer = mesh.index_buffer;

    if (is_pbr) {
        const pbr_mat = if (mesh.material) |m| m.pbr else null;
        const albedo_tex = if (pbr_mat) |p| (p.albedo_texture orelse env.default_white.*) else env.default_white.*;
        const normal_tex = if (pbr_mat) |p| (p.normal_texture orelse env.default_normal.*) else env.default_normal.*;
        const mr_tex = if (pbr_mat) |p| (p.metallic_roughness_texture orelse env.default_white.*) else env.default_white.*;
        const emissive_tex = if (pbr_mat) |p| (p.emissive_texture orelse env.default_white.*) else env.default_white.*;
        const occlusion_tex = if (pbr_mat) |p| (p.occlusion_texture orelse env.default_white.*) else env.default_white.*;

        bind.views[inst_pbr_shd.VIEW_albedo_tex] = albedo_tex.view;
        bind.views[inst_pbr_shd.VIEW_normal_tex] = normal_tex.view;
        bind.views[inst_pbr_shd.VIEW_metallic_roughness_tex] = mr_tex.view;
        bind.views[inst_pbr_shd.VIEW_emissive_tex] = emissive_tex.view;
        bind.views[inst_pbr_shd.VIEW_occlusion_tex] = occlusion_tex.view;
        bind.samplers[inst_pbr_shd.SMP_smp] = albedo_tex.sampler;
        // Data-slot sampler: first present data texture wins, mirroring the
        // regular draw path.
        const data_sampler_tex = if (pbr_mat) |p|
            (p.normal_texture orelse p.metallic_roughness_texture orelse p.occlusion_texture orelse p.emissive_texture orelse albedo_tex)
        else
            albedo_tex;
        bind.samplers[inst_pbr_shd.SMP_data_smp] = data_sampler_tex.sampler;

        // Environment IBL Cubemap & Shadow Depth Map
        const cube = (if (pbr_mat) |p| p.environment_texture else null) orelse env.sky_texture orelse env.default_cube.*;
        bind.views[inst_pbr_shd.VIEW_env_tex] = cube.view;
        bind.samplers[inst_pbr_shd.SMP_env_smp] = cube.sampler;

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

        const mat_albedo = if (pbr_mat) |p| p.getAlbedoColor4() else [4]f32{ 1, 1, 1, 1 };
        const metallic = if (pbr_mat) |p| p.metallic else 0.0;
        const roughness = if (pbr_mat) |p| p.roughness else 0.5;
        const env_intensity = if (pbr_mat) |p| env.ibl_intensity * p.environment_intensity else env.ibl_intensity;
        const emissive_col = if (pbr_mat) |p| [4]f32{ p.emissive_color.r, p.emissive_color.g, p.emissive_color.b, 1.0 } else [4]f32{ 0, 0, 0, 1 };
        const occlusion_strength = if (pbr_mat) |p| p.occlusion_strength else 1.0;

        const f = frameUniformsFor(env, mesh, ctx);
        const inst_fs = inst_pbr_shd.FsParams{
            .eye_pos = f.eye_pos,
            .light_dir = f.light_dir,
            .light_color = f.light_color,
            .ambient_color = f.ambient_color,
            .base_color_factor = mat_albedo,
            .pbr_factors = .{ metallic, roughness, occlusion_strength, env_intensity },
            .emissive_factor = emissive_col,
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
            .alpha_cutoff = uniforms.alphaCutoffFor(mesh.material),
            .normal_scale = if (pbr_mat) |p| p.normal_scale else 1.0,
        };
        sg.applyUniforms(inst_pbr_shd.UB_fs_params, sg.asRange(&inst_fs));
    } else {
        // Documented limitation: instanced shader-material meshes render with
        // the default standard material in this version (the hook materials
        // are compiled from the non-instanced templates).
        const std_mat = if (mesh.material) |m| switch (m) {
            .standard => |s| s,
            .pbr => env.default_material,
            .shader_material => env.default_material,
        } else env.default_material;
        const tex = if (std_mat.diffuse_texture) |t| t else env.default_white.*;

        bind.views[inst_shd.VIEW_diffuse_tex] = tex.view;
        bind.samplers[inst_shd.SMP_smp] = tex.sampler;

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

        const f = frameUniformsFor(env, mesh, ctx);
        const inst_fs = inst_shd.FsParams{
            .eye_pos = f.eye_pos,
            .light_dir = f.light_dir,
            .light_color = f.light_color,
            .ambient_color = f.ambient_color,
            .diffuse_color = std_mat.getDiffuseColor4(),
            .alpha_cutoff = uniforms.alphaCutoffFor(mesh.material),
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

    sg.draw(0, mesh.index_count, mesh.visible_instance_count);
    env.stats.main_draw_calls += 1;
    env.stats.draw_calls += 1;
    env.stats.triangles += (mesh.index_count / 3) * mesh.visible_instance_count;
}
