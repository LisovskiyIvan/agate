const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_pbr_shd = @import("instanced_pbr_shader");

const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;

const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const material_mod = @import("../material.zig");
const PBRMaterial = material_mod.PBRMaterial;
const Material = material_mod.Material;
const passes = @import("../passes/mod.zig");

const render_queue = @import("render_queue.zig");
const RenderMeshItem = render_queue.RenderMeshItem;
const RenderInstancedBatch = render_queue.RenderInstancedBatch;
const morph_gpu = @import("../mesh/morph_gpu.zig");
const MAX_BONES = @import("../animation/skeleton.zig").MAX_BONES;
const uniforms = @import("uniforms.zig");
const probe_layer = @import("probe_layer.zig");
const clustered_layer = @import("clustered_lights.zig");
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
    // Reflection-probe pack for the frame (wave 25): snapshot-owned borrow
    // (see SceneFrameSnapshot.probe_pack). Per regular draw the nearest
    // enabled+captured probe containing the object wins (no blending);
    // instanced batches skip probes in v1 (no single object position) and
    // always take the legacy path.
    probes: []const probe_layer.ProbeFrameEntry = &.{},
    // Clustered forward lights (wave 30): render-owned tile cache borrow
    // (Scene.clustered, rebuilt per view from the staged snapshot). Null
    // binds zero views (legacy path; only defensive — both render paths
    // thread the cache). The bound triple is the real storage views when
    // a GPU upload landed, else the shared dummy; the count/live uniform
    // in FrameContext gates the shader loop either way.
    clustered: ?*const clustered_layer.ClusteredGpuCache = null,
    // Which per-view buffer slot of the clustered cache this draw binds
    // (0 = primary, 1+ = secondary PIP views). Set per view by
    // renderSceneView from its own slot; every other construction site
    // keeps the default (primary). Single-view frames always bind slot 0.
    clustered_slot: usize = 0,
    // Shadow map views/samplers from the CSM/spot atlas.
    shadow_pass: *const passes.ShadowPass,
    // Scene-level fragment uniform inputs; mesh.receive_shadows is patched
    // per draw before building the shader uniforms.
    shadow_uniforms: uniforms.ShadowState,
    capture_opaque_only: bool = false,
    refraction_view: sg.View = .{},
    refraction_sampler: sg.Sampler = .{},
    refraction_view_proj: Mat4 = Mat4.identity,
    refraction_capture: [4]f32 = .{ 0, 1, 1, 0 }, // valid, width, height
};

// Draws one regular (non-instanced) queue item: pipeline select, bind,
// uniforms, draw. Shared by the opaque pass and the transparent pass
// (pipeline choice follows item.transparent; double-sided items additionally
// resolve to the cull-off twins via pipelines.forRegularItem).
// Cutout items ride the opaque pass; their alpha_cutoff uniform enables the
// in-shader discard. Updates stats.
// Reads only render-owned item/draw_record data plus queue storage slices
// (skins/shaders after all prepare allocations).
pub fn drawRegularItem(
    env: *const Environment,
    item: RenderMeshItem,
    ctx: *const FrameContext,
    current_pipeline_id: *u32,
    skins: []const [MAX_BONES]Mat4,
    shaders: []const material_mod.ShaderDrawSnapshot,
    coats: []const material_mod.CoatParams,
) void {
    if (item.index_count == 0) return;
    if (sg.isvalid()) {
        if (item.vertex_buffer.id == 0 or sg.queryBufferState(item.vertex_buffer) != .VALID) return;
        if (item.index_buffer.id != 0 and sg.queryBufferState(item.index_buffer) != .VALID) return;
    }

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

    bind.views[pbr_shd.VIEW_albedo_tex] = rec.albedo_view;
    bind.views[pbr_shd.VIEW_normal_tex] = rec.normal_view;
    bind.views[pbr_shd.VIEW_metallic_roughness_tex] = rec.mr_view;
    bind.views[pbr_shd.VIEW_emissive_tex] = rec.emissive_view;
    bind.views[pbr_shd.VIEW_occlusion_tex] = rec.occlusion_view;
    // PBR layers v1: coat/fabric masks (white fallback staged in the
    // record when unset); sampled through data_smp, no extra sampler.
    bind.views[pbr_shd.VIEW_clearcoat_tex] = rec.clearcoat_view;
    bind.views[pbr_shd.VIEW_sheen_tex] = rec.sheen_view;
    bind.views[pbr_shd.VIEW_refraction_tex] = if (env.refraction_view.id != 0) env.refraction_view else env.default_white.view;
    bind.samplers[pbr_shd.SMP_refraction_smp] = if (env.refraction_sampler.id != 0) env.refraction_sampler else env.default_white.sampler;
    bind.samplers[pbr_shd.SMP_smp] = rec.albedo_sampler;
    bind.samplers[pbr_shd.SMP_data_smp] = rec.data_sampler;
    // Babylon env-BRDF lookup: scales the analytic specular sum and the
    // specular IBL (coloredEnergyConservationFactor).
    bind.views[pbr_shd.VIEW_brdf_lut_tex] = rec.brdf_lut_view;
    bind.samplers[pbr_shd.SMP_brdf_lut_smp] = rec.brdf_lut_sampler;

    // Environment IBL Cubemap & Shadow Depth Map
    const cube_view = rec.env_view orelse (env.sky_texture orelse env.default_cube).view;
    const cube_sampler = rec.env_sampler orelse (env.sky_texture orelse env.default_cube).sampler;
    bind.views[pbr_shd.VIEW_env_tex] = cube_view;
    bind.samplers[pbr_shd.SMP_env_smp] = cube_sampler;

    bind.views[pbr_shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
    bind.views[pbr_shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
    bind.views[pbr_shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
    bind.views[pbr_shd.VIEW_point_shadow_tex] = env.shadow_pass.point_texture_view;
    bind.samplers[pbr_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
    bind.samplers[pbr_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;
    bind.views[pbr_shd.VIEW_morph_tex] = morph_view;
    bind.samplers[pbr_shd.SMP_morph_smp] = env.pipelines.morph_sampler;

    // Reflection probe (wave 25, wave C.2): winning probe cubes or the default
    // cube with zeroed params (legacy path) when none applies.
    const prb = probeForDraw(env, item.model);
    bind.views[pbr_shd.VIEW_probe_tex] = prb.view;
    bind.views[pbr_shd.VIEW_probe2_tex] = prb.view2;
    bind.samplers[pbr_shd.SMP_probe_smp] = prb.sampler;

    // Clustered tile storage (wave 30): real views when live, dummy
    // otherwise (always valid binds; the uniform gates the loop).
    bindClusteredViews(&bind, pbr_shd, env);

    sg.applyBindings(bind);

    const vs_params = pbr_shd.VsParams{
        .mvp = mvp,
        .model = model,
    };
    sg.applyUniforms(pbr_shd.UB_vs_params, sg.asRange(&vs_params));

    const skel_bones: ?*const [MAX_BONES]Mat4 = if (item.is_skinned)
        // Corrupted skinned index: skip draw rather than upload stale bones.
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
    const coat = resolveCoat(coats, item.coat_index);
    const fs_params = pbr_shd.FsParams{
        .eye_pos = f.eye_pos,
        .light_dir = f.light_dir,
        .light_color = f.light_color,
        .ambient_color = f.ambient_color,
        .hemi_dir_intensity = f.hemi_dir_intensity,
        .hemi_diffuse = f.hemi_diffuse,
        .base_color_factor = rec.base_color,
        .pbr_factors = rec.pbr_factors,
        .emissive_factor = rec.emissive_color,
        .alpha_cutoff = rec.alpha_cutoff,
        .normal_scale = rec.normal_scale,
        .uv_matrix = rec.uv_matrices,
        .uv_offset = rec.uv_offsets,
        .channel_selectors = rec.channel_selectors,
        .clearcoat_factors = coat.clearcoat_factors,
        .clearcoat_color = coat.clearcoat_color,
        .sheen_factors = coat.sheen_factors,
        .sheen_color = coat.sheen_color,
        .anisotropy_factors = coat.anisotropy_factors,
        .transmission_factors = coat.transmission_factors,
        .transmission_color = coat.transmission_color,
        .sss_factors = coat.sss_factors,
        .sss_color = coat.sss_color,
        .clearcoat_uv_matrix = coat.clearcoat_uv_matrix,
        .clearcoat_uv_offset = coat.clearcoat_uv_offset,
        .sheen_uv_matrix = coat.sheen_uv_matrix,
        .sheen_uv_offset = coat.sheen_uv_offset,
        .refraction_factors = coat.refraction_factors,
        .refraction_view_proj = env.refraction_view_proj,
        .refraction_capture = env.refraction_capture,
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
        .point_view_proj = f.point_view_proj,
        .point_shadow_params = f.point_shadow_params,
        .directional_dir = f.directional_dir,
        .directional_color_int = f.directional_color_int,
        .probe_params = prb.params,
        .probe2_params = prb.params2,
        .probe_box = prb.box,
        .area_center_int = f.area_center_int,
        .area_right = f.area_right,
        .area_up = f.area_up,
        .area_color = f.area_color,
        .clustered_params = f.clustered_params,
        .clustered_viewport = f.clustered_viewport,
    };
    if (skel_bones != null) {
        sg.applyUniforms(skinned_pbr_shd.UB_fs_params, sg.asRange(&fs_params));
    } else {
        sg.applyUniforms(pbr_shd.UB_fs_params, sg.asRange(&fs_params));
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
    // Sidedness of hook material comes from snapshot (sm.double_sided) without decal forcing.
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
        // PBR-base hook material: full PBR lighting with engine defaults
        // for maps the material does not override. Texture is from prepare snapshot.
        bind.views[pbr_shd.VIEW_albedo_tex] = snap.tex_view;
        bind.views[pbr_shd.VIEW_normal_tex] = env.default_normal.view;
        bind.views[pbr_shd.VIEW_metallic_roughness_tex] = env.default_white.view;
        bind.views[pbr_shd.VIEW_emissive_tex] = env.default_white.view;
        bind.views[pbr_shd.VIEW_occlusion_tex] = env.default_white.view;
        // Hook materials carry no coat/fabric maps: white keeps the
        // mask slots valid (identity sampling).
        bind.views[pbr_shd.VIEW_clearcoat_tex] = env.default_white.view;
        bind.views[pbr_shd.VIEW_sheen_tex] = env.default_white.view;
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
        bind.views[pbr_shd.VIEW_point_shadow_tex] = env.shadow_pass.point_texture_view;
        bind.samplers[pbr_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
        bind.samplers[pbr_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;

        bind.views[pbr_shd.VIEW_morph_tex] = morph_view;
        bind.views[pbr_shd.VIEW_refraction_tex] = env.default_white.view;
        bind.samplers[pbr_shd.SMP_refraction_smp] = env.default_white.sampler;
        bind.samplers[pbr_shd.SMP_morph_smp] = env.pipelines.morph_sampler;
        // Reflection probe (wave 25): hook materials resolve like
        // regular draws (per-object selection from the model).
        const prb_hook = probeForDraw(env, item.model);
        bind.views[pbr_shd.VIEW_probe_tex] = prb_hook.view;
        bind.views[pbr_shd.VIEW_probe2_tex] = prb_hook.view2;
        bind.samplers[pbr_shd.SMP_probe_smp] = prb_hook.sampler;
        // Babylon env-BRDF lookup: scales the analytic specular sum and the
        // specular IBL (coloredEnergyConservationFactor).
        const rec = item.draw_record;
        bind.views[pbr_shd.VIEW_brdf_lut_tex] = if (rec.brdf_lut_view.id != 0) rec.brdf_lut_view else env.default_white.view;
        bind.samplers[pbr_shd.SMP_brdf_lut_smp] = if (rec.brdf_lut_sampler.id != 0) rec.brdf_lut_sampler else env.default_white.sampler;
        // Clustered tile storage (wave 30): see the regular PBR branch.
        bindClusteredViews(&bind, pbr_shd, env);
        sg.applyBindings(bind);

        const vs_params = pbr_shd.VsParams{ .mvp = mvp, .model = item.model };
        sg.applyUniforms(entry.vs_ub, sg.asRange(&vs_params));
        sg.applyUniforms(pbr_shd.UB_vs_morph, sg.asRange(&vsMorphUniform(pbr_shd, morph_uniforms)));

        const fs_params = pbr_shd.FsParams{
            .eye_pos = f.eye_pos,
            .light_dir = f.light_dir,
            .light_color = f.light_color,
            .ambient_color = f.ambient_color,
            .hemi_dir_intensity = f.hemi_dir_intensity,
            .hemi_diffuse = f.hemi_diffuse,
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
            // Hook materials carry no coat/fabric layers: neutral-disabled.
            .clearcoat_factors = material_mod.CoatParams.neutral.clearcoat_factors,
            .clearcoat_color = material_mod.CoatParams.neutral.clearcoat_color,
            .sheen_factors = material_mod.CoatParams.neutral.sheen_factors,
            .sheen_color = material_mod.CoatParams.neutral.sheen_color,
            .anisotropy_factors = material_mod.CoatParams.neutral.anisotropy_factors,
            .transmission_factors = material_mod.CoatParams.neutral.transmission_factors,
            .transmission_color = material_mod.CoatParams.neutral.transmission_color,
            .sss_factors = material_mod.CoatParams.neutral.sss_factors,
            .sss_color = material_mod.CoatParams.neutral.sss_color,
            .clearcoat_uv_matrix = material_mod.CoatParams.neutral.clearcoat_uv_matrix,
            .clearcoat_uv_offset = material_mod.CoatParams.neutral.clearcoat_uv_offset,
            .sheen_uv_matrix = material_mod.CoatParams.neutral.sheen_uv_matrix,
            .sheen_uv_offset = material_mod.CoatParams.neutral.sheen_uv_offset,
            .refraction_factors = material_mod.CoatParams.neutral.refraction_factors,
            .refraction_view_proj = Mat4.identity,
            .refraction_capture = .{ 0, 1, 1, 0 },
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
            .point_view_proj = f.point_view_proj,
            .point_shadow_params = f.point_shadow_params,
            .directional_dir = f.directional_dir,
            .directional_color_int = f.directional_color_int,
            .probe_params = prb_hook.params,
            .probe2_params = prb_hook.params2,
            .probe_box = prb_hook.box,
            .area_center_int = f.area_center_int,
            .area_right = f.area_right,
            .area_up = f.area_up,
            .area_color = f.area_color,
            .clustered_params = f.clustered_params,
            .clustered_viewport = f.clustered_viewport,
        };
        sg.applyUniforms(entry.fs_ub, sg.asRange(&fs_params));
    } else {
        // Runtime-registered custom source: vertex/index binds plus the
        // material texture at view slot 0 (sokol tolerates binding slots the
        // shader does not declare; shaders that declare nothing get white).
        // Contract: UB 0 carries {mat4 mvp, mat4 model} like every forward
        // shader (runtime sources must declare it, see registerRuntime docs).
        // Texture view slot 0 is from snapshot; second (texture1) is view slot 1.
        // Shaders not declaring the slot receive snapshot white default without errors.
        bind.views[0] = snap.tex_view;
        bind.samplers[0] = snap.tex_sampler;
        bind.views[1] = snap.tex1_view;
        bind.samplers[1] = snap.tex1_sampler;
        sg.applyBindings(bind);

        const vs_params = pbr_shd.VsParams{ .mvp = mvp, .model = item.model };
        sg.applyUniforms(entry.vs_ub, sg.asRange(&vs_params));
    }

    // User uniform block (declarative param table -> packed storage);
    // materials with vertex-stage params carry the same payload on the
    // fs-stage and vs-stage blocks. The upload is exactly the entry's
    // declared wire size (user_bytes): hook blocks always span the full
    // 128 bytes, external shaders declare their own window (v1: <= 2x
    // vec4) — sokol validates the size, so a short slice is required.
    if (entry.user_ub) |ub| {
        const bytes = shader_material.uniformBytes(&snap.uniforms);
        sg.applyUniforms(ub, sg.asRange(bytes[0..@min(entry.user_bytes, bytes.len)]));
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

pub const ProbeDrawState = struct {
    params: [4]f32,
    params2: [4]f32,
    box: [2][4]f32,
    view: sg.View,
    view2: sg.View,
    sampler: sg.Sampler,
};

/// Per-draw reflection-probe resolution (wave 25, wave C.2, wave Q.2): selects up to
/// two overlapping enabled+captured probes containing the object's world
/// position with smooth continuous falloff weights and box bounds; otherwise the legacy path
/// (zeroed params, so the shader takes its bit-identical no-probe branch).
/// Pure (no GPU calls).
fn probeForDrawPos(env: *const Environment, pos: Vec3) ProbeDrawState {
    const sel = probe_layer.selectProbes(env.probes, pos);
    var out = ProbeDrawState{
        .params = .{ 0.0, 0.0, 0.0, 0.0 },
        .params2 = .{ 0.0, 0.0, 0.0, 0.0 },
        .box = .{
            .{ 0.0, 0.0, 0.0, 0.0 },
            .{ 0.0, 0.0, 0.0, 0.0 },
        },
        .view = env.default_cube.view,
        .view2 = env.default_cube.view,
        .sampler = env.default_cube.sampler,
    };
    if (sel.primary) |p0| {
        out.params = .{ 1.0, p0.probe.intensity, p0.probe.max_probe_lod, p0.weight };
        out.box[0] = .{ p0.probe.position.x, p0.probe.position.y, p0.probe.position.z, p0.probe.box_extents };
        out.view = p0.probe.view;
        out.sampler = p0.probe.sampler;
    }
    if (sel.secondary) |p1| {
        out.params2 = .{ 1.0, p1.probe.intensity, p1.probe.max_probe_lod, p1.weight };
        out.box[1] = .{ p1.probe.position.x, p1.probe.position.y, p1.probe.position.z, p1.probe.box_extents };
        out.view2 = p1.probe.view;
    }
    return out;
}

pub fn probeForDraw(env: *const Environment, model: Mat4) ProbeDrawState {
    const pos = Vec3.new(model.m[12], model.m[13], model.m[14]);
    return probeForDrawPos(env, pos);
}

// Binds the clustered tile storage views for one draw: the real views of
// this draw's view slot when that slot's GPU side mirrors its view's
// staged tiles, else the shared dummy (declared SSBO slots stay valid;
// the count/live uniform gates the shader loop). The module's VIEW_*
// consts come from its own generated shader module (identical 12/13/14
// slots in all five forward shaders). Pure (no GPU calls).
fn bindClusteredViews(bind: *sg.Bindings, comptime module: anytype, env: *const Environment) void {
    const v: clustered_layer.ClusterBindingViews = if (env.clustered) |c| c.bindingViewsForSlot(env.clustered_slot) else .{};
    bind.views[module.VIEW_ssbo_cluster_lights] = v.lights;
    bind.views[module.VIEW_ssbo_cluster_tiles] = v.tiles;
    bind.views[module.VIEW_ssbo_cluster_indices] = v.indices;
}

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

// Resolves the coat/fabric factors for a draw: the owned side-table copy by
// index, CoatParams.neutral when the lobe is off (null) or the index is
// stale (unreachable via the builders). Pure (no GPU calls).
pub fn resolveCoat(coats: []const material_mod.CoatParams, index: ?u32) material_mod.CoatParams {
    if (render_queue.coatAt(coats, index)) |cp| return cp.*;
    return material_mod.CoatParams.neutral;
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
pub fn pbrUvMatrices(pbr_mat: ?*const PBRMaterial) [5][4]f32 {
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

pub fn pbrUvOffsets(pbr_mat: ?*const PBRMaterial) [5][4]f32 {
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
pub fn pbrChannelSelectors(pbr_mat: ?*const PBRMaterial) [4]f32 {
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
pub fn drawInstancedBatch(env: *const Environment, batch: RenderInstancedBatch, ctx: *const FrameContext, current_pipeline_id: *u32, coats: []const material_mod.CoatParams) void {
    if (batch.visible_instance_count == 0 or batch.instance_buffer.id == 0) return;
    if (sg.isvalid()) {
        if (batch.vertex_buffer.id == 0 or sg.queryBufferState(batch.vertex_buffer) != .VALID) return;
        if (batch.index_buffer.id != 0 and sg.queryBufferState(batch.index_buffer) != .VALID) return;
        if (sg.queryBufferState(batch.instance_buffer) != .VALID) return;
    }

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
    bind.views[inst_pbr_shd.VIEW_albedo_tex] = rec.albedo_view;
    bind.views[inst_pbr_shd.VIEW_normal_tex] = rec.normal_view;
    bind.views[inst_pbr_shd.VIEW_metallic_roughness_tex] = rec.mr_view;
    bind.views[inst_pbr_shd.VIEW_emissive_tex] = rec.emissive_view;
    bind.views[inst_pbr_shd.VIEW_occlusion_tex] = rec.occlusion_view;
    // PBR layers v1: see the regular PBR branch above.
    bind.views[inst_pbr_shd.VIEW_clearcoat_tex] = rec.clearcoat_view;
    bind.views[inst_pbr_shd.VIEW_sheen_tex] = rec.sheen_view;
    bind.views[inst_pbr_shd.VIEW_refraction_tex] = if (env.refraction_view.id != 0) env.refraction_view else env.default_white.view;
    bind.samplers[inst_pbr_shd.SMP_refraction_smp] = if (env.refraction_sampler.id != 0) env.refraction_sampler else env.default_white.sampler;
    bind.samplers[inst_pbr_shd.SMP_smp] = rec.albedo_sampler;
    bind.samplers[inst_pbr_shd.SMP_data_smp] = rec.data_sampler;
    // Babylon env-BRDF lookup: scales the analytic specular sum and the
    // specular IBL (coloredEnergyConservationFactor).
    bind.views[inst_pbr_shd.VIEW_brdf_lut_tex] = rec.brdf_lut_view;
    bind.samplers[inst_pbr_shd.SMP_brdf_lut_smp] = rec.brdf_lut_sampler;

    // Environment IBL Cubemap & Shadow Depth Map
    const cube_view = rec.env_view orelse (if (env.sky_texture) |s| s.view else env.default_cube.view);
    const cube_sampler = rec.env_sampler orelse (if (env.sky_texture) |s| s.sampler else env.default_cube.sampler);
    bind.views[inst_pbr_shd.VIEW_env_tex] = cube_view;
    bind.samplers[inst_pbr_shd.SMP_env_smp] = cube_sampler;

    bind.views[inst_pbr_shd.VIEW_shadow_tex] = env.shadow_pass.texture_view;
    bind.views[inst_pbr_shd.VIEW_shadow_depth_tex] = env.shadow_pass.texture_view;
    bind.views[inst_pbr_shd.VIEW_spot_shadow_tex] = env.shadow_pass.spot_texture_view;
    bind.views[inst_pbr_shd.VIEW_point_shadow_tex] = env.shadow_pass.point_texture_view;
    bind.samplers[inst_pbr_shd.SMP_shadow_smp] = env.shadow_pass.sampler;
    bind.samplers[inst_pbr_shd.SMP_depth_smp] = env.shadow_pass.depth_sampler;

    // Reflection probe (wave 25, wave C.2): selected probe cubes for batch.world_center
    // or default cube with zeroed params when none applies.
    const prb = probeForDrawPos(env, batch.world_center);
    bind.views[inst_pbr_shd.VIEW_probe_tex] = prb.view;
    bind.views[inst_pbr_shd.VIEW_probe2_tex] = prb.view2;
    bind.samplers[inst_pbr_shd.SMP_probe_smp] = prb.sampler;

    // Clustered tile storage (wave 30): see the regular PBR branch.
    bindClusteredViews(&bind, inst_pbr_shd, env);

    sg.applyBindings(bind);

    const inst_vs = inst_pbr_shd.VsParams{
        .view_proj = ctx.view_proj,
    };
    sg.applyUniforms(inst_pbr_shd.UB_vs_params, sg.asRange(&inst_vs));

    const f = frameUniformsForState(env.shadow_uniforms, batch.receive_shadows, ctx);
    const coat = resolveCoat(coats, batch.coat_index);
    const inst_fs = inst_pbr_shd.FsParams{
        .eye_pos = f.eye_pos,
        .light_dir = f.light_dir,
        .light_color = f.light_color,
        .ambient_color = f.ambient_color,
        .hemi_dir_intensity = f.hemi_dir_intensity,
        .hemi_diffuse = f.hemi_diffuse,
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
        .point_view_proj = f.point_view_proj,
        .point_shadow_params = f.point_shadow_params,
        .directional_dir = f.directional_dir,
        .directional_color_int = f.directional_color_int,
        .probe_params = prb.params,
        .probe2_params = prb.params2,
        .probe_box = prb.box,
        .area_center_int = f.area_center_int,
        .area_right = f.area_right,
        .area_up = f.area_up,
        .area_color = f.area_color,
        .clustered_params = f.clustered_params,
        .clustered_viewport = f.clustered_viewport,
        .alpha_cutoff = rec.alpha_cutoff,
        .normal_scale = rec.normal_scale,
        .uv_matrix = rec.uv_matrices,
        .uv_offset = rec.uv_offsets,
        .channel_selectors = rec.channel_selectors,
        .clearcoat_factors = coat.clearcoat_factors,
        .clearcoat_color = coat.clearcoat_color,
        .sheen_factors = coat.sheen_factors,
        .sheen_color = coat.sheen_color,
        .anisotropy_factors = coat.anisotropy_factors,
        .transmission_factors = coat.transmission_factors,
        .transmission_color = coat.transmission_color,
        .sss_factors = coat.sss_factors,
        .sss_color = coat.sss_color,
        .clearcoat_uv_matrix = coat.clearcoat_uv_matrix,
        .clearcoat_uv_offset = coat.clearcoat_uv_offset,
        .sheen_uv_matrix = coat.sheen_uv_matrix,
        .sheen_uv_offset = coat.sheen_uv_offset,
        .refraction_factors = coat.refraction_factors,
        .refraction_view_proj = env.refraction_view_proj,
        .refraction_capture = env.refraction_capture,
    };
    sg.applyUniforms(inst_pbr_shd.UB_fs_params, sg.asRange(&inst_fs));

    sg.draw(0, batch.index_count, batch.visible_instance_count);
    env.stats.main_draw_calls += 1;
    env.stats.draw_calls += 1;
    env.stats.triangles += (batch.index_count / 3) * batch.visible_instance_count;
}
