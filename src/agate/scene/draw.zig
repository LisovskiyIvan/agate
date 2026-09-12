const sokol = @import("sokol");
const sg = sokol.gfx;
const shd = @import("shader");
const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_shd = @import("instanced_shader");

const math = @import("math");
const Mat4 = math.Mat4;

const render_queue = @import("render_queue.zig");
const uniforms = @import("uniforms.zig");
const pipelines = @import("pipelines.zig");

pub const FrameContext = uniforms.FrameContext;

// Draws one regular (non-instanced) queue item: pipeline select, bind,
// uniforms, draw. Shared by the opaque pass and the transparent pass
// (pipeline choice follows item.transparent; double-sided items additionally
// resolve to the cull-off twins via pipelines.pipelineForRegularItem).
// Cutout items ride the opaque pass; their alpha_cutoff uniform enables the
// in-shader discard. Updates stats.
// `scene` is generic (anytype) to avoid a scene.zig import cycle; it must
// expose the Scene render state (pipelines, default textures, shadow pass,
// stats, frameUniforms, default_material, skybox/ibl fields).
pub fn drawRegularItem(scene: anytype, item: anytype, ctx: FrameContext, current_pipeline_id: *u32) void {
    const mesh = item.mesh;
    const model = item.model;
    const mvp = Mat4.mul(ctx.view_proj, model);

    const pip_id = pipelines.pipelineForRegularItem(scene, item);

    if (pip_id == 0) return;
    if (pip_id != current_pipeline_id.*) {
        sg.applyPipeline(.{ .id = pip_id });
        current_pipeline_id.* = pip_id;
        scene.stats.pipeline_switches += 1;
    }

    var bind = sg.Bindings{};
    bind.vertex_buffers[0] = mesh.vertex_buffer;
    bind.index_buffer = mesh.index_buffer;

    if (item.is_pbr) {
        const pbr_mat = if (mesh.material) |m| m.pbr else null;
        const albedo_tex = if (pbr_mat) |p| (p.albedo_texture orelse scene.default_white_texture) else scene.default_white_texture;
        const normal_tex = if (pbr_mat) |p| (p.normal_texture orelse scene.default_normal_texture) else scene.default_normal_texture;
        const mr_tex = if (pbr_mat) |p| (p.metallic_roughness_texture orelse scene.default_white_texture) else scene.default_white_texture;
        const emissive_tex = if (pbr_mat) |p| (p.emissive_texture orelse scene.default_white_texture) else scene.default_white_texture;
        const occlusion_tex = if (pbr_mat) |p| (p.occlusion_texture orelse scene.default_white_texture) else scene.default_white_texture;

        bind.views[pbr_shd.VIEW_albedo_tex] = albedo_tex.view;
        bind.views[pbr_shd.VIEW_normal_tex] = normal_tex.view;
        bind.views[pbr_shd.VIEW_metallic_roughness_tex] = mr_tex.view;
        bind.views[pbr_shd.VIEW_emissive_tex] = emissive_tex.view;
        bind.views[pbr_shd.VIEW_occlusion_tex] = occlusion_tex.view;
        bind.samplers[pbr_shd.SMP_smp] = albedo_tex.sampler;

        // Environment IBL Cubemap & Shadow Depth Map: a material-level
        // environment texture (e.g. an HDR probe) overrides the skybox.
        const cube = (if (pbr_mat) |p| p.environment_texture else null) orelse scene.skybox_texture orelse scene.default_cube_texture;
        bind.views[pbr_shd.VIEW_env_tex] = cube.view;
        bind.samplers[pbr_shd.SMP_env_smp] = cube.sampler;

        bind.views[pbr_shd.VIEW_shadow_tex] = scene.shadow_pass.texture_view;
        bind.views[pbr_shd.VIEW_shadow_depth_tex] = scene.shadow_pass.texture_view;
        bind.views[pbr_shd.VIEW_spot_shadow_tex] = scene.shadow_pass.spot_texture_view;
        bind.samplers[pbr_shd.SMP_shadow_smp] = scene.shadow_pass.sampler;
        bind.samplers[pbr_shd.SMP_depth_smp] = scene.shadow_pass.depth_sampler;

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
        }

        const mat_albedo = if (pbr_mat) |p| p.getAlbedoColor4() else [4]f32{ 1, 1, 1, 1 };
        const metallic = if (pbr_mat) |p| p.metallic else 0.0;
        const roughness = if (pbr_mat) |p| p.roughness else 0.5;
        const env_intensity = if (pbr_mat) |p| scene.ibl_intensity * p.environment_intensity else scene.ibl_intensity;
        const emissive_col = if (pbr_mat) |p| [4]f32{ p.emissive_color.r, p.emissive_color.g, p.emissive_color.b, 1.0 } else [4]f32{ 0, 0, 0, 1 };
        const occlusion_strength = if (pbr_mat) |p| p.occlusion_strength else 1.0;

        const f = scene.frameUniforms(mesh, ctx);
        const fs_params = pbr_shd.FsParams{
            .eye_pos = f.eye_pos,
            .light_dir = f.light_dir,
            .light_color = f.light_color,
            .ambient_color = f.ambient_color,
            .base_color_factor = mat_albedo,
            .pbr_factors = .{ metallic, roughness, occlusion_strength, env_intensity },
            .emissive_factor = emissive_col,
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
        if (mesh.skeleton != null) {
            sg.applyUniforms(skinned_pbr_shd.UB_fs_params, sg.asRange(&fs_params));
        } else {
            sg.applyUniforms(pbr_shd.UB_fs_params, sg.asRange(&fs_params));
        }
    } else {
        const std_mat = if (mesh.material) |m| m.standard else &scene.default_material;
        const tex = if (std_mat.diffuse_texture) |t| t else scene.default_white_texture;

        bind.views[shd.VIEW_diffuse_tex] = tex.view;
        bind.samplers[shd.SMP_smp] = tex.sampler;

        bind.views[shd.VIEW_shadow_tex] = scene.shadow_pass.texture_view;
        bind.views[shd.VIEW_shadow_depth_tex] = scene.shadow_pass.texture_view;
        bind.views[shd.VIEW_spot_shadow_tex] = scene.shadow_pass.spot_texture_view;
        bind.samplers[shd.SMP_shadow_smp] = scene.shadow_pass.sampler;
        bind.samplers[shd.SMP_depth_smp] = scene.shadow_pass.depth_sampler;

        sg.applyBindings(bind);

        const vs_params = shd.VsParams{
            .mvp = mvp,
            .model = model,
        };
        sg.applyUniforms(shd.UB_vs_params, sg.asRange(&vs_params));

        const f = scene.frameUniforms(mesh, ctx);
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
    scene.stats.draw_calls += 1;
    scene.stats.triangles += mesh.index_count / 3;
}

// Draws one instanced mesh with the currently visible instance buffer.
// Transparent instanced meshes use the blend twin pipeline and are drawn
// as-is (no per-instance back-to-front sort); the caller draws them
// after all opaque geometry. Double-sided materials select the cull-off
// twins (opaque or blend) when the scene provides them. Updates stats.
pub fn drawInstancedMesh(scene: anytype, mesh: anytype, ctx: FrameContext, current_pipeline_id: *u32) void {
    if (mesh.instances.items.len == 0) return;
    if (mesh.visible_instance_count == 0 or mesh.instance_buffer.id == 0) return;

    const transparent = render_queue.materialIsTransparent(mesh.material);
    const is_u32 = mesh.index_type == .UINT32;
    // Double-sided instanced meshes use the cull-off twins when the scene
    // provides them; otherwise the regular pipelines (legacy behavior).
    const pip_id = pipelines.pipelineForInstancedMesh(scene, transparent, is_u32, render_queue.materialIsDoubleSided(mesh.material));
    if (pip_id == 0) return;
    if (pip_id != current_pipeline_id.*) {
        sg.applyPipeline(.{ .id = pip_id });
        current_pipeline_id.* = pip_id;
        scene.stats.pipeline_switches += 1;
    }

    var bind = sg.Bindings{};
    bind.vertex_buffers[0] = mesh.vertex_buffer;
    bind.vertex_buffers[1] = mesh.instance_buffer;
    bind.index_buffer = mesh.index_buffer;

    const std_mat = if (mesh.material) |m| switch (m) {
        .standard => |s| s,
        .pbr => &scene.default_material,
    } else &scene.default_material;
    const tex = if (std_mat.diffuse_texture) |t| t else scene.default_white_texture;

    bind.views[inst_shd.VIEW_diffuse_tex] = tex.view;
    bind.samplers[inst_shd.SMP_smp] = tex.sampler;

    bind.views[inst_shd.VIEW_shadow_tex] = scene.shadow_pass.texture_view;
    bind.views[inst_shd.VIEW_shadow_depth_tex] = scene.shadow_pass.texture_view;
    bind.views[inst_shd.VIEW_spot_shadow_tex] = scene.shadow_pass.spot_texture_view;
    bind.samplers[inst_shd.SMP_shadow_smp] = scene.shadow_pass.sampler;
    bind.samplers[inst_shd.SMP_depth_smp] = scene.shadow_pass.depth_sampler;

    sg.applyBindings(bind);

    const inst_vs = inst_shd.VsParams{
        .view_proj = ctx.view_proj,
    };
    sg.applyUniforms(inst_shd.UB_vs_params, sg.asRange(&inst_vs));

    const f = scene.frameUniforms(mesh, ctx);
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

    sg.draw(0, mesh.index_count, mesh.visible_instance_count);
    scene.stats.draw_calls += 1;
    scene.stats.triangles += (mesh.index_count / 3) * mesh.visible_instance_count;
}
