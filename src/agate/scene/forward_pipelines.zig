const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const shd = @import("shader");
const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_shd = @import("instanced_shader");
const inst_pbr_shd = @import("instanced_pbr_shader");

const scene_pipelines = @import("pipelines.zig");
const render_queue = @import("render_queue.zig");
const RenderMeshItem = render_queue.RenderMeshItem;

/// All forward-rendering GPU pipelines in one place: the opaque u16/u32
/// pairs and their transparent blend twins for the 5 shader families, plus
/// the double-sided (cull-off) twin set. Created once at Scene init; the
/// per-item selection helpers keep the legacy mapping bit-identical.
pub const ForwardPipelines = struct {
    // Mesh Forward Pipelines
    pipeline_u16: sg.Pipeline = .{},
    pipeline_u32: sg.Pipeline = .{},
    pipeline_pbr_u16: sg.Pipeline = .{},
    pipeline_pbr_u32: sg.Pipeline = .{},
    pipeline_skinned_pbr_u16: sg.Pipeline = .{},
    pipeline_skinned_pbr_u32: sg.Pipeline = .{},
    pipeline_instanced_u16: sg.Pipeline = .{},
    pipeline_instanced_u32: sg.Pipeline = .{},
    pipeline_instanced_pbr_u16: sg.Pipeline = .{},
    pipeline_instanced_pbr_u32: sg.Pipeline = .{},

    // Transparent twin pipelines: same shaders/layouts as above, but with
    // alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA), depth test on and
    // depth write off. Selected per item when item.transparent is true.
    pipeline_blend_u16: sg.Pipeline = .{},
    pipeline_blend_u32: sg.Pipeline = .{},
    pipeline_pbr_blend_u16: sg.Pipeline = .{},
    pipeline_pbr_blend_u32: sg.Pipeline = .{},
    pipeline_skinned_pbr_blend_u16: sg.Pipeline = .{},
    pipeline_skinned_pbr_blend_u32: sg.Pipeline = .{},
    pipeline_instanced_blend_u16: sg.Pipeline = .{},
    pipeline_instanced_blend_u32: sg.Pipeline = .{},
    pipeline_instanced_pbr_blend_u16: sg.Pipeline = .{},
    pipeline_instanced_pbr_blend_u32: sg.Pipeline = .{},

    // Double-sided (cull-off) twins for every family; selected per item when
    // the material sets double_sided. Created in init, freed in deinit.
    ds_pipelines: scene_pipelines.DoubleSidedPipelines = .{},

    // GPU-morph bind target for draws without morphs: binding/uniform state
    // persists across draws in sokol, so every forward draw must bind a
    // valid morph view. Non-GPU-morph meshes get this 1x1 zero RGBA32F
    // texture with the morph enable uniform off (draw.zig). The nearest
    // nonfiltering sampler matches the mesh delta sampling (RGBA32F is
    // unfilterable on Metal). Created in init, freed in deinit.
    default_morph_image: sg.Image = .{},
    default_morph_view: sg.View = .{},
    morph_sampler: sg.Sampler = .{},

    /// Builds every pipeline (GPU calls). Panics if a base pipeline fails
    /// to create — same loud failure as the legacy Scene.initPipelines.
    pub fn init() ForwardPipelines {
        var self: ForwardPipelines = .{};

        const zero_texel = [_]f32{ 0, 0, 0, 0 };
        var img_desc = sg.ImageDesc{
            .width = 1,
            .height = 1,
            .pixel_format = .RGBA32F,
        };
        img_desc.data.mip_levels[0] = sg.asRange(&zero_texel);
        self.default_morph_image = sg.makeImage(img_desc);
        self.default_morph_view = sg.makeView(.{ .texture = .{ .image = self.default_morph_image } });
        self.morph_sampler = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        // One shader handle per family, shared by the opaque/blend pairs and
        // the double-sided twins.
        const family_shaders = scene_pipelines.DoubleSidedSourceShaders{
            .standard = sg.makeShader(shd.standardShaderDesc(sg.queryBackend())),
            .pbr = sg.makeShader(pbr_shd.pbrShaderDesc(sg.queryBackend())),
            .instanced = sg.makeShader(inst_shd.instancedShaderDesc(sg.queryBackend())),
            .instanced_pbr = sg.makeShader(inst_pbr_shd.instancedPbrShaderDesc(sg.queryBackend())),
            .skinned_pbr = sg.makeShader(skinned_pbr_shd.skinnedPbrShaderDesc(sg.queryBackend())),
        };
        const specs = [_]struct {
            shader: sg.Shader,
            family: scene_pipelines.PipelineFamily,
            opaque_u16: *sg.Pipeline,
            opaque_u32: *sg.Pipeline,
            blend_u16: *sg.Pipeline,
            blend_u32: *sg.Pipeline,
        }{
            .{
                .shader = family_shaders.standard,
                .family = .standard,
                .opaque_u16 = &self.pipeline_u16,
                .opaque_u32 = &self.pipeline_u32,
                .blend_u16 = &self.pipeline_blend_u16,
                .blend_u32 = &self.pipeline_blend_u32,
            },
            .{
                .shader = family_shaders.pbr,
                .family = .pbr,
                .opaque_u16 = &self.pipeline_pbr_u16,
                .opaque_u32 = &self.pipeline_pbr_u32,
                .blend_u16 = &self.pipeline_pbr_blend_u16,
                .blend_u32 = &self.pipeline_pbr_blend_u32,
            },
            .{
                .shader = family_shaders.instanced,
                .family = .instanced,
                .opaque_u16 = &self.pipeline_instanced_u16,
                .opaque_u32 = &self.pipeline_instanced_u32,
                .blend_u16 = &self.pipeline_instanced_blend_u16,
                .blend_u32 = &self.pipeline_instanced_blend_u32,
            },
            .{
                .shader = family_shaders.instanced_pbr,
                .family = .instanced_pbr,
                .opaque_u16 = &self.pipeline_instanced_pbr_u16,
                .opaque_u32 = &self.pipeline_instanced_pbr_u32,
                .blend_u16 = &self.pipeline_instanced_pbr_blend_u16,
                .blend_u32 = &self.pipeline_instanced_pbr_blend_u32,
            },
            .{
                .shader = family_shaders.skinned_pbr,
                .family = .skinned_pbr,
                .opaque_u16 = &self.pipeline_skinned_pbr_u16,
                .opaque_u32 = &self.pipeline_skinned_pbr_u32,
                .blend_u16 = &self.pipeline_skinned_pbr_blend_u16,
                .blend_u32 = &self.pipeline_skinned_pbr_blend_u32,
            },
        };

        for (specs) |spec| {
            var desc = sg.PipelineDesc{
                .shader = spec.shader,
                .index_type = .UINT16,
                .depth = .{
                    .compare = .LESS_EQUAL,
                    .write_enabled = true,
                },
                .cull_mode = .BACK,
                .face_winding = .CCW,
            };
            scene_pipelines.pipelineLayoutFor(spec.family, &desc);
            scene_pipelines.makePipelinePair(desc, spec.opaque_u16, spec.opaque_u32, spec.blend_u16, spec.blend_u32);
        }

        self.ds_pipelines.initFromShaders(family_shaders);

        inline for (.{
            .{ .pipe = self.pipeline_u16, .msg = "pipeline_u16 failed to create!" },
            .{ .pipe = self.pipeline_u32, .msg = "pipeline_u32 failed to create!" },
            .{ .pipe = self.pipeline_pbr_u16, .msg = "pipeline_pbr_u16 failed to create!" },
            .{ .pipe = self.pipeline_pbr_u32, .msg = "pipeline_pbr_u32 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_u16, .msg = "pipeline_skinned_pbr_u16 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_u32, .msg = "pipeline_skinned_pbr_u32 failed to create!" },
            .{ .pipe = self.pipeline_instanced_u16, .msg = "pipeline_instanced_u16 failed to create!" },
            .{ .pipe = self.pipeline_instanced_u32, .msg = "pipeline_instanced_u32 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_u16, .msg = "pipeline_instanced_pbr_u16 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_u32, .msg = "pipeline_instanced_pbr_u32 failed to create!" },
            .{ .pipe = self.pipeline_blend_u16, .msg = "pipeline_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_blend_u32, .msg = "pipeline_blend_u32 failed to create!" },
            .{ .pipe = self.pipeline_pbr_blend_u16, .msg = "pipeline_pbr_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_pbr_blend_u32, .msg = "pipeline_pbr_blend_u32 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_blend_u16, .msg = "pipeline_skinned_pbr_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_blend_u32, .msg = "pipeline_skinned_pbr_blend_u32 failed to create!" },
            .{ .pipe = self.pipeline_instanced_blend_u16, .msg = "pipeline_instanced_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_instanced_blend_u32, .msg = "pipeline_instanced_blend_u32 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_blend_u16, .msg = "pipeline_instanced_pbr_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_blend_u32, .msg = "pipeline_instanced_pbr_blend_u32 failed to create!" },
        }) |entry| {
            if (entry.pipe.id == 0) @panic(entry.msg);
        }

        return self;
    }

    pub fn deinit(self: *ForwardPipelines) void {
        sg.destroyImage(self.default_morph_image);
        sg.destroyView(self.default_morph_view);
        sg.destroySampler(self.morph_sampler);
        sg.destroyPipeline(self.pipeline_u16);
        sg.destroyPipeline(self.pipeline_u32);
        sg.destroyPipeline(self.pipeline_pbr_u16);
        sg.destroyPipeline(self.pipeline_pbr_u32);
        sg.destroyPipeline(self.pipeline_skinned_pbr_u16);
        sg.destroyPipeline(self.pipeline_skinned_pbr_u32);
        sg.destroyPipeline(self.pipeline_instanced_u16);
        sg.destroyPipeline(self.pipeline_instanced_u32);
        sg.destroyPipeline(self.pipeline_blend_u16);
        sg.destroyPipeline(self.pipeline_blend_u32);
        sg.destroyPipeline(self.pipeline_pbr_blend_u16);
        sg.destroyPipeline(self.pipeline_pbr_blend_u32);
        sg.destroyPipeline(self.pipeline_skinned_pbr_blend_u16);
        sg.destroyPipeline(self.pipeline_skinned_pbr_blend_u32);
        sg.destroyPipeline(self.pipeline_instanced_blend_u16);
        sg.destroyPipeline(self.pipeline_instanced_blend_u32);
        sg.destroyPipeline(self.pipeline_instanced_pbr_u16);
        sg.destroyPipeline(self.pipeline_instanced_pbr_u32);
        sg.destroyPipeline(self.pipeline_instanced_pbr_blend_u16);
        sg.destroyPipeline(self.pipeline_instanced_pbr_blend_u32);
        self.ds_pipelines.deinit();
    }

    // Selects the forward pipeline for a regular (non-instanced) queue item.
    // Transparent items resolve to the blend twins; opaque selection is
    // identical to the legacy logic, so existing pipeline ids are untouched.
    pub fn forRegularItem(self: *const ForwardPipelines, item: RenderMeshItem) u32 {
        return scene_pipelines.pipelineForRegularItem(self, item);
    }

    // Selects the instanced pipeline. Same double-sided contract as
    // forRegularItem: cull-off twin when requested and available, regular
    // pipeline otherwise.
    pub fn forInstancedMesh(self: *const ForwardPipelines, is_pbr: bool, transparent: bool, is_u32: bool, double_sided: bool) u32 {
        return scene_pipelines.pipelineForInstancedMesh(self, is_pbr, transparent, is_u32, double_sided);
    }
};

test "pipeline selection follows transparency flag" {
    const pipelines = ForwardPipelines{
        .pipeline_u16 = .{ .id = 11 },
        .pipeline_u32 = .{ .id = 12 },
        .pipeline_blend_u16 = .{ .id = 13 },
        .pipeline_blend_u32 = .{ .id = 14 },
        .pipeline_pbr_u16 = .{ .id = 21 },
        .pipeline_pbr_u32 = .{ .id = 22 },
        .pipeline_pbr_blend_u16 = .{ .id = 23 },
        .pipeline_pbr_blend_u32 = .{ .id = 24 },
        .pipeline_skinned_pbr_u16 = .{ .id = 31 },
        .pipeline_skinned_pbr_u32 = .{ .id = 32 },
        .pipeline_skinned_pbr_blend_u16 = .{ .id = 33 },
        .pipeline_skinned_pbr_blend_u32 = .{ .id = 34 },
    };

    const Mesh = @import("../mesh.zig").Mesh;
    const Skeleton = @import("../animation/skeleton.zig").Skeleton;
    const Mat4 = @import("math").Mat4;

    var mesh_obj: Mesh = undefined;
    mesh_obj.index_type = .UINT16;
    mesh_obj.skeleton = null;
    // forRegularItem probes mesh material for double-sidedness.
    mesh_obj.material = null;

    const opaque_std = RenderMeshItem{ .mesh = &mesh_obj, .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = false, .texture_id = 0, .transparent = false };
    var blend_std = opaque_std;
    blend_std.transparent = true;
    try std.testing.expect(pipelines.forRegularItem(opaque_std) == 11);
    try std.testing.expect(pipelines.forRegularItem(blend_std) == 13);

    mesh_obj.index_type = .UINT32;
    try std.testing.expect(pipelines.forRegularItem(opaque_std) == 12);
    try std.testing.expect(pipelines.forRegularItem(blend_std) == 14);

    const opaque_pbr = RenderMeshItem{ .mesh = &mesh_obj, .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = true, .texture_id = 0, .transparent = false };
    var blend_pbr = opaque_pbr;
    blend_pbr.transparent = true;
    try std.testing.expect(pipelines.forRegularItem(opaque_pbr) == 22);
    try std.testing.expect(pipelines.forRegularItem(blend_pbr) == 24);

    var skel: Skeleton = undefined;
    mesh_obj.skeleton = &skel;
    try std.testing.expect(pipelines.forRegularItem(opaque_pbr) == 32);
    try std.testing.expect(pipelines.forRegularItem(blend_pbr) == 34);
    mesh_obj.index_type = .UINT16;
    try std.testing.expect(pipelines.forRegularItem(opaque_pbr) == 31);
    try std.testing.expect(pipelines.forRegularItem(blend_pbr) == 33);
}
