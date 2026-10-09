const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_pbr_shd = @import("instanced_pbr_shader");

const scene_pipelines = @import("pipelines.zig");
const render_queue = @import("render_queue.zig");
const RenderMeshItem = render_queue.RenderMeshItem;
const shader_material = @import("../shader_material.zig");
pub const TargetShape = @import("../target_shape.zig").TargetShape;
pub const defaultDepthFormat = @import("../target_shape.zig").defaultDepthFormat;

/// Base descriptor funnel for every main-target pipeline (defined in
/// pipelines.zig; re-exported here because this file is the pipeline front
/// door). See the definition for the sample-count contract.
pub const forwardDesc = scene_pipelines.forwardDesc;
pub const forwardDescForShape = scene_pipelines.forwardDescForShape;

/// Cache-slot key mixing a shader-material registration key with the complete
/// target shape (color format, depth format, stencil format, and sample count).
pub fn shaderMaterialKeyForShape(key: u64, shape: TargetShape) u64 {
    var h = std.hash.Wyhash.init(0x6d73_6161_2020_2020); // "msaa    "
    h.update(std.mem.asBytes(&key));
    const shape_hash = shape.hash();
    h.update(std.mem.asBytes(&shape_hash));
    return h.final();
}

pub fn shaderMaterialKey(key: u64, sample_count: i32, color_format: sg.PixelFormat) u64 {
    return shaderMaterialKeyForShape(key, .{
        .sample_count = sample_count,
        .color_format = color_format,
        .depth_format = defaultDepthFormat(),
    });
}

/// GPU pipeline set for one registered shader material: 8 pipelines shaped
/// exactly like the built-in families (opaque/blend x u16/u32 plus their
/// cull-off twins). Created lazily by ShaderMaterialCache on first use.
pub const ShaderMaterialSet = struct {
    key: u64 = 0,
    shader: sg.Shader = .{},
    opaque_u16: sg.Pipeline = .{},
    opaque_u32: sg.Pipeline = .{},
    blend_u16: sg.Pipeline = .{},
    blend_u32: sg.Pipeline = .{},
    ds_opaque_u16: sg.Pipeline = .{},
    ds_opaque_u32: sg.Pipeline = .{},
    ds_blend_u16: sg.Pipeline = .{},
    ds_blend_u32: sg.Pipeline = .{},

    /// Same contract as pipelineForRegularItem: cull-off twin when
    /// requested and available, blend twin when transparent, regular
    /// pipeline otherwise.
    pub fn pipelineFor(self: *const ShaderMaterialSet, transparent: bool, is_u32: bool, double_sided: bool) u32 {
        if (double_sided) {
            const id = switch (transparent) {
                true => if (is_u32) self.ds_blend_u32.id else self.ds_blend_u16.id,
                false => if (is_u32) self.ds_opaque_u32.id else self.ds_opaque_u16.id,
            };
            if (id != 0) return id;
        }
        return switch (transparent) {
            true => if (is_u32) self.blend_u32.id else self.blend_u16.id,
            false => if (is_u32) self.opaque_u32.id else self.opaque_u16.id,
        };
    }
};

/// Lazy pipeline cache for registered shader materials. Entries are created
/// on first use keyed by the registration key (Wyhash of the shader name,
/// see shader_material.keyForName); lookups are O(slots) linear scans, the
/// same determinism contract as the built-in pipeline table. Slots hold 8
/// pipelines + 1 shader each; a full cache makes further materials skip
/// drawing (overflow_count tracks it for diagnostics).
pub const ShaderMaterialCache = struct {
    pub const max_entries = 32;

    slots: [max_entries]ShaderMaterialSet = @splat(.{}),
    overflow_count: u32 = 0,
    /// Target shape the cached pipelines are built for, mixed
    /// into the slot key so shape variants coexist (see shaderMaterialKeyForShape).
    shape: TargetShape = .{},
    sample_count: i32 = 1,
    color_format: sg.PixelFormat = .RGBA16F,

    pub fn lookup(self: *const ShaderMaterialCache, key: u64) ?*const ShaderMaterialSet {
        const slot_key = shaderMaterialKeyForShape(key, self.shape);
        for (&self.slots) |*slot| {
            if (slot.key == slot_key and slot.opaque_u16.id != 0) return slot;
        }
        return null;
    }

    /// GPU calls on cache miss. Null when the key is unregistered, the cache
    /// is full, or pipeline creation failed (the draw path skips the mesh —
    /// a material bug must not take the frame down; core pipeline failures
    /// in ForwardPipelines.init still panic loudly).
    pub fn getOrCreate(self: *ShaderMaterialCache, key: u64) ?*const ShaderMaterialSet {
        if (self.lookup(key)) |set| return set;
        const entry = shader_material.entryForKey(key) orelse return null;
        const slot_key = shaderMaterialKeyForShape(key, self.shape);
        for (&self.slots) |*slot| {
            if (slot.key == 0 and slot.opaque_u16.id == 0 and slot.shader.id == 0) {
                const shader = entry.make_shader(sg.queryBackend());
                if (shader.id == 0) return null;
                slot.shader = shader;
                var desc = scene_pipelines.forwardDescForShape(shader, self.shape);
                scene_pipelines.pipelineLayoutFor(switch (entry.base) {
                    .standard, .pbr => .pbr,
                }, &desc);
                scene_pipelines.makePipelinePair(desc, &slot.opaque_u16, &slot.opaque_u32, &slot.blend_u16, &slot.blend_u32);
                scene_pipelines.makeCullOffPair(desc, &slot.ds_opaque_u16, &slot.ds_opaque_u32, &slot.ds_blend_u16, &slot.ds_blend_u32);
                slot.key = slot_key;
                if (slot.opaque_u16.id == 0) return null; // creation failed
                return slot;
            }
        }
        self.overflow_count += 1;
        return null;
    }

    pub fn deinit(self: *ShaderMaterialCache) void {
        for (&self.slots) |*slot| {
            inline for (.{
                &slot.opaque_u16,    &slot.opaque_u32,
                &slot.blend_u16,     &slot.blend_u32,
                &slot.ds_opaque_u16, &slot.ds_opaque_u32,
                &slot.ds_blend_u16,  &slot.ds_blend_u32,
            }) |pipe| {
                if (pipe.*.id != 0) sg.destroyPipeline(pipe.*);
                pipe.* = .{};
            }
            if (slot.shader.id != 0) sg.destroyShader(slot.shader);
            slot.shader = .{};
            slot.key = 0;
        }
        self.overflow_count = 0;
    }
};

/// All forward-rendering GPU pipelines in one place: the opaque u16/u32
/// pairs and their transparent blend twins for the 3 shader families, plus
/// the double-sided (cull-off) twin set. Created once at Scene init; the
/// per-item selection helpers keep the established mapping bit-identical.
pub const ForwardPipelines = struct {
    // Mesh Forward Pipelines
    pipeline_pbr_u16: sg.Pipeline = .{},
    pipeline_pbr_u32: sg.Pipeline = .{},
    pipeline_skinned_pbr_u16: sg.Pipeline = .{},
    pipeline_skinned_pbr_u32: sg.Pipeline = .{},
    pipeline_instanced_pbr_u16: sg.Pipeline = .{},
    pipeline_instanced_pbr_u32: sg.Pipeline = .{},

    // Transparent twin pipelines: same shaders/layouts as above, but with
    // alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA), depth test on and
    // depth write off. Selected per item when item.transparent is true.
    pipeline_pbr_blend_u16: sg.Pipeline = .{},
    pipeline_pbr_blend_u32: sg.Pipeline = .{},
    pipeline_skinned_pbr_blend_u16: sg.Pipeline = .{},
    pipeline_skinned_pbr_blend_u32: sg.Pipeline = .{},
    pipeline_instanced_pbr_blend_u16: sg.Pipeline = .{},
    pipeline_instanced_pbr_blend_u32: sg.Pipeline = .{},

    // Double-sided (cull-off) twins for every family; selected per item when
    // the material sets double_sided. Created in init, freed in deinit.
    ds_pipelines: scene_pipelines.DoubleSidedPipelines = .{},

    // Lazy pipeline cache for registered shader materials (built-in families
    // above are eager; shader materials pay pipeline creation on first use).
    shader_materials: ShaderMaterialCache = .{},

    // GPU-morph bind target for draws without morphs: binding/uniform state
    // persists across draws in sokol, so every forward draw must bind a
    // valid morph view. Non-GPU-morph meshes get this 1x1 zero RGBA32F
    // texture with the morph enable uniform off (draw.zig). The nearest
    // nonfiltering sampler matches the mesh delta sampling (RGBA32F is
    // unfilterable on Metal). Created in init, freed in deinit.
    default_morph_image: sg.Image = .{},
    default_morph_view: sg.View = .{},
    morph_sampler: sg.Sampler = .{},

    /// Complete target shape every pipeline in this set was built for.
    shape: TargetShape = .{},
    sample_count: i32 = 1,
    color_format: sg.PixelFormat = .RGBA16F,
    depth_format: sg.PixelFormat = .DEPTH,

    family_shaders: ?scene_pipelines.DoubleSidedSourceShaders = null,
    owns_shaders: bool = false,

    /// Builds every pipeline for an explicit target shape. Panics if a base pipeline fails to create.
    pub fn initForShape(target_shape: TargetShape) ForwardPipelines {
        const family_shaders = scene_pipelines.DoubleSidedSourceShaders{
            .pbr = sg.makeShader(pbr_shd.pbrShaderDesc(sg.queryBackend())),
            .instanced_pbr = sg.makeShader(inst_pbr_shd.instancedPbrShaderDesc(sg.queryBackend())),
            .skinned_pbr = sg.makeShader(skinned_pbr_shd.skinnedPbrShaderDesc(sg.queryBackend())),
        };
        var self = initWithShadersForShape(target_shape, family_shaders);
        self.owns_shaders = true;
        return self;
    }

    /// Builds every pipeline for sample count + color format (environment depth).
    pub fn init(sample_count: i32, color_format: sg.PixelFormat) ForwardPipelines {
        return initForShape(.{
            .sample_count = sample_count,
            .color_format = color_format,
            .depth_format = defaultDepthFormat(),
        });
    }

    /// Variant of init that borrows pre-compiled family shaders for an explicit target shape.
    pub fn initWithShadersForShape(target_shape: TargetShape, family_shaders: scene_pipelines.DoubleSidedSourceShaders) ForwardPipelines {
        const resolved = target_shape.resolveEnvironment();
        var self: ForwardPipelines = .{};
        self.shape = resolved;
        self.sample_count = resolved.sample_count;
        self.color_format = resolved.color_format;
        self.depth_format = resolved.depth_format;
        self.family_shaders = family_shaders;
        self.owns_shaders = false;

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

        const specs = [_]struct {
            shader: sg.Shader,
            family: scene_pipelines.PipelineFamily,
            opaque_u16: *sg.Pipeline,
            opaque_u32: *sg.Pipeline,
            blend_u16: *sg.Pipeline,
            blend_u32: *sg.Pipeline,
        }{
            .{
                .shader = family_shaders.pbr,
                .family = .pbr,
                .opaque_u16 = &self.pipeline_pbr_u16,
                .opaque_u32 = &self.pipeline_pbr_u32,
                .blend_u16 = &self.pipeline_pbr_blend_u16,
                .blend_u32 = &self.pipeline_pbr_blend_u32,
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
            var desc = forwardDescForShape(spec.shader, resolved);
            scene_pipelines.pipelineLayoutFor(spec.family, &desc);
            scene_pipelines.makePipelinePair(desc, spec.opaque_u16, spec.opaque_u32, spec.blend_u16, spec.blend_u32);
        }

        self.ds_pipelines.initFromShadersForShape(family_shaders, resolved);

        // The shape-aware shader-material cache: lazily filled pipelines
        // built for the same target shape as the eager sets above.
        self.shader_materials.shape = resolved;
        self.shader_materials.sample_count = resolved.sample_count;
        self.shader_materials.color_format = resolved.color_format;

        inline for (.{
            .{ .pipe = self.pipeline_pbr_u16, .msg = "pipeline_pbr_u16 failed to create!" },
            .{ .pipe = self.pipeline_pbr_u32, .msg = "pipeline_pbr_u32 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_u16, .msg = "pipeline_skinned_pbr_u16 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_u32, .msg = "pipeline_skinned_pbr_u32 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_u16, .msg = "pipeline_instanced_pbr_u16 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_u32, .msg = "pipeline_instanced_pbr_u32 failed to create!" },
            .{ .pipe = self.pipeline_pbr_blend_u16, .msg = "pipeline_pbr_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_pbr_blend_u32, .msg = "pipeline_pbr_blend_u32 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_blend_u16, .msg = "pipeline_skinned_pbr_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_skinned_pbr_blend_u32, .msg = "pipeline_skinned_pbr_blend_u32 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_blend_u16, .msg = "pipeline_instanced_pbr_blend_u16 failed to create!" },
            .{ .pipe = self.pipeline_instanced_pbr_blend_u32, .msg = "pipeline_instanced_pbr_blend_u32 failed to create!" },
        }) |entry| {
            if (entry.pipe.id == 0) @panic(entry.msg);
        }

        return self;
    }

    pub fn initWithShaders(sample_count: i32, color_format: sg.PixelFormat, family_shaders: scene_pipelines.DoubleSidedSourceShaders) ForwardPipelines {
        return initWithShadersForShape(.{
            .sample_count = sample_count,
            .color_format = color_format,
            .depth_format = defaultDepthFormat(),
        }, family_shaders);
    }

    pub fn deinit(self: *ForwardPipelines) void {
        sg.destroyImage(self.default_morph_image);
        sg.destroyView(self.default_morph_view);
        sg.destroySampler(self.morph_sampler);
        sg.destroyPipeline(self.pipeline_pbr_u16);
        sg.destroyPipeline(self.pipeline_pbr_u32);
        sg.destroyPipeline(self.pipeline_skinned_pbr_u16);
        sg.destroyPipeline(self.pipeline_skinned_pbr_u32);
        sg.destroyPipeline(self.pipeline_pbr_blend_u16);
        sg.destroyPipeline(self.pipeline_pbr_blend_u32);
        sg.destroyPipeline(self.pipeline_skinned_pbr_blend_u16);
        sg.destroyPipeline(self.pipeline_skinned_pbr_blend_u32);
        sg.destroyPipeline(self.pipeline_instanced_pbr_u16);
        sg.destroyPipeline(self.pipeline_instanced_pbr_u32);
        sg.destroyPipeline(self.pipeline_instanced_pbr_blend_u16);
        sg.destroyPipeline(self.pipeline_instanced_pbr_blend_u32);
        self.ds_pipelines.deinit();
        self.shader_materials.deinit();
        if (self.owns_shaders) {
            if (self.family_shaders) |fs| {
                if (fs.pbr.id != 0) sg.destroyShader(fs.pbr);
                if (fs.instanced_pbr.id != 0) sg.destroyShader(fs.instanced_pbr);
                if (fs.skinned_pbr.id != 0) sg.destroyShader(fs.skinned_pbr);
            }
        }
        self.family_shaders = null;
        self.owns_shaders = false;
    }

    // Selects the forward pipeline for a regular (non-instanced) queue item.
    // Transparent items resolve to the blend twins; opaque selection is
    // identical to the established mapping, so existing pipeline ids are untouched.
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
