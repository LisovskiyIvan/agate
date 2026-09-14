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
const shader_material = @import("../shader_material.zig");

/// Base descriptor funnel for every main-target pipeline (defined in
/// pipelines.zig; re-exported here because this file is the pipeline front
/// door). See the definition for the sample-count contract.
pub const forwardBaseDesc = scene_pipelines.forwardBaseDesc;

/// Cache-slot key mixing a shader-material registration key with the target
/// sample count, so 1x and MSAA pipeline sets for the same shader coexist in
/// one cache without colliding.
pub fn shaderMaterialSlotKey(key: u64, sample_count: i32) u64 {
    var h = std.hash.Wyhash.init(0x6d73_6161_2020_2020); // "msaa    "
    h.update(std.mem.asBytes(&key));
    h.update(std.mem.asBytes(&sample_count));
    return h.final();
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
    /// Main-target sample count the cached pipelines are built for; mixed
    /// into the slot key so 1x and MSAA variants coexist (see
    /// shaderMaterialSlotKey).
    sample_count: i32 = 1,

    pub fn lookup(self: *const ShaderMaterialCache, key: u64) ?*const ShaderMaterialSet {
        const slot_key = shaderMaterialSlotKey(key, self.sample_count);
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
        const slot_key = shaderMaterialSlotKey(key, self.sample_count);
        for (&self.slots) |*slot| {
            if (slot.key == 0 and slot.opaque_u16.id == 0 and slot.shader.id == 0) {
                const shader = entry.make_shader(sg.queryBackend());
                if (shader.id == 0) return null;
                slot.shader = shader;
                var desc = forwardBaseDesc(shader, self.sample_count);
                scene_pipelines.pipelineLayoutFor(switch (entry.base) {
                    .standard => .standard,
                    .pbr => .pbr,
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

    /// Main-target sample count every pipeline in this set was built with
    /// (1 = legacy set; > 1 via initSampled for the MSAA twin in Scene).
    sample_count: i32 = 1,

    family_shaders: ?scene_pipelines.DoubleSidedSourceShaders = null,
    owns_shaders: bool = false,

    /// Builds the default 1x pipeline set (GPU calls). Panics if a base
    /// pipeline fails to create — same loud failure as the legacy
    /// Scene.initPipelines.
    pub fn init() ForwardPipelines {
        return initSampled(1);
    }

    /// Builds every pipeline for a specific main-target sample count (GPU
    /// calls). Sokol requires pipeline.sample_count to equal the sample
    /// count of every attachment of the pass it is applied in, so Scene
    /// keeps one set per active target shape and never mixes them within a
    /// frame. Panics if a base pipeline fails to create.
    pub fn initSampled(sample_count: i32) ForwardPipelines {
        const family_shaders = scene_pipelines.DoubleSidedSourceShaders{
            .standard = sg.makeShader(shd.standardShaderDesc(sg.queryBackend())),
            .pbr = sg.makeShader(pbr_shd.pbrShaderDesc(sg.queryBackend())),
            .instanced = sg.makeShader(inst_shd.instancedShaderDesc(sg.queryBackend())),
            .instanced_pbr = sg.makeShader(inst_pbr_shd.instancedPbrShaderDesc(sg.queryBackend())),
            .skinned_pbr = sg.makeShader(skinned_pbr_shd.skinnedPbrShaderDesc(sg.queryBackend())),
        };
        var self = initSampledWithShaders(sample_count, family_shaders);
        self.owns_shaders = true;
        return self;
    }

    /// Variant of initSampled that borrows pre-compiled family shaders (e.g. from
    /// the 1x ForwardPipelines in Scene) instead of compiling new ones.
    pub fn initSampledWithShaders(sample_count: i32, family_shaders: scene_pipelines.DoubleSidedSourceShaders) ForwardPipelines {
        var self: ForwardPipelines = .{};
        self.sample_count = sample_count;
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
            var desc = forwardBaseDesc(spec.shader, sample_count);
            scene_pipelines.pipelineLayoutFor(spec.family, &desc);
            scene_pipelines.makePipelinePair(desc, spec.opaque_u16, spec.opaque_u32, spec.blend_u16, spec.blend_u32);
        }

        self.ds_pipelines.initFromShaders(family_shaders, sample_count);

        // The sample-aware shader-material cache: lazily filled pipelines
        // built for the same target shape as the eager sets above.
        self.shader_materials.sample_count = sample_count;

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
        self.shader_materials.deinit();
        if (self.owns_shaders) {
            if (self.family_shaders) |fs| {
                if (fs.standard.id != 0) sg.destroyShader(fs.standard);
                if (fs.pbr.id != 0) sg.destroyShader(fs.pbr);
                if (fs.instanced.id != 0) sg.destroyShader(fs.instanced);
                if (fs.instanced_pbr.id != 0) sg.destroyShader(fs.instanced_pbr);
                if (fs.skinned_pbr.id != 0) sg.destroyShader(fs.skinned_pbr);
            }
        }
        self.family_shaders = null;
        self.owns_shaders = false;
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

test "ShaderMaterialSet.pipelineFor mirrors the built-in selection contract" {
    const set = ShaderMaterialSet{
        .key = 42,
        .opaque_u16 = .{ .id = 101 },
        .opaque_u32 = .{ .id = 102 },
        .blend_u16 = .{ .id = 103 },
        .blend_u32 = .{ .id = 104 },
        .ds_opaque_u16 = .{ .id = 111 },
        .ds_opaque_u32 = .{ .id = 112 },
        .ds_blend_u16 = .{ .id = 113 },
        .ds_blend_u32 = .{ .id = 114 },
    };
    try std.testing.expectEqual(@as(u32, 101), set.pipelineFor(false, false, false));
    try std.testing.expectEqual(@as(u32, 102), set.pipelineFor(false, true, false));
    try std.testing.expectEqual(@as(u32, 103), set.pipelineFor(true, false, false));
    try std.testing.expectEqual(@as(u32, 104), set.pipelineFor(true, true, false));
    // Double-sided resolves to the cull-off twins.
    try std.testing.expectEqual(@as(u32, 111), set.pipelineFor(false, false, true));
    try std.testing.expectEqual(@as(u32, 114), set.pipelineFor(true, true, true));

    // Missing cull-off twin (id 0) falls back to the regular pipeline.
    var partial = ShaderMaterialSet{
        .opaque_u16 = .{ .id = 201 },
        .blend_u32 = .{ .id = 204 },
    };
    try std.testing.expectEqual(@as(u32, 201), partial.pipelineFor(false, false, true));
    try std.testing.expectEqual(@as(u32, 204), partial.pipelineFor(true, true, true));
    _ = &partial;
}

test "ShaderMaterialCache lookup is key-based and miss-safe without GPU" {
    var cache = ShaderMaterialCache{};
    // No GPU resources were created: every lookup misses.
    try std.testing.expect(cache.lookup(0) == null);
    try std.testing.expect(cache.lookup(shader_material.keyForName("ramp_wave")) == null);
    // entryForKey resolves the registration even though the cache is cold.
    try std.testing.expect(shader_material.entryForKey(shader_material.keyForName("ramp_wave")) != null);
    try std.testing.expect(shader_material.entryForKey(shader_material.keyForName("nope")) == null);
}

// --- MSAA pipeline-table consistency (GPU-free contracts). Visual AA
// quality cannot be unit-tested; that verification is the agate smoke run
// (`agate --frames 120 --msaa 4` must complete without sokol validation
// errors, the loudest failure mode being a pipeline/attachment sample-count
// mismatch).

/// Number of sg.Pipeline-typed fields of a struct, via comptime reflection.
fn countPipelineFields(comptime T: type) usize {
    var n: usize = 0;
    for (@typeInfo(T).@"struct".fields) |f| {
        if (f.type == sg.Pipeline) n += 1;
    }
    return n;
}

test "pipeline tables cover every pipeline field (sample-count twins stay in sync)" {
    // 5 families x opaque/blend x u16/u32. If a new pipeline field is added
    // to ForwardPipelines, initSampled's panic list and deinit must grow
    // with it — the count change fails this test and forces the review.
    const n_forward = comptime countPipelineFields(ForwardPipelines);
    try std.testing.expectEqual(@as(usize, 20), n_forward);
    // 5 families x opaque/blend x u16/u32 (cull-off twins).
    const n_ds = comptime countPipelineFields(scene_pipelines.DoubleSidedPipelines);
    try std.testing.expectEqual(@as(usize, 20), n_ds);
    // opaque/blend x u16/u32 x regular/ds per registered shader material.
    const n_shader_mat = comptime countPipelineFields(ShaderMaterialSet);
    try std.testing.expectEqual(@as(usize, 8), n_shader_mat);
}

test "forwardBaseDesc funnels the sample count into the pipeline descriptor" {
    const desc = forwardBaseDesc(.{}, 4);
    try std.testing.expectEqual(@as(i32, 4), desc.sample_count);
    try std.testing.expectEqual(sg.IndexType.UINT16, desc.index_type);
    try std.testing.expect(desc.depth.compare == .LESS_EQUAL);
    try std.testing.expect(desc.depth.write_enabled);
    try std.testing.expect(desc.cull_mode == .BACK);
    // Default stays the legacy 1x shape.
    try std.testing.expectEqual(@as(i32, 1), forwardBaseDesc(.{}, 1).sample_count);
}

test "shaderMaterialSlotKey separates sample-count variants and is stable" {
    const key = shader_material.keyForName("ramp_wave");
    const k1 = shaderMaterialSlotKey(key, 1);
    const k4 = shaderMaterialSlotKey(key, 4);
    try std.testing.expect(k1 != k4);
    try std.testing.expectEqual(k1, shaderMaterialSlotKey(key, 1));
    try std.testing.expectEqual(k4, shaderMaterialSlotKey(key, 4));
    // Different registration keys stay distinct within one sample count.
    try std.testing.expect(k1 != shaderMaterialSlotKey(key + 1, 1));
}

test "sampled ForwardPipelines default field keeps the legacy shape" {
    const fw = ForwardPipelines{};
    try std.testing.expectEqual(@as(i32, 1), fw.sample_count);
    try std.testing.expectEqual(@as(i32, 1), fw.shader_materials.sample_count);
}
