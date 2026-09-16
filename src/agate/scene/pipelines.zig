const sokol = @import("sokol");
const sg = sokol.gfx;
const shd = @import("shader");
const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_shd = @import("instanced_shader");
const inst_pbr_shd = @import("instanced_pbr_shader");

const Vertex = @import("../mesh.zig").Vertex;
const Mat4 = @import("math").Mat4;
const render_queue = @import("render_queue.zig");

// One shader + one vertex layout feeds an opaque u16/u32 pair plus its
// transparent blend twins. Only initPipelines uses this table.
pub const PipelineFamily = enum { standard, pbr, instanced, skinned_pbr, instanced_pbr };

// Single funnel for the base descriptor of every main-target pipeline
// (built-in families, double-sided twins, shader-material sets): the base
// depth/cull state plus the target sample count. Sokol validation rejects
// sg_apply_pipeline when pipeline.sample_count differs from any attachment
// image of the current pass (color AND depth), so every pipeline that can
// draw into the main target must be built through here with the same count
// as the target (scene/msaa.zig decides that count; the 1x set and the
// sampled twin sets in Scene must never be mixed within a frame).
pub fn forwardBaseDesc(shader: sg.Shader, sample_count: i32) sg.PipelineDesc {
    return .{
        .shader = shader,
        .index_type = .UINT16,
        .depth = .{
            .compare = .LESS_EQUAL,
            .write_enabled = true,
        },
        .cull_mode = .BACK,
        .face_winding = .CCW,
        .sample_count = sample_count,
    };
}

// Fills the vertex layout for a family exactly as the legacy hand-written
// descs did (same buffers, attr slots, formats, offsets). Any shader-side
// layout risk lives here: attr changes must mirror the matching *.glsl.
pub fn pipelineLayoutFor(family: PipelineFamily, desc: *sg.PipelineDesc) void {
    switch (family) {
        .standard => {
            desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
            desc.layout.attrs[shd.ATTR_standard_position] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
            desc.layout.attrs[shd.ATTR_standard_normal] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
            desc.layout.attrs[shd.ATTR_standard_color0] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
            desc.layout.attrs[shd.ATTR_standard_texcoord0] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };
        },
        .pbr => {
            desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
            desc.layout.attrs[pbr_shd.ATTR_pbr_position] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
            desc.layout.attrs[pbr_shd.ATTR_pbr_normal] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
            desc.layout.attrs[pbr_shd.ATTR_pbr_color0] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
            desc.layout.attrs[pbr_shd.ATTR_pbr_texcoord0] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };
            desc.layout.attrs[pbr_shd.ATTR_pbr_tangent] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "tangent") };
        },
        .instanced => {
            desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
            desc.layout.attrs[inst_shd.ATTR_instanced_position] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
            desc.layout.attrs[inst_shd.ATTR_instanced_normal] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
            desc.layout.attrs[inst_shd.ATTR_instanced_color0] = .{ .buffer_index = 0, .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
            desc.layout.attrs[inst_shd.ATTR_instanced_texcoord0] = .{ .buffer_index = 0, .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };

            desc.layout.buffers[1] = .{
                .step_func = .PER_INSTANCE,
                .step_rate = 1,
                .stride = @sizeOf(Mat4),
            };
            desc.layout.attrs[inst_shd.ATTR_instanced_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
            desc.layout.attrs[inst_shd.ATTR_instanced_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
            desc.layout.attrs[inst_shd.ATTR_instanced_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
            desc.layout.attrs[inst_shd.ATTR_instanced_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };
        },
        .instanced_pbr => {
            desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_position] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_normal] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_tangent] = .{ .buffer_index = 0, .format = .FLOAT4, .offset = @offsetOf(Vertex, "tangent") };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_color0] = .{ .buffer_index = 0, .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_texcoord0] = .{ .buffer_index = 0, .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };

            desc.layout.buffers[1] = .{
                .step_func = .PER_INSTANCE,
                .step_rate = 1,
                .stride = @sizeOf(Mat4),
            };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };
        },
        .skinned_pbr => {
            desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_position] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_normal] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_color0] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_texcoord0] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_tangent] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "tangent") };
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_joints] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "joints") };
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_weights] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "weights") };
        },
    }
}

// Creates an opaque u16/u32 pair plus transparent blend twins from one
// base desc. Order and index_type switches match the legacy code.
pub fn makePipelinePair(base: sg.PipelineDesc, opaque_u16: *sg.Pipeline, opaque_u32: *sg.Pipeline, blend_u16: *sg.Pipeline, blend_u32: *sg.Pipeline) void {
    var desc = base;
    desc.index_type = .UINT16;
    opaque_u16.* = sg.makePipeline(desc);
    desc.index_type = .UINT32;
    opaque_u32.* = sg.makePipeline(desc);
    desc.index_type = .UINT16;
    blend_u16.* = sg.makePipeline(render_queue.blendDescFor(desc));
    desc.index_type = .UINT32;
    blend_u32.* = sg.makePipeline(render_queue.blendDescFor(desc));
}

// Derives a double-sided (face culling disabled) desc from an opaque base
// desc: same shader/layout/depth/blend state, only cull_mode forced to
// .NONE. Pure function (no GPU calls). Composes with blendDescFor: blend
// twins keep .NONE because blendDescFor never touches cull_mode.
pub fn cullOffDescFor(base: sg.PipelineDesc) sg.PipelineDesc {
    var desc = base;
    desc.cull_mode = .NONE;
    return desc;
}

// Creates the double-sided twins of one family: cull-off opaque u16/u32
// pair plus cull-off transparent blend twins. Same order/index_type
// switches as makePipelinePair (GPU calls).
pub fn makeCullOffPair(base: sg.PipelineDesc, opaque_u16: *sg.Pipeline, opaque_u32: *sg.Pipeline, blend_u16: *sg.Pipeline, blend_u32: *sg.Pipeline) void {
    makePipelinePair(cullOffDescFor(base), opaque_u16, opaque_u32, blend_u16, blend_u32);
}

// Family shaders needed to build the double-sided set. The Scene already
// creates these shader handles in initPipelines; it can retain them and
// pass them here (or extend the specs table to also feed initFromShaders).
pub const DoubleSidedSourceShaders = struct {
    standard: sg.Shader,
    pbr: sg.Shader,
    instanced: sg.Shader,
    instanced_pbr: sg.Shader,
    skinned_pbr: sg.Shader,
};

// Double-sided (cull-off) pipeline twins for every family: opaque u16/u32
// plus blend u16/u32, i.e. 20 pipelines total (5 families x opaque/blend x
// u16/u32). Instanced twins are included: same table, no extra code.
// Intended embedding (Scene gains ONE field, init/deinit forwarded):
//   ds_pipelines: DoubleSidedPipelines = .{},
//   self.ds_pipelines.initFromShaders(.{ .standard = std_sh, ... });
//   self.ds_pipelines.deinit();
// NOTE: back-face lighting still uses geometric normals (no
// gl_FrontFacing flip in the shaders); back faces may shade dark. This
// matches the minimal cull-off scope.
pub const DoubleSidedPipelines = struct {
    standard_u16: sg.Pipeline = .{},
    standard_u32: sg.Pipeline = .{},
    standard_blend_u16: sg.Pipeline = .{},
    standard_blend_u32: sg.Pipeline = .{},
    pbr_u16: sg.Pipeline = .{},
    pbr_u32: sg.Pipeline = .{},
    pbr_blend_u16: sg.Pipeline = .{},
    pbr_blend_u32: sg.Pipeline = .{},
    skinned_pbr_u16: sg.Pipeline = .{},
    skinned_pbr_u32: sg.Pipeline = .{},
    skinned_pbr_blend_u16: sg.Pipeline = .{},
    skinned_pbr_blend_u32: sg.Pipeline = .{},
    instanced_u16: sg.Pipeline = .{},
    instanced_u32: sg.Pipeline = .{},
    instanced_blend_u16: sg.Pipeline = .{},
    instanced_blend_u32: sg.Pipeline = .{},
    instanced_pbr_u16: sg.Pipeline = .{},
    instanced_pbr_u32: sg.Pipeline = .{},
    instanced_pbr_blend_u16: sg.Pipeline = .{},
    instanced_pbr_blend_u32: sg.Pipeline = .{},

    // Builds all 20 cull-off twins from the family shaders (GPU calls).
    // Base descs mirror Scene.initPipelines (depth LESS_EQUAL/write on,
    // BACK cull, CCW winding) plus the main-target sample count (sokol
    // requires pipelines to match the attachment sample count they draw
    // into); makeCullOffPair forces cull off.
    pub fn initFromShaders(self: *DoubleSidedPipelines, shaders: DoubleSidedSourceShaders, sample_count: i32) void {
        const specs = [_]struct {
            shader: sg.Shader,
            family: PipelineFamily,
            opaque_u16: *sg.Pipeline,
            opaque_u32: *sg.Pipeline,
            blend_u16: *sg.Pipeline,
            blend_u32: *sg.Pipeline,
        }{
            .{
                .shader = shaders.standard,
                .family = .standard,
                .opaque_u16 = &self.standard_u16,
                .opaque_u32 = &self.standard_u32,
                .blend_u16 = &self.standard_blend_u16,
                .blend_u32 = &self.standard_blend_u32,
            },
            .{
                .shader = shaders.pbr,
                .family = .pbr,
                .opaque_u16 = &self.pbr_u16,
                .opaque_u32 = &self.pbr_u32,
                .blend_u16 = &self.pbr_blend_u16,
                .blend_u32 = &self.pbr_blend_u32,
            },
            .{
                .shader = shaders.instanced,
                .family = .instanced,
                .opaque_u16 = &self.instanced_u16,
                .opaque_u32 = &self.instanced_u32,
                .blend_u16 = &self.instanced_blend_u16,
                .blend_u32 = &self.instanced_blend_u32,
            },
            .{
                .shader = shaders.instanced_pbr,
                .family = .instanced_pbr,
                .opaque_u16 = &self.instanced_pbr_u16,
                .opaque_u32 = &self.instanced_pbr_u32,
                .blend_u16 = &self.instanced_pbr_blend_u16,
                .blend_u32 = &self.instanced_pbr_blend_u32,
            },
            .{
                .shader = shaders.skinned_pbr,
                .family = .skinned_pbr,
                .opaque_u16 = &self.skinned_pbr_u16,
                .opaque_u32 = &self.skinned_pbr_u32,
                .blend_u16 = &self.skinned_pbr_blend_u16,
                .blend_u32 = &self.skinned_pbr_blend_u32,
            },
        };

        for (specs) |spec| {
            var desc = forwardBaseDesc(spec.shader, sample_count);
            pipelineLayoutFor(spec.family, &desc);
            makeCullOffPair(desc, spec.opaque_u16, spec.opaque_u32, spec.blend_u16, spec.blend_u32);
        }
    }

    pub fn deinit(self: *DoubleSidedPipelines) void {
        inline for (.{
            &self.standard_u16,            &self.standard_u32,
            &self.standard_blend_u16,      &self.standard_blend_u32,
            &self.pbr_u16,                 &self.pbr_u32,
            &self.pbr_blend_u16,           &self.pbr_blend_u32,
            &self.skinned_pbr_u16,         &self.skinned_pbr_u32,
            &self.skinned_pbr_blend_u16,   &self.skinned_pbr_blend_u32,
            &self.instanced_u16,           &self.instanced_u32,
            &self.instanced_blend_u16,     &self.instanced_blend_u32,
            &self.instanced_pbr_u16,       &self.instanced_pbr_u32,
            &self.instanced_pbr_blend_u16, &self.instanced_pbr_blend_u32,
        }) |pipe| {
            if (pipe.*.id != 0) sg.destroyPipeline(pipe.*);
            pipe.* = .{};
        }
    }
};

// Selects the forward pipeline for a regular (non-instanced) queue item.
// Transparent items resolve to the blend twins; opaque selection is
// identical to the legacy logic, so existing pipeline ids are untouched.
// Double-sided items (item.double_sided OR mesh material double_sided)
// resolve to the cull-off twins in scene.ds_pipelines when that set is
// present and the twin id is non-zero, otherwise they fall back to the
// regular pipelines. Scenes without a ds_pipelines field (legacy) and
// items without a double_sided field behave exactly as before.
// Cutout items are opaque (transparent == false) and resolve to opaque ids.
// `scene` is generic (anytype) to avoid a scene.zig import cycle; it must
// expose the 12 pipeline_* fields. `item` must expose .mesh/.is_pbr/.transparent.
pub fn pipelineForRegularItem(scene: anytype, item: anytype) u32 {
    const is_u32 = blk: {
        if (@hasField(@TypeOf(item), "mesh")) {
            if (@typeInfo(@TypeOf(item.mesh)) == .optional) {
                if (item.mesh) |m| break :blk (m.index_type == .UINT32);
            } else {
                break :blk (item.mesh.index_type == .UINT32);
            }
        }
        if (@hasField(@TypeOf(item), "index_type") and item.index_type == .UINT32) break :blk true;
        if (@hasField(@TypeOf(item), "is_u32") and item.is_u32) break :blk true;
        break :blk false;
    };

    const is_skinned = blk: {
        if (@hasField(@TypeOf(item), "mesh")) {
            if (@typeInfo(@TypeOf(item.mesh)) == .optional) {
                if (item.mesh) |m| break :blk (m.skeleton != null);
            } else {
                break :blk (item.mesh.skeleton != null);
            }
        }
        if (@hasField(@TypeOf(item), "is_skinned") and item.is_skinned) break :blk true;
        if (@hasField(@TypeOf(item), "skin_matrices") and item.skin_matrices != null) break :blk true;
        break :blk false;
    };

    const ds = sceneDoubleSided(scene);
    const want_ds = if (ds) |_| itemDoubleSided(item) else false;
    if (item.is_pbr) {
        if (is_skinned) {
            if (item.transparent) {
                if (want_ds) {
                    const id = if (is_u32) ds.?.skinned_pbr_blend_u32.id else ds.?.skinned_pbr_blend_u16.id;
                    if (id != 0) return id;
                }
                return if (is_u32) scene.pipeline_skinned_pbr_blend_u32.id else scene.pipeline_skinned_pbr_blend_u16.id;
            }
            if (want_ds) {
                const id = if (is_u32) ds.?.skinned_pbr_u32.id else ds.?.skinned_pbr_u16.id;
                if (id != 0) return id;
            }
            return if (is_u32) scene.pipeline_skinned_pbr_u32.id else scene.pipeline_skinned_pbr_u16.id;
        }
        if (item.transparent) {
            if (want_ds) {
                const id = if (is_u32) ds.?.pbr_blend_u32.id else ds.?.pbr_blend_u16.id;
                if (id != 0) return id;
            }
            return if (is_u32) scene.pipeline_pbr_blend_u32.id else scene.pipeline_pbr_blend_u16.id;
        }
        if (want_ds) {
            const id = if (is_u32) ds.?.pbr_u32.id else ds.?.pbr_u16.id;
            if (id != 0) return id;
        }
        return if (is_u32) scene.pipeline_pbr_u32.id else scene.pipeline_pbr_u16.id;
    }
    if (item.transparent) {
        if (want_ds) {
            const id = if (is_u32) ds.?.standard_blend_u32.id else ds.?.standard_blend_u16.id;
            if (id != 0) return id;
        }
        return if (is_u32) scene.pipeline_blend_u32.id else scene.pipeline_blend_u16.id;
    }
    if (want_ds) {
        const id = if (is_u32) ds.?.standard_u32.id else ds.?.standard_u16.id;
        if (id != 0) return id;
    }
    return if (is_u32) scene.pipeline_u32.id else scene.pipeline_u16.id;
}

test "cullOffDescFor disables culling and preserves everything else" {
    const std = @import("std");
    const base = sg.PipelineDesc{
        .shader = .{},
        .index_type = .UINT32,
        .depth = .{ .compare = .LESS_EQUAL, .write_enabled = true },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    const cull_off = cullOffDescFor(base);
    try std.testing.expect(cull_off.cull_mode == .NONE);
    try std.testing.expect(cull_off.depth.write_enabled);
    try std.testing.expect(cull_off.depth.compare == .LESS_EQUAL);
    try std.testing.expect(cull_off.index_type == .UINT32);
    try std.testing.expect(cull_off.face_winding == .CCW);
    // Pure function: the base desc is left untouched.
    try std.testing.expect(base.cull_mode == .BACK);

    // Composes with blendDescFor: blend twins stay cull-off.
    const blend_off = render_queue.blendDescFor(cull_off);
    try std.testing.expect(blend_off.cull_mode == .NONE);
    try std.testing.expect(blend_off.colors[0].blend.enabled);
    try std.testing.expect(!blend_off.depth.write_enabled);
}

// --- GPU-free selection tests use lightweight mocks (no Mesh import:
// mesh.zig -> scene.zig -> pipelines.zig would be an import cycle). ---

const TestMaterial = @import("../material.zig");
const TestFakeSkel = struct {};

const TestMesh = struct {
    index_type: sg.IndexType = .UINT16,
    skeleton: ?*TestFakeSkel = null,
    material: ?TestMaterial.Material = null,
};

const TestItem = struct {
    mesh: *TestMesh,
    is_pbr: bool = false,
    transparent: bool = false,
    double_sided: bool = false,
};

fn testLegacyScene() struct {
    pipeline_u16: sg.Pipeline,
    pipeline_u32: sg.Pipeline,
    pipeline_pbr_u16: sg.Pipeline,
    pipeline_pbr_u32: sg.Pipeline,
    pipeline_skinned_pbr_u16: sg.Pipeline,
    pipeline_skinned_pbr_u32: sg.Pipeline,
    pipeline_instanced_u16: sg.Pipeline,
    pipeline_instanced_u32: sg.Pipeline,
    pipeline_instanced_pbr_u16: sg.Pipeline,
    pipeline_instanced_pbr_u32: sg.Pipeline,
    pipeline_blend_u16: sg.Pipeline,
    pipeline_blend_u32: sg.Pipeline,
    pipeline_pbr_blend_u16: sg.Pipeline,
    pipeline_pbr_blend_u32: sg.Pipeline,
    pipeline_skinned_pbr_blend_u16: sg.Pipeline,
    pipeline_skinned_pbr_blend_u32: sg.Pipeline,
    pipeline_instanced_blend_u16: sg.Pipeline,
    pipeline_instanced_blend_u32: sg.Pipeline,
    pipeline_instanced_pbr_blend_u16: sg.Pipeline,
    pipeline_instanced_pbr_blend_u32: sg.Pipeline,
} {
    return .{
        .pipeline_u16 = .{ .id = 11 },
        .pipeline_u32 = .{ .id = 12 },
        .pipeline_pbr_u16 = .{ .id = 21 },
        .pipeline_pbr_u32 = .{ .id = 22 },
        .pipeline_skinned_pbr_u16 = .{ .id = 31 },
        .pipeline_skinned_pbr_u32 = .{ .id = 32 },
        .pipeline_instanced_u16 = .{ .id = 41 },
        .pipeline_instanced_u32 = .{ .id = 42 },
        .pipeline_instanced_pbr_u16 = .{ .id = 51 },
        .pipeline_instanced_pbr_u32 = .{ .id = 52 },
        .pipeline_blend_u16 = .{ .id = 13 },
        .pipeline_blend_u32 = .{ .id = 14 },
        .pipeline_pbr_blend_u16 = .{ .id = 23 },
        .pipeline_pbr_blend_u32 = .{ .id = 24 },
        .pipeline_skinned_pbr_blend_u16 = .{ .id = 33 },
        .pipeline_skinned_pbr_blend_u32 = .{ .id = 34 },
        .pipeline_instanced_blend_u16 = .{ .id = 43 },
        .pipeline_instanced_blend_u32 = .{ .id = 44 },
        .pipeline_instanced_pbr_blend_u16 = .{ .id = 53 },
        .pipeline_instanced_pbr_blend_u32 = .{ .id = 54 },
    };
}

const TestDsScene = struct {
    pipeline_u16: sg.Pipeline,
    pipeline_u32: sg.Pipeline,
    pipeline_pbr_u16: sg.Pipeline,
    pipeline_pbr_u32: sg.Pipeline,
    pipeline_skinned_pbr_u16: sg.Pipeline,
    pipeline_skinned_pbr_u32: sg.Pipeline,
    pipeline_instanced_u16: sg.Pipeline,
    pipeline_instanced_u32: sg.Pipeline,
    pipeline_instanced_pbr_u16: sg.Pipeline,
    pipeline_instanced_pbr_u32: sg.Pipeline,
    pipeline_blend_u16: sg.Pipeline,
    pipeline_blend_u32: sg.Pipeline,
    pipeline_pbr_blend_u16: sg.Pipeline,
    pipeline_pbr_blend_u32: sg.Pipeline,
    pipeline_skinned_pbr_blend_u16: sg.Pipeline,
    pipeline_skinned_pbr_blend_u32: sg.Pipeline,
    pipeline_instanced_blend_u16: sg.Pipeline,
    pipeline_instanced_blend_u32: sg.Pipeline,
    pipeline_instanced_pbr_blend_u16: sg.Pipeline,
    pipeline_instanced_pbr_blend_u32: sg.Pipeline,
    ds_pipelines: DoubleSidedPipelines,
};

test "legacy scenes keep existing pipeline ids; cutout resolves opaque" {
    const std = @import("std");
    const scene = testLegacyScene();
    var mesh = TestMesh{};
    var skel = TestFakeSkel{};

    // Opaque/transparent matrix, u16 + u32, standard + pbr + skinned.
    var item = TestItem{ .mesh = &mesh };
    try std.testing.expectEqual(@as(u32, 11), pipelineForRegularItem(scene, item));
    item.transparent = true;
    try std.testing.expectEqual(@as(u32, 13), pipelineForRegularItem(scene, item));
    item.transparent = false;
    mesh.index_type = .UINT32;
    try std.testing.expectEqual(@as(u32, 12), pipelineForRegularItem(scene, item));

    item.is_pbr = true;
    mesh.index_type = .UINT16;
    try std.testing.expectEqual(@as(u32, 21), pipelineForRegularItem(scene, item));
    item.transparent = true;
    try std.testing.expectEqual(@as(u32, 23), pipelineForRegularItem(scene, item));

    mesh.skeleton = &skel;
    item.transparent = false;
    try std.testing.expectEqual(@as(u32, 31), pipelineForRegularItem(scene, item));
    item.transparent = true;
    try std.testing.expectEqual(@as(u32, 33), pipelineForRegularItem(scene, item));

    // Cutout material classifies opaque (transparent == false) and must
    // resolve to the opaque pipeline id, never a blend twin.
    var cutout_mat = TestMaterial.StandardMaterial.init("c");
    cutout_mat.alpha_mode = .cutout;
    mesh.skeleton = null;
    mesh.material = .{ .standard = &cutout_mat };
    item = TestItem{ .mesh = &mesh, .transparent = false };
    try std.testing.expectEqual(@as(u32, 11), pipelineForRegularItem(scene, item));
    mesh.index_type = .UINT32;
    try std.testing.expectEqual(@as(u32, 12), pipelineForRegularItem(scene, item));
    item.is_pbr = true;
    mesh.index_type = .UINT16;
    // Note: std cutout material on a pbr item still picks the pbr pipe;
    // the point is opaque (not blend) selection.
    try std.testing.expectEqual(@as(u32, 21), pipelineForRegularItem(scene, item));
}

test "double-sided items select cull-off twins, with regular fallback" {
    const std = @import("std");
    const legacy = testLegacyScene();
    const scene = TestDsScene{
        .pipeline_u16 = legacy.pipeline_u16,
        .pipeline_u32 = legacy.pipeline_u32,
        .pipeline_pbr_u16 = legacy.pipeline_pbr_u16,
        .pipeline_pbr_u32 = legacy.pipeline_pbr_u32,
        .pipeline_skinned_pbr_u16 = legacy.pipeline_skinned_pbr_u16,
        .pipeline_skinned_pbr_u32 = legacy.pipeline_skinned_pbr_u32,
        .pipeline_instanced_u16 = legacy.pipeline_instanced_u16,
        .pipeline_instanced_u32 = legacy.pipeline_instanced_u32,
        .pipeline_instanced_pbr_u16 = legacy.pipeline_instanced_pbr_u16,
        .pipeline_instanced_pbr_u32 = legacy.pipeline_instanced_pbr_u32,
        .pipeline_blend_u16 = legacy.pipeline_blend_u16,
        .pipeline_blend_u32 = legacy.pipeline_blend_u32,
        .pipeline_pbr_blend_u16 = legacy.pipeline_pbr_blend_u16,
        .pipeline_pbr_blend_u32 = legacy.pipeline_pbr_blend_u32,
        .pipeline_skinned_pbr_blend_u16 = legacy.pipeline_skinned_pbr_blend_u16,
        .pipeline_skinned_pbr_blend_u32 = legacy.pipeline_skinned_pbr_blend_u32,
        .pipeline_instanced_blend_u16 = legacy.pipeline_instanced_blend_u16,
        .pipeline_instanced_blend_u32 = legacy.pipeline_instanced_blend_u32,
        .pipeline_instanced_pbr_blend_u16 = legacy.pipeline_instanced_pbr_blend_u16,
        .pipeline_instanced_pbr_blend_u32 = legacy.pipeline_instanced_pbr_blend_u32,
        .ds_pipelines = .{
            .standard_u16 = .{ .id = 111 },
            .standard_u32 = .{ .id = 112 },
            .standard_blend_u16 = .{ .id = 113 },
            .pbr_u16 = .{ .id = 121 },
            .pbr_blend_u32 = .{ .id = 124 },
            .instanced_u16 = .{ .id = 141 },
            .instanced_pbr_u16 = .{ .id = 151 },
        },
    };
    var mesh = TestMesh{};
    var mat = TestMaterial.StandardMaterial.init("ds");
    mat.double_sided = true;

    // Explicit item flag selects the cull-off twin.
    var item = TestItem{ .mesh = &mesh, .double_sided = true };
    try std.testing.expectEqual(@as(u32, 111), pipelineForRegularItem(scene, item));
    mesh.index_type = .UINT32;
    try std.testing.expectEqual(@as(u32, 112), pipelineForRegularItem(scene, item));
    mesh.index_type = .UINT16;

    // Blend + double-sided selects the cull-off blend twin.
    item.transparent = true;
    try std.testing.expectEqual(@as(u32, 113), pipelineForRegularItem(scene, item));

    // Missing twin (id 0) falls back to the regular pipeline.
    mesh.index_type = .UINT32;
    try std.testing.expectEqual(@as(u32, 14), pipelineForRegularItem(scene, item));
    mesh.index_type = .UINT16;
    item.transparent = false;

    // Material-derived: flag unset, but mesh material is double-sided.
    mesh.material = .{ .standard = &mat };
    item = TestItem{ .mesh = &mesh };
    try std.testing.expectEqual(@as(u32, 111), pipelineForRegularItem(scene, item));

    // Single-sided items keep regular ids even when the ds set exists.
    mat.double_sided = false;
    try std.testing.expectEqual(@as(u32, 11), pipelineForRegularItem(scene, item));
    item.double_sided = false;
    mesh.material = null;
    try std.testing.expectEqual(@as(u32, 11), pipelineForRegularItem(scene, item));

    // PBR family honors the same contract.
    item.is_pbr = true;
    item.double_sided = true;
    try std.testing.expectEqual(@as(u32, 121), pipelineForRegularItem(scene, item));
    item.transparent = true;
    mesh.index_type = .UINT32;
    try std.testing.expectEqual(@as(u32, 124), pipelineForRegularItem(scene, item));

    // Instanced helper: ds twin, blend twin fallback, legacy scene.
    try std.testing.expectEqual(@as(u32, 141), pipelineForInstancedMesh(scene, false, false, false, true));
    try std.testing.expectEqual(@as(u32, 43), pipelineForInstancedMesh(scene, false, true, false, true));
    try std.testing.expectEqual(@as(u32, 41), pipelineForInstancedMesh(scene, false, false, false, false));
    try std.testing.expectEqual(@as(u32, 44), pipelineForInstancedMesh(scene, false, true, true, false));
    try std.testing.expectEqual(@as(u32, 41), pipelineForInstancedMesh(legacy, false, false, false, true));
    try std.testing.expectEqual(@as(u32, 43), pipelineForInstancedMesh(legacy, false, true, false, true));

    // Instanced PBR:
    try std.testing.expectEqual(@as(u32, 151), pipelineForInstancedMesh(scene, true, false, false, true));
    try std.testing.expectEqual(@as(u32, 51), pipelineForInstancedMesh(scene, true, false, false, false));
    try std.testing.expectEqual(@as(u32, 54), pipelineForInstancedMesh(scene, true, true, true, false));
    try std.testing.expectEqual(@as(u32, 51), pipelineForInstancedMesh(legacy, true, false, false, true));
}

// Selects the instanced pipeline. Same double-sided contract as
// pipelineForRegularItem: cull-off twin when requested and available,
// regular pipeline otherwise (legacy scenes behave exactly as before).
// `scene` must expose the pipeline_instanced_* fields.
pub fn pipelineForInstancedMesh(scene: anytype, is_pbr: bool, transparent: bool, is_u32: bool, double_sided: bool) u32 {
    if (sceneDoubleSided(scene)) |ds| {
        if (double_sided) {
            if (is_pbr) {
                const id = if (transparent)
                    (if (is_u32) ds.instanced_pbr_blend_u32.id else ds.instanced_pbr_blend_u16.id)
                else
                    (if (is_u32) ds.instanced_pbr_u32.id else ds.instanced_pbr_u16.id);
                if (id != 0) return id;
            } else {
                const id = if (transparent)
                    (if (is_u32) ds.instanced_blend_u32.id else ds.instanced_blend_u16.id)
                else
                    (if (is_u32) ds.instanced_u32.id else ds.instanced_u16.id);
                if (id != 0) return id;
            }
        }
    }
    if (is_pbr) {
        if (transparent) {
            return if (is_u32) scene.pipeline_instanced_pbr_blend_u32.id else scene.pipeline_instanced_pbr_blend_u16.id;
        }
        return if (is_u32) scene.pipeline_instanced_pbr_u32.id else scene.pipeline_instanced_pbr_u16.id;
    }
    if (transparent) {
        return if (is_u32) scene.pipeline_instanced_blend_u32.id else scene.pipeline_instanced_blend_u16.id;
    }
    return if (is_u32) scene.pipeline_instanced_u32.id else scene.pipeline_instanced_u16.id;
}

// Scene-side double-sided set, if the scene embeds one as `ds_pipelines`
// (see DoubleSidedPipelines). Missing field (legacy Scene) means no
// double-sided pipelines: selection falls back to regular ids. Accepts
// both Scene values and pointers; the set is returned BY VALUE because
// `&scene.ds_pipelines` would dangle for by-value args (the field would
// live in this function's stack frame).
fn sceneDoubleSided(scene: anytype) ?DoubleSidedPipelines {
    const S = @TypeOf(scene);
    const T = switch (@typeInfo(S)) {
        .pointer => |p| p.child,
        else => S,
    };
    if (@typeInfo(T) == .@"struct" and @hasField(T, "ds_pipelines")) {
        return scene.ds_pipelines;
    }
    return null;
}

// True when the queue item wants face culling disabled: either the item
// carries an explicit double_sided flag, or its mesh material is
// double-sided (covers queue items built before the flag existed).
fn itemDoubleSided(item: anytype) bool {
    const I = @TypeOf(item);
    if (@typeInfo(I) == .@"struct" and @hasField(I, "double_sided")) {
        if (item.double_sided) return true;
    }
    return meshDoubleSided(item.mesh);
}

// Derives double-sidedness from a mesh-like value exposing .material.
// Meshes without a .material field (foreign mocks) count as single-sided.
fn meshDoubleSided(mesh: anytype) bool {
    const M = @TypeOf(mesh);
    const T = switch (@typeInfo(M)) {
        .pointer => |p| p.child,
        else => M,
    };
    if (@typeInfo(T) == .@"struct" and @hasField(T, "material")) {
        return render_queue.materialIsDoubleSided(mesh.material);
    }
    return false;
}
