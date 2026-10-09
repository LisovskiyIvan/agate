const sokol = @import("sokol");
const sg = sokol.gfx;
const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_pbr_shd = @import("instanced_pbr_shader");

const Vertex = @import("../mesh.zig").Vertex;
const Mat4 = @import("math").Mat4;
const render_queue = @import("render_queue.zig");
pub const TargetShape = @import("../target_shape.zig").TargetShape;
pub const defaultDepthFormat = @import("../target_shape.zig").defaultDepthFormat;

// One shader + one vertex layout feeds an opaque u16/u32 pair plus its
// transparent blend twins. Only initPipelines uses this table.
pub const PipelineFamily = enum { pbr, skinned_pbr, instanced_pbr };

// Single funnel for the base descriptor of every main-target pipeline
// (built-in families, double-sided twins, shader-material sets): the base
// depth/cull state plus the exact target sample count, depth format, and color format.
// Sokol validation rejects sg_apply_pipeline when pipeline.sample_count or
// depth.pixel_format differs from any attachment image of the current pass,
// so every pipeline that draws into the main target is built through here
// with the complete target shape.
pub fn forwardDescForShape(shader: sg.Shader, shape: TargetShape) sg.PipelineDesc {
    const resolved = shape.resolveEnvironment();
    var desc = sg.PipelineDesc{
        .shader = shader,
        .index_type = .UINT16,
        .depth = .{
            .pixel_format = resolved.depth_format,
            .compare = .LESS_EQUAL,
            .write_enabled = true,
        },
        .cull_mode = .BACK,
        .face_winding = .CCW,
        .sample_count = resolved.sample_count,
    };
    desc.colors[0].pixel_format = resolved.color_format;
    return desc;
}

pub fn forwardDesc(shader: sg.Shader, sample_count: i32, color_format: sg.PixelFormat) sg.PipelineDesc {
    return forwardDescForShape(shader, .{
        .sample_count = sample_count,
        .color_format = color_format,
        .depth_format = defaultDepthFormat(),
    });
}

// Fills the vertex layout for a family matching the established per-family
// configs (same buffers, attr slots, formats, offsets). Any shader-side
// layout risk lives here: attr changes must mirror the matching *.glsl.
pub fn pipelineLayoutFor(family: PipelineFamily, desc: *sg.PipelineDesc) void {
    switch (family) {
        .pbr => {
            desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
            desc.layout.attrs[pbr_shd.ATTR_pbr_position] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
            desc.layout.attrs[pbr_shd.ATTR_pbr_normal] = .{ .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
            desc.layout.attrs[pbr_shd.ATTR_pbr_color0] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
            desc.layout.attrs[pbr_shd.ATTR_pbr_texcoord0] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };
            desc.layout.attrs[pbr_shd.ATTR_pbr_texcoord1] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv1") };
            desc.layout.attrs[pbr_shd.ATTR_pbr_tangent] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "tangent") };
        },
        .instanced_pbr => {
            desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_position] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "position") };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_normal] = .{ .buffer_index = 0, .format = .FLOAT3, .offset = @offsetOf(Vertex, "normal") };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_tangent] = .{ .buffer_index = 0, .format = .FLOAT4, .offset = @offsetOf(Vertex, "tangent") };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_color0] = .{ .buffer_index = 0, .format = .FLOAT4, .offset = @offsetOf(Vertex, "color") };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_texcoord0] = .{ .buffer_index = 0, .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv") };
            desc.layout.attrs[inst_pbr_shd.ATTR_instanced_pbr_texcoord1] = .{ .buffer_index = 0, .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv1") };

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
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_texcoord1] = .{ .format = .FLOAT2, .offset = @offsetOf(Vertex, "uv1") };
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_tangent] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "tangent") };
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_joints] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "joints") };
            desc.layout.attrs[skinned_pbr_shd.ATTR_skinned_pbr_weights] = .{ .format = .FLOAT4, .offset = @offsetOf(Vertex, "weights") };
        },
    }
}

// Creates an opaque u16/u32 pair plus transparent blend twins from one
// base desc. Order and index_type switches match the selection order.
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
    pbr: sg.Shader,
    instanced_pbr: sg.Shader,
    skinned_pbr: sg.Shader,
};

// Double-sided (cull-off) pipeline twins for every family: opaque u16/u32
// plus blend u16/u32, i.e. 12 pipelines total (3 families x opaque/blend x
// u16/u32). Instanced twins are included: same table, no extra code.
// Intended embedding (Scene gains ONE field, init/deinit forwarded):
//   ds_pipelines: DoubleSidedPipelines = .{},
//   self.ds_pipelines.initFromShaders(.{ .pbr = pbr_sh, ... });
//   self.ds_pipelines.deinit();
// NOTE: back-face lighting still uses geometric normals (no
// gl_FrontFacing flip in the shaders); back faces may shade dark. This
// matches the minimal cull-off scope.
pub const DoubleSidedPipelines = struct {
    pbr_u16: sg.Pipeline = .{},
    pbr_u32: sg.Pipeline = .{},
    pbr_blend_u16: sg.Pipeline = .{},
    pbr_blend_u32: sg.Pipeline = .{},
    skinned_pbr_u16: sg.Pipeline = .{},
    skinned_pbr_u32: sg.Pipeline = .{},
    skinned_pbr_blend_u16: sg.Pipeline = .{},
    skinned_pbr_blend_u32: sg.Pipeline = .{},
    instanced_pbr_u16: sg.Pipeline = .{},
    instanced_pbr_u32: sg.Pipeline = .{},
    instanced_pbr_blend_u16: sg.Pipeline = .{},
    instanced_pbr_blend_u32: sg.Pipeline = .{},

    // Builds all 12 cull-off twins from the family shaders (GPU calls).
    // Base descs mirror Scene.initPipelines (depth LESS_EQUAL/write on,
    // BACK cull, CCW winding) plus the main-target sample count and color
    // format (sokol requires pipelines to match the attachment sample
    // count they draw into); makeCullOffPair forces cull off.
    pub fn initFromShadersForShape(self: *DoubleSidedPipelines, shaders: DoubleSidedSourceShaders, shape: TargetShape) void {
        const specs = [_]struct {
            shader: sg.Shader,
            family: PipelineFamily,
            opaque_u16: *sg.Pipeline,
            opaque_u32: *sg.Pipeline,
            blend_u16: *sg.Pipeline,
            blend_u32: *sg.Pipeline,
        }{
            .{
                .shader = shaders.pbr,
                .family = .pbr,
                .opaque_u16 = &self.pbr_u16,
                .opaque_u32 = &self.pbr_u32,
                .blend_u16 = &self.pbr_blend_u16,
                .blend_u32 = &self.pbr_blend_u32,
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
            var desc = forwardDescForShape(spec.shader, shape);
            pipelineLayoutFor(spec.family, &desc);
            makeCullOffPair(desc, spec.opaque_u16, spec.opaque_u32, spec.blend_u16, spec.blend_u32);
        }
    }

    pub fn initFromShaders(self: *DoubleSidedPipelines, shaders: DoubleSidedSourceShaders, sample_count: i32, color_format: sg.PixelFormat) void {
        self.initFromShadersForShape(shaders, .{
            .sample_count = sample_count,
            .color_format = color_format,
            .depth_format = defaultDepthFormat(),
        });
    }

    pub fn deinit(self: *DoubleSidedPipelines) void {
        inline for (.{
            &self.pbr_u16,                 &self.pbr_u32,
            &self.pbr_blend_u16,           &self.pbr_blend_u32,
            &self.skinned_pbr_u16,         &self.skinned_pbr_u32,
            &self.skinned_pbr_blend_u16,   &self.skinned_pbr_blend_u32,
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
// identical to the established mapping, so existing pipeline ids are untouched.
// Reads ONLY render-owned snapshots (is_u32/is_skinned/double_sided flags):
// the draw phase never touches live Mesh/Material. `scene` is generic (anytype)
// to avoid a scene.zig import cycle; it must expose the pipeline_*
// fields. `item` snapshot flags (.is_u32/.index_type and .is_skinned/.skin_index
// fallbacks cover foreign mocks; no live mesh pointer is referenced).
// Cutout items are opaque (transparent == false) and resolve to opaque ids.
pub fn pipelineForRegularItem(scene: anytype, item: anytype) u32 {
    const is_u32 = blk: {
        if (@hasField(@TypeOf(item), "is_u32") and item.is_u32) break :blk true;
        if (@hasField(@TypeOf(item), "index_type") and item.index_type == .UINT32) break :blk true;
        break :blk false;
    };

    const is_skinned = blk: {
        if (@hasField(@TypeOf(item), "is_skinned") and item.is_skinned) break :blk true;
        if (@hasField(@TypeOf(item), "skin_index") and item.skin_index != null) break :blk true;
        break :blk false;
    };

    const ds = sceneDoubleSided(scene);
    const want_ds = if (ds) |_| itemDoubleSided(item) else false;

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

// Selects the instanced pipeline. Same double-sided contract as
// pipelineForRegularItem: cull-off twin when requested and available,
// regular pipeline otherwise (scenes without the set behave exactly as before).
// `scene` must expose the pipeline_instanced_* fields.
pub fn pipelineForInstancedMesh(scene: anytype, is_pbr: bool, transparent: bool, is_u32: bool, double_sided: bool) u32 {
    _ = is_pbr;
    if (sceneDoubleSided(scene)) |ds| {
        if (double_sided) {
            const id = if (transparent)
                (if (is_u32) ds.instanced_pbr_blend_u32.id else ds.instanced_pbr_blend_u16.id)
            else
                (if (is_u32) ds.instanced_pbr_u32.id else ds.instanced_pbr_u16.id);
            if (id != 0) return id;
        }
    }
    if (transparent) {
        return if (is_u32) scene.pipeline_instanced_pbr_blend_u32.id else scene.pipeline_instanced_pbr_blend_u16.id;
    }
    return if (is_u32) scene.pipeline_instanced_pbr_u32.id else scene.pipeline_instanced_pbr_u16.id;
}

// Scene-side double-sided set, if the scene embeds one as `ds_pipelines`
// (see DoubleSidedPipelines). Missing field (scenes without the set) means no
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

// True when the queue item wants face culling disabled: the item carries
// an explicit double_sided snapshot (no live material is read in the draw phase).
fn itemDoubleSided(item: anytype) bool {
    const I = @TypeOf(item);
    if (@typeInfo(I) == .@"struct" and @hasField(I, "double_sided")) {
        return item.double_sided;
    }
    return false;
}
