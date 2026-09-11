const sokol = @import("sokol");
const sg = sokol.gfx;
const shd = @import("shader");
const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_shd = @import("instanced_shader");

const Vertex = @import("../mesh.zig").Vertex;
const Mat4 = @import("math").Mat4;
const render_queue = @import("render_queue.zig");

// One shader + one vertex layout feeds an opaque u16/u32 pair plus its
// transparent blend twins. Only initPipelines uses this table.
pub const PipelineFamily = enum { standard, pbr, instanced, skinned_pbr };

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

// Selects the forward pipeline for a regular (non-instanced) queue item.
// Transparent items resolve to the blend twins; opaque selection is
// identical to the legacy logic, so existing pipeline ids are untouched.
// `scene` is generic (anytype) to avoid a scene.zig import cycle; it must
// expose the 12 pipeline_* fields. `item` must expose .mesh/.is_pbr/.transparent.
pub fn pipelineForRegularItem(scene: anytype, item: anytype) u32 {
    const mesh = item.mesh;
    const is_u32 = mesh.index_type == .UINT32;
    if (item.is_pbr) {
        if (mesh.skeleton != null) {
            if (item.transparent) {
                return if (is_u32) scene.pipeline_skinned_pbr_blend_u32.id else scene.pipeline_skinned_pbr_blend_u16.id;
            }
            return if (is_u32) scene.pipeline_skinned_pbr_u32.id else scene.pipeline_skinned_pbr_u16.id;
        }
        if (item.transparent) {
            return if (is_u32) scene.pipeline_pbr_blend_u32.id else scene.pipeline_pbr_blend_u16.id;
        }
        return if (is_u32) scene.pipeline_pbr_u32.id else scene.pipeline_pbr_u16.id;
    }
    if (item.transparent) {
        return if (is_u32) scene.pipeline_blend_u32.id else scene.pipeline_blend_u16.id;
    }
    return if (is_u32) scene.pipeline_u32.id else scene.pipeline_u16.id;
}
