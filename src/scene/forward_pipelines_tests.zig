const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;

const forward_pipelines = @import("forward_pipelines.zig");
const scene_pipelines = @import("pipelines.zig");
const render_queue = @import("render_queue.zig");
const RenderMeshItem = render_queue.RenderMeshItem;
const shader_material = @import("../shader_material.zig");

const ForwardPipelines = forward_pipelines.ForwardPipelines;
const ShaderMaterialSet = forward_pipelines.ShaderMaterialSet;
const ShaderMaterialCache = forward_pipelines.ShaderMaterialCache;
const forwardDesc = forward_pipelines.forwardDesc;
const shaderMaterialKey = forward_pipelines.shaderMaterialKey;

fn countPipelineFields(comptime T: type) usize {
    var n: usize = 0;
    for (@typeInfo(T).@"struct".fields) |f| {
        if (f.type == sg.Pipeline) n += 1;
    }
    return n;
}

test "pipeline selection follows transparency flag" {
    const pipelines = ForwardPipelines{
        .pipeline_pbr_u16 = .{ .id = 21 },
        .pipeline_pbr_u32 = .{ .id = 22 },
        .pipeline_pbr_blend_u16 = .{ .id = 23 },
        .pipeline_pbr_blend_u32 = .{ .id = 24 },
        .pipeline_skinned_pbr_u16 = .{ .id = 31 },
        .pipeline_skinned_pbr_u32 = .{ .id = 32 },
        .pipeline_skinned_pbr_blend_u16 = .{ .id = 33 },
        .pipeline_skinned_pbr_blend_u32 = .{ .id = 34 },
    };

    const opaque_pbr = RenderMeshItem{ .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = true, .texture_id = 0, .transparent = false };
    var blend_pbr = opaque_pbr;
    blend_pbr.transparent = true;
    try std.testing.expect(pipelines.forRegularItem(opaque_pbr) == 21);
    try std.testing.expect(pipelines.forRegularItem(blend_pbr) == 23);

    var opaque_u32 = opaque_pbr;
    opaque_u32.is_u32 = true;
    var blend_u32 = blend_pbr;
    blend_u32.is_u32 = true;
    try std.testing.expect(pipelines.forRegularItem(opaque_u32) == 22);
    try std.testing.expect(pipelines.forRegularItem(blend_u32) == 24);

    var skinned_pbr = opaque_pbr;
    skinned_pbr.is_skinned = true;
    var skinned_blend = blend_pbr;
    skinned_blend.is_skinned = true;
    var skinned_u32 = skinned_pbr;
    skinned_u32.is_u32 = true;
    var skinned_blend_u32 = skinned_blend;
    skinned_blend_u32.is_u32 = true;
    try std.testing.expect(pipelines.forRegularItem(skinned_u32) == 32);
    try std.testing.expect(pipelines.forRegularItem(skinned_blend_u32) == 34);
    try std.testing.expect(pipelines.forRegularItem(skinned_pbr) == 31);
    try std.testing.expect(pipelines.forRegularItem(skinned_blend) == 33);
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
    try std.testing.expectEqual(@as(u32, 111), set.pipelineFor(false, false, true));
    try std.testing.expectEqual(@as(u32, 114), set.pipelineFor(true, true, true));

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
    try std.testing.expect(cache.lookup(0) == null);
    try std.testing.expect(cache.lookup(shader_material.keyForName("ramp_wave")) == null);
    try std.testing.expect(shader_material.entryForKey(shader_material.keyForName("ramp_wave")) != null);
    try std.testing.expect(shader_material.entryForKey(shader_material.keyForName("nope")) == null);
}

test "pipeline tables cover every pipeline field (sample-count twins stay in sync)" {
    const n_forward = comptime countPipelineFields(ForwardPipelines);
    try std.testing.expectEqual(@as(usize, 12), n_forward);
    const n_ds = comptime countPipelineFields(scene_pipelines.DoubleSidedPipelines);
    try std.testing.expectEqual(@as(usize, 12), n_ds);
    const n_shader_mat = comptime countPipelineFields(ShaderMaterialSet);
    try std.testing.expectEqual(@as(usize, 8), n_shader_mat);
}

test "forwardDesc funnels the exact target shape into the pipeline descriptor" {
    const desc = forwardDesc(.{}, 4, .RGBA16F);
    try std.testing.expectEqual(@as(i32, 4), desc.sample_count);
    try std.testing.expect(desc.colors[0].pixel_format == .RGBA16F);
    try std.testing.expectEqual(sg.IndexType.UINT16, desc.index_type);
    try std.testing.expect(desc.depth.compare == .LESS_EQUAL);
    try std.testing.expect(desc.depth.write_enabled);
    try std.testing.expect(desc.cull_mode == .BACK);
    const ldr = forwardDesc(.{}, 1, .BGRA8);
    try std.testing.expectEqual(@as(i32, 1), ldr.sample_count);
    try std.testing.expect(ldr.colors[0].pixel_format == .BGRA8);
}

test "shaderMaterialKey separates sample-count AND format variants" {
    const key = shader_material.keyForName("ramp_wave");
    const hdr1 = shaderMaterialKey(key, 1, .RGBA16F);
    const ldr1 = shaderMaterialKey(key, 1, .BGRA8);
    const hdr4 = shaderMaterialKey(key, 4, .RGBA16F);
    const ldr4 = shaderMaterialKey(key, 4, .BGRA8);
    try std.testing.expect(ldr1 != hdr1);
    try std.testing.expect(hdr1 != hdr4);
    try std.testing.expect(ldr4 != hdr4);
    try std.testing.expect(ldr1 != ldr4);
    try std.testing.expectEqual(hdr1, shaderMaterialKey(key, 1, .RGBA16F));
    try std.testing.expectEqual(ldr4, shaderMaterialKey(key, 4, .BGRA8));
    try std.testing.expect(hdr1 != shaderMaterialKey(key + 1, 1, .RGBA16F));
}

test "ShaderMaterialCache defaults to the 1x RGBA16F shape" {
    const cache = ShaderMaterialCache{};
    try std.testing.expectEqual(@as(i32, 1), cache.sample_count);
    try std.testing.expect(cache.color_format == .RGBA16F);
}
