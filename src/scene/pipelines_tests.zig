const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const render_queue = @import("render_queue.zig");
const pipelines = @import("pipelines.zig");
const forwardDesc = pipelines.forwardDesc;
const cullOffDescFor = pipelines.cullOffDescFor;
const DoubleSidedPipelines = pipelines.DoubleSidedPipelines;
const pipelineForRegularItem = pipelines.pipelineForRegularItem;
const pipelineForInstancedMesh = pipelines.pipelineForInstancedMesh;

test "forwardDesc pins the exact target shape without leaking depth/cull/blend state" {
    const hdr = forwardDesc(.{}, 1, .RGBA16F);
    try std.testing.expect(hdr.colors[0].pixel_format == .RGBA16F);
    // Invariants shared with the base desc: depth/cull/winding/samples.
    try std.testing.expect(hdr.depth.compare == .LESS_EQUAL);
    try std.testing.expect(hdr.depth.write_enabled);
    try std.testing.expect(hdr.cull_mode == .BACK);
    try std.testing.expect(hdr.face_winding == .CCW);
    try std.testing.expectEqual(sg.IndexType.UINT16, hdr.index_type);
    const msaa_hdr = forwardDesc(.{}, 4, .RGBA16F);
    try std.testing.expect(msaa_hdr.colors[0].pixel_format == .RGBA16F);
    try std.testing.expectEqual(@as(i32, 4), msaa_hdr.sample_count);
    try std.testing.expect(msaa_hdr.depth.write_enabled);
    // Exact second shape: 1x BGRA8 pins its own format and count.
    const ldr = forwardDesc(.{}, 1, .BGRA8);
    try std.testing.expect(ldr.colors[0].pixel_format == .BGRA8);
    try std.testing.expectEqual(@as(i32, 1), ldr.sample_count);
    // Composes with blendDescFor: HDR blend twins keep the format, gain
    // blend, and drop the depth write (transparent-twin contract).
    const blend_hdr = render_queue.blendDescFor(hdr);
    try std.testing.expect(blend_hdr.colors[0].pixel_format == .RGBA16F);
    try std.testing.expect(blend_hdr.colors[0].blend.enabled);
    try std.testing.expect(!blend_hdr.depth.write_enabled);
    try std.testing.expect(blend_hdr.depth.compare == .LESS_EQUAL);
    // Composes with cullOffDescFor: double-sided twins keep the format.
    const ds_hdr = cullOffDescFor(hdr);
    try std.testing.expect(ds_hdr.colors[0].pixel_format == .RGBA16F);
    try std.testing.expect(ds_hdr.cull_mode == .NONE);
    try std.testing.expect(ds_hdr.depth.write_enabled);
}

test "cullOffDescFor disables culling and preserves everything else" {
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

const TestItem = struct {
    transparent: bool = false,
    double_sided: bool = false,
    is_u32: bool = false,
    is_skinned: bool = false,
    skin_index: ?u32 = null,
};

fn testLegacyScene() struct {
    pipeline_pbr_u16: sg.Pipeline,
    pipeline_pbr_u32: sg.Pipeline,
    pipeline_skinned_pbr_u16: sg.Pipeline,
    pipeline_skinned_pbr_u32: sg.Pipeline,
    pipeline_instanced_pbr_u16: sg.Pipeline,
    pipeline_instanced_pbr_u32: sg.Pipeline,
    pipeline_pbr_blend_u16: sg.Pipeline,
    pipeline_pbr_blend_u32: sg.Pipeline,
    pipeline_skinned_pbr_blend_u16: sg.Pipeline,
    pipeline_skinned_pbr_blend_u32: sg.Pipeline,
    pipeline_instanced_pbr_blend_u16: sg.Pipeline,
    pipeline_instanced_pbr_blend_u32: sg.Pipeline,
} {
    return .{
        .pipeline_pbr_u16 = .{ .id = 21 },
        .pipeline_pbr_u32 = .{ .id = 22 },
        .pipeline_skinned_pbr_u16 = .{ .id = 31 },
        .pipeline_skinned_pbr_u32 = .{ .id = 32 },
        .pipeline_instanced_pbr_u16 = .{ .id = 51 },
        .pipeline_instanced_pbr_u32 = .{ .id = 52 },
        .pipeline_pbr_blend_u16 = .{ .id = 23 },
        .pipeline_pbr_blend_u32 = .{ .id = 24 },
        .pipeline_skinned_pbr_blend_u16 = .{ .id = 33 },
        .pipeline_skinned_pbr_blend_u32 = .{ .id = 34 },
        .pipeline_instanced_pbr_blend_u16 = .{ .id = 53 },
        .pipeline_instanced_pbr_blend_u32 = .{ .id = 54 },
    };
}

const TestDsScene = struct {
    pipeline_pbr_u16: sg.Pipeline,
    pipeline_pbr_u32: sg.Pipeline,
    pipeline_skinned_pbr_u16: sg.Pipeline,
    pipeline_skinned_pbr_u32: sg.Pipeline,
    pipeline_instanced_pbr_u16: sg.Pipeline,
    pipeline_instanced_pbr_u32: sg.Pipeline,
    pipeline_pbr_blend_u16: sg.Pipeline,
    pipeline_pbr_blend_u32: sg.Pipeline,
    pipeline_skinned_pbr_blend_u16: sg.Pipeline,
    pipeline_skinned_pbr_blend_u32: sg.Pipeline,
    pipeline_instanced_pbr_blend_u16: sg.Pipeline,
    pipeline_instanced_pbr_blend_u32: sg.Pipeline,
    ds_pipelines: DoubleSidedPipelines,
};

test "forward scenes keep existing pipeline ids; cutout resolves opaque" {
    const scene = testLegacyScene();

    // Opaque/transparent matrix, u16 + u32, pbr + skinned.
    // Flags come exclusively from render-owned snapshots (no live mesh referenced).
    var item = TestItem{};
    try std.testing.expectEqual(@as(u32, 21), pipelineForRegularItem(scene, item));
    item.transparent = true;
    try std.testing.expectEqual(@as(u32, 23), pipelineForRegularItem(scene, item));
    item.transparent = false;
    item.is_u32 = true;
    try std.testing.expectEqual(@as(u32, 22), pipelineForRegularItem(scene, item));

    item.is_u32 = false;
    item.is_skinned = true;
    item.transparent = false;
    try std.testing.expectEqual(@as(u32, 31), pipelineForRegularItem(scene, item));
    item.transparent = true;
    try std.testing.expectEqual(@as(u32, 33), pipelineForRegularItem(scene, item));
    // When is_skinned flag is omitted, skin_index also selects the skinned pipeline.
    item = TestItem{ .skin_index = 5 };
    try std.testing.expectEqual(@as(u32, 31), pipelineForRegularItem(scene, item));

    // Cutout material classifies opaque (transparent == false) and must
    // resolve to the opaque pipeline id, never a blend twin.
    item = TestItem{ .transparent = false };
    try std.testing.expectEqual(@as(u32, 21), pipelineForRegularItem(scene, item));
    item.is_u32 = true;
    try std.testing.expectEqual(@as(u32, 22), pipelineForRegularItem(scene, item));
}

test "double-sided items select cull-off twins, with regular fallback" {
    const legacy = testLegacyScene();
    const scene = TestDsScene{
        .pipeline_pbr_u16 = legacy.pipeline_pbr_u16,
        .pipeline_pbr_u32 = legacy.pipeline_pbr_u32,
        .pipeline_skinned_pbr_u16 = legacy.pipeline_skinned_pbr_u16,
        .pipeline_skinned_pbr_u32 = legacy.pipeline_skinned_pbr_u32,
        .pipeline_instanced_pbr_u16 = legacy.pipeline_instanced_pbr_u16,
        .pipeline_instanced_pbr_u32 = legacy.pipeline_instanced_pbr_u32,
        .pipeline_pbr_blend_u16 = legacy.pipeline_pbr_blend_u16,
        .pipeline_pbr_blend_u32 = legacy.pipeline_pbr_blend_u32,
        .pipeline_skinned_pbr_blend_u16 = legacy.pipeline_skinned_pbr_blend_u16,
        .pipeline_skinned_pbr_blend_u32 = legacy.pipeline_skinned_pbr_blend_u32,
        .pipeline_instanced_pbr_blend_u16 = legacy.pipeline_instanced_pbr_blend_u16,
        .pipeline_instanced_pbr_blend_u32 = legacy.pipeline_instanced_pbr_blend_u32,
        .ds_pipelines = .{
            .pbr_u16 = .{ .id = 121 },
            .pbr_blend_u32 = .{ .id = 124 },
            .instanced_pbr_u16 = .{ .id = 151 },
        },
    };
    var item = TestItem{ .double_sided = true };
    try std.testing.expectEqual(@as(u32, 121), pipelineForRegularItem(scene, item));

    // Missing twin (id 0) falls back to the regular pipeline.
    item.is_u32 = true;
    item.transparent = false;
    try std.testing.expectEqual(@as(u32, 22), pipelineForRegularItem(scene, item));

    // Blend + double-sided selects the cull-off blend twin.
    item.transparent = true;
    try std.testing.expectEqual(@as(u32, 124), pipelineForRegularItem(scene, item));

    // Single-sided items keep regular ids even when the ds set exists.
    item = TestItem{};
    try std.testing.expectEqual(@as(u32, 21), pipelineForRegularItem(scene, item));

    // Instanced PBR:
    try std.testing.expectEqual(@as(u32, 151), pipelineForInstancedMesh(scene, true, false, false, true));
    try std.testing.expectEqual(@as(u32, 51), pipelineForInstancedMesh(scene, true, false, false, false));
    try std.testing.expectEqual(@as(u32, 54), pipelineForInstancedMesh(scene, true, true, true, false));
    try std.testing.expectEqual(@as(u32, 51), pipelineForInstancedMesh(legacy, true, false, false, true));
}
