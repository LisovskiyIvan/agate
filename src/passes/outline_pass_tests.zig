//! Tests for `passes/outline_pass.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const sg = @import("sokol").gfx;
const outline_shd = @import("outline_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color4 = math.Color4;
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;
const Vertex = mesh_mod.Vertex;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const scene_render_queue = @import("../scene/render_queue.zig");
const prod = @import("outline_pass.zig");
const clampWidthPx = prod.clampWidthPx;
const max_width_px = prod.max_width_px;
const ndcExpandForViewport = prod.ndcExpandForViewport;
const expandVertex = prod.expandVertex;
const outlineParamsFor = prod.outlineParamsFor;
const default_depth_bias = prod.default_depth_bias;
const shouldOutlineMesh = prod.shouldOutlineMesh;
const makeOutlineDrawItem = prod.makeOutlineDrawItem;
const configureOutlineDesc = prod.configureOutlineDesc;
const configureOutlineInstDesc = prod.configureOutlineInstDesc;
const configureOutlineCutoutDesc = prod.configureOutlineCutoutDesc;
const configureOutlineSkinnedDesc = prod.configureOutlineSkinnedDesc;
const OutlinePass = prod.OutlinePass;

test "clampWidthPx clamps to [0, max] and sanitizes non-finite" {
    try std.testing.expectEqual(@as(f32, 0.0), clampWidthPx(-3.0));
    try std.testing.expectEqual(@as(f32, 0.0), clampWidthPx(0.0));
    try std.testing.expectEqual(@as(f32, 2.5), clampWidthPx(2.5));
    try std.testing.expectEqual(max_width_px, clampWidthPx(max_width_px + 100.0));
    try std.testing.expectEqual(@as(f32, 0.0), clampWidthPx(std.math.nan(f32)));
    try std.testing.expectEqual(@as(f32, 0.0), clampWidthPx(std.math.inf(f32)));
}

test "expandVertex offsets position along normal by scale" {
    const pos = [3]f32{ 1.0, 2.0, 3.0 };
    const nrm = [3]f32{ 0.0, 1.0, 0.0 };
    try std.testing.expectEqual([3]f32{ 1.0, 2.5, 3.0 }, expandVertex(pos, nrm, 0.5));
    try std.testing.expectEqual(pos, expandVertex(pos, nrm, 0.0));
    try std.testing.expectEqual([3]f32{ 1.0, 1.0, 3.0 }, expandVertex(pos, nrm, -1.0));
    try std.testing.expectEqual(pos, expandVertex(pos, nrm, std.math.nan(f32)));
    try std.testing.expectEqual(pos, expandVertex(pos, nrm, std.math.inf(f32)));
}

test "ndcExpandForViewport converts px width to NDC without div-by-zero" {
    const e = ndcExpandForViewport(2.0, 800.0, 600.0);
    try std.testing.expectApproxEqAbs(2.0 * 2.0 / 800.0, e[0], 1e-6);
    try std.testing.expectApproxEqAbs(2.0 * 2.0 / 600.0, e[1], 1e-6);
    // Degenerate viewports never produce Inf/NaN.
    for ([_]f32{ 0.0, -10.0, std.math.nan(f32), std.math.inf(f32) }) |bad| {
        try std.testing.expect(std.math.isFinite(ndcExpandForViewport(2.0, bad, bad)[0]));
        try std.testing.expect(std.math.isFinite(ndcExpandForViewport(2.0, bad, bad)[1]));
    }
    // Width clamping flows through.
    const c = ndcExpandForViewport(max_width_px + 100.0, 800.0, 600.0);
    try std.testing.expectApproxEqAbs(max_width_px * 2.0 / 800.0, c[0], 1e-6);
}

test "configureOutlineDesc sets inverse-hull state" {
    var desc = std.mem.zeroes(sg.PipelineDesc);
    configureOutlineDesc(&desc);
    try std.testing.expect(desc.cull_mode == .FRONT);
    try std.testing.expect(desc.depth.compare == .LESS_EQUAL);
    try std.testing.expect(!desc.depth.write_enabled);
    try std.testing.expect(desc.colors[0].blend.enabled);
    try std.testing.expect(desc.colors[0].blend.src_factor_rgb == .SRC_ALPHA);
    try std.testing.expect(desc.colors[0].blend.dst_factor_rgb == .ONE_MINUS_SRC_ALPHA);
    try std.testing.expectEqual(@sizeOf(Vertex), desc.layout.buffers[0].stride);
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_position].format == .FLOAT3);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "position"))),
        desc.layout.attrs[outline_shd.ATTR_outline_position].offset,
    );
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_normal].format == .FLOAT3);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "normal"))),
        desc.layout.attrs[outline_shd.ATTR_outline_normal].offset,
    );
}

test "configureOutlineInstDesc binds per-instance matrix buffer" {
    var desc = std.mem.zeroes(sg.PipelineDesc);
    configureOutlineInstDesc(&desc);
    try std.testing.expect(desc.cull_mode == .FRONT);
    try std.testing.expectEqual(@sizeOf(Vertex), desc.layout.buffers[0].stride);
    try std.testing.expectEqual(@sizeOf(Mat4), desc.layout.buffers[1].stride);
    try std.testing.expect(desc.layout.buffers[1].step_func == .PER_INSTANCE);
    inline for (0..4) |i| {
        const attr = switch (i) {
            0 => outline_shd.ATTR_outline_inst_inst_mat0,
            1 => outline_shd.ATTR_outline_inst_inst_mat1,
            2 => outline_shd.ATTR_outline_inst_inst_mat2,
            else => outline_shd.ATTR_outline_inst_inst_mat3,
        };
        try std.testing.expectEqual(@as(i32, 1), desc.layout.attrs[attr].buffer_index);
        try std.testing.expectEqual(@as(i32, @intCast(i * 16)), desc.layout.attrs[attr].offset);
        try std.testing.expect(desc.layout.attrs[attr].format == .FLOAT4);
    }
}

test "configureOutlineSkinnedDesc binds skin attributes" {
    var desc = std.mem.zeroes(sg.PipelineDesc);
    configureOutlineSkinnedDesc(&desc);
    try std.testing.expect(desc.cull_mode == .FRONT);
    try std.testing.expectEqual(@sizeOf(Vertex), desc.layout.buffers[0].stride);
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_skinned_joints].format == .FLOAT4);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "joints"))),
        desc.layout.attrs[outline_shd.ATTR_outline_skinned_joints].offset,
    );
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_skinned_weights].format == .FLOAT4);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "weights"))),
        desc.layout.attrs[outline_shd.ATTR_outline_skinned_weights].offset,
    );
}

test "shouldOutlineMesh filters invisible and empty meshes" {
    var mesh = Mesh{
        .name = "outline-test",
        .vertex_buffer = .{ .id = 1 },
        .index_buffer = .{ .id = 2 },
        .index_count = 36,
    };
    try std.testing.expect(shouldOutlineMesh(&mesh));

    mesh.is_visible = false;
    try std.testing.expect(!shouldOutlineMesh(&mesh));
    mesh.is_visible = true;

    mesh.index_count = 0;
    try std.testing.expect(!shouldOutlineMesh(&mesh));
    mesh.index_count = 36;

    mesh.vertex_buffer = .{};
    try std.testing.expect(!shouldOutlineMesh(&mesh));
    mesh.vertex_buffer = .{ .id = 1 };

    // Skinned and instanced meshes take dedicated pipeline families, so they
    // stay outline-eligible here (render() routes them).
    var skel: Skeleton = undefined;
    mesh.skeleton = &skel;
    try std.testing.expect(shouldOutlineMesh(&mesh));
    mesh.skeleton = null;
    mesh.instances.items.len = 3;
    try std.testing.expect(shouldOutlineMesh(&mesh));
}

test "render with empty mesh list is a GPU-free no-op" {
    var pass = OutlinePass{};
    // Zero pipelines + empty list must return before any sokol call.
    pass.render(Mat4.identity, Vec3.zero, &.{}, Color4.white, 2.0);
    // Zero width is a no-op even with a (skipped) mesh present.
    var mesh = Mesh{
        .name = "outline-test",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    const meshes = [_]*Mesh{&mesh};
    pass.render(Mat4.identity, Vec3.zero, &meshes, Color4.white, 0.0);
}

test "outlineParamsFor packs clamped width and current viewport" {
    OutlinePass.resize(800, 600);
    const p = outlineParamsFor(2.0);
    try std.testing.expectEqual([4]f32{ 2.0, 800.0, 600.0, default_depth_bias }, p);
    const clamped = outlineParamsFor(max_width_px + 100.0);
    try std.testing.expectEqual(max_width_px, clamped[0]);
}

test "configureOutlineCutoutDesc binds position and uv without culling" {
    var desc = std.mem.zeroes(sg.PipelineDesc);
    configureOutlineCutoutDesc(&desc);
    try std.testing.expect(desc.cull_mode == .NONE);
    try std.testing.expect(desc.depth.compare == .LESS_EQUAL);
    try std.testing.expect(!desc.depth.write_enabled);
    try std.testing.expectEqual(@sizeOf(Vertex), desc.layout.buffers[0].stride);
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_cutout_position].format == .FLOAT3);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "position"))),
        desc.layout.attrs[outline_shd.ATTR_outline_cutout_position].offset,
    );
    try std.testing.expect(desc.layout.attrs[outline_shd.ATTR_outline_cutout_texcoord0].format == .FLOAT2);
    try std.testing.expectEqual(
        @as(i32, @intCast(@offsetOf(Vertex, "uv"))),
        desc.layout.attrs[outline_shd.ATTR_outline_cutout_texcoord0].offset,
    );
}

// Prepared outline item does not reference live data: model, bone copy,
// and cutout snapshot survive mutation of TRS/material and skeleton republishing.
test "P4: outline item owns model, skin and cutout snapshots" {
    const ally = std.testing.allocator;
    const skel = try Skeleton.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(1, 0, 0);
    skel.update();

    var mesh = Mesh{
        .name = "outline_skinned",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(4, 0, 0),
        .skeleton = skel,
    };
    var skins: scene_render_queue.SkinStorage = .empty;
    defer skins.deinit(ally);

    const it = makeOutlineDrawItem(ally, &skins, &mesh, 7, 3, .published) orelse return error.TestUnexpectedResult;
    try std.testing.expect(it.skin_index != null);
    try std.testing.expectEqual(@as(usize, 1), skins.items.len);
    try std.testing.expect(it.source_uid != 0);
    try std.testing.expectEqual(it.source_uid, mesh.uid);
    try std.testing.expectEqual(@as(u32, 3), it.source_mesh);

    mesh.position = Vec3.new(99, 99, 99);
    skel.bones[0].local_position = Vec3.new(5, 0, 0);
    skel.update();
    skel.update();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), skel.getRenderSkinMatrices()[0].m[12], 1e-4);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), it.model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), skins.items[it.skin_index.?][0].m[12], 1e-4);

    // Cutout snapshot: view/sampler/cutoff are copied; live material is not read.
    const material_mod = @import("../material.zig");
    const texture_mod = @import("../texture.zig");
    var cut_mat = material_mod.PBRMaterial.init("outline_cut");
    cut_mat.alpha_mode = .cutout;
    cut_mat.alpha_cutoff = 0.3;
    cut_mat.albedo_texture = texture_mod.Texture{
        .image = .{},
        .view = .{ .id = 77 },
        .sampler = .{ .id = 78 },
        .width = 4,
        .height = 4,
    };
    var rigid = Mesh{
        .name = "outline_cutout",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = .{ .pbr = &cut_mat },
    };
    var skins2: scene_render_queue.SkinStorage = .empty;
    defer skins2.deinit(ally);
    const cut = makeOutlineDrawItem(ally, &skins2, &rigid, 7, 5, .published) orelse return error.TestUnexpectedResult;
    try std.testing.expect(cut.is_cutout);
    try std.testing.expectEqual(@as(usize, 0), skins2.items.len);
    try std.testing.expect(cut.source_uid != 0);
    try std.testing.expectEqual(@as(u32, 5), cut.source_mesh);

    cut_mat.alpha_cutoff = 0.9;
    cut_mat.albedo_texture = null;
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), cut.cutout_cutoff, 1e-6);
    try std.testing.expectEqual(@as(u32, 77), cut.cutout_view.?.id);
    try std.testing.expectEqual(@as(u32, 78), cut.cutout_sampler.?.id);
}

// OOM of bone copy: makeOutlineDrawItem returns null (caller skips item);
// unskinned mesh builds without allocations even under failing allocator.
test "P4: outline skin OOM returns null, rigid build stays allocation-free" {
    const ally = std.testing.allocator;
    const skel = try Skeleton.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(1, 0, 0);
    skel.update();

    var skinned = Mesh{
        .name = "oom_outline_skinned",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .skeleton = skel,
    };
    var rigid = Mesh{
        .name = "oom_outline_rigid",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };

    var limited = FailNthP4Outline{ .backing = ally, .fail_on = 1 };
    var skins: scene_render_queue.SkinStorage = .empty;
    defer skins.deinit(limited.allocator());
    // First allocation (bone copy) fails — item is not built.
    try std.testing.expect(makeOutlineDrawItem(limited.allocator(), &skins, &skinned, 0, 0, .published) == null);
    try std.testing.expectEqual(@as(usize, 0), skins.items.len);
    // Rigid path makes no allocations — succeeds under failing allocator.
    const it = makeOutlineDrawItem(limited.allocator(), &skins, &rigid, 0, 1, .published) orelse return error.TestUnexpectedResult;
    try std.testing.expect(it.skin_index == null);
    try std.testing.expectEqual(@as(usize, 0), skins.items.len);
}

const FailNthP4Outline = struct {
    backing: std.mem.Allocator,
    fail_on: usize,
    count: usize = 0,

    fn allocator(self: *FailNthP4Outline) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *FailNthP4Outline = @ptrCast(@alignCast(ctx));
        self.count += 1;
        if (self.count == self.fail_on) return null;
        return self.backing.rawAlloc(len, alignment, ra);
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *FailNthP4Outline = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory, alignment, new_len, ra);
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *FailNthP4Outline = @ptrCast(@alignCast(ctx));
        return self.backing.rawRemap(memory, alignment, new_len, ra);
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *FailNthP4Outline = @ptrCast(@alignCast(ctx));
        return self.backing.rawFree(memory, alignment, ra);
    }
};

test "stage-2A: outline item carries source uid and list index" {
    const ally = std.testing.allocator;
    var m0 = Mesh{
        .name = "outline0",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    var m1 = Mesh{
        .name = "outline1",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    var skins: scene_render_queue.SkinStorage = .empty;
    defer skins.deinit(ally);
    const it0 = makeOutlineDrawItem(ally, &skins, &m0, 11, 0, .published) orelse return error.TestUnexpectedResult;
    const it1 = makeOutlineDrawItem(ally, &skins, &m1, 11, 1, .published) orelse return error.TestUnexpectedResult;
    try std.testing.expect(it0.source_uid != 0);
    try std.testing.expect(it1.source_uid != 0);
    try std.testing.expect(it0.source_uid != it1.source_uid);
    try std.testing.expectEqual(m0.uid, it0.source_uid);
    try std.testing.expectEqual(m1.uid, it1.source_uid);
    try std.testing.expectEqual(@as(u32, 0), it0.source_mesh);
    try std.testing.expectEqual(@as(u32, 1), it1.source_mesh);
}
