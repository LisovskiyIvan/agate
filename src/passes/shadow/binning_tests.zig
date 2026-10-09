const std = @import("std");
const Mesh = @import("../../mesh.zig").Mesh;
const ShadowPass = @import("core.zig").ShadowPass;
const jobs = @import("../../jobs.zig");

test "parallel shadow binning produces serial-identical results" {
    const ally = std.testing.allocator;
    const count = 300;
    const meshes = try ally.alloc(Mesh, count);
    defer ally.free(meshes);
    const ptrs = try ally.alloc(*Mesh, count);
    defer ally.free(ptrs);

    for (0..count) |i| {
        meshes[i] = Mesh{
            .name = "m",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 3,
            .index_type = if (i % 3 == 0) .UINT32 else .UINT16,
            .cast_shadows = (i % 5 != 0),
            .is_lod_child = (i % 11 == 0),
            .is_decal = (i % 13 == 0),
        };
        ptrs[i] = &meshes[i];
    }

    var pass_s = ShadowPass{
        .allocator = ally,
        .binned_meshes = .empty,
        .image = .{},
        .attachment_view = .{},
        .texture_view = .{},
        .sampler = .{},
        .depth_sampler = .{},
        .spot_image = .{},
        .spot_attachment_view = .{},
        .spot_texture_view = .{},
        .spot_needs_clear = false,
        .point_image = .{},
        .point_attachment_view = .{},
        .point_texture_view = .{},
        .point_needs_clear = false,
        .pipeline_u16 = .{},
        .pipeline_u32 = .{},
        .inst_pipeline_u16 = .{},
        .inst_pipeline_u32 = .{},
        .skinned_pipeline_u16 = .{},
        .skinned_pipeline_u32 = .{},
        .shadow_shader = .{},
        .inst_shader = .{},
        .skinned_shader = .{},
    };
    defer pass_s.binned_meshes.deinit(ally);
    defer pass_s.binned_source.deinit(ally);

    var pass_p = ShadowPass{
        .allocator = ally,
        .binned_meshes = .empty,
        .image = .{},
        .attachment_view = .{},
        .texture_view = .{},
        .sampler = .{},
        .depth_sampler = .{},
        .spot_image = .{},
        .spot_attachment_view = .{},
        .spot_texture_view = .{},
        .spot_needs_clear = false,
        .point_image = .{},
        .point_attachment_view = .{},
        .point_texture_view = .{},
        .point_needs_clear = false,
        .pipeline_u16 = .{},
        .pipeline_u32 = .{},
        .inst_pipeline_u16 = .{},
        .inst_pipeline_u32 = .{},
        .skinned_pipeline_u16 = .{},
        .skinned_pipeline_u32 = .{},
        .shadow_shader = .{},
        .inst_shader = .{},
        .skinned_shader = .{},
    };
    defer pass_p.binned_meshes.deinit(ally);
    defer pass_p.binned_source.deinit(ally);

    // Serial
    const res_s = pass_s.binMeshes(ptrs, null);

    // Parallel
    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();
    const res_p = pass_p.binMeshes(ptrs, pool);

    // Verify
    try std.testing.expectEqual(res_s.counts, res_p.counts);
    try std.testing.expectEqual(res_s.offsets, res_p.offsets);
    try std.testing.expectEqual(pass_s.binned_meshes.items.len, pass_p.binned_meshes.items.len);
    try std.testing.expect(pass_s.binned_meshes.items.len > 0);

    for (pass_s.binned_meshes.items, pass_p.binned_meshes.items) |m_s, m_p| {
        try std.testing.expectEqual(m_s, m_p);
    }
}
