//! Mesh binning: grouping shadow casters by pipeline bucket. Split out of
//! `shadow_pass.zig` (facade).
//!
//! `binMeshes` bins the input mesh list into the pass-owned
//! `binned_meshes`/`binned_source` scratch in `bucket_order`, with a
//! parallel counting + scatter path when a job pool with workers is
//! available. Takes the pass as `anytype` so this module never imports
//! `core.zig` or the facade back (same discipline as `particles/*`,
//! `profiler/*`); the `Bucket` vocabulary lives in `types.zig`.
//!
//! Moved tests reach `core.ShadowPass` through a block-scoped import that
//! exists only in test builds.
const std = @import("std");
const mesh_mod = @import("../../mesh.zig");
const Mesh = mesh_mod.Mesh;
const jobs = @import("../../jobs.zig");

const types = @import("types.zig");

const ParallelShadowBinning = struct {
    meshes: []const *Mesh,
    span: usize,
    chunk_counts: [][6]usize,
    chunk_offsets: [][6]usize,
    out_binned: []*Mesh,
    out_source: []u32,

    fn countChunkRange(pass: *ParallelShadowBinning, start: usize, end: usize) void {
        for (start..end) |chunk_id| {
            const lo = chunk_id * pass.span;
            if (lo >= pass.meshes.len) {
                pass.chunk_counts[chunk_id] = .{ 0, 0, 0, 0, 0, 0 };
                continue;
            }
            const hi = @min(lo + pass.span, pass.meshes.len);
            var local: [6]usize = .{ 0, 0, 0, 0, 0, 0 };
            for (pass.meshes[lo..hi]) |mesh| {
                if (!mesh.cast_shadows or mesh.is_lod_child or mesh.is_decal) continue;
                local[@intFromEnum(types.bucketFor(mesh))] += 1;
            }
            pass.chunk_counts[chunk_id] = local;
        }
    }

    fn scatterChunkRange(pass: *ParallelShadowBinning, start: usize, end: usize) void {
        for (start..end) |chunk_id| {
            const lo = chunk_id * pass.span;
            if (lo >= pass.meshes.len) continue;
            const hi = @min(lo + pass.span, pass.meshes.len);
            var cursors = pass.chunk_offsets[chunk_id];
            for (pass.meshes[lo..hi], lo..) |mesh, src_idx| {
                if (!mesh.cast_shadows or mesh.is_lod_child or mesh.is_decal) continue;
                const b = @intFromEnum(types.bucketFor(mesh));
                pass.out_binned[cursors[b]] = mesh;
                pass.out_source[cursors[b]] = @intCast(src_idx);
                cursors[b] += 1;
            }
        }
    }
};

pub fn binMeshes(
    self: anytype,
    meshes: []const *Mesh,
    pool: ?*jobs.Pool,
) types.BinResult {
    var counts: [6]usize = .{ 0, 0, 0, 0, 0, 0 };
    var offsets: [6]usize = undefined;

    const min_meshes_for_parallel: usize = 128;
    if (pool != null and pool.?.workerCount() > 0 and meshes.len >= min_meshes_for_parallel) {
        const p = pool.?;
        const chunk_count = @min((p.workerCount() + 1) * 2, 32);
        const span = (meshes.len + chunk_count - 1) / chunk_count;

        var chunk_counts_buf: [32][6]usize = undefined;
        var chunk_offsets_buf: [32][6]usize = undefined;
        const chunk_counts = chunk_counts_buf[0..chunk_count];
        const chunk_offsets = chunk_offsets_buf[0..chunk_count];

        var pass = ParallelShadowBinning{
            .meshes = meshes,
            .span = span,
            .chunk_counts = chunk_counts,
            .chunk_offsets = chunk_offsets,
            .out_binned = &.{},
            .out_source = &.{},
        };

        p.forkJoin(ParallelShadowBinning, &pass, ParallelShadowBinning.countChunkRange, chunk_count);

        for (0..chunk_count) |c| {
            for (0..6) |b| {
                counts[b] += chunk_counts[c][b];
            }
        }

        var total: usize = 0;
        for (0..6) |b| {
            offsets[b] = total;
            total += counts[b];
        }

        if (total > self.binned_meshes.items.len) {
            // Атомарность публикации: рост не удался — список заимствованных
            // указателей очищается (меши могли быть destroyMesh между кадрами,
            // prepare НЕ ДОЛЖЕН их читать), возвращается пустой coherent-снимок.
            self.binned_meshes.resize(self.allocator, total) catch {
                self.binned_meshes.clearRetainingCapacity();
                self.binned_source.clearRetainingCapacity();
                return .{
                    .counts = [_]usize{0} ** 6,
                    .offsets = [_]usize{0} ** 6,
                };
            };
            self.binned_source.resize(self.allocator, total) catch {
                self.binned_meshes.clearRetainingCapacity();
                self.binned_source.clearRetainingCapacity();
                return .{
                    .counts = [_]usize{0} ** 6,
                    .offsets = [_]usize{0} ** 6,
                };
            };
        } else {
            self.binned_meshes.shrinkRetainingCapacity(total);
            self.binned_source.shrinkRetainingCapacity(total);
        }

        for (0..6) |b| {
            var cur = offsets[b];
            for (0..chunk_count) |c| {
                chunk_offsets[c][b] = cur;
                cur += chunk_counts[c][b];
            }
        }

        pass.out_binned = self.binned_meshes.items;
        pass.out_source = self.binned_source.items;
        p.forkJoin(ParallelShadowBinning, &pass, ParallelShadowBinning.scatterChunkRange, chunk_count);

        return .{ .counts = counts, .offsets = offsets };
    }

    // Serial fallback
    for (meshes) |mesh| {
        if (!mesh.cast_shadows or mesh.is_lod_child or mesh.is_decal) continue;
        counts[@intFromEnum(types.bucketFor(mesh))] += 1;
    }

    var total: usize = 0;
    var cursors: [6]usize = undefined;
    for (0..6) |b| {
        offsets[b] = total;
        cursors[b] = total;
        total += counts[b];
    }

    if (total > self.binned_meshes.items.len) {
        // Атомарность публикации: см. выше — старые указатели не
        // удерживаются, prepare ничего из них не читает.
        self.binned_meshes.resize(self.allocator, total) catch {
            self.binned_meshes.clearRetainingCapacity();
            self.binned_source.clearRetainingCapacity();
            return .{
                .counts = [_]usize{0} ** 6,
                .offsets = [_]usize{0} ** 6,
            };
        };
        self.binned_source.resize(self.allocator, total) catch {
            self.binned_meshes.clearRetainingCapacity();
            self.binned_source.clearRetainingCapacity();
            return .{
                .counts = [_]usize{0} ** 6,
                .offsets = [_]usize{0} ** 6,
            };
        };
    } else {
        self.binned_meshes.shrinkRetainingCapacity(total);
        self.binned_source.shrinkRetainingCapacity(total);
    }

    for (meshes, 0..) |mesh, src_idx| {
        if (!mesh.cast_shadows or mesh.is_lod_child or mesh.is_decal) continue;
        const b = @intFromEnum(types.bucketFor(mesh));
        self.binned_meshes.items[cursors[b]] = mesh;
        self.binned_source.items[cursors[b]] = @intCast(src_idx);
        cursors[b] += 1;
    }

    return .{ .counts = counts, .offsets = offsets };
}

test "parallel shadow binning produces serial-identical results" {
    const ShadowPass = @import("core.zig").ShadowPass;
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
