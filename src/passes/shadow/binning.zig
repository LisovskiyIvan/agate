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
            // Atomic publish: if growth fails, clear borrowed pointer list
            // (meshes may have been destroyed between frames) and return empty snapshot.
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
        // Atomic publish: stale pointers cleared on OOM, returning empty snapshot.
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
