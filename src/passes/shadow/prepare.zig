//! Render-owned shadow snapshot builder. Split out of `shadow_pass.zig` (facade).
//!
//! `prepareInto` is the core prepare algorithm, parameterized by destination payload:
//! it bins meshes into the pass-owned `binned_meshes` scratch (via `binning.zig`),
//! then snapshots items and bone matrix copies into `out`.
const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const mesh_mod = @import("../../mesh.zig");
const Mesh = mesh_mod.Mesh;
const scene_render_queue = @import("../../scene/render_queue.zig");
const jobs = @import("../../jobs.zig");

const types = @import("types.zig");

/// Core prepare algorithm, parameterized by the destination payload: bins
/// meshes into the pass-owned `binned_meshes` scratch, then snapshots
/// items + skin copies into `out`. `out` may be the standalone
/// `prepared` (via `prepare`) or a leased Scene frame slot.
pub fn prepareInto(
    self: anytype,
    out: anytype,
    meshes: []const *Mesh,
    cache_key: u64,
    instance_source: mesh_mod.InstanceSource,
    pool: ?*jobs.Pool,
) types.BinResult {
    for (meshes) |m| _ = m.ensureUid();
    const binned = self.binMeshes(meshes, pool);
    const total = self.binned_meshes.items.len;
    if (total > out.items.items.len) {
        out.items.resize(self.allocator, total) catch {
            // Atomic publish: if growth fails, return an empty coherent snapshot
            // (items + skins + counts) rather than stale items with cleared skins.
            out.items.clearRetainingCapacity();
            out.skins.clearRetainingCapacity();
            out.bin = .{
                .counts = [_]usize{0} ** 6,
                .offsets = [_]usize{0} ** 6,
            };
            return out.bin;
        };
    } else {
        out.items.shrinkRetainingCapacity(total);
    }
    out.skins.clearRetainingCapacity();

    for (self.binned_meshes.items, 0..) |mesh, idx| {
        const is_inst = mesh.instances.items.len > 0;
        // Instanced meshes read the frame's staged render state;
        // regular meshes use the fresh world cache.
        const staged = mesh.instanceRenderSource(instance_source).*;
        const aabb_w = if (!is_inst) scene_render_queue.worldAABBCached(cache_key, mesh) else staged.bounds;
        const model = if (!is_inst) scene_render_queue.worldMatrixCached(cache_key, mesh) else Mat4.identity;
        // Render-owned skin copy. Item layout must remain 1:1 with binned_meshes,
        // so OOM flags the item gpu_pending to skip in renderBuckets without shifting bucket slices.
        var skin_index: ?u32 = null;
        var skin_oom = false;
        if (mesh.skeleton) |skel| {
            const src = skel.getRenderSkinMatrices();
            out.skins.ensureUnusedCapacity(self.allocator, 1) catch {
                skin_oom = true;
            };
            if (!skin_oom) {
                skin_index = @intCast(out.skins.items.len);
                out.skins.appendAssumeCapacity(src.*);
            }
        }
        const ext = aabb_w.extents();
        const max_dim = @max(ext.x, @max(ext.y, ext.z));

        // Distant-cascade shadow LOD stand-in (render-owned snapshot of the
        // coarsest QEM-simplified child; null = fail safe to high-poly).
        const shadow_lod = types.shadowLodMesh(mesh);

        out.items.items[idx] = .{
            .vertex_buffer = mesh.vertex_buffer,
            .index_buffer = mesh.index_buffer,
            .index_count = mesh.index_count,
            .instance_buffer = staged.buffer,
            .visible_instance_count = staged.count,
            .model = model,
            .world_aabb = aabb_w,
            .max_dim = max_dim,
            .skin_index = skin_index,
            .bucket = types.bucketFor(mesh),
            .is_instanced = is_inst,
            .gpu_pending = mesh.gpu_pending or skin_oom,
            // Instanced batch: drawable iff the staged buffer holds at least
            // one matrix (visible source mesh and/or visible instances).
            .is_visible = if (is_inst) staged.count > 0 else mesh.is_visible,
            .source_uid = mesh.uid,
            .source_mesh = self.binned_source.items[idx],
            .lod_vertex_buffer = if (shadow_lod) |lm| lm.vertex_buffer else .{},
            .lod_index_buffer = if (shadow_lod) |lm| lm.index_buffer else .{},
            .lod_index_count = if (shadow_lod) |lm| lm.index_count else 0,
            .has_shadow_lod = shadow_lod != null,
        };
    }

    out.bin = binned;
    return binned;
}
