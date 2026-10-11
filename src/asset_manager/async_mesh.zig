//! Async mesh parsing support. Split out of `asset_manager.zig`.
//!
//! - `AsyncMeshKind` / `asyncKindForPath`: worker-safe kinds (OBJ/STL/PLY;
//!   GLB/GLTF stay sync-only) selected by file extension.
//! - `parseMeshBytesToGeometry`: raw bytes to owned CPU geometry (worker
//!   side; mirrors the sync scene-less branch exactly).
//! - `AsyncMeshJob`: heap handoff between the `TaskRunner` worker and the
//!   caller thread (named by `AssetTask.async_job` in `task.zig`).
//!
//! The worker entry (`asyncMeshRun`) and the test-only hold gate stay
//! defined in the `asset_manager.zig` facade: tests take
//! `&asset_manager.async_test_hold`, so the gate must keep definition
//! identity at that path, and its sole reader stays with it.

const std = @import("std");

const mesh_mod = @import("../mesh.zig");
const mesh_cpu = @import("mesh_cpu.zig");
const obj_loader = @import("../loader/obj.zig");
const stl_loader = @import("../loader/stl.zig");
const ply_loader = @import("../loader/ply.zig");

const GeometryData = mesh_mod.GeometryData;
const buildCpuGeometry = mesh_cpu.buildCpuGeometry;

/// Worker-safe mesh kinds for the async path: pure CPU parse with no Scene,
/// no `sg.*`, no cache. GLB/GLTF are excluded (cgltf + materials/textures +
/// sg work is not worker-safe) and stay sync-only.
const AsyncMeshKind = enum { obj, stl, ply };

pub fn asyncKindForPath(path: []const u8) ?AsyncMeshKind {
    if (std.mem.endsWith(u8, path, ".obj")) return .obj;
    if (std.mem.endsWith(u8, path, ".stl")) return .stl;
    if (std.mem.endsWith(u8, path, ".ply")) return .ply;
    return null;
}

/// Parses raw file bytes into owned CPU geometry (worker side). Mirrors the
/// sync scene-less branch exactly (same loaders, same `buildCpuGeometry`
/// inputs) so async and sync scene-less results are identical. No Scene, no
/// `sg.*`, no cache — allocator must be thread-safe.
pub fn parseMeshBytesToGeometry(allocator: std.mem.Allocator, kind: AsyncMeshKind, bytes: []const u8) !GeometryData {
    switch (kind) {
        .obj => {
            var data = try obj_loader.parse(allocator, bytes);
            defer data.deinit(allocator);
            return buildCpuGeometry(allocator, data.positions, data.normals, data.uvs, null, data.indices);
        },
        .stl => {
            var data = try stl_loader.parse(allocator, bytes);
            defer data.deinit(allocator);
            const n = data.vertex_count;
            const uvs = try allocator.alloc(f32, n * 2);
            defer allocator.free(uvs);
            @memset(uvs, 0);
            return buildCpuGeometry(allocator, data.positions, data.normals, uvs, null, data.indices);
        },
        .ply => {
            var data = try ply_loader.parse(allocator, bytes);
            defer data.deinit(allocator);
            return buildCpuGeometry(
                allocator,
                data.positions,
                data.normals,
                data.uvs,
                if (data.has_colors) data.colors else null,
                data.indices,
            );
        },
    }
}
/// Heap handoff between the `TaskRunner` worker and the caller thread. The
/// worker writes `worker_id`, then `started` (release), then parses into
/// `result`/`err`, then `done` (release). The caller reads `result`/`err`
/// only after `done == true` (acquire). The worker never touches the
/// `AssetTask` or the `Scene`; `path` bytes are borrowed from the caller
/// (same lifetime rule as `AssetTask.path`).
pub const AsyncMeshJob = struct {
    kind: AsyncMeshKind,
    /// BORROWED path bytes (caller keeps alive until reset/deinit).
    path: []const u8,
    allocator: std.mem.Allocator,
    started: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    worker_id: ?std.Thread.Id = null,
    result: ?GeometryData = null,
    err: ?anyerror = null,
};
