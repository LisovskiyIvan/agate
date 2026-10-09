//! Mesh -> OBJ exporter (CPU geometry only, no Scene dependency).
//!
//! Exports `Mesh.cpu_positions` / `Mesh.cpu_indices` as Wavefront OBJ text.
//! Meshes without CPU geometry (empty `cpu_positions`) are skipped: glTF
//! imports and geometry builders retain it, GPU-only meshes do not.
//!
//! Normals: flat per-triangle face normals; byte-identical normals share one
//! `vn` line so coplanar faces keep the importer from splitting shared
//! corners (faces use the `v//vn` form). The STL exporter uses the same
//! convention.
//! UVs are dropped (no `vt`); the OBJ importer defaults them to zero, so a
//! round-trip stays lossless on positions and triangle counts.
//! Materials: with `emit_materials`, meshes carrying a Standard or PBR
//! material get `usemtl` lines plus a single `mtllib scene.mtl` header; the
//! MTL payload itself comes from `writeMtlAlloc` (diffuse/albedo as `Kd`,
//! alpha as `d`; metallic/roughness/emissive/textures are NOT exported).
//!
//! Transforms: `apply_world_transform == false` exports raw CPU coordinates.
//! When true, positions go through `mesh.getWorldMatrix()` (TRS composed
//! with base_matrix, parents and bone attachments). Skinning and morph
//! weights are NOT applied: skinned/morphed meshes export in bind/base pose,
//! and normals under non-uniform scale are direction-transformed (exact for
//! rigid and uniform transforms only).
//! Out-of-range indices drop the triangle; a trailing index tail (< 3) is
//! ignored. Meshes with empty `cpu_indices` export as a triangle soup over
//! `cpu_positions` (trailing vertices < 3 are ignored).

const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;

const Mesh = @import("../mesh.zig").Mesh;

/// Name of the companion material file referenced by the `mtllib` line.
/// Save `writeMtlAlloc` output under this name next to the `.obj` file.
pub const mtl_filename: []const u8 = "scene.mtl";

pub const ObjExportOptions = struct {
    visible_only: bool = false,
    apply_world_transform: bool = false,
    emit_materials: bool = true,
};

/// Sanitizes an object/material name for OBJ/MTL output: every byte outside
/// `[A-Za-z0-9._-]` (spaces, controls, all non-ASCII bytes including UTF-8)
/// becomes `_`. Empty input becomes `"mesh"`. Length is preserved, so
/// distinct names usually stay distinct (collisions merge silently).
pub fn sanitizeNameAlloc(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    if (name.len == 0) return allocator.dupe(u8, "mesh");
    const out = try allocator.alloc(u8, name.len);
    errdefer allocator.free(out);
    for (name, 0..) |c, i| {
        const ok = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or c == '.' or c == '_' or c == '-';
        out[i] = if (ok) c else '_';
    }
    return out;
}

fn shouldExport(mesh: *const Mesh, options: ObjExportOptions) bool {
    if (options.visible_only and !mesh.is_visible) return false;
    if (mesh.cpu_positions.len == 0) return false;
    return true;
}

fn exportPosition(mesh: *const Mesh, world: ?Mat4, index: usize) Vec3 {
    const p = mesh.cpu_positions[index];
    return if (world) |m| m.transformPoint(p) else p;
}

/// Flat face normal with an up fallback for degenerate triangles
/// (mirrors the OBJ importer's fallback).
fn faceNormal(a: Vec3, b: Vec3, c: Vec3) Vec3 {
    const n = b.sub(a).cross(c.sub(a));
    if (n.lengthSq() < 1e-12) return Vec3.up;
    return n.normalize();
}

/// Quantizes a normal component for vn dedup (1e-5, as in the STL importer).
/// NaN maps to 0 so degenerate input cannot trap @intFromFloat.
fn quantNormal(x: f32) i32 {
    if (std.math.isNan(x)) return 0;
    return @intFromFloat(@round(x * 100000.0));
}

fn materialName(mesh: *const Mesh) ?[]const u8 {
    const mat = mesh.material orelse return null;
    return mat.name();
}

/// Exports meshes as OBJ text (owned, caller frees with allocator).
/// Empty input (or nothing exportable) yields an empty slice.
pub fn writeObjAlloc(
    allocator: std.mem.Allocator,
    meshes: []const *Mesh,
    options: ObjExportOptions,
) ![]u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    if (options.emit_materials) {
        var need_mtl = false;
        for (meshes) |mesh| {
            if (!shouldExport(mesh, options)) continue;
            if (mesh.material != null) {
                need_mtl = true;
                break;
            }
        }
        if (need_mtl) {
            const header = try std.fmt.allocPrint(allocator, "mtllib {s}\n", .{mtl_filename});
            defer allocator.free(header);
            try buf.appendSlice(allocator, header);
        }
    }

    // Global 1-based bases: OBJ indices are file-wide, not per object.
    var base_v: usize = 0;
    var base_n: usize = 0;
    for (meshes) |mesh| {
        if (!shouldExport(mesh, options)) continue;

        const clean = try sanitizeNameAlloc(allocator, mesh.name);
        defer allocator.free(clean);
        const o_line = try std.fmt.allocPrint(allocator, "o {s}\n", .{clean});
        defer allocator.free(o_line);
        try buf.appendSlice(allocator, o_line);

        if (options.emit_materials) {
            if (materialName(mesh)) |raw| {
                const mat_clean = try sanitizeNameAlloc(allocator, raw);
                defer allocator.free(mat_clean);
                const u_line = try std.fmt.allocPrint(allocator, "usemtl {s}\n", .{mat_clean});
                defer allocator.free(u_line);
                try buf.appendSlice(allocator, u_line);
            }
        }

        const world: ?Mat4 = if (options.apply_world_transform) mesh.getWorldMatrix() else null;

        for (0..mesh.cpu_positions.len) |vi| {
            const p = exportPosition(mesh, world, vi);
            const v_line = try std.fmt.allocPrint(allocator, "v {d} {d} {d}\n", .{ p.x, p.y, p.z });
            defer allocator.free(v_line);
            try buf.appendSlice(allocator, v_line);
        }

        // Collect valid triangles: indexed triples, or a soup when the mesh
        // carries no indices. Out-of-range indices drop the triangle.
        var tris: std.ArrayListUnmanaged([3]u32) = .empty;
        defer tris.deinit(allocator);
        if (mesh.cpu_indices.len > 0) {
            var t: usize = 0;
            while (t + 2 < mesh.cpu_indices.len) : (t += 3) {
                const a = mesh.cpu_indices[t];
                const b = mesh.cpu_indices[t + 1];
                const c = mesh.cpu_indices[t + 2];
                if (a >= mesh.cpu_positions.len or b >= mesh.cpu_positions.len or c >= mesh.cpu_positions.len) continue;
                try tris.append(allocator, .{ a, b, c });
            }
        } else {
            var v: usize = 0;
            while (v + 2 < mesh.cpu_positions.len) : (v += 3) {
                try tris.append(allocator, .{ @intCast(v), @intCast(v + 1), @intCast(v + 2) });
            }
        }

        // vn dedup: identical normals share one line. The OBJ importer keys
        // vertices on (position, uv, normal), so repeating the same normal
        // per face would split corners shared by coplanar faces and inflate
        // the round-trip vertex count.
        var n_cache: std.AutoHashMapUnmanaged([3]i32, usize) = .empty;
        defer n_cache.deinit(allocator);
        var face_ni: std.ArrayListUnmanaged(usize) = .empty;
        defer face_ni.deinit(allocator);

        for (tris.items) |tri| {
            const n = faceNormal(
                exportPosition(mesh, world, tri[0]),
                exportPosition(mesh, world, tri[1]),
                exportPosition(mesh, world, tri[2]),
            );
            const key = [3]i32{ quantNormal(n.x), quantNormal(n.y), quantNormal(n.z) };
            const gop = try n_cache.getOrPut(allocator, key);
            if (!gop.found_existing) {
                // Key is already inserted here, so count()-1 is its index.
                gop.value_ptr.* = n_cache.count() - 1;
                const n_line = try std.fmt.allocPrint(allocator, "vn {d} {d} {d}\n", .{ n.x, n.y, n.z });
                defer allocator.free(n_line);
                try buf.appendSlice(allocator, n_line);
            }
            try face_ni.append(allocator, gop.value_ptr.*);
        }

        var ni: usize = 0;
        for (tris.items) |tri| {
            ni += 1;
            const nidx = base_n + face_ni.items[ni - 1] + 1;
            const f_line = try std.fmt.allocPrint(
                allocator,
                "f {d}//{d} {d}//{d} {d}//{d}\n",
                .{
                    tri[0] + base_v + 1, nidx,
                    tri[1] + base_v + 1, nidx,
                    tri[2] + base_v + 1, nidx,
                },
            );
            defer allocator.free(f_line);
            try buf.appendSlice(allocator, f_line);
        }

        base_v += mesh.cpu_positions.len;
        base_n += n_cache.count();
    }

    return buf.toOwnedSlice(allocator);
}

/// Exports the MTL companion for `writeObjAlloc` (owned, caller frees).
/// One `newmtl` per distinct sanitized material name (first wins on
/// collision), with `Kd` from diffuse/albedo and `d` from alpha.
/// Meshes without materials contribute nothing; empty input yields empty.
pub fn writeMtlAlloc(
    allocator: std.mem.Allocator,
    meshes: []const *Mesh,
    options: ObjExportOptions,
) ![]u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    var seen: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (seen.items) |s| allocator.free(s);
        seen.deinit(allocator);
    }

    for (meshes) |mesh| {
        if (!shouldExport(mesh, options)) continue;
        const mat = mesh.material orelse continue;
        const raw_name = mat.name();
        const clean = try sanitizeNameAlloc(allocator, raw_name);
        errdefer allocator.free(clean);
        var dup = false;
        for (seen.items) |s| {
            if (std.mem.eql(u8, s, clean)) {
                dup = true;
                break;
            }
        }
        if (dup) {
            allocator.free(clean);
            continue;
        }

        const base = mat.baseColor3();
        const kd: [3]f32 = .{ base.r, base.g, base.b };
        const alpha: f32 = mat.alpha();
        const block = try std.fmt.allocPrint(
            allocator,
            "newmtl {s}\nKd {d} {d} {d}\nd {d}\n",
            .{ clean, kd[0], kd[1], kd[2], alpha },
        );
        defer allocator.free(block);
        try buf.appendSlice(allocator, block);
        // Last fallible op: ownership of clean moves to seen, which disarms
        // the errdefer above (nothing after this point can fail).
        try seen.append(allocator, clean);
    }

    return buf.toOwnedSlice(allocator);
}
