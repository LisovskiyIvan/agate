//! Mesh -> OBJ exporter (CPU geometry only, no Scene dependency).
//!
//! Exports `Mesh.cpu_positions` / `Mesh.cpu_indices` as Wavefront OBJ text.
//! Meshes without CPU geometry (empty `cpu_positions`) are skipped: glTF
//! imports and geometry builders retain it, GPU-only meshes do not.
//!
//! Normals: flat per-triangle face normals (one `vn` per triangle, faces use
//! the `v//vn` form). The STL exporter uses the same convention.
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

fn materialName(mesh: *const Mesh) ?[]const u8 {
    const mat = mesh.material orelse return null;
    return switch (mat) {
        .standard => |s| s.name,
        .pbr => |p| p.name,
    };
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

        for (tris.items) |tri| {
            const n = faceNormal(
                exportPosition(mesh, world, tri[0]),
                exportPosition(mesh, world, tri[1]),
                exportPosition(mesh, world, tri[2]),
            );
            const n_line = try std.fmt.allocPrint(allocator, "vn {d} {d} {d}\n", .{ n.x, n.y, n.z });
            defer allocator.free(n_line);
            try buf.appendSlice(allocator, n_line);
        }

        var ni: usize = 0;
        for (tris.items) |tri| {
            ni += 1;
            const f_line = try std.fmt.allocPrint(
                allocator,
                "f {d}//{d} {d}//{d} {d}//{d}\n",
                .{
                    tri[0] + base_v + 1, base_n + ni,
                    tri[1] + base_v + 1, base_n + ni,
                    tri[2] + base_v + 1, base_n + ni,
                },
            );
            defer allocator.free(f_line);
            try buf.appendSlice(allocator, f_line);
        }

        base_v += mesh.cpu_positions.len;
        base_n += tris.items.len;
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
        const raw_name = switch (mat) {
            .standard => |s| s.name,
            .pbr => |p| p.name,
        };
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

        const kd: [3]f32 = switch (mat) {
            .standard => |s| .{ s.diffuse_color.r, s.diffuse_color.g, s.diffuse_color.b },
            .pbr => |p| .{ p.albedo_color.r, p.albedo_color.g, p.albedo_color.b },
        };
        const alpha: f32 = switch (mat) {
            .standard => |s| s.alpha,
            .pbr => |p| p.alpha,
        };
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

// ---- tests (GPU-free) ----

const obj_loader = @import("../loader/obj.zig");
const MaterialModule = @import("../material.zig");
const PBRMaterial = MaterialModule.PBRMaterial;
const Color3 = math.Color3;

fn makeTriMesh(positions: []Vec3, indices: []u32) Mesh {
    return .{
        .name = "tri",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .cpu_positions = positions,
        .cpu_indices = indices,
    };
}

test "obj export round-trip single triangle" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    var list = [_]*Mesh{&mesh};
    const bytes = try writeObjAlloc(alloc, &list, .{});
    defer alloc.free(bytes);
    var data = try obj_loader.parse(alloc, bytes);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());
    try std.testing.expectEqual(@as(usize, 3), data.indices.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.positions[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.positions[7], 1e-6);
}

test "obj export round-trip quad keeps vertex and face counts" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(1, 0, 0),
        Vec3.new(1, 1, 0),
        Vec3.new(0, 1, 0),
    };
    var indices = [_]u32{ 0, 1, 2, 0, 2, 3 };
    var mesh = Mesh{
        .name = "quad",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 6,
        .cpu_positions = &positions,
        .cpu_indices = &indices,
    };
    var list = [_]*Mesh{&mesh};
    const bytes = try writeObjAlloc(alloc, &list, .{});
    defer alloc.free(bytes);
    var data = try obj_loader.parse(alloc, bytes);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4), data.vertexCount());
    try std.testing.expectEqual(@as(usize, 6), data.indices.len);
}

test "obj export skips mesh without cpu geometry" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    var bare = Mesh{
        .name = "bare",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    var list = [_]*Mesh{ &mesh, &bare };
    const bytes = try writeObjAlloc(alloc, &list, .{});
    defer alloc.free(bytes);
    var data = try obj_loader.parse(alloc, bytes);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());
    try std.testing.expectEqual(@as(usize, 3), data.indices.len);
}

test "obj export empty list yields empty output" {
    const alloc = std.testing.allocator;
    const empty: []const *Mesh = &.{};
    const bytes = try writeObjAlloc(alloc, empty, .{});
    defer alloc.free(bytes);
    try std.testing.expectEqual(@as(usize, 0), bytes.len);
}

test "obj export visible_only skips hidden meshes" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var shown = makeTriMesh(&positions, &indices);
    var hidden = Mesh{
        .name = "hidden",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .is_visible = false,
        .cpu_positions = &positions,
        .cpu_indices = &indices,
    };
    var list = [_]*Mesh{ &shown, &hidden };
    const filtered = try writeObjAlloc(alloc, &list, .{ .visible_only = true });
    defer alloc.free(filtered);
    var data = try obj_loader.parse(alloc, filtered);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());

    const unfiltered = try writeObjAlloc(alloc, &list, .{ .visible_only = false });
    defer alloc.free(unfiltered);
    var data_all = try obj_loader.parse(alloc, unfiltered);
    defer data_all.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 6), data_all.indices.len);
}

test "obj export world transform applies mesh translation" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    mesh.position = Vec3.new(10, 0, 0);
    var list = [_]*Mesh{&mesh};

    const world = try writeObjAlloc(alloc, &list, .{ .apply_world_transform = true });
    defer alloc.free(world);
    var data_w = try obj_loader.parse(alloc, world);
    defer data_w.deinit(alloc);
    try std.testing.expectApproxEqAbs(@as(f32, 10), data_w.positions[0], 1e-5);

    const local = try writeObjAlloc(alloc, &list, .{ .apply_world_transform = false });
    defer alloc.free(local);
    var data_l = try obj_loader.parse(alloc, local);
    defer data_l.deinit(alloc);
    try std.testing.expectApproxEqAbs(@as(f32, 0), data_l.positions[0], 1e-6);
}

test "obj sanitize replaces spaces and unicode bytes" {
    const alloc = std.testing.allocator;
    const ascii = try sanitizeNameAlloc(alloc, "hello world!");
    defer alloc.free(ascii);
    try std.testing.expectEqualStrings("hello_world_", ascii);

    const uni_src = "mesh Привет";
    const uni = try sanitizeNameAlloc(alloc, uni_src);
    defer alloc.free(uni);
    // 1:1 byte mapping: spaces and every UTF-8 byte become '_'.
    try std.testing.expectEqual(uni_src.len, uni.len);
    for (uni) |c| {
        try std.testing.expect(c < 128 and c != ' ');
    }

    const empty_name = try sanitizeNameAlloc(alloc, "");
    defer alloc.free(empty_name);
    try std.testing.expectEqualStrings("mesh", empty_name);

    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    mesh.name = "my mesh";
    var list = [_]*Mesh{&mesh};
    const bytes = try writeObjAlloc(alloc, &list, .{});
    defer alloc.free(bytes);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "o my_mesh\n") != null);
}

test "obj export mtl carries pbr albedo as Kd" {
    const alloc = std.testing.allocator;
    var pbr = PBRMaterial.init("red_mat");
    pbr.albedo_color = Color3.new(1, 0, 0);
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    mesh.material = .{ .pbr = &pbr };
    var list = [_]*Mesh{&mesh};

    const obj = try writeObjAlloc(alloc, &list, .{});
    defer alloc.free(obj);
    try std.testing.expect(std.mem.indexOf(u8, obj, "mtllib scene.mtl\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, obj, "usemtl red_mat\n") != null);

    const mtl = try writeMtlAlloc(alloc, &list, .{});
    defer alloc.free(mtl);
    try std.testing.expect(std.mem.indexOf(u8, mtl, "newmtl red_mat\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, mtl, "Kd 1 0 0\n") != null);
}
