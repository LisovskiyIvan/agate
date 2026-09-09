const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;

const StandardMaterial = @import("material.zig").StandardMaterial;
const PBRMaterial = @import("material.zig").PBRMaterial;
const Material = @import("material.zig").Material;
const Scene = @import("scene.zig").Scene;
const Skeleton = @import("animation/skeleton.zig").Skeleton;

pub const Vertex = extern struct {
    position: [3]f32,
    normal: [3]f32,
    color: [4]f32,
    uv: [2]f32,
    tangent: [4]f32 = .{ 1.0, 0.0, 0.0, 1.0 },
    joints: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 },
    weights: [4]f32 = .{ 1.0, 0.0, 0.0, 0.0 },
};

pub const CullingStrategy = enum {
    frustum,
    always_render,
};

pub const InstancedMesh = struct {
    name: []const u8,
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero, // Euler angles in degrees
    scaling: Vec3 = Vec3.one,
    is_visible: bool = true,
    cast_shadows: bool = true,
    receive_shadows: bool = true,
    culling_strategy: CullingStrategy = .frustum,
    source_mesh: *Mesh,

    pub fn getWorldMatrix(self: InstancedMesh) Mat4 {
        const trs = Mat4.fromRotationTranslationScale(self.position, self.rotation, self.scaling);
        return Mat4.mul(trs, self.source_mesh.base_matrix);
    }

    pub fn getWorldBoundingBox(self: InstancedMesh) BoundingBox {
        return self.source_mesh.local_bounding_box.transform(self.getWorldMatrix());
    }
};

pub const Mesh = struct {
    name: []const u8,
    owns_name: bool = false,
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero, // Euler angles in degrees
    scaling: Vec3 = Vec3.one,

    vertex_buffer: sg.Buffer,
    index_buffer: sg.Buffer,
    index_count: u32,
    index_type: sg.IndexType = .UINT16,
    material: ?Material = null,
    parent: ?*Mesh = null,
    skeleton: ?*Skeleton = null,
    base_matrix: Mat4 = Mat4.identity,

    // Culling, Shadows & Visibility
    is_visible: bool = true,
    cast_shadows: bool = true,
    receive_shadows: bool = true,
    culling_strategy: CullingStrategy = .frustum,
    local_bounding_box: BoundingBox = BoundingBox.zero,

    // Instancing support
    instances: std.ArrayListUnmanaged(*InstancedMesh) = .empty,
    instance_buffer: sg.Buffer = .{},
    instance_buffer_capacity: usize = 0,
    visible_instance_count: u32 = 0,
    // Per-frame transform cache (Scene.worldMatrixCached fills these once per render()).
    cached_matrix: Mat4 = Mat4.identity,
    cached_aabb: BoundingBox = BoundingBox.zero,
    cached_frame: u64 = std.math.maxInt(u64),
    // Instance buffer upload dedup: skip sg.updateBuffer when data unchanged.
    instance_hash: u64 = 0,
    instance_uploaded_count: usize = 0,

    pub fn createInstance(self: *Mesh, scene: *Scene, name: []const u8) !*InstancedMesh {
        const inst = try scene.allocator.create(InstancedMesh);
        inst.* = .{
            .name = name,
            .source_mesh = self,
        };
        try self.instances.append(scene.allocator, inst);
        return inst;
    }

    pub fn setStandardMaterial(self: *Mesh, mat: *StandardMaterial) void {
        self.material = .{ .standard = mat };
    }

    pub fn setPBRMaterial(self: *Mesh, mat: *PBRMaterial) void {
        self.material = .{ .pbr = mat };
    }

    pub fn getWorldMatrix(self: Mesh) Mat4 {
        const trs = Mat4.fromRotationTranslationScale(self.position, self.rotation, self.scaling);
        const local = Mat4.mul(trs, self.base_matrix);
        if (self.parent) |p| {
            return Mat4.mul(p.getWorldMatrix(), local);
        }
        return local;
    }

    pub fn getWorldBoundingBox(self: Mesh) BoundingBox {
        return self.local_bounding_box.transform(self.getWorldMatrix());
    }

    pub fn deinit(self: *Mesh, allocator: std.mem.Allocator) void {
        sg.destroyBuffer(self.vertex_buffer);
        sg.destroyBuffer(self.index_buffer);
        if (self.instance_buffer.id != 0) {
            sg.destroyBuffer(self.instance_buffer);
        }
        for (self.instances.items) |inst| {
            allocator.destroy(inst);
        }
        self.instances.deinit(allocator);
        if (self.owns_name and self.name.len > 0) {
            allocator.free(self.name);
        }
    }
};

pub const BoxOptions = struct {
    size: f32 = 1.0,
    width: ?f32 = null,
    height: ?f32 = null,
    depth: ?f32 = null,
    face_colors: ?[6]Color4 = null,
};

pub const SphereOptions = struct {
    diameter: f32 = 1.0,
    segments: u32 = 24,
    color: Color4 = Color4.white,
};

pub const GroundOptions = struct {
    width: f32 = 10.0,
    height: f32 = 10.0,
    subdivisions: u32 = 1,
    color: Color4 = Color4.white,
};

pub const CylinderOptions = struct {
    height: f32 = 2.0,
    diameter: f32 = 1.0,
    tessellation: u32 = 24,
    color: Color4 = Color4.white,
};

pub const CapsuleOptions = struct {
    radius: f32 = 0.5,
    height: f32 = 2.0, // total height including hemisphere caps
    tessellation: u32 = 16, // radial slices
    cap_subdivisions: u32 = 8, // rings per cap
    color: Color4 = Color4.white,
};

pub fn computeTangents(vertices: []Vertex, indices: ?[]const u32, indices16: ?[]const u16) void {
    for (vertices) |*v| {
        v.tangent = .{ 0, 0, 0, 1 };
    }

    const num_indices = if (indices) |idx| idx.len else if (indices16) |idx16| idx16.len else vertices.len;
    var tri_i: usize = 0;
    while (tri_i + 2 < num_indices) : (tri_i += 3) {
        const idx0: usize = if (indices) |idx| idx[tri_i] else if (indices16) |idx16| idx16[tri_i] else tri_i;
        const idx1: usize = if (indices) |idx| idx[tri_i + 1] else if (indices16) |idx16| idx16[tri_i + 1] else tri_i + 1;
        const idx2: usize = if (indices) |idx| idx[tri_i + 2] else if (indices16) |idx16| idx16[tri_i + 2] else tri_i + 2;

        if (idx0 >= vertices.len or idx1 >= vertices.len or idx2 >= vertices.len) continue;

        const v0 = vertices[idx0];
        const v1 = vertices[idx1];
        const v2 = vertices[idx2];

        const edge1 = Vec3.new(v1.position[0] - v0.position[0], v1.position[1] - v0.position[1], v1.position[2] - v0.position[2]);
        const edge2 = Vec3.new(v2.position[0] - v0.position[0], v2.position[1] - v0.position[1], v2.position[2] - v0.position[2]);

        const delta_u1 = v1.uv[0] - v0.uv[0];
        const delta_v1 = v1.uv[1] - v0.uv[1];
        const delta_u2 = v2.uv[0] - v0.uv[0];
        const delta_v2 = v2.uv[1] - v0.uv[1];

        const det = delta_u1 * delta_v2 - delta_u2 * delta_v1;
        if (@abs(det) > 1e-6) {
            const r = 1.0 / det;
            const tangent = Vec3.new(
                (edge1.x * delta_v2 - edge2.x * delta_v1) * r,
                (edge1.y * delta_v2 - edge2.y * delta_v1) * r,
                (edge1.z * delta_v2 - edge2.z * delta_v1) * r,
            );

            vertices[idx0].tangent[0] += tangent.x;
            vertices[idx0].tangent[1] += tangent.y;
            vertices[idx0].tangent[2] += tangent.z;

            vertices[idx1].tangent[0] += tangent.x;
            vertices[idx1].tangent[1] += tangent.y;
            vertices[idx1].tangent[2] += tangent.z;

            vertices[idx2].tangent[0] += tangent.x;
            vertices[idx2].tangent[1] += tangent.y;
            vertices[idx2].tangent[2] += tangent.z;
        }
    }

    for (vertices) |*v| {
        const n = Vec3.new(v.normal[0], v.normal[1], v.normal[2]);
        var t = Vec3.new(v.tangent[0], v.tangent[1], v.tangent[2]);

        if (t.lengthSq() < 1e-6) {
            t = if (@abs(n.x) > 0.9) Vec3.new(0, 1, 0) else Vec3.new(1, 0, 0);
        }

        const t_proj = t.sub(n.scale(n.dot(t)));
        if (t_proj.lengthSq() > 1e-6) {
            const t_norm = t_proj.normalize();
            v.tangent[0] = t_norm.x;
            v.tangent[1] = t_norm.y;
            v.tangent[2] = t_norm.z;
            v.tangent[3] = 1.0;
        } else {
            v.tangent = .{ 1.0, 0.0, 0.0, 1.0 };
        }
    }
}

pub const MeshBuilder = struct {
    pub fn createBox(scene: *Scene, name: []const u8, options: BoxOptions) !*Mesh {
        const w = (options.width orelse options.size) * 0.5;
        const h = (options.height orelse options.size) * 0.5;
        const d = (options.depth orelse options.size) * 0.5;

        // Default vibrant face colors if not provided
        const colors = options.face_colors orelse [6]Color4{
            Color4.new(1.0, 0.2, 0.2, 1.0), // Front: Red
            Color4.new(0.2, 1.0, 0.2, 1.0), // Back: Green
            Color4.new(0.2, 0.4, 1.0, 1.0), // Left: Blue
            Color4.new(1.0, 0.6, 0.1, 1.0), // Right: Orange
            Color4.new(0.9, 0.2, 0.8, 1.0), // Top: Magenta
            Color4.new(0.2, 0.8, 0.9, 1.0), // Bottom: Cyan
        };

        var vertices: [24]Vertex = undefined;
        var vi: usize = 0;

        const Quad = struct {
            p: [4][3]f32,
            n: [3]f32,
            t: [4]f32,
            c: [4]f32,
        };

        const quads = [_]Quad{
            // Front (+Z)
            .{
                .p = .{ .{ -w, -h, d }, .{ w, -h, d }, .{ w, h, d }, .{ -w, h, d } },
                .n = .{ 0, 0, 1 },
                .t = .{ 1, 0, 0, 1 },
                .c = colors[0].toArray(),
            },
            // Back (-Z)
            .{
                .p = .{ .{ w, -h, -d }, .{ -w, -h, -d }, .{ -w, h, -d }, .{ w, h, -d } },
                .n = .{ 0, 0, -1 },
                .t = .{ -1, 0, 0, 1 },
                .c = colors[1].toArray(),
            },
            // Left (-X)
            .{
                .p = .{ .{ -w, -h, -d }, .{ -w, -h, d }, .{ -w, h, d }, .{ -w, h, -d } },
                .n = .{ -1, 0, 0 },
                .t = .{ 0, 0, 1, 1 },
                .c = colors[2].toArray(),
            },
            // Right (+X)
            .{
                .p = .{ .{ w, -h, d }, .{ w, -h, -d }, .{ w, h, -d }, .{ w, h, d } },
                .n = .{ 1, 0, 0 },
                .t = .{ 0, 0, -1, 1 },
                .c = colors[3].toArray(),
            },
            // Top (+Y)
            .{
                .p = .{ .{ -w, h, d }, .{ w, h, d }, .{ w, h, -d }, .{ -w, h, -d } },
                .n = .{ 0, 1, 0 },
                .t = .{ 1, 0, 0, 1 },
                .c = colors[4].toArray(),
            },
            // Bottom (-Y)
            .{
                .p = .{ .{ -w, -h, -d }, .{ w, -h, -d }, .{ w, -h, d }, .{ -w, -h, d } },
                .n = .{ 0, -1, 0 },
                .t = .{ 1, 0, 0, 1 },
                .c = colors[5].toArray(),
            },
        };

        const uvs = [4][2]f32{
            .{ 0.0, 0.0 },
            .{ 1.0, 0.0 },
            .{ 1.0, 1.0 },
            .{ 0.0, 1.0 },
        };

        for (quads) |q| {
            for (0..4) |i| {
                vertices[vi] = .{
                    .position = q.p[i],
                    .normal = q.n,
                    .color = q.c,
                    .uv = uvs[i],
                    .tangent = q.t,
                };
                vi += 1;
            }
        }

        var indices: [36]u16 = undefined;
        var ii: usize = 0;
        for (0..6) |face| {
            const base: u16 = @intCast(face * 4);
            indices[ii + 0] = base + 0;
            indices[ii + 1] = base + 1;
            indices[ii + 2] = base + 2;
            indices[ii + 3] = base + 0;
            indices[ii + 4] = base + 2;
            indices[ii + 5] = base + 3;
            ii += 6;
        }

        const vbuf = sg.makeBuffer(.{
            .data = sg.asRange(&vertices),
        });

        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&indices),
        });

        const mesh = try scene.allocator.create(Mesh);
        mesh.* = .{
            .name = name,
            .vertex_buffer = vbuf,
            .index_buffer = ibuf,
            .index_count = 36,
            .local_bounding_box = BoundingBox.init(
                Vec3.new(-w, -h, -d),
                Vec3.new(w, h, d),
            ),
        };

        try scene.meshes.append(scene.allocator, mesh);
        return mesh;
    }

    pub fn createGround(scene: *Scene, name: []const u8, options: GroundOptions) !*Mesh {
        const half_w = options.width * 0.5;
        const half_h = options.height * 0.5;
        const subs = @max(1, options.subdivisions);

        const vert_count = (subs + 1) * (subs + 1);
        const index_count = subs * subs * 6;

        const vertices = try scene.allocator.alloc(Vertex, vert_count);
        defer scene.allocator.free(vertices);

        const indices = try scene.allocator.alloc(u16, index_count);
        defer scene.allocator.free(indices);

        var vi: usize = 0;
        for (0..subs + 1) |iz| {
            const fz = @as(f32, @floatFromInt(iz)) / @as(f32, @floatFromInt(subs));
            const z = -half_h + fz * options.height;
            for (0..subs + 1) |ix| {
                const fx = @as(f32, @floatFromInt(ix)) / @as(f32, @floatFromInt(subs));
                const x = -half_w + fx * options.width;
                vertices[vi] = .{
                    .position = .{ x, 0.0, z },
                    .normal = .{ 0.0, 1.0, 0.0 },
                    .color = options.color.toArray(),
                    .uv = .{ fx, fz },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
                vi += 1;
            }
        }

        var ii: usize = 0;
        const row_stride = subs + 1;
        for (0..subs) |iz| {
            for (0..subs) |ix| {
                const p0: u16 = @intCast(iz * row_stride + ix);
                const p1: u16 = @intCast(iz * row_stride + (ix + 1));
                const p2: u16 = @intCast((iz + 1) * row_stride + (ix + 1));
                const p3: u16 = @intCast((iz + 1) * row_stride + ix);

                // CCW winding for +Y normal: (p0, p2, p1) and (p0, p3, p2)
                indices[ii + 0] = p0;
                indices[ii + 1] = p2;
                indices[ii + 2] = p1;
                indices[ii + 3] = p0;
                indices[ii + 4] = p3;
                indices[ii + 5] = p2;
                ii += 6;
            }
        }

        const vbuf = sg.makeBuffer(.{
            .data = sg.asRange(vertices),
        });

        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(indices),
        });

        const mesh = try scene.allocator.create(Mesh);
        mesh.* = .{
            .name = name,
            .vertex_buffer = vbuf,
            .index_buffer = ibuf,
            .index_count = @intCast(index_count),
            .local_bounding_box = BoundingBox.init(
                Vec3.new(-half_w, 0.0, -half_h),
                Vec3.new(half_w, 0.0, half_h),
            ),
        };

        try scene.meshes.append(scene.allocator, mesh);
        return mesh;
    }

    pub fn createSphere(scene: *Scene, name: []const u8, options: SphereOptions) !*Mesh {
        const radius = options.diameter * 0.5;
        const segs = @max(4, options.segments);
        const rings = segs;
        const slices = segs;

        const vert_count = (rings + 1) * (slices + 1);
        const index_count = rings * slices * 6;

        const vertices = try scene.allocator.alloc(Vertex, vert_count);
        defer scene.allocator.free(vertices);

        const indices = try scene.allocator.alloc(u16, index_count);
        defer scene.allocator.free(indices);

        const pi = std.math.pi;
        var vi: usize = 0;
        for (0..rings + 1) |r| {
            const v = @as(f32, @floatFromInt(r)) / @as(f32, @floatFromInt(rings));
            const phi = -pi * 0.5 + pi * v;
            const cos_phi = @cos(phi);
            const sin_phi = @sin(phi);

            for (0..slices + 1) |s| {
                const u = @as(f32, @floatFromInt(s)) / @as(f32, @floatFromInt(slices));
                const theta = 2.0 * pi * u;
                const cos_theta = @cos(theta);
                const sin_theta = @sin(theta);

                const nx = cos_phi * sin_theta;
                const ny = sin_phi;
                const nz = cos_phi * cos_theta;

                vertices[vi] = .{
                    .position = .{ nx * radius, ny * radius, nz * radius },
                    .normal = .{ nx, ny, nz },
                    .color = options.color.toArray(),
                    .uv = .{ u, v },
                    .tangent = .{ cos_theta, 0.0, -sin_theta, 1.0 },
                };
                vi += 1;
            }
        }

        var ii: usize = 0;
        const slice_stride = slices + 1;
        for (0..rings) |r| {
            for (0..slices) |s| {
                const p0: u16 = @intCast(r * slice_stride + s);
                const p1: u16 = @intCast(r * slice_stride + (s + 1));
                const p2: u16 = @intCast((r + 1) * slice_stride + (s + 1));
                const p3: u16 = @intCast((r + 1) * slice_stride + s);

                // CCW winding
                indices[ii + 0] = p0;
                indices[ii + 1] = p1;
                indices[ii + 2] = p2;
                indices[ii + 3] = p0;
                indices[ii + 4] = p2;
                indices[ii + 5] = p3;
                ii += 6;
            }
        }

        const vbuf = sg.makeBuffer(.{
            .data = sg.asRange(vertices),
        });

        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(indices),
        });

        const mesh = try scene.allocator.create(Mesh);
        mesh.* = .{
            .name = name,
            .vertex_buffer = vbuf,
            .index_buffer = ibuf,
            .index_count = @intCast(index_count),
            .local_bounding_box = BoundingBox.init(
                Vec3.new(-radius, -radius, -radius),
                Vec3.new(radius, radius, radius),
            ),
        };

        try scene.meshes.append(scene.allocator, mesh);
        return mesh;
    }

    pub fn createCylinder(scene: *Scene, name: []const u8, options: CylinderOptions) !*Mesh {
        const radius = options.diameter * 0.5;
        const half_h = options.height * 0.5;
        const tess = @max(3, options.tessellation);

        // Side vertices: (tess + 1) * 2
        // Top cap vertices: tess + 2
        // Bottom cap vertices: tess + 2
        const side_verts = (tess + 1) * 2;
        const cap_verts = (tess + 2) * 2;
        const total_verts = side_verts + cap_verts;

        const side_indices = tess * 6;
        const cap_indices = tess * 3 * 2;
        const total_indices = side_indices + cap_indices;

        const vertices = try scene.allocator.alloc(Vertex, total_verts);
        defer scene.allocator.free(vertices);

        const indices = try scene.allocator.alloc(u16, total_indices);
        defer scene.allocator.free(indices);

        const pi = std.math.pi;
        var vi: usize = 0;

        // 1. Sides
        for (0..tess + 1) |i| {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(tess));
            const theta = 2.0 * pi * u;
            const nx = @sin(theta);
            const nz = @cos(theta);

            // Bottom
            vertices[vi] = .{
                .position = .{ nx * radius, -half_h, nz * radius },
                .normal = .{ nx, 0.0, nz },
                .color = options.color.toArray(),
                .uv = .{ u, 0.0 },
                .tangent = .{ nz, 0.0, -nx, 1.0 },
            };
            vi += 1;

            // Top
            vertices[vi] = .{
                .position = .{ nx * radius, half_h, nz * radius },
                .normal = .{ nx, 0.0, nz },
                .color = options.color.toArray(),
                .uv = .{ u, 1.0 },
                .tangent = .{ nz, 0.0, -nx, 1.0 },
            };
            vi += 1;
        }

        var ii: usize = 0;
        for (0..tess) |i| {
            const b0: u16 = @intCast(i * 2);
            const t0: u16 = @intCast(i * 2 + 1);
            const b1: u16 = @intCast((i + 1) * 2);
            const t1: u16 = @intCast((i + 1) * 2 + 1);

            indices[ii + 0] = b0;
            indices[ii + 1] = b1;
            indices[ii + 2] = t1;
            indices[ii + 3] = b0;
            indices[ii + 4] = t1;
            indices[ii + 5] = t0;
            ii += 6;
        }

        // 2. Top cap
        const top_center_idx: u16 = @intCast(vi);
        vertices[vi] = .{
            .position = .{ 0.0, half_h, 0.0 },
            .normal = .{ 0.0, 1.0, 0.0 },
            .color = options.color.toArray(),
            .uv = .{ 0.5, 0.5 },
            .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
        };
        vi += 1;

        const top_ring_start: u16 = @intCast(vi);
        for (0..tess + 1) |i| {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(tess));
            const theta = 2.0 * pi * u;
            const x = @sin(theta);
            const z = @cos(theta);
            vertices[vi] = .{
                .position = .{ x * radius, half_h, z * radius },
                .normal = .{ 0.0, 1.0, 0.0 },
                .color = options.color.toArray(),
                .uv = .{ 0.5 + x * 0.5, 0.5 + z * 0.5 },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }

        for (0..tess) |i| {
            indices[ii + 0] = top_center_idx;
            indices[ii + 1] = top_ring_start + @as(u16, @intCast(i));
            indices[ii + 2] = top_ring_start + @as(u16, @intCast(i + 1));
            ii += 3;
        }

        // 3. Bottom cap
        const bot_center_idx: u16 = @intCast(vi);
        vertices[vi] = .{
            .position = .{ 0.0, -half_h, 0.0 },
            .normal = .{ 0.0, -1.0, 0.0 },
            .color = options.color.toArray(),
            .uv = .{ 0.5, 0.5 },
            .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
        };
        vi += 1;

        const bot_ring_start: u16 = @intCast(vi);
        for (0..tess + 1) |i| {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(tess));
            const theta = 2.0 * pi * u;
            const x = @sin(theta);
            const z = @cos(theta);
            vertices[vi] = .{
                .position = .{ x * radius, -half_h, z * radius },
                .normal = .{ 0.0, -1.0, 0.0 },
                .color = options.color.toArray(),
                .uv = .{ 0.5 + x * 0.5, 0.5 + z * 0.5 },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }

        for (0..tess) |i| {
            indices[ii + 0] = bot_center_idx;
            indices[ii + 1] = bot_ring_start + @as(u16, @intCast(i + 1));
            indices[ii + 2] = bot_ring_start + @as(u16, @intCast(i));
            ii += 3;
        }

        const vbuf = sg.makeBuffer(.{
            .data = sg.asRange(vertices),
        });

        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(indices),
        });

        const mesh = try scene.allocator.create(Mesh);
        mesh.* = .{
            .name = name,
            .vertex_buffer = vbuf,
            .index_buffer = ibuf,
            .index_count = @intCast(total_indices),
            .local_bounding_box = BoundingBox.init(
                Vec3.new(-radius, -half_h, -radius),
                Vec3.new(radius, half_h, radius),
            ),
        };

        try scene.meshes.append(scene.allocator, mesh);
        return mesh;
    }

    pub fn createCapsule(scene: *Scene, name: []const u8, options: CapsuleOptions) !*Mesh {
        const radius = options.radius;
        const total_height = @max(options.height, radius * 2.0);
        const half_h = (total_height - radius * 2.0) * 0.5;
        const slices = @max(4, options.tessellation);
        const cap_rings = @max(2, options.cap_subdivisions);

        // Ring layout from bottom to top:
        // - Bottom hemisphere: cap_rings rings (from south pole to bottom equator)
        // - Top of cylinder: 1 ring (bottom equator is at y = -half_h, top of cylinder is at y = +half_h)
        // - Top hemisphere: cap_rings rings (from top equator to north pole)
        // Total rings = (cap_rings + 1) + 1 + cap_rings = 2 * cap_rings + 2
        const total_rings = 2 * cap_rings + 2;
        const vert_count = total_rings * (slices + 1);
        const quad_rows = total_rings - 1;
        const index_count = quad_rows * slices * 6;

        const vertices = try scene.allocator.alloc(Vertex, vert_count);
        defer scene.allocator.free(vertices);

        const indices = try scene.allocator.alloc(u16, index_count);
        defer scene.allocator.free(indices);

        const pi = std.math.pi;
        var vi: usize = 0;

        for (0..total_rings) |r| {
            var phi: f32 = 0.0;
            var center_y: f32 = 0.0;

            if (r <= cap_rings) {
                // Bottom hemisphere: phi in [-pi/2, 0]
                const t = @as(f32, @floatFromInt(r)) / @as(f32, @floatFromInt(cap_rings));
                phi = -pi * 0.5 + t * (pi * 0.5);
                center_y = -half_h;
            } else if (r == cap_rings + 1) {
                // Top of cylinder body: phi = 0
                phi = 0.0;
                center_y = half_h;
            } else {
                // Top hemisphere: phi in [0, pi/2]
                const t = @as(f32, @floatFromInt(r - cap_rings - 1)) / @as(f32, @floatFromInt(cap_rings));
                phi = t * (pi * 0.5);
                center_y = half_h;
            }

            const cos_phi = @cos(phi);
            const sin_phi = @sin(phi);
            const v = @as(f32, @floatFromInt(r)) / @as(f32, @floatFromInt(quad_rows));

            for (0..slices + 1) |s| {
                const u = @as(f32, @floatFromInt(s)) / @as(f32, @floatFromInt(slices));
                const theta = 2.0 * pi * u;
                const cos_theta = @cos(theta);
                const sin_theta = @sin(theta);

                const nx = cos_phi * sin_theta;
                const ny = sin_phi;
                const nz = cos_phi * cos_theta;

                vertices[vi] = .{
                    .position = .{ nx * radius, center_y + ny * radius, nz * radius },
                    .normal = .{ nx, ny, nz },
                    .color = options.color.toArray(),
                    .uv = .{ u, v },
                    .tangent = .{ cos_theta, 0.0, -sin_theta, 1.0 },
                };
                vi += 1;
            }
        }

        var ii: usize = 0;
        const slice_stride: u16 = @intCast(slices + 1);
        for (0..quad_rows) |r| {
            const r_u16: u16 = @intCast(r);
            for (0..slices) |s| {
                const s_u16: u16 = @intCast(s);
                const p0 = r_u16 * slice_stride + s_u16;
                const p1 = r_u16 * slice_stride + (s_u16 + 1);
                const p2 = (r_u16 + 1) * slice_stride + (s_u16 + 1);
                const p3 = (r_u16 + 1) * slice_stride + s_u16;

                // CCW winding
                indices[ii + 0] = p0;
                indices[ii + 1] = p1;
                indices[ii + 2] = p2;
                indices[ii + 3] = p0;
                indices[ii + 4] = p2;
                indices[ii + 5] = p3;
                ii += 6;
            }
        }

        const vbuf = sg.makeBuffer(.{
            .data = sg.asRange(vertices),
        });

        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(indices),
        });

        const total_half = total_height * 0.5;
        const mesh = try scene.allocator.create(Mesh);
        mesh.* = .{
            .name = name,
            .vertex_buffer = vbuf,
            .index_buffer = ibuf,
            .index_count = @intCast(index_count),
            .local_bounding_box = BoundingBox.init(
                Vec3.new(-radius, -total_half, -radius),
                Vec3.new(radius, total_half, radius),
            ),
        };

        try scene.meshes.append(scene.allocator, mesh);
        return mesh;
    }
};
