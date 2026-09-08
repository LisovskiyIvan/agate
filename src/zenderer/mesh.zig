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

pub const Vertex = extern struct {
    position: [3]f32,
    normal: [3]f32,
    color: [4]f32,
    uv: [2]f32,
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
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero, // Euler angles in degrees
    scaling: Vec3 = Vec3.one,

    vertex_buffer: sg.Buffer,
    index_buffer: sg.Buffer,
    index_count: u32,
    index_type: sg.IndexType = .UINT16,
    material: ?Material = null,
    parent: ?*Mesh = null,
    base_matrix: Mat4 = Mat4.identity,

    // Culling & Visibility
    is_visible: bool = true,
    culling_strategy: CullingStrategy = .frustum,
    local_bounding_box: BoundingBox = BoundingBox.zero,

    // Instancing support
    instances: std.ArrayListUnmanaged(*InstancedMesh) = .empty,
    instance_buffer: sg.Buffer = .{},
    instance_buffer_capacity: usize = 0,

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
    }
};

pub const BoxOptions = struct {
    size: f32 = 1.0,
    width: ?f32 = null,
    height: ?f32 = null,
    depth: ?f32 = null,
    face_colors: ?[6]Color4 = null,
};

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
            c: [4]f32,
        };

        const quads = [_]Quad{
            // Front (+Z)
            .{
                .p = .{ .{ -w, -h, d }, .{ w, -h, d }, .{ w, h, d }, .{ -w, h, d } },
                .n = .{ 0, 0, 1 },
                .c = colors[0].toArray(),
            },
            // Back (-Z)
            .{
                .p = .{ .{ w, -h, -d }, .{ -w, -h, -d }, .{ -w, h, -d }, .{ w, h, -d } },
                .n = .{ 0, 0, -1 },
                .c = colors[1].toArray(),
            },
            // Left (-X)
            .{
                .p = .{ .{ -w, -h, -d }, .{ -w, -h, d }, .{ -w, h, d }, .{ -w, h, -d } },
                .n = .{ -1, 0, 0 },
                .c = colors[2].toArray(),
            },
            // Right (+X)
            .{
                .p = .{ .{ w, -h, d }, .{ w, -h, -d }, .{ w, h, -d }, .{ w, h, d } },
                .n = .{ 1, 0, 0 },
                .c = colors[3].toArray(),
            },
            // Top (+Y)
            .{
                .p = .{ .{ -w, h, d }, .{ w, h, d }, .{ w, h, -d }, .{ -w, h, -d } },
                .n = .{ 0, 1, 0 },
                .c = colors[4].toArray(),
            },
            // Bottom (-Y)
            .{
                .p = .{ .{ -w, -h, -d }, .{ w, -h, -d }, .{ w, -h, d }, .{ -w, -h, d } },
                .n = .{ 0, -1, 0 },
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
};
