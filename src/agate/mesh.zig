const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Vec2 = math.Vec2;
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

pub const BoneAttachment = struct {
    host_mesh: *Mesh,
    bone_index: usize,
    offset_matrix: Mat4 = Mat4.identity,
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
    attach_bone: ?BoneAttachment = null,
    base_matrix: Mat4 = Mat4.identity,

    // Culling, Shadows & Visibility
    is_visible: bool = true,
    cast_shadows: bool = true,
    receive_shadows: bool = true,
    culling_strategy: CullingStrategy = .frustum,
    local_bounding_box: BoundingBox = BoundingBox.zero,

    // Optional CPU-side geometry retained for physics collider creation
    // (convex hull / triangle mesh shapes). Owned by the scene allocator.
    cpu_positions: []Vec3 = &.{},
    cpu_indices: []u32 = &.{},

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

    /// Attaches this mesh to a specific bone socket of host_mesh.
    pub fn attachToBone(self: *Mesh, host_mesh: *Mesh, bone_index: usize) void {
        self.attach_bone = .{
            .host_mesh = host_mesh,
            .bone_index = bone_index,
            .offset_matrix = Mat4.identity,
        };
    }

    /// Attaches this mesh to a bone identified by name on host_mesh.
    pub fn attachToBoneByName(self: *Mesh, host_mesh: *Mesh, bone_name: []const u8) !void {
        const skel = host_mesh.skeleton orelse return error.NoSkeletonOnMesh;
        const idx = skel.findBoneIndex(bone_name) orelse return error.BoneNotFound;
        self.attachToBone(host_mesh, idx);
    }

    /// Detaches this mesh from its bone socket.
    pub fn detachFromBone(self: *Mesh) void {
        self.attach_bone = null;
    }

    pub fn getWorldMatrix(self: Mesh) Mat4 {
        const trs = Mat4.fromRotationTranslationScale(self.position, self.rotation, self.scaling);
        const local = Mat4.mul(trs, self.base_matrix);
        if (self.attach_bone) |att| {
            if (att.host_mesh.skeleton) |skel| {
                const host_mat = att.host_mesh.getWorldMatrix();
                const bone_mat = skel.getBoneWorldMatrix(att.bone_index, host_mat);
                const with_offset = Mat4.mul(bone_mat, att.offset_matrix);
                return Mat4.mul(with_offset, local);
            }
        }
        if (self.parent) |p| {
            return Mat4.mul(p.getWorldMatrix(), local);
        }
        return local;
    }

    pub fn getWorldBoundingBox(self: Mesh) BoundingBox {
        return self.local_bounding_box.transform(self.getWorldMatrix());
    }

    /// Retains a CPU copy of the geometry so physics colliders (convex hull,
    /// triangle mesh) can be built from it later. Owned by the allocator.
    pub fn retainCpuGeometry(self: *Mesh, allocator: std.mem.Allocator, vertices: []const Vertex, indices: []const u16) !void {
        const positions = try allocator.alloc(Vec3, vertices.len);
        errdefer allocator.free(positions);
        // Vec3 is extern 3xf32, same layout as Vertex.position: one copy each.
        for (vertices, 0..) |v, i| {
            positions[i] = @bitCast(v.position);
        }

        const index_u32 = try allocator.alloc(u32, indices.len);
        // Widening zero-extends: 8-wide SIMD chunks plus a scalar tail.
        var i: usize = 0;
        const widen_tail = indices.len & ~@as(usize, 7);
        while (i < widen_tail) : (i += 8) {
            const narrow: @Vector(8, u16) = indices[i..][0..8].*;
            const wide: @Vector(8, u32) = @as(@Vector(8, u32), narrow);
            index_u32[i..][0..8].* = wide;
        }
        while (i < indices.len) : (i += 1) {
            index_u32[i] = indices[i];
        }

        self.cpu_positions = positions;
        self.cpu_indices = index_u32;
    }

    /// Same as retainCpuGeometry but for sources that are already 32-bit indexed.
    pub fn retainCpuGeometryU32(self: *Mesh, allocator: std.mem.Allocator, vertices: []const Vertex, indices: []const u32) !void {
        const positions = try allocator.alloc(Vec3, vertices.len);
        errdefer allocator.free(positions);
        for (vertices, 0..) |v, i| {
            positions[i] = @bitCast(v.position);
        }

        const index_copy = try allocator.alloc(u32, indices.len);
        @memcpy(index_copy, indices);

        self.cpu_positions = positions;
        self.cpu_indices = index_copy;
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
        if (self.cpu_positions.len > 0) {
            allocator.free(self.cpu_positions);
        }
        if (self.cpu_indices.len > 0) {
            allocator.free(self.cpu_indices);
        }
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

pub const TerrainOptions = struct {
    /// x/z cell size and y height multiplier (matches HeightFieldOptions).
    scale: Vec3 = Vec3.new(1.0, 1.0, 1.0),
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

pub const TorusOptions = struct {
    diameter: f32 = 1.0,
    thickness: f32 = 0.5,
    tessellation: u32 = 24, // segments around both the ring and the tube
    color: Color4 = Color4.white,
};

pub const TorusKnotOptions = struct {
    radius: f32 = 1.0, // overall scale of the knot center curve
    tube: f32 = 0.4,
    radial_segments: u32 = 16, // segments around the tube
    tubular_segments: u32 = 128, // segments along the knot
    p: u32 = 2,
    q: u32 = 3,
    color: Color4 = Color4.white,
};

pub const DiscOptions = struct {
    radius: f32 = 0.5,
    tessellation: u32 = 32, // segments around the rim
    color: Color4 = Color4.white,
};

pub const RibbonOptions = struct {
    paths: []const []const Vec3, // at least 2 paths with equal point counts
    close_path: bool = false, // connect last point back to first within each path
    close_array: bool = false, // connect last path back to first path
    color: Color4 = Color4.white,
};

pub const LatheOptions = struct {
    shape: []const Vec3, // profile revolved around Y; x is radius, y is height
    tessellation: u32 = 24, // segments around the Y axis
    color: Color4 = Color4.white,
};

pub const PlaneOptions = struct {
    width: f32 = 1.0,
    height: f32 = 1.0,
    subdivisions_x: u32 = 1, // quads along X (clamped to >= 1)
    subdivisions_y: u32 = 1, // quads along Y (clamped to >= 1)
    uv_scale: Vec2 = Vec2.one, // UV multiplier applied per axis
    color: Color4 = Color4.white,
};

pub const TubeOptions = struct {
    path: []const Vec3, // at least 2 points
    radius: f32 = 0.1, // uniform radius, used when radii is null
    radii: ?[]const f32 = null, // per-point radius override (must match path length)
    tessellation: u32 = 8, // segments around the tube (clamped to >= 3)
    capped: bool = false, // flat end caps (ignored when closed)
    closed: bool = false, // connect the last point back to the first
    color: Color4 = Color4.white,
};

pub const LinesOptions = struct {
    points: []const Vec3, // at least 2 points
    width: f32 = 0.1, // ribbon width in world units
    colors: ?[]const Color4 = null, // per-point color override (must match points length)
    closed: bool = false, // connect the last point back to the first
    up: Vec3 = Vec3.up, // reference hint for the ribbon plane orientation
    color: Color4 = Color4.white,
};

pub const ExtrudeOptions = struct {
    profile: []const Vec2, // closed 2D outline in XY (CCW preferred, CW is normalized)
    depth: f32 = 1.0, // extrusion distance along +Z, from z = 0 to z = depth
    capped: bool = true, // front/back caps triangulated with ear clipping
    uv_scale: Vec2 = Vec2.one, // UV multiplier (arclength/depth on sides, XY on caps)
    color: Color4 = Color4.white,
};

inline fn accumulateTriangleTangent(vertices: []Vertex, idx0: usize, idx1: usize, idx2: usize) void {
    if (idx0 >= vertices.len or idx1 >= vertices.len or idx2 >= vertices.len) return;

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

pub fn computeTangents(vertices: []Vertex, indices: ?[]const u32, indices16: ?[]const u16) void {
    for (vertices) |*v| {
        v.tangent = .{ 0, 0, 0, 1 };
    }

    // One loop per index source so the per-triangle source selection branches
    // disappear; accumulation order (and hence shared-vertex sums) is unchanged.
    if (indices) |idx| {
        var tri_i: usize = 0;
        while (tri_i + 2 < idx.len) : (tri_i += 3) {
            accumulateTriangleTangent(vertices, idx[tri_i], idx[tri_i + 1], idx[tri_i + 2]);
        }
    } else if (indices16) |idx16| {
        var tri_i: usize = 0;
        while (tri_i + 2 < idx16.len) : (tri_i += 3) {
            accumulateTriangleTangent(vertices, idx16[tri_i], idx16[tri_i + 1], idx16[tri_i + 2]);
        }
    } else {
        var tri_i: usize = 0;
        while (tri_i + 2 < vertices.len) : (tri_i += 3) {
            accumulateTriangleTangent(vertices, tri_i, tri_i + 1, tri_i + 2);
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

// CPU-side geometry produced by the pure (non-GPU) builder helpers below.
// MeshBuilder uploads it to sokol buffers; unit tests inspect it directly.
const GeometryData = struct {
    vertices: []Vertex,
    indices: []u32,
    bounds: BoundingBox,

    fn deinit(self: *GeometryData, allocator: std.mem.Allocator) void {
        allocator.free(self.vertices);
        allocator.free(self.indices);
    }
};

// One table entry per unique revolution angle: sin/cos plus the normalized
// coordinate reused for UVs. Tables keep results bit-identical to per-vertex
// trig while computing each angle once per builder call.
const TrigEntry = struct {
    cos: f32,
    sin: f32,
    f: f32,
};

fn buildTorusData(allocator: std.mem.Allocator, options: TorusOptions) !GeometryData {
    const tess = @max(3, options.tessellation);
    const ring_radius = @max(0.0, options.diameter) * 0.5;
    const tube_radius = @max(0.0, options.thickness) * 0.5;
    const color = options.color.toArray();

    const side = tess + 1;
    const vertices = try allocator.alloc(Vertex, side * side);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, tess * tess * 6);
    errdefer allocator.free(indices);

    const pi = std.math.pi;
    const tess_f: f32 = @floatFromInt(tess);
    const ring_tab = try allocator.alloc(TrigEntry, side);
    defer allocator.free(ring_tab);
    for (0..side) |j| {
        const u = @as(f32, @floatFromInt(j)) / tess_f;
        const a = u * 2.0 * pi;
        ring_tab[j] = .{ .cos = @cos(a), .sin = @sin(a), .f = u };
    }
    const tube_tab = try allocator.alloc(TrigEntry, side);
    defer allocator.free(tube_tab);
    for (0..side) |i| {
        const v = @as(f32, @floatFromInt(i)) / tess_f;
        const a = v * 2.0 * pi;
        tube_tab[i] = .{ .cos = @cos(a), .sin = @sin(a), .f = v };
    }

    var vi: usize = 0;
    for (0..side) |j| {
        const cos_u = ring_tab[j].cos;
        const sin_u = ring_tab[j].sin;
        const uu = ring_tab[j].f;
        for (0..side) |i| {
            const cos_v = tube_tab[i].cos;
            const sin_v = tube_tab[i].sin;
            const cx = ring_radius + tube_radius * cos_v;
            vertices[vi] = .{
                .position = .{ cx * cos_u, tube_radius * sin_v, cx * sin_u },
                .normal = .{ cos_v * cos_u, sin_v, cos_v * sin_u },
                .color = color,
                .uv = .{ uu, tube_tab[i].f },
                .tangent = .{ -sin_u, 0.0, cos_u, 1.0 },
            };
            vi += 1;
        }
    }

    var ii: usize = 0;
    for (0..tess) |j| {
        for (0..tess) |i| {
            const a: u32 = @intCast(j * side + i);
            const b: u32 = @intCast(j * side + (i + 1));
            const c: u32 = @intCast((j + 1) * side + (i + 1));
            const d: u32 = @intCast((j + 1) * side + i);
            indices[ii + 0] = a;
            indices[ii + 1] = b;
            indices[ii + 2] = c;
            indices[ii + 3] = a;
            indices[ii + 4] = c;
            indices[ii + 5] = d;
            ii += 6;
        }
    }

    const extent = ring_radius + tube_radius;
    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-extent, -tube_radius, -extent),
            Vec3.new(extent, tube_radius, extent),
        ),
    };
}

// Center curve of a (p, q) torus knot, Y-up: the knot winds around the Y axis.
fn torusKnotCenter(p: u32, q: u32, scale: f32, t: f32) Vec3 {
    const pf = @as(f32, @floatFromInt(p));
    const qf = @as(f32, @floatFromInt(q));
    const k = 2.0 + @cos(qf * t);
    return Vec3.new(
        k * @cos(pf * t) * scale,
        @sin(qf * t) * scale,
        k * @sin(pf * t) * scale,
    );
}

fn buildTorusKnotData(allocator: std.mem.Allocator, options: TorusKnotOptions) !GeometryData {
    const radial = @max(3, options.radial_segments);
    const tubular = @max(3, options.tubular_segments);
    const p = @max(1, options.p);
    const q = @max(1, options.q);
    const tube_radius = @max(0.0, options.tube);
    // The raw curve reaches a radial extent of 3, so scale it to `radius`.
    const scale = @max(0.0, options.radius) / 3.0;
    const color = options.color.toArray();

    const ring = radial + 1;
    const rows = tubular + 1;
    const vertices = try allocator.alloc(Vertex, rows * ring);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, tubular * radial * 6);
    errdefer allocator.free(indices);

    const pi = std.math.pi;
    // Single frame buffer replaces separate center/tangent/normal/binormal
    // arrays; phases below fill one field at a time in the same order as before.
    const Frame = struct {
        center: Vec3,
        tangent: Vec3,
        n: Vec3,
        b: Vec3,
    };
    const frames = try allocator.alloc(Frame, rows);
    defer allocator.free(frames);
    const tubular_f: f32 = @floatFromInt(tubular);
    for (0..rows) |i| {
        const t = @as(f32, @floatFromInt(i)) / tubular_f * 2.0 * pi;
        frames[i].center = torusKnotCenter(p, q, scale, t);
    }

    // Wrapped central-difference tangents.
    for (0..rows) |i| {
        const prev = frames[(i + rows - 1) % rows].center;
        const next = frames[(i + 1) % rows].center;
        const d = next.sub(prev);
        frames[i].tangent = if (d.lengthSq() > 1e-12) d.normalize() else Vec3.forward;
    }

    // Parallel-transport frames keep the tube twist-free along the knot.
    var ref = Vec3.up;
    if (@abs(frames[0].tangent.dot(ref)) > 0.9) ref = Vec3.right;
    var n0 = ref.sub(frames[0].tangent.scale(frames[0].tangent.dot(ref)));
    if (n0.lengthSq() < 1e-12) n0 = Vec3.forward;
    n0 = n0.normalize();
    frames[0].n = n0;
    frames[0].b = frames[0].tangent.cross(n0);
    for (1..rows) |i| {
        const t = frames[i].tangent;
        var n = frames[i - 1].n.sub(t.scale(t.dot(frames[i - 1].n)));
        if (n.lengthSq() < 1e-12) {
            n = frames[i - 1].n;
        } else {
            n = n.normalize();
        }
        frames[i].n = n;
        frames[i].b = t.cross(n);
    }

    // Each unique tube angle computed once, shared by all rings.
    const radial_f: f32 = @floatFromInt(radial);
    const tube_tab = try allocator.alloc(TrigEntry, ring);
    defer allocator.free(tube_tab);
    for (0..ring) |j| {
        const v = @as(f32, @floatFromInt(j)) / radial_f;
        const a = v * 2.0 * pi;
        tube_tab[j] = .{ .cos = @cos(a), .sin = @sin(a), .f = v };
    }

    var min_v = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    var max_v = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
    var vi: usize = 0;
    for (0..rows) |i| {
        // Cache the ring frame in locals: one indexed load per field per ring.
        const fr = frames[i];
        const uu = @as(f32, @floatFromInt(i)) / tubular_f;
        for (0..ring) |j| {
            const cos_v = tube_tab[j].cos;
            const sin_v = tube_tab[j].sin;
            const offset = fr.n.scale(cos_v).add(fr.b.scale(sin_v));
            const pos = fr.center.add(offset.scale(tube_radius));
            vertices[vi] = .{
                .position = pos.toArray(),
                .normal = offset.toArray(),
                .color = color,
                .uv = .{ uu, tube_tab[j].f },
                .tangent = .{ fr.tangent.x, fr.tangent.y, fr.tangent.z, 1.0 },
            };
            min_v = Vec3.new(@min(min_v.x, pos.x), @min(min_v.y, pos.y), @min(min_v.z, pos.z));
            max_v = Vec3.new(@max(max_v.x, pos.x), @max(max_v.y, pos.y), @max(max_v.z, pos.z));
            vi += 1;
        }
    }

    var ii: usize = 0;
    for (0..tubular) |i| {
        for (0..radial) |j| {
            const a: u32 = @intCast(i * ring + j);
            const b: u32 = @intCast(i * ring + (j + 1));
            const c: u32 = @intCast((i + 1) * ring + (j + 1));
            const d: u32 = @intCast((i + 1) * ring + j);
            indices[ii + 0] = a;
            indices[ii + 1] = b;
            indices[ii + 2] = c;
            indices[ii + 3] = a;
            indices[ii + 4] = c;
            indices[ii + 5] = d;
            ii += 6;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(min_v, max_v),
    };
}

fn buildDiscData(allocator: std.mem.Allocator, options: DiscOptions) !GeometryData {
    const tess = @max(3, options.tessellation);
    const radius = @max(0.0, options.radius);
    const color = options.color.toArray();

    const vertices = try allocator.alloc(Vertex, tess + 2);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, tess * 3);
    errdefer allocator.free(indices);

    vertices[0] = .{
        .position = .{ 0.0, 0.0, 0.0 },
        .normal = .{ 0.0, 1.0, 0.0 },
        .color = color,
        .uv = .{ 0.5, 0.5 },
        .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
    };

    const pi = std.math.pi;
    for (0..tess + 1) |j| {
        const theta = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(tess)) * 2.0 * pi;
        const x = @sin(theta);
        const z = @cos(theta);
        vertices[1 + j] = .{
            .position = .{ x * radius, 0.0, z * radius },
            .normal = .{ 0.0, 1.0, 0.0 },
            .color = color,
            .uv = .{ 0.5 + x * 0.5, 0.5 + z * 0.5 },
            .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
        };
    }

    for (0..tess) |j| {
        indices[j * 3 + 0] = 0;
        indices[j * 3 + 1] = @intCast(1 + j);
        indices[j * 3 + 2] = @intCast(1 + j + 1);
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-radius, 0.0, -radius),
            Vec3.new(radius, 0.0, radius),
        ),
    };
}

fn buildRibbonData(allocator: std.mem.Allocator, options: RibbonOptions) !GeometryData {
    const paths = options.paths;
    if (paths.len < 2) return error.InvalidRibbon;
    const count = paths[0].len;
    if (count < 2) return error.InvalidRibbon;
    for (paths[1..]) |path| {
        if (path.len != count) return error.InvalidRibbon;
    }

    const num_paths = paths.len;
    const color = options.color.toArray();
    const close_path = options.close_path;
    const close_array = options.close_array;
    const row_quads = if (close_array) num_paths else num_paths - 1;
    const col_quads = if (close_path) count else count - 1;
    // UV denominators are loop-invariant: hoist the branches out of the vertex loop.
    const u_denom: f32 = if (close_array) @floatFromInt(num_paths) else @floatFromInt(num_paths - 1);
    const v_denom: f32 = if (close_path) @floatFromInt(count) else @floatFromInt(count - 1);

    const vertices = try allocator.alloc(Vertex, num_paths * count);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, row_quads * col_quads * 6);
    errdefer allocator.free(indices);

    var min_v = paths[0][0];
    var max_v = paths[0][0];
    for (0..num_paths) |pi_| {
        const uu = @as(f32, @floatFromInt(pi_)) / u_denom;
        for (0..count) |ci| {
            const pos = paths[pi_][ci];
            min_v = Vec3.new(@min(min_v.x, pos.x), @min(min_v.y, pos.y), @min(min_v.z, pos.z));
            max_v = Vec3.new(@max(max_v.x, pos.x), @max(max_v.y, pos.y), @max(max_v.z, pos.z));
            vertices[pi_ * count + ci] = .{
                .position = pos.toArray(),
                .normal = .{ 0.0, 0.0, 0.0 },
                .color = color,
                .uv = .{ uu, @as(f32, @floatFromInt(ci)) / v_denom },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
        }
    }

    var ii: usize = 0;
    for (0..row_quads) |pi_| {
        // Row wrap resolved once per row instead of per quad.
        const qi = if (close_array) (pi_ + 1) % num_paths else pi_ + 1;
        for (0..col_quads) |ci| {
            const cj = if (close_path) (ci + 1) % count else ci + 1;
            const a: u32 = @intCast(pi_ * count + ci);
            const b: u32 = @intCast(pi_ * count + cj);
            const c: u32 = @intCast(qi * count + cj);
            const d: u32 = @intCast(qi * count + ci);
            indices[ii + 0] = a;
            indices[ii + 1] = b;
            indices[ii + 2] = c;
            indices[ii + 3] = a;
            indices[ii + 4] = c;
            indices[ii + 5] = d;
            ii += 6;
        }
    }

    // Smooth normals: accumulate area-weighted face normals, then normalize.
    var tri: usize = 0;
    while (tri < indices.len) : (tri += 3) {
        const p0 = vertices[indices[tri]].position;
        const p1 = vertices[indices[tri + 1]].position;
        const p2 = vertices[indices[tri + 2]].position;
        const e1 = Vec3.new(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
        const e2 = Vec3.new(p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]);
        const fn_ = e1.cross(e2);
        for ([3]u32{ indices[tri], indices[tri + 1], indices[tri + 2] }) |idx| {
            vertices[idx].normal[0] += fn_.x;
            vertices[idx].normal[1] += fn_.y;
            vertices[idx].normal[2] += fn_.z;
        }
    }

    for (0..num_paths) |pi_| {
        const path = paths[pi_];
        for (0..count) |ci| {
            const v = &vertices[pi_ * count + ci];
            var n = Vec3.new(v.normal[0], v.normal[1], v.normal[2]);
            if (n.lengthSq() < 1e-12) {
                n = Vec3.up;
            } else {
                n = n.normalize();
            }
            v.normal = n.toArray();
            // Tangent follows the path direction, orthogonalized against the normal.
            const c0 = if (close_path) (ci + count - 1) % count else if (ci > 0) ci - 1 else 0;
            const c1 = if (close_path) (ci + 1) % count else @min(ci + 1, count - 1);
            var t = path[c1].sub(path[c0]);
            t = t.sub(n.scale(n.dot(t)));
            if (t.lengthSq() < 1e-12) {
                t = if (@abs(n.x) > 0.9) Vec3.new(0, 1, 0) else Vec3.new(1, 0, 0);
            } else {
                t = t.normalize();
            }
            v.tangent = .{ t.x, t.y, t.z, 1.0 };
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(min_v, max_v),
    };
}

fn buildLatheData(allocator: std.mem.Allocator, options: LatheOptions) !GeometryData {
    const shape = options.shape;
    if (shape.len < 2) return error.InvalidLathe;
    const tess = @max(3, options.tessellation);
    const color = options.color.toArray();
    const eps = 1e-6;

    var max_r: f32 = 0.0;
    var min_y = shape[0].y;
    var max_y = shape[0].y;
    for (shape) |pt| {
        max_r = @max(max_r, @max(0.0, pt.x));
        min_y = @min(min_y, pt.y);
        max_y = @max(max_y, pt.y);
    }
    const mid_y = (min_y + max_y) * 0.5;

    // Count revolved quads, skipping degenerate on-axis segments.
    var quads: usize = 0;
    for (0..shape.len - 1) |i| {
        if (@abs(shape[i].x) < eps and @abs(shape[i + 1].x) < eps) continue;
        quads += 1;
    }

    const side = tess + 1;
    const vertices = try allocator.alloc(Vertex, shape.len * side);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, quads * tess * 6);
    errdefer allocator.free(indices);

    const pi = std.math.pi;
    // Each unique revolution angle computed once, shared by all profile rows.
    const trig = try allocator.alloc(TrigEntry, side);
    defer allocator.free(trig);
    const tess_f: f32 = @floatFromInt(tess);
    for (0..side) |j| {
        const f = @as(f32, @floatFromInt(j)) / tess_f;
        const theta = f * 2.0 * pi;
        trig[j] = .{ .cos = @cos(theta), .sin = @sin(theta), .f = f };
    }
    const v_denom: f32 = @floatFromInt(shape.len - 1);

    for (shape, 0..) |pt, i| {
        const prev = shape[if (i > 0) i - 1 else 0];
        const next = shape[if (i + 1 < shape.len) i + 1 else shape.len - 1];
        const dr = next.x - prev.x;
        const dy = next.y - prev.y;
        const t_len = @sqrt(dr * dr + dy * dy);
        const degenerate = t_len < eps;
        // Outward normal of the revolved profile: (dy * sin, -dr, dy * cos).
        const ndr: f32 = if (degenerate) 0.0 else dr / t_len;
        const ndy: f32 = if (degenerate) 0.0 else dy / t_len;
        const cap_n: Vec3 = if (pt.y >= mid_y) Vec3.up else Vec3.down;
        const radius = @max(0.0, pt.x);
        const vv = @as(f32, @floatFromInt(i)) / v_denom;
        for (0..side) |j| {
            const sin_t = trig[j].sin;
            const cos_t = trig[j].cos;
            // (ndr, ndy) is unit, so ndr*ndr + ndy*(sin^2+cos^2) == 1: the
            // normal below is unit by construction, no normalize() needed.
            const n = if (degenerate) cap_n else Vec3.new(ndy * sin_t, -ndr, ndy * cos_t);
            vertices[i * side + j] = .{
                .position = .{ radius * sin_t, pt.y, radius * cos_t },
                .normal = n.toArray(),
                .color = color,
                .uv = .{ trig[j].f, vv },
                .tangent = .{ cos_t, 0.0, -sin_t, 1.0 },
            };
        }
    }

    var ii: usize = 0;
    for (0..shape.len - 1) |i| {
        if (@abs(shape[i].x) < eps and @abs(shape[i + 1].x) < eps) continue;
        for (0..tess) |j| {
            const a: u32 = @intCast(i * side + j);
            const b: u32 = @intCast(i * side + (j + 1));
            const c: u32 = @intCast((i + 1) * side + (j + 1));
            const d: u32 = @intCast((i + 1) * side + j);
            indices[ii + 0] = a;
            indices[ii + 1] = b;
            indices[ii + 2] = c;
            indices[ii + 3] = a;
            indices[ii + 4] = c;
            indices[ii + 5] = d;
            ii += 6;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-max_r, min_y, -max_r),
            Vec3.new(max_r, max_y, max_r),
        ),
    };
}

fn buildPlaneData(allocator: std.mem.Allocator, options: PlaneOptions) !GeometryData {
    const sx = @max(1, options.subdivisions_x);
    const sy = @max(1, options.subdivisions_y);
    const half_w = options.width * 0.5;
    const half_h = options.height * 0.5;
    const color = options.color.toArray();

    const cols = sx + 1;
    const rows = sy + 1;
    const vertices = try allocator.alloc(Vertex, cols * rows);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, sx * sy * 6);
    errdefer allocator.free(indices);

    const sx_f: f32 = @floatFromInt(sx);
    const sy_f: f32 = @floatFromInt(sy);
    var vi: usize = 0;
    for (0..rows) |iy| {
        const fy = @as(f32, @floatFromInt(iy)) / sy_f;
        const y = -half_h + fy * options.height;
        for (0..cols) |ix| {
            const fx = @as(f32, @floatFromInt(ix)) / sx_f;
            const x = -half_w + fx * options.width;
            vertices[vi] = .{
                .position = .{ x, y, 0.0 },
                .normal = .{ 0.0, 0.0, 1.0 },
                .color = color,
                .uv = .{ fx * options.uv_scale.x, fy * options.uv_scale.y },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }
    }

    var ii: usize = 0;
    for (0..sy) |iy| {
        for (0..sx) |ix| {
            const a: u32 = @intCast(iy * cols + ix);
            const b: u32 = @intCast(iy * cols + (ix + 1));
            const c: u32 = @intCast((iy + 1) * cols + (ix + 1));
            const d: u32 = @intCast((iy + 1) * cols + ix);
            // CCW winding for the +Z normal.
            indices[ii + 0] = a;
            indices[ii + 1] = b;
            indices[ii + 2] = c;
            indices[ii + 3] = a;
            indices[ii + 4] = c;
            indices[ii + 5] = d;
            ii += 6;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-half_w, -half_h, 0.0),
            Vec3.new(half_w, half_h, 0.0),
        ),
    };
}

// Tangents of a polyline via central differences (wrapped when closed).
// Zero-length neighborhoods fall back to the previous tangent so a repeated
// point never produces a NaN frame.
fn computePathTangents(points: []const Vec3, closed: bool, out: []Vec3) void {
    const n = points.len;
    for (0..n) |i| {
        const prev_idx = if (closed) (i + n - 1) % n else if (i > 0) i - 1 else i;
        const next_idx = if (closed) (i + 1) % n else if (i + 1 < n) i + 1 else i;
        const d = points[next_idx].sub(points[prev_idx]);
        if (d.lengthSq() > 1e-12) {
            out[i] = d.normalize();
        } else if (i > 0) {
            out[i] = out[i - 1];
        } else {
            out[i] = Vec3.forward;
        }
    }
}

// Parallel-transport normal/binormal frames along unit tangents: each normal
// is projected onto the plane perpendicular to the next tangent, which keeps
// the frame twist-free along the path. `up_hint` seeds the first normal;
// when closed the residual loop twist is measured and spread evenly so the
// last ring meets the first seamlessly.
fn parallelTransportFrames(tangents: []const Vec3, closed: bool, up_hint: Vec3, normals: []Vec3, binormals: []Vec3) void {
    const n = tangents.len;
    var ref = up_hint;
    if (ref.lengthSq() < 1e-12) ref = Vec3.up;
    if (@abs(tangents[0].dot(ref)) > 0.9) {
        ref = if (@abs(tangents[0].x) < 0.9) Vec3.right else Vec3.forward;
    }
    var n0 = ref.sub(tangents[0].scale(tangents[0].dot(ref)));
    if (n0.lengthSq() < 1e-12) n0 = Vec3.forward;
    normals[0] = n0.normalize();
    binormals[0] = tangents[0].cross(normals[0]);
    for (1..n) |i| {
        const t = tangents[i];
        var nn = normals[i - 1].sub(t.scale(t.dot(normals[i - 1])));
        if (nn.lengthSq() < 1e-12) {
            nn = normals[i - 1];
        } else {
            nn = nn.normalize();
        }
        normals[i] = nn;
        binormals[i] = t.cross(nn);
    }
    if (closed and n > 1) {
        // Transport the last normal onto the first tangent and measure the
        // residual twist against the first normal.
        const t0 = tangents[0];
        var wrap = normals[n - 1].sub(t0.scale(t0.dot(normals[n - 1])));
        if (wrap.lengthSq() > 1e-12) {
            wrap = wrap.normalize();
            const cos_a = std.math.clamp(wrap.dot(normals[0]), -1.0, 1.0);
            const sin_a = t0.dot(wrap.cross(normals[0]));
            const twist = std.math.atan2(sin_a, cos_a);
            if (@abs(twist) > 1e-6) {
                // Unwind the twist linearly: ring i rotates by twist * i / n
                // around its own tangent (Rodrigues rotation, normal stays
                // perpendicular to the tangent so no re-normalize is needed).
                const n_f: f32 = @floatFromInt(n);
                for (1..n) |i| {
                    const a = twist * (@as(f32, @floatFromInt(i)) / n_f);
                    const c = @cos(a);
                    const s = @sin(a);
                    const nn2 = normals[i];
                    const tt = tangents[i];
                    const rotated = nn2.scale(c).add(tt.cross(nn2).scale(s));
                    normals[i] = rotated;
                    binormals[i] = tt.cross(rotated);
                }
            }
        }
    }
}

// Cumulative arc lengths of a polyline (closing segment included when closed).
// `total` is at least 1e-9 so UV computation never divides by zero.
fn computePathLengths(points: []const Vec3, closed: bool, cumulative: []f32) f32 {
    const n = points.len;
    cumulative[0] = 0.0;
    for (1..n) |i| {
        cumulative[i] = cumulative[i - 1] + points[i].distance(points[i - 1]);
    }
    var total = cumulative[n - 1];
    if (closed) total += points[0].distance(points[n - 1]);
    if (total < 1e-9) total = 1.0;
    return total;
}

fn buildTubeData(allocator: std.mem.Allocator, options: TubeOptions) !GeometryData {
    const points = options.path;
    if (points.len < 2) return error.InvalidTube;
    if (options.radii) |radii| {
        if (radii.len != points.len) return error.InvalidTube;
    }
    const tess = @max(3, options.tessellation);
    const closed = options.closed;
    // Caps on a closed loop would duplicate the seam ring: ignore them.
    const capped = options.capped and !closed;
    const color = options.color.toArray();

    const n = points.len;
    const ring = tess + 1;

    // Per-point radius table: avoids branching on options.radii in the loop.
    const point_radii = try allocator.alloc(f32, n);
    defer allocator.free(point_radii);
    if (options.radii) |radii| {
        for (radii, 0..) |r, i| point_radii[i] = @max(0.0, r);
    } else {
        @memset(point_radii, @max(0.0, options.radius));
    }

    const tangents = try allocator.alloc(Vec3, n);
    defer allocator.free(tangents);
    computePathTangents(points, closed, tangents);
    const normals = try allocator.alloc(Vec3, n);
    defer allocator.free(normals);
    const binormals = try allocator.alloc(Vec3, n);
    defer allocator.free(binormals);
    parallelTransportFrames(tangents, closed, Vec3.up, normals, binormals);

    const cumulative = try allocator.alloc(f32, n);
    defer allocator.free(cumulative);
    const total_len = computePathLengths(points, closed, cumulative);

    // Each unique ring angle computed once, shared by all rows.
    const tube_tab = try allocator.alloc(TrigEntry, ring);
    defer allocator.free(tube_tab);
    const tess_f: f32 = @floatFromInt(tess);
    for (0..ring) |j| {
        const v = @as(f32, @floatFromInt(j)) / tess_f;
        const a = v * 2.0 * std.math.pi;
        tube_tab[j] = .{ .cos = @cos(a), .sin = @sin(a), .f = v };
    }

    const cap_verts = if (capped) 2 * ring + 2 else 0;
    const vertices = try allocator.alloc(Vertex, n * ring + cap_verts);
    errdefer allocator.free(vertices);
    const side_quads = if (closed) n else n - 1;
    const cap_indices = if (capped) 2 * tess * 3 else 0;
    const indices = try allocator.alloc(u32, side_quads * tess * 6 + cap_indices);
    errdefer allocator.free(indices);

    var min_v = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    var max_v = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
    var vi: usize = 0;
    for (0..n) |i| {
        // Arc-length U keeps texture density uniform along the path.
        const uu = cumulative[i] / total_len;
        for (0..ring) |j| {
            const dir = normals[i].scale(tube_tab[j].cos).add(binormals[i].scale(tube_tab[j].sin));
            const pos = points[i].add(dir.scale(point_radii[i]));
            vertices[vi] = .{
                .position = pos.toArray(),
                .normal = dir.toArray(),
                .color = color,
                .uv = .{ uu, tube_tab[j].f },
                .tangent = .{ tangents[i].x, tangents[i].y, tangents[i].z, 1.0 },
            };
            min_v = Vec3.new(@min(min_v.x, pos.x), @min(min_v.y, pos.y), @min(min_v.z, pos.z));
            max_v = Vec3.new(@max(max_v.x, pos.x), @max(max_v.y, pos.y), @max(max_v.z, pos.z));
            vi += 1;
        }
    }

    var ii: usize = 0;
    for (0..side_quads) |i| {
        const ni = if (closed) (i + 1) % n else i + 1;
        for (0..tess) |j| {
            const a: u32 = @intCast(i * ring + j);
            const b: u32 = @intCast(i * ring + (j + 1));
            const c: u32 = @intCast(ni * ring + (j + 1));
            const d: u32 = @intCast(ni * ring + j);
            indices[ii + 0] = a;
            indices[ii + 1] = b;
            indices[ii + 2] = c;
            indices[ii + 3] = a;
            indices[ii + 4] = c;
            indices[ii + 5] = d;
            ii += 6;
        }
    }

    if (capped) {
        // Flat end caps reuse the side ring positions with axial normals:
        // a duplicated ring plus a center vertex per cap.
        const cap_uv_c: [2]f32 = .{ 0.5, 0.5 };
        for ([2]usize{ 0, 1 }) |cap| {
            const row = if (cap == 0) 0 else n - 1;
            const axial = if (cap == 0) tangents[0].scale(-1.0) else tangents[n - 1];
            const center_idx: u32 = @intCast(vi);
            const cap_center = points[row];
            vertices[vi] = .{
                .position = cap_center.toArray(),
                .normal = axial.toArray(),
                .color = color,
                .uv = cap_uv_c,
                .tangent = .{ normals[row].x, normals[row].y, normals[row].z, 1.0 },
            };
            vi += 1;
            const ring_start: u32 = @intCast(vi);
            for (0..ring) |j| {
                const dir = normals[row].scale(tube_tab[j].cos).add(binormals[row].scale(tube_tab[j].sin));
                const pos = cap_center.add(dir.scale(point_radii[row]));
                vertices[vi] = .{
                    .position = pos.toArray(),
                    .normal = axial.toArray(),
                    .color = color,
                    .uv = .{ 0.5 + tube_tab[j].cos * 0.5, 0.5 + tube_tab[j].sin * 0.5 },
                    .tangent = .{ normals[row].x, normals[row].y, normals[row].z, 1.0 },
                };
                vi += 1;
            }
            for (0..tess) |j| {
                const j0: u32 = @intCast(j);
                const j1: u32 = @intCast(j + 1);
                if (cap == 0) {
                    // Start cap faces -tangent.
                    indices[ii + 0] = center_idx;
                    indices[ii + 1] = ring_start + j1;
                    indices[ii + 2] = ring_start + j0;
                } else {
                    // End cap faces +tangent.
                    indices[ii + 0] = center_idx;
                    indices[ii + 1] = ring_start + j0;
                    indices[ii + 2] = ring_start + j1;
                }
                ii += 3;
            }
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(min_v, max_v),
    };
}

fn buildLinesData(allocator: std.mem.Allocator, options: LinesOptions) !GeometryData {
    const points = options.points;
    if (points.len < 2) return error.InvalidLines;
    if (options.colors) |cols| {
        if (cols.len != points.len) return error.InvalidLines;
    }
    const half_w = @max(0.0, options.width) * 0.5;
    const closed = options.closed;
    const uniform = options.color.toArray();

    const n = points.len;
    const tangents = try allocator.alloc(Vec3, n);
    defer allocator.free(tangents);
    computePathTangents(points, closed, tangents);
    // Side vectors (normals output) lie in the plane perpendicular to the
    // path and closest to the up hint; the ribbon face normal is the
    // binormal output (tangent x side).
    const sides = try allocator.alloc(Vec3, n);
    defer allocator.free(sides);
    const face_normals = try allocator.alloc(Vec3, n);
    defer allocator.free(face_normals);
    parallelTransportFrames(tangents, closed, options.up, sides, face_normals);

    const cumulative = try allocator.alloc(f32, n);
    defer allocator.free(cumulative);
    const total_len = computePathLengths(points, closed, cumulative);

    const vertices = try allocator.alloc(Vertex, 2 * n);
    errdefer allocator.free(vertices);
    const segs = if (closed) n else n - 1;
    const indices = try allocator.alloc(u32, segs * 6);
    errdefer allocator.free(indices);

    var min_v = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    var max_v = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
    for (0..n) |i| {
        const uu = cumulative[i] / total_len;
        const col = if (options.colors) |cols| cols[i].toArray() else uniform;
        const left = points[i].add(sides[i].scale(half_w));
        const right = points[i].sub(sides[i].scale(half_w));
        const normal = face_normals[i].toArray();
        const tangent = [4]f32{ tangents[i].x, tangents[i].y, tangents[i].z, 1.0 };
        vertices[2 * i] = .{
            .position = left.toArray(),
            .normal = normal,
            .color = col,
            .uv = .{ uu, 1.0 },
            .tangent = tangent,
        };
        vertices[2 * i + 1] = .{
            .position = right.toArray(),
            .normal = normal,
            .color = col,
            .uv = .{ uu, 0.0 },
            .tangent = tangent,
        };
        for ([2]Vec3{ left, right }) |p| {
            min_v = Vec3.new(@min(min_v.x, p.x), @min(min_v.y, p.y), @min(min_v.z, p.z));
            max_v = Vec3.new(@max(max_v.x, p.x), @max(max_v.y, p.y), @max(max_v.z, p.z));
        }
    }

    var ii: usize = 0;
    for (0..segs) |i| {
        const ni = if (closed) (i + 1) % n else i + 1;
        const a: u32 = @intCast(2 * i);
        const b: u32 = @intCast(2 * i + 1);
        const c: u32 = @intCast(2 * ni);
        const d: u32 = @intCast(2 * ni + 1);
        // CCW winding for the (tangent x side) face normal.
        indices[ii + 0] = a;
        indices[ii + 1] = b;
        indices[ii + 2] = c;
        indices[ii + 3] = b;
        indices[ii + 4] = d;
        indices[ii + 5] = c;
        ii += 6;
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(min_v, max_v),
    };
}

inline fn orient2d(a: Vec2, b: Vec2, c: Vec2) f32 {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// Containment in a CCW triangle, with points on the edges counting as
// inside: a vertex touching the ear triangle boundary blocks the ear, which
// keeps diagonals of concave outlines from leaking outside the polygon.
fn pointInTriangle2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) bool {
    const eps: f32 = 1e-9;
    return orient2d(a, b, p) >= -eps and orient2d(b, c, p) >= -eps and orient2d(c, a, p) >= -eps;
}

fn isEarTip(poly: []const Vec2, ip: usize, ic: usize, inext: usize, live: []const usize) bool {
    const a = poly[ip];
    const b = poly[ic];
    const c = poly[inext];
    // Reflex (or collinear) vertices can never be ear tips in a CCW polygon.
    if (orient2d(a, b, c) <= 1e-9) return false;
    for (live) |vi| {
        if (vi == ip or vi == ic or vi == inext) continue;
        if (pointInTriangle2d(poly[vi], a, b, c)) return false;
    }
    return true;
}

// Proper crossing of two segments (shared endpoints do not count;
// the caller only tests non-adjacent outline edges).
fn segmentsCross2d(a: Vec2, b: Vec2, c: Vec2, d: Vec2) bool {
    const o1 = orient2d(a, b, c);
    const o2 = orient2d(a, b, d);
    const o3 = orient2d(c, d, a);
    const o4 = orient2d(c, d, b);
    return ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and
        ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0));
}

fn buildExtrudeData(allocator: std.mem.Allocator, options: ExtrudeOptions) !GeometryData {
    const src = options.profile;
    if (src.len < 3) return error.InvalidExtrude;

    // Copy the profile, dropping consecutive duplicates and an explicitly
    // repeated closing point.
    const clean = try allocator.alloc(Vec2, src.len);
    defer allocator.free(clean);
    var m: usize = 0;
    for (src) |p| {
        if (m > 0) {
            const dx = p.x - clean[m - 1].x;
            const dy = p.y - clean[m - 1].y;
            if (dx * dx + dy * dy < 1e-12) continue;
        }
        clean[m] = p;
        m += 1;
    }
    if (m > 1) {
        const dx = clean[m - 1].x - clean[0].x;
        const dy = clean[m - 1].y - clean[0].y;
        if (dx * dx + dy * dy < 1e-12) m -= 1;
    }
    if (m < 3) return error.InvalidExtrude;
    const poly = clean[0..m];

    // Doubled signed area: zero means collinear/degenerate, negative means
    // clockwise (normalized to CCW so side normals point outward).
    var area2: f32 = 0.0;
    for (0..m) |i| {
        const a = poly[i];
        const b = poly[(i + 1) % m];
        area2 += a.x * b.y - b.x * a.y;
    }
    if (@abs(area2) < 1e-9) return error.InvalidExtrude;
    if (area2 < 0.0) {
        var lo: usize = 0;
        var hi: usize = m - 1;
        while (lo < hi) {
            const tmp = poly[lo];
            poly[lo] = poly[hi];
            poly[hi] = tmp;
            lo += 1;
            hi -= 1;
        }
        area2 = -area2;
    }

    // Reject self-intersecting (bow-tie and similar) outlines: only
    // non-adjacent edge pairs are tested.
    for (0..m) |i| {
        const a0 = poly[i];
        const a1 = poly[(i + 1) % m];
        for (i + 1..m) |j| {
            if (j == i + 1) continue;
            if (i == 0 and j == m - 1) continue;
            if (segmentsCross2d(a0, a1, poly[j], poly[(j + 1) % m])) {
                return error.InvalidExtrude;
            }
        }
    }

    // Ear-clipping triangulation of the cap polygon (supports concave shapes).
    const order = try allocator.alloc(usize, m);
    defer allocator.free(order);
    for (0..m) |k| order[k] = k;
    const cap_tris = try allocator.alloc([3]u32, m - 2);
    defer allocator.free(cap_tris);
    var remaining = m;
    var tri_count: usize = 0;
    var guard: usize = 0;
    while (remaining > 3) {
        // Quadratic bound: a simple polygon always has an ear, so exhausting
        // the guard means numerical trouble rather than a valid shape.
        guard += 1;
        if (guard > m * m + 1) return error.InvalidExtrude;
        var ear: ?usize = null;
        var k: usize = 0;
        while (k < remaining) : (k += 1) {
            const ip = order[(k + remaining - 1) % remaining];
            const ic = order[k];
            const inext = order[(k + 1) % remaining];
            if (isEarTip(poly, ip, ic, inext, order[0..remaining])) {
                ear = k;
                break;
            }
        }
        const e = ear orelse return error.InvalidExtrude;
        const et0: u32 = @intCast(order[(e + remaining - 1) % remaining]);
        const et1: u32 = @intCast(order[e]);
        const et2: u32 = @intCast(order[(e + 1) % remaining]);
        cap_tris[tri_count] = .{ et0, et1, et2 };
        tri_count += 1;
        var t = e;
        while (t + 1 < remaining) : (t += 1) order[t] = order[t + 1];
        remaining -= 1;
    }
    const ft0: u32 = @intCast(order[0]);
    const ft1: u32 = @intCast(order[1]);
    const ft2: u32 = @intCast(order[2]);
    cap_tris[tri_count] = .{ ft0, ft1, ft2 };
    tri_count += 1;

    const color = options.color.toArray();
    const depth = options.depth;
    const capped = options.capped;
    const side_verts = 4 * m;
    const cap_verts = if (capped) 2 * m else 0;
    const vertices = try allocator.alloc(Vertex, side_verts + cap_verts);
    errdefer allocator.free(vertices);
    const side_indices = 6 * m;
    const cap_indices = if (capped) 2 * (m - 2) * 3 else 0;
    const indices = try allocator.alloc(u32, side_indices + cap_indices);
    errdefer allocator.free(indices);

    var min_x = poly[0].x;
    var max_x = poly[0].x;
    var min_y = poly[0].y;
    var max_y = poly[0].y;
    for (poly[1..]) |p| {
        min_x = @min(min_x, p.x);
        max_x = @max(max_x, p.x);
        min_y = @min(min_y, p.y);
        max_y = @max(max_y, p.y);
    }

    // Side walls: one flat-shaded quad per outline edge. For a CCW outline
    // the outward normal is the right side of the edge direction.
    var arclen: f32 = 0.0;
    var vi: usize = 0;
    var ii: usize = 0;
    for (0..m) |i| {
        const a = poly[i];
        const b = poly[(i + 1) % m];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len = @sqrt(dx * dx + dy * dy);
        const inv = 1.0 / @max(len, 1e-9);
        const nx = dy * inv;
        const ny = -dx * inv;
        const us = arclen * options.uv_scale.x;
        arclen += len;
        const ue = arclen * options.uv_scale.x;
        const vs: f32 = 0.0;
        const ve = depth * options.uv_scale.y;
        const normal = [3]f32{ nx, ny, 0.0 };
        const tangent = [4]f32{ 0.0, 0.0, 1.0, 1.0 };
        const base: u32 = @intCast(vi);
        vertices[vi + 0] = .{ .position = .{ a.x, a.y, 0.0 }, .normal = normal, .color = color, .uv = .{ us, vs }, .tangent = tangent };
        vertices[vi + 1] = .{ .position = .{ b.x, b.y, 0.0 }, .normal = normal, .color = color, .uv = .{ ue, vs }, .tangent = tangent };
        vertices[vi + 2] = .{ .position = .{ b.x, b.y, depth }, .normal = normal, .color = color, .uv = .{ ue, ve }, .tangent = tangent };
        vertices[vi + 3] = .{ .position = .{ a.x, a.y, depth }, .normal = normal, .color = color, .uv = .{ us, ve }, .tangent = tangent };
        vi += 4;
        // CCW winding for the outward normal.
        indices[ii + 0] = base + 0;
        indices[ii + 1] = base + 1;
        indices[ii + 2] = base + 2;
        indices[ii + 3] = base + 0;
        indices[ii + 4] = base + 2;
        indices[ii + 5] = base + 3;
        ii += 6;
    }

    if (capped) {
        // Caps need axial normals, so they duplicate the outline positions.
        // Layout: front ring (z = depth, +Z), then back ring (z = 0, -Z).
        // Index layout mirrors it: front cap triangles first, then back cap.
        const front_base: u32 = @intCast(vi);
        for (poly) |p| {
            vertices[vi] = .{
                .position = .{ p.x, p.y, depth },
                .normal = .{ 0.0, 0.0, 1.0 },
                .color = color,
                .uv = .{ p.x * options.uv_scale.x, p.y * options.uv_scale.y },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }
        const back_base: u32 = @intCast(vi);
        for (poly) |p| {
            vertices[vi] = .{
                .position = .{ p.x, p.y, 0.0 },
                .normal = .{ 0.0, 0.0, -1.0 },
                .color = color,
                .uv = .{ p.x * options.uv_scale.x, p.y * options.uv_scale.y },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }
        for (cap_tris[0..tri_count]) |tri| {
            // Front cap keeps the CCW (+Z) orientation.
            indices[ii + 0] = front_base + tri[0];
            indices[ii + 1] = front_base + tri[1];
            indices[ii + 2] = front_base + tri[2];
            ii += 3;
        }
        for (cap_tris[0..tri_count]) |tri| {
            // Back cap is mirrored for the -Z normal.
            indices[ii + 0] = back_base + tri[0];
            indices[ii + 1] = back_base + tri[2];
            indices[ii + 2] = back_base + tri[1];
            ii += 3;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(min_x, min_y, @min(0.0, depth)),
            Vec3.new(max_x, max_y, @max(0.0, depth)),
        ),
    };
}

// Uploads CPU-side geometry to sokol buffers, narrowing indices to 16-bit
// when the vertex count allows it (matching the existing builders).
fn uploadGeometry(scene: *Scene, name: []const u8, data: GeometryData) !*Mesh {
    const vbuf = sg.makeBuffer(.{
        .data = sg.asRange(data.vertices),
    });

    const mesh = try scene.allocator.create(Mesh);
    errdefer scene.allocator.destroy(mesh);

    if (data.vertices.len <= std.math.maxInt(u16)) {
        // CPU copy first: it needs the original u32 indices.
        try mesh.retainCpuGeometryU32(scene.allocator, data.vertices, data.indices);
        // The struct is assigned wholesale below, so keep the retained slices
        // from being wiped by the default field values.
        const cpu_positions = mesh.cpu_positions;
        const cpu_indices = mesh.cpu_indices;
        // Narrow the u32 source buffer in place instead of allocating a temp
        // indices16 array. u16[k] lands at byte 2k, strictly behind the u32[k]
        // read at byte 4k, so a forward pass never clobbers unread data.
        // The caller frees this buffer right after upload either way.
        const wide = data.indices;
        const narrow_ptr: [*]u16 = @ptrCast(wide.ptr);
        const indices16 = narrow_ptr[0..wide.len];
        var k: usize = 0;
        const narrow_tail = wide.len & ~@as(usize, 7);
        while (k < narrow_tail) : (k += 8) {
            const v: @Vector(8, u32) = wide[k..][0..8].*;
            const n: @Vector(8, u16) = @truncate(v);
            (narrow_ptr + k)[0..8].* = n;
        }
        while (k < wide.len) : (k += 1) {
            indices16[k] = @intCast(wide[k]);
        }
        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(indices16),
        });
        mesh.* = .{
            .name = name,
            .vertex_buffer = vbuf,
            .index_buffer = ibuf,
            .index_count = @intCast(data.indices.len),
            .local_bounding_box = data.bounds,
            .cpu_positions = cpu_positions,
            .cpu_indices = cpu_indices,
        };
    } else {
        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(data.indices),
        });
        mesh.* = .{
            .name = name,
            .vertex_buffer = vbuf,
            .index_buffer = ibuf,
            .index_count = @intCast(data.indices.len),
            .index_type = .UINT32,
            .local_bounding_box = data.bounds,
        };
        try mesh.retainCpuGeometryU32(scene.allocator, data.vertices, data.indices);
    }

    try scene.meshes.append(scene.allocator, mesh);
    return mesh;
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
        try mesh.retainCpuGeometry(scene.allocator, &vertices, &indices);

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

        const color = options.color.toArray();
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
                    .color = color,
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
        try mesh.retainCpuGeometry(scene.allocator, vertices, indices);

        try scene.meshes.append(scene.allocator, mesh);
        return mesh;
    }

    /// Builds a height-map grid mesh. `heights` is row-major:
    /// index = row * count_x + column, matching the Box3D height field layout.
    /// The local origin is the (0, min_height, 0) corner.
    pub fn createTerrain(
        scene: *Scene,
        name: []const u8,
        heights: []const f32,
        count_x: u32,
        count_z: u32,
        options: TerrainOptions,
    ) !*Mesh {
        const expected = @as(usize, count_x) * count_z;
        if (count_x < 2 or count_z < 2 or heights.len != expected) {
            return error.InvalidTerrainDimensions;
        }

        const vert_count = expected;
        const index_count = (count_x - 1) * (count_z - 1) * 6;
        const sx = options.scale.x;
        const sy = options.scale.y;
        const sz = options.scale.z;

        const vertices = try scene.allocator.alloc(Vertex, vert_count);
        defer scene.allocator.free(vertices);

        var min_y: f32 = heights[0] * sy;
        var max_y: f32 = heights[0] * sy;
        const color = options.color.toArray();
        const u_denom: f32 = @floatFromInt(count_x - 1);
        const v_denom: f32 = @floatFromInt(count_z - 1);

        var vi: usize = 0;
        for (0..count_z) |row| {
            for (0..count_x) |col| {
                const x = @as(f32, @floatFromInt(col)) * sx;
                const z = @as(f32, @floatFromInt(row)) * sz;
                const h = heights[row * count_x + col] * sy;
                min_y = @min(min_y, h);
                max_y = @max(max_y, h);

                // Central-difference normal.
                const col_l = if (col > 0) col - 1 else col;
                const col_r = if (col + 1 < count_x) col + 1 else col;
                const row_u = if (row > 0) row - 1 else row;
                const row_d = if (row + 1 < count_z) row + 1 else row;
                const dh_dx = (heights[row * count_x + col_r] - heights[row * count_x + col_l]) * sy;
                const dh_dz = (heights[row_d * count_x + col] - heights[row_u * count_x + col]) * sy;
                const dx = @as(f32, @floatFromInt(col_r - col_l)) * sx;
                const dz = @as(f32, @floatFromInt(row_d - row_u)) * sz;
                const tx = Vec3.new(if (dx > 0.0) dx else 1.0, dh_dx, 0.0);
                const tz = Vec3.new(0.0, dh_dz, if (dz > 0.0) dz else 1.0);
                const normal = Vec3.new(
                    tz.y * tx.z - tz.z * tx.y,
                    tz.z * tx.x - tz.x * tx.z,
                    tz.x * tx.y - tz.y * tx.x,
                ).normalize();

                vertices[vi] = .{
                    .position = .{ x, h, z },
                    .normal = .{ normal.x, normal.y, normal.z },
                    .color = color,
                    .uv = .{
                        @as(f32, @floatFromInt(col)) / u_denom,
                        @as(f32, @floatFromInt(row)) / v_denom,
                    },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
                vi += 1;
            }
        }

        const indices = try scene.allocator.alloc(u32, index_count);
        defer scene.allocator.free(indices);
        var ii: usize = 0;
        const row_stride = count_x;
        for (0..count_z - 1) |row| {
            for (0..count_x - 1) |col| {
                const p0: u32 = @intCast(row * row_stride + col);
                const p1: u32 = @intCast(row * row_stride + col + 1);
                const p2: u32 = @intCast((row + 1) * row_stride + col + 1);
                const p3: u32 = @intCast((row + 1) * row_stride + col);

                // CCW winding for +Y normal.
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
            .index_count = index_count,
            .index_type = .UINT32,
            .local_bounding_box = BoundingBox.init(
                Vec3.new(0.0, min_y, 0.0),
                Vec3.new(@as(f32, @floatFromInt(count_x - 1)) * sx, max_y, @as(f32, @floatFromInt(count_z - 1)) * sz),
            ),
        };
        try mesh.retainCpuGeometryU32(scene.allocator, vertices, indices);

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
        const color = options.color.toArray();
        // Each unique slice angle computed once, shared by all rings.
        const slice_tab = try scene.allocator.alloc(TrigEntry, slices + 1);
        defer scene.allocator.free(slice_tab);
        const slices_f: f32 = @floatFromInt(slices);
        for (0..slices + 1) |s| {
            const u = @as(f32, @floatFromInt(s)) / slices_f;
            const theta = 2.0 * pi * u;
            slice_tab[s] = .{ .cos = @cos(theta), .sin = @sin(theta), .f = u };
        }

        var vi: usize = 0;
        for (0..rings + 1) |r| {
            const v = @as(f32, @floatFromInt(r)) / @as(f32, @floatFromInt(rings));
            const phi = -pi * 0.5 + pi * v;
            const cos_phi = @cos(phi);
            const sin_phi = @sin(phi);

            for (0..slices + 1) |s| {
                const u = slice_tab[s].f;
                const cos_theta = slice_tab[s].cos;
                const sin_theta = slice_tab[s].sin;

                const nx = cos_phi * sin_theta;
                const ny = sin_phi;
                const nz = cos_phi * cos_theta;

                vertices[vi] = .{
                    .position = .{ nx * radius, ny * radius, nz * radius },
                    .normal = .{ nx, ny, nz },
                    .color = color,
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
        try mesh.retainCpuGeometry(scene.allocator, vertices, indices);

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
        const color = options.color.toArray();
        // One ring table shared by the side and both cap loops.
        const ring_tab = try scene.allocator.alloc(TrigEntry, tess + 1);
        defer scene.allocator.free(ring_tab);
        const tess_f: f32 = @floatFromInt(tess);
        for (0..tess + 1) |i| {
            const u = @as(f32, @floatFromInt(i)) / tess_f;
            const theta = 2.0 * pi * u;
            ring_tab[i] = .{ .cos = @cos(theta), .sin = @sin(theta), .f = u };
        }
        var vi: usize = 0;

        // 1. Sides
        for (0..tess + 1) |i| {
            const u = ring_tab[i].f;
            const nx = ring_tab[i].sin;
            const nz = ring_tab[i].cos;

            // Bottom
            vertices[vi] = .{
                .position = .{ nx * radius, -half_h, nz * radius },
                .normal = .{ nx, 0.0, nz },
                .color = color,
                .uv = .{ u, 0.0 },
                .tangent = .{ nz, 0.0, -nx, 1.0 },
            };
            vi += 1;

            // Top
            vertices[vi] = .{
                .position = .{ nx * radius, half_h, nz * radius },
                .normal = .{ nx, 0.0, nz },
                .color = color,
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
            .color = color,
            .uv = .{ 0.5, 0.5 },
            .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
        };
        vi += 1;

        const top_ring_start: u16 = @intCast(vi);
        for (0..tess + 1) |i| {
            const x = ring_tab[i].sin;
            const z = ring_tab[i].cos;
            vertices[vi] = .{
                .position = .{ x * radius, half_h, z * radius },
                .normal = .{ 0.0, 1.0, 0.0 },
                .color = color,
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
            .color = color,
            .uv = .{ 0.5, 0.5 },
            .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
        };
        vi += 1;

        const bot_ring_start: u16 = @intCast(vi);
        for (0..tess + 1) |i| {
            const x = ring_tab[i].sin;
            const z = ring_tab[i].cos;
            vertices[vi] = .{
                .position = .{ x * radius, -half_h, z * radius },
                .normal = .{ 0.0, -1.0, 0.0 },
                .color = color,
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
        try mesh.retainCpuGeometry(scene.allocator, vertices, indices);

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
        const color = options.color.toArray();
        // Each unique slice angle computed once, shared by all rings.
        const slice_tab = try scene.allocator.alloc(TrigEntry, slices + 1);
        defer scene.allocator.free(slice_tab);
        const slices_f: f32 = @floatFromInt(slices);
        for (0..slices + 1) |s| {
            const u = @as(f32, @floatFromInt(s)) / slices_f;
            const theta = 2.0 * pi * u;
            slice_tab[s] = .{ .cos = @cos(theta), .sin = @sin(theta), .f = u };
        }
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
                const u = slice_tab[s].f;
                const cos_theta = slice_tab[s].cos;
                const sin_theta = slice_tab[s].sin;

                const nx = cos_phi * sin_theta;
                const ny = sin_phi;
                const nz = cos_phi * cos_theta;

                vertices[vi] = .{
                    .position = .{ nx * radius, center_y + ny * radius, nz * radius },
                    .normal = .{ nx, ny, nz },
                    .color = color,
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
        try mesh.retainCpuGeometry(scene.allocator, vertices, indices);

        try scene.meshes.append(scene.allocator, mesh);
        return mesh;
    }

    pub fn createTorus(scene: *Scene, name: []const u8, options: TorusOptions) !*Mesh {
        var data = try buildTorusData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createTorusKnot(scene: *Scene, name: []const u8, options: TorusKnotOptions) !*Mesh {
        var data = try buildTorusKnotData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createDisc(scene: *Scene, name: []const u8, options: DiscOptions) !*Mesh {
        var data = try buildDiscData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createRibbon(scene: *Scene, name: []const u8, options: RibbonOptions) !*Mesh {
        var data = try buildRibbonData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createLathe(scene: *Scene, name: []const u8, options: LatheOptions) !*Mesh {
        var data = try buildLatheData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createPlane(scene: *Scene, name: []const u8, options: PlaneOptions) !*Mesh {
        var data = try buildPlaneData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createTube(scene: *Scene, name: []const u8, options: TubeOptions) !*Mesh {
        var data = try buildTubeData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createLines(scene: *Scene, name: []const u8, options: LinesOptions) !*Mesh {
        var data = try buildLinesData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }

    pub fn createExtrude(scene: *Scene, name: []const u8, options: ExtrudeOptions) !*Mesh {
        var data = try buildExtrudeData(scene.allocator, options);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }
};

test "Mesh attachToBone world matrix computation" {
    const ally = std.testing.allocator;
    const skel = try Skeleton.init(ally, 2);
    defer skel.deinit();

    skel.bones[0].local_position = Vec3.new(2, 0, 0);
    skel.bones[1].parent_index = 0;
    skel.bones[1].local_position = Vec3.new(0, 3, 0);
    skel.update();

    var host_mesh: Mesh = undefined;
    host_mesh = .{
        .name = "host",
        .position = Vec3.new(10, 20, 30),
        .rotation = Vec3.zero,
        .scaling = Vec3.one,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .skeleton = skel,
    };

    var attached_mesh: Mesh = undefined;
    attached_mesh = .{
        .name = "sword",
        .position = Vec3.new(0, 0, 1), // local offset relative to bone
        .rotation = Vec3.zero,
        .scaling = Vec3.one,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };

    attached_mesh.attachToBone(&host_mesh, 1);

    const world = attached_mesh.getWorldMatrix();
    const pos = world.getTranslation();

    // host (10, 20, 30) + bone1 (2, 3, 0) + local (0, 0, 1) = (12, 23, 31)
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), pos.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 23.0), pos.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), pos.z, 1e-4);
}

fn expectNormalsNormalized(vertices: []const Vertex) !void {
    for (vertices) |v| {
        const n = Vec3.new(v.normal[0], v.normal[1], v.normal[2]);
        try std.testing.expectApproxEqAbs(1.0, n.length(), 1e-4);
    }
}

fn expectIndicesInBounds(indices: []const u32, vertex_count: usize) !void {
    for (indices) |ix| {
        try std.testing.expect(ix < vertex_count);
    }
}

test "MeshBuilder torus geometry" {
    const ally = std.testing.allocator;
    var data = try buildTorusData(ally, .{ .diameter = 2.0, .thickness = 0.5, .tessellation = 8 });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 81), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 384), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Ring radius 1.0, tube radius 0.25: symmetric bounds.
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.25), data.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), data.bounds.max.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -0.25), data.bounds.min.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), data.bounds.max.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.25), data.bounds.min.z, 1e-4);

    for (data.vertices) |v| {
        try std.testing.expect(v.uv[0] >= 0.0 and v.uv[0] <= 1.0);
        try std.testing.expect(v.uv[1] >= 0.0 and v.uv[1] <= 1.0);
    }
}

test "MeshBuilder torus knot geometry" {
    const ally = std.testing.allocator;
    const tubular: usize = 16;
    const radial: usize = 6;
    var data = try buildTorusKnotData(ally, .{
        .radius = 3.0,
        .tube = 0.4,
        .radial_segments = 6,
        .tubular_segments = 16,
    });
    defer data.deinit(ally);

    try std.testing.expectEqual((tubular + 1) * (radial + 1), data.vertices.len);
    try std.testing.expectEqual(tubular * radial * 6, data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Knot is centered on the origin: bounds roughly symmetric.
    const center = data.bounds.center();
    try std.testing.expect(@abs(center.x) < 0.2);
    try std.testing.expect(@abs(center.y) < 0.2);
    try std.testing.expect(@abs(center.z) < 0.2);
    // Outer extent stays within radius + tube (plus a small margin).
    try std.testing.expect(data.bounds.max.x <= 3.4 + 1e-3);
    try std.testing.expect(data.bounds.min.x >= -3.4 - 1e-3);
}

test "MeshBuilder disc geometry" {
    const ally = std.testing.allocator;
    var data = try buildDiscData(ally, .{ .radius = 2.0, .tessellation = 8 });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 10), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 24), data.indices.len);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Flat in XZ, normal +Y, UVs mapped from disc coordinates.
    for (data.vertices) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v.position[1], 1e-6);
        try std.testing.expectEqual([3]f32{ 0.0, 1.0, 0.0 }, v.normal);
        try std.testing.expect(v.uv[0] >= 0.0 and v.uv[0] <= 1.0);
        try std.testing.expect(v.uv[1] >= 0.0 and v.uv[1] <= 1.0);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), data.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), data.bounds.min.z, 1e-4);
}

test "MeshBuilder ribbon geometry" {
    const ally = std.testing.allocator;
    const path_a = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(1, 0, 0),
        Vec3.new(2, 0, 0),
    };
    const path_b = [_]Vec3{
        Vec3.new(0, 0, 1),
        Vec3.new(1, 1, 1),
        Vec3.new(2, 0, 1),
    };
    const paths = [_][]const Vec3{ &path_a, &path_b };

    var data = try buildRibbonData(ally, .{ .paths = &paths });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 6), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 12), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Closed variants produce the wrapped quad counts.
    var closed = try buildRibbonData(ally, .{ .paths = &paths, .close_path = true, .close_array = false });
    defer closed.deinit(ally);
    try std.testing.expectEqual(@as(usize, 18), closed.indices.len);

    var closed_both = try buildRibbonData(ally, .{ .paths = &paths, .close_path = true, .close_array = true });
    defer closed_both.deinit(ally);
    try std.testing.expectEqual(@as(usize, 36), closed_both.indices.len);
}

test "MeshBuilder ribbon rejects invalid paths" {
    const ally = std.testing.allocator;
    const short = [_]Vec3{Vec3.new(0, 0, 0)};
    const ok_path = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0) };

    const single = [_][]const Vec3{&ok_path};
    try std.testing.expectError(error.InvalidRibbon, buildRibbonData(ally, .{ .paths = &single }));

    const one_short = [_][]const Vec3{ &ok_path, &short };
    try std.testing.expectError(error.InvalidRibbon, buildRibbonData(ally, .{ .paths = &one_short }));

    const longer = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(2, 0, 0) };
    const mismatched = [_][]const Vec3{ &ok_path, &longer };
    try std.testing.expectError(error.InvalidRibbon, buildRibbonData(ally, .{ .paths = &mismatched }));
}

test "MeshBuilder lathe geometry" {
    const ally = std.testing.allocator;
    const profile = [_]Vec3{
        Vec3.new(0.5, 0.0, 0.0),
        Vec3.new(0.5, 2.0, 0.0),
    };
    var data = try buildLatheData(ally, .{ .shape = &profile, .tessellation = 8 });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 18), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 48), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Cylinder wall: radial normals, symmetric XZ bounds.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), data.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.y, 1e-4);

    // Degenerate on-axis segment is skipped: only one quad strip remains.
    const with_axis = [_]Vec3{
        Vec3.new(0.0, 0.0, 0.0),
        Vec3.new(0.0, 1.0, 0.0),
        Vec3.new(0.5, 1.0, 0.0),
    };
    var deg = try buildLatheData(ally, .{ .shape = &with_axis, .tessellation = 8 });
    defer deg.deinit(ally);
    try std.testing.expectEqual(@as(usize, 27), deg.vertices.len);
    try std.testing.expectEqual(@as(usize, 48), deg.indices.len);
    try expectNormalsNormalized(deg.vertices);

    const single = [_]Vec3{Vec3.new(0.5, 0.0, 0.0)};
    try std.testing.expectError(error.InvalidLathe, buildLatheData(ally, .{ .shape = &single }));
}

test "MeshBuilder plane geometry" {
    const ally = std.testing.allocator;
    var data = try buildPlaneData(ally, .{
        .width = 2.0,
        .height = 4.0,
        .subdivisions_x = 3,
        .subdivisions_y = 2,
        .uv_scale = Vec2.new(2.0, 3.0),
    });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 12), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 36), data.indices.len);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Centered in XY, flat at z = 0, normals +Z.
    for (data.vertices) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v.position[2], 1e-6);
        try std.testing.expectEqual([3]f32{ 0.0, 0.0, 1.0 }, v.normal);
    }
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), data.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), data.bounds.min.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.z, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.max.z, 1e-6);

    // UVs span the per-axis scale.
    var max_u: f32 = 0.0;
    var max_v: f32 = 0.0;
    for (data.vertices) |v| {
        max_u = @max(max_u, v.uv[0]);
        max_v = @max(max_v, v.uv[1]);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), max_u, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), max_v, 1e-4);

    // Zero subdivisions clamp to a single quad.
    var flat = try buildPlaneData(ally, .{ .subdivisions_x = 0, .subdivisions_y = 0 });
    defer flat.deinit(ally);
    try std.testing.expectEqual(@as(usize, 4), flat.vertices.len);
    try std.testing.expectEqual(@as(usize, 6), flat.indices.len);
}

test "MeshBuilder tube geometry" {
    const ally = std.testing.allocator;
    const path = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(0, 1, 0),
        Vec3.new(0, 2, 0),
    };
    var data = try buildTubeData(ally, .{ .path = &path, .radius = 0.5, .tessellation = 6 });
    defer data.deinit(ally);

    // Open tube: one ring per point, one quad strip per segment.
    try std.testing.expectEqual(@as(usize, 21), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 72), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Straight path along Y: AABB is the path extent plus the radius in XZ.
    // (Ring samples may not hit the exact Z extremes, so those assert range.)
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), data.bounds.min.x, 1e-4);
    try std.testing.expect(data.bounds.max.z <= 0.5 + 1e-4);
    try std.testing.expect(data.bounds.min.z >= -0.5 - 1e-4);
    try std.testing.expect(data.bounds.max.z > 0.4);
    try std.testing.expect(data.bounds.min.z < -0.4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.y, 1e-4);

    // U runs along the path, V around the tube.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.vertices[0].uv[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.vertices[14].uv[0], 1e-6);
    for (data.vertices) |v| {
        try std.testing.expect(v.uv[0] >= 0.0 and v.uv[0] <= 1.0);
        try std.testing.expect(v.uv[1] >= 0.0 and v.uv[1] <= 1.0);
    }
}

test "MeshBuilder tube capped closed radii" {
    const ally = std.testing.allocator;
    const square = [_]Vec3{
        Vec3.new(-1, 0, -1),
        Vec3.new(1, 0, -1),
        Vec3.new(1, 0, 1),
        Vec3.new(-1, 0, 1),
    };
    const radii = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    var data = try buildTubeData(ally, .{
        .path = &square,
        .radii = &radii,
        .tessellation = 4,
        .closed = true,
        .capped = true, // ignored on closed loops
    });
    defer data.deinit(ally);

    // Closed loop: one ring per point, wrapped strips, no caps.
    try std.testing.expectEqual(@as(usize, 20), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 96), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Per-point radii: the first ring vertex sits one radius from its center.
    for (0..4) |i| {
        const v = data.vertices[i * 5];
        const px = v.position[0] - square[i].x;
        const py = v.position[1] - square[i].y;
        const pz = v.position[2] - square[i].z;
        try std.testing.expectApproxEqAbs(radii[i], @sqrt(px * px + py * py + pz * pz), 1e-4);
    }

    // Capped open tube adds a duplicated ring plus a center vertex per cap.
    var capped = try buildTubeData(ally, .{
        .path = &square,
        .radius = 0.1,
        .tessellation = 4,
        .capped = true,
    });
    defer capped.deinit(ally);
    try std.testing.expectEqual(@as(usize, 4 * 5 + 2 * 5 + 2), capped.vertices.len);
    try std.testing.expectEqual(@as(usize, 3 * 4 * 6 + 2 * 4 * 3), capped.indices.len);
    try expectNormalsNormalized(capped.vertices);
    try expectIndicesInBounds(capped.indices, capped.vertices.len);
}

test "MeshBuilder tube rejects invalid input" {
    const ally = std.testing.allocator;
    const single = [_]Vec3{Vec3.new(0, 0, 0)};
    try std.testing.expectError(error.InvalidTube, buildTubeData(ally, .{ .path = &single }));

    const ok_path = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(0, 1, 0) };
    const bad_radii = [_]f32{0.1};
    try std.testing.expectError(error.InvalidTube, buildTubeData(ally, .{ .path = &ok_path, .radii = &bad_radii }));
}

test "MeshBuilder lines geometry" {
    const ally = std.testing.allocator;
    const points = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(2, 0, 0),
        Vec3.new(4, 0, 0),
    };
    const colors = [_]Color4{
        Color4.new(1, 0, 0, 1),
        Color4.new(0, 1, 0, 1),
        Color4.new(0, 0, 1, 1),
    };
    var data = try buildLinesData(ally, .{ .points = &points, .width = 2.0, .colors = &colors });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 6), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 12), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Ribbon width spans both sides of the center line.
    for (0..3) |i| {
        const l = Vec3.new(data.vertices[2 * i].position[0], data.vertices[2 * i].position[1], data.vertices[2 * i].position[2]);
        const r = Vec3.new(data.vertices[2 * i + 1].position[0], data.vertices[2 * i + 1].position[1], data.vertices[2 * i + 1].position[2]);
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), l.distance(r), 1e-4);
        // Per-point colors apply to both vertices of the pair.
        try std.testing.expectEqual(colors[i].toArray(), data.vertices[2 * i].color);
        try std.testing.expectEqual(colors[i].toArray(), data.vertices[2 * i + 1].color);
    }

    // U follows the arc length: equal segments give 0, 0.5, 1.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.vertices[0].uv[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data.vertices[2].uv[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.vertices[4].uv[0], 1e-5);

    // Closed loop wraps the last strip back to the first, without caps.
    var closed = try buildLinesData(ally, .{ .points = &points, .width = 1.0, .closed = true });
    defer closed.deinit(ally);
    try std.testing.expectEqual(@as(usize, 6), closed.vertices.len);
    try std.testing.expectEqual(@as(usize, 18), closed.indices.len);
    try expectNormalsNormalized(closed.vertices);
    try expectIndicesInBounds(closed.indices, closed.vertices.len);
}

test "MeshBuilder lines rejects invalid input" {
    const ally = std.testing.allocator;
    const single = [_]Vec3{Vec3.new(0, 0, 0)};
    try std.testing.expectError(error.InvalidLines, buildLinesData(ally, .{ .points = &single }));

    const ok_points = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0) };
    const bad_colors = [_]Color4{Color4.white};
    try std.testing.expectError(error.InvalidLines, buildLinesData(ally, .{ .points = &ok_points, .colors = &bad_colors }));
}

test "MeshBuilder extrude convex geometry" {
    const ally = std.testing.allocator;
    const square = [_]Vec2{
        Vec2.new(0, 0),
        Vec2.new(1, 0),
        Vec2.new(1, 1),
        Vec2.new(0, 1),
    };
    var data = try buildExtrudeData(ally, .{ .profile = &square, .depth = 2.0 });
    defer data.deinit(ally);

    // 4 outline points: 4 side quads (16 verts) + 2 cap rings (8 verts);
    // 24 side indices + 2 caps x 2 triangles x 3.
    try std.testing.expectEqual(@as(usize, 24), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 36), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // AABB covers the profile footprint and the extrusion depth.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.bounds.max.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.bounds.max.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.z, 1e-5);

    // First side quad is the bottom edge: outward normal -Y.
    try std.testing.expectEqual([3]f32{ 0.0, -1.0, 0.0 }, data.vertices[0].normal);
    // Caps carry axial normals: front ring starts at vertex 16.
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 1.0 }, data.vertices[16].normal);
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, -1.0 }, data.vertices[20].normal);

    // Clockwise input is normalized: same counts, still outward normals.
    const square_cw = [_]Vec2{
        Vec2.new(0, 0),
        Vec2.new(0, 1),
        Vec2.new(1, 1),
        Vec2.new(1, 0),
    };
    var cw = try buildExtrudeData(ally, .{ .profile = &square_cw, .depth = 2.0 });
    defer cw.deinit(ally);
    try std.testing.expectEqual(data.vertices.len, cw.vertices.len);
    try std.testing.expectEqual(data.indices.len, cw.indices.len);
    try expectNormalsNormalized(cw.vertices);
    var found_outward = false;
    for (cw.vertices[0..16]) |v| {
        if (v.normal[0] == 0.0 and v.normal[1] == -1.0 and v.normal[2] == 0.0) found_outward = true;
    }
    try std.testing.expect(found_outward);

    // Uncapped extrusion keeps only the side walls.
    var open = try buildExtrudeData(ally, .{ .profile = &square, .depth = 2.0, .capped = false });
    defer open.deinit(ally);
    try std.testing.expectEqual(@as(usize, 16), open.vertices.len);
    try std.testing.expectEqual(@as(usize, 24), open.indices.len);
}

test "MeshBuilder extrude concave L-shape geometry" {
    const ally = std.testing.allocator;
    // 2x2 square with the top-right 1x1 quadrant removed: area 3, reflex at (1, 1).
    const ell = [_]Vec2{
        Vec2.new(0, 0),
        Vec2.new(2, 0),
        Vec2.new(2, 1),
        Vec2.new(1, 1),
        Vec2.new(1, 2),
        Vec2.new(0, 2),
    };
    var data = try buildExtrudeData(ally, .{ .profile = &ell, .depth = 1.0 });
    defer data.deinit(ally);

    // 6 outline points: 24 side verts + 12 cap verts; each cap has 6 - 2 triangles.
    try std.testing.expectEqual(@as(usize, 36), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 60), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Front cap triangles (indices 36..48) must tile the L area of 3.
    var cap_area: f32 = 0.0;
    var t: usize = 36;
    while (t < 48) : (t += 3) {
        const p0 = data.vertices[data.indices[t]].position;
        const p1 = data.vertices[data.indices[t + 1]].position;
        const p2 = data.vertices[data.indices[t + 2]].position;
        const e1x = p1[0] - p0[0];
        const e1y = p1[1] - p0[1];
        const e2x = p2[0] - p0[0];
        const e2y = p2[1] - p0[1];
        cap_area += @abs(e1x * e2y - e2x * e1y) * 0.5;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), cap_area, 1e-4);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.z, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.bounds.max.z, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.y, 1e-4);
}

test "MeshBuilder extrude rejects invalid input" {
    const ally = std.testing.allocator;
    const two = [_]Vec2{ Vec2.new(0, 0), Vec2.new(1, 0) };
    try std.testing.expectError(error.InvalidExtrude, buildExtrudeData(ally, .{ .profile = &two }));

    // Collinear points have zero area.
    const collinear = [_]Vec2{ Vec2.new(0, 0), Vec2.new(1, 0), Vec2.new(2, 0) };
    try std.testing.expectError(error.InvalidExtrude, buildExtrudeData(ally, .{ .profile = &collinear }));

    // Bow-tie outline self-intersects.
    const bowtie = [_]Vec2{ Vec2.new(0, 0), Vec2.new(1, 1), Vec2.new(1, 0), Vec2.new(0, 1) };
    try std.testing.expectError(error.InvalidExtrude, buildExtrudeData(ally, .{ .profile = &bowtie }));
}
