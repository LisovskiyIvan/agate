const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;

const Mesh = @import("mesh.zig").Mesh;
const Vertex = @import("types.zig").Vertex;
const Scene = @import("../scene.zig").Scene;
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");

pub const TrailOptions = struct {
    /// Width of the ribbon at the head (most recent position).
    diameter: f32 = 0.35,
    /// Maximum number of segments along the trail.
    segments: u32 = 64,
    /// Lifetime of a trail node in seconds before disappearing.
    lifetime: f32 = 1.2,
    /// Minimum movement distance between recorded nodes (avoids clustering when stationary).
    min_distance: f32 = 0.04,
    /// Whether the ribbon tapers from diameter at head to 0 at tail.
    taper: bool = true,
    /// Color at the head of the trail.
    color_start: Color4 = Color4.new(1.0, 1.0, 1.0, 1.0),
    /// Color at the tail of the trail (alpha typically 0.0 for smooth fade-out).
    color_end: Color4 = Color4.new(1.0, 1.0, 1.0, 0.0),
    /// Whether to start recording automatically upon creation.
    auto_start: bool = true,
};

pub const TrailNode = struct {
    position: Vec3,
    age: f32 = 0.0,
};

/// Dynamic ribbon mesh following a target object, bone, or world position.
/// Similar to Babylon.js TrailMesh: generates camera-facing billboard quad strips
/// with customizable width tapering, lifetime fade-out, and vertex gradients.
pub const TrailMesh = struct {
    allocator: std.mem.Allocator,
    scene: *Scene,
    mesh: *Mesh,
    target: ?*Mesh = null,
    target_offset: Vec3 = Vec3.zero,
    options: TrailOptions,
    nodes: std.ArrayListUnmanaged(TrailNode) = .empty,
    vertices: []Vertex,
    indices: []u16,
    is_active: bool = true,
    last_position: ?Vec3 = null,
    /// Stage 3: update() stages CPU data and sets this flag; the sg
    /// upload happens in flushGpuUploads on the render side.
    gpu_dirty: bool = false,
    /// True when the vertex/index buffers could not be created at init time
    /// (off-context `createTrailMesh`): flushGpuUploads creates them on the
    /// render side once a context is available, then uploads staged data.
    buffers_pending: bool = false,
    pending_vertex_count: usize = 0,
    pending_index_count: usize = 0,
    pending_min_pt: Vec3 = undefined,
    pending_max_pt: Vec3 = undefined,

    pub fn init(scene: *Scene, name: []const u8, options: TrailOptions) !*TrailMesh {
        const allocator = scene.allocator;
        const max_segments = @max(options.segments, 2);
        const max_nodes = max_segments + 1;
        const max_verts = max_nodes * 2;
        const max_indices = max_segments * 6;

        const vertices = try allocator.alloc(Vertex, max_verts);
        errdefer allocator.free(vertices);
        const indices = try allocator.alloc(u16, max_indices);
        errdefer allocator.free(indices);

        // Dynamic vertex & index buffers. Off-context construction (runtime
        // spawn on the game thread) defers creation to flushGpuUploads.
        const deferred = !gpu_thread.isOnContextThread();
        const vb = if (deferred) sg.Buffer{} else sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .dynamic_update = true },
            .size = max_verts * @sizeOf(Vertex),
        });
        const ib = if (deferred) sg.Buffer{} else sg.makeBuffer(.{
            .usage = .{ .index_buffer = true, .dynamic_update = true },
            .size = max_indices * @sizeOf(u16),
        });

        const mesh = try allocator.create(Mesh);
        errdefer allocator.destroy(mesh);
        mesh.* = .{
            .name = name,
            .vertex_buffer = vb,
            .index_buffer = ib,
            .index_count = 0,
            .index_type = .UINT16,
            .local_bounding_box = BoundingBox.zero,
            .cast_shadows = false,
        };
        try scene.meshes.append(allocator, mesh);

        const self = try allocator.create(TrailMesh);
        self.* = .{
            .allocator = allocator,
            .scene = scene,
            .mesh = mesh,
            .options = options,
            .vertices = vertices,
            .indices = indices,
            .is_active = options.auto_start,
            .buffers_pending = deferred,
        };
        return self;
    }

    /// Attaches the trail to a host mesh with an optional local-space offset.
    pub fn setTarget(self: *TrailMesh, target: *Mesh, offset: Vec3) void {
        self.target = target;
        self.target_offset = offset;
    }

    /// Manually appends a new trail position.
    pub fn addNode(self: *TrailMesh, position: Vec3) !void {
        try self.nodes.insert(self.allocator, 0, .{
            .position = position,
            .age = 0.0,
        });
        self.last_position = position;

        // Cap to maximum allowed nodes
        const max_nodes = self.options.segments + 1;
        while (self.nodes.items.len > max_nodes) {
            _ = self.nodes.pop();
        }
    }

    /// Clears all active trail points.
    pub fn reset(self: *TrailMesh) void {
        self.nodes.clearRetainingCapacity();
        self.last_position = null;
        self.mesh.index_count = 0;
    }

    /// Updates trail history and rebuilds ribbon geometry.
    /// `camera_pos` is used for billboard ribbon orientation facing the viewer.
    pub fn update(self: *TrailMesh, dt: f32, camera_pos: Vec3) void {
        if (!self.is_active) return;

        // 1. Sample target movement
        if (self.target) |tgt| {
            const world_mat = tgt.getWorldMatrix();
            const cur_pos = world_mat.transformPoint(self.target_offset);
            const dist = if (self.last_position) |lp| cur_pos.distance(lp) else (self.options.min_distance + 1.0);
            if (dist >= self.options.min_distance) {
                self.addNode(cur_pos) catch {};
            }
        }

        // 2. Age nodes
        for (self.nodes.items) |*node| {
            node.age += dt;
        }

        // 3. Prune expired nodes from tail
        while (self.nodes.items.len > 0) {
            const tail_idx = self.nodes.items.len - 1;
            if (self.nodes.items[tail_idx].age > self.options.lifetime) {
                _ = self.nodes.pop();
            } else {
                break;
            }
        }

        const count = self.nodes.items.len;
        if (count < 2) {
            self.mesh.index_count = 0;
            return;
        }

        // 4. Generate ribbon geometry
        var min_pt = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
        var max_pt = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));

        var vi: usize = 0;
        for (self.nodes.items, 0..) |node, i| {
            const t = std.math.clamp(node.age / self.options.lifetime, 0.0, 1.0);
            const width = self.options.diameter * (if (self.options.taper) (1.0 - t) else 1.0);

            // Compute movement tangent along trail
            var dir: Vec3 = undefined;
            if (i == 0) {
                dir = self.nodes.items[0].position.sub(self.nodes.items[1].position);
            } else if (i == count - 1) {
                dir = self.nodes.items[count - 2].position.sub(self.nodes.items[count - 1].position);
            } else {
                dir = self.nodes.items[i - 1].position.sub(self.nodes.items[i + 1].position);
            }

            const dir_len_sq = dir.lengthSq();
            if (dir_len_sq > 1e-8) {
                dir = dir.scale(1.0 / @sqrt(dir_len_sq));
            } else {
                dir = Vec3.new(0.0, 1.0, 0.0);
            }

            // Billboard side vector perpendicular to movement and camera vector
            var to_cam = camera_pos.sub(node.position);
            const cam_len_sq = to_cam.lengthSq();
            if (cam_len_sq > 1e-8) {
                to_cam = to_cam.scale(1.0 / @sqrt(cam_len_sq));
            } else {
                to_cam = Vec3.new(0.0, 0.0, 1.0);
            }

            var side = dir.cross(to_cam);
            var side_len_sq = side.lengthSq();
            if (side_len_sq < 1e-6) {
                side = dir.cross(Vec3.up);
                side_len_sq = side.lengthSq();
                if (side_len_sq < 1e-6) {
                    side = dir.cross(Vec3.new(1.0, 0.0, 0.0));
                    side_len_sq = side.lengthSq();
                }
            }
            if (side_len_sq > 1e-8) {
                side = side.scale(1.0 / @sqrt(side_len_sq));
            } else {
                side = Vec3.new(1.0, 0.0, 0.0);
            }

            const normal = side.cross(dir).normalize();
            const col = Color4.lerp(self.options.color_start, self.options.color_end, t);
            const half_offset = side.scale(width * 0.5);

            const v_left = node.position.sub(half_offset);
            const v_right = node.position.add(half_offset);

            min_pt.x = @min(min_pt.x, @min(v_left.x, v_right.x));
            min_pt.y = @min(min_pt.y, @min(v_left.y, v_right.y));
            min_pt.z = @min(min_pt.z, @min(v_left.z, v_right.z));
            max_pt.x = @max(max_pt.x, @max(v_left.x, v_right.x));
            max_pt.y = @max(max_pt.y, @max(v_left.y, v_right.y));
            max_pt.z = @max(max_pt.z, @max(v_left.z, v_right.z));

            self.vertices[vi + 0] = .{
                .position = v_left.toArray(),
                .normal = normal.toArray(),
                .color = col.toArray(),
                .uv = .{ 0.0, t },
                .tangent = .{ side.x, side.y, side.z, 1.0 },
            };
            self.vertices[vi + 1] = .{
                .position = v_right.toArray(),
                .normal = normal.toArray(),
                .color = col.toArray(),
                .uv = .{ 1.0, t },
                .tangent = .{ side.x, side.y, side.z, 1.0 },
            };
            vi += 2;
        }

        // Quad strip indices
        var ii: usize = 0;
        var s: u16 = 0;
        while (s < count - 1) : (s += 1) {
            const base = s * 2;
            self.indices[ii + 0] = base + 0;
            self.indices[ii + 1] = base + 1;
            self.indices[ii + 2] = base + 2;

            self.indices[ii + 3] = base + 1;
            self.indices[ii + 4] = base + 3;
            self.indices[ii + 5] = base + 2;
            ii += 6;
        }

        // Stage CPU data only; the sg buffer upload happens in
        // `flushGpuUploads` on the render side (sg is single-context —
        // the update phase must stay free of sg.* calls).
        self.pending_vertex_count = vi;
        self.pending_index_count = ii;
        self.pending_min_pt = min_pt;
        self.pending_max_pt = max_pt;
        self.gpu_dirty = true;
    }

    /// Uploads staged trail geometry. Runs on the sg-context thread
    /// (Scene.render start), never during the update phase.
    pub fn flushGpuUploads(self: *TrailMesh) void {
        // Deferred construction (off-context spawn): create the buffers
        // here, on the render side, then upload whatever was staged.
        if (self.buffers_pending and sg.isvalid()) {
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = self.vertices.len * @sizeOf(Vertex),
            });
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true, .dynamic_update = true },
                .size = self.indices.len * @sizeOf(u16),
            });
            if (vb.id != 0 and ib.id != 0) {
                self.mesh.vertex_buffer = vb;
                self.mesh.index_buffer = ib;
                self.buffers_pending = false;
                if (self.pending_index_count > 0) self.gpu_dirty = true;
            } else {
                // Partial creation (pool exhaustion): destroy whatever was
                // created so the retry next frame does not leak handles.
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
            }
        }
        if (!self.gpu_dirty) return;
        self.gpu_dirty = false;
        if (self.mesh.vertex_buffer.id != 0) {
            sg.updateBuffer(self.mesh.vertex_buffer, sg.asRange(self.vertices[0..self.pending_vertex_count]));
            // Учёт динамики: только staged-префикс вершин.
            upload_meter.record(self.pending_vertex_count * @sizeOf(Vertex));
        }
        if (self.mesh.index_buffer.id != 0) {
            sg.updateBuffer(self.mesh.index_buffer, sg.asRange(self.indices[0..self.pending_index_count]));
            // Учёт динамики: только staged-префикс индексов (u16).
            upload_meter.record(self.pending_index_count * @sizeOf(u16));
        }
        self.mesh.index_count = @intCast(self.pending_index_count);
        self.mesh.local_bounding_box = BoundingBox.init(self.pending_min_pt, self.pending_max_pt);
        self.mesh.cached_aabb = self.mesh.local_bounding_box;
    }

    pub fn deinit(self: *TrailMesh) void {
        self.nodes.deinit(self.allocator);
        self.allocator.free(self.vertices);
        self.allocator.free(self.indices);
    }
};
