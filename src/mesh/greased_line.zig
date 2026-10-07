//! Agate Greased Line Subsystem.
//!
//! Thick, visually rich 3D lines and polylines with variable widths,
//! camera-facing billboarding, multi-path support, and UV/color gradients.
//! Inspired by Babylon.js GreasedLineMesh and Three.js MeshLine.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;

const Mesh = @import("mesh.zig").Mesh;
const Vertex = @import("types.zig").Vertex;
const GeometryData = @import("types.zig").GeometryData;
const Scene = @import("../scene.zig").Scene;
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const tangents = @import("tangents.zig");

pub const GreasedLineUVMode = enum {
    /// U coordinate ranges from 0.0 at the start to 1.0 at the end of the line.
    relative,
    /// U coordinate matches world-space arc length (useful for tiled textures and dashes).
    unit_length,
};

pub const GreasedLineColorMode = enum {
    /// Uniform single color across the line.
    single,
    /// Per-vertex colors provided in options.colors.
    per_vertex,
    /// Color gradient linearly interpolated between start and end.
    gradient,
};

pub const GreasedLineOptions = struct {
    /// Single polyline path (used if `paths` is empty).
    points: []const Vec3 = &.{},
    /// Multiple polyline paths rendered as a single unified draw call.
    paths: []const []const Vec3 = &.{},
    /// Base line thickness/width in world units.
    width: f32 = 0.1,
    /// Optional per-vertex width array.
    widths: ?[]const f32 = null,
    /// Base uniform color of the line.
    color: Color4 = Color4.white,
    /// Optional end color for gradient color mode.
    color_end: Color4 = Color4.white,
    /// Optional per-vertex colors.
    colors: ?[]const Color4 = null,
    /// Color distribution mode.
    color_mode: GreasedLineColorMode = .single,
    /// Fixed extrusion direction. If null, line automatically billboards towards camera.
    up: ?Vec3 = null,
    /// Static camera position used when building static billboard geometry.
    camera_pos: ?Vec3 = null,
    /// Whether each path is a closed loop (connects last point back to first point).
    closed: bool = false,
    /// UV mapping mode along the line length.
    uv_mode: GreasedLineUVMode = .relative,
    /// UV tiling scale multiplier.
    uv_scale: f32 = 1.0,
    /// Dash ratio (0.0 = solid continuous line, 0.5 = equal dash and gap).
    dash_ratio: f32 = 0.0,
    /// Length of a single dash cycle in UV units.
    dash_length: f32 = 1.0,
    /// Offset of the dash pattern.
    dash_offset: f32 = 0.0,
    /// Limit to prevent excessive sharp corner spikes (miter limit).
    miter_limit: f32 = 3.0,
};

/// Builds static CPU-side GeometryData for a Greased Line from options.
pub fn buildGreasedLineData(allocator: std.mem.Allocator, options: GreasedLineOptions) !GeometryData {
    const multi_paths: []const []const Vec3 = if (options.paths.len > 0)
        options.paths
    else if (options.points.len > 0)
        &[_][]const Vec3{options.points}
    else
        return error.InvalidPoints;

    var total_points: usize = 0;
    var total_segments: usize = 0;
    for (multi_paths) |path| {
        const min_pts: usize = if (options.closed) 3 else 2;
        if (path.len < min_pts) return error.InvalidPoints;
        total_points += path.len;
        total_segments += if (options.closed) path.len else path.len - 1;
    }

    if (options.widths) |w| {
        if (w.len != total_points) return error.MismatchedAttributes;
    }
    if (options.colors) |c| {
        if (c.len != total_points) return error.MismatchedAttributes;
    }

    const total_verts = total_points * 2;
    const total_indices = total_segments * 6;

    const vertices = try allocator.alloc(Vertex, total_verts);
    errdefer allocator.free(vertices);

    const indices = try allocator.alloc(u32, total_indices);
    errdefer allocator.free(indices);

    var vert_offset: usize = 0;
    var idx_offset: usize = 0;
    var global_pt_idx: usize = 0;

    var min_pt = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    var max_pt = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));

    for (multi_paths) |path| {
        const n = path.len;
        const segs = if (options.closed) n else n - 1;

        // 1. Arc length computation along path
        const cumulative = try allocator.alloc(f32, n);
        defer allocator.free(cumulative);
        cumulative[0] = 0.0;
        var total_len: f32 = 0.0;
        for (0..n - 1) |i| {
            total_len += path[i].distance(path[i + 1]);
            cumulative[i + 1] = total_len;
        }
        if (options.closed) {
            total_len += path[n - 1].distance(path[0]);
        }
        if (total_len < 1e-7) total_len = 1.0;

        // 2. Tangent and side vectors per node
        const tangents_arr = try allocator.alloc(Vec3, n);
        defer allocator.free(tangents_arr);
        const sides = try allocator.alloc(Vec3, n);
        defer allocator.free(sides);
        const face_normals = try allocator.alloc(Vec3, n);
        defer allocator.free(face_normals);

        computeGreasedPathFrames(path, options.closed, options.up, options.camera_pos, options.miter_limit, tangents_arr, sides, face_normals);

        // 3. Generate vertices for this path
        const base_vert_idx: u32 = @intCast(vert_offset);

        for (0..n) |i| {
            const pt = path[i];
            const pt_global = global_pt_idx;
            global_pt_idx += 1;

            const w = if (options.widths) |w_arr| w_arr[pt_global] else options.width;
            const half_w = @max(w, 0.0) * 0.5;

            // U coordinate
            const u_coord: f32 = switch (options.uv_mode) {
                .relative => (cumulative[i] / total_len) * options.uv_scale,
                .unit_length => cumulative[i] * options.uv_scale,
            };

            // Color computation
            const col: Color4 = switch (options.color_mode) {
                .single => options.color,
                .per_vertex => if (options.colors) |c_arr| c_arr[pt_global] else options.color,
                .gradient => Color4.lerp(options.color, options.color_end, cumulative[i] / total_len),
            };

            const side = sides[i];
            const left_pos = pt.add(side.scale(half_w));
            const right_pos = pt.sub(side.scale(half_w));
            const norm = face_normals[i].toArray();
            const tan = [4]f32{ tangents_arr[i].x, tangents_arr[i].y, tangents_arr[i].z, options.dash_ratio };

            // Left ribbon vertex
            vertices[vert_offset + 2 * i] = .{
                .position = left_pos.toArray(),
                .normal = norm,
                .color = col.toArray(),
                .uv = .{ u_coord, 1.0 },
                .tangent = tan,
            };

            // Right ribbon vertex
            vertices[vert_offset + 2 * i + 1] = .{
                .position = right_pos.toArray(),
                .normal = norm,
                .color = col.toArray(),
                .uv = .{ u_coord, 0.0 },
                .tangent = tan,
            };

            min_pt = Vec3.new(
                @min(min_pt.x, @min(left_pos.x, right_pos.x)),
                @min(min_pt.y, @min(left_pos.y, right_pos.y)),
                @min(min_pt.z, @min(left_pos.z, right_pos.z)),
            );
            max_pt = Vec3.new(
                @max(max_pt.x, @max(left_pos.x, right_pos.x)),
                @max(max_pt.y, @max(left_pos.y, right_pos.y)),
                @max(max_pt.z, @max(left_pos.z, right_pos.z)),
            );
        }

        // 4. Generate triangle indices for this path
        for (0..segs) |i| {
            const next_i = if (options.closed) (i + 1) % n else i + 1;
            const a: u32 = base_vert_idx + @as(u32, @intCast(2 * i));
            const b: u32 = base_vert_idx + @as(u32, @intCast(2 * i + 1));
            const c: u32 = base_vert_idx + @as(u32, @intCast(2 * next_i));
            const d: u32 = base_vert_idx + @as(u32, @intCast(2 * next_i + 1));

            indices[idx_offset + 0] = a;
            indices[idx_offset + 1] = b;
            indices[idx_offset + 2] = c;
            indices[idx_offset + 3] = b;
            indices[idx_offset + 4] = d;
            indices[idx_offset + 5] = c;
            idx_offset += 6;
        }

        vert_offset += 2 * n;
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(min_pt, max_pt),
    };
}

/// Computes tangents, side normals, and face normals with miter joint smoothing.
fn computeGreasedPathFrames(
    path: []const Vec3,
    closed: bool,
    up_opt: ?Vec3,
    cam_opt: ?Vec3,
    miter_limit: f32,
    out_tangents: []Vec3,
    out_sides: []Vec3,
    out_normals: []Vec3,
) void {
    const n = path.len;
    if (n == 0) return;

    // Pass 1: compute node tangents
    for (0..n) |i| {
        var t: Vec3 = undefined;
        if (closed) {
            const prev = if (i == 0) n - 1 else i - 1;
            const next = (i + 1) % n;
            const d_in = path[i].sub(path[prev]).normalize();
            const d_out = path[next].sub(path[i]).normalize();
            t = d_in.add(d_out);
        } else {
            if (i == 0) {
                t = path[1].sub(path[0]);
            } else if (i == n - 1) {
                t = path[n - 1].sub(path[n - 2]);
            } else {
                const d_in = path[i].sub(path[i - 1]).normalize();
                const d_out = path[i + 1].sub(path[i]).normalize();
                t = d_in.add(d_out);
            }
        }

        const len_sq = t.lengthSq();
        if (len_sq > 1e-8) {
            out_tangents[i] = t.scale(1.0 / @sqrt(len_sq));
        } else if (i + 1 < n) {
            out_tangents[i] = path[i + 1].sub(path[i]).normalize();
        } else {
            out_tangents[i] = Vec3.new(0.0, 0.0, 1.0);
        }
    }

    // Pass 2: compute side vectors and apply miter scaling
    for (0..n) |i| {
        const t = out_tangents[i];
        const pt = path[i];

        var side: Vec3 = undefined;

        if (cam_opt) |cam_pos| {
            // Camera-facing billboarding
            var to_cam = cam_pos.sub(pt);
            const cam_len_sq = to_cam.lengthSq();
            if (cam_len_sq > 1e-8) {
                to_cam = to_cam.scale(1.0 / @sqrt(cam_len_sq));
            } else {
                to_cam = Vec3.new(0.0, 0.0, 1.0);
            }

            side = t.cross(to_cam);
            const side_sq = side.lengthSq();
            if (side_sq > 1e-8) {
                side = side.scale(1.0 / @sqrt(side_sq));
            } else {
                side = t.cross(up_opt orelse Vec3.up);
                const s2 = side.lengthSq();
                if (s2 > 1e-8) {
                    side = side.scale(1.0 / @sqrt(s2));
                } else {
                    side = tangents.pickOrthogonal(t);
                }
            }
        } else if (up_opt) |up_vec| {
            // Fixed extrusion direction
            side = t.cross(up_vec);
            const side_sq = side.lengthSq();
            if (side_sq > 1e-8) {
                side = side.scale(1.0 / @sqrt(side_sq));
            } else {
                side = tangents.pickOrthogonal(t);
            }
        } else {
            // Default 3D frame
            side = t.cross(Vec3.up);
            const side_sq = side.lengthSq();
            if (side_sq > 1e-8) {
                side = side.scale(1.0 / @sqrt(side_sq));
            } else {
                side = tangents.pickOrthogonal(t);
            }
        }

        // Miter scale at intermediate corners to maintain constant visual line width
        var miter_scale: f32 = 1.0;
        if ((closed or (i > 0 and i < n - 1))) {
            const prev = if (i == 0) n - 1 else i - 1;
            const d_in = path[i].sub(path[prev]).normalize();
            const dot = std.math.clamp(d_in.dot(t), 0.15, 1.0);
            miter_scale = std.math.clamp(1.0 / dot, 1.0, miter_limit);
        }

        out_sides[i] = side.scale(miter_scale);
        out_normals[i] = out_sides[i].cross(t).normalize();
    }
}

/// Dynamic GreasedLine mesh that can update its control points and widths,
/// and re-orient towards the camera in real-time.
pub const GreasedLineMesh = struct {
    allocator: std.mem.Allocator,
    scene: *Scene,
    mesh: *Mesh,
    options: GreasedLineOptions,
    points_buffer: std.ArrayListUnmanaged(Vec3) = .empty,
    widths_buffer: std.ArrayListUnmanaged(f32) = .empty,
    vertices: []Vertex,
    indices: []u32,
    gpu_dirty: bool = false,
    /// Vertex data has never reached the GPU (or the buffers were just
    /// created): the next flush uploads both vertex and index buffers.
    /// Cleared after the first full upload; later rebuilds update vertices
    /// only (indices depend on topology, which does not change).
    gpu_needs_full_upload: bool = false,

    pub fn init(scene: *Scene, name: []const u8, options: GreasedLineOptions) !*GreasedLineMesh {
        const allocator = scene.allocator;
        var initial_data = try buildGreasedLineData(allocator, options);
        defer initial_data.deinit(allocator);

        const total_verts = initial_data.vertices.len;
        const total_indices = initial_data.indices.len;

        const vertices = try allocator.dupe(Vertex, initial_data.vertices);
        errdefer allocator.free(vertices);

        const indices = try allocator.dupe(u32, initial_data.indices);
        errdefer allocator.free(indices);

        const deferred = !gpu_thread.isOnContextThread() or !sg.isvalid();
        // Immediate creation makes empty dynamic buffers; the actual data
        // upload happens in flushGpuUploads on the context thread. Uploading
        // here would collide with the first frame's flush — it runs before
        // the first sg.commit and therefore lands in the same sokol frame,
        // and sokol allows only one update per buffer per frame
        // (VALIDATE_UPDATEBUF_ONCE). `.data` at creation is not an option:
        // this sokol rejects desc.data for .write_* (dynamic_update) buffers.
        const vb = if (deferred) sg.Buffer{} else sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .dynamic_update = true },
            .size = total_verts * @sizeOf(Vertex),
        });
        const ib = if (deferred) sg.Buffer{} else sg.makeBuffer(.{
            .usage = .{ .index_buffer = true, .dynamic_update = true },
            .size = total_indices * @sizeOf(u32),
        });

        const mesh = try allocator.create(Mesh);
        errdefer allocator.destroy(mesh);

        mesh.* = .{
            .name = name,
            .vertex_buffer = vb,
            .index_buffer = ib,
            .vertex_count = @intCast(total_verts),
            .index_count = @intCast(total_indices),
            .index_type = .UINT32,
            .local_bounding_box = initial_data.bounds,
            .cast_shadows = false,
        };
        try mesh.retainCpuGeometryU32(allocator, vertices, indices);

        const greased = try allocator.create(GreasedLineMesh);
        greased.* = .{
            .allocator = allocator,
            .scene = scene,
            .mesh = mesh,
            .options = options,
            .vertices = vertices,
            .indices = indices,
            // Buffers are empty until the first flush (immediate creation has
            // no data yet; deferred creation has no buffers at all).
            .gpu_dirty = true,
            .gpu_needs_full_upload = true,
        };

        if (options.points.len > 0) {
            try greased.points_buffer.appendSlice(allocator, options.points);
        }

        try scene.meshes.append(allocator, mesh);
        return greased;
    }

    pub fn deinit(self: *GreasedLineMesh) void {
        self.points_buffer.deinit(self.allocator);
        self.widths_buffer.deinit(self.allocator);
        self.allocator.free(self.vertices);
        self.allocator.free(self.indices);
        self.allocator.destroy(self);
    }

    /// Dynamically updates the control points of the line.
    pub fn setPoints(self: *GreasedLineMesh, new_points: []const Vec3) !void {
        if (new_points.len < 2) return;
        self.points_buffer.clearRetainingCapacity();
        try self.points_buffer.appendSlice(self.allocator, new_points);
        self.options.points = self.points_buffer.items;
        self.options.paths = &.{};
        self.rebuildGeometry(self.options.camera_pos orelse Vec3.zero);
    }

    /// Dynamically updates the global thickness of the line.
    pub fn setWidth(self: *GreasedLineMesh, width: f32) void {
        self.options.width = @max(width, 0.0);
        self.rebuildGeometry(self.options.camera_pos orelse Vec3.zero);
    }

    /// Dynamically updates the line color.
    pub fn setColor(self: *GreasedLineMesh, color: Color4) void {
        self.options.color = color;
        self.rebuildGeometry(self.options.camera_pos orelse Vec3.zero);
    }

    /// Updates the camera-facing orientation for billboarding.
    pub fn update(self: *GreasedLineMesh, camera_pos: Vec3) void {
        self.options.camera_pos = camera_pos;
        self.rebuildGeometry(camera_pos);
    }

    fn rebuildGeometry(self: *GreasedLineMesh, camera_pos: Vec3) void {
        var opts = self.options;
        opts.camera_pos = camera_pos;

        const data = buildGreasedLineData(self.allocator, opts) catch return;
        defer {
            self.allocator.free(data.vertices);
            self.allocator.free(data.indices);
        }

        const copy_verts = @min(self.vertices.len, data.vertices.len);
        @memcpy(self.vertices[0..copy_verts], data.vertices[0..copy_verts]);

        const copy_idx = @min(self.indices.len, data.indices.len);
        @memcpy(self.indices[0..copy_idx], data.indices[0..copy_idx]);

        self.mesh.vertex_count = @intCast(copy_verts);
        self.mesh.index_count = @intCast(copy_idx);
        self.mesh.local_bounding_box = data.bounds;

        self.gpu_dirty = true;
    }

    /// Uploads dirty CPU geometry to the GPU buffers. Runs on the context
    /// thread (Scene.flushPendingGpuUploads). Never uploads the same buffer
    /// twice in one frame: the first call after creation does the full
    /// upload (vertex + index), later rebuilds update vertices only.
    pub fn flushGpuUploads(self: *GreasedLineMesh) void {
        if (!self.gpu_dirty) return;
        if (!sg.isvalid()) return; // keep dirty: retry once a context exists
        if (self.mesh.vertex_buffer.id == 0) {
            if (self.vertices.len == 0) return;
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = self.vertices.len * @sizeOf(Vertex),
            });
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true, .dynamic_update = true },
                .size = self.indices.len * @sizeOf(u32),
            });
            if (vb.id == 0 or ib.id == 0) {
                // Partial creation (pool exhaustion): destroy what was made
                // and retry next frame instead of leaking handles.
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
                return;
            }
            self.mesh.vertex_buffer = vb;
            self.mesh.index_buffer = ib;
            self.gpu_needs_full_upload = true;
        }
        if (self.vertices.len > 0) {
            sg.updateBuffer(self.mesh.vertex_buffer, sg.asRange(self.vertices));
            // Учёт динамики: весь вершинный массив линии.
            upload_meter.record(self.vertices.len * @sizeOf(Vertex));
        }
        if (self.gpu_needs_full_upload and self.indices.len > 0) {
            sg.updateBuffer(self.mesh.index_buffer, sg.asRange(self.indices));
            // Учёт динамики: индексный массив (полная заливка при создании).
            upload_meter.record(self.indices.len * @sizeOf(u32));
            self.gpu_needs_full_upload = false;
        }
        self.gpu_dirty = false;
    }
};
