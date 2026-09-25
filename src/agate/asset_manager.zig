//! AssetManager: batch asset loading pipeline with progress tracking and caching.
//!
//! Inspired by Babylon.js AssetsManager:
//! - Task types:
//!   - TextFileTask: loads UTF-8 text files (JSON, configs, shaders)
//!   - BinaryFileTask: loads raw binary data (buffers, audio data, packets)
//!   - TextureTask: loads 2D textures (integrated with Texture decode / GPU upload)
//!   - MeshTask: loads 3D models (.glb, .gltf, .obj, .stl, .ply) into a Scene
//!     (scene != null, via SceneLoader/appendToScene). Scene-less mesh tasks
//!     retain owned CPU `GeometryData` in `task.mesh_result` (OBJ/STL/PLY;
//!     GLB/GLTF without a scene is `error.MeshRequiresScene`).
//!   - CustomTask: user-defined arbitrary loading routine
//! - AssetCache:
//!   - In-memory cache by path/URL to prevent duplicate disk reads and duplicate allocations
//!   - Query, store, invalidate, or bulk-clear cached assets
//! - Progress & Diagnostics:
//!   - Total, completed, failed, and remaining task counts
//!   - Normalized progress in [0.0 .. 1.0]
//!   - Task states: .pending, .running, .completed, .failed
//! - Execution Models: synchronous `loadSync` / stepped `loadStep` (task-count
//!   budget only — NOT a wall-time frame budget; each task runs I/O +
//!   parse/decode/upload inline). Context-thread rule: texture/mesh GPU
//!   uploads require the sg-context thread (`error.TextureRequiresContextThread`
//!   otherwise). A manager has one caller at a time: it is not internally
//!   synchronized. Text/binary work can run on a worker; custom tasks must
//!   obey their own thread contract. True async is the
//!   existing `assets.UploadQueue` (worker decode, context drain) — note its
//!   `.async_textures` option offloads texture decode, not mesh geometry;
//!   GPU uploads still run on the context thread.
//!   No second worker pipeline lives here (see `AssetManager` docs).
//! - Event Callbacks:
//!   - `onTaskSuccess`, `onTaskError`, `onProgress`, `onFinish`
//!
//! Ownership + lifetime:
//! - `AssetTask.name`/`path` are BORROWED: the caller must keep them alive
//!   until `reset`/`resetAll`/`deinit`. The manager never dupes or frees them.
//! - `text_result`/`binary_result` are OWNED by the task (`owned_*`) when
//!   `use_cache == false`, else BORROWED from the cache (freed by
//!   `cache.clear`/`deinit`, never by `reset`).
//! - `texture_result` is OWNED by the task (`owned_texture`) only when
//!   `use_cache == false`; cached hits AND newly-cached uploads are owned by
//!   the cache (task holds a borrowed alias — `reset` must NOT deinit it).
//! - `mesh_result` (scene-less mesh tasks) is OWNED CPU `GeometryData`
//!   (freed by `reset`, or moved out via `takeMeshGeometry` and freed with
//!   the manager's allocator). Reset/cache teardown of GPU textures is
//!   context-thread only, after prepared frames stop borrowing their handles.

const std = @import("std");

const Scene = @import("scene.zig").Scene;
const Mesh = @import("mesh.zig").Mesh;
const GeometryData = @import("mesh.zig").GeometryData;
const Vertex = @import("mesh.zig").Vertex;
const computeTangents = @import("mesh.zig").computeTangents;
const Texture = @import("texture.zig").Texture;
const SceneLoader = @import("loader/scene_loader.zig").SceneLoader;
const obj_loader = @import("loader/obj.zig");
const stl_loader = @import("loader/stl.zig");
const ply_loader = @import("loader/ply.zig");
const gpu_thread = @import("gpu_thread.zig");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const BoundingBox = math.BoundingBox;

pub const TaskState = enum(u8) {
    pending,
    running,
    completed,
    failed,
};

pub const TaskType = enum(u8) {
    text,
    binary,
    texture,
    mesh,
    custom,
};

pub const AssetTask = struct {
    /// BORROWED: caller keeps alive until manager reset/deinit.
    name: []const u8,
    /// BORROWED: caller keeps alive until manager reset/deinit.
    path: []const u8 = "",
    task_type: TaskType,
    state: TaskState = .pending,
    error_result: ?anyerror = null,
    use_cache: bool = true,

    // Results
    /// OWNED iff owned_text (use_cache == false); else borrowed from cache.
    text_result: ?[]const u8 = null,
    /// OWNED iff owned_binary (use_cache == false); else borrowed from cache.
    binary_result: ?[]const u8 = null,
    /// OWNED iff owned_texture (use_cache == false AND upload succeeded);
    /// cached hits and newly-cached uploads are borrowed from the cache.
    texture_result: ?Texture = null,
    /// Meshes spawned into scene (scene != null), or 1 with owned CPU
    /// geometry in `mesh_result` (scene-less OBJ/STL/PLY). Zero until
    /// success; never 1 on failure.
    mesh_count: usize = 0,
    /// Validated CPU vertex/index totals for mesh tasks (scene loads sum the
    /// spawned meshes; scene-less loads count `mesh_result`).
    mesh_vertex_count: usize = 0,
    mesh_index_count: usize = 0,
    /// Scene-less mesh result: OWNED CPU geometry (`owned_mesh`), freed by
    /// `reset` or moved out via `takeMeshGeometry`. Null for scene loads
    /// (meshes live in the Scene) and until success.
    mesh_result: ?GeometryData = null,

    // Task-specific context
    scene: ?*Scene = null,
    texture_options: Texture.Options = .{},
    custom_run: ?*const fn (*AssetTask, std.mem.Allocator) anyerror!void = null,
    user_ctx: ?*anyopaque = null,

    // Internal ownership tracking
    owned_text: bool = false,
    owned_binary: bool = false,
    owned_texture: bool = false,
    owned_mesh: bool = false,
    from_cache: bool = false,

    /// Moves the scene-less mesh geometry out; the task no longer owns it
    /// (subsequent `reset` won't free it). Returns null when there is none.
    pub fn takeMeshGeometry(self: *AssetTask) ?GeometryData {
        if (!self.owned_mesh) return null;
        const g = self.mesh_result orelse return null;
        self.mesh_result = null;
        self.owned_mesh = false;
        self.mesh_count = 0;
        self.mesh_vertex_count = 0;
        self.mesh_index_count = 0;
        return g;
    }

    pub fn isSuccess(self: *const AssetTask) bool {
        return self.state == .completed;
    }

    pub fn isFailed(self: *const AssetTask) bool {
        return self.state == .failed;
    }
};

/// In-memory asset cache indexing loaded resources by file path or identifier.
pub const AssetCache = struct {
    allocator: std.mem.Allocator,
    text_map: std.StringHashMapUnmanaged([]const u8) = .empty,
    binary_map: std.StringHashMapUnmanaged([]const u8) = .empty,
    texture_map: std.StringHashMapUnmanaged(Texture) = .empty,

    pub fn init(allocator: std.mem.Allocator) AssetCache {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AssetCache) void {
        self.clear();
    }

    pub fn clear(self: *AssetCache) void {
        var text_it = self.text_map.iterator();
        while (text_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.text_map.deinit(self.allocator);
        self.text_map = .empty;

        var bin_it = self.binary_map.iterator();
        while (bin_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.binary_map.deinit(self.allocator);
        self.binary_map = .empty;

        var tex_it = self.texture_map.iterator();
        while (tex_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.texture_map.deinit(self.allocator);
        self.texture_map = .empty;
    }

    pub fn hasText(self: *const AssetCache, key: []const u8) bool {
        return self.text_map.contains(key);
    }

    pub fn getText(self: *const AssetCache, key: []const u8) ?[]const u8 {
        return self.text_map.get(key);
    }

    pub fn putText(self: *AssetCache, key: []const u8, text: []const u8) ![]const u8 {
        const key_dupe = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_dupe);
        const text_dupe = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(text_dupe);
        const gop = try self.text_map.getOrPut(self.allocator, key);
        if (gop.found_existing) {
            // Overwrite: keep the original key, free the dupe + old value.
            self.allocator.free(key_dupe);
            self.allocator.free(gop.value_ptr.*);
            gop.value_ptr.* = text_dupe;
            return gop.value_ptr.*;
        }
        gop.key_ptr.* = key_dupe;
        gop.value_ptr.* = text_dupe;
        return text_dupe;
    }

    pub fn hasBinary(self: *const AssetCache, key: []const u8) bool {
        return self.binary_map.contains(key);
    }

    pub fn getBinary(self: *const AssetCache, key: []const u8) ?[]const u8 {
        return self.binary_map.get(key);
    }

    pub fn putBinary(self: *AssetCache, key: []const u8, data: []const u8) ![]const u8 {
        const key_dupe = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_dupe);
        const data_dupe = try self.allocator.dupe(u8, data);
        errdefer self.allocator.free(data_dupe);
        const gop = try self.binary_map.getOrPut(self.allocator, key);
        if (gop.found_existing) {
            self.allocator.free(key_dupe);
            self.allocator.free(gop.value_ptr.*);
            gop.value_ptr.* = data_dupe;
            return gop.value_ptr.*;
        }
        gop.key_ptr.* = key_dupe;
        gop.value_ptr.* = data_dupe;
        return data_dupe;
    }

    pub fn hasTexture(self: *const AssetCache, key: []const u8) bool {
        return self.texture_map.contains(key);
    }

    pub fn getTexture(self: *const AssetCache, key: []const u8) ?Texture {
        return self.texture_map.get(key);
    }

    pub fn putTexture(self: *AssetCache, key: []const u8, texture: Texture) !void {
        const key_dupe = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_dupe);
        const gop = try self.texture_map.getOrPut(self.allocator, key);
        if (gop.found_existing) {
            // Self-put of the identical GPU handles (e.g. re-caching the
            // borrowed alias) is a no-op: deinit would destroy the live
            // texture out from under the inserted alias.
            if (textureHandlesEqual(gop.value_ptr.*, texture)) {
                self.allocator.free(key_dupe);
                return;
            }
            // Overwrite: keep the original key, destroy the old GPU texture.
            // Transfer semantics: `texture` ownership moves to the cache;
            // never pass a borrowed cache alias with different handles.
            self.allocator.free(key_dupe);
            gop.value_ptr.deinit();
            gop.value_ptr.* = texture;
            return;
        }
        gop.key_ptr.* = key_dupe;
        gop.value_ptr.* = texture;
    }

    pub fn count(self: *const AssetCache) usize {
        return self.text_map.count() + self.binary_map.count() + self.texture_map.count();
    }
};

/// Pure GPU-handle identity for textures (no sg calls): image/view/sampler
/// ids plus dimensions. Used to make `putTexture` self-put safe.
pub fn textureHandlesEqual(a: Texture, b: Texture) bool {
    return a.image.id == b.image.id and
        a.view.id == b.view.id and
        a.sampler.id == b.sampler.id and
        a.width == b.width and
        a.height == b.height;
}

fn readFileBytesAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const len = try file.length(io);
    const n = std.math.cast(usize, len) orelse return error.FileTooLarge;
    const buf = try allocator.alloc(u8, n);
    errdefer allocator.free(buf);
    const read = try file.readPositionalAll(io, buf, 0);
    if (read < buf.len) return error.UnexpectedEof;
    return buf;
}

/// Sums vertex/index totals over meshes spawned into a scene.
fn sumSpawned(spawned: []*Mesh) struct { verts: usize, idx: usize } {
    var v: usize = 0;
    var idx: usize = 0;
    for (spawned) |m| {
        v += m.vertex_count;
        idx += m.index_count;
    }
    return .{ .verts = v, .idx = idx };
}

/// Builds owned CPU `GeometryData` from parsed loader output (positions /
/// normals / uvs / indices, white color; PLY passes its own colors). Bounds
/// + tangents mirror the `appendToScene` upload path; no GPU calls.
fn buildCpuGeometry(
    allocator: std.mem.Allocator,
    positions: []const f32,
    normals: []const f32,
    uvs: []const f32,
    colors: ?[]const f32,
    indices: []const u32,
) !GeometryData {
    const n = positions.len / 3;
    const vertices = try allocator.alloc(Vertex, n);
    errdefer allocator.free(vertices);
    for (0..n) |i| {
        vertices[i] = .{
            .position = .{ positions[3 * i], positions[3 * i + 1], positions[3 * i + 2] },
            .normal = .{ normals[3 * i], normals[3 * i + 1], normals[3 * i + 2] },
            .color = if (colors) |c| .{ c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3] } else .{ 1, 1, 1, 1 },
            .uv = .{ uvs[2 * i], uvs[2 * i + 1] },
        };
    }
    var min_p = math.Vec3.new(vertices[0].position[0], vertices[0].position[1], vertices[0].position[2]);
    var max_p = min_p;
    for (vertices) |vert| {
        min_p.x = @min(min_p.x, vert.position[0]);
        min_p.y = @min(min_p.y, vert.position[1]);
        min_p.z = @min(min_p.z, vert.position[2]);
        max_p.x = @max(max_p.x, vert.position[0]);
        max_p.y = @max(max_p.y, vert.position[1]);
        max_p.z = @max(max_p.z, vert.position[2]);
    }
    const owned_idx = try allocator.dupe(u32, indices);
    errdefer allocator.free(owned_idx);
    computeTangents(vertices, owned_idx, null);
    return .{
        .vertices = vertices,
        .indices = owned_idx,
        .bounds = BoundingBox.init(min_p, max_p),
    };
}

pub const TaskSuccessFn = *const fn (manager: *AssetManager, task: *AssetTask) void;
pub const TaskErrorFn = *const fn (manager: *AssetManager, task: *AssetTask, err: anyerror) void;
pub const ProgressFn = *const fn (manager: *AssetManager, remaining: usize, total: usize, task: *AssetTask) void;
pub const FinishFn = *const fn (manager: *AssetManager) void;

/// Central asset loading manager: synchronous batch loading with progress,
/// caching, and stable task pointers. See module docs for the execution /
/// context-thread contract (sync only; `UploadQueue` is the async route).
pub const AssetManager = struct {
    allocator: std.mem.Allocator,
    /// Individually allocated tasks: returned `*AssetTask` pointers stay
    /// stable across later `add*` calls (the pointer array may reallocate,
    /// the task allocations do not move). Callbacks may safely enqueue new
    /// tasks and keep using the current task pointer. Callbacks must NOT
    /// call `reset`/`resetAll`/`deinit` (that would free the running task).
    tasks: std.ArrayListUnmanaged(*AssetTask) = .empty,
    cache: AssetCache,

    completed_count: usize = 0,
    failed_count: usize = 0,
    current_index: usize = 0,

    // Callbacks
    onTaskSuccess: ?TaskSuccessFn = null,
    onTaskError: ?TaskErrorFn = null,
    onProgress: ?ProgressFn = null,
    onFinish: ?FinishFn = null,
    user_data: ?*anyopaque = null,

    pub fn init(allocator: std.mem.Allocator) AssetManager {
        return .{
            .allocator = allocator,
            .cache = AssetCache.init(allocator),
        };
    }

    pub fn deinit(self: *AssetManager) void {
        self.reset();
        self.cache.deinit();
    }

    /// Clears task queue and frees task-owned data, keeping the cache intact.
    /// Borrowed cache aliases (text/binary/texture) are NOT freed/deinited.
    pub fn reset(self: *AssetManager) void {
        for (self.tasks.items) |task| {
            if (task.owned_text and task.text_result != null) {
                self.allocator.free(task.text_result.?);
            }
            if (task.owned_binary and task.binary_result != null) {
                self.allocator.free(task.binary_result.?);
            }
            if (task.owned_texture and task.texture_result != null) {
                task.texture_result.?.deinit();
            }
            if (task.owned_mesh and task.mesh_result != null) {
                task.mesh_result.?.deinit(self.allocator);
            }
            self.allocator.destroy(task);
        }
        self.tasks.deinit(self.allocator);
        self.tasks = .empty;
        self.completed_count = 0;
        self.failed_count = 0;
        self.current_index = 0;
    }

    /// Resets both the task list and the underlying asset cache.
    pub fn resetAll(self: *AssetManager) void {
        self.reset();
        self.cache.clear();
    }

    pub fn addTextFileTask(self: *AssetManager, name: []const u8, path: []const u8) !*AssetTask {
        const task = try self.allocator.create(AssetTask);
        errdefer self.allocator.destroy(task);
        task.* = .{
            .name = name,
            .path = path,
            .task_type = .text,
        };
        try self.tasks.append(self.allocator, task);
        return task;
    }

    pub fn addBinaryFileTask(self: *AssetManager, name: []const u8, path: []const u8) !*AssetTask {
        const task = try self.allocator.create(AssetTask);
        errdefer self.allocator.destroy(task);
        task.* = .{
            .name = name,
            .path = path,
            .task_type = .binary,
        };
        try self.tasks.append(self.allocator, task);
        return task;
    }

    pub fn addTextureTask(self: *AssetManager, name: []const u8, path: []const u8, options: Texture.Options) !*AssetTask {
        const task = try self.allocator.create(AssetTask);
        errdefer self.allocator.destroy(task);
        task.* = .{
            .name = name,
            .path = path,
            .task_type = .texture,
            .texture_options = options,
        };
        try self.tasks.append(self.allocator, task);
        return task;
    }

    pub fn addMeshTask(self: *AssetManager, name: []const u8, path: []const u8, scene: ?*Scene) !*AssetTask {
        const task = try self.allocator.create(AssetTask);
        errdefer self.allocator.destroy(task);
        task.* = .{
            .name = name,
            .path = path,
            .task_type = .mesh,
            .scene = scene,
        };
        try self.tasks.append(self.allocator, task);
        return task;
    }

    pub fn addCustomTask(
        self: *AssetManager,
        name: []const u8,
        run_fn: *const fn (*AssetTask, std.mem.Allocator) anyerror!void,
        user_ctx: ?*anyopaque,
    ) !*AssetTask {
        const task = try self.allocator.create(AssetTask);
        errdefer self.allocator.destroy(task);
        task.* = .{
            .name = name,
            .task_type = .custom,
            .custom_run = run_fn,
            .user_ctx = user_ctx,
        };
        try self.tasks.append(self.allocator, task);
        return task;
    }

    pub fn getTaskByName(self: *AssetManager, name: []const u8) ?*AssetTask {
        for (self.tasks.items) |t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    pub fn remainingCount(self: *const AssetManager) usize {
        const processed = self.completed_count + self.failed_count;
        if (processed >= self.tasks.items.len) return 0;
        return self.tasks.items.len - processed;
    }

    pub fn totalCount(self: *const AssetManager) usize {
        return self.tasks.items.len;
    }

    pub fn progress(self: *const AssetManager) f32 {
        if (self.tasks.items.len == 0) return 1.0;
        const done: f32 = @floatFromInt(self.completed_count + self.failed_count);
        const total: f32 = @floatFromInt(self.tasks.items.len);
        return done / total;
    }

    pub fn isDone(self: *const AssetManager) bool {
        return self.remainingCount() == 0;
    }

    pub fn hasErrors(self: *const AssetManager) bool {
        return self.failed_count > 0;
    }

    fn executeTask(self: *AssetManager, task: *AssetTask) anyerror!void {
        switch (task.task_type) {
            .text => {
                if (task.use_cache and self.cache.hasText(task.path)) {
                    task.text_result = self.cache.getText(task.path);
                    task.from_cache = true;
                    return;
                }
                const bytes = try readFileBytesAlloc(self.allocator, task.path);
                errdefer self.allocator.free(bytes);
                if (task.use_cache) {
                    task.text_result = try self.cache.putText(task.path, bytes);
                    self.allocator.free(bytes);
                    task.from_cache = false;
                } else {
                    task.text_result = bytes;
                    task.owned_text = true;
                }
            },
            .binary => {
                if (task.use_cache and self.cache.hasBinary(task.path)) {
                    task.binary_result = self.cache.getBinary(task.path);
                    task.from_cache = true;
                    return;
                }
                const bytes = try readFileBytesAlloc(self.allocator, task.path);
                errdefer self.allocator.free(bytes);
                if (task.use_cache) {
                    task.binary_result = try self.cache.putBinary(task.path, bytes);
                    self.allocator.free(bytes);
                    task.from_cache = false;
                } else {
                    task.binary_result = bytes;
                    task.owned_binary = true;
                }
            },
            .texture => {
                if (task.use_cache and self.cache.hasTexture(task.path)) {
                    task.texture_result = self.cache.getTexture(task.path);
                    task.from_cache = true;
                    task.owned_texture = false;
                    return;
                }
                if (sg.isvalid()) {
                    // Live GPU upload: context thread only. Worker threads
                    // must use UploadQueue (decode off-thread, drain here).
                    if (!gpu_thread.isOnContextThread()) return error.TextureRequiresContextThread;
                    var tex = try Texture.fromFile(self.allocator, task.path, task.texture_options);
                    errdefer tex.deinit();
                    if (task.use_cache) {
                        // Cache takes GPU ownership; the task keeps a borrowed alias.
                        try self.cache.putTexture(task.path, tex);
                        task.texture_result = tex;
                        task.owned_texture = false;
                        task.from_cache = false;
                    } else {
                        task.texture_result = tex;
                        task.owned_texture = true;
                    }
                } else {
                    // Headless CPU decode check (no GPU upload possible):
                    // validates the file decodes, leaves texture_result null.
                    var raw = try Texture.decodeImageFile(self.allocator, task.path, .{});
                    raw.deinit(self.allocator);
                    task.texture_result = null;
                    task.owned_texture = false;
                }
            },
            .mesh => {
                const is_glb = std.mem.endsWith(u8, task.path, ".glb") or std.mem.endsWith(u8, task.path, ".gltf");
                const is_obj = std.mem.endsWith(u8, task.path, ".obj");
                const is_stl = std.mem.endsWith(u8, task.path, ".stl");
                const is_ply = std.mem.endsWith(u8, task.path, ".ply");
                if (task.scene) |sc| {
                    if (is_glb) {
                        const spawned = try SceneLoader.appendGlb(sc, task.path);
                        defer sc.allocator.free(spawned);
                        const totals = sumSpawned(spawned);
                        task.mesh_count = spawned.len;
                        task.mesh_vertex_count = totals.verts;
                        task.mesh_index_count = totals.idx;
                    } else if (is_obj or is_stl or is_ply) {
                        const bytes = try readFileBytesAlloc(self.allocator, task.path);
                        defer self.allocator.free(bytes);
                        const spawned = if (is_obj)
                            try obj_loader.appendToScene(sc, self.allocator, task.name, bytes)
                        else if (is_stl)
                            try stl_loader.appendToScene(sc, self.allocator, task.name, bytes)
                        else
                            try ply_loader.appendToScene(sc, self.allocator, task.name, bytes);
                        defer self.allocator.free(spawned);
                        const totals = sumSpawned(spawned);
                        task.mesh_count = spawned.len;
                        task.mesh_vertex_count = totals.verts;
                        task.mesh_index_count = totals.idx;
                    } else {
                        return error.UnsupportedMeshFormat;
                    }
                } else {
                    // Scene-less: retain owned CPU geometry (OBJ/STL/PLY).
                    // GLB/GLTF has no CPU-only path — explicit error.
                    if (is_glb) return error.MeshRequiresScene;
                    const bytes = try readFileBytesAlloc(self.allocator, task.path);
                    defer self.allocator.free(bytes);
                    var geom: GeometryData = undefined;
                    if (is_obj) {
                        var data = try obj_loader.parse(self.allocator, bytes);
                        defer data.deinit(self.allocator);
                        geom = try buildCpuGeometry(self.allocator, data.positions, data.normals, data.uvs, null, data.indices);
                    } else if (is_stl) {
                        var data = try stl_loader.parse(self.allocator, bytes);
                        defer data.deinit(self.allocator);
                        const n = data.vertex_count;
                        const uvs = try self.allocator.alloc(f32, n * 2);
                        defer self.allocator.free(uvs);
                        @memset(uvs, 0);
                        geom = try buildCpuGeometry(self.allocator, data.positions, data.normals, uvs, null, data.indices);
                    } else if (is_ply) {
                        var data = try ply_loader.parse(self.allocator, bytes);
                        defer data.deinit(self.allocator);
                        geom = try buildCpuGeometry(
                            self.allocator,
                            data.positions,
                            data.normals,
                            data.uvs,
                            if (data.has_colors) data.colors else null,
                            data.indices,
                        );
                    } else {
                        return error.UnsupportedMeshFormat;
                    }
                    errdefer geom.deinit(self.allocator);
                    task.mesh_result = geom;
                    task.owned_mesh = true;
                    task.mesh_count = 1;
                    task.mesh_vertex_count = geom.vertices.len;
                    task.mesh_index_count = geom.indices.len;
                }
            },
            .custom => {
                if (task.custom_run) |run| {
                    try run(task, self.allocator);
                }
            },
        }
    }

    /// Runs up to `max_tasks` pending tasks (sync; see module docs).
    /// Callbacks may enqueue via `add*` (pointers stable); they must not
    /// call `reset`/`resetAll`/`deinit`.
    pub fn loadStep(self: *AssetManager, max_tasks: usize) bool {
        var executed: usize = 0;
        while (self.current_index < self.tasks.items.len and executed < max_tasks) {
            // Re-read the pointer per iteration: callbacks may append (which
            // may reallocate the pointer array) — the task allocations
            // themselves stay stable.
            const task = self.tasks.items[self.current_index];
            self.current_index += 1;

            if (task.state != .pending) continue;

            task.state = .running;
            if (self.executeTask(task)) {
                task.state = .completed;
                self.completed_count += 1;
                if (self.onTaskSuccess) |cb| cb(self, task);
            } else |err| {
                task.state = .failed;
                task.error_result = err;
                self.failed_count += 1;
                if (self.onTaskError) |cb| cb(self, task, err);
            }

            executed += 1;
            if (self.onProgress) |cb| cb(self, self.remainingCount(), self.tasks.items.len, task);
        }

        const done = self.isDone();
        if (done and self.onFinish != null) {
            const cb = self.onFinish.?;
            self.onFinish = null; // Fire once
            cb(self);
        }
        return done;
    }

    /// Loads all tasks synchronously to completion.
    pub fn loadSync(self: *AssetManager) void {
        _ = self.loadStep(std.math.maxInt(usize));
    }
};

test "AssetManager text and binary loading with cache and progress" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var mgr = AssetManager.init(allocator);
    defer mgr.deinit();

    // Create scratch files for test
    const io = std.Io.Threaded.global_single_threaded.io();
    const test_txt_path = "test_asset_file.txt";
    const test_bin_path = "test_asset_file.bin";

    {
        const f = try std.Io.Dir.cwd().createFile(io, test_txt_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, "Hello AssetManager", 0);
    }
    defer std.Io.Dir.cwd().deleteFile(io, test_txt_path) catch {};

    {
        const f = try std.Io.Dir.cwd().createFile(io, test_bin_path, .{});
        defer f.close(io);
        const raw_bin = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF, 0x42 };
        _ = try f.writePositionalAll(io, &raw_bin, 0);
    }
    defer std.Io.Dir.cwd().deleteFile(io, test_bin_path) catch {};

    _ = try mgr.addTextFileTask("text_sample", test_txt_path);
    _ = try mgr.addBinaryFileTask("bin_sample", test_bin_path);

    try testing.expectEqual(@as(usize, 2), mgr.totalCount());
    try testing.expectEqual(@as(usize, 2), mgr.remainingCount());
    try testing.expectApproxEqAbs(@as(f32, 0.0), mgr.progress(), 1e-4);

    // Step 1: run first task
    const done1 = mgr.loadStep(1);
    try testing.expect(!done1);
    try testing.expectEqual(@as(usize, 1), mgr.remainingCount());
    try testing.expectApproxEqAbs(@as(f32, 0.5), mgr.progress(), 1e-4);

    const t1 = mgr.getTaskByName("text_sample").?;
    try testing.expect(t1.isSuccess());
    try testing.expectEqualStrings("Hello AssetManager", t1.text_result.?);
    try testing.expect(!t1.from_cache);

    // Step 2: run second task
    const done2 = mgr.loadStep(1);
    try testing.expect(done2);
    try testing.expect(mgr.isDone());
    try testing.expectApproxEqAbs(@as(f32, 1.0), mgr.progress(), 1e-4);

    const t2 = mgr.getTaskByName("bin_sample").?;
    try testing.expect(t2.isSuccess());
    try testing.expectEqual(@as(usize, 5), t2.binary_result.?.len);
    try testing.expectEqual(@as(u8, 0xDE), t2.binary_result.?[0]);

    // Verify cache has both
    try testing.expect(mgr.cache.hasText(test_txt_path));
    try testing.expect(mgr.cache.hasBinary(test_bin_path));

    // Reset tasks and load text again; should hit cache
    mgr.reset();
    try testing.expectEqual(@as(usize, 0), mgr.totalCount());

    _ = try mgr.addTextFileTask("cached_text", test_txt_path);
    mgr.loadSync();

    const t_cached = mgr.getTaskByName("cached_text").?;
    try testing.expect(t_cached.isSuccess());
    try testing.expect(t_cached.from_cache);
    try testing.expectEqualStrings("Hello AssetManager", t_cached.text_result.?);
}

test "AssetManager custom task, error reporting, and callbacks" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var mgr = AssetManager.init(allocator);
    defer mgr.deinit();

    const Context = struct {
        success_count: usize = 0,
        error_count: usize = 0,
        progress_calls: usize = 0,
        finished: bool = false,
        counter: u32 = 0,
    };
    var ctx = Context{};
    mgr.user_data = &ctx;

    mgr.onTaskSuccess = struct {
        fn cb(m: *AssetManager, _: *AssetTask) void {
            const c: *Context = @ptrCast(@alignCast(m.user_data.?));
            c.success_count += 1;
        }
    }.cb;

    mgr.onTaskError = struct {
        fn cb(m: *AssetManager, _: *AssetTask, _: anyerror) void {
            const c: *Context = @ptrCast(@alignCast(m.user_data.?));
            c.error_count += 1;
        }
    }.cb;

    mgr.onProgress = struct {
        fn cb(m: *AssetManager, _: usize, _: usize, _: *AssetTask) void {
            const c: *Context = @ptrCast(@alignCast(m.user_data.?));
            c.progress_calls += 1;
        }
    }.cb;

    mgr.onFinish = struct {
        fn cb(m: *AssetManager) void {
            const c: *Context = @ptrCast(@alignCast(m.user_data.?));
            c.finished = true;
        }
    }.cb;

    const customGood = struct {
        fn run(task: *AssetTask, _: std.mem.Allocator) !void {
            const c: *Context = @ptrCast(@alignCast(task.user_ctx.?));
            c.counter += 10;
        }
    }.run;

    const customBad = struct {
        fn run(_: *AssetTask, _: std.mem.Allocator) !void {
            return error.SimulatedFailure;
        }
    }.run;

    _ = try mgr.addCustomTask("good_task", customGood, &ctx);
    _ = try mgr.addCustomTask("bad_task", customBad, &ctx);

    mgr.loadSync();

    try testing.expect(mgr.isDone());
    try testing.expect(mgr.hasErrors());
    try testing.expectEqual(@as(usize, 1), mgr.completed_count);
    try testing.expectEqual(@as(usize, 1), mgr.failed_count);
    try testing.expectEqual(@as(u32, 10), ctx.counter);
    try testing.expectEqual(@as(usize, 1), ctx.success_count);
    try testing.expectEqual(@as(usize, 1), ctx.error_count);
    try testing.expectEqual(@as(usize, 2), ctx.progress_calls);
    try testing.expect(ctx.finished);
}

test "AssetManager task pointers stay stable across growth and callbacks" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var mgr = AssetManager.init(allocator);
    defer mgr.deinit();

    const io = std.Io.Threaded.global_single_threaded.io();
    const tmp_path = "test_stable_first.txt";
    {
        const f = try std.Io.Dir.cwd().createFile(io, tmp_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, "stable", 0);
    }
    defer std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};

    const first = try mgr.addTextFileTask("first_task", tmp_path);
    _ = try mgr.addCustomTask("filler", struct {
        fn run(_: *AssetTask, _: std.mem.Allocator) !void {}
    }.run, null);

    const Ctx = struct {
        first_ptr: ?*AssetTask = null,
        saw_success: bool = false,
        progress_calls: usize = 0,
        progress_saw_first: bool = false,
        progress_first_state_ok: bool = false,
    };
    var ctx = Ctx{ .first_ptr = first };
    mgr.user_data = &ctx;
    mgr.onTaskSuccess = struct {
        fn cb(m: *AssetManager, task: *AssetTask) void {
            const c: *Ctx = @ptrCast(@alignCast(m.user_data.?));
            if (task == c.first_ptr) {
                c.saw_success = true;
                // Force pointer-array reallocation inside the callback: the
                // running task allocation must not move.
                var i: usize = 0;
                while (i < 64) : (i += 1) {
                    _ = m.addCustomTask("enqueued", struct {
                        fn run(_: *AssetTask, _: std.mem.Allocator) !void {}
                    }.run, null) catch return;
                }
            }
        }
    }.cb;
    mgr.onProgress = struct {
        fn cb(m: *AssetManager, _: usize, _: usize, task: *AssetTask) void {
            const c: *Ctx = @ptrCast(@alignCast(m.user_data.?));
            c.progress_calls += 1;
            // Identity + state must survive the in-callback growth above;
            // stored to ctx so the check can't be optimized away.
            if (task == c.first_ptr) {
                c.progress_saw_first = true;
                c.progress_first_state_ok = task.isSuccess() and
                    std.mem.eql(u8, task.name, "first_task");
            }
        }
    }.cb;

    mgr.loadSync();
    try testing.expect(ctx.saw_success);
    try testing.expect(ctx.progress_saw_first);
    try testing.expect(ctx.progress_first_state_ok);
    // 2 initial + 64 enqueued, each fires onProgress exactly once.
    try testing.expectEqual(@as(usize, 66), ctx.progress_calls);
    try testing.expect(mgr.isDone());
    try testing.expect(!mgr.hasErrors());
    try testing.expect(mgr.getTaskByName("first_task") == first);
    try testing.expect(first.isSuccess());
}

test "AssetManager mesh scene-less retains geometry, scene load spawns + cleanup" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const obj_path = "test_mesh_task.obj";
    const obj_text =
        \\v 0 0 0
        \\v 1 0 0
        \\v 0 1 0
        \\f 1 2 3
    ;
    {
        const f = try std.Io.Dir.cwd().createFile(io, obj_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, obj_text, 0);
    }
    defer std.Io.Dir.cwd().deleteFile(io, obj_path) catch {};

    // Scene-less: retains owned CPU geometry with real vertices/indices.
    {
        var mgr = AssetManager.init(allocator);
        defer mgr.deinit();
        _ = try mgr.addMeshTask("tri_noscene", obj_path, null);
        mgr.loadSync();
        const t = mgr.getTaskByName("tri_noscene").?;
        try testing.expect(t.isSuccess());
        try testing.expectEqual(@as(usize, 1), t.mesh_count);
        try testing.expect(t.owned_mesh);
        const g = t.mesh_result.?;
        try testing.expectEqual(@as(usize, 3), g.vertices.len);
        try testing.expectEqual(@as(usize, 3), g.indices.len);
        try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, g.indices);
        try testing.expectEqual(@as(f32, 1), g.vertices[1].position[0]);
        try testing.expectEqual(@as(usize, 3), t.mesh_vertex_count);
        try testing.expectEqual(@as(usize, 3), t.mesh_index_count);
        // take moves ownership out; reset must not double-free.
        var taken = t.takeMeshGeometry().?;
        defer taken.deinit(allocator);
        try testing.expect(!t.owned_mesh);
        try testing.expect(t.mesh_result == null);
    }

    // Scene load: real entity appears with matching counts; cleanup frees it.
    {
        var mgr = AssetManager.init(allocator);
        defer mgr.deinit();
        const testScene = @import("testing.zig").testScene;
        var scene = testScene(allocator);
        // testScene leaves GPU passes undefined; only allocator/meshes are
        // touched by the deferred upload path (no sg context in tests).
        defer {
            for (scene.meshes.items) |m| {
                m.deinit(allocator);
                allocator.destroy(m);
            }
            scene.meshes.deinit(allocator);
            scene.profiler.deinit();
        }
        _ = try mgr.addMeshTask("tri_scene", obj_path, &scene);
        mgr.loadSync();
        const t = mgr.getTaskByName("tri_scene").?;
        try testing.expect(t.isSuccess());
        try testing.expectEqual(@as(usize, 1), t.mesh_count);
        try testing.expectEqual(@as(usize, 1), scene.meshes.items.len);
        try testing.expectEqual(t.mesh_vertex_count, @as(usize, scene.meshes.items[0].vertex_count));
        try testing.expectEqual(t.mesh_index_count, @as(usize, scene.meshes.items[0].index_count));
        try testing.expectEqual(@as(usize, 3), t.mesh_vertex_count);
        try testing.expectEqual(@as(usize, 3), t.mesh_index_count);
    }
}

test "AssetManager mesh failures are explicit, never fake success" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const bad_obj_path = "test_mesh_bad.obj";
    {
        const f = try std.Io.Dir.cwd().createFile(io, bad_obj_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, "this is not a mesh {{{ }}}", 0);
    }
    defer std.Io.Dir.cwd().deleteFile(io, bad_obj_path) catch {};

    const unsupported_path = "test_mesh_task.xyz";
    {
        const f = try std.Io.Dir.cwd().createFile(io, unsupported_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, "junk", 0);
    }
    defer std.Io.Dir.cwd().deleteFile(io, unsupported_path) catch {};

    var mgr = AssetManager.init(allocator);
    defer mgr.deinit();

    _ = try mgr.addMeshTask("missing_file", "no_such_mesh_file.obj", null);
    _ = try mgr.addMeshTask("bad_obj", bad_obj_path, null);
    _ = try mgr.addMeshTask("unsupported_ext", unsupported_path, null);
    _ = try mgr.addMeshTask("glb_needs_scene", "whatever.glb", null);

    mgr.loadSync();

    try testing.expectEqual(@as(usize, 0), mgr.completed_count);
    try testing.expectEqual(@as(usize, 4), mgr.failed_count);
    for ([_][]const u8{ "missing_file", "bad_obj", "unsupported_ext", "glb_needs_scene" }) |name| {
        const t = mgr.getTaskByName(name).?;
        try testing.expect(t.isFailed());
        try testing.expect(!t.isSuccess());
        try testing.expect(t.error_result != null);
        try testing.expectEqual(@as(usize, 0), t.mesh_count);
        try testing.expectEqual(@as(usize, 0), t.mesh_vertex_count);
        try testing.expectEqual(@as(usize, 0), t.mesh_index_count);
        try testing.expect(t.mesh_result == null);
        try testing.expect(!t.owned_mesh);
    }
    try testing.expect(mgr.getTaskByName("glb_needs_scene").?.error_result.? == error.MeshRequiresScene);
    try testing.expect(mgr.getTaskByName("unsupported_ext").?.error_result.? == error.UnsupportedMeshFormat);
}

test "AssetCache overwrite replaces value without leaking keys" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var cache = AssetCache.init(allocator);
    defer cache.deinit();

    _ = try cache.putText("k", "first");
    _ = try cache.putText("k", "second");
    try testing.expectEqual(@as(usize, 1), cache.count());
    try testing.expectEqualStrings("second", cache.getText("k").?);

    _ = try cache.putBinary("b", "123");
    _ = try cache.putBinary("b", "4567");
    try testing.expectEqualStrings("4567", cache.getBinary("b").?);
}

test "textureHandlesEqual compares GPU identity without touching sg" {
    const testing = std.testing;
    var a = std.mem.zeroes(Texture);
    var b = std.mem.zeroes(Texture);
    a.width = 4;
    a.height = 4;
    b.width = 4;
    b.height = 4;
    try testing.expect(textureHandlesEqual(a, b));
    b.height = 8;
    try testing.expect(!textureHandlesEqual(a, b));
    b.height = 4;
    b.image.id += 1;
    try testing.expect(!textureHandlesEqual(a, b));
}
