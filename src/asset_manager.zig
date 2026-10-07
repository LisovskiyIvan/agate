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
//! - Opt-in async mesh parsing (`addMeshTaskAsync` + `enableMeshAsync`):
//!   OBJ/STL/PLY file read + parse + CPU geometry build run on a
//!   `jobs.TaskRunner` worker; `loadStep`/`loadSync` (caller/context thread)
//!   drain completions without blocking (`loadStep`) and only there append
//!   to the Scene (`uploadGeometry` + morph-base retain, mirroring
//!   `appendToScene`) or publish owned `mesh_result`. No second worker
//!   pipeline: the existing `jobs.TaskRunner` is reused (UploadQueue stays
//!   the texture route). GLB/GLTF stay sync-only (`addMeshTaskAsync`
//!   rejects them with `error.UnsupportedMeshFormat`): cgltf parsing plus
//!   material/texture/sg work is not worker-safe. GPU upload still happens
//!   on the context thread — async only moves I/O + parse off it.
//!   `loadStep` polls async completions non-blockingly; `max_tasks` bounds
//!   submits AND completions drained per call. `loadSync` blocks until every
//!   async worker result is drained. See `AssetManager` docs.
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
//! - Async mesh tasks (`is_async`): `name`/`path` stay BORROWED with the same
//!   lifetime rule. The worker never touches the `AssetTask` or the `Scene`:
//!   it owns a heap `AsyncMeshJob` (borrowed `path` bytes, manager
//!   allocator) and publishes owned CPU `GeometryData` into the job. The
//!   caller thread moves that geometry into `mesh_result` (scene-less) or
//!   into the Scene (upload + retain, freeing the worker copy) during
//!   `loadStep`/`loadSync` drain. Unclaimed worker geometry is freed by
//!   `reset`/`deinit` after joining the worker (release/acquire handshake
//!   on `done`). The manager allocator must be thread-safe for the async
//!   path (worker allocates/parses on it). Thread rule: worker = file read
//!   + parse only (no `sg.*`, no Scene, no callbacks); caller = Scene
//!   append + callbacks + GPU upload (context thread, still required).

const std = @import("std");

const Scene = @import("scene.zig").Scene;
const Mesh = @import("mesh.zig").Mesh;
const GeometryData = @import("mesh.zig").GeometryData;
const Vertex = @import("mesh.zig").Vertex;
const computeTangents = @import("mesh.zig").computeTangents;
const uploadGeometry = @import("mesh.zig").uploadGeometry;
const Texture = @import("texture.zig").Texture;
const SceneLoader = @import("loader/scene_loader.zig").SceneLoader;
const obj_loader = @import("loader/obj.zig");
const stl_loader = @import("loader/stl.zig");
const ply_loader = @import("loader/ply.zig");
const gpu_thread = @import("gpu_thread.zig");
const jobs = @import("jobs.zig");
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

    // Opt-in async mesh parsing (see `addMeshTaskAsync`). False for every
    // other task type. Async tasks run file read + parse on a
    // `jobs.TaskRunner` worker and finalize on the caller thread.
    is_async: bool = false,
    /// In-flight worker handoff (non-null while `.running` async). Owned by
    /// the manager: freed on drain (`loadStep`/`loadSync`) or, unclaimed, by
    /// `reset`/`deinit` after the worker handshake. Never touched by the
    /// worker's task-state path (the worker only writes the job).
    async_job: ?*AsyncMeshJob = null,
    /// Worker thread id that parsed this task (copied on drain; null until
    /// the first successful or failed drain). Test observable proving the
    /// parse left the calling thread.
    async_worker_id: ?std.Thread.Id = null,

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

/// Worker-safe mesh kinds for the async path: pure CPU parse with no Scene,
/// no `sg.*`, no cache. GLB/GLTF are excluded (cgltf + materials/textures +
/// sg work is not worker-safe) and stay sync-only.
const AsyncMeshKind = enum { obj, stl, ply };

fn asyncKindForPath(path: []const u8) ?AsyncMeshKind {
    if (std.mem.endsWith(u8, path, ".obj")) return .obj;
    if (std.mem.endsWith(u8, path, ".stl")) return .stl;
    if (std.mem.endsWith(u8, path, ".ply")) return .ply;
    return null;
}

/// Parses raw file bytes into owned CPU geometry (worker side). Mirrors the
/// sync scene-less branch exactly (same loaders, same `buildCpuGeometry`
/// inputs) so async and sync scene-less results are identical. No Scene, no
/// `sg.*`, no cache — allocator must be thread-safe.
fn parseMeshBytesToGeometry(allocator: std.mem.Allocator, kind: AsyncMeshKind, bytes: []const u8) !GeometryData {
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

/// Test-only hold gate: while true, async workers spin after publishing
/// `started`/`worker_id` and before parsing. Lets a test observe `.running`
/// deterministically, then release. Never set with a zero-thread (inline)
/// runner — the posting thread would spin on its own gate.
pub var async_test_hold: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);

/// Heap handoff between the `TaskRunner` worker and the caller thread. The
/// worker writes `worker_id`, then `started` (release), then parses into
/// `result`/`err`, then `done` (release). The caller reads `result`/`err`
/// only after `done == true` (acquire). The worker never touches the
/// `AssetTask` or the `Scene`; `path` bytes are borrowed from the caller
/// (same lifetime rule as `AssetTask.path`).
const AsyncMeshJob = struct {
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

/// `jobs.TaskRunner` entry: file read + parse + CPU geometry build. No
/// `sg.*`, no Scene mutation, no callbacks, no task-state writes.
fn asyncMeshRun(ctx: *anyopaque) void {
    const job: *AsyncMeshJob = @ptrCast(@alignCast(ctx));
    job.worker_id = std.Thread.getCurrentId();
    job.started.store(true, .release);
    while (async_test_hold.load(.acquire)) jobs.sleepNs(50_000);
    const bytes = readFileBytesAlloc(job.allocator, job.path) catch |e| {
        job.err = e;
        job.done.store(true, .release);
        return;
    };
    defer job.allocator.free(bytes);
    const geom = parseMeshBytesToGeometry(job.allocator, job.kind, bytes) catch |e| {
        job.err = e;
        job.done.store(true, .release);
        return;
    };
    job.result = geom;
    job.done.store(true, .release);
}

pub const TaskSuccessFn = *const fn (manager: *AssetManager, task: *AssetTask) void;
pub const TaskErrorFn = *const fn (manager: *AssetManager, task: *AssetTask, err: anyerror) void;
pub const ProgressFn = *const fn (manager: *AssetManager, remaining: usize, total: usize, task: *AssetTask) void;
pub const FinishFn = *const fn (manager: *AssetManager) void;

/// Central asset loading manager: synchronous batch loading with progress,
/// caching, and stable task pointers, plus an opt-in async mesh path.
/// See module docs for the execution / context-thread contract.
/// (`UploadQueue` stays the texture async route; async mesh reuses the
/// existing `jobs.TaskRunner` — no second worker pipeline.)
///
/// Threading: one caller at a time on `loadStep`/`loadSync`/`reset`/
/// `deinit` (not internally synchronized). Async workers only touch their
/// `AsyncMeshJob` (never the task, Scene, `sg.*`, or callbacks); the caller
/// thread owns every task-state transition, Scene append, and callback.
pub const AssetManager = struct {
    allocator: std.mem.Allocator,
    /// Individually allocated tasks: returned `*AssetTask` pointers stay
    /// stable across later `add*` calls (the pointer array may reallocate,
    /// the task allocations do not move). Callbacks may safely enqueue new
    /// tasks and keep using the current task pointer. Callbacks must NOT
    /// call `reset`/`resetAll`/`deinit` (that would free the running task).
    tasks: std.ArrayListUnmanaged(*AssetTask) = .empty,
    cache: AssetCache,
    /// Lazily created worker pool for `addMeshTaskAsync` tasks (owned iff
    /// `async_owned`). Kept across `reset` for reuse; joined in `deinit`
    /// (and `disableMeshAsync`). Null until first async use.
    async_runner: ?*jobs.TaskRunner = null,
    async_owned: bool = false,

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
        if (self.async_runner) |runner| {
            if (self.async_owned) runner.deinit();
            self.async_runner = null;
            self.async_owned = false;
        }
    }

    /// Clears task queue and frees task-owned data, keeping the cache intact.
    /// Borrowed cache aliases (text/binary/texture) are NOT freed/deinited.
    /// Async safety: in-flight worker jobs are joined (spin on `done` with a
    /// short park — workers only parse, so they always finish) and their
    /// unclaimed geometry freed; the runner itself is kept for reuse
    /// (joined in `deinit`/`disableMeshAsync`). Caller thread only.
    pub fn reset(self: *AssetManager) void {
        for (self.tasks.items) |task| {
            if (task.async_job) |job| {
                while (!job.done.load(.acquire)) jobs.sleepNs(50_000);
                if (job.result) |*g| g.deinit(self.allocator);
                self.allocator.destroy(job);
                task.async_job = null;
            }
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

    /// Opt-in async mesh task: OBJ/STL/PLY file read + parse + CPU geometry
    /// build run on a `jobs.TaskRunner` worker; Scene append (or owned
    /// `mesh_result` publish) happens on the caller thread during
    /// `loadStep`/`loadSync` drain. GLB/GLTF and unknown extensions are
    /// rejected here with `error.UnsupportedMeshFormat` — they stay
    /// sync-only via `addMeshTask` (cgltf + materials/textures + `sg.*`
    /// work is not worker-safe). The runner is created lazily (1 worker)
    /// when the first async task submits; call `enableMeshAsync` first to
    /// pick the thread count. Nothing runs until `loadStep`/`loadSync`
    /// (same as sync tasks). Requires a thread-safe manager allocator.
    pub fn addMeshTaskAsync(self: *AssetManager, name: []const u8, path: []const u8, scene: ?*Scene) !*AssetTask {
        if (asyncKindForPath(path) == null) return error.UnsupportedMeshFormat;
        const task = try self.allocator.create(AssetTask);
        errdefer self.allocator.destroy(task);
        task.* = .{
            .name = name,
            .path = path,
            .task_type = .mesh,
            .scene = scene,
            .is_async = true,
        };
        try self.tasks.append(self.allocator, task);
        return task;
    }

    /// Creates the async worker pool explicitly (idempotent). Without this,
    /// the first async submit lazily creates 1 worker. `thread_count == 0`
    /// creates an inline runner (tasks run on the posting thread — still
    /// correct, no parallelism). Caller thread only.
    pub fn enableMeshAsync(self: *AssetManager, thread_count: usize) !void {
        if (self.async_runner != null) return;
        self.async_runner = try jobs.TaskRunner.init(self.allocator, thread_count);
        self.async_owned = true;
    }

    /// Joins the async workers and drops the pool; queued/in-flight jobs
    /// run to completion inside `TaskRunner.deinit` (their results stay
    /// parked in their jobs for a later `loadStep` drain or `reset` free).
    /// New async submits lazily recreate the pool. Caller thread only.
    pub fn disableMeshAsync(self: *AssetManager) void {
        if (self.async_runner) |runner| {
            if (self.async_owned) runner.deinit();
            self.async_runner = null;
            self.async_owned = false;
        }
    }

    fn ensureAsyncRunner(self: *AssetManager) !void {
        if (self.async_runner != null) return;
        try self.enableMeshAsync(1);
    }

    /// Posts a pending async mesh task to the runner (caller thread).
    /// Sets `.running`; completion (or failure) is published later by
    /// `drainAsyncMesh` on the caller thread. Never touches the Scene.
    fn submitAsyncMesh(self: *AssetManager, task: *AssetTask) anyerror!void {
        const kind = asyncKindForPath(task.path) orelse return error.UnsupportedMeshFormat;
        try self.ensureAsyncRunner();
        const job = try self.allocator.create(AsyncMeshJob);
        job.* = .{
            .kind = kind,
            .path = task.path,
            .allocator = self.allocator,
        };
        task.async_job = job;
        task.state = .running;
        self.async_runner.?.post(@ptrCast(job), asyncMeshRun);
    }

    /// Finalizes one worker-finished async task on the caller thread:
    /// publishes `mesh_result` (scene-less) or appends to the Scene via
    /// `uploadGeometry` + name dupe + morph-base retain (mirroring
    /// `appendToScene`; the worker copy is then freed). Failures set
    /// `error_result` exactly like the sync path. Frees the job.
    /// Precondition: `job.done.load(.acquire) == true`.
    fn drainAsyncMesh(self: *AssetManager, task: *AssetTask, job: *AsyncMeshJob) void {
        task.async_worker_id = job.worker_id;
        if (job.err) |e| {
            task.state = .failed;
            task.error_result = e;
            self.failed_count += 1;
            self.allocator.destroy(job);
            task.async_job = null;
            if (self.onTaskError) |cb| cb(self, task, e);
            if (self.onProgress) |cb| cb(self, self.remainingCount(), self.tasks.items.len, task);
            return;
        }
        var geom = job.result orelse {
            task.state = .failed;
            task.error_result = error.AsyncMeshMissingResult;
            self.failed_count += 1;
            self.allocator.destroy(job);
            task.async_job = null;
            if (self.onTaskError) |cb| cb(self, task, error.AsyncMeshMissingResult);
            if (self.onProgress) |cb| cb(self, self.remainingCount(), self.tasks.items.len, task);
            return;
        };
        if (task.scene) |sc| {
            const owned_name = sc.allocator.dupe(u8, task.name) catch |e| {
                geom.deinit(self.allocator);
                task.state = .failed;
                task.error_result = e;
                self.failed_count += 1;
                self.allocator.destroy(job);
                task.async_job = null;
                if (self.onTaskError) |cb| cb(self, task, e);
                if (self.onProgress) |cb| cb(self, self.remainingCount(), self.tasks.items.len, task);
                return;
            };
            errdefer sc.allocator.free(owned_name);
            const mesh_obj = uploadGeometry(sc, owned_name, geom) catch |e| {
                geom.deinit(self.allocator);
                task.state = .failed;
                task.error_result = e;
                self.failed_count += 1;
                self.allocator.destroy(job);
                task.async_job = null;
                if (self.onTaskError) |cb| cb(self, task, e);
                if (self.onProgress) |cb| cb(self, self.remainingCount(), self.tasks.items.len, task);
                return;
            };
            mesh_obj.owns_name = true;
            mesh_obj.retainMorphBase(sc.allocator, geom.vertices) catch |e| {
                geom.deinit(self.allocator);
                task.state = .failed;
                task.error_result = e;
                self.failed_count += 1;
                self.allocator.destroy(job);
                task.async_job = null;
                if (self.onTaskError) |cb| cb(self, task, e);
                if (self.onProgress) |cb| cb(self, self.remainingCount(), self.tasks.items.len, task);
                return;
            };
            task.mesh_count = 1;
            task.mesh_vertex_count = geom.vertices.len;
            task.mesh_index_count = geom.indices.len;
            geom.deinit(self.allocator);
            task.state = .completed;
            self.completed_count += 1;
            self.allocator.destroy(job);
            task.async_job = null;
            if (self.onTaskSuccess) |cb| cb(self, task);
            if (self.onProgress) |cb| cb(self, self.remainingCount(), self.tasks.items.len, task);
        } else {
            task.mesh_result = geom;
            task.owned_mesh = true;
            task.mesh_count = 1;
            task.mesh_vertex_count = geom.vertices.len;
            task.mesh_index_count = geom.indices.len;
            task.state = .completed;
            self.completed_count += 1;
            self.allocator.destroy(job);
            task.async_job = null;
            if (self.onTaskSuccess) |cb| cb(self, task);
            if (self.onProgress) |cb| cb(self, self.remainingCount(), self.tasks.items.len, task);
        }
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

    /// Runs up to `max_tasks` tasks (sync inline + async submit/drain; see
    /// module docs). Polls async completions NON-BLOCKINGLY first (each
    /// drained completion consumes 1 of `max_tasks`), then runs/submits
    /// pending tasks in order (sync tasks execute inline; async mesh tasks
    /// post to the runner and go `.running` without firing progress yet —
    /// progress/success/error for async tasks fire on the later drain, on
    /// the caller thread). A task never reports success before its result
    /// is available. Callbacks may enqueue via `add*` (pointers stable);
    /// they must not call `reset`/`resetAll`/`deinit`.
    pub fn loadStep(self: *AssetManager, max_tasks: usize) bool {
        var executed: usize = 0;
        // Phase 1: drain worker-finished async tasks (non-blocking poll).
        // Scan the whole list: submitted tasks sit behind `current_index`.
        if (executed < max_tasks) {
            var i: usize = 0;
            while (i < self.tasks.items.len and executed < max_tasks) {
                // Re-read per iteration: drain fires callbacks that may
                // append (pointer array may reallocate; tasks stay stable).
                const task = self.tasks.items[i];
                i += 1;
                const job = task.async_job orelse continue;
                if (task.state != .running) continue;
                if (!job.done.load(.acquire)) continue;
                self.drainAsyncMesh(task, job);
                executed += 1;
            }
        }
        while (self.current_index < self.tasks.items.len and executed < max_tasks) {
            // Re-read the pointer per iteration: callbacks may append (which
            // may reallocate the pointer array) — the task allocations
            // themselves stay stable.
            const task = self.tasks.items[self.current_index];
            self.current_index += 1;

            if (task.state != .pending) continue;

            if (task.is_async and task.task_type == .mesh) {
                if (self.submitAsyncMesh(task)) {
                    // Submitted: `.running`, no callback yet (fires on drain).
                    executed += 1;
                } else |err| {
                    task.state = .failed;
                    task.error_result = err;
                    self.failed_count += 1;
                    if (self.onTaskError) |cb| cb(self, task, err);
                    executed += 1;
                    if (self.onProgress) |cb| cb(self, self.remainingCount(), self.tasks.items.len, task);
                }
                continue;
            }

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

    /// Loads all tasks to completion. Sync tasks run inline; async mesh
    /// tasks BLOCK here until every worker result is drained (short parks
    /// between non-blocking polls — never a hot spin). Caller/context
    /// thread: Scene appends and GPU uploads still happen here.
    pub fn loadSync(self: *AssetManager) void {
        while (true) {
            const done = self.loadStep(std.math.maxInt(usize));
            if (done) return;
            jobs.sleepNs(1_000_000);
        }
    }
};

test "AssetManager text and binary loading with cache and progress" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var mgr = AssetManager.init(allocator);
    defer mgr.deinit();

    // Scratch files live in an isolated per-run directory (unique,
    // auto-cleaned): fixed CWD-relative names would collide with a
    // concurrent `zig build test` run of the same suite.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const test_txt_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/test_asset_file.txt", .{tmp.sub_path});
    defer allocator.free(test_txt_path);
    const test_bin_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/test_asset_file.bin", .{tmp.sub_path});
    defer allocator.free(test_bin_path);

    const io = std.Io.Threaded.global_single_threaded.io();

    {
        const f = try std.Io.Dir.cwd().createFile(io, test_txt_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, "Hello AssetManager", 0);
    }

    {
        const f = try std.Io.Dir.cwd().createFile(io, test_bin_path, .{});
        defer f.close(io);
        const raw_bin = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF, 0x42 };
        _ = try f.writePositionalAll(io, &raw_bin, 0);
    }

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
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/test_stable_first.txt", .{tmp.sub_path});
    defer allocator.free(tmp_path);
    {
        const f = try std.Io.Dir.cwd().createFile(io, tmp_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, "stable", 0);
    }

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
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const obj_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/test_mesh_task.obj", .{tmp.sub_path});
    defer allocator.free(obj_path);
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
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const bad_obj_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/test_mesh_bad.obj", .{tmp.sub_path});
    defer allocator.free(bad_obj_path);
    {
        const f = try std.Io.Dir.cwd().createFile(io, bad_obj_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, "this is not a mesh {{{ }}}", 0);
    }

    const unsupported_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/test_mesh_task.xyz", .{tmp.sub_path});
    defer allocator.free(unsupported_path);
    {
        const f = try std.Io.Dir.cwd().createFile(io, unsupported_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, "junk", 0);
    }

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

test "AssetManager async mesh parses on a worker thread (gated handshake)" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const obj_path = "test_async_mesh.obj";
    {
        const f = try std.Io.Dir.cwd().createFile(io, obj_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", 0);
    }
    defer std.Io.Dir.cwd().deleteFile(io, obj_path) catch {};

    var mgr = AssetManager.init(allocator);
    defer mgr.deinit();
    try mgr.enableMeshAsync(1);

    const caller_id = std.Thread.getCurrentId();
    async_test_hold.store(true, .release);
    defer async_test_hold.store(false, .release);
    const t = try mgr.addMeshTaskAsync("tri_async", obj_path, null);
    try testing.expect(t.is_async);
    try testing.expectEqual(TaskState.pending, t.state);

    // Submit only: must go `.running` without completing (worker is gated).
    const done_submit = mgr.loadStep(1);
    try testing.expect(!done_submit);
    try testing.expectEqual(TaskState.running, t.state);
    try testing.expect(!t.isSuccess());
    try testing.expect(t.mesh_result == null);

    // Handshake: wait (bounded) for the worker to start and block in the
    // gate — proves the parse left the calling thread, with no timing luck.
    const job = t.async_job.?;
    var spins: usize = 0;
    while (!job.started.load(.acquire)) : (spins += 1) {
        if (spins > 100_000) return error.AsyncWorkerNeverStarted;
        jobs.sleepNs(50_000);
    }
    try testing.expect(job.worker_id.? != caller_id);
    // Still running while the worker is blocked: no fake success.
    try testing.expectEqual(TaskState.running, t.state);
    try testing.expect(!t.isSuccess());

    async_test_hold.store(false, .release);
    mgr.loadSync();

    try testing.expect(t.isSuccess());
    try testing.expectEqual(@as(usize, 1), t.mesh_count);
    try testing.expect(t.owned_mesh);
    try testing.expectEqual(@as(usize, 3), t.mesh_result.?.vertices.len);
    try testing.expectEqual(@as(usize, 3), t.mesh_result.?.indices.len);
    try testing.expectEqual(@as(usize, 3), t.mesh_vertex_count);
    try testing.expectEqual(@as(usize, 3), t.mesh_index_count);
    try testing.expect(t.async_worker_id.? != caller_id);
    try testing.expectEqual(@as(usize, 1), mgr.completed_count);
    try testing.expectEqual(@as(usize, 0), mgr.failed_count);
    try testing.expect(mgr.isDone());
}

test "AssetManager async mesh scene load + failures drain on caller" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const obj_path = "test_async_scene.obj";
    {
        const f = try std.Io.Dir.cwd().createFile(io, obj_path, .{});
        defer f.close(io);
        _ = try f.writePositionalAll(io, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", 0);
    }
    defer std.Io.Dir.cwd().deleteFile(io, obj_path) catch {};

    var mgr = AssetManager.init(allocator);
    defer mgr.deinit();
    try mgr.enableMeshAsync(1);

    const testScene = @import("testing.zig").testScene;
    var scene = testScene(allocator);
    defer {
        for (scene.meshes.items) |m| {
            m.deinit(allocator);
            allocator.destroy(m);
        }
        scene.meshes.deinit(allocator);
        scene.profiler.deinit();
    }

    const t = try mgr.addMeshTaskAsync("tri_scene_async", obj_path, &scene);
    const bad = try mgr.addMeshTaskAsync("missing_async", "no_such_async_file.obj", null);
    // GLB stays sync-only: rejected at submit time, explicitly.
    try testing.expectError(error.UnsupportedMeshFormat, mgr.addMeshTaskAsync("glb_async", "x.glb", &scene));

    mgr.loadSync();

    try testing.expect(t.isSuccess());
    try testing.expectEqual(@as(usize, 1), t.mesh_count);
    try testing.expectEqual(@as(usize, 1), scene.meshes.items.len);
    try testing.expectEqual(t.mesh_vertex_count, @as(usize, scene.meshes.items[0].vertex_count));
    try testing.expectEqual(t.mesh_index_count, @as(usize, scene.meshes.items[0].index_count));
    try testing.expect(t.mesh_result == null);
    try testing.expect(!t.owned_mesh);

    try testing.expect(bad.isFailed());
    try testing.expect(bad.error_result != null);
    try testing.expectEqual(@as(usize, 0), bad.mesh_count);
    try testing.expect(bad.mesh_result == null);
    try testing.expect(!bad.owned_mesh);
    try testing.expectEqual(@as(usize, 1), mgr.completed_count);
    try testing.expectEqual(@as(usize, 1), mgr.failed_count);

    // Reset frees everything (including any unclaimed worker payload path)
    // with a clean testing allocator; runner survives for reuse.
    mgr.reset();
    try testing.expectEqual(@as(usize, 0), mgr.totalCount());
    try testing.expect(mgr.async_runner != null);
}
