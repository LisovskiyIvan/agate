//! AssetManager owner. Split out of `asset_manager.zig` (facade).
//!
//! This module owns the `AssetManager` type: task queue, progress, the
//! `add*` constructors, the per-format `executeTask` dispatch, the stepped
//! (`loadStep`) / blocking (`loadSync`) drivers, and the async mesh
//! submit/drain glue. Task types live in `task.zig`, the cache in
//! `cache.zig`, CPU mesh helpers in `mesh_cpu.zig`, async parse support in
//! `async_mesh.zig` (reached through the local aliases below so moved
//! method bodies are byte-identical); the worker entry itself
//! (`asyncMeshRun`) stays in the facade with the test gate it reads.
//!
//! Anti-cycle rule: leaves never import the `asset_manager.zig` facade
//! back. The one edge into it (`facade.asyncMeshRun`, posted by
//! `submitAsyncMesh`) is a leaf-ward function-pointer use from this owner
//! module, resolved lazily in the method body.

const std = @import("std");

const scene_mod = @import("../scene.zig");
const mesh_mod = @import("../mesh.zig");
const texture_mod = @import("../texture.zig");
const loader_mod = @import("../loader/scene_loader.zig");
const obj_loader = @import("../loader/obj.zig");
const stl_loader = @import("../loader/stl.zig");
const ply_loader = @import("../loader/ply.zig");
const gpu_thread = @import("../gpu_thread.zig");
const jobs = @import("../jobs.zig");
const sokol = @import("sokol");
const sg = sokol.gfx;
const facade = @import("../asset_manager.zig");
const task_mod = @import("task.zig");
const cache_mod = @import("cache.zig");
const mesh_cpu = @import("mesh_cpu.zig");
const async_mesh = @import("async_mesh.zig");

const Scene = scene_mod.Scene;
const GeometryData = mesh_mod.GeometryData;
const uploadGeometry = mesh_mod.uploadGeometry;
const Texture = texture_mod.Texture;
const SceneLoader = loader_mod.SceneLoader;
const AssetTask = task_mod.AssetTask;
const AssetCache = cache_mod.AssetCache;
const AsyncMeshJob = async_mesh.AsyncMeshJob;
const readFileBytesAlloc = mesh_cpu.readFileBytesAlloc;
const sumSpawned = mesh_cpu.sumSpawned;
const buildCpuGeometry = mesh_cpu.buildCpuGeometry;
const asyncKindForPath = async_mesh.asyncKindForPath;
const asyncMeshRun = facade.asyncMeshRun;

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
    /// Stale scene targets skipped without touching the scene: async drains
    /// plus sync executes that found their `task.scene` torn down (or from
    /// a recycled generation), plus targets detached by `clearSceneRefs`.
    /// Monotonic diagnostic; never resets the task outcome by itself (each
    /// stale task still lands `.failed` with `error.StaleScene`, or
    /// completes scene-less after an explicit detach).
    stale_scene_drops: usize = 0,

    // Callbacks
    onTaskSuccess: ?TaskSuccessFn = null,
    onTaskError: ?TaskErrorFn = null,
    onProgress: ?ProgressFn = null,
    onFinish: ?FinishFn = null,
    user_data: ?*anyopaque = null,
    /// Latch: `onFinish` already fired for the current done-generation.
    /// Rearmed whenever not done (at `loadStep` entry, covering every `add*`
    /// path) and by `reset`; set BEFORE invoking `onFinish`.
    finish_notified: bool = false,

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
                // Owned-texture teardown routes through the guarded
                // Texture.deinit (context-thread debug tripwire when live
                // handles are destroyed); no separate assert here.
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
        self.finish_notified = false;
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
            .scene_uid = if (scene) |sc| sc.lifetime_uid else 0,
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
            .scene_uid = if (scene) |sc| sc.lifetime_uid else 0,
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

    /// True when a scene target must not be touched: teardown began
    /// (`scene_alive` cleared by `Scene.deinit`) or the live scene reports
    /// a different lifetime generation than the one captured at submit time
    /// (storage freed and recycled for a new scene). Best effort: reading
    /// these two fields through the stored pointer is only sound while the
    /// storage is still mapped — the binding contract is `clearSceneRefs`
    /// before the Scene storage is freed.
    fn sceneTargetStale(sc: *const Scene, scene_uid: u64) bool {
        if (!sc.scene_alive) return true;
        if (scene_uid != 0 and sc.lifetime_uid != scene_uid) return true;
        return false;
    }

    /// Detaches every task targeting `scene` (sync and async, any state):
    /// their scene pointer is cleared so later drains/executes never touch
    /// the scene, even after it is deinited and its storage freed or
    /// recycled. Detached async mesh tasks complete as scene-less
    /// owned-geometry publishes (`mesh_result`); detached sync mesh tasks
    /// take the scene-less CPU path. Returns the detached count and bumps
    /// `stale_scene_drops` by the same amount. CALL THIS before
    /// deiniting/freeing a Scene that any live task still targets — it is
    /// the only check that stays sound once the storage is unmapped.
    /// Caller thread only.
    pub fn clearSceneRefs(self: *AssetManager, scene: *Scene) usize {
        var detached: usize = 0;
        for (self.tasks.items) |task| {
            if (task.scene == scene) {
                task.scene = null;
                task.scene_uid = 0;
                detached += 1;
            }
        }
        self.stale_scene_drops += detached;
        return detached;
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
        if (task.scene) |sc| {
            if (sceneTargetStale(sc, task.scene_uid)) {
                // The scene died (or its storage was recycled for a new
                // scene) while the worker parsed: never touch it. Free the
                // worker geometry and fail closed.
                if (job.result) |*g| g.deinit(self.allocator);
                task.state = .failed;
                task.error_result = error.StaleScene;
                self.failed_count += 1;
                self.stale_scene_drops += 1;
                self.allocator.destroy(job);
                task.async_job = null;
                if (self.onTaskError) |cb| cb(self, task, error.StaleScene);
                if (self.onProgress) |cb| cb(self, self.remainingCount(), self.tasks.items.len, task);
                return;
            }
        }
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
                    if (sceneTargetStale(sc, task.scene_uid)) {
                        // Sync mirror of the async-drain guard above: the
                        // scene died (or was recycled) between submit and
                        // execute. Count it and fail closed without touching
                        // the scene; `loadStep` records the failure.
                        self.stale_scene_drops += 1;
                        return error.StaleScene;
                    }
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
    /// Returns true once every task has completed or failed. `onFinish`
    /// fires once per completed batch (stays installed across batches and
    /// `reset`; polls while already done do not re-fire; clearing it to
    /// null disables). A task enqueued from `onFinish` makes the return
    /// re-evaluate to false so `loadSync` keeps draining.
    pub fn loadStep(self: *AssetManager, max_tasks: usize) bool {
        // Rearm: any pending work (including tasks appended after a prior
        // completion) starts a new generation that may fire `onFinish` again.
        if (!self.isDone()) self.finish_notified = false;
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

        var done = self.isDone();
        if (done and !self.finish_notified) {
            if (self.onFinish) |cb| {
                self.finish_notified = true; // Before callback (reentrancy).
                cb(self);
                // The callback may have enqueued: report the post-callback
                // state so `loadSync` does not exit with pending tasks.
                done = self.isDone();
            }
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
