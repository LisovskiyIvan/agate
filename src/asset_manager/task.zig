//! Asset task types. Split out of `asset_manager.zig` (facade).
//!
//! - `TaskState` / `TaskType`: per-task status and kind.
//! - `AssetTask`: one queued load (borrowed name/path; owned-or-cached
//!   results) plus `takeMeshGeometry` / `isSuccess` / `isFailed`.
//! - The in-flight worker handoff type (`async_job`) lives in
//!   `async_mesh.zig`; this module only names it.

const std = @import("std");

const scene_mod = @import("../scene.zig");
const mesh_mod = @import("../mesh.zig");
const texture_mod = @import("../texture.zig");
const async_mesh = @import("async_mesh.zig");

const Scene = scene_mod.Scene;
const GeometryData = mesh_mod.GeometryData;
const Texture = texture_mod.Texture;
const AsyncMeshJob = async_mesh.AsyncMeshJob;

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
    /// Scene lifetime generation captured at submit time (`addMeshTask` /
    /// `addMeshTaskAsync`): the drain/execute paths fail closed when the
    /// live scene no longer reports this generation (or reports teardown),
    /// instead of appending into a recycled or torn-down scene. 0 =
    /// uncaptured (legacy/test scenes): the generation check is skipped
    /// but the teardown flag still applies.
    scene_uid: u64 = 0,
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
