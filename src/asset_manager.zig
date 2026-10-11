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
//!   - `onFinish` fires once per completed batch: it stays installed across
//!     batches and `reset`, polls while already done do not re-fire, and
//!     setting it to null disables it.
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
//!
//! Layout (split of this module; `asset_manager.zig` stays the entry point
//! so `agate.asset_manager` and all existing imports keep working):
//!
//! - `asset_manager/task.zig` — `TaskState`, `TaskType`, `AssetTask`.
//! - `asset_manager/cache.zig` — `AssetCache`, `textureHandlesEqual`.
//! - `asset_manager/mesh_cpu.zig` — file read, `sumSpawned`,
//!   `buildCpuGeometry`.
//! - `asset_manager/async_mesh.zig` — async kinds, `parseMeshBytesToGeometry`,
//!   `AsyncMeshJob`.
//! - `asset_manager/core.zig` — callback aliases, `AssetManager` owner.
//!
//! Retained below (not re-exports): the test-only `async_test_hold` gate
//! plus its sole reader `asyncMeshRun`. Tests take `&async_test_hold`, so
//! the gate must keep definition identity at this path; the worker entry
//! stays with the gate it spins on. `core.zig` posts it by name.

const std = @import("std");

const jobs = @import("jobs.zig");
const core = @import("asset_manager/core.zig");
const task = @import("asset_manager/task.zig");
const cache = @import("asset_manager/cache.zig");
const mesh_cpu = @import("asset_manager/mesh_cpu.zig");
const async_mesh = @import("asset_manager/async_mesh.zig");

const AsyncMeshJob = async_mesh.AsyncMeshJob;
const readFileBytesAlloc = mesh_cpu.readFileBytesAlloc;
const parseMeshBytesToGeometry = async_mesh.parseMeshBytesToGeometry;

pub const AssetManager = core.AssetManager;
pub const AssetTask = task.AssetTask;
pub const AssetCache = cache.AssetCache;
pub const textureHandlesEqual = cache.textureHandlesEqual;

/// Test-only hold gate: while true, async workers spin after publishing
/// `started`/`worker_id` and before parsing. Lets a test observe `.running`
/// deterministically, then release. Never set with a zero-thread (inline)
/// runner — the posting thread would spin on its own gate.
pub var async_test_hold: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);

/// `jobs.TaskRunner` entry: file read + parse + CPU geometry build. No
/// `sg.*`, no Scene mutation, no callbacks, no task-state writes.
pub fn asyncMeshRun(ctx: *anyopaque) void {
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
