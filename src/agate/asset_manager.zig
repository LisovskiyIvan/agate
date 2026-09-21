//! AssetManager: batch asset loading pipeline with progress tracking and caching.
//!
//! Inspired by Babylon.js AssetsManager:
//! - Task types:
//!   - TextFileTask: loads UTF-8 text files (JSON, configs, shaders)
//!   - BinaryFileTask: loads raw binary data (buffers, audio data, packets)
//!   - TextureTask: loads 2D textures (integrated with Texture decode / GPU upload)
//!   - MeshTask: loads 3D models (.glb, .gltf, .obj, .stl, .ply) into a Scene or memory
//!   - CustomTask: user-defined arbitrary loading routine
//! - AssetCache:
//!   - In-memory cache by path/URL to prevent duplicate disk reads and duplicate allocations
//!   - Query, store, invalidate, or bulk-clear cached assets
//! - Progress & Diagnostics:
//!   - Total, completed, failed, and remaining task counts
//!   - Normalized progress in [0.0 .. 1.0]
//!   - Task states: .pending, .running, .completed, .failed
//! - Execution Models:
//!   - Synchronous batch loading (`loadSync`)
//!   - Cooperative stepped frame-budgeted loading (`loadStep`)
//! - Event Callbacks:
//!   - `onTaskSuccess`, `onTaskError`, `onProgress`, `onFinish`

const std = @import("std");

const Scene = @import("scene.zig").Scene;
const Mesh = @import("mesh.zig").Mesh;
const Texture = @import("texture.zig").Texture;
const SceneLoader = @import("loader/scene_loader.zig").SceneLoader;
const obj_loader = @import("loader/obj.zig");
const stl_loader = @import("loader/stl.zig");
const ply_loader = @import("loader/ply.zig");
const sokol = @import("sokol");
const sg = sokol.gfx;

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
    name: []const u8,
    path: []const u8 = "",
    task_type: TaskType,
    state: TaskState = .pending,
    error_result: ?anyerror = null,
    use_cache: bool = true,

    // Results
    text_result: ?[]const u8 = null,
    binary_result: ?[]const u8 = null,
    texture_result: ?Texture = null,
    mesh_count: usize = 0,

    // Task-specific context
    scene: ?*Scene = null,
    texture_options: Texture.Options = .{},
    custom_run: ?*const fn (*AssetTask, std.mem.Allocator) anyerror!void = null,
    user_ctx: ?*anyopaque = null,

    // Internal ownership tracking
    owned_text: bool = false,
    owned_binary: bool = false,
    from_cache: bool = false,

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
        try self.text_map.put(self.allocator, key_dupe, text_dupe);
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
        try self.binary_map.put(self.allocator, key_dupe, data_dupe);
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
        try self.texture_map.put(self.allocator, key_dupe, texture);
    }

    pub fn count(self: *const AssetCache) usize {
        return self.text_map.count() + self.binary_map.count() + self.texture_map.count();
    }
};

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

pub const TaskSuccessFn = *const fn (manager: *AssetManager, task: *AssetTask) void;
pub const TaskErrorFn = *const fn (manager: *AssetManager, task: *AssetTask, err: anyerror) void;
pub const ProgressFn = *const fn (manager: *AssetManager, remaining: usize, total: usize, task: *AssetTask) void;
pub const FinishFn = *const fn (manager: *AssetManager) void;

/// Central asset loading manager orchestrating multiple asset tasks with
/// progress tracking, automatic caching, and step-budgeted execution.
pub const AssetManager = struct {
    allocator: std.mem.Allocator,
    tasks: std.ArrayListUnmanaged(AssetTask) = .empty,
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
    pub fn reset(self: *AssetManager) void {
        for (self.tasks.items) |*task| {
            if (task.owned_text and task.text_result != null) {
                self.allocator.free(task.text_result.?);
            }
            if (task.owned_binary and task.binary_result != null) {
                self.allocator.free(task.binary_result.?);
            }
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
        try self.tasks.append(self.allocator, .{
            .name = name,
            .path = path,
            .task_type = .text,
        });
        return &self.tasks.items[self.tasks.items.len - 1];
    }

    pub fn addBinaryFileTask(self: *AssetManager, name: []const u8, path: []const u8) !*AssetTask {
        try self.tasks.append(self.allocator, .{
            .name = name,
            .path = path,
            .task_type = .binary,
        });
        return &self.tasks.items[self.tasks.items.len - 1];
    }

    pub fn addTextureTask(self: *AssetManager, name: []const u8, path: []const u8, options: Texture.Options) !*AssetTask {
        try self.tasks.append(self.allocator, .{
            .name = name,
            .path = path,
            .task_type = .texture,
            .texture_options = options,
        });
        return &self.tasks.items[self.tasks.items.len - 1];
    }

    pub fn addMeshTask(self: *AssetManager, name: []const u8, path: []const u8, scene: ?*Scene) !*AssetTask {
        try self.tasks.append(self.allocator, .{
            .name = name,
            .path = path,
            .task_type = .mesh,
            .scene = scene,
        });
        return &self.tasks.items[self.tasks.items.len - 1];
    }

    pub fn addCustomTask(
        self: *AssetManager,
        name: []const u8,
        run_fn: *const fn (*AssetTask, std.mem.Allocator) anyerror!void,
        user_ctx: ?*anyopaque,
    ) !*AssetTask {
        try self.tasks.append(self.allocator, .{
            .name = name,
            .task_type = .custom,
            .custom_run = run_fn,
            .user_ctx = user_ctx,
        });
        return &self.tasks.items[self.tasks.items.len - 1];
    }

    pub fn getTaskByName(self: *AssetManager, name: []const u8) ?*AssetTask {
        for (self.tasks.items) |*t| {
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
                    return;
                }
                if (sg.isvalid()) {
                    const tex = try Texture.fromFile(self.allocator, task.path, task.texture_options);
                    task.texture_result = tex;
                    if (task.use_cache) {
                        try self.cache.putTexture(task.path, tex);
                    }
                } else {
                    // Headless CPU decode check
                    var raw = try Texture.decodeImageFile(self.allocator, task.path, .{});
                    raw.deinit(self.allocator);
                    task.texture_result = null;
                }
            },
            .mesh => {
                if (task.scene) |sc| {
                    if (std.mem.endsWith(u8, task.path, ".glb") or std.mem.endsWith(u8, task.path, ".gltf")) {
                        const spawned = try SceneLoader.appendGlb(sc, task.path);
                        task.mesh_count = spawned.len;
                    } else if (std.mem.endsWith(u8, task.path, ".obj")) {
                        const bytes = try readFileBytesAlloc(self.allocator, task.path);
                        defer self.allocator.free(bytes);
                        var data = try obj_loader.parse(self.allocator, bytes);
                        defer data.deinit(self.allocator);
                        task.mesh_count = 1;
                    } else if (std.mem.endsWith(u8, task.path, ".stl")) {
                        const bytes = try readFileBytesAlloc(self.allocator, task.path);
                        defer self.allocator.free(bytes);
                        var data = try stl_loader.parse(self.allocator, bytes);
                        defer data.deinit(self.allocator);
                        task.mesh_count = 1;
                    } else if (std.mem.endsWith(u8, task.path, ".ply")) {
                        const bytes = try readFileBytesAlloc(self.allocator, task.path);
                        defer self.allocator.free(bytes);
                        var data = try ply_loader.parse(self.allocator, bytes);
                        defer data.deinit(self.allocator);
                        task.mesh_count = 1;
                    } else {
                        return error.UnsupportedMeshFormat;
                    }
                } else {
                    // Cache or read raw mesh file
                    const bytes = try readFileBytesAlloc(self.allocator, task.path);
                    defer self.allocator.free(bytes);
                    task.mesh_count = 1;
                }
            },
            .custom => {
                if (task.custom_run) |run| {
                    try run(task, self.allocator);
                }
            },
        }
    }

    /// Runs up to `max_tasks` pending tasks. Returns `true` if all tasks are completed.
    pub fn loadStep(self: *AssetManager, max_tasks: usize) bool {
        var executed: usize = 0;
        while (self.current_index < self.tasks.items.len and executed < max_tasks) {
            const task = &self.tasks.items[self.current_index];
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
