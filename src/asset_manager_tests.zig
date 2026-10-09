const std = @import("std");
const am = @import("asset_manager.zig");
const AssetManager = am.AssetManager;
const AssetCache = am.AssetCache;
const AssetTask = am.AssetTask;
const TaskState = am.TaskState;
const textureHandlesEqual = am.textureHandlesEqual;
const Scene = @import("scene.zig").Scene;
const Mesh = @import("mesh.zig").Mesh;
const Texture = @import("texture.zig").Texture;
const sokol = @import("sokol");
const sg = sokol.gfx;
const jobs = @import("jobs.zig");
const async_test_hold = &am.async_test_hold;

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
